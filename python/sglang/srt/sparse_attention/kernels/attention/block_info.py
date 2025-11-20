# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from sglang.srt.sparse_attention.kernels.attention.seqlen_info import SeqlenInfoQK

@dataclass(frozen=True)
class BlockInfo:
    m_block_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    sink_size: cutlass.Constexpr[int] = 0
    enable_streaming: cutlass.Constexpr[bool] = False

    @cute.jit
    def get_n_block_min_max(
        self, seqlen_info: SeqlenInfoQK, m_block: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size)
        if cutlass.const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = n_idx if cutlass.const_expr(self.is_causal) else n_idx + self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.n_block_size))
        n_block_min = 0
        if cutlass.const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.n_block_size, 0)
        return n_block_min, n_block_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        """If we have separate iterations with causal or local masking at the start, where do we stop"""
        m_idx_min = m_block * self.m_block_size
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if cutlass.const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.n_block_size)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        """If we have separate iterations with local masking at the end, where do we stop the non-masked iterations"""
        if cutlass.const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.n_block_size))


    @cute.jit
    def get_streaming_mask_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: cutlass.Int32,
        position_ids: Optional[cute.Tensor] = None,
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        """
        Get the start and end of the streaming mask for the given m_block.
        
        For streaming attention, a query at position i can attend to:
        1. Sink region: [0, sink_size)
        2. Local window: [i - window_size_left + 1, i] (within causal constraint)
        
        Returns (n_block_min, n_block_max) where n_block in [n_block_min, n_block_max) 
        need to be processed.
        """
        # Calculate n_block_max based on causal constraint
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size)
        
        if cutlass.const_expr(self.is_causal):
            # For causal attention, the rightmost position a query at m_block can attend to
            # is determined by the maximum row index in this block
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            # Adjust for query/key length difference
            n_idx_max = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_block_max = min(
                n_block_max, cute.ceil_div(n_idx_max, self.n_block_size)
            )
        
        # Calculate n_block_min based on local window constraint
        # The sink region [0, sink_size) always needs to be included
        n_block_min = 0
        
        if self.enable_streaming and self.window_size_left is not None:
            # For streaming attention with local window, the leftmost position 
            # (excluding sink) that needs to be processed is determined by the 
            # minimum row index in this block
            m_idx_min = m_block * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            
            # Adjust for query/key length difference
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            
            # The left boundary of the local window (excluding sink)
            n_idx_left = n_idx - self.window_size_left + 1
            
            # We need to process blocks that overlap with either:
            # 1. The sink region [0, sink_size), or
            # 2. The local window [n_idx_left, n_idx_max]
            
            if self.sink_size > 0:
                # When sink is present, we must start from block 0 to include the sink region.
                # Even if there's a gap between sink and local window, we return a single
                # continuous range [0, n_block_max). The gap will be masked out by
                # apply_streaming_mask() which checks each element individually.
                n_block_min = 0
            else:
                # No sink, only local window - start from the left boundary of the window
                n_block_min = cutlass.max(n_idx_left // self.n_block_size, 0)
        
        return n_block_min, n_block_max

        # q_pos_min = position_ids[0]
        # q_pos_max = position_ids[self.m_block_size - 1]

        # k_pos_max = q_pos_max 
        # n_block_max = cute.ceil_div(k_pos_max + 1, self.n_block_size)

        # n_block_max = cutlass.min(n_block_max, cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size))

        # n_block_min = 0

        # if self.enable_streaming:
        #     if self.sink_size > 0:
        #         n_block_min = 0 
        #     elif self.window_size_left is not None:
        #         k_pos_min_window = q_pos_min - self.window_size_left

        #         n_block_min = cutlass.max(0, k_pos_min_window // self.n_block_size)

        # return n_block_min, n_block_max
