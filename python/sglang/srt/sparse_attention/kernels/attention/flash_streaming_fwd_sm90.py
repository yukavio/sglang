# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic

from sglang.srt.sparse_attention.kernels.attention import hopper_helpers as sm90_utils
from sglang.srt.sparse_attention.kernels.attention import utils
from sglang.srt.sparse_attention.kernels.attention import pipeline
from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from sglang.srt.sparse_attention.kernels.attention.softmax import Softmax
from sglang.srt.sparse_attention.kernels.attention.seqlen_info import SeqlenInfoQK
from sglang.srt.sparse_attention.kernels.attention.block_info import BlockInfo
from sglang.srt.sparse_attention.kernels.attention.pack_gqa import PackGQA
from sglang.srt.sparse_attention.kernels.attention.named_barrier import NamedBarrierFwd
from sglang.srt.sparse_attention.kernels.attention.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, SingleTileLPTScheduler, SingleTileVarlenScheduler, ParamsBase
from sglang.srt.sparse_attention.kernels.attention.flash_fwd_sm90 import FlashAttentionForwardSm90

class FlashStreamingForwardSm90(FlashAttentionForwardSm90):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cute.jit
    def __call__(self, *args, **kwargs):
        pass

    @cute.jit
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softcap_val: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        groupwise: bool,
        sink_size: Optional[int] = None,
        enable_streaming: bool = False,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # TMA prefetch
        if warp_idx == 0:
            if const_expr(tma_atom_Q is not None):
                cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier init
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            # if tidx < 2:
            #     # barrierO num threads should be self.num_mma_threads
            #     cute.arch.mbarrier_init(mbar_ptr_Q + tidx, 1 if tidx == 0 else self.num_mma_threads)
            cute.arch.mbarrier_init(mbar_ptr_Q, 1 if const_expr(self.use_tma_Q) else self.num_Q_load_threads)
            # cute.arch.mbarrier_init(mbar_ptr_Q + 1, self.num_mma_threads)
        # We rely on pipeline_k and pipeline_v to initialize the mbarrier fence and sync
        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group
        )
        pipeline_k = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_k_bytes,
            init_wait=False,
        )
        pipeline_v = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_v_bytes,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        # TODO: how to get sQ_pi for cp.async if pack_gqa?
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type)
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)
        if const_expr(sP_layout is not None):
            sP_pi = storage.sP.get_tensor(sP_layout)
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        else:
            sP, sP_pi = None, None
        # reuse sQ's data iterator
        sO_pi = storage.sQ.get_tensor(sO_layout)
        # TODO: idk why not using sO_pi is faster
        sO = cute.make_tensor(cute.recast_ptr(sO_pi.iterator, sO_layout.inner, dtype=sO_pi.element_type), sO_layout.outer)

        block_info = BlockInfo(
            self.m_block_size, self.n_block_size, self.is_causal, self.is_local,
            window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            sink_size=sink_size,
            enable_streaming=enable_streaming,
        )

        SeqlenInfoCls = partial(
            SeqlenInfoQK, seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0] if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ, mSeqUsedK=mSeqUsedK,
        )

        AttentionMaskCls = partial(
            AttentionMask, self.m_block_size, self.n_block_size,
            window_size_left=window_size_left, window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            sink_size=sink_size,
            enable_streaming=enable_streaming,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Producer
        if warp_idx < 4:  # Producer
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                groupwise,
            )

        else:
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_pv_rs,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softcap_val,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
            )
        
        