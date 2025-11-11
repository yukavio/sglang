
from typing import Optional
import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .flash_streaming_fwd_sm90 import FlashStreamingForwardSm90

# Mapping from PyTorch dtypes to Cutlass dtypes
torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

def convert_blockmask(blockmask, causal):
    """Convert from the 0-1 format to the format used by the CUDA code.
    0 means the block is skipped.
    nonzero means the block is not skipped.
    Argument:
        blockmask: (row, col): a 0-1 tensor
    Return:
        blockmask_converted: (col, row), dtype torch.int32: for each column, it contains the row
            indices of the nonzero blocks, padded with -1 to reach length @row.
            The indices are multiplied by 4, with the smallest bit used to encode whether
            it is the first nonzero in its row, and the 2nd smallest bit to encode whether it is
            the last nonzero in its row..
    """
    assert not causal
    # TD [2022-05-13]: The indexing and sorting is very tricky
    nrow, ncol = blockmask.shape
    # Sort does not support bool on CUDA
    blockmask = blockmask.to(dtype=torch.uint8)
    nonzero_val, nonzero_sorted_rowidx = blockmask.sort(dim=0, stable=True, descending=True)
    nonzero_unsorted_rowidx = nonzero_sorted_rowidx.argsort(dim=0)
    last_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True).indices[:, -1]
    last_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), last_nonzero_col_per_row
    ]
    first_nonzero_col_per_row = blockmask.sort(dim=-1, stable=True, descending=True).indices[:, 0]
    first_nonzero_col_per_row_after_sort = nonzero_unsorted_rowidx[
        torch.arange(nrow, device=blockmask.device), first_nonzero_col_per_row
    ]
    nonzero_idx = nonzero_sorted_rowidx * 4
    nonzero_idx[last_nonzero_col_per_row_after_sort, last_nonzero_col_per_row] += 2
    nonzero_idx[first_nonzero_col_per_row_after_sort, first_nonzero_col_per_row] += 1
    nonzero_idx[nonzero_val == 0] = -1
    return nonzero_idx.T.contiguous().to(dtype=torch.int32)


def convert_blockmask_row_reverse(blockmask, causal):
    """Convert blockmask to row-reverse format for forward pass.
    
    This is similar to convert_blockmask but organizes the mask in row-major order,
    which is more efficient for forward pass iteration.
    
    Args:
        blockmask: (nrow, ncol, num_blockmasks): Block mask tensor
        causal: Whether causal masking is applied
        
    Returns:
        Row-reverse format blockmask suitable for forward pass
    """
    # For now, we use the same conversion as the original
    # TODO: Implement proper row-reverse conversion if needed
    if blockmask.ndim == 3:
        # Handle multiple blockmasks
        nrow, ncol, num_masks = blockmask.shape
        result = []
        for i in range(num_masks):
            result.append(convert_blockmask(blockmask[:, :, i], causal))
        return torch.stack(result, dim=0)  # (num_masks, ncol, nrow)
    else:
        return convert_blockmask(blockmask, causal)


def replace_ones_with_count(tensor):
    """Replace 1s in the tensor with cumulative counts.
    
    Used to process head_mask_type, replacing positions with value 1 with their cumulative count.
    Example: [0, 1, 0, 1, 1, 0] -> [0, 1, 0, 2, 3, 0]
    
    Shape and meaning of head_mask_type:
    - shape: (num_heads,) - 1D tensor with length equal to the total number of attention heads
    - Each element indicates the type of the corresponding attention head:
      * 0: Dense Attention head
        Computes the full attention matrix, all query-key pairs participate in computation
      * 1: Block Sparse Attention head
        Uses precomputed blockmask to define sparse patterns, only computes blocks marked as 1 in the mask
        Suitable for custom sparse patterns (e.g., random sparse, fixed patterns, etc.)
      * -1: Streaming Attention head
        Uses sink+local pattern defined by streaming_info
        Suitable for incremental inference scenarios (e.g., StreamingLLM), retaining initial tokens (sink) and recent tokens (local window)
    
    Purpose of this function:
    Assigns unique IDs (1, 2, 3, ...) to all block sparse heads with value 1,
    so the CUDA kernel can index the corresponding blockmask data using this ID.
    Example: head_mask_type[i]=2 means the i-th head uses the 2nd blockmask.
    
    Args:
        tensor: Input tensor with shape (num_heads,), containing 0, 1, and -1, indicating the type of each attention head
        
    Returns:
        tuple: 
            - Modified tensor: shape remains (num_heads,), positions originally with 1 are replaced with cumulative counts (1, 2, 3, ...),
              used as blockmask index IDs; other positions remain unchanged (0 or -1)
            - Total count of 1s: Total number of block sparse heads, indicating how many blockmasks need to be prepared
              (each block sparse head corresponds to an independent sparse pattern)
    """
    ones_mask = tensor == 1
    ones_num = ones_mask.sum()
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    count = count * ones_mask
    tensor = tensor.masked_scatter(ones_mask, count[ones_mask])
    return tensor, ones_num


def _block_sparse_attn_forward(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    m_block_dim, n_block_dim,
    head_mask_type,
    streaming_info,
    row_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    softmax_scale,
    is_causal,
    exact_streaming,
    return_softmax,
    window_size_left,
    window_size_right
):
    """Block sparse attention forward pass - low-level CUDA kernel invocation.
    
    This function directly calls the CUDA kernel to execute block sparse attention computation.
    It serves as a thin wrapper around the C++/CUDA implementation, handling the interface
    between Python and the compiled kernel.
    
    Args:
        q: Query tensor, shape (total_q, num_heads, head_dim)
            - total_q: sum of all query sequence lengths in the batch
            - num_heads: number of attention heads
            - head_dim: dimension of each attention head
        k: Key tensor, shape (total_k, num_heads_kv, head_dim)
            - total_k: sum of all key sequence lengths in the batch
            - num_heads_kv: number of key/value heads (can be less than num_heads for GQA)
        v: Value tensor, shape (total_k, num_heads_kv, head_dim_v)
            - Same shape as k, but potentially different head dimension
        cu_seqlens_q: Cumulative sequence lengths for queries, shape (batch_size + 1,)
            - Used for variable-length sequences (varlen mode)
            - cu_seqlens_q[i] indicates the starting position of the i-th sequence
            - Example: [0, 128, 256, 512] means 3 sequences with lengths 128, 128, 256
        cu_seqlens_k: Cumulative sequence lengths for keys, shape (batch_size + 1,)
            - Similar to cu_seqlens_q but for key/value sequences
        m_block_dim: Block dimension for query (M dimension), typically 128
            - Defines the granularity of sparse pattern in the query dimension
            - Queries are divided into blocks of size m_block_dim
        n_block_dim: Block dimension for key (N dimension), typically 128
            - Defines the granularity of sparse pattern in the key dimension
            - Keys are divided into blocks of size n_block_dim
        head_mask_type: Per-head mask type indicator, shape (num_heads,)
            - 0: Dense attention (compute full attention matrix)
            - >0: Block sparse attention (use blockmask with this ID)
            - -1: Streaming attention (use streaming_info pattern)
        streaming_info: Streaming attention configuration for incremental inference
            - Contains sink size (number of initial tokens to always attend to)
            - Contains local window size (number of recent tokens to attend to)
            - Used when head_mask_type == -1
        row_blockmask: Row-organized block mask, shape (num_blocksparse_heads, M_blocks, N_blocks)
            - Defines which blocks are computed (1) or skipped (0/-1)
            - Row-reverse format: optimized for forward pass iteration
            - Each blocksparse head has its own mask pattern
        max_seqlen_q_: Maximum query sequence length in the batch
            - Used for memory allocation and boundary checking
        max_seqlen_k_: Maximum key sequence length in the batch
            - Used for memory allocation and boundary checking
        p_dropout: Dropout probability (0.0 to 1.0)
            - Applied to attention weights after softmax
            - 0.0 means no dropout
        softmax_scale: Scaling factor for attention scores
            - Typically 1/sqrt(head_dim) to stabilize gradients
            - Applied before softmax: softmax(Q @ K^T * scale)
        is_causal: Whether to apply causal masking
            - True: mask out future positions (for autoregressive models)
            - False: allow attending to all positions
        exact_streaming: Whether to use exact streaming computation
            - True: precise computation for streaming pattern
            - False: approximate computation (faster but less accurate)
        return_softmax: Whether to return the softmax attention matrix
            - True: return attention weights (for visualization/analysis)
            - False: only return attention output (saves memory)
        window_size_left: Left window size for sliding window attention
            - Number of positions to the left that can be attended to
            - None means no left window constraint
        window_size_right: Right window size for sliding window attention
            - Number of positions to the right that can be attended to
            - None means no right window constraint
        
    Returns:
        tuple: (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state)
            - out: Attention output, shape (total_q, num_heads, head_dim_v)
                Final result of attention computation
            - q: Query tensor (possibly modified/padded by kernel)
                Returned for potential use in backward pass
            - k: Key tensor (possibly modified/padded by kernel)
                Returned for potential use in backward pass
            - v: Value tensor (possibly modified/padded by kernel)
                Returned for potential use in backward pass
            - out_padded: Padded output tensor (if padding was applied)
                Used internally by the kernel for alignment
            - softmax_lse: Log-sum-exp of softmax, shape (batch_size, num_heads, max_seqlen_q)
                Needed for numerically stable backward pass
                LSE = log(sum(exp(attention_scores)))
            - S_dmask: Dropout mask applied to attention weights
                Needed for backward pass to apply same dropout pattern
                None if p_dropout == 0.0
            - rng_state: Random number generator state
                Needed to reproduce dropout pattern in backward pass
                None if p_dropout == 0.0
    
    Note:
        This function is a direct wrapper around the CUDA kernel. The actual computation
        happens in C++/CUDA code (block_sparse_attn_cuda.fwd_block). The last argument
        (None) is reserved for future extensions.
    """
    # Call the CUDA kernel for block sparse attention forward pass
    # The kernel is implemented in C++/CUDA and registered via PyBind11
    # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = block_sparse_attn_cuda.fwd_block(
    #     q, k, v,  # Query, Key, Value tensors
    #     cu_seqlens_q, cu_seqlens_k,  # Cumulative sequence lengths for varlen mode
    #     m_block_dim, n_block_dim,  # Block dimensions for sparse pattern
    #     head_mask_type,  # Per-head attention type indicator
    #     streaming_info,  # Streaming attention configuration
    #     row_blockmask,  # Block mask defining sparse pattern
    #     max_seqlen_q_, max_seqlen_k_,  # Maximum sequence lengths
    #     p_dropout,  # Dropout probability
    #     softmax_scale,  # Attention score scaling factor
    #     is_causal,  # Causal masking flag
    #     exact_streaming,  # Exact streaming computation flag
    #     return_softmax,  # Whether to return attention weights
    #     window_size_left,  # Left sliding window size
    #     window_size_right,  # Right sliding window size
    #     None  # Reserved for future extensions (e.g., custom mask)
    # )
    
    # Return all outputs from the CUDA kernel
    # These will be used by the autograd function for backward pass
    # return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state

    # Get device compute capability and validate
    compute_capability = torch.cuda.get_device_capability()[0]
    assert compute_capability in [9], "Unsupported compute capability. Supported: 9.x"
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Extract tensor shapes
    total_q, num_head, head_dim = q.shape
    total_k, num_head_kv, _ = k.shape
    head_dim_v = v.shape[-1]
    batch_size = cu_seqlens_q.shape[0] - 1 if cu_seqlens_q is not None else 1
    qhead_per_kvhead = num_head // num_head_kv
    
    # Set default softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Validate inputs
    assert q.dtype in [torch.float16, torch.bfloat16], "q must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be on CUDA device"
    
    # Create output tensors
    out = torch.empty_like(q[..., :head_dim_v])  # (total_q, num_head, head_dim_v)
    softmax_lse = torch.empty(num_head, total_q, dtype=torch.float32, device=q.device)
    
    # Handle dropout (not implemented yet, just create placeholders)
    S_dmask = None
    rng_state = None
    if p_dropout > 0.0:
        # TODO: Implement dropout support
        raise NotImplementedError("Dropout is not yet implemented for block sparse attention")
    
    # Convert torch tensors to cute tensors
    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out)
    ]
    lse_tensor = from_dlpack(softmax_lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
    
    # Convert cu_seqlens tensors if provided
    cu_seqlens_q_tensor = from_dlpack(cu_seqlens_q.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0) if cu_seqlens_q is not None else None
    cu_seqlens_k_tensor = from_dlpack(cu_seqlens_k.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0) if cu_seqlens_k is not None else None
    
    # Handle window sizes
    local = window_size_left is not None or window_size_right is not None
    if is_causal:
        window_size_right = 0
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            is_causal, local = True, False
        else:
            is_causal, local = False, True
    
    # Create compilation key for kernel caching
    compile_key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead, 
        is_causal, local,
        cu_seqlens_q is not None, cu_seqlens_k is not None,
        window_size_left is not None, window_size_right is not None,
        m_block_dim, n_block_dim,
        compute_capability,
    )
    
    # Initialize compile cache if it doesn't exist
    if not hasattr(_block_sparse_attn_forward, 'compile_cache'):
        _block_sparse_attn_forward.compile_cache = {}
    
    # Compile kernel if not cached
    if compile_key not in _block_sparse_attn_forward.compile_cache:
        from .flash_fwd_sm90 import FlashAttentionForwardSm90
        
        # Create FlashAttentionForwardSm90 instance
        fa_fwd = FlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=is_causal,
            is_local=local,
            pack_gqa=False,  # For block sparse, we don't use pack_gqa for now
            m_block_size=m_block_dim,
            n_block_size=n_block_dim,
            num_stages=2,
            num_threads=384,  # 3 warp groups for producer, 1 for consumer
            Q_in_regs=False,
            groupwise=False,
        )
        
        # Compile the kernel
        _block_sparse_attn_forward.compile_cache[compile_key] = cute.compile(
            fa_fwd, 
            q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, 
            softmax_scale, current_stream,
            cu_seqlens_q_tensor, cu_seqlens_k_tensor, 
            None, None,  # seqused_q, seqused_k
            None,  # page_table
            None,  # softcap
            window_size_left, window_size_right,
            None,  # learnable_sink
        )
    
    # Execute the compiled kernel
    _block_sparse_attn_forward.compile_cache[compile_key](
        q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
        softmax_scale, current_stream,
        cu_seqlens_q_tensor, cu_seqlens_k_tensor,
        None, None,  # seqused_q, seqused_k
        None,  # page_table
        None,  # softcap
        window_size_left, window_size_right,
        None,  # learnable_sink
    )
    
    # Return outputs in the expected format
    # Note: out_padded is set to out for now (no padding applied)
    return out, q, k, v, out, softmax_lse, S_dmask, rng_state
    

class BlockSparseAttnFun(torch.autograd.Function):
    """Block sparse attention autograd function.
    
    Implements PyTorch's autograd interface, supporting forward and backward propagation.
    This is the main block sparse attention implementation that does not return attention weights.
    """
    
    @staticmethod
    def forward(ctx,
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                m_block_dim, n_block_dim,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q_, max_seqlen_k_,
                p_dropout,
                softmax_scale,
                is_causal,
                exact_streaming,
                return_softmax,
                window_size_left,
                window_size_right, deterministic=False):
        """Forward propagation.
        
        Executes block sparse attention computation and saves intermediate results needed for backward pass.
        
        Key steps:
        1. Set default softmax scale factor (if not provided)
        2. Convert block mask to row-reverse format (for forward pass)
        3. Validate exact_streaming prerequisites
        4. Call CUDA kernel to perform computation
        5. Save tensors and parameters needed for backward pass
        
        Args:
            ctx: Context object to save information for backward pass
            q: Query tensor, shape (total_q, num_heads, head_dim)
            k: Key tensor, shape (total_k, num_heads, head_dim)
            v: Value tensor, shape (total_k, num_heads, head_dim)
            cu_seqlens_q: Cumulative sequence lengths for queries, shape (batch_size + 1,)
            cu_seqlens_k: Cumulative sequence lengths for keys, shape (batch_size + 1,)
            m_block_dim: Block dimension for query (M dimension), typically 128
            n_block_dim: Block dimension for key (N dimension), typically 128
            head_mask_type: Tensor indicating attention type for each head, shape (num_heads,)
                           0: dense attention, 1+: block sparse (with blockmask ID), -1: streaming
            streaming_info: Streaming attention configuration (sink size, local window size)
            base_blockmask: Block mask tensor defining sparse pattern, shape (nrow, ncol, num_blockmasks)
            max_seqlen_q_: Maximum query sequence length in the batch
            max_seqlen_k_: Maximum key sequence length in the batch
            p_dropout: Dropout probability
            softmax_scale: Scaling factor for softmax, defaults to 1/sqrt(head_dim)
            is_causal: Whether to apply causal masking
            exact_streaming: Whether to use exact streaming attention mode
            return_softmax: Whether to return softmax probabilities (not used in this function)
            window_size_left: Left window size for local attention (-1 means no limit)
            window_size_right: Right window size for local attention (-1 means no limit)
            deterministic: Whether to use deterministic computation
            
        Returns:
            out: Output tensor, shape (total_q, num_heads, head_dim)
        """
        # Save rng_state because the backward pass will regenerate the dropout mask
        # Set default softmax scale to 1/sqrt(head_dim) if not provided
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        
        # Convert block mask to row-reverse format for efficient CUDA kernel processing
        # This format allows the kernel to quickly find which blocks to compute for each row
        if base_blockmask is not None:
            row_blockmask = convert_blockmask_row_reverse(base_blockmask, is_causal)
        else:
            row_blockmask = None
        
        # Validate exact_streaming mode requirements
        # Exact streaming requires streaming_info and causal masking to be enabled
        if exact_streaming:
            assert streaming_info is not None
            assert is_causal
        
        # Call the CUDA kernel to perform block sparse attention computation
        # Returns output, potentially modified q/k/v, padded output, softmax statistics,
        # dropout mask, and RNG state for backward pass
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _block_sparse_attn_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            m_block_dim, n_block_dim,
            head_mask_type,
            streaming_info,
            row_blockmask,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            softmax_scale,
            is_causal,
            exact_streaming,
            return_softmax=False,
            window_size_left=window_size_left,
            window_size_right=window_size_right
        )
        
        
        # Save scalar parameters as context attributes
        # These define the block structure and attention configuration
        ctx.m_block_dim = m_block_dim
        ctx.n_block_dim = n_block_dim
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.max_seqlen_q_ = max_seqlen_q_
        ctx.max_seqlen_k_ = max_seqlen_k_
        ctx.p_dropout = p_dropout
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        ctx.exact_streaming = exact_streaming
        ctx.deterministic = deterministic
        
        return out


def block_sparse_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    base_blockmask: torch.Tensor,
    max_seqlens_q_: int,
    max_seqlens_k_: int,
    p_dropout: float = 0.0,
    deterministic: bool = False,
    softmax_scale: float = None,
    is_causal: bool = False,
    exact_straming: bool = False,
    return_attn_probs: bool =  False
):  
    head_mask_type, blocksparse_head_num = replace_ones_with_count(head_mask_type)
    if base_blockmask is not None:
        assert base_blockmask.shape[1] == blocksparse_head_num, "base_blockmask.shape[1] must be equal to blocksparse_head_num"

    func = BlockSparseAttnFun
    return func.apply(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        128, 128,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlens_q_, max_seqlens_k_,
        p_dropout,
        softmax_scale,
        is_causal,
        exact_straming,
        return_attn_probs,
        -1, -1,
        deterministic
    )


def block_streaming_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_mask_type: torch.Tensor,
    streaming_info: torch.Tensor,
    max_seqlen_q_: int,
    max_seqlen_k_: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
):
    # TODO: Implement block streaming attention
    pass
