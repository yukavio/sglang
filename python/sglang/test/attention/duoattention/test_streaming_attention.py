import pytest
import torch
import math

from sglang.srt.sparse_attention.kernels.attention.streaming_sparse_attention_interface import (
    streaming_sparse_attn_func,
)
from sglang.test.attention.duoattention.streaming_attention_ref import (
    block_streaming_attention_ref,
)


def is_hopper():
    """Check if the current GPU is Hopper (compute capability 9.x)"""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


@pytest.mark.skipif(not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)")
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("sink_size", [4, 8])
@pytest.mark.parametrize("local_size", [32, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_streaming_attention(seqlen, dtype, sink_size, local_size, batch_size):
    device = torch.device("cuda")

    num_heads = 4
    head_dim = 64

    # Create batch format tensors
    q = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)

    total_tokens = batch_size * seqlen
    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    k_varlen = k.reshape(total_tokens, num_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Test CUDA implementation (batch format)
    out_cuda, _ = streaming_sparse_attn_func(
        q, k, v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(local_size - 1, 0),
        learnable_sink=None,
        sink_size=sink_size,
        enable_streaming=True,
        softcap=0.0,
        pack_gqa=False,
        groupwise=False,
        position_ids=None,
    )


    head_mask_type = torch.full(
        (num_heads,), -1, dtype=torch.int32, device=device)
    out_ref_varlen, _ = block_streaming_attention_ref(
        q_varlen, k_varlen, v_varlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        head_mask_type=head_mask_type,
        sink_size=sink_size,
        local_size=local_size,
        softmax_scale=softmax_scale,
        is_causal=True,
    )
    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Check output shape
    assert out_cuda.shape == (batch_size, seqlen, num_heads, head_dim), \
        f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(f"Streaming attention test passed for seqlen={seqlen}, dtype={dtype}, sink_size={sink_size}, local_size={local_size}, batch_size={batch_size}!")


def test_chunked_streaming_attention():
    device = torch.device("cuda")
    num_heads = 4
    head_dim = 64
    seqlen = 4096
    dtype = torch.bfloat16
    sink_size = 4
    local_size = 32
    batch_size = 1
    
    q = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    
    total_tokens = batch_size * seqlen

    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    k_varlen = k.reshape(total_tokens, num_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )
    position_ids = torch.arange(
        0, total_tokens, dtype=torch.int32, device=device
    )

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Test CUDA implementation (batch format)
    # out_cuda, _ = streaming_sparse_attn_func(
    #     q, k, v,
    #     cu_seqlens_q=None,
    #     cu_seqlens_k=None,
    #     seqused_q=None,
    #     seqused_k=None,
    #     page_table=None,
    #     softmax_scale=softmax_scale,
    #     causal=True,
    #     window_size=(local_size - 1, 0),
    #     learnable_sink=None,
    #     sink_size=sink_size,
    #     enable_streaming=True,
    #     softcap=0.0,
    #     pack_gqa=False,
    #     groupwise=False,
    #     position_ids=None,
    # )

    chunk_size = 256
    num_chunks = total_tokens // chunk_size

    k_cache = []
    v_cache = []
    chunked_outputs = []

    for i in range(num_chunks): 
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        q_chunk = q[:, start_idx:end_idx, :, :]
        k_chunk = k[:, start_idx:end_idx, :, :]
        v_chunk = v[:, start_idx:end_idx, :, :]

        k_cache.append(k_chunk)
        v_cache.append(v_chunk)

        k_context = torch.cat(k_cache, dim=1)
        v_context = torch.cat(v_cache, dim=1)

        pos_ids_chunk = torch.arange(start_idx, end_idx, dtype=torch.int32, device=device)
        pos_ids_chunk = pos_ids_chunk.unsqueeze(0).expand(batch_size, -1)

        print(f"    - q_chunk shape: {q_chunk.shape}")
        print(f"    - k_context shape: {k_context.shape}")
        print(f"    - position_ids: from {pos_ids_chunk[0, 0]} to {pos_ids_chunk[0, -1]}")

        out_chunk, _ = streaming_sparse_attn_func(
            q_chunk, k_context, v_context,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            page_table=None,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(local_size - 1, 0),
            learnable_sink=None,
            sink_size=sink_size,
            enable_streaming=True,
            softcap=0.0,
            pack_gqa=False,
            groupwise=False,
            position_ids=pos_ids_chunk,
        )

        chunked_outputs.append(out_chunk)

    out_cuda = torch.cat(chunked_outputs, dim=1)

    head_mask_type = torch.full(
        (num_heads,), -1, dtype=torch.int32, device=device)
    out_ref_varlen, _ = block_streaming_attention_ref(
        q_varlen, k_varlen, v_varlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        head_mask_type=head_mask_type,
        sink_size=sink_size,
        local_size=local_size,
        softmax_scale=softmax_scale,
        is_causal=True,
    )

    # Check output shape
    assert out_cuda.shape == (batch_size, seqlen, num_heads, head_dim), \
        f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(f"Chunked streaming attention test passed for seqlen={seqlen}, dtype={dtype}, sink_size={sink_size}, local_size={local_size}, batch_size={batch_size}!")