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
# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# @pytest.mark.parametrize("sink_size", [4, 8])
# @pytest.mark.parametrize("local_size", [64, 128])
# @pytest.mark.parametrize("seqlen", [256, 512])

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("sink_size", [4])
@pytest.mark.parametrize("local_size", [64])
@pytest.mark.parametrize("seqlen", [256, 512])
def test_streaming_attention_batch(dtype, sink_size, local_size, seqlen):
    """Test streaming sparse attention with standard batch format.

    This test compares the output of the CUDA kernel implementation against
    a reference PyTorch implementation using standard (batch, seqlen, heads, dim) format.

    Args:
        dtype: Data type for tensors (bfloat16 or float16)
        sink_size: Number of sink tokens at the beginning
        local_size: Size of the local sliding window
        seqlen: Sequence length for each sample in the batch
    """
    device = torch.device("cuda")
    batch_size = 2
    num_heads = 8
    head_dim = 64

    # Generate random input tensors in standard batch format
    # Shape: (batch_size, seqlen, num_heads, head_dim)
    q = torch.randn(batch_size, seqlen, num_heads, head_dim,
                    dtype=dtype, device=device, requires_grad=False)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim,
                    dtype=dtype, device=device, requires_grad=False)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim,
                    dtype=dtype, device=device, requires_grad=False)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Call the CUDA kernel implementation (standard batch format, no cu_seqlens)
    # window_size_left = (local_size - 1) to implement a sliding window
    window_size_left = local_size - 1
    window_size_right = 0  # causal, so can't see future

    out_cuda, _ = streaming_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q=None,  # No varlen
        cu_seqlens_k=None,  # No varlen
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(window_size_left, window_size_right),
        learnable_sink=None,
        sink_size=sink_size,
        enable_streaming=True,
        softcap=0.0,
        pack_gqa=False,  # Disable GQA for simple test
        groupwise=False,
    )

    # Prepare reference implementation inputs (needs varlen format)
    # Convert batch format to varlen format for reference
    total_tokens = batch_size * seqlen
    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    k_varlen = k.reshape(total_tokens, num_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_heads, head_dim)

    cu_seqlens = torch.tensor(
        [i * seqlen for i in range(batch_size + 1)], dtype=torch.int32, device=device)

    # head_mask_type: -1 for all heads means all use streaming attention
    head_mask_type = torch.full(
        (num_heads,), -1, dtype=torch.int32, device=device)

    out_ref_varlen, _ = block_streaming_attention_ref(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        head_mask_type=head_mask_type,
        sink_size=sink_size,
        local_size=local_size,
        p_dropout=0.0,
        softmax_scale=softmax_scale,
        is_causal=True,
        dropout_mask=None,
        query_padding_mask=None,
        key_padding_mask=None,
        return_attn_probs=False,
    )

    # Convert reference output back to batch format
    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Compare outputs
    # Use a reasonable tolerance for floating point comparison
    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 1e-2

    torch.testing.assert_close(
        out_cuda,
        out_ref,
        atol=atol,
        rtol=rtol,
        msg=f"Streaming attention output mismatch for dtype={dtype}, sink_size={sink_size}, "
        f"local_size={local_size}, seqlen={seqlen}"
    )

    print(
        f"Test passed: dtype={dtype}, sink_size={sink_size}, local_size={local_size}, seqlen={seqlen}")


@pytest.mark.skipif(not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)")
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("sink_size", [4, 8])
@pytest.mark.parametrize("local_size", [32])
def test_streaming_attention(seqlen, dtype, sink_size, local_size):
    """Basic test with fixed parameters to quickly verify functionality."""
    device = torch.device("cuda")

    batch_size = 1
    num_heads = 4
    head_dim = 64

    # Create batch format tensors
    q = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)

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
    )

    # Test reference implementation (needs varlen format)
    q_varlen = q.reshape(seqlen, num_heads, head_dim)
    k_varlen = k.reshape(seqlen, num_heads, head_dim)
    v_varlen = v.reshape(seqlen, num_heads, head_dim)
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

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
    torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)

    print("✓ Basic streaming attention test passed!")


@pytest.mark.skipif(not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)")
def test_streaming_attention_different_sink_sizes():
    """Test with various sink sizes to ensure correctness."""
    device = torch.device("cuda")
    dtype = torch.float16

    batch_size = 1
    seqlen = 256
    num_heads = 8
    head_dim = 128
    local_size = 64

    q = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads,
                    head_dim, dtype=dtype, device=device)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Test different sink sizes
    for sink_size in [0, 2, 4, 8, 16]:
        out_cuda, _ = streaming_sparse_attn_func(
            q, k, v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(local_size - 1, 0),
            sink_size=sink_size,
            enable_streaming=True,
            pack_gqa=False,
        )

        # Convert to varlen for reference
        q_varlen = q.reshape(seqlen, num_heads, head_dim)
        k_varlen = k.reshape(seqlen, num_heads, head_dim)
        v_varlen = v.reshape(seqlen, num_heads, head_dim)
        cu_seqlens = torch.tensor(
            [0, seqlen], dtype=torch.int32, device=device)

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
        out_ref = out_ref_varlen.reshape(
            batch_size, seqlen, num_heads, head_dim)

        torch.testing.assert_close(out_cuda, out_ref, atol=5e-3, rtol=1e-2)
        print(f"✓ Test passed for sink_size={sink_size}")


if __name__ == "__main__":
    # Run basic tests
    if is_hopper():
        print("Running streaming attention tests on Hopper GPU...")
        test_streaming_attention()
    else:
        print("Streaming attention tests require Hopper GPU (SM 9.0). Skipping tests.")
