import math

import pytest
import torch

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


@pytest.mark.skipif(
    not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("sink_size", [4, 8])
@pytest.mark.parametrize("local_size", [32, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_streaming_attention(seqlen, dtype, sink_size, local_size, batch_size):
    device = torch.device("cuda")

    num_heads = 4
    head_dim = 64

    # Create batch format tensors
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)

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
        q,
        k,
        v,
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

    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)
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
        softmax_scale=softmax_scale,
        is_causal=True,
    )
    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Check output shape
    assert out_cuda.shape == (
        batch_size,
        seqlen,
        num_heads,
        head_dim,
    ), f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(
        f"Streaming attention test passed for seqlen={seqlen}, dtype={dtype}, sink_size={sink_size}, local_size={local_size}, batch_size={batch_size}!"
    )


@pytest.mark.skipif(
    not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("sink_size", [4, 8])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_chunked_streaming_attention(seqlen, sink_size, chunk_size, batch_size):
    device = torch.device("cuda")
    num_heads = 4
    head_dim = 64
    dtype = torch.bfloat16
    local_size = 32

    q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)

    total_tokens = batch_size * seqlen

    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    k_varlen = k.reshape(total_tokens, num_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # num_chunks = total_tokens // chunk_size
    num_chunks = seqlen // chunk_size

    k_cache = []
    v_cache = []
    chunked_outputs = []

    print(f"num_chunks: {num_chunks}")

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

        pos_ids_chunk = torch.arange(
            start_idx, end_idx, dtype=torch.int32, device=device
        )
        pos_ids_chunk = pos_ids_chunk.unsqueeze(0).expand(batch_size, -1).contiguous()

        out_chunk, _ = streaming_sparse_attn_func(
            q_chunk,
            k_context,
            v_context,
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

    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)
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
        softmax_scale=softmax_scale,
        is_causal=True,
    )
    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Check output shape
    assert out_cuda.shape == (
        batch_size,
        seqlen,
        num_heads,
        head_dim,
    ), f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    print(f"out_cuda: {out_cuda}")
    print(f"out_ref: {out_ref}")

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(
        f"Chunked streaming attention test passed for seqlen={seqlen}, dtype={dtype}, sink_size={sink_size}, local_size={local_size}, batch_size={batch_size}!"
    )


@pytest.mark.skipif(
    not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("sink_size", [4, 8])
@pytest.mark.parametrize("local_size", [32])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_paged_streaming_attention(
    seqlen, page_size, sink_size, local_size, batch_size
):
    device = torch.device("cuda")
    num_heads = 4
    head_dim = 64
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)

    total_tokens = batch_size * seqlen

    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    k_varlen = k.reshape(total_tokens, num_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )

    num_blocks_per_seq = (seqlen + page_size - 1) // page_size
    max_num_blocks = num_blocks_per_seq * batch_size * 2

    k_cache = torch.zeros(
        max_num_blocks, page_size, num_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.zeros(
        max_num_blocks, page_size, num_heads, head_dim, dtype=dtype, device=device
    )

    page_table = torch.zeros(
        batch_size, num_blocks_per_seq, dtype=torch.int32, device=device
    )

    all_random_indices = torch.randperm(
        max_num_blocks, device=device, dtype=torch.int32
    )

    for b in range(batch_size):
        start_idx = b * num_blocks_per_seq
        end_idx = start_idx + num_blocks_per_seq

        available_indices = all_random_indices[start_idx:end_idx]
        page_table[b] = available_indices

        for i, block_idx in enumerate(available_indices):
            start_token = i * page_size
            end_token = min((i + 1) * page_size, seqlen)
            valid_len = end_token - start_token

            if valid_len > 0:
                k_cache[block_idx, :valid_len, :, :] = k[b, start_token:end_token, :, :]
                v_cache[block_idx, :valid_len, :, :] = v[b, start_token:end_token, :, :]

        softmax_scale = 1.0 / math.sqrt(head_dim)

        seqused_k = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)

        out_cuda, _ = streaming_sparse_attn_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=seqused_k,
            page_table=page_table,
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
            m_block_size=128,
            n_block_size=page_size,
        )

    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)

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
        softmax_scale=softmax_scale,
        is_causal=True,
    )

    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Check output shape
    assert out_cuda.shape == (
        batch_size,
        seqlen,
        num_heads,
        head_dim,
    ), f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(
        f"Paged streaming attention test passed for seqlen={seqlen}, dtype={dtype}, batch_size={batch_size}!"
    )


@pytest.mark.skipif(
    not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("sink_size", [4, 8])
@pytest.mark.parametrize("local_size", [32])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_paged_chunked_streaming_attention(
    seqlen, page_size, chunk_size, sink_size, local_size, batch_size
):
    device = torch.device("cuda")
    num_heads = 4
    head_dim = 64
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)

    total_tokens = batch_size * seqlen

    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    k_varlen = k.reshape(total_tokens, num_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )

    num_blocks_per_seq = (seqlen + page_size - 1) // page_size
    max_num_blocks = num_blocks_per_seq * batch_size * 2

    k_cache = torch.zeros(
        max_num_blocks, page_size, num_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.zeros(
        max_num_blocks, page_size, num_heads, head_dim, dtype=dtype, device=device
    )

    page_table = torch.zeros(
        batch_size, num_blocks_per_seq, dtype=torch.int32, device=device
    )
    all_random_indices = torch.randperm(
        max_num_blocks, device=device, dtype=torch.int32
    )

    for b in range(batch_size):
        start_idx = b * num_blocks_per_seq
        end_idx = start_idx + num_blocks_per_seq

        available_indices = all_random_indices[start_idx:end_idx]
        page_table[b] = available_indices

        for i, block_idx in enumerate(available_indices):
            start_token = i * page_size
            end_token = min((i + 1) * page_size, seqlen)

            valid_len = end_token - start_token

            if valid_len > 0:
                k_cache[block_idx, :valid_len, :, :] = k[b, start_token:end_token, :, :]
                v_cache[block_idx, :valid_len, :, :] = v[b, start_token:end_token, :, :]

    softmax_scale = 1.0 / math.sqrt(head_dim)

    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    chunked_outputs = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        q_chunk = q[:, start_idx:end_idx, :, :]

        pos_ids_chunk = torch.arange(
            start_idx, end_idx, dtype=torch.int32, device=device
        )
        pos_ids_chunk = pos_ids_chunk.unsqueeze(0).expand(batch_size, -1).contiguous()

        current_seqsed_k = torch.full(
            (batch_size,), end_idx, dtype=torch.int32, device=device
        )

        out_chunk, _ = streaming_sparse_attn_func(
            q_chunk,
            k_cache,
            v_cache,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=current_seqsed_k,
            page_table=page_table,
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
            m_block_size=128,
            n_block_size=page_size,
        )

        chunked_outputs.append(out_chunk)

    out_cuda = torch.cat(chunked_outputs, dim=1)

    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)

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
        softmax_scale=softmax_scale,
        is_causal=True,
    )

    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # Check output shape
    assert out_cuda.shape == (
        batch_size,
        seqlen,
        num_heads,
        head_dim,
    ), f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(
        f"Paged streaming attention test passed for seqlen={seqlen}, dtype={dtype}, batch_size={batch_size}!"
    )


@pytest.mark.skipif(
    not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
def test_streaming_attention_gqa(
    seqlen
):
    device = torch.device("cuda")
    batch_size=1
    page_size=32
    sink_size=8
    local_size=16
    gqa_group_size=2
    num_heads = 4  # must be divisible by gqa_group_size
    num_kv_heads = num_heads // gqa_group_size
    head_dim = 64
    dtype = torch.bfloat16

    # Q: [B, seqlen, num_heads, head_dim]
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    # K/V: [B, seqlen, num_kv_heads, head_dim]
    k = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)

    total_tokens = batch_size * seqlen

    # Q: [total_tokens, num_heads, head_dim]
    q_varlen = q.reshape(total_tokens, num_heads, head_dim)
    # K/V: [total_tokens, num_kv_heads, head_dim]
    k_varlen = k.reshape(total_tokens, num_kv_heads, head_dim)
    v_varlen = v.reshape(total_tokens, num_kv_heads, head_dim)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )

    num_blocks_per_seq = (seqlen + page_size - 1) // page_size
    max_num_blocks = num_blocks_per_seq * batch_size * 2

    # K/V paged cache: [max_num_blocks, page_size, num_kv_heads, head_dim]
    k_cache = torch.zeros(
        max_num_blocks, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.zeros(
        max_num_blocks, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    page_table = torch.zeros(
        batch_size, num_blocks_per_seq, dtype=torch.int32, device=device
    )
    all_random_indices = torch.randperm(
        max_num_blocks, device=device, dtype=torch.int32
    )

    for b in range(batch_size):
        start_idx = b * num_blocks_per_seq
        end_idx = start_idx + num_blocks_per_seq
        available_indices = all_random_indices[start_idx:end_idx]
        page_table[b] = available_indices

        for i, block_idx in enumerate(available_indices):
            start_token = i * page_size
            end_token = min((i + 1) * page_size, seqlen)
            valid_len = end_token - start_token

            if valid_len > 0:
                k_cache[block_idx, :valid_len, :, :] = k[b, start_token:end_token, :, :]
                v_cache[block_idx, :valid_len, :, :] = v[b, start_token:end_token, :, :]

    softmax_scale = 1.0 / math.sqrt(head_dim)
    seqused_k = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)

    out_cuda, _ = streaming_sparse_attn_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(local_size - 1, 0),
        learnable_sink=None,
        sink_size=sink_size,
        enable_streaming=True,
        softcap=0.0,
        pack_gqa=True,  # Enable GQA feature here
        groupwise=False,
        position_ids=None,
        m_block_size=128,
        n_block_size=page_size,
    )

    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)

    # For reference implementation, we must duplicate k/v heads for each group to match num_heads
    # vllm/gptq-baseline does the same for GQA ref
    expanded_k_varlen = k_varlen.repeat_interleave(gqa_group_size, dim=1)
    expanded_v_varlen = v_varlen.repeat_interleave(gqa_group_size, dim=1)
    # shape: [N, num_heads, head_dim]

    out_ref_varlen, _ = block_streaming_attention_ref(
        q_varlen,
        expanded_k_varlen,
        expanded_v_varlen,
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
    assert out_cuda.shape == (
        batch_size,
        seqlen,
        num_heads,
        head_dim,
    ), f"Expected shape {(batch_size, seqlen, num_heads, head_dim)}, got {out_cuda.shape}"

    # Check output values
    torch.testing.assert_close(out_cuda, out_ref, atol=5e-1, rtol=5e-1)

    print(
        f"Paged streaming attention with GQA test passed for seqlen={seqlen}, dtype={dtype}, batch_size={batch_size}, num_heads={num_heads}, gqa_group_size={gqa_group_size}!"
    )


@pytest.mark.skipif(
    not is_hopper(), reason="Streaming attention requires Hopper GPU (SM 9.0)"
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096])
def test_streaming_attention_gqa_with_varlen(seqlen):
    """
    这个测试专门验证 CUDA Kernel 在 GQA (`pack_gqa=True`) 和
    Varlen (`cu_seqlens_q` is not None) 两种模式同时开启时的正确性。
    这精确地模拟了 sglang 集成环境中的失败场景。
    """
    # 1. 参数设置 (使用较小的值以便快速调试)
    device = torch.device("cuda")
    batch_size = 2
    page_size = 32
    sink_size = 8
    local_size = 16
    gqa_group_size = 2
    num_heads = 4
    num_kv_heads = num_heads // gqa_group_size
    head_dim = 64
    dtype = torch.bfloat16

    # 2. 创建输入数据
    # Q: [B, seqlen, num_heads, head_dim]
    q_padded = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    # K/V: [B, seqlen, num_kv_heads, head_dim]
    k_padded = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_padded = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=dtype, device=device)

    # 3. 构造 Varlen 输入 (这是与之前测试的核心区别)
    total_tokens = batch_size * seqlen
    
    # Q (Varlen): [total_tokens, num_heads, head_dim]
    q_varlen = q_padded.reshape(total_tokens, num_heads, head_dim)
    
    # K/V (Varlen): [total_tokens, num_kv_heads, head_dim]
    k_varlen = k_padded.reshape(total_tokens, num_kv_heads, head_dim)
    v_varlen = v_padded.reshape(total_tokens, num_kv_heads, head_dim)
    
    # cu_seqlens_q: 描述 Varlen 张量的序列边界
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )

    # 4. 构造 Paged K/V Cache
    num_blocks_per_seq = (seqlen + page_size - 1) // page_size
    max_num_blocks = num_blocks_per_seq * batch_size * 2
    
    k_cache = torch.zeros(max_num_blocks, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(max_num_blocks, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    page_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
    all_random_indices = torch.randperm(max_num_blocks, device=device, dtype=torch.int32)

    for b in range(batch_size):
        start_idx = b * num_blocks_per_seq
        end_idx = start_idx + num_blocks_per_seq
        available_indices = all_random_indices[start_idx:end_idx]
        page_table[b] = available_indices

        for i, block_idx in enumerate(available_indices):
            start_token = i * page_size
            end_token = min((i + 1) * page_size, seqlen)
            if (valid_len := end_token - start_token) > 0:
                k_cache[block_idx, :valid_len] = k_padded[b, start_token:end_token]
                v_cache[block_idx, :valid_len] = v_padded[b, start_token:end_token]

    # 5. 调用待测的 CUDA Kernel
    softmax_scale = 1.0 / math.sqrt(head_dim)
    seqused_k = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)

    # 核心调用：使用 q_varlen 和 cu_seqlens_q
    out_cuda_varlen, _ = streaming_sparse_attn_func(
        q_varlen,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,  # <-- 使用 Varlen
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(local_size - 1, 0),
        sink_size=sink_size,
        enable_streaming=True,
        pack_gqa=True,  # <-- 开启 GQA
        position_ids=None,
        # 其他参数
        learnable_sink=None, softcap=0.0, groupwise=False, m_block_size=128, n_block_size=page_size,
    )
    # Kernel 返回的是 Varlen 格式，需要 reshape 以便比较
    out_cuda = out_cuda_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # 6. 调用 PyTorch 参考实现
    head_mask_type = torch.full((num_heads,), -1, dtype=torch.int32, device=device)
    
    # 参考实现需要手动扩展 K/V head
    expanded_k_varlen = k_varlen.repeat_interleave(gqa_group_size, dim=1)
    expanded_v_varlen = v_varlen.repeat_interleave(gqa_group_size, dim=1)

    out_ref_varlen, _ = block_streaming_attention_ref(
        q=q_varlen,
        k=expanded_k_varlen,
        v=expanded_v_varlen,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_q,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        head_mask_type=head_mask_type,
        sink_size=sink_size,
        local_size=local_size,
        softmax_scale=softmax_scale,
        is_causal=True,
    )
    out_ref = out_ref_varlen.reshape(batch_size, seqlen, num_heads, head_dim)

    # 7. 断言比较结果
    print("Shape of CUDA output:", out_cuda.shape)
    print("Shape of Reference output:", out_ref.shape)
    assert out_cuda.shape == out_ref.shape, "Output shapes do not match!"

    # 使用一个相对严格的容差来捕捉计算错误
    torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)

    print(f"Paged streaming attention with GQA and Varlen test passed!")


