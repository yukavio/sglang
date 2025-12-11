import os
import json
import numpy as np
from typing import List, Optional, Tuple

def load_attn_pattern(attn_load_dir: str) -> Tuple[List[List[List[int]]], int, int]:
    full_attention_heads = np.loadtxt(
        os.path.join(attn_load_dir, "full_attention_heads.tsv"),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    config = json.load(open(os.path.join(attn_load_dir, "config.json")))
    sink_size = config["sink_size"]
    recent_size = config["recent_size"]
    return full_attention_heads, sink_size, recent_size


def sparsify_attention_heads(
    full_attention_heads, 
    threshold=None,
    sparsity=None
):
    # add a very small random noise to full_attention_heads to break ties
    full_attention_heads += np.random.uniform(0, 1e-6, full_attention_heads.shape)

    if sparsity is not None:
        # ignore the threshold and use the sparsity
        # set the sparsity small values to 0 and others to 1
        threshold = np.quantile(full_attention_heads, sparsity)
    else:
        assert threshold is not None, "Either threshold or sparsity must be provided"

    if sparsity >= 1:
        # all heads are pruned
        threshold = 2
    if sparsity <= 0:
        # no heads are pruned
        threshold = -1

    full_attention_heads = (full_attention_heads >= threshold).astype(float)
    sparsity = 1 - np.mean(full_attention_heads)
    return full_attention_heads, sparsity


def count_attention_head_types(
    full_attention_heads: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统计每个 layer 中 full attention heads 和 streaming attention heads 的数量。
    
    Args:
        full_attention_heads: 二维数组，形状为 [num_layers, num_heads]，
                             每个值表示该 head 是 full attention 的概率
        threshold: 阈值，用于判断是 full attention 还是 streaming attention。
                   如果概率值 >= threshold，则认为是 full attention head
    
    Returns:
        full_heads_count: 一维数组，每个值表示每个 layer 的 full attention heads 数量
        streaming_heads_count: 一维数组，每个值表示每个 layer 的 streaming attention heads 数量
    """
    # 将概率值转换为二值判断：>= threshold 为 full attention (1)，否则为 streaming attention (0)
    is_full_attention = (full_attention_heads >= threshold).astype(int)
    
    # 统计每个 layer 的 full attention heads 数量（即值为 1 的数量）
    full_heads_count = np.sum(is_full_attention, axis=1)
    
    # 统计每个 layer 的 streaming attention heads 数量（即值为 0 的数量）
    streaming_heads_count = full_attention_heads.shape[1] - full_heads_count
    
    return full_heads_count, streaming_heads_count
