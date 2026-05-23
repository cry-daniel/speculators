from __future__ import annotations

from typing import Iterable

import torch


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def block_sparse_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: Iterable[int],
    block_size: int,
) -> torch.Tensor:
    seq_len = k.shape[-2]
    indices = []
    for block in sorted(set(int(b) for b in selected_blocks)):
        start = max(0, block * block_size)
        end = min(seq_len, start + block_size)
        if start < end:
            indices.extend(range(start, end))
    if not indices:
        return torch.zeros_like(q)
    device = k.device
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    return dense_attention(q, k.index_select(-2, idx), v.index_select(-2, idx))


def estimate_hbm_bytes(batch: int, heads: int, selected_tokens: int, head_dim: int, dtype_bytes: int) -> int:
    return batch * heads * selected_tokens * head_dim * dtype_bytes * 2

