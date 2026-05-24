#!/usr/bin/env python3
"""Proxy microbenchmark for sparse verifier KV layouts."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path
from typing import Any


def estimate_hbm_bytes(
    union_blocks: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int,
) -> int:
    return num_layers * num_kv_heads * union_blocks * block_size * head_dim * bytes_per_elem * 2


def union_blocks_for_layout(
    layout: str,
    num_blocks: int,
    k: int,
    topk_per_token: int,
    shared_budget: int,
    private_max: int,
) -> int:
    if layout == "dense_verifier":
        return num_blocks
    if layout == "independent_topk":
        return min(num_blocks, topk_per_token * k)
    if layout in {"snapkv_static", "shared_only"}:
        return min(num_blocks, shared_budget)
    if layout in {"speclink_fixed", "speclink_prob", "speclink_prob_fallback"}:
        return min(num_blocks, shared_budget + private_max * k)
    raise ValueError(f"unsupported layout: {layout}")


def sync(device: str, torch: Any) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_one(args: argparse.Namespace, layout: str, k: int, context_len: int, block_size: int) -> dict[str, Any]:
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    num_blocks = max(1, math.ceil(context_len / block_size))
    union_blocks = union_blocks_for_layout(
        layout,
        num_blocks,
        k,
        args.topk_per_token,
        args.shared_budget,
        args.private_max,
    )
    selected_tokens = max(1, union_blocks * block_size)
    q = torch.randn(k, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    key = torch.randn(selected_tokens, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    value = torch.randn(selected_tokens, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)

    def step() -> Any:
        scores = torch.einsum("khd,thd->kht", q, key) / math.sqrt(args.head_dim)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype)
        return torch.einsum("kht,thd->khd", probs, value)

    for _ in range(args.warmup):
        step()
    sync(device, torch)
    times: list[float] = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        step()
        sync(device, torch)
        times.append((time.perf_counter() - start) * 1000.0)
    bytes_estimate = estimate_hbm_bytes(
        union_blocks,
        block_size,
        args.num_layers,
        args.num_kv_heads,
        args.head_dim,
        args.bytes_per_elem,
    )
    return {
        "backend": "torch_union_attention_proxy",
        "device": device,
        "layout": layout,
        "num_spec_tokens": k,
        "context_len": context_len,
        "block_size": block_size,
        "union_blocks": union_blocks,
        "selected_tokens": selected_tokens,
        "kernel_or_proxy_time_ms_mean": statistics.mean(times),
        "kernel_or_proxy_time_ms_p50": statistics.median(times),
        "bytes_estimate": bytes_estimate,
        "achieved_tokens_per_s_proxy": (k / (statistics.mean(times) / 1000.0)) if times else None,
        "index_overhead_ms": None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "backend",
        "device",
        "layout",
        "num_spec_tokens",
        "context_len",
        "block_size",
        "union_blocks",
        "kernel_or_proxy_time_ms_mean",
        "bytes_estimate",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--layouts",
        default="dense_verifier,independent_topk,snapkv_static,shared_only,speclink_prob",
    )
    parser.add_argument("--k-values", default="2,4,8,12")
    parser.add_argument("--context-lens", default="512,2048,4096,8192")
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--topk-per-token", type=int, default=8)
    parser.add_argument("--shared-budget", type=int, default=4)
    parser.add_argument("--private-max", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bytes-per-elem", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = []
    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    for context_len in parse_ints(args.context_lens):
        for block_size in parse_ints(args.block_sizes):
            for k in parse_ints(args.k_values):
                for layout in layouts:
                    rows.append(bench_one(args, layout, k, context_len, block_size))
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
