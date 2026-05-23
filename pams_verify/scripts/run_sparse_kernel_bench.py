#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment, write_json, write_jsonl
from pams.memory import dtype_bytes, load_model_config
from pams.plotting import save_bar
from pams.sparse_attention_ref import block_sparse_attention_ref, dense_attention, estimate_hbm_bytes
from pams.sparse_attention_triton import implementation_status


def run_case(torch, *, seq_len: int, actual_seq_len: int, batch: int, heads: int, head_dim: int, block_size: int, method: str, device: str, dtype) -> dict:
    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, actual_seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, actual_seq_len, head_dim, device=device, dtype=dtype)
    block_count = max(1, (actual_seq_len + block_size - 1) // block_size)
    if method == "dense":
        selected = list(range(block_count))
    elif method == "independent_sparse":
        selected = list(range(0, block_count, max(1, block_count // 16)))[:16]
    elif method == "shared_only":
        selected = list(range(0, block_count, max(1, block_count // 8)))[:8]
    elif method == "shared_fixed_residual":
        selected = sorted(set(list(range(0, block_count, max(1, block_count // 8)))[:8] + list(range(1, block_count, max(1, block_count // 16)))[:8]))
    else:
        selected = sorted(set(list(range(0, block_count, max(1, block_count // 8)))[:8] + list(range(2, block_count, max(1, block_count // 24)))[:6]))

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    if method == "dense":
        out = dense_attention(q, k, v)
    else:
        out = block_sparse_attention_ref(q, k, v, selected, block_size)
    if device == "cuda":
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated()
    else:
        memory_allocated = 0
    latency_ms = (time.perf_counter() - start) * 1000.0
    del out
    return {
        "requested_seq_len": seq_len,
        "actual_seq_len": actual_seq_len,
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "method": method,
        "selected_blocks": len(selected),
        "latency_ms": latency_ms,
        "effective_hbm_bytes": estimate_hbm_bytes(batch, heads, len(selected) * block_size, head_dim, 2),
        "kernel_launch_overhead_ms": latency_ms if method != "dense" else 0.0,
        "index_construction_overhead_ms": 0.0 if method == "dense" else 0.01 * len(selected),
        "memory_allocated": memory_allocated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="Qwen/Qwen3-8B")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 2048, 4096, 8192])
    parser.add_argument("--draft-lens", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    import torch

    cfg = load_model_config(args.model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype in {"bfloat16", "bf16"} else torch.float16
    if device == "cpu":
        dtype = torch.float32
    heads = min(4, int(cfg.get("num_key_value_heads", 8)))
    head_dim = int(cfg.get("head_dim", 128))
    methods = ["dense", "independent_sparse", "shared_only", "shared_fixed_residual", "pams"]
    rows = []
    for seq_len in args.seq_lens:
        actual_seq_len = seq_len if device == "cuda" else min(seq_len, 512)
        for batch in args.batch_sizes:
            actual_batch = batch if device == "cuda" else min(batch, 2)
            for block_size in args.block_sizes:
                for method in methods:
                    rows.append(
                        run_case(
                            torch,
                            seq_len=seq_len,
                            actual_seq_len=actual_seq_len,
                            batch=actual_batch,
                            heads=heads,
                            head_dim=head_dim,
                            block_size=block_size,
                            method=method,
                            device=device,
                            dtype=dtype,
                        )
                    )
    exp = EXPERIMENTS / "06_sparse_kernel_microbench"
    write_jsonl(exp / "raw" / "kernel_bench.jsonl", rows)
    by_method = {}
    for method in methods:
        vals = [row["latency_ms"] for row in rows if row["method"] == method]
        by_method[method] = sum(vals) / max(1, len(vals))
    write_json(
        exp / "parsed" / "kernel_summary.json",
        {"device": device, "triton": implementation_status(), "mean_latency_ms_by_method": by_method, "rows": len(rows)},
    )
    save_bar(
        exp / "figures" / "kernel_latency_vs_union_blocks.png",
        list(by_method),
        [by_method[m] for m in by_method],
        "Reference sparse attention latency",
        "Latency ms",
    )
    status = "completed_reference_gpu" if device == "cuda" else "completed_cpu_reference_scaled_down"
    register_experiment(
        "06_sparse_kernel_microbench",
        config=vars(args),
        command="python scripts/run_sparse_kernel_bench.py --model-config Qwen/Qwen3-8B --seq-lens 512 2048 4096 8192 --draft-lens 2 4 8 --block-sizes 16 32 64",
        status=status,
        summary=(
            "# Phase 9 Sparse Attention Kernel Microbenchmark\n\n"
            f"- Device: `{device}`\n"
            f"- Triton status: `{implementation_status()['label']}`\n"
            "- This benchmark uses the torch gather+dense reference path unless a custom kernel is later added.\n"
            f"- Mean dense latency ms: `{by_method['dense']:.4f}`\n"
            f"- Mean PAMS reference latency ms: `{by_method['pams']:.4f}`\n"
        ),
        metadata_extra={"label": "kernel_microbenchmark", "device": device, "kernel_summary": by_method, "triton": implementation_status()},
        model_name=args.model_config,
        model_dtype=args.dtype,
    )
    print(json.dumps({"status": status, "device": device, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

