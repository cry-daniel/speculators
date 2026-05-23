#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment, write_json
from pams.memory import estimate_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    args = parser.parse_args()
    estimate = estimate_memory(
        args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    out = EXPERIMENTS / "00_env" / "memory_estimate.json"
    write_json(out, estimate)
    status = "completed"
    if args.max_model_len >= 16384 and not estimate["safe_16384_with_20pct_headroom"]:
        status = "unsafe_16384_context"
    summary = [
        "# Phase 0 Memory Estimate",
        "",
        f"- Model: `{args.model}`",
        f"- Dtype: `{args.dtype}`",
        f"- GPU memory utilization: `{args.gpu_memory_utilization}`",
        f"- Estimated weight memory GB: `{estimate['model_weight_memory_gb']}`",
        f"- KV bytes/token: `{estimate['kv_bytes_per_token']}`",
        f"- Max KV tokens under budget: `{estimate['max_kv_tokens_under_budget']}`",
        f"- Requested KV tokens: `{estimate['requested_kv_tokens']}`",
        f"- Requested headroom ratio: `{estimate['requested_headroom_ratio']}`",
        f"- Recommended max_model_len: `{estimate['recommended_max_model_len']}`",
        f"- Recommended max_num_seqs: `{estimate['recommended_max_num_seqs']}`",
        "",
        "OOM degrade order: max_num_seqs, max_model_len, num_prompts, dtype_or_kv_cache_dtype.",
    ]
    register_experiment(
        "00_env",
        config=vars(args),
        command="python scripts/estimate_memory.py --model Qwen/Qwen3-8B --dtype bfloat16",
        status=status,
        summary="\n".join(summary),
        metadata_extra={"memory_estimate_json": str(out), "memory_estimate": estimate},
        model_name=args.model,
        model_dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    print(estimate)


if __name__ == "__main__":
    main()

