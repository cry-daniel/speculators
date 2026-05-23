#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.attention_trace import collect_synthetic_traces
from pams.configs import EXPERIMENTS, register_experiment, write_json
from pams.workloads import WORKLOADS, generate_workload, split_rows, specs_as_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--workloads", nargs="+", default=["short_chat", "medium_sharegpt_like", "long_rag_4k"])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--splits", nargs="+", default=["calibration", "validation", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--draft-len", type=int, default=4)
    args = parser.parse_args()
    exp = EXPERIMENTS / "02_trace_collection"
    raw = exp / "raw"
    workload_dir = raw / "workloads"
    workload_dir.mkdir(parents=True, exist_ok=True)
    for name in args.workloads:
        rows = generate_workload(WORKLOADS[name], args.seed)
        from pams.configs import write_jsonl

        write_jsonl(workload_dir / f"{name}.jsonl", rows)
        for split, split_rows_ in split_rows(rows, args.seed).items():
            write_jsonl(workload_dir / f"{name}_{split}.jsonl", split_rows_)
    paths = collect_synthetic_traces(
        raw,
        workload_names=args.workloads,
        seed=args.seed,
        block_sizes=args.block_sizes,
        draft_len=args.draft_len,
    )
    counts = {}
    for split, path in paths.items():
        counts[split] = sum(1 for _ in path.open("r", encoding="utf-8"))
    write_json(exp / "parsed" / "trace_counts.json", counts)
    write_json(exp / "parsed" / "workloads.json", specs_as_dict())
    summary = [
        "# Phase 4 Trace Collection",
        "",
        "Current run used deterministic synthetic fallback traces because online vLLM/HF draft attention extraction is not available in the sandboxed process.",
        "Dense target labels in these traces are synthetic audit labels and are not used by the online planner.",
        "",
        "Trace files:",
    ]
    summary.extend(f"- {split}: `{path}` rows={counts[split]}" for split, path in paths.items())
    register_experiment(
        "02_trace_collection",
        config=vars(args),
        command="python scripts/run_trace_collection.py --target-model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B --workloads short_chat medium_sharegpt_like long_rag_4k --max-model-len 8192 --splits calibration validation test",
        status="completed_synthetic_fallback",
        summary="\n".join(summary),
        metadata_extra={"trace_counts": counts, "trace_source": "synthetic_fallback"},
        model_name=args.target_model,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    print(counts)


if __name__ == "__main__":
    main()

