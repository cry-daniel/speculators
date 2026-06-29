#!/usr/bin/env python3
"""Run a focused SR24 base_only_24 speed-ceiling sweep.

This helper does not implement a new benchmark path. It sequentially launches
the existing GuideLLM/vLLM matrix runner with different SR24 target scopes, then
aggregates the dense-vs-base-only speed ceiling into one CSV/Markdown report.

Run from examples/evaluate/eval-guidellm:

  conda run -n spec python scripts/run_sr24_baseonly_scope_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
MATRIX_RUNNER = EVAL_ROOT / "scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py"


@dataclass(frozen=True)
class ScopeCase:
    label: str
    target_leafs: str
    base_only_by_leaf: str
    gate_up_split: str = "none"


DEFAULT_CASES = [
    ScopeCase(
        label="safe_gateup16_31_down8_15",
        target_leafs="gate_up_proj,down_proj",
        base_only_by_leaf="gate_up_proj=16-31;down_proj=8-15",
    ),
    ScopeCase(
        label="gateup16_31",
        target_leafs="gate_up_proj",
        base_only_by_leaf="gate_up_proj=16-31",
    ),
    ScopeCase(
        label="down8_15",
        target_leafs="down_proj",
        base_only_by_leaf="down_proj=8-15",
    ),
    ScopeCase(
        label="down16_31",
        target_leafs="down_proj",
        base_only_by_leaf="down_proj=16-31",
    ),
    ScopeCase(
        label="gateup16_31_down16_31",
        target_leafs="gate_up_proj,down_proj",
        base_only_by_leaf="gate_up_proj=16-31;down_proj=16-31",
    ),
    ScopeCase(
        label="gateup_all",
        target_leafs="gate_up_proj",
        base_only_by_leaf="gate_up_proj=0-31",
    ),
    ScopeCase(
        label="mlp_all",
        target_leafs="gate_up_proj,down_proj",
        base_only_by_leaf="gate_up_proj=0-31;down_proj=0-31",
    ),
    ScopeCase(
        label="accuracy_tail_gateup31_up_sparse",
        target_leafs="gate_up_proj",
        base_only_by_leaf="gate_up_proj=31",
        gate_up_split="up_sparse",
    ),
]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def quote_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_summary_rows(root: Path) -> list[dict[str, str]]:
    summary_path = root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def case_command(
    args: argparse.Namespace,
    case: ScopeCase,
    case_final_root: Path,
    case_work_root: Path,
    port_base: int,
) -> list[str]:
    return [
        sys.executable,
        str(MATRIX_RUNNER),
        "--final-root",
        str(case_final_root),
        "--work-root",
        str(case_work_root),
        "--methods",
        "dense_baseline,base_only_24",
        "--datasets",
        args.dataset,
        "--batch-sizes",
        str(args.batch_size),
        "--repeats",
        "1",
        "--fixed-total-requests",
        str(args.fixed_total_requests),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--warmup-s",
        str(args.warmup_s),
        "--measurement-s",
        str(args.measurement_s),
        "--cooldown-s",
        str(args.cooldown_s),
        "--port-base",
        str(port_base),
        "--eagle3-k",
        str(args.eagle3_k),
        "--sr24-preset",
        "manual",
        "--sr24-backend",
        "torch_sparse",
        "--sr24-residual-backend",
        "dense_rows",
        "--sr24-residual-device",
        "cuda",
        "--sr24-require-gpu-residual",
        "--sr24-target-leafs",
        case.target_leafs,
        "--sr24-residual-target-leafs",
        "none",
        "--sr24-base-only-layer-ids-by-leaf",
        case.base_only_by_leaf,
        "--sr24-reduce-cpu-sync",
        "--no-sr24-sync-mask-state",
        "--sr24-static-mask-state",
        "no_residual",
        "--sr24-static-mask-buffer",
        "--sr24-allow-cudagraph",
        "--sr24-stats-interval",
        str(args.sr24_stats_interval),
        "--sr24-gate-up-split",
        case.gate_up_split,
    ]


def summarize_case(label: str, rows: list[dict[str, str]]) -> dict[str, str | float]:
    dense = next((row for row in rows if row.get("method") == "dense_baseline"), None)
    base = next((row for row in rows if row.get("method") == "base_only_24"), None)
    if dense is None or base is None:
        raise RuntimeError(f"{label}: missing dense or base_only_24 row")
    dense_full = parse_float(dense.get("full_batch_output_tokens_per_second"))
    base_full = parse_float(base.get("full_batch_output_tokens_per_second"))
    dense_total = parse_float(dense.get("total_output_tokens_per_second"))
    base_total = parse_float(base.get("total_output_tokens_per_second"))
    return {
        "case": label,
        "status_dense": dense.get("status", ""),
        "status_base_only": base.get("status", ""),
        "dense_full_batch_tps": dense_full or "",
        "base_only_full_batch_tps": base_full or "",
        "base_only_full_speedup": (
            (base_full / dense_full) if dense_full and base_full else ""
        ),
        "dense_total_tps": dense_total or "",
        "base_only_total_tps": base_total or "",
        "base_only_total_speedup": (
            (base_total / dense_total) if dense_total and base_total else ""
        ),
        "accepted_draft_per_step": (
            parse_float(base.get("spec_avg_accepted_draft_tokens_per_step")) or ""
        ),
        "avg_gpu_util_pct": parse_float(base.get("avg_gpu_util_pct")) or "",
        "cudagraph_counts": (
            base.get("sr24_cudagraph_mode_counts")
            or base.get("server_cudagraph_profile_counts")
            or ""
        ),
        "module_count": parse_float(base.get("sr24_module_count_attached")) or "",
        "storage_over_dense": parse_float(base.get("sr24_storage_over_dense")) or "",
        "target_leafs": base.get("sr24_target_leafs", ""),
        "base_only_layer_ids_by_leaf": base.get(
            "sr24_base_only_layer_ids_by_leaf", ""
        ),
    }


def write_outputs(root: Path, commands: list[str], summaries: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "case",
        "status_dense",
        "status_base_only",
        "dense_full_batch_tps",
        "base_only_full_batch_tps",
        "base_only_full_speedup",
        "dense_total_tps",
        "base_only_total_tps",
        "base_only_total_speedup",
        "accepted_draft_per_step",
        "avg_gpu_util_pct",
        "cudagraph_counts",
        "module_count",
        "storage_over_dense",
        "target_leafs",
        "base_only_layer_ids_by_leaf",
    ]
    with (root / "commands.sh").open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(command + "\n")
    with (root / "scope_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    with (root / "scope_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)

    lines = [
        "# SR24 Base-Only Scope Speed Ceiling",
        "",
        "This is a speed-ceiling sweep. It runs only `dense_baseline` and "
        "`base_only_24`; it does not claim quality safety for any sparse scope.",
        "",
        "| case | base-only full tok/s | dense full tok/s | full speedup | "
        "accepted draft/step | GPU util | modules | graph |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {case} | {base:.3f} | {dense:.3f} | {speed:.3f}x | "
            "{acc:.3f} | {util:.3f}% | {mods:.0f} | `{graph}` |".format(
                case=row["case"],
                base=float(row["base_only_full_batch_tps"] or 0.0),
                dense=float(row["dense_full_batch_tps"] or 0.0),
                speed=float(row["base_only_full_speedup"] or 0.0),
                acc=float(row["accepted_draft_per_step"] or 0.0),
                util=float(row["avg_gpu_util_pct"] or 0.0),
                mods=float(row["module_count"] or 0.0),
                graph=row["cudagraph_counts"],
            )
        )
    lines.extend(
        [
            "",
            "Read: scopes below `1.2x` base-only full-batch speedup do not have "
            "enough headroom for `speclink_t08` after residual correction, unless "
            "the mixed sparse/residual operator changes.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EVAL_ROOT / "results.bak" / f"sr24_baseonly_scope_sweep_{timestamp()}",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=EVAL_ROOT / "temp" / f"sr24_baseonly_scope_sweep_work_{timestamp()}",
    )
    parser.add_argument("--dataset", default="math_reasoning")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fixed-total-requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--measurement-s", type=float, default=8.0)
    parser.add_argument("--cooldown-s", type=float, default=1.0)
    parser.add_argument("--eagle3-k", type=int, default=8)
    parser.add_argument("--sr24-stats-interval", type=int, default=32)
    parser.add_argument("--port-base", type=int, default=8110)
    parser.add_argument(
        "--cases",
        default=",".join(case.label for case in DEFAULT_CASES),
        help="Comma-separated case labels from the built-in scope set.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only aggregate existing per-case summary.csv files under output-root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {item.strip() for item in args.cases.split(",") if item.strip()}
    cases = [case for case in DEFAULT_CASES if case.label in selected]
    missing = sorted(selected - {case.label for case in DEFAULT_CASES})
    if missing:
        raise SystemExit(f"unknown case labels: {missing}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    summaries: list[dict[str, str | float]] = []
    if args.analyze_only:
        for case in cases:
            case_final_root = args.output_root / case.label
            summaries.append(summarize_case(case.label, load_summary_rows(case_final_root)))
        write_outputs(args.output_root, commands, summaries)
        print(f"wrote {args.output_root / 'report.md'}")
        return

    for index, case in enumerate(cases):
        case_final_root = args.output_root / case.label
        case_work_root = args.work_root / case.label
        command = case_command(
            args,
            case,
            case_final_root,
            case_work_root,
            args.port_base + index * 10,
        )
        rendered = f"(cd {shlex.quote(str(EVAL_ROOT))} && {quote_cmd(command)})"
        commands.append(rendered)
        print(rendered, flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=EVAL_ROOT, check=True)
        summaries.append(summarize_case(case.label, load_summary_rows(case_final_root)))
        write_outputs(args.output_root, commands, summaries)

    if args.dry_run:
        write_outputs(args.output_root, commands, summaries)
        print(f"dry-run commands written to {args.output_root / 'commands.sh'}")
    else:
        print(f"wrote {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
