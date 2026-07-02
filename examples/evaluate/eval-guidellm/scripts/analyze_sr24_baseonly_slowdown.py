#!/usr/bin/env python3
"""Summarize why SR24 base_only_24 is slow or fast in existing runs.

This is intentionally an offline reducer over matrix-run artifacts. It does not
launch vLLM. Point it at one or more result roots that contain `summary.csv`;
if a root also has `run_config.json`, the report includes the relevant serving
configuration knobs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(row: dict[str, Any], key: str) -> int | None:
    value = _float(row, key)
    if value is None:
        return None
    return int(value)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _graph_counts(row: dict[str, Any], root: Path, method: str,
                  batch_size: int | None) -> dict[str, int]:
    counts = _json_dict(row.get("sr24_cudagraph_mode_counts"))
    if counts:
        return {str(k): int(v) for k, v in counts.items()}

    # Some clean runs store the graph summary only in the work root.
    run_config = _read_json(root / "run_config.json") or {}
    work_root_value = run_config.get("work_root")
    if not work_root_value or not method or batch_size is None:
        return {}
    work_root = Path(work_root_value)
    candidates = list(work_root.glob(
        f"{method}/bs{batch_size}/rep*/**/cudagraph_stats.jsonl"))
    candidates += list(work_root.glob(
        f"{method}/bs{batch_size}/rep*/cudagraph_stats.jsonl"))
    if not candidates:
        return {}
    latest: dict[str, int] = {}
    for line in candidates[0].read_text().splitlines():
        event = _json_dict(line)
        event_counts = _json_dict(event.get("cudagraph_mode_counts"))
        if event_counts:
            latest = {str(k): int(v) for k, v in event_counts.items()}
    return latest


def _none_frac(counts: dict[str, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    return float(counts.get("NONE", 0)) / float(total)


def _full_frac(counts: dict[str, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    return float(counts.get("FULL", 0)) / float(total)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _collect_root(root: Path) -> list[dict[str, Any]]:
    summary_path = root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.csv under {root}")

    run_config = _read_json(root / "run_config.json") or {}
    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            method = str(row.get("method") or "")
            batch_size = _int(row, "batch_size")
            counts = _graph_counts(row, root, method, batch_size)
            rows.append({
                "root": str(root.resolve()),
                "method": method,
                "dataset": row.get("dataset") or "",
                "batch_size": batch_size,
                "status": row.get("status") or "",
                "max_tokens": _int(row, "max_new_tokens"),
                "max_num_batched_tokens": run_config.get("max_num_batched_tokens"),
                "disable_chunked_prefill": run_config.get("disable_chunked_prefill"),
                "vllm_compilation_config": run_config.get("vllm_compilation_config") or "",
                "target_leafs": row.get("sr24_target_leafs") or run_config.get("sr24_target_leafs") or "",
                "module_count_attached": _int(row, "sr24_module_count_attached"),
                "full_batch_tps": _float(row, "full_batch_output_tokens_per_second"),
                "total_tps": _float(row, "total_output_tokens_per_second"),
                "tpot_ms": _float(row, "tpot_ms_mean"),
                "gpu_util": _float(row, "avg_gpu_util_pct"),
                "accept_rate": _float(row, "spec_acceptance_rate"),
                "accepted_per_step": _float(row, "spec_avg_accepted_draft_tokens_per_step"),
                "selected_per_step": _float(row, "spec_avg_selected_draft_tokens_per_step"),
                "cudagraph_counts": counts,
                "cudagraph_full_frac": _full_frac(counts),
                "cudagraph_none_frac": _none_frac(counts),
            })
    return rows


def _diagnose(row: dict[str, Any], dense: dict[str, Any] | None) -> str:
    if row["method"] != "base_only_24":
        return ""
    parts: list[str] = []
    speedup = row.get("speedup_vs_dense")
    base_acc = row.get("accepted_per_step")
    dense_acc = dense.get("accepted_per_step") if dense else None
    base_util = row.get("gpu_util")
    dense_util = dense.get("gpu_util") if dense else None
    none_frac = row.get("cudagraph_none_frac")
    modules = row.get("module_count_attached")

    if speedup is not None and speedup < 1.0:
        if base_acc is not None and dense_acc is not None and base_acc >= dense_acc:
            parts.append("not_acceptance_limited")
        if base_util is not None and dense_util is not None and base_util + 10.0 < dense_util:
            parts.append("gpu_underutilized")
        if none_frac is not None and none_frac > 0.20:
            parts.append("many_cudagraph_NONE")
        if modules is not None and modules > 64:
            parts.append("overbroad_module_scope")
    else:
        if base_acc is not None and dense_acc is not None and base_acc >= dense_acc:
            parts.append("acceptance_ok")
        if base_util is not None and base_util >= 80.0:
            parts.append("gpu_util_ok")
        if none_frac is not None and none_frac < 0.05:
            parts.append("graph_ok")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True,
                        help="Result roots containing summary.csv")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for root_arg in args.roots:
        rows.extend(_collect_root(Path(root_arg)))

    dense_by_case: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    dense_by_dataset_batch: dict[tuple[str, int | None], dict[str, Any]] = {}
    for row in rows:
        if row["method"] in {"dense_baseline", "vllm_eagle3", "eagle3_dense"}:
            key = (row["root"], row["dataset"], row["batch_size"])
            dense_by_case.setdefault(key, row)
            fallback_key = (row["dataset"], row["batch_size"])
            dense_by_dataset_batch.setdefault(fallback_key, row)

    for row in rows:
        key = (row["root"], row["dataset"], row["batch_size"])
        dense = dense_by_case.get(key) or dense_by_dataset_batch.get(
            (row["dataset"], row["batch_size"])
        )
        dense_tps = dense.get("full_batch_tps") if dense else None
        row["dense_full_batch_tps"] = dense_tps
        row["speedup_vs_dense"] = (
            row["full_batch_tps"] / dense_tps
            if row.get("full_batch_tps") is not None and dense_tps else None
        )
        row["diagnosis"] = _diagnose(row, dense)

    csv_fields = [
        "root", "method", "dataset", "batch_size", "status", "max_tokens",
        "max_num_batched_tokens", "disable_chunked_prefill",
        "module_count_attached", "target_leafs", "full_batch_tps",
        "dense_full_batch_tps", "speedup_vs_dense", "total_tps", "tpot_ms",
        "gpu_util", "accept_rate", "accepted_per_step", "selected_per_step",
        "cudagraph_full_frac", "cudagraph_none_frac", "cudagraph_counts",
        "diagnosis",
    ]
    with (output_root / "baseonly_diagnosis.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in csv_fields}
            out["cudagraph_counts"] = json.dumps(row.get("cudagraph_counts") or {},
                                                 sort_keys=True)
            writer.writerow(out)

    base_rows = [r for r in rows if r["method"] == "base_only_24"]
    slow_rows = [
        r for r in base_rows
        if r.get("speedup_vs_dense") is not None
        and float(r["speedup_vs_dense"]) < 1.0
    ]
    fast_rows = [
        r for r in base_rows
        if r.get("speedup_vs_dense") is not None
        and float(r["speedup_vs_dense"]) >= 1.0
    ]
    slow_not_acceptance = [
        r for r in slow_rows if "not_acceptance_limited" in (r.get("diagnosis") or "")
    ]
    slow_gpu_graph = [
        r for r in slow_rows
        if (
            "gpu_underutilized" in (r.get("diagnosis") or "")
            or "many_cudagraph_NONE" in (r.get("diagnosis") or "")
        )
    ]
    healthy_fast = [
        r for r in fast_rows
        if (
            "acceptance_ok" in (r.get("diagnosis") or "")
            and "gpu_util_ok" in (r.get("diagnosis") or "")
        )
    ]
    report = [
        "# SR24 base_only_24 slowdown diagnosis",
        "",
        "This report compares existing runs and classifies base-only slowdown "
        "using full-batch throughput, accepted draft tokens per step, GPU "
        "utilization, CUDA Graph FULL/NONE counts, and attached SR24 module "
        "scope.",
        "",
        "## Aggregate Read",
        "",
        f"- base_only_24 rows analyzed: `{len(base_rows)}`",
        f"- rows slower than dense: `{len(slow_rows)}`",
        f"- slower rows that are not acceptance-limited: `{len(slow_not_acceptance)}`",
        f"- slower rows with GPU underutilization or CUDA Graph loss: `{len(slow_gpu_graph)}`",
        f"- rows faster than dense with healthy acceptance/GPU util: `{len(healthy_fast)}`",
        "",
        "Interpretation: if the slower rows still accept at least as many draft "
        "tokens as dense, the issue is not speculative acceptance. If the "
        "healthy fast rows exist for narrower/graph-enabled scopes, base sparse "
        "compute has real headroom; the failed rows are serving-shape, graph, "
        "or overbroad-scope problems.",
        "",
        "| root | full-batch tok/s | speedup | accepted/step | dense accepted/step | GPU util | graph FULL/NONE | modules | diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in base_rows:
        key = (row["root"], row["dataset"], row["batch_size"])
        dense = dense_by_case.get(key) or dense_by_dataset_batch.get(
            (row["dataset"], row["batch_size"])
        ) or {}
        counts = row.get("cudagraph_counts") or {}
        report.append(
            "| "
            + " | ".join([
                Path(str(row["root"])).name,
                _fmt(row.get("full_batch_tps"), 3),
                _fmt(row.get("speedup_vs_dense"), 3),
                _fmt(row.get("accepted_per_step"), 3),
                _fmt(dense.get("accepted_per_step"), 3),
                _fmt(row.get("gpu_util"), 3),
                json.dumps(counts, sort_keys=True),
                _fmt(row.get("module_count_attached"), 0),
                row.get("diagnosis") or "",
            ])
            + " |"
        )

    report += [
        "",
        "## Read",
        "",
        "- If `accepted/step` is not lower than dense, the base-only run is not "
        "slow because accepted length collapsed.",
        "- If GPU utilization is far below dense and CUDA Graph `NONE` is high, "
        "the slowdown is a serving/graph/config issue rather than the sparse "
        "kernel's upper bound.",
        "- If the attached module count is broad, compare it against the "
        "narrow MLP-only rows before drawing conclusions about the intended "
        "`base_only_24` upper bound.",
        "",
        f"CSV: `{(output_root / 'baseonly_diagnosis.csv').resolve()}`",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
