#!/usr/bin/env python3
"""Summarize SR24 all_corrected_24 backend/device evidence from result roots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing summary.csv under {root}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["root"] = str(root.resolve())
            rows.append(row)
    return rows


def classify(row: dict[str, Any], dense: dict[str, Any] | None) -> str:
    if row.get("method") != "all_corrected_24":
        return ""
    if as_float(row.get("full_batch_output_tokens_per_second")) is None:
        return "no_serving_result"
    parts: list[str] = []
    backend = str(row.get("sr24_residual_backend") or "")
    device_counts = str(row.get("sr24_residual_device_counts") or "")
    runtime_gpu = str(row.get("sr24_compressed_residual_runtime_on_gpu") or "")
    non_gpu = str(row.get("sr24_compressed_residual_non_gpu_modules") or "")
    speedup = as_float(row.get("full_speedup_vs_dense"))
    accepted = as_float(row.get("spec_avg_accepted_draft_tokens_per_step"))
    dense_accepted = (
        as_float(dense.get("spec_avg_accepted_draft_tokens_per_step"))
        if dense is not None
        else None
    )
    if backend == "compressed_dense":
        if "cuda" in device_counts and runtime_gpu == "True" and not non_gpu:
            parts.append("compressed_dense_gpu_resident")
        else:
            parts.append("compressed_dense_gpu_not_proven")
    if accepted is not None and dense_accepted is not None:
        if accepted >= dense_accepted * 0.95:
            parts.append("acceptance_intact")
        else:
            parts.append("acceptance_lower")
    if speedup is not None:
        if speedup >= 1.0:
            parts.append("not_slower_than_dense")
        else:
            parts.append("slower_than_dense")
    if backend in {"torch_sparse", "compressed_dense"}:
        parts.append("two_pass_exact_correction")
    return ",".join(parts)


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "root",
        "method",
        "batch_size",
        "full_batch_tps",
        "dense_full_batch_tps",
        "full_speedup_vs_dense",
        "total_tps",
        "accepted_per_step",
        "dense_accepted_per_step",
        "gpu_util",
        "graph_counts",
        "residual_backend",
        "residual_device",
        "residual_backend_counts",
        "residual_device_counts",
        "compressed_residual_runtime_on_gpu",
        "compressed_residual_non_gpu_modules",
        "diagnosis",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    for root in args.roots:
        raw_rows.extend(read_rows(Path(root)))

    dense_by_case: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        if row.get("method") == "dense_baseline":
            dense_by_case[(row["root"], row.get("dataset", ""), row.get("batch_size", ""))] = row

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("method") not in {"dense_baseline", "all_corrected_24"}:
            continue
        dense = dense_by_case.get(
            (row["root"], row.get("dataset", ""), row.get("batch_size", ""))
        )
        full_tps = as_float(row.get("full_batch_output_tokens_per_second"))
        dense_full = (
            as_float(dense.get("full_batch_output_tokens_per_second"))
            if dense is not None
            else None
        )
        row["full_speedup_vs_dense"] = (
            full_tps / dense_full if full_tps is not None and dense_full else None
        )
        row["diagnosis"] = classify(row, dense)
        rows.append({
            "root": row["root"],
            "method": row.get("method", ""),
            "batch_size": row.get("batch_size", ""),
            "full_batch_tps": row.get("full_batch_output_tokens_per_second", ""),
            "dense_full_batch_tps": dense.get("full_batch_output_tokens_per_second", "") if dense else "",
            "full_speedup_vs_dense": row["full_speedup_vs_dense"],
            "total_tps": row.get("total_output_tokens_per_second", ""),
            "accepted_per_step": row.get("spec_avg_accepted_draft_tokens_per_step", ""),
            "dense_accepted_per_step": dense.get("spec_avg_accepted_draft_tokens_per_step", "") if dense else "",
            "gpu_util": row.get("avg_gpu_util_pct", ""),
            "graph_counts": row.get("server_cudagraph_profile_counts", "") or row.get("sr24_cudagraph_mode_counts", ""),
            "residual_backend": row.get("sr24_residual_backend", ""),
            "residual_device": row.get("sr24_residual_device", ""),
            "residual_backend_counts": row.get("sr24_residual_backend_counts", ""),
            "residual_device_counts": row.get("sr24_residual_device_counts", ""),
            "compressed_residual_runtime_on_gpu": row.get("sr24_compressed_residual_runtime_on_gpu", ""),
            "compressed_residual_non_gpu_modules": row.get("sr24_compressed_residual_non_gpu_modules", ""),
            "diagnosis": row["diagnosis"],
        })

    corrected = [row for row in rows if row["method"] == "all_corrected_24"]
    measured_corrected = [
        row for row in corrected if row.get("diagnosis") != "no_serving_result"
    ]
    gpu_resident = [
        row for row in measured_corrected
        if "compressed_dense_gpu_resident" in row.get("diagnosis", "")
    ]
    slower_intact = [
        row for row in measured_corrected
        if (
            "slower_than_dense" in row.get("diagnosis", "")
            and "acceptance_intact" in row.get("diagnosis", "")
        )
    ]
    write_csv(output_root / "allcorrected_diagnosis.csv", rows)

    report = [
        "# SR24 all_corrected_24 backend diagnosis",
        "",
        f"- all_corrected_24 rows analyzed: `{len(corrected)}`",
        f"- all_corrected_24 rows with serving throughput: `{len(measured_corrected)}`",
        f"- compressed_dense rows proven GPU-resident: `{len(gpu_resident)}`",
        f"- slower-than-dense rows with intact acceptance: `{len(slower_intact)}`",
        "",
        "Interpretation: GPU-resident `compressed_dense` rules out CPU residual "
        "fallback as the main bottleneck.  When accepted draft length is intact "
        "but throughput is below dense, the current exact path is losing to "
        "two-pass sparse-base plus residual-correction work.",
        "",
        "| root | backend | full tok/s | dense full tok/s | speedup | accepted/step | dense accepted/step | GPU util | graph | device counts | diagnosis |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in corrected:
        report.append(
            "| "
            + " | ".join([
                Path(str(row["root"])).name,
                str(row["residual_backend"]),
                fmt(row["full_batch_tps"]),
                fmt(row["dense_full_batch_tps"]),
                fmt(row["full_speedup_vs_dense"]),
                fmt(row["accepted_per_step"]),
                fmt(row["dense_accepted_per_step"]),
                fmt(row["gpu_util"]),
                str(row["graph_counts"]),
                str(row["residual_device_counts"]),
                str(row["diagnosis"]),
            ])
            + " |"
        )
    report.extend([
        "",
        f"CSV: `{(output_root / 'allcorrected_diagnosis.csv').resolve()}`",
    ])
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output_root.resolve())


if __name__ == "__main__":
    main()
