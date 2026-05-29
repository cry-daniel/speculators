#!/usr/bin/env python3
"""Fit a simple confidence-to-acceptance calibration table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.speclink_cv.core import find_trace_files, write_csv, write_json


def collect_rows(trace_root: Path, workloads: set[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in find_trace_files(trace_root):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("dataset_label")) not in workloads:
                continue
            if item.get("reached") != 1 or item.get("accepted_local") is None:
                continue
            rows.append(
                {
                    "workload": str(item.get("dataset_label")),
                    "model_label": str(item.get("model_label")),
                    "method": str(item.get("method")),
                    "draft_position": int(item.get("draft_position") or 0),
                    "confidence": float(item.get("draft_selected_prob") or 0.0),
                    "accepted": int(item.get("accepted_local") or 0),
                    "dataset_index": int(item.get("dataset_index") or 0),
                }
            )
    return rows


def bin_index(confidence: float, num_bins: int) -> int:
    return min(num_bins - 1, max(0, int(confidence * num_bins)))


def fit(rows: list[dict[str, float | int | str]], num_bins: int) -> dict:
    bins = [
        {"bin": idx, "confidence_sum": 0.0, "accepted_sum": 0.0, "count": 0}
        for idx in range(num_bins)
    ]
    for row in rows:
        idx = bin_index(float(row["confidence"]), num_bins)
        bins[idx]["confidence_sum"] += float(row["confidence"])
        bins[idx]["accepted_sum"] += float(row["accepted"])
        bins[idx]["count"] += 1
    table = []
    global_rate = (
        sum(float(row["accepted"]) for row in rows) / len(rows) if rows else 0.5
    )
    for item in bins:
        count = int(item["count"])
        table.append(
            {
                "bin": item["bin"],
                "left": item["bin"] / num_bins,
                "right": (item["bin"] + 1) / num_bins,
                "count": count,
                "mean_confidence": item["confidence_sum"] / count if count else math.nan,
                "acceptance_rate": item["accepted_sum"] / count if count else global_rate,
            }
        )
    return {
        "model_type": "binning",
        "num_bins": num_bins,
        "global_acceptance_rate": global_rate,
        "bins": table,
        "notes": "Simple empirical binning over reached draft tokens only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", default="math,mtbench")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--split", choices=["all", "even", "odd"], default="even")
    args = parser.parse_args()
    workloads = {item.strip() for item in args.workloads.split(",") if item.strip()}
    rows = collect_rows(args.trace_root, workloads)
    if args.split == "even":
        rows = [row for row in rows if int(row["dataset_index"]) % 2 == 0]
    elif args.split == "odd":
        rows = [row for row in rows if int(row["dataset_index"]) % 2 == 1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = fit(rows, args.num_bins)
    write_json(args.output_dir / "calibration_model.json", model)
    write_csv(args.output_dir / "calibration_table.csv", model["bins"])
    (args.output_dir / "calibration_report.md").write_text(
        "# Confidence Calibration\n\n"
        f"- split: {args.split}\n"
        f"- rows: {len(rows)}\n"
        f"- bins: {args.num_bins}\n"
        f"- global acceptance rate: {model['global_acceptance_rate']:.4f}\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote {args.output_dir / 'calibration_model.json'}")


if __name__ == "__main__":
    main()
