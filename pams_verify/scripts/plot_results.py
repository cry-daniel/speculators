#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS
from pams.plotting import save_bar


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EXPERIMENTS)
    args = parser.parse_args()
    union = load(args.root / "03_union_problem" / "parsed" / "union_metrics.json").get("results", [])
    if union:
        save_bar(
            args.root / "03_union_problem" / "figures" / "union_growth_vs_draft_len.png",
            [r["method"] for r in union],
            [r["union_growth_ratio"] for r in union],
            "Union growth by method",
            "Ratio",
        )
    mask = load(args.root / "05_mask_planner_offline" / "parsed" / "mask_planner_metrics.json").get("results", [])
    if mask:
        save_bar(
            args.root / "05_mask_planner_offline" / "figures" / "pareto_quality_vs_loaded_blocks.png",
            [r["method"] for r in mask],
            [r["accepted_tokens_per_loaded_block"] for r in mask],
            "Offline quality per block",
            "Accepted/block",
        )
    kernel = load(args.root / "06_sparse_kernel_microbench" / "parsed" / "kernel_summary.json").get("mean_latency_ms_by_method", {})
    if kernel:
        save_bar(
            args.root / "06_sparse_kernel_microbench" / "figures" / "kernel_latency_vs_union_blocks.png",
            list(kernel),
            [kernel[k] for k in kernel],
            "Reference kernel latency",
            "Latency ms",
        )
    # Required figures that may be unavailable still get explicit zero-valued placeholders.
    placeholders = [
        ("10_end2end/figures/end2end_itl.png", "End-to-end ITL unavailable"),
        ("10_end2end/figures/end2end_throughput.png", "End-to-end throughput unavailable"),
        ("11_correctness_quality/figures/false_accept_false_reject.png", "Correctness audit"),
        ("12_ablations/figures/ablation_summary.png", "Ablation summary"),
    ]
    for rel, title in placeholders:
        path = args.root / rel
        if not path.exists():
            save_bar(path, ["unavailable"], [0.0], title, "value")
    print(json.dumps({"status": "figures_updated"}, indent=2))


if __name__ == "__main__":
    main()

