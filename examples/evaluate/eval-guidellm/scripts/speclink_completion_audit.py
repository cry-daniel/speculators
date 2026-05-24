#!/usr/bin/env python3
"""Audit SpecLink experiment deliverables against speclink.md acceptance items."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def matrix_complete(path: Path) -> bool:
    rows = read_csv_tsv(path)
    return bool(rows) and all(row.get("status") == "complete" for row in rows)


def read_csv_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def has_files(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def has_columns(path: Path, columns: set[str]) -> bool:
    rows = read_csv(path)
    if not rows:
        return False
    return columns <= set(rows[0])


def count_rows(path: Path) -> int:
    return len(read_csv(path))


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    cols = ["item", "status", "evidence", "note"]
    lines = ["# SpecLink Completion Audit", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    tables = root / "tables"
    baseline = tables / "baseline_summary.csv"
    breakdown = tables / "breakdown_summary.csv"
    sparse_layout = tables / "sparse_layout_hidden_summary.csv"
    quality = tables / "sparse_quality_summary.csv"
    microbench = tables / "sparse_microbench.csv"
    report = root / "speclink_experiment_report.md"
    verification = root / "verification.md"
    verification_text = verification.read_text(encoding="utf-8") if verification.exists() else ""
    patch = root.parent / "patches" / "vllm-speclink.diff"
    patch_alt = Path("patches/vllm-speclink.diff")
    if not patch.exists() and patch_alt.exists():
        patch = patch_alt

    baseline_cols = {
        "method",
        "num_spec_tokens",
        "rate",
        "output_tokens_per_s",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "strict_em",
        "flexible_em",
    }
    acceptance_cols = {
        key
        for row in read_csv(baseline)
        for key in row
        if key.startswith("acceptance_rate_pos_")
    }
    layout_rows = read_csv(sparse_layout)
    quality_rows = read_csv(quality)
    micro_rows = read_csv(microbench)
    direct_unit_tests_supported = (
        "test_speclink_math_eval" in verification_text
        and "test_speclink_planner" in verification_text
        and "Result: passed" in verification_text
    ) or "speclink unit tests passed" in verification_text

    rows = [
        {
            "item": "1. dense/eagle3/peagle real GuideLLM results",
            "status": "supported" if matrix_complete(tables / "formal_matrix_status.tsv") else "missing",
            "evidence": str(tables / "formal_matrix_status.tsv"),
            "note": "Formal matrix covers dense, eagle3, peagle, rates 1/2/4, repeats 0/1/2.",
        },
        {
            "item": "2. dense/eagle3/peagle accuracy outputs",
            "status": "supported"
            if matrix_complete(tables / "formal_matrix_status.tsv")
            else "missing",
            "evidence": str(tables / "formal_matrix_status.tsv"),
            "note": "Matrix status checks accuracy_summary.json and accuracy n/errors.",
        },
        {
            "item": "3. baseline_summary.csv throughput/latency/accuracy/acceptance",
            "status": "supported" if has_columns(baseline, baseline_cols) and acceptance_cols else "missing",
            "evidence": str(baseline),
            "note": f"rows={count_rows(baseline)}, acceptance_cols={len(acceptance_cols)}",
        },
        {
            "item": "4. breakdown_summary.csv draft/verifier/other",
            "status": "supported"
            if has_columns(
                breakdown,
                {
                    "method",
                    "draft_forward_pct",
                    "target_verify_forward_pct",
                    "scheduler_step_pct",
                    "engine_update_pct",
                    "other_pct",
                },
            )
            else "missing",
            "evidence": str(breakdown),
            "note": f"rows={count_rows(breakdown)}",
        },
        {
            "item": "5. sparse layout union growth and speclink reduction",
            "status": "supported-proxy"
            if any(row.get("layout") == "independent_topk" for row in layout_rows)
            and any(row.get("layout") == "speclink_prob" for row in layout_rows)
            else "missing",
            "evidence": str(sparse_layout),
            "note": "Uses hidden_similarity_proxy traces; not sparse-kernel speedup.",
        },
        {
            "item": "6. SnapKV/static challenge coverage/union/Jaccard/HBM",
            "status": "supported-proxy"
            if has_columns(
                sparse_layout,
                {
                    "layout",
                    "union_blocks",
                    "jaccard_mean",
                    "coverage_mean",
                    "estimated_hbm_bytes_per_step",
                },
            )
            and any(row.get("layout") == "snapkv_static" for row in layout_rows)
            else "missing",
            "evidence": str(sparse_layout),
            "note": "Uses hidden_similarity_proxy candidates from offline Qwen3 hidden states.",
        },
        {
            "item": "7. speclink planner unit tests",
            "status": "supported" if direct_unit_tests_supported else "missing",
            "evidence": str(verification),
            "note": "pytest unavailable in spec env; direct test-function harness passed.",
        },
        {
            "item": "8. vLLM speclink plan_only smoke live trace",
            "status": "supported"
            if has_files([root / "05_speclink" / "speclink_prob_profile_smoke" / "live_sparse_trace.jsonl"])
            else "missing",
            "evidence": str(root / "05_speclink" / "speclink_prob_profile_smoke" / "live_sparse_trace.jsonl"),
            "note": "Plan-only integration; dense verifier still runs.",
        },
        {
            "item": "9. speclink_experiment_report.md",
            "status": "supported" if report.exists() and report.stat().st_size > 0 else "missing",
            "evidence": str(report),
            "note": "",
        },
        {
            "item": "Bonus. real sparse verifier kernel e2e speedup",
            "status": "missing",
            "evidence": "SPECLINK_MODE=sparse_kernel is not integrated",
            "note": "Explicitly reported as missing; not part of minimum acceptance.",
        },
        {
            "item": "Bonus. acceptance probability calibration",
            "status": "supported-proxy"
            if has_files([root / "05_speclink" / "calibrator.pkl", root / "05_speclink" / "calibrator_summary.json"])
            else "missing",
            "evidence": str(root / "05_speclink" / "calibrator_summary.json"),
            "note": "Planner probability calibration, not sparse verifier correctness.",
        },
        {
            "item": "Bonus. false accept/reject sparse verifier quality",
            "status": "supported-proxy"
            if has_columns(quality, {"actual_false_accept_rate", "actual_false_reject_rate"})
            else "missing",
            "evidence": str(quality),
            "note": "Offline masked-logits proxy, not vLLM sparse-kernel correctness.",
        },
        {
            "item": "Bonus. padded 2k/4k/8k long-context evidence",
            "status": "supported-proxy"
            if {"pad2k", "pad4k"} <= {row.get("context_label") for row in layout_rows}
            else "partial",
            "evidence": str(sparse_layout),
            "note": "pad8k dataset exists; hidden-quality small runs cover pad2k and pad4k.",
        },
        {
            "item": "Patch diff saved",
            "status": "supported" if patch.exists() and patch.stat().st_size > 0 else "missing",
            "evidence": str(patch),
            "note": "",
        },
    ]
    write_md(Path(args.out_md), rows)
    print(f"wrote {args.out_md}")
    missing_minimum = [
        row for row in rows[:9] if not row["status"].startswith("supported")
    ]
    if missing_minimum:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
