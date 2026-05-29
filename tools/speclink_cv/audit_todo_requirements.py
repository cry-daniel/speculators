#!/usr/bin/env python3
"""Audit current SpecLink-CV artifacts against TODO.md requirements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_ROOT = (
    REPO_ROOT
    / "examples/evaluate/eval-guidellm/results"
    / "speclink_cv_correctness_audit_20260526_230036"
)
EVAL_GUIDELLM_ROOT = REPO_ROOT / "examples/evaluate/eval-guidellm"
TEMP_ROOT = EVAL_GUIDELLM_ROOT / "temp"


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_exists(root: Path, relative: str) -> bool:
    return (root / relative).exists()


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern))


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def csv_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def current_gate_summary(root: Path) -> dict[str, Any]:
    rows = read_csv(root / "09_reports/token_id_correctness_summary.csv")
    chunked = [
        row
        for row in rows
        if str(row.get("method", "")).startswith("speclink_cv_chunked")
    ]
    exactsafe = [row for row in rows if row.get("method") == "speclink_cv_exactsafe"]
    chunked_failed = [
        row
        for row in chunked
        if safe_float(row.get("exact_match_vs_eagle3")) != 1.0
    ]
    exactsafe_failed = [
        row
        for row in exactsafe
        if safe_float(row.get("exact_match_vs_eagle3")) != 1.0
    ]
    return {
        "token_rows": len(rows),
        "chunked_rows": len(chunked),
        "chunked_failed": len(chunked_failed),
        "exactsafe_rows": len(exactsafe),
        "exactsafe_failed": len(exactsafe_failed),
    }


def batch_invariant_summary(root: Path) -> dict[str, Any]:
    bs8_rows = read_csv(root / "54_batch_invariant_chunked_matrix/summary.csv")
    extension_rows = read_csv(root / "58_batch_invariant_bs16_bs32_matrix/summary.csv")
    rows = bs8_rows + extension_rows
    failed = [row for row in rows if not csv_bool(row.get("matched"))]
    bs8_failed = [row for row in bs8_rows if not csv_bool(row.get("matched"))]
    extension_failed = [
        row for row in extension_rows if not csv_bool(row.get("matched"))
    ]
    labels = [
        f"{row.get('model')}/{row.get('dataset')}/K={row.get('K')}"
        f"/bs={row.get('batch_size')}"
        for row in failed
    ]
    return {
        "rows": len(rows),
        "failed": len(failed),
        "passed": len(rows) - len(failed),
        "all_passed": bool(rows) and not failed,
        "bs8_rows": len(bs8_rows),
        "bs8_failed": len(bs8_failed),
        "bs8_passed": len(bs8_rows) - len(bs8_failed),
        "bs8_all_passed": bool(bs8_rows) and not bs8_failed,
        "extension_rows": len(extension_rows),
        "extension_failed": len(extension_failed),
        "extension_passed": len(extension_rows) - len(extension_failed),
        "failure_labels": ", ".join(labels),
    }


def grouped_fallback_summary(root: Path) -> dict[str, Any]:
    bs8_rows = read_csv(root / "52_grouped_k8_k12_cross_model_data_bs8/summary.csv")
    extension_rows = read_csv(
        root / "59_grouped_batchwide_fallback_bs16_bs32_matrix/summary.csv"
    )
    rows = bs8_rows + extension_rows
    failed = [row for row in rows if not csv_bool(row.get("matched"))]
    extension_failed = [
        row for row in extension_rows if not csv_bool(row.get("matched"))
    ]
    labels = [
        f"{row.get('model')}/{row.get('dataset')}/K={row.get('K')}"
        f"/bs={row.get('batch_size')}"
        for row in failed
    ]
    extension_labels = [
        f"{row.get('model')}/{row.get('dataset')}/K={row.get('K')}"
        f"/bs={row.get('batch_size')}"
        for row in extension_failed
    ]
    return {
        "rows": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "extension_rows": len(extension_rows),
        "extension_passed": len(extension_rows) - len(extension_failed),
        "extension_failed": len(extension_failed),
        "failure_labels": ", ".join(labels),
        "extension_failure_labels": ", ".join(extension_labels),
    }


def long_generation_summary(root: Path) -> dict[str, Any]:
    chunked_rows = read_csv(root / "60_qwen_mtbench_k8_bs8_t64_batchinv_gate/summary.csv")
    exactsafe_rows = read_csv(root / "61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/summary.csv")
    grouped_rows = read_csv(root / "63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/summary.csv")
    debug_rows = read_csv(root / "64_qwen_mtbench_k8_bs8_t64_debug_plain/summary.csv")
    reject_confirm_rows = read_csv(
        root / "65_qwen_mtbench_k8_bs8_t64_reject_confirm_gate/summary.csv"
    )
    confirm_all_rows = read_csv(
        root / "66_qwen_mtbench_k8_bs8_t64_confirm_all_gate/summary.csv"
    )
    confirm_all_barrier_rows = read_csv(
        root / "67_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_gate/summary.csv"
    )
    prefixreject_dense_rows = read_csv(
        root / "68_qwen_mtbench_k8_bs8_t64_prefixreject_dense8_gate/summary.csv"
    )
    grouped_confirm_all_barrier_rows = read_csv(
        root
        / "69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate/summary.csv"
    )
    context_debug_rows = read_csv(
        root / "70_qwen_mtbench_k8_bs8_t64_context_debug_gate/summary.csv"
    )
    recompute_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_recompute_qwen_mtbench_k8_bs8_t64_20260527/summary.csv"
    )
    recompute_barrier_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/summary.csv"
    )
    batched_dense_worker_ordered_rows = read_csv(
        TEMP_ROOT
        / (
            "speclink_cv_batched_dense_worker_ordered_recompute_barrier_"
            "qwen_mtbench_k8_bs8_t64_20260527/summary.csv"
        )
    )
    prefix_nokv_rows = read_csv(
        TEMP_ROOT
        / (
            "speclink_cv_prefix_nokv_recompute_barrier_"
            "qwen_mtbench_k8_bs8_t64_20260527/summary.csv"
        )
    )
    chunked = chunked_rows[0] if chunked_rows else {}
    exactsafe = exactsafe_rows[0] if exactsafe_rows else {}
    grouped = grouped_rows[0] if grouped_rows else {}
    debug = debug_rows[0] if debug_rows else {}
    reject_confirm = reject_confirm_rows[0] if reject_confirm_rows else {}
    confirm_all = confirm_all_rows[0] if confirm_all_rows else {}
    confirm_all_barrier = (
        confirm_all_barrier_rows[0] if confirm_all_barrier_rows else {}
    )
    prefixreject_dense = (
        prefixreject_dense_rows[0] if prefixreject_dense_rows else {}
    )
    grouped_confirm_all_barrier = (
        grouped_confirm_all_barrier_rows[0]
        if grouped_confirm_all_barrier_rows
        else {}
    )
    context_debug = context_debug_rows[0] if context_debug_rows else {}
    recompute = recompute_rows[0] if recompute_rows else {}
    recompute_barrier = (
        recompute_barrier_rows[0] if recompute_barrier_rows else {}
    )
    batched_dense_worker_ordered = (
        batched_dense_worker_ordered_rows[0]
        if batched_dense_worker_ordered_rows
        else {}
    )
    prefix_nokv = prefix_nokv_rows[0] if prefix_nokv_rows else {}
    return {
        "chunked_rows": len(chunked_rows),
        "chunked_matched": csv_bool(chunked.get("matched")),
        "chunked_matched_count": chunked.get("matched_count", ""),
        "chunked_total_count": chunked.get("total_count", ""),
        "chunked_first_mismatch_token": chunked.get("first_mismatch_token_index", ""),
        "exactsafe_rows": len(exactsafe_rows),
        "exactsafe_matched": csv_bool(exactsafe.get("matched")),
        "exactsafe_matched_count": exactsafe.get("matched_count", ""),
        "exactsafe_total_count": exactsafe.get("total_count", ""),
        "grouped_rows": len(grouped_rows),
        "grouped_matched": csv_bool(grouped.get("matched")),
        "grouped_matched_count": grouped.get("matched_count", ""),
        "grouped_total_count": grouped.get("total_count", ""),
        "grouped_first_mismatch_token": grouped.get(
            "first_mismatch_token_index", ""
        ),
        "debug_rows": len(debug_rows),
        "debug_matched": csv_bool(debug.get("matched")),
        "debug_matched_count": debug.get("matched_count", ""),
        "debug_total_count": debug.get("total_count", ""),
        "debug_first_mismatch_token": debug.get("first_mismatch_token_index", ""),
        "reject_confirm_rows": len(reject_confirm_rows),
        "reject_confirm_matched": csv_bool(reject_confirm.get("matched")),
        "reject_confirm_matched_count": reject_confirm.get("matched_count", ""),
        "reject_confirm_total_count": reject_confirm.get("total_count", ""),
        "reject_confirm_first_mismatch_token": reject_confirm.get(
            "first_mismatch_token_index", ""
        ),
        "confirm_all_rows": len(confirm_all_rows),
        "confirm_all_matched": csv_bool(confirm_all.get("matched")),
        "confirm_all_matched_count": confirm_all.get("matched_count", ""),
        "confirm_all_total_count": confirm_all.get("total_count", ""),
        "confirm_all_first_mismatch_token": confirm_all.get(
            "first_mismatch_token_index", ""
        ),
        "confirm_all_barrier_rows": len(confirm_all_barrier_rows),
        "confirm_all_barrier_matched": csv_bool(
            confirm_all_barrier.get("matched")
        ),
        "confirm_all_barrier_matched_count": confirm_all_barrier.get(
            "matched_count", ""
        ),
        "confirm_all_barrier_total_count": confirm_all_barrier.get(
            "total_count", ""
        ),
        "confirm_all_barrier_first_mismatch_token": confirm_all_barrier.get(
            "first_mismatch_token_index", ""
        ),
        "prefixreject_dense_rows": len(prefixreject_dense_rows),
        "prefixreject_dense_matched": csv_bool(prefixreject_dense.get("matched")),
        "prefixreject_dense_matched_count": prefixreject_dense.get(
            "matched_count", ""
        ),
        "prefixreject_dense_total_count": prefixreject_dense.get("total_count", ""),
        "prefixreject_dense_first_mismatch_token": prefixreject_dense.get(
            "first_mismatch_token_index", ""
        ),
        "grouped_confirm_all_barrier_rows": len(grouped_confirm_all_barrier_rows),
        "grouped_confirm_all_barrier_matched": csv_bool(
            grouped_confirm_all_barrier.get("matched")
        ),
        "grouped_confirm_all_barrier_matched_count": grouped_confirm_all_barrier.get(
            "matched_count", ""
        ),
        "grouped_confirm_all_barrier_total_count": grouped_confirm_all_barrier.get(
            "total_count", ""
        ),
        "grouped_confirm_all_barrier_first_mismatch_token": grouped_confirm_all_barrier.get(
            "first_mismatch_token_index", ""
        ),
        "context_debug_rows": len(context_debug_rows),
        "context_debug_matched": csv_bool(context_debug.get("matched")),
        "context_debug_matched_count": context_debug.get("matched_count", ""),
        "context_debug_total_count": context_debug.get("total_count", ""),
        "context_debug_first_mismatch_token": context_debug.get(
            "first_mismatch_token_index", ""
        ),
        "recompute_rows": len(recompute_rows),
        "recompute_matched": csv_bool(recompute.get("matched")),
        "recompute_matched_count": recompute.get("matched_count", ""),
        "recompute_total_count": recompute.get("total_count", ""),
        "recompute_first_mismatch_token": recompute.get(
            "first_mismatch_token_index", ""
        ),
        "recompute_barrier_rows": len(recompute_barrier_rows),
        "recompute_barrier_matched": csv_bool(recompute_barrier.get("matched")),
        "recompute_barrier_matched_count": recompute_barrier.get(
            "matched_count", ""
        ),
        "recompute_barrier_total_count": recompute_barrier.get("total_count", ""),
        "recompute_barrier_first_mismatch_token": recompute_barrier.get(
            "first_mismatch_token_index", ""
        ),
        "batched_dense_worker_ordered_rows": len(
            batched_dense_worker_ordered_rows
        ),
        "batched_dense_worker_ordered_matched": csv_bool(
            batched_dense_worker_ordered.get("matched")
        ),
        "batched_dense_worker_ordered_matched_count": (
            batched_dense_worker_ordered.get("matched_count", "")
        ),
        "batched_dense_worker_ordered_total_count": (
            batched_dense_worker_ordered.get("total_count", "")
        ),
        "batched_dense_worker_ordered_first_mismatch_token": (
            batched_dense_worker_ordered.get("first_mismatch_token_index", "")
        ),
        "prefix_nokv_rows": len(prefix_nokv_rows),
        "prefix_nokv_matched": csv_bool(prefix_nokv.get("matched")),
        "prefix_nokv_matched_count": prefix_nokv.get("matched_count", ""),
        "prefix_nokv_total_count": prefix_nokv.get("total_count", ""),
        "prefix_nokv_first_mismatch_token": prefix_nokv.get(
            "first_mismatch_token_index", ""
        ),
    }


def latest_tie_fix_summary() -> dict[str, Any]:
    t12_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t12_tiefix_20260528/summary.csv"
    )
    t32_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_tiefix_20260528/summary.csv"
    )
    row0_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_row0_debug_20260528/summary.csv"
    )
    nobarrier_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_nobarrier_tiefix_20260528/summary.csv"
    )
    exactsafe_rows = read_csv(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_exactsafe_tiefix_20260528/summary.csv"
    )
    history_kv = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_historykv_20260528"
        / "000_qwen3_8b_mtbench_k8_bs16_chunked_confirm_all_barrier"
        / "kv_cache_drift.json"
    )
    t12 = t12_rows[0] if t12_rows else {}
    t32 = t32_rows[0] if t32_rows else {}
    row0 = row0_rows[0] if row0_rows else {}
    nobarrier = nobarrier_rows[0] if nobarrier_rows else {}
    exactsafe = exactsafe_rows[0] if exactsafe_rows else {}
    history_summary = history_kv.get("summary") or {}
    return {
        "t12_rows": len(t12_rows),
        "t12_matched": csv_bool(t12.get("matched")),
        "t12_matched_count": t12.get("matched_count", ""),
        "t12_total_count": t12.get("total_count", ""),
        "t32_rows": len(t32_rows),
        "t32_matched": csv_bool(t32.get("matched")),
        "t32_matched_count": t32.get("matched_count", ""),
        "t32_total_count": t32.get("total_count", ""),
        "t32_first_mismatch_token": t32.get("first_mismatch_token_index", ""),
        "row0_rows": len(row0_rows),
        "row0_matched": csv_bool(row0.get("matched")),
        "row0_matched_count": row0.get("matched_count", ""),
        "row0_total_count": row0.get("total_count", ""),
        "row0_first_mismatch_token": row0.get("first_mismatch_token_index", ""),
        "nobarrier_rows": len(nobarrier_rows),
        "nobarrier_matched": csv_bool(nobarrier.get("matched")),
        "nobarrier_matched_count": nobarrier.get("matched_count", ""),
        "nobarrier_total_count": nobarrier.get("total_count", ""),
        "nobarrier_first_mismatch_token": nobarrier.get(
            "first_mismatch_token_index", ""
        ),
        "exactsafe_rows": len(exactsafe_rows),
        "exactsafe_matched": csv_bool(exactsafe.get("matched")),
        "exactsafe_matched_count": exactsafe.get("matched_count", ""),
        "exactsafe_total_count": exactsafe.get("total_count", ""),
        "history_kv_rows": 1 if history_summary else 0,
        "history_kv_checksum_field": history_summary.get("checksum_field", ""),
        "history_kv_baseline_upto": history_summary.get(
            "baseline_upto_position", ""
        ),
        "history_kv_cv_upto": history_summary.get("cv_upto_position", ""),
        "history_kv_compared_entries": history_summary.get(
            "compared_entries", ""
        ),
        "history_kv_checksum_mismatches": history_summary.get(
            "checksum_mismatches", ""
        ),
    }


def correctness_case_summary(relative_json: str) -> dict[str, Any]:
    payload = read_json(TEMP_ROOT / relative_json)
    matched_items = payload.get("matched_items") or []
    mismatches = payload.get("mismatches") or []
    first = mismatches[0] if mismatches else {}
    return {
        "exists": bool(payload),
        "matched": csv_bool(payload.get("matched")),
        "matched_count": sum(1 for item in matched_items if item is True),
        "total_count": len(matched_items),
        "first_mismatch_token": first.get("first_diff_index", ""),
        "baseline_token": (
            first.get("baseline_token_ids", [""])[first.get("first_diff_index", -1)]
            if isinstance(first.get("first_diff_index"), int)
            and first.get("baseline_token_ids")
            and first.get("first_diff_index", -1) >= 0
            and first.get("first_diff_index", -1) < len(first.get("baseline_token_ids"))
            else ""
        ),
        "cv_token": (
            first.get("speclink_cv_token_ids", [""])[
                first.get("first_diff_index", -1)
            ]
            if isinstance(first.get("first_diff_index"), int)
            and first.get("speclink_cv_token_ids")
            and first.get("first_diff_index", -1) >= 0
            and first.get("first_diff_index", -1)
            < len(first.get("speclink_cv_token_ids"))
            else ""
        ),
    }


def profile_event_summary(relative_jsonl: str, event_name: str) -> dict[str, int]:
    path = TEMP_ROOT / relative_jsonl
    count = 0
    released = 0
    if not path.exists():
        return {"count": 0, "released": 0}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != event_name:
                continue
            count += 1
            released += sum(int(x) for x in payload.get("released_block_counts", []))
    return {"count": count, "released": released}


def representative_probe_summary() -> dict[str, Any]:
    bs1 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs1_t32_representative_20260528/"
        "000_qwen3_8b_mtbench_k8_bs1_h4/correctness.json"
    )
    bs4 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs4_t32_representative_20260528/"
        "000_qwen3_8b_mtbench_k8_bs4_h4/correctness.json"
    )
    bs8 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_representative_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_h4/correctness.json"
    )
    dense0 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_dense0_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_h4_dense0/correctness.json"
    )
    batchedprefix_dense0 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_batchedprefix_dense0_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_h4_batchedprefix_dense0/correctness.json"
    )
    skiprollback_dense0 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_skiprollback_dense0_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_h4_skiprollback_dense0/correctness.json"
    )
    rejectconfirm_dense0 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_dense0_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_h4_rejectconfirm_dense0/correctness.json"
    )
    exactsafe_afterfix = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs16_t32_exactsafe_afterfix_20260528/"
        "000_qwen3_8b_mtbench_k8_bs16_exactsafe_afterfix/correctness.json"
    )
    kvdebug = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_kvdebug_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm/correctness.json"
    )
    kvdebug_kv = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_kvdebug_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm"
        / "kv_cache_drift.json"
    )
    kvdebug_active = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_kvdebug_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm"
        / "active_batch_drift.json"
    )
    confirm_fullactive = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_confirm_fullactive_afterprogress_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm/correctness.json"
    )
    confirm_fullactive_active = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_confirm_fullactive_afterprogress_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm"
        / "active_batch_drift.json"
    )
    confirmall_barrier_h4 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_confirmall_barrier_h4_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm_all_barrier/correctness.json"
    )
    confirmall_barrier_h4_active = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_confirmall_barrier_h4_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm_all_barrier"
        / "active_batch_drift.json"
    )
    lockstep_h4 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_h4_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/correctness.json"
    )
    lockstep_h4_active = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_h4_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked"
        / "active_batch_drift.json"
    )
    lockstep_batchsuffix = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_batchsuffix_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/correctness.json"
    )
    lockstep_prefixdense8 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_prefixdense8_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/correctness.json"
    )
    lockstep_rejectconfirm = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_lockstep_rejectconfirm_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm/correctness.json"
    )
    proposer_drift = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_proposer_debug_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked"
        / "proposer_drift.json"
    )
    recompute_blockrollback = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_recompute_blockrollback_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked_recompute/correctness.json"
    )
    drafteps001 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_drafteps001_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/correctness.json"
    )
    grouped_drafteps001 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_grouped_drafteps001_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked_grouped_batchwide_prefixreject_barrier/"
        "correctness.json"
    )
    qwen_math_grouped_drafteps001 = correctness_case_summary(
        "speclink_cv_qwen_math_k8_bs8_t32_grouped_drafteps001_20260528/"
        "000_qwen3_8b_math_k8_bs8_chunked_grouped_batchwide_prefixreject_barrier/"
        "correctness.json"
    )
    llama_grouped_drafteps001 = correctness_case_summary(
        "speclink_cv_llama_math_k8_bs8_t32_grouped_drafteps001_20260528/"
        "000_llama3_1_8b_math_k8_bs8_chunked_grouped_batchwide_prefixreject_barrier/"
        "correctness.json"
    )
    llama_mtbench_grouped_drafteps001 = correctness_case_summary(
        "speclink_cv_llama_mtbench_k8_bs8_t32_grouped_drafteps001_20260528/"
        "000_llama3_1_8b_mtbench_k8_bs8_chunked_grouped_batchwide_prefixreject_barrier/"
        "correctness.json"
    )
    dlmrollback = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_dlmrollback_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/correctness.json"
    )
    dlmrollback_dense0 = correctness_case_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_dlmrollback_dense0_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/correctness.json"
    )
    dlmrollback_dense0_drift = read_json(
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_dlmrollback_dense0_debug_20260528"
        / "000_qwen3_8b_mtbench_k8_bs8_chunked"
        / "proposer_drift.json"
    )
    rollback = profile_event_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_skiprollback_dense0_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_h4_skiprollback_dense0/"
        "speclink_cv_profile.jsonl",
        "prefix_skip_suffix_block_rollback",
    )
    dlmrollback_adjust = profile_event_summary(
        "speclink_cv_qwen_mtbench_k8_bs8_t32_dlmrollback_debug_20260528/"
        "000_qwen3_8b_mtbench_k8_bs8_chunked/"
        "speclink_cv_events.jsonl",
        "prefix_reject_drafter_rollback_adjusted",
    )
    return {
        "bs1": bs1,
        "bs4": bs4,
        "bs8": bs8,
        "dense0": dense0,
        "batchedprefix_dense0": batchedprefix_dense0,
        "skiprollback_dense0": skiprollback_dense0,
        "rejectconfirm_dense0": rejectconfirm_dense0,
        "exactsafe_afterfix": exactsafe_afterfix,
        "kvdebug": kvdebug,
        "kvdebug_checksum_entries": (
            (kvdebug_kv.get("summary") or {}).get("compared_entries", "")
        ),
        "kvdebug_checksum_mismatches": (
            (kvdebug_kv.get("summary") or {}).get("checksum_mismatches", "")
        ),
        "kvdebug_active_equal": (
            (kvdebug_active.get("summary") or {}).get(
                "active_request_ordinals_equal", ""
            )
        ),
        "kvdebug_missing_active": (
            (kvdebug_active.get("summary") or {}).get("missing_active_in_cv", "")
        ),
        "confirm_fullactive": confirm_fullactive,
        "confirm_fullactive_missing_active": (
            (confirm_fullactive_active.get("summary") or {}).get(
                "missing_active_in_cv", ""
            )
        ),
        "confirm_fullactive_slot_equal": (
            (confirm_fullactive_active.get("summary") or {}).get(
                "target_slot_mapping_gid0_equal", ""
            )
        ),
        "confirm_fullactive_scheduled_equal": (
            (confirm_fullactive_active.get("summary") or {}).get(
                "scheduled_drafts_equal", ""
            )
        ),
        "confirmall_barrier_h4": confirmall_barrier_h4,
        "confirmall_barrier_h4_missing_active": (
            (confirmall_barrier_h4_active.get("summary") or {}).get(
                "missing_active_in_cv", ""
            )
        ),
        "confirmall_barrier_h4_slot_equal": (
            (confirmall_barrier_h4_active.get("summary") or {}).get(
                "target_slot_mapping_gid0_equal", ""
            )
        ),
        "confirmall_barrier_h4_scheduled_equal": (
            (confirmall_barrier_h4_active.get("summary") or {}).get(
                "scheduled_drafts_equal", ""
            )
        ),
        "lockstep_h4": lockstep_h4,
        "lockstep_h4_active_equal": (
            (lockstep_h4_active.get("summary") or {}).get(
                "active_request_ordinals_equal", ""
            )
        ),
        "lockstep_h4_missing_active": (
            (lockstep_h4_active.get("summary") or {}).get(
                "missing_active_in_cv", ""
            )
        ),
        "lockstep_h4_scheduled_equal": (
            (lockstep_h4_active.get("summary") or {}).get(
                "scheduled_drafts_equal", ""
            )
        ),
        "lockstep_h4_slot_equal": (
            (lockstep_h4_active.get("summary") or {}).get(
                "target_slot_mapping_gid0_equal", ""
            )
        ),
        "lockstep_batchsuffix": lockstep_batchsuffix,
        "lockstep_prefixdense8": lockstep_prefixdense8,
        "lockstep_rejectconfirm": lockstep_rejectconfirm,
        "proposer_drift_reason": proposer_drift.get(
            "first_drift_reason", ""
        ),
        "proposer_drift_pairs": proposer_drift.get("compared_pairs", ""),
        "proposer_drift_first": proposer_drift.get("first_drift") or {},
        "recompute_blockrollback": recompute_blockrollback,
        "drafteps001": drafteps001,
        "grouped_drafteps001": grouped_drafteps001,
        "qwen_math_grouped_drafteps001": qwen_math_grouped_drafteps001,
        "llama_grouped_drafteps001": llama_grouped_drafteps001,
        "llama_mtbench_grouped_drafteps001": llama_mtbench_grouped_drafteps001,
        "dlmrollback": dlmrollback,
        "dlmrollback_dense0": dlmrollback_dense0,
        "dlmrollback_dense0_drift_reason": dlmrollback_dense0_drift.get(
            "first_drift_reason", ""
        ),
        "dlmrollback_dense0_drift_first": (
            dlmrollback_dense0_drift.get("first_drift") or {}
        ),
        "dlmrollback_adjust_count": dlmrollback_adjust["count"],
        "rollback_count": rollback["count"],
        "rollback_released": rollback["released"],
    }


def guidellm_summary(root: Path) -> dict[str, Any]:
    rows = read_csv(root / "09_reports/summary_metrics.csv")
    guidellm = [
        row for row in rows
        if row.get("measurement_type") == "guidellm_end_to_end"
    ]
    cv_rows = [row for row in guidellm if str(row.get("method", "")).startswith("cv_")]
    valid = [
        row
        for row in cv_rows
        if row.get("speedup_claim_status") == "valid_exact_chunked"
    ]
    quality_valid = [
        row
        for row in cv_rows
        if row.get("speedup_claim_status") == "valid_quality_preserving_chunked"
    ]
    math_quality = [
        row for row in guidellm
        if row.get("source") == "math_quality_followup"
    ]
    contribution = [
        row for row in rows
        if row.get("measurement_type") == "contribution_ablation"
    ]
    steady = [
        row for row in rows
        if row.get("measurement_type") == "steady_state_throughput"
    ]
    return {
        "guidellm_rows": len(guidellm),
        "cv_rows": len(cv_rows),
        "valid_cv_rows": len(valid),
        "quality_valid_cv_rows": len(quality_valid),
        "math_quality_rows": len(math_quality),
        "contribution_rows": len(contribution),
        "steady_rows": len(steady),
    }


def unit_summary(root: Path) -> dict[str, Any]:
    rows = read_csv(root / "02_unit_tests/unit_test_summary.csv")
    return {
        "unit_rows": len(rows),
        "unit_failed": sum(1 for row in rows if row.get("status") != "pass"),
    }


def build_audit(root: Path) -> list[dict[str, Any]]:
    gate = current_gate_summary(root)
    batch_invariant = batch_invariant_summary(root)
    grouped_fallback = grouped_fallback_summary(root)
    long_generation = long_generation_summary(root)
    tie_fix = latest_tie_fix_summary()
    representative = representative_probe_summary()
    proposer_first = representative.get("proposer_drift_first") or {}
    proposer_hidden_delta = proposer_first.get("hidden_delta") or {}
    dlmrollback_dense0_first = (
        representative.get("dlmrollback_dense0_drift_first") or {}
    )
    dlmrollback_dense0_hidden_delta = (
        dlmrollback_dense0_first.get("hidden_delta") or {}
    )
    guide = guidellm_summary(root)
    unit = unit_summary(root)
    grouped_probe = read_json(
        root / "51_grouped_batchwide_prefixreject_probe/correctness.json"
    )
    grouped_matrix = read_csv(
        root / "52_grouped_k8_k12_cross_model_data_bs8/summary.csv"
    )
    grouped_matrix_failed = [
        row for row in grouped_matrix if not csv_bool(row.get("matched"))
    ]
    worker_diag = read_csv(
        root / "53_worker_ordered_lowmargin_llama_math_k8/summary.csv"
    )
    worker_pass = next(
        (
            row
            for row in worker_diag
            if row.get("case") == "confidence_minbenefit_oneshot_pass"
        ),
        {},
    )
    if grouped_matrix:
        failure_labels = ", ".join(
            f"{row.get('model')}/{row.get('dataset')}/K={row.get('K')}"
            for row in grouped_matrix_failed
        )
        grouped_probe_note = (
            "Grouped batch-wide prefix-reject matrix passed "
            f"{len(grouped_matrix) - len(grouped_matrix_failed)}/"
            f"{len(grouped_matrix)} rows"
            + (f"; failures: {failure_labels}." if failure_labels else ".")
        )
        if grouped_fallback["extension_rows"]:
            grouped_probe_note += (
                " The bs=16/32 grouped fallback extension passed "
                f"{grouped_fallback['extension_passed']}/"
                f"{grouped_fallback['extension_rows']} rows"
                + (
                    "."
                    if not grouped_fallback["extension_failure_labels"]
                    else (
                        "; failures: "
                        f"{grouped_fallback['extension_failure_labels']}."
                    )
                )
            )
    elif grouped_probe.get("matched"):
        grouped_probe_note = (
            "Latest grouped batch-wide prefix-reject probe passed 8/8 for "
            "Qwen3/math K=8 bs=8."
        )
    else:
        grouped_probe_note = "Latest grouped batch-wide prefix-reject probe is missing or failed."
    calibration_metrics = read_json(
        root / "03_confidence_calibration/eval/calibration_metrics.json"
    )
    calibration_rows = calibration_metrics.get("rows", "")
    calibration_ece = calibration_metrics.get("ece", "")
    calibration_brier = calibration_metrics.get("brier", "")
    verify_cost_rows = max(
        0,
        sum(
            1
            for _ in (
                root
                / "07_roofline_packing/verify_cost_proxy/verify_cost_lookup.csv"
            ).open(encoding="utf-8")
        )
        - 1,
    ) if file_exists(
        root, "07_roofline_packing/verify_cost_proxy/verify_cost_lookup.csv"
    ) else 0
    has_valid_hk = gate["chunked_rows"] > 0 and gate["chunked_failed"] == 0
    has_batch_invariant_bs8_hk = bool(batch_invariant["bs8_all_passed"])
    has_batch_invariant_full_hk = bool(batch_invariant["all_passed"])
    has_valid_end_to_end = guide["valid_cv_rows"] > 0
    rows = [
        {
            "area": "result_tree",
            "requirement": "TODO result tree with 00_env through 09_reports, logs, patches, scripts",
            "status": "complete"
            if all(file_exists(root, name) for name in [
                "00_env",
                "01_impl_notes",
                "02_unit_tests",
                "03_confidence_calibration",
                "04_baselines",
                "05_cv_ablation",
                "06_scheduler_queue",
                "07_roofline_packing",
                "08_figures",
                "09_reports",
                "logs",
                "patches",
                "scripts",
            ])
            else "incomplete",
            "evidence": str(root),
            "notes": "Directory scaffold exists in the audit root.",
        },
        {
            "area": "environment",
            "requirement": "env_check.py output env_report.md and env_report.json",
            "status": "complete"
            if file_exists(root, "00_env/env_report.md")
            and file_exists(root, "00_env/env_report.json")
            else "missing",
            "evidence": "00_env/env_report.md; 00_env/env_report.json",
            "notes": "",
        },
        {
            "area": "runtime_flags",
            "requirement": "SpecLink-CV enable/confidence/async/roofline/candidate/log/profile/debug flags",
            "status": "complete",
            "evidence": "vllm/vllm/speclink_cv.py; AGENTS.md",
            "notes": "Implemented as SPECLINK_CV_* environment variables rather than public vLLM CLI args.",
        },
        {
            "area": "vllm_integration",
            "requirement": "Live vLLM prefix/suffix verifier path, not simulator-only",
            "status": (
                "partial_batch_invariant_bs8_exact"
                if has_batch_invariant_bs8_hk
                else "partial_blocked_by_correctness"
            ),
            "evidence": "vllm/vllm/v1/core/sched/scheduler.py; live correctness gate artifacts",
            "notes": (
                "h<K path runs in live vLLM. Historical non-batch-invariant "
                "gates still expose verifier-shape drift, but "
                "`VLLM_BATCH_INVARIANT=1` makes the archived bs=8 plain "
                "chunked matrix pass "
                f"{batch_invariant['bs8_passed']}/{batch_invariant['bs8_rows']} "
                "rows. The bs=16/32 extension passes only "
                f"{batch_invariant['extension_passed']}/"
                f"{batch_invariant['extension_rows']} rows."
            ),
        },
        {
            "area": "confidence",
            "requirement": "DLM confidence trace and calibration tooling",
            "status": (
                "complete_trace_evaluated_blocked_by_correctness"
                if calibration_metrics
                else "partial"
            ),
            "evidence": (
                "03_confidence_calibration/trace_collection; "
                "03_confidence_calibration/fit; "
                "03_confidence_calibration/eval"
            ),
            "notes": (
                f"trace rows eval={calibration_rows}; ECE={calibration_ece}; "
                f"Brier={calibration_brier}. End-to-end valid "
                "confidence-guided speedup still needs a batch-invariant "
                "GuideLLM rerun."
            ),
        },
        {
            "area": "state_machine",
            "requirement": "Request state machine for prefix reject, prefix accept, suffix verify, commit",
            "status": "complete_unit_tested",
            "evidence": "tools/speclink_cv/core.py; tools/speclink_cv/test_state_machine.py",
            "notes": "",
        },
        {
            "area": "async_queue",
            "requirement": "Async verification queue with independent request scheduling and starvation guard",
            "status": "complete_unit_tested_live_profiled",
            "evidence": "tools/speclink_cv/core.py; scheduler.py; test_async_queue.py; speclink_cv_profile.jsonl",
            "notes": "",
        },
        {
            "area": "roofline",
            "requirement": "Roofline-aware packing or empirical proxy plus prediction/fallback logging",
            "status": "partial_trace_proxy" if verify_cost_rows else "partial",
            "evidence": (
                "07_roofline_packing/verify_cost_proxy; "
                "tools/speclink_cv/profile_verify_cost.py; "
                "test_roofline_packing.py"
            ),
            "notes": (
                f"trace-derived lookup rows={verify_cost_rows}; this is not "
                "hardware timing. Hardware-accurate roofline validation still "
                "needs a batch-invariant GuideLLM rerun."
            ),
        },
        {
            "area": "unit_tests",
            "requirement": "Focused unit tests and summaries",
            "status": "complete" if unit["unit_rows"] >= 6 and unit["unit_failed"] == 0 else "incomplete",
            "evidence": "02_unit_tests/unit_test_summary.csv",
            "notes": f"tests={unit['unit_rows']} failed={unit['unit_failed']}",
        },
        {
            "area": "correctness",
            "requirement": "Greedy SpecLink-CV output must match EAGLE3 one-shot",
            "status": (
                "failed_batch_invariant_full_matrix"
                if batch_invariant["extension_failed"] > 0
                else
                "partial_batch_invariant_smoke_passed"
                if has_batch_invariant_bs8_hk
                else "failed"
                if gate["chunked_failed"] > 0
                else "complete"
                if has_valid_hk
                else "missing"
            ),
            "evidence": "09_reports/token_id_correctness_summary.csv",
            "notes": (
                f"h<K rows={gate['chunked_rows']} failed={gate['chunked_failed']}; "
                f"exactsafe rows={gate['exactsafe_rows']} failed={gate['exactsafe_failed']}. "
                f"{grouped_probe_note} Historical non-batch-invariant live "
                "h<K chunking remains unsafe because at least one grouped-"
                "fallback row still fails. The worker-ordered diagnostic "
                "keeps that failure after row alignment, while exact-safe "
                "one-shot fallback passes 8/8. With `VLLM_BATCH_INVARIANT=1`, "
                "the bs=8 plain chunked matrix in "
                "`54_batch_invariant_chunked_matrix/` passes "
                f"{batch_invariant['bs8_passed']}/"
                f"{batch_invariant['bs8_rows']} rows, but the bs=16/32 "
                "extension in `58_batch_invariant_bs16_bs32_matrix/` passes "
                f"only {batch_invariant['extension_passed']}/"
                f"{batch_invariant['extension_rows']} rows"
                + (
                    "."
                    if not batch_invariant["failure_labels"]
                    else f"; failures: {batch_invariant['failure_labels']}."
                )
                + (
                    " The grouped batch-wide prefix-reject fallback is a "
                    "partial recovery: it passes "
                    f"{grouped_fallback['extension_passed']}/"
                    f"{grouped_fallback['extension_rows']} bs=16/32 rows"
                    + (
                        "."
                        if not grouped_fallback["extension_failure_labels"]
                        else (
                            "; remaining grouped failures: "
                            f"{grouped_fallback['extension_failure_labels']}."
                        )
                    )
                    if grouped_fallback["extension_rows"]
                    else ""
                )
                + (
                    " The longer Qwen3/MTBench/K=8/bs=8 64-token "
                    "batch-invariant gate fails "
                    f"{long_generation['chunked_matched_count']}/"
                    f"{long_generation['chunked_total_count']} with first "
                    "mismatch token "
                    f"{long_generation['chunked_first_mismatch_token']}, "
                    "grouped batch-wide fallback improves only to "
                    f"{long_generation['grouped_matched_count']}/"
                    f"{long_generation['grouped_total_count']}, "
                    "while exact-safe fallback passes "
                    f"{long_generation['exactsafe_matched_count']}/"
                    f"{long_generation['exactsafe_total_count']}."
                    + (
                        " The debug 8-prompt rerun attributes the first "
                        "mismatch to a `prefix_rejected_skip_suffix` commit "
                        "at token "
                        f"{long_generation['debug_first_mismatch_token']}; "
                        "the paired prefix-reject full-K confirmation gate "
                        "still matches only "
                        f"{long_generation['reject_confirm_matched_count']}/"
                        f"{long_generation['reject_confirm_total_count']}; "
                        "confirm-all and barrier confirm-all both still match "
                        "only "
                        f"{long_generation['confirm_all_matched_count']}/"
                        f"{long_generation['confirm_all_total_count']}; "
                        "prefix-reject dense8 realignment also matches only "
                        f"{long_generation['prefixreject_dense_matched_count']}/"
                        f"{long_generation['prefixreject_dense_total_count']}; "
                        "grouped full-K confirmation barrier also matches only "
                        f"{long_generation['grouped_confirm_all_barrier_matched_count']}/"
                        f"{long_generation['grouped_confirm_all_barrier_total_count']}; "
                        "the context-debug rerun also matches only "
                        f"{long_generation['context_debug_matched_count']}/"
                        f"{long_generation['context_debug_total_count']}."
                        if long_generation["debug_rows"]
                        else ""
                    )
                    if long_generation["chunked_rows"]
                    else ""
                )
                + (
                    " The 2026-05-28 sampler tie-breaking fix makes the "
                    "Qwen3/MTBench/K=8/bs=16/max12 gate pass "
                    f"{tie_fix['t12_matched_count']}/"
                    f"{tie_fix['t12_total_count']}, but the max32 rerun "
                    "still passes only "
                    f"{tie_fix['t32_matched_count']}/"
                    f"{tie_fix['t32_total_count']} with first mismatch token "
                    f"{tie_fix['t32_first_mismatch_token']}. The no-barrier "
                    "max32 rerun also fails "
                    f"{tie_fix['nobarrier_matched_count']}/"
                    f"{tie_fix['nobarrier_total_count']}, while exact-safe "
                    "fallback passes "
                    f"{tie_fix['exactsafe_matched_count']}/"
                    f"{tie_fix['exactsafe_total_count']}."
                    + (
                        " The bounded history-KV rerun compares "
                        f"{tie_fix['history_kv_compared_entries']} "
                        "layer-0 entries before verifier position "
                        f"{tie_fix['history_kv_baseline_upto']} and finds "
                        f"{tie_fix['history_kv_checksum_mismatches']} "
                        "checksum mismatches."
                        if tie_fix["history_kv_rows"]
                        else ""
                    )
                    if tie_fix["t32_rows"]
                    else ""
                )
                + (
                    " The 2026-05-28 representative h=4,K=8 Qwen3/MTBench "
                    "max32 probes narrow the current failure boundary: bs=1 "
                    "passes "
                    f"{representative['bs1']['matched_count']}/"
                    f"{representative['bs1']['total_count']}, bs=4 passes "
                    f"{representative['bs4']['matched_count']}/"
                    f"{representative['bs4']['total_count']}, but bs=8 "
                    "fails "
                    f"{representative['bs8']['matched_count']}/"
                    f"{representative['bs8']['total_count']} at token "
                    f"{representative['bs8']['first_mismatch_token']} "
                    f"({representative['bs8']['baseline_token']} vs "
                    f"{representative['bs8']['cv_token']}). Disabling dense "
                    "realignment still fails "
                    f"{representative['dense0']['matched_count']}/"
                    f"{representative['dense0']['total_count']}; batched "
                    "prefix plus global barrier still fails "
                    f"{representative['batchedprefix_dense0']['matched_count']}/"
                    f"{representative['batchedprefix_dense0']['total_count']}; "
                    "prefix skip-suffix block rollback executes "
                    f"{representative['rollback_count']} rollback events and "
                    f"releases {representative['rollback_released']} blocks "
                    "but still fails "
                    f"{representative['skiprollback_dense0']['matched_count']}/"
                    f"{representative['skiprollback_dense0']['total_count']}. "
                    "Prefix-reject full-K confirmation aligns the current "
                    "position/slot mapping with one-shot but still fails "
                    f"{representative['rejectconfirm_dense0']['matched_count']}/"
                    f"{representative['rejectconfirm_dense0']['total_count']}, "
                    "which points to accumulated historical KV, EAGLE hidden "
                    "state, or numerical state drift before the current "
                    "verifier call. The exact-safe afterfix probe passes "
                    f"{representative['exactsafe_afterfix']['matched_count']}/"
                    f"{representative['exactsafe_afterfix']['total_count']}. "
                    "The bounded KV debug rerun for the same bs=8 failure "
                    "still fails "
                    f"{representative['kvdebug']['matched_count']}/"
                    f"{representative['kvdebug']['total_count']}, but "
                    "history-KV checksums match "
                    f"{representative['kvdebug_checksum_entries']} entries "
                    "with "
                    f"{representative['kvdebug_checksum_mismatches']} "
                    "mismatches; active ordinals still differ, missing "
                    f"{representative['kvdebug_missing_active']} in CV. "
                    "Expanding prefix-reject full-K confirmation to the "
                    "current full active set still fails "
                    f"{representative['confirm_fullactive']['matched_count']}/"
                    f"{representative['confirm_fullactive']['total_count']} "
                    "and still misses active ordinals "
                    f"{representative['confirm_fullactive_missing_active']}. "
                    "Forced h=4 confirm-all with the global batch barrier "
                    "also fails "
                    f"{representative['confirmall_barrier_h4']['matched_count']}/"
                    f"{representative['confirmall_barrier_h4']['total_count']}; "
                    "its scheduled drafts match the baseline but active "
                    "ordinals still miss "
                    f"{representative['confirmall_barrier_h4_missing_active']}. "
                    "The lockstep iteration-barrier diagnostic holds resolved "
                    "requests until the whole prefix batch finishes, and it "
                    "does make active ordinals equal, but it still fails "
                    f"{representative['lockstep_h4']['matched_count']}/"
                    f"{representative['lockstep_h4']['total_count']} because "
                    "the later DLM draft already differs from one-shot. "
                    "Batched suffix verification under the same lockstep "
                    "barrier also fails "
                    f"{representative['lockstep_batchsuffix']['matched_count']}/"
                    f"{representative['lockstep_batchsuffix']['total_count']}; "
                    "adding 8 prefix-reject dense realign steps also fails "
                    f"{representative['lockstep_prefixdense8']['matched_count']}/"
                    f"{representative['lockstep_prefixdense8']['total_count']}; "
                    "lockstep plus prefix-reject full-K confirmation also "
                    "fails "
                    f"{representative['lockstep_rejectconfirm']['matched_count']}/"
                    f"{representative['lockstep_rejectconfirm']['total_count']}. "
                    "A 2026-05-28 draft-preferred tie tolerance probe with "
                    "`SPECLINK_CV_DRAFT_ACCEPT_EPS=0.01` still fails plain "
                    "h<K chunking "
                    f"{representative['drafteps001']['matched_count']}/"
                    f"{representative['drafteps001']['total_count']}, while "
                    "the grouped batch-wide full-K fallback plus the same "
                    "epsilon passes the Qwen3/MTBench representative case "
                    f"{representative['grouped_drafteps001']['matched_count']}/"
                    f"{representative['grouped_drafteps001']['total_count']}; "
                    "it also passes the Qwen3/math, Llama3.1/math, and "
                    "Llama3.1/MTBench K=8/bs=8 short probes "
                    f"{representative['qwen_math_grouped_drafteps001']['matched_count']}/"
                    f"{representative['qwen_math_grouped_drafteps001']['total_count']}, "
                    f"{representative['llama_grouped_drafteps001']['matched_count']}/"
                    f"{representative['llama_grouped_drafteps001']['total_count']}, "
                    f"{representative['llama_mtbench_grouped_drafteps001']['matched_count']}/"
                    f"{representative['llama_mtbench_grouped_drafteps001']['total_count']}. "
                    "These passing cases are a correctness fallback, not a "
                    "suffix-pruning speedup, because prefix rows are discarded "
                    "and requeued for full-K confirmation. A 2026-05-28 "
                    "DLM rollback attempt added skipped suffix length to the "
                    "EAGLE drafter rollback on prefix reject; that was later "
                    "identified as incorrect because the worker-side rejected "
                    "count is relative to the current prefix verifier forward, "
                    "which never included the skipped suffix. The scheduler "
                    "still discards the suffix logically, and suffix "
                    "verification can update DLM state before returned drafts "
                    "are optionally dropped. The representative Qwen3/MTBench "
                    "K=8/bs=8/max32 gate still fails "
                    f"{representative['dlmrollback']['matched_count']}/"
                    f"{representative['dlmrollback']['total_count']}; with "
                    "dense realignment disabled it also fails "
                    f"{representative['dlmrollback_dense0']['matched_count']}/"
                    f"{representative['dlmrollback_dense0']['total_count']}. "
                    "The historical debug run recorded "
                    f"{representative['dlmrollback_adjust_count']} "
                    "prefix-reject rollback-adjustment events; treat those as "
                    "evidence of the over-rollback bug, not as the desired "
                    "steady-state behavior."
                    if representative["bs8"]["exists"]
                    else ""
                )
            ),
        },
        {
            "area": "shape_drift_diagnosis",
            "requirement": "Explain the observed h<K correctness failure from raw verifier evidence",
            "status": "complete"
            if file_exists(
                root,
                "22_verifier_shape_drift_topk/verifier_shape_drift_topk.md",
            )
            else "missing",
            "evidence": (
                "22_verifier_shape_drift_topk/verifier_shape_drift_topk.md; "
                "69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate/"
                "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm_all_barrier/"
                "active_batch_drift.md; "
                "70_qwen_mtbench_k8_bs8_t64_context_debug_gate/"
                "000_qwen3_8b_mtbench_k8_bs8_chunked_confirm_all_barrier/"
                "active_batch_drift.md; "
                "temp/speclink_cv_qwen_mtbench_k8_bs8_t32_rejectconfirm_dense0_20260528/"
                "000_qwen3_8b_mtbench_k8_bs8_h4_rejectconfirm_dense0/"
                "token_timeline.md"
            ),
            "notes": (
                "Top-k boundary comparison shows zero prefix argmax mismatches "
                "but a low-margin boundary argmax flip between chunked h<K "
                "verification and full-K one-shot. A later batch-fill barrier "
                "fix makes batch-size-8 confirm-all pass, proving the earlier "
                "barrier dispatched before the waiting queue drained. However "
                "plain h<K, low-margin guard, accept-only confirmation, "
                "reject-only confirmation, forced h=1, and dense-realign step "
                "diagnostics still fail. Prefix-reject dense realignment also "
                "fails. A later offline margin sweep shows threshold 1.5 can "
                "retrospectively catch the failed requests, but live threshold "
                "1.5 and batch-wide low-margin fallback still fail 7/8. "
                "Forced h=5/6/7 probes also fail, and h=6/7 show prefix "
                "target argmax mismatches inside matched full drafts. "
                "The K=8 Triton-attention backend probe still fails 5/8, "
                "and the smoke path already uses enforce_eager=True, so the "
                "remaining drift is not fixed by disabling CUDA graph replay "
                "or switching attention backend. The latest batch-wide "
                "prefix-reject fallback fixes the observed Qwen3/math K=8 bs=8 "
                "failure by grouping the requeued full-K confirmations instead "
                "of isolating them as row 0, and that smoke passes 8/8. "
                "Verifier margin is a risk signal, not a correctness fix; "
                "the grouped fallback now has broader matrix evidence and "
                "still fails Llama3.1/math. At bs=16/32 it passes "
                f"{grouped_fallback['extension_passed']}/"
                f"{grouped_fallback['extension_rows']} rows, with remaining "
                "failures concentrated on Llama/math. The earlier bs=8 "
                "failing "
                "trace shows request 0's second prefix has a prefix decision "
                "and target-argmax mismatch: CV accepts 4 prefix tokens for "
                "`[5944, 11, 584, 1205]`, while one-shot baseline accepts 3, "
                "flipping `we first need` vs `we need to first`. "
                "The worker-ordered low-margin probe fixes request row order "
                "but still has one prefix decision mismatch and one full-"
                "confirmation target mismatch; the paired one-shot fallback "
                f"row passes {worker_pass.get('matched_prompts', '8')}/"
                f"{worker_pass.get('total_prompts', '8')}. "
                "The later batch-invariant matrix shows the practical "
                "exactness condition for the current implementation: enabling "
                "`VLLM_BATCH_INVARIANT=1` removes the observed cross-shape "
                "drift in the archived Qwen/Llama, math/MTBench, K=8/K=12, "
                "batch-size-8 smoke. The later 64-token Qwen3/MTBench gate "
                "shows that this condition is still length-scoped: plain h<K "
                f"matches only {long_generation['chunked_matched_count']}/"
                f"{long_generation['chunked_total_count']} at max_tokens=64, "
                "grouped fallback matches only "
                f"{long_generation['grouped_matched_count']}/"
                f"{long_generation['grouped_total_count']}, "
                "while exact-safe passes "
                f"{long_generation['exactsafe_matched_count']}/"
                f"{long_generation['exactsafe_total_count']}. "
                "The debug 8-prompt rerun localizes the first long-output "
                "mismatch to a `prefix_rejected_skip_suffix` commit at token "
                f"{long_generation['debug_first_mismatch_token']}, and "
                "prefix-reject full-K confirmation alone still matches only "
                f"{long_generation['reject_confirm_matched_count']}/"
                f"{long_generation['reject_confirm_total_count']}. "
                "Confirm-all and barrier confirm-all also fail "
                f"{long_generation['confirm_all_matched_count']}/"
                f"{long_generation['confirm_all_total_count']}, so the "
                "long-output failure is not just a missing confirmation branch. "
                "Prefix-reject dense8 realignment also fails "
                f"{long_generation['prefixreject_dense_matched_count']}/"
                f"{long_generation['prefixreject_dense_total_count']}, so "
                "immediate drafter reuse after prefix rejection is not the "
                "full explanation either. Grouping every forced full-K "
                "confirmation under the global batch barrier also fails "
                f"{long_generation['grouped_confirm_all_barrier_matched_count']}/"
                f"{long_generation['grouped_confirm_all_barrier_total_count']}; "
                "its token timeline shows the same scheduled full-K draft IDs "
                "as baseline but a different first target argmax. The paired "
                "`active_batch_drift.*` diagnostic shows the active request "
                "ordinals are also identical `[0,1,2,3,4,5,6,7]`; baseline "
                "prefers token 2213 over 2033 by 0.125 logit, while CV ties "
                "both at 31.625 and picks 2033. That leaves accumulated "
                "verifier/KV or numerical state drift, not isolated "
                "confirmation scheduling or coarse active-set mismatch, as the "
                "remaining explanation. The context-debug rerun in "
                "`70_qwen_mtbench_k8_bs8_t64_context_debug_gate/` still "
                f"matches only {long_generation['context_debug_matched_count']}/"
                f"{long_generation['context_debug_total_count']} at token "
                f"{long_generation['context_debug_first_mismatch_token']}; "
                "its active-batch diagnostic confirms equal "
                "`num_computed_tokens_cpu`, equal `num_tokens_no_spec`, equal "
                "worker output-token count, and equal final 16 committed "
                "context token IDs at the mismatch. That rules out committed "
                "context length or token-tail mismatch as the remaining "
                "explanation. The same diagnostic records different physical "
                "`block_ids_tail` values, so KV layout or accumulated "
                "numerical state remains in scope."
                + (
                    " A follow-up committed-prefix KV recompute diagnostic "
                    "also fails: without the global barrier it matches only "
                    f"{long_generation['recompute_matched_count']}/"
                    f"{long_generation['recompute_total_count']}, and with "
                    "the barrier it matches only "
                    f"{long_generation['recompute_barrier_matched_count']}/"
                    f"{long_generation['recompute_barrier_total_count']}; "
                    "the barrier timeline attributes the first mismatch to a "
                    "dense target step at token "
                    f"{long_generation['recompute_barrier_first_mismatch_token']}. "
                    "This rules out committed-token KV recompute as a "
                    "standalone recovery for the full-K EAGLE3 verifier shape."
                    if long_generation["recompute_barrier_rows"]
                    else ""
                )
                + (
                    " The worker-ordered batched dense-realign recompute "
                    "diagnostic also fails "
                    f"{long_generation['batched_dense_worker_ordered_matched_count']}/"
                    f"{long_generation['batched_dense_worker_ordered_total_count']} "
                    "at token "
                    f"{long_generation['batched_dense_worker_ordered_first_mismatch_token']}; "
                    "its active-batch diagnostic confirms row order and "
                    "context tail are now equal, leaving physical KV layout or "
                    "accumulated numerical state as the remaining suspect."
                    if long_generation["batched_dense_worker_ordered_rows"]
                    else ""
                )
                + (
                    " The prefix no-KV-write probe also fails "
                    f"{long_generation['prefix_nokv_matched_count']}/"
                    f"{long_generation['prefix_nokv_total_count']} at token "
                    f"{long_generation['prefix_nokv_first_mismatch_token']}; "
                    "its active-batch diagnostic shows the baseline first "
                    "verifier row is isolated while CV still verifies a full "
                    "active batch, so simply suppressing prefix-probe KV "
                    "writes is not an equivalent one-shot replay."
                    if long_generation["prefix_nokv_rows"]
                    else ""
                )
                + (
                    " The 2026-05-28 tie-fix debug adds a shorter bs=16 "
                    "reproduction: max12 passes "
                    f"{tie_fix['t12_matched_count']}/"
                    f"{tie_fix['t12_total_count']} after disabling implicit "
                    "draft preference on exact ties, but max32 still matches "
                    f"{tie_fix['t32_matched_count']}/"
                    f"{tie_fix['t32_total_count']} at token "
                    f"{tie_fix['t32_first_mismatch_token']}. The row-0 debug "
                    "shows baseline token 2213 versus CV token 2033 on a "
                    "low-margin verifier row with matching logical context "
                    "tail and different physical block tails. A bounded "
                    "history-KV checksum rerun compares "
                    f"{tie_fix['history_kv_compared_entries']} layer-0 "
                    "entries before verifier position "
                    f"{tie_fix['history_kv_baseline_upto']} and finds "
                    f"{tie_fix['history_kv_checksum_mismatches']} checksum "
                    "mismatches, narrowing the suspect set toward active-batch "
                    "shape, current slot layout, or numerical drift rather "
                    "than simple historical KV content corruption. The "
                    "no-barrier "
                    "rerun still fails "
                    f"{tie_fix['nobarrier_matched_count']}/"
                    f"{tie_fix['nobarrier_total_count']}, while exact-safe "
                    f"passes {tie_fix['exactsafe_matched_count']}/"
                    f"{tie_fix['exactsafe_total_count']}."
                    if tie_fix["t32_rows"]
                    else ""
                )
                + (
                    " The representative h=4,K=8 Qwen3/MTBench max32 probes "
                    "make the failure smaller and cleaner: bs=1 and bs=4 "
                    "match exactly, while bs=8 fails only request 0 at token "
                    f"{representative['bs8']['first_mismatch_token']}. "
                    "The dense0 and batched-prefix/global-barrier variants "
                    "fail at the same token, so the failure is not caused by "
                    "the post-reject dense realignment queue or by isolated "
                    "prefix scheduling alone. The diagnostic skip-suffix "
                    "rollback path releases "
                    f"{representative['rollback_released']} tail blocks over "
                    f"{representative['rollback_count']} events and still "
                    "fails. The prefix-reject full-K confirmation run has "
                    "the same current target positions `[86..93]` and slot "
                    "mapping `[806..813]` as baseline, but its verifier "
                    "argmax is 2033 where baseline is 2213. That rules out "
                    "the current verifier token metadata as a sufficient "
                    "explanation. A follow-up KV-debug rerun compares "
                    f"{representative['kvdebug_checksum_entries']} "
                    "bounded pre-target history-KV entries and finds "
                    f"{representative['kvdebug_checksum_mismatches']} "
                    "checksum mismatches; its active-batch diagnostic still "
                    "shows CV missing active ordinals "
                    f"{representative['kvdebug_missing_active']}. This shifts "
                    "the next fix from dirty historical KV toward reproducing "
                    "the one-shot active-batch shape or understanding the "
                    "batch-shape numerical path. The follow-up "
                    "full-active-set confirmation diagnostic still fails "
                    f"{representative['confirm_fullactive']['matched_count']}/"
                    f"{representative['confirm_fullactive']['total_count']} "
                    "and still misses active ordinals "
                    f"{representative['confirm_fullactive_missing_active']}, "
                    "because one request has already progressed out of the "
                    "batch by the time request 0 reaches the failing full-K "
                    "confirmation. Forced h=4 confirm-all plus the global "
                    "barrier also fails "
                    f"{representative['confirmall_barrier_h4']['matched_count']}/"
                    f"{representative['confirmall_barrier_h4']['total_count']}; "
                    "the scheduled draft IDs match, but active ordinals still "
                    "miss "
                    f"{representative['confirmall_barrier_h4_missing_active']} "
                    "and the target slot mapping differs. That makes exact "
                    "one-shot equivalence look like a global timeline/shape "
                    "property, not a local confirmation-branch repair. "
                    "A stricter lockstep iteration barrier removes the active "
                    "ordinal gap (`active_request_ordinals_equal="
                    f"{representative['lockstep_h4_active_equal']}`), but the "
                    "same gate still fails "
                    f"{representative['lockstep_h4']['matched_count']}/"
                    f"{representative['lockstep_h4']['total_count']}; the "
                    "scheduled draft IDs no longer match the one-shot verifier "
                    "(`scheduled_drafts_equal="
                    f"{representative['lockstep_h4_scheduled_equal']}`), and "
                    "slot mapping still differs (`target_slot_mapping_gid0_equal="
                    f"{representative['lockstep_h4_slot_equal']}`). Batching "
                    "the suffix phase under lockstep still fails "
                    f"{representative['lockstep_batchsuffix']['matched_count']}/"
                    f"{representative['lockstep_batchsuffix']['total_count']}. "
                    "Prefix-reject dense realignment under lockstep also fails "
                    f"{representative['lockstep_prefixdense8']['matched_count']}/"
                    f"{representative['lockstep_prefixdense8']['total_count']}, "
                    "moving the mismatch to a dense target step. Lockstep plus "
                    "prefix-reject full-K confirmation still fails "
                    f"{representative['lockstep_rejectconfirm']['matched_count']}/"
                    f"{representative['lockstep_rejectconfirm']['total_count']}, "
                    "so confirmation after an h<K probe is not enough once the "
                    "trajectory has drifted. The bounded proposer drift trace "
                    "then compares "
                    f"{representative['proposer_drift_pairs']} proposer pairs "
                    "and finds the first drift reason is "
                    f"`{representative['proposer_drift_reason']}`: at request "
                    f"{proposer_first.get('request_ordinal', '')}, output "
                    f"{proposer_first.get('worker_output_token_count', '')}, "
                    "the next token still matches but the target-hidden "
                    "checksum delta has abs_sum "
                    f"{proposer_hidden_delta.get('abs_sum', '')}, and the "
                    "EAGLE draft diverges. A recompute plus physical block "
                    "rollback probe still fails "
                    f"{representative['recompute_blockrollback']['matched_count']}/"
                    f"{representative['recompute_blockrollback']['total_count']} "
                    "at token "
                    f"{representative['recompute_blockrollback']['first_mismatch_token']}. "
                    "A draft-preferred tie tolerance of 0.01 does not repair "
                    "plain h<K chunking, but grouped batch-wide full-K "
                    "confirmation with the same tolerance passes the short "
                    "2-model x 2-dataset K=8/bs=8 representative probes. "
                    "The remaining suspect for true suffix-pruning is therefore DLM/EAGLE "
                    "target-hidden state or physical slot-layout drift "
                    "accumulated by earlier h<K verifier steps. A follow-up "
                    "DLM rollback audit supersedes the old skipped-suffix "
                    "rollback interpretation: skipped suffix tokens must be "
                    "discarded by scheduler/request bookkeeping, not added to "
                    "the worker-side current-forward rejected count. The old "
                    "Qwen3/MTBench K=8/bs=8/max32 case still fails at token 31 "
                    "under that historical probe. With dense "
                    "realignment disabled, the bounded proposer trace finds "
                    "first drift reason "
                    f"`{representative['dlmrollback_dense0_drift_reason']}` "
                    "at output "
                    f"{dlmrollback_dense0_first.get('worker_output_token_count', '')}: "
                    "the next token still matches, but scheduled drafts and "
                    "later EAGLE drafts diverge, with target-hidden abs_sum "
                    "delta "
                    f"{dlmrollback_dense0_hidden_delta.get('abs_sum', '')}. "
                    "This narrows the remaining issue away from simple DLM "
                    "suffix rollback and toward target-hidden/slot-layout or "
                    "active-shape numerical drift."
                    if representative["bs8"]["exists"]
                    else ""
                )
            ),
        },
        {
            "area": "datasets_models_k",
            "requirement": "Qwen3 and Llama3.1 on math and MTBench for K=8 and K=12",
            "status": (
                "complete_batch_invariant_smoke"
                if has_batch_invariant_bs8_hk
                else "partial"
            ),
            "evidence": "13_live_correctness_gate; 19_cross_model_dataset_k8_bs8; 21_k12_cross_model_data_bs8; 52_grouped_k8_k12_cross_model_data_bs8; 53_worker_ordered_lowmargin_llama_math_k8; 54_batch_invariant_chunked_matrix; 58_batch_invariant_bs16_bs32_matrix; 59_grouped_batchwide_fallback_bs16_bs32_matrix; 60_qwen_mtbench_k8_bs8_t64_batchinv_gate; 61_qwen_mtbench_k8_bs8_t64_exactsafe_gate; 63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate; 64_qwen_mtbench_k8_bs8_t64_debug_plain; 65_qwen_mtbench_k8_bs8_t64_reject_confirm_gate; 66_qwen_mtbench_k8_bs8_t64_confirm_all_gate; 67_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_gate; 68_qwen_mtbench_k8_bs8_t64_prefixreject_dense8_gate; 69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate; 70_qwen_mtbench_k8_bs8_t64_context_debug_gate; temp/speclink_cv_qwen_mtbench_k8_bs8_t32_proposer_debug_20260528; temp/speclink_cv_qwen_mtbench_k8_bs8_t32_recompute_blockrollback_20260528; temp/speclink_cv_qwen_mtbench_k8_bs8_t32_drafteps001_20260528; temp/speclink_cv_qwen_mtbench_k8_bs8_t32_grouped_drafteps001_20260528; temp/speclink_cv_qwen_math_k8_bs8_t32_grouped_drafteps001_20260528; temp/speclink_cv_llama_math_k8_bs8_t32_grouped_drafteps001_20260528; temp/speclink_cv_llama_mtbench_k8_bs8_t32_grouped_drafteps001_20260528",
            "notes": (
                "Token gates cover both models/datasets for K=8 and K=12 at "
                "bs=8. Without batch-invariant execution, grouped fallback "
                "passes 7/8 rows and fails Llama3.1/math/K=8. With "
                "`VLLM_BATCH_INVARIANT=1`, plain h<K chunked verification "
                "passes all bs=8 rows across Qwen3/Llama3.1 and math/MTBench. "
                "The bs=16/32 extension is not complete: "
                f"{batch_invariant['extension_passed']}/"
                f"{batch_invariant['extension_rows']} rows pass. Grouped "
                "batch-wide fallback improves bs=16/32 to "
                f"{grouped_fallback['extension_passed']}/"
                f"{grouped_fallback['extension_rows']} rows, but still fails "
                "Llama/math. The 64-token Qwen3/MTBench/K=8/bs=8 follow-up "
                "also fails under plain batch-invariant h<K, and grouped "
                "fallback still fails 22/24, so the bs=8 positive smoke is "
                "not length-general. The debug 8-prompt rerun localizes the "
                "first mismatch to a prefix-reject skip-suffix commit at token "
                f"{long_generation['debug_first_mismatch_token']}; "
                "prefix-reject full-K confirmation alone still fails "
                f"{long_generation['reject_confirm_matched_count']}/"
                f"{long_generation['reject_confirm_total_count']}, and "
                "confirm-all/barrier-confirm-all still fail "
                f"{long_generation['confirm_all_matched_count']}/"
                f"{long_generation['confirm_all_total_count']}. "
                "Prefix-reject dense8 realignment also fails "
                f"{long_generation['prefixreject_dense_matched_count']}/"
                f"{long_generation['prefixreject_dense_total_count']}; "
                "grouped full-K confirmation barrier also fails "
                f"{long_generation['grouped_confirm_all_barrier_matched_count']}/"
                f"{long_generation['grouped_confirm_all_barrier_total_count']}."
            ),
        },
        {
            "area": "batch_sizes",
            "requirement": "Batch sizes 8,16,32",
            "status": "failed_batch_invariant_bs16_bs32",
            "evidence": "18_qwen_math_k8_batch_sweep; 20_qwen_math_k12_batch_sweep; 58_batch_invariant_bs16_bs32_matrix; 59_grouped_batchwide_fallback_bs16_bs32_matrix; 60_qwen_mtbench_k8_bs8_t64_batchinv_gate; 63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate",
            "notes": (
                "Qwen/math covers bs=8/16/32 for K=8 and K=12 without "
                "batch-invariant execution and fails h<K at all three sizes. "
                "The batch-invariant bs=16/32 matrix was run and passes only "
                f"{batch_invariant['extension_passed']}/"
                f"{batch_invariant['extension_rows']} rows, so the TODO "
                "batch-size correctness requirement is not satisfied. The "
                "grouped batch-wide prefix-reject fallback passes "
                f"{grouped_fallback['extension_passed']}/"
                f"{grouped_fallback['extension_rows']} bs=16/32 rows, but "
                "still fails Llama/math and is not a complete solution."
            ),
        },
        {
            "area": "baselines_ablation",
            "requirement": "pure vLLM, EAGLE3 one-shot, 2x2x2 CV ablations, best CV",
            "status": (
                "partial_batch_invariant_guidellm_smoke"
                if has_valid_end_to_end
                else "blocked_by_missing_batch_invariant_guidellm"
                if has_batch_invariant_bs8_hk
                else "blocked_by_correctness"
            ),
            "evidence": "09_reports/summary_metrics.csv",
            "notes": (
                f"GuideLLM rows={guide['guidellm_rows']} cv_rows={guide['cv_rows']} "
                f"valid_cv_rows={guide['valid_cv_rows']} "
                f"quality_valid_cv_rows={guide['quality_valid_cv_rows']} "
                f"math_quality_followup_rows={guide['math_quality_rows']} "
                f"steady_rows={guide['steady_rows']} "
                f"contribution_rows={guide['contribution_rows']}. The archived "
                "batch-invariant smoke and Qwen3/math/K=8/bs=8 pilot validate "
                "the serving path, but the valid CV rows are slower than "
                "EAGLE3 and cover only one model/dataset/K family. The newer "
                "math-quality follow-up includes quality-preserving staged CV "
                "rows and steady-state/contribution diagnostics under the "
                "relaxed EM-preservation metric, but those rows do not satisfy "
                "the original exact token-id TODO gate. Full GuideLLM baselines "
                "and 2x2x2 exact ablations still need to be rerun across the "
                "TODO matrix before a best-CV exactness claim."
            ),
        },
        {
            "area": "figures",
            "requirement": "Figures with source CSVs",
            "status": (
                "complete_negative_artifacts"
                if count_files(root, "08_figures/*.png") >= 16
                and count_files(root, "08_figures/*.csv") >= 16
                else "partial"
            ),
            "evidence": "08_figures/",
            "notes": f"png={count_files(root, '08_figures/*.png')} csv={count_files(root, '08_figures/*.csv')}; many are pilot/negative-result figures.",
        },
        {
            "area": "final_report",
            "requirement": "SPECLINK_CV_REPORT.md, summary_metrics.csv/json, patch snapshot",
            "status": (
                "complete_correctness_audit_pending_guidellm"
                if has_batch_invariant_bs8_hk
                else "complete_negative_report"
            ),
            "evidence": "09_reports/SPECLINK_CV_REPORT.md; 09_reports/summary_metrics.csv; patches/vllm_speclink_cv.diff",
            "notes": (
                "Report includes the positive bs=8 batch-invariant matrix and "
                "the negative bs=16/32 extension; it is not a successful "
                "speedup report."
            ),
        },
        {
            "area": "anti_cheat",
            "requirement": "Do not claim simulated or mismatched speedups",
            "status": "complete",
            "evidence": "speedup_claim_status fields; SPECLINK_CV_REPORT.md",
            "notes": "Invalid h<K and exact-safe fallback rows are explicitly blocked from speedup claims.",
        },
    ]
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = [
        "# SpecLink-CV TODO Requirement Audit",
        "",
        "This file audits the current artifacts against `TODO.md`. It is not a "
        "success report: items marked `failed` or `blocked_by_correctness` still "
        "prevent the active goal from being complete.",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Requirements",
            "",
            "| area | status | requirement | evidence | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['area']} | {row['status']} | {row['requirement']} | "
            f"`{row['evidence']}` | {row['notes']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit(root: Path) -> list[dict[str, Any]]:
    rows = build_audit(root)
    write_csv(root / "09_reports/TODO_REQUIREMENT_AUDIT.csv", rows)
    write_json(root / "09_reports/TODO_REQUIREMENT_AUDIT.json", {"rows": rows})
    write_markdown(root / "09_reports/TODO_REQUIREMENT_AUDIT.md", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.audit_root.resolve()
    rows = write_audit(root)
    print(
        json.dumps(
            {
                "audit_root": str(root),
                "rows": len(rows),
                "output_md": str(root / "09_reports/TODO_REQUIREMENT_AUDIT.md"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
