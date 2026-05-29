#!/usr/bin/env python3
"""Finalize a SpecLink-CV correctness audit result tree.

The GuideLLM matrix report is useful for end-to-end metrics, but the live
SpecLink-CV gate is stricter: token IDs must match EAGLE3 one-shot before any
h<K speedup can be claimed.  This script collects the already-run token-id
gates and rewrites the top-level report so the final artifact reflects the
current correctness evidence instead of an older pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.speclink_cv.audit_todo_requirements import write_audit

DEFAULT_AUDIT_ROOT = (
    REPO_ROOT
    / "examples/evaluate/eval-guidellm/results"
    / "speclink_cv_correctness_audit_20260526_230036"
)
EVAL_GUIDELLM_ROOT = REPO_ROOT / "examples/evaluate/eval-guidellm"
TEMP_ROOT = EVAL_GUIDELLM_ROOT / "temp"

SUMMARY_COLUMNS = [
    "measurement_type",
    "source",
    "model",
    "dataset",
    "K",
    "batch_size",
    "method",
    "throughput",
    "speedup_vs_eagle3",
    "ttft_p95",
    "itl_p95",
    "e2e_p95",
    "exact_match_vs_eagle3",
    "selected_h_avg",
    "skipped_suffix_ratio",
    "extra_tlm_forwards_per_request",
    "queue_wait_p95",
    "gpu_util",
    "fallback_ratio",
    "quality_score",
    "quality_correct",
    "quality_evaluable",
    "quality_gate_status",
    "steady_gen_tps_mean",
    "steady_speedup_vs_eagle3",
    "end_to_end_speedup_vs_eagle3",
    "dlm_suffix_saving_speedup",
    "batch_scheduling_speedup",
    "status",
    "speedup_claim_valid",
    "speedup_claim_status",
    "matched_count",
    "total_count",
    "first_mismatch_index",
    "first_mismatch_token_index",
    "evidence_path",
]

LATEST_MATH_QUALITY_ROOT = (
    TEMP_ROOT / "speclink_cv_math_k8_k12_bs8_bs32_staged_quality_20260528"
)
LATEST_CONTRIBUTION_ROOT = (
    TEMP_ROOT / "speclink_cv_contribution_ablation_k12_bs16_20260528"
)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(SUMMARY_COLUMNS)
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


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def collect_token_gate_rows(root: Path) -> list[dict[str, Any]]:
    candidates = [
        root / "13_live_correctness_gate/combined_summary.csv",
        root / "18_qwen_math_k8_batch_sweep/combined_summary.csv",
        root / "19_cross_model_dataset_k8_bs8/combined_summary.csv",
        root / "20_qwen_math_k12_batch_sweep/combined_summary.csv",
        root / "21_k12_cross_model_data_bs8/combined_summary.csv",
        root / "52_grouped_k8_k12_cross_model_data_bs8/summary.csv",
        root / "54_batch_invariant_chunked_matrix/summary.csv",
        root / "57_batch_invariant_guidellm_prompt_gate_t32/summary.csv",
        root / "58_batch_invariant_bs16_bs32_matrix/summary.csv",
        root / "59_grouped_batchwide_fallback_bs16_bs32_matrix/summary.csv",
        root / "60_qwen_mtbench_k8_bs8_t64_batchinv_gate/summary.csv",
        root / "61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/summary.csv",
        root / "63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/summary.csv",
        root / "64_qwen_mtbench_k8_bs8_t64_debug_plain/summary.csv",
        root / "65_qwen_mtbench_k8_bs8_t64_reject_confirm_gate/summary.csv",
        root / "66_qwen_mtbench_k8_bs8_t64_confirm_all_gate/summary.csv",
        root / "67_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_gate/summary.csv",
        root / "68_qwen_mtbench_k8_bs8_t64_prefixreject_dense8_gate/summary.csv",
        root
        / "69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate/summary.csv",
        root / "70_qwen_mtbench_k8_bs8_t64_context_debug_gate/summary.csv",
    ]
    supplemental_temp_candidates = [
        TEMP_ROOT / "speclink_cv_tie_prefer_draft_confirm_all_barrier_20260527/summary.csv",
        TEMP_ROOT / "speclink_cv_tie_prefer_draft_confirm_all_isolated_20260527/summary.csv",
        TEMP_ROOT / "speclink_cv_exactsafe_after_tie_20260527/summary.csv",
        TEMP_ROOT / "speclink_cv_tie_argmax_llama_math_k8_bs16_20260527/summary.csv",
        TEMP_ROOT / "speclink_cv_recovery_qwen_math_k8_bs32_20260527/summary.csv",
        TEMP_ROOT / "speclink_cv_recompute_qwen_mtbench_k8_bs8_t64_20260527/summary.csv",
        TEMP_ROOT / "speclink_cv_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/summary.csv",
        TEMP_ROOT
        / (
            "speclink_cv_batched_dense_worker_ordered_recompute_barrier_"
            "qwen_mtbench_k8_bs8_t64_20260527/summary.csv"
        ),
        TEMP_ROOT
        / (
            "speclink_cv_prefix_nokv_recompute_barrier_"
            "qwen_mtbench_k8_bs8_t64_20260527/summary.csv"
        ),
        TEMP_ROOT
        / "speclink_cv_llama_math_k8_bs8_t16_barrier_argmax_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs8_t32_barrier_argmax_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_argmax_barrier_matrix_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t12_tiefix_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_tiefix_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_row0_debug_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_nobarrier_tiefix_20260528/summary.csv",
        TEMP_ROOT
        / "speclink_cv_qwen_mtbench_k8_bs16_t32_exactsafe_tiefix_20260528/summary.csv",
    ]
    candidates.extend(supplemental_temp_candidates)
    candidates.extend(sorted((root / "14_backend_probe").glob("*/summary.csv")))
    candidates.extend(sorted((root / "15_suffix_replay_probe").glob("*/summary.csv")))
    candidates.extend(sorted((root / "16_prefix_reject_confirm_probe").glob("*/summary.csv")))
    candidates.extend(sorted((root / "17_exactsafe_after_confirm_probe").glob("*/summary.csv")))
    candidates.append(root / "24_confirm_all_isolated_probe/summary.csv")
    candidates.append(root / "25_confirm_all_batched_probe/summary.csv")
    correctness_candidates = [
        (
            root / "26_low_margin_probe/correctness.json",
            "chunked_low_margin",
            8,
        ),
        (
            root / "27_low_margin_reject_confirm_probe/correctness.json",
            "chunked_reject_confirm_low_margin",
            8,
        ),
        (
            root
            / "28_low_margin_reject_confirm_isolated_probe/correctness.json",
            "chunked_reject_confirm_low_margin_isolated",
            8,
        ),
        (
            root / "29_bs1_shape_probe/correctness.json",
            "chunked_bs1",
            1,
        ),
        (
            root / "30_bs1_confirm_all_probe/correctness.json",
            "chunked_confirm_all_bs1",
            1,
        ),
        (
            root / "31_barrier_confirm_all_probe/correctness.json",
            "chunked_confirm_all_barrier",
            8,
        ),
        (
            root / "32_barrier_batchfill_confirm_all_probe/correctness.json",
            "chunked_confirm_all_barrier_batchfill",
            8,
        ),
        (
            root / "33_barrier_batchfill_plain_probe/correctness.json",
            "chunked_barrier_batchfill",
            8,
        ),
        (
            root / "34_barrier_batchfill_margin_probe/correctness.json",
            "chunked_low_margin_barrier_batchfill",
            8,
        ),
        (
            root / "35_barrier_batchfill_accept_confirm_probe/correctness.json",
            "chunked_accept_confirm_barrier_batchfill",
            8,
        ),
        (
            root / "36_barrier_batchfill_reject_confirm_probe/correctness.json",
            "chunked_reject_confirm_barrier_batchfill",
            8,
        ),
        (
            root / "37_barrier_batchfill_h1_plain_probe/correctness.json",
            "chunked_h1_barrier_batchfill",
            8,
        ),
        (
            root / "38_barrier_batchfill_dense0_probe/correctness.json",
            "chunked_dense0_barrier_batchfill",
            8,
        ),
        (
            root / "39_barrier_batchfill_dense1_probe/correctness.json",
            "chunked_dense1_barrier_batchfill",
            8,
        ),
        (
            root
            / "40_barrier_batchfill_prefixreject_dense1_probe/correctness.json",
            "chunked_prefixreject_dense1_barrier_batchfill",
            8,
        ),
        (
            root
            / "41_barrier_batchfill_prefixreject_dense8_probe/correctness.json",
            "chunked_prefixreject_dense8_barrier_batchfill",
            8,
        ),
        (
            root / "43_barrier_batchfill_margin15_probe/correctness.json",
            "chunked_low_margin15_barrier_batchfill",
            8,
        ),
        (
            root / "44_barrier_batchwide_margin15_probe/correctness.json",
            "chunked_batchwide_low_margin15_barrier_batchfill",
            8,
        ),
        (
            root
            / "45_barrier_batchwide_margin15_refined_probe/correctness.json",
            "chunked_batchwide_low_margin15_refined",
            8,
        ),
        (
            root
            / "46_barrier_batchwide_margin15_reuses_confirm_probe/correctness.json",
            "chunked_batchwide_low_margin15_reuses_confirm",
            8,
        ),
        (
            root / "47_barrier_batchfill_h5_probe/correctness.json",
            "chunked_h5_barrier_batchfill",
            8,
        ),
        (
            root / "48_barrier_batchfill_h6_probe/correctness.json",
            "chunked_h6_barrier_batchfill",
            8,
        ),
        (
            root / "49_barrier_batchfill_h7_probe/correctness.json",
            "chunked_h7_barrier_batchfill",
            8,
        ),
        (
            root / "50_barrier_batchfill_triton_k8_probe/correctness.json",
            "chunked_triton_attn_k8_barrier_batchfill",
            8,
        ),
        (
            root / "51_grouped_batchwide_prefixreject_probe/correctness.json",
            "chunked_grouped_batchwide_prefixreject_barrier_batchfill",
            8,
        ),
        (
            root
            / "53_worker_ordered_lowmargin_llama_math_k8/"
            / "worker_ordered_lowmargin_full_replay_fail/correctness.json",
            "chunked_worker_ordered_lowmargin_full_replay",
            8,
        ),
        (
            root
            / "53_worker_ordered_lowmargin_llama_math_k8/"
            / "confidence_minbenefit_oneshot_pass/correctness.json",
            "exactsafe",
            8,
        ),
    ]

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def append_row(raw: dict[str, Any], path: Path) -> None:
            mode = str(raw.get("mode", ""))
            matched_count = safe_int(raw.get("matched_count")) or 0
            total_count = safe_int(raw.get("total_count")) or 0
            exact = matched_count / total_count if total_count else ""
            k = safe_int(raw.get("K"))
            selected_h = ""
            method = f"speclink_cv_{mode}"
            selected_h_override = raw.get("selected_h")
            if mode == "chunked":
                selected_h = (k // 2) if k else ""
                speedup_status = (
                    "invalid_correctness_mismatch"
                    if exact != 1.0
                    else "token_gate_pass_no_throughput"
                )
            elif mode in {"chunked_confirm", "chunked_confirm_all"}:
                selected_h = (k // 2) if k else ""
                speedup_status = (
                    "invalid_correctness_mismatch"
                    if exact != 1.0
                    else "token_gate_pass_no_throughput"
                )
            elif mode.startswith("chunked_"):
                selected_h = (k // 2) if k else ""
                speedup_status = (
                    "invalid_correctness_mismatch"
                    if exact != 1.0
                    else "token_gate_pass_no_throughput"
                )
            elif mode == "exactsafe":
                selected_h = k or ""
                speedup_status = "invalid_no_live_chunking"
            else:
                speedup_status = "diagnostic"
            if selected_h_override not in (None, ""):
                selected_h = selected_h_override
            source = raw.get("source")
            if source:
                evidence = source
            elif path.is_relative_to(root):
                evidence = rel(path, root)
            else:
                evidence = raw.get("output_dir") or raw.get("output_json") or rel(path, root)
            key = (
                raw.get("model"),
                raw.get("dataset"),
                raw.get("K"),
                raw.get("batch_size"),
                raw.get("num_prompts"),
                raw.get("max_tokens"),
                mode,
                evidence,
            )
            if key in seen:
                return
            seen.add(key)
            rows.append(
                {
                    "measurement_type": "token_id_correctness_gate",
                    "model": raw.get("model", ""),
                    "dataset": raw.get("dataset", ""),
                    "K": raw.get("K", ""),
                    "batch_size": raw.get("batch_size", ""),
                    "method": method,
                    "throughput": "",
                    "speedup_vs_eagle3": "",
                    "ttft_p95": "",
                    "itl_p95": "",
                    "e2e_p95": "",
                    "exact_match_vs_eagle3": exact,
                    "selected_h_avg": selected_h,
                    "skipped_suffix_ratio": "",
                    "extra_tlm_forwards_per_request": "",
                    "queue_wait_p95": "",
                    "gpu_util": "",
                    "fallback_ratio": 1.0 if mode == "exactsafe" else 0.0,
                    "status": raw.get("status", ""),
                    "speedup_claim_valid": 0,
                    "speedup_claim_status": speedup_status,
                    "matched_count": matched_count,
                    "total_count": total_count,
                    "first_mismatch_index": raw.get("first_mismatch_index", ""),
                    "first_mismatch_token_index": raw.get(
                        "first_mismatch_token_index", ""
                    ),
                    "evidence_path": evidence,
                    "num_prompts": raw.get("num_prompts", ""),
                    "max_tokens": raw.get("max_tokens", ""),
                    "raw_summary": rel(path, root),
                }
            )
    for path in candidates:
        for raw in read_csv(path):
            append_row(raw, path)

    for path, mode, fallback_batch_size in correctness_candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        matched_items = list(data.get("matched_items") or [])
        mismatches = list(data.get("mismatches") or [])
        first_mismatch = mismatches[0] if mismatches else {}
        model = str(data.get("model") or "")
        prompts_jsonl = str(data.get("prompts_jsonl") or "")
        if "Qwen3" in model:
            model_name = "qwen3_8b"
        elif "Llama" in model:
            model_name = "llama3_1_8b"
        else:
            model_name = model
        dataset = "math" if "math_reasoning" in prompts_jsonl else prompts_jsonl
        append_row(
            {
                "model": model_name,
                "dataset": dataset,
                "K": data.get("num_spec_tokens", ""),
                "batch_size": data.get("max_num_seqs", fallback_batch_size),
                "selected_h": (
                    data.get("force_prefix_len")
                    if data.get("force_prefix_len")
                    else (
                        (int(data.get("num_spec_tokens") or 0) // 2)
                        if mode.startswith("chunked")
                        else ""
                    )
                ),
                "num_prompts": data.get("num_prompts", ""),
                "max_tokens": data.get("max_tokens", ""),
                "mode": mode,
                "matched": str(bool(data.get("matched"))).lower(),
                "matched_count": sum(1 for item in matched_items if item),
                "total_count": len(matched_items),
                "status": "pass" if data.get("matched") else "fail",
                "first_mismatch_index": first_mismatch.get("index", ""),
                "first_mismatch_token_index": first_mismatch.get(
                    "first_diff_index", ""
                ),
                "output_json": rel(path, root),
                "source": rel(path, root),
            },
            path,
        )
    return rows


def collect_guidellm_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [
        root / "current_code_guidellm_3case/summary_metrics.csv",
        root / "current_code_guidellm_3case_denseiso/summary_metrics.csv",
        root / "current_code_guidellm_3case_exactsafe/summary_metrics.csv",
        root / "55_batch_invariant_guidellm_smoke/summary_metrics.csv",
        root / "56_batch_invariant_guidellm_qwen_math_k8_bs8_pilot/summary_metrics.csv",
        root / "62_qwen_mtbench_k8_bs8_guidellm_batchinv_ablation/summary_metrics.csv",
        LATEST_MATH_QUALITY_ROOT / "summary_metrics.csv",
        LATEST_CONTRIBUTION_ROOT / "batched/summary_metrics.csv",
        LATEST_CONTRIBUTION_ROOT / "singleton_live/summary_metrics.csv",
    ]
    seen: set[tuple[Any, ...]] = set()
    for path in candidates:
        for raw in read_csv(path):
            measurement_type = raw.get("measurement_type", "guidellm_end_to_end")
            if measurement_type != "guidellm_end_to_end":
                continue
            row = {key: raw.get(key, "") for key in SUMMARY_COLUMNS}
            row["measurement_type"] = measurement_type
            row["model"] = raw.get("model", "")
            row["dataset"] = raw.get("dataset", "")
            row["K"] = raw.get("K", "")
            row["batch_size"] = raw.get("batch_size", "")
            row["method"] = raw.get("method", "")
            row["status"] = raw.get("status", "")
            row["source"] = (
                "math_quality_followup"
                if path.is_relative_to(LATEST_MATH_QUALITY_ROOT)
                else "contribution_batched_followup"
                if path.is_relative_to(LATEST_CONTRIBUTION_ROOT / "batched")
                else "contribution_singleton_followup"
                if path.is_relative_to(LATEST_CONTRIBUTION_ROOT / "singleton_live")
                else "archived_guidellm"
            )
            row["output_dir"] = raw.get("output_dir", "")
            row["evidence_path"] = rel(path, root)
            key = (
                row["model"],
                row["dataset"],
                row["K"],
                row["batch_size"],
                row["method"],
                row["output_dir"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def collect_followup_diagnostic_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostic_paths = [
        (
            LATEST_MATH_QUALITY_ROOT / "09_reports/steady_state_throughput.csv",
            "steady_state_throughput",
            "math_quality_followup",
        ),
        (
            LATEST_CONTRIBUTION_ROOT / "09_reports/steady_state_throughput.csv",
            "steady_state_throughput",
            "contribution_followup",
        ),
        (
            LATEST_CONTRIBUTION_ROOT / "09_reports/contribution_ablation.csv",
            "contribution_ablation",
            "contribution_followup",
        ),
    ]
    seen: set[tuple[Any, ...]] = set()
    for path, measurement_type, source in diagnostic_paths:
        for raw in read_csv(path):
            row = dict(raw)
            row["measurement_type"] = measurement_type
            row["source"] = source
            row["evidence_path"] = rel(path, root)
            key = (
                measurement_type,
                source,
                row.get("model"),
                row.get("dataset"),
                row.get("K"),
                row.get("batch_size"),
                row.get("method"),
                row.get("source_group"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def latest_math_quality_rows() -> list[dict[str, Any]]:
    summary_rows = read_csv(LATEST_MATH_QUALITY_ROOT / "summary_metrics.csv")
    steady_rows = read_csv(
        LATEST_MATH_QUALITY_ROOT / "09_reports/steady_state_throughput.csv"
    )
    by_key = {
        (
            row.get("model"),
            row.get("dataset"),
            row.get("K"),
            row.get("batch_size"),
            row.get("method"),
        ): row
        for row in summary_rows
    }
    steady_by_key = {
        (
            row.get("model"),
            row.get("dataset"),
            row.get("K"),
            row.get("batch_size"),
            row.get("method"),
        ): row
        for row in steady_rows
    }
    rows: list[dict[str, Any]] = []
    for key, cv_row in by_key.items():
        model, dataset, k, batch_size, method = key
        if method != "cv_half_async_staged_simple":
            continue
        baseline = by_key.get((model, dataset, k, batch_size, "eagle3_oneshot"), {})
        steady = steady_by_key.get(key, {})
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "K": k,
                "batch_size": batch_size,
                "baseline_tps": baseline.get("throughput", ""),
                "cv_tps": cv_row.get("throughput", ""),
                "e2e_speedup": cv_row.get("speedup_vs_eagle3", ""),
                "steady_speedup": steady.get("steady_speedup_vs_eagle3", ""),
                "quality_score": cv_row.get("quality_score", ""),
                "quality_correct": cv_row.get("quality_correct", ""),
                "quality_evaluable": cv_row.get("quality_evaluable", ""),
                "quality_gate_status": cv_row.get("quality_gate_status", ""),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("model", "")),
            safe_int(row.get("K")) or 0,
            safe_int(row.get("batch_size")) or 0,
        ),
    )


def render_math_quality_followup(
    math_quality_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "",
        "## Math-Quality Performance Follow-up",
        "",
        "The latest math-only follow-up is included as performance evidence "
        "under the relaxed metric requested after the strict token-id work: "
        "math answer EM must be preserved, but the output is not required to "
        "be byte-for-byte identical to EAGLE3. These rows are therefore useful "
        "for performance direction and quality checks, but they do not close "
        "the original TODO exact-greedy correctness gate.",
        "",
    ]
    if math_quality_rows:
        lines.extend(
            [
                "| model | K | batch | EAGLE3 TPS | CV TPS | E2E speedup | steady speedup | quality | gate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in math_quality_rows:
            quality = (
                f"{row.get('quality_correct', '')}/"
                f"{row.get('quality_evaluable', '')}"
            )
            lines.append(
                f"| {row.get('model', '')} | {row.get('K', '')} | "
                f"{row.get('batch_size', '')} | {fmt(row.get('baseline_tps'), 1)} | "
                f"{fmt(row.get('cv_tps'), 1)} | {fmt(row.get('e2e_speedup'))} | "
                f"{fmt(row.get('steady_speedup'))} | {quality} | "
                f"{row.get('quality_gate_status', '')} |"
            )
    else:
        lines.append(
            "No `speclink_cv_math_k8_k12_bs8_bs32_staged_quality_20260528` "
            "summary was found under `temp/`."
        )

    lines.extend(
        [
            "",
            "The steady-state column is parsed from vLLM periodic throughput "
            "logs using only samples where `Running >= 0.8 * batch_size`; blank "
            "entries mean the run finished before enough periodic vLLM samples "
            "were emitted. This separates full-batch serving behavior from the "
            "fixed-request warmup/drain tail included in GuideLLM end-to-end TPS.",
            "",
        ]
    )
    if contribution_rows:
        lines.extend(
            [
                "The K=12/bs=16 contribution ablation keeps skip-suffix enabled "
                "in every CV row, then separates TLM suffix skip alone, staged "
                "DLM suffix saving, and batched scheduling:",
                "",
                "| model | EAGLE3 TPS | non-staged TPS | staged TPS | singleton TPS | TLM-only speedup | staged speedup | DLM saving | batch scheduling | quality gate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(contribution_rows, key=lambda item: str(item.get("model", ""))):
            lines.append(
                f"| {row.get('model', '')} | {fmt(row.get('baseline_tps'), 1)} | "
                f"{fmt(row.get('nonstaged_batched_tps'), 1)} | "
                f"{fmt(row.get('staged_batched_tps'), 1)} | "
                f"{fmt(row.get('staged_singleton_tps'), 1)} | "
                f"{fmt(row.get('nonstaged_batched_speedup_vs_eagle3'))} | "
                f"{fmt(row.get('staged_batched_speedup_vs_eagle3'))} | "
                f"{fmt(row.get('dlm_suffix_saving_speedup'))} | "
                f"{fmt(row.get('batch_scheduling_speedup'))} | "
                f"{row.get('quality_gate_staged', '')} |"
            )
    else:
        lines.append(
            "No `speclink_cv_contribution_ablation_k12_bs16_20260528` "
            "contribution summary was found under `temp/`."
        )
    lines.extend(
        [
            "",
            "The ablation confirms the current performance priority: skipping "
            "the verified suffix is necessary but not sufficient. Staged DLM "
            "suffix saving gives an additional gain over non-staged CV, while "
            "singleton live verification is much slower than batched CV even "
            "before considering tail drain.",
        ]
    )
    return lines


def summarize_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunked = [
        row
        for row in rows
        if str(row.get("method", "")).startswith("speclink_cv_chunked")
    ]
    exactsafe = [row for row in rows if row.get("method") == "speclink_cv_exactsafe"]
    failed_chunked = [
        row
        for row in chunked
        if safe_float(row.get("exact_match_vs_eagle3")) != 1.0
    ]
    passed_chunked = [
        row
        for row in chunked
        if safe_float(row.get("exact_match_vs_eagle3")) == 1.0
    ]
    failed_exactsafe = [
        row
        for row in exactsafe
        if safe_float(row.get("exact_match_vs_eagle3")) != 1.0
    ]
    return {
        "chunked_rows": len(chunked),
        "chunked_passed": len(passed_chunked),
        "chunked_failed": len(failed_chunked),
        "exactsafe_rows": len(exactsafe),
        "exactsafe_failed": len(failed_exactsafe),
        "models": sorted({str(row.get("model")) for row in rows if row.get("model")}),
        "datasets": sorted(
            {str(row.get("dataset")) for row in rows if row.get("dataset")}
        ),
        "ks": sorted({str(row.get("K")) for row in rows if row.get("K")}),
        "batch_sizes": sorted(
            {str(row.get("batch_size")) for row in rows if row.get("batch_size")}
        ),
        "failed_chunked_rows": failed_chunked,
    }


def write_patch_snapshot(root: Path) -> None:
    paths = [
        "AGENTS.md",
        "TODO.md",
        "tools",
        "vllm/vllm/speclink_cv.py",
        "vllm/vllm/speclink_confidence_trace.py",
        "vllm/vllm/v1/core/sched/output.py",
        "vllm/vllm/v1/core/sched/scheduler.py",
        "vllm/vllm/v1/outputs.py",
        "vllm/vllm/v1/request.py",
        "vllm/vllm/v1/sample/rejection_sampler.py",
        "vllm/vllm/v1/sample/sampler.py",
        "vllm/vllm/v1/spec_decode/llm_base_proposer.py",
        "vllm/vllm/v1/worker/gpu_model_runner.py",
        "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py",
        "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_full.sh",
    ]
    tracked_diff = subprocess.run(
        ["git", "diff", "--"] + paths,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--"] + paths,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    chunks = [tracked_diff.stdout]
    for relative in sorted(line for line in untracked.stdout.splitlines() if line):
        file_diff = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", relative],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        chunks.append(file_diff.stdout)
    patch_dir = root / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    full_diff = "\n".join(chunk for chunk in chunks if chunk)
    (patch_dir / "vllm_speclink_cv.diff").write_text(
        full_diff, encoding="utf-8"
    )
    # Keep the older name for compatibility with earlier notes.
    (patch_dir / "speclink_cv.diff").write_text(full_diff, encoding="utf-8")


def write_report(
    root: Path,
    combined_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> None:
    summary = summarize_gate(token_rows)
    report = root / "09_reports" / "SPECLINK_CV_REPORT.md"
    failed = summary["failed_chunked_rows"]
    calibration_metrics = read_json(
        root / "03_confidence_calibration/eval/calibration_metrics.json"
    )
    calibration_trace_rows = read_csv(
        root / "03_confidence_calibration/trace_collection/trace_manifest.csv"
    )
    verify_cost_rows = read_csv(
        root / "07_roofline_packing/verify_cost_proxy/verify_cost_lookup.csv"
    )
    math_quality_followup_rows = latest_math_quality_rows()
    contribution_followup_rows = read_csv(
        LATEST_CONTRIBUTION_ROOT / "09_reports/contribution_ablation.csv"
    )
    lines = [
        "# SPECLINK_CV_REPORT",
        "",
        "## Current Verdict",
        "",
        "SpecLink-CV live h<K chunked verification is implemented and tested. "
        "Historical non-batch-invariant h<K probes are not correctness-stable "
        "in the current vLLM/EAGLE3 path. `VLLM_BATCH_INVARIANT=1` makes the "
        "archived batch-size-8 plain chunked matrix pass, but it does not "
        "solve the full TODO correctness gate: the bs=16/32 extension passes "
        "only 8/16 rows. The newer grouped batch-wide prefix-reject fallback "
        "improves the bs=16/32 diagnostic to 13/16 rows, but it still fails "
        "Llama3.1/math in three cases, so it is not a general correctness "
        "fix. The latest 64-token Qwen3/MTBench debug trace attributes the "
        "first token mismatch to a `prefix_rejected_skip_suffix` commit, and "
        "the paired prefix-reject full-K confirmation gate still fails 7/8, "
        "so reject-only confirmation is not enough. Stronger confirm-all and "
        "confirm-all plus global-batch-barrier probes also fail 7/8 at the "
        "same token, and forcing 8 dense TLM realignment steps after every "
        "prefix reject also fails 7/8. A follow-up grouped full-K confirmation "
        "barrier that schedules all forced confirmations together still fails "
        "7/8; its token timeline shows the same full-K draft IDs as the "
        "baseline but a different first target argmax. The paired "
        "`active_batch_drift.*` diagnostic also shows the active request "
        "ordinals are the same `[0,1,2,3,4,5,6,7]`; the flip is a low-margin "
        "verifier result (`2213` beats `2033` by 0.125 in baseline, while CV "
        "ties both at 31.625 and picks `2033`). The follow-up context-debug "
        "gate in `70_qwen_mtbench_k8_bs8_t64_context_debug_gate/` reproduces "
        "the same 7/8 failure and confirms the committed context is identical "
        "at the mismatch (`num_computed_tokens_cpu=86`, `num_tokens_no_spec=87`, "
        "`worker_output_token_count=31`, and the final 16 context token IDs "
        "match exactly). The same diagnostic now also records that the physical "
        "KV block tails differ between the two independent runs, so the "
        "remaining suspect is accumulated verifier/KV layout or numerical "
        "state drift rather than a missing confirmation branch, isolated "
        "confirmation scheduling, coarse active-set mismatch, context-token "
        "mismatch, or immediate drafter reuse.",
        "",
        "Supplemental current-code probes from 2026-05-27 are included in "
        "the token-id summary when the local `temp/` artifacts are present. "
        "They show that reserving the full one-shot slot range plus a "
        "draft-aware batch-invariant greedy tie policy moves the "
        "Qwen3/MTBench/K=8/bs=8/64-token confirm-all-barrier mismatch from "
        "token 31 to token 48, but still matches only 7/8; the isolated "
        "confirm-all variant also matches only 7/8. The paired exact-safe "
        "fallback passes 8/8. A follow-up `chunked_recompute` diagnostic "
        "rolls back the computed-token cursor after committed prefix chunks "
        "so the next TLM step recomputes the committed tokens and overwrites "
        "h<K KV state. It still fails the same Qwen3/MTBench/K=8/bs=8/64-token "
        "gate: 3/8 without the global barrier and 6/8 with the barrier, both "
        "with the first request diverging at token 31. The barrier timeline "
        "attributes the first mismatch to a dense target step, which means "
        "recomputing committed KV alone is not enough to emulate the EAGLE3 "
        "full-K verifier shape. A follow-up batched dense-realign probe also "
        "forces worker input-batch row order back to request ordinal order; it "
        "still matches only 5/8, and its active-batch diagnostic shows equal "
        "active ordinals, equal committed context tail, and different physical "
        "KV block tails at the same token-31 flip. A no-KV-write prefix probe "
        "suppresses KV writes for the h<K prefix verifier forward, but it "
        "fails earlier: 0/8 with the first mismatch at token 1. Its active "
        "drift diagnostic shows the baseline first verifier row is isolated "
        "while CV still verifies a full active batch, so suppressing prefix KV "
        "writes without a full equivalent replay is not a recovery. These "
        "probes reinforce the current verdict: "
        "true live h<K chunking remains unsafe, while one-shot fallback is "
        "the only correctness-preserving default.",
        "",
        "The 2026-05-28 sampler tie-breaking follow-up tightened "
        "`greedy_sample_with_preferred_tokens`: under "
        "`VLLM_BATCH_INVARIANT=1`, `SPECLINK_CV_DRAFT_ACCEPT_EPS<=0` now "
        "returns the stable greedy token instead of accepting an exact-tied "
        "draft token. This fixes the short Qwen3/MTBench/K=8/bs=16/max12 "
        "argmax failure in "
        "`speclink_cv_qwen_mtbench_k8_bs16_t12_tiefix_20260528/`, but the "
        "original max32 setting still fails 13/16 in "
        "`speclink_cv_qwen_mtbench_k8_bs16_t32_tiefix_20260528/`. The bounded "
        "row-0 debug in "
        "`speclink_cv_qwen_mtbench_k8_bs16_t32_row0_debug_20260528/` shows "
        "another low-margin verifier flip at token 31: baseline emits "
        "`2213`, while CV emits `2033`. The logical context tail matches, but "
        "the physical block tails diverge early (`[2,3,4,5]` versus "
        "`[2,3,4,98]` after the first decode step). The no-barrier rerun "
        "also fails at token 31, while the paired exact-safe guard passes. "
        "The bounded history-KV checksum rerun in "
        "`speclink_cv_qwen_mtbench_k8_bs16_t32_historykv_20260528/` compares "
        "the eight logical positions before the failing verifier target "
        "position (`78..85`, layer 0) and finds identical K/V checksums even "
        "though the active request set and physical slots differ. "
        "Therefore the sampler tie fix is necessary but not sufficient; the "
        "remaining problem is narrower than simple historical KV content "
        "corruption: active-batch shape, current slot layout, or accumulated "
        "numerical drift remain in scope for live h<K chunking.",
        "",
        "The default runtime therefore uses the exact-safe guard: "
        "`SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=0` falls back to full-K "
        "one-shot verification. That path is exact in the recorded gates, but "
        "it is not a chunked-verification speedup. To test the current exact "
        "h<K path, run with `SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=1` and "
        "`VLLM_BATCH_INVARIANT=1`. No matrix-level end-to-end speedup claim is "
        "valid until the GuideLLM baselines and ablations are rerun under that "
        "same exact mode.",
        "",
        "## Scope",
        "",
        f"- output root: `{root}`",
        f"- token-id gate rows: {len(token_rows)}",
        f"- GuideLLM pilot rows: {sum(1 for row in combined_rows if row.get('measurement_type') == 'guidellm_end_to_end')}",
        f"- follow-up diagnostic rows: {sum(1 for row in combined_rows if row.get('measurement_type') in {'steady_state_throughput', 'contribution_ablation'})}",
        f"- models covered by token gates: `{', '.join(summary['models'])}`",
        f"- datasets covered by token gates: `{', '.join(summary['datasets'])}`",
        f"- K covered by token gates: `{', '.join(summary['ks'])}`",
        f"- batch sizes covered by token gates: `{', '.join(summary['batch_sizes'])}`",
        "",
        "## Token-Id Correctness Gate",
        "",
        f"- live h<K rows: {summary['chunked_rows']}",
        f"- live h<K passed: {summary['chunked_passed']}",
        f"- live h<K failed: {summary['chunked_failed']}",
        f"- exact-safe rows: {summary['exactsafe_rows']}",
        f"- exact-safe failed: {summary['exactsafe_failed']}",
        "",
        "| model | dataset | K | batch | mode | matched | status | first mismatch token | evidence |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in token_rows:
        mode = str(row.get("method", "")).replace("speclink_cv_", "")
        matched = f"{row.get('matched_count')}/{row.get('total_count')}"
        lines.append(
            f"| {row.get('model')} | {row.get('dataset')} | {row.get('K')} | "
            f"{row.get('batch_size')} | {mode} | {matched} | {row.get('status')} | "
            f"{row.get('first_mismatch_token_index', '')} | `{row.get('raw_summary', row.get('evidence_path', ''))}` |"
        )
    lines.extend(
        [
            "",
            "## GuideLLM Pilot",
            "",
            "The recorded GuideLLM pilot covers Qwen3/math/K=8/bs=8 with pure "
            "vLLM, EAGLE3 one-shot, and the 2 x 2 x 2 CV ablations. Those "
            "rows are useful as raw serving evidence, but every live-chunked "
            "CV row is either `invalid_correctness_mismatch` or "
            "`invalid_no_live_chunking`. They must not be reported as valid "
            "SpecLink-CV speedups.",
            "",
            "The newer batch-invariant serving smoke in "
            "`55_batch_invariant_guidellm_smoke/` covers Qwen3/math/K=8/bs=1 "
            "with EAGLE3 one-shot and `cv_half_async_roofline`, using "
            "`VLLM_BATCH_INVARIANT=1` and true h<K chunking. It starts vLLM, "
            "runs GuideLLM, emits live `verify_chunk_*` profile events, and "
            "text-matches EAGLE3 for the two sampled requests. It is a "
            "serving-path smoke only: the CV row is slower than EAGLE3 "
            "one-shot and the full batch-invariant ablation matrix is still "
            "not run.",
            "",
            "The Qwen/math/K=8/bs=8 batch-invariant GuideLLM pilot in "
            "`56_batch_invariant_guidellm_qwen_math_k8_bs8_pilot/` runs pure "
            "vLLM, EAGLE3 one-shot, and all 8 CV ablations for 8 requests and "
            "32 output tokens. Every run completes, but every CV row is slower "
            "than EAGLE3. The best text-exact CV row is "
            "`cv_conf_async_simple` at about 0.253x EAGLE3 throughput. Several "
            "other CV rows have GuideLLM text exact-match below 1.0; the "
            "paired prompt-subset token gate in "
            "`57_batch_invariant_guidellm_prompt_gate_t32/` passes 8/8, so "
            "those text mismatches should be treated as conservative serving "
            "evidence rather than the strict token-id result.",
            "",
            "The broader bs=16/32 batch-invariant matrix in "
            "`58_batch_invariant_bs16_bs32_matrix/` is negative: only 8/16 "
            "rows pass. Every Qwen bs=32 row fails, Qwen/MTBench/K=8/bs=16 "
            "fails, and Llama/math fails for K=8 at both bs=16 and bs=32 and "
            "for K=12 at bs=32. This means batch-invariant execution is not a "
            "general correctness fix for the requested batch-size sweep.",
            "",
            "The grouped batch-wide fallback rerun in "
            "`59_grouped_batchwide_fallback_bs16_bs32_matrix/` requeues every "
            "row in a rejecting prefix batch for grouped full-K confirmation. "
            "It passes all Qwen rows and all Llama/MTBench rows at bs=16/32, "
            "but still fails Llama/math/K=8 at bs=16 and bs=32 and "
            "Llama/math/K=12 at bs=16. This narrows the remaining failure "
            "surface, but it also confirms that grouped fallback is not a "
            "universal exactness recovery.",
            "",
            "The Qwen/MTBench/K=8/bs=8 batch-invariant GuideLLM ablation in "
            "`62_qwen_mtbench_k8_bs8_guidellm_batchinv_ablation/` runs pure "
            "vLLM, EAGLE3 one-shot, and all 8 CV ablations for 24 requests and "
            "64 output tokens. All runs complete, but every CV row is slower "
            "than EAGLE3 and has GuideLLM text exact-match below 1.0. The "
            "paired strict token-id gate in "
            "`60_qwen_mtbench_k8_bs8_t64_batchinv_gate/` confirms this is not "
            "just a text-comparison artifact: live h<K matches only 21/24 "
            "prompts, with the first mismatch at token 31. The exact-safe "
            "fallback in `61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/` passes "
            "24/24. The grouped batch-wide prefix-reject fallback in "
            "`63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/` completes "
            "after the batch-fill barrier fix but still only matches 22/24. "
            "Therefore the shorter `54` bs=8 token gate does not generalize "
            "to longer 64-token MTBench generation.",
            *render_math_quality_followup(
                math_quality_followup_rows,
                contribution_followup_rows,
            ),
            "",
            "## Verifier Shape Drift Diagnosis",
            "",
            "The strongest current failure evidence is the top-k debug trace in "
            "`22_verifier_shape_drift_topk/`. It compares h<K chunked prefix "
            "verifier events against full-K EAGLE3 one-shot verifier events for "
            "the same draft token sequence. In the Qwen3/math/K=8/bs=8 debug "
            "smoke, 26 prefix-accepted boundary events had a matching one-shot "
            "full draft; 25 matched the boundary argmax and 1 differed. The "
            "enhanced analyzer also compared the 104 prefix target positions "
            "inside those matched chunks and found zero prefix argmax "
            "mismatches, localizing the observed drift to the first boundary "
            "position after the accepted prefix.",
            "",
            "The mismatching apple-orchard event used h=4 and draft prefix "
            "`[198, 32313, 11, 1077]`. The chunked prefix bonus argmax was "
            "`594`, while the one-shot boundary argmax was `752`. Their top-k "
            "sets overlap: chunked top-k starts `[594, 752, ...]`, one-shot "
            "top-k starts `[752, 594, ...]`, with nearly tied logits. This "
            "supports the current diagnosis that live h<K changes target "
            "verifier shape enough to flip greedy token identity in low-margin "
            "boundary cases. It is not an async queue bookkeeping-only issue.",
            "",
            "A later batched-prefix probe fixed a real worker-side hygiene bug: "
            "when the drafter is intentionally skipped for SpecLink-CV "
            "realignment, vLLM must return empty draft lists rather than "
            "`[0] * K` token drafts. The zero-draft pollution is gone in "
            "`23_batched_prefix_emptydraft_probe/`, but the same Qwen3/math/K=8/"
            "bs=8 h<K gate still matches only 6/8 prompts. That fix is "
            "necessary but not sufficient for exact live h<K correctness.",
            "",
            "The confirm-all probes in `24_confirm_all_isolated_probe/` and "
            "`25_confirm_all_batched_probe/` tried to recover exactness by "
            "discarding both accepted and rejected prefix outputs, then "
            "requeueing the original full-K draft for one-shot confirmation. "
            "Both probes improved to 7/8 but still failed. The isolated probe "
            "kept the apple-orchard mismatch because the confirmation ran as a "
            "different batch row. The batched probe fixed apple-orchard but "
            "still failed Hallie after later step grouping changed the "
            "target/drafter state. This shows that next-step full-K "
            "confirmation is not equivalent to confirming inside the original "
            "one-shot verifier step.",
            "",
            "`analyze_prefix_step_equivalence.py` makes this diagnosis "
            "step-local. In the isolated confirm-all run, 39/49 prefix steps "
            "had a matching baseline full draft, but 3 of those later full-K "
            "confirmations still had target-argmax differences. In the batched "
            "confirm-all run, only 6/60 prefix steps had a matching baseline "
            "full draft; 54 were already on draft sequences absent from the "
            "one-shot baseline. This separates numerical verifier-shape drift "
            "from later target/drafter state drift.",
            "",
            "The low-margin fallback probes in `26_low_margin_probe/`, "
            "`27_low_margin_reject_confirm_probe/`, and "
            "`28_low_margin_reject_confirm_isolated_probe/` added a verifier "
            "top-1/top-2 margin guard: when a prefix target or bonus row had "
            "margin <= 0.5, the worker discarded the prefix output and "
            "requeued the original full-K draft. This did not restore "
            "correctness. Guard-only still failed apple-orchard at token 5; "
            "guard plus reject confirmation failed Hallie under batched "
            "prefix; isolated guard plus reject confirmation failed both "
            "Hallie and apple-orchard. This rules out a simple confidence/"
            "margin-only fix for exact greedy equivalence.",
            "",
            "The batch-shape probes in `29_bs1_shape_probe/`, "
            "`30_bs1_confirm_all_probe/`, and "
            "`31_barrier_confirm_all_probe/` first showed that batch-size 1 "
            "plain h<K still fails, batch-size 1 confirm-all passes, and the "
            "original batch-size 8 global barrier confirm-all still failed. "
            "The follow-up `32_barrier_batchfill_confirm_all_probe/` fixed a "
            "real scheduler bug: the barrier now waits for the waiting queue "
            "to drain before dispatching prefix chunks, and confirm-all passes "
            "8/8. This is a diagnostic exactness recovery, not a speedup path. "
            "`33_barrier_batchfill_plain_probe/` still fails 6/8, "
            "`34_barrier_batchfill_margin_probe/` fails 5/8, accept-only and "
            "reject-only confirmation both fail, and forced h=1 still fails "
            "5/8. The dense-realign diagnostics in "
            "`38_barrier_batchfill_dense0_probe/` and "
            "`39_barrier_batchfill_dense1_probe/` also fail 5/8. Disabling "
            "dense realignment removes the dense guard from the trace but not "
            "the token drift; using one dense realign step changes which "
            "prompts fail, but still does not restore exactness. The prefix-"
            "reject realignment probes in "
            "`40_barrier_batchfill_prefixreject_dense1_probe/` and "
            "`41_barrier_batchfill_prefixreject_dense8_probe/` both fail 4/8, "
            "so suppressing the immediate drafter after prefix rejection is "
            "also not a standalone fix.",
            "",
            "The later margin-threshold audit in "
            "`42_prefix_margin_guard_33_probe/` shows why threshold tuning was "
            "tempting but insufficient: on the plain batch-fill trace, "
            "threshold 0.5 only covers one of the two failed request ordinals, "
            "while threshold 1.5 retrospectively covers both but would already "
            "fallback 60% of prefix chunks. Live threshold 1.5 probes still "
            "fail. `43_barrier_batchfill_margin15_probe/` improves to 7/8 "
            "but fails apple-orchard at token 5. Batch-wide low-margin "
            "fallback also fails 7/8 in `44_barrier_batchwide_margin15_probe/`; "
            "the refined candidate/mask path in "
            "`45_barrier_batchwide_margin15_refined_probe/` and the scheduler "
            "path that reuses prefix accept/reject confirmation in "
            "`46_barrier_batchwide_margin15_reuses_confirm_probe/` still fail "
            "7/8. Therefore verifier margin can identify risky chunks, but "
            "it is not a correctness fix for live h<K chunking.",
            "",
            "Forced larger-prefix probes also failed. "
            "`47_barrier_batchfill_h5_probe/` still matches only 6/8, while "
            "`48_barrier_batchfill_h6_probe/` and "
            "`49_barrier_batchfill_h7_probe/` match only 5/8. Their "
            "`prefix_step_equivalence.*` reports are stronger than the h=4 "
            "boundary case: h=6 and h=7 each have 5 prefix target argmax "
            "mismatches among matched baseline full drafts. That means the "
            "shape-sensitive drift is not limited to the prefix bonus boundary; "
            "larger h can flip target rows inside the prefix itself. The true "
            "h<K shortcut is still not exact.",
            "",
            "The backend diagnostic in "
            "`50_barrier_batchfill_triton_k8_probe/` reran the Qwen3/math K=8 "
            "batch-fill barrier path with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. "
            "It still matches only 5/8 prompts, with Hallie, farmer, and "
            "apple-orchard diverging at token indexes 13, 5, and 5. The "
            "`live_correctness_smoke.py` path already uses `enforce_eager=True`, "
            "so this is not explained by CUDA graph replay alone, and switching "
            "from the default attention backend to Triton attention does not "
            "remove the live h<K drift.",
            "",
            "The latest grouped-confirmation diagnostic in "
            "`51_grouped_batchwide_prefixreject_probe/` fixes the specific "
            "K=8 Qwen3/math batch-fill apple-orchard failure. When any request "
            "in a prefix batch rejects, the scheduler now requeues every row "
            "from that prefix batch for full-K confirmation and schedules those "
            "confirmations together, instead of isolating them one by one as "
            "row 0. This probe matches 8/8 prompts. Its "
            "`prefix_step_equivalence.*` files still show residual shape "
            "sensitivity in comparable prefix/full-confirmation logits, so this "
            "is a repaired smoke gate for the batch-wide fallback path, not yet "
            "a full proof across K, model, dataset, and batch-size settings.",
            "",
            "The broader grouped-fallback rerun in "
            "`52_grouped_k8_k12_cross_model_data_bs8/` covers Qwen3-8B and "
            "Llama3.1-8B on math and MTBench for K=8 and K=12 at batch size "
            "8. It passes 7/8 rows but fails Llama3.1/math/K=8 with 7/8 "
            "matching prompts and the first mismatch at token index 9. The "
            "failing request says `we first need` in the EAGLE3 one-shot "
            "baseline but `we need to first` in the grouped fallback path. "
            "The prefix-step equivalence trace localizes one prefix decision "
            "mismatch and one prefix target-argmax mismatch for request 0's "
            "second prefix. For draft prefix `[5944, 11, 584, 1205]`, the "
            "CV prefix probe accepts all 4 prefix tokens while the matching "
            "baseline one-shot verifier accepts only 3. This keeps live h<K "
            "chunking in the negative-correctness category.",
            "",
            "The worker-ordered low-margin diagnostic in "
            "`53_worker_ordered_lowmargin_llama_math_k8/` narrows the failure "
            "further. The worker now reorders the persistent input batch by "
            "request ordinal under the global barrier, so request 0 moves from "
            "row 7 to row 0 before verification. That fixes the row-binding "
            "symptom, but the full-replay probe still matches only 7/8 prompts "
            "and reports one prefix decision mismatch plus one full-confirmation "
            "target mismatch. The paired exact-safe run with confidence sizing "
            "and `SPECLINK_CV_MIN_BENEFIT=999` forces one-shot fallback and "
            "matches 8/8. Therefore row alignment is necessary but not "
            "sufficient; the remaining issue is the h<K verifier shape itself.",
            "",
            "The batch-invariant matrix in "
            "`54_batch_invariant_chunked_matrix/` is the first positive live "
            "h<K token-id gate. With `VLLM_BATCH_INVARIANT=1`, plain "
            "`mode=chunked` plus batched-prefix verification passes all 8 rows "
            "for Qwen3-8B and Llama3.1-8B on math and MTBench at K=8 and K=12, "
            "batch size 8, with 8/8 prompts matched in every row. This makes "
            "batch-invariant execution a useful exactness condition for the "
            "current bs=8 SpecLink-CV path. It is not a broad proof.",
            "",
            "The bs=16/32 extension in "
            "`58_batch_invariant_bs16_bs32_matrix/` shows the limit of that "
            "condition. It passes 8/16 rows and fails 8/16. Batch size 32 is "
            "especially weak: every Qwen bs=32 row fails, and Llama/math "
            "fails at bs=32 for both K values. Therefore the full TODO "
            "batch-size correctness gate is still failed.",
            "",
            "`59_grouped_batchwide_fallback_bs16_bs32_matrix/` tests the best "
            "current recovery strategy under the same bs=16/32 pressure: when "
            "any row in a prefix verifier batch rejects, every row in that "
            "prefix batch is requeued for grouped full-K confirmation. This "
            "passes 13/16 rows, including all Qwen rows, but the Llama/math "
            "failures remain. The grouped fallback is therefore a useful "
            "diagnostic and a partial recovery path, not a completed "
            "SpecLink-CV correctness solution.",
            "",
            "Longer generation length is also a correctness stressor. "
            "`60_qwen_mtbench_k8_bs8_t64_batchinv_gate/` uses "
            "`VLLM_BATCH_INVARIANT=1`, Qwen3/MTBench/K=8/bs=8, 24 prompts, "
            "and 64 output tokens. Plain h<K chunking fails 21/24 with the "
            "first mismatch at token 31. The paired exact-safe fallback in "
            "`61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/` passes 24/24. This "
            "keeps the bs=8 batch-invariant result scoped to the shorter "
            "smoke lengths where it was measured. The paired grouped fallback "
            "gate in `63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/` also "
            "fails 22/24, so grouped full-K confirmation after prefix rejects "
            "does not close the long-output MTBench case.",
            "",
            "`64_qwen_mtbench_k8_bs8_t64_debug_plain/` contains the matching "
            "debug run and `token_timeline.*` attribution. For request 0, the "
            "first mismatch is token index 31: EAGLE3 one-shot emits token "
            "`2213`, while SpecLink-CV emits `2033`. The attributing CV "
            "segment is `prefix_rejected_skip_suffix` at event index 286, "
            "whose prefix verifier target argmax starts `[2033, 13, 1084, "
            "752]`; the baseline segment covering the same token index is a "
            "full-K `spec_step_output` whose verifier target argmax starts "
            "`[2213, 13, 576, 4024, 1736, 4658, ...]`. "
            "`65_qwen_mtbench_k8_bs8_t64_reject_confirm_gate/` then enables "
            "prefix-reject full-K confirmation for the same 8-prompt setting "
            "and still matches only 7/8 with first mismatch token 31. This "
            "rules out a single unconfirmed reject shortcut. The stronger "
            "`66_qwen_mtbench_k8_bs8_t64_confirm_all_gate/` and "
            "`67_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_gate/` probes "
            "discard both accepted and rejected prefix outputs and requeue "
            "full-K confirmation; both still match only 7/8 with first "
            "mismatch token 31. The barrier variant also requires all running "
            "requests to dispatch prefix chunks together. These results narrow "
            "the long-output failure to accumulated verifier/KV shape state "
            "drift rather than a missing confirmation branch. "
            "`68_qwen_mtbench_k8_bs8_t64_prefixreject_dense8_gate/` then "
            "forces 8 dense TLM realignment steps after every prefix reject; "
            "it still matches only 7/8 with first mismatch token 31, so "
            "immediate drafter reuse after prefix rejection is not sufficient "
            "to explain the failure. "
            "`69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate/` "
            "then groups every forced full-K confirmation under the global "
            "batch barrier, not only batch-wide prefix-reject confirmations. "
            "It still matches only 7/8 at token 31. Its `token_timeline.*` "
            "moves the attributing CV segment from `prefix_rejected_skip_suffix` "
            "to a full-K `spec_step_output`: CV and baseline verify the same "
            "scheduled draft IDs `[2213, 13, 576, 4024, 1736, 1030, 13, 576]`, "
            "but the first target argmax differs (`2033` in CV versus `2213` "
            "in baseline). The new `active_batch_drift.*` diagnostic confirms "
            "both verifier batches have active request ordinals "
            "`[0,1,2,3,4,5,6,7]`; baseline has logits `2213=31.75`, "
            "`2033=31.625`, while CV has `2213=31.625`, `2033=31.625`. "
            "This rules out isolated full-K confirmation scheduling and "
            "coarse active-set mismatch as the remaining explanation. "
            "`70_qwen_mtbench_k8_bs8_t64_context_debug_gate/` reruns the same "
            "8-prompt gate after extending `verifier_step_debug`; it again "
            "matches only 7/8 and the first mismatch is still token 31. Its "
            "`active_batch_drift.*` shows `num_computed_tokens_cpu=86`, "
            "`num_tokens_no_spec=87`, `worker_output_token_count=31`, and the "
            "same final 16 context token IDs on both paths, so the remaining "
            "drift is not explained by committed context length or token-tail "
            "mismatch either. The physical `block_ids_tail` values differ, "
            "which keeps KV layout or accumulated numerical state in scope.",
            "",
            "## TODO Questions",
            "",
            "1. pure vLLM vs EAGLE3 vs best SpecLink-CV: the pilot has pure "
            "vLLM and EAGLE3 metrics, and the batch-invariant serving smoke "
            "plus the Qwen/math/K=8/bs=8 pilot have valid h<K CV rows. There "
            "is still no full-matrix best SpecLink-CV row because only one "
            "batch-invariant ablation scenario has been rerun.",
            "2. K=8 vs K=12: both K=8 and K=12 show h<K failures on Qwen3/math; "
            "K=12 did not remove the shape-sensitive mismatch.",
            "3. batch size 8/16/32: Qwen3/math failed h<K at bs=8,16,32 for "
            "both K=8 and K=12 without batch-invariant execution. With "
            "`VLLM_BATCH_INVARIANT=1`, bs=8 passes but the bs=16/32 extension "
            "still fails 8/16 rows. The grouped batch-wide fallback improves "
            "the bs=16/32 matrix to 13/16, but Llama/math still fails. A "
            "longer Qwen3/MTBench/K=8/bs=8 64-token gate also fails 21/24 "
            "under batch-invariant h<K; grouped fallback improves this only "
            "to 22/24, while exact-safe passes 24/24. The debug 8-prompt "
            "rerun fails 7/8 and localizes request 0's first mismatch to a "
            "`prefix_rejected_skip_suffix` commit at token 31; reject-only "
            "full-K confirmation, confirm-all, and barrier confirm-all all "
            "still fail 7/8. Prefix-reject dense8 realignment and grouped "
            "full-K confirmation barrier also fail 7/8.",
            "4. DLM confidence predicts acceptance: confidence trace tooling "
            "exists and the trace calibration artifact has "
            f"{calibration_metrics.get('rows', 'unknown')} odd-split rows, "
            f"ECE `{calibration_metrics.get('ece', 'unknown')}`, and Brier "
            f"`{calibration_metrics.get('brier', 'unknown')}`. Confidence "
            "cannot rescue correctness because it only chooses h and never "
            "changes TLM accept/reject semantics.",
            "5. fixed half vs confidence-guided chunk: no valid performance "
            "winner can be selected under the correctness gate.",
            "6. sync vs async: async queue hooks and profile events exist, but "
            "async live h<K rows are invalid while correctness is not exact.",
            "7. roofline packing: roofline fallback/profile hooks exist; no "
            "valid speedup can be attributed to packing yet. The archived "
            f"verify-cost proxy has {len(verify_cost_rows)} trace-derived "
            "lookup rows and is explicitly not a hardware timing profile.",
            "8. source of benefit: skipped suffix tokens are recorded in "
            "invalid h<K runs, but those savings cannot be claimed as benefits.",
            "9. extra TLM forward/repeated KV: diagnostic replay and confirm "
            "probes show extra forwards do not restore exactness.",
            "10. correctness: not fully preserved for live h<K chunking "
            "without batch-invariant execution. The grouped batch-wide "
            "prefix-reject fallback passes the Qwen3/math K=8 bs=8 smoke in "
            "`51_grouped_batchwide_prefixreject_probe/`, but the broader "
            "`52_grouped_k8_k12_cross_model_data_bs8/` matrix still fails "
            "Llama3.1/math/K=8. With `VLLM_BATCH_INVARIANT=1`, the newer "
            "`54_batch_invariant_chunked_matrix/` plain h<K matrix passes "
            "8/8 rows across Qwen3/Llama, math/MTBench, and K=8/K=12 at "
            "batch size 8, but `58_batch_invariant_bs16_bs32_matrix/` fails "
            "8/16 rows at batch sizes 16 and 32. The later grouped fallback "
            "bs=16/32 matrix passes 13/16 rows, with the remaining failures "
            "all on Llama/math.",
            " The 64-token Qwen3/MTBench debug follow-up shows the first "
            "long-output mismatch is produced by a prefix-reject skip-suffix "
            "commit, and prefix-reject full-K confirmation alone still fails "
            "7/8. Confirm-all, barrier confirm-all, prefix-reject dense8, "
            "and grouped full-K confirmation barrier also fail 7/8.",
            "11. global best: none. The batch-invariant GuideLLM smoke/pilot "
            "rows that pass the local speedup gate are slower than EAGLE3 "
            "one-shot and cover only Qwen3/math/K=8.",
            "12. unsuitable scenarios: current Qwen3/math and Llama3.1 "
            "math token gates show live h<K chunking is unsafe; "
            "Qwen3/MTBench K=8/K=12 and Llama/math K=12 passed small gates, "
            "but those passes are not enough to override failures in other "
            "same-model/same-K scenarios.",
            "13. vLLM limits: verifier shape, KV/cache rollback, scheduler "
            "integration, and eager/backend numerical shape sensitivity remain "
            "open issues. The archived Triton-attention probe still fails, so "
            "backend switching is not a standalone fix.",
            "",
            "## Artifacts",
            "",
            "- `09_reports/summary_metrics.csv` and `.json`: combined GuideLLM "
            "pilot plus token-id correctness rows.",
            "- `09_reports/token_id_correctness_summary.csv` and `.json`: "
            "strict token-id correctness evidence.",
            "- `09_reports/CORRECTNESS_AUDIT.md`: detailed chronological audit.",
            "- `09_reports/TODO_REQUIREMENT_AUDIT.md`: requirement-level "
            "completion audit against `TODO.md`.",
            "- `patches/vllm_speclink_cv.diff`: current implementation diff.",
            "- `08_figures/*.csv`: source data for generated figures.",
            "- `03_confidence_calibration/`: trace manifest, binning "
            "calibration model, odd-split ECE/Brier metrics, and reliability "
            f"diagram from {len(calibration_trace_rows)} trace files.",
            "- `07_roofline_packing/verify_cost_proxy/`: trace-derived "
            "verify-cost lookup used only as a roofline-packing code-path "
            "proxy, not as end-to-end timing evidence.",
            "- `22_verifier_shape_drift_topk/`: top-k boundary diagnosis for "
            "the h<K verifier shape drift.",
            "- `23_batched_prefix_emptydraft_probe/`: batched-prefix diagnostic "
            "after removing zero-draft pollution from drafter-skip steps.",
            "- `24_confirm_all_isolated_probe/` and "
            "`25_confirm_all_batched_probe/`: full-K confirmation diagnostics "
            "for both accepted and rejected prefix chunks, including "
            "`prefix_step_equivalence.*` step-local analysis.",
            "- `26_low_margin_probe/`, `27_low_margin_reject_confirm_probe/`, "
            "and `28_low_margin_reject_confirm_isolated_probe/`: verifier "
            "low-margin fallback diagnostics with `prefix_step_equivalence.*`.",
            "- `29_bs1_shape_probe/`, `30_bs1_confirm_all_probe/`, and "
            "`31_barrier_confirm_all_probe/`: batch-shape diagnostics showing "
            "why the first coarse batch barrier was insufficient.",
            "- `32_barrier_batchfill_confirm_all_probe/` through "
            "`41_barrier_batchfill_prefixreject_dense8_probe/`: batch-fill barrier "
            "follow-ups showing confirm-all now passes, while plain h<K, "
            "low-margin guard, accept-only confirmation, reject-only "
            "confirmation, forced h=1, and dense-realign step diagnostics "
            "still fail.",
            "- `42_prefix_margin_guard_33_probe/` through "
            "`46_barrier_batchwide_margin15_reuses_confirm_probe/`: "
            "retrospective and live threshold 1.5 diagnostics showing that "
            "higher verifier-margin fallback and batch-wide fallback still "
            "do not satisfy strict token-id equivalence.",
            "- `47_barrier_batchfill_h5_probe/` through "
            "`49_barrier_batchfill_h7_probe/`: forced larger-prefix diagnostics "
            "showing that h=5/6/7 still fail, with h=6/7 exhibiting prefix "
            "target argmax mismatches inside matched full drafts.",
            "- `50_barrier_batchfill_triton_k8_probe/`: K=8 Triton-attention "
            "backend diagnostic showing that backend switching does not restore "
            "strict token-id equivalence.",
            "- `51_grouped_batchwide_prefixreject_probe/`: K=8 Qwen3/math "
            "batch-wide prefix-reject fallback with grouped full-K confirmations; "
            "the smoke matches 8/8 and includes `prefix_step_equivalence.*`.",
            "- `52_grouped_k8_k12_cross_model_data_bs8/`: grouped batch-wide "
            "prefix-reject fallback matrix for Qwen3/Llama, math/MTBench, "
            "K=8/K=12 at batch size 8; 7/8 rows pass and "
            "Llama3.1/math/K=8 fails 7/8.",
            "- `53_worker_ordered_lowmargin_llama_math_k8/`: worker row-order "
            "diagnostic for the remaining Llama3.1/math/K=8 failure; row "
            "alignment is fixed but h<K full replay still fails, while "
            "forced one-shot fallback passes 8/8.",
            "- `54_batch_invariant_chunked_matrix/`: positive plain h<K "
            "correctness matrix with `VLLM_BATCH_INVARIANT=1`; Qwen3/Llama, "
            "math/MTBench, K=8/K=12, batch size 8 all pass 8/8 prompts.",
            "- `55_batch_invariant_guidellm_smoke/`: minimal real GuideLLM "
            "serving smoke for the same batch-invariant h<K path; it validates "
            "startup, profiling, parsing, and text exact-match on Qwen3/math/"
            "K=8/bs=1, but it is not a full performance matrix.",
            "- `56_batch_invariant_guidellm_qwen_math_k8_bs8_pilot/`: "
            "Qwen3/math/K=8/bs=8 GuideLLM pilot with pure vLLM, EAGLE3, and "
            "all 8 CV ablations; all rows run, but CV is slower than EAGLE3.",
            "- `57_batch_invariant_guidellm_prompt_gate_t32/`: strict "
            "token-id gate over the exact prompt subset used by the "
            "Qwen3/math/K=8/bs=8 GuideLLM pilot; plain chunked passes 8/8 at "
            "32 output tokens.",
            "- `58_batch_invariant_bs16_bs32_matrix/`: bs=16/32 extension of "
            "the batch-invariant token-id gate; only 8/16 rows pass, so the "
            "full batch-size correctness requirement remains unsatisfied.",
            "- `59_grouped_batchwide_fallback_bs16_bs32_matrix/`: grouped "
            "batch-wide prefix-reject fallback at bs=16/32; 13/16 rows pass. "
            "All Qwen and Llama/MTBench rows pass, but Llama/math/K=8 "
            "fails at bs=16 and bs=32 and Llama/math/K=12 fails at bs=16.",
            "- `60_qwen_mtbench_k8_bs8_t64_batchinv_gate/`: longer "
            "Qwen3/MTBench/K=8/bs=8 token-id gate with "
            "`VLLM_BATCH_INVARIANT=1`; plain h<K chunking matches only 21/24 "
            "prompts at 64 output tokens.",
            "- `61_qwen_mtbench_k8_bs8_t64_exactsafe_gate/`: paired "
            "exact-safe fallback for the same Qwen3/MTBench/K=8/bs=8 "
            "64-token setting; passes 24/24.",
            "- `63_qwen_mtbench_k8_bs8_t64_grouped_fallback_gate/`: paired "
            "grouped batch-wide prefix-reject fallback for the same "
            "64-token setting; after the barrier wait fix it completes, but "
            "still matches only 22/24.",
            "- `64_qwen_mtbench_k8_bs8_t64_debug_plain/`: 8-prompt debug "
            "rerun for Qwen3/MTBench/K=8/bs=8 at 64 output tokens, with "
            "`token_timeline.*` attributing request 0's first mismatch at "
            "token 31 to `prefix_rejected_skip_suffix`.",
            "- `65_qwen_mtbench_k8_bs8_t64_reject_confirm_gate/`: paired "
            "prefix-reject full-K confirmation gate for the same 8-prompt "
            "setting; still matches only 7/8, first mismatch token 31.",
            "- `66_qwen_mtbench_k8_bs8_t64_confirm_all_gate/`: paired "
            "confirm-all gate for the same 8-prompt setting; accepted and "
            "rejected prefixes are both discarded and requeued for full-K "
            "confirmation, but the run still matches only 7/8.",
            "- `67_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_gate/`: "
            "confirm-all plus global-batch-barrier gate for the same setting; "
            "it still matches only 7/8, so coarse batch-step alignment does "
            "not recover the long-output failure.",
            "- `68_qwen_mtbench_k8_bs8_t64_prefixreject_dense8_gate/`: "
            "prefix-reject dense realignment diagnostic for the same setting; "
            "forcing 8 dense TLM steps after every prefix reject still matches "
            "only 7/8.",
            "- `69_qwen_mtbench_k8_bs8_t64_confirm_all_barrier_grouped_gate/`: "
            "grouped full-K confirmation barrier diagnostic for the same "
            "setting; it still matches only 7/8, and `token_timeline.*` plus "
            "`active_batch_drift.*` show the full-K confirmation verifies the "
            "same draft IDs and active request ordinals as baseline while the "
            "first target argmax flips from baseline token 2213 to CV token "
            "2033 on a near-tie.",
            "- `70_qwen_mtbench_k8_bs8_t64_context_debug_gate/`: same "
            "Qwen3/MTBench/K=8/bs=8/64-token 8-prompt diagnostic after adding "
            "context-length and context-tail fields to `verifier_step_debug`; "
            "it still matches only 7/8, and the mismatching row has identical "
            "computed length, no-spec length, output-token count, active "
            "request ordinals, scheduled draft IDs, and final 16 context "
            "token IDs.",
            "- Supplemental 2026-05-27 current-code probes under `temp/`: "
            "`speclink_cv_tie_prefer_draft_confirm_all_barrier_20260527/`, "
            "`speclink_cv_tie_prefer_draft_confirm_all_isolated_20260527/`, "
            "`speclink_cv_exactsafe_after_tie_20260527/`, "
            "`speclink_cv_tie_argmax_llama_math_k8_bs16_20260527/`, and "
            "`speclink_cv_recovery_qwen_math_k8_bs32_20260527/`, plus "
            "`speclink_cv_recompute_qwen_mtbench_k8_bs8_t64_20260527/` and "
            "`speclink_cv_recompute_barrier_qwen_mtbench_k8_bs8_t64_20260527/`, "
            "and `speclink_cv_batched_dense_worker_ordered_recompute_barrier_"
            "qwen_mtbench_k8_bs8_t64_20260527/`, plus "
            "`speclink_cv_prefix_nokv_recompute_barrier_qwen_mtbench_k8_bs8_t64_"
            "20260527/`, and the 2026-05-28 tie-fix probes "
            "`speclink_cv_qwen_mtbench_k8_bs16_t12_tiefix_20260528/`, "
            "`speclink_cv_qwen_mtbench_k8_bs16_t32_tiefix_20260528/`, "
            "`speclink_cv_qwen_mtbench_k8_bs16_t32_row0_debug_20260528/`, "
            "`speclink_cv_qwen_mtbench_k8_bs16_t32_historykv_20260528/`, "
            "`speclink_cv_qwen_mtbench_k8_bs16_t32_nobarrier_tiefix_20260528/`, "
            "and "
            "`speclink_cv_qwen_mtbench_k8_bs16_t32_exactsafe_tiefix_20260528/`. "
            "These are "
            "included in the token summary when present, but remain local "
            "supplemental diagnostics rather than standalone final result "
            "roots.",
            "- `62_qwen_mtbench_k8_bs8_guidellm_batchinv_ablation/`: "
            "Qwen3/MTBench/K=8/bs=8 GuideLLM ablation with pure vLLM, "
            "EAGLE3, and all 8 CV methods at 24 requests and 64 output "
            "tokens; all CV rows are slower than EAGLE3 and fail the "
            "GuideLLM text exact-match speedup gate.",
            "- `13_live_correctness_gate/`, `18_qwen_math_k8_batch_sweep/`, "
            "`19_cross_model_dataset_k8_bs8/`, `20_qwen_math_k12_batch_sweep/`, "
            "`21_k12_cross_model_data_bs8/`: raw correctness gate artifacts.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    warnings = [
        row
        for row in combined_rows
        if str(row.get("method", "")).startswith("speclink_cv")
        and safe_float(row.get("exact_match_vs_eagle3")) != 1.0
    ]
    write_csv(root / "09_reports" / "correctness_warnings.csv", warnings)
    invalid = [
        row
        for row in combined_rows
        if str(row.get("method", "")).startswith("speclink_cv")
        and str(row.get("speedup_claim_status", "")) != "valid_exact_chunked"
    ]
    write_csv(root / "09_reports" / "speedup_claim_warnings.csv", invalid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.audit_root.resolve()
    if not root.exists():
        raise SystemExit(f"audit root does not exist: {root}")
    token_rows = collect_token_gate_rows(root)
    guidellm_rows = collect_guidellm_rows(root)
    diagnostic_rows = collect_followup_diagnostic_rows(root)
    combined_rows = guidellm_rows + diagnostic_rows + token_rows
    write_csv(root / "09_reports" / "token_id_correctness_summary.csv", token_rows)
    write_json(
        root / "09_reports" / "token_id_correctness_summary.json",
        {"summary": summarize_gate(token_rows), "rows": token_rows},
    )
    write_csv(root / "09_reports" / "summary_metrics.csv", combined_rows)
    write_json(root / "09_reports" / "summary_metrics.json", {"rows": combined_rows})
    write_csv(root / "summary_metrics.csv", combined_rows)
    write_json(root / "summary_metrics.json", {"rows": combined_rows})
    write_patch_snapshot(root)
    write_report(root, combined_rows, token_rows)
    write_audit(root)
    print(f"[INFO] finalized correctness audit: {root}")
    print(
        "[INFO] token gate rows: "
        f"{len(token_rows)}; follow-up diagnostic rows: {len(diagnostic_rows)}; "
        f"combined summary rows: {len(combined_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
