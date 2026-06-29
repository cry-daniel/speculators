#!/usr/bin/env python3
"""Summarize SR24 throughput and component breakdown artifacts.

Example:
  conda run -n spec python scripts/summarize_sr24_breakdown.py \
    --roots results.bak/sr24_breakdown_lowsync_graphon_flushfix_bs64_64req_20260624 \
            results.bak/sr24_breakdown_eager_linear_exact_bs64_64req_20260624 \
    --output-root results.bak/sr24_breakdown_component_summary_20260624
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]

LINEAR_PROFILE_BUCKETS = {
    "gate_up_proj": "by_leaf_gate_up_proj",
    "gate_up_proj_layers_16_31": "by_leaf_gate_up_proj__layers_16_31",
    "down_proj": "by_leaf_down_proj",
    "down_proj_layers_8_15": "by_leaf_down_proj__layers_8_15",
    "down_proj_layers_16_31": "by_leaf_down_proj__layers_16_31",
}


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return EVAL_ROOT / path


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _load_json_object_or_last_jsonl(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    latest: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            latest = value
    return latest


def _graph_counts(row: dict[str, str], breakdown: dict[str, Any]) -> str:
    raw = row.get("sr24_cudagraph_mode_counts") or ""
    if raw:
        return raw
    counts = breakdown.get("counts") or {}
    graph = {
        key.removeprefix("cudagraph_mode_"): int(value)
        for key, value in counts.items()
        if key.startswith("cudagraph_mode_")
    }
    return json.dumps(graph, sort_keys=True) if graph else ""


def _parse_graph_counts(raw: Any) -> dict[str, int]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return {
            str(key): int(value)
            for key, value in raw.items()
            if _as_float(value) is not None
        }
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(count)
        for key, count in value.items()
        if _as_float(count) is not None
    }


def _graph_fraction(raw: Any, mode: str) -> float | None:
    counts = _parse_graph_counts(raw)
    total = sum(counts.values())
    if total <= 0:
        return None
    return counts.get(mode, 0) / total


def _row_has_linear_timing(row: dict[str, Any]) -> bool:
    timing_keys = (
        "base_sparse_linear_cuda_ms_per_call",
        "row_routed_gate_up_cuda_ms_per_call",
        "row_routed_gate_up_base_sparse_cuda_ms_per_call",
        "row_routed_gate_up_dense_gemm_cuda_ms_per_call",
        "residual_dense_gemm_cuda_ms_per_call",
        "residual_sparse_gemm_cuda_ms_per_call",
        "bucket_triton_dense_gemm_scatter_cuda_ms_per_call",
        "compressed_residual_cuda_ms_per_event",
    )
    return any(_as_float(row.get(key)) is not None for key in timing_keys)


def _residual_correction_ms(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    for key in (
        "bucket_triton_dense_gemm_scatter_cuda_ms_per_call",
        "residual_dense_gemm_cuda_ms_per_call",
        "residual_sparse_gemm_cuda_ms_per_call",
        "compressed_residual_cuda_ms_per_event",
    ):
        value = _fmt(row.get(key))
        if value:
            return value
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _classify_row(row: dict[str, Any]) -> str:
    if _as_float(row.get("full_batch_output_tps")) is None:
        return "diagnostic"
    if _row_has_linear_timing(row):
        return "diagnostic_component_timing"
    if row.get("breakdown_path"):
        return "diagnostic_gpu_counts"
    sync_ms = _as_float(row.get("scheduler_mask_state_sync_cpu_ms_per_call"))
    if sync_ms is not None and sync_ms > 1.0:
        return "diagnostic_sync"
    if _truthy(row.get("sr24_sync_mask_state")):
        return "diagnostic_sync"
    if row.get("sr24_sync_reduced_stats") not in (None, "") and not _truthy(
            row.get("sr24_sync_reduced_stats")):
        return "diagnostic_sync"
    mask_state_wall_ms = _as_float(
        row.get("scheduler_mask_state_wall_cpu_ms_per_step"))
    if mask_state_wall_ms is not None and mask_state_wall_ms > 1.0:
        return "diagnostic_sync"
    routing_loop_ms = _as_float(
        row.get("scheduler_request_routing_loop_cpu_ms_per_call")
    )
    if routing_loop_ms is not None and routing_loop_ms > 1.0:
        return "diagnostic_scheduler_timing"
    routing_loop_wall_ms = _as_float(
        row.get("scheduler_request_routing_loop_wall_cpu_ms_per_step"))
    if routing_loop_wall_ms is not None and routing_loop_wall_ms > 1.0:
        return "diagnostic_scheduler_timing"
    if _truthy(row.get("sr24_stats_exact")):
        return "diagnostic_exact_stats"
    return "clean_serving"


def _sum_keys(data: dict[str, Any], keys: list[str]) -> float | None:
    values = [_as_float(data.get(key)) for key in keys]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values)


def _suffix_keys(keys: list[str], suffix: str) -> list[str]:
    return [f"{key}__{suffix}" for key in keys]


def _ratio(numer: Any, denom: Any) -> float | None:
    n = _as_float(numer)
    d = _as_float(denom)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _ratio_sum(numer_keys: list[str], denom_keys: list[str],
               counts: dict[str, Any]) -> float | None:
    numer = sum(
        value
        for value in (_as_float(counts.get(key)) for key in numer_keys)
        if value is not None
    )
    denom = sum(
        value
        for value in (_as_float(counts.get(key)) for key in denom_keys)
        if value is not None
    )
    if denom == 0:
        return None
    return numer / denom


def _read_summary_rows(root: Path) -> list[dict[str, str]]:
    summary_path = next(
        (
            path
            for path in (
                root / "summary.csv",
                root / "median_summary.csv",
                root / "breakdown_summary.csv",
            )
            if path.exists()
        ),
        None,
    )
    if summary_path is None:
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_already_summarized_row(row: dict[str, str]) -> bool:
    return "row_kind" in row and (
        "total_output_tps" in row
        or "scheduler_mask_build_cpu_ms_per_step" in row
        or "base_sparse_linear_cuda_ms_per_call" in row
    )


def _summarize_row(root: Path, row: dict[str, str]) -> dict[str, Any]:
    work_dir = Path(row.get("work_dir") or "")
    candidate_breakdowns = [
        work_dir / "speclink_sr24_breakdown.json",
        work_dir.parent / "speclink_sr24_breakdown.json",
    ]
    breakdown_path = next(
        (path for path in candidate_breakdowns if path.exists()),
        candidate_breakdowns[0],
    )
    breakdown = (
        _load_json_object_or_last_jsonl(breakdown_path)
        if breakdown_path.exists()
        else {}
    )
    counts = breakdown.get("counts") or {}
    cpu_avg = breakdown.get("cpu_avg_ms") or {}
    cuda_avg = breakdown.get("cuda_avg_ms") or {}
    cuda_ms = breakdown.get("cuda_ms") or {}
    cuda_calls = breakdown.get("cuda_calls") or {}
    derived = breakdown.get("derived") or {}

    base_sparse_keys = [
        "base_sparse_linear_cuda_ms",
        "route_bucket_dense_rows_base_sparse_gemm_cuda_ms",
        "route_all_residual_rows_base_sparse_gemm_cuda_ms",
    ]
    residual_dense_keys = [
        "residual_dense_gemm_cuda_ms",
        "residual_dense_full_gemm_cuda_ms",
        "residual_dense_rows_gemm_cuda_ms",
        "residual_dense_rows_full_gemm_cuda_ms",
        "route_bucket_dense_rows_dense_gemm_cuda_ms",
        "route_all_residual_rows_dense_gemm_cuda_ms",
    ]
    residual_sparse_keys = [
        "residual_sparse_gemm_cuda_ms",
        "residual_sparse_full_gemm_cuda_ms",
    ]
    bucket_triton_dense_gemm_keys = [
        "bucket_triton_dense_gemm_scatter_cuda_ms",
    ]
    base_sparse_total = _sum_keys(cuda_ms, base_sparse_keys)
    base_sparse_calls = _sum_keys(cuda_calls, base_sparse_keys)
    residual_dense_total = _sum_keys(cuda_ms, residual_dense_keys)
    residual_dense_calls = _sum_keys(cuda_calls, residual_dense_keys)
    residual_sparse_total = _sum_keys(cuda_ms, residual_sparse_keys)
    residual_sparse_calls = _sum_keys(cuda_calls, residual_sparse_keys)
    bucket_triton_dense_gemm_total = _sum_keys(
        cuda_ms, bucket_triton_dense_gemm_keys
    )
    bucket_triton_dense_gemm_calls = _sum_keys(
        cuda_calls, bucket_triton_dense_gemm_keys
    )
    gather_scatter_total = _sum_keys(
        cuda_ms,
        [
            "gather_input_index_select_cuda_ms",
            "gather_base_index_select_cuda_ms",
            "bucket_delta_compute_cuda_ms",
            "bucket_triton_override_cuda_ms",
            "scatter_index_add_cuda_ms",
            "residual_sparse_full_select_cuda_ms",
            "residual_dense_full_select_cuda_ms",
            "compressed_residual_gather_cuda_ms",
            "compressed_residual_slice_copy_cuda_ms",
            "compressed_residual_add_cuda_ms",
            "compressed_residual_scatter_cuda_ms",
            "compressed_residual_mask_add_cuda_ms",
            "residual_dense_rows_gather_cuda_ms",
            "residual_dense_rows_clone_cuda_ms",
            "residual_dense_rows_index_copy_cuda_ms",
            "route_bucket_dense_rows_route_build_cuda_ms",
            "route_bucket_dense_rows_base_gather_cuda_ms",
            "route_bucket_dense_rows_dense_gather_cuda_ms",
            "route_bucket_dense_rows_base_index_copy_cuda_ms",
            "route_bucket_dense_rows_dense_index_copy_cuda_ms",
            "route_all_residual_rows_route_build_cuda_ms",
            "route_all_residual_rows_base_gather_cuda_ms",
            "route_all_residual_rows_dense_gather_cuda_ms",
            "route_all_residual_rows_base_index_copy_cuda_ms",
            "route_all_residual_rows_dense_index_copy_cuda_ms",
        ],
    )
    gather_scatter_calls = _sum_keys(
        cuda_calls,
        [
            "gather_input_index_select_cuda_ms",
            "gather_base_index_select_cuda_ms",
            "bucket_delta_compute_cuda_ms",
            "bucket_triton_override_cuda_ms",
            "scatter_index_add_cuda_ms",
            "residual_sparse_full_select_cuda_ms",
            "residual_dense_full_select_cuda_ms",
            "compressed_residual_gather_cuda_ms",
            "compressed_residual_slice_copy_cuda_ms",
            "compressed_residual_add_cuda_ms",
            "compressed_residual_scatter_cuda_ms",
            "compressed_residual_mask_add_cuda_ms",
            "residual_dense_rows_gather_cuda_ms",
            "residual_dense_rows_clone_cuda_ms",
            "residual_dense_rows_index_copy_cuda_ms",
            "route_bucket_dense_rows_route_build_cuda_ms",
            "route_bucket_dense_rows_base_gather_cuda_ms",
            "route_bucket_dense_rows_dense_gather_cuda_ms",
            "route_bucket_dense_rows_base_index_copy_cuda_ms",
            "route_bucket_dense_rows_dense_index_copy_cuda_ms",
            "route_all_residual_rows_route_build_cuda_ms",
            "route_all_residual_rows_base_gather_cuda_ms",
            "route_all_residual_rows_dense_gather_cuda_ms",
            "route_all_residual_rows_base_index_copy_cuda_ms",
            "route_all_residual_rows_dense_index_copy_cuda_ms",
        ],
    )
    route_all_gather_scatter_keys = [
        "route_all_residual_rows_route_build_cuda_ms",
        "route_all_residual_rows_base_gather_cuda_ms",
        "route_all_residual_rows_dense_gather_cuda_ms",
        "route_all_residual_rows_base_index_copy_cuda_ms",
        "route_all_residual_rows_dense_index_copy_cuda_ms",
    ]
    route_all_gather_scatter_total = _sum_keys(
        cuda_ms, route_all_gather_scatter_keys
    )
    compressed_residual_overhead_total = _sum_keys(
        cuda_ms,
        [
            "compressed_residual_triton_cuda_ms",
            "compressed_residual_materialize_cuda_ms",
            "compressed_residual_gemm_cuda_ms",
            "compressed_residual_gather_cuda_ms",
            "compressed_residual_slice_copy_cuda_ms",
            "compressed_residual_add_cuda_ms",
            "compressed_residual_scatter_cuda_ms",
            "compressed_residual_mask_add_cuda_ms",
        ],
    )
    compressed_residual_overhead_calls = _sum_keys(
        cuda_calls,
        [
            "compressed_residual_triton_cuda_ms",
            "compressed_residual_materialize_cuda_ms",
            "compressed_residual_gemm_cuda_ms",
            "compressed_residual_gather_cuda_ms",
            "compressed_residual_slice_copy_cuda_ms",
            "compressed_residual_add_cuda_ms",
            "compressed_residual_scatter_cuda_ms",
            "compressed_residual_mask_add_cuda_ms",
        ],
    )
    row_routed_mlp_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_mlp_route_build_cuda_ms",
            "row_routed_mlp_dense_gather_cuda_ms",
            "row_routed_mlp_dense_gate_up_cuda_ms",
            "row_routed_mlp_base_gather_cuda_ms",
            "row_routed_mlp_base_gate_up_cuda_ms",
            "row_routed_mlp_gate_up_cat_cuda_ms",
            "row_routed_mlp_act_cuda_ms",
            "row_routed_mlp_down_sparse_cuda_ms",
            "row_routed_mlp_triton_assemble_cuda_ms",
            "row_routed_mlp_index_copy_cuda_ms",
        ],
    )
    row_routed_mlp_reuse_base_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_mlp_dense_gather_cuda_ms",
            "row_routed_mlp_dense_gate_up_cuda_ms",
            "row_routed_mlp_dense_act_cuda_ms",
            "row_routed_mlp_dense_down_cuda_ms",
            "row_routed_mlp_reuse_base_gate_up_cuda_ms",
            "row_routed_mlp_reuse_base_act_cuda_ms",
            "row_routed_mlp_reuse_base_down_sparse_cuda_ms",
            "row_routed_mlp_reuse_base_index_copy_cuda_ms",
        ],
    )
    row_routed_mlp_reuse_base_dense_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_mlp_dense_gather_cuda_ms",
            "row_routed_mlp_dense_gate_up_cuda_ms",
            "row_routed_mlp_dense_act_cuda_ms",
            "row_routed_mlp_dense_down_cuda_ms",
            "row_routed_mlp_reuse_base_index_copy_cuda_ms",
        ],
    )
    row_routed_mlp_reuse_base_base_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_mlp_reuse_base_gate_up_cuda_ms",
            "row_routed_mlp_reuse_base_act_cuda_ms",
            "row_routed_mlp_reuse_base_down_sparse_cuda_ms",
        ],
    )
    row_routed_mlp_reuse_base_gather_scatter_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_mlp_dense_gather_cuda_ms",
            "row_routed_mlp_reuse_base_index_copy_cuda_ms",
        ],
    )
    row_routed_gate_up_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_gate_up_route_build_cuda_ms",
            "row_routed_gate_up_dense_gather_cuda_ms",
            "row_routed_gate_up_dense_gemm_cuda_ms",
            "row_routed_gate_up_base_gather_cuda_ms",
            "row_routed_gate_up_base_sparse_cuda_ms",
            "row_routed_gate_up_index_copy_cuda_ms",
            "row_routed_gate_up_full_dense_cuda_ms",
        ],
    )
    row_routed_gate_up_base_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_gate_up_base_gather_cuda_ms",
            "row_routed_gate_up_base_sparse_cuda_ms",
        ],
    )
    row_routed_gate_up_dense_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_gate_up_dense_gather_cuda_ms",
            "row_routed_gate_up_dense_gemm_cuda_ms",
            "row_routed_gate_up_full_dense_cuda_ms",
        ],
    )
    row_routed_gate_up_gather_scatter_total = _sum_keys(
        cuda_ms,
        [
            "row_routed_gate_up_route_build_cuda_ms",
            "row_routed_gate_up_dense_gather_cuda_ms",
            "row_routed_gate_up_base_gather_cuda_ms",
            "row_routed_gate_up_index_copy_cuda_ms",
        ],
    )
    adaptive_dense_fallback_total = _sum_keys(
        cuda_ms,
        ["adaptive_dense_fallback_gemm_cuda_ms"],
    )
    adaptive_dense_fallback_calls = _sum_keys(
        cuda_calls,
        ["adaptive_dense_fallback_gemm_cuda_ms"],
    )

    result = {
        "source_root": str(root.resolve()),
        "method": row.get("method", ""),
        "dataset": row.get("dataset", ""),
        "batch_size": row.get("batch_size", ""),
        "max_new_tokens": row.get("max_new_tokens", ""),
        "status": row.get("status", ""),
        "total_output_tps": row.get("total_output_tokens_per_second", ""),
        "full_batch_output_tps": row.get("full_batch_output_tokens_per_second", ""),
        "avg_gpu_util_pct": row.get("avg_gpu_util_pct", ""),
        "peak_gpu_util_pct": row.get("peak_gpu_util_pct", ""),
        "tpot_ms_mean": row.get("tpot_ms_mean", ""),
        "spec_acceptance_rate": row.get("spec_acceptance_rate", ""),
        "avg_accepted_draft_tokens_per_step": row.get(
            "spec_avg_accepted_draft_tokens_per_step", ""
        ),
        "avg_selected_draft_tokens_per_step": row.get(
            "spec_avg_selected_draft_tokens_per_step", ""
        ),
        "cudagraph_mode_counts": _graph_counts(row, breakdown),
        "cudagraph_full_fraction": _graph_fraction(
            _graph_counts(row, breakdown), "FULL"
        ),
        "cudagraph_none_fraction": _graph_fraction(
            _graph_counts(row, breakdown), "NONE"
        ),
        "sr24_sync_mask_state": row.get("sr24_sync_mask_state", ""),
        "sr24_stats_exact": row.get("sr24_stats_exact", ""),
        "sr24_bucket_calls": row.get("sr24_bucket_calls", ""),
        "sr24_bucket_candidate_rows": row.get("sr24_bucket_candidate_rows", ""),
        "sr24_bucket_active_rows": row.get("sr24_bucket_active_rows", ""),
        "sr24_bucket_total_rows": row.get("sr24_bucket_total_rows", ""),
        "sr24_bucket_residual_requested_rows": row.get(
            "sr24_bucket_residual_requested_rows", ""
        ),
        "sr24_bucket_candidate_rows_per_call": row.get(
            "sr24_bucket_candidate_rows_per_call", ""
        ),
        "sr24_bucket_active_rows_per_call": row.get(
            "sr24_bucket_active_rows_per_call", ""
        ),
        "sr24_bucket_active_fraction_of_requested": (
            row.get("sr24_bucket_active_fraction_of_requested", "")
            or _ratio(
                row.get("sr24_bucket_active_rows"),
                row.get("sr24_bucket_residual_requested_rows"),
            )
        ),
        "scheduler_mask_build_cpu_ms_per_step": (
            cpu_avg.get("scheduler_mask_build_cpu_ms")
            or row.get("sr24_scheduler_mask_wall_cpu_ms_per_step", "")
        ),
        "scheduler_mask_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_mask_wall_cpu_ms_per_step", ""
        ),
        "scheduler_materialize_counts_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_materialize_counts_wall_cpu_ms_per_step", ""
        ),
        "scheduler_pending_scores_pop_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_pending_scores_pop_wall_cpu_ms_per_step", ""
        ),
        "scheduler_batched_mask_builder_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_batched_mask_builder_wall_cpu_ms_per_step", ""
        ),
        "scheduler_request_routing_loop_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_request_routing_loop_wall_cpu_ms_per_step", ""
        ),
        "scheduler_batch_all_apply_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_batch_all_apply_wall_cpu_ms_per_step", ""
        ),
        "scheduler_mask_state_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_mask_state_wall_cpu_ms_per_step", ""
        ),
        "scheduler_static_mask_copy_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_static_mask_copy_wall_cpu_ms_per_step", ""
        ),
        "scheduler_row_index_bucket_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_row_index_bucket_wall_cpu_ms_per_step", ""
        ),
        "scheduler_residual_bucket_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_residual_bucket_wall_cpu_ms_per_step", ""
        ),
        "scheduler_mixed_row_indices_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_mixed_row_indices_wall_cpu_ms_per_step", ""
        ),
        "scheduler_direct_cpu_route_rows_wall_cpu_ms_per_step": row.get(
            "sr24_scheduler_direct_cpu_route_rows_wall_cpu_ms_per_step", ""
        ),
        "scheduler_materialize_counts_cpu_ms_per_call": cpu_avg.get(
            "scheduler_materialize_counts_cpu_ms"
        ),
        "scheduler_mask_init_cpu_ms_per_call": cpu_avg.get(
            "scheduler_mask_init_cpu_ms"
        ),
        "scheduler_mask_init_cuda_ms_per_call": cuda_avg.get(
            "scheduler_mask_init_cuda_ms"
        ),
        "scheduler_pending_scores_pop_cpu_ms_per_call": cpu_avg.get(
            "scheduler_pending_scores_pop_cpu_ms"
        ),
        "scheduler_request_routing_loop_cpu_ms_per_call": cpu_avg.get(
            "scheduler_request_routing_loop_cpu_ms"
        ),
        "scheduler_score_policy_cuda_ms_per_call": cuda_avg.get(
            "scheduler_score_policy_cuda_ms"
        ),
        "scheduler_mask_write_cuda_ms_per_call": cuda_avg.get(
            "scheduler_mask_write_cuda_ms"
        ),
        "scheduler_mask_state_sync_cpu_ms_per_call": cpu_avg.get(
            "scheduler_mask_state_sync_cpu_ms"
        ),
        "scheduler_mask_state_sum_cuda_ms_per_call": cuda_avg.get(
            "scheduler_mask_state_sum_cuda_ms"
        ),
        "scheduler_bucket_build_cpu_ms_per_call": cpu_avg.get(
            "scheduler_bucket_build_cpu_ms"
        ),
        "scheduler_bucket_topk_cuda_ms_total": cuda_ms.get(
            "scheduler_bucket_topk_cuda_ms"
        ),
        "scheduler_bucket_topk_cuda_ms_per_call": cuda_avg.get(
            "scheduler_bucket_topk_cuda_ms"
        ),
        "scheduler_direct_position_bucket_cpu_ms_per_call": cpu_avg.get(
            "scheduler_direct_position_bucket_cpu_ms"
        ),
        "scheduler_direct_position_bucket_cuda_ms_per_call": cuda_avg.get(
            "scheduler_direct_position_bucket_cuda_ms"
        ),
        "scheduler_direct_position_bucket_builds": counts.get(
            "scheduler_direct_position_bucket_builds"
        ),
        "scheduler_direct_position_bucket_vector_builds": counts.get(
            "scheduler_direct_position_bucket_vector_builds"
        ),
        "scheduler_direct_position_bucket_rows_per_build": _ratio(
            counts.get("scheduler_direct_position_bucket_rows"),
            counts.get("scheduler_direct_position_bucket_builds"),
        ),
        "scheduler_batched_mask_kernel_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_kernel_cuda_ms"
        ),
        "scheduler_batched_mask_uniform_direct_kernel_cuda_ms_per_step":
        cuda_avg.get("scheduler_batched_mask_uniform_direct_kernel_cuda_ms"),
        "scheduler_batched_mask_indexed_kernel_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_indexed_kernel_cuda_ms"
        ),
        "scheduler_batched_mask_req_kernel_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_req_kernel_cuda_ms"
        ),
        "scheduler_batched_mask_req_indexed_kernel_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_req_indexed_kernel_cuda_ms"
        ),
        "scheduler_batched_mask_setup_cpu_ms_per_step": cpu_avg.get(
            "scheduler_batched_mask_setup_cpu_ms"
        ),
        "scheduler_batched_mask_tensor_setup_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_tensor_setup_cuda_ms"
        ),
        "scheduler_batched_mask_score_rows_setup_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_score_rows_setup_cuda_ms"
        ),
        "scheduler_batched_mask_score_matrix_setup_cuda_ms_per_step": cuda_avg.get(
            "scheduler_batched_mask_score_matrix_setup_cuda_ms"
        ),
        "scheduler_dense_fallback_steps": counts.get(
            "scheduler_dense_fallback_steps"
        ),
        "mask_state_all_residual_steps": counts.get("mask_state_all_residual"),
        "mask_state_mixed_steps": counts.get("mask_state_mixed"),
        "mask_state_no_residual_steps": counts.get("mask_state_no_residual"),
        "base_sparse_linear_cuda_ms_total": base_sparse_total,
        "base_sparse_linear_cuda_ms_per_call": _ratio(
            base_sparse_total, base_sparse_calls
        ),
        "residual_dense_gemm_cuda_ms_total": residual_dense_total,
        "residual_dense_gemm_cuda_ms_per_call": _ratio(
            residual_dense_total, residual_dense_calls
        ),
        "residual_sparse_gemm_cuda_ms_total": residual_sparse_total,
        "residual_sparse_gemm_cuda_ms_per_call": _ratio(
            residual_sparse_total, residual_sparse_calls
        ),
        "compressed_residual_cuda_ms_total": compressed_residual_overhead_total,
        "compressed_residual_cuda_ms_per_event": _ratio(
            compressed_residual_overhead_total,
            compressed_residual_overhead_calls,
        ),
        "compressed_residual_materialize_cuda_ms_total": cuda_ms.get(
            "compressed_residual_materialize_cuda_ms"
        ),
        "compressed_residual_materialize_cuda_ms_per_call": cuda_avg.get(
            "compressed_residual_materialize_cuda_ms"
        ),
        "compressed_residual_triton_cuda_ms_total": cuda_ms.get(
            "compressed_residual_triton_cuda_ms"
        ),
        "compressed_residual_triton_cuda_ms_per_call": cuda_avg.get(
            "compressed_residual_triton_cuda_ms"
        ),
        "compressed_residual_gemm_cuda_ms_total": cuda_ms.get(
            "compressed_residual_gemm_cuda_ms"
        ),
        "compressed_residual_gemm_cuda_ms_per_call": cuda_avg.get(
            "compressed_residual_gemm_cuda_ms"
        ),
        "compressed_residual_add_cuda_ms_total": cuda_ms.get(
            "compressed_residual_add_cuda_ms"
        ),
        "compressed_residual_add_cuda_ms_per_call": cuda_avg.get(
            "compressed_residual_add_cuda_ms"
        ),
        "bucket_triton_override_cuda_ms_per_call": cuda_avg.get(
            "bucket_triton_override_cuda_ms"
        ),
        "bucket_triton_dense_gemm_scatter_cuda_ms_total":
        bucket_triton_dense_gemm_total,
        "bucket_triton_dense_gemm_scatter_cuda_ms_per_call": _ratio(
            bucket_triton_dense_gemm_total,
            bucket_triton_dense_gemm_calls,
        ),
        "bucket_triton_dense_gemm_scatter_rows_per_call": _ratio(
            counts.get("bucket_triton_dense_gemm_scatter_rows"),
            counts.get("bucket_triton_dense_gemm_scatter_calls"),
        ),
        "compressed_residual_cached_weight_hits": counts.get(
            "compressed_residual_cached_weight_hits"
        ),
        "compressed_residual_cached_weight_misses": counts.get(
            "compressed_residual_cached_weight_misses"
        ),
        "gather_scatter_cuda_ms_total": gather_scatter_total,
        "gather_scatter_cuda_ms_per_call": _ratio(
            gather_scatter_total, gather_scatter_calls
        ),
        "route_all_gather_scatter_cuda_ms_total": route_all_gather_scatter_total,
        "route_all_gather_scatter_cuda_ms_per_linear_call": _ratio(
            route_all_gather_scatter_total,
            counts.get("route_all_residual_rows_calls"),
        ),
        "route_all_base_gather_cuda_ms_per_call": cuda_avg.get(
            "route_all_residual_rows_base_gather_cuda_ms"
        ),
        "route_all_dense_gather_cuda_ms_per_call": cuda_avg.get(
            "route_all_residual_rows_dense_gather_cuda_ms"
        ),
        "route_all_base_index_copy_cuda_ms_per_call": cuda_avg.get(
            "route_all_residual_rows_base_index_copy_cuda_ms"
        ),
        "route_all_dense_index_copy_cuda_ms_per_call": cuda_avg.get(
            "route_all_residual_rows_dense_index_copy_cuda_ms"
        ),
        "row_routed_mlp_cuda_ms_total": row_routed_mlp_total,
        "row_routed_mlp_cuda_ms_per_call": _ratio(
            row_routed_mlp_total, counts.get("row_routed_mlp_calls")
        ),
        "row_routed_mlp_reuse_base_cuda_ms_total": (
            row_routed_mlp_reuse_base_total
        ),
        "row_routed_mlp_reuse_base_cuda_ms_per_call": _ratio(
            row_routed_mlp_reuse_base_total,
            counts.get("row_routed_mlp_reuse_base_output_calls"),
        ),
        "row_routed_mlp_reuse_base_dense_cuda_ms_total": (
            row_routed_mlp_reuse_base_dense_total
        ),
        "row_routed_mlp_reuse_base_dense_cuda_ms_per_call": _ratio(
            row_routed_mlp_reuse_base_dense_total,
            counts.get("row_routed_mlp_reuse_base_output_calls"),
        ),
        "row_routed_mlp_reuse_base_base_cuda_ms_total": (
            row_routed_mlp_reuse_base_base_total
        ),
        "row_routed_mlp_reuse_base_base_cuda_ms_per_call": _ratio(
            row_routed_mlp_reuse_base_base_total,
            counts.get("row_routed_mlp_reuse_base_output_calls"),
        ),
        "row_routed_mlp_reuse_base_gather_scatter_cuda_ms_total": (
            row_routed_mlp_reuse_base_gather_scatter_total
        ),
        "row_routed_mlp_reuse_base_gather_scatter_cuda_ms_per_call": _ratio(
            row_routed_mlp_reuse_base_gather_scatter_total,
            counts.get("row_routed_mlp_reuse_base_output_calls"),
        ),
        "adaptive_dense_fallback_cuda_ms_total":
        adaptive_dense_fallback_total,
        "adaptive_dense_fallback_cuda_ms_per_call": _ratio(
            adaptive_dense_fallback_total,
            adaptive_dense_fallback_calls,
        ),
        "adaptive_dense_fallback_calls": counts.get(
            "adaptive_dense_fallback_calls"
        ),
        "adaptive_dense_fallback_rows_per_call": _ratio(
            counts.get("adaptive_dense_fallback_rows"),
            counts.get("adaptive_dense_fallback_calls"),
        ),
        "adaptive_dense_fallback_candidate_rows_per_call": _ratio(
            counts.get("adaptive_dense_fallback_candidate_rows"),
            counts.get("adaptive_dense_fallback_calls"),
        ),
        "row_routed_mlp_route_build_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_route_build_cuda_ms"
        ),
        "row_routed_mlp_dense_gather_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_dense_gather_cuda_ms"
        ),
        "row_routed_mlp_dense_gate_up_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_dense_gate_up_cuda_ms"
        ),
        "row_routed_mlp_dense_act_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_dense_act_cuda_ms"
        ),
        "row_routed_mlp_dense_down_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_dense_down_cuda_ms"
        ),
        "row_routed_mlp_base_gather_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_base_gather_cuda_ms"
        ),
        "row_routed_mlp_base_gate_up_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_base_gate_up_cuda_ms"
        ),
        "row_routed_mlp_gate_up_cat_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_gate_up_cat_cuda_ms"
        ),
        "row_routed_mlp_act_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_act_cuda_ms"
        ),
        "row_routed_mlp_down_sparse_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_down_sparse_cuda_ms"
        ),
        "row_routed_mlp_reuse_base_gate_up_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_reuse_base_gate_up_cuda_ms"
        ),
        "row_routed_mlp_reuse_base_act_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_reuse_base_act_cuda_ms"
        ),
        "row_routed_mlp_reuse_base_down_sparse_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_reuse_base_down_sparse_cuda_ms"
        ),
        "row_routed_mlp_reuse_base_index_copy_cuda_ms_per_call": cuda_avg.get(
            "row_routed_mlp_reuse_base_index_copy_cuda_ms"
        ),
        "row_routed_mlp_assemble_cuda_ms_per_call": (
            cuda_avg.get("row_routed_mlp_triton_assemble_cuda_ms")
            or cuda_avg.get("row_routed_mlp_index_copy_cuda_ms")
        ),
        "row_routed_mlp_calls": counts.get("row_routed_mlp_calls"),
        "row_routed_mlp_reuse_base_output_calls": counts.get(
            "row_routed_mlp_reuse_base_output_calls"
        ),
        "row_routed_mlp_rows_per_call": _ratio(
            counts.get("row_routed_mlp_rows"),
            counts.get("row_routed_mlp_calls"),
        ),
        "row_routed_mlp_reuse_base_rows_per_call": _ratio(
            counts.get("row_routed_mlp_reuse_base_output_rows"),
            counts.get("row_routed_mlp_reuse_base_output_calls"),
        ),
        "row_routed_mlp_dense_rows_per_call": _ratio(
            counts.get("row_routed_mlp_dense_rows"),
            counts.get("row_routed_mlp_calls"),
        ),
        "row_routed_mlp_reuse_base_dense_rows_per_call": _ratio(
            counts.get("row_routed_mlp_reuse_base_output_dense_rows"),
            counts.get("row_routed_mlp_reuse_base_output_calls"),
        ),
        "row_routed_mlp_base_rows_per_call": _ratio(
            counts.get("row_routed_mlp_base_rows"),
            counts.get("row_routed_mlp_calls"),
        ),
        "row_routed_mlp_skipped_small_dense_rows": counts.get(
            "row_routed_mlp_skipped_small_dense_rows"
        ),
        "row_routed_mlp_skipped_dense_rows_per_skip": _ratio(
            counts.get("row_routed_mlp_skipped_dense_rows"),
            counts.get("row_routed_mlp_skipped_small_dense_rows"),
        ),
        "row_routed_mlp_skipped_total_rows_per_skip": _ratio(
            counts.get("row_routed_mlp_skipped_total_rows"),
            counts.get("row_routed_mlp_skipped_small_dense_rows"),
        ),
        "row_routed_gate_up_cuda_ms_total": row_routed_gate_up_total,
        "row_routed_gate_up_cuda_ms_per_call": _ratio(
            row_routed_gate_up_total,
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_base_cuda_ms_total": row_routed_gate_up_base_total,
        "row_routed_gate_up_base_cuda_ms_per_call": _ratio(
            row_routed_gate_up_base_total,
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_dense_cuda_ms_total": row_routed_gate_up_dense_total,
        "row_routed_gate_up_dense_cuda_ms_per_call": _ratio(
            row_routed_gate_up_dense_total,
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_gather_scatter_cuda_ms_total": (
            row_routed_gate_up_gather_scatter_total
        ),
        "row_routed_gate_up_gather_scatter_cuda_ms_per_call": _ratio(
            row_routed_gate_up_gather_scatter_total,
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_route_build_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_route_build_cuda_ms"
        ),
        "row_routed_gate_up_dense_gather_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_dense_gather_cuda_ms"
        ),
        "row_routed_gate_up_dense_gemm_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_dense_gemm_cuda_ms"
        ),
        "row_routed_gate_up_base_gather_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_base_gather_cuda_ms"
        ),
        "row_routed_gate_up_base_sparse_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_base_sparse_cuda_ms"
        ),
        "row_routed_gate_up_index_copy_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_index_copy_cuda_ms"
        ),
        "row_routed_gate_up_full_dense_cuda_ms_per_call": cuda_avg.get(
            "row_routed_gate_up_full_dense_cuda_ms"
        ),
        "row_routed_gate_up_calls": counts.get("row_routed_gate_up_calls"),
        "row_routed_gate_up_full_dense_calls": counts.get(
            "row_routed_gate_up_full_dense_calls"
        ),
        "row_routed_gate_up_cached_plan_hits": counts.get(
            "row_routed_gate_up_cached_plan_hits"
        ),
        "row_routed_gate_up_rows_per_call": _ratio(
            counts.get("row_routed_gate_up_rows"),
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_dense_rows_per_call": _ratio(
            counts.get("row_routed_gate_up_dense_rows"),
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_base_rows_per_call": _ratio(
            counts.get("row_routed_gate_up_base_rows"),
            counts.get("row_routed_gate_up_calls"),
        ),
        "row_routed_gate_up_skipped_small_dense_rows": counts.get(
            "row_routed_gate_up_skipped_small_dense_rows"
        ),
        "row_routed_gate_up_skipped_large_dense_rows": counts.get(
            "row_routed_gate_up_skipped_large_dense_rows"
        ),
        "row_routed_gate_up_skipped_large_base_rows": counts.get(
            "row_routed_gate_up_skipped_large_base_rows"
        ),
        "row_routed_gate_up_skipped_dense_rows_per_skip": _ratio(
            counts.get("row_routed_gate_up_skipped_dense_rows"),
            _sum_keys(
                counts,
                [
                    "row_routed_gate_up_skipped_small_dense_rows",
                    "row_routed_gate_up_skipped_large_dense_rows",
                    "row_routed_gate_up_skipped_large_base_rows",
                ],
            ),
        ),
        "row_routed_gate_up_skipped_total_rows_per_skip": _ratio(
            counts.get("row_routed_gate_up_skipped_total_rows"),
            _sum_keys(
                counts,
                [
                    "row_routed_gate_up_skipped_small_dense_rows",
                    "row_routed_gate_up_skipped_large_dense_rows",
                    "row_routed_gate_up_skipped_large_base_rows",
                ],
            ),
        ),
        "verify_steps": counts.get("verify_steps"),
        "batched_mask_builder_steps": counts.get("batched_mask_builder_steps"),
        "batched_mask_builder_indexed_steps": counts.get(
            "batched_mask_builder_indexed_steps"
        ),
        "batched_mask_builder_uniform_direct_steps": counts.get(
            "batched_mask_builder_uniform_direct_steps"
        ),
        "batched_mask_builder_gpu_count_steps": counts.get(
            "batched_mask_builder_gpu_count_steps"
        ),
        "batched_mask_builder_gpu_count_indexed_fallback_steps": counts.get(
            "batched_mask_builder_gpu_count_indexed_fallback_steps"
        ),
        "batched_mask_builder_direct_score_rows_steps": counts.get(
            "batched_mask_builder_direct_score_rows_steps"
        ),
        "avg_scheduled_tokens_per_step": derived.get(
            "avg_scheduled_tokens_per_step"
        ),
        "draft_residual_rows": counts.get("gpu_residual_draft_tokens")
        or counts.get("residual_draft_tokens")
        or row.get("sr24_residual_draft_tokens", ""),
        "draft_base_rows": counts.get("gpu_base_only_draft_tokens")
        or counts.get("base_only_draft_tokens")
        or row.get("sr24_base_only_draft_tokens", ""),
        "non_draft_residual_rows": counts.get("gpu_residual_non_draft_tokens")
        or counts.get("residual_non_draft_tokens")
        or row.get("sr24_residual_non_draft_tokens", ""),
        "non_draft_base_rows": counts.get("gpu_base_only_non_draft_tokens")
        or counts.get("base_only_non_draft_tokens")
        or row.get("sr24_base_only_non_draft_tokens", ""),
        "draft_residual_fraction": _ratio_sum(
            ["gpu_residual_draft_tokens", "residual_draft_tokens"],
            [
                "gpu_residual_draft_tokens",
                "residual_draft_tokens",
                "gpu_base_only_draft_tokens",
                "base_only_draft_tokens",
            ],
            counts,
        ),
        "non_draft_residual_fraction": _ratio_sum(
            ["gpu_residual_non_draft_tokens", "residual_non_draft_tokens"],
            [
                "gpu_residual_non_draft_tokens",
                "residual_non_draft_tokens",
                "gpu_base_only_non_draft_tokens",
                "base_only_non_draft_tokens",
            ],
            counts,
        ),
        "bucket_fill_ratio": derived.get("bucket_fill_ratio"),
        "avg_bucket_candidate_rows": derived.get("avg_bucket_candidate_rows"),
        "avg_bucket_active_rows": derived.get("avg_bucket_active_rows"),
        "gpu_bucket_active_rows": counts.get("gpu_bucket_active_rows"),
        "gpu_bucket_active_rows_per_step": _ratio(
            counts.get("gpu_bucket_active_rows"),
            counts.get("verify_steps"),
        ),
        "avg_bucket_actual_over_requested": (
            row.get("sr24_bucket_active_fraction_of_requested", "")
            or _ratio(
                row.get("sr24_bucket_active_rows"),
                row.get("sr24_bucket_residual_requested_rows"),
            )
        ),
        "base_sparse_linear_rows_per_call": (
            _ratio(
                counts.get("base_sparse_linear_rows"),
                counts.get("base_sparse_linear_calls"),
            )
            or _ratio(
                counts.get("route_bucket_dense_rows_base_rows"),
                counts.get("route_bucket_dense_rows_calls"),
            )
            or _ratio(
                counts.get("route_all_residual_rows_base_rows"),
                counts.get("route_all_residual_rows_calls"),
            )
        ),
        "dense_rows_bucket_rows_per_call": (
            _ratio(
                counts.get("dense_rows_bucket_rows"),
                counts.get("dense_rows_bucket_linear_calls"),
            )
            or _ratio(
                counts.get("route_bucket_dense_rows_dense_rows"),
                counts.get("route_bucket_dense_rows_calls"),
            )
            or _ratio(
                counts.get("route_all_residual_rows_dense_rows"),
                counts.get("route_all_residual_rows_calls"),
            )
        ),
        "residual_dense_rows_per_call": (
            _ratio(
                counts.get("residual_dense_rows"),
                counts.get("residual_dense_rows_calls"),
            )
            or _ratio(
                counts.get("route_all_residual_rows_dense_rows"),
                counts.get("route_all_residual_rows_calls"),
            )
        ),
        "residual_dense_rows_total_rows_per_call": _ratio(
            counts.get("residual_dense_rows_total_rows"),
            counts.get("residual_dense_rows_calls"),
        ),
        "residual_dense_rows_fraction": (
            _ratio(
                counts.get("residual_dense_rows"),
                counts.get("residual_dense_rows_total_rows"),
            )
            or _ratio(
                counts.get("route_bucket_dense_rows_dense_rows"),
                counts.get("route_bucket_dense_rows_rows"),
            )
            or _ratio(
                counts.get("route_all_residual_rows_dense_rows"),
                counts.get("route_all_residual_rows_rows"),
            )
        ),
        "route_bucket_enabled": bool(
            counts.get("bucket_calls") or counts.get("route_all_residual_rows_calls")
        ),
        "breakdown_path": str(breakdown_path.resolve())
        if breakdown_path.exists()
        else "",
        "work_dir": str(work_dir.resolve()) if work_dir.exists() else str(work_dir),
    }
    for bucket_label, suffix in LINEAR_PROFILE_BUCKETS.items():
        base_total = _sum_keys(cuda_ms, _suffix_keys(base_sparse_keys, suffix))
        base_calls = _sum_keys(cuda_calls, _suffix_keys(base_sparse_keys, suffix))
        residual_dense_bucket_total = _sum_keys(
            cuda_ms, _suffix_keys(residual_dense_keys, suffix)
        )
        residual_dense_bucket_calls = _sum_keys(
            cuda_calls, _suffix_keys(residual_dense_keys, suffix)
        )
        residual_sparse_bucket_total = _sum_keys(
            cuda_ms, _suffix_keys(residual_sparse_keys, suffix)
        )
        residual_sparse_bucket_calls = _sum_keys(
            cuda_calls, _suffix_keys(residual_sparse_keys, suffix)
        )
        bucket_triton_dense_total = _sum_keys(
            cuda_ms, _suffix_keys(bucket_triton_dense_gemm_keys, suffix)
        )
        bucket_triton_dense_calls = _sum_keys(
            cuda_calls, _suffix_keys(bucket_triton_dense_gemm_keys, suffix)
        )
        result[f"{bucket_label}_base_sparse_linear_cuda_ms_per_call"] = _ratio(
            base_total, base_calls
        )
        result[f"{bucket_label}_residual_dense_gemm_cuda_ms_per_call"] = _ratio(
            residual_dense_bucket_total, residual_dense_bucket_calls
        )
        result[f"{bucket_label}_residual_sparse_gemm_cuda_ms_per_call"] = _ratio(
            residual_sparse_bucket_total, residual_sparse_bucket_calls
        )
        result[
            f"{bucket_label}_bucket_triton_dense_gemm_scatter_cuda_ms_per_call"
        ] = _ratio(bucket_triton_dense_total, bucket_triton_dense_calls)
        result[f"{bucket_label}_base_sparse_linear_rows_per_call"] = (
            _ratio(
                counts.get(f"base_sparse_linear_rows__{suffix}"),
                counts.get(f"base_sparse_linear_calls__{suffix}"),
            )
            or _ratio(
                counts.get(f"route_bucket_dense_rows_base_rows__{suffix}"),
                counts.get(f"route_bucket_dense_rows_calls__{suffix}"),
            )
            or _ratio(
                counts.get(f"route_all_residual_rows_base_rows__{suffix}"),
                counts.get(f"route_all_residual_rows_calls__{suffix}"),
            )
        )
        result[f"{bucket_label}_residual_dense_rows_per_call"] = (
            _ratio(
                counts.get(f"residual_dense_rows__{suffix}"),
                counts.get(f"residual_dense_rows_calls__{suffix}"),
            )
            or _ratio(
                counts.get(f"dense_rows_bucket_rows__{suffix}"),
                counts.get(f"dense_rows_bucket_linear_calls__{suffix}"),
            )
            or _ratio(
                counts.get(f"route_bucket_dense_rows_dense_rows__{suffix}"),
                counts.get(f"route_bucket_dense_rows_calls__{suffix}"),
            )
            or _ratio(
                counts.get(f"route_all_residual_rows_dense_rows__{suffix}"),
                counts.get(f"route_all_residual_rows_calls__{suffix}"),
            )
        )
        result[f"{bucket_label}_bucket_triton_dense_gemm_scatter_rows_per_call"] = (
            _ratio(
                counts.get(f"bucket_triton_dense_gemm_scatter_rows__{suffix}"),
                counts.get(f"bucket_triton_dense_gemm_scatter_calls__{suffix}"),
            )
        )
    result["row_kind"] = _classify_row(result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _attach_relative_speedups(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str], dict[str, Any]] = {}
    same_root_baselines: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("method") not in {"dense_baseline", "vllm_eagle3"}:
            continue
        if row.get("row_kind") != "clean_serving":
            continue
        key = (
            str(row.get("dataset", "")),
            str(row.get("batch_size", "")),
            str(row.get("max_new_tokens", "")),
        )
        current = baselines.get(key)
        if current is None or (
            (_as_float(row.get("full_batch_output_tps")) or float("-inf"))
            > (_as_float(current.get("full_batch_output_tps")) or float("-inf"))
        ):
            baselines[key] = row
        same_key = (str(row.get("source_root", "")), *key)
        same_root_baselines[same_key] = row

    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("batch_size", "")),
            str(row.get("max_new_tokens", "")),
        )
        same_key = (str(row.get("source_root", "")), *key)
        same_root_baseline = same_root_baselines.get(same_key)
        if same_root_baseline is None:
            row["same_root_dense_reference_full_batch_tps"] = ""
            row["full_batch_speedup_vs_same_root_dense"] = ""
            row["total_tps_speedup_vs_same_root_dense"] = ""
        else:
            row["same_root_dense_reference_full_batch_tps"] = (
                same_root_baseline.get("full_batch_output_tps", "")
            )
            row["full_batch_speedup_vs_same_root_dense"] = _ratio(
                row.get("full_batch_output_tps"),
                same_root_baseline.get("full_batch_output_tps"),
            )
            row["total_tps_speedup_vs_same_root_dense"] = _ratio(
                row.get("total_output_tps"),
                same_root_baseline.get("total_output_tps"),
            )

        baseline = baselines.get(key)
        if baseline is None:
            row["best_clean_dense_reference_full_batch_tps"] = ""
            row["full_batch_speedup_vs_best_clean_dense"] = ""
            row["total_tps_speedup_vs_best_clean_dense"] = ""
            continue
        row["best_clean_dense_reference_full_batch_tps"] = baseline.get(
            "full_batch_output_tps", ""
        )
        row["full_batch_speedup_vs_best_clean_dense"] = _ratio(
            row.get("full_batch_output_tps"),
            baseline.get("full_batch_output_tps"),
        )
        row["total_tps_speedup_vs_best_clean_dense"] = _ratio(
            row.get("total_output_tps"),
            baseline.get("total_output_tps"),
        )


def _make_per_leaf_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        for bucket_label in LINEAR_PROFILE_BUCKETS:
            base_ms = row.get(
                f"{bucket_label}_base_sparse_linear_cuda_ms_per_call"
            )
            residual_dense_ms = row.get(
                f"{bucket_label}_residual_dense_gemm_cuda_ms_per_call"
            )
            residual_sparse_ms = row.get(
                f"{bucket_label}_residual_sparse_gemm_cuda_ms_per_call"
            )
            base_rows = row.get(
                f"{bucket_label}_base_sparse_linear_rows_per_call"
            )
            residual_rows = row.get(
                f"{bucket_label}_residual_dense_rows_per_call"
            )
            if not any(
                _as_float(value) is not None
                for value in (
                    base_ms,
                    residual_dense_ms,
                    residual_sparse_ms,
                    base_rows,
                    residual_rows,
                )
            ):
                continue
            out.append(
                {
                    "source_index": idx,
                    "source_root": row.get("source_root", ""),
                    "method": row.get("method", ""),
                    "dataset": row.get("dataset", ""),
                    "batch_size": row.get("batch_size", ""),
                    "max_new_tokens": row.get("max_new_tokens", ""),
                    "row_kind": row.get("row_kind", ""),
                    "bucket": bucket_label,
                    "base_sparse_linear_cuda_ms_per_call": base_ms,
                    "residual_dense_gemm_cuda_ms_per_call": residual_dense_ms,
                    "residual_sparse_gemm_cuda_ms_per_call": residual_sparse_ms,
                    "base_sparse_linear_rows_per_call": base_rows,
                    "residual_dense_rows_per_call": residual_rows,
                    "breakdown_path": row.get("breakdown_path", ""),
                    "work_dir": row.get("work_dir", ""),
                }
            )
    return out


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    def _clean_serving_candidate(row: dict[str, Any]) -> bool:
        return row.get("row_kind") == "clean_serving"

    def _find_row(
        method: str,
        *,
        require_linear: bool = False,
        require_scheduler: bool = False,
        prefer_clean_serving: bool = False,
        prefer_graph: bool = False,
        prefer_request_loop: bool = False,
    ) -> dict[str, Any] | None:
        candidates = [row for row in rows if row.get("method") == method]
        if require_linear:
            candidates = [
                row
                for row in candidates
                if _as_float(row.get("base_sparse_linear_cuda_ms_per_call")) is not None
            ]
        if require_scheduler:
            candidates = [
                row
                for row in candidates
                if _as_float(row.get("scheduler_mask_build_cpu_ms_per_step"))
                is not None
            ]
        if not candidates:
            return None
        if prefer_clean_serving:
            clean = [
                row
                for row in candidates
                if _clean_serving_candidate(row)
            ]
            if clean:
                return max(
                    clean,
                    key=lambda row: _as_float(row.get("full_batch_output_tps"))
                    or float("-inf"),
                )
        if prefer_request_loop:
            loop_candidates = [
                row
                for row in candidates
                if _as_float(
                    row.get("scheduler_request_routing_loop_cpu_ms_per_call")
                )
                is not None
            ]
            if loop_candidates:
                return loop_candidates[-1]
        graph_candidates = [
            row
            for row in candidates
            if "FULL" in str(row.get("cudagraph_mode_counts", ""))
        ]
        clean_graph_candidates = [
            row for row in graph_candidates if _clean_serving_candidate(row)
        ]
        if prefer_graph and clean_graph_candidates:
            return clean_graph_candidates[-1]
        if prefer_graph and graph_candidates:
            return graph_candidates[-1]
        # When the caller passes several roots for the same method, later roots
        # are usually the newer variant in the ablation. Summarize that latest
        # comparable row while still keeping every row in the detailed tables.
        return graph_candidates[-1] if graph_candidates else candidates[-1]

    def _value(row: dict[str, Any] | None, key: str, digits: int = 3) -> str:
        if row is None:
            return ""
        return _fmt(row.get(key), digits)

    def _raw(row: dict[str, Any] | None, key: str) -> str:
        if row is None:
            return ""
        value = row.get(key, "")
        return "" if value is None else str(value)

    def _gather_value(row: dict[str, Any] | None) -> str:
        if row is None:
            return ""
        route_all = _fmt(
            row.get("route_all_gather_scatter_cuda_ms_per_linear_call")
        )
        if route_all:
            return f"{route_all} per linear"
        event_avg = _fmt(row.get("gather_scatter_cuda_ms_per_call"))
        return f"{event_avg} per event" if event_avg else ""

    def _fallback_value(row: dict[str, Any] | None) -> str:
        if row is None:
            return ""
        scheduler_steps = _fmt(row.get("scheduler_dense_fallback_steps"), 0)
        adaptive_calls = _fmt(row.get("adaptive_dense_fallback_calls"), 0)
        parts = []
        if scheduler_steps:
            parts.append(f"scheduler={scheduler_steps}")
        if adaptive_calls:
            parts.append(f"adaptive={adaptive_calls}")
        return "; ".join(parts)

    def _fmt_na(row: dict[str, Any] | None, key: str,
                digits: int = 3) -> str:
        value = _value(row, key, digits)
        return value if value else "n/a"

    def _find_routing_row(method: str) -> dict[str, Any] | None:
        candidates = []
        for row in rows:
            if row.get("method") != method:
                continue
            has_rows = any(
                row.get(key) not in (None, "")
                for key in (
                    "draft_residual_rows",
                    "draft_base_rows",
                    "non_draft_residual_rows",
                    "non_draft_base_rows",
                    "bucket_fill_ratio",
                )
            )
            if has_rows:
                candidates.append(row)
        if not candidates:
            return None
        complete_candidates = [
            row
            for row in candidates
            if all(
                row.get(key) not in (None, "")
                for key in (
                    "draft_residual_rows",
                    "draft_base_rows",
                    "non_draft_residual_rows",
                    "non_draft_base_rows",
                )
            )
        ]
        if complete_candidates:
            candidates = complete_candidates
        bucket_candidates = [
            row
            for row in candidates
            if _as_float(row.get("bucket_fill_ratio")) is not None
        ]
        return bucket_candidates[-1] if bucket_candidates else candidates[-1]

    def _same_root_dense_for(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        for candidate in rows:
            if candidate.get("row_kind") != "clean_serving":
                continue
            if candidate.get("method") not in {"dense_baseline", "vllm_eagle3"}:
                continue
            if candidate.get("source_root") != row.get("source_root"):
                continue
            if candidate.get("dataset") != row.get("dataset"):
                continue
            if candidate.get("batch_size") != row.get("batch_size"):
                continue
            if candidate.get("max_new_tokens") != row.get("max_new_tokens"):
                continue
            return candidate
        return None

    dense_serving_row = _find_row("dense_baseline", prefer_clean_serving=True)
    if dense_serving_row is None:
        dense_serving_row = _find_row("vllm_eagle3", prefer_clean_serving=True)
    speclink_serving_row = _find_row("speclink_t08", prefer_clean_serving=True)
    paired_dense_serving_row = _same_root_dense_for(speclink_serving_row)
    if paired_dense_serving_row is not None:
        dense_serving_row = paired_dense_serving_row
    dense_graph_row = _find_row("dense_baseline", prefer_graph=True)
    if dense_graph_row is None:
        dense_graph_row = _find_row("vllm_eagle3", prefer_graph=True)
    speclink_graph_row = _find_row("speclink_t08", prefer_graph=True)
    base_only_row = _find_row("base_only_24")
    base_only_scheduler_row = _find_row("base_only_24", require_scheduler=True)
    speclink_scheduler_row = _find_row(
        "speclink_t08",
        require_scheduler=True,
        prefer_request_loop=True,
    )
    speclink_clean_scheduler_rows = [
        row
        for row in rows
        if row.get("method") == "speclink_t08"
        and row.get("row_kind") == "clean_serving"
        and _as_float(row.get("scheduler_mask_build_cpu_ms_per_step")) is not None
    ]
    speclink_clean_scheduler_row = (
        speclink_clean_scheduler_rows[-1]
        if speclink_clean_scheduler_rows
        else None
    )
    speclink_linear_row = _find_row("speclink_t08", require_linear=True)
    speclink_routing_row = _find_routing_row("speclink_t08")
    residual_ms = _residual_correction_ms(speclink_linear_row)

    speclink_routing = ""
    if speclink_routing_row is not None:
        bucket_fill = _fmt_na(speclink_routing_row, "bucket_fill_ratio")
        residual_row_fraction = _fmt_na(
            speclink_routing_row,
            "residual_dense_rows_fraction",
        )
        speclink_routing = (
            f"draft {_fmt(speclink_routing_row.get('draft_residual_rows'), 0)}/"
            f"{_fmt(speclink_routing_row.get('draft_base_rows'), 0)}, "
            f"non-draft {_fmt(speclink_routing_row.get('non_draft_residual_rows'), 0)}/"
            f"{_fmt(speclink_routing_row.get('non_draft_base_rows'), 0)}, "
            f"bucket fill {bucket_fill}, correction row fraction "
            f"{residual_row_fraction}"
        )
    elif speclink_serving_row is not None:
        speclink_routing = (
            f"non-draft residual/base "
            f"{_fmt(speclink_serving_row.get('non_draft_residual_rows'), 0)}/"
            f"{_fmt(speclink_serving_row.get('non_draft_base_rows'), 0)}"
        )

    clean_speedup = _as_float(
        speclink_serving_row.get("full_batch_speedup_vs_same_root_dense")
        if speclink_serving_row is not None
        else None
    )
    gpu_util_delta = None
    if dense_serving_row is not None and speclink_serving_row is not None:
        dense_util = _as_float(dense_serving_row.get("avg_gpu_util_pct"))
        sr24_util = _as_float(speclink_serving_row.get("avg_gpu_util_pct"))
        if dense_util is not None and sr24_util is not None:
            gpu_util_delta = sr24_util - dense_util
    sr24_none_frac = _as_float(
        speclink_graph_row.get("cudagraph_none_fraction")
        if speclink_graph_row is not None
        else None
    )
    interpretation = []
    if clean_speedup is not None:
        if clean_speedup < 0.98:
            interpretation.append(
                "Clean serving SR24 is slower than dense, so the bottleneck is "
                "not just a diagnostic-profiler artifact."
            )
        elif clean_speedup < 1.05:
            interpretation.append(
                "Clean serving SR24 is roughly tied with dense; this is useful "
                "for ablation, but still far from a material speedup."
            )
        else:
            interpretation.append(
                "Clean serving SR24 is faster than dense in this selected row; "
                "quality and repeatability are the next gates."
            )
    if gpu_util_delta is not None and abs(gpu_util_delta) <= 5.0:
        interpretation.append(
            "GPU utilization is similar between dense and SR24, so the current "
            "clean-serving gap is more consistent with inefficient mixed "
            "operators or extra useful-work cost than with a mostly idle GPU."
        )
    if sr24_none_frac is not None and sr24_none_frac <= 0.05:
        interpretation.append(
            "SR24 has low CUDA-Graph NONE fraction in the selected graph row; "
            "CUDA Graph coverage is not the first suspect."
        )
    clean_scheduler_ms = _as_float(
        speclink_clean_scheduler_row.get("scheduler_mask_build_cpu_ms_per_step")
        if speclink_clean_scheduler_row is not None
        else None
    )
    diag_scheduler_loop_ms = _as_float(
        speclink_scheduler_row.get("scheduler_request_routing_loop_cpu_ms_per_call")
        if speclink_scheduler_row is not None
        else None
    )
    if clean_scheduler_ms is not None and clean_scheduler_ms <= 1.0:
        interpretation.append(
            "The low-sync clean scheduler path is already sub-ms per step in "
            "the selected row; large scheduler numbers mostly come from "
            "exact-routing or sync diagnostics."
        )
    if diag_scheduler_loop_ms is not None and diag_scheduler_loop_ms > 1.0:
        interpretation.append(
            "Diagnostic rows still show CPU request routing/mask build as a "
            "large sync-heavy cost; treat it as a separate ablation from clean "
            "throughput."
        )
    if _as_float(
        speclink_linear_row.get("base_sparse_linear_cuda_ms_per_call")
        if speclink_linear_row is not None
        else None
    ):
        interpretation.append(
            "Linear diagnostics localize the GPU-side cost to the sparse base "
            "matmul plus residual correction path; gather/scatter is secondary "
            "unless future rows contradict it."
        )

    lines: list[str] = [
        "# SR24 Component Breakdown Summary",
        "",
            "This summary joins serving metrics from `summary.csv` with SR24 component",
            "breakdown fields from each run's `speclink_sr24_breakdown.json`.",
            "",
            "Use graph-on low-sync rows for serving behavior. Use eager linear rows",
            "only to localize component time inside SR24 Linear hooks.",
            "Scheduler and linear micro-timings are diagnostic and add overhead.",
            "",
            "## Short Read",
            "",
    ]
    for item in interpretation:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Bottleneck Diagnosis",
            "",
            "| part | measured item | current value | diagnosis |",
            "| --- | --- | ---: | --- |",
            "| clean serving throughput | full-batch output tokens/s without per-linear CUDA timing | "
            f"dense {_value(dense_serving_row, 'full_batch_output_tps')} tok/s, "
            f"SR24 {_value(speclink_serving_row, 'full_batch_output_tps')} tok/s "
            f"({_value(speclink_serving_row, 'full_batch_speedup_vs_same_root_dense')}x same-root dense) | "
            "This is the end-to-end number to optimize; component rows below are diagnostic and can add overhead. |",
            "| scheduler / mask build, clean path | low-sync per-step residual-mask construction time in graph-capable serving | "
            f"{_value(speclink_clean_scheduler_row, 'scheduler_mask_build_cpu_ms_per_step')} ms/step "
            f"(`base_only_24`: {_value(base_only_scheduler_row, 'scheduler_mask_build_cpu_ms_per_step')} ms/step) | "
            "This is the scheduler cost that should be compared with clean serving throughput. |",
            "| scheduler / mask build, exact diagnostic | per-step residual-mask, bucket, and routing construction with exact stats enabled | "
            f"{_value(speclink_scheduler_row, 'scheduler_mask_build_cpu_ms_per_step')} ms/step | "
            "This localizes CPU/GPU sync and routing overhead, but should not be treated as the clean serving cost. |",
            "| dense fallback | steps that converted a high-residual mixed step to all-residual dense target output | "
            f"{_fallback_value(speclink_scheduler_row)} fallback calls; "
            f"mask states all/mixed/no = {_value(speclink_scheduler_row, 'mask_state_all_residual_steps', 0)}/"
            f"{_value(speclink_scheduler_row, 'mask_state_mixed_steps', 0)}/"
            f"{_value(speclink_scheduler_row, 'mask_state_no_residual_steps', 0)} | "
            "This is accuracy-conservative, but if it requires CPU mask-state sync it can erase any throughput benefit. |",
            "| request routing loop, exact diagnostic | CPU time inside the per-request routing loop | "
            f"{_value(speclink_scheduler_row, 'scheduler_request_routing_loop_cpu_ms_per_call')} ms/step | "
            "This separates routing itself from mask init/topk; it is usually the first thing to eliminate with a batched GPU builder. |",
            "| base sparse linear | SR24 sparse-base matmul time for the selected leafs | "
            f"{_value(speclink_linear_row, 'base_sparse_linear_cuda_ms_per_call')} ms/call | "
            "If this dominates, the sparse base pass is not cheap enough to compensate for the extra correction path. |",
            "| residual correction | dense/sparse/compressed residual correction GEMM time | "
            f"{residual_ms} ms/call | "
            "This is the selected-row correction cost; compare it with base sparse before optimizing gather/scatter. |",
            "| gather/scatter | index_select, delta compute, and index_add bucket assembly time | "
            f"{_gather_value(speclink_linear_row)} | "
            "If this stays small, it is not the current main bottleneck. |",
            "| routing statistics | draft/non-draft residual/base rows and bucket fill | "
            f"{speclink_routing} | "
            "Low bucket fill would indicate wasted correction capacity; high fill shifts attention to scheduler and operator cost. |",
            "| CUDA Graph | dense vs SR24 graph-mode counts | "
            f"dense `{_raw(dense_graph_row, 'cudagraph_mode_counts')}`, SR24 `{_raw(speclink_graph_row, 'cudagraph_mode_counts')}` | "
            "Many extra `NONE` steps in SR24 would mean graph coverage is hurting serving throughput. |",
            "| GPU util | sampled average GPU utilization | "
            f"dense {_value(dense_serving_row, 'avg_gpu_util_pct')}%, SR24 {_value(speclink_serving_row, 'avg_gpu_util_pct')}% | "
            "Low SR24 util with similar graph coverage points to CPU wait, small kernels, or an underfilled mixed sparse/residual path. |",
            "",
            "## Requested Breakdown Matrix",
            "",
        "| source | row kind | method | bs | scheduler/mask build ms/step | dense fallback steps | mask all/mixed/no | base sparse ms/call | residual correction ms/call | gather/scatter | routing residual/base rows | correction row frac | bucket fill | CUDA Graph modes | GPU util % | full-batch tok/s | same-root speedup | best-dense speedup |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        residual_ms = _residual_correction_ms(row)
        lines.append(
            "| {idx} | {kind} | `{method}` | {bs} | {scheduler} | {fallback} | {mask_states} | {base} | {residual} | {gather} | {dr}/{db} draft, {nr}/{nb} non-draft | {row_frac} | {bucket_fill} | `{graph}` | {gpu} | {full} | {same_speedup} | {best_speedup} |".format(
                idx=idx,
                kind=row.get("row_kind", ""),
                method=row.get("method", ""),
                bs=row.get("batch_size", ""),
                scheduler=_fmt(row.get("scheduler_mask_build_cpu_ms_per_step")),
                fallback=_fallback_value(row),
                mask_states=(
                    f"{_fmt(row.get('mask_state_all_residual_steps'), 0)}/"
                    f"{_fmt(row.get('mask_state_mixed_steps'), 0)}/"
                    f"{_fmt(row.get('mask_state_no_residual_steps'), 0)}"
                ),
                base=_fmt(row.get("base_sparse_linear_cuda_ms_per_call")),
                residual=_fmt(residual_ms),
                gather=_gather_value(row),
                dr=_fmt(row.get("draft_residual_rows"), 0),
                db=_fmt(row.get("draft_base_rows"), 0),
                nr=_fmt(row.get("non_draft_residual_rows"), 0),
                nb=_fmt(row.get("non_draft_base_rows"), 0),
                row_frac=_fmt(row.get("residual_dense_rows_fraction")),
                bucket_fill=(
                    _fmt(row.get("bucket_fill_ratio"))
                    if _as_float(row.get("bucket_fill_ratio")) is not None
                    else "n/a"
                ),
                graph=row.get("cudagraph_mode_counts", ""),
                gpu=_fmt(row.get("avg_gpu_util_pct")),
                full=_fmt(row.get("full_batch_output_tps")),
                same_speedup=_fmt(
                    row.get("full_batch_speedup_vs_same_root_dense")
                ),
                best_speedup=_fmt(
                    row.get("full_batch_speedup_vs_best_clean_dense")
                ),
            )
        )

    lines.extend(
        [
            "",
            "Read this first: high scheduler ms/step points to request routing, mask",
            "construction, bucket selection, or CPU/GPU synchronization. High base",
            "sparse ms/call means the semi-structured base matmul is the dominant",
            "linear cost. High residual/gather/scatter means the selective correction",
            "path is too expensive. Many `NONE` CUDA Graph steps or low GPU util",
            "usually indicate shape churn or small-kernel underutilization.",
            "",
            "## Throughput, Graph, And Utilization",
        "",
        "| source | row kind | method | bs | total tok/s | full-batch tok/s | same-root speedup | best-dense speedup | GPU util % | TPOT ms | CUDA Graph | FULL frac | NONE frac |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | {kind} | `{method}` | {bs} | {total} | {full} | {same_speedup} | {best_speedup} | {gpu} | {tpot} | `{graph}` | {full_frac} | {none_frac} |".format(
                idx=idx,
                kind=row.get("row_kind", ""),
                method=row.get("method", ""),
                bs=row.get("batch_size", ""),
                total=_fmt(row.get("total_output_tps")),
                full=_fmt(row.get("full_batch_output_tps")),
                same_speedup=_fmt(
                    row.get("full_batch_speedup_vs_same_root_dense")
                ),
                best_speedup=_fmt(
                    row.get("full_batch_speedup_vs_best_clean_dense")
                ),
                gpu=_fmt(row.get("avg_gpu_util_pct")),
                tpot=_fmt(row.get("tpot_ms_mean")),
                graph=row.get("cudagraph_mode_counts", ""),
                full_frac=_fmt(row.get("cudagraph_full_fraction")),
                none_frac=_fmt(row.get("cudagraph_none_fraction")),
            )
        )

    lines.extend(
        [
            "",
            "## Scheduler And Routing",
            "",
            "| source | method | scheduler mask ms/step | bucket topk ms/call | avg scheduled tokens/step | draft residual/base rows | non-draft residual/base rows | draft residual frac | non-draft residual frac | correction row frac | bucket fill | bucket rows active/candidate | bucket actual/requested | bucket actual/requested frac |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{method}` | {mask} | {topk} | {sched} | {dr}/{db} | {nr}/{nb} | {draft_frac} | {non_draft_frac} | {row_frac} | {fill} | {active}/{candidate} | {actual}/{requested} | {actual_frac} |".format(
                idx=idx,
                method=row.get("method", ""),
                mask=_fmt(row.get("scheduler_mask_build_cpu_ms_per_step")),
                topk=_fmt(row.get("scheduler_bucket_topk_cuda_ms_per_call")),
                sched=_fmt(row.get("avg_scheduled_tokens_per_step")),
                dr=_fmt(row.get("draft_residual_rows"), 0),
                db=_fmt(row.get("draft_base_rows"), 0),
                nr=_fmt(row.get("non_draft_residual_rows"), 0),
                nb=_fmt(row.get("non_draft_base_rows"), 0),
                draft_frac=_fmt(row.get("draft_residual_fraction")),
                non_draft_frac=_fmt(row.get("non_draft_residual_fraction")),
                row_frac=_fmt(row.get("residual_dense_rows_fraction")),
                fill=(
                    _fmt(row.get("bucket_fill_ratio"))
                    if _as_float(row.get("bucket_fill_ratio")) is not None
                    else "n/a"
                ),
                active=(
                    _fmt(row.get("avg_bucket_active_rows"))
                    if _as_float(row.get("avg_bucket_active_rows")) is not None
                    else "n/a"
                ),
                candidate=(
                    _fmt(row.get("avg_bucket_candidate_rows"))
                    if _as_float(row.get("avg_bucket_candidate_rows")) is not None
                    else "n/a"
                ),
                actual=(
                    _fmt(row.get("sr24_bucket_active_rows"), 0)
                    if _as_float(row.get("sr24_bucket_active_rows")) is not None
                    else "n/a"
                ),
                requested=(
                    _fmt(row.get("sr24_bucket_residual_requested_rows"), 0)
                    if _as_float(row.get("sr24_bucket_residual_requested_rows"))
                    is not None
                    else "n/a"
                ),
                actual_frac=(
                    _fmt(row.get("sr24_bucket_active_fraction_of_requested"))
                    if _as_float(row.get("sr24_bucket_active_fraction_of_requested"))
                    is not None
                    else "n/a"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Scheduler Mask-Build Breakdown",
            "",
            "| source | method | counts CPU | mask init CPU/CUDA | pending pop CPU | request loop CPU | score policy CUDA | mask write CUDA | mask-state sync CPU/CUDA | bucket build CPU/topk CUDA | direct position CPU/CUDA | direct position vector/builds | direct position rows/build |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{method}` | {counts} | {mask_cpu}/{mask_cuda} | {pending} | {loop} | {score_cuda} | {write_cuda} | {sync_cpu}/{sync_cuda} | {bucket_cpu}/{bucket_cuda} | {direct_cpu}/{direct_cuda} | {direct_vector}/{direct_builds} | {direct_rows} |".format(
                idx=idx,
                method=row.get("method", ""),
                counts=_fmt(
                    row.get("scheduler_materialize_counts_cpu_ms_per_call")
                ),
                mask_cpu=_fmt(row.get("scheduler_mask_init_cpu_ms_per_call")),
                mask_cuda=_fmt(row.get("scheduler_mask_init_cuda_ms_per_call")),
                pending=_fmt(
                    row.get("scheduler_pending_scores_pop_cpu_ms_per_call")
                ),
                loop=_fmt(
                    row.get("scheduler_request_routing_loop_cpu_ms_per_call")
                ),
                score_cuda=_fmt(
                    row.get("scheduler_score_policy_cuda_ms_per_call")
                ),
                write_cuda=_fmt(row.get("scheduler_mask_write_cuda_ms_per_call")),
                sync_cpu=_fmt(
                    row.get("scheduler_mask_state_sync_cpu_ms_per_call")
                ),
                sync_cuda=_fmt(row.get("scheduler_mask_state_sum_cuda_ms_per_call")),
                bucket_cpu=_fmt(row.get("scheduler_bucket_build_cpu_ms_per_call")),
                bucket_cuda=_fmt(row.get("scheduler_bucket_topk_cuda_ms_per_call")),
                direct_cpu=_fmt(
                    row.get("scheduler_direct_position_bucket_cpu_ms_per_call")
                ),
                direct_cuda=_fmt(
                    row.get("scheduler_direct_position_bucket_cuda_ms_per_call")
                ),
                direct_vector=_fmt(
                    row.get("scheduler_direct_position_bucket_vector_builds"), 0
                ),
                direct_builds=_fmt(
                    row.get("scheduler_direct_position_bucket_builds"), 0
                ),
                direct_rows=_fmt(
                    row.get("scheduler_direct_position_bucket_rows_per_build"), 0
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Clean Runtime Scheduler Wall Time",
            "",
            "These values come from SR24 runtime stats and use CPU wall-clock timers only; they do not create CUDA events or force tensor synchronization.",
            "",
            "| source | method | total mask wall ms/step | counts | pending pop | batched builder | request loop | batch-all apply | mask-state | static copy | bucket/index | bucket build | mixed row indices |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{method}` | {total} | {counts} | {pending} | {batched} | {loop} | {batch_all} | {state} | {static_copy} | {bucket} | {bucket_build} | {row_indices} |".format(
                idx=idx,
                method=row.get("method", ""),
                total=_fmt(row.get("scheduler_mask_wall_cpu_ms_per_step")),
                counts=_fmt(
                    row.get("scheduler_materialize_counts_wall_cpu_ms_per_step")
                ),
                pending=_fmt(
                    row.get("scheduler_pending_scores_pop_wall_cpu_ms_per_step")
                ),
                batched=_fmt(
                    row.get("scheduler_batched_mask_builder_wall_cpu_ms_per_step")
                ),
                loop=_fmt(
                    row.get("scheduler_request_routing_loop_wall_cpu_ms_per_step")
                ),
                batch_all=_fmt(
                    row.get("scheduler_batch_all_apply_wall_cpu_ms_per_step")
                ),
                state=_fmt(row.get("scheduler_mask_state_wall_cpu_ms_per_step")),
                static_copy=_fmt(
                    row.get("scheduler_static_mask_copy_wall_cpu_ms_per_step")
                ),
                bucket=_fmt(
                    row.get("scheduler_row_index_bucket_wall_cpu_ms_per_step")
                ),
                bucket_build=_fmt(
                    row.get("scheduler_residual_bucket_wall_cpu_ms_per_step")
                ),
                row_indices=_fmt(
                    row.get("scheduler_mixed_row_indices_wall_cpu_ms_per_step")
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Batched Mask Builder",
            "",
            "| source | method | batched steps | uniform steps | indexed steps | GPU-count steps | GPU-count fallback steps | direct score-row steps | setup CPU | tensor setup CUDA | score rows CUDA | score matrix CUDA | uniform kernel | batched kernel | indexed kernel | req kernel | req indexed kernel |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{method}` | {steps} | {uniform_steps} | {indexed_steps} | {gpu_count_steps} | {gpu_count_fallback_steps} | {direct_score_steps} | {setup} | {tensor_setup} | {score_rows} | {score_matrix} | {uniform_kernel} | {kernel} | {indexed_kernel} | {req_kernel} | {req_indexed_kernel} |".format(
                idx=idx,
                method=row.get("method", ""),
                steps=_fmt(row.get("batched_mask_builder_steps"), 0),
                uniform_steps=_fmt(
                    row.get("batched_mask_builder_uniform_direct_steps"), 0
                ),
                indexed_steps=_fmt(row.get("batched_mask_builder_indexed_steps"), 0),
                gpu_count_steps=_fmt(
                    row.get("batched_mask_builder_gpu_count_steps"), 0
                ),
                gpu_count_fallback_steps=_fmt(
                    row.get(
                        "batched_mask_builder_gpu_count_indexed_fallback_steps"
                    ),
                    0,
                ),
                direct_score_steps=_fmt(
                    row.get("batched_mask_builder_direct_score_rows_steps"), 0
                ),
                setup=_fmt(row.get("scheduler_batched_mask_setup_cpu_ms_per_step")),
                tensor_setup=_fmt(
                    row.get("scheduler_batched_mask_tensor_setup_cuda_ms_per_step")
                ),
                score_rows=_fmt(
                    row.get(
                        "scheduler_batched_mask_score_rows_setup_cuda_ms_per_step"
                    )
                ),
                score_matrix=_fmt(
                    row.get(
                        "scheduler_batched_mask_score_matrix_setup_cuda_ms_per_step"
                    )
                ),
                uniform_kernel=_fmt(
                    row.get(
                        "scheduler_batched_mask_uniform_direct_kernel_cuda_ms_per_step"
                    )
                ),
                kernel=_fmt(row.get("scheduler_batched_mask_kernel_cuda_ms_per_step")),
                indexed_kernel=_fmt(
                    row.get("scheduler_batched_mask_indexed_kernel_cuda_ms_per_step")
                ),
                req_kernel=_fmt(
                    row.get("scheduler_batched_mask_req_kernel_cuda_ms_per_step")
                ),
                req_indexed_kernel=_fmt(
                    row.get(
                        "scheduler_batched_mask_req_indexed_kernel_cuda_ms_per_step"
                    )
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Linear Components",
            "",
        "| source | method | base sparse ms/call | residual dense GEMM ms/call | residual sparse GEMM ms/call | bucket Triton dense GEMM/scatter ms/call | bucket Triton override ms/call | compressed residual ms/event | compressed Triton ms/call | compressed materialize ms/call | compressed GEMM ms/call | cached weight hit/miss | route-all gather/scatter ms/linear | gather/scatter ms/event | base rows/call | residual rows/call | dense bucket rows/call | bucket Triton rows/call |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{method}` | {base} | {residual} | {residual_sparse} | {bucket_triton_dense} | {bucket_override} | {compressed} | {triton_residual} | {materialize} | {compressed_gemm} | {hits}/{misses} | {route_all_gather} | {gather} | {base_rows} | {residual_rows} | {dense_rows} | {bucket_triton_rows} |".format(
                idx=idx,
                method=row.get("method", ""),
                base=_fmt(row.get("base_sparse_linear_cuda_ms_per_call")),
                residual=_fmt(row.get("residual_dense_gemm_cuda_ms_per_call")),
                residual_sparse=_fmt(
                    row.get("residual_sparse_gemm_cuda_ms_per_call")
                ),
                bucket_triton_dense=_fmt(
                    row.get(
                        "bucket_triton_dense_gemm_scatter_cuda_ms_per_call"
                    )
                ),
                bucket_override=_fmt(
                    row.get("bucket_triton_override_cuda_ms_per_call")
                ),
                compressed=_fmt(row.get("compressed_residual_cuda_ms_per_event")),
                triton_residual=_fmt(
                    row.get("compressed_residual_triton_cuda_ms_per_call")
                ),
                materialize=_fmt(
                    row.get("compressed_residual_materialize_cuda_ms_per_call")
                ),
                compressed_gemm=_fmt(
                    row.get("compressed_residual_gemm_cuda_ms_per_call")
                ),
                hits=_fmt(
                    row.get("compressed_residual_cached_weight_hits"), 0
                ),
                misses=_fmt(
                    row.get("compressed_residual_cached_weight_misses"), 0
                ),
                route_all_gather=_fmt(
                    row.get("route_all_gather_scatter_cuda_ms_per_linear_call")
                ),
                gather=_fmt(row.get("gather_scatter_cuda_ms_per_call")),
                base_rows=_fmt(row.get("base_sparse_linear_rows_per_call")),
                residual_rows=_fmt(row.get("residual_dense_rows_per_call")),
                dense_rows=_fmt(row.get("dense_rows_bucket_rows_per_call")),
                bucket_triton_rows=_fmt(
                    row.get("bucket_triton_dense_gemm_scatter_rows_per_call")
                ),
            )
        )

    per_leaf_rows = _make_per_leaf_rows(rows)
    if per_leaf_rows:
        lines.extend(
            [
                "",
                "## Per-Leaf Linear Components",
                "",
                "| source | method | bucket | base sparse ms/call | residual dense GEMM ms/call | residual sparse GEMM ms/call | base rows/call | residual rows/call |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in per_leaf_rows:
            lines.append(
                "| {idx} | `{method}` | `{bucket}` | {base_ms} | {residual_dense_ms} | {residual_sparse_ms} | {base_rows} | {residual_rows} |".format(
                    idx=row.get("source_index", ""),
                    method=row.get("method", ""),
                    bucket=row.get("bucket", ""),
                    base_ms=_fmt(
                        row.get("base_sparse_linear_cuda_ms_per_call")
                    ),
                    residual_dense_ms=_fmt(
                        row.get("residual_dense_gemm_cuda_ms_per_call")
                    ),
                    residual_sparse_ms=_fmt(
                        row.get("residual_sparse_gemm_cuda_ms_per_call")
                    ),
                    base_rows=_fmt(
                        row.get("base_sparse_linear_rows_per_call")
                    ),
                    residual_rows=_fmt(
                        row.get("residual_dense_rows_per_call")
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Row-Routed MLP Components",
            "",
            "| source | method | calls | reuse calls | skipped | total ms/call | reuse total | reuse base | reuse dense | route | dense gather | dense gate_up | dense act | dense down | base gather | base gate_up | reuse base gate_up | reuse base act | sparse down | reuse sparse down | assemble | reuse copy | rows/base/dense per call | reuse rows/dense per call | skipped dense/total rows |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{method}` | {calls} | {reuse_calls} | {skipped} | {total} | {reuse_total} | {reuse_base} | {reuse_dense} | {route} | {dense_gather} | {dense_gate_up} | {dense_act} | {dense_down} | {base_gather} | {base_gate_up} | {reuse_base_gate_up} | {reuse_base_act} | {down} | {reuse_down} | {assemble} | {reuse_copy} | {rows}/{base_rows}/{dense_rows} | {reuse_rows}/{reuse_dense_rows} | {skipped_dense}/{skipped_rows} |".format(
                idx=idx,
                method=row.get("method", ""),
                calls=_fmt(row.get("row_routed_mlp_calls"), 0),
                reuse_calls=_fmt(
                    row.get("row_routed_mlp_reuse_base_output_calls"), 0
                ),
                skipped=_fmt(row.get("row_routed_mlp_skipped_small_dense_rows"), 0),
                total=_fmt(row.get("row_routed_mlp_cuda_ms_per_call")),
                reuse_total=_fmt(
                    row.get("row_routed_mlp_reuse_base_cuda_ms_per_call")
                ),
                reuse_base=_fmt(
                    row.get("row_routed_mlp_reuse_base_base_cuda_ms_per_call")
                ),
                reuse_dense=_fmt(
                    row.get("row_routed_mlp_reuse_base_dense_cuda_ms_per_call")
                ),
                route=_fmt(row.get("row_routed_mlp_route_build_cuda_ms_per_call")),
                dense_gather=_fmt(
                    row.get("row_routed_mlp_dense_gather_cuda_ms_per_call")
                ),
                dense_gate_up=_fmt(
                    row.get("row_routed_mlp_dense_gate_up_cuda_ms_per_call")
                ),
                dense_act=_fmt(row.get("row_routed_mlp_dense_act_cuda_ms_per_call")),
                dense_down=_fmt(
                    row.get("row_routed_mlp_dense_down_cuda_ms_per_call")
                ),
                base_gather=_fmt(
                    row.get("row_routed_mlp_base_gather_cuda_ms_per_call")
                ),
                base_gate_up=_fmt(
                    row.get("row_routed_mlp_base_gate_up_cuda_ms_per_call")
                ),
                reuse_base_gate_up=_fmt(
                    row.get("row_routed_mlp_reuse_base_gate_up_cuda_ms_per_call")
                ),
                reuse_base_act=_fmt(
                    row.get("row_routed_mlp_reuse_base_act_cuda_ms_per_call")
                ),
                cat=_fmt(row.get("row_routed_mlp_gate_up_cat_cuda_ms_per_call")),
                act=_fmt(row.get("row_routed_mlp_act_cuda_ms_per_call")),
                down=_fmt(row.get("row_routed_mlp_down_sparse_cuda_ms_per_call")),
                reuse_down=_fmt(
                    row.get(
                        "row_routed_mlp_reuse_base_down_sparse_cuda_ms_per_call"
                    )
                ),
                assemble=_fmt(row.get("row_routed_mlp_assemble_cuda_ms_per_call")),
                reuse_copy=_fmt(
                    row.get("row_routed_mlp_reuse_base_index_copy_cuda_ms_per_call")
                ),
                rows=_fmt(row.get("row_routed_mlp_rows_per_call")),
                base_rows=_fmt(row.get("row_routed_mlp_base_rows_per_call")),
                dense_rows=_fmt(row.get("row_routed_mlp_dense_rows_per_call")),
                reuse_rows=_fmt(
                    row.get("row_routed_mlp_reuse_base_rows_per_call")
                ),
                reuse_dense_rows=_fmt(
                    row.get("row_routed_mlp_reuse_base_dense_rows_per_call")
                ),
                skipped_dense=_fmt(
                    row.get("row_routed_mlp_skipped_dense_rows_per_skip")
                ),
                skipped_rows=_fmt(
                    row.get("row_routed_mlp_skipped_total_rows_per_skip")
                ),
            )
        )

    lines.extend(["", "## Sources", ""])
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx}. root: `{row['source_root']}`")
        if row.get("breakdown_path"):
            lines.append(f"   breakdown: `{row['breakdown_path']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        type=Path,
        help="One or more final result roots containing summary.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where breakdown_summary.csv and report.md are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for root_arg in args.roots:
        root = _resolve_path(root_arg)
        for summary_row in _read_summary_rows(root):
            if not summary_row.get("method"):
                continue
            if _is_already_summarized_row(summary_row):
                row = dict(summary_row)
                row.setdefault("source_root", str(root.resolve()))
                rows.append(row)
            else:
                rows.append(_summarize_row(root, summary_row))

    if not rows:
        raise SystemExit("no summary rows found")

    _attach_relative_speedups(rows)
    _write_csv(output_root / "breakdown_summary.csv", rows)
    per_leaf_rows = _make_per_leaf_rows(rows)
    if per_leaf_rows:
        _write_csv(output_root / "per_leaf_linear_breakdown.csv", per_leaf_rows)
    _write_report(output_root / "report.md", rows)
    print(output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
