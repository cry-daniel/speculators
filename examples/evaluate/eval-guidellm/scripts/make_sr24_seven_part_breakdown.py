#!/usr/bin/env python3
"""Build a seven-part SR24 slowdown breakdown from existing artifacts.

This is an offline reducer. It does not launch vLLM and does not add timing
overhead to serving. Point it at one or more result roots that contain
`summary.csv`, `median_summary.csv`, or `breakdown_summary.csv`; if the root has
per-run `speclink_sr24_breakdown.json` files, component timings are joined in.

Example:
  conda run -n spec python scripts/make_sr24_seven_part_breakdown.py \
    --roots results.bak/sr24_user_requested_breakdown_bs64_math_k8_nosync_20260625 \
    --output-root results.bak/sr24_seven_part_breakdown_TIMESTAMP
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import summarize_sr24_breakdown as sr24_summary


EVAL_ROOT = Path(__file__).resolve().parents[1]


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
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


def _fmt_int(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return str(int(round(number)))


def _parse_graph_counts(value: Any) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        count = _as_float(raw_count)
        if count is None or count <= 0:
            continue
        counts[str(key)] = int(round(count))
    return counts


def _fmt_graph_counts(value: Any) -> str:
    counts = _parse_graph_counts(value)
    if not counts:
        return ""
    return json.dumps(counts, sort_keys=True)


def _graph_fraction_from_counts(value: Any, mode: str) -> float | None:
    counts = _parse_graph_counts(value)
    total = sum(counts.values())
    if total <= 0:
        return None
    return counts.get(mode, 0) / total


def _row_graph_counts_value(row: dict[str, Any]) -> Any:
    counts = _parse_graph_counts(row.get("cudagraph_mode_counts"))
    if counts:
        return counts
    return row.get("server_cudagraph_profile_counts")


def _kv(label: str, value: Any, suffix: str = "", digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{label}={number:.{digits}f}{suffix}"


def _kv_int(label: str, value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{label}={int(round(number))}"


def _compact_join(parts: list[str]) -> str:
    return ", ".join(part for part in parts if part)


def _ratio(numer: Any, denom: Any) -> float | None:
    n = _as_float(numer)
    d = _as_float(denom)
    if n is None or d in (None, 0.0):
        return None
    return n / d


def _sum_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    total = 0.0
    seen = False
    for key in keys:
        value = _as_float(row.get(key))
        if value is None:
            continue
        total += value
        seen = True
    return total if seen else None


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _resolve_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return EVAL_ROOT / path


def _load_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_arg in roots:
        root = _resolve_root(root_arg)
        for summary_row in sr24_summary._read_summary_rows(root):
            if not summary_row.get("method"):
                continue
            if sr24_summary._is_already_summarized_row(summary_row):
                row = dict(summary_row)
                row.setdefault("source_root", str(root.resolve()))
                if not row.get("total_output_tps"):
                    row["total_output_tps"] = row.get(
                        "total_output_tokens_per_second", ""
                    )
                if not row.get("full_batch_output_tps"):
                    row["full_batch_output_tps"] = row.get(
                        "full_batch_output_tokens_per_second", ""
                    )
                rows.append(row)
            else:
                rows.append(sr24_summary._summarize_row(root, summary_row))
    if not rows:
        raise SystemExit("no summary rows found")
    rows = _dedupe_rows(rows)
    sr24_summary._attach_relative_speedups(rows)
    _attach_same_root_total_speedups(rows)
    _attach_clean_runtime_graph(rows)
    return rows


def _graph_key(row: dict[str, Any], *, include_max_tokens: bool) -> tuple[str, ...]:
    key = (
        str(row.get("method", "")),
        str(row.get("dataset", "")),
        str(row.get("batch_size", "")),
    )
    if include_max_tokens:
        return (*key, str(row.get("max_new_tokens", "")))
    return key


def _attach_clean_runtime_graph(rows: list[dict[str, Any]]) -> None:
    """Copy clean runtime CUDA Graph stats onto matching clean serving rows.

    Matrix roots often keep the clean throughput row and the lower-overhead
    runtime-stat row as separate rows. Joining the graph stats by shape keeps
    the representative table readable without changing throughput selection.
    """
    exact_graph_rows: dict[tuple[str, ...], dict[str, Any]] = {}
    loose_graph_rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        if not _is_clean_like_serving(row):
            continue
        counts = _parse_graph_counts(_row_graph_counts_value(row))
        if not counts:
            continue
        source_root = str(row.get("source_root", ""))
        for key, graph_rows in (
            (_graph_key(row, include_max_tokens=True), exact_graph_rows),
            (_graph_key(row, include_max_tokens=False), loose_graph_rows),
        ):
            current = graph_rows.get(key)
            if current is None or "/clean_runtime_stats" in source_root:
                graph_rows[key] = row

    for row in rows:
        if _parse_graph_counts(_row_graph_counts_value(row)):
            continue
        source = exact_graph_rows.get(
            _graph_key(row, include_max_tokens=True)
        ) or loose_graph_rows.get(_graph_key(row, include_max_tokens=False))
        if source is None:
            continue
        counts = _parse_graph_counts(_row_graph_counts_value(source))
        if not counts:
            continue
        row["cudagraph_mode_counts"] = json.dumps(counts, sort_keys=True)
        for key in (
            "cudagraph_full_fraction",
            "cudagraph_none_fraction",
            "cudagraph_full_steps",
            "cudagraph_none_steps",
            "cudagraph_total_steps",
        ):
            if not row.get(key) and source.get(key) not in (None, ""):
                row[key] = source.get(key)
        if not row.get("cudagraph_full_fraction"):
            row["cudagraph_full_fraction"] = _graph_fraction_from_counts(
                counts, "FULL")
        if not row.get("cudagraph_none_fraction"):
            row["cudagraph_none_fraction"] = _graph_fraction_from_counts(
                counts, "NONE")


def _load_operator_microbench_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "rows",
        "out_features",
        "in_features",
        "residual_fraction",
        "dense_graph_ms",
        "base_sparse_graph_ms",
        "bucket_delta_inplace_graph_ms",
    }
    for root_arg in roots:
        root = _resolve_root(root_arg)
        summary_path = root / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(
                set(reader.fieldnames)
            ):
                continue
            for row in reader:
                row = dict(row)
                row["source_root"] = str(root.resolve())
                row["row_kind"] = "operator_microbench"
                row["method"] = "component_microbench"
                rows.append(row)
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate rows when a raw root and a derived summary root are mixed.

    The seven-part reducer can consume both raw matrix roots and the output of
    summarize_sr24_breakdown.py. Passing both is harmless for the numbers but
    makes the report repeat every row, which obscures the actual slowdown read.
    Prefer the first occurrence because raw roots preserve the clearest source.
    """
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("source_root", "")),
            str(row.get("method", "")),
            str(row.get("dataset", "")),
            str(row.get("batch_size", "")),
            str(row.get("max_new_tokens", "")),
            str(row.get("row_kind", "")),
            str(row.get("work_dir", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _attach_same_root_total_speedups(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("method") not in {"dense_baseline", "vllm_eagle3"}:
            continue
        key = (
            str(row.get("source_root", "")),
            str(row.get("dataset", "")),
            str(row.get("batch_size", "")),
            str(row.get("max_new_tokens", "")),
        )
        current = baselines.get(key)
        if current is None or (
            (_as_float(row.get("total_output_tps")) or float("-inf"))
            > (_as_float(current.get("total_output_tps")) or float("-inf"))
        ):
            baselines[key] = row

    for row in rows:
        key = (
            str(row.get("source_root", "")),
            str(row.get("dataset", "")),
            str(row.get("batch_size", "")),
            str(row.get("max_new_tokens", "")),
        )
        baseline = baselines.get(key)
        if baseline is None:
            continue
        row.setdefault(
            "same_root_dense_reference_total_tps",
            baseline.get("total_output_tps", ""),
        )
        if not row.get("total_tps_speedup_vs_same_root_dense"):
            row["total_tps_speedup_vs_same_root_dense"] = _ratio(
                row.get("total_output_tps"), baseline.get("total_output_tps")
            )


def _part_value(row: dict[str, Any], part: str) -> str:
    if part == "scheduler_mask":
        return _compact_join(
            [
                _kv(
                    "wall",
                    row.get("scheduler_mask_wall_cpu_ms_per_step"),
                    "ms/step",
                ),
                _kv(
                    "mask",
                    row.get("scheduler_mask_build_cpu_ms_per_step"),
                    "ms/step",
                ),
                _kv(
                    "wall_loop",
                    row.get("scheduler_request_routing_loop_wall_cpu_ms_per_step"),
                    "ms",
                ),
                _kv(
                    "row_bucket",
                    row.get("scheduler_row_index_bucket_wall_cpu_ms_per_step"),
                    "ms",
                ),
                _kv(
                    "bucket_build",
                    row.get("scheduler_residual_bucket_wall_cpu_ms_per_step"),
                    "ms",
                ),
                _kv(
                    "row_indices",
                    row.get("scheduler_mixed_row_indices_wall_cpu_ms_per_step"),
                    "ms",
                ),
                _kv(
                    "direct_cpu_rows",
                    row.get(
                        "scheduler_direct_cpu_route_rows_wall_cpu_ms_per_step"
                    ),
                    "ms",
                ),
                _kv(
                    "batched_req",
                    row.get("scheduler_batched_mask_req_kernel_cuda_ms_per_step"),
                    "ms",
                ),
                _kv(
                    "indexed",
                    row.get(
                        "scheduler_batched_mask_indexed_kernel_cuda_ms_per_step"
                    ),
                    "ms",
                ),
                _kv("topk", row.get("scheduler_bucket_topk_cuda_ms_per_call"), "ms"),
                _kv(
                    "sync",
                    row.get("scheduler_mask_state_sync_cpu_ms_per_call"),
                    "ms",
                ),
                _kv(
                    "request_loop",
                    row.get("scheduler_request_routing_loop_cpu_ms_per_call"),
                    "ms",
                ),
            ]
        )
    if part == "base_sparse":
        gate_up_base_ms = row.get("row_routed_gate_up_base_sparse_cuda_ms_per_call")
        if _as_float(gate_up_base_ms) is not None:
            return _compact_join(
                [
                    _kv("row_routed_gate_up_base_sparse", gate_up_base_ms, "ms/call"),
                    _kv(
                        "base_gather",
                        row.get("row_routed_gate_up_base_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "base_rows/call",
                        row.get("row_routed_gate_up_base_rows_per_call"),
                    ),
                    _kv(
                        "total_rows/call",
                        row.get("row_routed_gate_up_rows_per_call"),
                    ),
                ]
            )
        routed_base_ms = _sum_float(
            row,
            (
                "row_routed_mlp_base_gather_cuda_ms_per_call",
                "row_routed_mlp_base_gate_up_cuda_ms_per_call",
                "row_routed_mlp_base_down_sparse_cuda_ms_per_call",
                "row_routed_mlp_down_sparse_cuda_ms_per_call",
            ),
        )
        if routed_base_ms is not None:
            return _compact_join(
                [
                    _kv("row_routed_base_total", routed_base_ms, "ms/call"),
                    _kv(
                        "base_gather",
                        row.get("row_routed_mlp_base_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "base_gate_up",
                        row.get("row_routed_mlp_base_gate_up_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "base_down_sparse",
                        _first_float(
                            row,
                            (
                                "row_routed_mlp_base_down_sparse_cuda_ms_per_call",
                                "row_routed_mlp_down_sparse_cuda_ms_per_call",
                            ),
                        ),
                        "ms",
                    ),
                    _kv(
                        "base_rows/call",
                        row.get("row_routed_mlp_base_rows_per_call"),
                    ),
                ]
            )
        reuse_base_ms = row.get("row_routed_mlp_reuse_base_base_cuda_ms_per_call")
        if _as_float(reuse_base_ms) is not None:
            return _compact_join(
                [
                    _kv("reuse_base_sparse", reuse_base_ms, "ms/call"),
                    _kv(
                        "gate_up",
                        row.get("row_routed_mlp_reuse_base_gate_up_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "act",
                        row.get("row_routed_mlp_reuse_base_act_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "down",
                        row.get(
                            "row_routed_mlp_reuse_base_down_sparse_cuda_ms_per_call"
                        ),
                        "ms",
                    ),
                ]
            )
        return _compact_join(
            [
                _kv(
                    "gate_up16-31",
                    row.get(
                        "gate_up_proj_layers_16_31_base_sparse_linear_cuda_ms_per_call"
                    ),
                    "ms/call",
                ),
                _kv(
                    "base_sparse",
                    row.get("base_sparse_linear_cuda_ms_per_call"),
                    "ms/call",
                ),
                _kv(
                    "gate_up16-31_rows",
                    row.get("gate_up_proj_layers_16_31_base_sparse_linear_rows_per_call"),
                ),
                _kv("rows/call", row.get("base_sparse_linear_rows_per_call")),
            ]
        )
    if part == "residual_correction":
        gate_up_dense_ms = row.get("row_routed_gate_up_dense_gemm_cuda_ms_per_call")
        if _as_float(gate_up_dense_ms) is not None:
            return _compact_join(
                [
                    _kv("row_routed_gate_up_dense_gemm", gate_up_dense_ms, "ms/call"),
                    _kv(
                        "dense_gather",
                        row.get("row_routed_gate_up_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_rows/call",
                        row.get("row_routed_gate_up_dense_rows_per_call"),
                    ),
                    _kv_int(
                        "cached_plan_hits",
                        row.get("row_routed_gate_up_cached_plan_hits"),
                    ),
                ]
            )
        routed_dense_ms = _sum_float(
            row,
            (
                "row_routed_mlp_dense_gather_cuda_ms_per_call",
                "row_routed_mlp_dense_gate_up_cuda_ms_per_call",
                "row_routed_mlp_dense_act_cuda_ms_per_call",
                "row_routed_mlp_dense_down_cuda_ms_per_call",
            ),
        )
        if routed_dense_ms is not None:
            return _compact_join(
                [
                    _kv("row_routed_dense_total", routed_dense_ms, "ms/call"),
                    _kv(
                        "dense_gather",
                        row.get("row_routed_mlp_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_gate_up",
                        row.get("row_routed_mlp_dense_gate_up_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_act",
                        row.get("row_routed_mlp_dense_act_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_down",
                        row.get("row_routed_mlp_dense_down_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_rows/call",
                        row.get("row_routed_mlp_dense_rows_per_call"),
                    ),
                ]
            )
        reuse_dense_ms = row.get("row_routed_mlp_reuse_base_dense_cuda_ms_per_call")
        if _as_float(reuse_dense_ms) is not None:
            return _compact_join(
                [
                    _kv("reuse_dense_total", reuse_dense_ms, "ms/call"),
                    _kv(
                        "dense_gather",
                        row.get("row_routed_mlp_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_gate_up",
                        row.get("row_routed_mlp_dense_gate_up_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_act",
                        row.get("row_routed_mlp_dense_act_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_down",
                        row.get("row_routed_mlp_dense_down_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "copy",
                        row.get(
                            "row_routed_mlp_reuse_base_index_copy_cuda_ms_per_call"
                        ),
                        "ms",
                    ),
                    _kv(
                        "dense_rows/call",
                        row.get("row_routed_mlp_reuse_base_dense_rows_per_call"),
                    ),
                ]
            )
        compressed_total_ms = _sum_float(
            row,
            (
                "compressed_residual_materialize_cuda_ms_per_call",
                "compressed_residual_gemm_cuda_ms_per_call",
                "compressed_residual_add_cuda_ms_per_call",
            ),
        )
        if compressed_total_ms is not None:
            return _compact_join(
                [
                    _kv("compressed_total", compressed_total_ms, "ms/call"),
                    _kv(
                        "materialize",
                        row.get("compressed_residual_materialize_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "gemm",
                        row.get("compressed_residual_gemm_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "add",
                        row.get("compressed_residual_add_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv_int(
                        "cache_hits",
                        row.get("compressed_residual_cached_weight_hits"),
                    ),
                    _kv_int(
                        "cache_misses",
                        row.get("compressed_residual_cached_weight_misses"),
                    ),
                ]
            )
        bucket_triton_ms = row.get(
            "bucket_triton_dense_gemm_scatter_cuda_ms_per_call"
        )
        if _as_float(bucket_triton_ms) is not None:
            return _compact_join(
                [
                    _kv(
                        "bucket_triton_dense_gemm_scatter",
                        bucket_triton_ms,
                        "ms/call",
                    ),
                    _kv(
                        "gate_up16-31_bucket_triton",
                        row.get(
                            "gate_up_proj_layers_16_31_"
                            "bucket_triton_dense_gemm_scatter_cuda_ms_per_call"
                        ),
                        "ms/call",
                    ),
                    _kv(
                        "bucket_triton_rows/call",
                        row.get(
                            "bucket_triton_dense_gemm_scatter_rows_per_call"
                        ),
                    ),
                    _kv(
                        "dense_rows/call",
                        row.get("dense_rows_bucket_rows_per_call"),
                    ),
                ]
            )
        residual_ms = _first_float(
            row,
            (
                "residual_dense_gemm_cuda_ms_per_call",
                "residual_sparse_gemm_cuda_ms_per_call",
                "adaptive_dense_fallback_cuda_ms_per_call",
            ),
        )
        residual_name = _first_value(
            row,
            (
                "residual_dense_gemm_cuda_ms_per_call",
                "residual_sparse_gemm_cuda_ms_per_call",
                "adaptive_dense_fallback_cuda_ms_per_call",
            ),
        )
        if residual_ms is None and residual_name == "":
            return ""
        return _compact_join(
            [
                _kv(
                    "gate_up16-31_dense",
                    row.get(
                        "gate_up_proj_layers_16_31_residual_dense_gemm_cuda_ms_per_call"
                    ),
                    "ms/call",
                ),
                _kv("residual", residual_ms, "ms/call_or_event"),
                _kv(
                    "gate_up16-31_rows",
                    row.get("gate_up_proj_layers_16_31_residual_dense_rows_per_call"),
                ),
                _kv("dense_rows/call", row.get("residual_dense_rows_per_call")),
                _kv("bucket_rows/call", row.get("dense_rows_bucket_rows_per_call")),
                _kv_int("adaptive_calls", row.get("adaptive_dense_fallback_calls")),
            ]
        )
    if part == "gather_scatter":
        gate_up_gather_ms = row.get(
            "row_routed_gate_up_gather_scatter_cuda_ms_per_call"
        )
        if _as_float(gate_up_gather_ms) is not None:
            return _compact_join(
                [
                    _kv("row_routed_gate_up_gather_scatter", gate_up_gather_ms, "ms/call"),
                    _kv(
                        "route",
                        row.get("row_routed_gate_up_route_build_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_gather",
                        row.get("row_routed_gate_up_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "base_gather",
                        row.get("row_routed_gate_up_base_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "index_copy",
                        row.get("row_routed_gate_up_index_copy_cuda_ms_per_call"),
                        "ms",
                    ),
                ]
            )
        route_all_gather_ms = row.get(
            "route_all_gather_scatter_cuda_ms_per_linear_call"
        )
        if _as_float(route_all_gather_ms) is not None:
            return _compact_join(
                [
                    _kv("route_all_gather_scatter", route_all_gather_ms, "ms/linear"),
                    _kv(
                        "base_gather",
                        row.get("route_all_base_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_gather",
                        row.get("route_all_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "base_copy",
                        row.get("route_all_base_index_copy_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "dense_copy",
                        row.get("route_all_dense_index_copy_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "event_avg",
                        row.get("gather_scatter_cuda_ms_per_call"),
                        "ms/event",
                    ),
                ]
            )
        routed_gather_ms = _sum_float(
            row,
            (
                "row_routed_mlp_dense_gather_cuda_ms_per_call",
                "row_routed_mlp_base_gather_cuda_ms_per_call",
                "row_routed_mlp_assemble_cuda_ms_per_call",
            ),
        )
        if routed_gather_ms is not None:
            return _compact_join(
                [
                    _kv("row_routed_gather_scatter", routed_gather_ms, "ms/call"),
                    _kv(
                        "dense_gather",
                        row.get("row_routed_mlp_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "base_gather",
                        row.get("row_routed_mlp_base_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "assemble",
                        row.get("row_routed_mlp_assemble_cuda_ms_per_call"),
                        "ms",
                    ),
                ]
            )
        reuse_gather_ms = row.get(
            "row_routed_mlp_reuse_base_gather_scatter_cuda_ms_per_call"
        )
        if _as_float(reuse_gather_ms) is not None:
            return _compact_join(
                [
                    _kv("reuse_gather_scatter", reuse_gather_ms, "ms/call"),
                    _kv(
                        "dense_gather",
                        row.get("row_routed_mlp_dense_gather_cuda_ms_per_call"),
                        "ms",
                    ),
                    _kv(
                        "copy",
                        row.get(
                            "row_routed_mlp_reuse_base_index_copy_cuda_ms_per_call"
                        ),
                        "ms",
                    ),
                ]
            )
        return _kv(
            "gather_scatter", row.get("gather_scatter_cuda_ms_per_call"), "ms/call"
        )
    if part == "routing":
        draft_residual = _fmt_int(row.get("draft_residual_rows"))
        draft_base = _fmt_int(row.get("draft_base_rows"))
        non_draft_residual = _fmt_int(row.get("non_draft_residual_rows"))
        non_draft_base = _fmt_int(row.get("non_draft_base_rows"))
        pieces: list[str] = []
        if draft_residual or draft_base:
            pieces.append(f"draft residual/base={draft_residual}/{draft_base}")
        if non_draft_residual or non_draft_base:
            pieces.append(
                f"non-draft residual/base={non_draft_residual}/{non_draft_base}"
            )
        pieces.extend(
            [
                _kv("draft_residual_frac", row.get("draft_residual_fraction")),
                _kv("non_draft_residual_frac", row.get("non_draft_residual_fraction")),
                _kv("correction_frac", row.get("residual_dense_rows_fraction")),
                _kv("bucket_fill", row.get("bucket_fill_ratio")),
                _kv(
                    "avg_bucket_active",
                    row.get("avg_bucket_active_rows")
                    or row.get("gpu_bucket_active_rows_per_step"),
                ),
                _kv(
                    "avg_bucket_candidate",
                    row.get("avg_bucket_candidate_rows")
                    or row.get("sr24_bucket_candidate_rows_per_call"),
                ),
                _compact_join(
                    [
                        "bucket_actual/requested="
                        f"{_fmt_int(row.get('sr24_bucket_active_rows'))}/"
                        f"{_fmt_int(row.get('sr24_bucket_residual_requested_rows'))}"
                    ]
                )
                if (
                    (
                        _as_float(row.get("sr24_bucket_active_rows"))
                        not in (None, 0.0)
                    )
                    or _fmt_int(row.get("sr24_bucket_residual_requested_rows"))
                )
                else "",
                _kv(
                    "bucket_actual_frac",
                    row.get("sr24_bucket_active_fraction_of_requested"),
                ),
            ]
        )
        return _compact_join(pieces)
    if part == "cuda_graph":
        graph = _fmt_graph_counts(_row_graph_counts_value(row))
        return _compact_join(
            [
                f"modes={graph}" if graph else "",
                _kv("FULL", row.get("cudagraph_full_fraction")),
                _kv("NONE", row.get("cudagraph_none_fraction")),
            ]
        )
    if part == "gpu_util":
        speedup = _first_value(
            row,
            (
                "full_batch_speedup_vs_same_root_dense",
                "total_tps_speedup_vs_same_root_dense",
                "total_tps_speedup_vs_best_clean_dense",
            ),
        )
        return _compact_join(
            [
                _kv("avg", row.get("avg_gpu_util_pct"), "%"),
                _kv("peak", row.get("peak_gpu_util_pct"), "%"),
                _kv("full_tps", row.get("full_batch_output_tps")),
                _kv("total_tps", row.get("total_output_tps")),
                _kv("speedup", speedup, "x"),
            ]
        )
    raise ValueError(f"unknown part {part}")


def _part_read(row: dict[str, Any], part: str) -> str:
    row_kind = str(row.get("row_kind", ""))
    if part == "scheduler_mask":
        mask_ms = _as_float(row.get("scheduler_mask_build_cpu_ms_per_step"))
        wall_ms = _as_float(row.get("scheduler_mask_wall_cpu_ms_per_step"))
        row_bucket_ms = _as_float(
            row.get("scheduler_row_index_bucket_wall_cpu_ms_per_step")
        )
        sync_ms = _as_float(row.get("scheduler_mask_state_sync_cpu_ms_per_call"))
        loop_ms = _as_float(row.get("scheduler_request_routing_loop_cpu_ms_per_call"))
        if row_kind.startswith("diagnostic") and (
            (wall_ms is not None and wall_ms > 1.0)
            or (mask_ms is not None and mask_ms > 1.0)
        ):
            return "sync-heavy diagnostic path; useful for ablation, not clean serving"
        if row_bucket_ms is not None and row_bucket_ms > 1.0:
            return "dynamic row-index/bucket construction is the visible bottleneck"
        if (sync_ms and sync_ms > 1.0) or (loop_ms and loop_ms > 1.0):
            return "sync-heavy diagnostic path; useful for ablation, not clean serving"
        if wall_ms is not None and wall_ms <= 1.0:
            return "clean wall-clock mask build is sub-ms"
        if wall_ms is not None:
            return "clean mask/bucket build is visible in serving"
        if mask_ms is not None and mask_ms <= 1.0:
            return "clean mask build is sub-ms"
        if mask_ms is not None:
            return "mask build is a candidate bottleneck"
        return "not measured in this row"
    if part == "base_sparse":
        base_ms = (
            _as_float(row.get("base_sparse_linear_cuda_ms_per_call"))
            or _as_float(row.get("row_routed_gate_up_base_sparse_cuda_ms_per_call"))
            or _as_float(row.get("row_routed_mlp_reuse_base_base_cuda_ms_per_call"))
            or _sum_float(
                row,
                (
                    "row_routed_mlp_base_gather_cuda_ms_per_call",
                    "row_routed_mlp_base_gate_up_cuda_ms_per_call",
                    "row_routed_mlp_base_down_sparse_cuda_ms_per_call",
                    "row_routed_mlp_down_sparse_cuda_ms_per_call",
                ),
            )
        )
        if base_ms is None:
            return "not measured in this row"
        if base_ms >= 0.5:
            return "sparse base is a large GPU-side cost"
        return "sparse base is cheap enough only if correction stays low"
    if part == "residual_correction":
        residual_ms = _sum_float(
            row,
            (
                "compressed_residual_materialize_cuda_ms_per_call",
                "compressed_residual_gemm_cuda_ms_per_call",
                "compressed_residual_add_cuda_ms_per_call",
            ),
        )
        if residual_ms is None:
            residual_ms = _first_float(
                row,
                (
                    "row_routed_gate_up_dense_gemm_cuda_ms_per_call",
                    "row_routed_mlp_reuse_base_dense_cuda_ms_per_call",
                    "row_routed_mlp_reuse_base_dense_cuda_ms_per_call",
                    "row_routed_mlp_dense_gate_up_cuda_ms_per_call",
                    "residual_dense_gemm_cuda_ms_per_call",
                    "residual_sparse_gemm_cuda_ms_per_call",
                    "adaptive_dense_fallback_cuda_ms_per_call",
                ),
            )
        if residual_ms is None:
            return "not measured in this row"
        base_ms = (
            _as_float(row.get("base_sparse_linear_cuda_ms_per_call"))
            or _as_float(row.get("row_routed_gate_up_base_sparse_cuda_ms_per_call"))
            or _as_float(row.get("row_routed_mlp_reuse_base_base_cuda_ms_per_call"))
            or _sum_float(
                row,
                (
                    "row_routed_mlp_base_gather_cuda_ms_per_call",
                    "row_routed_mlp_base_gate_up_cuda_ms_per_call",
                    "row_routed_mlp_base_down_sparse_cuda_ms_per_call",
                    "row_routed_mlp_down_sparse_cuda_ms_per_call",
                ),
            )
        )
        if base_ms is not None and residual_ms >= 0.25 * base_ms:
            return "correction is large relative to sparse base"
        return "correction is secondary for this row"
    if part == "gather_scatter":
        gather_ms = (
            _as_float(row.get("gather_scatter_cuda_ms_per_call"))
            or _as_float(
                row.get("row_routed_gate_up_gather_scatter_cuda_ms_per_call")
            )
            or _as_float(
                row.get("row_routed_mlp_reuse_base_gather_scatter_cuda_ms_per_call")
            )
            or _sum_float(
                row,
                (
                    "row_routed_mlp_dense_gather_cuda_ms_per_call",
                    "row_routed_mlp_base_gather_cuda_ms_per_call",
                    "row_routed_mlp_assemble_cuda_ms_per_call",
                ),
            )
        )
        residual_ms = _first_float(
            row,
            (
                "row_routed_gate_up_dense_gemm_cuda_ms_per_call",
                "row_routed_mlp_reuse_base_dense_cuda_ms_per_call",
                "row_routed_mlp_dense_gate_up_cuda_ms_per_call",
                "residual_dense_gemm_cuda_ms_per_call",
                "compressed_residual_gemm_cuda_ms_per_call",
            ),
        )
        if gather_ms is None:
            return "not measured in this row"
        if residual_ms is not None and gather_ms > residual_ms:
            return "assembly dominates correction; prioritize fused route"
        if gather_ms > 0.05:
            return "assembly is visible enough to erase small sparse gains"
        return "assembly is not the first bottleneck"
    if part == "routing":
        correction_frac = _as_float(row.get("residual_dense_rows_fraction"))
        draft_frac = _as_float(row.get("draft_residual_fraction"))
        non_draft_frac = _as_float(row.get("non_draft_residual_fraction"))
        bucket_fill = _as_float(row.get("bucket_fill_ratio"))
        has_route_counts = any(
            _as_float(row.get(key)) is not None
            for key in (
                "draft_residual_rows",
                "draft_base_rows",
                "non_draft_residual_rows",
                "non_draft_base_rows",
                "sr24_bucket_candidate_rows",
                "sr24_bucket_active_rows",
                "sr24_bucket_residual_requested_rows",
            )
        )
        if correction_frac is not None and correction_frac >= 0.125:
            return "correction fraction is outside the measured gate/up positive region"
        if draft_frac is not None and draft_frac >= 0.5:
            return "many draft rows still require residual correction"
        if correction_frac is not None:
            return "routing is sparse-friendly if quality holds"
        if (
            (draft_frac is not None and draft_frac > 0.0)
            or (non_draft_frac is not None and non_draft_frac > 0.0)
        ):
            return "routing correction is visible; compare it against sparse-base cost"
        if bucket_fill is not None:
            if bucket_fill >= 0.8:
                return "bucket capacity is mostly used; operator cost matters more than fill"
            return "bucket fill is low; selection capacity may be wasted"
        if has_route_counts:
            return "routing counts available; no residual fraction derived for this row"
        return "not measured in this row"
    if part == "cuda_graph":
        none_frac = _as_float(row.get("cudagraph_none_fraction"))
        if none_frac is None:
            return "not measured in this row"
        if none_frac >= 0.2:
            return "CUDA Graph misses may contribute"
        return "CUDA Graph coverage is not the first suspect"
    if part == "gpu_util":
        util = _as_float(row.get("avg_gpu_util_pct"))
        if util is None:
            return "not measured in this row"
        if row_kind.startswith("diagnostic") and util < 70.0:
            return "diagnostic synchronization lowers util; use clean row for real serving util"
        if util < 70.0:
            return "GPU underutilization is likely"
        return "GPU is busy; focus on useful-work efficiency"
    raise ValueError(f"unknown part {part}")


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts = (
        "scheduler_mask",
        "base_sparse",
        "residual_correction",
        "gather_scatter",
        "routing",
        "cuda_graph",
        "gpu_util",
    )
    compact: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        for part in parts:
            compact.append(
                {
                    "source_index": idx,
                    "source_root": row.get("source_root", ""),
                    "method": row.get("method", ""),
                    "dataset": row.get("dataset", ""),
                    "batch_size": row.get("batch_size", ""),
                    "max_new_tokens": row.get("max_new_tokens", ""),
                    "row_kind": row.get("row_kind", ""),
                    "part": part,
                    "evidence": _part_value(row, part),
                    "read": _part_read(row, part),
                }
            )
    return compact


def _has_component_timing(row: dict[str, Any]) -> bool:
    keys = (
        "base_sparse_linear_cuda_ms_per_call",
        "row_routed_gate_up_cuda_ms_per_call",
        "row_routed_gate_up_base_sparse_cuda_ms_per_call",
        "row_routed_gate_up_dense_gemm_cuda_ms_per_call",
        "row_routed_mlp_cuda_ms_per_call",
        "row_routed_mlp_base_gate_up_cuda_ms_per_call",
        "compressed_residual_gemm_cuda_ms_per_call",
        "residual_dense_gemm_cuda_ms_per_call",
        "residual_sparse_gemm_cuda_ms_per_call",
    )
    return any(_as_float(row.get(key)) is not None for key in keys)


def _is_clean_like_serving(row: dict[str, Any]) -> bool:
    """Rows with exact SR24 aggregate stats can still be serving rows.

    The matrix runner marks all-corrected runs with exact aggregate SR24 stats
    as diagnostic_exact_stats even when no per-linear CUDA events were enabled.
    Treat those as clean-like for throughput/GPU-util selection, while keeping
    rows with component timing as diagnostic only.
    """
    row_kind = str(row.get("row_kind", ""))
    if row_kind == "clean_serving":
        return True
    return row_kind == "diagnostic_exact_stats" and not _has_component_timing(row)


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


def _choose_representative(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("method") == method]
    if not candidates:
        return None
    clean = [row for row in candidates if _is_clean_like_serving(row)]
    if clean:
        graph_clean = [
            row for row in clean
            if _parse_graph_counts(_row_graph_counts_value(row))
        ]
        if graph_clean:
            clean = graph_clean
        return max(
            clean,
            key=lambda row: _first_float(
                row, ("full_batch_output_tps", "total_output_tps")
            )
            or float("-inf"),
        )
    linear = [
        row
        for row in candidates
        if _as_float(row.get("base_sparse_linear_cuda_ms_per_call")) is not None
    ]
    if linear:
        return linear[-1]
    return candidates[-1]


def _row_speedup(row: dict[str, Any]) -> float | None:
    return _first_float(
        row,
        (
            "full_batch_speedup_vs_same_root_dense",
            "total_tps_speedup_vs_same_root_dense",
            "total_tps_speedup_vs_best_clean_dense",
        ),
    )


def _choose_clean_by_speed(
    rows: list[dict[str, Any]],
    method: str,
    *,
    fastest: bool,
    prefer_graph_counts: bool = True,
    prefer_primary_clean_root: bool = True,
    prefer_relative_speedup: bool = False,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("method") == method
        and _is_clean_like_serving(row)
        and _first_float(row, ("full_batch_output_tps", "total_output_tps")) is not None
    ]
    if not candidates:
        return None
    if prefer_primary_clean_root:
        primary_candidates = [
            row
            for row in candidates
            if "/clean_serving" in str(row.get("source_root", ""))
        ]
        if primary_candidates:
            candidates = primary_candidates
    if prefer_graph_counts:
        graph_candidates = [
            row for row in candidates
            if _parse_graph_counts(_row_graph_counts_value(row))
        ]
        if graph_candidates:
            candidates = graph_candidates
    def sort_key(row: dict[str, Any]) -> float:
        if prefer_relative_speedup:
            relative = _row_speedup(row)
            if relative is not None:
                return relative
        return (
            _first_float(row, ("full_batch_output_tps", "total_output_tps"))
            or float("-inf")
        )

    return sorted(candidates, key=sort_key, reverse=fastest)[0]


def _choose_linear_timing(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("method") == method
        and (
            _as_float(row.get("base_sparse_linear_cuda_ms_per_call")) is not None
            or _as_float(row.get("row_routed_gate_up_base_sparse_cuda_ms_per_call"))
            is not None
            or _as_float(row.get("row_routed_mlp_cuda_ms_per_call")) is not None
            or _as_float(row.get("row_routed_mlp_base_gate_up_cuda_ms_per_call"))
            is not None
        )
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: _first_float(
            row,
            (
                "row_routed_gate_up_base_sparse_cuda_ms_per_call",
                "base_sparse_linear_cuda_ms_per_call",
                "row_routed_mlp_cuda_ms_per_call",
                "row_routed_mlp_base_gate_up_cuda_ms_per_call",
            ),
        )
        or float("-inf"),
    )


def _choose_clean_scheduler_timing(
    rows: list[dict[str, Any]],
    method: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("method") == method
        and _is_clean_like_serving(row)
        and (
            _as_float(row.get("scheduler_mask_wall_cpu_ms_per_step")) is not None
            or _as_float(row.get("scheduler_mask_build_cpu_ms_per_step")) is not None
        )
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: _first_float(
            row, ("full_batch_output_tps", "total_output_tps")
        )
        or float("-inf"),
    )


def _choose_routing_timing(rows: list[dict[str, Any]],
                           method: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("method") == method
        and (
            _as_float(row.get("base_sparse_linear_cuda_ms_per_call")) is not None
            or _as_float(row.get("row_routed_gate_up_base_sparse_cuda_ms_per_call"))
            is not None
            or _as_float(row.get("row_routed_mlp_cuda_ms_per_call")) is not None
            or _as_float(row.get("row_routed_mlp_base_gate_up_cuda_ms_per_call"))
            is not None
        )
        and _as_float(row.get("draft_residual_fraction")) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: _as_float(row.get("draft_residual_fraction"))
        or float("-inf"),
    )


def _key_read(row: dict[str, Any] | None, label: str) -> str:
    if row is None:
        return ""
    util = _as_float(row.get("avg_gpu_util_pct"))
    none = _as_float(row.get("cudagraph_none_fraction"))
    base_ms = (
        _as_float(row.get("base_sparse_linear_cuda_ms_per_call"))
        or _as_float(row.get("row_routed_gate_up_base_sparse_cuda_ms_per_call"))
        or _as_float(row.get("row_routed_mlp_reuse_base_base_cuda_ms_per_call"))
        or _sum_float(
            row,
            (
                "row_routed_mlp_base_gather_cuda_ms_per_call",
                "row_routed_mlp_base_gate_up_cuda_ms_per_call",
                "row_routed_mlp_base_down_sparse_cuda_ms_per_call",
                "row_routed_mlp_down_sparse_cuda_ms_per_call",
            ),
        )
    )
    residual_ms = _sum_float(
        row,
        (
            "compressed_residual_materialize_cuda_ms_per_call",
            "compressed_residual_gemm_cuda_ms_per_call",
            "compressed_residual_add_cuda_ms_per_call",
        ),
    )
    if residual_ms is None:
        residual_ms = _first_float(
            row,
            (
                "row_routed_gate_up_dense_gemm_cuda_ms_per_call",
                "row_routed_mlp_cuda_ms_per_call",
                "row_routed_mlp_reuse_base_dense_cuda_ms_per_call",
                "row_routed_mlp_dense_gate_up_cuda_ms_per_call",
                "residual_dense_gemm_cuda_ms_per_call",
                "residual_sparse_gemm_cuda_ms_per_call",
                "adaptive_dense_fallback_cuda_ms_per_call",
            ),
        )
    draft_frac = _as_float(row.get("draft_residual_fraction"))
    label_l = label.lower()
    if "slow" in label_l and "base_only" in label_l:
        return "slow base_only is graph/util/scope limited, not acceptance limited"
    if "fast" in label_l and "base_only" in label_l:
        return "base-only upper bound is real when scope and graph coverage are good"
    if "clean speclink" in label_l:
        if none is not None and none >= 0.2:
            return "clean mixed SR24 loses CUDA Graph coverage; combine graph fix with operator work"
        if util is not None and util < 80.0:
            return "clean mixed SR24 underutilizes the GPU; reduce small kernels and CPU waits"
        return "clean mixed SR24 is graph/util healthy; optimize useful-work efficiency"
    if str(row.get("row_kind", "")).startswith("diagnostic") and util is not None and util < 70.0:
        return "diagnostic timing/sync row; use it for localization, not clean serving throughput"
    if base_ms is not None and residual_ms is not None:
        if draft_frac is not None and draft_frac > 0.5:
            return "quality-safe routing sends too many rows through residual correction"
        return "operator timing row; compare sparse base plus correction against dense"
    if util is not None and util < 70.0:
        return "GPU underutilization is likely"
    if none is not None and none >= 0.2:
        return "CUDA Graph misses may contribute"
    return "reference row"


def _append_key_row(lines: list[str], label: str,
                    row: dict[str, Any] | None) -> None:
    if row is None:
        lines.append(f"| {label} | n/a | n/a |  |  |  |  |  |  |  |  |")
        return
    speedup = _row_speedup(row)
    lines.append(
        "| {label} | `{method}` | {kind} | {bs} | {full_tps} | {total_tps} | {speedup} | {util} | {none_frac} | {base_ms} | {residual_ms} | {read} |".format(
            label=label,
            method=row.get("method", ""),
            kind=row.get("row_kind", ""),
            bs=row.get("batch_size", ""),
            full_tps=_fmt(row.get("full_batch_output_tps")),
            total_tps=_fmt(row.get("total_output_tps")),
            speedup=_fmt(speedup),
            util=_fmt(row.get("avg_gpu_util_pct")),
            none_frac=_fmt(row.get("cudagraph_none_fraction")),
            base_ms=_fmt(
                _first_float(
                    row,
                    (
                        "row_routed_gate_up_base_sparse_cuda_ms_per_call",
                        "base_sparse_linear_cuda_ms_per_call",
                        "row_routed_mlp_reuse_base_base_cuda_ms_per_call",
                        "row_routed_mlp_base_gate_up_cuda_ms_per_call",
                    ),
                )
            ),
            residual_ms=_fmt(
                _sum_float(
                    row,
                    (
                        "compressed_residual_materialize_cuda_ms_per_call",
                        "compressed_residual_gemm_cuda_ms_per_call",
                        "compressed_residual_add_cuda_ms_per_call",
                    ),
                )
                if _sum_float(
                    row,
                    (
                        "compressed_residual_materialize_cuda_ms_per_call",
                        "compressed_residual_gemm_cuda_ms_per_call",
                        "compressed_residual_add_cuda_ms_per_call",
                    ),
                )
                is not None
                else _first_float(
                    row,
                    (
                        "row_routed_gate_up_dense_gemm_cuda_ms_per_call",
                        "row_routed_mlp_cuda_ms_per_call",
                        "row_routed_mlp_reuse_base_dense_cuda_ms_per_call",
                        "row_routed_mlp_dense_gate_up_cuda_ms_per_call",
                        "residual_dense_gemm_cuda_ms_per_call",
                        "residual_sparse_gemm_cuda_ms_per_call",
                        "adaptive_dense_fallback_cuda_ms_per_call",
                    ),
                )
            ),
            read=_key_read(row, label),
        )
    )


def _diagnosis_cell(row: dict[str, Any] | None, part: str) -> str:
    if row is None:
        return ""
    return _part_value(row, part)


def _diagnosis_read(
    *,
    clean: dict[str, Any] | None,
    diagnostic: dict[str, Any] | None,
    part: str,
) -> str:
    diag_read = _part_read(diagnostic, part) if diagnostic is not None else ""
    clean_read = _part_read(clean, part) if clean is not None else ""
    if part in {"scheduler_mask", "cuda_graph", "gpu_util"}:
        if clean_read and clean_read != "not measured in this row":
            return clean_read
    if diag_read and diag_read != "not measured in this row":
        return diag_read
    return clean_read


def _ratio_text(numer: Any, denom: Any) -> str:
    value = _ratio(numer, denom)
    if value is None:
        return ""
    return f"{value:.2f}x"


def _microbench_read(row: dict[str, Any]) -> str:
    dense = _as_float(row.get("dense_graph_ms")) or _as_float(row.get("dense_ms"))
    base = _as_float(row.get("base_sparse_graph_ms")) or _as_float(
        row.get("base_sparse_ms")
    )
    mixed = _as_float(row.get("bucket_delta_inplace_graph_ms")) or _as_float(
        row.get("bucket_delta_inplace_ms")
    )
    copy = _as_float(row.get("bucket_dense_copy_inplace_graph_ms")) or _as_float(
        row.get("bucket_dense_copy_inplace_ms")
    )
    gather = _as_float(row.get("gather_scatter_cuda_ms"))
    residual = _as_float(row.get("residual_dense_gemm_cuda_ms"))
    if dense is None:
        return "missing dense reference"
    if base is not None and mixed is not None and base < dense and mixed >= dense:
        return "sparse base has headroom, but correction/assembly erases it"
    if mixed is not None and mixed < dense:
        return "current mixed proxy can beat dense for this shape/fraction"
    if copy is not None and mixed is not None and copy < mixed:
        return "dense-copy bucket reduces mixed overhead; validate in serving"
    if gather is not None and residual is not None and gather > residual:
        return "gather/scatter is larger than dense-row GEMM"
    return "operator bound; compare against serving row before optimizing"


def _write_operator_microbench_section(
    lines: list[str],
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    lines.extend(
        [
            "",
            "## Operator Microbench",
            "",
            "These rows are isolated Linear-shape GPU timings. They do not measure",
            "GuideLLM/vLLM serving throughput, but they show whether the current",
            "sparse-base plus residual-correction operator can beat dense after",
            "scheduler overhead and CUDA Graph misses are removed.",
            "",
            "| source | shape rows/out/in | residual frac | residual rows | bucket fill | dense graph ms | base sparse graph ms | base/dense | current mixed graph ms | mixed/dense | dense-row GEMM ms | gather+scatter ms | bucket dense-copy graph ms | copy/dense | compressed dense-delta graph ms | comp/dense | routed split graph ms | routed/dense | prefix concat graph ms | prefix/dense | read |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        dense_ref = row.get("dense_graph_ms") or row.get("dense_ms")
        lines.append(
            "| {idx} | {shape} | {frac} | {res_rows} | {fill} | {dense} | {base} | {base_ratio} | {mixed} | {mixed_ratio} | {residual} | {gather} | {copy} | {copy_ratio} | {comp} | {comp_ratio} | {routed} | {routed_ratio} | {prefix} | {prefix_ratio} | {read} |".format(
                idx=idx,
                shape=(
                    f"{row.get('rows', '')}/{row.get('out_features', '')}/"
                    f"{row.get('in_features', '')}"
                ),
                frac=_fmt(row.get("residual_fraction")),
                res_rows=_fmt(row.get("residual_rows"), 0),
                fill=_fmt(row.get("bucket_fill_ratio")),
                dense=_fmt(row.get("dense_graph_ms")),
                base=_fmt(row.get("base_sparse_graph_ms")),
                base_ratio=_ratio_text(row.get("base_sparse_graph_ms"), dense_ref),
                mixed=_fmt(row.get("bucket_delta_inplace_graph_ms")),
                mixed_ratio=_ratio_text(
                    row.get("bucket_delta_inplace_graph_ms"), dense_ref
                ),
                residual=_fmt(row.get("residual_dense_gemm_cuda_ms")),
                gather=_fmt(row.get("gather_scatter_cuda_ms")),
                copy=_fmt(row.get("bucket_dense_copy_inplace_graph_ms")),
                copy_ratio=_ratio_text(
                    row.get("bucket_dense_copy_inplace_graph_ms"), dense_ref
                ),
                comp=_fmt(row.get("compressed_delta_dense_inplace_graph_ms")),
                comp_ratio=_ratio_text(
                    row.get("compressed_delta_dense_inplace_graph_ms"), dense_ref
                ),
                routed=_fmt(row.get("routed_split_graph_ms")),
                routed_ratio=_ratio_text(row.get("routed_split_graph_ms"), dense_ref),
                prefix=_fmt(row.get("routed_prefix_concat_graph_ms")),
                prefix_ratio=_ratio_text(
                    row.get("routed_prefix_concat_graph_ms"), dense_ref
                ),
                read=_microbench_read(row),
            )
        )


def _write_slowdown_diagnosis(
    lines: list[str],
    *,
    clean_speclink: dict[str, Any] | None,
    speclink_linear_diag: dict[str, Any] | None,
) -> None:
    parts = [
        (
            "scheduler / mask build",
            "Construct residual masks, bucket rows, and per-step scheduling state.",
            "scheduler_mask",
        ),
        (
            "base sparse linear",
            "Sparse base GEMM, with gate_up_proj layers 16-31 shown when available.",
            "base_sparse",
        ),
        (
            "residual correction",
            "Dense-row or compressed residual correction cost.",
            "residual_correction",
        ),
        (
            "gather/scatter",
            "index_select/index_add_/bucket assembly overhead around correction.",
            "gather_scatter",
        ),
        (
            "routing statistics",
            "Residual/base split for draft and non-draft rows plus bucket fill.",
            "routing",
        ),
        (
            "CUDA Graph",
            "FULL/NONE graph coverage for the same serving path.",
            "cuda_graph",
        ),
        (
            "GPU util",
            "Whether the slowdown is underutilization or inefficient full-GPU work.",
            "gpu_util",
        ),
    ]
    lines.extend(
        [
            "",
            "## Slowdown Diagnosis",
            "",
            "This table maps the current evidence to the exact components that need",
            "to be checked before another controller sweep. Clean serving rows are",
            "used for throughput, graph coverage, and GPU util. Diagnostic rows are",
            "used only for localized CUDA-event and routing evidence because their",
            "synchronization overhead can distort tok/s.",
            "",
            "| part | what is measured | clean serving evidence | diagnostic evidence | current read |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for label, measured, part in parts:
        lines.append(
            "| {label} | {measured} | {clean} | {diag} | {read} |".format(
                label=label,
                measured=measured,
                clean=_diagnosis_cell(clean_speclink, part),
                diag=_diagnosis_cell(speclink_linear_diag, part),
                read=_diagnosis_read(
                    clean=clean_speclink,
                    diagnostic=speclink_linear_diag,
                    part=part,
                ),
            )
        )


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    compact: list[dict[str, Any]],
    operator_microbench_rows: list[dict[str, Any]],
) -> None:
    dense = _choose_representative(rows, "dense_baseline") or _choose_representative(
        rows, "vllm_eagle3"
    )
    base_only = _choose_representative(rows, "base_only_24")
    all_corrected = _choose_representative(rows, "all_corrected_24")
    speclink = _choose_representative(rows, "speclink_t08")
    fast_base_only = _choose_clean_by_speed(rows, "base_only_24", fastest=True)
    slow_base_only = _choose_clean_by_speed(rows, "base_only_24", fastest=False)
    clean_speclink = _choose_clean_by_speed(
        rows,
        "speclink_t08",
        fastest=True,
        prefer_primary_clean_root=False,
    )
    slow_speclink = _choose_clean_by_speed(
        rows,
        "speclink_t08",
        fastest=False,
        prefer_primary_clean_root=False,
        prefer_relative_speedup=True,
    )
    speclink_routing_diag = _choose_routing_timing(rows, "speclink_t08")
    speclink_linear_diag = _choose_linear_timing(rows, "speclink_t08")
    speclink_clean_scheduler = _choose_clean_scheduler_timing(
        rows, "speclink_t08"
    )
    clean_all_corrected = _choose_clean_by_speed(
        rows, "all_corrected_24", fastest=True
    )
    all_corrected_diag = (
        _choose_linear_timing(rows, "all_corrected_24")
        or _choose_representative(rows, "all_corrected_24")
    )

    lines = [
        "# SR24 Seven-Part Slowdown Breakdown",
        "",
        "This report is an offline summary over existing artifacts. It is meant to",
        "answer the fixed slowdown questions before another controller sweep.",
        "",
        "## How To Read This Report",
        "",
        "- `clean_serving` rows are the end-to-end throughput and GPU-utilization",
        "  reference. These rows should use low-sync counters.",
        "- `diagnostic` rows localize scheduler and Linear costs with CUDA events",
        "  and exact routing counters. Their tok/s and GPU util can be distorted",
        "  by synchronization and should not override the clean rows.",
        "- Operator microbench rows, when present in the source artifacts, are",
        "  isolated kernel-shape evidence. They answer whether a sparse/residual",
        "  operator can beat dense even after graph and Python overhead are removed.",
        "",
        "## Representative Rows",
        "",
        "| label | method | row kind | bs | full-batch tok/s | total tok/s | same-root speedup | GPU util % | CUDA Graph |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in (
        ("dense", dense),
        ("base_only_24", base_only),
        ("all_corrected_24", all_corrected),
        ("speclink_t08", speclink),
    ):
        if row is None:
            lines.append(f"| {label} | n/a | n/a |  |  |  |  |  |  |")
            continue
        speedup = _first_value(
            row,
            (
                "full_batch_speedup_vs_same_root_dense",
                "total_tps_speedup_vs_same_root_dense",
                "total_tps_speedup_vs_best_clean_dense",
            ),
        )
        lines.append(
            "| {label} | `{method}` | {kind} | {bs} | {full_tps} | {total_tps} | {speedup} | {util} | `{graph}` |".format(
                label=label,
                method=row.get("method", ""),
                kind=row.get("row_kind", ""),
                bs=row.get("batch_size", ""),
                full_tps=_fmt(row.get("full_batch_output_tps")),
                total_tps=_fmt(row.get("total_output_tps")),
                speedup=_fmt(speedup),
                util=_fmt(row.get("avg_gpu_util_pct")),
                graph=_fmt_graph_counts(_row_graph_counts_value(row)),
            )
        )

    lines.extend(
        [
            "",
            "## Key Reads",
            "",
            "| label | method | row kind | bs | full-batch tok/s | total tok/s | speedup | GPU util % | graph NONE frac | base sparse ms/call | correction ms/call | read |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    _append_key_row(lines, "fast base_only upper bound", fast_base_only)
    _append_key_row(lines, "slow broad base_only", slow_base_only)
    _append_key_row(lines, "clean speclink_t08", clean_speclink)
    if slow_speclink is not None and slow_speclink is not clean_speclink:
        _append_key_row(
            lines,
            "slow/underutilized speclink_t08 ablation",
            slow_speclink,
        )
    _append_key_row(lines, "speclink_t08 mixed routing diagnostic", speclink_routing_diag)
    _append_key_row(lines, "speclink_t08 linear diagnostic", speclink_linear_diag)
    _append_key_row(lines, "clean all_corrected_24", clean_all_corrected)
    _append_key_row(lines, "all_corrected_24 diagnostic", all_corrected_diag)

    _write_slowdown_diagnosis(
        lines,
        clean_speclink=speclink_clean_scheduler
        or clean_speclink
        or clean_all_corrected,
        speclink_linear_diag=speclink_linear_diag or all_corrected_diag,
    )

    _write_operator_microbench_section(lines, operator_microbench_rows)

    lines.extend(
        [
            "",
            "## Seven-Part Matrix",
            "",
            "| source | method | row kind | part | evidence | read |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compact:
        lines.append(
            "| {idx} | `{method}` | {kind} | {part} | {evidence} | {read} |".format(
                idx=row["source_index"],
                method=row["method"],
                kind=row["row_kind"],
                part=row["part"],
                evidence=row["evidence"],
                read=row["read"],
            )
        )

    lines.extend(
        [
            "",
            "## Current Decision Rules",
            "",
            "- If `base_only_24` has normal accepted length but low GPU util or many CUDA Graph `NONE` steps, treat it as a serving/operator-shape issue rather than an acceptance issue.",
            "- If `all_corrected_24` reports `compressed_residual_*` on CUDA but has high materialize/GEMM time, the backend is GPU-resident but not a speed path without a fused residual operator.",
            "- If `speclink_t08` keeps high GPU util and good CUDA Graph coverage but remains below dense, optimize the mixed sparse+residual operator or reduce residual rows before changing scheduler policy.",
            "- If a `speclink_t08` ablation drops below 70% average GPU util, treat it as a small-kernel/fragmentation path even if it improves routing quality or acceptance.",
            "- If accepted base-only tokens appear in stable request-id traces, fix quality by protecting accepted/write rows; aggregate accuracy cancellation is not enough.",
            "",
            "## Sources",
            "",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx}. `{row.get('source_root', '')}`")
        if row.get("breakdown_path"):
            lines.append(f"   breakdown: `{row.get('breakdown_path')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.roots)
    operator_microbench_rows = _load_operator_microbench_rows(args.roots)
    compact = _compact_rows(rows)
    _write_csv(output_root / "seven_part_breakdown.csv", compact)
    _write_csv(output_root / "joined_rows.csv", rows)
    if operator_microbench_rows:
        _write_csv(output_root / "operator_microbench.csv", operator_microbench_rows)
    _write_report(output_root / "report.md", rows, compact, operator_microbench_rows)
    print(output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
