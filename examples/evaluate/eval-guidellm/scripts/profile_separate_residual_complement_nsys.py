#!/usr/bin/env python3
"""Profile and aggregate the separate residual-complement GPU timeline.

The default driver profiles the representative Qwen3-8B and Qwen3-14B
``o_proj, M=512`` cases.  A worker reproduces the production topology exactly:
all-row cuSPARSELt base and dense-row CUTLASS complement execute on distinct
streams, followed by base gather, add, and scatter on the origin stream.

Each calibration launch is enclosed by a narrow NVTX stage.  The analyzer
joins CUDA runtime launch correlation IDs to the calibration kernels, then
uses kernel/grid/block signatures and stable CUDA Graph node IDs to label
node-level replays.  Graph-created virtual streams deliberately are not
matched to eager calibration stream IDs.  The measured timeline therefore
retains the formal graph scheduling behavior without guessing from demangled
kernel names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sparse24_benchmark_common as common
from speculators.speclink import (
    COMPLEMENT_CTA_VARIANTS,
    SPARSE_RESIDUAL_SMEM,
    TP1_FUSED_WEIGHT_SHAPES,
    cusparselt_sparse_residual_residual_linear,
    cusparselt_sparse_residual_sparse_linear,
    prepare_cusparselt_sparse_residual_weight,
    prepare_online_sparse24_weight,
    select_cusparselt_algorithm,
)


NSYS = Path("/usr/local/cuda-13.2/bin/nsys")
MODELS = ("qwen3_8b", "qwen3_14b")
MODEL_LABELS = {"qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B"}
DEFAULT_DENSE_FRACTION = Fraction(1, 8)
DEFAULT_M = 512
DEFAULT_SEED = 20260721
DEFAULT_WARMUP = 100
DEFAULT_REPETITIONS = 20
STAGES = (
    "activation_gather",
    "base",
    "complement",
    "merge_base_gather",
    "merge_add",
    "merge_scatter",
)
STAGE_LABELS = {
    "activation_gather": "Activation gather",
    "base": "cuSPARSELt base",
    "complement": "Complement HMMA.SP",
    "merge_base_gather": "Merge: base gather",
    "merge_add": "Merge: add",
    "merge_scatter": "Merge: scatter",
}


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def launch_profiled_separate(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    runtime: Any,
    resources: common.MultiStreamResources,
    *,
    complement_variant: str = "auto",
) -> torch.Tensor:
    """The production separate path with stage-only NVTX annotations."""

    origin = torch.cuda.current_stream(x.device)
    resources.fork_event.record(origin)
    resources.dense_stream.wait_event(resources.fork_event)
    resources.sparse_stream.wait_event(resources.fork_event)
    with torch.cuda.stream(resources.sparse_stream):
        with nvtx_range("stage:base"):
            base = cusparselt_sparse_residual_sparse_linear(x, runtime)
    resources.sparse_done_event.record(resources.sparse_stream)
    with torch.cuda.stream(resources.dense_stream):
        with nvtx_range("stage:activation_gather"):
            dense_x = x.index_select(0, dense_indices)
        with nvtx_range("stage:complement"):
            correction = cusparselt_sparse_residual_residual_linear(
                dense_x, runtime, variant=complement_variant
            )
    resources.dense_done_event.record(resources.dense_stream)
    origin.wait_event(resources.sparse_done_event)
    origin.wait_event(resources.dense_done_event)
    with nvtx_range("stage:merge_base_gather"):
        corrected = base.index_select(0, dense_indices)
    with nvtx_range("stage:merge_add"):
        corrected.add_(correction)
    with nvtx_range("stage:merge_scatter"):
        base.index_copy_(0, dense_indices, corrected)
    return base


def correctness(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    result: dict[str, float | bool] = {
        "correct": bool(
            torch.allclose(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
        ),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }
    if not result["correct"]:
        raise RuntimeError(f"separate residual-complement mismatch: {result}")
    return result


def worker(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run the worker with real GPU access")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    n, k = TP1_FUSED_WEIGHT_SHAPES[args.model]["o"]
    case = common.ShapeCase(args.model, "o", args.m, k, n)
    route_record = common.generate_routes(
        [args.m], [args.dense_fraction], args.seed
    )["routes"][common.route_key(args.m, args.dense_fraction)]
    route = common.route_from_record(route_record, device)
    weight, weight24 = common.make_synthetic_weight(case, args.seed, device)
    x = common.make_input(case, args.seed, device, purpose="separate_nsys")
    canonical = prepare_online_sparse24_weight(
        weight, weight24, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical, sparse_weight=weight24
    )
    algorithm_id = select_cusparselt_algorithm(runtime.cusparselt, x)
    resources = common.create_multistream_resources(device)

    dense_x = x.index_select(0, route.dense_indices)
    dense_reference = F.linear(dense_x, weight)
    sparse_reference = F.linear(x, weight24)
    expected = sparse_reference.clone()
    expected.index_copy_(0, route.dense_indices, dense_reference)

    call = lambda: launch_profiled_separate(
        x,
        route.dense_indices,
        runtime,
        resources,
        complement_variant=args.complement_variant,
    )
    captured = common.capture_multistream_graph(
        call, resources, warmup=3, device=device
    )
    for _ in range(args.warmup):
        captured.graph.replay()
    torch.cuda.synchronize(device)
    check = correctness(captured.output, expected)

    timing_samples: list[float] = []
    if args.timing_trials:
        timing_samples = [
            common.steady_graph_sample(captured, args.replays_per_sample)
            for _ in range(args.timing_trials)
        ]

    payload = {
        "case": case.key,
        "model": args.model,
        "projection": "o",
        "M": args.m,
        "N": n,
        "K": k,
        "dense_rows": route.dense_count,
        "sparse_rows": route.sparse_count,
        "dense_fraction": str(args.dense_fraction),
        "complement_variant": args.complement_variant,
        "warmup_graph_replays": args.warmup,
        "cusparselt_algorithm_id": algorithm_id,
        "correctness": check,
    }
    if timing_samples:
        payload["formal_e2e_us"] = {
            **common.summarize(timing_samples),
            "samples_us": timing_samples,
            "trials": args.timing_trials,
            "replays_per_sample": args.replays_per_sample,
            "timing_boundary": "natural_steady_state_no_eviction",
        }
    if args.timing_only:
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0

    torch.cuda.profiler.start()
    # One eager calibration on the same streams establishes an exact mapping
    # from NVTX stage to kernel/grid/block/stream signature.  Its kernels are
    # excluded from the measured iterations below.
    with torch.cuda.stream(resources.capture_stream):
        launch_profiled_separate(
            x,
            route.dense_indices,
            runtime,
            resources,
            complement_variant=args.complement_variant,
        )
    torch.cuda.synchronize(device)
    for repetition in range(args.repetitions):
        with nvtx_range(f"iteration:{repetition}"):
            captured.graph.replay()
    torch.cuda.synchronize(device)
    torch.cuda.profiler.stop()

    payload.update(
        {
            "profiled_iterations": args.repetitions,
            "profiled_execution": "node-level CUDA Graph replay",
        }
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def nsys_rows(sqlite_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    connection = sqlite3.connect(sqlite_path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        kernel_tables = sorted(
            table
            for table in tables
            if table.startswith("CUPTI_ACTIVITY_KIND_")
            and table.endswith("KERNEL")
        )
        if len(kernel_tables) != 1:
            raise RuntimeError(f"unexpected NSYS kernel tables: {kernel_tables}")
        kernel_table = kernel_tables[0]
        required_tables = {
            kernel_table,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "NVTX_EVENTS",
            "StringIds",
        }
        missing_tables = required_tables - tables
        if missing_tables:
            raise RuntimeError(f"NSYS export missing {sorted(missing_tables)}")
        required_kernel_columns = {
            "start",
            "end",
            "streamId",
            "correlationId",
            "graphId",
            "graphNodeId",
            "gridX",
            "blockX",
            "shortName",
        }
        missing_columns = required_kernel_columns - table_columns(
            connection, kernel_table
        )
        if missing_columns:
            raise RuntimeError(
                f"NSYS kernel table missing {sorted(missing_columns)}"
            )

        range_query = """
            SELECT e.start, e.end, COALESCE(e.text, s.value), e.globalTid
            FROM NVTX_EVENTS AS e
            LEFT JOIN StringIds AS s ON s.id = e.textId
            WHERE e.end IS NOT NULL
              AND (e.text LIKE 'stage:%' OR e.text LIKE 'iteration:%'
                   OR s.value LIKE 'stage:%' OR s.value LIKE 'iteration:%')
            ORDER BY e.start
        """
        ranges = [
            {
                "start_ns": int(start),
                "end_ns": int(end),
                "message": str(message),
                "global_tid": int(global_tid),
            }
            for start, end, message, global_tid in connection.execute(range_query)
        ]
        stage_ranges = [
            item for item in ranges if item["message"].startswith("stage:")
        ]
        iteration_ranges = [
            item for item in ranges if item["message"].startswith("iteration:")
        ]
        if not stage_ranges or not iteration_ranges:
            raise RuntimeError("NSYS export contains no profile NVTX ranges")

        runtime_query = """
            SELECT r.start, r.end, r.globalTid, r.correlationId, s.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            LEFT JOIN StringIds AS s ON s.id = r.nameId
            ORDER BY r.start
        """
        runtimes_by_correlation: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for start, end, global_tid, correlation_id, name in connection.execute(
            runtime_query
        ):
            runtimes_by_correlation[int(correlation_id)].append(
                {
                    "start_ns": int(start),
                    "end_ns": int(end),
                    "global_tid": int(global_tid),
                    "runtime_name": str(name),
                }
            )

        raw_kernels: list[dict[str, Any]] = []
        kernel_query = f"""
            SELECT k.start, k.end, k.streamId, k.correlationId,
                   k.graphId, k.graphNodeId, k.gridX, k.blockX, s.value
            FROM {kernel_table} AS k
            LEFT JOIN StringIds AS s ON s.id = k.shortName
            ORDER BY k.start
        """
        for (
            start,
            end,
            stream_id,
            correlation_id,
            graph_id,
            graph_node_id,
            grid_x,
            block_x,
            kernel_name,
        ) in connection.execute(kernel_query):
            raw_kernels.append(
                {
                    "start_ns": int(start),
                    "end_ns": int(end),
                    "duration_us": (int(end) - int(start)) / 1000.0,
                    "stream_id": int(stream_id),
                    "grid_x": int(grid_x),
                    "block_x": int(block_x),
                    "correlation_id": int(correlation_id),
                    "graph_id": None if graph_id is None else int(graph_id),
                    "graph_node_id": (
                        None if graph_node_id is None else int(graph_node_id)
                    ),
                    "kernel_name": str(kernel_name),
                }
            )

        def enclosing_ranges(
            runtime: dict[str, Any], candidates: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            return [
                item
                for item in candidates
                if item["global_tid"] == runtime["global_tid"]
                and item["start_ns"] <= runtime["start_ns"] <= item["end_ns"]
            ]

        def signature(kernel: dict[str, Any]) -> tuple[str, int, int]:
            return (
                str(kernel["kernel_name"]),
                int(kernel["grid_x"]),
                int(kernel["block_x"]),
            )

        # Calibration kernels have a stage NVTX range but no iteration range.
        # Eager streams and CUDA Graph virtual streams have unrelated IDs, so
        # streamId must not be part of the signature.  A signature can map to
        # two stages (the two index_select kernels); graph-stream affinity
        # disambiguates that known collision below.
        signature_to_stages: dict[tuple[str, int, int], set[str]] = defaultdict(
            set
        )
        for kernel in raw_kernels:
            if kernel["graph_node_id"] is not None:
                continue
            for runtime in runtimes_by_correlation.get(
                int(kernel["correlation_id"]), []
            ):
                stages = enclosing_ranges(runtime, stage_ranges)
                iterations = enclosing_ranges(runtime, iteration_ranges)
                if not stages or iterations:
                    continue
                stage_range = min(
                    stages,
                    key=lambda item: item["end_ns"] - item["start_ns"],
                )
                stage = stage_range["message"].removeprefix("stage:")
                signature_to_stages[signature(kernel)].add(stage)
        calibrated_stages = {
            stage for stages in signature_to_stages.values() for stage in stages
        }
        if calibrated_stages != set(STAGES):
            raise RuntimeError(
                "calibration did not identify every stage: "
                f"{sorted(calibrated_stages)}"
            )

        # Correlation IDs are not a reliable per-replay label when many graph
        # launches are queued.  graphNodeId is stable across replays, so the
        # N-th occurrence of each node belongs to replay N.
        graph_kernels = [
            kernel for kernel in raw_kernels if kernel["graph_node_id"] is not None
        ]
        by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for kernel in graph_kernels:
            by_node[int(kernel["graph_node_id"])].append(kernel)
        if not by_node:
            raise RuntimeError("NSYS export contains no CUDA Graph node kernels")
        expected_iterations = {
            int(item["message"].removeprefix("iteration:"))
            for item in iteration_ranges
        }
        if expected_iterations != set(range(len(expected_iterations))):
            raise RuntimeError(
                f"unexpected iteration NVTX labels: {sorted(expected_iterations)}"
            )
        wrong_counts = {
            node_id: len(occurrences)
            for node_id, occurrences in by_node.items()
            if len(occurrences) != len(expected_iterations)
        }
        if wrong_counts:
            raise RuntimeError(
                "CUDA Graph node occurrence counts do not match profiled replays: "
                f"{wrong_counts}"
            )

        node_to_stage: dict[int, str] = {}
        ambiguous_nodes: dict[int, set[str]] = {}
        for node_id, occurrences in by_node.items():
            keys = {signature(kernel) for kernel in occurrences}
            streams = {int(kernel["stream_id"]) for kernel in occurrences}
            if len(keys) != 1 or len(streams) != 1:
                raise RuntimeError(
                    f"graph node {node_id} changed signature/stream across replays"
                )
            candidates = signature_to_stages.get(next(iter(keys)), set())
            if len(candidates) == 1:
                node_to_stage[node_id] = next(iter(candidates))
            elif candidates:
                ambiguous_nodes[node_id] = candidates

        complement_streams = {
            int(by_node[node_id][0]["stream_id"])
            for node_id, stage in node_to_stage.items()
            if stage == "complement"
        }
        if len(complement_streams) != 1:
            raise RuntimeError(
                "could not identify the CUDA Graph complement stream: "
                f"{sorted(complement_streams)}"
            )
        complement_stream = next(iter(complement_streams))
        gather_collision = {"activation_gather", "merge_base_gather"}
        for node_id, candidates in ambiguous_nodes.items():
            if candidates != gather_collision:
                raise RuntimeError(
                    f"unsupported ambiguous graph node {node_id}: "
                    f"{sorted(candidates)}"
                )
            node_to_stage[node_id] = (
                "activation_gather"
                if int(by_node[node_id][0]["stream_id"]) == complement_stream
                else "merge_base_gather"
            )

        mapped_stages = set(node_to_stage.values())
        if mapped_stages != set(STAGES):
            raise RuntimeError(
                f"graph nodes did not cover every stage: {sorted(mapped_stages)}"
            )

        rows: list[dict[str, Any]] = []
        unassigned = [
            str(kernel["kernel_name"])
            for kernel in graph_kernels
            if int(kernel["graph_node_id"]) not in node_to_stage
        ]
        for node_id, occurrences in by_node.items():
            if node_id not in node_to_stage:
                continue
            ordered = sorted(occurrences, key=lambda item: int(item["start_ns"]))
            for iteration, kernel in enumerate(ordered):
                runtime_candidates = runtimes_by_correlation.get(
                    int(kernel["correlation_id"]), []
                )
                runtime_name = (
                    str(runtime_candidates[0]["runtime_name"])
                    if runtime_candidates
                    else "CUDA Graph node"
                )
                rows.append(
                    {
                        "iteration": iteration,
                        "stage": node_to_stage[node_id],
                        "start_ns": kernel["start_ns"],
                        "end_ns": kernel["end_ns"],
                        "duration_us": kernel["duration_us"],
                        "stream_id": kernel["stream_id"],
                        "grid_x": kernel["grid_x"],
                        "block_x": kernel["block_x"],
                        "correlation_id": kernel["correlation_id"],
                        "graph_id": kernel["graph_id"],
                        "graph_node_id": node_id,
                        "runtime_name": runtime_name,
                        "kernel_name": kernel["kernel_name"],
                    }
                )
    finally:
        connection.close()
    if not rows:
        raise RuntimeError(f"no staged GPU kernels found in {sqlite_path}")
    return rows, unassigned


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )


def distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def interval(rows: Sequence[dict[str, Any]]) -> tuple[int, int]:
    if not rows:
        raise ValueError("interval requires at least one kernel")
    return min(int(row["start_ns"]) for row in rows), max(
        int(row["end_ns"]) for row in rows
    )


def overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def aggregate_case(
    model: str,
    rows: list[dict[str, Any]],
    repetitions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_iteration[int(row["iteration"])].append(row)
    if set(by_iteration) != set(range(repetitions)):
        raise RuntimeError(
            f"{model}: expected iterations 0..{repetitions - 1}, "
            f"got {sorted(by_iteration)}"
        )

    iteration_rows: list[dict[str, Any]] = []
    stage_values: dict[str, list[float]] = defaultdict(list)
    stage_kernel_sums: dict[str, list[float]] = defaultdict(list)
    for iteration in range(repetitions):
        current = by_iteration[iteration]
        staged = {
            stage: [row for row in current if row["stage"] == stage]
            for stage in STAGES
        }
        missing = [stage for stage, stage_rows in staged.items() if not stage_rows]
        if missing:
            raise RuntimeError(f"{model} iteration {iteration} missing {missing}")
        intervals = {stage: interval(stage_rows) for stage, stage_rows in staged.items()}
        for stage, (start, end) in intervals.items():
            stage_values[stage].append((end - start) / 1000.0)
            stage_kernel_sums[stage].append(
                sum(float(row["duration_us"]) for row in staged[stage])
            )

        base = intervals["base"]
        gather = intervals["activation_gather"]
        complement = intervals["complement"]
        dense_branch = (gather[0], complement[1])
        merge = (
            intervals["merge_base_gather"][0],
            intervals["merge_scatter"][1],
        )
        first = min(base[0], dense_branch[0])
        branch_end = max(base[1], dense_branch[1])
        total = (merge[1] - first) / 1000.0
        iteration_rows.append(
            {
                "model": model,
                "iteration": iteration,
                "timeline_start_ns": first,
                "timeline_end_ns": merge[1],
                "timeline_total_us": total,
                "premerge_union_us": (branch_end - first) / 1000.0,
                "merge_interval_us": (merge[1] - merge[0]) / 1000.0,
                "base_dense_branch_overlap_us": overlap(base, dense_branch)
                / 1000.0,
                "base_complement_overlap_us": overlap(base, complement) / 1000.0,
                "base_duration_us": (base[1] - base[0]) / 1000.0,
                "dense_branch_duration_us": (dense_branch[1] - dense_branch[0])
                / 1000.0,
                "complement_exposed_after_base_us": max(
                    0, complement[1] - base[1]
                )
                / 1000.0,
                "base_exposed_after_complement_us": max(
                    0, base[1] - complement[1]
                )
                / 1000.0,
                "wait_to_merge_us": max(0, merge[0] - branch_end) / 1000.0,
            }
        )

    stage_rows = []
    for stage in STAGES:
        stage_rows.append(
            {
                "model": model,
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                **distribution(stage_values[stage]),
                "kernel_sum_median_us": statistics.median(
                    stage_kernel_sums[stage]
                ),
                "kernels_per_iteration": len(
                    [row for row in by_iteration[0] if row["stage"] == stage]
                ),
            }
        )
    return iteration_rows, stage_rows


def read_formal() -> dict[str, dict[str, float]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "results/residual_complement_fused_all_shapes_20260721_formal/formal_summary.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["projection"] != "o" or int(row["M"]) != DEFAULT_M:
            continue
        result[row["model"]][row["method"]] = float(row["median_us"])
    return dict(result)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def representative_iteration(
    model: str, iteration_rows: list[dict[str, Any]]
) -> int:
    candidates = [row for row in iteration_rows if row["model"] == model]
    median = statistics.median(float(row["timeline_total_us"]) for row in candidates)
    return int(
        min(candidates, key=lambda row: abs(float(row["timeline_total_us"]) - median))[
            "iteration"
        ]
    )


def plot_timeline(
    output_root: Path,
    kernel_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
) -> None:
    colors = {
        "activation_gather": "#72B7B2",
        "base": "#4C78A8",
        "complement": "#E45756",
        "merge_base_gather": "#F2CF5B",
        "merge_add": "#B279A2",
        "merge_scatter": "#FF9DA6",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.8), sharex=False)
    y_positions = {"base": 1.0, "dense": 0.0, "merge": -1.0}
    for axis, model in zip(axes, MODELS, strict=True):
        iteration = representative_iteration(model, iteration_rows)
        selected = [
            row
            for row in kernel_rows
            if row["model"] == model and int(row["iteration"]) == iteration
        ]
        origin = min(int(row["start_ns"]) for row in selected)
        for row in selected:
            stage = str(row["stage"])
            lane = (
                "base"
                if stage == "base"
                else "merge"
                if stage.startswith("merge_")
                else "dense"
            )
            start = (int(row["start_ns"]) - origin) / 1000.0
            width = float(row["duration_us"])
            axis.barh(
                y_positions[lane],
                width,
                left=start,
                height=0.55,
                color=colors[stage],
                edgecolor="black",
                linewidth=0.3,
                label=STAGE_LABELS[stage],
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8, loc="upper right")
        axis.set_yticks(
            [y_positions["merge"], y_positions["dense"], y_positions["base"]],
            ["Origin merge", "Dense branch", "Base branch"],
        )
        axis.set_title(
            f"{MODEL_LABELS[model]} o_proj, M=512, representative iteration {iteration}"
        )
        axis.set_xlabel("GPU timeline from first staged kernel (us)")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "separate_nsys_timeline.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_stage_medians(
    output_root: Path, stage_rows: list[dict[str, Any]]
) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 4.8))
    x = np.arange(len(STAGES))
    width = 0.36
    for index, model in enumerate(MODELS):
        lookup = {
            str(row["stage"]): float(row["median_us"])
            for row in stage_rows
            if row["model"] == model
        }
        axis.bar(
            x + (index - 0.5) * width,
            [lookup[stage] for stage in STAGES],
            width,
            label=MODEL_LABELS[model],
        )
    axis.set_xticks(x, [STAGE_LABELS[stage] for stage in STAGES], rotation=18, ha="right")
    axis.set_ylabel("Median GPU kernel interval (us)")
    axis.set_title("Separate residual-complement stage durations (non-additive)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "separate_stage_medians.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def format_value(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    output_root: Path,
    iteration_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    formal: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Separate residual-complement NSYS breakdown",
        "",
        "The GPU stage durations below are medians across 20 post-warmup CUDA Graph replays. "
        "Base and dense branches overlap, so stage durations are intentionally non-additive. "
        "Formal E2E remains the 10 x 1000 CUDA Graph median.",
        "",
        "| Model | Formal separate | Formal pure 2:4 | NSYS timeline | Base | Act gather | Complement | Merge | Base/complement overlap | Complement tail after base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        iterations = [row for row in iteration_rows if row["model"] == model]
        stages = {
            str(row["stage"]): float(row["median_us"])
            for row in stage_rows
            if row["model"] == model
        }
        metric = lambda key: statistics.median(
            float(row[key]) for row in iterations
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS[model],
                    format_value(formal[model]["residual_complement_current"]),
                    format_value(formal[model]["cusparselt_pure24"]),
                    format_value(metric("timeline_total_us")),
                    format_value(stages["base"]),
                    format_value(stages["activation_gather"]),
                    format_value(stages["complement"]),
                    format_value(metric("merge_interval_us")),
                    format_value(metric("base_complement_overlap_us")),
                    format_value(metric("complement_exposed_after_base_us")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Activation gather, complement, and merge are distinct GPU work; the current "
            "baseline does not fuse any of them.",
            "- The timeline exposes whether the correction finishes inside the all-row base "
            "window or creates a critical-path tail.",
            "- The first implementation target is an indexed serial in-place correction for "
            "tail-dominated shapes, followed by a single concurrent merge kernel.",
            "",
            "Artifacts: [raw kernel timeline](kernel_timeline.csv), "
            "[iteration metrics](iteration_summary.csv), "
            "[stage medians](stage_summary.csv), "
            "[timeline figure](figures/separate_nsys_timeline.png), and "
            "[stage figure](figures/separate_stage_medians.png).",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_worker_payload(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "case" in value:
            return value
    raise RuntimeError(f"worker emitted no JSON payload:\n{output}")


def driver(args: argparse.Namespace) -> int:
    if not NSYS.is_file():
        raise RuntimeError(f"Nsight Systems is missing: {NSYS}")
    common.require_idle_gpu(args.device_index)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    nsys_dir = output_root / "nsys"
    nsys_dir.mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    payloads = []
    all_kernel_rows: list[dict[str, Any]] = []
    all_iteration_rows: list[dict[str, Any]] = []
    all_stage_rows: list[dict[str, Any]] = []
    unassigned_by_model: dict[str, list[str]] = {}

    for model in args.models:
        common.require_idle_gpu(args.device_index)
        prefix = nsys_dir / f"{model}_o_m{args.m}_separate"
        command = [
            str(NSYS),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
            "--export=sqlite",
            f"--output={prefix}",
            sys.executable,
            str(script),
            "--worker",
            "--model",
            model,
            "--m",
            str(args.m),
            "--dense-fraction",
            str(args.dense_fraction),
            "--complement-variant",
            args.complement_variant,
            "--device-index",
            str(args.device_index),
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
        ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[4],
        )
        (nsys_dir / f"{model}_worker.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        payloads.append(parse_worker_payload(completed.stdout))
        sqlite_path = prefix.with_suffix(".sqlite")
        if not sqlite_path.is_file():
            raise RuntimeError(f"NSYS did not export {sqlite_path}")
        kernel_rows, unassigned = nsys_rows(sqlite_path)
        for row in kernel_rows:
            row["model"] = model
        iteration_rows, stage_rows = aggregate_case(
            model, kernel_rows, args.repetitions
        )
        all_kernel_rows.extend(kernel_rows)
        all_iteration_rows.extend(iteration_rows)
        all_stage_rows.extend(stage_rows)
        unassigned_by_model[model] = unassigned

    formal = read_formal()
    write_csv(output_root / "kernel_timeline.csv", all_kernel_rows)
    write_csv(output_root / "iteration_summary.csv", all_iteration_rows)
    write_csv(output_root / "stage_summary.csv", all_stage_rows)
    common.write_json(output_root / "worker_payloads.json", payloads)
    common.write_json(
        output_root / "analysis_metadata.json",
        {
            "models": list(args.models),
            "M": args.m,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "stage_attribution": (
                "eager NVTX/runtime calibration + stable CUDA Graph node ID; "
                "node occurrence rank assigns replay"
            ),
            "unassigned_profiled_kernels": unassigned_by_model,
            "formal_source": (
                "results/residual_complement_fused_all_shapes_20260721_formal/"
                "formal_summary.csv"
            ),
        },
    )
    plot_timeline(output_root, all_kernel_rows, all_iteration_rows)
    plot_stage_medians(output_root, all_stage_rows)
    write_report(output_root, all_iteration_rows, all_stage_rows, formal)
    print(output_root)
    return 0


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    if not models or len(set(models)) != len(models):
        raise argparse.ArgumentTypeError("models must be a non-empty unique list")
    unknown = set(models) - set(MODELS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported models: {sorted(unknown)}")
    return models


def parse_fraction(value: str) -> Fraction:
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(f"invalid fraction: {value}") from error
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("dense fraction must be in (0, 1]")
    return fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--models", type=parse_models, default=MODELS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument(
        "--dense-fraction", type=parse_fraction, default=DEFAULT_DENSE_FRACTION
    )
    parser.add_argument(
        "--complement-variant",
        choices=tuple(COMPLEMENT_CTA_VARIANTS),
        default="auto",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--timing-trials", type=int, default=0)
    parser.add_argument("--replays-per-sample", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "results/separate_residual_complement_nsys_breakdown_20260721"
        ),
    )
    args = parser.parse_args()
    if args.m <= 0 or args.m % 8:
        parser.error("M must be positive and divisible by 8")
    if args.warmup <= 0 or args.repetitions <= 0:
        parser.error("warmup and repetitions must be positive")
    if args.timing_trials < 0 or args.replays_per_sample <= 0:
        parser.error("timing trials must be non-negative and replays positive")
    if args.timing_only and args.timing_trials <= 0:
        parser.error("--timing-only requires positive --timing-trials")
    if args.worker and args.model is None:
        parser.error("--worker requires --model")
    if not args.worker and args.dense_fraction != DEFAULT_DENSE_FRACTION:
        parser.error("the built-in report driver supports only dense fraction 1/8")
    if not args.worker and args.complement_variant != "auto":
        parser.error("the built-in report driver supports only complement variant auto")
    return args


def main() -> int:
    args = parse_args()
    return worker(args) if args.worker else driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
