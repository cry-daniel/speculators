#!/usr/bin/env python3
"""Formal residual-complement fused benchmark over all requested LLM linears.

The current method runs a prepared cuSPARSELt base 2:4 GEMM for every row and
a concurrent CUTLASS complement 2:4 GEMM for dense-routed rows, followed by an
add/scatter merge.  The fused method instead partitions rows: sparse rows run
the prepared base GEMM, while dense rows run one CUTLASS kernel whose two
HMMA.SP operations share activation B and metadata E stages, one FP32
accumulator, and one epilogue.

Formal latency uses 100 warmup CUDA Graph replays and 10 samples of 1000
continuous replays.  One Event pair surrounds each sample and synchronization
is outside the replay loop.  Five method orders and their reverses provide
pairwise 5:5 order balance.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import sparse24_benchmark_common as common  # noqa: E402
from sparse24_benchmark_common import (  # noqa: E402
    CapturedGraph,
    MultiStreamResources,
    capture_graph,
    capture_multistream_graph,
    create_multistream_resources,
)
from residual_complement_runtime import (  # noqa: E402
    launch_fused,
    launch_separate,
)
from speculators.speclink import (  # noqa: E402
    FUSED_BASE_COMPLEMENT_VARIANTS,
    SPARSE_RESIDUAL_SMEM,
    TP1_FUSED_WEIGHT_SHAPES,
    cusparselt_sparse_residual_fused_dense_linear,
    cusparselt_sparse_residual_fused_kernel_attributes,
    cusparselt_sparse_residual_kernel_attributes,
    cusparselt_sparse_residual_sparse_linear,
    prepare_cusparselt_sparse_residual_weight,
    prepare_online_sparse24_weight,
    select_cusparselt_algorithm,
)


SCRIPT_PATH = Path(__file__).resolve()
EVAL_ROOT = SCRIPT_PATH.parent.parent

CURRENT = "residual_complement_current"
FUSED = "residual_complement_fused"
CUBLAS = "cublas_dense"
PURE24 = "cusparselt_pure24"
METHODS = (CURRENT, FUSED, CUBLAS, PURE24)
METHOD_LABELS = {
    CURRENT: "Current: all-row base + complement",
    FUSED: "Fused: sparse base || dense dual-HMMA.SP",
    CUBLAS: "cuBLAS dense",
    PURE24: "cuSPARSELt pure 2:4",
}

MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
PROJECTIONS = ("qkv", "o", "gate_up", "down")
M_VALUES = (512, 1024, 2048)
DENSE_FRACTION = Fraction(1, 8)
WARMUP_REPLAYS = 100
TRIALS = 10
REPLAYS_PER_SAMPLE = 1000
CAPTURE_WARMUP = 3
SCREEN_WARMUP = 100
SCREEN_TRIALS = 5
SCREEN_REPLAYS = 1000
DEFAULT_SEED = 20260721
DEFAULT_RTOL = 5e-2
DEFAULT_ATOL = 5e-2


def parse_csv_choices(
    value: str, allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError(f"{label} must be nonempty")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {label}: {unknown}")
    if len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError(f"duplicate {label}")
    return selected


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("M values must be integers") from error
    if not selected or any(item not in M_VALUES for item in selected):
        raise argparse.ArgumentTypeError(f"M values must be selected from {M_VALUES}")
    if len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("duplicate M values")
    return selected


def cases_for(
    models: Sequence[str], projections: Sequence[str], m_values: Sequence[int]
) -> list[common.ShapeCase]:
    return [
        common.ShapeCase(model, projection, m, k, n)
        for model in models
        for projection in projections
        for m in m_values
        for n, k in (TP1_FUSED_WEIGHT_SHAPES[model][projection],)
    ]


def method_orders() -> tuple[tuple[str, ...], ...]:
    base = (
        (CURRENT, FUSED, CUBLAS, PURE24),
        (FUSED, CUBLAS, PURE24, CURRENT),
        (CUBLAS, PURE24, CURRENT, FUSED),
        (PURE24, CURRENT, FUSED, CUBLAS),
        (CURRENT, CUBLAS, FUSED, PURE24),
    )
    orders = base + tuple(tuple(reversed(order)) for order in base)
    if len(orders) != TRIALS:
        raise AssertionError("formal order count changed")
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            before = sum(order.index(left) < order.index(right) for order in orders)
            if before != TRIALS // 2:
                raise AssertionError(f"unbalanced pair {left}/{right}: {before}")
    return orders


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: Sequence[float]) -> dict[str, float]:
    if len(values) != TRIALS:
        raise RuntimeError(f"expected {TRIALS} formal samples")
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
        "mean_us": statistics.mean(values),
        "cv": statistics.pstdev(values) / statistics.mean(values),
    }


def correctness(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    record = {
        "correct": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "rtol": rtol,
        "atol": atol,
    }
    if not record["correct"]:
        raise RuntimeError(f"correctness failure: {record}")
    return record


def screen_fused_variants(
    dense_x: torch.Tensor,
    runtime: Any,
    dense_reference: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for variant in FUSED_BASE_COMPLEMENT_VARIANTS:
        call = lambda variant=variant: cusparselt_sparse_residual_fused_dense_linear(
            dense_x, runtime, variant=variant
        )
        eager = call()
        check = correctness(eager, dense_reference, rtol=rtol, atol=atol)
        for _ in range(SCREEN_WARMUP):
            call()
        torch.cuda.synchronize(dense_x.device)
        samples: list[float] = []
        for _ in range(SCREEN_TRIALS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(SCREEN_REPLAYS):
                call()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / SCREEN_REPLAYS)
        attributes = cusparselt_sparse_residual_fused_kernel_attributes(
            dense_x.shape[0], runtime.n, variant=variant
        )
        records.append(
            {
                "variant": variant,
                "median_us": statistics.median(samples),
                "samples_us": samples,
                "correctness": check,
                "kernel_attributes": attributes,
            }
        )
    winner = min(records, key=lambda row: float(row["median_us"]))["variant"]
    return str(winner), records


def screen_captured_fused_e2e_variants(
    captured_by_variant: dict[str, CapturedGraph],
) -> tuple[str, list[dict[str, Any]]]:
    """Select using the exact graph instances retained for formal timing.

    Short screens can select a transient concurrent-CTA schedule.  Each graph
    is therefore kept alive, warmed for the same 100 replays as the formal
    protocol, and measured over five 1000-replay intervals before selection.
    """
    records: list[dict[str, Any]] = []
    for variant in FUSED_BASE_COMPLEMENT_VARIANTS:
        captured = captured_by_variant[variant]
        for _ in range(SCREEN_WARMUP):
            captured.graph.replay()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(SCREEN_TRIALS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(SCREEN_REPLAYS):
                captured.graph.replay()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / SCREEN_REPLAYS)
        records.append(
            {
                "variant": variant,
                "e2e_median_us": statistics.median(samples),
                "e2e_p90_us": percentile(samples, 0.9),
                "e2e_samples_us": samples,
            }
        )
    winner = min(records, key=lambda row: float(row["e2e_p90_us"]))[
        "variant"
    ]
    return str(winner), records


def formal_measure(
    captured: dict[str, CapturedGraph],
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    for method in METHODS:
        for _ in range(WARMUP_REPLAYS):
            captured[method].graph.replay()
    torch.cuda.synchronize()
    values = {method: [] for method in METHODS}
    raw: list[dict[str, Any]] = []
    for trial, order in enumerate(method_orders()):
        for position, method in enumerate(order):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter()
            start.record()
            for _ in range(REPLAYS_PER_SAMPLE):
                captured[method].graph.replay()
            end.record()
            enqueue_ms = (time.perf_counter() - host_start) * 1000.0
            end.synchronize()
            total_ms = float(start.elapsed_time(end))
            latency_us = total_ms * 1000.0 / REPLAYS_PER_SAMPLE
            values[method].append(latency_us)
            raw.append(
                {
                    "trial": trial,
                    "position": position,
                    "order": ",".join(order),
                    "method": method,
                    "latency_us": latency_us,
                    "total_gpu_ms": total_ms,
                    "host_enqueue_ms": enqueue_ms,
                    "graph_replays": REPLAYS_PER_SAMPLE,
                    "cuda_event_pairs": 1,
                    "synchronize_inside_replay_loop": False,
                }
            )
    return values, raw


def run_worker(args: argparse.Namespace, case: common.ShapeCase) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    routes = common.generate_routes([case.m], [DENSE_FRACTION], args.seed)
    route_record = routes["routes"][common.route_key(case.m, DENSE_FRACTION)]
    route = common.route_from_record(route_record, device)
    dense_rows = case.m // 8
    sparse_rows = case.m - dense_rows
    if (route.dense_count, route.sparse_count) != (dense_rows, sparse_rows):
        raise RuntimeError("route is not exactly 1/8 dense")

    weight, weight24 = common.make_synthetic_weight(case, args.seed, device)
    x = common.make_input(case, args.seed, device, purpose="residual_fused")
    dense_x = x.index_select(0, route.dense_indices).contiguous()
    sparse_x = x.index_select(0, route.sparse_indices).contiguous()
    dense_reference = F.linear(dense_x, weight)
    sparse_reference = F.linear(sparse_x, weight24)
    hybrid_reference = torch.empty(
        (case.m, case.n), device=device, dtype=torch.bfloat16
    )
    hybrid_reference.index_copy_(0, route.dense_indices, dense_reference)
    hybrid_reference.index_copy_(0, route.sparse_indices, sparse_reference)
    cublas_reference = F.linear(x, weight)
    pure_reference = F.linear(x, weight24)

    canonical = prepare_online_sparse24_weight(
        weight, weight24, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical, sparse_weight=weight24
    )
    expected_storage = 2 * case.n * case.k + case.n * case.k // 8
    if runtime.persistent_bytes() != expected_storage:
        raise RuntimeError("one-weight persistent byte contract failed")

    dense_kernel_winner, dense_kernel_tuning = screen_fused_variants(
        dense_x,
        runtime,
        dense_reference,
        rtol=args.rtol,
        atol=args.atol,
    )

    full_algorithm = select_cusparselt_algorithm(runtime.cusparselt, x)
    current_resources = create_multistream_resources(device)
    current_call = lambda: launch_separate(
        x, route.dense_indices, runtime, current_resources
    )
    current_captured = capture_multistream_graph(
        current_call,
        current_resources,
        warmup=CAPTURE_WARMUP,
        device=device,
    )
    pure_captured = capture_graph(
        lambda: cusparselt_sparse_residual_sparse_linear(x, runtime),
        warmup=CAPTURE_WARMUP,
        unroll=1,
    )

    sparse_algorithm = select_cusparselt_algorithm(runtime.cusparselt, sparse_x)
    fused_captured_by_variant: dict[str, CapturedGraph] = {}
    fused_resources_by_variant: dict[str, MultiStreamResources] = {}
    fused_outputs_by_variant: dict[str, torch.Tensor] = {}
    fused_checks_by_variant: dict[str, dict[str, Any]] = {}
    for variant in FUSED_BASE_COMPLEMENT_VARIANTS:
        resources = create_multistream_resources(device)
        output = torch.empty(
            (case.m, case.n), device=device, dtype=torch.bfloat16
        )
        call = lambda variant=variant, resources=resources, output=output: (
            launch_fused(
                x,
                route.dense_indices,
                route.sparse_indices,
                runtime,
                output,
                resources,
                variant=variant,
            )
        )
        captured_variant = capture_multistream_graph(
            call,
            resources,
            warmup=CAPTURE_WARMUP,
            device=device,
        )
        fused_captured_by_variant[variant] = captured_variant
        fused_resources_by_variant[variant] = resources
        fused_outputs_by_variant[variant] = output
        fused_checks_by_variant[variant] = correctness(
            captured_variant.output,
            hybrid_reference,
            rtol=args.rtol,
            atol=args.atol,
        )

    cublas_captured = capture_graph(
        lambda: F.linear(x, weight), warmup=CAPTURE_WARMUP, unroll=1
    )
    winner, e2e_tuning = screen_captured_fused_e2e_variants(
        fused_captured_by_variant
    )
    dense_by_variant = {
        str(record["variant"]): record for record in dense_kernel_tuning
    }
    e2e_by_variant = {str(record["variant"]): record for record in e2e_tuning}
    tuning = []
    for variant in FUSED_BASE_COMPLEMENT_VARIANTS:
        dense_record = dense_by_variant[variant]
        e2e_record = e2e_by_variant[variant]
        tuning.append(
            {
                "variant": variant,
                "dense_kernel_median_us": dense_record["median_us"],
                "dense_kernel_samples_us": dense_record["samples_us"],
                "dense_kernel_correctness": dense_record["correctness"],
                "kernel_attributes": dense_record["kernel_attributes"],
                "e2e_median_us": e2e_record["e2e_median_us"],
                "e2e_p90_us": e2e_record["e2e_p90_us"],
                "e2e_samples_us": e2e_record["e2e_samples_us"],
                "e2e_correctness": fused_checks_by_variant[variant],
                "dense_kernel_selected": variant == dense_kernel_winner,
                "e2e_selected": variant == winner,
            }
        )
    fused_captured = fused_captured_by_variant[winner]
    captured = {
        CURRENT: current_captured,
        FUSED: fused_captured,
        CUBLAS: cublas_captured,
        PURE24: pure_captured,
    }
    expected = {
        CURRENT: hybrid_reference,
        FUSED: hybrid_reference,
        CUBLAS: cublas_reference,
        PURE24: pure_reference,
    }
    checks = {
        method: correctness(
            captured[method].output,
            expected[method],
            rtol=args.rtol,
            atol=args.atol,
        )
        for method in METHODS
    }

    del canonical, weight24, dense_reference, sparse_reference
    del hybrid_reference, cublas_reference, pure_reference, dense_x, sparse_x
    gc.collect()
    torch.cuda.empty_cache()

    values, raw = formal_measure(captured)
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        storage_bytes = {
            CURRENT: expected_storage,
            FUSED: expected_storage,
            CUBLAS: 2 * case.n * case.k,
            PURE24: case.n * case.k + case.n * case.k // 8,
        }[method]
        summaries.append(
            {
                "case": case.key,
                "model": case.model,
                "projection": case.projection,
                "M": case.m,
                "N": case.n,
                "K": case.k,
                "dense_rows": dense_rows,
                "sparse_rows": sparse_rows,
                "method": method,
                "method_label": METHOD_LABELS[method],
                **summary(values[method]),
                "persistent_weight_bytes": storage_bytes,
                "fused_variant": winner if method == FUSED else None,
                "full_m_algorithm_id": full_algorithm,
                "sparse_rows_algorithm_id": sparse_algorithm,
                "correctness": checks[method],
            }
        )
    for row in raw:
        row.update(
            {
                "case": case.key,
                "model": case.model,
                "projection": case.projection,
                "M": case.m,
                "N": case.n,
                "K": case.k,
            }
        )
    return {
        "case": asdict(case),
        "summaries": summaries,
        "raw": raw,
        "tuning": tuning,
        "winner": winner,
        "dense_kernel_winner": dense_kernel_winner,
        "storage": {
            "representation": "cusparselt_packed_base_values_and_metadata+compact_complement",
            "persistent_bytes": expected_storage,
            "metadata_payloads": 1,
            "dense_weight_copies": 0,
        },
        "current_complement_attributes": (
            cusparselt_sparse_residual_kernel_attributes(
                dense_rows, case.n, residual_only=True
            )
        ),
        "formal_protocol": {
            "warmup_replays": WARMUP_REPLAYS,
            "trials": TRIALS,
            "replays_per_sample": REPLAYS_PER_SAMPLE,
            "order": "five orders plus exact reverses; every pair 5:5",
            "cache_state": "natural steady state; no explicit eviction between graph replays",
        },
        "fused_screen_protocol": {
            "warmup_replays": SCREEN_WARMUP,
            "trials": SCREEN_TRIALS,
            "replays_per_sample": SCREEN_REPLAYS,
            "selection_target": "complete two-stream E2E graph P90",
            "graph_identity": "same retained graph instance used by formal timing",
        },
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    flattened: list[dict[str, Any]] = []
    keys: list[str] = []
    for row in rows:
        flat = {
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in row.items()
        }
        flattened.append(flat)
        for key in flat:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flattened)


def read_shards(work_root: Path, cases: Sequence[common.ShapeCase]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for case in cases:
        path = work_root / "shards" / f"{case.key}.json"
        if not path.is_file():
            raise RuntimeError(f"missing formal shard: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("case") != asdict(case):
            raise RuntimeError(f"shard identity mismatch: {path}")
        if len(payload.get("summaries", ())) != len(METHODS):
            raise RuntimeError(f"incomplete summaries: {path}")
        if len(payload.get("raw", ())) != len(METHODS) * TRIALS:
            raise RuntimeError(f"incomplete raw samples: {path}")
        payloads.append(payload)
    return payloads


def plot_results(rows: Sequence[dict[str, Any]], output_root: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = {
        (str(row["model"]), str(row["projection"]), int(row["M"]), str(row["method"])): row
        for row in rows
    }
    available = {
        (str(row["model"]), str(row["projection"])) for row in rows
    }
    row_keys = [
        (model, projection)
        for model in MODELS
        for projection in PROJECTIONS
        if (model, projection) in available
    ]
    m_values = [
        m for m in M_VALUES if any(int(row["M"]) == m for row in rows)
    ]
    labels = [f"{model}\n{projection}" for model, projection in row_keys]
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for axis, method, title in (
        (axes[0], CURRENT, "Current residual-complement / cuBLAS speedup"),
        (axes[1], FUSED, "Fused residual-complement / cuBLAS speedup"),
    ):
        matrix = np.array(
            [
                [
                    float(by_key[(model, projection, m, CUBLAS)]["median_us"])
                    / float(by_key[(model, projection, m, method)]["median_us"])
                    for m in m_values
                ]
                for model, projection in row_keys
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.8, vmax=1.8)
        axis.set_title(title)
        axis.set_xticks(range(len(m_values)), [f"M={m}" for m in m_values])
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}x",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axis, label="speedup vs cuBLAS")
    figure_dir = output_root / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure.savefig(figure_dir / "speedup_vs_cublas_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axes_grid = plt.subplots(
        len(m_values),
        1,
        figsize=(18, 4 * len(m_values)),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_grid[:, 0]
    x = np.arange(len(row_keys), dtype=float)
    width = 0.2
    for axis, m in zip(axes, m_values, strict=True):
        for offset_index, method in enumerate(METHODS):
            medians = [
                float(by_key[(model, projection, m, method)]["median_us"])
                for model, projection in row_keys
            ]
            offset = (offset_index - 1.5) * width
            axis.bar(x + offset, medians, width=width, label=METHOD_LABELS[method])
        axis.set_yscale("log")
        axis.set_ylabel("Median latency (us, log)")
        axis.set_title(f"M={m}")
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9)
    figure.savefig(figure_dir / "absolute_latency_all_shapes.png", dpi=180)
    plt.close(figure)


def write_report(rows: Sequence[dict[str, Any]], output_root: Path) -> None:
    by_key = {
        (str(row["model"]), str(row["projection"]), int(row["M"]), str(row["method"])): row
        for row in rows
    }
    lines = [
        "# Residual-complement fused: all model linears",
        "",
        "每个 case 固定 1/8 dense rows 与 7/8 structured-2:4 rows。当前方案是",
        "全 M 行 cuSPARSELt base 与 dense rows complement HMMA.SP 并发，随后 add/scatter。",
        "fused 方案把 sparse rows 的 base 分支与 dense rows 的 dual-HMMA.SP 分支并发；",
        "dual-HMMA.SP 在一个 CUTLASS kernel 内每个 K-stage 只加载一次 activation B 和",
        "唯一 metadata E，两份 compact A 更新同一个 FP32 accumulator，并只执行一次 epilogue。",
        "",
        "正式计时为 100 warmups、10 samples x 1000 CUDA Graph replays，方法顺序 pairwise 5:5；",
        "报告 median [P10, P90]，单位 us。结果是 natural steady-state，不在 replay 间做 cache eviction。",
        "三个 fused 配置均先 capture 并保持存活，以 100 warmups、5 samples x 1000 replays",
        "筛选完整双 stream E2E，并以 P90 选出稳健配置；正式计时复用胜出配置的同一个 graph 实例。",
        "",
        "| Model | Layer | M | Current | Fused | cuBLAS | pure 2:4 | Fused/current | Fused/cuBLAS | Variant |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    available = {
        (str(row["model"]), str(row["projection"])) for row in rows
    }
    row_keys = [
        (model, projection)
        for model in MODELS
        for projection in PROJECTIONS
        if (model, projection) in available
    ]
    m_values = [
        m for m in M_VALUES if any(int(row["M"]) == m for row in rows)
    ]
    fused_wins = 0
    cublas_wins = 0
    total = 0
    for model, projection in row_keys:
        for m in m_values:
            current = by_key[(model, projection, m, CURRENT)]
            fused = by_key[(model, projection, m, FUSED)]
            cublas = by_key[(model, projection, m, CUBLAS)]
            pure = by_key[(model, projection, m, PURE24)]
            current_us = float(current["median_us"])
            fused_us = float(fused["median_us"])
            cublas_us = float(cublas["median_us"])
            pure_us = float(pure["median_us"])
            fused_wins += fused_us < current_us
            cublas_wins += fused_us < cublas_us
            total += 1
            lines.append(
                f"| {model} | {projection} | {m} | "
                f"{current_us:.3f} [{current['p10_us']:.3f}, {current['p90_us']:.3f}] | "
                f"{fused_us:.3f} [{fused['p10_us']:.3f}, {fused['p90_us']:.3f}] | "
                f"{cublas_us:.3f} | {pure_us:.3f} | "
                f"{current_us / fused_us:.3f}x | {cublas_us / fused_us:.3f}x | "
                f"{fused['fused_variant']} |"
            )
    lines.extend(
        [
            "",
            f"Fused 在 {fused_wins}/{total} 个 case 快于 current，在 "
            f"{cublas_wins}/{total} 个 case 快于 cuBLAS dense。",
            "",
            "- [Formal summary](formal_summary.csv)",
            "- [Raw 10-sample data](formal_raw.csv)",
            "- [Per-shape fused configuration screening](fused_tuning.csv)",
            "- [Speedup heatmap](figures/speedup_vs_cublas_heatmap.png)",
            "- [Absolute latency plot](figures/absolute_latency_all_shapes.png)",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(
    payloads: Sequence[dict[str, Any]], output_root: Path, idle_records: Sequence[dict[str, Any]]
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [row for payload in payloads for row in payload["summaries"]]
    raw = [row for payload in payloads for row in payload["raw"]]
    tuning = []
    for payload in payloads:
        case = payload["case"]
        for record in payload["tuning"]:
            tuning.append(
                {
                    **case,
                    "variant": record["variant"],
                    "dense_kernel_median_us": record["dense_kernel_median_us"],
                    "e2e_median_us": record["e2e_median_us"],
                    "e2e_p90_us": record.get(
                        "e2e_p90_us",
                        percentile(record["e2e_samples_us"], 0.9),
                    ),
                    "selected": record["variant"] == payload["winner"],
                    "dense_kernel_selected": record["dense_kernel_selected"],
                    "dense_kernel_samples_us": record[
                        "dense_kernel_samples_us"
                    ],
                    "e2e_samples_us": record["e2e_samples_us"],
                    "kernel_attributes": record["kernel_attributes"],
                    "dense_kernel_correctness": record[
                        "dense_kernel_correctness"
                    ],
                    "e2e_correctness": record["e2e_correctness"],
                }
            )
    write_csv(output_root / "formal_summary.csv", summaries)
    write_csv(output_root / "formal_raw.csv", raw)
    write_csv(output_root / "fused_tuning.csv", tuning)
    metadata = {
        "measurement_date": datetime.now().astimezone().isoformat(),
        "models": list(dict.fromkeys(row["model"] for row in summaries)),
        "projections": list(dict.fromkeys(row["projection"] for row in summaries)),
        "M_values": list(dict.fromkeys(int(row["M"]) for row in summaries)),
        "cases": len(payloads),
        "methods": list(METHODS),
        "formal_protocol": {
            "warmup_replays": WARMUP_REPLAYS,
            "trials": TRIALS,
            "replays_per_sample": REPLAYS_PER_SAMPLE,
            "synchronization_inside_interval": False,
            "order_balance": "every method pair 5:5",
            "cache_state": "natural steady state continuous graph replay",
        },
        "fused_screen_protocol": {
            "warmup_replays": SCREEN_WARMUP,
            "trials": SCREEN_TRIALS,
            "replays_per_sample": SCREEN_REPLAYS,
            "selection_target": "complete two-stream E2E graph P90",
            "graph_identity": "same retained graph instance used by formal timing",
        },
        "gpu_idle_checks": list(idle_records),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    (output_root / "measurement_provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_results(summaries, output_root)
    write_report(summaries, output_root)


def worker_command(
    args: argparse.Namespace, case: common.ShapeCase, shard: Path
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--worker-model",
        case.model,
        "--worker-projection",
        case.projection,
        "--worker-m",
        str(case.m),
        "--worker-output",
        str(shard),
        "--device-index",
        str(args.device_index),
        "--seed",
        str(args.seed),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--projections", default=",".join(PROJECTIONS))
    parser.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--worker-projection", help=argparse.SUPPRESS)
    parser.add_argument("--worker-m", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.models = parse_csv_choices(args.models, MODELS, "models")
    args.projections = parse_csv_choices(
        args.projections, PROJECTIONS, "projections"
    )
    args.m_values = parse_int_csv(args.m_values)
    return args


def main() -> int:
    args = parse_args()
    if args.worker:
        if (
            args.worker_model not in MODELS
            or args.worker_projection not in PROJECTIONS
            or args.worker_m not in M_VALUES
            or args.worker_output is None
        ):
            raise ValueError("incomplete worker identity")
        n, k = TP1_FUSED_WEIGHT_SHAPES[args.worker_model][args.worker_projection]
        case = common.ShapeCase(
            args.worker_model, args.worker_projection, args.worker_m, k, n
        )
        payload = run_worker(args, case)
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root
        or EVAL_ROOT / "results" / f"residual_complement_fused_all_shapes_{timestamp}"
    ).resolve()
    work_root = (
        args.work_root
        or EVAL_ROOT / "temp" / f"residual_complement_fused_work_{timestamp}"
    ).resolve()
    (work_root / "shards").mkdir(parents=True, exist_ok=True)
    idle_root = work_root / "idle"
    idle_root.mkdir(parents=True, exist_ok=True)
    cases = cases_for(args.models, args.projections, args.m_values)
    prior_idle_by_case: dict[str, dict[str, Any]] = {}
    prior_provenance = output_root / "measurement_provenance.json"
    if args.resume and prior_provenance.is_file():
        prior_payload = json.loads(prior_provenance.read_text(encoding="utf-8"))
        prior_idle_by_case = {
            str(record["case"]): record
            for record in prior_payload.get("gpu_idle_checks", ())
        }
    idle_records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        shard = work_root / "shards" / f"{case.key}.json"
        idle_path = idle_root / f"{case.key}.json"
        if args.resume and shard.is_file():
            print(f"[{index}/{len(cases)}] reuse {case.key}", flush=True)
            prior_idle = (
                json.loads(idle_path.read_text(encoding="utf-8"))
                if idle_path.is_file()
                else prior_idle_by_case.get(case.key)
            )
            if prior_idle is not None:
                idle_records.append(prior_idle)
            continue
        idle = common.require_idle_gpu(args.device_index)
        idle["case"] = case.key
        idle_records.append(idle)
        idle_path.write_text(
            json.dumps(idle, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[{index}/{len(cases)}] run {case.key}", flush=True)
        subprocess.run(worker_command(args, case, shard), check=True, cwd=EVAL_ROOT)
        time.sleep(0.5)
    if len(idle_records) != len(cases):
        missing = sorted(
            {case.key for case in cases}
            - {str(record["case"]) for record in idle_records}
        )
        raise RuntimeError(f"missing GPU idle provenance for cases: {missing}")
    payloads = read_shards(work_root, cases)
    finalize(payloads, output_root, idle_records)
    run_manifest = {
        "output_root": str(output_root),
        "work_root": str(work_root),
        "cases": [asdict(case) for case in cases],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output_root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
