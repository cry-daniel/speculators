#!/usr/bin/env python3
"""Formal Split-K=2 separate residual-complement sweep over real LLM linears.

The coordinator launches one worker per (model, projection), so each large
synthetic weight is prepared only once.  Every worker evaluates M in
{512,1024,2048} and dense fractions in {1/8,1/4,1/2}.  Split-K=2 mainloops
are screened with HBM-cold replays; the selected hybrid graph, BF16 cuBLAS
dense, and pure 2:4 cuSPARSELt are then measured with 100 warmups and
10 x 1000 CUDA Graph replays.  A 256 MiB untimed eviction precedes every
measured replay.
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm

import sparse24_benchmark_common as common
from residual_complement_runtime import launch_separate
from speculators.speclink import (
    SPARSE_RESIDUAL_SMEM,
    TP1_FUSED_WEIGHT_SHAPES,
    prepare_cusparselt_sparse_residual_weight,
    prepare_online_sparse24_weight,
    cusparselt_sparse_residual_sparse_linear,
    select_cusparselt_algorithm,
)


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parent.parent
MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
PROJECTIONS = ("qkv", "o", "gate_up", "down")
M_VALUES = (512, 1024, 2048)
FRACTIONS = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama3-70B",
}
SPLIT_VARIANTS = (
    "b_resident_feature64_token64_b2a1",
    "b_resident_feature128_token64_b2a1",
    "feature64_token64_s4",
    "feature128_token64_s4",
)
P40_VARIANT = "b_resident_feature64_token64_b2a1_p40"
SEED = 20260721


def parse_csv(value: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(f"invalid {label}: {unknown or value!r}")
    return selected


def parse_m_values(value: str) -> tuple[int, ...]:
    selected = tuple(int(item) for item in value.split(",") if item)
    if not selected or any(item not in M_VALUES for item in selected):
        raise argparse.ArgumentTypeError(f"M must be selected from {M_VALUES}")
    return selected


def parse_fractions(value: str) -> tuple[Fraction, ...]:
    selected = tuple(Fraction(item) for item in value.split(",") if item)
    if not selected or any(item not in FRACTIONS for item in selected):
        raise argparse.ArgumentTypeError(
            f"fractions must be selected from {tuple(map(str, FRACTIONS))}"
        )
    return selected


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
        "mean_us": statistics.mean(values),
    }


def correctness(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    result = {
        "correct": bool(torch.allclose(actual, expected, rtol=5e-2, atol=5e-2)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }
    if not result["correct"]:
        raise RuntimeError(f"correctness failure: {result}")
    return result


class ColdTimer:
    def __init__(self, device: torch.device, max_replays: int, eviction_mib: int):
        self.eviction = torch.empty(
            eviction_mib * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.eviction.zero_()
        self.starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(max_replays)
        ]
        self.ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(max_replays)
        ]
        torch.cuda.synchronize(device)

    def sample(self, captured: common.CapturedGraph, replays: int) -> float:
        if replays > len(self.starts):
            raise ValueError("event pool is too small")
        for index in range(replays):
            self.eviction.add_(1)
            self.starts[index].record()
            captured.graph.replay()
            self.ends[index].record()
        torch.cuda.synchronize(self.eviction.device)
        return 1000.0 * sum(
            self.starts[index].elapsed_time(self.ends[index])
            for index in range(replays)
        ) / replays


def candidates_for(n: int) -> tuple[str, ...]:
    if n == 24576:
        return (*SPLIT_VARIANTS, "b_resident_feature128_token64_b2a1_p192")
    if n == 28672:
        return (*SPLIT_VARIANTS, "b_resident_feature128_token64_b2a1_p224")
    if n == 5120:
        return (P40_VARIANT, *SPLIT_VARIANTS[1:])
    return SPLIT_VARIANTS


def feature_tile(variant: str) -> int:
    return 128 if "feature128" in variant else 64


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = args.worker_model
    projection = args.worker_projection
    n, k = TP1_FUSED_WEIGHT_SHAPES[model][projection]
    weight_case = common.ShapeCase(model, projection, max(args.m_values), k, n)
    weight, weight24 = common.make_synthetic_weight(weight_case, args.seed, device)
    canonical = prepare_online_sparse24_weight(
        weight, weight24, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical, sparse_weight=weight24
    )
    persistent_bytes = runtime.persistent_bytes()
    del canonical
    gc.collect()
    torch.cuda.empty_cache()

    timer = ColdTimer(device, args.replays, args.eviction_mib)
    cases: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []

    for m in args.m_values:
        case = common.ShapeCase(model, projection, m, k, n)
        x = common.make_input(case, args.seed, device, purpose="splitk2_all")
        algorithm_id = select_cusparselt_algorithm(runtime.cusparselt, x)
        cublas = common.capture_graph(
            lambda: F.linear(x, weight), warmup=args.capture_warmup
        )
        pure_sparse = common.capture_graph(
            lambda: cusparselt_sparse_residual_sparse_linear(x, runtime),
            warmup=args.capture_warmup,
        )
        base_reference = F.linear(x, weight24)
        pure_correctness = correctness(pure_sparse.output, base_reference)
        routes = common.generate_routes([m], args.fractions, args.seed)

        for fraction in args.fractions:
            record = routes["routes"][common.route_key(m, fraction)]
            route = common.route_from_record(record, device)
            dense_x = x.index_select(0, route.dense_indices).contiguous()
            dense_reference = F.linear(dense_x, weight)
            expected = base_reference.clone()
            expected.index_copy_(0, route.dense_indices, dense_reference)

            captured_by_variant: dict[str, common.CapturedGraph] = {}
            resources_by_variant: dict[str, Any] = {}
            checks: dict[str, dict[str, Any]] = {}
            for variant in candidates_for(n):
                resources = common.create_multistream_resources(device)
                resources_by_variant[variant] = resources
                graph = common.capture_multistream_graph(
                    lambda resources=resources, variant=variant: launch_separate(
                        x,
                        route.dense_indices,
                        runtime,
                        resources,
                        complement_variant="feature128_token64_s4",
                        complement_first=True,
                        optimized_gather=True,
                        optimized_merge=True,
                        splitk2_complement=True,
                        splitk2_variant=variant,
                    ),
                    resources,
                    warmup=args.capture_warmup,
                    device=device,
                )
                captured_by_variant[variant] = graph
                checks[variant] = correctness(graph.output, expected)

            for graph in captured_by_variant.values():
                for _ in range(args.screen_warmup):
                    graph.graph.replay()
            torch.cuda.synchronize(device)
            screen_samples: dict[str, list[float]] = {
                variant: [] for variant in captured_by_variant
            }
            for trial in range(args.screen_trials):
                order = list(captured_by_variant)
                if trial % 2:
                    order.reverse()
                for position, variant in enumerate(order):
                    latency = timer.sample(
                        captured_by_variant[variant], args.screen_replays
                    )
                    screen_samples[variant].append(latency)
                    screens.append(
                        {
                            "model": model,
                            "projection": projection,
                            "M": m,
                            "dense_fraction": str(fraction),
                            "dense_rows": route.dense_count,
                            "variant": variant,
                            "trial": trial,
                            "position": position,
                            "latency_us": latency,
                        }
                    )
            selected = min(
                screen_samples,
                key=lambda variant: statistics.median(screen_samples[variant]),
            )
            split = captured_by_variant[selected]

            for graph in (cublas, pure_sparse, split):
                for _ in range(args.warmup):
                    graph.graph.replay()
            torch.cuda.synchronize(device)
            formal = {
                "cublas_dense": [],
                "cusparselt_pure_24": [],
                "splitk2_separate": [],
            }
            for trial in range(args.trials):
                methods = (
                    "cublas_dense",
                    "cusparselt_pure_24",
                    "splitk2_separate",
                )
                shift = trial % len(methods)
                order = methods[shift:] + methods[:shift]
                graphs = {
                    "cublas_dense": cublas,
                    "cusparselt_pure_24": pure_sparse,
                    "splitk2_separate": split,
                }
                for position, method in enumerate(order):
                    latency = timer.sample(graphs[method], args.replays)
                    formal[method].append(latency)
                    raw.append(
                        {
                            "model": model,
                            "projection": projection,
                            "M": m,
                            "N": n,
                            "K": k,
                            "dense_fraction": str(fraction),
                            "dense_rows": route.dense_count,
                            "selected_variant": selected,
                            "trial": trial,
                            "position": position,
                            "method": method,
                            "latency_us": latency,
                        }
                    )

            cublas_summary = summarize(formal["cublas_dense"])
            pure_summary = summarize(formal["cusparselt_pure_24"])
            split_summary = summarize(formal["splitk2_separate"])
            speedup = cublas_summary["median_us"] / split_summary["median_us"]
            ctas = (
                math.ceil(n / feature_tile(selected))
                * math.ceil(route.dense_count / 64)
                * 2
            )
            case_result = {
                "case": case.key,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "projection": projection,
                "M": m,
                "N": n,
                "K": k,
                "dense_fraction": str(fraction),
                "dense_rows": route.dense_count,
                "sparse_rows": route.sparse_count,
                "selected_variant": selected,
                "splitk2_ctas": ctas,
                "cusparselt_algorithm_id": algorithm_id,
                "persistent_bytes": persistent_bytes,
                "splitk2_workspace_bytes": 2 * 2 * route.dense_count * n,
                "cublas_median_us": cublas_summary["median_us"],
                "cublas_p10_us": cublas_summary["p10_us"],
                "cublas_p90_us": cublas_summary["p90_us"],
                "cusparselt_pure_24_median_us": pure_summary["median_us"],
                "cusparselt_pure_24_p10_us": pure_summary["p10_us"],
                "cusparselt_pure_24_p90_us": pure_summary["p90_us"],
                "splitk2_median_us": split_summary["median_us"],
                "splitk2_p10_us": split_summary["p10_us"],
                "splitk2_p90_us": split_summary["p90_us"],
                "hybrid_method": "separate_splitk2",
                "hybrid_median_us": split_summary["median_us"],
                "hybrid_p10_us": split_summary["p10_us"],
                "hybrid_p90_us": split_summary["p90_us"],
                "speedup_vs_cublas": speedup,
                "pure_24_speedup_vs_cublas": (
                    cublas_summary["median_us"] / pure_summary["median_us"]
                ),
                "hybrid_speedup_vs_pure_24": (
                    pure_summary["median_us"] / split_summary["median_us"]
                ),
                "faster_than_cublas": speedup > 1.0,
                "speedup_ge_1_1": speedup >= 1.1,
                "speedup_ge_1_3": speedup >= 1.3,
                "correctness": checks[selected],
                "pure_24_correctness": pure_correctness,
                "all_candidate_correct": all(
                    value["correct"] for value in checks.values()
                ),
                "screen_median_us": {
                    variant: statistics.median(values)
                    for variant, values in screen_samples.items()
                },
            }
            cases.append(case_result)
            print(
                f"{model}/{projection} M={m} D={fraction}: "
                f"hybrid={split_summary['median_us']:.3f} us, "
                f"dense={cublas_summary['median_us']:.3f} us, "
                f"pure24={pure_summary['median_us']:.3f} us, "
                f"hybrid/dense={speedup:.3f}x "
                f"[{selected}]",
                flush=True,
            )

            del expected, dense_reference, dense_x, route
            del captured_by_variant, resources_by_variant, checks, split
            gc.collect()
            torch.cuda.empty_cache()

        del cublas, pure_sparse, base_reference, x
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "model": model,
        "projection": projection,
        "N": n,
        "K": k,
        "cases": cases,
        "raw": raw,
        "screens": screens,
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group(keys: Sequence[str]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[name] for name in keys)
            buckets.setdefault(key, []).append(row)
        result = []
        for key, selected in sorted(buckets.items()):
            speedups = [float(row["speedup_vs_cublas"]) for row in selected]
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "cases": len(selected),
                    "geomean_speedup": geometric_mean(speedups),
                    "min_speedup": min(speedups),
                    "max_speedup": max(speedups),
                    "faster_cases": sum(value > 1.0 for value in speedups),
                    "ge_1_1_cases": sum(value >= 1.1 for value in speedups),
                    "ge_1_3_cases": sum(value >= 1.3 for value in speedups),
                }
            )
        return result

    speedups = [float(row["speedup_vs_cublas"]) for row in rows]
    pure_speedups = [float(row["pure_24_speedup_vs_cublas"]) for row in rows]
    hybrid_vs_pure = [float(row["hybrid_speedup_vs_pure_24"]) for row in rows]
    return {
        "overall": {
            "cases": len(rows),
            "geomean_speedup": geometric_mean(speedups),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "faster_cases": sum(value > 1.0 for value in speedups),
            "ge_1_1_cases": sum(value >= 1.1 for value in speedups),
            "ge_1_3_cases": sum(value >= 1.3 for value in speedups),
            "pure_24_geomean_speedup_vs_cublas": geometric_mean(pure_speedups),
            "hybrid_geomean_speedup_vs_pure_24": geometric_mean(hybrid_vs_pure),
        },
        "by_model": group(("model",)),
        "by_projection": group(("projection",)),
        "by_M": group(("M",)),
        "by_dense_fraction": group(("dense_fraction",)),
        "by_model_M": group(("model", "M")),
    }


def plot_speedups(output: Path, rows: list[dict[str, Any]]) -> None:
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    projections = [
        projection for projection in PROJECTIONS
        if any(row["projection"] == projection for row in rows)
    ]
    m_values = sorted({int(row["M"]) for row in rows})
    fractions = [str(value) for value in FRACTIONS if any(row["dense_fraction"] == str(value) for row in rows)]
    figure, axes = plt.subplots(
        len(models), len(m_values),
        figsize=(4.5 * len(m_values), 3.7 * len(models)),
        squeeze=False,
    )
    norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.8)
    image = None
    for row_index, model in enumerate(models):
        for column_index, m in enumerate(m_values):
            axis = axes[row_index][column_index]
            matrix = np.full((len(fractions), len(projections)), np.nan)
            for fi, fraction in enumerate(fractions):
                for pi, projection in enumerate(projections):
                    match = next(
                        (
                            row for row in rows
                            if row["model"] == model
                            and int(row["M"]) == m
                            and row["dense_fraction"] == fraction
                            and row["projection"] == projection
                        ),
                        None,
                    )
                    if match is not None:
                        matrix[fi, pi] = float(match["speedup_vs_cublas"])
            image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
            axis.set_xticks(range(len(projections)), projections, rotation=25)
            axis.set_yticks(range(len(fractions)), fractions)
            axis.set_title(f"{MODEL_LABELS[model]}, M={m}")
            axis.set_ylabel("Dense fraction")
            for fi in range(len(fractions)):
                for pi in range(len(projections)):
                    if np.isfinite(matrix[fi, pi]):
                        axis.text(
                            pi, fi, f"{matrix[fi, pi]:.2f}x",
                            ha="center", va="center", fontsize=8,
                            color="black",
                        )
    # Use a dedicated colorbar axis.  Passing the full axes array directly to
    # Figure.colorbar lets Matplotlib reclaim space from the last plot column;
    # with this 5x3 matrix that can place the colorbar on top of the panels.
    figure.subplots_adjust(
        left=0.06, right=0.89, top=0.88, bottom=0.10,
        hspace=0.58, wspace=0.22,
    )
    if image is not None:
        colorbar_axis = figure.add_axes((0.92, 0.15, 0.015, 0.70))
        figure.colorbar(
            image,
            cax=colorbar_axis,
            label="Speedup vs BF16 cuBLAS dense",
        )
    figure.suptitle(
        "Residual-complement hybrid speedup", fontsize=15, y=0.985
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_absolute_comparison(output: Path, rows: list[dict[str, Any]]) -> None:
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    projections = [
        projection for projection in PROJECTIONS
        if any(row["projection"] == projection for row in rows)
    ]
    figure, axes = plt.subplots(
        len(models), len(projections),
        figsize=(5.0 * len(projections), 3.6 * len(models)),
        squeeze=False,
    )
    colors = ("#4c78a8", "#59a14f", "#f28e2b")
    for ri, model in enumerate(models):
        for ci, projection in enumerate(projections):
            axis = axes[ri][ci]
            selected = sorted(
                (
                    row for row in rows
                    if row["model"] == model and row["projection"] == projection
                ),
                key=lambda row: (int(row["M"]), Fraction(row["dense_fraction"])),
            )
            labels = [
                f"M{row['M']}\nD={row['dense_fraction']}" for row in selected
            ]
            positions = np.arange(len(selected), dtype=np.float64)
            width = 0.25
            values = (
                [float(row["cublas_median_us"]) for row in selected],
                [float(row["cusparselt_pure_24_median_us"]) for row in selected],
                [
                    float(row.get("hybrid_median_us", row["splitk2_median_us"]))
                    for row in selected
                ],
            )
            for offset, data, label, color in zip(
                (-width, 0.0, width),
                values,
                ("Dense cuBLAS", "Pure 2:4 cuSPARSELt", "Hybrid"),
                colors,
                strict=True,
            ):
                axis.bar(positions + offset, data, width, label=label, color=color)
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.set_ylabel("Median latency (us)")
            axis.set_title(f"{MODEL_LABELS[model]} {projection}")
            axis.grid(axis="y", alpha=0.25)
            if ri == 0 and ci == len(projections) - 1:
                axis.legend(fontsize=8)
    figure.suptitle("Linear operator: absolute HBM-cold latency", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(
    output: Path,
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    overall = analysis["overall"]
    lines = [
        "# Residual-complement hybrid vs BF16 cuBLAS dense",
        "",
        f"Scope: {len(set(row['model'] for row in rows))} models, "
        f"projections={tuple(args.projections)}, M={tuple(args.m_values)}, dense fractions="
        f"{tuple(map(str, args.fractions))}; {len(rows)} cases.",
        "",
        f"Protocol: GPU idle; {args.warmup} warmups; {args.trials} x "
        f"{args.replays} CUDA Graph replays; an untimed {args.eviction_mib} MiB "
        "eviction before every measured replay; median with P10/P90 in us.",
        "",
        "The hybrid implementation is selected per measured case. Most rows use "
        "concurrent separate Split-K2; rows marked `optimized_fused` use sparse-row "
        "cuSPARSELt plus dense-row dual-HMMA.SP token partitioning.",
        "",
        "## Overall",
        "",
        f"- Geometric-mean speedup: **{overall['geomean_speedup']:.4f}x**.",
        f"- Pure 2:4 cuSPARSELt geometric-mean speedup vs cuBLAS: "
        f"**{overall['pure_24_geomean_speedup_vs_cublas']:.4f}x**.",
        f"- Hybrid geometric mean relative to pure 2:4: "
        f"**{overall['hybrid_geomean_speedup_vs_pure_24']:.4f}x**.",
        f"- Faster than cuBLAS: {overall['faster_cases']}/{overall['cases']} cases.",
        f"- At least 1.1x: {overall['ge_1_1_cases']}/{overall['cases']} cases.",
        f"- At least 1.3x: {overall['ge_1_3_cases']}/{overall['cases']} cases.",
        f"- Range: {overall['min_speedup']:.4f}x to {overall['max_speedup']:.4f}x.",
        "",
        "## Geometric mean by model",
        "",
        "| Model | Cases | Geomean | Faster | >=1.1x | >=1.3x |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["by_model"]:
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['cases']} | "
            f"{row['geomean_speedup']:.4f}x | {row['faster_cases']} | "
            f"{row['ge_1_1_cases']} | {row['ge_1_3_cases']} |"
        )
    lines.extend(["", "## All per-layer speedups", ""])
    for model in args.models:
        for m in args.m_values:
            lines.extend(
                [
                    f"### {MODEL_LABELS[model]}, M={m}",
                    "",
                    "| Dense fraction | " + " | ".join(args.projections) + " |",
                    "|---|" + "---:|" * len(args.projections),
                ]
            )
            for fraction in args.fractions:
                values = []
                for projection in args.projections:
                    match = next(
                        row for row in rows
                        if row["model"] == model
                        and int(row["M"]) == m
                        and row["dense_fraction"] == str(fraction)
                        and row["projection"] == projection
                    )
                    values.append(float(match["speedup_vs_cublas"]))
                lines.append(
                    f"| {fraction} | " + " | ".join(f"{value:.3f}x" for value in values) + " |"
                )
            lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `summary.csv`: all case medians, P10/P90, selected variant and speedup.",
            "- `raw.csv`: all 10 formal measurements per method and case.",
            "- `screen.csv`: HBM-cold Split-K variant-screen measurements.",
            "- `analysis.json`: aggregate geometric means and threshold counts.",
            "- `figures/absolute_latency_comparison.png`: measured dense, pure 2:4, and hybrid latency.",
            "- `figures/speedup_heatmaps.png`: complete model/M/fraction/layer matrix.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    worker_files: list[Path] = []
    for model in args.models:
        for projection in args.projections:
            common.require_idle_gpu(args.device_index)
            worker_output = work / f"{model}__{projection}.json"
            command = [
                sys.executable,
                str(SCRIPT),
                "--worker",
                "--worker-model", model,
                "--worker-projection", projection,
                "--worker-output", str(worker_output),
                "--device-index", str(args.device_index),
                "--m-values", ",".join(map(str, args.m_values)),
                "--fractions", ",".join(map(str, args.fractions)),
                "--seed", str(args.seed),
                "--capture-warmup", str(args.capture_warmup),
                "--screen-warmup", str(args.screen_warmup),
                "--screen-trials", str(args.screen_trials),
                "--screen-replays", str(args.screen_replays),
                "--warmup", str(args.warmup),
                "--trials", str(args.trials),
                "--replays", str(args.replays),
                "--eviction-mib", str(args.eviction_mib),
            ]
            print(f"[worker] {model}/{projection}", flush=True)
            subprocess.run(command, cwd=EVAL_ROOT, check=True)
            worker_files.append(worker_output)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    for path in worker_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["cases"])
        raw.extend(payload["raw"])
        screens.extend(payload["screens"])
    expected = len(args.models) * len(args.projections) * len(args.m_values) * len(args.fractions)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} cases, got {len(rows)}")
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["correctness"] = json.dumps(flat["correctness"], sort_keys=True)
        flat["pure_24_correctness"] = json.dumps(
            flat["pure_24_correctness"], sort_keys=True
        )
        flat["screen_median_us"] = json.dumps(flat["screen_median_us"], sort_keys=True)
        flat_rows.append(flat)
    write_csv(output / "summary.csv", flat_rows)
    write_csv(output / "raw.csv", raw)
    write_csv(output / "screen.csv", screens)
    analysis = aggregate(rows)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "models": list(args.models),
        "projections": list(args.projections),
        "M_values": list(args.m_values),
        "dense_fractions": list(map(str, args.fractions)),
        "cases": len(rows),
        "protocol": {
            "capture_warmup": args.capture_warmup,
            "screen_warmup": args.screen_warmup,
            "screen_trials": args.screen_trials,
            "screen_replays": args.screen_replays,
            "formal_warmup": args.warmup,
            "formal_trials": args.trials,
            "formal_replays": args.replays,
            "eviction_mib_per_replay": args.eviction_mib,
            "synchronize_inside_replay_loop": False,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plot_absolute_comparison(figures / "absolute_latency_comparison.png", rows)
    plot_speedups(figures / "speedup_heatmaps.png", rows)
    write_report(output, rows, analysis, args)
    print(output)
    print(json.dumps(analysis["overall"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-projection", choices=PROJECTIONS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--projections", default=",".join(PROJECTIONS))
    parser.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path, default=EVAL_ROOT / "temp/splitk2_all_models_layers")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--screen-warmup", type=int, default=20)
    parser.add_argument("--screen-trials", type=int, default=3)
    parser.add_argument("--screen-replays", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--eviction-mib", type=int, default=256)
    args = parser.parse_args()
    args.models = parse_csv(args.models, MODELS, "models")
    args.projections = parse_csv(args.projections, PROJECTIONS, "projections")
    args.m_values = parse_m_values(args.m_values)
    args.fractions = parse_fractions(args.fractions)
    if args.worker:
        if not args.worker_model or not args.worker_projection or not args.worker_output:
            parser.error("worker mode requires model, projection, and output")
    elif args.output_root is None:
        parser.error("coordinator mode requires --output-root")
    if min(
        args.capture_warmup,
        args.screen_warmup,
        args.screen_trials,
        args.screen_replays,
        args.warmup,
        args.trials,
        args.replays,
        args.eviction_mib,
    ) <= 0:
        parser.error("protocol counts must be positive")
    if args.trials != 10:
        parser.error("formal protocol requires exactly 10 trials")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        run_worker(arguments)
    else:
        run_coordinator(arguments)
