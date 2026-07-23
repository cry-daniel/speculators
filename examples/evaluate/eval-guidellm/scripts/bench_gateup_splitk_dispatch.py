#!/usr/bin/env python3
"""Formal gate_up dispatch benchmark under one HBM-cold protocol.

For every selected (model, M, dense fraction), this benchmark measures three
graphs in the same run: BF16 cuBLAS dense, the current non-Split-K separate
residual-complement kernel, and the best of three Split-K=2 candidates.  The
Split-K candidate is selected by a short HBM-cold screen.  Formal numbers use
100 warmups and 10 x 1000 CUDA Graph replays with an untimed 256 MiB eviction
before every measured replay.

The coordinator can also merge these freshly measured gate_up rows with the
previous formal qkv/o/down rows.  That paper-facing view contains only dense
fractions 1/8 and 1/4; historical 1/2 measurements remain untouched in their
original result directory but are not copied or plotted here.
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
    select_cusparselt_algorithm,
)


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parent.parent
MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
M_VALUES = (512, 1024, 2048)
FRACTIONS = (Fraction(1, 8), Fraction(1, 4))
PROJECTIONS = ("qkv", "o", "gate_up", "down")
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama3-70B",
}
NON_SPLIT_VARIANT = "feature128_token64_s4"
SPLIT_VARIANTS = (
    "b_resident_feature64_token64_b2a1",
    "b_resident_feature128_token64_b2a1",
    "feature64_token64_s4",
    "feature128_token64_s4",
    "b_resident_feature128_token128_b2a1",
    "b_resident_feature256_token128_b2a1",
    "b_resident_feature128_token128_b2a2",
    "b_resident_feature128_token128_b3a2",
)


def split_variants_for(n: int) -> tuple[str, ...]:
    exact = {
        24576: (
            "b_resident_feature128_token64_b2a1_p192",
            "b_resident_feature128_token128_b2a1_p192",
            "b_resident_feature256_token128_b2a1_p192",
            "b_resident_feature128_token128_b2a2_p192",
            "b_resident_feature128_token128_b3a2_p192",
        ),
        28672: (
            "b_resident_feature128_token64_b2a1_p224",
            "b_resident_feature128_token128_b2a1_p224",
            "b_resident_feature256_token128_b2a1_p224",
            "b_resident_feature128_token128_b2a2_p224",
            "b_resident_feature128_token128_b3a2_p224",
        ),
    }.get(n, ())
    return (*SPLIT_VARIANTS, *exact)


def non_split_variant_for(n: int) -> str:
    return {
        24576: "b_resident_feature128_token64_b2a1_p192",
        28672: "b_resident_feature128_token64_b2a1_p224",
    }.get(n, "b_resident_feature128_token64_b2a1")
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


def capture_separate(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    runtime: Any,
    device: torch.device,
    *,
    splitk2: bool,
    non_split_variant: str = NON_SPLIT_VARIANT,
    split_variant: str = SPLIT_VARIANTS[0],
    capture_warmup: int,
    resources: Any | None = None,
) -> tuple[common.CapturedGraph, Any]:
    if resources is None:
        resources = common.create_multistream_resources(device)
    graph = common.capture_multistream_graph(
        lambda: launch_separate(
            x,
            dense_indices,
            runtime,
            resources,
            complement_variant=non_split_variant,
            complement_first=True,
            optimized_gather=True,
            optimized_merge=True,
            splitk2_complement=splitk2,
            splitk2_variant=split_variant,
        ),
        resources,
        warmup=capture_warmup,
        device=device,
    )
    return graph, resources


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
    n, k = TP1_FUSED_WEIGHT_SHAPES[model]["gate_up"]
    weight_case = common.ShapeCase(model, "gate_up", max(args.m_values), k, n)
    weight, weight24 = common.make_synthetic_weight(weight_case, args.seed, device)
    canonical = prepare_online_sparse24_weight(
        weight, weight24, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical,
        sparse_weight=weight24,
        algorithm_id=max(args.cusparselt_algorithm_id, 0),
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
        case = common.ShapeCase(model, "gate_up", m, k, n)
        x = common.make_input(case, args.seed, device, purpose="gateup_dispatch")
        if args.cusparselt_algorithm_id < 0:
            algorithm_id = select_cusparselt_algorithm(runtime.cusparselt, x)
        else:
            algorithm_id = args.cusparselt_algorithm_id
        cublas = common.capture_graph(
            lambda: F.linear(x, weight), warmup=args.capture_warmup
        )
        base_reference = F.linear(x, weight24)
        routes = common.generate_routes([m], args.fractions, args.seed)

        for fraction in args.fractions:
            record = routes["routes"][common.route_key(m, fraction)]
            route = common.route_from_record(record, device)
            dense_x = x.index_select(0, route.dense_indices).contiguous()
            dense_reference = F.linear(dense_x, weight)
            expected = base_reference.clone()
            expected.index_copy_(0, route.dense_indices, dense_reference)

            non_split, non_split_resources = capture_separate(
                x,
                route.dense_indices,
                runtime,
                device,
                splitk2=False,
                non_split_variant=non_split_variant_for(n),
                capture_warmup=args.capture_warmup,
            )
            checks: dict[str, dict[str, Any]] = {
                "non_split": correctness(non_split.output, expected)
            }
            split_graphs: dict[str, common.CapturedGraph] = {}
            split_resources: dict[str, Any] = {}
            for variant in split_variants_for(n):
                graph, resources = capture_separate(
                    x,
                    route.dense_indices,
                    runtime,
                    device,
                    splitk2=True,
                    split_variant=variant,
                    capture_warmup=args.capture_warmup,
                )
                split_graphs[variant] = graph
                split_resources[variant] = resources
                checks[variant] = correctness(graph.output, expected)

            for graph in split_graphs.values():
                for _ in range(args.screen_warmup):
                    graph.graph.replay()
            torch.cuda.synchronize(device)
            screen_samples = {variant: [] for variant in split_graphs}
            for trial in range(args.screen_trials):
                order = list(split_graphs)
                if trial % 2:
                    order.reverse()
                for position, variant in enumerate(order):
                    latency = timer.sample(split_graphs[variant], args.screen_replays)
                    screen_samples[variant].append(latency)
                    screens.append(
                        {
                            "model": model,
                            "projection": "gate_up",
                            "M": m,
                            "N": n,
                            "K": k,
                            "dense_fraction": str(fraction),
                            "dense_rows": route.dense_count,
                            "variant": variant,
                            "trial": trial,
                            "position": position,
                            "latency_us": latency,
                        }
                    )
            selected_split_variant = min(
                screen_samples,
                key=lambda variant: statistics.median(screen_samples[variant]),
            )
            selected_split = split_graphs[selected_split_variant]

            graphs = {
                "cublas_dense": cublas,
                "non_split_separate": non_split,
                "splitk2_separate": selected_split,
            }
            for graph in graphs.values():
                for _ in range(args.warmup):
                    graph.graph.replay()
            torch.cuda.synchronize(device)
            formal = {method: [] for method in graphs}
            methods = tuple(graphs)
            for trial in range(args.trials):
                shift = trial % len(methods)
                order = methods[shift:] + methods[:shift]
                if (trial // len(methods)) % 2:
                    order = tuple(reversed(order))
                for position, method in enumerate(order):
                    latency = timer.sample(graphs[method], args.replays)
                    formal[method].append(latency)
                    raw.append(
                        {
                            "model": model,
                            "projection": "gate_up",
                            "M": m,
                            "N": n,
                            "K": k,
                            "dense_fraction": str(fraction),
                            "dense_rows": route.dense_count,
                            "selected_split_variant": selected_split_variant,
                            "trial": trial,
                            "position": position,
                            "method": method,
                            "latency_us": latency,
                        }
                    )

            summaries = {method: summarize(values) for method, values in formal.items()}
            cublas_median = summaries["cublas_dense"]["median_us"]
            non_split_median = summaries["non_split_separate"]["median_us"]
            split_median = summaries["splitk2_separate"]["median_us"]
            selected_method = (
                "non_split_separate"
                if non_split_median <= split_median
                else "splitk2_separate"
            )
            selected_summary = summaries[selected_method]
            cases.append(
                {
                    "case": case.key,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "projection": "gate_up",
                    "M": m,
                    "N": n,
                    "K": k,
                    "dense_fraction": str(fraction),
                    "dense_rows": route.dense_count,
                    "sparse_rows": route.sparse_count,
                    "non_split_variant": non_split_variant_for(n),
                    "selected_split_variant": selected_split_variant,
                    "selected_method": selected_method,
                    "cusparselt_algorithm_id": algorithm_id,
                    "persistent_bytes": persistent_bytes,
                    "splitk2_workspace_bytes": 2 * 2 * route.dense_count * n,
                    "cublas_median_us": cublas_median,
                    "cublas_p10_us": summaries["cublas_dense"]["p10_us"],
                    "cublas_p90_us": summaries["cublas_dense"]["p90_us"],
                    "non_split_median_us": non_split_median,
                    "non_split_p10_us": summaries["non_split_separate"]["p10_us"],
                    "non_split_p90_us": summaries["non_split_separate"]["p90_us"],
                    "non_split_speedup_vs_cublas": cublas_median / non_split_median,
                    "splitk2_median_us": split_median,
                    "splitk2_p10_us": summaries["splitk2_separate"]["p10_us"],
                    "splitk2_p90_us": summaries["splitk2_separate"]["p90_us"],
                    "splitk2_speedup_vs_cublas": cublas_median / split_median,
                    "selected_median_us": selected_summary["median_us"],
                    "selected_p10_us": selected_summary["p10_us"],
                    "selected_p90_us": selected_summary["p90_us"],
                    "speedup_vs_cublas": cublas_median / selected_summary["median_us"],
                    "non_split_speedup_vs_splitk2": split_median / non_split_median,
                    "checks": checks,
                    "all_candidate_correct": all(value["correct"] for value in checks.values()),
                    "screen_median_us": {
                        variant: statistics.median(values)
                        for variant, values in screen_samples.items()
                    },
                }
            )
            print(
                f"{model}/gate_up M={m} D={fraction}: "
                f"non-split={non_split_median:.3f} us, "
                f"split={split_median:.3f} us [{selected_split_variant}], "
                f"cuBLAS={cublas_median:.3f} us -> {selected_method}",
                flush=True,
            )

            del expected, dense_reference, dense_x, route
            del non_split, non_split_resources, selected_split
            del split_graphs, split_resources, checks, graphs
            gc.collect()
            torch.cuda.empty_cache()

        del cublas, base_reference, x
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "model": model,
        "projection": "gate_up",
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
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(selected: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row["speedup_vs_cublas"]) for row in selected]
        return {
            "cases": len(selected),
            "geomean_speedup": geometric_mean(values),
            "min_speedup": min(values),
            "max_speedup": max(values),
            "faster_cases": sum(value > 1.0 for value in values),
            "ge_1_1_cases": sum(value >= 1.1 for value in values),
            "ge_1_3_cases": sum(value >= 1.3 for value in values),
        }

    result: dict[str, Any] = {"overall": summarize_group(rows)}
    for label, key in (
        ("by_model", "model"),
        ("by_projection", "projection"),
        ("by_M", "M"),
        ("by_dense_fraction", "dense_fraction"),
    ):
        groups = []
        for value in sorted({row[key] for row in rows}, key=str):
            selected = [row for row in rows if row[key] == value]
            groups.append({key: value, **summarize_group(selected)})
        result[label] = groups
    return result


def plot_gateup(output: Path, rows: list[dict[str, Any]]) -> None:
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    m_values = sorted({int(row["M"]) for row in rows})
    figure, axes = plt.subplots(
        len(models), len(m_values),
        figsize=(4.4 * len(m_values), 2.7 * len(models)),
        squeeze=False,
        sharey=True,
    )
    fractions = tuple(map(str, FRACTIONS))
    x_positions = np.arange(len(fractions))
    width = 0.34
    for row_index, model in enumerate(models):
        for column_index, m in enumerate(m_values):
            axis = axes[row_index][column_index]
            selected = {
                row["dense_fraction"]: row
                for row in rows
                if row["model"] == model and int(row["M"]) == m
            }
            non_split = [
                float(selected[fraction]["non_split_speedup_vs_cublas"])
                for fraction in fractions
            ]
            split = [
                float(selected[fraction]["splitk2_speedup_vs_cublas"])
                for fraction in fractions
            ]
            axis.bar(x_positions - width / 2, non_split, width, label="non-Split-K")
            axis.bar(x_positions + width / 2, split, width, label="Split-K=2")
            axis.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
            axis.set_xticks(x_positions, fractions)
            axis.set_title(f"{MODEL_LABELS[model]}, M={m}")
            axis.set_ylim(0.85, 1.65)
            axis.grid(axis="y", alpha=0.25)
            if column_index == 0:
                axis.set_ylabel("Speedup vs cuBLAS")
            for position, values in ((-width / 2, non_split), (width / 2, split)):
                for index, value in enumerate(values):
                    axis.text(
                        x_positions[index] + position,
                        value + 0.012,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("gate_up: HBM-cold non-Split-K vs Split-K=2", y=0.997)
    figure.tight_layout(rect=(0, 0, 1, 0.942))
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_paper_heatmaps(output: Path, rows: list[dict[str, Any]]) -> None:
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    m_values = sorted({int(row["M"]) for row in rows})
    fractions = tuple(map(str, FRACTIONS))
    figure, axes = plt.subplots(
        len(models), len(m_values),
        figsize=(4.5 * len(m_values), 3.0 * len(models)),
        squeeze=False,
    )
    norm = TwoSlopeNorm(vmin=0.7, vcenter=1.0, vmax=1.8)
    image = None
    for row_index, model in enumerate(models):
        for column_index, m in enumerate(m_values):
            axis = axes[row_index][column_index]
            matrix = np.full((len(fractions), len(PROJECTIONS)), np.nan)
            for fi, fraction in enumerate(fractions):
                for pi, projection in enumerate(PROJECTIONS):
                    match = next(
                        row for row in rows
                        if row["model"] == model
                        and int(row["M"]) == m
                        and row["dense_fraction"] == fraction
                        and row["projection"] == projection
                    )
                    matrix[fi, pi] = float(match["speedup_vs_cublas"])
            image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
            axis.set_xticks(range(len(PROJECTIONS)), PROJECTIONS, rotation=25)
            axis.set_yticks(range(len(fractions)), fractions)
            axis.set_title(f"{MODEL_LABELS[model]}, M={m}")
            axis.set_ylabel("Dense fraction")
            for fi in range(len(fractions)):
                for pi in range(len(PROJECTIONS)):
                    axis.text(
                        pi, fi, f"{matrix[fi, pi]:.2f}x",
                        ha="center", va="center", fontsize=8, color="black",
                    )
    figure.subplots_adjust(
        left=0.06, right=0.89, top=0.95, bottom=0.05,
        hspace=0.48, wspace=0.22,
    )
    if image is not None:
        colorbar_axis = figure.add_axes((0.92, 0.15, 0.015, 0.70))
        figure.colorbar(
            image, cax=colorbar_axis, label="Speedup vs BF16 cuBLAS dense"
        )
    figure.suptitle(
        "Separate residual-complement speedup "
        "(fixed Split-K=2; dense fractions 1/8 and 1/4)",
        fontsize=15,
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def normalize_old_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(row)
    normalized["M"] = int(row["M"])
    normalized["N"] = int(row["N"])
    normalized["K"] = int(row["K"])
    normalized["dense_rows"] = int(row["dense_rows"])
    normalized["sparse_rows"] = int(row["sparse_rows"])
    normalized["cublas_median_us"] = float(row["cublas_median_us"])
    normalized["selected_median_us"] = float(row["splitk2_median_us"])
    normalized["selected_p10_us"] = float(row["splitk2_p10_us"])
    normalized["selected_p90_us"] = float(row["splitk2_p90_us"])
    normalized["speedup_vs_cublas"] = float(row["speedup_vs_cublas"])
    normalized["selected_method"] = "splitk2_separate"
    normalized["result_source"] = "prior_formal_splitk2_same_protocol"
    return normalized


def build_paper_rows(
    gate_rows: list[dict[str, Any]], prior_root: Path
) -> list[dict[str, Any]]:
    summary_path = prior_root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing prior formal summary: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    paper_rows = [
        normalize_old_row(row)
        for row in old_rows
        if row["projection"] != "gate_up"
        and row["dense_fraction"] in {"1/8", "1/4"}
    ]
    for row in gate_rows:
        copied = dict(row)
        copied["selected_method"] = "splitk2_separate"
        copied["selected_median_us"] = float(row["splitk2_median_us"])
        copied["selected_p10_us"] = float(row["splitk2_p10_us"])
        copied["selected_p90_us"] = float(row["splitk2_p90_us"])
        copied["speedup_vs_cublas"] = float(
            row["splitk2_speedup_vs_cublas"]
        )
        copied["result_source"] = "fresh_gateup_fixed_splitk2_same_protocol"
        paper_rows.append(copied)
    expected = len(MODELS) * len(PROJECTIONS) * len(M_VALUES) * len(FRACTIONS)
    keys = {
        (row["model"], row["projection"], int(row["M"]), row["dense_fraction"])
        for row in paper_rows
    }
    if len(paper_rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"paper matrix must contain {expected} unique cases, got "
            f"{len(paper_rows)} rows and {len(keys)} keys"
        )
    return sorted(
        paper_rows,
        key=lambda row: (
            MODELS.index(row["model"]),
            M_VALUES.index(int(row["M"])),
            FRACTIONS.index(Fraction(row["dense_fraction"])),
            PROJECTIONS.index(row["projection"]),
        ),
    )


def write_report(
    output: Path,
    gate_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    gate_analysis: dict[str, Any],
    paper_analysis: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    non_split_values = [
        float(row["non_split_speedup_vs_cublas"]) for row in gate_rows
    ]
    split_values = [
        float(row["splitk2_speedup_vs_cublas"]) for row in gate_rows
    ]
    lines = [
        "# Paper-facing separate residual-complement results",
        "",
        "Only dense fractions **1/8 and 1/4** are included. Historical 1/2 "
        "rows were not reused and are not plotted.",
        "",
        "## Protocol",
        "",
        f"GPU idle; {args.warmup} warmups; {args.trials} x {args.replays} CUDA "
        f"Graph replays; untimed {args.eviction_mib} MiB eviction before every "
        "measured replay; one synchronization after each replay batch; median "
        "with P10/P90 in microseconds.",
        "",
        "All 30 gate_up cases were freshly measured in one run per case with "
        "cuBLAS dense, current non-Split-K, and the screened Split-K=2 graph.",
        "",
        "## Fresh gate_up fixed Split-K=2 result",
        "",
        f"- non-Split-K geometric-mean speedup vs cuBLAS: "
        f"**{geometric_mean(non_split_values):.4f}x**.",
        f"- Split-K=2 geometric-mean speedup vs cuBLAS: "
        f"**{geometric_mean(split_values):.4f}x**.",
        "",
        "The final paper matrix uses Split-K=2 for every gate_up case. "
        "non-Split-K remains only as a diagnostic comparison and does not "
        "affect any paper-facing speedup or aggregate.",
        "",
        "| Model | M | Dense fraction | cuBLAS us | non-Split-K us | "
        "Split-K=2 us | Split-K2 speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['M']} | "
            f"{row['dense_fraction']} | {row['cublas_median_us']:.3f} | "
            f"{row['non_split_median_us']:.3f} | {row['splitk2_median_us']:.3f} | "
            f"{row['splitk2_speedup_vs_cublas']:.3f}x |"
        )
    overall = paper_analysis["overall"]
    lines.extend(
        [
            "",
            "## Final 120-case paper matrix",
            "",
            f"- Geometric-mean speedup: **{overall['geomean_speedup']:.4f}x**.",
            f"- Faster than cuBLAS: {overall['faster_cases']}/{overall['cases']}.",
            f"- At least 1.1x: {overall['ge_1_1_cases']}/{overall['cases']}.",
            f"- At least 1.3x: {overall['ge_1_3_cases']}/{overall['cases']}.",
            "",
            "## Artifacts",
            "",
            "- `gate_up_summary.csv`: the 30 fresh gate_up case medians and diagnostic comparison.",
            "- `gate_up_raw.csv`: all 10 formal samples for all three methods.",
            "- `gate_up_screen.csv`: Split-K=2 candidate screen measurements.",
            "- `paper_summary_1over8_1over4.csv`: the final 120-case display matrix.",
            "- `figures/gate_up_non_split_vs_splitk2.png`: same-run gate_up comparison.",
            "- `figures/speedup_heatmaps_1over8_1over4.png`: final paper heatmap.",
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
        common.require_idle_gpu(args.device_index)
        worker_output = work / f"{model}__gate_up.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--worker-model", model,
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
        print(f"[worker] {model}/gate_up", flush=True)
        subprocess.run(command, cwd=EVAL_ROOT, check=True)
        worker_files.append(worker_output)

    gate_rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    for path in worker_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        gate_rows.extend(payload["cases"])
        raw.extend(payload["raw"])
        screens.extend(payload["screens"])
    expected = len(args.models) * len(args.m_values) * len(args.fractions)
    if len(gate_rows) != expected:
        raise RuntimeError(f"expected {expected} gate_up cases, got {len(gate_rows)}")
    if not all(row["all_candidate_correct"] for row in gate_rows):
        raise RuntimeError("at least one gate_up candidate failed correctness")

    flat_gate_rows = []
    for row in gate_rows:
        flat = dict(row)
        flat["checks"] = json.dumps(flat["checks"], sort_keys=True)
        flat["screen_median_us"] = json.dumps(flat["screen_median_us"], sort_keys=True)
        flat_gate_rows.append(flat)
    write_csv(output / "gate_up_summary.csv", flat_gate_rows)
    write_csv(output / "gate_up_raw.csv", raw)
    write_csv(output / "gate_up_screen.csv", screens)

    paper_rows = build_paper_rows(gate_rows, args.prior_formal_root.resolve())
    fixed_split_gate_rows = [
        row for row in paper_rows if row["projection"] == "gate_up"
    ]
    gate_analysis = aggregate(fixed_split_gate_rows)
    paper_analysis = aggregate(paper_rows)
    write_csv(output / "paper_summary_1over8_1over4.csv", paper_rows)
    (output / "gate_up_analysis.json").write_text(
        json.dumps(gate_analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "paper_analysis_1over8_1over4.json").write_text(
        json.dumps(paper_analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "models": list(args.models),
        "projection": "gate_up",
        "M_values": list(args.m_values),
        "dense_fractions": list(map(str, args.fractions)),
        "gate_up_cases": len(gate_rows),
        "paper_display_cases": len(paper_rows),
        "paper_display_excludes_dense_fraction": ["1/2"],
        "gate_up_paper_value_semantics": (
            "fixed Split-K=2 for every gate_up case; non-Split-K is retained "
            "only as a diagnostic comparison"
        ),
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
            "formal_methods_measured_same_run": [
                "cublas_dense", "non_split_separate", "splitk2_separate"
            ],
        },
        "prior_formal_root_for_qkv_o_down": str(args.prior_formal_root.resolve()),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plot_gateup(figures / "gate_up_non_split_vs_splitk2.png", gate_rows)
    plot_paper_heatmaps(
        figures / "speedup_heatmaps_1over8_1over4.png", paper_rows
    )
    write_report(output, gate_rows, paper_rows, gate_analysis, paper_analysis, args)
    print(output)
    print(json.dumps({
        "gate_up": gate_analysis["overall"],
        "paper_120_cases": paper_analysis["overall"],
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    parser.add_argument("--fractions", default=",".join(map(str, FRACTIONS)))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=EVAL_ROOT / "temp/gateup_splitk_dispatch",
    )
    parser.add_argument("--prior-formal-root", type=Path)
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
    parser.add_argument(
        "--cusparselt-algorithm-id",
        type=int,
        default=-1,
        help="fixed base algorithm; -1 uses cuSPARSELt search",
    )
    args = parser.parse_args()
    args.models = parse_csv(args.models, MODELS, "models")
    args.m_values = parse_m_values(args.m_values)
    args.fractions = parse_fractions(args.fractions)
    if args.worker:
        if not args.worker_model or not args.worker_output:
            parser.error("worker mode requires model and output")
    elif args.output_root is None or args.prior_formal_root is None:
        parser.error("coordinator mode requires output-root and prior-formal-root")
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
    if args.cusparselt_algorithm_id < -1:
        parser.error("cuSPARSELt algorithm id must be -1 or non-negative")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        run_worker(arguments)
    else:
        run_coordinator(arguments)
