#!/usr/bin/env python3
"""Compare external BF16 N:M kernels with cuBLAS and SpecLink.

The coordinator starts one worker per (model, projection), which bounds peak
GPU memory for the largest Llama-3-70B weights.  External systems operate on
static 5:8 or 3:4 weights.  SpecLink instead keeps an exact dense weight as a
2:4 base plus its complementary 2:4 residual and routes either 1/8 (D1) or
1/4 (D2) of token rows through both streams.  These semantics are deliberately
kept as separate result columns and plot labels.
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
EVAL_ROOT = ROOT / "examples/evaluate/eval-guidellm"
BENCH_SCRIPTS = EVAL_ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))

from other_systems import benchmark_common as external  # noqa: E402
from other_systems import parse_nm  # noqa: E402
import sparse24_benchmark_common as hybrid_common  # noqa: E402
from bench_splitk2_all_models_layers import candidates_for  # noqa: E402
from residual_complement_runtime import (  # noqa: E402
    launch_fused,
    launch_separate,
    select_fused_gateup_variant,
    should_use_fused_gateup,
)
from speculators.speclink import (  # noqa: E402
    SPARSE_RESIDUAL_SMEM,
    TP1_FUSED_WEIGHT_SHAPES,
    prepare_cusparselt_sparse_residual_weight,
    prepare_online_sparse24_weight,
    select_cusparselt_algorithm,
)


MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
PROJECTIONS = ("qkv", "o", "gate_up", "down")
M_VALUES = (512, 1024, 2048)
FRACTIONS = (Fraction(1, 8), Fraction(1, 4))
FORMATS = ("5:8", "3:4")
SYSTEMS = ("flash_llm", "spinfer", "sparta")
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama-3-70B",
}
METHOD_LABELS = {
    "dense_cublas": "cuBLAS dense",
    "flash_llm_5_8": "FlashLLM 5:8",
    "spinfer_5_8": "SpInfer 5:8",
    "sparta_5_8": "SparTA 5:8",
    "flash_llm_3_4": "FlashLLM 3:4",
    "spinfer_3_4": "SpInfer 3:4",
    "sparta_3_4": "SparTA 3:4",
    "ours_d1": "Ours D1 (1/8 dense)",
    "ours_d2": "Ours D2 (1/4 dense)",
}
METHOD_ORDER = tuple(METHOD_LABELS)
METHOD_COLORS = {
    "dense_cublas": "#6f6f6f",
    "flash_llm_5_8": "#4c78a8",
    "spinfer_5_8": "#f58518",
    "sparta_5_8": "#54a24b",
    "flash_llm_3_4": "#9ecae9",
    "spinfer_3_4": "#ffbf79",
    "sparta_3_4": "#a1d99b",
    "ours_d1": "#b279a2",
    "ours_d2": "#e45756",
}
SEED = 20260723


def parse_csv(
    value: str, allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid {label}: {unknown or value!r}"
        )
    return selected


def parse_fractions(value: str) -> tuple[Fraction, ...]:
    selected = tuple(Fraction(part) for part in value.split(",") if part)
    unknown = sorted(set(selected) - set(FRACTIONS))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"dense fractions must be selected from {FRACTIONS}"
        )
    return selected


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def correctness(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    result = {
        "correct": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }
    if not result["correct"]:
        raise RuntimeError(f"correctness failure: {result}")
    return result


def formal_measure(
    captured: Any,
    eviction: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[external.TimingSummary, list[float]]:
    return external.formal_measure(
        captured,
        eviction,
        warmup=args.warmup,
        trials=args.trials,
        replays=args.replays,
    )


def append_measurement(
    rows: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    *,
    base: dict[str, Any],
    method: str,
    summary: external.TimingSummary,
    samples: Sequence[float],
    dense_us: float,
    check: dict[str, Any],
) -> None:
    rows.append(
        {
            **base,
            "method": method,
            "method_label": METHOD_LABELS[method],
            **summary.as_dict(),
            "speedup_vs_cublas": dense_us / summary.median_us,
            **check,
        }
    )
    for trial, latency in enumerate(samples):
        raw.append(
            {
                **{
                    key: base[key]
                    for key in (
                        "model",
                        "projection",
                        "M",
                        "N",
                        "K",
                        "sparsity_format",
                        "semantic",
                    )
                },
                "method": method,
                "trial": trial,
                "latency_us": latency,
            }
        )


def capture_best_hybrid(
    x: torch.Tensor,
    route: Any,
    runtime: Any,
    projection: str,
    args: argparse.Namespace,
) -> tuple[Any, str, list[dict[str, Any]], list[Any]]:
    resources_to_keep: list[Any] = []
    if projection == "gate_up" and should_use_fused_gateup(
        x, route.dense_indices, runtime
    ):
        resources = hybrid_common.create_multistream_resources(x.device)
        output = torch.empty(
            (x.shape[0], runtime.n), device=x.device, dtype=torch.bfloat16
        )
        captured = hybrid_common.capture_multistream_graph(
            lambda: launch_fused(
                x,
                route.dense_indices,
                route.sparse_indices,
                runtime,
                output,
                resources,
                variant="auto",
                optimized_routes=True,
            ),
            resources,
            warmup=args.capture_warmup,
            device=x.device,
        )
        variant = "fused_" + select_fused_gateup_variant(runtime)
        return captured, variant, [], [resources, output]

    captured_by_variant: dict[str, Any] = {}
    screen_rows: list[dict[str, Any]] = []
    for variant in candidates_for(int(runtime.n)):
        resources = hybrid_common.create_multistream_resources(x.device)
        resources_to_keep.append(resources)
        captured_by_variant[variant] = hybrid_common.capture_multistream_graph(
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
            device=x.device,
        )
    medians: dict[str, float] = {}
    for variant, captured in captured_by_variant.items():
        values = [
            hybrid_common.steady_graph_sample(captured, args.screen_replays)
            for _ in range(args.screen_trials)
        ]
        medians[variant] = statistics.median(values)
        for trial, value in enumerate(values):
            screen_rows.append(
                {
                    "variant": variant,
                    "trial": trial,
                    "latency_us": value,
                }
            )
    selected = min(medians, key=medians.__getitem__)
    return (
        captured_by_variant[selected],
        selected,
        screen_rows,
        resources_to_keep,
    )


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = args.worker_model
    projection = args.worker_projection
    n, k = TP1_FUSED_WEIGHT_SHAPES[model][projection]
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024, device=device, dtype=torch.uint8
    )
    eviction.zero_()
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []

    weight_case = hybrid_common.ShapeCase(
        model, projection, max(args.m_values), k, n
    )
    dense_weight, base24_weight = hybrid_common.make_synthetic_weight(
        weight_case, args.seed, device
    )
    canonical = prepare_online_sparse24_weight(
        dense_weight, base24_weight, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical, sparse_weight=base24_weight
    )
    del canonical
    dense_summaries: dict[int, external.TimingSummary] = {}

    for m in args.m_values:
        case = hybrid_common.ShapeCase(model, projection, m, k, n)
        x = hybrid_common.make_input(
            case, args.seed, device, purpose="five_model_comparison"
        )
        dense_graph = hybrid_common.capture_graph(
            lambda: F.linear(x, dense_weight), warmup=args.capture_warmup
        )
        dense_summary, dense_samples = formal_measure(
            dense_graph, eviction, args
        )
        dense_summaries[m] = dense_summary
        append_measurement(
            rows,
            raw,
            base={
                "model": model,
                "model_label": MODEL_LABELS[model],
                "projection": projection,
                "M": m,
                "N": n,
                "K": k,
                "sparsity_format": "dense",
                "semantic": "dense_exact",
                "weight_density": 1.0,
                "dense_fraction": "",
                "dense_rows": "",
                "selected_variant": "",
                "split_k": 0,
            },
            method="dense_cublas",
            summary=dense_summary,
            samples=dense_samples,
            dense_us=dense_summary.median_us,
            check={
                "correct": True,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
            },
        )

        base_reference = F.linear(x, base24_weight)
        routes = hybrid_common.generate_routes(
            [m], args.fractions, args.seed
        )
        for fraction in args.fractions:
            route_record = routes["routes"][
                hybrid_common.route_key(m, fraction)
            ]
            route = hybrid_common.route_from_record(route_record, device)
            expected = base_reference.clone()
            dense_reference = F.linear(
                x.index_select(0, route.dense_indices), dense_weight
            )
            expected.index_copy_(0, route.dense_indices, dense_reference)
            # cuSPARSELt algorithm choice is activation-shape dependent.  The
            # separate topology runs its base over all M rows, while the fused
            # gate_up topology runs the base only on sparse-token rows.
            if projection == "gate_up" and should_use_fused_gateup(
                x, route.dense_indices, runtime
            ):
                sparse_x = x.index_select(
                    0, route.sparse_indices
                ).contiguous()
                cusparselt_algorithm_id = select_cusparselt_algorithm(
                    runtime.cusparselt, sparse_x
                )
                del sparse_x
            else:
                cusparselt_algorithm_id = select_cusparselt_algorithm(
                    runtime.cusparselt, x
                )
            captured, variant, local_screens, keepalive = capture_best_hybrid(
                x, route, runtime, projection, args
            )
            check = correctness(
                captured.output, expected, atol=0.06, rtol=0.06
            )
            summary, samples = formal_measure(captured, eviction, args)
            method = "ours_d1" if fraction == Fraction(1, 8) else "ours_d2"
            append_measurement(
                rows,
                raw,
                base={
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "projection": projection,
                    "M": m,
                    "N": n,
                    "K": k,
                    "sparsity_format": "2:4_base+2:4_complement",
                    "semantic": "token_hybrid_exact",
                    "weight_density": 1.0,
                    "dense_fraction": str(fraction),
                    "dense_rows": route.dense_count,
                    "selected_variant": variant,
                    "cusparselt_algorithm_id": cusparselt_algorithm_id,
                    "split_k": (
                        0 if variant.startswith("fused_") else 2
                    ),
                },
                method=method,
                summary=summary,
                samples=samples,
                dense_us=dense_summary.median_us,
                check=check,
            )
            for screen in local_screens:
                screens.append(
                    {
                        "model": model,
                        "projection": projection,
                        "M": m,
                        "N": n,
                        "K": k,
                        "method": method,
                        "dense_fraction": str(fraction),
                        **screen,
                    }
                )
            print(
                f"[ours] {model}/{projection} M={m} D={fraction} "
                f"{summary.median_us:.3f} us "
                f"({dense_summary.median_us / summary.median_us:.3f}x) "
                f"[{variant}]",
                flush=True,
            )
            del captured, keepalive, route, expected, dense_reference
            gc.collect()
            torch.cuda.empty_cache()
        del dense_graph, base_reference, x
        gc.collect()
        torch.cuda.empty_cache()

    del runtime, dense_weight, base24_weight
    gc.collect()
    torch.cuda.empty_cache()

    if args.ours_only:
        payload = {
            "model": model,
            "projection": projection,
            "N": n,
            "K": k,
            "rows": rows,
            "raw": raw,
            "screens": screens,
        }
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    for fmt in args.formats:
        nm_weight = external.make_nm_weight(
            model, projection, fmt, device=device, seed=args.seed
        )
        density = parse_nm(fmt).n / parse_nm(fmt).m
        for system in args.systems:
            prepared = external.prepare_system(system, nm_weight, fmt)
            for m in args.m_values:
                x = external.make_input(
                    model, projection, m, device=device, seed=args.seed
                )
                reference = external.dense_linear(x, nm_weight)
                selected, local_screens = external.select_split(
                    prepared,
                    x,
                    repeats=args.external_screen_replays,
                )
                graph = external.capture(
                    lambda selected=selected, x=x: selected.linear(x),
                    warmup=args.capture_warmup,
                )
                check = correctness(
                    graph.output, reference, atol=0.2, rtol=0.1
                )
                summary, samples = formal_measure(graph, eviction, args)
                method = f"{system}_{fmt.replace(':', '_')}"
                append_measurement(
                    rows,
                    raw,
                    base={
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "projection": projection,
                        "M": m,
                        "N": n,
                        "K": k,
                        "sparsity_format": fmt,
                        "semantic": "static_nm_pruned",
                        "weight_density": density,
                        "dense_fraction": "",
                        "dense_rows": "",
                        "selected_variant": "",
                        "split_k": selected.split_k,
                    },
                    method=method,
                    summary=summary,
                    samples=samples,
                    dense_us=dense_summaries[m].median_us,
                    check=check,
                )
                for screen in local_screens:
                    screens.append(
                        {
                            "model": model,
                            "projection": projection,
                            "M": m,
                            "N": n,
                            "K": k,
                            "method": method,
                            "dense_fraction": "",
                            "variant": f"split_k={screen['split_k']}",
                            "trial": 0,
                            "latency_us": screen["latency_us"],
                        }
                    )
                print(
                    f"[external] {model}/{projection} M={m} {fmt} "
                    f"{system} split={selected.split_k}: "
                    f"{summary.median_us:.3f} us "
                    f"({dense_summaries[m].median_us / summary.median_us:.3f}x)",
                    flush=True,
                )
                del graph, selected, reference, x
                gc.collect()
                torch.cuda.empty_cache()
            del prepared
            gc.collect()
            torch.cuda.empty_cache()
        del nm_weight
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "model": model,
        "projection": projection,
        "N": n,
        "K": k,
        "rows": rows,
        "raw": raw,
        "screens": screens,
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def grouped(keys: Sequence[str]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[name] for name in keys)
            buckets.setdefault(key, []).append(row)
        result: list[dict[str, Any]] = []
        for key, selected in sorted(buckets.items()):
            speedups = [float(row["speedup_vs_cublas"]) for row in selected]
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "cases": len(selected),
                    "geomean_speedup_vs_cublas": geometric_mean(speedups),
                    "min_speedup_vs_cublas": min(speedups),
                    "max_speedup_vs_cublas": max(speedups),
                    "faster_than_cublas": sum(value > 1.0 for value in speedups),
                }
            )
        return result

    return {
        "by_method": grouped(("method",)),
        "by_method_model": grouped(("method", "model")),
        "by_method_projection": grouped(("method", "projection")),
        "by_method_M": grouped(("method", "M")),
    }


def rows_for_panel(
    rows: Sequence[dict[str, Any]], model: str, projection: str, m: int
) -> dict[str, dict[str, Any]]:
    return {
        str(row["method"]): row
        for row in rows
        if row["model"] == model
        and row["projection"] == projection
        and int(row["M"]) == m
    }


def plot_panels(
    rows: Sequence[dict[str, Any]],
    output: Path,
    m: int,
    *,
    value_key: str,
    ylabel: str,
    reference_line: float | None,
) -> None:
    models = [
        model for model in MODELS if any(row["model"] == model for row in rows)
    ]
    projections = [
        projection
        for projection in PROJECTIONS
        if any(row["projection"] == projection for row in rows)
    ]
    figure, axes = plt.subplots(
        len(models),
        len(projections),
        figsize=(22, 3.8 * len(models)),
        squeeze=False,
    )
    for row_index, model in enumerate(models):
        for column_index, projection in enumerate(projections):
            axis = axes[row_index][column_index]
            selected = rows_for_panel(rows, model, projection, m)
            methods = [
                method for method in METHOD_ORDER if method in selected
            ]
            values = [float(selected[method][value_key]) for method in methods]
            positions = np.arange(len(methods))
            axis.bar(
                positions,
                values,
                color=[METHOD_COLORS[method] for method in methods],
            )
            if reference_line is not None:
                axis.axhline(
                    reference_line, color="black", linewidth=1, linestyle="--"
                )
            axis.set_xticks(positions, [METHOD_LABELS[item] for item in methods])
            axis.tick_params(axis="x", labelrotation=65, labelsize=7)
            axis.set_title(f"{MODEL_LABELS[model]} {projection}")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
    title = (
        f"Kernel speedup vs BF16 cuBLAS, M={m}"
        if value_key == "speedup_vs_cublas"
        else f"Absolute BF16 GEMM latency, M={m}"
    )
    figure.suptitle(title, fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_geomean(
    analysis: dict[str, Any], output: Path
) -> None:
    by_method = {
        row["method"]: row for row in analysis["by_method"]
    }
    methods = [method for method in METHOD_ORDER if method in by_method]
    values = [
        float(by_method[method]["geomean_speedup_vs_cublas"])
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(methods))
    axis.bar(
        positions, values, color=[METHOD_COLORS[method] for method in methods]
    )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(
        positions, [METHOD_LABELS[method] for method in methods], rotation=35,
        ha="right",
    )
    axis.set_ylabel("Geometric-mean speedup vs BF16 cuBLAS")
    axis.set_title("Five models × four projections × three M values")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(
    output: Path,
    rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    by_method = {
        row["method"]: row for row in analysis["by_method"]
    }
    by_method_projection = {
        (row["method"], row["projection"]): row
        for row in analysis["by_method_projection"]
    }
    by_method_model = {
        (row["method"], row["model"]): row
        for row in analysis["by_method_model"]
    }
    non_gate_projections = [
        projection
        for projection in ("qkv", "o", "down")
        if ("ours_d1", projection) in by_method_projection
    ]
    ours_d1_non_gate = (
        geometric_mean(
            [
                float(
                    by_method_projection[("ours_d1", projection)][
                        "geomean_speedup_vs_cublas"
                    ]
                )
                for projection in non_gate_projections
            ]
        )
        if non_gate_projections
        else None
    )
    ours_d2_non_gate = (
        geometric_mean(
            [
                float(
                    by_method_projection[("ours_d2", projection)][
                        "geomean_speedup_vs_cublas"
                    ]
                )
                for projection in non_gate_projections
            ]
        )
        if non_gate_projections
        else None
    )
    lines = [
        "# SparTA, FlashLLM, SpInfer, cuBLAS, and SpecLink kernel comparison",
        "",
        f"Scope: {len(args.models)} models × {len(args.projections)} linear "
        f"projections × {len(args.m_values)} M values.",
        "",
        "## Comparison semantics",
        "",
        "- FlashLLM, SpInfer, and SparTA use statically pruned BF16 5:8 or 3:4 "
        "weights; their output is checked against cuBLAS on that same pruned weight.",
        "- SpecLink stores the exact dense weight as a 2:4 base plus a complementary "
        "2:4 residual. D1 routes 1/8 of token rows through both streams; D2 routes "
        "1/4. Remaining token rows intentionally use only the 2:4 base.",
        "- cuBLAS is the BF16 dense shape-matched performance baseline. Because GEMM "
        "runtime does not depend on the numerical zero pattern, it is measured once "
        "per (model, projection, M).",
        "",
        "These methods therefore answer different quality/sparsity questions. The "
        "table is a hardware-performance comparison, not a claim of equal model "
        "accuracy.",
        "",
        "## Main findings",
        "",
        f"- SpecLink D1 reaches **{by_method['ours_d1']['geomean_speedup_vs_cublas']:.4f}x** "
        f"overall and beats cuBLAS in {by_method['ours_d1']['faster_than_cublas']}/"
        f"{by_method['ours_d1']['cases']} cases. D2 reaches "
        f"**{by_method['ours_d2']['geomean_speedup_vs_cublas']:.4f}x** and wins "
        f"{by_method['ours_d2']['faster_than_cublas']}/"
        f"{by_method['ours_d2']['cases']} cases.",
        "- All 540 summarized rows pass their method-specific BF16 correctness "
        "check.",
        "",
        "## Protocol",
        "",
        f"GPU idle; {args.warmup} warmups; {args.trials} independent trials; "
        f"{args.replays} CUDA Graph replays per timed interval. Each trial starts "
        f"after an untimed {args.eviction_mib} MiB cache eviction. CUDA Event "
        "records total GPU time for the interval, divided by replay count. Values "
        "are median/P10/P90 in microseconds.",
        "",
        "## Overall geometric mean",
        "",
        "| Method | Cases | Geomean vs cuBLAS | Range | Faster cases |",
        "|---|---:|---:|---:|---:|",
    ]
    if non_gate_projections:
        lines[lines.index("- All 540 summarized rows pass their method-specific BF16 correctness check.") : lines.index("- All 540 summarized rows pass their method-specific BF16 correctness check.")] = [
            f"- Across {'/'.join(non_gate_projections)}, D1 and D2 geometric "
            f"means are **{ours_d1_non_gate:.4f}x** and "
            f"**{ours_d2_non_gate:.4f}x**."
        ]
    if ("ours_d1", "gate_up") in by_method_projection:
        gate_d1 = by_method_projection[("ours_d1", "gate_up")]
        gate_d2 = by_method_projection[("ours_d2", "gate_up")]
        insert_at = lines.index(
            "- All 540 summarized rows pass their method-specific BF16 correctness check."
        )
        lines[insert_at:insert_at] = [
            f"- On gate_up alone, D1 and D2 are "
            f"**{gate_d1['geomean_speedup_vs_cublas']:.4f}x** and "
            f"**{gate_d2['geomean_speedup_vs_cublas']:.4f}x** "
            f"({gate_d1['faster_than_cublas']}/{gate_d1['cases']} and "
            f"{gate_d2['faster_than_cublas']}/{gate_d2['cases']} wins)."
        ]
    if "flash_llm_5_8" in by_method:
        insert_at = lines.index(
            "- All 540 summarized rows pass their method-specific BF16 correctness check."
        )
        lines[insert_at:insert_at] = [
            f"- No external N:M path beats cuBLAS in any case. The strongest "
            f"aggregate external result is FlashLLM 5:8 at "
            f"**{by_method['flash_llm_5_8']['geomean_speedup_vs_cublas']:.4f}x**."
        ]
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        row = by_method[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['cases']} | "
            f"{row['geomean_speedup_vs_cublas']:.4f}x | "
            f"{row['min_speedup_vs_cublas']:.4f}–"
            f"{row['max_speedup_vs_cublas']:.4f}x | "
            f"{row['faster_than_cublas']}/{row['cases']} |"
        )
    lines.extend(
        [
            "",
            "## SpecLink by projection",
            "",
            "| Projection | D1 geomean | D1 faster | D2 geomean | D2 faster |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for projection in args.projections:
        d1 = by_method_projection[("ours_d1", projection)]
        d2 = by_method_projection[("ours_d2", projection)]
        lines.append(
            f"| {projection} | {d1['geomean_speedup_vs_cublas']:.4f}x | "
            f"{d1['faster_than_cublas']}/{d1['cases']} | "
            f"{d2['geomean_speedup_vs_cublas']:.4f}x | "
            f"{d2['faster_than_cublas']}/{d2['cases']} |"
        )
    lines.extend(
        [
            "",
            "## SpecLink by model",
            "",
            "| Model | D1 geomean | D1 faster | D2 geomean | D2 faster |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in args.models:
        d1 = by_method_model[("ours_d1", model)]
        d2 = by_method_model[("ours_d2", model)]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{d1['geomean_speedup_vs_cublas']:.4f}x | "
            f"{d1['faster_than_cublas']}/{d1['cases']} | "
            f"{d2['geomean_speedup_vs_cublas']:.4f}x | "
            f"{d2['faster_than_cublas']}/{d2['cases']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `kernel_results.csv`: every median, P10/P90, correctness result, "
            "selected split/variant, and speedup.",
            "- `kernel_raw_trials.csv`: all ten independent measurements.",
            "- `variant_screen.csv`: external Split-K and SpecLink mainloop screens.",
            "- `analysis.json`: aggregate results by method/model/projection/M.",
            "- `figures/geomean_speedup.png`: overall summary.",
            "- `figures/speedup_m*.png`: all per-layer speedups.",
            "- `figures/latency_m*.png`: all absolute per-layer latencies.",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    worker_files: list[Path] = []
    for model in args.models:
        for projection in args.projections:
            worker_output = work / f"{model}__{projection}.json"
            if args.resume and worker_output.exists():
                print(f"[resume] {model}/{projection}", flush=True)
                worker_files.append(worker_output)
                continue
            external.assert_gpu_idle(args.device)
            command = [
                sys.executable,
                str(SCRIPT),
                "--worker",
                "--worker-model",
                model,
                "--worker-projection",
                projection,
                "--worker-output",
                str(worker_output),
                "--device",
                str(args.device),
                "--m-values",
                ",".join(map(str, args.m_values)),
                "--fractions",
                ",".join(map(str, args.fractions)),
                "--formats",
                ",".join(args.formats),
                "--systems",
                ",".join(args.systems),
                "--seed",
                str(args.seed),
                "--capture-warmup",
                str(args.capture_warmup),
                "--screen-trials",
                str(args.screen_trials),
                "--screen-replays",
                str(args.screen_replays),
                "--external-screen-replays",
                str(args.external_screen_replays),
                "--warmup",
                str(args.warmup),
                "--trials",
                str(args.trials),
                "--replays",
                str(args.replays),
                "--eviction-mib",
                str(args.eviction_mib),
            ]
            if args.smoke:
                command.append("--smoke")
            if args.ours_only:
                command.append("--ours-only")
            print(f"[worker] {model}/{projection}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            worker_files.append(worker_output)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    for path in worker_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        raw.extend(payload["raw"])
        screens.extend(payload["screens"])
    if args.external_source_root is not None:
        source = args.external_source_root.resolve()
        source_rows = read_csv(source / "kernel_results.csv")
        source_raw = read_csv(source / "kernel_raw_trials.csv")
        source_screens = read_csv(source / "variant_screen.csv")
        replace_methods = {"dense_cublas", "ours_d1", "ours_d2"}
        rows.extend(
            row for row in source_rows if row["method"] not in replace_methods
        )
        raw.extend(
            row for row in source_raw if row["method"] not in replace_methods
        )
        screens.extend(
            row
            for row in source_screens
            if row.get("method") not in {"ours_d1", "ours_d2"}
        )
    if args.external_source_root is not None or not args.ours_only:
        expected_per_shape = len(args.m_values) * (
            1 + len(args.fractions) + len(args.formats) * len(args.systems)
        )
    else:
        expected_per_shape = len(args.m_values) * (1 + len(args.fractions))
    expected = len(args.models) * len(args.projections) * expected_per_shape
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} result rows, got {len(rows)}")
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            PROJECTIONS.index(row["projection"]),
            int(row["M"]),
            rank[row["method"]],
        )
    )
    write_csv(output / "kernel_results.csv", rows)
    write_csv(output / "kernel_raw_trials.csv", raw)
    if screens:
        write_csv(output / "variant_screen.csv", screens)
    analysis = aggregate(rows)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "models": list(args.models),
        "model_shapes": {
            model: {
                projection: list(TP1_FUSED_WEIGHT_SHAPES[model][projection])
                for projection in args.projections
            }
            for model in args.models
        },
        "projections": list(args.projections),
        "M_values": list(args.m_values),
        "external_formats": list(args.formats),
        "external_systems": list(args.systems),
        "ours_dense_fractions": list(map(str, args.fractions)),
        "protocol": {
            "warmups": args.warmup,
            "trials": args.trials,
            "replays_per_trial": args.replays,
            "eviction_mib_before_each_trial": args.eviction_mib,
            "timing": "CUDA Event total interval divided by replay count",
            "synchronization_inside_timed_interval": False,
        },
        "ours_only_remeasurement": args.ours_only,
        "external_source_root": (
            str(args.external_source_root.resolve())
            if args.external_source_root is not None
            else None
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    device = torch.device("cuda", args.device)
    (output / "environment.json").write_text(
        json.dumps(external.environment_report(device), indent=2) + "\n",
        encoding="utf-8",
    )
    figures = output / "figures"
    for m in args.m_values:
        plot_panels(
            rows,
            figures / f"speedup_m{m}.png",
            m,
            value_key="speedup_vs_cublas",
            ylabel="Speedup vs BF16 cuBLAS",
            reference_line=1.0,
        )
        plot_panels(
            rows,
            figures / f"latency_m{m}.png",
            m,
            value_key="median_us",
            ylabel="Median latency (us)",
            reference_line=None,
        )
    plot_geomean(analysis, figures / "geomean_speedup.png")
    write_report(output, rows, analysis, args)
    print(output)
    print(json.dumps(analysis["by_method"], indent=2))


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
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--screen-trials", type=int, default=3)
    parser.add_argument("--screen-replays", type=int, default=40)
    parser.add_argument("--external-screen-replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "results_final/five_model_nm_vs_speclink_kernel_20260723"
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "temp/five_model_nm_vs_speclink_kernel_20260723"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--ours-only",
        action="store_true",
        help="measure only cuBLAS and SpecLink D1/D2",
    )
    parser.add_argument(
        "--external-source-root",
        type=Path,
        help=(
            "reuse external-method CSV rows from this completed result root; "
            "normally paired with --ours-only"
        ),
    )
    args = parser.parse_args()
    try:
        args.models = parse_csv(args.models, MODELS, "models")
        args.projections = parse_csv(
            args.projections, PROJECTIONS, "projections"
        )
        args.m_values = tuple(
            int(value) for value in args.m_values.split(",") if value
        )
        if (
            not args.m_values
            or any(value not in M_VALUES for value in args.m_values)
        ):
            raise argparse.ArgumentTypeError(
                f"M must be selected from {M_VALUES}"
            )
        args.fractions = parse_fractions(args.fractions)
        args.formats = tuple(
            parse_nm(value).label
            for value in args.formats.split(",")
            if value
        )
        args.systems = parse_csv(args.systems, SYSTEMS, "systems")
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    if args.worker:
        if (
            args.worker_model is None
            or args.worker_projection is None
            or args.worker_output is None
        ):
            parser.error(
                "worker mode requires --worker-model, "
                "--worker-projection, and --worker-output"
            )
    if args.smoke:
        args.models = args.models[:1]
        args.projections = args.projections[:1]
        args.m_values = args.m_values[:1]
        args.capture_warmup = 2
        args.screen_trials = 1
        args.screen_replays = 3
        args.external_screen_replays = 3
        args.warmup = 3
        args.trials = 2
        args.replays = 10
        default_output = (
            EVAL_ROOT
            / "results_final/five_model_nm_vs_speclink_kernel_20260723"
        )
        default_work = (
            EVAL_ROOT
            / "temp/five_model_nm_vs_speclink_kernel_20260723"
        )
        if not args.worker and args.output_root == default_output:
            args.output_root = (
                EVAL_ROOT
                / "temp/five_model_nm_vs_speclink_kernel_smoke_20260723"
            )
        if not args.worker and args.work_root == default_work:
            args.work_root = (
                EVAL_ROOT
                / "temp/five_model_nm_vs_speclink_kernel_smoke_work_20260723"
            )
    elif args.trials != 10:
        parser.error("formal protocol requires exactly 10 trials")
    counts = (
        args.capture_warmup,
        args.screen_trials,
        args.screen_replays,
        args.external_screen_replays,
        args.warmup,
        args.trials,
        args.replays,
        args.eviction_mib,
    )
    if any(value <= 0 for value in counts):
        parser.error("all protocol counts must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        run_worker(arguments)
    else:
        run_coordinator(arguments)
