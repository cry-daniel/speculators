#!/usr/bin/env python3
"""Formal full-decoder-layer comparison for five model shapes.

The complete layer contains RMSNorm, fused QKV, optional Qwen Q/K RMSNorm,
RoPE, explicit GQA attention, softmax, O projection, residual, RMSNorm,
Gate/Up, SiLU-times-Up, Down projection, and the final residual.

SparTA, FlashLLM, and SpInfer use static 5:8 weights for all four linears.
SpecLink uses its exact dense weight decomposition and routes 1/8 of token
rows through the complementary 2:4 stream.  The pure-2:4 cuSPARSELt path is
reported as a non-exact-output upper bound.  Each request contributes one
current token plus seven draft tokens, so B=64/128/256 maps to
M=512/1024/2048 at context length 128.
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
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
EVAL_ROOT = ROOT / "examples/evaluate/eval-guidellm"
BENCH_SCRIPTS = EVAL_ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))

from other_systems import benchmark_common as timing  # noqa: E402
from other_systems import bench_layer as external_layer  # noqa: E402
import bench_decoder_layer_residual_complement as hybrid_layer  # noqa: E402
from speculators.speclink import (  # noqa: E402
    TP1_FUSED_WEIGHT_SHAPES,
    select_cusparselt_algorithm,
)


MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
BATCH_SIZES = (64, 128, 256)
METHODS = (
    "sparta_5_8",
    "flash_llm_5_8",
    "spinfer_5_8",
    "dense_cublas",
    "speclink_d1",
    "pure_24_upper_bound",
)
EXTERNAL_METHODS = {
    "sparta": "sparta_5_8",
    "flash_llm": "flash_llm_5_8",
    "spinfer": "spinfer_5_8",
}
METHOD_LABELS = {
    "sparta_5_8": "SparTA 5:8",
    "flash_llm_5_8": "FlashLLM 5:8",
    "spinfer_5_8": "SpInfer 5:8",
    "dense_cublas": "cuBLAS Dense",
    "speclink_d1": "SpecLink D1 (1/8 dense)",
    "pure_24_upper_bound": "2:4 Upper Bound",
}
METHOD_COLORS = {
    "sparta_5_8": "#54a24b",
    "flash_llm_5_8": "#4c78a8",
    "spinfer_5_8": "#f58518",
    "dense_cublas": "#777777",
    "speclink_d1": "#b279a2",
    "pure_24_upper_bound": "#222222",
}
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama3-70B",
}
SEED = 20260723


def parse_csv(
    value: str, allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid {label}: {unknown or value!r}"
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


def check_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    check = {
        "correct": bool(
            torch.allclose(actual, expected, atol=atol, rtol=rtol)
        ),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }
    if not check["correct"]:
        raise RuntimeError(f"full-layer correctness failure: {check}")
    return check


def append_measurement(
    rows: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    *,
    model: str,
    batch: int,
    method: str,
    summary: timing.TimingSummary,
    samples: Sequence[float],
    check: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "batch_size": batch,
            "M": batch * 8,
            "draft_tokens_per_request": 7,
            "context_length_including_current": 128,
            "max_visible_length": 135,
            "method": method,
            "method_label": METHOD_LABELS[method],
            **summary.as_dict(),
            **check,
            **(extra or {}),
        }
    )
    raw.extend(
        {
            "model": model,
            "batch_size": batch,
            "M": batch * 8,
            "method": method,
            "trial": trial,
            "latency_us": latency,
        }
        for trial, latency in enumerate(samples)
    )


def measure_graph(
    captured: Any,
    eviction: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[timing.TimingSummary, list[float]]:
    return timing.formal_measure(
        captured,
        eviction,
        warmup=args.warmup,
        trials=args.trials,
        replays=args.replays,
    )


def retune_hybrid_algorithms(
    state: hybrid_layer.LayerState,
    model: str,
    m: int,
    device: torch.device,
    seed: int,
) -> dict[str, int]:
    selected: dict[str, int] = {}
    for name, projection in state.projections.items():
        n, k = TP1_FUSED_WEIGHT_SHAPES[model][name]
        case = hybrid_layer.common.ShapeCase(model, name, m, k, n)
        sample = hybrid_layer.common.make_input(
            case, seed, device, purpose="full_layer_per_m_tune"
        )
        algorithm_id = select_cusparselt_algorithm(
            projection.runtime.cusparselt, sample
        )
        projection.algorithm_id = algorithm_id
        selected[name] = algorithm_id
        del sample
    return selected


def run_hybrid_methods(
    args: argparse.Namespace,
    model: str,
    device: torch.device,
    eviction: torch.Tensor,
    rows: list[dict[str, Any]],
    raw: list[dict[str, Any]],
) -> None:
    state = hybrid_layer.prepare_layer(
        model, max(args.batch_sizes), device, args.seed
    )
    for batch in args.batch_sizes:
        m = batch * 8
        algorithm_ids = retune_hybrid_algorithms(
            state, model, m, device, args.seed
        )
        hidden = hybrid_layer.make_hidden(model, batch, device, args.seed)
        confidence = hybrid_layer.make_confidence(batch, device, args.seed)
        dense_indices = hybrid_layer.dense_indices_from_confidence(
            confidence, 1, "global"
        )
        sparse_mask = torch.ones(m, dtype=torch.bool, device=device)
        sparse_mask[dense_indices] = False
        sparse_indices = (
            torch.nonzero(sparse_mask, as_tuple=False)
            .flatten()
            .contiguous()
        )

        dense_graph = hybrid_layer.common.capture_graph(
            lambda: hybrid_layer.layer_forward(
                hidden, state, method="dense"
            ),
            warmup=args.capture_warmup,
        )
        pure_graph = hybrid_layer.common.capture_graph(
            lambda: hybrid_layer.layer_forward(
                hidden, state, method="pure_sparse"
            ),
            warmup=args.capture_warmup,
        )
        speclink_graph = hybrid_layer.common.capture_multistream_graph(
            lambda: hybrid_layer.layer_forward(
                hidden,
                state,
                method="residual_complement",
                dense_indices=dense_indices,
                sparse_indices=sparse_indices,
            ),
            state.resources,
            warmup=args.capture_warmup,
            device=device,
        )

        pure_reference = hybrid_layer.layer_forward(
            hidden, state, method="pure_sparse"
        )
        pure_check = check_outputs(
            pure_graph.output, pure_reference, atol=0.2, rtol=0.1
        )
        timer = hybrid_layer.PhaseEvents()
        speclink_reference = hybrid_layer.layer_forward(
            hidden,
            state,
            method="residual_complement",
            dense_indices=dense_indices,
            timer=timer,
        )
        timer.finish()
        speclink_check = check_outputs(
            speclink_graph.output,
            speclink_reference,
            atol=0.2,
            rtol=0.1,
        )

        for method, graph, check, extra in (
            (
                "dense_cublas",
                dense_graph,
                {
                    "correct": True,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                },
                {
                    "weight_semantic": "exact_dense",
                    "dense_fraction": "8/8",
                    "dense_rows": m,
                    "sparse_rows": 0,
                },
            ),
            (
                "pure_24_upper_bound",
                pure_graph,
                pure_check,
                {
                    "weight_semantic": "all_tokens_pure_2_4",
                    "dense_fraction": "0/8",
                    "dense_rows": 0,
                    "sparse_rows": m,
                    "cusparselt_algorithm_ids": json.dumps(
                        algorithm_ids, sort_keys=True
                    ),
                },
            ),
            (
                "speclink_d1",
                speclink_graph,
                speclink_check,
                {
                    "weight_semantic": "token_hybrid_2_4_complement",
                    "dense_fraction": "1/8",
                    "dense_rows": int(dense_indices.numel()),
                    "sparse_rows": int(sparse_indices.numel()),
                    "cusparselt_algorithm_ids": json.dumps(
                        algorithm_ids, sort_keys=True
                    ),
                },
            ),
        ):
            summary, samples = measure_graph(graph, eviction, args)
            append_measurement(
                rows,
                raw,
                model=model,
                batch=batch,
                method=method,
                summary=summary,
                samples=samples,
                check=check,
                extra=extra,
            )
            print(
                f"[layer] {model} M={m} {method}: "
                f"{summary.median_us:.3f} us",
                flush=True,
            )

        del (
            hidden,
            confidence,
            dense_indices,
            sparse_indices,
            sparse_mask,
            dense_graph,
            pure_graph,
            speclink_graph,
            pure_reference,
            speclink_reference,
        )
        gc.collect()
        torch.cuda.empty_cache()
    del state
    gc.collect()
    torch.cuda.empty_cache()


def run_external_methods(
    args: argparse.Namespace,
    model: str,
    device: torch.device,
    eviction: torch.Tensor,
    rows: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    screens: list[dict[str, Any]],
) -> None:
    for system in args.external_systems:
        method = EXTERNAL_METHODS[system]
        state = external_layer.prepare_layer(
            model,
            system,
            "5:8",
            max(args.batch_sizes),
            device,
            args.seed,
        )
        for batch in args.batch_sizes:
            m = batch * 8
            for screen in external_layer.tune_layer_splits(
                state,
                model,
                m,
                device,
                args.seed,
                args.split_screen_repeats,
            ):
                screens.append(
                    {
                        "model": model,
                        "method": method,
                        "batch_size": batch,
                        **screen,
                    }
                )
            hidden = hybrid_layer.make_hidden(
                model, batch, device, args.seed
            )
            dense_reference = external_layer.layer_forward(
                hidden, state, external=False
            )
            graph = timing.capture(
                lambda: external_layer.layer_forward(
                    hidden, state, external=True
                ),
                warmup=args.capture_warmup,
            )
            check = check_outputs(
                graph.output,
                dense_reference,
                atol=0.5,
                rtol=0.2,
            )
            summary, samples = measure_graph(graph, eviction, args)
            append_measurement(
                rows,
                raw,
                model=model,
                batch=batch,
                method=method,
                summary=summary,
                samples=samples,
                check=check,
                extra={
                    "weight_semantic": "static_5_8_pruned",
                    "dense_fraction": "",
                    "dense_rows": "",
                    "sparse_rows": "",
                    "projection_splits": json.dumps(
                        {
                            name: projection.split_by_m[m]
                            for name, projection in state.projections.items()
                        },
                        sort_keys=True,
                    ),
                },
            )
            print(
                f"[layer] {model} M={m} {method}: "
                f"{summary.median_us:.3f} us",
                flush=True,
            )
            del hidden, dense_reference, graph
            gc.collect()
            torch.cuda.empty_cache()
        del state
        gc.collect()
        torch.cuda.empty_cache()


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    eviction.zero_()
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    run_hybrid_methods(
        args, args.worker_model, device, eviction, rows, raw
    )
    run_external_methods(
        args, args.worker_model, device, eviction, rows, raw, screens
    )
    payload = {
        "model": args.worker_model,
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
    return math.exp(statistics.mean(math.log(float(value)) for value in values))


def normalize(rows: list[dict[str, Any]]) -> None:
    sparta = {
        (row["model"], int(row["M"])): float(row["median_us"])
        for row in rows
        if row["method"] == "sparta_5_8"
    }
    for row in rows:
        baseline = sparta[(row["model"], int(row["M"]))]
        row["sparta_5_8_median_us"] = baseline
        row["speedup_vs_sparta_5_8"] = (
            baseline / float(row["median_us"])
        )


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["method"]))
        buckets.setdefault(key, []).append(
            float(row["speedup_vs_sparta_5_8"])
        )
    result: list[dict[str, Any]] = []
    for (model, method), values in sorted(buckets.items()):
        result.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "cases": len(values),
                "geomean_speedup_vs_sparta_5_8": geometric_mean(values),
                "min_speedup_vs_sparta_5_8": min(values),
                "max_speedup_vs_sparta_5_8": max(values),
            }
        )
    for method in METHODS:
        values = [
            float(row["speedup_vs_sparta_5_8"])
            for row in rows
            if row["method"] == method
        ]
        result.append(
            {
                "model": "all",
                "model_label": "All five models",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "cases": len(values),
                "geomean_speedup_vs_sparta_5_8": geometric_mean(values),
                "min_speedup_vs_sparta_5_8": min(values),
                "max_speedup_vs_sparta_5_8": max(values),
            }
        )
    return result


def style_bar(container: Any, method: str) -> None:
    if method != "pure_24_upper_bound":
        return
    for bar in container:
        bar.set_facecolor("none")
        bar.set_edgecolor(METHOD_COLORS[method])
        bar.set_linewidth(1.6)
        bar.set_linestyle("--")
        bar.set_hatch("//")


def plot_model(
    rows: Sequence[dict[str, Any]], model: str, output: Path
) -> None:
    selected = [row for row in rows if row["model"] == model]
    indexed = {
        (int(row["M"]), str(row["method"])): row for row in selected
    }
    m_values = sorted({int(row["M"]) for row in selected})
    positions = np.arange(len(m_values), dtype=np.float64)
    width = 0.135
    offsets = (
        np.arange(len(METHODS), dtype=np.float64)
        - (len(METHODS) - 1) / 2
    ) * width
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for method_index, method in enumerate(METHODS):
        latency = [
            float(indexed[(m, method)]["median_us"]) for m in m_values
        ]
        speedup = [
            float(indexed[(m, method)]["speedup_vs_sparta_5_8"])
            for m in m_values
        ]
        kwargs = {
            "color": (
                "none"
                if method == "pure_24_upper_bound"
                else METHOD_COLORS[method]
            ),
            "edgecolor": METHOD_COLORS[method],
            "label": METHOD_LABELS[method],
        }
        latency_bars = axes[0].bar(
            positions + offsets[method_index],
            latency,
            width,
            **kwargs,
        )
        speedup_bars = axes[1].bar(
            positions + offsets[method_index],
            speedup,
            width,
            **kwargs,
        )
        style_bar(latency_bars, method)
        style_bar(speedup_bars, method)
    axes[0].set_ylabel("Full-layer median latency (μs), lower is better")
    axes[1].set_ylabel("Speedup over SparTA 5:8 (×), higher is better")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    for axis in axes:
        axis.set_xticks(positions, [f"M={m}" for m in m_values])
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        f"{MODEL_LABELS[model]} complete decoder layer, context=128",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_model_geomean(
    summary: Sequence[dict[str, Any]], output: Path
) -> None:
    selected = [row for row in summary if row["model"] != "all"]
    models = [
        model
        for model in MODELS
        if any(row["model"] == model for row in selected)
    ]
    indexed = {
        (row["model"], row["method"]): float(
            row["geomean_speedup_vs_sparta_5_8"]
        )
        for row in selected
    }
    positions = np.arange(len(models), dtype=np.float64)
    width = 0.135
    offsets = (
        np.arange(len(METHODS), dtype=np.float64)
        - (len(METHODS) - 1) / 2
    ) * width
    figure, axis = plt.subplots(figsize=(14, 5.8))
    for method_index, method in enumerate(METHODS):
        values = [indexed[(model, method)] for model in models]
        bars = axis.bar(
            positions + offsets[method_index],
            values,
            width,
            color=(
                "none"
                if method == "pure_24_upper_bound"
                else METHOD_COLORS[method]
            ),
            edgecolor=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        style_bar(bars, method)
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(
        positions, [MODEL_LABELS[model] for model in models]
    )
    axis.set_ylabel(
        "Full-layer geomean speedup over SparTA 5:8 (×)"
    )
    axis.set_title("Geometric mean over M=512/1024/2048")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=3, frameon=False, loc="upper center")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(
    output: Path,
    rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    indexed = {
        (row["model"], int(row["M"]), row["method"]): row for row in rows
    }
    aggregate_index = {
        (row["model"], row["method"]): row for row in summary
    }
    lines = [
        "# Five-model complete decoder-layer comparison",
        "",
        "The timed layer contains RMSNorm, fused QKV, optional Qwen Q/K "
        "RMSNorm, RoPE, explicit GQA attention, softmax, O projection, "
        "residual, RMSNorm, Gate/Up, SiLU-times-Up, Down projection, and "
        "the final residual.",
        "",
        "Every request contributes one current token and seven draft tokens. "
        "Thus B=64/128/256 corresponds to M=512/1024/2048. The context "
        "length including the current token is 128.",
        "",
        "## Semantics",
        "",
        "- SparTA, FlashLLM, and SpInfer use static BF16 5:8 weights on all "
        "four linears.",
        "- cuBLAS uses the exact dense weights.",
        "- SpecLink D1 sends 1/8 of token rows through both the 2:4 base and "
        "complement; the remaining 7/8 use the 2:4 base.",
        "- `2:4 Upper Bound` sends every row through only the 2:4 base and "
        "does not produce an exact dense-layer output.",
        "",
        "## Protocol",
        "",
        f"GPU idle; {args.warmup} warmups; {args.trials} trials; "
        f"{args.replays} continuous CUDA Graph replays per trial; "
        f"{args.eviction_mib} MiB cache eviction before each trial. CUDA "
        "Event measures the whole replay interval and the result is divided "
        "by the replay count. All values are in microseconds.",
        "",
        "## Geometric mean over all five models and three M values",
        "",
        "| Method | Cases | Speedup vs SparTA 5:8 | Range |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate_index[("all", method)]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['cases']} | "
            f"{row['geomean_speedup_vs_sparta_5_8']:.4f}x | "
            f"{row['min_speedup_vs_sparta_5_8']:.4f}–"
            f"{row['max_speedup_vs_sparta_5_8']:.4f}x |"
        )
    overall_speclink = float(
        aggregate_index[("all", "speclink_d1")][
            "geomean_speedup_vs_sparta_5_8"
        ]
    )
    overall_dense = float(
        aggregate_index[("all", "dense_cublas")][
            "geomean_speedup_vs_sparta_5_8"
        ]
    )
    overall_upper = float(
        aggregate_index[("all", "pure_24_upper_bound")][
            "geomean_speedup_vs_sparta_5_8"
        ]
    )
    lines.extend(
        [
            "",
            "SpecLink is "
            f"**{overall_speclink / overall_dense:.4f}x** faster than "
            "the complete cuBLAS dense layer and reaches "
            f"**{overall_speclink / overall_upper:.2%}** of the measured "
            "pure-2:4 upper-bound performance.",
            "",
            "## Geometric mean by model",
            "",
            "| Model | "
            + " | ".join(METHOD_LABELS[method] for method in METHODS)
            + " |",
            "|---|" + "---:|" * len(METHODS),
        ]
    )
    for model in args.models:
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            + " | ".join(
                f"{aggregate_index[(model, method)]['geomean_speedup_vs_sparta_5_8']:.4f}x"
                for method in METHODS
            )
            + " |"
        )
    for model in args.models:
        lines.extend(
            [
                "",
                f"## {MODEL_LABELS[model]}",
                "",
                "### Absolute complete-layer latency",
                "",
                "| M | "
                + " | ".join(METHOD_LABELS[method] for method in METHODS)
                + " |",
                "|---:|" + "---:|" * len(METHODS),
            ]
        )
        for m in (batch * 8 for batch in args.batch_sizes):
            values = [
                float(indexed[(model, m, method)]["median_us"])
                for method in METHODS
            ]
            lines.append(
                f"| {m} | "
                + " | ".join(f"{value:.3f} μs" for value in values)
                + " |"
            )
        lines.extend(
            [
                "",
                "### Speedup over SparTA 5:8",
                "",
                "| M | "
                + " | ".join(METHOD_LABELS[method] for method in METHODS)
                + " |",
                "|---:|" + "---:|" * len(METHODS),
            ]
        )
        for m in (batch * 8 for batch in args.batch_sizes):
            values = [
                float(
                    indexed[(model, m, method)][
                        "speedup_vs_sparta_5_8"
                    ]
                )
                for method in METHODS
            ]
            lines.append(
                f"| {m} | "
                + " | ".join(f"{value:.4f}x" for value in values)
                + " |"
            )
        lines.extend(
            [
                "",
                "Geometric mean over the three M values: "
                + ", ".join(
                    f"{METHOD_LABELS[method]} "
                    f"{aggregate_index[(model, method)]['geomean_speedup_vs_sparta_5_8']:.4f}x"
                    for method in METHODS
                )
                + ".",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `full_layer_results.csv`: medians, P10/P90, correctness, "
            "selected algorithms/splits, and SparTA-normalized speedups.",
            "- `full_layer_raw_trials.csv`: all ten formal trials.",
            "- `external_split_screen.csv`: external-kernel Split-K screens.",
            "- `full_layer_summary.csv`: per-model and overall geometric means.",
            "- `figures/MODEL_full_layer.png`: absolute time and normalized "
            "speedup for each model.",
            "- `figures/geomean_by_model.png`: model-level summary.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    worker_files: list[Path] = []
    for model in args.models:
        worker_output = work / f"{model}.json"
        if args.resume and worker_output.exists():
            print(f"[resume] {model}", flush=True)
            worker_files.append(worker_output)
            continue
        timing.assert_gpu_idle(args.device)
        command = [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--worker-model",
            model,
            "--worker-output",
            str(worker_output),
            "--batch-sizes",
            ",".join(map(str, args.batch_sizes)),
            "--external-systems",
            ",".join(args.external_systems),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--capture-warmup",
            str(args.capture_warmup),
            "--warmup",
            str(args.warmup),
            "--trials",
            str(args.trials),
            "--replays",
            str(args.replays),
            "--eviction-mib",
            str(args.eviction_mib),
            "--split-screen-repeats",
            str(args.split_screen_repeats),
        ]
        if args.smoke:
            command.append("--smoke")
        print(f"[worker] {model}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        worker_files.append(worker_output)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    for worker_file in worker_files:
        payload = json.loads(worker_file.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        raw.extend(payload["raw"])
        screens.extend(payload["screens"])
    expected_rows = (
        len(args.models) * len(args.batch_sizes) * len(METHODS)
    )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} result rows, got {len(rows)}"
        )
    expected_raw = expected_rows * args.trials
    if len(raw) != expected_raw:
        raise RuntimeError(
            f"expected {expected_raw} raw trials, got {len(raw)}"
        )
    rank = {method: index for index, method in enumerate(METHODS)}
    rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            int(row["M"]),
            rank[row["method"]],
        )
    )
    normalize(rows)
    summary = aggregate(rows)
    write_csv(output / "full_layer_results.csv", rows)
    write_csv(output / "full_layer_raw_trials.csv", raw)
    write_csv(output / "full_layer_summary.csv", summary)
    if screens:
        write_csv(output / "external_split_screen.csv", screens)
    metadata = {
        "models": list(args.models),
        "batch_sizes": list(args.batch_sizes),
        "M_values": [batch * 8 for batch in args.batch_sizes],
        "context_length_including_current": 128,
        "draft_tokens_per_request": 7,
        "methods": list(METHODS),
        "protocol": {
            "capture_warmup": args.capture_warmup,
            "warmups": args.warmup,
            "trials": args.trials,
            "replays_per_trial": args.replays,
            "eviction_mib_before_each_trial": args.eviction_mib,
            "timing": "CUDA Event total interval divided by replay count",
            "synchronization_inside_timed_interval": False,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figures = output / "figures"
    for model in args.models:
        plot_model(rows, model, figures / f"{model}_full_layer.png")
    plot_model_geomean(summary, figures / "geomean_by_model.png")
    write_report(output / "report.md", rows, summary, args)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--batch-sizes", default=",".join(map(str, BATCH_SIZES))
    )
    parser.add_argument(
        "--external-systems",
        default="sparta,flash_llm,spinfer",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--split-screen-repeats", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "results_final/"
            "five_model_full_layer_5_8_vs_speclink_d1_20260723"
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "temp/"
            "five_model_full_layer_5_8_vs_speclink_d1_20260723"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        args.models = parse_csv(args.models, MODELS, "models")
        args.external_systems = parse_csv(
            args.external_systems,
            tuple(EXTERNAL_METHODS),
            "external systems",
        )
        args.batch_sizes = tuple(
            int(item) for item in args.batch_sizes.split(",") if item
        )
        if (
            not args.batch_sizes
            or any(item not in BATCH_SIZES for item in args.batch_sizes)
        ):
            raise argparse.ArgumentTypeError(
                f"batch sizes must be selected from {BATCH_SIZES}"
            )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.worker and (
        args.worker_model is None or args.worker_output is None
    ):
        parser.error(
            "worker mode requires --worker-model and --worker-output"
        )
    if args.smoke:
        args.models = args.models[:1]
        args.batch_sizes = args.batch_sizes[:1]
        args.capture_warmup = 2
        args.warmup = 3
        args.trials = 2
        args.replays = 5
        args.split_screen_repeats = 2
        default_output = (
            EVAL_ROOT
            / "results_final/"
            "five_model_full_layer_5_8_vs_speclink_d1_20260723"
        )
        default_work = (
            EVAL_ROOT
            / "temp/"
            "five_model_full_layer_5_8_vs_speclink_d1_20260723"
        )
        if not args.worker and args.output_root == default_output:
            args.output_root = (
                EVAL_ROOT / "temp/full_layer_five_model_smoke"
            )
        if not args.worker and args.work_root == default_work:
            args.work_root = (
                EVAL_ROOT / "temp/full_layer_five_model_smoke_work"
            )
    elif args.trials != 10:
        parser.error("formal protocol requires exactly 10 trials")
    counts = (
        args.capture_warmup,
        args.warmup,
        args.trials,
        args.replays,
        args.eviction_mib,
        args.split_screen_repeats,
    )
    if any(value <= 0 for value in counts):
        parser.error("all protocol counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
