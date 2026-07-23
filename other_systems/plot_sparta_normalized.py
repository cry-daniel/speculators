#!/usr/bin/env python3
"""Replot the formal five-model kernel results normalized to SparTA 5:8."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    "qwen3_8b",
    "llama3_1_8b",
    "qwen3_14b",
    "qwen3_32b",
    "llama3_70b",
)
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama-3-70B",
}
PROJECTIONS = ("qkv", "o", "gate_up", "down")
M_VALUES = (512, 1024, 2048)
METHODS = (
    "sparta_5_8",
    "flash_llm_5_8",
    "spinfer_5_8",
    "dense_cublas",
    "ours_d1",
    "pure_24_upper_bound",
)
METHOD_LABELS = {
    "sparta_5_8": "SparTA 5:8",
    "flash_llm_5_8": "FlashLLM 5:8",
    "spinfer_5_8": "SpInfer 5:8",
    "dense_cublas": "cuBLAS dense",
    "ours_d1": "SpecLink D1 (1/8 dense)",
    "pure_24_upper_bound": "2:4 Upper Bound",
}
METHOD_COLORS = {
    "sparta_5_8": "#54a24b",
    "flash_llm_5_8": "#4c78a8",
    "spinfer_5_8": "#f58518",
    "dense_cublas": "#777777",
    "ours_d1": "#b279a2",
    "pure_24_upper_bound": "#222222",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def normalize(
    rows: list[dict[str, str]],
    upper_bound_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = [*rows, *upper_bound_rows]
    indexed = {
        (
            row["model"],
            row["projection"],
            int(row["M"]),
            row["method"],
        ): row
        for row in rows
    }
    normalized: list[dict[str, Any]] = []
    for model in MODELS:
        for projection in PROJECTIONS:
            for m in M_VALUES:
                sparta = indexed[(model, projection, m, "sparta_5_8")]
                sparta_us = float(sparta["median_us"])
                for method in METHODS:
                    source = indexed[(model, projection, m, method)]
                    method_us = float(source["median_us"])
                    normalized.append(
                        {
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "projection": projection,
                            "M": m,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "sparsity": (
                                "1/8 dense tokens"
                                if method == "ours_d1"
                                else (
                                    "pure 2:4"
                                    if method == "pure_24_upper_bound"
                                    else (
                                        "dense"
                                        if method == "dense_cublas"
                                        else "5:8"
                                    )
                                )
                            ),
                            "median_us": method_us,
                            "sparta_5_8_median_us": sparta_us,
                            "speedup_vs_sparta_5_8": sparta_us / method_us,
                        }
                    )
    return normalized


def aggregate(
    rows: list[dict[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        buckets.setdefault(key, []).append(
            float(row["speedup_vs_sparta_5_8"])
        )
    return [
        {
            **dict(zip(keys, key, strict=True)),
            "cases": len(values),
            "geomean_speedup_vs_sparta_5_8": geometric_mean(values),
            "min_speedup_vs_sparta_5_8": min(values),
            "max_speedup_vs_sparta_5_8": max(values),
        }
        for key, values in sorted(buckets.items())
    ]


def plot_matrix(rows: list[dict[str, Any]], output: Path) -> None:
    indexed = {
        (
            row["model"],
            row["projection"],
            int(row["M"]),
            row["method"],
        ): float(row["speedup_vs_sparta_5_8"])
        for row in rows
    }
    figure, axes = plt.subplots(
        len(M_VALUES),
        len(MODELS),
        figsize=(24, 13),
        sharey=True,
        squeeze=False,
    )
    width = 0.135
    positions = np.arange(len(PROJECTIONS), dtype=np.float64)
    offsets = (
        np.arange(len(METHODS), dtype=np.float64)
        - (len(METHODS) - 1) / 2
    ) * width
    max_value = max(float(row["speedup_vs_sparta_5_8"]) for row in rows)
    for row_index, m in enumerate(M_VALUES):
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index][column_index]
            for method_index, method in enumerate(METHODS):
                values = [
                    indexed[(model, projection, m, method)]
                    for projection in PROJECTIONS
                ]
                upper_bound = method == "pure_24_upper_bound"
                axis.bar(
                    positions + offsets[method_index],
                    values,
                    width,
                    color=(
                        "none" if upper_bound else METHOD_COLORS[method]
                    ),
                    edgecolor=METHOD_COLORS[method],
                    linewidth=1.5 if upper_bound else 0.5,
                    linestyle="--" if upper_bound else "-",
                    hatch="//" if upper_bound else None,
                    label=METHOD_LABELS[method],
                )
            axis.axhline(
                1.0, color="black", linewidth=1, linestyle="--", alpha=0.8
            )
            axis.set_xticks(positions, PROJECTIONS, rotation=20)
            axis.set_ylim(0, max_value * 1.08)
            axis.grid(axis="y", alpha=0.22)
            axis.set_title(f"{MODEL_LABELS[model]}, M={m}")
            if column_index == 0:
                axis.set_ylabel("Speedup over SparTA 5:8 (×)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(
        "BF16 linear kernels normalized to SparTA 5:8",
        fontsize=17,
        y=0.965,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_geomean(summary: list[dict[str, Any]], output: Path) -> None:
    indexed = {row["method"]: row for row in summary}
    values = [
        float(indexed[method]["geomean_speedup_vs_sparta_5_8"])
        for method in METHODS
    ]
    positions = np.arange(len(METHODS))
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    bars = axis.bar(
        positions,
        values,
        color=[
            (
                "none"
                if method == "pure_24_upper_bound"
                else METHOD_COLORS[method]
            )
            for method in METHODS
        ],
        edgecolor=[METHOD_COLORS[method] for method in METHODS],
        linewidth=[
            1.8 if method == "pure_24_upper_bound" else 0.5
            for method in METHODS
        ],
    )
    upper = bars[METHODS.index("pure_24_upper_bound")]
    upper.set_linestyle("--")
    upper.set_hatch("//")
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(
        positions,
        [METHOD_LABELS[method] for method in METHODS],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Geometric-mean speedup over SparTA 5:8 (×)")
    axis.set_title("Five models × four projections × three M values")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_model_layer(
    summary: list[dict[str, Any]], output: Path
) -> None:
    indexed = {
        (row["model"], row["projection"], row["method"]): float(
            row["geomean_speedup_vs_sparta_5_8"]
        )
        for row in summary
    }
    figure, axes = plt.subplots(
        1,
        len(MODELS),
        figsize=(24, 5.5),
        sharey=True,
        squeeze=False,
    )
    width = 0.135
    positions = np.arange(len(PROJECTIONS), dtype=np.float64)
    offsets = (
        np.arange(len(METHODS), dtype=np.float64)
        - (len(METHODS) - 1) / 2
    ) * width
    max_value = max(indexed.values())
    for model_index, model in enumerate(MODELS):
        axis = axes[0][model_index]
        for method_index, method in enumerate(METHODS):
            values = [
                indexed[(model, projection, method)]
                for projection in PROJECTIONS
            ]
            upper_bound = method == "pure_24_upper_bound"
            axis.bar(
                positions + offsets[method_index],
                values,
                width,
                color="none" if upper_bound else METHOD_COLORS[method],
                edgecolor=METHOD_COLORS[method],
                linewidth=1.5 if upper_bound else 0.5,
                linestyle="--" if upper_bound else "-",
                hatch="//" if upper_bound else None,
                label=METHOD_LABELS[method],
            )
        axis.axhline(
            1.0, color="black", linewidth=1, linestyle="--", alpha=0.8
        )
        axis.set_xticks(positions, PROJECTIONS, rotation=20)
        axis.set_ylim(0, max_value * 1.08)
        axis.grid(axis="y", alpha=0.22)
        axis.set_title(MODEL_LABELS[model])
        if model_index == 0:
            axis.set_ylabel("Geomean speedup over SparTA 5:8 (×)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    figure.suptitle(
        "BF16 linear kernels by model and projection "
        "(geomean over M=512/1024/2048)",
        fontsize=17,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def write_report(
    output: Path,
    overall: list[dict[str, Any]],
    by_projection: list[dict[str, Any]],
    by_model_projection: list[dict[str, Any]],
) -> None:
    overall_by_method = {row["method"]: row for row in overall}
    projection_index = {
        (row["method"], row["projection"]): row for row in by_projection
    }
    model_projection_index = {
        (row["method"], row["model"], row["projection"]): row
        for row in by_model_projection
    }
    lines = [
        "# SparTA-normalized 5:8 comparison",
        "",
        "Normalization is performed independently for every "
        "`(model, projection, M)` shape:",
        "",
        "`speedup_vs_sparta = SparTA_5:8_latency / method_latency`",
        "",
        "Therefore SparTA is exactly 1.0 for every shape and values above 1.0 "
        "are faster than SparTA. FlashLLM, SpInfer, and SparTA use 5:8 weights; "
        "SpecLink uses D1, i.e. 1/8 dense-token rows. The hollow dashed "
        "`2:4 Upper Bound` is measured with cuSPARSELt while sending every "
        "token row through only the 2:4 base; it is not an exact dense-output "
        "method.",
        "",
        "## Overall geometric mean",
        "",
        "| Method | Cases | Speedup vs SparTA 5:8 | Range |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = overall_by_method[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['cases']} | "
            f"{row['geomean_speedup_vs_sparta_5_8']:.4f}x | "
            f"{row['min_speedup_vs_sparta_5_8']:.4f}–"
            f"{row['max_speedup_vs_sparta_5_8']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "## Geometric mean by projection",
            "",
            "| Projection | "
            + " | ".join(METHOD_LABELS[method] for method in METHODS)
            + " |",
            "|---|" + "---:|" * len(METHODS),
        ]
    )
    for projection in PROJECTIONS:
        values = [
            float(
                projection_index[(method, projection)][
                    "geomean_speedup_vs_sparta_5_8"
                ]
            )
            for method in METHODS
        ]
        lines.append(
            f"| {projection} | "
            + " | ".join(f"{value:.4f}x" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Geometric mean by model and projection",
            "",
            "Each entry is the geometric mean over M=512, 1024, and 2048.",
        ]
    )
    for model in MODELS:
        lines.extend(
            [
                "",
                f"### {MODEL_LABELS[model]}",
                "",
                "| Projection | "
                + " | ".join(METHOD_LABELS[method] for method in METHODS)
                + " |",
                "|---|" + "---:|" * len(METHODS),
            ]
        )
        for projection in PROJECTIONS:
            values = [
                float(
                    model_projection_index[(method, model, projection)][
                        "geomean_speedup_vs_sparta_5_8"
                    ]
                )
                for method in METHODS
            ]
            lines.append(
                f"| {projection} | "
                + " | ".join(f"{value:.4f}x" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `sparta_normalized.csv`: every shape and absolute latency.",
            "- `sparta_normalized_summary.csv`: aggregate geometric means.",
            "- `sparta_normalized_by_model_layer.csv`: model/projection "
            "geometric means over the three M values.",
            "- `pure24_upper_bound.csv`: formal pure-2:4 cuSPARSELt medians.",
            "- `figures/speedup_vs_sparta_5_8.png`: all model/layer/M panels.",
            "- `figures/geomean_vs_sparta_5_8.png`: overall comparison.",
            "- `figures/geomean_by_model_layer_vs_sparta_5_8.png`: "
            "per-model linear-layer comparison.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "examples/evaluate/eval-guidellm/results_final/"
            "five_model_nm_vs_speclink_kernel_20260723"
        ),
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    normalized = normalize(
        read_csv(root / "kernel_results.csv"),
        read_csv(root / "pure24_upper_bound.csv"),
    )
    overall = aggregate(normalized, ("method",))
    by_projection = aggregate(normalized, ("method", "projection"))
    by_model = aggregate(normalized, ("method", "model"))
    by_m = aggregate(normalized, ("method", "M"))
    by_model_projection = aggregate(
        normalized, ("method", "model", "projection")
    )
    write_csv(root / "sparta_normalized.csv", normalized)
    write_csv(
        root / "sparta_normalized_by_model_layer.csv",
        by_model_projection,
    )
    write_csv(
        root / "sparta_normalized_summary.csv",
        [
            {"group": "overall", **row} for row in overall
        ]
        + [
            {"group": "projection", **row} for row in by_projection
        ]
        + [{"group": "model", **row} for row in by_model]
        + [{"group": "M", **row} for row in by_m]
        + [
            {"group": "model_layer", **row}
            for row in by_model_projection
        ],
    )
    plot_matrix(
        normalized, root / "figures/speedup_vs_sparta_5_8.png"
    )
    plot_geomean(
        overall, root / "figures/geomean_vs_sparta_5_8.png"
    )
    plot_model_layer(
        by_model_projection,
        root / "figures/geomean_by_model_layer_vs_sparta_5_8.png",
    )
    write_report(
        root / "sparta_normalized_report.md",
        overall,
        by_projection,
        by_model_projection,
    )
    print(
        json.dumps(
            {
                row["method"]: row["geomean_speedup_vs_sparta_5_8"]
                for row in overall
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
