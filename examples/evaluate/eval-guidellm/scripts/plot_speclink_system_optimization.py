#!/usr/bin/env python3
"""Plot the strict quality/throughput summary for SpecLink candidates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_frontier(rows: list[dict[str, str]], output_dir: Path, title: str) -> None:
    labels = [
        row.get("label") or f"{row['model']}\n{row['task']}"
        for row in rows
    ]
    speedups = [float(row["performance_speedup"]) for row in rows]
    deltas = [float(row["quality_delta_pp"]) for row in rows]
    colors = [
        "#2a9d8f" if row["meets_target"] == "True" else "#e76f51"
        for row in rows
    ]

    fig, (speed_ax, quality_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 5.8),
        sharey=True,
    )
    positions = list(range(len(rows)))
    speed_ax.barh(positions, speedups, color=colors, height=0.68)
    speed_ax.axvline(
        1.4,
        color="#264653",
        linestyle="--",
        linewidth=1.4,
        label="target 1.4x",
    )
    speed_ax.axvline(
        1.3,
        color="#6c757d",
        linestyle=":",
        linewidth=1.2,
        label="hard floor 1.3x",
    )
    speed_ax.set_xlabel("Throughput speedup vs dense EAGLE3")
    speed_ax.set_yticks(positions, labels)
    speed_ax.set_xlim(
        min(0.8, min(speedups) - 0.05),
        max(1.65, max(speedups) + 0.08),
    )
    speed_ax.invert_yaxis()
    speed_ax.legend(frameon=False, loc="lower right")
    speed_ax.grid(axis="x", alpha=0.2)

    quality_ax.barh(positions, deltas, color=colors, height=0.68)
    quality_ax.axvline(
        -5.0,
        color="#264653",
        linestyle="--",
        linewidth=1.4,
        label="quality floor -5pp",
    )
    quality_ax.axvline(0.0, color="#6c757d", linewidth=0.8)
    quality_ax.set_xlabel("Absolute accuracy delta (pp)")
    quality_ax.set_xlim(min(-12.5, min(deltas) - 1.0), 2.0)
    quality_ax.legend(frameon=False, loc="lower left")
    quality_ax.grid(axis="x", alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"quality_throughput_frontier.{suffix}",
            dpi=180,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_throughput_matrix(
    rows: list[dict[str, str]], output_dir: Path, title: str
) -> None:
    performance = [
        row
        for row in rows
        if row.get("phase") == "performance" and row.get("speedup_median")
    ]
    if not performance:
        return
    facets = sorted({(row["model"], row["task"]) for row in performance})
    columns = min(2, len(facets))
    figure_rows = math.ceil(len(facets) / columns)
    fig, axes = plt.subplots(
        figure_rows,
        columns,
        figsize=(6.2 * columns, 4.7 * figure_rows),
        squeeze=False,
    )
    image = None
    for axis, (model, task) in zip(axes.flat, facets):
        facet_rows = [
            row
            for row in performance
            if row["model"] == model and row["task"] == task
        ]
        batch_sizes = sorted({int(row["batch_size"]) for row in facet_rows})
        k_values = sorted({int(row["k"]) for row in facet_rows})
        values = np.full((len(batch_sizes), len(k_values)), np.nan)
        minimums = np.full_like(values, np.nan)
        repeats = np.zeros_like(values, dtype=np.int32)
        batch_index = {value: index for index, value in enumerate(batch_sizes)}
        k_index = {value: index for index, value in enumerate(k_values)}
        for row in facet_rows:
            y = batch_index[int(row["batch_size"])]
            x = k_index[int(row["k"])]
            values[y, x] = float(row["speedup_median"])
            if row.get("speedup_min"):
                minimums[y, x] = float(row["speedup_min"])
            repeats[y, x] = int(row.get("repeat_count") or 0)
        image = axis.imshow(values, cmap="RdYlGn", vmin=1.2, vmax=1.7, aspect="auto")
        for y in range(len(batch_sizes)):
            for x in range(len(k_values)):
                value = values[y, x]
                if np.isnan(value):
                    continue
                minimum = minimums[y, x]
                detail = f"{value:.3f}x"
                if repeats[y, x] > 1 and not np.isnan(minimum):
                    detail += f"\nmin {minimum:.3f}x"
                axis.text(
                    x,
                    y,
                    detail,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        axis.set_xticks(range(len(k_values)), [str(value) for value in k_values])
        axis.set_yticks(
            range(len(batch_sizes)), [str(value) for value in batch_sizes]
        )
        axis.set_xlabel("Speculative K")
        axis.set_ylabel("Batch size / concurrency")
        axis.set_title(f"{model} | {task}")
    for axis in axes.flat[len(facets) :]:
        axis.set_visible(False)
    if image is not None:
        color_axis = fig.add_axes((0.92, 0.16, 0.018, 0.68))
        fig.colorbar(image, cax=color_axis, label="Speedup vs dense EAGLE3")
    fig.suptitle(title)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.09, right=0.88, wspace=0.28)
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"throughput_matrix.{suffix}",
            dpi=180,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_accuracy(rows: list[dict[str, str]], output_dir: Path) -> None:
    accuracy = [
        row
        for row in rows
        if row.get("phase") == "accuracy" and row.get("accuracy_delta_pp")
    ]
    if not accuracy:
        return
    accuracy.sort(
        key=lambda row: (
            row["model"],
            row["task"],
            int(row["batch_size"]),
            int(row["k"]),
        )
    )
    labels = [
        f"{row['model']} | {row['task']} | bs={row['batch_size']} | K={row['k']}"
        for row in accuracy
    ]
    deltas = [float(row["accuracy_delta_pp"]) for row in accuracy]
    colors = ["#2a9d8f" if value >= -5.0 else "#e76f51" for value in deltas]
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.45 * len(accuracy) + 1.8)))
    positions = list(range(len(accuracy)))
    axis.barh(positions, deltas, color=colors, height=0.68)
    axis.axvline(-5.0, color="#264653", linestyle="--", label="floor -5pp")
    axis.axvline(0.0, color="#6c757d", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Absolute accuracy delta (percentage points)")
    axis.set_title("Matched-concurrency LM-eval accuracy vs same-K dense EAGLE3")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"accuracy_delta.{suffix}",
            dpi=180,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="SpecLink strict same-K system matrix",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise ValueError(f"no rows found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "speedup_median" in rows[0]:
        plot_throughput_matrix(rows, args.output_dir, args.title)
        plot_accuracy(rows, args.output_dir)
    elif {"performance_speedup", "quality_delta_pp"}.issubset(rows[0]):
        plot_frontier(rows, args.output_dir, args.title)
    else:
        raise ValueError(f"unsupported input columns: {sorted(rows[0])}")


if __name__ == "__main__":
    main()
