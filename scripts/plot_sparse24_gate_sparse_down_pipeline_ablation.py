#!/usr/bin/env python3
"""Compare whole-batch sparse Gate/SwiGLU -> sparse Down pipelines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    configs = tuple(dict.fromkeys(str(row["config"]) for row in rows))
    colors = {
        "128x64_gate_128x64_sparse_down": "#176B87",
        "256x32_gate_256x32_sparse_down": "#B33F40",
    }
    markers = {"qwen3_8b": "o", "llama3_1_8b": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    for model in models:
        for config in configs:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == model and row["config"] == config
                ),
                key=lambda row: int(row["rows"]),
            )
            if not selected:
                continue
            short_config = str(config).split("_gate_")[0]
            label = f"{model} {short_config}"
            x = [int(row["rows"]) for row in selected]
            axes[0].plot(
                x,
                [float(row["pipeline_gate_sparse_down_ms"]) for row in selected],
                marker=markers[model],
                color=colors.get(config, "#555555"),
                linestyle="-" if model == "qwen3_8b" else "--",
                label=label,
            )
            axes[1].plot(
                x,
                [float(row["pipeline_speedup"]) for row in selected],
                marker=markers[model],
                color=colors.get(config, "#555555"),
                linestyle="-" if model == "qwen3_8b" else "--",
                label=label,
            )

    axes[0].set_title("Whole-batch W24 pipeline latency")
    axes[0].set_xlabel("Verifier rows")
    axes[0].set_ylabel("CUDA Graph latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_title("Single launch vs same-tile two launches")
    axes[1].set_xlabel("Verifier rows")
    axes[1].set_ylabel("Separate / pipeline")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.suptitle("Kernel upper bound only; not token-mixed SpecLink", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Sparse MLP pipeline configuration ablation",
        "",
        "This is a whole-batch W24 kernel upper bound. It is not a final "
        "token-mixed SpecLink result.",
        "",
        "| Model | Rows | Config | Pipeline (ms) | Launch speedup | Correct |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        correct = (
            float(row["max_abs_diff"]) == 0.0
            and int(row["counter_max_after_graph"]) == 0
        )
        lines.append(
            f"| {row['model']} | {row['rows']} | {row['config']} | "
            f"{float(row['pipeline_gate_sparse_down_ms']):.4f} | "
            f"{float(row['pipeline_speedup']):.3f}x | "
            f"{'yes' if correct else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in args.input_csv:
        rows.extend(read_rows(path))
    if not rows:
        raise ValueError("no input rows")
    rows.sort(key=lambda row: (str(row["model"]), int(row["rows"]), str(row["config"])))
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "gate_sparse_down_pipeline_ablation.csv", rows)
    write_plot(args.output_root / "gate_sparse_down_pipeline_ablation.png", rows)
    write_report(args.output_root / "report.md", rows)
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
