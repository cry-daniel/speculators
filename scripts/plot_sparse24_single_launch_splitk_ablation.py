#!/usr/bin/env python3
"""Summarize and plot single-launch sparse split-K Down ablations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _measurement(path: Path, model: str) -> dict[str, str]:
    rows = [
        row
        for row in _read_csv(path)
        if row.get("status") == "ok" and row.get("model") == model
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one successful {model} measurement in {path}"
        )
    return rows[0]


def _system_row(
    model: str,
    repeat: int,
    control_path: Path,
    candidate_path: Path,
) -> dict[str, str | int | float]:
    control = _measurement(control_path, model)
    candidate = _measurement(candidate_path, model)
    for field in ("model", "task", "batch_size", "k"):
        if control[field] != candidate[field]:
            raise ValueError(
                f"unmatched {field} for repeat {repeat}: "
                f"{control[field]} != {candidate[field]}"
            )
    if control["model"] != model:
        raise ValueError(f"expected model {model}, got {control['model']}")
    control_tps = float(control["request_output_tokens_per_second"])
    candidate_tps = float(candidate["request_output_tokens_per_second"])
    return {
        "model": model,
        "repeat": repeat,
        "task": control["task"],
        "batch_size": int(control["batch_size"]),
        "k": int(control["k"]),
        "control_tps": control_tps,
        "candidate_tps": candidate_tps,
        "candidate_over_control": candidate_tps / control_tps,
        "control_score": float(control["score"]),
        "candidate_score": float(candidate["score"]),
        "score_delta_pp": 100.0
        * (float(candidate["score"]) - float(control["score"])),
        "control_output_tokens": int(control["output_tokens"]),
        "candidate_output_tokens": int(candidate["output_tokens"]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _best_micro_rows(
    rows: list[dict[str, str]],
    speedup_column: str,
) -> dict[str, list[dict[str, str]]]:
    best: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        key = (row["model"], int(row["batch_size"]), int(row["K"]))
        previous = best.get(key)
        if previous is None or float(row[speedup_column]) > float(
            previous[speedup_column]
        ):
            best[key] = row
    grouped: dict[str, list[dict[str, str]]] = {}
    for key in sorted(best, key=lambda item: (item[0], item[1], item[2])):
        grouped.setdefault(key[0], []).append(best[key])
    return grouped


def _plot(
    micro_rows: list[dict[str, str]],
    system_rows: list[dict[str, str | int | float]],
    output: Path,
    *,
    speedup_column: str,
    micro_title: str,
    candidate_label: str,
) -> None:
    import matplotlib.pyplot as plt

    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    markers = {"qwen3_8b": "o", "llama3_1_8b": "s"}
    models = list(dict.fromkeys(str(row["model"]) for row in system_rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    grouped_micro = _best_micro_rows(micro_rows, speedup_column)
    labels = [
        f"{batch_size}/{k}"
        for batch_size in (16, 32, 64)
        for k in (6, 8, 10)
    ]
    for model in models:
        selected = grouped_micro[model]
        by_shape = {
            (int(row["batch_size"]), int(row["K"])): row
            for row in selected
        }
        values = [
            float(by_shape[(batch_size, k)][speedup_column])
            for batch_size in (16, 32, 64)
            for k in (6, 8, 10)
        ]
        axes[0].plot(
            range(len(labels)),
            values,
            color=colors[model],
            marker=markers[model],
            linewidth=1.6,
            label=model,
        )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title(micro_title)
    axes[0].set_xlabel("Batch size / K")
    axes[0].set_ylabel("Standard / split-K latency")
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for index, model in enumerate(models):
        selected = [row for row in system_rows if row["model"] == model]
        ratios = [float(row["candidate_over_control"]) for row in selected]
        offsets = [
            index + (repeat_index - (len(ratios) - 1) / 2) * 0.08
            for repeat_index in range(len(ratios))
        ]
        axes[1].scatter(
            offsets,
            ratios,
            color=colors[model],
            marker=markers[model],
            s=42,
        )
        median_ratio = statistics.median(ratios)
        axes[1].plot(
            [index - 0.2, index + 0.2],
            [median_ratio, median_ratio],
            color=colors[model],
            linewidth=2.2,
        )
        axes[1].text(
            index,
            max(ratios) + 0.008,
            f"median {median_ratio:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_title("LM-eval system A/B, bs32 K8")
    axes[1].set_xticks(range(len(models)), models)
    axes[1].set_ylabel(f"{candidate_label} / control throughput")
    axes[1].grid(axis="y", alpha=0.25)
    all_ratios = [
        float(row["candidate_over_control"]) for row in system_rows
    ]
    axes[1].set_ylim(
        bottom=min(0.97, min(all_ratios) - 0.025),
        top=max(1.04, max(all_ratios) + 0.03),
    )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-csv", type=Path, required=True)
    parser.add_argument(
        "--system",
        nargs=4,
        action="append",
        metavar=("MODEL", "REPEAT", "CONTROL_CSV", "CANDIDATE_CSV"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--micro-speedup-column",
        default="single_launch_exact_down_speedup",
    )
    parser.add_argument(
        "--micro-title",
        default="Best serial split-K sparse Down kernel",
    )
    parser.add_argument("--candidate-label", default="Single-launch")
    parser.add_argument(
        "--plot-name",
        default="single_launch_splitk_ablation.png",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    system_rows = [
        _system_row(model, int(repeat), Path(control), Path(candidate))
        for model, repeat, control, candidate in args.system
    ]
    summary_rows: list[dict[str, object]] = []
    for model in dict.fromkeys(str(row["model"]) for row in system_rows):
        selected = [row for row in system_rows if row["model"] == model]
        ratios = [float(row["candidate_over_control"]) for row in selected]
        score_deltas = [float(row["score_delta_pp"]) for row in selected]
        summary_rows.append(
            {
                "model": model,
                "repeats": len(selected),
                "median_paired_ratio": statistics.median(ratios),
                "minimum_paired_ratio": min(ratios),
                "maximum_paired_ratio": max(ratios),
                "median_score_delta_pp": statistics.median(score_deltas),
            }
        )

    _write_csv(args.output_root / "system_ab.csv", system_rows)
    _write_csv(args.output_root / "system_ab_summary.csv", summary_rows)
    _plot(
        _read_csv(args.micro_csv),
        system_rows,
        args.output_root / args.plot_name,
        speedup_column=args.micro_speedup_column,
        micro_title=args.micro_title,
        candidate_label=args.candidate_label,
    )
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
