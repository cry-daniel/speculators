#!/usr/bin/env python3
"""Aggregate and plot paired SpecLink system ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


COLORS = {
    "baseline": "#457B9D",
    "candidate": "#E76F51",
}


def parse_run(value: str) -> tuple[str, str, int, Path]:
    try:
        identity, path_text = value.split("=", 1)
        model, variant, repeat_text = identity.split(":", 2)
        repeat = int(repeat_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "run must be MODEL:VARIANT:REPEAT=PATH"
        ) from error
    return model, variant, repeat, Path(path_text).expanduser().resolve()


def find_summary(root: Path) -> dict[str, Any]:
    paths = sorted(root.rglob("summary.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one summary.json under {root}, found {len(paths)}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one summary row in {paths[0]}")
    return payload[0]


def collect(specs: list[tuple[str, str, int, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, variant, repeat, root in specs:
        summary = find_summary(root)
        actual_model = str(summary["model_label"])
        if actual_model != model:
            raise ValueError(f"model mismatch: requested {model}, found {actual_model}")
        rows.append(
            {
                "model": model,
                "variant": variant,
                "repeat": repeat,
                "score": float(summary["score"]),
                "elapsed_seconds": float(summary["request_elapsed_seconds"]),
                "output_tokens": int(summary["output_tokens"]),
                "throughput_tok_s": float(
                    summary["request_output_tokens_per_second"]
                ),
                "accepted_tokens": float(summary["spec_accepted_tokens"]),
                "draft_tokens": float(summary["spec_draft_tokens"]),
                "acceptance_rate": float(summary["spec_acceptance_rate"]),
                "summary_path": str(root),
            }
        )
    return rows


def paired_rows(
    rows: list[dict[str, Any]], baseline: str, candidate: str
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model"]), int(row["repeat"]), str(row["variant"])): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    identities = sorted({(str(row["model"]), int(row["repeat"])) for row in rows})
    for model, repeat in identities:
        base = lookup.get((model, repeat, baseline))
        tuned = lookup.get((model, repeat, candidate))
        if base is None or tuned is None:
            raise ValueError(f"missing pair for {model} repeat {repeat}")
        pairs.append(
            {
                "model": model,
                "repeat": repeat,
                "baseline_tok_s": base["throughput_tok_s"],
                "candidate_tok_s": tuned["throughput_tok_s"],
                "speedup": tuned["throughput_tok_s"] / base["throughput_tok_s"],
                "baseline_acceptance": base["acceptance_rate"],
                "candidate_acceptance": tuned["acceptance_rate"],
                "acceptance_delta_pp": 100.0
                * (tuned["acceptance_rate"] - base["acceptance_rate"]),
                "baseline_score": base["score"],
                "candidate_score": tuned["score"],
            }
        )
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    output_root: Path,
    baseline: str,
    candidate: str,
    title: str,
) -> None:
    models = sorted({str(row["model"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for model_index, model in enumerate(models):
        model_rows = [row for row in rows if row["model"] == model]
        for row in model_rows:
            variant_offset = -0.13 if row["variant"] == baseline else 0.13
            color = COLORS[
                "baseline" if row["variant"] == baseline else "candidate"
            ]
            axes[0].scatter(
                model_index + variant_offset,
                row["throughput_tok_s"],
                color=color,
                s=42,
                zorder=3,
            )
            axes[2].scatter(
                model_index + variant_offset,
                100.0 * row["acceptance_rate"],
                color=color,
                s=42,
                zorder=3,
            )
        for repeat in sorted({int(row["repeat"]) for row in model_rows}):
            base = next(
                row
                for row in model_rows
                if row["repeat"] == repeat and row["variant"] == baseline
            )
            tuned = next(
                row
                for row in model_rows
                if row["repeat"] == repeat and row["variant"] == candidate
            )
            axes[0].plot(
                [model_index - 0.13, model_index + 0.13],
                [base["throughput_tok_s"], tuned["throughput_tok_s"]],
                color="#9CA3AF",
                linewidth=1.0,
                zorder=1,
            )
            axes[2].plot(
                [model_index - 0.13, model_index + 0.13],
                [100.0 * base["acceptance_rate"], 100.0 * tuned["acceptance_rate"]],
                color="#9CA3AF",
                linewidth=1.0,
                zorder=1,
            )

    for model_index, model in enumerate(models):
        model_pairs = [row for row in pairs if row["model"] == model]
        values = [float(row["speedup"]) for row in model_pairs]
        axes[1].scatter(
            [model_index] * len(values), values, color="#2A9D8F", s=46, zorder=3
        )
        median_value = statistics.median(values)
        axes[1].plot(
            [model_index - 0.18, model_index + 0.18],
            [median_value, median_value],
            color="#111827",
            linewidth=2.2,
        )
        axes[1].text(
            model_index,
            median_value,
            f" {median_value:.3f}x",
            va="bottom",
            ha="center",
            fontsize=9,
        )

    axes[0].set_title("End-to-end throughput")
    axes[0].set_ylabel("Output tokens / second")
    axes[1].set_title(f"Paired {candidate} / {baseline}")
    axes[1].set_ylabel("Throughput ratio")
    axes[1].axhline(1.0, color="#6B7280", linestyle="--", linewidth=1.1)
    axes[2].set_title("EAGLE3 acceptance")
    axes[2].set_ylabel("Accepted draft tokens (%)")
    for axis in axes:
        axis.set_xticks(range(len(models)), models)
        axis.grid(axis="y", alpha=0.22)

    baseline_handle = plt.Line2D(
        [0], [0], marker="o", linestyle="", color=COLORS["baseline"], label=baseline
    )
    candidate_handle = plt.Line2D(
        [0], [0], marker="o", linestyle="", color=COLORS["candidate"], label=candidate
    )
    axes[0].legend(handles=[baseline_handle, candidate_handle], frameon=False)
    axes[2].legend(handles=[baseline_handle, candidate_handle], frameon=False)
    fig.suptitle(title)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "paired_system_ablation.png", dpi=180, facecolor="white")
    fig.savefig(figures / "paired_system_ablation.svg", facecolor="white")
    plt.close(fig)


def write_report(
    output_root: Path,
    pairs: list[dict[str, Any]],
    baseline: str,
    candidate: str,
) -> None:
    lines = [
        "# Paired System Ablation",
        "",
        f"Candidate `{candidate}` compared with `{baseline}` using matched output-token counts.",
        "",
        "| Model | Repeat | Throughput ratio | Acceptance delta | Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['model']} | {row['repeat']} | {row['speedup']:.3f}x | "
            f"{row['acceptance_delta_pp']:+.2f}pp | "
            f"{row['baseline_score']:.4f} -> {row['candidate_score']:.4f} |"
        )
    lines.extend(["", "Per-model median throughput ratios:", ""])
    for model in sorted({str(row["model"]) for row in pairs}):
        values = [float(row["speedup"]) for row in pairs if row["model"] == model]
        lines.append(f"- `{model}`: `{statistics.median(values):.3f}x`")
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--baseline", default="control")
    parser.add_argument("--candidate", default="tuned")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--title", default="SpecLink paired system ablation")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = collect(args.run)
    pairs = paired_rows(rows, args.baseline, args.candidate)
    write_csv(output_root / "runs.csv", rows)
    write_csv(output_root / "paired.csv", pairs)
    plot(rows, pairs, output_root, args.baseline, args.candidate, args.title)
    write_report(output_root, pairs, args.baseline, args.candidate)
    print(output_root)


if __name__ == "__main__":
    main()
