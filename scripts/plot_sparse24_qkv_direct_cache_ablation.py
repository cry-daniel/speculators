#!/usr/bin/env python3
"""Plot the QKV direct-cache microbenchmark and system A/B ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _system_rows(
    model: str, control_path: Path, candidate_path: Path
) -> list[dict[str, str | int | float]]:
    control = {
        int(row["repeat"]): row
        for row in _read_csv(control_path)
        if row.get("status") == "ok"
    }
    candidate = {
        int(row["repeat"]): row
        for row in _read_csv(candidate_path)
        if row.get("status") == "ok"
    }
    repeats = sorted(control.keys() & candidate.keys())
    if not repeats:
        raise ValueError(f"no matched successful repeats for {model}")
    output: list[dict[str, str | int | float]] = []
    for repeat in repeats:
        control_row = control[repeat]
        candidate_row = candidate[repeat]
        control_tps = float(control_row["request_output_tokens_per_second"])
        candidate_tps = float(candidate_row["request_output_tokens_per_second"])
        output.append(
            {
                "model": model,
                "repeat": repeat,
                "control_tps": control_tps,
                "candidate_tps": candidate_tps,
                "candidate_over_control": candidate_tps / control_tps,
                "control_score": float(control_row["score"]),
                "candidate_score": float(candidate_row["score"]),
                "control_output_tokens": int(control_row["output_tokens"]),
                "candidate_output_tokens": int(candidate_row["output_tokens"]),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    micro_rows: list[dict[str, str]],
    system_rows: list[dict[str, str | int | float]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in system_rows))
    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

    for model in models:
        selected = [row for row in micro_rows if row["model"] == model]
        axes[0].plot(
            [int(row["rows"]) for row in selected],
            [float(row["speedup"]) for row in selected],
            marker="o",
            color=colors.get(model),
            label=model,
        )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title("QKV post-op + KV-cache microbenchmark")
    axes[0].set_xlabel("Verifier rows")
    axes[0].set_ylabel("Separate / direct-cache latency")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for index, model in enumerate(models):
        selected = [row for row in system_rows if row["model"] == model]
        control = [float(row["control_tps"]) for row in selected]
        candidate = [float(row["candidate_tps"]) for row in selected]
        axes[1].scatter(
            [index - 0.12] * len(control),
            control,
            marker="o",
            color="#666666",
            label="separate cache update" if index == 0 else None,
        )
        axes[1].scatter(
            [index + 0.12] * len(candidate),
            candidate,
            marker="s",
            color=colors.get(model),
            label="direct-cache" if index == 0 else None,
        )
        control_median = statistics.median(control)
        candidate_median = statistics.median(candidate)
        axes[1].plot(
            [index - 0.2, index - 0.04],
            [control_median, control_median],
            color="#333333",
            linewidth=2,
        )
        axes[1].plot(
            [index + 0.04, index + 0.2],
            [candidate_median, candidate_median],
            color=colors.get(model),
            linewidth=2,
        )
        axes[1].text(
            index,
            max(control + candidate) * 1.015,
            f"{candidate_median / control_median:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_title("LM-eval system A/B, bs32 K8")
    axes[1].set_xticks(range(len(models)), models)
    axes[1].set_ylabel("Client-observed output tokens/s")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.04)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-csv", type=Path, required=True)
    parser.add_argument(
        "--system",
        nargs=3,
        action="append",
        metavar=("MODEL", "CONTROL_CSV", "CANDIDATE_CSV"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    micro_rows = _read_csv(args.micro_csv)
    system_rows: list[dict[str, str | int | float]] = []
    for model, control_path, candidate_path in args.system:
        system_rows.extend(
            _system_rows(model, Path(control_path), Path(candidate_path))
        )
    summary_rows: list[dict[str, object]] = []
    for model in dict.fromkeys(str(row["model"]) for row in system_rows):
        selected = [row for row in system_rows if row["model"] == model]
        control = [float(row["control_tps"]) for row in selected]
        candidate = [float(row["candidate_tps"]) for row in selected]
        ratios = [float(row["candidate_over_control"]) for row in selected]
        summary_rows.append(
            {
                "model": model,
                "repeats": len(selected),
                "control_median_tps": statistics.median(control),
                "candidate_median_tps": statistics.median(candidate),
                "median_tps_ratio": statistics.median(candidate)
                / statistics.median(control),
                "median_paired_ratio": statistics.median(ratios),
                "minimum_paired_ratio": min(ratios),
            }
        )
    _write_csv(args.output_root / "system_ablation.csv", system_rows)
    _write_csv(args.output_root / "system_ablation_summary.csv", summary_rows)
    _plot(
        micro_rows,
        system_rows,
        args.output_root / "qkv_direct_cache_ablation.png",
    )
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
