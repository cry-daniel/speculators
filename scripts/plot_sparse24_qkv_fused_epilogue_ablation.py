#!/usr/bin/env python3
"""Summarize fused sparse-QKV epilogue micro and system ablations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _acceptance_rate(measurement: dict[str, str]) -> float:
    summary_path = Path(measurement["run_dir"]) / "summary.csv"
    rows = [
        row
        for row in _read_csv(summary_path)
        if row.get("mode") == "token_dense_dynamic"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one token-dense row in {summary_path}")
    return float(rows[0]["spec_acceptance_rate"])


def _system_row(
    model: str,
    repeat: int,
    control_path: Path,
    candidate_path: Path,
) -> dict[str, object]:
    control = _measurement(control_path, model)
    candidate = _measurement(candidate_path, model)
    for field in ("model", "task", "batch_size", "k", "service_config_id"):
        if control[field] != candidate[field]:
            raise ValueError(
                f"unmatched {field} for {model} repeat {repeat}: "
                f"{control[field]} != {candidate[field]}"
            )
    control_tps = float(control["request_output_tokens_per_second"])
    candidate_tps = float(candidate["request_output_tokens_per_second"])
    control_acceptance = _acceptance_rate(control)
    candidate_acceptance = _acceptance_rate(candidate)
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
        "control_acceptance_rate": control_acceptance,
        "candidate_acceptance_rate": candidate_acceptance,
        "acceptance_delta_pp": 100.0
        * (candidate_acceptance - control_acceptance),
    }


def _micro_rows(paths: list[Path]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in paths:
        for row in _read_csv(path):
            # The service control used by this ablation keeps the generic
            # two-stage QKV post-op.  The persistent epilogue under test is
            # reported separately as ``single_launch_qkv_ms``.  The old plot
            # accidentally read ``paired_fused_qkv_ms`` as the candidate,
            # which is still a two-launch vec8 post-op path.
            current = float(row.get("paired_gather_residual_qkv_ms") or "nan")
            fused = float(row.get("single_launch_qkv_ms") or "nan")
            if current != current or fused != fused or fused <= 0:
                continue
            output.append(
                {
                    "model": row["model"],
                    "rows": int(row["rows"]),
                    "dense_rows": int(row["dense_rows"]),
                    "current_qkv_ms": current,
                    "fused_qkv_ms": fused,
                    "current_over_fused": current / fused,
                    "fused_max_abs_diff": float(
                        row.get("single_launch_qkv_max_abs_diff") or "nan"
                    ),
                }
            )
    return sorted(
        output,
        key=lambda row: (str(row["model"]), int(row["rows"])),
    )


def _plot(
    micro_rows: list[dict[str, object]],
    system_rows: list[dict[str, object]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    markers = {"qwen3_8b": "o", "llama3_1_8b": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    labels = [
        f"{row['model'].split('_')[0]}\nM{row['rows']}/D{row['dense_rows']}"
        for row in micro_rows
    ]
    values = [float(row["current_over_fused"]) for row in micro_rows]
    axes[0].bar(
        range(len(values)),
        values,
        color=[colors[str(row["model"])] for row in micro_rows],
        width=0.72,
    )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title("QKV + routed residual + norm/RoPE")
    axes[0].set_ylabel("Current two-stage / fused latency")
    axes[0].set_xticks(range(len(labels)), labels, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(
        min(0.98, min(values) - 0.01),
        max(1.05, max(values) + 0.01),
    )

    models = list(dict.fromkeys(str(row["model"]) for row in system_rows))
    for index, model in enumerate(models):
        selected = [row for row in system_rows if row["model"] == model]
        ratios = [float(row["candidate_over_control"]) for row in selected]
        offsets = [
            index + (i - (len(ratios) - 1) / 2) * 0.08
            for i in range(len(ratios))
        ]
        axes[1].scatter(
            offsets,
            ratios,
            color=colors[model],
            marker=markers[model],
            s=46,
        )
        median = statistics.median(ratios)
        axes[1].plot(
            [index - 0.22, index + 0.22],
            [median, median],
            color=colors[model],
            linewidth=2.2,
        )
        batch_size = int(selected[0]["batch_size"])
        k = int(selected[0]["k"])
        axes[1].text(
            index,
            max(ratios) + 0.012,
            f"bs{batch_size}/K{k}\nmedian {median:.3f}x",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_title("LM-eval matched-shape system A/B")
    axes[1].set_ylabel("Fused / control throughput")
    axes[1].set_xticks(range(len(models)), models)
    axes[1].grid(axis="y", alpha=0.25)
    ratios = [float(row["candidate_over_control"]) for row in system_rows]
    axes[1].set_ylim(min(0.80, min(ratios) - 0.025), max(1.08, max(ratios) + 0.04))

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-csv", type=Path, action="append", required=True)
    parser.add_argument(
        "--system",
        nargs=4,
        action="append",
        metavar=("MODEL", "REPEAT", "CONTROL_CSV", "CANDIDATE_CSV"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    micro_rows = _micro_rows(args.micro_csv)
    system_rows = [
        _system_row(model, int(repeat), Path(control), Path(candidate))
        for model, repeat, control, candidate in args.system
    ]
    summary_rows: list[dict[str, object]] = []
    for model in dict.fromkeys(str(row["model"]) for row in system_rows):
        selected = [row for row in system_rows if row["model"] == model]
        ratios = [float(row["candidate_over_control"]) for row in selected]
        summary_rows.append(
            {
                "model": model,
                "batch_size": int(selected[0]["batch_size"]),
                "k": int(selected[0]["k"]),
                "repeats": len(selected),
                "median_paired_ratio": statistics.median(ratios),
                "minimum_paired_ratio": min(ratios),
                "maximum_paired_ratio": max(ratios),
                "median_acceptance_delta_pp": statistics.median(
                    float(row["acceptance_delta_pp"]) for row in selected
                ),
            }
        )

    _write_csv(args.output_root / "micro_summary.csv", micro_rows)
    _write_csv(args.output_root / "system_ab.csv", system_rows)
    _write_csv(args.output_root / "system_ab_summary.csv", summary_rows)
    _plot(
        micro_rows,
        system_rows,
        args.output_root / "qkv_fused_epilogue_ablation.png",
    )
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
