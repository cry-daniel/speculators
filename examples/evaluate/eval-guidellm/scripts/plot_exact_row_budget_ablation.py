#!/usr/bin/env python3
"""Plot exact dense-row budget ablations from sparse24 microbenchmarks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECTIONS = {
    (4096, 6144): ("qkv", "QKV"),
    (4096, 24576): ("gate_up", "Gate/up"),
    (12288, 4096): ("down", "Down"),
}
PROJECTION_ORDER = ("qkv", "gate_up", "down")
COLORS = {
    "qkv": "#2A9D8F",
    "gate_up": "#E9C46A",
    "down": "#E76F51",
    "weighted": "#374151",
}


def is_true(value: str | bool | None) -> bool:
    return str(value).strip().lower() == "true"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def projection(row: dict[str, str]) -> tuple[str, str] | None:
    return PROJECTIONS.get((int(row["K"]), int(row["N"])))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    pure: dict[str, dict[str, str]] = {}
    exact: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        current_projection = projection(row)
        if current_projection is None or not is_true(row.get("pass")):
            continue
        projection_name, _label = current_projection
        backend = row.get("backend", "")
        if backend == "device_sparse_gemm_view":
            existing = pure.get(projection_name)
            if existing is None or float(row["sparse_ms"]) < float(existing["sparse_ms"]):
                pure[projection_name] = row
            continue
        if (
            row.get("linear_strategy") != "full_sparse_residual"
            or "complement" not in backend
            or row.get("row_selection") != "random_sorted"
        ):
            continue
        dense_rows = int(float(row["dense_rows"]))
        key = (dense_rows, projection_name)
        existing = exact.get(key)
        if existing is None or float(row["sparse_ms"]) < float(existing["sparse_ms"]):
            exact[key] = row

    if set(pure) != set(PROJECTION_ORDER):
        raise ValueError("missing pure 2:4 projection rows")
    budgets = sorted({budget for budget, _projection in exact})
    output: list[dict[str, str | int | float]] = []
    for budget in (0, *budgets):
        dense_total = 0.0
        routed_total = 0.0
        for projection_name in PROJECTION_ORDER:
            source = pure[projection_name] if budget == 0 else exact[(budget, projection_name)]
            dense_ms = float(source["dense_ms"])
            routed_ms = float(source["sparse_ms"])
            dense_total += dense_ms
            routed_total += routed_ms
            output.append(
                {
                    "dense_rows": budget,
                    "projection": projection_name,
                    "dense_ms": dense_ms,
                    "routed_ms": routed_ms,
                    "speedup": dense_ms / routed_ms,
                    "backend": source["backend"],
                }
            )
        output.append(
            {
                "dense_rows": budget,
                "projection": "weighted",
                "dense_ms": dense_total,
                "routed_ms": routed_total,
                "speedup": dense_total / routed_total,
                "backend": "weighted_sum",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, str | int | float]], output_root: Path, title: str) -> None:
    lookup = {
        (int(row["dense_rows"]), str(row["projection"])): float(row["speedup"])
        for row in rows
    }
    budgets = sorted({budget for budget, _projection in lookup})
    fig, (axis, weighted_axis) = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        constrained_layout=True,
    )
    for projection_name in (*PROJECTION_ORDER, "weighted"):
        label = (
            "Weighted linears"
            if projection_name == "weighted"
            else PROJECTIONS[
                next(shape for shape, value in PROJECTIONS.items() if value[0] == projection_name)
            ][1]
        )
        axis.plot(
            budgets,
            [lookup[(budget, projection_name)] for budget in budgets],
            marker="o",
            linewidth=2.0,
            color=COLORS[projection_name],
            label=label,
        )
    axis.axhline(1.4, color="#B91C1C", linewidth=1.4, linestyle="--", label="Target 1.4x")
    axis.axhline(1.0, color="#6B7280", linewidth=1.0, linestyle=":")
    axis.set_title("Projection speedup")
    axis.set_xlabel("Exact dense rows per verify batch")
    axis.set_ylabel("Speedup vs dense GEMM")
    axis.set_xticks(budgets)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=8, ncol=2)

    weighted_axis.bar(
        [str(budget) for budget in budgets],
        [lookup[(budget, "weighted")] for budget in budgets],
        color=["#457B9D" if budget == 0 else "#374151" for budget in budgets],
    )
    weighted_axis.axhline(1.4, color="#B91C1C", linewidth=1.4, linestyle="--", label="Target 1.4x")
    weighted_axis.axhline(1.0, color="#6B7280", linewidth=1.0, linestyle=":")
    weighted_axis.set_title("Weighted verifier linears")
    weighted_axis.set_xlabel("Exact dense rows per verify batch")
    weighted_axis.set_ylabel("Speedup vs dense GEMM")
    weighted_axis.grid(axis="y", alpha=0.22)
    weighted_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "exact_row_budget_ablation.png", dpi=180)
    fig.savefig(figures / "exact_row_budget_ablation.svg")
    plt.close(fig)


def write_report(
    output_root: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    weighted = [row for row in rows if row["projection"] == "weighted"]
    lines = [
        "# Exact Row Budget Ablation",
        "",
        "CUDA-event projection microbenchmark for Qwen3-8B `bs=16`, `K=6`, "
        "with 112 verifier rows. `D=0` is whole-batch 2:4; `D>0` exactly "
        "reconstructs selected rows with the complementary 2:4 weights.",
        "",
        "| Exact dense rows | Weighted projection speedup |",
        "|---:|---:|",
    ]
    for row in weighted:
        lines.append(f"| {row['dense_rows']} | {float(row['speedup']):.3f}x |")
    lines.extend(
        [
            "",
            "Even one exact dense row requires streaming the complementary weight "
            "matrix. Gate/up and down are bandwidth-bound, so all nonzero budgets "
            "remain near dense speed. These are diagnostic projection timings, not "
            "end-to-end throughput claims.",
            "",
            "See `figures/exact_row_budget_ablation.png` and "
            "`exact_row_budget_ablation.csv`.",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Exact row-routing budget ablation (Qwen3-8B bs=16, K=6)",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = summarize(load_rows(args.input.expanduser().resolve()))
    write_csv(output_root / "exact_row_budget_ablation.csv", rows)
    plot(rows, output_root, args.title)
    write_report(output_root, rows)
    print(output_root)


if __name__ == "__main__":
    main()
