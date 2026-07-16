#!/usr/bin/env python3
"""Plot exact dense-row routing feasibility from CUTLASS microbenchmarks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXACT_STRATEGIES = {
    "full_sparse_residual",
    "full_sparse_dense_override",
    "split_dense_sparse",
}
PROJECTION_ORDER = ("qkv", "gate_up", "down")
PROJECTION_LABELS = {
    "qkv": "QKV",
    "gate_up": "Gate/up",
    "down": "Down",
}


@dataclass(frozen=True)
class ProjectionResult:
    projection: str
    rows: int
    dense_rows: int
    strategy: str
    backend: str
    dense_ms: float
    exact_ms: float
    pure24_ms: float

    @property
    def exact_speedup(self) -> float:
        return self.dense_ms / self.exact_ms

    @property
    def pure24_speedup(self) -> float:
        return self.dense_ms / self.pure24_ms


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be LABEL=BENCHMARK_ROOT")
    label, path = value.split("=", 1)
    label = label.strip()
    root = Path(path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("case label cannot be empty")
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"benchmark root does not exist: {root}")
    return label, root


def projection_name(row: dict[str, str]) -> str | None:
    k = int(row["K"])
    n = int(row["N"])
    if (k, n) == (4096, 6144):
        return "qkv"
    if k == 4096 and n > 6144:
        return "gate_up"
    if n == 4096 and k > 4096:
        return "down"
    return None


def display_case_label(label: str) -> str:
    parts = label.split("_")
    if len(parts) <= 2:
        return label
    return f"{'_'.join(parts[:2])}\n{' '.join(parts[2:])}"


def load_case(root: Path) -> dict[str, ProjectionResult]:
    candidates = [
        root / "selective_dense_best.csv",
        root / "selective_dense_e2e_best.csv",
        root / "selective_dense_random_e2e_best.csv",
        root / "summary.csv",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"no selective-dense CSV found under {root}")

    best: dict[str, ProjectionResult] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            projection = projection_name(row)
            if projection is None:
                continue
            strategy = row.get("linear_strategy", "")
            if strategy not in EXACT_STRATEGIES:
                continue
            if row.get("pass", "").strip().lower() not in {"true", "1"}:
                continue
            try:
                result = ProjectionResult(
                    projection=projection,
                    rows=int(row["M"]),
                    dense_rows=int(row["dense_rows"]),
                    strategy=strategy,
                    backend=row["backend"],
                    dense_ms=float(row["dense_ms"]),
                    exact_ms=float(row["sparse_ms"]),
                    pure24_ms=float(row["pure24_ms"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            current = best.get(projection)
            if current is None or result.exact_ms < current.exact_ms:
                best[projection] = result

    missing = set(PROJECTION_ORDER).difference(best)
    if missing:
        raise RuntimeError(f"{source} is missing projections: {sorted(missing)}")
    return best


def write_summary(
    output_root: Path,
    cases: list[tuple[str, dict[str, ProjectionResult]]],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for label, projections in cases:
        dense_total = sum(projections[name].dense_ms for name in PROJECTION_ORDER)
        exact_total = sum(projections[name].exact_ms for name in PROJECTION_ORDER)
        pure_total = sum(projections[name].pure24_ms for name in PROJECTION_ORDER)
        for name in PROJECTION_ORDER:
            result = projections[name]
            rows.append(
                {
                    "case": label,
                    "projection": name,
                    "rows": result.rows,
                    "dense_rows": result.dense_rows,
                    "strategy": result.strategy,
                    "backend": result.backend,
                    "dense_ms": result.dense_ms,
                    "exact_ms": result.exact_ms,
                    "pure24_ms": result.pure24_ms,
                    "exact_speedup": result.exact_speedup,
                    "pure24_speedup": result.pure24_speedup,
                }
            )
        rows.append(
            {
                "case": label,
                "projection": "weighted_all",
                "rows": projections["qkv"].rows,
                "dense_rows": projections["qkv"].dense_rows,
                "strategy": "best_exact_per_projection",
                "backend": "weighted_sum",
                "dense_ms": dense_total,
                "exact_ms": exact_total,
                "pure24_ms": pure_total,
                "exact_speedup": dense_total / exact_total,
                "pure24_speedup": dense_total / pure_total,
            }
        )

    csv_path = output_root / "exact_row_routing_feasibility.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_summary(
    output_root: Path,
    labels: list[str],
    rows: list[dict[str, str | int | float]],
    target: float,
) -> None:
    lookup = {
        (str(row["case"]), str(row["projection"])): row for row in rows
    }
    x = np.arange(len(labels), dtype=np.float64)
    colors = {
        "qkv": "#2A9D8F",
        "gate_up": "#E9C46A",
        "down": "#E76F51",
        "weighted_all": "#374151",
    }

    display_labels = [display_case_label(label) for label in labels]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12.5, 2.2 * len(labels)), 4.8),
        constrained_layout=True,
    )
    width = 0.19
    for offset, projection in enumerate((*PROJECTION_ORDER, "weighted_all")):
        values = [
            float(lookup[(label, projection)]["exact_speedup"])
            for label in labels
        ]
        axes[0].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            color=colors[projection],
            label=(
                "Weighted linears"
                if projection == "weighted_all"
                else PROJECTION_LABELS[projection]
            ),
        )
    target_label = f"End-to-end target {target:.1f}x"
    axes[0].axhline(target, color="#B91C1C", linewidth=1.5, label=target_label)
    axes[0].axhline(1.0, color="#6B7280", linewidth=1.0, linestyle=":")
    axes[0].set_title("Exact row-mixed speedup")
    axes[0].set_ylabel("Speedup vs dense")
    axes[0].set_xticks(x, display_labels)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    exact = [
        float(lookup[(label, "weighted_all")]["exact_speedup"])
        for label in labels
    ]
    pure = [
        float(lookup[(label, "weighted_all")]["pure24_speedup"])
        for label in labels
    ]
    axes[1].bar(x - width / 2, exact, width, color="#374151", label="Exact row-mixed")
    axes[1].bar(x + width / 2, pure, width, color="#457B9D", label="Whole-batch 2:4 upper bound")
    axes[1].axhline(target, color="#B91C1C", linewidth=1.5, label=target_label)
    axes[1].axhline(1.0, color="#6B7280", linewidth=1.0, linestyle=":")
    axes[1].set_title("Weighted verifier linears")
    axes[1].set_ylabel("Speedup vs dense")
    axes[1].set_xticks(x, display_labels)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "exact_row_routing_feasibility.png", dpi=180)
    fig.savefig(figures / "exact_row_routing_feasibility.svg")
    plt.close(fig)


def write_report(
    output_root: Path,
    labels: list[str],
    rows: list[dict[str, str | int | float]],
    target: float,
) -> None:
    lookup = {
        (str(row["case"]), str(row["projection"])): row for row in rows
    }
    lines = [
        "# Exact Row-Routing Feasibility",
        "",
        "These are CUDA-event microbenchmarks for exact within-batch routing. "
        "Selected rows reconstruct dense weights; other rows use one 2:4 mask.",
        "",
        "| Case | Rows | Dense rows | QKV | Gate/up | Down | Weighted exact | Weighted pure 2:4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        qkv = lookup[(label, "qkv")]
        gate = lookup[(label, "gate_up")]
        down = lookup[(label, "down")]
        weighted = lookup[(label, "weighted_all")]
        lines.append(
            f"| {label} | {qkv['rows']} | {qkv['dense_rows']} | "
            f"{float(qkv['exact_speedup']):.3f}x | "
            f"{float(gate['exact_speedup']):.3f}x | "
            f"{float(down['exact_speedup']):.3f}x | "
            f"{float(weighted['exact_speedup']):.3f}x | "
            f"{float(weighted['pure24_speedup']):.3f}x |"
        )
    lines.extend(
        [
            "",
            "At small row counts, exact routing with any dense rows streams both "
            "the primary 2:4 weights and their complementary residual weights. "
            "Their FP16 value payload is approximately one dense weight matrix, "
            "plus sparse metadata, so bandwidth-bound gate/up and down projections "
            "cannot retain the whole-batch 2:4 speedup through row-count tiling alone.",
            "",
            f"The reference target is {target:.1f}x end-to-end. These values cover "
            "only QKV, gate/up, and down projections, so they are diagnostic rather "
            "than end-to-end claims. The whole-batch 2:4 bars are retained only as "
            "an upper-bound comparison and are not token-mixed SpecLink results.",
            "",
            "See `figures/exact_row_routing_feasibility.png` and "
            "`exact_row_routing_feasibility.csv`.",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-speedup", type=float, default=1.4)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [(label, load_case(root)) for label, root in args.case]
    rows = write_summary(output_root, cases)
    labels = [label for label, _ in cases]
    plot_summary(output_root, labels, rows, args.target_speedup)
    write_report(output_root, labels, rows, args.target_speedup)
    print(output_root)


if __name__ == "__main__":
    main()
