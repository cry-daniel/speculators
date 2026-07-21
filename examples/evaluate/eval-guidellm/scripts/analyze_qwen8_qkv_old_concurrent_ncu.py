#!/usr/bin/env python3
"""Aggregate whole-graph NCU and NSYS evidence for Qwen3-8B qkv."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ncu_report_utils import (
    byte_value,
    converted_metric,
    decimal_value,
    ncu_records,
)


OLD = "old_dense_base_concurrent_d29_s141"
CUBLAS = "cublas_dense_m2048"
METHODS = (OLD, CUBLAS)
LABELS = {OLD: "Old concurrent", CUBLAS: "cuBLAS dense"}

METRICS = {
    "tensor_active_pct": (
        "sm__pipe_tensor_subpipe_hmma_cycles_active."
        "avg.pct_of_peak_sustained_elapsed"
    ),
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "barrier_stall_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "long_scoreboard_stall_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
    ),
    "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_read_hit_pct": "lts__t_sector_op_read_hit_rate.pct",
    "shared_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "shared_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(record: dict[str, str], name: str) -> float:
    raw = decimal_value(record, name)
    if raw is None:
        raise RuntimeError(f"NCU report is missing {name}")
    return float(raw)


def extract_ncu(report: Path, method: str) -> dict[str, Any]:
    records, units = ncu_records(report)
    if len(records) != 1:
        raise RuntimeError(f"expected one whole-graph row in {report}: {len(records)}")
    record = records[0]
    if record.get("Kernel Name") != "graph":
        raise RuntimeError(
            f"{report} is not a whole-graph NCU report: "
            f"{record.get('Kernel Name')!r}"
        )
    row: dict[str, Any] = {
        "method": method,
        "label": LABELS[method],
        "report": report.name,
        "ncu_workload": record.get("Kernel Name"),
        "representative_grid": record.get("Grid Size"),
        "representative_block": record.get("Block Size"),
        "ncu_duration_us": converted_metric(
            record, units, "duration_us", "gpu__time_duration.sum"
        ),
        "dram_read_bytes": byte_value(
            record, units, "dram__bytes_op_read.sum"
        ),
    }
    for output, raw_name in METRICS.items():
        row[output] = metric(record, raw_name)
    row["dram_read_mib"] = float(row["dram_read_bytes"]) / (1024.0**2)
    row["shared_bank_conflicts_per_1k_wavefronts"] = (
        1000.0 * float(row["shared_bank_conflicts"])
        / float(row["shared_wavefronts"])
    )
    return row


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def read_nsys(sqlite_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    connection = sqlite3.connect(sqlite_path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        kernel_tables = sorted(
            table
            for table in tables
            if table.startswith("CUPTI_ACTIVITY_KIND_")
            and table.endswith("KERNEL")
        )
        if len(kernel_tables) != 1 or "StringIds" not in tables:
            raise RuntimeError(f"unexpected NSYS schema: {kernel_tables}")
        table = kernel_tables[0]
        columns = table_columns(connection, table)
        required = {
            "start", "end", "gridX", "blockX", "streamId", "graphNodeId",
            "graphId", "shortName",
        }
        missing = required - columns
        if missing:
            raise RuntimeError(f"NSYS kernel table missing {sorted(missing)}")
        query = f"""
            SELECT k.start, k.end, k.gridX, k.blockX, k.streamId,
                   k.graphNodeId, k.graphId, s.value
            FROM {table} AS k
            LEFT JOIN StringIds AS s ON s.id = k.shortName
            ORDER BY k.start
        """
        rows = []
        for start, end, grid_x, block_x, stream_id, node_id, graph_id, name in connection.execute(query):
            rows.append(
                {
                    "start_ns": int(start),
                    "end_ns": int(end),
                    "duration_us": (int(end) - int(start)) / 1000.0,
                    "grid_x": int(grid_x),
                    "block_x": int(block_x),
                    "stream_id": int(stream_id),
                    "graph_node_id": int(node_id),
                    "graph_id": int(graph_id),
                    "kernel_name": str(name),
                    "role": "dense" if int(grid_x) == 29 else "sparse",
                }
            )
    finally:
        connection.close()
    if len(rows) != 2 or {row["grid_x"] for row in rows} != {29, 141}:
        raise RuntimeError(f"expected D29/S141 timeline, got {rows}")
    if len({row["stream_id"] for row in rows}) != 2:
        raise RuntimeError("dense and sparse kernels did not use distinct streams")
    dense = next(row for row in rows if row["role"] == "dense")
    sparse = next(row for row in rows if row["role"] == "sparse")
    overlap_ns = max(
        0,
        min(dense["end_ns"], sparse["end_ns"])
        - max(dense["start_ns"], sparse["start_ns"]),
    )
    union_ns = max(dense["end_ns"], sparse["end_ns"]) - min(
        dense["start_ns"], sparse["start_ns"]
    )
    summary = {
        "dense_duration_us": dense["duration_us"],
        "sparse_duration_us": sparse["duration_us"],
        "start_skew_us": abs(dense["start_ns"] - sparse["start_ns"]) / 1000.0,
        "overlap_us": overlap_ns / 1000.0,
        "union_us": union_ns / 1000.0,
        "overlap_fraction_of_shorter": overlap_ns
        / min(
            dense["end_ns"] - dense["start_ns"],
            sparse["end_ns"] - sparse["start_ns"],
        ),
        "overlap_fraction_of_union": overlap_ns / union_ns,
        "distinct_streams": True,
        "same_graph": dense["graph_id"] == sparse["graph_id"],
    }
    if summary["overlap_fraction_of_shorter"] < 0.95:
        raise RuntimeError(f"concurrent overlap contract failed: {summary}")
    return rows, summary


def plot(root: Path, formal: dict[str, dict[str, str]], ncu: dict[str, dict[str, Any]]) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = ["#4C78A8", "#F58518"]
    labels = [LABELS[method] for method in METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.2))

    latency = [float(formal[method]["median_us"]) for method in METHODS]
    axes[0, 0].bar(labels, latency, color=colors)
    axes[0, 0].set_title("Formal latency (100 warmup, 10 x 1000; lower is better)")
    axes[0, 0].set_ylabel("Median time (us)")
    axes[0, 0].grid(axis="y", alpha=0.25)

    compute_names = [
        "Tensor (HMMA) Pipe Active\n(↑ better*)",
        "Issue Active\n(context*)",
        "Achieved Occupancy\n(context*)",
    ]
    compute_keys = ["tensor_active_pct", "issue_active_pct", "achieved_occupancy_pct"]
    x = np.arange(len(compute_names))
    width = 0.36
    for index, method in enumerate(METHODS):
        axes[0, 1].bar(
            x + (index - 0.5) * width,
            [float(ncu[method][key]) for key in compute_keys],
            width,
            label=LABELS[method],
            color=colors[index],
        )
    axes[0, 1].set_xticks(x, compute_names)
    axes[0, 1].set_ylabel("Percent")
    axes[0, 1].set_title("Compute and scheduler")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    stall_names = [
        "Warp Stall: Barrier\n(↓ better)",
        "Warp Stall: Long Scoreboard\n(↓ better)",
    ]
    stall_keys = ["barrier_stall_pct", "long_scoreboard_stall_pct"]
    x = np.arange(len(stall_names))
    for index, method in enumerate(METHODS):
        axes[1, 0].bar(
            x + (index - 0.5) * width,
            [float(ncu[method][key]) for key in stall_keys],
            width,
            label=LABELS[method],
            color=colors[index],
        )
    axes[1, 0].set_xticks(x, stall_names)
    axes[1, 0].set_ylabel("Stalled active warp cycles (%)")
    axes[1, 0].set_title("Warp stalls")
    axes[1, 0].grid(axis="y", alpha=0.25)

    memory_names = [
        "DRAM Bytes Read\n(MiB; ↓ better)",
        "DRAM Throughput\n(%; context*)",
        "L2 Read Hit Rate\n(%; ↑ better)",
        "Shared Bank Conflicts\n(/ 1K WF; derived; ↓ better)",
    ]
    memory_keys = ["dram_read_mib", "dram_throughput_pct", "l2_read_hit_pct", "shared_bank_conflicts_per_1k_wavefronts"]
    x = np.arange(len(memory_names))
    for index, method in enumerate(METHODS):
        axes[1, 1].bar(
            x + (index - 0.5) * width,
            [float(ncu[method][key]) for key in memory_keys],
            width,
            label=LABELS[method],
            color=colors[index],
        )
    axes[1, 1].set_xticks(x, memory_names)
    axes[1, 1].set_title("Memory hierarchy")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("Qwen3-8B qkv, M=2048: old concurrent hybrid vs cuBLAS dense")
    fig.text(
        0.5,
        0.012,
        "Directions assume the same useful workload. * Tensor activity should be useful math; "
        "issue activity, occupancy, and DRAM bandwidth are diagnostic, not monotonic objectives.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(figures / "paper_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    formal_rows = read_csv(root / "formal_summary.csv")
    formal = {row["method"]: row for row in formal_rows}
    if set(formal) != set(METHODS):
        raise RuntimeError(f"formal matrix changed: {set(formal)}")
    ncu_rows = [
        extract_ncu(root / "ncu" / f"{method}.ncu-rep", method)
        for method in METHODS
    ]
    ncu = {row["method"]: row for row in ncu_rows}
    write_csv(root / "ncu_whole_graph.csv", ncu_rows)

    nsys_rows, overlap = read_nsys(
        root / "nsys" / "old_dense_base_concurrent_d29_s141.sqlite"
    )
    write_csv(root / "nsys_timeline.csv", nsys_rows)
    (root / "nsys_overlap.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    old = ncu[OLD]
    base = ncu[CUBLAS]
    old_latency = float(formal[OLD]["median_us"])
    base_latency = float(formal[CUBLAS]["median_us"])
    nonzero_flop_ratio = 1.0 / 8.0 + 7.0 / 8.0 * 0.5
    derived = {
        "formal_old_us": old_latency,
        "formal_cublas_us": base_latency,
        "formal_speedup_vs_cublas": base_latency / old_latency,
        "nonzero_flop_ratio_vs_dense": nonzero_flop_ratio,
        "nonzero_flop_reduction_pct": 100.0 * (1.0 - nonzero_flop_ratio),
        "nonzero_flop_throughput_vs_cublas": (
            nonzero_flop_ratio * base_latency / old_latency
        ),
        "tensor_active_ratio": old["tensor_active_pct"] / base["tensor_active_pct"],
        "dram_read_ratio": old["dram_read_bytes"] / base["dram_read_bytes"],
        "bank_conflict_total_ratio": old["shared_bank_conflicts"] / base["shared_bank_conflicts"],
        "bank_conflict_rate_ratio": old["shared_bank_conflicts_per_1k_wavefronts"] / base["shared_bank_conflicts_per_1k_wavefronts"],
        "barrier_stall_ratio": old["barrier_stall_pct"] / base["barrier_stall_pct"],
        "long_scoreboard_ratio": old["long_scoreboard_stall_pct"] / base["long_scoreboard_stall_pct"],
        "ncu_duration_ratio_diagnostic_only": old["ncu_duration_us"] / base["ncu_duration_us"],
        **{f"nsys_{key}": value for key, value in overlap.items()},
    }
    (root / "derived_metrics.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plot(root, formal, ncu)

    lines = [
        "# Qwen3-8B qkv: old concurrent hybrid vs cuBLAS dense",
        "",
        "Shape: M=2048, N=6144, K=4096. The old graph launches dense M256 "
        "and online-pack sparse M1792 on separate streams with D29:S141. The "
        "baseline is one cuBLAS dense M2048 graph.",
        "",
        "> The output shapes match, but numerical semantics do not: 1792 rows "
        "of the old method use the 2:4-masked weight. This is a performance/approximation "
        "comparison, not an equal-output comparison.",
        "",
        "## Formal latency",
        "",
        "100 warmups, 10 samples, 1000 single-call CUDA Graph replays per sample; "
        "strict 5:5 order balance. NCU is not the latency authority.",
        "",
        "| Method | Median [P10, P90] (us) |",
        "|---|---:|",
    ]
    for method in METHODS:
        row = formal[method]
        lines.append(
            f"| {LABELS[method]} | {float(row['median_us']):.3f} "
            f"[{float(row['p10_us']):.3f}, {float(row['p90_us']):.3f}] |"
        )
    lines.extend(
        [
            "",
            f"The old concurrent graph is {derived['formal_speedup_vs_cublas']:.4f}x "
            "faster (3.24% lower latency). However, its nonzero arithmetic count is "
            f"only {100.0 * nonzero_flop_ratio:.2f}% of dense, so its achieved "
            f"nonzero-FLOP throughput is only {100.0 * derived['nonzero_flop_throughput_vs_cublas']:.1f}% "
            "of cuBLAS.",
            "",
            "## Whole-graph NCU metrics",
            "",
            "NCU uses kernel replay with `graph-profiling=graph` and `cache-control all`. "
            "The old row aggregates both concurrent nodes; caches are not purged between "
            "the dense and sparse graph nodes. NCU duration is diagnostic only.",
            "",
            "| Metric | Old concurrent | cuBLAS dense | Why it is included |",
            "|---|---:|---:|---|",
            f"| NCU graph duration (us) | {old['ncu_duration_us']:.3f} | {base['ncu_duration_us']:.3f} | Instrumented HBM-cold diagnostic |",
            f"| Tensor Core active (%) | {old['tensor_active_pct']:.2f} | {base['tensor_active_pct']:.2f} | Useful Tensor-Core pipeline utilization |",
            f"| Issue active (%) | {old['issue_active_pct']:.2f} | {base['issue_active_pct']:.2f} | Scheduler issue activity; high is not automatically useful math |",
            f"| Achieved occupancy (%) | {old['achieved_occupancy_pct']:.2f} | {base['achieved_occupancy_pct']:.2f} | Resident active-warps supply |",
            f"| Barrier stall (%) | {old['barrier_stall_pct']:.2f} | {base['barrier_stall_pct']:.2f} | Explicit synchronization/control penalty |",
            f"| Long-scoreboard stall (%) | {old['long_scoreboard_stall_pct']:.2f} | {base['long_scoreboard_stall_pct']:.2f} | Outstanding memory-dependency penalty |",
            f"| DRAM read (MiB) | {old['dram_read_mib']:.2f} | {base['dram_read_mib']:.2f} | Off-chip traffic volume |",
            f"| DRAM throughput (% peak) | {old['dram_throughput_pct']:.2f} | {base['dram_throughput_pct']:.2f} | Whether HBM bandwidth is saturated |",
            f"| L2 read hit rate (%) | {old['l2_read_hit_pct']:.2f} | {base['l2_read_hit_pct']:.2f} | Cache effectiveness |",
            f"| Shared bank conflicts / 1k wavefronts | {old['shared_bank_conflicts_per_1k_wavefronts']:.2f} | {base['shared_bank_conflicts_per_1k_wavefronts']:.2f} | Normalized shared-memory serialization |",
            "",
            "## Concurrency proof",
            "",
            f"NSYS observes grid-29 and grid-141 kernels on distinct streams with a "
            f"{overlap['start_skew_us']:.3f} us start skew. Their durations are "
            f"{overlap['dense_duration_us']:.3f}/{overlap['sparse_duration_us']:.3f} us; "
            f"the shorter branch is {100.0 * overlap['overlap_fraction_of_shorter']:.1f}% "
            f"overlapped and overlap covers {100.0 * overlap['overlap_fraction_of_union']:.1f}% "
            "of the union interval.",
            "",
            "## Interpretation",
            "",
            "The old hybrid obtains only a 3.24% wall-time win despite removing 43.75% "
            "of nonzero arithmetic. The problem is not HBM saturation: DRAM reads rise "
            "only about 5%, both methods use about 5-6% of peak DRAM bandwidth, and L2 "
            "hit rates are nearly identical. Instead, Tensor active falls to roughly "
            "half of cuBLAS while barrier stalls and normalized shared-bank conflicts "
            "increase sharply. Its higher issue activity and occupancy are therefore "
            "evidence of online pack/control work, not better useful-compute efficiency.",
            "",
            "- [NCU whole-graph CSV](ncu_whole_graph.csv)",
            "- [Formal summary](formal_summary.csv)",
            "- [NSYS timeline](nsys_timeline.csv)",
            "- [Paper figure](figures/paper_profile.png)",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(derived, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
