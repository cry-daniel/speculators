#!/usr/bin/env python3
"""Summarize SR24 live grouped-queue shadow decisions.

The live shadow queue is intentionally default-off and does not change serving
behavior.  It emits ``sr24_grouped_queue_shadow_decision`` events into the same
grouping trace used by the offline queue planner.  This script checks whether a
live shadow run produces the expected group/fallback mix before a real scheduler
queue is implemented.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


BS_RE = re.compile(r"/bs(\d+)(?:/|$)")


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def infer_batch_size(path: Path, rows: list[dict[str, Any]]) -> int:
    match = BS_RE.search(str(path))
    if match:
        return int(match.group(1))
    for row in rows:
        key = row.get("key")
        if isinstance(key, list) and key:
            batch = _as_int(key[0])
            if batch > 0:
                return batch
    return 0


def load_shadow_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "sr24_grouped_queue_shadow_decision":
                rows.append(row)
    return rows


def summarize_rows(
    *,
    trace_path: Path,
    batch_size: int,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            batch_size,
            _as_int(row.get("max_wait_blocks")),
            row.get("action", ""),
            row.get("reason", ""),
        )
        by_key[key].append(row)

    output: list[dict[str, Any]] = []
    for (batch, wait, action, reason), group in sorted(
        by_key.items(),
        key=lambda item: (
            _as_int(item[0][0]),
            _as_int(item[0][1]),
            str(item[0][2]),
            str(item[0][3]),
        ),
    ):
        blocks = sum(_as_int(row.get("block_count")) for row in group)
        dense_rows = sum(_as_int(row.get("dense_rows")) for row in group)
        base_rows = sum(_as_int(row.get("base_rows")) for row in group)
        dense_units = sum(_as_float(row.get("dense_time_units")) for row in group)
        mixed_units = sum(_as_float(row.get("mixed_time_units")) for row in group)
        output.append({
            "trace_path": str(trace_path),
            "batch_size": batch,
            "max_wait_blocks": wait,
            "action": action,
            "reason": reason,
            "events": len(group),
            "blocks": blocks,
            "dense_rows": dense_rows,
            "base_rows": base_rows,
            "avg_dense_fill": statistics.fmean(
                _as_float(row.get("dense_fill")) for row in group
            ),
            "avg_base_fill": statistics.fmean(
                _as_float(row.get("base_fill")) for row in group
            ),
            "avg_wait_blocks": statistics.fmean(
                _as_float(row.get("wait_blocks")) for row in group
            ),
            "estimated_mlp_local_speedup": (
                dense_units / mixed_units if mixed_units > 0 else 1.0
            ),
        })
    return output


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("batch_size"),
            row.get("max_wait_blocks"),
            row.get("action"),
            row.get("reason"),
        )
        by_key[key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(
        by_key.items(),
        key=lambda item: (
            _as_int(item[0][0]),
            _as_int(item[0][1]),
            str(item[0][2]),
            str(item[0][3]),
        ),
    ):
        events = sum(_as_int(row.get("events")) for row in group)
        blocks = sum(_as_int(row.get("blocks")) for row in group)
        dense_rows = sum(_as_int(row.get("dense_rows")) for row in group)
        base_rows = sum(_as_int(row.get("base_rows")) for row in group)
        weighted_speed = [
            (_as_float(row.get("estimated_mlp_local_speedup")),
             _as_int(row.get("blocks")))
            for row in group
        ]
        total_weight = sum(weight for _, weight in weighted_speed)
        speed = (
            sum(value * weight for value, weight in weighted_speed)
            / max(total_weight, 1)
        )
        output.append({
            "batch_size": key[0],
            "max_wait_blocks": key[1],
            "action": key[2],
            "reason": key[3],
            "events": events,
            "blocks": blocks,
            "dense_rows": dense_rows,
            "base_rows": base_rows,
            "estimated_mlp_local_speedup": speed,
        })
    return output


def load_offline_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compare_with_offline(
    shadow_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    offline_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in offline_rows:
        key = (
            _as_int(row.get("batch_size")),
            _as_int(row.get("max_wait_blocks")),
            row.get("action", ""),
            row.get("reason", ""),
        )
        offline_by_key[key] = row

    output: list[dict[str, Any]] = []
    for row in shadow_rows:
        key = (
            _as_int(row.get("batch_size")),
            _as_int(row.get("max_wait_blocks")),
            row.get("action", ""),
            row.get("reason", ""),
        )
        offline = offline_by_key.get(key)
        if offline is None:
            output.append({
                **row,
                "offline_present": False,
                "events_delta": "",
                "blocks_delta": "",
                "dense_rows_delta": "",
                "base_rows_delta": "",
            })
            continue
        output.append({
            **row,
            "offline_present": True,
            "offline_events": _as_int(offline.get("events")),
            "offline_blocks": _as_int(offline.get("blocks")),
            "offline_dense_rows": _as_int(offline.get("dense_rows")),
            "offline_base_rows": _as_int(offline.get("base_rows")),
            "events_delta": _as_int(row.get("events")) - _as_int(
                offline.get("events")
            ),
            "blocks_delta": _as_int(row.get("blocks")) - _as_int(
                offline.get("blocks")
            ),
            "dense_rows_delta": _as_int(row.get("dense_rows")) - _as_int(
                offline.get("dense_rows")
            ),
            "base_rows_delta": _as_int(row.get("base_rows")) - _as_int(
                offline.get("base_rows")
            ),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    aggregate_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# SR24 Grouped Queue Shadow Analysis\n\n")
        f.write(
            "This report summarizes default-off live shadow queue decisions. "
            "The shadow queue does not delay verifier execution and is not a "
            "throughput result; it validates the online grouping/fallback "
            "logic before a real scheduler queue is implemented.\n\n"
        )
        f.write(
            "| bs | max wait | action | reason | events | blocks | dense rows | "
            "base rows | est local speedup |\n"
        )
        f.write("|---:|---:|---|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            f.write(
                f"| {row['batch_size']} | {row['max_wait_blocks']} | "
                f"{row['action']} | {row['reason']} | {row['events']} | "
                f"{row['blocks']} | {row['dense_rows']} | {row['base_rows']} | "
                f"{_fmt(row['estimated_mlp_local_speedup'])}x |\n"
            )
        if comparison_rows:
            f.write("\n## Offline Plan Delta\n\n")
            f.write(
                "| bs | max wait | action | reason | offline present | "
                "events delta | blocks delta | dense rows delta | "
                "base rows delta |\n"
            )
            f.write("|---:|---:|---|---|:---:|---:|---:|---:|---:|\n")
            for row in comparison_rows:
                f.write(
                    f"| {row['batch_size']} | {row['max_wait_blocks']} | "
                    f"{row['action']} | {row['reason']} | "
                    f"{'yes' if row['offline_present'] else 'no'} | "
                    f"{row['events_delta']} | {row['blocks_delta']} | "
                    f"{row['dense_rows_delta']} | {row['base_rows_delta']} |\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SR24 live grouped-queue shadow trace events."
    )
    parser.add_argument(
        "--trace-glob",
        action="append",
        required=True,
        help="Glob for speclink_sr24_grouping_trace.jsonl files.",
    )
    parser.add_argument(
        "--offline-plan-summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional queue_plan_summary.csv from "
            "analyze_sr24_grouping_queue_trace.py for parity comparison."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    trace_paths: list[Path] = []
    for pattern in args.trace_glob:
        trace_paths.extend(
            Path(path).resolve() for path in glob.glob(pattern, recursive=True)
        )
    trace_paths = sorted(set(trace_paths))
    if not trace_paths:
        raise SystemExit("no trace files matched")

    per_trace_rows: list[dict[str, Any]] = []
    raw_count_rows: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        shadow_rows = load_shadow_rows(trace_path)
        batch_size = infer_batch_size(trace_path, shadow_rows)
        raw_count_rows.append({
            "trace_path": str(trace_path),
            "batch_size": batch_size,
            "shadow_decision_events": len(shadow_rows),
        })
        per_trace_rows.extend(
            summarize_rows(
                trace_path=trace_path,
                batch_size=batch_size,
                rows=shadow_rows,
            )
        )

    aggregate_rows = aggregate_summary(per_trace_rows)
    comparison_rows: list[dict[str, Any]] = []
    if args.offline_plan_summary_csv is not None:
        offline_rows = load_offline_summary(args.offline_plan_summary_csv)
        comparison_rows = compare_with_offline(aggregate_rows, offline_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "shadow_trace_counts.csv", raw_count_rows)
    write_csv(args.output_root / "shadow_summary_by_trace.csv", per_trace_rows)
    write_csv(args.output_root / "shadow_summary.csv", aggregate_rows)
    write_csv(args.output_root / "shadow_offline_delta.csv", comparison_rows)
    write_report(
        args.output_root / "shadow_report.md",
        aggregate_rows=aggregate_rows,
        comparison_rows=comparison_rows,
    )
    print((args.output_root / "shadow_report.md").resolve())


if __name__ == "__main__":
    main()
