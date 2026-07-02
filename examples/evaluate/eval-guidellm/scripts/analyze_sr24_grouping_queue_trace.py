#!/usr/bin/env python3
"""Estimate useful SR24 verifier-block grouping from live trace JSONL files.

This is an offline scheduler diagnostic.  It does not change vLLM behavior and
does not prove that delaying verification is latency-safe.  It answers a
smaller systems question: if the live scheduler had a queue of compatible
fixed-prefix verifier blocks, how often could it feed the packed/grouped MLP
operator shape requested by the planner?
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BS_RE = re.compile(r"/bs(\d+)(?:/|$)")


@dataclass(frozen=True)
class Block:
    index: int
    key: tuple[Any, ...]
    batch_size: int
    dense_rows: int
    base_rows: int
    target_dense_rows: int
    target_base_rows: int
    local_speedup: float


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


def infer_batch_size(path: Path, records: list[dict[str, Any]]) -> int:
    match = BS_RE.search(str(path))
    if match:
        return int(match.group(1))
    for record in records:
        value = _as_int(record.get("policy_batch"))
        if value:
            return value
    return 0


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "sr24_grouping_opportunity":
                records.append(row)
    return records


def load_policy(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = payload.get("policies")
    if not isinstance(policies, list):
        raise ValueError(f"policy file {path} does not contain a policies list")
    out: dict[int, dict[str, Any]] = {}
    for item in policies:
        if not isinstance(item, dict):
            continue
        batch = _as_int(item.get("batch_size"))
        if batch > 0:
            out[batch] = item
    return out


def compatible_blocks(
    path: Path,
    *,
    policy_by_batch: dict[int, dict[str, Any]] | None = None,
) -> tuple[int, list[Block], dict[str, int]]:
    records = load_records(path)
    batch_size = infer_batch_size(path, records)
    policy_by_batch = policy_by_batch or {}
    override_policy = policy_by_batch.get(batch_size)
    counters = {
        "trace_records": len(records),
        "compact_records": 0,
        "eligible_records": 0,
        "noncompact_records": 0,
        "incompatible_records": 0,
        "policy_override_records": 0,
    }
    blocks: list[Block] = []
    for record in records:
        if not bool(record.get("compact_spec_batch")):
            counters["noncompact_records"] += 1
            continue
        counters["compact_records"] += 1
        valid_width = _as_int(record.get("valid_width"))
        scheduled_width = _as_int(record.get("scheduled_width"))
        prefix = _as_int(record.get("prefix"))
        if override_policy is not None:
            policy = override_policy
            counters["policy_override_records"] += 1
        else:
            policy = record
        policy_batch = _as_int(policy.get("batch_size") or record.get("policy_batch"),
                               batch_size)
        policy_k = _as_int(policy.get("k") or record.get("policy_k"))
        policy_prefix = _as_int(policy.get("prefix") or record.get("policy_prefix"),
                                prefix)
        dense_rows = _as_int(record.get("dense_rows"))
        base_rows = _as_int(record.get("base_rows"))
        target_dense_rows = _as_int(
            policy.get("grouped_dense_rows") or record.get("grouped_dense_rows")
        )
        target_base_rows = _as_int(
            policy.get("grouped_base_rows") or record.get("grouped_base_rows")
        )
        speedup = _as_float(
            policy.get("mixed_local_speedup_vs_dense")
            or record.get("mixed_local_speedup_vs_dense"),
            1.0,
        )
        compatible = (
            (override_policy is not None or bool(record.get("policy_compatible")))
            and bool(record.get("descriptor_available"))
            and scheduled_width > 0
            and valid_width > 0
            and policy_batch > 0
            and dense_rows > 0
            and base_rows > 0
            and target_dense_rows > 0
            and target_base_rows > 0
            and policy_k in (0, valid_width)
            and policy_prefix == prefix
            and speedup > 1.0
        )
        if not compatible:
            counters["incompatible_records"] += 1
            continue
        key = (
            policy_batch,
            scheduled_width,
            valid_width,
            prefix,
            target_dense_rows,
            target_base_rows,
        )
        blocks.append(
            Block(
                index=len(blocks),
                key=key,
                batch_size=batch_size or policy_batch,
                dense_rows=dense_rows,
                base_rows=base_rows,
                target_dense_rows=target_dense_rows,
                target_base_rows=target_base_rows,
                local_speedup=speedup,
            )
        )
        counters["eligible_records"] += 1
    return batch_size, blocks, counters


@dataclass
class SimStats:
    groups: int = 0
    grouped_blocks: int = 0
    fallback_blocks: int = 0
    grouped_dense_rows: int = 0
    grouped_base_rows: int = 0
    fallback_dense_rows: int = 0
    fallback_base_rows: int = 0
    wait_blocks: list[int] | None = None
    fill_dense: list[float] | None = None
    fill_base: list[float] | None = None
    dense_time_units: float = 0.0
    mixed_time_units: float = 0.0

    def __post_init__(self) -> None:
        if self.wait_blocks is None:
            self.wait_blocks = []
        if self.fill_dense is None:
            self.fill_dense = []
        if self.fill_base is None:
            self.fill_base = []


@dataclass(frozen=True)
class PlanEvent:
    action: str
    reason: str
    key: tuple[Any, ...]
    start_block_index: int
    end_block_index: int
    block_count: int
    block_indices: tuple[int, ...]
    dense_rows: int
    base_rows: int
    target_dense_rows: int
    target_base_rows: int
    wait_blocks: int
    dense_fill: float
    base_fill: float
    local_speedup: float
    dense_time_units: float
    mixed_time_units: float


def simulate_key_plan(
    blocks: list[Block],
    max_wait_blocks: int,
) -> tuple[SimStats, list[PlanEvent]]:
    stats = SimStats()
    events: list[PlanEvent] = []
    i = 0
    n = len(blocks)
    while i < n:
        first = blocks[i]
        dense_sum = 0
        base_sum = 0
        found_j: int | None = None
        max_j = min(n - 1, i + max_wait_blocks)
        for j in range(i, max_j + 1):
            dense_sum += blocks[j].dense_rows
            base_sum += blocks[j].base_rows
            if (
                dense_sum >= first.target_dense_rows
                and base_sum >= first.target_base_rows
            ):
                found_j = j
                break
        if found_j is None:
            stats.fallback_blocks += 1
            stats.fallback_dense_rows += first.dense_rows
            stats.fallback_base_rows += first.base_rows
            stats.dense_time_units += 1.0
            stats.mixed_time_units += 1.0
            reason = (
                "tail_underfilled"
                if i + max_wait_blocks >= n - 1
                else "timeout_underfilled"
            )
            events.append(
                PlanEvent(
                    action="fallback",
                    reason=reason,
                    key=first.key,
                    start_block_index=first.index,
                    end_block_index=first.index,
                    block_count=1,
                    block_indices=(first.index,),
                    dense_rows=first.dense_rows,
                    base_rows=first.base_rows,
                    target_dense_rows=first.target_dense_rows,
                    target_base_rows=first.target_base_rows,
                    wait_blocks=0,
                    dense_fill=first.dense_rows / max(first.target_dense_rows, 1),
                    base_fill=first.base_rows / max(first.target_base_rows, 1),
                    local_speedup=1.0,
                    dense_time_units=1.0,
                    mixed_time_units=1.0,
                )
            )
            i += 1
            continue

        group = blocks[i:found_j + 1]
        group_dense = sum(block.dense_rows for block in group)
        group_base = sum(block.base_rows for block in group)
        group_speedup = statistics.fmean(block.local_speedup for block in group)
        group_len = len(group)
        mixed_time = float(group_len) / max(group_speedup, 1.0)
        stats.groups += 1
        stats.grouped_blocks += group_len
        stats.grouped_dense_rows += group_dense
        stats.grouped_base_rows += group_base
        stats.dense_time_units += float(group_len)
        stats.mixed_time_units += mixed_time
        stats.wait_blocks.append(group_len - 1)
        stats.fill_dense.append(group_dense / max(first.target_dense_rows, 1))
        stats.fill_base.append(group_base / max(first.target_base_rows, 1))
        events.append(
            PlanEvent(
                action="group",
                reason="target_reached",
                key=first.key,
                start_block_index=group[0].index,
                end_block_index=group[-1].index,
                block_count=group_len,
                block_indices=tuple(block.index for block in group),
                dense_rows=group_dense,
                base_rows=group_base,
                target_dense_rows=first.target_dense_rows,
                target_base_rows=first.target_base_rows,
                wait_blocks=group_len - 1,
                dense_fill=group_dense / max(first.target_dense_rows, 1),
                base_fill=group_base / max(first.target_base_rows, 1),
                local_speedup=group_speedup,
                dense_time_units=float(group_len),
                mixed_time_units=mixed_time,
            )
        )
        i = found_j + 1
    return stats, events


def simulate_key(blocks: list[Block], max_wait_blocks: int) -> SimStats:
    return simulate_key_plan(blocks, max_wait_blocks)[0]


def merge_stats(stats_list: list[SimStats]) -> SimStats:
    merged = SimStats()
    for stats in stats_list:
        merged.groups += stats.groups
        merged.grouped_blocks += stats.grouped_blocks
        merged.fallback_blocks += stats.fallback_blocks
        merged.grouped_dense_rows += stats.grouped_dense_rows
        merged.grouped_base_rows += stats.grouped_base_rows
        merged.fallback_dense_rows += stats.fallback_dense_rows
        merged.fallback_base_rows += stats.fallback_base_rows
        merged.dense_time_units += stats.dense_time_units
        merged.mixed_time_units += stats.mixed_time_units
        merged.wait_blocks.extend(stats.wait_blocks or [])
        merged.fill_dense.extend(stats.fill_dense or [])
        merged.fill_base.extend(stats.fill_base or [])
    return merged


def summarize(
    *,
    trace_path: Path,
    batch_size: int,
    blocks: list[Block],
    counters: dict[str, int],
    max_wait_blocks: int,
) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], list[Block]] = defaultdict(list)
    for block in blocks:
        by_key[block.key].append(block)
    stats = merge_stats([
        simulate_key(group, max_wait_blocks)
        for group in by_key.values()
    ])
    eligible = len(blocks)
    row_total = sum(block.dense_rows + block.base_rows for block in blocks)
    grouped_rows = stats.grouped_dense_rows + stats.grouped_base_rows
    return {
        "trace_path": str(trace_path),
        "batch_size": batch_size,
        "max_wait_blocks": max_wait_blocks,
        **counters,
        "eligible_blocks": eligible,
        "groups": stats.groups,
        "grouped_blocks": stats.grouped_blocks,
        "fallback_blocks": stats.fallback_blocks,
        "grouped_block_pct": (
            100.0 * stats.grouped_blocks / eligible if eligible else 0.0
        ),
        "grouped_row_pct": (
            100.0 * grouped_rows / row_total if row_total else 0.0
        ),
        "avg_wait_blocks": (
            statistics.fmean(stats.wait_blocks) if stats.wait_blocks else 0.0
        ),
        "p90_wait_blocks": (
            sorted(stats.wait_blocks)[int(0.9 * (len(stats.wait_blocks) - 1))]
            if stats.wait_blocks else 0
        ),
        "avg_blocks_per_group": (
            stats.grouped_blocks / stats.groups if stats.groups else 0.0
        ),
        "avg_dense_fill": (
            statistics.fmean(stats.fill_dense) if stats.fill_dense else 0.0
        ),
        "avg_base_fill": (
            statistics.fmean(stats.fill_base) if stats.fill_base else 0.0
        ),
        "estimated_mlp_local_speedup": (
            stats.dense_time_units / stats.mixed_time_units
            if stats.mixed_time_units > 0 else 1.0
        ),
    }


def _json_key(key: tuple[Any, ...]) -> str:
    return json.dumps(list(key), separators=(",", ":"))


def _indices_summary(indices: tuple[int, ...]) -> str:
    if not indices:
        return ""
    if len(indices) == 1:
        return str(indices[0])
    contiguous = all(
        indices[pos] == indices[pos - 1] + 1
        for pos in range(1, len(indices))
    )
    if contiguous:
        return f"{indices[0]}-{indices[-1]}"
    return ",".join(str(index) for index in indices)


def build_plan_rows(
    *,
    trace_path: Path,
    batch_size: int,
    blocks: list[Block],
    max_wait_blocks: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], list[Block]] = defaultdict(list)
    for block in blocks:
        by_key[block.key].append(block)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(by_key.items(), key=lambda item: _json_key(item[0])):
        _, events = simulate_key_plan(group, max_wait_blocks)
        for event in events:
            rows.append({
                "trace_path": str(trace_path),
                "batch_size": batch_size,
                "max_wait_blocks": max_wait_blocks,
                "key": _json_key(key),
                "action": event.action,
                "reason": event.reason,
                "start_block_index": event.start_block_index,
                "end_block_index": event.end_block_index,
                "block_count": event.block_count,
                "block_indices": _indices_summary(event.block_indices),
                "dense_rows": event.dense_rows,
                "base_rows": event.base_rows,
                "target_dense_rows": event.target_dense_rows,
                "target_base_rows": event.target_base_rows,
                "dense_fill": event.dense_fill,
                "base_fill": event.base_fill,
                "wait_blocks": event.wait_blocks,
                "local_speedup": event.local_speedup,
                "estimated_dense_time_units": event.dense_time_units,
                "estimated_mixed_time_units": event.mixed_time_units,
            })
    return rows


def summarize_plan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    for (batch_size, max_wait, action, reason), group in sorted(
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
        dense_units = sum(_as_float(row.get("estimated_dense_time_units"))
                          for row in group)
        mixed_units = sum(_as_float(row.get("estimated_mixed_time_units"))
                          for row in group)
        output.append({
            "batch_size": batch_size,
            "max_wait_blocks": max_wait,
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def derive_recommendations(
    rows: list[dict[str, Any]],
    *,
    target_speedup: float,
) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        batch_size = _as_int(row.get("batch_size"))
        if batch_size > 0:
            by_batch[batch_size].append(row)

    output: list[dict[str, Any]] = []
    for batch_size, batch_rows in sorted(by_batch.items()):
        batch_rows = sorted(batch_rows, key=lambda row: int(row["max_wait_blocks"]))
        target_rows = [
            row for row in batch_rows
            if _as_float(row.get("estimated_mlp_local_speedup")) >= target_speedup
        ]
        best_row = max(
            batch_rows,
            key=lambda row: _as_float(row.get("estimated_mlp_local_speedup")),
        )
        target_row = target_rows[0] if target_rows else None
        no_wait = next(
            (row for row in batch_rows if int(row["max_wait_blocks"]) == 0),
            batch_rows[0],
        )
        if target_row is None:
            design_read = "queue_upper_bound_below_target"
        else:
            min_wait = int(target_row["max_wait_blocks"])
            if min_wait == 0:
                design_read = "single_block_viable"
            elif min_wait <= 1:
                design_read = "short_wait_queue_candidate"
            else:
                design_read = "long_wait_queue_upper_bound"

        output.append({
            "batch_size": batch_size,
            "target_speedup": target_speedup,
            "single_block_estimated_mlp_local_speedup":
                _as_float(no_wait.get("estimated_mlp_local_speedup")),
            "target_reached": target_row is not None,
            "min_wait_blocks_for_target": (
                int(target_row["max_wait_blocks"]) if target_row is not None else ""
            ),
            "speedup_at_min_wait": (
                _as_float(target_row.get("estimated_mlp_local_speedup"))
                if target_row is not None else ""
            ),
            "grouped_block_pct_at_min_wait": (
                _as_float(target_row.get("grouped_block_pct"))
                if target_row is not None else ""
            ),
            "grouped_row_pct_at_min_wait": (
                _as_float(target_row.get("grouped_row_pct"))
                if target_row is not None else ""
            ),
            "avg_wait_blocks_at_min_wait": (
                _as_float(target_row.get("avg_wait_blocks"))
                if target_row is not None else ""
            ),
            "best_wait_blocks": int(best_row["max_wait_blocks"]),
            "best_estimated_mlp_local_speedup":
                _as_float(best_row.get("estimated_mlp_local_speedup")),
            "best_grouped_block_pct": _as_float(best_row.get("grouped_block_pct")),
            "best_grouped_row_pct": _as_float(best_row.get("grouped_row_pct")),
            "design_read": design_read,
        })
    return output


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    plan_summary_rows: list[dict[str, Any]],
    *,
    target_speedup: float,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# SR24 Grouping Queue Trace Analysis\n\n")
        f.write(
            "This is an offline upper-bound diagnostic over live grouping trace "
            "JSONL files.  It treats each compact verification step as one "
            "verifier block and estimates how often a queue could group enough "
            "compatible blocks to feed the packed MLP planner target.  It does "
            "not prove the live scheduler can delay verification safely.\n\n"
        )
        f.write(
            "| bs | max wait blocks | eligible blocks | grouped block % | "
            "grouped row % | policy override records | avg wait | avg blocks/group | dense fill | "
            "base fill | est MLP local speedup |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda item: (
            int(item["batch_size"]), int(item["max_wait_blocks"]))
        ):
            f.write(
                f"| {row['batch_size']} | {row['max_wait_blocks']} | "
                f"{row['eligible_blocks']} | "
                f"{_fmt(row['grouped_block_pct'])} | "
                f"{_fmt(row['grouped_row_pct'])} | "
                f"{row.get('policy_override_records', 0)} | "
                f"{_fmt(row['avg_wait_blocks'])} | "
                f"{_fmt(row['avg_blocks_per_group'])} | "
                f"{_fmt(row['avg_dense_fill'])} | "
                f"{_fmt(row['avg_base_fill'])} | "
                f"{_fmt(row['estimated_mlp_local_speedup'])}x |\n"
            )
        f.write("\n## Read\n\n")
        f.write(
            "- `max wait blocks=0` means no cross-step queue; only a single "
            "verifier block can feed the grouped operator.\n"
        )
        f.write(
            "- Larger wait budgets are an optimistic queue upper bound.  The "
            "implementation would need dependency-safe buffering and fallback "
            "for tail blocks.\n"
        )
        f.write(
            "- `estimated MLP local speedup` uses the planner's local mixed "
            "speedup for grouped blocks and dense fallback for ungrouped blocks; "
            "it is not an end-to-end tok/s prediction.\n"
        )
        f.write("\n## Design Recommendations\n\n")
        f.write(
            f"Target local MLP speedup for this diagnostic: `{target_speedup:.3f}x`.\n\n"
        )
        f.write(
            "| bs | single-block est | target reached | min wait for target | "
            "speedup at min wait | grouped block % | grouped row % | "
            "best wait | best est speedup | design read |\n"
        )
        f.write("|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in recommendations:
            f.write(
                f"| {row['batch_size']} | "
                f"{_fmt(row['single_block_estimated_mlp_local_speedup'])}x | "
                f"{'yes' if row['target_reached'] else 'no'} | "
                f"{row['min_wait_blocks_for_target']} | "
                f"{_fmt(row['speedup_at_min_wait'])}x | "
                f"{_fmt(row['grouped_block_pct_at_min_wait'])} | "
                f"{_fmt(row['grouped_row_pct_at_min_wait'])} | "
                f"{row['best_wait_blocks']} | "
                f"{_fmt(row['best_estimated_mlp_local_speedup'])}x | "
                f"{row['design_read']} |\n"
            )
        f.write("\n")
        f.write(
            "Read `long_wait_queue_upper_bound` as an algorithm/design warning: "
            "the trace says enough compatible work exists only after delaying "
            "multiple verifier blocks, so a live serving implementation needs a "
            "dependency-safe queue and dense fallback. `queue_upper_bound_below_target` "
            "means even this optimistic queue model does not reach the requested "
            "local operator target; the sparse operator itself must improve.\n"
        )
        if plan_summary_rows:
            f.write("\n## Live Queue Plan Artifact\n\n")
            f.write(
                "`queue_plan.csv` and `queue_plan.jsonl` contain one row per "
                "offline group/fallback decision.  The key implementation "
                "fields are `action`, `reason`, `block_indices`, row counts, "
                "fill ratios, and `wait_blocks`; these are the rows to replay "
                "when moving from this trace diagnostic to a live scheduler "
                "queue.\n\n"
            )
            f.write(
                "| bs | max wait | action | reason | events | blocks | "
                "avg wait | dense fill | base fill | est MLP local speedup |\n"
            )
            f.write("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
            for row in plan_summary_rows:
                f.write(
                    f"| {row['batch_size']} | {row['max_wait_blocks']} | "
                    f"{row['action']} | {row['reason']} | "
                    f"{row['events']} | {row['blocks']} | "
                    f"{_fmt(row['avg_wait_blocks'])} | "
                    f"{_fmt(row['avg_dense_fill'])} | "
                    f"{_fmt(row['avg_base_fill'])} | "
                    f"{_fmt(row['estimated_mlp_local_speedup'])}x |\n"
                )
            f.write("\n")
            f.write(
                "This plan is still trace-local.  A live implementation must "
                "preserve request dependencies, cap queue age, and fall back to "
                "dense when the queue cannot fill the packed operator target.\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SR24 verifier-block grouping trace queue potential."
    )
    parser.add_argument(
        "--trace-glob",
        action="append",
        required=True,
        help="Glob for speclink_sr24_grouping_trace.jsonl files.",
    )
    parser.add_argument(
        "--max-wait-blocks",
        default="0,1,3,7",
        help="Comma-separated future verifier-block wait budgets.",
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        default=1.2,
        help="Local MLP speedup target used for recommendation rows.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help=(
            "Optional scheduler policy JSON. When set, per-batch policy rows "
            "override the trace's historical policy fields, useful for "
            "reanalyzing old traces with a refreshed grouped-operator policy."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    trace_paths: list[Path] = []
    for pattern in args.trace_glob:
        trace_paths.extend(Path(path).resolve() for path in glob.glob(pattern))
    trace_paths = sorted(set(trace_paths))
    if not trace_paths:
        raise SystemExit("no trace files matched")

    waits = [
        max(0, int(item.strip()))
        for item in args.max_wait_blocks.split(",")
        if item.strip()
    ]
    if not waits:
        raise SystemExit("empty --max-wait-blocks")

    policy_by_batch = load_policy(args.policy_path)
    rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        batch_size, blocks, counters = compatible_blocks(
            trace_path,
            policy_by_batch=policy_by_batch,
        )
        for wait in waits:
            rows.append(
                summarize(
                    trace_path=trace_path,
                    batch_size=batch_size,
                    blocks=blocks,
                    counters=counters,
                    max_wait_blocks=wait,
                )
            )
            plan_rows.extend(
                build_plan_rows(
                    trace_path=trace_path,
                    batch_size=batch_size,
                    blocks=blocks,
                    max_wait_blocks=wait,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.policy_path is not None:
        (args.output_root / "policy_path.txt").write_text(
            str(args.policy_path.resolve()) + "\n",
            encoding="utf-8",
        )
    recommendations = derive_recommendations(
        rows,
        target_speedup=float(args.target_speedup),
    )
    plan_summary_rows = summarize_plan_rows(plan_rows)
    write_csv(args.output_root / "queue_summary.csv", rows)
    write_csv(args.output_root / "queue_recommendations.csv", recommendations)
    write_csv(args.output_root / "queue_plan.csv", plan_rows)
    write_jsonl(args.output_root / "queue_plan.jsonl", plan_rows)
    write_csv(args.output_root / "queue_plan_summary.csv", plan_summary_rows)
    (args.output_root / "queue_recommendations.json").write_text(
        json.dumps(recommendations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_root / "queue_report.md",
        rows,
        recommendations,
        plan_summary_rows,
        target_speedup=float(args.target_speedup),
    )
    print((args.output_root / "queue_report.md").resolve())


if __name__ == "__main__":
    main()
