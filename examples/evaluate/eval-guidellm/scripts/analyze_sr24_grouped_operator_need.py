#!/usr/bin/env python3
"""Summarize SR24 grouped/fused operator demand from a live grouping trace.

This is a diagnosis helper for the next SR24 optimization step.  It does not
claim serving speedup.  It answers two narrower questions from
``speclink_sr24_grouping_trace.jsonl``:

1. How often does the live scheduler fall back because the packed mixed MLP is
   not implemented for grouped verifier blocks?
2. For those fallbacks, how many are near the target effective batch already,
   and how many would require a real cross-step grouped verifier queue?

The script intentionally keeps the output small enough to inspect directly in
``results.bak`` after each smoke run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("empty integer list")
    return out


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def row_weight(row: dict[str, Any]) -> int:
    dense_rows = as_int(row.get("dense_rows"))
    base_rows = as_int(row.get("base_rows"))
    return max(0, dense_rows + base_rows)


def summarize(rows: list[dict[str, Any]], tolerances: list[int]) -> dict[str, Any]:
    reason_counts = Counter(
        str(row.get("single_block_fallback_reason") or "none") for row in rows
    )
    action_counts = Counter(str(row.get("policy_action") or "") for row in rows)
    operator_counts = Counter(str(row.get("mixed_operator") or "") for row in rows)
    by_action_reason = Counter(
        (
            str(row.get("policy_action") or ""),
            str(row.get("single_block_fallback_reason") or "none"),
            str(row.get("mixed_operator") or ""),
            bool(row.get("operator_supported_live")),
        )
        for row in rows
    )

    op_unimplemented = [
        row
        for row in rows
        if str(row.get("single_block_fallback_reason") or "")
        == "operator_unimplemented"
    ]
    underfilled = [
        row
        for row in rows
        if str(row.get("single_block_fallback_reason") or "") == "underfilled"
    ]
    mixed_rows = [
        row
        for row in rows
        if str(row.get("policy_action") or "") == "use_mixed_single_block"
        and str(row.get("single_block_fallback_reason") or "") == ""
    ]

    tolerance_rows: list[dict[str, Any]] = []
    for tol in tolerances:
        near = 0
        cross = 0
        weighted_near = 0
        weighted_cross = 0
        for row in op_unimplemented:
            target = as_int(row.get("target_effective_batch_size"))
            active = as_int(row.get("active_requests"))
            weight = row_weight(row)
            if target > 0 and target - active <= tol:
                near += 1
                weighted_near += weight
            else:
                cross += 1
                weighted_cross += weight
        total = max(1, len(op_unimplemented))
        weighted_total = max(1, weighted_near + weighted_cross)
        tolerance_rows.append({
            "near_full_tolerance": tol,
            "operator_unimplemented_steps": len(op_unimplemented),
            "near_target_steps": near,
            "cross_step_queue_steps": cross,
            "near_target_step_fraction": near / total,
            "near_target_row_fraction": weighted_near / weighted_total,
            "near_target_rows": weighted_near,
            "cross_step_queue_rows": weighted_cross,
        })

    reason_rows: list[dict[str, Any]] = []
    grouped_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_reason[str(row.get("single_block_fallback_reason") or "none")].append(
            row
        )
    for reason, group in sorted(grouped_by_reason.items()):
        weights = [row_weight(row) for row in group]
        active = [as_int(row.get("active_requests")) for row in group]
        reason_rows.append({
            "reason": reason,
            "steps": len(group),
            "row_weight": sum(weights),
            "avg_active_requests": mean(active) if active else 0.0,
            "min_active_requests": min(active) if active else 0,
            "max_active_requests": max(active) if active else 0,
        })

    local_weight = 0.0
    local_mixed_cost = 0.0
    local_rows_with_speedup = 0
    for row in op_unimplemented:
        speedup = as_float(row.get("mixed_local_speedup_vs_dense"))
        weight = row_weight(row)
        if speedup is None or speedup <= 0 or weight <= 0:
            continue
        local_weight += weight
        local_mixed_cost += weight / speedup
        local_rows_with_speedup += weight
    local_operator_speedup = (
        local_weight / local_mixed_cost if local_mixed_cost > 0 else None
    )

    active_hist = Counter(as_int(row.get("active_requests")) for row in op_unimplemented)
    active_hist_rows = [
        {
            "active_requests": active,
            "operator_unimplemented_steps": count,
        }
        for active, count in sorted(active_hist.items())
    ]

    return {
        "total_steps": len(rows),
        "reason_counts": dict(reason_counts),
        "action_counts": dict(action_counts),
        "operator_counts": dict(operator_counts),
        "by_action_reason": [
            {
                "policy_action": action,
                "fallback_reason": reason,
                "mixed_operator": operator,
                "operator_supported_live": supported,
                "steps": count,
            }
            for (action, reason, operator, supported), count in by_action_reason.most_common()
        ],
        "operator_unimplemented_steps": len(op_unimplemented),
        "underfilled_steps": len(underfilled),
        "mixed_single_block_steps": len(mixed_rows),
        "operator_unimplemented_row_weight": sum(row_weight(row) for row in op_unimplemented),
        "underfilled_row_weight": sum(row_weight(row) for row in underfilled),
        "mixed_single_block_row_weight": sum(row_weight(row) for row in mixed_rows),
        "local_operator_speedup_if_grouped_replaced_fallback": local_operator_speedup,
        "local_operator_rows_with_speedup": local_rows_with_speedup,
        "tolerance_rows": tolerance_rows,
        "reason_rows": reason_rows,
        "active_hist_rows": active_hist_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def write_markdown(path: Path, trace: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# SR24 Grouped Operator Need")
    lines.append("")
    lines.append(f"- trace: `{trace}`")
    lines.append(f"- total grouping events: `{summary['total_steps']}`")
    lines.append(
        "- local operator-only speedup if all grouped fallback rows used the "
        "planner mixed operator: "
        f"`{fmt_float(summary['local_operator_speedup_if_grouped_replaced_fallback'])}x`"
    )
    lines.append("")
    lines.append("## Action/Reason Counts")
    lines.append("")
    lines.append("| policy action | fallback reason | operator | supported | steps |")
    lines.append("|---|---|---|---:|---:|")
    for row in summary["by_action_reason"]:
        lines.append(
            "| {policy_action} | {fallback_reason} | {mixed_operator} | "
            "{operator_supported_live} | {steps} |".format(**row)
        )
    lines.append("")
    lines.append("## Near-Target vs Cross-Step Queue")
    lines.append("")
    lines.append(
        "| near-full tolerance | op-unimplemented steps | near-target steps | "
        "cross-step queue steps | near-target row fraction |"
    )
    lines.append("|---:|---:|---:|---:|---:|")
    for row in summary["tolerance_rows"]:
        lines.append(
            "| {near_full_tolerance} | {operator_unimplemented_steps} | "
            "{near_target_steps} | {cross_step_queue_steps} | "
            "{near_target_row_fraction:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## Reason Summary")
    lines.append("")
    lines.append(
        "| reason | steps | row weight | avg active requests | min active | max active |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["reason_rows"]:
        lines.append(
            "| {reason} | {steps} | {row_weight} | {avg_active_requests:.2f} | "
            "{min_active_requests} | {max_active_requests} |".format(**row)
        )
    lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append(
        "Rows in `operator_unimplemented` are the work a real grouped/fused "
        "packed MLP must cover.  A high near-target fraction means policy "
        "selection/capacity padding can recover some steps.  A high cross-step "
        "fraction means the implementation needs a verifier-block queue or a "
        "larger fused operator shape; opening the serial PyTorch branch is not "
        "expected to help serving throughput."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to results.bak with a timestamp.",
    )
    parser.add_argument(
        "--near-full-tolerances",
        type=parse_int_list,
        default=parse_int_list("0,1,2,4,8"),
    )
    args = parser.parse_args()

    trace = args.trace.resolve()
    rows = load_trace(trace)
    summary = summarize(rows, args.near_full_tolerances)
    output_root = args.output_root
    if output_root is None:
        output_root = (
            Path("examples/evaluate/eval-guidellm/results.bak")
            / f"sr24_grouped_operator_need_{timestamp()}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "reason_summary.csv", summary["reason_rows"])
    write_csv(output_root / "near_target_summary.csv", summary["tolerance_rows"])
    write_csv(output_root / "active_request_histogram.csv", summary["active_hist_rows"])
    write_csv(output_root / "action_reason_summary.csv", summary["by_action_reason"])
    write_markdown(output_root / "summary.md", trace, summary)
    print(output_root.resolve())


if __name__ == "__main__":
    main()
