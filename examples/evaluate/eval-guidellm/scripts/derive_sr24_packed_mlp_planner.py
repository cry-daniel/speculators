#!/usr/bin/env python3
"""Derive SR24 packed-MLP serving planner hints from microbench CSV rows.

The packed verifier-block microbench reports several isolated operator paths.
This reducer turns those rows into a small table answering two live-serving
questions:

1. For the current row count, should the mixed dense/sparse MLP be used or
   should the branch fall back to dense?
2. If mixed is not profitable yet, how many useful verifier blocks would need
   to be grouped before the mixed operator reaches the target speedup?

It is an offline analysis only. It does not launch vLLM and does not claim that
future autoregressive decode steps can be coalesced safely.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(round(number))


def _fmt(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _fmt_speedup(value: Any) -> str:
    rendered = _fmt(value)
    return "" if not rendered else f"{rendered}x"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _speedup(dense_ms: float | None, candidate_ms: float | None) -> float | None:
    if dense_ms is None or candidate_ms is None or candidate_ms <= 0.0:
        return None
    return dense_ms / candidate_ms


def _row_key(row: dict[str, str]) -> tuple[int, int]:
    batch = _as_int(row.get("batch_size"))
    prefix = _as_int(row.get("prefix"))
    if batch is None or prefix is None:
        raise ValueError(f"missing batch/prefix in row: {row}")
    return batch, prefix


def derive_rows(rows: list[dict[str, str]], *, target_speedup: float) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for row in rows:
        dense_ms = _as_float(row.get("dense_graph_ms")) or _as_float(row.get("dense_ms"))
        serial_dense_ms = _as_float(
            row.get("serial_dense_reference_graph_ms")
        ) or _as_float(row.get("serial_dense_reference_ms"))
        packed_ms = _as_float(row.get("packed_graph_ms")) or _as_float(
            row.get("packed_ms")
        )
        parallel_ms = _as_float(row.get("packed_parallel_graph_ms")) or _as_float(
            row.get("packed_parallel_ms")
        )
        padded_ms = _as_float(
            row.get("packed_capacity_padded_graph_ms")
        ) or _as_float(row.get("packed_capacity_padded_ms"))
        best_name = "dense_fallback"
        best_ms = dense_ms
        mixed_best_name = ""
        mixed_best_ms = None
        for name, candidate_ms in (
            ("packed", packed_ms),
            ("packed_parallel", parallel_ms),
            ("packed_padded", padded_ms),
        ):
            if candidate_ms is None:
                continue
            if mixed_best_ms is None or candidate_ms < mixed_best_ms:
                mixed_best_name = name
                mixed_best_ms = candidate_ms
            if best_ms is None or candidate_ms < best_ms:
                best_name = name
                best_ms = candidate_ms
        best_speedup = _speedup(dense_ms, best_ms)
        mixed_best_speedup = _speedup(dense_ms, mixed_best_ms)
        mixed_best_serial_speedup = _speedup(serial_dense_ms, mixed_best_ms)
        parallel_speedup = _speedup(dense_ms, parallel_ms)
        padded_speedup = _speedup(dense_ms, padded_ms)
        parallel_serial_speedup = _speedup(serial_dense_ms, parallel_ms)
        padded_serial_speedup = _speedup(serial_dense_ms, padded_ms)
        derived.append({
            "batch_size": _as_int(row.get("batch_size")),
            "prefix": _as_int(row.get("prefix")),
            "coalesce_factor": _as_int(row.get("coalesce_factor")) or 1,
            "effective_batch_size": _as_int(row.get("effective_batch_size")),
            "k": _as_int(row.get("k")),
            "dense_rows": _as_int(row.get("dense_rows")),
            "base_rows": _as_int(row.get("base_rows")),
            "dense_capacity": _as_int(row.get("dense_capacity")),
            "base_capacity": _as_int(row.get("base_capacity")),
            "dense_fill": _as_float(row.get("dense_capacity_fill")),
            "base_fill": _as_float(row.get("base_capacity_fill")),
            "dense_graph_ms": dense_ms,
            "serial_dense_graph_ms": serial_dense_ms,
            "packed_parallel_graph_ms": parallel_ms,
            "packed_padded_graph_ms": padded_ms,
            "best_operator": best_name,
            "best_graph_ms": best_ms,
            "best_speedup_vs_dense": best_speedup,
            "mixed_best_operator": mixed_best_name,
            "mixed_best_graph_ms": mixed_best_ms,
            "mixed_best_speedup_vs_dense": mixed_best_speedup,
            "mixed_best_speedup_vs_serial_dense": mixed_best_serial_speedup,
            "parallel_speedup_vs_dense": parallel_speedup,
            "padded_speedup_vs_dense": padded_speedup,
            "parallel_speedup_vs_serial_dense": parallel_serial_speedup,
            "padded_speedup_vs_serial_dense": padded_serial_speedup,
            "meets_target": (
                mixed_best_speedup is not None
                and mixed_best_speedup >= target_speedup
            ),
            "planner_action": (
                "use_mixed"
                if mixed_best_speedup is not None
                and mixed_best_speedup >= target_speedup
                else "dense_fallback_or_group_more"
            ),
        })
    return derived


def derive_grouping(derived: list[dict[str, Any]], *, target_speedup: float) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in derived:
        batch = row.get("batch_size")
        prefix = row.get("prefix")
        if batch is None or prefix is None:
            continue
        by_key[(int(batch), int(prefix))].append(row)

    output: list[dict[str, Any]] = []
    for (batch, prefix), group in sorted(by_key.items()):
        group = sorted(group, key=lambda item: int(item.get("coalesce_factor") or 1))
        winner = None
        serial_winner = None
        for row in group:
            speedup = row.get("mixed_best_speedup_vs_dense")
            if speedup is not None and float(speedup) >= target_speedup:
                winner = row
                break
        for row in group:
            speedup = row.get("mixed_best_speedup_vs_serial_dense")
            if speedup is not None and float(speedup) >= target_speedup:
                serial_winner = row
                break
        best_observed = max(
            group,
            key=lambda item: float(item.get("best_speedup_vs_dense") or 0.0),
        )
        output.append({
            "batch_size": batch,
            "prefix": prefix,
            "target_speedup": target_speedup,
            "required_coalesce_factor": (
                winner.get("coalesce_factor") if winner is not None else ""
            ),
            "required_effective_batch_size": (
                winner.get("effective_batch_size") if winner is not None else ""
            ),
            "required_operator": winner.get("mixed_best_operator") if winner else "",
            "required_speedup": (
                winner.get("mixed_best_speedup_vs_dense") if winner else ""
            ),
            "serial_required_coalesce_factor": (
                serial_winner.get("coalesce_factor")
                if serial_winner is not None
                else ""
            ),
            "serial_required_effective_batch_size": (
                serial_winner.get("effective_batch_size")
                if serial_winner is not None
                else ""
            ),
            "serial_required_operator": (
                serial_winner.get("mixed_best_operator") if serial_winner else ""
            ),
            "serial_required_speedup": (
                serial_winner.get("mixed_best_speedup_vs_serial_dense")
                if serial_winner
                else ""
            ),
            "best_observed_coalesce_factor": best_observed.get("coalesce_factor"),
            "best_observed_effective_batch_size": best_observed.get(
                "effective_batch_size"
            ),
            "best_observed_operator": best_observed.get("best_operator"),
            "best_observed_speedup": best_observed.get("best_speedup_vs_dense"),
            "planner_read": (
                "mixed_ready"
                if winner is not None
                else "needs_more_grouping_or_dense_fallback"
            ),
        })
    return output


def derive_batch_policy(
    per_row: list[dict[str, Any]],
    grouping: list[dict[str, Any]],
    *,
    target_speedup: float,
    policy_prefix: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize the live policy implication per original batch size.

    `grouping` answers the question for a fixed prefix. This reducer answers the
    next scheduler question: with the current batch size, is any single-block
    prefix profitable, and if not, what is the smallest useful-row coalescing
    factor over all prefixes that would make a mixed operator locally viable?
    """
    rows_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in per_row:
        batch = row.get("batch_size")
        if batch is None:
            continue
        rows_by_batch[int(batch)].append(row)

    grouping_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in grouping:
        batch = row.get("batch_size")
        if batch is None:
            continue
        grouping_by_batch[int(batch)].append(row)

    output: list[dict[str, Any]] = []
    for batch, rows in sorted(rows_by_batch.items()):
        if policy_prefix is not None:
            rows = [
                row for row in rows
                if _as_int(row.get("prefix")) == int(policy_prefix)
            ]
            if not rows:
                continue
        single_rows = [
            row for row in rows if int(row.get("coalesce_factor") or 1) == 1
        ]
        best_single = max(
            single_rows or rows,
            key=lambda item: float(item.get("mixed_best_speedup_vs_dense") or 0.0),
        )
        ready_group_rows = [
            row for row in grouping_by_batch.get(batch, [])
            if row.get("required_coalesce_factor") not in (None, "")
            and (
                policy_prefix is None
                or _as_int(row.get("prefix")) == int(policy_prefix)
            )
        ]
        best_group = None
        if ready_group_rows:
            best_group = min(
                ready_group_rows,
                key=lambda item: (
                    int(item.get("required_coalesce_factor") or 10**9),
                    int(item.get("prefix") or 10**9),
                ),
            )
        single_speedup = best_single.get("mixed_best_speedup_vs_dense")
        single_ready = (
            single_speedup is not None and float(single_speedup) >= target_speedup
        )
        action = "use_mixed_single_block" if single_ready else "dense_fallback"
        if not single_ready and best_group is not None:
            action = "dense_fallback_until_grouped"
        output.append({
            "batch_size": batch,
            "target_speedup": target_speedup,
            "best_single_prefix": best_single.get("prefix"),
            "best_single_effective_batch_size": best_single.get(
                "effective_batch_size"
            ),
            "best_single_dense_rows": best_single.get("dense_rows"),
            "best_single_base_rows": best_single.get("base_rows"),
            "best_single_operator": best_single.get("mixed_best_operator"),
            "best_single_mixed_local_speedup": single_speedup,
            "required_group_prefix": (
                best_group.get("prefix") if best_group is not None else ""
            ),
            "required_coalesce_factor": (
                best_group.get("required_coalesce_factor")
                if best_group is not None
                else ""
            ),
            "required_effective_batch_size": (
                best_group.get("required_effective_batch_size")
                if best_group is not None
                else ""
            ),
            "required_operator": (
                best_group.get("required_operator")
                if best_group is not None
                else ""
            ),
            "required_mixed_local_speedup": (
                best_group.get("required_speedup")
                if best_group is not None
                else ""
            ),
            "planner_action": action,
        })
    return output


def derive_scheduler_policy(
    per_row: list[dict[str, Any]],
    batch_policy: list[dict[str, Any]],
    *,
    target_speedup: float,
    source_csv: Path,
    policy_prefix: int | None = None,
) -> dict[str, Any]:
    """Build a machine-readable policy for the live grouped-operator design.

    The policy deliberately separates an operator-local decision from a live
    scheduling claim.  A row with `dense_fallback_until_grouped` means the
    current single verifier block is not profitable, but the microbench has an
    operator-local point that would be profitable if the scheduler can safely
    group that many useful verifier blocks with the same module weights.
    """
    rows_by_shape: dict[tuple[int, int, int], dict[str, Any]] = {}
    single_by_shape: dict[tuple[int, int], dict[str, Any]] = {}
    for row in per_row:
        batch = row.get("batch_size")
        prefix = row.get("prefix")
        coalesce = row.get("coalesce_factor")
        if batch is None or prefix is None or coalesce is None:
            continue
        key = (int(batch), int(prefix), int(coalesce))
        rows_by_shape[key] = row
        if int(coalesce) == 1:
            single_by_shape[(int(batch), int(prefix))] = row

    policies: list[dict[str, Any]] = []
    for row in batch_policy:
        batch = int(row.get("batch_size"))
        action = str(row.get("planner_action") or "dense_fallback")
        if action == "use_mixed_single_block":
            prefix = _as_int(row.get("best_single_prefix"))
            coalesce = 1
        elif action == "dense_fallback_until_grouped":
            prefix = _as_int(row.get("required_group_prefix"))
            coalesce = _as_int(row.get("required_coalesce_factor"))
        else:
            prefix = _as_int(row.get("best_single_prefix"))
            coalesce = None

        selected = (
            rows_by_shape.get((batch, int(prefix), int(coalesce)))
            if prefix is not None and coalesce is not None
            else None
        )
        single = (
            single_by_shape.get((batch, int(prefix)))
            if prefix is not None
            else None
        )
        dense_rows_per_block = None
        base_rows_per_block = None
        if single is not None:
            dense_rows_per_block = single.get("dense_rows")
            base_rows_per_block = single.get("base_rows")
        elif selected is not None and coalesce:
            dense_rows = selected.get("dense_rows")
            base_rows = selected.get("base_rows")
            dense_rows_per_block = (
                int(dense_rows) // int(coalesce)
                if dense_rows not in (None, "") else None
            )
            base_rows_per_block = (
                int(base_rows) // int(coalesce)
                if base_rows not in (None, "") else None
            )

        policies.append({
            "batch_size": batch,
            "planner_action": action,
            "target_speedup": target_speedup,
            "prefix": prefix,
            "k": selected.get("k") if selected is not None else (
                single.get("k") if single is not None else None
            ),
            "fallback_when_underfilled": "dense",
            "min_grouped_verifier_blocks": coalesce,
            "target_effective_batch_size": (
                selected.get("effective_batch_size")
                if selected is not None else ""
            ),
            "mixed_operator": (
                selected.get("mixed_best_operator")
                if selected is not None else ""
            ),
            "mixed_local_speedup_vs_dense": (
                selected.get("mixed_best_speedup_vs_dense")
                if selected is not None else ""
            ),
            "dense_rows_per_verifier_block": dense_rows_per_block,
            "base_rows_per_verifier_block": base_rows_per_block,
            "grouped_dense_rows": (
                selected.get("dense_rows") if selected is not None else ""
            ),
            "grouped_base_rows": (
                selected.get("base_rows") if selected is not None else ""
            ),
            "dense_capacity": (
                selected.get("dense_capacity") if selected is not None else ""
            ),
            "base_capacity": (
                selected.get("base_capacity") if selected is not None else ""
            ),
        })

    return {
        "schema_version": 1,
        "source_summary_csv": str(source_csv),
        "target_speedup": target_speedup,
        "policy_prefix": policy_prefix if policy_prefix is not None else "",
        "default_action": "dense_fallback",
        "scope": "operator_local_policy_not_live_serving_claim",
        "dependency_constraints": [
            "Only group verifier blocks that use the same module weights.",
            "Do not change decode dependencies; grouping is valid only after all grouped inputs are ready.",
            "Use dense fallback when the minimum useful grouped blocks are not available within the scheduler latency budget.",
            "The policy assumes fixed-prefix route descriptors and disjoint dense-important / 2:4-sparse rows.",
        ],
        "policies": policies,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_outputs(
    output_root: Path,
    per_row: list[dict[str, Any]],
    grouping: list[dict[str, Any]],
    batch_policy: list[dict[str, Any]],
    scheduler_policy: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "operator_planner_rows.csv", per_row)
    _write_csv(output_root / "operator_planner_grouping.csv", grouping)
    _write_csv(output_root / "operator_planner_batch_policy.csv", batch_policy)
    with (output_root / "scheduler_policy.json").open("w", encoding="utf-8") as f:
        json.dump(scheduler_policy, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_root / "operator_planner.md").open("w", encoding="utf-8") as f:
        f.write("# SR24 Packed MLP Operator Planner\n\n")
        f.write(
            "This offline planner reads packed verifier-block microbench rows. "
            "It does not prove decode-step coalescing is dependency-safe; it "
            "only states what row fill would be needed for the operator itself.\n\n"
        )
        policy_prefix = scheduler_policy.get("policy_prefix")
        if policy_prefix not in (None, ""):
            f.write(
                f"Policy rows are constrained to fixed prefix `{policy_prefix}` "
                "to match one live fixed-prefix server configuration. Treat this "
                "as quality-safe only if a separate accuracy gate has validated "
                "that prefix.\n\n"
            )
        f.write("## Per-Row Decision\n\n")
        f.write(
            "| bs | prefix | coalesce | effective bs | dense rows | base rows | "
            "planner op | mixed op | mixed local speedup | "
            "mixed serial speedup | dense fill | "
            "base fill | action |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|\n")
        for row in per_row:
            f.write(
                f"| {row.get('batch_size')} | {row.get('prefix')} | "
                f"{row.get('coalesce_factor')} | "
                f"{row.get('effective_batch_size')} | {row.get('dense_rows')} | "
                f"{row.get('base_rows')} | {row.get('best_operator')} | "
                f"{row.get('mixed_best_operator')} | "
                f"{_fmt_speedup(row.get('mixed_best_speedup_vs_dense'))} | "
                f"{_fmt_speedup(row.get('mixed_best_speedup_vs_serial_dense'))} | "
                f"{_fmt(row.get('dense_fill'))} | {_fmt(row.get('base_fill'))} | "
                f"{row.get('planner_action')} |\n"
            )
        f.write("\n## Required Useful-Row Grouping\n\n")
        f.write(
            "| bs | prefix | local required coalesce | local effective bs | "
            "local operator | local speedup | serial required coalesce | "
            "serial effective bs | serial operator | serial speedup | read |\n"
        )
        f.write("|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---|\n")
        for row in grouping:
            f.write(
                f"| {row.get('batch_size')} | {row.get('prefix')} | "
                f"{row.get('required_coalesce_factor')} | "
                f"{row.get('required_effective_batch_size')} | "
                f"{row.get('required_operator')} | "
                f"{_fmt_speedup(row.get('required_speedup'))} | "
                f"{row.get('serial_required_coalesce_factor')} | "
                f"{row.get('serial_required_effective_batch_size')} | "
                f"{row.get('serial_required_operator')} | "
                f"{_fmt_speedup(row.get('serial_required_speedup'))} | "
                f"{row.get('planner_read')} |\n"
            )
        f.write("\n## Batch Policy\n\n")
        f.write(
            "| bs | best single prefix | best single speedup | "
            "required group prefix | required coalesce | required effective bs | "
            "required speedup | action |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in batch_policy:
            f.write(
                f"| {row.get('batch_size')} | "
                f"{row.get('best_single_prefix')} | "
                f"{_fmt_speedup(row.get('best_single_mixed_local_speedup'))} | "
                f"{row.get('required_group_prefix')} | "
                f"{row.get('required_coalesce_factor')} | "
                f"{row.get('required_effective_batch_size')} | "
                f"{_fmt_speedup(row.get('required_mixed_local_speedup'))} | "
                f"{row.get('planner_action')} |\n"
            )
        f.write("\n## Scheduler Policy JSON\n\n")
        f.write(
            "A machine-readable policy is written to `scheduler_policy.json`. "
            "It is intentionally scoped as an operator-local policy, not a live "
            "serving speed claim. Rows with `dense_fallback_until_grouped` mean "
            "the live scheduler should use dense fallback unless it can group "
            "the listed number of ready verifier blocks for the same module "
            "weights without violating decode dependencies or latency budget.\n"
        )
        f.write("\n## Interpretation\n\n")
        f.write(
            "- `local speedup` compares the mixed dense/sparse operator against a "
            "dense operator over the same grouped fixed bucket. This is the "
            "operator-level requirement for replacing a dense verifier MLP.\n"
        )
        f.write(
            "- `serial speedup` compares the grouped mixed operator against "
            "running the original ungrouped dense block repeatedly. This is an "
            "optimistic scheduler upper bound; it is not valid unless the live "
            "scheduler can safely coalesce those useful rows without changing "
            "decode dependencies or latency goals.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive SR24 packed MLP planner hints from a microbench CSV."
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--target-speedup", type=float, default=1.2)
    parser.add_argument(
        "--policy-prefix",
        type=int,
        default=-1,
        help=(
            "If non-negative, derive the batch/scheduler policy using only "
            "this fixed prefix. Use this when quality evidence requires a "
            "specific prefix even if another prefix is faster in the "
            "microbench."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    summary_csv = args.summary_csv.resolve()
    output_root = args.output_root or summary_csv.parent
    rows = _load_rows(summary_csv)
    per_row = derive_rows(rows, target_speedup=args.target_speedup)
    grouping = derive_grouping(per_row, target_speedup=args.target_speedup)
    policy_prefix = args.policy_prefix if args.policy_prefix >= 0 else None
    batch_policy = derive_batch_policy(
        per_row,
        grouping,
        target_speedup=args.target_speedup,
        policy_prefix=policy_prefix,
    )
    scheduler_policy = derive_scheduler_policy(
        per_row,
        batch_policy,
        target_speedup=args.target_speedup,
        source_csv=summary_csv,
        policy_prefix=policy_prefix,
    )
    write_outputs(output_root, per_row, grouping, batch_policy, scheduler_policy)
    print((output_root / "operator_planner.md").resolve())


if __name__ == "__main__":
    main()
