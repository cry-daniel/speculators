#!/usr/bin/env python3
"""Summarize SR24 residual routing against actual verifier acceptance labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "accepted_local" not in row:
                continue
            rows.append(row)
    return rows


def _key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("run_id", "unknown")),
        str(row.get("model_label", "unknown")),
        str(row.get("dataset_label", "unknown")),
        str(row.get("method", "unknown")),
        str(row.get("sr24_policy", "unknown")),
    )


def _ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def _requested_residual(row: dict[str, Any]) -> bool:
    return row.get("sr24_uses_residual") == 1


def _effective_residual(row: dict[str, Any]) -> bool:
    value = row.get("sr24_effective_residual")
    if value is None:
        return _requested_residual(row)
    return value == 1


def _score_value(row: dict[str, Any]) -> tuple[bool, float]:
    score = row.get("draft_selected_prob")
    if score is None:
        return False, float("nan")
    try:
        return True, float(score)
    except (TypeError, ValueError):
        return False, float("nan")


def _group_steps(
    group: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    steps: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        steps[(str(row.get("request_id")), int(row.get("step_id") or -1))].append(row)
    for step_rows in steps.values():
        step_rows.sort(key=lambda row: int(row.get("draft_position") or 0))
    return steps


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        reached = [row for row in group if row.get("reached") == 1]
        unreached = [row for row in group if row.get("reached") != 1]
        accepted = [row for row in group if row.get("accepted_local") == 1]
        rejected = [row for row in group if row.get("accepted_local") == 0]
        residual = [row for row in group if _requested_residual(row)]
        base_only = [row for row in group if not _requested_residual(row)]
        effective_residual = [row for row in group if _effective_residual(row)]
        effective_base = [row for row in group if not _effective_residual(row)]
        reached_base = [row for row in reached if not _requested_residual(row)]
        reached_effective_base = [
            row for row in reached if not _effective_residual(row)
        ]
        unreached_effective_base = [
            row for row in unreached if not _effective_residual(row)
        ]
        accepted_base = [row for row in accepted if not _requested_residual(row)]
        accepted_effective_base = [
            row for row in accepted if not _effective_residual(row)
        ]
        rejected_base = [row for row in rejected if not _requested_residual(row)]
        accepted_residual = [row for row in accepted if _requested_residual(row)]
        accepted_effective_residual = [
            row for row in accepted if _effective_residual(row)
        ]

        steps = _group_steps(group)
        step_count = len(steps)
        steps_with_reached_base = 0
        steps_with_unreached_effective_base = 0
        steps_with_accepted_base = 0
        steps_with_reject_base = 0
        accepted_counts: list[int] = []
        residual_counts: list[int] = []
        reached_base_counts: list[int] = []
        unreached_effective_base_counts: list[int] = []
        for step_rows in steps.values():
            accepted_counts.append(
                max(int(row.get("num_accepted_in_step") or 0) for row in step_rows)
            )
            residual_counts.append(
                sum(1 for row in step_rows if _requested_residual(row))
            )
            reached_base_count = sum(
                1
                for row in step_rows
                if row.get("reached") == 1 and not _requested_residual(row)
            )
            reached_base_counts.append(reached_base_count)
            unreached_effective_base_count = sum(
                1
                for row in step_rows
                if row.get("reached") != 1 and not _effective_residual(row)
            )
            unreached_effective_base_counts.append(unreached_effective_base_count)
            if reached_base_count:
                steps_with_reached_base += 1
            if unreached_effective_base_count:
                steps_with_unreached_effective_base += 1
            if any(
                row.get("accepted_local") == 1 and not _requested_residual(row)
                for row in step_rows
            ):
                steps_with_accepted_base += 1
            if any(
                row.get("accepted_local") == 0 and not _requested_residual(row)
                for row in step_rows
            ):
                steps_with_reject_base += 1

        score_bins = Counter()
        for row in accepted_base:
            score = row.get("draft_selected_prob")
            if score is None:
                score_bins["missing"] += 1
            else:
                score_bins[f"{int(float(score) * 10) / 10:.1f}"] += 1
        effective_score_bins = Counter()
        for row in accepted_effective_base:
            score = row.get("draft_selected_prob")
            if score is None:
                effective_score_bins["missing"] += 1
            else:
                effective_score_bins[f"{int(float(score) * 10) / 10:.1f}"] += 1

        run_id, model_label, dataset_label, method, policy = key
        out.append(
            {
                "run_id": run_id,
                "model_label": model_label,
                "dataset_label": dataset_label,
                "method": method,
                "sr24_policy": policy,
                "tokens": len(group),
                "reached_tokens": len(reached),
                "unreached_tokens": len(unreached),
                "accepted_tokens": len(accepted),
                "rejected_tokens": len(rejected),
                "sr24_residual_tokens": len(residual),
                "sr24_base_only_tokens": len(base_only),
                "sr24_effective_residual_tokens": len(effective_residual),
                "sr24_effective_base_only_tokens": len(effective_base),
                "reached_base_only_tokens": len(reached_base),
                "reached_effective_base_only_tokens": len(reached_effective_base),
                "unreached_effective_base_only_tokens":
                len(unreached_effective_base),
                "accepted_base_only_tokens": len(accepted_base),
                "accepted_effective_base_only_tokens": len(accepted_effective_base),
                "accepted_residual_tokens": len(accepted_residual),
                "accepted_effective_residual_tokens":
                len(accepted_effective_residual),
                "rejected_base_only_tokens": len(rejected_base),
                "sr24_residual_fraction": _ratio(len(residual), len(group)),
                "sr24_effective_residual_fraction":
                _ratio(len(effective_residual), len(group)),
                "reached_base_only_fraction": _ratio(len(reached_base), len(reached)),
                "reached_effective_base_only_fraction":
                _ratio(len(reached_effective_base), len(reached)),
                "unreached_effective_base_only_fraction":
                _ratio(len(unreached_effective_base), len(unreached)),
                "accepted_base_only_fraction": _ratio(len(accepted_base), len(accepted)),
                "accepted_effective_base_only_fraction":
                _ratio(len(accepted_effective_base), len(accepted)),
                "rejected_base_only_fraction": _ratio(len(rejected_base), len(rejected)),
                "steps": step_count,
                "steps_with_reached_base_only": steps_with_reached_base,
                "steps_with_unreached_effective_base_only":
                steps_with_unreached_effective_base,
                "steps_with_accepted_base_only": steps_with_accepted_base,
                "steps_with_rejected_base_only": steps_with_reject_base,
                "steps_with_reached_base_only_fraction":
                _ratio(steps_with_reached_base, step_count),
                "steps_with_unreached_effective_base_only_fraction":
                _ratio(steps_with_unreached_effective_base, step_count),
                "steps_with_accepted_base_only_fraction":
                _ratio(steps_with_accepted_base, step_count),
                "steps_with_rejected_base_only_fraction":
                _ratio(steps_with_reject_base, step_count),
                "mean_accepted_tokens_per_step":
                _ratio(sum(accepted_counts), len(accepted_counts)),
                "mean_sr24_residual_rows_per_step":
                _ratio(sum(residual_counts), len(residual_counts)),
                "mean_reached_base_rows_per_step":
                _ratio(sum(reached_base_counts), len(reached_base_counts)),
                "mean_unreached_effective_base_rows_per_step":
                _ratio(
                    sum(unreached_effective_base_counts),
                    len(unreached_effective_base_counts),
                ),
                "accepted_base_score_bins": json.dumps(
                    dict(sorted(score_bins.items())), sort_keys=True
                ),
                "accepted_effective_base_score_bins": json.dumps(
                    dict(sorted(effective_score_bins.items())), sort_keys=True
                ),
            }
        )
    return out


def summarize_by_position(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        key = (*_key(row), int(row.get("draft_position") or -1))
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        run_id, model_label, dataset_label, method, policy, draft_position = key
        reached = [row for row in group if row.get("reached") == 1]
        unreached = [row for row in group if row.get("reached") != 1]
        accepted = [row for row in group if row.get("accepted_local") == 1]
        rejected = [row for row in group if row.get("accepted_local") == 0]
        residual = [row for row in group if row.get("sr24_uses_residual") == 1]
        accepted_base = [
            row for row in accepted if row.get("sr24_uses_residual") == 0
        ]
        reached_base = [
            row for row in reached if row.get("sr24_uses_residual") == 0
        ]
        unreached_base = [
            row for row in unreached if row.get("sr24_uses_residual") == 0
        ]
        out.append(
            {
                "run_id": run_id,
                "model_label": model_label,
                "dataset_label": dataset_label,
                "method": method,
                "sr24_policy": policy,
                "draft_position": draft_position,
                "tokens": len(group),
                "reached_tokens": len(reached),
                "unreached_tokens": len(unreached),
                "accepted_tokens": len(accepted),
                "rejected_tokens": len(rejected),
                "sr24_residual_tokens": len(residual),
                "accepted_base_only_tokens": len(accepted_base),
                "reached_base_only_tokens": len(reached_base),
                "unreached_base_only_tokens": len(unreached_base),
                "sr24_residual_fraction": _ratio(len(residual), len(group)),
                "accepted_rate": _ratio(len(accepted), len(group)),
                "accepted_base_only_fraction":
                _ratio(len(accepted_base), len(accepted)),
                "reached_base_only_fraction": _ratio(len(reached_base), len(reached)),
                "unreached_base_only_fraction":
                _ratio(len(unreached_base), len(unreached)),
            }
        )
    return out


def summarize_prefix_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project quality-risk counters if the first N draft positions are dense.

    This is an offline upper-bound analysis over an existing trace: for a
    candidate prefix N, a token is treated as residual if it was already
    residual in the trace or if draft_position <= N. It does not predict
    changed acceptance after changing the verifier path, but it directly shows
    how much accepted base-only risk remains and how many residual rows the
    policy would add.
    """
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    max_position = 0
    for row in rows:
        grouped[_key(row)].append(row)
        try:
            max_position = max(max_position, int(row.get("draft_position") or 0))
        except (TypeError, ValueError):
            pass

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        run_id, model_label, dataset_label, method, policy = key
        steps = _group_steps(group)
        step_count = len(steps)
        for prefix_len in range(0, max_position + 1):
            projected_residual = 0
            projected_base = 0
            accepted = 0
            accepted_base = 0
            reached = 0
            reached_base = 0
            unreached = 0
            unreached_base = 0
            rejected = 0
            rejected_base = 0
            steps_with_accepted_base = 0
            steps_with_rejected_base = 0
            steps_with_unreached_base = 0
            residual_counts: list[int] = []
            accepted_counts: list[int] = []
            for step_rows in steps.values():
                step_accepted_base = 0
                step_rejected_base = 0
                step_unreached_base = 0
                step_residual = 0
                accepted_counts.append(
                    max(int(row.get("num_accepted_in_step") or 0) for row in step_rows)
                )
                for row in step_rows:
                    pos = int(row.get("draft_position") or 0)
                    uses_residual = (
                        row.get("sr24_uses_residual") == 1
                        or (prefix_len > 0 and pos <= prefix_len)
                    )
                    if uses_residual:
                        projected_residual += 1
                        step_residual += 1
                    else:
                        projected_base += 1
                    if row.get("reached") == 1:
                        reached += 1
                        if not uses_residual:
                            reached_base += 1
                    else:
                        unreached += 1
                        if not uses_residual:
                            unreached_base += 1
                            step_unreached_base += 1
                    if row.get("accepted_local") == 1:
                        accepted += 1
                        if not uses_residual:
                            accepted_base += 1
                            step_accepted_base += 1
                    if row.get("accepted_local") == 0:
                        rejected += 1
                        if not uses_residual:
                            rejected_base += 1
                            step_rejected_base += 1
                residual_counts.append(step_residual)
                if step_accepted_base:
                    steps_with_accepted_base += 1
                if step_rejected_base:
                    steps_with_rejected_base += 1
                if step_unreached_base:
                    steps_with_unreached_base += 1
            out.append(
                {
                    "run_id": run_id,
                    "model_label": model_label,
                    "dataset_label": dataset_label,
                    "method": method,
                    "sr24_policy": policy,
                    "projected_prefix_residual_len": prefix_len,
                    "tokens": len(group),
                    "projected_residual_tokens": projected_residual,
                    "projected_base_only_tokens": projected_base,
                    "accepted_tokens": accepted,
                    "accepted_base_only_tokens": accepted_base,
                    "reached_tokens": reached,
                    "reached_base_only_tokens": reached_base,
                    "unreached_tokens": unreached,
                    "unreached_base_only_tokens": unreached_base,
                    "rejected_tokens": rejected,
                    "rejected_base_only_tokens": rejected_base,
                    "projected_residual_fraction":
                    _ratio(projected_residual, len(group)),
                    "accepted_base_only_fraction":
                    _ratio(accepted_base, accepted),
                    "reached_base_only_fraction": _ratio(reached_base, reached),
                    "unreached_base_only_fraction":
                    _ratio(unreached_base, unreached),
                    "rejected_base_only_fraction": _ratio(rejected_base, rejected),
                    "steps": step_count,
                    "steps_with_accepted_base_only": steps_with_accepted_base,
                    "steps_with_rejected_base_only": steps_with_rejected_base,
                    "steps_with_unreached_base_only": steps_with_unreached_base,
                    "steps_with_accepted_base_only_fraction":
                    _ratio(steps_with_accepted_base, step_count),
                    "steps_with_rejected_base_only_fraction":
                    _ratio(steps_with_rejected_base, step_count),
                    "steps_with_unreached_base_only_fraction":
                    _ratio(steps_with_unreached_base, step_count),
                    "mean_accepted_tokens_per_step":
                    _ratio(sum(accepted_counts), len(accepted_counts)),
                    "mean_projected_residual_rows_per_step":
                    _ratio(sum(residual_counts), len(residual_counts)),
                }
            )
    return out


def summarize_score_policy_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project residual routing from draft score thresholds.

    The current speed/quality conflict is about accepted base-only tokens. This
    projection tests a risk-oriented rule: make draft rows residual when their
    DLM selected-token probability is high enough to be likely accepted, plus
    an optional dense prefix. It is intentionally offline and uses the traced
    acceptance labels only for measuring residual/base-only risk. The
    threshold comparison mirrors runtime `high_confidence`: score > threshold.
    """
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_key(row)].append(row)

    thresholds = [round(i / 10, 1) for i in range(0, 11)]
    prefixes = list(range(0, 5))
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        run_id, model_label, dataset_label, method, policy = key
        steps = _group_steps(group)
        step_count = len(steps)
        for prefix_len in prefixes:
            for cutoff in thresholds:
                projected_residual = 0
                accepted = 0
                accepted_base = 0
                reached = 0
                reached_base = 0
                unreached = 0
                unreached_base = 0
                rejected = 0
                rejected_base = 0
                missing_scores = 0
                steps_with_accepted_base = 0
                steps_with_rejected_base = 0
                steps_with_unreached_base = 0
                residual_counts: list[int] = []
                accepted_counts: list[int] = []
                for step_rows in steps.values():
                    step_accepted_base = 0
                    step_rejected_base = 0
                    step_unreached_base = 0
                    step_residual = 0
                    accepted_counts.append(
                        max(int(row.get("num_accepted_in_step") or 0)
                            for row in step_rows)
                    )
                    for row in step_rows:
                        pos = int(row.get("draft_position") or 0)
                        score_present, score_value = _score_value(row)
                        if not score_present:
                            missing_scores += 1
                        uses_residual = (
                            (prefix_len > 0 and pos <= prefix_len)
                            or (score_present and score_value > cutoff)
                            or not score_present
                        )
                        if uses_residual:
                            projected_residual += 1
                            step_residual += 1
                        if row.get("reached") == 1:
                            reached += 1
                            if not uses_residual:
                                reached_base += 1
                        else:
                            unreached += 1
                            if not uses_residual:
                                unreached_base += 1
                                step_unreached_base += 1
                        if row.get("accepted_local") == 1:
                            accepted += 1
                            if not uses_residual:
                                accepted_base += 1
                                step_accepted_base += 1
                        if row.get("accepted_local") == 0:
                            rejected += 1
                            if not uses_residual:
                                rejected_base += 1
                                step_rejected_base += 1
                    residual_counts.append(step_residual)
                    if step_accepted_base:
                        steps_with_accepted_base += 1
                    if step_rejected_base:
                        steps_with_rejected_base += 1
                    if step_unreached_base:
                        steps_with_unreached_base += 1
                out.append(
                    {
                        "run_id": run_id,
                        "model_label": model_label,
                        "dataset_label": dataset_label,
                        "method": method,
                        "source_sr24_policy": policy,
                        "projected_policy": "high_confidence_plus_prefix",
                        "projected_prefix_residual_len": prefix_len,
                        "projected_score_threshold": cutoff,
                        "tokens": len(group),
                        "missing_score_tokens": missing_scores,
                        "projected_residual_tokens": projected_residual,
                        "projected_residual_fraction":
                        _ratio(projected_residual, len(group)),
                        "accepted_tokens": accepted,
                        "accepted_base_only_tokens": accepted_base,
                        "accepted_base_only_fraction":
                        _ratio(accepted_base, accepted),
                        "reached_tokens": reached,
                        "reached_base_only_tokens": reached_base,
                        "reached_base_only_fraction": _ratio(reached_base, reached),
                        "unreached_tokens": unreached,
                        "unreached_base_only_tokens": unreached_base,
                        "unreached_base_only_fraction":
                        _ratio(unreached_base, unreached),
                        "rejected_tokens": rejected,
                        "rejected_base_only_tokens": rejected_base,
                        "rejected_base_only_fraction": _ratio(rejected_base, rejected),
                        "steps": step_count,
                        "steps_with_accepted_base_only": steps_with_accepted_base,
                        "steps_with_rejected_base_only": steps_with_rejected_base,
                        "steps_with_unreached_base_only": steps_with_unreached_base,
                        "steps_with_accepted_base_only_fraction":
                        _ratio(steps_with_accepted_base, step_count),
                        "steps_with_rejected_base_only_fraction":
                        _ratio(steps_with_rejected_base, step_count),
                        "steps_with_unreached_base_only_fraction":
                        _ratio(steps_with_unreached_base, step_count),
                        "mean_accepted_tokens_per_step":
                        _ratio(sum(accepted_counts), len(accepted_counts)),
                        "mean_projected_residual_rows_per_step":
                        _ratio(sum(residual_counts), len(residual_counts)),
                    }
                )
    return out


def summarize_critical_prefix_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project runtime `critical_prefix` routing over a completed trace.

    Runtime critical-prefix routing corrects all high-confidence prefix rows
    until the first low-confidence row, includes that first low-confidence row,
    and optionally includes `extra_after_low` more rows. If no low-confidence
    row exists, the whole draft is treated as residual because the complete
    prefix is likely to be accepted.
    """
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    max_position = 0
    for row in rows:
        grouped[_key(row)].append(row)
        try:
            max_position = max(max_position, int(row.get("draft_position") or 0))
        except (TypeError, ValueError):
            pass

    thresholds = [round(i / 10, 1) for i in range(0, 11)]
    prefixes = list(range(0, max_position + 1))
    extras = list(range(0, min(max_position, 4) + 1))
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        run_id, model_label, dataset_label, method, policy = key
        steps = _group_steps(group)
        step_count = len(steps)
        for prefix_len in prefixes:
            for extra_after_low in extras:
                for cutoff in thresholds:
                    projected_residual = 0
                    accepted = 0
                    accepted_base = 0
                    reached = 0
                    reached_base = 0
                    unreached = 0
                    unreached_base = 0
                    rejected = 0
                    rejected_base = 0
                    missing_scores = 0
                    steps_with_accepted_base = 0
                    steps_with_rejected_base = 0
                    steps_with_unreached_base = 0
                    residual_counts: list[int] = []
                    accepted_counts: list[int] = []
                    for step_rows in steps.values():
                        first_low_pos: int | None = None
                        for row in step_rows:
                            pos = int(row.get("draft_position") or 0)
                            score_present, score_value = _score_value(row)
                            if score_present and score_value <= cutoff:
                                first_low_pos = pos
                                break
                        step_accepted_base = 0
                        step_rejected_base = 0
                        step_unreached_base = 0
                        step_residual = 0
                        accepted_counts.append(
                            max(int(row.get("num_accepted_in_step") or 0)
                                for row in step_rows)
                        )
                        for row in step_rows:
                            pos = int(row.get("draft_position") or 0)
                            score_present, _score = _score_value(row)
                            if not score_present:
                                missing_scores += 1
                            if first_low_pos is None:
                                critical_prefix = True
                            else:
                                critical_prefix = pos <= first_low_pos + extra_after_low
                            uses_residual = (
                                critical_prefix
                                or (prefix_len > 0 and pos <= prefix_len)
                                or not score_present
                            )
                            if uses_residual:
                                projected_residual += 1
                                step_residual += 1
                            if row.get("reached") == 1:
                                reached += 1
                                if not uses_residual:
                                    reached_base += 1
                            else:
                                unreached += 1
                                if not uses_residual:
                                    unreached_base += 1
                                    step_unreached_base += 1
                            if row.get("accepted_local") == 1:
                                accepted += 1
                                if not uses_residual:
                                    accepted_base += 1
                                    step_accepted_base += 1
                            if row.get("accepted_local") == 0:
                                rejected += 1
                                if not uses_residual:
                                    rejected_base += 1
                                    step_rejected_base += 1
                        residual_counts.append(step_residual)
                        if step_accepted_base:
                            steps_with_accepted_base += 1
                        if step_rejected_base:
                            steps_with_rejected_base += 1
                        if step_unreached_base:
                            steps_with_unreached_base += 1
                    out.append(
                        {
                            "run_id": run_id,
                            "model_label": model_label,
                            "dataset_label": dataset_label,
                            "method": method,
                            "source_sr24_policy": policy,
                            "projected_policy": "critical_prefix",
                            "projected_prefix_residual_len": prefix_len,
                            "projected_extra_after_low": extra_after_low,
                            "projected_score_threshold": cutoff,
                            "tokens": len(group),
                            "missing_score_tokens": missing_scores,
                            "projected_residual_tokens": projected_residual,
                            "projected_residual_fraction":
                            _ratio(projected_residual, len(group)),
                            "accepted_tokens": accepted,
                            "accepted_base_only_tokens": accepted_base,
                            "accepted_base_only_fraction":
                            _ratio(accepted_base, accepted),
                            "reached_tokens": reached,
                            "reached_base_only_tokens": reached_base,
                            "reached_base_only_fraction":
                            _ratio(reached_base, reached),
                            "unreached_tokens": unreached,
                            "unreached_base_only_tokens": unreached_base,
                            "unreached_base_only_fraction":
                            _ratio(unreached_base, unreached),
                            "rejected_tokens": rejected,
                            "rejected_base_only_tokens": rejected_base,
                            "rejected_base_only_fraction":
                            _ratio(rejected_base, rejected),
                            "steps": step_count,
                            "steps_with_accepted_base_only":
                            steps_with_accepted_base,
                            "steps_with_rejected_base_only":
                            steps_with_rejected_base,
                            "steps_with_unreached_base_only":
                            steps_with_unreached_base,
                            "steps_with_accepted_base_only_fraction":
                            _ratio(steps_with_accepted_base, step_count),
                            "steps_with_rejected_base_only_fraction":
                            _ratio(steps_with_rejected_base, step_count),
                            "steps_with_unreached_base_only_fraction":
                            _ratio(steps_with_unreached_base, step_count),
                            "mean_accepted_tokens_per_step":
                            _ratio(sum(accepted_counts), len(accepted_counts)),
                            "mean_projected_residual_rows_per_step":
                            _ratio(sum(residual_counts), len(residual_counts)),
                        }
                    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], trace_path: Path) -> None:
    lines = [
        "# SR24 Acceptance Trace Summary",
        "",
        f"- trace: `{trace_path.resolve()}`",
        "",
        "| model | dataset | method | policy | tokens | requested residual frac | effective residual frac | accepted effective-base frac | rejected requested-base frac | reached effective-base frac | unreached effective-base frac | steps accepted-base | steps rejected-base | steps unreached-base | mean accepted/step | mean residual rows/step |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {dataset} | {method} | {policy} | {tokens} | {residual:.4f} | {effective_residual:.4f} | {accepted_effective_base:.4f} | {rejected_base:.4f} | {reached_effective_base:.4f} | {unreached_effective_base:.4f} | {steps_base:.4f} | {steps_reject_base:.4f} | {steps_unreached_base:.4f} | {accepted_step:.4f} | {residual_step:.4f} |".format(
                model=row["model_label"],
                dataset=row["dataset_label"],
                method=row["method"],
                policy=row["sr24_policy"],
                tokens=row["tokens"],
                residual=row.get("sr24_residual_fraction") or 0.0,
                effective_residual=(
                    row.get("sr24_effective_residual_fraction") or 0.0
                ),
                accepted_base=row.get("accepted_base_only_fraction") or 0.0,
                accepted_effective_base=(
                    row.get("accepted_effective_base_only_fraction") or 0.0
                ),
                rejected_base=row.get("rejected_base_only_fraction") or 0.0,
                reached_effective_base=(
                    row.get("reached_effective_base_only_fraction") or 0.0
                ),
                unreached_effective_base=(
                    row.get("unreached_effective_base_only_fraction") or 0.0
                ),
                steps_base=row.get("steps_with_accepted_base_only_fraction") or 0.0,
                steps_reject_base=(
                    row.get("steps_with_rejected_base_only_fraction") or 0.0
                ),
                steps_unreached_base=(
                    row.get("steps_with_unreached_effective_base_only_fraction")
                    or 0.0
                ),
                accepted_step=row.get("mean_accepted_tokens_per_step") or 0.0,
                residual_step=row.get("mean_sr24_residual_rows_per_step") or 0.0,
            )
        )
    if rows:
        lines.extend(
            [
                "",
                "## Current-Policy Diagnosis",
                "",
                "Accepted base-only tokens are the quality-risk tokens: they were",
                "accepted by the verifier while running through the SR24 effective",
                "base-only path. Rejected base-only tokens are also risky: the",
                "first rejected token's target logits choose the recovered token.",
                "Unreached base-only suffix rows are listed separately because they",
                "are not accepted/rejected locally but may still expose scheduler or",
                "KV-side equivalence bugs when selective routing is not identical to",
                "the all-corrected path.",
                "The requested-base metric uses the routing mask; the effective-base",
                "metric also treats residual rows excluded by a fixed bucket cap as",
                "base-only.",
                "",
                "| model | dataset | policy | accepted effective-base frac | rejected requested-base frac | unreached effective-base frac | effective base score bins |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| {model} | {dataset} | {policy} | {accepted_effective_base:.4f} | {rejected_base:.4f} | {unreached_effective_base:.4f} | `{bins}` |".format(
                    model=row["model_label"],
                    dataset=row["dataset_label"],
                    policy=row["sr24_policy"],
                    accepted_effective_base=(
                        row.get("accepted_effective_base_only_fraction") or 0.0
                    ),
                    rejected_base=row.get("rejected_base_only_fraction") or 0.0,
                    unreached_effective_base=(
                        row.get("unreached_effective_base_only_fraction") or 0.0
                    ),
                    bins=row.get("accepted_effective_base_score_bins")
                    or row.get("accepted_base_score_bins")
                    or "{}",
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_position_report(path: Path, position_rows: list[dict[str, Any]]) -> None:
    lines = [
        "",
        "## By Draft Position",
        "",
        "| position | tokens | reached | unreached | accepted | residual frac | accepted base-only frac | reached base-only frac | unreached base-only frac |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in position_rows:
        lines.append(
            "| {pos} | {tokens} | {reached} | {unreached} | {accepted} | {residual:.4f} | {accepted_base:.4f} | {reached_base:.4f} | {unreached_base:.4f} |".format(
                pos=row["draft_position"],
                tokens=row["tokens"],
                reached=row["reached_tokens"],
                unreached=row["unreached_tokens"],
                accepted=row["accepted_tokens"],
                residual=row.get("sr24_residual_fraction") or 0.0,
                accepted_base=row.get("accepted_base_only_fraction") or 0.0,
                reached_base=row.get("reached_base_only_fraction") or 0.0,
                unreached_base=row.get("unreached_base_only_fraction") or 0.0,
            )
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def append_prefix_projection_report(
    path: Path,
    projection_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "",
        "## Prefix Residual Projection",
        "",
        "This is an offline projection over the existing trace. `prefix=N` means",
        "draft positions `1..N` are treated as residual/dense even if the traced",
        "run used base-only for them.",
        "",
        "| prefix | residual frac | accepted base-only frac | rejected base-only frac | reached base-only frac | unreached base-only frac | steps accepted-base | steps rejected-base | steps unreached-base | mean accepted/step | mean residual rows/step |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in projection_rows:
        lines.append(
            "| {prefix} | {residual:.4f} | {accepted_base:.4f} | {rejected_base:.4f} | {reached_base:.4f} | {unreached_base:.4f} | {steps_base:.4f} | {steps_reject_base:.4f} | {steps_unreached_base:.4f} | {accepted_step:.4f} | {residual_step:.4f} |".format(
                prefix=row["projected_prefix_residual_len"],
                residual=row.get("projected_residual_fraction") or 0.0,
                accepted_base=row.get("accepted_base_only_fraction") or 0.0,
                rejected_base=row.get("rejected_base_only_fraction") or 0.0,
                reached_base=row.get("reached_base_only_fraction") or 0.0,
                unreached_base=row.get("unreached_base_only_fraction") or 0.0,
                steps_base=row.get("steps_with_accepted_base_only_fraction") or 0.0,
                steps_reject_base=(
                    row.get("steps_with_rejected_base_only_fraction") or 0.0
                ),
                steps_unreached_base=(
                    row.get("steps_with_unreached_base_only_fraction") or 0.0
                ),
                accepted_step=row.get("mean_accepted_tokens_per_step") or 0.0,
                residual_step=row.get("mean_projected_residual_rows_per_step") or 0.0,
            )
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def append_score_projection_report(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    # Keep the report compact: show the lowest residual fraction that reaches
    # each accepted-base-only target. Also surface rejected-base-only tokens:
    # the first rejected token's target logits choose the recovered token, so it
    # is a quality-risk token even though it was not accepted.
    targets = [0.20, 0.10, 0.05, 0.02, 0.00]
    best_rows: list[dict[str, Any]] = []
    for target in targets:
        candidates = [
            row for row in rows
            if (row.get("accepted_base_only_fraction") or 0.0) <= target
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (
                row.get("projected_residual_fraction") or 1.0,
                row.get("projected_prefix_residual_len") or 0,
                row.get("projected_score_threshold") or 0.0,
            ),
        )
        best = dict(best)
        best["accepted_base_only_target"] = target
        best_rows.append(best)

    lines = [
        "",
        "## Score-Threshold Projection",
        "",
        "Offline projection for a quality-risk routing rule:",
        "`residual = draft_selected_prob > threshold OR draft_position <= prefix`.",
        "Missing scores are treated as residual.",
        "",
        "| accepted-base target | prefix | threshold | residual frac | accepted base-only frac | rejected base-only frac | reached base-only frac | unreached base-only frac | steps accepted-base | steps rejected-base | steps unreached-base | mean residual rows/step |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows:
        lines.append(
            "| {target:.2f} | {prefix} | {threshold:.1f} | {residual:.4f} | {accepted_base:.4f} | {rejected_base:.4f} | {reached_base:.4f} | {unreached_base:.4f} | {steps_base:.4f} | {steps_reject_base:.4f} | {steps_unreached_base:.4f} | {residual_step:.4f} |".format(
                target=row["accepted_base_only_target"],
                prefix=row["projected_prefix_residual_len"],
                threshold=row["projected_score_threshold"],
                residual=row.get("projected_residual_fraction") or 0.0,
                accepted_base=row.get("accepted_base_only_fraction") or 0.0,
                rejected_base=row.get("rejected_base_only_fraction") or 0.0,
                reached_base=row.get("reached_base_only_fraction") or 0.0,
                unreached_base=row.get("unreached_base_only_fraction") or 0.0,
                steps_base=row.get("steps_with_accepted_base_only_fraction") or 0.0,
                steps_reject_base=(
                    row.get("steps_with_rejected_base_only_fraction") or 0.0
                ),
                steps_unreached_base=(
                    row.get("steps_with_unreached_base_only_fraction") or 0.0
                ),
                residual_step=row.get("mean_projected_residual_rows_per_step") or 0.0,
            )
        )
    critical_rows: list[dict[str, Any]] = []
    for target in targets:
        candidates = []
        for row in rows:
            accepted_base = row.get("accepted_base_only_fraction") or 0.0
            rejected_base = row.get("rejected_base_only_fraction") or 0.0
            if max(accepted_base, rejected_base) <= target:
                candidates.append(row)
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (
                row.get("projected_residual_fraction") or 1.0,
                row.get("projected_prefix_residual_len") or 0,
                row.get("projected_score_threshold") or 0.0,
            ),
        )
        best = dict(best)
        best["critical_base_only_target"] = target
        critical_rows.append(best)
    lines.extend([
        "",
        "Critical-base projection constrains both accepted and rejected",
        "base-only risk by `max(accepted_base_only, rejected_base_only)`.",
        "",
        "| critical-base target | prefix | threshold | residual frac | accepted base-only frac | rejected base-only frac | unreached base-only frac | steps accepted-base | steps rejected-base | steps unreached-base | mean residual rows/step |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in critical_rows:
        lines.append(
            "| {target:.2f} | {prefix} | {threshold:.1f} | {residual:.4f} | {accepted_base:.4f} | {rejected_base:.4f} | {unreached_base:.4f} | {steps_base:.4f} | {steps_reject_base:.4f} | {steps_unreached_base:.4f} | {residual_step:.4f} |".format(
                target=row["critical_base_only_target"],
                prefix=row["projected_prefix_residual_len"],
                threshold=row["projected_score_threshold"],
                residual=row.get("projected_residual_fraction") or 0.0,
                accepted_base=row.get("accepted_base_only_fraction") or 0.0,
                rejected_base=row.get("rejected_base_only_fraction") or 0.0,
                unreached_base=row.get("unreached_base_only_fraction") or 0.0,
                steps_base=row.get("steps_with_accepted_base_only_fraction") or 0.0,
                steps_reject_base=(
                    row.get("steps_with_rejected_base_only_fraction") or 0.0
                ),
                steps_unreached_base=(
                    row.get("steps_with_unreached_base_only_fraction") or 0.0
                ),
                residual_step=row.get("mean_projected_residual_rows_per_step") or 0.0,
            )
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def append_critical_prefix_projection_report(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    targets = [0.20, 0.10, 0.05, 0.02, 0.00]
    critical_rows: list[dict[str, Any]] = []
    for target in targets:
        candidates = []
        for row in rows:
            accepted_base = row.get("accepted_base_only_fraction") or 0.0
            rejected_base = row.get("rejected_base_only_fraction") or 0.0
            if max(accepted_base, rejected_base) <= target:
                candidates.append(row)
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (
                row.get("projected_residual_fraction") or 1.0,
                row.get("projected_prefix_residual_len") or 0,
                row.get("projected_extra_after_low") or 0,
                row.get("projected_score_threshold") or 0.0,
            ),
        )
        best = dict(best)
        best["critical_base_only_target"] = target
        critical_rows.append(best)

    lines = [
        "",
        "## Critical-Prefix Projection",
        "",
        "Offline projection for the runtime `critical_prefix` policy. It marks",
        "the high-confidence prefix through the first low-confidence token as",
        "residual, then optionally includes `extra_after_low` more draft rows.",
        "If no low-confidence token appears, the whole draft is residual.",
        "",
        "| critical-base target | prefix | extra | threshold | residual frac | accepted base-only frac | rejected base-only frac | unreached base-only frac | steps accepted-base | steps rejected-base | steps unreached-base | mean residual rows/step |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in critical_rows:
        lines.append(
            "| {target:.2f} | {prefix} | {extra} | {threshold:.1f} | {residual:.4f} | {accepted_base:.4f} | {rejected_base:.4f} | {unreached_base:.4f} | {steps_base:.4f} | {steps_reject_base:.4f} | {steps_unreached_base:.4f} | {residual_step:.4f} |".format(
                target=row["critical_base_only_target"],
                prefix=row["projected_prefix_residual_len"],
                extra=row["projected_extra_after_low"],
                threshold=row["projected_score_threshold"],
                residual=row.get("projected_residual_fraction") or 0.0,
                accepted_base=row.get("accepted_base_only_fraction") or 0.0,
                rejected_base=row.get("rejected_base_only_fraction") or 0.0,
                unreached_base=row.get("unreached_base_only_fraction") or 0.0,
                steps_base=row.get("steps_with_accepted_base_only_fraction") or 0.0,
                steps_reject_base=(
                    row.get("steps_with_rejected_base_only_fraction") or 0.0
                ),
                steps_unreached_base=(
                    row.get("steps_with_unreached_base_only_fraction") or 0.0
                ),
                residual_step=row.get("mean_projected_residual_rows_per_step") or 0.0,
            )
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    trace_rows = load_rows(args.trace)
    rows = summarize(trace_rows)
    position_rows = summarize_by_position(trace_rows)
    projection_rows = summarize_prefix_projection(trace_rows)
    score_projection_rows = summarize_score_policy_projection(trace_rows)
    critical_projection_rows = summarize_critical_prefix_projection(trace_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "sr24_acceptance_trace_summary.csv", rows)
    write_csv(
        args.output_root / "sr24_acceptance_trace_by_position.csv",
        position_rows,
    )
    write_csv(args.output_root / "sr24_prefix_projection.csv", projection_rows)
    write_csv(
        args.output_root / "sr24_score_policy_projection.csv",
        score_projection_rows,
    )
    write_csv(
        args.output_root / "sr24_critical_prefix_projection.csv",
        critical_projection_rows,
    )
    (args.output_root / "sr24_acceptance_trace_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(args.output_root / "report.md", rows, args.trace)
    append_position_report(args.output_root / "report.md", position_rows)
    append_prefix_projection_report(args.output_root / "report.md", projection_rows)
    append_score_projection_report(
        args.output_root / "report.md",
        score_projection_rows,
    )
    append_critical_prefix_projection_report(
        args.output_root / "report.md",
        critical_projection_rows,
    )
    print(args.output_root / "report.md")


if __name__ == "__main__":
    main()
