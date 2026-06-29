#!/usr/bin/env python3
"""Summarize request-level risk features from an SR24 confidence trace.

This is an offline diagnostic for SR24 quality regressions. It joins
`analyze_sr24_sample_divergence.py` output with `speclink_confidence_trace.jsonl`
and reports whether early speculative signals separate dense-correct/SR24-wrong
requests from the rest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None or not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _load_divergence(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            request_id = row.get("request_id")
            if request_id:
                rows[request_id] = row
    return rows


def _load_trace(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = row.get("request_id")
            if request_id:
                by_request[request_id].append(row)
    for rows in by_request.values():
        rows.sort(
            key=lambda row: (
                int(row.get("step_id") or 0),
                int(row.get("draft_position") or 0),
            )
        )
    return by_request


def _step_rows(rows: list[dict[str, Any]], max_steps: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        step_id = int(row.get("step_id") or 0)
        if step_id <= max_steps:
            out.append(row)
    return out


def _summarize_request(
    request_id: str,
    label: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    max_steps: int,
) -> dict[str, Any]:
    early = _step_rows(rows, max_steps)
    reached = [row for row in early if row.get("reached") == 1]
    rejected = [row for row in early if row.get("accepted_local") == 0]
    accepted = [row for row in early if row.get("accepted_local") == 1]
    first_step = [row for row in rows if int(row.get("step_id") or 0) == 1]
    first_reached = [row for row in first_step if row.get("reached") == 1]
    accepted_per_step: dict[int, int] = {}
    first_reject_per_step: dict[int, int] = {}
    for row in early:
        step_id = int(row.get("step_id") or 0)
        accepted_per_step[step_id] = int(row.get("num_accepted_in_step") or 0)
        first_reject = _float(row.get("first_reject_position"))
        if first_reject is not None:
            first_reject_per_step[step_id] = int(first_reject)

    def vals(key: str, source: list[dict[str, Any]] = early) -> list[float]:
        out: list[float] = []
        for row in source:
            value = _float(row.get(key))
            if value is not None:
                out.append(value)
        return out

    draft_probs = vals("draft_selected_prob", reached)
    target_probs = vals("target_selected_prob", reached)
    entropies = vals("draft_entropy", reached)
    target_rank = vals("target_rank_of_draft_token", reached)
    accepted_lengths = list(accepted_per_step.values())
    first_rejects = list(first_reject_per_step.values())
    return {
        "request_id": request_id,
        "doc_id": label.get("doc_id", ""),
        "regression": label.get("regression", ""),
        "improvement": label.get("improvement", ""),
        "dense_correct": label.get("dense_correct", ""),
        "experiment_correct": label.get("experiment_correct", ""),
        "target": label.get("target", ""),
        "trace_rows": len(rows),
        "early_rows": len(early),
        "early_reached_rows": len(reached),
        "early_accepted_rows": len(accepted),
        "early_rejected_rows": len(rejected),
        "early_reject_rate": (
            len(rejected) / len(reached) if reached else ""
        ),
        "early_accept_rate": (
            len(accepted) / len(reached) if reached else ""
        ),
        "early_accepted_len_mean": _mean(accepted_lengths),
        "early_accepted_len_min": min(accepted_lengths) if accepted_lengths else "",
        "early_first_reject_min": min(first_rejects) if first_rejects else "",
        "first_step_accepted_len": (
            int(first_step[0].get("num_accepted_in_step") or 0)
            if first_step else ""
        ),
        "first_step_first_reached_draft_prob": (
            _float(first_reached[0].get("draft_selected_prob"))
            if first_reached else ""
        ),
        "first_step_first_reached_target_rank": (
            _float(first_reached[0].get("target_rank_of_draft_token"))
            if first_reached else ""
        ),
        "early_draft_prob_mean": _mean(draft_probs),
        "early_draft_prob_min": min(draft_probs) if draft_probs else "",
        "early_target_prob_mean": _mean(target_probs),
        "early_target_rank_mean": _mean(target_rank),
        "early_target_rank_max": max(target_rank) if target_rank else "",
        "early_entropy_mean": _mean(entropies),
        "early_entropy_max": max(entropies) if entropies else "",
        "mask_states": ",".join(
            sorted({str(row.get("sr24_mask_state")) for row in rows
                    if row.get("sr24_mask_state") is not None})
        ),
        "base_only_scope": ",".join(
            sorted({str(row.get("sr24_base_only_layer_ids_by_leaf")) for row in rows
                    if row.get("sr24_base_only_layer_ids_by_leaf")})
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, subset in (
        ("regression", [row for row in rows if row.get("regression") == "True"]),
        ("non_regression", [row for row in rows if row.get("regression") != "True"]),
        ("all", rows),
    ):
        item: dict[str, Any] = {"group": label, "requests": len(subset)}
        for key in (
            "early_reject_rate",
            "early_accept_rate",
            "early_accepted_len_mean",
            "first_step_accepted_len",
            "first_step_first_reached_draft_prob",
            "first_step_first_reached_target_rank",
            "early_draft_prob_mean",
            "early_draft_prob_min",
            "early_target_rank_mean",
            "early_target_rank_max",
            "early_entropy_mean",
            "early_entropy_max",
        ):
            values = [_float(row.get(key)) for row in subset]
            values = [value for value in values if value is not None]
            item[f"{key}_mean"] = _mean(values)
            item[f"{key}_min"] = min(values) if values else ""
            item[f"{key}_max"] = max(values) if values else ""
        out.append(item)
    return out


def _threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    features = (
        ("early_entropy_max", ">="),
        ("early_target_rank_max", ">="),
        ("early_reject_rate", ">="),
        ("early_draft_prob_min", "<="),
        ("first_step_first_reached_draft_prob", "<="),
    )
    total_reg = sum(row.get("regression") == "True" for row in rows)
    for feature, direction in features:
        values = sorted({
            value for row in rows
            if (value := _float(row.get(feature))) is not None
        })
        if not values:
            continue
        if len(values) > 24:
            indices = sorted({
                int(round(i * (len(values) - 1) / 23))
                for i in range(24)
            })
            thresholds = [values[i] for i in indices]
        else:
            thresholds = values
        for threshold in thresholds:
            if direction == ">=":
                flagged = [
                    row for row in rows
                    if (value := _float(row.get(feature))) is not None
                    and value >= threshold
                ]
            else:
                flagged = [
                    row for row in rows
                    if (value := _float(row.get(feature))) is not None
                    and value <= threshold
                ]
            if not flagged:
                continue
            caught = sum(row.get("regression") == "True" for row in flagged)
            false_pos = len(flagged) - caught
            candidates.append({
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "flagged": len(flagged),
                "caught_regressions": caught,
                "total_regressions": total_reg,
                "recall": caught / total_reg if total_reg else "",
                "precision": caught / len(flagged),
                "false_positive": false_pos,
            })
    candidates.sort(
        key=lambda row: (
            _float(row.get("recall")) or 0.0,
            _float(row.get("precision")) or 0.0,
            -int(row.get("flagged") or 0),
        ),
        reverse=True,
    )
    return candidates


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# SR24 Trace Risk Analysis",
        "",
        f"- divergence csv: `{args.divergence_csv.resolve()}`",
        f"- trace jsonl: `{args.trace_jsonl.resolve()}`",
        f"- max early steps: `{args.max_steps}`",
        f"- requests: `{len(rows)}`",
        f"- regressions: `{sum(row.get('regression') == 'True' for row in rows)}`",
        "",
        "## Group Summary",
        "",
        "| group | requests | early reject rate | early accepted len | first-step accepted len | early draft prob min | early target rank max | early entropy max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {group} | {requests} | {reject} | {accepted} | {first} | {draft_min} | {rank_max} | {entropy_max} |".format(
                group=row["group"],
                requests=row["requests"],
                reject=_fmt(row.get("early_reject_rate_mean")),
                accepted=_fmt(row.get("early_accepted_len_mean_mean")),
                first=_fmt(row.get("first_step_accepted_len_mean")),
                draft_min=_fmt(row.get("early_draft_prob_min_mean")),
                rank_max=_fmt(row.get("early_target_rank_max_mean")),
                entropy_max=_fmt(row.get("early_entropy_max_mean")),
            )
        )
    lines.extend([
        "",
        "## Top Threshold Candidates",
        "",
        "| feature | dir | threshold | flagged | caught | recall | precision |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in thresholds[:20]:
        lines.append(
            "| {feature} | {direction} | {threshold} | {flagged} | {caught} | {recall} | {precision} |".format(
                feature=row["feature"],
                direction=row["direction"],
                threshold=_fmt(row["threshold"]),
                flagged=row["flagged"],
                caught=row["caught_regressions"],
                recall=_fmt(row["recall"]),
                precision=_fmt(row["precision"]),
            )
        )
    lines.extend([
        "",
        "## Regression Requests",
        "",
        "| doc_id | target | first step accepted | early reject rate | early target rank max | early entropy max | dense -> SR24 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        if row.get("regression") != "True":
            continue
        lines.append(
            "| {doc_id} | `{target}` | {first} | {reject} | {rank} | {entropy} | {dense} -> {exp} |".format(
                doc_id=row.get("doc_id", ""),
                target=row.get("target", ""),
                first=row.get("first_step_accepted_len", ""),
                reject=_fmt(row.get("early_reject_rate")),
                rank=_fmt(row.get("early_target_rank_max")),
                entropy=_fmt(row.get("early_entropy_max")),
                dense=row.get("dense_correct", ""),
                exp=row.get("experiment_correct", ""),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divergence-csv", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels = _load_divergence(args.divergence_csv)
    traces = _load_trace(args.trace_jsonl)
    rows = [
        _summarize_request(
            request_id,
            labels[request_id],
            traces.get(request_id, []),
            max_steps=args.max_steps,
        )
        for request_id in sorted(labels)
    ]
    summary = _group_summary(rows)
    thresholds = _threshold_rows(rows)
    _write_csv(args.output_root / "request_risk_features.csv", rows)
    _write_csv(args.output_root / "group_summary.csv", summary)
    _write_csv(args.output_root / "threshold_candidates.csv", thresholds)
    _write_report(args.output_root / "report.md", rows, summary, thresholds, args)
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
