#!/usr/bin/env python3
"""Compare bounded EAGLE proposer debug events between baseline and CV runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--cv-events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def request_ordinal(request_id: str) -> int | str:
    raw = request_id.split("-", 1)[0]
    return int(raw) if raw.isdigit() else request_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def proposer_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for index, event in enumerate(read_jsonl(path)):
        if event.get("event") != "proposer_step_debug":
            continue
        item = dict(event)
        item["event_index"] = index
        item["request_ordinal"] = request_ordinal(str(event.get("request_id", "")))
        events.append(item)
    return events


def hidden_delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, float | None]:
    fields = ("sum", "abs_sum", "max_abs")
    result: dict[str, float | None] = {}
    left = left or {}
    right = right or {}
    for field in fields:
        if field in left and field in right:
            result[field] = float(right[field]) - float(left[field])
        else:
            result[field] = None
    return result


def row_summary(
    baseline: dict[str, Any], cv: dict[str, Any]
) -> dict[str, Any]:
    baseline_draft = baseline.get("draft_token_ids") or []
    cv_draft = cv.get("draft_token_ids") or []
    draft_first_diff = None
    for index, (left, right) in enumerate(zip(baseline_draft, cv_draft)):
        if left != right:
            draft_first_diff = index
            break
    if draft_first_diff is None and len(baseline_draft) != len(cv_draft):
        draft_first_diff = min(len(baseline_draft), len(cv_draft))

    return {
        "request_ordinal": baseline["request_ordinal"],
        "worker_output_token_count": baseline.get("worker_output_token_count"),
        "baseline_event_index": baseline.get("event_index"),
        "cv_event_index": cv.get("event_index"),
        "active_ordinals_equal": baseline.get("active_request_ordinals")
        == cv.get("active_request_ordinals"),
        "baseline_active_ordinals": baseline.get("active_request_ordinals"),
        "cv_active_ordinals": cv.get("active_request_ordinals"),
        "next_token_equal": baseline.get("next_token_id") == cv.get("next_token_id"),
        "baseline_next_token": baseline.get("next_token_id"),
        "cv_next_token": cv.get("next_token_id"),
        "sample_position_equal": baseline.get("sample_position")
        == cv.get("sample_position"),
        "baseline_sample_position": baseline.get("sample_position"),
        "cv_sample_position": cv.get("sample_position"),
        "sample_slot_equal": baseline.get("sample_slot_mapping")
        == cv.get("sample_slot_mapping"),
        "baseline_sample_slot": baseline.get("sample_slot_mapping"),
        "cv_sample_slot": cv.get("sample_slot_mapping"),
        "target_token_equal": baseline.get("sample_target_token_id")
        == cv.get("sample_target_token_id"),
        "baseline_target_token": baseline.get("sample_target_token_id"),
        "cv_target_token": cv.get("sample_target_token_id"),
        "scheduled_draft_equal": baseline.get("scheduled_spec_token_ids")
        == cv.get("scheduled_spec_token_ids"),
        "baseline_scheduled_draft": baseline.get("scheduled_spec_token_ids"),
        "cv_scheduled_draft": cv.get("scheduled_spec_token_ids"),
        "draft_equal": baseline_draft == cv_draft,
        "draft_first_diff": draft_first_diff,
        "baseline_draft": baseline_draft,
        "cv_draft": cv_draft,
        "hidden_delta": hidden_delta(
            baseline.get("target_hidden_checksum"),
            cv.get("target_hidden_checksum"),
        ),
        "baseline_hidden_checksum": baseline.get("target_hidden_checksum"),
        "cv_hidden_checksum": cv.get("target_hidden_checksum"),
    }


def build_rows(
    baseline_events: list[dict[str, Any]], cv_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cv_by_key = {
        (event.get("request_ordinal"), event.get("worker_output_token_count")): event
        for event in cv_events
    }
    rows = []
    for baseline in baseline_events:
        key = (
            baseline.get("request_ordinal"),
            baseline.get("worker_output_token_count"),
        )
        cv = cv_by_key.get(key)
        if cv is None:
            continue
        rows.append(row_summary(baseline, cv))
    rows.sort(key=lambda row: (row["request_ordinal"], row["worker_output_token_count"]))
    return rows


def compact_list(value: Any, max_items: int = 8) -> str:
    if not isinstance(value, list):
        return str(value)
    shown = value[:max_items]
    suffix = "" if len(value) <= max_items else f"...(+{len(value) - max_items})"
    return f"{shown}{suffix}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "request_ordinal",
        "worker_output_token_count",
        "active_ordinals_equal",
        "next_token_equal",
        "sample_position_equal",
        "sample_slot_equal",
        "target_token_equal",
        "scheduled_draft_equal",
        "draft_equal",
        "draft_first_diff",
        "baseline_next_token",
        "cv_next_token",
        "baseline_sample_position",
        "cv_sample_position",
        "baseline_sample_slot",
        "cv_sample_slot",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Proposer Drift",
        "",
        f"- compared proposer events: `{payload['compared_pairs']}`",
        f"- first drift: `{payload.get('first_drift_reason', 'none')}`",
    ]
    first = payload.get("first_drift")
    if first:
        lines.extend(
            [
                "",
                "## First Drift",
                "",
                f"- request ordinal: `{first['request_ordinal']}`",
                f"- worker output token count: `{first['worker_output_token_count']}`",
                f"- active ordinals equal: `{first['active_ordinals_equal']}`",
                f"- next token baseline/CV: `{first['baseline_next_token']}` / `{first['cv_next_token']}`",
                f"- sample position baseline/CV: `{first['baseline_sample_position']}` / `{first['cv_sample_position']}`",
                f"- sample slot baseline/CV: `{first['baseline_sample_slot']}` / `{first['cv_sample_slot']}`",
                f"- scheduled draft equal: `{first['scheduled_draft_equal']}`",
                f"- draft equal: `{first['draft_equal']}`",
                f"- draft first diff: `{first['draft_first_diff']}`",
                f"- hidden delta: `{first['hidden_delta']}`",
                f"- baseline draft: `{compact_list(first['baseline_draft'])}`",
                f"- CV draft: `{compact_list(first['cv_draft'])}`",
                f"- baseline scheduled draft: `{compact_list(first['baseline_scheduled_draft'])}`",
                f"- CV scheduled draft: `{compact_list(first['cv_scheduled_draft'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| request | out tokens | active ok | next ok | hidden delta abs | scheduled ok | draft ok | draft diff |",
            "|---:|---:|---|---|---:|---|---|---:|",
        ]
    )
    for row in payload["rows"][:32]:
        hidden_abs = (row.get("hidden_delta") or {}).get("abs_sum")
        hidden_abs_text = "" if hidden_abs is None else f"{hidden_abs:.6g}"
        lines.append(
            "| {request_ordinal} | {worker_output_token_count} | {active_ordinals_equal} | "
            "{next_token_equal} | {hidden_abs} | {scheduled_draft_equal} | "
            "{draft_equal} | {draft_first_diff} |".format(
                hidden_abs=hidden_abs_text,
                **row,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline = proposer_events(args.baseline_events)
    cv = proposer_events(args.cv_events)
    rows = build_rows(baseline, cv)
    first_drift = None
    first_reason = "none"
    for row in rows:
        if not row["active_ordinals_equal"]:
            first_drift = row
            first_reason = "active_request_set"
            break
        if not row["next_token_equal"]:
            first_drift = row
            first_reason = "next_token"
            break
        if row["hidden_delta"].get("sum") not in (0.0, None):
            first_drift = row
            first_reason = "target_hidden_checksum"
            break
        if not row["draft_equal"]:
            first_drift = row
            first_reason = "draft_token_ids"
            break

    payload = {
        "baseline_proposer_events": len(baseline),
        "cv_proposer_events": len(cv),
        "compared_pairs": len(rows),
        "first_drift_reason": first_reason,
        "first_drift": first_drift,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_md(args.output_md, payload)


if __name__ == "__main__":
    main()
