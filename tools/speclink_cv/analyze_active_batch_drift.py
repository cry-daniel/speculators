#!/usr/bin/env python3
"""Compare verifier batch context at a token mismatch.

This diagnostic consumes ``token_timeline.json`` plus the baseline/CV debug
event streams.  It does not judge correctness by itself; it records whether the
verifier row that produced the first mismatch was evaluated with the same active
batch shape, scheduled draft IDs, and target-model top-k logits as the one-shot
baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-json", type=Path, required=True)
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--cv-events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def token_list(value: Any) -> list[int]:
    return [int(item) for item in (value or [])]


def request_ordinal(request_id: Any) -> int | None:
    raw = str(request_id or "")
    prefix = raw.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def verifier_group(events: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    if index < 0 or index >= len(events):
        raise SystemExit(f"verifier event index out of range: {index}")
    event = events[index]
    if event.get("event") not in {"verifier_step_debug", "dense_step_debug"}:
        raise SystemExit(
            f"event index {index} is not verifier_step_debug/dense_step_debug"
        )
    stage = event.get("stage")
    event_name = event.get("event")

    left = index
    while (
        left > 0
        and events[left - 1].get("event") == event_name
        and events[left - 1].get("stage") == stage
    ):
        left -= 1

    right = index + 1
    while (
        right < len(events)
        and events[right].get("event") == event_name
        and events[right].get("stage") == stage
    ):
        right += 1

    group: list[dict[str, Any]] = []
    for event_index in range(left, right):
        row = dict(events[event_index])
        row["_event_index"] = event_index
        group.append(row)
    return group


def active_ordinals(group: list[dict[str, Any]]) -> list[int]:
    ordinals: list[int] = []
    for row in group:
        has_draft = bool(token_list(row.get("scheduled_spec_token_ids")))
        has_dense_token = (
            row.get("event") == "dense_step_debug"
            and int(row.get("num_scheduled_tokens") or 0) > 0
        )
        if not has_draft and not has_dense_token:
            continue
        ordinal = request_ordinal(row.get("request_id"))
        if ordinal is not None:
            ordinals.append(ordinal)
    return ordinals


def row_for_request(group: list[dict[str, Any]], request_index: int) -> dict[str, Any]:
    for row in group:
        if request_ordinal(row.get("request_id")) == request_index:
            return row
    raise SystemExit(f"request ordinal {request_index} not found in verifier group")


def topk_value(row: dict[str, Any], position: int, token_id: int) -> float | None:
    token_rows = row.get("target_topk_token_ids") or []
    value_rows = row.get("target_topk_values") or []
    if position < 0 or position >= len(token_rows) or position >= len(value_rows):
        return None
    for item, value in zip(token_rows[position], value_rows[position]):
        if int(item) == int(token_id):
            return float(value)
    return None


def topk_margin(row: dict[str, Any], position: int) -> float | None:
    value_rows = row.get("target_topk_values") or []
    if position < 0 or position >= len(value_rows):
        return None
    values = [float(item) for item in value_rows[position]]
    if len(values) < 2:
        return None
    return values[0] - values[1]


def first_diff(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def row_summary(source: str, group: list[dict[str, Any]], request_index: int) -> dict[str, Any]:
    row = row_for_request(group, request_index)
    active = active_ordinals(group)
    scheduled = token_list(row.get("scheduled_spec_token_ids"))
    argmax = token_list(row.get("target_argmax_token_ids"))
    sampled = token_list(row.get("sampled_token_ids"))
    return {
        "source": source,
        "verifier_event_index": row.get("_event_index", ""),
        "stage": row.get("stage", ""),
        "row_index": row.get("row_index", ""),
        "request_id": row.get("request_id", ""),
        "num_computed_tokens_cpu": row.get("num_computed_tokens_cpu", ""),
        "num_tokens_no_spec": row.get("num_tokens_no_spec", ""),
        "num_prompt_tokens": row.get("num_prompt_tokens", ""),
        "worker_output_token_count": row.get("worker_output_token_count", ""),
        "context_tail_start": row.get("context_tail_start", ""),
        "context_tail_token_ids": token_list(row.get("context_tail_token_ids")),
        "block_ids_tail": row.get("block_ids_tail") or [],
        "group_rows": len(group),
        "active_rows": len(active),
        "active_request_ordinals": active,
        "scheduled_spec_token_ids": scheduled,
        "target_argmax_token_ids": argmax,
        "sampled_token_ids": sampled,
        "first_target_argmax": argmax[0] if argmax else "",
        "first_scheduled_draft": scheduled[0] if scheduled else "",
        "first_position_margin": topk_margin(row, 0),
        "baseline_token_value": None,
        "cv_token_value": None,
        "target_topk_token_ids_pos0": token_list((row.get("target_topk_token_ids") or [[]])[0])
        if row.get("target_topk_token_ids")
        else [],
        "target_topk_values_pos0": [
            float(item) for item in ((row.get("target_topk_values") or [[]])[0])
        ]
        if row.get("target_topk_values")
        else [],
        "target_position_ids": token_list(row.get("target_position_ids")),
        "target_slot_mapping_gid0": token_list(row.get("target_slot_mapping_gid0")),
        "bonus_position_id": row.get("bonus_position_id", ""),
        "bonus_slot_mapping_gid0": row.get("bonus_slot_mapping_gid0", ""),
        "query_start": row.get("query_start", ""),
        "query_end": row.get("query_end", ""),
        "seq_len": row.get("seq_len", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "verifier_event_index",
        "stage",
        "row_index",
        "request_id",
        "group_rows",
        "active_rows",
        "active_request_ordinals",
        "num_computed_tokens_cpu",
        "num_tokens_no_spec",
        "num_prompt_tokens",
        "worker_output_token_count",
        "context_tail_start",
        "context_tail_token_ids",
        "block_ids_tail",
        "query_start",
        "query_end",
        "seq_len",
        "first_scheduled_draft",
        "first_target_argmax",
        "first_position_margin",
        "baseline_token_value",
        "cv_token_value",
        "target_position_ids",
        "target_slot_mapping_gid0",
        "bonus_position_id",
        "bonus_slot_mapping_gid0",
        "scheduled_spec_token_ids",
        "target_argmax_token_ids",
        "sampled_token_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, (list, dict)):
                    out[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Active Batch Drift Diagnostic",
        "",
        f"- request_index: {summary['request_index']}",
        f"- first_mismatch_token_index: {summary['first_mismatch_token_index']}",
        f"- baseline_token: {summary['baseline_token']}",
        f"- speclink_cv_token: {summary['speclink_cv_token']}",
        f"- scheduled_drafts_equal: {summary['scheduled_drafts_equal']}",
        f"- first_argmax_equal: {summary['first_argmax_equal']}",
        f"- target_argmax_equal: {summary['target_argmax_equal']}",
        f"- target_argmax_first_diff: {summary['target_argmax_first_diff']}",
        f"- active_request_ordinals_equal: {summary['active_request_ordinals_equal']}",
        f"- num_computed_tokens_equal: {summary['num_computed_tokens_equal']}",
        f"- num_tokens_no_spec_equal: {summary['num_tokens_no_spec_equal']}",
        f"- context_tail_equal: {summary['context_tail_equal']}",
        f"- block_ids_tail_equal: {summary['block_ids_tail_equal']}",
        f"- target_position_ids_equal: {summary['target_position_ids_equal']}",
        f"- target_slot_mapping_gid0_equal: {summary['target_slot_mapping_gid0_equal']}",
        f"- missing_active_in_cv: {summary['missing_active_in_cv']}",
        f"- extra_active_in_cv: {summary['extra_active_in_cv']}",
        "",
        "| source | verifier event | row | active rows | computed | no-spec | out toks | first draft | first argmax | margin | baseline token logit | cv token logit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {source} | {event} | {row_index} | {active_rows} | "
            "{computed} | {no_spec} | {out_toks} | {draft} | {argmax} | "
            "{margin} | {baseline_value} | {cv_value} |".format(
                source=row["source"],
                event=row["verifier_event_index"],
                row_index=row["row_index"],
                active_rows=row["active_rows"],
                computed=row["num_computed_tokens_cpu"],
                no_spec=row["num_tokens_no_spec"],
                out_toks=row["worker_output_token_count"],
                draft=row["first_scheduled_draft"],
                argmax=row["first_target_argmax"],
                margin=fmt(row["first_position_margin"]),
                baseline_value=fmt(row["baseline_token_value"]),
                cv_value=fmt(row["cv_token_value"]),
            )
        )
    lines.extend(
        [
            "",
            "Active request ordinals:",
            "",
            f"- baseline: `{rows[0]['active_request_ordinals']}`",
            f"- speclink_cv: `{rows[1]['active_request_ordinals']}`",
            "",
            "Context tail token IDs, when present in the debug trace:",
            "",
            f"- baseline start {rows[0]['context_tail_start']}: "
            f"`{rows[0]['context_tail_token_ids']}`",
            f"- speclink_cv start {rows[1]['context_tail_start']}: "
            f"`{rows[1]['context_tail_token_ids']}`",
            "",
            "Verifier position IDs and current-token slot mappings:",
            "",
            f"- baseline positions: `{rows[0]['target_position_ids']}`",
            f"- speclink_cv positions: `{rows[1]['target_position_ids']}`",
            f"- baseline slot gid0: `{rows[0]['target_slot_mapping_gid0']}`",
            f"- speclink_cv slot gid0: `{rows[1]['target_slot_mapping_gid0']}`",
            "",
            "Physical KV block tails, when present in the debug trace:",
            "",
            f"- baseline: `{rows[0]['block_ids_tail']}`",
            f"- speclink_cv: `{rows[1]['block_ids_tail']}`",
            "",
            "Interpretation: this file only diagnoses the failing trace. If the "
            "scheduled drafts are equal but the active request set or first-token "
            "top-k logits differ, the failure is consistent with verifier/KV, "
            "physical KV layout, or batch-shape drift rather than a missing "
            "full-K confirmation branch.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_payload(timeline: dict[str, Any], baseline_events: list[dict[str, Any]], cv_events: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_segment = timeline["baseline_segment_at_mismatch"]
    cv_segment = timeline["speclink_cv_segment_at_mismatch"]
    request_index = int(timeline["request_index"])
    baseline_token = int(timeline["baseline_token"])
    cv_token = int(timeline["speclink_cv_token"])

    baseline_group = verifier_group(
        baseline_events, int(baseline_segment["verifier_event_index"])
    )
    cv_group = verifier_group(cv_events, int(cv_segment["verifier_event_index"]))
    rows = [
        row_summary("baseline", baseline_group, request_index),
        row_summary("speclink_cv", cv_group, request_index),
    ]
    for row in rows:
        source_group = baseline_group if row["source"] == "baseline" else cv_group
        source_row = row_for_request(source_group, request_index)
        row["baseline_token_value"] = topk_value(source_row, 0, baseline_token)
        row["cv_token_value"] = topk_value(source_row, 0, cv_token)

    baseline_active = set(rows[0]["active_request_ordinals"])
    cv_active = set(rows[1]["active_request_ordinals"])

    def equal_or_unknown(key: str) -> bool | str:
        left = rows[0].get(key)
        right = rows[1].get(key)
        if left in ("", [], None) and right in ("", [], None):
            return "unknown"
        return left == right

    summary = {
        "request_index": request_index,
        "first_mismatch_token_index": timeline.get("first_mismatch_token_index"),
        "baseline_token": baseline_token,
        "speclink_cv_token": cv_token,
        "scheduled_drafts_equal": rows[0]["scheduled_spec_token_ids"]
        == rows[1]["scheduled_spec_token_ids"],
        "first_argmax_equal": rows[0]["first_target_argmax"]
        == rows[1]["first_target_argmax"],
        "target_argmax_equal": rows[0]["target_argmax_token_ids"]
        == rows[1]["target_argmax_token_ids"],
        "target_argmax_first_diff": first_diff(
            rows[0]["target_argmax_token_ids"], rows[1]["target_argmax_token_ids"]
        ),
        "active_request_ordinals_equal": rows[0]["active_request_ordinals"]
        == rows[1]["active_request_ordinals"],
        "num_computed_tokens_equal": equal_or_unknown("num_computed_tokens_cpu"),
        "num_tokens_no_spec_equal": equal_or_unknown("num_tokens_no_spec"),
        "context_tail_equal": equal_or_unknown("context_tail_token_ids"),
        "block_ids_tail_equal": equal_or_unknown("block_ids_tail"),
        "target_position_ids_equal": equal_or_unknown("target_position_ids"),
        "target_slot_mapping_gid0_equal": equal_or_unknown(
            "target_slot_mapping_gid0"
        ),
        "missing_active_in_cv": sorted(baseline_active - cv_active),
        "extra_active_in_cv": sorted(cv_active - baseline_active),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    args = parse_args()
    payload = build_payload(
        read_json(args.timeline_json),
        read_jsonl(args.baseline_events),
        read_jsonl(args.cv_events),
    )
    write_json(args.output_json, payload)
    write_csv(args.output_csv, payload["rows"])
    write_markdown(args.output_md, payload)


if __name__ == "__main__":
    main()
