#!/usr/bin/env python3
"""Classify where a SpecLink-CV correctness mismatch first appears.

This diagnostic answers whether the first output mismatch is produced by the
first prefix verification, a later suffix/full-K verification, or a dense step.
It also checks the visible "over-accept" hypothesis by verifying whether all
CV output segments before the mismatch still match the baseline and whether the
failing prefix-reject segment emitted more tokens than ``num_accepted + 1``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from tools.speclink_cv.analyze_token_timeline import (
        build_segments,
        first_diff,
        read_jsonl,
        segment_covering,
        token_list,
    )
except ModuleNotFoundError:  # pragma: no cover - script-path execution
    from analyze_token_timeline import (  # type: ignore
        build_segments,
        first_diff,
        read_jsonl,
        segment_covering,
        token_list,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--correctness-json", type=Path)
    parser.add_argument("--baseline-events", type=Path)
    parser.add_argument("--cv-events", type=Path)
    parser.add_argument("--search-window", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def request_ordinal(request_id: Any) -> int | None:
    raw = str(request_id or "")
    prefix = raw.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def event_name(row: dict[str, Any] | None) -> str:
    return "" if row is None else str(row.get("event", ""))


def int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def verification_attempt_index(
    events: list[dict[str, Any]],
    *,
    request_index: int,
    verifier_event_index: int | None,
) -> int | None:
    if verifier_event_index is None:
        return None
    count = 0
    for index, event in enumerate(events):
        if index > verifier_event_index:
            break
        name = event.get("event")
        stage = event.get("stage")
        if name not in {"verifier_step_debug", "dense_step_debug"}:
            continue
        if stage not in {"before_prefix_bonus_mask", "after_sample"}:
            continue
        if request_ordinal(event.get("request_id")) != request_index:
            continue
        count += 1
    return count


def classify_segment(segment: dict[str, Any] | None) -> str:
    name = event_name(segment)
    if name == "prefix_rejected_skip_suffix":
        return "prefix_first_chunk_rejected"
    if name == "prefix_accepted_requeue_suffix":
        return "prefix_first_chunk_accepted_suffix_requeued"
    if name == "spec_step_output":
        prefix_len = None if segment is None else segment.get("verifier_prefix_chunk_len")
        if prefix_len in ("", None):
            return "full_k_or_suffix_verification"
        return "prefix_or_suffix_verification"
    if name == "dense_gap":
        return "unlogged_dense_step"
    if name == "":
        return "missing_segment"
    return name


def emitted_expected_count(segment: dict[str, Any] | None) -> int | None:
    if segment is None:
        return None
    name = event_name(segment)
    accepted = int_or_none(segment.get("num_accepted"))
    selected_h = int_or_none(segment.get("selected_h"))
    if name == "prefix_rejected_skip_suffix" and accepted is not None:
        return accepted + 1
    if name == "prefix_accepted_requeue_suffix" and selected_h is not None:
        return selected_h
    if name == "spec_step_output" and accepted is not None:
        return accepted + 1
    return None


def compact_segment(segment: dict[str, Any] | None) -> dict[str, Any]:
    if segment is None:
        return {}
    keys = [
        "event",
        "event_index",
        "start_token_index",
        "end_token_index",
        "length",
        "num_accepted",
        "selected_h",
        "k",
        "suffix_len",
        "token_ids",
        "baseline_token_ids",
        "cv_token_ids",
        "verifier_event_index",
        "verifier_prefix_chunk_len",
        "verifier_suffix_chunk_len",
        "verifier_scheduled_spec_token_ids",
        "verifier_target_argmax_token_ids",
        "verifier_sampled_token_ids",
        "verifier_bonus_argmax_token_id",
        "verifier_target_position_ids",
        "verifier_target_slot_mapping_gid0",
        "verifier_bonus_position_id",
        "verifier_bonus_slot_mapping_gid0",
        "verifier_query_start",
        "verifier_query_end",
        "verifier_seq_len",
        "verifier_row_index",
    ]
    return {key: segment.get(key) for key in keys}


def classify_request(
    *,
    request_index: int,
    baseline_tokens: list[int],
    cv_tokens: list[int],
    baseline_events: list[dict[str, Any]],
    cv_events: list[dict[str, Any]],
    search_window: int,
) -> dict[str, Any] | None:
    mismatch = first_diff(baseline_tokens, cv_tokens)
    if mismatch is None:
        return None

    baseline_segments, baseline_errors = build_segments(
        source="baseline",
        events=baseline_events,
        request_index=request_index,
        output_tokens=baseline_tokens,
        baseline_tokens=baseline_tokens,
        cv_tokens=cv_tokens,
        search_window=search_window,
    )
    cv_segments, cv_errors = build_segments(
        source="speclink_cv",
        events=cv_events,
        request_index=request_index,
        output_tokens=cv_tokens,
        baseline_tokens=baseline_tokens,
        cv_tokens=cv_tokens,
        search_window=search_window,
    )
    baseline_segment = segment_covering(baseline_segments, mismatch)
    cv_segment = segment_covering(cv_segments, mismatch)
    previous_segments = [
        row for row in cv_segments if int(row["end_token_index"]) <= mismatch
    ]
    previous_segment = previous_segments[-1] if previous_segments else None
    previous_all_match = all(
        bool(row.get("matches_baseline_range")) for row in previous_segments
    )
    previous_prefix_accepts = sum(
        1 for row in previous_segments if event_name(row) == "prefix_accepted_requeue_suffix"
    )
    previous_prefix_rejects = sum(
        1 for row in previous_segments if event_name(row) == "prefix_rejected_skip_suffix"
    )
    previous_full_steps = sum(
        1 for row in previous_segments if event_name(row) == "spec_step_output"
    )

    verifier_index = int_or_none(
        None if cv_segment is None else cv_segment.get("verifier_event_index")
    )
    emitted_expected = emitted_expected_count(cv_segment)
    emitted_actual = int_or_none(None if cv_segment is None else cv_segment.get("length"))
    accepted_before_failure = int_or_none(
        None if cv_segment is None else cv_segment.get("num_accepted")
    )
    visible_over_accept = (
        emitted_expected is not None
        and emitted_actual is not None
        and emitted_actual > emitted_expected
    )

    return {
        "request_index": request_index,
        "first_mismatch_token_index": mismatch,
        "baseline_token": baseline_tokens[mismatch]
        if mismatch < len(baseline_tokens)
        else None,
        "speclink_cv_token": cv_tokens[mismatch] if mismatch < len(cv_tokens) else None,
        "failure_class": classify_segment(cv_segment),
        "cv_event": event_name(cv_segment),
        "cv_event_index": None if cv_segment is None else cv_segment.get("event_index"),
        "cv_verifier_event_index": verifier_index,
        "cv_verification_attempt_index": verification_attempt_index(
            cv_events,
            request_index=request_index,
            verifier_event_index=verifier_index,
        ),
        "baseline_event": event_name(baseline_segment),
        "baseline_event_index": None
        if baseline_segment is None
        else baseline_segment.get("event_index"),
        "previous_cv_event": event_name(previous_segment),
        "previous_cv_event_index": None
        if previous_segment is None
        else previous_segment.get("event_index"),
        "previous_cv_segments_all_match": previous_all_match,
        "previous_prefix_accepts": previous_prefix_accepts,
        "previous_prefix_rejects": previous_prefix_rejects,
        "previous_full_or_suffix_steps": previous_full_steps,
        "accepted_before_failure": accepted_before_failure,
        "failure_offset_within_verifier": accepted_before_failure,
        "emitted_expected_count": emitted_expected,
        "emitted_actual_count": emitted_actual,
        "visible_over_accept_in_failing_segment": visible_over_accept,
        "alignment_errors": len(baseline_errors) + len(cv_errors),
        "cv_segment": compact_segment(cv_segment),
        "baseline_segment": compact_segment(baseline_segment),
    }


CSV_FIELDS = [
    "request_index",
    "first_mismatch_token_index",
    "baseline_token",
    "speclink_cv_token",
    "failure_class",
    "cv_event",
    "cv_event_index",
    "cv_verifier_event_index",
    "cv_verification_attempt_index",
    "baseline_event",
    "baseline_event_index",
    "previous_cv_event",
    "previous_cv_event_index",
    "previous_cv_segments_all_match",
    "previous_prefix_accepts",
    "previous_prefix_rejects",
    "previous_full_or_suffix_steps",
    "accepted_before_failure",
    "failure_offset_within_verifier",
    "emitted_expected_count",
    "emitted_actual_count",
    "visible_over_accept_in_failing_segment",
    "alignment_errors",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    summary = payload["summary"]
    lines = [
        "# Verification Failure Stage",
        "",
        f"- mismatched requests: `{summary['mismatched_requests']}`",
        f"- prefix first-chunk rejected: `{summary['prefix_first_chunk_rejected']}`",
        f"- prefix accepted then suffix requeued: `{summary['prefix_first_chunk_accepted_suffix_requeued']}`",
        f"- full-K or suffix verification: `{summary['full_k_or_suffix_verification']}`",
        f"- visible over-accept segments: `{summary['visible_over_accept_segments']}`",
        "",
        "| request | token | class | CV event | attempt | accepted before failure | prev event | prev ok | emits | base token | CV token |",
        "| ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {request_index} | {first_mismatch_token_index} | {failure_class} | "
            "{cv_event} | {cv_verification_attempt_index} | {accepted_before_failure} | {previous_cv_event} | "
            "{previous_cv_segments_all_match} | {emitted_actual_count}/{emitted_expected_count} | "
            "{baseline_token} | {speclink_cv_token} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.case_dir is not None:
        return (
            args.case_dir / "correctness.json",
            args.case_dir / "baseline_events.jsonl",
            args.case_dir / "speclink_cv_events.jsonl",
        )
    if args.correctness_json and args.baseline_events and args.cv_events:
        return args.correctness_json, args.baseline_events, args.cv_events
    raise SystemExit("pass either --case-dir or all explicit input paths")


def main() -> int:
    args = parse_args()
    correctness_path, baseline_events_path, cv_events_path = resolve_inputs(args)
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    baseline_outputs = correctness.get("baseline", {}).get("outputs") or []
    cv_outputs = correctness.get("speclink_cv", {}).get("outputs") or []
    baseline_events = read_jsonl(baseline_events_path)
    cv_events = read_jsonl(cv_events_path)

    rows: list[dict[str, Any]] = []
    for request_index, (baseline_output, cv_output) in enumerate(
        zip(baseline_outputs, cv_outputs)
    ):
        row = classify_request(
            request_index=request_index,
            baseline_tokens=token_list(baseline_output.get("token_ids")),
            cv_tokens=token_list(cv_output.get("token_ids")),
            baseline_events=baseline_events,
            cv_events=cv_events,
            search_window=args.search_window,
        )
        if row is not None:
            rows.append(row)

    summary = {
        "mismatched_requests": len(rows),
        "prefix_first_chunk_rejected": sum(
            1 for row in rows if row["failure_class"] == "prefix_first_chunk_rejected"
        ),
        "prefix_first_chunk_accepted_suffix_requeued": sum(
            1
            for row in rows
            if row["failure_class"] == "prefix_first_chunk_accepted_suffix_requeued"
        ),
        "full_k_or_suffix_verification": sum(
            1 for row in rows if row["failure_class"] == "full_k_or_suffix_verification"
        ),
        "visible_over_accept_segments": sum(
            1 for row in rows if row["visible_over_accept_in_failing_segment"]
        ),
    }
    payload = {
        "summary": summary,
        "inputs": {
            "correctness_json": str(correctness_path),
            "baseline_events": str(baseline_events_path),
            "cv_events": str(cv_events_path),
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
