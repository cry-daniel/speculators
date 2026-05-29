#!/usr/bin/env python3
"""Attribute token mismatches to SpecLink-CV scheduler events.

The verbose debug JSONL logs only record speculative/chunked verifier outputs.
Dense target-model steps are not emitted as ``spec_step_output`` events, so a
plain event count is not enough to explain where a final token mismatch came
from.  This diagnostic aligns each logged generated-token chunk against the
final output token IDs from ``correctness.json`` and inserts synthetic
``dense_gap`` segments for unlogged dense target steps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_EVENTS = {
    "dense_step_output",
    "spec_step_output",
    "prefix_accepted_requeue_suffix",
    "prefix_rejected_skip_suffix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-json", type=Path, required=True)
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--cv-events", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--search-window", type=int, default=128)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def request_ordinal(request_id: str) -> int | None:
    prefix = request_id.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def token_list(value: Any) -> list[int]:
    return [int(item) for item in (value or [])]


def first_diff(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def find_subsequence(
    sequence: list[int],
    needle: list[int],
    start: int,
    search_window: int,
) -> tuple[int, int, bool] | None:
    if not needle:
        return start, 0, False
    end = min(len(sequence) - len(needle) + 1, start + search_window + 1)
    for index in range(start, max(start, end)):
        if sequence[index : index + len(needle)] == needle:
            return index, len(needle), False
    # The final generation can be cut by max_tokens, so the logged event may
    # contain a suffix that is longer than the returned output token list.
    partial_end = min(len(sequence), start + search_window + len(needle))
    for index in range(start, partial_end):
        remaining = sequence[index:]
        if remaining and needle[: len(remaining)] == remaining:
            return index, len(remaining), True
    return None


def summarize_verifier(event: dict[str, Any] | None) -> dict[str, Any]:
    if not event:
        return {
            "verifier_event_index": "",
            "verifier_prefix_chunk_len": "",
            "verifier_suffix_chunk_len": "",
            "verifier_scheduled_spec_token_ids": [],
            "verifier_target_argmax_token_ids": [],
            "verifier_sampled_token_ids": [],
            "verifier_bonus_argmax_token_id": "",
            "verifier_query_input_token_ids": [],
            "verifier_target_input_token_ids": [],
            "verifier_row_index": "",
        }
    return {
        "verifier_event_index": event.get("_event_index", ""),
        "verifier_prefix_chunk_len": event.get("prefix_chunk_len"),
        "verifier_suffix_chunk_len": event.get("suffix_chunk_len"),
        "verifier_scheduled_spec_token_ids": token_list(
            event.get("scheduled_spec_token_ids")
        ),
        "verifier_target_argmax_token_ids": token_list(
            event.get("target_argmax_token_ids")
        ),
        "verifier_sampled_token_ids": token_list(event.get("sampled_token_ids")),
        "verifier_bonus_argmax_token_id": event.get("bonus_argmax_token_id", ""),
        "verifier_query_input_token_ids": token_list(
            event.get("query_input_token_ids")
        ),
        "verifier_target_input_token_ids": token_list(
            event.get("target_input_token_ids")
        ),
        "verifier_target_position_ids": token_list(
            event.get("target_position_ids")
        ),
        "verifier_target_slot_mapping_gid0": token_list(
            event.get("target_slot_mapping_gid0")
        ),
        "verifier_bonus_position_id": event.get("bonus_position_id", ""),
        "verifier_bonus_slot_mapping_gid0": event.get(
            "bonus_slot_mapping_gid0", ""
        ),
        "verifier_query_start": event.get("query_start", ""),
        "verifier_query_end": event.get("query_end", ""),
        "verifier_seq_len": event.get("seq_len", ""),
        "verifier_row_index": event.get("row_index", ""),
    }


def build_segments(
    *,
    source: str,
    events: list[dict[str, Any]],
    request_index: int,
    output_tokens: list[int],
    baseline_tokens: list[int],
    cv_tokens: list[int],
    search_window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    position = 0
    last_verifier: dict[str, Any] | None = None

    def add_segment(
        *,
        event_name: str,
        event_index: int | str,
        start: int,
        end: int,
        generated_tokens: list[int],
        event: dict[str, Any] | None = None,
    ) -> None:
        baseline_slice = baseline_tokens[start:end]
        cv_slice = cv_tokens[start:end]
        row: dict[str, Any] = {
            "source": source,
            "event": event_name,
            "event_index": event_index,
            "request_index": request_index,
            "start_token_index": start,
            "end_token_index": end,
            "length": end - start,
            "token_ids": list(generated_tokens),
            "baseline_token_ids": baseline_slice,
            "cv_token_ids": cv_slice,
            "matches_baseline_range": baseline_slice == cv_slice,
            "range_first_diff_offset": first_diff(baseline_slice, cv_slice),
            "num_accepted": "" if event is None else event.get("num_accepted", ""),
            "scheduled_spec_token_ids": []
            if event is None
            else token_list(event.get("scheduled_spec_token_ids")),
            "prefix_tokens": []
            if event is None
            else token_list(event.get("prefix_tokens")),
            "suffix_tokens": []
            if event is None
            else token_list(event.get("suffix_tokens")),
            "k": "" if event is None else event.get("k", ""),
            "selected_h": "" if event is None else event.get("selected_h", ""),
            "suffix_len": "" if event is None else event.get("suffix_len", ""),
        }
        row.update(summarize_verifier(last_verifier))
        segments.append(row)

    for event_index, raw_event in enumerate(events):
        event = dict(raw_event)
        event["_event_index"] = event_index
        if request_ordinal(str(event.get("request_id", ""))) != request_index:
            continue
        name = str(event.get("event", ""))
        if name == "dense_step_debug" and event.get("stage") == "after_sample":
            last_verifier = event
            continue
        if name == "verifier_step_debug" and event.get("stage") == (
            "before_prefix_bonus_mask"
        ):
            last_verifier = event
            continue
        if name not in OUTPUT_EVENTS:
            continue
        generated = token_list(event.get("generated_token_ids"))
        if not generated:
            continue
        match = find_subsequence(
            output_tokens,
            generated,
            position,
            search_window=search_window,
        )
        if match is None:
            errors.append(
                {
                    "source": source,
                    "event_index": event_index,
                    "event": name,
                    "position": position,
                    "generated_token_ids": generated,
                    "near_output_tokens": output_tokens[
                        position : position + search_window
                    ],
                }
            )
            break
        start, matched_len, truncated_by_output = match
        if start > position:
            add_segment(
                event_name="dense_gap",
                event_index="",
                start=position,
                end=start,
                generated_tokens=output_tokens[position:start],
                event=None,
            )
        add_segment(
            event_name=(
                f"{name}_truncated_by_output"
                if truncated_by_output
                else name
            ),
            event_index=event_index,
            start=start,
            end=start + matched_len,
            generated_tokens=generated[:matched_len],
            event=event,
        )
        position = start + matched_len

    if position < len(output_tokens):
        add_segment(
            event_name="dense_gap",
            event_index="",
            start=position,
            end=len(output_tokens),
            generated_tokens=output_tokens[position:],
            event=None,
        )
    return segments, errors


def segment_covering(
    segments: list[dict[str, Any]], token_index: int | None
) -> dict[str, Any] | None:
    if token_index is None:
        return None
    for segment in segments:
        if (
            int(segment["start_token_index"])
            <= token_index
            < int(segment["end_token_index"])
        ):
            return segment
    return None


CSV_COLUMNS = [
    "source",
    "event",
    "event_index",
    "request_index",
    "start_token_index",
    "end_token_index",
    "length",
    "matches_baseline_range",
    "range_first_diff_offset",
    "num_accepted",
    "k",
    "selected_h",
    "suffix_len",
    "token_ids",
    "baseline_token_ids",
    "cv_token_ids",
    "scheduled_spec_token_ids",
    "prefix_tokens",
    "suffix_tokens",
    "verifier_event_index",
    "verifier_prefix_chunk_len",
    "verifier_suffix_chunk_len",
    "verifier_scheduled_spec_token_ids",
    "verifier_target_argmax_token_ids",
    "verifier_sampled_token_ids",
    "verifier_bonus_argmax_token_id",
    "verifier_query_input_token_ids",
    "verifier_target_input_token_ids",
    "verifier_target_position_ids",
    "verifier_target_slot_mapping_gid0",
    "verifier_bonus_position_id",
    "verifier_bonus_slot_mapping_gid0",
    "verifier_query_start",
    "verifier_query_end",
    "verifier_seq_len",
    "verifier_row_index",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def write_md(
    path: Path,
    *,
    first_mismatch: int | None,
    baseline_token: int | None,
    cv_token: int | None,
    baseline_segment: dict[str, Any] | None,
    cv_segment: dict[str, Any] | None,
    errors: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SpecLink-CV Token Timeline",
        "",
        f"- first_mismatch_token_index: {first_mismatch}",
        f"- baseline_token: {baseline_token}",
        f"- speclink_cv_token: {cv_token}",
        f"- alignment_errors: {len(errors)}",
        "",
        "## Attributing Segment",
        "",
    ]
    if cv_segment is None:
        lines.append("No CV segment covered the first mismatch.")
    else:
        lines.extend(
            [
                f"- source: {cv_segment['source']}",
                f"- event: {cv_segment['event']}",
                f"- event_index: {cv_segment['event_index']}",
                (
                    "- token_range: "
                    f"{cv_segment['start_token_index']}.."
                    f"{cv_segment['end_token_index']}"
                ),
                f"- token_ids: `{cv_segment['token_ids']}`",
                f"- baseline_range: `{cv_segment['baseline_token_ids']}`",
                (
                    "- verifier_target_argmax_token_ids: "
                    f"`{cv_segment['verifier_target_argmax_token_ids']}`"
                ),
                (
                    "- verifier_scheduled_spec_token_ids: "
                    f"`{cv_segment['verifier_scheduled_spec_token_ids']}`"
                ),
                (
                    "- verifier_query_input_token_ids: "
                    f"`{cv_segment['verifier_query_input_token_ids']}`"
                ),
                (
                    "- verifier_target_input_token_ids: "
                    f"`{cv_segment['verifier_target_input_token_ids']}`"
                ),
                (
                    "- verifier_target_position_ids: "
                    f"`{cv_segment['verifier_target_position_ids']}`"
                ),
                (
                    "- verifier_target_slot_mapping_gid0: "
                    f"`{cv_segment['verifier_target_slot_mapping_gid0']}`"
                ),
                f"- verifier_prefix_chunk_len: {cv_segment['verifier_prefix_chunk_len']}",
                "",
            ]
        )
    if baseline_segment is not None:
        lines.extend(
            [
                "## Baseline Segment Covering Same Index",
                "",
                f"- event: {baseline_segment['event']}",
                f"- event_index: {baseline_segment['event_index']}",
                (
                    "- token_range: "
                    f"{baseline_segment['start_token_index']}.."
                    f"{baseline_segment['end_token_index']}"
                ),
                f"- token_ids: `{baseline_segment['token_ids']}`",
                (
                    "- verifier_target_argmax_token_ids: "
                    f"`{baseline_segment['verifier_target_argmax_token_ids']}`"
                ),
                (
                    "- verifier_scheduled_spec_token_ids: "
                    f"`{baseline_segment['verifier_scheduled_spec_token_ids']}`"
                ),
                (
                    "- verifier_query_input_token_ids: "
                    f"`{baseline_segment['verifier_query_input_token_ids']}`"
                ),
                (
                    "- verifier_target_input_token_ids: "
                    f"`{baseline_segment['verifier_target_input_token_ids']}`"
                ),
                (
                    "- verifier_target_position_ids: "
                    f"`{baseline_segment['verifier_target_position_ids']}`"
                ),
                (
                    "- verifier_target_slot_mapping_gid0: "
                    f"`{baseline_segment['verifier_target_slot_mapping_gid0']}`"
                ),
                "",
            ]
        )
    if errors:
        lines.extend(["## Alignment Errors", ""])
        for error in errors[:5]:
            lines.append(f"- `{json.dumps(error, ensure_ascii=False)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    correctness = json.loads(args.correctness_json.read_text(encoding="utf-8"))
    baseline_outputs = correctness.get("baseline", {}).get("outputs") or []
    cv_outputs = correctness.get("speclink_cv", {}).get("outputs") or []
    if args.request_index >= len(baseline_outputs):
        raise SystemExit(f"request index {args.request_index} missing in baseline")
    if args.request_index >= len(cv_outputs):
        raise SystemExit(f"request index {args.request_index} missing in CV")

    baseline_tokens = token_list(baseline_outputs[args.request_index].get("token_ids"))
    cv_tokens = token_list(cv_outputs[args.request_index].get("token_ids"))
    mismatch = first_diff(baseline_tokens, cv_tokens)

    baseline_segments, baseline_errors = build_segments(
        source="baseline",
        events=read_jsonl(args.baseline_events),
        request_index=args.request_index,
        output_tokens=baseline_tokens,
        baseline_tokens=baseline_tokens,
        cv_tokens=cv_tokens,
        search_window=args.search_window,
    )
    cv_segments, cv_errors = build_segments(
        source="speclink_cv",
        events=read_jsonl(args.cv_events),
        request_index=args.request_index,
        output_tokens=cv_tokens,
        baseline_tokens=baseline_tokens,
        cv_tokens=cv_tokens,
        search_window=args.search_window,
    )
    rows = baseline_segments + cv_segments
    errors = baseline_errors + cv_errors
    baseline_segment = segment_covering(baseline_segments, mismatch)
    cv_segment = segment_covering(cv_segments, mismatch)
    payload = {
        "request_index": args.request_index,
        "first_mismatch_token_index": mismatch,
        "baseline_token": None if mismatch is None else baseline_tokens[mismatch],
        "speclink_cv_token": None if mismatch is None else cv_tokens[mismatch],
        "baseline_segment_at_mismatch": baseline_segment,
        "speclink_cv_segment_at_mismatch": cv_segment,
        "alignment_errors": errors,
        "segments": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_md(
        args.output_md,
        first_mismatch=mismatch,
        baseline_token=None if mismatch is None else baseline_tokens[mismatch],
        cv_token=None if mismatch is None else cv_tokens[mismatch],
        baseline_segment=baseline_segment,
        cv_segment=cv_segment,
        errors=errors,
    )
    print(
        json.dumps(
            {
                "request_index": args.request_index,
                "first_mismatch_token_index": mismatch,
                "cv_event": None if cv_segment is None else cv_segment["event"],
                "cv_event_index": None
                if cv_segment is None
                else cv_segment["event_index"],
                "alignment_errors": len(errors),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
