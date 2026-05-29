#!/usr/bin/env python3
"""Compare one-shot and chunked verifier boundary argmax events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--cv-events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_one_shot_index(
    events: list[dict[str, Any]],
) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    index: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "verifier_step_debug":
            continue
        if event.get("stage") != "before_prefix_bonus_mask":
            continue
        tokens = tuple(int(x) for x in event.get("scheduled_spec_token_ids") or [])
        if not tokens:
            continue
        index.setdefault(tokens, []).append(event)
    return index


def request_ordinal(request_id: str) -> int | None:
    prefix = request_id.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def build_one_shot_ordinal_index(
    events: list[dict[str, Any]],
) -> dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]]:
    index: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "verifier_step_debug":
            continue
        if event.get("stage") != "before_prefix_bonus_mask":
            continue
        ordinal = request_ordinal(str(event.get("request_id", "")))
        if ordinal is None:
            continue
        tokens = tuple(int(x) for x in event.get("scheduled_spec_token_ids") or [])
        if not tokens:
            continue
        index.setdefault((ordinal, tokens), []).append(event)
    return index


def analyze(
    baseline_events: list[dict[str, Any]],
    cv_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_index = build_one_shot_index(baseline_events)
    baseline_ordinal_index = build_one_shot_ordinal_index(baseline_events)
    pending_prefix: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for event in cv_events:
        name = event.get("event")
        req_id = str(event.get("request_id", ""))
        if name == "prefix_scheduled":
            prefix = [int(x) for x in event.get("prefix_tokens") or []]
            suffix = [int(x) for x in event.get("suffix_tokens") or []]
            if prefix and suffix:
                pending_prefix[req_id] = {
                    "request_id": req_id,
                    "selected_h": int(event.get("selected_h") or len(prefix)),
                    "prefix_tokens": prefix,
                    "suffix_tokens": suffix,
                    "full_draft_tokens": prefix + suffix,
                    "queue_wait_ms": event.get("queue_wait_ms"),
                }
            continue

        if name != "verifier_step_debug":
            continue
        if event.get("stage") != "before_prefix_bonus_mask":
            continue
        prefix_len = event.get("prefix_chunk_len")
        if prefix_len is None:
            continue
        plan = pending_prefix.get(req_id)
        if plan is None:
            continue

        full_draft = tuple(plan["full_draft_tokens"])
        ordinal = request_ordinal(req_id)
        baseline_matches = (
            baseline_ordinal_index.get((ordinal, full_draft), [])
            if ordinal is not None
            else []
        )
        if not baseline_matches:
            baseline_matches = baseline_index.get(full_draft, [])
        selected_h = int(plan["selected_h"])
        baseline_boundary_argmax = None
        baseline_boundary_topk = []
        baseline_boundary_topk_values = []
        baseline_request_id = ""
        prefix_argmax_mismatches: list[dict[str, Any]] = []
        prefix_argmax_total = 0
        prefix_argmax_match_count = 0
        if baseline_matches:
            baseline_event = baseline_matches[0]
            baseline_request_id = str(baseline_event.get("request_id", ""))
            target_argmax = baseline_event.get("target_argmax_token_ids") or []
            cv_target_argmax = event.get("target_argmax_token_ids") or []
            cv_target_topk = event.get("target_topk_token_ids") or []
            baseline_target_topk = baseline_event.get("target_topk_token_ids") or []
            for pos in range(min(selected_h, len(target_argmax), len(cv_target_argmax))):
                prefix_argmax_total += 1
                cv_argmax = int(cv_target_argmax[pos])
                baseline_argmax = int(target_argmax[pos])
                if cv_argmax == baseline_argmax:
                    prefix_argmax_match_count += 1
                    continue
                prefix_argmax_mismatches.append(
                    {
                        "position": pos,
                        "cv_argmax": cv_argmax,
                        "baseline_argmax": baseline_argmax,
                        "cv_topk": (
                            [int(x) for x in cv_target_topk[pos]]
                            if pos < len(cv_target_topk)
                            else []
                        ),
                        "baseline_topk": (
                            [int(x) for x in baseline_target_topk[pos]]
                            if pos < len(baseline_target_topk)
                            else []
                        ),
                    }
                )
            if selected_h < len(target_argmax):
                baseline_boundary_argmax = int(target_argmax[selected_h])
            target_topk = baseline_event.get("target_topk_token_ids") or []
            if selected_h < len(target_topk):
                baseline_boundary_topk = [
                    int(x) for x in target_topk[selected_h]
                ]
            target_topk_values = baseline_event.get("target_topk_values") or []
            if selected_h < len(target_topk_values):
                baseline_boundary_topk_values = [
                    float(x) for x in target_topk_values[selected_h]
                ]

        cv_bonus_argmax = event.get("bonus_argmax_token_id")
        if cv_bonus_argmax is not None:
            cv_bonus_argmax = int(cv_bonus_argmax)
        cv_bonus_topk = [int(x) for x in event.get("bonus_topk_token_ids") or []]
        cv_bonus_topk_values = [
            float(x) for x in event.get("bonus_topk_values") or []
        ]
        cv_contains_baseline = (
            baseline_boundary_argmax in cv_bonus_topk
            if baseline_boundary_argmax is not None
            else False
        )
        baseline_contains_cv = (
            cv_bonus_argmax in baseline_boundary_topk
            if cv_bonus_argmax is not None
            else False
        )
        records.append(
            {
                "request_id": req_id,
                "baseline_request_id": baseline_request_id,
                "selected_h": selected_h,
                "full_draft_tokens": list(full_draft),
                "prefix_tokens": plan["prefix_tokens"],
                "suffix_tokens": plan["suffix_tokens"],
                "cv_bonus_argmax": cv_bonus_argmax,
                "baseline_boundary_argmax": baseline_boundary_argmax,
                "cv_bonus_topk": cv_bonus_topk,
                "cv_bonus_topk_values": cv_bonus_topk_values,
                "baseline_boundary_topk": baseline_boundary_topk,
                "baseline_boundary_topk_values": baseline_boundary_topk_values,
                "cv_topk_contains_baseline_argmax": cv_contains_baseline,
                "baseline_topk_contains_cv_argmax": baseline_contains_cv,
                "prefix_argmax_match_count": prefix_argmax_match_count,
                "prefix_argmax_total": prefix_argmax_total,
                "prefix_argmax_mismatch_count": len(prefix_argmax_mismatches),
                "prefix_argmax_mismatches": prefix_argmax_mismatches,
                "argmax_match": (
                    baseline_boundary_argmax is not None
                    and cv_bonus_argmax == baseline_boundary_argmax
                ),
                "comparison_status": (
                    "match"
                    if baseline_boundary_argmax is not None
                    and cv_bonus_argmax == baseline_boundary_argmax
                    else "argmax_mismatch"
                    if baseline_boundary_argmax is not None
                    else "no_matching_one_shot_draft"
                ),
                "baseline_match_count": len(baseline_matches),
                "queue_wait_ms": plan.get("queue_wait_ms"),
            }
        )

    return records


def write_outputs(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "request_id",
        "baseline_request_id",
        "selected_h",
        "cv_bonus_argmax",
        "baseline_boundary_argmax",
        "cv_bonus_topk",
        "baseline_boundary_topk",
        "cv_topk_contains_baseline_argmax",
        "baseline_topk_contains_cv_argmax",
        "prefix_argmax_match_count",
        "prefix_argmax_total",
        "prefix_argmax_mismatch_count",
        "prefix_argmax_mismatches",
        "argmax_match",
        "comparison_status",
        "baseline_match_count",
        "queue_wait_ms",
        "prefix_tokens",
        "suffix_tokens",
        "full_draft_tokens",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in (
                "prefix_tokens",
                "suffix_tokens",
                "full_draft_tokens",
                "cv_bonus_topk",
                "baseline_boundary_topk",
                "prefix_argmax_mismatches",
            ):
                row[key] = json.dumps(row[key])
            writer.writerow({field: row.get(field) for field in fields})

    total = len(records)
    comparable = [
        record for record in records
        if record["comparison_status"] != "no_matching_one_shot_draft"
    ]
    matched = sum(1 for record in comparable if record["argmax_match"])
    mismatched_records = [
        record for record in comparable if not record["argmax_match"]
    ]
    unmatched = total - len(comparable)
    mismatches_with_topk = [
        record for record in mismatched_records
        if record.get("cv_bonus_topk") and record.get("baseline_boundary_topk")
    ]
    cv_contains_baseline = sum(
        1 for record in mismatches_with_topk
        if record.get("cv_topk_contains_baseline_argmax")
    )
    baseline_contains_cv = sum(
        1 for record in mismatches_with_topk
        if record.get("baseline_topk_contains_cv_argmax")
    )
    prefix_compared = sum(int(record.get("prefix_argmax_total") or 0)
                          for record in comparable)
    prefix_mismatches = sum(int(record.get("prefix_argmax_mismatch_count") or 0)
                            for record in comparable)
    lines = [
        "# Verifier Shape Drift Analysis",
        "",
        f"- compared prefix-accepted boundary events: `{total}`",
        f"- events with a matching one-shot full draft: `{len(comparable)}`",
        f"- prefix target positions compared with one-shot: `{prefix_compared}`",
        f"- prefix target argmax mismatches: `{prefix_mismatches}`",
        f"- boundary argmax matches among matched drafts: `{matched}`",
        f"- boundary argmax mismatches among matched drafts: `{len(mismatched_records)}`",
        f"- mismatches with top-k evidence: `{len(mismatches_with_topk)}`",
        f"- top-k evidence rows where CV top-k contains one-shot argmax: `{cv_contains_baseline}`",
        f"- top-k evidence rows where one-shot top-k contains CV argmax: `{baseline_contains_cv}`",
        f"- events without a matching one-shot full draft: `{unmatched}`",
        "",
        "A mismatch means the full-K one-shot verifier target argmax at the first"
        " suffix position differs from the h<K chunked prefix verifier bonus"
        " argmax for the same full draft token sequence.",
        "",
        "| request_id | h | cv bonus | one-shot boundary | prefix mismatches | top-k overlap | full draft prefix |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in mismatched_records[:20]:
        topk_status = (
            f"cv_has_base={record.get('cv_topk_contains_baseline_argmax')}, "
            f"base_has_cv={record.get('baseline_topk_contains_cv_argmax')}"
            if record.get("cv_bonus_topk") and record.get("baseline_boundary_topk")
            else "not_available"
        )
        lines.append(
            "| {request_id} | {selected_h} | {cv_bonus_argmax} | "
            "{baseline_boundary_argmax} | {prefix_argmax_mismatch_count} | "
            "{topk_status} | {prefix_tokens} |".format(
                topk_status=topk_status,
                **record,
            )
        )
    prefix_mismatch_records = [
        record for record in comparable
        if int(record.get("prefix_argmax_mismatch_count") or 0) > 0
    ]
    if prefix_mismatch_records:
        lines.extend(
            [
                "",
                "## Prefix Target Argmax Mismatches",
                "",
                "| request_id | h | mismatch_count | first mismatch | full draft prefix |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for record in prefix_mismatch_records[:20]:
            mismatches = record.get("prefix_argmax_mismatches") or []
            first = mismatches[0] if mismatches else {}
            first_text = (
                f"pos={first.get('position')} cv={first.get('cv_argmax')} "
                f"one_shot={first.get('baseline_argmax')}"
            )
            lines.append(
                "| {request_id} | {selected_h} | "
                "{prefix_argmax_mismatch_count} | {first_text} | "
                "{prefix_tokens} |".format(first_text=first_text, **record)
            )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = analyze(read_jsonl(args.baseline_events), read_jsonl(args.cv_events))
    write_outputs(records, args)
    print(
        json.dumps(
            {
                "records": len(records),
                "comparable_records": sum(
                    1 for record in records
                    if record["comparison_status"] != "no_matching_one_shot_draft"
                ),
                "argmax_mismatches": sum(
                    1 for record in records
                    if record["comparison_status"] == "argmax_mismatch"
                ),
                "output_json": str(args.output_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
