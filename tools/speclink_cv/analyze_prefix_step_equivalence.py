#!/usr/bin/env python3
"""Compare h<K prefix probes with full-K one-shot verifier steps.

This diagnostic is intentionally step-local. It answers two questions from raw
debug JSONL:

1. Did the prefix probe make the same accept/reject decision that the full-K
   one-shot verifier would make for the same draft?
2. If a later full-K confirmation was requeued, did that confirmation have the
   same target argmax vector as the baseline one-shot verifier for the same
   request ordinal and draft?
"""

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
            if line:
                rows.append(json.loads(line))
    return rows


def request_ordinal(request_id: str) -> int | None:
    prefix = request_id.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def token_tuple(event: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in event.get("scheduled_spec_token_ids") or [])


def build_full_step_indexes(
    events: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]],
    dict[tuple[int, ...], list[dict[str, Any]]],
]:
    by_ordinal: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = {}
    by_tokens: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "verifier_step_debug":
            continue
        if event.get("stage") != "before_prefix_bonus_mask":
            continue
        if event.get("prefix_chunk_len") is not None:
            continue
        tokens = token_tuple(event)
        if not tokens:
            continue
        by_tokens.setdefault(tokens, []).append(event)
        ordinal = request_ordinal(str(event.get("request_id", "")))
        if ordinal is not None:
            by_ordinal.setdefault((ordinal, tokens), []).append(event)
    return by_ordinal, by_tokens


def valid_sampled_tokens(event: dict[str, Any]) -> list[int]:
    tokens: list[int] = []
    for raw in event.get("sampled_token_ids") or []:
        token = int(raw)
        if token >= 0:
            tokens.append(token)
    return tokens


def accepted_count(draft_tokens: list[int], target_argmax: list[int]) -> int:
    count = 0
    for draft, target in zip(draft_tokens, target_argmax):
        if int(draft) != int(target):
            break
        count += 1
    return count


def first_diff(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def pick_baseline(
    request_id: str,
    full_draft: tuple[int, ...],
    by_ordinal: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]],
    by_tokens: dict[tuple[int, ...], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    ordinal = request_ordinal(request_id)
    if ordinal is not None:
        matches = by_ordinal.get((ordinal, full_draft), [])
        if matches:
            return matches[0], "ordinal_and_draft"
    matches = by_tokens.get(full_draft, [])
    if matches:
        return matches[0], "draft_only"
    return None, "missing"


def analyze(
    baseline_events: list[dict[str, Any]],
    cv_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_ordinal, baseline_by_tokens = build_full_step_indexes(
        baseline_events
    )
    cv_full_by_ordinal, cv_full_by_tokens = build_full_step_indexes(cv_events)
    pending: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for event in cv_events:
        name = event.get("event")
        req_id = str(event.get("request_id", ""))
        if name == "prefix_scheduled":
            prefix = [int(x) for x in event.get("prefix_tokens") or []]
            suffix = [int(x) for x in event.get("suffix_tokens") or []]
            if prefix and suffix:
                pending[req_id] = {
                    "request_id": req_id,
                    "selected_h": int(event.get("selected_h") or len(prefix)),
                    "prefix_tokens": prefix,
                    "suffix_tokens": suffix,
                    "full_draft_tokens": prefix + suffix,
                    "queue_wait_ms": event.get("queue_wait_ms"),
                    "row_index": None,
                }
            continue

        if name != "verifier_step_debug":
            continue
        if event.get("stage") != "before_prefix_bonus_mask":
            continue
        prefix_len = event.get("prefix_chunk_len")
        if prefix_len is None:
            continue
        plan = pending.get(req_id)
        if plan is None:
            continue

        full_draft_tuple = tuple(plan["full_draft_tokens"])
        baseline, baseline_match_type = pick_baseline(
            req_id, full_draft_tuple, baseline_by_ordinal, baseline_by_tokens
        )
        cv_full, cv_full_match_type = pick_baseline(
            req_id, full_draft_tuple, cv_full_by_ordinal, cv_full_by_tokens
        )
        prefix_tokens = list(plan["prefix_tokens"])
        sampled = valid_sampled_tokens(event)
        prefix_probe_accepted = max(0, len(sampled) - 1)
        prefix_probe_accepted = min(prefix_probe_accepted, len(prefix_tokens))
        prefix_probe_full_accept = prefix_probe_accepted == len(prefix_tokens)
        prefix_target = [int(x) for x in event.get("target_argmax_token_ids") or []]
        prefix_target_accept = accepted_count(prefix_tokens, prefix_target)

        baseline_accept = None
        baseline_full_accept = None
        baseline_prefix_diff = None
        baseline_target = []
        if baseline is not None:
            baseline_target = [
                int(x) for x in baseline.get("target_argmax_token_ids") or []
            ]
            baseline_accept = accepted_count(prefix_tokens, baseline_target)
            baseline_full_accept = baseline_accept == len(prefix_tokens)
            baseline_prefix_diff = first_diff(
                prefix_target[: len(prefix_tokens)],
                baseline_target[: len(prefix_tokens)],
            )

        cv_full_diff = None
        cv_full_target = []
        if cv_full is not None and baseline is not None:
            cv_full_target = [
                int(x) for x in cv_full.get("target_argmax_token_ids") or []
            ]
            cv_full_diff = first_diff(
                cv_full_target[: len(plan["full_draft_tokens"])],
                baseline_target[: len(plan["full_draft_tokens"])],
            )

        records.append(
            {
                "request_id": req_id,
                "request_ordinal": request_ordinal(req_id),
                "selected_h": plan["selected_h"],
                "k": len(plan["full_draft_tokens"]),
                "row_index": event.get("row_index"),
                "queue_wait_ms": plan.get("queue_wait_ms"),
                "baseline_match_type": baseline_match_type,
                "cv_full_match_type": cv_full_match_type,
                "prefix_probe_accepted": prefix_probe_accepted,
                "prefix_probe_full_accept": prefix_probe_full_accept,
                "prefix_target_accept": prefix_target_accept,
                "baseline_prefix_accept": baseline_accept,
                "baseline_prefix_full_accept": baseline_full_accept,
                "prefix_decision_matches_baseline": (
                    ""
                    if baseline_full_accept is None
                    else prefix_probe_full_accept == baseline_full_accept
                ),
                "prefix_target_argmax_diff": (
                    "" if baseline_prefix_diff is None else baseline_prefix_diff
                ),
                "cv_full_target_argmax_diff": (
                    "" if cv_full_diff is None else cv_full_diff
                ),
                "full_draft_tokens": list(plan["full_draft_tokens"]),
                "prefix_target_argmax": prefix_target,
                "baseline_target_argmax": baseline_target,
                "cv_full_target_argmax": cv_full_target,
            }
        )
    return records


CSV_FIELDS = [
    "request_id",
    "request_ordinal",
    "selected_h",
    "k",
    "row_index",
    "queue_wait_ms",
    "baseline_match_type",
    "cv_full_match_type",
    "prefix_probe_accepted",
    "prefix_probe_full_accept",
    "prefix_target_accept",
    "baseline_prefix_accept",
    "baseline_prefix_full_accept",
    "prefix_decision_matches_baseline",
    "prefix_target_argmax_diff",
    "cv_full_target_argmax_diff",
    "full_draft_tokens",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in CSV_FIELDS}
            out["full_draft_tokens"] = json.dumps(
                row.get("full_draft_tokens", []), separators=(",", ":")
            )
            writer.writerow(out)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_baseline = [r for r in rows if r["baseline_match_type"] != "missing"]
    missing_baseline = [r for r in rows if r["baseline_match_type"] == "missing"]
    decision_known = [
        r for r in matched_baseline if r["prefix_decision_matches_baseline"] != ""
    ]
    decision_mismatch = [
        r for r in decision_known if not r["prefix_decision_matches_baseline"]
    ]
    prefix_argmax_mismatch = [
        r for r in matched_baseline if r["prefix_target_argmax_diff"] not in ("", None)
    ]
    cv_full_known = [
        r for r in matched_baseline if r["cv_full_match_type"] != "missing"
    ]
    cv_full_mismatch = [
        r for r in cv_full_known if r["cv_full_target_argmax_diff"] not in ("", None)
    ]
    return {
        "prefix_steps": len(rows),
        "matched_baseline_full_draft": len(matched_baseline),
        "missing_baseline_full_draft": len(missing_baseline),
        "prefix_decision_mismatches": len(decision_mismatch),
        "prefix_target_argmax_mismatches": len(prefix_argmax_mismatch),
        "cv_full_confirmations_compared": len(cv_full_known),
        "cv_full_confirmation_target_mismatches": len(cv_full_mismatch),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prefix Step Equivalence Analysis",
        "",
        f"- prefix steps: `{summary['prefix_steps']}`",
        f"- matched baseline full drafts: `{summary['matched_baseline_full_draft']}`",
        f"- missing baseline full drafts: `{summary['missing_baseline_full_draft']}`",
        f"- prefix decision mismatches: `{summary['prefix_decision_mismatches']}`",
        f"- prefix target argmax mismatches: `{summary['prefix_target_argmax_mismatches']}`",
        f"- CV full confirmations compared: `{summary['cv_full_confirmations_compared']}`",
        f"- CV full confirmation target mismatches: `{summary['cv_full_confirmation_target_mismatches']}`",
        "",
        "A missing baseline full draft means the live CV path has already changed",
        "the draft sequence relative to the one-shot baseline for that request",
        "ordinal, so later output equivalence cannot be established step-locally.",
        "",
        "## Notable Rows",
        "",
        "| request | h | baseline match | decision ok | prefix diff | full-confirm diff | draft prefix |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    notable = [
        row
        for row in rows
        if row["baseline_match_type"] == "missing"
        or row["prefix_decision_matches_baseline"] is False
        or row["prefix_target_argmax_diff"] not in ("", None)
        or row["cv_full_target_argmax_diff"] not in ("", None)
    ]
    for row in notable[:50]:
        draft = row.get("full_draft_tokens", [])[: row.get("selected_h", 0)]
        lines.append(
            "| {request_id} | {selected_h} | {baseline_match_type} | "
            "{prefix_decision_matches_baseline} | {prefix_target_argmax_diff} | "
            "{cv_full_target_argmax_diff} | {draft} |".format(
                draft=draft,
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = analyze(read_jsonl(args.baseline_events), read_jsonl(args.cv_events))
    summary = summarize(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
