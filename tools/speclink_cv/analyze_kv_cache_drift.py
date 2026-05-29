#!/usr/bin/env python3
"""Compare bounded logical-position KV checksums at a mismatch.

The verifier metadata can match while logits differ if the current query reads
different historical KV content. This diagnostic consumes a token timeline and
debug JSONL files produced with bounded ``--kv-debug-*`` options. Newer debug
events include both a legacy seq-tail checksum window and a pre-target history
window; by default this tool compares the history window when present.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from tools.speclink_cv.analyze_token_timeline import read_jsonl
except ModuleNotFoundError:  # pragma: no cover - script-path execution
    from analyze_token_timeline import read_jsonl  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-json", type=Path, required=True)
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--cv-events", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--checksum-field",
        choices=["auto", "history", "tail"],
        default="auto",
        help=(
            "Which event checksum field to compare. auto prefers the bounded "
            "history window before the first verifier target position, then "
            "falls back to the legacy tail window."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_at(events: list[dict[str, Any]], index: Any) -> dict[str, Any]:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        raise SystemExit(f"invalid verifier event index: {index!r}") from None
    if idx < 0 or idx >= len(events):
        raise SystemExit(f"verifier event index out of range: {idx}")
    return events[idx]


def checksum_index(
    event: dict[str, Any],
    *,
    field_name: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for pos_row in event.get(field_name) or []:
        logical_pos = int(pos_row.get("logical_pos"))
        for layer_row in pos_row.get("layers") or []:
            layer = int(layer_row.get("layer"))
            rows[(logical_pos, layer)] = {
                "logical_pos": logical_pos,
                "token_id": pos_row.get("token_id"),
                "block_id": pos_row.get("block_id"),
                "block_offset": pos_row.get("block_offset"),
                "physical_slot": pos_row.get("physical_slot"),
                "layer": layer,
                "k_sum": layer_row.get("k_sum"),
                "k_abs_sum": layer_row.get("k_abs_sum"),
                "k_max_abs": layer_row.get("k_max_abs"),
                "v_sum": layer_row.get("v_sum"),
                "v_abs_sum": layer_row.get("v_abs_sum"),
                "v_max_abs": layer_row.get("v_max_abs"),
            }
    return rows


def first_mismatch_segment(timeline: dict[str, Any], source: str) -> dict[str, Any]:
    key = (
        "baseline_segment_at_mismatch"
        if source == "baseline"
        else "speclink_cv_segment_at_mismatch"
    )
    segment = timeline.get(key)
    if not segment:
        raise SystemExit(f"missing {key} in token timeline")
    return segment


CHECKSUM_FIELDS = [
    "k_sum",
    "k_abs_sum",
    "k_max_abs",
    "v_sum",
    "v_abs_sum",
    "v_max_abs",
]


def compare_rows(
    baseline_event: dict[str, Any],
    cv_event: dict[str, Any],
    *,
    field_name: str,
    atol: float,
) -> list[dict[str, Any]]:
    baseline = checksum_index(baseline_event, field_name=field_name)
    cv = checksum_index(cv_event, field_name=field_name)
    keys = sorted(set(baseline) | set(cv))
    rows: list[dict[str, Any]] = []
    for key in keys:
        left = baseline.get(key)
        right = cv.get(key)
        row: dict[str, Any] = {
            "logical_pos": key[0],
            "layer": key[1],
            "present_baseline": left is not None,
            "present_cv": right is not None,
            "token_id_baseline": None if left is None else left.get("token_id"),
            "token_id_cv": None if right is None else right.get("token_id"),
            "block_id_baseline": None if left is None else left.get("block_id"),
            "block_id_cv": None if right is None else right.get("block_id"),
            "block_offset_baseline": None
            if left is None
            else left.get("block_offset"),
            "block_offset_cv": None if right is None else right.get("block_offset"),
            "physical_slot_baseline": None
            if left is None
            else left.get("physical_slot"),
            "physical_slot_cv": None if right is None else right.get("physical_slot"),
            "checksum_equal": left is not None and right is not None,
            "max_abs_delta": "",
        }
        max_delta = 0.0
        if left is None or right is None:
            row["checksum_equal"] = False
        else:
            for field in CHECKSUM_FIELDS:
                l_val = float(left.get(field) or 0.0)
                r_val = float(right.get(field) or 0.0)
                delta = abs(l_val - r_val)
                max_delta = max(max_delta, delta)
                row[f"{field}_baseline"] = l_val
                row[f"{field}_cv"] = r_val
                row[f"{field}_delta"] = delta
            row["max_abs_delta"] = max_delta
            row["checksum_equal"] = max_delta <= atol
        rows.append(row)
    return rows


CSV_FIELDS = [
    "logical_pos",
    "layer",
    "present_baseline",
    "present_cv",
    "checksum_equal",
    "max_abs_delta",
    "token_id_baseline",
    "token_id_cv",
    "block_id_baseline",
    "block_id_cv",
    "block_offset_baseline",
    "block_offset_cv",
    "physical_slot_baseline",
    "physical_slot_cv",
    *[
        f"{field}_{suffix}"
        for field in CHECKSUM_FIELDS
        for suffix in ("baseline", "cv", "delta")
    ],
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def write_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    first = summary.get("first_checksum_mismatch")
    lines = [
        "# KV Cache Drift Diagnostic",
        "",
        f"- request_index: `{summary['request_index']}`",
        f"- first_mismatch_token_index: `{summary['first_mismatch_token_index']}`",
        f"- baseline_verifier_event_index: `{summary['baseline_verifier_event_index']}`",
        f"- cv_verifier_event_index: `{summary['cv_verifier_event_index']}`",
        f"- checksum_field: `{summary['checksum_field']}`",
        f"- baseline_upto_position: `{summary.get('baseline_upto_position', '')}`",
        f"- cv_upto_position: `{summary.get('cv_upto_position', '')}`",
        f"- compared_entries: `{summary['compared_entries']}`",
        f"- checksum_mismatches: `{summary['checksum_mismatches']}`",
        f"- missing_checksum_entries: `{summary['missing_checksum_entries']}`",
        "",
    ]
    if first:
        lines.extend(
            [
                "First checksum mismatch:",
                "",
                f"- logical_pos: `{first['logical_pos']}`",
                f"- layer: `{first['layer']}`",
                f"- token_id baseline/CV: `{first['token_id_baseline']}` / `{first['token_id_cv']}`",
                f"- block_id baseline/CV: `{first['block_id_baseline']}` / `{first['block_id_cv']}`",
                f"- physical_slot baseline/CV: `{first['physical_slot_baseline']}` / `{first['physical_slot_cv']}`",
                f"- max_abs_delta: `{first['max_abs_delta']}`",
                "",
            ]
        )
    lines.extend(
        [
            "| logical pos | layer | token | block base | block CV | slot base | slot CV | equal | max delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in rows[:50]:
        lines.append(
            "| {logical_pos} | {layer} | {token_id_baseline} | "
            "{block_id_baseline} | {block_id_cv} | {physical_slot_baseline} | "
            "{physical_slot_cv} | {checksum_equal} | {max_abs_delta} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    timeline = read_json(args.timeline_json)
    baseline_segment = first_mismatch_segment(timeline, "baseline")
    cv_segment = first_mismatch_segment(timeline, "speclink_cv")
    baseline_event = event_at(
        read_jsonl(args.baseline_events),
        baseline_segment.get("verifier_event_index"),
    )
    cv_event = event_at(
        read_jsonl(args.cv_events),
        cv_segment.get("verifier_event_index"),
    )
    if args.checksum_field == "history":
        field_name = "history_kv_cache_checksums"
    elif args.checksum_field == "tail":
        field_name = "kv_cache_checksums"
    else:
        history_available = bool(
            baseline_event.get("history_kv_cache_checksums")
            and cv_event.get("history_kv_cache_checksums")
        )
        field_name = (
            "history_kv_cache_checksums"
            if history_available
            else "kv_cache_checksums"
        )
    rows = compare_rows(
        baseline_event,
        cv_event,
        field_name=field_name,
        atol=args.atol,
    )
    mismatches = [row for row in rows if not row["checksum_equal"]]
    missing = [
        row
        for row in rows
        if not row["present_baseline"] or not row["present_cv"]
    ]
    summary = {
        "request_index": timeline.get("request_index"),
        "first_mismatch_token_index": timeline.get("first_mismatch_token_index"),
        "baseline_verifier_event_index": baseline_segment.get(
            "verifier_event_index"
        ),
        "cv_verifier_event_index": cv_segment.get("verifier_event_index"),
        "checksum_field": field_name,
        "baseline_upto_position": baseline_event.get(
            "history_kv_cache_upto_position"
            if field_name == "history_kv_cache_checksums"
            else "tail_kv_cache_upto_position"
        ),
        "cv_upto_position": cv_event.get(
            "history_kv_cache_upto_position"
            if field_name == "history_kv_cache_checksums"
            else "tail_kv_cache_upto_position"
        ),
        "compared_entries": len(rows),
        "checksum_mismatches": len(mismatches),
        "missing_checksum_entries": len(missing),
        "first_checksum_mismatch": mismatches[0] if mismatches else None,
    }
    payload = {"summary": summary, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_md(args.output_md, payload)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
