#!/usr/bin/env python3
"""Align SR24 serving replay responses with confidence trace records.

This diagnostic expects responses from `replay_sr24_serving_samples.py`, which
sets stable request ids, and a `speclink_confidence_trace.jsonl` written by the
same serving process. It avoids request-order guesses.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_trace(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if request_id:
            by_request[request_id].append(row)
    return by_request


def find_trace_rows(
    trace_by_request: dict[str, list[dict[str, Any]]],
    expected_request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    rows = trace_by_request.get(expected_request_id)
    if rows is not None:
        return expected_request_id, rows
    prefix = f"{expected_request_id}-"
    matches = [
        (request_id, rows)
        for request_id, rows in trace_by_request.items()
        if request_id.startswith(prefix)
    ]
    if len(matches) == 1:
        return matches[0]
    return "", []


def trace_output_ids(rows: list[dict[str, Any]]) -> list[int]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            step_id = int(row.get("step_id") or -1)
        except (TypeError, ValueError):
            step_id = -1
        by_step[step_id].append(row)
    output_ids: list[int] = []
    for step_id in sorted(by_step):
        step_rows = by_step[step_id]
        if not step_rows:
            continue
        values = step_rows[0].get("output_token_ids_valid") or []
        if isinstance(values, list):
            output_ids.extend(int(token_id) for token_id in values)
    return output_ids


def common_prefix_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def summarize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reached_base = 0
    accepted_base = 0
    rejected_base = 0
    accepted_residual = 0
    reached_residual = 0
    step_ids: set[int] = set()
    for row in rows:
        try:
            step_ids.add(int(row.get("step_id") or -1))
        except (TypeError, ValueError):
            pass
        reached = row.get("reached") == 1
        accepted = row.get("accepted_local") == 1
        rejected = row.get("accepted_local") == 0
        is_base = row.get("sr24_uses_residual") == 0
        is_residual = row.get("sr24_uses_residual") == 1
        if reached and is_base:
            reached_base += 1
        if accepted and is_base:
            accepted_base += 1
        if rejected and is_base:
            rejected_base += 1
        if reached and is_residual:
            reached_residual += 1
        if accepted and is_residual:
            accepted_residual += 1
    return {
        "trace_rows": len(rows),
        "trace_steps": len(step_ids),
        "trace_reached_base_only": reached_base,
        "trace_accepted_base_only": accepted_base,
        "trace_rejected_base_only": rejected_base,
        "trace_reached_residual": reached_residual,
        "trace_accepted_residual": accepted_residual,
    }


def analyze(args: argparse.Namespace) -> list[dict[str, Any]]:
    replay_rows = load_jsonl(args.replay_output)
    trace_by_request = load_trace(args.confidence_trace)
    out: list[dict[str, Any]] = []
    for replay in replay_rows:
        request_id = str(
            replay.get("expected_trace_request_id")
            or f"cmpl-{replay.get('request_id')}-0"
        )
        matched_request_id, trace_rows = find_trace_rows(trace_by_request, request_id)
        response_ids = replay.get("token_ids") or []
        if not isinstance(response_ids, list):
            response_ids = []
        response_ids = [int(token_id) for token_id in response_ids]
        traced_ids = trace_output_ids(trace_rows)
        prefix = common_prefix_len(response_ids, traced_ids)
        row = {
            "doc_id": replay.get("doc_id"),
            "sample_index": replay.get("sample_index"),
            "target_doc": replay.get("target_doc"),
            "request_id": replay.get("request_id"),
            "expected_trace_request_id": request_id,
            "matched_trace_request_id": matched_request_id,
            "trace_found": bool(trace_rows),
            "response_token_count": len(response_ids),
            "trace_output_token_count": len(traced_ids),
            "trace_response_prefix_tokens": prefix,
            "trace_reconstructs_full_response": (
                len(response_ids) == len(traced_ids) and prefix == len(response_ids)
            ),
            "first_mismatch_response_token": (
                response_ids[prefix] if prefix < len(response_ids) else ""
            ),
            "first_mismatch_trace_token": (
                traced_ids[prefix] if prefix < len(traced_ids) else ""
            ),
            "finish_reason": replay.get("finish_reason"),
            "reference_exact_match": replay.get("reference_exact_match"),
        }
        row.update(summarize_trace(trace_rows))
        out.append(row)
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


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    trace_found = sum(1 for row in rows if row.get("trace_found"))
    full = sum(1 for row in rows if row.get("trace_reconstructs_full_response"))
    accepted_base = sum(int(row.get("trace_accepted_base_only") or 0) for row in rows)
    reached_base = sum(int(row.get("trace_reached_base_only") or 0) for row in rows)
    lines = [
        "# SR24 Serving Replay Trace Alignment",
        "",
        f"- replay output: `{args.replay_output.resolve()}`",
        f"- confidence trace: `{args.confidence_trace.resolve()}`",
        f"- rows: `{len(rows)}`",
        f"- trace found: `{trace_found}`",
        f"- full response reconstructed from trace: `{full}`",
        f"- reached base-only tokens: `{reached_base}`",
        f"- accepted base-only tokens: `{accepted_base}`",
        "",
        "| doc_id | row | trace | response toks | trace toks | prefix | full | accepted base | rejected base | finish |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {doc} | {sample} | {trace} | {resp} | {traced} | {prefix} | {full} | {accepted_base} | {rejected_base} | {finish} |".format(
                doc=row.get("doc_id"),
                sample=row.get("sample_index"),
                trace=row.get("trace_found"),
                resp=row.get("response_token_count"),
                traced=row.get("trace_output_token_count"),
                prefix=row.get("trace_response_prefix_tokens"),
                full=row.get("trace_reconstructs_full_response"),
                accepted_base=row.get("trace_accepted_base_only"),
                rejected_base=row.get("trace_rejected_base_only"),
                finish=row.get("finish_reason"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--confidence-trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = analyze(args)
    write_csv(args.output_root / "trace_alignment.csv", rows)
    (args.output_root / "trace_alignment.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(args.output_root / "report.md", rows, args)
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
