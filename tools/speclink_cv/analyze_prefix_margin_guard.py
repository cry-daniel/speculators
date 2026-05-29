#!/usr/bin/env python3
"""Retrospective analysis for the SpecLink-CV prefix margin fallback.

The runtime low-margin guard falls back to a full one-shot confirmation when
the minimum top1-top2 verifier logit margin across prefix target rows and the
bonus row is below a threshold.  This script reads a debug event JSONL and
reports how many live h<K prefix chunks would be converted to one-shot
confirmation for a threshold sweep.

This is a diagnostic only.  A promising threshold still needs a live GPU
correctness run because the guard changes the subsequent scheduler trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = "0,0.125,0.25,0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-events", type=Path, required=True)
    parser.add_argument("--correctness-json", type=Path)
    parser.add_argument("--prefix-equivalence-json", type=Path)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
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


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def request_ordinal(request_id: str) -> int | None:
    prefix = request_id.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return sorted(set(values))


def top2_margin(values: Any) -> float | None:
    if not isinstance(values, list) or len(values) < 2:
        return None
    return float(values[0]) - float(values[1])


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def valid_sampled_tokens(event: dict[str, Any]) -> list[int]:
    tokens: list[int] = []
    for raw in event.get("sampled_token_ids") or []:
        token = int(raw)
        if token >= 0:
            tokens.append(token)
    return tokens


def load_final_matches(correctness: dict[str, Any]) -> dict[int, bool]:
    matched_items = correctness.get("matched_items")
    if not isinstance(matched_items, list):
        return {}
    return {index: bool(value) for index, value in enumerate(matched_items)}


def load_prefix_equivalence(
    data: dict[str, Any],
) -> dict[tuple[str, tuple[int, ...]], dict[str, Any]]:
    index: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for row in data.get("rows") or []:
        req_id = str(row.get("request_id", ""))
        full_draft = tuple(int(x) for x in row.get("full_draft_tokens") or [])
        if req_id and full_draft:
            index[(req_id, full_draft)] = row
    return index


def analyze(
    events: list[dict[str, Any]],
    final_matches: dict[int, bool],
    equivalence_index: dict[tuple[str, tuple[int, ...]], dict[str, Any]],
) -> list[dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    step_index = 0
    rows: list[dict[str, Any]] = []

    for event in events:
        name = event.get("event")
        req_id = str(event.get("request_id", ""))
        if name == "prefix_scheduled":
            prefix = [int(x) for x in event.get("prefix_tokens") or []]
            suffix = [int(x) for x in event.get("suffix_tokens") or []]
            if prefix and suffix:
                pending[req_id] = {
                    "request_id": req_id,
                    "request_ordinal": request_ordinal(req_id),
                    "selected_h": int(event.get("selected_h") or len(prefix)),
                    "k": int(event.get("k") or len(prefix) + len(suffix)),
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
        plan = pending.get(req_id)
        if plan is None:
            continue

        prefix_len = int(prefix_len)
        target_margins: list[float] = []
        for values in (event.get("target_topk_values") or [])[:prefix_len]:
            margin = top2_margin(values)
            if margin is not None:
                target_margins.append(margin)
        bonus_margin = top2_margin(event.get("bonus_topk_values") or [])
        margins = list(target_margins)
        if bonus_margin is not None:
            margins.append(bonus_margin)
        min_margin = min(margins) if margins else None
        min_margin_source = ""
        if min_margin is not None:
            min_margin_source = (
                "bonus"
                if bonus_margin is not None and min_margin == bonus_margin
                else f"target_{target_margins.index(min_margin)}"
            )

        sampled = valid_sampled_tokens(event)
        accepted = max(0, len(sampled) - 1)
        accepted = min(accepted, prefix_len)
        ordinal = plan["request_ordinal"]
        full_draft = tuple(plan["full_draft_tokens"])
        equiv = equivalence_index.get((req_id, full_draft), {})
        final_matched = (
            final_matches.get(ordinal) if ordinal is not None else None
        )

        rows.append(
            {
                "step_index": step_index,
                "request_id": req_id,
                "request_ordinal": ordinal,
                "final_output_matched": final_matched,
                "selected_h": plan["selected_h"],
                "k": plan["k"],
                "prefix_len": prefix_len,
                "suffix_len": len(plan["suffix_tokens"]),
                "prefix_probe_accepted": accepted,
                "prefix_probe_full_accept": accepted == prefix_len,
                "min_margin": min_margin,
                "min_margin_source": min_margin_source,
                "target_min_margin": (
                    min(target_margins) if target_margins else None
                ),
                "bonus_margin": bonus_margin,
                "target_margins": target_margins,
                "queue_wait_ms": plan.get("queue_wait_ms"),
                "baseline_match_type": equiv.get("baseline_match_type", ""),
                "prefix_decision_matches_baseline": equiv.get(
                    "prefix_decision_matches_baseline", ""
                ),
                "prefix_target_argmax_diff": equiv.get(
                    "prefix_target_argmax_diff", ""
                ),
                "full_draft_tokens": list(full_draft),
                "prefix_tokens": plan["prefix_tokens"],
                "suffix_tokens": plan["suffix_tokens"],
            }
        )
        step_index += 1

    return rows


def summarize_thresholds(
    rows: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    all_ordinals = {
        int(row["request_ordinal"])
        for row in rows
        if row.get("request_ordinal") is not None
    }
    failed_ordinals = {
        int(row["request_ordinal"])
        for row in rows
        if row.get("request_ordinal") is not None
        and row.get("final_output_matched") is False
    }
    total = len(rows)
    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        triggered = [
            row
            for row in rows
            if row.get("min_margin") is not None
            and float(row["min_margin"]) <= threshold
        ]
        triggered_ordinals = {
            int(row["request_ordinal"])
            for row in triggered
            if row.get("request_ordinal") is not None
        }
        failed_triggered = failed_ordinals & triggered_ordinals
        matched_triggered = (all_ordinals - failed_ordinals) & triggered_ordinals
        summaries.append(
            {
                "threshold": threshold,
                "fallback_chunks": len(triggered),
                "fallback_fraction": len(triggered) / total if total else 0.0,
                "live_hk_chunks_remaining": total - len(triggered),
                "live_hk_fraction_remaining": (
                    (total - len(triggered)) / total if total else 0.0
                ),
                "triggered_ordinals": sorted(triggered_ordinals),
                "failed_ordinals_triggered": sorted(failed_triggered),
                "failed_ordinals_missed": sorted(failed_ordinals - triggered_ordinals),
                "matched_ordinals_triggered": sorted(matched_triggered),
                "final_failure_recall": (
                    len(failed_triggered) / len(failed_ordinals)
                    if failed_ordinals
                    else None
                ),
                "final_failure_precision": (
                    len(failed_triggered) / len(triggered_ordinals)
                    if triggered_ordinals
                    else None
                ),
            }
        )
    return summaries


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [
        float(row["min_margin"])
        for row in rows
        if row.get("min_margin") is not None
    ]
    final_failed = sorted(
        {
            int(row["request_ordinal"])
            for row in rows
            if row.get("request_ordinal") is not None
            and row.get("final_output_matched") is False
        }
    )
    final_matched = sorted(
        {
            int(row["request_ordinal"])
            for row in rows
            if row.get("request_ordinal") is not None
            and row.get("final_output_matched") is True
        }
    )
    comparable = [
        row for row in rows if row.get("baseline_match_type") not in ("", "missing")
    ]
    missing = [row for row in rows if row.get("baseline_match_type") == "missing"]
    return {
        "prefix_chunks": len(rows),
        "final_failed_ordinals": final_failed,
        "final_matched_ordinals": final_matched,
        "comparable_prefix_chunks": len(comparable),
        "missing_baseline_prefix_chunks": len(missing),
        "prefix_full_accept_chunks": sum(
            1 for row in rows if row.get("prefix_probe_full_accept")
        ),
        "prefix_reject_chunks": sum(
            1 for row in rows if not row.get("prefix_probe_full_accept")
        ),
        "min_margin": {
            "count": len(margins),
            "min": min(margins) if margins else None,
            "p10": quantile(margins, 0.10),
            "p25": quantile(margins, 0.25),
            "p50": quantile(margins, 0.50),
            "p75": quantile(margins, 0.75),
            "p90": quantile(margins, 0.90),
            "max": max(margins) if margins else None,
        },
    }


CSV_FIELDS = [
    "step_index",
    "request_id",
    "request_ordinal",
    "final_output_matched",
    "selected_h",
    "k",
    "prefix_probe_accepted",
    "prefix_probe_full_accept",
    "min_margin",
    "min_margin_source",
    "target_min_margin",
    "bonus_margin",
    "baseline_match_type",
    "prefix_decision_matches_baseline",
    "prefix_target_argmax_diff",
    "queue_wait_ms",
    "target_margins",
    "prefix_tokens",
    "suffix_tokens",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in CSV_FIELDS}
            for key in ("target_margins", "prefix_tokens", "suffix_tokens"):
                out[key] = json.dumps(out[key], separators=(",", ":"))
            writer.writerow(out)


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4g}"


def write_markdown(
    path: Path,
    row_summary: dict[str, Any],
    threshold_summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin = row_summary["min_margin"]
    lines = [
        "# Prefix Margin Guard Analysis",
        "",
        "This is an offline diagnostic. It estimates which h<K prefix chunks",
        "would fall back to one-shot confirmation for each margin threshold.",
        "A threshold must still be validated by a live correctness smoke.",
        "",
        f"- prefix chunks: `{row_summary['prefix_chunks']}`",
        f"- final failed request ordinals: `{row_summary['final_failed_ordinals']}`",
        f"- comparable prefix chunks: `{row_summary['comparable_prefix_chunks']}`",
        f"- missing baseline prefix chunks: `{row_summary['missing_baseline_prefix_chunks']}`",
        f"- prefix full-accept chunks: `{row_summary['prefix_full_accept_chunks']}`",
        f"- prefix reject chunks: `{row_summary['prefix_reject_chunks']}`",
        f"- min margin p10/p50/p90: `{fmt_float(margin['p10'])}` / "
        f"`{fmt_float(margin['p50'])}` / `{fmt_float(margin['p90'])}`",
        "",
        "## Threshold Sweep",
        "",
        "| threshold | fallback chunks | live chunks left | failure recall | failure precision | failed missed | matched degraded |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in threshold_summary:
        lines.append(
            "| {threshold:.4g} | {fallback_chunks} ({fallback_fraction:.1%}) | "
            "{live_hk_chunks_remaining} ({live_hk_fraction_remaining:.1%}) | "
            "{recall} | {precision} | {missed} | {matched} |".format(
                threshold=float(item["threshold"]),
                fallback_chunks=item["fallback_chunks"],
                fallback_fraction=item["fallback_fraction"],
                live_hk_chunks_remaining=item["live_hk_chunks_remaining"],
                live_hk_fraction_remaining=item["live_hk_fraction_remaining"],
                recall=fmt_float(item["final_failure_recall"]),
                precision=fmt_float(item["final_failure_precision"]),
                missed=item["failed_ordinals_missed"],
                matched=item["matched_ordinals_triggered"],
            )
        )

    notable = [
        row
        for row in rows
        if row.get("final_output_matched") is False
        or row.get("min_margin_source") == "bonus"
        or row.get("baseline_match_type") == "missing"
    ]
    lines.extend(
        [
            "",
            "## Notable Prefix Chunks",
            "",
            "| step | request | final ok | h | accepted | min margin | source | baseline match | prefix tokens |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in notable[:80]:
        display = dict(row)
        display["min_margin_display"] = fmt_float(row.get("min_margin"))
        display["prefix_tokens_display"] = row.get("prefix_tokens", [])[
            : row.get("selected_h", 0)
        ]
        lines.append(
            "| {step_index} | {request_id} | {final_output_matched} | "
            "{selected_h} | {prefix_probe_accepted} | "
            "{min_margin_display} | {min_margin_source} | "
            "{baseline_match_type} | {prefix_tokens_display} |".format(
                **display,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    rows = analyze(
        read_jsonl(args.cv_events),
        load_final_matches(read_json(args.correctness_json)),
        load_prefix_equivalence(read_json(args.prefix_equivalence_json)),
    )
    row_summary = summarize_rows(rows)
    threshold_summary = summarize_thresholds(rows, thresholds)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "summary": row_summary,
                "thresholds": threshold_summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, row_summary, threshold_summary, rows)
    print(
        json.dumps(
            {
                "prefix_chunks": row_summary["prefix_chunks"],
                "final_failed_ordinals": row_summary["final_failed_ordinals"],
                "output_json": str(args.output_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
