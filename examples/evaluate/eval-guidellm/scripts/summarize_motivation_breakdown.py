#!/usr/bin/env python3
"""Summarize motivation breakdown experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


SPEC_RE = re.compile(
    r"SpecDecoding metrics: .*?"
    r"Mean acceptance length: (?P<mean>[0-9.]+), "
    r"Accepted throughput: (?P<accepted_tps>[0-9.]+) tokens/s, "
    r"Drafted throughput: (?P<drafted_tps>[0-9.]+) tokens/s, "
    r"Accepted: (?P<accepted>[0-9]+) tokens, "
    r"Drafted: (?P<drafted>[0-9]+) tokens, "
    r"Per-position acceptance rate: (?P<rates>[0-9., ]+), "
    r"Avg Draft acceptance rate: (?P<avg>[0-9.]+)%"
)


SUMMARY_FIELDS = [
    "algo",
    "batch_size",
    "num_spec_tokens",
    "status",
    "run_dir",
    "prompt_tokens",
    "output_tokens",
    "requests",
    "guidellm_output_tps",
    "guidellm_total_tps",
    "guidellm_latency_mean_ms",
    "guidellm_completed_requests",
    "guidellm_completed_output_tokens",
    "events_total",
    "decode_events",
    "avg_active_requests",
    "avg_scheduled_tokens",
    "decode_total_ms_per_iter",
    "decode_verify_ms_per_iter",
    "decode_verify_qkv_proj_ms_per_iter",
    "decode_verify_attention_ms_per_iter",
    "decode_verify_ffn_ms_per_iter",
    "decode_verify_model_other_ms_per_iter",
    "decode_draft_ms_per_iter",
    "decode_accept_reject_ms_per_iter",
    "decode_other_ms_per_iter",
    "decode_verify_pct",
    "decode_verify_qkv_proj_pct_of_verify",
    "decode_verify_attention_pct_of_verify",
    "decode_verify_ffn_pct_of_verify",
    "decode_verify_model_other_pct_of_verify",
    "decode_draft_pct",
    "decode_accept_reject_pct",
    "decode_other_pct",
    "verify_detail_events",
    "prefill_events",
    "prefill_total_ms",
    "prefill_verify_ms",
    "draft_tokens",
    "spec_accepted_tokens",
    "spec_drafted_tokens",
    "spec_avg_acceptance_pct",
    "spec_mean_acceptance_length",
]


CONCISE_FIELDS = [
    "model",
    "batch_size",
    "num_spec_tokens",
    "decode_verify_pct",
    "decode_draft_pct",
    "decode_others_pct",
    "tokens_per_decode_iter",
    "e2e_latency_mean_s",
]


VERIFY_DETAIL_FIELDS = [
    "model",
    "batch_size",
    "num_spec_tokens",
    "verify_qkv_proj_pct",
    "verify_attention_pct",
    "verify_ffn_pct",
    "verify_model_other_pct",
    "verify_qkv_proj_ms_per_iter",
    "verify_attention_ms_per_iter",
    "verify_ffn_ms_per_iter",
    "verify_model_other_ms_per_iter",
    "verify_total_ms_per_iter",
]


RAW_EVENT_FIELDS = [
    "algo",
    "batch_size",
    "num_spec_tokens",
    "phase",
    "ts",
    "active_requests",
    "scheduled_tokens",
    "max_scheduled_tokens",
    "num_tokens_unpadded",
    "num_tokens_padded",
    "num_draft_tokens",
    "verify_forward_ms",
    "verify_qkv_proj_ms",
    "verify_attention_ms",
    "verify_ffn_ms",
    "verify_model_other_ms",
    "draft_forward_ms",
    "accept_reject_ms",
    "other_ms",
    "measured_total_ms",
    "use_spec_decode",
    "spec_method",
    "run_dir",
]


@dataclass
class SpecMetrics:
    accepted_tokens: int = 0
    drafted_tokens: int = 0
    avg_acceptance_pct: float | None = None
    mean_acceptance_length: float | None = None
    weighted_rates: list[float] | None = None


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def sum_field(rows: list[dict[str, Any]], key: str) -> float:
    return sum(to_float(row.get(key)) or 0.0 for row in rows)


def mean_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [to_float(row.get(key)) for row in rows]
    return mean([value for value in values if value is not None])


def first_path(data: Any, paths: list[list[str]]) -> Any:
    for path in paths:
        value = data
        ok = True
        for part in path:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                if idx >= len(value):
                    ok = False
                    break
                value = value[idx]
            else:
                ok = False
                break
        if ok:
            return value
    return None


def find_numeric_by_suffix(data: Any, suffix: tuple[str, ...]) -> float | None:
    found: list[float] = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + (str(key),))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                walk(item, path + (str(idx),))
        elif path[-len(suffix) :] == suffix:
            number = to_float(value)
            if number is not None:
                found.append(number)

    walk(data, ())
    return found[0] if found else None


def parse_guidellm_json(path: Path) -> dict[str, float | None]:
    data = load_json(path)
    if data is None:
        return {
            "guidellm_output_tps": None,
            "guidellm_total_tps": None,
            "guidellm_latency_mean_ms": None,
            "guidellm_completed_requests": None,
            "guidellm_completed_output_tokens": None,
        }
    latency_ms = (
        to_float(
            first_path(
                data,
                [
                    [
                        "benchmarks",
                        "0",
                        "metrics",
                        "request_latency_ms",
                        "successful",
                        "mean",
                    ],
                    ["metrics", "request_latency_ms", "successful", "mean"],
                ],
            )
        )
        or find_numeric_by_suffix(
            data, ("request_latency_ms", "successful", "mean")
        )
    )
    latency_s = (
        to_float(
            first_path(
                data,
                [
                    [
                        "benchmarks",
                        "0",
                        "metrics",
                        "request_latency",
                        "successful",
                        "mean",
                    ],
                    ["metrics", "request_latency", "successful", "mean"],
                ],
            )
        )
        or find_numeric_by_suffix(data, ("request_latency", "successful", "mean"))
    )
    if latency_ms is None and latency_s is not None:
        latency_ms = latency_s * 1000.0

    return {
        "guidellm_output_tps": (
            to_float(
                first_path(
                    data,
                    [
                        [
                            "benchmarks",
                            "0",
                            "metrics",
                            "output_tokens_per_second",
                            "successful",
                            "mean",
                        ],
                        [
                            "metrics",
                            "output_tokens_per_second",
                            "successful",
                            "mean",
                        ],
                    ],
                )
            )
            or find_numeric_by_suffix(
                data, ("output_tokens_per_second", "successful", "mean")
            )
        ),
        "guidellm_total_tps": (
            to_float(
                first_path(
                    data,
                    [
                        [
                            "benchmarks",
                            "0",
                            "metrics",
                            "tokens_per_second",
                            "successful",
                            "mean",
                        ],
                        ["metrics", "tokens_per_second", "successful", "mean"],
                    ],
                )
            )
            or find_numeric_by_suffix(data, ("tokens_per_second", "successful", "mean"))
        ),
        "guidellm_latency_mean_ms": (
            latency_ms
        ),
        "guidellm_completed_requests": (
            to_float(
                first_path(
                    data,
                    [
                        ["benchmarks", "0", "metrics", "request_totals", "successful"],
                        ["metrics", "request_totals", "successful"],
                    ],
                )
            )
            or find_numeric_by_suffix(data, ("request_totals", "successful"))
        ),
        "guidellm_completed_output_tokens": (
            to_float(
                first_path(
                    data,
                    [
                        [
                            "benchmarks",
                            "0",
                            "metrics",
                            "output_token_count",
                            "successful",
                            "total_sum",
                        ],
                        [
                            "metrics",
                            "output_token_count",
                            "successful",
                            "total_sum",
                        ],
                    ],
                )
            )
            or find_numeric_by_suffix(
                data, ("output_token_count", "successful", "total_sum")
            )
        ),
    }


def parse_guidellm_log(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    match = re.search(r"Output throughput:\s*([0-9.]+)", text)
    if match:
        out["guidellm_output_tps"] = float(match.group(1))
    match = re.search(r"Total throughput:\s*([0-9.]+)", text)
    if match:
        out["guidellm_total_tps"] = float(match.group(1))
    return out


def parse_spec_metrics(path: Path) -> SpecMetrics:
    metrics = SpecMetrics()
    if not path.exists():
        return metrics
    sections = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in SPEC_RE.finditer(text):
        drafted = int(match.group("drafted"))
        accepted = int(match.group("accepted"))
        rates = [float(item.strip()) for item in match.group("rates").split(",")]
        sections.append(
            {
                "drafted": drafted,
                "accepted": accepted,
                "mean": float(match.group("mean")),
                "avg": float(match.group("avg")),
                "rates": rates,
            }
        )
    if not sections:
        return metrics

    metrics.accepted_tokens = sum(item["accepted"] for item in sections)
    metrics.drafted_tokens = sum(item["drafted"] for item in sections)
    if metrics.drafted_tokens:
        metrics.avg_acceptance_pct = (
            metrics.accepted_tokens / metrics.drafted_tokens * 100.0
        )
    metrics.mean_acceptance_length = mean([item["mean"] for item in sections])

    max_len = max(len(item["rates"]) for item in sections)
    weighted_rates = []
    for idx in range(max_len):
        numerator = 0.0
        denominator = 0
        for item in sections:
            if idx < len(item["rates"]):
                numerator += item["rates"][idx] * item["drafted"]
                denominator += item["drafted"]
        weighted_rates.append(numerator / denominator if denominator else 0.0)
    metrics.weighted_rates = weighted_rates
    return metrics


def summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    meta = load_json(run_dir / "metadata.json") or {}
    events = load_jsonl(run_dir / "breakdown_events.jsonl")
    for event in events:
        event["run_dir"] = str(run_dir)

    decode_events = [
        event
        for event in events
        if event.get("phase") == "decode" and event.get("use_spec_decode") is True
    ]
    if not decode_events:
        decode_events = [event for event in events if event.get("phase") == "decode"]
    detail_keys = [
        "verify_qkv_proj_ms",
        "verify_attention_ms",
        "verify_ffn_ms",
        "verify_model_other_ms",
    ]
    verify_detail_events = [
        event
        for event in decode_events
        if any(to_float(event.get(key)) is not None for key in detail_keys)
    ]
    prefill_events = [event for event in events if event.get("phase") == "prefill"]

    decode_total = sum_field(decode_events, "measured_total_ms")
    decode_verify = sum_field(decode_events, "verify_forward_ms")
    decode_verify_qkv = (
        sum_field(decode_events, "verify_qkv_proj_ms")
        if verify_detail_events
        else None
    )
    decode_verify_attention = (
        sum_field(decode_events, "verify_attention_ms")
        if verify_detail_events
        else None
    )
    decode_verify_ffn = (
        sum_field(decode_events, "verify_ffn_ms") if verify_detail_events else None
    )
    decode_verify_model_other = (
        sum_field(decode_events, "verify_model_other_ms")
        if verify_detail_events
        else None
    )
    decode_draft = sum_field(decode_events, "draft_forward_ms")
    decode_accept = sum_field(decode_events, "accept_reject_ms")
    decode_other = sum_field(decode_events, "other_ms")
    decode_count = len(decode_events)

    def per_iter(value: float) -> float | None:
        return value / decode_count if decode_count else None

    def maybe_per_iter(value: float | None) -> float | None:
        return value / decode_count if value is not None and decode_count else None

    def pct(value: float) -> float | None:
        return value / decode_total if decode_total else None

    def verify_pct(value: float | None) -> float | None:
        return value / decode_verify if value is not None and decode_verify else None

    guidellm = parse_guidellm_json(run_dir / "guidellm_results.json")
    guidellm.update(
        {
            key: value
            for key, value in parse_guidellm_log(run_dir / "guidellm_output.log").items()
            if value is not None
        }
    )
    spec = parse_spec_metrics(run_dir / "vllm_server.log")

    row = {
        "algo": meta.get("algo"),
        "batch_size": meta.get("batch_size"),
        "num_spec_tokens": meta.get("num_spec_tokens"),
        "status": meta.get("status", "unknown"),
        "run_dir": str(run_dir),
        "prompt_tokens": meta.get("prompt_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "requests": meta.get("requests"),
        "guidellm_output_tps": guidellm.get("guidellm_output_tps"),
        "guidellm_total_tps": guidellm.get("guidellm_total_tps"),
        "guidellm_latency_mean_ms": guidellm.get("guidellm_latency_mean_ms"),
        "guidellm_completed_requests": guidellm.get("guidellm_completed_requests"),
        "guidellm_completed_output_tokens": guidellm.get(
            "guidellm_completed_output_tokens"
        ),
        "events_total": len(events),
        "decode_events": decode_count,
        "avg_active_requests": mean_field(decode_events, "active_requests"),
        "avg_scheduled_tokens": mean_field(decode_events, "scheduled_tokens"),
        "decode_total_ms_per_iter": per_iter(decode_total),
        "decode_verify_ms_per_iter": per_iter(decode_verify),
        "decode_verify_qkv_proj_ms_per_iter": maybe_per_iter(decode_verify_qkv),
        "decode_verify_attention_ms_per_iter": maybe_per_iter(
            decode_verify_attention
        ),
        "decode_verify_ffn_ms_per_iter": maybe_per_iter(decode_verify_ffn),
        "decode_verify_model_other_ms_per_iter": maybe_per_iter(
            decode_verify_model_other
        ),
        "decode_draft_ms_per_iter": per_iter(decode_draft),
        "decode_accept_reject_ms_per_iter": per_iter(decode_accept),
        "decode_other_ms_per_iter": per_iter(decode_other),
        "decode_verify_pct": pct(decode_verify),
        "decode_verify_qkv_proj_pct_of_verify": verify_pct(decode_verify_qkv),
        "decode_verify_attention_pct_of_verify": verify_pct(
            decode_verify_attention
        ),
        "decode_verify_ffn_pct_of_verify": verify_pct(decode_verify_ffn),
        "decode_verify_model_other_pct_of_verify": verify_pct(
            decode_verify_model_other
        ),
        "decode_draft_pct": pct(decode_draft),
        "decode_accept_reject_pct": pct(decode_accept),
        "decode_other_pct": pct(decode_other),
        "verify_detail_events": len(verify_detail_events),
        "prefill_events": len(prefill_events),
        "prefill_total_ms": sum_field(prefill_events, "measured_total_ms"),
        "prefill_verify_ms": sum_field(prefill_events, "verify_forward_ms"),
        "draft_tokens": sum(
            int(to_float(event.get("num_draft_tokens")) or 0)
            for event in decode_events
        ),
        "spec_accepted_tokens": spec.accepted_tokens,
        "spec_drafted_tokens": spec.drafted_tokens,
        "spec_avg_acceptance_pct": spec.avg_acceptance_pct,
        "spec_mean_acceptance_length": spec.mean_acceptance_length,
    }

    acceptance_rows = []
    if spec.weighted_rates:
        for idx, rate in enumerate(spec.weighted_rates):
            acceptance_rows.append(
                {
                    "algo": row["algo"],
                    "batch_size": row["batch_size"],
                    "num_spec_tokens": row["num_spec_tokens"],
                    "position": idx,
                    "acceptance_rate": rate,
                    "run_dir": str(run_dir),
                }
            )

    return row, events, acceptance_rows


def build_concise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rounded(value: float | None, digits: int) -> float | None:
        return round(value, digits) if value is not None else None

    concise_rows = []
    for row in rows:
        decode_events = to_float(row.get("decode_events")) or 0.0
        completed_output_tokens = to_float(
            row.get("guidellm_completed_output_tokens")
        )
        if completed_output_tokens is None:
            completed_output_tokens = (
                (to_float(row.get("output_tokens")) or 0.0)
                * (to_float(row.get("requests")) or 0.0)
            )
        accept_pct = to_float(row.get("decode_accept_reject_pct")) or 0.0
        other_pct = to_float(row.get("decode_other_pct")) or 0.0
        latency_ms = to_float(row.get("guidellm_latency_mean_ms"))
        tokens_per_decode_iter = (
            completed_output_tokens / decode_events
            if decode_events
            else None
        )
        latency_s = latency_ms / 1000.0 if latency_ms is not None else None
        concise_rows.append(
            {
                "model": row.get("algo"),
                "batch_size": row.get("batch_size"),
                "num_spec_tokens": row.get("num_spec_tokens"),
                "decode_verify_pct": rounded(
                    (to_float(row.get("decode_verify_pct")) or 0.0) * 100.0,
                    2,
                ),
                "decode_draft_pct": rounded(
                    (to_float(row.get("decode_draft_pct")) or 0.0) * 100.0,
                    2,
                ),
                "decode_others_pct": rounded(
                    (accept_pct + other_pct) * 100.0, 2
                ),
                "tokens_per_decode_iter": rounded(tokens_per_decode_iter, 2),
                "e2e_latency_mean_s": rounded(latency_s, 3),
            }
        )
    return concise_rows


def build_verify_detail_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rounded(value: float | None, digits: int) -> float | None:
        return round(value, digits) if value is not None else None

    detail_rows = []
    for row in rows:
        if not (to_float(row.get("verify_detail_events")) or 0.0):
            continue
        detail_rows.append(
            {
                "model": row.get("algo"),
                "batch_size": row.get("batch_size"),
                "num_spec_tokens": row.get("num_spec_tokens"),
                "verify_qkv_proj_pct": rounded(
                    (
                        to_float(row.get("decode_verify_qkv_proj_pct_of_verify"))
                        or 0.0
                    )
                    * 100.0,
                    2,
                ),
                "verify_attention_pct": rounded(
                    (
                        to_float(row.get("decode_verify_attention_pct_of_verify"))
                        or 0.0
                    )
                    * 100.0,
                    2,
                ),
                "verify_ffn_pct": rounded(
                    (to_float(row.get("decode_verify_ffn_pct_of_verify")) or 0.0)
                    * 100.0,
                    2,
                ),
                "verify_model_other_pct": rounded(
                    (
                        to_float(
                            row.get("decode_verify_model_other_pct_of_verify")
                        )
                        or 0.0
                    )
                    * 100.0,
                    2,
                ),
                "verify_qkv_proj_ms_per_iter": rounded(
                    to_float(row.get("decode_verify_qkv_proj_ms_per_iter")), 3
                ),
                "verify_attention_ms_per_iter": rounded(
                    to_float(row.get("decode_verify_attention_ms_per_iter")), 3
                ),
                "verify_ffn_ms_per_iter": rounded(
                    to_float(row.get("decode_verify_ffn_ms_per_iter")), 3
                ),
                "verify_model_other_ms_per_iter": rounded(
                    to_float(row.get("decode_verify_model_other_ms_per_iter")), 3
                ),
                "verify_total_ms_per_iter": rounded(
                    to_float(row.get("decode_verify_ms_per_iter")), 3
                ),
            }
        )
    return detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def excel_col(idx: int) -> str:
    col = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        col = chr(ord("A") + rem) + col
    return col


def cell_xml(row_idx: int, col_idx: int, value: Any) -> str:
    ref = f"{excel_col(col_idx)}{row_idx + 1}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return f'<c r="{ref}"/>'
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def sheet_xml(rows: list[dict[str, Any]], fields: list[str]) -> str:
    table = [dict(zip(fields, fields, strict=True))]
    table.extend(rows)
    row_xml = []
    for row_idx, row in enumerate(table):
        cells = "".join(cell_xml(row_idx, col_idx, row.get(field)) for col_idx, field in enumerate(fields))
        row_xml.append(f'<row r="{row_idx + 1}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(row_xml)
        + "</sheetData></worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(
            f'<sheet name="{escape(name)}" sheetId="{idx}" '
            f'r:id="rId{idx}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(sheets)
        + "</sheets></workbook>"
    )


def workbook_rels(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def content_types(sheet_count: int) -> str:
    sheets = []
    for idx in range(1, sheet_count + 1):
        sheets.append(
            '<Override PartName="/xl/worksheets/sheet'
            f'{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(sheets)
        + "</Types>"
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[dict[str, Any]], list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(sheets)))
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>',
        )
        zf.writestr("xl/workbook.xml", workbook_xml([sheet[0] for sheet in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels(len(sheets)))
        for idx, (_name, rows, fields) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows, fields))


def draw_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    usable = [
        row
        for row in rows
        if to_float(row.get("decode_total_ms_per_iter"))
        and row.get("algo") is not None
        and row.get("batch_size") is not None
        and row.get("num_spec_tokens") is not None
    ]
    width = 1500
    height = 650
    if not usable:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220">'
            '<text x="30" y="60" font-family="Arial" font-size="18">No breakdown data found.</text>'
            "</svg>",
            encoding="utf-8",
        )
        return

    ks = sorted({int(row["num_spec_tokens"]) for row in usable})
    batches = sorted({int(row["batch_size"]) for row in usable})
    algos = [algo for algo in ["eagle3", "peagle"] if any(row["algo"] == algo for row in usable)]
    colors = {
        "verify": "#4c78a8",
        "draft": "#f58518",
        "accept": "#54a24b",
        "other": "#9d755d",
    }
    comp_keys = [
        ("verify", "decode_verify_ms_per_iter"),
        ("draft", "decode_draft_ms_per_iter"),
        ("accept", "decode_accept_reject_ms_per_iter"),
        ("other", "decode_other_ms_per_iter"),
    ]
    panel_gap = 40
    margin_left = 70
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    panel_width = (width - margin_left - margin_right - panel_gap * (len(ks) - 1)) / len(ks)
    plot_height = height - margin_top - margin_bottom
    max_total = max(to_float(row.get("decode_total_ms_per_iter")) or 0.0 for row in usable)
    max_total = max_total * 1.18 if max_total else 1.0

    def row_for(k: int, batch: int, algo: str) -> dict[str, Any] | None:
        for row in usable:
            if (
                int(row["num_spec_tokens"]) == k
                and int(row["batch_size"]) == batch
                and row["algo"] == algo
            ):
                return row
        return None

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="34" font-family="Arial" font-size="22" font-weight="700">Motivation Breakdown: decode iteration time</text>',
        '<text x="30" y="56" font-family="Arial" font-size="12" fill="#555">Component labels are placed directly on the bars; values are per vLLM decode iteration.</text>',
    ]

    for panel_idx, k in enumerate(ks):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = margin_top
        parts.append(
            f'<text x="{x0}" y="{y0 - 18}" font-family="Arial" font-size="16" font-weight="700">NUM_SPEC_TOKENS={k}</text>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{y0 + plot_height}" x2="{x0 + panel_width}" y2="{y0 + plot_height}" stroke="#222"/>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}" stroke="#222"/>'
        )
        for tick_idx in range(5):
            value = max_total * tick_idx / 4
            y = y0 + plot_height - value / max_total * plot_height
            parts.append(
                f'<line x1="{x0 - 4}" y1="{y}" x2="{x0 + panel_width}" y2="{y}" stroke="#e6e6e6"/>'
            )
            if panel_idx == 0:
                parts.append(
                    f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#555">{value:.0f}</text>'
                )

        group_width = panel_width / max(len(batches), 1)
        bar_width = min(24, group_width / (len(algos) + 1.8))
        for batch_idx, batch in enumerate(batches):
            group_center = x0 + group_width * batch_idx + group_width / 2
            parts.append(
                f'<text x="{group_center}" y="{y0 + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="11">bs={batch}</text>'
            )
            for algo_idx, algo in enumerate(algos):
                row = row_for(k, batch, algo)
                if row is None:
                    continue
                bar_x = group_center - (len(algos) * bar_width + (len(algos) - 1) * 5) / 2 + algo_idx * (bar_width + 5)
                y_cursor = y0 + plot_height
                total = to_float(row.get("decode_total_ms_per_iter")) or 0.0
                for label, key in comp_keys:
                    value = to_float(row.get(key)) or 0.0
                    h = value / max_total * plot_height
                    y_cursor -= h
                    parts.append(
                        f'<rect x="{bar_x}" y="{y_cursor}" width="{bar_width}" height="{h}" fill="{colors[label]}"/>'
                    )
                    if h >= 18:
                        parts.append(
                            f'<text x="{bar_x + bar_width / 2}" y="{y_cursor + h / 2 + 3}" text-anchor="middle" font-family="Arial" font-size="9" fill="white">{label}</text>'
                        )
                parts.append(
                    f'<text x="{bar_x + bar_width / 2}" y="{max(y0 + 10, y_cursor - 5)}" text-anchor="middle" font-family="Arial" font-size="10" fill="#222">{algo}</text>'
                )
                parts.append(
                    f'<text x="{bar_x + bar_width / 2}" y="{y0 + plot_height + 39}" text-anchor="middle" font-family="Arial" font-size="10" fill="#444">{total:.0f}</text>'
                )

    parts.append(
        f'<text x="{margin_left - 48}" y="{margin_top + plot_height / 2}" transform="rotate(-90 {margin_left - 48} {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="12">ms / decode iteration</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_verify_detail_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    usable = [
        row
        for row in rows
        if to_float(row.get("verify_detail_events"))
        and to_float(row.get("decode_verify_ms_per_iter"))
        and row.get("algo") is not None
        and row.get("batch_size") is not None
        and row.get("num_spec_tokens") is not None
    ]
    width = 1500
    height = 650
    if not usable:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220">'
            '<text x="30" y="60" font-family="Arial" font-size="18">No verify-detail data found.</text>'
            "</svg>",
            encoding="utf-8",
        )
        return

    ks = sorted({int(row["num_spec_tokens"]) for row in usable})
    batches = sorted({int(row["batch_size"]) for row in usable})
    algos = [algo for algo in ["eagle3", "peagle"] if any(row["algo"] == algo for row in usable)]
    colors = {
        "qkv": "#4c78a8",
        "attn": "#f58518",
        "ffn": "#54a24b",
        "other": "#9d755d",
    }
    comp_keys = [
        ("qkv", "decode_verify_qkv_proj_ms_per_iter"),
        ("attn", "decode_verify_attention_ms_per_iter"),
        ("ffn", "decode_verify_ffn_ms_per_iter"),
        ("other", "decode_verify_model_other_ms_per_iter"),
    ]
    panel_gap = 40
    margin_left = 70
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    panel_width = (width - margin_left - margin_right - panel_gap * (len(ks) - 1)) / len(ks)
    plot_height = height - margin_top - margin_bottom
    max_total = max(to_float(row.get("decode_verify_ms_per_iter")) or 0.0 for row in usable)
    max_total = max_total * 1.18 if max_total else 1.0

    def row_for(k: int, batch: int, algo: str) -> dict[str, Any] | None:
        for row in usable:
            if (
                int(row["num_spec_tokens"]) == k
                and int(row["batch_size"]) == batch
                and row["algo"] == algo
            ):
                return row
        return None

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="34" font-family="Arial" font-size="22" font-weight="700">Verify Breakdown: QKV / Attention / FFN</text>',
        '<text x="30" y="56" font-family="Arial" font-size="12" fill="#555">Values are per vLLM decode iteration; labels are drawn directly on bars.</text>',
    ]

    for panel_idx, k in enumerate(ks):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        y0 = margin_top
        parts.append(
            f'<text x="{x0}" y="{y0 - 18}" font-family="Arial" font-size="16" font-weight="700">NUM_SPEC_TOKENS={k}</text>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{y0 + plot_height}" x2="{x0 + panel_width}" y2="{y0 + plot_height}" stroke="#222"/>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}" stroke="#222"/>'
        )
        for tick_idx in range(5):
            value = max_total * tick_idx / 4
            y = y0 + plot_height - value / max_total * plot_height
            parts.append(
                f'<line x1="{x0 - 4}" y1="{y}" x2="{x0 + panel_width}" y2="{y}" stroke="#e6e6e6"/>'
            )
            if panel_idx == 0:
                parts.append(
                    f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#555">{value:.0f}</text>'
                )

        group_width = panel_width / max(len(batches), 1)
        bar_width = min(24, group_width / (len(algos) + 1.8))
        for batch_idx, batch in enumerate(batches):
            group_center = x0 + group_width * batch_idx + group_width / 2
            parts.append(
                f'<text x="{group_center}" y="{y0 + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="11">bs={batch}</text>'
            )
            for algo_idx, algo in enumerate(algos):
                row = row_for(k, batch, algo)
                if row is None:
                    continue
                bar_x = group_center - (len(algos) * bar_width + (len(algos) - 1) * 5) / 2 + algo_idx * (bar_width + 5)
                y_cursor = y0 + plot_height
                total = to_float(row.get("decode_verify_ms_per_iter")) or 0.0
                for label, key in comp_keys:
                    value = to_float(row.get(key)) or 0.0
                    h = value / max_total * plot_height
                    y_cursor -= h
                    parts.append(
                        f'<rect x="{bar_x}" y="{y_cursor}" width="{bar_width}" height="{h}" fill="{colors[label]}"/>'
                    )
                    if h >= 18:
                        parts.append(
                            f'<text x="{bar_x + bar_width / 2}" y="{y_cursor + h / 2 + 3}" text-anchor="middle" font-family="Arial" font-size="9" fill="white">{label}</text>'
                        )
                parts.append(
                    f'<text x="{bar_x + bar_width / 2}" y="{max(y0 + 10, y_cursor - 5)}" text-anchor="middle" font-family="Arial" font-size="10" fill="#222">{algo}</text>'
                )
                parts.append(
                    f'<text x="{bar_x + bar_width / 2}" y="{y0 + plot_height + 39}" text-anchor="middle" font-family="Arial" font-size="10" fill="#444">{total:.0f}</text>'
                )

    parts.append(
        f'<text x="{margin_left - 48}" y="{margin_top + plot_height / 2}" transform="rotate(-90 {margin_left - 48} {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="12">verify ms / decode iteration</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize motivation breakdown run directories."
    )
    parser.add_argument("output_root", type=Path, help="motivation_breakdown output root")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    run_root = output_root / "runs"
    run_dirs = sorted(path for path in run_root.glob("*") if path.is_dir())
    rows: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row, events, acceptance = summarize_run(run_dir)
        rows.append(row)
        raw_events.extend(events)
        acceptance_rows.extend(acceptance)

    rows.sort(
        key=lambda row: (
            str(row.get("algo")),
            int(row.get("batch_size") or 0),
            int(row.get("num_spec_tokens") or 0),
        )
    )
    raw_events.sort(
        key=lambda row: (
            str(row.get("algo")),
            int(row.get("batch_size") or 0),
            int(row.get("num_spec_tokens") or 0),
            float(row.get("ts") or 0.0),
        )
    )
    concise_rows = build_concise_rows(rows)
    verify_detail_rows = build_verify_detail_rows(rows)

    write_csv(output_root / "concise_summary.csv", concise_rows, CONCISE_FIELDS)
    write_csv(
        output_root / "verify_detail_summary.csv",
        verify_detail_rows,
        VERIFY_DETAIL_FIELDS,
    )
    write_csv(output_root / "summary.csv", rows, SUMMARY_FIELDS)
    write_csv(output_root / "raw_events.csv", raw_events, RAW_EVENT_FIELDS)
    write_csv(
        output_root / "acceptance.csv",
        acceptance_rows,
        ["algo", "batch_size", "num_spec_tokens", "position", "acceptance_rate", "run_dir"],
    )
    write_xlsx(
        output_root / "motivation_breakdown.xlsx",
        [
            ("concise_summary", concise_rows, CONCISE_FIELDS),
            ("verify_detail", verify_detail_rows, VERIFY_DETAIL_FIELDS),
            ("summary", rows, SUMMARY_FIELDS),
            ("raw_events", raw_events, RAW_EVENT_FIELDS),
            (
                "acceptance",
                acceptance_rows,
                [
                    "algo",
                    "batch_size",
                    "num_spec_tokens",
                    "position",
                    "acceptance_rate",
                    "run_dir",
                ],
            ),
        ],
    )
    draw_svg(output_root / "motivation_breakdown.svg", rows)
    draw_verify_detail_svg(output_root / "motivation_verify_breakdown.svg", rows)
    print(f"[INFO] Wrote concise summary: {output_root / 'concise_summary.csv'}")
    print(
        f"[INFO] Wrote verify detail:   {output_root / 'verify_detail_summary.csv'}"
    )
    print(f"[INFO] Wrote summary: {output_root / 'summary.csv'}")
    print(f"[INFO] Wrote Excel:   {output_root / 'motivation_breakdown.xlsx'}")
    print(f"[INFO] Wrote figure:  {output_root / 'motivation_breakdown.svg'}")
    print(
        f"[INFO] Wrote verify figure: {output_root / 'motivation_verify_breakdown.svg'}"
    )


if __name__ == "__main__":
    main()
