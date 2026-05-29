#!/usr/bin/env python3
"""Merge live SpecLink-CV correctness gate summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize(rows: list[dict[str, Any]], sources: list[str]) -> dict[str, Any]:
    total = len(rows)
    matched = sum(1 for row in rows if is_true(row.get("matched")))
    by_model: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row.get("model", ""))
        bucket = by_model.setdefault(key, {"total": 0, "matched": 0})
        bucket["total"] += 1
        if is_true(row.get("matched")):
            bucket["matched"] += 1
    return {
        "sources": sources,
        "total_cases": total,
        "matched_cases": matched,
        "failed_cases": total - matched,
        "all_matched": total > 0 and matched == total,
        "by_model": by_model,
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Live Correctness Gate Summary",
        "",
        f"- total_cases: `{summary['total_cases']}`",
        f"- matched_cases: `{summary['matched_cases']}`",
        f"- failed_cases: `{summary['failed_cases']}`",
        f"- all_matched: `{summary['all_matched']}`",
        "",
        "Sources:",
        "",
    ]
    lines.extend(f"- `{source}`" for source in summary["sources"])
    lines.extend(
        [
            "",
            "| model | dataset | K | batch | prompts | max tokens | mode | greedy eps | matched | matched count | total | first mismatch token |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {model} | {dataset} | {K} | {batch_size} | {num_prompts} | "
            "{max_tokens} | {mode} | {greedy_eps} | {matched} | "
            "{matched_count} | {total_count} | {first_mismatch_token_index} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Note: `greedy_eps` is a diagnostic near-tie guard for",
            "`VLLM_BATCH_INVARIANT=1`; passing rows here are correctness-gate",
            "evidence for the listed bounded prompts and token budget, not a",
            "throughput claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in args.summary:
        sources.append(str(path))
        source_rows = read_rows(path)
        for row in source_rows:
            row = dict(row)
            row["source_summary"] = str(path)
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, sources)
    write_csv(args.output_dir / "combined_summary.csv", rows)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "combined_summary.md", rows, summary)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "total_cases": summary["total_cases"],
                "failed_cases": summary["failed_cases"],
            }
        )
    )
    return 1 if summary["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
