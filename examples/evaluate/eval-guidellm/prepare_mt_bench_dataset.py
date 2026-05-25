#!/usr/bin/env python3
"""Download and convert MT-Bench questions for completion-style tracing."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_prompt(turns: list[Any]) -> str:
    parts: list[str] = []
    for idx, turn in enumerate(turns, start=1):
        parts.append(f"User turn {idx}:\n{str(turn)}")
    return "\n\n".join(parts) + "\n\nAssistant:"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-out", type=Path, default=Path("data/mt_bench_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/mt_bench.jsonl"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force or not args.raw_out.exists():
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(args.url, timeout=120) as response:
            args.raw_out.write_bytes(response.read())

    converted: list[dict[str, Any]] = []
    for row in read_jsonl(args.raw_out):
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise SystemExit(f"MT-Bench row has no turns: {row.get('question_id')}")
        prompt = build_prompt(turns)
        converted.append(
            {
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "dataset_label": "mtbench",
                "turns": turns,
                "reference": row.get("reference"),
                "prompt": prompt,
            }
        )

    if len(converted) != 80:
        raise SystemExit(f"Expected 80 MT-Bench rows, got {len(converted)}")
    write_jsonl(args.output, converted)
    print(f"[INFO] Wrote {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()
