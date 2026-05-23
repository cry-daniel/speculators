#!/usr/bin/env python3
"""Create a deterministic repeated-request dataset from the first N JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-n", default=5, type=int)
    parser.add_argument("--repeat", default=16, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == args.first_n:
                break

    if len(rows) != args.first_n:
        raise ValueError(f"Expected {args.first_n} rows, found {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for repeat_idx in range(args.repeat):
            for row in rows:
                item = dict(row)
                item["question_id"] = f"{row.get('question_id')}_r{repeat_idx:02d}"
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
