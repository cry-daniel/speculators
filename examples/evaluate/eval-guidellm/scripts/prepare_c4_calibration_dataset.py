#!/usr/bin/env python3
"""Prepare a fixed C4 prompt sample for structured 2:4 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "c4_calibration"
    / "c4_calibration_512_seed42.jsonl"
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="allenai/c4")
    parser.add_argument("--config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--num-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise SystemExit("--num-examples must be positive")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")

    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("The `datasets` package is required in the spec env") from exc

    stream = load_dataset(
        args.dataset_id,
        args.config,
        split=args.split,
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(stream):
        text = str(item.get(args.text_field, "")).strip()
        if len(text) < args.min_chars:
            continue
        rows.append(
            {
                "prompt": text,
                "reference": "",
                "source_id": f"c4-{args.seed}-{idx:09d}",
                "dataset": args.dataset_id,
                "config": args.config,
                "split": args.split,
            }
        )
        if len(rows) >= args.num_examples:
            break

    if len(rows) != args.num_examples:
        raise SystemExit(f"Only collected {len(rows)} rows; expected {args.num_examples}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_chars = [len(row["prompt"]) for row in rows]
    prompt_words = [len(row["prompt"].split()) for row in rows]
    stats = {
        "count": len(rows),
        "avg_prompt_chars": mean(prompt_chars),
        "avg_prompt_words": mean(prompt_words),
        "min_prompt_chars": min(prompt_chars),
        "max_prompt_chars": max(prompt_chars),
    }
    manifest = {
        "dataset_id": args.dataset_id,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "num_examples": args.num_examples,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "min_chars": args.min_chars,
        "output": str(args.output.resolve()),
        "stats": stats,
    }
    write_json(args.output.parent / "manifest.json", manifest)
    write_json(args.output.parent / "stats.json", stats)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
