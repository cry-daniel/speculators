#!/usr/bin/env python3
"""Probe math_reasoning.jsonl without assuming fixed field names."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from speculators.speclink.math_eval import (  # noqa: E402
    PROMPT_FIELDS,
    REFERENCE_FIELDS,
    build_math_prompt,
    first_present,
    load_dataset,
    read_jsonl,
    write_jsonl,
)


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def load_tokenizer(model: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] tokenizer unavailable for {model}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_jsonl(args.dataset)
    normalized = load_dataset(args.dataset)
    tokenizer = load_tokenizer(args.model)

    token_lengths: list[int] = []
    if tokenizer is not None:
        for record in normalized:
            token_lengths.append(
                len(tokenizer.encode(build_math_prompt(record.prompt), add_special_tokens=False))
            )

    examples = []
    for index, record in enumerate(raw[:5]):
        examples.append(
            {
                "index": index,
                "fields": sorted(record.keys()),
                "prompt_field": (first_present(record, PROMPT_FIELDS) or ["", ""])[0],
                "reference_field": (first_present(record, REFERENCE_FIELDS) or ["", ""])[0],
                "sample": record,
            }
        )

    profile = {
        "dataset": str(args.dataset),
        "num_records": len(raw),
        "prompt_field_candidates": list(PROMPT_FIELDS),
        "reference_field_candidates": list(REFERENCE_FIELDS),
        "observed_fields": sorted({key for row in raw for key in row.keys()}),
        "tokenizer_model": args.model,
        "tokenizer_available": tokenizer is not None,
        "prompt_token_lengths": {
            "min": min(token_lengths) if token_lengths else None,
            "mean": statistics.mean(token_lengths) if token_lengths else None,
            "p50": percentile(token_lengths, 50),
            "p90": percentile(token_lengths, 90),
            "p95": percentile(token_lengths, 95),
            "max": max(token_lengths) if token_lengths else None,
        },
    }
    (out_dir / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(out_dir / "dataset_examples.jsonl", examples)
    print(f"wrote {out_dir / 'dataset_profile.json'}")
    print(f"wrote {out_dir / 'dataset_examples.jsonl'}")


if __name__ == "__main__":
    main()
