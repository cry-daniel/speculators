#!/usr/bin/env python3
"""Create padded math_reasoning variants with distractor context."""

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

from speculators.speclink.math_eval import load_dataset, write_jsonl  # noqa: E402


def load_tokenizer(model: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] tokenizer unavailable for {model}: {exc}", file=sys.stderr)
        return None


def token_len(tokenizer: Any | None, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text.split()))
    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def make_prompt(records: list[Any], index: int, target_tokens: int, tokenizer: Any | None) -> str:
    target = records[index]
    if target_tokens <= 0:
        return target.prompt

    parts = [
        "The following are unrelated worked examples. They are distractor context.",
        "Solve only the final problem after the marker FINAL PROBLEM.",
        "",
    ]
    cursor = index + 1
    while token_len(tokenizer, "\n".join(parts) + target.prompt) < target_tokens:
        distractor = records[cursor % len(records)]
        if distractor.id != target.id:
            parts.append(f"Unrelated example {len(parts)}:")
            parts.append(distractor.prompt)
            parts.append(f"Reference answer: {distractor.reference_raw}")
            parts.append("")
        cursor += 1
        if cursor - index > len(records) * 8:
            break
    parts.append("FINAL PROBLEM:")
    parts.append(target.prompt)
    return "\n".join(parts)


def summarize(lengths: list[int]) -> dict[str, Any]:
    return {
        "min": min(lengths) if lengths else None,
        "mean": statistics.mean(lengths) if lengths else None,
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p95": percentile(lengths, 95),
        "max": max(lengths) if lengths else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--targets",
        default="0,2048,4096",
        help="Comma-separated target prompt token lengths. Use 0 for original.",
    )
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    tokenizer = load_tokenizer(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for target in [int(item) for item in args.targets.split(",") if item.strip()]:
        suffix = "orig" if target <= 0 else f"pad{target // 1000}k"
        rows = []
        lengths: list[int] = []
        for index, record in enumerate(records):
            prompt = make_prompt(records, index, target, tokenizer)
            length = token_len(tokenizer, prompt)
            lengths.append(length)
            rows.append(
                {
                    "id": record.id,
                    "question_id": record.id,
                    "prompt": prompt,
                    "reference": record.reference_raw,
                    "padding_target_tokens": target,
                    "prompt_tokens": length,
                    "source_dataset": str(args.dataset),
                }
            )
        path = out_dir / f"math_reasoning_{suffix}.jsonl"
        write_jsonl(path, rows)
        manifest.append(
            {
                "target_tokens": target,
                "path": str(path),
                "num_records": len(rows),
                "prompt_token_lengths": summarize(lengths),
            }
        )
        print(f"wrote {path}")

    manifest_path = out_dir / "padded_math_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
