#!/usr/bin/env python3
"""Prepare Databricks Dolly 15k as JSONL prompts split by category."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "dolly"
)
DATASET_ID = "databricks/databricks-dolly-15k"


def build_prompt(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()
    if context:
        return f"Instruction:\n{instruction}\n\nContext:\n{context}\n\nAnswer:"
    return f"Instruction:\n{instruction}\n\nAnswer:"


def normalize_row(row: dict[str, Any], source_id: str) -> dict[str, Any]:
    instruction = str(row.get("instruction") or "").strip()
    context = str(row.get("context") or "").strip()
    response = str(row.get("response") or "").strip()
    category = str(row.get("category") or "unknown").strip() or "unknown"
    return {
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": category,
        "prompt": build_prompt(instruction, context),
        "reference": response,
        "source_id": source_id,
    }


def load_dolly() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "The `datasets` package is required to prepare Dolly. "
            "Install it in the `spec` conda env or use an existing HF cache."
        ) from exc
    try:
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load {DATASET_ID}. Check network access or the local "
            f"Hugging Face cache. Original error: {exc!r}"
        ) from exc
    rows = [normalize_row(dict(row), f"dolly-{idx:06d}") for idx, row in enumerate(dataset)]
    if not rows:
        raise SystemExit(f"{DATASET_ID} loaded zero rows; refusing to write empty files")
    return rows


def parse_categories(value: str) -> set[str] | None:
    categories = {item.strip() for item in value.split(",") if item.strip()}
    return categories or None


def filter_and_sample(
    rows: list[dict[str, Any]],
    *,
    categories: set[str] | None,
    limit_per_category: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if categories is not None and row["category"] not in categories:
            continue
        grouped[row["category"]].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        items = list(grouped[category])
        if limit_per_category is not None and limit_per_category > 0:
            rng.shuffle(items)
            items = items[:limit_per_category]
            items.sort(key=lambda item: item["source_id"])
        selected.extend(items)
    if not selected:
        raise SystemExit("No Dolly rows matched the requested categories/limits")
    return selected


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stats_for(category: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg_chars(field: str) -> float:
        return sum(len(str(row.get(field) or "")) for row in rows) / len(rows)

    return {
        "category": category,
        "num_examples": len(rows),
        "avg_instruction_chars": avg_chars("instruction"),
        "avg_context_chars": avg_chars("context"),
        "avg_response_chars": avg_chars("response"),
        "avg_prompt_chars": avg_chars("prompt"),
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, *, force: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise SystemExit(f"{output_dir} already exists and is not empty; pass --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    by_category_dir = output_dir / "by_category"
    by_category_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    all_path = output_dir / "dolly_all.jsonl"
    write_jsonl(all_path, rows)
    category_paths: dict[str, str] = {}
    stats_rows: list[dict[str, Any]] = []
    for category in sorted(grouped):
        category_path = by_category_dir / f"{category}.jsonl"
        write_jsonl(category_path, grouped[category])
        category_paths[category] = str(category_path.resolve())
        stats_rows.append(stats_for(category, grouped[category]))

    stats_path = output_dir / "dolly_category_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "category",
            "num_examples",
            "avg_instruction_chars",
            "avg_context_chars",
            "avg_response_chars",
            "avg_prompt_chars",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    manifest = {
        "dataset": DATASET_ID,
        "output_dir": str(output_dir.resolve()),
        "dolly_all": str(all_path.resolve()),
        "by_category_dir": str(by_category_dir.resolve()),
        "category_stats": str(stats_path.resolve()),
        "categories": {
            row["category"]: int(row["num_examples"]) for row in stats_rows
        },
        "category_paths": category_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--categories", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_dolly()
    selected = filter_and_sample(
        rows,
        categories=parse_categories(args.categories),
        limit_per_category=args.limit_per_category or None,
        seed=args.seed,
    )
    manifest = write_outputs(selected, args.output_dir, force=args.force)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
