#!/usr/bin/env python3
"""Freeze exactly 256 official lm-eval 0.4.12 examples per benchmark.

The grouped Minerva Math, BBH, and MMLU-Pro tasks interpret ``--limit`` per
leaf task, so ``--limit 256`` would evaluate thousands of examples.  This
script deterministically stratifies 256 examples across each group's official
leaf tasks and writes the exact ``--samples`` maps consumed by lm-eval.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
from typing import Any

from datasets import load_dataset


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parent.parent
DEFAULT_OUTPUT = EVAL_ROOT / "data" / "lm_eval_official_256_seed42"
MINERVA_LEAVES = {
    "algebra": "minerva_math_algebra",
    "counting_and_probability": "minerva_math_counting_and_prob",
    "geometry": "minerva_math_geometry",
    "intermediate_algebra": "minerva_math_intermediate_algebra",
    "number_theory": "minerva_math_num_theory",
    "prealgebra": "minerva_math_prealgebra",
    "precalculus": "minerva_math_precalc",
}
BBH_CATEGORIES = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)
MMLU_PRO_CATEGORIES = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocation(labels: list[str], total: int) -> dict[str, int]:
    quotient, remainder = divmod(total, len(labels))
    return {
        label: quotient + int(index < remainder)
        for index, label in enumerate(sorted(labels))
    }


def sampled_indices(population: int, count: int, seed: int) -> list[int]:
    if population < count:
        raise ValueError(f"cannot select {count} from {population}")
    return sorted(random.Random(seed).sample(range(population), count))


def prepare_gsm8k(count: int, seed: int) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = sampled_indices(len(dataset), count, seed)
    details = [
        {
            "leaf_task": "gsm8k_cot",
            "sample_index": index,
            "source_id": f"gsm8k:test:{index}",
        }
        for index in indices
    ]
    return {"gsm8k_cot": indices}, details


def prepare_minerva(
    count: int, seed: int
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    counts = allocation(list(MINERVA_LEAVES), count)
    samples: dict[str, list[int]] = {}
    details: list[dict[str, Any]] = []
    for category_index, category in enumerate(sorted(MINERVA_LEAVES)):
        leaf = MINERVA_LEAVES[category]
        dataset = load_dataset(
            "EleutherAI/hendrycks_math", category, split="test"
        )
        indices = sampled_indices(
            len(dataset), counts[category], seed * 1009 + category_index
        )
        samples[leaf] = indices
        details.extend(
            {
                "category": category,
                "leaf_task": leaf,
                "sample_index": index,
                "source_id": f"minerva_math:{category}:{index}",
            }
            for index in indices
        )
    return samples, details


def prepare_bbh(count: int, seed: int) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    counts = allocation(list(BBH_CATEGORIES), count)
    samples: dict[str, list[int]] = {}
    details: list[dict[str, Any]] = []
    for category_index, category in enumerate(sorted(BBH_CATEGORIES)):
        dataset = load_dataset("SaylorTwift/bbh", category, split="test")
        indices = sampled_indices(
            len(dataset), counts[category], seed * 2017 + category_index
        )
        leaf = f"bbh_cot_fewshot_{category}"
        samples[leaf] = indices
        details.extend(
            {
                "category": category,
                "leaf_task": leaf,
                "sample_index": index,
                "source_id": f"bbh:{category}:{index}",
            }
            for index in indices
        )
    return samples, details


def prepare_mmlu_pro(
    count: int, seed: int
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default", split="test")
    by_category: dict[str, list[tuple[int, str]]] = {
        category: [] for category in MMLU_PRO_CATEGORIES
    }
    for global_index, row in enumerate(dataset):
        category = str(row["category"])
        if category not in by_category:
            raise ValueError(f"unexpected MMLU-Pro category: {category}")
        source_id = str(row.get("question_id", global_index))
        by_category[category].append((global_index, source_id))

    counts = allocation(list(MMLU_PRO_CATEGORIES), count)
    samples: dict[str, list[int]] = {}
    details: list[dict[str, Any]] = []
    for category_index, category in enumerate(sorted(MMLU_PRO_CATEGORIES)):
        population = by_category[category]
        indices = sampled_indices(
            len(population), counts[category], seed * 3001 + category_index
        )
        leaf = f"mmlu_pro_{category.replace(' ', '_')}"
        samples[leaf] = indices
        details.extend(
            {
                "category": category,
                "global_dataset_index": population[index][0],
                "leaf_task": leaf,
                "sample_index": index,
                "source_id": f"mmlu_pro:{population[index][1]}",
            }
            for index in indices
        )
    return samples, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if importlib.metadata.version("lm-eval") != "0.4.12":
        raise RuntimeError("this protocol requires lm-eval 0.4.12")
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    builders = {
        "gsm8k_cot": prepare_gsm8k,
        "minerva_math": prepare_minerva,
        "bbh_cot_fewshot": prepare_bbh,
        "mmlu_pro": prepare_mmlu_pro,
    }
    manifest: dict[str, Any] = {
        "lm_eval_version": "0.4.12",
        "seed": args.seed,
        "num_examples_per_benchmark": args.num_examples,
        "selection": "category-stratified random sample; exact leaf indices frozen",
        "benchmarks": {},
    }
    for benchmark, builder in builders.items():
        samples_path = output / f"{benchmark}_samples.json"
        detail_path = output / f"{benchmark}_selection.json"
        if samples_path.exists() and detail_path.exists() and not args.force:
            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            details = json.loads(detail_path.read_text(encoding="utf-8"))
        else:
            samples, details = builder(args.num_examples, args.seed)
            write_json(samples_path, samples)
            write_json(detail_path, details)
        selected = sum(len(indices) for indices in samples.values())
        if selected != args.num_examples or len(details) != args.num_examples:
            raise RuntimeError(
                f"{benchmark} selected {selected}/{len(details)}, expected "
                f"{args.num_examples}"
            )
        identities = {
            (str(row["leaf_task"]), int(row["sample_index"])) for row in details
        }
        if len(identities) != args.num_examples:
            raise RuntimeError(f"{benchmark} contains duplicate leaf indices")
        manifest["benchmarks"][benchmark] = {
            "samples_path": str(samples_path.relative_to(EVAL_ROOT)),
            "samples_sha256": sha256(samples_path),
            "selection_path": str(detail_path.relative_to(EVAL_ROOT)),
            "selection_sha256": sha256(detail_path),
            "rows": selected,
            "leaf_tasks": len(samples),
            "category_counts": dict(
                sorted(Counter(row.get("category", "main") for row in details).items())
            ),
        }
    write_json(output / "manifest.json", manifest)
    print(output)


if __name__ == "__main__":
    main()
