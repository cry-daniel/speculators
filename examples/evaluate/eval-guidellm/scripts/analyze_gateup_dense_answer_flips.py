#!/usr/bin/env python3
"""Compare per-example answer flips for original D1 and gate/up-dense D1."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
DEFAULT_BASELINE_ROOT = (
    EVAL_ROOT
    / "results_final"
    / "lm_eval_official_cot_fewshot_dense_d0_d8_qwen_llama_256_bs64_seed42_20260724_0030"
    / "llama3_1_8b"
)
DEFAULT_GATEUP_ROOT = (
    EVAL_ROOT
    / "results_final"
    / "lm_eval_official_llama_d1_gateup_dense_256_bs64_seed42_20260724"
    / "llama3_1_8b"
    / "d1"
)
DEFAULT_TOKENIZER = SPECULATORS_ROOT.parent / "models" / "llama-3.1-8b-instruct"
TASKS = ("gsm8k", "minerva", "bbh", "mmlu_pro")


def response_text(row: dict[str, Any]) -> str:
    responses = row.get("resps") or [[""]]
    return str(responses[0][0])


def prediction_text(row: dict[str, Any]) -> str:
    response = row.get("filtered_resps")
    if isinstance(response, list) and response:
        return str(response[0])
    return str(response or "")


def invalid_prediction(row: dict[str, Any]) -> bool:
    return prediction_text(row).strip().lower() in {
        "",
        "none",
        "invalid",
        "[invalid]",
    }


def target_option_text(row: dict[str, Any]) -> str:
    doc = row.get("doc") or {}
    target = str(row.get("target") or "").strip()
    options = doc.get("options")
    if isinstance(options, list) and len(target) == 1 and target.isalpha():
        index = ord(target.upper()) - ord("A")
        if 0 <= index < len(options):
            return str(options[index]).strip()
    prompt = str(doc.get("input") or doc.get("question") or "")
    if re.fullmatch(r"\(?[A-Z]\)?", target):
        letter = target.strip("()")
        match = re.search(
            rf"\({letter}\)\s*(.*?)(?=\s*\([A-Z]\)|$)",
            prompt,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
    return ""


def question_text(row: dict[str, Any]) -> str:
    doc = row.get("doc") or {}
    return str(
        doc.get("question")
        or doc.get("problem")
        or doc.get("input")
        or ""
    )


def load_task(root: Path, task: str) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    pattern = str(root / task / "lm_eval_output" / "**" / "samples_*.jsonl")
    for raw_path in glob.glob(pattern, recursive=True):
        path = Path(raw_path)
        leaf = path.name.split("_2026-", 1)[0].removeprefix("samples_")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if task == "gsm8k" and row.get("filter") != "flexible-extract":
                    continue
                metric = "math_verify" if task == "minerva" else "exact_match"
                row["correct"] = int(float(row[metric]))
                row["task_leaf"] = leaf
                output[(str(row["doc_hash"]), str(row["filter"]))] = row
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> None:
    roots = {
        "old": args.baseline_root / "d1",
        "gateup_dense": args.gateup_root,
        "dense": args.baseline_root / "dense_eagle3",
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    output_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"tasks": {}, "overall_invalid_predictions": {}}

    loaded: dict[str, dict[str, dict[tuple[str, str], dict[str, Any]]]] = {}
    for method, root in roots.items():
        loaded[method] = {task: load_task(root, task) for task in TASKS}
        summary["overall_invalid_predictions"][method] = {
            task: sum(
                invalid_prediction(row)
                for row in loaded[method][task].values()
            )
            for task in ("bbh", "mmlu_pro")
        }

    for task in TASKS:
        old_rows = loaded["old"][task]
        gate_rows = loaded["gateup_dense"][task]
        dense_rows = loaded["dense"][task]
        if old_rows.keys() != gate_rows.keys() or old_rows.keys() != dense_rows.keys():
            raise RuntimeError(f"{task}: fixed examples are not aligned")
        counts: Counter[str] = Counter()
        for key, old in old_rows.items():
            gate = gate_rows[key]
            dense = dense_rows[key]
            if int(old["correct"]) == int(gate["correct"]):
                continue
            direction = (
                "wrong_to_correct"
                if int(old["correct"]) == 0
                else "correct_to_wrong"
            )
            gate_raw = response_text(gate)
            option_text = target_option_text(gate)
            gate_invalid = invalid_prediction(gate)
            gate_tokens = len(
                tokenizer.encode(gate_raw, add_special_tokens=False)
            )
            counts[direction] += 1
            counts[f"{direction}_dense_correct"] += int(dense["correct"])
            counts[f"{direction}_dense_wrong"] += 1 - int(dense["correct"])
            if direction == "correct_to_wrong":
                counts["correct_to_wrong_invalid_extract"] += int(gate_invalid)
                counts["correct_to_wrong_max_tokens"] += int(
                    gate_tokens >= args.max_gen_toks - 1
                )
                counts["correct_to_wrong_invalid_but_target_text_present"] += int(
                    gate_invalid
                    and bool(option_text)
                    and option_text.lower() in gate_raw.lower()
                )
            output_rows.append(
                {
                    "task": task,
                    "task_leaf": old["task_leaf"],
                    "doc_hash": key[0],
                    "filter": key[1],
                    "direction": direction,
                    "target": old.get("target"),
                    "target_option_text": option_text,
                    "question": question_text(old),
                    "old_correct": old["correct"],
                    "gateup_dense_correct": gate["correct"],
                    "dense_correct": dense["correct"],
                    "old_prediction": prediction_text(old),
                    "gateup_dense_prediction": prediction_text(gate),
                    "dense_prediction": prediction_text(dense),
                    "gateup_dense_invalid_extract": gate_invalid,
                    "gateup_dense_response_tokens": gate_tokens,
                    "gateup_dense_hit_max_tokens": (
                        gate_tokens >= args.max_gen_toks - 1
                    ),
                    "gateup_dense_target_text_present": (
                        bool(option_text)
                        and option_text.lower() in gate_raw.lower()
                    ),
                    "old_response": response_text(old),
                    "gateup_dense_response": gate_raw,
                    "dense_response": response_text(dense),
                }
            )
        summary["tasks"][task] = dict(counts)

    total = Counter()
    for task_counts in summary["tasks"].values():
        total.update(task_counts)
    summary["total"] = dict(total)
    summary["num_aligned_examples"] = len(TASKS) * 256
    summary["num_flipped_examples"] = len(output_rows)
    summary["max_gen_toks"] = args.max_gen_toks

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_rows.sort(
        key=lambda row: (
            str(row["task"]),
            str(row["direction"]),
            str(row["task_leaf"]),
            str(row["doc_hash"]),
        )
    )
    write_csv(args.output_root / "answer_flips.csv", output_rows)
    write_json(args.output_root / "answer_flip_summary.json", summary)
    print(args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument("--gateup-root", type=Path, default=DEFAULT_GATEUP_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-gen-toks", type=int, default=256)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_GATEUP_ROOT.parents[1]
        / "answer_flip_analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
