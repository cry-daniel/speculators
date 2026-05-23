from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .configs import write_jsonl


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    input_len: int | list[tuple[float, int]]
    output_len: int
    num_prompts: int
    max_model_len: int
    concurrency: list[int]


WORKLOADS: dict[str, WorkloadSpec] = {
    "short_mtbench_like": WorkloadSpec("short_mtbench_like", 512, 256, 128, 4096, [1, 2, 4, 8, 16]),
    "short_chat": WorkloadSpec("short_chat", 128, 128, 128, 2048, [1, 2, 4, 8, 16]),
    "medium_sharegpt_like": WorkloadSpec(
        "medium_sharegpt_like",
        [(0.30, 256), (0.30, 512), (0.25, 1024), (0.15, 2048)],
        256,
        128,
        4096,
        [1, 2, 4, 8, 16],
    ),
    "long_rag_4k": WorkloadSpec("long_rag_4k", 4096, 256, 64, 8192, [1, 2, 4, 8]),
    "long_rag_8k": WorkloadSpec("long_rag_8k", 8192, 256, 32, 16384, [1, 2, 4]),
    "long_output": WorkloadSpec("long_output", 512, 1024, 64, 4096, [1, 2, 4, 8]),
    "mixed_5090_safe": WorkloadSpec("mixed_5090_safe", 512, 256, 128, 8192, [1, 2, 4, 8, 16]),
}


def workload_specs(names: Iterable[str] | None = None) -> list[WorkloadSpec]:
    if names is None:
        return list(WORKLOADS.values())
    return [WORKLOADS[name] for name in names]


def choose_input_len(spec: WorkloadSpec, rng: random.Random, index: int) -> int:
    if isinstance(spec.input_len, int):
        return spec.input_len
    threshold = rng.random()
    cumulative = 0.0
    for prob, value in spec.input_len:
        cumulative += prob
        if threshold <= cumulative:
            return value
    return spec.input_len[-1][1]


def _token_budget_text(target_tokens: int, seed: int) -> str:
    # This is a tokenizer-independent approximation used for length-controlled
    # serving benchmarks. Real token counts are measured later when a tokenizer is available.
    rng = random.Random(seed)
    words = []
    vocab = [
        "reason", "compute", "verify", "context", "answer", "detail", "step",
        "evidence", "constraint", "memory", "system", "draft", "target",
        "block", "token", "latency",
    ]
    for _ in range(max(1, target_tokens * 3 // 4)):
        words.append(rng.choice(vocab))
    return " ".join(words)


def generate_workload(spec: WorkloadSpec, seed: int = 0) -> list[dict]:
    rng = random.Random(seed + sum(ord(c) for c in spec.name))
    rows = []
    if spec.name == "mixed_5090_safe":
        mix = [
            ("short_chat", 0.30),
            ("short_mtbench_like", 0.25),
            ("medium_sharegpt_like", 0.25),
            ("long_rag_4k", 0.10),
            ("long_output", 0.10),
        ]
        expanded: list[str] = []
        for name, frac in mix:
            expanded.extend([name] * int(round(spec.num_prompts * frac)))
        while len(expanded) < spec.num_prompts:
            expanded.append("short_chat")
        expanded = expanded[: spec.num_prompts]
        rng.shuffle(expanded)
        for idx, child_name in enumerate(expanded):
            child_spec = WORKLOADS[child_name]
            input_len = choose_input_len(child_spec, rng, idx)
            output_len = child_spec.output_len
            rows.append(make_prompt_row(spec.name, idx, input_len, output_len, child_name, seed))
        return rows

    for idx in range(spec.num_prompts):
        input_len = choose_input_len(spec, rng, idx)
        rows.append(make_prompt_row(spec.name, idx, input_len, spec.output_len, spec.name, seed))
    return rows


def make_prompt_row(workload: str, idx: int, input_len: int, output_len: int, source: str, seed: int) -> dict:
    context = _token_budget_text(input_len, seed * 100_000 + idx)
    return {
        "prompt_id": f"{workload}_{idx:04d}",
        "workload": workload,
        "source_workload": source,
        "input_len_target": input_len,
        "output_len_target": output_len,
        "max_tokens": output_len,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the provided context and answer with concise reasoning.\n\n"
                    f"Context:\n{context}\n\nQuestion: What is the most relevant verification detail?"
                ),
            }
        ],
    }


def split_rows(rows: list[dict], seed: int = 0) -> dict[str, list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    cal_end = max(1, int(n * 0.40))
    val_end = max(cal_end + 1, int(n * 0.70))
    return {
        "calibration": shuffled[:cal_end],
        "validation": shuffled[cal_end:val_end],
        "test": shuffled[val_end:],
    }


def specs_as_dict() -> dict[str, dict]:
    return {name: asdict(spec) for name, spec in WORKLOADS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for spec in workload_specs(args.workloads):
        rows = generate_workload(spec, args.seed)
        write_jsonl(args.output_dir / f"{spec.name}.jsonl", rows)
        for split, split_rows_ in split_rows(rows, args.seed).items():
            write_jsonl(args.output_dir / f"{spec.name}_{split}.jsonl", split_rows_)


if __name__ == "__main__":
    main()

