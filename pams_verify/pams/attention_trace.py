from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from .configs import write_jsonl
from .workloads import WORKLOADS, generate_workload, split_rows


def _scores_for_blocks(rng: random.Random, center: int, total_blocks: int, width: int, count: int) -> dict[int, float]:
    candidates: dict[int, float] = {}
    target = min(count, total_blocks)
    offsets = [0]
    for delta in range(1, total_blocks + 1):
        offsets.extend([-delta, delta])
        if len(offsets) >= target * 3:
            break
    for offset in offsets:
        block = center + offset
        if block < 0 or block >= total_blocks:
            continue
        distance = abs(block - center)
        jitter = rng.random() * 0.05
        candidates[block] = 1.0 / (1.0 + distance / max(1, width)) + jitter
        if len(candidates) >= target:
            break
    return candidates


def synthetic_trace_for_prompts(
    prompts: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
    block_sizes: list[int],
    draft_len: int = 4,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed + sum(ord(c) for c in split))
    for prompt_idx, prompt in enumerate(prompts):
        input_len = int(prompt["input_len_target"])
        rounds = max(2, min(12, int(prompt["output_len_target"]) // max(1, draft_len * 8)))
        workload_bias = {
            "short_chat": 0.35,
            "short_mtbench_like": 0.25,
            "medium_sharegpt_like": 0.12,
            "long_rag_4k": -0.05,
            "long_rag_8k": -0.12,
            "long_output": 0.05,
        }.get(str(prompt.get("source_workload", prompt.get("workload"))), 0.0)
        for round_idx in range(rounds):
            block_id = f"{prompt['prompt_id']}:{round_idx}"
            first_rejection_seen = False
            prev_accept_ma = 0.7
            for token_index in range(draft_len):
                dense_accept_prob = max(
                    0.03,
                    min(0.96, 0.82 - 0.12 * token_index + workload_bias + rng.gauss(0.0, 0.08)),
                )
                accepted = rng.random() < dense_accept_prob and not first_rejection_seen
                if not accepted:
                    first_rejection_seen = True
                draft_prob = max(0.01, min(0.99, dense_accept_prob + rng.gauss(0.0, 0.10)))
                entropy = max(0.05, 1.8 - 1.2 * draft_prob + rng.random() * 0.2)
                margin = (draft_prob - 0.5) * 5.0 + rng.gauss(0.0, 0.45)
                for block_size in block_sizes:
                    total_blocks = max(1, (input_len + round_idx * draft_len + token_index + block_size - 1) // block_size)
                    base_center = int(total_blocks * (0.60 + 0.08 * rng.random()))
                    token_shift = token_index * max(1, total_blocks // 16)
                    draft_center = max(0, min(total_blocks - 1, base_center + token_shift + rng.randint(-2, 2)))
                    target_center = max(0, min(total_blocks - 1, base_center + int(token_index * total_blocks / 24) + rng.randint(-2, 2)))
                    draft_scores = _scores_for_blocks(rng, draft_center, total_blocks, max(1, total_blocks // 12), 24)
                    target_scores = _scores_for_blocks(rng, target_center, total_blocks, max(1, total_blocks // 14), 12)
                    top_blocks = [b for b, _ in sorted(target_scores.items(), key=lambda item: (-item[1], item[0]))[:8]]
                    rows.append(
                        {
                            "trace_source": "synthetic_fallback_no_online_target_masks",
                            "split": split,
                            "prompt_id": prompt["prompt_id"],
                            "workload": prompt["workload"],
                            "block_id": f"{block_id}:bs{block_size}",
                            "round_index": round_idx,
                            "token_position": input_len + round_idx * draft_len + token_index,
                            "token_index": token_index,
                            "draft_token_id": 1000 + ((prompt_idx * 97 + round_idx * 13 + token_index) % 50000),
                            "draft_probability": draft_prob,
                            "draft_logit_margin": margin,
                            "draft_entropy": entropy,
                            "draft_topk_alternatives": [
                                [1000 + ((prompt_idx * 97 + round_idx * 13 + token_index + k) % 50000), max(0.0, draft_prob - 0.05 * k)]
                                for k in range(1, 6)
                            ],
                            "draft_hidden_available": False,
                            "draft_attention_available": True,
                            "candidate_blocks": draft_scores,
                            "target_dense_top_blocks": top_blocks,
                            "target_dense_logprob_for_draft": -max(0.01, 1.0 - dense_accept_prob),
                            "target_dense_logits_margin": margin + rng.gauss(0.0, 0.35),
                            "accepted": int(accepted),
                            "reached": int(not first_rejection_seen or accepted),
                            "first_rejection_position": token_index if first_rejection_seen else None,
                            "dense_target_verifier_decision": int(accepted),
                            "proposed_draft_length": draft_len,
                            "block_size": block_size,
                            "sequence_length": input_len,
                            "previous_round_acceptance_moving_average": prev_accept_ma,
                        }
                    )
                    prev_accept_ma = 0.8 * prev_accept_ma + 0.2 * float(accepted)
    return rows


def collect_synthetic_traces(
    output_dir: Path,
    *,
    workload_names: list[str],
    seed: int,
    block_sizes: list[int],
    draft_len: int,
) -> dict[str, Path]:
    merged: dict[str, list[dict[str, Any]]] = {"calibration": [], "validation": [], "test": []}
    for workload_name in workload_names:
        rows = generate_workload(WORKLOADS[workload_name], seed)
        splits = split_rows(rows, seed)
        for split_name, prompts in splits.items():
            merged[split_name].extend(
                synthetic_trace_for_prompts(prompts, split=split_name, seed=seed, block_sizes=block_sizes, draft_len=draft_len)
            )
    paths = {}
    for split_name, rows in merged.items():
        path = output_dir / f"trace_{split_name}.jsonl"
        write_jsonl(path, rows)
        paths[split_name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", default=["short_chat", "medium_sharegpt_like", "long_rag_4k"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--draft-len", type=int, default=4)
    args = parser.parse_args()
    collect_synthetic_traces(
        args.output_dir,
        workload_names=args.workloads,
        seed=args.seed,
        block_sizes=args.block_sizes,
        draft_len=args.draft_len,
    )


if __name__ == "__main__":
    main()
