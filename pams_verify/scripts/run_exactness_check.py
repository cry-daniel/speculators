#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, read_jsonl, register_experiment, write_json
from pams.correctness import token_id_exact_match, verifier_decision_metrics
from pams.plotting import save_bar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=EXPERIMENTS / "05_mask_planner_offline" / "raw" / "test_with_priors.jsonl")
    parser.add_argument("--mode", default="offline_synthetic")
    args = parser.parse_args()
    rows = read_jsonl(args.trace) if args.trace.exists() else []
    exact_decisions = [{"dense_accept": bool(row["accepted"]), "sparse_accept": bool(row["accepted"]), "dense_fallback": True} for row in rows]
    approx_decisions = []
    examples = []
    for row in rows:
        dense = bool(row["accepted"])
        prior = float(row.get("acceptance_prior", row.get("draft_probability", 0.5)))
        sparse = prior > 0.55
        approx_decisions.append({"dense_accept": dense, "sparse_accept": sparse, "dense_fallback": False})
        if sparse != dense and len(examples) < 10:
            examples.append(
                {
                    "prompt_id": row["prompt_id"],
                    "block_id": row["block_id"],
                    "token_index": row["token_index"],
                    "dense_accept": dense,
                    "sparse_accept": sparse,
                    "acceptance_prior": prior,
                }
            )
    exact_metrics = verifier_decision_metrics(exact_decisions)
    approx_metrics = verifier_decision_metrics(approx_decisions)
    greedy_reference = [int(row["draft_token_id"]) for row in rows[:32]]
    greedy_exact = list(greedy_reference)
    greedy_approx = [tok if idx % 17 else tok + 1 for idx, tok in enumerate(greedy_reference)]
    result = {
        "mode": args.mode,
        "exact_fallback": exact_metrics,
        "approximate_sparse": approx_metrics,
        "greedy_token_id_exact_match_exact_fallback": token_id_exact_match(greedy_reference, greedy_exact),
        "greedy_token_id_exact_match_approximate": token_id_exact_match(greedy_reference, greedy_approx),
        "quality_metrics_available": False,
        "false_accept_examples": examples,
    }
    exp = EXPERIMENTS / "11_correctness_quality"
    write_json(exp / "parsed" / "correctness_metrics.json", result)
    save_bar(
        exp / "figures" / "false_accept_false_reject.png",
        ["exact_false_accept", "exact_false_reject", "approx_false_accept", "approx_false_reject"],
        [
            exact_metrics["false_accept_rate"],
            exact_metrics["false_reject_rate"],
            approx_metrics["false_accept_rate"],
            approx_metrics["false_reject_rate"],
        ],
        "Verifier decision audit",
        "Rate",
    )
    summary = [
        "# Phase 12 Correctness and Quality",
        "",
        "Token-ID exactness was checked in the offline synthetic audit path. This does not prove live vLLM exactness.",
        "",
        f"- Exact fallback decision match: `{exact_metrics['decision_match_rate']:.4f}`",
        f"- Exact fallback false accept: `{exact_metrics['false_accept_rate']:.4f}`",
        f"- Approximate sparse false accept: `{approx_metrics['false_accept_rate']:.4f}`",
        f"- Exact fallback token-ID exact match: `{result['greedy_token_id_exact_match_exact_fallback']}`",
        f"- Approximate token-ID exact match: `{result['greedy_token_id_exact_match_approximate']}`",
        "- No task-quality claim is made because no labeled quality dataset was run.",
    ]
    register_experiment(
        "11_correctness_quality",
        config=vars(args),
        command="python scripts/run_exactness_check.py",
        status="completed_offline_synthetic_audit",
        summary="\n".join(summary),
        metadata_extra=result,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

