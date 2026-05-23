#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, read_jsonl, register_experiment, write_json
from pams.correctness import verifier_decision_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--trace", type=Path, default=EXPERIMENTS / "05_mask_planner_offline" / "raw" / "test_with_priors.jsonl")
    args = parser.parse_args()
    rows = read_jsonl(args.trace) if args.trace.exists() else []
    exact_decisions = []
    approx_decisions = []
    for row in rows:
        dense = bool(row["accepted"])
        risk = float(row.get("risk_score", 0.0))
        prior = float(row.get("acceptance_prior", row.get("draft_probability", 0.5)))
        early = int(row.get("token_index", 0)) <= 1
        exact_decisions.append({"dense_accept": dense, "sparse_accept": dense, "dense_fallback": True})
        fallback = risk > 0.65 or (early and 0.35 <= prior <= 0.75)
        sparse_accept = dense if fallback else prior > 0.55
        approx_decisions.append({"dense_accept": dense, "sparse_accept": sparse_accept, "dense_fallback": fallback})
    exact_metrics = verifier_decision_metrics(exact_decisions)
    approx_metrics = verifier_decision_metrics(approx_decisions)
    result = {
        "integration": "C_sparse_prefilter_dense_fallback",
        "patched_vllm": False,
        "ran_live_vllm": False,
        "exact_fallback": exact_metrics,
        "approximate_fallback": approx_metrics,
        "outcome": "offline_prefilter_simulation_no_live_vllm",
    }
    exp = EXPERIMENTS / "09_vllm_integration_c_fallback_prefilter"
    write_json(exp / "parsed" / "integration_c_result.json", result)
    summary = [
        "# Integration C: Sparse Prefilter + Dense Fallback",
        "",
        "Two fallback policies were evaluated on the trace as an offline proxy. No live vLLM prefilter path ran in this turn.",
        "",
        f"- Exact fallback decision match: `{exact_metrics['decision_match_rate']:.4f}`",
        f"- Exact fallback false accept: `{exact_metrics['false_accept_rate']:.4f}`",
        f"- Approx fallback false accept: `{approx_metrics['false_accept_rate']:.4f}`",
        f"- Approx fallback dense fallback ratio: `{approx_metrics['dense_fallback_rate']:.4f}`",
    ]
    register_experiment(
        "09_vllm_integration_c_fallback_prefilter",
        config=vars(args),
        command="python scripts/run_vllm_integration_c.py --target-model Qwen/Qwen3-8B",
        status="attempted_offline_prefilter_only",
        summary="\n".join(summary),
        metadata_extra=result,
        model_name=args.target_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

