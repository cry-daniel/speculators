#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, read_jsonl, register_experiment, write_json
from pams.vllm_hooks import inspect_vllm_for_pams_hooks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--trace", type=Path, default=EXPERIMENTS / "05_mask_planner_offline" / "raw" / "test_with_priors.jsonl")
    args = parser.parse_args()
    rows = read_jsonl(args.trace) if args.trace.exists() else []
    inspected = inspect_vllm_for_pams_hooks()
    total = len(rows)
    high_risk = sum(1 for row in rows if float(row.get("risk_score", 0.0)) > 0.65)
    low_reach_suffix = sum(1 for row in rows if int(row.get("token_index", 0)) >= 2 and float(row.get("prefix_reach_probability", 1.0)) < 0.35)
    result = {
        "integration": "A_scheduler_hook",
        "exact_mode": True,
        "patched_vllm": False,
        "compiled": False,
        "ran_live_vllm": False,
        "trace_rows": total,
        "high_risk_tokens": high_risk,
        "low_reach_suffix_tokens": low_reach_suffix,
        "vllm_inspection": inspected,
        "outcome": "offline_policy_registered_no_live_vllm_hook",
    }
    exp = EXPERIMENTS / "07_vllm_integration_a_scheduler_hook"
    write_json(exp / "parsed" / "integration_a_result.json", result)
    summary = [
        "# Integration A: Scheduler / Verification-Planner Hook",
        "",
        "Attempted a low-risk exact policy using PAMS risk scores to shorten low-reach suffixes and bypass high-risk speculation.",
        "",
        "Result: no live vLLM scheduler hook was applied in this run. The installed vLLM package was inspected and no PAMS feature flag was present.",
        "",
        f"- Trace rows analyzed: `{total}`",
        f"- High-risk tokens: `{high_risk}`",
        f"- Low-reach suffix tokens: `{low_reach_suffix}`",
        f"- vLLM imported: `{inspected['vllm_imported']}`",
        f"- vLLM version: `{inspected['version']}`",
    ]
    register_experiment(
        "07_vllm_integration_a_scheduler_hook",
        config=vars(args),
        command="python scripts/run_vllm_integration_a.py --target-model Qwen/Qwen3-8B",
        status="attempted_no_live_patch",
        summary="\n".join(summary),
        metadata_extra=result,
        model_name=args.target_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

