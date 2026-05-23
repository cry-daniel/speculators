#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment, write_json
from pams.memory import estimate_memory
from pams.plotting import save_bar


def maybe_import_guidellm_status() -> dict:
    status_path = REPO_ROOT / "examples" / "evaluate" / "eval-guidellm" / "results" / "math_num_spec_tokens_sweep_20260522_173539" / "status.tsv"
    if not status_path.exists():
        return {"available": False, "path": str(status_path), "rows": []}
    rows = []
    for line in status_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        rows.append(parts)
    return {"available": True, "path": str(status_path), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--workloads", nargs="+", default=["short_chat", "short_mtbench_like", "medium_sharegpt_like", "long_rag_4k", "long_output", "mixed_5090_safe"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--respect-memory-estimator", default="true")
    args = parser.parse_args()
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    memory = estimate_memory(args.target_model, max_num_seqs=max(args.concurrency))
    guidellm = maybe_import_guidellm_status()
    methods = [
        "dense_no_spec",
        "vllm_ngram_4",
        "vllm_ngram_8",
        "model_draft_fixed_4",
        "model_draft_fixed_8",
        "independent_sparse_verifier",
        "shared_fixed_residual",
        "pams",
        "pams_fallback_exact",
        "pams_fallback_approximate",
    ]
    matrix = []
    for workload in args.workloads:
        for conc in args.concurrency:
            for method in methods:
                if method.startswith("pams") or method in {"independent_sparse_verifier", "shared_fixed_residual"}:
                    status = "blocked_no_patched_vllm_sparse_verifier"
                elif not cuda_available:
                    status = "blocked_cuda_unavailable_in_process"
                else:
                    status = "registered_not_executed"
                matrix.append({"workload": workload, "concurrency": conc, "method": method, "status": status})
    exp = EXPERIMENTS / "10_end2end"
    write_json(exp / "raw" / "imported_guidellm_status.json", guidellm)
    write_json(
        exp / "parsed" / "end2end_matrix.json",
        {
            "pams_end2end_available": False,
            "cuda_available_in_process": cuda_available,
            "vllm_command": shutil.which("vllm"),
            "memory_estimate": memory,
            "matrix": matrix,
        },
    )
    save_bar(
        exp / "figures" / "end2end_itl.png",
        ["dense", "standard_spec", "pams"],
        [0.0, 0.0, 0.0],
        "End-to-end ITL unavailable",
        "Mean ITL improvement",
    )
    save_bar(
        exp / "figures" / "end2end_throughput.png",
        ["dense", "standard_spec", "pams"],
        [0.0, 0.0, 0.0],
        "End-to-end throughput unavailable",
        "Output tok/s improvement",
    )
    summary = [
        "# Phase 11 End-to-End Matrix",
        "",
        "The end-to-end PAMS matrix is registered but not satisfied. Integration B did not produce a patched vLLM sparse verifier path, so PAMS methods are blocked.",
        "",
        f"- CUDA available to torch in this process: `{cuda_available}`",
        f"- Imported prior GuideLLM status file available: `{guidellm['available']}`",
        f"- Matrix entries: `{len(matrix)}`",
        "- Final GO criteria cannot be met without a live patched-vLLM PAMS result.",
    ]
    register_experiment(
        "10_end2end",
        config=vars(args) | {"methods": methods},
        command="python scripts/run_end2end_matrix.py --target-model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B --workloads short_chat short_mtbench_like medium_sharegpt_like long_rag_4k long_output mixed_5090_safe --concurrency 1 2 4 8 --respect-memory-estimator true",
        status="blocked_no_patched_vllm_sparse_verifier",
        summary="\n".join(summary),
        metadata_extra={"pams_end2end_available": False, "matrix_entries": len(matrix), "cuda_available_in_process": cuda_available},
        model_name=args.target_model,
        max_model_len=memory["recommended_max_model_len"],
    )
    register_experiment(
        "13_failures_oom",
        config={"source": "run_end2end_matrix", "oom_records": []},
        command="python scripts/run_end2end_matrix.py --target-model Qwen/Qwen3-8B ...",
        status="completed_no_oom_observed",
        summary=(
            "# Phase 13 Failure / OOM Log\n\n"
            "No live PAMS vLLM run was executed, and no OOM occurred during the implemented offline/reference stages. "
            "Future live vLLM OOMs should be appended under this folder with exact command, stderr, and degraded retry settings."
        ),
        metadata_extra={"oom_records": [], "degrade_order": memory["oom_degrade_order"]},
        model_name=args.target_model,
        max_model_len=memory["recommended_max_model_len"],
    )
    print(json.dumps({"pams_end2end_available": False, "matrix_entries": len(matrix)}, indent=2))


if __name__ == "__main__":
    main()
