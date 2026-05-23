#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, failure_record, register_experiment, write_json


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=False)
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workloads", nargs="+", default=["short_chat", "short_mtbench_like"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--execute-vllm", action="store_true")
    args = parser.parse_args()

    import torch

    cuda_available = bool(torch.cuda.is_available())
    vllm_bin = shutil.which("vllm")
    help_code, help_out = run(["vllm", "bench", "serve", "--help"]) if vllm_bin else (127, "vllm command not found")
    baselines = [
        "dense_no_spec",
        "vllm_ngram_4",
        "vllm_ngram_8",
        "vllm_suffix_8",
        "vllm_suffix_16",
        "model_draft_qwen3_0p6b_4",
        "model_draft_qwen3_0p6b_8",
        "eagle3",
        "p_eagle",
    ]
    runs = []
    failures = []
    for workload in args.workloads:
        for conc in args.concurrency:
            for method in baselines:
                status = "registered_not_executed"
                reason = "execute_vllm flag not set"
                if not cuda_available:
                    status = "blocked_cuda_unavailable_in_process"
                    reason = "torch.cuda.is_available() is false in this process"
                elif help_code != 0:
                    status = "blocked_vllm_bench_unavailable"
                    reason = help_out.strip()[:500]
                elif args.execute_vllm:
                    status = "not_implemented_live_runner"
                    reason = "live vLLM baseline command construction is intentionally gated for manual GPU execution"
                if status.startswith("blocked") or status.startswith("not_implemented"):
                    failures.append(failure_record(status, reason, {"workload": workload, "concurrency": conc, "method": method}))
                runs.append(
                    {
                        "workload": workload,
                        "concurrency": conc,
                        "method": method,
                        "status": status,
                        "reason": reason,
                    }
                )
    exp = EXPERIMENTS / "01_dense_baselines"
    write_json(exp / "raw" / "baselines_registered.json", {"runs": runs})
    write_json(exp / "parsed" / "baseline_failures.json", {"failures": failures})
    if not cuda_available:
        block_reason = "Live vLLM benchmark execution is blocked because CUDA is not visible to torch in this process."
    elif help_code != 0:
        block_reason = "Live vLLM benchmark execution is blocked because the `vllm bench serve` CLI is unavailable in this environment."
    else:
        block_reason = "Baseline commands were registered; pass --execute-vllm for a manually supervised live run."
    summary = [
        "# Phase 3 Dense and Standard Speculative Baselines",
        "",
        f"Baseline configurations were registered before execution. {block_reason}",
        "",
        f"- vLLM command found: `{bool(vllm_bin)}`",
        f"- `vllm bench serve --help` exit code: `{help_code}`",
        f"- torch CUDA available: `{cuda_available}`",
        f"- registered runs: `{len(runs)}`",
        f"- blocked/unsupported records: `{len(failures)}`",
    ]
    register_experiment(
        "01_dense_baselines",
        config=vars(args) | {"baselines": baselines},
        command="python scripts/run_dense_baselines.py --target-model Qwen/Qwen3-8B --workloads short_chat short_mtbench_like --max-model-len 4096 --concurrency 1 2 4 --num-prompts 32",
        status="blocked_cuda_unavailable_in_process" if not cuda_available else "registered_not_executed",
        summary="\n".join(summary),
        stdout=help_out,
        stderr="" if cuda_available else "torch.cuda.is_available() returned false",
        metadata_extra={"runs_registered": len(runs), "failures": failures, "vllm_help_exit_code": help_code},
        model_name=args.target_model,
        max_model_len=args.max_model_len,
    )
    print(json.dumps({"runs": len(runs), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
