#!/usr/bin/env python3
"""Measure dense, old indexed, and fused residual-complement prefill paths."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import run_llama_gsm8k_eagle3_residual_d1 as gsm
from residual_24_feasibility import load_gsm8k
from run_structured_24_spec_quality import start_vllm_server, stop_process


METHODS = (
    "eagle3_dense",
    "residual_complement_prefill_indexed",
    "residual_complement_prefill_fused",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    alpha = position - lower
    return ordered[lower] * (1 - alpha) + ordered[upper] * alpha


def run_batch(
    *,
    port: int,
    model: str,
    prompts: list[str],
    concurrency: int,
    run_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                gsm.post_completion,
                port=port,
                model=model,
                prompt=prompt,
                max_tokens=1,
                min_tokens=1,
                ignore_eos=True,
                request_id=f"prefill-{run_id}-{index:04d}",
                timeout_s=timeout_s,
            )
            for index, prompt in enumerate(prompts)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    errors = [row["error"] for row in results if row["error"]]
    if errors:
        raise RuntimeError(f"prefill batch failed: {errors[:3]}")
    completion_tokens = sum(int(row["completion_tokens"]) for row in results)
    if completion_tokens != len(prompts):
        raise RuntimeError(
            f"expected one output token per prompt, got {completion_tokens}"
        )
    return {
        "elapsed_sec": elapsed,
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in results),
        "completion_tokens": completion_tokens,
    }


def method_env(args: argparse.Namespace, method: str, case_dir: Path) -> dict[str, str]:
    env = gsm.make_env(args, method, case_dir)
    if method == "residual_complement_prefill_indexed":
        env["SPECLINK_TOKEN_DENSE_PREFILL_FUSED"] = "0"
    elif method == "residual_complement_prefill_fused":
        env["SPECLINK_TOKEN_DENSE_PREFILL_FUSED"] = "1"
    if args.trace_breakdown:
        env.update(
            {
                "SPECLINK_BREAKDOWN": "1",
                "SPECLINK_BREAKDOWN_SYNC": "0",
                "SPECLINK_BREAKDOWN_ALGO": method,
                "SPECLINK_BREAKDOWN_OUT": str(
                    (case_dir / "breakdown_events.jsonl").resolve()
                ),
            }
        )
    return env


def run_method(
    args: argparse.Namespace,
    method: str,
    prompts: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_dir = args.output_root / method
    case_dir.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        process, port = start_vllm_server(
            args,
            base_model=str(args.model.resolve()),
            speculator_model=str(args.speculator.resolve()),
            case_dir=case_dir,
            env=method_env(args, method, case_dir),
        )
        run_batch(
            port=port,
            model=str(args.model.resolve()),
            prompts=prompts,
            concurrency=args.concurrency,
            run_id=f"{method}-warmup",
            timeout_s=args.request_timeout_s,
        )
        raw: list[dict[str, Any]] = []
        for trial in range(args.trials):
            record = run_batch(
                port=port,
                model=str(args.model.resolve()),
                prompts=prompts,
                concurrency=args.concurrency,
                run_id=f"{method}-t{trial:02d}",
                timeout_s=args.request_timeout_s,
            )
            record.update({"method": method, "trial": trial})
            raw.append(record)
            print(
                f"[{method}] trial {trial + 1}/{args.trials}: "
                f"{record['elapsed_sec']:.6f} s",
                flush=True,
            )
        samples = [float(row["elapsed_sec"]) for row in raw]
        summary = {
            "method": method,
            "trials": args.trials,
            "requests_per_trial": len(prompts),
            "concurrency": args.concurrency,
            "prompt_tokens_per_trial": raw[0]["prompt_tokens"],
            "completion_tokens_per_trial": raw[0]["completion_tokens"],
            "median_sec": statistics.median(samples),
            "p10_sec": percentile(samples, 0.1),
            "p90_sec": percentile(samples, 0.9),
            "samples_sec": samples,
        }
        return summary, raw
    finally:
        stop_process(process)
        time.sleep(args.server_shutdown_settle_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", type=Path, default=gsm.DEFAULT_MODEL)
    parser.add_argument("--speculator", type=Path, default=gsm.DEFAULT_SPECULATOR)
    parser.add_argument(
        "--calibration-cache-root",
        type=Path,
        default=gsm.DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    )
    parser.add_argument("--num-questions", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated subset of dense,indexed,fused method names.",
    )
    parser.add_argument("--trace-breakdown", action="store_true")
    parser.add_argument("--num-spec-tokens", type=int, default=7)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--port-base", type=int, default=8080)
    parser.add_argument("--health-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = gsm.EVAL_ROOT / f"results/prefill_residual_complement_{stamp}"
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = set(args.methods) - set(METHODS)
    if not args.methods or unknown:
        parser.error(f"invalid --methods selection: {sorted(unknown)}")

    # Fields shared with the main GSM8K runner and vLLM command builder.
    args.dense_eighths = 1
    args.disable_step_stats = True
    args.prefill_fused = True
    args.compile_graph = True
    args.enforce_eager = False
    args.enable_prefix_caching = False
    args.cudagraph_capture_sizes = list(
        range(8, args.max_num_seqs * 8 + 1, 8)
    )
    args.compilation_config = {
        "mode": "VLLM_COMPILE",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
        "max_cudagraph_capture_size": args.cudagraph_capture_sizes[-1],
        "cudagraph_num_of_warmups": 1,
    }
    return args


def main() -> None:
    args = parse_args()
    pack = load_gsm8k(args.num_questions, args.seed)
    if pack.error or len(pack.rows) != args.num_questions:
        raise SystemExit(f"failed to load GSM8K: {pack.error or len(pack.rows)}")
    prompts = [str(row["prompt"]) for row in pack.rows]
    summaries: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for method in args.methods:
        summary, method_raw = run_method(args, method, prompts)
        summaries.append(summary)
        raw.extend(method_raw)

    dense_row = next(
        (row for row in summaries if row["method"] == "eagle3_dense"), None
    )
    dense = float(dense_row["median_sec"]) if dense_row is not None else None
    for row in summaries:
        row["time_over_dense"] = (
            float(row["median_sec"]) / dense if dense is not None else None
        )
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [key for key in summaries[0] if key != "samples_sec"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output_root / "raw.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    print(args.output_root)


if __name__ == "__main__":
    main()
