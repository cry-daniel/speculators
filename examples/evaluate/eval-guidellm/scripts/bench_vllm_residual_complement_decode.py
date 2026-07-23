#!/usr/bin/env python3
"""Decode-only vLLM EAGLE3 benchmark for quota-routed residual complement.

One model load is reused across all batch sizes and routing quotas for a given
backend.  Token-dense workers run in-process V1 with one fixed CUDA Graph per
quota/scope worker, so the GPU-resident confidence route can be updated between
replays without recapturing the model.  Per-request draft length is seven,
making each verifier step one current + seven draft rows (M=8B).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt

import sparse24_benchmark_common as common


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parent.parent
REPO_ROOT = EVAL_ROOT.parents[2]
MODELS = ("qwen3_8b", "llama3_1_8b")
BATCH_SIZES = (64, 128, 256)
SCOPES = ("global", "per_request")
EIGHTHS = tuple(range(1, 9))
SPEC_COUNTER = re.compile(
    r"^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens))"
    r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"
)
QWEN_CACHE = Path(
    "/ACALAB/stu1/.cache/huggingface/hub/models--Qwen--Qwen3-8B"
)
QWEN_REVISION = (QWEN_CACHE / "refs/main").read_text(encoding="utf-8").strip()
MODEL_PATHS = {
    "qwen3_8b": str((QWEN_CACHE / "snapshots" / QWEN_REVISION).resolve()),
    "llama3_1_8b": str((REPO_ROOT.parent / "models/llama-3.1-8b-instruct").resolve()),
}
SPECULATOR_PATHS = {
    "qwen3_8b": str((REPO_ROOT.parent / "models/qwen3-8b-eagle3-speculator").resolve()),
    "llama3_1_8b": str((REPO_ROOT.parent / "models/llama-3.1-8b-eagle3-speculator").resolve()),
}
CALIBRATION_ROOT = (
    EVAL_ROOT / "data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512"
).resolve()


def parse_csv(value: str, allowed: Sequence[str], name: str) -> tuple[str, ...]:
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(f"invalid {name}: {unknown or value}")
    return selected


def parse_int_csv(value: str, allowed: Sequence[int], name: str) -> tuple[int, ...]:
    selected = tuple(int(part) for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(f"invalid {name}: {unknown or value}")
    return selected


def configure_worker(args: argparse.Namespace) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.worker_backend == "dense":
        os.environ["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
        os.environ["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
        return
    os.environ.update(
        {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": args.worker_model,
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(CALIBRATION_ROOT),
            "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_MODE": "high_confidence_dense",
            "SPECLINK_TOKEN_DENSE_BACKEND": "residual_complement_splitk2",
            "SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS": str(args.eighths[0]),
            "SPECLINK_TOKEN_DENSE_ROUTING_SCOPE": args.scopes[0],
            "SPECLINK_TOKEN_DENSE_EXPECTED_ROWS": str(max(args.batch_sizes) * 8),
            "SPECLINK_TOKEN_DENSE_STATS_INTERVAL": "1",
            "SPECLINK_TOKEN_DENSE_STATS_DETAIL": "0",
            "SPECLINK_TOKEN_DENSE_GRAPH_ROUTE": "1",
            "SPECLINK_TOKEN_DENSE_GRAPH_ROWS": ",".join(
                str(batch * 8) for batch in args.batch_sizes
            ),
        }
    )


def output_decode_interval(outputs: list[Any], elapsed: float) -> tuple[float, int]:
    token_count = sum(len(candidate.token_ids) for out in outputs for candidate in out.outputs)
    starts = []
    finishes = []
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        first = getattr(metrics, "first_token_time", None)
        if first is None:
            first = getattr(metrics, "first_token_ts", None)
        finish = getattr(metrics, "finished_time", None)
        if finish is None:
            finish = getattr(metrics, "last_token_ts", None)
        if first is not None and finish is not None:
            starts.append(float(first))
            finishes.append(float(finish))
    decode_elapsed = max(finishes) - min(starts) if starts and finishes else elapsed
    return max(decode_elapsed, 1e-9), token_count


def load_route_totals(path: Path) -> dict[str, int]:
    """Load the final cumulative route record written during formal trials."""

    if not path.exists():
        return {}
    final: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            final = json.loads(line)
    keys = (
        "steps",
        "total_scheduled_tokens",
        "total_draft_tokens",
        "dense_draft_tokens",
        "sparse_draft_tokens",
        "missing_score_tokens",
    )
    return {key: int(final.get(key, 0)) for key in keys}


def scrape_acceptance_counters() -> dict[str, float]:
    """Read in-process vLLM Prometheus counters for both backends."""

    import prometheus_client

    text = prometheus_client.generate_latest().decode("utf-8", errors="replace")
    counters: dict[str, float] = {}
    for line in text.splitlines():
        match = SPEC_COUNTER.match(line.strip())
        if match:
            key = match.group(1)
            counters[key] = counters.get(key, 0.0) + float(match.group(2))
    return counters


def counter_delta(
    before: dict[str, float], after: dict[str, float], suffix: str
) -> float:
    key = f"vllm:spec_decode_num_{suffix}"
    return float(after.get(key, 0.0) - before.get(key, 0.0))


def run_generation(llm: Any, prompts: list[Any], params: Any) -> dict[str, float]:
    before = scrape_acceptance_counters()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    after = scrape_acceptance_counters()
    decode_elapsed, output_tokens = output_decode_interval(outputs, elapsed)
    request_durations = []
    for out in outputs:
        metrics = getattr(out, "metrics", None)
        first = getattr(metrics, "first_token_time", None)
        if first is None:
            first = getattr(metrics, "first_token_ts", None)
        finish = getattr(metrics, "finished_time", None)
        if finish is None:
            finish = getattr(metrics, "last_token_ts", None)
        if first is not None and finish is not None:
            request_durations.append(max(0.0, float(finish) - float(first)))
    drafts = counter_delta(before, after, "drafts")
    drafted_tokens = counter_delta(before, after, "draft_tokens")
    accepted_tokens = counter_delta(before, after, "accepted_tokens")
    return {
        "wall_sec": elapsed,
        "decode_sec": decode_elapsed,
        "output_tokens": output_tokens,
        "wall_tokens_per_sec": output_tokens / elapsed,
        "decode_tokens_per_sec": output_tokens / decode_elapsed,
        "mean_request_decode_ms": 1000.0
        * (statistics.mean(request_durations) if request_durations else decode_elapsed),
        "spec_decode_drafts": drafts,
        "spec_decode_draft_tokens": drafted_tokens,
        "spec_decode_accepted_tokens": accepted_tokens,
        "mean_acceptance_length": (
            1.0 + accepted_tokens / drafts if drafts else float("nan")
        ),
        "draft_acceptance_rate": (
            accepted_tokens / drafted_tokens
            if drafted_tokens
            else float("nan")
        ),
    }


def run_worker(args: argparse.Namespace) -> None:
    configure_worker(args)
    from vllm import LLM, SamplingParams, TokensPrompt

    model = args.worker_model
    capture_sizes = sorted({batch * 8 for batch in args.batch_sizes})
    llm = LLM(
        model=MODEL_PATHS[model],
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.batch_sizes),
        max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=not args.collect_acceptance,
        enforce_eager=False,
        enable_prefix_caching=True,
        generation_config="vllm",
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": max(capture_sizes),
        },
        speculative_config={
            "model": SPECULATOR_PATHS[model],
            "num_speculative_tokens": 7,
            "method": "eagle3",
            "max_model_len": args.max_model_len,
        },
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        min_tokens=args.output_tokens,
        ignore_eos=True,
        detokenize=False,
        seed=args.seed,
    )
    warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=min(8, args.output_tokens),
        min_tokens=min(8, args.output_tokens),
        ignore_eos=True,
        detokenize=False,
        seed=args.seed,
    )
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    cases = [("baseline", 8)] if args.worker_backend == "dense" else [
        (scope, eighths) for scope in args.scopes for eighths in args.eighths
    ]
    for batch in args.batch_sizes:
        # Unique final IDs prevent cross-request prefix sharing.  An untimed
        # one-token run installs each exact 128-token prefix in vLLM's prefix
        # cache, so subsequent wall intervals measure decoding rather than
        # prompt prefill.
        prompts = [
            TokensPrompt(
                prompt_token_ids=[42] * (args.prompt_tokens - 1) + [1000 + index]
            )
            for index in range(batch)
        ]
        # vLLM does not retain the final, potentially extendable cache block of
        # a completed request.  Install one extra token so the exact formal
        # 128-token prompt ends at a completed block boundary and is wholly
        # reusable.  The timed requests below still have length 128.
        cache_prompts = [
            TokensPrompt(prompt_token_ids=list(prompt["prompt_token_ids"]) + [77])
            for prompt in prompts
        ]
        prefill_only = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
            detokenize=False,
            seed=args.seed,
        )
        llm.generate(cache_prompts, prefill_only, use_tqdm=False)
        for scope, eighths in cases:
            stats_file = args.worker_output.parent / (
                f"{model}__{args.worker_backend}__b{batch}__{scope}__d{eighths}.jsonl"
            )
            if stats_file.exists():
                stats_file.unlink()
            if args.worker_backend != "dense":
                os.environ["SPECLINK_TOKEN_DENSE_ROUTING_SCOPE"] = scope
                os.environ["SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS"] = str(eighths)
                if args.route_stats:
                    os.environ["SPECLINK_TOKEN_DENSE_STATS_PATH"] = str(stats_file)
                else:
                    os.environ.pop("SPECLINK_TOKEN_DENSE_STATS_PATH", None)
                from vllm.speclink_token_dense import reset_runtime_state

                reset_runtime_state()
            # Warm the exact K=7 verification shape after changing quota.
            llm.generate(prompts, warmup_params, use_tqdm=False)
            # Formal routing statistics must not include model initialization,
            # prefix installation, or the shape/quota warmup above.
            if args.worker_backend != "dense":
                reset_runtime_state()
                if stats_file.exists():
                    stats_file.unlink()
            samples = []
            for trial in range(args.trials):
                metrics = run_generation(llm, prompts, params)
                record = {
                    "model": model,
                    "backend": args.worker_backend,
                    "batch_size": batch,
                    "M": batch * 8,
                    "draft_tokens_per_request": 7,
                    "routing_scope": scope,
                    "dense_fraction": f"{eighths}/8",
                    "trial": trial,
                    **metrics,
                }
                samples.append(record)
                raw.append(record)
                print(
                    f"{model} {args.worker_backend} B={batch} {scope} D={eighths}/8 "
                    f"trial={trial}: {metrics['decode_tokens_per_sec']:.2f} tok/s",
                    flush=True,
                )
            decode_rates = [float(row["decode_tokens_per_sec"]) for row in samples]
            wall_rates = [float(row["wall_tokens_per_sec"]) for row in samples]
            request_ms = [float(row["mean_request_decode_ms"]) for row in samples]
            total_drafts = sum(float(row["spec_decode_drafts"]) for row in samples)
            total_draft_tokens = sum(
                float(row["spec_decode_draft_tokens"]) for row in samples
            )
            total_accepted_tokens = sum(
                float(row["spec_decode_accepted_tokens"]) for row in samples
            )
            route_totals = (
                load_route_totals(stats_file)
                if args.worker_backend != "dense"
                else {}
            )
            request_steps = max(
                0,
                route_totals.get("total_scheduled_tokens", 0)
                - route_totals.get("total_draft_tokens", 0),
            )
            formal_output_tokens = sum(int(row["output_tokens"]) for row in samples)
            rows.append(
                {
                    "model": model,
                    "backend": args.worker_backend,
                    "batch_size": batch,
                    "M": batch * 8,
                    "draft_tokens_per_request": 7,
                    "routing_scope": scope,
                    "dense_fraction": f"{eighths}/8",
                    "decode_tokens_per_sec_median": statistics.median(decode_rates),
                    "decode_tokens_per_sec_p10": common.percentile(decode_rates, 0.1),
                    "decode_tokens_per_sec_p90": common.percentile(decode_rates, 0.9),
                    "wall_tokens_per_sec_median": statistics.median(wall_rates),
                    "mean_request_decode_ms_median": statistics.median(request_ms),
                    "mean_acceptance_length": (
                        1.0 + total_accepted_tokens / total_drafts
                        if total_drafts
                        else None
                    ),
                    "draft_acceptance_rate": (
                        total_accepted_tokens / total_draft_tokens
                        if total_draft_tokens
                        else None
                    ),
                    "spec_decode_drafts": total_drafts,
                    "spec_decode_draft_tokens": total_draft_tokens,
                    "spec_decode_accepted_tokens": total_accepted_tokens,
                    **route_totals,
                    "mean_emitted_tokens_per_request_step": (
                        formal_output_tokens / request_steps if request_steps else None
                    ),
                    "stats_path": str(stats_file) if args.worker_backend != "dense" else "",
                }
            )
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps({"rows": rows, "raw": raw}, indent=2) + "\n",
        encoding="utf-8",
    )
    # LLM has no public shutdown method in this vendored 0.20 API; the worker
    # is intentionally one model/backend per process, so process exit releases
    # the in-process engine and NCCL state.
    del llm


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    worker_files = []
    for model in args.models:
        jobs = [("dense", "baseline", 8)] + [
            ("residual_complement", scope, eighths)
            for scope in args.scopes
            for eighths in args.eighths
        ]
        for backend, scope, eighths in jobs:
            worker = work / f"{model}__{backend}__{scope}__d{eighths}.json"
            if args.analyze_only:
                if not worker.exists():
                    raise FileNotFoundError(f"missing worker result: {worker}")
                worker_files.append(worker)
                continue
            common.require_idle_gpu(args.device_index)
            command = [
                sys.executable,
                str(SCRIPT),
                "--worker",
                "--worker-model", model,
                "--worker-backend", backend,
                "--worker-output", str(worker),
                "--batch-sizes", ",".join(map(str, args.batch_sizes)),
                "--scopes", scope if backend != "dense" else "global",
                "--eighths", str(eighths if backend != "dense" else 1),
                "--trials", str(args.trials),
                "--prompt-tokens", str(args.prompt_tokens),
                "--output-tokens", str(args.output_tokens),
                "--max-model-len", str(args.max_model_len),
                "--max-num-batched-tokens", str(args.max_num_batched_tokens),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--seed", str(args.seed),
            ]
            if args.route_stats:
                command.append("--route-stats")
            if args.collect_acceptance:
                command.append("--collect-acceptance")
            subprocess.run(command, cwd=EVAL_ROOT, check=True)
            worker_files.append(worker)
    rows = []
    raw = []
    for worker in worker_files:
        payload = json.loads(worker.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        raw.extend(payload["raw"])
    dense = {
        (row["model"], int(row["batch_size"])): row
        for row in rows
        if row["backend"] == "dense"
    }
    merged = []
    for row in rows:
        if row["backend"] == "dense":
            continue
        baseline = dense[(row["model"], int(row["batch_size"]))]
        item = dict(row)
        item["dense_decode_tokens_per_sec_median"] = baseline[
            "decode_tokens_per_sec_median"
        ]
        item["dense_wall_tokens_per_sec_median"] = baseline[
            "wall_tokens_per_sec_median"
        ]
        item["dense_mean_acceptance_length"] = baseline[
            "mean_acceptance_length"
        ]
        item["speedup_vs_dense"] = float(row["decode_tokens_per_sec_median"]) / float(
            baseline["decode_tokens_per_sec_median"]
        )
        item["wall_speedup_vs_dense"] = float(
            row["wall_tokens_per_sec_median"]
        ) / float(baseline["wall_tokens_per_sec_median"])
        merged.append(item)
    write_csv(output / "summary.csv", merged)
    write_csv(output / "backend_summary.csv", rows)
    write_csv(output / "raw.csv", raw)
    speedups = [float(row["speedup_vs_dense"]) for row in merged]
    wall_speedups = [float(row["wall_speedup_vs_dense"]) for row in merged]

    def grouped(field: str, value: Any, metric: str) -> float:
        return geometric_mean(
            [float(row[metric]) for row in merged if row[field] == value]
        )
    analysis = {
        "cases": len(merged),
        "decode_cases_ge_1_1": sum(value >= 1.1 for value in speedups),
        "wall_cases_ge_1_1": sum(value >= 1.1 for value in wall_speedups),
        "geomean_speedup": geometric_mean(speedups),
        "geomean_wall_speedup": geometric_mean(wall_speedups),
        "by_scope": {
            scope: geometric_mean(
                [float(row["speedup_vs_dense"]) for row in merged if row["routing_scope"] == scope]
            )
            for scope in args.scopes
        },
        "by_fraction": {
            fraction: {
                "decode": grouped("dense_fraction", fraction, "speedup_vs_dense"),
                "wall": grouped(
                    "dense_fraction", fraction, "wall_speedup_vs_dense"
                ),
            }
            for fraction in sorted({row["dense_fraction"] for row in merged})
        },
        "by_batch_size": {
            str(batch): {
                "decode": geometric_mean(
                    [
                        float(row["speedup_vs_dense"])
                        for row in merged
                        if int(row["batch_size"]) == batch
                    ]
                ),
                "wall": geometric_mean(
                    [
                        float(row["wall_speedup_vs_dense"])
                        for row in merged
                        if int(row["batch_size"]) == batch
                    ]
                ),
            }
            for batch in args.batch_sizes
        },
        "by_model": {
            model: {
                "decode": grouped("model", model, "speedup_vs_dense"),
                "wall": grouped("model", model, "wall_speedup_vs_dense"),
            }
            for model in args.models
        },
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "models": list(args.models),
        "batch_sizes": list(args.batch_sizes),
        "dense_fractions_eighths": list(args.eighths),
        "routing_scopes": list(args.scopes),
        "draft_tokens_per_request": 7,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens_per_request": args.output_tokens,
        "trials": args.trials,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "cudagraph_capture_rows": [batch * 8 for batch in args.batch_sizes],
        "decode_interval": "earliest first-token timestamp to latest completion",
        "wall_interval": "complete LLM.generate call including cache-tail prefill",
        "gpu_exclusivity": "idle check before every worker process",
        "worker_results": [str(path) for path in worker_files],
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(
        len(args.models),
        len(args.batch_sizes),
        figsize=(5 * len(args.batch_sizes), 4 * len(args.models)),
        squeeze=False,
    )
    for ri, model in enumerate(args.models):
        for ci, batch in enumerate(args.batch_sizes):
            axis = axes[ri][ci]
            for scope in args.scopes:
                marker = "o" if scope == "global" else "s"
                selected = sorted(
                    (
                        row for row in merged
                        if row["model"] == model
                        and int(row["batch_size"]) == batch
                        and row["routing_scope"] == scope
                    ),
                    key=lambda row: int(row["dense_fraction"].split("/")[0]),
                )
                axis.plot(
                    fractions := [
                        int(row["dense_fraction"].split("/")[0]) / 8
                        for row in selected
                    ],
                    values := [float(row["speedup_vs_dense"]) for row in selected],
                    marker=marker,
                    label=scope,
                )
                for x_value, y_value in zip(fractions, values, strict=True):
                    axis.annotate(
                        f"{y_value:.3f}x",
                        (x_value, y_value),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.axhline(1.1, color="gray", linestyle=":", linewidth=0.8)
            axis.set_ylim(0.98, 1.33)
            axis.set_title(f"{model}, B={batch}, M={batch * 8}")
            axis.set_xlabel("Dense-token fraction")
            axis.set_ylabel("Decode throughput speedup")
            axis.grid(alpha=0.25)
            if ri == 0 and ci == len(args.batch_sizes) - 1:
                axis.legend()
    figure.tight_layout()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    figure.savefig(figures / "vllm_decode_speedup.png", dpi=220)
    figure.savefig(figures / "vllm_decode_speedup.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(
        len(args.models),
        len(args.batch_sizes),
        figsize=(5 * len(args.batch_sizes), 4 * len(args.models)),
        squeeze=False,
    )
    for ri, model in enumerate(args.models):
        for ci, batch in enumerate(args.batch_sizes):
            axis = axes[ri][ci]
            selected = sorted(
                (
                    row
                    for row in merged
                    if row["model"] == model
                    and int(row["batch_size"]) == batch
                    and row["routing_scope"] == "global"
                ),
                key=lambda row: int(row["dense_fraction"].split("/")[0]),
            )
            axis.plot(
                fractions := [
                    int(row["dense_fraction"].split("/")[0]) / 8
                    for row in selected
                ],
                values := [
                    float(row["wall_speedup_vs_dense"]) for row in selected
                ],
                marker="o",
                color="#dd8452",
            )
            for x_value, y_value in zip(fractions, values, strict=True):
                axis.annotate(
                    f"{y_value:.3f}x",
                    (x_value, y_value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.axhline(1.1, color="gray", linestyle=":", linewidth=0.8)
            axis.set_ylim(0.98, 1.28)
            axis.set_title(f"{model}, B={batch}, M={batch * 8}")
            axis.set_xlabel("Dense-token fraction")
            axis.set_ylabel("Full-call wall throughput speedup")
            axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "vllm_wall_speedup.png", dpi=220)
    figure.savefig(figures / "vllm_wall_speedup.pdf")
    plt.close(figure)
    report = [
        "# vLLM EAGLE3 residual-complement decode benchmark",
        "",
        f"Geometric-mean decode-throughput speedup over dense EAGLE3: {analysis['geomean_speedup']:.4f}x.",
        f"Geometric-mean full-call wall-throughput speedup: {analysis['geomean_wall_speedup']:.4f}x.",
        "",
        (
            "Each exact 128-token prefix is installed in the vLLM prefix cache "
            "before warmup. The timed interval is the complete cached-prefix "
            f"generation call ({args.output_tokens} output tokens/request). "
            "Decode throughput uses the interval from the earliest first-token "
            "timestamp to the latest completion timestamp; wall throughput also "
            "includes the remaining block-aligned prefill. K=7, so each full "
            "verifier step contains 8 rows per active request."
        ),
        "",
        "| Model | B | Dense fraction | Dense decode tok/s | Hybrid decode tok/s | Decode speedup | Wall speedup | Dense accept length | Hybrid accept length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged:
        report.append(
            f"| {row['model']} | {row['batch_size']} | {row['dense_fraction']} | "
            f"{float(row['dense_decode_tokens_per_sec_median']):.2f} | "
            f"{float(row['decode_tokens_per_sec_median']):.2f} | "
            f"{float(row['speedup_vs_dense']):.4f}x | "
            f"{float(row['wall_speedup_vs_dense']):.4f}x | "
            f"{float(row['dense_mean_acceptance_length']):.4f} | "
            f"{float(row['mean_acceptance_length']):.4f} |"
        )
    report.extend(
        [
            "",
            "The 1.1x target applies to the steady decode window. Full-call wall "
            "throughput is reported separately because it includes the one-time "
            "32-token prefix-cache tail prefill.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(analysis, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Regenerate summaries and figures from existing worker JSON files.",
    )
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-backend", choices=("dense", "residual_complement"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)))
    parser.add_argument("--scopes", default=",".join(SCOPES))
    parser.add_argument("--eighths", default=",".join(map(str, EIGHTHS)))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path, default=EVAL_ROOT / "temp/vllm_residual_complement_decode")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=272)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help=(
            "Scheduler token budget. 8192 keeps the B256 cached-prefix tail "
            "out of mixed eager verifier steps."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--route-stats",
        action="store_true",
        help="Enable token-route stats bookkeeping; disabled for clean timing.",
    )
    parser.add_argument(
        "--collect-acceptance",
        action="store_true",
        help=(
            "Enable vLLM speculative metrics. Use a separate pass so metric "
            "collection does not contaminate clean throughput timing."
        ),
    )
    args = parser.parse_args()
    args.models = parse_csv(args.models, MODELS, "models")
    args.batch_sizes = parse_int_csv(args.batch_sizes, BATCH_SIZES, "batch sizes")
    args.scopes = parse_csv(args.scopes, SCOPES, "scopes")
    args.eighths = parse_int_csv(args.eighths, EIGHTHS, "eighths")
    if args.worker and (
        args.worker_model is None
        or args.worker_backend is None
        or args.worker_output is None
    ):
        parser.error("worker requires model/backend/output")
    if (
        args.worker
        and args.worker_backend == "residual_complement"
        and (len(args.scopes) != 1 or len(args.eighths) != 1)
    ):
        parser.error(
            "CUDA-graph residual workers require exactly one scope and one quota"
        )
    if not args.worker and args.output_root is None:
        parser.error("coordinator requires --output-root")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        run_coordinator(parsed)
