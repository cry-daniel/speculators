#!/usr/bin/env python3
"""Benchmark exact DLM selected-token log-probability reductions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import chain
from pathlib import Path
from typing import Callable

import torch

import sparse24_benchmark_common as common
from vllm.speclink_token_dense import (
    _greedy_token_logprob_kernel,
    greedy_sample_with_logprob,
)


WARMUPS = 10
TRIALS = 10
ITERATIONS = 100
SHAPES = (
    ("qwen3_8b", 64, 151936),
    ("qwen3_8b", 128, 151936),
    ("qwen3_8b", 256, 151936),
    ("llama3_1_8b", 64, 128256),
    ("llama3_1_8b", 128, 128256),
    ("llama3_1_8b", 256, 128256),
)
METHODS = ("argmax", "single_cta_logprob", "two_stage_logprob")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def method_orders() -> tuple[tuple[str, ...], ...]:
    base = (
        ("argmax", "single_cta_logprob", "two_stage_logprob"),
        ("single_cta_logprob", "two_stage_logprob", "argmax"),
        ("two_stage_logprob", "argmax", "single_cta_logprob"),
        ("argmax", "two_stage_logprob", "single_cta_logprob"),
        ("two_stage_logprob", "single_cta_logprob", "argmax"),
    )
    return base + tuple(tuple(reversed(order)) for order in base)


def event_sample(call: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERATIONS):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / ITERATIONS


def run_shape(model: str, rows: int, vocab: int) -> list[dict[str, object]]:
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.bfloat16)
    old_ids = torch.empty(rows, device="cuda", dtype=torch.int64)
    old_logprobs = torch.empty(rows, device="cuda", dtype=torch.float32)

    def argmax() -> torch.Tensor:
        return logits.argmax(dim=-1)

    def single_cta() -> tuple[torch.Tensor, torch.Tensor]:
        _greedy_token_logprob_kernel[(rows,)](
            old_ids,
            old_logprobs,
            logits,
            logits.stride(0),
            vocab,
            BLOCK_SIZE=1024,
        )
        return old_ids, old_logprobs

    def two_stage() -> tuple[torch.Tensor, torch.Tensor]:
        return greedy_sample_with_logprob(logits)

    calls: dict[str, Callable[[], object]] = {
        "argmax": argmax,
        "single_cta_logprob": single_cta,
        "two_stage_logprob": two_stage,
    }
    for call in calls.values():
        for _ in range(WARMUPS):
            call()
    torch.cuda.synchronize()
    samples = {method: [] for method in METHODS}
    for order in method_orders():
        for method in order:
            samples[method].append(event_sample(calls[method]))

    new_ids, new_logprobs = two_stage()
    reference_values, reference_ids = torch.max(logits.float(), dim=-1)
    reference_logprobs = reference_values - torch.logsumexp(logits.float(), dim=-1)
    ids_equal = bool(torch.equal(new_ids, reference_ids))
    max_logprob_error = float((new_logprobs - reference_logprobs).abs().max())
    return [
        {
            "model": model,
            "rows": rows,
            "vocab": vocab,
            "method": method,
            "median_us": statistics.median(samples[method]),
            "p10_us": percentile(samples[method], 0.1),
            "p90_us": percentile(samples[method], 0.9),
            "samples_us": json.dumps(samples[method]),
            "token_ids_equal": ids_equal if method == "two_stage_logprob" else "",
            "max_logprob_abs_error": (
                max_logprob_error if method == "two_stage_logprob" else ""
            ),
        }
        for method in METHODS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    common.require_idle_gpu(args.device_index)
    rows = list(chain.from_iterable(run_shape(*shape) for shape in SHAPES))
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "confidence_reduction.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "warmups": WARMUPS,
        "trials": TRIALS,
        "iterations_per_trial": ITERATIONS,
        "timing": "one CUDA Event interval per trial; synchronization outside interval",
        "order_balance": "five orders and their reverses",
        "gpu_idle_check": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    (args.output_root / "confidence_reduction_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_root / "confidence_reduction.csv")


if __name__ == "__main__":
    main()
