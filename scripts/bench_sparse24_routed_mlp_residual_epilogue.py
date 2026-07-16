#!/usr/bin/env python3
"""Benchmark exact routed Down merge directly into the model residual."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm import _custom_ops as vllm_ops  # noqa: E402
from vllm.speclink_kernel import (  # noqa: E402
    sparse24_transpose_add_routed_residual,
    sparse24_transpose_add_routed_residual_rmsnorm,
)


HIDDEN_SIZE = 4096
EPSILON = 1e-6
ROWS = (112, 144, 176, 224, 288, 352, 448, 576, 704)
DENSE_COUNTS = {
    "qwen3_8b": (16, 16, 20, 28, 32, 32, 32, 32, 32),
    "llama3_1_8b": (30, 40, 50, 60, 64, 64, 64, 64, 64),
}


def _graph_median_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            captured_output = fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    measurements: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) / repeat)
    del captured_output
    return statistics.median(measurements)


def _run_case(
    model: str,
    rows: int,
    dense_count: int,
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> dict[str, float | int | str]:
    generator = torch.Generator(device="cuda").manual_seed(3100 + rows + dense_count)
    leading_dim = (rows + 7) // 8 * 8
    dense_run = (dense_count + 7) // 8 * 8
    logical_full = torch.randn(
        (rows, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ) * 1e-4
    logical_correction = torch.randn(
        (dense_count, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ) * 1e-4
    full = torch.empty_strided(
        (rows, HIDDEN_SIZE),
        (1, leading_dim),
        device="cuda",
        dtype=torch.float16,
    )
    correction = torch.empty_strided(
        (dense_run, HIDDEN_SIZE),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    full.copy_(logical_full)
    correction[:dense_count].copy_(logical_correction)

    dense_rows = torch.randperm(rows, device="cuda", generator=generator)[
        :dense_count
    ].sort().values.to(torch.int64)
    dense_slots = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    dense_slots[dense_rows] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    initial_residual = torch.randn(
        (rows, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight = torch.randn(
        (HIDDEN_SIZE,),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )

    baseline_hidden = torch.empty_like(initial_residual)
    baseline_residual = initial_residual.clone()

    def baseline() -> torch.Tensor:
        sparse24_transpose_add_routed_residual(
            full,
            correction,
            dense_slots,
            dense_count=dense_count,
            out=baseline_hidden,
        )
        vllm_ops.fused_add_rms_norm(
            baseline_hidden,
            baseline_residual,
            weight,
            EPSILON,
        )
        return baseline_hidden

    fused_hidden = torch.empty_like(initial_residual)
    fused_residual = initial_residual.clone()
    square_partials = torch.empty(
        (rows, HIDDEN_SIZE // 32), device="cuda", dtype=torch.float32
    )

    def fused() -> torch.Tensor:
        return sparse24_transpose_add_routed_residual_rmsnorm(
            full,
            correction,
            dense_slots,
            fused_residual,
            weight,
            dense_count=dense_count,
            epsilon=EPSILON,
            out=fused_hidden,
            square_partials=square_partials,
        )

    expected = baseline().clone()
    expected_residual = baseline_residual.clone()
    actual = fused().clone()
    torch.cuda.synchronize()
    output_diff = float((actual - expected).abs().max().item())
    residual_diff = float(
        (fused_residual - expected_residual).abs().max().item()
    )

    baseline_residual.copy_(initial_residual)
    fused_residual.copy_(initial_residual)
    baseline_ms = _graph_median_ms(
        baseline, warmup=warmup, repeat=repeat, trials=trials
    )
    fused_ms = _graph_median_ms(
        fused, warmup=warmup, repeat=repeat, trials=trials
    )
    return {
        "model": model,
        "rows": rows,
        "dense_count": dense_count,
        "baseline_ms": baseline_ms,
        "fused_ms": fused_ms,
        "speedup": baseline_ms / fused_ms,
        "max_output_abs_diff": output_diff,
        "max_residual_abs_diff": residual_diff,
    }


def _plot(results: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    for model in DENSE_COUNTS:
        rows = [row for row in results if row["model"] == model]
        color = colors[model]
        axes[0].plot(
            [int(row["rows"]) for row in rows],
            [float(row["baseline_ms"]) for row in rows],
            marker="o",
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"{model} baseline",
        )
        axes[0].plot(
            [int(row["rows"]) for row in rows],
            [float(row["fused_ms"]) for row in rows],
            marker="s",
            color=color,
            label=f"{model} fused",
        )
        axes[1].plot(
            [int(row["rows"]) for row in rows],
            [float(row["speedup"]) for row in rows],
            marker="o",
            color=color,
            label=model,
        )
    axes[0].set_title("Routed Down + residual/RMSNorm")
    axes[0].set_xlabel("Verifier rows")
    axes[0].set_ylabel("CUDA-graph latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_title("Fused epilogue speedup")
    axes[1].set_xlabel("Verifier rows")
    axes[1].set_ylabel("Baseline / fused")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", default="qwen3_8b,llama3_1_8b"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    invalid = [model for model in models if model not in DENSE_COUNTS]
    if not models or invalid:
        parser.error(f"unsupported models: {invalid}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_root = args.output_root or (
        REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_routed_mlp_residual_epilogue_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(
            model,
            rows,
            dense_count,
            warmup=args.warmup,
            repeat=args.repeat,
            trials=args.trials,
        )
        for model in models
        for rows, dense_count in zip(ROWS, DENSE_COUNTS[model], strict=True)
    ]
    csv_path = output_root / "routed_mlp_residual_epilogue.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, output_root / "routed_mlp_residual_epilogue.png")
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
