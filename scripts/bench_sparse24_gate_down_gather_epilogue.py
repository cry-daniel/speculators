#!/usr/bin/env python3
"""Benchmark fused routed SwiGLU correction plus compact Down-input store."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    sparse24_gather_rows_,
    sparse24_routed_swiglu_correction_,
    sparse24_routed_swiglu_correction_gather_,
)


MODELS = {
    "qwen3_8b": {"intermediate": 12288, "ratio": 0.125, "cap": 32},
    "llama3_1_8b": {"intermediate": 14336, "ratio": 0.3125, "cap": 64},
}


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def _csv_models(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in values if item not in MODELS]
    if not values or invalid:
        raise argparse.ArgumentTypeError(f"unsupported models: {invalid}")
    return values


def _dense_counts(model: str, batch_size: int, k: int) -> tuple[int, int]:
    config = MODELS[model]
    scored_rows = batch_size * k
    ratio_budget = int(float(config["ratio"]) * scored_rows + 0.5)
    dense_count = min(
        scored_rows,
        int(config["cap"]),
        max(batch_size, ratio_budget),
    )
    return dense_count, max(8, math.ceil(dense_count / 8) * 8)


def _capture(fn: Callable[[], torch.Tensor], repeat: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            output = fn()
    torch.cuda.synchronize()
    del output
    return graph


def _sample_ms(graph: torch.cuda.CUDAGraph, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _paired_median_ms(
    baseline: Callable[[], torch.Tensor],
    fused: Callable[[], torch.Tensor],
    *,
    repeat: int,
    trials: int,
) -> tuple[float, float]:
    baseline_graph = _capture(baseline, repeat)
    fused_graph = _capture(fused, repeat)
    baseline_samples: list[float] = []
    fused_samples: list[float] = []
    for trial in range(trials):
        if trial % 2:
            fused_samples.append(_sample_ms(fused_graph, repeat))
            baseline_samples.append(_sample_ms(baseline_graph, repeat))
        else:
            baseline_samples.append(_sample_ms(baseline_graph, repeat))
            fused_samples.append(_sample_ms(fused_graph, repeat))
    return statistics.median(baseline_samples), statistics.median(fused_samples)


def _run_case(
    model: str,
    batch_size: int,
    k: int,
    generator: torch.Generator,
    *,
    repeat: int,
    trials: int,
) -> dict[str, object]:
    rows = batch_size * (k + 1)
    dense_count, dense_run = _dense_counts(model, batch_size, k)
    intermediate = int(MODELS[model]["intermediate"])
    gate_up = intermediate * 2

    dense_base = torch.randn(
        (dense_count, gate_up),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense_residual = torch.randn(
        (dense_count, gate_up),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense_base.mul_(0.1)
    dense_residual.mul_(0.1)
    dense_rows = torch.randperm(rows, device="cuda", generator=generator)[
        :dense_count
    ].sort().values.to(torch.int32)
    sparse_hidden = torch.randn(
        (rows, intermediate),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    baseline_out = sparse_hidden.clone()
    fused_out = sparse_hidden.clone()
    baseline_dense = torch.empty(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    fused_dense = torch.empty_like(baseline_dense)

    def baseline() -> torch.Tensor:
        sparse24_routed_swiglu_correction_(
            dense_base,
            dense_residual,
            dense_rows,
            baseline_out,
        )
        sparse24_gather_rows_(
            baseline_out,
            dense_rows,
            baseline_dense[:dense_count],
        )
        return baseline_out

    def fused() -> torch.Tensor:
        sparse24_routed_swiglu_correction_gather_(
            dense_base,
            dense_residual,
            dense_rows,
            fused_out,
            fused_dense,
        )
        return fused_out

    baseline()
    fused()
    torch.cuda.synchronize()
    output_diff = float(
        (baseline_out.float() - fused_out.float()).abs().max().item()
    )
    dense_diff = float(
        (
            baseline_dense[:dense_count].float()
            - fused_dense[:dense_count].float()
        )
        .abs()
        .max()
        .item()
    )
    if output_diff != 0.0 or dense_diff != 0.0:
        raise RuntimeError(
            f"fused correction/gather mismatch for {model} bs={batch_size} "
            f"K={k}: output={output_diff}, dense={dense_diff}"
        )

    baseline_ms, fused_ms = _paired_median_ms(
        baseline,
        fused,
        repeat=repeat,
        trials=trials,
    )
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_count": dense_count,
        "dense_run": dense_run,
        "intermediate_size": intermediate,
        "baseline_correction_gather_ms": baseline_ms,
        "fused_correction_gather_ms": fused_ms,
        "epilogue_speedup": baseline_ms / fused_ms,
        "output_max_abs_diff": output_diff,
        "dense_hidden_max_abs_diff": dense_diff,
    }


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    markers = {"qwen3_8b": "o", "llama3_1_8b": "s"}
    cases = [(batch, k) for batch in (16, 32, 64) for k in (6, 8, 10)]
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for model in MODELS:
        selected = {
            (int(row["batch_size"]), int(row["K"])): row
            for row in rows
            if row["model"] == model
        }
        if not selected:
            continue
        axis.plot(
            range(len(cases)),
            [float(selected[case]["epilogue_speedup"]) for case in cases],
            color=colors[model],
            marker=markers[model],
            linewidth=1.6,
            label=model,
        )
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_title("Gate correction + compact Down-input epilogue")
    axis.set_xlabel("Batch size / K")
    axis.set_ylabel("Correction + gather / fused latency")
    axis.set_xticks(
        range(len(cases)),
        [f"{batch}/{k}" for batch, k in cases],
        rotation=35,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=_csv_models, default=tuple(MODELS))
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(16, 32, 64))
    parser.add_argument("--k-values", type=_csv_ints, default=(6, 8, 10))
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_gate_down_gather_epilogue_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.repeat <= 0 or args.trials <= 0:
        raise ValueError("repeat and trials must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(20260716)
    rows = [
        _run_case(
            model,
            batch_size,
            k,
            generator,
            repeat=args.repeat,
            trials=args.trials,
        )
        for model in args.models
        for batch_size in args.batch_sizes
        for k in args.k_values
    ]
    csv_path = args.output_root / "gate_down_gather_epilogue.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output_root / "gate_down_gather_epilogue.png")
    for row in rows:
        print(
            f"{row['model']} bs={row['batch_size']} K={row['K']} "
            f"{float(row['baseline_correction_gather_ms']):.4f}->"
            f"{float(row['fused_correction_gather_ms']):.4f} ms "
            f"speedup={float(row['epilogue_speedup']):.3f}x"
        )
    print(csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
