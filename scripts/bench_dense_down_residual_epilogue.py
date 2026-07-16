#!/usr/bin/env python3
"""Benchmark dense Down GEMM with an in-epilogue residual add."""

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
    dense_cutlass_weight_t_gemm_add,
)


PROJECTIONS = {
    "qwen_down": (12288, 4096),
    "llama_down": (14336, 4096),
}
DEFAULT_ROWS = "112,144,176,224,288,352,448,576,704"
DEFAULT_CONFIGS = (
    "torch_addmm,auto,64x64x64_s4,64x128x64_s3,"
    "fp32_64x64x64_s4,fp32_64x128x64_s3"
)
EPSILON = 1e-6


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item.strip())
    if not rows or any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("rows must be positive comma-separated ints")
    return rows


def _capture_unrolled(
    fn: Callable[[], torch.Tensor], unroll: int
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            captured = fn()
    del captured
    torch.cuda.synchronize()
    return graph


def _time_graph(
    fn: Callable[[], torch.Tensor],
    *,
    unroll: int,
    replays: int,
    trials: int,
) -> float:
    graph = _capture_unrolled(fn, unroll)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (replays * unroll))
    return statistics.median(samples)


def _run_projection(
    label: str,
    rows: tuple[int, ...],
    configs: tuple[str, ...],
    *,
    seed: int,
    unroll: int,
    replays: int,
    trials: int,
) -> list[dict[str, object]]:
    k, n = PROJECTIONS[label]
    generator = torch.Generator(device="cuda").manual_seed(seed + k)
    weight_t = torch.randn(
        (n, k), device="cuda", dtype=torch.float16, generator=generator
    ).mul_(0.01)
    norm_weight = torch.randn(
        (n,), device="cuda", dtype=torch.float16, generator=generator
    ).mul_(0.02).add_(1.0)
    results: list[dict[str, object]] = []
    for m in rows:
        x = torch.randn(
            (m, k), device="cuda", dtype=torch.float16, generator=generator
        ).mul_(0.1)
        residual_initial = torch.randn(
            (m, n), device="cuda", dtype=torch.float16, generator=generator
        ).mul_(0.1)

        baseline_hidden = torch.empty_like(residual_initial)
        baseline_residual = residual_initial.clone()

        def baseline() -> torch.Tensor:
            torch.mm(x, weight_t.t(), out=baseline_hidden)
            vllm_ops.fused_add_rms_norm(
                baseline_hidden, baseline_residual, norm_weight, EPSILON
            )
            return baseline_hidden

        baseline_residual.copy_(residual_initial)
        expected = baseline().clone()
        expected_residual = baseline_residual.clone()
        torch.cuda.synchronize()
        baseline_residual.copy_(residual_initial)
        baseline_ms = _time_graph(
            baseline, unroll=unroll, replays=replays, trials=trials
        )

        for config in configs:
            fused_hidden = torch.empty_like(residual_initial)
            fused_residual = residual_initial.clone()
            if config == "torch_addmm":
                accumulator = "torch"
            elif config.startswith("fp32_"):
                accumulator = "fp32"
            else:
                accumulator = "fp16"
            device_config = config.removeprefix("fp32_")

            def fused(config: str = config) -> torch.Tensor:
                if config == "torch_addmm":
                    torch.addmm(
                        fused_residual,
                        x,
                        weight_t.t(),
                        out=fused_residual,
                    )
                else:
                    dense_cutlass_weight_t_gemm_add(
                        x,
                        weight_t,
                        fused_residual,
                        out=fused_residual,
                        accumulator=accumulator,
                        device_config=device_config,
                    )
                vllm_ops.rms_norm(
                    fused_hidden,
                    fused_residual,
                    norm_weight,
                    EPSILON,
                )
                return fused_hidden

            fused_residual.copy_(residual_initial)
            actual = fused().clone()
            actual_residual = fused_residual.clone()
            torch.cuda.synchronize()
            output_max_abs_diff = float(
                (expected.float() - actual.float()).abs().max().item()
            )
            residual_max_abs_diff = float(
                (expected_residual.float() - actual_residual.float())
                .abs()
                .max()
                .item()
            )
            output_mean_abs_diff = float(
                (expected.float() - actual.float()).abs().mean().item()
            )
            fused_residual.copy_(residual_initial)
            fused_ms = _time_graph(
                fused, unroll=unroll, replays=replays, trials=trials
            )
            row = {
                "projection": label,
                "M": m,
                "K": k,
                "N": n,
                "config": config,
                "accumulator": accumulator,
                "baseline_ms": baseline_ms,
                "fused_ms": fused_ms,
                "speedup": baseline_ms / fused_ms,
                "output_max_abs_diff": output_max_abs_diff,
                "output_mean_abs_diff": output_mean_abs_diff,
                "residual_max_abs_diff": residual_max_abs_diff,
            }
            results.append(row)
            print(
                f"{label} M={m} config={config}: "
                f"baseline={baseline_ms:.4f} ms fused={fused_ms:.4f} ms "
                f"speedup={row['speedup']:.3f}x "
                f"output_diff={output_max_abs_diff:.4g} "
                f"residual_diff={residual_max_abs_diff:.4g}",
                flush=True,
            )
            del fused_hidden, fused_residual, actual, actual_residual
        del (
            x,
            residual_initial,
            baseline_hidden,
            baseline_residual,
            expected,
            expected_residual,
        )
    del weight_t, norm_weight
    torch.cuda.empty_cache()
    return results


def _plot(results: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    series = list(
        dict.fromkeys(
            (str(row["projection"]), str(row["config"])) for row in results
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for label, config in series:
        subset = [
            row
            for row in results
            if row["projection"] == label and row["config"] == config
        ]
        x = [int(row["M"]) for row in subset]
        axes[0].plot(
            x,
            [float(row["fused_ms"]) for row in subset],
            marker="o",
            label=f"{label}: {config}",
        )
        axes[1].plot(
            x,
            [float(row["speedup"]) for row in subset],
            marker="o",
            label=f"{label}: {config}",
        )
    baseline_rows = [
        row for row in results if row["config"] == series[0][1]
    ]
    for label in dict.fromkeys(str(row["projection"]) for row in baseline_rows):
        subset = [row for row in baseline_rows if row["projection"] == label]
        axes[0].plot(
            [int(row["M"]) for row in subset],
            [float(row["baseline_ms"]) for row in subset],
            linestyle="--",
            label=f"{label}: baseline",
        )
    axes[0].set_xlabel("Verifier rows (M)")
    axes[0].set_ylabel("Down + residual norm (ms)")
    axes[0].set_title("MLP tail latency")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xlabel("Verifier rows (M)")
    axes[1].set_ylabel("Speedup vs cuBLAS + fused norm")
    axes[1].set_title("Residual epilogue speedup")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections", type=_parse_csv, default=tuple(PROJECTIONS)
    )
    parser.add_argument("--rows", type=_parse_rows, default=_parse_rows(DEFAULT_ROWS))
    parser.add_argument(
        "--configs", type=_parse_csv, default=_parse_csv(DEFAULT_CONFIGS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    invalid = [label for label in args.projections if label not in PROJECTIONS]
    if invalid:
        parser.error(f"unknown projections: {','.join(invalid)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_dir = args.output_dir or (
        REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"dense_down_residual_epilogue_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for label in args.projections:
        results.extend(
            _run_projection(
                label,
                args.rows,
                args.configs,
                seed=args.seed,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
        )

    csv_path = output_dir / "dense_down_residual_epilogue.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, output_dir / "dense_down_residual_epilogue.png")
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
