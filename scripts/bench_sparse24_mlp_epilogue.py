#!/usr/bin/env python3
"""Benchmark fused sparse Down materialization and residual add."""

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
    sparse24_transpose_add_residual_,
    sparse24_transpose_add_rmsnorm,
    sparse24_transpose_output_contiguous,
)


HIDDEN_SIZE = 4096
EPSILON = 1e-6
DEFAULT_ROWS = (112, 144, 176, 224, 288, 352, 448, 576, 704)
DEFAULT_CONFIGS = ("2", "4", "8", "16", "32")


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item.strip())
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be positive comma-separated ints")
    return rows


def _parse_configs(value: str) -> tuple[str, ...]:
    configs = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in configs if item not in DEFAULT_CONFIGS]
    if not configs or invalid:
        raise argparse.ArgumentTypeError(
            "configs must be drawn from " + ",".join(DEFAULT_CONFIGS)
        )
    return configs


def _graph_median_ms(
    fn: Callable[[], object],
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
    rows: int,
    *,
    configs: tuple[str, ...],
    warmup: int,
    repeat: int,
    trials: int,
) -> list[dict[str, float | int | str]]:
    generator = torch.Generator(device="cuda").manual_seed(2000 + rows)
    logical_down = (
        torch.randn(
            (rows, HIDDEN_SIZE),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 1e-3
    )
    leading_dim = ((rows + 7) // 8) * 8
    transposed_down = torch.empty_strided(
        logical_down.shape,
        (1, leading_dim),
        device="cuda",
        dtype=torch.float16,
    )
    transposed_down.copy_(logical_down)
    residual_initial = torch.randn(
        logical_down.shape,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.float16)

    baseline_hidden = torch.empty_like(logical_down)
    baseline_residual = residual_initial.clone()

    def baseline() -> torch.Tensor:
        sparse24_transpose_output_contiguous(
            transposed_down,
            out=baseline_hidden,
        )
        vllm_ops.fused_add_rms_norm(
            baseline_hidden,
            baseline_residual,
            weight,
            EPSILON,
        )
        return baseline_hidden

    fused_hidden = torch.empty_like(logical_down)
    fused_residual = residual_initial.clone()

    def fused() -> torch.Tensor:
        sparse24_transpose_add_residual_(transposed_down, fused_residual)
        vllm_ops.rms_norm(
            fused_hidden,
            fused_residual,
            weight,
            EPSILON,
        )
        return fused_hidden

    expected = baseline().clone()
    expected_residual = baseline_residual.clone()
    actual = fused().clone()
    torch.cuda.synchronize()
    output_diff = float((actual - expected).abs().max().item())
    residual_diff = float(
        (fused_residual - expected_residual).abs().max().item()
    )
    baseline_ms = _graph_median_ms(
        baseline, warmup=warmup, repeat=repeat, trials=trials
    )
    fused_ms = _graph_median_ms(
        fused, warmup=warmup, repeat=repeat, trials=trials
    )
    results: list[dict[str, float | int | str]] = [
        {
            "rows": rows,
            "variant": "two_stage",
            "baseline_ms": baseline_ms,
            "fused_ms": fused_ms,
            "speedup": baseline_ms / fused_ms,
            "max_output_abs_diff": output_diff,
            "max_residual_abs_diff": residual_diff,
        }
    ]
    for config in configs:
        single_hidden = torch.empty_like(logical_down)
        single_residual = residual_initial.clone()

        def single_fused(config: str = config) -> torch.Tensor:
            return sparse24_transpose_add_rmsnorm(
                transposed_down,
                single_residual,
                weight,
                epsilon=EPSILON,
                out=single_hidden,
                epilogue_config=config,
            )

        actual_single = single_fused().clone()
        torch.cuda.synchronize()
        single_output_diff = float(
            (actual_single - expected).abs().max().item()
        )
        single_residual_diff = float(
            (single_residual - expected_residual).abs().max().item()
        )
        single_ms = _graph_median_ms(
            single_fused,
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        results.append(
            {
                "rows": rows,
                "variant": f"single_rows_{config}",
                "baseline_ms": baseline_ms,
                "fused_ms": single_ms,
                "speedup": baseline_ms / single_ms,
                "max_output_abs_diff": single_output_diff,
                "max_residual_abs_diff": single_residual_diff,
            }
        )
    return results


def _plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    variants = sorted({str(row["variant"]) for row in rows})
    baseline_rows = [row for row in rows if row["variant"] == variants[0]]
    x = [int(row["rows"]) for row in baseline_rows]
    axes[0].plot(
        x,
        [float(row["baseline_ms"]) for row in baseline_rows],
        marker="o",
        color="#555555",
        label="Transpose + fused add/RMSNorm",
    )
    colors = ("#176B87", "#B33F40", "#6C5B7B", "#2A9D8F", "#E09F3E", "#9E2A2B")
    for variant, color in zip(variants, colors, strict=False):
        selected = [row for row in rows if row["variant"] == variant]
        axes[0].plot(
            [int(row["rows"]) for row in selected],
            [float(row["fused_ms"]) for row in selected],
            marker="s",
            color=color,
            label=variant,
        )
    axes[0].set_title("MLP Down post-op latency")
    axes[0].set_xlabel("Verifier rows (bs x (K+1))")
    axes[0].set_ylabel("CUDA-graph latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for variant, color in zip(variants, colors, strict=False):
        selected = [row for row in rows if row["variant"] == variant]
        axes[1].plot(
            [int(row["rows"]) for row in selected],
            [float(row["speedup"]) for row in selected],
            marker="o",
            color=color,
            label=variant,
        )
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_title("MLP Down post-op speedup")
    axes[1].set_xlabel("Verifier rows (bs x (K+1))")
    axes[1].set_ylabel("Speedup")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument("--configs", type=_parse_configs, default=DEFAULT_CONFIGS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_mlp_epilogue_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = [
        result
        for rows in args.rows
        for result in _run_case(
            rows,
            configs=args.configs,
            warmup=args.warmup,
            repeat=args.repeat,
            trials=args.trials,
        )
    ]
    csv_path = args.output_root / "mlp_epilogue_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, args.output_root / "mlp_epilogue_speedup.png")
    print(csv_path)
    for row in results:
        print(
            f"M={int(row['rows']):>3} "
            f"{str(row['variant']):<16} "
            f"{float(row['baseline_ms']):.4f}->{float(row['fused_ms']):.4f} ms "
            f"speedup={float(row['speedup']):.3f}x "
            f"output_diff={float(row['max_output_abs_diff']):.5f} "
            f"residual_diff={float(row['max_residual_abs_diff']):.5f}"
        )


if __name__ == "__main__":
    main()
