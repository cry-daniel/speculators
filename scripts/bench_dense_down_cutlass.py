#!/usr/bin/env python3
"""Benchmark dense verifier projections before adding a fused epilogue."""

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

from vllm.speclink_kernel import (  # noqa: E402
    dense_cutlass_device_gemm,
    dense_cutlass_weight_t_gemm,
)


PROJECTIONS = {
    "qwen_gate": (4096, 24576),
    "qwen_down": (12288, 4096),
    "qwen_o": (4096, 4096),
    "llama_gate": (4096, 28672),
    "llama_down": (14336, 4096),
    "llama_o": (4096, 4096),
}
DEFAULT_ROWS = "112,144,176,224,288,352,448,576,704"
DEFAULT_CONFIGS = "auto"
DEFAULT_ACCUMULATORS = "fp32,fp16"


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


def _graph_median_ms(
    graph: torch.cuda.CUDAGraph,
    *,
    unroll: int,
    replays: int,
    trials: int,
) -> float:
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


def _time(
    fn: Callable[[], torch.Tensor],
    *,
    unroll: int,
    replays: int,
    trials: int,
) -> float:
    graph = _capture_unrolled(fn, unroll)
    return _graph_median_ms(
        graph, unroll=unroll, replays=replays, trials=trials
    )


def _run_projection(
    label: str,
    rows: tuple[int, ...],
    *,
    configs: tuple[str, ...],
    accumulators: tuple[str, ...],
    seed: int,
    unroll: int,
    replays: int,
    trials: int,
) -> list[dict[str, object]]:
    k, n = PROJECTIONS[label]
    generator = torch.Generator(device="cuda").manual_seed(seed + k + n)
    # weight_nk matches vLLM's LinearBase storage. weight_kn is benchmark-only
    # and lets the existing row-major CUTLASS entry point run without timing a
    # transpose or allocating one per invocation.
    weight_nk = torch.randn(
        (n, k), device="cuda", dtype=torch.float16, generator=generator
    ).mul_(0.01)
    weight_kn = weight_nk.t().contiguous()
    results: list[dict[str, object]] = []
    for m in rows:
        x = torch.randn(
            (m, k), device="cuda", dtype=torch.float16, generator=generator
        ).mul_(0.02)
        out_t = torch.empty((m, n), device="cuda", dtype=torch.float16)
        out_kn = torch.empty_like(out_t)
        out_cutlass_rowmajor = torch.empty_like(out_t)
        out_cutlass_weight_t = torch.empty_like(out_t)

        def torch_weight_t() -> torch.Tensor:
            return torch.mm(x, weight_nk.t(), out=out_t)

        def torch_rowmajor() -> torch.Tensor:
            return torch.mm(x, weight_kn, out=out_kn)

        torch_weight_t()
        torch_rowmajor()
        torch.cuda.synchronize()
        torch_weight_t_ms = _time(
            torch_weight_t, unroll=unroll, replays=replays, trials=trials
        )
        torch_rowmajor_ms = _time(
            torch_rowmajor, unroll=unroll, replays=replays, trials=trials
        )
        for accumulator in accumulators:
            for config in configs:

                def cutlass_rowmajor(
                    accumulator: str = accumulator, config: str = config
                ) -> torch.Tensor:
                    return dense_cutlass_device_gemm(
                        x,
                        weight_kn,
                        out=out_cutlass_rowmajor,
                        accumulator=accumulator,
                        device_config=config,
                    )

                def cutlass_weight_t(
                    accumulator: str = accumulator, config: str = config
                ) -> torch.Tensor:
                    return dense_cutlass_weight_t_gemm(
                        x,
                        weight_nk,
                        out=out_cutlass_weight_t,
                        accumulator=accumulator,
                        device_config=config,
                    )

                cutlass_rowmajor()
                cutlass_weight_t()
                torch.cuda.synchronize()
                max_abs_diff = float(
                    (out_t.float() - out_cutlass_weight_t.float())
                    .abs()
                    .max()
                    .item()
                )
                mean_abs_diff = float(
                    (out_t.float() - out_cutlass_weight_t.float())
                    .abs()
                    .mean()
                    .item()
                )
                cutlass_rowmajor_ms = _time(
                    cutlass_rowmajor,
                    unroll=unroll,
                    replays=replays,
                    trials=trials,
                )
                cutlass_weight_t_ms = _time(
                    cutlass_weight_t,
                    unroll=unroll,
                    replays=replays,
                    trials=trials,
                )
                row = {
                    "projection": label,
                    "M": m,
                    "K": k,
                    "N": n,
                    "config": config,
                    "accumulator": accumulator,
                    "torch_weight_t_ms": torch_weight_t_ms,
                    "torch_rowmajor_ms": torch_rowmajor_ms,
                    "cutlass_rowmajor_ms": cutlass_rowmajor_ms,
                    "cutlass_weight_t_ms": cutlass_weight_t_ms,
                    "weight_t_speedup_vs_torch": (
                        torch_weight_t_ms / cutlass_weight_t_ms
                    ),
                    "rowmajor_speedup_vs_torch": (
                        torch_rowmajor_ms / cutlass_rowmajor_ms
                    ),
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                }
                results.append(row)
                print(
                    f"{label} M={m} config={config} accum={accumulator}: "
                    f"torch(W.T)={torch_weight_t_ms:.4f} ms "
                    f"cutlass(W.T)={cutlass_weight_t_ms:.4f} ms "
                    f"speedup={row['weight_t_speedup_vs_torch']:.3f}x "
                    f"rowmajor={cutlass_rowmajor_ms:.4f} ms "
                    f"max_diff={max_abs_diff:.3g}",
                    flush=True,
                )
        del x, out_t, out_kn, out_cutlass_rowmajor, out_cutlass_weight_t
    del weight_nk, weight_kn
    torch.cuda.empty_cache()
    return results


def _plot(results: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    series = list(
        dict.fromkeys(
            (
                str(row["projection"]),
                str(row["config"]),
                str(row["accumulator"]),
            )
            for row in results
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for label, config, accumulator in series:
        subset = [
            row
            for row in results
            if row["projection"] == label
            and row["config"] == config
            and row["accumulator"] == accumulator
        ]
        x = [int(row["M"]) for row in subset]
        axes[0].plot(
            x,
            [float(row["torch_weight_t_ms"]) for row in subset],
            marker="o",
            label=f"{label}: cuBLAS" if config == series[0][1] else "_nolegend_",
        )
        axes[0].plot(
            x,
            [float(row["cutlass_weight_t_ms"]) for row in subset],
            marker="x",
            linestyle="--",
            label=f"{label}: {config}/{accumulator}",
        )
        axes[1].plot(
            x,
            [float(row["weight_t_speedup_vs_torch"]) for row in subset],
            marker="o",
            label=f"{label}: {config}/{accumulator}",
        )
    axes[0].set_xlabel("Verifier rows (M)")
    axes[0].set_ylabel("Kernel time (ms)")
    axes[0].set_title("Dense projection latency")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xlabel("Verifier rows (M)")
    axes[1].set_ylabel("Speedup vs cuBLAS")
    axes[1].set_title("CUTLASS direct weight_t path")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projections",
        type=_parse_csv,
        default=tuple(PROJECTIONS),
        help="Comma-separated projection labels",
    )
    parser.add_argument("--rows", type=_parse_rows, default=_parse_rows(DEFAULT_ROWS))
    parser.add_argument(
        "--configs", type=_parse_csv, default=_parse_csv(DEFAULT_CONFIGS)
    )
    parser.add_argument(
        "--accumulators",
        type=_parse_csv,
        default=_parse_csv(DEFAULT_ACCUMULATORS),
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
    invalid_accumulators = [
        item for item in args.accumulators if item not in {"fp16", "fp32"}
    ]
    if invalid_accumulators:
        parser.error(f"unknown accumulators: {','.join(invalid_accumulators)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_dir = args.output_dir or (
        REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"dense_down_cutlass_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for label in args.projections:
        results.extend(
            _run_projection(
                label,
                args.rows,
                configs=args.configs,
                accumulators=args.accumulators,
                seed=args.seed,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
        )

    csv_path = output_dir / "dense_down_cutlass.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, output_dir / "dense_down_cutlass.png")
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
