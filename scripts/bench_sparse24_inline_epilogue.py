#!/usr/bin/env python3
"""Benchmark the experimental inline-transpose CUTLASS sparse epilogue."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
)


DEFAULT_ROWS = "112,144,176,224,288,352,448,576,704"
DEFAULT_CONFIGS = "auto"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def capture_unrolled(fn: Callable[[], object], unroll: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    torch.cuda.synchronize()
    return graph


def graph_milliseconds(
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


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weight = torch.randn(
        (args.k, args.n),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    values, meta = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=args.k,
    )
    del weight, weight24, packed

    rows: list[dict[str, object]] = []
    for config in args.configs:
        for m in args.rows:
            x = torch.randn(
                (m, args.k),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            output_view = torch.empty_strided(
                (m, args.n),
                (1, m),
                device="cuda",
                dtype=torch.float16,
            )
            output_contiguous = torch.empty(
                (m, args.n), device="cuda", dtype=torch.float16
            )
            workspace = torch.empty(
                (args.n, m), device="cuda", dtype=torch.float16
            )
            def view_fn() -> torch.Tensor:
                return sparse24_cutlass_device_gemm_prepacked(
                    x,
                    values,
                    meta,
                    out=output_view,
                    device_config=config,
                )

            def contiguous_fn() -> torch.Tensor:
                return sparse24_cutlass_device_gemm_prepacked(
                    x,
                    values,
                    meta,
                    contiguous_output=True,
                    out=output_contiguous,
                    workspace=workspace,
                    device_config=config,
                )

            view_graph = capture_unrolled(view_fn, args.unroll)
            contiguous_graph = capture_unrolled(contiguous_fn, args.unroll)
            view_ms = graph_milliseconds(
                view_graph,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
            contiguous_ms = graph_milliseconds(
                contiguous_graph,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
            for store_mode in args.stores:
                output_inline = torch.empty_like(output_contiguous)

                def inline_fn(store_mode: str = store_mode) -> torch.Tensor:
                    return sparse24_cutlass_inline_transpose_gemm_prepacked(
                        x,
                        values,
                        meta,
                        out=output_inline,
                        config=config,
                        store_mode=store_mode,
                    )

                contiguous_fn()
                inline_fn()
                torch.cuda.synchronize()
                max_abs_diff = float(
                    (output_contiguous.float() - output_inline.float())
                    .abs()
                    .max()
                    .item()
                )
                close = bool(
                    torch.allclose(
                        output_contiguous,
                        output_inline,
                        rtol=args.rtol,
                        atol=args.atol,
                    )
                )
                if not close:
                    raise RuntimeError(
                        f"inline epilogue mismatch for M={m}, config={config}, "
                        f"store={store_mode}: max_abs_diff={max_abs_diff}"
                    )

                inline_graph = capture_unrolled(inline_fn, args.unroll)
                inline_ms = graph_milliseconds(
                    inline_graph,
                    unroll=args.unroll,
                    replays=args.replays,
                    trials=args.trials,
                )
                row = {
                    "M": m,
                    "K": args.k,
                    "N": args.n,
                    "config": config,
                    "store_mode": store_mode,
                    "view_ms": view_ms,
                    "contiguous_ms": contiguous_ms,
                    "inline_ms": inline_ms,
                    "inline_speedup_vs_contiguous": contiguous_ms / inline_ms,
                    "inline_slowdown_vs_view": inline_ms / view_ms,
                    "transpose_fraction": (contiguous_ms - view_ms)
                    / contiguous_ms,
                    "max_abs_diff": max_abs_diff,
                    "pass": close,
                }
                rows.append(row)
                print(
                    f"M={m} config={config} store={store_mode} "
                    f"view={view_ms:.4f} ms contiguous={contiguous_ms:.4f} ms "
                    f"inline={inline_ms:.4f} ms "
                    f"speedup={row['inline_speedup_vs_contiguous']:.3f}x",
                    flush=True,
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = sorted(
        {(str(row["config"]), str(row["store_mode"])) for row in rows}
    )
    for config, store_mode in labels:
        selected = sorted(
            (
                row
                for row in rows
                if row["config"] == config and row["store_mode"] == store_mode
            ),
            key=lambda row: int(row["M"]),
        )
        m_values = [int(row["M"]) for row in selected]
        speedups = [float(row["inline_speedup_vs_contiguous"]) for row in selected]
        overheads = [float(row["inline_slowdown_vs_view"]) for row in selected]
        label = f"{config}/{store_mode}"
        axes[0].plot(m_values, speedups, marker="o", label=label)
        axes[1].plot(m_values, overheads, marker="o", label=label)
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("Inline vs GEMM + transpose")
    axes[0].set_ylabel("Speedup")
    axes[1].set_title("Inline cost vs no-transpose lower bound")
    axes[1].set_ylabel("Slowdown")
    for axis in axes:
        axis.set_xlabel("Verification rows (M)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=parse_csv_ints, default=parse_csv_ints(DEFAULT_ROWS))
    parser.add_argument("--configs", type=parse_csv_strings, default=parse_csv_strings(DEFAULT_CONFIGS))
    parser.add_argument(
        "--stores",
        type=parse_csv_strings,
        default=parse_csv_strings("scalar,vector"),
    )
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    csv_path = args.output_root / "inline_epilogue_benchmark.csv"
    plot_path = args.output_root / "inline_epilogue_speedup.png"
    write_csv(csv_path, rows)
    write_plot(plot_path, rows)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
