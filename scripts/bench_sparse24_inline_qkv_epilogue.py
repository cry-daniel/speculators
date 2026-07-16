#!/usr/bin/env python3
"""Benchmark sparse QKV GEMM with inline Q/K normalization and RoPE."""

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

from vllm import _custom_ops as vllm_ops  # noqa: E402
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_qkv_postop_prepacked,
    sparse24_qkv_transpose_postop,
    sparse24_transpose_output_contiguous,
)


MODEL_WIDTH = 4096
Q_SIZE = 4096
KV_SIZE = 1024
HEAD_DIM = 128
OUTPUT_SIZE = Q_SIZE + 2 * KV_SIZE
DEFAULT_ROWS = "112,144,176,224,288,352,448,576,704"
DEFAULT_INLINE_CONFIGS = "auto,128x32x64_s4,128x32x64_s4_sw2,128x32x64_s4_sw4"


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


def prepare_weight(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (MODEL_WIDTH, OUTPUT_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    return prepare_cutlass_sparse24_device_gemm(
        packed.values, packed.meta, layout=packed.layout, K=MODEL_WIDTH
    )


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    values, meta = prepare_weight(generator)
    if args.model == "qwen3_8b":
        q_weight = 1.0 + 0.05 * torch.randn(
            HEAD_DIM,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        k_weight = 1.0 + 0.05 * torch.randn(
            HEAD_DIM,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        normalize_qk = True
        epsilon = 1e-6
    else:
        q_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float16)
        k_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float16)
        normalize_qk = False
        epsilon = 0.0
    angles = torch.randn(
        (4096, HEAD_DIM // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()

    results: list[dict[str, object]] = []
    for rows in args.rows:
        x = torch.randn(
            (rows, MODEL_WIDTH),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        positions = torch.arange(rows, device="cuda", dtype=torch.int64)
        transposed = torch.empty_strided(
            (rows, OUTPUT_SIZE),
            (1, rows),
            device="cuda",
            dtype=torch.float16,
        )
        baseline_output = torch.empty(
            (rows, OUTPUT_SIZE), device="cuda", dtype=torch.float16
        )
        separate_fused_output = torch.empty_like(baseline_output)
        inline_outputs = {
            config: torch.empty_like(baseline_output)
            for config in args.inline_configs
        }
        baseline_q = baseline_output[:, :Q_SIZE]
        baseline_k = baseline_output[:, Q_SIZE : Q_SIZE + KV_SIZE]

        def gemm() -> torch.Tensor:
            return sparse24_cutlass_device_gemm_prepacked(
                x, values, meta, out=transposed, device_config="auto"
            )

        def baseline_fn() -> torch.Tensor:
            gemm()
            sparse24_transpose_output_contiguous(
                transposed, out=baseline_output
            )
            if normalize_qk:
                vllm_ops.fused_qk_norm_rope(
                    baseline_output,
                    Q_SIZE // HEAD_DIM,
                    KV_SIZE // HEAD_DIM,
                    KV_SIZE // HEAD_DIM,
                    HEAD_DIM,
                    epsilon,
                    q_weight,
                    k_weight,
                    cache,
                    True,
                    positions,
                    -1,
                )
            else:
                vllm_ops.rotary_embedding(
                    positions,
                    baseline_q,
                    baseline_k,
                    HEAD_DIM,
                    cache,
                    True,
                )
            return baseline_output

        def separate_fused_fn() -> torch.Tensor:
            gemm()
            return sparse24_qkv_transpose_postop(
                transposed,
                cache,
                positions,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                head_dim=HEAD_DIM,
                epsilon=epsilon,
                is_neox=True,
                q_weight=q_weight if normalize_qk else None,
                k_weight=k_weight if normalize_qk else None,
                out=separate_fused_output,
                postop_config="16x4",
            )

        def inline_fn(config: str) -> torch.Tensor:
            return sparse24_cutlass_qkv_postop_prepacked(
                x,
                values,
                meta,
                q_weight,
                k_weight,
                cache,
                positions,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                epsilon=epsilon,
                normalize_qk=normalize_qk,
                out=inline_outputs[config],
                config=config,
            )

        baseline_fn()
        separate_fused_fn()
        for config in args.inline_configs:
            inline_fn(config)
        torch.cuda.synchronize()
        separate_diff = float(
            (baseline_output.float() - separate_fused_output.float())
            .abs()
            .max()
            .item()
        )
        if not torch.allclose(
            baseline_output, separate_fused_output, rtol=args.rtol, atol=args.atol
        ):
            raise RuntimeError(
                f"separate QKV post-op mismatch M={rows}: {separate_diff}"
            )

        baseline_graph = capture_unrolled(baseline_fn, args.unroll)
        separate_graph = capture_unrolled(separate_fused_fn, args.unroll)
        baseline_ms = graph_milliseconds(
            baseline_graph,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
        )
        separate_ms = graph_milliseconds(
            separate_graph,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
        )
        for config in args.inline_configs:
            inline_output = inline_outputs[config]
            inline_diff = float(
                (baseline_output.float() - inline_output.float())
                .abs()
                .max()
                .item()
            )
            if not torch.allclose(
                baseline_output, inline_output, rtol=args.rtol, atol=args.atol
            ):
                raise RuntimeError(
                    f"inline QKV mismatch M={rows}, config={config}: "
                    f"{inline_diff}"
                )
            inline_graph = capture_unrolled(
                lambda config=config: inline_fn(config), args.unroll
            )
            inline_ms = graph_milliseconds(
                inline_graph,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
            result = {
                "model": args.model,
                "M": rows,
                "inline_config": config,
                "baseline_ms": baseline_ms,
                "separate_fused_postop_ms": separate_ms,
                "inline_epilogue_ms": inline_ms,
                "separate_speedup": baseline_ms / separate_ms,
                "inline_speedup": baseline_ms / inline_ms,
                "inline_vs_separate_speedup": separate_ms / inline_ms,
                "separate_max_abs_diff": separate_diff,
                "inline_max_abs_diff": inline_diff,
            }
            results.append(result)
            print(
                f"{args.model} M={rows} config={config} "
                f"baseline={baseline_ms:.4f} ms separate={separate_ms:.4f} ms "
                f"inline={inline_ms:.4f} ms "
                f"inline_speedup={result['inline_speedup']:.3f}x "
                f"inline_vs_separate="
                f"{result['inline_vs_separate_speedup']:.3f}x",
                flush=True,
            )
    return results


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    first_config = str(rows[0]["inline_config"])
    baseline_rows = [
        row for row in rows if str(row["inline_config"]) == first_config
    ]
    axis.plot(
        [int(row["M"]) for row in baseline_rows],
        [float(row["separate_speedup"]) for row in baseline_rows],
        marker="o",
        label="Separate fused post-op",
    )
    for config in sorted({str(row["inline_config"]) for row in rows}):
        selected = sorted(
            (row for row in rows if str(row["inline_config"]) == config),
            key=lambda row: int(row["M"]),
        )
        axis.plot(
            [int(row["M"]) for row in selected],
            [float(row["inline_speedup"]) for row in selected],
            marker="s",
            label=f"Inline {config}",
        )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Verification rows (M)")
    axis.set_ylabel("Speedup vs standard QKV path")
    axis.set_title(f"{rows[0]['model']} sparse QKV epilogue")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("qwen3_8b", "llama3_1_8b"), default="qwen3_8b"
    )
    parser.add_argument(
        "--rows", type=parse_csv_ints, default=parse_csv_ints(DEFAULT_ROWS)
    )
    parser.add_argument(
        "--inline-configs",
        type=parse_csv_strings,
        default=parse_csv_strings(DEFAULT_INLINE_CONFIGS),
    )
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
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "inline_qkv_epilogue_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = args.output_root / "inline_qkv_epilogue_speedup.png"
    write_plot(plot_path, rows)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
