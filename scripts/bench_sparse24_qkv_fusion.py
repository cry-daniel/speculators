#!/usr/bin/env python3
"""Benchmark sparse-layout QKV post-ops and SM120 QK fusion schedules."""

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
    sparse24_qkv_transpose_postop,
    sparse24_transpose_output_contiguous,
)


Q_SIZE = 4096
KV_SIZE = 1024
HEAD_DIM = 128
OUTPUT_SIZE = Q_SIZE + 2 * KV_SIZE
Q_HEADS = Q_SIZE // HEAD_DIM
KV_HEADS = KV_SIZE // HEAD_DIM
EPSILON = 1e-6
DEFAULT_ROWS = (112, 144, 176, 224, 288, 352, 448, 576, 704)
DEFAULT_POSTOP_CONFIGS = (
    "16x4",
    "16x8",
    "32x4",
    "32x8",
    "64x4",
    "64x8",
)


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item.strip())
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be positive comma-separated ints")
    return rows


def _parse_configs(value: str) -> tuple[str, ...]:
    configs = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in configs if item not in DEFAULT_POSTOP_CONFIGS]
    if not configs or invalid:
        raise argparse.ArgumentTypeError(
            "post-op configs must be drawn from " + ",".join(DEFAULT_POSTOP_CONFIGS)
        )
    return configs


def _median_cuda_ms(
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


def _make_inputs(rows: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(1000 + rows)
    logical = torch.randn(
        (rows, OUTPUT_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    q_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float16)
    k_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float16)
    angles = torch.randn(
        (4096, HEAD_DIM // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
    position_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
    leading_dim = ((rows + 7) // 8) * 8
    transposed = torch.empty_strided(
        logical.shape,
        (1, leading_dim),
        device="cuda",
        dtype=torch.float16,
    )
    transposed.copy_(logical)
    return logical, transposed, q_weight, k_weight, cos_sin_cache, position_ids


def _qwen_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    token_heads_per_warp: int,
) -> torch.Tensor:
    vllm_ops.fused_qk_norm_rope(
        qkv,
        Q_HEADS,
        KV_HEADS,
        KV_HEADS,
        HEAD_DIM,
        EPSILON,
        q_weight,
        k_weight,
        cos_sin_cache,
        True,
        position_ids,
        token_heads_per_warp,
    )
    return qkv


def _llama_rope(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    q, k, _v = qkv.split((Q_SIZE, KV_SIZE, KV_SIZE), dim=-1)
    vllm_ops.rotary_embedding(
        position_ids,
        q,
        k,
        HEAD_DIM,
        cos_sin_cache,
        True,
    )
    return qkv


def _run_case(
    rows: int,
    *,
    postop_configs: tuple[str, ...],
    warmup: int,
    repeat: int,
    trials: int,
) -> list[dict[str, float | int | str]]:
    logical, transposed, q_weight, k_weight, cache, positions = _make_inputs(rows)
    results: list[dict[str, float | int | str]] = []
    qwen_baseline_out = torch.empty_like(logical)
    qwen_fused_out = {
        config: torch.empty_like(logical) for config in postop_configs
    }

    def qwen_baseline() -> torch.Tensor:
        qkv = sparse24_transpose_output_contiguous(
            transposed, out=qwen_baseline_out
        )
        return _qwen_norm_rope(
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            token_heads_per_warp=-1,
        )

    def qwen_fused(config: str) -> torch.Tensor:
        return sparse24_qkv_transpose_postop(
            transposed,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            epsilon=EPSILON,
            is_neox=True,
            q_weight=q_weight,
            k_weight=k_weight,
            out=qwen_fused_out[config],
            postop_config=config,
        )

    qwen_expected = qwen_baseline()
    qwen_baseline_ms = _median_cuda_ms(
        qwen_baseline, warmup=warmup, repeat=repeat, trials=trials
    )
    for config in postop_configs:
        qwen_actual = qwen_fused(config)
        qwen_diff = float((qwen_actual - qwen_expected).abs().max().item())
        qwen_fused_ms = _median_cuda_ms(
            lambda config=config: qwen_fused(config),
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        results.append(
            {
                "rows": rows,
                "case": "qwen_transposed_postop",
                "variant": config,
                "baseline_ms": qwen_baseline_ms,
                "variant_ms": qwen_fused_ms,
                "speedup": qwen_baseline_ms / qwen_fused_ms,
                "max_abs_diff": qwen_diff,
            }
        )

    llama_baseline_out = torch.empty_like(logical)
    llama_fused_out = {
        config: torch.empty_like(logical) for config in postop_configs
    }

    def llama_baseline() -> torch.Tensor:
        return _llama_rope(
            sparse24_transpose_output_contiguous(
                transposed, out=llama_baseline_out
            ),
            cache,
            positions,
        )

    def llama_fused(config: str) -> torch.Tensor:
        return sparse24_qkv_transpose_postop(
            transposed,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            epsilon=0.0,
            is_neox=True,
            out=llama_fused_out[config],
            postop_config=config,
        )

    llama_expected = llama_baseline()
    llama_baseline_ms = _median_cuda_ms(
        llama_baseline, warmup=warmup, repeat=repeat, trials=trials
    )
    for config in postop_configs:
        llama_actual = llama_fused(config)
        llama_diff = float((llama_actual - llama_expected).abs().max().item())
        llama_fused_ms = _median_cuda_ms(
            lambda config=config: llama_fused(config),
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        results.append(
            {
                "rows": rows,
                "case": "llama_transposed_postop",
                "variant": config,
                "baseline_ms": llama_baseline_ms,
                "variant_ms": llama_fused_ms,
                "speedup": llama_baseline_ms / llama_fused_ms,
                "max_abs_diff": llama_diff,
            }
        )

    schedule_inputs = {
        value: logical.clone() for value in (-1, 1, 2, 4, 8)
    }

    def run_schedule(value: int) -> torch.Tensor:
        return _qwen_norm_rope(
            schedule_inputs[value],
            q_weight,
            k_weight,
            cache,
            positions,
            token_heads_per_warp=value,
        )

    schedule_expected = logical.clone()
    _qwen_norm_rope(
        schedule_expected,
        q_weight,
        k_weight,
        cache,
        positions,
        token_heads_per_warp=1,
    )
    schedule_ms: dict[int, float] = {}
    schedule_diff: dict[int, float] = {}
    for value in (-1, 1, 2, 4, 8):
        correctness_input = logical.clone()
        _qwen_norm_rope(
            correctness_input,
            q_weight,
            k_weight,
            cache,
            positions,
            token_heads_per_warp=value,
        )
        schedule_diff[value] = float(
            (correctness_input - schedule_expected).abs().max().item()
        )
        schedule_ms[value] = _median_cuda_ms(
            lambda value=value: run_schedule(value),
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
    for value in (1, 2, 4, 8):
        results.append(
            {
                "rows": rows,
                "case": "qwen_contiguous_schedule",
                "variant": f"heads_per_warp_{value}",
                "baseline_ms": schedule_ms[-1],
                "variant_ms": schedule_ms[value],
                "speedup": schedule_ms[-1] / schedule_ms[value],
                "max_abs_diff": schedule_diff[value],
            }
        )
    return results


def _plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.1), sharey=False)
    for axis, case, title in (
        (axes[0], "qwen_transposed_postop", "Qwen norm + RoPE"),
        (axes[1], "llama_transposed_postop", "Llama RoPE"),
    ):
        variants = sorted({str(row["variant"]) for row in rows if row["case"] == case})
        colors = plt.get_cmap("tab10").colors
        for index, variant in enumerate(variants):
            selected = [
                row
                for row in rows
                if row["case"] == case and row["variant"] == variant
            ]
            axis.plot(
                [int(row["rows"]) for row in selected],
                [float(row["speedup"]) for row in selected],
                marker="o",
                label=variant,
                color=colors[index % len(colors)],
            )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Verifier rows (bs x (K+1))")
        axis.set_ylabel("Speedup")
        axis.grid(alpha=0.25)
        axis.legend(title="Rows x lanes", frameon=False, ncol=2)

    schedule_colors = ("#6C5B7B", "#2A9D8F", "#E09F3E", "#9E2A2B")
    for value, color in zip((1, 2, 4, 8), schedule_colors, strict=True):
        variant = f"heads_per_warp_{value}"
        selected = [
            row
            for row in rows
            if row["case"] == "qwen_contiguous_schedule"
            and row["variant"] == variant
        ]
        axes[2].plot(
            [int(row["rows"]) for row in selected],
            [float(row["speedup"]) for row in selected],
            marker="o",
            label=str(value),
            color=color,
        )
    axes[2].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[2].set_title("SM120 contiguous QK schedule")
    axes[2].set_xlabel("Verifier rows (bs x (K+1))")
    axes[2].set_ylabel("Speedup vs auto")
    axes[2].grid(alpha=0.25)
    axes[2].legend(title="Heads / warp", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_parse_rows, default=DEFAULT_ROWS)
    parser.add_argument(
        "--postop-configs",
        type=_parse_configs,
        default=DEFAULT_POSTOP_CONFIGS,
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_qkv_fusion_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int | str]] = []
    for rows in args.rows:
        results.extend(
            _run_case(
                rows,
                postop_configs=args.postop_configs,
                warmup=args.warmup,
                repeat=args.repeat,
                trials=args.trials,
            )
        )
    csv_path = args.output_root / "qkv_fusion_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, args.output_root / "qkv_fusion_speedup.png")
    print(csv_path)
    for row in results:
        print(
            f"M={int(row['rows']):>3} {str(row['case']):<28} "
            f"{str(row['variant']):<16} {float(row['baseline_ms']):.4f}->"
            f"{float(row['variant_ms']):.4f} ms "
            f"speedup={float(row['speedup']):.3f}x "
            f"diff={float(row['max_abs_diff']):.5f}"
        )


if __name__ == "__main__":
    main()
