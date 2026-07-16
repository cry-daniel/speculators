#!/usr/bin/env python3
"""Benchmark routed QKV post-op fused with the FP16 KV-cache store."""

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
    sparse24_qkv_add_routed_residual_postop_cache_inplace_,
    sparse24_qkv_add_routed_residual_postop_inplace_,
)


Q_SIZE = 4096
KV_SIZE = 1024
HEAD_DIM = 128
OUTPUT_SIZE = Q_SIZE + 2 * KV_SIZE
BLOCK_SIZE = 16
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
    postop_config: str,
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> dict[str, float | int | str]:
    generator = torch.Generator(device="cuda").manual_seed(4100 + rows + dense_count)
    normalize_qk = model == "qwen3_8b"
    initial_qkv = torch.randn(
        (rows, OUTPUT_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ) * 1e-2
    correction = torch.randn(
        (dense_count, OUTPUT_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ) * 1e-2
    dense_rows = torch.randperm(rows, device="cuda", generator=generator)[
        :dense_count
    ].sort().values.to(torch.int64)
    dense_slots = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    dense_slots[dense_rows] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    positions = torch.randint(
        0, 2048, (rows,), device="cuda", dtype=torch.int64, generator=generator
    )
    rope = torch.randn(
        (2048, HEAD_DIM),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    q_weight = torch.randn(
        (HEAD_DIM,), device="cuda", dtype=torch.float16, generator=generator
    )
    k_weight = torch.randn(
        (HEAD_DIM,), device="cuda", dtype=torch.float16, generator=generator
    )
    num_blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_shape = (num_blocks, BLOCK_SIZE, KV_SIZE // HEAD_DIM, HEAD_DIM)
    slot_mapping = torch.randperm(
        num_blocks * BLOCK_SIZE, device="cuda", generator=generator
    )[:rows].to(torch.int64)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    baseline_qkv = initial_qkv.clone()
    baseline_key_cache = torch.zeros(
        cache_shape, device="cuda", dtype=torch.float16
    )
    baseline_value_cache = torch.zeros_like(baseline_key_cache)

    def baseline() -> torch.Tensor:
        sparse24_qkv_add_routed_residual_postop_inplace_(
            baseline_qkv,
            correction,
            dense_slots,
            rope,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            epsilon=EPSILON,
            is_neox=True,
            q_weight=q_weight if normalize_qk else None,
            k_weight=k_weight if normalize_qk else None,
            postop_config="vec8",
        )
        _, key, value = baseline_qkv.split((Q_SIZE, KV_SIZE, KV_SIZE), dim=-1)
        vllm_ops.reshape_and_cache_flash(
            key.view(rows, KV_SIZE // HEAD_DIM, HEAD_DIM),
            value.view(rows, KV_SIZE // HEAD_DIM, HEAD_DIM),
            baseline_key_cache,
            baseline_value_cache,
            slot_mapping,
            "auto",
            scale,
            scale,
        )
        return baseline_qkv

    fused_qkv = initial_qkv.clone()
    fused_key_cache = torch.zeros_like(baseline_key_cache)
    fused_value_cache = torch.zeros_like(baseline_value_cache)

    def fused() -> torch.Tensor:
        return sparse24_qkv_add_routed_residual_postop_cache_inplace_(
            fused_qkv,
            correction,
            dense_slots,
            rope,
            positions,
            slot_mapping,
            fused_key_cache,
            fused_value_cache,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            epsilon=EPSILON,
            is_neox=True,
            q_weight=q_weight if normalize_qk else None,
            k_weight=k_weight if normalize_qk else None,
            postop_config=postop_config,
        )

    expected = baseline().clone()
    actual = fused().clone()
    torch.cuda.synchronize()
    q_diff = float((actual[:, :Q_SIZE] - expected[:, :Q_SIZE]).abs().max().item())
    key_diff = float((fused_key_cache - baseline_key_cache).abs().max().item())
    value_diff = float(
        (fused_value_cache - baseline_value_cache).abs().max().item()
    )
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
        "postop_config": postop_config,
        "baseline_ms": baseline_ms,
        "fused_ms": fused_ms,
        "speedup": baseline_ms / fused_ms,
        "max_q_abs_diff": q_diff,
        "max_key_cache_abs_diff": key_diff,
        "max_value_cache_abs_diff": value_diff,
    }


def _plot(results: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    configs = list(dict.fromkeys(str(row["postop_config"]) for row in results))
    line_styles = ("-", "--", ":", "-.")
    for model, color in colors.items():
        for config, line_style in zip(configs, line_styles, strict=False):
            selected = [
                row
                for row in results
                if row["model"] == model and row["postop_config"] == config
            ]
            axes[0].plot(
                [int(row["rows"]) for row in selected],
                [float(row["fused_ms"]) for row in selected],
                marker="s",
                linestyle=line_style,
                color=color,
                label=f"{model} {config}",
            )
            axes[1].plot(
                [int(row["rows"]) for row in selected],
                [float(row["speedup"]) for row in selected],
                marker="o",
                linestyle=line_style,
                color=color,
                label=f"{model} {config}",
            )
    axes[0].set_title("QKV post-op + FP16 KV-cache store")
    axes[0].set_xlabel("Verifier rows")
    axes[0].set_ylabel("Fused CUDA-graph latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_title("Direct-cache epilogue speedup")
    axes[1].set_xlabel("Verifier rows")
    axes[1].set_ylabel("Separate / fused")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--postop-configs", default="vec8")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    invalid = [model for model in models if model not in DENSE_COUNTS]
    if not models or invalid:
        parser.error(f"unsupported models: {invalid}")
    configs = tuple(
        item.strip() for item in args.postop_configs.split(",") if item.strip()
    )
    invalid_configs = [
        config
        for config in configs
        if config not in {"vec8", "vec16", "vec32", "vec64"}
    ]
    if not configs or invalid_configs:
        parser.error(f"unsupported post-op configs: {invalid_configs}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_root = args.output_root or (
        REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_qkv_cache_epilogue_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(
            model,
            rows,
            dense_count,
            postop_config,
            warmup=args.warmup,
            repeat=args.repeat,
            trials=args.trials,
        )
        for model in models
        for postop_config in configs
        for rows, dense_count in zip(ROWS, DENSE_COUNTS[model], strict=True)
    ]
    csv_path = output_root / "qkv_cache_epilogue.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, output_root / "qkv_cache_epilogue.png")
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
