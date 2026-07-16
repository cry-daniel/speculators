#!/usr/bin/env python3
"""Benchmark a whole-batch W24 Gate/SwiGLU -> W24 Down pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_gate_dense_down_pipeline import (  # noqa: E402
    MODELS,
    prepare_gate_up_layouts,
)
from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_cutlass_gate_sparse_down_pipeline_prepacked,
)


def prepare_down(
    model_width: int,
    intermediate: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (intermediate, model_width),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    return prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=intermediate,
    )


def run_case(
    model: str,
    rows: int,
    gate_values: torch.Tensor,
    gate_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    hidden_size = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    x = torch.randn(
        (rows, hidden_size),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.1)
    hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    output = torch.empty(
        (rows, hidden_size), device="cuda", dtype=torch.float16
    )
    down_row_tile = 32 if args.config.startswith("256x32") else 64
    counters = torch.zeros(
        (rows + down_row_tile - 1) // down_row_tile,
        device="cuda",
        dtype=torch.int32,
    )

    def launch(stage: str) -> torch.Tensor:
        sparse24_cutlass_gate_sparse_down_pipeline_prepacked(
            x,
            gate_values,
            gate_meta,
            down_values,
            down_meta,
            hidden=hidden,
            out=output,
            row_counters=counters,
            config=args.config,
            worker_blocks=args.worker_blocks,
            stage=stage,
        )
        return hidden if stage == "gate_only" else output

    def separate() -> torch.Tensor:
        launch("gate_only")
        return launch("down_only")

    def pipeline() -> torch.Tensor:
        return launch("full")

    expected = separate().clone()
    actual = pipeline().clone()
    torch.cuda.synchronize()
    max_abs_diff = float(
        (actual.float() - expected.float()).abs().max().item()
    )
    mean_abs_diff = float(
        (actual.float() - expected.float()).abs().mean().item()
    )
    counter_max = int(counters.abs().max().item())
    if max_abs_diff > args.output_atol:
        raise RuntimeError(
            f"sparse pipeline mismatch for {model} M={rows}: "
            f"{max_abs_diff:.6f}"
        )
    if counter_max != 0:
        raise RuntimeError(
            f"sparse pipeline counters did not reset for {model} M={rows}: "
            f"{counter_max}"
        )

    separate_ms, pipeline_ms = paired_graph_median_ms(
        separate,
        pipeline,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    gate_ms, _ = paired_graph_median_ms(
        lambda: launch("gate_only"),
        lambda: launch("gate_only"),
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    down_ms, _ = paired_graph_median_ms(
        lambda: launch("down_only"),
        lambda: launch("down_only"),
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    counter_max_after_graph = int(counters.abs().max().item())
    if counter_max_after_graph != 0:
        raise RuntimeError(
            f"sparse pipeline counters did not reset after graph replay for "
            f"{model} M={rows}: {counter_max_after_graph}"
        )
    return {
        "model": model,
        "rows": rows,
        "config": args.config,
        "worker_blocks": args.worker_blocks,
        "separate_gate_sparse_down_ms": separate_ms,
        "pipeline_gate_sparse_down_ms": pipeline_ms,
        "pipeline_speedup": separate_ms / pipeline_ms,
        "gate_only_ms": gate_ms,
        "down_only_ms": down_ms,
        "stage_sum_ms": gate_ms + down_ms,
        "pipeline_vs_stage_sum_speedup": (gate_ms + down_ms) / pipeline_ms,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "counter_max": counter_max,
        "counter_max_after_graph": counter_max_after_graph,
        "scope": "whole_batch_w24_kernel_upper_bound",
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        x = [int(row["rows"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["separate_gate_sparse_down_ms"]) for row in selected],
            marker="o",
            linestyle="--",
            color=colors[model],
            label=f"{model} separate",
        )
        axes[0].plot(
            x,
            [float(row["pipeline_gate_sparse_down_ms"]) for row in selected],
            marker="s",
            color=colors[model],
            label=f"{model} pipeline",
        )
        axes[1].plot(
            x,
            [float(row["pipeline_speedup"]) for row in selected],
            marker="o",
            color=colors[model],
            label=model,
        )
    axes[0].set_title("Whole-batch W24 MLP kernel upper bound")
    axes[0].set_xlabel("Verifier rows")
    axes[0].set_ylabel("CUDA Graph latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_title("Single-launch pipeline speedup")
    axes[1].set_xlabel("Verifier rows")
    axes[1].set_ylabel("Separate / pipeline")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Gate/SwiGLU -> sparse Down pipeline upper bound",
        "",
        "This is a whole-batch W24 kernel ablation, not a final token-mixed result.",
        "",
        "| Model | Rows | Separate (ms) | Pipeline (ms) | Speedup | Max diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['rows']} | "
            f"{float(row['separate_gate_sparse_down_ms']):.4f} | "
            f"{float(row['pipeline_gate_sparse_down_ms']):.4f} | "
            f"{float(row['pipeline_speedup']):.3f}x | "
            f"{float(row['max_abs_diff']):.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_csv_strings, default=tuple(MODELS))
    parser.add_argument("--rows", type=parse_csv_ints, default=(112, 144, 176))
    parser.add_argument("--worker-blocks", type=int, default=0)
    parser.add_argument(
        "--config",
        choices=(
            "256x64_gate_256x64_sparse_down",
            "128x64_gate_128x64_sparse_down",
            "256x32_gate_256x32_sparse_down",
            "256x64_gate_256x64_sparse_down_dynamic_owners",
            "128x64_gate_128x64_sparse_down_dynamic_owners",
            "256x32_gate_256x32_sparse_down_dynamic_owners",
            "128x64_gate_128x64_sparse_down_grid_barrier",
        ),
        default="256x64_gate_256x64_sparse_down",
    )
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=3)
    parser.add_argument("--output-atol", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_models = [model for model in args.models if model not in MODELS]
    if invalid_models:
        raise ValueError(f"unsupported models: {invalid_models}")
    if args.worker_blocks < 0 or args.worker_blocks == 1:
        raise ValueError("--worker-blocks must be zero or at least two")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results: list[dict[str, object]] = []
    for model in args.models:
        hidden_size = int(MODELS[model]["hidden"])
        intermediate = int(MODELS[model]["intermediate"])
        gate_layouts = prepare_gate_up_layouts(
            hidden_size, intermediate, generator
        )
        channels = 64 if args.config.startswith("128x64") else 128
        gate_values, gate_meta = gate_layouts[channels]
        down_values, down_meta = prepare_down(
            hidden_size, intermediate, generator
        )
        for rows in args.rows:
            result = run_case(
                model,
                rows,
                gate_values,
                gate_meta,
                down_values,
                down_meta,
                generator,
                args,
            )
            results.append(result)
            print(result, flush=True)
        del gate_layouts, down_values, down_meta
        torch.cuda.empty_cache()

    csv_path = args.output_root / "gate_sparse_down_pipeline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "gate_sparse_down_pipeline.png", results)
    write_report(args.output_root / "report.md", results)
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
