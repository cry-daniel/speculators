#!/usr/bin/env python3
"""Benchmark a persistent W24 Gate/SwiGLU -> dense Down pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_gate_up_swiglu,
    sparse24_cutlass_gate_dense_down_pipeline_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
)


MODELS = {
    "qwen3_8b": {"hidden": 4096, "intermediate": 12288},
    "llama3_1_8b": {"hidden": 4096, "intermediate": 14336},
}


def prepare_gate_up_layouts(
    model_width: int,
    intermediate: int,
    generator: torch.Generator,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    output_size = 2 * intermediate
    weight = torch.randn(
        (model_width, output_size),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    layouts = {}
    for channels in (64, 128):
        layouts[channels] = prepare_cutlass_sparse24_gate_up_swiglu(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=model_width,
            channels_per_half_tile=channels,
        )
    return layouts


def run_case(
    model: str,
    rows: int,
    baseline_gate_values: torch.Tensor,
    baseline_gate_meta: torch.Tensor,
    pipeline_gate_values: torch.Tensor,
    pipeline_gate_meta: torch.Tensor,
    down_weight: torch.Tensor,
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
    baseline_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    baseline_out = torch.empty(
        (rows, hidden_size), device="cuda", dtype=torch.float16
    )
    pipeline_hidden = torch.empty_like(baseline_hidden)
    pipeline_out = torch.empty_like(baseline_out)
    down_row_tile = 64 if args.config.endswith("64x64_down") else 128
    row_counters = torch.zeros(
        (rows + down_row_tile - 1) // down_row_tile,
        device="cuda",
        dtype=torch.int32,
    )

    def baseline_gate() -> torch.Tensor:
        sparse24_cutlass_gate_up_swiglu_prepacked(
            x,
            baseline_gate_values,
            baseline_gate_meta,
            out=baseline_hidden,
            config="256x64x64_s3_sw4",
        )
        return baseline_hidden

    def baseline_down() -> torch.Tensor:
        return torch.mm(
            baseline_hidden, down_weight.t(), out=baseline_out
        )

    def baseline() -> torch.Tensor:
        baseline_gate()
        return baseline_down()

    def run_pipeline(stage: str) -> torch.Tensor:
        sparse24_cutlass_gate_dense_down_pipeline_prepacked(
            x,
            pipeline_gate_values,
            pipeline_gate_meta,
            down_weight,
            hidden=pipeline_hidden,
            out=pipeline_out,
            row_counters=row_counters,
            config=args.config,
            worker_blocks=args.worker_blocks,
            stage=stage,
        )
        return pipeline_hidden if stage == "gate_only" else pipeline_out

    def pipeline() -> torch.Tensor:
        return run_pipeline("full")

    def pipeline_gate() -> torch.Tensor:
        return run_pipeline("gate_only")

    def pipeline_down() -> torch.Tensor:
        return run_pipeline("down_only")

    baseline()
    pipeline()
    torch.cuda.synchronize()
    hidden_max_abs_diff = float(
        (baseline_hidden.float() - pipeline_hidden.float()).abs().max().item()
    )
    output_max_abs_diff = float(
        (baseline_out.float() - pipeline_out.float()).abs().max().item()
    )
    output_mean_abs_diff = float(
        (baseline_out.float() - pipeline_out.float()).abs().mean().item()
    )
    counter_max = int(row_counters.abs().max().item())
    if hidden_max_abs_diff > args.hidden_atol:
        raise RuntimeError(
            f"Gate hidden mismatch for {model} M={rows}: "
            f"{hidden_max_abs_diff:.6f}"
        )
    if output_max_abs_diff > args.output_atol:
        raise RuntimeError(
            f"Down output mismatch for {model} M={rows}: "
            f"{output_max_abs_diff:.6f}"
        )
    if counter_max != 0:
        raise RuntimeError(
            f"pipeline counters did not reset for {model} M={rows}: "
            f"{counter_max}"
        )

    baseline_ms, pipeline_ms = paired_graph_median_ms(
        baseline,
        pipeline,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    baseline_gate_ms, pipeline_gate_ms = paired_graph_median_ms(
        baseline_gate,
        pipeline_gate,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    baseline_down_ms, pipeline_down_ms = paired_graph_median_ms(
        baseline_down,
        pipeline_down,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    counter_max_after_graph = int(row_counters.abs().max().item())
    if counter_max_after_graph != 0:
        raise RuntimeError(
            f"pipeline counters did not reset after graph replay for "
            f"{model} M={rows}: {counter_max_after_graph}"
        )
    return {
        "model": model,
        "rows": rows,
        "worker_blocks": args.worker_blocks,
        "config": args.config,
        "baseline_gate_dense_down_ms": baseline_ms,
        "pipeline_gate_dense_down_ms": pipeline_ms,
        "pipeline_speedup": baseline_ms / pipeline_ms,
        "baseline_gate_ms": baseline_gate_ms,
        "pipeline_gate_only_ms": pipeline_gate_ms,
        "gate_only_speedup": baseline_gate_ms / pipeline_gate_ms,
        "baseline_down_ms": baseline_down_ms,
        "pipeline_down_only_ms": pipeline_down_ms,
        "down_only_speedup": baseline_down_ms / pipeline_down_ms,
        "hidden_max_abs_diff": hidden_max_abs_diff,
        "output_max_abs_diff": output_max_abs_diff,
        "output_mean_abs_diff": output_mean_abs_diff,
        "counter_max": counter_max,
        "counter_max_after_graph": counter_max_after_graph,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))
    colors = {"qwen3_8b": "#176B87", "llama3_1_8b": "#B33F40"}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        x = [int(row["rows"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["baseline_gate_dense_down_ms"]) for row in selected],
            marker="o",
            linestyle="--",
            color=colors[model],
            label=f"{model} separate",
        )
        axes[0].plot(
            x,
            [float(row["pipeline_gate_dense_down_ms"]) for row in selected],
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
        axes[2].plot(
            x,
            [float(row["gate_only_speedup"]) for row in selected],
            marker="o",
            color=colors[model],
            label=f"{model} Gate",
        )
        axes[2].plot(
            x,
            [float(row["down_only_speedup"]) for row in selected],
            marker="s",
            linestyle="--",
            color=colors[model],
            label=f"{model} Down",
        )
    axes[0].set_title("W24 Gate/SwiGLU -> dense Down")
    axes[0].set_xlabel("Verifier rows")
    axes[0].set_ylabel("CUDA Graph latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[1].set_title("Persistent pipeline speedup")
    axes[1].set_xlabel("Verifier rows")
    axes[1].set_ylabel("Separate / pipeline")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[2].axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    axes[2].set_title("Stage-only speedup")
    axes[2].set_xlabel("Verifier rows")
    axes[2].set_ylabel("Existing / persistent")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# W24 Gate/SwiGLU -> dense Down persistent pipeline",
        "",
        "| Model | Rows | Separate (ms) | Pipeline (ms) | Speedup | Gate-only | Down-only | Max diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['rows']} | "
            f"{float(row['baseline_gate_dense_down_ms']):.4f} | "
            f"{float(row['pipeline_gate_dense_down_ms']):.4f} | "
            f"{float(row['pipeline_speedup']):.3f}x | "
            f"{float(row['gate_only_speedup']):.3f}x | "
            f"{float(row['down_only_speedup']):.3f}x | "
            f"{float(row['output_max_abs_diff']):.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_csv_strings, default=tuple(MODELS))
    parser.add_argument("--rows", type=parse_csv_ints, default=(112, 288, 576))
    parser.add_argument("--worker-blocks", type=int, default=0)
    parser.add_argument(
        "--config",
        choices=(
            "256x64_gate_64x64_down",
            "256x64_gate_64x128_down",
            "256x64_gate_128x128_down",
            "128x64_gate_64x128_down_w32x64",
        ),
        default="256x64_gate_64x128_down",
    )
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=3)
    parser.add_argument("--hidden-atol", type=float, default=0.01)
    parser.add_argument("--output-atol", type=float, default=0.25)
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
        baseline_gate_values, baseline_gate_meta = gate_layouts[128]
        pipeline_channels = (
            64 if args.config.startswith("128x64_gate") else 128
        )
        pipeline_gate_values, pipeline_gate_meta = gate_layouts[
            pipeline_channels
        ]
        down_weight = torch.randn(
            (hidden_size, intermediate),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.02)
        for rows in args.rows:
            result = run_case(
                model,
                rows,
                baseline_gate_values,
                baseline_gate_meta,
                pipeline_gate_values,
                pipeline_gate_meta,
                down_weight,
                generator,
                args,
            )
            results.append(result)
            print(result, flush=True)
        del gate_layouts, down_weight
        torch.cuda.empty_cache()

    csv_path = args.output_root / "gate_dense_down_pipeline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "gate_dense_down_pipeline.png", results)
    write_report(args.output_root / "report.md", results)
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
