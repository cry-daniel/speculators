#!/usr/bin/env python3
"""Benchmark a fused residual sparse-GEMM correction/SwiGLU epilogue."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from bench_sparse24_paired_residual import (  # noqa: E402
    padded_rows,
    route_indices,
)
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    assert_24_weight,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    prepare_cutlass_sparse24_gate_up_swiglu,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_residual_correction_swiglu_prepacked,
    sparse24_routed_swiglu_correction_,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
DEFAULT_CONFIGS = (
    "256x32x64_s3_sw4",
    "256x64x64_s3_sw4",
)


def prepare_residual_weights(
    model_width: int,
    intermediate: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output_size = 2 * intermediate
    weight = torch.randn(
        (model_width, output_size),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    residual, _ = apply_random_24_mask(weight, generator=generator)
    assert_24_weight(residual)
    packed = pack_24(residual, layout="n_major")
    standard_values, standard_meta = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=model_width,
    )
    fused_values, fused_meta = prepare_cutlass_sparse24_gate_up_swiglu(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=model_width,
    )
    return standard_values, standard_meta, fused_values, fused_meta


def run_case(
    model: str,
    batch_size: int,
    k: int,
    config: str,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    dense_fraction: float,
    min_dense_per_request: int,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> dict[str, object]:
    model_width = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    output_size = 2 * intermediate
    rows = batch_size * (k + 1)
    dense_rows, _ = route_indices(
        batch_size,
        k,
        dense_fraction=dense_fraction,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
    )
    dense_count = int(dense_rows.numel())
    dense_run = padded_rows(dense_count)
    standard_values, standard_meta, fused_values, fused_meta = weights

    dense_x = torch.zeros(
        (dense_run, model_width), device="cuda", dtype=torch.float16
    )
    dense_x[:dense_count].normal_(mean=0.0, std=0.1, generator=generator)
    dense_base = torch.randn(
        (dense_count, output_size),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense_base.mul_(0.1)
    residual = torch.empty(
        (dense_run, output_size), device="cuda", dtype=torch.float16
    )
    baseline_out = torch.zeros(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    fused_out = torch.zeros_like(baseline_out)
    dual_out = torch.zeros_like(baseline_out)
    dense_hidden = torch.empty(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )

    def baseline() -> torch.Tensor:
        sparse24_cutlass_inline_transpose_gemm_prepacked(
            dense_x,
            standard_values,
            standard_meta,
            out=residual,
            config="auto",
            store_mode="vector",
        )
        return sparse24_routed_swiglu_correction_(
            dense_base,
            residual[:dense_count],
            dense_rows,
            baseline_out,
        )

    def fused() -> torch.Tensor:
        return sparse24_cutlass_residual_correction_swiglu_prepacked(
            dense_x,
            fused_values,
            fused_meta,
            dense_base,
            dense_rows,
            fused_out,
            config=config,
        )

    def fused_dual_output() -> torch.Tensor:
        return sparse24_cutlass_residual_correction_swiglu_prepacked(
            dense_x,
            fused_values,
            fused_meta,
            dense_base,
            dense_rows,
            dual_out,
            dense_hidden=dense_hidden,
            config=config,
        )

    baseline_actual = baseline().clone()
    fused_actual = fused().clone()
    dual_actual = fused_dual_output().clone()
    torch.cuda.synchronize()
    selected_baseline = baseline_actual[dense_rows.long()]
    selected_fused = fused_actual[dense_rows.long()]
    max_abs_diff = float(
        (selected_fused.float() - selected_baseline.float()).abs().max().item()
    )
    dual_max_abs_diff = float(
        (
            dual_actual[dense_rows.long()].float()
            - selected_baseline.float()
        )
        .abs()
        .max()
        .item()
    )
    compact_max_abs_diff = float(
        (dense_hidden[:dense_count].float() - selected_baseline.float())
        .abs()
        .max()
        .item()
    )
    if not torch.allclose(
        selected_fused, selected_baseline, rtol=5e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"fused correction mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={max_abs_diff}"
        )
    if not torch.allclose(
        dual_actual[dense_rows.long()],
        selected_baseline,
        rtol=5e-2,
        atol=2e-1,
    ) or not torch.allclose(
        dense_hidden[:dense_count],
        selected_baseline,
        rtol=5e-2,
        atol=2e-1,
    ):
        raise RuntimeError(
            f"dual-output correction mismatch for {model} "
            f"bs={batch_size} K={k}: full={dual_max_abs_diff}, "
            f"compact={compact_max_abs_diff}"
        )

    baseline_ms, fused_ms = paired_graph_median_ms(
        baseline,
        fused,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    fused_control_ms, fused_dual_ms = paired_graph_median_ms(
        fused,
        fused_dual_output,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": dense_count,
        "dense_fraction_actual": dense_count / rows,
        "config": config,
        "accumulator": os.getenv("SPECLINK_SPARSE24_ACCUMULATOR", "fp32"),
        "baseline_residual_correction_ms": baseline_ms,
        "fused_residual_correction_ms": fused_ms,
        "speedup_vs_baseline": baseline_ms / fused_ms,
        "fused_control_ms": fused_control_ms,
        "fused_dual_output_ms": fused_dual_ms,
        "dual_output_speedup_vs_fused": fused_control_ms / fused_dual_ms,
        "dual_output_speedup_vs_baseline": baseline_ms / fused_dual_ms,
        "max_abs_diff": max_abs_diff,
        "dual_max_abs_diff": dual_max_abs_diff,
        "compact_max_abs_diff": compact_max_abs_diff,
        "pass": True,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    configs = list(dict.fromkeys(str(row["config"]) for row in rows))
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.4 * len(models), 4.6), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        model_rows = [row for row in rows if row["model"] == model]
        points = sorted(
            {(int(row["batch_size"]), int(row["K"])) for row in model_rows}
        )
        x_values = list(range(len(points)))
        for config, color in zip(configs, colors, strict=False):
            values = [
                float(
                    next(
                        row["dual_output_speedup_vs_baseline"]
                        for row in model_rows
                        if row["config"] == config
                        and int(row["batch_size"]) == batch_size
                        and int(row["K"]) == k
                    )
                )
                for batch_size, k in points
            ]
            axis.plot(
                x_values,
                values,
                marker="o",
                linewidth=1.8,
                color=color,
                label=config,
            )
        axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_xticks(x_values)
        axis.set_xticklabels(
            [f"bs{batch_size}/K{k}" for batch_size, k in points],
            rotation=35,
            ha="right",
        )
        axis.set_title(model)
        axis.set_ylabel("Dual-output speedup vs GEMM + correction")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Residual Correction SwiGLU Epilogue\n\n")
        handle.write(
            "The baseline is a standard residual sparse GEMM followed by a "
            "separate FP16 correction/SwiGLU scatter kernel.\n\n"
        )
        handle.write(
            "| Model | bs | K | Dense rows | Config | Fused | Dual | "
            "Dual/fused | Diff |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---|---:|---:|---:|---:|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['batch_size']} | {row['K']} | "
                f"{row['dense_rows']} | {row['config']} | "
                f"{float(row['speedup_vs_baseline']):.3f}x | "
                f"{float(row['dual_output_speedup_vs_baseline']):.3f}x | "
                f"{float(row['dual_output_speedup_vs_fused']):.3f}x | "
                f"{float(row['max_abs_diff']):.6f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=parse_csv_strings,
        default=("qwen3_8b", "llama3_1_8b"),
    )
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--k-values", type=parse_csv_ints, default=DEFAULT_K_VALUES)
    parser.add_argument(
        "--configs", type=parse_csv_strings, default=DEFAULT_CONFIGS
    )
    parser.add_argument("--qwen-dense-fraction", type=float, default=0.125)
    parser.add_argument("--llama-dense-fraction", type=float, default=0.3125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument(
        "--accumulator",
        choices=("fp32", "fp16", "fp16_gate", "fp16_qkv_gate"),
        default="fp16_qkv_gate",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--graph-warmup-replays", type=int, default=30)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_models = [model for model in args.models if model not in MODELS]
    invalid_configs = [
        config for config in args.configs if config not in DEFAULT_CONFIGS
    ]
    if invalid_models or invalid_configs:
        raise ValueError(
            f"unsupported models={invalid_models}, configs={invalid_configs}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.accumulator == "fp32":
        os.environ.pop("SPECLINK_SPARSE24_ACCUMULATOR", None)
    else:
        os.environ["SPECLINK_SPARSE24_ACCUMULATOR"] = args.accumulator
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    for model in args.models:
        model_width = int(MODELS[model]["hidden"])
        intermediate = int(MODELS[model]["intermediate"])
        weights = prepare_residual_weights(
            model_width, intermediate, generator
        )
        dense_fraction = (
            args.qwen_dense_fraction
            if model == "qwen3_8b"
            else args.llama_dense_fraction
        )
        for batch_size in args.batch_sizes:
            for k in args.k_values:
                for config in args.configs:
                    result = run_case(
                        model,
                        batch_size,
                        k,
                        config,
                        weights,
                        dense_fraction=dense_fraction,
                        min_dense_per_request=args.min_dense_per_request,
                        generator=generator,
                        unroll=args.unroll,
                        replays=args.replays,
                        trials=args.trials,
                        graph_warmup_replays=args.graph_warmup_replays,
                    )
                    rows.append(result)
                    print(
                        f"{model} bs={batch_size} K={k} "
                        f"dense={result['dense_rows']}/{result['rows']} "
                        f"config={config} "
                        "dual="
                        f"{float(result['dual_output_speedup_vs_baseline']):.3f}x "
                        "dual/fused="
                        f"{float(result['dual_output_speedup_vs_fused']):.3f}x",
                        flush=True,
                    )
        del weights
        torch.cuda.empty_cache()

    csv_path = args.output_root / "residual_correction_swiglu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(args.output_root / "report.md", rows)
    write_plot(args.output_root / "residual_correction_swiglu.png", rows)
    print(args.output_root, flush=True)


if __name__ == "__main__":
    main()
