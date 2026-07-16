#!/usr/bin/env python3
"""Benchmark a split indexed-sparse/dual-sparse exact routed SwiGLU stage."""

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
    sparse24_cutlass_dual_swiglu_prepacked,
    sparse24_cutlass_indexed_swiglu_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_,
    sparse24_routed_swiglu_correction_,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
DEFAULT_CONFIGS = (
    "256x32x64_s3_sw4",
    "256x64x64_s3_sw4",
)
PackedWeights = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def prepare_weights(
    model_width: int,
    intermediate: int,
    generator: torch.Generator,
) -> PackedWeights:
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
    residual24 = weight - weight24
    assert_24_weight(weight24)
    assert_24_weight(residual24)

    def pack_interleaved(
        sparse_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_24(sparse_weight, layout="n_major")
        return prepare_cutlass_sparse24_gate_up_swiglu(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=model_width,
        )

    full_values, full_meta = pack_interleaved(weight24)
    residual_interleaved_values, residual_interleaved_meta = (
        pack_interleaved(residual24)
    )
    residual_packed = pack_24(residual24, layout="n_major")
    residual_values, residual_meta = prepare_cutlass_sparse24_device_gemm(
        residual_packed.values,
        residual_packed.meta,
        layout=residual_packed.layout,
        K=model_width,
    )
    return (
        full_values,
        full_meta,
        residual_values,
        residual_meta,
        residual_interleaved_values,
        residual_interleaved_meta,
    )


def run_case(
    model: str,
    batch_size: int,
    k: int,
    config: str,
    weights: PackedWeights,
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
    dense_rows, sparse_rows = route_indices(
        batch_size,
        k,
        dense_fraction=dense_fraction,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
    )
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    dense_run = padded_rows(dense_count)
    sparse_run = padded_rows(sparse_count)
    dense_slots = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    dense_slots[dense_rows.long()] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    (
        full_values,
        full_meta,
        residual_values,
        residual_meta,
        residual_interleaved_values,
        residual_interleaved_meta,
    ) = weights

    x = torch.randn(
        (rows, model_width),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)
    dense_x = torch.zeros(
        (dense_run, model_width), device="cuda", dtype=torch.float16
    )
    sparse_x = torch.zeros(
        (sparse_run, model_width), device="cuda", dtype=torch.float16
    )

    baseline_out = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    dense_base = torch.empty(
        (dense_count, output_size), device="cuda", dtype=torch.float16
    )
    gate_residual = torch.empty(
        (dense_run, output_size), device="cuda", dtype=torch.float16
    )
    split_out = torch.empty_like(baseline_out)

    baseline_full_stream = torch.cuda.Stream()
    baseline_residual_stream = torch.cuda.Stream()
    split_sparse_stream = torch.cuda.Stream()
    split_dense_stream = torch.cuda.Stream()

    def baseline() -> torch.Tensor:
        current = torch.cuda.current_stream()
        baseline_full_stream.wait_stream(current)
        baseline_residual_stream.wait_stream(current)
        with torch.cuda.stream(baseline_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                full_values,
                full_meta,
                dense_slots,
                dense_count=dense_count,
                out=baseline_out,
                dense_base=dense_base,
                config="256x64x64_s3_sw4",
            )
        with torch.cuda.stream(baseline_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                dense_x,
                residual_values,
                residual_meta,
                out=gate_residual,
                config="auto",
                store_mode="vector",
            )
        current.wait_stream(baseline_full_stream)
        current.wait_stream(baseline_residual_stream)
        return sparse24_routed_swiglu_correction_(
            dense_base,
            gate_residual[:dense_count],
            dense_rows,
            baseline_out,
        )

    def split_dual() -> torch.Tensor:
        current = torch.cuda.current_stream()
        split_sparse_stream.wait_stream(current)
        split_dense_stream.wait_stream(current)
        with torch.cuda.stream(split_sparse_stream):
            sparse24_gather_rows_(x, sparse_rows, sparse_x[:sparse_count])
            sparse24_cutlass_indexed_swiglu_prepacked(
                sparse_x,
                full_values,
                full_meta,
                sparse_rows,
                split_out,
                config=config,
            )
        with torch.cuda.stream(split_dense_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_dual_swiglu_prepacked(
                dense_x,
                full_values,
                full_meta,
                residual_interleaved_values,
                residual_interleaved_meta,
                dense_rows,
                split_out,
                config=config,
            )
        current.wait_stream(split_sparse_stream)
        current.wait_stream(split_dense_stream)
        return split_out

    baseline_actual = baseline().clone()
    split_actual = split_dual().clone()
    torch.cuda.synchronize()
    max_abs_diff = float(
        (split_actual.float() - baseline_actual.float()).abs().max().item()
    )
    if not torch.allclose(
        split_actual, baseline_actual, rtol=5e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"split dual mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={max_abs_diff}"
        )

    baseline_ms, split_ms = paired_graph_median_ms(
        baseline,
        split_dual,
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
        "sparse_rows": sparse_count,
        "dense_fraction_actual": dense_count / rows,
        "config": config,
        "accumulator": os.getenv("SPECLINK_SPARSE24_ACCUMULATOR", "fp32"),
        "baseline_gate_ms": baseline_ms,
        "split_dual_gate_ms": split_ms,
        "speedup_vs_baseline": baseline_ms / split_ms,
        "max_abs_diff": max_abs_diff,
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
                        row["speedup_vs_baseline"]
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
        axis.set_ylabel("Exact gate-stage speedup vs full + residual")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Dual Sparse Routed SwiGLU\n\n")
        handle.write(
            "The candidate runs indexed W24 for sparse rows and sequential "
            "W24/R24 accumulation inside one CTA for dense rows.\n\n"
        )
        handle.write("| Model | bs | K | Dense | Config | Speedup | Diff |\n")
        handle.write("|---|---:|---:|---:|---|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['batch_size']} | {row['K']} | "
                f"{row['dense_rows']}/{row['rows']} | {row['config']} | "
                f"{float(row['speedup_vs_baseline']):.3f}x | "
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
    if args.accumulator == "fp32":
        os.environ.pop("SPECLINK_SPARSE24_ACCUMULATOR", None)
    else:
        os.environ["SPECLINK_SPARSE24_ACCUMULATOR"] = args.accumulator
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results: list[dict[str, object]] = []
    for model in args.models:
        model_width = int(MODELS[model]["hidden"])
        intermediate = int(MODELS[model]["intermediate"])
        weights = prepare_weights(model_width, intermediate, generator)
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
                    results.append(result)
                    print(
                        f"{model} bs={batch_size} K={k} "
                        f"dense={result['dense_rows']}/{result['rows']} "
                        f"config={config} "
                        f"speedup={float(result['speedup_vs_baseline']):.3f}x",
                        flush=True,
                    )
        del weights
        torch.cuda.empty_cache()

    csv_path = args.output_root / "dual_routed_swiglu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_report(args.output_root / "report.md", results)
    write_plot(args.output_root / "dual_routed_swiglu.png", results)
    print(args.output_root, flush=True)


if __name__ == "__main__":
    main()
