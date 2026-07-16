#!/usr/bin/env python3
"""Benchmark exact mixed-row QKV with a routed sparse output epilogue."""

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
from bench_sparse24_paired_residual import (  # noqa: E402
    padded_rows,
    route_indices,
)
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    assert_24_weight,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_routed_output_gemm_prepacked,
    sparse24_gather_rows_,
    sparse24_routed_linear_correction_,
)


MODEL_WIDTH = 4096
QKV_WIDTH = 6144
DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
DEFAULT_CONFIGS = (
    "64x64x64_s6",
    "128x32x64_s4_sw4",
    "128x64x64_s5",
    "256x64x64_s3",
    "256x64x64_s3_sw4",
)


def prepare_weights(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (MODEL_WIDTH, QKV_WIDTH),
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

    full_packed = pack_24(weight24, layout="n_major")
    full_values, full_meta = prepare_cutlass_sparse24_device_gemm(
        full_packed.values,
        full_packed.meta,
        layout=full_packed.layout,
        K=MODEL_WIDTH,
    )
    residual_packed = pack_24(residual24, layout="n_major")
    residual_values, residual_meta = prepare_cutlass_sparse24_device_gemm(
        residual_packed.values,
        residual_packed.meta,
        layout=residual_packed.layout,
        K=MODEL_WIDTH,
    )
    return full_values, full_meta, residual_values, residual_meta


def run_case(
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
    full_values, full_meta, residual_values, residual_meta = weights
    rows = batch_size * (k + 1)
    dense_rows, _sparse_rows = route_indices(
        batch_size,
        k,
        dense_fraction=dense_fraction,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
    )
    dense_count = int(dense_rows.numel())
    dense_run = padded_rows(dense_count)
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows.long()] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    x = torch.randn(
        (rows, MODEL_WIDTH),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)
    dense_x = torch.zeros(
        (dense_run, MODEL_WIDTH), device="cuda", dtype=torch.float16
    )

    baseline_output = torch.empty(
        (rows, QKV_WIDTH), device="cuda", dtype=torch.float16
    )
    baseline_workspace = torch.empty(
        (QKV_WIDTH, rows), device="cuda", dtype=torch.float16
    )
    baseline_residual = torch.empty(
        (dense_run, QKV_WIDTH), device="cuda", dtype=torch.float16
    )
    baseline_residual_workspace = torch.empty(
        (QKV_WIDTH, dense_run), device="cuda", dtype=torch.float16
    )
    routed_output = torch.empty_like(baseline_output)
    routed_base = torch.empty(
        (dense_count, QKV_WIDTH), device="cuda", dtype=torch.float16
    )
    routed_residual = torch.empty_like(baseline_residual)
    routed_residual_workspace = torch.empty_like(baseline_residual_workspace)

    baseline_full_stream = torch.cuda.Stream()
    baseline_residual_stream = torch.cuda.Stream()
    routed_full_stream = torch.cuda.Stream()
    routed_residual_stream = torch.cuda.Stream()

    def baseline_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        baseline_full_stream.wait_stream(current)
        baseline_residual_stream.wait_stream(current)
        with torch.cuda.stream(baseline_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                full_values,
                full_meta,
                contiguous_output=True,
                out=baseline_output,
                workspace=baseline_workspace,
                device_config="auto",
            )
        with torch.cuda.stream(baseline_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                dense_x,
                residual_values,
                residual_meta,
                contiguous_output=True,
                out=baseline_residual,
                workspace=baseline_residual_workspace,
                device_config="auto",
            )
        current.wait_stream(baseline_full_stream)
        current.wait_stream(baseline_residual_stream)
        return sparse24_add_indexed_rows_contiguous_(
            baseline_output,
            baseline_residual[:dense_count],
            dense_rows,
        )

    def routed_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        routed_full_stream.wait_stream(current)
        routed_residual_stream.wait_stream(current)
        with torch.cuda.stream(routed_full_stream):
            sparse24_cutlass_routed_output_gemm_prepacked(
                x,
                full_values,
                full_meta,
                dense_slots,
                dense_count=dense_count,
                out=routed_output,
                dense_base=routed_base,
                config=config,
            )
        with torch.cuda.stream(routed_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                dense_x,
                residual_values,
                residual_meta,
                contiguous_output=True,
                out=routed_residual,
                workspace=routed_residual_workspace,
                device_config="auto",
            )
        current.wait_stream(routed_full_stream)
        current.wait_stream(routed_residual_stream)
        return sparse24_routed_linear_correction_(
            routed_base,
            routed_residual[:dense_count],
            dense_rows,
            routed_output,
        )

    expected = baseline_exact().clone()
    actual = routed_exact().clone()
    torch.cuda.synchronize()
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    if not torch.allclose(actual, expected, rtol=3e-2, atol=1e-1):
        raise RuntimeError(
            f"routed QKV mismatch bs={batch_size} K={k} config={config}: "
            f"max_abs_diff={max_abs_diff}"
        )
    baseline_ms, routed_ms = paired_graph_median_ms(
        baseline_exact,
        routed_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    return {
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": dense_count,
        "dense_fraction_actual": dense_count / rows,
        "config": config,
        "baseline_exact_qkv_ms": baseline_ms,
        "routed_exact_qkv_ms": routed_ms,
        "routed_qkv_speedup": baseline_ms / routed_ms,
        "max_abs_diff": max_abs_diff,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    configs = list(dict.fromkeys(str(row["config"]) for row in rows))
    figure, axes = plt.subplots(
        1,
        len(configs),
        figsize=(5.2 * len(configs), 4.1),
        squeeze=False,
        sharey=True,
    )
    for axis, config in zip(axes[0], configs, strict=True):
        selected = [row for row in rows if row["config"] == config]
        for k in sorted({int(row["K"]) for row in selected}):
            by_k = [row for row in selected if int(row["K"]) == k]
            axis.plot(
                [int(row["batch_size"]) for row in by_k],
                [float(row["routed_qkv_speedup"]) for row in by_k],
                marker="o",
                color=colors[k],
                label=f"K={k}",
            )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(config)
        axis.set_xlabel("Batch size")
        axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0][0].set_ylabel("Routed QKV speedup")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--k-values", type=parse_csv_ints, default=DEFAULT_K_VALUES)
    parser.add_argument("--configs", type=parse_csv_strings, default=DEFAULT_CONFIGS)
    parser.add_argument("--dense-fraction", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=30)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_configs = [item for item in args.configs if item not in DEFAULT_CONFIGS]
    if invalid_configs:
        raise ValueError(f"unsupported routed QKV configs: {invalid_configs}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weights = prepare_weights(generator)

    results: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for k in args.k_values:
            for config in args.configs:
                result = run_case(
                    batch_size,
                    k,
                    config,
                    weights,
                    dense_fraction=args.dense_fraction,
                    min_dense_per_request=args.min_dense_per_request,
                    generator=generator,
                    unroll=args.unroll,
                    replays=args.replays,
                    trials=args.trials,
                    graph_warmup_replays=args.graph_warmup_replays,
                )
                results.append(result)
                print(
                    f"bs={batch_size} K={k} "
                    f"dense={int(result['dense_rows'])}/{int(result['rows'])} "
                    f"config={config} "
                    f"speedup={float(result['routed_qkv_speedup']):.3f}x",
                    flush=True,
                )

    csv_path = args.output_root / "routed_qkv_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "routed_qkv_speedup.png", results)
    print(args.output_root)


if __name__ == "__main__":
    main()
