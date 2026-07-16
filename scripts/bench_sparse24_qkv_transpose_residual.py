#!/usr/bin/env python3
"""Benchmark fused QKV transpose plus routed residual correction."""

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
)
from bench_sparse24_paired_residual import (  # noqa: E402
    padded_rows,
    route_indices,
)
from bench_sparse24_routed_qkv import (  # noqa: E402
    MODEL_WIDTH,
    QKV_WIDTH,
    prepare_weights,
)
from vllm.speclink_kernel import (  # noqa: E402
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_paired_persistent_gemm_prepacked,
    sparse24_gather_rows_,
    sparse24_transpose_add_routed_residual,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)


def run_case(
    batch_size: int,
    k: int,
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
    ).mul_(0.1)
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
    fused_full = torch.empty_strided(
        (rows, QKV_WIDTH),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    fused_residual = torch.empty_strided(
        (dense_run, QKV_WIDTH),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    fused_output = torch.empty_like(baseline_output)
    paired_full = torch.empty_strided(
        (rows, QKV_WIDTH),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    paired_residual = torch.empty_strided(
        (dense_run, QKV_WIDTH),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    paired_output = torch.empty_like(baseline_output)

    baseline_full_stream = torch.cuda.Stream()
    baseline_residual_stream = torch.cuda.Stream()
    fused_full_stream = torch.cuda.Stream()
    fused_residual_stream = torch.cuda.Stream()

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

    def fused_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        fused_full_stream.wait_stream(current)
        fused_residual_stream.wait_stream(current)
        with torch.cuda.stream(fused_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                full_values,
                full_meta,
                contiguous_output=False,
                out=fused_full,
                device_config="auto",
            )
        with torch.cuda.stream(fused_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                dense_x,
                residual_values,
                residual_meta,
                contiguous_output=False,
                out=fused_residual,
                device_config="auto",
            )
        current.wait_stream(fused_full_stream)
        current.wait_stream(fused_residual_stream)
        return sparse24_transpose_add_routed_residual(
            fused_full,
            fused_residual,
            dense_slots,
            dense_count=dense_count,
            out=fused_output,
        )

    def paired_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
        sparse24_cutlass_paired_persistent_gemm_prepacked(
            x,
            full_values,
            full_meta,
            dense_x,
            residual_values,
            residual_meta,
            full_out=paired_full,
            residual_out=paired_residual,
            schedule="interleaved",
        )
        return sparse24_transpose_add_routed_residual(
            paired_full,
            paired_residual,
            dense_slots,
            dense_count=dense_count,
            out=paired_output,
        )

    expected = baseline_exact().clone()
    actual = fused_exact().clone()
    paired_actual = paired_exact().clone()
    torch.cuda.synchronize()
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    paired_max_abs_diff = float(
        (paired_actual.float() - expected.float()).abs().max().item()
    )
    if not torch.allclose(actual, expected, rtol=3e-2, atol=1e-1):
        raise RuntimeError(
            f"fused QKV mismatch bs={batch_size} K={k}: "
            f"max_abs_diff={max_abs_diff}"
        )
    if not torch.allclose(paired_actual, expected, rtol=3e-2, atol=1e-1):
        raise RuntimeError(
            f"paired-persistent QKV mismatch bs={batch_size} K={k}: "
            f"max_abs_diff={paired_max_abs_diff}"
        )
    baseline_ms, fused_ms = paired_graph_median_ms(
        baseline_exact,
        fused_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    fused_control_ms, paired_ms = paired_graph_median_ms(
        fused_exact,
        paired_exact,
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
        "baseline_exact_qkv_ms": baseline_ms,
        "fused_transpose_residual_ms": fused_ms,
        "fused_qkv_speedup": baseline_ms / fused_ms,
        "max_abs_diff": max_abs_diff,
        "paired_control_ms": fused_control_ms,
        "paired_persistent_qkv_ms": paired_ms,
        "paired_persistent_speedup_vs_fused": fused_control_ms / paired_ms,
        "paired_persistent_max_abs_diff": paired_max_abs_diff,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for k in sorted({int(row["K"]) for row in rows}):
        selected = [row for row in rows if int(row["K"]) == k]
        axis.plot(
            [int(row["batch_size"]) for row in selected],
            [float(row["fused_qkv_speedup"]) for row in selected],
            marker="o",
            color=colors[k],
            label=f"K={k}",
        )
        axis.plot(
            [int(row["batch_size"]) for row in selected],
            [
                float(row["paired_persistent_speedup_vs_fused"])
                for row in selected
            ],
            marker="s",
            color=colors[k],
            linestyle="--",
            label=f"K={k}, persistent vs fused",
        )
    axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("QKV speedup")
    axis.set_xticks(sorted({int(row["batch_size"]) for row in rows}))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--k-values", type=parse_csv_ints, default=DEFAULT_K_VALUES)
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
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weights = prepare_weights(generator)
    results: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for k in args.k_values:
            result = run_case(
                batch_size,
                k,
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
                f"fused={float(result['fused_qkv_speedup']):.3f}x "
                "persistent_vs_fused="
                f"{float(result['paired_persistent_speedup_vs_fused']):.3f}x",
                flush=True,
            )
    csv_path = args.output_root / "qkv_transpose_residual_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "qkv_transpose_residual_speedup.png", results)
    print(args.output_root)


if __name__ == "__main__":
    main()
