#!/usr/bin/env python3
"""Benchmark explicit and single-launch split-K exact sparse Down paths."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_transposed_to_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_device_splitk_indexed_add_gemm_prepacked,
    sparse24_cutlass_device_splitk_gemm_prepacked,
    sparse24_cutlass_device_strided_input_gemm_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_signal_ready_,
    sparse24_transpose_add_routed_residual,
    sparse24_transpose_add_routed_splitk_residual,
)


MODELS = {
    "qwen3_8b": {
        "intermediate": 12288,
        "hidden": 4096,
        "ratio": 0.125,
        "cap": 32,
    },
    "llama3_1_8b": {
        "intermediate": 14336,
        "hidden": 4096,
        "ratio": 0.3125,
        "cap": 64,
    },
}


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def _csv_models(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in values if item not in MODELS]
    if not values or invalid:
        raise argparse.ArgumentTypeError(f"unsupported models: {invalid}")
    return values


def _capture_unrolled(
    fn: Callable[[], torch.Tensor],
    *,
    repeat: int,
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            captured = fn()
    torch.cuda.synchronize()
    del captured
    return graph


def _graph_sample_ms(
    graph: torch.cuda.CUDAGraph,
    *,
    repeat: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _paired_graph_median_ms(
    standard_fn: Callable[[], torch.Tensor],
    split_fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> tuple[float, float]:
    standard_graph = _capture_unrolled(standard_fn, repeat=repeat)
    split_graph = _capture_unrolled(split_fn, repeat=repeat)
    for _ in range(warmup):
        standard_graph.replay()
        split_graph.replay()
    torch.cuda.synchronize()

    standard_samples: list[float] = []
    split_samples: list[float] = []
    for trial in range(trials):
        if trial % 2 == 0:
            standard_samples.append(
                _graph_sample_ms(standard_graph, repeat=repeat)
            )
            split_samples.append(_graph_sample_ms(split_graph, repeat=repeat))
        else:
            split_samples.append(_graph_sample_ms(split_graph, repeat=repeat))
            standard_samples.append(
                _graph_sample_ms(standard_graph, repeat=repeat)
            )
    return statistics.median(standard_samples), statistics.median(split_samples)


def _prepack(weight24: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_24(weight24, layout="n_major")
    return prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=int(weight24.shape[0]),
    )


def _prepare_down_weights(
    model: str,
    split_k_values: tuple[int, ...],
    generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, tuple[tuple[int, int, torch.Tensor, torch.Tensor], ...]],
]:
    K = int(MODELS[model]["intermediate"])
    N = int(MODELS[model]["hidden"])
    weight = torch.randn(
        (K, N), device="cuda", dtype=torch.float16, generator=generator
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    full24, _ = apply_random_24_mask(weight, generator=generator)
    residual24 = weight - full24
    full_values, full_metadata = _prepack(full24)
    residual_values, residual_metadata = _prepack(residual24)

    split_weights: dict[
        int, tuple[tuple[int, int, torch.Tensor, torch.Tensor], ...]
    ] = {}
    for split_k in split_k_values:
        if K % split_k:
            raise ValueError(f"{model} K={K} is not divisible by split={split_k}")
        slice_k = K // split_k
        if slice_k % 64:
            raise ValueError(
                f"{model} split={split_k} gives K slice {slice_k}, not divisible by 64"
            )
        slices: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for split in range(split_k):
            start = split * slice_k
            end = start + slice_k
            values = residual_values[:, start // 2 : end // 2]
            metadata_start = (start // 32) * N * 2
            metadata_end = metadata_start + (slice_k // 32) * N * 2
            metadata = residual_metadata[metadata_start:metadata_end]
            slices.append((start, end, values, metadata))
        split_weights[split_k] = tuple(slices)

    return (
        weight,
        full_values,
        full_metadata,
        residual_values,
        residual_metadata,
        split_weights,
    )


def _dense_counts(model: str, batch_size: int, k: int) -> tuple[int, int]:
    config = MODELS[model]
    scored_rows = batch_size * k
    ratio_budget = int(float(config["ratio"]) * scored_rows + 0.5)
    dense_count = min(
        scored_rows,
        int(config["cap"]),
        max(batch_size, ratio_budget),
    )
    dense_run = max(8, math.ceil(dense_count / 8) * 8)
    return dense_count, dense_run


def _inline_down_config(rows: int) -> str:
    """Best common Qwen/Llama Down config from the bs>=16 shape sweep."""
    if rows <= 144:
        return "128x32x64_s4_sw4"
    if rows <= 288:
        return "128x64x64_s5"
    if rows <= 576:
        return "256x64x64_s3"
    return "128x64x64_s5"


def _run_case(
    model: str,
    batch_size: int,
    k: int,
    split_k_values: tuple[int, ...],
    dense_weight: torch.Tensor,
    full_values: torch.Tensor,
    full_metadata: torch.Tensor,
    residual_values: torch.Tensor,
    residual_metadata: torch.Tensor,
    split_weights: dict[
        int, tuple[tuple[int, int, torch.Tensor, torch.Tensor], ...]
    ],
    generator: torch.Generator,
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> list[dict[str, object]]:
    rows = batch_size * (k + 1)
    dense_count, dense_run = _dense_counts(model, batch_size, k)
    K = int(MODELS[model]["intermediate"])
    N = int(MODELS[model]["hidden"])
    inline_down_config = _inline_down_config(rows)

    full_x = torch.randn(
        (rows, K), device="cuda", dtype=torch.float16, generator=generator
    )
    full_x.mul_(0.1)
    dense_rows = torch.arange(dense_count, device="cuda", dtype=torch.int64)
    dense_rows_i32 = dense_rows.to(dtype=torch.int32)
    dense_x = torch.zeros(
        (dense_run, K), device="cuda", dtype=torch.float16
    )
    dense_x[:dense_count].copy_(full_x.index_select(0, dense_rows))
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[:dense_count] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )

    standard_full_out = torch.empty_strided(
        (rows, N), (1, rows), device="cuda", dtype=torch.float16
    )
    standard_residual_out = torch.empty_strided(
        (dense_run, N), (1, dense_run), device="cuda", dtype=torch.float16
    )
    standard_output = torch.empty(
        (rows, N), device="cuda", dtype=torch.float16
    )

    results: list[dict[str, object]] = []
    for split_k in split_k_values:
        split_full_out = torch.empty_strided(
            (rows, N), (1, rows), device="cuda", dtype=torch.float16
        )
        split_partials = torch.empty_strided(
            (split_k, dense_run, N),
            (N * dense_run, 1, dense_run),
            device="cuda",
            dtype=torch.float16,
        )
        split_output = torch.empty_like(standard_output)
        serial_residual_out = torch.empty_strided(
            (dense_run, N),
            (1, dense_run),
            device="cuda",
            dtype=torch.float16,
        )
        serial_output = torch.empty_like(standard_output)
        serial_workspace = torch.zeros(
            math.ceil(N / 256) * math.ceil(dense_run / 32),
            device="cuda",
            dtype=torch.int32,
        )
        fused_output = torch.empty_like(standard_output)
        fused_residual_out = torch.empty_strided(
            (dense_run, N),
            (1, dense_run),
            device="cuda",
            dtype=torch.float16,
        )
        fused_workspace = torch.zeros_like(serial_workspace)
        fused_ready_state = torch.zeros(
            2, device="cuda", dtype=torch.int32
        )
        tiled_output = torch.empty_like(standard_output)
        tiled_residual_out = torch.empty_strided(
            (dense_run, N),
            (1, dense_run),
            device="cuda",
            dtype=torch.float16,
        )
        tiled_workspace = torch.zeros_like(serial_workspace)
        slices = split_weights[split_k]
        x_slices = tuple(dense_x[:, start:end] for start, end, _, _ in slices)

        standard_full_stream = torch.cuda.Stream()
        standard_residual_stream = torch.cuda.Stream()
        split_full_stream = torch.cuda.Stream()
        split_residual_streams = tuple(torch.cuda.Stream() for _ in slices)
        serial_full_stream = torch.cuda.Stream()
        serial_residual_stream = torch.cuda.Stream()
        fused_full_stream = torch.cuda.Stream()
        fused_residual_stream = torch.cuda.Stream()
        tiled_full_stream = torch.cuda.Stream()
        tiled_residual_stream = torch.cuda.Stream()

        def standard_exact() -> torch.Tensor:
            current = torch.cuda.current_stream()
            standard_full_stream.wait_stream(current)
            standard_residual_stream.wait_stream(current)
            with torch.cuda.stream(standard_full_stream):
                sparse24_cutlass_device_gemm_prepacked(
                    full_x,
                    full_values,
                    full_metadata,
                    out=standard_full_out,
                    device_config="auto",
                )
            with torch.cuda.stream(standard_residual_stream):
                sparse24_cutlass_device_gemm_prepacked(
                    dense_x,
                    residual_values,
                    residual_metadata,
                    out=standard_residual_out,
                    device_config="auto",
                )
            current.wait_stream(standard_full_stream)
            current.wait_stream(standard_residual_stream)
            return sparse24_transpose_add_routed_residual(
                standard_full_out,
                standard_residual_out,
                dense_slots,
                dense_count=dense_count,
                out=standard_output,
            )

        def split_exact() -> torch.Tensor:
            current = torch.cuda.current_stream()
            split_full_stream.wait_stream(current)
            for stream in split_residual_streams:
                stream.wait_stream(current)
            with torch.cuda.stream(split_full_stream):
                sparse24_cutlass_device_gemm_prepacked(
                    full_x,
                    full_values,
                    full_metadata,
                    out=split_full_out,
                    device_config="auto",
                )
            for split, (stream, x_slice, packed_slice) in enumerate(
                zip(split_residual_streams, x_slices, slices, strict=True)
            ):
                _, _, values, metadata = packed_slice
                with torch.cuda.stream(stream):
                    sparse24_cutlass_device_strided_input_gemm_prepacked(
                        x_slice,
                        values,
                        metadata,
                        out=split_partials[split],
                    )
            current.wait_stream(split_full_stream)
            for stream in split_residual_streams:
                current.wait_stream(stream)
            return sparse24_transpose_add_routed_splitk_residual(
                split_full_out,
                split_partials,
                dense_slots,
                dense_count=dense_count,
                out=split_output,
            )

        def serial_exact() -> torch.Tensor:
            current = torch.cuda.current_stream()
            serial_full_stream.wait_stream(current)
            serial_residual_stream.wait_stream(current)
            with torch.cuda.stream(serial_full_stream):
                sparse24_cutlass_device_gemm_prepacked(
                    full_x,
                    full_values,
                    full_metadata,
                    out=split_full_out,
                    device_config="auto",
                )
            with torch.cuda.stream(serial_residual_stream):
                sparse24_cutlass_device_splitk_gemm_prepacked(
                    dense_x,
                    residual_values,
                    residual_metadata,
                    split_k_slices=split_k,
                    out=serial_residual_out,
                    workspace=serial_workspace,
                )
            current.wait_stream(serial_full_stream)
            current.wait_stream(serial_residual_stream)
            return sparse24_transpose_add_routed_residual(
                split_full_out,
                serial_residual_out,
                dense_slots,
                dense_count=dense_count,
                out=serial_output,
            )

        def fused_indexed_add_exact() -> torch.Tensor:
            current = torch.cuda.current_stream()
            fused_full_stream.wait_stream(current)
            fused_residual_stream.wait_stream(current)
            with torch.cuda.stream(fused_full_stream):
                sparse24_cutlass_inline_transpose_gemm_prepacked(
                    full_x,
                    full_values,
                    full_metadata,
                    out=fused_output,
                    config=inline_down_config,
                    store_mode="vector",
                )
                sparse24_cutlass_signal_ready_(fused_ready_state)
            with torch.cuda.stream(fused_residual_stream):
                sparse24_cutlass_device_splitk_indexed_add_gemm_prepacked(
                    dense_x,
                    residual_values,
                    residual_metadata,
                    dense_rows_i32,
                    fused_output,
                    split_k_slices=split_k,
                    out=fused_residual_out,
                    workspace=fused_workspace,
                    ready_state=fused_ready_state,
                )
            current.wait_stream(fused_full_stream)
            current.wait_stream(fused_residual_stream)
            return fused_output

        def tiled_indexed_add_exact() -> torch.Tensor:
            current = torch.cuda.current_stream()
            tiled_full_stream.wait_stream(current)
            tiled_residual_stream.wait_stream(current)
            with torch.cuda.stream(tiled_full_stream):
                sparse24_cutlass_inline_transpose_gemm_prepacked(
                    full_x,
                    full_values,
                    full_metadata,
                    out=tiled_output,
                    config=inline_down_config,
                    store_mode="vector",
                )
            with torch.cuda.stream(tiled_residual_stream):
                sparse24_cutlass_device_splitk_gemm_prepacked(
                    dense_x,
                    residual_values,
                    residual_metadata,
                    split_k_slices=split_k,
                    out=tiled_residual_out,
                    workspace=tiled_workspace,
                )
            current.wait_stream(tiled_full_stream)
            current.wait_stream(tiled_residual_stream)
            return sparse24_add_indexed_rows_transposed_to_contiguous_(
                tiled_output,
                tiled_residual_out,
                dense_rows_i32,
            )

        standard_actual = standard_exact().clone()
        split_actual = split_exact().clone()
        serial_actual = serial_exact().clone()
        fused_actual = fused_indexed_add_exact().clone()
        tiled_actual = tiled_indexed_add_exact().clone()
        torch.cuda.synchronize()
        split_vs_standard = float(
            (split_actual.float() - standard_actual.float()).abs().max().item()
        )
        if not torch.allclose(
            split_actual, standard_actual, rtol=2e-2, atol=8e-2
        ):
            raise RuntimeError(
                f"parallel split-K mismatch for {model} bs={batch_size} K={k} "
                f"split={split_k}: max_abs_diff={split_vs_standard}"
            )
        serial_vs_standard = float(
            (serial_actual.float() - standard_actual.float()).abs().max().item()
        )
        if not torch.allclose(
            serial_actual, standard_actual, rtol=2e-2, atol=8e-2
        ):
            raise RuntimeError(
                f"single-launch split-K mismatch for {model} "
                f"bs={batch_size} K={k} split={split_k}: "
                f"max_abs_diff={serial_vs_standard}"
            )
        fused_vs_standard = float(
            (fused_actual.float() - standard_actual.float()).abs().max().item()
        )
        if not torch.allclose(
            fused_actual, standard_actual, rtol=2e-2, atol=8e-2
        ):
            raise RuntimeError(
                f"split-K indexed-add mismatch for {model} "
                f"bs={batch_size} K={k} split={split_k}: "
                f"max_abs_diff={fused_vs_standard}"
            )
        ready_state_after = tuple(
            int(value) for value in fused_ready_state.cpu()
        )
        if ready_state_after != (0, 0):
            raise RuntimeError(
                f"split-K indexed-add state did not reset: "
                f"{ready_state_after}"
            )
        tiled_vs_standard = float(
            (tiled_actual.float() - standard_actual.float()).abs().max().item()
        )
        if not torch.allclose(
            tiled_actual, standard_actual, rtol=2e-2, atol=8e-2
        ):
            raise RuntimeError(
                f"tiled indexed-add mismatch for {model} "
                f"bs={batch_size} K={k} split={split_k}: "
                f"max_abs_diff={tiled_vs_standard}"
            )

        expected = torch.mm(full_x, dense_weight)
        standard_dense_diff = float(
            (
                standard_actual.index_select(0, dense_rows).float()
                - expected.index_select(0, dense_rows).float()
            )
            .abs()
            .max()
            .item()
        )
        del expected
        standard_ms, split_ms = _paired_graph_median_ms(
            standard_exact,
            split_exact,
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        serial_standard_ms, serial_ms = _paired_graph_median_ms(
            standard_exact,
            serial_exact,
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        fused_standard_ms, fused_ms = _paired_graph_median_ms(
            standard_exact,
            fused_indexed_add_exact,
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        tiled_standard_ms, tiled_ms = _paired_graph_median_ms(
            standard_exact,
            tiled_indexed_add_exact,
            warmup=warmup,
            repeat=repeat,
            trials=trials,
        )
        base_ctas = math.ceil(N / 256) * math.ceil(dense_run / 32)
        results.append(
            {
                "model": model,
                "batch_size": batch_size,
                "K": k,
                "rows": rows,
                "dense_count": dense_count,
                "dense_run": dense_run,
                "indexed_add_full_config": inline_down_config,
                "split_k_slices": split_k,
                "base_residual_ctas": base_ctas,
                "split_residual_ctas": base_ctas * split_k,
                "partial_workspace_bytes": split_partials.numel()
                * split_partials.element_size(),
                "standard_exact_down_ms": standard_ms,
                "split_exact_down_ms": split_ms,
                "exact_down_speedup": standard_ms / split_ms,
                "single_launch_standard_exact_down_ms": serial_standard_ms,
                "single_launch_split_exact_down_ms": serial_ms,
                "single_launch_exact_down_speedup": (
                    serial_standard_ms / serial_ms
                ),
                "single_launch_workspace_bytes": (
                    serial_workspace.numel() * serial_workspace.element_size()
                ),
                "indexed_add_standard_exact_down_ms": fused_standard_ms,
                "indexed_add_split_exact_down_ms": fused_ms,
                "indexed_add_exact_down_speedup": fused_standard_ms
                / fused_ms,
                "indexed_add_workspace_bytes": (
                    fused_workspace.numel() * fused_workspace.element_size()
                    + fused_ready_state.numel()
                    * fused_ready_state.element_size()
                ),
                "tiled_indexed_add_standard_exact_down_ms": tiled_standard_ms,
                "tiled_indexed_add_exact_down_ms": tiled_ms,
                "tiled_indexed_add_exact_down_speedup": tiled_standard_ms
                / tiled_ms,
                "tiled_indexed_add_workspace_bytes": (
                    tiled_workspace.numel() * tiled_workspace.element_size()
                ),
                "standard_dense_row_max_abs_diff": standard_dense_diff,
                "split_vs_standard_max_abs_diff": split_vs_standard,
                "single_launch_vs_standard_max_abs_diff": serial_vs_standard,
                "indexed_add_vs_standard_max_abs_diff": fused_vs_standard,
                "tiled_indexed_add_vs_standard_max_abs_diff": (
                    tiled_vs_standard
                ),
                "indexed_add_ready_state_0": ready_state_after[0],
                "indexed_add_ready_state_1": ready_state_after[1],
            }
        )
    return results


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    figure, axes = plt.subplots(
        1, len(models), figsize=(7.2 * len(models), 4.4), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        selected_model = [row for row in rows if row["model"] == model]
        case_order = sorted(
            {
                (int(row["batch_size"]), int(row["K"]))
                for row in selected_model
            }
        )
        for split_k in sorted(
            {int(row["split_k_slices"]) for row in selected_model}
        ):
            selected = {
                (int(row["batch_size"]), int(row["K"])): row
                for row in selected_model
                if int(row["split_k_slices"]) == split_k
            }
            axis.plot(
                range(len(case_order)),
                [float(selected[case]["exact_down_speedup"]) for case in case_order],
                marker="o",
                label=f"parallel split-K {split_k}",
            )
            axis.plot(
                range(len(case_order)),
                [
                    float(selected[case]["single_launch_exact_down_speedup"])
                    for case in case_order
                ],
                marker="s",
                linestyle="--",
                label=f"single-launch split-K {split_k}",
            )
            axis.plot(
                range(len(case_order)),
                [
                    float(selected[case]["indexed_add_exact_down_speedup"])
                    for case in case_order
                ],
                marker="^",
                linestyle=":",
                label=f"indexed-add split-K {split_k}",
            )
            axis.plot(
                range(len(case_order)),
                [
                    float(
                        selected[case][
                            "tiled_indexed_add_exact_down_speedup"
                        ]
                    )
                    for case in case_order
                ],
                marker="D",
                linestyle="-.",
                label=f"tiled indexed-add split-K {split_k}",
            )
        labels = [f"bs{batch}/K{k}" for batch, k in case_order]
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
        axis.set_title(model)
        axis.set_ylabel("Exact Down speedup")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=_csv_models, default=tuple(MODELS))
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(16, 32, 64))
    parser.add_argument("--k-values", type=_csv_ints, default=(6, 8, 10))
    parser.add_argument("--split-k-values", type=_csv_ints, default=(2, 4, 8))
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_parallel_splitk_down_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.repeat <= 0 or args.trials <= 0 or args.warmup < 0:
        raise ValueError("repeat/trials must be positive and warmup non-negative")
    args.output_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cuda").manual_seed(20260715)
    results: list[dict[str, object]] = []
    for model in args.models:
        (
            dense_weight,
            full_values,
            full_metadata,
            residual_values,
            residual_metadata,
            split_weights,
        ) = _prepare_down_weights(model, args.split_k_values, generator)
        for batch_size in args.batch_sizes:
            for k in args.k_values:
                results.extend(
                    _run_case(
                        model,
                        batch_size,
                        k,
                        args.split_k_values,
                        dense_weight,
                        full_values,
                        full_metadata,
                        residual_values,
                        residual_metadata,
                        split_weights,
                        generator,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        trials=args.trials,
                    )
                )
        del (
            dense_weight,
            full_values,
            full_metadata,
            residual_values,
            residual_metadata,
            split_weights,
        )

    csv_path = args.output_root / "parallel_splitk_down_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    _plot(results, args.output_root / "parallel_splitk_down_speedup.png")
    print(csv_path)
    for row in results:
        print(
            f"{row['model']} bs={row['batch_size']} K={row['K']} "
            f"dense={row['dense_count']}/{row['dense_run']} "
            f"split={row['split_k_slices']} "
            f"{float(row['standard_exact_down_ms']):.4f}->"
            f"{float(row['split_exact_down_ms']):.4f} ms "
            f"speedup={float(row['exact_down_speedup']):.3f}x "
            f"single={float(row['single_launch_split_exact_down_ms']):.4f} ms "
            f"single_speedup="
            f"{float(row['single_launch_exact_down_speedup']):.3f}x "
            f"indexed_add="
            f"{float(row['indexed_add_split_exact_down_ms']):.4f} ms "
            f"indexed_speedup="
            f"{float(row['indexed_add_exact_down_speedup']):.3f}x "
            f"tiled_indexed_add="
            f"{float(row['tiled_indexed_add_exact_down_ms']):.4f} ms "
            f"tiled_speedup="
            f"{float(row['tiled_indexed_add_exact_down_speedup']):.3f}x "
            f"diff={float(row['single_launch_vs_standard_max_abs_diff']):.5f}"
        )


if __name__ == "__main__":
    main()
