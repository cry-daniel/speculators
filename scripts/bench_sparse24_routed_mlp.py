#!/usr/bin/env python3
"""Benchmark the selected fused-layout path across an exact routed MLP."""

from __future__ import annotations

import argparse
import csv
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
from bench_sparse24_routed_swiglu import (  # noqa: E402
    DEFAULT_CONFIGS,
    prepare_gate_up_weights,
)
from vllm import _custom_ops as _vllm_custom_ops  # noqa: E402,F401
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    assert_24_weight,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_add_indexed_rows_strided_,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_paired_gather_residual_prepacked,
    sparse24_cutlass_paired_persistent_gemm_prepacked,
    sparse24_cutlass_paired_persistent_routed_swiglu_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_,
    sparse24_gather_rows_strided_,
    sparse24_routed_swiglu_correction_,
    sparse24_routed_swiglu_correction_gather_,
    sparse24_silu_and_mul_transposed,
    sparse24_transpose_add_routed_residual,
    sparse24_transpose_input_to_strided,
    sparse24_transpose_output_contiguous,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
SUPPORTED_CONFIGS = (
    *DEFAULT_CONFIGS,
    "256x32x64_s3_sw4",
    "256x64x64_s2_sw4",
)
SUPPORTED_PAIRED_DOWN_CONFIGS = (
    "auto",
    "256x64_full_256x64_residual",
    "256x128_full_256x64_residual",
    "256x128_full_256x128_residual",
)
SUPPORTED_PAIRED_GATHER_DOWN_CONFIGS = (
    "128x64_full_128x64_residual_contiguous",
    "256x32_full_256x32_residual_contiguous",
    "64x64_full_64x64_residual_contiguous",
    "256x32_full_128x32_residual_contiguous",
    "256x64_full_256x64_residual_contiguous",
)
PackedGateUp = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
PackedDown = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def prepare_down_weights(
    intermediate: int,
    model_width: int,
    generator: torch.Generator,
) -> PackedDown:
    weight = torch.randn(
        (intermediate, model_width),
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
        K=intermediate,
    )
    residual_packed = pack_24(residual24, layout="n_major")
    residual_values, residual_meta = prepare_cutlass_sparse24_device_gemm(
        residual_packed.values,
        residual_packed.meta,
        layout=residual_packed.layout,
        K=intermediate,
    )
    return weight, full_values, full_meta, residual_values, residual_meta


def run_case(
    model: str,
    batch_size: int,
    k: int,
    config: str,
    gate_weights: PackedGateUp,
    down_weights: PackedDown,
    *,
    paired_down_config: str,
    paired_gather_down_config: str,
    route_mode: str,
    dense_fraction: float,
    dense_cap: int,
    min_dense_per_request: int,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
    profile_iterations: int,
    profile_variant: str,
) -> dict[str, object]:
    model_width = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    gate_up_size = 2 * intermediate
    control_config = "256x64x64_s3_sw4"
    (
        dense_gate_weight,
        gate_full_values,
        gate_full_meta,
        gate_routed_values,
        gate_routed_meta,
        gate_residual_values,
        gate_residual_meta,
        _gate_residual_fp8,
        _gate_residual_fp8_scale,
    ) = gate_weights
    (
        dense_down_weight,
        down_full_values,
        down_full_meta,
        down_residual_values,
        down_residual_meta,
    ) = down_weights

    rows = batch_size * (k + 1)
    dense_rows, _sparse_rows = route_indices(
        batch_size,
        k,
        dense_fraction=dense_fraction,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
        route_mode=route_mode,
        dense_cap=dense_cap,
    )
    dense_count = int(dense_rows.numel())
    dense_run = padded_rows(dense_count)
    dense_slots = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
    dense_slots[dense_rows.long()] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )

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

    baseline_gate = torch.empty_strided(
        (rows, gate_up_size),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_gate_residual = torch.empty_strided(
        (dense_run, gate_up_size),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_hidden = torch.empty_strided(
        (rows, intermediate),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_dense_hidden = torch.zeros(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    ).as_strided((dense_run, intermediate), (1, dense_run))
    baseline_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_down_residual = torch.empty_strided(
        (dense_run, model_width),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_output = torch.empty(
        (rows, model_width), device="cuda", dtype=torch.float16
    )
    dense_gate_up = torch.empty(
        (rows, gate_up_size), device="cuda", dtype=torch.float16
    )
    dense_hidden_output = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    dense_mlp_output = torch.empty_like(baseline_output)

    routed_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    routed_hidden_transposed = torch.empty_strided(
        routed_hidden.shape,
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    routed_dense_base = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    routed_gate_residual = torch.empty(
        (dense_run, gate_up_size), device="cuda", dtype=torch.float16
    )
    routed_gate_workspace = torch.empty(
        (gate_up_size, dense_run), device="cuda", dtype=torch.float16
    )
    routed_dense_hidden = torch.zeros(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    ).as_strided((dense_run, intermediate), (1, dense_run))
    routed_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    routed_down_residual = torch.empty_strided(
        (dense_run, model_width),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    routed_output = torch.empty_like(baseline_output)

    fused_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    fused_hidden_transposed = torch.empty_strided(
        fused_hidden.shape,
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    fused_dense_base = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    fused_gate_residual = torch.empty(
        (dense_run, gate_up_size), device="cuda", dtype=torch.float16
    )
    fused_dense_hidden = torch.zeros(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    fused_dense_hidden_transposed = torch.zeros(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    ).as_strided((dense_run, intermediate), (1, dense_run))
    fused_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    fused_down_residual = torch.empty_strided(
        (dense_run, model_width),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    fused_output = torch.empty_like(baseline_output)

    paired_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    paired_hidden_transposed = torch.empty_strided(
        paired_hidden.shape,
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    paired_dense_base = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    paired_gate_residual = torch.empty(
        (dense_run, gate_up_size), device="cuda", dtype=torch.float16
    )
    paired_dense_hidden = torch.empty(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    paired_dense_hidden_transposed = torch.empty_strided(
        (dense_run, intermediate),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    paired_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    paired_down_residual = torch.empty_strided(
        (dense_run, model_width),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    paired_output = torch.empty_like(baseline_output)
    paired_both_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    paired_both_down_residual = torch.empty_strided(
        (dense_run, model_width),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    paired_both_output = torch.empty_like(baseline_output)
    paired_gather_down = torch.empty_like(baseline_output)
    paired_gather_down_residual = torch.empty(
        (dense_count, model_width), device="cuda", dtype=torch.float16
    )

    dense_overwrite_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    dense_overwrite_base = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    dense_overwrite_gate_up = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    dense_overwrite_dense_hidden = torch.empty(
        (dense_count, intermediate), device="cuda", dtype=torch.float16
    )
    dense_overwrite_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    dense_overwrite_dense_down = torch.empty(
        (dense_count, model_width), device="cuda", dtype=torch.float16
    )
    dense_overwrite_output = torch.empty_like(baseline_output)

    gather_fused_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    gather_fused_hidden_transposed = torch.empty_strided(
        gather_fused_hidden.shape,
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    gather_fused_dense_base = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    gather_fused_gate_residual = torch.empty(
        (dense_run, gate_up_size), device="cuda", dtype=torch.float16
    )
    gather_fused_dense_hidden = torch.empty(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    gather_fused_down = torch.empty_strided(
        (rows, model_width),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    gather_fused_down_residual = torch.empty_strided(
        (dense_run, model_width),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    gather_fused_output = torch.empty_like(baseline_output)

    # Qwen's largest tested M=704 shape is faster with the original B-row input.
    use_contiguous_down = not (
        model == "qwen3_8b" and rows >= 704 and intermediate == 12288
    )

    baseline_full_stream = torch.cuda.Stream()
    baseline_residual_stream = torch.cuda.Stream()
    routed_full_stream = torch.cuda.Stream()
    routed_residual_stream = torch.cuda.Stream()
    fused_full_stream = torch.cuda.Stream()
    fused_residual_stream = torch.cuda.Stream()
    paired_full_stream = torch.cuda.Stream()
    paired_residual_stream = torch.cuda.Stream()
    dense_overwrite_full_stream = torch.cuda.Stream()
    dense_overwrite_dense_stream = torch.cuda.Stream()
    gather_fused_full_stream = torch.cuda.Stream()
    gather_fused_residual_stream = torch.cuda.Stream()

    def dense_mlp() -> torch.Tensor:
        torch.mm(x, dense_gate_weight, out=dense_gate_up)
        torch.ops._C.silu_and_mul(dense_hidden_output, dense_gate_up)
        return torch.mm(
            dense_hidden_output, dense_down_weight, out=dense_mlp_output
        )

    def baseline_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        baseline_full_stream.wait_stream(current)
        baseline_residual_stream.wait_stream(current)
        with torch.cuda.stream(baseline_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                gate_full_values,
                gate_full_meta,
                out=baseline_gate,
                device_config="auto",
            )
        with torch.cuda.stream(baseline_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                dense_x,
                gate_residual_values,
                gate_residual_meta,
                out=baseline_gate_residual,
                device_config="auto",
            )
        current.wait_stream(baseline_full_stream)
        current.wait_stream(baseline_residual_stream)
        sparse24_add_indexed_rows_strided_(
            baseline_gate,
            baseline_gate_residual[:dense_count],
            dense_rows,
        )
        sparse24_silu_and_mul_transposed(baseline_gate, out=baseline_hidden)

        baseline_full_stream.wait_stream(current)
        baseline_residual_stream.wait_stream(current)
        with torch.cuda.stream(baseline_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                baseline_hidden,
                down_full_values,
                down_full_meta,
                input_transposed=True,
                out=baseline_down,
                device_config="auto",
            )
        with torch.cuda.stream(baseline_residual_stream):
            sparse24_gather_rows_strided_(
                baseline_hidden,
                dense_rows,
                baseline_dense_hidden[:dense_count],
            )
            sparse24_cutlass_device_gemm_prepacked(
                baseline_dense_hidden,
                down_residual_values,
                down_residual_meta,
                input_transposed=True,
                out=baseline_down_residual,
                device_config="auto",
            )
        current.wait_stream(baseline_full_stream)
        current.wait_stream(baseline_residual_stream)
        sparse24_add_indexed_rows_strided_(
            baseline_down,
            baseline_down_residual[:dense_count],
            dense_rows,
        )
        return sparse24_transpose_output_contiguous(
            baseline_down, out=baseline_output
        )

    def routed_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        routed_full_stream.wait_stream(current)
        routed_residual_stream.wait_stream(current)
        with torch.cuda.stream(routed_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate_routed_values,
                gate_routed_meta,
                dense_slots,
                dense_count=dense_count,
                out=routed_hidden,
                dense_base=routed_dense_base,
                config=control_config,
            )
        with torch.cuda.stream(routed_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                dense_x,
                gate_residual_values,
                gate_residual_meta,
                contiguous_output=True,
                out=routed_gate_residual,
                workspace=routed_gate_workspace,
                device_config="auto",
            )
        current.wait_stream(routed_full_stream)
        current.wait_stream(routed_residual_stream)
        sparse24_routed_swiglu_correction_(
            routed_dense_base,
            routed_gate_residual[:dense_count],
            dense_rows,
            routed_hidden,
        )
        sparse24_transpose_input_to_strided(
            routed_hidden, out=routed_hidden_transposed
        )

        routed_full_stream.wait_stream(current)
        routed_residual_stream.wait_stream(current)
        with torch.cuda.stream(routed_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                routed_hidden_transposed,
                down_full_values,
                down_full_meta,
                input_transposed=True,
                out=routed_down,
                device_config="auto",
            )
        with torch.cuda.stream(routed_residual_stream):
            sparse24_gather_rows_strided_(
                routed_hidden_transposed,
                dense_rows,
                routed_dense_hidden[:dense_count],
            )
            sparse24_cutlass_device_gemm_prepacked(
                routed_dense_hidden,
                down_residual_values,
                down_residual_meta,
                input_transposed=True,
                out=routed_down_residual,
                device_config="auto",
            )
        current.wait_stream(routed_full_stream)
        current.wait_stream(routed_residual_stream)
        return sparse24_transpose_add_routed_residual(
            routed_down,
            routed_down_residual,
            dense_slots,
            dense_count=dense_count,
            out=routed_output,
        )

    def fused_layout_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        fused_full_stream.wait_stream(current)
        fused_residual_stream.wait_stream(current)
        with torch.cuda.stream(fused_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate_routed_values,
                gate_routed_meta,
                dense_slots,
                dense_count=dense_count,
                out=fused_hidden,
                dense_base=fused_dense_base,
                config=control_config,
            )
        with torch.cuda.stream(fused_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                dense_x,
                gate_residual_values,
                gate_residual_meta,
                out=fused_gate_residual,
                config="auto",
                store_mode="vector",
            )
        current.wait_stream(fused_full_stream)
        current.wait_stream(fused_residual_stream)
        sparse24_routed_swiglu_correction_(
            fused_dense_base,
            fused_gate_residual[:dense_count],
            dense_rows,
            fused_hidden,
        )

        if not use_contiguous_down:
            sparse24_transpose_input_to_strided(
                fused_hidden, out=fused_hidden_transposed
            )

        fused_full_stream.wait_stream(current)
        fused_residual_stream.wait_stream(current)
        with torch.cuda.stream(fused_full_stream):
            if use_contiguous_down:
                sparse24_cutlass_device_gemm_prepacked(
                    fused_hidden,
                    down_full_values,
                    down_full_meta,
                    out=fused_down,
                    device_config="auto",
                )
            else:
                sparse24_cutlass_device_gemm_prepacked(
                    fused_hidden_transposed,
                    down_full_values,
                    down_full_meta,
                    input_transposed=True,
                    out=fused_down,
                    device_config="auto",
                )
        with torch.cuda.stream(fused_residual_stream):
            if use_contiguous_down:
                sparse24_gather_rows_(
                    fused_hidden,
                    dense_rows,
                    fused_dense_hidden[:dense_count],
                )
                sparse24_cutlass_device_gemm_prepacked(
                    fused_dense_hidden,
                    down_residual_values,
                    down_residual_meta,
                    out=fused_down_residual,
                    device_config="auto",
                )
            else:
                sparse24_gather_rows_strided_(
                    fused_hidden_transposed,
                    dense_rows,
                    fused_dense_hidden_transposed[:dense_count],
                )
                sparse24_cutlass_device_gemm_prepacked(
                    fused_dense_hidden_transposed,
                    down_residual_values,
                    down_residual_meta,
                    input_transposed=True,
                    out=fused_down_residual,
                    device_config="auto",
                )
        current.wait_stream(fused_full_stream)
        current.wait_stream(fused_residual_stream)
        return sparse24_transpose_add_routed_residual(
            fused_down,
            fused_down_residual,
            dense_slots,
            dense_count=dense_count,
            out=fused_output,
        )

    def fused_correction_gather_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        gather_fused_full_stream.wait_stream(current)
        gather_fused_residual_stream.wait_stream(current)
        with torch.cuda.stream(gather_fused_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate_routed_values,
                gate_routed_meta,
                dense_slots,
                dense_count=dense_count,
                out=gather_fused_hidden,
                dense_base=gather_fused_dense_base,
                config=control_config,
            )
        with torch.cuda.stream(gather_fused_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                dense_x,
                gate_residual_values,
                gate_residual_meta,
                out=gather_fused_gate_residual,
                config="auto",
                store_mode="vector",
            )
        current.wait_stream(gather_fused_full_stream)
        current.wait_stream(gather_fused_residual_stream)
        sparse24_routed_swiglu_correction_gather_(
            gather_fused_dense_base,
            gather_fused_gate_residual[:dense_count],
            dense_rows,
            gather_fused_hidden,
            gather_fused_dense_hidden,
        )

        if not use_contiguous_down:
            sparse24_transpose_input_to_strided(
                gather_fused_hidden, out=gather_fused_hidden_transposed
            )

        gather_fused_full_stream.wait_stream(current)
        gather_fused_residual_stream.wait_stream(current)
        with torch.cuda.stream(gather_fused_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                (
                    gather_fused_hidden
                    if use_contiguous_down
                    else gather_fused_hidden_transposed
                ),
                down_full_values,
                down_full_meta,
                input_transposed=not use_contiguous_down,
                out=gather_fused_down,
                device_config="auto",
            )
        with torch.cuda.stream(gather_fused_residual_stream):
            sparse24_cutlass_device_gemm_prepacked(
                gather_fused_dense_hidden,
                down_residual_values,
                down_residual_meta,
                out=gather_fused_down_residual,
                device_config="auto",
            )
        current.wait_stream(gather_fused_full_stream)
        current.wait_stream(gather_fused_residual_stream)
        return sparse24_transpose_add_routed_residual(
            gather_fused_down,
            gather_fused_down_residual,
            dense_slots,
            dense_count=dense_count,
            out=gather_fused_output,
        )

    def paired_gate_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
        sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
            x,
            gate_routed_values,
            gate_routed_meta,
            dense_slots,
            dense_x,
            gate_residual_values,
            gate_residual_meta,
            dense_count=dense_count,
            full_out=paired_hidden,
            dense_base=paired_dense_base,
            residual_out=paired_gate_residual,
            schedule="interleaved",
            config=config,
        )
        sparse24_routed_swiglu_correction_(
            paired_dense_base,
            paired_gate_residual[:dense_count],
            dense_rows,
            paired_hidden,
        )

        if not use_contiguous_down:
            sparse24_transpose_input_to_strided(
                paired_hidden, out=paired_hidden_transposed
            )

        current = torch.cuda.current_stream()
        paired_full_stream.wait_stream(current)
        paired_residual_stream.wait_stream(current)
        with torch.cuda.stream(paired_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                (
                    paired_hidden
                    if use_contiguous_down
                    else paired_hidden_transposed
                ),
                down_full_values,
                down_full_meta,
                input_transposed=not use_contiguous_down,
                out=paired_down,
                device_config="auto",
            )
        with torch.cuda.stream(paired_residual_stream):
            if use_contiguous_down:
                sparse24_gather_rows_(
                    paired_hidden,
                    dense_rows,
                    paired_dense_hidden[:dense_count],
                )
                residual_input = paired_dense_hidden
            else:
                sparse24_gather_rows_strided_(
                    paired_hidden_transposed,
                    dense_rows,
                    paired_dense_hidden_transposed[:dense_count],
                )
                residual_input = paired_dense_hidden_transposed
            sparse24_cutlass_device_gemm_prepacked(
                residual_input,
                down_residual_values,
                down_residual_meta,
                input_transposed=not use_contiguous_down,
                out=paired_down_residual,
                device_config="auto",
            )
        current.wait_stream(paired_full_stream)
        current.wait_stream(paired_residual_stream)
        return sparse24_transpose_add_routed_residual(
            paired_down,
            paired_down_residual,
            dense_slots,
            dense_count=dense_count,
            out=paired_output,
        )

    def paired_gate_down_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
        sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
            x,
            gate_routed_values,
            gate_routed_meta,
            dense_slots,
            dense_x,
            gate_residual_values,
            gate_residual_meta,
            dense_count=dense_count,
            full_out=paired_hidden,
            dense_base=paired_dense_base,
            residual_out=paired_gate_residual,
            schedule="interleaved",
            config=config,
        )
        sparse24_routed_swiglu_correction_(
            paired_dense_base,
            paired_gate_residual[:dense_count],
            dense_rows,
            paired_hidden,
        )
        sparse24_gather_rows_(
            paired_hidden,
            dense_rows,
            paired_dense_hidden[:dense_count],
        )
        sparse24_cutlass_paired_persistent_gemm_prepacked(
            paired_hidden,
            down_full_values,
            down_full_meta,
            paired_dense_hidden,
            down_residual_values,
            down_residual_meta,
            full_out=paired_both_down,
            residual_out=paired_both_down_residual,
            schedule="interleaved",
            config=paired_down_config,
        )
        return sparse24_transpose_add_routed_residual(
            paired_both_down,
            paired_both_down_residual,
            dense_slots,
            dense_count=dense_count,
            out=paired_both_output,
        )

    def paired_gate_gather_down_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
        sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
            x,
            gate_routed_values,
            gate_routed_meta,
            dense_slots,
            dense_x,
            gate_residual_values,
            gate_residual_meta,
            dense_count=dense_count,
            full_out=paired_hidden,
            dense_base=paired_dense_base,
            residual_out=paired_gate_residual,
            schedule="interleaved",
            config=config,
        )
        sparse24_routed_swiglu_correction_(
            paired_dense_base,
            paired_gate_residual[:dense_count],
            dense_rows,
            paired_hidden,
        )
        sparse24_cutlass_paired_gather_residual_prepacked(
            paired_hidden,
            down_full_values,
            down_full_meta,
            down_residual_values,
            down_residual_meta,
            dense_rows,
            full_out=paired_gather_down,
            residual_out=paired_gather_down_residual,
            schedule="interleaved",
            config=paired_gather_down_config,
        )
        return sparse24_add_indexed_rows_contiguous_(
            paired_gather_down,
            paired_gather_down_residual,
            dense_rows,
        )

    def dense_row_overwrite_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
        current = torch.cuda.current_stream()
        dense_overwrite_full_stream.wait_stream(current)
        dense_overwrite_dense_stream.wait_stream(current)
        with torch.cuda.stream(dense_overwrite_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate_routed_values,
                gate_routed_meta,
                dense_slots,
                dense_count=dense_count,
                out=dense_overwrite_hidden,
                dense_base=dense_overwrite_base,
                config=control_config,
            )
        with torch.cuda.stream(dense_overwrite_dense_stream):
            torch.mm(
                dense_x[:dense_count],
                dense_gate_weight,
                out=dense_overwrite_gate_up,
            )
            torch.ops._C.silu_and_mul(
                dense_overwrite_dense_hidden,
                dense_overwrite_gate_up,
            )
        current.wait_stream(dense_overwrite_full_stream)
        current.wait_stream(dense_overwrite_dense_stream)
        sparse24_copy_indexed_rows_contiguous_(
            dense_overwrite_hidden,
            dense_overwrite_dense_hidden,
            dense_rows,
        )

        dense_overwrite_full_stream.wait_stream(current)
        dense_overwrite_dense_stream.wait_stream(current)
        with torch.cuda.stream(dense_overwrite_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                dense_overwrite_hidden,
                down_full_values,
                down_full_meta,
                out=dense_overwrite_down,
                device_config="auto",
            )
        with torch.cuda.stream(dense_overwrite_dense_stream):
            torch.mm(
                dense_overwrite_dense_hidden,
                dense_down_weight,
                out=dense_overwrite_dense_down,
            )
        current.wait_stream(dense_overwrite_full_stream)
        current.wait_stream(dense_overwrite_dense_stream)
        sparse24_transpose_output_contiguous(
            dense_overwrite_down, out=dense_overwrite_output
        )
        return sparse24_copy_indexed_rows_contiguous_(
            dense_overwrite_output,
            dense_overwrite_dense_down,
            dense_rows,
        )

    dense_expected = dense_mlp().clone()
    expected = baseline_exact().clone()
    routed_actual = routed_exact().clone()
    fused_actual = fused_layout_exact().clone()
    gather_fused_actual = fused_correction_gather_exact().clone()
    paired_actual = paired_gate_exact().clone()
    paired_both_actual = paired_gate_down_exact().clone()
    paired_gather_actual = paired_gate_gather_down_exact().clone()
    dense_overwrite_actual = dense_row_overwrite_exact().clone()
    torch.cuda.synchronize()
    routed_max_abs_diff = float(
        (routed_actual.float() - expected.float()).abs().max().item()
    )
    fused_max_abs_diff = float(
        (fused_actual.float() - expected.float()).abs().max().item()
    )
    gather_fused_max_abs_diff = float(
        (gather_fused_actual.float() - expected.float()).abs().max().item()
    )
    paired_max_abs_diff = float(
        (paired_actual.float() - expected.float()).abs().max().item()
    )
    paired_both_max_abs_diff = float(
        (paired_both_actual.float() - expected.float()).abs().max().item()
    )
    paired_gather_max_abs_diff = float(
        (paired_gather_actual.float() - expected.float()).abs().max().item()
    )
    dense_overwrite_max_abs_diff = float(
        (dense_overwrite_actual.float() - expected.float()).abs().max().item()
    )
    selected_dense_max_abs_diff = float(
        (
            gather_fused_actual[dense_rows.long()].float()
            - dense_expected[dense_rows.long()].float()
        )
        .abs()
        .max()
        .item()
    )
    if not torch.allclose(routed_actual, expected, rtol=5e-2, atol=2e-1):
        raise RuntimeError(
            f"routed MLP mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={routed_max_abs_diff}"
        )
    if not torch.allclose(fused_actual, expected, rtol=5e-2, atol=2e-1):
        raise RuntimeError(
            f"fused-layout MLP mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={fused_max_abs_diff}"
        )
    if not torch.allclose(
        gather_fused_actual, expected, rtol=5e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"fused correction/gather MLP mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={gather_fused_max_abs_diff}"
        )
    if not torch.allclose(paired_actual, expected, rtol=5e-2, atol=2e-1):
        raise RuntimeError(
            f"paired-gate MLP mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={paired_max_abs_diff}"
        )
    if not torch.allclose(
        paired_both_actual, expected, rtol=5e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"paired gate/down MLP mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_both_max_abs_diff}"
        )
    if not torch.allclose(
        paired_gather_actual, expected, rtol=5e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"paired gather Down MLP mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_gather_max_abs_diff}"
        )
    if not torch.allclose(
        dense_overwrite_actual, expected, rtol=5e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"dense-row overwrite MLP mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={dense_overwrite_max_abs_diff}"
        )
    if not torch.allclose(
        gather_fused_actual[dense_rows.long()],
        dense_expected[dense_rows.long()],
        rtol=5e-2,
        atol=2e-1,
    ):
        raise RuntimeError(
            f"selected dense MLP rows mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={selected_dense_max_abs_diff}"
        )

    if profile_iterations > 0:
        profile_functions = {
            "fused_layout": fused_layout_exact,
            "paired_gate": paired_gate_exact,
            "dense_overwrite": dense_row_overwrite_exact,
        }
        profile_fn = profile_functions[profile_variant]
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"speclink_{profile_variant}_mlp")
        for _ in range(profile_iterations):
            profile_fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    baseline_ms, routed_ms = paired_graph_median_ms(
        baseline_exact,
        routed_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    routed_control_ms, fused_ms = paired_graph_median_ms(
        routed_exact,
        fused_layout_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    fused_control_ms, gather_fused_ms = paired_graph_median_ms(
        fused_layout_exact,
        fused_correction_gather_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    paired_control_ms, paired_gate_ms = paired_graph_median_ms(
        fused_layout_exact,
        paired_gate_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    paired_down_control_ms, paired_gate_down_ms = paired_graph_median_ms(
        paired_gate_exact,
        paired_gate_down_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    paired_gather_control_ms, paired_gate_gather_down_ms = (
        paired_graph_median_ms(
            paired_gate_exact,
            paired_gate_gather_down_exact,
            unroll=unroll,
            replays=replays,
            trials=trials,
            graph_warmup_replays=graph_warmup_replays,
        )
    )
    dense_overwrite_control_ms, dense_overwrite_ms = paired_graph_median_ms(
        paired_gate_exact,
        dense_row_overwrite_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    dense_mlp_ms, fused_dense_control_ms = paired_graph_median_ms(
        dense_mlp,
        fused_layout_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    routed_speedup = baseline_ms / routed_ms
    fused_vs_routed = routed_control_ms / fused_ms
    gather_fused_vs_layout = fused_control_ms / gather_fused_ms
    paired_gate_vs_layout = paired_control_ms / paired_gate_ms
    paired_down_vs_gate = paired_down_control_ms / paired_gate_down_ms
    paired_gather_down_vs_gate = (
        paired_gather_control_ms / paired_gate_gather_down_ms
    )
    dense_overwrite_vs_paired = (
        dense_overwrite_control_ms / dense_overwrite_ms
    )
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "route_mode": route_mode,
        "dense_rows": dense_count,
        "dense_fraction_actual": dense_count / rows,
        "config": config,
        "paired_down_config": paired_down_config,
        "paired_gather_down_config": paired_gather_down_config,
        "baseline_exact_mlp_ms": baseline_ms,
        "routed_exact_mlp_ms": routed_ms,
        "routed_mlp_speedup": routed_speedup,
        "routed_max_abs_diff": routed_max_abs_diff,
        "fused_layout_routed_control_ms": routed_control_ms,
        "fused_layout_mlp_ms": fused_ms,
        "fused_layout_speedup_vs_routed": fused_vs_routed,
        "fused_layout_speedup_vs_baseline": routed_speedup * fused_vs_routed,
        "fused_layout_max_abs_diff": fused_max_abs_diff,
        "fused_down_layout": "contiguous" if use_contiguous_down else "b_row",
        "fused_correction_gather_control_ms": fused_control_ms,
        "fused_correction_gather_mlp_ms": gather_fused_ms,
        "fused_correction_gather_speedup_vs_layout": (
            gather_fused_vs_layout
        ),
        "fused_correction_gather_speedup_vs_routed": (
            fused_vs_routed * gather_fused_vs_layout
        ),
        "fused_correction_gather_speedup_vs_baseline": (
            routed_speedup * fused_vs_routed * gather_fused_vs_layout
        ),
        "fused_correction_gather_max_abs_diff": (
            gather_fused_max_abs_diff
        ),
        "paired_gate_control_ms": paired_control_ms,
        "paired_gate_mlp_ms": paired_gate_ms,
        "paired_gate_speedup_vs_layout": paired_gate_vs_layout,
        "paired_gate_max_abs_diff": paired_max_abs_diff,
        "paired_down_control_ms": paired_down_control_ms,
        "paired_gate_down_mlp_ms": paired_gate_down_ms,
        "paired_down_speedup_vs_paired_gate": paired_down_vs_gate,
        "paired_gate_down_speedup_vs_layout": (
            paired_gate_vs_layout * paired_down_vs_gate
        ),
        "paired_gate_down_max_abs_diff": paired_both_max_abs_diff,
        "paired_gather_down_control_ms": paired_gather_control_ms,
        "paired_gate_gather_down_mlp_ms": paired_gate_gather_down_ms,
        "paired_gather_down_speedup_vs_paired_gate": (
            paired_gather_down_vs_gate
        ),
        "paired_gather_down_speedup_vs_layout": (
            paired_gate_vs_layout * paired_gather_down_vs_gate
        ),
        "paired_gather_down_max_abs_diff": paired_gather_max_abs_diff,
        "dense_overwrite_control_ms": dense_overwrite_control_ms,
        "dense_overwrite_mlp_ms": dense_overwrite_ms,
        "dense_overwrite_speedup_vs_paired_gate": (
            dense_overwrite_vs_paired
        ),
        "dense_overwrite_speedup_vs_layout": (
            paired_gate_vs_layout * dense_overwrite_vs_paired
        ),
        "dense_overwrite_max_abs_diff": dense_overwrite_max_abs_diff,
        "dense_overwrite_weight_bytes": (
            dense_gate_weight.numel() + dense_down_weight.numel()
        )
        * dense_gate_weight.element_size(),
        "dense_mlp_ms": dense_mlp_ms,
        "fused_layout_dense_control_ms": fused_dense_control_ms,
        "fused_layout_speedup_vs_dense_mlp": (
            dense_mlp_ms / fused_dense_control_ms
        ),
        "selected_dense_max_abs_diff": selected_dense_max_abs_diff,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    configs = list(dict.fromkeys(str(row["config"]) for row in rows))
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    line_styles = ["-", "--"]
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        selected = [row for row in rows if row["model"] == model]
        for config_index, config in enumerate(configs):
            for k in sorted({int(row["K"]) for row in selected}):
                by_key = [
                    row
                    for row in selected
                    if row["config"] == config and int(row["K"]) == k
                ]
                if not by_key:
                    continue
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["fused_layout_speedup_vs_routed"])
                        * float(
                            row[
                                "fused_correction_gather_speedup_vs_layout"
                            ]
                        )
                        for row in by_key
                    ],
                    marker="o",
                    color=colors[k],
                    linestyle=line_styles[config_index % len(line_styles)],
                    label=f"K={k}, {config}",
                )
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["paired_gate_speedup_vs_layout"])
                        for row in by_key
                    ],
                    marker="s",
                    color=colors[k],
                    linestyle=":",
                    label=f"K={k}, paired gate",
                )
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["paired_gate_down_speedup_vs_layout"])
                        for row in by_key
                    ],
                    marker="^",
                    color=colors[k],
                    linestyle="--",
                    label=f"K={k}, paired gate+down",
                )
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["dense_overwrite_speedup_vs_layout"])
                        for row in by_key
                    ],
                    marker="D",
                    color=colors[k],
                    linestyle="-.",
                    label=f"K={k}, dense-row overwrite",
                )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model}: exact MLP")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Speedup vs fused-layout routed MLP")
        axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_csv_strings, default=tuple(MODELS))
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--k-values", type=parse_csv_ints, default=DEFAULT_K_VALUES)
    parser.add_argument("--configs", type=parse_csv_strings, default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--paired-down-config",
        choices=SUPPORTED_PAIRED_DOWN_CONFIGS,
        default="auto",
    )
    parser.add_argument(
        "--paired-gather-down-config",
        choices=SUPPORTED_PAIRED_GATHER_DOWN_CONFIGS,
        default="128x64_full_128x64_residual_contiguous",
    )
    parser.add_argument(
        "--route-mode",
        choices=("ratio_total", "bonus_dense", "draft_ratio_cap"),
        default="ratio_total",
    )
    parser.add_argument("--dense-fraction", type=float, default=0.125)
    parser.add_argument("--dense-cap", type=int, default=-1)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=30)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument(
        "--profile-variant",
        choices=("fused_layout", "paired_gate", "dense_overwrite"),
        default="paired_gate",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_models = [model for model in args.models if model not in MODELS]
    invalid_configs = [
        config for config in args.configs if config not in SUPPORTED_CONFIGS
    ]
    if invalid_models or invalid_configs:
        raise ValueError(
            f"unsupported models={invalid_models}, configs={invalid_configs}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    results: list[dict[str, object]] = []
    for model in args.models:
        model_width = int(MODELS[model]["hidden"])
        intermediate = int(MODELS[model]["intermediate"])
        gate_weights = prepare_gate_up_weights(
            model_width, intermediate, generator
        )
        down_weights = prepare_down_weights(
            intermediate, model_width, generator
        )
        for batch_size in args.batch_sizes:
            for k in args.k_values:
                for config in args.configs:
                    result = run_case(
                        model,
                        batch_size,
                        k,
                        config,
                        gate_weights,
                        down_weights,
                        paired_down_config=args.paired_down_config,
                        paired_gather_down_config=(
                            args.paired_gather_down_config
                        ),
                        route_mode=args.route_mode,
                        dense_fraction=args.dense_fraction,
                        dense_cap=args.dense_cap,
                        min_dense_per_request=args.min_dense_per_request,
                        generator=generator,
                        unroll=args.unroll,
                        replays=args.replays,
                        trials=args.trials,
                        graph_warmup_replays=args.graph_warmup_replays,
                        profile_iterations=args.profile_iterations,
                        profile_variant=args.profile_variant,
                    )
                    results.append(result)
                    print(
                        f"{model} bs={batch_size} K={k} "
                        f"route={args.route_mode} "
                        f"dense={int(result['dense_rows'])}/{int(result['rows'])} "
                        f"config={config} "
                        f"paired_down={args.paired_down_config} "
                        "paired_gather_down="
                        f"{args.paired_gather_down_config} "
                        f"layout={result['fused_down_layout']} "
                        "fused_gather_vs_layout="
                        f"{float(result['fused_correction_gather_speedup_vs_layout']):.3f}x "
                        "paired_gate_vs_layout="
                        f"{float(result['paired_gate_speedup_vs_layout']):.3f}x "
                        "paired_down_vs_gate="
                        f"{float(result['paired_down_speedup_vs_paired_gate']):.3f}x "
                        "paired_gather_down_vs_gate="
                        f"{float(result['paired_gather_down_speedup_vs_paired_gate']):.3f}x "
                        "dense_overwrite_vs_paired="
                        f"{float(result['dense_overwrite_speedup_vs_paired_gate']):.3f}x",
                        flush=True,
                    )
        del gate_weights, down_weights
        torch.cuda.empty_cache()

    csv_path = args.output_root / "routed_mlp_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "routed_mlp_speedup.png", results)
    print(args.output_root)


if __name__ == "__main__":
    main()
