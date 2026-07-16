#!/usr/bin/env python3
"""Benchmark exact mixed-row gate/up with a routed sparse SwiGLU epilogue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm import _custom_ops as vllm_ops  # noqa: E402
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
    sparse24_add_indexed_rows_strided_,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
    sparse24_cutlass_grouped_owner_swiglu_prepacked,
    sparse24_cutlass_paired_fused_routed_swiglu_prepacked,
    sparse24_cutlass_paired_gather_routed_swiglu_prepacked,
    sparse24_cutlass_paired_persistent_routed_swiglu_prepacked,
    sparse24_cutlass_paired_self_contained_routed_swiglu_prepacked,
    sparse24_cutlass_residual_correction_swiglu_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_,
    sparse24_routed_swiglu_correction_,
    sparse24_silu_and_mul_transposed_to_contiguous,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
DEFAULT_CONFIGS = (
    "256x64x64_s3",
    "256x64x64_s3_sw4",
)
SUPPORTED_CONFIGS = (
    *DEFAULT_CONFIGS,
    "256x32x64_s3_sw4",
    "256x64x64_s2_sw4",
)
SUPPORTED_PAIRED_CONFIGS = (
    *SUPPORTED_CONFIGS,
    "256x64_full_256x32_residual_s3_sw4",
)
SUPPORTED_PAIRED_FUSED_CONFIGS = (
    *SUPPORTED_CONFIGS,
    "256x64x64_s3_sw4_fast_silu",
    "256x64_full_256x32_residual_s3_sw4",
)
DEFAULT_PAIRED_SCHEDULES = ("partitioned", "interleaved")
SUPPORTED_PAIRED_FUSED_SCHEDULES = ("partitioned", "shared_phased")


def prepare_gate_up_weights(
    model_width: int,
    intermediate: int,
    generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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

    full_packed = pack_24(weight24, layout="n_major")
    full_values, full_meta = prepare_cutlass_sparse24_device_gemm(
        full_packed.values,
        full_packed.meta,
        layout=full_packed.layout,
        K=model_width,
    )
    routed_values, routed_meta = prepare_cutlass_sparse24_gate_up_swiglu(
        full_packed.values,
        full_packed.meta,
        layout=full_packed.layout,
        K=model_width,
    )
    residual_packed = pack_24(residual24, layout="n_major")
    residual_values, residual_meta = prepare_cutlass_sparse24_device_gemm(
        residual_packed.values,
        residual_packed.meta,
        layout=residual_packed.layout,
        K=model_width,
    )
    residual_routed_values, residual_routed_meta = (
        prepare_cutlass_sparse24_gate_up_swiglu(
            residual_packed.values,
            residual_packed.meta,
            layout=residual_packed.layout,
            K=model_width,
        )
    )
    residual_fp8_t, residual_fp8_scale = vllm_ops.scaled_fp8_quant(
        residual24.t().contiguous(),
        use_per_token_if_dynamic=True,
    )
    residual_fp8 = residual_fp8_t.t()
    return (
        weight,
        full_values,
        full_meta,
        routed_values,
        routed_meta,
        residual_values,
        residual_meta,
        residual_routed_values,
        residual_routed_meta,
        residual_fp8,
        residual_fp8_scale,
    )


def run_case(
    model: str,
    batch_size: int,
    k: int,
    config: str,
    paired_schedule: str,
    paired_worker_blocks: int,
    weights: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    dense_fraction: float,
    min_dense_per_request: int,
    route_mode: str,
    dense_cap: int,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
    benchmark_fp8_residual: bool,
    paired_control_config: str,
    paired_config: str,
    paired_fused_control_config: str,
    paired_fused_config: str,
    paired_fused_schedule: str,
    grouped_owner_tiles: int,
) -> dict[str, object]:
    model_width = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    gate_up_size = 2 * intermediate
    (
        dense_weight,
        full_values,
        full_meta,
        routed_values,
        routed_meta,
        residual_values,
        residual_meta,
        residual_routed_values,
        residual_routed_meta,
        residual_fp8,
        residual_fp8_scale,
    ) = weights

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
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
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
    dense_input = torch.zeros(
        (dense_run, model_width), device="cuda", dtype=torch.float16
    )

    baseline_gate_up = torch.empty_strided(
        (rows, gate_up_size),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_residual = torch.empty_strided(
        (dense_run, gate_up_size),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )

    routed_hidden = torch.empty_like(baseline_hidden)
    routed_dense_base = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    routed_residual = torch.empty(
        (dense_run, gate_up_size), device="cuda", dtype=torch.float16
    )
    routed_residual_workspace = torch.empty(
        (gate_up_size, dense_run), device="cuda", dtype=torch.float16
    )
    paired_hidden = torch.empty_like(routed_hidden)
    paired_dense_base = torch.empty_like(routed_dense_base)
    paired_residual = torch.empty_like(routed_residual)
    paired_control_hidden = torch.empty_like(routed_hidden)
    paired_control_dense_base = torch.empty_like(routed_dense_base)
    paired_control_residual = torch.empty_like(routed_residual)
    self_contained_hidden = torch.empty_like(routed_hidden)
    self_contained_dense_base = torch.empty_like(routed_dense_base)
    grouped_owner_hidden = torch.empty_like(routed_hidden)
    grouped_owner_dense_base = torch.empty_like(routed_dense_base)
    self_contained_config = (
        "256x32x64_s3_sw4"
        if paired_fused_config == "256x32x64_s3_sw4"
        else "256x64x64_s3_sw4"
    )
    fused_hidden = torch.empty_like(routed_hidden)
    fused_dense_base = torch.empty_like(routed_dense_base)
    fused_feature_counters = torch.zeros(
        gate_up_size // 256, device="cuda", dtype=torch.int32
    )
    fused_control_hidden = torch.empty_like(routed_hidden)
    fused_control_dense_base = torch.empty_like(routed_dense_base)
    fused_control_feature_counters = torch.zeros_like(
        fused_feature_counters
    )
    compact_fused_hidden = torch.empty_like(routed_hidden)
    compact_fused_dense_base = torch.empty_like(routed_dense_base)
    compact_fused_feature_counters = torch.zeros_like(
        fused_feature_counters
    )
    standalone_correction_hidden = torch.empty_like(routed_hidden)
    overwrite_hidden = torch.empty_like(routed_hidden)
    overwrite_dense_gate = torch.empty(
        (dense_count, gate_up_size), device="cuda", dtype=torch.float16
    )
    overwrite_dense_hidden = torch.empty(
        (dense_count, intermediate), device="cuda", dtype=torch.float16
    )
    fp8_hidden = torch.empty_like(routed_hidden)
    fp8_dense_base = torch.empty_like(routed_dense_base)
    fp8_residual = torch.empty_like(routed_residual)
    fp8_input = torch.empty(
        dense_input.shape,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    fp8_token_scale = torch.empty(
        (dense_run, 1), device="cuda", dtype=torch.float32
    )
    fp8_static_scale = (
        x.float().abs().amax().clamp_min(1.0e-6) / 448.0
    ).reshape(1)

    baseline_full_stream = torch.cuda.Stream()
    baseline_residual_stream = torch.cuda.Stream()
    routed_full_stream = torch.cuda.Stream()
    routed_residual_stream = torch.cuda.Stream()

    def baseline_full_stage() -> torch.Tensor:
        sparse24_cutlass_device_gemm_prepacked(
            x,
            full_values,
            full_meta,
            out=baseline_gate_up,
            device_config="auto",
        )
        return sparse24_silu_and_mul_transposed_to_contiguous(
            baseline_gate_up, out=baseline_hidden
        )

    def routed_full_stage() -> torch.Tensor:
        sparse24_cutlass_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            dense_slots,
            dense_count=dense_count,
            out=routed_hidden,
            dense_base=routed_dense_base,
            config=config,
        )
        return routed_hidden

    def baseline_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        baseline_full_stream.wait_stream(current)
        baseline_residual_stream.wait_stream(current)
        with torch.cuda.stream(baseline_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                full_values,
                full_meta,
                out=baseline_gate_up,
                device_config="auto",
            )
        with torch.cuda.stream(baseline_residual_stream):
            sparse24_gather_rows_(
                x, dense_rows, dense_input[:dense_count]
            )
            sparse24_cutlass_device_gemm_prepacked(
                dense_input,
                residual_values,
                residual_meta,
                out=baseline_residual,
                device_config="auto",
            )
        current.wait_stream(baseline_full_stream)
        current.wait_stream(baseline_residual_stream)
        sparse24_add_indexed_rows_strided_(
            baseline_gate_up,
            baseline_residual[:dense_count],
            dense_rows,
        )
        return sparse24_silu_and_mul_transposed_to_contiguous(
            baseline_gate_up, out=baseline_hidden
        )

    def routed_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        routed_full_stream.wait_stream(current)
        routed_residual_stream.wait_stream(current)
        with torch.cuda.stream(routed_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                routed_values,
                routed_meta,
                dense_slots,
                dense_count=dense_count,
                out=routed_hidden,
                dense_base=routed_dense_base,
                config=config,
            )
        with torch.cuda.stream(routed_residual_stream):
            sparse24_gather_rows_(
                x, dense_rows, dense_input[:dense_count]
            )
            sparse24_cutlass_device_gemm_prepacked(
                dense_input,
                residual_values,
                residual_meta,
                contiguous_output=True,
                out=routed_residual,
                workspace=routed_residual_workspace,
                device_config="auto",
            )
        current.wait_stream(routed_full_stream)
        current.wait_stream(routed_residual_stream)
        return sparse24_routed_swiglu_correction_(
            routed_dense_base,
            routed_residual[:dense_count],
            dense_rows,
            routed_hidden,
        )

    def paired_persistent_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
        sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            dense_slots,
            dense_input,
            residual_values,
            residual_meta,
            dense_count=dense_count,
            full_out=paired_hidden,
            dense_base=paired_dense_base,
            residual_out=paired_residual,
            schedule=paired_schedule,
            config=paired_config,
            worker_blocks=paired_worker_blocks,
        )
        return sparse24_routed_swiglu_correction_(
            paired_dense_base,
            paired_residual[:dense_count],
            dense_rows,
            paired_hidden,
        )

    def paired_control_exact() -> torch.Tensor:
        if not paired_control_config:
            raise RuntimeError("paired control config is not set")
        sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
        sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            dense_slots,
            dense_input,
            residual_values,
            residual_meta,
            dense_count=dense_count,
            full_out=paired_control_hidden,
            dense_base=paired_control_dense_base,
            residual_out=paired_control_residual,
            schedule=paired_schedule,
            config=paired_control_config,
            worker_blocks=paired_worker_blocks,
        )
        return sparse24_routed_swiglu_correction_(
            paired_control_dense_base,
            paired_control_residual[:dense_count],
            dense_rows,
            paired_control_hidden,
        )

    def paired_gather_exact() -> torch.Tensor:
        sparse24_cutlass_paired_gather_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            dense_slots,
            dense_rows,
            residual_values,
            residual_meta,
            full_out=paired_hidden,
            dense_base=paired_dense_base,
            residual_out=paired_residual[:dense_count],
            schedule=paired_schedule,
            config=paired_config,
            worker_blocks=paired_worker_blocks,
        )
        return sparse24_routed_swiglu_correction_(
            paired_dense_base,
            paired_residual[:dense_count],
            dense_rows,
            paired_hidden,
        )

    def paired_fused_exact() -> torch.Tensor:
        sparse24_cutlass_paired_fused_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            residual_routed_values,
            residual_routed_meta,
            dense_rows,
            dense_slots,
            out=fused_hidden,
            dense_base=fused_dense_base,
            feature_counters=fused_feature_counters,
            config=paired_fused_config,
            worker_blocks=paired_worker_blocks,
            schedule=paired_fused_schedule,
        )
        return fused_hidden

    def paired_fused_control_exact() -> torch.Tensor:
        if not paired_fused_control_config:
            raise RuntimeError("paired fused control config is not set")
        sparse24_cutlass_paired_fused_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            residual_routed_values,
            residual_routed_meta,
            dense_rows,
            dense_slots,
            out=fused_control_hidden,
            dense_base=fused_control_dense_base,
            feature_counters=fused_control_feature_counters,
            config=paired_fused_control_config,
            worker_blocks=paired_worker_blocks,
            schedule=paired_fused_schedule,
        )
        return fused_control_hidden

    def self_contained_exact() -> torch.Tensor:
        sparse24_cutlass_paired_self_contained_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            residual_routed_values,
            residual_routed_meta,
            dense_rows,
            dense_slots,
            out=self_contained_hidden,
            dense_base=self_contained_dense_base,
            config=self_contained_config,
            worker_blocks=paired_worker_blocks,
        )
        return self_contained_hidden

    def grouped_owner_exact() -> torch.Tensor:
        sparse24_cutlass_grouped_owner_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            residual_routed_values,
            residual_routed_meta,
            dense_rows,
            dense_slots,
            out=grouped_owner_hidden,
            dense_base=grouped_owner_dense_base,
            group_tiles=grouped_owner_tiles,
            config="256x32x64_s3_sw4",
        )
        return grouped_owner_hidden

    def paired_compact_fused_exact() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
        sparse24_cutlass_paired_fused_routed_swiglu_prepacked(
            x,
            routed_values,
            routed_meta,
            residual_routed_values,
            residual_routed_meta,
            dense_rows,
            dense_slots,
            compact_residual_x=dense_input,
            out=compact_fused_hidden,
            dense_base=compact_fused_dense_base,
            feature_counters=compact_fused_feature_counters,
            config=paired_fused_config,
            worker_blocks=paired_worker_blocks,
            schedule=paired_fused_schedule,
        )
        return compact_fused_hidden

    def dense_overwrite_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        routed_full_stream.wait_stream(current)
        routed_residual_stream.wait_stream(current)
        with torch.cuda.stream(routed_full_stream):
            sparse24_cutlass_gate_up_swiglu_prepacked(
                x,
                routed_values,
                routed_meta,
                out=overwrite_hidden,
                config=config,
            )
        with torch.cuda.stream(routed_residual_stream):
            sparse24_gather_rows_(
                x, dense_rows, dense_input[:dense_count]
            )
            torch.mm(
                dense_input[:dense_count],
                dense_weight,
                out=overwrite_dense_gate,
            )
            torch.ops._C.silu_and_mul(
                overwrite_dense_hidden, overwrite_dense_gate
            )
        current.wait_stream(routed_full_stream)
        current.wait_stream(routed_residual_stream)
        return sparse24_copy_indexed_rows_contiguous_(
            overwrite_hidden,
            overwrite_dense_hidden,
            dense_rows,
        )

    def fp8_residual_exact(dynamic_per_token: bool) -> torch.Tensor:
        current = torch.cuda.current_stream()
        routed_full_stream.wait_stream(current)
        routed_residual_stream.wait_stream(current)
        with torch.cuda.stream(routed_full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                routed_values,
                routed_meta,
                dense_slots,
                dense_count=dense_count,
                out=fp8_hidden,
                dense_base=fp8_dense_base,
                config=config,
            )
        with torch.cuda.stream(routed_residual_stream):
            sparse24_gather_rows_(
                x, dense_rows, dense_input[:dense_count]
            )
            if dynamic_per_token:
                torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                    fp8_input,
                    dense_input,
                    fp8_token_scale,
                    None,
                )
                input_scale = fp8_token_scale
            else:
                vllm_ops.scaled_fp8_quant(
                    dense_input,
                    fp8_static_scale,
                    output=fp8_input,
                )
                input_scale = fp8_static_scale
            torch.ops._C.cutlass_scaled_mm(
                fp8_residual,
                fp8_input,
                residual_fp8,
                input_scale,
                residual_fp8_scale,
                None,
            )
        current.wait_stream(routed_full_stream)
        current.wait_stream(routed_residual_stream)
        return sparse24_routed_swiglu_correction_(
            fp8_dense_base,
            fp8_residual[:dense_count],
            dense_rows,
            fp8_hidden,
        )

    expected = baseline_exact().clone()
    actual = routed_exact().clone()
    paired_actual = paired_persistent_exact().clone()
    paired_control_actual = (
        paired_control_exact().clone() if paired_control_config else None
    )
    paired_gather_actual = paired_gather_exact().clone()
    self_contained_actual = self_contained_exact().clone()
    grouped_owner_actual = grouped_owner_exact().clone()
    paired_fused_actual = paired_fused_exact().clone()
    paired_fused_control_actual = (
        paired_fused_control_exact().clone()
        if paired_fused_control_config
        else None
    )
    paired_compact_fused_actual = paired_compact_fused_exact().clone()
    standalone_correction_actual = None
    if config in {"256x64x64_s3_sw4", "256x32x64_s3_sw4"}:
        standalone_correction_hidden.copy_(paired_fused_actual)
        sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
        sparse24_cutlass_residual_correction_swiglu_prepacked(
            dense_input,
            residual_routed_values,
            residual_routed_meta,
            fused_dense_base,
            dense_rows,
            standalone_correction_hidden,
            config=config,
        )
        standalone_correction_actual = standalone_correction_hidden.clone()
    benchmark_dense_overwrite = config != "256x64x64_s2_sw4"
    dense_overwrite_actual = (
        dense_overwrite_exact().clone() if benchmark_dense_overwrite else None
    )
    fp8_static_actual = (
        fp8_residual_exact(False).clone()
        if benchmark_fp8_residual
        else None
    )
    fp8_token_actual = (
        fp8_residual_exact(True).clone()
        if benchmark_fp8_residual
        else None
    )
    torch.cuda.synchronize()
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    paired_max_abs_diff = float(
        (paired_actual.float() - expected.float()).abs().max().item()
    )
    paired_control_max_abs_diff = (
        float(
            (paired_control_actual.float() - expected.float())
            .abs()
            .max()
            .item()
        )
        if paired_control_actual is not None
        else None
    )
    paired_gather_max_abs_diff = float(
        (paired_gather_actual.float() - expected.float()).abs().max().item()
    )
    self_contained_max_abs_diff = float(
        (self_contained_actual.float() - expected.float()).abs().max().item()
    )
    grouped_owner_max_abs_diff = float(
        (grouped_owner_actual.float() - expected.float()).abs().max().item()
    )
    paired_base_max_abs_diff = float(
        (paired_dense_base.float() - routed_dense_base.float())
        .abs()
        .max()
        .item()
    )
    paired_residual_max_abs_diff = float(
        (
            paired_residual[:dense_count].float()
            - routed_residual[:dense_count].float()
        )
        .abs()
        .max()
        .item()
    )
    paired_fused_max_abs_diff = float(
        (paired_fused_actual.float() - expected.float()).abs().max().item()
    )
    paired_fused_control_max_abs_diff = (
        float(
            (paired_fused_control_actual.float() - expected.float())
            .abs()
            .max()
            .item()
        )
        if paired_fused_control_actual is not None
        else None
    )
    paired_fused_vs_persistent_max_abs_diff = float(
        (paired_fused_actual.float() - paired_actual.float())
        .abs()
        .max()
        .item()
    )
    paired_fused_dense_max_abs_diff = float(
        (
            paired_fused_actual[dense_rows.long()].float()
            - expected[dense_rows.long()].float()
        )
        .abs()
        .max()
        .item()
    )
    sparse_mask = torch.ones(rows, device="cuda", dtype=torch.bool)
    sparse_mask[dense_rows.long()] = False
    paired_fused_sparse_max_abs_diff = float(
        (
            paired_fused_actual[sparse_mask].float()
            - expected[sparse_mask].float()
        )
        .abs()
        .max()
        .item()
    )
    paired_fused_base_max_abs_diff = float(
        (fused_dense_base.float() - routed_dense_base.float())
        .abs()
        .max()
        .item()
    )
    paired_fused_counter_max = int(
        fused_feature_counters.abs().max().item()
    )
    paired_fused_control_counter_max = int(
        fused_control_feature_counters.abs().max().item()
    )
    paired_compact_fused_max_abs_diff = float(
        (paired_compact_fused_actual.float() - expected.float())
        .abs()
        .max()
        .item()
    )
    paired_compact_fused_counter_max = int(
        compact_fused_feature_counters.abs().max().item()
    )
    standalone_correction_max_abs_diff = None
    fused_vs_standalone_max_abs_diff = None
    if standalone_correction_actual is not None:
        standalone_correction_max_abs_diff = float(
            (
                standalone_correction_actual[dense_rows.long()].float()
                - expected[dense_rows.long()].float()
            )
            .abs()
            .max()
            .item()
        )
        fused_vs_standalone_max_abs_diff = float(
            (
                paired_fused_actual[dense_rows.long()].float()
                - standalone_correction_actual[dense_rows.long()].float()
            )
            .abs()
            .max()
            .item()
        )
    dense_overwrite_max_abs_diff = (
        float(
            (dense_overwrite_actual.float() - expected.float())
            .abs()
            .max()
            .item()
        )
        if dense_overwrite_actual is not None
        else None
    )
    if not torch.allclose(actual, expected, rtol=3e-2, atol=1e-1):
        raise RuntimeError(
            f"routed SwiGLU mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={max_abs_diff}"
        )
    if not torch.allclose(
        paired_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"paired routed SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: max_abs_diff={paired_max_abs_diff}"
        )
    if paired_control_actual is not None and not torch.allclose(
        paired_control_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"paired control SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_control_max_abs_diff}"
        )
    if not torch.allclose(
        paired_gather_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"paired gather routed SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_gather_max_abs_diff}"
        )
    if not torch.allclose(
        self_contained_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"self-contained routed SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={self_contained_max_abs_diff}"
        )
    if not torch.allclose(
        grouped_owner_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"grouped-owner routed SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={grouped_owner_max_abs_diff}"
        )
    if not torch.allclose(
        paired_fused_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"paired fused routed SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_fused_max_abs_diff}, "
            f"vs_persistent={paired_fused_vs_persistent_max_abs_diff}, "
            f"persistent={paired_max_abs_diff}, "
            f"dense={paired_fused_dense_max_abs_diff}, "
            f"sparse={paired_fused_sparse_max_abs_diff}, "
            f"base={paired_fused_base_max_abs_diff}, "
            f"paired_base={paired_base_max_abs_diff}, "
            f"paired_residual={paired_residual_max_abs_diff}, "
            f"counter={paired_fused_counter_max}, "
            f"standalone={standalone_correction_max_abs_diff}, "
            f"vs_standalone={fused_vs_standalone_max_abs_diff}"
        )
    if paired_fused_control_actual is not None and not torch.allclose(
        paired_fused_control_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"paired fused control SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_fused_control_max_abs_diff}"
        )
    if paired_fused_counter_max != 0:
        raise RuntimeError(
            "paired fused routed SwiGLU did not reset feature counters: "
            f"max={paired_fused_counter_max}"
        )
    if paired_fused_control_counter_max != 0:
        raise RuntimeError(
            "paired fused control did not reset feature counters: "
            f"max={paired_fused_control_counter_max}"
        )
    if not torch.allclose(
        paired_compact_fused_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"paired compact fused routed SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={paired_compact_fused_max_abs_diff}"
        )
    if paired_compact_fused_counter_max != 0:
        raise RuntimeError(
            "paired compact fused routed SwiGLU did not reset counters: "
            f"max={paired_compact_fused_counter_max}"
        )
    if dense_overwrite_actual is not None and not torch.allclose(
        dense_overwrite_actual, expected, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"dense-overwrite SwiGLU mismatch for {model} "
            f"bs={batch_size} K={k}: "
            f"max_abs_diff={dense_overwrite_max_abs_diff}"
        )

    fp8_static_max_abs_diff = None
    fp8_token_max_abs_diff = None
    if fp8_static_actual is not None:
        fp8_static_max_abs_diff = float(
            (fp8_static_actual.float() - expected.float()).abs().max().item()
        )
    if fp8_token_actual is not None:
        fp8_token_max_abs_diff = float(
            (fp8_token_actual.float() - expected.float()).abs().max().item()
        )

    baseline_ms, routed_ms = paired_graph_median_ms(
        baseline_exact,
        routed_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    baseline_full_ms, routed_full_ms = paired_graph_median_ms(
        baseline_full_stage,
        routed_full_stage,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    routed_control_ms, paired_persistent_ms = paired_graph_median_ms(
        routed_exact,
        paired_persistent_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    if paired_control_config:
        paired_direct_control_ms, paired_direct_candidate_ms = (
            paired_graph_median_ms(
                paired_control_exact,
                paired_persistent_exact,
                unroll=unroll,
                replays=replays,
                trials=trials,
                graph_warmup_replays=graph_warmup_replays,
            )
        )
    else:
        paired_direct_control_ms = float("nan")
        paired_direct_candidate_ms = float("nan")
    paired_gather_control_ms, paired_gather_ms = paired_graph_median_ms(
        paired_persistent_exact,
        paired_gather_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    self_contained_control_ms, self_contained_ms = paired_graph_median_ms(
        paired_persistent_exact,
        self_contained_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    grouped_owner_control_ms, grouped_owner_ms = paired_graph_median_ms(
        paired_persistent_exact,
        grouped_owner_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    paired_control_ms, paired_fused_ms = paired_graph_median_ms(
        paired_persistent_exact,
        paired_fused_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    if paired_fused_control_config:
        (
            paired_fused_direct_control_ms,
            paired_fused_direct_candidate_ms,
        ) = paired_graph_median_ms(
            paired_fused_control_exact,
            paired_fused_exact,
            unroll=unroll,
            replays=replays,
            trials=trials,
            graph_warmup_replays=graph_warmup_replays,
        )
    else:
        paired_fused_direct_control_ms = float("nan")
        paired_fused_direct_candidate_ms = float("nan")
    compact_control_ms, paired_compact_fused_ms = paired_graph_median_ms(
        paired_persistent_exact,
        paired_compact_fused_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    torch.cuda.synchronize()
    paired_fused_counter_max_after_graph = int(
        fused_feature_counters.abs().max().item()
    )
    paired_fused_control_counter_max_after_graph = int(
        fused_control_feature_counters.abs().max().item()
    )
    paired_compact_fused_counter_max_after_graph = int(
        compact_fused_feature_counters.abs().max().item()
    )
    if paired_fused_counter_max_after_graph != 0:
        raise RuntimeError(
            "paired fused routed SwiGLU graph replay left nonzero counters: "
            f"max={paired_fused_counter_max_after_graph}"
        )
    if paired_fused_control_counter_max_after_graph != 0:
        raise RuntimeError(
            "paired fused control graph replay left nonzero counters: "
            f"max={paired_fused_control_counter_max_after_graph}"
        )
    if paired_compact_fused_counter_max_after_graph != 0:
        raise RuntimeError(
            "paired compact fused routed SwiGLU graph replay left nonzero "
            f"counters: max={paired_compact_fused_counter_max_after_graph}"
        )
    dense_overwrite_control_ms = None
    dense_overwrite_ms = None
    if benchmark_dense_overwrite:
        dense_overwrite_control_ms, dense_overwrite_ms = paired_graph_median_ms(
            routed_exact,
            dense_overwrite_exact,
            unroll=unroll,
            replays=replays,
            trials=trials,
            graph_warmup_replays=graph_warmup_replays,
        )
    fp8_static_control_ms = None
    fp8_static_ms = None
    fp8_token_control_ms = None
    fp8_token_ms = None
    if benchmark_fp8_residual:
        fp8_static_control_ms, fp8_static_ms = paired_graph_median_ms(
            routed_exact,
            lambda: fp8_residual_exact(False),
            unroll=unroll,
            replays=replays,
            trials=trials,
            graph_warmup_replays=graph_warmup_replays,
        )
        fp8_token_control_ms, fp8_token_ms = paired_graph_median_ms(
            routed_exact,
            lambda: fp8_residual_exact(True),
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
        "route_mode": route_mode,
        "dense_cap": dense_cap,
        "config": config,
        "paired_control_config": paired_control_config,
        "paired_config": paired_config,
        "paired_fused_control_config": paired_fused_control_config,
        "paired_fused_config": paired_fused_config,
        "paired_fused_schedule": paired_fused_schedule,
        "paired_schedule": paired_schedule,
        "paired_worker_blocks": paired_worker_blocks,
        "baseline_exact_ms": baseline_ms,
        "routed_exact_ms": routed_ms,
        "routed_exact_speedup": baseline_ms / routed_ms,
        "baseline_full_activation_ms": baseline_full_ms,
        "routed_full_stage_ms": routed_full_ms,
        "routed_full_stage_speedup": baseline_full_ms / routed_full_ms,
        "max_abs_diff": max_abs_diff,
        "paired_persistent_routed_control_ms": routed_control_ms,
        "paired_persistent_exact_ms": paired_persistent_ms,
        "paired_persistent_speedup_vs_routed": (
            routed_control_ms / paired_persistent_ms
        ),
        "paired_persistent_max_abs_diff": paired_max_abs_diff,
        "paired_control_max_abs_diff": paired_control_max_abs_diff,
        "paired_direct_control_ms": paired_direct_control_ms,
        "paired_direct_candidate_ms": paired_direct_candidate_ms,
        "paired_direct_candidate_speedup": (
            paired_direct_control_ms / paired_direct_candidate_ms
        ),
        "paired_gather_control_ms": paired_gather_control_ms,
        "paired_gather_exact_ms": paired_gather_ms,
        "paired_gather_speedup_vs_persistent": (
            paired_gather_control_ms / paired_gather_ms
        ),
        "paired_gather_max_abs_diff": paired_gather_max_abs_diff,
        "self_contained_config": self_contained_config,
        "self_contained_control_ms": self_contained_control_ms,
        "self_contained_exact_ms": self_contained_ms,
        "self_contained_speedup_vs_persistent": (
            self_contained_control_ms / self_contained_ms
        ),
        "self_contained_max_abs_diff": self_contained_max_abs_diff,
        "grouped_owner_tiles": grouped_owner_tiles,
        "grouped_owner_control_ms": grouped_owner_control_ms,
        "grouped_owner_exact_ms": grouped_owner_ms,
        "grouped_owner_speedup_vs_persistent": (
            grouped_owner_control_ms / grouped_owner_ms
        ),
        "grouped_owner_max_abs_diff": grouped_owner_max_abs_diff,
        "paired_base_max_abs_diff": paired_base_max_abs_diff,
        "paired_residual_max_abs_diff": paired_residual_max_abs_diff,
        "paired_fused_control_ms": paired_control_ms,
        "paired_fused_exact_ms": paired_fused_ms,
        "paired_fused_speedup_vs_persistent": (
            paired_control_ms / paired_fused_ms
        ),
        "paired_fused_max_abs_diff": paired_fused_max_abs_diff,
        "paired_fused_control_max_abs_diff": (
            paired_fused_control_max_abs_diff
        ),
        "paired_fused_direct_control_ms": paired_fused_direct_control_ms,
        "paired_fused_direct_candidate_ms": (
            paired_fused_direct_candidate_ms
        ),
        "paired_fused_direct_candidate_speedup": (
            paired_fused_direct_control_ms / paired_fused_direct_candidate_ms
        ),
        "paired_fused_vs_persistent_max_abs_diff": (
            paired_fused_vs_persistent_max_abs_diff
        ),
        "paired_fused_dense_max_abs_diff": paired_fused_dense_max_abs_diff,
        "paired_fused_sparse_max_abs_diff": paired_fused_sparse_max_abs_diff,
        "paired_fused_base_max_abs_diff": paired_fused_base_max_abs_diff,
        "paired_fused_counter_max": paired_fused_counter_max,
        "paired_fused_control_counter_max": (
            paired_fused_control_counter_max
        ),
        "paired_fused_counter_max_after_graph": (
            paired_fused_counter_max_after_graph
        ),
        "paired_fused_control_counter_max_after_graph": (
            paired_fused_control_counter_max_after_graph
        ),
        "paired_compact_fused_control_ms": compact_control_ms,
        "paired_compact_fused_exact_ms": paired_compact_fused_ms,
        "paired_compact_fused_speedup_vs_persistent": (
            compact_control_ms / paired_compact_fused_ms
        ),
        "paired_compact_fused_max_abs_diff": (
            paired_compact_fused_max_abs_diff
        ),
        "paired_compact_fused_counter_max": (
            paired_compact_fused_counter_max
        ),
        "paired_compact_fused_counter_max_after_graph": (
            paired_compact_fused_counter_max_after_graph
        ),
        "standalone_correction_max_abs_diff": (
            standalone_correction_max_abs_diff
        ),
        "fused_vs_standalone_max_abs_diff": (
            fused_vs_standalone_max_abs_diff
        ),
        "dense_overwrite_control_ms": dense_overwrite_control_ms,
        "dense_overwrite_exact_ms": dense_overwrite_ms,
        "dense_overwrite_speedup_vs_routed": (
            dense_overwrite_control_ms / dense_overwrite_ms
            if dense_overwrite_ms is not None
            else None
        ),
        "dense_overwrite_max_abs_diff": dense_overwrite_max_abs_diff,
        "fp8_static_control_ms": fp8_static_control_ms,
        "fp8_static_exact_ms": fp8_static_ms,
        "fp8_static_speedup_vs_routed": (
            fp8_static_control_ms / fp8_static_ms
            if fp8_static_ms is not None
            else None
        ),
        "fp8_static_max_abs_diff": fp8_static_max_abs_diff,
        "fp8_token_control_ms": fp8_token_control_ms,
        "fp8_token_exact_ms": fp8_token_ms,
        "fp8_token_speedup_vs_routed": (
            fp8_token_control_ms / fp8_token_ms
            if fp8_token_ms is not None
            else None
        ),
        "fp8_token_max_abs_diff": fp8_token_max_abs_diff,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    base_schedule = (
        "partitioned"
        if any(row["paired_schedule"] == "partitioned" for row in rows)
        else str(rows[0]["paired_schedule"])
    )
    rows = [row for row in rows if row["paired_schedule"] == base_schedule]
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    variants = list(
        dict.fromkeys(
            (
                str(row["config"]),
                str(row["paired_schedule"]),
                int(row["paired_worker_blocks"]),
            )
            for row in rows
        )
    )
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    line_styles = ["-", "--"]
    figure, axes = plt.subplots(
        2, len(models), figsize=(6.2 * len(models), 7.2), squeeze=False
    )
    for column, model in enumerate(models):
        selected = [row for row in rows if row["model"] == model]
        for config_index, (config, schedule, worker_blocks) in enumerate(
            variants
        ):
            for k in sorted({int(row["K"]) for row in selected}):
                by_key = [
                    row
                    for row in selected
                    if row["config"] == config
                    and row["paired_schedule"] == schedule
                    and int(row["paired_worker_blocks"]) == worker_blocks
                    and int(row["K"]) == k
                ]
                if not by_key:
                    continue
                label = f"K={k}, {config}, workers={worker_blocks or 'auto'}"
                axes[0][column].plot(
                    [int(row["batch_size"]) for row in by_key],
                    [float(row["routed_exact_speedup"]) for row in by_key],
                    marker="o",
                    color=colors[k],
                    linestyle=line_styles[config_index % len(line_styles)],
                    label=label,
                )
                axes[1][column].plot(
                    [int(row["batch_size"]) for row in by_key],
                    [float(row["routed_full_stage_speedup"]) for row in by_key],
                    marker="s",
                    color=colors[k],
                    linestyle=line_styles[config_index % len(line_styles)],
                    label=label,
                )
        for row_index, title in enumerate(
            ("Exact mixed gate/up", "Full W24 epilogue stage")
        ):
            axis = axes[row_index][column]
            axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
            axis.set_title(f"{model}: {title}")
            axis.set_xlabel("Batch size")
            axis.set_ylabel("Speedup")
            axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_paired_persistent_plot(
    path: Path, rows: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    variants = list(
        dict.fromkeys(
            (
                str(row["config"]),
                str(row["paired_schedule"]),
                int(row["paired_worker_blocks"]),
            )
            for row in rows
        )
    )
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    line_styles = ["-", "--"]
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        selected = [row for row in rows if row["model"] == model]
        for variant_index, (config, schedule, worker_blocks) in enumerate(
            variants
        ):
            for k in sorted({int(row["K"]) for row in selected}):
                by_key = [
                    row
                    for row in selected
                    if row["config"] == config
                    and row["paired_schedule"] == schedule
                    and int(row["paired_worker_blocks"]) == worker_blocks
                    and int(row["K"]) == k
                ]
                if not by_key:
                    continue
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["paired_persistent_speedup_vs_routed"])
                        for row in by_key
                    ],
                    marker="o" if schedule == "partitioned" else "s",
                    color=colors[k],
                    linestyle=line_styles[variant_index % len(line_styles)],
                    label=(
                        f"K={k}, {config}, {schedule}, "
                        f"workers={worker_blocks or 'auto'}"
                    ),
                )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model}: paired-persistent gate/up")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Speedup vs two-stream routed exact")
        axis.set_xticks(
            sorted({int(row["batch_size"]) for row in selected})
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_fused_epilogue_plot(
    path: Path, rows: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    configs = list(dict.fromkeys(str(row["config"]) for row in rows))
    backends = (
        (
            "grouped_owner_speedup_vs_persistent",
            "grouped owner",
            "v",
        ),
        (
            "self_contained_speedup_vs_persistent",
            "self-contained W24+R24",
            "D",
        ),
        (
            "paired_gather_speedup_vs_persistent",
            "gather in residual mainloop",
            "^",
        ),
        ("paired_fused_speedup_vs_persistent", "gathered", "o"),
        (
            "paired_compact_fused_speedup_vs_persistent",
            "compact",
            "s",
        ),
    )
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
                for field, backend, marker in backends:
                    axis.plot(
                        [int(row["batch_size"]) for row in by_key],
                        [float(row[field]) for row in by_key],
                        marker=marker,
                        color=colors[k],
                        linestyle="-" if config_index % 2 == 0 else "--",
                        label=f"K={k}, {config}, {backend}",
                    )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model}: paired Gate/Up ablations")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Speedup vs current paired path")
        axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_fp8_residual_plot(
    path: Path, rows: list[dict[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in rows
        if row["fp8_token_speedup_vs_routed"] is not None
    ]
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    backends = (
        ("fp8_static_speedup_vs_routed", "FP8 static scale", "-", "o"),
        ("fp8_token_speedup_vs_routed", "FP8 token scale", "--", "s"),
    )
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        selected = [row for row in rows if row["model"] == model]
        for field, backend, line_style, marker in backends:
            for k in sorted({int(row["K"]) for row in selected}):
                by_key = [row for row in selected if int(row["K"]) == k]
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [float(row[field]) for row in by_key],
                    marker=marker,
                    color=colors[k],
                    linestyle=line_style,
                    label=f"K={k}, {backend}",
                )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model}: FP8 dense-row residual")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Speedup vs FP16 2:4 residual")
        axis.set_xticks(
            sorted({int(row["batch_size"]) for row in selected})
        )
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
        "--paired-control-config",
        default="",
        help="optional paired config for direct interleaved A/B timing",
    )
    parser.add_argument(
        "--paired-config",
        default="",
        help="optional paired W24/R24 config; defaults to each --configs item",
    )
    parser.add_argument(
        "--paired-fused-control-config",
        default="",
        help="optional fused config for direct interleaved A/B timing",
    )
    parser.add_argument(
        "--paired-fused-config",
        default="",
        help="optional fused-epilogue config; defaults to each --configs item",
    )
    parser.add_argument(
        "--paired-fused-schedule",
        choices=SUPPORTED_PAIRED_FUSED_SCHEDULES,
        default="partitioned",
    )
    parser.add_argument(
        "--paired-schedules",
        type=parse_csv_strings,
        default=DEFAULT_PAIRED_SCHEDULES,
    )
    parser.add_argument(
        "--paired-worker-blocks", type=parse_csv_ints, default=(0,)
    )
    parser.add_argument(
        "--grouped-owner-tiles",
        type=int,
        choices=(1, 2, 3, 4),
        default=2,
    )
    parser.add_argument("--dense-fraction", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument(
        "--route-mode",
        choices=("ratio_total", "bonus_dense", "draft_ratio_cap"),
        default="ratio_total",
    )
    parser.add_argument("--dense-cap", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=30)
    parser.add_argument("--benchmark-fp8-residual", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_models = [model for model in args.models if model not in MODELS]
    invalid_configs = [
        config for config in args.configs if config not in SUPPORTED_CONFIGS
    ]
    invalid_schedules = [
        schedule
        for schedule in args.paired_schedules
        if schedule not in DEFAULT_PAIRED_SCHEDULES
    ]
    invalid_fused_config = (
        args.paired_fused_config
        and args.paired_fused_config not in SUPPORTED_PAIRED_FUSED_CONFIGS
    )
    invalid_fused_control_config = (
        args.paired_fused_control_config
        and args.paired_fused_control_config
        not in SUPPORTED_PAIRED_FUSED_CONFIGS
    )
    invalid_paired_config = (
        args.paired_config
        and args.paired_config not in SUPPORTED_PAIRED_CONFIGS
    )
    invalid_paired_control_config = (
        args.paired_control_config
        and args.paired_control_config not in SUPPORTED_PAIRED_CONFIGS
    )
    if (
        invalid_models
        or invalid_configs
        or invalid_schedules
        or invalid_paired_control_config
        or invalid_paired_config
        or invalid_fused_control_config
        or invalid_fused_config
    ):
        raise ValueError(
            "unsupported "
            f"models={invalid_models}, configs={invalid_configs}, "
            f"paired_schedules={invalid_schedules}, "
            f"paired_control_config={args.paired_control_config!r}, "
            f"paired_config={args.paired_config!r}, "
            "paired_fused_control_config="
            f"{args.paired_fused_control_config!r}, "
            f"paired_fused_config={args.paired_fused_config!r}"
        )
    if any(
        worker_blocks == 1 or worker_blocks < 0
        for worker_blocks in args.paired_worker_blocks
    ):
        raise ValueError("--paired-worker-blocks must be 0 (auto) or at least 2")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    results: list[dict[str, object]] = []
    for model in args.models:
        weights = prepare_gate_up_weights(
            int(MODELS[model]["hidden"]),
            int(MODELS[model]["intermediate"]),
            generator,
        )
        for batch_size in args.batch_sizes:
            for k in args.k_values:
                for config in args.configs:
                    for paired_schedule in args.paired_schedules:
                        for worker_blocks in args.paired_worker_blocks:
                            result = run_case(
                                model,
                                batch_size,
                                k,
                                config,
                                paired_schedule,
                                worker_blocks,
                                weights,
                                dense_fraction=args.dense_fraction,
                                min_dense_per_request=(
                                    args.min_dense_per_request
                                ),
                                route_mode=args.route_mode,
                                dense_cap=args.dense_cap,
                                generator=generator,
                                unroll=args.unroll,
                                replays=args.replays,
                                trials=args.trials,
                                graph_warmup_replays=(
                                    args.graph_warmup_replays
                                ),
                                benchmark_fp8_residual=(
                                    args.benchmark_fp8_residual
                                ),
                                paired_control_config=(
                                    args.paired_control_config
                                ),
                                paired_config=(
                                    args.paired_config or config
                                ),
                                paired_fused_control_config=(
                                    args.paired_fused_control_config
                                ),
                                paired_fused_config=(
                                    args.paired_fused_config or config
                                ),
                                paired_fused_schedule=(
                                    args.paired_fused_schedule
                                ),
                                grouped_owner_tiles=args.grouped_owner_tiles,
                            )
                            results.append(result)
                            print(
                                f"{model} bs={batch_size} K={k} "
                                f"dense={int(result['dense_rows'])}/"
                                f"{int(result['rows'])} config={config} "
                                f"schedule={paired_schedule} "
                                f"fused_schedule={args.paired_fused_schedule} "
                                f"workers={worker_blocks} "
                                "paired="
                                f"{float(result['paired_persistent_speedup_vs_routed']):.3f}x "
                                "self="
                                f"{float(result['self_contained_speedup_vs_persistent']):.3f}x "
                                "owner="
                                f"{float(result['grouped_owner_speedup_vs_persistent']):.3f}x "
                                "fused="
                                f"{float(result['paired_fused_speedup_vs_persistent']):.3f}x "
                                "compact_fused="
                                f"{float(result['paired_compact_fused_speedup_vs_persistent']):.3f}x",
                                flush=True,
                            )
                            if args.benchmark_fp8_residual:
                                print(
                                    "  fp8 residual: static="
                                    f"{float(result['fp8_static_speedup_vs_routed']):.3f}x, "
                                    "token="
                                    f"{float(result['fp8_token_speedup_vs_routed']):.3f}x, "
                                    "max_diff="
                                    f"{float(result['fp8_token_max_abs_diff']):.5f}",
                                    flush=True,
                                )
        del weights
        torch.cuda.empty_cache()

    csv_path = args.output_root / "routed_swiglu_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "routed_swiglu_speedup.png", results)
    write_paired_persistent_plot(
        args.output_root / "paired_persistent_speedup.png", results
    )
    write_fused_epilogue_plot(
        args.output_root / "paired_fused_epilogue_speedup.png", results
    )
    if args.benchmark_fp8_residual:
        write_fp8_residual_plot(
            args.output_root / "fp8_residual_speedup.png", results
        )
    print(args.output_root)


if __name__ == "__main__":
    main()
