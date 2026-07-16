#!/usr/bin/env python3
"""Benchmark exact row-routed QKV chains with CUTLASS and cuSPARSELt.

The benchmark includes the full sparse GEMM, complementary 2:4 residual for
dense rows, indexed addition, layout materialization, and QKV post-processing.
Packing and routing-score selection are intentionally outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm import _custom_ops as vllm_ops  # noqa: E402
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    dense_cutlass_simt_weight_t_gemm,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_add_indexed_rows_strided_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_gather_gemm_prepacked,
    sparse24_cutlass_paired_persistent_gemm_prepacked,
    sparse24_gather_rows_,
    sparse24_qkv_transpose_add_routed_residual_postop,
    sparse24_qkv_transpose_postop,
    sparse24_transpose_add_routed_residual,
    sparse24_transpose_output_contiguous,
)


IN_FEATURES = 4096
Q_SIZE = 4096
KV_SIZE = 1024
OUT_FEATURES = Q_SIZE + 2 * KV_SIZE
HEAD_DIM = 128
Q_HEADS = Q_SIZE // HEAD_DIM
KV_HEADS = KV_SIZE // HEAD_DIM
EPSILON = 1.0e-6


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated ints")
    return result


def _csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def _event_time_ms(fn: Callable[[], torch.Tensor], repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeat)


def _median_graph_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = fn()
    torch.cuda.synchronize()

    def replay() -> torch.Tensor:
        graph.replay()
        return captured_output

    for _ in range(warmup):
        replay()
    torch.cuda.synchronize()
    samples = [_event_time_ms(replay, repeat) for _ in range(trials)]
    del graph, captured_output
    return float(statistics.median(samples)), samples


def _pack_cutlass(weight_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_24(weight_kn.contiguous(), layout="n_major")
    return prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=int(weight_kn.shape[0]),
    )


def _pack_cslt(weight_kn: torch.Tensor) -> torch.Tensor:
    previous = bool(SparseSemiStructuredTensor._FORCE_CUTLASS)
    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        sparse_weight = to_sparse_semi_structured(weight_kn.t().contiguous())
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = previous
    return sparse_weight.packed


def _apply_qwen_postop(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    vllm_ops.fused_qk_norm_rope(
        qkv,
        Q_HEADS,
        KV_HEADS,
        KV_HEADS,
        HEAD_DIM,
        EPSILON,
        q_weight,
        k_weight,
        cache,
        True,
        positions,
        -1,
    )
    return qkv


def _apply_llama_postop(
    qkv: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    q, k, _v = qkv.split((Q_SIZE, KV_SIZE, KV_SIZE), dim=-1)
    vllm_ops.rotary_embedding(
        positions,
        q,
        k,
        HEAD_DIM,
        cache,
        True,
    )
    return qkv


def _route_dense_count(
    mode: str,
    *,
    batch_size: int,
    num_spec_tokens: int,
    rows: int,
    dense_ratio: float,
) -> int:
    if mode == "ratio_total":
        count = math.ceil(rows * dense_ratio)
    elif mode == "bonus_dense":
        count = batch_size + math.ceil(
            batch_size * num_spec_tokens * dense_ratio
        )
    else:
        raise ValueError(f"unsupported route mode: {mode}")
    return min(rows - 1, max(1, count))


def _run_case(
    *,
    model: str,
    batch_size: int,
    num_spec_tokens: int,
    route_mode: str,
    dense_ratio: float,
    cslt_alg_id: int,
    parallel: bool,
    full_stream_priority: int,
    residual_stream_priority: int,
    residual_first: bool,
    seed: int,
    warmup: int,
    repeat: int,
    trials: int,
    selected_variants: set[str] | None,
) -> list[dict[str, object]]:
    rows = batch_size * (num_spec_tokens + 1)
    dense_count = _route_dense_count(
        route_mode,
        batch_size=batch_size,
        num_spec_tokens=num_spec_tokens,
        rows=rows,
        dense_ratio=dense_ratio,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (rows, IN_FEATURES),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight_kn = (
        torch.randn(
            (IN_FEATURES, OUT_FEATURES),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 0.02
    ).contiguous()
    weight_kn = torch.where(
        weight_kn == 0,
        torch.full_like(weight_kn, 1.0e-3),
        weight_kn,
    ).contiguous()
    weight24_kn, _mask = apply_random_24_mask(weight_kn, generator=generator)
    weight24_kn = weight24_kn.contiguous()
    residual_kn = (weight_kn - weight24_kn).contiguous()
    residual_weight_t = residual_kn.t().contiguous()
    residual_fp8_t, residual_fp8_scale = vllm_ops.scaled_fp8_quant(
        residual_weight_t,
        use_per_token_if_dynamic=True,
    )
    residual_fp8 = residual_fp8_t.t()
    full_values, full_meta = _pack_cutlass(weight24_kn)
    residual_values, residual_meta = _pack_cutlass(residual_kn)
    cslt_packed = _pack_cslt(weight24_kn)
    cslt_residual_packed = _pack_cslt(residual_kn)

    dense_rows = torch.randperm(rows, device="cuda", generator=generator)[
        :dense_count
    ].sort().values.to(dtype=torch.int32)
    dense_slot_by_row = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slot_by_row.scatter_(
        0,
        dense_rows.to(dtype=torch.int64),
        torch.arange(dense_count, device="cuda", dtype=torch.int32),
    )
    dense_run = (dense_count + 7) // 8 * 8
    dense_input = torch.zeros(
        (dense_run, IN_FEATURES), device="cuda", dtype=torch.float16
    )
    residual_contiguous_out = torch.empty(
        (dense_run, OUT_FEATURES), device="cuda", dtype=torch.float16
    )
    residual_workspace = torch.empty(
        (OUT_FEATURES, dense_run), device="cuda", dtype=torch.float16
    )
    residual_view_out = torch.empty_strided(
        (dense_run, OUT_FEATURES),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    gather_residual_view_out = torch.empty_strided(
        (dense_count, OUT_FEATURES),
        (1, dense_run),
        device="cuda",
        dtype=torch.float16,
    )
    fp8_input = torch.empty(
        dense_input.shape, device="cuda", dtype=torch.float8_e4m3fn
    )
    fp8_token_scale = torch.empty(
        (dense_run, 1), device="cuda", dtype=torch.float32
    )
    fp8_static_scale = torch.tensor(
        [0.02], device="cuda", dtype=torch.float32
    )
    fp8_residual_out = torch.empty(
        (dense_run, OUT_FEATURES), device="cuda", dtype=torch.float16
    )
    cutlass_contiguous_out = torch.empty(
        (rows, OUT_FEATURES), device="cuda", dtype=torch.float16
    )
    cutlass_workspace = torch.empty(
        (OUT_FEATURES, rows), device="cuda", dtype=torch.float16
    )
    cutlass_view_out = torch.empty_strided(
        (rows, OUT_FEATURES),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    materialized_out = torch.empty(
        (rows, OUT_FEATURES), device="cuda", dtype=torch.float16
    )
    dense_out = torch.empty_like(materialized_out)
    fused_out = torch.empty_like(materialized_out)
    q_weight = torch.randn(
        HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator
    )
    k_weight = torch.randn(
        HEAD_DIM, device="cuda", dtype=torch.float16, generator=generator
    )
    angles = torch.randn(
        (4096, HEAD_DIM // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half().contiguous()
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)

    full_stream = torch.cuda.Stream(priority=full_stream_priority)
    residual_stream = torch.cuda.Stream(priority=residual_stream_priority)

    def run_parallel(
        full_fn: Callable[[], torch.Tensor],
        residual_fn: Callable[[], torch.Tensor],
        *,
        gather_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not parallel:
            full_out = full_fn()
            if gather_input:
                sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
            return full_out, residual_fn()
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        if residual_first:
            with torch.cuda.stream(residual_stream):
                if gather_input:
                    sparse24_gather_rows_(
                        x, dense_rows, dense_input[:dense_count]
                    )
                selected_residual_out = residual_fn()
            with torch.cuda.stream(full_stream):
                full_out = full_fn()
        else:
            with torch.cuda.stream(full_stream):
                full_out = full_fn()
            with torch.cuda.stream(residual_stream):
                if gather_input:
                    sparse24_gather_rows_(
                        x, dense_rows, dense_input[:dense_count]
                    )
                selected_residual_out = residual_fn()
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return full_out, selected_residual_out

    def qwen_separate(qkv: torch.Tensor) -> torch.Tensor:
        return _apply_qwen_postop(qkv, q_weight, k_weight, cache, positions)

    def llama_separate(qkv: torch.Tensor) -> torch.Tensor:
        return _apply_llama_postop(qkv, cache, positions)

    separate_postop = qwen_separate if model == "qwen3_8b" else llama_separate

    def dense_separate() -> torch.Tensor:
        torch.mm(x, weight_kn, out=dense_out)
        return separate_postop(dense_out)

    fused_kwargs: dict[str, object] = {}
    if model == "qwen3_8b":
        fused_kwargs.update(
            epsilon=EPSILON,
            q_weight=q_weight,
            k_weight=k_weight,
        )
    else:
        fused_kwargs.update(epsilon=0.0)

    def residual_add_contiguous(
        output: torch.Tensor, residual_output: torch.Tensor
    ) -> torch.Tensor:
        return sparse24_add_indexed_rows_contiguous_(
            output, residual_output[:dense_count], dense_rows
        )

    def residual_add_strided(
        output: torch.Tensor, residual_output: torch.Tensor
    ) -> torch.Tensor:
        return sparse24_add_indexed_rows_strided_(
            output, residual_output[:dense_count], dense_rows
        )

    def residual_contiguous() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            dense_input,
            residual_values,
            residual_meta,
            contiguous_output=True,
            out=residual_contiguous_out,
            workspace=residual_workspace,
            device_config="auto",
        )

    def residual_view() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            dense_input,
            residual_values,
            residual_meta,
            contiguous_output=False,
            out=residual_view_out,
            device_config="auto",
        )

    def gather_residual_view(config: str) -> torch.Tensor:
        return sparse24_cutlass_gather_gemm_prepacked(
            x,
            residual_values,
            residual_meta,
            dense_rows,
            out=gather_residual_view_out,
            config=config,
        )

    def cutlass_full_contiguous() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            x,
            full_values,
            full_meta,
            contiguous_output=True,
            out=cutlass_contiguous_out,
            workspace=cutlass_workspace,
            device_config="auto",
        )

    def cutlass_full_view() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            x,
            full_values,
            full_meta,
            contiguous_output=False,
            out=cutlass_view_out,
            device_config="auto",
        )

    def cslt_full_view() -> torch.Tensor:
        return torch._cslt_sparse_mm(
            cslt_packed,
            x.t(),
            transpose_result=False,
            alg_id=cslt_alg_id,
        ).t()

    def cslt_residual_view() -> torch.Tensor:
        return torch._cslt_sparse_mm(
            cslt_residual_packed,
            dense_input.t(),
            transpose_result=False,
            alg_id=cslt_alg_id,
        ).t()

    def simt_residual_view(config: str) -> torch.Tensor:
        return dense_cutlass_simt_weight_t_gemm(
            dense_input,
            residual_weight_t,
            out=residual_view_out,
            config=config,
        )

    def fp8_residual_contiguous(dynamic_per_token: bool) -> torch.Tensor:
        if dynamic_per_token:
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                fp8_input, dense_input, fp8_token_scale, None
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
            fp8_residual_out,
            fp8_input,
            residual_fp8,
            input_scale,
            residual_fp8_scale,
            None,
        )
        return fp8_residual_out

    def current_cutlass() -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_contiguous, residual_contiguous
        )
        residual_add_contiguous(output, selected_residual)
        return separate_postop(output)

    def cutlass_fused() -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_view, residual_view
        )
        residual_add_strided(output, selected_residual)
        return sparse24_qkv_transpose_postop(
            output,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cslt_separate() -> torch.Tensor:
        output, selected_residual = run_parallel(cslt_full_view, residual_view)
        residual_add_strided(output, selected_residual)
        sparse24_transpose_output_contiguous(output, out=materialized_out)
        return separate_postop(materialized_out)

    def cslt_fused() -> torch.Tensor:
        output, selected_residual = run_parallel(cslt_full_view, residual_view)
        residual_add_strided(output, selected_residual)
        return sparse24_qkv_transpose_postop(
            output,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cutlass_routed_residual_fused() -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_view, residual_view
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cutlass_gather_residual_routed_fused(config: str) -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_view,
            lambda: gather_residual_view(config),
            gather_input=False,
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def paired_persistent_routed_residual_fused(
        schedule: str,
    ) -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
        output, selected_residual = (
            sparse24_cutlass_paired_persistent_gemm_prepacked(
                x,
                full_values,
                full_meta,
                dense_input,
                residual_values,
                residual_meta,
                full_out=cutlass_view_out,
                residual_out=residual_view_out,
                schedule=schedule,
            )
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cslt_routed_residual_fused() -> torch.Tensor:
        output, selected_residual = run_parallel(cslt_full_view, residual_view)
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cutlass_full_cslt_residual_fused() -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_view, cslt_residual_view
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cslt_both_routed_residual_fused() -> torch.Tensor:
        output, selected_residual = run_parallel(
            cslt_full_view, cslt_residual_view
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cslt_full_simt_residual_fused(config: str) -> torch.Tensor:
        output, selected_residual = run_parallel(
            cslt_full_view, lambda: simt_residual_view(config)
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cslt_full_fp8_residual_fused(
        dynamic_per_token: bool,
    ) -> torch.Tensor:
        output, selected_residual = run_parallel(
            cslt_full_view,
            lambda: fp8_residual_contiguous(dynamic_per_token),
        )
        return sparse24_qkv_transpose_add_routed_residual_postop(
            output,
            selected_residual,
            dense_slot_by_row,
            cache,
            positions,
            q_size=Q_SIZE,
            kv_size=KV_SIZE,
            head_dim=HEAD_DIM,
            is_neox=True,
            out=fused_out,
            **fused_kwargs,
        )

    def cutlass_routed_materialize_separate() -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_view, residual_view
        )
        sparse24_transpose_add_routed_residual(
            output,
            selected_residual,
            dense_slot_by_row,
            dense_count=dense_count,
            out=materialized_out,
        )
        return separate_postop(materialized_out)

    def cutlass_gather_residual_routed_materialize_separate(
        config: str,
    ) -> torch.Tensor:
        output, selected_residual = run_parallel(
            cutlass_full_view,
            lambda: gather_residual_view(config),
            gather_input=False,
        )
        sparse24_transpose_add_routed_residual(
            output,
            selected_residual,
            dense_slot_by_row,
            dense_count=dense_count,
            out=materialized_out,
        )
        return separate_postop(materialized_out)

    def cslt_routed_materialize_separate() -> torch.Tensor:
        output, selected_residual = run_parallel(cslt_full_view, residual_view)
        sparse24_transpose_add_routed_residual(
            output,
            selected_residual,
            dense_slot_by_row,
            dense_count=dense_count,
            out=materialized_out,
        )
        return separate_postop(materialized_out)

    variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("cutlass_current", current_cutlass),
        ("cutlass_fused_epilogue", cutlass_fused),
        ("cusparselt_separate_epilogue", cslt_separate),
        ("cusparselt_fused_epilogue", cslt_fused),
        (
            "cutlass_routed_residual_fused_epilogue",
            cutlass_routed_residual_fused,
        ),
        (
            "cutlass_gather_residual_256x32_routed_fused_epilogue",
            lambda: cutlass_gather_residual_routed_fused(
                "256x32x64_s3_sw4"
            ),
        ),
        (
            "cutlass_gather_residual_256x64_routed_fused_epilogue",
            lambda: cutlass_gather_residual_routed_fused(
                "256x64x64_s3_sw4"
            ),
        ),
        (
            "cutlass_gather_residual_128x32_routed_fused_epilogue",
            lambda: cutlass_gather_residual_routed_fused(
                "128x32x64_s4_sw4"
            ),
        ),
        (
            "paired_persistent_partitioned_routed_residual_fused_epilogue",
            lambda: paired_persistent_routed_residual_fused("partitioned"),
        ),
        (
            "paired_persistent_interleaved_routed_residual_fused_epilogue",
            lambda: paired_persistent_routed_residual_fused("interleaved"),
        ),
        (
            "cusparselt_routed_residual_fused_epilogue",
            cslt_routed_residual_fused,
        ),
        (
            "cutlass_full_cusparselt_residual_fused_epilogue",
            cutlass_full_cslt_residual_fused,
        ),
        (
            "cusparselt_both_routed_residual_fused_epilogue",
            cslt_both_routed_residual_fused,
        ),
        (
            "cusparselt_full_simt64_residual_fused_epilogue",
            lambda: cslt_full_simt_residual_fused("64x64x8"),
        ),
        (
            "cusparselt_full_simt128_residual_fused_epilogue",
            lambda: cslt_full_simt_residual_fused("128x64x8"),
        ),
        (
            "cusparselt_full_fp8_static_residual_fused_epilogue",
            lambda: cslt_full_fp8_residual_fused(False),
        ),
        (
            "cusparselt_full_fp8_token_residual_fused_epilogue",
            lambda: cslt_full_fp8_residual_fused(True),
        ),
        (
            "cutlass_routed_materialize_separate_epilogue",
            cutlass_routed_materialize_separate,
        ),
        (
            "cutlass_gather_residual_256x32_routed_materialize_separate_epilogue",
            lambda: cutlass_gather_residual_routed_materialize_separate(
                "256x32x64_s3_sw4"
            ),
        ),
        (
            "cutlass_gather_residual_256x64_routed_materialize_separate_epilogue",
            lambda: cutlass_gather_residual_routed_materialize_separate(
                "256x64x64_s3_sw4"
            ),
        ),
        (
            "cutlass_gather_residual_128x32_routed_materialize_separate_epilogue",
            lambda: cutlass_gather_residual_routed_materialize_separate(
                "128x32x64_s4_sw4"
            ),
        ),
        (
            "cusparselt_routed_materialize_separate_epilogue",
            cslt_routed_materialize_separate,
        ),
    ]
    if selected_variants:
        available = {label for label, _fn in variants}
        unknown = sorted(selected_variants - available)
        if unknown:
            raise ValueError(f"unsupported variants: {unknown}")
        if "cutlass_current" not in selected_variants:
            raise ValueError("selected variants must include cutlass_current")
        variants = [
            (label, fn)
            for label, fn in variants
            if label in selected_variants
        ]
    outputs: dict[str, torch.Tensor] = {}
    for label, fn in variants:
        outputs[label] = fn().clone()
    torch.cuda.synchronize()
    expected = outputs["cutlass_current"]
    dense_graph_ms, dense_samples = _median_graph_ms(
        dense_separate,
        warmup=warmup,
        repeat=repeat,
        trials=trials,
    )
    result_rows: list[dict[str, object]] = []
    for label, fn in variants:
        graph_ms, samples = _median_graph_ms(
            fn, warmup=warmup, repeat=repeat, trials=trials
        )
        actual = fn()
        torch.cuda.synchronize()
        max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
        result_rows.append(
            {
                "model": model,
                "batch_size": batch_size,
                "num_spec_tokens": num_spec_tokens,
                "rows": rows,
                "route_mode": route_mode,
                "dense_ratio": dense_ratio,
                "dense_count": dense_count,
                "dense_fraction": dense_count / rows,
                "cslt_alg_id": cslt_alg_id,
                "parallel": parallel,
                "full_stream_priority": full_stream_priority,
                "residual_stream_priority": residual_stream_priority,
                "residual_first": residual_first,
                "variant": label,
                "graph_ms": graph_ms,
                "graph_samples_ms": ";".join(f"{value:.6f}" for value in samples),
                "dense_graph_ms": dense_graph_ms,
                "dense_graph_samples_ms": ";".join(
                    f"{value:.6f}" for value in dense_samples
                ),
                "max_abs_diff_vs_current": max_abs_diff,
                "pass": bool(
                    torch.allclose(actual, expected, rtol=4.0e-2, atol=4.0e-2)
                ),
            }
        )
    baseline_ms = float(result_rows[0]["graph_ms"])
    for row in result_rows:
        row["speedup_vs_current"] = baseline_ms / float(row["graph_ms"])
        row["speedup_vs_dense"] = dense_graph_ms / float(row["graph_ms"])

    del outputs, expected, variants
    del cslt_packed, cslt_residual_packed
    del full_values, full_meta, residual_values, residual_meta
    del residual_fp8, residual_fp8_t, residual_fp8_scale
    del fp8_input, fp8_token_scale
    del fp8_static_scale, fp8_residual_out
    del weight_kn, weight24_kn, residual_kn, residual_weight_t, x
    gc.collect()
    torch.cuda.empty_cache()
    return result_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    candidates = [row for row in rows if row["variant"] != "cutlass_current"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Exact Row-Routed QKV Backend and Epilogue Ablation\n\n")
        handle.write(
            "Times include full 2:4 GEMM, complementary residual for dense "
            "rows, indexed add, and QKV norm/RoPE post-processing.\n\n"
        )
        handle.write(
            "| model | bs | K | M | route | dense rows | variant | ms | vs current | vs dense | diff | pass |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|\n"
        )
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['batch_size']} | "
                f"{row['num_spec_tokens']} | {row['rows']} | "
                f"{row['route_mode']} | {row['dense_count']} | "
                f"{row['variant']} | {float(row['graph_ms']):.6f} | "
                f"{float(row['speedup_vs_current']):.3f}x | "
                f"{float(row['speedup_vs_dense']):.3f}x | "
                f"{float(row['max_abs_diff_vs_current']):.5f} | "
                f"{row['pass']} |\n"
            )
        if candidates:
            best = max(candidates, key=lambda row: float(row["speedup_vs_current"]))
            handle.write(
                "\nBest candidate: "
                f"`{best['variant']}` at {best['model']} bs={best['batch_size']} "
                f"K={best['num_spec_tokens']} ({float(best['speedup_vs_current']):.3f}x).\n"
            )


def _write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    variants = [
        "cutlass_fused_epilogue",
        "cusparselt_separate_epilogue",
        "cusparselt_fused_epilogue",
        "cutlass_routed_residual_fused_epilogue",
        "cutlass_gather_residual_256x32_routed_fused_epilogue",
        "cutlass_gather_residual_256x64_routed_fused_epilogue",
        "cutlass_gather_residual_128x32_routed_fused_epilogue",
        "paired_persistent_partitioned_routed_residual_fused_epilogue",
        "paired_persistent_interleaved_routed_residual_fused_epilogue",
        "cusparselt_routed_residual_fused_epilogue",
        "cutlass_full_cusparselt_residual_fused_epilogue",
        "cusparselt_both_routed_residual_fused_epilogue",
        "cusparselt_full_simt64_residual_fused_epilogue",
        "cusparselt_full_simt128_residual_fused_epilogue",
        "cusparselt_full_fp8_static_residual_fused_epilogue",
        "cusparselt_full_fp8_token_residual_fused_epilogue",
        "cutlass_routed_materialize_separate_epilogue",
        "cutlass_gather_residual_256x32_routed_materialize_separate_epilogue",
        "cutlass_gather_residual_256x64_routed_materialize_separate_epilogue",
        "cutlass_gather_residual_128x32_routed_materialize_separate_epilogue",
        "cusparselt_routed_materialize_separate_epilogue",
    ]
    available = {str(row["variant"]) for row in rows}
    variants = [variant for variant in variants if variant in available]
    colors = {
        "cutlass_fused_epilogue": "#1f77b4",
        "cusparselt_separate_epilogue": "#ff7f0e",
        "cusparselt_fused_epilogue": "#2ca02c",
        "cutlass_routed_residual_fused_epilogue": "#d62728",
        "cutlass_gather_residual_256x32_routed_fused_epilogue": "#ff7f0e",
        "cutlass_gather_residual_256x64_routed_fused_epilogue": "#2ca02c",
        "cutlass_gather_residual_128x32_routed_fused_epilogue": "#9467bd",
        "paired_persistent_partitioned_routed_residual_fused_epilogue": "#17becf",
        "paired_persistent_interleaved_routed_residual_fused_epilogue": "#1f77b4",
        "cusparselt_routed_residual_fused_epilogue": "#9467bd",
        "cutlass_full_cusparselt_residual_fused_epilogue": "#bcbd22",
        "cusparselt_both_routed_residual_fused_epilogue": "#7f7f7f",
        "cusparselt_full_simt64_residual_fused_epilogue": "#2ca02c",
        "cusparselt_full_simt128_residual_fused_epilogue": "#8c564b",
        "cusparselt_full_fp8_static_residual_fused_epilogue": "#ff7f0e",
        "cusparselt_full_fp8_token_residual_fused_epilogue": "#d62728",
        "cutlass_routed_materialize_separate_epilogue": "#8c564b",
        "cutlass_gather_residual_256x32_routed_materialize_separate_epilogue": "#17becf",
        "cutlass_gather_residual_256x64_routed_materialize_separate_epilogue": "#bcbd22",
        "cutlass_gather_residual_128x32_routed_materialize_separate_epilogue": "#7f7f7f",
        "cusparselt_routed_materialize_separate_epilogue": "#e377c2",
    }
    points = sorted(
        {
            (
                str(row["model"]),
                int(row["batch_size"]),
                int(row["num_spec_tokens"]),
                str(row["route_mode"]),
            )
            for row in rows
        }
    )
    x_values = list(range(len(points)))
    figure, axis = plt.subplots(figsize=(max(8, len(points) * 1.2), 4.8))
    for variant in variants:
        values = []
        for model, batch_size, num_spec_tokens, route_mode in points:
            match = next(
                row
                for row in rows
                if row["model"] == model
                and int(row["batch_size"]) == batch_size
                and int(row["num_spec_tokens"]) == num_spec_tokens
                and row["route_mode"] == route_mode
                and row["variant"] == variant
            )
            values.append(float(match["speedup_vs_current"]))
        axis.plot(
            x_values,
            values,
            marker="o",
            linewidth=1.8,
            color=colors[variant],
            label=variant,
        )
    axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        [f"{model}\nbs{bs}/K{k}\n{route}" for model, bs, k, route in points],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("CUDA-graph speedup vs current exact QKV chain")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=_csv_strings,
        default=["qwen3_8b", "llama3_1_8b"],
    )
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[64])
    parser.add_argument("--k-values", type=_csv_ints, default=[8])
    parser.add_argument(
        "--route-modes",
        type=_csv_strings,
        default=["ratio_total", "bonus_dense"],
    )
    parser.add_argument("--dense-ratio", type=float, default=0.125)
    parser.add_argument("--cslt-alg-id", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap the full and complementary residual GEMMs.",
    )
    parser.add_argument("--full-stream-priority", type=int, default=0)
    parser.add_argument("--residual-stream-priority", type=int, default=0)
    parser.add_argument(
        "--residual-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Submit gather/residual work before the full sparse GEMM.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument(
        "--variants",
        type=_csv_strings,
        help="optional subset; must include cutlass_current",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with real GPU access")
    unsupported = sorted(set(args.models) - {"qwen3_8b", "llama3_1_8b"})
    if unsupported:
        raise ValueError(f"unsupported models: {unsupported}")
    unsupported_routes = sorted(
        set(args.route_modes) - {"ratio_total", "bonus_dense"}
    )
    if unsupported_routes:
        raise ValueError(f"unsupported route modes: {unsupported_routes}")
    if not 0.0 < args.dense_ratio < 1.0:
        raise ValueError("--dense-ratio must be between 0 and 1")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_cslt_qkv_fusion_{stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    case_index = 0
    for model in args.models:
        for batch_size in args.batch_sizes:
            for num_spec_tokens in args.k_values:
                for route_mode in args.route_modes:
                    case_index += 1
                    print(
                        f"[{case_index:02d}] {model} bs={batch_size} "
                        f"K={num_spec_tokens} route={route_mode}",
                        flush=True,
                    )
                    case_rows = _run_case(
                        model=model,
                        batch_size=batch_size,
                        num_spec_tokens=num_spec_tokens,
                        route_mode=route_mode,
                        dense_ratio=args.dense_ratio,
                        cslt_alg_id=args.cslt_alg_id,
                        parallel=args.parallel,
                        full_stream_priority=args.full_stream_priority,
                        residual_stream_priority=args.residual_stream_priority,
                        residual_first=args.residual_first,
                        seed=args.seed + case_index,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        trials=args.trials,
                        selected_variants=(
                            set(args.variants) if args.variants else None
                        ),
                    )
                    rows.extend(case_rows)
                    _write_csv(output_root / "qkv_fusion_ablation.csv", rows)
                    for row in case_rows:
                        print(
                            f"  {row['variant']}: {float(row['graph_ms']):.6f} ms, "
                            f"current={float(row['speedup_vs_current']):.3f}x, "
                            f"dense={float(row['speedup_vs_dense']):.3f}x, "
                            f"diff={float(row['max_abs_diff_vs_current']):.5f}, "
                            f"pass={row['pass']}",
                            flush=True,
                        )
    _write_report(output_root / "report.md", rows)
    _write_plot(output_root / "qkv_fusion_ablation.png", rows)
    metadata = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "models": args.models,
        "batch_sizes": args.batch_sizes,
        "k_values": args.k_values,
        "route_modes": args.route_modes,
        "dense_ratio": args.dense_ratio,
        "cslt_alg_id": args.cslt_alg_id,
        "parallel": args.parallel,
        "full_stream_priority": args.full_stream_priority,
        "residual_stream_priority": args.residual_stream_priority,
        "residual_first": args.residual_first,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "variants": args.variants or "all",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
