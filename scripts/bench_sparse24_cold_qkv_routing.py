#!/usr/bin/env python3
"""Benchmark exact mixed-row QKV routes with layer-cold weights."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_heterogeneous_routing import (  # noqa: E402
    QKV_OUTPUTS,
    make_route,
    prepare_weight,
)
from bench_sparse24_cslt_qkv_fusion import (  # noqa: E402
    EPSILON,
    HEAD_DIM,
    KV_SIZE,
    Q_SIZE,
    _apply_llama_postop,
    _apply_qwen_postop,
)
from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from vllm.speclink_kernel import (  # noqa: E402
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_heterogeneous_linear_prepacked,
    sparse24_cutlass_grouped_owner_qkv_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_paired_gather_residual_prepacked,
    sparse24_cutlass_paired_gather_residual_qkv_prepacked,
    sparse24_cutlass_paired_fused_routed_qkv_epilogue_prepacked,
    sparse24_cutlass_paired_finalize_qkv_prepacked,
    sparse24_cutlass_paired_inplace_residual_prepacked,
    sparse24_cutlass_routed_exact_linear_prepacked,
    sparse24_gather_rows_,
    sparse24_qkv_add_routed_residual_postop_inplace_,
    sparse24_qkv_transpose_add_routed_residual_postop,
    sparse24_sub_indexed_rows_contiguous_,
    sparse24_transpose_add_routed_residual,
)


@dataclass
class LayerState:
    x: torch.Tensor
    dense_weight: torch.Tensor
    sparse_values: torch.Tensor
    sparse_meta: torch.Tensor
    residual_values: torch.Tensor
    residual_meta: torch.Tensor
    dense_out: torch.Tensor
    static_out: torch.Tensor
    heterogeneous_out: torch.Tensor
    exact_out: torch.Tensor
    paired_full_out: torch.Tensor
    paired_residual_out: torch.Tensor
    paired_out: torch.Tensor
    paired_grid_barrier: torch.Tensor
    paired_qkv_epilogue_out: torch.Tensor
    paired_qkv_dense_base: torch.Tensor
    paired_qkv_counters: torch.Tensor
    dense_base_out: torch.Tensor
    sparse_correction_x: torch.Tensor
    sparse_correction_out: torch.Tensor
    inplace_out: torch.Tensor
    inplace_counters: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor


def prepare_layers(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    dense_count: int,
    paired_contiguous: bool,
    layer_count: int,
    generator: torch.Generator,
) -> list[LayerState]:
    states: list[LayerState] = []
    sparse_count = rows - dense_count
    sparse_run = (sparse_count + 7) // 8 * 8

    def paired_output(output_rows: int) -> torch.Tensor:
        if paired_contiguous:
            return torch.empty(
                (output_rows, out_features),
                device="cuda",
                dtype=torch.float16,
            )
        return torch.empty_strided(
            (output_rows, out_features),
            (1, output_rows),
            device="cuda",
            dtype=torch.float16,
        )

    for _ in range(layer_count):
        x = torch.randn(
            (rows, hidden),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.1)
        (
            weight,
            weight24,
            sparse_values,
            sparse_meta,
            residual_values,
            residual_meta,
        ) = prepare_weight(hidden, out_features, generator)
        dense_weight = weight.t().contiguous()
        del weight, weight24
        states.append(
            LayerState(
                x=x,
                dense_weight=dense_weight,
                sparse_values=sparse_values,
                sparse_meta=sparse_meta,
                residual_values=residual_values,
                residual_meta=residual_meta,
                dense_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                static_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                heterogeneous_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                exact_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                paired_full_out=paired_output(rows),
                paired_residual_out=paired_output(dense_count),
                paired_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                paired_grid_barrier=torch.zeros(
                    2, device="cuda", dtype=torch.int32
                ),
                paired_qkv_epilogue_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                paired_qkv_dense_base=torch.empty(
                    (dense_count, out_features),
                    device="cuda",
                    dtype=torch.float16,
                ),
                paired_qkv_counters=torch.zeros(
                    out_features // 256,
                    device="cuda",
                    dtype=torch.int32,
                ),
                dense_base_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                sparse_correction_x=torch.zeros(
                    (sparse_run, hidden), device="cuda", dtype=torch.float16
                ),
                sparse_correction_out=torch.empty(
                    (sparse_run, out_features),
                    device="cuda",
                    dtype=torch.float16,
                ),
                inplace_out=torch.empty(
                    (rows, out_features), device="cuda", dtype=torch.float16
                ),
                inplace_counters=torch.zeros(
                    (out_features + 127) // 128,
                    device="cuda",
                    dtype=torch.int32,
                ),
                q_weight=torch.randn(
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.float16,
                    generator=generator,
                ),
                k_weight=torch.randn(
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.float16,
                    generator=generator,
                ),
            )
        )
    return states


def run_point(
    *,
    model: str,
    batch_size: int,
    k: int,
    explicit_case: tuple[int, int] | None,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    hidden = int(MODELS[model]["hidden"])
    out_features = int(QKV_OUTPUTS[model])
    if explicit_case is None:
        rows = batch_size * (k + 1)
        dense_rows, sparse_rows = make_route(
            batch_size,
            k,
            dense_ratio=args.dense_ratio,
            min_dense_per_request=args.min_dense_per_request,
            generator=generator,
            dense_cap=args.dense_cap,
        )
        case_label = ""
    else:
        rows, explicit_dense_count = explicit_case
        permutation = torch.randperm(
            rows, device="cuda", generator=generator
        )
        dense_rows = permutation[:explicit_dense_count].sort().values
        dense_rows = dense_rows.to(torch.int32).contiguous()
        sparse_rows = permutation[explicit_dense_count:].sort().values
        sparse_rows = sparse_rows.to(torch.int32).contiguous()
        case_label = f"m{rows}_d{explicit_dense_count}"
    dense_count = int(dense_rows.numel())
    paired_contiguous = args.paired_config.endswith("_contiguous")
    single_launch_supported = args.paired_config == (
        "256x64_full_256x32_residual_contiguous"
    )
    dense_slot_by_row = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slot_by_row[dense_rows.long()] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    states = prepare_layers(
        rows=rows,
        hidden=hidden,
        out_features=out_features,
        dense_count=dense_count,
        paired_contiguous=paired_contiguous,
        layer_count=args.layers,
        generator=generator,
    )
    angles = torch.randn(
        (4096, HEAD_DIM // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cos_sin_cache = torch.cat(
        (angles.cos(), angles.sin()), dim=-1
    ).half().contiguous()
    position_ids = torch.arange(rows, device="cuda", dtype=torch.int64)

    from vllm.speclink_linear import _qkv_heterogeneous_config

    heterogeneous_config = (
        _qkv_heterogeneous_config(
            rows,
            int(dense_rows.numel()),
            out_features,
        )
        if args.heterogeneous_config == "production"
        else args.heterogeneous_config
    )
    dense_base_stream = torch.cuda.Stream()
    sparse_correction_stream = torch.cuda.Stream()
    sparse_count = int(sparse_rows.numel())

    def dense_fn() -> torch.Tensor:
        for state in states:
            torch.mm(state.x, state.dense_weight.t(), out=state.dense_out)
        return states[-1].dense_out

    def static_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                out=state.static_out,
                config=args.static_config,
                store_mode="vector",
            )
        return states[-1].static_out

    def heterogeneous_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_heterogeneous_linear_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.dense_weight,
                dense_rows,
                sparse_rows,
                out=state.heterogeneous_out,
                config=heterogeneous_config,
            )
        return states[-1].heterogeneous_out

    def exact_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_routed_exact_linear_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                sparse_rows,
                out=state.exact_out,
                config=args.exact_config,
            )
        return states[-1].exact_out

    def dense_base_sparse_correction_fn() -> torch.Tensor:
        for state in states:
            current_stream = torch.cuda.current_stream()
            dense_base_stream.wait_stream(current_stream)
            sparse_correction_stream.wait_stream(current_stream)
            with torch.cuda.stream(dense_base_stream):
                torch.mm(
                    state.x,
                    state.dense_weight.t(),
                    out=state.dense_base_out,
                )
            with torch.cuda.stream(sparse_correction_stream):
                sparse24_gather_rows_(
                    state.x,
                    sparse_rows,
                    state.sparse_correction_x[:sparse_count],
                )
                sparse24_cutlass_inline_transpose_gemm_prepacked(
                    state.sparse_correction_x,
                    state.residual_values,
                    state.residual_meta,
                    out=state.sparse_correction_out,
                    config=args.static_config,
                    store_mode="vector",
                )
            current_stream.wait_stream(dense_base_stream)
            current_stream.wait_stream(sparse_correction_stream)
            sparse24_sub_indexed_rows_contiguous_(
                state.dense_base_out,
                state.sparse_correction_out[:sparse_count],
                sparse_rows,
            )
        return states[-1].dense_base_out

    def paired_gather_residual_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_paired_gather_residual_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                full_out=state.paired_full_out,
                residual_out=state.paired_residual_out,
                schedule=args.paired_schedule,
                config=args.paired_config,
                worker_blocks=args.paired_worker_blocks,
            )
            if paired_contiguous:
                sparse24_add_indexed_rows_contiguous_(
                    state.paired_full_out,
                    state.paired_residual_out,
                    dense_rows,
                )
            else:
                sparse24_transpose_add_routed_residual(
                    state.paired_full_out,
                    state.paired_residual_out,
                    dense_slot_by_row,
                    dense_count=dense_count,
                    out=state.paired_out,
                )
        return (
            states[-1].paired_full_out
            if paired_contiguous
            else states[-1].paired_out
        )

    def inplace_residual_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_paired_inplace_residual_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                out=state.inplace_out,
                feature_counters=state.inplace_counters,
                config=args.inplace_config,
                worker_blocks=args.inplace_worker_blocks,
            )
        return states[-1].inplace_out

    def dense_qkv_fn() -> torch.Tensor:
        for state in states:
            torch.mm(state.x, state.dense_weight.t(), out=state.dense_out)
            if model == "qwen3_8b":
                _apply_qwen_postop(
                    state.dense_out,
                    state.q_weight,
                    state.k_weight,
                    cos_sin_cache,
                    position_ids,
                )
            else:
                _apply_llama_postop(
                    state.dense_out,
                    cos_sin_cache,
                    position_ids,
                )
        return states[-1].dense_out

    def exact_qkv_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_routed_exact_linear_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                sparse_rows,
                out=state.exact_out,
                config=args.exact_config,
            )
            if model == "qwen3_8b":
                _apply_qwen_postop(
                    state.exact_out,
                    state.q_weight,
                    state.k_weight,
                    cos_sin_cache,
                    position_ids,
                )
            else:
                _apply_llama_postop(
                    state.exact_out,
                    cos_sin_cache,
                    position_ids,
                )
        return states[-1].exact_out

    def paired_gather_residual_qkv_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_paired_gather_residual_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                full_out=state.paired_full_out,
                residual_out=state.paired_residual_out,
                schedule=args.paired_schedule,
                config=args.paired_config,
                worker_blocks=args.paired_worker_blocks,
            )
            fused_kwargs: dict[str, object]
            if model == "qwen3_8b":
                fused_kwargs = {
                    "epsilon": EPSILON,
                    "q_weight": state.q_weight,
                    "k_weight": state.k_weight,
                }
            else:
                fused_kwargs = {"epsilon": 0.0}
            if paired_contiguous:
                sparse24_add_indexed_rows_contiguous_(
                    state.paired_full_out,
                    state.paired_residual_out,
                    dense_rows,
                )
                if model == "qwen3_8b":
                    _apply_qwen_postop(
                        state.paired_full_out,
                        state.q_weight,
                        state.k_weight,
                        cos_sin_cache,
                        position_ids,
                    )
                else:
                    _apply_llama_postop(
                        state.paired_full_out,
                        cos_sin_cache,
                        position_ids,
                    )
            else:
                sparse24_qkv_transpose_add_routed_residual_postop(
                    state.paired_full_out,
                    state.paired_residual_out,
                    dense_slot_by_row,
                    cos_sin_cache,
                    position_ids,
                    q_size=Q_SIZE,
                    kv_size=KV_SIZE,
                    head_dim=HEAD_DIM,
                    is_neox=True,
                    out=state.paired_out,
                    postop_config=args.qkv_postop_config,
                    **fused_kwargs,
                )
        return (
            states[-1].paired_full_out
            if paired_contiguous
            else states[-1].paired_out
        )

    def paired_gather_fused_qkv_fn() -> torch.Tensor:
        if not paired_contiguous:
            raise RuntimeError(
                "the in-place fused QKV epilogue requires a contiguous "
                "paired-output config"
            )
        for state in states:
            sparse24_cutlass_paired_gather_residual_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                full_out=state.paired_full_out,
                residual_out=state.paired_residual_out,
                schedule=args.paired_schedule,
                config=args.paired_config,
                worker_blocks=args.paired_worker_blocks,
            )
            fused_kwargs: dict[str, object]
            if model == "qwen3_8b":
                fused_kwargs = {
                    "epsilon": EPSILON,
                    "q_weight": state.q_weight,
                    "k_weight": state.k_weight,
                }
            else:
                fused_kwargs = {"epsilon": 0.0}
            sparse24_qkv_add_routed_residual_postop_inplace_(
                state.paired_full_out,
                state.paired_residual_out,
                dense_slot_by_row,
                cos_sin_cache,
                position_ids,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                head_dim=HEAD_DIM,
                is_neox=True,
                postop_config=args.qkv_postop_config,
                **fused_kwargs,
            )
        return states[-1].paired_full_out

    def paired_single_launch_qkv_fn() -> torch.Tensor:
        if args.paired_config != (
            "256x64_full_256x32_residual_contiguous"
        ):
            raise RuntimeError(
                "single-launch paired QKV currently requires config 13"
            )
        for state in states:
            fused_kwargs: dict[str, object]
            if model == "qwen3_8b":
                fused_kwargs = {
                    "epsilon": EPSILON,
                    "q_weight": state.q_weight,
                    "k_weight": state.k_weight,
                }
            else:
                fused_kwargs = {"epsilon": 0.0}
            sparse24_cutlass_paired_gather_residual_qkv_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                dense_slot_by_row,
                cos_sin_cache,
                position_ids,
                state.paired_grid_barrier,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                head_dim=HEAD_DIM,
                is_neox=True,
                full_out=state.paired_full_out,
                residual_out=state.paired_residual_out,
                schedule=args.paired_schedule,
                config=args.paired_config,
                worker_blocks=args.paired_worker_blocks,
                **fused_kwargs,
            )
        return states[-1].paired_full_out

    def paired_epilogue_qkv_fn() -> torch.Tensor:
        if not single_launch_supported:
            raise RuntimeError(
                "routed QKV epilogue currently requires paired config 13"
            )
        for state in states:
            fused_kwargs: dict[str, object]
            if model == "qwen3_8b":
                fused_kwargs = {
                    "epsilon": EPSILON,
                    "q_weight": state.q_weight,
                    "k_weight": state.k_weight,
                }
            else:
                fused_kwargs = {"epsilon": 0.0}
            sparse24_cutlass_paired_fused_routed_qkv_epilogue_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                dense_slot_by_row,
                state.paired_qkv_dense_base,
                cos_sin_cache,
                position_ids,
                state.paired_qkv_counters,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                head_dim=HEAD_DIM,
                is_neox=True,
                out=state.paired_qkv_epilogue_out,
                worker_blocks=args.qkv_epilogue_worker_blocks,
                residual_worker_blocks=(
                    args.qkv_epilogue_residual_worker_blocks
                ),
                **fused_kwargs,
            )
        return states[-1].paired_qkv_epilogue_out

    def paired_finalize_qkv_fn() -> torch.Tensor:
        if not paired_contiguous:
            raise RuntimeError(
                "distributed QKV finalization requires contiguous outputs"
            )
        for state in states:
            fused_kwargs: dict[str, object]
            if model == "qwen3_8b":
                fused_kwargs = {
                    "epsilon": EPSILON,
                    "q_weight": state.q_weight,
                    "k_weight": state.k_weight,
                }
            else:
                fused_kwargs = {"epsilon": 0.0}
            sparse24_cutlass_paired_finalize_qkv_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                cos_sin_cache,
                position_ids,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                head_dim=HEAD_DIM,
                is_neox=True,
                out=state.paired_qkv_epilogue_out,
                residual_out=state.paired_residual_out,
                feature_counters=state.paired_qkv_counters,
                config=args.qkv_finalize_config,
                worker_blocks=args.qkv_finalize_worker_blocks,
                schedule=args.qkv_finalize_schedule,
                **fused_kwargs,
            )
        return states[-1].paired_qkv_epilogue_out

    def grouped_owner_qkv_fn() -> torch.Tensor:
        for state in states:
            fused_kwargs: dict[str, object]
            if model == "qwen3_8b":
                fused_kwargs = {
                    "epsilon": EPSILON,
                    "q_weight": state.q_weight,
                    "k_weight": state.k_weight,
                }
            else:
                fused_kwargs = {"epsilon": 0.0}
            sparse24_cutlass_grouped_owner_qkv_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                dense_slot_by_row,
                state.paired_qkv_dense_base,
                cos_sin_cache,
                position_ids,
                q_size=Q_SIZE,
                kv_size=KV_SIZE,
                head_dim=HEAD_DIM,
                is_neox=True,
                out=state.paired_qkv_epilogue_out,
                group_tiles=args.grouped_owner_qkv_group_tiles,
                config=args.grouped_owner_qkv_config,
                **fused_kwargs,
            )
        return states[-1].paired_qkv_epilogue_out

    def dense_base_sparse_correction_qkv_fn() -> torch.Tensor:
        dense_base_sparse_correction_fn()
        for state in states:
            if model == "qwen3_8b":
                _apply_qwen_postop(
                    state.dense_base_out,
                    state.q_weight,
                    state.k_weight,
                    cos_sin_cache,
                    position_ids,
                )
            else:
                _apply_llama_postop(
                    state.dense_base_out,
                    cos_sin_cache,
                    position_ids,
                )
        return states[-1].dense_base_out

    def inplace_residual_qkv_fn() -> torch.Tensor:
        for state in states:
            sparse24_cutlass_paired_inplace_residual_prepacked(
                state.x,
                state.sparse_values,
                state.sparse_meta,
                state.residual_values,
                state.residual_meta,
                dense_rows,
                out=state.inplace_out,
                feature_counters=state.inplace_counters,
                config=args.inplace_config,
                worker_blocks=args.inplace_worker_blocks,
            )
            if model == "qwen3_8b":
                _apply_qwen_postop(
                    state.inplace_out,
                    state.q_weight,
                    state.k_weight,
                    cos_sin_cache,
                    position_ids,
                )
            else:
                _apply_llama_postop(
                    state.inplace_out,
                    cos_sin_cache,
                    position_ids,
                )
        return states[-1].inplace_out

    heterogeneous_fn()
    exact_fn()
    dense_base_sparse_correction_fn()
    paired_gather_residual_fn()
    inplace_residual_fn()
    torch.cuda.synchronize()
    max_abs_diff = max(
        float(
            (state.heterogeneous_out.float() - state.exact_out.float())
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if max_abs_diff > args.atol:
        raise RuntimeError(
            "heterogeneous and complementary 2:4 routes disagree: "
            f"max_abs_diff={max_abs_diff:.6f}"
        )
    dense_base_max_abs_diff = max(
        float(
            (state.heterogeneous_out.float() - state.dense_base_out.float())
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if dense_base_max_abs_diff > args.atol:
        raise RuntimeError(
            "heterogeneous and dense-base sparse-correction routes disagree: "
            f"max_abs_diff={dense_base_max_abs_diff:.6f}"
        )
    paired_max_abs_diff = max(
        float(
            (
                state.heterogeneous_out.float()
                - (
                    state.paired_full_out.float()
                    if paired_contiguous
                    else state.paired_out.float()
                )
            )
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if paired_max_abs_diff > args.atol:
        raise RuntimeError(
            "heterogeneous and paired gather residual routes disagree: "
            f"max_abs_diff={paired_max_abs_diff:.6f}"
        )
    inplace_max_abs_diff = max(
        float(
            (state.heterogeneous_out.float() - state.inplace_out.float())
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if inplace_max_abs_diff > args.atol:
        raise RuntimeError(
            "heterogeneous and in-place residual routes disagree: "
            f"max_abs_diff={inplace_max_abs_diff:.6f}"
        )
    counter_max = max(
        int(state.inplace_counters.abs().max().item()) for state in states
    )
    if counter_max != 0:
        raise RuntimeError(
            f"in-place residual feature counters did not reset: {counter_max}"
        )

    exact_qkv_fn()
    dense_base_sparse_correction_qkv_fn()
    paired_gather_residual_qkv_fn()
    inplace_residual_qkv_fn()
    torch.cuda.synchronize()
    paired_qkv_max_abs_diff = max(
        float(
            (
                state.exact_out.float()
                - (
                    state.paired_full_out.float()
                    if paired_contiguous
                    else state.paired_out.float()
                )
            )
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if paired_qkv_max_abs_diff > args.qkv_atol:
        raise RuntimeError(
            "exact and paired gather residual full QKV stages disagree: "
            f"max_abs_diff={paired_qkv_max_abs_diff:.6f}"
        )
    inplace_qkv_max_abs_diff = max(
        float(
            (state.exact_out.float() - state.inplace_out.float())
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if inplace_qkv_max_abs_diff > args.qkv_atol:
        raise RuntimeError(
            "exact and in-place residual full QKV stages disagree: "
            f"max_abs_diff={inplace_qkv_max_abs_diff:.6f}"
        )
    paired_fused_qkv_max_abs_diff = float("nan")
    if paired_contiguous:
        exact_qkv_fn()
        paired_gather_fused_qkv_fn()
        torch.cuda.synchronize()
        paired_fused_qkv_max_abs_diff = max(
            float(
                (state.exact_out.float() - state.paired_full_out.float())
                .abs()
                .max()
                .item()
            )
            for state in states
        )
        if paired_fused_qkv_max_abs_diff > args.qkv_atol:
            raise RuntimeError(
                "exact and fused contiguous paired QKV stages disagree: "
                f"max_abs_diff={paired_fused_qkv_max_abs_diff:.6f}"
            )
    single_launch_qkv_max_abs_diff = float("nan")
    if single_launch_supported:
        exact_qkv_fn()
        paired_single_launch_qkv_fn()
        torch.cuda.synchronize()
        single_launch_qkv_max_abs_diff = max(
            float(
                (state.exact_out.float() - state.paired_full_out.float())
                .abs()
                .max()
                .item()
            )
            for state in states
        )
        if single_launch_qkv_max_abs_diff > args.qkv_atol:
            raise RuntimeError(
                "exact and single-launch paired QKV stages disagree: "
                f"max_abs_diff={single_launch_qkv_max_abs_diff:.6f}"
            )
        barrier_arrivals = max(
            int(state.paired_grid_barrier[0].item()) for state in states
        )
        if barrier_arrivals != 0:
            raise RuntimeError(
                "single-launch paired QKV barrier did not reset: "
                f"arrivals={barrier_arrivals}"
            )
    epilogue_qkv_max_abs_diff = float("nan")
    if single_launch_supported:
        exact_qkv_fn()
        paired_epilogue_qkv_fn()
        torch.cuda.synchronize()
        epilogue_qkv_max_abs_diff = max(
            float(
                (
                    state.exact_out.float()
                    - state.paired_qkv_epilogue_out.float()
                )
                .abs()
                .max()
                .item()
            )
            for state in states
        )
        if epilogue_qkv_max_abs_diff > args.qkv_atol:
            raise RuntimeError(
                "exact and routed QKV epilogues disagree: "
                f"max_abs_diff={epilogue_qkv_max_abs_diff:.6f}"
            )
        counter_max = max(
            int(state.paired_qkv_counters.abs().max().item())
            for state in states
        )
        if counter_max != 0:
            raise RuntimeError(
                "routed QKV epilogue counters did not reset: "
                f"max={counter_max}"
            )

    finalize_qkv_max_abs_diff = float("nan")
    if paired_contiguous:
        exact_qkv_fn()
        paired_finalize_qkv_fn()
        torch.cuda.synchronize()
        finalize_qkv_max_abs_diff = max(
            float(
                (
                    state.exact_out.float()
                    - state.paired_qkv_epilogue_out.float()
                )
                .abs()
                .max()
                .item()
            )
            for state in states
        )
        if finalize_qkv_max_abs_diff > args.qkv_atol:
            raise RuntimeError(
                "exact and distributed finalized QKV disagree: "
                f"max_abs_diff={finalize_qkv_max_abs_diff:.6f}"
            )
        counter_max = max(
            int(state.paired_qkv_counters.abs().max().item())
            for state in states
        )
        if counter_max != 0:
            raise RuntimeError(
                "distributed QKV finalizer counters did not reset: "
                f"max={counter_max}"
            )

    exact_qkv_fn()
    grouped_owner_qkv_fn()
    torch.cuda.synchronize()
    grouped_owner_qkv_max_abs_diff = max(
        float(
            (
                state.exact_out.float()
                - state.paired_qkv_epilogue_out.float()
            )
            .abs()
            .max()
            .item()
        )
        for state in states
    )
    if grouped_owner_qkv_max_abs_diff > args.qkv_atol:
        raise RuntimeError(
            "exact and grouped-owner QKV epilogues disagree: "
            f"max_abs_diff={grouped_owner_qkv_max_abs_diff:.6f}"
        )

    timings: dict[str, float] = {}
    candidates = [
        ("heterogeneous", heterogeneous_fn),
        ("complementary_exact", exact_fn),
        ("dense_base_sparse_correction", dense_base_sparse_correction_fn),
        ("paired_gather_residual", paired_gather_residual_fn),
        ("paired_inplace_residual", inplace_residual_fn),
    ]
    static_supported = rows % 8 == 0
    if static_supported:
        candidates.insert(0, ("static_sparse", static_fn))
    else:
        timings["dense_for_static_sparse_ms"] = float("nan")
        timings["static_sparse_ms"] = float("nan")

    for name, candidate in candidates:
        dense_ms, candidate_ms = paired_graph_median_ms(
            dense_fn,
            candidate,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        timings[f"dense_for_{name}_ms"] = dense_ms / args.layers
        timings[f"{name}_ms"] = candidate_ms / args.layers

    dense_qkv_ms, paired_qkv_ms = paired_graph_median_ms(
        dense_qkv_fn,
        paired_gather_residual_qkv_fn,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    timings["dense_qkv_ms"] = dense_qkv_ms / args.layers
    timings["paired_gather_residual_qkv_ms"] = paired_qkv_ms / args.layers
    if paired_contiguous:
        fused_dense_qkv_ms, paired_fused_qkv_ms = paired_graph_median_ms(
            dense_qkv_fn,
            paired_gather_fused_qkv_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        timings["dense_for_paired_fused_qkv_ms"] = (
            fused_dense_qkv_ms / args.layers
        )
        timings["paired_fused_qkv_ms"] = paired_fused_qkv_ms / args.layers
    else:
        timings["dense_for_paired_fused_qkv_ms"] = float("nan")
        timings["paired_fused_qkv_ms"] = float("nan")
    if single_launch_supported:
        single_dense_qkv_ms, single_launch_qkv_ms = paired_graph_median_ms(
            dense_qkv_fn,
            paired_single_launch_qkv_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        timings["dense_for_single_launch_qkv_ms"] = (
            single_dense_qkv_ms / args.layers
        )
        timings["single_launch_qkv_ms"] = single_launch_qkv_ms / args.layers
    else:
        timings["dense_for_single_launch_qkv_ms"] = float("nan")
        timings["single_launch_qkv_ms"] = float("nan")
    if single_launch_supported:
        epilogue_dense_qkv_ms, epilogue_qkv_ms = paired_graph_median_ms(
            dense_qkv_fn,
            paired_epilogue_qkv_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        timings["dense_for_epilogue_qkv_ms"] = (
            epilogue_dense_qkv_ms / args.layers
        )
        timings["epilogue_qkv_ms"] = epilogue_qkv_ms / args.layers
    else:
        timings["dense_for_epilogue_qkv_ms"] = float("nan")
        timings["epilogue_qkv_ms"] = float("nan")
    if paired_contiguous:
        finalize_dense_qkv_ms, finalize_qkv_ms = paired_graph_median_ms(
            dense_qkv_fn,
            paired_finalize_qkv_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        timings["dense_for_finalize_qkv_ms"] = (
            finalize_dense_qkv_ms / args.layers
        )
        timings["finalize_qkv_ms"] = finalize_qkv_ms / args.layers
    else:
        timings["dense_for_finalize_qkv_ms"] = float("nan")
        timings["finalize_qkv_ms"] = float("nan")
    grouped_dense_qkv_ms, grouped_owner_qkv_ms = paired_graph_median_ms(
        dense_qkv_fn,
        grouped_owner_qkv_fn,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    timings["dense_for_grouped_owner_qkv_ms"] = (
        grouped_dense_qkv_ms / args.layers
    )
    timings["grouped_owner_qkv_ms"] = grouped_owner_qkv_ms / args.layers
    dense_base_qkv_ms, dense_base_sparse_qkv_ms = paired_graph_median_ms(
        dense_qkv_fn,
        dense_base_sparse_correction_qkv_fn,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    timings["dense_for_dense_base_sparse_qkv_ms"] = (
        dense_base_qkv_ms / args.layers
    )
    timings["dense_base_sparse_qkv_ms"] = (
        dense_base_sparse_qkv_ms / args.layers
    )
    inplace_dense_qkv_ms, inplace_qkv_ms = paired_graph_median_ms(
        dense_qkv_fn,
        inplace_residual_qkv_fn,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    timings["dense_for_inplace_qkv_ms"] = inplace_dense_qkv_ms / args.layers
    timings["paired_inplace_residual_qkv_ms"] = inplace_qkv_ms / args.layers

    dense_bytes = states[0].dense_weight.numel() * states[0].dense_weight.element_size()
    sparse_bytes = (
        states[0].sparse_values.numel() * states[0].sparse_values.element_size()
        + states[0].sparse_meta.numel() * states[0].sparse_meta.element_size()
    )
    residual_bytes = (
        states[0].residual_values.numel()
        * states[0].residual_values.element_size()
        + states[0].residual_meta.numel()
        * states[0].residual_meta.element_size()
    )
    row = {
        "model": model,
        "case": case_label,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "layers": args.layers,
        "dense_cap": args.dense_cap,
        "dense_rows": int(dense_rows.numel()),
        "sparse_rows": int(sparse_rows.numel()),
        "dense_fraction": int(dense_rows.numel()) / rows,
        "heterogeneous_config": heterogeneous_config,
        "exact_config": args.exact_config,
        "paired_config": args.paired_config,
        "paired_schedule": args.paired_schedule,
        "paired_worker_blocks": args.paired_worker_blocks,
        "qkv_epilogue_worker_blocks": args.qkv_epilogue_worker_blocks,
        "qkv_epilogue_residual_worker_blocks": (
            args.qkv_epilogue_residual_worker_blocks
        ),
        "qkv_finalize_config": args.qkv_finalize_config,
        "qkv_finalize_worker_blocks": args.qkv_finalize_worker_blocks,
        "qkv_finalize_schedule": args.qkv_finalize_schedule,
        "grouped_owner_qkv_group_tiles": args.grouped_owner_qkv_group_tiles,
        "grouped_owner_qkv_config": args.grouped_owner_qkv_config,
        "inplace_config": args.inplace_config,
        "inplace_worker_blocks": args.inplace_worker_blocks,
        "static_config": args.static_config,
        "static_supported": static_supported,
        "dense_weight_mib": dense_bytes / (1 << 20),
        "heterogeneous_weight_mib": (dense_bytes + sparse_bytes) / (1 << 20),
        "complementary_weight_mib": (sparse_bytes + residual_bytes) / (1 << 20),
        **timings,
        "static_sparse_speedup": (
            timings["dense_for_static_sparse_ms"]
            / timings["static_sparse_ms"]
        ),
        "heterogeneous_speedup": (
            timings["dense_for_heterogeneous_ms"]
            / timings["heterogeneous_ms"]
        ),
        "complementary_exact_speedup": (
            timings["dense_for_complementary_exact_ms"]
            / timings["complementary_exact_ms"]
        ),
        "dense_base_sparse_correction_speedup": (
            timings["dense_for_dense_base_sparse_correction_ms"]
            / timings["dense_base_sparse_correction_ms"]
        ),
        "dense_base_sparse_qkv_speedup": (
            timings["dense_for_dense_base_sparse_qkv_ms"]
            / timings["dense_base_sparse_qkv_ms"]
        ),
        "paired_gather_residual_speedup": (
            timings["dense_for_paired_gather_residual_ms"]
            / timings["paired_gather_residual_ms"]
        ),
        "paired_gather_residual_qkv_speedup": (
            timings["dense_qkv_ms"]
            / timings["paired_gather_residual_qkv_ms"]
        ),
        "paired_fused_qkv_speedup": (
            timings["dense_for_paired_fused_qkv_ms"]
            / timings["paired_fused_qkv_ms"]
        ),
        "single_launch_qkv_speedup": (
            timings["dense_for_single_launch_qkv_ms"]
            / timings["single_launch_qkv_ms"]
        ),
        "epilogue_qkv_speedup": (
            timings["dense_for_epilogue_qkv_ms"]
            / timings["epilogue_qkv_ms"]
        ),
        "finalize_qkv_speedup": (
            timings["dense_for_finalize_qkv_ms"]
            / timings["finalize_qkv_ms"]
        ),
        "grouped_owner_qkv_speedup": (
            timings["dense_for_grouped_owner_qkv_ms"]
            / timings["grouped_owner_qkv_ms"]
        ),
        "paired_inplace_residual_speedup": (
            timings["dense_for_paired_inplace_residual_ms"]
            / timings["paired_inplace_residual_ms"]
        ),
        "paired_inplace_residual_qkv_speedup": (
            timings["dense_for_inplace_qkv_ms"]
            / timings["paired_inplace_residual_qkv_ms"]
        ),
        "max_abs_diff": max_abs_diff,
        "dense_base_max_abs_diff": dense_base_max_abs_diff,
        "paired_max_abs_diff": paired_max_abs_diff,
        "paired_qkv_max_abs_diff": paired_qkv_max_abs_diff,
        "paired_fused_qkv_max_abs_diff": paired_fused_qkv_max_abs_diff,
        "single_launch_qkv_max_abs_diff": single_launch_qkv_max_abs_diff,
        "epilogue_qkv_max_abs_diff": epilogue_qkv_max_abs_diff,
        "finalize_qkv_max_abs_diff": finalize_qkv_max_abs_diff,
        "grouped_owner_qkv_max_abs_diff": grouped_owner_qkv_max_abs_diff,
        "inplace_max_abs_diff": inplace_max_abs_diff,
        "inplace_qkv_max_abs_diff": inplace_qkv_max_abs_diff,
    }
    del states, dense_rows, sparse_rows, dense_slot_by_row
    gc.collect()
    torch.cuda.empty_cache()
    return row


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        (
            f"{row['model'].replace('_8b', '')}\n{row['case']}"
            if row["case"]
            else f"{row['model'].replace('_8b', '')}\nbs{row['batch_size']}/K{row['K']}"
        )
        for row in rows
    ]
    positions = list(range(len(rows)))
    width = 0.074
    figure, axis = plt.subplots(figsize=(max(7.5, 1.35 * len(rows)), 4.6))
    for offset, field, label, color in (
        (-5.0 * width, "static_sparse_speedup", "Whole-batch 2:4 upper", "#457B9D"),
        (-4.0 * width, "heterogeneous_speedup", "W24 + dense W", "#D45D79"),
        (-3.0 * width, "complementary_exact_speedup", "Split rows: W24 + R24", "#E9C46A"),
        (-2.0 * width, "dense_base_sparse_qkv_speedup", "Dense base - sparse R24", "#1D3557"),
        (-1.0 * width, "paired_gather_residual_qkv_speedup", "Paired + add", "#2A9D8F"),
        (0.0, "paired_fused_qkv_speedup", "Paired + vec8 post-op", "#F4A261"),
        (1.0 * width, "single_launch_qkv_speedup", "Global-barrier QKV", "#E76F51"),
        (2.0 * width, "epilogue_qkv_speedup", "Inline QKV epilogue", "#588157"),
        (3.0 * width, "finalize_qkv_speedup", "Distributed finalizer", "#3A7D44"),
        (4.0 * width, "grouped_owner_qkv_speedup", "Grouped-owner QKV", "#8A5A44"),
        (5.0 * width, "paired_inplace_residual_qkv_speedup", "Paired in-place", "#6A4C93"),
    ):
        axis.bar(
            [position + offset for position in positions],
            [float(row[field]) for row in rows],
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(1.0, color="#222222", linewidth=1)
    axis.set_ylabel("Layer-cold QKV speedup vs dense")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Layer-cold mixed-row QKV routing",
        "",
        "Each timing rotates independent layer weights inside one CUDA Graph, so a",
        "single layer cannot remain resident in L2 between calls.",
        "",
        "| Model | bs/K | Rows | Dense rows | Static 2:4 | W24 + W | Split-row W24 + R24 | Dense base - sparse R24 | Paired projection | Paired full QKV | Paired + vec8 | Global-barrier QKV | Inline QKV epilogue | Distributed finalizer | Grouped-owner QKV | In-place projection | In-place full QKV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        point_label = (
            str(row["case"])
            if row["case"]
            else f"{row['batch_size']}/{row['K']}"
        )
        lines.append(
            f"| {row['model']} | {point_label} | {row['rows']} | "
            f"{row['dense_rows']} | {float(row['static_sparse_speedup']):.3f}x | "
            f"{float(row['heterogeneous_speedup']):.3f}x | "
            f"{float(row['complementary_exact_speedup']):.3f}x | "
            f"{float(row['dense_base_sparse_qkv_speedup']):.3f}x | "
            f"{float(row['paired_gather_residual_speedup']):.3f}x | "
            f"{float(row['paired_gather_residual_qkv_speedup']):.3f}x | "
            f"{float(row['paired_fused_qkv_speedup']):.3f}x | "
            f"{float(row['single_launch_qkv_speedup']):.3f}x | "
            f"{float(row['epilogue_qkv_speedup']):.3f}x | "
            f"{float(row['finalize_qkv_speedup']):.3f}x | "
            f"{float(row['grouped_owner_qkv_speedup']):.3f}x | "
            f"{float(row['paired_inplace_residual_speedup']):.3f}x | "
            f"{float(row['paired_inplace_residual_qkv_speedup']):.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument("--batch-sizes", default="16")
    parser.add_argument("--k-values", default="6")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dense-ratio", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument("--dense-cap", type=int, default=-1)
    parser.add_argument(
        "--explicit-cases",
        default="",
        help=(
            "comma-separated live row shapes as ROWS:DENSE_ROWS; when set, "
            "the batch/K matrix is skipped"
        ),
    )
    parser.add_argument("--heterogeneous-config", default="production")
    parser.add_argument("--exact-config", default="128x32x64_s4_sw4")
    parser.add_argument("--static-config", default="256x32x64_s3_sw4")
    parser.add_argument(
        "--paired-config", default="256x64_full_256x64_residual"
    )
    parser.add_argument(
        "--paired-schedule", choices=("partitioned", "interleaved"),
        default="partitioned",
    )
    parser.add_argument("--paired-worker-blocks", type=int, default=0)
    parser.add_argument("--qkv-epilogue-worker-blocks", type=int, default=0)
    parser.add_argument(
        "--qkv-epilogue-residual-worker-blocks", type=int, default=0
    )
    parser.add_argument(
        "--qkv-finalize-config",
        choices=(
            "256x32_full_256x32_residual_finalize_qkv",
            "256x64_full_256x64_residual_finalize_qkv",
            "256x64_full_256x32_residual_finalize_qkv",
        ),
        default="256x64_full_256x32_residual_finalize_qkv",
    )
    parser.add_argument("--qkv-finalize-worker-blocks", type=int, default=0)
    parser.add_argument(
        "--qkv-finalize-schedule",
        choices=("partitioned", "interleaved"),
        default="partitioned",
    )
    parser.add_argument("--grouped-owner-qkv-group-tiles", type=int, default=2)
    parser.add_argument(
        "--grouped-owner-qkv-config", default="256x32x64_s3_w64x32"
    )
    parser.add_argument(
        "--inplace-config",
        default="256x32_full_256x32_residual_inplace",
    )
    parser.add_argument("--inplace-worker-blocks", type=int, default=0)
    parser.add_argument("--sparse-accumulator", default="fp16_qkv_gate")
    parser.add_argument("--unroll", type=int, default=1)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--qkv-atol", type=float, default=0.25)
    parser.add_argument("--qkv-postop-config", default="16x8")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.layers < 2:
        raise ValueError("--layers must be at least 2 for a cold-weight test")
    if args.dense_cap < -1:
        raise ValueError("--dense-cap must be -1 or non-negative")
    os.environ["SPECLINK_SPARSE24_ACCUMULATOR"] = args.sparse_accumulator
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    explicit_cases: list[tuple[int, int]] = []
    if args.explicit_cases.strip():
        for item in parse_csv_strings(args.explicit_cases):
            try:
                row_text, dense_text = item.split(":", maxsplit=1)
                explicit_case = (int(row_text), int(dense_text))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid --explicit-cases item {item!r}; expected ROWS:DENSE_ROWS"
                ) from error
            explicit_rows, explicit_dense = explicit_case
            if explicit_rows <= 0 or not 0 < explicit_dense < explicit_rows:
                raise ValueError(
                    "explicit cases require 0 < DENSE_ROWS < ROWS, got "
                    f"{item!r}"
                )
            explicit_cases.append(explicit_case)
    for model in parse_csv_strings(args.models):
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r}")
        if explicit_cases:
            for explicit_case in explicit_cases:
                row = run_point(
                    model=model,
                    batch_size=0,
                    k=0,
                    explicit_case=explicit_case,
                    generator=generator,
                    args=args,
                )
                rows.append(row)
                print(row, flush=True)
            continue
        for batch_size in parse_csv_ints(args.batch_sizes):
            for k in parse_csv_ints(args.k_values):
                row = run_point(
                    model=model,
                    batch_size=batch_size,
                    k=k,
                    explicit_case=None,
                    generator=generator,
                    args=args,
                )
                rows.append(row)
                print(row, flush=True)

    csv_path = args.output_root / "cold_qkv_routing.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_plot(args.output_root / "cold_qkv_routing.png", rows)
    write_report(args.output_root / "report.md", rows)
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
