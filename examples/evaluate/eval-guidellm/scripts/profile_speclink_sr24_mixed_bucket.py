#!/usr/bin/env python3
"""Microbenchmark SR24 mixed sparse-base + dense-row residual bucket paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.sparse import to_sparse_semi_structured


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be ROWS,OUT,IN")
    return parts[0], parts[1], parts[2]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_base_24(weight: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % 4:
        raise ValueError("in_features must be divisible by 4")
    grouped = weight.view(out_features, in_features // 4, 4)
    keep_idx = grouped.abs().topk(2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    base = torch.zeros_like(grouped)
    base[keep] = grouped[keep]
    return base.view_as(weight).contiguous()


def direct_cslt_linear(
    x: torch.Tensor,
    sparse_weight: Any,
    *,
    alg_id: int = 0,
) -> torch.Tensor:
    dense_input = sparse_weight._pad_dense_input(x)
    out = torch._cslt_sparse_mm(
        sparse_weight.packed,
        dense_input.t().contiguous(),
        transpose_result=False,
        alg_id=alg_id,
    ).t()
    return out[: x.shape[0], :]


@triton.jit
def _routed_assemble_kernel(
    dense_output,
    dense_rows,
    base_output,
    base_rows,
    output,
    dense_count: tl.constexpr,
    total_rows: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    col_mask = cols < out_features
    is_dense = row < dense_count
    base_row = row - dense_count
    dense_dst_row = tl.load(dense_rows + row, mask=is_dense, other=0)
    base_dst_row = tl.load(base_rows + base_row, mask=~is_dense, other=0)
    dst_row = tl.where(is_dense, dense_dst_row, base_dst_row)
    dense_vals = tl.load(
        dense_output + row * out_features + cols,
        mask=is_dense & col_mask,
        other=0.0,
    )
    base_vals = tl.load(
        base_output + base_row * out_features + cols,
        mask=(~is_dense) & col_mask,
        other=0.0,
    )
    values = dense_vals + base_vals
    tl.store(output + dst_row * out_features + cols, values, mask=col_mask)


def triton_routed_assemble(
    dense_output: torch.Tensor,
    dense_rows: torch.Tensor,
    base_output: torch.Tensor,
    base_rows: torch.Tensor,
    *,
    total_rows: int,
    out_features: int,
    block_n: int = 1024,
) -> torch.Tensor:
    output = torch.empty(
        (total_rows, out_features),
        device=dense_output.device,
        dtype=dense_output.dtype,
    )
    dense_count = int(dense_output.shape[0])
    base_count = int(base_output.shape[0])
    if dense_count + base_count != total_rows:
        raise RuntimeError(
            "routed assembly row mismatch: "
            f"dense={dense_count}, base={base_count}, total={total_rows}"
        )
    grid = (total_rows, triton.cdiv(out_features, block_n))
    _routed_assemble_kernel[grid](
        dense_output,
        dense_rows,
        base_output,
        base_rows,
        output,
        dense_count,
        total_rows,
        out_features,
        block_n,
        num_warps=8,
    )
    return output


@triton.jit
def _bucket_override_kernel(
    dense_output,
    bucket_rows,
    bucket_values,
    output,
    bucket_count: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    col_mask = cols < out_features
    dst_row = tl.load(bucket_rows + row, mask=row < bucket_count, other=0)
    use_dense = tl.load(bucket_values + row, mask=row < bucket_count, other=0) != 0
    vals = tl.load(
        dense_output + row * out_features + cols,
        mask=(row < bucket_count) & use_dense & col_mask,
        other=0.0,
    )
    tl.store(
        output + dst_row * out_features + cols,
        vals,
        mask=(row < bucket_count) & use_dense & col_mask,
    )


def triton_bucket_override(
    base_output: torch.Tensor,
    dense_output: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
    *,
    out_features: int,
    block_n: int = 1024,
) -> torch.Tensor:
    base_output = base_output.contiguous()
    output = base_output.clone()
    bucket_rows = bucket_rows.contiguous()
    bucket_values = bucket_values.to(dtype=torch.bool).contiguous()
    dense_output = dense_output.contiguous()
    bucket_count = int(bucket_rows.numel())
    grid = (bucket_count, triton.cdiv(out_features, block_n))
    _bucket_override_kernel[grid](
        dense_output,
        bucket_rows,
        bucket_values,
        output,
        bucket_count,
        out_features,
        block_n,
        num_warps=8,
    )
    return output


def triton_bucket_override_inplace(
    base_output: torch.Tensor,
    dense_output: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
    *,
    out_features: int,
    block_n: int = 1024,
) -> torch.Tensor:
    base_output = base_output.contiguous()
    bucket_rows = bucket_rows.contiguous()
    bucket_values = bucket_values.to(dtype=torch.bool).contiguous()
    dense_output = dense_output.contiguous()
    bucket_count = int(bucket_rows.numel())
    grid = (bucket_count, triton.cdiv(out_features, block_n))
    _bucket_override_kernel[grid](
        dense_output,
        bucket_rows,
        bucket_values,
        base_output,
        bucket_count,
        out_features,
        block_n,
        num_warps=8,
    )
    return base_output


@triton.jit
def _bucket_dense_gemm_scatter_kernel(
    input_ptr,
    weight_ptr,
    bucket_rows_ptr,
    bucket_values_ptr,
    output_ptr,
    bucket_count: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    input_stride_m: tl.constexpr,
    input_stride_k: tl.constexpr,
    weight_stride_n: tl.constexpr,
    weight_stride_k: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    bucket_valid = offs_m < bucket_count
    bucket_rows = tl.load(bucket_rows_ptr + offs_m, mask=bucket_valid, other=0)
    bucket_values = tl.load(
        bucket_values_ptr + offs_m,
        mask=bucket_valid,
        other=0,
    )
    row_valid = bucket_valid & (bucket_values != 0)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in range(0, in_features, block_k):
        offs_k = k_start + tl.arange(0, block_k)
        k_mask = offs_k < in_features
        x = tl.load(
            input_ptr
            + bucket_rows[:, None] * input_stride_m
            + offs_k[None, :] * input_stride_k,
            mask=row_valid[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + offs_n[:, None] * weight_stride_n
            + offs_k[None, :] * weight_stride_k,
            mask=(offs_n[:, None] < out_features) & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        output_ptr
        + bucket_rows[:, None] * output_stride_m
        + offs_n[None, :] * output_stride_n,
        acc,
        mask=row_valid[:, None] & (offs_n[None, :] < out_features),
    )


def triton_bucket_dense_gemm_scatter_inplace(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
    base_output: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    input_contig = input_tensor.contiguous()
    weight_contig = dense_weight.contiguous()
    output_contig = base_output.contiguous()
    bucket_rows_contig = bucket_rows.to(
        device=input_tensor.device,
        dtype=torch.long,
        non_blocking=True,
    ).contiguous()
    bucket_values_contig = bucket_values.to(
        device=input_tensor.device,
        dtype=torch.bool,
        non_blocking=True,
    ).contiguous()
    bucket_count = int(bucket_rows_contig.numel())
    if bucket_count <= 0:
        return output_contig
    in_features = int(input_contig.shape[1])
    out_features = int(weight_contig.shape[0])
    grid = (
        triton.cdiv(bucket_count, block_m),
        triton.cdiv(out_features, block_n),
    )
    _bucket_dense_gemm_scatter_kernel[grid](
        input_contig,
        weight_contig,
        bucket_rows_contig,
        bucket_values_contig,
        output_contig,
        bucket_count,
        in_features,
        out_features,
        int(input_contig.stride(0)),
        int(input_contig.stride(1)),
        int(weight_contig.stride(0)),
        int(weight_contig.stride(1)),
        int(output_contig.stride(0)),
        int(output_contig.stride(1)),
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )
    return output_contig


def time_call(fn: Callable[[], Any], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def time_graph(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float | None, str | None]:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / repeats), None
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def fmt_ms(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def fmt_ratio(value: Any, base: float) -> str:
    return "" if value is None else f"{float(value) / base:.2f}x"


def fmt_delta_ms(value: Any, lower_bound: Any) -> str:
    if value is None or lower_bound is None:
        return ""
    return f"{float(value) - float(lower_bound):.4f}"


def fmt_target_gap(value: Any, target: float) -> str:
    if value is None:
        return ""
    return f"{float(value) - target:.4f}"


def run_case(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    bucket_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    triton_block_n: int,
    triton_dense_block_m: int,
    triton_dense_block_n: int,
    triton_dense_block_k: int,
) -> dict[str, Any]:
    bucket_size = min(rows, bucket_size)
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    dense_weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
    base_weight = make_base_24(dense_weight)
    base_sparse = to_sparse_semi_structured(base_weight)
    bucket_rows = torch.arange(bucket_size, device="cuda", dtype=torch.long)
    bucket_values = torch.ones(bucket_size, device="cuda", dtype=torch.bool)
    base_rows = torch.arange(bucket_size, rows, device="cuda", dtype=torch.long)
    torch.cuda.synchronize()

    def dense_full() -> torch.Tensor:
        return F.linear(x, dense_weight)

    def base_sparse_full() -> torch.Tensor:
        return F.linear(x, base_sparse)

    def base_cslt_full() -> torch.Tensor:
        return direct_cslt_linear(x, base_sparse, alg_id=0)

    def bucket_delta() -> torch.Tensor:
        base_output = base_sparse_full()
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        base_selected = base_output.index_select(0, bucket_rows)
        output = base_output.clone()
        output.index_add_(0, bucket_rows, dense_output - base_selected)
        return output

    def bucket_delta_inplace() -> torch.Tensor:
        base_output = base_sparse_full()
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        base_selected = base_output.index_select(0, bucket_rows)
        base_output.index_add_(0, bucket_rows, dense_output - base_selected)
        return base_output

    def bucket_delta_no_scatter() -> torch.Tensor:
        base_output = base_sparse_full()
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        base_selected = base_output.index_select(0, bucket_rows)
        return dense_output - base_selected

    def bucket_replace() -> torch.Tensor:
        base_output = base_sparse_full()
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        output = base_output.clone()
        output.index_copy_(0, bucket_rows, dense_output)
        return output

    def bucket_override_triton() -> torch.Tensor:
        base_output = base_sparse_full()
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        return triton_bucket_override(
            base_output,
            dense_output,
            bucket_rows,
            bucket_values,
            out_features=out_features,
            block_n=triton_block_n,
        )

    def bucket_override_triton_inplace() -> torch.Tensor:
        base_output = base_sparse_full()
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        return triton_bucket_override_inplace(
            base_output,
            dense_output,
            bucket_rows,
            bucket_values,
            out_features=out_features,
            block_n=triton_block_n,
        )

    def bucket_triton_dense_gemm_scatter() -> torch.Tensor:
        base_output = base_sparse_full()
        return triton_bucket_dense_gemm_scatter_inplace(
            x,
            dense_weight,
            bucket_rows,
            bucket_values,
            base_output,
            block_m=triton_dense_block_m,
            block_n=triton_dense_block_n,
            block_k=triton_dense_block_k,
        )

    def routed_bucket() -> torch.Tensor:
        output = torch.empty((rows, out_features), device="cuda", dtype=dtype)
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        output.index_copy_(0, bucket_rows, dense_output)
        if int(base_rows.numel()) > 0:
            base_input = x.index_select(0, base_rows)
            base_output = F.linear(base_input, base_sparse)
            output.index_copy_(0, base_rows, base_output)
        return output

    def routed_bucket_triton_assemble() -> torch.Tensor:
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight).contiguous()
        if int(base_rows.numel()) > 0:
            base_input = x.index_select(0, base_rows)
            base_output = F.linear(base_input, base_sparse).contiguous()
        else:
            base_output = torch.empty((0, out_features), device="cuda", dtype=dtype)
        return triton_routed_assemble(
            dense_output,
            bucket_rows,
            base_output,
            base_rows,
            total_rows=rows,
            out_features=out_features,
            block_n=triton_block_n,
        )

    def routed_no_scatter() -> tuple[torch.Tensor, torch.Tensor | None]:
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        base_output = None
        if int(base_rows.numel()) > 0:
            base_input = x.index_select(0, base_rows)
            base_output = F.linear(base_input, base_sparse)
        return dense_output, base_output

    def routed_bucket_cslt() -> torch.Tensor:
        output = torch.empty((rows, out_features), device="cuda", dtype=dtype)
        dense_input = x.index_select(0, bucket_rows)
        dense_output = F.linear(dense_input, dense_weight)
        output.index_copy_(0, bucket_rows, dense_output)
        if int(base_rows.numel()) > 0:
            base_input = x.index_select(0, base_rows)
            base_output = direct_cslt_linear(base_input, base_sparse, alg_id=0)
            output.index_copy_(0, base_rows, base_output)
        return output

    dense_out = dense_full()
    base_out = base_sparse_full()
    expected = base_out.clone()
    expected.index_copy_(0, bucket_rows, dense_out.index_select(0, bucket_rows))
    bucket_delta_diff = float((bucket_delta() - expected).abs().max().item())
    bucket_delta_inplace_diff = float(
        (bucket_delta_inplace() - expected).abs().max().item()
    )
    bucket_replace_diff = float((bucket_replace() - expected).abs().max().item())
    bucket_override_triton_diff = float(
        (bucket_override_triton() - expected).abs().max().item()
    )
    bucket_override_triton_inplace_diff = float(
        (bucket_override_triton_inplace() - expected).abs().max().item()
    )
    bucket_triton_dense_gemm_scatter_diff = float(
        (bucket_triton_dense_gemm_scatter() - expected).abs().max().item()
    )
    routed_diff = float((routed_bucket() - expected).abs().max().item())
    routed_triton_diff = float(
        (routed_bucket_triton_assemble() - expected).abs().max().item()
    )
    routed_cslt_diff = float((routed_bucket_cslt() - expected).abs().max().item())

    dense_ms = time_call(dense_full, warmup=warmup, repeats=repeats)
    base_sparse_ms = time_call(base_sparse_full, warmup=warmup, repeats=repeats)
    base_cslt_ms = time_call(base_cslt_full, warmup=warmup, repeats=repeats)
    bucket_delta_ms = time_call(bucket_delta, warmup=warmup, repeats=repeats)
    bucket_delta_inplace_ms = time_call(
        bucket_delta_inplace, warmup=warmup, repeats=repeats
    )
    bucket_delta_no_scatter_ms = time_call(
        bucket_delta_no_scatter, warmup=warmup, repeats=repeats
    )
    bucket_replace_ms = time_call(bucket_replace, warmup=warmup, repeats=repeats)
    bucket_override_triton_ms = time_call(
        bucket_override_triton, warmup=warmup, repeats=repeats
    )
    bucket_override_triton_inplace_ms = time_call(
        bucket_override_triton_inplace, warmup=warmup, repeats=repeats
    )
    bucket_triton_dense_gemm_scatter_ms = time_call(
        bucket_triton_dense_gemm_scatter, warmup=warmup, repeats=repeats
    )
    routed_no_scatter_ms = time_call(
        routed_no_scatter, warmup=warmup, repeats=repeats
    )
    routed_ms = time_call(routed_bucket, warmup=warmup, repeats=repeats)
    routed_triton_ms = time_call(
        routed_bucket_triton_assemble, warmup=warmup, repeats=repeats
    )
    routed_cslt_ms = time_call(routed_bucket_cslt, warmup=warmup, repeats=repeats)
    dense_graph_ms, dense_graph_error = time_graph(
        dense_full, warmup=warmup, repeats=repeats
    )
    base_sparse_graph_ms, base_sparse_graph_error = time_graph(
        base_sparse_full, warmup=warmup, repeats=repeats
    )
    base_cslt_graph_ms, base_cslt_graph_error = time_graph(
        base_cslt_full, warmup=warmup, repeats=repeats
    )
    bucket_delta_graph_ms, bucket_delta_graph_error = time_graph(
        bucket_delta, warmup=warmup, repeats=repeats
    )
    bucket_delta_inplace_graph_ms, bucket_delta_inplace_graph_error = time_graph(
        bucket_delta_inplace, warmup=warmup, repeats=repeats
    )
    bucket_delta_no_scatter_graph_ms, bucket_delta_no_scatter_graph_error = (
        time_graph(bucket_delta_no_scatter, warmup=warmup, repeats=repeats)
    )
    bucket_replace_graph_ms, bucket_replace_graph_error = time_graph(
        bucket_replace, warmup=warmup, repeats=repeats
    )
    bucket_override_triton_graph_ms, bucket_override_triton_graph_error = time_graph(
        bucket_override_triton, warmup=warmup, repeats=repeats
    )
    (
        bucket_override_triton_inplace_graph_ms,
        bucket_override_triton_inplace_graph_error,
    ) = time_graph(bucket_override_triton_inplace, warmup=warmup, repeats=repeats)
    (
        bucket_triton_dense_gemm_scatter_graph_ms,
        bucket_triton_dense_gemm_scatter_graph_error,
    ) = time_graph(bucket_triton_dense_gemm_scatter, warmup=warmup, repeats=repeats)
    routed_no_scatter_graph_ms, routed_no_scatter_graph_error = time_graph(
        routed_no_scatter, warmup=warmup, repeats=repeats
    )
    routed_graph_ms, routed_graph_error = time_graph(
        routed_bucket, warmup=warmup, repeats=repeats
    )
    routed_triton_graph_ms, routed_triton_graph_error = time_graph(
        routed_bucket_triton_assemble, warmup=warmup, repeats=repeats
    )
    routed_cslt_graph_ms, routed_cslt_graph_error = time_graph(
        routed_bucket_cslt, warmup=warmup, repeats=repeats
    )

    return {
        "rows": rows,
        "out_features": out_features,
        "in_features": in_features,
        "bucket_size": bucket_size,
        "triton_block_n": triton_block_n,
        "triton_dense_block_m": triton_dense_block_m,
        "triton_dense_block_n": triton_dense_block_n,
        "triton_dense_block_k": triton_dense_block_k,
        "dtype": str(dtype).replace("torch.", ""),
        "dense_ms": dense_ms,
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "base_sparse_ms": base_sparse_ms,
        "base_sparse_graph_ms": base_sparse_graph_ms,
        "base_sparse_graph_error": base_sparse_graph_error,
        "base_cslt_ms": base_cslt_ms,
        "base_cslt_graph_ms": base_cslt_graph_ms,
        "base_cslt_graph_error": base_cslt_graph_error,
        "bucket_delta_ms": bucket_delta_ms,
        "bucket_delta_inplace_ms": bucket_delta_inplace_ms,
        "bucket_delta_no_scatter_ms": bucket_delta_no_scatter_ms,
        "bucket_replace_ms": bucket_replace_ms,
        "bucket_override_triton_ms": bucket_override_triton_ms,
        "bucket_override_triton_inplace_ms": bucket_override_triton_inplace_ms,
        "bucket_triton_dense_gemm_scatter_ms":
        bucket_triton_dense_gemm_scatter_ms,
        "routed_no_scatter_ms": routed_no_scatter_ms,
        "routed_ms": routed_ms,
        "routed_triton_ms": routed_triton_ms,
        "routed_cslt_ms": routed_cslt_ms,
        "bucket_delta_graph_ms": bucket_delta_graph_ms,
        "bucket_delta_graph_error": bucket_delta_graph_error,
        "bucket_delta_inplace_graph_ms": bucket_delta_inplace_graph_ms,
        "bucket_delta_inplace_graph_error": bucket_delta_inplace_graph_error,
        "bucket_delta_no_scatter_graph_ms": bucket_delta_no_scatter_graph_ms,
        "bucket_delta_no_scatter_graph_error": bucket_delta_no_scatter_graph_error,
        "bucket_replace_graph_ms": bucket_replace_graph_ms,
        "bucket_replace_graph_error": bucket_replace_graph_error,
        "bucket_override_triton_graph_ms": bucket_override_triton_graph_ms,
        "bucket_override_triton_graph_error": bucket_override_triton_graph_error,
        "bucket_override_triton_inplace_graph_ms":
        bucket_override_triton_inplace_graph_ms,
        "bucket_override_triton_inplace_graph_error":
        bucket_override_triton_inplace_graph_error,
        "bucket_triton_dense_gemm_scatter_graph_ms":
        bucket_triton_dense_gemm_scatter_graph_ms,
        "bucket_triton_dense_gemm_scatter_graph_error":
        bucket_triton_dense_gemm_scatter_graph_error,
        "routed_no_scatter_graph_ms": routed_no_scatter_graph_ms,
        "routed_no_scatter_graph_error": routed_no_scatter_graph_error,
        "routed_graph_ms": routed_graph_ms,
        "routed_graph_error": routed_graph_error,
        "routed_triton_graph_ms": routed_triton_graph_ms,
        "routed_triton_graph_error": routed_triton_graph_error,
        "routed_cslt_graph_ms": routed_cslt_graph_ms,
        "routed_cslt_graph_error": routed_cslt_graph_error,
        "bucket_delta_diff": bucket_delta_diff,
        "bucket_delta_inplace_diff": bucket_delta_inplace_diff,
        "bucket_replace_diff": bucket_replace_diff,
        "bucket_override_triton_diff": bucket_override_triton_diff,
        "bucket_override_triton_inplace_diff":
        bucket_override_triton_inplace_diff,
        "bucket_triton_dense_gemm_scatter_diff":
        bucket_triton_dense_gemm_scatter_diff,
        "routed_diff": routed_diff,
        "routed_triton_diff": routed_triton_diff,
        "routed_cslt_diff": routed_cslt_diff,
    }


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SR24 Mixed Bucket Microbenchmark\n\n")
        handle.write(
            "This is a CUDA microbenchmark for the current dynamic SR24 mixed "
            "path. `bucket_delta` matches the serving-safe bucket implementation: "
            "run the sparse base on all rows, run dense on selected rows, then "
            "scatter the dense-minus-base delta. `bucket_delta_inplace` is the "
            "clone-free serving variant. `bucket_delta_no_scatter` keeps "
            "the required GEMMs and base-row read but skips output assembly. "
            "`bucket_replace` is a lower-cost variant that is only equivalent "
            "when every bucket row is actually selected for dense correction. "
            "`routed` avoids sparse work on bucket rows but splits the work into "
            "separate base and dense GEMMs; `routed_no_scatter` times those GEMMs "
            "without output assembly.\n\n"
        )
        handle.write("| rows | out | in | bucket | override block N | dense GEMM block | dense graph | dense/1.2 target | base sparse graph | bucket delta graph | bucket delta inplace graph | clone removal ms | delta no-scatter graph | inplace assembly ms | bucket replace graph | bucket Triton override graph | bucket Triton override in-place graph | bucket Triton dense-GEMM scatter graph | routed no-scatter graph | routed graph | routed Triton graph | routed assembly ms | routed Triton assembly ms | routed cslt graph | delta target gap | inplace target gap | routed no-scatter target gap | routed target gap | delta/dense graph | inplace/dense graph | dense-GEMM scatter/dense graph | delta no-scatter/dense | replace/dense graph | override/dense graph | override in-place/dense graph | routed no-scatter/dense | routed/dense graph | routed Triton/dense | max diff |\n")
        handle.write("|-----:|----:|---:|-------:|-----------------:|-----------------:|------------:|-----------------:|------------------:|-------------------:|-----------------------------:|----------------:|-----------------------:|-------------------:|---------------------:|-----------------------------:|--------------------------------------:|---------------------------------------:|------------------------:|-------------:|--------------------:|------------------:|------------------------:|------------------:|-----------------:|-------------------:|-----------------------------:|-----------------:|------------------:|--------------------:|------------------------------:|-------------------------:|--------------------:|---------------------:|--------------------------------:|------------------------:|-------------------:|--------------------:|---------:|\n")
        for row in rows:
            dense = float(row["dense_ms"])
            dense_graph = float(row["dense_graph_ms"] or dense)
            target_1p2 = dense_graph / 1.2
            max_diff = max(
                float(row["bucket_delta_diff"]),
                float(row["bucket_delta_inplace_diff"]),
                float(row["bucket_replace_diff"]),
                float(row["bucket_override_triton_diff"]),
                float(row["bucket_override_triton_inplace_diff"]),
                float(row["bucket_triton_dense_gemm_scatter_diff"]),
                float(row["routed_diff"]),
                float(row["routed_triton_diff"]),
                float(row["routed_cslt_diff"]),
            )
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                f"{row['bucket_size']} | "
                f"{row['triton_block_n']} | "
                f"{row['triton_dense_block_m']}x{row['triton_dense_block_n']}x{row['triton_dense_block_k']} | "
                f"{fmt_ms(row['dense_graph_ms']) or row['dense_graph_error']} | "
                f"{target_1p2:.4f} | "
                f"{fmt_ms(row['base_sparse_graph_ms']) or row['base_sparse_graph_error']} | "
                f"{fmt_ms(row['bucket_delta_graph_ms']) or row['bucket_delta_graph_error']} | "
                f"{fmt_ms(row['bucket_delta_inplace_graph_ms']) or row['bucket_delta_inplace_graph_error']} | "
                f"{fmt_delta_ms(row['bucket_delta_graph_ms'], row['bucket_delta_inplace_graph_ms'])} | "
                f"{fmt_ms(row['bucket_delta_no_scatter_graph_ms']) or row['bucket_delta_no_scatter_graph_error']} | "
                f"{fmt_delta_ms(row['bucket_delta_inplace_graph_ms'], row['bucket_delta_no_scatter_graph_ms'])} | "
                f"{fmt_ms(row['bucket_replace_graph_ms']) or row['bucket_replace_graph_error']} | "
                f"{fmt_ms(row['bucket_override_triton_graph_ms']) or row['bucket_override_triton_graph_error']} | "
                f"{fmt_ms(row['bucket_override_triton_inplace_graph_ms']) or row['bucket_override_triton_inplace_graph_error']} | "
                f"{fmt_ms(row['bucket_triton_dense_gemm_scatter_graph_ms']) or row['bucket_triton_dense_gemm_scatter_graph_error']} | "
                f"{fmt_ms(row['routed_no_scatter_graph_ms']) or row['routed_no_scatter_graph_error']} | "
                f"{fmt_ms(row['routed_graph_ms']) or row['routed_graph_error']} | "
                f"{fmt_ms(row['routed_triton_graph_ms']) or row['routed_triton_graph_error']} | "
                f"{fmt_delta_ms(row['routed_graph_ms'], row['routed_no_scatter_graph_ms'])} | "
                f"{fmt_delta_ms(row['routed_triton_graph_ms'], row['routed_no_scatter_graph_ms'])} | "
                f"{fmt_ms(row['routed_cslt_graph_ms']) or row['routed_cslt_graph_error']} | "
                f"{fmt_target_gap(row['bucket_delta_graph_ms'], target_1p2)} | "
                f"{fmt_target_gap(row['bucket_delta_inplace_graph_ms'], target_1p2)} | "
                f"{fmt_target_gap(row['routed_no_scatter_graph_ms'], target_1p2)} | "
                f"{fmt_target_gap(row['routed_graph_ms'], target_1p2)} | "
                f"{fmt_ratio(row['bucket_delta_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['bucket_delta_inplace_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['bucket_triton_dense_gemm_scatter_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['bucket_delta_no_scatter_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['bucket_replace_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['bucket_override_triton_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['bucket_override_triton_inplace_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['routed_no_scatter_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['routed_graph_ms'], dense_graph)} | "
                f"{fmt_ratio(row['routed_triton_graph_ms'], dense_graph)} | "
                f"{max_diff:.4g} |\n"
            )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SR24 mixed bucket probe")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    output_root = args.output_root or (
        Path("examples/evaluate/eval-guidellm/results.bak")
        / f"speclink_sr24_mixed_bucket_probe_{timestamp()}"
    )
    results: list[dict[str, Any]] = []
    for shape in args.shape:
        rows, out_features, in_features = shape
        for bucket_size in args.bucket_sizes:
            for block_m in args.triton_dense_block_m:
                for block_n in args.triton_dense_block_n:
                    for block_k in args.triton_dense_block_k:
                        results.append(
                            run_case(
                                rows=rows,
                                out_features=out_features,
                                in_features=in_features,
                                bucket_size=bucket_size,
                                dtype=dtype,
                                warmup=args.warmup,
                                repeats=args.repeats,
                                triton_block_n=args.triton_block_n,
                                triton_dense_block_m=block_m,
                                triton_dense_block_n=block_n,
                                triton_dense_block_k=block_k,
                            )
                        )
    write_report(output_root, results)
    print(output_root.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=[],
        help="ROWS,OUT,IN. Defaults to Llama gate_up rows=512.",
    )
    parser.add_argument("--bucket-sizes", type=parse_int_list, default=[64, 128, 256])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--triton-block-n", type=int, default=1024)
    parser.add_argument(
        "--triton-dense-block-m",
        type=parse_int_list,
        default=[16],
        help="Comma-separated BLOCK_M values for the Triton dense-GEMM scatter kernel.",
    )
    parser.add_argument(
        "--triton-dense-block-n",
        type=parse_int_list,
        default=[32],
        help="Comma-separated BLOCK_N values for the Triton dense-GEMM scatter kernel.",
    )
    parser.add_argument(
        "--triton-dense-block-k",
        type=parse_int_list,
        default=[128],
        help="Comma-separated BLOCK_K values for the Triton dense-GEMM scatter kernel.",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(512, 28672, 4096)]
    run(args)


if __name__ == "__main__":
    main()
