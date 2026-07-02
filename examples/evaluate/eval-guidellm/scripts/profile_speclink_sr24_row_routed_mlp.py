#!/usr/bin/env python3
"""Microbenchmark SR24 row-routed Llama MLP variants.

This checks whether routing at the whole-MLP level is more promising than
routing inside the fused gate_up Linear. The current Linear-level route has to
assemble a [rows, 2 * intermediate] tensor. The MLP-level route computes dense
gate_up rows and sparse gate_up rows separately, concatenates them through the
activation and down projection, and only assembles the final [rows, hidden]
output.

The `sparse_gateup_dense_down` variants probe a narrower lossy design: keep the
large gate/up projection sparse on unimportant rows, but use the normal dense
down projection for all rows. This tests whether cuSPARSELt down projection
latency dominates at the row counts used by serving.

The `overlap_streams` variant keeps the exact-down route semantics but launches
the dense important-row branch and sparse base-row branch on separate CUDA
streams before the final assemble. It is a microbenchmark for the PPoPP-style
question of whether independent dense/sparse work can be overlapped instead of
serialized.
"""

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


def silu_and_mul(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    return F.silu(gate) * up


@triton.jit
def _assemble_rows_kernel(
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
    tl.store(
        output + dst_row * out_features + cols,
        dense_vals + base_vals,
        mask=col_mask,
    )


def assemble_rows(
    dense_output: torch.Tensor,
    dense_rows: torch.Tensor,
    base_output: torch.Tensor,
    base_rows: torch.Tensor,
    *,
    total_rows: int,
    out_features: int,
    block_n: int,
) -> torch.Tensor:
    output = torch.empty(
        (total_rows, out_features),
        dtype=dense_output.dtype,
        device=dense_output.device,
    )
    grid = (total_rows, triton.cdiv(out_features, block_n))
    _assemble_rows_kernel[grid](
        dense_output.contiguous(),
        dense_rows.contiguous(),
        base_output.contiguous(),
        base_rows.contiguous(),
        output,
        int(dense_output.shape[0]),
        int(total_rows),
        int(out_features),
        int(block_n),
        num_warps=8,
    )
    return output


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


def fmt_ratio(value: Any, base: float | None) -> str:
    if value is None or base is None or base <= 0:
        return ""
    return f"{float(value) / base:.2f}x"


def fmt_target_gap(value: Any, target: float | None) -> str:
    if value is None or target is None:
        return ""
    return f"{float(value) - target:.4f}"


def run_case(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    bucket_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    block_n: int,
) -> dict[str, Any]:
    bucket_size = min(rows, bucket_size)
    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    gate_up_weight = torch.randn(
        intermediate_size * 2,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    down_weight = torch.randn(
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=dtype,
    )
    gate_up_base = to_sparse_semi_structured(make_base_24(gate_up_weight))
    down_base = to_sparse_semi_structured(make_base_24(down_weight))
    dense_rows = torch.arange(bucket_size, device="cuda", dtype=torch.long)
    base_rows = torch.arange(bucket_size, rows, device="cuda", dtype=torch.long)
    dense_stream = torch.cuda.Stream()
    base_stream = torch.cuda.Stream()

    def select_input(row_ids: torch.Tensor) -> torch.Tensor:
        if int(row_ids.numel()) == 0:
            return torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return x.index_select(0, row_ids)

    def dense_gate_up_linear(inp: torch.Tensor) -> torch.Tensor:
        if int(inp.shape[0]) == 0:
            return torch.empty(
                0, intermediate_size * 2, device="cuda", dtype=dtype
            )
        return F.linear(inp, gate_up_weight)

    def base_gate_up_linear(inp: torch.Tensor) -> torch.Tensor:
        if int(inp.shape[0]) == 0:
            return torch.empty(
                0, intermediate_size * 2, device="cuda", dtype=dtype
            )
        return F.linear(inp, gate_up_base)

    def dense_down_linear(act: torch.Tensor) -> torch.Tensor:
        if int(act.shape[0]) == 0:
            return torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return F.linear(act, down_weight)

    def base_down_linear(act: torch.Tensor) -> torch.Tensor:
        if int(act.shape[0]) == 0:
            return torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return F.linear(act, down_base)

    def dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_weight)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_weight)

    def base_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_base)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_base)

    def linear_level_replace_then_down() -> torch.Tensor:
        base_gate_up = F.linear(x, gate_up_base)
        dense_gate_up = dense_gate_up_linear(select_input(dense_rows))
        gate_up = base_gate_up.clone()
        gate_up.index_copy_(0, dense_rows, dense_gate_up)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_base)

    def row_routed_mlp_index_copy() -> torch.Tensor:
        dense_gate_up = dense_gate_up_linear(select_input(dense_rows))
        base_gate_up = base_gate_up_linear(select_input(base_rows))
        routed_gate_up = torch.cat([dense_gate_up, base_gate_up], dim=0)
        routed_act = silu_and_mul(routed_gate_up, intermediate_size)
        routed_down = F.linear(routed_act, down_base)
        output = torch.empty((rows, hidden_size), device="cuda", dtype=dtype)
        output.index_copy_(0, dense_rows, routed_down[:bucket_size])
        output.index_copy_(0, base_rows, routed_down[bucket_size:])
        return output

    def row_routed_mlp_triton_assemble() -> torch.Tensor:
        dense_gate_up = dense_gate_up_linear(select_input(dense_rows))
        base_gate_up = base_gate_up_linear(select_input(base_rows))
        routed_gate_up = torch.cat([dense_gate_up, base_gate_up], dim=0)
        routed_act = silu_and_mul(routed_gate_up, intermediate_size)
        routed_down = F.linear(routed_act, down_base)
        dense_out = routed_down[:bucket_size]
        base_out = routed_down[bucket_size:]
        return assemble_rows(
            dense_out,
            dense_rows,
            base_out,
            base_rows,
            total_rows=rows,
            out_features=hidden_size,
            block_n=block_n,
        )

    def row_routed_exact_down_index_copy() -> torch.Tensor:
        dense_input = select_input(dense_rows)
        base_input = select_input(base_rows)
        dense_gate_up = dense_gate_up_linear(dense_input)
        base_gate_up = base_gate_up_linear(base_input)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        base_act = silu_and_mul(base_gate_up, intermediate_size)
        dense_down = dense_down_linear(dense_act)
        base_down = base_down_linear(base_act)
        output = torch.empty((rows, hidden_size), device="cuda", dtype=dtype)
        output.index_copy_(0, dense_rows, dense_down)
        output.index_copy_(0, base_rows, base_down)
        return output

    def row_routed_exact_down_triton_assemble() -> torch.Tensor:
        dense_input = select_input(dense_rows)
        base_input = select_input(base_rows)
        dense_gate_up = dense_gate_up_linear(dense_input)
        base_gate_up = base_gate_up_linear(base_input)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        base_act = silu_and_mul(base_gate_up, intermediate_size)
        dense_down = dense_down_linear(dense_act)
        base_down = base_down_linear(base_act)
        return assemble_rows(
            dense_down,
            dense_rows,
            base_down,
            base_rows,
            total_rows=rows,
            out_features=hidden_size,
            block_n=block_n,
        )

    def row_routed_exact_down_overlap_streams_index_copy() -> torch.Tensor:
        with torch.cuda.stream(dense_stream):
            dense_input = select_input(dense_rows)
            dense_gate_up = dense_gate_up_linear(dense_input)
            dense_act = silu_and_mul(dense_gate_up, intermediate_size)
            dense_down = dense_down_linear(dense_act)
        with torch.cuda.stream(base_stream):
            base_input = select_input(base_rows)
            base_gate_up = base_gate_up_linear(base_input)
            base_act = silu_and_mul(base_gate_up, intermediate_size)
            base_down = base_down_linear(base_act)
        torch.cuda.current_stream().wait_stream(dense_stream)
        torch.cuda.current_stream().wait_stream(base_stream)
        output = torch.empty((rows, hidden_size), device="cuda", dtype=dtype)
        output.index_copy_(0, dense_rows, dense_down)
        output.index_copy_(0, base_rows, base_down)
        return output

    def row_routed_exact_down_overlap_streams_triton_assemble() -> torch.Tensor:
        with torch.cuda.stream(dense_stream):
            dense_input = select_input(dense_rows)
            dense_gate_up = dense_gate_up_linear(dense_input)
            dense_act = silu_and_mul(dense_gate_up, intermediate_size)
            dense_down = dense_down_linear(dense_act)
        with torch.cuda.stream(base_stream):
            base_input = select_input(base_rows)
            base_gate_up = base_gate_up_linear(base_input)
            base_act = silu_and_mul(base_gate_up, intermediate_size)
            base_down = base_down_linear(base_act)
        torch.cuda.current_stream().wait_stream(dense_stream)
        torch.cuda.current_stream().wait_stream(base_stream)
        return assemble_rows(
            dense_down,
            dense_rows,
            base_down,
            base_rows,
            total_rows=rows,
            out_features=hidden_size,
            block_n=block_n,
        )

    def row_routed_sparse_gateup_dense_down_index_copy() -> torch.Tensor:
        dense_input = select_input(dense_rows)
        base_input = select_input(base_rows)
        dense_gate_up = dense_gate_up_linear(dense_input)
        base_gate_up = base_gate_up_linear(base_input)
        routed_act = torch.cat(
            [
                silu_and_mul(dense_gate_up, intermediate_size),
                silu_and_mul(base_gate_up, intermediate_size),
            ],
            dim=0,
        )
        routed_down = dense_down_linear(routed_act)
        output = torch.empty((rows, hidden_size), device="cuda", dtype=dtype)
        output.index_copy_(0, dense_rows, routed_down[:bucket_size])
        output.index_copy_(0, base_rows, routed_down[bucket_size:])
        return output

    def row_routed_sparse_gateup_dense_down_triton_assemble() -> torch.Tensor:
        dense_input = select_input(dense_rows)
        base_input = select_input(base_rows)
        dense_gate_up = dense_gate_up_linear(dense_input)
        base_gate_up = base_gate_up_linear(base_input)
        routed_act = torch.cat(
            [
                silu_and_mul(dense_gate_up, intermediate_size),
                silu_and_mul(base_gate_up, intermediate_size),
            ],
            dim=0,
        )
        routed_down = dense_down_linear(routed_act)
        return assemble_rows(
            routed_down[:bucket_size],
            dense_rows,
            routed_down[bucket_size:],
            base_rows,
            total_rows=rows,
            out_features=hidden_size,
            block_n=block_n,
        )

    def row_routed_sparse_gateup_dense_down_contiguous_cat() -> torch.Tensor:
        dense_gate_up = dense_gate_up_linear(x[:bucket_size])
        base_gate_up = base_gate_up_linear(x[bucket_size:])
        routed_act = torch.cat(
            [
                silu_and_mul(dense_gate_up, intermediate_size),
                silu_and_mul(base_gate_up, intermediate_size),
            ],
            dim=0,
        )
        return dense_down_linear(routed_act)

    def row_routed_exact_down_contiguous_cat() -> torch.Tensor:
        """Fastpath for prefix dense rows and suffix base rows.

        This mirrors the live `SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH` MLP
        path: avoid `index_select` and final `index_copy_` when the route is
        already a contiguous prefix/suffix split.
        """
        dense_gate_up = dense_gate_up_linear(x[:bucket_size])
        base_gate_up = base_gate_up_linear(x[bucket_size:])
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        base_act = silu_and_mul(base_gate_up, intermediate_size)
        dense_down = dense_down_linear(dense_act)
        base_down = base_down_linear(base_act)
        return torch.cat([dense_down, base_down], dim=0)

    def row_routed_exact_down_no_final_assemble() -> tuple[torch.Tensor, torch.Tensor]:
        """Lower-bound exact-down route when downstream can consume routed rows.

        Live serving still needs the final hidden states in original row order,
        so this is not directly usable. It separates route compute cost from
        final assembly cost.
        """
        dense_input = select_input(dense_rows)
        base_input = select_input(base_rows)
        dense_gate_up = dense_gate_up_linear(dense_input)
        base_gate_up = base_gate_up_linear(base_input)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        base_act = silu_and_mul(base_gate_up, intermediate_size)
        dense_down = dense_down_linear(dense_act)
        base_down = base_down_linear(base_act)
        return dense_down, base_down

    def row_routed_exact_down_reuse_base_output() -> torch.Tensor:
        """Graph-friendlier exact-down route used by the live reuse-base probe.

        It avoids building the base-row complement. The tradeoff is extra sparse
        base MLP work on rows that will later be overwritten by exact dense rows.
        """
        dense_input = select_input(dense_rows)
        dense_gate_up = dense_gate_up_linear(dense_input)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        dense_down = dense_down_linear(dense_act)
        base_gate_up = base_gate_up_linear(x)
        base_act = silu_and_mul(base_gate_up, intermediate_size)
        base_down = base_down_linear(base_act)
        base_down.index_copy_(0, dense_rows, dense_down)
        return base_down

    expected = linear_level_replace_then_down()
    row_copy = row_routed_mlp_index_copy()
    row_triton = row_routed_mlp_triton_assemble()
    exact_down_copy = row_routed_exact_down_index_copy()
    exact_down_triton = row_routed_exact_down_triton_assemble()
    exact_down_overlap_copy = row_routed_exact_down_overlap_streams_index_copy()
    exact_down_overlap_triton = row_routed_exact_down_overlap_streams_triton_assemble()
    exact_down_contiguous = row_routed_exact_down_contiguous_cat()
    exact_down_reuse_base = row_routed_exact_down_reuse_base_output()
    sparse_gateup_dense_down_copy = (
        row_routed_sparse_gateup_dense_down_index_copy()
    )
    sparse_gateup_dense_down_triton = (
        row_routed_sparse_gateup_dense_down_triton_assemble()
    )
    sparse_gateup_dense_down_contiguous = (
        row_routed_sparse_gateup_dense_down_contiguous_cat()
    )

    dense_input_pre = select_input(dense_rows)
    base_input_pre = select_input(base_rows)
    dense_gate_up_pre = dense_gate_up_linear(dense_input_pre)
    base_gate_up_pre = base_gate_up_linear(base_input_pre)
    dense_act_pre = silu_and_mul(dense_gate_up_pre, intermediate_size)
    base_act_pre = silu_and_mul(base_gate_up_pre, intermediate_size)
    mixed_act_pre = torch.cat([dense_act_pre, base_act_pre], dim=0)
    dense_down_pre = dense_down_linear(dense_act_pre)
    base_down_pre = base_down_linear(base_act_pre)

    def index_copy_assemble_only() -> torch.Tensor:
        output = torch.empty((rows, hidden_size), device="cuda", dtype=dtype)
        output.index_copy_(0, dense_rows, dense_down_pre)
        output.index_copy_(0, base_rows, base_down_pre)
        return output

    def triton_assemble_only() -> torch.Tensor:
        return assemble_rows(
            dense_down_pre,
            dense_rows,
            base_down_pre,
            base_rows,
            total_rows=rows,
            out_features=hidden_size,
            block_n=block_n,
        )
    torch.cuda.synchronize()

    dense_graph_ms, dense_graph_error = time_graph(
        dense_mlp, warmup=warmup, repeats=repeats
    )
    base_graph_ms, base_graph_error = time_graph(
        base_mlp, warmup=warmup, repeats=repeats
    )
    linear_graph_ms, linear_graph_error = time_graph(
        linear_level_replace_then_down, warmup=warmup, repeats=repeats
    )
    row_copy_graph_ms, row_copy_graph_error = time_graph(
        row_routed_mlp_index_copy, warmup=warmup, repeats=repeats
    )
    row_triton_graph_ms, row_triton_graph_error = time_graph(
        row_routed_mlp_triton_assemble, warmup=warmup, repeats=repeats
    )
    exact_down_copy_graph_ms, exact_down_copy_graph_error = time_graph(
        row_routed_exact_down_index_copy, warmup=warmup, repeats=repeats
    )
    exact_down_triton_graph_ms, exact_down_triton_graph_error = time_graph(
        row_routed_exact_down_triton_assemble, warmup=warmup, repeats=repeats
    )
    exact_down_overlap_copy_graph_ms, exact_down_overlap_copy_graph_error = time_graph(
        row_routed_exact_down_overlap_streams_index_copy,
        warmup=warmup,
        repeats=repeats,
    )
    (
        exact_down_overlap_triton_graph_ms,
        exact_down_overlap_triton_graph_error,
    ) = time_graph(
        row_routed_exact_down_overlap_streams_triton_assemble,
        warmup=warmup,
        repeats=repeats,
    )
    (
        sparse_gateup_dense_down_copy_graph_ms,
        sparse_gateup_dense_down_copy_graph_error,
    ) = time_graph(
        row_routed_sparse_gateup_dense_down_index_copy,
        warmup=warmup,
        repeats=repeats,
    )
    (
        sparse_gateup_dense_down_triton_graph_ms,
        sparse_gateup_dense_down_triton_graph_error,
    ) = time_graph(
        row_routed_sparse_gateup_dense_down_triton_assemble,
        warmup=warmup,
        repeats=repeats,
    )
    (
        sparse_gateup_dense_down_contiguous_graph_ms,
        sparse_gateup_dense_down_contiguous_graph_error,
    ) = time_graph(
        row_routed_sparse_gateup_dense_down_contiguous_cat,
        warmup=warmup,
        repeats=repeats,
    )
    exact_down_contiguous_graph_ms, exact_down_contiguous_graph_error = time_graph(
        row_routed_exact_down_contiguous_cat, warmup=warmup, repeats=repeats
    )
    exact_down_no_assemble_graph_ms, exact_down_no_assemble_graph_error = time_graph(
        row_routed_exact_down_no_final_assemble, warmup=warmup, repeats=repeats
    )
    exact_down_reuse_base_graph_ms, exact_down_reuse_base_graph_error = time_graph(
        row_routed_exact_down_reuse_base_output, warmup=warmup, repeats=repeats
    )

    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "bucket_size": bucket_size,
        "dtype": str(dtype).replace("torch.", ""),
        "dense_ms": time_call(dense_mlp, warmup=warmup, repeats=repeats),
        "base_ms": time_call(base_mlp, warmup=warmup, repeats=repeats),
        "linear_level_ms": time_call(
            linear_level_replace_then_down, warmup=warmup, repeats=repeats
        ),
        "row_routed_index_copy_ms": time_call(
            row_routed_mlp_index_copy, warmup=warmup, repeats=repeats
        ),
        "row_routed_triton_assemble_ms": time_call(
            row_routed_mlp_triton_assemble, warmup=warmup, repeats=repeats
        ),
        "row_routed_exact_down_index_copy_ms": time_call(
            row_routed_exact_down_index_copy, warmup=warmup, repeats=repeats
        ),
        "row_routed_exact_down_triton_assemble_ms": time_call(
            row_routed_exact_down_triton_assemble, warmup=warmup, repeats=repeats
        ),
        "row_routed_exact_down_overlap_streams_index_copy_ms": time_call(
            row_routed_exact_down_overlap_streams_index_copy,
            warmup=warmup,
            repeats=repeats,
        ),
        "row_routed_exact_down_overlap_streams_triton_assemble_ms": time_call(
            row_routed_exact_down_overlap_streams_triton_assemble,
            warmup=warmup,
            repeats=repeats,
        ),
        "row_routed_sparse_gateup_dense_down_index_copy_ms": time_call(
            row_routed_sparse_gateup_dense_down_index_copy,
            warmup=warmup,
            repeats=repeats,
        ),
        "row_routed_sparse_gateup_dense_down_triton_assemble_ms": time_call(
            row_routed_sparse_gateup_dense_down_triton_assemble,
            warmup=warmup,
            repeats=repeats,
        ),
        "row_routed_sparse_gateup_dense_down_contiguous_cat_ms": time_call(
            row_routed_sparse_gateup_dense_down_contiguous_cat,
            warmup=warmup,
            repeats=repeats,
        ),
        "row_routed_exact_down_contiguous_cat_ms": time_call(
            row_routed_exact_down_contiguous_cat, warmup=warmup, repeats=repeats
        ),
        "row_routed_exact_down_no_final_assemble_ms": time_call(
            row_routed_exact_down_no_final_assemble, warmup=warmup, repeats=repeats
        ),
        "row_routed_exact_down_reuse_base_output_ms": time_call(
            row_routed_exact_down_reuse_base_output, warmup=warmup, repeats=repeats
        ),
        "exact_down_dense_gather_ms": time_call(
            lambda: select_input(dense_rows), warmup=warmup, repeats=repeats
        ),
        "exact_down_base_gather_ms": time_call(
            lambda: select_input(base_rows), warmup=warmup, repeats=repeats
        ),
        "exact_down_dense_gate_up_ms": time_call(
            lambda: dense_gate_up_linear(dense_input_pre),
            warmup=warmup,
            repeats=repeats,
        ),
        "exact_down_base_gate_up_ms": time_call(
            lambda: base_gate_up_linear(base_input_pre),
            warmup=warmup,
            repeats=repeats,
        ),
        "exact_down_dense_act_ms": time_call(
            lambda: silu_and_mul(dense_gate_up_pre, intermediate_size),
            warmup=warmup,
            repeats=repeats,
        ),
        "exact_down_base_act_ms": time_call(
            lambda: silu_and_mul(base_gate_up_pre, intermediate_size),
            warmup=warmup,
            repeats=repeats,
        ),
        "exact_down_dense_down_ms": time_call(
            lambda: dense_down_linear(dense_act_pre),
            warmup=warmup,
            repeats=repeats,
        ),
        "exact_down_base_down_ms": time_call(
            lambda: base_down_linear(base_act_pre),
            warmup=warmup,
            repeats=repeats,
        ),
        "sparse_gateup_dense_down_full_down_ms": time_call(
            lambda: dense_down_linear(mixed_act_pre),
            warmup=warmup,
            repeats=repeats,
        ),
        "exact_down_index_copy_assemble_ms": time_call(
            index_copy_assemble_only, warmup=warmup, repeats=repeats
        ),
        "exact_down_triton_assemble_only_ms": time_call(
            triton_assemble_only, warmup=warmup, repeats=repeats
        ),
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "base_graph_ms": base_graph_ms,
        "base_graph_error": base_graph_error,
        "linear_level_graph_ms": linear_graph_ms,
        "linear_level_graph_error": linear_graph_error,
        "row_routed_index_copy_graph_ms": row_copy_graph_ms,
        "row_routed_index_copy_graph_error": row_copy_graph_error,
        "row_routed_triton_assemble_graph_ms": row_triton_graph_ms,
        "row_routed_triton_assemble_graph_error": row_triton_graph_error,
        "row_routed_exact_down_index_copy_graph_ms": exact_down_copy_graph_ms,
        "row_routed_exact_down_index_copy_graph_error": exact_down_copy_graph_error,
        "row_routed_exact_down_triton_assemble_graph_ms": exact_down_triton_graph_ms,
        "row_routed_exact_down_triton_assemble_graph_error":
        exact_down_triton_graph_error,
        "row_routed_exact_down_overlap_streams_index_copy_graph_ms":
        exact_down_overlap_copy_graph_ms,
        "row_routed_exact_down_overlap_streams_index_copy_graph_error":
        exact_down_overlap_copy_graph_error,
        "row_routed_exact_down_overlap_streams_triton_assemble_graph_ms":
        exact_down_overlap_triton_graph_ms,
        "row_routed_exact_down_overlap_streams_triton_assemble_graph_error":
        exact_down_overlap_triton_graph_error,
        "row_routed_sparse_gateup_dense_down_index_copy_graph_ms":
        sparse_gateup_dense_down_copy_graph_ms,
        "row_routed_sparse_gateup_dense_down_index_copy_graph_error":
        sparse_gateup_dense_down_copy_graph_error,
        "row_routed_sparse_gateup_dense_down_triton_assemble_graph_ms":
        sparse_gateup_dense_down_triton_graph_ms,
        "row_routed_sparse_gateup_dense_down_triton_assemble_graph_error":
        sparse_gateup_dense_down_triton_graph_error,
        "row_routed_sparse_gateup_dense_down_contiguous_cat_graph_ms":
        sparse_gateup_dense_down_contiguous_graph_ms,
        "row_routed_sparse_gateup_dense_down_contiguous_cat_graph_error":
        sparse_gateup_dense_down_contiguous_graph_error,
        "row_routed_exact_down_contiguous_cat_graph_ms":
        exact_down_contiguous_graph_ms,
        "row_routed_exact_down_contiguous_cat_graph_error":
        exact_down_contiguous_graph_error,
        "row_routed_exact_down_no_final_assemble_graph_ms":
        exact_down_no_assemble_graph_ms,
        "row_routed_exact_down_no_final_assemble_graph_error":
        exact_down_no_assemble_graph_error,
        "row_routed_exact_down_reuse_base_output_graph_ms":
        exact_down_reuse_base_graph_ms,
        "row_routed_exact_down_reuse_base_output_graph_error":
        exact_down_reuse_base_graph_error,
        "row_routed_index_copy_max_diff": float((row_copy - expected).abs().max().item()),
        "row_routed_triton_assemble_max_diff": float(
            (row_triton - expected).abs().max().item()
        ),
        "row_routed_exact_down_assemble_max_diff": float(
            (exact_down_copy - exact_down_triton).abs().max().item()
        ),
        "row_routed_exact_down_overlap_streams_index_copy_max_diff": float(
            (exact_down_copy - exact_down_overlap_copy).abs().max().item()
        ),
        "row_routed_exact_down_overlap_streams_triton_assemble_max_diff": float(
            (exact_down_copy - exact_down_overlap_triton).abs().max().item()
        ),
        "row_routed_sparse_gateup_dense_down_assemble_max_diff": float(
            (
                sparse_gateup_dense_down_copy
                - sparse_gateup_dense_down_triton
            ).abs().max().item()
        ),
        "row_routed_sparse_gateup_dense_down_contiguous_cat_max_diff": float(
            (
                sparse_gateup_dense_down_copy
                - sparse_gateup_dense_down_contiguous
            ).abs().max().item()
        ),
        "row_routed_exact_down_contiguous_cat_max_diff": float(
            (exact_down_copy - exact_down_contiguous).abs().max().item()
        ),
        "row_routed_exact_down_reuse_base_max_diff": float(
            (exact_down_copy - exact_down_reuse_base).abs().max().item()
        ),
    }


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SR24 Row-Routed MLP Microbenchmark\n\n")
        handle.write(
            "`linear_level` matches a gate_up row-replace path followed by a "
            "base down projection. `row_routed_*` avoids assembling the large "
            "gate_up tensor in original row order and assembles only the final "
            "hidden-size MLP output. `exact-down` additionally protects the "
            "same dense rows through the down projection; that is the more "
            "quality-conservative route used to judge whether row routing can "
            "still be fast after correcting the full MLP rows.\n\n"
        )
        handle.write("| rows | hidden | inter | bucket | dense graph | dense/1.2 target | base graph | linear-level graph | row-routed Triton graph | exact-down Triton graph | exact-down overlap Triton graph | sparse-gateup dense-down Triton graph | sparse-gateup dense-down contiguous graph | exact-down contiguous graph | exact-down no-assemble graph | exact-down reuse-base graph | overlap target gap | sparse-gateup dense-down target gap | exact-down target gap | contiguous target gap | reuse-base target gap | row-Triton / dense | overlap / dense | sparse-gateup dense-down / dense | sparse-gateup dense-down contiguous / dense | exact-down / dense | contiguous / dense | reuse-base / dense | max diff |\n")
        handle.write("|-----:|-------:|------:|-------:|------------:|-----------------:|-----------:|-------------------:|------------------------:|------------------------:|--------------------------------:|------------------------------------:|---------------------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|-------------------:|---------------------------------:|----------------------:|----------------------:|----------------------:|------------------:|---------------:|--------------------------------:|-------------------------------------------:|-------------------:|-------------------:|--------------------:|---------:|\n")
        for row in rows:
            dense = row.get("dense_graph_ms") or row.get("dense_ms")
            target_1p2 = float(dense) / 1.2 if dense is not None else None
            max_diff = max(
                float(row["row_routed_index_copy_max_diff"]),
                float(row["row_routed_triton_assemble_max_diff"]),
                float(row["row_routed_exact_down_assemble_max_diff"]),
                float(row["row_routed_exact_down_overlap_streams_index_copy_max_diff"]),
                float(row["row_routed_exact_down_overlap_streams_triton_assemble_max_diff"]),
                float(row["row_routed_sparse_gateup_dense_down_assemble_max_diff"]),
                float(
                    row[
                        "row_routed_sparse_gateup_dense_down_contiguous_cat_max_diff"
                    ]
                ),
                float(row["row_routed_exact_down_contiguous_cat_max_diff"]),
                float(row["row_routed_exact_down_reuse_base_max_diff"]),
            )
            handle.write(
                f"| {row['rows']} | {row['hidden_size']} | "
                f"{row['intermediate_size']} | {row['bucket_size']} | "
                f"{fmt_ms(row.get('dense_graph_ms')) or row.get('dense_graph_error')} | "
                f"{target_1p2:.4f} | "
                f"{fmt_ms(row.get('base_graph_ms')) or row.get('base_graph_error')} | "
                f"{fmt_ms(row.get('linear_level_graph_ms')) or row.get('linear_level_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_triton_assemble_graph_ms')) or row.get('row_routed_triton_assemble_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_exact_down_triton_assemble_graph_ms')) or row.get('row_routed_exact_down_triton_assemble_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_exact_down_overlap_streams_triton_assemble_graph_ms')) or row.get('row_routed_exact_down_overlap_streams_triton_assemble_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_sparse_gateup_dense_down_triton_assemble_graph_ms')) or row.get('row_routed_sparse_gateup_dense_down_triton_assemble_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_sparse_gateup_dense_down_contiguous_cat_graph_ms')) or row.get('row_routed_sparse_gateup_dense_down_contiguous_cat_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_exact_down_contiguous_cat_graph_ms')) or row.get('row_routed_exact_down_contiguous_cat_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_exact_down_no_final_assemble_graph_ms')) or row.get('row_routed_exact_down_no_final_assemble_graph_error')} | "
                f"{fmt_ms(row.get('row_routed_exact_down_reuse_base_output_graph_ms')) or row.get('row_routed_exact_down_reuse_base_output_graph_error')} | "
                f"{fmt_target_gap(row.get('row_routed_exact_down_overlap_streams_triton_assemble_graph_ms'), target_1p2)} | "
                f"{fmt_target_gap(row.get('row_routed_sparse_gateup_dense_down_triton_assemble_graph_ms'), target_1p2)} | "
                f"{fmt_target_gap(row.get('row_routed_exact_down_triton_assemble_graph_ms'), target_1p2)} | "
                f"{fmt_target_gap(row.get('row_routed_exact_down_contiguous_cat_graph_ms'), target_1p2)} | "
                f"{fmt_target_gap(row.get('row_routed_exact_down_reuse_base_output_graph_ms'), target_1p2)} | "
                f"{fmt_ratio(row.get('row_routed_triton_assemble_graph_ms'), dense)} | "
                f"{fmt_ratio(row.get('row_routed_exact_down_overlap_streams_triton_assemble_graph_ms'), dense)} | "
                f"{fmt_ratio(row.get('row_routed_sparse_gateup_dense_down_triton_assemble_graph_ms'), dense)} | "
                f"{fmt_ratio(row.get('row_routed_sparse_gateup_dense_down_contiguous_cat_graph_ms'), dense)} | "
                f"{fmt_ratio(row.get('row_routed_exact_down_triton_assemble_graph_ms'), dense)} | "
                f"{fmt_ratio(row.get('row_routed_exact_down_contiguous_cat_graph_ms'), dense)} | "
                f"{fmt_ratio(row.get('row_routed_exact_down_reuse_base_output_graph_ms'), dense)} | "
                f"{max_diff:.4g} |\n"
            )

        handle.write("\n## Exact-Down Subcomponent Timing\n\n")
        handle.write(
            "The rows below use normal CUDA event timing, not graph replay. "
            "They isolate the quality-conservative exact-down path into "
            "gather, gate/up, activation, down, and final assembly costs.\n\n"
        )
        handle.write("| bucket | dense gather | base gather | dense gate/up | base gate/up | dense act | base act | dense down | base down | full dense down | index assemble | Triton assemble |\n")
        handle.write("|-------:|-------------:|------------:|--------------:|-------------:|----------:|---------:|-----------:|----------:|----------------:|---------------:|----------------:|\n")
        for row in rows:
            handle.write(
                f"| {row['bucket_size']} | "
                f"{fmt_ms(row.get('exact_down_dense_gather_ms'))} | "
                f"{fmt_ms(row.get('exact_down_base_gather_ms'))} | "
                f"{fmt_ms(row.get('exact_down_dense_gate_up_ms'))} | "
                f"{fmt_ms(row.get('exact_down_base_gate_up_ms'))} | "
                f"{fmt_ms(row.get('exact_down_dense_act_ms'))} | "
                f"{fmt_ms(row.get('exact_down_base_act_ms'))} | "
                f"{fmt_ms(row.get('exact_down_dense_down_ms'))} | "
                f"{fmt_ms(row.get('exact_down_base_down_ms'))} | "
                f"{fmt_ms(row.get('sparse_gateup_dense_down_full_down_ms'))} | "
                f"{fmt_ms(row.get('exact_down_index_copy_assemble_ms'))} | "
                f"{fmt_ms(row.get('exact_down_triton_assemble_only_ms'))} |\n"
            )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SR24 row-routed MLP probe")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    output_root = args.output_root or (
        Path("examples/evaluate/eval-guidellm/results.bak")
        / f"speclink_sr24_row_routed_mlp_probe_{timestamp()}"
    )
    rows = [
        run_case(
            rows=args.rows,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            bucket_size=bucket,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            block_n=args.triton_block_n,
        )
        for bucket in args.bucket_sizes
    ]
    write_report(output_root, rows)
    print(output_root.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile SR24 row-routed MLP variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--bucket-sizes", type=parse_int_list, default=[16, 32, 64, 128])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--triton-block-n", type=int, default=1024)
    parser.add_argument("--output-root", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
