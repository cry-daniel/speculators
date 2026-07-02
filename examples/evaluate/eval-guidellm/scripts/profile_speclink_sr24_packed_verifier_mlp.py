#!/usr/bin/env python3
"""Profile a packed verifier-block SR24 mixed MLP.

This isolates the next intended SpecLink operator shape:

  hidden[batch, K + 1, hidden]
  dense rows = first `prefix` draft rows plus verifier bonus row
  sparse rows = remaining draft rows

The benchmark compares the full dense MLP, full 2:4 base MLP, and a packed
fixed-block mixed MLP that keeps dense-important rows and sparse-only rows
disjoint. It is a standalone CUDA microbench; it does not launch vLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    return values


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


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
) -> tuple[float | None, str]:
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
        return float(start.elapsed_time(end) / repeats), ""
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def fmt_ms(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def fmt_ratio(value: Any, base: Any) -> str:
    if value in (None, "") or base in (None, ""):
        return ""
    base_f = float(base)
    if base_f <= 0:
        return ""
    return f"{float(value) / base_f:.3f}x"


def run_case(
    *,
    batch_size: int,
    coalesce_factor: int,
    k: int,
    prefix: int,
    capacity_multiple: int,
    min_dense_capacity: int,
    min_base_capacity: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    width = k + 1
    valid_width = k
    coalesce_factor = max(1, int(coalesce_factor))
    effective_batch_size = batch_size * coalesce_factor
    single_block_rows = batch_size * width
    rows = effective_batch_size * width
    prefix = max(0, min(prefix, k))
    dense_rows_per_request = prefix + 1
    base_rows_per_request = max(0, k - prefix)
    single_dense_count = batch_size * dense_rows_per_request
    single_base_count = batch_size * base_rows_per_request
    dense_count = effective_batch_size * dense_rows_per_request
    base_count = effective_batch_size * base_rows_per_request
    dense_capacity = max(
        dense_count,
        min_dense_capacity,
        round_up(dense_count, capacity_multiple),
    )
    base_capacity = max(
        base_count,
        min_base_capacity if base_count > 0 else 0,
        round_up(base_count, capacity_multiple) if base_count > 0 else 0,
    )

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
    x_blocks = x.reshape(effective_batch_size, width, hidden_size)
    x_single = x[:single_block_rows]
    dense_buffer = torch.zeros(dense_capacity, hidden_size, device="cuda", dtype=dtype)
    base_buffer = torch.zeros(base_capacity, hidden_size, device="cuda", dtype=dtype)

    def dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_weight)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_weight)

    def base_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_base)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_base)

    def single_dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x_single, gate_up_weight)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_weight)

    def dense_input_from_blocks() -> torch.Tensor:
        dense_parts: list[torch.Tensor] = []
        if prefix > 0:
            dense_parts.append(x_blocks[:, :prefix, :].reshape(-1, hidden_size))
        dense_parts.append(
            x_blocks[:, valid_width:valid_width + 1, :].reshape(-1, hidden_size)
        )
        return dense_parts[0] if len(dense_parts) == 1 else torch.cat(
            dense_parts, dim=0
        )

    def base_input_from_blocks() -> torch.Tensor:
        if base_rows_per_request <= 0:
            return torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return x_blocks[:, prefix:valid_width, :].reshape(-1, hidden_size)

    dense_stream = torch.cuda.Stream()
    base_stream = torch.cuda.Stream()

    def packed_mlp() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()

        dense_gate_up = F.linear(dense_input, gate_up_weight)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        dense_down = F.linear(dense_act, down_weight)

        if base_count > 0:
            base_gate_up = F.linear(base_input, gate_up_base)
            base_act = silu_and_mul(base_gate_up, intermediate_size)
            base_down = F.linear(base_act, down_base)
        else:
            base_down = torch.empty(0, hidden_size, device="cuda", dtype=dtype)

        output_blocks = torch.empty(
            effective_batch_size,
            width,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        dense_prefix_rows = effective_batch_size * prefix
        if prefix > 0:
            output_blocks[:, :prefix, :].copy_(
                dense_down[:dense_prefix_rows].reshape(
                    effective_batch_size, prefix, hidden_size
                )
            )
        output_blocks[:, valid_width, :].copy_(
            dense_down[dense_prefix_rows:].reshape(
                effective_batch_size, hidden_size
            )
        )
        if base_count > 0:
            output_blocks[:, prefix:valid_width, :].copy_(
                base_down.reshape(
                    effective_batch_size, base_rows_per_request, hidden_size
                )
            )
        return output_blocks.reshape(rows, hidden_size).contiguous()

    def packed_parallel_mlp() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        launch_stream = torch.cuda.current_stream()

        dense_stream.wait_stream(launch_stream)
        with torch.cuda.stream(dense_stream):
            dense_gate_up = F.linear(dense_input, gate_up_weight)
            dense_act = silu_and_mul(dense_gate_up, intermediate_size)
            dense_down = F.linear(dense_act, down_weight)

        if base_count > 0:
            base_stream.wait_stream(launch_stream)
            with torch.cuda.stream(base_stream):
                base_gate_up = F.linear(base_input, gate_up_base)
                base_act = silu_and_mul(base_gate_up, intermediate_size)
                base_down = F.linear(base_act, down_base)
            launch_stream.wait_stream(base_stream)
        else:
            base_down = torch.empty(0, hidden_size, device="cuda", dtype=dtype)

        launch_stream.wait_stream(dense_stream)
        output_blocks = torch.empty(
            effective_batch_size,
            width,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        dense_prefix_rows = effective_batch_size * prefix
        if prefix > 0:
            output_blocks[:, :prefix, :].copy_(
                dense_down[:dense_prefix_rows].reshape(
                    effective_batch_size, prefix, hidden_size
                )
            )
        output_blocks[:, valid_width, :].copy_(
            dense_down[dense_prefix_rows:].reshape(
                effective_batch_size, hidden_size
            )
        )
        if base_count > 0:
            output_blocks[:, prefix:valid_width, :].copy_(
                base_down.reshape(
                    effective_batch_size, base_rows_per_request, hidden_size
                )
            )
        return output_blocks.reshape(rows, hidden_size).contiguous()

    def packed_no_final_assemble() -> tuple[torch.Tensor, torch.Tensor]:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        dense_gate_up = F.linear(dense_input, gate_up_weight)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        dense_down = F.linear(dense_act, down_weight)
        if base_count > 0:
            base_gate_up = F.linear(base_input, gate_up_base)
            base_act = silu_and_mul(base_gate_up, intermediate_size)
            base_down = F.linear(base_act, down_base)
        else:
            base_down = torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return dense_down, base_down

    def packed_parallel_no_final_assemble() -> tuple[torch.Tensor, torch.Tensor]:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        launch_stream = torch.cuda.current_stream()

        dense_stream.wait_stream(launch_stream)
        with torch.cuda.stream(dense_stream):
            dense_gate_up = F.linear(dense_input, gate_up_weight)
            dense_act = silu_and_mul(dense_gate_up, intermediate_size)
            dense_down = F.linear(dense_act, down_weight)

        if base_count > 0:
            base_stream.wait_stream(launch_stream)
            with torch.cuda.stream(base_stream):
                base_gate_up = F.linear(base_input, gate_up_base)
                base_act = silu_and_mul(base_gate_up, intermediate_size)
                base_down = F.linear(base_act, down_base)
            launch_stream.wait_stream(base_stream)
        else:
            base_down = torch.empty(0, hidden_size, device="cuda", dtype=dtype)

        launch_stream.wait_stream(dense_stream)
        return dense_down, base_down

    def packed_reuse_base_output() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        dense_gate_up = F.linear(dense_input, gate_up_weight)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        dense_down = F.linear(dense_act, down_weight)

        base_gate_up = F.linear(x, gate_up_base)
        base_act = silu_and_mul(base_gate_up, intermediate_size)
        base_out = F.linear(base_act, down_base).reshape(
            effective_batch_size, width, hidden_size
        )

        dense_prefix_rows = effective_batch_size * prefix
        if prefix > 0:
            base_out[:, :prefix, :].copy_(
                dense_down[:dense_prefix_rows].reshape(
                    effective_batch_size, prefix, hidden_size
                )
            )
        base_out[:, valid_width, :].copy_(
            dense_down[dense_prefix_rows:].reshape(
                effective_batch_size, hidden_size
            )
        )
        return base_out.reshape(rows, hidden_size).contiguous()

    def packed_capacity_padded_mlp() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        dense_buffer[:dense_count].copy_(dense_input)
        dense_gate_up = F.linear(dense_buffer, gate_up_weight)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        dense_down = F.linear(dense_act, down_weight)

        if base_count > 0:
            base_input = base_input_from_blocks()
            base_buffer[:base_count].copy_(base_input)
            base_gate_up = F.linear(base_buffer, gate_up_base)
            base_act = silu_and_mul(base_gate_up, intermediate_size)
            base_down = F.linear(base_act, down_base)
        else:
            base_down = torch.empty(0, hidden_size, device="cuda", dtype=dtype)

        output_blocks = torch.empty(
            effective_batch_size,
            width,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        dense_prefix_rows = effective_batch_size * prefix
        if prefix > 0:
            output_blocks[:, :prefix, :].copy_(
                dense_down[:dense_prefix_rows].reshape(
                    effective_batch_size, prefix, hidden_size
                )
            )
        output_blocks[:, valid_width, :].copy_(
            dense_down[dense_prefix_rows:dense_count].reshape(
                effective_batch_size, hidden_size
            )
        )
        if base_count > 0:
            output_blocks[:, prefix:valid_width, :].copy_(
                base_down[:base_count].reshape(
                    effective_batch_size, base_rows_per_request, hidden_size
                )
            )
        return output_blocks.reshape(rows, hidden_size).contiguous()

    def packed_capacity_padded_no_final_assemble() -> tuple[torch.Tensor, torch.Tensor]:
        dense_input = dense_input_from_blocks()
        dense_buffer[:dense_count].copy_(dense_input)
        dense_gate_up = F.linear(dense_buffer, gate_up_weight)
        dense_act = silu_and_mul(dense_gate_up, intermediate_size)
        dense_down = F.linear(dense_act, down_weight)

        if base_count > 0:
            base_input = base_input_from_blocks()
            base_buffer[:base_count].copy_(base_input)
            base_gate_up = F.linear(base_buffer, gate_up_base)
            base_act = silu_and_mul(base_gate_up, intermediate_size)
            base_down = F.linear(base_act, down_base)
        else:
            base_down = torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return dense_down[:dense_count], base_down[:base_count]

    # Run correctness-shape checks once before timing.
    dense_out = dense_mlp()
    single_dense_out = single_dense_mlp()
    base_out = base_mlp()
    packed_out = packed_mlp()
    packed_parallel_out = packed_parallel_mlp()
    reuse_out = packed_reuse_base_output()
    padded_out = packed_capacity_padded_mlp()
    dense_branch, base_branch = packed_no_final_assemble()
    parallel_dense_branch, parallel_base_branch = packed_parallel_no_final_assemble()
    padded_dense_branch, padded_base_branch = packed_capacity_padded_no_final_assemble()
    torch.cuda.synchronize()

    dense_graph_ms, dense_graph_error = time_graph(
        dense_mlp, warmup=warmup, repeats=repeats
    )
    single_dense_graph_ms, single_dense_graph_error = time_graph(
        single_dense_mlp, warmup=warmup, repeats=repeats
    )
    base_graph_ms, base_graph_error = time_graph(
        base_mlp, warmup=warmup, repeats=repeats
    )
    packed_graph_ms, packed_graph_error = time_graph(
        packed_mlp, warmup=warmup, repeats=repeats
    )
    no_assemble_graph_ms, no_assemble_graph_error = time_graph(
        packed_no_final_assemble, warmup=warmup, repeats=repeats
    )
    parallel_graph_ms, parallel_graph_error = time_graph(
        packed_parallel_mlp, warmup=warmup, repeats=repeats
    )
    parallel_no_assemble_graph_ms, parallel_no_assemble_graph_error = time_graph(
        packed_parallel_no_final_assemble, warmup=warmup, repeats=repeats
    )
    reuse_graph_ms, reuse_graph_error = time_graph(
        packed_reuse_base_output, warmup=warmup, repeats=repeats
    )
    padded_graph_ms, padded_graph_error = time_graph(
        packed_capacity_padded_mlp, warmup=warmup, repeats=repeats
    )
    padded_no_assemble_graph_ms, padded_no_assemble_graph_error = time_graph(
        packed_capacity_padded_no_final_assemble, warmup=warmup, repeats=repeats
    )

    dense_ms = time_call(dense_mlp, warmup=warmup, repeats=repeats)
    single_dense_ms = time_call(single_dense_mlp, warmup=warmup, repeats=repeats)
    base_ms = time_call(base_mlp, warmup=warmup, repeats=repeats)
    packed_ms = time_call(packed_mlp, warmup=warmup, repeats=repeats)
    no_assemble_ms = time_call(
        packed_no_final_assemble, warmup=warmup, repeats=repeats
    )
    parallel_ms = time_call(packed_parallel_mlp, warmup=warmup, repeats=repeats)
    parallel_no_assemble_ms = time_call(
        packed_parallel_no_final_assemble, warmup=warmup, repeats=repeats
    )
    reuse_ms = time_call(packed_reuse_base_output, warmup=warmup, repeats=repeats)
    padded_ms = time_call(packed_capacity_padded_mlp, warmup=warmup, repeats=repeats)
    padded_no_assemble_ms = time_call(
        packed_capacity_padded_no_final_assemble, warmup=warmup, repeats=repeats
    )

    return {
        "batch_size": batch_size,
        "coalesce_factor": coalesce_factor,
        "effective_batch_size": effective_batch_size,
        "k": k,
        "width": width,
        "rows": rows,
        "single_block_rows": single_block_rows,
        "prefix": prefix,
        "dense_rows_per_request": dense_rows_per_request,
        "base_rows_per_request": base_rows_per_request,
        "single_dense_rows": single_dense_count,
        "single_base_rows": single_base_count,
        "dense_rows": dense_count,
        "base_rows": base_count,
        "capacity_multiple": capacity_multiple,
        "dense_capacity": dense_capacity,
        "base_capacity": base_capacity,
        "dense_capacity_fill": (
            dense_count / dense_capacity if dense_capacity > 0 else None
        ),
        "base_capacity_fill": base_count / base_capacity if base_capacity > 0 else None,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype).replace("torch.", ""),
        "dense_ms": dense_ms,
        "single_dense_ms": single_dense_ms,
        "serial_dense_reference_ms": single_dense_ms * coalesce_factor,
        "base_ms": base_ms,
        "packed_ms": packed_ms,
        "packed_no_assemble_ms": no_assemble_ms,
        "packed_parallel_ms": parallel_ms,
        "packed_parallel_no_assemble_ms": parallel_no_assemble_ms,
        "packed_reuse_base_ms": reuse_ms,
        "packed_capacity_padded_ms": padded_ms,
        "packed_capacity_padded_no_assemble_ms": padded_no_assemble_ms,
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "single_dense_graph_ms": single_dense_graph_ms,
        "single_dense_graph_error": single_dense_graph_error,
        "serial_dense_reference_graph_ms": (
            single_dense_graph_ms * coalesce_factor
            if single_dense_graph_ms is not None
            else None
        ),
        "base_graph_ms": base_graph_ms,
        "base_graph_error": base_graph_error,
        "packed_graph_ms": packed_graph_ms,
        "packed_graph_error": packed_graph_error,
        "packed_no_assemble_graph_ms": no_assemble_graph_ms,
        "packed_no_assemble_graph_error": no_assemble_graph_error,
        "packed_parallel_graph_ms": parallel_graph_ms,
        "packed_parallel_graph_error": parallel_graph_error,
        "packed_parallel_no_assemble_graph_ms": parallel_no_assemble_graph_ms,
        "packed_parallel_no_assemble_graph_error": parallel_no_assemble_graph_error,
        "packed_reuse_base_graph_ms": reuse_graph_ms,
        "packed_reuse_base_graph_error": reuse_graph_error,
        "packed_capacity_padded_graph_ms": padded_graph_ms,
        "packed_capacity_padded_graph_error": padded_graph_error,
        "packed_capacity_padded_no_assemble_graph_ms": padded_no_assemble_graph_ms,
        "packed_capacity_padded_no_assemble_graph_error": (
            padded_no_assemble_graph_error
        ),
        "base_vs_dense_graph_ratio": (
            base_graph_ms / dense_graph_ms
            if base_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_vs_dense_graph_ratio": (
            packed_graph_ms / dense_graph_ms
            if packed_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_no_assemble_vs_dense_graph_ratio": (
            no_assemble_graph_ms / dense_graph_ms
            if no_assemble_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_parallel_vs_dense_graph_ratio": (
            parallel_graph_ms / dense_graph_ms
            if parallel_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_parallel_no_assemble_vs_dense_graph_ratio": (
            parallel_no_assemble_graph_ms / dense_graph_ms
            if parallel_no_assemble_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_reuse_base_vs_dense_graph_ratio": (
            reuse_graph_ms / dense_graph_ms
            if reuse_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_capacity_padded_vs_dense_graph_ratio": (
            padded_graph_ms / dense_graph_ms
            if padded_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_capacity_padded_no_assemble_vs_dense_graph_ratio": (
            padded_no_assemble_graph_ms / dense_graph_ms
            if padded_no_assemble_graph_ms is not None and dense_graph_ms
            else None
        ),
        "packed_parallel_vs_serial_dense_graph_speedup": (
            (single_dense_graph_ms * coalesce_factor) / parallel_graph_ms
            if single_dense_graph_ms is not None
            and parallel_graph_ms is not None
            and parallel_graph_ms > 0
            else None
        ),
        "packed_vs_serial_dense_graph_speedup": (
            (single_dense_graph_ms * coalesce_factor) / packed_graph_ms
            if single_dense_graph_ms is not None
            and packed_graph_ms is not None
            and packed_graph_ms > 0
            else None
        ),
        "coalesced_dense_vs_serial_dense_graph_speedup": (
            (single_dense_graph_ms * coalesce_factor) / dense_graph_ms
            if single_dense_graph_ms is not None
            and dense_graph_ms is not None
            and dense_graph_ms > 0
            else None
        ),
        "dense_output_norm": float(dense_out.norm().item()),
        "single_dense_output_norm": float(single_dense_out.norm().item()),
        "base_output_norm": float(base_out.norm().item()),
        "packed_output_norm": float(packed_out.norm().item()),
        "packed_parallel_output_norm": float(packed_parallel_out.norm().item()),
        "reuse_output_norm": float(reuse_out.norm().item()),
        "padded_output_norm": float(padded_out.norm().item()),
        "packed_branch_rows": int(dense_branch.shape[0] + base_branch.shape[0]),
        "packed_parallel_branch_rows": int(
            parallel_dense_branch.shape[0] + parallel_base_branch.shape[0]
        ),
        "packed_capacity_padded_branch_rows": int(
            padded_dense_branch.shape[0] + padded_base_branch.shape[0]
        ),
    }


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no rows to write")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# SR24 Packed Verifier-Block MLP Microbenchmark\n\n")
        f.write(
            "Rows are shaped as `[batch, K + 1, hidden]`. `packed` routes "
            "the first `prefix` draft rows plus the verifier bonus row through "
            "dense MLP and the remaining draft rows through 2:4 sparse MLP.\n\n"
        )
        f.write(
            "| bs | coalesce | effective bs | K | prefix | rows | dense rows | "
            "base rows | dense graph ms | serial dense graph ms | "
            "base graph ms | packed graph ms | no-assemble graph ms | "
            "parallel graph ms | parallel no-asm graph ms | reuse-base graph ms | "
            "padded graph ms | packed/dense | parallel/dense | padded/dense | "
            "parallel speedup | coalesced dense vs serial | "
            "parallel vs serial speedup |\n"
        )
        f.write(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            dense_graph = row.get("dense_graph_ms") or row.get("dense_ms")
            serial_dense_graph = (
                row.get("serial_dense_reference_graph_ms")
                or row.get("serial_dense_reference_ms")
            )
            packed_graph = row.get("packed_graph_ms")
            parallel_graph = row.get("packed_parallel_graph_ms")
            padded_graph = row.get("packed_capacity_padded_graph_ms")
            parallel_speedup = (
                float(dense_graph) / float(parallel_graph)
                if dense_graph not in (None, "") and parallel_graph not in (None, "")
                else None
            )
            coalesced_dense_serial_speedup = row.get(
                "coalesced_dense_vs_serial_dense_graph_speedup"
            )
            parallel_serial_speedup = row.get(
                "packed_parallel_vs_serial_dense_graph_speedup"
            )
            f.write(
                f"| {row['batch_size']} | {row['coalesce_factor']} | "
                f"{row['effective_batch_size']} | {row['k']} | "
                f"{row['prefix']} | {row['rows']} | {row['dense_rows']} | "
                f"{row['base_rows']} | "
                f"{fmt_ms(row.get('dense_graph_ms')) or row.get('dense_graph_error')} | "
                f"{fmt_ms(serial_dense_graph)} | "
                f"{fmt_ms(row.get('base_graph_ms')) or row.get('base_graph_error')} | "
                f"{fmt_ms(row.get('packed_graph_ms')) or row.get('packed_graph_error')} | "
                f"{fmt_ms(row.get('packed_no_assemble_graph_ms')) or row.get('packed_no_assemble_graph_error')} | "
                f"{fmt_ms(row.get('packed_parallel_graph_ms')) or row.get('packed_parallel_graph_error')} | "
                f"{fmt_ms(row.get('packed_parallel_no_assemble_graph_ms')) or row.get('packed_parallel_no_assemble_graph_error')} | "
                f"{fmt_ms(row.get('packed_reuse_base_graph_ms')) or row.get('packed_reuse_base_graph_error')} | "
                f"{fmt_ms(row.get('packed_capacity_padded_graph_ms')) or row.get('packed_capacity_padded_graph_error')} | "
                f"{fmt_ratio(row.get('packed_graph_ms'), dense_graph)} | "
                f"{fmt_ratio(row.get('packed_parallel_graph_ms'), dense_graph)} | "
                f"{fmt_ratio(row.get('packed_capacity_padded_graph_ms'), dense_graph)} | "
                f"{'' if parallel_speedup is None else f'{parallel_speedup:.3f}x'} | "
                f"{'' if coalesced_dense_serial_speedup in (None, '') else f'{float(coalesced_dense_serial_speedup):.3f}x'} | "
                f"{'' if parallel_serial_speedup in (None, '') else f'{float(parallel_serial_speedup):.3f}x'} |\n"
            )

        f.write("\n## Fixed-Capacity Fill\n\n")
        f.write(
            "| bs | coalesce | K | prefix | capacity multiple | "
            "dense active/cap | base active/cap | dense fill | base fill |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            dense_fill = row.get("dense_capacity_fill")
            base_fill = row.get("base_capacity_fill")
            f.write(
                f"| {row['batch_size']} | {row['coalesce_factor']} | "
                f"{row['k']} | {row['prefix']} | {row['capacity_multiple']} | "
                f"{row['dense_rows']}/{row['dense_capacity']} | "
                f"{row['base_rows']}/{row['base_capacity']} | "
                f"{'' if dense_fill is None else f'{float(dense_fill):.3f}'} | "
                f"{'' if base_fill is None else f'{float(base_fill):.3f}'} |\n"
            )

        f.write("\n## Useful-Row Coalescing\n\n")
        f.write(
            "`coalesce > 1` simulates grouping that many independent verifier "
            "blocks with the same weights into one larger dense/sparse branch. "
            "`parallel vs serial dense` compares the grouped parallel mixed path "
            "against running the original batch-size dense MLP once per block. "
            "This is an optimistic upper bound for useful-row coalescing; it is "
            "not a proof that decode steps can be delayed or reordered safely.\n"
        )

        f.write("\n## Read\n\n")
        f.write(
            "- `packed/dense < 0.80x` is the rough standalone target for a "
            "future live path to have a chance at `>=1.2x` after scheduler and "
            "serving overhead.\n"
        )
        f.write(
            "- `no-assemble` is a lower bound showing whether final block "
            "assembly is the main remaining cost.\n"
        )
        f.write(
            "- `reuse-base` computes sparse MLP for all rows and overwrites "
            "dense rows; it is graph-friendly but violates the no-duplicate-work "
            "goal for important rows.\n"
        )
        f.write(
            "- `parallel` launches the dense-important and sparse-base branches "
            "on separate CUDA streams, then joins before final assembly.\n"
        )
        f.write(
            "- `padded` copies active rows into fixed-capacity dense/base "
            "buffers before the branch GEMMs. It measures whether padding to "
            "stable tensor-core-friendly capacities is a useful way to keep "
            "small sparse branches filled. A useful padded path must beat the "
            "unpadded packed path by enough to repay the extra dummy rows.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument(
        "--coalesce-factors",
        default="1",
        help=(
            "Comma-separated useful verifier-block coalescing factors. A factor "
            "of 4 benchmarks one grouped branch over four original batch-size "
            "blocks and reports speedup over four serial dense blocks."
        ),
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--prefixes", default="1,2,4")
    parser.add_argument(
        "--capacity-multiple",
        type=int,
        default=128,
        help="Round fixed dense/base branch capacities up to this row multiple.",
    )
    parser.add_argument(
        "--min-dense-capacity",
        type=int,
        default=0,
        help="Minimum fixed capacity for the dense-important branch.",
    )
    parser.add_argument(
        "--min-base-capacity",
        type=int,
        default=0,
        help="Minimum fixed capacity for the sparse-base branch when nonempty.",
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results.bak") / f"sr24_packed_verifier_mlp_{timestamp()}",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rows: list[dict[str, Any]] = []
    for batch_size in parse_int_list(args.batch_sizes):
        for coalesce_factor in parse_int_list(args.coalesce_factors):
            for prefix in parse_int_list(args.prefixes):
                rows.append(
                    run_case(
                        batch_size=batch_size,
                        coalesce_factor=coalesce_factor,
                        k=args.k,
                        prefix=prefix,
                        capacity_multiple=args.capacity_multiple,
                        min_dense_capacity=args.min_dense_capacity,
                        min_base_capacity=args.min_base_capacity,
                        hidden_size=args.hidden_size,
                        intermediate_size=args.intermediate_size,
                        dtype=dtype,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                )
    write_outputs(args.output_root, rows)
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
