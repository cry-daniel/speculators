#!/usr/bin/env python3
"""Prototype a Triton small-M 2:4 base MLP kernel for SR24.

This is intentionally a microbenchmark, not a serving path.  The current SR24
evidence shows that bs8/bs16 are blocked by small-M 2:4 sparse MLP latency.  This
script tests a simple GPU-resident data format:

  values[n, group, 0:2] and absolute k positions k0/k1[n, group]

The Triton kernel computes the same output as a dense zeroed 2:4 base weight,
but it does not use NVIDIA sparse tensor cores.  A positive result would justify
turning this into a real CUDA/Triton operator; a negative result means the next
step must be a tensor-core-backed grouped/fused design instead of scalar Triton
gather-multiply loops.
"""

from __future__ import annotations

import argparse
import csv
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
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_block_configs(value: str) -> list[tuple[int, int, int]]:
    configs: list[tuple[int, int, int]] = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = item.replace("x", " ").split()
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"block config {item!r} must look like 16x16x32"
            )
        configs.append(tuple(int(part) for part in parts))  # type: ignore[arg-type]
    return configs


def silu_and_mul(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    return F.silu(gate) * up


def make_base24_artifacts(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    if in_features % 4:
        raise ValueError("in_features must be divisible by 4")
    groups = in_features // 4
    grouped = weight.view(out_features, groups, 4)
    keep_idx = grouped.abs().topk(2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)

    base_grouped = torch.zeros_like(grouped)
    base_grouped[keep] = grouped[keep]
    base_dense = base_grouped.view_as(weight).contiguous()

    pos = torch.arange(4, device=weight.device, dtype=torch.int32).view(1, 1, 4)
    kept_pos = pos.expand_as(grouped)[keep].view(out_features, groups, 2)
    values = grouped[keep].view(out_features, groups, 2).contiguous()
    k_base = (torch.arange(groups, device=weight.device, dtype=torch.int32) * 4).view(
        1, groups
    )
    k0 = (kept_pos[:, :, 0].to(torch.int32) + k_base).contiguous()
    k1 = (kept_pos[:, :, 1].to(torch.int32) + k_base).contiguous()
    return base_dense, values, k0, k1


@triton.jit
def _base24_abspos_linear_kernel(
    x_ptr,
    values_ptr,
    k0_ptr,
    k1_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_G: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group_start in tl.range(0, G, BLOCK_G):
        offs_g = group_start + tl.arange(0, BLOCK_G)
        ng_valid = (offs_n[:, None] < N) & (offs_g[None, :] < G)
        pos_base = offs_n[:, None] * G + offs_g[None, :]
        k0 = tl.load(k0_ptr + pos_base, mask=ng_valid, other=0).to(tl.int32)
        k1 = tl.load(k1_ptr + pos_base, mask=ng_valid, other=0).to(tl.int32)
        value_base = (offs_n[:, None] * G + offs_g[None, :]) * 2
        v0 = tl.load(values_ptr + value_base, mask=ng_valid, other=0.0)
        v1 = tl.load(values_ptr + value_base + 1, mask=ng_valid, other=0.0)

        x_valid = (offs_m[:, None, None] < M) & ng_valid[None, :, :]
        x0 = tl.load(
            x_ptr + offs_m[:, None, None] * K + k0[None, :, :],
            mask=x_valid,
            other=0.0,
        )
        x1 = tl.load(
            x_ptr + offs_m[:, None, None] * K + k1[None, :, :],
            mask=x_valid,
            other=0.0,
        )
        prod = x0.to(tl.float32) * v0[None, :, :] + x1.to(tl.float32) * v1[
            None, :, :
        ]
        acc += tl.sum(prod, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_base24_linear(
    x: torch.Tensor,
    *,
    values: torch.Tensor,
    k0: torch.Tensor,
    k1: torch.Tensor,
    out_features: int,
    in_features: int,
    block_m: int,
    block_n: int,
    block_g: int,
) -> torch.Tensor:
    rows = int(x.shape[0])
    groups = in_features // 4
    out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _base24_abspos_linear_kernel[grid](
        x.contiguous(),
        values,
        k0,
        k1,
        out,
        rows,
        out_features,
        in_features,
        groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_G=block_g,
        num_warps=4,
    )
    return out


def direct_cslt_linear(
    x: torch.Tensor,
    sparse_weight: Any,
    *,
    alg_id: int,
) -> torch.Tensor:
    packed = getattr(sparse_weight, "packed", None)
    if packed is None:
        raise RuntimeError("sparse weight does not expose packed cuSPARSELt data")
    padded_x = sparse_weight._pad_dense_input(x)
    out = torch._cslt_sparse_mm(
        packed,
        padded_x.t().contiguous(),
        transpose_result=False,
        alg_id=alg_id,
    ).t()
    return out[: x.shape[0], :]


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


def time_or_error(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float | None, str | None]:
    try:
        return time_call(fn, warmup=warmup, repeats=repeats), None
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def run_case(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    block_config: tuple[int, int, int],
    input_scale: float,
    weight_scale: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    block_m, block_n, block_g = block_config
    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype) * input_scale
    gate_up_weight = torch.randn(
        intermediate_size * 2,
        hidden_size,
        device="cuda",
        dtype=dtype,
    ) * weight_scale
    down_weight = torch.randn(
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=dtype,
    ) * weight_scale
    (
        gate_up_base_dense,
        gate_up_values,
        gate_up_k0,
        gate_up_k1,
    ) = make_base24_artifacts(gate_up_weight)
    (
        down_base_dense,
        down_values,
        down_k0,
        down_k1,
    ) = make_base24_artifacts(down_weight)
    gate_up_sparse = to_sparse_semi_structured(gate_up_base_dense)
    down_sparse = to_sparse_semi_structured(down_base_dense)

    def dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_weight)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_weight)

    def base_dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_base_dense)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_base_dense)

    def cslt_mlp_alg0() -> torch.Tensor:
        gate_up = direct_cslt_linear(x, gate_up_sparse, alg_id=0)
        act = silu_and_mul(gate_up, intermediate_size)
        return direct_cslt_linear(act, down_sparse, alg_id=0)

    def cslt_mlp_alg1() -> torch.Tensor:
        gate_up = direct_cslt_linear(x, gate_up_sparse, alg_id=1)
        act = silu_and_mul(gate_up, intermediate_size)
        return direct_cslt_linear(act, down_sparse, alg_id=1)

    def triton_mlp() -> torch.Tensor:
        gate_up = triton_base24_linear(
            x,
            values=gate_up_values,
            k0=gate_up_k0,
            k1=gate_up_k1,
            out_features=intermediate_size * 2,
            in_features=hidden_size,
            block_m=block_m,
            block_n=block_n,
            block_g=block_g,
        )
        act = silu_and_mul(gate_up, intermediate_size)
        return triton_base24_linear(
            act,
            values=down_values,
            k0=down_k0,
            k1=down_k1,
            out_features=hidden_size,
            in_features=intermediate_size,
            block_m=block_m,
            block_n=block_n,
            block_g=block_g,
        )

    base_out = base_dense_mlp()
    triton_out = triton_mlp()
    cslt0_out, cslt0_error = None, None
    try:
        cslt0_out = cslt_mlp_alg0()
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        cslt0_error = f"{type(exc).__name__}: {exc}"
    cslt1_out, cslt1_error = None, None
    try:
        cslt1_out = cslt_mlp_alg1()
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        cslt1_error = f"{type(exc).__name__}: {exc}"

    dense_graph_ms, dense_graph_error = time_graph(
        dense_mlp, warmup=warmup, repeats=repeats
    )
    base_dense_graph_ms, base_dense_graph_error = time_graph(
        base_dense_mlp, warmup=warmup, repeats=repeats
    )
    triton_graph_ms, triton_graph_error = time_graph(
        triton_mlp, warmup=warmup, repeats=repeats
    )
    cslt0_graph_ms, cslt0_graph_error = time_graph(
        cslt_mlp_alg0, warmup=warmup, repeats=repeats
    )
    cslt1_graph_ms, cslt1_graph_error = time_graph(
        cslt_mlp_alg1, warmup=warmup, repeats=repeats
    )
    cslt0_ms, cslt0_time_error = time_or_error(
        cslt_mlp_alg0, warmup=warmup, repeats=repeats
    )
    cslt1_ms, cslt1_time_error = time_or_error(
        cslt_mlp_alg1, warmup=warmup, repeats=repeats
    )

    dense_ref = dense_graph_ms or time_call(dense_mlp, warmup=warmup, repeats=repeats)
    triton_ref = triton_graph_ms or time_call(
        triton_mlp, warmup=warmup, repeats=repeats
    )
    base_absmax = float(base_out.abs().max().item())
    denom = base_absmax if base_absmax > 0 else 1.0
    triton_abs_diff = float((triton_out - base_out).abs().max().item())
    cslt0_abs_diff = (
        None
        if cslt0_out is None
        else float((cslt0_out - base_out).abs().max().item())
    )
    cslt1_abs_diff = (
        None
        if cslt1_out is None
        else float((cslt1_out - base_out).abs().max().item())
    )
    row = {
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype).replace("torch.", ""),
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "block_m": block_m,
        "block_n": block_n,
        "block_g": block_g,
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "base_dense_graph_ms": base_dense_graph_ms,
        "base_dense_graph_error": base_dense_graph_error,
        "cslt_alg0_graph_ms": cslt0_graph_ms,
        "cslt_alg0_graph_error": cslt0_graph_error or cslt0_error,
        "cslt_alg1_graph_ms": cslt1_graph_ms,
        "cslt_alg1_graph_error": cslt1_graph_error or cslt1_error,
        "triton_base24_graph_ms": triton_graph_ms,
        "triton_base24_graph_error": triton_graph_error,
        "cslt_alg0_ms": cslt0_ms,
        "cslt_alg0_error": cslt0_time_error or cslt0_error,
        "cslt_alg1_ms": cslt1_ms,
        "cslt_alg1_error": cslt1_time_error or cslt1_error,
        "triton_base24_ms": time_call(triton_mlp, warmup=warmup, repeats=repeats),
        "triton_vs_dense_speedup": dense_ref / triton_ref if triton_ref else None,
        "triton_over_dense": triton_ref / dense_ref if dense_ref else None,
        "base_out_absmax": base_absmax,
        "triton_vs_base_dense_max_diff": triton_abs_diff,
        "triton_vs_base_dense_rel_max_diff": triton_abs_diff / denom,
        "cslt_alg0_vs_base_dense_max_diff": cslt0_abs_diff,
        "cslt_alg0_vs_base_dense_rel_max_diff": (
            None if cslt0_abs_diff is None else cslt0_abs_diff / denom
        ),
        "cslt_alg1_vs_base_dense_max_diff": cslt1_abs_diff,
        "cslt_alg1_vs_base_dense_rel_max_diff": (
            None if cslt1_abs_diff is None else cslt1_abs_diff / denom
        ),
    }
    for key in ("cslt_alg0_graph_ms", "cslt_alg1_graph_ms", "base_dense_graph_ms"):
        value = row.get(key)
        if value is not None and dense_ref:
            row[f"{key}_speedup_vs_dense"] = dense_ref / float(value)
    return row


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def fmt_ratio(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}x"


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# SR24 Triton Base 2:4 MLP Prototype\n\n")
        f.write(
            "| rows | block | dense graph ms | cuSPARSELt alg0 graph ms | "
            "cuSPARSELt alg1 graph ms | Triton base24 graph ms | "
            "Triton speedup vs dense | Triton/base diff |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            block = f"{row['block_m']}x{row['block_n']}x{row['block_g']}"
            f.write(
                f"| {row['rows']} | {block} | "
                f"{fmt(row.get('dense_graph_ms')) or row.get('dense_graph_error', '')} | "
                f"{fmt(row.get('cslt_alg0_graph_ms')) or row.get('cslt_alg0_graph_error', '')} | "
                f"{fmt(row.get('cslt_alg1_graph_ms')) or row.get('cslt_alg1_graph_error', '')} | "
                f"{fmt(row.get('triton_base24_graph_ms')) or row.get('triton_base24_graph_error', '')} | "
                f"{fmt_ratio(row.get('triton_vs_dense_speedup'))} | "
                f"{fmt(row.get('triton_vs_base_dense_max_diff'))}"
                f" (rel {fmt(row.get('triton_vs_base_dense_rel_max_diff'))}) |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a custom Triton base 2:4 MLP prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=parse_int_list, default=[72, 144, 288, 576])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument(
        "--block-configs",
        type=parse_block_configs,
        default=parse_block_configs("16x16x32,16x32x32"),
        help="Comma-separated BLOCK_M x BLOCK_N x BLOCK_G configs.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton base 2:4 MLP probe")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    output_root = args.output_root or (
        Path("examples/evaluate/eval-guidellm/results.bak")
        / f"sr24_triton_base24_mlp_{timestamp()}"
    )
    rows: list[dict[str, Any]] = []
    for row_count in args.rows:
        for block_config in args.block_configs:
            rows.append(
                run_case(
                    rows=row_count,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.intermediate_size,
                    dtype=dtype,
                    block_config=block_config,
                    input_scale=args.input_scale,
                    weight_scale=args.weight_scale,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )
    write_outputs(output_root, rows)
    print(output_root.resolve())


if __name__ == "__main__":
    main()
