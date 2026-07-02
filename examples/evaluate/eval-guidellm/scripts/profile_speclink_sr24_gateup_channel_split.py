#!/usr/bin/env python3
"""Microbenchmark paired-channel SR24 split for fused Llama gate_up_proj.

The fused gate_up projection has rows laid out as:

    [gate channels..., up channels...]

This benchmark keeps both rows for selected intermediate channels dense and
uses a 2:4 sparse base for the remaining paired channels. It also times the
cost of reassembling the result back into the original fused row order.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
    return f"{float(value) / base:.3f}x"


def fmt_speedup(value: Any, base: float | None) -> str:
    if value is None or base is None or float(value) <= 0:
        return ""
    return f"{base / float(value):.3f}x"


def select_channel_rows(
    weight: torch.Tensor,
    *,
    intermediate_size: int,
    dense_fraction: float,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= dense_fraction <= 1.0:
        raise ValueError("dense_fraction must be in [0, 1]")
    dense_count = int(round(intermediate_size * dense_fraction))
    dense_count = min(max(dense_count, 0), intermediate_size)
    channel_ids = torch.arange(intermediate_size, device=weight.device)
    if dense_count == 0:
        dense_channels = channel_ids[:0]
    elif dense_count == intermediate_size:
        dense_channels = channel_ids
    elif strategy == "front":
        dense_channels = channel_ids[:dense_count]
    elif strategy == "norm":
        gate_score = weight[:intermediate_size].float().pow(2).mean(dim=1)
        up_score = weight[intermediate_size:].float().pow(2).mean(dim=1)
        scores = gate_score + up_score
        dense_channels = scores.topk(dense_count, largest=True, sorted=True).indices
    else:
        raise ValueError(f"unsupported strategy: {strategy}")

    dense_mask = torch.zeros(intermediate_size, device=weight.device, dtype=torch.bool)
    dense_mask[dense_channels] = True
    sparse_channels = channel_ids[~dense_mask]
    dense_rows = torch.cat([dense_channels, dense_channels + intermediate_size])
    sparse_rows = torch.cat([sparse_channels, sparse_channels + intermediate_size])
    return (
        dense_channels.to(torch.long).contiguous(),
        sparse_channels.to(torch.long).contiguous(),
        dense_rows.to(torch.long).contiguous(),
        sparse_rows.to(torch.long).contiguous(),
    )


def run_case(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    dense_fraction: float,
    strategy: str,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    out_features = intermediate_size * 2
    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(out_features, hidden_size, device="cuda", dtype=dtype)
    down_weight = torch.randn(hidden_size,
                              intermediate_size,
                              device="cuda",
                              dtype=dtype)
    base_weight = make_base_24(weight)
    sparse_base_full = to_sparse_semi_structured(base_weight)
    dense_channels, sparse_channels, dense_rows, sparse_rows = select_channel_rows(
        weight,
        intermediate_size=intermediate_size,
        dense_fraction=dense_fraction,
        strategy=strategy,
    )
    grouped_channels = torch.cat([dense_channels, sparse_channels])
    grouped_down_weight = down_weight.index_select(1, grouped_channels).contiguous()
    dense_weight = weight.index_select(0, dense_rows).contiguous()
    sparse_weight_dense = base_weight.index_select(0, sparse_rows).contiguous()
    sparse_weight = to_sparse_semi_structured(sparse_weight_dense)
    dense_rows_cpu = dense_rows.detach().cpu()
    sparse_rows_cpu = sparse_rows.detach().cpu()
    fused_act_available = hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "silu_and_mul")

    split_output = torch.empty(rows, out_features, device="cuda", dtype=dtype)

    def dense_full() -> torch.Tensor:
        return F.linear(x, weight)

    def sparse_full() -> torch.Tensor:
        return direct_cslt_linear(x, sparse_base_full)

    def split_matmuls() -> tuple[torch.Tensor, torch.Tensor]:
        dense_part = F.linear(x, dense_weight)
        sparse_part = direct_cslt_linear(x, sparse_weight)
        return dense_part, sparse_part

    def split_with_index_copy() -> torch.Tensor:
        dense_part, sparse_part = split_matmuls()
        split_output.index_copy_(1, dense_rows, dense_part)
        split_output.index_copy_(1, sparse_rows, sparse_part)
        return split_output

    def split_with_new_empty() -> torch.Tensor:
        dense_part, sparse_part = split_matmuls()
        out = torch.empty(rows, out_features, device="cuda", dtype=dtype)
        out.index_copy_(1, dense_rows, dense_part)
        out.index_copy_(1, sparse_rows, sparse_part)
        return out

    def dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x, weight)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        return F.linear(act, down_weight)

    def split_grouped_mlp() -> torch.Tensor:
        dense_part, sparse_part = split_matmuls()
        dense_gate, dense_up = dense_part.chunk(2, dim=-1)
        sparse_gate, sparse_up = sparse_part.chunk(2, dim=-1)
        dense_act = F.silu(dense_gate) * dense_up
        sparse_act = F.silu(sparse_gate) * sparse_up
        grouped_act = torch.cat([dense_act, sparse_act], dim=-1)
        return F.linear(grouped_act, grouped_down_weight)

    def split_grouped_mlp_fused_act() -> torch.Tensor:
        dense_part, sparse_part = split_matmuls()
        dense_gate, dense_up = dense_part.chunk(2, dim=-1)
        sparse_gate, sparse_up = sparse_part.chunk(2, dim=-1)
        grouped_gate = torch.cat([dense_gate, sparse_gate], dim=-1)
        grouped_up = torch.cat([dense_up, sparse_up], dim=-1)
        grouped_gate_up = torch.cat([grouped_gate, grouped_up], dim=-1)
        grouped_act = torch.empty(
            (rows, intermediate_size),
            device=grouped_gate_up.device,
            dtype=grouped_gate_up.dtype,
        )
        torch.ops._C.silu_and_mul(grouped_act, grouped_gate_up)
        return F.linear(grouped_act, grouped_down_weight)

    # Sanity: split output should equal a dense matmul where sparse rows use the
    # 2:4 base and dense rows use original weights.
    expected_weight = base_weight.clone()
    expected_weight.index_copy_(0, dense_rows, dense_weight)
    expected = F.linear(x, expected_weight)
    actual = split_with_index_copy()
    max_abs_diff = float((expected - actual).abs().max().detach().cpu())
    torch.cuda.synchronize()

    dense_ms = time_call(dense_full, warmup=warmup, repeats=repeats)
    dense_graph_ms, dense_graph_error = time_graph(
        dense_full, warmup=warmup, repeats=repeats)
    sparse_ms = time_call(sparse_full, warmup=warmup, repeats=repeats)
    sparse_graph_ms, sparse_graph_error = time_graph(
        sparse_full, warmup=warmup, repeats=repeats)
    split_matmuls_ms = time_call(split_matmuls, warmup=warmup, repeats=repeats)
    split_matmuls_graph_ms, split_matmuls_graph_error = time_graph(
        split_matmuls, warmup=warmup, repeats=repeats)
    split_index_ms = time_call(
        split_with_index_copy, warmup=warmup, repeats=repeats)
    split_index_graph_ms, split_index_graph_error = time_graph(
        split_with_index_copy, warmup=warmup, repeats=repeats)
    split_new_ms = time_call(split_with_new_empty, warmup=warmup, repeats=repeats)
    split_new_graph_ms, split_new_graph_error = time_graph(
        split_with_new_empty, warmup=warmup, repeats=repeats)
    dense_mlp_ms = time_call(dense_mlp, warmup=warmup, repeats=repeats)
    dense_mlp_graph_ms, dense_mlp_graph_error = time_graph(
        dense_mlp, warmup=warmup, repeats=repeats)
    split_grouped_mlp_ms = time_call(
        split_grouped_mlp, warmup=warmup, repeats=repeats)
    split_grouped_mlp_graph_ms, split_grouped_mlp_graph_error = time_graph(
        split_grouped_mlp, warmup=warmup, repeats=repeats)
    if fused_act_available:
        split_grouped_mlp_fused_act_ms = time_call(
            split_grouped_mlp_fused_act, warmup=warmup, repeats=repeats)
        (
            split_grouped_mlp_fused_act_graph_ms,
            split_grouped_mlp_fused_act_graph_error,
        ) = time_graph(split_grouped_mlp_fused_act,
                       warmup=warmup,
                       repeats=repeats)
    else:
        split_grouped_mlp_fused_act_ms = None
        split_grouped_mlp_fused_act_graph_ms = None
        split_grouped_mlp_fused_act_graph_error = (
            "torch.ops._C.silu_and_mul unavailable outside vLLM serve"
        )

    assembly_ms = split_index_ms - split_matmuls_ms
    assembly_graph_ms = (
        split_index_graph_ms - split_matmuls_graph_ms
        if split_index_graph_ms is not None and split_matmuls_graph_ms is not None
        else None
    )

    return {
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "out_features": out_features,
        "dtype": str(dtype).replace("torch.", ""),
        "strategy": strategy,
        "dense_fraction": dense_fraction,
        "dense_channel_count": int(dense_rows.numel() // 2),
        "sparse_channel_count": int(sparse_rows.numel() // 2),
        "dense_row_count": int(dense_rows.numel()),
        "sparse_row_count": int(sparse_rows.numel()),
        "dense_rows_preview": dense_rows_cpu[:16].tolist(),
        "sparse_rows_preview": sparse_rows_cpu[:16].tolist(),
        "max_abs_diff": max_abs_diff,
        "dense_ms": dense_ms,
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "sparse_full_ms": sparse_ms,
        "sparse_full_graph_ms": sparse_graph_ms,
        "sparse_full_graph_error": sparse_graph_error,
        "split_matmuls_ms": split_matmuls_ms,
        "split_matmuls_graph_ms": split_matmuls_graph_ms,
        "split_matmuls_graph_error": split_matmuls_graph_error,
        "split_index_copy_ms": split_index_ms,
        "split_index_copy_graph_ms": split_index_graph_ms,
        "split_index_copy_graph_error": split_index_graph_error,
        "split_new_empty_ms": split_new_ms,
        "split_new_empty_graph_ms": split_new_graph_ms,
        "split_new_empty_graph_error": split_new_graph_error,
        "assembly_ms": assembly_ms,
        "assembly_graph_ms": assembly_graph_ms,
        "split_index_speedup_vs_dense": dense_ms / split_index_ms,
        "split_index_graph_speedup_vs_dense_graph": (
            dense_graph_ms / split_index_graph_ms
            if dense_graph_ms is not None and split_index_graph_ms is not None
            else None
        ),
        "split_matmuls_speedup_vs_dense": dense_ms / split_matmuls_ms,
        "split_matmuls_graph_speedup_vs_dense_graph": (
            dense_graph_ms / split_matmuls_graph_ms
            if dense_graph_ms is not None and split_matmuls_graph_ms is not None
            else None
        ),
        "dense_mlp_ms": dense_mlp_ms,
        "dense_mlp_graph_ms": dense_mlp_graph_ms,
        "dense_mlp_graph_error": dense_mlp_graph_error,
        "split_grouped_mlp_ms": split_grouped_mlp_ms,
        "split_grouped_mlp_graph_ms": split_grouped_mlp_graph_ms,
        "split_grouped_mlp_graph_error": split_grouped_mlp_graph_error,
        "split_grouped_mlp_speedup_vs_dense_mlp": dense_mlp_ms
        / split_grouped_mlp_ms,
        "split_grouped_mlp_graph_speedup_vs_dense_mlp_graph": (
            dense_mlp_graph_ms / split_grouped_mlp_graph_ms
            if dense_mlp_graph_ms is not None
            and split_grouped_mlp_graph_ms is not None
            else None
        ),
        "split_grouped_mlp_fused_act_ms": split_grouped_mlp_fused_act_ms,
        "split_grouped_mlp_fused_act_graph_ms":
        split_grouped_mlp_fused_act_graph_ms,
        "split_grouped_mlp_fused_act_graph_error":
        split_grouped_mlp_fused_act_graph_error,
        "split_grouped_mlp_fused_act_speedup_vs_dense_mlp": (
            dense_mlp_ms / split_grouped_mlp_fused_act_ms
            if split_grouped_mlp_fused_act_ms is not None else None
        ),
        "split_grouped_mlp_fused_act_graph_speedup_vs_dense_mlp_graph": (
            dense_mlp_graph_ms / split_grouped_mlp_fused_act_graph_ms
            if dense_mlp_graph_ms is not None
            and split_grouped_mlp_fused_act_graph_ms is not None
            else None
        ),
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "| strategy | dense frac | dense graph ms | split matmuls graph ms | "
        "split assembled graph ms | split assembled speedup | "
        "dense MLP graph ms | grouped split MLP graph ms | "
        "grouped split MLP speedup | fused-act grouped MLP graph ms | "
        "fused-act grouped speedup | max abs diff |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        dense_graph = row.get("dense_graph_ms")
        lines.append(
            "| "
            f"{row['strategy']} | "
            f"{row['dense_fraction']:.3f} | "
            f"{fmt_ms(dense_graph)} | "
            f"{fmt_ms(row.get('split_matmuls_graph_ms'))} | "
            f"{fmt_ms(row.get('split_index_copy_graph_ms'))} | "
            f"{fmt_speedup(row.get('split_index_copy_graph_ms'), dense_graph)} | "
            f"{fmt_ms(row.get('dense_mlp_graph_ms'))} | "
            f"{fmt_ms(row.get('split_grouped_mlp_graph_ms'))} | "
            f"{fmt_speedup(row.get('split_grouped_mlp_graph_ms'), row.get('dense_mlp_graph_ms'))} | "
            f"{fmt_ms(row.get('split_grouped_mlp_fused_act_graph_ms'))} | "
            f"{fmt_speedup(row.get('split_grouped_mlp_fused_act_graph_ms'), row.get('dense_mlp_graph_ms'))} | "
            f"{row['max_abs_diff']:.6f} |\n"
        )
    path.write_text(
        "# SR24 Gate/Up Paired-Channel Split Microbench\n\n"
        "This measures a standalone fused `gate_up_proj` shape. Dense selected "
        "intermediate channels keep both gate and up rows dense; the remaining "
        "paired rows use 2:4 sparse weights. The assembled path includes the "
        "column `index_copy_` needed to restore the original fused row order. "
        "The grouped MLP path avoids that scatter by keeping the gate/up "
        "intermediate channels grouped and pre-permuting the down_proj input "
        "columns. The fused-act grouped path keeps the same grouped channel "
        "order but uses vLLM's fused SiluAndMul activation.\n\n"
        + "".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dense-fractions", type=parse_float_list,
                        default=parse_float_list("0.125,0.25,0.5"))
    parser.add_argument("--strategies", default="norm,front")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"],
                        default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microbench")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    output_root = args.output_root
    if output_root is None:
        output_root = (
            Path("examples/evaluate/eval-guidellm/results.bak")
            / f"sr24_gateup_channel_split_microbench_{timestamp()}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    strategies = [item.strip() for item in args.strategies.split(",")
                  if item.strip()]
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for dense_fraction in args.dense_fractions:
            row = run_case(
                rows=args.rows,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                dense_fraction=dense_fraction,
                strategy=strategy,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    (output_root / "raw_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_root / "summary.md", rows)
    print(f"wrote {output_root.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
