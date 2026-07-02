#!/usr/bin/env python3
"""Profile the SR24 fixed-block mixed down_proj operator shape.

This isolates the current down-only 8pp candidate:

  activation[batch, K + 1, intermediate]
  dense rows = first `prefix` draft rows plus verifier bonus row
  sparse rows = remaining draft rows

Unlike the full MLP probe, this measures only down_proj. It answers whether the
down-only route has enough local operator headroom before running expensive
vLLM/GuideLLM serving experiments.
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


def speedup(numerator: Any, denominator: Any) -> float | None:
    if numerator in (None, "") or denominator in (None, ""):
        return None
    denom = float(denominator)
    if denom <= 0:
        return None
    return float(numerator) / denom


def run_case(
    *,
    batch_size: int,
    coalesce_factor: int,
    k: int,
    prefix: int,
    min_dense_rows: int,
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
    protected_dense_width = prefix + 1
    protected_dense_count = effective_batch_size * protected_dense_width
    promoted_width = 0
    if min_dense_rows > protected_dense_count:
        need_rows = min_dense_rows - protected_dense_count
        promoted_width = min(
            k - prefix,
            (need_rows + effective_batch_size - 1) // effective_batch_size,
        )
    dense_rows_per_request = protected_dense_width + promoted_width
    base_rows_per_request = max(0, k - prefix - promoted_width)
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

    x = torch.randn(rows, intermediate_size, device="cuda", dtype=dtype)
    down_weight = torch.randn(
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=dtype,
    )
    down_base = to_sparse_semi_structured(make_base_24(down_weight))
    x_blocks = x.reshape(effective_batch_size, width, intermediate_size)
    x_single = x[:single_block_rows]
    dense_buffer = torch.zeros(
        dense_capacity, intermediate_size, device="cuda", dtype=dtype
    )
    base_buffer = torch.zeros(
        base_capacity, intermediate_size, device="cuda", dtype=dtype
    )

    def dense_down() -> torch.Tensor:
        return F.linear(x, down_weight)

    def base_down() -> torch.Tensor:
        return F.linear(x, down_base)

    def single_dense_down() -> torch.Tensor:
        return F.linear(x_single, down_weight)

    def dense_input_from_blocks() -> torch.Tensor:
        dense_parts: list[torch.Tensor] = []
        if prefix > 0:
            dense_parts.append(x_blocks[:, :prefix, :].reshape(-1, intermediate_size))
        if promoted_width > 0:
            dense_parts.append(
                x_blocks[:, prefix:prefix + promoted_width, :].reshape(
                    -1, intermediate_size
                )
            )
        dense_parts.append(
            x_blocks[:, valid_width:valid_width + 1, :].reshape(
                -1, intermediate_size
            )
        )
        return dense_parts[0] if len(dense_parts) == 1 else torch.cat(
            dense_parts, dim=0
        )

    def base_input_from_blocks() -> torch.Tensor:
        if base_rows_per_request <= 0:
            return torch.empty(0, intermediate_size, device="cuda", dtype=dtype)
        base_start = prefix + promoted_width
        return x_blocks[:, base_start:valid_width, :].reshape(-1, intermediate_size)

    dense_stream = torch.cuda.Stream()
    base_stream = torch.cuda.Stream()

    def assemble(dense_out: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        output_blocks = torch.empty(
            effective_batch_size,
            width,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        dense_prefix_rows = effective_batch_size * prefix
        dense_promoted_rows = effective_batch_size * promoted_width
        dense_bonus_start = dense_prefix_rows + dense_promoted_rows
        if prefix > 0:
            output_blocks[:, :prefix, :].copy_(
                dense_out[:dense_prefix_rows].reshape(
                    effective_batch_size, prefix, hidden_size
                )
            )
        if promoted_width > 0:
            output_blocks[:, prefix:prefix + promoted_width, :].copy_(
                dense_out[dense_prefix_rows:dense_bonus_start].reshape(
                    effective_batch_size, promoted_width, hidden_size
                )
            )
        output_blocks[:, valid_width, :].copy_(
            dense_out[dense_bonus_start:dense_count].reshape(
                effective_batch_size, hidden_size
            )
        )
        if base_count > 0:
            output_blocks[:, prefix + promoted_width:valid_width, :].copy_(
                base_out[:base_count].reshape(
                    effective_batch_size, base_rows_per_request, hidden_size
                )
            )
        return output_blocks.reshape(rows, hidden_size).contiguous()

    def packed_down() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        dense_out = F.linear(dense_input, down_weight)
        if base_count > 0:
            base_out = F.linear(base_input, down_base)
        else:
            base_out = torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return assemble(dense_out, base_out)

    def packed_parallel_down() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        launch_stream = torch.cuda.current_stream()

        dense_stream.wait_stream(launch_stream)
        with torch.cuda.stream(dense_stream):
            dense_out = F.linear(dense_input, down_weight)

        if base_count > 0:
            base_stream.wait_stream(launch_stream)
            with torch.cuda.stream(base_stream):
                base_out = F.linear(base_input, down_base)
            launch_stream.wait_stream(base_stream)
        else:
            base_out = torch.empty(0, hidden_size, device="cuda", dtype=dtype)

        launch_stream.wait_stream(dense_stream)
        return assemble(dense_out, base_out)

    def packed_no_final_assemble() -> tuple[torch.Tensor, torch.Tensor]:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        dense_out = F.linear(dense_input, down_weight)
        if base_count > 0:
            base_out = F.linear(base_input, down_base)
        else:
            base_out = torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return dense_out, base_out

    def packed_parallel_no_final_assemble() -> tuple[torch.Tensor, torch.Tensor]:
        dense_input = dense_input_from_blocks()
        base_input = base_input_from_blocks()
        launch_stream = torch.cuda.current_stream()

        dense_stream.wait_stream(launch_stream)
        with torch.cuda.stream(dense_stream):
            dense_out = F.linear(dense_input, down_weight)

        if base_count > 0:
            base_stream.wait_stream(launch_stream)
            with torch.cuda.stream(base_stream):
                base_out = F.linear(base_input, down_base)
            launch_stream.wait_stream(base_stream)
        else:
            base_out = torch.empty(0, hidden_size, device="cuda", dtype=dtype)

        launch_stream.wait_stream(dense_stream)
        return dense_out, base_out

    def packed_capacity_padded_down() -> torch.Tensor:
        dense_input = dense_input_from_blocks()
        dense_buffer[:dense_count].copy_(dense_input)
        dense_out = F.linear(dense_buffer, down_weight)
        if base_count > 0:
            base_input = base_input_from_blocks()
            base_buffer[:base_count].copy_(base_input)
            base_out = F.linear(base_buffer, down_base)
        else:
            base_out = torch.empty(0, hidden_size, device="cuda", dtype=dtype)
        return assemble(dense_out[:dense_count], base_out[:base_count])

    dense_norm = float(dense_down().norm().item())
    base_norm = float(base_down().norm().item())
    packed_norm = float(packed_down().norm().item())
    parallel_norm = float(packed_parallel_down().norm().item())
    padded_norm = float(packed_capacity_padded_down().norm().item())
    torch.cuda.synchronize()

    dense_graph_ms, dense_graph_error = time_graph(
        dense_down, warmup=warmup, repeats=repeats
    )
    single_dense_graph_ms, single_dense_graph_error = time_graph(
        single_dense_down, warmup=warmup, repeats=repeats
    )
    base_graph_ms, base_graph_error = time_graph(
        base_down, warmup=warmup, repeats=repeats
    )
    packed_graph_ms, packed_graph_error = time_graph(
        packed_down, warmup=warmup, repeats=repeats
    )
    no_assemble_graph_ms, no_assemble_graph_error = time_graph(
        packed_no_final_assemble, warmup=warmup, repeats=repeats
    )
    parallel_graph_ms, parallel_graph_error = time_graph(
        packed_parallel_down, warmup=warmup, repeats=repeats
    )
    parallel_no_assemble_graph_ms, parallel_no_assemble_graph_error = time_graph(
        packed_parallel_no_final_assemble, warmup=warmup, repeats=repeats
    )
    padded_graph_ms, padded_graph_error = time_graph(
        packed_capacity_padded_down, warmup=warmup, repeats=repeats
    )

    dense_ms = time_call(dense_down, warmup=warmup, repeats=repeats)
    single_dense_ms = time_call(single_dense_down, warmup=warmup, repeats=repeats)
    base_ms = time_call(base_down, warmup=warmup, repeats=repeats)
    packed_ms = time_call(packed_down, warmup=warmup, repeats=repeats)
    no_assemble_ms = time_call(
        packed_no_final_assemble, warmup=warmup, repeats=repeats
    )
    parallel_ms = time_call(packed_parallel_down, warmup=warmup, repeats=repeats)
    parallel_no_assemble_ms = time_call(
        packed_parallel_no_final_assemble, warmup=warmup, repeats=repeats
    )
    padded_ms = time_call(
        packed_capacity_padded_down, warmup=warmup, repeats=repeats
    )

    serial_dense_graph = (
        single_dense_graph_ms * coalesce_factor
        if single_dense_graph_ms is not None
        else None
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
        "promoted_width": promoted_width,
        "min_dense_rows": min_dense_rows,
        "dense_rows_per_request": dense_rows_per_request,
        "base_rows_per_request": base_rows_per_request,
        "single_dense_rows": single_dense_count,
        "single_base_rows": single_base_count,
        "dense_rows": dense_count,
        "base_rows": base_count,
        "capacity_multiple": capacity_multiple,
        "dense_capacity": dense_capacity,
        "base_capacity": base_capacity,
        "dense_capacity_fill": dense_count / dense_capacity,
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
        "packed_capacity_padded_ms": padded_ms,
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "single_dense_graph_ms": single_dense_graph_ms,
        "single_dense_graph_error": single_dense_graph_error,
        "serial_dense_reference_graph_ms": serial_dense_graph,
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
        "packed_capacity_padded_graph_ms": padded_graph_ms,
        "packed_capacity_padded_graph_error": padded_graph_error,
        "base_vs_dense_graph_ratio": speedup(base_graph_ms, dense_graph_ms),
        "packed_vs_dense_graph_ratio": speedup(packed_graph_ms, dense_graph_ms),
        "packed_no_assemble_vs_dense_graph_ratio": speedup(
            no_assemble_graph_ms,
            dense_graph_ms,
        ),
        "packed_parallel_vs_dense_graph_ratio": speedup(
            parallel_graph_ms,
            dense_graph_ms,
        ),
        "packed_parallel_no_assemble_vs_dense_graph_ratio": speedup(
            parallel_no_assemble_graph_ms,
            dense_graph_ms,
        ),
        "packed_capacity_padded_vs_dense_graph_ratio": speedup(
            padded_graph_ms,
            dense_graph_ms,
        ),
        "packed_vs_serial_dense_graph_speedup": speedup(
            serial_dense_graph,
            packed_graph_ms,
        ),
        "parallel_vs_serial_dense_graph_speedup": speedup(
            serial_dense_graph,
            parallel_graph_ms,
        ),
        "coalesced_dense_vs_serial_dense_graph_speedup": speedup(
            serial_dense_graph,
            dense_graph_ms,
        ),
        "dense_output_norm": dense_norm,
        "base_output_norm": base_norm,
        "packed_output_norm": packed_norm,
        "packed_parallel_output_norm": parallel_norm,
        "padded_output_norm": padded_norm,
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
        f.write("# SR24 Packed DownProj Microbenchmark\n\n")
        f.write(
            "Rows are shaped as `[batch, K + 1, intermediate]`. `packed` routes "
            "the first `prefix` draft rows plus the verifier bonus row through "
            "dense down_proj and the remaining draft rows through 2:4 sparse "
            "down_proj. `promoted_width` means additional low-priority draft "
            "positions were intentionally run dense to fill a small dense branch.\n\n"
        )
        f.write(
            "| bs | coalesce | effective bs | K | prefix | promoted | rows | "
            "dense rows | base rows | dense graph ms | serial dense graph ms | "
            "base graph ms | packed graph ms | no-assemble graph ms | "
            "parallel graph ms | parallel no-asm graph ms | padded graph ms | "
            "packed/dense | parallel/dense | padded/dense | packed vs serial | "
            "parallel vs serial |\n"
        )
        f.write(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            dense_graph = row.get("dense_graph_ms") or row.get("dense_ms")
            serial_dense_graph = (
                row.get("serial_dense_reference_graph_ms")
                or row.get("serial_dense_reference_ms")
            )
            f.write(
                f"| {row['batch_size']} | {row['coalesce_factor']} | "
                f"{row['effective_batch_size']} | {row['k']} | "
                f"{row['prefix']} | {row['promoted_width']} | {row['rows']} | "
                f"{row['dense_rows']} | {row['base_rows']} | "
                f"{fmt_ms(row.get('dense_graph_ms')) or row.get('dense_graph_error')} | "
                f"{fmt_ms(serial_dense_graph)} | "
                f"{fmt_ms(row.get('base_graph_ms')) or row.get('base_graph_error')} | "
                f"{fmt_ms(row.get('packed_graph_ms')) or row.get('packed_graph_error')} | "
                f"{fmt_ms(row.get('packed_no_assemble_graph_ms')) or row.get('packed_no_assemble_graph_error')} | "
                f"{fmt_ms(row.get('packed_parallel_graph_ms')) or row.get('packed_parallel_graph_error')} | "
                f"{fmt_ms(row.get('packed_parallel_no_assemble_graph_ms')) or row.get('packed_parallel_no_assemble_graph_error')} | "
                f"{fmt_ms(row.get('packed_capacity_padded_graph_ms')) or row.get('packed_capacity_padded_graph_error')} | "
                f"{fmt_ratio(row.get('packed_graph_ms'), dense_graph)} | "
                f"{fmt_ratio(row.get('packed_parallel_graph_ms'), dense_graph)} | "
                f"{fmt_ratio(row.get('packed_capacity_padded_graph_ms'), dense_graph)} | "
                f"{fmt_ratio(serial_dense_graph, row.get('packed_graph_ms'))} | "
                f"{fmt_ratio(serial_dense_graph, row.get('packed_parallel_graph_ms'))} |\n"
            )

        f.write("\n## Read\n\n")
        f.write(
            "- `packed/dense < 0.80x` is the rough local target for a live "
            "operator to have a chance at `>=1.2x` after serving overhead.\n"
        )
        f.write(
            "- `coalesce > 1` simulates grouping independent verifier blocks with "
            "the same weights. It is an optimistic upper bound for a future "
            "grouped verifier queue, not a serving claim.\n"
        )
        f.write(
            "- `parallel` launches dense-important and sparse-base branches on "
            "separate CUDA streams. If it loses here, Python stream overlap is "
            "not the right live optimization.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--coalesce-factors", default="1")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--prefixes", default="2,4")
    parser.add_argument(
        "--min-dense-rows",
        type=int,
        default=0,
        help=(
            "Promote additional draft positions to dense until this dense-row "
            "count is reached. This models useful-row fill when the important "
            "set is too small."
        ),
    )
    parser.add_argument("--capacity-multiple", type=int, default=128)
    parser.add_argument("--min-dense-capacity", type=int, default=0)
    parser.add_argument("--min-base-capacity", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results.bak") / f"sr24_packed_downproj_{timestamp()}",
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
                        min_dense_rows=args.min_dense_rows,
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
