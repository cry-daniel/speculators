#!/usr/bin/env python3
"""Profile the SR24 compressed_dense residual kernel.

This microbenchmark isolates the all_corrected/compressed_dense residual path:
it compares the custom packed residual Triton matmul against materializing the
dense residual weight and running a normal torch GEMM. It does not launch vLLM.

Example:
  cd examples/evaluate/eval-guidellm
  conda run -n spec python scripts/profile_speclink_sr24_compressed_residual_kernel.py \
    --shape 512,28672,4096 --shape 512,4096,14336 \
    --block-n 16,32,64,128 --block-g 16,32,64,128
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
import triton

from vllm.speclink_sr24 import (
    _compressed_residual_matmul_kernel,
    _compute_keep_mask_24,
    _pack_keep_mask,
    _unpacked_group_bytes_to_keep,
    _expand_mask_bytes,
)


EVAL_ROOT = Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be ROWS,OUT,IN")
    return parts[0], parts[1], parts[2]


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    return values


def time_cuda(fn: Callable[[], Any], *, warmup: int, repeats: int) -> float:
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


def time_graph(fn: Callable[[], Any], *, warmup: int, repeats: int) -> tuple[float | None, str]:
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


def make_case(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    dense_weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
    keep = _compute_keep_mask_24(dense_weight).to(device="cuda")
    groups = in_features // 4
    residual_values = (
        dense_weight.view(out_features, groups, 4)[~keep].contiguous()
    )
    mask_bytes = _pack_keep_mask(keep).to(device="cuda", dtype=torch.uint8)
    group_bytes = _expand_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=groups,
        device=torch.device("cuda"),
    )
    keep_roundtrip = _unpacked_group_bytes_to_keep(
        group_bytes,
        device=torch.device("cuda"),
    )
    residual_weight = torch.zeros_like(dense_weight)
    residual_weight.view(out_features, groups, 4).masked_scatter_(
        torch.logical_not(keep_roundtrip),
        residual_values,
    )
    torch.cuda.synchronize()
    return {
        "x": x,
        "dense_weight": dense_weight,
        "keep": keep,
        "residual_values": residual_values,
        "mask_bytes": mask_bytes,
        "residual_weight": residual_weight,
    }


def run_triton_kernel(
    *,
    x: torch.Tensor,
    residual_values: torch.Tensor,
    mask_bytes: torch.Tensor,
    out_features: int,
    block_m: int,
    block_n: int,
    block_g: int,
) -> torch.Tensor:
    rows = int(x.shape[0])
    in_features = int(x.shape[1])
    groups = in_features // 4
    out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _compressed_residual_matmul_kernel[grid](
        x,
        residual_values,
        mask_bytes,
        out,
        rows,
        out_features,
        groups,
        int(x.stride(0)),
        int(x.stride(1)),
        int(mask_bytes.stride(0)),
        int(out.stride(0)),
        int(out.stride(1)),
        block_m,
        block_n,
        block_g,
        num_warps=4,
        num_stages=3,
    )
    return out


def profile_shape(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    block_m_values: list[int],
    block_n_values: list[int],
    block_g_values: list[int],
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    case = make_case(
        rows=rows,
        out_features=out_features,
        in_features=in_features,
        dtype=dtype,
    )
    x = case["x"]
    residual_values = case["residual_values"]
    mask_bytes = case["mask_bytes"]
    residual_weight = case["residual_weight"]

    def dense_residual() -> torch.Tensor:
        return F.linear(x, residual_weight, bias=None)

    dense_ms = time_cuda(dense_residual, warmup=warmup, repeats=repeats)
    dense_graph_ms, dense_graph_error = time_graph(
        dense_residual,
        warmup=warmup,
        repeats=repeats,
    )
    reference = dense_residual()
    torch.cuda.synchronize()

    out: list[dict[str, Any]] = []
    for block_m in block_m_values:
        for block_n in block_n_values:
            for block_g in block_g_values:
                def triton_case() -> torch.Tensor:
                    return run_triton_kernel(
                        x=x,
                        residual_values=residual_values,
                        mask_bytes=mask_bytes,
                        out_features=out_features,
                        block_m=block_m,
                        block_n=block_n,
                        block_g=block_g,
                    )

                error = ""
                try:
                    triton_ms = time_cuda(
                        triton_case,
                        warmup=warmup,
                        repeats=repeats,
                    )
                    triton_graph_ms, triton_graph_error = time_graph(
                        triton_case,
                        warmup=warmup,
                        repeats=repeats,
                    )
                    candidate = triton_case()
                    torch.cuda.synchronize()
                    max_abs_diff = float((candidate - reference).abs().max().item())
                except Exception as exc:  # noqa: BLE001
                    torch.cuda.synchronize()
                    triton_ms = None
                    triton_graph_ms = None
                    triton_graph_error = ""
                    max_abs_diff = None
                    error = f"{type(exc).__name__}: {exc}"
                dense_ref = dense_graph_ms if dense_graph_ms is not None else dense_ms
                triton_ref = (
                    triton_graph_ms if triton_graph_ms is not None else triton_ms
                )
                out.append(
                    {
                        "rows": rows,
                        "out_features": out_features,
                        "in_features": in_features,
                        "dtype": str(dtype).replace("torch.", ""),
                        "block_m": block_m,
                        "block_n": block_n,
                        "block_g": block_g,
                        "dense_residual_ms": dense_ms,
                        "dense_residual_graph_ms": dense_graph_ms,
                        "dense_residual_graph_error": dense_graph_error,
                        "triton_ms": triton_ms,
                        "triton_graph_ms": triton_graph_ms,
                        "triton_graph_error": triton_graph_error,
                        "triton_vs_dense_graph": (
                            triton_ref / dense_ref
                            if triton_ref is not None and dense_ref
                            else None
                        ),
                        "max_abs_diff": max_abs_diff,
                        "error": error,
                    }
                )
    return out


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        row
        for row in rows
        if row.get("triton_vs_dense_graph") is not None and not row.get("error")
    ]
    best_by_shape: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in valid_rows:
        key = (
            int(row["rows"]),
            int(row["out_features"]),
            int(row["in_features"]),
        )
        current = best_by_shape.get(key)
        if current is None or float(row["triton_vs_dense_graph"]) < float(
            current["triton_vs_dense_graph"]
        ):
            best_by_shape[key] = row

    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SR24 Compressed Residual Kernel Profile\n\n")
        handle.write(
            "Lower `triton/dense` is better. Dense means materialized residual "
            "weight plus torch GEMM; Triton means packed residual values plus "
            "the custom `_compressed_residual_matmul_kernel`.\n\n"
        )
        if best_by_shape:
            slower_shapes = sum(
                1
                for row in best_by_shape.values()
                if float(row["triton_vs_dense_graph"]) >= 1.0
            )
            large_diff_shapes = sum(
                1
                for row in best_by_shape.values()
                if float(row["max_abs_diff"]) > 0.05
            )
            if slower_shapes or large_diff_shapes:
                handle.write(
                    "Diagnostic read: keep this Triton residual-only path "
                    "disabled for serving unless both `triton/dense < 1` and "
                    "`max abs diff` are acceptable for the target dtype. It "
                    f"is slower on {slower_shapes}/{len(best_by_shape)} best "
                    f"shape rows and has `max abs diff > 0.05` on "
                    f"{large_diff_shapes}/{len(best_by_shape)} best shape "
                    "rows.\n\n"
                )
        handle.write(
            "| rows | out | in | best block_m | best block_n | best block_g | "
            "dense graph ms | triton graph ms | triton/dense | max abs diff |\n"
        )
        handle.write(
            "|-----:|----:|---:|-------------:|-------------:|-------------:|"
            "---------------:|----------------:|-------------:|-------------:|\n"
        )
        for key in sorted(best_by_shape):
            row = best_by_shape[key]
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} "
                f"| {row['block_m']} | {row['block_n']} | {row['block_g']} "
                f"| {float(row['dense_residual_graph_ms']):.4f} "
                f"| {float(row['triton_graph_ms']):.4f} "
                f"| {float(row['triton_vs_dense_graph']):.3f} "
                f"| {float(row['max_abs_diff']):.6f} |\n"
            )
        handle.write("\nFull sweep CSV:\n\n")
        handle.write(f"`{(output_root / 'summary.csv').resolve()}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EVAL_ROOT
        / "results.bak"
        / f"sr24_compressed_residual_kernel_profile_{timestamp()}",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=[],
        help="ROWS,OUT,IN. May be repeated.",
    )
    parser.add_argument("--block-m", type=parse_int_csv, default="8,16,32")
    parser.add_argument("--block-n", type=parse_int_csv, default="16,32,64,128")
    parser.add_argument("--block-g", type=parse_int_csv, default="16,32,64,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    shapes = args.shape or [(512, 28672, 4096), (512, 4096, 14336)]
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        rows.extend(
            profile_shape(
                rows=shape[0],
                out_features=shape[1],
                in_features=shape[2],
                dtype=dtype,
                block_m_values=args.block_m,
                block_n_values=args.block_n,
                block_g_values=args.block_g,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
        torch.cuda.empty_cache()
    write_outputs(args.output_root.resolve(), rows)
    print((args.output_root / "summary.md").resolve())
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
