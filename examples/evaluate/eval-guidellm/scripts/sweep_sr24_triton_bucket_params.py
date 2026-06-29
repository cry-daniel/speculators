#!/usr/bin/env python3
"""Sweep SR24 Triton bucket dense-GEMM parameters.

This is a narrow diagnostic for the current mixed residual path. It reuses the
component microbench and varies only the Triton bucket GEMM tile sizes, so a
bad result here means the existing Python/Triton bucket kernel is not a likely
serving speed path.

Example:
  conda run -n spec python scripts/sweep_sr24_triton_bucket_params.py \
    --shape 512,28672,4096 --residual-fractions 0.0625,0.125 \
    --bucket-size 64 --warmup 10 --repeats 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from profile_speclink_sr24_component_breakdown import (
    component_case,
    parse_float_list,
    parse_shape,
)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("empty int list")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("all values must be positive")
    return out


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SR24 Triton bucket sweep")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    output_root = args.output_root or (
        Path("results.bak") / f"sr24_triton_bucket_param_sweep_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM"] = "1"
    rows: list[dict[str, Any]] = []
    for shape_rows, out_features, in_features in args.shape:
        for residual_fraction in args.residual_fractions:
            for block_m in args.block_m:
                for block_n in args.block_n:
                    for block_k in args.block_k:
                        os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_M"] = (
                            str(block_m)
                        )
                        os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_N"] = (
                            str(block_n)
                        )
                        os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_K"] = (
                            str(block_k)
                        )
                        result = component_case(
                            rows=shape_rows,
                            out_features=out_features,
                            in_features=in_features,
                            residual_fraction=residual_fraction,
                            bucket_size=args.bucket_size,
                            dtype=dtype,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                        result["triton_block_m"] = block_m
                        result["triton_block_n"] = block_n
                        result["triton_block_k"] = block_k
                        dense_graph = result.get("dense_graph_ms")
                        triton_graph = result.get(
                            "bucket_triton_dense_gemm_scatter_graph_ms"
                        )
                        if dense_graph and triton_graph:
                            result["triton_bucket_vs_dense_graph"] = (
                                float(triton_graph) / float(dense_graph)
                            )
                        mixed_graph = result.get("bucket_delta_inplace_graph_ms")
                        if mixed_graph and triton_graph:
                            result["triton_bucket_vs_bucket_delta_graph"] = (
                                float(triton_graph) / float(mixed_graph)
                            )
                        rows.append(result)
                        print(
                            "shape="
                            f"{shape_rows}x{out_features}x{in_features} "
                            f"frac={residual_fraction:g} "
                            f"bm/bn/bk={block_m}/{block_n}/{block_k} "
                            "triton_graph="
                            f"{fmt(triton_graph)}ms"
                        )

    csv_path = output_root / "sweep.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best = sorted(
        rows,
        key=lambda row: (
            float(row.get("bucket_triton_dense_gemm_scatter_graph_ms") or 1e30),
            int(row["triton_block_m"]),
            int(row["triton_block_n"]),
            int(row["triton_block_k"]),
        ),
    )
    summary = {
        "rows": len(rows),
        "best": best[: min(10, len(best))],
        "csv": str(csv_path.resolve()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SR24 Triton Bucket Parameter Sweep\n\n")
        handle.write(f"- rows: {len(rows)}\n")
        handle.write(f"- csv: `{csv_path.resolve()}`\n\n")
        handle.write("| shape | residual frac | block M/N/K | dense graph ms | bucket delta graph ms | triton graph ms | triton/dense | triton/bucket-delta |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in best[: min(20, len(best))]:
            shape = f"{row['rows']}x{row['out_features']}x{row['in_features']}"
            block = (
                f"{row['triton_block_m']}/"
                f"{row['triton_block_n']}/"
                f"{row['triton_block_k']}"
            )
            handle.write(
                "| "
                f"{shape} | {row['residual_fraction']} | {block} | "
                f"{fmt(row.get('dense_graph_ms'))} | "
                f"{fmt(row.get('bucket_delta_inplace_graph_ms'))} | "
                f"{fmt(row.get('bucket_triton_dense_gemm_scatter_graph_ms'))} | "
                f"{fmt(row.get('triton_bucket_vs_dense_graph'))} | "
                f"{fmt(row.get('triton_bucket_vs_bucket_delta_graph'))} |\n"
            )
    print(output_root.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument(
        "--residual-fractions",
        type=parse_float_list,
        default=[0.0625, 0.125, 0.25],
    )
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--block-m", type=parse_int_list, default=[8, 16, 32])
    parser.add_argument("--block-n", type=parse_int_list, default=[32, 64, 128])
    parser.add_argument("--block-k", type=parse_int_list, default=[64, 128, 256])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(512, 28672, 4096), (512, 4096, 14336)]
    run(args)


if __name__ == "__main__":
    main()
