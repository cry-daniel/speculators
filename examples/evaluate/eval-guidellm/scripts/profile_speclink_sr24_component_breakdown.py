#!/usr/bin/env python3
"""Component-level SR24 GPU microbreakdown.

This diagnostic is intentionally narrower than a full vLLM serving run.  It
answers the first question before another end-to-end sweep: for representative
Llama MLP Linear shapes, where does the current 2:4 + residual correction path
spend time?

Example:
  conda run -n spec python scripts/profile_speclink_sr24_component_breakdown.py \
    --shape 512,28672,4096 --shape 512,4096,14336 \
    --residual-fractions 0.125,0.25,0.5,0.875 \
    --bucket-size 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

try:
    from vllm.speclink_sr24 import (
        _compressed_residual_triton_linear as sr24_compressed_residual_triton_linear,
    )
    from vllm.speclink_sr24 import (
        _pack_keep_mask as sr24_pack_keep_mask,
    )
    from vllm.speclink_sr24 import (
        _triton_bucket_dense_gemm_scatter_inplace as sr24_triton_bucket_dense_gemm_scatter_inplace,
    )
except Exception:  # noqa: BLE001
    sr24_compressed_residual_triton_linear = None
    sr24_pack_keep_mask = None
    sr24_triton_bucket_dense_gemm_scatter_inplace = None


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be ROWS,OUT,IN")
    return parts[0], parts[1], parts[2]


def parse_float_list(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("empty float list")
    for item in out:
        if item < 0.0 or item > 1.0:
            raise argparse.ArgumentTypeError("fractions must be in [0, 1]")
    return out


def make_base_24_and_keep(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    if in_features % 4:
        raise ValueError("in_features must be divisible by 4")
    grouped = weight.view(out_features, in_features // 4, 4)
    keep_idx = grouped.abs().topk(2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    base = torch.zeros_like(grouped)
    base[keep] = grouped[keep]
    return base.view_as(weight).contiguous(), keep


def make_base_24(weight: torch.Tensor) -> torch.Tensor:
    base, _ = make_base_24_and_keep(weight)
    return base


def time_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> float:
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


def time_cpu(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    sync: bool = False,
) -> float:
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


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
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def fmt_ratio(value: Any, denom: Any) -> str:
    if value is None or value == "" or denom is None or denom == "":
        return ""
    denom_f = float(denom)
    if denom_f == 0.0:
        return ""
    return f"{float(value) / denom_f:.2f}x"


def component_case(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    residual_fraction: float,
    bucket_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    residual_rows_count = min(rows, max(0, int(round(rows * residual_fraction))))
    bucket_candidate_rows = min(rows, max(0, int(bucket_size)))
    active_bucket_rows = min(residual_rows_count, bucket_candidate_rows)

    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    dense_weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
    base_weight, keep = make_base_24_and_keep(dense_weight)
    residual_weight = (dense_weight - base_weight).contiguous()
    base_sparse = to_sparse_semi_structured(base_weight)
    residual_sparse = to_sparse_semi_structured(residual_weight)
    groups = in_features // 4
    residual_values = dense_weight.view(out_features, groups, 4)[~keep].contiguous()
    compressed_module = None
    if sr24_pack_keep_mask is not None:
        compressed_module = SimpleNamespace(
            _speclink_sr24_weight_shape=(out_features, in_features),
            _speclink_sr24_usable_in=in_features,
            _speclink_sr24_weight_dtype=dtype,
            _speclink_sr24_residual_values=residual_values,
            _speclink_sr24_base_mask_bytes=sr24_pack_keep_mask(keep).to(
                device="cuda", non_blocking=True
            ),
        )
        os.environ["SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON"] = "1"

    all_rows = torch.arange(rows, device="cuda", dtype=torch.long)
    residual_rows = all_rows[:residual_rows_count].contiguous()
    base_rows = all_rows[residual_rows_count:].contiguous()
    bucket_rows = all_rows[:bucket_candidate_rows].contiguous()
    bucket_values_bool = torch.zeros(
        bucket_candidate_rows,
        device="cuda",
        dtype=torch.bool,
    )
    if active_bucket_rows > 0:
        bucket_values_bool[:active_bucket_rows] = True
    priority = torch.empty(rows, device="cuda", dtype=torch.float32)
    priority[:residual_rows_count].uniform_(0.5, 1.0)
    priority[residual_rows_count:].uniform_(0.0, 0.5)
    scores = torch.rand(rows, device="cuda", dtype=torch.float32)
    score_threshold = torch.tensor(0.8, device="cuda", dtype=torch.float32)
    request_starts = list(range(0, rows, 9))
    request_counts = [min(9, rows - start) for start in request_starts]
    torch.cuda.synchronize()

    def dense_full() -> torch.Tensor:
        return F.linear(x, dense_weight)

    def base_sparse_full() -> torch.Tensor:
        return F.linear(x, base_sparse)

    def mask_build_cpu_loop() -> torch.Tensor:
        mask = torch.empty(rows, dtype=torch.bool, device="cuda")
        mask.fill_(False)
        left = residual_rows_count
        for start, count in zip(request_starts, request_counts):
            if left <= 0:
                break
            take = min(count, left)
            mask[start:start + take].fill_(True)
            left -= take
        return mask

    def mask_build_vectorized_cuda() -> torch.Tensor:
        return scores < score_threshold

    def bucket_topk() -> torch.Tensor:
        if bucket_candidate_rows <= 0:
            return torch.empty(0, device="cuda", dtype=torch.long)
        return torch.topk(priority, k=bucket_candidate_rows, largest=True).indices

    dense_output = dense_full()
    base_output = base_sparse_full()
    residual_input = (
        x.index_select(0, residual_rows)
        if residual_rows_count > 0
        else torch.empty(0, in_features, device="cuda", dtype=dtype)
    )
    residual_dense_output = (
        F.linear(residual_input, dense_weight)
        if residual_rows_count > 0
        else torch.empty(0, out_features, device="cuda", dtype=dtype)
    )
    base_selected = (
        base_output.index_select(0, residual_rows)
        if residual_rows_count > 0
        else torch.empty(0, out_features, device="cuda", dtype=dtype)
    )
    delta = residual_dense_output - base_selected
    torch.cuda.synchronize()

    def gather_residual_input() -> torch.Tensor:
        return x.index_select(0, residual_rows)

    def residual_dense_gemm() -> torch.Tensor:
        return F.linear(residual_input, dense_weight)

    def residual_delta_dense_gemm() -> torch.Tensor:
        return F.linear(residual_input, residual_weight)

    def residual_delta_sparse_gemm() -> torch.Tensor:
        return F.linear(residual_input, residual_sparse)

    def gather_base_output() -> torch.Tensor:
        return base_output.index_select(0, residual_rows)

    def delta_compute() -> torch.Tensor:
        return residual_dense_output - base_selected

    def output_clone() -> torch.Tensor:
        return base_output.clone()

    def scatter_index_add() -> torch.Tensor:
        out = base_output.clone()
        out.index_add_(0, residual_rows, delta)
        return out

    def scatter_index_copy() -> torch.Tensor:
        out = base_output.clone()
        out.index_copy_(0, residual_rows, residual_dense_output)
        return out

    def bucket_delta_inplace() -> torch.Tensor:
        out = F.linear(x, base_sparse)
        dense_input = x.index_select(0, residual_rows)
        dense_rows = F.linear(dense_input, dense_weight)
        base_rows_selected = out.index_select(0, residual_rows)
        out.index_add_(0, residual_rows, dense_rows - base_rows_selected)
        return out

    def bucket_dense_copy_inplace() -> torch.Tensor:
        out = F.linear(x, base_sparse)
        if bucket_candidate_rows <= 0:
            return out
        # Conservative serving variant: overwrite every selected bucket row
        # with exact dense output. Rows with bucket_values=False are corrected
        # too, which is accuracy-safe and avoids delta/index_add overhead.
        dense_input = x.index_select(0, bucket_rows)
        dense_rows = F.linear(dense_input, dense_weight)
        out.index_copy_(0, bucket_rows, dense_rows)
        return out

    def compressed_delta_dense_inplace() -> torch.Tensor:
        out = F.linear(x, base_sparse)
        dense_input = x.index_select(0, residual_rows)
        residual_rows_output = F.linear(dense_input, residual_weight)
        out.index_add_(0, residual_rows, residual_rows_output)
        return out

    def compressed_delta_sparse_inplace() -> torch.Tensor:
        out = F.linear(x, base_sparse)
        dense_input = x.index_select(0, residual_rows)
        residual_rows_output = F.linear(dense_input, residual_sparse)
        out.index_add_(0, residual_rows, residual_rows_output)
        return out

    def compressed_delta_triton_inplace() -> torch.Tensor:
        if (
            sr24_compressed_residual_triton_linear is None
            or compressed_module is None
        ):
            raise RuntimeError("SR24 compressed residual Triton path unavailable")
        out = F.linear(x, base_sparse)
        dense_input = x.index_select(0, residual_rows)
        residual_rows_output = sr24_compressed_residual_triton_linear(
            compressed_module,
            dense_input,
            dtype=dtype,
            device=x.device,
        )
        if residual_rows_output is None:
            raise RuntimeError("SR24 compressed residual Triton returned None")
        out.index_add_(0, residual_rows, residual_rows_output)
        return out

    def bucket_triton_dense_gemm_scatter() -> torch.Tensor:
        out = F.linear(x, base_sparse)
        if (
            sr24_triton_bucket_dense_gemm_scatter_inplace is not None
            and bucket_candidate_rows > 0
        ):
            ok = sr24_triton_bucket_dense_gemm_scatter_inplace(
                x,
                dense_weight,
                bucket_rows,
                bucket_values_bool,
                out,
                None,
            )
            if not ok:
                raise RuntimeError("SR24 Triton bucket dense GEMM returned False")
        return out

    def routed_split() -> torch.Tensor:
        out = torch.empty((rows, out_features), device="cuda", dtype=dtype)
        if residual_rows_count > 0:
            dense_input = x.index_select(0, residual_rows)
            dense_rows = F.linear(dense_input, dense_weight)
            out.index_copy_(0, residual_rows, dense_rows)
        if int(base_rows.numel()) > 0:
            base_input = x.index_select(0, base_rows)
            base_part = F.linear(base_input, base_sparse)
            out.index_copy_(0, base_rows, base_part)
        return out

    def routed_prefix_concat() -> torch.Tensor:
        """Idealized row-routed upper bound when route order is already grouped.

        The synthetic residual rows are a prefix and base rows are the suffix,
        so concatenating dense-prefix output and sparse-base suffix output keeps
        the original row order. Real serving cannot assume this after priority
        bucket selection, but this measures the best case for "do not compute
        sparse base on corrected rows" without scatter assembly.
        """
        if residual_rows_count <= 0:
            return F.linear(x, base_sparse)
        dense_part = F.linear(x[:residual_rows_count], dense_weight)
        if residual_rows_count >= rows:
            return dense_part
        base_part = F.linear(x[residual_rows_count:], base_sparse)
        return torch.cat([dense_part, base_part], dim=0)

    dense_ms = time_cuda(dense_full, warmup=warmup, repeats=repeats)
    base_sparse_ms = time_cuda(base_sparse_full, warmup=warmup, repeats=repeats)
    mask_cpu_ms = time_cpu(
        mask_build_cpu_loop, warmup=warmup, repeats=repeats, sync=True
    )
    mask_vectorized_ms = time_cuda(
        mask_build_vectorized_cuda, warmup=warmup, repeats=repeats
    )
    bucket_topk_ms = time_cuda(bucket_topk, warmup=warmup, repeats=repeats)

    if residual_rows_count > 0:
        gather_input_ms = time_cuda(
            gather_residual_input, warmup=warmup, repeats=repeats
        )
        residual_dense_gemm_ms = time_cuda(
            residual_dense_gemm, warmup=warmup, repeats=repeats
        )
        residual_delta_dense_gemm_ms = time_cuda(
            residual_delta_dense_gemm, warmup=warmup, repeats=repeats
        )
        residual_delta_sparse_gemm_ms = time_cuda(
            residual_delta_sparse_gemm, warmup=warmup, repeats=repeats
        )
        gather_base_ms = time_cuda(gather_base_output, warmup=warmup, repeats=repeats)
        delta_ms = time_cuda(delta_compute, warmup=warmup, repeats=repeats)
        clone_ms = time_cuda(output_clone, warmup=warmup, repeats=repeats)
        scatter_add_ms = time_cuda(scatter_index_add, warmup=warmup, repeats=repeats)
        scatter_copy_ms = time_cuda(scatter_index_copy, warmup=warmup, repeats=repeats)
        bucket_delta_ms = time_cuda(
            bucket_delta_inplace, warmup=warmup, repeats=repeats
        )
        bucket_dense_copy_ms = time_cuda(
            bucket_dense_copy_inplace, warmup=warmup, repeats=repeats
        )
        compressed_delta_dense_ms = time_cuda(
            compressed_delta_dense_inplace, warmup=warmup, repeats=repeats
        )
        compressed_delta_sparse_ms = time_cuda(
            compressed_delta_sparse_inplace, warmup=warmup, repeats=repeats
        )
        if sr24_compressed_residual_triton_linear is not None:
            compressed_delta_triton_ms = time_cuda(
                compressed_delta_triton_inplace, warmup=warmup, repeats=repeats
            )
        else:
            compressed_delta_triton_ms = None
        if sr24_triton_bucket_dense_gemm_scatter_inplace is not None:
            os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM"] = "1"
            bucket_triton_dense_gemm_scatter_ms = time_cuda(
                bucket_triton_dense_gemm_scatter, warmup=warmup, repeats=repeats
            )
        else:
            bucket_triton_dense_gemm_scatter_ms = None
        routed_split_ms = time_cuda(routed_split, warmup=warmup, repeats=repeats)
        routed_prefix_concat_ms = time_cuda(
            routed_prefix_concat, warmup=warmup, repeats=repeats
        )
    else:
        gather_input_ms = 0.0
        residual_dense_gemm_ms = 0.0
        residual_delta_dense_gemm_ms = 0.0
        residual_delta_sparse_gemm_ms = 0.0
        gather_base_ms = 0.0
        delta_ms = 0.0
        clone_ms = 0.0
        scatter_add_ms = 0.0
        scatter_copy_ms = 0.0
        bucket_delta_ms = base_sparse_ms
        bucket_dense_copy_ms = base_sparse_ms
        compressed_delta_dense_ms = base_sparse_ms
        compressed_delta_sparse_ms = base_sparse_ms
        compressed_delta_triton_ms = base_sparse_ms
        bucket_triton_dense_gemm_scatter_ms = base_sparse_ms
        routed_split_ms = base_sparse_ms
        routed_prefix_concat_ms = base_sparse_ms

    dense_graph_ms, dense_graph_error = time_graph(
        dense_full, warmup=warmup, repeats=repeats
    )
    base_sparse_graph_ms, base_sparse_graph_error = time_graph(
        base_sparse_full, warmup=warmup, repeats=repeats
    )
    bucket_delta_graph_ms, bucket_delta_graph_error = (
        time_graph(bucket_delta_inplace, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    bucket_dense_copy_graph_ms, bucket_dense_copy_graph_error = (
        time_graph(bucket_dense_copy_inplace, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    compressed_delta_dense_graph_ms, compressed_delta_dense_graph_error = (
        time_graph(compressed_delta_dense_inplace, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    compressed_delta_sparse_graph_ms, compressed_delta_sparse_graph_error = (
        time_graph(compressed_delta_sparse_inplace, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    compressed_delta_triton_graph_ms, compressed_delta_triton_graph_error = (
        time_graph(compressed_delta_triton_inplace, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        and sr24_compressed_residual_triton_linear is not None
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    bucket_triton_dense_gemm_scatter_graph_ms, (
        bucket_triton_dense_gemm_scatter_graph_error
    ) = (
        time_graph(
            bucket_triton_dense_gemm_scatter,
            warmup=warmup,
            repeats=repeats,
        )
        if residual_rows_count > 0
        and sr24_triton_bucket_dense_gemm_scatter_inplace is not None
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    routed_split_graph_ms, routed_split_graph_error = (
        time_graph(routed_split, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )
    routed_prefix_concat_graph_ms, routed_prefix_concat_graph_error = (
        time_graph(routed_prefix_concat, warmup=warmup, repeats=repeats)
        if residual_rows_count > 0
        else (base_sparse_graph_ms, base_sparse_graph_error)
    )

    correction_components_ms = (
        gather_input_ms
        + residual_dense_gemm_ms
        + gather_base_ms
        + delta_ms
        + scatter_add_ms
    )
    gather_scatter_ms = gather_input_ms + gather_base_ms + delta_ms + scatter_add_ms

    max_diff = float(
        (
            dense_output.index_select(0, residual_rows) - residual_dense_output
        ).abs().max().item()
    ) if residual_rows_count > 0 else 0.0

    return {
        "rows": rows,
        "out_features": out_features,
        "in_features": in_features,
        "dtype": str(dtype).replace("torch.", ""),
        "residual_fraction": residual_fraction,
        "residual_rows": residual_rows_count,
        "base_rows": int(base_rows.numel()),
        "bucket_candidate_rows": bucket_candidate_rows,
        "bucket_active_rows": active_bucket_rows,
        "bucket_fill_ratio": (
            active_bucket_rows / bucket_candidate_rows
            if bucket_candidate_rows
            else None
        ),
        "dense_ms": dense_ms,
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "base_sparse_ms": base_sparse_ms,
        "base_sparse_graph_ms": base_sparse_graph_ms,
        "base_sparse_graph_error": base_sparse_graph_error,
        "scheduler_mask_cpu_loop_ms": mask_cpu_ms,
        "scheduler_mask_vectorized_cuda_ms": mask_vectorized_ms,
        "scheduler_bucket_topk_cuda_ms": bucket_topk_ms,
        "residual_gather_input_cuda_ms": gather_input_ms,
        "residual_dense_gemm_cuda_ms": residual_dense_gemm_ms,
        "residual_delta_dense_gemm_cuda_ms": residual_delta_dense_gemm_ms,
        "residual_delta_sparse_gemm_cuda_ms": residual_delta_sparse_gemm_ms,
        "residual_gather_base_cuda_ms": gather_base_ms,
        "residual_delta_cuda_ms": delta_ms,
        "residual_output_clone_cuda_ms": clone_ms,
        "residual_scatter_index_add_with_clone_cuda_ms": scatter_add_ms,
        "residual_scatter_index_copy_with_clone_cuda_ms": scatter_copy_ms,
        "gather_scatter_cuda_ms": gather_scatter_ms,
        "correction_components_cuda_ms": correction_components_ms,
        "bucket_delta_inplace_ms": bucket_delta_ms,
        "bucket_delta_inplace_graph_ms": bucket_delta_graph_ms,
        "bucket_delta_inplace_graph_error": bucket_delta_graph_error,
        "bucket_dense_copy_inplace_ms": bucket_dense_copy_ms,
        "bucket_dense_copy_inplace_graph_ms": bucket_dense_copy_graph_ms,
        "bucket_dense_copy_inplace_graph_error": bucket_dense_copy_graph_error,
        "compressed_delta_dense_inplace_ms": compressed_delta_dense_ms,
        "compressed_delta_dense_inplace_graph_ms": compressed_delta_dense_graph_ms,
        "compressed_delta_dense_inplace_graph_error": (
            compressed_delta_dense_graph_error
        ),
        "compressed_delta_sparse_inplace_ms": compressed_delta_sparse_ms,
        "compressed_delta_sparse_inplace_graph_ms": compressed_delta_sparse_graph_ms,
        "compressed_delta_sparse_inplace_graph_error": (
            compressed_delta_sparse_graph_error
        ),
        "compressed_delta_triton_inplace_ms": compressed_delta_triton_ms,
        "compressed_delta_triton_inplace_graph_ms": compressed_delta_triton_graph_ms,
        "compressed_delta_triton_inplace_graph_error": (
            compressed_delta_triton_graph_error
        ),
        "bucket_triton_dense_gemm_scatter_ms": (
            bucket_triton_dense_gemm_scatter_ms
        ),
        "bucket_triton_dense_gemm_scatter_graph_ms": (
            bucket_triton_dense_gemm_scatter_graph_ms
        ),
        "bucket_triton_dense_gemm_scatter_graph_error": (
            bucket_triton_dense_gemm_scatter_graph_error
        ),
        "routed_split_ms": routed_split_ms,
        "routed_split_graph_ms": routed_split_graph_ms,
        "routed_split_graph_error": routed_split_graph_error,
        "routed_prefix_concat_ms": routed_prefix_concat_ms,
        "routed_prefix_concat_graph_ms": routed_prefix_concat_graph_ms,
        "routed_prefix_concat_graph_error": routed_prefix_concat_graph_error,
        "residual_dense_row_max_diff": max_diff,
    }


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

    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SR24 Component Breakdown\n\n")
        handle.write(
            "This GPU microbenchmark isolates the main costs in the current "
            "SR24 path. `bucket_delta_inplace` is the serving-like mixed path: "
            "sparse base for all rows, dense full Linear for residual rows, "
            "then dense-minus-base scatter back into the base output. "
            "`compressed_delta_dense_inplace` simulates the current cached "
            "compressed-residual path after the residual weight has already "
            "been materialized as a dense tensor: sparse base for all rows, "
            "dense residual-delta GEMM for residual rows, then index-add. "
            "`compressed_delta_sparse_inplace` is an exact but more idealized "
            "base sparse plus residual sparse GEMM path. "
            "GPU utilization is not measured by this script; use the serving "
            "breakdown runner for `avg_gpu_util_pct` and CUDA Graph FULL/NONE "
            "step counts.\n\n"
        )
        handle.write(
            "| rows | out | in | residual frac | residual rows | bucket fill | "
            "dense graph | base sparse graph | base/dense | mask CPU loop | "
            "mask vector CUDA | bucket topk | dense-row GEMM | gather+scatter | "
            "component sum | mixed total graph | mixed/dense | bucket dense "
            "copy graph | copy/dense | compressed dense delta graph | "
            "comp-dense/dense | compressed sparse delta graph | "
            "comp-sparse/dense | compressed Triton delta graph | "
            "comp-triton/dense | Triton dense scatter graph | Triton/dense | "
            "routed split graph | routed/dense | prefix concat graph | "
            "prefix/dense | graph status |\n"
        )
        handle.write(
            "|-----:|----:|---:|--------------:|--------------:|------------:|"
            "------------:|------------------:|-----------:|--------------:|"
            "-----------------:|------------:|---------------:|---------------:|"
            "--------------:|------------------:|------------:|-----------------------:|"
            "-----------:|-----------------------:|----------------:|"
            "------------------------:|-----------------:|-------------------:|-------------:|"
            "-------------------------------:|-------------------:|-------------------:|"
            "------------:|--------------------:|-------------:|-------------|\n"
        )
        for row in rows:
            dense_ref = row.get("dense_graph_ms") or row.get("dense_ms")
            bucket_fill = (
                ""
                if row["bucket_fill_ratio"] is None
                else f"{float(row['bucket_fill_ratio']):.3f}"
            )
            graph_status = "FULL"
            errors = [
                row.get("dense_graph_error"),
                row.get("base_sparse_graph_error"),
                row.get("bucket_delta_inplace_graph_error"),
                row.get("bucket_dense_copy_inplace_graph_error"),
                row.get("compressed_delta_dense_inplace_graph_error"),
                row.get("compressed_delta_sparse_inplace_graph_error"),
                row.get("compressed_delta_triton_inplace_graph_error"),
                row.get("bucket_triton_dense_gemm_scatter_graph_error"),
                row.get("routed_split_graph_error"),
                row.get("routed_prefix_concat_graph_error"),
            ]
            if any(errors):
                graph_status = "; ".join(str(err) for err in errors if err)
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                f"{float(row['residual_fraction']):.3f} | {row['residual_rows']} | "
                f"{bucket_fill} | "
                f"{fmt_ms(row['dense_graph_ms'])} | "
                f"{fmt_ms(row['base_sparse_graph_ms'])} | "
                f"{fmt_ratio(row['base_sparse_graph_ms'], dense_ref)} | "
                f"{fmt_ms(row['scheduler_mask_cpu_loop_ms'])} | "
                f"{fmt_ms(row['scheduler_mask_vectorized_cuda_ms'])} | "
                f"{fmt_ms(row['scheduler_bucket_topk_cuda_ms'])} | "
                f"{fmt_ms(row['residual_dense_gemm_cuda_ms'])} | "
                f"{fmt_ms(row['gather_scatter_cuda_ms'])} | "
                f"{fmt_ms(row['correction_components_cuda_ms'])} | "
                f"{fmt_ms(row['bucket_delta_inplace_graph_ms'])} | "
                f"{fmt_ratio(row['bucket_delta_inplace_graph_ms'], dense_ref)} | "
                f"{fmt_ms(row.get('bucket_dense_copy_inplace_graph_ms'))} | "
                f"{fmt_ratio(row.get('bucket_dense_copy_inplace_graph_ms'), dense_ref)} | "
                f"{fmt_ms(row.get('compressed_delta_dense_inplace_graph_ms'))} | "
                f"{fmt_ratio(row.get('compressed_delta_dense_inplace_graph_ms'), dense_ref)} | "
                f"{fmt_ms(row.get('compressed_delta_sparse_inplace_graph_ms'))} | "
                f"{fmt_ratio(row.get('compressed_delta_sparse_inplace_graph_ms'), dense_ref)} | "
                f"{fmt_ms(row.get('compressed_delta_triton_inplace_graph_ms'))} | "
                f"{fmt_ratio(row.get('compressed_delta_triton_inplace_graph_ms'), dense_ref)} | "
                f"{fmt_ms(row.get('bucket_triton_dense_gemm_scatter_graph_ms'))} | "
                f"{fmt_ratio(row.get('bucket_triton_dense_gemm_scatter_graph_ms'), dense_ref)} | "
                f"{fmt_ms(row['routed_split_graph_ms'])} | "
                f"{fmt_ratio(row['routed_split_graph_ms'], dense_ref)} | "
                f"{fmt_ms(row['routed_prefix_concat_graph_ms'])} | "
                f"{fmt_ratio(row['routed_prefix_concat_graph_ms'], dense_ref)} | "
                f"{graph_status} |\n"
            )

        handle.write("\n## Reading the table\n\n")
        handle.write(
            "- `base sparse graph` shows the best-case 2:4 base Linear cost when "
            "it can be captured.\n"
            "- `dense-row GEMM` is the dense correction GEMM for rows routed to "
            "residual/dense verification.\n"
            "- `gather+scatter` includes row gather, base-row read, delta, and "
            "scatter with output clone; this is the overhead that remains even "
            "if the dense-row GEMM is small.\n"
            "- `mixed total graph` is the closest microbench proxy for current "
            "selective SR24 per Linear call.\n"
            "- `bucket dense copy graph` is the conservative bucket variant "
            "that overwrites selected bucket rows with exact dense output "
            "instead of computing dense-minus-base deltas.\n"
            "- `compressed dense delta graph` removes the dense-minus-base "
            "gather from the current dense-row correction, but still uses a "
            "dense residual GEMM with many zeros.\n"
            "- `compressed sparse delta graph` measures whether an exact "
            "residual 2:4 GEMM could make the two-pass base+residual path "
            "competitive before writing a fused kernel.\n"
            "- `compressed Triton delta graph` is the actual direct "
            "compressed-residual Triton path from `vllm.speclink_sr24`: it "
            "reads packed GPU mask bytes plus GPU-resident residual values "
            "instead of materializing a dense residual matrix.\n"
            "- `routed split graph` avoids sparse-base work on corrected rows "
            "but still gathers and scatters both routes back to original row "
            "order.\n"
            "- `prefix concat graph` is an idealized upper bound where routed "
            "rows are already grouped, so no scatter assembly is needed.\n"
        )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SR24 component breakdown")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    output_root = args.output_root or (
        Path("examples/evaluate/eval-guidellm/results.bak")
        / f"speclink_sr24_component_breakdown_{timestamp()}"
    )
    rows: list[dict[str, Any]] = []
    for shape in args.shape:
        shape_rows, out_features, in_features = shape
        for residual_fraction in args.residual_fractions:
            rows.append(
                component_case(
                    rows=shape_rows,
                    out_features=out_features,
                    in_features=in_features,
                    residual_fraction=residual_fraction,
                    bucket_size=args.bucket_size,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )
    write_outputs(output_root, rows)
    print(output_root.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=[],
        help="ROWS,OUT,IN. Defaults to Llama gate_up and down rows=512.",
    )
    parser.add_argument(
        "--residual-fractions",
        type=parse_float_list,
        default=[0.125, 0.25, 0.5, 0.875],
        help="Comma-separated residual row fractions to probe.",
    )
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(512, 28672, 4096), (512, 4096, 14336)]
    run(args)


if __name__ == "__main__":
    main()
