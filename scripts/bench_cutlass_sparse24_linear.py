#!/usr/bin/env python3
"""Benchmark the final CUTLASS SparseGemm sparse24 linear backend."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    assert_24_weight,
    dense_cutlass_device_gemm,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_strided_,
    sparse24_add_prefix_strided_,
    sparse24_copy_indexed_rows_strided_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_gather_rows_,
    sparse_mma_status,
)


@dataclass(frozen=True)
class ShapeCase:
    label: str
    shape: tuple[int, int, int]


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,K,N")
    return parts[0], parts[1], parts[2]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_strategy_list(value: str) -> list[str]:
    allowed = {
        "full_sparse_residual",
        "full_sparse_dense_override",
        "split_dense_sparse",
        "all",
    }
    strategies = parse_str_list(value)
    bad = [item for item in strategies if item not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unsupported strategy {bad[0]!r}; use full_sparse_residual, "
            "full_sparse_dense_override, split_dense_sparse, or all"
        )
    if "all" in strategies:
        return ["all"]
    return strategies


def parse_fraction_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "/" in item:
            numerator, denominator = item.split("/", 1)
            fraction = float(numerator) / float(denominator)
        else:
            fraction = float(item)
        if fraction <= 0.0 or fraction >= 1.0:
            raise argparse.ArgumentTypeError(
                f"fraction must be in (0, 1), got {item!r}"
            )
        out.append(fraction)
    return out


def fraction_label(fraction: float) -> str:
    common = {
        0.125: "1/8",
        0.25: "1/4",
        0.5: "1/2",
    }
    for key, label in common.items():
        if abs(fraction - key) < 1.0e-9:
            return label
    return f"{fraction:.4g}"


def dense_row_count(M: int, dense_fraction: float) -> int:
    dense_rows = int(round(M * dense_fraction))
    return min(max(dense_rows, 1), M - 1)


def selective_dense_fractions_for_shape(args: argparse.Namespace, M: int) -> list[float]:
    fractions = list(args.selective_dense_fractions)
    for count in args.selective_dense_counts:
        if count <= 0 or count >= M:
            continue
        fractions.append(float(count) / float(M))
    out: list[float] = []
    seen: set[int] = set()
    for fraction in fractions:
        dense_rows = dense_row_count(M, fraction)
        if dense_rows in seen:
            continue
        seen.add(dense_rows)
        out.append(fraction)
    return out


def row_selection_seed(base_seed: int, M: int, dense_fraction: float) -> int:
    fraction_tag = int(round(dense_fraction * 1_000_000))
    return int(base_seed + M * 9176 + fraction_tag * 1009)


def backend_linear_strategy(backend: str) -> str:
    if "split_dense_sparse" in backend or backend.endswith("_split_view"):
        return "split_dense_sparse"
    if "dense_override" in backend:
        return "full_sparse_dense_override"
    if "complement" in backend:
        return "full_sparse_residual"
    return ""


def make_dense_row_indices(
    *,
    M: int,
    dense_fraction: float,
    row_selection: str,
    seed: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, str]:
    dense_rows = dense_row_count(M, dense_fraction)
    if row_selection == "prefix":
        indices64 = torch.arange(dense_rows, device=device, dtype=torch.int64)
        label = "prefix"
    elif row_selection == "random":
        generator = torch.Generator(device=device).manual_seed(seed)
        indices64 = torch.randperm(M, device=device, generator=generator)[:dense_rows]
        indices64 = torch.sort(indices64).values.contiguous()
        label = "random_sorted"
    else:
        raise ValueError(f"unsupported row_selection {row_selection!r}")
    return dense_rows, indices64, indices64.to(torch.int32).contiguous(), label


def tensor_sha1(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.sha1(data).hexdigest()


def tensor_preview(tensor: torch.Tensor, *, limit: int = 16) -> str:
    values = tensor.detach().cpu().tolist()
    preview = ";".join(str(int(value)) for value in values[:limit])
    if len(values) > limit:
        preview += ";..."
    return preview


def ideal_selective_speedup_bound(dense_fraction: float) -> float:
    return 1.0 / (dense_fraction + 0.5 * (1.0 - dense_fraction))


def selective_sparse24_device_configs(
    *,
    requested_device_config: str,
    M: int,
    dense_rows: int,
    K: int,
    N: int,
) -> tuple[str | None, str | None]:
    """Return full/residual SparseGemm config overrides for selective random rows."""

    if requested_device_config != "auto":
        return None, None
    full_override = os.environ.get(
        "SPECLINK_SPARSE24_SELECTIVE_FULL_DEVICE_CONFIG"
    )
    residual_override = os.environ.get(
        "SPECLINK_SPARSE24_SELECTIVE_RESIDUAL_DEVICE_CONFIG"
    )
    if full_override or residual_override:
        return (
            full_override or None,
            residual_override or None,
        )
    env_override = os.environ.get("SPECLINK_SPARSE24_SELECTIVE_DEVICE_CONFIG")
    if env_override and env_override != "auto":
        return env_override, env_override
    if dense_rows * 4 > M:
        return None, None

    config: str | None = None
    if K == 4096 and N == 4096:
        if M == 256 and dense_rows <= 32:
            config = "128x64x64_s4"
        elif M == 256 and dense_rows <= 64:
            config = "128x64x64_s3"
        elif M == 512 and dense_rows >= 128:
            config = "256x64x64_s3_sw4"
    if K == 4096 and N == 6144:
        if M == 512 and dense_rows >= 128:
            config = "128x64x64_s4"
    if K == 4096 and N == 24576:
        if M == 256 and dense_rows <= 32:
            config = "256x64x64_s3_sw4"
        elif M == 512 and dense_rows == 128:
            config = "128x128x64_s3_sw4"
    if K == 4096 and N == 28672:
        if M == 512 and dense_rows == 128:
            config = "128x128x64_s3_sw4"
    if N == 4096 and K >= 8192:
        if M == 256 and dense_rows <= 32:
            config = None
        elif M == 256 and dense_rows <= 64:
            config = "128x64x64_s5"
        elif M == 512 and dense_rows <= 64:
            config = "256x64x64_s3_sw4"
        elif M == 512 and dense_rows <= 128:
            config = "256x64x64_s3"
    if config is None:
        return None, None
    return config, config


def _config_shape_cases(
    *,
    model_label: str,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    m_values: list[int],
    mlp_projections: str,
) -> list[ShapeCase]:
    qkv_out = (num_attention_heads + 2 * num_key_value_heads) * head_dim
    projections = [
        ("qkv_proj", hidden_size, qkv_out),
        ("o_proj", hidden_size, hidden_size),
    ]
    if mlp_projections in ("fused", "both"):
        projections.append(("gate_up_proj", hidden_size, 2 * intermediate_size))
    if mlp_projections in ("split", "both"):
        projections.extend(
            [
                ("gate_proj", hidden_size, intermediate_size),
                ("up_proj", hidden_size, intermediate_size),
            ]
        )
    projections.append(("down_proj", intermediate_size, hidden_size))
    return [
        ShapeCase(f"{model_label}:{projection}:M{M}", (M, K, N))
        for M in m_values
        for projection, K, N in projections
    ]


def preset_shape_cases(
    preset: str,
    m_values: list[int],
    *,
    mlp_projections: str = "fused",
) -> list[ShapeCase]:
    if mlp_projections not in ("fused", "split", "both"):
        raise ValueError(f"unsupported MLP projection mode {mlp_projections!r}")
    legacy = [
        ShapeCase("legacy_decode_m8_4096", (8, 4096, 4096)),
        ShapeCase("legacy_decode_m64_4096", (64, 4096, 4096)),
        ShapeCase("legacy_decode_m128_4096", (128, 4096, 4096)),
    ]
    qwen3_8b = _config_shape_cases(
        model_label="qwen3_8b",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        m_values=m_values,
        mlp_projections=mlp_projections,
    )
    llama3_1_8b = _config_shape_cases(
        model_label="llama3_1_8b",
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        m_values=m_values,
        mlp_projections=mlp_projections,
    )
    if preset == "legacy":
        return legacy
    if preset == "qwen3_8b":
        return qwen3_8b
    if preset == "llama3_1_8b":
        return llama3_1_8b
    if preset == "serving":
        return qwen3_8b + llama3_1_8b
    if preset == "all":
        return legacy + qwen3_8b + llama3_1_8b
    raise ValueError(f"unsupported preset {preset!r}")


def cuda_event_bench(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeat)


def median_event_bench(
    fn,
    *,
    warmup: int,
    repeat: int,
    measure_trials: int,
) -> tuple[float, list[float]]:
    if measure_trials <= 0:
        raise ValueError(f"measure_trials must be positive, got {measure_trials}")
    samples = [
        cuda_event_bench(fn, warmup=warmup, repeat=repeat)
        for _ in range(measure_trials)
    ]
    return float(statistics.median(samples)), samples


def fmt_samples(values: list[float]) -> str:
    return ";".join(f"{value:.6g}" for value in values)


def max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected.float().abs().clamp_min(1.0e-6)).max().item())
    return max_abs, max_rel


def tflops(flops: float, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return flops / (ms * 1.0e-3) / 1.0e12


def make_tensors(
    shape: tuple[int, int, int],
    *,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    M, K, N = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.randn((M, K), device="cuda", dtype=torch.float16, generator=generator) * 0.25
    W = torch.randn((K, N), device="cuda", dtype=torch.float16, generator=generator) * 0.25
    W = torch.where(W == 0, torch.full_like(W, 1.0e-3), W).contiguous()
    W24, _meta = apply_random_24_mask(W, generator=generator)
    assert_24_weight(W24)
    W_comp = (W - W24).contiguous()
    assert_24_weight(W_comp)
    packed = pack_24(W24, layout="n_major")
    a_values, a_meta_e = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=K,
    )
    packed_comp = pack_24(W_comp, layout="n_major")
    comp_a_values, comp_a_meta_e = prepare_cutlass_sparse24_device_gemm(
        packed_comp.values,
        packed_comp.meta,
        layout=packed_comp.layout,
        K=K,
    )
    return (
        X.contiguous(),
        W.contiguous(),
        W24.contiguous(),
        a_values,
        a_meta_e,
        comp_a_values,
        comp_a_meta_e,
    )


def sync_parallel_streams(
    dense_stream: torch.cuda.Stream,
    sparse_stream: torch.cuda.Stream,
) -> None:
    current = torch.cuda.current_stream()
    dense_stream.wait_stream(current)
    sparse_stream.wait_stream(current)
    current.wait_stream(dense_stream)
    current.wait_stream(sparse_stream)


def make_parallel_selective_fn(
    *,
    dense_fn,
    sparse_fn,
    dense_stream: torch.cuda.Stream,
    sparse_stream: torch.cuda.Stream,
):
    def run_parallel() -> None:
        current = torch.cuda.current_stream()
        dense_stream.wait_stream(current)
        sparse_stream.wait_stream(current)
        with torch.cuda.stream(dense_stream):
            dense_fn()
        with torch.cuda.stream(sparse_stream):
            sparse_fn()
        current.wait_stream(dense_stream)
        current.wait_stream(sparse_stream)

    return run_parallel


def make_cuda_graph_fn(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    def replay() -> None:
        graph.replay()

    return replay


def make_cuda_graph_or_error(fn):
    try:
        return make_cuda_graph_fn(fn)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"

        def raise_capture_error() -> None:
            raise RuntimeError(f"CUDA graph capture failed: {message}")

        return raise_capture_error


def run_selective_dense_case(
    *,
    shape_case: ShapeCase,
    X: torch.Tensor,
    W: torch.Tensor,
    W24: torch.Tensor,
    Y24_ref: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    comp_a_values: torch.Tensor,
    comp_a_meta_e: torch.Tensor,
    dense_ms: float,
    dense_ms_samples: list[float],
    pure24_ms: float,
    pure24_backend: str,
    dense_fraction: float,
    device_config: str,
    pad_m_multiple: int,
    warmup: int,
    repeat: int,
    measure_trials: int,
    reuse_output: bool,
    row_selection: str,
    random_gather_backend: str,
    selective_dense_strategies: list[str],
    row_seed: int,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    M, K, N = shape_case.shape
    dense_rows, dense_row_indices64, dense_row_indices32, row_selection_label = (
        make_dense_row_indices(
            M=M,
            dense_fraction=dense_fraction,
            row_selection=row_selection,
            seed=row_seed,
            device=X.device,
        )
    )
    sparse_rows = M - dense_rows
    prefix_selection = row_selection_label == "prefix"
    full_m_run = ((M + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple
    sparse_m_run = ((sparse_rows + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple
    comp_m_run = ((dense_rows + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple
    selective_full_device_config, selective_residual_device_config = (
        selective_sparse24_device_configs(
            requested_device_config=device_config,
            M=M,
            dense_rows=dense_rows,
            K=K,
            N=N,
    )
    )

    X_dense_ref = X[:dense_rows] if prefix_selection else X.index_select(0, dense_row_indices64)
    dense_ref = X_dense_ref @ W
    sparse_ref = Y24_ref[dense_rows:] if prefix_selection else torch.empty((0, N), device=X.device, dtype=torch.float16)
    unselected_mask = None
    sparse_full_unselected_ref = None
    unselected_row_indices64 = None
    unselected_row_indices32 = None
    if not prefix_selection:
        unselected_mask = torch.ones(M, device=X.device, dtype=torch.bool)
        unselected_mask[dense_row_indices64] = False
        unselected_row_indices64 = unselected_mask.nonzero(as_tuple=False).flatten()
        unselected_row_indices32 = unselected_row_indices64.to(torch.int32)
        sparse_full_unselected_ref = Y24_ref[unselected_mask]

    dense_out = torch.empty((dense_rows, N), device=X.device, dtype=torch.float16)
    dense_cutlass_out = torch.empty(
        (dense_rows, N),
        device=X.device,
        dtype=torch.float16,
    )
    sparse_out_view = torch.empty_strided(
        (sparse_m_run, N),
        (1, sparse_m_run),
        device=X.device,
        dtype=torch.float16,
    )
    full_sparse_out_view = torch.empty_strided(
        (full_m_run, N),
        (1, full_m_run),
        device=X.device,
        dtype=torch.float16,
    )
    full_sparse_logical_view = full_sparse_out_view[:M]
    comp_out_view = torch.empty_strided(
        (comp_m_run, N),
        (1, comp_m_run),
        device=X.device,
        dtype=torch.float16,
    )
    comp_prefix_view = comp_out_view[:dense_rows]
    dense_override_out_view = torch.empty_strided(
        (comp_m_run, N),
        (1, comp_m_run),
        device=X.device,
        dtype=torch.float16,
    )
    dense_override_prefix_view = dense_override_out_view[:dense_rows]
    if prefix_selection:
        X_dense = X[:dense_rows]
        X_sparse = X[dense_rows:]
        X_sparse_random = None
    else:
        X_dense = torch.empty((dense_rows, K), device=X.device, dtype=torch.float16)
        X_sparse = None
        X_sparse_random = torch.empty((sparse_rows, K), device=X.device, dtype=torch.float16)
    row_indices_preview = tensor_preview(dense_row_indices32)
    row_indices_sha1 = tensor_sha1(dense_row_indices32)

    def gather_dense_rows() -> torch.Tensor:
        if prefix_selection:
            return X_dense
        if random_gather_backend == "cutlass":
            return sparse24_gather_rows_(X, dense_row_indices32, X_dense)
        if random_gather_backend == "torch":
            return torch.index_select(X, 0, dense_row_indices64, out=X_dense)
        raise ValueError(f"unsupported random_gather_backend {random_gather_backend!r}")

    def gather_sparse_rows() -> torch.Tensor:
        if prefix_selection:
            return X_sparse
        if (
            X_sparse_random is None
            or unselected_row_indices64 is None
            or unselected_row_indices32 is None
        ):
            raise RuntimeError("random sparse-row gather state is not initialized")
        if random_gather_backend == "cutlass":
            return sparse24_gather_rows_(X, unselected_row_indices32, X_sparse_random)
        if random_gather_backend == "torch":
            return torch.index_select(X, 0, unselected_row_indices64, out=X_sparse_random)
        raise ValueError(f"unsupported random_gather_backend {random_gather_backend!r}")

    def dense_part() -> torch.Tensor:
        return torch.mm(gather_dense_rows(), W, out=dense_out)

    def dense_part_cutlass() -> torch.Tensor:
        return dense_cutlass_device_gemm(gather_dense_rows(), W, out=dense_cutlass_out)

    def sparse_part_view() -> torch.Tensor:
        if X_sparse is None:
            raise RuntimeError("split sparse_part_view is only defined for prefix row selection")
        return sparse24_cutlass_device_gemm_prepacked(
            X_sparse,
            a_values,
            a_meta_e,
            contiguous_output=False,
            out=sparse_out_view,
            workspace=None,
            pad_m_multiple=pad_m_multiple,
        )

    def split_sparse_part_view() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            gather_sparse_rows(),
            a_values,
            a_meta_e,
            contiguous_output=False,
            out=sparse_out_view,
            workspace=None,
            pad_m_multiple=pad_m_multiple,
        )

    def full_sparse_part_view() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            X,
            a_values,
            a_meta_e,
            contiguous_output=False,
            out=full_sparse_out_view,
            workspace=None,
            pad_m_multiple=pad_m_multiple,
            device_config=selective_full_device_config,
        )

    def complement_dense_part_view() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            gather_dense_rows(),
            comp_a_values,
            comp_a_meta_e,
            contiguous_output=False,
            out=comp_out_view,
            workspace=None,
            pad_m_multiple=pad_m_multiple,
            device_config=selective_residual_device_config,
        )

    def dense_override_part_view() -> torch.Tensor:
        return torch.mm(gather_dense_rows(), W, out=dense_override_prefix_view)

    def serial_split_view() -> None:
        dense_part()
        sparse_part_view()

    dense_stream = torch.cuda.Stream()
    sparse_stream = torch.cuda.Stream()
    parallel_split_view = make_parallel_selective_fn(
        dense_fn=dense_part,
        sparse_fn=sparse_part_view,
        dense_stream=dense_stream,
        sparse_stream=sparse_stream,
    )
    cutlass_dense_stream = torch.cuda.Stream()
    cutlass_sparse_stream = torch.cuda.Stream()

    def cutlass_serial_split_view() -> None:
        dense_part_cutlass()
        sparse_part_view()

    cutlass_parallel_split_view = make_parallel_selective_fn(
        dense_fn=dense_part_cutlass,
        sparse_fn=sparse_part_view,
        dense_stream=cutlass_dense_stream,
        sparse_stream=cutlass_sparse_stream,
    )

    full_sparse_stream = torch.cuda.Stream()
    complement_stream = torch.cuda.Stream()

    def parallel_complement_view() -> None:
        current = torch.cuda.current_stream()
        full_sparse_stream.wait_stream(current)
        complement_stream.wait_stream(current)
        with torch.cuda.stream(full_sparse_stream):
            full_sparse_part_view()
        with torch.cuda.stream(complement_stream):
            complement_dense_part_view()
        current.wait_stream(full_sparse_stream)
        current.wait_stream(complement_stream)
        full_sparse_out_view[:dense_rows].add_(comp_prefix_view)

    def add_prefix_custom() -> torch.Tensor:
        return sparse24_add_prefix_strided_(
            full_sparse_out_view,
            comp_prefix_view,
            dense_rows=dense_rows,
        )

    def add_indexed_rows_custom() -> torch.Tensor:
        return sparse24_add_indexed_rows_strided_(
            full_sparse_out_view,
            comp_prefix_view,
            dense_row_indices32,
        )

    def add_residual_custom() -> torch.Tensor:
        if prefix_selection:
            return add_prefix_custom()
        return add_indexed_rows_custom()

    def copy_indexed_rows_custom() -> torch.Tensor:
        return sparse24_copy_indexed_rows_strided_(
            full_sparse_out_view,
            dense_override_prefix_view,
            dense_row_indices32,
        )

    def copy_unselected_rows_custom() -> torch.Tensor:
        if unselected_row_indices32 is None:
            raise RuntimeError("random sparse-row scatter state is not initialized")
        return sparse24_copy_indexed_rows_strided_(
            full_sparse_out_view,
            sparse_out_view[:sparse_rows],
            unselected_row_indices32,
        )

    def parallel_complement_custom_add_view() -> None:
        current = torch.cuda.current_stream()
        full_sparse_stream.wait_stream(current)
        complement_stream.wait_stream(current)
        with torch.cuda.stream(full_sparse_stream):
            full_sparse_part_view()
        with torch.cuda.stream(complement_stream):
            complement_dense_part_view()
        current.wait_stream(full_sparse_stream)
        current.wait_stream(complement_stream)
        add_residual_custom()

    def serial_complement_view() -> None:
        full_sparse_part_view()
        complement_dense_part_view()
        full_sparse_out_view[:dense_rows].add_(comp_prefix_view)

    def serial_complement_custom_add_view() -> None:
        full_sparse_part_view()
        complement_dense_part_view()
        add_residual_custom()

    def serial_dense_override_view() -> None:
        full_sparse_part_view()
        dense_override_part_view()
        copy_indexed_rows_custom()

    def parallel_dense_override_view() -> None:
        current = torch.cuda.current_stream()
        full_sparse_stream.wait_stream(current)
        complement_stream.wait_stream(current)
        with torch.cuda.stream(full_sparse_stream):
            full_sparse_part_view()
        with torch.cuda.stream(complement_stream):
            dense_override_part_view()
        current.wait_stream(full_sparse_stream)
        current.wait_stream(complement_stream)
        copy_indexed_rows_custom()

    def serial_split_dense_sparse_view() -> None:
        dense_override_part_view()
        split_sparse_part_view()
        copy_indexed_rows_custom()
        copy_unselected_rows_custom()

    def parallel_split_dense_sparse_view() -> None:
        current = torch.cuda.current_stream()
        full_sparse_stream.wait_stream(current)
        complement_stream.wait_stream(current)
        with torch.cuda.stream(full_sparse_stream):
            split_sparse_part_view()
        with torch.cuda.stream(complement_stream):
            dense_override_part_view()
        current.wait_stream(full_sparse_stream)
        current.wait_stream(complement_stream)
        copy_indexed_rows_custom()
        copy_unselected_rows_custom()

    if prefix_selection:
        graph_serial_split_view = make_cuda_graph_or_error(serial_split_view)
        graph_cutlass_serial_split_view = make_cuda_graph_or_error(cutlass_serial_split_view)
        graph_parallel_complement_view = make_cuda_graph_or_error(parallel_complement_view)
        graph_parallel_complement_custom_add_view = make_cuda_graph_or_error(
            parallel_complement_custom_add_view
        )
        graph_complement_view = make_cuda_graph_or_error(serial_complement_view)
        graph_complement_custom_add_view = make_cuda_graph_or_error(
            serial_complement_custom_add_view
        )
        graph_dense_override_view = make_cuda_graph_or_error(serial_dense_override_view)
        graph_parallel_dense_override_view = make_cuda_graph_or_error(
            parallel_dense_override_view
        )
        backend_fns = [
            ("selective_dense_serial_split_view", serial_split_view),
            ("selective_dense_parallel_split_view", parallel_split_view),
            ("selective_dense_cutlass_serial_split_view", cutlass_serial_split_view),
            ("selective_dense_cutlass_parallel_split_view", cutlass_parallel_split_view),
            ("selective_dense_parallel_complement_view", parallel_complement_view),
            (
                "selective_dense_parallel_complement_custom_add_view",
                parallel_complement_custom_add_view,
            ),
            ("selective_dense_graph_serial_split_view", graph_serial_split_view),
            (
                "selective_dense_cutlass_graph_serial_split_view",
                graph_cutlass_serial_split_view,
            ),
            ("selective_dense_graph_parallel_complement_view", graph_parallel_complement_view),
            (
                "selective_dense_graph_parallel_complement_custom_add_view",
                graph_parallel_complement_custom_add_view,
            ),
            ("selective_dense_graph_complement_view", graph_complement_view),
            (
                "selective_dense_graph_complement_custom_add_view",
                graph_complement_custom_add_view,
            ),
            (
                "selective_dense_complement_dense_override_view",
                serial_dense_override_view,
            ),
            (
                "selective_dense_parallel_complement_dense_override_view",
                parallel_dense_override_view,
            ),
            (
                "selective_dense_graph_complement_dense_override_view",
                graph_dense_override_view,
            ),
            (
                "selective_dense_graph_parallel_complement_dense_override_view",
                graph_parallel_dense_override_view,
            ),
        ]
    else:
        graph_parallel_complement_custom_add_view = make_cuda_graph_or_error(
            parallel_complement_custom_add_view
        )
        graph_complement_custom_add_view = make_cuda_graph_or_error(
            serial_complement_custom_add_view
        )
        graph_parallel_dense_override_view = make_cuda_graph_or_error(
            parallel_dense_override_view
        )
        graph_dense_override_view = make_cuda_graph_or_error(serial_dense_override_view)
        graph_parallel_split_dense_sparse_view = make_cuda_graph_or_error(
            parallel_split_dense_sparse_view
        )
        graph_split_dense_sparse_view = make_cuda_graph_or_error(serial_split_dense_sparse_view)
        backend_fns = [
            (
                "selective_dense_random_complement_fused_residual_view",
                serial_complement_custom_add_view,
            ),
            (
                "selective_dense_random_parallel_complement_fused_residual_view",
                parallel_complement_custom_add_view,
            ),
            (
                "selective_dense_random_graph_complement_fused_residual_view",
                graph_complement_custom_add_view,
            ),
            (
                "selective_dense_random_graph_parallel_complement_fused_residual_view",
                graph_parallel_complement_custom_add_view,
            ),
            (
                "selective_dense_random_complement_dense_override_view",
                serial_dense_override_view,
            ),
            (
                "selective_dense_random_parallel_complement_dense_override_view",
                parallel_dense_override_view,
            ),
            (
                "selective_dense_random_graph_complement_dense_override_view",
                graph_dense_override_view,
            ),
            (
                "selective_dense_random_graph_parallel_complement_dense_override_view",
                graph_parallel_dense_override_view,
            ),
            (
                "selective_dense_random_complement_split_dense_sparse_view",
                serial_split_dense_sparse_view,
            ),
            (
                "selective_dense_random_parallel_complement_split_dense_sparse_view",
                parallel_split_dense_sparse_view,
            ),
            (
                "selective_dense_random_graph_complement_split_dense_sparse_view",
                graph_split_dense_sparse_view,
            ),
            (
                "selective_dense_random_graph_parallel_complement_split_dense_sparse_view",
                graph_parallel_split_dense_sparse_view,
            ),
        ]

    if selective_dense_strategies != ["all"]:
        allowed_strategies = set(selective_dense_strategies)
        backend_fns = [
            (backend, fn)
            for backend, fn in backend_fns
            if backend_linear_strategy(backend) in allowed_strategies
        ]

    rows: list[dict[str, Any]] = []
    for backend, fn in backend_fns:
        strategy = backend_linear_strategy(backend)
        try:
            fn()
            torch.cuda.synchronize()
            if "complement" in backend:
                if prefix_selection:
                    dense_actual = full_sparse_logical_view[:dense_rows]
                    sparse_actual = full_sparse_logical_view[dense_rows:]
                    sparse_expected = sparse_ref
                else:
                    dense_actual = full_sparse_logical_view.index_select(
                        0, dense_row_indices64
                    )
                    if unselected_mask is None or sparse_full_unselected_ref is None:
                        raise RuntimeError("random row validation state is not initialized")
                    sparse_actual = full_sparse_logical_view[unselected_mask]
                    sparse_expected = sparse_full_unselected_ref
                dense_abs, dense_rel = max_errors(dense_actual, dense_ref)
                sparse_abs, sparse_rel = max_errors(
                    sparse_actual, sparse_expected
                )
                passed = bool(
                    torch.allclose(
                        dense_actual, dense_ref, rtol=rtol, atol=atol
                    )
                    and torch.allclose(
                        sparse_actual, sparse_expected, rtol=rtol, atol=atol
                    )
                )
            else:
                dense_actual = (
                    dense_cutlass_out
                    if backend.startswith("selective_dense_cutlass_")
                    else dense_out
                )
                dense_abs, dense_rel = max_errors(dense_actual, dense_ref)
                sparse_abs, sparse_rel = max_errors(sparse_out_view[:sparse_rows], sparse_ref)
                passed = bool(
                    torch.allclose(dense_actual, dense_ref, rtol=rtol, atol=atol)
                    and torch.allclose(sparse_out_view[:sparse_rows], sparse_ref, rtol=rtol, atol=atol)
                )
            mixed_ms, mixed_samples = median_event_bench(
                fn,
                warmup=warmup,
                repeat=repeat,
                measure_trials=measure_trials,
            )
            error = ""
        except Exception as exc:
            torch.cuda.synchronize()
            dense_abs = sparse_abs = float("nan")
            dense_rel = sparse_rel = float("nan")
            passed = False
            mixed_ms = float("nan")
            mixed_samples = []
            error = f"{type(exc).__name__}: {exc}"
        speedup = dense_ms / mixed_ms if mixed_ms > 0 else float("nan")
        speedup_vs_pure24 = pure24_ms / mixed_ms if mixed_ms > 0 and pure24_ms > 0 else float("nan")
        ideal_bound = ideal_selective_speedup_bound(dense_fraction)
        rows.append(
            {
                "shape_label": shape_case.label,
                "M": M,
                "M_padded": full_m_run,
                "pad_m_multiple": pad_m_multiple,
                "K": K,
                "N": N,
                "backend": backend,
                "linear_strategy": strategy,
                "output_reuse": reuse_output,
                "device_config": device_config,
                "selective_full_device_config": selective_full_device_config or "",
                "residual_device_config": selective_residual_device_config or "",
                "dense_fraction": dense_fraction,
                "dense_fraction_label": fraction_label(dense_fraction),
                "dense_rows": dense_rows,
                "sparse_rows": sparse_rows,
                "pure24_backend": pure24_backend,
                "pure24_ms": pure24_ms,
                "dense_ms": dense_ms,
                "dense_ms_samples": fmt_samples(dense_ms_samples),
                "sparse_ms": mixed_ms,
                "sparse_ms_samples": fmt_samples(mixed_samples),
                "speedup": speedup,
                "speedup_vs_pure24": speedup_vs_pure24,
                "slowdown_vs_pure24": mixed_ms / pure24_ms if pure24_ms > 0 else float("nan"),
                "ideal_mixed_speedup_bound": ideal_bound,
                "speedup_fraction_of_ideal_bound": (
                    speedup / ideal_bound if ideal_bound > 0 else float("nan")
                ),
                "dense_tflops": tflops(2.0 * M * K * N, dense_ms),
                "sparse_dense_equiv_tflops": tflops(2.0 * M * K * N, mixed_ms),
                "sparse_actual_nonzero_tflops": tflops(
                    2.0 * (dense_rows * K * N + 0.5 * sparse_rows * K * N),
                    mixed_ms,
                ),
                "row_selection": row_selection_label,
                "row_seed": "" if prefix_selection else row_seed,
                "dense_row_indices_preview": row_indices_preview,
                "dense_row_indices_sha1": row_indices_sha1,
                "gather_in_timing": not prefix_selection,
                "gather_backend": "" if prefix_selection else random_gather_backend,
                "scatter_in_timing": "complement" in backend,
                "random_sampling_in_timing": False,
                "max_abs_err": max(dense_abs, sparse_abs),
                "max_rel_err": max(dense_rel, sparse_rel),
                "dense_max_abs_err": dense_abs,
                "sparse_max_abs_err": sparse_abs,
                "pass": passed,
                "error": error,
            }
        )
    return rows


def run_case(
    shape_case: ShapeCase,
    *,
    seed: int,
    warmup: int,
    repeat: int,
    measure_trials: int,
    device_config_values: list[str],
    pad_m_multiple_values: list[int],
    selective_dense_fractions: list[float],
    reuse_output: bool,
    row_selection: str,
    random_gather_backend: str,
    selective_dense_strategies: list[str],
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    M, K, N = shape_case.shape
    if K % 64 != 0 or N % 32 != 0:
        raise ValueError(f"final CUTLASS backend requires K % 64 == 0 and N % 32 == 0: {shape_case}")

    X, W, W24, a_values, a_meta_e, comp_a_values, comp_a_meta_e = make_tensors(
        shape_case.shape,
        seed=seed,
    )
    Y_ref = X @ W24
    torch.cuda.synchronize()
    if reuse_output:
        dense_out = torch.empty((M, N), device=X.device, dtype=torch.float16)

        def dense_fn() -> torch.Tensor:
            return torch.mm(X, W, out=dense_out)
    else:
        dense_fn = lambda: X @ W
    dense_ms, dense_ms_samples = median_event_bench(
        dense_fn,
        warmup=warmup,
        repeat=repeat,
        measure_trials=measure_trials,
    )

    dense_flops = 2.0 * M * K * N
    actual_sparse_flops = dense_flops * 0.5
    rows: list[dict[str, Any]] = []
    for device_config in device_config_values:
        os.environ["SPECLINK_SPARSE24_DEVICE_CONFIG"] = device_config
        for pad_m_multiple in pad_m_multiple_values:
            if pad_m_multiple < 8 or pad_m_multiple % 8 != 0:
                raise ValueError(
                    f"pad_m_multiple must be a positive multiple of 8, got {pad_m_multiple}"
                )
            M_run = ((M + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple
            sparse_out = (
                torch.empty((M_run, N), device=X.device, dtype=torch.float16)
                if reuse_output
                else None
            )
            sparse_workspace = (
                torch.empty((N, M_run), device=X.device, dtype=torch.float16)
                if reuse_output
                else None
            )
            sparse_view_out = (
                torch.empty_strided(
                    (M_run, N),
                    (1, M_run),
                    device=X.device,
                    dtype=torch.float16,
                )
                if reuse_output
                else None
            )
            for backend, contiguous_output, out, workspace in [
                (
                    "device_sparse_gemm",
                    True,
                    sparse_out,
                    sparse_workspace,
                ),
                ("device_sparse_gemm_view", False, sparse_view_out, None),
            ]:
                base_row = {
                    "shape_label": shape_case.label,
                    "M": M,
                    "M_padded": M_run,
                    "pad_m_multiple": pad_m_multiple,
                    "K": K,
                    "N": N,
                        "backend": backend,
                        "linear_strategy": "",
                        "output_reuse": reuse_output,
                    "device_config": device_config,
                    "dense_ms": dense_ms,
                    "dense_ms_samples": fmt_samples(dense_ms_samples),
                }
                try:
                    Y = sparse24_cutlass_device_gemm_prepacked(
                        X,
                        a_values,
                        a_meta_e,
                        contiguous_output=contiguous_output,
                        out=out,
                        workspace=workspace,
                        pad_m_multiple=pad_m_multiple,
                    )
                    torch.cuda.synchronize()
                    max_abs, max_rel = max_errors(Y, Y_ref)
                    passed = bool(torch.allclose(Y, Y_ref, rtol=rtol, atol=atol))
                    sparse_ms, sparse_ms_samples = median_event_bench(
                        lambda: sparse24_cutlass_device_gemm_prepacked(
                            X,
                            a_values,
                            a_meta_e,
                            contiguous_output=contiguous_output,
                            out=out,
                            workspace=workspace,
                            pad_m_multiple=pad_m_multiple,
                        ),
                        warmup=warmup,
                        repeat=repeat,
                        measure_trials=measure_trials,
                    )
                    error = ""
                except Exception as exc:
                    torch.cuda.synchronize()
                    max_abs = float("nan")
                    max_rel = float("nan")
                    passed = False
                    sparse_ms = float("nan")
                    sparse_ms_samples = []
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        **base_row,
                        "dense_fraction": "",
                        "dense_fraction_label": "",
                        "dense_rows": "",
                        "sparse_rows": "",
                        "pure24_backend": "",
                        "pure24_ms": "",
                        "row_selection": "",
                        "row_seed": "",
                        "dense_row_indices_preview": "",
                        "dense_row_indices_sha1": "",
                        "gather_in_timing": "",
                        "gather_backend": "",
                        "scatter_in_timing": "",
                        "random_sampling_in_timing": "",
                        "sparse_ms": sparse_ms,
                        "sparse_ms_samples": fmt_samples(sparse_ms_samples),
                        "speedup": dense_ms / sparse_ms if sparse_ms > 0 else float("nan"),
                        "speedup_vs_pure24": "",
                        "slowdown_vs_pure24": "",
                        "dense_tflops": tflops(dense_flops, dense_ms),
                        "sparse_dense_equiv_tflops": tflops(dense_flops, sparse_ms),
                        "sparse_actual_nonzero_tflops": tflops(actual_sparse_flops, sparse_ms),
                        "max_abs_err": max_abs,
                        "max_rel_err": max_rel,
                        "pass": passed,
                        "error": error,
                    }
                )
            pure24_candidates = [
                row
                for row in rows
                if row["shape_label"] == shape_case.label
                and row["device_config"] == device_config
                and row["pad_m_multiple"] == pad_m_multiple
                and row["backend"] == "device_sparse_gemm_view"
                and row["pass"]
                and math.isfinite(float(row["sparse_ms"]))
            ]
            if selective_dense_fractions and pure24_candidates:
                pure24_row = min(
                    pure24_candidates,
                    key=lambda row: float(row["sparse_ms"]),
                )
                for dense_fraction in selective_dense_fractions:
                    rows.extend(
                        run_selective_dense_case(
                            shape_case=shape_case,
                            X=X,
                            W=W,
                            W24=W24,
                            Y24_ref=Y_ref,
                            a_values=a_values,
                            a_meta_e=a_meta_e,
                            comp_a_values=comp_a_values,
                            comp_a_meta_e=comp_a_meta_e,
                            dense_ms=dense_ms,
                            dense_ms_samples=dense_ms_samples,
                            pure24_ms=float(pure24_row["sparse_ms"]),
                            pure24_backend=str(pure24_row["backend"]),
                            dense_fraction=dense_fraction,
                            device_config=device_config,
                            pad_m_multiple=pad_m_multiple,
                            warmup=warmup,
                            repeat=repeat,
                            measure_trials=measure_trials,
                            reuse_output=reuse_output,
                            row_selection=row_selection,
                            random_gather_backend=random_gather_backend,
                            selective_dense_strategies=selective_dense_strategies,
                            row_seed=row_selection_seed(seed, M, dense_fraction),
                            rtol=rtol,
                            atol=atol,
                        )
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def row_passes(row: dict[str, Any]) -> bool:
    value = row.get("pass")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def write_report(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    passing = [row for row in rows if row_passes(row)]
    best_by_shape: list[dict[str, Any]] = []
    for shape in sorted({(row["shape_label"], row["M"], row["K"], row["N"]) for row in rows}):
        candidates = [
            row
            for row in passing
            if (row["shape_label"], row["M"], row["K"], row["N"]) == shape
            and math.isfinite(as_float(row.get("sparse_ms")))
        ]
        if candidates:
            best_by_shape.append(min(candidates, key=lambda row: as_float(row.get("sparse_ms"))))
    speedups = [
        as_float(row.get("speedup"))
        for row in best_by_shape
        if as_float(row.get("speedup")) > 0
    ]
    geomean = statistics.geometric_mean(speedups) if speedups else float("nan")

    with path.open("w") as handle:
        handle.write("# CUTLASS Sparse24 Linear Benchmark\n\n")
        handle.write("- Backend: final CUTLASS device SparseGemm path.\n")
        handle.write("- `device_sparse_gemm_view` skips the final transpose and is the fastest measured output form.\n")
        handle.write(f"- Best-row geomean speedup over dense cuBLAS: `{fmt(geomean)}`.\n")
        handle.write(f"- Passing rows: `{len(passing)} / {len(rows)}`.\n\n")
        handle.write("## Environment\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(metadata, indent=2, sort_keys=True))
        handle.write("\n```\n\n")
        handle.write("## Results\n\n")
        columns = [
            "shape_label",
            "M",
            "M_padded",
            "pad_m_multiple",
            "K",
            "N",
            "backend",
            "dense_fraction_label",
            "dense_rows",
            "sparse_rows",
            "row_selection",
            "row_seed",
            "dense_row_indices_preview",
            "dense_row_indices_sha1",
            "gather_in_timing",
            "gather_backend",
            "scatter_in_timing",
            "random_sampling_in_timing",
            "output_reuse",
            "device_config",
            "dense_ms",
            "dense_ms_samples",
            "pure24_ms",
            "sparse_ms",
            "sparse_ms_samples",
            "speedup",
            "speedup_vs_pure24",
            "slowdown_vs_pure24",
            "ideal_mixed_speedup_bound",
            "speedup_fraction_of_ideal_bound",
            "dense_tflops",
            "sparse_dense_equiv_tflops",
            "sparse_actual_nonzero_tflops",
            "max_abs_err",
            "pass",
            "error",
        ]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |\n")
        handle.write("\n## Best Per Shape\n\n")
        for row in best_by_shape:
            handle.write(
                f"- {row['shape_label']} M={row['M']} K={row['K']} N={row['N']}: "
                f"backend={row['backend']}, config={row['device_config']}, "
                f"pad_m_multiple={row['pad_m_multiple']}, "
                f"sparse_ms={fmt(row['sparse_ms'])}, speedup={fmt(row['speedup'])}\n"
            )


def shape_label_parts(label: str) -> tuple[str, str, int]:
    parts = label.split(":")
    model_label = parts[0] if parts else label
    projection = parts[1] if len(parts) > 1 else label
    m_value = 0
    if len(parts) > 2 and parts[2].startswith("M"):
        try:
            m_value = int(parts[2][1:])
        except ValueError:
            m_value = 0
    return model_label, projection, m_value


def write_speedup_plot(output_root: Path, rows: list[dict[str, Any]]) -> None:
    plot_rows: list[tuple[str, str, int, float]] = []
    for row in rows:
        if row.get("backend") != "device_sparse_gemm_view" or not row_passes(row):
            continue
        try:
            speedup = float(row.get("speedup"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(speedup):
            continue
        model_label, projection, m_value = shape_label_parts(str(row.get("shape_label") or ""))
        plot_rows.append((model_label, projection, m_value, speedup))
    if not plot_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projection_order = {
        "qkv_proj": 0,
        "o_proj": 1,
        "gate_up_proj": 2,
        "gate_proj": 3,
        "up_proj": 4,
        "down_proj": 5,
    }
    models = sorted({item[0] for item in plot_rows})
    fig_width = max(10.0, max(1, len(plot_rows)) * 0.34)
    fig, axes_obj = plt.subplots(
        len(models),
        1,
        figsize=(fig_width, max(3.2, 3.0 * len(models))),
        squeeze=False,
    )
    axes = [row[0] for row in axes_obj]
    for ax, model_label in zip(axes, models):
        points = sorted(
            (item for item in plot_rows if item[0] == model_label),
            key=lambda item: (projection_order.get(item[1], 99), item[2], item[1]),
        )
        labels = [f"{projection}\nM={m_value}" for _, projection, m_value, _ in points]
        values = [speedup for *_, speedup in points]
        ax.bar(range(len(points)), values, color="#4C78A8")
        ax.axhline(1.0, color="#D62728", linewidth=1.0, linestyle="--")
        ax.set_title(model_label)
        ax.set_ylabel("speedup vs dense")
        ax.set_ylim(0.0, max(2.0, max(values) * 1.15))
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(range(len(points)), labels, rotation=45, ha="right")
    fig.suptitle("CUTLASS 2:4 Linear speedup over dense cuBLAS")
    fig.tight_layout()
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "sparse24_linear_speedup.png", dpi=200)
    plt.close(fig)


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def selective_backend_produces_full_output(backend: str) -> bool:
    """Return whether a selective backend materializes one complete output tensor."""

    return (
        "complement" in backend
        or "dense_override" in backend
        or "random_complement_split_dense_sparse" in backend
    )


def selective_best_rows(
    rows: list[dict[str, Any]],
    *,
    end_to_end_only: bool = False,
    row_selection_filter: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        backend = str(row.get("backend") or "")
        if not backend.startswith("selective_dense_") or not row_passes(row):
            continue
        if end_to_end_only and not selective_backend_produces_full_output(backend):
            continue
        if (
            row_selection_filter is not None
            and str(row.get("row_selection") or "") != row_selection_filter
        ):
            continue
        speedup = as_float(row.get("speedup"))
        if not math.isfinite(speedup):
            continue
        key = (str(row.get("shape_label") or ""), str(row.get("dense_fraction_label") or ""))
        groups.setdefault(key, []).append(row)
    best: list[dict[str, Any]] = []
    for candidates in groups.values():
        row = dict(max(candidates, key=lambda row: as_float(row.get("speedup"))))
        dense_fraction = as_float(row.get("dense_fraction"))
        if math.isfinite(dense_fraction):
            ideal_bound = ideal_selective_speedup_bound(dense_fraction)
            row["ideal_mixed_speedup_bound"] = ideal_bound
            row["speedup_fraction_of_ideal_bound"] = (
                as_float(row.get("speedup")) / ideal_bound
                if ideal_bound > 0
                else float("nan")
            )
        best.append(row)
    return sorted(
        best,
        key=lambda row: (
            shape_label_parts(str(row.get("shape_label") or "")),
            as_float(row.get("dense_fraction")),
        ),
    )


def write_selective_dense_end_to_end_outputs(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    row_selection_filter: str | None = None,
    output_prefix: str = "selective_dense_e2e",
    title: str = "Selective Dense End-to-End Best Backends",
    description: str | None = None,
) -> None:
    best = selective_best_rows(
        rows,
        end_to_end_only=True,
        row_selection_filter=row_selection_filter,
    )
    if not best:
        return
    write_csv(output_root / f"{output_prefix}_best.csv", best)

    by_fraction: dict[str, list[float]] = {}
    for row in best:
        label = str(row.get("dense_fraction_label") or "")
        by_fraction.setdefault(label, []).append(as_float(row.get("speedup")))

    with (output_root / f"{output_prefix}_best.md").open("w") as handle:
        handle.write(f"# {title}\n\n")
        if description is None:
            description = (
                "This stricter view only considers complement backends that materialize "
                "one complete Linear output tensor. Split-only backends are excluded "
                "because they leave dense and sparse row ranges in separate buffers."
            )
        handle.write(description.rstrip() + "\n\n")
        handle.write(
            "| dense_fraction | cases | >=1.4x | ideal_bound | min | median | geomean | geomean/ideal |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for label in sorted(
            by_fraction,
            key=lambda item: as_float(
                next(
                    row
                    for row in best
                    if row.get("dense_fraction_label") == item
                ).get("dense_fraction")
            ),
        ):
            values = [value for value in by_fraction[label] if math.isfinite(value)]
            if not values:
                continue
            dense_fraction = as_float(
                next(row for row in best if row.get("dense_fraction_label") == label).get(
                    "dense_fraction"
                )
            )
            ideal_bound = ideal_selective_speedup_bound(dense_fraction)
            geomean = statistics.geometric_mean(values)
            handle.write(
                f"| {label} | {len(values)} | {sum(value >= 1.4 for value in values)} | "
                f"{ideal_bound:.4f} | {min(values):.4f} | "
                f"{statistics.median(values):.4f} | {geomean:.4f} | "
                f"{(geomean / ideal_bound):.4f} |\n"
            )
        handle.write("\n## Rows Below 1.4x\n\n")
        low_rows = [row for row in best if as_float(row.get("speedup")) < 1.4]
        if not low_rows:
            handle.write("All end-to-end selective-dense rows are at least 1.4x dense.\n")
        else:
            for row in sorted(low_rows, key=lambda item: as_float(item.get("speedup"))):
                handle.write(
                    f"- {row['shape_label']} fraction={row['dense_fraction_label']} "
                    f"backend={row['backend']} speedup={fmt(as_float(row['speedup']))} "
                    f"slowdown_vs_pure24={fmt(as_float(row['slowdown_vs_pure24']))}\n"
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fractions = sorted(
        {str(row.get("dense_fraction_label") or "") for row in best},
        key=lambda label: as_float(
            next(row for row in best if row.get("dense_fraction_label") == label).get(
                "dense_fraction"
            )
        ),
    )
    x = list(range(len(fractions)))
    medians = [
        statistics.median(
            [
                as_float(row.get("speedup"))
                for row in best
                if row.get("dense_fraction_label") == label
            ]
        )
        for label in fractions
    ]
    geomeans = [
        statistics.geometric_mean(
            [
                as_float(row.get("speedup"))
                for row in best
                if row.get("dense_fraction_label") == label
                and as_float(row.get("speedup")) > 0
            ]
        )
        for label in fractions
    ]
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([value - width / 2 for value in x], medians, width, label="median")
    ax.bar([value + width / 2 for value in x], geomeans, width, label="geomean")
    ax.axhline(1.4, color="#D62728", linestyle="--", linewidth=1.2, label="1.4x target")
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=0.9)
    ax.set_xticks(x, fractions)
    ax.set_xlabel("dense fraction")
    ax.set_ylabel("speedup vs dense cuBLAS")
    ax.set_ylim(0.0, max(1.8, max(medians + geomeans) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle(title.replace(" Backends", " Speedup"))
    fig.tight_layout()
    fig.savefig(figures_dir / f"{output_prefix}_best_speedup.png", dpi=200)
    plt.close(fig)


def write_selective_dense_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    best = selective_best_rows(rows)
    if not best:
        return
    write_csv(output_root / "selective_dense_best.csv", best)

    by_fraction: dict[str, list[float]] = {}
    for row in best:
        label = str(row.get("dense_fraction_label") or "")
        by_fraction.setdefault(label, []).append(as_float(row.get("speedup")))

    with (output_root / "selective_dense_best.md").open("w") as handle:
        handle.write("# Selective Dense Best Backends\n\n")
        handle.write(
            "Best backend is selected independently for each model/projection/M/fraction "
            "from the measured selective-dense variants.\n\n"
        )
        handle.write(
            "| dense_fraction | cases | >=1.4x | ideal_bound | min | median | geomean | geomean/ideal |\n"
        )
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for label in sorted(
            by_fraction,
            key=lambda item: as_float(best[next(i for i, row in enumerate(best) if row.get("dense_fraction_label") == item)].get("dense_fraction")),
        ):
            values = [value for value in by_fraction[label] if math.isfinite(value)]
            if not values:
                continue
            ideal_bound = ideal_selective_speedup_bound(
                as_float(
                    next(
                        row
                        for row in best
                        if row.get("dense_fraction_label") == label
                    ).get("dense_fraction")
                )
            )
            geomean = statistics.geometric_mean(values)
            handle.write(
                f"| {label} | {len(values)} | {sum(value >= 1.4 for value in values)} | "
                f"{ideal_bound:.4f} | {min(values):.4f} | "
                f"{statistics.median(values):.4f} | {geomean:.4f} | "
                f"{(geomean / ideal_bound):.4f} |\n"
            )
        handle.write(
            "\nThe ideal bound assumes selected dense rows use dense tensor cores and "
            "the remaining rows get an exact 2x sparse tensor-core compute-rate gain, "
            "with no launch, memory, or scheduling overhead.\n"
        )
        handle.write("\n## Rows Below 1.4x\n\n")
        low_rows = [row for row in best if as_float(row.get("speedup")) < 1.4]
        if not low_rows:
            handle.write("All best selective-dense rows are at least 1.4x dense.\n")
        else:
            for row in sorted(low_rows, key=lambda item: as_float(item.get("speedup"))):
                handle.write(
                    f"- {row['shape_label']} fraction={row['dense_fraction_label']} "
                    f"backend={row['backend']} speedup={fmt(as_float(row['speedup']))} "
                    f"slowdown_vs_pure24={fmt(as_float(row['slowdown_vs_pure24']))}\n"
                )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    projection_order = {
        "qkv_proj": 0,
        "o_proj": 1,
        "gate_up_proj": 2,
        "gate_proj": 3,
        "up_proj": 4,
        "down_proj": 5,
    }
    fractions = sorted(
        {str(row.get("dense_fraction_label") or "") for row in best},
        key=lambda label: as_float(
            next(row for row in best if row.get("dense_fraction_label") == label).get(
                "dense_fraction"
            )
        ),
    )
    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", "-.", ":"]
    models = sorted({shape_label_parts(str(row.get("shape_label") or ""))[0] for row in best})
    fig_width = max(10.0, len({row.get("shape_label") for row in best}) * 0.22)
    fig, axes_obj = plt.subplots(
        len(models),
        1,
        figsize=(fig_width, max(3.4, 3.2 * len(models))),
        squeeze=False,
    )
    axes = [row[0] for row in axes_obj]
    for ax, model_label in zip(axes, models):
        model_rows = [row for row in best if shape_label_parts(str(row.get("shape_label") or ""))[0] == model_label]
        shape_keys = sorted(
            {str(row.get("shape_label") or "") for row in model_rows},
            key=lambda label: (
                projection_order.get(shape_label_parts(label)[1], 99),
                shape_label_parts(label)[2],
                shape_label_parts(label)[1],
            ),
        )
        x_index = {label: index for index, label in enumerate(shape_keys)}
        for idx, fraction in enumerate(fractions):
            fraction_rows = [
                row
                for row in model_rows
                if str(row.get("dense_fraction_label") or "") == fraction
            ]
            xs = [x_index[str(row.get("shape_label") or "")] for row in fraction_rows]
            ys = [as_float(row.get("speedup")) for row in fraction_rows]
            ax.plot(
                xs,
                ys,
                label=f"dense {fraction}",
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=1.5,
                markersize=4,
            )
        ax.axhline(1.4, color="#D62728", linewidth=1.0, linestyle="--", label="1.4x target")
        ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle=":")
        ax.set_title(model_label)
        ax.set_ylabel("best speedup vs dense cuBLAS")
        values = [as_float(row.get("speedup")) for row in model_rows]
        ax.set_ylim(0.0, max(2.0, max(values) * 1.15))
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(
            range(len(shape_keys)),
            [f"{shape_label_parts(label)[1]}\nM={shape_label_parts(label)[2]}" for label in shape_keys],
            rotation=45,
            ha="right",
        )
        ax.legend(ncols=min(4, len(fractions) + 1), fontsize=8)
    fig.suptitle("Selective dense + 2:4 best measured speedup")
    fig.tight_layout()
    fig.savefig(figures_dir / "selective_dense_best_speedup.png", dpi=200)
    plt.close(fig)


def proportional_selective_speedup_bound(
    dense_fraction: float,
    pure24_speedup: float,
) -> float:
    if pure24_speedup <= 0 or not math.isfinite(pure24_speedup):
        return float("nan")
    return 1.0 / (dense_fraction + (1.0 - dense_fraction) / pure24_speedup)


def write_selective_dense_feasibility(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    end_to_end_only: bool = False,
    output_stem: str = "selective_dense_feasibility",
    row_selection_filter: str | None = None,
) -> None:
    best = selective_best_rows(
        rows,
        end_to_end_only=end_to_end_only,
        row_selection_filter=row_selection_filter,
    )
    if not best:
        return

    audit_rows: list[dict[str, Any]] = []
    for row in best:
        dense_fraction = as_float(row.get("dense_fraction"))
        dense_ms = as_float(row.get("dense_ms"))
        pure24_ms = as_float(row.get("pure24_ms"))
        speedup = as_float(row.get("speedup"))
        if not math.isfinite(dense_fraction):
            continue
        ideal_bound = ideal_selective_speedup_bound(dense_fraction)
        pure24_speedup = (
            dense_ms / pure24_ms
            if dense_ms > 0 and pure24_ms > 0
            else float("nan")
        )
        proportional_bound = proportional_selective_speedup_bound(
            dense_fraction,
            pure24_speedup,
        )
        if speedup >= 1.4:
            status = "measured_ge_1p4"
        elif ideal_bound < 1.4:
            status = "ideal_compute_bound_lt_1p4"
        elif math.isfinite(proportional_bound) and proportional_bound < 1.4:
            status = "measured_pure24_bound_lt_1p4"
        else:
            status = "overhead_or_schedule_limited"
        audit_rows.append(
            {
                "shape_label": row.get("shape_label"),
                "M": row.get("M"),
                "K": row.get("K"),
                "N": row.get("N"),
                "dense_fraction_label": row.get("dense_fraction_label"),
                "dense_fraction": dense_fraction,
                "backend": row.get("backend"),
                "dense_ms": dense_ms,
                "pure24_ms": pure24_ms,
                "mixed_ms": as_float(row.get("sparse_ms")),
                "pure24_speedup": pure24_speedup,
                "mixed_speedup": speedup,
                "speedup_vs_pure24": as_float(row.get("speedup_vs_pure24")),
                "slowdown_vs_pure24": as_float(row.get("slowdown_vs_pure24")),
                "ideal_mixed_speedup_bound": ideal_bound,
                "proportional_bound_from_pure24": proportional_bound,
                "mixed_fraction_of_ideal_bound": (
                    speedup / ideal_bound if ideal_bound > 0 else float("nan")
                ),
                "mixed_fraction_of_proportional_bound": (
                    speedup / proportional_bound
                    if proportional_bound > 0 and math.isfinite(proportional_bound)
                    else float("nan")
                ),
                "target_1p4_margin": speedup - 1.4,
                "status": status,
            }
        )

    if not audit_rows:
        return
    write_csv(output_root / f"{output_stem}.csv", audit_rows)

    fractions = sorted(
        {str(row["dense_fraction_label"]) for row in audit_rows},
        key=lambda label: as_float(
            next(row for row in audit_rows if row["dense_fraction_label"] == label)[
                "dense_fraction"
            ]
        ),
    )
    with (output_root / f"{output_stem}.md").open("w") as handle:
        title = (
            "Selective Dense End-to-End Feasibility Audit"
            if end_to_end_only
            else "Selective Dense Feasibility Audit"
        )
        handle.write(f"# {title}\n\n")
        if end_to_end_only:
            handle.write(
                "Only complement backends are included here, so every measured row "
                "materializes one complete Linear output tensor.\n\n"
            )
        handle.write(
            "The ideal bound uses a dense-row cost of 1x and a sparse-row cost of "
            "0.5x. The proportional bound replaces the ideal 2x sparse speedup "
            "with the measured pure 2:4 speedup for the same shape.\n\n"
        )
        columns = [
            "dense_fraction",
            "cases",
            "measured_ge_1p4",
            "ideal_bound_lt_1p4",
            "pure24_bound_lt_1p4",
            "overhead_limited",
            "geomean_measured",
            "geomean_proportional_bound",
            "ideal_bound",
        ]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for label in fractions:
            group = [row for row in audit_rows if row["dense_fraction_label"] == label]
            measured = [row["mixed_speedup"] for row in group if row["mixed_speedup"] > 0]
            proportional = [
                row["proportional_bound_from_pure24"]
                for row in group
                if row["proportional_bound_from_pure24"] > 0
                and math.isfinite(row["proportional_bound_from_pure24"])
            ]
            status_counts = {
                status: sum(row["status"] == status for row in group)
                for status in {
                    "measured_ge_1p4",
                    "ideal_compute_bound_lt_1p4",
                    "measured_pure24_bound_lt_1p4",
                    "overhead_or_schedule_limited",
                }
            }
            ideal_bound = ideal_selective_speedup_bound(
                as_float(group[0]["dense_fraction"])
            )
            handle.write(
                f"| {label} | {len(group)} | "
                f"{status_counts['measured_ge_1p4']} | "
                f"{status_counts['ideal_compute_bound_lt_1p4']} | "
                f"{status_counts['measured_pure24_bound_lt_1p4']} | "
                f"{status_counts['overhead_or_schedule_limited']} | "
                f"{(statistics.geometric_mean(measured) if measured else float('nan')):.4f} | "
                f"{(statistics.geometric_mean(proportional) if proportional else float('nan')):.4f} | "
                f"{ideal_bound:.4f} |\n"
            )
        handle.write("\n## Worst Gaps To 1.4x\n\n")
        for row in sorted(audit_rows, key=lambda item: item["target_1p4_margin"])[:16]:
            handle.write(
                f"- {row['shape_label']} fraction={row['dense_fraction_label']} "
                f"speedup={row['mixed_speedup']:.4f}, "
                f"ideal_bound={row['ideal_mixed_speedup_bound']:.4f}, "
                f"proportional_bound={row['proportional_bound_from_pure24']:.4f}, "
                f"status={row['status']}\n"
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    x = list(range(len(fractions)))
    measured_values: list[float] = []
    proportional_values: list[float] = []
    ideal_values: list[float] = []
    for label in fractions:
        group = [row for row in audit_rows if row["dense_fraction_label"] == label]
        measured_values.append(
            statistics.geometric_mean(
                [row["mixed_speedup"] for row in group if row["mixed_speedup"] > 0]
            )
            if any(row["mixed_speedup"] > 0 for row in group)
            else float("nan")
        )
        positive_proportional = [
            row["proportional_bound_from_pure24"]
            for row in group
            if row["proportional_bound_from_pure24"] > 0
            and math.isfinite(row["proportional_bound_from_pure24"])
        ]
        proportional_values.append(
            statistics.geometric_mean(positive_proportional)
            if positive_proportional
            else float("nan")
        )
        ideal_values.append(
            ideal_selective_speedup_bound(as_float(group[0]["dense_fraction"]))
        )
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    measured_label = "measured e2e best" if end_to_end_only else "measured best"
    ax.bar([value - width for value in x], measured_values, width, label=measured_label)
    ax.bar(x, proportional_values, width, label="bound from pure 2:4")
    ax.bar([value + width for value in x], ideal_values, width, label="ideal 2x sparse bound")
    ax.axhline(1.4, color="#D62728", linestyle="--", linewidth=1.2, label="1.4x target")
    ax.set_xticks(x, fractions)
    ax.set_xlabel("dense fraction")
    ax.set_ylabel("speedup vs dense cuBLAS")
    ax.set_ylim(0.0, max(2.0, max(ideal_values) * 1.12))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    if end_to_end_only:
        fig.suptitle("Selective dense end-to-end feasibility")
    else:
        fig.suptitle("Selective dense feasibility vs measured and ideal bounds")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{output_stem}.png", dpi=200)
    plt.close(fig)


def metadata_for_run(
    args: argparse.Namespace,
    *,
    stamp: str,
) -> dict[str, Any]:
    device = torch.cuda.get_device_properties(0)
    return {
        "timestamp": stamp,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": {
            "name": device.name,
            "major": device.major,
            "minor": device.minor,
            "total_memory": device.total_memory,
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "measure_trials": args.measure_trials,
        "rtol": args.rtol,
        "atol": args.atol,
        "device_config_list": args.device_config_list,
        "pad_m_multiple_list": args.pad_m_multiple_list,
        "reuse_output": args.reuse_output,
        "preset": "custom" if args.shape else args.preset,
        "mlp_projections": args.mlp_projections,
        "m_list": args.m_list,
        "selective_dense_fractions": args.selective_dense_fractions,
        "selective_dense_counts": args.selective_dense_counts,
        "selective_dense_strategies": args.selective_dense_strategies,
        "row_selection": args.row_selection,
        "random_gather_backend": args.random_gather_backend,
        "sparse_mma_status": sparse_mma_status(),
        "isolate_shapes": args.isolate_shapes,
    }


def refresh_outputs(output_root: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2)
    )
    write_report(output_root / "report.md", rows, metadata)
    write_speedup_plot(output_root, rows)
    write_selective_dense_outputs(output_root, rows)
    write_selective_dense_end_to_end_outputs(output_root, rows)
    write_selective_dense_end_to_end_outputs(
        output_root,
        rows,
        row_selection_filter="random_sorted",
        output_prefix="selective_dense_random_e2e",
        title="Random Row Selective Dense End-to-End Best Backends",
        description=(
            "Random-row runs select the requested fraction of rows with a fixed seed, "
            "sort the selected indices for locality, gather those rows inside the timed "
            "region, then materialize one complete output tensor. The benchmark includes "
            "sparse-base plus 2:4 residual scatter-add, sparse-base plus dense "
            "selected-row override, and a split dense-selected/sparse-unselected path; "
            "the best row is selected per shape/fraction."
        ),
    )
    write_selective_dense_feasibility(output_root, rows)
    write_selective_dense_feasibility(
        output_root,
        rows,
        end_to_end_only=True,
        output_stem="selective_dense_e2e_feasibility",
    )
    write_selective_dense_feasibility(
        output_root,
        rows,
        end_to_end_only=True,
        output_stem="selective_dense_random_e2e_feasibility",
        row_selection_filter="random_sorted",
    )


def safe_shape_dir_name(shape_case: ShapeCase) -> str:
    return (
        shape_case.label.replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("=", "")
    )


def run_isolated_shape_cases(
    args: argparse.Namespace,
    shape_cases: list[ShapeCase],
    *,
    stamp: str,
) -> None:
    metadata = metadata_for_run(args, stamp=stamp)
    rows: list[dict[str, Any]] = []
    shape_root = args.output_root / "shape_runs"
    shape_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    for index, shape_case in enumerate(shape_cases):
        M, K, N = shape_case.shape
        case_root = shape_root / f"{index:02d}_{safe_shape_dir_name(shape_case)}"
        command = [
            sys.executable,
            str(script),
            "--shape",
            f"{M},{K},{N}",
            "--seed",
            str(args.seed + index),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--measure-trials",
            str(args.measure_trials),
            "--device-config-list",
            ",".join(args.device_config_list),
            "--pad-m-multiple-list",
            ",".join(str(value) for value in args.pad_m_multiple_list),
            "--selective-dense-fractions",
            ",".join(fraction_label(value) for value in args.selective_dense_fractions),
            "--selective-dense-counts",
            ",".join(str(value) for value in args.selective_dense_counts),
            "--selective-dense-strategies",
            ",".join(args.selective_dense_strategies),
            "--row-selection",
            args.row_selection,
            "--random-gather-backend",
            args.random_gather_backend,
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
            "--output-root",
            str(case_root),
        ]
        if args.reuse_output:
            command.append("--reuse-output")
        print(
            f"[isolated {index + 1:02d}/{len(shape_cases):02d}] "
            f"{shape_case.label} M={M} K={K} N={N}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"isolated shape run failed for {shape_case.label} "
                f"with return code {completed.returncode}"
            )
        case_summary = case_root / "summary.csv"
        case_rows = read_csv_rows(case_summary)
        for row in case_rows:
            row["shape_label"] = shape_case.label
        rows.extend(case_rows)
        refresh_outputs(args.output_root, rows, metadata)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Wrote {args.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help="Benchmark explicit shape as M,K,N. Can be repeated. When set, --preset is ignored.",
    )
    parser.add_argument(
        "--preset",
        choices=("legacy", "qwen3_8b", "llama3_1_8b", "serving", "all"),
        default="legacy",
    )
    parser.add_argument(
        "--mlp-projections",
        choices=("fused", "split", "both"),
        default="fused",
    )
    parser.add_argument("--m-list", type=parse_int_list, default=parse_int_list("1,8,32,128"))
    parser.add_argument(
        "--device-config-list",
        type=parse_str_list,
        default=parse_str_list("auto"),
        help=(
            "Comma-separated CUTLASS SparseGemm configs. Use auto for the final "
            "measured policy."
        ),
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Preallocate dense/sparse output buffers and the sparse transpose workspace.",
    )
    parser.add_argument(
        "--pad-m-multiple-list",
        type=parse_int_list,
        default=parse_int_list("8"),
        help="Comma-separated M padding multiples to sweep; each must be a multiple of 8.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--measure-trials",
        type=int,
        default=5,
        help="Run each timing measurement this many times and report the median.",
    )
    parser.add_argument(
        "--selective-dense-fractions",
        type=parse_fraction_list,
        default=[],
        help=(
            "Optional comma-separated dense-row fractions for mixed selective-dense "
            "runs, e.g. 1/8,1/4,1/2."
        ),
    )
    parser.add_argument(
        "--selective-dense-counts",
        type=parse_int_list,
        default=[],
        help=(
            "Optional exact dense row counts. Each count is converted to a "
            "per-shape fraction, so 16,32,64,128 can be swept across model "
            "linear shapes."
        ),
    )
    parser.add_argument(
        "--selective-dense-strategies",
        type=parse_strategy_list,
        default=parse_strategy_list(
            "full_sparse_residual,full_sparse_dense_override,split_dense_sparse"
        ),
        help=(
            "Comma-separated mixed linear strategies to run: "
            "full_sparse_residual, full_sparse_dense_override, "
            "split_dense_sparse, or all."
        ),
    )
    parser.add_argument(
        "--row-selection",
        choices=("prefix", "random"),
        default="prefix",
        help=(
            "Dense/residual row selection policy for --selective-dense-fractions. "
            "random chooses a fixed random subset and sorts the indices before "
            "gather/scatter; the random sampling itself is not timed."
        ),
    )
    parser.add_argument(
        "--random-gather-backend",
        choices=("cutlass", "torch"),
        default="cutlass",
        help="Gather implementation for --row-selection random. cutlass uses the local vectorized CUDA kernel.",
    )
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--isolate-shapes",
        action="store_true",
        help=(
            "Run each shape in a child process and combine outputs. This avoids "
            "CUDA graph private-pool accumulation during graph-heavy sweeps."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark; run with real GPU access")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_root is None:
        args.output_root = (
            REPO_ROOT
            / "examples/evaluate/eval-guidellm/temp"
            / f"cutlass_sparse24_linear_bench_{stamp}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.shape:
        shape_cases = [
            ShapeCase(f"custom_{M}x{K}x{N}", (M, K, N))
            for M, K, N in args.shape
        ]
    else:
        shape_cases = preset_shape_cases(
            args.preset,
            args.m_list,
            mlp_projections=args.mlp_projections,
        )

    if args.isolate_shapes and len(shape_cases) > 1:
        run_isolated_shape_cases(args, shape_cases, stamp=stamp)
        return

    metadata = metadata_for_run(args, stamp=stamp)

    rows: list[dict[str, Any]] = []
    for index, shape_case in enumerate(shape_cases):
        M, K, N = shape_case.shape
        print(f"[{index + 1:02d}] {shape_case.label} M={M} K={K} N={N}")
        case_rows = run_case(
            shape_case,
            seed=args.seed + index,
            warmup=args.warmup,
            repeat=args.repeat,
            measure_trials=args.measure_trials,
            device_config_values=args.device_config_list,
            pad_m_multiple_values=args.pad_m_multiple_list,
            selective_dense_fractions=selective_dense_fractions_for_shape(args, M),
            reuse_output=args.reuse_output,
            row_selection=args.row_selection,
            random_gather_backend=args.random_gather_backend,
            selective_dense_strategies=args.selective_dense_strategies,
            rtol=args.rtol,
            atol=args.atol,
        )
        rows.extend(case_rows)
        refresh_outputs(args.output_root, rows, metadata)
        for row in case_rows:
            print(
                "  backend={backend} config={device_config} "
                "pad_m={pad_m_multiple} "
                "dense={dense_ms:.4f}ms sparse={sparse_ms:.4f}ms "
                "speedup={speedup:.4f} pass={pass}".format(**row)
            )
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Wrote {args.output_root}")


if __name__ == "__main__":
    main()
