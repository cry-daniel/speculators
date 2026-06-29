#!/usr/bin/env python3
"""Microbenchmark and profiler probe for the SR24 torch_sparse backend."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional benchmark dependency
    triton = None
    tl = None


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@contextmanager
def sparse_backend_overrides(
    *,
    force_cutlass: bool | None = None,
    alg_id: int | None = None,
    fuse_transpose: bool | None = None,
):
    """Temporarily override PyTorch semi-structured sparse backend knobs."""
    from torch.sparse import SparseSemiStructuredTensor

    old_force_cutlass = getattr(
        SparseSemiStructuredTensor, "_FORCE_CUTLASS", None
    )
    old_alg_id = getattr(SparseSemiStructuredTensor, "_DEFAULT_ALG_ID", None)
    old_fuse_transpose = getattr(
        SparseSemiStructuredTensor, "_FUSE_TRANSPOSE", None
    )
    if force_cutlass is not None and old_force_cutlass is not None:
        SparseSemiStructuredTensor._FORCE_CUTLASS = bool(force_cutlass)
    if alg_id is not None and old_alg_id is not None:
        SparseSemiStructuredTensor._DEFAULT_ALG_ID = int(alg_id)
    if fuse_transpose is not None and old_fuse_transpose is not None:
        SparseSemiStructuredTensor._FUSE_TRANSPOSE = bool(fuse_transpose)
    try:
        yield SparseSemiStructuredTensor
    finally:
        if old_force_cutlass is not None:
            SparseSemiStructuredTensor._FORCE_CUTLASS = old_force_cutlass
        if old_alg_id is not None:
            SparseSemiStructuredTensor._DEFAULT_ALG_ID = old_alg_id
        if old_fuse_transpose is not None:
            SparseSemiStructuredTensor._FUSE_TRANSPOSE = old_fuse_transpose


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be ROWS,OUT,IN")
    return parts[0], parts[1], parts[2]


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("tile sizes must be positive")
    return values


def parse_optional_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"default", "auto", "none"}:
        return None
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected default/auto/none, true/false, or 1/0"
    )


def make_complementary_24(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    if in_features % 4:
        raise ValueError("in_features must be divisible by 4")
    grouped = weight.view(out_features, in_features // 4, 4)
    idx = grouped.abs().topk(2, dim=-1).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool)
    keep.scatter_(-1, idx, True)
    base = torch.zeros_like(grouped)
    residual = torch.zeros_like(grouped)
    base[keep] = grouped[keep]
    residual[~keep] = grouped[~keep]
    residual_values = grouped[~keep].contiguous()
    return (
        base.view_as(weight).contiguous(),
        residual.view_as(weight).contiguous(),
        keep,
        residual_values,
    )


def pack_keep_mask(keep: torch.Tensor) -> torch.Tensor:
    bits = torch.tensor([1, 2, 4, 8], dtype=torch.uint8, device=keep.device)
    group_bytes = (
        keep.to(torch.uint8) * bits.view(1, 1, 4)
    ).sum(dim=-1).to(torch.uint8)
    groups = int(group_bytes.shape[1])
    if groups % 2:
        pad = torch.zeros(
            (int(group_bytes.shape[0]), 1),
            dtype=torch.uint8,
            device=group_bytes.device,
        )
        group_bytes = torch.cat([group_bytes, pad], dim=1)
    return (group_bytes[:, 0::2] | (group_bytes[:, 1::2] << 4)).contiguous()


def expand_keep_mask(
    mask_bytes: torch.Tensor,
    *,
    out_features: int,
    groups: int,
    device: torch.device,
) -> torch.Tensor:
    packed = mask_bytes.to(device=device, dtype=torch.uint8, non_blocking=True)
    unpacked = torch.empty(
        (out_features, packed.shape[1] * 2),
        dtype=torch.uint8,
        device=device,
    )
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    bits = torch.tensor([1, 2, 4, 8], dtype=torch.uint8, device=device)
    return (unpacked[:, :groups].unsqueeze(-1) & bits.view(1, 1, 4)).ne(0)


def compressed_residual_weight(
    *,
    mask_bytes: torch.Tensor,
    residual_values: torch.Tensor,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    groups = in_features // 4
    keep = expand_keep_mask(
        mask_bytes,
        out_features=out_features,
        groups=groups,
        device=device,
    )
    values = residual_values.to(device=device, dtype=dtype, non_blocking=True)
    residual_weight = torch.zeros(
        (out_features, in_features),
        device=device,
        dtype=dtype,
    )
    residual_weight.view(out_features, groups, 4)[~keep] = values
    return residual_weight


def residual_position_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos0: list[int] = []
    pos1: list[int] = []
    for bits in range(16):
        positions = [idx for idx in range(4) if bits & (1 << idx)]
        if len(positions) < 2:
            positions = (positions + [0, 0])[:2]
        pos0.append(positions[0])
        pos1.append(positions[1])
    return (
        torch.tensor(pos0, dtype=torch.int32, device=device),
        torch.tensor(pos1, dtype=torch.int32, device=device),
    )


def residual_positions_from_keep(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Predecode complementary 2:4 positions into GPU-resident tensors.

    This is setup-time work for the microbench. The serving hot path should be
    able to reuse these tiny position tensors instead of decoding packed masks
    inside every residual matmul tile.
    """
    if keep.ndim != 3 or keep.shape[-1] != 4:
        raise ValueError("keep must have shape [out_features, groups, 4]")
    positions = torch.arange(4, dtype=torch.uint8, device=keep.device).view(1, 1, 4)
    residual_positions = positions.expand_as(keep)[torch.logical_not(keep)]
    residual_positions = residual_positions.view(keep.shape[0], keep.shape[1], 2)
    return (
        residual_positions[:, :, 0].contiguous(),
        residual_positions[:, :, 1].contiguous(),
    )


def residual_absolute_positions_from_keep(
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predecode complementary positions into absolute K indices.

    This removes the per-tile `group * 4 + pos` arithmetic from the Triton
    residual matmul and measures whether address computation, rather than the
    irregular residual gather pattern itself, is the main bottleneck.
    """
    pos0, pos1 = residual_positions_from_keep(keep)
    groups = int(keep.shape[1])
    base = (torch.arange(groups, dtype=torch.int32, device=keep.device) * 4).view(
        1, groups
    )
    return (
        (pos0.to(torch.int32) + base).contiguous(),
        (pos1.to(torch.int32) + base).contiguous(),
    )


if triton is not None:

    @triton.jit
    def _residual24_linear_kernel(
        x_ptr,
        residual_values_ptr,
        mask_bytes_ptr,
        out_ptr,
        pos0_lut_ptr,
        pos1_lut_ptr,
        m_size,
        N: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        PACKED_G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_MN: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_mn = tl.arange(0, BLOCK_MN)
        offs_m = pid_m * BLOCK_M + offs_mn // BLOCK_N
        offs_n = pid_n * BLOCK_N + offs_mn % BLOCK_N
        acc = tl.zeros((BLOCK_MN,), dtype=tl.float32)

        for group_start in tl.range(0, G, BLOCK_G):
            offs_g = group_start + tl.arange(0, BLOCK_G)
            valid = (
                (offs_m[:, None] < m_size)
                & (offs_n[:, None] < N)
                & (offs_g[None, :] < G)
            )
            packed = tl.load(
                mask_bytes_ptr
                + offs_n[:, None] * PACKED_G
                + (offs_g[None, :] // 2),
                mask=valid,
                other=0,
            )
            group_byte = tl.where(
                (offs_g[None, :] & 1) == 0,
                packed & 0x0F,
                (packed >> 4) & 0x0F,
            )
            residual_bits = group_byte ^ 0x0F
            pos0 = tl.load(pos0_lut_ptr + residual_bits)
            pos1 = tl.load(pos1_lut_ptr + residual_bits)
            x0 = tl.load(
                x_ptr + offs_m[:, None] * K + offs_g[None, :] * 4 + pos0,
                mask=valid,
                other=0.0,
            )
            x1 = tl.load(
                x_ptr + offs_m[:, None] * K + offs_g[None, :] * 4 + pos1,
                mask=valid,
                other=0.0,
            )
            val_base = offs_n[:, None] * (G * 2) + offs_g[None, :] * 2
            v0 = tl.load(
                residual_values_ptr + val_base,
                mask=valid,
                other=0.0,
            )
            v1 = tl.load(
                residual_values_ptr + val_base + 1,
                mask=valid,
                other=0.0,
            )
            acc += tl.sum(x0.to(tl.float32) * v0 + x1.to(tl.float32) * v1, axis=1)

        tl.store(
            out_ptr + offs_m * N + offs_n,
            acc,
            mask=(offs_m < m_size) & (offs_n < N),
        )

    @triton.jit
    def _residual24_linear_tiled_kernel(
        x_ptr,
        residual_values_ptr,
        mask_bytes_ptr,
        out_ptr,
        pos0_lut_ptr,
        pos1_lut_ptr,
        m_size,
        N: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        PACKED_G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for group_start in tl.range(0, G, BLOCK_G):
            offs_g = group_start + tl.arange(0, BLOCK_G)
            ng_valid = (offs_n[:, None] < N) & (offs_g[None, :] < G)
            packed = tl.load(
                mask_bytes_ptr
                + offs_n[:, None] * PACKED_G
                + (offs_g[None, :] // 2),
                mask=ng_valid,
                other=0,
            )
            group_byte = tl.where(
                (offs_g[None, :] & 1) == 0,
                packed & 0x0F,
                (packed >> 4) & 0x0F,
            )
            residual_bits = group_byte ^ 0x0F
            pos0 = tl.load(pos0_lut_ptr + residual_bits)
            pos1 = tl.load(pos1_lut_ptr + residual_bits)
            val_base = offs_n[:, None] * (G * 2) + offs_g[None, :] * 2
            v0 = tl.load(
                residual_values_ptr + val_base,
                mask=ng_valid,
                other=0.0,
            )
            v1 = tl.load(
                residual_values_ptr + val_base + 1,
                mask=ng_valid,
                other=0.0,
            )

            x_valid = (offs_m[:, None, None] < m_size) & ng_valid[None, :, :]
            x0 = tl.load(
                x_ptr
                + offs_m[:, None, None] * K
                + offs_g[None, None, :] * 4
                + pos0[None, :, :],
                mask=x_valid,
                other=0.0,
            )
            x1 = tl.load(
                x_ptr
                + offs_m[:, None, None] * K
                + offs_g[None, None, :] * 4
                + pos1[None, :, :],
                mask=x_valid,
                other=0.0,
            )
            prod = (
                x0.to(tl.float32) * v0[None, :, :]
                + x1.to(tl.float32) * v1[None, :, :]
            )
            acc += tl.sum(prod, axis=2)

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _residual24_linear_pos_tiled_kernel(
        x_ptr,
        residual_values_ptr,
        pos0_ptr,
        pos1_ptr,
        out_ptr,
        m_size,
        N: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for group_start in tl.range(0, G, BLOCK_G):
            offs_g = group_start + tl.arange(0, BLOCK_G)
            ng_valid = (offs_n[:, None] < N) & (offs_g[None, :] < G)
            pos_base = offs_n[:, None] * G + offs_g[None, :]
            pos0 = tl.load(pos0_ptr + pos_base, mask=ng_valid, other=0).to(tl.int32)
            pos1 = tl.load(pos1_ptr + pos_base, mask=ng_valid, other=0).to(tl.int32)
            val_base = offs_n[:, None] * (G * 2) + offs_g[None, :] * 2
            v0 = tl.load(
                residual_values_ptr + val_base,
                mask=ng_valid,
                other=0.0,
            )
            v1 = tl.load(
                residual_values_ptr + val_base + 1,
                mask=ng_valid,
                other=0.0,
            )

            x_valid = (offs_m[:, None, None] < m_size) & ng_valid[None, :, :]
            x0 = tl.load(
                x_ptr
                + offs_m[:, None, None] * K
                + offs_g[None, None, :] * 4
                + pos0[None, :, :],
                mask=x_valid,
                other=0.0,
            )
            x1 = tl.load(
                x_ptr
                + offs_m[:, None, None] * K
                + offs_g[None, None, :] * 4
                + pos1[None, :, :],
                mask=x_valid,
                other=0.0,
            )
            prod = (
                x0.to(tl.float32) * v0[None, :, :]
                + x1.to(tl.float32) * v1[None, :, :]
            )
            acc += tl.sum(prod, axis=2)

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _residual24_linear_abspos_tiled_kernel(
        x_ptr,
        residual_values_ptr,
        k0_ptr,
        k1_ptr,
        out_ptr,
        m_size,
        N: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
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
            val_base = offs_n[:, None] * (G * 2) + offs_g[None, :] * 2
            v0 = tl.load(
                residual_values_ptr + val_base,
                mask=ng_valid,
                other=0.0,
            )
            v1 = tl.load(
                residual_values_ptr + val_base + 1,
                mask=ng_valid,
                other=0.0,
            )

            x_valid = (offs_m[:, None, None] < m_size) & ng_valid[None, :, :]
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
            prod = (
                x0.to(tl.float32) * v0[None, :, :]
                + x1.to(tl.float32) * v1[None, :, :]
            )
            acc += tl.sum(prod, axis=2)

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < N),
        )


def triton_residual24_linear(
    x: torch.Tensor,
    *,
    mask_bytes: torch.Tensor,
    residual_values: torch.Tensor,
    out_features: int,
    in_features: int,
    block_m: int = 16,
    block_n: int = 16,
    block_g: int = 32,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    rows = int(x.shape[0])
    groups = in_features // 4
    packed_groups = (groups + 1) // 2
    if tuple(mask_bytes.shape) != (out_features, packed_groups):
        raise RuntimeError(
            f"mask_bytes shape {tuple(mask_bytes.shape)} does not match "
            f"{(out_features, packed_groups)}"
        )
    values = residual_values.to(device=x.device, dtype=x.dtype, non_blocking=True)
    masks = mask_bytes.to(device=x.device, dtype=torch.uint8, non_blocking=True)
    pos0, pos1 = residual_position_luts(x.device)
    out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _residual24_linear_kernel[grid](
        x,
        values,
        masks,
        out,
        pos0,
        pos1,
        rows,
        out_features,
        in_features,
        groups,
        packed_groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_MN=block_m * block_n,
        BLOCK_G=block_g,
    )
    return out


def triton_residual24_linear_tiled(
    x: torch.Tensor,
    *,
    mask_bytes: torch.Tensor,
    residual_values: torch.Tensor,
    out_features: int,
    in_features: int,
    block_m: int = 16,
    block_n: int = 16,
    block_g: int = 32,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    rows = int(x.shape[0])
    groups = in_features // 4
    packed_groups = (groups + 1) // 2
    if tuple(mask_bytes.shape) != (out_features, packed_groups):
        raise RuntimeError(
            f"mask_bytes shape {tuple(mask_bytes.shape)} does not match "
            f"{(out_features, packed_groups)}"
        )
    values = residual_values.to(device=x.device, dtype=x.dtype, non_blocking=True)
    masks = mask_bytes.to(device=x.device, dtype=torch.uint8, non_blocking=True)
    pos0, pos1 = residual_position_luts(x.device)
    out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _residual24_linear_tiled_kernel[grid](
        x,
        values,
        masks,
        out,
        pos0,
        pos1,
        rows,
        out_features,
        in_features,
        groups,
        packed_groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_G=block_g,
        num_warps=4,
    )
    return out


def triton_residual24_linear_pos_tiled(
    x: torch.Tensor,
    *,
    pos0: torch.Tensor,
    pos1: torch.Tensor,
    residual_values: torch.Tensor,
    out_features: int,
    in_features: int,
    block_m: int = 16,
    block_n: int = 16,
    block_g: int = 32,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    rows = int(x.shape[0])
    groups = in_features // 4
    if tuple(pos0.shape) != (out_features, groups):
        raise RuntimeError(
            f"pos0 shape {tuple(pos0.shape)} does not match {(out_features, groups)}"
        )
    if tuple(pos1.shape) != (out_features, groups):
        raise RuntimeError(
            f"pos1 shape {tuple(pos1.shape)} does not match {(out_features, groups)}"
        )
    values = residual_values.to(device=x.device, dtype=x.dtype, non_blocking=True)
    pos0_cuda = pos0.to(device=x.device, dtype=torch.uint8, non_blocking=True)
    pos1_cuda = pos1.to(device=x.device, dtype=torch.uint8, non_blocking=True)
    out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _residual24_linear_pos_tiled_kernel[grid](
        x,
        values,
        pos0_cuda,
        pos1_cuda,
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


def triton_residual24_linear_abspos_tiled(
    x: torch.Tensor,
    *,
    k0: torch.Tensor,
    k1: torch.Tensor,
    residual_values: torch.Tensor,
    out_features: int,
    in_features: int,
    block_m: int = 16,
    block_n: int = 16,
    block_g: int = 32,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    rows = int(x.shape[0])
    groups = in_features // 4
    if tuple(k0.shape) != (out_features, groups):
        raise RuntimeError(
            f"k0 shape {tuple(k0.shape)} does not match {(out_features, groups)}"
        )
    if tuple(k1.shape) != (out_features, groups):
        raise RuntimeError(
            f"k1 shape {tuple(k1.shape)} does not match {(out_features, groups)}"
        )
    values = residual_values.to(device=x.device, dtype=x.dtype, non_blocking=True)
    k0_cuda = k0.to(device=x.device, dtype=torch.int32, non_blocking=True)
    k1_cuda = k1.to(device=x.device, dtype=torch.int32, non_blocking=True)
    out = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _residual24_linear_abspos_tiled_kernel[grid](
        x,
        values,
        k0_cuda,
        k1_cuda,
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


def time_call(fn, *, warmup: int, repeats: int) -> float:
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


def call_tensor_or_error(fn) -> tuple[torch.Tensor | None, str | None]:
    try:
        out = fn()
        torch.cuda.synchronize()
        return out, None
    except Exception as exc:
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def time_call_or_error(fn, *, warmup: int, repeats: int) -> tuple[float | None, str | None]:
    try:
        return time_call(fn, warmup=warmup, repeats=repeats), None
    except Exception as exc:
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def time_cuda_graph_call(
    fn,
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
    except Exception as exc:
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def private_sparse_mm_linear(x: torch.Tensor, sparse_weight: Any) -> torch.Tensor:
    padded_x = sparse_weight._pad_dense_input(x)
    out = sparse_weight._mm(padded_x.t().contiguous()).t()
    return out[: x.shape[0], :]


def direct_cslt_linear(
    x: torch.Tensor,
    sparse_weight: Any,
    *,
    alg_id: int = 0,
) -> torch.Tensor:
    packed = getattr(sparse_weight, "packed", None)
    if packed is None:
        raise RuntimeError(
            f"{type(sparse_weight).__name__} does not expose cuSPARSELt packed weight"
        )
    padded_x = sparse_weight._pad_dense_input(x)
    out = torch._cslt_sparse_mm(
        packed,
        padded_x.t().contiguous(),
        transpose_result=False,
        alg_id=alg_id,
    ).t()
    return out[: x.shape[0], :]


def profile_sparse_call(x: torch.Tensor, sparse_weight: Any) -> list[dict[str, Any]]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        F.linear(x, sparse_weight)
        torch.cuda.synchronize()
    events = []
    for item in prof.key_averages():
        key = str(item.key)
        if any(token in key.lower() for token in ("sparse", "cusparse", "cslt", "semi")):
            events.append(
                {
                    "key": key,
                    "cpu_time_total_us": float(item.cpu_time_total),
                    "cuda_time_total_us": float(
                        getattr(
                            item,
                            "cuda_time_total",
                            getattr(item, "device_time_total", 0.0),
                        )
                    ),
                    "count": int(item.count),
                    "input_shapes": str(item.input_shapes),
                }
            )
    return events


def semi_structured_storage_bytes(
    tensor: Any,
    *,
    logical_value_bytes: int,
) -> tuple[int, int]:
    value_bytes = 0
    meta_bytes = 0
    seen: set[int] = set()

    def add_tensor(item: Any, *, metadata: bool) -> None:
        nonlocal value_bytes, meta_bytes
        if not isinstance(item, torch.Tensor):
            return
        ptr = int(item.untyped_storage().data_ptr())
        if ptr in seen:
            return
        seen.add(ptr)
        size = int(item.untyped_storage().nbytes())
        if metadata:
            meta_bytes += size
        else:
            value_bytes += size

    for attr in ("values", "packed", "packed_t"):
        try:
            item = getattr(tensor, attr)
            add_tensor(item() if callable(item) else item, metadata=False)
        except Exception:
            pass
    for attr in ("indices", "meta", "meta_t"):
        try:
            item = getattr(tensor, attr)
            add_tensor(item() if callable(item) else item, metadata=True)
        except Exception:
            pass
    if meta_bytes == 0 and value_bytes > logical_value_bytes:
        meta_bytes = value_bytes - logical_value_bytes
        value_bytes = logical_value_bytes
    return value_bytes, meta_bytes


def run_case(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    skip_triton_residual: bool = False,
) -> dict[str, Any]:
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    dense_weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
    base_weight, residual_weight, keep, residual_values = make_complementary_24(
        dense_weight
    )
    base_sparse = to_sparse_semi_structured(base_weight)
    residual_sparse = to_sparse_semi_structured(residual_weight)
    from torch.sparse import SparseSemiStructuredTensor

    sparse_force_cutlass = getattr(
        SparseSemiStructuredTensor, "_FORCE_CUTLASS", None
    )
    sparse_default_alg_id = getattr(
        SparseSemiStructuredTensor, "_DEFAULT_ALG_ID", None
    )
    sparse_fuse_transpose = getattr(
        SparseSemiStructuredTensor, "_FUSE_TRANSPOSE", None
    )
    mask_bytes_cpu = pack_keep_mask(keep.detach()).cpu()
    mask_bytes_cuda = mask_bytes_cpu.to(device="cuda", non_blocking=True)
    residual_values_cpu = residual_values.detach().cpu()
    residual_values_cuda = residual_values.detach().contiguous()
    residual_pos0_cuda, residual_pos1_cuda = residual_positions_from_keep(keep.detach())
    residual_k0_cuda, residual_k1_cuda = residual_absolute_positions_from_keep(
        keep.detach()
    )
    cached_residual_weight_cuda = compressed_residual_weight(
        mask_bytes=mask_bytes_cuda,
        residual_values=residual_values_cuda,
        out_features=out_features,
        in_features=in_features,
        dtype=dtype,
        device=x.device,
    )
    torch.cuda.synchronize()

    dense = lambda: F.linear(x, dense_weight)
    base_dense = lambda: F.linear(x, base_weight)
    base_sparse_call = lambda: F.linear(x, base_sparse)
    base_private_mm_call = lambda: private_sparse_mm_linear(x, base_sparse)
    base_direct_cslt_alg0 = lambda: direct_cslt_linear(x, base_sparse, alg_id=0)
    base_direct_cslt_alg1 = lambda: direct_cslt_linear(x, base_sparse, alg_id=1)
    all_corrected_sparse = lambda: F.linear(x, base_sparse) + F.linear(x, residual_sparse)
    all_corrected_direct_cslt_alg0 = lambda: (
        direct_cslt_linear(x, base_sparse, alg_id=0)
        + direct_cslt_linear(x, residual_sparse, alg_id=0)
    )
    all_corrected_direct_cslt_alg1 = lambda: (
        direct_cslt_linear(x, base_sparse, alg_id=1)
        + direct_cslt_linear(x, residual_sparse, alg_id=1)
    )
    dense_residual = lambda: F.linear(x, residual_weight)
    compressed_residual_cpu = lambda: F.linear(
        x,
        compressed_residual_weight(
            mask_bytes=mask_bytes_cpu,
            residual_values=residual_values_cpu,
            out_features=out_features,
            in_features=in_features,
            dtype=dtype,
            device=x.device,
        ),
    )
    compressed_residual_cuda = lambda: F.linear(
        x,
        compressed_residual_weight(
            mask_bytes=mask_bytes_cuda,
            residual_values=residual_values_cuda,
            out_features=out_features,
            in_features=in_features,
            dtype=dtype,
            device=x.device,
        ),
    )
    compressed_residual_cached_cuda = lambda: F.linear(x, cached_residual_weight_cuda)
    triton_residual = (
        lambda: triton_residual24_linear(
            x,
            mask_bytes=mask_bytes_cuda,
            residual_values=residual_values_cuda,
            out_features=out_features,
            in_features=in_features,
        )
        if triton is not None
        else None
    )
    triton_tiled_residual = (
        lambda: triton_residual24_linear_tiled(
            x,
            mask_bytes=mask_bytes_cuda,
            residual_values=residual_values_cuda,
            out_features=out_features,
            in_features=in_features,
        )
        if triton is not None
        else None
    )
    triton_pos_tiled_residual = (
        lambda: triton_residual24_linear_pos_tiled(
            x,
            pos0=residual_pos0_cuda,
            pos1=residual_pos1_cuda,
            residual_values=residual_values_cuda,
            out_features=out_features,
            in_features=in_features,
        )
        if triton is not None
        else None
    )
    triton_abspos_tiled_residual = (
        lambda: triton_residual24_linear_abspos_tiled(
            x,
            k0=residual_k0_cuda,
            k1=residual_k1_cuda,
            residual_values=residual_values_cuda,
            out_features=out_features,
            in_features=in_features,
        )
        if triton is not None
        else None
    )
    all_corrected_compressed_cpu = lambda: F.linear(
        x, base_sparse
    ) + compressed_residual_cpu()
    all_corrected_compressed_cuda = lambda: F.linear(
        x, base_sparse
    ) + compressed_residual_cuda()
    all_corrected_compressed_cached_cuda = lambda: (
        F.linear(x, base_sparse) + compressed_residual_cached_cuda()
    )
    all_corrected_triton_residual = (
        lambda: F.linear(x, base_sparse) + triton_residual()
        if triton is not None
        else None
    )
    all_corrected_triton_tiled_residual = (
        lambda: F.linear(x, base_sparse) + triton_tiled_residual()
        if triton is not None
        else None
    )
    all_corrected_triton_pos_tiled_residual = (
        lambda: F.linear(x, base_sparse) + triton_pos_tiled_residual()
        if triton is not None
        else None
    )
    all_corrected_triton_abspos_tiled_residual = (
        lambda: F.linear(x, base_sparse) + triton_abspos_tiled_residual()
        if triton is not None
        else None
    )

    dense_out = dense()
    corrected_out = all_corrected_sparse()
    private_mm_out = base_private_mm_call()
    direct_cslt_alg0_out, direct_cslt_alg0_error = call_tensor_or_error(
        base_direct_cslt_alg0
    )
    direct_cslt_alg1_out, direct_cslt_alg1_error = call_tensor_or_error(
        base_direct_cslt_alg1
    )
    all_corrected_direct_cslt_alg0_out, all_corrected_direct_cslt_alg0_error = (
        call_tensor_or_error(all_corrected_direct_cslt_alg0)
    )
    all_corrected_direct_cslt_alg1_out, all_corrected_direct_cslt_alg1_error = (
        call_tensor_or_error(all_corrected_direct_cslt_alg1)
    )
    compressed_cuda_out = F.linear(x, base_sparse) + compressed_residual_cuda()
    if skip_triton_residual:
        triton_residual_out = None
        triton_corrected_out = None
        triton_tiled_residual_out = None
        triton_tiled_corrected_out = None
        triton_pos_tiled_residual_out = None
        triton_pos_tiled_corrected_out = None
        triton_abspos_tiled_residual_out = None
        triton_abspos_tiled_corrected_out = None
        triton_tiled_error = "skipped"
        triton_pos_tiled_error = "skipped"
        triton_abspos_tiled_error = "skipped"
    else:
        triton_residual_out = triton_residual() if triton is not None else None
        triton_corrected_out = (
            F.linear(x, base_sparse) + triton_residual_out
            if triton_residual_out is not None
            else None
        )
        triton_tiled_residual_out, triton_tiled_error = (
            call_tensor_or_error(triton_tiled_residual)
            if triton is not None
            else (None, None)
        )
        triton_tiled_corrected_out = (
            F.linear(x, base_sparse) + triton_tiled_residual_out
            if triton_tiled_residual_out is not None
            else None
        )
        triton_pos_tiled_residual_out, triton_pos_tiled_error = (
            call_tensor_or_error(triton_pos_tiled_residual)
            if triton is not None
            else (None, None)
        )
        triton_pos_tiled_corrected_out = (
            F.linear(x, base_sparse) + triton_pos_tiled_residual_out
            if triton_pos_tiled_residual_out is not None
            else None
        )
        triton_abspos_tiled_residual_out, triton_abspos_tiled_error = (
            call_tensor_or_error(triton_abspos_tiled_residual)
            if triton is not None
            else (None, None)
        )
        triton_abspos_tiled_corrected_out = (
            F.linear(x, base_sparse) + triton_abspos_tiled_residual_out
            if triton_abspos_tiled_residual_out is not None
            else None
        )
    torch.cuda.synchronize()
    max_abs_diff = float((dense_out - corrected_out).abs().max().item())
    mean_abs_diff = float((dense_out - corrected_out).abs().mean().item())
    compressed_max_abs_diff = float(
        (dense_out - compressed_cuda_out).abs().max().item()
    )
    private_mm_max_abs_diff = float(
        (base_dense() - private_mm_out).abs().max().item()
    )
    direct_cslt_alg0_max_abs_diff = (
        float((base_dense() - direct_cslt_alg0_out).abs().max().item())
        if direct_cslt_alg0_out is not None
        else None
    )
    direct_cslt_alg1_max_abs_diff = (
        float((base_dense() - direct_cslt_alg1_out).abs().max().item())
        if direct_cslt_alg1_out is not None
        else None
    )
    all_corrected_direct_cslt_alg0_max_abs_diff = (
        float((dense_out - all_corrected_direct_cslt_alg0_out).abs().max().item())
        if all_corrected_direct_cslt_alg0_out is not None
        else None
    )
    all_corrected_direct_cslt_alg1_max_abs_diff = (
        float((dense_out - all_corrected_direct_cslt_alg1_out).abs().max().item())
        if all_corrected_direct_cslt_alg1_out is not None
        else None
    )
    triton_residual_max_abs_diff = (
        float((dense_residual() - triton_residual_out).abs().max().item())
        if triton_residual_out is not None
        else None
    )
    triton_corrected_max_abs_diff = (
        float((dense_out - triton_corrected_out).abs().max().item())
        if triton_corrected_out is not None
        else None
    )
    triton_tiled_residual_max_abs_diff = (
        float((dense_residual() - triton_tiled_residual_out).abs().max().item())
        if triton_tiled_residual_out is not None
        else None
    )
    triton_tiled_corrected_max_abs_diff = (
        float((dense_out - triton_tiled_corrected_out).abs().max().item())
        if triton_tiled_corrected_out is not None
        else None
    )
    triton_pos_tiled_residual_max_abs_diff = (
        float((dense_residual() - triton_pos_tiled_residual_out).abs().max().item())
        if triton_pos_tiled_residual_out is not None
        else None
    )
    triton_pos_tiled_corrected_max_abs_diff = (
        float((dense_out - triton_pos_tiled_corrected_out).abs().max().item())
        if triton_pos_tiled_corrected_out is not None
        else None
    )
    triton_abspos_tiled_residual_max_abs_diff = (
        float((dense_residual() - triton_abspos_tiled_residual_out).abs().max().item())
        if triton_abspos_tiled_residual_out is not None
        else None
    )
    triton_abspos_tiled_corrected_max_abs_diff = (
        float((dense_out - triton_abspos_tiled_corrected_out).abs().max().item())
        if triton_abspos_tiled_corrected_out is not None
        else None
    )

    logical_value_bytes = int(
        dense_weight.numel() // 2 * dense_weight.element_size()
    )
    base_value_bytes, base_meta_bytes = semi_structured_storage_bytes(
        base_sparse,
        logical_value_bytes=logical_value_bytes,
    )
    residual_value_bytes, residual_meta_bytes = semi_structured_storage_bytes(
        residual_sparse,
        logical_value_bytes=logical_value_bytes,
    )
    base_sparse_graph_ms, base_sparse_graph_error = time_cuda_graph_call(
        base_sparse_call, warmup=warmup, repeats=repeats
    )
    dense_graph_ms, dense_graph_error = time_cuda_graph_call(
        dense, warmup=warmup, repeats=repeats
    )
    private_mm_graph_ms, private_mm_graph_error = time_cuda_graph_call(
        base_private_mm_call, warmup=warmup, repeats=repeats
    )
    direct_cslt_alg0_graph_ms, direct_cslt_alg0_graph_error = time_cuda_graph_call(
        base_direct_cslt_alg0, warmup=warmup, repeats=repeats
    )
    direct_cslt_alg1_graph_ms, direct_cslt_alg1_graph_error = time_cuda_graph_call(
        base_direct_cslt_alg1, warmup=warmup, repeats=repeats
    )
    direct_cslt_alg0_ms, direct_cslt_alg0_time_error = (
        time_call_or_error(
            base_direct_cslt_alg0,
            warmup=warmup,
            repeats=repeats,
        )
        if direct_cslt_alg0_out is not None
        else (None, direct_cslt_alg0_error)
    )
    direct_cslt_alg1_ms, direct_cslt_alg1_time_error = (
        time_call_or_error(
            base_direct_cslt_alg1,
            warmup=warmup,
            repeats=repeats,
        )
        if direct_cslt_alg1_out is not None
        else (None, direct_cslt_alg1_error)
    )
    all_corrected_sparse_graph_ms, all_corrected_sparse_graph_error = (
        time_cuda_graph_call(all_corrected_sparse, warmup=warmup, repeats=repeats)
    )
    (
        all_corrected_compressed_cached_graph_ms,
        all_corrected_compressed_cached_graph_error,
    ) = time_cuda_graph_call(
        all_corrected_compressed_cached_cuda,
        warmup=warmup,
        repeats=repeats,
    )
    (
        all_corrected_direct_cslt_alg0_ms,
        all_corrected_direct_cslt_alg0_time_error,
    ) = (
        time_call_or_error(
            all_corrected_direct_cslt_alg0,
            warmup=warmup,
            repeats=repeats,
        )
        if all_corrected_direct_cslt_alg0_out is not None
        else (None, all_corrected_direct_cslt_alg0_error)
    )
    (
        all_corrected_direct_cslt_alg1_ms,
        all_corrected_direct_cslt_alg1_time_error,
    ) = (
        time_call_or_error(
            all_corrected_direct_cslt_alg1,
            warmup=warmup,
            repeats=repeats,
        )
        if all_corrected_direct_cslt_alg1_out is not None
        else (None, all_corrected_direct_cslt_alg1_error)
    )
    (
        all_corrected_direct_cslt_alg0_graph_ms,
        all_corrected_direct_cslt_alg0_graph_error,
    ) = (
        time_cuda_graph_call(
            all_corrected_direct_cslt_alg0,
            warmup=warmup,
            repeats=repeats,
        )
        if all_corrected_direct_cslt_alg0_out is not None
        else (None, all_corrected_direct_cslt_alg0_error)
    )
    (
        all_corrected_direct_cslt_alg1_graph_ms,
        all_corrected_direct_cslt_alg1_graph_error,
    ) = (
        time_cuda_graph_call(
            all_corrected_direct_cslt_alg1,
            warmup=warmup,
            repeats=repeats,
        )
        if all_corrected_direct_cslt_alg1_out is not None
        else (None, all_corrected_direct_cslt_alg1_error)
    )
    triton_tiled_residual_ms, triton_tiled_residual_time_error = (
        time_call_or_error(triton_tiled_residual, warmup=warmup, repeats=repeats)
        if triton_tiled_residual_out is not None
        else (None, triton_tiled_error)
    )
    (
        all_corrected_triton_tiled_residual_ms,
        all_corrected_triton_tiled_residual_error,
    ) = (
        time_call_or_error(
            all_corrected_triton_tiled_residual,
            warmup=warmup,
            repeats=repeats,
        )
        if triton_tiled_residual_out is not None
        else (None, triton_tiled_error)
    )
    triton_pos_tiled_residual_ms, triton_pos_tiled_residual_time_error = (
        time_call_or_error(triton_pos_tiled_residual, warmup=warmup, repeats=repeats)
        if triton_pos_tiled_residual_out is not None
        else (None, triton_pos_tiled_error)
    )
    triton_abspos_tiled_residual_ms, triton_abspos_tiled_residual_time_error = (
        time_call_or_error(
            triton_abspos_tiled_residual,
            warmup=warmup,
            repeats=repeats,
        )
        if triton_abspos_tiled_residual_out is not None
        else (None, triton_abspos_tiled_error)
    )
    (
        all_corrected_triton_pos_tiled_residual_ms,
        all_corrected_triton_pos_tiled_residual_error,
    ) = (
        time_call_or_error(
            all_corrected_triton_pos_tiled_residual,
            warmup=warmup,
            repeats=repeats,
        )
        if triton_pos_tiled_residual_out is not None
        else (None, triton_pos_tiled_error)
    )
    (
        all_corrected_triton_abspos_tiled_residual_ms,
        all_corrected_triton_abspos_tiled_residual_error,
    ) = (
        time_call_or_error(
            all_corrected_triton_abspos_tiled_residual,
            warmup=warmup,
            repeats=repeats,
        )
        if triton_abspos_tiled_residual_out is not None
        else (None, triton_abspos_tiled_error)
    )
    (
        all_corrected_triton_pos_tiled_graph_ms,
        all_corrected_triton_pos_tiled_graph_error,
    ) = (
        time_cuda_graph_call(
            all_corrected_triton_pos_tiled_residual,
            warmup=warmup,
            repeats=repeats,
        )
        if triton_pos_tiled_residual_out is not None
        else (None, triton_pos_tiled_error)
    )
    (
        all_corrected_triton_abspos_tiled_graph_ms,
        all_corrected_triton_abspos_tiled_graph_error,
    ) = (
        time_cuda_graph_call(
            all_corrected_triton_abspos_tiled_residual,
            warmup=warmup,
            repeats=repeats,
        )
        if triton_abspos_tiled_residual_out is not None
        else (None, triton_abspos_tiled_error)
    )

    return {
        "rows": rows,
        "out_features": out_features,
        "in_features": in_features,
        "dtype": str(dtype).replace("torch.", ""),
        "sparse_force_cutlass": sparse_force_cutlass,
        "sparse_default_alg_id": sparse_default_alg_id,
        "sparse_fuse_transpose": sparse_fuse_transpose,
        "backend_class": type(base_sparse).__name__,
        "base_sparse_alg_id_cusparselt": getattr(
            base_sparse, "alg_id_cusparselt", None
        ),
        "base_sparse_fuse_transpose_cusparselt": getattr(
            base_sparse, "fuse_transpose_cusparselt", None
        ),
        "mask_bytes_cpu_device": str(mask_bytes_cpu.device),
        "mask_bytes_cuda_device": str(mask_bytes_cuda.device),
        "residual_values_cpu_device": str(residual_values_cpu.device),
        "residual_values_cuda_device": str(residual_values_cuda.device),
        "cached_residual_weight_cuda_device": str(cached_residual_weight_cuda.device),
        "residual_pos0_cuda_device": str(residual_pos0_cuda.device),
        "residual_pos1_cuda_device": str(residual_pos1_cuda.device),
        "residual_k0_cuda_device": str(residual_k0_cuda.device),
        "residual_k1_cuda_device": str(residual_k1_cuda.device),
        "dense_ms": time_call(dense, warmup=warmup, repeats=repeats),
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "base_dense_zero_ms": time_call(base_dense, warmup=warmup, repeats=repeats),
        "base_sparse_ms": time_call(base_sparse_call, warmup=warmup, repeats=repeats),
        "base_private_mm_ms": time_call(
            base_private_mm_call, warmup=warmup, repeats=repeats
        ),
        "base_direct_cslt_alg0_ms": direct_cslt_alg0_ms,
        "base_direct_cslt_alg0_error": direct_cslt_alg0_time_error,
        "base_direct_cslt_alg1_ms": direct_cslt_alg1_ms,
        "base_direct_cslt_alg1_error": direct_cslt_alg1_time_error,
        "base_sparse_graph_ms": base_sparse_graph_ms,
        "base_sparse_graph_error": base_sparse_graph_error,
        "base_private_mm_graph_ms": private_mm_graph_ms,
        "base_private_mm_graph_error": private_mm_graph_error,
        "base_direct_cslt_alg0_graph_ms": direct_cslt_alg0_graph_ms,
        "base_direct_cslt_alg0_graph_error": direct_cslt_alg0_graph_error,
        "base_direct_cslt_alg1_graph_ms": direct_cslt_alg1_graph_ms,
        "base_direct_cslt_alg1_graph_error": direct_cslt_alg1_graph_error,
        "dense_residual_ms": time_call(dense_residual, warmup=warmup, repeats=repeats),
        "compressed_residual_cpu_ms": time_call(
            compressed_residual_cpu, warmup=warmup, repeats=repeats
        ),
        "compressed_residual_cuda_ms": time_call(
            compressed_residual_cuda, warmup=warmup, repeats=repeats
        ),
        "compressed_residual_cached_cuda_ms": time_call(
            compressed_residual_cached_cuda, warmup=warmup, repeats=repeats
        ),
        "triton_residual_ms": (
            time_call(triton_residual, warmup=warmup, repeats=repeats)
            if triton is not None and triton_residual_out is not None
            else None
        ),
        "triton_tiled_residual_ms": triton_tiled_residual_ms,
        "triton_tiled_residual_error": triton_tiled_residual_time_error,
        "triton_pos_tiled_residual_ms": triton_pos_tiled_residual_ms,
        "triton_pos_tiled_residual_error": triton_pos_tiled_residual_time_error,
        "triton_abspos_tiled_residual_ms": triton_abspos_tiled_residual_ms,
        "triton_abspos_tiled_residual_error":
            triton_abspos_tiled_residual_time_error,
        "all_corrected_sparse_ms": time_call(
            all_corrected_sparse, warmup=warmup, repeats=repeats
        ),
        "all_corrected_sparse_graph_ms": all_corrected_sparse_graph_ms,
        "all_corrected_sparse_graph_error": all_corrected_sparse_graph_error,
        "all_corrected_direct_cslt_alg0_ms": all_corrected_direct_cslt_alg0_ms,
        "all_corrected_direct_cslt_alg0_error":
            all_corrected_direct_cslt_alg0_time_error,
        "all_corrected_direct_cslt_alg0_graph_ms":
            all_corrected_direct_cslt_alg0_graph_ms,
        "all_corrected_direct_cslt_alg0_graph_error":
            all_corrected_direct_cslt_alg0_graph_error,
        "all_corrected_direct_cslt_alg1_ms": all_corrected_direct_cslt_alg1_ms,
        "all_corrected_direct_cslt_alg1_error":
            all_corrected_direct_cslt_alg1_time_error,
        "all_corrected_direct_cslt_alg1_graph_ms":
            all_corrected_direct_cslt_alg1_graph_ms,
        "all_corrected_direct_cslt_alg1_graph_error":
            all_corrected_direct_cslt_alg1_graph_error,
        "all_corrected_compressed_cpu_ms": time_call(
            all_corrected_compressed_cpu, warmup=warmup, repeats=repeats
        ),
        "all_corrected_compressed_cuda_ms": time_call(
            all_corrected_compressed_cuda, warmup=warmup, repeats=repeats
        ),
        "all_corrected_compressed_cached_cuda_ms": time_call(
            all_corrected_compressed_cached_cuda, warmup=warmup, repeats=repeats
        ),
        "all_corrected_compressed_cached_graph_ms":
            all_corrected_compressed_cached_graph_ms,
        "all_corrected_compressed_cached_graph_error":
            all_corrected_compressed_cached_graph_error,
        "all_corrected_triton_residual_ms": (
            time_call(all_corrected_triton_residual, warmup=warmup, repeats=repeats)
            if triton is not None and triton_residual_out is not None
            else None
        ),
        "all_corrected_triton_tiled_residual_ms":
            all_corrected_triton_tiled_residual_ms,
        "all_corrected_triton_tiled_residual_error":
            all_corrected_triton_tiled_residual_error,
        "all_corrected_triton_pos_tiled_residual_ms":
            all_corrected_triton_pos_tiled_residual_ms,
        "all_corrected_triton_pos_tiled_residual_error":
            all_corrected_triton_pos_tiled_residual_error,
        "all_corrected_triton_pos_tiled_graph_ms":
            all_corrected_triton_pos_tiled_graph_ms,
        "all_corrected_triton_pos_tiled_graph_error":
            all_corrected_triton_pos_tiled_graph_error,
        "all_corrected_triton_abspos_tiled_residual_ms":
            all_corrected_triton_abspos_tiled_residual_ms,
        "all_corrected_triton_abspos_tiled_residual_error":
            all_corrected_triton_abspos_tiled_residual_error,
        "all_corrected_triton_abspos_tiled_graph_ms":
            all_corrected_triton_abspos_tiled_graph_ms,
        "all_corrected_triton_abspos_tiled_graph_error":
            all_corrected_triton_abspos_tiled_graph_error,
        "max_abs_diff": max_abs_diff,
        "compressed_max_abs_diff": compressed_max_abs_diff,
        "private_mm_max_abs_diff": private_mm_max_abs_diff,
        "direct_cslt_alg0_max_abs_diff": direct_cslt_alg0_max_abs_diff,
        "direct_cslt_alg1_max_abs_diff": direct_cslt_alg1_max_abs_diff,
        "all_corrected_direct_cslt_alg0_max_abs_diff":
            all_corrected_direct_cslt_alg0_max_abs_diff,
        "all_corrected_direct_cslt_alg1_max_abs_diff":
            all_corrected_direct_cslt_alg1_max_abs_diff,
        "triton_residual_max_abs_diff": triton_residual_max_abs_diff,
        "triton_corrected_max_abs_diff": triton_corrected_max_abs_diff,
        "triton_tiled_residual_max_abs_diff": triton_tiled_residual_max_abs_diff,
        "triton_tiled_corrected_max_abs_diff": triton_tiled_corrected_max_abs_diff,
        "triton_pos_tiled_residual_max_abs_diff":
            triton_pos_tiled_residual_max_abs_diff,
        "triton_pos_tiled_corrected_max_abs_diff":
            triton_pos_tiled_corrected_max_abs_diff,
        "triton_abspos_tiled_residual_max_abs_diff":
            triton_abspos_tiled_residual_max_abs_diff,
        "triton_abspos_tiled_corrected_max_abs_diff":
            triton_abspos_tiled_corrected_max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "dense_weight_bytes": int(dense_weight.numel() * dense_weight.element_size()),
        "base_sparse_value_bytes": base_value_bytes,
        "base_sparse_metadata_bytes": base_meta_bytes,
        "residual_sparse_value_bytes": residual_value_bytes,
        "residual_sparse_metadata_bytes": residual_meta_bytes,
        "base_plus_residual_storage_over_dense": (
            base_value_bytes + base_meta_bytes + residual_value_bytes + residual_meta_bytes
        )
        / float(dense_weight.numel() * dense_weight.element_size()),
        "profiler_sparse_events": profile_sparse_call(x, base_sparse),
    }


def run_triton_tile_sweep_case(
    *,
    rows: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    block_ms: list[int],
    block_ns: list[int],
    block_gs: list[int],
) -> list[dict[str, Any]]:
    if triton is None:
        return []

    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    dense_weight = torch.randn(out_features, in_features, device="cuda", dtype=dtype)
    base_weight, residual_weight, keep, residual_values = make_complementary_24(
        dense_weight
    )
    base_sparse = to_sparse_semi_structured(base_weight)
    residual_values = residual_values.detach().contiguous()
    pos0, pos1 = residual_positions_from_keep(keep.detach())
    k0, k1 = residual_absolute_positions_from_keep(keep.detach())
    torch.cuda.synchronize()

    dense_graph_ms, dense_graph_error = time_cuda_graph_call(
        lambda: F.linear(x, dense_weight),
        warmup=warmup,
        repeats=repeats,
    )
    base_sparse_graph_ms, base_sparse_graph_error = time_cuda_graph_call(
        lambda: F.linear(x, base_sparse),
        warmup=warmup,
        repeats=repeats,
    )
    dense_residual_graph_ms, dense_residual_graph_error = time_cuda_graph_call(
        lambda: F.linear(x, residual_weight),
        warmup=warmup,
        repeats=repeats,
    )

    out: list[dict[str, Any]] = []
    for block_m in block_ms:
        for block_n in block_ns:
            for block_g in block_gs:

                def pos_residual() -> torch.Tensor:
                    return triton_residual24_linear_pos_tiled(
                        x,
                        pos0=pos0,
                        pos1=pos1,
                        residual_values=residual_values,
                        out_features=out_features,
                        in_features=in_features,
                        block_m=block_m,
                        block_n=block_n,
                        block_g=block_g,
                    )

                def abspos_residual() -> torch.Tensor:
                    return triton_residual24_linear_abspos_tiled(
                        x,
                        k0=k0,
                        k1=k1,
                        residual_values=residual_values,
                        out_features=out_features,
                        in_features=in_features,
                        block_m=block_m,
                        block_n=block_n,
                        block_g=block_g,
                    )

                def pos_all_corrected() -> torch.Tensor:
                    return F.linear(x, base_sparse) + pos_residual()

                def abspos_all_corrected() -> torch.Tensor:
                    return F.linear(x, base_sparse) + abspos_residual()

                pos_ms, pos_error = time_cuda_graph_call(
                    pos_residual, warmup=warmup, repeats=repeats
                )
                abspos_ms, abspos_error = time_cuda_graph_call(
                    abspos_residual, warmup=warmup, repeats=repeats
                )
                pos_all_ms, pos_all_error = time_cuda_graph_call(
                    pos_all_corrected, warmup=warmup, repeats=repeats
                )
                abspos_all_ms, abspos_all_error = time_cuda_graph_call(
                    abspos_all_corrected, warmup=warmup, repeats=repeats
                )

                target_ms = (
                    dense_graph_ms / 1.2
                    if dense_graph_ms is not None
                    else None
                )
                for kernel, residual_ms, residual_error, all_ms, all_error in (
                    ("pos_tiled", pos_ms, pos_error, pos_all_ms, pos_all_error),
                    (
                        "abspos_tiled",
                        abspos_ms,
                        abspos_error,
                        abspos_all_ms,
                        abspos_all_error,
                    ),
                ):
                    out.append({
                        "rows": rows,
                        "out_features": out_features,
                        "in_features": in_features,
                        "dtype": str(dtype).replace("torch.", ""),
                        "kernel": kernel,
                        "block_m": block_m,
                        "block_n": block_n,
                        "block_g": block_g,
                        "dense_graph_ms": dense_graph_ms,
                        "dense_graph_error": dense_graph_error,
                        "dense_1p2_target_ms": target_ms,
                        "base_sparse_graph_ms": base_sparse_graph_ms,
                        "base_sparse_graph_error": base_sparse_graph_error,
                        "dense_residual_graph_ms": dense_residual_graph_ms,
                        "dense_residual_graph_error": dense_residual_graph_error,
                        "triton_residual_graph_ms": residual_ms,
                        "triton_residual_graph_error": residual_error,
                        "all_corrected_triton_graph_ms": all_ms,
                        "all_corrected_triton_graph_error": all_error,
                        "all_corrected_vs_dense_graph": (
                            all_ms / dense_graph_ms
                            if all_ms is not None
                            and dense_graph_ms not in (None, 0.0)
                            else None
                        ),
                        "needed_reduction_to_1p2": (
                            all_ms / target_ms
                            if all_ms is not None and target_ms not in (None, 0.0)
                            else None
                        ),
                    })
    return out


def write_triton_tile_sweep(output_root: Path,
                            rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_root / "triton_tile_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def as_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def fmt(value: Any) -> str:
        number = as_float(value)
        return "" if number is None else f"{number:.4f}"

    def fmt_ratio(value: Any, denom: Any) -> str:
        number = as_float(value)
        denom_number = as_float(denom)
        if number is None or denom_number in (None, 0.0):
            return ""
        return f"{number / denom_number:.2f}x"

    def row_key(row: dict[str, Any]) -> float:
        return as_float(row.get("all_corrected_triton_graph_ms")) or float("inf")

    best_rows = sorted(rows, key=row_key)[:10]
    with (output_root / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Triton Residual Tile Sweep\n\n")
        handle.write(
            "This sweep keeps the current residual 2:4 Triton algorithm and "
            "only changes tile sizes. It separates a bad tile choice from an "
            "algorithmic bottleneck.\n\n"
        )
        handle.write(
            "| rows | out | in | kernel | block_m | block_n | block_g | "
            "dense graph | target 1.2x | residual graph | all-corrected graph | "
            "all/dense | needed reduction | error |\n"
        )
        handle.write(
            "|-----:|----:|---:|--------|--------:|--------:|--------:|"
            "------------:|-----------:|---------------:|--------------------:|"
            "----------:|----------------:|-------|\n"
        )
        for row in best_rows:
            error = (
                row.get("all_corrected_triton_graph_error")
                or row.get("triton_residual_graph_error")
                or ""
            )
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                f"`{row['kernel']}` | {row['block_m']} | {row['block_n']} | "
                f"{row['block_g']} | {fmt(row.get('dense_graph_ms'))} | "
                f"{fmt(row.get('dense_1p2_target_ms'))} | "
                f"{fmt(row.get('triton_residual_graph_ms'))} | "
                f"{fmt(row.get('all_corrected_triton_graph_ms'))} | "
                f"{fmt_ratio(row.get('all_corrected_triton_graph_ms'), row.get('dense_graph_ms'))} | "
                f"{fmt_ratio(row.get('all_corrected_triton_graph_ms'), row.get('dense_1p2_target_ms'))} | "
                f"{str(error).replace('|', '/')[:120]} |\n"
            )


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    def fmt_ms(value: Any) -> str:
        return "" if value is None else f"{float(value):.4f}"

    def fmt_ratio(value: Any, base: float) -> str:
        return "" if value is None else f"{float(value) / base:.1f}x"

    def fmt_speedup(value: Any, base: float) -> str:
        return "" if value is None else f"{base / float(value):.2f}x"

    def fmt_diff(value: Any) -> str:
        return "" if value is None else f"{float(value):.4g}"

    def fmt_graph(value: Any, error: Any) -> str:
        if value is not None:
            return f"{float(value):.4f}"
        if not error:
            return ""
        return str(error).replace("|", "/")[:120]

    def as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def best_candidate(row: dict[str, Any]) -> tuple[str, float] | None:
        candidates = [
            ("all_sparse_graph", row.get("all_corrected_sparse_graph_ms")),
            ("direct_cslt_alg0_graph", row.get("all_corrected_direct_cslt_alg0_graph_ms")),
            ("direct_cslt_alg1_graph", row.get("all_corrected_direct_cslt_alg1_graph_ms")),
            (
                "compressed_cached_graph",
                row.get("all_corrected_compressed_cached_graph_ms"),
            ),
            (
                "triton_pos_tiled_graph",
                row.get("all_corrected_triton_pos_tiled_graph_ms"),
            ),
            (
                "triton_abspos_tiled_graph",
                row.get("all_corrected_triton_abspos_tiled_graph_ms"),
            ),
        ]
        valid = [(name, value) for name, value in candidates if as_float(value) is not None]
        if not valid:
            return None
        name, value = min(valid, key=lambda item: float(item[1]))
        return name, float(value)

    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SpecLink SR24 Sparse Backend Probe\n\n")
        handle.write(
            "This probe verifies PyTorch `SparseSemiStructuredTensorCUSPARSELT` "
            "execution for representative Linear shapes. It is a microbenchmark, "
            "not an end-to-end serving result. The compressed columns include "
            "runtime dense residual materialization from packed masks and "
            "complementary values, matching the current SR24 compressed-dense "
            "prototype path. The Triton residual column is an experimental "
            "GPU-only compressed residual matmul that avoids materializing a "
            "dense residual weight. The private `_mm` and direct `_cslt` columns "
            "separate PyTorch dispatch overhead from the underlying cuSPARSELt "
            "matmul cost.\n\n"
        )
        handle.write("## Compressed Residual Device Check\n\n")
        handle.write(
            "The `*_cuda_device` columns must be CUDA devices for the GPU-side "
            "compressed residual paths. CPU tensors are included only for the "
            "explicit CPU materialization baseline.\n\n"
        )
        handle.write("## Sparse Backend Configuration\n\n")
        handle.write(
            "| rows | backend class | force CUTLASS | default alg id | fuse transpose | weight alg id | weight fuse transpose |\n"
        )
        handle.write(
            "|-----:|---------------|---------------|---------------:|----------------|--------------:|-----------------------|\n"
        )
        for row in rows:
            handle.write(
                f"| {int(row['rows'])} | "
                f"`{row.get('backend_class', '')}` | "
                f"{row.get('sparse_force_cutlass', '')} | "
                f"{row.get('sparse_default_alg_id', '')} | "
                f"{row.get('sparse_fuse_transpose', '')} | "
                f"{row.get('base_sparse_alg_id_cusparselt', '')} | "
                f"{row.get('base_sparse_fuse_transpose_cusparselt', '')} |\n"
            )
        handle.write("\n")
        handle.write(
            "| rows | mask CPU | mask CUDA | residual values CPU | residual values CUDA | cached dense residual | pos0 CUDA | pos1 CUDA | k0 CUDA | k1 CUDA |\n"
        )
        handle.write("|-----:|----------|-----------|---------------------|----------------------|-----------------------|-----------|-----------|---------|---------|\n")
        for row in rows:
            handle.write(
                f"| {int(row['rows'])} | "
                f"{row.get('mask_bytes_cpu_device', '')} | "
                f"{row.get('mask_bytes_cuda_device', '')} | "
                f"{row.get('residual_values_cpu_device', '')} | "
                f"{row.get('residual_values_cuda_device', '')} | "
                f"{row.get('cached_residual_weight_cuda_device', '')} | "
                f"{row.get('residual_pos0_cuda_device', '')} | "
                f"{row.get('residual_pos1_cuda_device', '')} | "
                f"{row.get('residual_k0_cuda_device', '')} | "
                f"{row.get('residual_k1_cuda_device', '')} |\n"
            )
        handle.write("\n")
        handle.write("| rows | out | in | dense ms | dense graph ms/error | base dense-zero ms | base sparse ms | sparse graph ms/error | private _mm ms | private graph ms/error | raw cslt alg0 ms | cslt0 graph ms/error | raw cslt alg1 ms | cslt1 graph ms/error | dense residual ms | compressed residual CPU ms | compressed residual CUDA ms | compressed residual cached CUDA ms | Triton residual ms | Triton tiled residual ms/error | Triton pos tiled residual ms/error | all sparse ms | all compressed CUDA ms | all compressed cached CUDA ms | all Triton residual ms | all Triton tiled ms/error | all Triton pos tiled ms/error | dense-zero/dense | sparse/dense | private/dense | cslt0/dense | cslt1/dense | comp cuda/dense | comp cached/dense | Triton all/dense | Triton tiled all/dense | Triton pos tiled all/dense | cslt0 diff | cslt1 diff | comp diff | Triton diff | Triton tiled diff | Triton pos tiled diff | backend | sparse events |\n")
        handle.write("|-----:|----:|---:|---------:|---------------------:|-------------------:|---------------:|----------------------:|---------------:|-----------------------:|-----------------:|----------------------:|-----------------:|----------------------:|------------------:|---------------------------:|----------------------------:|-----------------------------------:|-------------------:|-------------------------------:|-----------------------------------:|--------------:|-----------------------:|------------------------------:|-----------------------:|----------------------------:|--------------------------------:|-----------------:|-------------:|--------------:|------------:|------------:|----------------:|-------------------:|-----------------:|-----------------------:|---------------------------:|-----------:|-----------:|----------:|------------:|------------------:|----------------------:|---------|---------------|\n")
        for row in rows:
            events = ", ".join(event["key"] for event in row["profiler_sparse_events"]) or "missing"
            dense_ms = float(row["dense_ms"])
            base_dense_ms = float(row["base_dense_zero_ms"])
            base_ms = float(row["base_sparse_ms"])
            private_mm = float(row["base_private_mm_ms"])
            direct_cslt_alg0 = row.get("base_direct_cslt_alg0_ms")
            direct_cslt_alg1 = row.get("base_direct_cslt_alg1_ms")
            all_ms = float(row["all_corrected_sparse_ms"])
            comp_cuda_ms = float(row["all_corrected_compressed_cuda_ms"])
            comp_cached_ms = float(row["all_corrected_compressed_cached_cuda_ms"])
            triton_all_ms = row.get("all_corrected_triton_residual_ms")
            triton_tiled_all_ms = row.get(
                "all_corrected_triton_tiled_residual_ms"
            )
            triton_pos_tiled_all_ms = row.get(
                "all_corrected_triton_pos_tiled_residual_ms"
            )
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                f"{dense_ms:.4f} | "
                f"{fmt_graph(row.get('dense_graph_ms'), row.get('dense_graph_error'))} | "
                f"{base_dense_ms:.4f} | "
                f"{base_ms:.4f} | "
                f"{fmt_graph(row.get('base_sparse_graph_ms'), row.get('base_sparse_graph_error'))} | "
                f"{private_mm:.4f} | "
                f"{fmt_graph(row.get('base_private_mm_graph_ms'), row.get('base_private_mm_graph_error'))} | "
                f"{fmt_graph(direct_cslt_alg0, row.get('base_direct_cslt_alg0_error'))} | "
                f"{fmt_graph(row.get('base_direct_cslt_alg0_graph_ms'), row.get('base_direct_cslt_alg0_graph_error'))} | "
                f"{fmt_graph(direct_cslt_alg1, row.get('base_direct_cslt_alg1_error'))} | "
                f"{fmt_graph(row.get('base_direct_cslt_alg1_graph_ms'), row.get('base_direct_cslt_alg1_graph_error'))} | "
                f"{float(row['dense_residual_ms']):.4f} | "
                f"{float(row['compressed_residual_cpu_ms']):.4f} | "
                f"{float(row['compressed_residual_cuda_ms']):.4f} | "
                f"{float(row['compressed_residual_cached_cuda_ms']):.4f} | "
                f"{fmt_ms(row.get('triton_residual_ms'))} | "
                f"{fmt_graph(row.get('triton_tiled_residual_ms'), row.get('triton_tiled_residual_error'))} | "
                f"{fmt_graph(row.get('triton_pos_tiled_residual_ms'), row.get('triton_pos_tiled_residual_error'))} | "
                f"{all_ms:.4f} | "
                f"{comp_cuda_ms:.4f} | "
                f"{comp_cached_ms:.4f} | "
                f"{fmt_ms(triton_all_ms)} | "
                f"{fmt_graph(triton_tiled_all_ms, row.get('all_corrected_triton_tiled_residual_error'))} | "
                f"{fmt_graph(triton_pos_tiled_all_ms, row.get('all_corrected_triton_pos_tiled_residual_error'))} | "
                f"{base_dense_ms / dense_ms:.1f}x | "
                f"{base_ms / dense_ms:.1f}x | "
                f"{private_mm / dense_ms:.1f}x | "
                f"{fmt_ratio(direct_cslt_alg0, dense_ms)} | "
                f"{fmt_ratio(direct_cslt_alg1, dense_ms)} | "
                f"{comp_cuda_ms / dense_ms:.1f}x | "
                f"{comp_cached_ms / dense_ms:.1f}x | "
                f"{fmt_ratio(triton_all_ms, dense_ms)} | "
                f"{fmt_ratio(triton_tiled_all_ms, dense_ms)} | "
                f"{fmt_ratio(triton_pos_tiled_all_ms, dense_ms)} | "
                f"{fmt_diff(row.get('direct_cslt_alg0_max_abs_diff'))} | "
                f"{fmt_diff(row.get('direct_cslt_alg1_max_abs_diff'))} | "
                f"{row['compressed_max_abs_diff']:.4g} | "
                f"{fmt_diff(row.get('triton_corrected_max_abs_diff'))} | "
                f"{fmt_diff(row.get('triton_tiled_corrected_max_abs_diff'))} | "
                f"{fmt_diff(row.get('triton_pos_tiled_corrected_max_abs_diff'))} | "
                f"{row['backend_class']} | {events} |\n"
            )
        handle.write("\n## All-Corrected Graph Candidates\n\n")
        handle.write(
            "These columns isolate the candidate path for exact `all_corrected_24`: "
            "one 2:4 base GEMM plus one complementary 2:4 residual GEMM. The graph "
            "columns replay a captured CUDA Graph, so they mostly remove Python and "
            "CPU launch overhead. If these graph timings are still not competitive "
            "with dense, a real fused packed base+residual kernel is needed before "
            "this path is worth wiring into serving.\n\n"
        )
        handle.write("| rows | out | in | dense ms | dense graph ms/error | all sparse eager ms | all sparse graph ms/error | all direct cslt0 eager ms/error | all direct cslt0 graph ms/error | all direct cslt1 eager ms/error | all direct cslt1 graph ms/error | all compressed CUDA ms | all compressed cached CUDA ms | all compressed cached graph ms/error | all Triton pos tiled graph ms/error | all Triton abspos graph ms/error | graph sparse speedup vs dense graph | graph cslt0 speedup vs dense graph | graph cslt1 speedup vs dense graph | graph compressed cached speedup vs dense graph | graph pos tiled speedup vs dense graph | graph abspos speedup vs dense graph | cslt0 corrected diff | cslt1 corrected diff | pos tiled corrected diff | abspos corrected diff |\n")
        handle.write("|-----:|----:|---:|---------:|---------------------:|--------------------:|--------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|-----------------------:|------------------------------:|-----------------------------------:|-----------------------------------:|--------------------------------:|----------------------------------:|---------------------------------:|---------------------------------:|---------------------------------------------:|------------------------------------:|---------------------------------:|---------------------:|---------------------:|------------------------:|----------------------:|\n")
        for row in rows:
            dense_ms = float(row["dense_ms"])
            dense_graph_ms = row.get("dense_graph_ms") or dense_ms
            all_sparse_ms = float(row["all_corrected_sparse_ms"])
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                f"{dense_ms:.4f} | "
                f"{fmt_graph(row.get('dense_graph_ms'), row.get('dense_graph_error'))} | "
                f"{all_sparse_ms:.4f} | "
                f"{fmt_graph(row.get('all_corrected_sparse_graph_ms'), row.get('all_corrected_sparse_graph_error'))} | "
                f"{fmt_graph(row.get('all_corrected_direct_cslt_alg0_ms'), row.get('all_corrected_direct_cslt_alg0_error'))} | "
                f"{fmt_graph(row.get('all_corrected_direct_cslt_alg0_graph_ms'), row.get('all_corrected_direct_cslt_alg0_graph_error'))} | "
                f"{fmt_graph(row.get('all_corrected_direct_cslt_alg1_ms'), row.get('all_corrected_direct_cslt_alg1_error'))} | "
                f"{fmt_graph(row.get('all_corrected_direct_cslt_alg1_graph_ms'), row.get('all_corrected_direct_cslt_alg1_graph_error'))} | "
                f"{float(row['all_corrected_compressed_cuda_ms']):.4f} | "
                f"{float(row['all_corrected_compressed_cached_cuda_ms']):.4f} | "
                f"{fmt_graph(row.get('all_corrected_compressed_cached_graph_ms'), row.get('all_corrected_compressed_cached_graph_error'))} | "
                f"{fmt_graph(row.get('all_corrected_triton_pos_tiled_graph_ms'), row.get('all_corrected_triton_pos_tiled_graph_error'))} | "
                f"{fmt_graph(row.get('all_corrected_triton_abspos_tiled_graph_ms'), row.get('all_corrected_triton_abspos_tiled_graph_error'))} | "
                f"{fmt_speedup(row.get('all_corrected_sparse_graph_ms'), dense_graph_ms)} | "
                f"{fmt_speedup(row.get('all_corrected_direct_cslt_alg0_graph_ms'), dense_graph_ms)} | "
                f"{fmt_speedup(row.get('all_corrected_direct_cslt_alg1_graph_ms'), dense_graph_ms)} | "
                f"{fmt_speedup(row.get('all_corrected_compressed_cached_graph_ms'), dense_graph_ms)} | "
                f"{fmt_speedup(row.get('all_corrected_triton_pos_tiled_graph_ms'), dense_graph_ms)} | "
                f"{fmt_speedup(row.get('all_corrected_triton_abspos_tiled_graph_ms'), dense_graph_ms)} | "
                f"{fmt_diff(row.get('all_corrected_direct_cslt_alg0_max_abs_diff'))} | "
                f"{fmt_diff(row.get('all_corrected_direct_cslt_alg1_max_abs_diff'))} | "
                f"{fmt_diff(row.get('triton_pos_tiled_corrected_max_abs_diff'))} | "
                f"{fmt_diff(row.get('triton_abspos_tiled_corrected_max_abs_diff'))} |\n"
            )

        handle.write("\n## Fused All-Corrected Requirement\n\n")
        handle.write(
            "Exact `all_corrected_24` cannot skip the complementary 2:4 weights. "
            "A useful implementation therefore needs a fused packed operator whose "
            "single-call time is below dense, not just two separate sparse GEMMs. "
            "The target below uses `dense_graph_ms / 1.2`, matching the later "
            "`speclink_t08` goal of at least 1.2x over dense. A candidate whose "
            "`needed reduction` is much larger than 1 still needs a new CUDA/Triton "
            "kernel shape rather than another Python/vLLM integration flag.\n\n"
        )
        handle.write(
            "| rows | out | in | dense graph ms | target for 1.2x ms | best current exact graph path | best current ms | current vs dense | needed reduction to 1.2x | verdict |\n"
        )
        handle.write(
            "|-----:|----:|---:|---------------:|-------------------:|-------------------------------|----------------:|-----------------:|-------------------------:|---------|\n"
        )
        for row in rows:
            dense_graph_ms = as_float(row.get("dense_graph_ms")) or float(row["dense_ms"])
            target_ms = dense_graph_ms / 1.2
            best = best_candidate(row)
            if best is None:
                handle.write(
                    f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                    f"{dense_graph_ms:.4f} | {target_ms:.4f} | n/a |  |  |  | no graph candidate |\n"
                )
                continue
            best_name, best_ms = best
            current_vs_dense = best_ms / dense_graph_ms
            needed_reduction = best_ms / target_ms
            verdict = (
                "already meets target"
                if needed_reduction <= 1.0
                else "needs fused packed kernel"
            )
            handle.write(
                f"| {row['rows']} | {row['out_features']} | {row['in_features']} | "
                f"{dense_graph_ms:.4f} | {target_ms:.4f} | `{best_name}` | "
                f"{best_ms:.4f} | {current_vs_dense:.2f}x dense time | "
                f"{needed_reduction:.2f}x faster than current | {verdict} |\n"
            )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for semi-structured sparse backend probe")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = args.shape or [(64, 4096, 4096), (64, 6144, 4096)]
    output_root = args.output_root or (
        Path("examples/evaluate/eval-guidellm/results.bak")
        / f"speclink_sr24_sparse_backend_probe_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    with sparse_backend_overrides(
        force_cutlass=args.sparse_force_cutlass,
        alg_id=args.sparse_alg_id,
        fuse_transpose=args.sparse_fuse_transpose,
    ):
        rows = [
            run_case(
                rows=shape[0],
                out_features=shape[1],
                in_features=shape[2],
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                skip_triton_residual=args.skip_triton_residual,
            )
            for shape in shapes
        ]
        sweep_rows: list[dict[str, Any]] = []
        if args.triton_tile_sweep:
            for shape in shapes:
                sweep_rows.extend(
                    run_triton_tile_sweep_case(
                        rows=shape[0],
                        out_features=shape[1],
                        in_features=shape[2],
                        dtype=dtype,
                        warmup=args.triton_sweep_warmup,
                        repeats=args.triton_sweep_repeats,
                        block_ms=args.triton_block_ms,
                        block_ns=args.triton_block_ns,
                        block_gs=args.triton_block_gs,
                    )
                )
    (output_root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_root, rows)
    write_triton_tile_sweep(output_root, sweep_rows)
    print(output_root.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile PyTorch semi-structured sparse backend for SR24.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=None,
        help="ROWS,OUT,IN shape. Can be repeated.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--skip-triton-residual",
        action="store_true",
        help=(
            "Skip the experimental scalar Triton residual 2:4 kernels. Use this "
            "for large-shape all-corrected diagnostics when the goal is to "
            "compare dense, PyTorch sparse, direct cuSPARSELt, and compressed "
            "cached paths without spending minutes on already-known slow Triton "
            "variants."
        ),
    )
    parser.add_argument(
        "--triton-tile-sweep",
        action="store_true",
        help=(
            "Sweep residual 2:4 Triton tile sizes and append the best "
            "all-corrected graph candidates to summary.md."
        ),
    )
    parser.add_argument("--triton-block-ms",
                        type=parse_int_csv,
                        default=[8, 16, 32])
    parser.add_argument("--triton-block-ns",
                        type=parse_int_csv,
                        default=[16, 32, 64])
    parser.add_argument("--triton-block-gs",
                        type=parse_int_csv,
                        default=[16, 32, 64])
    parser.add_argument("--triton-sweep-warmup", type=int, default=3)
    parser.add_argument("--triton-sweep-repeats", type=int, default=10)
    parser.add_argument(
        "--sparse-force-cutlass",
        type=parse_optional_bool,
        default=None,
        help=(
            "Override torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS. "
            "Use true/false or default."
        ),
    )
    parser.add_argument(
        "--sparse-alg-id",
        type=int,
        default=None,
        help="Override torch.sparse.SparseSemiStructuredTensor._DEFAULT_ALG_ID.",
    )
    parser.add_argument(
        "--sparse-fuse-transpose",
        type=parse_optional_bool,
        default=None,
        help=(
            "Override torch.sparse.SparseSemiStructuredTensor._FUSE_TRANSPOSE. "
            "Use true/false or default."
        ),
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
