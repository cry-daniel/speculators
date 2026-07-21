"""Canonical 2:4 selector packing and CUTLASS metadata reordering."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import torch

from ._cuda_extension import load_cuda_extension


SPARSE24_METADATA_NIBBLES = (0x4, 0x8, 0x9, 0xC, 0xD, 0xE)


def unpack_metadata(metadata: torch.Tensor) -> torch.Tensor:
    low = metadata.bitwise_and(0x0F)
    high = metadata.bitwise_right_shift(4).bitwise_and(0x0F)
    return torch.stack((low, high), dim=-1).flatten(-2).contiguous()


def pack_metadata(nibbles: torch.Tensor) -> torch.Tensor:
    if nibbles.dtype != torch.uint8 or nibbles.shape[-1] % 2:
        raise ValueError("metadata nibbles must be uint8 with an even last dimension")
    paired = nibbles.view(*nibbles.shape[:-1], nibbles.shape[-1] // 2, 2)
    return paired[..., 0].bitwise_or(paired[..., 1].bitwise_left_shift(4)).contiguous()


def positions_from_metadata(
    metadata: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nibbles = unpack_metadata(metadata)
    retained = torch.stack(
        (
            nibbles.bitwise_and(0x3).to(torch.int64),
            nibbles.bitwise_right_shift(2).bitwise_and(0x3).to(torch.int64),
        ),
        dim=-1,
    )
    positions = torch.arange(4, device=metadata.device, dtype=torch.int64)
    expanded = positions.view(1, 1, 4).expand(*nibbles.shape, 4)
    mask = torch.zeros_like(expanded, dtype=torch.bool)
    mask.scatter_(-1, retained, True)
    complement = expanded.masked_select(~mask).view(*nibbles.shape, 2)
    return retained, complement


def validate_metadata(metadata: torch.Tensor, shape: tuple[int, int]) -> None:
    n, k = shape
    if (
        not isinstance(metadata, torch.Tensor)
        or metadata.dtype != torch.uint8
        or tuple(metadata.shape) != (n, k // 8)
        or not metadata.is_contiguous()
    ):
        raise ValueError("metadata must be contiguous uint8 with shape [N,K/8]")
    nibbles = unpack_metadata(metadata)
    valid = torch.zeros_like(nibbles, dtype=torch.bool)
    for nibble in SPARSE24_METADATA_NIBBLES:
        valid.logical_or_(nibbles.eq(nibble))
    if not bool(valid.all().item()):
        raise ValueError("metadata contains an invalid 2:4 selector nibble")


@functools.lru_cache(maxsize=1)
def _extension() -> Any:
    root = Path(__file__).resolve().parents[3]
    csrc = Path(__file__).resolve().parent / "csrc"
    build_dir = Path(
        os.environ.get(
            "SPECLINK_SPARSE24_METADATA_BUILD_DIR",
            root / "temp/torch_extensions/sparse24_cutlass_metadata_cuda130",
        )
    )
    return load_cuda_extension(
        name="speclink_sparse24_cutlass_metadata_cuda",
        sources=(
            csrc / "sparse24_cutlass_metadata.cpp",
            csrc / "sparse24_cutlass_metadata.cu",
        ),
        build_dir=build_dir,
        include_cutlass=False,
        verbose_env="SPECLINK_SPARSE24_METADATA_VERBOSE_BUILD",
    )


def reorder_cutlass_sparse24_metadata(
    metadata: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Return CUTLASS ``ColumnMajorInterleaved<2>`` ElementE metadata."""

    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError(f"invalid logical [N,K] shape {shape!r}")
    n, k = shape
    validate_metadata(metadata, shape)
    if not metadata.is_cuda:
        raise ValueError("CUTLASS metadata reorder requires CUDA metadata")
    if n % 32 or k % 32:
        raise ValueError("CUTLASS metadata reorder requires N,K multiples of 32")
    reordered = _extension().reorder_metadata(metadata, k)
    if (
        reordered.dtype != torch.int16
        or tuple(reordered.shape) != (n, k // 16)
        or not reordered.is_contiguous()
        or reordered.device != metadata.device
    ):
        raise RuntimeError("CUTLASS metadata reorder returned an invalid tensor")
    return reordered


__all__ = [
    "SPARSE24_METADATA_NIBBLES",
    "pack_metadata",
    "positions_from_metadata",
    "reorder_cutlass_sparse24_metadata",
    "unpack_metadata",
    "validate_metadata",
]
