"""Frozen old dense-base concurrent branches used by the paper NCU study.

This module intentionally exposes only the two independently launched BM64
branches from the former optimization ladder.  The stored representation is
one dense BF16 weight plus canonical 2:4 metadata; the sparse branch constructs
its HMMA.SP operand online, while the dense branch consumes the dense weight.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from pathlib import Path
from typing import Any

import torch

from ._cuda_extension import load_cuda_extension
from .online_sparse24 import pack_sparse24_components
from .sparse24_cutlass_metadata import reorder_cutlass_sparse24_metadata


OLD_CONCURRENT_DENSE_BRANCH = "dense"
OLD_CONCURRENT_SPARSE_BRANCH = "sparse"
_BRANCH_IDS = {
    OLD_CONCURRENT_DENSE_BRANCH: 0,
    OLD_CONCURRENT_SPARSE_BRANCH: 1,
}
_ATTRIBUTE_NAMES = (
    "num_regs",
    "dynamic_smem_bytes",
    "local_bytes",
    "max_threads_per_block",
    "active_blocks_per_sm",
    "theoretical_occupancy_pct",
)


@dataclass(frozen=True, slots=True)
class OldConcurrentWeight:
    """The exact one-weight representation consumed by the old branches."""

    shape: tuple[int, int]
    dense_weight: torch.Tensor
    reordered_metadata: torch.Tensor

    def __post_init__(self) -> None:
        n, k = self.shape
        if (
            self.dense_weight.dtype != torch.bfloat16
            or tuple(self.dense_weight.shape) != self.shape
            or not self.dense_weight.is_cuda
            or not self.dense_weight.is_contiguous()
        ):
            raise ValueError("dense_weight must be contiguous CUDA BF16 [N,K]")
        if n % 64 or k % 64:
            raise ValueError("old concurrent N and K must be multiples of 64")
        if (
            self.reordered_metadata.dtype != torch.int16
            or tuple(self.reordered_metadata.shape) != (n, k // 16)
            or self.reordered_metadata.device != self.dense_weight.device
            or not self.reordered_metadata.is_contiguous()
        ):
            raise ValueError(
                "reordered_metadata must be contiguous int16 [N,K/16]"
            )

    def persistent_bytes_by_component(self) -> dict[str, int]:
        return {
            "dense_weight": self.dense_weight.numel()
            * self.dense_weight.element_size(),
            "metadata": self.reordered_metadata.numel()
            * self.reordered_metadata.element_size(),
        }

    def persistent_bytes(self) -> int:
        return sum(self.persistent_bytes_by_component().values())


def prepare_old_concurrent_weight(
    dense_weight: torch.Tensor,
    weight24: torch.Tensor | None = None,
    *,
    chunk_rows: int = 512,
) -> OldConcurrentWeight:
    metadata, _, _ = pack_sparse24_components(
        dense_weight,
        weight24,
        need_values=False,
        need_residual=False,
        chunk_rows=chunk_rows,
    )
    reordered = reorder_cutlass_sparse24_metadata(
        metadata, tuple(dense_weight.shape)
    )
    return OldConcurrentWeight(
        shape=tuple(dense_weight.shape),
        dense_weight=dense_weight,
        reordered_metadata=reordered,
    )


@functools.lru_cache(maxsize=1)
def _extension() -> Any:
    root = Path(__file__).resolve().parents[3]
    csrc = Path(__file__).resolve().parent / "csrc"
    build_dir = Path(
        os.environ.get(
            "SPECLINK_OLD_CONCURRENT_BUILD_DIR",
            root / "temp/torch_extensions/old_concurrent_cutlass_cuda130",
        )
    )
    return load_cuda_extension(
        name="speclink_old_concurrent_cutlass_cuda",
        sources=(
            csrc / "old_concurrent_cutlass_sparse24.cpp",
            csrc / "old_concurrent_cutlass_sparse24.cu",
        ),
        required=(
            csrc / "old_concurrent_sidecar_mma.h",
            csrc / "cutlass_transpose_epilogue_visitor.h",
        ),
        build_dir=build_dir,
        verbose_env="SPECLINK_OLD_CONCURRENT_VERBOSE_BUILD",
    )


def old_concurrent_branch_linear_out(
    x: torch.Tensor,
    prepared: OldConcurrentWeight,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    output: torch.Tensor,
    *,
    branch: str,
    persistent_blocks: int | None = None,
) -> torch.Tensor:
    if branch not in _BRANCH_IDS:
        raise ValueError(f"branch must be one of {tuple(_BRANCH_IDS)}")
    if persistent_blocks is None:
        persistent_blocks = torch.cuda.get_device_properties(x.device).multi_processor_count
    if not isinstance(persistent_blocks, int) or persistent_blocks <= 0:
        raise ValueError("persistent_blocks must be a positive integer")
    return _extension().branch_forward_out(
        x,
        dense_indices,
        sparse_indices,
        prepared.dense_weight,
        prepared.reordered_metadata,
        output,
        _BRANCH_IDS[branch],
        persistent_blocks,
    )


def old_concurrent_kernel_attributes(branch: str) -> dict[str, int]:
    if branch not in _BRANCH_IDS:
        raise ValueError(f"branch must be one of {tuple(_BRANCH_IDS)}")
    values = _extension().kernel_attributes(_BRANCH_IDS[branch])
    attributes = dict(zip(_ATTRIBUTE_NAMES, map(int, values), strict=True))
    dense = branch == OLD_CONCURRENT_DENSE_BRANCH
    attributes.update(
        {
            "actual_threads_per_block": 512,
            "dense_warps": 16 if dense else 0,
            "sparse_warps": 0 if dense else 16,
            "route_waves_m2048_f1_8": 1 if dense else 7,
            "output_tile_features": 64,
            "persistent_blocks_per_sm": 1,
        }
    )
    return attributes


__all__ = [
    "OLD_CONCURRENT_DENSE_BRANCH",
    "OLD_CONCURRENT_SPARSE_BRANCH",
    "OldConcurrentWeight",
    "old_concurrent_branch_linear_out",
    "old_concurrent_kernel_attributes",
    "prepare_old_concurrent_weight",
]
