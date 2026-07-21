"""Canonical one-weight layout for residual-complement 2:4 GEMM.

For each K4 group, ``packed_values`` stores the two base values, ``residual``
stores the complementary two values, and one selector nibble records the base
positions.  Two nibbles share one byte, so metadata occupies exactly ``N*K/8``
bytes.  Weight preparation is setup-only and never part of a timed GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sparse24_cutlass_metadata import (
    SPARSE24_METADATA_NIBBLES,
    pack_metadata,
    positions_from_metadata,
    validate_metadata,
)


SPARSE_RESIDUAL_SMEM = "sparse_residual_smem"
_DEFAULT_CHUNK_ROWS = 512


def _require_bf16_matrix(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a 2D BF16 tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{name} must have positive dimensions")


def _require_bf16_weight(tensor: torch.Tensor, name: str) -> None:
    _require_bf16_matrix(tensor, name)
    if tensor.shape[1] % 8:
        raise ValueError(f"{name} K must be divisible by 8")


def _top2_positions(groups: torch.Tensor) -> torch.Tensor:
    ranking = torch.argsort(
        groups.abs().float(), dim=-1, descending=True, stable=True
    )
    return ranking[..., :2].sort(dim=-1).values


def _positions_from_sparse_weight(
    dense_groups: torch.Tensor,
    sparse_groups: torch.Tensor,
    *,
    row_offset: int,
) -> torch.Tensor:
    structural = sparse_groups.ne(0)
    counts = structural.sum(dim=-1)
    if not bool(counts.eq(2).all().item()):
        first = counts.ne(2).nonzero(as_tuple=False)[0]
        raise ValueError(
            "weight24 must contain exactly two numerical nonzeros per K4: "
            f"row {row_offset + int(first[0])}, group {int(first[1])}"
        )
    positions = torch.arange(
        4, device=dense_groups.device, dtype=torch.int64
    ).view(1, 1, 4).expand_as(structural)
    retained = positions.masked_select(structural).view(
        *structural.shape[:-1], 2
    )
    if not torch.equal(
        dense_groups.gather(-1, retained), sparse_groups.gather(-1, retained)
    ):
        raise ValueError("weight24 retained values must match dense_weight exactly")
    return retained


def pack_sparse24_components(
    dense_weight: torch.Tensor,
    weight24: torch.Tensor | None = None,
    *,
    need_values: bool = True,
    need_residual: bool = True,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Pack canonical selectors and optional base/complement values."""

    _require_bf16_weight(dense_weight, "dense_weight")
    if weight24 is not None:
        _require_bf16_weight(weight24, "weight24")
        if weight24.shape != dense_weight.shape:
            raise ValueError("weight24 shape must match dense_weight")
        if weight24.device != dense_weight.device:
            raise ValueError("weight24 and dense_weight must share a device")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")

    n, k = dense_weight.shape
    metadata = torch.empty((n, k // 8), dtype=torch.uint8, device=dense_weight.device)
    packed = (
        torch.empty((n, k // 2), dtype=torch.bfloat16, device=dense_weight.device)
        if need_values
        else None
    )
    residual = (
        torch.empty((n, k // 2), dtype=torch.bfloat16, device=dense_weight.device)
        if need_residual
        else None
    )

    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        dense_groups = dense_weight[start:stop].view(stop - start, k // 4, 4)
        retained = (
            _top2_positions(dense_groups)
            if weight24 is None
            else _positions_from_sparse_weight(
                dense_groups,
                weight24[start:stop].view(stop - start, k // 4, 4),
                row_offset=start,
            )
        )
        nibble = retained[..., 0].bitwise_or(retained[..., 1].bitwise_left_shift(2))
        metadata[start:stop].copy_(pack_metadata(nibble.to(torch.uint8)))
        if packed is not None:
            packed[start:stop].copy_(
                dense_groups.gather(-1, retained).reshape(stop - start, k // 2)
            )
        if residual is not None:
            positions = torch.arange(
                4, device=dense_weight.device, dtype=torch.int64
            ).view(1, 1, 4).expand(stop - start, k // 4, 4)
            mask = torch.zeros_like(positions, dtype=torch.bool)
            mask.scatter_(-1, retained, True)
            complement = positions.masked_select(~mask).view(stop - start, k // 4, 2)
            residual[start:stop].copy_(
                dense_groups.gather(-1, complement).reshape(stop - start, k // 2)
            )
    return metadata.contiguous(), packed, residual


def _storage_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


@dataclass(frozen=True, slots=True)
class OnlineSparse24Weight:
    """Setup representation containing base, complement, and one metadata."""

    shape: tuple[int, int]
    metadata: torch.Tensor
    packed_values: torch.Tensor
    residual: torch.Tensor
    variant: str = SPARSE_RESIDUAL_SMEM

    def __post_init__(self) -> None:
        if self.variant != SPARSE_RESIDUAL_SMEM:
            raise ValueError(f"unsupported representation {self.variant!r}")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(not isinstance(value, int) or value <= 0 for value in self.shape)
        ):
            raise ValueError(f"invalid logical [N,K] shape {self.shape!r}")
        n, k = self.shape
        validate_metadata(self.metadata, self.shape)
        for tensor, name in (
            (self.packed_values, "packed_values"),
            (self.residual, "residual"),
        ):
            _require_bf16_matrix(tensor, name)
            if tuple(tensor.shape) != (n, k // 2):
                raise ValueError(f"{name} must have shape [N,K/2]")
            if tensor.device != self.metadata.device:
                raise ValueError(f"{name} and metadata must share a device")

    @property
    def n(self) -> int:
        return self.shape[0]

    @property
    def k(self) -> int:
        return self.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return self.metadata.device

    @property
    def storage_family(self) -> str:
        return "sparse_residual"

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "packed_values": self.packed_values,
            "metadata": self.metadata,
            "residual": self.residual,
        }

    def persistent_bytes(self) -> int:
        return sum(_storage_nbytes(tensor) for tensor in self.persistent_tensors().values())

    def persistent_bytes_by_component(self) -> dict[str, int]:
        return {
            name: _storage_nbytes(tensor)
            for name, tensor in self.persistent_tensors().items()
        }

    def reconstruct_sparse(self) -> torch.Tensor:
        retained, _ = positions_from_metadata(self.metadata)
        groups = torch.zeros(
            (self.n, self.k // 4, 4), dtype=torch.bfloat16, device=self.device
        )
        groups.scatter_(-1, retained, self.packed_values.view(self.n, self.k // 4, 2))
        return groups.view(self.shape).contiguous()

    def reconstruct_residual_sparse(self) -> torch.Tensor:
        _, complement = positions_from_metadata(self.metadata)
        groups = torch.zeros(
            (self.n, self.k // 4, 4), dtype=torch.bfloat16, device=self.device
        )
        groups.scatter_(-1, complement, self.residual.view(self.n, self.k // 4, 2))
        return groups.view(self.shape).contiguous()

    def reconstruct_dense(self) -> torch.Tensor:
        retained, complement = positions_from_metadata(self.metadata)
        groups = torch.empty(
            (self.n, self.k // 4, 4), dtype=torch.bfloat16, device=self.device
        )
        groups.scatter_(-1, retained, self.packed_values.view(self.n, self.k // 4, 2))
        groups.scatter_(-1, complement, self.residual.view(self.n, self.k // 4, 2))
        return groups.view(self.shape).contiguous()


def prepare_online_sparse24_weight(
    dense_weight: torch.Tensor,
    weight24: torch.Tensor | None = None,
    *,
    variant: str = SPARSE_RESIDUAL_SMEM,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
) -> OnlineSparse24Weight:
    if variant != SPARSE_RESIDUAL_SMEM:
        raise ValueError(f"unsupported representation {variant!r}")
    metadata, packed, residual = pack_sparse24_components(
        dense_weight, weight24, chunk_rows=chunk_rows
    )
    assert packed is not None and residual is not None
    return OnlineSparse24Weight(
        shape=tuple(dense_weight.shape),
        metadata=metadata,
        packed_values=packed,
        residual=residual,
    )


__all__ = [
    "OnlineSparse24Weight",
    "SPARSE24_METADATA_NIBBLES",
    "SPARSE_RESIDUAL_SMEM",
    "pack_sparse24_components",
    "prepare_online_sparse24_weight",
]
