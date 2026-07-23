"""Deterministic exact N:M masking shared by all external baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NMFormat:
    n: int
    m: int

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.n > self.m:
            raise ValueError(f"N:M must satisfy 0 < N <= M, got {self.n}:{self.m}")

    @property
    def label(self) -> str:
        return f"{self.n}:{self.m}"

    @property
    def density(self) -> float:
        return self.n / self.m


def parse_nm(value: str | NMFormat) -> NMFormat:
    if isinstance(value, NMFormat):
        return value
    fields = str(value).strip().split(":")
    if len(fields) != 2:
        raise ValueError(f"expected N:M such as 5:8 or 3:4, got {value!r}")
    try:
        return NMFormat(int(fields[0]), int(fields[1]))
    except ValueError as error:
        raise ValueError(f"invalid N:M format {value!r}") from error


def _balanced_group_counts(fmt: NMFormat) -> list[int]:
    """Allocate N survivors over K4 groups without starving a SparTA group.

    Flash-LLM and SpInfer do not require this distribution, but sharing one
    deterministic mask makes backend comparisons meaningful.  For density at
    least 1/2 and M divisible by four, every K4 receives at least two values,
    which lets the SpInfer artifact's SparTA decomposition form an exact 2:4
    base and a non-overlapping residual.
    """

    if fmt.m % 4 != 0:
        return []
    groups = fmt.m // 4
    base = fmt.n // groups
    remainder = fmt.n % groups
    counts = [base] * groups
    for index in range(remainder):
        counts[groups - remainder + index] += 1
    if any(count > 4 for count in counts):
        raise AssertionError("internal N:M group allocation exceeds four")
    return counts


def apply_nm_mask(
    weight: torch.Tensor,
    fmt: str | NMFormat,
    *,
    row_chunk: int = 256,
) -> torch.Tensor:
    """Keep exactly N largest-magnitude values per consecutive K-axis M group."""

    fmt = parse_nm(fmt)
    if weight.ndim != 2 or weight.shape[1] % fmt.m != 0:
        raise ValueError(
            f"weight must be [Nout,K] with K divisible by {fmt.m}, "
            f"got {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise ValueError("weight must have a floating-point dtype")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    # For the paper's 5:8 and 3:4 formats, make the distribution inside each
    # M-group deterministic and SparTA-compatible while preserving exact N:M.
    counts = _balanced_group_counts(fmt)
    result = torch.zeros_like(weight)
    for start in range(0, weight.shape[0], row_chunk):
        stop = min(weight.shape[0], start + row_chunk)
        # A generated BF16 normal tensor can very rarely contain an exact zero.
        # Both artifact compressors use value!=0 as structural metadata, so
        # replace such candidates before selecting the structural mask.
        source = weight[start:stop].masked_fill(
            weight[start:stop].eq(0), torch.finfo(weight.dtype).tiny
        )
        if counts and min(counts) >= 2:
            pieces = source.reshape(source.shape[0], -1, fmt.m).split(4, dim=-1)
            kept = []
            for piece, count in zip(pieces, counts, strict=True):
                indices = piece.abs().topk(count, dim=-1, sorted=False).indices
                mask = torch.zeros_like(piece, dtype=torch.bool)
                mask.scatter_(-1, indices, True)
                kept.append(piece.masked_fill(~mask, 0))
            result[start:stop] = torch.cat(kept, dim=-1).reshape_as(source)
        else:
            grouped = source.reshape(source.shape[0], -1, fmt.m)
            indices = grouped.abs().topk(fmt.n, dim=-1, sorted=False).indices
            mask = torch.zeros_like(grouped, dtype=torch.bool)
            mask.scatter_(-1, indices, True)
            result[start:stop] = grouped.masked_fill(~mask, 0).reshape_as(source)

    validate_nm(result, fmt)
    return result.contiguous()


def validate_nm(weight: torch.Tensor, fmt: str | NMFormat) -> None:
    fmt = parse_nm(fmt)
    if weight.ndim != 2 or weight.shape[1] % fmt.m != 0:
        raise ValueError("weight shape is incompatible with N:M")
    for start in range(0, weight.shape[0], 256):
        stop = min(weight.shape[0], start + 256)
        counts = (
            weight[start:stop].reshape(stop - start, -1, fmt.m).ne(0).sum(dim=-1)
        )
        if not bool(counts.eq(fmt.n).all().item()):
            first = counts.ne(fmt.n).nonzero(as_tuple=False)[0]
            raise ValueError(
                f"weight is not exact {fmt.label}: row={start + int(first[0])}, "
                f"group={int(first[1])}, nnz={int(counts[tuple(first)].item())}"
            )


def split_sparta_base_residual(
    weight_nm: torch.Tensor,
    fmt: str | NMFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce SpInfer's SparTA split: first 2 values/K4 plus residual.

    The at-most-2:4 base is suitable for cuSPARSELt and the residual is suitable
    for SpInfer.  This is exact even when a K4 contains fewer than two values.
    """

    fmt = parse_nm(fmt)
    validate_nm(weight_nm, fmt)
    if weight_nm.shape[1] % 4 != 0:
        raise ValueError("SparTA decomposition requires K divisible by four")
    base = torch.zeros_like(weight_nm)
    for start in range(0, weight_nm.shape[0], 256):
        stop = min(weight_nm.shape[0], start + 256)
        groups = weight_nm[start:stop].reshape(stop - start, -1, 4)
        indices = groups.abs().topk(2, dim=-1, sorted=False).indices
        base_mask = torch.zeros_like(groups, dtype=torch.bool)
        base_mask.scatter_(-1, indices, True)
        base[start:stop] = groups.masked_fill(~base_mask, 0).reshape(
            stop - start, weight_nm.shape[1]
        )
    base = base.contiguous()
    residual = (weight_nm - base).contiguous()
    if not torch.equal(base + residual, weight_nm):
        raise AssertionError("SparTA split does not reconstruct the N:M weight")
    return base, residual
