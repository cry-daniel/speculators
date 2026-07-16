"""Packing helpers for 2:4 structured sparse linear weights.

The public linear semantics are always ``Y = X @ W`` with dense activations
``X[M, K]`` and a 2:4 sparse weight ``W[K, N]``.  Compression is offline and is
not included in benchmark latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Sparse24Layout = Literal["k_major", "n_major"]

PATTERN_TO_POS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)

PATTERN_TO_ORDERED_META_NIBBLE: tuple[int, ...] = (
    0x4,  # (0, 1)
    0x8,  # (0, 2)
    0xC,  # (0, 3)
    0x9,  # (1, 2)
    0xD,  # (1, 3)
    0xE,  # (2, 3)
)


@dataclass(frozen=True)
class Packed24Weight:
    """Compressed 2:4 weight.

    Attributes:
        values: Nonzero values.  ``k_major`` layout is ``[K / 4, N, 2]`` and
            ``n_major`` layout is ``[N, K / 4, 2]``.
        meta: Pattern IDs in ``[0, 5]``.  ``k_major`` layout is ``[K / 4, N]``
            and ``n_major`` layout is ``[N, K / 4]``.
        K: Original dense weight K dimension.
        N: Original dense weight N dimension.
        layout: Compression layout used by ``values`` and ``meta``.
    """

    values: torch.Tensor
    meta: torch.Tensor
    K: int
    N: int
    layout: Sparse24Layout


def _check_weight_shape(W: torch.Tensor) -> tuple[int, int, int]:
    if W.ndim != 2:
        raise ValueError(f"W must be rank 2 [K, N], got shape {tuple(W.shape)}")
    K, N = W.shape
    if K % 4 != 0:
        raise ValueError(f"K must be divisible by 4 for 2:4, got K={K}")
    return K, N, K // 4


def _normalize_layout(layout: str) -> Sparse24Layout:
    if layout not in ("k_major", "n_major"):
        raise ValueError(f"unsupported 2:4 layout {layout!r}")
    return layout  # type: ignore[return-value]


def assert_24_weight(W: torch.Tensor) -> None:
    """Assert every ``W[4*g:4*g+4, n]`` group has exactly two nonzeros.

    Args:
        W: Dense materialized 2:4 weight with shape ``[K, N]``.

    Raises:
        AssertionError: If any K-axis group has a nonzero count other than two.
        ValueError: If ``W`` is not rank-2 or ``K`` is not divisible by 4.
    """

    K, N, groups = _check_weight_shape(W)
    counts = W.reshape(groups, 4, N).ne(0).sum(dim=1)
    bad = torch.nonzero(counts.ne(2), as_tuple=False)
    if bad.numel() == 0:
        return
    g, n = bad[0].tolist()
    count = int(counts[g, n].item())
    raise AssertionError(
        f"W is not 2:4 at group={g}, n={n}: nonzero_count={count}; "
        f"shape=({K}, {N})"
    )


def apply_random_24_mask(
    W: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent random 2-of-4 K-axis mask to dense ``W[K, N]``.

    The random pattern is sampled independently for every ``(K / 4, N)`` group,
    so all six legal patterns can appear in different output channels.

    Returns:
        A tuple ``(W_24, meta)`` where ``W_24`` is dense materialized 2:4 and
        ``meta`` is the ``k_major`` pattern tensor with shape ``[K / 4, N]``.
    """

    _K, N, groups = _check_weight_shape(W)
    meta = torch.randint(
        0,
        len(PATTERN_TO_POS),
        (groups, N),
        device=W.device,
        generator=generator,
        dtype=torch.int64,
    )
    pos_lut = torch.tensor(PATTERN_TO_POS, device=W.device, dtype=torch.int64)
    pos = torch.arange(4, device=W.device).view(1, 4, 1)
    pos0 = pos_lut[meta, 0].view(groups, 1, N)
    pos1 = pos_lut[meta, 1].view(groups, 1, N)
    mask = (pos == pos0) | (pos == pos1)
    W24 = torch.where(
        mask,
        W.reshape(groups, 4, N),
        torch.zeros((), device=W.device, dtype=W.dtype),
    )
    return W24.reshape_as(W), meta.to(torch.uint8)


def pack_24(W: torch.Tensor, *, layout: Sparse24Layout = "k_major") -> Packed24Weight:
    """Pack dense materialized 2:4 ``W[K, N]`` into values and metadata.

    The input must already be 2:4 sparse.  Use ``apply_random_24_mask`` or an
    activation-aware pruning method before calling this function.
    """

    layout = _normalize_layout(layout)
    K, N, groups = _check_weight_shape(W)
    assert_24_weight(W)
    Wg = W.reshape(groups, 4, N)
    nz = Wg.ne(0)
    meta = torch.zeros((groups, N), device=W.device, dtype=torch.uint8)
    values = torch.zeros((groups, N, 2), device=W.device, dtype=W.dtype)
    matched = torch.zeros((groups, N), device=W.device, dtype=torch.bool)

    for pattern_id, (p0, p1) in enumerate(PATTERN_TO_POS):
        pattern_mask = nz[:, p0, :] & nz[:, p1, :]
        for p in range(4):
            if p not in (p0, p1):
                pattern_mask = pattern_mask & ~nz[:, p, :]
        meta = torch.where(pattern_mask, torch.full_like(meta, pattern_id), meta)
        values[:, :, 0] = torch.where(pattern_mask, Wg[:, p0, :], values[:, :, 0])
        values[:, :, 1] = torch.where(pattern_mask, Wg[:, p1, :], values[:, :, 1])
        matched |= pattern_mask

    if not bool(matched.all().item()):
        bad = torch.nonzero(~matched, as_tuple=False)[0].tolist()
        raise AssertionError(
            f"failed to encode 2:4 pattern at group={bad[0]}, n={bad[1]}"
        )

    if layout == "n_major":
        return Packed24Weight(
            values=values.permute(1, 0, 2).contiguous(),
            meta=meta.t().contiguous(),
            K=K,
            N=N,
            layout=layout,
        )
    return Packed24Weight(
        values=values.contiguous(),
        meta=meta.contiguous(),
        K=K,
        N=N,
        layout=layout,
    )


def pack_24_from_n_major_group_bytes(
    weight_nk: torch.Tensor,
    group_bytes: torch.Tensor,
    row_scale: torch.Tensor | None = None,
) -> Packed24Weight:
    """Pack ``weight_nk[N, K]`` from n-major 2:4 keep-bit masks.

    This is the low-memory path for dynamic vLLM routing. It produces the same
    representation as ``pack_24(masked_weight.t(), layout="n_major")`` without
    materializing the masked ``[K, N]`` transpose or running dense nonzero-count
    validation over it.
    """

    if weight_nk.ndim != 2:
        raise ValueError(
            f"weight_nk must be rank 2 [N, K], got shape {tuple(weight_nk.shape)}"
        )
    N, K = weight_nk.shape
    if K % 4 != 0:
        raise ValueError(f"K must be divisible by 4 for 2:4, got K={K}")
    groups = K // 4
    if tuple(group_bytes.shape) != (N, groups):
        raise ValueError(
            f"group_bytes must have shape {(N, groups)}, got {tuple(group_bytes.shape)}"
        )
    if row_scale is not None and row_scale.numel() != N:
        raise ValueError(f"row_scale length {row_scale.numel()} does not match N={N}")

    group_bytes = group_bytes.to(device=weight_nk.device, dtype=torch.uint8)
    meta = torch.empty((N, groups), device=weight_nk.device, dtype=torch.uint8)
    matched = torch.zeros((N, groups), device=weight_nk.device, dtype=torch.bool)
    for pattern_id, (p0, p1) in enumerate(PATTERN_TO_POS):
        pattern_mask = group_bytes.eq((1 << p0) | (1 << p1))
        meta = torch.where(pattern_mask, torch.full_like(meta, pattern_id), meta)
        matched |= pattern_mask
    if not bool(matched.all().item()):
        bad = (~matched).nonzero(as_tuple=False)
        n, group = bad[0].tolist()
        raise AssertionError(
            f"group_bytes is not 2:4 at n={n}, group={group}: "
            f"mask=0x{int(group_bytes[n, group].item()):x}"
        )
    del matched

    wg = weight_nk.reshape(N, groups, 4)
    values = torch.empty((N, groups, 2), device=weight_nk.device, dtype=weight_nk.dtype)
    for pattern_id, (p0, p1) in enumerate(PATTERN_TO_POS):
        pattern_mask = meta.eq(pattern_id)
        values[:, :, 0] = torch.where(pattern_mask, wg[:, :, p0], values[:, :, 0])
        values[:, :, 1] = torch.where(pattern_mask, wg[:, :, p1], values[:, :, 1])
    if row_scale is not None:
        scale = row_scale.to(device=weight_nk.device, dtype=weight_nk.dtype)
        values = values * scale.view(N, 1, 1)

    return Packed24Weight(
        values=values.contiguous(),
        meta=meta.contiguous(),
        K=K,
        N=N,
        layout="n_major",
    )


def pattern_meta_to_ordered_nibbles(meta: torch.Tensor) -> torch.Tensor:
    """Convert pattern IDs ``0..5`` to PTX ordered-metadata 4-bit values.

    PTX ``mma.sp::ordered_metadata`` for f16/bf16 2:4 accepts the six nibbles
    ``0x4, 0x8, 0x9, 0xc, 0xd, 0xe``. The returned tensor keeps one nibble per
    input element; packing those nibbles into the per-lane 32-bit metadata
    register is kernel-layout dependent and is handled separately.
    """

    if meta.dtype != torch.uint8:
        raise ValueError(f"meta must be torch.uint8 pattern IDs, got {meta.dtype}")
    bad = (meta > 5).nonzero(as_tuple=False)
    if bad.numel() != 0:
        idx = bad[0].tolist()
        raise ValueError(f"meta pattern ID out of range at {idx}: {int(meta[tuple(idx)])}")
    lut = torch.tensor(
        PATTERN_TO_ORDERED_META_NIBBLE,
        device=meta.device,
        dtype=torch.uint8,
    )
    return lut[meta.long()]


def pack_ordered_metadata_u32(meta: torch.Tensor) -> torch.Tensor:
    """Pack pattern IDs into PTX ordered metadata words.

    The input stores one pattern ID per logical 4-wide sparse group.  The output
    packs eight PTX ordered-metadata nibbles into one 32-bit word along the last
    dimension. This is the high-level ordered-metadata word used by local
    emulation and diagnostics; the CUTLASS backend applies the required
    shared-memory E-layout swizzle when it feeds arbitrary patterns to
    ``mma.sp``.
    """

    if meta.ndim < 1:
        raise ValueError("meta must have at least one dimension")
    groups = meta.shape[-1]
    if groups % 8 != 0:
        raise ValueError(f"metadata group count must be divisible by 8, got {groups}")
    ordered = pattern_meta_to_ordered_nibbles(meta)
    chunks = ordered.reshape(*ordered.shape[:-1], groups // 8, 8).to(torch.int64)
    shifts = torch.arange(8, device=meta.device, dtype=torch.int64) * 4
    packed = (chunks << shifts).sum(dim=-1)
    return packed.to(torch.uint32)


def pack_mma_sp_operand_a(
    values: torch.Tensor,
    meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack compressed 2:4 rows for ``mma.sp`` sparse operand A.

    Args:
        values: Nonzero values with shape ``[..., K / 4, 2]``. For this repo's
            linear convention, pass ``pack_24(W, layout="n_major").values`` so
            the rows are ``W.T`` rows.
        meta: Pattern IDs with shape ``[..., K / 4]`` matching ``values``.

    Returns:
        ``(a_values, a_meta)`` where ``a_values`` is ``[..., K / 2]`` and
        ``a_meta`` is ``[..., K / 32]`` with dtype ``torch.uint32``.
    """

    if values.ndim < 2 or values.shape[-1] != 2:
        raise ValueError(
            f"values must have shape [..., groups, 2], got {tuple(values.shape)}"
        )
    if meta.shape != values.shape[:-1]:
        raise ValueError(
            f"values/meta shape mismatch: values={tuple(values.shape)}, meta={tuple(meta.shape)}"
        )
    groups = values.shape[-2]
    if groups % 8 != 0:
        raise ValueError(f"K / 4 group count must be divisible by 8, got {groups}")
    a_values = values.contiguous().reshape(*values.shape[:-2], groups * 2)
    a_meta = pack_ordered_metadata_u32(meta)
    return a_values, a_meta


def decompress_24(
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    layout: Sparse24Layout = "k_major",
) -> torch.Tensor:
    """Decompress 2:4 ``values`` and ``meta`` into dense ``W[K, N]``.

    Args:
        values: ``[K / 4, N, 2]`` for ``k_major`` or ``[N, K / 4, 2]`` for
            ``n_major``.
        meta: Pattern IDs with matching leading dimensions.
        layout: Compression layout.

    Returns:
        Dense materialized 2:4 weight ``[K, N]``.
    """

    layout = _normalize_layout(layout)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"values must have shape [*, *, 2], got {tuple(values.shape)}")
    if meta.ndim != 2:
        raise ValueError(f"meta must be rank 2, got {tuple(meta.shape)}")
    if layout == "n_major":
        values = values.permute(1, 0, 2).contiguous()
        meta = meta.t().contiguous()
    groups, N, two = values.shape
    if two != 2 or meta.shape != (groups, N):
        raise ValueError(
            f"values/meta shape mismatch: values={tuple(values.shape)}, meta={tuple(meta.shape)}"
        )
    Wg = torch.zeros((groups, 4, N), device=values.device, dtype=values.dtype)
    for pattern_id, (p0, p1) in enumerate(PATTERN_TO_POS):
        pattern_mask = meta.eq(pattern_id)
        Wg[:, p0, :] = torch.where(pattern_mask, values[:, :, 0], Wg[:, p0, :])
        Wg[:, p1, :] = torch.where(pattern_mask, values[:, :, 1], Wg[:, p1, :])
    return Wg.reshape(groups * 4, N)
