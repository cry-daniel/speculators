# SPDX-License-Identifier: Apache-2.0
"""Env-gated TLM-only activation-aware 2:4 masking for SpecLink studies.

This module is intentionally independent from the eval scripts so vLLM can
apply masks at model-load time without importing experiment-only code.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch


TARGET_LEAFS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
}

FUSED_CACHE_LEAFS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

TOKEN_DENSE_PROJECTION_LEAFS = {
    "none": set(),
    "all": TARGET_LEAFS,
    "attention": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
    },
    "mlp": {
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
    },
    "gate_up": {
        "gate_proj",
        "up_proj",
        "gate_up_proj",
    },
    "down": {"down_proj"},
    "qkv": {"q_proj", "k_proj", "v_proj", "qkv_proj"},
    "o": {"o_proj"},
    "o_gate_up": {
        "o_proj",
        "gate_proj",
        "up_proj",
        "gate_up_proj",
    },
    "qkv_down": {
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "down_proj",
    },
    "qkv_gate_up_down": {
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "gate_proj",
        "up_proj",
        "gate_up_proj",
        "down_proj",
    },
    "o_down": {"o_proj", "down_proj"},
    "attention_down": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "down_proj",
    },
    "attention_gate_up": {
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "gate_up_proj",
    },
}

_MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)
_BIT_COUNTS = torch.tensor(
    [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4],
    dtype=torch.int16,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _layer_index(module_name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    if not match:
        return None
    return int(match.group(1))


def _module_leaf(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


@torch.no_grad()
def _pack_qkv_cusparselt_24(
    weight: torch.Tensor,
    group_bytes: torch.Tensor,
) -> torch.Tensor:
    """Create the cuSPARSELt full-mask pack used by large-M QKV."""

    from torch.sparse import to_sparse_semi_structured
    from torch.sparse.semi_structured import SparseSemiStructuredTensor

    out_features, in_features = map(int, weight.shape)
    groups = in_features // 4
    bits = _MASK_BITS.to(device=weight.device)
    keep = (
        group_bytes.to(device=weight.device, dtype=torch.uint8).unsqueeze(-1)
        & bits.view(1, 1, 4)
    ).ne(0)
    masked = weight.contiguous().clone()
    masked.view(out_features, groups, 4).masked_fill_(~keep, 0)
    previous = bool(SparseSemiStructuredTensor._FORCE_CUTLASS)
    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
        packed = to_sparse_semi_structured(masked).packed
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = previous
    return packed


def _parse_layer_indices(value: str) -> set[int]:
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start < 0 or end < 0:
                raise ValueError("layer indices must be non-negative")
            step = 1 if end >= start else -1
            layers.update(range(start, end + step, step))
        else:
            layer = int(part)
            if layer < 0:
                raise ValueError("layer indices must be non-negative")
            layers.add(layer)
    return layers


def _token_dense_projection_enabled(policy: str, leaf: str) -> bool:
    selected = TOKEN_DENSE_PROJECTION_LEAFS.get(policy)
    if selected is None:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_PROJECTION_POLICY must be one of "
            f"{sorted(TOKEN_DENSE_PROJECTION_LEAFS)}, got {policy!r}"
        )
    return leaf in selected


def _module_is_skipped(module_name: str) -> bool:
    lowered = module_name.lower()
    return (
        module_name == "lm_head"
        or module_name.endswith(".lm_head")
        or "embed_tokens" in lowered
        or "embedding" in lowered
        or ".wte" in lowered
    )


def _iter_target_modules(model: Any) -> list[tuple[str, Any, torch.Tensor]]:
    out: list[tuple[str, Any, torch.Tensor]] = []
    for name, module in model.named_modules():
        if _module_is_skipped(name):
            continue
        leaf = _module_leaf(name)
        if leaf not in TARGET_LEAFS:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        usable_in = (int(weight.shape[1]) // 4) * 4
        if usable_in <= 0:
            continue
        out.append((name, module, weight))
    return out


def _load_activation_scales(cache_root: Path, model_label: str) -> dict[str, torch.Tensor]:
    tensor_path = cache_root / f"{model_label}.pt"
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"missing activation RMS cache for {model_label}: {tensor_path}"
        )
    scales = torch.load(tensor_path, map_location="cpu")
    if not isinstance(scales, dict) or not scales:
        raise RuntimeError(f"invalid activation RMS cache: {tensor_path}")
    return {str(name): value for name, value in scales.items()}


def _load_mask_cache(cache_path: str) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"missing structured 2:4 mask cache: {path}")
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict) or "masks" not in cache:
        raise RuntimeError(f"invalid structured 2:4 mask cache: {path}")
    return cache


def _load_group_covariance_cache(cache_path: str) -> dict[str, torch.Tensor]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"missing grouped covariance cache: {path}")
    cache = torch.load(path, map_location="cpu")
    covariances = cache.get("covariances") if isinstance(cache, dict) else None
    if not isinstance(covariances, dict) or not covariances:
        raise RuntimeError(f"invalid grouped covariance cache: {path}")
    return {str(name): value for name, value in covariances.items()}


def _scale_for_module(
    module_name: str,
    activation_scales: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if module_name in activation_scales:
        return activation_scales[module_name]

    leaf = _module_leaf(module_name)
    candidate_leafs = FUSED_CACHE_LEAFS.get(leaf, (leaf,))
    prefix = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    for candidate_leaf in candidate_leafs:
        candidate = f"{prefix}.{candidate_leaf}" if prefix else candidate_leaf
        if candidate in activation_scales:
            return activation_scales[candidate]
    return None


def _group_covariance_for_module(
    module_name: str,
    covariances: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if module_name in covariances:
        return covariances[module_name]
    leaf = _module_leaf(module_name)
    candidate_leafs = FUSED_CACHE_LEAFS.get(leaf, ())
    prefix = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    for candidate_leaf in candidate_leafs:
        candidate = f"{prefix}.{candidate_leaf}" if prefix else candidate_leaf
        covariance = covariances.get(candidate)
        if covariance is not None:
            return covariance
    return None


def _cache_values_for_module(
    module_name: str,
    cache: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    masks = cache.get("masks", {})
    row_scales = cache.get("row_scales", {})
    if module_name in masks:
        return masks[module_name], row_scales.get(module_name)

    leaf = _module_leaf(module_name)
    candidate_leafs = FUSED_CACHE_LEAFS.get(leaf)
    if not candidate_leafs:
        return None, None
    prefix = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    mask_parts = []
    scale_parts = []
    for candidate_leaf in candidate_leafs:
        candidate = f"{prefix}.{candidate_leaf}" if prefix else candidate_leaf
        mask = masks.get(candidate)
        if mask is None:
            return None, None
        mask_parts.append(mask)
        scale = row_scales.get(candidate)
        if scale is not None:
            scale_parts.append(scale)
    scale_out = torch.cat(scale_parts, dim=0) if len(scale_parts) == len(mask_parts) else None
    return torch.cat(mask_parts, dim=0), scale_out


def _expand_cached_mask_bytes(
    mask_bytes: torch.Tensor,
    *,
    out_features: int,
    groups: int,
) -> torch.Tensor:
    expected = (out_features, groups)
    if tuple(mask_bytes.shape) == expected:
        return mask_bytes.to(dtype=torch.uint8)

    packed_expected = (out_features, (groups + 1) // 2)
    if tuple(mask_bytes.shape) == packed_expected:
        packed = mask_bytes.to(dtype=torch.uint8)
        unpacked = torch.empty(
            (out_features, packed_expected[1] * 2),
            dtype=torch.uint8,
            device=packed.device,
        )
        unpacked[:, 0::2] = packed & 0x0F
        unpacked[:, 1::2] = (packed >> 4) & 0x0F
        return unpacked[:, :groups]

    raise RuntimeError(
        f"cached 2:4 mask shape {tuple(mask_bytes.shape)} does not match "
        f"{expected} or packed {packed_expected}"
    )


@torch.no_grad()
def _variance_preserving_row_scale(
    weight: torch.Tensor,
    group_bytes: torch.Tensor,
    activation_scale: torch.Tensor | None,
    *,
    max_scale: float,
    chunk_groups: int = 128,
) -> torch.Tensor:
    """Match each dense output row's activation-weighted second moment."""

    out_features, in_features = map(int, weight.shape)
    groups = in_features // 4
    if tuple(group_bytes.shape) != (out_features, groups):
        raise RuntimeError(
            "variance row scale mask shape mismatch: "
            f"got {tuple(group_bytes.shape)}, expected {(out_features, groups)}"
        )
    if not math.isfinite(max_scale) or max_scale < 1.0:
        raise RuntimeError(
            "SPECLINK_SPARSE24_ROW_SCALE_MAX must be finite and at least 1.0"
        )

    if activation_scale is None:
        activation_energy = torch.ones(
            in_features,
            device=weight.device,
            dtype=torch.float32,
        )
    else:
        if activation_scale.numel() < in_features:
            raise RuntimeError(
                "activation RMS length is shorter than the sparse weight input: "
                f"{activation_scale.numel()} < {in_features}"
            )
        activation_energy = activation_scale[:in_features].to(
            device=weight.device,
            dtype=torch.float32,
        ).square()

    total_energy = torch.zeros(
        out_features,
        device=weight.device,
        dtype=torch.float32,
    )
    kept_energy = torch.zeros_like(total_energy)
    group_bytes = group_bytes.to(device=weight.device, dtype=torch.uint8)
    for group_start in range(0, groups, chunk_groups):
        group_end = min(group_start + chunk_groups, groups)
        column_start = group_start * 4
        column_end = group_end * 4
        energy = weight[:, column_start:column_end].float().square()
        energy.mul_(activation_energy[column_start:column_end].unsqueeze(0))
        energy = energy.view(out_features, group_end - group_start, 4)
        total_energy.add_(energy.sum(dim=(1, 2)))
        chunk_mask = group_bytes[:, group_start:group_end]
        for position in range(4):
            kept_energy.add_(
                (
                    energy[:, :, position]
                    * ((chunk_mask >> position) & 1).to(dtype=energy.dtype)
                ).sum(dim=1)
            )

    positive = total_energy > 0
    scale = torch.ones_like(total_energy)
    scale[positive] = torch.sqrt(
        total_energy[positive]
        / kept_energy[positive].clamp_min(torch.finfo(torch.float32).tiny)
    )
    return scale.clamp_(min=1.0, max=max_scale).to(dtype=weight.dtype)


@torch.no_grad()
def _reconstruct_grouped_24_weight(
    weight: torch.Tensor,
    group_bytes: torch.Tensor,
    covariance: torch.Tensor,
    *,
    damping: float = 1e-4,
    max_ratio: float = 2.0,
    row_chunk: int = 512,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit retained weights to dense group outputs under a 4x4 covariance."""

    out_features, in_features = map(int, weight.shape)
    groups = in_features // 4
    if tuple(group_bytes.shape) != (out_features, groups):
        raise RuntimeError("group reconstruction mask shape mismatch")
    if covariance.ndim != 3 or tuple(covariance.shape[1:]) != (4, 4):
        raise RuntimeError(
            f"group covariance must have shape [groups,4,4], got {tuple(covariance.shape)}"
        )
    if int(covariance.shape[0]) < groups:
        raise RuntimeError(
            f"group covariance is too short: {int(covariance.shape[0])} < {groups}"
        )
    if not math.isfinite(damping) or damping < 0.0:
        raise RuntimeError("group reconstruction damping must be finite and non-negative")
    if not math.isfinite(max_ratio) or max_ratio < 1.0:
        raise RuntimeError("group reconstruction max_ratio must be at least 1.0")

    covariance = covariance[:groups].to(
        device=weight.device,
        dtype=torch.float32,
    )
    group_bytes = group_bytes.to(device=weight.device, dtype=torch.uint8)
    reconstructed = torch.zeros_like(weight)
    if groups * 4 < in_features:
        reconstructed[:, groups * 4 :].copy_(weight[:, groups * 4 :])
    reconstructed_view = reconstructed[:, : groups * 4].view(
        out_features, groups, 4
    )
    diagonal_mean = covariance.diagonal(dim1=1, dim2=2).mean(dim=1)
    regularizer = damping * diagonal_mean + 1e-8
    options = (
        (0x3, 0, 1),
        (0x5, 0, 2),
        (0x9, 0, 3),
        (0x6, 1, 2),
        (0xA, 1, 3),
        (0xC, 2, 3),
    )
    dense_error = 0.0
    reconstructed_error = 0.0
    max_coefficient_ratio = 0.0

    for start in range(0, out_features, row_chunk):
        end = min(start + row_chunk, out_features)
        dense_group = (
            weight[start:end, : groups * 4]
            .float()
            .view(end - start, groups, 4)
        )
        fitted = torch.zeros_like(dense_group)
        limit = dense_group.abs().amax(dim=2).clamp_min_(1e-8) * max_ratio
        for option_byte, first, second in options:
            selected = group_bytes[start:end].eq(option_byte)
            rhs_first = torch.einsum(
                "gk,rgk->rg", covariance[:, first, :], dense_group
            )
            rhs_second = torch.einsum(
                "gk,rgk->rg", covariance[:, second, :], dense_group
            )
            c00 = covariance[:, first, first] + regularizer
            c11 = covariance[:, second, second] + regularizer
            c01 = covariance[:, first, second]
            determinant = (c00 * c11 - c01.square()).clamp_min_(1e-12)
            coefficient_first = (
                rhs_first * c11 - rhs_second * c01
            ) / determinant
            coefficient_second = (
                rhs_second * c00 - rhs_first * c01
            ) / determinant
            coefficient_first.clamp_(min=-limit, max=limit)
            coefficient_second.clamp_(min=-limit, max=limit)
            fitted[:, :, first] = torch.where(
                selected, coefficient_first, fitted[:, :, first]
            )
            fitted[:, :, second] = torch.where(
                selected, coefficient_second, fitted[:, :, second]
            )
        reconstructed_view[start:end].copy_(fitted.to(dtype=weight.dtype))

        original_sparse = dense_group.clone()
        for position in range(4):
            original_sparse[:, :, position].mul_(
                ((group_bytes[start:end] >> position) & 1).to(torch.float32)
            )
        original_delta = dense_group - original_sparse
        fitted_delta = dense_group - fitted
        dense_error += float(
            torch.einsum(
                "rgi,gij,rgj->", original_delta, covariance, original_delta
            ).item()
        )
        reconstructed_error += float(
            torch.einsum(
                "rgi,gij,rgj->", fitted_delta, covariance, fitted_delta
            ).item()
        )
        ratio = fitted.abs() / dense_group.abs().amax(dim=2, keepdim=True).clamp_min_(
            1e-8
        )
        max_coefficient_ratio = max(
            max_coefficient_ratio,
            float(ratio.max().item()),
        )

    return reconstructed, {
        "group_reconstruction_error_ratio": (
            reconstructed_error / dense_error if dense_error > 0.0 else 0.0
        ),
        "group_reconstruction_max_coefficient_ratio": max_coefficient_ratio,
    }


def _selected_layers(modules: list[tuple[str, Any, torch.Tensor]]) -> list[int]:
    return sorted(
        {
            layer
            for name, _, _ in modules
            for layer in [_layer_index(name)]
            if layer is not None
        }
    )


def _should_keep_dense(
    *,
    policy: str,
    layer: int | None,
    layers: list[int],
    keep_n: int,
) -> bool:
    if layer is None:
        return False
    keep_n = max(0, keep_n)
    if keep_n == 0:
        return False
    if policy == "keep_first":
        return layer in set(layers[:keep_n])
    if policy == "keep_last":
        return layer in set(layers[-keep_n:])
    if policy == "keep_first_last":
        return layer in set(layers[:keep_n]) or layer in set(layers[-keep_n:])
    return False


def _compute_keep_mask_24(
    weight: torch.Tensor,
    activation_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, str]:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    view = weight[:, :usable_in].view(out_features, usable_in // 4, 4)
    score = view.detach().abs().float()
    method = "magnitude"
    if activation_scale is not None and activation_scale.numel() >= usable_in:
        scale = activation_scale[:usable_in].to(device=weight.device, dtype=score.dtype)
        score = score * scale.view(1, usable_in // 4, 4)
        method = "activation_aware"
    keep_idx = score.topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return keep, method


def _pack_keep_mask(keep: torch.Tensor) -> torch.Tensor:
    bits = _MASK_BITS.to(device=keep.device)
    group_bytes = (keep.to(torch.uint8) * bits.view(1, 1, 4)).sum(dim=-1)
    groups = int(group_bytes.shape[1])
    if groups % 2:
        pad = torch.zeros(
            (int(group_bytes.shape[0]), 1),
            dtype=group_bytes.dtype,
            device=group_bytes.device,
        )
        group_bytes = torch.cat([group_bytes, pad], dim=1)
    packed = group_bytes[:, 0::2] | (group_bytes[:, 1::2] << 4)
    return packed.cpu()


def _count_zeroed_from_group_bytes(mask_bytes: torch.Tensor, total: int) -> int:
    group_bytes = mask_bytes.to(dtype=torch.long)
    bit_counts = _BIT_COUNTS.to(device=group_bytes.device)
    kept = int(bit_counts[group_bytes & 0x0F].sum().item())
    return total - kept


def _mask_weight_24(
    weight: torch.Tensor,
    activation_scale: torch.Tensor | None,
) -> tuple[int, int, str]:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    keep, method = _compute_keep_mask_24(weight, activation_scale)
    with torch.no_grad():
        view = weight[:, :usable_in].view(out_features, usable_in // 4, 4)
        view.masked_fill_(~keep, 0)
    total = out_features * usable_in
    zeroed = int((~keep).sum().item())
    return total, zeroed, method


def _attach_computed_mask_24(
    module: Any,
    weight: torch.Tensor,
    activation_scale: torch.Tensor | None,
) -> tuple[int, int, str]:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    keep, method = _compute_keep_mask_24(weight, activation_scale)
    module._speclink_24_mask_bytes = _pack_keep_mask(keep).to(
        device=weight.device, non_blocking=True
    )
    module._speclink_24_row_scale = None
    total = out_features * usable_in
    zeroed = int((~keep).sum().item())
    return total, zeroed, f"token_dense_{method}"


def _apply_cached_mask_24(
    weight: torch.Tensor,
    mask_bytes: torch.Tensor,
    row_scale: torch.Tensor | None,
) -> tuple[int, int, str]:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    mask_bytes = _expand_cached_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=groups,
    )
    bits = _MASK_BITS.to(device=weight.device)
    keep = (
        mask_bytes.to(device=weight.device, dtype=torch.uint8).unsqueeze(-1)
        & bits.view(1, 1, 4)
    ).ne(0)
    with torch.no_grad():
        view = weight[:, :usable_in].view(out_features, groups, 4)
        view.masked_fill_(~keep, 0)
        if row_scale is not None:
            scale = row_scale.to(device=weight.device, dtype=weight.dtype)
            if scale.numel() != out_features:
                raise RuntimeError(
                    f"cached row_scale length {scale.numel()} does not match {out_features}"
                )
            weight[:, :usable_in].mul_(scale.view(-1, 1))
    total = out_features * usable_in
    zeroed = int((~keep).sum().item())
    return total, zeroed, "cached_mask_row_scale" if row_scale is not None else "cached_mask"


def _attach_cached_mask_24(
    module: Any,
    weight: torch.Tensor,
    mask_bytes: torch.Tensor,
    row_scale: torch.Tensor | None,
) -> tuple[int, int, str]:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    group_bytes = _expand_cached_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=groups,
    )
    module._speclink_24_mask_bytes = _pack_group_bytes(group_bytes).to(
        device=weight.device, non_blocking=True
    )
    module._speclink_24_row_scale = (
        row_scale.to(device=weight.device, non_blocking=True)
        if row_scale is not None
        else None
    )
    total = out_features * usable_in
    zeroed = _count_zeroed_from_group_bytes(group_bytes, total)
    method = "cached_mask_row_scale" if row_scale is not None else "cached_mask"
    return total, zeroed, f"token_dense_{method}"


def _pack_group_bytes(group_bytes: torch.Tensor) -> torch.Tensor:
    group_bytes = group_bytes.to(dtype=torch.uint8)
    groups = int(group_bytes.shape[1])
    if groups % 2:
        pad = torch.zeros(
            (int(group_bytes.shape[0]), 1),
            dtype=torch.uint8,
            device=group_bytes.device,
        )
        group_bytes = torch.cat([group_bytes, pad], dim=1)
    packed = group_bytes[:, 0::2] | (group_bytes[:, 1::2] << 4)
    return packed.cpu()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _import_speclink_kernel_backend() -> tuple[Any, Any]:
    try:
        from vllm.speclink_kernel import (
            pack_24_from_n_major_group_bytes,
            prepare_cutlass_sparse24_device_gemm,
        )
    except Exception as exc:
        raise RuntimeError(
            "SpecLink token-dense requires the repo-local "
            "vllm.speclink_kernel package"
        ) from exc
    return pack_24_from_n_major_group_bytes, prepare_cutlass_sparse24_device_gemm


def _record_cutlass_skip(
    stats: dict[str, Any] | None,
    module_name: str,
    reason: str,
) -> None:
    if stats is None:
        return
    stats.setdefault("cutlass_sparse24_skipped_modules", []).append(
        {"module": module_name, "reason": reason}
    )


def _strict_kernel_error(
    module_name: str,
    reason: str,
    stats: dict[str, Any] | None = None,
) -> None:
    _record_cutlass_skip(stats, module_name, reason)
    raise RuntimeError(
        f"SpecLink token-dense requires strict 2:4 prepack for {module_name}; "
        f"{reason}"
    )


def _cutlass_supported_weight(
    module_name: str,
    weight: torch.Tensor,
    stats: dict[str, Any] | None = None,
) -> bool:
    if not weight.is_cuda:
        _record_cutlass_skip(stats, module_name, "weight_not_cuda")
        return False
    if weight.dtype != torch.float16:
        _record_cutlass_skip(stats, module_name, f"unsupported_dtype:{weight.dtype}")
        return False
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    if in_features % 64 != 0:
        _record_cutlass_skip(
            stats,
            module_name,
            f"in_features_not_multiple_of_64:{in_features}",
        )
        return False
    if out_features % 32 != 0:
        _record_cutlass_skip(
            stats,
            module_name,
            f"out_features_not_multiple_of_32:{out_features}",
        )
        return False
    return True


def _attach_speclink_kernel_prepack(
    module: Any,
    module_name: str,
    weight: torch.Tensor,
    stats: dict[str, Any] | None = None,
    *,
    activation_scale: torch.Tensor | None = None,
) -> None:
    if getattr(module, "_speclink_selective_dense_enabled", False):
        return
    mask_bytes = getattr(module, "_speclink_24_mask_bytes", None)
    if mask_bytes is None:
        _strict_kernel_error(module_name, "missing_mask", stats)
    if not _cutlass_supported_weight(module_name, weight, stats):
        _strict_kernel_error(module_name, "unsupported_weight", stats)

    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    strategy = os.environ.get(
        "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY",
        "auto",
    ).strip()
    sparse_backend = os.environ.get(
        "SPECLINK_SPARSE24_BACKEND", "cutlass"
    ).strip().lower()
    if sparse_backend != "cutlass":
        _strict_kernel_error(
            module_name,
            f"unsupported_sparse_backend:{sparse_backend}",
            stats,
        )
    release_dense_weight_requested = _env_flag(
        "SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT"
    )
    retain_dense_policy = os.environ.get(
        "SPECLINK_SPARSE24_RETAIN_DENSE_WEIGHT", "none"
    ).strip().lower()
    leaf = _module_leaf(module_name)
    if retain_dense_policy == "none":
        retain_dense_weight = False
    elif retain_dense_policy == "qkv":
        retain_dense_weight = leaf in {"qkv_proj", "q_proj", "k_proj", "v_proj"}
    elif retain_dense_policy == "attention":
        retain_dense_weight = leaf in {
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        }
    else:
        _strict_kernel_error(
            module_name,
            f"unsupported_retain_dense_weight_policy:{retain_dense_policy}",
            stats,
        )
    release_dense_weight = release_dense_weight_requested and not retain_dense_weight
    row_scale = getattr(module, "_speclink_24_row_scale", None)
    row_scale_mode = os.environ.get(
        "SPECLINK_SPARSE24_ROW_SCALE_MODE", "cache"
    ).strip().lower()
    variance_scale_projection_policy = os.environ.get(
        "SPECLINK_SPARSE24_VARIANCE_SCALE_PROJECTION_POLICY", "all"
    ).strip().lower()
    if row_scale_mode not in {"none", "cache", "variance"}:
        _strict_kernel_error(
            module_name,
            f"unsupported_sparse_row_scale_mode:{row_scale_mode}",
            stats,
        )
    row_scale_max = float(
        os.environ.get("SPECLINK_SPARSE24_ROW_SCALE_MAX", "1.25")
    )
    if not math.isfinite(row_scale_max) or row_scale_max < 1.0:
        _strict_kernel_error(
            module_name,
            f"invalid_sparse_row_scale_max:{row_scale_max}",
            stats,
        )
    value_scale = float(os.environ.get("SPECLINK_SPARSE24_VALUE_SCALE", "1.0"))
    if leaf in {"gate_proj", "up_proj", "gate_up_proj"}:
        value_scale = float(
            os.environ.get(
                "SPECLINK_SPARSE24_GATE_UP_VALUE_SCALE",
                str(value_scale),
            )
        )
    if not math.isfinite(value_scale) or value_scale <= 0.0:
        _strict_kernel_error(
            module_name,
            f"invalid_sparse_value_scale:{value_scale}",
            stats,
        )
    if strategy not in {
        "auto",
        "full_sparse_residual",
        "full_sparse_dense_override",
        "split_dense_sparse",
        "sparse_only_decode",
    }:
        _strict_kernel_error(module_name, f"unsupported_strategy:{strategy}", stats)
    group_bytes = _expand_cached_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=in_features // 4,
    )
    bit_counts = _BIT_COUNTS.to(device=group_bytes.device)
    bad = bit_counts[(group_bytes & 0x0F).to(dtype=torch.long)].ne(2)
    if bool(bad.any().item()):
        bad_row, bad_group = bad.nonzero(as_tuple=False)[0].tolist()
        _strict_kernel_error(
            module_name,
            "mask_group_not_2to4:"
            f"row={bad_row}:group={bad_group}:"
            f"mask=0x{int(group_bytes[bad_row, bad_group].item()):x}",
            stats,
        )

    prepack_weight = weight
    reconstruction_stats: dict[str, float] | None = None
    if _env_flag("SPECLINK_SPARSE24_GROUP_RECONSTRUCTION", "0"):
        covariance = getattr(module, "_speclink_24_group_covariance", None)
        if covariance is not None:
            if strategy == "full_sparse_residual":
                _strict_kernel_error(
                    module_name,
                    "group_reconstruction_incompatible_with_exact_residual",
                    stats,
                )
            prepack_weight, reconstruction_stats = _reconstruct_grouped_24_weight(
                weight,
                group_bytes,
                covariance,
            )

    effective_row_scale = None if row_scale_mode == "none" else row_scale
    variance_scale_enabled = _token_dense_projection_enabled(
        variance_scale_projection_policy,
        leaf,
    )
    if row_scale_mode == "variance" and variance_scale_enabled:
        effective_row_scale = _variance_preserving_row_scale(
            weight,
            group_bytes,
            activation_scale,
            max_scale=row_scale_max,
        )
    if effective_row_scale is not None:
        effective_row_scale = effective_row_scale.to(
            device=weight.device,
            dtype=weight.dtype,
        )
    if value_scale != 1.0:
        if effective_row_scale is None:
            effective_row_scale = torch.full(
                (out_features,),
                value_scale,
                device=weight.device,
                dtype=weight.dtype,
            )
        else:
            effective_row_scale = effective_row_scale * value_scale

    pack_group_bytes = group_bytes
    pack_weight = prepack_weight
    pack_row_scale = effective_row_scale
    gate_up_hybrid = os.environ.get(
        "SPECLINK_SPARSE24_GATE_UP_HYBRID", "none"
    ).strip().lower()
    hybrid_sparse_first = False
    if leaf == "gate_up_proj" and gate_up_hybrid != "none":
        if gate_up_hybrid not in {"up_sparse", "gate_sparse"}:
            _strict_kernel_error(
                module_name,
                f"unsupported_gate_up_hybrid:{gate_up_hybrid}",
                stats,
            )
        if strategy != "sparse_only_decode":
            _strict_kernel_error(
                module_name,
                "gate_up_hybrid_requires_sparse_only_decode",
                stats,
            )
        if out_features % 2:
            _strict_kernel_error(
                module_name,
                f"gate_up_hybrid_requires_even_out_features:{out_features}",
                stats,
            )
        split = out_features // 2
        hybrid_sparse_first = gate_up_hybrid == "gate_sparse"
        sparse_slice = slice(0, split) if hybrid_sparse_first else slice(split, None)
        pack_group_bytes = group_bytes[sparse_slice]
        pack_weight = prepack_weight[sparse_slice]
        if effective_row_scale is not None:
            pack_row_scale = effective_row_scale[sparse_slice]
    selective_mixed_rows = bool(
        getattr(module, "_speclink_selective_mixed_rows", True)
    )
    qkv_paired_requested = (
        leaf == "qkv_proj"
        and in_features == 4096
        and out_features in {5120, 6144}
        and _env_flag("SPECLINK_SPARSE24_QKV_PAIRED_ROUTING", "0")
    )
    qkv_paired_routing = (
        qkv_paired_requested and effective_row_scale is None
    )
    # A pure-static projection still needs the complementary 2:4 pack when
    # its dense weight is released: verify rows use W24, while prefill and
    # calls outside a verify context reconstruct W exactly as W24 + R24.
    pack_residual = (selective_mixed_rows or release_dense_weight) and (
        strategy == "full_sparse_residual"
        or (strategy == "auto" and effective_row_scale is None)
        or qkv_paired_routing
    )
    if pack_residual and effective_row_scale is not None:
        _strict_kernel_error(
            module_name,
            "row_scale_residual_is_not_exact_2to4",
            stats,
        )
    if release_dense_weight and not pack_residual:
        _strict_kernel_error(
            module_name,
            "release_dense_weight_requires_exact_residual_prepack",
            stats,
        )

    inline_swiglu_mlp = (
        leaf == "gate_up_proj"
        and gate_up_hybrid == "none"
        and strategy in {"full_sparse_dense_override", "split_dense_sparse"}
        and _env_flag("SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP", "0")
        and _env_flag("SPECLINK_TOKEN_DENSE_INLINE_SWIGLU_MLP", "0")
    )
    routed_swiglu_mlp = (
        leaf == "gate_up_proj"
        and gate_up_hybrid == "none"
        and selective_mixed_rows
        and pack_residual
        and _env_flag("SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP", "0")
        and _env_flag("SPECLINK_TOKEN_DENSE_ROUTED_SWIGLU_MLP", "0")
    )
    sparse_gate_dense_down_mlp = (
        leaf == "gate_up_proj"
        and gate_up_hybrid == "none"
        and not selective_mixed_rows
        and _env_flag(
            "SPECLINK_TOKEN_DENSE_SPARSE_GATE_DENSE_DOWN", "0"
        )
    )
    interleaved_swiglu_mlp = (
        inline_swiglu_mlp
        or routed_swiglu_mlp
        or sparse_gate_dense_down_mlp
    )
    qkv_cusparselt = (
        leaf == "qkv_proj"
        and in_features == 4096
        and out_features == 6144
        and selective_mixed_rows
        and pack_residual
        and _env_flag("SPECLINK_SPARSE24_QKV_CUSPARSELT", "0")
    )
    if interleaved_swiglu_mlp and out_features % 256:
        _strict_kernel_error(
            module_name,
            f"interleaved_swiglu_requires_out_features_multiple_256:{out_features}",
            stats,
        )

    try:
        pack_from_group_bytes, prepare_device_gemm = (
            _import_speclink_kernel_backend()
        )
        packed_full = pack_from_group_bytes(
            pack_weight,
            pack_group_bytes,
            pack_row_scale,
        )
        if interleaved_swiglu_mlp:
            from vllm.speclink_kernel import (
                prepare_cutlass_sparse24_gate_up_swiglu,
            )

            full_a_values, full_a_meta_e = (
                prepare_cutlass_sparse24_gate_up_swiglu(
                    packed_full.values,
                    packed_full.meta,
                    layout=packed_full.layout,
                    K=packed_full.K,
                )
            )
        else:
            full_a_values, full_a_meta_e = prepare_device_gemm(
                packed_full.values,
                packed_full.meta,
                layout=packed_full.layout,
                K=packed_full.K,
            )
        if pack_residual:
            residual_group_bytes = (group_bytes ^ 0x0F).to(dtype=torch.uint8)
            packed_residual = pack_from_group_bytes(
                weight, residual_group_bytes, None
            )
            residual_a_values, residual_a_meta_e = prepare_device_gemm(
                packed_residual.values,
                packed_residual.meta,
                layout=packed_residual.layout,
                K=packed_residual.K,
            )
        else:
            residual_a_values = None
            residual_a_meta_e = None
        qkv_cusparselt_packed = (
            _pack_qkv_cusparselt_24(prepack_weight, group_bytes)
            if qkv_cusparselt
            else None
        )
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _strict_kernel_error(module_name, "prepack_oom", stats)
    except Exception as exc:
        _strict_kernel_error(module_name, f"prepack_failed:{exc}", stats)

    module._speclink_selective_dense_enabled = True
    module._speclink_sparse24_backend = sparse_backend
    module._speclink_sparse24_full_a_values = full_a_values
    module._speclink_sparse24_full_a_meta_e = full_a_meta_e
    module._speclink_sparse24_gate_up_interleaved = interleaved_swiglu_mlp
    module._speclink_sparse24_routed_swiglu = routed_swiglu_mlp
    module._speclink_sparse24_qkv_paired = (
        qkv_paired_routing and selective_mixed_rows
    )
    module._speclink_sparse24_sparse_gate_dense_down = (
        sparse_gate_dense_down_mlp
    )
    if residual_a_values is not None and residual_a_meta_e is not None:
        module._speclink_sparse24_residual_a_values = residual_a_values
        module._speclink_sparse24_residual_a_meta_e = residual_a_meta_e
    if qkv_cusparselt_packed is not None:
        module._speclink_sparse24_qkv_cusparselt_packed = (
            qkv_cusparselt_packed
        )
    module._speclink_sparse24_in_features = in_features
    module._speclink_sparse24_out_features = out_features
    module._speclink_gate_up_hybrid = (
        gate_up_hybrid if leaf == "gate_up_proj" else "none"
    )
    module._speclink_gate_up_hybrid_sparse_first = hybrid_sparse_first
    if leaf == "gate_up_proj" and gate_up_hybrid != "none":
        from vllm.speclink_linear import prepare_gate_up_hybrid_streams

        module._speclink_gate_up_hybrid_parallel = (
            prepare_gate_up_hybrid_streams(weight.device)
        )
    if leaf == "gate_up_proj" and (
        _env_flag("SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP", "0")
        or sparse_gate_dense_down_mlp
    ):
        from vllm.speclink_mlp import prepare_mixed_mlp_streams

        prepare_mixed_mlp_streams(weight.device)
    if strategy in {"split_dense_sparse", "full_sparse_dense_override"}:
        from vllm.speclink_linear import prepare_mixed_linear_streams

        module._speclink_parallel_split = prepare_mixed_linear_streams(
            weight.device
        )
    module._speclink_sparse24_module_name = module_name
    qkv_parallel_residual = False
    if pack_residual:
        from vllm.speclink_linear import prepare_qkv_parallel_residual_streams

        qkv_parallel_residual = prepare_qkv_parallel_residual_streams(
            module_name,
            in_features,
            out_features,
            weight.device,
        )
    module._speclink_qkv_parallel_residual = qkv_parallel_residual
    module._speclink_sparse24_linear_strategy = strategy
    module._speclink_sparse24_value_scale = value_scale
    module._speclink_sparse24_row_scale_mode = row_scale_mode
    module._speclink_sparse24_dynamic_cutlass_enabled = True
    if sparse_backend == "cutlass":
        module._speclink_sparse24_dynamic_cutlass_a_values = full_a_values
        module._speclink_sparse24_dynamic_cutlass_a_meta_e = full_a_meta_e
    module._speclink_sparse24_dynamic_cutlass_in_features = in_features
    module._speclink_sparse24_dynamic_cutlass_out_features = out_features
    module._speclink_sparse24_dynamic_cutlass_module_name = module_name
    if release_dense_weight:
        module._speclink_sparse24_dense_weight_released = True
        with torch.no_grad():
            weight.data = torch.empty(
                0,
                device=weight.device,
                dtype=weight.dtype,
            )
    if stats is not None:
        if sparse_backend == "cutlass":
            stats.setdefault(
                "cutlass_sparse24_dynamic_prepack_module_names", []
            ).append(module_name)
        stats.setdefault("speclink_kernel_prepack_module_names", []).append(
            module_name
        )
        if inline_swiglu_mlp:
            stats.setdefault(
                "speclink_kernel_inline_swiglu_mlp_module_names", []
            ).append(module_name)
        if routed_swiglu_mlp:
            stats.setdefault(
                "speclink_kernel_routed_swiglu_mlp_module_names", []
            ).append(module_name)
        if sparse_gate_dense_down_mlp:
            stats.setdefault(
                "speclink_kernel_sparse_gate_dense_down_module_names", []
            ).append(module_name)
        stats.setdefault("speclink_kernel_backend_module_names", {}).setdefault(
            sparse_backend, []
        ).append(module_name)
        stats["speclink_kernel_sparse_value_scale"] = value_scale
        stats["speclink_kernel_row_scale_mode"] = row_scale_mode
        stats["speclink_kernel_variance_scale_projection_policy"] = (
            variance_scale_projection_policy
        )
        stats["speclink_kernel_row_scale_max"] = row_scale_max
        if reconstruction_stats is not None:
            stats.setdefault(
                "speclink_kernel_group_reconstruction_module_stats", []
            ).append({"module": module_name, **reconstruction_stats})
        if effective_row_scale is not None:
            scale_float = effective_row_scale.float()
            stats.setdefault("speclink_kernel_row_scale_module_stats", []).append(
                {
                    "module": module_name,
                    "min": float(scale_float.min().item()),
                    "mean": float(scale_float.mean().item()),
                    "max": float(scale_float.max().item()),
                }
            )
        if qkv_parallel_residual:
            stats.setdefault(
                "speclink_kernel_qkv_parallel_residual_module_names", []
            ).append(module_name)
        if qkv_cusparselt_packed is not None:
            stats.setdefault(
                "speclink_kernel_qkv_cusparselt_module_names", []
            ).append(module_name)
        if pack_residual:
            stats.setdefault(
                "speclink_kernel_residual_prepack_module_names", []
            ).append(module_name)
        if release_dense_weight:
            stats.setdefault(
                "speclink_kernel_released_dense_weight_module_names", []
            ).append(module_name)
        elif release_dense_weight_requested and retain_dense_weight:
            stats.setdefault(
                "speclink_kernel_retained_dense_weight_module_names", []
            ).append(module_name)


def apply_structured_24_from_env(
    model: Any,
    *,
    logger: Any | None = None,
    context: str = "target_model",
) -> dict[str, Any] | None:
    """Apply the requested 2:4 mask to the already-loaded target model.

    Returns the stats dict when enabled, otherwise None. The caller must invoke
    this before loading the drafter/speculator so only the TLM is modified.
    """

    if not _env_flag("SPECLINK_STRUCTURED_24_ENABLE"):
        return None

    model_label = os.environ.get("SPECLINK_STRUCTURED_24_MODEL_LABEL", "").strip()
    cache_root_raw = os.environ.get("SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT", "")
    policy = os.environ.get("SPECLINK_STRUCTURED_24_POLICY", "all_sparse").strip()
    layer_index_raw = os.environ.get("SPECLINK_STRUCTURED_24_LAYER_INDEX", "").strip()
    keep_n = int(os.environ.get("SPECLINK_STRUCTURED_24_KEEP_N", "0") or "0")
    stats_path_raw = os.environ.get("SPECLINK_STRUCTURED_24_STATS_PATH", "").strip()
    mask_cache_raw = os.environ.get("SPECLINK_STRUCTURED_24_MASK_CACHE", "").strip()
    group_reconstruction = _env_flag(
        "SPECLINK_SPARSE24_GROUP_RECONSTRUCTION", "0"
    )
    group_covariance_cache_raw = os.environ.get(
        "SPECLINK_SPARSE24_GROUP_COVARIANCE_CACHE", ""
    ).strip()
    cache_strict = _env_flag("SPECLINK_STRUCTURED_24_CACHE_STRICT", "1")
    token_dense = _env_flag("SPECLINK_TOKEN_DENSE_ENABLE", "0")
    token_dense_linear_strategy = os.environ.get(
        "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY",
        "auto",
    ).strip()
    token_dense_dense_selection = os.environ.get(
        "SPECLINK_TOKEN_DENSE_DENSE_SELECTION",
        "highest",
    ).strip()
    token_dense_pure_batch_routes = (
        token_dense_linear_strategy != "sparse_only_decode"
        and token_dense_dense_selection
        in {"batch_adaptive", "batch_alternating", "batch_confidence"}
    )
    token_dense_mlp_strategy = os.environ.get(
        "SPECLINK_TOKEN_DENSE_MLP_STRATEGY", "auto"
    ).strip()
    token_dense_projection_policy = os.environ.get(
        "SPECLINK_TOKEN_DENSE_PROJECTION_POLICY", "all"
    ).strip()
    token_dense_mixed_projection_policy = os.environ.get(
        "SPECLINK_TOKEN_DENSE_MIXED_PROJECTION_POLICY", "all"
    ).strip()
    token_dense_mixed_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_MIXED_LAYERS", ""
    ).strip()
    token_dense_mlp_static_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_MLP_STATIC_LAYERS", ""
    ).strip()
    token_dense_o_sparse_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_O_SPARSE_LAYERS", ""
    ).strip()
    token_dense_gate_up_dense_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_GATE_UP_DENSE_LAYERS", ""
    ).strip()
    token_dense_down_dense_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_DOWN_DENSE_LAYERS", ""
    ).strip()
    token_dense_attention_dense_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_ATTENTION_DENSE_LAYERS", ""
    ).strip()
    token_dense_dense_layers_raw = os.environ.get(
        "SPECLINK_TOKEN_DENSE_DENSE_LAYERS", ""
    ).strip()
    try:
        token_dense_gate_up_dense_layers = _parse_layer_indices(
            token_dense_gate_up_dense_layers_raw
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "invalid SPECLINK_TOKEN_DENSE_GATE_UP_DENSE_LAYERS="
            f"{token_dense_gate_up_dense_layers_raw!r}: {exc}"
        ) from exc
    try:
        token_dense_down_dense_layers = _parse_layer_indices(
            token_dense_down_dense_layers_raw
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "invalid SPECLINK_TOKEN_DENSE_DOWN_DENSE_LAYERS="
            f"{token_dense_down_dense_layers_raw!r}: {exc}"
        ) from exc
    try:
        token_dense_attention_dense_layers = _parse_layer_indices(
            token_dense_attention_dense_layers_raw
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "invalid SPECLINK_TOKEN_DENSE_ATTENTION_DENSE_LAYERS="
            f"{token_dense_attention_dense_layers_raw!r}: {exc}"
        ) from exc
    try:
        token_dense_dense_layers = _parse_layer_indices(
            token_dense_dense_layers_raw
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "invalid SPECLINK_TOKEN_DENSE_DENSE_LAYERS="
            f"{token_dense_dense_layers_raw!r}: {exc}"
        ) from exc
    token_dense_all_layers_mixed = token_dense_mixed_layers_raw in {"", "all"}
    if token_dense_mixed_layers_raw == "none":
        token_dense_mixed_layers: set[int] = set()
    elif token_dense_all_layers_mixed:
        token_dense_mixed_layers = set()
    else:
        try:
            token_dense_mixed_layers = _parse_layer_indices(
                token_dense_mixed_layers_raw
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "invalid SPECLINK_TOKEN_DENSE_MIXED_LAYERS="
                f"{token_dense_mixed_layers_raw!r}: {exc}"
            ) from exc
    try:
        token_dense_mlp_static_layers = _parse_layer_indices(
            token_dense_mlp_static_layers_raw
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "invalid SPECLINK_TOKEN_DENSE_MLP_STATIC_LAYERS="
            f"{token_dense_mlp_static_layers_raw!r}: {exc}"
        ) from exc
    token_dense_all_o_layers_sparse = token_dense_o_sparse_layers_raw in {
        "",
        "all",
    }
    if token_dense_o_sparse_layers_raw == "none":
        token_dense_o_sparse_layers: set[int] = set()
    elif token_dense_all_o_layers_sparse:
        token_dense_o_sparse_layers = set()
    else:
        try:
            token_dense_o_sparse_layers = _parse_layer_indices(
                token_dense_o_sparse_layers_raw
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "invalid SPECLINK_TOKEN_DENSE_O_SPARSE_LAYERS="
                f"{token_dense_o_sparse_layers_raw!r}: {exc}"
            ) from exc
    token_dense_row_scale_mode = os.environ.get(
        "SPECLINK_SPARSE24_ROW_SCALE_MODE", "cache"
    ).strip().lower()
    token_dense_sparse_accumulator = os.environ.get(
        "SPECLINK_SPARSE24_ACCUMULATOR", "fp32"
    ).strip().lower()
    token_dense_sparse_backend = os.environ.get(
        "SPECLINK_SPARSE24_BACKEND", "cutlass"
    ).strip().lower()
    token_dense_gate_up_value_scale = float(
        os.environ.get("SPECLINK_SPARSE24_GATE_UP_VALUE_SCALE", "1.0")
    )
    token_dense_gate_up_hybrid = os.environ.get(
        "SPECLINK_SPARSE24_GATE_UP_HYBRID", "none"
    ).strip().lower()
    if token_dense_gate_up_hybrid not in {"none", "up_sparse", "gate_sparse"}:
        raise RuntimeError(
            "SPECLINK_SPARSE24_GATE_UP_HYBRID must be none, up_sparse, "
            "or gate_sparse"
        )
    if (
        token_dense_gate_up_hybrid != "none"
        and token_dense_linear_strategy != "sparse_only_decode"
    ):
        raise RuntimeError(
            "gate/up hybrid currently requires sparse_only_decode"
        )
    if (
        not math.isfinite(token_dense_gate_up_value_scale)
        or token_dense_gate_up_value_scale <= 0.0
    ):
        raise RuntimeError(
            "SPECLINK_SPARSE24_GATE_UP_VALUE_SCALE must be finite and positive"
        )
    dynamic_cutlass_env_requested = _env_flag(
        "SPECLINK_STRUCTURED_24_DYNAMIC_CUTLASS_BACKEND", "0"
    )
    dynamic_cutlass_requested = token_dense or dynamic_cutlass_env_requested
    dynamic_cutlass_active = token_dense

    if not model_label:
        raise RuntimeError("SPECLINK_STRUCTURED_24_MODEL_LABEL is required")
    if not cache_root_raw:
        raise RuntimeError("SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT is required")
    if policy not in {
        "dense",
        "single_layer",
        "all_sparse",
        "keep_first",
        "keep_last",
        "keep_first_last",
    }:
        raise RuntimeError(f"unsupported SPECLINK_STRUCTURED_24_POLICY={policy}")
    if token_dense:
        _token_dense_projection_enabled(token_dense_projection_policy, "qkv_proj")
        _token_dense_projection_enabled(
            token_dense_mixed_projection_policy,
            "qkv_proj",
        )
        if token_dense_row_scale_mode not in {"none", "cache", "variance"}:
            raise RuntimeError(
                "SPECLINK_SPARSE24_ROW_SCALE_MODE must be none, cache, or "
                "variance"
            )
        if token_dense_sparse_accumulator not in {
            "fp32",
            "fp16",
            "fp16_gate",
            "fp16_gate_down",
            "fp16_qkv_gate",
        }:
            raise RuntimeError(
                "SPECLINK_SPARSE24_ACCUMULATOR must be fp32, fp16, "
                "fp16_gate, fp16_gate_down, or fp16_qkv_gate"
            )
        if token_dense_sparse_backend != "cutlass":
            raise RuntimeError(
                "SPECLINK_SPARSE24_BACKEND must be cutlass"
            )

    modules = _iter_target_modules(model)
    layers = _selected_layers(modules)
    unknown_gate_up_dense_layers = token_dense_gate_up_dense_layers.difference(
        layers
    )
    unknown_down_dense_layers = token_dense_down_dense_layers.difference(layers)
    unknown_attention_dense_layers = token_dense_attention_dense_layers.difference(
        layers
    )
    unknown_dense_layers = token_dense_dense_layers.difference(layers)
    unknown_mixed_layers = token_dense_mixed_layers.difference(layers)
    unknown_mlp_static_layers = token_dense_mlp_static_layers.difference(
        layers
    )
    unknown_o_sparse_layers = token_dense_o_sparse_layers.difference(layers)
    if token_dense and unknown_gate_up_dense_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_GATE_UP_DENSE_LAYERS contains layers not "
            f"present in the target model: {sorted(unknown_gate_up_dense_layers)}"
        )
    if token_dense and unknown_dense_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_LAYERS contains layers not present "
            f"in the target model: {sorted(unknown_dense_layers)}"
        )
    if token_dense and unknown_down_dense_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DOWN_DENSE_LAYERS contains layers not "
            f"present in the target model: {sorted(unknown_down_dense_layers)}"
        )
    if token_dense and unknown_attention_dense_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_ATTENTION_DENSE_LAYERS contains layers not "
            f"present in the target model: {sorted(unknown_attention_dense_layers)}"
        )
    if token_dense and unknown_mixed_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_MIXED_LAYERS contains layers not present "
            f"in the target model: {sorted(unknown_mixed_layers)}"
        )
    if token_dense and unknown_mlp_static_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_MLP_STATIC_LAYERS contains layers not "
            f"present in the target model: {sorted(unknown_mlp_static_layers)}"
        )
    if token_dense and unknown_o_sparse_layers:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_O_SPARSE_LAYERS contains layers not present "
            f"in the target model: {sorted(unknown_o_sparse_layers)}"
        )
    activation_scales = _load_activation_scales(Path(cache_root_raw), model_label)
    mask_cache = _load_mask_cache(mask_cache_raw) if mask_cache_raw else None
    if group_reconstruction and not group_covariance_cache_raw:
        raise RuntimeError(
            "SPECLINK_SPARSE24_GROUP_COVARIANCE_CACHE is required when grouped "
            "weight reconstruction is enabled"
        )
    group_covariances = (
        _load_group_covariance_cache(group_covariance_cache_raw)
        if group_reconstruction
        else {}
    )
    single_layer = int(layer_index_raw) if layer_index_raw else None
    if policy == "single_layer" and single_layer is None:
        raise RuntimeError("SPECLINK_STRUCTURED_24_LAYER_INDEX is required for single_layer")
    if dynamic_cutlass_active:
        _import_speclink_kernel_backend()

    stats: dict[str, Any] = {
        "enabled": True,
        "context": context,
        "model_label": model_label,
        "policy": policy,
        "layer_index": single_layer,
        "keep_n": keep_n,
        "calibration_cache_root": str(Path(cache_root_raw).resolve()),
        "mask_cache": str(Path(mask_cache_raw).resolve()) if mask_cache_raw else "",
        "mask_cache_method": (
            (mask_cache.get("metadata", {}) or {}).get("method", "") if mask_cache else ""
        ),
        "token_dense_enabled": token_dense,
        "cutlass_sparse24_dynamic_backend_requested": dynamic_cutlass_requested,
        "cutlass_sparse24_dynamic_backend_enabled": False,
        "cutlass_sparse24_dynamic_backend_disabled_reason": (
            "not_token_dense"
            if dynamic_cutlass_env_requested and not token_dense
            else ""
        ),
        "cutlass_sparse24_dynamic_prepack_module_count": 0,
        "cutlass_sparse24_dynamic_prepack_module_names": [],
        "speclink_kernel_backend_enabled": False,
        "speclink_kernel_strict": token_dense,
        "speclink_kernel_linear_strategy": token_dense_linear_strategy,
        "speclink_kernel_dense_selection": token_dense_dense_selection,
        "speclink_kernel_pure_batch_routes": token_dense_pure_batch_routes,
        "speclink_kernel_mlp_strategy": token_dense_mlp_strategy,
        "speclink_kernel_projection_policy": token_dense_projection_policy,
        "speclink_kernel_mixed_projection_policy": (
            token_dense_mixed_projection_policy
        ),
        "speclink_kernel_all_layers_mixed": token_dense_all_layers_mixed,
        "speclink_kernel_mixed_layers": sorted(token_dense_mixed_layers),
        "speclink_kernel_mlp_static_layers": sorted(
            token_dense_mlp_static_layers
        ),
        "speclink_kernel_all_o_layers_sparse": (
            token_dense_all_o_layers_sparse
        ),
        "speclink_kernel_o_sparse_layers": sorted(
            token_dense_o_sparse_layers
        ),
        "speclink_kernel_mixed_module_names": [],
        "speclink_kernel_sparse_only_module_names": [],
        "speclink_kernel_gate_up_dense_layers": sorted(
            token_dense_gate_up_dense_layers
        ),
        "speclink_kernel_down_dense_layers": sorted(
            token_dense_down_dense_layers
        ),
        "speclink_kernel_attention_dense_layers": sorted(
            token_dense_attention_dense_layers
        ),
        "speclink_kernel_dense_layers": sorted(token_dense_dense_layers),
        "speclink_kernel_row_scale_mode": token_dense_row_scale_mode,
        "speclink_kernel_sparse_accumulator": token_dense_sparse_accumulator,
        "speclink_kernel_sparse_backend": token_dense_sparse_backend,
        "speclink_kernel_gate_up_value_scale": (
            token_dense_gate_up_value_scale
        ),
        "speclink_kernel_gate_up_hybrid": token_dense_gate_up_hybrid,
        "speclink_kernel_group_reconstruction": group_reconstruction,
        "speclink_kernel_group_covariance_cache": (
            str(Path(group_covariance_cache_raw).resolve())
            if group_covariance_cache_raw
            else ""
        ),
        "speclink_kernel_prepack_module_count": 0,
        "speclink_kernel_prepack_module_names": [],
        "speclink_kernel_inline_swiglu_mlp_module_count": 0,
        "speclink_kernel_inline_swiglu_mlp_module_names": [],
        "speclink_kernel_routed_swiglu_mlp_module_count": 0,
        "speclink_kernel_routed_swiglu_mlp_module_names": [],
        "speclink_kernel_sparse_gate_dense_down_module_count": 0,
        "speclink_kernel_sparse_gate_dense_down_module_names": [],
        "speclink_kernel_residual_prepack_module_names": [],
        "speclink_kernel_qkv_cusparselt_module_names": [],
        "speclink_kernel_release_dense_weight_requested": _env_flag(
            "SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT"
        ),
        "speclink_kernel_retain_dense_weight_policy": os.environ.get(
            "SPECLINK_SPARSE24_RETAIN_DENSE_WEIGHT", "none"
        ).strip().lower(),
        "speclink_kernel_released_dense_weight_module_names": [],
        "speclink_kernel_retained_dense_weight_module_names": [],
        "cutlass_sparse24_skipped_modules": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "module_count_seen": len(modules),
        "layers_seen": layers,
        "total_masked_weight_count": 0,
        "zeroed_weight_count": 0,
        "scope_target_weight_count": 0,
        "dense_keep_weight_count": 0,
        "actual_sparsity": 0.0,
        "effective_sparse_fraction": 0.0,
        "masked_module_names": [],
        "dense_keep_module_names": [],
        "missing_activation_scale_modules": [],
        "missing_cached_mask_modules": [],
        "per_module": [],
    }

    for name, _module, weight in modules:
        layer = _layer_index(name)
        leaf = _module_leaf(name)
        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])
        usable_in = (in_features // 4) * 4
        total = out_features * usable_in
        stats["scope_target_weight_count"] += total

        should_mask = policy != "dense"
        keep_dense_method = "dense_keep"
        if policy == "single_layer":
            should_mask = layer == single_layer
        elif policy in {"keep_first", "keep_last", "keep_first_last"}:
            should_mask = not _should_keep_dense(
                policy=policy,
                layer=layer,
                layers=layers,
                keep_n=keep_n,
            )
        if token_dense and not _token_dense_projection_enabled(
            token_dense_projection_policy, leaf
        ):
            should_mask = False
            keep_dense_method = (
                "token_dense_projection_"
                f"{token_dense_projection_policy}_dense"
            )
        if (
            token_dense
            and leaf == "o_proj"
            and not token_dense_all_o_layers_sparse
            and layer not in token_dense_o_sparse_layers
        ):
            should_mask = False
            keep_dense_method = "token_dense_o_layer_dense"
        if (
            token_dense
            and layer in token_dense_gate_up_dense_layers
            and leaf in {"gate_proj", "up_proj", "gate_up_proj"}
        ):
            should_mask = False
            keep_dense_method = "token_dense_gate_up_layer_dense"
        if (
            token_dense
            and layer in token_dense_down_dense_layers
            and leaf == "down_proj"
        ):
            should_mask = False
            keep_dense_method = "token_dense_down_layer_dense"
        if (
            token_dense
            and layer in token_dense_attention_dense_layers
            and leaf in {"q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj"}
        ):
            should_mask = False
            keep_dense_method = "token_dense_attention_layer_dense"
        if token_dense and layer in token_dense_dense_layers:
            should_mask = False
            keep_dense_method = "token_dense_explicit_layer_dense"
        if (
            token_dense
            and leaf == "down_proj"
            and (
                token_dense_mlp_strategy == "gate_only"
                or (
                    token_dense_mlp_strategy == "auto"
                    and token_dense_linear_strategy != "sparse_only_decode"
                )
            )
        ):
            should_mask = False
            keep_dense_method = "token_dense_mlp_gate_only_down_dense"
        if not should_mask:
            if token_dense:
                _module._speclink_selective_dense_bypass = True
            stats["dense_keep_weight_count"] += total
            stats["dense_keep_module_names"].append(name)
            stats["per_module"].append(
                {
                    "module": name,
                    "leaf": leaf,
                    "layer": layer,
                    "shape": [out_features, in_features],
                    "masked_weight_count": 0,
                    "zeroed_weight_count": 0,
                    "kept_dense": True,
                    "mask_method": keep_dense_method,
                }
            )
            continue

        method = "activation_aware"
        dynamic_cutlass_prepacked = False
        activation_scale = _scale_for_module(name, activation_scales)
        if activation_scale is None:
            stats["missing_activation_scale_modules"].append(name)
        if mask_cache is not None:
            mask_bytes, row_scale = _cache_values_for_module(name, mask_cache)
            if mask_bytes is None:
                stats["missing_cached_mask_modules"].append(name)
                if cache_strict:
                    raise RuntimeError(f"missing cached 2:4 mask for {name}")
                if token_dense:
                    masked_total, zeroed, method = _attach_computed_mask_24(
                        _module,
                        weight,
                        activation_scale,
                    )
                else:
                    masked_total, zeroed, method = _mask_weight_24(
                        weight,
                        activation_scale,
                    )
            else:
                if token_dense:
                    masked_total, zeroed, method = _attach_cached_mask_24(
                        _module,
                        weight,
                        mask_bytes,
                        row_scale,
                    )
                else:
                    masked_total, zeroed, method = _apply_cached_mask_24(
                        weight,
                        mask_bytes,
                        row_scale,
                    )
        else:
            if token_dense:
                masked_total, zeroed, method = _attach_computed_mask_24(
                    _module,
                    weight,
                    activation_scale,
                )
            else:
                masked_total, zeroed, method = _mask_weight_24(weight, activation_scale)
        if token_dense and group_reconstruction:
            covariance = _group_covariance_for_module(name, group_covariances)
            if covariance is not None:
                _module._speclink_24_group_covariance = covariance
        if dynamic_cutlass_active:
            mixed_rows = (
                token_dense_linear_strategy != "sparse_only_decode"
                and (
                    token_dense_all_layers_mixed
                    or layer in token_dense_mixed_layers
                )
                and _token_dense_projection_enabled(
                    token_dense_mixed_projection_policy,
                    leaf,
                )
                and not (
                    layer in token_dense_mlp_static_layers
                    and leaf
                    in {"gate_proj", "up_proj", "gate_up_proj", "down_proj"}
                )
            )
            _module._speclink_selective_mixed_rows = mixed_rows
            _attach_speclink_kernel_prepack(
                _module,
                name,
                weight,
                stats,
                activation_scale=activation_scale,
            )
            dynamic_cutlass_prepacked = True
            method = f"{method}_speclink_kernel"
            stats[
                "speclink_kernel_mixed_module_names"
                if mixed_rows
                else "speclink_kernel_sparse_only_module_names"
            ].append(name)
        stats["total_masked_weight_count"] += masked_total
        stats["zeroed_weight_count"] += zeroed
        stats["masked_module_names"].append(name)
        stats["per_module"].append(
            {
                "module": name,
                "leaf": leaf,
                "layer": layer,
                "shape": [out_features, in_features],
                "masked_weight_count": masked_total,
                "zeroed_weight_count": zeroed,
                "actual_sparsity": zeroed / masked_total if masked_total else 0.0,
                "kept_dense": False,
                "mask_method": method,
                "cutlass_sparse24_dynamic_backend": dynamic_cutlass_prepacked,
                "mixed_rows": (
                    bool(getattr(_module, "_speclink_selective_mixed_rows", False))
                    if token_dense
                    else False
                ),
            }
        )

    masked_total = int(stats["total_masked_weight_count"])
    scope_total = int(stats["scope_target_weight_count"])
    stats["actual_sparsity"] = (
        int(stats["zeroed_weight_count"]) / masked_total if masked_total else 0.0
    )
    stats["effective_sparse_fraction"] = (
        int(stats["zeroed_weight_count"]) / scope_total if scope_total else 0.0
    )
    stats["cutlass_sparse24_dynamic_prepack_module_count"] = len(
        stats["cutlass_sparse24_dynamic_prepack_module_names"]
    )
    stats["cutlass_sparse24_dynamic_backend_enabled"] = (
        stats["cutlass_sparse24_dynamic_prepack_module_count"] > 0
    )
    stats["speclink_kernel_prepack_module_count"] = len(
        stats["speclink_kernel_prepack_module_names"]
    )
    stats["speclink_kernel_inline_swiglu_mlp_module_count"] = len(
        stats["speclink_kernel_inline_swiglu_mlp_module_names"]
    )
    stats["speclink_kernel_routed_swiglu_mlp_module_count"] = len(
        stats["speclink_kernel_routed_swiglu_mlp_module_names"]
    )
    stats["speclink_kernel_sparse_gate_dense_down_module_count"] = len(
        stats["speclink_kernel_sparse_gate_dense_down_module_names"]
    )
    stats["speclink_kernel_residual_prepack_module_count"] = len(
        stats["speclink_kernel_residual_prepack_module_names"]
    )
    stats["speclink_kernel_qkv_cusparselt_module_count"] = len(
        stats["speclink_kernel_qkv_cusparselt_module_names"]
    )
    stats["speclink_kernel_backend_enabled"] = (
        stats["speclink_kernel_prepack_module_count"] > 0
    )
    stats["speclink_kernel_released_dense_weight_module_count"] = len(
        stats["speclink_kernel_released_dense_weight_module_names"]
    )
    stats["speclink_kernel_retained_dense_weight_module_count"] = len(
        stats["speclink_kernel_retained_dense_weight_module_names"]
    )
    stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Token-dense mode computes masks from dense GPU weights but does not keep
    # those temporary scoring tensors. Release the allocator cache before vLLM
    # profiles KV-cache capacity.
    if token_dense and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if stats_path_raw:
        _write_json(Path(stats_path_raw), stats)
    if logger is not None:
        logger.info(
            "Applied SpecLink TLM-only 2:4 mask: model=%s policy=%s "
            "token_dense=%s masked_modules=%d effective_sparse_fraction=%.4f",
            model_label,
            policy,
            token_dense,
            len(stats["masked_module_names"]),
            stats["effective_sparse_fraction"],
        )
    return stats
