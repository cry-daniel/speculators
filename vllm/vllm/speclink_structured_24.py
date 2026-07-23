# SPDX-License-Identifier: Apache-2.0
"""Env-gated TLM-only activation-aware 2:4 masking for SpecLink studies.

This module is intentionally independent from the eval scripts so vLLM can
apply masks at model-load time without importing experiment-only code.
"""

from __future__ import annotations

import json
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
    cache_strict = _env_flag("SPECLINK_STRUCTURED_24_CACHE_STRICT", "1")
    token_dense = _env_flag("SPECLINK_TOKEN_DENSE_ENABLE", "0")
    token_dense_backend = os.environ.get(
        "SPECLINK_TOKEN_DENSE_BACKEND", "legacy_dense_first"
    ).strip()
    if token_dense_backend not in {
        "legacy_dense_first",
        "residual_complement_splitk2",
    }:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_BACKEND must be legacy_dense_first or "
            "residual_complement_splitk2"
        )
    if token_dense_backend == "residual_complement_splitk2" and not token_dense:
        raise RuntimeError(
            "residual_complement_splitk2 requires SPECLINK_TOKEN_DENSE_ENABLE=1"
        )

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

    modules = _iter_target_modules(model)
    layers = _selected_layers(modules)
    # Do not keep a second Python reference to every dense Parameter while
    # converting modules one by one.  residual-complement replaces
    # ``module.weight`` with its one-weight runtime; retaining the original
    # tensors in this list would keep the entire dense checkpoint alive and
    # double peak GPU memory until this function returns.
    module_entries = [(name, module) for name, module, _weight in modules]
    del modules
    activation_scales = _load_activation_scales(Path(cache_root_raw), model_label)
    mask_cache = _load_mask_cache(mask_cache_raw) if mask_cache_raw else None
    single_layer = int(layer_index_raw) if layer_index_raw else None
    if policy == "single_layer" and single_layer is None:
        raise RuntimeError("SPECLINK_STRUCTURED_24_LAYER_INDEX is required for single_layer")

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
        "token_dense_backend": token_dense_backend,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "module_count_seen": len(module_entries),
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
        "residual_complement_persistent_bytes": 0,
        "released_dense_weight_bytes": 0,
    }

    for name, _module in module_entries:
        weight = getattr(_module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(f"target linear {name} lost its weight before conversion")
        layer = _layer_index(name)
        leaf = _module_leaf(name)
        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])
        usable_in = (in_features // 4) * 4
        total = out_features * usable_in
        stats["scope_target_weight_count"] += total

        should_mask = policy != "dense"
        if policy == "single_layer":
            should_mask = layer == single_layer
        elif policy in {"keep_first", "keep_last", "keep_first_last"}:
            should_mask = not _should_keep_dense(
                policy=policy,
                layer=layer,
                layers=layers,
                keep_n=keep_n,
            )

        if not should_mask:
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
                    "mask_method": "dense_keep",
                }
            )
            continue

        method = "activation_aware"
        if mask_cache is not None:
            mask_bytes, row_scale = _cache_values_for_module(name, mask_cache)
            if mask_bytes is None:
                stats["missing_cached_mask_modules"].append(name)
                if cache_strict:
                    raise RuntimeError(f"missing cached 2:4 mask for {name}")
                activation_scale = _scale_for_module(name, activation_scales)
                if activation_scale is None:
                    stats["missing_activation_scale_modules"].append(name)
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
            activation_scale = _scale_for_module(name, activation_scales)
            if activation_scale is None:
                stats["missing_activation_scale_modules"].append(name)
            if token_dense:
                masked_total, zeroed, method = _attach_computed_mask_24(
                    _module,
                    weight,
                    activation_scale,
                )
            else:
                masked_total, zeroed, method = _mask_weight_24(weight, activation_scale)
        residual_runtime: dict[str, Any] | None = None
        if token_dense_backend == "residual_complement_splitk2":
            from vllm.speclink_token_dense import prepare_residual_complement_module

            residual_runtime = prepare_residual_complement_module(_module)
            stats["residual_complement_persistent_bytes"] += int(
                residual_runtime["persistent_bytes"]
            )
            stats["released_dense_weight_bytes"] += int(
                residual_runtime["released_dense_bytes"]
            )
            # Each large temporary sparse reconstruction is setup-only.  Give
            # the next module the released allocator blocks immediately.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                "residual_complement_runtime": residual_runtime,
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
