#!/usr/bin/env python3
"""Generate offline 2:4 masks, save pruned HF models, and run lm-eval.

This runner keeps the implementation intentionally local to the SpecLink
evaluation stack. It extracts the mask-selection ideas needed for the current
experiments without importing full external training frameworks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
SPECLINK_ROOT = SPECULATORS_ROOT.parent
RESULTS_ROOT = EVAL_ROOT / "results"
TEMP_ROOT = EVAL_ROOT / "temp"
MASKS_ROOT = EVAL_ROOT / "data" / "c4_calibration" / "offline_24_masks"
DEFAULT_MASK_ROOT = MASKS_ROOT / "c4_512_seed42_bf16_max512"

sys.path.insert(0, str(SCRIPT_DIR))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    DEFAULT_C4_CALIBRATION_PROMPTS,
    LAYER_SENSITIVITY_DEFAULT_MODELS,
    QUALITY_MASK_TARGETS,
    dtype_from_arg,
    ensure_quality_dependencies,
    ensure_tokenizer,
    load_activation_cache,
    load_calibration_prompt_file,
    parse_csv_list,
    parse_model_id_overrides,
    write_json,
)
from run_structured_24_spec_quality import EAGLE3_SPECULATORS  # noqa: E402

PRUNING_METHODS = ("wanda", "proxsparse", "maskllm")
METHODS = (*PRUNING_METHODS, "original")
ORIGINAL_METHOD = "original"
TARGET_MODULES = QUALITY_MASK_TARGETS["all"]
MASK_BITS = None
OPTION_GROUP_BYTES = None


@dataclass(frozen=True)
class RunPaths:
    mask_root: Path
    pruned_model_root: Path
    output_root: Path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def require_torch() -> Any:
    ensure_quality_dependencies()
    import torch

    return torch


def require_transformers() -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


def init_mask_constants(device: Any | None = None) -> tuple[Any, Any]:
    global MASK_BITS, OPTION_GROUP_BYTES
    torch = require_torch()
    if MASK_BITS is None:
        MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)
    if OPTION_GROUP_BYTES is None:
        OPTION_GROUP_BYTES = torch.tensor([3, 5, 9, 6, 10, 12], dtype=torch.uint8)
    bits = MASK_BITS
    options = OPTION_GROUP_BYTES
    if device is not None:
        bits = bits.to(device=device)
        options = options.to(device=device)
    return bits, options


def set_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_paths(args: argparse.Namespace) -> RunPaths:
    stamp = getattr(args, "_stamp", None) or timestamp()
    args._stamp = stamp
    output_root = args.output_root
    if output_root is None:
        output_root = RESULTS_ROOT / f"offline_24_pruning_gsm8k64_{stamp}"
    pruned_model_root = args.pruned_model_root
    if pruned_model_root is None:
        pruned_model_root = TEMP_ROOT / f"offline_24_pruned_models_{stamp}"
    return RunPaths(
        mask_root=(args.mask_root or DEFAULT_MASK_ROOT).resolve(),
        pruned_model_root=pruned_model_root.resolve(),
        output_root=output_root.resolve(),
    )


def resolve_models(args: argparse.Namespace) -> dict[str, str]:
    model_ids = dict(LAYER_SENSITIVITY_DEFAULT_MODELS)
    model_ids.update(parse_model_id_overrides(args.model_id))
    selected = parse_csv_list(args.models)
    missing = [label for label in selected if label not in model_ids]
    if missing:
        raise ValueError(f"unknown model labels: {', '.join(missing)}")
    return {label: model_ids[label] for label in selected}


def resolve_methods(args: argparse.Namespace, *, pruning_only: bool = False) -> list[str]:
    methods = parse_csv_list(args.methods)
    unsupported = [method for method in methods if method not in METHODS]
    if unsupported:
        raise ValueError(f"unsupported methods: {', '.join(unsupported)}")
    if pruning_only:
        methods = [method for method in methods if method != ORIGINAL_METHOD]
    return methods


def module_leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def module_is_skipped(name: str) -> bool:
    lowered = name.lower()
    return (
        name == "lm_head"
        or name.endswith(".lm_head")
        or "embed_tokens" in lowered
        or "embedding" in lowered
        or ".wte" in lowered
    )


def iter_target_linear_modules(model: Any) -> list[tuple[str, Any]]:
    torch = require_torch()
    from torch import nn

    out = []
    targets = set(TARGET_MODULES)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_is_skipped(name) or module_leaf(name) not in targets:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if (int(weight.shape[1]) // 4) * 4 <= 0:
            continue
        out.append((name, module))
    return out


def load_model_and_tokenizer(
    model_id: str,
    *,
    dtype: Any,
    device: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, Any]:
    torch = require_torch()
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    ensure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        model.to(device)
    model.eval()
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def activation_scale_for(
    module_name: str,
    activation_scales: dict[str, Any],
    *,
    usable_in: int,
    device: Any,
    dtype: Any,
) -> Any | None:
    torch = require_torch()
    scale = activation_scales.get(module_name)
    if scale is None:
        return None
    if not isinstance(scale, torch.Tensor) or scale.numel() < usable_in:
        return None
    return scale[:usable_in].to(device=device, dtype=dtype)


def score_view(weight: Any, activation_scale: Any | None) -> tuple[Any, int, int, int]:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    view = weight.detach()[:, :usable_in].float().view(out_features, groups, 4)
    score = view.abs()
    if activation_scale is not None:
        score = score * activation_scale.float().view(1, groups, 4)
    return score, out_features, usable_in, groups


def keep_to_group_bytes(keep: Any) -> Any:
    bits, _ = init_mask_constants(keep.device)
    return (keep.to(dtype=bits.dtype) * bits.view(1, 1, 4)).sum(dim=-1).to(dtype=bits.dtype)


def group_bytes_to_keep(group_bytes: Any, *, device: Any) -> Any:
    bits, _ = init_mask_constants(device)
    return (group_bytes.to(device=device, dtype=bits.dtype).unsqueeze(-1) & bits.view(1, 1, 4)).ne(0)


def pack_group_bytes(group_bytes: Any) -> Any:
    torch = require_torch()
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


def unpack_group_bytes(mask_bytes: Any, *, out_features: int, groups: int) -> Any:
    torch = require_torch()
    mask_bytes = mask_bytes.to(dtype=torch.uint8)
    if tuple(mask_bytes.shape) == (out_features, groups):
        return mask_bytes
    packed_expected = (out_features, (groups + 1) // 2)
    if tuple(mask_bytes.shape) != packed_expected:
        raise RuntimeError(
            f"mask shape {tuple(mask_bytes.shape)} does not match "
            f"{(out_features, groups)} or packed {packed_expected}"
        )
    unpacked = torch.empty(
        (out_features, packed_expected[1] * 2),
        dtype=torch.uint8,
        device=mask_bytes.device,
    )
    unpacked[:, 0::2] = mask_bytes & 0x0F
    unpacked[:, 1::2] = (mask_bytes >> 4) & 0x0F
    return unpacked[:, :groups]


def top2_mask_from_score(score: Any) -> Any:
    torch = require_torch()
    keep_idx = score.topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return keep


def top2_group_bytes_from_score(score: Any, chunk_size: int) -> Any:
    torch = require_torch()
    bits, _ = init_mask_constants(score.device)
    out_features = int(score.shape[0])
    groups = int(score.shape[1])
    flat = score.reshape(-1, 4)
    out = torch.empty(flat.shape[0], dtype=torch.uint8, device=score.device)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        keep_idx = chunk.topk(k=2, dim=-1, largest=True, sorted=False).indices
        keep = torch.zeros_like(chunk, dtype=torch.bool)
        keep.scatter_(-1, keep_idx, True)
        out[start : start + chunk_size] = (
            keep.to(dtype=torch.uint8) * bits.view(1, 4)
        ).sum(dim=-1)
    return out.view(out_features, groups)


def mask_stats_template(method: str, model_label: str) -> dict[str, Any]:
    return {
        "method": method,
        "model_label": model_label,
        "target_modules": list(TARGET_MODULES),
        "module_count": 0,
        "total_masked_weight_count": 0,
        "zeroed_weight_count": 0,
        "actual_sparsity": 0.0,
        "missing_activation_scale_modules": [],
        "per_module": [],
    }


def add_module_stats(
    stats: dict[str, Any],
    *,
    module: str,
    shape: list[int],
    usable_in: int,
    zeroed: int,
    mask_method: str,
) -> None:
    total = int(shape[0] * usable_in)
    stats["module_count"] += 1
    stats["total_masked_weight_count"] += total
    stats["zeroed_weight_count"] += int(zeroed)
    stats["per_module"].append(
        {
            "module": module,
            "shape": shape,
            "usable_in_features": int(usable_in),
            "masked_weight_count": total,
            "zeroed_weight_count": int(zeroed),
            "actual_sparsity": int(zeroed) / total if total else 0.0,
            "mask_method": mask_method,
        }
    )


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    total = int(stats.get("total_masked_weight_count") or 0)
    stats["actual_sparsity"] = (
        int(stats.get("zeroed_weight_count") or 0) / total if total else 0.0
    )
    return stats


def generate_wanda_masks(
    model: Any,
    *,
    model_label: str,
    activation_scales: dict[str, Any],
    group_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    masks: dict[str, Any] = {}
    stats = mask_stats_template("wanda", model_label)
    for name, module in iter_target_linear_modules(model):
        weight = module.weight
        usable_in = (int(weight.shape[1]) // 4) * 4
        scale = activation_scale_for(
            name,
            activation_scales,
            usable_in=usable_in,
            device=weight.device,
            dtype=weight.dtype,
        )
        if scale is None:
            stats["missing_activation_scale_modules"].append(name)
        score, out_features, _, groups = score_view(weight, scale)
        group_bytes = top2_group_bytes_from_score(score, group_chunk_size)
        masks[name] = pack_group_bytes(group_bytes)
        add_module_stats(
            stats,
            module=name,
            shape=[int(weight.shape[0]), int(weight.shape[1])],
            usable_in=usable_in,
            zeroed=int(out_features * groups * 2),
            mask_method="wanda_activation_rms",
        )
        del score, group_bytes
    return masks, finalize_stats(stats)


def soft_threshold_nonneg(x: Any, tau: Any) -> Any:
    return (x - tau).clamp_min(0.0)


def proxsparse_prox_op_4(score: Any, lambda_value: float, iter_num: int) -> Any:
    torch = require_torch()
    sorted_score, original_indices = torch.sort(score, descending=True, dim=1)

    w2 = sorted_score.clone()
    w2[:, 2] = 0
    w2[:, 3] = 0

    w3 = torch.zeros_like(sorted_score)
    for _ in range(iter_num):
        prev = w3.clone()
        w3[:, 2] = soft_threshold_nonneg(sorted_score[:, 2], lambda_value * (w3[:, 0] * w3[:, 1]))
        w3[:, 1] = soft_threshold_nonneg(sorted_score[:, 1], lambda_value * (w3[:, 0] * w3[:, 2]))
        w3[:, 0] = soft_threshold_nonneg(sorted_score[:, 0], lambda_value * (w3[:, 1] * w3[:, 2]))
        if torch.sum(torch.abs(w3 - prev)) < 1e-8:
            break

    w4 = torch.zeros_like(sorted_score)
    for _ in range(iter_num):
        prev = w4.clone()
        w4[:, 3] = soft_threshold_nonneg(
            sorted_score[:, 3],
            lambda_value * (w4[:, 0] * w4[:, 1] + w4[:, 1] * w4[:, 2] + w4[:, 2] * w4[:, 0]),
        )
        w4[:, 2] = soft_threshold_nonneg(
            sorted_score[:, 2],
            lambda_value * (w4[:, 0] * w4[:, 1] + w4[:, 1] * w4[:, 3] + w4[:, 3] * w4[:, 0]),
        )
        w4[:, 1] = soft_threshold_nonneg(
            sorted_score[:, 1],
            lambda_value * (w4[:, 0] * w4[:, 2] + w4[:, 2] * w4[:, 3] + w4[:, 3] * w4[:, 0]),
        )
        w4[:, 0] = soft_threshold_nonneg(
            sorted_score[:, 0],
            lambda_value * (w4[:, 1] * w4[:, 2] + w4[:, 2] * w4[:, 3] + w4[:, 3] * w4[:, 1]),
        )
        if torch.sum(torch.abs(w4 - prev)) < 1e-8:
            break

    def reg(w: Any) -> Any:
        return (
            (w[:, 0] * w[:, 1] * w[:, 2]).abs()
            + (w[:, 1] * w[:, 2] * w[:, 3]).abs()
            + (w[:, 2] * w[:, 3] * w[:, 0]).abs()
            + (w[:, 3] * w[:, 0] * w[:, 1]).abs()
        )

    def obj(w: Any) -> Any:
        return 0.5 * torch.norm(w - sorted_score, p=2, dim=1).pow(2) + lambda_value * reg(w)

    choices = torch.min(torch.stack([obj(w2), obj(w3), obj(w4)]), dim=0).indices
    best = torch.where(choices.unsqueeze(1) == 0, w2, torch.where(choices.unsqueeze(1) == 1, w3, w4))
    reordered = torch.zeros_like(best)
    row_indices = torch.arange(best.shape[0], device=best.device).unsqueeze(1).expand_as(original_indices)
    reordered[row_indices, original_indices] = best
    return reordered


def generate_proxsparse_masks(
    model: Any,
    *,
    model_label: str,
    activation_scales: dict[str, Any],
    lambda_value: float,
    iter_num: int,
    group_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    masks: dict[str, Any] = {}
    stats = mask_stats_template("proxsparse", model_label)
    stats["proxsparse_lambda"] = lambda_value
    stats["proxsparse_iter_num"] = iter_num
    for name, module in iter_target_linear_modules(model):
        weight = module.weight
        usable_in = (int(weight.shape[1]) // 4) * 4
        scale = activation_scale_for(
            name,
            activation_scales,
            usable_in=usable_in,
            device=weight.device,
            dtype=weight.dtype,
        )
        if scale is None:
            stats["missing_activation_scale_modules"].append(name)
        score, out_features, _, groups = score_view(weight, scale)
        flat = score.reshape(-1, 4)
        group_bytes = torch.empty(flat.shape[0], dtype=torch.uint8, device=score.device)
        chunk_size = max(1, int(group_chunk_size))
        for start in range(0, flat.shape[0], chunk_size):
            prox_score = proxsparse_prox_op_4(
                flat[start : start + chunk_size],
                lambda_value,
                iter_num,
            )
            group_bytes[start : start + chunk_size] = top2_group_bytes_from_score(
                prox_score.view(-1, 1, 4),
                chunk_size,
            ).view(-1)
            del prox_score
        group_bytes = group_bytes.view(out_features, groups)
        masks[name] = pack_group_bytes(group_bytes)
        add_module_stats(
            stats,
            module=name,
            shape=[int(weight.shape[0]), int(weight.shape[1])],
            usable_in=usable_in,
            zeroed=int(out_features * groups * 2),
            mask_method="proxsparse_prox_activation_rms",
        )
        del score, flat, group_bytes
    return masks, finalize_stats(stats)


def maskllm_option_losses(score: Any) -> Any:
    torch = require_torch()
    option_keep = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        device=score.device,
        dtype=score.dtype,
    )
    removed = 1.0 - option_keep.view(1, 1, 6, 4)
    return (score.pow(2).unsqueeze(2) * removed).sum(dim=-1)


def generate_maskllm_masks(
    model: Any,
    *,
    model_label: str,
    activation_scales: dict[str, Any],
    steps: int,
    lr: float,
    temp_start: float,
    temp_end: float,
    scale_start: float,
    scale_end: float,
    prior_strength: float,
    seed: int,
    group_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    import torch.nn.functional as F

    masks: dict[str, Any] = {}
    stats = mask_stats_template("maskllm", model_label)
    stats.update(
        {
            "maskllm_steps": steps,
            "maskllm_lr": lr,
            "maskllm_temperature_range": [temp_start, temp_end],
            "maskllm_scale_range": [scale_start, scale_end],
            "maskllm_prior_strength": prior_strength,
        }
    )
    _, option_bytes = init_mask_constants()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for name, module in iter_target_linear_modules(model):
        weight = module.weight
        usable_in = (int(weight.shape[1]) // 4) * 4
        scale = activation_scale_for(
            name,
            activation_scales,
            usable_in=usable_in,
            device=weight.device,
            dtype=weight.dtype,
        )
        if scale is None:
            stats["missing_activation_scale_modules"].append(name)
        score, out_features, _, groups = score_view(weight, scale)
        flat = score.reshape(-1, 1, 4)
        flat_group_bytes = torch.empty(flat.shape[0], dtype=torch.uint8, device=score.device)
        chunk_size = max(1, int(group_chunk_size))
        option_bytes_device = option_bytes.to(device=score.device)
        for start in range(0, flat.shape[0], chunk_size):
            chunk_score = flat[start : start + chunk_size]
            losses = maskllm_option_losses(chunk_score).view(-1, 6).detach()
            norm = losses.detach().mean().clamp_min(1e-8)
            gate = (-losses / norm * prior_strength).float()
            if steps > 0:
                noise = torch.randn(gate.shape, generator=generator, dtype=gate.dtype).to(gate.device) * 0.01
                gate = (gate + noise).requires_grad_(True)
                optimizer = torch.optim.SGD([gate], lr=lr)
                for step in range(steps):
                    ratio = step / max(steps - 1, 1)
                    temperature = temp_start + (temp_end - temp_start) * ratio
                    scale_multiplier = scale_start + (scale_end - scale_start) * ratio
                    optimizer.zero_grad(set_to_none=True)
                    probs = F.gumbel_softmax(
                        gate * scale_multiplier,
                        tau=max(temperature, 1e-4),
                        hard=False,
                        dim=-1,
                    )
                    loss = (probs * losses).mean()
                    loss.backward()
                    optimizer.step()
                choice = gate.detach().argmax(dim=-1)
            else:
                choice = losses.argmin(dim=-1)
            flat_group_bytes[start : start + chunk_size] = option_bytes_device[choice].to(dtype=torch.uint8)
            del chunk_score, losses, gate, choice

        group_bytes = flat_group_bytes.view(out_features, groups)
        masks[name] = pack_group_bytes(group_bytes)
        add_module_stats(
            stats,
            module=name,
            shape=[int(weight.shape[0]), int(weight.shape[1])],
            usable_in=usable_in,
            zeroed=int(out_features * groups * 2),
            mask_method="maskllm_gumbel_activation_rms",
        )
        del score, flat, flat_group_bytes, group_bytes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return masks, finalize_stats(stats)


def mask_path(paths: RunPaths, model_label: str, method: str) -> Path:
    return paths.mask_root / model_label / f"{method}.pt"


def pruned_model_path(paths: RunPaths, model_label: str, method: str) -> Path:
    return paths.pruned_model_root / model_label / method


def save_mask_cache(
    path: Path,
    *,
    masks: dict[str, Any],
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "metadata": metadata,
        "stats": stats,
        "masks": {name: tensor.cpu() for name, tensor in sorted(masks.items())},
        "row_scales": {},
    }
    torch = require_torch()
    torch.save(cache, path)
    write_json(path.with_suffix(".json"), {"metadata": metadata, "stats": stats})


def load_mask_cache(path: Path) -> dict[str, Any]:
    torch = require_torch()
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict) or "masks" not in cache:
        raise RuntimeError(f"invalid mask cache: {path}")
    return cache


def regression_scale_from_moments(
    cross: Any,
    sparse_energy: Any,
    *,
    min_scale: float,
    max_scale: float,
) -> Any:
    """Return the channelwise least-squares scale for dense ~= scale*sparse."""

    torch = require_torch()
    if not (0.0 < min_scale <= max_scale and math.isfinite(max_scale)):
        raise ValueError("row-scale bounds must satisfy 0 < min <= max < inf")
    scale = torch.ones_like(cross, dtype=torch.float32)
    valid = sparse_energy > torch.finfo(torch.float32).tiny
    scale[valid] = cross[valid] / sparse_energy[valid]
    return scale.clamp_(min=min_scale, max=max_scale)


def collect_regression_row_scales(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    mask_cache: dict[str, Any],
    *,
    max_seq_len: int,
    batch_size: int,
    min_scale: float,
    max_scale: float,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit output-channel scales on dense-model C4 activations."""

    torch = require_torch()
    import torch.nn.functional as F

    if batch_size <= 0:
        raise ValueError("row-scale calibration batch size must be positive")
    masks = mask_cache.get("masks", {})
    modules = {
        name: module
        for name, module in iter_target_linear_modules(model)
        if name in masks
    }
    if not modules:
        raise RuntimeError("mask cache does not match any target model modules")

    cross: dict[str, Any] = {}
    sparse_energy: dict[str, Any] = {}
    dense_energy: dict[str, Any] = {}
    token_counts: dict[str, int] = {}
    current_token_mask: Any | None = None
    handles = []

    def make_hook(name: str):
        mask_bytes = masks[name]

        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs or not isinstance(output, torch.Tensor):
                return
            x = inputs[0].detach()
            dense = output.detach()
            flat_x = x.reshape(-1, x.shape[-1])
            flat_dense = dense.reshape(-1, dense.shape[-1])
            if current_token_mask is not None and current_token_mask.numel() == flat_x.shape[0]:
                selected = current_token_mask
                flat_x = flat_x[selected]
                flat_dense = flat_dense[selected]
            if flat_x.numel() == 0:
                return

            weight = module.weight.detach()
            out_features, in_features = map(int, weight.shape)
            groups = in_features // 4
            group_bytes = unpack_group_bytes(
                mask_bytes,
                out_features=out_features,
                groups=groups,
            ).to(device=weight.device, non_blocking=True)
            keep = group_bytes_to_keep(group_bytes, device=weight.device).reshape(
                out_features, groups * 4
            )
            sparse_weight = weight[:, : groups * 4] * keep.to(dtype=weight.dtype)
            sparse = F.linear(flat_x[:, : groups * 4], sparse_weight, None)
            if module.bias is not None:
                flat_dense = flat_dense - module.bias.detach().reshape(1, -1)

            dense_f = flat_dense.float()
            sparse_f = sparse.float()
            batch_cross = (dense_f * sparse_f).sum(dim=0)
            batch_sparse_energy = sparse_f.square().sum(dim=0)
            batch_dense_energy = dense_f.square().sum(dim=0)
            if name not in cross:
                cross[name] = torch.zeros_like(batch_cross)
                sparse_energy[name] = torch.zeros_like(batch_sparse_energy)
                dense_energy[name] = torch.zeros_like(batch_dense_energy)
                token_counts[name] = 0
            cross[name].add_(batch_cross)
            sparse_energy[name].add_(batch_sparse_energy)
            dense_energy[name].add_(batch_dense_energy)
            token_counts[name] += int(flat_x.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                )
                current_token_mask = encoded.get("attention_mask")
                if current_token_mask is not None:
                    current_token_mask = current_token_mask.reshape(-1).bool().to(device)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
                print(
                    f"  row-scale calibration prompts "
                    f"{min(start + batch_size, len(prompts))}/{len(prompts)}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    row_scales: dict[str, Any] = {}
    module_stats: dict[str, Any] = {}
    for name in sorted(cross):
        scale = regression_scale_from_moments(
            cross[name],
            sparse_energy[name],
            min_scale=min_scale,
            max_scale=max_scale,
        )
        row_scales[name] = scale.to(dtype=torch.float16).cpu()
        before = (
            dense_energy[name] - 2.0 * cross[name] + sparse_energy[name]
        ).clamp_min_(0.0)
        after = (
            dense_energy[name]
            - 2.0 * scale * cross[name]
            + scale.square() * sparse_energy[name]
        ).clamp_min_(0.0)
        module_stats[name] = {
            "tokens": token_counts[name],
            "scale_min": float(scale.min().item()),
            "scale_mean": float(scale.mean().item()),
            "scale_max": float(scale.max().item()),
            "relative_mse_before": float(
                before.sum().div(dense_energy[name].sum().clamp_min(1e-12)).item()
            ),
            "relative_mse_after": float(
                after.sum().div(dense_energy[name].sum().clamp_min(1e-12)).item()
            ),
        }
    return row_scales, {
        "num_prompts": len(prompts),
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "num_modules": len(row_scales),
        "modules": module_stats,
    }


def calibrate_row_scales(args: argparse.Namespace) -> list[Path]:
    torch = require_torch()
    set_seed(args.seed)
    paths = selected_paths(args)
    model_ids = resolve_models(args)
    methods = resolve_methods(args, pruning_only=True)
    dtype = dtype_from_arg(args.dtype)
    prompts = load_calibration_prompt_file(
        args.calibration_prompts,
        args.calibration_num_examples,
        args.seed,
    )
    output_mask_root = args.row_scale_output_root
    if output_mask_root is None:
        output_mask_root = paths.mask_root.with_name(
            f"{paths.mask_root.name}_{args.row_scale_suffix}"
        )
    output_mask_root = output_mask_root.resolve()
    output_mask_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_mask_root / "run_config.json",
        {
            "argv": sys.argv,
            "models": model_ids,
            "methods": methods,
            "source_mask_root": str(paths.mask_root),
            "output_mask_root": str(output_mask_root),
            "calibration_prompts": str(args.calibration_prompts.resolve()),
            "calibration_num_examples": len(prompts),
            "calibration_max_seq_len": args.calibration_max_seq_len,
            "row_scale_batch_size": args.row_scale_batch_size,
            "row_scale_min": args.row_scale_min,
            "row_scale_max": args.row_scale_max,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "created_at": timestamp(),
        },
    )
    generated: list[Path] = []
    for model_label, model_id in model_ids.items():
        model, tokenizer = load_model_and_tokenizer(
            model_id,
            dtype=dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        try:
            for method in methods:
                source_path = mask_path(paths, model_label, method)
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"missing source mask cache {source_path}; run generate-masks first"
                    )
                output_path = output_mask_root / model_label / f"{method}.pt"
                if output_path.exists() and not args.force:
                    print(f"[INFO] Reusing row-scale cache: {output_path}", flush=True)
                    generated.append(output_path)
                    continue
                cache = load_mask_cache(source_path)
                print(
                    f"[INFO] Calibrating row scales for {model_label}/{method}",
                    flush=True,
                )
                row_scales, calibration_stats = collect_regression_row_scales(
                    model,
                    tokenizer,
                    prompts,
                    cache,
                    max_seq_len=args.calibration_max_seq_len,
                    batch_size=args.row_scale_batch_size,
                    min_scale=args.row_scale_min,
                    max_scale=args.row_scale_max,
                    device=args.device,
                )
                output_cache = dict(cache)
                output_cache["row_scales"] = row_scales
                output_cache["metadata"] = dict(cache.get("metadata", {}))
                output_cache["metadata"]["row_scale_calibration"] = {
                    "source_method": method,
                    "source_path": str(source_path.resolve()),
                    "prompt_path": str(args.calibration_prompts.resolve()),
                    **{key: value for key, value in calibration_stats.items() if key != "modules"},
                }
                output_cache["stats"] = dict(cache.get("stats", {}))
                output_cache["stats"]["row_scale_calibration"] = calibration_stats
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output_cache, output_path)
                write_json(
                    output_path.with_suffix(".json"),
                    {
                        "metadata": output_cache["metadata"],
                        "row_scale_calibration": calibration_stats,
                    },
                )
                generated.append(output_path)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return generated


def generate_masks(args: argparse.Namespace) -> list[Path]:
    torch = require_torch()
    set_seed(args.seed)
    paths = selected_paths(args)
    model_ids = resolve_models(args)
    methods = resolve_methods(args, pruning_only=True)
    dtype = dtype_from_arg(args.dtype)
    prompts = load_calibration_prompt_file(
        args.calibration_prompts,
        args.calibration_num_examples,
        args.seed,
    )
    generated: list[Path] = []

    paths.mask_root.mkdir(parents=True, exist_ok=True)
    write_json(
        paths.mask_root / "run_config.json",
        {
            "argv": sys.argv,
            "models": model_ids,
            "methods": methods,
            "calibration_prompts": str(args.calibration_prompts.resolve()),
            "calibration_num_examples": len(prompts),
            "calibration_max_seq_len": args.calibration_max_seq_len,
            "calibration_cache_root": str(args.calibration_cache_root.resolve()),
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "created_at": timestamp(),
        },
    )

    for model_label, model_id in model_ids.items():
        activation_scales, activation_metadata = load_activation_cache(
            args.calibration_cache_root,
            model_label,
        )
        model, tokenizer = load_model_and_tokenizer(
            model_id,
            dtype=dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        try:
            for method in methods:
                out_path = mask_path(paths, model_label, method)
                if out_path.exists() and not args.force:
                    print(f"[INFO] Reusing mask cache: {out_path}", flush=True)
                    generated.append(out_path)
                    continue
                print(f"[INFO] Generating {method} mask for {model_label}", flush=True)
                if method == "wanda":
                    masks, stats = generate_wanda_masks(
                        model,
                        model_label=model_label,
                        activation_scales=activation_scales,
                        group_chunk_size=args.group_chunk_size,
                    )
                elif method == "proxsparse":
                    masks, stats = generate_proxsparse_masks(
                        model,
                        model_label=model_label,
                        activation_scales=activation_scales,
                        lambda_value=args.proxsparse_lambda,
                        iter_num=args.proxsparse_iters,
                        group_chunk_size=args.group_chunk_size,
                    )
                elif method == "maskllm":
                    masks, stats = generate_maskllm_masks(
                        model,
                        model_label=model_label,
                        activation_scales=activation_scales,
                        steps=args.maskllm_steps,
                        lr=args.maskllm_lr,
                        temp_start=args.maskllm_temp_start,
                        temp_end=args.maskllm_temp_end,
                        scale_start=args.maskllm_scale_start,
                        scale_end=args.maskllm_scale_end,
                        prior_strength=args.maskllm_prior_strength,
                        seed=args.seed,
                        group_chunk_size=args.group_chunk_size,
                    )
                else:
                    raise ValueError(f"unsupported method: {method}")
                metadata = {
                    "method": method,
                    "model_label": model_label,
                    "model_id": model_id,
                    "mask_format": "packed_2x4_group_bytes",
                    "source": "speclink_offline_24_pruning",
                    "external_references": {
                        "wanda": "https://github.com/locuslab/wanda",
                        "proxsparse": "https://github.com/amazon-science/ProxSparse",
                        "maskllm": "https://github.com/NVlabs/MaskLLM",
                    },
                    "calibration_prompts": str(args.calibration_prompts.resolve()),
                    "calibration_num_examples": len(prompts),
                    "calibration_max_seq_len": args.calibration_max_seq_len,
                    "calibration_cache": activation_metadata,
                    "created_at": timestamp(),
                }
                save_mask_cache(out_path, masks=masks, metadata=metadata, stats=stats)
                generated.append(out_path)
                print(f"[INFO] Wrote mask cache: {out_path}", flush=True)
                del masks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return generated


def apply_masks_to_model(model: Any, masks: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    stats: dict[str, Any] = {
        "module_count": 0,
        "total_masked_weight_count": 0,
        "zeroed_weight_count": 0,
        "actual_sparsity": 0.0,
        "missing_mask_modules": [],
        "unexpected_mask_modules": sorted(set(masks)),
        "per_module": [],
    }
    with torch.no_grad():
        for name, module in iter_target_linear_modules(model):
            stats["unexpected_mask_modules"] = [
                item for item in stats["unexpected_mask_modules"] if item != name
            ]
            mask_bytes = masks.get(name)
            if mask_bytes is None:
                stats["missing_mask_modules"].append(name)
                continue
            weight = module.weight
            out_features = int(weight.shape[0])
            in_features = int(weight.shape[1])
            usable_in = (in_features // 4) * 4
            groups = usable_in // 4
            group_bytes = unpack_group_bytes(mask_bytes, out_features=out_features, groups=groups)
            keep = group_bytes_to_keep(group_bytes, device=weight.device)
            keep_counts = keep.sum(dim=-1)
            if not bool((keep_counts == 2).all().item()):
                raise RuntimeError(f"mask for {name} is not exactly 2:4")
            view = weight[:, :usable_in].view(out_features, groups, 4)
            view.masked_fill_(~keep, 0)
            total = out_features * usable_in
            zeroed = int((~keep).sum().item())
            stats["module_count"] += 1
            stats["total_masked_weight_count"] += total
            stats["zeroed_weight_count"] += zeroed
            stats["per_module"].append(
                {
                    "module": name,
                    "shape": [out_features, in_features],
                    "usable_in_features": usable_in,
                    "zeroed_weight_count": zeroed,
                    "masked_weight_count": total,
                    "actual_sparsity": zeroed / total if total else 0.0,
                }
            )
    total = int(stats["total_masked_weight_count"])
    stats["actual_sparsity"] = int(stats["zeroed_weight_count"]) / total if total else 0.0
    return stats


def materialize_models(args: argparse.Namespace) -> list[Path]:
    torch = require_torch()
    set_seed(args.seed)
    paths = selected_paths(args)
    model_ids = resolve_models(args)
    methods = resolve_methods(args, pruning_only=True)
    dtype = dtype_from_arg(args.dtype)
    materialized: list[Path] = []
    paths.pruned_model_root.mkdir(parents=True, exist_ok=True)

    for model_label, model_id in model_ids.items():
        for method in methods:
            out_dir = pruned_model_path(paths, model_label, method)
            done_path = out_dir / "speclink_offline_24_pruned_model.json"
            if done_path.exists() and not args.force:
                print(f"[INFO] Reusing pruned model: {out_dir}", flush=True)
                materialized.append(out_dir)
                continue
            cache_path = mask_path(paths, model_label, method)
            if not cache_path.exists():
                raise FileNotFoundError(f"missing mask cache for {model_label}/{method}: {cache_path}")
            cache = load_mask_cache(cache_path)
            print(f"[INFO] Materializing {model_label}/{method} -> {out_dir}", flush=True)
            model, tokenizer = load_model_and_tokenizer(
                model_id,
                dtype=dtype,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
            try:
                apply_stats = apply_masks_to_model(model, cache["masks"])
                out_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(
                    out_dir,
                    safe_serialization=True,
                    max_shard_size=args.max_shard_size,
                )
                tokenizer.save_pretrained(out_dir)
                metadata = {
                    "model_label": model_label,
                    "base_model_id": model_id,
                    "method": method,
                    "mask_cache": str(cache_path.resolve()),
                    "apply_stats": apply_stats,
                    "created_at": timestamp(),
                }
                write_json(done_path, metadata)
                materialized.append(out_dir)
                print(f"[INFO] Wrote pruned model: {out_dir}", flush=True)
            finally:
                del model
                del tokenizer
                del cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return materialized


def append_command(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")


def decode_sparse_lm_eval_modes(raw_modes: str) -> str:
    """Route decode-only sparse runs through the token-dense verifier path.

    The dense speculative mode disables SpecLink structured 2:4 before vLLM
    starts. For offline-mask decode-only runs, use token_dense_d0 together with
    sparse_only_decode so every decode verifier row uses the cached 2:4 mask
    while prefill remains dense.
    """
    modes = []
    for mode in parse_csv_list(raw_modes):
        if mode in {"eagle3_dense", "eagle3_dense_eager"}:
            mode = "token_dense_d0"
        if mode == "dense_ar":
            continue
        if mode not in modes:
            modes.append(mode)
    return ",".join(modes or ["token_dense_d0"])


def lm_eval_modes_for_method(
    raw_modes: str,
    *,
    decode_sparse_only: bool,
    method: str,
) -> str:
    if decode_sparse_only and method != ORIGINAL_METHOD:
        return decode_sparse_lm_eval_modes(raw_modes)
    return raw_modes


def run_lm_eval(args: argparse.Namespace) -> None:
    paths = selected_paths(args)
    model_ids = resolve_models(args)
    methods = resolve_methods(args)
    failures: list[dict[str, Any]] = []
    commands_path = paths.output_root / "commands.sh"
    if commands_path.exists() and args.force:
        commands_path.unlink()

    for model_label in model_ids:
        for method in methods:
            mask_cache_for_decode: Path | None = None
            if method == ORIGINAL_METHOD:
                model_dir = Path(model_ids[model_label]).resolve()
            elif args.decode_sparse_only:
                model_dir = Path(model_ids[model_label]).resolve()
                mask_cache_for_decode = mask_path(paths, model_label, method)
                if not mask_cache_for_decode.exists():
                    raise FileNotFoundError(
                        f"missing mask cache for {model_label}/{method}: "
                        f"{mask_cache_for_decode}"
                    )
            else:
                model_dir = pruned_model_path(paths, model_label, method)
                if not model_dir.exists():
                    raise FileNotFoundError(f"missing pruned model: {model_dir}")
            eval_dir = paths.output_root / "lm_eval" / model_label / method
            lm_eval_modes = lm_eval_modes_for_method(
                args.lm_eval_modes,
                decode_sparse_only=args.decode_sparse_only,
                method=method,
            )
            command = [
                sys.executable,
                "-u",
                str(SCRIPT_DIR / "run_lm_eval_accuracy.py"),
                "--models",
                model_label,
                "--mode",
                lm_eval_modes,
                "--task",
                args.lm_eval_task,
                "--limit",
                str(args.lm_eval_limit),
                "--model-path",
                str(model_dir),
                "--tokenizer-path",
                str(model_dir),
                "--output-dir",
                str(eval_dir),
                "--num-spec-tokens",
                str(args.num_spec_tokens),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--port-base",
                str(args.port_base),
                "--health-timeout-s",
                str(args.health_timeout_s),
                "--request-timeout-s",
                str(args.request_timeout_s),
                "--batch-size",
                str(args.batch_size),
                "--num-concurrent",
                str(args.num_concurrent),
            ]
            if mask_cache_for_decode is not None:
                command.extend(
                    [
                        "--dtype",
                        "fp16",
                        "--token-dense-mask-method",
                        "inherit",
                        "--token-dense-linear-strategy",
                        "sparse_only_decode",
                        "--token-dense-cudagraph-mode",
                        "none",
                    ]
                )
            if args.apply_chat_template:
                command.append("--apply-chat-template")
            if args.enforce_eager:
                command.append("--enforce-eager")
            if args.resume_lm_eval:
                command.append("--resume")
            append_command(commands_path, command)
            env = os.environ.copy()
            if mask_cache_for_decode is not None:
                env["SPECLINK_STRUCTURED_24_MASK_CACHE"] = str(
                    mask_cache_for_decode.resolve()
                )
                env["SPECLINK_STRUCTURED_24_CACHE_STRICT"] = "1"
                write_json(
                    eval_dir / "decode_sparse_env.json",
                    {
                        "prefill": "dense",
                        "decode": "token_dense_structured_24",
                        "lm_eval_modes": lm_eval_modes,
                        "routing_mode": (
                            "token_dense_d0 with sparse_only_decode routes every "
                            "decode verifier row through the selected cached 2:4 "
                            "mask; prefill stays dense"
                        ),
                        "mask_cache": str(mask_cache_for_decode.resolve()),
                        "method": method,
                        "model_label": model_label,
                    },
                )
            print(f"[INFO] Running lm-eval for {model_label}/{method}", flush=True)
            completed = subprocess.run(
                command,
                cwd=str(SPECULATORS_ROOT),
                env=env,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                failures.append(
                    {
                        "model_label": model_label,
                        "method": method,
                        "returncode": completed.returncode,
                        "eval_dir": str(eval_dir),
                    }
                )
                write_json(paths.output_root / "lm_eval_failures.json", failures)
                if not args.keep_going:
                    raise RuntimeError(f"lm-eval failed for {model_label}/{method}")
    rows = write_combined_summary(paths)
    write_accuracy_plots(args, paths, rows)
    if failures:
        write_json(paths.output_root / "lm_eval_failures.json", failures)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


ACCURACY_METRIC_PRIORITY = [
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match,get_response",
    "exact_match,none",
    "exact_match",
    "acc,none",
    "acc",
    "acc_norm,none",
    "pass@1,create_test",
    "pass@1",
]


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def result_sample_count(data: dict[str, Any], task_name: str) -> int | None:
    for section in ("n-samples", "n_samples"):
        values = data.get(section)
        if not isinstance(values, dict):
            continue
        raw = values.get(task_name)
        if isinstance(raw, dict):
            raw = raw.get("effective") or raw.get("original")
        count = as_int(raw)
        if count is not None:
            return count
    return None


def choose_accuracy_metric(task_result: dict[str, Any]) -> tuple[str, float | None]:
    for key in ACCURACY_METRIC_PRIORITY:
        if key in task_result:
            return key, as_float(task_result.get(key))
    for key, value in task_result.items():
        if (
            key.endswith("_stderr")
            or key in {"alias", "name", "samples", "sample_len"}
        ):
            continue
        score = as_float(value)
        if score is not None:
            return key, score
    return "", None


def enrich_with_result_json(row: dict[str, Any]) -> None:
    task_name = str(row.get("task_result_name") or row.get("task") or "")
    result_path = Path(str(row.get("result_path") or ""))
    if not task_name or not result_path.exists():
        row["accuracy"] = row.get("score", "")
        row["accuracy_metric"] = row.get("metric", "")
        return
    data = json.loads(result_path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    task_result = results.get(task_name)
    if not isinstance(task_result, dict) and len(results) == 1:
        task_name, task_result = next(iter(results.items()))
        row["task_result_name"] = task_name
    if not isinstance(task_result, dict):
        row["accuracy"] = row.get("score", "")
        row["accuracy_metric"] = row.get("metric", "")
        return

    metric, accuracy = choose_accuracy_metric(task_result)
    samples = result_sample_count(data, task_name) or as_int(row.get("samples"))
    if accuracy is not None:
        row["metric"] = metric
        row["accuracy_metric"] = metric
        row["score"] = accuracy
        row["accuracy"] = accuracy
        if samples is not None:
            row["samples"] = samples
            row["correct"] = int(round(float(accuracy) * int(samples)))
    else:
        row["accuracy"] = row.get("score", "")
        row["accuracy_metric"] = row.get("metric", "")


def write_combined_summary(paths: RunPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted((paths.output_root / "lm_eval").rglob("summary.csv")):
        parts = summary.relative_to(paths.output_root / "lm_eval").parts
        if len(parts) < 3:
            continue
        model_label, method = parts[0], parts[1]
        for row in read_csv_rows(summary):
            row["pruning_method"] = method
            row["source_model_label"] = model_label
            row["summary_path"] = str(summary)
            enrich_with_result_json(row)
            rows.append(row)

    dense_scores: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.get("mode") != "dense_ar":
            continue
        accuracy = as_float(row.get("accuracy"))
        if accuracy is None:
            continue
        key = (
            str(row.get("source_model_label")),
            str(row.get("pruning_method")),
            str(row.get("task_result_name") or row.get("task") or ""),
        )
        dense_scores[key] = accuracy
    for row in rows:
        key = (
            str(row.get("source_model_label")),
            str(row.get("pruning_method")),
            str(row.get("task_result_name") or row.get("task") or ""),
        )
        dense = dense_scores.get(key)
        accuracy = as_float(row.get("accuracy"))
        row["dense_ar_accuracy"] = dense if dense is not None else ""
        row["delta_pp_vs_dense_ar"] = (
            (accuracy - dense) * 100.0
            if accuracy is not None and dense is not None
            else ""
        )

    out_csv = paths.output_root / "offline_24_lm_eval_summary.csv"
    out_json = paths.output_root / "offline_24_lm_eval_summary.json"
    fields = [
        "source_model_label",
        "pruning_method",
        "mode",
        "task",
        "task_result_name",
        "metric",
        "accuracy",
        "correct",
        "samples",
        "dense_ar_accuracy",
        "delta_pp_vs_dense_ar",
        "status",
        "spec_acceptance_rate",
        "spec_accepted_tokens",
        "spec_draft_tokens",
        "result_path",
        "summary_path",
    ]
    paths.output_root.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (paths.output_root / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Offline 2:4 lm-eval Summary\n\n")
        handle.write("| Model | Method | Mode | Task | Metric | Accuracy | Correct | Samples | Delta pp vs AR | Spec accept | Status |\n")
        handle.write("|------|--------|------|------|--------|---------:|--------:|--------:|---------------:|------------:|--------|\n")
        for row in rows:
            accuracy = as_float(row.get("accuracy"))
            delta = as_float(row.get("delta_pp_vs_dense_ar"))
            accept = as_float(row.get("spec_acceptance_rate"))
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("source_model_label", "")),
                        str(row.get("pruning_method", "")),
                        str(row.get("mode", "")),
                        str(row.get("task_result_name") or row.get("task") or ""),
                        str(row.get("metric", "")),
                        f"{accuracy:.4f}" if accuracy is not None else "",
                        str(row.get("correct", "")),
                        str(row.get("samples", "")),
                        f"{delta:.2f}" if delta is not None else "",
                        f"{accept:.4f}" if accept is not None else "",
                        str(row.get("status", "")),
                    ]
                )
                + " |\n"
            )
    return rows


def token_dense_budget(mode: str) -> int | None:
    prefix = "token_dense_d"
    if not mode.startswith(prefix):
        return None
    raw = mode.removeprefix(prefix)
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value in {16, 32, 64, 128} else None


def method_display(method: str) -> str:
    return {
        "wanda": "Wanda",
        "proxsparse": "ProxSparse",
        "maskllm": "MaskLLM",
    }.get(method, method)


def model_color(model_label: str) -> str:
    return {
        "qwen3_8b": "#1f77b4",
        "llama3_1_8b": "#ff7f0e",
    }.get(model_label, "#4C78A8")


def method_marker(method: str) -> str:
    return {
        "wanda": "s",
        "proxsparse": "^",
        "maskllm": "D",
        "token_dense": "o",
    }.get(method, "o")


def method_linestyle(method: str) -> str:
    return {
        "wanda": "--",
        "proxsparse": "-.",
        "maskllm": ":",
        "token_dense": "-",
    }.get(method, "--")


def offline_accuracy_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        mode = str(row.get("mode") or "")
        if mode != "eagle3_dense" and not mode.startswith("token_dense_"):
            continue
        task_name = str(row.get("task_result_name") or row.get("task") or "")
        if task_name != "gsm8k_cot":
            continue
        method = str(row.get("pruning_method") or "")
        if method not in PRUNING_METHODS:
            continue
        accuracy = as_float(row.get("accuracy"))
        if accuracy is None or not math.isfinite(accuracy):
            continue
        points.append(
            {
                "model_label": str(
                    row.get("source_model_label") or row.get("model_label") or ""
                ),
                "method": method,
                "mode": mode,
                "accuracy": accuracy,
                "samples": row.get("samples", ""),
                "source": "offline_24",
            }
        )
    return points


def token_dense_accuracy_points(summary_paths: list[Path]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        for row in read_csv_rows(summary_path):
            if row.get("status") != "ok":
                continue
            task_name = str(row.get("task_result_name") or row.get("task") or "")
            if task_name != "gsm8k_cot":
                continue
            mode = str(row.get("mode") or "")
            budget = token_dense_budget(mode)
            if budget is None:
                continue
            accuracy = as_float(row.get("score"))
            if accuracy is None or not math.isfinite(accuracy):
                continue
            points.append(
                {
                    "model_label": str(row.get("model_label") or ""),
                    "method": mode,
                    "budget": budget,
                    "accuracy": accuracy,
                    "samples": row.get("samples", ""),
                    "source": "token_dense",
                    "summary_path": str(summary_path),
                }
            )
    return points


def write_plot_source_csv(path: Path, points: list[dict[str, Any]]) -> None:
    if not points:
        return
    fields = sorted({key for point in points for key in point})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)


def write_offline_accuracy_plot(figures_dir: Path, points: list[dict[str, Any]]) -> None:
    if not points:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    method_order = ["wanda", "proxsparse", "maskllm"]
    model_order = sorted({point["model_label"] for point in points})
    offsets = {
        model: (-0.12 if index == 0 else 0.12 if index == 1 else 0.0)
        for index, model in enumerate(model_order)
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for point in points:
        x = method_order.index(str(point["method"])) + offsets.get(
            str(point["model_label"]), 0.0
        )
        y = float(point["accuracy"])
        method = str(point["method"])
        color = model_color(str(point["model_label"]))
        ax.plot(
            [x - 0.07, x + 0.07],
            [y, y],
            color=color,
            linestyle=method_linestyle(method),
            marker=method_marker(method),
            markersize=6,
            linewidth=1.8,
        )
    ax.set_title("GSM8K CoT offline 2:4 pruning accuracy")
    ax.set_ylabel("Exact match")
    ax.set_xticks(
        range(len(method_order)), [method_display(method) for method in method_order]
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    model_handles = [
        Line2D([0], [0], color=model_color(model), linewidth=2.0, label=model)
        for model in model_order
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=method_linestyle(method),
            marker=method_marker(method),
            linewidth=1.8,
            label=method_display(method),
        )
        for method in method_order
    ]
    first = ax.legend(handles=model_handles, title="Model", loc="lower left")
    ax.add_artist(first)
    ax.legend(handles=method_handles, title="Method", loc="lower right")
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "offline_24_gsm8k_cot_accuracy.png", dpi=200)
    plt.close(fig)


def write_combined_accuracy_plot(
    figures_dir: Path,
    offline_points: list[dict[str, Any]],
    token_dense_points: list[dict[str, Any]],
) -> None:
    if not offline_points and not token_dense_points:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.5),
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    model_order = sorted(
        {point["model_label"] for point in offline_points + token_dense_points}
    )
    ax = axes[0]
    for model in model_order:
        series = sorted(
            [point for point in token_dense_points if point["model_label"] == model],
            key=lambda point: float(point["budget"]),
        )
        if not series:
            continue
        ax.plot(
            [float(point["budget"]) for point in series],
            [float(point["accuracy"]) for point in series],
            color=model_color(model),
            linestyle=method_linestyle("token_dense"),
            marker=method_marker("token_dense"),
            linewidth=1.8,
            markersize=5,
        )
    ax.set_title("token_dense dense budget")
    ax.set_xlabel("dense draft-token budget")
    ax.set_ylabel("GSM8K CoT exact match")
    ax.set_xticks([16, 32, 64, 128])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    method_order = ["wanda", "proxsparse", "maskllm"]
    offsets = {
        model: (-0.12 if index == 0 else 0.12 if index == 1 else 0.0)
        for index, model in enumerate(model_order)
    }
    for point in offline_points:
        method = str(point["method"])
        model = str(point["model_label"])
        x = method_order.index(method) + offsets.get(model, 0.0)
        y = float(point["accuracy"])
        ax.plot(
            [x - 0.07, x + 0.07],
            [y, y],
            color=model_color(model),
            linestyle=method_linestyle(method),
            marker=method_marker(method),
            markersize=6,
            linewidth=1.8,
        )
    ax.set_title("offline 2:4 pruning")
    ax.set_xticks(
        range(len(method_order)), [method_display(method) for method in method_order]
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    model_handles = [
        Line2D([0], [0], color=model_color(model), linewidth=2.0, label=model)
        for model in model_order
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            marker="o",
            linewidth=1.8,
            label="token_dense",
        ),
        *[
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=method_linestyle(method),
                marker=method_marker(method),
                linewidth=1.8,
                label=method_display(method),
            )
            for method in method_order
        ],
    ]
    fig.legend(
        handles=model_handles,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.33, -0.02),
        ncol=len(model_handles),
    )
    fig.legend(
        handles=method_handles,
        title="Method",
        loc="lower center",
        bbox_to_anchor=(0.73, -0.02),
        ncol=2,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "gsm8k_cot_accuracy_method_comparison.png", dpi=200)
    plt.close(fig)


def write_accuracy_plots(
    args: argparse.Namespace,
    paths: RunPaths,
    rows: list[dict[str, Any]],
) -> None:
    figures_dir = paths.output_root / "figures"
    offline_points = offline_accuracy_points(rows)
    comparison_summaries = [
        path.resolve() for path in (args.comparison_summary or []) if path.exists()
    ]
    token_dense_points = token_dense_accuracy_points(comparison_summaries)
    write_plot_source_csv(
        paths.output_root / "accuracy_plot_rows.csv",
        offline_points + token_dense_points,
    )
    write_offline_accuracy_plot(figures_dir, offline_points)
    write_combined_accuracy_plot(figures_dir, offline_points, token_dense_points)


def run_all(args: argparse.Namespace) -> None:
    paths = selected_paths(args)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        paths.output_root / "run_config.json",
        {
            "argv": sys.argv,
            "mask_root": str(paths.mask_root),
            "pruned_model_root": str(paths.pruned_model_root),
            "output_root": str(paths.output_root),
            "created_at": timestamp(),
        },
    )
    generate_masks(args)
    materialize_models(args)
    run_lm_eval(args)
    print(paths.output_root)


def self_test() -> None:
    torch = require_torch()
    set_seed(123)
    score = torch.tensor([[[1.0, 4.0, 3.0, 2.0], [0.5, 0.7, 0.1, 0.9]]])
    keep = top2_mask_from_score(score)
    group_bytes = keep_to_group_bytes(keep)
    packed = pack_group_bytes(group_bytes)
    unpacked = unpack_group_bytes(packed, out_features=1, groups=2)
    restored = group_bytes_to_keep(unpacked, device=score.device)
    assert torch.equal(keep, restored)
    prox = proxsparse_prox_op_4(score.reshape(-1, 4), 0.1, 4)
    assert prox.shape == (2, 4)
    losses = maskllm_option_losses(score)
    assert losses.shape == (1, 2, 6)
    scale = regression_scale_from_moments(
        torch.tensor([4.0]),
        torch.tensor([2.0]),
        min_scale=0.5,
        max_scale=3.0,
    )
    assert torch.equal(scale, torch.tensor([2.0]))
    print("offline_24_pruning self-test ok")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--methods", default="wanda,proxsparse,maskllm")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--pruned-model-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--calibration-prompts", type=Path, default=DEFAULT_C4_CALIBRATION_PROMPTS)
    parser.add_argument("--calibration-cache-root", type=Path, default=DEFAULT_C4_CALIBRATION_CACHE_ROOT)
    parser.add_argument("--calibration-num-examples", type=int, default=512)
    parser.add_argument("--calibration-max-seq-len", type=int, default=512)
    parser.add_argument("--row-scale-output-root", type=Path, default=None)
    parser.add_argument("--row-scale-suffix", default="regression")
    parser.add_argument("--row-scale-batch-size", type=int, default=1)
    parser.add_argument("--row-scale-min", type=float, default=0.5)
    parser.add_argument("--row-scale-max", type=float, default=1.5)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--group-chunk-size", type=int, default=1048576)
    parser.add_argument("--proxsparse-lambda", type=float, default=0.85)
    parser.add_argument("--proxsparse-iters", type=int, default=30)
    parser.add_argument("--maskllm-steps", type=int, default=6)
    parser.add_argument("--maskllm-lr", type=float, default=1.0)
    parser.add_argument("--maskllm-temp-start", type=float, default=4.0)
    parser.add_argument("--maskllm-temp-end", type=float, default=0.05)
    parser.add_argument("--maskllm-scale-start", type=float, default=1.0)
    parser.add_argument("--maskllm-scale-end", type=float, default=5.0)
    parser.add_argument("--maskllm-prior-strength", type=float, default=2.0)
    parser.add_argument("--lm-eval-task", default="gsm8k_cot")
    parser.add_argument("--lm-eval-limit", default="64")
    parser.add_argument("--lm-eval-modes", default="dense_ar,eagle3_dense")
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--port-base", type=int, default=8260)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume-lm-eval", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--decode-sparse-only",
        action="store_true",
        help=(
            "Load the original dense model and attach the selected offline 2:4 "
            "mask only to token-dense decode/verification routing. Prefill stays dense."
        ),
    )
    parser.add_argument(
        "--comparison-summary",
        action="append",
        type=Path,
        default=[],
        help="Optional token_dense/lm-eval summary CSV to include in the combined accuracy plot.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Wanda/ProxSparse/MaskLLM 2:4 pruning and lm-eval runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in [
        "generate-masks",
        "calibrate-row-scales",
        "materialize-models",
        "run-lm-eval",
        "run-all",
    ]:
        sub = subparsers.add_parser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_common_args(sub)
    subparsers.add_parser("self-test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "generate-masks":
        generate_masks(args)
    elif args.command == "calibrate-row-scales":
        for path in calibrate_row_scales(args):
            print(path)
    elif args.command == "materialize-models":
        materialize_models(args)
    elif args.command == "run-lm-eval":
        run_lm_eval(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
