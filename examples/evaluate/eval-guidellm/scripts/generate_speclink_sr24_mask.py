#!/usr/bin/env python3
"""Export activation-aware 2:4 masks for the SpecLink SR24 vLLM prototype.

The output is a torch `.pt` cache accepted by `SPECLINK_SR24_MASK_PATH`.
It stores one packed uint8 mask per unfused HF Linear module. Local vLLM fuses
q/k/v and gate/up at load time and reconstructs those fused masks from these
module names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import residual_24_feasibility as r24  # noqa: E402


MASK_BITS = None
TARGET_LEAFS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def pack_keep_mask(keep: Any) -> Any:
    global MASK_BITS
    torch = r24.torch
    if MASK_BITS is None:
        MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8, device=keep.device)
    group_bytes = (
        keep.to(torch.uint8) * MASK_BITS.view(1, 1, 4)
    ).sum(dim=-1).to(torch.uint8)
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


def compute_keep_mask(weight: Any, activation_scale: Any | None) -> tuple[Any, str]:
    torch = r24.torch
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    grouped = weight[:, :usable_in].detach().abs().float().view(
        out_features, usable_in // 4, 4
    )
    method = "magnitude"
    if activation_scale is not None and int(activation_scale.numel()) >= usable_in:
        scale = activation_scale[:usable_in].to(device=weight.device, dtype=grouped.dtype)
        grouped = grouped * scale.view(1, usable_in // 4, 4)
        method = "activation_aware"
    keep_idx = grouped.topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return keep, method


def compute_gate_up_pair_keep_masks(
    gate_weight: Any,
    up_weight: Any,
    activation_scale: Any | None,
    score_mode: str,
) -> tuple[Any, Any, str]:
    """Use one shared 2:4 input pattern for each gate/up channel pair."""
    torch = r24.torch
    if tuple(gate_weight.shape) != tuple(up_weight.shape):
        raise RuntimeError(
            "gate/up pair-aware masks require matching shapes, got "
            f"{tuple(gate_weight.shape)} and {tuple(up_weight.shape)}"
        )
    out_features = int(gate_weight.shape[0])
    in_features = int(gate_weight.shape[1])
    usable_in = (in_features // 4) * 4
    gate_score = gate_weight[:, :usable_in].detach().abs().float().view(
        out_features, usable_in // 4, 4
    )
    up_score = up_weight[:, :usable_in].detach().abs().float().view(
        out_features, usable_in // 4, 4
    )
    if score_mode == "sum":
        score = gate_score + up_score
    elif score_mode == "min":
        score = torch.minimum(gate_score, up_score)
    elif score_mode == "max":
        score = torch.maximum(gate_score, up_score)
    elif score_mode == "product":
        score = gate_score * up_score
    elif score_mode == "balanced_sum":
        score = gate_score + up_score + 2.0 * torch.minimum(gate_score, up_score)
    else:
        raise RuntimeError(f"unsupported gate/up pair score mode: {score_mode}")
    method = f"magnitude_pair_gate_up_{score_mode}"
    if activation_scale is not None and int(activation_scale.numel()) >= usable_in:
        scale = activation_scale[:usable_in].to(
            device=gate_weight.device, dtype=score.dtype
        )
        score = score * scale.view(1, usable_in // 4, 4)
        method = f"activation_aware_pair_gate_up_{score_mode}"
    keep_idx = score.topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return keep, keep.clone(), method


def module_is_target(name: str, module: Any) -> bool:
    if not isinstance(module, r24.nn.Linear):
        return False
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in TARGET_LEAFS:
        return False
    lowered = name.lower()
    return not (
        name == "lm_head"
        or name.endswith(".lm_head")
        or "embed_tokens" in lowered
        or "embedding" in lowered
    )


def run(args: argparse.Namespace) -> None:
    r24.ensure_quality_dependencies()
    torch = r24.torch
    r24.set_seed(args.seed)
    dtype = r24.dtype_from_arg(args.dtype)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model_ids = dict(r24.LAYER_SENSITIVITY_DEFAULT_MODELS)
    model_ids.update(r24.parse_model_id_overrides(args.model_id))
    model_id = args.model_path or model_ids.get(args.model_label)
    if not model_id:
        raise ValueError(f"unknown model label: {args.model_label}")

    activation_scales, calibration_stats = r24.load_activation_cache(
        args.calibration_cache_root,
        args.model_label,
        required_modules=r24.QUALITY_MASK_TARGETS["all"],
    )
    model, tokenizer = r24.load_model_and_tokenizer(
        model_id,
        dtype,
        args.device,
        args.trust_remote_code,
        args.local_files_only,
    )
    del tokenizer

    masks: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        handled_pair_modules: set[str] = set()
        named_modules = dict(model.named_modules())
        if args.gate_up_pair_aware:
            for gate_name, gate_module in named_modules.items():
                if gate_name.rsplit(".", 1)[-1] != "gate_proj":
                    continue
                prefix = gate_name.rsplit(".", 1)[0] if "." in gate_name else ""
                up_name = f"{prefix}.up_proj" if prefix else "up_proj"
                up_module = named_modules.get(up_name)
                if not module_is_target(gate_name, gate_module):
                    continue
                if up_module is None or not module_is_target(up_name, up_module):
                    continue
                gate_weight = gate_module.weight
                up_weight = up_module.weight
                usable_in = (int(gate_weight.shape[1]) // 4) * 4
                if usable_in <= 0:
                    continue
                scale = activation_scales.get(gate_name)
                if scale is None:
                    scale = activation_scales.get(up_name)
                gate_keep, up_keep, method = compute_gate_up_pair_keep_masks(
                    gate_weight, up_weight, scale, args.gate_up_pair_score
                )
                if not bool((gate_keep.sum(dim=-1) == 2).all().item()):
                    raise RuntimeError(f"generated non-2:4 mask for {gate_name}")
                if not bool((up_keep.sum(dim=-1) == 2).all().item()):
                    raise RuntimeError(f"generated non-2:4 mask for {up_name}")
                masks[gate_name] = pack_keep_mask(gate_keep)
                masks[up_name] = pack_keep_mask(up_keep)
                handled_pair_modules.update({gate_name, up_name})
                for module_name, module, keep in (
                    (gate_name, gate_module, gate_keep),
                    (up_name, up_module, up_keep),
                ):
                    weight = module.weight
                    rows.append(
                        {
                            "module": module_name,
                            "leaf": module_name.rsplit(".", 1)[-1],
                            "shape": [int(weight.shape[0]), int(weight.shape[1])],
                            "usable_in": usable_in,
                            "mask_method": method,
                            "packed_shape": list(masks[module_name].shape),
                        }
                    )
        for name, module in model.named_modules():
            if not module_is_target(name, module):
                continue
            if name in handled_pair_modules:
                continue
            weight = module.weight
            usable_in = (int(weight.shape[1]) // 4) * 4
            if usable_in <= 0:
                continue
            scale = activation_scales.get(name)
            keep, method = compute_keep_mask(weight, scale)
            if not bool((keep.sum(dim=-1) == 2).all().item()):
                raise RuntimeError(f"generated non-2:4 mask for {name}")
            masks[name] = pack_keep_mask(keep)
            rows.append(
                {
                    "module": name,
                    "leaf": name.rsplit(".", 1)[-1],
                    "shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "usable_in": usable_in,
                    "mask_method": method,
                    "packed_shape": list(masks[name].shape),
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = args.output_path or (
        EVAL_ROOT
        / "data"
        / "c4_calibration"
        / "sr24_masks"
        / f"{args.model_label}_activation_aware_24.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "masks": masks,
        "metadata": {
            "method": (
                f"activation_aware_pair_gate_up_{args.gate_up_pair_score}"
                if args.gate_up_pair_aware
                else "activation_aware"
            ),
            "model_label": args.model_label,
            "model_id": model_id,
            "dtype": args.dtype,
            "target_modules": list(TARGET_LEAFS),
            "gate_up_pair_aware": bool(args.gate_up_pair_aware),
            "gate_up_pair_score": args.gate_up_pair_score,
            "calibration": calibration_stats,
            "module_count": len(masks),
            "created_at": r24.timestamp(),
            "argv": sys.argv,
        },
        "per_module": rows,
    }
    torch.save(cache, output_path)
    (output_path.with_suffix(".json")).write_text(
        json.dumps(cache["metadata"] | {"per_module": rows}, indent=2),
        encoding="utf-8",
    )
    print(output_path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate activation-aware 2:4 mask cache for SR24.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-label", default="llama3_1_8b")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--calibration-cache-root",
        type=Path,
        default=r24.DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--gate-up-pair-aware",
        action="store_true",
        help=(
            "For matching gate_proj/up_proj modules, select one shared 2:4 "
            "input pattern per channel using combined gate+up importance. "
            "This targets fused gate_up_proj serving paths."
        ),
    )
    parser.add_argument(
        "--gate-up-pair-score",
        choices=["sum", "min", "max", "product", "balanced_sum"],
        default="sum",
        help=(
            "Importance score for --gate-up-pair-aware. sum preserves the "
            "initial pair-aware behavior; min/product favor input positions "
            "important to both gate and up rows."
        ),
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
