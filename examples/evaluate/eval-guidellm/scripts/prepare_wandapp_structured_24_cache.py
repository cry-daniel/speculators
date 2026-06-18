#!/usr/bin/env python3
"""Prepare Wanda++-style 2:4 mask caches from the fixed C4 calibration set.

This does not mutate or save model checkpoints. It stores compact mask caches
that the vLLM TLM-only hook can apply at model-load time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "data" / "c4_calibration" / "wandapp_masks"

sys.path.insert(0, str(SCRIPT_DIR))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_PROMPTS,
    LAYER_SENSITIVITY_DEFAULT_MODELS,
    QUALITY_MASK_TARGETS,
    dtype_from_arg,
    ensure_quality_dependencies,
    iter_target_linear_modules,
    load_calibration_prompt_file,
    load_model_and_tokenizer,
    parse_csv_list,
    parse_model_id_overrides,
    set_seed,
    write_json,
)


MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pack_nibbles(mask_bytes: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit group masks into one uint8 column."""

    if mask_bytes.dtype != torch.uint8:
        mask_bytes = mask_bytes.to(dtype=torch.uint8)
    if mask_bytes.shape[1] % 2:
        pad = torch.zeros((mask_bytes.shape[0], 1), dtype=torch.uint8)
        mask_bytes = torch.cat([mask_bytes, pad], dim=1)
    low = mask_bytes[:, 0::2] & 0x0F
    high = (mask_bytes[:, 1::2] & 0x0F) << 4
    return low | high


def mask_bytes_from_keep(keep: torch.Tensor) -> torch.Tensor:
    bits = MASK_BITS.to(device=keep.device)
    return (keep.to(dtype=torch.uint8) * bits.view(1, 1, 4)).sum(dim=-1).to(dtype=torch.uint8)


def collect_module_inputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    target_modules: list[tuple[str, Any, int, int, int]],
    max_seq_len: int,
    batch_size: int,
    device: str,
    max_tokens_per_module: int,
    storage_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    stores: dict[str, list[torch.Tensor]] = {name: [] for name, *_ in target_modules}
    counts: dict[str, int] = {name: 0 for name, *_ in target_modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            remaining = max_tokens_per_module - counts[name]
            if remaining <= 0 or not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach()
            if x.ndim == 1:
                x = x.reshape(1, -1)
            elif x.ndim > 2:
                x = x.reshape(-1, x.shape[-1])
            if x.ndim != 2 or x.shape[0] == 0:
                return
            take = min(remaining, int(x.shape[0]))
            if take < int(x.shape[0]):
                indices = torch.linspace(
                    0,
                    int(x.shape[0]) - 1,
                    steps=take,
                    device=x.device,
                ).long()
                x = x.index_select(0, indices)
            else:
                x = x[:take]
            stores[name].append(x.to(device="cpu", dtype=storage_dtype))
            counts[name] += take

        return hook

    for name, module, *_ in target_modules:
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        for start in range(0, len(prompts), batch_size):
            if all(value >= max_tokens_per_module for value in counts.values()):
                break
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    merged: dict[str, torch.Tensor] = {}
    for name, chunks in stores.items():
        if chunks:
            merged[name] = torch.cat(chunks, dim=0)[:max_tokens_per_module].contiguous()
    return merged, counts


def _score_chunk(
    *,
    weight_chunk: torch.Tensor,
    inputs: torch.Tensor,
    rms: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Local output sensitivity: rows whose outputs have larger normalized
    # response gradient get a larger multiplier inside their 4-wide groups.
    outputs = inputs.matmul(weight_chunk.t())
    denom = outputs.square().sum(dim=0).sqrt().clamp_min(1e-8)
    grad_proxy = (outputs / denom.view(1, -1)).t().matmul(inputs)
    grad_proxy = grad_proxy.abs()
    grad_norm = grad_proxy / grad_proxy.mean(dim=1, keepdim=True).clamp_min(1e-8)
    grad_norm = grad_norm.clamp(max=10.0)
    score = weight_chunk.abs() * rms.view(1, -1) * (1.0 + alpha * grad_norm)
    return score, outputs


def build_mask_for_module(
    module_name: str,
    module: Any,
    inputs_cpu: torch.Tensor,
    *,
    method: str,
    score_device: str,
    row_chunk: int,
    alpha: float,
    ro_scale_clamp_min: float,
    ro_scale_clamp_max: float,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    weight = module.weight.detach()
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    if usable_in <= 0:
        raise RuntimeError(f"{module_name} has no 4-aligned input dimension")
    inputs = inputs_cpu[:, :usable_in].to(device=score_device, dtype=torch.float32)
    rms = inputs.square().mean(dim=0).sqrt().clamp_min(1e-8)
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    row_chunk = max(1, int(row_chunk))

    for start in range(0, out_features, row_chunk):
        end = min(out_features, start + row_chunk)
        weight_chunk = weight[start:end, :usable_in].to(device=score_device, dtype=torch.float32)
        score, dense_outputs = _score_chunk(
            weight_chunk=weight_chunk,
            inputs=inputs,
            rms=rms,
            alpha=alpha,
        )
        score_view = score.view(end - start, groups, 4)
        keep_idx = score_view.topk(k=2, dim=-1, largest=True, sorted=False).indices
        keep = torch.zeros_like(score_view, dtype=torch.bool)
        keep.scatter_(-1, keep_idx, True)
        mask_bytes = mask_bytes_from_keep(keep).cpu()
        packed_chunks.append(pack_nibbles(mask_bytes))

        if method == "wandapp_ro":
            keep_flat = keep.reshape(end - start, usable_in)
            sparse_weight = weight_chunk * keep_flat.to(dtype=weight_chunk.dtype)
            sparse_outputs = inputs.matmul(sparse_weight.t())
            numerator = (sparse_outputs * dense_outputs).sum(dim=0)
            denominator = sparse_outputs.square().sum(dim=0).clamp_min(1e-8)
            scale = numerator / denominator
            scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
            scale = scale.clamp(ro_scale_clamp_min, ro_scale_clamp_max)
            scale_chunks.append(scale.to(device="cpu", dtype=torch.float16))

        del weight_chunk, score, dense_outputs, keep, keep_idx
        if score_device == "cuda":
            torch.cuda.empty_cache()

    mask_packed = torch.cat(packed_chunks, dim=0).contiguous()
    row_scale = torch.cat(scale_chunks, dim=0).contiguous() if scale_chunks else None
    stats = {
        "module": module_name,
        "shape": [out_features, in_features],
        "usable_in": usable_in,
        "groups": groups,
        "packed_mask_shape": list(mask_packed.shape),
        "calibration_tokens": int(inputs_cpu.shape[0]),
        "method": method,
        "alpha": alpha,
        "row_scale_mean": float(row_scale.float().mean().item()) if row_scale is not None else None,
        "row_scale_std": float(row_scale.float().std().item()) if row_scale is not None and row_scale.numel() > 1 else None,
    }
    return mask_packed, row_scale, stats


def build_cache_for_model(args: argparse.Namespace, model_label: str, model_id: str) -> list[Path]:
    dtype = dtype_from_arg(args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype,
        args.device,
        args.trust_remote_code,
        args.local_files_only,
    )
    try:
        target_modules = iter_target_linear_modules(
            model,
            QUALITY_MASK_TARGETS["all"],
            skip_lm_head=True,
            skip_embeddings=True,
        )
        print(
            f"[INFO] {model_label}: collecting C4 inputs for {len(target_modules)} modules",
            flush=True,
        )
        prompts = load_calibration_prompt_file(args.calibration_prompts, args.num_prompts, args.seed)
        inputs_by_module, token_counts = collect_module_inputs(
            model,
            tokenizer,
            prompts,
            target_modules=target_modules,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
            max_tokens_per_module=args.max_tokens_per_module,
            storage_dtype=torch.float16,
        )

        written: list[Path] = []
        module_lookup = {name: module for name, module, *_ in target_modules}
        for method in parse_csv_list(args.methods):
            if method not in {"wandapp_rgs", "wandapp_ro"}:
                raise ValueError(f"unsupported method: {method}")
            masks: dict[str, torch.Tensor] = {}
            row_scales: dict[str, torch.Tensor] = {}
            per_module: list[dict[str, Any]] = []
            for idx, (name, _module, *_rest) in enumerate(target_modules, start=1):
                module_inputs = inputs_by_module.get(name)
                if module_inputs is None or module_inputs.numel() == 0:
                    raise RuntimeError(f"missing collected inputs for {model_label}/{name}")
                print(
                    f"[INFO] {model_label}/{method}: {idx}/{len(target_modules)} {name}",
                    flush=True,
                )
                mask, row_scale, stats = build_mask_for_module(
                    name,
                    module_lookup[name],
                    module_inputs,
                    method=method,
                    score_device=args.score_device,
                    row_chunk=args.score_row_chunk,
                    alpha=args.alpha,
                    ro_scale_clamp_min=args.ro_scale_clamp_min,
                    ro_scale_clamp_max=args.ro_scale_clamp_max,
                )
                masks[name] = mask
                if row_scale is not None:
                    row_scales[name] = row_scale
                per_module.append(stats)

            output_path = args.output_root / f"{model_label}_{method}.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": {
                        "format": "speclink_structured_24_mask_cache_v1",
                        "mask_packing": "two_4bit_group_masks_per_uint8",
                        "method": method,
                        "model_label": model_label,
                        "model_id": model_id,
                        "created_at": timestamp(),
                        "calibration_prompts": str(args.calibration_prompts.resolve()),
                        "num_prompts": len(prompts),
                        "max_seq_len": args.max_seq_len,
                        "max_tokens_per_module": args.max_tokens_per_module,
                        "alpha": args.alpha,
                        "score_device": args.score_device,
                        "score_row_chunk": args.score_row_chunk,
                        "ro_scale_clamp": [args.ro_scale_clamp_min, args.ro_scale_clamp_max],
                        "note": (
                            "Wanda++-style RGS-lite mask; wandapp_ro also stores "
                            "output-row least-squares scales. Original checkpoint "
                            "weights are not modified."
                        ),
                    },
                    "token_counts": token_counts,
                    "masks": masks,
                    "row_scales": row_scales,
                    "per_module": per_module,
                },
                output_path,
            )
            written.append(output_path)
            write_json(
                output_path.with_suffix(".json"),
                {
                    "path": str(output_path.resolve()),
                    "method": method,
                    "model_label": model_label,
                    "model_id": model_id,
                    "modules": len(per_module),
                    "num_prompts": len(prompts),
                    "max_tokens_per_module": args.max_tokens_per_module,
                    "created_at": timestamp(),
                },
            )
        return written
    finally:
        del model
        del tokenizer
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    ensure_quality_dependencies()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.score_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA score_device requested but torch.cuda.is_available() is false")
    args.output_root.mkdir(parents=True, exist_ok=True)
    model_ids = dict(LAYER_SENSITIVITY_DEFAULT_MODELS)
    model_ids.update(parse_model_id_overrides(args.model_id))
    selected_models = parse_csv_list(args.models)
    write_json(
        args.output_root / "run_config.json",
        {
            "argv": sys.argv,
            "models": selected_models,
            "model_ids": model_ids,
            "methods": parse_csv_list(args.methods),
            "calibration_prompts": str(args.calibration_prompts.resolve()),
            "num_prompts": args.num_prompts,
            "max_seq_len": args.max_seq_len,
            "max_tokens_per_module": args.max_tokens_per_module,
            "dtype": args.dtype,
            "device": args.device,
            "score_device": args.score_device,
            "created_at": timestamp(),
        },
    )

    manifest: dict[str, Any] = {"created_at": timestamp(), "caches": []}
    for model_label in selected_models:
        model_id = model_ids.get(model_label)
        if not model_id:
            raise ValueError(f"unknown model label: {model_label}")
        for path in build_cache_for_model(args, model_label, model_id):
            manifest["caches"].append(str(path.resolve()))
    write_json(args.output_root / "manifest.json", manifest)
    print(args.output_root.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare compact Wanda++-style 2:4 mask caches from C4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--model-id", action="append", default=[], help="Override base model as LABEL=PATH_OR_ID.")
    parser.add_argument("--methods", default="wandapp_rgs,wandapp_ro")
    parser.add_argument("--calibration-prompts", type=Path, default=DEFAULT_C4_CALIBRATION_PROMPTS)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-module", type=int, default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--score-row-chunk", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--ro-scale-clamp-min", type=float, default=0.25)
    parser.add_argument("--ro-scale-clamp-max", type=float, default=4.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
