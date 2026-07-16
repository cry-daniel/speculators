#!/usr/bin/env python3
"""Build grouped C4 covariances and covariance-aware gate/up 2:4 masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = (
    EVAL_ROOT / "data/c4_calibration/c4_calibration_512_seed42.jsonl"
)
DEFAULT_MASK_ROOT = (
    EVAL_ROOT
    / "data/c4_calibration/offline_24_masks/c4_512_seed42_bf16_max512"
)
DEFAULT_MODELS = {
    "qwen3_8b": EVAL_ROOT.parents[3] / "models/qwen3-8b",
    "llama3_1_8b": EVAL_ROOT.parents[3] / "models/llama-3.1-8b-instruct",
}
KEEP_OPTIONS = torch.tensor(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ],
    dtype=torch.long,
)
OPTION_BYTES = torch.tensor([0x3, 0x5, 0x9, 0x6, 0xA, 0xC], dtype=torch.uint8)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        overrides[label.strip()] = Path(path).expanduser().resolve()
    return overrides


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or row.get("prompt") or "").strip()
            if text:
                prompts.append(text)
            if 0 < limit <= len(prompts):
                break
    if not prompts:
        raise RuntimeError(f"no prompts found in {path}")
    return prompts


def pack_group_bytes(group_bytes: torch.Tensor) -> torch.Tensor:
    groups = int(group_bytes.shape[1])
    if groups % 2:
        group_bytes = torch.cat(
            [
                group_bytes,
                torch.zeros(
                    (int(group_bytes.shape[0]), 1),
                    device=group_bytes.device,
                    dtype=torch.uint8,
                ),
            ],
            dim=1,
        )
    return (
        group_bytes[:, 0::2] | (group_bytes[:, 1::2] << 4)
    ).cpu()


def unpack_group_bytes(
    mask_bytes: torch.Tensor,
    *,
    out_features: int,
    groups: int,
) -> torch.Tensor:
    if tuple(mask_bytes.shape) == (out_features, groups):
        return mask_bytes.to(dtype=torch.uint8)
    expected = (out_features, (groups + 1) // 2)
    if tuple(mask_bytes.shape) != expected:
        raise RuntimeError(
            f"mask shape {tuple(mask_bytes.shape)} does not match {expected}"
        )
    unpacked = torch.empty(
        (out_features, expected[1] * 2), dtype=torch.uint8
    )
    unpacked[:, 0::2] = mask_bytes & 0x0F
    unpacked[:, 1::2] = (mask_bytes >> 4) & 0x0F
    return unpacked[:, :groups]


def collect_group_covariances(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    target_leafs: set[str],
    max_seq_len: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            usable = (int(x.shape[1]) // 4) * 4
            grouped = x[:, :usable].view(int(x.shape[0]), usable // 4, 4)
            batch_sum = torch.einsum("tgi,tgj->gij", grouped, grouped)
            if name not in sums:
                sums[name] = torch.zeros_like(batch_sum)
                counts[name] = 0
            sums[name].add_(batch_sum)
            counts[name] += int(grouped.shape[0])

        return hook

    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and leaf in target_leafs:
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    try:
        with torch.no_grad():
            for index, prompt in enumerate(prompts, start=1):
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
                print(f"  calibration prompt {index}/{len(prompts)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache

    return {
        name: (value / max(counts[name], 1)).cpu()
        for name, value in sums.items()
    }, counts


def covariance_mask(
    weight: torch.Tensor,
    covariance: torch.Tensor,
    *,
    row_chunk: int,
) -> torch.Tensor:
    out_features, in_features = map(int, weight.shape)
    groups = in_features // 4
    covariance = covariance[:groups].to(device=weight.device, dtype=torch.float32)
    option_bytes = OPTION_BYTES.to(device=weight.device)
    group_bytes = torch.empty(
        (out_features, groups), device=weight.device, dtype=torch.uint8
    )
    all_indices = {0, 1, 2, 3}
    removed_options = [
        sorted(all_indices.difference(option.tolist())) for option in KEEP_OPTIONS
    ]
    for start in range(0, out_features, row_chunk):
        end = min(start + row_chunk, out_features)
        grouped = (
            weight[start:end, : groups * 4]
            .detach()
            .float()
            .view(end - start, groups, 4)
        )
        losses: list[torch.Tensor] = []
        for first, second in removed_options:
            w0 = grouped[:, :, first]
            w1 = grouped[:, :, second]
            losses.append(
                w0.square() * covariance[:, first, first]
                + 2.0 * w0 * w1 * covariance[:, first, second]
                + w1.square() * covariance[:, second, second]
            )
        best = torch.stack(losses, dim=-1).argmin(dim=-1)
        group_bytes[start:end] = option_bytes[best]
    return group_bytes


def build_cache(
    model: nn.Module,
    *,
    model_label: str,
    base_cache: dict[str, Any],
    covariances: dict[str, torch.Tensor],
    counts: dict[str, int],
    row_chunk: int,
    prompt_path: Path,
    prompt_count: int,
    max_seq_len: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    masks = dict(base_cache["masks"])
    changed_groups = 0
    total_gate_groups = 0
    per_module: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in {"gate_proj", "up_proj"}:
            continue
        covariance_name = name.rsplit(".", 1)[0] + ".gate_proj"
        covariance = covariances.get(covariance_name)
        if covariance is None:
            raise RuntimeError(f"missing grouped covariance for {name}")
        group_bytes = covariance_mask(
            module.weight,
            covariance,
            row_chunk=row_chunk,
        )
        groups = int(module.weight.shape[1]) // 4
        old_group_bytes = unpack_group_bytes(
            masks[name],
            out_features=int(module.weight.shape[0]),
            groups=groups,
        ).to(device=group_bytes.device)
        changed = int(group_bytes.ne(old_group_bytes).sum().item())
        total = int(group_bytes.numel())
        changed_groups += changed
        total_gate_groups += total
        masks[name] = pack_group_bytes(group_bytes)
        per_module.append(
            {
                "module": name,
                "covariance_source": covariance_name,
                "calibration_tokens": counts[covariance_name],
                "changed_groups_vs_wanda": changed,
                "total_groups": total,
                "changed_fraction_vs_wanda": changed / total if total else 0.0,
            }
        )
        print(
            f"  {name}: changed {changed}/{total} groups vs Wanda",
            flush=True,
        )

    metadata = {
        **dict(base_cache.get("metadata", {})),
        "method": "covwanda",
        "source": "speclink_grouped_covariance_gate_mask",
        "base_method": "wanda",
        "calibration_prompts": str(prompt_path.resolve()),
        "calibration_num_examples": prompt_count,
        "calibration_max_seq_len": max_seq_len,
    }
    stats = {
        "method": "covwanda",
        "model_label": model_label,
        "changed_gate_groups_vs_wanda": changed_groups,
        "total_gate_groups": total_gate_groups,
        "changed_gate_fraction_vs_wanda": (
            changed_groups / total_gate_groups if total_gate_groups else 0.0
        ),
        "per_module": per_module,
    }
    return {
        "metadata": metadata,
        "stats": stats,
        "masks": masks,
        "row_scales": {},
    }, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare grouped-covariance gate/up 2:4 mask caches."
    )
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--covariance-leafs",
        default="q_proj,gate_proj,down_proj",
        help=(
            "Linear inputs whose grouped 4x4 covariance is cached. q_proj is "
            "shared by fused QKV and gate_proj by fused gate/up."
        ),
    )
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument(
        "--covariance-output-name",
        default="gate_group_covariances.pt",
        help="Filename written below each model's mask-cache directory.",
    )
    parser.add_argument(
        "--covariance-only",
        action="store_true",
        help="Write grouped covariance without replacing the covwanda mask cache.",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.model_id)
    prompts = load_prompts(args.prompts, args.num_examples)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    covariance_leafs = set(parse_csv(args.covariance_leafs))
    if not covariance_leafs:
        raise ValueError("--covariance-leafs must not be empty")
    device = torch.device("cuda")
    for model_label in parse_csv(args.models):
        model_path = overrides.get(model_label, DEFAULT_MODELS.get(model_label))
        if model_path is None:
            raise ValueError(f"no model path configured for {model_label}")
        model_path = Path(model_path).resolve()
        print(f"[{model_label}] loading {model_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device)
        model.eval()
        covariances, counts = collect_group_covariances(
            model,
            tokenizer,
            prompts,
            target_leafs=covariance_leafs,
            max_seq_len=args.max_seq_len,
            device=device,
        )
        model_root = args.mask_root / model_label
        covariance_path = model_root / args.covariance_output_name
        torch.save(
            {
                "metadata": {
                    "model_label": model_label,
                    "calibration_prompts": str(args.prompts.resolve()),
                    "calibration_num_examples": len(prompts),
                    "calibration_max_seq_len": args.max_seq_len,
                    "dtype": args.dtype,
                    "covariance_leafs": sorted(covariance_leafs),
                },
                "covariances": covariances,
                "counts": counts,
            },
            covariance_path,
        )
        print(covariance_path, flush=True)
        if args.covariance_only:
            del model
            torch.cuda.empty_cache()
            continue
        base_path = model_root / "wanda.pt"
        base_cache = torch.load(base_path, map_location="cpu")
        cache, stats = build_cache(
            model,
            model_label=model_label,
            base_cache=base_cache,
            covariances=covariances,
            counts=counts,
            row_chunk=args.row_chunk,
            prompt_path=args.prompts,
            prompt_count=len(prompts),
            max_seq_len=args.max_seq_len,
        )
        output_path = model_root / "covwanda.pt"
        torch.save(cache, output_path)
        (model_root / "covwanda.json").write_text(
            json.dumps({"metadata": cache["metadata"], "stats": stats}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(output_path, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
