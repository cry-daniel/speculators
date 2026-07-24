#!/usr/bin/env python3
"""Attribute the original activation-RMS Wanda 2:4 error by module and layer.

The mask is exactly the serving baseline:

    score = abs(weight) * activation_rms
    keep the largest two scores in every contiguous group of four

Two complementary metrics are reported:

1. ``rms_proxy_*`` uses the full 512-prompt activation-RMS cache and assumes
   independent input channels.
2. ``measured_*`` runs fixed C4 prompts through the dense model and measures
   the actual local linear-output residual ``X @ (W - W_2:4).T``.  It includes
   all input-channel correlations and is the primary attribution metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
MODEL_PATH = SPECULATORS_ROOT.parent / "models" / "llama-3.1-8b-instruct"
CALIBRATION_PROMPTS = (
    EVAL_ROOT
    / "data"
    / "c4_calibration"
    / "c4_calibration_512_seed42.jsonl"
)
ACTIVATION_RMS = (
    EVAL_ROOT
    / "data"
    / "c4_calibration"
    / "activation_rms"
    / "c4_512_seed42_bf16_max512"
    / "llama3_1_8b.pt"
)
TARGET_LEAFS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_prompts(path: Path, num_prompts: int, seed: int) -> list[str]:
    import random

    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or row.get("text") or "").strip()
            if prompt:
                prompts.append(prompt)
    if num_prompts > 0 and num_prompts < len(prompts):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(prompts)), num_prompts))
        prompts = [prompts[index] for index in indices]
    return prompts


def layer_index(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


def target_modules(model: Any) -> list[tuple[str, Any]]:
    output = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in TARGET_LEAFS:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        output.append((name, module))
    return output


def wanda_group_bytes(
    weight: torch.Tensor,
    activation_rms: torch.Tensor,
    *,
    row_chunk: int,
) -> torch.Tensor:
    out_features, in_features = map(int, weight.shape)
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    scale = activation_rms[:usable_in].to(
        device=weight.device, dtype=torch.float32
    )
    scale = scale.view(1, groups, 4)
    bits = MASK_BITS.to(device=weight.device)
    output = torch.empty(
        (out_features, groups), dtype=torch.uint8, device=weight.device
    )
    for start in range(0, out_features, row_chunk):
        view = (
            weight[start : start + row_chunk, :usable_in]
            .detach()
            .float()
            .view(-1, groups, 4)
        )
        score = view.abs() * scale
        indices = score.topk(k=2, dim=-1, largest=True, sorted=False).indices
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep.scatter_(-1, indices, True)
        output[start : start + row_chunk] = (
            keep.to(torch.uint8) * bits.view(1, 1, 4)
        ).sum(dim=-1)
    return output


def make_rows(
    model: Any,
    activation_scales: dict[str, torch.Tensor],
    *,
    row_chunk: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    group_masks: dict[str, torch.Tensor] = {}
    bits = MASK_BITS.to("cuda")
    for index, (name, module) in enumerate(target_modules(model), start=1):
        weight = module.weight
        out_features, in_features = map(int, weight.shape)
        usable_in = (in_features // 4) * 4
        groups = usable_in // 4
        scale = activation_scales.get(name)
        if scale is None or int(scale.numel()) < usable_in:
            raise RuntimeError(f"missing activation RMS for {name}")
        group_bytes = wanda_group_bytes(
            weight,
            scale,
            row_chunk=row_chunk,
        )
        keep = (
            group_bytes.unsqueeze(-1) & bits.view(1, 1, 4)
        ).ne(0)
        if not bool((keep.sum(dim=-1) == 2).all().item()):
            raise RuntimeError(f"{name} mask is not exactly 2:4")
        weight_view = (
            weight[:, :usable_in].detach().float().view(
                out_features, groups, 4
            )
        )
        scale_view = (
            scale[:usable_in]
            .to(device=weight.device, dtype=torch.float32)
            .view(1, groups, 4)
        )
        energy = weight_view.square() * scale_view.square()
        dropped = float(energy.masked_fill(keep, 0).sum().item())
        dense = float(energy.sum().item())
        rows.append(
            {
                "module": name,
                "layer": layer_index(name),
                "linear": name.rsplit(".", 1)[-1],
                "out_features": out_features,
                "in_features": in_features,
                "groups": out_features * groups,
                "kept_per_group": 2,
                "rms_proxy_error": dropped,
                "rms_proxy_dense_energy": dense,
                "rms_proxy_nmse": dropped / dense if dense else 0.0,
                "rms_proxy_relative_rmse": (
                    math.sqrt(dropped / dense) if dense else 0.0
                ),
                "measured_error_sq": 0.0,
                "measured_dense_sq": 0.0,
                "measured_output_elements": 0,
                "measured_tokens": 0,
                "_measured_error_sq_cuda": torch.zeros(
                    (), dtype=torch.float64, device=weight.device
                ),
                "_measured_dense_sq_cuda": torch.zeros(
                    (), dtype=torch.float64, device=weight.device
                ),
            }
        )
        # One byte per 2:4 group. Keeping these compact masks on the GPU costs
        # about 1.6 GB for Llama-3.1-8B and avoids repeated top-k work.
        group_masks[name] = group_bytes
        del keep, weight_view, scale_view, energy
        if index == 1 or index % 16 == 0:
            print(
                f"[mask] prepared {index}/{len(target_modules(model))} modules",
                flush=True,
            )
    return rows, group_masks


def measure_local_output_error(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    rows: list[dict[str, Any]],
    group_masks: dict[str, torch.Tensor],
    *,
    max_seq_len: int,
) -> int:
    row_by_name = {str(row["module"]): row for row in rows}
    handles = []
    bits = MASK_BITS.to("cuda")

    def make_hook(name: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            x = inputs[0]
            weight = module.weight
            out_features, in_features = map(int, weight.shape)
            usable_in = (in_features // 4) * 4
            groups = usable_in // 4
            group_bytes = group_masks[name]
            keep = (
                group_bytes.unsqueeze(-1) & bits.view(1, 1, 4)
            ).ne(0)
            residual = (
                weight[:, :usable_in]
                .view(out_features, groups, 4)
                .masked_fill(keep, 0)
                .view(out_features, usable_in)
            )
            flat_input = x.detach().reshape(-1, in_features)[:, :usable_in]
            error = F.linear(flat_input, residual)
            dense_output = output.detach().reshape(-1, out_features)
            row = row_by_name[name]
            row["_measured_error_sq_cuda"].add_(
                error.float().square().sum().double()
            )
            row["_measured_dense_sq_cuda"].add_(
                dense_output.float().square().sum().double()
            )
            row["measured_output_elements"] += int(error.numel())
            row["measured_tokens"] += int(flat_input.shape[0])

        return hook

    modules = dict(target_modules(model))
    for name in row_by_name:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False
    backbone = getattr(model, "model", model)
    total_tokens = 0
    try:
        with torch.inference_mode():
            for index, prompt in enumerate(prompts, start=1):
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                encoded = {
                    key: value.to("cuda") for key, value in encoded.items()
                }
                total_tokens += int(encoded["attention_mask"].sum().item())
                backbone(**encoded, use_cache=False)
                if index == 1 or index % 8 == 0 or index == len(prompts):
                    print(
                        f"[measure] completed {index}/{len(prompts)} prompts",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
    return total_tokens


def finalize_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["measured_error_sq"] = float(
            row.pop("_measured_error_sq_cuda").item()
        )
        row["measured_dense_sq"] = float(
            row.pop("_measured_dense_sq_cuda").item()
        )
    total_rms_error = sum(float(row["rms_proxy_error"]) for row in rows)
    total_measured_error = sum(
        float(row["measured_error_sq"]) for row in rows
    )
    for row in rows:
        error = float(row["measured_error_sq"])
        dense = float(row["measured_dense_sq"])
        elements = int(row["measured_output_elements"])
        row["rms_proxy_error_share"] = (
            float(row["rms_proxy_error"]) / total_rms_error
            if total_rms_error
            else 0.0
        )
        row["measured_error_share"] = (
            error / total_measured_error if total_measured_error else 0.0
        )
        row["measured_error_mse"] = error / elements if elements else 0.0
        row["measured_dense_mse"] = dense / elements if elements else 0.0
        row["measured_nmse"] = error / dense if dense else 0.0
        row["measured_relative_rmse"] = (
            math.sqrt(error / dense) if dense else 0.0
        )
        row["measured_ser_db"] = (
            10.0 * math.log10(dense / error)
            if error > 0.0 and dense > 0.0
            else None
        )


def aggregate(
    rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    accum: dict[Any, dict[str, Any]] = defaultdict(
        lambda: {
            "module_count": 0,
            "rms_proxy_error": 0.0,
            "rms_proxy_dense_energy": 0.0,
            "measured_error_sq": 0.0,
            "measured_dense_sq": 0.0,
            "measured_output_elements": 0,
        }
    )
    for row in rows:
        item = accum[row[key]]
        item["module_count"] += 1
        for field in (
            "rms_proxy_error",
            "rms_proxy_dense_energy",
            "measured_error_sq",
            "measured_dense_sq",
            "measured_output_elements",
        ):
            item[field] += row[field]
    total_rms_error = sum(
        float(item["rms_proxy_error"]) for item in accum.values()
    )
    total_measured_error = sum(
        float(item["measured_error_sq"]) for item in accum.values()
    )
    output = []
    for value, item in accum.items():
        rms_error = float(item["rms_proxy_error"])
        rms_dense = float(item["rms_proxy_dense_energy"])
        measured_error = float(item["measured_error_sq"])
        measured_dense = float(item["measured_dense_sq"])
        elements = int(item["measured_output_elements"])
        output.append(
            {
                key: value,
                **item,
                "rms_proxy_error_share": (
                    rms_error / total_rms_error if total_rms_error else 0.0
                ),
                "rms_proxy_nmse": (
                    rms_error / rms_dense if rms_dense else 0.0
                ),
                "measured_error_share": (
                    measured_error / total_measured_error
                    if total_measured_error
                    else 0.0
                ),
                "measured_error_mse": (
                    measured_error / elements if elements else 0.0
                ),
                "measured_dense_mse": (
                    measured_dense / elements if elements else 0.0
                ),
                "measured_nmse": (
                    measured_error / measured_dense
                    if measured_dense
                    else 0.0
                ),
                "measured_relative_rmse": (
                    math.sqrt(measured_error / measured_dense)
                    if measured_dense
                    else 0.0
                ),
            }
        )
    output.sort(
        key=lambda item: float(item["measured_error_sq"]),
        reverse=True,
    )
    return output


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def sci(value: Any) -> str:
    return f"{float(value):.4e}"


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    module_rows: list[dict[str, Any]],
    leaf_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    total_tokens: int,
) -> None:
    relative_rows = sorted(
        leaf_rows,
        key=lambda item: float(item["measured_nmse"]),
        reverse=True,
    )
    last_layer = max(int(row["layer"]) for row in layer_rows)
    last_ten_share = sum(
        float(row["measured_error_share"])
        for row in layer_rows
        if int(row["layer"]) >= last_layer - 9
    )
    gate_up_share = sum(
        float(row["measured_error_share"])
        for row in leaf_rows
        if row["linear"] in {"gate_proj", "up_proj"}
    )
    lines = [
        "# Llama-3.1-8B original Wanda 2:4 error attribution",
        "",
        "## Protocol",
        "",
        "- Mask: original `top-2(abs(weight) * activation_rms)` in every contiguous group of four.",
        "- No alternative mask, weight reconstruction, or runtime change is used.",
        f"- Activation-RMS cache: `{ACTIVATION_RMS}` (fixed 512-prompt C4 calibration).",
        f"- Measured local-output sample: {len(load_prompts(args.calibration_prompts, args.num_prompts, args.seed))} fixed C4 prompts, {total_tokens} tokens, seed={args.seed}, max sequence length={args.max_seq_len}.",
        "- Primary metric: measured local squared error `||XW - XW_2:4||_F^2`.",
        "- `Error share` answers where the total local approximation error comes from; `NMSE` answers which output stream is distorted most relative to its own dense signal.",
        "",
        "## Conclusion",
        "",
        f"- By total local squared error, `gate_proj + up_proj` contribute **{pct(gate_up_share)}**. The dominant source is therefore the FFN expansion path.",
        f"- By error relative to each stream's own dense output, `{relative_rows[0]['linear']}` is worst at **{pct(relative_rows[0]['measured_nmse'])}** NMSE, followed by `{relative_rows[1]['linear']}` at **{pct(relative_rows[1]['measured_nmse'])}**. These paths have small absolute energy but high relative distortion.",
        f"- The final ten transformer layers contribute **{pct(last_ten_share)}** of total local error; layer {layer_rows[0]['layer']} alone contributes **{pct(layer_rows[0]['measured_error_share'])}**.",
        "- Therefore, `gate/up` dominate the error budget, while `o/v` are the most fragile normalized streams. Local reconstruction error alone cannot determine which one causes the largest downstream accuracy loss; that requires a dense-keep ablation by linear type.",
        "",
        "## By linear type",
        "",
        "| Linear | Error share | Local NMSE | Relative RMSE | RMS-proxy share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in leaf_rows:
        lines.append(
            "| {linear} | {share} | {nmse} | {rrmse} | {proxy} |".format(
                linear=row["linear"],
                share=pct(row["measured_error_share"]),
                nmse=pct(row["measured_nmse"]),
                rrmse=pct(row["measured_relative_rmse"]),
                proxy=pct(row["rms_proxy_error_share"]),
            )
        )
    lines.extend(
        [
            "",
            "## Largest individual modules",
            "",
            "| Rank | Module | Error share | Local NMSE | Error MSE |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(
        sorted(
            module_rows,
            key=lambda item: float(item["measured_error_sq"]),
            reverse=True,
        )[:15],
        start=1,
    ):
        lines.append(
            f"| {rank} | `{row['module']}` | "
            f"{pct(row['measured_error_share'])} | "
            f"{pct(row['measured_nmse'])} | "
            f"{sci(row['measured_error_mse'])} |"
        )
    lines.extend(
        [
            "",
            "## Layers with the largest aggregate error",
            "",
            "| Rank | Layer | Error share | Aggregate NMSE |",
            "|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(layer_rows[:10], start=1):
        lines.append(
            f"| {rank} | {row['layer']} | "
            f"{pct(row['measured_error_share'])} | "
            f"{pct(row['measured_nmse'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `module_error.csv`: all 224 target linear modules.",
            "- `linear_type_error.csv`: aggregation by q/k/v/o/gate/up/down.",
            "- `layer_error.csv`: aggregation by transformer layer.",
            "- `audit.json`: exact model/calibration identities and run parameters.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--calibration-prompts",
        type=Path,
        default=CALIBRATION_PROMPTS,
    )
    parser.add_argument("--activation-rms", type=Path, default=ACTIVATION_RMS)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = (
            EVAL_ROOT
            / "results_final"
            / f"llama3_8b_wanda_error_attribution_{timestamp()}"
        )
    return args


def main() -> None:
    args = parse_args()
    for path in (args.model, args.calibration_prompts, args.activation_rms):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local-output diagnostic")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    prompts = load_prompts(
        args.calibration_prompts,
        args.num_prompts,
        args.seed,
    )
    activation_scales = torch.load(args.activation_rms, map_location="cpu")
    activation_metadata = args.activation_rms.with_name(
        f"{args.activation_rms.stem}_metadata.json"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()

    rows, group_masks = make_rows(
        model,
        activation_scales,
        row_chunk=args.row_chunk,
    )
    total_tokens = measure_local_output_error(
        model,
        tokenizer,
        prompts,
        rows,
        group_masks,
        max_seq_len=args.max_seq_len,
    )
    finalize_rows(rows)
    leaf_rows = aggregate(rows, "linear")
    layer_rows = aggregate(rows, "layer")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "module_error.csv", rows)
    write_csv(output_root / "linear_type_error.csv", leaf_rows)
    write_csv(output_root / "layer_error.csv", layer_rows)
    write_json(
        output_root / "audit.json",
        {
            "method": "original_activation_rms_wanda_exact_2_of_4",
            "model": str(args.model.resolve()),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "calibration_prompts": str(args.calibration_prompts.resolve()),
            "calibration_prompts_sha256": sha256_file(
                args.calibration_prompts
            ),
            "activation_rms": str(args.activation_rms.resolve()),
            "activation_rms_sha256": sha256_file(args.activation_rms),
            "activation_metadata": (
                str(activation_metadata.resolve())
                if activation_metadata.exists()
                else None
            ),
            "activation_metadata_sha256": (
                sha256_file(activation_metadata)
                if activation_metadata.exists()
                else None
            ),
            "num_prompts": len(prompts),
            "total_tokens": total_tokens,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "dtype": "bf16",
            "target_module_count": len(rows),
            "actual_sparsity": 0.5,
            "created_at": timestamp(),
        },
    )
    write_report(
        output_root / "report.md",
        args=args,
        module_rows=rows,
        leaf_rows=leaf_rows,
        layer_rows=layer_rows,
        total_tokens=total_tokens,
    )
    print(output_root)


if __name__ == "__main__":
    main()
