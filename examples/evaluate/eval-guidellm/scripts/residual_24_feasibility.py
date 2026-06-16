#!/usr/bin/env python3
"""Residual 2:4 feasibility utilities.

Usage examples from examples/evaluate/eval-guidellm:

  # 1. Prototype lossless complementary 2:4 storage on Qwen/Llama shapes.
  conda run -n spec python scripts/residual_24_feasibility.py storage \
    --output-root results/residual_24_feasibility_TIMESTAMP

  # 2. Analyze confidence-barrier residual-token ratios from trace JSONL files.
  conda run -n spec python scripts/residual_24_feasibility.py barrier \
    temp results --output-root results/residual_24_feasibility_TIMESTAMP

  # 3. Collect verifier-detail timing from motivation_breakdown run roots.
  conda run -n spec python scripts/residual_24_feasibility.py verifier \
    results/residual_24_feasibility_TIMESTAMP/qwen_verifier_breakdown_run \
    results/residual_24_feasibility_TIMESTAMP/llama_verifier_breakdown_run \
    --output-root results/residual_24_feasibility_TIMESTAMP

  # 4. Combine verifier timing and barrier ratios into an analytical estimate.
  conda run -n spec python scripts/residual_24_feasibility.py speedup \
    --verifier-breakdown results/residual_24_feasibility_TIMESTAMP/verifier_breakdown.csv \
    --residual-barrier results/residual_24_feasibility_TIMESTAMP/residual_barrier_tradeoff.csv \
    --output-root results/residual_24_feasibility_TIMESTAMP

  # 5. Evaluate dense vs in-memory structured 2:4 masked model quality.
  conda run -n spec python scripts/residual_24_feasibility.py quality \
    --models qwen3_8b,llama3_1_8b \
    --mask-scopes none,attn,ffn,all \
    --datasets gsm8k,humaneval,math_reasoning,mtbench,dolly \
    --output-root results/structured_24_quality_TIMESTAMP

This script is analysis-only. It does not implement a fused 2:4 kernel and does
not claim measured end-to-end speedup. The quality subcommand applies masks in
memory and still runs dense PyTorch/Transformers kernels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)
DEFAULT_DATASETS = ("math", "mtbench")
DEFAULT_MODELS = ("qwen3_8b", "llama3_1_8b")
DEFAULT_METHODS = ("eagle3", "peagle")
DEFAULT_BATCHES = (8, 16, 32, 64)
DEFAULT_KS = (4, 8, 12)
DEFAULT_STORAGE_MODELS = [
    ("qwen3_8b", "Qwen/Qwen3-8B"),
    (
        "llama3_1_8b",
        "/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/llama-3.1-8b-instruct",
    ),
]

STORAGE_FIELDS = [
    "model",
    "model_path",
    "module",
    "out_features",
    "in_features",
    "dense_value_count",
    "base_value_count",
    "residual_value_count",
    "metadata_group_count",
    "metadata_bits_per_group",
    "value_bits",
    "dense_storage_bits",
    "base_residual_storage_bits",
    "shared_metadata_storage_bits",
    "estimated_storage_overhead_vs_dense_pct",
    "max_abs_reconstruction_error",
    "lossless_reconstruct",
    "status",
    "warning",
]

BARRIER_FIELDS = [
    "dataset",
    "model",
    "method",
    "K",
    "threshold",
    "residual_token_ratio",
    "reject_coverage",
    "full_accept_rate",
    "avg_h",
    "p50_h",
    "p90_h",
    "steps",
    "rejection_steps",
    "source_files",
    "no_drop_mode",
    "promising",
    "risk_note",
]

VERIFIER_FIELDS = [
    "model",
    "method",
    "batch_size",
    "K",
    "qkv_proj_pct",
    "attention_pct",
    "o_proj_pct",
    "ffn_pct",
    "verifier_other_pct",
    "linear_total_pct",
    "verify_total_ms_per_iter",
    "decode_events",
    "status",
    "note",
    "run_dir",
]

SPEEDUP_FIELDS = [
    "dataset",
    "model",
    "method",
    "batch_size",
    "K",
    "threshold",
    "linear_total_pct",
    "residual_token_ratio",
    "estimated_linear_speedup",
    "estimated_overall_speedup",
    "reject_coverage",
    "promising",
    "note",
]


@dataclass
class TokenRow:
    position: int
    probability: float
    accepted: int | None
    first_reject: int | None
    scheduled_k: int


@dataclass
class FileStats:
    path: Path
    counts: dict[tuple[str, str, str, int], int] = field(default_factory=dict)


def parse_csv_list(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH_OR_HF_ID")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("model label and path must be non-empty")
    return label, path


def config_int(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_shapes(config: Any) -> tuple[dict[str, tuple[int, int]], list[str]]:
    warnings: list[str] = []
    hidden_size = config_int(config, "hidden_size")
    intermediate_size = config_int(config, "intermediate_size")
    num_heads = config_int(config, "num_attention_heads")
    num_kv_heads = config_int(config, "num_key_value_heads") or num_heads
    head_dim = config_int(config, "head_dim")
    if head_dim is None and hidden_size is not None and num_heads:
        head_dim = hidden_size // num_heads
        if hidden_size % num_heads:
            warnings.append("hidden_size is not divisible by num_attention_heads")

    missing = [
        name
        for name, value in [
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("num_attention_heads", num_heads),
            ("num_key_value_heads", num_kv_heads),
            ("head_dim", head_dim),
        ]
        if value is None
    ]
    if missing:
        warnings.append(f"missing config fields: {', '.join(missing)}")
        return {}, warnings

    assert hidden_size is not None
    assert intermediate_size is not None
    assert num_heads is not None
    assert num_kv_heads is not None
    assert head_dim is not None

    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim
    return {
        "q_proj": (q_out, hidden_size),
        "k_proj": (kv_out, hidden_size),
        "v_proj": (kv_out, hidden_size),
        "o_proj": (hidden_size, q_out),
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
    }, warnings


def decompose_shape(
    out_features: int,
    in_features: int,
    *,
    dtype: Any,
    seed: int,
) -> tuple[bool, float]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dense = torch.randn((out_features, in_features), generator=generator, dtype=dtype)
    grouped = dense.reshape(-1, 4)
    top2_idx = grouped.abs().topk(k=2, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(1, top2_idx, True)
    base = torch.where(mask, grouped, torch.zeros((), dtype=dtype))
    residual = torch.where(~mask, grouped, torch.zeros((), dtype=dtype))
    reconstructed = (base + residual).reshape_as(dense)
    max_error = float((reconstructed - dense).abs().max().item())
    return max_error == 0.0, max_error


def storage_row(
    *,
    model: str,
    model_path: str,
    module: str,
    shape: tuple[int, int],
    value_bits: int,
    metadata_bits_per_group: int,
    dtype: Any,
    seed: int,
) -> dict[str, Any]:
    out_features, in_features = shape
    if in_features % 4 != 0:
        return {
            "model": model,
            "model_path": model_path,
            "module": module,
            "out_features": out_features,
            "in_features": in_features,
            "status": "skipped",
            "warning": "in_features is not divisible by 4",
        }

    dense_count = out_features * in_features
    groups = dense_count // 4
    base_count = groups * 2
    residual_count = groups * 2
    dense_bits = dense_count * value_bits
    value_bits_total = (base_count + residual_count) * value_bits
    metadata_bits = groups * metadata_bits_per_group
    overhead = ((value_bits_total + metadata_bits) / dense_bits) - 1.0
    ok, max_error = decompose_shape(
        out_features, in_features, dtype=dtype, seed=seed
    )
    return {
        "model": model,
        "model_path": model_path,
        "module": module,
        "out_features": out_features,
        "in_features": in_features,
        "dense_value_count": dense_count,
        "base_value_count": base_count,
        "residual_value_count": residual_count,
        "metadata_group_count": groups,
        "metadata_bits_per_group": metadata_bits_per_group,
        "value_bits": value_bits,
        "dense_storage_bits": dense_bits,
        "base_residual_storage_bits": value_bits_total,
        "shared_metadata_storage_bits": metadata_bits,
        "estimated_storage_overhead_vs_dense_pct": round(overhead * 100.0, 4),
        "max_abs_reconstruction_error": max_error,
        "lossless_reconstruct": ok,
        "status": "ok" if ok else "failed",
        "warning": "",
    }


def write_storage_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    overheads = [
        float(row["estimated_storage_overhead_vs_dense_pct"])
        for row in ok_rows
        if row.get("estimated_storage_overhead_vs_dense_pct") not in ("", None)
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Complementary 2:4 Storage Prototype\n\n")
        handle.write(
            "This is a Python-only random-weight prototype over the real "
            "Proj/FFN matrix shapes inferred from Hugging Face configs. "
            "It does not implement a fused 2:4 kernel.\n\n"
        )
        handle.write(f"- rows: {len(rows)}\n")
        handle.write(f"- ok rows: {len(ok_rows)}\n")
        handle.write(f"- non-ok rows: {len(failed_rows)}\n")
        if overheads:
            handle.write(
                f"- shared metadata overhead vs dense: "
                f"{min(overheads):.4f}% to {max(overheads):.4f}%\n"
            )
        handle.write("\n")
        handle.write("| model | module | shape [out,in] | lossless | overhead vs dense |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            overhead = row.get("estimated_storage_overhead_vs_dense_pct", "")
            handle.write(
                f"| {row.get('model', '')} | {row.get('module', '')} | "
                f"{row.get('out_features', '')} x {row.get('in_features', '')} | "
                f"{row.get('lossless_reconstruct', '')} | {overhead} |\n"
            )
        if failed_rows:
            handle.write("\n## Warnings\n\n")
            for row in failed_rows:
                handle.write(
                    f"- {row.get('model')} {row.get('module', '')}: "
                    f"{row.get('warning', row.get('status'))}\n"
                )


def run_storage(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    models = args.model or DEFAULT_STORAGE_MODELS
    rows: list[dict[str, Any]] = []
    for model_label, model_path in models:
        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=not args.allow_remote_config,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "model": model_label,
                    "model_path": model_path,
                    "status": "skipped",
                    "warning": f"failed to load config: {exc}",
                }
            )
            continue

        shapes, warnings = infer_shapes(config)
        if not shapes:
            rows.append(
                {
                    "model": model_label,
                    "model_path": model_path,
                    "status": "skipped",
                    "warning": "; ".join(warnings),
                }
            )
            continue

        for idx, (module, shape) in enumerate(shapes.items()):
            row = storage_row(
                model=model_label,
                model_path=model_path,
                module=module,
                shape=shape,
                value_bits=args.value_bits,
                metadata_bits_per_group=args.metadata_bits_per_group,
                dtype=dtype,
                seed=args.seed + idx,
            )
            if warnings:
                row["warning"] = "; ".join(
                    filter(None, [row.get("warning", ""), *warnings])
                )
            rows.append(row)
            gc.collect()

    csv_path = args.output_root / "complementary_24_storage.csv"
    md_path = args.output_root / "complementary_24_storage.md"
    write_csv(csv_path, rows, STORAGE_FIELDS)
    write_storage_markdown(md_path, rows)
    print(csv_path)
    print(md_path)


def iter_candidate_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            yield path
            continue
        for candidate in path.rglob("*.jsonl"):
            text = str(candidate)
            if "chunk_trace" in text:
                continue
            if "trace" not in candidate.name and "trace" not in str(candidate.parent):
                continue
            yield candidate


def infer_from_text(text: str, choices: Iterable[str]) -> str | None:
    lowered = text.lower()
    for choice in choices:
        if choice.lower() in lowered:
            return choice
    return None


def normalize_record(
    record: dict[str, Any],
    path: Path,
) -> tuple[str | None, str | None, str | None, int | None]:
    context = " ".join(
        str(record.get(key, ""))
        for key in ("request_id", "run_id", "sequence_id", "dataset_label")
    )
    context = f"{context} {path}"

    dataset = record.get("dataset_label")
    if not dataset or dataset == "unknown":
        dataset = infer_from_text(context, ("mtbench", "math", "synthetic_1000x1000"))
        if dataset is None and "speclink_confidence_acceptance" in str(path):
            dataset = "math"

    model = record.get("model_label")
    if not model or model == "unknown":
        model = infer_from_text(context, DEFAULT_MODELS)

    method = record.get("method") or record.get("spec_method")
    if not method or method == "unknown":
        method = infer_from_text(context, DEFAULT_METHODS)

    raw_k = (
        record.get("num_spec_tokens")
        or record.get("num_scheduled_draft_tokens")
        or record.get("trace_pending_tokens")
    )
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        match = re.search(r"[_-]k(\d+)", context.lower())
        k = int(match.group(1)) if match else None

    return (
        str(dataset) if dataset else None,
        str(model) if model else None,
        str(method) if method else None,
        k,
    )


def scan_file(path: Path) -> FileStats:
    stats = FileStats(path=path)
    try:
        handle = path.open("r", encoding="utf-8", errors="ignore")
    except OSError:
        return stats
    with handle:
        for line in handle:
            if "draft_selected_prob" not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or "num_accepted_in_step" not in record:
                continue
            dataset, model, method, k = normalize_record(record, path)
            if dataset and model and method and k is not None:
                key = (dataset, model, method, k)
                stats.counts[key] = stats.counts.get(key, 0) + 1
    return stats


def choose_sources(
    files: list[Path],
    desired: set[tuple[str, str, str, int]],
) -> dict[tuple[str, str, str, int], list[Path]]:
    best: dict[tuple[str, str, str, int], FileStats] = {}
    for path in files:
        stats = scan_file(path)
        for combo, count in stats.counts.items():
            if combo not in desired:
                continue
            old = best.get(combo)
            if old is None or count > old.counts.get(combo, 0):
                best[combo] = stats
    return {combo: [stats.path] for combo, stats in best.items()}


def load_steps(
    sources: dict[tuple[str, str, str, int], list[Path]],
) -> dict[tuple[str, str, str, int], dict[tuple[str, int], list[TokenRow]]]:
    wanted_by_path: dict[Path, set[tuple[str, str, str, int]]] = defaultdict(set)
    for combo, paths in sources.items():
        for path in paths:
            wanted_by_path[path].add(combo)

    steps: dict[
        tuple[str, str, str, int], dict[tuple[str, int], list[TokenRow]]
    ] = defaultdict(lambda: defaultdict(list))
    for path, wanted in wanted_by_path.items():
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "draft_selected_prob" not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                dataset, model, method, k = normalize_record(record, path)
                if dataset is None or model is None or method is None or k is None:
                    continue
                combo = (dataset, model, method, k)
                if combo not in wanted:
                    continue
                position = to_int(record.get("draft_position"))
                probability = to_float(record.get("draft_selected_prob"))
                if position is None or probability is None or position < 1:
                    continue
                request_id = str(record.get("request_id") or record.get("sequence_id"))
                step_id = to_int(record.get("step_id"))
                if not request_id or step_id is None:
                    continue
                accepted = to_int(record.get("num_accepted_in_step"))
                first_reject = to_int(record.get("first_reject_position"))
                scheduled_k = (
                    to_int(record.get("num_scheduled_draft_tokens"))
                    or to_int(record.get("trace_pending_tokens"))
                    or k
                )
                steps[combo][(request_id, step_id)].append(
                    TokenRow(
                        position=position,
                        probability=max(probability, 1e-45),
                        accepted=accepted,
                        first_reject=first_reject,
                        scheduled_k=scheduled_k,
                    )
                )
    return steps


def barrier_h(rows: list[TokenRow], threshold: float, k: int, no_drop_mode: str) -> int:
    log_sum = 0.0
    for row in sorted(rows, key=lambda item: item.position):
        if row.position > k:
            continue
        log_sum += math.log(row.probability)
        if math.exp(log_sum) < threshold:
            return min(row.position, k)
    return k if no_drop_mode == "safe" else 0


def first_reject_position(rows: list[TokenRow], k: int) -> int | None:
    for row in rows:
        if row.first_reject is not None and 1 <= row.first_reject <= k:
            return row.first_reject
    accepted_values = [row.accepted for row in rows if row.accepted is not None]
    if not accepted_values:
        return None
    accepted = max(accepted_values)
    if accepted < k:
        return max(1, accepted + 1)
    return None


def summarize_combo(
    combo: tuple[str, str, str, int],
    step_rows: dict[tuple[str, int], list[TokenRow]],
    thresholds: list[float],
    sources: list[Path],
    no_drop_mode: str,
) -> list[dict[str, Any]]:
    dataset, model, method, k = combo
    out: list[dict[str, Any]] = []
    source_text = ";".join(str(path) for path in sources)
    for threshold in thresholds:
        hs: list[int] = []
        rejections = 0
        covered = 0
        full_accept = 0
        total_k = 0
        for rows in step_rows.values():
            if not rows:
                continue
            h = barrier_h(rows, threshold, k, no_drop_mode)
            hs.append(h)
            total_k += k
            reject_pos = first_reject_position(rows, k)
            if reject_pos is None:
                full_accept += 1
            else:
                rejections += 1
                if reject_pos <= h:
                    covered += 1
        if not hs:
            continue
        residual_ratio = sum(hs) / total_k if total_k else math.nan
        reject_coverage = covered / rejections if rejections else math.nan
        full_accept_rate = full_accept / len(hs)
        promising = (
            residual_ratio <= 0.4
            and (not math.isnan(reject_coverage))
            and reject_coverage >= 0.8
        )
        risk_note = (
            "promising"
            if promising
            else "high_risk" if residual_ratio > 0.6 else "needs_kernel_validation"
        )
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "K": k,
                "threshold": threshold,
                "residual_token_ratio": round(residual_ratio, 6),
                "reject_coverage": ""
                if math.isnan(reject_coverage)
                else round(reject_coverage, 6),
                "full_accept_rate": round(full_accept_rate, 6),
                "avg_h": round(mean(hs), 6),
                "p50_h": round(percentile(hs, 50), 6),
                "p90_h": round(percentile(hs, 90), 6),
                "steps": len(hs),
                "rejection_steps": rejections,
                "source_files": source_text,
                "no_drop_mode": no_drop_mode,
                "promising": promising,
                "risk_note": risk_note,
            }
        )
    return out


def write_barrier_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    missing: list[tuple[str, str, str, int]],
) -> None:
    promising = [row for row in rows if str(row.get("promising")) == "True"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Residual Barrier Tradeoff\n\n")
        handle.write(
            "The barrier is computed per decoding step from cumulative DLM "
            "draft-token confidence. The residual prefix length h is the first "
            "position whose cumulative confidence drops below the threshold; "
            "if no position crosses the threshold, h=0 unless safe mode is used.\n\n"
        )
        handle.write(f"- rows: {len(rows)}\n")
        handle.write(f"- promising rows: {len(promising)}\n")
        handle.write(f"- missing matrix cells: {len(missing)}\n\n")
        if missing:
            handle.write("## Missing Cells\n\n")
            for dataset, model, method, k in missing:
                handle.write(f"- {dataset}/{model}/{method}/K={k}\n")
            handle.write("\n")
        handle.write("## Best Rows By Cell\n\n")
        handle.write(
            "| dataset | model | method | K | threshold | residual ratio | reject coverage | avg h | note |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["dataset"], row["model"], row["method"], int(row["K"]))].append(row)
        for combo in sorted(grouped):
            candidates = grouped[combo]
            candidates.sort(
                key=lambda row: (
                    float(row["residual_token_ratio"]),
                    -float(row["reject_coverage"] or 0.0),
                )
            )
            row = candidates[0]
            handle.write(
                f"| {row['dataset']} | {row['model']} | {row['method']} | "
                f"{row['K']} | {row['threshold']} | {row['residual_token_ratio']} | "
                f"{row['reject_coverage']} | {row['avg_h']} | {row['risk_note']} |\n"
            )


def run_barrier(args: argparse.Namespace) -> None:
    eval_root = Path(__file__).resolve().parents[1]
    inputs = args.inputs or [eval_root / "temp", eval_root / "results"]
    datasets = parse_csv_list(args.datasets)
    models = parse_csv_list(args.models)
    methods = parse_csv_list(args.methods)
    ks = parse_csv_list(args.ks, int)
    thresholds = parse_csv_list(args.thresholds, float)

    desired = {
        (dataset, model, method, k)
        for dataset in datasets
        for model in models
        for method in methods
        for k in ks
        if not (model == "llama3_1_8b" and method == "peagle")
    }
    files = sorted(set(iter_candidate_files(inputs)))
    sources = choose_sources(files, desired)
    missing = sorted(desired - set(sources))
    steps = load_steps(sources)

    rows: list[dict[str, Any]] = []
    for combo in sorted(steps):
        rows.extend(
            summarize_combo(
                combo,
                steps[combo],
                thresholds,
                sources.get(combo, []),
                args.no_drop_mode,
            )
        )

    csv_path = args.output_root / "residual_barrier_tradeoff.csv"
    md_path = args.output_root / "residual_barrier_summary.md"
    write_csv(csv_path, rows, BARRIER_FIELDS)
    write_barrier_markdown(md_path, rows, missing)
    print(csv_path)
    print(md_path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def model_label(base_model: str) -> str:
    lowered = base_model.lower()
    if "llama" in lowered:
        return "llama3_1_8b"
    if "qwen" in lowered:
        return "qwen3_8b"
    return base_model or "unknown"


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    metadata = load_json(run_dir / "metadata.json")
    if not metadata:
        return None
    method = metadata.get("algo")
    batch_size = metadata.get("batch_size")
    k = metadata.get("num_spec_tokens")
    base_model = str(metadata.get("base_model", ""))
    if method is None or batch_size is None or k is None:
        return None
    events = load_jsonl(run_dir / "breakdown_events.jsonl")
    decode_events = [
        event
        for event in events
        if event.get("phase") == "decode" and event.get("use_spec_decode") is True
    ]
    if not decode_events:
        decode_events = [event for event in events if event.get("phase") == "decode"]
    if not decode_events:
        return {
            "model": model_label(base_model),
            "method": method,
            "batch_size": batch_size,
            "K": k,
            "status": "missing",
            "note": "no_decode_events",
            "run_dir": str(run_dir),
        }

    verify = sum(to_float(event.get("verify_forward_ms")) or 0.0 for event in decode_events)
    qkv = sum(to_float(event.get("verify_qkv_proj_ms")) or 0.0 for event in decode_events)
    attention = sum(to_float(event.get("verify_attention_ms")) or 0.0 for event in decode_events)
    ffn = sum(to_float(event.get("verify_ffn_ms")) or 0.0 for event in decode_events)
    other = sum(to_float(event.get("verify_model_other_ms")) or 0.0 for event in decode_events)
    has_detail = any(
        to_float(event.get(key)) is not None
        for event in decode_events
        for key in ("verify_qkv_proj_ms", "verify_attention_ms", "verify_ffn_ms")
    )
    if not verify or not has_detail:
        return {
            "model": model_label(base_model),
            "method": method,
            "batch_size": batch_size,
            "K": k,
            "status": "missing",
            "note": "verifier_detail_not_available",
            "run_dir": str(run_dir),
            "decode_events": len(decode_events),
        }

    def pct(value: float) -> float:
        return value / verify * 100.0 if verify else math.nan

    qkv_pct = pct(qkv)
    attention_pct = pct(attention)
    ffn_pct = pct(ffn)
    other_pct = pct(other)
    linear_total = qkv_pct + ffn_pct
    note = (
        "o_proj is included inside attention timing in current hooks; "
        "linear_total_pct is therefore a conservative qkv+ffn lower bound"
    )
    return {
        "model": model_label(base_model),
        "method": method,
        "batch_size": batch_size,
        "K": k,
        "qkv_proj_pct": round(qkv_pct, 6),
        "attention_pct": round(attention_pct, 6),
        "o_proj_pct": "",
        "ffn_pct": round(ffn_pct, 6),
        "verifier_other_pct": round(other_pct, 6),
        "linear_total_pct": round(linear_total, 6),
        "verify_total_ms_per_iter": round(verify / len(decode_events), 6),
        "decode_events": len(decode_events),
        "status": "ok",
        "note": note,
        "run_dir": str(run_dir),
    }


def collect_runs(roots: list[Path]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        metadata_paths = [root] if root.name == "metadata.json" else list(root.rglob("metadata.json"))
        for metadata_path in metadata_paths:
            run_dir = metadata_path.parent if metadata_path.name == "metadata.json" else metadata_path
            row = summarize_run(run_dir)
            if row is None:
                continue
            try:
                key = (
                    str(row["model"]),
                    str(row["method"]),
                    int(row["batch_size"]),
                    int(row["K"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            old = out.get(key)
            if old is None or (old.get("status") != "ok" and row.get("status") == "ok"):
                out[key] = row
    return out


def write_verifier_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    insufficient = [
        row for row in ok_rows if (to_float(row.get("linear_total_pct")) or 0.0) < 50.0
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Verifier Breakdown For Residual 2:4 Feasibility\n\n")
        handle.write(
            "Current verifier-detail instrumentation times qkv_proj, an "
            "attention block that includes o_proj, ffn, and model_other. "
            "It does not separately time o_proj without modifying model forward.\n\n"
        )
        handle.write(f"- rows: {len(rows)}\n")
        handle.write(f"- ok rows: {len(ok_rows)}\n")
        handle.write(f"- rows with linear_total_pct < 50%: {len(insufficient)}\n\n")
        handle.write("| model | method | bs | K | qkv % | attention % | ffn % | linear total % | status |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row.get('model', '')} | {row.get('method', '')} | "
                f"{row.get('batch_size', '')} | {row.get('K', '')} | "
                f"{row.get('qkv_proj_pct', '')} | {row.get('attention_pct', '')} | "
                f"{row.get('ffn_pct', '')} | {row.get('linear_total_pct', '')} | "
                f"{row.get('status', '')} |\n"
            )
        if insufficient:
            handle.write(
                "\nRows below 50% linear verifier time indicate residual 2:4 "
                "alone is likely insufficient unless o_proj or other hidden "
                "linear work is separately recovered by a lower-level kernel.\n"
            )


def run_verifier(args: argparse.Namespace) -> None:
    models = parse_csv_list(args.models)
    methods = parse_csv_list(args.methods)
    batches = parse_csv_list(args.batch_sizes, int)
    ks = parse_csv_list(args.ks, int)
    found = collect_runs(args.roots)

    rows: list[dict[str, Any]] = []
    for model in models:
        for method in methods:
            if model == "llama3_1_8b" and method == "peagle":
                continue
            for batch_size in batches:
                for k in ks:
                    key = (model, method, batch_size, k)
                    row = found.get(key)
                    if row is None:
                        row = {
                            "model": model,
                            "method": method,
                            "batch_size": batch_size,
                            "K": k,
                            "status": "missing",
                            "note": "no_breakdown_run_found",
                        }
                    rows.append(row)

    csv_path = args.output_root / "verifier_breakdown.csv"
    md_path = args.output_root / "verifier_breakdown.md"
    write_csv(csv_path, rows, VERIFIER_FIELDS)
    write_verifier_markdown(md_path, rows)
    print(csv_path)
    print(md_path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct_to_fraction(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def write_speedup_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    promising = [row for row in rows if row.get("promising") is True]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Residual 2:4 Theoretical Speedup Estimate\n\n")
        handle.write(
            "This is an analytical estimate only. It assumes the measured "
            "linear verifier fraction can run with complementary 2:4 base and "
            "residual work according to the residual-token ratio; it is not an "
            "end-to-end speedup claim.\n\n"
        )
        handle.write(f"- rows: {len(rows)}\n")
        handle.write(f"- rows meeting 1.3x / <=0.4 residual / >=0.8 coverage: {len(promising)}\n\n")
        handle.write(
            "| dataset | model | method | bs | K | tau | linear pct | residual ratio | overall speedup | note |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in sorted(
            rows,
            key=lambda item: (
                item["model"],
                item["method"],
                int(item["batch_size"]),
                int(item["K"]),
                item["dataset"],
                float(item["threshold"]),
            ),
        )[:200]:
            handle.write(
                f"| {row['dataset']} | {row['model']} | {row['method']} | "
                f"{row['batch_size']} | {row['K']} | {row['threshold']} | "
                f"{row['linear_total_pct']} | {row['residual_token_ratio']} | "
                f"{row['estimated_overall_speedup']} | {row['note']} |\n"
            )


def run_speedup(args: argparse.Namespace) -> None:
    verifier_rows = read_csv(args.verifier_breakdown)
    barrier_rows = read_csv(args.residual_barrier)
    barriers: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in barrier_rows:
        k_value = row.get("K") or row.get("num_spec_tokens")
        try:
            key = (row["model"], row["method"], int(k_value))
        except (KeyError, TypeError, ValueError):
            continue
        barriers[key].append(row)

    out: list[dict[str, Any]] = []
    for vrow in verifier_rows:
        status = vrow.get("status", "ok")
        if status and status != "ok":
            continue
        linear_pct_raw = to_float(vrow.get("linear_total_pct"))
        if linear_pct_raw is None:
            continue
        linear_fraction = pct_to_fraction(linear_pct_raw)
        try:
            key = (vrow["model"], vrow["method"], int(vrow["K"]))
        except (KeyError, TypeError, ValueError):
            continue
        for brow in barriers.get(key, []):
            residual_ratio = to_float(brow.get("residual_token_ratio"))
            reject_coverage = to_float(brow.get("reject_coverage"))
            if residual_ratio is None:
                continue
            linear_speedup = 1.0 / (0.5 + 0.5 * residual_ratio)
            overall_speedup = 1.0 / (
                1.0 - linear_fraction + linear_fraction / linear_speedup
            )
            promising = (
                overall_speedup >= 1.3
                and residual_ratio <= 0.4
                and reject_coverage is not None
                and reject_coverage >= 0.8
            )
            note = (
                "meets_thresholds"
                if promising
                else "linear_fraction_or_residual_ratio_insufficient"
            )
            out.append(
                {
                    "dataset": brow.get("dataset", ""),
                    "model": vrow["model"],
                    "method": vrow["method"],
                    "batch_size": vrow["batch_size"],
                    "K": vrow["K"],
                    "threshold": brow.get("threshold", ""),
                    "linear_total_pct": round(linear_fraction * 100.0, 6),
                    "residual_token_ratio": residual_ratio,
                    "estimated_linear_speedup": round(linear_speedup, 6),
                    "estimated_overall_speedup": round(overall_speedup, 6),
                    "reject_coverage": "" if reject_coverage is None else reject_coverage,
                    "promising": promising,
                    "note": note,
                }
            )

    csv_path = args.output_root / "residual_24_speedup_estimate.csv"
    md_path = args.output_root / "residual_24_speedup_estimate.md"
    write_csv(csv_path, out, SPEEDUP_FIELDS)
    write_speedup_markdown(md_path, out)
    print(csv_path)
    print(md_path)


# Heavy quality-evaluation dependencies are loaded only by the `quality`
# subcommand so the storage/barrier/verifier/speedup analysis paths stay light.
torch = None
nn = None
AutoModelForCausalLM = None
AutoTokenizer = None


def ensure_quality_dependencies() -> None:
    global torch, nn, AutoModelForCausalLM, AutoTokenizer
    if torch is not None:
        return
    import torch as torch_module
    import torch.nn as nn_module
    from transformers import AutoModelForCausalLM as auto_model_cls
    from transformers import AutoTokenizer as auto_tokenizer_cls

    torch = torch_module
    nn = nn_module
    AutoModelForCausalLM = auto_model_cls
    AutoTokenizer = auto_tokenizer_cls

QUALITY_SCRIPT_PATH = Path(__file__).resolve()
QUALITY_EVAL_ROOT = QUALITY_SCRIPT_PATH.parents[1]
QUALITY_DATA_ROOT = QUALITY_EVAL_ROOT / "data"
QUALITY_RESULTS_ROOT = QUALITY_EVAL_ROOT / "results"

QUALITY_DEFAULT_MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}

QUALITY_MASK_TARGETS = {
    "none": (),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "ffn": ("gate_proj", "up_proj", "down_proj"),
    "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}

QUALITY_CSV_FIELDS = [
    "model_label",
    "model_id",
    "mask_scope",
    "dataset",
    "metric_name",
    "metric_type",
    "dense_metric_value",
    "sparse_metric_value",
    "delta_vs_dense",
    "ratio_vs_dense",
    "num_examples",
    "ppl_mode",
    "accuracy_available",
    "pass_at_1_available",
    "dtype",
    "actual_sparsity",
    "zeroed_weight_count",
    "total_masked_weight_count",
    "skipped",
    "failed",
    "error",
]


@dataclass
class DatasetPack:
    name: str
    rows: list[dict[str, Any]]
    error: str = ""


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_model_id_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--model-id must use LABEL=MODEL_ID")
        label, model_id = value.split("=", 1)
        label = label.strip()
        model_id = model_id.strip()
        if not label or not model_id:
            raise argparse.ArgumentTypeError("--model-id label and value must be non-empty")
        overrides[label] = model_id
    return overrides


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_reference(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            text = normalize_reference(item)
            if text:
                return text
        return ""
    return str(value)


def first_text_field(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = normalize_reference(value)
        if text:
            return text
    return ""


def normalize_number(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = text.rstrip(".")
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None
    if "####" in text:
        lines = text.rsplit("####", 1)[-1].strip().splitlines()
        if lines:
            normalized = normalize_number(lines[0])
            if normalized:
                return normalized
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        normalized = normalize_number(boxed[-1])
        if normalized:
            return normalized
    numbers = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {value}")


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def finite_ppl(nll: float | None) -> float | None:
    if nll is None:
        return None
    if nll > 50:
        return float("inf")
    return math.exp(nll)


def module_is_skipped(name: str, skip_lm_head: bool, skip_embeddings: bool) -> bool:
    if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
        return True
    if skip_embeddings and any(part in name.lower() for part in ("embed_tokens", "wte", "embedding")):
        return True
    return False


def apply_24_mask_to_model(
    model: nn.Module,
    target_modules: tuple[str, ...],
    group_dim: str = "in",
    skip_lm_head: bool = True,
    skip_embeddings: bool = True,
) -> dict[str, Any]:
    """Apply in-place 2:4 masks to selected Linear weights.

    For a Linear weight shaped [out_features, in_features], group_dim="in"
    groups every 4 consecutive in_features values per output row, keeps the
    largest 2 by absolute value, and zeroes the other 2. Tails shorter than 4
    are left unchanged and reported.
    """

    if group_dim != "in":
        raise ValueError("only group_dim='in' is supported")

    stats: dict[str, Any] = {
        "target_modules": list(target_modules),
        "group_dim": group_dim,
        "skip_lm_head": skip_lm_head,
        "skip_embeddings": skip_embeddings,
        "total_masked_weight_count": 0,
        "zeroed_weight_count": 0,
        "actual_sparsity": 0.0,
        "masked_module_names": [],
        "per_module": [],
        "tail_warnings": [],
    }

    if not target_modules:
        return stats

    target_set = set(target_modules)
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in target_set:
                continue
            if module_is_skipped(name, skip_lm_head, skip_embeddings):
                continue
            weight = module.weight
            if weight.ndim != 2:
                continue
            out_features, in_features = weight.shape
            usable_in = (in_features // 4) * 4
            tail = in_features - usable_in
            if usable_in == 0:
                stats["tail_warnings"].append(
                    {"module": name, "in_features": in_features, "tail": tail}
                )
                continue

            view = weight[:, :usable_in].view(out_features, usable_in // 4, 4)
            keep_idx = view.abs().topk(k=2, dim=-1, largest=True, sorted=False).indices
            keep = torch.zeros_like(view, dtype=torch.bool)
            keep.scatter_(-1, keep_idx, True)
            zeroed = int((~keep).sum().item())
            total = int(keep.numel())
            view.masked_fill_(~keep, 0)

            module_stats = {
                "module": name,
                "shape": [int(out_features), int(in_features)],
                "masked_weight_count": total,
                "zeroed_weight_count": zeroed,
                "actual_sparsity": zeroed / total if total else 0.0,
                "unmasked_tail_in_features": int(tail),
            }
            stats["per_module"].append(module_stats)
            stats["masked_module_names"].append(name)
            stats["total_masked_weight_count"] += total
            stats["zeroed_weight_count"] += zeroed
            if tail:
                stats["tail_warnings"].append(
                    {"module": name, "in_features": int(in_features), "tail": int(tail)}
                )

    total = stats["total_masked_weight_count"]
    stats["actual_sparsity"] = (
        stats["zeroed_weight_count"] / total if total else 0.0
    )
    return stats


def load_gsm8k(limit: int | None, seed: int) -> DatasetPack:
    local = QUALITY_DATA_ROOT / "gsm8k.jsonl"
    try:
        if local.exists():
            raw_rows = read_jsonl(local)
        else:
            from datasets import load_dataset

            raw_rows = list(load_dataset("openai/gsm8k", "main", split="test"))
        rows = []
        for idx, row in enumerate(raw_rows):
            question = first_text_field(row, ("question", "prompt", "input"))
            answer = first_text_field(row, ("answer", "reference", "target", "output"))
            rows.append(
                {
                    "id": row.get("question_id", row.get("id", idx)),
                    "prompt": f"Question:\n{question}\n\nAnswer:",
                    "gold_text": answer,
                    "gold_answer": extract_final_answer(answer),
                }
            )
        return DatasetPack("gsm8k", sample_rows(rows, limit, seed))
    except Exception as exc:  # noqa: BLE001
        return DatasetPack("gsm8k", [], error=str(exc))


def load_humaneval(limit: int | None, seed: int) -> DatasetPack:
    local = QUALITY_DATA_ROOT / "humaneval.jsonl"
    try:
        if local.exists():
            raw_rows = read_jsonl(local)
        else:
            from datasets import load_dataset

            raw_rows = list(load_dataset("openai_humaneval", split="test"))
        rows = []
        for idx, row in enumerate(raw_rows):
            rows.append(
                {
                    "id": row.get("task_id", row.get("id", idx)),
                    "prompt": str(row.get("prompt", "")),
                    "entry_point": str(row.get("entry_point", "")),
                    "test": str(row.get("test", "")),
                }
            )
        return DatasetPack("humaneval", sample_rows(rows, limit, seed))
    except Exception as exc:  # noqa: BLE001
        return DatasetPack("humaneval", [], error=str(exc))


def load_math_reasoning(limit: int | None, seed: int) -> DatasetPack:
    local = QUALITY_DATA_ROOT / "math_reasoning.jsonl"
    try:
        raw_rows = read_jsonl(local)
        rows = []
        for idx, row in enumerate(raw_rows):
            prompt = first_text_field(row, ("prompt", "question", "input"))
            target = first_text_field(row, ("answer", "output", "response", "target", "reference"))
            rows.append(
                {
                    "id": row.get("question_id", row.get("id", idx)),
                    "prompt": prompt,
                    "target": target,
                    "gold_answer": extract_final_answer(target),
                }
            )
        return DatasetPack("math_reasoning", sample_rows(rows, limit, seed))
    except Exception as exc:  # noqa: BLE001
        return DatasetPack("math_reasoning", [], error=str(exc))


def load_mtbench(limit: int | None, seed: int) -> DatasetPack:
    local = QUALITY_DATA_ROOT / "mt_bench.jsonl"
    try:
        raw_rows = read_jsonl(local)
        rows = []
        for idx, row in enumerate(raw_rows):
            prompt = first_text_field(row, ("prompt", "question", "input"))
            target = first_text_field(
                row, ("answer", "output", "response", "target", "reference")
            )
            rows.append(
                {
                    "id": row.get("question_id", row.get("id", idx)),
                    "prompt": prompt,
                    "target": target,
                    "category": row.get("category", ""),
                }
            )
        return DatasetPack("mtbench", sample_rows(rows, limit, seed))
    except Exception as exc:  # noqa: BLE001
        return DatasetPack("mtbench", [], error=str(exc))


def load_dolly(limit: int | None, seed: int) -> DatasetPack:
    local = QUALITY_DATA_ROOT / "dolly" / "dolly_all.jsonl"
    try:
        if local.exists():
            raw_rows = read_jsonl(local)
        else:
            from datasets import load_dataset

            raw_rows = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
        rows = []
        for idx, row in enumerate(raw_rows):
            instruction = str(row.get("instruction", ""))
            context = str(row.get("context", "") or "")
            response = first_text_field(row, ("response", "reference", "target", "output"))
            if context.strip():
                prompt = f"Instruction:\n{instruction}\n\nContext:\n{context}\n\nResponse:"
            else:
                prompt = f"Instruction:\n{instruction}\n\nResponse:"
            rows.append(
                {
                    "id": row.get("source_id", row.get("id", idx)),
                    "instruction": instruction,
                    "context": context,
                    "prompt": prompt,
                    "target": response,
                    "category": row.get("category", ""),
                }
            )
        return DatasetPack("dolly", sample_rows(rows, limit, seed))
    except Exception as exc:  # noqa: BLE001
        return DatasetPack("dolly", [], error=str(exc))


def sample_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[idx] for idx in indices]


def load_datasets(args: argparse.Namespace) -> dict[str, DatasetPack]:
    datasets = parse_csv_list(args.datasets)
    out: dict[str, DatasetPack] = {}
    for name in datasets:
        if name == "gsm8k":
            out[name] = load_gsm8k(args.gsm8k_num_examples, args.seed)
        elif name == "humaneval":
            out[name] = load_humaneval(args.humaneval_num_examples, args.seed)
        elif name == "math_reasoning":
            out[name] = load_math_reasoning(args.math_num_examples, args.seed)
        elif name == "mtbench":
            out[name] = load_mtbench(args.mtbench_num_examples, args.seed)
        elif name == "dolly":
            out[name] = load_dolly(args.dolly_num_examples, args.seed)
        else:
            out[name] = DatasetPack(name, [], error=f"unknown dataset: {name}")
    return out


def ensure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def load_model_and_tokenizer(
    model_id: str,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, Any]:
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
    model.to(device)
    model.eval()
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> list[str]:
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(
            tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        )
    return outputs


def target_loss_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_seq_len: int,
    device: str,
) -> tuple[float | None, int, str]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        return None, 0, "empty_target"

    if len(target_ids) >= max_seq_len:
        target_ids = target_ids[:max_seq_len]
        prompt_ids = []
    else:
        prompt_budget = max_seq_len - len(target_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
    if not target_ids:
        return None, 0, "target_truncated_empty"

    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    labels = torch.tensor(
        [[-100] * len(prompt_ids) + target_ids], dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    active = shift_labels.ne(-100)
    token_count = int(active.sum().item())
    if token_count == 0:
        return None, 0, "no_target_tokens_after_shift"
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return float(loss_sum.item()), token_count, ""


def evaluate_ppl(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    *,
    max_seq_len: int,
    device: str,
    examples_path: Path,
) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    skipped = 0
    details: list[dict[str, Any]] = []
    for row in examples:
        loss_sum, token_count, skip_reason = target_loss_one(
            model,
            tokenizer,
            str(row["prompt"]),
            str(row["target"]),
            max_seq_len=max_seq_len,
            device=device,
        )
        if loss_sum is None or token_count == 0:
            skipped += 1
            details.append(
                {
                    "id": row.get("id"),
                    "skipped": True,
                    "skip_reason": skip_reason,
                }
            )
            continue
        total_loss += loss_sum
        total_tokens += token_count
        details.append(
            {
                "id": row.get("id"),
                "target_tokens": token_count,
                "nll": loss_sum / token_count,
                "skipped": False,
            }
        )
    append_jsonl(examples_path, details)
    nll = total_loss / total_tokens if total_tokens else None
    return {
        "nll": nll,
        "ppl": finite_ppl(nll),
        "num_examples": len(examples) - skipped,
        "skipped_examples": skipped,
        "avg_target_tokens": (total_tokens / (len(examples) - skipped)) if len(examples) > skipped else None,
    }


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    prompts = [row["prompt"] for row in pack.rows]
    outputs = generate_texts(
        model,
        tokenizer,
        prompts,
        max_new_tokens=256,
        max_seq_len=args.max_seq_len,
        batch_size=args.generation_batch_size,
        device=args.device,
    )
    details: list[dict[str, Any]] = []
    correct = 0
    counted = 0
    for row, output in zip(pack.rows, outputs, strict=True):
        gold = row.get("gold_answer")
        pred = extract_final_answer(output)
        is_counted = gold is not None
        is_correct = bool(is_counted and pred is not None and pred == gold)
        if is_counted:
            counted += 1
            correct += int(is_correct)
        details.append(
            {
                "id": row.get("id"),
                "prompt": row.get("prompt"),
                "generation": output,
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": is_correct if is_counted else None,
                "counted": is_counted,
            }
        )
    append_jsonl(run_dir / "generations_gsm8k.jsonl", details)
    return {
        "metric_name": "gsm8k_accuracy",
        "metric_type": "accuracy",
        "value": correct / counted if counted else None,
        "gsm8k_accuracy": correct / counted if counted else None,
        "gsm8k_correct": correct,
        "gsm8k_num_examples": counted,
        "accuracy_available": counted > 0,
        "num_examples": counted,
    }


def run_humaneval_test(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout_s: float,
) -> tuple[bool, str]:
    code = (
        prompt
        + completion
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.TemporaryDirectory(prefix="humaneval_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                },
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as exc:  # noqa: BLE001
            return False, f"subprocess_error: {exc}"
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout)[-1000:]


def evaluate_humaneval(
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    prompts = [row["prompt"] for row in pack.rows]
    outputs = generate_texts(
        model,
        tokenizer,
        prompts,
        max_new_tokens=512,
        max_seq_len=args.max_seq_len,
        batch_size=args.generation_batch_size,
        device=args.device,
    )
    details: list[dict[str, Any]] = []
    solved = 0
    skipped = False
    skip_reason = ""
    for row, output in zip(pack.rows, outputs, strict=True):
        try:
            ok, error = run_humaneval_test(
                row["prompt"],
                output,
                row["test"],
                row["entry_point"],
                args.humaneval_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            skipped = True
            skip_reason = f"functional correctness unavailable: {exc}"
            ok = False
            error = skip_reason
        solved += int(ok)
        details.append(
            {
                "id": row.get("id"),
                "prompt": row.get("prompt"),
                "generation": output,
                "entry_point": row.get("entry_point"),
                "passed": ok,
                "error": error,
            }
        )
    append_jsonl(run_dir / "generations_humaneval.jsonl", details)
    available = not skipped
    return {
        "metric_name": "humaneval_pass_at_1",
        "metric_type": "pass_at_1",
        "value": solved / len(pack.rows) if pack.rows and available else None,
        "humaneval_pass_at_1": solved / len(pack.rows) if pack.rows and available else None,
        "humaneval_solved": solved if available else None,
        "humaneval_num_examples": len(pack.rows),
        "pass_at_1_available": available,
        "skipped_pass_at_1": skipped,
        "skip_reason": skip_reason,
        "num_examples": len(pack.rows),
    }


def evaluate_math_reasoning(
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    rows_with_gold = [row for row in pack.rows if row.get("gold_answer") is not None]
    if rows_with_gold:
        prompts = [row["prompt"] for row in pack.rows]
        outputs = generate_texts(
            model,
            tokenizer,
            prompts,
            max_new_tokens=256,
            max_seq_len=args.max_seq_len,
            batch_size=args.generation_batch_size,
            device=args.device,
        )
        correct = 0
        counted = 0
        details: list[dict[str, Any]] = []
        for row, output in zip(pack.rows, outputs, strict=True):
            gold = row.get("gold_answer")
            pred = extract_final_answer(output)
            is_counted = gold is not None
            is_correct = bool(is_counted and pred is not None and gold == pred)
            if is_counted:
                counted += 1
                correct += int(is_correct)
            details.append(
                {
                    "id": row.get("id"),
                    "prompt": row.get("prompt"),
                    "generation": output,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "correct": is_correct if is_counted else None,
                    "counted": is_counted,
                }
            )
        append_jsonl(run_dir / "generations_math_reasoning.jsonl", details)
        return {
            "metric_name": "math_accuracy",
            "metric_type": "accuracy",
            "value": correct / counted if counted else None,
            "math_accuracy": correct / counted if counted else None,
            "math_correct": correct,
            "math_num_accuracy_examples": counted,
            "accuracy_available": counted > 0,
            "num_examples": counted,
            "metric_mode": "accuracy",
        }

    ppl_rows = [
        {"id": row.get("id"), "prompt": row["prompt"], "target": row.get("target", "")}
        for row in pack.rows
        if row.get("target")
    ]
    loss = evaluate_ppl(
        model,
        tokenizer,
        ppl_rows,
        max_seq_len=args.max_seq_len,
        device=args.device,
        examples_path=run_dir / "generations_math_reasoning.jsonl",
    )
    return {
        "metric_name": "math_ppl_fallback",
        "metric_type": "ppl",
        "value": loss["ppl"],
        "math_nll": loss["nll"],
        "math_ppl": loss["ppl"],
        "num_examples": loss["num_examples"],
        "accuracy_available": False,
        "metric_mode": "ppl_fallback",
        "fallback_reason": "no_extractable_gold_answers",
    }


def mtbench_has_reference(pack: DatasetPack) -> bool:
    return bool(pack.rows) and all(bool(row.get("target")) for row in pack.rows)


def dense_reference_path(output_root: Path, model_label: str) -> Path:
    return output_root / "dense_references" / model_label / "mtbench_dense_references.jsonl"


def generate_or_load_mtbench_refs(
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    output_root: Path,
    model_label: str,
) -> list[dict[str, Any]]:
    path = dense_reference_path(output_root, model_label)
    if path.exists():
        return read_jsonl(path)
    prompts = [row["prompt"] for row in pack.rows]
    outputs = generate_texts(
        model,
        tokenizer,
        prompts,
        max_new_tokens=512,
        max_seq_len=args.max_seq_len,
        batch_size=args.generation_batch_size,
        device=args.device,
    )
    refs = []
    for row, output in zip(pack.rows, outputs, strict=True):
        refs.append({"id": row.get("id"), "prompt": row["prompt"], "target": output})
    append_jsonl(path, refs)
    return refs


def evaluate_mtbench(
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    run_dir: Path,
    output_root: Path,
    model_label: str,
    mask_scope: str,
) -> dict[str, Any]:
    if mtbench_has_reference(pack):
        ppl_mode = "reference_target"
        eval_rows = [
            {"id": row.get("id"), "prompt": row["prompt"], "target": row.get("target", "")}
            for row in pack.rows
            if row.get("target")
        ]
    else:
        ppl_mode = "dense_reference"
        if mask_scope != "none" and not dense_reference_path(output_root, model_label).exists():
            raise RuntimeError(
                "MT-Bench dense references are missing; run dense mask_scope=none first"
            )
        refs = generate_or_load_mtbench_refs(
            model, tokenizer, pack, args, output_root, model_label
        )
        shutil.copyfile(
            dense_reference_path(output_root, model_label),
            run_dir / "mtbench_dense_references.jsonl",
        )
        eval_rows = refs

    loss = evaluate_ppl(
        model,
        tokenizer,
        eval_rows,
        max_seq_len=args.max_seq_len,
        device=args.device,
        examples_path=run_dir / "generations_mtbench.jsonl",
    )
    return {
        "metric_name": "mtbench_ppl",
        "metric_type": "ppl",
        "value": loss["ppl"],
        "mtbench_nll": loss["nll"],
        "mtbench_ppl": loss["ppl"],
        "mtbench_num_examples": loss["num_examples"],
        "ppl_mode": ppl_mode,
        "num_examples": loss["num_examples"],
    }


def evaluate_dolly(
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    eval_rows = [
        {"id": row.get("id"), "prompt": row["prompt"], "target": row.get("target", "")}
        for row in pack.rows
        if row.get("target")
    ]
    loss = evaluate_ppl(
        model,
        tokenizer,
        eval_rows,
        max_seq_len=args.max_seq_len,
        device=args.device,
        examples_path=run_dir / "dolly_loss_examples.jsonl",
    )
    return {
        "metric_name": "dolly_ppl",
        "metric_type": "ppl",
        "value": loss["ppl"],
        "dolly_nll": loss["nll"],
        "dolly_ppl": loss["ppl"],
        "dolly_num_examples": loss["num_examples"],
        "dolly_avg_target_tokens": loss["avg_target_tokens"],
        "num_examples": loss["num_examples"],
    }


def expected_metric(dataset: str) -> tuple[str, str]:
    if dataset == "gsm8k":
        return "gsm8k_accuracy", "accuracy"
    if dataset == "humaneval":
        return "humaneval_pass_at_1", "pass_at_1"
    if dataset == "math_reasoning":
        return "math_accuracy", "accuracy"
    if dataset == "mtbench":
        return "mtbench_ppl", "ppl"
    if dataset == "dolly":
        return "dolly_ppl", "ppl"
    return f"{dataset}_metric", "unknown"


def failed_metric(dataset: str, error: str) -> dict[str, Any]:
    metric_name, metric_type = expected_metric(dataset)
    return {
        "metric_name": metric_name,
        "metric_type": metric_type,
        "value": None,
        "num_examples": 0,
        "failed": True,
        "error": error,
    }


def evaluate_dataset(
    dataset: str,
    model: Any,
    tokenizer: Any,
    pack: DatasetPack,
    args: argparse.Namespace,
    run_dir: Path,
    output_root: Path,
    model_label: str,
    mask_scope: str,
) -> dict[str, Any]:
    if pack.error:
        return failed_metric(dataset, f"dataset_load_failed: {pack.error}")
    if not pack.rows:
        return failed_metric(dataset, "dataset_empty")
    if dataset == "gsm8k":
        return evaluate_gsm8k(model, tokenizer, pack, args, run_dir)
    if dataset == "humaneval":
        return evaluate_humaneval(model, tokenizer, pack, args, run_dir)
    if dataset == "math_reasoning":
        return evaluate_math_reasoning(model, tokenizer, pack, args, run_dir)
    if dataset == "mtbench":
        return evaluate_mtbench(
            model, tokenizer, pack, args, run_dir, output_root, model_label, mask_scope
        )
    if dataset == "dolly":
        return evaluate_dolly(model, tokenizer, pack, args, run_dir)
    return failed_metric(dataset, f"unknown dataset: {dataset}")


def make_run_dir(output_root: Path, model_label: str, mask_scope: str) -> Path:
    run_dir = output_root / "runs" / f"{model_label}_{mask_scope}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def empty_mask_stats(mask_scope: str) -> dict[str, Any]:
    return {
        "mask_scope": mask_scope,
        "target_modules": list(QUALITY_MASK_TARGETS.get(mask_scope, ())),
        "total_masked_weight_count": 0,
        "zeroed_weight_count": 0,
        "actual_sparsity": 0.0,
        "masked_module_names": [],
        "per_module": [],
        "tail_warnings": [],
    }


def run_one_scope(
    model_label: str,
    model_id: str,
    mask_scope: str,
    datasets: dict[str, DatasetPack],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = make_run_dir(output_root, model_label, mask_scope)
    run_config = {
        "model_label": model_label,
        "model_id": model_id,
        "mask_scope": mask_scope,
        "datasets": list(datasets),
        "dtype": args.dtype,
        "device": args.device,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "started_at": timestamp(),
    }
    write_json(run_dir / "run_config.json", run_config)

    try:
        dtype = dtype_from_arg(args.dtype)
        model, tokenizer = load_model_and_tokenizer(
            model_id,
            dtype,
            args.device,
            args.trust_remote_code,
            args.local_files_only,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"model_load_failed: {exc}"
        mask_stats = empty_mask_stats(mask_scope)
        write_json(run_dir / "mask_stats.json", mask_stats)
        metrics = {
            dataset: failed_metric(dataset, error)
            for dataset in datasets
        }
        write_json(run_dir / "metrics.json", {"failed": True, "error": error, "datasets": metrics})
        return {
            "run_dir": str(run_dir),
            "failed": True,
            "error": error,
            "mask_stats": mask_stats,
            "metrics": metrics,
        }

    try:
        if mask_scope == "none":
            mask_stats = empty_mask_stats(mask_scope)
        else:
            mask_stats = apply_24_mask_to_model(
                model,
                QUALITY_MASK_TARGETS[mask_scope],
                group_dim="in",
                skip_lm_head=True,
                skip_embeddings=True,
            )
            mask_stats["mask_scope"] = mask_scope
        write_json(run_dir / "mask_stats.json", mask_stats)

        if args.save_masked_model and mask_scope != "none":
            save_dir = run_dir / "masked_model"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        metrics: dict[str, Any] = {}
        for dataset_name, pack in datasets.items():
            started = time.time()
            try:
                metric = evaluate_dataset(
                    dataset_name,
                    model,
                    tokenizer,
                    pack,
                    args,
                    run_dir,
                    output_root,
                    model_label,
                    mask_scope,
                )
            except Exception as exc:  # noqa: BLE001
                metric = failed_metric(dataset_name, str(exc))
            metric["elapsed_sec"] = round(time.time() - started, 3)
            metric.setdefault("failed", False)
            metric.setdefault("error", "")
            metric.setdefault("skipped", False)
            metrics[dataset_name] = metric
            write_json(run_dir / "metrics.json", {"failed": False, "datasets": metrics})

        return {
            "run_dir": str(run_dir),
            "failed": False,
            "error": "",
            "mask_stats": mask_stats,
            "metrics": metrics,
        }
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def order_mask_scopes(scopes: list[str]) -> list[str]:
    unknown = [scope for scope in scopes if scope not in QUALITY_MASK_TARGETS]
    if unknown:
        raise ValueError(f"unknown mask scopes: {unknown}")
    ordered = []
    if "none" in scopes:
        ordered.append("none")
    ordered.extend([scope for scope in scopes if scope != "none"])
    return ordered


def metric_value(metric: dict[str, Any]) -> float | None:
    return safe_float(metric.get("value"))


def build_csv_rows(
    all_results: dict[tuple[str, str], dict[str, Any]],
    model_ids: dict[str, str],
    dtype: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dense_by_model_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for (model_label, mask_scope), result in all_results.items():
        if mask_scope != "none":
            continue
        for dataset, metric in result["metrics"].items():
            dense_by_model_dataset[(model_label, dataset)] = metric

    for (model_label, mask_scope), result in sorted(all_results.items()):
        mask_stats = result["mask_stats"]
        for dataset, metric in result["metrics"].items():
            dense_metric = dense_by_model_dataset.get((model_label, dataset), {})
            dense_value = metric_value(dense_metric)
            sparse_value = metric_value(metric)
            metric_type = metric.get("metric_type", "")
            delta = None
            ratio = None
            if dense_value is not None and sparse_value is not None:
                delta = sparse_value - dense_value
                if metric_type == "ppl" and dense_value:
                    ratio = sparse_value / dense_value
            rows.append(
                {
                    "model_label": model_label,
                    "model_id": model_ids.get(model_label, ""),
                    "mask_scope": mask_scope,
                    "dataset": dataset,
                    "metric_name": metric.get("metric_name", ""),
                    "metric_type": metric_type,
                    "dense_metric_value": dense_value,
                    "sparse_metric_value": sparse_value,
                    "delta_vs_dense": delta,
                    "ratio_vs_dense": ratio,
                    "num_examples": metric.get("num_examples", 0),
                    "ppl_mode": metric.get("ppl_mode", ""),
                    "accuracy_available": bool(metric.get("accuracy_available", metric_type == "accuracy" and sparse_value is not None)),
                    "pass_at_1_available": bool(metric.get("pass_at_1_available", metric_type == "pass_at_1" and sparse_value is not None)),
                    "dtype": dtype,
                    "actual_sparsity": mask_stats.get("actual_sparsity", 0.0),
                    "zeroed_weight_count": mask_stats.get("zeroed_weight_count", 0),
                    "total_masked_weight_count": mask_stats.get("total_masked_weight_count", 0),
                    "skipped": bool(metric.get("skipped", False) or metric.get("skipped_pass_at_1", False)),
                    "failed": bool(metric.get("failed", result.get("failed", False))),
                    "error": metric.get("error", result.get("error", "")),
                }
            )
    return rows


def write_csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    if math.isinf(number):
        return "inf"
    return f"{number:.6g}"


def risk_for_row(row: dict[str, Any]) -> str:
    if row.get("failed"):
        return "failed"
    metric_type = row.get("metric_type")
    delta = safe_float(row.get("delta_vs_dense"))
    ratio = safe_float(row.get("ratio_vs_dense"))
    dataset = row.get("dataset")
    if metric_type in ("accuracy", "pass_at_1") and delta is not None:
        if delta < -0.05 and dataset in ("gsm8k", "humaneval", "math_reasoning"):
            return "high risk"
    if metric_type == "ppl" and ratio is not None and ratio > 1.10:
        return "high risk"
    return ""


def write_summary_md(
    path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    smoke_cmd = (
        "cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators\n\n"
        "conda run -n spec python -u \\\n"
        "  examples/evaluate/eval-guidellm/scripts/evaluate_structured_24_quality.py \\\n"
        "  --smoke \\\n"
        "  --output-root examples/evaluate/eval-guidellm/results/structured_24_quality_smoke"
    )
    full_cmd = (
        "conda run -n spec python -u \\\n"
        "  examples/evaluate/eval-guidellm/scripts/evaluate_structured_24_quality.py \\\n"
        "  --models qwen3_8b,llama3_1_8b \\\n"
        "  --mask-scopes none,attn,ffn,all \\\n"
        "  --datasets gsm8k,humaneval,math_reasoning,mtbench,dolly \\\n"
        "  --gsm8k-num-examples 128 \\\n"
        "  --humaneval-num-examples 64 \\\n"
        "  --math-num-examples 128 \\\n"
        "  --dolly-num-examples 128 \\\n"
        "  --mtbench-num-examples 80 \\\n"
        "  --dtype bf16 \\\n"
        "  --output-root examples/evaluate/eval-guidellm/results/structured_24_quality_full"
    )
    actual_cmd = ""
    run_config_path = output_root / "run_config.json"
    if run_config_path.exists():
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            argv = run_config.get("argv")
            if isinstance(argv, list) and argv:
                actual_cmd = " ".join(shlex.quote(str(part)) for part in argv)
                actual_cmd = actual_cmd.replace(
                    "examples/evaluate/eval-guidellm/scripts/residual_24_feasibility.py quality",
                    "examples/evaluate/eval-guidellm/scripts/evaluate_structured_24_quality.py",
                )
        except Exception:  # noqa: BLE001
            actual_cmd = ""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Structured 2:4 Masked-Weight Quality Evaluation\n\n")
        handle.write(f"Output root: `{output_root.resolve()}`\n\n")
        handle.write(
            "This experiment compares dense weights against in-memory 2:4 masked "
            "weights while still running dense PyTorch/Transformers kernels. It "
            "is a quality-loss evaluation, not a real sparse-kernel speedup result.\n\n"
        )
        handle.write("## Commands\n\n")
        handle.write("Smoke:\n\n```bash\n" + smoke_cmd + "\n```\n\n")
        handle.write("Full:\n\n```bash\n" + full_cmd + "\n```\n\n")
        if actual_cmd:
            handle.write("Actual command for this result:\n\n```bash\n")
            handle.write("cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators\n")
            handle.write(actual_cmd + "\n```\n\n")
        handle.write("## Metric Rules\n\n")
        handle.write(
            "- GSM8K: final-answer exact-match accuracy only.\n"
            "- HumanEval: pass@1 via subprocess functional correctness only.\n"
            "- Math Reasoning: final-answer exact-match accuracy when extractable; PPL fallback only if not extractable.\n"
            "- MT-Bench: NLL/PPL only; dense-reference PPL is not an official MT-Bench judge score.\n"
            "- Dolly: target-response NLL/PPL only; no accuracy is reported.\n\n"
        )
        handle.write("## Results\n\n")
        handle.write(
            "| model | dataset | mask scope | metric | dense | masked | delta | ratio | examples | ppl mode | risk | error |\n"
        )
        handle.write("|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['model_label']} | {row['dataset']} | {row['mask_scope']} | "
                f"{row['metric_name']} | {format_value(row['dense_metric_value'])} | "
                f"{format_value(row['sparse_metric_value'])} | {format_value(row['delta_vs_dense'])} | "
                f"{format_value(row['ratio_vs_dense'])} | {row['num_examples']} | "
                f"{row.get('ppl_mode', '')} | {risk_for_row(row)} | {row.get('error', '')} |\n"
            )
        handle.write("\n## Masking\n\n")
        handle.write(
            "For every selected `nn.Linear.weight` shaped `[out_features, in_features]`, "
            "weights are grouped along `in_features` in consecutive groups of 4. "
            "The two largest absolute values are kept and the other two are zeroed. "
            "Tails shorter than 4 are left unmasked and reported in `mask_stats.json`.\n\n"
        )
        handle.write(
            "Mask scopes: `attn` masks q/k/v/o projections, `ffn` masks gate/up/down "
            "projections, and `all` masks both. `lm_head`, embeddings, norms, rotary "
            "embedding, and non-Linear modules are not masked.\n"
        )


def write_run_config(output_root: Path, args: argparse.Namespace, model_ids: dict[str, str]) -> None:
    data = {
        "argv": sys.argv,
        "models": parse_csv_list(args.models),
        "model_ids": model_ids,
        "mask_scopes": parse_csv_list(args.mask_scopes),
        "datasets": parse_csv_list(args.datasets),
        "dtype": args.dtype,
        "device": args.device,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "smoke": args.smoke,
        "created_at": timestamp(),
    }
    write_json(output_root / "run_config.json", data)


def summarize_existing(output_root: Path, dtype: str) -> None:
    config = json.loads((output_root / "run_config.json").read_text(encoding="utf-8"))
    model_ids = config.get("model_ids", QUALITY_DEFAULT_MODELS)
    all_results: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in sorted((output_root / "runs").glob("*_*")):
        run_config_path = run_dir / "run_config.json"
        metrics_path = run_dir / "metrics.json"
        mask_stats_path = run_dir / "mask_stats.json"
        if not run_config_path.exists() or not metrics_path.exists() or not mask_stats_path.exists():
            continue
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        mask_stats = json.loads(mask_stats_path.read_text(encoding="utf-8"))
        model_label = run_config["model_label"]
        mask_scope = run_config["mask_scope"]
        all_results[(model_label, mask_scope)] = {
            "failed": bool(metrics.get("failed", False)),
            "error": metrics.get("error", ""),
            "mask_stats": mask_stats,
            "metrics": metrics.get("datasets", {}),
        }
    rows = build_csv_rows(all_results, model_ids, dtype)
    write_csv_summary(output_root / "structured_24_quality.csv", rows)
    write_summary_md(output_root / "summary.md", rows, argparse.Namespace(), output_root)
    write_json(
        output_root / "all_metrics.json",
        {
            f"{model_label}/{mask_scope}": result
            for (model_label, mask_scope), result in all_results.items()
        },
    )


def configure_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.models = "qwen3_8b"
    args.mask_scopes = "none,all"
    args.datasets = "gsm8k,dolly"
    args.gsm8k_num_examples = 16
    args.dolly_num_examples = 16
    args.humaneval_num_examples = 16
    args.math_num_examples = 16
    args.mtbench_num_examples = 16



def run_quality(args: argparse.Namespace) -> None:
    configure_smoke(args)
    output_root = args.output_root or (
        QUALITY_RESULTS_ROOT / f"structured_24_quality_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summarize_existing:
        summarize_existing(output_root, args.dtype)
        print(output_root / "structured_24_quality.csv")
        print(output_root / "summary.md")
        return

    ensure_quality_dependencies()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model_ids = dict(QUALITY_DEFAULT_MODELS)
    model_ids.update(parse_model_id_overrides(args.model_id))
    selected_models = parse_csv_list(args.models)
    selected_scopes = order_mask_scopes(parse_csv_list(args.mask_scopes))
    selected_datasets = load_datasets(args)
    write_run_config(output_root, args, model_ids)

    all_results: dict[tuple[str, str], dict[str, Any]] = {}
    for model_label in selected_models:
        model_id = model_ids.get(model_label)
        if not model_id:
            for mask_scope in selected_scopes:
                all_results[(model_label, mask_scope)] = {
                    "failed": True,
                    "error": f"unknown model label: {model_label}",
                    "mask_stats": empty_mask_stats(mask_scope),
                    "metrics": {
                        dataset: failed_metric(dataset, f"unknown model label: {model_label}")
                        for dataset in selected_datasets
                    },
                }
            continue
        for mask_scope in selected_scopes:
            print(f"[INFO] Running model={model_label} mask_scope={mask_scope}", flush=True)
            result = run_one_scope(
                model_label,
                model_id,
                mask_scope,
                selected_datasets,
                args,
                output_root,
            )
            all_results[(model_label, mask_scope)] = result

    rows = build_csv_rows(all_results, model_ids, args.dtype)
    csv_path = output_root / "structured_24_quality.csv"
    summary_path = output_root / "summary.md"
    write_csv_summary(csv_path, rows)
    write_summary_md(summary_path, rows, args, output_root)
    write_json(
        output_root / "all_metrics.json",
        {
            f"{model_label}/{mask_scope}": result
            for (model_label, mask_scope), result in all_results.items()
        },
    )
    print(csv_path)
    print(summary_path)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    storage = subparsers.add_parser(
        "storage",
        help="Prototype lossless complementary 2:4 decomposition.",
    )
    storage.add_argument("--output-root", required=True, type=Path)
    storage.add_argument(
        "--model",
        action="append",
        type=parse_model,
        help="Model config as LABEL=PATH_OR_HF_ID. Defaults to Qwen3 and local Llama.",
    )
    storage.add_argument("--value-bits", type=int, default=16)
    storage.add_argument("--metadata-bits-per-group", type=int, default=4)
    storage.add_argument("--seed", type=int, default=1234)
    storage.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Random prototype dtype. float16 keeps memory bounded for 8B FFN shapes.",
    )
    storage.add_argument(
        "--allow-remote-config",
        action="store_true",
        help="Allow transformers to fetch configs if not present locally.",
    )
    storage.set_defaults(func=run_storage)

    barrier = subparsers.add_parser(
        "barrier",
        help="Analyze confidence-barrier residual-token ratios from traces.",
    )
    barrier.add_argument("inputs", nargs="*", type=Path, help="Trace files or roots.")
    barrier.add_argument("--output-root", required=True, type=Path)
    barrier.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    barrier.add_argument("--models", default=",".join(DEFAULT_MODELS))
    barrier.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    barrier.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    barrier.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    barrier.add_argument(
        "--no-drop-mode",
        choices=("default", "safe"),
        default="default",
        help="When no cumulative-confidence barrier is crossed, default uses h=0; safe uses h=K.",
    )
    barrier.set_defaults(func=run_barrier)

    verifier = subparsers.add_parser(
        "verifier",
        help="Collect verifier-detail timing into the residual 2:4 schema.",
    )
    verifier.add_argument("roots", nargs="+", type=Path, help="Motivation breakdown roots.")
    verifier.add_argument("--output-root", required=True, type=Path)
    verifier.add_argument("--models", default=",".join(DEFAULT_MODELS))
    verifier.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    verifier.add_argument("--batch-sizes", default=",".join(str(v) for v in DEFAULT_BATCHES))
    verifier.add_argument("--ks", default=",".join(str(v) for v in DEFAULT_KS))
    verifier.set_defaults(func=run_verifier)

    speedup = subparsers.add_parser(
        "speedup",
        help="Estimate theoretical residual 2:4 speedup from CSVs.",
    )
    speedup.add_argument("--verifier-breakdown", required=True, type=Path)
    speedup.add_argument("--residual-barrier", required=True, type=Path)
    speedup.add_argument("--output-root", required=True, type=Path)
    speedup.set_defaults(func=run_speedup)


    quality = subparsers.add_parser(
        "quality",
        help="Evaluate dense vs in-memory structured 2:4 masked model quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Evaluate quality loss from in-memory 2:4 masks on selected "
            "nn.Linear weights. This uses Transformers/PyTorch only; it does "
            "not use vLLM and does not claim real sparse-kernel speedup.\n\n"
            "Smoke example:\n"
            "  conda run -n spec python scripts/evaluate_structured_24_quality.py "
            "--smoke --output-root results/structured_24_quality_smoke\n\n"
            "Full example:\n"
            "  conda run -n spec python scripts/evaluate_structured_24_quality.py "
            "--models qwen3_8b,llama3_1_8b --mask-scopes none,attn,ffn,all "
            "--datasets gsm8k,humaneval,math_reasoning,mtbench,dolly "
            "--gsm8k-num-examples 128 --humaneval-num-examples 64 "
            "--math-num-examples 128 --dolly-num-examples 128 "
            "--mtbench-num-examples 80 --dtype bf16 "
            "--output-root results/structured_24_quality_full"
        ),
    )
    quality.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    quality.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Override model id/path as LABEL=MODEL_ID. Can be repeated.",
    )
    quality.add_argument("--mask-scopes", default="none,attn,ffn,all")
    quality.add_argument(
        "--datasets",
        default="gsm8k,humaneval,math_reasoning,mtbench,dolly",
    )
    quality.add_argument("--gsm8k-num-examples", type=int, default=128)
    quality.add_argument("--humaneval-num-examples", type=int, default=None)
    quality.add_argument("--math-num-examples", type=int, default=128)
    quality.add_argument("--mtbench-num-examples", type=int, default=80)
    quality.add_argument("--dolly-num-examples", type=int, default=128)
    quality.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    quality.add_argument("--device", default="cuda")
    quality.add_argument("--max-seq-len", type=int, default=2048)
    quality.add_argument("--output-root", type=Path, default=None)
    quality.add_argument("--smoke", action="store_true")
    quality.add_argument("--trust-remote-code", action="store_true")
    quality.add_argument("--local-files-only", action="store_true")
    quality.add_argument("--save-masked-model", action="store_true")
    quality.add_argument("--seed", type=int, default=42)
    quality.add_argument("--generation-batch-size", type=int, default=4)
    quality.add_argument("--humaneval-timeout", type=float, default=5.0)
    quality.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Regenerate structured_24_quality.csv and summary.md from existing run directories.",
    )
    quality.set_defaults(func=run_quality)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
