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

This script is analysis-only. It does not implement a fused 2:4 kernel and does
not claim measured end-to-end speedup.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
