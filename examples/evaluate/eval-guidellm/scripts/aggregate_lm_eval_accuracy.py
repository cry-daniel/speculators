#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

TASK_METRIC_PRIORITY = {
    "gsm8k_cot": [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match",
    ],
    "minerva_math500": [
        "exact_match,none",
        "exact_match",
        "score,none",
        "score",
    ],
    "gpqa_diamond_cot_zeroshot": [
        "exact_match,none",
        "exact_match",
        "acc,none",
        "acc",
    ],
    "ifeval": [
        "prompt_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_strict_acc,none",
        "inst_level_loose_acc,none",
    ],
    "humaneval_instruct": [
        "pass@1,create_test",
        "pass@1",
    ],
    "longbench_multi_news": [
        "score,none",
        "score",
        "rouge_score,none",
        "rouge_score",
    ],
}

METRIC_PRIORITY = [
    "score,none",
    "score",
    "qa_f1_score,none",
    "qa_f1_score",
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

TOKENIZER_CACHE: dict[str, Any] = {}
DENSE_REFERENCE_MODES = ("dense_baseline", "eagle3_dense", "dense_ar")
SPEC_DENSE_REFERENCE_MODES = ("dense_baseline", "eagle3_dense")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def failure_reason(run_dir: Path) -> str:
    priority_tokens = (
        "DatasetNotFoundError",
        "gated dataset",
        "Access to dataset",
        "Access to this dataset",
        "Cannot access gated repo",
        "CUDA out of memory",
        "out of memory",
        "RuntimeError:",
        "ValueError:",
    )
    fallback_tokens = (
        "Traceback",
        "ERROR",
        "OOM",
    )
    fallback = ""
    for name in ("lm_eval.log", "vllm_server.log"):
        path = run_dir / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if any(token in line for token in priority_tokens):
                return line.strip()[:500]
            if not fallback and any(token in line for token in fallback_tokens):
                fallback = line.strip()[:500]
    if fallback:
        return fallback
    return ""


def find_result_json(run_dir: Path) -> Path | None:
    candidates = [
        path
        for path in (run_dir / "lm_eval_output").rglob("*.json")
        if path.name != "run_meta.json"
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_samples_jsonl(run_dir: Path, task_name: str) -> Path | None:
    candidates = list((run_dir / "lm_eval_output").rglob(f"samples_{task_name}_*.jsonl"))
    if not candidates:
        candidates = list((run_dir / "lm_eval_output").rglob("samples_*.jsonl"))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_all_samples_jsonl(run_dir: Path, task_name: str) -> list[Path]:
    candidates = list((run_dir / "lm_eval_output").rglob(f"samples_{task_name}_*.jsonl"))
    if not candidates:
        candidates = list((run_dir / "lm_eval_output").rglob("samples_*.jsonl"))
    candidates.sort()
    return candidates


def choose_metric(task_result: dict[str, Any]) -> tuple[str, float | None]:
    for key in METRIC_PRIORITY:
        if key in task_result and not key.endswith("_stderr"):
            try:
                return key, float(task_result[key])
            except (TypeError, ValueError):
                return key, None
    for key, value in task_result.items():
        if key.endswith("_stderr") or key in {"alias", "samples"}:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def choose_metrics(task_name: str, task_result: dict[str, Any]) -> list[tuple[str, float | None]]:
    candidates = TASK_METRIC_PRIORITY.get(task_name, [])
    if not candidates:
        candidates = TASK_METRIC_PRIORITY.get(task_name.split(",", 1)[0], [])
    out: list[tuple[str, float | None]] = []
    for key in candidates:
        if key in task_result:
            try:
                out.append((key, float(task_result[key])))
            except (TypeError, ValueError):
                out.append((key, None))
    if out:
        # Most tasks have one official score. IFEval and LongBench intentionally
        # report multiple official metrics when available.
        if task_name == "ifeval" or task_name == "longbench_multi_news":
            return out
        return [out[0]]
    metric, score = choose_metric(task_result)
    return [(metric, score)] if metric else []


def sample_count(data: dict[str, Any], task: str) -> int | None:
    for section in ("n-samples", "n_samples"):
        values = data.get(section)
        if isinstance(values, dict):
            raw = values.get(task)
            if isinstance(raw, dict):
                raw = raw.get("effective") or raw.get("original")
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return None


def sample_key(sample: dict[str, Any]) -> str:
    for key in ("doc_id", "doc_hash", "prompt_hash"):
        if key in sample:
            return f"{key}:{sample[key]}"
    return json.dumps(sample.get("doc", sample), sort_keys=True, ensure_ascii=False)


def sample_correct(sample: dict[str, Any], metric: str) -> bool | None:
    metric_name = metric.split(",", 1)[0] if metric else ""
    candidates = [metric_name, "exact_match", "acc", "acc_norm", "pass@1"]
    for key in candidates:
        if key and key in sample:
            value = sample[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value) > 0.5
    return None


def generation_text(sample: dict[str, Any]) -> str | None:
    resps = sample.get("resps")
    if (
        isinstance(resps, list)
        and len(resps) == 1
        and isinstance(resps[0], list)
        and len(resps[0]) == 1
        and isinstance(resps[0][0], str)
    ):
        return resps[0][0]
    return None


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * pct
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (pos - low))


def load_tokenizer(tokenizer_path: str) -> Any | None:
    if not tokenizer_path:
        return None
    if tokenizer_path in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[tokenizer_path]
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception:
        tokenizer = None
    TOKENIZER_CACHE[tokenizer_path] = tokenizer
    return tokenizer


def output_length_stats(run_dir: Path, task_name: str, meta: dict[str, Any]) -> dict[str, Any]:
    tokenizer = load_tokenizer(str(meta.get("tokenizer_path") or meta.get("model_path") or ""))
    max_new_tokens = int(meta.get("max_new_tokens") or 0)
    token_counts: list[int] = []
    char_counts: list[int] = []
    seen: set[tuple[str, str]] = set()
    for sample_path in find_all_samples_jsonl(run_dir, task_name):
        with sample_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                dedup_key = (
                    sample_path.name,
                    str(sample.get("doc_id", sample.get("prompt_hash", len(seen)))),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                text = generation_text(sample)
                if text is None:
                    continue
                char_counts.append(len(text))
                if tokenizer is None:
                    token_counts.append(len(text.split()))
                else:
                    token_counts.append(
                        len(tokenizer.encode(text, add_special_tokens=False))
                    )
    clipped = (
        sum(1 for count in token_counts if max_new_tokens and count >= max_new_tokens)
        if token_counts
        else 0
    )
    non_clipped = len(token_counts) - clipped
    return {
        "output_sample_count": len(token_counts),
        "avg_output_tokens": statistics.fmean(token_counts) if token_counts else None,
        "p50_output_tokens": percentile(token_counts, 0.50),
        "p90_output_tokens": percentile(token_counts, 0.90),
        "max_output_tokens": max(token_counts) if token_counts else None,
        "avg_output_chars": statistics.fmean(char_counts) if char_counts else None,
        "max_token_clipped_count": clipped,
        # local-completions samples do not persist finish_reason; this is the
        # observable non-clipped completion count (EOS or task stop sequence).
        "eos_finished_count": non_clipped if token_counts else None,
    }


def summarize_token_dense_stats(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "token_dense_stats.jsonl"
    latest: dict[str, Any] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") in {"verify_token_mask_summary", "verify_token_mask"}:
                if int(record.get("total_draft_tokens") or 0) > 0:
                    latest = record
    return latest


def summarize_sr24_stats(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stats_path = run_dir / "speclink_sr24_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
        if isinstance(stats, dict):
            out.update(
                {
                    "sr24_mode": stats.get("mode"),
                    "sr24_backend": stats.get("backend"),
                    "sr24_residual_backend": stats.get("residual_backend"),
                    "sr24_residual_device": stats.get("residual_device"),
                    "sr24_require_gpu_residual": stats.get(
                        "require_gpu_residual"
                    ),
                    "sr24_threshold": stats.get("threshold"),
                    "sr24_prefix_threshold": stats.get(
                        "selective_prefix_threshold"
                    ),
                    "sr24_all_corrected_dense_fastpath": stats.get(
                        "all_corrected_dense_fastpath"
                    ),
                    "sr24_selective_correct_non_draft": stats.get(
                        "selective_correct_non_draft"
                    ),
                    "sr24_selective_non_draft_policy": stats.get(
                        "selective_non_draft_policy"
                    ),
                    "sr24_selective_residual_policy": stats.get(
                        "selective_residual_policy"
                    ),
                    "sr24_selective_extra_after_low": stats.get(
                        "selective_extra_after_low"
                    ),
                    "sr24_selective_min_prefix_residual": stats.get(
                        "selective_min_prefix_residual"
                    ),
                    "sr24_selective_max_residual_draft_rows": stats.get(
                        "selective_max_residual_draft_rows"
                    ),
                    "sr24_low_confidence_cap_by_risk": stats.get(
                        "low_confidence_cap_by_risk"
                    ),
                    "sr24_early_dense_tokens": stats.get("early_dense_tokens"),
                    "sr24_sync_mask_state": stats.get("sync_mask_state"),
                    "sr24_static_mask_state": stats.get("static_mask_state"),
                    "sr24_static_mask_buffer": stats.get("static_mask_buffer"),
                    "sr24_batched_mask_builder": stats.get(
                        "batched_mask_builder"
                    ),
                    "sr24_batched_uniform_direct": stats.get(
                        "batched_uniform_direct"
                    ),
                    "sr24_gate_up_split": stats.get("gate_up_split"),
                    "sr24_row_routed_mlp": stats.get("row_routed_mlp"),
                    "sr24_row_routed_mlp_min_dense_rows": stats.get(
                        "row_routed_mlp_min_dense_rows"
                    ),
                    "sr24_row_routed_mlp_max_dense_rows": stats.get(
                        "row_routed_mlp_max_dense_rows"
                    ),
                    "sr24_row_routed_mlp_max_base_rows": stats.get(
                        "row_routed_mlp_max_base_rows"
                    ),
                    "sr24_force_cudagraph_none_for_mixed": stats.get(
                        "force_cudagraph_none_for_mixed"
                    ),
                    "sr24_static_mask_buffer_capacity": stats.get(
                        "static_mask_buffer_capacity"
                    ),
                    "sr24_residual_bucket_size": stats.get("residual_bucket_size"),
                    "sr24_residual_bucket_priority": stats.get(
                        "residual_bucket_priority"
                    ),
                    "sr24_route_bucket_rows": stats.get("route_bucket_rows"),
                    "sr24_route_all_residual_rows": stats.get(
                        "route_all_residual_rows"
                    ),
                    "sr24_route_reuse_base_output": stats.get(
                        "route_reuse_base_output"
                    ),
                    "sr24_route_dense_fallback_fraction": stats.get(
                        "route_dense_fallback_fraction"
                    ),
                    "sr24_triton_route_assembly": stats.get(
                        "triton_route_assembly"
                    ),
                    "sr24_target_leafs": ",".join(stats.get("target_leafs") or []),
                    "sr24_residual_target_leafs":
                    ",".join(stats.get("residual_target_leafs") or []),
                    "sr24_base_only_layer_ids":
                    ",".join(
                        str(item) for item in (stats.get("base_only_layer_ids") or [])
                    ),
                    "sr24_base_only_layer_ids_by_leaf":
                    ";".join(
                        f"{leaf}={','.join(str(item) for item in layer_ids)}"
                        for leaf, layer_ids in (
                            stats.get("base_only_layer_ids_by_leaf") or {}
                        ).items()
                    ),
                    "sr24_residual_layer_ids_by_leaf":
                    ";".join(
                        f"{leaf}={','.join(str(item) for item in layer_ids)}"
                        for leaf, layer_ids in (
                            stats.get("residual_layer_ids_by_leaf") or {}
                        ).items()
                    ),
                    "sr24_residual_out_chunk": stats.get("residual_out_chunk"),
                    "sr24_extract_chunk_rows": stats.get(
                        "residual_extract_chunk_rows"
                    ),
                    "sr24_residual_extract_cpu_fallback_chunks": stats.get(
                        "residual_extract_cpu_fallback_chunks"
                    ),
                    "sr24_residual_extract_cpu_fallback_module_count": stats.get(
                        "residual_extract_cpu_fallback_module_count"
                    ),
                    "sr24_residual_backend_counts": json.dumps(
                        stats.get("residual_backend_counts") or {},
                        sort_keys=True,
                    ),
                    "sr24_residual_device_counts": json.dumps(
                        stats.get("residual_device_counts") or {},
                        sort_keys=True,
                    ),
                    "sr24_residual_cpu_module_count": stats.get(
                        "residual_cpu_module_count"
                    ),
                    "sr24_residual_cuda_module_count": stats.get(
                        "residual_cuda_module_count"
                    ),
                    "sr24_compressed_residual_runtime_on_gpu": stats.get(
                        "compressed_residual_runtime_on_gpu"
                    ),
                    "sr24_compressed_residual_non_gpu_modules": ",".join(
                        stats.get("compressed_residual_non_gpu_modules") or []
                    ),
                    "sr24_module_count_attached": stats.get("module_count_attached"),
                    "sr24_storage_over_dense": stats.get("storage_over_dense"),
                    "sr24_actual_weight_storage_bytes": stats.get(
                        "actual_weight_storage_bytes"
                    ),
                    "sr24_sparse_metadata_bytes": stats.get("sparse_metadata_bytes"),
                    "sr24_mask_metadata_bytes": stats.get("mask_metadata_bytes"),
                    "sr24_mask_path": stats.get("mask_path"),
                    "sr24_mask_cache_method": stats.get("mask_cache_method"),
                }
            )
    events_path = run_dir / "speclink_sr24_events.jsonl"
    latest_summary: dict[str, Any] = {}
    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") in {"sr24_verify_summary", "sr24_verify_mask"}:
                    latest_summary = record
    if latest_summary:
        out.update(
            {
                "sr24_total_draft_tokens": latest_summary.get("total_draft_tokens"),
                "sr24_total_valid_draft_tokens": latest_summary.get(
                    "total_valid_draft_tokens"
                ),
                "sr24_non_draft_tokens": latest_summary.get("non_draft_tokens"),
                "sr24_residual_draft_tokens": latest_summary.get(
                    "residual_draft_tokens"
                ),
                "sr24_base_only_draft_tokens": latest_summary.get(
                    "base_only_draft_tokens"
                ),
                "sr24_residual_non_draft_tokens": latest_summary.get(
                    "residual_non_draft_tokens"
                ),
                "sr24_base_only_non_draft_tokens": latest_summary.get(
                    "base_only_non_draft_tokens"
                ),
                "sr24_early_residual_draft_tokens": latest_summary.get(
                    "early_residual_draft_tokens"
                ),
                "sr24_early_residual_non_draft_tokens": latest_summary.get(
                    "early_residual_non_draft_tokens"
                ),
                "sr24_missing_score_tokens": latest_summary.get(
                    "missing_score_tokens"
                ),
                "sr24_sync_reduced_stats": latest_summary.get(
                    "sync_reduced_stats"
                ),
                "sr24_stats_exact": latest_summary.get("stats_exact"),
                "sr24_residual_draft_fraction": latest_summary.get(
                    "residual_draft_fraction"
                ),
                "sr24_bucket_calls": latest_summary.get("bucket_calls"),
                "sr24_bucket_candidate_rows": latest_summary.get(
                    "bucket_candidate_rows"
                ),
                "sr24_bucket_active_rows": latest_summary.get(
                    "bucket_active_rows"
                ),
                "sr24_bucket_total_rows": latest_summary.get("bucket_total_rows"),
                "sr24_bucket_residual_requested_rows": latest_summary.get(
                    "bucket_residual_requested_rows"
                ),
                "sr24_bucket_candidate_rows_per_call": latest_summary.get(
                    "bucket_candidate_rows_per_call"
                ),
                "sr24_bucket_active_rows_per_call": latest_summary.get(
                    "bucket_active_rows_per_call"
                ),
                "sr24_bucket_active_fraction_of_requested": latest_summary.get(
                    "bucket_active_fraction_of_requested"
                ),
                "sr24_residual_non_draft_fraction": (
                    (
                        latest_summary.get("residual_non_draft_tokens")
                        / latest_summary.get("non_draft_tokens")
                    )
                    if latest_summary.get("non_draft_tokens")
                    else None
                ),
            }
        )
    return out


def load_sample_map(run_dir: Path, task_name: str, metric: str) -> dict[str, dict[str, Any]]:
    sample_path = find_samples_jsonl(run_dir, task_name)
    if sample_path is None:
        return {}
    samples: dict[str, dict[str, Any]] = {}
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            correct = sample_correct(sample, metric)
            samples[sample_key(sample)] = {
                "correct": correct,
                "doc_id": sample.get("doc_id"),
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
                "target": sample.get("target"),
                "filtered_resps": sample.get("filtered_resps"),
                "resps": sample.get("resps"),
                "sample_path": str(sample_path),
            }
    return samples


def rows_from_runs(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(output_dir.rglob("run_meta.json")):
        run_dir = meta_path.parent
        meta = load_json(meta_path)
        result_path = find_result_json(run_dir)
        status = meta.get("status", "")
        meta_error = meta.get("error") or ""
        log_error = failure_reason(run_dir) if status != "ok" else ""
        if not result_path:
            task_name = str(meta.get("task") or "")
            rows.append(
                {
                    **meta,
                    "task_result_name": task_name,
                    "metric": "",
                    "score": "",
                    "samples": "",
                    "result_path": "",
                    "status": status or "missing_result",
                    "error": meta_error or failure_reason(run_dir),
                }
            )
            continue
        data = load_json(result_path)
        results = data.get("results", {})
        token_dense_stats = summarize_token_dense_stats(run_dir)
        sr24_stats = summarize_sr24_stats(run_dir)
        row_status = status
        row_error = meta_error or log_error
        if (
            str(meta.get("mode") or "").startswith("token_dense_")
            and status == "ok"
            and not token_dense_stats
        ):
            row_status = "failed"
            row_error = (
                "missing token_dense_stats.jsonl; token-dense routing was not "
                "observed during verification"
            )
        if (
            str(meta.get("mode") or "") in {"base_only_24", "all_corrected_24", "speclink_t08"}
            and status == "ok"
            and not sr24_stats
        ):
            row_status = "failed"
            row_error = "missing speclink_sr24 stats/events"
        for task_name, task_result in results.items():
            if not isinstance(task_result, dict):
                continue
            length_stats = output_length_stats(run_dir, task_name, meta)
            for metric, score in choose_metrics(task_name, task_result):
                rows.append(
                    {
                        **meta,
                        **length_stats,
                        "run_dir": str(run_dir),
                        "task_result_name": task_name,
                        "metric": metric,
                        "score": score,
                        "samples": sample_count(data, task_name),
                        "result_path": str(result_path),
                        "status": row_status,
                        "error": row_error,
                        "token_dense_threshold": token_dense_stats.get("threshold"),
                        "token_dense_dense_draft_fraction": token_dense_stats.get(
                            "dense_draft_fraction"
                        ),
                        "token_dense_sparse_draft_fraction": token_dense_stats.get(
                            "sparse_draft_fraction"
                        ),
                        "token_dense_missing_score_tokens": token_dense_stats.get(
                            "missing_score_tokens"
                        ),
                        **sr24_stats,
                    }
                )
    for skip_path in sorted(output_dir.rglob("skip.json")):
        skip = load_json(skip_path)
        rows.append(
            {
                "model_label": skip.get("model_label"),
                "mode": skip.get("mode"),
                "mode_group": "",
                "task": skip.get("task"),
                "task_result_name": skip.get("task"),
                "metric": "",
                "score": "",
                "samples": "",
                "status": "skipped",
                "error": skip.get("reason", ""),
                "result_path": "",
            }
        )
    return rows


def add_dense_comparisons(rows: list[dict[str, Any]]) -> None:
    dense_scores: dict[tuple[str, str, str], float] = {}
    dense_ar_scores: dict[tuple[str, str, str], float] = {}
    dense_spec_scores: dict[tuple[str, str, str], float] = {}
    for reference_mode in DENSE_REFERENCE_MODES:
        for row in rows:
            if row.get("mode") != reference_mode:
                continue
            score = row.get("score")
            key = (
                str(row.get("model_label")),
                str(row.get("task_result_name")),
                str(row.get("metric")),
            )
            if (
                key not in dense_scores
                and isinstance(score, (int, float))
                and math.isfinite(float(score))
            ):
                dense_scores[key] = float(score)
    for row in rows:
        mode = str(row.get("mode") or "")
        score = row.get("score")
        if not (isinstance(score, (int, float)) and math.isfinite(float(score))):
            continue
        key = (
            str(row.get("model_label")),
            str(row.get("task_result_name")),
            str(row.get("metric")),
        )
        if mode == "dense_ar" and key not in dense_ar_scores:
            dense_ar_scores[key] = float(score)
        if mode in SPEC_DENSE_REFERENCE_MODES and key not in dense_spec_scores:
            dense_spec_scores[key] = float(score)
    for row in rows:
        score = row.get("score")
        key = (
            str(row.get("model_label")),
            str(row.get("task_result_name")),
            str(row.get("metric")),
        )
        dense = dense_scores.get(key)
        dense_ar = dense_ar_scores.get(key)
        dense_spec = dense_spec_scores.get(key)
        row["dense_score"] = dense
        row["dense_ar_score"] = dense_ar
        row["dense_spec_score"] = dense_spec
        if isinstance(score, (int, float)) and dense is not None:
            row["delta_pp_vs_dense"] = (float(score) - dense) * 100.0
        else:
            row["delta_pp_vs_dense"] = ""
        if isinstance(score, (int, float)) and dense_ar is not None:
            row["delta_pp_vs_dense_ar"] = (float(score) - dense_ar) * 100.0
        else:
            row["delta_pp_vs_dense_ar"] = ""
        if isinstance(score, (int, float)) and dense_spec is not None:
            row["delta_pp_vs_dense_spec"] = (float(score) - dense_spec) * 100.0
        else:
            row["delta_pp_vs_dense_spec"] = ""


def add_paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    sample_maps: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}

    for reference_mode in DENSE_REFERENCE_MODES:
        for row in rows:
            if row.get("mode") == reference_mode and row.get("status") == "ok":
                key = (
                    str(row.get("model_label")),
                    str(row.get("task_result_name")),
                    str(row.get("metric")),
                )
                dense_rows.setdefault(key, row)

    paired_events: list[dict[str, Any]] = []
    for row in rows:
        task_name = str(row.get("task_result_name") or row.get("task") or "")
        model_label = str(row.get("model_label") or "")
        mode = str(row.get("mode") or "")
        run_dir = Path(str(row.get("run_dir") or ""))
        metric = str(row.get("metric") or "")
        row_key = (model_label, task_name, metric)
        sample_key_tuple = (model_label, mode, task_name)
        if row.get("status") != "ok" or not run_dir:
            continue
        sample_maps[sample_key_tuple] = load_sample_map(run_dir, task_name, metric)
        dense_row = dense_rows.get(row_key)
        if dense_row is None:
            continue
        dense_mode = str(dense_row.get("mode") or "")
        dense_tuple = (model_label, dense_mode, task_name)
        if dense_tuple not in sample_maps:
            dense_maps_run_dir = Path(str(dense_row.get("run_dir") or ""))
            sample_maps[dense_tuple] = load_sample_map(
                dense_maps_run_dir, task_name, str(dense_row.get("metric") or "")
            )
        dense_samples = sample_maps.get(dense_tuple, {})
        experiment_samples = sample_maps.get(sample_key_tuple, {})
        common_keys = sorted(set(dense_samples) & set(experiment_samples))

        counts = {
            "paired_samples": 0,
            "dense_correct": 0,
            "experimental_correct": 0,
            "both_correct": 0,
            "dense_correct_experimental_wrong": 0,
            "dense_wrong_experimental_correct": 0,
            "both_wrong": 0,
        }
        for key in common_keys:
            dense_correct = dense_samples[key].get("correct")
            experiment_correct = experiment_samples[key].get("correct")
            if dense_correct is None or experiment_correct is None:
                continue
            counts["paired_samples"] += 1
            counts["dense_correct"] += int(bool(dense_correct))
            counts["experimental_correct"] += int(bool(experiment_correct))
            if dense_correct and experiment_correct:
                counts["both_correct"] += 1
            elif dense_correct and not experiment_correct:
                counts["dense_correct_experimental_wrong"] += 1
                paired_events.append(
                    {
                        "event": "dense_correct_experimental_wrong",
                        "model_label": model_label,
                        "task": task_name,
                        "mode": mode,
                        "sample_key": key,
                        "dense": dense_samples[key],
                        "experimental": experiment_samples[key],
                    }
                )
            elif not dense_correct and experiment_correct:
                counts["dense_wrong_experimental_correct"] += 1
                paired_events.append(
                    {
                        "event": "dense_wrong_experimental_correct",
                        "model_label": model_label,
                        "task": task_name,
                        "mode": mode,
                        "sample_key": key,
                        "dense": dense_samples[key],
                        "experimental": experiment_samples[key],
                    }
                )
            else:
                counts["both_wrong"] += 1

        row.update(counts)
        dense_correct_count = counts["dense_correct"]
        if dense_correct_count:
            row["dense_correct_retention"] = counts["both_correct"] / dense_correct_count
        else:
            row["dense_correct_retention"] = ""

    return paired_events


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_label",
        "mode",
        "mode_group",
        "task",
        "task_result_name",
        "metric",
        "dense_score",
        "dense_ar_score",
        "dense_spec_score",
        "score",
        "delta_pp_vs_dense",
        "delta_pp_vs_dense_ar",
        "delta_pp_vs_dense_spec",
        "samples",
        "paired_samples",
        "dense_correct",
        "experimental_correct",
        "dense_correct_experimental_wrong",
        "dense_wrong_experimental_correct",
        "both_wrong",
        "dense_correct_retention",
        "status",
        "spec_acceptance_rate",
        "spec_accepted_tokens",
        "spec_draft_tokens",
        "output_sample_count",
        "avg_output_tokens",
        "p50_output_tokens",
        "p90_output_tokens",
        "max_output_tokens",
        "max_token_clipped_count",
        "eos_finished_count",
        "token_dense_threshold",
        "token_dense_dense_draft_fraction",
        "token_dense_sparse_draft_fraction",
        "token_dense_missing_score_tokens",
        "sr24_preset",
        "sr24_mode",
        "sr24_backend",
        "sr24_residual_backend",
        "sr24_residual_device",
        "sr24_require_gpu_residual",
        "sr24_threshold",
        "sr24_prefix_threshold",
        "sr24_all_corrected_dense_fastpath",
        "sr24_selective_correct_non_draft",
        "sr24_selective_non_draft_policy",
        "sr24_selective_residual_policy",
        "sr24_selective_extra_after_low",
        "sr24_selective_min_prefix_residual",
        "sr24_selective_max_residual_draft_rows",
        "sr24_low_confidence_cap_by_risk",
        "sr24_early_dense_tokens",
        "sr24_sync_mask_state",
        "sr24_static_mask_state",
        "sr24_static_mask_buffer",
        "sr24_batched_mask_builder",
        "sr24_batched_uniform_direct",
        "sr24_gate_up_split",
        "sr24_row_routed_mlp",
        "sr24_row_routed_mlp_min_dense_rows",
        "sr24_row_routed_mlp_max_dense_rows",
        "sr24_row_routed_mlp_max_base_rows",
        "sr24_force_cudagraph_none_for_mixed",
        "sr24_static_mask_buffer_capacity",
        "sr24_residual_bucket_size",
        "sr24_residual_bucket_priority",
        "sr24_route_bucket_rows",
        "sr24_route_all_residual_rows",
        "sr24_route_reuse_base_output",
        "sr24_route_dense_fallback_fraction",
        "sr24_triton_route_assembly",
        "sr24_target_leafs",
        "sr24_residual_target_leafs",
        "sr24_base_only_layer_ids",
        "sr24_base_only_layer_ids_by_leaf",
        "sr24_residual_layer_ids_by_leaf",
        "sr24_residual_out_chunk",
        "sr24_extract_chunk_rows",
        "sr24_residual_extract_cpu_fallback_chunks",
        "sr24_residual_extract_cpu_fallback_module_count",
        "sr24_residual_backend_counts",
        "sr24_residual_device_counts",
        "sr24_residual_cpu_module_count",
        "sr24_residual_cuda_module_count",
        "sr24_compressed_residual_runtime_on_gpu",
        "sr24_compressed_residual_non_gpu_modules",
        "sr24_residual_draft_fraction",
        "sr24_residual_non_draft_fraction",
        "sr24_residual_draft_tokens",
        "sr24_base_only_draft_tokens",
        "sr24_bucket_calls",
        "sr24_bucket_candidate_rows",
        "sr24_bucket_active_rows",
        "sr24_bucket_total_rows",
        "sr24_bucket_residual_requested_rows",
        "sr24_bucket_candidate_rows_per_call",
        "sr24_bucket_active_rows_per_call",
        "sr24_bucket_active_fraction_of_requested",
        "sr24_early_residual_draft_tokens",
        "sr24_early_residual_non_draft_tokens",
        "sr24_total_draft_tokens",
        "sr24_total_valid_draft_tokens",
        "sr24_non_draft_tokens",
        "sr24_residual_non_draft_tokens",
        "sr24_base_only_non_draft_tokens",
        "sr24_missing_score_tokens",
        "sr24_sync_reduced_stats",
        "sr24_stats_exact",
        "sr24_storage_over_dense",
        "sr24_actual_weight_storage_bytes",
        "sr24_sparse_metadata_bytes",
        "sr24_mask_metadata_bytes",
        "sr24_module_count_attached",
        "sr24_mask_cache_method",
        "sr24_mask_path",
        "result_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return ""
        return str(value)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# lm-eval Accuracy Report\n\n")
        handle.write(
            "Modes are dense target-only, dense EAGLE3, activation-aware, "
            "token_dense_t00-t10, and SR24 (`base_only_24`, "
            "`all_corrected_24`, `speclink_t08`).\n\n"
        )
        handle.write(
            "The `Dense` reference column uses `dense_baseline` when available, "
            "then `eagle3_dense`, otherwise `dense_ar`.\n\n"
        )
        handle.write(
            "`AR ref` is pure target-model autoregressive serving. `Spec ref` "
            "is dense EAGLE3 speculative serving (`dense_baseline` or "
            "`eagle3_dense`). SR24 accuracy debugging should inspect both, "
            "because dense speculative serving can differ from AR on reasoning "
            "samples even with `temperature=0`.\n\n"
        )
        handle.write(
            "Output length stats are computed from raw lm-eval completion samples. "
            "`eos_finished_count` is inferred as non-clipped completions because "
            "lm-eval local-completions samples do not persist API finish_reason.\n\n"
        )
        handle.write(
            "`Pair reg` counts dense-correct / experimental-wrong samples; "
            "`Pair imp` counts dense-wrong / experimental-correct samples. "
            "Use these paired counts to distinguish true quality preservation "
            "from aggregate-score cancellation.\n\n"
        )
        handle.write("| Task | Metric | Model | Mode | Dense | AR ref | Spec ref | Experimental | Delta pp | Delta vs AR | Delta vs Spec | Samples | Pair n | Pair reg | Pair imp | Dense retain | Avg out tok | P90 out tok | Clipped | Spec accept | Dense route | SR24 draft residual | SR24 non-draft residual | SR24 bucket actual/requested | SR24 bucket rows/call | SR24 early guard | SR24 policy | SR24 gate/up split | SR24 target leafs | SR24 residual leafs | SR24 base-only layers | SR24 residual layers | SR24 chunk | SR24 backend | Status |\n")
        handle.write("|------|--------|-------|------|------:|-------:|---------:|-------------:|---------:|------------:|--------------:|--------:|-------:|---------:|---------:|-------------:|------------:|------------:|--------:|------------:|------------:|-------------------:|-----------------------:|----------------------------:|---------------------:|-----------------|-------------|--------------------|-------------------|---------------------|-----------------------|------------------------|-----------:|-------------|--------|\n")
        for row in sorted(rows, key=lambda item: (str(item.get("task_result_name")), str(item.get("metric")), str(item.get("model_label")), str(item.get("mode")))):
            protocol = "generative" if "generative" in str(row.get("task")) or str(row.get("metric")).startswith("exact_match") else "official"
            sr24_policy = fmt(row.get("sr24_selective_residual_policy"))
            extra_after_low = row.get("sr24_selective_extra_after_low")
            if (
                sr24_policy
                and sr24_policy != ""
                and extra_after_low not in {None, "", 0, "0"}
            ):
                sr24_policy = f"{sr24_policy}+extra{extra_after_low}"
            prefix_threshold = row.get("sr24_prefix_threshold")
            if sr24_policy == "prefix_confidence" and prefix_threshold not in {
                None,
                "",
            }:
                sr24_policy = f"{sr24_policy}@{fmt(prefix_threshold)}"
            non_draft_policy = fmt(row.get("sr24_selective_non_draft_policy"))
            if non_draft_policy and non_draft_policy not in {"", "auto"}:
                sr24_policy = (
                    f"{sr24_policy}; non_draft={non_draft_policy}"
                    if sr24_policy else f"non_draft={non_draft_policy}"
                )
            early_guard = fmt(row.get("sr24_early_dense_tokens"))
            if early_guard:
                early_guard = (
                    f"{early_guard}; "
                    f"{fmt(row.get('sr24_early_residual_draft_tokens'))}/"
                    f"{fmt(row.get('sr24_early_residual_non_draft_tokens'))}"
                )
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("task_result_name") or row.get("task") or ""),
                        str(row.get("metric") or protocol),
                        str(row.get("model_label") or ""),
                        str(row.get("mode") or ""),
                        fmt(row.get("dense_score")),
                        fmt(row.get("dense_ar_score")),
                        fmt(row.get("dense_spec_score")),
                        fmt(row.get("score")),
                        fmt(row.get("delta_pp_vs_dense")),
                        fmt(row.get("delta_pp_vs_dense_ar")),
                        fmt(row.get("delta_pp_vs_dense_spec")),
                        fmt(row.get("samples")),
                        fmt(row.get("paired_samples")),
                        fmt(row.get("dense_correct_experimental_wrong")),
                        fmt(row.get("dense_wrong_experimental_correct")),
                        fmt(row.get("dense_correct_retention")),
                        fmt(row.get("avg_output_tokens")),
                        fmt(row.get("p90_output_tokens")),
                        fmt(row.get("max_token_clipped_count")),
                        fmt(row.get("spec_acceptance_rate")),
                        fmt(row.get("token_dense_dense_draft_fraction")),
                        fmt(row.get("sr24_residual_draft_fraction")),
                        fmt(row.get("sr24_residual_non_draft_fraction")),
                        (
                            f"{fmt(row.get('sr24_bucket_active_rows'))}/"
                            f"{fmt(row.get('sr24_bucket_residual_requested_rows'))}"
                        ),
                        (
                            f"{fmt(row.get('sr24_bucket_active_rows_per_call'))}/"
                            f"{fmt(row.get('sr24_bucket_candidate_rows_per_call'))}"
                        ),
                        early_guard,
                        sr24_policy,
                        fmt(row.get("sr24_gate_up_split")),
                        fmt(row.get("sr24_target_leafs")),
                        fmt(row.get("sr24_residual_target_leafs")),
                        fmt(
                            row.get("sr24_base_only_layer_ids_by_leaf")
                            or row.get("sr24_base_only_layer_ids")
                        ),
                        fmt(row.get("sr24_residual_layer_ids_by_leaf")),
                        fmt(row.get("sr24_residual_out_chunk")),
                        (
                            str(row.get("sr24_backend") or "")
                            + (
                                f"/{row.get('sr24_residual_backend')}"
                                if row.get("sr24_residual_backend")
                                else ""
                            )
                            + (
                                f"@{row.get('sr24_residual_device')}"
                                if row.get("sr24_residual_device")
                                else ""
                            )
                        ),
                        str(row.get("status") or ""),
                    ]
                )
                + " |\n"
            )
        skipped = [row for row in rows if row.get("status") == "skipped"]
        if skipped:
            handle.write("\n## Skipped\n\n")
            for row in skipped:
                handle.write(f"- {row.get('model_label')} / {row.get('mode')} / {row.get('task')}: {row.get('error')}\n")
        failed = [row for row in rows if row.get("status") == "failed"]
        if failed:
            handle.write("\n## Failed\n\n")
            for row in failed:
                handle.write(
                    f"- {row.get('model_label')} / {row.get('mode')} / "
                    f"{row.get('task')}: {row.get('error')}\n"
                )


def token_dense_index(mode: str) -> int | None:
    if not mode.startswith("token_dense_t"):
        return None
    try:
        return int(mode.rsplit("t", 1)[1])
    except ValueError:
        return None


def write_figures(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = output_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (fig_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return

    def is_longbench_row(row: dict[str, Any]) -> bool:
        task = str(row.get("task_result_name") or row.get("task") or "").lower()
        return task.startswith("longbench")

    td_rows = [
        row
        for row in rows
        if token_dense_index(str(row.get("mode") or "")) is not None
        and isinstance(row.get("score"), (int, float))
        and not is_longbench_row(row)
    ]
    x_labels = [f"t{i:02d}" for i in range(11)]
    x_values = list(range(11))

    def grouped(field: str) -> dict[tuple[str, str, str], list[float | None]]:
        out: dict[tuple[str, str, str], list[float | None]] = {}
        for row in td_rows:
            idx = token_dense_index(str(row.get("mode") or ""))
            if idx is None or idx < 0 or idx > 10:
                continue
            key = (
                str(row.get("model_label") or ""),
                str(row.get("task_result_name") or row.get("task") or ""),
                str(row.get("metric") or ""),
            )
            out.setdefault(key, [None] * 11)
            value = row.get(field)
            out[key][idx] = float(value) if isinstance(value, (int, float)) else None
        return out

    def dense_score_by_group() -> dict[tuple[str, str, str], float]:
        out: dict[tuple[str, str, str], float] = {}
        for row in td_rows:
            key = (
                str(row.get("model_label") or ""),
                str(row.get("task_result_name") or row.get("task") or ""),
                str(row.get("metric") or ""),
            )
            dense_score = row.get("dense_score")
            if (
                key not in out
                and isinstance(dense_score, (int, float))
                and math.isfinite(float(dense_score))
            ):
                out[key] = float(dense_score)
        return out

    score_groups = grouped("score")
    dense_score_groups = dense_score_by_group()
    for model in sorted({key[0] for key in score_groups}):
        keys = [key for key in sorted(score_groups) if key[0] == model]
        if not keys:
            continue
        cols = 2
        rows_n = math.ceil(len(keys) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(12, max(3, rows_n * 2.6)))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, key in zip(axes_list, keys):
            values = score_groups[key]
            ax.plot(x_values, values, marker="o")
            dense_score = dense_score_groups.get(key)
            if dense_score is not None:
                ax.axhline(
                    dense_score,
                    color="0.35",
                    linestyle="--",
                    linewidth=1.2,
                    label="dense",
                )
                ax.legend(fontsize=7, loc="best")
            ax.set_title(f"{key[1]} / {key[2]}", fontsize=9)
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_ylabel("absolute score")
            ax.grid(True, alpha=0.25)
        for ax in axes_list[len(keys) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"token_dense_absolute_score_{model}.png", dpi=180)
        plt.close(fig)

    for field, ylabel, filename in [
        ("avg_output_tokens", "avg output tokens", "avg_output_tokens_vs_threshold.png"),
        ("p90_output_tokens", "p90 output tokens", "p90_output_tokens_vs_threshold.png"),
        (
            "token_dense_dense_draft_fraction",
            "dense draft fraction",
            "dense_routing_fraction_vs_threshold.png",
        ),
    ]:
        groups = grouped(field)
        keys = sorted(groups)
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        for key in keys:
            values = groups[key]
            if not any(value is not None for value in values):
                continue
            ax.plot(x_values, values, marker="o", label=f"{key[0]}:{key[1]}:{key[2]}")
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("token-dense threshold")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    rows = rows_from_runs(output_dir)
    add_dense_comparisons(rows)
    paired_events = add_paired_comparisons(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "paired_regressions.jsonl").open("w", encoding="utf-8") as handle:
        for event in paired_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_report(output_dir / "report.md", rows)
    write_figures(output_dir, rows)
    print(output_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
