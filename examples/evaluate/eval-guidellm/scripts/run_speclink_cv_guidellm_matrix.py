#!/usr/bin/env python3
"""Run and summarize the SpecLink-CV serving benchmark matrix.

The runner is intentionally explicit: every case writes its config, server log,
benchmark result, SpecLink-CV event/profile JSONL, and a row in the summary
tables. It supports finite-request GuideLLM diagnostics and closed-loop
steady-state saturated throughput mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.speclink_cv.env_check import collect as collect_env
from tools.speclink_cv.env_check import write_markdown as write_env_markdown


RESULT_SUBDIRS = [
    "00_env",
    "01_impl_notes",
    "02_unit_tests",
    "03_confidence_calibration",
    "04_baselines",
    "05_cv_ablation",
    "06_scheduler_queue",
    "07_roofline_packing",
    "08_figures",
    "09_reports",
    "logs",
    "patches",
    "scripts",
]

FOCUSED_TESTS = [
    "test_chunk_decision",
    "test_state_machine",
    "test_async_queue",
    "test_roofline_packing",
    "test_correctness_smoke",
    "test_vllm_runtime_config",
    "test_sampler_draft_accept_eps",
    "test_math_answer_extraction",
]


METHODS: dict[str, dict[str, str]] = {
    "pure_vllm": {"spec": "0", "cv": "0", "confidence": "0", "async": "0", "roofline": "0"},
    "eagle3_oneshot": {"spec": "1", "cv": "0", "confidence": "0", "async": "0", "roofline": "0"},
    "cv_half_sync_simple": {"spec": "1", "cv": "1", "confidence": "0", "async": "0", "roofline": "0"},
    "cv_half_sync_roofline": {"spec": "1", "cv": "1", "confidence": "0", "async": "0", "roofline": "1"},
    "cv_half_async_simple": {"spec": "1", "cv": "1", "confidence": "0", "async": "1", "roofline": "0"},
    "cv_half_async_roofline": {"spec": "1", "cv": "1", "confidence": "0", "async": "1", "roofline": "1"},
    "cv_half_async_staged_simple": {"spec": "1", "cv": "1", "confidence": "0", "async": "1", "roofline": "0", "staged": "1"},
    "cv_conf_sync_simple": {"spec": "1", "cv": "1", "confidence": "1", "async": "0", "roofline": "0"},
    "cv_conf_sync_roofline": {"spec": "1", "cv": "1", "confidence": "1", "async": "0", "roofline": "1"},
    "cv_conf_async_simple": {"spec": "1", "cv": "1", "confidence": "1", "async": "1", "roofline": "0"},
    "cv_conf_async_roofline": {"spec": "1", "cv": "1", "confidence": "1", "async": "1", "roofline": "1"},
}

ANALYSIS_PROFILE_MAX_ROWS = int(
    os.environ.get("SPECLINK_CV_ANALYSIS_MAX_PROFILE_ROWS", "2000")
)
MATH_RELIABLE_MAX_TOKENS = int(
    os.environ.get("SPECLINK_CV_MATH_RELIABLE_MAX_TOKENS", "256")
)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_env_items(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env expects a non-empty KEY, got {item!r}")
        env[key] = value
    return env


def model_registry() -> dict[str, dict[str, str]]:
    models_root = REPO_ROOT.parent / "models"
    return {
        "qwen3_8b": {
            "base": env_default(
                "QWEN3_8B_BASE_MODEL",
                env_default("QWEN3_8B_MODEL", "Qwen/Qwen3-8B"),
            ),
            "speculator": env_default(
                "QWEN3_8B_EAGLE3_SPECULATOR_MODEL",
                env_default(
                    "EAGLE3_SPECULATOR_MODEL",
                    str(models_root / "qwen3-8b-eagle3-speculator"),
                ),
            ),
        },
        "llama3_1_8b": {
            "base": env_default(
                "LLAMA3_1_8B_BASE_MODEL",
                env_default("LLAMA3_1_8B_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            ),
            "speculator": env_default(
                "LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL",
                env_default(
                    "LLAMA_EAGLE3_SPECULATOR_MODEL",
                    str(models_root / "llama-3.1-8b-eagle3-speculator"),
                ),
            ),
        },
    }


def dataset_registry() -> dict[str, str]:
    return {
        "math": env_default(
            "MATH_REASONING_DATASET",
            env_default("MATH_DATASET", str(EVAL_DIR / "data/math_reasoning.jsonl")),
        ),
        "mtbench": env_default("MTBENCH_DATASET", str(EVAL_DIR / "data/mt_bench.jsonl")),
    }


def scalar(metrics: dict[str, Any], name: str, field: str = "mean") -> float | str:
    value = metrics.get(name, {})
    if not isinstance(value, dict):
        return ""
    section = value.get("successful") or value.get("total") or {}
    if field.startswith("p"):
        return section.get("percentiles", {}).get(field, "")
    return section.get(field, "")


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def mean_or_empty(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def parse_guidellm(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return {}
    bench = benchmarks[0]
    metrics = bench.get("metrics", {})
    requests = bench.get("requests", {})
    successful = requests.get("successful", [])
    errored = requests.get("errored", [])
    incomplete = requests.get("incomplete", [])
    return {
        "throughput": scalar(metrics, "output_tokens_per_second"),
        "total_tokens_per_second": scalar(metrics, "tokens_per_second"),
        "requests_per_second": scalar(metrics, "requests_per_second"),
        "actual_average_batch_size": scalar(metrics, "request_concurrency"),
        "ttft_p95": scalar(metrics, "time_to_first_token_ms", "p95"),
        "itl_p95": scalar(metrics, "inter_token_latency_ms", "p95"),
        "itl_p99": scalar(metrics, "inter_token_latency_ms", "p99"),
        "e2e_p95": scalar(metrics, "request_latency", "p95"),
        "e2e_p99": scalar(metrics, "request_latency", "p99"),
        "successful_requests": len(successful) if isinstance(successful, list) else "",
        "errored_requests": len(errored) if isinstance(errored, list) else "",
        "incomplete_requests": len(incomplete) if isinstance(incomplete, list) else "",
        "duration": bench.get("duration", ""),
        "benchmark_start_time": bench.get("start_time", ""),
        "benchmark_end_time": bench.get("end_time", ""),
    }


def parse_steady_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "throughput": data.get("output_tokens_per_second", ""),
        "total_tokens_per_second": data.get("output_tokens_per_second", ""),
        "requests_per_second": "",
        "actual_average_batch_size": data.get("concurrency", ""),
        "ttft_p95": "",
        "itl_p95": "",
        "itl_p99": "",
        "e2e_p95": "",
        "e2e_p99": "",
        "successful_requests": data.get("requests_completed", ""),
        "errored_requests": data.get("requests_errored", ""),
        "incomplete_requests": data.get("unfinished_worker_threads", ""),
        "duration": data.get("measurement_s", ""),
        "benchmark_start_time": data.get("measurement_start_epoch", ""),
        "benchmark_end_time": data.get("measurement_end_epoch", ""),
        "measurement_output_tokens": data.get("measurement_output_tokens", ""),
        "warmup_s": data.get("warmup_s", ""),
        "measurement_s": data.get("measurement_s", ""),
        "cooldown_s": data.get("cooldown_s", ""),
        "counting_mode": data.get("counting_mode", ""),
        "workload_hash": data.get("workload_hash", ""),
        "ignore_eos": data.get("ignore_eos", ""),
    }


def load_successful_outputs(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return []
    requests = benchmarks[0].get("requests", {})
    successful = requests.get("successful", [])
    if not isinstance(successful, list):
        return []
    outputs: list[str] = []
    for request in successful:
        if isinstance(request, dict):
            outputs.append(str(request.get("output", "")))
    return outputs


def request_fingerprint(request: dict[str, Any]) -> str:
    request_args = request.get("request_args", "")
    if request_args:
        try:
            parsed = json.loads(request_args)
            body = parsed.get("body", parsed)
            messages = body.get("messages")
            prompt = body.get("prompt")
            if messages is not None:
                return json.dumps({"messages": messages}, sort_keys=True, ensure_ascii=False)
            if prompt is not None:
                return json.dumps({"prompt": prompt}, sort_keys=True, ensure_ascii=False)
            return json.dumps(body, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(request_args)
    metrics = request.get("input_metrics", {})
    return json.dumps(
        {
            "prompt_tokens": request.get("prompt_tokens", ""),
            "text_tokens": metrics.get("text_tokens", ""),
            "text_characters": metrics.get("text_characters", ""),
        },
        sort_keys=True,
    )


def load_successful_output_map(path: Path) -> dict[str, str]:
    if not path.exists():
        sibling = path.with_name("steady_state_requests.jsonl")
        if sibling.exists():
            buckets: dict[str, list[str]] = defaultdict(list)
            with sibling.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    request = json.loads(line)
                    if request.get("ok"):
                        buckets[request_fingerprint(request)].append(
                            str(request.get("output", ""))
                        )
            output_map: dict[str, str] = {}
            for key, values in buckets.items():
                if len(values) == 1:
                    output_map[key] = values[0]
                else:
                    for idx, value in enumerate(values):
                        output_map[f"{key}#duplicate={idx}"] = value
            return output_map
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return {}
    requests = benchmarks[0].get("requests", {})
    successful = requests.get("successful", [])
    if not isinstance(successful, list):
        return {}
    buckets: dict[str, list[str]] = defaultdict(list)
    for request in successful:
        if isinstance(request, dict):
            buckets[request_fingerprint(request)].append(str(request.get("output", "")))
    output_map: dict[str, str] = {}
    for key, values in buckets.items():
        if len(values) == 1:
            output_map[key] = values[0]
        else:
            for idx, value in enumerate(values):
                output_map[f"{key}#duplicate={idx}"] = value
    return output_map


def load_successful_request_map(path: Path) -> dict[str, dict[str, Any]]:
    requests = load_successful_requests(path)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        buckets[request_fingerprint(request)].append(request)
    request_map: dict[str, dict[str, Any]] = {}
    for key, values in buckets.items():
        if len(values) == 1:
            request_map[key] = values[0]
        else:
            for idx, value in enumerate(values):
                request_map[f"{key}#duplicate={idx}"] = value
    return request_map


def load_successful_requests(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sibling = path.with_name("steady_state_requests.jsonl")
        if sibling.exists():
            rows: list[dict[str, Any]] = []
            with sibling.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    request = json.loads(line)
                    if request.get("ok"):
                        rows.append(request)
            return rows
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return []
    requests = benchmarks[0].get("requests", {})
    successful = requests.get("successful", [])
    return [request for request in successful if isinstance(request, dict)]


def normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def request_prompt_text(request: dict[str, Any]) -> str:
    raw = request.get("request_args", "")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        body = parsed.get("body", parsed)
        messages = body.get("messages")
        if messages:
            chunks: list[str] = []
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            chunks.append(str(part.get("text", "")))
                        elif isinstance(part, str):
                            chunks.append(part)
                else:
                    chunks.append(str(content))
            return normalize_prompt_text("\n".join(chunks))
        if body.get("prompt") is not None:
            return normalize_prompt_text(str(body.get("prompt", "")))
    except Exception:
        pass
    return normalize_prompt_text(str(raw))


def load_dataset_by_prompt(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = normalize_prompt_text(str(item.get("prompt", "")))
            if prompt:
                out[prompt] = item
    return out


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = value.replace(",", "")
    value = re.sub(r"^[\$£€]\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .。")
    return value


def extract_reference_answer(item: dict[str, Any]) -> str:
    refs = item.get("reference") or []
    text = refs[0] if isinstance(refs, list) and refs else str(refs)
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        return normalize_answer(match.group(1))
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_answer(numbers[-1]) if numbers else ""


def extract_predicted_answer(output: str) -> str:
    match = re.search(r"####\s*([^\n]+)", output)
    if match:
        return normalize_answer(match.group(1))
    match = re.search(r"\\boxed\{([^{}]+)\}", output)
    if match:
        return normalize_answer(match.group(1))
    number_re = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"
    answer_patterns = [
        rf"(?:final\s+)?(?:answer|result)\s*(?:is|=|:)\s*({number_re})",
        rf"(?:the\s+)?answer\s+is\s*[:=]?\s*({number_re})",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1])

    # Prefer the last number in conclusion sentences. A broad "so ..." regex
    # must not return the first operand in a final equation such as
    # "So, total is 24 + 10 = 34."
    conclusion_re = re.compile(
        r"(?:therefore|thus|hence|so)\b[^\n.。]*(?:[.。]|\n|$)",
        flags=re.IGNORECASE,
    )
    for segment in reversed(conclusion_re.findall(output)):
        numbers = re.findall(number_re, segment)
        if numbers:
            return normalize_answer(numbers[-1])

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        numbers = re.findall(number_re, line)
        if numbers:
            return normalize_answer(numbers[-1])
    numbers = re.findall(number_re, output)
    return normalize_answer(numbers[-1]) if numbers else ""


def normalized_text_similarity(left: str, right: str) -> float:
    left_norm = normalize_prompt_text(left.lower())
    right_norm = normalize_prompt_text(right.lower())
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def quality_for_run(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("status") != "ok":
        return {}
    run_dir = Path(str(row.get("output_dir", "")))
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    dataset_label = str(row.get("dataset", config.get("dataset_label", "")))
    dataset_path = Path(str(config.get("dataset", "")))
    requests = load_successful_requests(run_dir / "guidellm_results.json")
    max_tokens = safe_float(row.get("max_tokens")) or safe_float(config.get("max_tokens"))
    if (max_tokens is None or max_tokens == 0) and requests:
        try:
            raw = requests[0].get("request_args", "")
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            body = parsed.get("body", parsed)
            max_tokens = safe_float(body.get("max_tokens"))
        except Exception:
            max_tokens = None
    if max_tokens is None:
        max_tokens = 0.0
    outputs = [str(request.get("output", "")) for request in requests]
    non_empty = sum(1 for output in outputs if output.strip())
    quality_reliable = max_tokens <= 0 or max_tokens >= MATH_RELIABLE_MAX_TOKENS
    base: dict[str, Any] = {
        "quality_dataset": dataset_label,
        "quality_requests": len(requests),
        "quality_non_empty_rate": non_empty / len(requests) if requests else "",
        "quality_max_tokens": max_tokens,
        "quality_reliable": int(quality_reliable),
    }
    if dataset_label == "math":
        dataset = load_dataset_by_prompt(dataset_path)
        evaluable = 0
        correct = 0
        missing_pred = 0
        unmatched = 0
        for request in requests:
            item = dataset.get(request_prompt_text(request))
            if not item:
                unmatched += 1
                continue
            ref = extract_reference_answer(item)
            pred = extract_predicted_answer(str(request.get("output", "")))
            if not ref:
                continue
            evaluable += 1
            if not pred:
                missing_pred += 1
            if pred and pred == ref:
                correct += 1
        score = correct / evaluable if evaluable else ""
        return {
            **base,
            "quality_metric": "math_answer_em",
            "quality_score": score,
            "quality_correct": correct,
            "quality_evaluable": evaluable,
            "quality_unmatched": unmatched,
            "quality_missing_prediction": missing_pred,
            "quality_note": (
                "max_tokens_too_short_for_reliable_math"
                if not quality_reliable
                else "reference_answer_exact_match_unbounded"
                if max_tokens <= 0
                else "reference_answer_exact_match"
            ),
        }
    if dataset_label == "mtbench":
        return {
            **base,
            "quality_metric": "mtbench_proxy_no_reference",
            "quality_score": "",
            "quality_correct": "",
            "quality_evaluable": "",
            "quality_unmatched": "",
            "quality_missing_prediction": "",
            "quality_note": (
                "reference_null_no_judge_score; using baseline similarity proxy"
            ),
        }
    return {
        **base,
        "quality_metric": "unsupported_dataset",
        "quality_score": "",
        "quality_note": "no_quality_metric_for_dataset",
    }


def dataset_path_for_run(row: dict[str, Any]) -> Path | None:
    run_dir = Path(str(row.get("output_dir", "")))
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    dataset = config.get("dataset")
    return Path(str(dataset)) if dataset else None


def score_math_requests(
    dataset_by_prompt: dict[str, dict[str, Any]],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluable = 0
    correct = 0
    missing_pred = 0
    unmatched = 0
    for request in requests:
        item = dataset_by_prompt.get(request_prompt_text(request))
        if not item:
            unmatched += 1
            continue
        ref = extract_reference_answer(item)
        pred = extract_predicted_answer(str(request.get("output", "")))
        if not ref:
            continue
        evaluable += 1
        if not pred:
            missing_pred += 1
        if pred and pred == ref:
            correct += 1
    return {
        "score": correct / evaluable if evaluable else "",
        "correct": correct,
        "evaluable": evaluable,
        "unmatched": unmatched,
        "missing_prediction": missing_pred,
    }


def read_profile_rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    rows: list[dict[str, Any]] = []
    truncated = False
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if ANALYSIS_PROFILE_MAX_ROWS > 0 and len(rows) >= ANALYSIS_PROFILE_MAX_ROWS:
                truncated = True
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows, truncated


def parse_profile(path: Path, k: int) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows, truncated = read_profile_rows(path)
    log_limit_reached = any(row.get("event") == "jsonl_limit_reached" for row in rows)
    chunk_scheduled = [row for row in rows if row.get("event") == "verify_chunk_scheduled"]
    prefix_scheduled = [row for row in chunk_scheduled if row.get("phase") == "prefix"]
    suffix_scheduled = [row for row in chunk_scheduled if row.get("phase") == "suffix"]
    results = [row for row in rows if row.get("event") == "verify_chunk_result"]
    prefix_results = [row for row in results if row.get("phase") == "prefix"]
    dense_realign_results = [
        row
        for row in results
        if row.get("phase") == "dense_realign"
        and row.get("result") == "draft_dropped_for_dense_realign"
    ]
    staged_suffix_registered = [
        row for row in rows if row.get("event") == "staged_suffix_draft_registered"
    ]
    schedule_steps = [row for row in rows if row.get("event") == "schedule_step"]
    queue_steps = [row for row in rows if row.get("event") == "async_queue_step"]
    queue_waits = [
        float(row.get("queue_wait_ms", 0.0))
        for row in prefix_scheduled
        if row.get("queue_wait_ms") is not None
    ]
    decisions = [row for row in rows if row.get("event") == "verify_chunk_decision"]
    selected_h = [float(row.get("chunk_len", 0.0)) for row in prefix_scheduled]
    prefix_accept_counts = [
        float(row.get("num_accepted", 0.0)) for row in prefix_results
    ]
    skipped = sum(float(row.get("skipped_suffix_tokens", 0.0)) for row in prefix_results)
    extra = sum(float(row.get("extra_tlm_forward", 0.0)) for row in prefix_results)
    extra += sum(float(row.get("extra_tlm_forward", 1.0)) for row in dense_realign_results)
    verify_chunk_tokens = sum(
        float(row.get("chunk_len", 0.0)) for row in chunk_scheduled
    )
    verify_target_tokens_est = sum(
        float(row.get("scheduled_tokens_for_request", row.get("chunk_len", 0.0)))
        for row in chunk_scheduled
    )
    accepted_prefix = sum(1 for row in prefix_results if row.get("result") == "accepted_requeue_suffix")
    rejected_prefix = sum(1 for row in prefix_results if row.get("result") == "rejected_skip_suffix")
    fallback_count = sum(
        1
        for row in rows
        if row.get("fallback_reason")
        or "fallback" in str(row.get("reason", ""))
    )
    fallback_denominator = len(prefix_scheduled) + len(decisions)
    dispatch_steps = [row for row in queue_steps if "dispatch_count" in row]
    dispatch_counts = [
        float(row.get("dispatch_count", 0.0)) for row in dispatch_steps
    ]
    dispatch_tokens = [
        float(row.get("dispatch_tokens", 0.0)) for row in dispatch_steps
    ]
    dispatch_utils = [
        float(row.get("predicted_utilization", 0.0))
        for row in dispatch_steps
        if row.get("predicted_utilization") is not None
    ]
    dispatch_token_utils = [
        float(row.get("token_budget_utilization", 0.0))
        for row in dispatch_steps
        if row.get("token_budget_utilization") is not None
    ]
    dispatch_seq_utils = [
        float(row.get("seq_budget_utilization", 0.0))
        for row in dispatch_steps
        if row.get("seq_budget_utilization") is not None
    ]
    underfilled_dispatches = 0
    for row in dispatch_steps:
        dispatch_count = float(row.get("dispatch_count", 0.0))
        util = safe_float(row.get("predicted_utilization"))
        threshold = safe_float(row.get("util_threshold")) or 0.0
        if dispatch_count > 0 and util is not None and util < threshold:
            underfilled_dispatches += 1
    no_dispatches = sum(1 for value in dispatch_counts if value == 0)
    singleton_dispatches = sum(1 for value in dispatch_counts if value == 1)
    scheduled_spec_tokens = [
        float(row.get("scheduled_spec_tokens", 0.0)) for row in schedule_steps
    ]
    scheduled_spec_reqs = [
        float(row.get("scheduled_spec_reqs", 0.0)) for row in schedule_steps
    ]
    prefix_chunks_per_step = [
        float(row.get("prefix_chunks", 0.0)) for row in schedule_steps
    ]
    prefix_chunk_tokens_per_step = [
        float(row.get("prefix_chunk_tokens", 0.0)) for row in schedule_steps
    ]
    draft_tokens_generated_est = sum(
        float(row.get("chunk_len", 0.0))
        if row.get("staged_drafting")
        else float(row.get("k", 0.0))
        for row in prefix_scheduled
    ) + sum(
        float(row.get("registered_suffix_len", row.get("new_draft_len", 0.0)) or 0.0)
        for row in staged_suffix_registered
    )
    draft_tokens_full_k_est = sum(
        float(row.get("k", 0.0)) for row in prefix_scheduled
    )
    draft_tokens_accepted_est = sum(float(row.get("num_accepted", 0.0)) for row in results)
    draft_tokens_discarded_est = max(
        0.0, draft_tokens_generated_est - draft_tokens_accepted_est
    )

    return {
        "prefix_scheduled_count": len(prefix_scheduled) if prefix_scheduled else "",
        "prefix_result_count": len(prefix_results) if prefix_results else "",
        "suffix_scheduled_count": len(suffix_scheduled) if prefix_scheduled else "",
        "selected_h_avg": sum(selected_h) / len(selected_h) if selected_h else "",
        "selected_h_p95": percentile(selected_h, 95),
        "prefix_accepted_tokens_avg": mean_or_empty(prefix_accept_counts),
        "verify_chunk_tokens_scheduled": verify_chunk_tokens if chunk_scheduled else "",
        "verify_target_tokens_scheduled_est": (
            verify_target_tokens_est if chunk_scheduled else ""
        ),
        "verify_chunk_token_ratio_vs_oneshot_est": (
            verify_chunk_tokens / max(k * len(prefix_scheduled), 1)
            if prefix_scheduled
            else ""
        ),
        "verify_target_token_ratio_vs_oneshot_est": (
            verify_target_tokens_est / max((k + 1) * len(prefix_scheduled), 1)
            if prefix_scheduled
            else ""
        ),
        "skipped_suffix_tokens": skipped if prefix_results else "",
        "skipped_suffix_ratio": skipped / max(k * len(prefix_results), 1) if prefix_results else "",
        "extra_tlm_forwards_per_request": extra / len(prefix_results) if prefix_results else "",
        "dense_realign_steps": len(dense_realign_results),
        "queue_wait_p50": percentile(queue_waits, 50),
        "queue_wait_p95": percentile(queue_waits, 95),
        "queue_wait_p99": percentile(queue_waits, 99),
        "fallback_ratio": (
            fallback_count / max(fallback_denominator, 1)
            if fallback_denominator
            else ""
        ),
        "prefix_accepted_ratio": accepted_prefix / len(prefix_results) if prefix_results else "",
        "prefix_rejected_ratio": rejected_prefix / len(prefix_results) if prefix_results else "",
        "suffix_scheduled_ratio": len(suffix_scheduled) / len(prefix_results) if prefix_results else "",
        "verifier_calls_per_generated_token": (
            len(chunk_scheduled) / max(sum(float(row.get("chunk_len", 0.0)) for row in results), 1.0)
            if results
            else ""
        ),
        "staged_drafting_prefix_count": sum(
            1 for row in prefix_scheduled if row.get("staged_drafting")
        ),
        "staged_suffix_registered_count": len(staged_suffix_registered),
        "draft_tokens_generated_est": (
            draft_tokens_generated_est if prefix_scheduled else ""
        ),
        "draft_tokens_full_k_est": draft_tokens_full_k_est if prefix_scheduled else "",
        "draft_tokens_saved_by_staging_est": (
            max(0.0, draft_tokens_full_k_est - draft_tokens_generated_est)
            if prefix_scheduled
            else ""
        ),
        "draft_tokens_discarded_est": (
            draft_tokens_discarded_est if prefix_scheduled else ""
        ),
        "draft_discard_ratio_est": (
            draft_tokens_discarded_est / draft_tokens_generated_est
            if draft_tokens_generated_est
            else ""
        ),
        "async_queue_steps": len(queue_steps),
        "prefix_dispatch_count_avg": mean_or_empty(dispatch_counts),
        "prefix_dispatch_count_p95": percentile(dispatch_counts, 95),
        "prefix_dispatch_tokens_avg": mean_or_empty(dispatch_tokens),
        "prefix_dispatch_tokens_p95": percentile(dispatch_tokens, 95),
        "prefix_dispatch_util_avg": mean_or_empty(dispatch_utils),
        "prefix_dispatch_util_p95": percentile(dispatch_utils, 95),
        "prefix_dispatch_token_util_avg": mean_or_empty(dispatch_token_utils),
        "prefix_dispatch_seq_util_avg": mean_or_empty(dispatch_seq_utils),
        "prefix_underfilled_dispatch_ratio": (
            underfilled_dispatches / len(dispatch_steps) if dispatch_steps else ""
        ),
        "prefix_singleton_dispatch_ratio": (
            singleton_dispatches / len(dispatch_steps) if dispatch_steps else ""
        ),
        "prefix_no_dispatch_ratio": (
            no_dispatches / len(dispatch_steps) if dispatch_steps else ""
        ),
        "scheduled_spec_tokens_per_step": mean_or_empty(scheduled_spec_tokens),
        "scheduled_spec_reqs_per_step": mean_or_empty(scheduled_spec_reqs),
        "prefix_chunks_per_step": mean_or_empty(prefix_chunks_per_step),
        "prefix_chunk_tokens_per_step": mean_or_empty(prefix_chunk_tokens_per_step),
        "actual_scheduled_tokens_per_step": (
            sum(float(row.get("total_num_scheduled_tokens", 0.0)) for row in schedule_steps)
            / len(schedule_steps)
            if schedule_steps
            else ""
        ),
        "actual_scheduled_seqs_per_step": (
            sum(
                float(row.get("scheduled_running_reqs", 0.0))
                + float(row.get("scheduled_new_reqs", 0.0))
                + float(row.get("scheduled_resumed_reqs", 0.0))
                for row in schedule_steps
            )
            / len(schedule_steps)
            if schedule_steps
            else ""
        ),
        "profile_events": len(rows),
        "profile_log_limit_reached": int(log_limit_reached),
        "profile_analysis_truncated": int(truncated),
        "profile_analysis_max_rows": ANALYSIS_PROFILE_MAX_ROWS,
    }


def parse_acceptance(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("[") and line.endswith("]"):
            return line
    return ""


def _percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def gpu_util_summary(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {f"{prefix}sample_count": 0}
    return {
        f"{prefix}util": statistics.mean(values),
        f"{prefix}util_p50": _percentile(values, 0.50),
        f"{prefix}util_p95": _percentile(values, 0.95),
        f"{prefix}busy_ratio_50": sum(1 for item in values if item >= 50.0)
        / len(values),
        f"{prefix}busy_ratio_80": sum(1 for item in values if item >= 80.0)
        / len(values),
        f"{prefix}sample_count": len(values),
    }


def parse_gpu_util(
    path: Path,
    *,
    active_start_time: float | None = None,
    active_end_time: float | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {}
    gpu_utils: list[float] = []
    active_gpu_utils: list[float] = []
    powers: list[float] = []
    active_powers: list[float] = []
    memories: list[float] = []
    active_memories: list[float] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith("timestamp"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            sample_time: float | None = None
            try:
                sample_time = datetime.strptime(
                    parts[0], "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp()
            except ValueError:
                pass
            active = (
                sample_time is not None
                and active_start_time is not None
                and active_end_time is not None
                and active_start_time <= sample_time <= active_end_time
            )
            util: float | None = None
            power: float | None = None
            memory: float | None = None
            try:
                util = float(parts[1])
                gpu_utils.append(util)
            except ValueError:
                pass
            try:
                power = float(parts[2])
                powers.append(power)
            except ValueError:
                pass
            try:
                memory = float(parts[3])
                memories.append(memory)
            except ValueError:
                pass
            if active:
                if util is not None:
                    active_gpu_utils.append(util)
                if power is not None:
                    active_powers.append(power)
                if memory is not None:
                    active_memories.append(memory)
    if not gpu_utils:
        return {"gpu_sample_count": 0}
    metrics = {
        **gpu_util_summary(gpu_utils, "gpu_"),
        "gpu_power_avg": statistics.mean(powers) if powers else "",
        "gpu_memory_avg_mib": statistics.mean(memories) if memories else "",
    }
    if active_start_time is not None and active_end_time is not None:
        metrics.update(gpu_util_summary(active_gpu_utils, "gpu_active_"))
        metrics["gpu_active_power_avg"] = (
            statistics.mean(active_powers) if active_powers else ""
        )
        metrics["gpu_active_memory_avg_mib"] = (
            statistics.mean(active_memories) if active_memories else ""
        )
    return metrics


def row_from_run_dir(
    case: dict[str, Any],
    run_dir: Path,
    status: str | None = None,
    returncode: int | str = "",
    resumed: bool = False,
    error: str = "",
) -> dict[str, Any]:
    guidellm_results = run_dir / "guidellm_results.json"
    steady_state_results = run_dir / "steady_state_results.json"
    effective_status = status
    if effective_status is None:
        effective_status = (
            "ok"
            if guidellm_results.exists() or steady_state_results.exists()
            else "missing"
        )
    steady_state_metrics = parse_steady_state(steady_state_results)
    benchmark_metrics = steady_state_metrics or parse_guidellm(guidellm_results)
    row: dict[str, Any] = {
        "measurement_type": (
            "steady_state_saturated"
            if steady_state_metrics
            else "guidellm_end_to_end"
        ),
        "model": case["model_label"],
        "dataset": case["dataset_label"],
        "K": case["K"],
        "batch_size": case["batch_size"],
        "method": case["method"],
        "output_dir": str(run_dir),
        "status": effective_status,
        "returncode": returncode,
        **benchmark_metrics,
        **parse_profile(run_dir / "speclink_cv_profile.jsonl", int(case["K"])),
        "acceptance_rates": parse_acceptance(run_dir / "acceptance_analysis.txt"),
        "exact_match_vs_eagle3": "",
        "exact_match_source": "",
        "speedup_vs_eagle3": "",
        **parse_gpu_util(
            run_dir / "gpu_util.csv",
            active_start_time=safe_float(benchmark_metrics.get("benchmark_start_time")),
            active_end_time=safe_float(benchmark_metrics.get("benchmark_end_time")),
        ),
        "resumed": int(resumed),
    }
    if error:
        row["error"] = error
    return row


def parse_profile_distributions(path: Path, case: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {"selected_h": [], "first_reject": [], "queue_wait": [], "events": []}
    selected_h: list[dict[str, Any]] = []
    first_reject: list[dict[str, Any]] = []
    queue_wait: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    base = {
        "model": case["model_label"],
        "dataset": case["dataset_label"],
        "K": case["K"],
        "batch_size": case["batch_size"],
        "method": case["method"],
    }
    rows, truncated = read_profile_rows(path)
    for row in rows:
        event = row.get("event")
        if event == "verify_chunk_scheduled" and row.get("phase") == "prefix":
            selected_h.append({**base, "selected_h": row.get("chunk_len", "")})
            if row.get("queue_wait_ms") is not None:
                queue_wait.append({**base, "queue_wait_ms": row.get("queue_wait_ms", "")})
        elif event == "verify_chunk_result" and row.get("phase") == "prefix":
            accepted = int(row.get("num_accepted", 0))
            result = row.get("result", "")
            first_reject_position = accepted + 1 if result == "rejected_skip_suffix" else "prefix_survived"
            first_reject.append(
                {
                    **base,
                    "first_reject_position": first_reject_position,
                    "num_accepted": accepted,
                    "result": result,
                    "skipped_suffix_tokens": row.get("skipped_suffix_tokens", ""),
                    "extra_tlm_forward": row.get("extra_tlm_forward", ""),
                }
            )
        if event:
            events.append({**base, "event": event})
    if truncated:
        events.append({**base, "event": "profile_analysis_truncated"})
    return {"selected_h": selected_h, "first_reject": first_reject, "queue_wait": queue_wait, "events": events}


def local_no_proxy_value(env: dict[str, str]) -> str:
    values: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        for item in env.get(name, "").split(","):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    for item in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        if item not in values:
            values.append(item)
    return ",".join(values)


def wait_for_health(port: int, proc: subprocess.Popen[Any], timeout: int, log_path: Path) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"vLLM exited during startup rc={proc.returncode}\n{tail}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                sock.settimeout(2)
                sock.sendall(
                    b"GET /health HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Connection: close\r\n\r\n"
                )
                status = sock.recv(128).split(b"\r\n", 1)[0]
                if b" 200 " in status:
                    return
        except OSError:
            time.sleep(2)
    raise RuntimeError(f"vLLM health check timed out after {timeout}s")


def stop_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def start_gpu_sampler(
    path: Path,
    *,
    interval_ms: int,
) -> tuple[subprocess.Popen[Any] | None, Any | None]:
    if interval_ms <= 0:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.open("w", encoding="utf-8")
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used",
        "--format=csv,nounits",
        "-lms",
        str(interval_ms),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return proc, out
    except Exception:
        out.close()
        return None, None


def nsys_output_prefix(run_dir: Path, args: argparse.Namespace) -> Path:
    name = args.nsys_output_name.strip() or "vllm_guidellm_profile"
    return run_dir / name


def nsys_profile_command(
    cmd: list[str],
    run_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        "nsys",
        "profile",
        "-o",
        str(nsys_output_prefix(run_dir, args)),
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt,cublas",
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node",
        "--gpu-metrics-devices=cuda-visible",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat",
        *cmd,
    ]


def call_profile_endpoint(port: int, action: str, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as out:
        out.write(f"\n# {action}_profile\n")
        out.flush()
        return subprocess.run(
            [
                "curl",
                "--noproxy",
                "*",
                "-f",
                "-sS",
                "-X",
                "POST",
                f"http://127.0.0.1:{port}/{action}_profile",
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).returncode


def run_nsys_stats(run_dir: Path, args: argparse.Namespace) -> None:
    if not args.nsys_stats:
        return
    prefix = nsys_output_prefix(run_dir, args)
    reports = sorted(run_dir.glob(f"{prefix.name}*.nsys-rep"))
    stats_path = run_dir / "nsys_stats.txt"
    if not reports:
        stats_path.write_text(
            f"No Nsight Systems report found for prefix {prefix}\n",
            encoding="utf-8",
        )
        return
    with stats_path.open("w", encoding="utf-8") as out:
        for report in reports:
            out.write(f"# nsys stats {report.name}\n")
            out.flush()
            subprocess.run(
                ["nsys", "stats", str(report)],
                cwd=run_dir,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            summarize_nsys_sqlite(report.with_suffix(".sqlite"), run_dir, report.stem)
            if not args.keep_nsys_sqlite:
                sqlite_path = report.with_suffix(".sqlite")
                if sqlite_path.exists():
                    try:
                        sqlite_path.unlink()
                    except OSError as exc:
                        out.write(
                            f"\n# Failed to remove intermediate SQLite "
                            f"{sqlite_path.name}: {exc}\n"
                        )
            out.write("\n")


def summarize_nsys_sqlite(sqlite_path: Path, run_dir: Path, report_stem: str) -> None:
    if not sqlite_path.exists():
        return

    try:
        conn = sqlite3.connect(str(sqlite_path))
    except sqlite3.Error:
        return

    try:
        if not _sqlite_has_table(conn, "GPU_METRICS"):
            return
        gpu_rows = _collect_nsys_gpu_metrics(conn)
        api_rows = _collect_nsys_cuda_api_summary(conn)
        gap_summary = _collect_nsys_kernel_gap_summary(conn)
    except sqlite3.Error as exc:
        write_json(
            run_dir / f"{report_stem}_nsys_summary_error.json",
            {"error": str(exc), "sqlite": str(sqlite_path)},
        )
        return
    finally:
        conn.close()

    write_csv(run_dir / f"{report_stem}_gpu_metrics_summary.csv", gpu_rows)
    write_csv(run_dir / f"{report_stem}_cuda_api_summary.csv", api_rows)
    write_json(run_dir / f"{report_stem}_kernel_gap_summary.json", gap_summary)
    write_nsys_quick_summary(run_dir / f"{report_stem}_nsys_quick_summary.md", gpu_rows, api_rows, gap_summary)


def _sqlite_has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _collect_nsys_gpu_metrics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    wanted = {
        "GR Active [Throughput %]",
        "SMs Active [Throughput %]",
        "SM Issue [Throughput %]",
        "Tensor Active [Throughput %]",
        "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]",
    }
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for metric_name, value in conn.execute(
        """
        SELECT i.metricName, m.value
        FROM GPU_METRICS m
        JOIN TARGET_INFO_GPU_METRICS i
          ON m.typeId = i.typeId AND m.metricId = i.metricId
        WHERE i.metricName IN ({})
        """.format(",".join("?" for _ in wanted)),
        tuple(sorted(wanted)),
    ):
        values_by_metric[str(metric_name)].append(float(value))

    rows: list[dict[str, Any]] = []
    for metric_name in sorted(wanted):
        values = values_by_metric.get(metric_name, [])
        nonzero = [value for value in values if value > 0]
        rows.append(
            {
                "metric": metric_name,
                "samples": len(values),
                "mean": mean_or_empty(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "max": max(values) if values else "",
                "nonzero_samples": len(nonzero),
                "nonzero_mean": mean_or_empty(nonzero),
            }
        )
    return rows


def _collect_nsys_cuda_api_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _sqlite_has_table(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    rows: list[dict[str, Any]] = []
    for name, count, total_ns, avg_ns, max_ns in conn.execute(
        """
        SELECT s.value AS name,
               COUNT(*) AS calls,
               SUM(r.end - r.start) AS total_ns,
               AVG(r.end - r.start) AS avg_ns,
               MAX(r.end - r.start) AS max_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        GROUP BY r.nameId
        ORDER BY total_ns DESC
        LIMIT 50
        """
    ):
        rows.append(
            {
                "api": name,
                "calls": count,
                "total_ms": float(total_ns) / 1e6,
                "avg_us": float(avg_ns) / 1e3,
                "max_us": float(max_ns) / 1e3,
            }
        )
    return rows


def _collect_nsys_kernel_gap_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _sqlite_has_table(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return {}
    intervals = [
        (int(start), int(end))
        for start, end in conn.execute(
            """
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > start
            ORDER BY start
            """
        )
    ]
    if not intervals:
        return {}

    total_kernel_time_ns = sum(end - start for start, end in intervals)
    merged: list[tuple[int, int]] = []
    current_start, current_end = intervals[0]
    gaps: list[float] = []
    for start, end in intervals[1:]:
        if start > current_end:
            merged.append((current_start, current_end))
            gaps.append((start - current_end) / 1e3)
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    merged.append((current_start, current_end))

    gpu_busy_time_ns = sum(end - start for start, end in merged)
    trace_window_ns = merged[-1][1] - merged[0][0]
    gap_time_ns = max(0, trace_window_ns - gpu_busy_time_ns)
    return {
        "kernel_count": len(intervals),
        "merged_busy_ranges": len(merged),
        "trace_window_ms": trace_window_ns / 1e6,
        "sum_kernel_time_ms": total_kernel_time_ns / 1e6,
        "gpu_busy_union_ms": gpu_busy_time_ns / 1e6,
        "kernel_gap_total_ms": gap_time_ns / 1e6,
        "kernel_gap_ratio": (gap_time_ns / trace_window_ns) if trace_window_ns > 0 else "",
        "kernel_gap_count": len(gaps),
        "kernel_gap_p50_us": percentile(gaps, 50),
        "kernel_gap_p95_us": percentile(gaps, 95),
        "kernel_gap_max_us": max(gaps) if gaps else "",
    }


def write_nsys_quick_summary(
    path: Path,
    gpu_rows: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
    gap_summary: dict[str, Any],
) -> None:
    metric_map = {row["metric"]: row for row in gpu_rows}
    focus_metrics = [
        "SMs Active [Throughput %]",
        "Tensor Active [Throughput %]",
        "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]",
        "SM Issue [Throughput %]",
    ]
    lines = ["# Nsight Quick Summary", ""]
    lines.append("## GPU Metrics")
    lines.append("")
    lines.append("| metric | mean | p95 | max | nonzero mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for metric in focus_metrics:
        row = metric_map.get(metric, {})
        lines.append(
            "| {metric} | {mean} | {p95} | {max_v} | {nonzero_mean} |".format(
                metric=metric,
                mean=_fmt_float(row.get("mean")),
                p95=_fmt_float(row.get("p95")),
                max_v=_fmt_float(row.get("max")),
                nonzero_mean=_fmt_float(row.get("nonzero_mean")),
            )
        )
    lines.extend(["", "## Kernel Gaps", ""])
    lines.append(
        "- gap ratio: {ratio}; p95 gap: {p95} us; max gap: {max_gap} us; "
        "busy union: {busy} ms; trace window: {window} ms".format(
            ratio=_fmt_float(gap_summary.get("kernel_gap_ratio")),
            p95=_fmt_float(gap_summary.get("kernel_gap_p95_us")),
            max_gap=_fmt_float(gap_summary.get("kernel_gap_max_us")),
            busy=_fmt_float(gap_summary.get("gpu_busy_union_ms")),
            window=_fmt_float(gap_summary.get("trace_window_ms")),
        )
    )
    lines.extend(["", "## CUDA API Overhead", ""])
    lines.append("| api | calls | total ms | avg us | max us |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in api_rows[:10]:
        lines.append(
            "| {api} | {calls} | {total} | {avg} | {max_v} |".format(
                api=row.get("api", ""),
                calls=row.get("calls", ""),
                total=_fmt_float(row.get("total_ms")),
                avg=_fmt_float(row.get("avg_us")),
                max_v=_fmt_float(row.get("max_us")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: Any) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def create_result_tree(root: Path) -> None:
    for name in RESULT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def run_and_capture(cmd: list[str], cwd: Path, stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(cmd, cwd=cwd, text=True, stdout=out, stderr=err, check=False)
    return result.returncode


def write_setup_artifacts(root: Path, args: argparse.Namespace) -> None:
    create_result_tree(root)
    env_report = collect_env()
    write_json(root / "00_env" / "env_report.json", env_report)
    write_env_markdown(env_report, root / "00_env" / "env_report.md")
    (root / "01_impl_notes" / "implementation_notes.md").write_text(
        "# SpecLink-CV GuideLLM Implementation Notes\n\n"
        "- This run uses live `vllm serve`; `--benchmark-mode guidellm` records finite-request GuideLLM end-to-end metrics, while `--benchmark-mode steady_state` records closed-loop saturated output tokens/s.\n"
        "- In steady-state mode, `batch_size=N` means closed-loop concurrency N. Warmup, measurement window, and cooldown/drain are separate; only output tokens emitted during the fixed measurement window are counted.\n"
        "- Do not report finite-request GuideLLM makespan as serving throughput. Use the wording `steady-state throughput at concurrency N` or `saturated output tokens/s at concurrency N` for steady-state rows.\n"
        "- SpecLink-CV runtime switches are passed through `SPECLINK_CV_*` environment variables.\n"
        "- `cv_*` methods force `--no-async-scheduling` so the regular V1 scheduler executes the prefix/suffix verification hooks.\n"
        "- `SPECLINK_CV_ASYNC_QUEUE` controls the experiment's prefix verification queue and is independent from vLLM's own async scheduler mode.\n"
        "- `cv_half_async_staged_simple` is an experimental performance variant: the drafter first generates only the prefix width and drafts the suffix only after prefix acceptance.\n"
        "- `SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1` keeps suffix verification inside the regular batched scheduler path instead of isolating one suffix request per step; without it, skipped verifier work is often hidden by singleton suffix scheduling overhead.\n"
        "- `--nsys-profile` wraps the server command in Nsight Systems, sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, passes `--profiler-config.profiler cuda`, and brackets the benchmark client with `/start_profile` and `/stop_profile`.\n"
        "- Nsight runs write `nsys_stats.txt` plus quick CSV/Markdown summaries for SM/Tensor/DRAM activity, kernel gaps, and CUDA API overhead; intermediate SQLite exports are removed unless `--keep-nsys-sqlite` is set.\n"
        "- Live SpecLink-CV defaults to exact-safe mode. Unless `--allow-shape-drift-chunking` / `SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=1` is set, h<K CV rows fall back to EAGLE3 one-shot because verifier-shape drift has been observed. Such rows are exact but not valid chunked speedup claims.\n"
        "- For token-id exact h<K debugging, pass `--allow-shape-drift-chunking --env VLLM_BATCH_INVARIANT=1`. For math-quality performance runs, leave `VLLM_BATCH_INVARIANT=0` and require the math quality gate to pass.\n"
        "- The matrix runner supports `--resume`, `--case-offset`, and `--case-limit` so long runs can be split without losing completed cases.\n"
        "- Text-level exact-match is computed by aligning GuideLLM successful requests by `request_args` against the matching `eagle3_oneshot` run; token-id exact-match must be verified with `tools/speclink_cv/live_correctness_smoke.py`.\n"
        "- JSONL debug/profile output is bounded by `--log-max-events`, `--profile-max-events`, and `--analysis-profile-max-rows`; increase them only for a short representative diagnostic.\n",
        encoding="utf-8",
    )
    diff = subprocess.run(
        ["git", "diff", "--", "vllm", "tools", "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (root / "patches" / "speclink_cv.diff").write_text(diff.stdout, encoding="utf-8")
    (root / "patches" / "vllm_speclink_cv.diff").write_text(
        diff.stdout, encoding="utf-8"
    )
    write_json(root / "logs" / "run_config.json", vars(args))


def run_unit_tests(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in FOCUSED_TESTS:
        cmd = [sys.executable, "-m", f"tools.speclink_cv.{name}"]
        stdout = root / "02_unit_tests" / f"{name}.stdout.log"
        stderr = root / "02_unit_tests" / f"{name}.stderr.log"
        rc = run_and_capture(cmd, REPO_ROOT, stdout, stderr)
        rows.append({"test": name, "returncode": rc, "status": "pass" if rc == 0 else "fail"})
    write_csv(root / "02_unit_tests" / "unit_test_summary.csv", rows)
    write_json(root / "02_unit_tests" / "unit_test_summary.json", {"tests": rows})
    lines = ["# Unit Test Summary", ""]
    for row in rows:
        lines.append(f"- {row['test']}: {row['status']} rc={row['returncode']}")
    (root / "02_unit_tests" / "unit_test_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def make_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    models = model_registry()
    datasets = dataset_registry()
    selected_models = split_csv(args.models)
    selected_datasets = split_csv(args.datasets)
    selected_methods = split_csv(args.methods)
    selected_ks = [int(item) for item in split_csv(args.ks)]
    selected_batches = [int(item) for item in split_csv(args.batch_sizes)]
    cases: list[dict[str, Any]] = []
    for model_label in selected_models:
        if model_label not in models:
            raise SystemExit(f"unknown model label: {model_label}")
        for dataset_label in selected_datasets:
            if dataset_label not in datasets:
                raise SystemExit(f"unknown dataset label: {dataset_label}")
            for k in selected_ks:
                for batch_size in selected_batches:
                    for method in selected_methods:
                        if method not in METHODS:
                            raise SystemExit(f"unknown method: {method}")
                        cases.append(
                            {
                                "model_label": model_label,
                                "base_model": models[model_label]["base"],
                                "speculator_model": models[model_label]["speculator"],
                                "dataset_label": dataset_label,
                                "dataset": datasets[dataset_label],
                                "K": k,
                                "batch_size": batch_size,
                                "method": method,
                            }
                        )
    return cases


def server_command(case: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> list[str]:
    method_cfg = METHODS[case["method"]]
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        case["base_model"],
        "--seed",
        "42",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(args.port),
        "--max-num-seqs",
        str(case["batch_size"]),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    if args.enforce_eager or (method_cfg["cv"] == "1" and not args.allow_cv_cudagraph):
        cmd.append("--enforce-eager")
    if args.disable_vllm_async_scheduling or method_cfg["cv"] == "1":
        cmd.append("--no-async-scheduling")
    if args.nsys_profile:
        cmd.extend(["--profiler-config.profiler", "cuda"])
    if method_cfg["spec"] == "1":
        spec_config = {
            "model": case["speculator_model"],
            "num_speculative_tokens": int(case["K"]),
            "method": "eagle3",
            "max_model_len": int(args.max_model_len),
        }
        cmd.extend(["--speculative-config", json.dumps(spec_config)])
    if args.nsys_profile:
        cmd = nsys_profile_command(cmd, run_dir, args)
    return cmd


def guidellm_command(case: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> list[str]:
    body = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.max_tokens > 0:
        body["max_tokens"] = args.max_tokens
    return [
        sys.executable,
        "-m",
        "guidellm",
        "benchmark",
        "run",
        "--target",
        f"http://127.0.0.1:{args.port}",
        "--request-type",
        args.request_type,
        "--model",
        case["base_model"],
        "--processor",
        case["base_model"],
        "--data",
        case["dataset"],
        "--random-seed",
        str(args.random_seed),
        "--profile",
        "concurrent",
        "--rate",
        str(case["batch_size"]),
        "--max-requests",
        str(args.max_requests),
        "--output-path",
        str(run_dir / "guidellm_results.json"),
        "--backend-args",
        json.dumps({"extras": {"body": body}}),
    ]


def steady_state_command(case: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> list[str]:
    if args.max_tokens <= 0:
        raise SystemExit(
            "--benchmark-mode steady_state requires --max-tokens > 0; "
            "use --steady-state-ignore-eos for fixed output length"
        )
    if args.request_type not in {"completions", "chat_completions"}:
        raise SystemExit(
            "--benchmark-mode steady_state requires --request-type completions "
            "or chat_completions"
        )
    max_prompts = args.steady_state_max_prompts
    if max_prompts <= 0:
        max_prompts = args.max_requests
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/speclink_cv/steady_state_openai_benchmark.py"),
        "--target",
        f"http://127.0.0.1:{args.port}",
        "--model",
        case["base_model"],
        "--dataset",
        case["dataset"],
        "--run-label",
        (
            f"{case['model_label']}_{case['dataset_label']}_k{case['K']}_"
            f"bs{case['batch_size']}_{case['method']}"
        ),
        "--request-type",
        args.request_type,
        "--concurrency",
        str(case["batch_size"]),
        "--warmup-s",
        str(args.steady_state_warmup_s),
        "--measurement-s",
        str(args.steady_state_measurement_s),
        "--cooldown-s",
        str(args.steady_state_cooldown_s),
        "--bucket-s",
        str(args.steady_state_bucket_s),
        "--max-prompts",
        str(max_prompts),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.random_seed),
        "--timeout",
        str(args.steady_state_timeout),
        "--output-dir",
        str(run_dir),
    ]
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if not args.steady_state_ignore_eos:
        cmd.append("--no-ignore-eos")
    if args.steady_state_allow_final_usage_fallback:
        cmd.append("--allow-final-usage-fallback")
    return cmd


def case_env(case: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    method_cfg = METHODS[case["method"]]
    env = os.environ.copy()
    no_proxy = local_no_proxy_value(env)
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    env.update(
        {
            "SPECLINK_CV_ENABLE": method_cfg["cv"],
            "SPECLINK_CV_CONFIDENCE_SIZING": method_cfg["confidence"],
            "SPECLINK_CV_ASYNC_QUEUE": method_cfg["async"],
            "SPECLINK_CV_ROOFLINE_PACKING": method_cfg["roofline"],
            "SPECLINK_CV_STAGED_DRAFTING": method_cfg.get("staged", "0"),
            "SPECLINK_CV_LOG_JSONL": str(run_dir / "speclink_cv_events.jsonl"),
            "SPECLINK_CV_PROFILE_JSONL": str(run_dir / "speclink_cv_profile.jsonl"),
            "SPECLINK_CV_LOG_MAX_EVENTS": str(args.log_max_events),
            "SPECLINK_CV_PROFILE_MAX_EVENTS": str(args.profile_max_events),
            "SPECLINK_CV_UTIL_THRESHOLD": str(args.util_threshold),
            "SPECLINK_CV_MAX_QUEUE_WAIT_MS": str(args.max_queue_wait_ms),
            "SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP": str(args.max_verify_tokens_per_step),
            "SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP": str(args.max_verify_seqs_per_step),
            "SPECLINK_CV_ALLOW_BATCHED_PREFIX": "1"
            if args.allow_batched_prefix_verification
            else "0",
            "SPECLINK_CV_ALLOW_BATCHED_SUFFIX": "1"
            if args.allow_batched_prefix_verification
            else "0",
            "SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING": "1"
            if args.allow_shape_drift_chunking
            else "0",
        }
    )
    env.update(parse_env_items(args.env))
    if args.calibration_path:
        env["SPECLINK_CV_CALIBRATION_PATH"] = str(Path(args.calibration_path).resolve())
    if args.nsys_profile:
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return env


def run_case(case: dict[str, Any], args: argparse.Namespace, root: Path, commands: list[str]) -> dict[str, Any]:
    run_name = (
        f"{case['model_label']}_{case['dataset_label']}_k{case['K']}_"
        f"bs{case['batch_size']}_{case['method']}"
    )
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                **case,
                "extra_env": parse_env_items(args.env),
                "benchmark_mode": args.benchmark_mode,
                "max_tokens": args.max_tokens,
                "max_requests": args.max_requests,
                "steady_state_warmup_s": args.steady_state_warmup_s,
                "steady_state_measurement_s": args.steady_state_measurement_s,
                "steady_state_cooldown_s": args.steady_state_cooldown_s,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    server_log = run_dir / "vllm_server.log"
    guidellm_log = run_dir / "guidellm_output.log"
    benchmark_log = (
        run_dir / "steady_state_output.log"
        if args.benchmark_mode == "steady_state"
        else guidellm_log
    )
    acceptance_path = run_dir / "acceptance_analysis.txt"
    method_cfg = METHODS[case["method"]]
    server_cmd = server_command(case, args, run_dir)
    benchmark_cmd = (
        steady_state_command(case, args, run_dir)
        if args.benchmark_mode == "steady_state"
        else guidellm_command(case, args, run_dir)
    )
    env = case_env(case, args, run_dir)
    cv_env = " ".join(
        f"{key}={json.dumps(value)}"
        for key, value in env.items()
        if key.startswith("SPECLINK_CV_")
        or key == "VLLM_WORKER_MULTIPROC_METHOD"
        or key in parse_env_items(args.env)
    )
    profile_lines = []
    if args.nsys_profile:
        profile_lines.append(
            f"curl --noproxy '*' -f -sS -X POST "
            f"http://127.0.0.1:{args.port}/start_profile"
        )
    profile_lines.append(subprocess.list2cmdline(benchmark_cmd))
    if args.nsys_profile:
        profile_lines.append(
            f"curl --noproxy '*' -f -sS -X POST "
            f"http://127.0.0.1:{args.port}/stop_profile"
        )
        profile_lines.append(
            f"nsys stats {nsys_output_prefix(run_dir, args)}.nsys-rep"
        )
    command_text = (
        f"# {run_name}\n"
        + cv_env
        + " "
        + subprocess.list2cmdline(server_cmd)
        + "\n"
        + "\n".join(profile_lines)
        + "\n"
    )
    commands.append(command_text)
    (run_dir / "run_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {EVAL_DIR}\n"
        + command_text,
        encoding="utf-8",
    )
    (run_dir / "run_command.sh").chmod(0o755)
    base_row = {
        "measurement_type": (
            "steady_state_saturated"
            if args.benchmark_mode == "steady_state"
            else "guidellm_end_to_end"
        ),
        "model": case["model_label"],
        "dataset": case["dataset_label"],
        "K": case["K"],
        "batch_size": case["batch_size"],
        "method": case["method"],
        "output_dir": str(run_dir),
    }
    if args.dry_run:
        return {**base_row, "status": "planned"}
    result_path = (
        run_dir / "steady_state_results.json"
        if args.benchmark_mode == "steady_state"
        else run_dir / "guidellm_results.json"
    )
    if args.resume and result_path.exists():
        if method_cfg["spec"] == "1" and not acceptance_path.exists() and server_log.exists():
            parser = EVAL_DIR / "scripts/parse_logs.py"
            with acceptance_path.open("w", encoding="utf-8") as f:
                subprocess.run(
                    [sys.executable, str(parser), str(server_log)],
                    cwd=EVAL_DIR,
                    text=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        return row_from_run_dir(case, run_dir, status="ok", resumed=True)
    if port_is_open(args.port):
        return {**base_row, "status": "failed", "error": f"port {args.port} is already in use"}

    proc: subprocess.Popen[Any] | None = None
    gpu_sampler_proc: subprocess.Popen[Any] | None = None
    gpu_sampler_out: Any | None = None
    profile_started = False
    try:
        with server_log.open("w", encoding="utf-8") as server_out:
            proc = subprocess.Popen(
                server_cmd,
                cwd=EVAL_DIR,
                env=env,
                stdout=server_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        wait_for_health(args.port, proc, args.health_check_timeout, server_log)
        gpu_sampler_proc, gpu_sampler_out = start_gpu_sampler(
            run_dir / "gpu_util.csv",
            interval_ms=args.gpu_util_sampling_ms,
        )
        if args.nsys_profile:
            start_rc = call_profile_endpoint(
                args.port, "start", run_dir / "profile_control.log"
            )
            if start_rc != 0:
                raise RuntimeError(
                    f"start_profile failed with return code {start_rc}"
                )
            profile_started = True
        with benchmark_log.open("w", encoding="utf-8") as benchmark_out:
            benchmark_rc = subprocess.run(
                benchmark_cmd,
                cwd=EVAL_DIR,
                env=env,
                stdout=benchmark_out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            ).returncode
        if args.nsys_profile and profile_started:
            stop_rc = call_profile_endpoint(
                args.port, "stop", run_dir / "profile_control.log"
            )
            profile_started = False
            if stop_rc != 0:
                raise RuntimeError(
                    f"stop_profile failed with return code {stop_rc}"
                )
        if method_cfg["spec"] == "1":
            parser = EVAL_DIR / "scripts/parse_logs.py"
            with acceptance_path.open("w", encoding="utf-8") as f:
                subprocess.run(
                    [sys.executable, str(parser), str(server_log)],
                    cwd=EVAL_DIR,
                    text=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        return row_from_run_dir(
            case,
            run_dir,
            status="ok" if benchmark_rc == 0 else "failed",
            returncode=benchmark_rc,
        )
    except Exception as exc:  # noqa: BLE001
        return {**base_row, "status": "failed", "error": str(exc)}
    finally:
        if args.nsys_profile and profile_started:
            call_profile_endpoint(args.port, "stop", run_dir / "profile_control.log")
        stop_process(gpu_sampler_proc)
        if gpu_sampler_out is not None:
            gpu_sampler_out.close()
        stop_process(proc)
        if args.nsys_profile:
            run_nsys_stats(run_dir, args)


def collect_run_rows(root: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect rows from all existing run directories for the requested matrix.

    Long GuideLLM matrices are normally run with --case-offset/--case-limit.
    After each slice, summarize every completed or attempted run under the same
    output root so the top-level CSVs converge as slices finish. This must not
    be limited to the current method/model subset, because focused fill-in
    slices still need to preserve rows from earlier slices in the aggregate CSV.
    """
    rows: list[dict[str, Any]] = []
    case_by_name = {
        (
            f"{case['model_label']}_{case['dataset_label']}_k{case['K']}_"
            f"bs{case['batch_size']}_{case['method']}"
        ): case
        for case in cases
    }
    run_names = set(case_by_name)
    runs_root = root / "runs"
    if runs_root.exists():
        for config_path in runs_root.glob("*/config.json"):
            run_names.add(config_path.parent.name)
    for run_name in sorted(run_names):
        run_dir = runs_root / run_name
        config_path = run_dir / "config.json"
        if config_path.exists():
            case = json.loads(config_path.read_text(encoding="utf-8"))
            rows.append(row_from_run_dir(case, run_dir))
    return rows


def fill_speedups(rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if row.get("method") == "eagle3_oneshot" and row.get("status") == "ok":
            try:
                by_key[scenario_key(row)] = float(row["throughput"])
            except Exception:
                pass
    for row in rows:
        key = scenario_key(row)
        baseline = by_key.get(key)
        if baseline:
            try:
                row["speedup_vs_eagle3"] = float(row["throughput"]) / baseline
            except Exception:
                row["speedup_vs_eagle3"] = ""


def fill_exact_matches(rows: list[dict[str, Any]]) -> None:
    baseline_outputs: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in rows:
        key = scenario_key(row)
        if row.get("method") == "eagle3_oneshot" and row.get("status") == "ok":
            baseline_outputs[key] = load_successful_output_map(
                Path(str(row.get("output_dir", ""))) / "guidellm_results.json"
            )
            row["exact_match_vs_eagle3"] = 1.0
            row["exact_match_source"] = "guidellm_request_args_output_text_baseline"
            row["exact_match_compared_requests"] = len(baseline_outputs[key])
            row["exact_match_common_requests"] = len(baseline_outputs[key])

    for row in rows:
        method = str(row.get("method", ""))
        if not method.startswith("cv_") or row.get("status") != "ok":
            continue
        key = scenario_key(row)
        baseline = baseline_outputs.get(key, {})
        outputs = load_successful_output_map(
            Path(str(row.get("output_dir", ""))) / "guidellm_results.json"
        )
        if not baseline or not outputs:
            row["exact_match_vs_eagle3"] = ""
            row["exact_match_source"] = "missing_guidellm_request_args_output_text"
            row["exact_match_compared_requests"] = ""
            row["exact_match_common_requests"] = ""
            continue
        common = sorted(set(baseline) & set(outputs))
        if not common:
            row["exact_match_vs_eagle3"] = ""
            row["exact_match_source"] = "no_common_guidellm_request_args"
            row["exact_match_compared_requests"] = max(len(baseline), len(outputs))
            row["exact_match_common_requests"] = 0
            continue
        matches = sum(1 for item in common if baseline[item] == outputs[item])
        row["exact_match_vs_eagle3"] = matches / max(len(baseline), len(outputs))
        row["exact_match_source"] = "guidellm_request_args_output_text"
        row["exact_match_compared_requests"] = max(len(baseline), len(outputs))
        row["exact_match_common_requests"] = len(common)


def fill_quality_metrics(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    baseline_outputs: dict[tuple[Any, ...], dict[str, str]] = {}
    baseline_requests: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    baseline_quality: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        quality = quality_for_run(row)
        row.update(quality)
        key = scenario_key(row)
        if row.get("method") == "eagle3_oneshot":
            result_path = Path(str(row.get("output_dir", ""))) / "guidellm_results.json"
            baseline_outputs[key] = load_successful_output_map(result_path)
            baseline_requests[key] = load_successful_request_map(result_path)
            score = safe_float(row.get("quality_score"))
            if score is not None:
                baseline_quality[key] = score
            row["quality_gate_pass"] = 1 if row.get("quality_metric") else ""
            row["quality_gate_status"] = "baseline"

    for row in rows:
        if row.get("status") != "ok":
            continue
        key = scenario_key(row)
        method = str(row.get("method", ""))
        if not method.startswith("cv_"):
            continue
        baseline = baseline_outputs.get(key, {})
        baseline_request_map = baseline_requests.get(key, {})
        outputs = load_successful_output_map(
            Path(str(row.get("output_dir", ""))) / "guidellm_results.json"
        )
        request_map = load_successful_request_map(
            Path(str(row.get("output_dir", ""))) / "guidellm_results.json"
        )
        common = sorted(set(baseline) & set(outputs))
        similarities = [
            normalized_text_similarity(outputs[item], baseline[item]) for item in common
        ]
        row["quality_baseline_similarity"] = (
            sum(similarities) / len(similarities) if similarities else ""
        )
        row["quality_baseline_common_requests"] = len(common)
        if common:
            out_len = sum(len(outputs[item]) for item in common) / len(common)
            base_len = sum(len(baseline[item]) for item in common) / len(common)
            row["quality_length_ratio_vs_eagle3"] = (
                out_len / base_len if base_len > 0 else ""
            )
        score = safe_float(row.get("quality_score"))
        base_score = baseline_quality.get(key)
        gate_score = score
        gate_base_score = base_score
        if row.get("dataset") == "math" and common and baseline_request_map and request_map:
            dataset_path = dataset_path_for_run(row)
            if dataset_path is not None:
                dataset = load_dataset_by_prompt(dataset_path)
                cv_common = score_math_requests(
                    dataset, [request_map[item] for item in common if item in request_map]
                )
                base_common = score_math_requests(
                    dataset,
                    [
                        baseline_request_map[item]
                        for item in common
                        if item in baseline_request_map
                    ],
                )
                row["quality_common_requests"] = len(common)
                row["quality_common_score"] = cv_common["score"]
                row["quality_common_correct"] = cv_common["correct"]
                row["quality_common_evaluable"] = cv_common["evaluable"]
                row["quality_baseline_common_score"] = base_common["score"]
                row["quality_baseline_common_correct"] = base_common["correct"]
                row["quality_baseline_common_evaluable"] = base_common["evaluable"]
                common_score = safe_float(cv_common["score"])
                common_base_score = safe_float(base_common["score"])
                if common_score is not None and common_base_score is not None:
                    gate_score = common_score
                    gate_base_score = common_base_score
        if score is not None and base_score is not None:
            row["quality_delta_vs_eagle3"] = score - base_score
            if gate_score is not None and gate_base_score is not None:
                row["quality_gate_score"] = gate_score
                row["quality_gate_baseline_score"] = gate_base_score
                row["quality_gate_delta_vs_eagle3"] = gate_score - gate_base_score
            if int(row.get("quality_reliable") or 0) == 0:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "quality_unreliable_short_outputs"
            elif gate_base_score is None or gate_base_score <= 0.0:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "baseline_quality_uninformative"
            elif gate_score is not None and (
                gate_score + args.math_quality_drop_tolerance >= gate_base_score
            ):
                row["quality_gate_pass"] = 1
                row["quality_gate_status"] = "math_quality_preserved"
            else:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "math_quality_drop"
            continue
        if row.get("dataset") == "mtbench":
            similarity = safe_float(row.get("quality_baseline_similarity"))
            length_ratio = safe_float(row.get("quality_length_ratio_vs_eagle3"))
            if similarity is None:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "mtbench_missing_baseline_proxy"
            elif int(row.get("quality_reliable") or 0) == 0:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "quality_unreliable_short_outputs"
            elif (
                similarity >= args.mtbench_similarity_threshold
                and length_ratio is not None
                and args.mtbench_min_length_ratio <= length_ratio <= args.mtbench_max_length_ratio
            ):
                row["quality_gate_pass"] = 1
                row["quality_gate_status"] = "mtbench_proxy_no_large_drift"
            else:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "mtbench_proxy_large_drift"
        else:
            row["quality_gate_pass"] = 0
            row["quality_gate_status"] = "missing_quality_gate"


def fill_speedup_claim_validity(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        method = str(row.get("method", ""))
        if not method.startswith("cv_"):
            row["speedup_claim_valid"] = ""
            row["speedup_claim_status"] = ""
            continue
        if row.get("status") != "ok":
            row["speedup_claim_valid"] = 0
            row["speedup_claim_status"] = "invalid_run_status"
            continue
        selected_h = safe_float(row.get("selected_h_avg"))
        prefix_accept = safe_float(row.get("prefix_accepted_ratio"))
        prefix_reject = safe_float(row.get("prefix_rejected_ratio"))
        has_live_chunk = (
            selected_h is not None
            and (
                (prefix_accept is not None and prefix_accept > 0.0)
                or (prefix_reject is not None and prefix_reject > 0.0)
            )
        )
        if not has_live_chunk:
            row["speedup_claim_valid"] = 0
            row["speedup_claim_status"] = "invalid_no_live_chunking"
            continue
        if str(row.get("quality_gate_pass", "")) in {"1", "1.0", "True"}:
            row["speedup_claim_valid"] = 1
            row["speedup_claim_status"] = "valid_quality_preserving_chunked"
            continue
        if not row.get("quality_gate_status"):
            row["speedup_claim_valid"] = 0
            row["speedup_claim_status"] = "missing_quality_gate"
            continue
        row["speedup_claim_valid"] = 0
        row["speedup_claim_status"] = str(row.get("quality_gate_status"))


def scenario_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model", "")),
        str(row.get("dataset", "")),
        str(row.get("K", "")),
        str(row.get("batch_size", "")),
    )


def method_family(method: str) -> dict[str, str]:
    return {
        "confidence": "conf" if "_conf_" in method else "half",
        "queue": "async" if "_async_" in method else "sync",
        "packing": "roofline" if method.endswith("_roofline") else "simple",
    }


def best_cv_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not str(row.get("method", "")).startswith("cv_") or row.get("status") != "ok":
            continue
        if str(row.get("speedup_claim_valid", "")) not in {"1", "1.0"} and row.get("speedup_claim_valid") != 1:
            continue
        throughput = safe_float(row.get("throughput"))
        if throughput is None:
            continue
        key = scenario_key(row)
        prev = best.get(key)
        if prev is None or throughput > float(prev.get("throughput", 0.0)):
            best[key] = row
    return best


def figure_text(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGB", (1200, 720), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((40, 32), title, fill="#111", font=font)
        y = 84
        for line in lines[:36]:
            draw.text((40, y), line[:150], fill="#222", font=font)
            y += 18
        image.save(path)
    except Exception:
        path.with_suffix(".txt").write_text(title + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def draw_bar_figure(path: Path, title: str, rows: list[dict[str, Any]], label_key: str, value_key: str) -> None:
    if not rows:
        figure_text(path, title, ["No data available."])
        return
    try:
        from PIL import Image, ImageDraw, ImageFont

        width, height = 1280, 760
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((40, 28), title, fill="#111", font=font)
        plot_left, plot_top, plot_right, plot_bottom = 80, 90, 1220, 610
        values = [safe_float(row.get(value_key)) or 0.0 for row in rows]
        max_value = max(values + [1.0]) * 1.12
        bar_w = max(10, int((plot_right - plot_left) / max(len(rows), 1) * 0.62))
        gap = (plot_right - plot_left) / max(len(rows), 1)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#222")
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#222")
        for idx, row in enumerate(rows):
            value = safe_float(row.get(value_key)) or 0.0
            x_center = plot_left + gap * (idx + 0.5)
            x0 = x_center - bar_w / 2
            x1 = x_center + bar_w / 2
            y0 = plot_bottom - (value / max_value) * (plot_bottom - plot_top)
            color = "#0b7285" if idx % 2 == 0 else "#5f3dc4"
            draw.rectangle((x0, y0, x1, plot_bottom), fill=color)
            draw.text((x0, y0 - 16), f"{value:.3g}", fill="#111", font=font)
            label = str(row.get(label_key, ""))[:22]
            draw.text((x0 - 18, plot_bottom + 12), label, fill="#333", font=font)
        image.save(path)
    except Exception:
        figure_text(path, title, [f"{row.get(label_key)}: {row.get(value_key)}" for row in rows])


def draw_scatter_figure(path: Path, title: str, rows: list[dict[str, Any]], x_key: str, y_key: str, label_key: str) -> None:
    usable = [row for row in rows if safe_float(row.get(x_key)) is not None and safe_float(row.get(y_key)) is not None]
    if not usable:
        figure_text(path, title, ["No data available."])
        return
    try:
        from PIL import Image, ImageDraw, ImageFont

        width, height = 1280, 760
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((40, 28), title, fill="#111", font=font)
        left, top, right, bottom = 90, 90, 1200, 620
        xs = [safe_float(row.get(x_key)) or 0.0 for row in usable]
        ys = [safe_float(row.get(y_key)) or 0.0 for row in usable]
        max_x = max(xs + [0.1]) * 1.15
        max_y = max(ys + [0.1]) * 1.15
        draw.rectangle((left, top, right, bottom), outline="#222")
        draw.text((left + 420, bottom + 52), x_key, fill="#111", font=font)
        draw.text((18, top - 28), y_key, fill="#111", font=font)
        colors = ["#0b7285", "#5f3dc4", "#2b8a3e", "#c92a2a", "#e67700", "#364fc7"]
        for idx, row in enumerate(usable[:80]):
            x = left + ((safe_float(row.get(x_key)) or 0.0) / max_x) * (right - left)
            y = bottom - ((safe_float(row.get(y_key)) or 0.0) / max_y) * (bottom - top)
            color = colors[idx % len(colors)]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            if idx < 28:
                draw.text((min(x + 8, right - 170), max(top + 4, y - 8)), str(row.get(label_key, ""))[:24], fill=color, font=font)
        image.save(path)
    except Exception:
        figure_text(path, title, [f"{row.get(x_key)}, {row.get(y_key)}, {row.get(label_key)}" for row in usable])


def aggregate_mean(rows: list[dict[str, Any]], group_keys: list[str], value_key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = safe_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(item, "") for item in group_keys)
        groups.setdefault(key, []).append(value)
    out = []
    for key, values in sorted(groups.items()):
        row = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        row[value_key] = sum(values) / len(values)
        row["count"] = len(values)
        row["label"] = " ".join(str(part) for part in key)
        out.append(row)
    return out


def collect_distribution_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    selected_h: list[dict[str, Any]] = []
    first_reject: list[dict[str, Any]] = []
    queue_wait: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row in rows:
        output_dir = row.get("output_dir")
        if not output_dir:
            continue
        config_path = Path(str(output_dir)) / "config.json"
        profile_path = Path(str(output_dir)) / "speclink_cv_profile.jsonl"
        if not config_path.exists():
            continue
        case = json.loads(config_path.read_text(encoding="utf-8"))
        parsed = parse_profile_distributions(profile_path, case)
        selected_h.extend(parsed["selected_h"])
        first_reject.extend(parsed["first_reject"])
        queue_wait.extend(parsed["queue_wait"])
        events.extend(parsed["events"])
    return {"selected_h": selected_h, "first_reject": first_reject, "queue_wait": queue_wait, "events": events}


def build_figure_artifacts(root: Path, rows: list[dict[str, Any]]) -> None:
    fig_dir = root / "08_figures"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best = best_cv_rows(ok_rows)
    comparison: list[dict[str, Any]] = []
    for key in sorted({scenario_key(row) for row in ok_rows}):
        scenario_rows = [row for row in ok_rows if scenario_key(row) == key]
        for method in ("pure_vllm", "eagle3_oneshot"):
            for row in scenario_rows:
                if row.get("method") == method:
                    comparison.append({**row, "label": f"{row['model']} {row['dataset']} K{row['K']} bs{row['batch_size']} {method}"})
        if key in best:
            row = best[key]
            comparison.append({**row, "label": f"{row['model']} {row['dataset']} K{row['K']} bs{row['batch_size']} best_cv"})
    write_csv(fig_dir / "throughput_comparison.csv", comparison)
    draw_bar_figure(fig_dir / "throughput_comparison.png", "pure vLLM vs EAGLE3 vs best SpecLink-CV throughput", comparison[:36], "label", "throughput")
    write_csv(fig_dir / "itl_p95_comparison.csv", comparison)
    draw_bar_figure(fig_dir / "itl_p95_comparison.png", "p95 ITL comparison", comparison[:36], "label", "itl_p95")
    write_csv(fig_dir / "e2e_p95_comparison.csv", comparison)
    draw_bar_figure(fig_dir / "e2e_p95_comparison.png", "p95 end-to-end latency comparison", comparison[:36], "label", "e2e_p95")

    k_speedup = aggregate_mean([row for row in ok_rows if str(row.get("method", "")).startswith("cv_")], ["K", "method"], "speedup_vs_eagle3")
    write_csv(fig_dir / "k_speedup.csv", k_speedup)
    draw_bar_figure(fig_dir / "k_speedup.png", "K=8 vs K=12 speedup", k_speedup, "label", "speedup_vs_eagle3")

    batch_speedup = aggregate_mean([row for row in ok_rows if str(row.get("method", "")).startswith("cv_")], ["batch_size", "method"], "speedup_vs_eagle3")
    write_csv(fig_dir / "batch_speedup.csv", batch_speedup)
    draw_bar_figure(fig_dir / "batch_speedup.png", "Batch-size speedup", batch_speedup, "label", "speedup_vs_eagle3")

    dist = collect_distribution_rows(rows)
    write_csv(fig_dir / "selected_chunk_size_distribution.csv", dist["selected_h"])
    selected_h_counts = aggregate_mean(dist["selected_h"], ["method", "selected_h"], "selected_h")
    draw_bar_figure(fig_dir / "selected_chunk_size_distribution.png", "Selected chunk size distribution", selected_h_counts, "label", "count")
    write_csv(fig_dir / "first_reject_position_distribution.csv", dist["first_reject"])
    reject_counts: dict[tuple[Any, ...], int] = {}
    for row in dist["first_reject"]:
        key = (row.get("method"), row.get("first_reject_position"))
        reject_counts[key] = reject_counts.get(key, 0) + 1
    reject_rows = [
        {"method": key[0], "first_reject_position": key[1], "count": count, "label": f"{key[0]} {key[1]}"}
        for key, count in sorted(reject_counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
    ]
    write_csv(fig_dir / "first_reject_position_counts.csv", reject_rows)
    draw_bar_figure(fig_dir / "first_reject_position_distribution.png", "First reject position distribution", reject_rows[:40], "label", "count")

    queue_wait_rows = dist["queue_wait"]
    write_csv(fig_dir / "async_queue_wait_distribution.csv", queue_wait_rows)
    queue_wait_summary = aggregate_mean(queue_wait_rows, ["method"], "queue_wait_ms")
    draw_bar_figure(fig_dir / "async_queue_wait_distribution.png", "Async queue wait distribution", queue_wait_summary, "label", "queue_wait_ms")

    skipped = [row for row in ok_rows if str(row.get("method", "")).startswith("cv_")]
    scatter_rows = [{**row, "label": str(row.get("method", ""))} for row in skipped]
    write_csv(fig_dir / "skipped_suffix_vs_speedup.csv", scatter_rows)
    draw_scatter_figure(fig_dir / "skipped_suffix_vs_speedup.png", "Skipped suffix ratio vs speedup", scatter_rows, "skipped_suffix_ratio", "speedup_vs_eagle3", "label")
    write_csv(fig_dir / "extra_tlm_forwards_vs_speedup.csv", scatter_rows)
    draw_scatter_figure(fig_dir / "extra_tlm_forwards_vs_speedup.png", "Extra TLM forwards vs speedup", scatter_rows, "extra_tlm_forwards_per_request", "speedup_vs_eagle3", "label")

    half_conf = aggregate_mean(skipped, ["model", "dataset", "K", "batch_size", "method"], "speedup_vs_eagle3")
    for row in half_conf:
        row.update(method_family(str(row.get("method", ""))))
    write_csv(fig_dir / "fixed_half_vs_confidence_speedup.csv", half_conf)
    half_conf_summary = aggregate_mean(half_conf, ["confidence"], "speedup_vs_eagle3")
    draw_bar_figure(fig_dir / "fixed_half_vs_confidence_speedup.png", "Fixed half vs confidence-guided speedup", half_conf_summary, "label", "speedup_vs_eagle3")
    sync_async_summary = aggregate_mean(half_conf, ["queue"], "speedup_vs_eagle3")
    write_csv(fig_dir / "sync_vs_async_speedup.csv", sync_async_summary)
    draw_bar_figure(fig_dir / "sync_vs_async_speedup.png", "Sync vs async queue speedup", sync_async_summary, "label", "speedup_vs_eagle3")
    roofline_summary = aggregate_mean(half_conf, ["packing"], "speedup_vs_eagle3")
    write_csv(fig_dir / "simple_vs_roofline_speedup.csv", roofline_summary)
    draw_bar_figure(fig_dir / "simple_vs_roofline_speedup.png", "Simple vs roofline packing speedup", roofline_summary, "label", "speedup_vs_eagle3")
    heatmap = aggregate_mean(half_conf, ["confidence", "queue", "packing"], "speedup_vs_eagle3")
    write_csv(fig_dir / "ablation_heatmap.csv", heatmap)
    draw_bar_figure(fig_dir / "ablation_heatmap.png", "Ablation heatmap summary", heatmap, "label", "speedup_vs_eagle3")
    best_rows = list(best.values())
    for row in best_rows:
        row["label"] = f"{row['model']} {row['dataset']} K{row['K']} bs{row['batch_size']} {row['method']}"
    write_csv(fig_dir / "best_configuration_by_scenario.csv", best_rows)
    draw_bar_figure(fig_dir / "best_configuration_by_scenario.png", "Best SpecLink-CV configuration by scenario", best_rows[:40], "label", "throughput")

    confidence_status = [
        {"artifact": "reliability_diagram", "status": "not_generated_by_guidellm_runner", "note": "Use tools/speclink_cv/run_trace_experiment.py for calibration figures."}
    ]
    write_csv(fig_dir / "confidence_calibration_reliability.csv", confidence_status)
    figure_text(fig_dir / "confidence_calibration_reliability.png", "DLM confidence calibration reliability", [confidence_status[0]["note"]])


def write_guidellm_report(root: Path, rows: list[dict[str, Any]], unit_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    report = root / "09_reports" / "SPECLINK_CV_REPORT.md"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    planned_rows = [row for row in rows if row.get("status") == "planned"]
    failed_rows = [
        row
        for row in rows
        if row.get("status") not in {"ok", "planned"}
    ]
    cv_mismatch_rows = [
        row
        for row in ok_rows
        if str(row.get("method", "")).startswith("cv_")
        and str(row.get("speedup_claim_status", "")) != "invalid_no_live_chunking"
        and (safe_float(row.get("exact_match_vs_eagle3")) or 0.0) < 1.0
    ]
    invalid_speedup_rows = [
        row
        for row in ok_rows
        if str(row.get("method", "")).startswith("cv_")
        and str(row.get("speedup_claim_status", "")) != "valid_quality_preserving_chunked"
    ]
    write_csv(root / "09_reports" / "correctness_warnings.csv", cv_mismatch_rows)
    write_csv(root / "09_reports" / "speedup_claim_warnings.csv", invalid_speedup_rows)
    best = best_cv_rows(ok_rows)
    global_candidates = list(best.values())
    global_best = max(global_candidates, key=lambda row: safe_float(row.get("throughput")) or -1.0, default=None)
    actual_models = ",".join(sorted({str(row.get("model", "")) for row in rows if row.get("model")}))
    actual_datasets = ",".join(sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")}))
    actual_ks = ",".join(str(item) for item in sorted({int(row["K"]) for row in rows if str(row.get("K", "")).isdigit()}))
    actual_batches = ",".join(str(item) for item in sorted({int(row["batch_size"]) for row in rows if str(row.get("batch_size", "")).isdigit()}))
    measurement_types = ",".join(
        sorted({str(row.get("measurement_type", "")) for row in rows if row.get("measurement_type")})
    )
    lines = [
        "# SPECLINK_CV_REPORT",
        "",
        "## Run Scope",
        "",
        f"- output root: `{root}`",
        f"- cases: {len(rows)}",
        f"- ok: {len(ok_rows)}",
        f"- planned: {len(planned_rows)}",
        f"- failed/missing: {len(failed_rows)}",
        f"- models: `{actual_models or args.models}`",
        f"- datasets: `{actual_datasets or args.datasets}`",
        f"- K: `{actual_ks or args.ks}`",
        f"- batch sizes: `{actual_batches or args.batch_sizes}`",
        f"- measurement types: `{measurement_types or args.benchmark_mode}`",
        "",
        "## Unit Tests",
        "",
    ]
    for row in unit_rows:
        lines.append(f"- {row['test']}: {row['status']} rc={row['returncode']}")
    lines.extend(["", "## Main Findings", ""])
    if global_best:
        best_speedup = safe_float(global_best.get("speedup_vs_eagle3"))
        best_relation = (
            "throughput improvement"
            if best_speedup is not None and best_speedup > 1.0
            else "throughput slowdown"
        )
        lines.append(
            "- best valid SpecLink-CV row by the `throughput` column: "
            f"{global_best['model']} {global_best['dataset']} K={global_best['K']} "
            f"bs={global_best['batch_size']} {global_best['method']} "
            f"throughput={global_best.get('throughput')} "
            f"measurement_type={global_best.get('measurement_type')} "
            f"speedup_vs_eagle3={global_best.get('speedup_vs_eagle3')} "
            f"exact_match_vs_eagle3={global_best.get('exact_match_vs_eagle3')} "
            f"({best_relation} vs EAGLE3 one-shot)"
        )
    else:
        lines.append("- no successful SpecLink-CV row with exact-match evidence was available.")
    lines.extend(
        [
            "- `exact_match_vs_eagle3` in this report is cross-run text-level GuideLLM output matching after aligning requests by `request_args`. It is a drift diagnostic, not the default quality gate.",
            "- For `measurement_type=steady_state_saturated`, `throughput` means saturated output tokens/s at concurrency `batch_size`, counted only in the fixed measurement window. Warmup and drain are excluded.",
            "- For `measurement_type=guidellm_end_to_end`, `throughput` is finite-request GuideLLM output tokens/s and includes tail drain; use it for output/quality diagnostics, not for final serving-throughput claims.",
            "- For `math_reasoning`, the quality gate uses answer exact match against the dataset reference and allows small output drift when math answer quality is preserved. Use longer `--max-tokens` for reliable math EM; very short outputs are marked `quality_unreliable_short_outputs`.",
            "- MTBench currently has `reference=null` in the local dataset and is not part of the primary quality conclusion unless a judge is added; proxy similarity rows are reported only as diagnostics.",
            "- `cv_*` methods use the regular V1 scheduler with vLLM async scheduling disabled, while `SPECLINK_CV_ASYNC_QUEUE` controls the experiment's own verification queue.",
            "- Exact-safe mode is enabled by default. Because h<K chunked verifier shapes have shown greedy argmax drift versus full-K EAGLE3 one-shot, `SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING=0` falls CV rows back to EAGLE3 one-shot and the speedup gate marks them `invalid_no_live_chunking`.",
            "- Token-id exact h<K debugging still uses `VLLM_BATCH_INVARIANT=1` in addition to `--allow-shape-drift-chunking`. Performance rows should normally leave `VLLM_BATCH_INVARIANT=0` and use math quality preservation, not bit-for-bit EAGLE3 equality, as the quality gate.",
            "- `SPECLINK_CV_ALLOW_BATCHED_PREFIX=1` allows multiple prefix chunks per scheduler step. `SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1` does the same for accepted-prefix suffix chunks; without suffix batching, CV can lose most of the theoretical verifier saving to singleton suffix steps.",
            "- Any live-chunked `cv_*` row with a failing quality gate is excluded from speedup claims. Exact-safe fallback rows are still classified as `invalid_no_live_chunking`.",
            "- Sync conservative fallback rows can be exact but are not chunked verification wins. A `cv_*` row is eligible for best-CV selection only when `speedup_claim_status=valid_quality_preserving_chunked`; this means the row is a quality-preserving performance comparison, not necessarily a speedup over EAGLE3.",
            "- When suffix verification rejects, the conservative live path drops the same-step drafter output and runs dense realignment steps before resuming speculation. These steps are reported as `dense_realign_steps` and included in `extra_tlm_forwards_per_request`.",
            "",
            "## Correctness Warnings",
            "",
        ]
    )
    if not cv_mismatch_rows:
        lines.append("- none recorded.")
    else:
        for row in cv_mismatch_rows[:80]:
            lines.append(
                f"- {row.get('model')} {row.get('dataset')} K={row.get('K')} "
                f"bs={row.get('batch_size')} {row.get('method')}: "
                f"exact_match_vs_eagle3={row.get('exact_match_vs_eagle3')} "
                f"common_requests={row.get('exact_match_common_requests')}"
            )
    lines.extend(
        [
            "",
            "## Speedup Claim Gate",
            "",
        ]
    )
    successful_cv_rows = [
        row
        for row in ok_rows
        if str(row.get("method", "")).startswith("cv_")
    ]
    if not successful_cv_rows:
        lines.append("- no successful `cv_*` rows were recorded.")
    elif not invalid_speedup_rows:
        lines.append("- all successful `cv_*` rows are valid for performance comparison under the correctness/chunking gate.")
    else:
        for row in invalid_speedup_rows[:80]:
            lines.append(
                f"- {row.get('model')} {row.get('dataset')} K={row.get('K')} "
                f"bs={row.get('batch_size')} {row.get('method')}: "
                f"{row.get('speedup_claim_status')} "
                f"exact_match_vs_eagle3={row.get('exact_match_vs_eagle3')}"
            )
    lines.extend(
        [
            "",
            "## Failures",
            "",
        ]
    )
    if not failed_rows:
        lines.append("- none recorded.")
    else:
        for row in failed_rows[:80]:
            lines.append(f"- {row.get('model')} {row.get('dataset')} K={row.get('K')} bs={row.get('batch_size')} {row.get('method')}: {row.get('error', row.get('status'))}")
    lines.extend(["", "## Planned Cases", ""])
    if not planned_rows:
        lines.append("- none recorded.")
    else:
        for row in planned_rows[:80]:
            lines.append(
                f"- {row.get('model')} {row.get('dataset')} K={row.get('K')} "
                f"bs={row.get('batch_size')} {row.get('method')}"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `09_reports/summary_metrics.csv` and `.json`: primary table.",
            "- `08_figures/*.csv`: source data for every generated figure.",
            "- `runs/*/vllm_server.log`, `guidellm_output.log`/`steady_state_output.log`, `guidellm_results.json`/`steady_state_results.json`, and `speclink_cv_profile.jsonl`: raw evidence.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=EVAL_DIR / "results" / f"speclink_cv_{timestamp}")
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--datasets", default="math,mtbench")
    parser.add_argument("--ks", default="8,12")
    parser.add_argument("--batch-sizes", default="8,16,32")
    parser.add_argument(
        "--methods",
        default="pure_vllm,eagle3_oneshot,cv_half_sync_simple,cv_half_sync_roofline,cv_half_async_simple,cv_half_async_roofline,cv_conf_sync_simple,cv_conf_sync_roofline,cv_conf_async_simple,cv_conf_async_roofline",
    )
    parser.add_argument("--smoke", action="store_true", help="Override to a small one-case matrix.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed run directories that already contain the expected "
            "benchmark result JSON for the selected --benchmark-mode."
        ),
    )
    parser.add_argument("--case-offset", type=int, default=0, help="Skip this many planned cases before running/analyzing.")
    parser.add_argument("--case-limit", type=int, default=0, help="Run/analyze at most this many cases after --case-offset; 0 means no limit.")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8050")))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per request; set 0 to omit max_tokens from the request body.",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["guidellm", "steady_state"],
        default="guidellm",
        help=(
            "guidellm runs a finite-request GuideLLM benchmark for output/quality "
            "inspection. steady_state runs a closed-loop saturated throughput "
            "client where batch_size=N means concurrency N and drain tokens are "
            "excluded from output tokens/s."
        ),
    )
    parser.add_argument(
        "--steady-state-warmup-s",
        type=float,
        default=30.0,
        help="Warmup seconds before the fixed steady-state measurement window.",
    )
    parser.add_argument(
        "--steady-state-measurement-s",
        type=float,
        default=120.0,
        help="Fixed steady-state measurement window in seconds.",
    )
    parser.add_argument(
        "--steady-state-cooldown-s",
        type=float,
        default=30.0,
        help="Drain timeout after the steady-state measurement window; excluded from throughput.",
    )
    parser.add_argument(
        "--steady-state-bucket-s",
        type=float,
        default=1.0,
        help="Bucket width for steady_state_buckets.csv.",
    )
    parser.add_argument(
        "--steady-state-max-prompts",
        type=int,
        default=0,
        help="Number of dataset prompts cycled by the steady-state client; 0 reuses --max-requests.",
    )
    parser.add_argument(
        "--steady-state-timeout",
        type=float,
        default=1800.0,
        help="Per-request HTTP timeout for the steady-state client.",
    )
    parser.add_argument(
        "--steady-state-ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass ignore_eos to the OpenAI request in steady-state mode so fixed "
            "output lengths do not differ across methods."
        ),
    )
    parser.add_argument(
        "--steady-state-allow-final-usage-fallback",
        action="store_true",
        help=(
            "Allow assigning a whole request's final completion_tokens to its "
            "finish time if streaming continuous usage is unavailable. This is "
            "less accurate and should not be used for final throughput claims."
        ),
    )
    parser.add_argument(
        "--profile-max-events",
        type=int,
        default=200,
        help=(
            "Cap regular SpecLink-CV profile JSONL rows per run. Use 0 only "
            "for a short representative diagnostic."
        ),
    )
    parser.add_argument(
        "--log-max-events",
        type=int,
        default=200,
        help=(
            "Cap regular SpecLink-CV event JSONL rows per run. Use 0 only "
            "for a short representative diagnostic."
        ),
    )
    parser.add_argument(
        "--analysis-profile-max-rows",
        type=int,
        default=2000,
        help=(
            "Maximum profile JSONL rows read during summary/figure analysis; "
            "0 disables the analysis cap."
        ),
    )
    parser.add_argument("--request-type", default="chat_completions")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--math-quality-drop-tolerance",
        type=float,
        default=0.02,
        help=(
            "Allowed math answer EM drop versus EAGLE3 before a live h<K "
            "row is excluded from speedup claims."
        ),
    )
    parser.add_argument(
        "--mtbench-similarity-threshold",
        type=float,
        default=0.65,
        help="Diagnostic proxy threshold only; local MTBench has no judge/reference.",
    )
    parser.add_argument(
        "--mtbench-min-length-ratio",
        type=float,
        default=0.5,
        help="Diagnostic proxy lower length-ratio bound for MTBench rows.",
    )
    parser.add_argument(
        "--mtbench-max-length-ratio",
        type=float,
        default=1.8,
        help="Diagnostic proxy upper length-ratio bound for MTBench rows.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--allow-cv-cudagraph",
        action="store_true",
        help=(
            "Experimental: do not force --enforce-eager for cv_* methods. "
            "Default keeps eager mode for CV state-debug stability."
        ),
    )
    parser.add_argument(
        "--disable-vllm-async-scheduling",
        action="store_true",
        help="Pass --no-async-scheduling to every vLLM serve run, not only cv_* methods.",
    )
    parser.add_argument("--health-check-timeout", type=int, default=1800)
    parser.add_argument(
        "--gpu-util-sampling-ms",
        type=int,
        default=int(os.environ.get("GPU_UTIL_SAMPLING_MS", "0")),
        help=(
            "Sample nvidia-smi GPU utilization during each benchmark run into "
            "runs/*/gpu_util.csv. 0 disables sampling."
        ),
    )
    parser.add_argument(
        "--nsys-profile",
        action="store_true",
        help=(
            "Wrap vLLM serve in Nsight Systems and bracket the benchmark client with "
            "/start_profile and /stop_profile. Writes vllm_guidellm_profile*.nsys-rep "
            "and nsys_stats.txt in each run directory."
        ),
    )
    parser.add_argument(
        "--nsys-output-name",
        default="vllm_guidellm_profile",
        help="Nsight Systems output prefix relative to each run directory.",
    )
    parser.add_argument(
        "--nsys-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `nsys stats` on generated .nsys-rep files after server shutdown.",
    )
    parser.add_argument(
        "--keep-nsys-sqlite",
        action="store_true",
        help=(
            "Keep the intermediate .sqlite file exported by `nsys stats`. "
            "Default removes it to avoid large temporary artifacts."
        ),
    )
    parser.add_argument("--util-threshold", type=float, default=0.6)
    parser.add_argument("--max-queue-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-verify-tokens-per-step", type=int, default=0)
    parser.add_argument("--max-verify-seqs-per-step", type=int, default=0)
    parser.add_argument(
        "--allow-batched-prefix-verification",
        action="store_true",
        help=(
            "Experimental: allow several SpecLink-CV prefix chunks in one "
            "TLM step. The default conservative exact mode caps this at one. "
            "Suffix batching follows this flag unless overridden with "
            "`--env SPECLINK_CV_ALLOW_BATCHED_SUFFIX=0/1`."
        ),
    )
    parser.add_argument(
        "--allow-shape-drift-chunking",
        action="store_true",
        help=(
            "Experimental: allow live h<K chunking despite verifier-shape "
            "argmax drift evidence. Default exact-safe mode falls back to "
            "one-shot verification."
        ),
    )
    parser.add_argument("--calibration-path", default="")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra environment variable for both vLLM and benchmark child "
            "processes. Repeatable. Use VLLM_BATCH_INVARIANT=1 for token-id "
            "exact h<K debugging; keep it unset/0 for quality-gated "
            "performance runs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    global ANALYSIS_PROFILE_MAX_ROWS
    args = parse_args()
    ANALYSIS_PROFILE_MAX_ROWS = int(args.analysis_profile_max_rows)
    args.output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (Path.cwd() / args.output_root).resolve()
    )
    if args.calibration_path:
        args.calibration_path = str(
            Path(args.calibration_path).resolve()
            if not Path(args.calibration_path).is_absolute()
            else Path(args.calibration_path)
        )
    if args.benchmark_mode == "steady_state":
        if args.max_tokens <= 0:
            raise SystemExit("--benchmark-mode steady_state requires --max-tokens > 0")
        if args.steady_state_warmup_s < 0:
            raise SystemExit("--steady-state-warmup-s must be non-negative")
        if args.steady_state_measurement_s <= 0:
            raise SystemExit("--steady-state-measurement-s must be positive")
        if args.steady_state_cooldown_s < 0:
            raise SystemExit("--steady-state-cooldown-s must be non-negative")
        if args.steady_state_bucket_s <= 0:
            raise SystemExit("--steady-state-bucket-s must be positive")
    if args.smoke:
        args.models = "qwen3_8b"
        args.datasets = "math"
        args.ks = "8"
        args.batch_sizes = "1"
        args.methods = "eagle3_oneshot,cv_half_async_roofline"
        args.max_requests = min(args.max_requests, 4)
    cases_all = make_cases(args)
    args.total_planned_cases = len(cases_all)
    if args.case_offset < 0:
        raise SystemExit("--case-offset must be >= 0")
    if args.case_limit < 0:
        raise SystemExit("--case-limit must be >= 0")
    case_end = None if args.case_limit == 0 else args.case_offset + args.case_limit
    cases = cases_all[args.case_offset:case_end]
    args.selected_cases = len(cases)
    args.case_slice_requested = int(args.case_offset != 0 or args.case_limit != 0)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    create_result_tree(root)
    unit_rows: list[dict[str, Any]] = []
    if not args.analyze_only:
        write_setup_artifacts(root, args)
        if not args.dry_run and not args.skip_unit_tests:
            unit_rows = run_unit_tests(root)
    elif (root / "02_unit_tests" / "unit_test_summary.csv").exists():
        with (root / "02_unit_tests" / "unit_test_summary.csv").open(encoding="utf-8") as f:
            unit_rows = list(csv.DictReader(f))
    if args.analyze_only:
        rows = collect_run_rows(root, cases)
    else:
        slice_rows: list[dict[str, Any]] = []
        commands: list[str] = []
        for idx, case in enumerate(cases, 1):
            absolute_idx = args.case_offset + idx
            print(
                f"[INFO] Case {idx}/{len(cases)} (planned {absolute_idx}/{len(cases_all)}): "
                f"{case['model_label']} {case['dataset_label']} K={case['K']} "
                f"bs={case['batch_size']} {case['method']}"
            )
            row = run_case(case, args, root, commands)
            slice_rows.append(row)
            write_csv(root / "status_current_slice.csv", slice_rows)
        (root / "scripts/run_commands.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\ncd " + str(EVAL_DIR) + "\n\n" + "\n".join(commands),
            encoding="utf-8",
        )
        (root / "scripts/run_commands.sh").chmod(0o755)
        rows = slice_rows if args.dry_run else collect_run_rows(root, cases_all)
    write_csv(root / "status.csv", rows)
    fill_speedups(rows)
    fill_exact_matches(rows)
    fill_quality_metrics(rows, args)
    fill_speedup_claim_validity(rows)
    write_csv(root / "summary_metrics.csv", rows)
    write_json(root / "summary_metrics.json", {"rows": rows})
    write_csv(root / "09_reports" / "summary_metrics.csv", rows)
    write_json(root / "09_reports" / "summary_metrics.json", {"rows": rows})
    build_figure_artifacts(root, rows)
    write_guidellm_report(root, rows, unit_rows, args)
    report_lines = [
        "# SpecLink-CV GuideLLM Matrix",
        "",
        f"- cases: {len(rows)}",
        f"- total_planned_cases: {len(cases_all)}",
        f"- case_offset: {args.case_offset}",
        f"- case_limit: {args.case_limit}",
        f"- dry_run: {args.dry_run}",
        f"- analyze_only: {args.analyze_only}",
        f"- resume: {args.resume}",
        f"- output_root: `{root}`",
        "",
        "This runner can record either finite-request GuideLLM diagnostics or closed-loop saturated serving throughput. For steady-state rows, `batch_size=N` means concurrency N and `throughput` is saturated output tokens/s over the fixed measurement window, with warmup and drain excluded. For GuideLLM rows, `throughput` is finite-request end-to-end output tokens/s and should not be used as final serving-throughput evidence.",
        "",
        "## Status Counts",
        "",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status", ""))] = counts.get(str(row.get("status", "")), 0) + 1
    for key, value in sorted(counts.items()):
        report_lines.append(f"- {key}: {value}")
    (root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote matrix summary: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
