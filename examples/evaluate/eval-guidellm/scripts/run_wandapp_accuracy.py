#!/usr/bin/env python3
"""Token-dense threshold quality sweep for TLM-only 2:4.

All serving runs use vLLM + EAGLE3 speculative decoding. Only the target/base
large model is masked; the EAGLE3 drafter remains dense.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = EVAL_ROOT / "results"
RESULTS_BAK_ROOT = EVAL_ROOT / "results.bak"
DEFAULT_CACHE_ROOT = EVAL_ROOT / "data" / "c4_calibration" / "wandapp_masks"

sys.path.insert(0, str(SCRIPT_DIR))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    DatasetPack,
    LAYER_SENSITIVITY_DEFAULT_MODELS,
    append_jsonl,
    ensure_quality_dependencies,
    failed_metric,
    first_text_field,
    load_datasets,
    metric_value,
    parse_csv_list,
    parse_model_id_overrides,
    read_jsonl,
    sample_rows,
    set_seed,
    write_json,
)
from run_structured_24_spec_quality import (  # noqa: E402
    ACCURACY_DATASETS,
    DEFAULT_BASE_MODELS,
    EAGLE3_SPECULATORS,
    SparseCase,
    add_local_no_proxy,
    configure_local_no_proxy,
    run_accuracy_dataset,
    scrape_spec_metrics,
    metric_delta,
    start_vllm_server,
    stop_process,
)

GENERATED_PPL_DATASETS = {
    "mtbench",
    "humaneval",
    "dolly",
    "dolly_open_qa",
    "dolly_creative_writing",
    "dolly_summarization",
}

TOKENIZER_CACHE: dict[tuple[str, bool], Any] = {}

DOLLY_CATEGORY_DATASETS = {
    "dolly_open_qa": "open_qa",
    "dolly_creative_writing": "creative_writing",
    "dolly_summarization": "summarization",
}

DEFAULT_THRESHOLD_METHODS = ",".join(
    f"token_dense_t{threshold:02d}" for threshold in range(11)
)

CSV_FIELDS = [
    "model_label",
    "model_id",
    "method",
    "dataset",
    "metric_name",
    "metric_type",
    "dense_metric_value",
    "sparse_metric_value",
    "delta_vs_dense",
    "ratio_vs_dense",
    "dense_accuracy",
    "sparse_accuracy",
    "accuracy_drop",
    "recovery_vs_activation_aware",
    "dense_generated_ppl",
    "sparse_generated_ppl",
    "generated_ppl_ratio_vs_dense",
    "dense_generated_nll",
    "sparse_generated_nll",
    "generated_nll_delta_vs_dense",
    "num_examples",
    "output_tokens",
    "generated_logprob_tokens",
    "generated_logprob_missing",
    "request_error_count",
    "spec_acceptance_rate_case",
    "spec_accepted_tokens_case",
    "spec_draft_tokens_case",
    "effective_sparse_fraction",
    "mask_cache",
    "mask_cache_method",
    "missing_cached_mask_modules",
    "token_dense_threshold",
    "token_dense_dense_draft_fraction",
    "token_dense_sparse_draft_fraction",
    "token_dense_missing_score_tokens",
    "failed",
    "error",
]


@dataclass(frozen=True)
class MethodConfig:
    label: str
    base_method: str
    policy: str = "all_sparse"
    keep_n: int = 0
    token_dense_threshold: float | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_ppl(nll: float | None) -> float | None:
    if nll is None:
        return None
    try:
        if nll > 80:
            return float("inf")
        return math.exp(nll)
    except OverflowError:
        return float("inf")


def parse_method_config(label: str) -> MethodConfig:
    token_dense_match = re.fullmatch(r"token_dense_t(\d{1,3})", label)
    if token_dense_match:
        raw = int(token_dense_match.group(1))
        threshold = raw / 10.0 if raw <= 10 else raw / 100.0
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"unsupported token-dense threshold in {label!r}")
        return MethodConfig(
            label=label,
            base_method="token_dense",
            policy="all_sparse",
            token_dense_threshold=threshold,
        )
    match = re.fullmatch(
        r"(activation_aware|wandapp_rgs|wandapp_ro)"
        r"(?:_(keep_first_last|keep_first|keep_last)_(\d+))?",
        label,
    )
    if not match:
        raise ValueError(
            f"unsupported method label {label!r}; use activation_aware, "
            "wandapp_rgs, wandapp_ro, or METHOD_keep_first_N variants"
        )
    base_method = match.group(1)
    policy = match.group(2) or "all_sparse"
    keep_n = int(match.group(3) or "0")
    return MethodConfig(label=label, base_method=base_method, policy=policy, keep_n=keep_n)


def dataset_limit(args: argparse.Namespace, dataset_name: str) -> int | None:
    if dataset_name == "gsm8k":
        return args.gsm8k_num_examples
    if dataset_name == "math_reasoning":
        return args.math_num_examples
    if dataset_name == "mtbench":
        return args.mtbench_num_examples
    if dataset_name == "humaneval":
        return args.humaneval_num_examples
    if dataset_name == "dolly" or dataset_name in DOLLY_CATEGORY_DATASETS:
        return args.dolly_num_examples
    return None


def load_dolly_category_dataset(
    dataset_name: str,
    *,
    limit: int | None,
    seed: int,
) -> DatasetPack:
    category = DOLLY_CATEGORY_DATASETS[dataset_name]
    path = EVAL_ROOT / "data" / "dolly" / "by_category" / f"{category}.jsonl"
    try:
        raw_rows = read_jsonl(path)
        rows = []
        for idx, row in enumerate(raw_rows):
            instruction = str(row.get("instruction", ""))
            context = str(row.get("context", "") or "")
            prompt = first_text_field(row, ("prompt", "question", "input"))
            if not prompt:
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
                    "category": category,
                }
            )
        return DatasetPack(dataset_name, sample_rows(rows, limit, seed))
    except Exception as exc:  # noqa: BLE001
        return DatasetPack(dataset_name, [], error=str(exc))


def load_selected_datasets(args: argparse.Namespace) -> dict[str, DatasetPack]:
    selected = parse_csv_list(args.datasets)
    standard_names = [
        name
        for name in selected
        if name in {"gsm8k", "math_reasoning", "mtbench", "humaneval", "dolly"}
    ]
    datasets: dict[str, DatasetPack] = {}
    if standard_names:
        standard_args = argparse.Namespace(**vars(args))
        standard_args.datasets = ",".join(standard_names)
        datasets.update(load_datasets(standard_args))
    for name in selected:
        if name in DOLLY_CATEGORY_DATASETS:
            datasets[name] = load_dolly_category_dataset(
                name,
                limit=dataset_limit(args, name),
                seed=args.seed,
            )
        elif name not in datasets:
            datasets[name] = DatasetPack(name, [], error=f"unknown dataset: {name}")
    return datasets


def method_env(
    args: argparse.Namespace,
    *,
    model_label: str,
    method: MethodConfig,
    stats_path: Path,
) -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    env["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    if method.label == "dense":
        env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
        return env
    env.update(
        {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": model_label,
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(
                args.calibration_cache_root.resolve()
            ),
            "SPECLINK_STRUCTURED_24_POLICY": method.policy,
            "SPECLINK_STRUCTURED_24_KEEP_N": str(method.keep_n),
            "SPECLINK_STRUCTURED_24_STATS_PATH": str(stats_path.resolve()),
        }
    )
    if method.base_method == "token_dense":
        env.update(
            {
                "SPECLINK_TOKEN_DENSE_ENABLE": "1",
                "SPECLINK_TOKEN_DENSE_MODE": "high_confidence_dense",
                "SPECLINK_TOKEN_DENSE_THRESHOLD": str(
                    method.token_dense_threshold
                    if method.token_dense_threshold is not None
                    else 0.7
                ),
                "SPECLINK_TOKEN_DENSE_STATS_PATH": str(
                    (stats_path.parent / "token_dense_stats.jsonl").resolve()
                ),
                "SPECLINK_TOKEN_DENSE_STATS_DETAIL": "0",
            }
        )
    if method.base_method in {"wandapp_rgs", "wandapp_ro"}:
        cache_path = args.cache_root / f"{model_label}_{method.base_method}.pt"
        env["SPECLINK_STRUCTURED_24_MASK_CACHE"] = str(cache_path.resolve())
        env["SPECLINK_STRUCTURED_24_CACHE_STRICT"] = "1"
    return env


def summarize_token_dense_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    dense = 0
    sparse = 0
    draft = 0
    missing = 0
    steps = 0
    threshold_value = None
    latest_summary: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "verify_token_mask_summary":
                latest_summary = record
                continue
            if event != "verify_token_mask":
                continue
            steps += 1
            dense += int(record.get("dense_draft_tokens") or 0)
            sparse += int(record.get("sparse_draft_tokens") or 0)
            draft += int(record.get("total_draft_tokens") or 0)
            missing += int(record.get("missing_score_tokens") or 0)
            threshold_value = record.get("threshold", threshold_value)
    if latest_summary is not None:
        total_draft = int(latest_summary.get("total_draft_tokens") or 0)
        dense_draft = int(latest_summary.get("dense_draft_tokens") or 0)
        sparse_draft = int(latest_summary.get("sparse_draft_tokens") or 0)
        return {
            "steps": int(latest_summary.get("steps") or 0),
            "threshold": latest_summary.get("threshold"),
            "total_draft_tokens": total_draft,
            "dense_draft_tokens": dense_draft,
            "sparse_draft_tokens": sparse_draft,
            "missing_score_tokens": int(latest_summary.get("missing_score_tokens") or 0),
            "dense_draft_fraction": dense_draft / total_draft if total_draft else None,
            "sparse_draft_fraction": sparse_draft / total_draft if total_draft else None,
        }
    return {
        "steps": steps,
        "threshold": threshold_value,
        "total_draft_tokens": draft,
        "dense_draft_tokens": dense,
        "sparse_draft_tokens": sparse,
        "missing_score_tokens": missing,
        "dense_draft_fraction": dense / draft if draft else None,
        "sparse_draft_fraction": sparse / draft if draft else None,
    }


def metrics_complete(case_dir: Path, expected_datasets: set[str]) -> bool:
    path = case_dir / "metrics.json"
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    if data.get("status") != "ok":
        return False
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        return False
    if expected_datasets and not expected_datasets.issubset(set(datasets)):
        return False
    return not any(
        isinstance(metric, dict)
        and (metric.get("failed") or int(metric.get("request_error_count") or 0) > 0)
        for metric in datasets.values()
    )


def max_tokens_for_dataset(args: argparse.Namespace, dataset_name: str) -> int:
    if dataset_name in ACCURACY_DATASETS:
        return args.accuracy_max_tokens
    return args.ppl_max_tokens


def get_tokenizer(args: argparse.Namespace, model_id: str) -> Any:
    key = (model_id, bool(args.trust_remote_code))
    tokenizer = TOKENIZER_CACHE.get(key)
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def prepare_prompt_for_request(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    prompt: str,
    requested_max_tokens: int,
) -> tuple[str, int, int, int, bool, bool]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_prompt_tokens = len(token_ids)
    prompt_tokens = original_prompt_tokens
    prompt_truncated = False
    max_model_len = int(args.max_model_len)
    safety_margin = int(args.context_length_safety_margin)

    if prompt_tokens + safety_margin >= max_model_len:
        target_prompt_tokens = max(1, max_model_len - int(requested_max_tokens) - safety_margin)
        if prompt_tokens > target_prompt_tokens:
            head_tokens = target_prompt_tokens // 4
            tail_tokens = target_prompt_tokens - head_tokens
            if head_tokens > 0:
                token_ids = token_ids[:head_tokens] + token_ids[-tail_tokens:]
            else:
                token_ids = token_ids[-tail_tokens:]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > target_prompt_tokens:
                token_ids = token_ids[-target_prompt_tokens:]
                prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens = len(token_ids)
            prompt_truncated = True

    remaining = max_model_len - prompt_tokens - safety_margin
    effective = min(int(requested_max_tokens), max(0, remaining))
    return (
        prompt,
        effective,
        original_prompt_tokens,
        prompt_tokens,
        effective != int(requested_max_tokens),
        prompt_truncated,
    )


def post_completion_with_logprobs(
    *,
    port: int,
    model_id: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    timeout_s: float,
    logprobs: int,
) -> tuple[str, int | None, str | None, list[float | None], str]:
    body = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "request_id": request_id,
        "logprobs": logprobs,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return "", None, None, [], exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return "", None, None, [], repr(exc)
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    text = str(choice.get("text", "")) if choice else ""
    finish_reason = choice.get("finish_reason")
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    logprob_obj = choice.get("logprobs") or {}
    token_logprobs = logprob_obj.get("token_logprobs") or []
    parsed_logprobs: list[float | None] = []
    for value in token_logprobs:
        if value is None:
            parsed_logprobs.append(None)
        else:
            parsed_logprobs.append(float(value))
    return (
        text,
        int(completion_tokens) if completion_tokens is not None else None,
        str(finish_reason) if finish_reason is not None else None,
        parsed_logprobs,
        "",
    )


def run_generated_ppl_dataset(
    args: argparse.Namespace,
    *,
    port: int,
    model_id: str,
    model_label: str,
    case: SparseCase,
    dataset_name: str,
    pack: DatasetPack,
    case_dir: Path,
) -> dict[str, Any]:
    if pack.error:
        return failed_metric(dataset_name, f"dataset_load_failed: {pack.error}")
    details: list[dict[str, Any]] = []
    started = time.time()
    tokenizer = get_tokenizer(args, model_id)

    def evaluate_one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        request_id = f"tdppl-{model_label}-{case.label}-{dataset_name}-{idx:05d}"
        prompt = str(row.get("prompt", ""))
        requested_max_tokens = max_tokens_for_dataset(args, dataset_name)
        (
            request_prompt,
            effective_max_tokens,
            original_prompt_tokens,
            prompt_tokens,
            max_tokens_clipped,
            prompt_truncated,
        ) = prepare_prompt_for_request(
            args,
            tokenizer=tokenizer,
            prompt=prompt,
            requested_max_tokens=requested_max_tokens,
        )
        if effective_max_tokens <= 0:
            return {
                "idx": idx,
                "id": row.get("id"),
                "request_id": request_id,
                "error": (
                    f"prompt too long for max_model_len={args.max_model_len}: "
                    f"original_prompt_tokens={original_prompt_tokens}, "
                    f"prompt_tokens={prompt_tokens}, "
                    f"safety_margin={args.context_length_safety_margin}"
                ),
                "original_prompt_tokens": original_prompt_tokens,
                "prompt_tokens": prompt_tokens,
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": 0,
                "max_tokens_clipped": True,
                "prompt_truncated": prompt_truncated,
                "completion_tokens": 0,
                "logprob_tokens": 0,
                "missing_logprobs": 0,
                "nll_sum": 0.0,
            }
        text, completion_tokens, finish_reason, logprobs, error = post_completion_with_logprobs(
            port=port,
            model_id=model_id,
            prompt=request_prompt,
            max_tokens=effective_max_tokens,
            request_id=request_id,
            timeout_s=args.request_timeout_s,
            logprobs=args.completion_logprobs,
        )
        if error:
            return {
                "idx": idx,
                "id": row.get("id"),
                "request_id": request_id,
                "error": error,
                "original_prompt_tokens": original_prompt_tokens,
                "prompt_tokens": prompt_tokens,
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": effective_max_tokens,
                "max_tokens_clipped": max_tokens_clipped,
                "prompt_truncated": prompt_truncated,
                "completion_tokens": 0,
                "logprob_tokens": 0,
                "missing_logprobs": 0,
                "nll_sum": 0.0,
            }
        usable = [value for value in logprobs if value is not None and math.isfinite(value)]
        nll_sum = -sum(usable)
        logprob_tokens = len(usable)
        missing = len(logprobs) - logprob_tokens
        nll = nll_sum / logprob_tokens if logprob_tokens else None
        return {
            "idx": idx,
            "id": row.get("id"),
            "request_id": request_id,
            "prompt": row.get("prompt"),
            "generation": text,
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_tokens": prompt_tokens,
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "max_tokens_clipped": max_tokens_clipped,
            "prompt_truncated": prompt_truncated,
            "completion_tokens": completion_tokens or 0,
            "finish_reason": finish_reason,
            "logprob_tokens": logprob_tokens,
            "missing_logprobs": missing,
            "nll": nll,
            "ppl": finite_ppl(nll),
            "nll_sum": nll_sum,
            "error": "",
        }

    workers = max(1, int(args.ppl_concurrency))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, idx, row) for idx, row in enumerate(pack.rows)]
        for future in as_completed(futures):
            details.append(future.result())

    details.sort(key=lambda item: int(item.get("idx", 0)))
    append_jsonl(case_dir / f"generations_{dataset_name}.jsonl", details)
    errors = [item for item in details if item.get("error")]
    total_logprob_tokens = sum(int(item.get("logprob_tokens") or 0) for item in details)
    total_missing = sum(int(item.get("missing_logprobs") or 0) for item in details)
    total_output_tokens = sum(int(item.get("completion_tokens") or 0) for item in details)
    total_nll = sum(float(item.get("nll_sum") or 0.0) for item in details)
    clipped_count = sum(1 for item in details if item.get("max_tokens_clipped"))
    truncated_count = sum(1 for item in details if item.get("prompt_truncated"))
    nll = total_nll / total_logprob_tokens if total_logprob_tokens else None
    ppl = finite_ppl(nll)
    metric_name = f"{dataset_name}_generated_ppl"
    return {
        "metric_name": metric_name,
        "metric_type": "generated_ppl",
        "value": ppl,
        metric_name: ppl,
        "generated_nll": nll,
        "generated_ppl": ppl,
        "num_examples": len(details) - len(errors),
        "output_tokens": total_output_tokens,
        "generated_logprob_tokens": total_logprob_tokens,
        "generated_logprob_missing": total_missing,
        "request_error_count": len(errors),
        "max_tokens_clipped_count": clipped_count,
        "prompt_truncated_count": truncated_count,
        "elapsed_sec": round(time.time() - started, 3),
        "metric_source": "vllm_eagle3_speculative_generation_logprobs",
        "failed": total_logprob_tokens == 0,
        "error": f"{len(errors)} request errors" if errors else "",
    }


def run_method(
    args: argparse.Namespace,
    *,
    model_label: str,
    model_id: str,
    speculator_model: str,
    method: MethodConfig,
    datasets: dict[str, Any],
    case_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir.mkdir(parents=True, exist_ok=True)
    expected_datasets = {
        name
        for name in datasets
        if name in ACCURACY_DATASETS or name in GENERATED_PPL_DATASETS
    }
    if args.resume and metrics_complete(case_dir, expected_datasets):
        data = load_json(case_dir / "metrics.json")
        mask_stats = data.get("mask_stats", {})
        if isinstance(mask_stats, dict):
            mask_stats["token_dense_stats"] = data.get("token_dense_stats", {})
        return data.get("datasets", {}), mask_stats

    case = SparseCase(
        method.label,
        method.base_method,
        "dense" if method.label == "dense" else method.policy,
        keep_n=method.keep_n,
    )
    stats_path = case_dir / "vllm_structured_24_stats.json"
    env = method_env(args, model_label=model_label, method=method, stats_path=stats_path)
    process = None
    metrics: dict[str, Any] = {}
    mask_stats: dict[str, Any] = {}
    token_dense_stats: dict[str, Any] = {}
    old_enforce_eager = getattr(args, "enforce_eager", False)
    if method.base_method == "token_dense" and args.token_dense_enforce_eager:
        args.enforce_eager = True
    try:
        process, port = start_vllm_server(
            args,
            base_model=model_id,
            speculator_model=speculator_model,
            case_dir=case_dir,
            env=env,
        )
        before = scrape_spec_metrics(port)
        for dataset_name, pack in datasets.items():
            if dataset_name in ACCURACY_DATASETS:
                try:
                    metrics[dataset_name] = run_accuracy_dataset(
                        args,
                        port=port,
                        model_id=model_id,
                        model_label=model_label,
                        case=case,
                        dataset_name=dataset_name,
                        pack=pack,
                        case_dir=case_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    metrics[dataset_name] = failed_metric(dataset_name, str(exc))
            elif dataset_name in GENERATED_PPL_DATASETS:
                try:
                    metrics[dataset_name] = run_generated_ppl_dataset(
                        args,
                        port=port,
                        model_id=model_id,
                        model_label=model_label,
                        case=case,
                        dataset_name=dataset_name,
                        pack=pack,
                        case_dir=case_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    metrics[dataset_name] = failed_metric(dataset_name, str(exc))
            else:
                continue
            write_json(case_dir / "metrics.partial.json", metrics)
        after = scrape_spec_metrics(port)
        accepted = metric_delta(before, after, "vllm:spec_decode_num_accepted_tokens")
        drafted = metric_delta(before, after, "vllm:spec_decode_num_draft_tokens")
        for metric in metrics.values():
            metric["spec_accepted_tokens_case"] = accepted
            metric["spec_draft_tokens_case"] = drafted
            metric["spec_acceptance_rate_case"] = (
                accepted / drafted if accepted is not None and drafted else None
            )
        write_json(
            case_dir / "spec_counters.json",
            {
                "accepted_tokens": accepted,
                "draft_tokens": drafted,
                "acceptance_rate": accepted / drafted if accepted is not None and drafted else None,
            },
        )
    finally:
        stop_process(process)
        args.enforce_eager = old_enforce_eager
        if args.server_shutdown_settle_s > 0:
            time.sleep(args.server_shutdown_settle_s)

    if stats_path.exists():
        mask_stats = load_json(stats_path)
    token_dense_stats = summarize_token_dense_stats(case_dir / "token_dense_stats.jsonl")
    if token_dense_stats:
        mask_stats["token_dense_stats"] = token_dense_stats
    write_json(
        case_dir / "metrics.json",
        {
            "status": "ok",
            "model_label": model_label,
            "model_id": model_id,
            "speculator_model": speculator_model,
            "method": method.label,
            "base_method": method.base_method,
            "policy": method.policy,
            "keep_n": method.keep_n,
            "mask_stats": mask_stats,
            "token_dense_stats": token_dense_stats,
            "datasets": metrics,
        },
    )
    return metrics, mask_stats


def derive_token_dense_t00(
    *,
    model_label: str,
    model_id: str,
    speculator_model: str,
    dense_metrics: dict[str, Any],
    case_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir.mkdir(parents=True, exist_ok=True)
    method_metrics = copy.deepcopy(dense_metrics)
    draft_tokens = None
    for metric in method_metrics.values():
        metric["metric_source"] = (
            str(metric.get("metric_source") or "") + "_derived_token_dense_t00"
        ).strip("_")
        draft_tokens = metric.get("spec_draft_tokens_case", draft_tokens)
    draft_tokens_int = int(draft_tokens or 0)
    token_dense_stats = {
        "steps": 0,
        "threshold": 0.0,
        "total_draft_tokens": draft_tokens_int,
        "dense_draft_tokens": draft_tokens_int,
        "sparse_draft_tokens": 0,
        "missing_score_tokens": 0,
        "dense_draft_fraction": 1.0 if draft_tokens_int else None,
        "sparse_draft_fraction": 0.0 if draft_tokens_int else None,
        "derived_from": "dense",
    }
    mask_stats = {
        "effective_sparse_fraction": 0.5,
        "token_dense_stats": token_dense_stats,
        "derived_from": "dense",
    }
    write_json(
        case_dir / "metrics.json",
        {
            "status": "ok",
            "model_label": model_label,
            "model_id": model_id,
            "speculator_model": speculator_model,
            "method": "token_dense_t00",
            "base_method": "token_dense",
            "policy": "all_sparse",
            "keep_n": 0,
            "derived_from": "dense",
            "mask_stats": mask_stats,
            "token_dense_stats": token_dense_stats,
            "datasets": method_metrics,
        },
    )
    return method_metrics, mask_stats


def make_rows(
    *,
    model_label: str,
    model_id: str,
    method: str,
    dense_metrics: dict[str, Any],
    method_metrics: dict[str, Any],
    mask_stats: dict[str, Any],
    activation_aware_drops: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name, sparse_metric in method_metrics.items():
        metric_type = str(sparse_metric.get("metric_type") or "")
        dense_value = metric_value(dense_metrics.get(dataset_name, {}))
        sparse_value = metric_value(sparse_metric)
        drop = (
            dense_value - sparse_value
            if metric_type == "accuracy"
            and dense_value is not None
            and sparse_value is not None
            else None
        )
        delta = None
        ratio = None
        if dense_value is not None and sparse_value is not None:
            delta = sparse_value - dense_value if metric_type == "generated_ppl" else dense_value - sparse_value
            ratio = sparse_value / dense_value if dense_value else None
        aa_drop = activation_aware_drops.get(dataset_name)
        recovery = aa_drop - drop if aa_drop is not None and drop is not None else None
        token_dense_stats = mask_stats.get("token_dense_stats", {})
        dense_metric = dense_metrics.get(dataset_name, {})
        dense_nll = safe_float(dense_metric.get("generated_nll"))
        sparse_nll = safe_float(sparse_metric.get("generated_nll"))
        rows.append(
            {
                "model_label": model_label,
                "model_id": model_id,
                "method": method,
                "dataset": dataset_name,
                "metric_name": sparse_metric.get("metric_name", ""),
                "metric_type": metric_type,
                "dense_metric_value": dense_value,
                "sparse_metric_value": sparse_value,
                "delta_vs_dense": delta,
                "ratio_vs_dense": ratio,
                "dense_accuracy": dense_value if metric_type == "accuracy" else None,
                "sparse_accuracy": sparse_value if metric_type == "accuracy" else None,
                "accuracy_drop": drop,
                "recovery_vs_activation_aware": recovery,
                "dense_generated_ppl": dense_value if metric_type == "generated_ppl" else None,
                "sparse_generated_ppl": sparse_value if metric_type == "generated_ppl" else None,
                "generated_ppl_ratio_vs_dense": ratio if metric_type == "generated_ppl" else None,
                "dense_generated_nll": dense_nll,
                "sparse_generated_nll": sparse_nll,
                "generated_nll_delta_vs_dense": (
                    sparse_nll - dense_nll
                    if sparse_nll is not None and dense_nll is not None
                    else None
                ),
                "num_examples": sparse_metric.get("num_examples", dense_metrics.get(dataset_name, {}).get("num_examples", 0)),
                "output_tokens": sparse_metric.get("output_tokens"),
                "generated_logprob_tokens": sparse_metric.get("generated_logprob_tokens"),
                "generated_logprob_missing": sparse_metric.get("generated_logprob_missing"),
                "request_error_count": sparse_metric.get("request_error_count"),
                "spec_acceptance_rate_case": sparse_metric.get("spec_acceptance_rate_case"),
                "spec_accepted_tokens_case": sparse_metric.get("spec_accepted_tokens_case"),
                "spec_draft_tokens_case": sparse_metric.get("spec_draft_tokens_case"),
                "effective_sparse_fraction": mask_stats.get("effective_sparse_fraction"),
                "mask_cache": mask_stats.get("mask_cache", ""),
                "mask_cache_method": mask_stats.get("mask_cache_method", ""),
                "missing_cached_mask_modules": len(mask_stats.get("missing_cached_mask_modules", [])),
                "token_dense_threshold": token_dense_stats.get("threshold"),
                "token_dense_dense_draft_fraction": token_dense_stats.get("dense_draft_fraction"),
                "token_dense_sparse_draft_fraction": token_dense_stats.get("sparse_draft_fraction"),
                "token_dense_missing_score_tokens": token_dense_stats.get("missing_score_tokens"),
                "failed": bool(sparse_metric.get("failed", False)),
                "error": sparse_metric.get("error", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def threshold_methods(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["method"]) for row in rows}
    methods = [f"token_dense_t{value:02d}" for value in range(11)]
    return [method for method in methods if method in present]


def lookup_row(
    rows: list[dict[str, Any]],
    *,
    model_label: str,
    dataset: str,
    method: str,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["model_label"] == model_label
            and row["dataset"] == dataset
            and row["method"] == method
        ),
        None,
    )


def plot_metric_grid(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    metric_type: str,
    value_field: str,
    dense_field: str,
    reference_field: str | None,
    baseline_value: float | None = None,
    ylabel: str,
    filename: str,
) -> Path | None:
    valid = [row for row in rows if not row.get("failed") and row.get("metric_type") == metric_type]
    methods = threshold_methods(valid)
    if not valid or not methods:
        return None
    import matplotlib.pyplot as plt

    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model_label"]) for row in valid})
    datasets = sorted({str(row["dataset"]) for row in valid})
    fig, axes = plt.subplots(
        len(models),
        len(datasets),
        squeeze=False,
        figsize=(4.6 * len(datasets), 3.6 * len(models)),
        constrained_layout=True,
    )
    for row_idx, model_label in enumerate(models):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            values = []
            dense_value = baseline_value
            for method in methods:
                match = lookup_row(
                    valid,
                    model_label=model_label,
                    dataset=dataset,
                    method=method,
                )
                values.append(safe_float(match.get(value_field)) if match else None)
                if match and dense_value is None:
                    dense_value = safe_float(match.get(dense_field))
            x = list(range(len(methods)))
            ax.plot(
                x,
                [value if value is not None else math.nan for value in values],
                marker="o",
                linewidth=1.5,
                label="token_dense",
            )
            if dense_value is not None:
                ax.axhline(
                    dense_value,
                    color="black",
                    linewidth=1.0,
                    linestyle="--",
                    label="dense",
                )
            if reference_field:
                aa = lookup_row(
                    valid,
                    model_label=model_label,
                    dataset=dataset,
                    method="activation_aware",
                )
                aa_value = safe_float(aa.get(reference_field)) if aa else None
                if aa_value is not None:
                    ax.axhline(
                        aa_value,
                        color="tab:red",
                        linewidth=1.0,
                        linestyle=":",
                        label="activation_aware",
                    )
            ax.set_title(f"{model_label} / {dataset}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, [method.replace("token_dense_", "") for method in methods], rotation=0)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(loc="best", fontsize=8)
    path = fig_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_routing(rows: list[dict[str, Any]], output_root: Path) -> Path | None:
    valid = [row for row in rows if not row.get("failed")]
    methods = threshold_methods(valid)
    if not valid or not methods:
        return None
    import matplotlib.pyplot as plt

    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model_label"]) for row in valid})
    fig, axes = plt.subplots(
        1,
        len(models),
        squeeze=False,
        figsize=(4.8 * len(models), 3.6),
        constrained_layout=True,
    )
    for col_idx, model_label in enumerate(models):
        ax = axes[0][col_idx]
        values = []
        for method in methods:
            matches = [
                row
                for row in valid
                if row["model_label"] == model_label and row["method"] == method
            ]
            dense_fracs = [
                safe_float(row.get("token_dense_dense_draft_fraction"))
                for row in matches
            ]
            dense_fracs = [value for value in dense_fracs if value is not None]
            values.append(sum(dense_fracs) / len(dense_fracs) if dense_fracs else None)
        x = list(range(len(methods)))
        ax.plot(
            x,
            [value if value is not None else math.nan for value in values],
            marker="o",
        )
        ax.set_title(model_label)
        ax.set_ylabel("Dense draft-token fraction")
        ax.set_xticks(x, [method.replace("token_dense_", "") for method in methods])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
    path = fig_dir / "token_dense_routing_fraction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_results(rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    valid = [row for row in rows if not row.get("failed")]
    if not valid:
        return []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for path in [
        plot_metric_grid(
            rows=valid,
            output_root=output_root,
            metric_type="accuracy",
            value_field="sparse_accuracy",
            dense_field="dense_accuracy",
            reference_field="sparse_accuracy",
            ylabel="Absolute accuracy",
            filename="token_dense_absolute_accuracy.png",
        ),
        plot_metric_grid(
            rows=valid,
            output_root=output_root,
            metric_type="accuracy",
            value_field="accuracy_drop",
            dense_field="accuracy_drop",
            reference_field="accuracy_drop",
            baseline_value=0.0,
            ylabel="Accuracy drop vs dense",
            filename="token_dense_accuracy_drop.png",
        ),
        plot_metric_grid(
            rows=valid,
            output_root=output_root,
            metric_type="generated_ppl",
            value_field="sparse_generated_ppl",
            dense_field="dense_generated_ppl",
            reference_field="sparse_generated_ppl",
            ylabel="Generated PPL",
            filename="token_dense_generated_ppl.png",
        ),
        plot_metric_grid(
            rows=valid,
            output_root=output_root,
            metric_type="generated_ppl",
            value_field="generated_ppl_ratio_vs_dense",
            dense_field="ratio_vs_dense",
            reference_field="generated_ppl_ratio_vs_dense",
            baseline_value=1.0,
            ylabel="Generated PPL ratio vs dense",
            filename="token_dense_generated_ppl_ratio.png",
        ),
        plot_routing(valid, output_root),
    ]:
        if path is not None:
            paths.append(path)
    return paths


def write_summary(output_root: Path, rows: list[dict[str, Any]], figures: list[Path], args: argparse.Namespace) -> None:
    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Token-Dense Threshold Quality Sweep\n\n")
        handle.write(f"Output root: `{output_root.resolve()}`\n\n")
        handle.write("## Method\n\n")
        handle.write(
            "- Serving uses vLLM + EAGLE3 speculative decoding with K="
            f"{args.num_spec_tokens}.\n"
            "- Only the TLM/base large model is masked; the EAGLE3 drafter remains dense.\n"
            "- `activation_aware` is the existing C4 activation-RMS 2:4 baseline.\n"
            "- `wandapp_rgs` uses C4 inputs plus a local output-gradient sensitivity multiplier when choosing 2:4 masks.\n"
            "- `wandapp_ro` uses the same mask and adds an output-row least-squares scale. It is an RO-lite cache, not a saved sparse checkpoint.\n"
            "- `token_dense_tXX` keeps high-confidence draft-token rows dense when `draft_selected_prob` reaches the label threshold, e.g. `token_dense_t07` means 0.7; remaining draft-token rows use activation-aware 2:4, while prefill, non-draft, missing-score, and verifier bonus rows remain dense.\n"
            "- Method labels can include `_keep_first_N` to keep the first N transformer layers dense while masking the rest.\n"
            "- Accuracy is measured only on GSM8K and math_reasoning.\n"
            "- PPL is generated-token PPL from serving logprobs on MTBench, HumanEval, and selected Dolly categories; it is not teacher-forced reference PPL.\n"
            f"- Accuracy max tokens: `{args.accuracy_max_tokens}`; PPL max tokens: `{args.ppl_max_tokens}`.\n\n"
        )
        handle.write("## Inputs\n\n")
        handle.write(f"- models: `{args.models}`\n")
        handle.write(f"- datasets: `{args.datasets}`\n")
        handle.write(f"- cache root: `{args.cache_root.resolve()}`\n")
        handle.write(f"- calibration RMS root: `{args.calibration_cache_root.resolve()}`\n\n")
        if figures:
            handle.write("## Figures\n\n")
            for figure in figures:
                handle.write(f"- `{figure.resolve()}`\n")
            handle.write("\n")
        handle.write("## Quality Summary\n\n")
        handle.write("| model | dataset | method | metric | dense | sparse | delta | ratio | examples |\n")
        handle.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda item: (str(item["model_label"]), str(item["dataset"]), str(item["method"]))):
            handle.write(
                f"| {row['model_label']} | {row['dataset']} | {row['method']} | "
                f"{row.get('metric_type', '')} | "
                f"{safe_float(row.get('dense_metric_value')) or 0.0:.4f} | "
                f"{safe_float(row.get('sparse_metric_value')) or 0.0:.4f} | "
                f"{safe_float(row.get('delta_vs_dense')) or 0.0:.4f} | "
                f"{safe_float(row.get('ratio_vs_dense')) or 0.0:.4f} | "
                f"{int(row.get('num_examples') or 0)} |\n"
            )
        handle.write("\n## Files\n\n")
        handle.write("- `token_dense_threshold_quality.csv`: unified accuracy and generated-PPL comparison.\n")
        handle.write("- `token_dense_accuracy.csv`: accuracy-only comparison.\n")
        handle.write("- `token_dense_generated_ppl.csv`: generated-PPL-only comparison.\n")
        handle.write("- `wandapp_accuracy.csv`: compatibility copy of the unified table.\n")
        handle.write("- `runs/*/*/vllm_structured_24_stats.json`: vLLM-side mask proof and cache metadata.\n")


def configure_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.output_root = args.output_root or (RESULTS_BAK_ROOT / f"wandapp_accuracy_smoke_{timestamp()}")
    args.models = "qwen3_8b"
    args.methods = "token_dense_t00,token_dense_t10"
    args.datasets = "gsm8k,math_reasoning,mtbench,humaneval,dolly_open_qa,dolly_creative_writing,dolly_summarization"
    args.gsm8k_num_examples = 2
    args.math_num_examples = 2
    args.mtbench_num_examples = 2
    args.humaneval_num_examples = 2
    args.dolly_num_examples = 2
    args.accuracy_max_tokens = 32
    args.ppl_max_tokens = 32
    args.accuracy_concurrency = 2
    args.ppl_concurrency = 2
    args.max_num_seqs = 2


def run(args: argparse.Namespace) -> None:
    configure_local_no_proxy()
    configure_smoke(args)
    ensure_quality_dependencies()
    set_seed(args.seed)
    output_root = args.output_root or (RESULTS_ROOT / f"token_dense_threshold_quality_{timestamp()}")
    args.output_root = output_root
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = load_selected_datasets(args)
    datasets = {
        name: pack
        for name, pack in datasets.items()
        if name in ACCURACY_DATASETS or name in GENERATED_PPL_DATASETS
    }
    if not datasets:
        raise RuntimeError("no supported datasets selected")
    base_models = dict(DEFAULT_BASE_MODELS)
    base_models.update(LAYER_SENSITIVITY_DEFAULT_MODELS)
    base_models.update(parse_model_id_overrides(args.model_id))
    speculators = dict(EAGLE3_SPECULATORS)
    speculators.update(parse_model_id_overrides(args.speculator_model))
    selected_models = parse_csv_list(args.models)
    methods = parse_csv_list(args.methods)
    if "activation_aware" not in methods:
        methods = ["activation_aware", *methods]
    method_configs = [parse_method_config(method) for method in methods]
    dense_config = MethodConfig(label="dense", base_method="dense", policy="dense")

    write_json(
        output_root / "run_config.json",
        {
            "argv": sys.argv,
            "models": selected_models,
            "methods": methods,
            "datasets": list(datasets),
            "accuracy_datasets": [
                name for name in datasets if name in ACCURACY_DATASETS
            ],
            "generated_ppl_datasets": [
                name for name in datasets if name in GENERATED_PPL_DATASETS
            ],
            "base_models": base_models,
            "speculators": speculators,
            "cache_root": str(args.cache_root.resolve()),
            "calibration_cache_root": str(args.calibration_cache_root.resolve()),
            "num_spec_tokens": args.num_spec_tokens,
            "created_at": timestamp(),
            "smoke": args.smoke,
        },
    )

    rows: list[dict[str, Any]] = []
    for model_label in selected_models:
        model_id = base_models.get(model_label)
        speculator_model = speculators.get(model_label)
        if not model_id:
            raise ValueError(f"unknown model label: {model_label}")
        if not speculator_model:
            raise ValueError(f"missing EAGLE3 speculator for {model_label}")

        dense_metrics, _dense_stats = run_method(
            args,
            model_label=model_label,
            model_id=model_id,
            speculator_model=speculator_model,
            method=dense_config,
            datasets=datasets,
            case_dir=output_root / "runs" / model_label / "dense",
        )

        activation_aware_drops: dict[str, float] = {}
        for method_config in method_configs:
            case_dir = output_root / "runs" / model_label / method_config.label
            if (
                args.derive_t00_from_dense
                and method_config.base_method == "token_dense"
                and method_config.token_dense_threshold == 0.0
            ):
                if args.resume and metrics_complete(case_dir, set(datasets)):
                    data = load_json(case_dir / "metrics.json")
                    method_metrics = data.get("datasets", {})
                    mask_stats = data.get("mask_stats", {})
                    if isinstance(mask_stats, dict):
                        mask_stats["token_dense_stats"] = data.get("token_dense_stats", {})
                else:
                    method_metrics, mask_stats = derive_token_dense_t00(
                        model_label=model_label,
                        model_id=model_id,
                        speculator_model=speculator_model,
                        dense_metrics=dense_metrics,
                        case_dir=case_dir,
                    )
            else:
                method_metrics, mask_stats = run_method(
                    args,
                    model_label=model_label,
                    model_id=model_id,
                    speculator_model=speculator_model,
                    method=method_config,
                    datasets=datasets,
                    case_dir=case_dir,
                )
            method_rows = make_rows(
                model_label=model_label,
                model_id=model_id,
                method=method_config.label,
                dense_metrics=dense_metrics,
                method_metrics=method_metrics,
                mask_stats=mask_stats,
                activation_aware_drops=activation_aware_drops,
            )
            rows.extend(method_rows)
            if method_config.label == "activation_aware":
                activation_aware_drops = {
                    str(row["dataset"]): float(row["accuracy_drop"])
                    for row in method_rows
                    if row.get("accuracy_drop") is not None
                }
            write_outputs(output_root, rows, args)

    write_outputs(output_root, rows, args)
    print(output_root.resolve())


def write_outputs(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    write_csv(output_root / "token_dense_threshold_quality.csv", rows)
    write_csv(
        output_root / "token_dense_accuracy.csv",
        [row for row in rows if row.get("metric_type") == "accuracy"],
    )
    write_csv(
        output_root / "token_dense_generated_ppl.csv",
        [row for row in rows if row.get("metric_type") == "generated_ppl"],
    )
    write_csv(output_root / "wandapp_accuracy.csv", rows)
    figures = plot_results(rows, output_root)
    write_summary(output_root, rows, figures, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token-dense 2:4 threshold quality benchmark under EAGLE3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--model-id", action="append", default=[], help="Override base model as LABEL=PATH_OR_ID.")
    parser.add_argument("--speculator-model", action="append", default=[], help="Override EAGLE3 model as LABEL=PATH_OR_ID.")
    parser.add_argument("--methods", default=DEFAULT_THRESHOLD_METHODS)
    parser.add_argument(
        "--datasets",
        default="gsm8k,math_reasoning,mtbench,humaneval,dolly_open_qa,dolly_creative_writing,dolly_summarization",
    )
    parser.add_argument("--gsm8k-num-examples", type=int, default=0)
    parser.add_argument("--math-num-examples", type=int, default=80)
    parser.add_argument("--mtbench-num-examples", type=int, default=0)
    parser.add_argument("--dolly-num-examples", type=int, default=0)
    parser.add_argument("--humaneval-num-examples", type=int, default=None)
    parser.add_argument("--accuracy-max-tokens", type=int, default=512)
    parser.add_argument("--accuracy-concurrency", type=int, default=8)
    parser.add_argument("--ppl-max-tokens", type=int, default=512)
    parser.add_argument("--ppl-concurrency", type=int, default=8)
    parser.add_argument(
        "--context-length-safety-margin",
        type=int,
        default=8,
        help="Reserve this many tokens when clipping per-request generated-PPL max_tokens.",
    )
    parser.add_argument("--completion-logprobs", type=int, default=1)
    parser.add_argument("--derive-t00-from-dense", action="store_true", default=True)
    parser.add_argument("--no-derive-t00-from-dense", dest="derive_t00_from_dense", action="store_false")
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--port-base", type=int, default=8170)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=5.0)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--calibration-cache-root", type=Path, default=DEFAULT_C4_CALIBRATION_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--token-dense-enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
