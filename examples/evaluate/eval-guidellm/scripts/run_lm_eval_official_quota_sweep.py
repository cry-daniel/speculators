#!/usr/bin/env python3
"""Run official lm-eval 0.4.12 for dense EAGLE3 and SpecLink D0/8--D8/8.

Each model/quota pair starts one optimized vLLM EAGLE3 K=7 server and reuses it
for all four tasks.  This avoids reloading and repacking model weights for every
benchmark while preserving task-level timings and acceptance counters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
MODELS_ROOT = SPECULATORS_ROOT.parent / "models"
DATASET_ROOT = EVAL_ROOT / "data" / "lm_eval_official_256_seed42"
C4_CALIBRATION_PROMPTS = (
    EVAL_ROOT
    / "data"
    / "c4_calibration"
    / "c4_calibration_512_seed42.jsonl"
)
DEFAULT_CALIBRATION_ROOT = (
    EVAL_ROOT
    / "data"
    / "c4_calibration"
    / "activation_rms"
    / "c4_512_seed42_bf16_max512"
)

sys.path.insert(0, str(SCRIPT_DIR))
from run_llama_gsm8k_score_mode_comparison import warmup_full_batch  # noqa: E402
from run_structured_24_spec_quality import (  # noqa: E402
    add_local_no_proxy,
    start_vllm_server,
    stop_process,
)


MODEL_CONFIGS = {
    "qwen3_8b": {
        "display": "Qwen3-8B",
        "model": MODELS_ROOT / "qwen3-8b",
        "speculator": MODELS_ROOT / "qwen3-8b-eagle3-speculator",
    },
    "llama3_1_8b": {
        "display": "Llama-3.1-8B",
        "model": MODELS_ROOT / "llama-3.1-8b-instruct",
        "speculator": MODELS_ROOT / "llama-3.1-8b-eagle3-speculator",
    },
}
TASK_CONFIGS = {
    "gsm8k": {
        "display": "GSM8K CoT",
        "task": "gsm8k_cot",
        "manifest_key": "gsm8k_cot",
        "samples_file": "gsm8k_cot_samples.json",
        "metric_priority": (
            "exact_match,flexible-extract",
            "exact_match,strict-match",
        ),
    },
    "minerva": {
        "display": "Minerva Math",
        "task": "minerva_math",
        "manifest_key": "minerva_math",
        "samples_file": "minerva_math_samples.json",
        "metric_priority": ("math_verify,none", "exact_match,none"),
    },
    "bbh": {
        "display": "BBH CoT few-shot",
        "task": "bbh_cot_fewshot",
        "manifest_key": "bbh_cot_fewshot",
        "samples_file": "bbh_cot_fewshot_samples.json",
        "metric_priority": ("exact_match,get-answer",),
    },
    "mmlu_pro": {
        "display": "MMLU-Pro",
        "task": "mmlu_pro",
        "manifest_key": "mmlu_pro",
        "samples_file": "mmlu_pro_samples.json",
        "metric_priority": ("exact_match,custom-extract",),
    },
}
MAX_GEN_TOKS = 256
DENSE_METHOD = "dense_eagle3"
COUNTER_NAMES = (
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_eighths(value: str) -> list[int]:
    result: list[int] = []
    for item in csv_list(value):
        eighths = int(item)
        if not 0 <= eighths <= 8:
            raise ValueError(f"dense eighths must be in [0,8], got {eighths}")
        if eighths not in result:
            result.append(eighths)
    return result


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(SPECULATORS_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def selected_samples(task_key: str, limit: int = 0) -> dict[str, list[int]]:
    config = TASK_CONFIGS[task_key]
    source = DATASET_ROOT / str(config["samples_file"])
    full = json.loads(source.read_text(encoding="utf-8"))
    normalized = {
        str(task): [int(index) for index in indices]
        for task, indices in full.items()
    }
    if limit <= 0:
        return normalized

    # lm-eval treats an omitted/empty group leaf as "evaluate all", so smoke
    # runs must retain at least one frozen example for every leaf.
    limit = max(limit, len(normalized))
    output = {task: [] for task in normalized}
    remaining = limit
    depth = 0
    while remaining:
        progressed = False
        for task in normalized:
            indices = normalized[task]
            if depth >= len(indices):
                continue
            output[task].append(indices[depth])
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
        depth += 1
    if sum(map(len, output.values())) != limit:
        raise ValueError(f"{task_key} cannot provide {limit} fixed samples")
    return output


def logged_sample_count(run_dir: Path, result_path: Path | None) -> int:
    if result_path is None:
        return 0
    timestamp_suffix = result_path.stem.removeprefix("results_")
    total = 0
    for path in (run_dir / "lm_eval_output").rglob(
        f"samples_*_{timestamp_suffix}.jsonl"
    ):
        # One question may be logged once per filter (GSM8K has strict and
        # flexible extraction). Count unique document ids, not JSONL rows.
        doc_ids: set[int] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_ids.add(int(record["doc_id"]))
        total += len(doc_ids)
    return total


def use_qwen_sampling(
    model_label: str, *, llama_qwen_sampling: bool
) -> bool:
    return model_label == "qwen3_8b" or (
        model_label == "llama3_1_8b" and llama_qwen_sampling
    )


def chat_template_args(
    model_label: str, *, llama_qwen_sampling: bool
) -> dict[str, Any]:
    if use_qwen_sampling(
        model_label, llama_qwen_sampling=llama_qwen_sampling
    ):
        return {"enable_thinking": False}
    return {}


def generation_protocol(
    model_label: str, *, llama_qwen_sampling: bool
) -> dict[str, Any]:
    if use_qwen_sampling(
        model_label, llama_qwen_sampling=llama_qwen_sampling
    ):
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_gen_toks": MAX_GEN_TOKS,
            "thinking": False,
        }
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "max_gen_toks": MAX_GEN_TOKS,
        "thinking": None,
    }


def quota_label(eighths: int) -> str:
    return f"d{eighths}"


def method_label(eighths: int | None) -> str:
    return DENSE_METHOD if eighths is None else quota_label(eighths)


def quota_display(eighths: int | None) -> str:
    if eighths is None:
        return "Dense EAGLE3"
    return f"D{eighths}/8"


def method_order(eighths: int | None) -> int:
    return 9 if eighths is None else eighths


def graph_rows(args: argparse.Namespace) -> list[int]:
    width = args.num_spec_tokens + 1
    return list(range(width, width * args.max_num_seqs + 1, width))


def prepare_eval_tokenizer(
    *,
    model: Path,
    model_label: str,
    output_root: Path,
    llama_qwen_sampling: bool,
) -> Path:
    """Audit and return the checkpoint's unmodified official tokenizer."""

    config_path = model / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("chat_template"):
        raise ValueError(f"{config_path} has no official chat_template")
    wrapper = SCRIPT_DIR / "lm_eval_official_chat_template.py"
    write_json(
        output_root / "tokenizers" / model_label / "chat_template_audit.json",
        {
            "tokenizer": str(model.resolve()),
            "tokenizer_config": str(config_path.resolve()),
            "tokenizer_config_sha256": sha256_file(config_path),
            "chat_template_sha256": hashlib.sha256(
                str(config["chat_template"]).encode("utf-8")
            ).hexdigest(),
            "model_label": model_label,
            "apply_chat_template": True,
            "chat_template_source": (
                "checkpoint tokenizer_config.json -> chat_template"
            ),
            "chat_template_modified": False,
            "chat_template_args": chat_template_args(
                model_label,
                llama_qwen_sampling=llama_qwen_sampling,
            ),
            "lm_eval_entrypoint": str(wrapper.resolve()),
            "lm_eval_entrypoint_sha256": sha256_file(wrapper),
        },
    )
    return model.resolve()


def checkpoint_audit(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_files = sorted(path.glob("*.safetensors"))
    weights = [
        {"name": item.name, "bytes": item.stat().st_size}
        for item in weight_files
    ]
    manifest_payload = json.dumps(
        weights, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    index_path = path / "model.safetensors.index.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    return {
        "path": str(path.resolve()),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "torch_dtype": config.get("torch_dtype"),
        "config_sha256": sha256_file(config_path),
        "weight_files": weights,
        "weight_bytes": sum(int(item["bytes"]) for item in weights),
        "weight_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "weight_index_sha256": (
            sha256_file(index_path) if index_path.exists() else None
        ),
        "tokenizer_config_sha256": (
            sha256_file(tokenizer_config_path)
            if tokenizer_config_path.exists()
            else None
        ),
        "chat_template_source": (
            "checkpoint tokenizer_config.json -> chat_template"
            if tokenizer_config_path.exists()
            else None
        ),
        "identity_rule": (
            "config SHA256 + safetensors-index SHA256 + "
            "ordered weight filename/size manifest"
        ),
    }


def global_audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = DATASET_ROOT / "manifest.json"
    selection_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=str(SPECULATORS_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    models: dict[str, Any] = {}
    for model_label in args.models_list:
        config = MODEL_CONFIGS[model_label]
        models[model_label] = {
            "base": checkpoint_audit(Path(config["model"])),
            "speculator": checkpoint_audit(Path(config["speculator"])),
        }
    calibration_models = {
        model_label: {
            "activation_rms": str(
                (args.calibration_cache_root / f"{model_label}.pt").resolve()
            ),
            "activation_rms_sha256": sha256_file(
                args.calibration_cache_root / f"{model_label}.pt"
            ),
            "metadata": str(
                (
                    args.calibration_cache_root
                    / f"{model_label}_metadata.json"
                ).resolve()
            ),
            "metadata_sha256": sha256_file(
                args.calibration_cache_root
                / f"{model_label}_metadata.json"
            ),
        }
        for model_label in args.models_list
    }
    source_files = [
        SCRIPT,
        SCRIPT_DIR / "lm_eval_official_chat_template.py",
        SCRIPT_DIR / "prepare_lm_eval_official_256.py",
        SPECULATORS_ROOT / "vllm" / "vllm" / "speclink_token_dense.py",
        SPECULATORS_ROOT / "vllm" / "vllm" / "speclink_structured_24.py",
    ]
    return {
        "audit_schema": 1,
        "created_at": timestamp(),
        "protocol": {
            "lm_eval_version": importlib.metadata.version("lm-eval"),
            "seed": args.seed,
            "max_gen_toks": MAX_GEN_TOKS,
            "selection_manifest": str(manifest_path.resolve()),
            "selection_manifest_sha256": sha256_file(manifest_path),
            "selection": selection_manifest,
            "generation": {
                model_label: generation_protocol(
                    model_label,
                    llama_qwen_sampling=args.llama_qwen_sampling,
                )
                for model_label in args.models_list
            },
        },
        "source": {
            "repository": git_output("remote", "get-url", "origin"),
            "upstream_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "dirty_files": git_output("status", "--short").splitlines(),
            "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_file_sha256": {
                str(path.relative_to(SPECULATORS_ROOT)): sha256_file(path)
                for path in source_files
            },
        },
        "software": {
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "vllm": importlib.metadata.version("vllm"),
            "lm_eval": importlib.metadata.version("lm-eval"),
        },
        "models": models,
        "calibration": {
            "dataset": str(C4_CALIBRATION_PROMPTS.resolve()),
            "dataset_sha256": sha256_file(C4_CALIBRATION_PROMPTS),
            "dataset_rows": line_count(C4_CALIBRATION_PROMPTS),
            "cache_root": str(args.calibration_cache_root.resolve()),
            "models": calibration_models,
        },
    }


def case_audit(
    *,
    model_label: str,
    dense_eighths: int | None,
    case_dir: Path,
    global_record: dict[str, Any],
    dense_leafs: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    stats_path = case_dir / "vllm_structured_24_stats.json"
    runtime: dict[str, Any]
    sparsity: dict[str, Any]
    calibration_applied = dense_eighths is not None
    if dense_eighths is None:
        if stats_path.exists():
            errors.append("dense baseline unexpectedly emitted sparse stats")
        runtime = {
            "verified": not stats_path.exists(),
            "storage": "native_dense_checkpoint",
            "module_count": 0,
            "persistent_bytes": 0,
            "released_dense_bytes": 0,
        }
        sparsity = {
            "n": None,
            "m": None,
            "actual_weight_sparsity": 0.0,
            "verifier_dense_eighths": 8,
        }
    else:
        if not stats_path.exists():
            errors.append("structured 2:4 runtime stats file is missing")
            stats: dict[str, Any] = {}
        else:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        per_module = stats.get("per_module") or []
        masked_modules = [
            item for item in per_module if not item.get("kept_dense")
        ]
        dense_modules = [
            item for item in per_module if item.get("kept_dense")
        ]
        runtimes = [
            item.get("residual_complement_runtime") or {}
            for item in masked_modules
        ]
        expected_storage = "cusparselt_base_plus_complement_no_duplicate_metadata"
        scope_total = int(stats.get("scope_target_weight_count") or 0)
        zeroed_total = int(stats.get("zeroed_weight_count") or 0)
        effective_expected = zeroed_total / scope_total if scope_total else 0.0
        checks = {
            "enabled": stats.get("enabled") is True,
            "token_dense_enabled": stats.get("token_dense_enabled") is True,
            "model_label": stats.get("model_label") == model_label,
            "policy": stats.get("policy") == "all_sparse",
            "actual_sparsity": stats.get("actual_sparsity") == 0.5,
            "effective_sparse_fraction_consistent": math.isclose(
                float(stats.get("effective_sparse_fraction") or 0.0),
                effective_expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "requested_dense_leafs": (
                sorted(stats.get("dense_leafs") or [])
                == sorted(dense_leafs)
            ),
            "modules_present": bool(per_module),
            "module_count_matches": (
                int(stats.get("module_count_seen") or 0) == len(per_module)
            ),
            "no_missing_activation_scale": not stats.get(
                "missing_activation_scale_modules"
            ),
            "no_missing_cached_mask": not stats.get(
                "missing_cached_mask_modules"
            ),
            "all_masked_activation_aware": all(
                item.get("mask_method") == "token_dense_activation_aware"
                for item in masked_modules
            ),
            "all_dense_leaf_modules_kept": all(
                item.get("leaf") in dense_leafs
                and item.get("mask_method") == "dense_keep"
                and item.get("dense_keep_reason") == "dense_leaf"
                for item in dense_modules
            ),
            "no_unexpected_dense_modules": all(
                item.get("leaf") in dense_leafs for item in dense_modules
            ),
            "all_requested_dense_leafs_present": all(
                any(item.get("leaf") == leaf for item in dense_modules)
                for leaf in dense_leafs
            ),
            "all_runtime_storage": all(
                item.get("storage") == expected_storage for item in runtimes
            ),
            "all_runtime_shapes_match": all(
                item.get("shape") == runtime_item.get("shape")
                for item, runtime_item in zip(
                    masked_modules, runtimes, strict=True
                )
            ),
            "all_runtime_values_present": all(
                int(item.get("persistent_bytes") or 0) > 0
                and int(item.get("released_dense_bytes") or 0) > 0
                for item in runtimes
            ),
            "exact_two_of_four": (
                int(stats.get("zeroed_weight_count") or 0) * 2
                == int(stats.get("total_masked_weight_count") or -1)
            ),
        }
        errors.extend(name for name, passed in checks.items() if not passed)
        storages = sorted(
            {str(item.get("storage")) for item in runtimes if item}
        )
        runtime = {
            "verified": all(checks.values()),
            "storage": storages,
            "module_count": len(per_module),
            "masked_module_count": len(masked_modules),
            "dense_module_count": len(dense_modules),
            "dense_leafs": sorted(dense_leafs),
            "persistent_bytes": int(
                stats.get("residual_complement_persistent_bytes") or 0
            ),
            "released_dense_bytes": int(
                stats.get("released_dense_weight_bytes") or 0
            ),
            "packed_base_values_and_metadata": True,
            "complement_values": True,
            "duplicate_sparse_metadata": False,
            "checks": checks,
            "stats_path": str(stats_path.resolve()),
            "stats_sha256": (
                sha256_file(stats_path) if stats_path.exists() else None
            ),
        }
        sparsity = {
            "n": 2,
            "m": 4,
            "actual_weight_sparsity": stats.get("actual_sparsity"),
            "effective_sparse_fraction": stats.get(
                "effective_sparse_fraction"
            ),
            "verifier_dense_eighths": dense_eighths,
            "always_dense_leafs": sorted(dense_leafs),
        }
    record = {
        "audit_schema": 1,
        "created_at": timestamp(),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "model_label": model_label,
        "model": global_record["models"][model_label],
        "calibration": {
            **global_record["calibration"],
            "applied_to_mask": calibration_applied,
        },
        "upstream_commit": global_record["source"]["upstream_commit"],
        "working_tree_diff_sha256": global_record["source"][
            "working_tree_diff_sha256"
        ],
        "method": method_label(dense_eighths),
        "sparsity": sparsity,
        "runtime_sparse_value_loading": runtime,
    }
    write_json(case_dir / "case_audit.json", record)
    return record


def make_server_args(args: argparse.Namespace) -> argparse.Namespace:
    rows = graph_rows(args)
    return argparse.Namespace(
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_spec_tokens=args.num_spec_tokens,
        enforce_eager=False,
        enable_prefix_caching=False,
        compilation_config={
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": rows,
            "max_cudagraph_capture_size": rows[-1],
            "cudagraph_num_of_warmups": 1,
        },
        port_base=args.port_base,
        health_timeout_s=args.health_timeout_s,
    )


def make_env(
    args: argparse.Namespace,
    *,
    model_label: str,
    dense_eighths: int | None,
    case_dir: Path,
) -> dict[str, str]:
    rows = graph_rows(args)
    env = add_local_no_proxy(os.environ.copy())
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env.pop("VLLM_DISABLE_COMPILE_CACHE", None)
    env.pop("VLLM_CACHE_ROOT", None)
    for name in list(env):
        if name.startswith("SPECLINK_STRUCTURED_24_") or name.startswith(
            "SPECLINK_TOKEN_DENSE_"
        ):
            env.pop(name, None)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "SPECLINK_TRACE_CONFIDENCE": "0",
            "SPECLINK_STRUCTURED_24_ENABLE": "0",
            "SPECLINK_TOKEN_DENSE_ENABLE": "0",
        }
    )
    if dense_eighths is not None:
        env.update(
            {
                # The AOT graph closes over quota-specific dense/sparse index
                # workspace shapes, while vLLM's normal persistent cache key
                # does not include SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS.
                # Isolate caches by model/quota; disabling the cache entirely
                # takes a separate vLLM path that cannot compile inference
                # tensors with these persistent workspaces.
                "VLLM_CACHE_ROOT": str(
                    (
                        EVAL_ROOT
                        / "temp"
                        / "lm_eval_official_compile_cache"
                        / case_dir.parents[1].name
                        / model_label
                        / method_label(dense_eighths)
                    ).resolve()
                ),
                "SPECLINK_STRUCTURED_24_ENABLE": "1",
                "SPECLINK_STRUCTURED_24_MODEL_LABEL": model_label,
                "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(
                    args.calibration_cache_root.resolve()
                ),
                "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
                "SPECLINK_STRUCTURED_24_STATS_PATH": str(
                    (case_dir / "vllm_structured_24_stats.json").resolve()
                ),
                "SPECLINK_TOKEN_DENSE_ENABLE": "1",
                "SPECLINK_TOKEN_DENSE_MODE": "high_confidence_dense",
                "SPECLINK_TOKEN_DENSE_BACKEND": "residual_complement_splitk2",
                "SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS": str(dense_eighths),
                "SPECLINK_TOKEN_DENSE_ROUTING_SCOPE": "global",
                "SPECLINK_TOKEN_DENSE_SCORE_MODE": "prefix_product",
                "SPECLINK_TOKEN_DENSE_EXPECTED_ROWS": str(
                    args.max_num_seqs * (args.num_spec_tokens + 1)
                ),
                "SPECLINK_TOKEN_DENSE_STATS_INTERVAL": "0",
                "SPECLINK_TOKEN_DENSE_STATS_DETAIL": "0",
                "SPECLINK_TOKEN_DENSE_PREFILL_FUSED": "1",
                "SPECLINK_TOKEN_DENSE_GATEUP_FUSED": "1",
                "SPECLINK_TOKEN_DENSE_SPLITK2_VARIANT": "auto",
                "SPECLINK_TOKEN_DENSE_GRAPH_ROUTE": "1",
                "SPECLINK_TOKEN_DENSE_GRAPH_ROWS": ",".join(map(str, rows)),
            }
        )
        if args.dense_leafs_list:
            env["SPECLINK_STRUCTURED_24_DENSE_LEAFS"] = ",".join(
                args.dense_leafs_list
            )
    env.pop("SPECLINK_TOKEN_DENSE_STATS_PATH", None)
    return env


def scrape_counters(port: int) -> dict[str, float]:
    output = {name: 0.0 for name in COUNTER_NAMES}
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=10
        ) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return output
    for line in text.splitlines():
        for name in COUNTER_NAMES:
            match = re.match(
                rf"^{re.escape(name)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",
                line.strip(),
            )
            if match:
                output[name] += float(match.group(1))
                break
    return output


def counter_delta(
    before: dict[str, float],
    after: dict[str, float],
    name: str,
) -> float:
    return float(after.get(name, 0.0) - before.get(name, 0.0))


def model_args(
    *,
    port: int,
    model: Path,
    tokenizer: Path,
    args: argparse.Namespace,
    max_gen_toks: int,
) -> str:
    values = {
        "model": str(model.resolve()),
        "base_url": f"http://127.0.0.1:{port}/v1/completions",
        "tokenizer": str(tokenizer.resolve()),
        "tokenizer_backend": "huggingface",
        "max_length": args.max_model_len,
        "max_gen_toks": max_gen_toks,
        "num_concurrent": args.concurrency,
        "seed": args.seed,
        "timeout": args.request_timeout_s,
        "tokenized_requests": True,
        "trust_remote_code": False,
    }
    return ",".join(f"{key}={value}" for key, value in values.items())


def result_json(run_dir: Path) -> Path | None:
    candidates = [
        path
        for path in (run_dir / "lm_eval_output").rglob("*.json")
        if not path.name.startswith("samples_")
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def result_score(
    path: Path, task_key: str
) -> tuple[float | None, int | None, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = TASK_CONFIGS[task_key]
    task_name = str(config["task"])
    task_result = (
        (data.get("results") or {}).get(task_name)
        or (data.get("groups") or {}).get(task_name)
        or {}
    )
    metric = ""
    score: float | None = None
    for wanted in config["metric_priority"]:
        if wanted in task_result:
            metric = str(wanted)
            try:
                score = float(task_result[wanted])
            except (TypeError, ValueError):
                score = None
            break
    samples: int | None = None
    for section in ("n-samples", "n_samples"):
        raw = (data.get(section) or {}).get(task_name)
        if isinstance(raw, dict):
            raw = raw.get("effective") or raw.get("original")
        try:
            samples = int(raw)
        except (TypeError, ValueError):
            continue
        break
    return score, samples, metric


def run_task(
    args: argparse.Namespace,
    *,
    port: int,
    model_label: str,
    model: Path,
    tokenizer: Path,
    dense_eighths: int | None,
    task_key: str,
    run_dir: Path,
    audit_record: dict[str, Any],
) -> dict[str, Any]:
    config = TASK_CONFIGS[task_key]
    task_name = str(config["task"])
    max_gen_toks = MAX_GEN_TOKS
    sample_map = selected_samples(task_key, args.limit)
    expected_samples = sum(map(len, sample_map.values()))
    generation = generation_protocol(
        model_label,
        llama_qwen_sampling=args.llama_qwen_sampling,
    )
    gen_kwargs = [
        f"temperature={generation['temperature']}",
        f"top_p={generation['top_p']}",
        f"do_sample={generation['do_sample']}",
        f"max_gen_toks={MAX_GEN_TOKS}",
    ]
    if "top_k" in generation:
        gen_kwargs.extend(
            [
                f"top_k={generation['top_k']}",
                f"min_p={generation['min_p']}",
            ]
        )
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "spec",
        "python",
        str(SCRIPT_DIR / "lm_eval_official_chat_template.py"),
        "run",
        "--model",
        "local-completions",
        "--model_args",
        model_args(
            port=port,
            model=model,
            tokenizer=tokenizer,
            args=args,
            max_gen_toks=max_gen_toks,
        ),
        "--tasks",
        task_name,
        "--apply_chat_template",
        "--batch_size",
        str(args.batch_size),
        "--gen_kwargs",
        *gen_kwargs,
        "--seed",
        str(args.seed),
        "--samples",
        json.dumps(sample_map, separators=(",", ":")),
        "--output_path",
        str(run_dir / "lm_eval_output"),
        "--log_samples",
    ]
    write_json(
        run_dir / "lm_eval_command.json",
        {
            "command": command,
            "fixed_samples": sample_map,
            "expected_samples": expected_samples,
            "generation": generation,
        },
    )
    env = add_local_no_proxy(os.environ.copy())
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env.update(
        {
            "OPENAI_API_KEY": "EMPTY",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "SPECLINK_LMEVAL_CHAT_TEMPLATE_ARGS": json.dumps(
                chat_template_args(
                    model_label,
                    llama_qwen_sampling=args.llama_qwen_sampling,
                ),
                separators=(",", ":"),
            ),
        }
    )
    before = scrape_counters(port)
    started = time.perf_counter()
    with (run_dir / "lm_eval.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(EVAL_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    after = scrape_counters(port)
    result_path = result_json(run_dir)
    score = None
    samples = None
    metric = ""
    reported_samples = None
    if result_path is not None:
        score, reported_samples, metric = result_score(result_path, task_key)
    samples = logged_sample_count(run_dir, result_path)

    prompt_tokens = counter_delta(before, after, "vllm:prompt_tokens")
    generation_tokens = counter_delta(before, after, "vllm:generation_tokens")
    requests = counter_delta(before, after, "vllm:request_success")
    verification_steps = counter_delta(
        before, after, "vllm:spec_decode_num_drafts"
    )
    draft_tokens = counter_delta(
        before, after, "vllm:spec_decode_num_draft_tokens"
    )
    accepted_tokens = counter_delta(
        before, after, "vllm:spec_decode_num_accepted_tokens"
    )
    complete = (
        completed.returncode == 0
        and score is not None
        and samples == expected_samples
    )
    failure_reason = ""
    if completed.returncode:
        failure_reason = f"lm-eval returncode {completed.returncode}"
    elif score is None:
        failure_reason = "requested official aggregate metric is missing"
    elif samples != expected_samples:
        failure_reason = (
            f"logged sample count {samples} != expected {expected_samples}"
        )
    row = {
        "status": "ok" if complete else "failed",
        "failure_reason": failure_reason,
        "returncode": completed.returncode,
        "model_label": model_label,
        "model": MODEL_CONFIGS[model_label]["display"],
        "method": method_label(dense_eighths),
        "dense_eighths": dense_eighths,
        "quota": quota_display(dense_eighths),
        "mode": method_label(dense_eighths),
        "total_dense_fraction": (
            1.0 if dense_eighths is None else dense_eighths / 8.0
        ),
        "dense_draft_fraction": (
            1.0
            if dense_eighths is None
            else max(0.0, (dense_eighths - 1) / 7.0)
        ),
        "task_key": task_key,
        "task": config["display"],
        "lm_eval_task": task_name,
        "metric": metric,
        "accuracy": score,
        "samples": samples,
        "reported_group_samples": reported_samples,
        "expected_samples": expected_samples,
        "fixed_samples_sha256": sha256_file(
            DATASET_ROOT / str(config["samples_file"])
        ),
        "seed": args.seed,
        "max_gen_toks": MAX_GEN_TOKS,
        "generation_protocol": json.dumps(
            generation, sort_keys=True, separators=(",", ":")
        ),
        "chat_template_source": (
            "checkpoint tokenizer_config.json -> chat_template"
        ),
        "chat_template_modified": False,
        "chat_template_args": json.dumps(
            chat_template_args(
                model_label,
                llama_qwen_sampling=args.llama_qwen_sampling,
            ),
            separators=(",", ":"),
        ),
        "audit_pass": audit_record.get("status") == "passed",
        "audit_path": str((run_dir.parent / "case_audit.json").resolve()),
        "upstream_commit": audit_record.get("upstream_commit"),
        "nm_n": audit_record["sparsity"].get("n"),
        "nm_m": audit_record["sparsity"].get("m"),
        "runtime_sparse_value_loading_verified": audit_record[
            "runtime_sparse_value_loading"
        ].get("verified"),
        "elapsed_sec": elapsed,
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "requests_per_sec": requests / elapsed if elapsed and requests else None,
        "output_tokens_per_sec": generation_tokens / elapsed
        if elapsed and generation_tokens
        else None,
        "total_tokens_per_sec": (prompt_tokens + generation_tokens) / elapsed
        if elapsed and prompt_tokens + generation_tokens
        else None,
        "verification_steps": verification_steps,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "draft_acceptance_rate": accepted_tokens / draft_tokens
        if draft_tokens
        else None,
        "mean_acceptance_length": 1.0 + accepted_tokens / verification_steps
        if verification_steps
        else None,
        "result_path": str(result_path) if result_path else "",
        "created_at": timestamp(),
    }
    write_json(run_dir / "run_meta.json", row)
    return row


def load_existing_row(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_meta.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if (
        row.get("status") != "ok"
        or row.get("audit_pass") is not True
        or int(row.get("samples") or 0) != int(row.get("expected_samples") or -1)
    ):
        return None
    result_path = result_json(run_dir)
    if result_path is None:
        return None
    return row


def add_references(rows: list[dict[str, Any]]) -> None:
    dense_references: dict[tuple[str, str], dict[str, Any]] = {}
    d8_references: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["model_label"]), str(row["task_key"]))
        if row.get("method") == DENSE_METHOD:
            dense_references[key] = row
        elif row.get("dense_eighths") is not None and int(
            row["dense_eighths"]
        ) == 8:
            d8_references[key] = row
    for row in rows:
        key = (str(row["model_label"]), str(row["task_key"]))
        for prefix, reference in (
            ("dense_eagle3", dense_references.get(key)),
            ("d8", d8_references.get(key)),
        ):
            row[f"{prefix}_accuracy"] = (
                reference.get("accuracy") if reference else None
            )
            row[f"accuracy_delta_pp_vs_{prefix}"] = (
                (float(row["accuracy"]) - float(reference["accuracy"])) * 100.0
                if reference
                and row.get("accuracy") is not None
                and reference.get("accuracy") is not None
                else None
            )
            reference_throughput = (
                float(reference["output_tokens_per_sec"])
                if reference and reference.get("output_tokens_per_sec")
                else None
            )
            row[f"output_speedup_vs_{prefix}"] = (
                float(row["output_tokens_per_sec"]) / reference_throughput
                if reference_throughput and row.get("output_tokens_per_sec")
                else None
            )


def macro_rows(
    rows: list[dict[str, Any]], required_tasks: list[str]
) -> list[dict[str, Any]]:
    """Return only fully audited model/structure rows.

    A partial method is deliberately absent instead of appearing with a
    misleading macro over fewer tasks.
    """

    output: list[dict[str, Any]] = []
    for model_label in MODEL_CONFIGS:
        for dense_eighths in [None, *range(0, 9)]:
            method = method_label(dense_eighths)
            selected = {
                str(row["task_key"]): row
                for row in rows
                if row.get("status") == "ok"
                and row.get("audit_pass") is True
                and row["model_label"] == model_label
                and row.get("method") == method
                and int(row.get("samples") or 0)
                == int(row.get("expected_samples") or -1)
            }
            if set(selected) != set(required_tasks):
                continue
            accuracy_values = [
                float(selected[task]["accuracy"]) for task in required_tasks
            ]
            throughput_values = [
                float(selected[task]["output_tokens_per_sec"])
                for task in required_tasks
                if selected[task].get("output_tokens_per_sec")
            ]
            acceptance_values = [
                float(selected[task]["mean_acceptance_length"])
                for task in required_tasks
                if selected[task].get("mean_acceptance_length") is not None
            ]
            result = {
                "model_label": model_label,
                "model": MODEL_CONFIGS[model_label]["display"],
                "method": method,
                "dense_eighths": dense_eighths,
                "quota": quota_display(dense_eighths),
                "structure": (
                    "Dense"
                    if dense_eighths is None
                    else f"2:4 D{dense_eighths}/8"
                ),
                "nm_n": None if dense_eighths is None else 2,
                "nm_m": None if dense_eighths is None else 4,
                "tasks_completed": len(selected),
                "audit_pass": True,
                "macro_accuracy": statistics.mean(accuracy_values),
                "geomean_output_tokens_per_sec": math.exp(
                    statistics.mean(math.log(value) for value in throughput_values)
                )
                if throughput_values
                else None,
                "mean_acceptance_length": statistics.mean(acceptance_values)
                if acceptance_values
                else None,
            }
            for task in required_tasks:
                result[f"{task}_accuracy"] = float(
                    selected[task]["accuracy"]
                )
            output.append(result)
    dense_reference = {
        str(row["model_label"]): row
        for row in output
        if row.get("method") == DENSE_METHOD
    }
    d8_reference = {
        str(row["model_label"]): row
        for row in output
        if row.get("dense_eighths") is not None
        and int(row["dense_eighths"]) == 8
    }
    for row in output:
        for prefix, references in (
            ("dense_eagle3", dense_reference),
            ("d8", d8_reference),
        ):
            reference = references.get(str(row["model_label"]))
            row[f"macro_accuracy_delta_pp_vs_{prefix}"] = (
                (float(row["macro_accuracy"]) - float(reference["macro_accuracy"]))
                * 100
                if reference
                and row.get("macro_accuracy") is not None
                and reference.get("macro_accuracy") is not None
                else None
            )
            row[f"macro_accuracy_drop_pp_vs_{prefix}"] = (
                (float(reference["macro_accuracy"]) - float(row["macro_accuracy"]))
                * 100
                if reference
                and row.get("macro_accuracy") is not None
                and reference.get("macro_accuracy") is not None
                else None
            )
            row[f"geomean_output_speedup_vs_{prefix}"] = (
                float(row["geomean_output_tokens_per_sec"])
                / float(reference["geomean_output_tokens_per_sec"])
                if reference
                and row.get("geomean_output_tokens_per_sec")
                and reference.get("geomean_output_tokens_per_sec")
                else None
            )
        dense = dense_reference.get(str(row["model_label"]))
        for task in required_tasks:
            row[f"{task}_accuracy_drop_pp_vs_dense"] = (
                (
                    float(dense[f"{task}_accuracy"])
                    - float(row[f"{task}_accuracy"])
                )
                * 100
                if dense
                else None
            )
    return output


def plot_results(output_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(output_root / "plot_error.json", {"error": repr(exc)})
        return

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"qwen3_8b": "#2f6fb0", "llama3_1_8b": "#d46b27"}
    plotted_methods: list[int | None] = [*range(0, 9), None]
    x = list(range(len(plotted_methods)))
    labels = [
        "Dense" if value is None else f"D{value}/8" for value in plotted_methods
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True)
    for axis, task_key in zip(axes.flat, TASK_CONFIGS, strict=True):
        for model_label in MODEL_CONFIGS:
            selected = {
                str(row["method"]): float(row["accuracy"])
                for row in rows
                if row.get("status") == "ok"
                and row["task_key"] == task_key
                and row["model_label"] == model_label
                and row.get("accuracy") is not None
            }
            axis.plot(
                x,
                [
                    100.0 * selected.get(method_label(value), math.nan)
                    for value in plotted_methods
                ],
                marker="o",
                linewidth=2,
                color=colors[model_label],
                label=MODEL_CONFIGS[model_label]["display"],
            )
        axis.set_title(str(TASK_CONFIGS[task_key]["display"]))
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Dense-token quota")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "Official lm-eval CoT/few-shot accuracy: SpecLink D0/8–D8/8",
        y=0.985,
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(figures / "accuracy_by_quota.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True)
    for axis, task_key in zip(axes.flat, TASK_CONFIGS, strict=True):
        for model_label in MODEL_CONFIGS:
            selected = {
                str(row["method"]): float(row["output_tokens_per_sec"])
                for row in rows
                if row.get("status") == "ok"
                and row["task_key"] == task_key
                and row["model_label"] == model_label
                and row.get("output_tokens_per_sec")
            }
            axis.plot(
                x,
                [
                    selected.get(method_label(value), math.nan)
                    for value in plotted_methods
                ],
                marker="o",
                linewidth=2,
                color=colors[model_label],
                label=MODEL_CONFIGS[model_label]["display"],
            )
        axis.set_title(str(TASK_CONFIGS[task_key]["display"]))
        axis.set_ylabel("Output tokens/s")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Dense-token quota")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Official lm-eval vLLM throughput", y=0.985)
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(figures / "throughput_by_quota.png", dpi=180)
    plt.close(figure)


def write_report(
    output_root: Path,
    rows: list[dict[str, Any]],
    macros: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    def percent(value: Any) -> str:
        return "-" if value is None else f"{100 * float(value):.2f}%"

    def drop(value: Any) -> str:
        return "-" if value is None else f"{float(value):+.2f}"

    lines = [
        "# Qwen/Llama 8B official lm-eval 0.4.12 accuracy sweep",
        "",
        "## Protocol",
        "",
        "- Models: Qwen3-8B and Llama-3.1-8B; EAGLE3 K=7.",
        "- Methods: native dense EAGLE3 plus SpecLink D0/8 through D8/8.",
        "- D0 is the pure-2:4 verifier endpoint: all eight verifier rows use only the 2:4 base. Prefill remains exact dense.",
        "- D1--D8 keep the current verifier row dense and select d-1 of the seven draft rows for base+complement recovery.",
        f"- Serving batch: lm-eval batch size = HTTP concurrency = vLLM max-num-seqs = {args.batch_size}.",
        "- Execution: residual-complement Split-K2, global prefix-product routing, torch.compile, exact FULL_DECODE_ONLY CUDA Graph buckets, prefix cache off, route-stat writes off.",
        "- Tasks: official `gsm8k_cot` (8-shot), `minerva_math` (4-shot), `bbh_cot_fewshot` (3-shot), and `mmlu_pro` (5-shot). Each benchmark uses exactly 256 frozen seed-42 examples.",
        "- Prompting: lm-eval `--apply_chat_template` reads the unmodified checkpoint `tokenizer_config.json -> chat_template`; sampling cases receive only `enable_thinking=False` as a template argument. The Llama checkpoint template has no thinking branch, so this is an explicit no-op there.",
        (
            "- Generation: `max_gen_toks=256`; Qwen and Llama both use "
            "temperature=0.7, top_p=0.8, top_k=20, min_p=0."
            if args.llama_qwen_sampling
            else "- Generation: `max_gen_toks=256`; Qwen uses "
            "temperature=0.7, top_p=0.8, top_k=20, min_p=0; "
            "Llama uses greedy decoding."
        ),
        "- Metrics: GSM8K flexible-extract exact match, Minerva `math_verify`, BBH get-answer exact match, and MMLU-Pro custom-extract exact match.",
        "- Rows enter the table only after the case audit passes and all four tasks each log the exact requested sample count. Partial or failed cases are excluded.",
        "",
        "## Complete accuracy results",
        "",
        "Accuracy drop columns are percentage points relative to the same-model dense EAGLE3 baseline; positive means lower accuracy.",
        "",
        "| Model | Structure | GSM8K | Drop | Minerva | Drop | BBH | Drop | MMLU-Pro | Drop | Mean | Mean drop |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macros:
        lines.append(
            f"| {row['model']} | {row['structure']} | "
            f"{percent(row.get('gsm8k_accuracy'))} | "
            f"{drop(row.get('gsm8k_accuracy_drop_pp_vs_dense'))} | "
            f"{percent(row.get('minerva_accuracy'))} | "
            f"{drop(row.get('minerva_accuracy_drop_pp_vs_dense'))} | "
            f"{percent(row.get('bbh_accuracy'))} | "
            f"{drop(row.get('bbh_accuracy_drop_pp_vs_dense'))} | "
            f"{percent(row.get('mmlu_pro_accuracy'))} | "
            f"{drop(row.get('mmlu_pro_accuracy_drop_pp_vs_dense'))} | "
            f"{percent(row.get('macro_accuracy'))} | "
            f"{drop(row.get('macro_accuracy_drop_pp_vs_dense_eagle3'))} |"
        )
    failures = [row for row in rows if row.get("status") != "ok"]
    lines.extend(
        [
            "",
            "## Completeness and audit",
            "",
            f"- Fully included model/structure rows: {len(macros)}.",
            f"- Failed or incomplete task rows (excluded): {len(failures)}.",
            "- `experiment_audit.json` records model checkpoint identities, calibration data/cache hashes, source commit and dirty diff hash, lm-eval version, seed, and frozen sample maps.",
            "- Every `MODEL/METHOD/case_audit.json` verifies N:M, observed 50% weight sparsity, activation-aware calibration use, runtime base+complement sparse-value storage, module count, and released dense storage.",
            "- Every task `run_meta.json` records the actual sample JSONL count and requires it to equal the requested count.",
            "",
            "## Artifacts",
            "",
            "- `summary.csv`: one row per model/quota/task.",
            "- `accuracy_summary.csv`: only complete four-task audited rows, with all task accuracies and dense-relative drops.",
            "- `figures/accuracy_by_quota.png` and `figures/throughput_by_quota.png`.",
            "- `MODEL/dN/TASK/lm_eval_output/`: raw lm-eval result and sample JSONL.",
            "- `MODEL/dN/vllm_server.log`: model-load, compile, graph-capture, and kernel logs.",
            "",
            f"Command: `{shlex.join(sys.argv)}`",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_inputs(args: argparse.Namespace) -> None:
    if importlib.metadata.version("lm-eval") != "0.4.12":
        raise ValueError("formal protocol requires lm-eval exactly 0.4.12")
    if args.seed != 42:
        raise ValueError("formal protocol requires seed=42")
    if args.num_spec_tokens != 7:
        raise ValueError("eighth-based quotas require --num-spec-tokens 7")
    if args.server_start_retries < 1:
        raise ValueError("--server-start-retries must be positive")
    if args.dense_leafs_list and len(args.models_list) != 1:
        raise ValueError(
            "--dense-leafs is model-specific; select exactly one model to "
            "avoid applying a Llama preservation policy to Qwen (or vice versa)"
        )
    if not (
        args.batch_size == args.concurrency == args.max_num_seqs
    ) and not args.smoke:
        raise ValueError(
            "formal protocol requires batch-size=concurrency=max-num-seqs"
        )
    manifest = DATASET_ROOT / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} is missing; run prepare_lm_eval_official_256.py first"
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("lm_eval_version") != "0.4.12" or int(data["seed"]) != 42:
        raise ValueError("selection manifest version/seed does not match protocol")
    for task_key, config in TASK_CONFIGS.items():
        benchmark = str(config["manifest_key"])
        if int(data["benchmarks"][benchmark]["rows"]) != 256:
            raise ValueError(f"{benchmark} is not a fixed 256-example selection")
        samples_path = DATASET_ROOT / str(config["samples_file"])
        if sha256_file(samples_path) != data["benchmarks"][benchmark][
            "samples_sha256"
        ]:
            raise ValueError(f"{task_key} fixed samples hash mismatch")
        if sum(map(len, selected_samples(task_key).values())) != 256:
            raise ValueError(f"{task_key} fixed samples do not total 256")
    for model_label in args.models_list:
        if model_label not in MODEL_CONFIGS:
            raise ValueError(f"unknown model: {model_label}")
        for key in ("model", "speculator"):
            path = Path(MODEL_CONFIGS[model_label][key])
            if not path.exists():
                raise FileNotFoundError(path)
    for task_key in args.tasks_list:
        if task_key not in TASK_CONFIGS:
            raise ValueError(f"unknown task: {task_key}")
    if not args.calibration_cache_root.exists():
        raise FileNotFoundError(args.calibration_cache_root)
    if not C4_CALIBRATION_PROMPTS.exists():
        raise FileNotFoundError(C4_CALIBRATION_PROMPTS)
    if not args.smoke and args.limit:
        raise ValueError("formal results may not use --limit")


def run(args: argparse.Namespace) -> None:
    validate_inputs(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    server_args = make_server_args(args)
    audit = global_audit(args)
    write_json(output_root / "experiment_audit.json", audit)
    write_json(
        output_root / "run_config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not key.endswith("_list")
            },
            "models": args.models_list,
            "tasks": args.tasks_list,
            "eighths": args.eighths_list,
            "methods": [
                *([] if args.skip_dense_eagle3 else [DENSE_METHOD]),
                *(quota_label(value) for value in args.eighths_list),
            ],
            "graph_rows": graph_rows(args),
            "torch_version": importlib.metadata.version("torch"),
            "vllm_version": importlib.metadata.version("vllm"),
            "lm_eval_version": importlib.metadata.version("lm-eval"),
            "upstream_commit": audit["source"]["upstream_commit"],
            "working_tree_diff_sha256": audit["source"][
                "working_tree_diff_sha256"
            ],
            "selection_manifest_sha256": audit["protocol"][
                "selection_manifest_sha256"
            ],
            "max_gen_toks": MAX_GEN_TOKS,
            "created_at": timestamp(),
        },
    )

    rows: list[dict[str, Any]] = []
    methods: list[int | None] = [
        *([] if args.skip_dense_eagle3 else [None]),
        *args.eighths_list,
    ]
    for model_label in args.models_list:
        config = MODEL_CONFIGS[model_label]
        model = Path(config["model"]).resolve()
        speculator = Path(config["speculator"]).resolve()
        eval_tokenizer = prepare_eval_tokenizer(
            llama_qwen_sampling=args.llama_qwen_sampling,
            model=model,
            model_label=model_label,
            output_root=output_root,
        )
        for dense_eighths in methods:
            method = method_label(dense_eighths)
            case_dir = output_root / model_label / method
            pending: list[str] = []
            for task_key in args.tasks_list:
                existing = (
                    load_existing_row(case_dir / task_key) if args.resume else None
                )
                if existing is not None:
                    rows.append(existing)
                else:
                    pending.append(task_key)
            if not pending:
                continue

            print(
                f"[server] {model_label} {quota_display(dense_eighths)}; "
                f"pending={','.join(pending)}",
                flush=True,
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            env = make_env(
                args,
                model_label=model_label,
                dense_eighths=dense_eighths,
                case_dir=case_dir,
            )
            write_json(
                case_dir / "case_config.json",
                {
                    "model_label": model_label,
                    "method": method,
                    "dense_eighths": dense_eighths,
                    "total_dense_fraction": (
                        1.0 if dense_eighths is None else dense_eighths / 8.0
                    ),
                    "dense_draft_fraction": (
                        1.0
                        if dense_eighths is None
                        else max(0.0, (dense_eighths - 1) / 7.0)
                    ),
                    "num_spec_tokens": 7,
                    "score_mode": (
                        None if dense_eighths is None else "prefix_product"
                    ),
                    "routing_scope": None if dense_eighths is None else "global",
                    "batch_size": args.batch_size,
                    "concurrency": args.concurrency,
                    "max_num_seqs": args.max_num_seqs,
                    "compile_graph": True,
                    "persistent_compile_cache": True,
                    "quota_isolated_compile_cache": dense_eighths is not None,
                    "prefix_caching": False,
                    "step_stats": False,
                    "chat_template": True,
                    "chat_template_source": (
                        "checkpoint tokenizer_config.json -> chat_template"
                    ),
                    "chat_template_modified": False,
                    "chat_template_args": chat_template_args(
                        model_label,
                        llama_qwen_sampling=args.llama_qwen_sampling,
                    ),
                    "generation": generation_protocol(
                        model_label,
                        llama_qwen_sampling=args.llama_qwen_sampling,
                    ),
                    "seed": args.seed,
                    "max_gen_toks": MAX_GEN_TOKS,
                    "selection_manifest": str(
                        (DATASET_ROOT / "manifest.json").resolve()
                    ),
                    "continue_across_newline": True,
                    "always_dense_leafs": args.dense_leafs_list,
                },
            )
            process = None
            try:
                for start_attempt in range(1, args.server_start_retries + 1):
                    try:
                        process, port = start_vllm_server(
                            server_args,
                            base_model=str(model),
                            speculator_model=str(speculator),
                            case_dir=case_dir,
                            env=env,
                        )
                        break
                    except RuntimeError:
                        failed_log = case_dir / (
                            f"vllm_server_start_failure_{start_attempt}.log"
                        )
                        server_log = case_dir / "vllm_server.log"
                        if server_log.exists():
                            server_log.replace(failed_log)
                        if start_attempt >= args.server_start_retries:
                            raise
                        print(
                            f"[server-retry] {model_label} "
                            f"{quota_display(dense_eighths)} attempt "
                            f"{start_attempt}/{args.server_start_retries} failed",
                            flush=True,
                        )
                        time.sleep(args.server_shutdown_settle_s)
                if process is None:
                    raise RuntimeError("vLLM server did not start")
                case_record = case_audit(
                    model_label=model_label,
                    dense_eighths=dense_eighths,
                    case_dir=case_dir,
                    global_record=audit,
                    dense_leafs=args.dense_leafs_list,
                )
                if case_record["status"] != "passed":
                    raise RuntimeError(
                        f"case audit failed: {case_record['errors']}"
                    )
                warmup_args = argparse.Namespace(
                    concurrency=args.concurrency,
                    warmup_tokens=args.warmup_tokens,
                    request_timeout_s=args.request_timeout_s,
                )
                warmup_full_batch(port, str(model), warmup_args)
                for task_key in pending:
                    print(
                        f"[task] {model_label} "
                        f"{quota_display(dense_eighths)} {task_key}",
                        flush=True,
                    )
                    run_dir = case_dir / task_key
                    run_dir.mkdir(parents=True, exist_ok=True)
                    row = run_task(
                        args,
                        port=port,
                        model_label=model_label,
                        model=model,
                        tokenizer=eval_tokenizer,
                        dense_eighths=dense_eighths,
                        task_key=task_key,
                        run_dir=run_dir,
                        audit_record=case_record,
                    )
                    rows.append(row)
                    print(
                        f"[result] status={row['status']} "
                        f"accuracy={row['accuracy']} samples={row['samples']} "
                        f"output_tok_s={row['output_tokens_per_sec']} "
                        f"accept_len={row['mean_acceptance_length']}",
                        flush=True,
                    )
                    # Persist a usable partial summary after every task.
                    partial = sorted(
                        rows,
                        key=lambda item: (
                            str(item["model_label"]),
                            method_order(item.get("dense_eighths")),
                            str(item["task_key"]),
                        ),
                    )
                    add_references(partial)
                    write_csv(output_root / "summary.csv", partial)
            finally:
                stop_process(process)
                time.sleep(args.server_shutdown_settle_s)

    rows.sort(
        key=lambda item: (
            str(item["model_label"]),
            method_order(item.get("dense_eighths")),
            str(item["task_key"]),
        )
    )
    add_references(rows)
    macros = macro_rows(rows, args.tasks_list)
    write_csv(output_root / "summary.csv", rows)
    write_json(output_root / "summary.json", rows)
    write_csv(output_root / "accuracy_summary.csv", macros)
    write_json(output_root / "accuracy_summary.json", macros)
    plot_results(output_root, rows)
    write_report(output_root, rows, macros, args)
    print(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--tasks", default="gsm8k,minerva,bbh,mmlu_pro")
    parser.add_argument("--eighths", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument(
        "--skip-dense-eagle3",
        action="store_true",
        help="Do not run the native dense EAGLE3 reference.",
    )
    parser.add_argument(
        "--llama-qwen-sampling",
        action="store_true",
        help=(
            "Use Qwen's sampling protocol for Llama too: temperature=0.7, "
            "top_p=0.8, top_k=20, min_p=0, with thinking disabled."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-spec-tokens", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=8360)
    parser.add_argument("--health-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=int, default=1800)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=3.0)
    parser.add_argument("--server-start-retries", type=int, default=2)
    parser.add_argument(
        "--calibration-cache-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
    )
    parser.add_argument(
        "--dense-leafs",
        default="",
        help=(
            "Comma-separated fused vLLM linear leaf names kept fully dense "
            "for a single selected model, for example gate_up_proj,o_proj. "
            "This option requires --models to name exactly one model."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.models_list = csv_list(args.models)
    args.tasks_list = csv_list(args.tasks)
    args.eighths_list = parse_eighths(args.eighths)
    args.dense_leafs_list = csv_list(args.dense_leafs)
    if args.smoke:
        args.models_list = args.models_list[:1]
        args.eighths_list = args.eighths_list[:1]
        args.limit = args.limit or 4
        args.batch_size = min(args.batch_size, 4)
        args.concurrency = min(args.concurrency, 4)
        args.max_num_seqs = min(args.max_num_seqs, 4)
    if args.output_root is None:
        parent = EVAL_ROOT / ("temp" if args.smoke else "results_final")
        prefix = (
            "lm_eval_official_quota_smoke"
            if args.smoke
            else "lm_eval_official_quota"
        )
        args.output_root = parent / f"{prefix}_{timestamp()}"
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
