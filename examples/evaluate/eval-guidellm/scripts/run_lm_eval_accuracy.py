#!/usr/bin/env python3
"""Run lm-eval through this repo's vLLM/EAGLE3 serving path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
DEFAULT_OUTPUT_DIR = (
    EVAL_ROOT
    / "results"
    / "token_dense_lm_eval"
    / "lm_eval"
)
DEFAULT_CONFIG = EVAL_ROOT / "configs" / "lm_eval_accuracy.yaml"
LOCAL_TASKS = EVAL_ROOT / "eval_tasks"
DEFAULT_SR24_MASKS = {
    "llama3_1_8b": (
        EVAL_ROOT
        / "data/c4_calibration/sr24_masks/llama3_1_8b_activation_aware_24.pt"
    ),
}

sys.path.insert(0, str(SCRIPT_DIR))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    LAYER_SENSITIVITY_DEFAULT_MODELS,
    parse_csv_list,
    parse_model_id_overrides,
    write_json,
)
from run_structured_24_spec_quality import (  # noqa: E402
    DEFAULT_BASE_MODELS,
    EAGLE3_SPECULATORS,
    add_local_no_proxy,
    configure_local_no_proxy,
    find_free_port,
    scrape_spec_metrics,
    stop_process,
    wait_for_health,
)
from token_dense_methods import (  # noqa: E402
    MethodConfig,
    TOKEN_DENSE_METHODS,
    method_env,
    parse_method_config,
    timestamp,
)

HYBRID_METHODS = ["activation_aware", *TOKEN_DENSE_METHODS]
SR24_RUNTIME_MODES = {
    "base_only_24": "base_only",
    "all_corrected_24": "all_corrected",
    "speclink_t08": "selective",
}
SR24_METHODS = ["dense_baseline", *SR24_RUNTIME_MODES.keys()]
MODE_GROUPS = {
    "dense": ["dense_ar"],
    "speculative": ["eagle3_dense"],
    "hybrid": HYBRID_METHODS,
    "token_dense": TOKEN_DENSE_METHODS,
    "sr24": SR24_METHODS,
    "all": ["dense_ar", "eagle3_dense", *HYBRID_METHODS],
}
TASK_GROUPS = {
    "smoke": [
        "gsm8k_cot",
        "minerva_math500",
        "gpqa_diamond_cot_zeroshot",
        "ifeval",
        "humaneval_instruct",
        "longbench_multi_news",
    ],
    "all": [
        "gsm8k_cot",
        "minerva_math500",
        "gpqa_diamond_cot_zeroshot",
        "ifeval",
        "humaneval_instruct",
        "longbench_multi_news",
    ],
}
TASK_MAX_TOKENS = {
    "gsm8k_cot": 512,
    "minerva_math500": 512,
    "gpqa_diamond_cot_zeroshot": 512,
    "ifeval": 512,
    "humaneval_instruct": 512,
    "longbench_multi_news": 512,
}
TASK_KNOWN_TOTALS = {
    # These public tasks are smaller than the requested 200-row formal sample.
    "gpqa_diamond_cot_zeroshot": 198,
    "humaneval_instruct": 164,
}
TASK_PREFLIGHT_DATASETS = {
    # GPQA is gated on Hugging Face. Check access before starting a vLLM server
    # so an unauthorized token does not waste one GPU startup per mode.
    "gpqa_diamond_cot_zeroshot": ("Idavidrein/gpqa", "gpqa_diamond", "train[:1]"),
}
SR24_ITERATION_FIELDS = [
    "version",
    "git_diff_or_commit",
    "change_description",
    "hypothesis",
    "exact_command",
    "batch_size",
    "accuracy",
    "TPS",
    "speedup",
    "residual_ratio",
    "peak_VRAM",
    "profiling_observation",
    "kept_or_reverted",
    "reason",
]


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expand_modes(value: str) -> list[str]:
    out: list[str] = []
    for item in csv_list(value):
        if item in MODE_GROUPS:
            candidates = MODE_GROUPS[item]
        elif item.startswith("t") and item[1:].isdigit():
            candidates = [f"token_dense_{item}"]
        else:
            candidates = [item]
        for mode in candidates:
            if mode not in out:
                out.append(mode)
    return out


def expand_tasks(value: str) -> list[str]:
    out: list[str] = []
    for item in csv_list(value):
        candidates = TASK_GROUPS.get(item, [item])
        for task in candidates:
            if task not in out:
                out.append(task)
    return out


def manifest_request_count(args: argparse.Namespace) -> int:
    if args.manifest_size > 0:
        return args.manifest_size
    if args.limit:
        try:
            return max(0, int(float(args.limit)))
        except ValueError:
            return 0
    return 0


def ensure_task_manifest(
    args: argparse.Namespace,
    task: str,
) -> tuple[Path | None, str | None, int | None, str]:
    """Create or load an lm-eval --samples manifest for one task."""
    if not args.use_task_manifests:
        return None, None, None, ""
    requested = manifest_request_count(args)
    if requested <= 0:
        return None, None, None, ""
    actual = min(requested, TASK_KNOWN_TOTALS.get(task, requested))
    ids = list(range(actual))
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    path = args.manifest_dir / f"{task}_{requested}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing = data.get(task)
            if isinstance(existing, list):
                ids = [int(item) for item in existing]
                actual = len(ids)
        except Exception:
            # Rewrite malformed manifests instead of silently using bad samples.
            data = {task: ids}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        data = {task: ids}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    reason = ""
    if actual < requested:
        reason = f"task has fewer than requested samples; using {actual}/{requested}"
    return path, json.dumps({task: ids}), actual, reason


def extract_failure_line(text: str) -> str:
    priority_tokens = (
        "DatasetNotFoundError",
        "gated dataset",
        "Access to dataset",
        "Access to this dataset",
        "Cannot access gated repo",
        "RuntimeError:",
        "ValueError:",
        "ERROR",
    )
    fallback = ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in priority_tokens):
            return stripped[:500]
        if not fallback:
            fallback = stripped[:500]
    return fallback


def preflight_task_access(
    args: argparse.Namespace,
    task: str,
    run_dir: Path,
) -> str:
    dataset_info = TASK_PREFLIGHT_DATASETS.get(task)
    if dataset_info is None:
        return ""
    dataset_path, dataset_name, split = dataset_info
    env = add_local_no_proxy(os.environ.copy())
    configure_eval_cache_env(args, env)
    command = [
        sys.executable,
        "-c",
        (
            "from datasets import load_dataset; "
            f"load_dataset({dataset_path!r}, {dataset_name!r}, split={split!r})"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(EVAL_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        reason = extract_failure_line(stdout) or f"task preflight timed out for {task}"
        write_json(
            run_dir / "task_preflight.json",
            {
                "task": task,
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "split": split,
                "command": command,
                "returncode": "timeout",
                "error": reason,
            },
        )
        (run_dir / "task_preflight.log").write_text(stdout, encoding="utf-8")
        return reason
    if completed.returncode == 0:
        return ""
    reason = extract_failure_line(completed.stdout)
    write_json(
        run_dir / "task_preflight.json",
        {
            "task": task,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "split": split,
            "command": command,
            "returncode": completed.returncode,
            "error": reason,
        },
    )
    (run_dir / "task_preflight.log").write_text(completed.stdout, encoding="utf-8")
    return reason or f"task preflight failed for {task}"


def mode_method(mode: str) -> MethodConfig:
    if mode in {"dense_ar", "eagle3_dense", "dense_baseline"}:
        return MethodConfig(label="dense", base_method="dense", policy="dense")
    if mode in SR24_RUNTIME_MODES:
        return MethodConfig(label=mode, base_method="sr24", policy="dense")
    return parse_method_config(mode)


def mode_uses_spec(mode: str) -> bool:
    return mode != "dense_ar"


def mode_group(mode: str) -> str:
    if mode == "dense_ar":
        return "dense"
    if mode in {"eagle3_dense", "dense_baseline"}:
        return "speculative"
    if mode in SR24_RUNTIME_MODES:
        return "sr24"
    return "hybrid"


def sr24_runtime_mode(mode: str) -> str:
    try:
        return SR24_RUNTIME_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"not an SR24 mode: {mode}") from exc


def sr24_has_narrow_residual_scope(args: argparse.Namespace) -> bool:
    """Whether compressed residual prewarm is likely bounded enough to fit.

    Full-model all-corrected with `compressed_dense` can OOM on a 32GB GPU if
    the auto fastpath prewarms and caches dense residual weights for every
    linear module. The auto fastpath is intended for targeted operator
    ablations, so require an explicit leaf/layer scope before enabling it.
    """
    scoped_fields = (
        "sr24_target_leafs",
        "sr24_residual_target_leafs",
        "sr24_base_only_layer_ids",
        "sr24_base_only_layer_ids_by_leaf",
        "sr24_runtime_base_only_layer_ids_by_leaf",
        "sr24_residual_layer_ids_by_leaf",
    )
    return any(str(getattr(args, field, "") or "").strip() for field in scoped_fields)


def sr24_compressed_residual_runtime_settings(
    args: argparse.Namespace,
    mode: str,
) -> tuple[int, bool, bool, bool]:
    """Return effective compressed-residual settings for one SR24 mode.

    Keep lm-eval SR24 runs aligned with the GuideLLM throughput runner. The
    no-dense-fastpath all_corrected+compressed_dense path is an operator
    ablation; its best current form keeps the dense residual tensor GPU-resident
    and prewarmed instead of rebuilding it inside the forward path.
    """
    residual_out_chunk = int(args.sr24_residual_out_chunk)
    cache_weight = bool(args.sr24_cache_compressed_residual_weight)
    prewarm_weight = bool(getattr(args, "sr24_prewarm_compressed_residual_weight", False))
    auto_fastpath = (
        bool(getattr(args, "sr24_auto_compressed_residual_fastpath", True))
        and mode == "all_corrected_24"
        and args.sr24_backend == "torch_sparse"
        and args.sr24_residual_backend == "compressed_dense"
        and not args.sr24_all_corrected_dense_fastpath
        and not args.sr24_compressed_residual_triton
        and sr24_has_narrow_residual_scope(args)
    )
    if auto_fastpath:
        residual_out_chunk = 0
        cache_weight = True
        prewarm_weight = True
    return residual_out_chunk, cache_weight, prewarm_weight, auto_fastpath


def sr24_effective_direct_cslt_linear(
    args: argparse.Namespace,
    mode: str,
) -> bool:
    if bool(args.sr24_direct_cslt_linear):
        return True
    return (
        bool(getattr(args, "sr24_auto_direct_cslt_base_only", True))
        and mode == "base_only_24"
        and args.sr24_backend == "torch_sparse"
        and args.sr24_gate_up_split == "none"
    )


def sr24_compile_cache_root(args: argparse.Namespace, mode: str) -> Path:
    (
        residual_out_chunk,
        cache_compressed_residual_weight,
        prewarm_compressed_residual_weight,
        auto_compressed_residual_fastpath,
    ) = sr24_compressed_residual_runtime_settings(args, mode)
    fingerprint = {
        "cache_version": "opaque_dense_v1",
        "mode": mode,
        "runtime_mode": sr24_runtime_mode(mode),
        "backend": args.sr24_backend,
        "residual_backend": args.sr24_residual_backend,
        "residual_device": args.sr24_residual_device,
        "require_gpu_residual": args.sr24_require_gpu_residual,
        "threshold": args.sr24_threshold,
        "static_mask_state": args.sr24_static_mask_state,
        "static_all_residual_dense_fastpath":
        args.sr24_static_all_residual_dense_fastpath,
        "all_corrected_dense_fastpath": args.sr24_all_corrected_dense_fastpath,
        "full_residual_early_dense": args.sr24_full_residual_early_dense,
        "direct_cslt_linear": sr24_effective_direct_cslt_linear(args, mode),
        "auto_direct_cslt_base_only": args.sr24_auto_direct_cslt_base_only,
        "static_mask_buffer": args.sr24_static_mask_buffer,
        "batched_mask_builder": args.sr24_batched_mask_builder,
        "batched_uniform_direct": args.sr24_batched_uniform_direct,
        "gpu_count_mask_builder": args.sr24_gpu_count_mask_builder,
        "gate_up_split": args.sr24_gate_up_split,
        "gate_up_channel_dense_fraction":
        args.sr24_gate_up_channel_dense_fraction,
        "gate_up_channel_strategy": args.sr24_gate_up_channel_strategy,
        "gate_up_channel_fused_act": args.sr24_gate_up_channel_fused_act,
        "row_routed_mlp": args.sr24_row_routed_mlp,
        "row_routed_down_linear": args.sr24_row_routed_down_linear,
        "row_routed_mlp_reuse_base_output":
        args.sr24_row_routed_mlp_reuse_base_output,
        "row_routed_mlp_fixed_block_dense_fill":
        args.sr24_row_routed_mlp_fixed_block_dense_fill,
        "fixed_block_input_buffer": args.sr24_fixed_block_input_buffer,
        "fixed_block_output_buffer": args.sr24_fixed_block_output_buffer,
        "scheduler_policy_path": args.sr24_scheduler_policy_path,
        "scheduler_policy_dense_bypass":
        args.sr24_scheduler_policy_dense_bypass,
        "runtime_base_only_layer_ids_by_leaf":
        args.sr24_runtime_base_only_layer_ids_by_leaf,
        "row_routed_mlp_min_dense_rows": args.sr24_row_routed_mlp_min_dense_rows,
        "row_routed_mlp_min_dense_rows_by_leaf":
        args.sr24_row_routed_mlp_min_dense_rows_by_leaf,
        "row_routed_mlp_max_dense_rows": args.sr24_row_routed_mlp_max_dense_rows,
        "row_routed_mlp_max_dense_rows_by_leaf":
        args.sr24_row_routed_mlp_max_dense_rows_by_leaf,
        "row_routed_mlp_max_base_rows": args.sr24_row_routed_mlp_max_base_rows,
        "row_routed_mlp_max_base_rows_by_leaf":
        args.sr24_row_routed_mlp_max_base_rows_by_leaf,
        "route_all_skip_bucket": args.sr24_route_all_skip_bucket,
        "noverify_dense_mlp_fastpath":
        args.sr24_noverify_dense_mlp_fastpath,
        "selective_correct_non_draft": args.sr24_selective_correct_non_draft,
        "selective_non_draft_policy": args.sr24_selective_non_draft_policy,
        "selective_dense_nonverify_layer_ids_by_leaf":
        args.sr24_selective_dense_nonverify_layer_ids_by_leaf,
        "selective_dense_nonverify_max_rows":
        args.sr24_selective_dense_nonverify_max_rows,
        "selective_residual_policy": args.sr24_selective_residual_policy,
        "prefix_threshold": args.sr24_prefix_threshold,
        "selective_extra_after_low": args.sr24_selective_extra_after_low,
        "selective_min_prefix_residual":
        args.sr24_selective_min_prefix_residual,
        "selective_max_residual_draft_rows":
        args.sr24_selective_max_residual_draft_rows,
        "low_confidence_cap_by_risk": args.sr24_low_confidence_cap_by_risk,
        "early_dense_tokens": args.sr24_early_dense_tokens,
        "target_leafs": args.sr24_target_leafs,
        "residual_target_leafs": args.sr24_residual_target_leafs,
        "base_only_layer_ids": args.sr24_base_only_layer_ids,
        "base_only_layer_ids_by_leaf": args.sr24_base_only_layer_ids_by_leaf,
        "residual_layer_ids_by_leaf": args.sr24_residual_layer_ids_by_leaf,
        "residual_out_chunk": residual_out_chunk,
        "cache_compressed_residual_weight": cache_compressed_residual_weight,
        "prewarm_compressed_residual_weight": prewarm_compressed_residual_weight,
        "auto_compressed_residual_fastpath": auto_compressed_residual_fastpath,
        "compressed_residual_triton": args.sr24_compressed_residual_triton,
        "extract_chunk_rows": args.sr24_extract_chunk_rows,
        "residual_bucket_size": args.sr24_residual_bucket_size,
        "residual_bucket_scale_by_active":
        args.sr24_residual_bucket_scale_by_active,
        "residual_bucket_priority": args.sr24_residual_bucket_priority,
        "bonus_priority": args.sr24_bonus_priority,
        "draft_position_priority_scale": args.sr24_draft_position_priority_scale,
        "route_bucket_rows": args.sr24_route_bucket_rows,
        "route_all_residual_rows": args.sr24_route_all_residual_rows,
        "direct_cpu_route_rows": args.sr24_direct_cpu_route_rows,
        "route_reuse_base_output": args.sr24_route_reuse_base_output,
        "route_contiguous_fastpath": args.sr24_route_contiguous_fastpath,
        "route_dense_fallback_fraction":
        args.sr24_route_dense_fallback_fraction,
        "route_min_dense_rows": args.sr24_route_min_dense_rows,
        "route_min_base_rows": args.sr24_route_min_base_rows,
        "route_min_base_rows_by_leaf":
        args.sr24_route_min_base_rows_by_leaf,
        "route_max_dense_fraction": args.sr24_route_max_dense_fraction,
        "adaptive_dense_fallback": args.sr24_adaptive_dense_fallback,
        "adaptive_dense_fallback_no_residual_only":
        args.sr24_adaptive_dense_fallback_no_residual_only,
        "adaptive_dense_fallback_small_rows":
        args.sr24_adaptive_dense_fallback_small_rows,
        "adaptive_dense_fallback_gate_up_fraction":
        args.sr24_adaptive_dense_fallback_gate_up_fraction,
        "adaptive_dense_fallback_down_fraction":
        args.sr24_adaptive_dense_fallback_down_fraction,
        "adaptive_dense_fallback_small_down_no_residual":
        args.sr24_adaptive_dense_fallback_small_down_no_residual,
        "adaptive_dense_fallback_small_gate_up_no_residual":
        args.sr24_adaptive_dense_fallback_small_gate_up_no_residual,
        "triton_route_assembly": args.sr24_triton_route_assembly,
        "triton_bucket_override": args.sr24_triton_bucket_override,
        "triton_bucket_dense_gemm": args.sr24_triton_bucket_dense_gemm,
        "triton_bucket_scatter": args.sr24_triton_bucket_scatter,
        "triton_bucket_dense_block_m":
        args.sr24_triton_bucket_dense_block_m,
        "triton_bucket_dense_block_n":
        args.sr24_triton_bucket_dense_block_n,
        "triton_bucket_dense_block_k":
        args.sr24_triton_bucket_dense_block_k,
        "bucket_dense_copy": args.sr24_bucket_dense_copy,
        "bucket_dense_copy_active_only":
        args.sr24_bucket_dense_copy_active_only,
        "bucket_dense_compute_active_only":
        args.sr24_bucket_dense_compute_active_only,
        "bucket_dense_active_mask_fused":
        args.sr24_bucket_dense_active_mask_fused,
        "disable_runtime_stats": args.sr24_disable_runtime_stats,
        "reduce_cpu_sync": args.sr24_reduce_cpu_sync,
        "sync_mask_state": args.sr24_sync_mask_state,
        "breakdown": args.sr24_breakdown,
        "breakdown_linear": args.sr24_breakdown_linear,
        "breakdown_exact_routing": args.sr24_breakdown_exact_routing,
        "breakdown_gpu_counts": args.sr24_breakdown_gpu_counts,
        "breakdown_interval": args.sr24_breakdown_interval,
        "force_cudagraph_none_for_mixed":
        args.sr24_force_cudagraph_none_for_mixed,
        "dynamic_auto_cudagraph": args.sr24_dynamic_auto_cudagraph,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return EVAL_ROOT / "temp" / "vllm_compile_cache" / f"sr24_lmeval_{digest}"


def non_sr24_compile_cache_root(args: argparse.Namespace, mode: str) -> Path:
    method = mode_method(mode)
    fingerprint = {
        "cache_version": "non_sr24_lmeval_v1",
        "mode": mode,
        "method_label": method.label,
        "base_method": method.base_method,
        "policy": method.policy,
        "num_spec_tokens": int(args.num_spec_tokens),
        "max_num_seqs": int(args.max_num_seqs),
        "max_context_length": int(args.max_context_length),
        "enforce_eager": bool(args.enforce_eager),
        "vllm_compilation_config": args.vllm_compilation_config,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return EVAL_ROOT / "temp" / "vllm_compile_cache" / f"non_sr24_lmeval_{digest}"


def apply_sr24_preset(args: argparse.Namespace) -> None:
    preset = getattr(args, "sr24_preset", "manual")
    if preset == "manual":
        return
    if preset == "quality_safe_selective":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = (
            "gate_up_proj=16-31;down_proj=8-15"
        )
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.9
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = True
        args.sr24_adaptive_dense_fallback = True
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "down8_15_residual_only":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "down_proj"
        args.sr24_residual_target_leafs = "down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "down_proj=8-15"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.9
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = True
        args.sr24_adaptive_dense_fallback = True
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "quality_gateup_only":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "all_if_any_low"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.4
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 0
        args.sr24_residual_bucket_priority = False
        args.sr24_adaptive_dense_fallback = True
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "gateup_cap0_dense_guard":
        # Guarded selective candidate from the 2026-06-27 slowdown pass:
        # cap0 low-confidence correction on gate_up=16-31, plus a conservative
        # dense fallback when residual rows make the mixed sparse+correction
        # path unattractive. It is not paired-safe on the latest GSM8K-20
        # serving check, so treat it as a diagnostic candidate.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = False
        args.sr24_adaptive_dense_fallback = True
        args.sr24_adaptive_dense_fallback_gate_up_fraction = 0.05
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "gateup_cap0_maskstate_densefallback06":
        # Speed/quality candidate after the seven-part slowdown breakdown:
        # keep mixed steps guarded by eager execution, but synchronize the
        # per-step mask state once and promote high-residual steps to the exact
        # all-residual dense fastpath. This is conservative for quality because
        # it corrects extra rows instead of leaving them base-only, and it lets
        # promoted all-residual steps use normal vLLM CUDA Graph dispatch.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = True
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_route_dense_fallback_fraction = 0.6
        args.sr24_adaptive_dense_fallback = False
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "gateup_cap0_maskstate_densefallback00":
        # More conservative variant of gateup_cap0_maskstate_densefallback06:
        # any step with at least one residual row is promoted to all-residual.
        # This identifies whether remaining paired regressions are caused by
        # rare mixed steps. It is also the current quality-safe speed baseline:
        # use vLLM's default compile path because the dynamic route is exact
        # all/no-residual only, and the SR24-specific compile config slows this
        # dense-equivalent fallback without improving correctness.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = True
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_route_dense_fallback_fraction = 0.0
        args.sr24_adaptive_dense_fallback = False
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "gateup_cap0_graph_probe":
        # Explicit opt-in speed/quality candidate: same guard as
        # gateup_cap0_dense_guard, with dynamic-auto CUDA Graph enabled through
        # a stable bucket buffer. Keep this separate because GSM8K-10 found a
        # paired regression even when all draft rows were residual-corrected.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = False
        args.sr24_cudagraph_bucket = True
        # Match the throughput runner: direct cuSPARSELt remains graphable for
        # stable decode shapes, while mixed prefill/decode steps run eager to
        # avoid cuSPARSELt gather bounds failures under CUDA Graph replay.
        args.sr24_force_cudagraph_none_for_mixed = True
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_adaptive_dense_fallback = True
        args.sr24_adaptive_dense_fallback_gate_up_fraction = 0.05
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "speed_tradeoff_down16_base":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = "down_proj=16-31"
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_max_residual_draft_rows = 1
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = False
        args.sr24_adaptive_dense_fallback = True
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "riskcap2_bucket16_directcslt":
        # Post-breakdown speed candidate: keep the current bucket16/direct-
        # cuSPARSELt target scope, but reduce corrected draft rows to a small
        # mandatory prefix plus the most risky low-confidence positions. This
        # candidate must pass paired accuracy gates before it is used as a
        # throughput baseline.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = (
            "gate_up_proj=16-31;down_proj=8-15"
        )
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_max_residual_draft_rows = 2
        args.sr24_low_confidence_cap_by_risk = True
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 16
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        # Use vLLM's default compile path for the normal speed/quality
        # candidate. The SR24-specific FULL_DECODE_ONLY cache and row-routed
        # MLP path are separate ablations because paired GSM8K gates found
        # stable regressions there.
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        # Match the throughput runner: direct cuSPARSELt remains graphable for
        # stable decode shapes, while mixed prefill/decode steps run eager to
        # avoid cuSPARSELt gather bounds failures under CUDA Graph replay.
        args.sr24_force_cudagraph_none_for_mixed = True
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "lowresidual_gateup_riskcap2":
        # Match the throughput runner's current best measured gate-up-only
        # low-residual route. Keep this separate from
        # riskcap2_bucket16_directcslt, which also corrects down_proj and was
        # slower in the 2026-06-29 bucket follow-up.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_max_residual_draft_rows = 2
        args.sr24_low_confidence_cap_by_risk = True
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 8
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset in {"mlpall_lowconf_prefix5_tritonoverride", "mlpall_direct_prefix2"}:
        # All-MLP speed-target probe from the current SR24 slowdown pass.
        # It can be useful for lm-eval quality gates, but is not yet the
        # default quality-safe preset.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = ""
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.6
        args.sr24_selective_min_prefix_residual = 5
        if preset == "mlpall_direct_prefix2":
            args.sr24_selective_min_prefix_residual = 2
            args.sr24_direct_cslt_linear = True
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_allow_cudagraph = True
        args.sr24_triton_bucket_override = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset in {"mlpall_fixedprefix2_directcslt", "mlpall_fixedprefix2_graphsafe"}:
        # Score-free fixed route-table candidates matching the throughput
        # runner. They preserve a fixed prefix2 plus the bonus row and avoid
        # DLM selected-probability score collection.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = ""
        args.sr24_selective_residual_policy = "fixed_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_extra_after_low = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = preset == "mlpall_fixedprefix2_directcslt"
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = (
            preset == "mlpall_fixedprefix2_directcslt"
        )
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "mlpall_tilefill_prefix2_bucket32_cublas":
        # Match the GuideLLM throughput runner's tile-fill probe: fixed
        # bucket32 dense overwrite with cuBLAS, no active-only small GEMM.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = ""
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.6
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_extra_after_low = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_bucket_dense_copy_active_only = False
        args.sr24_bucket_dense_compute_active_only = False
        args.sr24_bucket_dense_active_mask_fused = False
        args.sr24_triton_bucket_override = False
        args.sr24_triton_bucket_dense_gemm = False
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset in {
        "lossy_prefix2_rowrouted_mlp",
        "lossy_prefix2_rowrouted_mlp_operator_guard",
    }:
        # 8pp-budget systems candidate: fixed route-table rows, disjoint
        # dense/sparse MLP branches, and no dense correction for rows that
        # already used the 2:4 sparse base. It protects the first two draft
        # rows plus the bonus row, then lets later draft rows remain sparse.
        # The fixed-prefix descriptor-only route plus reusable input buffers
        # are now part of the preset so quality and throughput runs do not
        # accidentally fall back to the older row-list/temporary-allocation
        # path. row_routed_mlp_min_dense_rows fills the dense branch with extra
        # low-priority rows when the protected set is too small for an
        # efficient GEMM tile. The operator_guard variant follows the packed
        # microbench planner instead: do not spend extra dense-fill work, and
        # fall back to dense unless the sparse-base branch has roughly the
        # K8/bs64 row count needed for a local 1.2x mixed operator.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = ""
        args.sr24_selective_residual_policy = "fixed_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_max_residual_draft_rows = 2
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 0
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_route_all_residual_rows = True
        args.sr24_route_all_skip_bucket = True
        args.sr24_direct_cpu_route_rows = False
        args.sr24_row_routed_mlp = True
        args.sr24_fixed_prefix_route_descriptor_only = True
        args.sr24_fixed_block_input_buffer = True
        args.sr24_row_routed_mlp_min_dense_rows = (
            0 if preset == "lossy_prefix2_rowrouted_mlp_operator_guard" else 32
        )
        args.sr24_route_contiguous_fastpath = True
        args.sr24_route_min_base_rows = (
            384 if preset == "lossy_prefix2_rowrouted_mlp_operator_guard" else 16
        )
        args.sr24_route_max_dense_fraction = 0.75
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset in {
        "gateup_res16_25_base26_31_critical4",
        "gateup_res16_25_base26_31_critical4_smallrow160",
    }:
        # Low-storage lossy candidate matching the throughput runner preset.
        # It keeps dense residual only for gate_up layers 16-25 and leaves
        # layers 26-31 sparse-only, reducing storage pressure while staying
        # within the historical GSM8K/Minerva 8pp quality budget.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = "gate_up_proj=26-31"
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-25"
        args.sr24_selective_residual_policy = "critical_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.7
        args.sr24_selective_extra_after_low = 4
        args.sr24_selective_min_prefix_residual = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = True
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 0
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        if preset == "gateup_res16_25_base26_31_critical4_smallrow160":
            args.sr24_adaptive_dense_fallback = True
            args.sr24_adaptive_dense_fallback_no_residual_only = True
            args.sr24_adaptive_dense_fallback_small_rows = 160
            args.sr24_adaptive_dense_fallback_small_gate_up_no_residual = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "fixedprefix4_bucket16_directcslt":
        # Middle ground between the quality-safe critical_prefix row and the
        # over-aggressive riskcap2 row: protect the first four draft rows, then
        # let later draft rows use the 2:4 base path. This tests whether the
        # dynamic first-low tail protection is actually needed for paired GSM8K
        # quality.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = (
            "gate_up_proj=16-31;down_proj=8-15"
        )
        args.sr24_selective_residual_policy = "fixed_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 16
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_route_all_residual_rows = True
        args.sr24_direct_cpu_route_rows = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "fixedprefix4_all_rowrouted_graph":
        # Graph-friendly systems probe matching the throughput runner:
        # protect a fixed prefix plus all non-draft rows, avoid direct CPU
        # route rows, and use MLP-level row routing so exact dense rows do not
        # also pay sparse-base work.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = (
            "gate_up_proj=16-31;down_proj=16-31"
        )
        args.sr24_selective_residual_policy = "fixed_prefix"
        args.sr24_selective_non_draft_policy = "all"
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_max_residual_draft_rows = 4
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 0
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_route_all_residual_rows = True
        args.sr24_route_all_skip_bucket = True
        args.sr24_direct_cpu_route_rows = False
        args.sr24_row_routed_mlp = True
        args.sr24_row_routed_mlp_min_dense_rows = 1
        args.sr24_route_contiguous_fastpath = True
        args.sr24_route_dense_fallback_fraction = 0.9
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = False
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "down0_15_fixedprefix4_directcslt":
        # Current doc2 precision/speed candidate from the 2026-06-29 slowdown
        # pass. It protects only early down_proj layers and keeps gate_up dense;
        # gate_up tail base-only/repaired variants broke or slowed the focused
        # GSM8K regression probes.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "down_proj"
        args.sr24_residual_target_leafs = "down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "down_proj=0-15"
        args.sr24_selective_residual_policy = "fixed_prefix"
        args.sr24_selective_non_draft_policy = "all"
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_max_residual_draft_rows = 4
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 0
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_route_all_residual_rows = True
        args.sr24_route_all_skip_bucket = True
        args.sr24_direct_cpu_route_rows = False
        args.sr24_route_dense_fallback_fraction = 0.9
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "criticalprefix4_bucket16_directcslt":
        # Quality/throughput candidate from the SR24 slowdown pass. Keep this
        # preset aligned with the GuideLLM throughput runner: the quality gate
        # must validate the exact same residual routing that serving measures.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = (
            "gate_up_proj=16-31;down_proj=8-15"
        )
        args.sr24_selective_residual_policy = "critical_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.6
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_extra_after_low = 1
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 16
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "criticalprefix_extra2_gateup_scaledbucket":
        # Trace-driven quality-first candidate from the 2026-06-29 acceptance
        # analysis. Protect a mandatory four-row accepted-prefix guard plus two
        # rows after the first low-confidence draft token. The per-active
        # bucket must fit that prefix and the bonus row; otherwise accepted
        # prefix rows can be demoted back to base-only by bucket truncation.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "critical_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_extra_after_low = 2
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_low_confidence_cap_by_risk = False
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 5
        args.sr24_residual_bucket_scale_by_active = True
        args.sr24_residual_bucket_priority = True
        args.sr24_bonus_priority = 0.5
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = False
        args.sr24_default_vllm_compile = False
        args.sr24_cudagraph_bucket = False
        args.sr24_force_cudagraph_none_for_mixed = True
        args.sr24_dynamic_auto_cudagraph = False
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "accuracy_first":
        # Down-proj tail sparsity is not accuracy-safe on current GSM8K gates.
        # Keep the gate half dense and sparsify only the up half for the
        # default accuracy-first candidate; use accuracy_gate_only for the older
        # fully fused gate_up tail ablation.
        base_only_by_leaf = "gate_up_proj=31"
        args.sr24_gate_up_split = "up_sparse"
    elif preset == "accuracy_gate_only":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "torch_sparse"
        args.sr24_residual_device = "auto"
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "none"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = "gate_up_proj=31"
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = True
        args.sr24_static_mask_state = "no_residual"
        args.sr24_static_all_residual_dense_fastpath = False
        args.sr24_static_mask_buffer = True
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    elif preset == "accuracy_down_only":
        base_only_by_leaf = "down_proj=31"
    elif preset in {"lossy_static_tail", "throughput_aggressive"}:
        base_only_by_leaf = "gate_up_proj=31;down_proj=30-31"
    else:
        raise ValueError(f"Unknown SR24 preset: {preset}")

    args.sr24_backend = "torch_sparse"
    args.sr24_residual_backend = "torch_sparse"
    args.sr24_residual_device = "auto"
    args.sr24_target_leafs = "qkv_proj,o_proj,gate_up_proj,down_proj"
    args.sr24_residual_target_leafs = "qkv_proj,o_proj"
    args.sr24_base_only_layer_ids = ""
    args.sr24_base_only_layer_ids_by_leaf = base_only_by_leaf
    args.sr24_reduce_cpu_sync = True
    args.sr24_sync_mask_state = True
    args.sr24_static_mask_state = "all_residual"
    args.sr24_static_all_residual_dense_fastpath = True
    args.sr24_static_mask_buffer = True
    args.sr24_allow_cudagraph = True
    args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)


SR24_PRESET_OVERRIDE_OPTIONS = {
    "sr24_threshold": "--sr24-threshold",
    "sr24_prefix_threshold": "--sr24-prefix-threshold",
    "sr24_selective_residual_policy": "--sr24-selective-residual-policy",
    "sr24_selective_non_draft_policy": "--sr24-selective-non-draft-policy",
    "sr24_selective_dense_nonverify_layer_ids_by_leaf":
    "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
    "sr24_selective_dense_nonverify_max_rows":
    "--sr24-selective-dense-nonverify-max-rows",
    "sr24_selective_extra_after_low": "--sr24-selective-extra-after-low",
    "sr24_selective_min_prefix_residual": "--sr24-selective-min-prefix-residual",
    "sr24_selective_max_residual_draft_rows":
    "--sr24-selective-max-residual-draft-rows",
    "sr24_residual_bucket_size": "--sr24-residual-bucket-size",
    "sr24_residual_bucket_scale_by_active":
    "--sr24-residual-bucket-scale-by-active",
    "sr24_residual_bucket_priority": "--sr24-residual-bucket-priority",
    "sr24_route_all_residual_rows": "--sr24-route-all-residual-rows",
    "sr24_route_all_skip_bucket": "--sr24-route-all-skip-bucket",
    "sr24_route_reuse_base_output": "--sr24-route-reuse-base-output",
    "sr24_route_contiguous_fastpath": "--sr24-route-contiguous-fastpath",
    "sr24_fixed_prefix_route_descriptor_only":
    "--sr24-fixed-prefix-route-descriptor-only",
    "sr24_scheduler_policy_path": "--sr24-scheduler-policy-path",
    "sr24_scheduler_policy_near_full_tolerance":
    "--sr24-scheduler-policy-near-full-tolerance",
    "sr24_scheduler_policy_allow_serial_packed_parallel":
    "--sr24-scheduler-policy-allow-serial-packed-parallel",
    "sr24_fixed_block_capacity_padding":
    "--sr24-fixed-block-capacity-padding",
    "sr24_fixed_block_capacity_zero_dummy":
    "--sr24-fixed-block-capacity-zero-dummy",
    "sr24_scheduler_policy_dense_bypass":
    "--sr24-scheduler-policy-dense-bypass",
    "sr24_route_overlap_streams": "--sr24-route-overlap-streams",
    "sr24_row_routed_mlp_fixed_block_dense_fill":
    "--sr24-row-routed-mlp-fixed-block-dense-fill",
    "sr24_fixed_block_input_buffer": "--sr24-fixed-block-input-buffer",
    "sr24_fixed_block_output_buffer": "--sr24-fixed-block-output-buffer",
    "sr24_runtime_base_only_layer_ids_by_leaf":
    "--sr24-runtime-base-only-layer-ids-by-leaf",
    "sr24_route_dense_fallback_fraction":
    "--sr24-route-dense-fallback-fraction",
    "sr24_route_min_dense_rows": "--sr24-route-min-dense-rows",
    "sr24_route_min_base_rows": "--sr24-route-min-base-rows",
    "sr24_route_min_base_rows_by_leaf":
    "--sr24-route-min-base-rows-by-leaf",
    "sr24_route_max_dense_fraction": "--sr24-route-max-dense-fraction",
    "sr24_adaptive_dense_fallback": "--sr24-adaptive-dense-fallback",
    "sr24_adaptive_dense_fallback_no_residual_only":
    "--sr24-adaptive-dense-fallback-no-residual-only",
    "sr24_adaptive_dense_fallback_small_rows":
    "--sr24-adaptive-dense-fallback-small-rows",
    "sr24_adaptive_dense_fallback_gate_up_fraction":
    "--sr24-adaptive-dense-fallback-gate-up-fraction",
    "sr24_adaptive_dense_fallback_down_fraction":
    "--sr24-adaptive-dense-fallback-down-fraction",
    "sr24_adaptive_dense_fallback_small_down_no_residual":
    "--sr24-adaptive-dense-fallback-small-down-no-residual",
    "sr24_adaptive_dense_fallback_small_gate_up_no_residual":
    "--sr24-adaptive-dense-fallback-small-gate-up-no-residual",
    "sr24_triton_route_assembly": "--sr24-triton-route-assembly",
    "sr24_triton_bucket_override": "--sr24-triton-bucket-override",
    "sr24_triton_bucket_dense_gemm": "--sr24-triton-bucket-dense-gemm",
    "sr24_triton_bucket_scatter": "--sr24-triton-bucket-scatter",
    "sr24_bucket_dense_copy": "--sr24-bucket-dense-copy",
    "sr24_bucket_dense_copy_active_only": "--sr24-bucket-dense-copy-active-only",
    "sr24_bucket_dense_compute_active_only":
    "--sr24-bucket-dense-compute-active-only",
    "sr24_bucket_dense_active_mask_fused":
    "--sr24-bucket-dense-active-mask-fused",
    "sr24_default_vllm_compile": "--sr24-default-vllm-compile",
    "sr24_cudagraph_bucket": "--sr24-cudagraph-bucket",
    "sr24_force_cudagraph_none_for_mixed":
    "--sr24-force-cudagraph-none-for-mixed",
    "sr24_dynamic_auto_cudagraph": "--sr24-dynamic-auto-cudagraph",
    "sr24_noverify_dense_mlp_fastpath":
    "--sr24-noverify-dense-mlp-fastpath",
    "sr24_cslt_small_m_alg_id_enable":
    "--sr24-cslt-small-m-alg-id-enable",
    "sr24_cslt_small_m_threshold":
    "--sr24-cslt-small-m-threshold",
    "sr24_cslt_small_m_alg_id": "--sr24-cslt-small-m-alg-id",
    "sr24_cslt_small_m_threshold_by_leaf":
    "--sr24-cslt-small-m-threshold-by-leaf",
    "sr24_cslt_small_m_alg_id_by_leaf":
    "--sr24-cslt-small-m-alg-id-by-leaf",
    "sr24_disable_runtime_stats": "--sr24-disable-runtime-stats",
    "sr24_target_leafs": "--sr24-target-leafs",
    "sr24_residual_target_leafs": "--sr24-residual-target-leafs",
    "sr24_base_only_layer_ids": "--sr24-base-only-layer-ids",
    "sr24_base_only_layer_ids_by_leaf": "--sr24-base-only-layer-ids-by-leaf",
    "sr24_residual_layer_ids_by_leaf": "--sr24-residual-layer-ids-by-leaf",
}


def _option_was_explicit(argv: list[str], option: str) -> bool:
    options = [option]
    if option.startswith("--"):
        options.append(f"--no-{option[2:]}")
    return any(
        arg == candidate or arg.startswith(f"{candidate}=")
        for arg in argv
        for candidate in options
    )


def capture_sr24_preset_overrides(
    args: argparse.Namespace,
    argv: list[str],
) -> dict[str, Any]:
    return {
        attr: getattr(args, attr)
        for attr, option in SR24_PRESET_OVERRIDE_OPTIONS.items()
        if _option_was_explicit(argv, option)
    }


def restore_sr24_preset_overrides(
    args: argparse.Namespace,
    overrides: dict[str, Any],
) -> None:
    for attr, value in overrides.items():
        setattr(args, attr, value)


def validate_sr24_runtime_base_only_compile(args: argparse.Namespace) -> None:
    scope = str(args.sr24_runtime_base_only_layer_ids_by_leaf or "").strip()
    if not scope:
        return
    modes = expand_modes(args.mode)
    if not any(mode in SR24_RUNTIME_MODES for mode in modes):
        return
    if not args.sr24_default_vllm_compile:
        return
    raise ValueError(
        "--sr24-runtime-base-only-layer-ids-by-leaf is a diagnostic eager-only "
        "ablation. Do not combine it with --sr24-default-vllm-compile: vLLM's "
        "shared decoder-layer graph can specialize the runtime branch to the "
        "wrong layer and silently corrupt SR24 quality."
    )


def sr24_requires_enforce_eager(args: argparse.Namespace, mode: str) -> bool:
    if mode not in SR24_RUNTIME_MODES:
        return False
    if mode == "base_only_24" and args.sr24_backend == "torch_sparse":
        # PyTorch semi-structured sparse tensors are the real 2:4 base path,
        # but the default vLLM/Inductor compile path can trace into the sparse
        # custom kernel during startup profiling and crash on lazy storage.
        # Keep base-only accuracy rows eager until that sparse op is graph-safe.
        return True
    if (
        mode == "speclink_t08"
        and args.sr24_direct_cpu_route_rows
        and args.sr24_route_all_residual_rows
    ):
        # Direct CPU row-list routing is a correctness/diagnostic path today.
        # It matches the mask semantics in eager mode, but GSM8K gates showed
        # divergence under vLLM compile/CUDA-Graph replay. Do not let default
        # compile or dynamic graph flags silently corrupt accuracy baselines.
        return True
    if args.sr24_default_vllm_compile:
        return False
    if args.sr24_allow_cudagraph and mode in {
        "base_only_24",
        "all_corrected_24",
    }:
        return False
    if mode == "speclink_t08" and speclink_t08_allows_cudagraph(args):
        return False
    if mode == "all_corrected_24" and args.sr24_all_corrected_dense_fastpath:
        return False
    if (
        mode == "base_only_24"
        and args.sr24_backend in {"dense_zero", "prototype"}
    ):
        return False
    return True


def speclink_t08_allows_cudagraph(args: argparse.Namespace) -> bool:
    if args.sr24_direct_cpu_route_rows and args.sr24_route_all_residual_rows:
        return False
    # Dynamic selective masks are updated outside the model call and are read
    # through SR24 global/context state. In graph/compile mode this has shown
    # correctness drift on GSM8K replays even when the debug mask is
    # all-residual near the divergence. Keep CUDA Graph enabled only for
    # static-mask ablations whose residual state is invariant across steps.
    allowed_states = {"all_residual", "no_residual"}
    if not args.sr24_force_cudagraph_none_for_mixed:
        # Experimental graph-safety ablation: static mixed keeps the SR24
        # residual-mask pointer stable and only updates its contents before the
        # target forward. Dynamic "auto" remains eager.
        allowed_states.add("mixed")
    fixed_prefix_route_all_can_use_graph = (
        args.sr24_route_all_residual_rows
        and args.sr24_selective_residual_policy == "fixed_prefix"
        and args.sr24_selective_non_draft_policy in {"all", "bonus"}
        and not args.sr24_direct_cpu_route_rows
        and args.sr24_route_all_skip_bucket
        and int(args.sr24_residual_bucket_size) <= 0
    )
    dynamic_auto_can_use_graph = (
        args.sr24_dynamic_auto_cudagraph
        and args.sr24_static_mask_state == "auto"
        and not args.sr24_force_cudagraph_none_for_mixed
        and not args.sr24_residual_bucket_scale_by_active
        and (
            not args.sr24_route_all_residual_rows
            or fixed_prefix_route_all_can_use_graph
        )
        and (
            int(args.sr24_residual_bucket_size) <= 0
            or args.sr24_cudagraph_bucket
        )
    )
    if (
        args.sr24_static_mask_state not in allowed_states
        and not dynamic_auto_can_use_graph
    ):
        return False
    return (
        args.sr24_allow_cudagraph
        and args.sr24_static_mask_buffer
        and args.sr24_reduce_cpu_sync
        and args.sr24_residual_backend in {"torch_sparse", "dense_rows"}
    )


def dense_baseline_aligns_to_sr24_eager(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "_dense_baseline_align_sr24_eager", False))


def effective_vllm_compilation_config(args: argparse.Namespace, mode: str) -> str:
    if args.vllm_compilation_config:
        return args.vllm_compilation_config
    if args.sr24_default_vllm_compile:
        return ""
    if (
        not args.sr24_allow_cudagraph
        or mode not in {"base_only_24", "speclink_t08"}
        or args.sr24_backend != "torch_sparse"
    ):
        return ""
    if mode == "speclink_t08" and not speclink_t08_allows_cudagraph(args):
        return ""
    verifier_tokens = args.max_num_seqs * (args.num_spec_tokens + 1)
    capture_size = max(1024, int(math.ceil(verifier_tokens / 16.0) * 16))
    return json.dumps(
        {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_cudagraph_capture_size": capture_size,
        },
        separators=(",", ":"),
    )


def configure_sr24_env(
    args: argparse.Namespace,
    *,
    mode: str,
    task: str,
    model_label: str,
    run_dir: Path,
) -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
    env["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["SPECLINK_SR24_ENABLE"] = "1"
    runtime_mode = sr24_runtime_mode(mode)
    is_all_corrected = runtime_mode == "all_corrected"
    env["SPECLINK_SR24_MODE"] = runtime_mode
    (
        residual_out_chunk,
        cache_compressed_residual_weight,
        prewarm_compressed_residual_weight,
        _auto_compressed_residual_fastpath,
    ) = sr24_compressed_residual_runtime_settings(args, mode)
    env["SPECLINK_SR24_BACKEND"] = args.sr24_backend
    env["SPECLINK_SR24_RESIDUAL_BACKEND"] = args.sr24_residual_backend
    env["SPECLINK_SR24_RESIDUAL_DEVICE"] = args.sr24_residual_device
    env["SPECLINK_SR24_REQUIRE_GPU_RESIDUAL"] = (
        "1" if args.sr24_require_gpu_residual else "0"
    )
    env["SPECLINK_SR24_THRESHOLD"] = str(args.sr24_threshold)
    env["SPECLINK_SR24_REDUCE_CPU_SYNC"] = (
        "1" if args.sr24_reduce_cpu_sync else "0"
    )
    env["SPECLINK_SR24_SYNC_MASK_STATE"] = (
        "1" if args.sr24_sync_mask_state else "0"
    )
    env["SPECLINK_SR24_STATIC_MASK_STATE"] = args.sr24_static_mask_state
    env["SPECLINK_SR24_STATIC_ALL_RESIDUAL_DENSE_FASTPATH"] = (
        "1" if args.sr24_static_all_residual_dense_fastpath else "0"
    )
    env["SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH"] = (
        "1" if args.sr24_all_corrected_dense_fastpath else "0"
    )
    env["SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_MAX_ROWS"] = str(
        args.sr24_base_only_dense_verify_max_rows
    )
    env["SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_LAYER_IDS"] = (
        args.sr24_base_only_dense_verify_layer_ids
    )
    env["SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_LAYER_IDS_BY_LEAF"] = (
        args.sr24_base_only_dense_verify_layer_ids_by_leaf
    )
    env["SPECLINK_SR24_RUNTIME_BASE_ONLY_LAYER_IDS_BY_LEAF"] = (
        args.sr24_runtime_base_only_layer_ids_by_leaf
    )
    env["SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE"] = (
        "1" if args.sr24_full_residual_early_dense else "0"
    )
    env["SPECLINK_SR24_NOVERIFY_DENSE_MLP_FASTPATH"] = (
        "1" if args.sr24_noverify_dense_mlp_fastpath else "0"
    )
    env["SPECLINK_SR24_STATIC_MASK_BUFFER"] = (
        "1" if args.sr24_static_mask_buffer else "0"
    )
    env["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = (
        "1" if args.sr24_batched_mask_builder else "0"
    )
    env["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = (
        "1" if args.sr24_batched_uniform_direct else "0"
    )
    env["SPECLINK_SR24_GPU_COUNT_MASK_BUILDER"] = (
        "1" if args.sr24_gpu_count_mask_builder else "0"
    )
    env["SPECLINK_SR24_DIRECT_CSLT_LINEAR"] = (
        "1" if sr24_effective_direct_cslt_linear(args, mode) else "0"
    )
    env["SPECLINK_SR24_CSLT_SMALL_M_ALG_ID_ENABLE"] = (
        "1" if args.sr24_cslt_small_m_alg_id_enable else "0"
    )
    env["SPECLINK_SR24_CSLT_SMALL_M_THRESHOLD"] = str(
        args.sr24_cslt_small_m_threshold
    )
    env["SPECLINK_SR24_CSLT_SMALL_M_ALG_ID"] = str(
        args.sr24_cslt_small_m_alg_id
    )
    env["SPECLINK_SR24_CSLT_SMALL_M_THRESHOLD_BY_LEAF"] = (
        args.sr24_cslt_small_m_threshold_by_leaf
    )
    env["SPECLINK_SR24_CSLT_SMALL_M_ALG_ID_BY_LEAF"] = (
        args.sr24_cslt_small_m_alg_id_by_leaf
    )
    env["SPECLINK_SR24_GATE_UP_SPLIT"] = args.sr24_gate_up_split
    env["SPECLINK_SR24_GATE_UP_CHANNEL_DENSE_FRACTION"] = str(
        args.sr24_gate_up_channel_dense_fraction
    )
    env["SPECLINK_SR24_GATE_UP_CHANNEL_STRATEGY"] = (
        args.sr24_gate_up_channel_strategy
    )
    env["SPECLINK_SR24_GATE_UP_CHANNEL_FUSED_ACT"] = (
        "1" if args.sr24_gate_up_channel_fused_act else "0"
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP"] = (
        "1" if args.sr24_row_routed_mlp else "0"
    )
    env["SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR"] = (
        "1" if args.sr24_row_routed_down_linear else "0"
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT"] = (
        "1" if args.sr24_row_routed_mlp_reuse_base_output else "0"
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_FIXED_BLOCK_DENSE_FILL"] = (
        "1" if args.sr24_row_routed_mlp_fixed_block_dense_fill else "0"
    )
    env["SPECLINK_SR24_FIXED_BLOCK_INPUT_BUFFER"] = (
        "1" if args.sr24_fixed_block_input_buffer else "0"
    )
    env["SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER"] = (
        "1" if args.sr24_fixed_block_output_buffer else "0"
    )
    env["SPECLINK_SR24_SCHEDULER_POLICY_PATH"] = (
        args.sr24_scheduler_policy_path or ""
    )
    env["SPECLINK_SR24_SCHEDULER_POLICY_NEAR_FULL_TOLERANCE"] = str(
        args.sr24_scheduler_policy_near_full_tolerance
    )
    env["SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_SERIAL_PACKED_PARALLEL"] = (
        "1" if args.sr24_scheduler_policy_allow_serial_packed_parallel else "0"
    )
    env["SPECLINK_SR24_FIXED_BLOCK_CAPACITY_PADDING"] = (
        "1" if args.sr24_fixed_block_capacity_padding else "0"
    )
    env["SPECLINK_SR24_FIXED_BLOCK_CAPACITY_ZERO_DUMMY"] = (
        "1" if args.sr24_fixed_block_capacity_zero_dummy else "0"
    )
    env["SPECLINK_SR24_SCHEDULER_POLICY_DENSE_BYPASS"] = (
        "1" if args.sr24_scheduler_policy_dense_bypass else "0"
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS"] = str(
        args.sr24_row_routed_mlp_min_dense_rows
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS_BY_LEAF"] = (
        args.sr24_row_routed_mlp_min_dense_rows_by_leaf
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS"] = str(
        args.sr24_row_routed_mlp_max_dense_rows
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS_BY_LEAF"] = (
        args.sr24_row_routed_mlp_max_dense_rows_by_leaf
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS"] = str(
        args.sr24_row_routed_mlp_max_base_rows
    )
    env["SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS_BY_LEAF"] = (
        args.sr24_row_routed_mlp_max_base_rows_by_leaf
    )
    env["SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED"] = (
        "1" if args.sr24_force_cudagraph_none_for_mixed else "0"
    )
    env["SPECLINK_SR24_MASK_BUFFER_CAPACITY"] = str(
        args.sr24_mask_buffer_capacity
    )
    env["SPECLINK_SR24_RESIDUAL_BUCKET_SIZE"] = str(
        args.sr24_residual_bucket_size
    )
    env["SPECLINK_SR24_RESIDUAL_BUCKET_SCALE_BY_ACTIVE"] = (
        "1" if args.sr24_residual_bucket_scale_by_active else "0"
    )
    env["SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY"] = (
        "1" if args.sr24_residual_bucket_priority else "0"
    )
    env["SPECLINK_SR24_BONUS_PRIORITY"] = str(args.sr24_bonus_priority)
    env["SPECLINK_SR24_DRAFT_POSITION_PRIORITY_SCALE"] = str(
        args.sr24_draft_position_priority_scale
    )
    env["SPECLINK_SR24_CUDAGRAPH_BUCKET"] = (
        "1" if args.sr24_cudagraph_bucket else "0"
    )
    env["SPECLINK_SR24_ROUTE_BUCKET_ROWS"] = (
        "1" if args.sr24_route_bucket_rows else "0"
    )
    env["SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS"] = (
        "1" if args.sr24_route_all_residual_rows else "0"
    )
    env["SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET"] = (
        "1" if args.sr24_route_all_skip_bucket else "0"
    )
    env["SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS"] = (
        "1" if args.sr24_direct_cpu_route_rows else "0"
    )
    env["SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT"] = (
        "1" if args.sr24_route_reuse_base_output else "0"
    )
    env["SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH"] = (
        "1" if args.sr24_route_contiguous_fastpath else "0"
    )
    env["SPECLINK_SR24_FIXED_PREFIX_ROUTE_DESCRIPTOR_ONLY"] = (
        "1" if args.sr24_fixed_prefix_route_descriptor_only else "0"
    )
    env["SPECLINK_SR24_ROUTE_OVERLAP_STREAMS"] = (
        "1" if args.sr24_route_overlap_streams else "0"
    )
    env["SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION"] = str(
        args.sr24_route_dense_fallback_fraction
    )
    env["SPECLINK_SR24_ROUTE_MIN_DENSE_ROWS"] = str(
        args.sr24_route_min_dense_rows
    )
    env["SPECLINK_SR24_ROUTE_MIN_BASE_ROWS"] = str(
        args.sr24_route_min_base_rows
    )
    env["SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"] = (
        args.sr24_route_min_base_rows_by_leaf
    )
    env["SPECLINK_SR24_ROUTE_MAX_DENSE_FRACTION"] = str(
        args.sr24_route_max_dense_fraction
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK"] = (
        "1" if args.sr24_adaptive_dense_fallback else "0"
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_NO_RESIDUAL_ONLY"] = (
        "1" if args.sr24_adaptive_dense_fallback_no_residual_only else "0"
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS"] = str(
        args.sr24_adaptive_dense_fallback_small_rows
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_GATE_UP_FRACTION"] = str(
        args.sr24_adaptive_dense_fallback_gate_up_fraction
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_DOWN_FRACTION"] = str(
        args.sr24_adaptive_dense_fallback_down_fraction
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL"] = (
        "1" if args.sr24_adaptive_dense_fallback_small_down_no_residual else "0"
    )
    env["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_GATE_UP_NO_RESIDUAL"] = (
        "1" if args.sr24_adaptive_dense_fallback_small_gate_up_no_residual else "0"
    )
    env["SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY"] = (
        "1" if args.sr24_triton_route_assembly else "0"
    )
    env["SPECLINK_SR24_TRITON_BUCKET_OVERRIDE"] = (
        "1" if args.sr24_triton_bucket_override else "0"
    )
    env["SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM"] = (
        "1" if args.sr24_triton_bucket_dense_gemm else "0"
    )
    env["SPECLINK_SR24_TRITON_BUCKET_SCATTER"] = (
        "1" if args.sr24_triton_bucket_scatter else "0"
    )
    env["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_M"] = str(
        args.sr24_triton_bucket_dense_block_m
    )
    env["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_N"] = str(
        args.sr24_triton_bucket_dense_block_n
    )
    env["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_K"] = str(
        args.sr24_triton_bucket_dense_block_k
    )
    env["SPECLINK_SR24_BUCKET_DENSE_COPY"] = (
        "1" if args.sr24_bucket_dense_copy else "0"
    )
    env["SPECLINK_SR24_BUCKET_DENSE_COPY_ACTIVE_ONLY"] = (
        "1" if args.sr24_bucket_dense_copy_active_only else "0"
    )
    env["SPECLINK_SR24_BUCKET_DENSE_COMPUTE_ACTIVE_ONLY"] = (
        "1" if args.sr24_bucket_dense_compute_active_only else "0"
    )
    env["SPECLINK_SR24_BUCKET_DENSE_ACTIVE_MASK_FUSED"] = (
        "1" if args.sr24_bucket_dense_active_mask_fused else "0"
    )
    env["SPECLINK_SR24_DISABLE_RUNTIME_STATS"] = (
        "1" if args.sr24_disable_runtime_stats else "0"
    )
    env["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = (
        "1" if args.sr24_selective_correct_non_draft else "0"
    )
    env["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = (
        args.sr24_selective_non_draft_policy
    )
    env["SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_LAYER_IDS_BY_LEAF"] = (
        args.sr24_selective_dense_nonverify_layer_ids_by_leaf
    )
    env["SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_MAX_ROWS"] = str(
        args.sr24_selective_dense_nonverify_max_rows
    )
    env["SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_STATIC_ROWS"] = str(
        args.max_num_seqs
    )
    env["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = (
        args.sr24_selective_residual_policy
    )
    env["SPECLINK_SR24_PREFIX_THRESHOLD"] = (
        "" if args.sr24_prefix_threshold < 0 else str(args.sr24_prefix_threshold)
    )
    env["SPECLINK_SR24_SELECTIVE_EXTRA_AFTER_LOW"] = str(
        args.sr24_selective_extra_after_low
    )
    env["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = str(
        args.sr24_selective_min_prefix_residual
    )
    env["SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS"] = str(
        args.sr24_selective_max_residual_draft_rows
    )
    env["SPECLINK_SR24_LOW_CONFIDENCE_CAP_BY_RISK"] = (
        "1" if args.sr24_low_confidence_cap_by_risk else "0"
    )
    env["SPECLINK_SR24_EARLY_DENSE_TOKENS"] = str(args.sr24_early_dense_tokens)
    if args.sr24_target_leafs:
        env["SPECLINK_SR24_TARGET_LEAFS"] = args.sr24_target_leafs
    else:
        env.pop("SPECLINK_SR24_TARGET_LEAFS", None)
    if is_all_corrected and args.sr24_target_leafs:
        env["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = args.sr24_target_leafs
    elif is_all_corrected:
        env.pop("SPECLINK_SR24_RESIDUAL_TARGET_LEAFS", None)
    elif args.sr24_residual_target_leafs:
        env["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = (
            args.sr24_residual_target_leafs
        )
    else:
        env.pop("SPECLINK_SR24_RESIDUAL_TARGET_LEAFS", None)
    if is_all_corrected:
        env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS", None)
        env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", None)
        env.pop("SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF", None)
    elif args.sr24_base_only_layer_ids:
        env["SPECLINK_SR24_BASE_ONLY_LAYER_IDS"] = args.sr24_base_only_layer_ids
        if args.sr24_base_only_layer_ids_by_leaf:
            env["SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF"] = (
                args.sr24_base_only_layer_ids_by_leaf
            )
        else:
            env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", None)
        if args.sr24_residual_layer_ids_by_leaf:
            env["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = (
                args.sr24_residual_layer_ids_by_leaf
            )
        else:
            env.pop("SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF", None)
    else:
        env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS", None)
        if args.sr24_base_only_layer_ids_by_leaf:
            env["SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF"] = (
                args.sr24_base_only_layer_ids_by_leaf
            )
        else:
            env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", None)
        if args.sr24_residual_layer_ids_by_leaf:
            env["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = (
                args.sr24_residual_layer_ids_by_leaf
            )
        else:
            env.pop("SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF", None)
    env["SPECLINK_SR24_RESIDUAL_OUT_CHUNK"] = str(residual_out_chunk)
    env["SPECLINK_SR24_CACHE_COMPRESSED_RESIDUAL_WEIGHT"] = (
        "1" if cache_compressed_residual_weight else "0"
    )
    env["SPECLINK_SR24_PREWARM_COMPRESSED_RESIDUAL_WEIGHT"] = (
        "1" if prewarm_compressed_residual_weight else "0"
    )
    env["SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON"] = (
        "1" if args.sr24_compressed_residual_triton else "0"
    )
    env["SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_M"] = str(
        args.sr24_compressed_residual_block_m
    )
    env["SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_N"] = str(
        args.sr24_compressed_residual_block_n
    )
    env["SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_G"] = str(
        args.sr24_compressed_residual_block_g
    )
    env["SPECLINK_SR24_EXTRACT_CHUNK_ROWS"] = str(args.sr24_extract_chunk_rows)
    env["SPECLINK_SR24_LOG"] = str((run_dir / "speclink_sr24_events.jsonl").resolve())
    env["SPECLINK_SR24_STATS_PATH"] = str(
        (run_dir / "speclink_sr24_stats.json").resolve()
    )
    env["SPECLINK_SR24_STATS_INTERVAL"] = str(args.sr24_stats_interval)
    sr24_breakdown_path = run_dir / "speclink_sr24_breakdown.json"
    env["SPECLINK_SR24_BREAKDOWN"] = "1" if args.sr24_breakdown else "0"
    env["SPECLINK_SR24_BREAKDOWN_PATH"] = str(sr24_breakdown_path.resolve())
    env["SPECLINK_SR24_BREAKDOWN_INTERVAL"] = str(args.sr24_breakdown_interval)
    env["SPECLINK_SR24_BREAKDOWN_LINEAR"] = (
        "1" if args.sr24_breakdown_linear else "0"
    )
    env["SPECLINK_SR24_BREAKDOWN_EXACT_ROUTING"] = (
        "1" if args.sr24_breakdown_exact_routing else "0"
    )
    env["SPECLINK_SR24_BREAKDOWN_SYNC_COUNTS"] = (
        "1" if args.sr24_breakdown_exact_routing else "0"
    )
    env["SPECLINK_SR24_BREAKDOWN_GPU_COUNTS"] = (
        "1" if args.sr24_breakdown_gpu_counts else "0"
    )
    mask_path = args.sr24_mask_path
    if mask_path is None and not args.sr24_disable_default_mask:
        default_mask = DEFAULT_SR24_MASKS.get(model_label)
        if default_mask is not None and default_mask.exists():
            mask_path = default_mask
    if mask_path:
        env["SPECLINK_SR24_MASK_PATH"] = str(mask_path.resolve())
    configure_confidence_trace_env(
        args,
        env=env,
        mode=mode,
        task=task,
        model_label=model_label,
        run_dir=run_dir,
    )
    return env


def disable_speclink_env(env: dict[str, str]) -> None:
    env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
    env["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    env["SPECLINK_SR24_ENABLE"] = "0"


def configure_confidence_trace_env(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    mode: str,
    task: str,
    model_label: str,
    run_dir: Path,
) -> None:
    """Enable confidence/acceptance trace only for explicit diagnostics."""
    if not args.trace_confidence or not mode_uses_spec(mode):
        env.pop("SPECLINK_TRACE_CONFIDENCE", None)
        return
    trace_path = run_dir / "speclink_confidence_trace.jsonl"
    env["SPECLINK_TRACE_CONFIDENCE"] = "1"
    env["SPECLINK_TRACE_OUTPUT"] = str(trace_path.resolve())
    env["SPECLINK_TRACE_RUN_ID"] = f"{model_label}_{mode}_{task}"
    env["SPECLINK_TRACE_MODEL_LABEL"] = model_label
    env["SPECLINK_TRACE_DATASET_LABEL"] = task
    env["SPECLINK_TRACE_METHOD"] = mode
    env["SPECLINK_TRACE_NUM_SPEC_TOKENS"] = str(args.num_spec_tokens)


def configure_eval_cache_env(args: argparse.Namespace, env: dict[str, str]) -> Path:
    cache_root = args.hf_cache_root.resolve()
    for child in ("home", "hf_home", "datasets", "evaluate", "tmp"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)
    if not env.get("HF_TOKEN"):
        for token_path in (
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ):
            if token_path.exists():
                token = token_path.read_text(encoding="utf-8").strip()
                if token:
                    env["HF_TOKEN"] = token
                    break
    env["HOME"] = str(cache_root / "home")
    env["HF_HOME"] = str(cache_root / "hf_home")
    env["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    env["HF_EVALUATE_CACHE"] = str(cache_root / "evaluate")
    env["EVALUATE_CACHE"] = str(cache_root / "evaluate")
    env["TMPDIR"] = str(cache_root / "tmp")
    return cache_root


def build_bwrap_command(command: list[str], *, run_dir: Path, cache_root: Path) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("HumanEval sandbox requested but bwrap is not installed")
    run_dir = run_dir.resolve()
    cache_root = cache_root.resolve()
    return [
        bwrap,
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(run_dir),
        str(run_dir),
        "--bind",
        str(cache_root),
        str(cache_root),
        "--setenv",
        "HOME",
        str(cache_root / "home"),
        "--setenv",
        "HF_HOME",
        str(cache_root / "hf_home"),
        "--setenv",
        "HF_DATASETS_CACHE",
        str(cache_root / "datasets"),
        "--setenv",
        "HF_EVALUATE_CACHE",
        str(cache_root / "evaluate"),
        "--setenv",
        "EVALUATE_CACHE",
        str(cache_root / "evaluate"),
        "--setenv",
        "TMPDIR",
        "/tmp",
        *command,
    ]


def build_vllm_command(
    args: argparse.Namespace,
    *,
    mode: str,
    model_path: str,
    speculator_path: str,
    port: int,
) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "spec",
        "vllm",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--seed",
        str(args.seed),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_context_length),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--generation-config",
        "vllm",
    ]
    if args.vllm_dtype:
        command.extend(["--dtype", args.vllm_dtype])
    compilation_config = effective_vllm_compilation_config(args, mode)
    if compilation_config:
        command.extend(["--compilation-config", compilation_config])
    if mode_uses_spec(mode):
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "model": speculator_path,
                        "num_speculative_tokens": args.num_spec_tokens,
                        "method": "eagle3",
                        "max_model_len": args.max_context_length,
                    }
                ),
            ]
        )
    if (
        args.enforce_eager
        or (
            mode == "dense_baseline"
            and dense_baseline_aligns_to_sr24_eager(args)
        )
        or mode.startswith("token_dense_")
        or sr24_requires_enforce_eager(args, mode)
    ):
        command.append("--enforce-eager")
    return command


def start_server(
    args: argparse.Namespace,
    *,
    mode: str,
    task: str,
    model_label: str,
    model_path: str,
    speculator_path: str,
    run_dir: Path,
) -> tuple[subprocess.Popen[Any], int]:
    port = find_free_port(args.port_base)
    method = mode_method(mode)
    stats_path = run_dir / "vllm_structured_24_stats.json"
    if mode in SR24_RUNTIME_MODES:
        env = configure_sr24_env(
            args,
            mode=mode,
            task=task,
            model_label=model_label,
            run_dir=run_dir,
        )
        sr24_cache_root = ""
        # SR24 uses env-gated Python branches that vLLM's default compile cache
        # key does not include. Always isolate SR24 graphs, including
        # --sr24-default-vllm-compile, so dense/eagle3 baselines never replay a
        # graph captured with SR24-only attributes.
        cache_root = sr24_compile_cache_root(args, mode)
        env["VLLM_CACHE_ROOT"] = str(cache_root)
        sr24_cache_root = str(cache_root)
        vllm_cache_root = str(cache_root)
    else:
        env = method_env(args, model_label=model_label, method=method, stats_path=stats_path)
        if mode in {"dense_ar", "eagle3_dense", "dense_baseline"}:
            disable_speclink_env(env)
        configure_confidence_trace_env(
            args,
            env=env,
            mode=mode,
            task=task,
            model_label=model_label,
            run_dir=run_dir,
        )
        sr24_cache_root = ""
        cache_root = non_sr24_compile_cache_root(args, mode)
        env["VLLM_CACHE_ROOT"] = str(cache_root)
        vllm_cache_root = str(cache_root)
    command = build_vllm_command(
        args,
        mode=mode,
        model_path=model_path,
        speculator_path=speculator_path,
        port=port,
    )
    write_json(
        run_dir / "server_command.json",
        {
            "command": command,
            "port": port,
            "speclink_env": {
                key: value
                for key, value in sorted(env.items())
                if key.startswith("SPECLINK_")
            },
            "sr24_compile_cache_root": sr24_cache_root,
            "vllm_compile_cache_root": vllm_cache_root,
        },
    )
    log = (run_dir / "vllm_server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(EVAL_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    wait_for_health(port, process, args.health_timeout_s)
    return process, port


def model_args_string(
    *,
    port: int,
    model_path: str,
    tokenizer_path: str,
    args: argparse.Namespace,
    max_gen_toks: int,
) -> str:
    values = {
        "model": model_path,
        "base_url": f"http://127.0.0.1:{port}/v1/completions",
        "tokenizer": tokenizer_path,
        "tokenizer_backend": "huggingface",
        "max_length": args.max_context_length,
        "max_gen_toks": max_gen_toks,
        "num_concurrent": args.num_concurrent,
        "seed": args.seed,
        "timeout": args.request_timeout_s,
        "tokenized_requests": True,
        "trust_remote_code": args.trust_remote_code,
    }
    return ",".join(f"{key}={value}" for key, value in values.items())


def run_lm_eval(
    args: argparse.Namespace,
    *,
    task: str,
    mode: str,
    model_path: str,
    tokenizer_path: str,
    port: int,
    run_dir: Path,
) -> int:
    max_gen_toks = args.max_new_tokens or TASK_MAX_TOKENS.get(task, 512)
    manifest_path, samples_json, manifest_count, manifest_note = ensure_task_manifest(
        args, task
    )
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "spec",
        "lm-eval",
        "run",
        "--model",
        "local-completions",
        "--model_args",
        model_args_string(
            port=port,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            args=args,
            max_gen_toks=max_gen_toks,
        ),
        "--tasks",
        task,
        "--batch_size",
        str(args.batch_size),
        "--gen_kwargs",
        "temperature=0",
        "top_p=1",
        "do_sample=False",
        f"max_gen_toks={max_gen_toks}",
        "--seed",
        str(args.seed),
        "--include_path",
        str(LOCAL_TASKS),
        "--output_path",
        str(run_dir / "lm_eval_output"),
        "--log_samples",
    ]
    if samples_json:
        command.extend(["--samples", samples_json])
    elif args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.apply_chat_template:
        command.append("--apply_chat_template")
    if task == "humaneval_instruct" and args.allow_unsafe_code:
        command.append("--confirm_run_unsafe_code")
    command_to_run = command
    env = add_local_no_proxy(os.environ.copy())
    env["OPENAI_API_KEY"] = "EMPTY"
    cache_root = configure_eval_cache_env(args, env)
    if task == "humaneval_instruct" and args.allow_unsafe_code:
        env["HF_ALLOW_CODE_EVAL"] = "1"
        if args.humaneval_sandbox in {"auto", "bwrap"}:
            command_to_run = build_bwrap_command(
                command,
                run_dir=run_dir,
                cache_root=cache_root,
            )
    write_json(
        run_dir / "lm_eval_command.json",
        {
            "command": command,
            "command_to_run": command_to_run,
            "manifest_path": str(manifest_path) if manifest_path else "",
            "manifest_count": manifest_count,
            "manifest_note": manifest_note,
            "max_gen_toks": max_gen_toks,
            "humaneval_sandbox": (
                args.humaneval_sandbox if task == "humaneval_instruct" else ""
            ),
            "cache_root": str(cache_root),
        },
    )
    with (run_dir / "lm_eval.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command_to_run,
            cwd=str(EVAL_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def should_skip_task(args: argparse.Namespace, task: str) -> str:
    if task == "humaneval_instruct" and not args.allow_unsafe_code:
        return (
            "HumanEval requires executing generated code. Skipped because "
            "--allow-unsafe-code was not set and no isolated execution backend "
            "was requested."
        )
    return ""


def write_skip(run_dir: Path, *, reason: str, task: str, mode: str, model_label: str) -> None:
    write_json(
        run_dir / "skip.json",
        {
            "status": "skipped",
            "reason": reason,
            "task": task,
            "mode": mode,
            "model_label": model_label,
            "created_at": timestamp(),
        },
    )


def completed_run(run_dir: Path) -> bool:
    if (run_dir / "skip.json").exists():
        return True
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("status") != "ok":
        return False
    mode = str(meta.get("mode") or "")
    if mode.startswith("token_dense_") and not token_dense_stats_valid(run_dir):
        return False
    if mode in SR24_RUNTIME_MODES and not sr24_stats_valid(run_dir):
        return False
    return bool(list((run_dir / "lm_eval_output").rglob("results_*.json")))


def token_dense_stats_valid(run_dir: Path) -> bool:
    stats_path = run_dir / "token_dense_stats.jsonl"
    if not stats_path.exists() or stats_path.stat().st_size <= 0:
        return False
    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("total_draft_tokens") or 0) > 0:
                return True
    return False


def sr24_stats_valid(run_dir: Path) -> bool:
    stats_path = run_dir / "speclink_sr24_stats.json"
    if not stats_path.exists() or stats_path.stat().st_size <= 0:
        return False
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(stats.get("module_count_attached") or 0) <= 0:
        return False
    if stats.get("dense_fastpath_noop") is True:
        return True
    if stats.get("linear_hooks_enabled") is False:
        return True
    if stats.get("runtime_stats_enabled") is False:
        return True
    event_path = run_dir / "speclink_sr24_events.jsonl"
    if not event_path.exists() or event_path.stat().st_size <= 0:
        return False
    with event_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") in {"sr24_verify_mask", "sr24_verify_summary"}:
                return True
    return False


def prepare_run_dir_for_rerun(run_dir: Path) -> None:
    for child in (
        "lm_eval_output",
        "lm_eval.log",
        "lm_eval_command.json",
        "run_meta.json",
        "server_command.json",
        "speclink_confidence_trace.jsonl",
        "task_preflight.json",
        "task_preflight.log",
        "token_dense_stats.jsonl",
        "speclink_sr24_events.jsonl",
        "speclink_sr24_stats.json",
        "speclink_sr24_breakdown.json",
        "vllm_server.log",
        "vllm_structured_24_stats.json",
        "skip.json",
    ):
        path = run_dir / child
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def write_environment(output_dir: Path) -> None:
    env_path = output_dir / "environment.txt"
    spec_python = Path(sys.executable).resolve()
    spec_bin = spec_python.parent
    lm_eval_bin = spec_bin / "lm-eval"
    commands = [
        [str(spec_python), "-m", "pip", "check"],
        [str(spec_python), "-c", "import sys; print(sys.version)"],
        [
            str(spec_python),
            "-c",
            "import importlib.metadata as m; "
            "print('lm-eval', m.version('lm-eval')); "
            "print('torch', m.version('torch')); "
            "print('vllm', m.version('vllm')); "
            "print('guidellm', m.version('guidellm'))",
        ],
        [str(lm_eval_bin), "--help"],
        [str(lm_eval_bin), "ls", "tasks"],
        [str(lm_eval_bin), "ls", "groups"],
    ]
    with env_path.open("w", encoding="utf-8") as handle:
        handle.write("# lm-eval environment\n\n")
        for command in commands:
            handle.write(f"$ {shlex.join(command)}\n")
            completed = subprocess.run(
                command,
                cwd=str(EVAL_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            handle.write(completed.stdout)
            handle.write("\n")


def ensure_sr24_iteration_logs(output_dir: Path, modes: list[str]) -> None:
    if not any(mode in SR24_METHODS for mode in modes):
        return
    csv_path = output_dir / "iteration_log.csv"
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SR24_ITERATION_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "version": "stage1_correctness_proto",
                    "git_diff_or_commit": "working_tree",
                    "change_description": (
                        "SR24 env-gated base/residual split with selectable "
                        "base and residual backends"
                    ),
                    "hypothesis": (
                        "Validate lossless base/residual split, then separate "
                        "real base 2:4 sparse execution from residual correction"
                    ),
                    "exact_command": shlex.join(sys.argv),
                    "kept_or_reverted": "kept",
                    "reason": "initial SR24 scaffold",
                }
            )
    md_path = output_dir / "iteration_log.md"
    if not md_path.exists():
        md_path.write_text(
            "# SpecLink SR24 Iteration Log\n\n"
            "Append one row to `iteration_log.csv` for every main mechanism "
            "change. Stage 1 is correctness-only: it must not be reported as a "
            "real NVIDIA 2:4 Sparse Tensor Core speedup unless the selected "
            "backend and profiler evidence are recorded here.\n",
            encoding="utf-8",
        )


def rerun_command(output_dir: Path) -> str:
    args = list(sys.argv[1:])
    rewritten: list[str] = []
    skip_next = False
    saw_output_dir = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--output-dir":
            rewritten.extend([arg, str(output_dir)])
            saw_output_dir = True
            skip_next = index + 1 < len(args)
            continue
        if arg.startswith("--output-dir="):
            rewritten.append(f"--output-dir={output_dir}")
            saw_output_dir = True
            continue
        rewritten.append(arg)
    if not saw_output_dir:
        rewritten.extend(["--output-dir", str(output_dir)])
    if "--resume" not in rewritten:
        rewritten.append("--resume")
    return shlex.join(["examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh", *rewritten])


def set_local_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def gpu_report() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def run(args: argparse.Namespace) -> None:
    configure_local_no_proxy()
    set_local_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    base_models = dict(DEFAULT_BASE_MODELS)
    base_models.update(LAYER_SENSITIVITY_DEFAULT_MODELS)
    base_models.update(parse_model_id_overrides(args.model_id))
    speculators = dict(EAGLE3_SPECULATORS)
    speculators.update(parse_model_id_overrides(args.speculator_model))
    model_labels = parse_csv_list(args.models)
    modes = expand_modes(args.mode)
    tasks = expand_tasks(args.task)
    args._dense_baseline_align_sr24_eager = (
        bool(args.align_dense_baseline_to_sr24_eager)
        and "dense_baseline" in modes
        and any(
            mode in SR24_RUNTIME_MODES and sr24_requires_enforce_eager(args, mode)
            for mode in modes
        )
    )
    ensure_sr24_iteration_logs(output_dir, modes)

    commands: list[str] = []
    task_preflight_failures: dict[str, str] = {}
    for model_label in model_labels:
        model_path = args.model_path or base_models.get(model_label)
        tokenizer_path = args.tokenizer_path or model_path
        speculator_path = args.speculator_model_path or speculators.get(model_label, "")
        if not model_path:
            raise ValueError(f"unknown model label: {model_label}")
        if not speculator_path:
            raise ValueError(f"missing speculator for {model_label}")
        for mode in modes:
            for task in tasks:
                run_dir = output_dir / model_label / mode / task
                run_dir.mkdir(parents=True, exist_ok=True)
                if args.resume and completed_run(run_dir):
                    continue
                prepare_run_dir_for_rerun(run_dir)
                reason = should_skip_task(args, task)
                if reason:
                    write_skip(run_dir, reason=reason, task=task, mode=mode, model_label=model_label)
                    continue
                if task in TASK_PREFLIGHT_DATASETS:
                    manifest_path, _, manifest_count, manifest_note = ensure_task_manifest(
                        args, task
                    )
                    if task not in task_preflight_failures:
                        task_preflight_failures[task] = preflight_task_access(
                            args, task, run_dir
                        )
                    preflight_reason = task_preflight_failures[task]
                    if preflight_reason:
                        write_json(
                            run_dir / "run_meta.json",
                            {
                                "status": "failed",
                                "returncode": 2,
                                "error": preflight_reason,
                                "model_label": model_label,
                                "model_path": model_path,
                                "tokenizer_path": tokenizer_path,
                                "speculator_path": speculator_path,
                                "mode": mode,
                                "mode_group": mode_group(mode),
                                "task": task,
                                "uses_speculative": mode_uses_spec(mode),
                                "max_new_tokens": args.max_new_tokens
                                or TASK_MAX_TOKENS.get(task, 512),
                                "use_task_manifests": args.use_task_manifests,
                                "manifest_size": manifest_request_count(args),
                                "manifest_dir": str(args.manifest_dir.resolve()),
                                "manifest_path": str(manifest_path) if manifest_path else "",
                                "manifest_count": manifest_count,
                                "manifest_note": manifest_note,
                                "hf_cache_root": str(args.hf_cache_root.resolve()),
                                "humaneval_sandbox": "",
                                "created_at": timestamp(),
                            },
                        )
                        continue
                process = None
                case_started_at = timestamp()
                case_start_time = time.perf_counter()
                try:
                    process, port = start_server(
                        args,
                        mode=mode,
                        task=task,
                        model_label=model_label,
                        model_path=model_path,
                        speculator_path=speculator_path,
                        run_dir=run_dir,
                    )
                    before = scrape_spec_metrics(port)
                    rc = run_lm_eval(
                        args,
                        task=task,
                        mode=mode,
                        model_path=model_path,
                        tokenizer_path=tokenizer_path,
                        port=port,
                        run_dir=run_dir,
                    )
                    after = scrape_spec_metrics(port)
                    case_ended_at = timestamp()
                    case_elapsed_seconds = time.perf_counter() - case_start_time
                    stats_error = ""
                    if (
                        rc == 0
                        and mode.startswith("token_dense_")
                        and not token_dense_stats_valid(run_dir)
                    ):
                        rc = 3
                        stats_error = (
                            "missing token_dense_stats.jsonl; token-dense "
                            "routing was not observed during verification"
                        )
                    if (
                        rc == 0
                        and mode in SR24_RUNTIME_MODES
                        and not sr24_stats_valid(run_dir)
                    ):
                        rc = 3
                        stats_error = (
                            "missing speclink_sr24 stats/events; SR24 routing "
                            "was not observed during verification"
                        )
                    sr24_static_stats: dict[str, Any] = {}
                    if mode in SR24_RUNTIME_MODES:
                        try:
                            sr24_static_stats = json.loads(
                                (run_dir / "speclink_sr24_stats.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        except Exception:
                            sr24_static_stats = {}
                    accepted = after.get("vllm:spec_decode_num_accepted_tokens", 0.0) - before.get(
                        "vllm:spec_decode_num_accepted_tokens", 0.0
                    )
                    drafted = after.get("vllm:spec_decode_num_draft_tokens", 0.0) - before.get(
                        "vllm:spec_decode_num_draft_tokens", 0.0
                    )
                    sr24_effective_residual_out_chunk = None
                    sr24_effective_cache_compressed = None
                    sr24_effective_prewarm_compressed = None
                    sr24_auto_compressed_fastpath = None
                    if mode in SR24_RUNTIME_MODES:
                        (
                            sr24_effective_residual_out_chunk,
                            sr24_effective_cache_compressed,
                            sr24_effective_prewarm_compressed,
                            sr24_auto_compressed_fastpath,
                        ) = sr24_compressed_residual_runtime_settings(args, mode)
                    write_json(
                        run_dir / "run_meta.json",
                        {
                            "status": "ok" if rc == 0 else "failed",
                            "returncode": rc,
                            "model_label": model_label,
                            "model_path": model_path,
                            "tokenizer_path": tokenizer_path,
                            "speculator_path": speculator_path,
                            "mode": mode,
                            "mode_group": mode_group(mode),
                            "vllm_dtype": args.vllm_dtype,
                            "sr24_preset": (
                                args.sr24_preset
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_mode": (
                                sr24_runtime_mode(mode)
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_threshold": (
                                args.sr24_threshold if mode in SR24_RUNTIME_MODES else None
                            ),
                            "sr24_backend": (
                                args.sr24_backend if mode in SR24_RUNTIME_MODES else ""
                            ),
                            "sr24_residual_backend": (
                                args.sr24_residual_backend
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_residual_device": (
                                args.sr24_residual_device
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_require_gpu_residual": (
                                args.sr24_require_gpu_residual
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_all_corrected_dense_fastpath": (
                                args.sr24_all_corrected_dense_fastpath
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_full_residual_early_dense": (
                                args.sr24_full_residual_early_dense
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_actual_residual_backend": (
                                sr24_static_stats.get("residual_backend", "")
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_actual_residual_device": (
                                sr24_static_stats.get("residual_device", "")
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_dense_fastpath_noop": (
                                sr24_static_stats.get("dense_fastpath_noop")
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_linear_hooks_enabled": (
                                sr24_static_stats.get("linear_hooks_enabled")
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_draft_scores_enabled": (
                                sr24_static_stats.get("draft_scores_enabled")
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_storage_over_dense": (
                                sr24_static_stats.get("storage_over_dense")
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_reduce_cpu_sync": (
                                args.sr24_reduce_cpu_sync
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_sync_mask_state": (
                                args.sr24_sync_mask_state
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_static_mask_state": (
                                args.sr24_static_mask_state
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_static_mask_buffer": (
                                args.sr24_static_mask_buffer
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_batched_mask_builder": (
                                args.sr24_batched_mask_builder
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_batched_uniform_direct": (
                                args.sr24_batched_uniform_direct
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_gpu_count_mask_builder": (
                                args.sr24_gpu_count_mask_builder
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_direct_cslt_linear": (
                                args.sr24_direct_cslt_linear
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_effective_direct_cslt_linear": (
                                sr24_effective_direct_cslt_linear(args, mode)
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_cslt_small_m_alg_id_enable": (
                                args.sr24_cslt_small_m_alg_id_enable
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_cslt_small_m_threshold": (
                                args.sr24_cslt_small_m_threshold
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_cslt_small_m_alg_id": (
                                args.sr24_cslt_small_m_alg_id
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_cslt_small_m_threshold_by_leaf": (
                                args.sr24_cslt_small_m_threshold_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_cslt_small_m_alg_id_by_leaf": (
                                args.sr24_cslt_small_m_alg_id_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_auto_direct_cslt_base_only": (
                                args.sr24_auto_direct_cslt_base_only
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_gate_up_split": (
                                args.sr24_gate_up_split
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_gate_up_channel_dense_fraction": (
                                args.sr24_gate_up_channel_dense_fraction
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_gate_up_channel_strategy": (
                                args.sr24_gate_up_channel_strategy
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_gate_up_channel_fused_act": (
                                args.sr24_gate_up_channel_fused_act
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_row_routed_mlp": (
                                args.sr24_row_routed_mlp
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_row_routed_mlp_reuse_base_output": (
                                args.sr24_row_routed_mlp_reuse_base_output
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_row_routed_mlp_min_dense_rows": (
                                args.sr24_row_routed_mlp_min_dense_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_row_routed_mlp_min_dense_rows_by_leaf": (
                                args.sr24_row_routed_mlp_min_dense_rows_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_row_routed_mlp_max_dense_rows": (
                                args.sr24_row_routed_mlp_max_dense_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_row_routed_mlp_max_dense_rows_by_leaf": (
                                args.sr24_row_routed_mlp_max_dense_rows_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_row_routed_mlp_max_base_rows": (
                                args.sr24_row_routed_mlp_max_base_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_row_routed_mlp_max_base_rows_by_leaf": (
                                args.sr24_row_routed_mlp_max_base_rows_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_scheduler_policy_path": (
                                args.sr24_scheduler_policy_path
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_scheduler_policy_dense_bypass": (
                                args.sr24_scheduler_policy_dense_bypass
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_force_cudagraph_none_for_mixed": (
                                args.sr24_force_cudagraph_none_for_mixed
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_selective_non_draft_policy": (
                                args.sr24_selective_non_draft_policy
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_selective_residual_policy": (
                                args.sr24_selective_residual_policy
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_prefix_threshold": (
                                args.sr24_prefix_threshold
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_selective_extra_after_low": (
                                args.sr24_selective_extra_after_low
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_selective_min_prefix_residual": (
                                args.sr24_selective_min_prefix_residual
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_selective_max_residual_draft_rows": (
                                args.sr24_selective_max_residual_draft_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_low_confidence_cap_by_risk": (
                                args.sr24_low_confidence_cap_by_risk
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_early_dense_tokens": (
                                args.sr24_early_dense_tokens
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_static_all_residual_dense_fastpath": (
                                args.sr24_static_all_residual_dense_fastpath
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_mask_buffer_capacity": (
                                args.sr24_mask_buffer_capacity
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_residual_bucket_scale_by_active": (
                                args.sr24_residual_bucket_scale_by_active
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_residual_bucket_priority": (
                                args.sr24_residual_bucket_priority
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_route_bucket_rows": (
                                args.sr24_route_bucket_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_route_all_residual_rows": (
                                args.sr24_route_all_residual_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_direct_cpu_route_rows": (
                                args.sr24_direct_cpu_route_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_route_reuse_base_output": (
                                args.sr24_route_reuse_base_output
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_route_contiguous_fastpath": (
                                args.sr24_route_contiguous_fastpath
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_route_dense_fallback_fraction": (
                                args.sr24_route_dense_fallback_fraction
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback": (
                                args.sr24_adaptive_dense_fallback
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback_no_residual_only": (
                                args.sr24_adaptive_dense_fallback_no_residual_only
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback_small_rows": (
                                args.sr24_adaptive_dense_fallback_small_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback_gate_up_fraction": (
                                args.sr24_adaptive_dense_fallback_gate_up_fraction
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback_down_fraction": (
                                args.sr24_adaptive_dense_fallback_down_fraction
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback_small_down_no_residual": (
                                args.sr24_adaptive_dense_fallback_small_down_no_residual
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_adaptive_dense_fallback_small_gate_up_no_residual": (
                                args.sr24_adaptive_dense_fallback_small_gate_up_no_residual
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_triton_route_assembly": (
                                args.sr24_triton_route_assembly
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_triton_bucket_override": (
                                args.sr24_triton_bucket_override
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_triton_bucket_dense_gemm": (
                                args.sr24_triton_bucket_dense_gemm
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_triton_bucket_scatter": (
                                args.sr24_triton_bucket_scatter
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_disable_runtime_stats": (
                                args.sr24_disable_runtime_stats
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_breakdown": (
                                args.sr24_breakdown
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_breakdown_linear": (
                                args.sr24_breakdown_linear
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_breakdown_exact_routing": (
                                args.sr24_breakdown_exact_routing
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_breakdown_gpu_counts": (
                                args.sr24_breakdown_gpu_counts
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_breakdown_interval": (
                                args.sr24_breakdown_interval
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_target_leafs": (
                                args.sr24_target_leafs
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_residual_target_leafs": (
                                args.sr24_residual_target_leafs
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_base_only_layer_ids": (
                                args.sr24_base_only_layer_ids
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_base_only_layer_ids_by_leaf": (
                                args.sr24_base_only_layer_ids_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_residual_layer_ids_by_leaf": (
                                args.sr24_residual_layer_ids_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_runtime_base_only_layer_ids_by_leaf": (
                                args.sr24_runtime_base_only_layer_ids_by_leaf
                                if mode in SR24_RUNTIME_MODES
                                else ""
                            ),
                            "sr24_residual_out_chunk": (
                                args.sr24_residual_out_chunk
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_effective_residual_out_chunk": (
                                sr24_effective_residual_out_chunk
                            ),
                            "sr24_cache_compressed_residual_weight": (
                                args.sr24_cache_compressed_residual_weight
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_effective_cache_compressed_residual_weight": (
                                sr24_effective_cache_compressed
                            ),
                            "sr24_prewarm_compressed_residual_weight": (
                                args.sr24_prewarm_compressed_residual_weight
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_effective_prewarm_compressed_residual_weight": (
                                sr24_effective_prewarm_compressed
                            ),
                            "sr24_auto_compressed_residual_fastpath": (
                                args.sr24_auto_compressed_residual_fastpath
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_effective_auto_compressed_residual_fastpath": (
                                sr24_auto_compressed_fastpath
                            ),
                            "sr24_compressed_residual_triton": (
                                args.sr24_compressed_residual_triton
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_compressed_residual_block_m": (
                                args.sr24_compressed_residual_block_m
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_compressed_residual_block_n": (
                                args.sr24_compressed_residual_block_n
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_compressed_residual_block_g": (
                                args.sr24_compressed_residual_block_g
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_extract_chunk_rows": (
                                args.sr24_extract_chunk_rows
                                if mode in SR24_RUNTIME_MODES
                                else None
                            ),
                            "sr24_mask_path": (
                                str(args.sr24_mask_path.resolve())
                                if mode in SR24_RUNTIME_MODES and args.sr24_mask_path
                                else ""
                            ),
                            "task": task,
                            "uses_speculative": mode_uses_spec(mode),
                            "trace_confidence": bool(
                                args.trace_confidence and mode_uses_spec(mode)
                            ),
                            "confidence_trace_path": str(
                                (run_dir / "speclink_confidence_trace.jsonl")
                                .resolve()
                            )
                            if args.trace_confidence and mode_uses_spec(mode)
                            else "",
                            "max_new_tokens": args.max_new_tokens
                            or TASK_MAX_TOKENS.get(task, 512),
                            "use_task_manifests": args.use_task_manifests,
                            "manifest_size": manifest_request_count(args),
                            "manifest_dir": str(args.manifest_dir.resolve()),
                            "hf_cache_root": str(args.hf_cache_root.resolve()),
                            "humaneval_sandbox": (
                                args.humaneval_sandbox
                                if task == "humaneval_instruct"
                                else ""
                            ),
                            "spec_accepted_tokens": accepted if drafted else None,
                            "spec_draft_tokens": drafted if drafted else None,
                            "spec_acceptance_rate": accepted / drafted if drafted else None,
                            "started_at": case_started_at,
                            "ended_at": case_ended_at,
                            "elapsed_seconds": case_elapsed_seconds,
                            "error": stats_error,
                            "created_at": timestamp(),
                        },
                    )
                    commands.append(str((run_dir / "lm_eval_command.json").resolve()))
                finally:
                    stop_process(process)
                    if args.server_shutdown_settle_s > 0:
                        time.sleep(args.server_shutdown_settle_s)

    write_json(
        output_dir / "run_config.json",
        {
            "models": model_labels,
            "modes": modes,
            "tasks": tasks,
            "output_dir": str(output_dir),
            "num_spec_tokens": args.num_spec_tokens,
            "vllm_dtype": args.vllm_dtype,
            "vllm_compilation_config": args.vllm_compilation_config,
            "sr24_preset": args.sr24_preset,
            "sr24_threshold": args.sr24_threshold,
            "sr24_backend": args.sr24_backend,
            "sr24_residual_backend": args.sr24_residual_backend,
            "sr24_residual_device": args.sr24_residual_device,
            "sr24_require_gpu_residual": args.sr24_require_gpu_residual,
            "sr24_all_corrected_dense_fastpath":
            args.sr24_all_corrected_dense_fastpath,
            "sr24_full_residual_early_dense":
            args.sr24_full_residual_early_dense,
            "sr24_reduce_cpu_sync": args.sr24_reduce_cpu_sync,
            "sr24_sync_mask_state": args.sr24_sync_mask_state,
            "sr24_static_mask_state": args.sr24_static_mask_state,
            "sr24_static_mask_buffer": args.sr24_static_mask_buffer,
            "sr24_batched_mask_builder": args.sr24_batched_mask_builder,
            "sr24_batched_uniform_direct": args.sr24_batched_uniform_direct,
            "sr24_gpu_count_mask_builder": args.sr24_gpu_count_mask_builder,
            "sr24_direct_cslt_linear": args.sr24_direct_cslt_linear,
            "sr24_cslt_small_m_alg_id_enable":
            args.sr24_cslt_small_m_alg_id_enable,
            "sr24_cslt_small_m_threshold": args.sr24_cslt_small_m_threshold,
            "sr24_cslt_small_m_alg_id": args.sr24_cslt_small_m_alg_id,
            "sr24_cslt_small_m_threshold_by_leaf":
            args.sr24_cslt_small_m_threshold_by_leaf,
            "sr24_cslt_small_m_alg_id_by_leaf":
            args.sr24_cslt_small_m_alg_id_by_leaf,
            "sr24_auto_direct_cslt_base_only":
            args.sr24_auto_direct_cslt_base_only,
            "sr24_gate_up_split": args.sr24_gate_up_split,
            "sr24_gate_up_channel_dense_fraction":
            args.sr24_gate_up_channel_dense_fraction,
            "sr24_gate_up_channel_strategy": args.sr24_gate_up_channel_strategy,
            "sr24_gate_up_channel_fused_act":
            args.sr24_gate_up_channel_fused_act,
            "sr24_row_routed_mlp": args.sr24_row_routed_mlp,
            "sr24_row_routed_mlp_reuse_base_output":
            args.sr24_row_routed_mlp_reuse_base_output,
            "sr24_row_routed_mlp_min_dense_rows":
            args.sr24_row_routed_mlp_min_dense_rows,
            "sr24_row_routed_mlp_min_dense_rows_by_leaf":
            args.sr24_row_routed_mlp_min_dense_rows_by_leaf,
            "sr24_row_routed_mlp_max_dense_rows":
            args.sr24_row_routed_mlp_max_dense_rows,
            "sr24_row_routed_mlp_max_dense_rows_by_leaf":
            args.sr24_row_routed_mlp_max_dense_rows_by_leaf,
            "sr24_row_routed_mlp_max_base_rows":
            args.sr24_row_routed_mlp_max_base_rows,
            "sr24_row_routed_mlp_max_base_rows_by_leaf":
            args.sr24_row_routed_mlp_max_base_rows_by_leaf,
            "sr24_scheduler_policy_path": args.sr24_scheduler_policy_path,
            "sr24_scheduler_policy_dense_bypass":
            args.sr24_scheduler_policy_dense_bypass,
            "sr24_force_cudagraph_none_for_mixed":
            args.sr24_force_cudagraph_none_for_mixed,
            "sr24_dynamic_auto_cudagraph": args.sr24_dynamic_auto_cudagraph,
            "sr24_selective_non_draft_policy":
            args.sr24_selective_non_draft_policy,
            "sr24_selective_dense_nonverify_layer_ids_by_leaf":
            args.sr24_selective_dense_nonverify_layer_ids_by_leaf,
            "sr24_selective_dense_nonverify_max_rows":
            args.sr24_selective_dense_nonverify_max_rows,
            "sr24_selective_residual_policy": args.sr24_selective_residual_policy,
            "sr24_prefix_threshold": args.sr24_prefix_threshold,
            "sr24_selective_extra_after_low": args.sr24_selective_extra_after_low,
            "sr24_selective_min_prefix_residual":
            args.sr24_selective_min_prefix_residual,
            "sr24_selective_max_residual_draft_rows":
            args.sr24_selective_max_residual_draft_rows,
            "sr24_low_confidence_cap_by_risk":
            args.sr24_low_confidence_cap_by_risk,
            "sr24_early_dense_tokens": args.sr24_early_dense_tokens,
            "sr24_static_all_residual_dense_fastpath":
            args.sr24_static_all_residual_dense_fastpath,
            "sr24_mask_buffer_capacity": args.sr24_mask_buffer_capacity,
            "sr24_residual_bucket_size": args.sr24_residual_bucket_size,
            "sr24_residual_bucket_scale_by_active":
            args.sr24_residual_bucket_scale_by_active,
            "sr24_residual_bucket_priority": args.sr24_residual_bucket_priority,
            "sr24_bonus_priority": args.sr24_bonus_priority,
            "sr24_draft_position_priority_scale":
            args.sr24_draft_position_priority_scale,
            "sr24_route_bucket_rows": args.sr24_route_bucket_rows,
            "sr24_route_all_residual_rows": args.sr24_route_all_residual_rows,
            "sr24_direct_cpu_route_rows": args.sr24_direct_cpu_route_rows,
            "sr24_route_reuse_base_output": args.sr24_route_reuse_base_output,
            "sr24_route_contiguous_fastpath":
            args.sr24_route_contiguous_fastpath,
            "sr24_route_overlap_streams": args.sr24_route_overlap_streams,
            "sr24_route_dense_fallback_fraction":
            args.sr24_route_dense_fallback_fraction,
            "sr24_route_min_dense_rows": args.sr24_route_min_dense_rows,
            "sr24_route_min_base_rows": args.sr24_route_min_base_rows,
            "sr24_route_min_base_rows_by_leaf":
            args.sr24_route_min_base_rows_by_leaf,
            "sr24_route_max_dense_fraction": args.sr24_route_max_dense_fraction,
            "sr24_triton_route_assembly": args.sr24_triton_route_assembly,
            "sr24_triton_bucket_override": args.sr24_triton_bucket_override,
            "sr24_triton_bucket_dense_gemm":
            args.sr24_triton_bucket_dense_gemm,
            "sr24_triton_bucket_scatter": args.sr24_triton_bucket_scatter,
            "sr24_triton_bucket_dense_block_m":
            args.sr24_triton_bucket_dense_block_m,
            "sr24_triton_bucket_dense_block_n":
            args.sr24_triton_bucket_dense_block_n,
            "sr24_triton_bucket_dense_block_k":
            args.sr24_triton_bucket_dense_block_k,
            "sr24_bucket_dense_copy": args.sr24_bucket_dense_copy,
            "sr24_bucket_dense_copy_active_only":
            args.sr24_bucket_dense_copy_active_only,
            "sr24_bucket_dense_compute_active_only":
            args.sr24_bucket_dense_compute_active_only,
            "sr24_bucket_dense_active_mask_fused":
            args.sr24_bucket_dense_active_mask_fused,
            "sr24_disable_runtime_stats": args.sr24_disable_runtime_stats,
            "sr24_breakdown": args.sr24_breakdown,
            "sr24_breakdown_linear": args.sr24_breakdown_linear,
            "sr24_breakdown_exact_routing": args.sr24_breakdown_exact_routing,
            "sr24_breakdown_gpu_counts": args.sr24_breakdown_gpu_counts,
            "sr24_breakdown_interval": args.sr24_breakdown_interval,
            "sr24_target_leafs": args.sr24_target_leafs,
            "sr24_residual_target_leafs": args.sr24_residual_target_leafs,
            "sr24_base_only_layer_ids": args.sr24_base_only_layer_ids,
            "sr24_base_only_layer_ids_by_leaf":
            args.sr24_base_only_layer_ids_by_leaf,
            "sr24_runtime_base_only_layer_ids_by_leaf":
            args.sr24_runtime_base_only_layer_ids_by_leaf,
            "sr24_base_only_dense_verify_max_rows":
            args.sr24_base_only_dense_verify_max_rows,
            "sr24_base_only_dense_verify_layer_ids":
            args.sr24_base_only_dense_verify_layer_ids,
            "sr24_base_only_dense_verify_layer_ids_by_leaf":
            args.sr24_base_only_dense_verify_layer_ids_by_leaf,
            "sr24_residual_layer_ids_by_leaf":
            args.sr24_residual_layer_ids_by_leaf,
            "sr24_residual_out_chunk": args.sr24_residual_out_chunk,
            "sr24_cache_compressed_residual_weight":
            args.sr24_cache_compressed_residual_weight,
            "sr24_prewarm_compressed_residual_weight":
            args.sr24_prewarm_compressed_residual_weight,
            "sr24_auto_compressed_residual_fastpath":
            args.sr24_auto_compressed_residual_fastpath,
            "sr24_compressed_residual_triton":
            args.sr24_compressed_residual_triton,
            "sr24_extract_chunk_rows": args.sr24_extract_chunk_rows,
            "sr24_disable_default_mask": args.sr24_disable_default_mask,
            "sr24_allow_cudagraph": args.sr24_allow_cudagraph,
            "sr24_default_vllm_compile": args.sr24_default_vllm_compile,
            "align_dense_baseline_to_sr24_eager":
            args.align_dense_baseline_to_sr24_eager,
            "dense_baseline_aligned_to_sr24_eager":
            dense_baseline_aligns_to_sr24_eager(args),
            "sr24_mask_path": (
                str(args.sr24_mask_path.resolve()) if args.sr24_mask_path else ""
            ),
            "max_context_length": args.max_context_length,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "use_task_manifests": args.use_task_manifests,
            "manifest_size": manifest_request_count(args),
            "manifest_dir": str(args.manifest_dir.resolve()),
            "hf_cache_root": str(args.hf_cache_root.resolve()),
            "humaneval_sandbox": args.humaneval_sandbox,
            "gpu_report": gpu_report(),
            "seed": args.seed,
            "created_at": timestamp(),
        },
    )
    with (output_dir / "commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("# Re-run this matrix\n")
        handle.write(rerun_command(output_dir) + "\n")
    if args.aggregate:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "aggregate_lm_eval_accuracy.py"),
                "--output-dir",
                str(output_dir),
            ],
            cwd=str(SPECULATORS_ROOT),
            check=False,
        )
    print(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lm-eval through SpecLink vLLM serving modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="all")
    parser.add_argument("--task", default="smoke")
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--limit", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--speculator-model-path", default="")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--speculator-model", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument(
        "--vllm-dtype",
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help=(
            "vLLM --dtype. Keep auto for normal runs; use half/float16 for "
            "SR24 split-matmul numerical ablations."
        ),
    )
    parser.add_argument(
        "--vllm-compilation-config",
        default="",
        help=(
            "Optional JSON string passed through to vLLM --compilation-config "
            "for compile/CUDA Graph ablations."
        ),
    )
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--port-base", type=int, default=8260)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=2.0)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--calibration-cache-root", type=Path, default=DEFAULT_C4_CALIBRATION_CACHE_ROOT)
    parser.add_argument(
        "--sr24-preset",
        choices=[
            "manual",
            "quality_safe_selective",
            "down8_15_residual_only",
            "quality_gateup_only",
            "gateup_cap0_dense_guard",
            "gateup_cap0_maskstate_densefallback06",
            "gateup_cap0_maskstate_densefallback00",
            "gateup_cap0_graph_probe",
            "speed_tradeoff_down16_base",
            "riskcap2_bucket16_directcslt",
            "lowresidual_gateup_riskcap2",
            "mlpall_lowconf_prefix5_tritonoverride",
            "mlpall_direct_prefix2",
            "mlpall_fixedprefix2_directcslt",
            "mlpall_fixedprefix2_graphsafe",
            "mlpall_tilefill_prefix2_bucket32_cublas",
            "lossy_prefix2_rowrouted_mlp",
            "lossy_prefix2_rowrouted_mlp_operator_guard",
            "gateup_res16_25_base26_31_critical4",
            "gateup_res16_25_base26_31_critical4_smallrow160",
            "fixedprefix4_bucket16_directcslt",
            "fixedprefix4_all_rowrouted_graph",
            "down0_15_fixedprefix4_directcslt",
            "criticalprefix4_bucket16_directcslt",
            "criticalprefix_extra2_gateup_scaledbucket",
            "accuracy_first",
            "accuracy_gate_only",
            "accuracy_down_only",
            "lossy_static_tail",
            "throughput_aggressive",
        ],
        default="manual",
        help=(
            "Apply a tested SR24 speclink_t08 preset. manual preserves all "
            "explicit SR24 flags. quality_safe_selective uses dynamic "
            "dense_rows residual on gate_up=16-31 and down=8-15 without any "
            "base-only layer filter. down8_15_residual_only touches only "
            "down_proj=8-15 with dense_rows residual and no base-only tail. "
            "quality_gateup_only is the current safer paired-quality reference: "
            "gate_up=16-31, all_if_any_low@0.4, prefix4, and no down base-only. "
            "gateup_cap0_dense_guard is a guarded selective candidate, not a "
            "paired-safe baseline: "
            "gate_up=16-31, low_confidence@0.8 cap0, bucket32, and adaptive "
            "dense fallback at residual fraction 0.05; it is near dense "
            "throughput, not the final speed target. "
            "gateup_cap0_maskstate_densefallback06 keeps mixed steps eager "
            "but promotes steps with >=60% residual rows to the exact "
            "all-residual dense fastpath so those steps can use CUDA Graph. "
            "gateup_cap0_maskstate_densefallback00 promotes any residual "
            "step to all-residual and is a stricter quality diagnostic. "
            "gateup_cap0_graph_probe keeps the same config but explicitly "
            "enables dynamic-auto CUDA Graph with a stable bucket; use it for "
            "graph precision probes, not as a quality-safe path. "
            "speed_tradeoff_down16_base is the current best speed/quality "
            "tradeoff probe: gate_up=16-31 cap1/bucket32 residual and "
            "down_proj=16-31 base-only; it is not paired-accuracy stable. "
            "riskcap2_bucket16_directcslt is a current SR24 speed candidate: "
            "gate_up=16-31 and down=8-15, low-confidence residual with "
            "prefix2 plus two risk-capped draft rows, bucket16 dense-copy "
            "correction, and direct cuSPARSELt sparse base. "
            "lowresidual_gateup_riskcap2 is the current best measured "
            "gate-up-only variant with bucket8 by default; use it when "
            "quality-gating the throughput route from the 2026-06-29 "
            "slowdown pass. "
            "mlpall_lowconf_prefix5_tritonoverride is the all-MLP speed "
            "target probe: gate_up/down all layers, low_confidence@0.6, "
            "prefix5, bucket32, dynamic CUDA Graph, and Triton bucket "
            "override; use it for quality gates of the 1.2x full-batch speed "
            "candidate, not as a proven safe default. "
            "mlpall_direct_prefix2 is the current 8pp-budget live candidate: "
            "same all-MLP scope, mandatory dense prefix2, and direct "
            "cuSPARSELt sparse base. "
            "mlpall_fixedprefix2_directcslt is the score-free route-table "
            "variant: all-MLP, fixed_prefix2+bonus, bucket32, and direct "
            "cuSPARSELt. "
            "lossy_prefix2_rowrouted_mlp is an 8pp-budget systems candidate: "
            "fixed-prefix route-table rows, MLP-level disjoint dense/sparse "
            "branches, descriptor-only route plans, reusable fixed-block "
            "input buffers, and min-dense-row filling to avoid tiny dense GEMMs. "
            "lossy_prefix2_rowrouted_mlp_operator_guard is the same fixed "
            "prefix2 route-table policy, but follows the packed-MLP planner: "
            "descriptor-only/input-buffer data format, no dense-fill "
            "promotion, and dense fallback until the sparse-base branch has "
            "about bs64/K8 row fill. "
            "fixedprefix4_bucket16_directcslt keeps the same operator scope "
            "but corrects only the first four draft rows plus the bonus row. "
            "fixedprefix4_all_rowrouted_graph protects a fixed prefix plus "
            "all non-draft rows, avoids direct CPU route rows, and uses "
            "MLP-level row routing to stop dense rows from also paying sparse "
            "base work; use it as the current graph-friendly systems probe. "
            "down0_15_fixedprefix4_directcslt protects down_proj=0-15 with "
            "fixed_prefix=4 and keeps gate_up dense; it is the current focused "
            "doc2-safe candidate before broader accuracy gates. "
            "criticalprefix4_bucket16_directcslt is the current quality-safe "
            "bucket16/direct-cuSPARSELt shape with critical_prefix@0.6, "
            "prefix4, and no extra row after the first low-confidence token. "
            "accuracy_first is now the conservative static tail candidate: "
            "qkv/o exact densefastpath plus base-only gate_up=31 with "
            "--sr24-gate-up-split up_sparse. accuracy_gate_only now attaches "
            "only the fully fused gate_up=31 sparse tail; accuracy_down_only "
            "keeps only down=31 as a negative/diagnostic ablation. "
            "lossy_static_tail and throughput_aggressive use "
            "gate_up=31,down=30-31; lossy_static_tail is the clearer name "
            "for the current 8pp-budget static-tail speed candidate."
        ),
    )
    parser.add_argument("--sr24-threshold", type=float, default=0.8)
    parser.add_argument(
        "--sr24-backend",
        choices=["torch_sparse", "prototype", "dense_zero"],
        default="torch_sparse",
        help=(
            "SR24 execution backend. torch_sparse uses PyTorch "
            "SparseSemiStructuredTensorCUSPARSELT; dense_zero/prototype keeps "
            "the zeroed 2:4 base weight dense-shaped for correctness and "
            "backend-isolation checks."
        ),
    )
    parser.add_argument(
        "--sr24-residual-backend",
        choices=["compressed_dense", "torch_sparse", "dense_rows"],
        default="dense_rows",
        help=(
            "Residual correction backend for SR24 torch_sparse mode. "
            "dense_rows keeps the original dense weight and replaces corrected "
            "rows with exact dense Linear outputs, so it is the default "
            "quality-safe diagnostic backend. compressed_dense keeps residual "
            "values compressed and materializes a dense residual matrix per "
            "corrected Linear call, but the sparse-base plus residual split "
            "GEMMs are not numerically identical to one dense bf16 GEMM; use it "
            "only for storage/operator ablations until a fused kernel is "
            "validated. torch_sparse "
            "tries a second PyTorch semi-structured sparse tensor and can OOM "
            "for full Llama-3.1-8B on a 32GB GPU."
        ),
    )
    parser.add_argument(
        "--sr24-residual-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Storage device for compressed SR24 residual values. auto uses "
            "GPU-resident residual values for the current performance path; "
            "use cpu explicitly only as a memory fallback or CPU-transfer "
            "ablation."
        ),
    )
    parser.add_argument(
        "--sr24-require-gpu-residual",
        action="store_true",
        help=(
            "Fail SR24 model attach if compressed_dense residual values are "
            "not GPU-resident. This is a diagnostic guard for all_corrected_24 "
            "and speclink_t08 performance runs."
        ),
    )
    parser.add_argument(
        "--sr24-mask-path",
        type=Path,
        default=None,
        help=(
            "Optional activation-aware 2:4 mask cache for SPECLINK_SR24_MODE. "
            "If omitted, known model labels such as llama3_1_8b use the "
            "repo-local C4 activation-aware mask when it exists."
        ),
    )
    parser.add_argument(
        "--sr24-disable-default-mask",
        action="store_true",
        help=(
            "Disable repo-local default SR24 activation-aware masks. Use this "
            "only for explicit magnitude-mask ablations."
        ),
    )
    parser.add_argument(
        "--sr24-reduce-cpu-sync",
        action="store_true",
        help=(
            "SR24 ablation: avoid per-step CPU scalar reductions for selective "
            "draft-row residual counts and use the masked full-row residual path. "
            "Residual fraction fields are approximate/incomplete in this mode."
        ),
    )
    parser.add_argument(
        "--sr24-allow-cudagraph",
        action="store_true",
        help=(
            "SR24 CPU/launch-overhead ablation: do not force --enforce-eager "
            "for base_only_24/all_corrected_24, allowing vLLM CUDA Graph "
            "dispatch. Selective speclink_t08 is allowed only for static "
            "all_residual/no_residual mask-state ablations; dynamic "
            "auto/mixed masks stay eager for correctness."
        ),
    )
    parser.add_argument(
        "--sr24-default-vllm-compile",
        action="store_true",
        help=(
            "Do not force eager or an SR24-specific compilation config for "
            "SR24 modes. This is a compile-compatibility ablation."
        ),
    )
    parser.add_argument(
        "--sr24-stats-interval",
        type=int,
        default=1,
        help=(
            "Flush SR24 verify summary every N decode steps. Keep 1 for exact "
            "diagnostics; use a larger value for CPU-overhead throughput "
            "ablations."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown",
        action="store_true",
        help=(
            "Write an SR24 component breakdown JSON with scheduler/mask, "
            "routing, bucket, CUDA Graph, and optional Linear-hook timing. "
            "Profiling-only; use clean runs without this flag for throughput "
            "claims."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-linear",
        action="store_true",
        help=(
            "Also time SR24 Linear internals such as sparse base, residual "
            "correction, gather, and scatter. This uses CUDA events and is a "
            "sync-heavy diagnostic path."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-exact-routing",
        action="store_true",
        help=(
            "Synchronize GPU scalars to report exact residual/base routing "
            "counts and bucket fill ratio. This intentionally measures CPU/GPU "
            "sync overhead."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-gpu-counts",
        action="store_true",
        help=(
            "Accumulate residual/base routing counts and bucket active rows on "
            "GPU and read them only when the breakdown is flushed. This is a "
            "lower-CPU-sync routing diagnostic than --sr24-breakdown-exact-routing."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-interval",
        type=int,
        default=2000,
        help="Flush SR24 breakdown after roughly this many CUDA timing events.",
    )
    parser.add_argument(
        "--trace-confidence",
        action="store_true",
        help=(
            "Enable SPECLINK_TRACE_CONFIDENCE for speculative lm-eval serving "
            "runs and write speclink_confidence_trace.jsonl under each run "
            "directory. Diagnostic only; it adds CPU-side trace overhead."
        ),
    )
    parser.add_argument(
        "--sr24-sync-mask-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --sr24-reduce-cpu-sync, synchronize once per verify step to "
            "classify the residual mask as all_residual/no_residual/mixed. "
            "Use --no-sr24-sync-mask-state for the stricter no-sync ablation."
        ),
    )
    parser.add_argument(
        "--sr24-static-mask-state",
        choices=["auto", "all_residual", "no_residual", "mixed"],
        default="auto",
        help=(
            "Static SR24 residual-mask state override for CPU-sync/CUDA Graph "
            "ablations. auto preserves normal behavior."
        ),
    )
    parser.add_argument(
        "--sr24-static-all-residual-dense-fastpath",
        action="store_true",
        help=(
            "When --sr24-static-mask-state=all_residual, keep the original "
            "dense Linear for residual-corrected target leafs. This is valid "
            "only for leafs whose residual path covers every row."
        ),
    )
    parser.add_argument(
        "--sr24-all-corrected-dense-fastpath",
        dest="sr24_all_corrected_dense_fastpath",
        action="store_true",
        default=True,
        help=(
            "For all_corrected_24, use the algebraically equivalent original "
            "dense Linear instead of sparse base plus residual correction."
        ),
    )
    parser.add_argument(
        "--no-sr24-all-corrected-dense-fastpath",
        dest="sr24_all_corrected_dense_fastpath",
        action="store_false",
        help="Disable the all_corrected_24 dense fastpath for ablation.",
    )
    parser.add_argument(
        "--sr24-full-residual-early-dense",
        action="store_true",
        help=(
            "When a selective verify step is known to be all-residual, run the "
            "dense target Linear directly instead of sparse base plus dense-row "
            "correction. This is required for early-dense accuracy guards to be "
            "numerically equivalent to dense on fully protected steps."
        ),
    )
    parser.add_argument(
        "--sr24-noverify-dense-mlp-fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For selective/all_corrected SR24 forwards without an active verify "
            "mask, run the full Llama MLP through dense weights before trying "
            "row-routed/sparse hooks. Use --no-sr24-noverify-dense-mlp-fastpath "
            "to A/B the older per-Linear exact-dense fallback."
        ),
    )
    parser.add_argument(
        "--sr24-selective-correct-non-draft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For selective SR24 modes, correct non-draft scheduled tokens "
            "with residual and gate only draft-token rows by DLM confidence."
        ),
    )
    parser.add_argument(
        "--sr24-selective-non-draft-policy",
        choices=["auto", "all", "none", "bonus", "predicted_full_accept"],
        default="auto",
        help=(
            "Selective SR24 non-draft/bonus-row residual policy. auto preserves "
            "--sr24-selective-correct-non-draft behavior; bonus corrects only "
            "the speculative bonus row; predicted_full_accept corrects the "
            "bonus row only when all draft scores are present and above the "
            "threshold."
        ),
    )
    parser.add_argument(
        "--sr24-selective-dense-nonverify-layer-ids-by-leaf",
        default="",
        help=(
            "Optional selective-mode no-verify dense guard scope. Empty preserves "
            "the historical behavior where --sr24-selective-correct-non-draft "
            "keeps every no-mask/non-draft forward dense. Use values such as "
            "'gate_up_proj=0-15;down_proj=0-15', or 'none' to let no-mask rows "
            "use the 2:4 sparse base."
        ),
    )
    parser.add_argument(
        "--sr24-selective-dense-nonverify-max-rows",
        type=int,
        default=0,
        help=(
            "Selective-mode no-verify small-shape dense guard. When >0, "
            "no-mask forwards with at most this many rows use dense weights "
            "even if their layer scope would otherwise use the 2:4 sparse "
            "base."
        ),
    )
    parser.add_argument(
        "--sr24-static-mask-buffer",
        action="store_true",
        help=(
            "Use a reusable GPU bool buffer for selective residual masks. "
            "Required for graph-safe speclink_t08 ablations."
        ),
    )
    parser.add_argument(
        "--sr24-batched-mask-builder",
        action="store_true",
        help=(
            "Experimental selective critical_prefix+bonus path: build draft "
            "residual mask rows with one batched kernel instead of per-request "
            "fragments. Supports critical_prefix, all_if_any_low, "
            "low_confidence, and high_confidence when early_dense_tokens is "
            "disabled; min_prefix_residual is supported. Matches the "
            "throughput runner flag."
        ),
    )
    parser.add_argument(
        "--sr24-batched-uniform-direct",
        action="store_true",
        help=(
            "SR24 CPU-sync ablation: when every request has uniform K and "
            "direct score rows, skip CPU-side per-request metadata copies in "
            "the batched mask builder. Requires --sr24-batched-mask-builder."
        ),
    )
    parser.add_argument(
        "--sr24-cudagraph-bucket",
        action="store_true",
        help=(
            "Experimental graph-correctness diagnostic. Allow selective SR24 "
            "residual buckets to use persistent CUDA Graph buffers. Default "
            "is off because GSM8K probes currently show quality regressions."
        ),
    )
    parser.add_argument(
        "--sr24-dynamic-auto-cudagraph",
        action="store_true",
        help=(
            "Experimental SR24 ablation. When paired with "
            "--sr24-allow-cudagraph, --no-sr24-force-cudagraph-none-for-mixed, "
            "--sr24-static-mask-state=auto, and persistent mask/bucket buffers, "
            "launch dynamic auto/mixed speclink_t08 without --enforce-eager. "
            "Default is off because dynamic graph correctness must be checked "
            "before final quality claims."
        ),
    )
    parser.add_argument(
        "--sr24-gpu-count-mask-builder",
        action="store_true",
        help=(
            "SR24 CPU-sync ablation: use scheduler count tensors already on "
            "GPU inside the batched mask builder when compatible. Off by "
            "default because it is a measured ablation, not the baseline."
        ),
    )
    parser.add_argument(
        "--sr24-direct-cslt-linear",
        action="store_true",
        help=(
            "SR24 torch-sparse ablation: call the opaque direct cuSPARSELt "
            "Linear path instead of PyTorch F.linear dispatch for sparse base "
            "weights. This matches the throughput runner flag."
        ),
    )
    parser.add_argument(
        "--sr24-auto-direct-cslt-base-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically use the direct cuSPARSELt Linear path for "
            "base_only_24 torch_sparse runs. speclink_t08 and all_corrected_24 "
            "remain controlled by --sr24-direct-cslt-linear."
        ),
    )
    parser.add_argument(
        "--sr24-cslt-small-m-alg-id-enable",
        action="store_true",
        help=(
            "SR24 operator ablation: for direct cuSPARSELt sparse Linear calls "
            "with rows <= --sr24-cslt-small-m-threshold, use "
            "--sr24-cslt-small-m-alg-id instead of the tensor default alg_id."
        ),
    )
    parser.add_argument(
        "--sr24-cslt-small-m-threshold",
        type=int,
        default=96,
        help="Row-count threshold for --sr24-cslt-small-m-alg-id-enable.",
    )
    parser.add_argument(
        "--sr24-cslt-small-m-alg-id",
        type=int,
        default=1,
        help="cuSPARSELt alg_id to use for small-M SR24 sparse Linear calls.",
    )
    parser.add_argument(
        "--sr24-cslt-small-m-threshold-by-leaf",
        default="",
        help=(
            "Optional comma-separated leaf=rows overrides for small-M "
            "cuSPARSELt alg selection, for example down_proj=256."
        ),
    )
    parser.add_argument(
        "--sr24-cslt-small-m-alg-id-by-leaf",
        default="",
        help=(
            "Optional comma-separated leaf=alg_id overrides for small-M "
            "cuSPARSELt alg selection, for example down_proj=1."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-dense-verify-max-rows",
        type=int,
        default=0,
        help=(
            "Base-only diagnostic ablation: keep a dense copy and use dense "
            "Linear for base_only_24 verifier steps with at most this many "
            "scheduled rows. Default 0 disables the verify-step dense fallback."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-dense-verify-layer-ids",
        default="",
        help=(
            "Optional comma/range layer filter for "
            "--sr24-base-only-dense-verify-max-rows."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-dense-verify-layer-ids-by-leaf",
        default="",
        help=(
            "Optional leaf-specific dense verify fallback filter, e.g. "
            "'gate_up_proj=31;down_proj=31'."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-split",
        choices=["none", "up_sparse", "gate_sparse", "channel_pair"],
        default="none",
        help=(
            "SR24 base-only gate_up_proj ablation. up_sparse keeps the gate "
            "half dense and sparsifies only the up half; gate_sparse does the "
            "opposite. channel_pair keeps selected intermediate gate/up "
            "channel pairs dense and sparsifies the remaining pairs. Default "
            "none preserves the fused gate_up_proj behavior."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-channel-dense-fraction",
        type=float,
        default=0.125,
        help=(
            "For --sr24-gate-up-split channel_pair, fraction of intermediate "
            "gate/up channel pairs kept dense. The rest are converted to 2:4."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-channel-strategy",
        choices=["norm", "magnitude", "first", "last"],
        default="norm",
        help=(
            "For --sr24-gate-up-split channel_pair, choose which gate/up "
            "channel pairs remain dense."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-channel-fused-act",
        action="store_true",
        help=(
            "For --sr24-gate-up-split channel_pair, reassemble grouped "
            "gate/up activations and call the original activation instead of "
            "manual silu(gate) * up."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp",
        action="store_true",
        help=(
            "Experimental mixed-mask MLP-level routing path. It computes dense "
            "gate_up for residual rows, sparse gate_up/down for base rows, and "
            "assembles only the final hidden-size MLP output. Default off."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-down-linear",
        action="store_true",
        help=(
            "Experimental Linear-level down_proj routing path. It computes "
            "dense down_proj only for residual rows and sparse down_proj only "
            "for base rows, avoiding sparse-base work on rows that are later "
            "overwritten. Default off."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-reuse-base-output",
        action="store_true",
        help=(
            "Experimental row-routed MLP variant: compute sparse-base MLP for "
            "all rows and overwrite selected dense rows, avoiding per-step "
            "base-row complement construction. Default off."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-fixed-block-dense-fill",
        action="store_true",
        help=(
            "For fixed-prefix row-routed MLP, promote adjacent sparse/base "
            "rows into the dense branch until "
            "--sr24-row-routed-mlp-min-dense-rows is reached. This keeps the "
            "route table graph-stable and tests tile-fill effects; default "
            "off preserves older fixed-block semantics."
        ),
    )
    parser.add_argument(
        "--sr24-fixed-block-input-buffer",
        action="store_true",
        help=(
            "For fixed-prefix row-routed MLP, assemble dense/base branch inputs "
            "into reusable graph-stable temporary buffers instead of allocating "
            "new tensors through torch.cat/reshape. This is an explicit "
            "data-format/allocator-overhead ablation."
        ),
    )
    parser.add_argument(
        "--sr24-fixed-block-output-buffer",
        action="store_true",
        help=(
            "For fixed-prefix row-routed MLP, assemble the final dense/base "
            "branch output into a per-module reusable workspace. This is a "
            "narrow allocator/format ablation and is off by default."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-min-dense-rows",
        type=int,
        default=128,
        help=(
            "Minimum residual/dense rows required before the experimental "
            "row-routed MLP path is used. Smaller mixed steps fall back to the "
            "normal Linear-level residual path."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-min-dense-rows-by-leaf",
        default="",
        help=(
            "Optional per-leaf override for --sr24-row-routed-mlp-min-dense-rows, "
            "for example gate_up_proj=128;down_proj=64."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-max-dense-rows",
        type=int,
        default=0,
        help=(
            "Optional maximum residual/dense rows for the experimental "
            "row-routed MLP path. Values above this fall back to the normal "
            "Linear-level residual path. Use 0 to disable the upper guard."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-max-dense-rows-by-leaf",
        default="",
        help=(
            "Optional per-leaf override for --sr24-row-routed-mlp-max-dense-rows. "
            "Use 0 for a leaf to disable its upper guard."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-max-base-rows",
        type=int,
        default=0,
        help=(
            "Optional maximum base/sparse rows for the experimental "
            "row-routed MLP path. This avoids known-slow dynamic sparse MLP "
            "shapes where almost all rows are base rows. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-max-base-rows-by-leaf",
        default="",
        help=(
            "Optional per-leaf override for --sr24-row-routed-mlp-max-base-rows. "
            "Use this as a shape-aware planner guard for gate_up/down."
        ),
    )
    parser.add_argument(
        "--sr24-force-cudagraph-none-for-mixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force CUDA Graph NONE for mixed SR24 verify plans. This is the "
            "default correctness guard. Use "
            "--no-sr24-force-cudagraph-none-for-mixed only for static-mixed "
            "mask-buffer graph-safety ablations."
        ),
    )
    parser.add_argument(
        "--sr24-mask-buffer-capacity",
        type=int,
        default=16384,
        help="Capacity in tokens for the reusable SR24 residual-mask buffer.",
    )
    parser.add_argument(
        "--sr24-residual-bucket-size",
        type=int,
        default=0,
        help=(
            "Reduce-sync mixed-mask ablation for torch_sparse residuals. When "
            ">0, residual correction uses a fixed-size GPU top-k bucket from "
            "the residual mask. Default 0 disables it; values smaller than the "
            "true residual-row count are approximate."
        ),
    )
    parser.add_argument(
        "--sr24-residual-bucket-scale-by-active",
        action="store_true",
        help=(
            "Interpret --sr24-residual-bucket-size as a per-active-request "
            "budget in scheduler-built buckets. Default keeps the historical "
            "global bucket size."
        ),
    )
    parser.add_argument(
        "--sr24-residual-bucket-priority",
        action="store_true",
        help=(
            "When residual bucket size is positive, choose capped residual rows "
            "by SR24 priority scores instead of the bool mask's arbitrary top-k. "
            "Non-draft rows, missing-score rows, low-confidence draft rows, and "
            "early draft rows are prioritized in that order."
        ),
    )
    parser.add_argument(
        "--sr24-bonus-priority",
        type=float,
        default=4.0,
        help=(
            "Priority score assigned to speculative bonus/non-draft rows when "
            "--sr24-residual-bucket-priority is active. Lower values reserve "
            "capped buckets for draft verification rows first."
        ),
    )
    parser.add_argument(
        "--sr24-draft-position-priority-scale",
        type=float,
        default=0.0,
        help=(
            "Additional priority for earlier draft positions in capped "
            "residual buckets. Default 0 preserves the existing policy."
        ),
    )
    parser.add_argument(
        "--sr24-route-bucket-rows",
        action="store_true",
        help=(
            "For torch_sparse + dense_rows residual bucket ablations, route "
            "bucket rows directly to dense Linear and non-bucket rows to sparse "
            "base Linear instead of computing sparse base for every row first."
        ),
    )
    parser.add_argument(
        "--sr24-route-all-residual-rows",
        action="store_true",
        help=(
            "For torch_sparse + dense_rows residual ablations, route every "
            "residual row directly to dense Linear and route only non-residual "
            "rows to sparse base Linear. This preserves the selective mask "
            "exactly and avoids sparse base work on corrected rows."
        ),
    )
    parser.add_argument(
        "--sr24-route-all-skip-bucket",
        action="store_true",
        help=(
            "For route_all_residual_rows, skip residual-bucket construction and "
            "route the full residual/base split directly. This matches the "
            "throughput runner flag used by fixed-prefix down0-15 diagnostics."
        ),
    )
    parser.add_argument(
        "--sr24-direct-cpu-route-rows",
        action="store_true",
        help=(
            "For route_all_residual_rows, build residual/base row-index tensors "
            "directly from request metadata on CPU instead of calling CUDA "
            "nonzero() on the residual mask. This is a CPU-sync ablation for "
            "small K speculative batches."
        ),
    )
    parser.add_argument(
        "--sr24-route-reuse-base-output",
        action="store_true",
        help=(
            "For torch_sparse + dense_rows mixed-mask ablations, keep the full "
            "sparse base output that was already computed and run dense Linear "
            "only for residual rows, then overwrite those rows. This avoids the "
            "full dense+where path without splitting the base sparse GEMM."
        ),
    )
    parser.add_argument(
        "--sr24-route-contiguous-fastpath",
        action="store_true",
        help=(
            "Experimental routed dense_rows fast path: when routed dense rows "
            "are a contiguous prefix or suffix, compute dense and sparse "
            "slices directly and concatenate them instead of using "
            "index_select/index_copy assembly. Only affects explicit routed "
            "row ablations."
        ),
    )
    parser.add_argument(
        "--sr24-fixed-prefix-route-descriptor-only",
        action="store_true",
        help=(
            "For fixed-prefix route_all_residual_rows + row-routed MLP, pass "
            "only the compact fixed-prefix route descriptor through the verify "
            "plan instead of residual/base row-index tensors. This is an "
            "explicit data-format optimization ablation for descriptor-safe "
            "fixed-prefix plans."
        ),
    )
    parser.add_argument(
        "--sr24-scheduler-policy-path",
        default="",
        help=(
            "Optional path to scheduler_policy.json generated by "
            "derive_sr24_packed_mlp_planner.py. When set, descriptor-safe "
            "fixed-block row-routed MLP uses the operator-local policy for "
            "mixed-vs-dense fallback decisions."
        ),
    )
    parser.add_argument(
        "--sr24-scheduler-policy-near-full-tolerance",
        type=int,
        default=0,
        help=(
            "When >0, allow active verifier-block counts within this distance "
            "below a larger scheduler_policy batch to reuse that larger "
            "policy. Pair with --sr24-fixed-block-capacity-padding to test "
            "near-full fixed-capacity mixed MLP plans."
        ),
    )
    parser.add_argument(
        "--sr24-fixed-block-capacity-padding",
        action="store_true",
        help=(
            "Pad near-full fixed-block dense/base MLP inputs to the matched "
            "scheduler-policy capacity with dummy rows and assemble only real "
            "rows. This is an explicit system-operator ablation for avoiding "
            "dense fallback on near-full verifier blocks."
        ),
    )
    parser.add_argument(
        "--sr24-fixed-block-capacity-zero-dummy",
        action="store_true",
        help=(
            "When fixed-block capacity padding is enabled, zero-fill padded "
            "dummy rows before dense/sparse MLP branches. The default leaves "
            "dummy rows undefined because their outputs are discarded and "
            "row-wise GEMMs do not mix rows."
        ),
    )
    parser.add_argument(
        "--sr24-scheduler-policy-dense-bypass",
        action="store_true",
        help=(
            "When scheduler_policy.json selects dense fallback for an "
            "underfilled fixed-block mixed operator, bypass the SR24 MLP hook "
            "and run the original dense vLLM MLP."
        ),
    )
    parser.add_argument(
        "--sr24-scheduler-policy-allow-serial-packed-parallel",
        action="store_true",
        help=(
            "Allow single-block packed_parallel policy rows to use the current "
            "serial fixed-block dense/sparse MLP path without Python stream "
            "overlap. This is a systems ablation; it is not a fused/grouped "
            "packed operator."
        ),
    )
    parser.add_argument(
        "--sr24-route-overlap-streams",
        action="store_true",
        help=(
            "Experimental routed dense_rows ablation: launch base 2:4 rows and "
            "important dense rows on separate CUDA streams before assembling "
            "the output. This is forced out of CUDA Graph mixed replay and is "
            "meant to test operator overlap."
        ),
    )
    parser.add_argument(
        "--sr24-route-dense-fallback-fraction",
        type=float,
        default=1.1,
        help=(
            "If residual rows cover at least this fraction of the current "
            "verify step, use the all-residual dense fastpath instead of mixed "
            "sparse/residual routing. For --sr24-route-all-residual-rows this "
            "also controls the split-route dense fallback. Values outside [0,1] "
            "disable the fallback. This is conservative for accuracy because it "
            "uses the dense target output for extra rows."
        ),
    )
    parser.add_argument(
        "--sr24-route-min-dense-rows",
        type=int,
        default=0,
        help=(
            "For SR24 routed dense_rows paths, skip split routing unless at "
            "least this many residual/dense rows are present. Default 0 keeps "
            "the historical behavior."
        ),
    )
    parser.add_argument(
        "--sr24-route-min-base-rows",
        type=int,
        default=0,
        help=(
            "For SR24 routed dense_rows paths, use a conservative full-dense "
            "fallback when fewer than this many base/sparse rows remain. "
            "Default 0 keeps the historical behavior."
        ),
    )
    parser.add_argument(
        "--sr24-route-min-base-rows-by-leaf",
        default="",
        help=(
            "Optional per-leaf override for --sr24-route-min-base-rows, e.g. "
            "'gate_up_proj=384;down_proj=128'. This avoids underfilled "
            "gate/up sparse branches without disabling cheaper down-proj "
            "splits."
        ),
    )
    parser.add_argument(
        "--sr24-route-max-dense-fraction",
        type=float,
        default=1.1,
        help=(
            "For SR24 routed dense_rows paths, use a conservative full-dense "
            "fallback when residual/dense rows exceed this fraction of the "
            "step. Values outside [0,1] disable this guard."
        ),
    )
    parser.add_argument(
        "--sr24-adaptive-dense-fallback",
        action="store_true",
        help=(
            "Enable shape-aware SR24 dense fallback for torch_sparse+dense_rows "
            "selective runs. It uses dense TLM Linear when the current leaf, "
            "row count, and residual bucket size match microbench-proven slow "
            "mixed sparse+correction cases. Accuracy is conservative because "
            "fallback corrects extra rows instead of leaving them base-only."
        ),
    )
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-no-residual-only",
        dest="sr24_adaptive_dense_fallback_no_residual_only",
        action="store_true",
        help=(
            "Restrict adaptive dense fallback to no-residual sparse-only "
            "steps. This isolates small sparse-base GEMM overhead without "
            "turning mixed residual-correction steps into full dense Linear."
        ),
    )
    parser.add_argument(
        "--no-sr24-adaptive-dense-fallback-no-residual-only",
        dest="sr24_adaptive_dense_fallback_no_residual_only",
        action="store_false",
        help="Allow adaptive dense fallback on both no-residual and mixed steps.",
    )
    parser.set_defaults(sr24_adaptive_dense_fallback_no_residual_only=False)
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-small-rows",
        type=int,
        default=128,
        help="Row-count cutoff for SR24 adaptive dense fallback small-shape rules.",
    )
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-gate-up-fraction",
        type=float,
        default=0.10,
        help=(
            "For gate_up_proj above the small-row cutoff, dense fallback when "
            "candidate dense/residual rows cover at least this fraction."
        ),
    )
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-down-fraction",
        type=float,
        default=0.25,
        help=(
            "For down_proj above the small-row cutoff, dense fallback when "
            "candidate dense/residual rows cover at least this fraction."
        ),
    )
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-small-down-no-residual",
        dest="sr24_adaptive_dense_fallback_small_down_no_residual",
        action="store_true",
        help=(
            "Enable the small-row down_proj dense fallback when the current "
            "step has no residual rows."
        ),
    )
    parser.add_argument(
        "--no-sr24-adaptive-dense-fallback-small-down-no-residual",
        dest="sr24_adaptive_dense_fallback_small_down_no_residual",
        action="store_false",
        help=(
            "Disable the small-row down_proj dense fallback when the current "
            "step has no residual rows. This is on by default because the "
            "64-row down_proj microbench shows sparse base slower than dense."
        ),
    )
    parser.set_defaults(sr24_adaptive_dense_fallback_small_down_no_residual=True)
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-small-gate-up-no-residual",
        dest="sr24_adaptive_dense_fallback_small_gate_up_no_residual",
        action="store_true",
        help=(
            "Enable the small-row gate_up_proj dense fallback when the current "
            "step has no residual rows. This is off by default because it "
            "trades away sparse-base work on low-row steps."
        ),
    )
    parser.add_argument(
        "--no-sr24-adaptive-dense-fallback-small-gate-up-no-residual",
        dest="sr24_adaptive_dense_fallback_small_gate_up_no_residual",
        action="store_false",
        help=(
            "Disable the small-row gate_up_proj dense fallback when the "
            "current step has no residual rows."
        ),
    )
    parser.set_defaults(sr24_adaptive_dense_fallback_small_gate_up_no_residual=False)
    parser.add_argument(
        "--sr24-triton-route-assembly",
        action="store_true",
        help=(
            "Use the experimental Triton routed output assembly kernel for "
            "--sr24-route-bucket-rows. Default is the PyTorch index_copy_ path."
        ),
    )
    parser.add_argument(
        "--sr24-triton-bucket-override",
        action="store_true",
        help=(
            "For torch_sparse + dense_rows residual bucket ablations, run sparse "
            "base on all rows and overwrite active bucket rows with dense output "
            "using a Triton kernel instead of base-row gather, delta, and index_add_."
        ),
    )
    parser.add_argument(
        "--sr24-triton-bucket-dense-gemm",
        action="store_true",
        help=(
            "Experimental fused correction prototype for torch_sparse + "
            "dense_rows residual buckets: compute bucket-row dense GEMM in "
            "Triton and scatter directly into the sparse base output."
        ),
    )
    parser.add_argument(
        "--sr24-triton-bucket-scatter",
        action="store_true",
        help=(
            "Use a Triton active-row scatter after the normal dense bucket GEMM. "
            "This keeps cuBLAS dense numerics while avoiding PyTorch "
            "index_select/where/index_copy_ writeback overhead."
        ),
    )
    parser.add_argument(
        "--sr24-triton-bucket-dense-block-m",
        type=int,
        default=16,
        help="Triton block_m for --sr24-triton-bucket-dense-gemm.",
    )
    parser.add_argument(
        "--sr24-triton-bucket-dense-block-n",
        type=int,
        default=32,
        help="Triton block_n for --sr24-triton-bucket-dense-gemm.",
    )
    parser.add_argument(
        "--sr24-triton-bucket-dense-block-k",
        type=int,
        default=128,
        help="Triton block_k for --sr24-triton-bucket-dense-gemm.",
    )
    parser.add_argument(
        "--sr24-bucket-dense-copy",
        action="store_true",
        help=(
            "Dense_rows bucket correction ablation: after the bucket dense "
            "GEMM, overwrite every bucket row with dense output via index_copy_ "
            "instead of gather-base/delta/index_add_. Padded bucket rows become "
            "dense, which is quality-conservative but may change acceptance."
        ),
    )
    parser.add_argument(
        "--sr24-bucket-dense-copy-active-only",
        action="store_true",
        help=(
            "When --sr24-bucket-dense-copy is enabled, preserve bucket rows "
            "whose runtime bucket value is inactive. This is a graph-safety "
            "ablation for static bucket replay."
        ),
    )
    parser.add_argument(
        "--sr24-bucket-dense-compute-active-only",
        action="store_true",
        help=(
            "When both --sr24-bucket-dense-copy and "
            "--sr24-bucket-dense-copy-active-only are enabled, gather and run "
            "dense GEMM only for active bucket rows. This avoids dense work for "
            "unimportant rows already handled by the sparse base path, but uses "
            "variable active-row shapes."
        ),
    )
    parser.add_argument(
        "--sr24-bucket-dense-active-mask-fused",
        action="store_true",
        help=(
            "For active-only bucket correction, keep the fixed bucket shape "
            "and let the Triton fused GEMM+scatter use bucket_values as an "
            "active mask. This avoids Python nonzero/index_select compression "
            "and can remain CUDA-Graph compatible."
        ),
    )
    parser.add_argument(
        "--sr24-disable-runtime-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "CPU/Python overhead ablation: disable SR24 runtime verify summary "
            "and CUDA graph counter updates."
        ),
    )
    parser.add_argument(
        "--sr24-selective-residual-policy",
        choices=[
            "critical_prefix",
            "all_if_any_low",
            "batch_all_if_any_low",
            "low_confidence",
            "high_confidence",
            "prefix_confidence",
            "fixed_prefix",
        ],
        default="critical_prefix",
        help=(
            "Selective SR24 draft-row policy. critical_prefix corrects the "
            "verifier-logit prefix through the first low-confidence draft "
            "token; all_if_any_low corrects all draft rows in a request step "
            "if any draft token is low-confidence; batch_all_if_any_low "
            "corrects every scheduled row in the whole verify step if any "
            "draft token in the batch is low-confidence; low_confidence/"
            "high_confidence are per-row policies. low_confidence marks only "
            "missing or low-confidence draft rows residual and can use the "
            "batched mask builder for low-overhead speed gates. "
            "prefix_confidence corrects rows while the DLM selected-token "
            "prefix product remains above --sr24-prefix-threshold. "
            "fixed_prefix corrects only the first "
            "--sr24-selective-min-prefix-residual draft rows plus the bonus "
            "row and avoids DLM selected-probability score routing."
        ),
    )
    parser.add_argument(
        "--sr24-prefix-threshold",
        type=float,
        default=-1.0,
        help=(
            "Threshold for --sr24-selective-residual-policy=prefix_confidence. "
            "Negative means reuse --sr24-threshold."
        ),
    )
    parser.add_argument(
        "--sr24-selective-extra-after-low",
        type=int,
        default=0,
        help=(
            "For critical_prefix selective SR24, additionally correct this many "
            "draft rows after the first low-confidence row. Default 0 preserves "
            "the original critical_prefix behavior."
        ),
    )
    parser.add_argument(
        "--sr24-selective-min-prefix-residual",
        type=int,
        default=0,
        help=(
            "Selective SR24 accuracy diagnostic. Force the first N draft rows "
            "of every speculative verify step through residual correction, then "
            "apply the selected confidence policy to the remaining rows. "
            "Default 0 preserves the existing policy."
        ),
    )
    parser.add_argument(
        "--sr24-selective-max-residual-draft-rows",
        type=int,
        default=0,
        help=(
            "Selective SR24 speed/quality diagnostic. When >0, cap each "
            "request's residual-corrected draft rows after preserving "
            "--sr24-selective-min-prefix-residual; low_confidence keeps its "
            "existing low-confidence/risk selection, while prefix-style "
            "policies keep the earliest selected residual rows. Bonus/non-draft "
            "rows still follow --sr24-selective-non-draft-policy. Default 0 "
            "leaves the selected policy uncapped."
        ),
    )
    parser.add_argument(
        "--sr24-low-confidence-cap-by-risk",
        action="store_true",
        help=(
            "When low_confidence is capped by "
            "--sr24-selective-max-residual-draft-rows, select the lowest-"
            "confidence draft rows within the request instead of the first "
            "low-confidence rows. This is a selective residual importance "
            "routing ablation."
        ),
    )
    parser.add_argument(
        "--sr24-early-dense-tokens",
        type=int,
        default=0,
        help=(
            "Selective SR24 accuracy guard. When >0, draft and bonus rows whose "
            "request generated length is still inside this prefix are forced "
            "through the residual/dense-corrected path. Default 0 disables the "
            "guard and avoids the extra generated-length context."
        ),
    )
    parser.add_argument(
        "--sr24-target-leafs",
        default="",
        help=(
            "Comma-separated SR24 target Linear leaf names. Empty means all "
            "supported Llama leaves: qkv_proj,o_proj,gate_up_proj,down_proj. "
            "Use this for accuracy/speed ablations such as attention-only "
            "or MLP-only sparse replacement."
        ),
    )
    parser.add_argument(
        "--sr24-residual-target-leafs",
        default="",
        help=(
            "Comma-separated subset of SR24 target Linear leaf names that keep "
            "the residual correction path. Empty means the same set as "
            "--sr24-target-leafs; use 'none' for pure base-only targets."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-layer-ids",
        default="",
        help=(
            "Comma-separated layer ids/ranges for target leafs that do not "
            "keep residual correction. Empty allows those base-only leafs in "
            "all layers, e.g. '0,1,30-31' keeps other layers dense."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-layer-ids-by-leaf",
        default="",
        help=(
            "Semicolon-separated per-leaf layer ids/ranges for target leafs "
            "without residual correction, e.g. "
            "'gate_up_proj=31;down_proj=30-31'. Overrides "
            "--sr24-base-only-layer-ids for listed leafs; unlisted base-only "
            "leafs are skipped when this option is non-empty."
        ),
    )
    parser.add_argument(
        "--sr24-residual-layer-ids-by-leaf",
        default="",
        help=(
            "Semicolon-separated per-leaf layer ids/ranges for residual target "
            "leafs that should use dynamic token-level residual correction, "
            "e.g. 'gate_up_proj=31;down_proj=31'. When set, residual target "
            "leafs not listed here can use per-module densefastpath, and "
            "unlisted layers of listed leafs are left dense."
        ),
    )
    parser.add_argument(
        "--sr24-runtime-base-only-layer-ids-by-leaf",
        default="",
        help=(
            "Semicolon-separated per-leaf layer ids/ranges that should skip "
            "runtime residual correction while still being attached with the "
            "same dense_rows SR24 storage as other layers, e.g. "
            "'gate_up_proj=26-31;down_proj=26-31'. This keeps the vLLM compile "
            "data format uniform and is off by default."
        ),
    )
    parser.add_argument(
        "--sr24-residual-out-chunk",
        type=int,
        default=4096,
        help=(
            "Output-channel chunk size for materializing compressed SR24 "
            "residual weights at runtime. Set <=0 to restore full residual "
            "dense materialization for ablation."
        ),
    )
    parser.add_argument(
        "--sr24-cache-compressed-residual-weight",
        action="store_true",
        help=(
            "Diagnostic compressed_dense optimization: cache the dense GPU "
            "residual tensor after first materialization instead of rebuilding "
            "it on every Linear call."
        ),
    )
    parser.add_argument(
        "--sr24-prewarm-compressed-residual-weight",
        action="store_true",
        help=(
            "When compressed residual weight caching is enabled, materialize "
            "the dense GPU residual tensor during SR24 model attach instead "
            "of the first Linear call. Used automatically for no-fastpath "
            "all_corrected_24 compressed_dense unless disabled by "
            "--no-sr24-auto-compressed-residual-fastpath."
        ),
    )
    parser.add_argument(
        "--sr24-auto-compressed-residual-fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For all_corrected_24 with torch_sparse/compressed_dense and the "
            "dense fastpath disabled, automatically use the best current "
            "compressed residual path: cache the GPU residual weight, prewarm "
            "it at model attach, and set residual_out_chunk=0. Use "
            "--no-sr24-auto-compressed-residual-fastpath to reproduce the "
            "older chunked materialization ablation."
        ),
    )
    parser.add_argument(
        "--sr24-compressed-residual-triton",
        action="store_true",
        help=(
            "Experimental diagnostic compressed_dense residual path: compute "
            "the residual-only matmul directly from GPU-resident compressed "
            "2:4 values with a Triton kernel. Current refresh measurements "
            "show this is slower than materializing the residual weight and "
            "running torch GEMM, so do not use it as the all_corrected_24 "
            "speed path."
        ),
    )
    parser.add_argument(
        "--sr24-compressed-residual-block-m",
        type=int,
        default=32,
        help=(
            "Triton block_m for --sr24-compressed-residual-triton. The default "
            "is the best local diagnostic setting from the 2026-06-28 "
            "compressed residual sweep."
        ),
    )
    parser.add_argument(
        "--sr24-compressed-residual-block-n",
        type=int,
        default=128,
        help=(
            "Triton block_n for --sr24-compressed-residual-triton. This path "
            "is still diagnostic; materialized residual GEMM remains faster."
        ),
    )
    parser.add_argument(
        "--sr24-compressed-residual-block-g",
        type=int,
        default=16,
        help="Triton group-block size for --sr24-compressed-residual-triton.",
    )
    parser.add_argument(
        "--sr24-extract-chunk-rows",
        type=int,
        default=128,
        help=(
            "Output-row chunk size used only while extracting compressed SR24 "
            "residual values during model loading."
        ),
    )
    parser.add_argument("--use-task-manifests", action="store_true")
    parser.add_argument("--manifest-size", type=int, default=0)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=EVAL_ROOT / "configs" / "task_manifests",
    )
    parser.add_argument(
        "--hf-cache-root",
        type=Path,
        default=EVAL_ROOT / "temp" / "hf_lm_eval_cache",
    )
    parser.add_argument(
        "--humaneval-sandbox",
        choices=["auto", "bwrap", "none"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--align-dense-baseline-to-sr24-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For paired accuracy reports, if dense_baseline is run together "
            "with an SR24 mode that must use --enforce-eager, also run "
            "dense_baseline with --enforce-eager. This avoids false paired "
            "regressions from comparing graph/compiled dense serving against "
            "eager SR24 serving."
        ),
    )
    parser.add_argument("--allow-unsafe-code", action="store_true")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--aggregate", action="store_true", default=True)
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sr24_preset_overrides = capture_sr24_preset_overrides(args, sys.argv[1:])
    apply_sr24_preset(args)
    restore_sr24_preset_overrides(args, sr24_preset_overrides)
    if getattr(args, "sr24_base_only_allow_compile", False):
        # This flag is meant to exercise the graph-capable base_only SR24
        # path. Without sr24_allow_cudagraph, command construction still adds
        # --enforce-eager and the accuracy run no longer matches throughput
        # base-only compile ablations.
        args.sr24_allow_cudagraph = True
    if args.sr24_breakdown:
        # Breakdown mode uses Python-side counters/locks and CUDA events for
        # diagnostics. Keep it eager even when a preset or explicit override
        # requested default vLLM compile; otherwise Dynamo can trace into the
        # diagnostic lock during profile-run.
        args.sr24_default_vllm_compile = False
        args.sr24_allow_cudagraph = False
        args.sr24_dynamic_auto_cudagraph = False
        args.sr24_force_cudagraph_none_for_mixed = True
        args.sr24_cudagraph_bucket = False
        args.vllm_compilation_config = ""
    validate_sr24_runtime_base_only_compile(args)
    args.sr24_gate_up_channel_dense_fraction = min(
        max(float(args.sr24_gate_up_channel_dense_fraction), 0.0), 1.0
    )
    args.sr24_extract_chunk_rows = max(1, int(args.sr24_extract_chunk_rows))
    args.sr24_mask_buffer_capacity = max(0, int(args.sr24_mask_buffer_capacity))
    args.sr24_route_min_dense_rows = max(0, int(args.sr24_route_min_dense_rows))
    args.sr24_route_min_base_rows = max(0, int(args.sr24_route_min_base_rows))
    args.sr24_selective_max_residual_draft_rows = max(
        0, int(args.sr24_selective_max_residual_draft_rows)
    )
    args.sr24_triton_bucket_dense_block_m = max(
        1, int(args.sr24_triton_bucket_dense_block_m)
    )
    args.sr24_triton_bucket_dense_block_n = max(
        1, int(args.sr24_triton_bucket_dense_block_n)
    )
    args.sr24_triton_bucket_dense_block_k = max(
        1, int(args.sr24_triton_bucket_dense_block_k)
    )
    args.sr24_compressed_residual_block_m = max(
        1, int(args.sr24_compressed_residual_block_m)
    )
    args.sr24_compressed_residual_block_n = max(
        1, int(args.sr24_compressed_residual_block_n)
    )
    args.sr24_compressed_residual_block_g = max(
        1, int(args.sr24_compressed_residual_block_g)
    )
    run(args)


if __name__ == "__main__":
    main()
