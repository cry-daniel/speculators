#!/usr/bin/env python3
"""Run strict same-K dense EAGLE3 versus SpecLink LM-eval matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EVAL_ROOT.parents[2]
LM_EVAL_RUNNER = SCRIPT_DIR / "run_lm_eval_accuracy.py"
DEFAULT_MODELS = ("qwen3_8b", "llama3_1_8b")
DEFAULT_TASKS = ("gsm8k_cot", "math_reasoning")
DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
RUNTIME_IMPLEMENTATION_FILES = (
    "examples/evaluate/eval-guidellm/eval_tasks/gsm8k_local.yaml",
    "examples/evaluate/eval-guidellm/eval_tasks/math_reasoning.yaml",
    "examples/evaluate/eval-guidellm/eval_tasks/math_reasoning_utils.py",
    "examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.py",
    "examples/evaluate/eval-guidellm/scripts/run_speclink_system_matrix.py",
    "examples/evaluate/eval-guidellm/scripts/token_dense_methods.py",
    "vllm/vllm/forward_context.py",
    "vllm/vllm/model_executor/layers/batch_invariant.py",
    "vllm/vllm/model_executor/models/llama.py",
    "vllm/vllm/model_executor/models/qwen2.py",
    "vllm/vllm/model_executor/models/qwen3.py",
    "vllm/vllm/speclink_kernel/__init__.py",
    "vllm/vllm/speclink_kernel/cutlass_backend.py",
    "vllm/vllm/speclink_kernel/cutlass_sparse24_linear.cu",
    "vllm/vllm/speclink_kernel/pack.py",
    "vllm/vllm/speclink_linear.py",
    "vllm/vllm/speclink_mlp.py",
    "vllm/vllm/speclink_structured_24.py",
    "vllm/vllm/speclink_token_dense.py",
    "vllm/vllm/v1/core/sched/async_scheduler.py",
    "vllm/vllm/v1/core/sched/scheduler.py",
    "vllm/vllm/v1/cudagraph_dispatcher.py",
    "vllm/vllm/v1/sample/rejection_sampler.py",
    "vllm/vllm/v1/spec_decode/llm_base_proposer.py",
    "vllm/vllm/v1/worker/gpu/sample/logprob.py",
    "vllm/vllm/v1/worker/gpu_model_runner.py",
)


@dataclass(frozen=True)
class RunCase:
    phase: str
    model: str
    task: str
    batch_size: int
    k: int
    repeat: int
    variant: str

    @property
    def key(self) -> tuple[str, str, str, int, int, int, str]:
        return (
            self.phase,
            self.model,
            self.task,
            self.batch_size,
            self.k,
            self.repeat,
            self.variant,
        )


DEFAULT_CANDIDATE_CONFIG: dict[str, Any] = {
    "projection_policy": "all",
    "mixed_projection_policy": "all",
    "mixed_layers": "all",
    "mlp_static_layers": "",
    "o_sparse_layers": "all",
    "linear_strategy": "split_dense_sparse",
    "mlp_strategy": "linear",
    "fused_batch_mlp": True,
    "inline_swiglu_mlp": False,
    "full_sparse_override_mlp": False,
    "full_sparse_override_mlp_min_rows": 224,
    "routed_swiglu_mlp": False,
    "sparse_gate_dense_down": False,
    "direct_gate_residual_epilogue": True,
    "gate_down_gather_epilogue": False,
    "direct_store_gate_up": False,
    "cutlass_down_fp16": False,
    "contiguous_down_input": True,
    "uninitialized_routed_workspace": True,
    "paired_persistent_gate": True,
    "paired_gate_shape_tuning": False,
    "paired_fused_gate_epilogue": False,
    "paired_persistent_down": False,
    "paired_gather_down": False,
    "paired_inplace_down": False,
    "paired_inplace_o": False,
    "parallel_splitk_down": False,
    "single_launch_splitk_down": False,
    "tiled_indexed_splitk_down": False,
    "indexed_down_epilogue": False,
    "indexed_output_epilogue": False,
    "qkv_heterogeneous_routing": False,
    "qkv_heterogeneous_max_rows": 704,
    "qkv_paired_routing": False,
    "qkv_paired_max_rows": 704,
    "qkv_active_wave_c12": False,
    "qkv_vec4_postop": False,
    "qkv_fused_epilogue": False,
    "qkv_direct_cache": False,
    "parallel_mixed_override": True,
    "layer_policy": "keep_first_last",
    "keep_n": 2,
    "dense_layers": "",
    "attention_dense_layers": "",
    "gate_up_dense_layers": "",
    "down_dense_layers": "",
    "dense_ratio": 0.125,
    "dense_min_per_request": 1,
    "dense_cap": -1,
    "dense_selection": "balanced_confidence",
    "adaptive_dense_max_requests": 1,
    "batch_confidence_threshold": 0.364,
    "batch_route_block_steps": 1,
    "batch_route_initial_credit": 0.0,
    "balanced_start_position": 0,
    "sparse_bonus": False,
    "sparse_unscored_decode": False,
    "cudagraph_mode": "full_decode_only",
    "compilation_mode": "vllm_compile",
    "sparse_output_mode": "contiguous",
    "sparse_accumulator": "fp16_qkv_gate",
    "sparse_value_scale": 1.0,
    "gate_up_value_scale": 1.0,
    "gate_up_hybrid": "none",
    "group_reconstruction": False,
    "group_covariance_cache": "",
    "row_scale_mode": "cache",
    "variance_scale_projection_policy": "all",
    "row_scale_max": 1.25,
    "mask_method": "wanda",
    "mask_root": "",
    "release_dense_weights": False,
    "retain_dense_weight": "none",
    "flashinfer_autotune": True,
    "production_fast": True,
}


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in csv_strings(value)]


def parse_variants(value: str) -> list[str]:
    variants = csv_strings(value)
    if not variants or any(item not in {"dense", "speclink"} for item in variants):
        raise argparse.ArgumentTypeError("variants must contain dense and/or speclink")
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("variants must not contain duplicates")
    return variants


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_candidate_configs(path: Path | None, models: list[str]) -> dict[str, dict[str, Any]]:
    raw: dict[str, Any] = {}
    if path is not None:
        loaded = read_json(path)
        if not isinstance(loaded, dict):
            raise ValueError("candidate config must be a JSON object")
        raw = loaded
    defaults = dict(DEFAULT_CANDIDATE_CONFIG)
    if isinstance(raw.get("defaults"), dict):
        defaults.update(raw["defaults"])
    per_model = raw.get("models", raw)
    configs: dict[str, dict[str, Any]] = {}
    for model in models:
        config = dict(defaults)
        model_config = per_model.get(model, {}) if isinstance(per_model, dict) else {}
        if isinstance(model_config, dict):
            config.update(model_config)
        configs[model] = config
    return configs


def config_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def implementation_id() -> str:
    digest = hashlib.sha1()
    for relative_path in RUNTIME_IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"runtime implementation file is missing: {path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def variant_config_id(
    variant: str,
    config: dict[str, Any],
    runtime_id: str,
) -> str:
    prefix = "dense" if variant == "dense" else config_id(config)
    return f"{prefix}_{runtime_id}"


def service_config_id(
    args: argparse.Namespace,
    case: RunCase,
    config: dict[str, Any],
    runtime_id: str,
) -> str:
    limit = (
        args.performance_limit
        if case.phase == "performance"
        else args.accuracy_limit
    )
    payload = {
        "runtime_id": runtime_id,
        "batch_size": case.batch_size,
        "num_concurrent": case_num_concurrent(args, case),
        "max_num_seqs": case.batch_size,
        "k": case.k,
        "limit": str(limit),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_context_length": args.max_context_length,
        "max_new_tokens": args.max_new_tokens,
        "server_warmup_batches": args.server_warmup_batches,
        "server_warmup_max_tokens": args.server_warmup_max_tokens,
        "server_warmup_lm_eval_batches": (
            args.server_warmup_lm_eval_batches
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend.lower(),
        "seed": args.seed,
        "cudagraph_mode": config["cudagraph_mode"],
        "compilation_mode": config["compilation_mode"],
        "flashinfer_autotune": bool(config.get("flashinfer_autotune", True)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def gpu_state() -> tuple[int, int] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    first = completed.stdout.strip().splitlines()[0]
    memory, utilization = (int(part.strip()) for part in first.split(",")[:2])
    return memory, utilization


def wait_for_gpu(args: argparse.Namespace) -> None:
    if args.skip_gpu_wait:
        return
    while True:
        state = gpu_state()
        if state is None:
            raise RuntimeError("nvidia-smi failed while waiting for the GPU")
        memory_mb, utilization = state
        if memory_mb <= args.gpu_idle_memory_mb and utilization <= args.gpu_idle_utilization:
            return
        print(
            f"GPU busy: memory={memory_mb} MiB utilization={utilization}%; "
            f"checking again in {args.gpu_check_interval_s}s",
            flush=True,
        )
        time.sleep(args.gpu_check_interval_s)


def case_run_dir(
    output_root: Path,
    case: RunCase,
    config: dict[str, Any],
    runtime_id: str,
) -> Path:
    current_config_id = variant_config_id(case.variant, config, runtime_id)
    variant_dir = (
        current_config_id
        if case.variant == "dense"
        else f"speclink_{current_config_id}"
    )
    return (
        output_root
        / "runs"
        / case.phase
        / case.model
        / case.task
        / f"bs{case.batch_size}_k{case.k}"
        / f"repeat{case.repeat}"
        / variant_dir
    )


def candidate_args(config: dict[str, Any]) -> list[str]:
    values = [
        "--token-dense-projection-policy",
        str(config["projection_policy"]),
        "--token-dense-linear-strategy",
        str(config["linear_strategy"]),
        "--token-dense-mixed-projection-policy",
        str(config["mixed_projection_policy"]),
        "--token-dense-mixed-layers",
        str(config["mixed_layers"]),
        "--token-dense-mlp-static-layers",
        str(config["mlp_static_layers"]),
        "--token-dense-o-sparse-layers",
        str(config["o_sparse_layers"]),
        "--token-dense-mlp-strategy",
        str(config["mlp_strategy"]),
        "--token-dense-layer-policy",
        str(config["layer_policy"]),
        "--token-dense-keep-n",
        str(config["keep_n"]),
        "--token-dense-dense-layers",
        str(config["dense_layers"]),
        "--token-dense-attention-dense-layers",
        str(config["attention_dense_layers"]),
        "--token-dense-gate-up-dense-layers",
        str(config["gate_up_dense_layers"]),
        "--token-dense-down-dense-layers",
        str(config["down_dense_layers"]),
        "--token-dense-dense-ratio",
        str(config["dense_ratio"]),
        "--token-dense-dense-min-per-request",
        str(config["dense_min_per_request"]),
        "--token-dense-dense-cap",
        str(config["dense_cap"]),
        "--token-dense-dense-selection",
        str(config["dense_selection"]),
        "--token-dense-batch-confidence-threshold",
        str(config["batch_confidence_threshold"]),
        "--token-dense-adaptive-dense-max-requests",
        str(config["adaptive_dense_max_requests"]),
        "--token-dense-batch-route-block-steps",
        str(config["batch_route_block_steps"]),
        "--token-dense-batch-route-initial-credit",
        str(config["batch_route_initial_credit"]),
        "--token-dense-balanced-start-position",
        str(config["balanced_start_position"]),
        "--token-dense-sparse-output-mode",
        str(config["sparse_output_mode"]),
        "--token-dense-sparse-accumulator",
        str(config["sparse_accumulator"]),
        "--token-dense-sparse-value-scale",
        str(config["sparse_value_scale"]),
        "--token-dense-gate-up-value-scale",
        str(config["gate_up_value_scale"]),
        "--token-dense-gate-up-hybrid",
        str(config["gate_up_hybrid"]),
        "--token-dense-row-scale-mode",
        str(config["row_scale_mode"]),
        "--token-dense-variance-scale-projection-policy",
        str(config["variance_scale_projection_policy"]),
        "--token-dense-row-scale-max",
        str(config["row_scale_max"]),
        "--token-dense-mask-method",
        str(config["mask_method"]),
    ]
    if bool(config.get("production_fast", True)):
        values.append("--production-fast")
    if bool(config.get("fused_batch_mlp", False)):
        values.append("--token-dense-fused-batch-mlp")
    if bool(config.get("inline_swiglu_mlp", False)):
        values.append("--token-dense-inline-swiglu-mlp")
    if bool(config.get("full_sparse_override_mlp", False)):
        values.extend(
            [
                "--token-dense-full-sparse-override-mlp",
                "--token-dense-full-sparse-override-mlp-min-rows",
                str(config.get("full_sparse_override_mlp_min_rows", 224)),
            ]
        )
    if bool(config.get("routed_swiglu_mlp", False)):
        values.append("--token-dense-routed-swiglu-mlp")
    if bool(config.get("sparse_gate_dense_down", False)):
        values.append("--token-dense-sparse-gate-dense-down")
    if not bool(config.get("direct_gate_residual_epilogue", True)):
        values.append("--no-token-dense-direct-gate-residual-epilogue")
    if bool(config.get("gate_down_gather_epilogue", False)):
        values.append("--token-dense-gate-down-gather-epilogue")
    if bool(config.get("direct_store_gate_up", False)):
        values.append("--token-dense-direct-store-gate-up")
    if bool(config.get("cutlass_down_fp16", False)):
        values.append("--token-dense-cutlass-down-fp16")
    if not bool(config.get("contiguous_down_input", True)):
        values.append("--no-token-dense-contiguous-down-input")
    if not bool(config.get("uninitialized_routed_workspace", True)):
        values.append("--no-token-dense-uninitialized-routed-workspace")
    if not bool(config.get("paired_persistent_gate", True)):
        values.append("--no-token-dense-paired-persistent-gate")
    if bool(config.get("paired_gate_shape_tuning", False)):
        values.append("--token-dense-paired-gate-shape-tuning")
    if bool(config.get("paired_fused_gate_epilogue", False)):
        values.append("--token-dense-paired-fused-gate-epilogue")
    if bool(config.get("paired_persistent_down", False)):
        values.append("--token-dense-paired-persistent-down")
    if bool(config.get("paired_gather_down", False)):
        values.append("--token-dense-paired-gather-down")
    if bool(config.get("paired_inplace_down", False)):
        values.append("--token-dense-paired-inplace-down")
    if bool(config.get("paired_inplace_o", False)):
        values.append("--token-dense-paired-inplace-o")
    if bool(config.get("parallel_splitk_down", False)):
        values.append("--token-dense-parallel-splitk-down")
    if bool(config.get("single_launch_splitk_down", False)):
        values.append("--token-dense-single-launch-splitk-down")
    if bool(config.get("tiled_indexed_splitk_down", False)):
        values.append("--token-dense-tiled-indexed-splitk-down")
    if bool(config.get("indexed_down_epilogue", False)):
        values.append("--token-dense-indexed-down-epilogue")
    if bool(config.get("indexed_output_epilogue", False)):
        values.append("--token-dense-indexed-output-epilogue")
    if bool(config.get("qkv_heterogeneous_routing", False)):
        values.extend(
            [
                "--token-dense-qkv-heterogeneous-routing",
                "--token-dense-qkv-heterogeneous-max-rows",
                str(config.get("qkv_heterogeneous_max_rows", 704)),
            ]
        )
    if bool(config.get("qkv_paired_routing", False)):
        values.extend(
            [
                "--token-dense-qkv-paired-routing",
                "--token-dense-qkv-paired-max-rows",
                str(config.get("qkv_paired_max_rows", 704)),
            ]
        )
    if bool(config.get("qkv_active_wave_c12", False)):
        values.append("--token-dense-qkv-active-wave-c12")
    if bool(config.get("qkv_vec4_postop", False)):
        values.append("--token-dense-qkv-vec4-postop")
    if bool(config.get("qkv_fused_epilogue", False)):
        values.append("--token-dense-qkv-fused-epilogue")
    if bool(config.get("qkv_direct_cache", False)):
        values.append("--token-dense-qkv-direct-cache")
    if bool(config.get("group_reconstruction", False)):
        values.append("--token-dense-group-reconstruction")
    if str(config.get("group_covariance_cache", "")).strip():
        values.extend(
            [
                "--token-dense-group-covariance-cache",
                str(config["group_covariance_cache"]),
            ]
        )
    if str(config.get("mask_root", "")).strip():
        values.extend(
            [
                "--token-dense-mask-root",
                str(config["mask_root"]),
            ]
        )
    if not bool(config.get("parallel_mixed_override", True)):
        values.append("--no-token-dense-parallel-mixed-override")
    if bool(config.get("sparse_bonus", False)):
        values.append("--token-dense-sparse-bonus")
    if bool(config.get("sparse_unscored_decode", False)):
        values.append("--token-dense-sparse-unscored-decode")
    if bool(config.get("release_dense_weights", False)):
        values.append("--token-dense-release-dense-weights")
    retain_dense_weight = str(config.get("retain_dense_weight", "none"))
    if retain_dense_weight != "none":
        values.extend(
            ["--token-dense-retain-dense-weight", retain_dense_weight]
        )
    if bool(config.get("flashinfer_autotune", True)):
        values.append("--token-dense-flashinfer-autotune")
    return values


def case_num_concurrent(args: argparse.Namespace, case: RunCase) -> int:
    return args.num_concurrent or case.batch_size


def build_command(
    args: argparse.Namespace,
    case: RunCase,
    run_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    is_performance = case.phase == "performance"
    limit = args.performance_limit if is_performance else args.accuracy_limit
    command = [
        sys.executable,
        "-u",
        str(LM_EVAL_RUNNER),
        "--mode",
        "eagle3_dense" if case.variant == "dense" else "token_dense_dynamic",
        "--models",
        case.model,
        "--task",
        case.task,
        "--output-dir",
        str(run_dir),
        "--num-spec-tokens",
        str(case.k),
        "--batch-size",
        str(case.batch_size),
        "--num-concurrent",
        str(case_num_concurrent(args, case)),
        "--max-num-seqs",
        str(case.batch_size),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-context-length",
        str(args.max_context_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--server-warmup-batches",
        str(args.server_warmup_batches),
        "--server-warmup-max-tokens",
        str(args.server_warmup_max_tokens),
        "--server-warmup-lm-eval-batches",
        str(args.server_warmup_lm_eval_batches),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--port-base",
        str(args.port_base),
        "--token-dense-cudagraph-mode",
        str(config["cudagraph_mode"]),
        "--token-dense-compilation-mode",
        str(config["compilation_mode"]),
        "--resume",
    ]
    if args.attention_backend.lower() != "auto":
        command.extend(["--attention-backend", args.attention_backend])
    if limit:
        command.extend(["--limit", str(limit)])
    if case.variant == "speclink":
        command.extend(candidate_args(config))
    return command


def result_row(run_dir: Path, mode: str) -> dict[str, str] | None:
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        return None
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") == mode:
                return row
    return None


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


MEASUREMENT_FIELDS = (
    "phase",
    "model",
    "task",
    "batch_size",
    "k",
    "repeat",
    "variant",
    "config_id",
    "service_config_id",
    "implementation_id",
    "status",
    "score",
    "samples",
    "request_elapsed_seconds",
    "output_tokens",
    "request_output_tokens_per_second",
    "run_dir",
    "command",
)


def load_measurements(path: Path) -> dict[tuple[str, str, str, int, int, int, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[tuple[str, str, str, int, int, int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["phase"],
                row["model"],
                row["task"],
                int(row["batch_size"]),
                int(row["k"]),
                int(row["repeat"]),
                row["variant"],
            )
            rows[key] = row
    return rows


def write_measurements(path: Path, rows: dict[tuple[Any, ...], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({field: rows[key].get(field, "") for field in MEASUREMENT_FIELDS})


def measurement_from_result(
    case: RunCase,
    run_dir: Path,
    command: list[str],
    config: dict[str, Any],
    current_service_config_id: str,
    runtime_id: str,
    returncode: int,
) -> dict[str, Any]:
    mode = "eagle3_dense" if case.variant == "dense" else "token_dense_dynamic"
    parsed = result_row(run_dir, mode) or {}
    status = parsed.get("status") or ("failed" if returncode else "missing_result")
    return {
        "phase": case.phase,
        "model": case.model,
        "task": case.task,
        "batch_size": case.batch_size,
        "k": case.k,
        "repeat": case.repeat,
        "variant": case.variant,
        "config_id": variant_config_id(case.variant, config, runtime_id),
        "service_config_id": current_service_config_id,
        "implementation_id": runtime_id,
        "status": status,
        "score": parsed.get("score", ""),
        "samples": parsed.get("samples", ""),
        "request_elapsed_seconds": parsed.get("request_elapsed_seconds", ""),
        "output_tokens": parsed.get("output_tokens", ""),
        "request_output_tokens_per_second": parsed.get(
            "request_output_tokens_per_second", ""
        ),
        "run_dir": str(run_dir),
        "command": shlex.join(command),
    }


def build_cases(args: argparse.Namespace) -> list[RunCase]:
    phases = ("performance", "accuracy") if args.phase == "both" else (args.phase,)
    cases: list[RunCase] = []
    point_index = 0
    for phase in phases:
        repeats = args.performance_repeats if phase == "performance" else 1
        batch_sizes = (
            args.batch_sizes
            if phase == "performance"
            else accuracy_batch_sizes(args)
        )
        for model in args.models:
            for task in args.tasks:
                for batch_size in batch_sizes:
                    for k in args.k_values:
                        point_index += 1
                        for repeat in range(1, repeats + 1):
                            variants = list(args.variants)
                            if len(variants) > 1 and (point_index + repeat) % 2:
                                variants.reverse()
                            for variant in variants:
                                cases.append(
                                    RunCase(
                                        phase=phase,
                                        model=model,
                                        task=task,
                                        batch_size=batch_size,
                                        k=k,
                                        repeat=repeat,
                                        variant=variant,
                                    )
                                )
    return cases


def accuracy_batch_sizes(args: argparse.Namespace) -> list[int]:
    """Return matched-load accuracy batch sizes.

    ``--accuracy-batch-size`` remains as a narrow compatibility override for
    old serial diagnostics. Strict final runs use ``--batch-sizes`` for both
    phases, or the explicit plural override.
    """

    if args.accuracy_batch_size is not None:
        return [args.accuracy_batch_size]
    if args.accuracy_batch_sizes is not None:
        return args.accuracy_batch_sizes
    return args.batch_sizes


COMPARISON_FIELDS = (
    "phase",
    "model",
    "task",
    "batch_size",
    "k",
    "repeat_count",
    "dense_score",
    "speclink_score",
    "accuracy_delta_pp",
    "dense_request_tok_s_median",
    "speclink_request_tok_s_median",
    "speedup_median",
    "speedup_min",
    "accuracy_within_5pp",
    "speedup_median_at_least_1_4x",
    "speedup_min_at_least_1_3x",
    "service_config_match",
    "complete",
)


def build_comparisons(
    rows: dict[tuple[Any, ...], dict[str, Any]],
    *,
    performance_repeats: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int], dict[int, dict[str, dict[str, Any]]]] = {}
    for row in rows.values():
        key = (
            str(row["phase"]),
            str(row["model"]),
            str(row["task"]),
            int(row["batch_size"]),
            int(row["k"]),
        )
        repeat = int(row["repeat"])
        grouped.setdefault(key, {}).setdefault(repeat, {})[str(row["variant"])] = row

    comparisons: list[dict[str, Any]] = []
    for key in sorted(grouped):
        phase, model, task, batch_size, k = key
        repeat_pairs = grouped[key]
        speedups: list[float] = []
        dense_rates: list[float] = []
        speclink_rates: list[float] = []
        dense_scores: list[float] = []
        speclink_scores: list[float] = []
        expected_repeats = performance_repeats if phase == "performance" else 1
        complete = set(repeat_pairs) == set(range(1, expected_repeats + 1))
        service_config_match = True
        for variants in repeat_pairs.values():
            dense = variants.get("dense")
            sparse = variants.get("speclink")
            if not dense or not sparse:
                complete = False
                service_config_match = False
                continue
            dense_service_config = str(dense.get("service_config_id", ""))
            sparse_service_config = str(sparse.get("service_config_id", ""))
            if (
                not dense_service_config
                or dense_service_config != sparse_service_config
            ):
                complete = False
                service_config_match = False
                continue
            if dense.get("status") != "ok" or sparse.get("status") != "ok":
                complete = False
            dense_rate = as_float(dense.get("request_output_tokens_per_second"))
            sparse_rate = as_float(sparse.get("request_output_tokens_per_second"))
            if dense_rate and sparse_rate:
                dense_rates.append(dense_rate)
                speclink_rates.append(sparse_rate)
                speedups.append(sparse_rate / dense_rate)
            elif phase == "performance":
                complete = False
            dense_score = as_float(dense.get("score"))
            sparse_score = as_float(sparse.get("score"))
            if dense_score is not None and sparse_score is not None:
                dense_scores.append(dense_score)
                speclink_scores.append(sparse_score)
            elif phase == "accuracy":
                complete = False
        dense_score = statistics.median(dense_scores) if dense_scores else None
        sparse_score = statistics.median(speclink_scores) if speclink_scores else None
        delta_pp = (
            (sparse_score - dense_score) * 100.0
            if dense_score is not None and sparse_score is not None
            else None
        )
        speedup_median = statistics.median(speedups) if speedups else None
        speedup_min = min(speedups) if speedups else None
        comparisons.append(
            {
                "phase": phase,
                "model": model,
                "task": task,
                "batch_size": batch_size,
                "k": k,
                "repeat_count": len(speedups) if phase == "performance" else len(dense_scores),
                "dense_score": dense_score,
                "speclink_score": sparse_score,
                "accuracy_delta_pp": delta_pp,
                "dense_request_tok_s_median": (
                    statistics.median(dense_rates) if dense_rates else None
                ),
                "speclink_request_tok_s_median": (
                    statistics.median(speclink_rates) if speclink_rates else None
                ),
                "speedup_median": speedup_median,
                "speedup_min": speedup_min,
                "accuracy_within_5pp": (
                    delta_pp >= -5.0
                    if phase == "accuracy" and delta_pp is not None
                    else ""
                ),
                "speedup_median_at_least_1_4x": (
                    speedup_median >= 1.4
                    if phase == "performance" and speedup_median is not None
                    else ""
                ),
                "speedup_min_at_least_1_3x": (
                    speedup_min >= 1.3
                    if phase == "performance" and speedup_min is not None
                    else ""
                ),
                "service_config_match": service_config_match,
                "complete": complete,
            }
        )
    return comparisons


def write_comparisons(output_root: Path, comparisons: list[dict[str, Any]]) -> None:
    path = output_root / "comparisons.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({field: row.get(field, "") for field in COMPARISON_FIELDS})

    performance = [row for row in comparisons if row["phase"] == "performance"]
    accuracy = [row for row in comparisons if row["phase"] == "accuracy"]
    performance_ok = all(
        row["complete"]
        and row["speedup_median_at_least_1_4x"]
        and row["speedup_min_at_least_1_3x"]
        for row in performance
    )
    accuracy_ok = all(
        row["complete"] and row["accuracy_within_5pp"] for row in accuracy
    )
    lines = [
        "# SpecLink strict same-K matrix",
        "",
        f"Performance points: {len(performance)}; all pass: "
        f"{performance_ok if performance else 'not run'}",
        f"Matched-load accuracy points: {len(accuracy)}; all pass: "
        f"{accuracy_ok if accuracy else 'not run'}",
        "",
        "Primary throughput is client-observed request-phase output tokens/s. ",
        "Each speedup pairs the same model, task, batch size, K, serving budget, and repeat.",
        "Scores emitted during performance runs are diagnostics only. The accuracy gate uses "
        "only phase=accuracy runs at the matching matrix batch size and concurrency.",
        "",
        "| Phase | Model | Task | bs | K | Delta pp | Median speedup | Min speedup | Pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        is_accuracy = row["phase"] == "accuracy"
        delta = row["accuracy_delta_pp"] if is_accuracy else None
        median = None if is_accuracy else row["speedup_median"]
        minimum = None if is_accuracy else row["speedup_min"]
        passed = row["complete"] and (
            row["accuracy_within_5pp"]
            if row["phase"] == "accuracy"
            else (
                row["speedup_median_at_least_1_4x"]
                and row["speedup_min_at_least_1_3x"]
            )
        )
        lines.append(
            "| {phase} | {model} | {task} | {batch_size} | {k} | {delta} | "
            "{median} | {minimum} | {passed} |".format(
                **row,
                delta="" if delta is None else f"{delta:.2f}",
                median="" if median is None else f"{median:.3f}x",
                minimum="" if minimum is None else f"{minimum:.3f}x",
                passed=passed,
            )
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_root = args.output_root.resolve()
    selected_accuracy_batch_sizes = accuracy_batch_sizes(args)
    if not selected_accuracy_batch_sizes or any(
        batch_size <= 0 for batch_size in selected_accuracy_batch_sizes
    ):
        raise ValueError("accuracy batch sizes must be positive")
    if args.performance_repeats <= 0:
        raise ValueError("--performance-repeats must be positive")
    if args.num_concurrent is not None and args.num_concurrent <= 0:
        raise ValueError("--num-concurrent must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = load_candidate_configs(args.candidate_config, args.models)
    runtime_id = implementation_id()
    invocation = {
        "models": args.models,
        "tasks": args.tasks,
        "batch_sizes": args.batch_sizes,
        "k_values": args.k_values,
        "phase": args.phase,
        "performance_repeats": args.performance_repeats,
        "num_concurrent": args.num_concurrent or "batch_size",
        "performance_limit": args.performance_limit,
        "accuracy_limit": args.accuracy_limit,
        "accuracy_batch_sizes": selected_accuracy_batch_sizes,
        "candidate_configs": configs,
        "implementation_id": runtime_id,
        "implementation_files": list(RUNTIME_IMPLEMENTATION_FILES),
        "variants": args.variants,
        "reference_measurements": (
            str(args.reference_measurements.resolve())
            if args.reference_measurements
            else ""
        ),
        "protocol": {
            "same_k": True,
            "fresh_server_per_measurement": True,
            "performance_num_concurrent": args.num_concurrent or "batch_size",
            "accuracy_num_concurrent": args.num_concurrent or "batch_size",
            "accuracy_batch_sizes": selected_accuracy_batch_sizes,
            "accuracy_route": "candidate routing enabled",
            "interleaved_variants": True,
        },
        "created_at": stamp(),
    }
    with (args.output_root / "invocations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(invocation, sort_keys=True) + "\n")
    write_json(args.output_root / "run_config.json", invocation)
    measurements_path = args.output_root / "measurements.csv"
    measurements = load_measurements(measurements_path)
    if args.reference_measurements:
        for key, row in load_measurements(args.reference_measurements).items():
            if row.get("variant") == "dense" and key not in measurements:
                measurements[key] = row
        write_measurements(measurements_path, measurements)
    cases = build_cases(args)
    commands_path = args.output_root / "commands.jsonl"

    for index, case in enumerate(cases, 1):
        config = configs[case.model]
        run_dir = case_run_dir(args.output_root, case, config, runtime_id)
        command = build_command(args, case, run_dir, config)
        expected_service_config_id = service_config_id(
            args,
            case,
            config,
            runtime_id,
        )
        existing = measurements.get(case.key)
        expected_config_id = variant_config_id(
            case.variant,
            config,
            runtime_id,
        )
        if (
            args.resume
            and existing
            and existing.get("status") == "ok"
            and existing.get("config_id") == expected_config_id
            and existing.get("service_config_id") == expected_service_config_id
        ):
            print(f"[{index}/{len(cases)}] resume skip {case.key}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        wait_for_gpu(args)
        run_dir.mkdir(parents=True, exist_ok=True)
        with commands_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"case": case.key, "command": command, "timestamp": time.time()}
                )
                + "\n"
            )
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str((EVAL_ROOT / "temp" / "matplotlib").resolve()))
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
        measurements[case.key] = measurement_from_result(
            case,
            run_dir,
            command,
            config,
            expected_service_config_id,
            runtime_id,
            completed.returncode,
        )
        write_measurements(measurements_path, measurements)
        write_comparisons(
            args.output_root,
            build_comparisons(
                measurements,
                performance_repeats=args.performance_repeats,
            ),
        )
        if completed.returncode != 0 and args.fail_fast:
            raise RuntimeError(f"matrix case failed: {case.key}")

    if not args.dry_run:
        write_comparisons(
            args.output_root,
            build_comparisons(
                measurements,
                performance_repeats=args.performance_repeats,
            ),
        )
    print(args.output_root.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", choices=("performance", "accuracy", "both"), default="both")
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=["dense", "speclink"],
    )
    parser.add_argument("--models", type=csv_strings, default=list(DEFAULT_MODELS))
    parser.add_argument("--tasks", type=csv_strings, default=list(DEFAULT_TASKS))
    parser.add_argument("--batch-sizes", type=csv_ints, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--k-values", type=csv_ints, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--performance-repeats", type=int, default=3)
    parser.add_argument(
        "--num-concurrent",
        type=int,
        help=(
            "Concurrent lm-eval HTTP batches. By default this equals the matrix "
            "batch size; use 1 only for serial diagnostics."
        ),
    )
    parser.add_argument("--performance-limit", default="128")
    parser.add_argument("--accuracy-limit", default="")
    parser.add_argument(
        "--accuracy-batch-sizes",
        type=csv_ints,
        help="optional accuracy-only matrix; defaults to --batch-sizes",
    )
    parser.add_argument(
        "--accuracy-batch-size",
        type=int,
        help="deprecated single-size override for legacy diagnostics",
    )
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--reference-measurements", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EVAL_ROOT / "temp" / f"speclink_same_k_matrix_{stamp()}",
    )
    parser.add_argument("--max-context-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--server-warmup-batches", type=int, default=1)
    parser.add_argument("--server-warmup-max-tokens", type=int, default=16)
    parser.add_argument(
        "--server-warmup-lm-eval-batches", type=int, default=1
    )
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port-base", type=int, default=8260)
    parser.add_argument("--gpu-check-interval-s", type=int, default=600)
    parser.add_argument("--gpu-idle-memory-mb", type=int, default=1024)
    parser.add_argument("--gpu-idle-utilization", type=int, default=10)
    parser.add_argument("--skip-gpu-wait", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
