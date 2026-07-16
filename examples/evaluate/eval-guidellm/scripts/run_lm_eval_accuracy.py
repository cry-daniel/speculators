#!/usr/bin/env python3
"""Run lm-eval through this repo's vLLM/EAGLE3 serving path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib import request

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
    DEFAULT_TOKEN_DENSE_MASK_ROOT,
    MethodConfig,
    TOKEN_DENSE_METHODS,
    method_env,
    parse_method_config,
    timestamp,
)

HYBRID_METHODS = ["activation_aware", *TOKEN_DENSE_METHODS]
MODE_GROUPS = {
    "dense": ["dense_ar"],
    "speculative": ["eagle3_dense"],
    "speculative_eager": ["eagle3_dense_eager"],
    "hybrid": HYBRID_METHODS,
    "token_dense": TOKEN_DENSE_METHODS,
    "all": ["dense_ar", "eagle3_dense", *HYBRID_METHODS],
}
TASK_GROUPS = {
    "smoke": ["logiqa_generative", "gsm8k_cot"],
    "math_reasoning": ["math_reasoning"],
    "gsm8k_math_reasoning": ["gsm8k_cot", "math_reasoning"],
    "all": [
        "agieval_logiqa_en",
        "logiqa_generative",
        "gsm8k_cot",
        "hendrycks_math",
        "mmlu_generative",
        "humaneval_instruct",
        "longbenchv2_generative",
    ],
}
TASK_MAX_TOKENS = {
    "agieval_logiqa_en": 16,
    "logiqa_generative": 16,
    "gsm8k_local": 512,
    "gsm8k_cot": 512,
    "gsm8k_cot_zeroshot": 512,
    "math_reasoning": 512,
    "hendrycks_math500": 1024,
    "hendrycks_math": 1024,
    "mmlu_generative": 16,
    "humaneval_instruct": 512,
    "longbench2": 16,
    "longbenchv2_generative": 16,
}


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def expand_modes(value: str) -> list[str]:
    out: list[str] = []
    for item in csv_list(value):
        if item in MODE_GROUPS:
            candidates = MODE_GROUPS[item]
        elif item.startswith("d") and item[1:].isdigit():
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


def mode_method(mode: str) -> MethodConfig:
    if mode in {"dense_ar", "eagle3_dense", "eagle3_dense_eager"}:
        return MethodConfig(label="dense", base_method="dense", policy="dense")
    return parse_method_config(mode)


def mode_uses_spec(mode: str) -> bool:
    return mode != "dense_ar"


def mode_group(mode: str) -> str:
    if mode == "dense_ar":
        return "dense"
    if mode in {"eagle3_dense", "eagle3_dense_eager"}:
        return "speculative"
    return "hybrid"


def build_vllm_command(
    args: argparse.Namespace,
    *,
    mode: str,
    model_path: str,
    speculator_path: str,
    port: int,
) -> list[str]:
    vllm_dtype = {
        "fp16": "float16",
        "bf16": "bfloat16",
    }.get(str(args.dtype), str(args.dtype))
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
        "--dtype",
        vllm_dtype,
    ]
    attention_backend = str(
        getattr(args, "attention_backend", "auto")
    ).strip()
    if attention_backend and attention_backend.lower() != "auto":
        command.extend(["--attention-backend", attention_backend])
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
    if mode_uses_spec(mode):
        if (
            mode.startswith("token_dense_")
            and not args.token_dense_flashinfer_autotune
        ):
            command.append("--no-enable-flashinfer-autotune")
        if (
            not args.enforce_eager
            and not (
                mode.startswith("token_dense_")
                and args.token_dense_enforce_eager
            )
            and (
                mode.startswith("token_dense_")
                or args.token_dense_cudagraph_mode != "none"
            )
        ):
            cg_mode = str(args.token_dense_cudagraph_mode).lower()
            if cg_mode not in {"none", "full", "full_decode_only"}:
                raise ValueError(
                    "unsupported token-dense CUDA graph mode "
                    f"{cg_mode!r}"
                )
            cudagraph_mode = {
                "none": "NONE",
                "full": "FULL_AND_PIECEWISE",
                "full_decode_only": "FULL_DECODE_ONLY",
            }[cg_mode]
            command.extend(
                [
                    "--compilation-config",
                    json.dumps(
                        {
                            "mode": args.token_dense_compilation_mode.upper(),
                            "cudagraph_mode": cudagraph_mode,
                        }
                    ),
                ]
            )
    if (
        args.enforce_eager
        or (mode.startswith("token_dense_") and args.token_dense_enforce_eager)
        or mode.startswith("activation_aware")
        or mode == "eagle3_dense_eager"
    ):
        command.append("--enforce-eager")
    return command


def start_server(
    args: argparse.Namespace,
    *,
    mode: str,
    model_label: str,
    model_path: str,
    speculator_path: str,
    run_dir: Path,
) -> tuple[subprocess.Popen[Any], int]:
    port = find_free_port(args.port_base)
    method = mode_method(mode)
    if method.base_method == "token_dense":
        method = replace(
            method,
            policy=args.token_dense_layer_policy,
            keep_n=args.token_dense_keep_n,
        )
    stats_path = run_dir / "vllm_structured_24_stats.json"
    env = method_env(args, model_label=model_label, method=method, stats_path=stats_path)
    if mode == "dense_ar":
        env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
        env["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    command = build_vllm_command(
        args,
        mode=mode,
        model_path=model_path,
        speculator_path=speculator_path,
        port=port,
    )
    write_json(run_dir / "server_command.json", {"command": command, "port": port})
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


def warmup_server(
    args: argparse.Namespace,
    *,
    port: int,
    model_path: str,
    run_dir: Path,
) -> None:
    """Warm the exact serving batch before collecting throughput metrics."""

    if args.server_warmup_batches <= 0:
        return
    prompt_count = args.num_concurrent
    payload = {
        "model": model_path,
        "prompt": ["Q: What is 1 + 1?\nA:"] * prompt_count,
        "max_tokens": args.server_warmup_max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    encoded = json.dumps(payload).encode("utf-8")
    url = f"http://127.0.0.1:{port}/v1/completions"
    started = time.perf_counter()
    completed_batches = 0
    try:
        for _ in range(args.server_warmup_batches):
            warmup_request = request.Request(
                url,
                data=encoded,
                headers={
                    "Authorization": "Bearer EMPTY",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with request.urlopen(
                warmup_request, timeout=args.request_timeout_s
            ) as response:
                response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"warmup request failed with HTTP {response.status}"
                    )
            completed_batches += 1
    finally:
        write_json(
            run_dir / "server_warmup.json",
            {
                "requested_batches": args.server_warmup_batches,
                "completed_batches": completed_batches,
                "prompts_per_batch": prompt_count,
                "max_tokens": args.server_warmup_max_tokens,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )


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
    warmup: bool = False,
    limit_override: int | None = None,
) -> int:
    max_gen_toks = args.max_new_tokens or TASK_MAX_TOKENS.get(task, 512)
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
        str(
            run_dir
            / ("lm_eval_warmup_output" if warmup else "lm_eval_output")
        ),
    ]
    if not warmup:
        command.append("--log_samples")
    limit = limit_override if limit_override is not None else args.limit
    if limit:
        command.extend(["--limit", str(limit)])
    if args.apply_chat_template:
        command.append("--apply_chat_template")
    if task == "humaneval_instruct" and args.allow_unsafe_code:
        command.append("--confirm_run_unsafe_code")
    command_name = (
        "lm_eval_warmup_command.json" if warmup else "lm_eval_command.json"
    )
    write_json(run_dir / command_name, {"command": command})
    env = add_local_no_proxy(os.environ.copy())
    env["OPENAI_API_KEY"] = "EMPTY"
    # lm-eval is an HTTP client in this workflow. Hiding CUDA prevents its
    # PyTorch seed/setup path from holding roughly 1 GiB in a second process.
    env["CUDA_VISIBLE_DEVICES"] = ""
    if task == "humaneval_instruct" and args.allow_unsafe_code:
        env["HF_ALLOW_CODE_EVAL"] = "1"
    log_name = "lm_eval_warmup.log" if warmup else "lm_eval.log"
    started = time.perf_counter()
    with (run_dir / log_name).open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(EVAL_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if warmup:
        write_json(
            run_dir / "lm_eval_warmup_meta.json",
            {
                "returncode": int(completed.returncode),
                "limit": limit,
                "elapsed_seconds": time.perf_counter() - started,
            },
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


def aggregate_outputs(
    output_dir: Path,
    baseline_dir: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "aggregate_lm_eval_accuracy.py"),
        "--output-dir",
        str(output_dir),
    ]
    if baseline_dir is not None:
        command.extend(["--baseline-dir", str(baseline_dir.resolve())])
    subprocess.run(
        command,
        cwd=str(SPECULATORS_ROOT),
        check=False,
    )


def set_local_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def run(args: argparse.Namespace) -> None:
    configure_local_no_proxy()
    set_local_seed(args.seed)
    if args.server_warmup_batches < 0:
        raise ValueError("--server-warmup-batches must be non-negative")
    if args.server_warmup_lm_eval_batches < 0:
        raise ValueError(
            "--server-warmup-lm-eval-batches must be non-negative"
        )
    if args.server_warmup_max_tokens <= 0:
        raise ValueError("--server-warmup-max-tokens must be positive")
    if (
        not math.isfinite(args.token_dense_sparse_value_scale)
        or args.token_dense_sparse_value_scale <= 0.0
    ):
        raise ValueError(
            "--token-dense-sparse-value-scale must be finite and positive"
        )
    if (
        not math.isfinite(args.token_dense_gate_up_value_scale)
        or args.token_dense_gate_up_value_scale <= 0.0
    ):
        raise ValueError(
            "--token-dense-gate-up-value-scale must be finite and positive"
        )
    if (
        not math.isfinite(args.token_dense_row_scale_max)
        or args.token_dense_row_scale_max < 1.0
    ):
        raise ValueError(
            "--token-dense-row-scale-max must be finite and at least 1.0"
        )
    if (
        not math.isfinite(args.token_dense_dense_ratio)
        or not 0.0 <= args.token_dense_dense_ratio <= 1.0
    ):
        raise ValueError("--token-dense-dense-ratio must be in [0, 1]")
    if args.token_dense_dense_min_per_request < 0:
        raise ValueError(
            "--token-dense-dense-min-per-request must be non-negative"
        )
    if args.token_dense_dense_cap < -1:
        raise ValueError(
            "--token-dense-dense-cap must be -1 or a non-negative integer"
        )
    if args.token_dense_qkv_heterogeneous_max_rows <= 0:
        raise ValueError(
            "--token-dense-qkv-heterogeneous-max-rows must be positive"
        )
    if args.token_dense_qkv_paired_max_rows <= 0:
        raise ValueError(
            "--token-dense-qkv-paired-max-rows must be positive"
        )
    if args.token_dense_full_sparse_override_mlp_min_rows <= 0:
        raise ValueError(
            "--token-dense-full-sparse-override-mlp-min-rows must be positive"
        )
    if args.token_dense_balanced_start_position < 0:
        raise ValueError(
            "--token-dense-balanced-start-position must be non-negative"
        )
    if args.token_dense_batch_route_block_steps <= 0:
        raise ValueError(
            "--token-dense-batch-route-block-steps must be positive"
        )
    if args.token_dense_adaptive_dense_max_requests < 0:
        raise ValueError(
            "--token-dense-adaptive-dense-max-requests must be non-negative"
        )
    if (
        not math.isfinite(args.token_dense_batch_route_initial_credit)
        or not 0.0 <= args.token_dense_batch_route_initial_credit < 1.0
    ):
        raise ValueError(
            "--token-dense-batch-route-initial-credit must be in [0, 1)"
        )
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
    if args.token_dense_routed_swiglu_mlp and (
        not args.token_dense_fused_batch_mlp
        or args.token_dense_linear_strategy != "full_sparse_residual"
        or args.token_dense_mlp_strategy != "linear"
    ):
        raise ValueError(
            "--token-dense-routed-swiglu-mlp requires "
            "--token-dense-fused-batch-mlp, "
            "--token-dense-linear-strategy=full_sparse_residual, and "
            "--token-dense-mlp-strategy=linear"
        )
    if (
        args.token_dense_qkv_heterogeneous_routing
        and args.token_dense_linear_strategy
        not in {"full_sparse_residual", "full_sparse_dense_override"}
    ):
        raise ValueError(
            "--token-dense-qkv-heterogeneous-routing requires "
            "--token-dense-linear-strategy=full_sparse_residual or "
            "full_sparse_dense_override"
        )
    if (
        args.token_dense_qkv_paired_routing
        and args.token_dense_linear_strategy
        not in {"full_sparse_residual", "full_sparse_dense_override"}
    ):
        raise ValueError(
            "--token-dense-qkv-paired-routing requires "
            "--token-dense-linear-strategy=full_sparse_residual or "
            "full_sparse_dense_override"
        )
    if (
        args.token_dense_qkv_vec4_postop
        and not args.token_dense_qkv_paired_routing
    ):
        raise ValueError(
            "--token-dense-qkv-vec4-postop requires "
            "--token-dense-qkv-paired-routing"
        )
    if (
        args.token_dense_qkv_active_wave_c12
        and not args.token_dense_qkv_paired_routing
    ):
        raise ValueError(
            "--token-dense-qkv-active-wave-c12 requires "
            "--token-dense-qkv-paired-routing"
        )
    if (
        args.token_dense_qkv_fused_epilogue
        and not args.token_dense_qkv_paired_routing
    ):
        raise ValueError(
            "--token-dense-qkv-fused-epilogue requires "
            "--token-dense-qkv-paired-routing"
        )
    if (
        args.token_dense_qkv_direct_cache
        and not args.token_dense_qkv_vec4_postop
    ):
        raise ValueError(
            "--token-dense-qkv-direct-cache requires "
            "--token-dense-qkv-vec4-postop"
        )
    if args.token_dense_full_sparse_override_mlp and (
        not args.token_dense_fused_batch_mlp
        or not args.token_dense_inline_swiglu_mlp
        or args.token_dense_linear_strategy != "full_sparse_dense_override"
        or args.token_dense_mlp_strategy != "linear"
    ):
        raise ValueError(
            "--token-dense-full-sparse-override-mlp requires fused batch "
            "MLP, inline SwiGLU, full_sparse_dense_override, and linear MLP"
        )
    if (
        args.token_dense_qkv_heterogeneous_routing
        and args.token_dense_release_dense_weights
        and args.token_dense_retain_dense_weight not in {"qkv", "attention"}
    ):
        raise ValueError(
            "heterogeneous QKV routing with released dense weights requires "
            "--token-dense-retain-dense-weight=qkv or attention"
        )
    if (
        args.token_dense_routed_swiglu_mlp
        and args.token_dense_inline_swiglu_mlp
    ):
        raise ValueError(
            "routed and non-routed inline SwiGLU modes are mutually exclusive"
        )
    if (
        any(mode.startswith("token_dense_") for mode in modes)
        and args.token_dense_linear_strategy == "full_sparse_residual"
        and (
            args.token_dense_sparse_value_scale != 1.0
            or args.token_dense_gate_up_value_scale != 1.0
            or args.token_dense_group_reconstruction
        )
    ):
        raise ValueError(
            "explicit sparse value scaling or grouped reconstruction is "
            "incompatible with exact full_sparse_residual"
        )

    commands: list[str] = []
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
                if args.resume and (
                    list((run_dir / "lm_eval_output").rglob("*.json"))
                    or (run_dir / "skip.json").exists()
                ):
                    continue
                reason = should_skip_task(args, task)
                if reason:
                    write_skip(run_dir, reason=reason, task=task, mode=mode, model_label=model_label)
                    if args.aggregate:
                        aggregate_outputs(output_dir, args.baseline_dir)
                    continue
                process = None
                try:
                    process, port = start_server(
                        args,
                        mode=mode,
                        model_label=model_label,
                        model_path=model_path,
                        speculator_path=speculator_path,
                        run_dir=run_dir,
                    )
                    warmup_server(
                        args,
                        port=port,
                        model_path=model_path,
                        run_dir=run_dir,
                    )
                    if args.server_warmup_lm_eval_batches > 0:
                        warmup_rc = run_lm_eval(
                            args,
                            task=task,
                            mode=mode,
                            model_path=model_path,
                            tokenizer_path=tokenizer_path,
                            port=port,
                            run_dir=run_dir,
                            warmup=True,
                            limit_override=(
                                args.server_warmup_lm_eval_batches
                                * args.num_concurrent
                            ),
                        )
                        if warmup_rc != 0:
                            raise RuntimeError(
                                "lm-eval server warmup failed with return "
                                f"code {warmup_rc}: {run_dir}"
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
                    accepted = after.get("vllm:spec_decode_num_accepted_tokens", 0.0) - before.get(
                        "vllm:spec_decode_num_accepted_tokens", 0.0
                    )
                    drafted = after.get("vllm:spec_decode_num_draft_tokens", 0.0) - before.get(
                        "vllm:spec_decode_num_draft_tokens", 0.0
                    )
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
                            "task": task,
                            "num_spec_tokens": args.num_spec_tokens,
                            "batch_size": args.batch_size,
                            "num_concurrent": args.num_concurrent,
                            "max_num_seqs": args.max_num_seqs,
                            "max_num_batched_tokens": (
                                args.max_num_batched_tokens
                            ),
                            "max_context_length": args.max_context_length,
                            "server_warmup_batches": (
                                args.server_warmup_batches
                            ),
                            "server_warmup_max_tokens": (
                                args.server_warmup_max_tokens
                            ),
                            "server_warmup_lm_eval_batches": (
                                args.server_warmup_lm_eval_batches
                            ),
                            "dtype": args.dtype,
                            "token_dense_linear_strategy": (
                                args.token_dense_linear_strategy
                            ),
                            "token_dense_mlp_strategy": (
                                args.token_dense_mlp_strategy
                            ),
                            "token_dense_projection_policy": (
                                args.token_dense_projection_policy
                            ),
                            "token_dense_mixed_projection_policy": (
                                args.token_dense_mixed_projection_policy
                            ),
                            "token_dense_o_sparse_layers": (
                                args.token_dense_o_sparse_layers
                            ),
                            "token_dense_mixed_layers": (
                                args.token_dense_mixed_layers
                            ),
                            "token_dense_mlp_static_layers": (
                                args.token_dense_mlp_static_layers
                            ),
                            "token_dense_gate_up_dense_layers": (
                                args.token_dense_gate_up_dense_layers
                            ),
                            "token_dense_down_dense_layers": (
                                args.token_dense_down_dense_layers
                            ),
                            "token_dense_attention_dense_layers": (
                                args.token_dense_attention_dense_layers
                            ),
                            "token_dense_dense_layers": (
                                args.token_dense_dense_layers
                            ),
                            "token_dense_dense_ratio": (
                                args.token_dense_dense_ratio
                            ),
                            "token_dense_dense_min_per_request": (
                                args.token_dense_dense_min_per_request
                            ),
                            "token_dense_dense_cap": (
                                args.token_dense_dense_cap
                            ),
                            "token_dense_dense_selection": (
                                args.token_dense_dense_selection
                            ),
                            "token_dense_balanced_start_position": (
                                args.token_dense_balanced_start_position
                            ),
                            "token_dense_sparse_value_scale": (
                                args.token_dense_sparse_value_scale
                            ),
                            "token_dense_gate_up_value_scale": (
                                args.token_dense_gate_up_value_scale
                            ),
                            "token_dense_gate_up_hybrid": (
                                args.token_dense_gate_up_hybrid
                            ),
                            "token_dense_fused_batch_mlp": (
                                args.token_dense_fused_batch_mlp
                            ),
                            "token_dense_inline_swiglu_mlp": (
                                args.token_dense_inline_swiglu_mlp
                            ),
                            "token_dense_routed_swiglu_mlp": (
                                args.token_dense_routed_swiglu_mlp
                            ),
                            "token_dense_sparse_gate_dense_down": (
                                args.token_dense_sparse_gate_dense_down
                            ),
                            "token_dense_direct_store_gate_up": (
                                args.token_dense_direct_store_gate_up
                            ),
                            "token_dense_cutlass_down_fp16": (
                                args.token_dense_cutlass_down_fp16
                            ),
                            "token_dense_direct_gate_residual_epilogue": (
                                args.token_dense_direct_gate_residual_epilogue
                            ),
                            "token_dense_gate_down_gather_epilogue": (
                                args.token_dense_gate_down_gather_epilogue
                            ),
                            "token_dense_contiguous_down_input": (
                                args.token_dense_contiguous_down_input
                            ),
                            "token_dense_uninitialized_routed_workspace": (
                                args.token_dense_uninitialized_routed_workspace
                            ),
                            "token_dense_paired_persistent_gate": (
                                args.token_dense_paired_persistent_gate
                            ),
                            "token_dense_paired_gate_shape_tuning": (
                                args.token_dense_paired_gate_shape_tuning
                            ),
                            "token_dense_paired_fused_gate_epilogue": (
                                args.token_dense_paired_fused_gate_epilogue
                            ),
                            "token_dense_paired_persistent_down": (
                                args.token_dense_paired_persistent_down
                            ),
                            "token_dense_paired_gather_down": (
                                args.token_dense_paired_gather_down
                            ),
                            "token_dense_paired_inplace_down": (
                                args.token_dense_paired_inplace_down
                            ),
                            "token_dense_paired_inplace_o": (
                                args.token_dense_paired_inplace_o
                            ),
                            "token_dense_parallel_splitk_down": (
                                args.token_dense_parallel_splitk_down
                            ),
                            "token_dense_single_launch_splitk_down": (
                                args.token_dense_single_launch_splitk_down
                            ),
                            "token_dense_tiled_indexed_splitk_down": (
                                args.token_dense_tiled_indexed_splitk_down
                            ),
                            "token_dense_indexed_down_epilogue": (
                                args.token_dense_indexed_down_epilogue
                            ),
                            "token_dense_indexed_output_epilogue": (
                                args.token_dense_indexed_output_epilogue
                            ),
                            "token_dense_group_reconstruction": (
                                args.token_dense_group_reconstruction
                            ),
                            "token_dense_row_scale_mode": (
                                args.token_dense_row_scale_mode
                            ),
                            "token_dense_variance_scale_projection_policy": (
                                args.token_dense_variance_scale_projection_policy
                            ),
                            "token_dense_row_scale_max": (
                                args.token_dense_row_scale_max
                            ),
                            "token_dense_sparse_output_mode": (
                                args.token_dense_sparse_output_mode
                            ),
                            "token_dense_sparse_accumulator": (
                                args.token_dense_sparse_accumulator
                            ),
                            "token_dense_sparse_backend": (
                                args.token_dense_sparse_backend
                            ),
                            "token_dense_mask_method": (
                                args.token_dense_mask_method
                            ),
                            "uses_speculative": mode_uses_spec(mode),
                            "spec_accepted_tokens": accepted if drafted else None,
                            "spec_draft_tokens": drafted if drafted else None,
                            "spec_acceptance_rate": accepted / drafted if drafted else None,
                            "created_at": timestamp(),
                        },
                    )
                    commands.append(str((run_dir / "lm_eval_command.json").resolve()))
                    if args.aggregate:
                        aggregate_outputs(output_dir, args.baseline_dir)
                    if rc != 0:
                        raise RuntimeError(
                            f"lm-eval failed with return code {rc}: {run_dir}"
                        )
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
            "baseline_dir": (
                str(args.baseline_dir.resolve()) if args.baseline_dir else ""
            ),
            "num_spec_tokens": args.num_spec_tokens,
            "batch_size": args.batch_size,
            "num_concurrent": args.num_concurrent,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "max_context_length": args.max_context_length,
            "max_new_tokens": args.max_new_tokens,
            "server_warmup_batches": args.server_warmup_batches,
            "server_warmup_max_tokens": args.server_warmup_max_tokens,
            "server_warmup_lm_eval_batches": (
                args.server_warmup_lm_eval_batches
            ),
            "limit": args.limit,
            "seed": args.seed,
            "token_dense_linear_strategy": args.token_dense_linear_strategy,
            "token_dense_mlp_strategy": args.token_dense_mlp_strategy,
            "token_dense_projection_policy": (
                args.token_dense_projection_policy
            ),
            "token_dense_mixed_projection_policy": (
                args.token_dense_mixed_projection_policy
            ),
            "token_dense_o_sparse_layers": (
                args.token_dense_o_sparse_layers
            ),
            "token_dense_mixed_layers": args.token_dense_mixed_layers,
            "token_dense_mlp_static_layers": (
                args.token_dense_mlp_static_layers
            ),
            "token_dense_gate_up_dense_layers": (
                args.token_dense_gate_up_dense_layers
            ),
            "token_dense_down_dense_layers": (
                args.token_dense_down_dense_layers
            ),
            "token_dense_attention_dense_layers": (
                args.token_dense_attention_dense_layers
            ),
            "token_dense_dense_layers": args.token_dense_dense_layers,
            "token_dense_dense_ratio": args.token_dense_dense_ratio,
            "token_dense_dense_min_per_request": (
                args.token_dense_dense_min_per_request
            ),
            "token_dense_dense_cap": args.token_dense_dense_cap,
            "token_dense_dense_selection": args.token_dense_dense_selection,
            "token_dense_balanced_start_position": (
                args.token_dense_balanced_start_position
            ),
            "token_dense_layer_policy": args.token_dense_layer_policy,
            "token_dense_keep_n": args.token_dense_keep_n,
            "token_dense_score_backend": args.token_dense_score_backend,
            "token_dense_fast_plan": args.token_dense_fast_plan,
            "token_dense_reuse_buffers": args.token_dense_reuse_buffers,
            "token_dense_contiguous_scatter": (
                args.token_dense_contiguous_scatter
            ),
            "token_dense_sparse_value_scale": (
                args.token_dense_sparse_value_scale
            ),
            "token_dense_gate_up_value_scale": (
                args.token_dense_gate_up_value_scale
            ),
            "token_dense_gate_up_hybrid": args.token_dense_gate_up_hybrid,
            "token_dense_fused_batch_mlp": (
                args.token_dense_fused_batch_mlp
            ),
            "token_dense_inline_swiglu_mlp": (
                args.token_dense_inline_swiglu_mlp
            ),
            "token_dense_routed_swiglu_mlp": (
                args.token_dense_routed_swiglu_mlp
            ),
            "token_dense_sparse_gate_dense_down": (
                args.token_dense_sparse_gate_dense_down
            ),
            "token_dense_direct_store_gate_up": (
                args.token_dense_direct_store_gate_up
            ),
            "token_dense_cutlass_down_fp16": (
                args.token_dense_cutlass_down_fp16
            ),
            "token_dense_direct_gate_residual_epilogue": (
                args.token_dense_direct_gate_residual_epilogue
            ),
            "token_dense_gate_down_gather_epilogue": (
                args.token_dense_gate_down_gather_epilogue
            ),
            "token_dense_contiguous_down_input": (
                args.token_dense_contiguous_down_input
            ),
            "token_dense_uninitialized_routed_workspace": (
                args.token_dense_uninitialized_routed_workspace
            ),
            "token_dense_paired_persistent_gate": (
                args.token_dense_paired_persistent_gate
            ),
            "token_dense_paired_gate_shape_tuning": (
                args.token_dense_paired_gate_shape_tuning
            ),
            "token_dense_paired_fused_gate_epilogue": (
                args.token_dense_paired_fused_gate_epilogue
            ),
            "token_dense_paired_persistent_down": (
                args.token_dense_paired_persistent_down
            ),
            "token_dense_paired_gather_down": (
                args.token_dense_paired_gather_down
            ),
            "token_dense_paired_inplace_down": (
                args.token_dense_paired_inplace_down
            ),
            "token_dense_paired_inplace_o": (
                args.token_dense_paired_inplace_o
            ),
            "token_dense_parallel_splitk_down": (
                args.token_dense_parallel_splitk_down
            ),
            "token_dense_single_launch_splitk_down": (
                args.token_dense_single_launch_splitk_down
            ),
            "token_dense_tiled_indexed_splitk_down": (
                args.token_dense_tiled_indexed_splitk_down
            ),
            "token_dense_indexed_down_epilogue": (
                args.token_dense_indexed_down_epilogue
            ),
            "token_dense_indexed_output_epilogue": (
                args.token_dense_indexed_output_epilogue
            ),
            "token_dense_group_reconstruction": (
                args.token_dense_group_reconstruction
            ),
            "token_dense_row_scale_mode": args.token_dense_row_scale_mode,
            "token_dense_variance_scale_projection_policy": (
                args.token_dense_variance_scale_projection_policy
            ),
            "token_dense_row_scale_max": args.token_dense_row_scale_max,
            "token_dense_sparse_output_mode": (
                args.token_dense_sparse_output_mode
            ),
            "token_dense_sparse_accumulator": (
                args.token_dense_sparse_accumulator
            ),
            "token_dense_sparse_backend": args.token_dense_sparse_backend,
            "token_dense_cudagraph_mode": args.token_dense_cudagraph_mode,
            "token_dense_compilation_mode": (
                args.token_dense_compilation_mode
            ),
            "token_dense_flashinfer_autotune": (
                args.token_dense_flashinfer_autotune
            ),
            "token_dense_isolate_decode_batches": (
                args.token_dense_isolate_decode_batches
            ),
            "token_dense_release_dense_weights": (
                args.token_dense_release_dense_weights
            ),
            "token_dense_mask_root": str(args.token_dense_mask_root),
            "token_dense_mask_method": args.token_dense_mask_method,
            "created_at": timestamp(),
        },
    )
    with (output_dir / "commands.sh").open("w", encoding="utf-8") as handle:
        handle.write("# Re-run this matrix\n")
        handle.write(rerun_command(output_dir) + "\n")
    if args.aggregate:
        aggregate_outputs(output_dir, args.baseline_dir)
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
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument(
        "--token-dense-linear-strategy",
        choices=(
            "auto",
            "full_sparse_residual",
            "full_sparse_dense_override",
            "split_dense_sparse",
            "sparse_only_decode",
        ),
        default="auto",
    )
    parser.add_argument(
        "--token-dense-mlp-strategy",
        choices=(
            "auto",
            "gate_only",
            "linear",
        ),
        default="auto",
    )
    parser.add_argument(
        "--token-dense-fused-batch-mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse pure batch-routed gate/up, SwiGLU, and down into one "
            "custom op while preserving transposed sparse intermediates."
        ),
    )
    parser.add_argument(
        "--token-dense-inline-swiglu-mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply SwiGLU in the sparse gate/up CUTLASS epilogue. Requires "
            "--token-dense-fused-batch-mlp and an eligible linear strategy."
        ),
    )
    parser.add_argument(
        "--token-dense-full-sparse-override-mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use profiled exact mixed-row Gate+SwiGLU/Down epilogues with "
            "heterogeneous, persistent, or parallel backends by row count."
        ),
    )
    parser.add_argument(
        "--token-dense-full-sparse-override-mlp-min-rows",
        type=int,
        default=224,
        help="Smallest verifier row count eligible for fused override MLP.",
    )
    parser.add_argument(
        "--token-dense-routed-swiglu-mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse row routing and SwiGLU into the sparse gate/up epilogue. "
            "Requires fused batch MLP with full_sparse_residual."
        ),
    )
    parser.add_argument(
        "--token-dense-sparse-gate-dense-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse all-sparse Gate/Up with SwiGLU while keeping Down dense. "
            "This requires fixed sparse Gate/Up modules and dense Down."
        ),
    )
    parser.add_argument(
        "--token-dense-direct-gate-residual-epilogue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write gate/up residual GEMM output directly in contiguous "
            "layout instead of launching a separate transpose."
        ),
    )
    parser.add_argument(
        "--token-dense-gate-down-gather-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write exact routed SwiGLU rows to both the full hidden output "
            "and compact dense-row Down input in one correction epilogue."
        ),
    )
    parser.add_argument(
        "--token-dense-direct-store-gate-up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store eligible static 2:4 Gate/Up GEMMs directly in row-major "
            "layout from the CUTLASS epilogue."
        ),
    )
    parser.add_argument(
        "--token-dense-cutlass-down-fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the tuned CUTLASS B-column FP16-accumulator kernel for "
            "eligible dense Down projections in sparse Gate/Up layers."
        ),
    )
    parser.add_argument(
        "--token-dense-contiguous-down-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Feed row-major routed SwiGLU output directly to Down GEMMs for "
            "eligible shapes instead of materializing a full transpose."
        ),
    )
    parser.add_argument(
        "--token-dense-uninitialized-routed-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip zero fills for padded routed-MLP workspaces whose padding "
            "rows are excluded from all outputs."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-persistent-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Schedule routed sparse gate/up and its dense-row residual in "
            "one persistent CTA grid for benchmarked verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-gate-shape-tuning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the measured partitioned schedule and worker-grid map for "
            "eligible paired Gate/SwiGLU verifier waves."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-fused-gate-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse paired W24/R24 Gate correction and SwiGLU into the residual "
            "epilogue for eligible verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-persistent-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the paired persistent full/residual Down kernel only for "
            "benchmarked positive verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-gather-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse full sparse Down with gathered dense-row residual Down and "
            "apply the compact correction only for benchmarked shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-inplace-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run exact paired W24/R24 Down and apply the dense-row residual "
            "inside the selected verifier-wave kernel."
        ),
    )
    parser.add_argument(
        "--token-dense-paired-inplace-o",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run exact paired W24/R24 O projection and apply the dense-row "
            "residual inside benchmarked bs>=16 verifier-wave kernels."
        ),
    )
    parser.add_argument(
        "--token-dense-parallel-splitk-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split the exact dense-row Down residual over concurrent K slices "
            "for benchmarked positive verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-single-launch-splitk-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one serial split-K sparse CUTLASS launch for the exact "
            "dense-row Down residual on benchmarked verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-tiled-indexed-splitk-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run full sparse Down directly to contiguous output, overlap a "
            "serial split-K dense-row correction, then transpose-add only "
            "the routed rows for measured-positive verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-indexed-down-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scatter sparse Down rows directly from the CUTLASS epilogue "
            "instead of materializing and merging a compact output."
        ),
    )
    parser.add_argument(
        "--token-dense-indexed-output-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scatter sparse QKV/O rows directly from the CUTLASS epilogue "
            "instead of materializing and merging compact outputs."
        ),
    )
    parser.add_argument(
        "--token-dense-qkv-heterogeneous-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the profiled one-launch exact dense-row plus 2:4-row Qwen "
            "QKV kernel, with shape-based fallback."
        ),
    )
    parser.add_argument(
        "--token-dense-qkv-heterogeneous-max-rows",
        type=int,
        default=704,
        help="Largest verifier row count eligible for the fused QKV route.",
    )
    parser.add_argument(
        "--token-dense-qkv-paired-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the profiled one-grid W24 plus gathered complementary R24 "
            "QKV backend at its positive verifier shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-qkv-paired-max-rows",
        type=int,
        default=704,
        help="Largest verifier row count eligible for paired QKV routing.",
    )
    parser.add_argument(
        "--token-dense-qkv-active-wave-c12",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ablate the layer-cold C12 QKV tile across Qwen active decode "
            "waves; disabled by default after the system A/B regression."
        ),
    )
    parser.add_argument(
        "--token-dense-qkv-vec4-postop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse routed residual addition, Q/K normalization, and Neox "
            "RoPE with the paired row-major QKV output."
        ),
    )
    parser.add_argument(
        "--token-dense-qkv-fused-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run paired W24/R24 QKV, residual correction, Q/K RMSNorm, "
            "and RoPE in one resident-grid launch at profiled C13 shapes."
        ),
    )
    parser.add_argument(
        "--token-dense-qkv-direct-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store fused post-RoPE K/V directly into the FP16 FlashAttention "
            "cache and remove the separate cache-update kernel."
        ),
    )
    parser.add_argument(
        "--token-dense-parallel-mixed-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlap the full-batch 2:4 GEMM with the selected-row dense "
            "GEMM before overwriting dense rows."
        ),
    )
    parser.add_argument(
        "--token-dense-projection-policy",
        choices=(
            "none",
            "all",
            "mlp",
            "attention",
            "gate_up",
            "down",
            "qkv",
            "o",
            "o_gate_up",
            "qkv_down",
            "qkv_gate_up_down",
            "o_down",
            "attention_down",
            "attention_gate_up",
        ),
        default="all",
    )
    parser.add_argument(
        "--token-dense-mixed-projection-policy",
        choices=(
            "none",
            "all",
            "mlp",
            "attention",
            "gate_up",
            "down",
            "qkv",
            "o",
            "o_gate_up",
            "qkv_down",
            "qkv_gate_up_down",
            "o_down",
            "attention_down",
            "attention_gate_up",
        ),
        default="all",
        help=(
            "Selected projections that honor dynamic dense rows. Other "
            "masked projections run every verification row through 2:4."
        ),
    )
    parser.add_argument(
        "--token-dense-mixed-layers",
        default="",
        help=(
            "Layers that honor dynamic dense rows. Empty or all selects all "
            "masked layers; none makes every masked layer pure 2:4."
        ),
    )
    parser.add_argument(
        "--token-dense-mlp-static-layers",
        default="",
        help=(
            "MLP layers that ignore dynamic dense rows and run every verify "
            "row through 2:4. Attention routing is unaffected."
        ),
    )
    parser.add_argument(
        "--token-dense-o-sparse-layers",
        default="",
        help=(
            "O-projection layers eligible for dynamic dense/2:4 routing. "
            "Empty or all selects every otherwise eligible layer; none keeps "
            "all O projections dense."
        ),
    )
    parser.add_argument(
        "--token-dense-gate-up-dense-layers",
        default="",
        help=(
            "Comma-separated gate_up layers or ranges to keep dense, e.g. "
            "4,10,15,24-25. Other selected projections remain sparse."
        ),
    )
    parser.add_argument(
        "--token-dense-dense-layers",
        default="",
        help=(
            "Comma-separated layers or ranges whose selected projections stay "
            "dense, e.g. 8,11-13."
        ),
    )
    parser.add_argument(
        "--token-dense-down-dense-layers",
        default="",
        help=(
            "Comma-separated layers or ranges whose down projection stays "
            "dense, e.g. 6,15."
        ),
    )
    parser.add_argument(
        "--token-dense-attention-dense-layers",
        default="",
        help=(
            "Comma-separated layers or ranges whose attention projections "
            "stay dense, e.g. 6,15."
        ),
    )
    parser.add_argument(
        "--token-dense-layer-policy",
        choices=("all_sparse", "keep_first", "keep_last", "keep_first_last"),
        default="all_sparse",
    )
    parser.add_argument("--token-dense-keep-n", type=int, default=0)
    parser.add_argument(
        "--token-dense-dense-ratio",
        type=float,
        default=0.125,
        help=(
            "Dynamic budget ratio over scored draft-verification rows. Used "
            "only by token_dense_dynamic."
        ),
    )
    parser.add_argument(
        "--token-dense-dense-min-per-request",
        type=int,
        default=1,
        help=(
            "Dynamic budget floor per active request. Forced first/bonus rows "
            "are excluded from this budget."
        ),
    )
    parser.add_argument(
        "--token-dense-dense-cap",
        type=int,
        default=-1,
        help="Dynamic dense-row cap; -1 disables the cap.",
    )
    parser.add_argument(
        "--token-dense-dense-selection",
        choices=(
            "batch_adaptive",
            "balanced_confidence",
            "balanced_low_confidence",
            "balanced_prefix",
            "batch_alternating",
            "batch_confidence",
            "highest",
            "lowest",
            "request_highest",
            "request_lowest",
            "request_contiguous",
        ),
        default="highest",
    )
    parser.add_argument(
        "--token-dense-batch-confidence-threshold",
        type=float,
        default=0.364,
    )
    parser.add_argument(
        "--token-dense-adaptive-dense-max-requests",
        type=int,
        default=1,
        help=(
            "For batch_adaptive routing, verification batches at or below "
            "this active-request count stay dense; larger batches use 2:4."
        ),
    )
    parser.add_argument(
        "--token-dense-batch-route-block-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--token-dense-batch-route-initial-credit",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--token-dense-balanced-start-position",
        type=int,
        default=0,
        help=(
            "First per-request draft position protected by balanced routing; "
            "later positions wrap around after the final draft position."
        ),
    )
    parser.add_argument(
        "--token-dense-sparse-bonus",
        action="store_true",
        help=(
            "Route verifier bonus rows through 2:4 so the fixed dense budget "
            "can protect more frequently consumed draft-verification rows."
        ),
    )
    parser.add_argument(
        "--token-dense-sparse-unscored-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route decode rows without drafter confidence through 2:4 while "
            "keeping prefill rows dense."
        ),
    )
    parser.add_argument(
        "--token-dense-score-backend",
        choices=("torch_softmax", "triton_selected", "triton_fused"),
        default="triton_fused",
    )
    parser.add_argument(
        "--token-dense-fast-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-reuse-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-contiguous-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-sparse-value-scale",
        type=float,
        default=1.0,
        help="Scale retained 2:4 values at prepack time without runtime work.",
    )
    parser.add_argument(
        "--token-dense-gate-up-value-scale",
        type=float,
        default=1.0,
        help=(
            "Scale retained gate/up 2:4 values at prepack time without "
            "changing attention or down projections."
        ),
    )
    parser.add_argument(
        "--token-dense-group-reconstruction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reconstruct retained 2:4 weights from the fixed grouped C4 "
            "covariance cache before CUTLASS prepack."
        ),
    )
    parser.add_argument(
        "--token-dense-group-covariance-cache",
        type=Path,
        help=(
            "Explicit grouped C4 covariance cache. Defaults to the model's "
            "gate_group_covariances.pt cache."
        ),
    )
    parser.add_argument(
        "--token-dense-row-scale-mode",
        choices=("none", "cache", "variance"),
        default="cache",
        help=(
            "Disable row scaling, use cached scales, or compute "
            "activation-weighted variance-preserving scales during prepack."
        ),
    )
    parser.add_argument(
        "--token-dense-variance-scale-projection-policy",
        choices=(
            "none",
            "all",
            "mlp",
            "attention",
            "gate_up",
            "down",
            "qkv",
            "o",
            "o_gate_up",
            "qkv_down",
            "qkv_gate_up_down",
            "o_down",
            "attention_down",
            "attention_gate_up",
        ),
        default="all",
        help="Projection subset that receives variance-preserving row scales.",
    )
    parser.add_argument(
        "--token-dense-row-scale-max",
        type=float,
        default=1.25,
        help="Upper clamp for variance-preserving per-row scales.",
    )
    parser.add_argument(
        "--token-dense-sparse-output-mode",
        choices=(
            "contiguous",
            "fused_mlp",
            "view_mlp",
            "view_mlp_o",
        ),
        default="contiguous",
        help=(
            "Keep every sparse output row-major or preserve CUTLASS views for "
            "MLP and attention output projections."
        ),
    )
    parser.add_argument(
        "--token-dense-sparse-accumulator",
        choices=(
            "fp32",
            "fp16",
            "fp16_gate",
            "fp16_gate_down",
            "fp16_qkv_gate",
        ),
        default="fp32",
        help=(
            "Accumulator policy for supported CUTLASS sparse decode tiles; "
            "fp16_gate keeps QKV/output/down on FP32; fp16_gate_down keeps "
            "QKV/output on FP32; fp16_qkv_gate keeps output/down on FP32."
        ),
    )
    parser.add_argument(
        "--token-dense-gate-up-hybrid",
        choices=("none", "up_sparse", "gate_sparse"),
        default="none",
    )
    parser.set_defaults(token_dense_sparse_backend="cutlass")
    parser.add_argument("--token-dense-mask-root", type=Path, default=DEFAULT_TOKEN_DENSE_MASK_ROOT)
    parser.add_argument(
        "--token-dense-mask-method",
        choices=(
            "wanda",
            "covwanda",
            "proxsparse",
            "maskllm",
            "inherit",
            "none",
        ),
        default="wanda",
    )
    parser.add_argument("--token-dense-enforce-eager", action="store_true")
    parser.add_argument(
        "--no-token-dense-enforce-eager",
        dest="token_dense_enforce_eager",
        action="store_false",
    )
    parser.set_defaults(token_dense_enforce_eager=False)
    parser.add_argument(
        "--token-dense-cudagraph-mode",
        choices=("none", "full", "full_decode_only"),
        default="none",
        help=(
            "CUDA graph mode for token-dense routing. full matches vLLM's "
            "default FULL_AND_PIECEWISE mode."
        ),
    )
    parser.add_argument(
        "--token-dense-compilation-mode",
        choices=("none", "vllm_compile"),
        default="none",
    )
    parser.add_argument(
        "--token-dense-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-isolate-decode-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Isolate pure decode verification batches. Disabled by default "
            "to preserve vLLM continuous batching; mixed batches fall back "
            "to dense target execution."
        ),
    )
    parser.add_argument(
        "--token-dense-release-dense-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--token-dense-retain-dense-weight",
        choices=("none", "qkv", "attention"),
        default="none",
    )
    parser.add_argument(
        "--attention-backend",
        default="auto",
        help="Common vLLM attention backend for dense and SpecLink runs.",
    )
    parser.add_argument(
        "--production-fast",
        action="store_true",
        help="Disable token-dense debug stats and use static hot-path env reads.",
    )
    parser.add_argument("--port-base", type=int, default=8260)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--server-warmup-batches", type=int, default=1)
    parser.add_argument("--server-warmup-max-tokens", type=int, default=16)
    parser.add_argument(
        "--server-warmup-lm-eval-batches", type=int, default=0
    )
    parser.add_argument("--server-shutdown-settle-s", type=float, default=2.0)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--calibration-cache-root", type=Path, default=DEFAULT_C4_CALIBRATION_CACHE_ROOT)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--allow-unsafe-code", action="store_true")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--aggregate", action="store_true", default=True)
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
