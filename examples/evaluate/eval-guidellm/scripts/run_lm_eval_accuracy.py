#!/usr/bin/env python3
"""Run lm-eval through this repo's vLLM/EAGLE3 serving path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
MODE_GROUPS = {
    "dense": ["dense_ar"],
    "speculative": ["eagle3_dense"],
    "hybrid": HYBRID_METHODS,
    "token_dense": TOKEN_DENSE_METHODS,
    "all": ["dense_ar", "eagle3_dense", *HYBRID_METHODS],
}
TASK_GROUPS = {
    "smoke": ["logiqa_generative", "gsm8k_cot"],
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
    "gsm8k_cot": 512,
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


def mode_method(mode: str) -> MethodConfig:
    if mode in {"dense_ar", "eagle3_dense"}:
        return MethodConfig(label="dense", base_method="dense", policy="dense")
    return parse_method_config(mode)


def mode_uses_spec(mode: str) -> bool:
    return mode != "dense_ar"


def mode_group(mode: str) -> str:
    if mode == "dense_ar":
        return "dense"
    if mode == "eagle3_dense":
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
    if args.enforce_eager or mode.startswith("token_dense_"):
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
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.apply_chat_template:
        command.append("--apply_chat_template")
    if task == "humaneval_instruct" and args.allow_unsafe_code:
        command.append("--confirm_run_unsafe_code")
    write_json(run_dir / "lm_eval_command.json", {"command": command})
    env = add_local_no_proxy(os.environ.copy())
    env["OPENAI_API_KEY"] = "EMPTY"
    if task == "humaneval_instruct" and args.allow_unsafe_code:
        env["HF_ALLOW_CODE_EVAL"] = "1"
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
                            "uses_speculative": mode_uses_spec(mode),
                            "spec_accepted_tokens": accepted if drafted else None,
                            "spec_draft_tokens": drafted if drafted else None,
                            "spec_acceptance_rate": accepted / drafted if drafted else None,
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
            "max_context_length": args.max_context_length,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
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
