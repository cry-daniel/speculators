#!/usr/bin/env python3
"""Run TLM-only activation-aware 2:4 quality under vLLM EAGLE3 serving.

The large target model is sparsified in vLLM by SPECLINK_STRUCTURED_24_* env
vars. The EAGLE3 drafter remains dense. PPL is computed as reference-target
TLM loss with the same dense/sparse TLM mask, while task accuracy is generated
through vLLM speculative decoding.

Example:
  conda run -n spec python scripts/run_structured_24_spec_quality.py --smoke

Full default matrix:
  conda run -n spec python scripts/run_structured_24_spec_quality.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
SPECLINK_ROOT = SPECULATORS_ROOT.parent
MODELS_ROOT = SPECLINK_ROOT / "models"
RESULTS_ROOT = EVAL_ROOT / "results"
RESULTS_BAK_ROOT = EVAL_ROOT / "results.bak"

sys.path.insert(0, str(SCRIPT_DIR))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    LAYER_SENSITIVITY_DEFAULT_MODELS,
    QUALITY_MASK_TARGETS,
    append_jsonl,
    apply_24_mask_to_model,
    dtype_from_arg,
    ensure_quality_dependencies,
    evaluate_dataset,
    extract_final_answer,
    failed_metric,
    group_target_modules_by_layer,
    load_activation_cache,
    load_datasets,
    load_model_and_tokenizer,
    metric_value,
    parse_csv_list,
    parse_layer_indices,
    parse_model_id_overrides,
    restore_module_weights,
    set_seed,
    snapshot_module_weights,
    write_json,
)

EAGLE3_SPECULATORS = {
    "qwen3_8b": str(MODELS_ROOT / "qwen3-8b-eagle3-speculator"),
    "llama3_1_8b": str(MODELS_ROOT / "llama-3.1-8b-eagle3-speculator"),
}

DEFAULT_BASE_MODELS = {
    "qwen3_8b": str(MODELS_ROOT / "qwen3-8b"),
    "llama3_1_8b": str(MODELS_ROOT / "llama-3.1-8b-instruct"),
}

ACCURACY_DATASETS = {"gsm8k", "math_reasoning"}
PPL_DATASETS = {"mtbench", "dolly"}
SPEC_COUNTER_RE = (
    r"^(vllm:spec_decode_num_(?:accepted_tokens|draft_tokens))"
    r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"
)


@dataclass(frozen=True)
class SparseCase:
    label: str
    group: str
    policy: str
    layer_index: int | None = None
    keep_n: int = 0


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def add_local_no_proxy(env: dict[str, str]) -> dict[str, str]:
    merged = env.copy()
    for key in [
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_SHLVL",
        "PYTHONPATH",
    ]:
        merged.pop(key, None)
    current = [item for item in merged.get("NO_PROXY", "").split(",") if item]
    for item in ["localhost", "127.0.0.1", "0.0.0.0"]:
        if item not in current:
            current.append(item)
    merged["NO_PROXY"] = ",".join(current)
    merged["no_proxy"] = merged["NO_PROXY"]
    return merged


def configure_local_no_proxy() -> None:
    merged = add_local_no_proxy(os.environ.copy())
    os.environ["NO_PROXY"] = merged["NO_PROXY"]
    os.environ["no_proxy"] = merged["no_proxy"]


def find_free_port(start: int) -> int:
    for port in range(start, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free TCP port found")


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    pgid = process.pid
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        pass

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and group_exists():
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.25)
    if group_exists():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        kill_deadline = time.monotonic() + 10
        while time.monotonic() < kill_deadline and group_exists():
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.25)


def wait_for_health(port: int, process: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2.0)
    raise TimeoutError(f"server on port {port} did not become healthy")


def scrape_spec_metrics(port: int) -> dict[str, float]:
    import re

    pattern = re.compile(SPEC_COUNTER_RE)
    metrics: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return metrics
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            metrics[match.group(1)] = metrics.get(match.group(1), 0.0) + float(
                match.group(2)
            )
    return metrics


def metric_delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    if key not in after:
        return None
    return after[key] - before.get(key, 0.0)


def post_completion(
    *,
    port: int,
    model_id: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    timeout_s: float,
) -> tuple[str, int | None, str]:
    body = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "request_id": request_id,
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
        return "", None, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return "", None, repr(exc)
    choices = data.get("choices") or []
    text = str(choices[0].get("text", "")) if choices else ""
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return text, int(completion_tokens) if completion_tokens is not None else None, ""


def build_cases(layers: list[int], args: argparse.Namespace) -> list[SparseCase]:
    cases: list[SparseCase] = [SparseCase("dense", "dense", "dense")]
    if args.include_layer_sensitivity:
        for layer in layers:
            cases.append(
                SparseCase(
                    f"layer_{layer:02d}",
                    "layer_sensitivity",
                    "single_layer",
                    layer_index=layer,
                )
            )
    if args.include_dense_keep:
        cases.append(SparseCase("all_sparse", "dense_keep", "all_sparse"))
        for keep_n in (1, 2, 3):
            cases.append(SparseCase(f"keep_first_{keep_n}", "dense_keep", "keep_first", keep_n=keep_n))
            cases.append(SparseCase(f"keep_last_{keep_n}", "dense_keep", "keep_last", keep_n=keep_n))
            cases.append(
                SparseCase(
                    f"keep_first_last_{keep_n}",
                    "dense_keep",
                    "keep_first_last",
                    keep_n=keep_n,
                )
            )
    return cases


def case_env(
    args: argparse.Namespace,
    *,
    model_label: str,
    case: SparseCase,
    stats_path: Path,
) -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    if case.policy == "dense":
        env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
        return env
    env.update(
        {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": model_label,
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(
                args.calibration_cache_root.resolve()
            ),
            "SPECLINK_STRUCTURED_24_POLICY": case.policy,
            "SPECLINK_STRUCTURED_24_KEEP_N": str(case.keep_n),
            "SPECLINK_STRUCTURED_24_STATS_PATH": str(stats_path.resolve()),
        }
    )
    if case.layer_index is not None:
        env["SPECLINK_STRUCTURED_24_LAYER_INDEX"] = str(case.layer_index)
    return env


def build_vllm_command(
    args: argparse.Namespace,
    *,
    base_model: str,
    speculator_model: str,
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
        base_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--seed",
        str(args.seed),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--generation-config",
        "vllm",
        "--speculative-config",
        json.dumps(
            {
                "model": speculator_model,
                "num_speculative_tokens": args.num_spec_tokens,
                "method": "eagle3",
                "max_model_len": args.max_model_len,
            }
        ),
    ]
    if getattr(args, "enforce_eager", False):
        command.append("--enforce-eager")
    enable_prefix_caching = getattr(args, "enable_prefix_caching", None)
    if enable_prefix_caching is True:
        command.append("--enable-prefix-caching")
    elif enable_prefix_caching is False:
        command.append("--no-enable-prefix-caching")
    compilation_config = getattr(args, "compilation_config", None)
    if compilation_config is not None:
        command.extend(
            ["--compilation-config", json.dumps(compilation_config, separators=(",", ":"))]
        )
    return command


def start_vllm_server(
    args: argparse.Namespace,
    *,
    base_model: str,
    speculator_model: str,
    case_dir: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[Any], int]:
    port = find_free_port(args.port_base)
    command = build_vllm_command(
        args,
        base_model=base_model,
        speculator_model=speculator_model,
        port=port,
    )
    write_json(case_dir / "server_command.json", {"command": command, "port": port})
    log_path = case_dir / "vllm_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("$ " + " ".join(command) + "\n\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=str(EVAL_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_health(port, process, args.health_timeout_s)
    except Exception:
        stop_process(process)
        raise
    return process, port


def apply_case_mask_offline(
    model: Any,
    *,
    model_label: str,
    case: SparseCase,
    calibration_cache_root: Path,
    layer_modules: dict[int, list[str]],
) -> dict[str, Any]:
    if case.policy == "dense":
        return {
            "policy": "dense",
            "total_masked_weight_count": 0,
            "zeroed_weight_count": 0,
            "actual_sparsity": 0.0,
            "effective_sparse_fraction": 0.0,
            "dense_keep_module_names": [],
            "masked_module_names": [],
            "per_module": [],
        }
    activation_scales, calibration_stats = load_activation_cache(
        calibration_cache_root,
        model_label,
        required_modules=QUALITY_MASK_TARGETS["all"],
    )
    dense_keep_modules: set[str] = set()
    only_module_names: set[str] | None = None
    if case.policy == "single_layer":
        if case.layer_index is None:
            raise RuntimeError("single_layer case missing layer_index")
        only_module_names = set(layer_modules[case.layer_index])
    elif case.policy in {"keep_first", "keep_last", "keep_first_last"}:
        layers = sorted(layer_modules)
        keep_layers: set[int] = set()
        if case.policy in {"keep_first", "keep_first_last"}:
            keep_layers.update(layers[: case.keep_n])
        if case.policy in {"keep_last", "keep_first_last"}:
            keep_layers.update(layers[-case.keep_n :])
        dense_keep_modules = {
            module_name
            for layer in keep_layers
            for module_name in layer_modules.get(layer, [])
        }
    mask_stats = apply_24_mask_to_model(
        model,
        QUALITY_MASK_TARGETS["all"],
        group_dim="in",
        skip_lm_head=True,
        skip_embeddings=True,
        mask_method="activation_aware",
        activation_scales=activation_scales,
        dense_keep_modules=dense_keep_modules,
        only_module_names=only_module_names,
    )
    mask_stats["policy"] = case.policy
    mask_stats["layer_index"] = case.layer_index
    mask_stats["keep_n"] = case.keep_n
    mask_stats["calibration"] = calibration_stats
    return mask_stats


def run_ppl_metrics(
    args: argparse.Namespace,
    *,
    model_label: str,
    model_id: str,
    case: SparseCase,
    datasets: dict[str, Any],
    case_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_quality_dependencies()
    dtype = dtype_from_arg(args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype,
        args.device,
        args.trust_remote_code,
        args.local_files_only,
    )
    try:
        layer_modules = group_target_modules_by_layer(model, QUALITY_MASK_TARGETS["all"])
        snapshots = {}
        if case.policy == "single_layer" and case.layer_index is not None:
            snapshots = snapshot_module_weights(model, layer_modules[case.layer_index])
        mask_stats = apply_case_mask_offline(
            model,
            model_label=model_label,
            case=case,
            calibration_cache_root=args.calibration_cache_root,
            layer_modules=layer_modules,
        )
        write_json(case_dir / "offline_mask_stats.json", mask_stats)

        metrics: dict[str, Any] = {}
        for dataset_name, pack in datasets.items():
            if dataset_name not in PPL_DATASETS:
                continue
            started = time.time()
            try:
                metric = evaluate_dataset(
                    dataset_name,
                    model,
                    tokenizer,
                    pack,
                    args,
                    case_dir,
                    args.output_root,
                    model_label,
                    "none" if case.policy == "dense" else "all",
                )
            except Exception as exc:  # noqa: BLE001
                metric = failed_metric(dataset_name, str(exc))
            metric["elapsed_sec"] = round(time.time() - started, 3)
            metric["metric_source"] = "offline_tlm_reference_loss_same_mask"
            metric.setdefault("failed", False)
            metric.setdefault("error", "")
            metrics[dataset_name] = metric
        if snapshots:
            restore_module_weights(model, snapshots)
        return metrics, mask_stats
    finally:
        del model
        del tokenizer
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def max_tokens_for_dataset(args: argparse.Namespace, dataset_name: str) -> int:
    if dataset_name == "gsm8k":
        return args.accuracy_max_tokens
    if dataset_name == "math_reasoning":
        return args.accuracy_max_tokens
    return args.accuracy_max_tokens


def run_accuracy_dataset(
    args: argparse.Namespace,
    *,
    port: int,
    model_id: str,
    model_label: str,
    case: SparseCase,
    dataset_name: str,
    pack: Any,
    case_dir: Path,
) -> dict[str, Any]:
    if pack.error:
        return failed_metric(dataset_name, f"dataset_load_failed: {pack.error}")
    details: list[dict[str, Any]] = []
    started = time.time()

    def evaluate_one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        gold = row.get("gold_answer")
        request_id = f"s24-{model_label}-{case.label}-{dataset_name}-{idx:05d}"
        text, completion_tokens, error = post_completion(
            port=port,
            model_id=model_id,
            prompt=row["prompt"],
            max_tokens=max_tokens_for_dataset(args, dataset_name),
            request_id=request_id,
            timeout_s=args.request_timeout_s,
        )
        if error:
            return {
                "idx": idx,
                "id": row.get("id"),
                "request_id": request_id,
                "error": error,
                "counted": False,
                "correct": None,
                "completion_tokens": 0,
            }
        pred = extract_final_answer(text)
        is_counted = gold is not None
        is_correct = bool(is_counted and pred is not None and pred == gold)
        return {
            "idx": idx,
            "id": row.get("id"),
            "request_id": request_id,
            "prompt": row.get("prompt"),
            "generation": text,
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": is_correct if is_counted else None,
            "counted": is_counted,
            "completion_tokens": completion_tokens,
            "error": "",
        }

    workers = max(1, int(args.accuracy_concurrency))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(evaluate_one, idx, row)
            for idx, row in enumerate(pack.rows)
        ]
        for future in as_completed(futures):
            details.append(future.result())

    details.sort(key=lambda item: int(item.get("idx", 0)))
    counted = sum(1 for item in details if item.get("counted"))
    correct = sum(1 for item in details if item.get("counted") and item.get("correct"))
    output_tokens = sum(int(item.get("completion_tokens") or 0) for item in details)
    append_jsonl(case_dir / f"generations_{dataset_name}.jsonl", details)
    accuracy = correct / counted if counted else None
    metric_name = "gsm8k_accuracy" if dataset_name == "gsm8k" else "math_accuracy"
    return {
        "metric_name": metric_name,
        "metric_type": "accuracy",
        "value": accuracy,
        metric_name: accuracy,
        "correct": correct,
        "num_examples": counted,
        "accuracy_available": counted > 0,
        "output_tokens": output_tokens,
        "elapsed_sec": round(time.time() - started, 3),
        "metric_source": "vllm_eagle3_speculative_generation",
        "failed": False,
        "error": "",
    }


def run_accuracy_metrics(
    args: argparse.Namespace,
    *,
    model_label: str,
    model_id: str,
    speculator_model: str,
    case: SparseCase,
    datasets: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    if not any(name in ACCURACY_DATASETS for name in datasets):
        return {}
    stats_path = case_dir / "vllm_structured_24_stats.json"
    env = case_env(args, model_label=model_label, case=case, stats_path=stats_path)
    process = None
    port = -1
    metrics: dict[str, Any] = {}
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
            if dataset_name not in ACCURACY_DATASETS:
                continue
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
            write_json(case_dir / "accuracy_metrics.partial.json", metrics)
        after = scrape_spec_metrics(port)
        accepted = metric_delta(before, after, "vllm:spec_decode_num_accepted_tokens")
        drafted = metric_delta(before, after, "vllm:spec_decode_num_draft_tokens")
        write_json(
            case_dir / "spec_counters.json",
            {
                "accepted_tokens": accepted,
                "draft_tokens": drafted,
                "acceptance_rate": accepted / drafted if accepted is not None and drafted else None,
            },
        )
        for metric in metrics.values():
            metric["spec_accepted_tokens_case"] = accepted
            metric["spec_draft_tokens_case"] = drafted
            metric["spec_acceptance_rate_case"] = (
                accepted / drafted if accepted is not None and drafted else None
            )
    finally:
        stop_process(process)
        if args.server_shutdown_settle_s > 0:
            time.sleep(args.server_shutdown_settle_s)
    return metrics


def metric_row(
    *,
    model_label: str,
    model_id: str,
    case: SparseCase,
    dataset_name: str,
    dense_metric: dict[str, Any],
    sparse_metric: dict[str, Any],
    mask_stats: dict[str, Any],
) -> dict[str, Any]:
    dense_value = metric_value(dense_metric)
    sparse_value = metric_value(sparse_metric)
    metric_type = sparse_metric.get("metric_type", dense_metric.get("metric_type", ""))
    delta = None
    ratio = None
    accuracy_drop = None
    if dense_value is not None and sparse_value is not None:
        delta = sparse_value - dense_value
        if metric_type == "ppl" and dense_value:
            ratio = sparse_value / dense_value
        elif metric_type in {"accuracy", "pass_at_1"}:
            accuracy_drop = dense_value - sparse_value
    return {
        "model_label": model_label,
        "model_id": model_id,
        "case": case.label,
        "case_group": case.group,
        "policy": case.policy,
        "layer_index": case.layer_index,
        "keep_n": case.keep_n,
        "dataset": dataset_name,
        "metric_name": sparse_metric.get("metric_name", dense_metric.get("metric_name", "")),
        "metric_type": metric_type,
        "dense_metric_value": dense_value,
        "sparse_metric_value": sparse_value,
        "delta_vs_dense": delta,
        "ratio_vs_dense": ratio,
        "accuracy_drop": accuracy_drop,
        "num_examples": sparse_metric.get("num_examples", dense_metric.get("num_examples", 0)),
        "metric_source": sparse_metric.get("metric_source", ""),
        "ppl_mode": sparse_metric.get("ppl_mode", dense_metric.get("ppl_mode", "")),
        "spec_acceptance_rate_case": sparse_metric.get("spec_acceptance_rate_case"),
        "spec_accepted_tokens_case": sparse_metric.get("spec_accepted_tokens_case"),
        "spec_draft_tokens_case": sparse_metric.get("spec_draft_tokens_case"),
        "zeroed_weight_count": mask_stats.get("zeroed_weight_count", 0),
        "total_masked_weight_count": mask_stats.get("total_masked_weight_count", 0),
        "effective_sparse_fraction": mask_stats.get("effective_sparse_fraction", 0.0),
        "failed": bool(sparse_metric.get("failed", False)),
        "error": sparse_metric.get("error", ""),
    }


CSV_FIELDS = [
    "model_label",
    "model_id",
    "case",
    "case_group",
    "policy",
    "layer_index",
    "keep_n",
    "dataset",
    "metric_name",
    "metric_type",
    "dense_metric_value",
    "sparse_metric_value",
    "delta_vs_dense",
    "ratio_vs_dense",
    "accuracy_drop",
    "num_examples",
    "metric_source",
    "ppl_mode",
    "spec_acceptance_rate_case",
    "spec_accepted_tokens_case",
    "spec_draft_tokens_case",
    "zeroed_weight_count",
    "total_masked_weight_count",
    "effective_sparse_fraction",
    "failed",
    "error",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def plot_results(rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    valid = [row for row in rows if not row.get("failed")]
    if not valid:
        return []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    layer_rows = [row for row in valid if row.get("case_group") == "layer_sensitivity"]
    if layer_rows:
        models = sorted({str(row["model_label"]) for row in layer_rows})
        fig, axes = plt.subplots(
            len(models),
            2,
            squeeze=False,
            figsize=(13, max(3.8, 3.4 * len(models))),
            constrained_layout=True,
        )
        for row_idx, model_label in enumerate(models):
            model_rows = [row for row in layer_rows if row["model_label"] == model_label]
            ppl_ax = axes[row_idx][0]
            acc_ax = axes[row_idx][1]
            for dataset in sorted({row["dataset"] for row in model_rows if row["metric_type"] == "ppl"}):
                pairs = sorted(
                    (
                        int(row["layer_index"]),
                        safe_float(row.get("ratio_vs_dense")),
                    )
                    for row in model_rows
                    if row["dataset"] == dataset and row["metric_type"] == "ppl"
                )
                pairs = [(x, y) for x, y in pairs if y is not None]
                if pairs:
                    ppl_ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=dataset)
            for dataset in sorted({row["dataset"] for row in model_rows if row["metric_type"] == "accuracy"}):
                pairs = sorted(
                    (
                        int(row["layer_index"]),
                        safe_float(row.get("accuracy_drop")),
                    )
                    for row in model_rows
                    if row["dataset"] == dataset and row["metric_type"] == "accuracy"
                )
                pairs = [(x, y) for x, y in pairs if y is not None]
                if pairs:
                    acc_ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=dataset)
            ppl_ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            acc_ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ppl_ax.set_title(f"{model_label}: PPL loss by single sparse layer")
            acc_ax.set_title(f"{model_label}: ACC drop by single sparse layer")
            ppl_ax.set_xlabel("Transformer layer")
            acc_ax.set_xlabel("Transformer layer")
            ppl_ax.set_ylabel("PPL ratio vs dense")
            acc_ax.set_ylabel("Accuracy drop vs dense")
            ppl_ax.grid(True, alpha=0.25)
            acc_ax.grid(True, alpha=0.25)
            ppl_ax.legend(loc="best")
            acc_ax.legend(loc="best")
        path = fig_dir / "layer_sensitivity_spec.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    keep_rows = [row for row in valid if row.get("case_group") == "dense_keep"]
    if keep_rows:
        models = sorted({str(row["model_label"]) for row in keep_rows})
        fig, axes = plt.subplots(
            len(models),
            2,
            squeeze=False,
            figsize=(14, max(4.0, 3.6 * len(models))),
            constrained_layout=True,
        )
        for row_idx, model_label in enumerate(models):
            model_rows = [row for row in keep_rows if row["model_label"] == model_label]
            labels = []
            ppl_values = []
            acc_values = []
            for case in [
                "all_sparse",
                "keep_first_1",
                "keep_first_2",
                "keep_first_3",
                "keep_last_1",
                "keep_last_2",
                "keep_last_3",
                "keep_first_last_1",
                "keep_first_last_2",
                "keep_first_last_3",
            ]:
                case_rows = [row for row in model_rows if row["case"] == case]
                ppl = [
                    safe_float(row.get("ratio_vs_dense"))
                    for row in case_rows
                    if row["metric_type"] == "ppl"
                ]
                acc = [
                    safe_float(row.get("accuracy_drop"))
                    for row in case_rows
                    if row["metric_type"] == "accuracy"
                ]
                if ppl or acc:
                    labels.append(case)
                    ppl_values.append(mean_or_none(ppl) or 0.0)
                    acc_values.append(mean_or_none(acc) or 0.0)
            x = list(range(len(labels)))
            axes[row_idx][0].bar(x, ppl_values)
            axes[row_idx][0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axes[row_idx][0].set_title(f"{model_label}: avg PPL ratio")
            axes[row_idx][0].set_xticks(x, labels, rotation=45, ha="right")
            axes[row_idx][0].grid(True, axis="y", alpha=0.25)
            axes[row_idx][1].bar(x, acc_values)
            axes[row_idx][1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            axes[row_idx][1].set_title(f"{model_label}: avg ACC drop")
            axes[row_idx][1].set_xticks(x, labels, rotation=45, ha="right")
            axes[row_idx][1].grid(True, axis="y", alpha=0.25)
        path = fig_dir / "dense_keep_spec.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def summarize_scores(rows: list[dict[str, Any]], metric_type: str, field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("metric_type") != metric_type or row.get("failed"):
            continue
        value = safe_float(row.get(field))
        if value is None:
            continue
        grouped.setdefault((str(row["model_label"]), str(row["case"])), []).append(value)
    out = []
    for (model_label, case), values in grouped.items():
        if values:
            out.append(
                {
                    "model_label": model_label,
                    "case": case,
                    "score": sum(values) / len(values),
                    "datasets": len(values),
                }
            )
    out.sort(key=lambda row: (row["model_label"], -row["score"], row["case"]))
    return out


def write_summary(output_root: Path, rows: list[dict[str, Any]], figures: list[Path], args: argparse.Namespace) -> None:
    layer_rows = [row for row in rows if row.get("case_group") == "layer_sensitivity"]
    keep_rows = [row for row in rows if row.get("case_group") == "dense_keep"]
    ppl_layer_scores = summarize_scores(layer_rows, "ppl", "ratio_vs_dense")
    acc_layer_scores = summarize_scores(layer_rows, "accuracy", "accuracy_drop")
    ppl_keep_scores = summarize_scores(keep_rows, "ppl", "ratio_vs_dense")
    acc_keep_scores = summarize_scores(keep_rows, "accuracy", "accuracy_drop")

    def write_top_scores(
        handle: Any,
        title: str,
        scores: list[dict[str, Any]],
        score_label: str,
        *,
        limit_per_model: int,
    ) -> None:
        handle.write(f"{title}\n\n")
        handle.write(f"| model | case | {score_label} | datasets |\n")
        handle.write("|---|---|---:|---:|\n")
        for model_label in sorted({str(row["model_label"]) for row in scores}):
            model_scores = [
                row for row in scores if str(row["model_label"]) == model_label
            ]
            model_scores.sort(key=lambda row: (-float(row["score"]), str(row["case"])))
            for row in model_scores[:limit_per_model]:
                handle.write(
                    f"| {row['model_label']} | {row['case']} | "
                    f"{row['score']:.6f} | {row['datasets']} |\n"
                )
        handle.write("\n")

    with (output_root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# TLM-only 2:4 Under EAGLE3 Speculative Inference\n\n")
        handle.write(f"Output root: `{output_root.resolve()}`\n\n")
        handle.write("## Method\n\n")
        handle.write(
            "- Serving accuracy uses vLLM + EAGLE3 speculative decoding with K=8.\n"
            "- 2:4 activation-aware masks are applied only to the TLM/base model; the EAGLE3 drafter stays dense.\n"
            "- PPL is dense-vs-sparse TLM reference loss with the same mask policy, not a generated-output metric.\n"
            "- ACC means task accuracy from generated answers on GSM8K and math_reasoning.\n\n"
        )
        handle.write("## Inputs\n\n")
        handle.write(f"- models: `{args.models}`\n")
        handle.write(f"- datasets: `{args.datasets}`\n")
        handle.write(f"- calibration cache: `{args.calibration_cache_root.resolve()}`\n")
        handle.write(f"- num spec tokens: `{args.num_spec_tokens}`\n")
        handle.write(f"- max model len: `{args.max_model_len}`\n\n")
        if figures:
            handle.write("## Figures\n\n")
            for figure in figures:
                handle.write(f"- `{figure.resolve()}`\n")
            handle.write("\n")
        write_top_scores(
            handle,
            "## Most Sensitive Single Layers By PPL",
            ppl_layer_scores,
            "avg PPL ratio",
            limit_per_model=8,
        )
        write_top_scores(
            handle,
            "## Most Sensitive Single Layers By ACC",
            acc_layer_scores,
            "avg ACC drop",
            limit_per_model=8,
        )
        write_top_scores(
            handle,
            "## Dense Keep Policies By PPL Loss",
            ppl_keep_scores,
            "avg PPL ratio",
            limit_per_model=20,
        )
        write_top_scores(
            handle,
            "## Dense Keep Policies By ACC Drop",
            acc_keep_scores,
            "avg ACC drop",
            limit_per_model=20,
        )
        handle.write("## Files\n\n")
        handle.write("- `structured_24_spec_quality.csv`: all rows.\n")
        handle.write("- `layer_sensitivity.csv`: single-layer rows.\n")
        handle.write("- `dense_keep_compare.csv`: all-sparse and dense-keep rows.\n")
        handle.write("- `runs/*/metrics.json`: per-case raw metrics and mask stats.\n")


def case_complete(case_dir: Path) -> bool:
    metrics_path = case_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("status") != "ok":
        return False
    metrics = data.get("datasets", {})
    if not isinstance(metrics, dict):
        return False
    return not any(
        isinstance(metric, dict) and metric.get("failed")
        for metric in metrics.values()
    )


def load_case_metrics(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    return data.get("datasets", {}), data.get("mask_stats", {})


def discover_layers(args: argparse.Namespace, model_label: str, model_id: str) -> list[int]:
    ensure_quality_dependencies()
    dtype = dtype_from_arg(args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype,
        args.device,
        args.trust_remote_code,
        args.local_files_only,
    )
    try:
        grouped = group_target_modules_by_layer(model, QUALITY_MASK_TARGETS["all"])
        requested = parse_layer_indices(args.layers)
        layers = requested or sorted(grouped)
        return [layer for layer in layers if layer in grouped]
    finally:
        del model
        del tokenizer
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def configure_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.output_root = args.output_root or (
        RESULTS_BAK_ROOT / f"structured_24_spec_quality_smoke_{timestamp()}"
    )
    args.models = "qwen3_8b"
    args.layers = "0"
    args.datasets = "mtbench,dolly,gsm8k,math_reasoning"
    args.mtbench_num_examples = 2
    args.dolly_num_examples = 2
    args.gsm8k_num_examples = 2
    args.math_num_examples = 2


def run(args: argparse.Namespace) -> None:
    configure_local_no_proxy()
    configure_smoke(args)
    output_root = args.output_root or (
        RESULTS_ROOT / f"structured_24_spec_tlm_eagle3_k8_{timestamp()}"
    )
    args.output_root = output_root
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_quality_dependencies()
    set_seed(args.seed)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    base_models = dict(DEFAULT_BASE_MODELS)
    base_models.update(LAYER_SENSITIVITY_DEFAULT_MODELS)
    base_models.update(parse_model_id_overrides(args.model_id))
    speculators = dict(EAGLE3_SPECULATORS)
    speculators.update(parse_model_id_overrides(args.speculator_model))
    selected_models = parse_csv_list(args.models)
    datasets = load_datasets(args)

    write_json(
        output_root / "run_config.json",
        {
            "argv": sys.argv,
            "models": selected_models,
            "base_models": base_models,
            "speculators": speculators,
            "datasets": parse_csv_list(args.datasets),
            "num_spec_tokens": args.num_spec_tokens,
            "calibration_cache_root": str(args.calibration_cache_root.resolve()),
            "created_at": timestamp(),
            "smoke": args.smoke,
        },
    )

    all_rows: list[dict[str, Any]] = []
    dense_metrics_by_model: dict[str, dict[str, Any]] = {}

    for model_label in selected_models:
        model_id = base_models.get(model_label)
        speculator_model = speculators.get(model_label)
        if not model_id:
            raise ValueError(f"unknown model label: {model_label}")
        if not speculator_model:
            raise ValueError(f"missing EAGLE3 speculator for {model_label}")
        layers = discover_layers(args, model_label, model_id)
        if not layers:
            raise RuntimeError(f"no layers discovered for {model_label}")
        cases = build_cases(layers, args)

        for case in cases:
            case_dir = output_root / "runs" / model_label / case.label
            case_dir.mkdir(parents=True, exist_ok=True)
            if args.resume and case_complete(case_dir):
                print(f"[SKIP] {model_label}/{case.label}", flush=True)
                metrics, mask_stats = load_case_metrics(case_dir)
            else:
                print(f"[RUN] {model_label}/{case.label}", flush=True)
                ppl_metrics, offline_mask_stats = run_ppl_metrics(
                    args,
                    model_label=model_label,
                    model_id=model_id,
                    case=case,
                    datasets=datasets,
                    case_dir=case_dir,
                )
                accuracy_metrics = run_accuracy_metrics(
                    args,
                    model_label=model_label,
                    model_id=model_id,
                    speculator_model=speculator_model,
                    case=case,
                    datasets=datasets,
                    case_dir=case_dir,
                )
                metrics = {**ppl_metrics, **accuracy_metrics}
                mask_stats = offline_mask_stats
                write_json(
                    case_dir / "metrics.json",
                    {
                        "status": "ok",
                        "model_label": model_label,
                        "model_id": model_id,
                        "speculator_model": speculator_model,
                        "case": case.__dict__,
                        "mask_stats": mask_stats,
                        "datasets": metrics,
                    },
                )

            if case.policy == "dense":
                dense_metrics_by_model[model_label] = metrics
                continue
            dense_metrics = dense_metrics_by_model.get(model_label)
            if dense_metrics is None:
                dense_dir = output_root / "runs" / model_label / "dense"
                if case_complete(dense_dir):
                    dense_metrics, _ = load_case_metrics(dense_dir)
                    dense_metrics_by_model[model_label] = dense_metrics
                else:
                    raise RuntimeError(f"dense baseline missing before {model_label}/{case.label}")
            for dataset_name, sparse_metric in metrics.items():
                row = metric_row(
                    model_label=model_label,
                    model_id=model_id,
                    case=case,
                    dataset_name=dataset_name,
                    dense_metric=dense_metrics.get(dataset_name, {}),
                    sparse_metric=sparse_metric,
                    mask_stats=mask_stats,
                )
                all_rows.append(row)
            write_outputs(output_root, all_rows, args)

    write_outputs(output_root, all_rows, args)
    print(output_root.resolve())


def write_outputs(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    write_csv(output_root / "structured_24_spec_quality.csv", rows)
    write_csv(
        output_root / "layer_sensitivity.csv",
        [row for row in rows if row.get("case_group") == "layer_sensitivity"],
    )
    write_csv(
        output_root / "dense_keep_compare.csv",
        [row for row in rows if row.get("case_group") == "dense_keep"],
    )
    figures = plot_results(rows, output_root)
    write_summary(output_root, rows, figures, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TLM-only 2:4 quality under vLLM EAGLE3 speculative decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--model-id", action="append", default=[], help="Override base model as LABEL=PATH_OR_ID.")
    parser.add_argument("--speculator-model", action="append", default=[], help="Override EAGLE3 model as LABEL=PATH_OR_ID.")
    parser.add_argument("--layers", default="", help="Layer indices or ranges for sensitivity; empty means all.")
    parser.add_argument("--datasets", default="mtbench,dolly,gsm8k,math_reasoning")
    parser.add_argument("--mtbench-num-examples", type=int, default=40)
    parser.add_argument("--dolly-num-examples", type=int, default=64)
    parser.add_argument("--gsm8k-num-examples", type=int, default=64)
    parser.add_argument("--math-num-examples", type=int, default=64)
    parser.add_argument("--humaneval-num-examples", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--accuracy-max-tokens", type=int, default=256)
    parser.add_argument("--accuracy-concurrency", type=int, default=4)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--port-base", type=int, default=8120)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--calibration-cache-root", type=Path, default=DEFAULT_C4_CALIBRATION_CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-layer-sensitivity", action="store_true", default=True)
    parser.add_argument("--skip-layer-sensitivity", dest="include_layer_sensitivity", action="store_false")
    parser.add_argument("--include-dense-keep", action="store_true", default=True)
    parser.add_argument("--skip-dense-keep", dest="include_dense_keep", action="store_false")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
