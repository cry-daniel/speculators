#!/usr/bin/env python3
"""Run Llama-3.1 AR/FastDraft/Smurfs/EAGLE3 throughput matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any


SPECLINK_ROOT = Path("/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink")
SPECULATORS_ROOT = SPECLINK_ROOT / "speculators"
EVAL_ROOT = SPECULATORS_ROOT / "examples/evaluate/eval-guidellm"
BASE_MODEL = SPECLINK_ROOT / "models/llama-3.1-8b-instruct"
FASTDRAFT_MODEL = SPECLINK_ROOT / "models/llama-3.1-8b-fastdraft-150m-int8-hf"
EAGLE3_SPECULATOR_MODEL = SPECLINK_ROOT / "models/llama-3.1-8b-eagle3-speculator"
SMURFS_DYNAMIC_ENV_PREFIX = "SPECLINK_SMURFS_DYNAMIC_"

BASE_DATASETS = {
    "math_reasoning": EVAL_ROOT / "data/math_reasoning.jsonl",
    "mtbench": EVAL_ROOT / "data/mt_bench.jsonl",
    "gsm8k": EVAL_ROOT / "data/gsm8k.jsonl",
    "humaneval": EVAL_ROOT / "data/humaneval.jsonl",
}
DATASETS = BASE_DATASETS.copy()
METHODS = ("vllm_ar", "vllm_fastdraft", "smurfs_fastdraft", "vllm_eagle3")
PROM_COUNTER_RE = re.compile(
    r"^(vllm:spec_decode_num_(?:accepted_tokens|draft_tokens))"
    r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"
)


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


def find_free_port(start: int) -> int:
    for port in range(start, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free TCP port found")


def wait_for_health(port: int, process: subprocess.Popen[Any],
                    timeout_s: float) -> None:
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


def fetch_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def start_logged(command: list[str], *, cwd: Path, log_path: Path,
                 env: dict[str, str]) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("$ " + " ".join(command) + "\n\n")
    log_file.flush()
    return subprocess.Popen(command,
                            cwd=str(cwd),
                            env=env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                            preexec_fn=os.setsid)


def run_logged(command: list[str], *, cwd: Path, log_path: Path,
               env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(command,
                                   cwd=str(cwd),
                                   env=env,
                                   stdout=log_file,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with code {completed.returncode}; see {log_path}")


def stat_mean(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key, {}).get("successful", {}).get("mean")
    return float(value) if value is not None else None


def stat_sum(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key, {}).get("successful", {}).get("total_sum")
    return float(value) if value is not None else None


def parse_guidellm(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = data["benchmarks"][0]
    metrics = benchmark["metrics"]
    scheduler_state = benchmark.get("scheduler_state", {})
    return {
        "successful_requests": scheduler_state.get("successful_requests"),
        "errored_requests": scheduler_state.get("errored_requests"),
        "duration": benchmark.get("duration"),
        "output_tokens_per_second": stat_mean(metrics, "output_tokens_per_second"),
        "tokens_per_second": stat_mean(metrics, "tokens_per_second"),
        "request_latency_mean": stat_mean(metrics, "request_latency"),
        "ttft_ms_mean": stat_mean(metrics, "time_to_first_token_ms"),
        "tpot_ms_mean": stat_mean(metrics, "time_per_output_token_ms"),
        "prompt_tokens_total": stat_sum(metrics, "prompt_token_count"),
        "output_tokens_total": stat_sum(metrics, "output_token_count"),
        "total_tokens_total": stat_sum(metrics, "total_token_count"),
    }


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("question") or row.get("text")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(BASE_MODEL),
                                             trust_remote_code=True)
    except Exception:
        return None


def estimate_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text.split()))
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text.split()))


def post_stream(
    *,
    url: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[str, int | None, float | None, float, str | None]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_token_time: float | None = None
    text_parts: list[str] = []
    completion_tokens: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    item = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = item.get("usage")
                if isinstance(usage, dict):
                    usage_tokens = usage.get("completion_tokens")
                    if usage_tokens is not None:
                        completion_tokens = int(usage_tokens)
                choices = item.get("choices") or []
                if choices:
                    text = choices[0].get("text")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        text_parts.append(str(text))
        ended = time.perf_counter()
        return "".join(text_parts), completion_tokens, first_token_time, ended, None
    except urllib.error.HTTPError as exc:
        ended = time.perf_counter()
        return "", None, first_token_time, ended, exc.read().decode(
            "utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        ended = time.perf_counter()
        return "", None, first_token_time, ended, repr(exc)


def scrape_spec_metrics(port: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                    timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return metrics
    for line in text.splitlines():
        match = PROM_COUNTER_RE.match(line.strip())
        if not match:
            continue
        metrics[match.group(1)] = metrics.get(match.group(1), 0.0) + float(
            match.group(2))
    return metrics


def metric_delta(before: dict[str, float], after: dict[str, float],
                 name: str) -> float | None:
    if name not in after:
        return None
    return after[name] - before.get(name, 0.0)


def run_streaming_case(
    args: argparse.Namespace,
    *,
    target: str,
    method: str,
    dataset_name: str,
    batch_size: int,
    run_dir: Path,
) -> dict[str, Any]:
    prompts = load_prompts(DATASETS[dataset_name])
    tokenizer = load_tokenizer()
    start = time.perf_counter()
    warmup_end = start + args.warmup_s
    measurement_end = warmup_end + args.measurement_s
    cooldown_end = measurement_end + args.cooldown_s
    prompt_index = 0
    prompt_lock = threading.Lock()
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    stop_event = threading.Event()
    url = f"{target}/v1/completions"

    def next_prompt() -> tuple[int, str]:
        nonlocal prompt_index
        with prompt_lock:
            index = prompt_index
            prompt_index += 1
        return index, prompts[index % len(prompts)]

    def worker(worker_id: int) -> None:
        while not stop_event.is_set() and time.perf_counter() < cooldown_end:
            dataset_index, prompt = next_prompt()
            request_id = (
                f"llama31-{method}-{dataset_name}-bs{batch_size}-"
                f"w{worker_id}-p{dataset_index:08d}"
            )
            body = {
                "model": str(BASE_MODEL),
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
                "request_id": request_id,
            }
            sent = time.perf_counter()
            text, usage_tokens, first_token, ended, error = post_stream(
                url=url,
                body=body,
                timeout=args.request_timeout_s,
            )
            if error is not None:
                stop_event.set()
            output_tokens = usage_tokens
            if output_tokens is None:
                output_tokens = estimate_tokens(tokenizer, text)
            ttft = None if first_token is None else first_token - sent
            latency = ended - sent
            tpot = None
            if first_token is not None and output_tokens > 1:
                tpot = (ended - first_token) / (output_tokens - 1)
            with records_lock:
                records.append({
                    "ok": error is None,
                    "error": error,
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "dataset_index": dataset_index,
                    "sent_s": sent - start,
                    "first_token_s": (None if first_token is None else
                                      first_token - start),
                    "ended_s": ended - start,
                    "latency_s": latency,
                    "ttft_s": ttft,
                    "tpot_s": tpot,
                    "output_tokens": output_tokens,
                })

    metrics_before = scrape_spec_metrics(int(target.rsplit(":", 1)[1]))
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [
            pool.submit(worker, worker_id) for worker_id in range(batch_size)
        ]
        time.sleep(max(args.warmup_s + args.measurement_s +
                       args.cooldown_s, 0.0))
        stop_event.set()
        for future in futures:
            future.result()
    metrics_after = scrape_spec_metrics(int(target.rsplit(":", 1)[1]))

    records = sorted(records, key=lambda item: item["sent_s"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "stream_records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    ok_records = [record for record in records if record["ok"]]
    errored_records = [record for record in records if not record["ok"]]
    first_sent = min((float(record["sent_s"]) for record in ok_records),
                     default=0.0)
    last_ended = max((float(record["ended_s"]) for record in ok_records),
                     default=0.0)
    total_elapsed_s = max(last_ended - first_sent, 0.0)
    output_tokens_total = sum(int(record["output_tokens"])
                              for record in ok_records)

    measurement_start_s = warmup_end - start
    measurement_end_s = measurement_end - start
    measured_completed = [
        record for record in ok_records
        if measurement_start_s <= float(record["ended_s"]) < measurement_end_s
    ]
    steady_output_tokens_completed = sum(
        int(record["output_tokens"]) for record in measured_completed)
    steady_output_tokens_prorated = 0.0
    for record in ok_records:
        generation_start = record["first_token_s"]
        if generation_start is None:
            generation_start = record["sent_s"]
        generation_end = float(record["ended_s"])
        if generation_end <= generation_start:
            continue
        overlap = max(
            0.0,
            min(generation_end, measurement_end_s) -
            max(float(generation_start), measurement_start_s),
        )
        if overlap <= 0:
            continue
        steady_output_tokens_prorated += (
            int(record["output_tokens"]) *
            (overlap / (generation_end - float(generation_start))))

    ttfts = [float(record["ttft_s"]) for record in ok_records
             if record["ttft_s"] is not None]
    tpots = [float(record["tpot_s"]) for record in ok_records
             if record["tpot_s"] is not None]
    accepted = metric_delta(metrics_before, metrics_after,
                            "vllm:spec_decode_num_accepted_tokens")
    drafted = metric_delta(metrics_before, metrics_after,
                           "vllm:spec_decode_num_draft_tokens")
    acceptance_rate = (
        accepted / drafted if accepted is not None and drafted and drafted > 0
        else None)
    return {
        "status": "ok" if not errored_records else "error",
        "successful_requests": len(ok_records),
        "errored_requests": len(errored_records),
        "duration": total_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "total_output_tokens_per_second":
        (output_tokens_total / total_elapsed_s if total_elapsed_s > 0 else None),
        "steady_state_output_tokens_per_second":
        (steady_output_tokens_prorated / args.measurement_s
         if args.measurement_s > 0 else None),
        "steady_state_output_tokens_prorated": steady_output_tokens_prorated,
        "steady_state_output_tokens_completed": steady_output_tokens_completed,
        "steady_state_completed_requests": len(measured_completed),
        "output_tokens_total": output_tokens_total,
        "request_latency_mean": mean(
            [float(record["latency_s"]) for record in ok_records]),
        "ttft_ms_mean": (mean(ttfts) * 1000.0 if ttfts else None),
        "tpot_ms_mean": (mean(tpots) * 1000.0 if tpots else None),
        "ttft_ms_p50": (percentile(ttfts, 0.5) * 1000.0 if ttfts else None),
        "ttft_ms_p90": (percentile(ttfts, 0.9) * 1000.0 if ttfts else None),
        "tpot_ms_p50": (percentile(tpots, 0.5) * 1000.0 if tpots else None),
        "tpot_ms_p90": (percentile(tpots, 0.9) * 1000.0 if tpots else None),
        "spec_acceptance_rate": acceptance_rate,
        "spec_accepted_tokens": accepted,
        "spec_draft_tokens": drafted,
        "first_error": errored_records[0]["error"] if errored_records else "",
    }


def dataset_count(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def materialize_benchmark_dataset(
    name: str,
    source_path: Path,
    min_rows: int,
) -> tuple[Path, int, int]:
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    source_rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_count = len(source_rows)
    if source_count >= min_rows:
        return source_path, source_count, source_count

    output_path = source_path.with_name(
        f"{source_path.stem}_guidellm_min{min_rows}.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(min_rows):
            item = dict(source_rows[idx % source_count])
            item["benchmark_repeat_index"] = idx // source_count
            item["benchmark_source_row"] = idx % source_count
            item["benchmark_source_dataset"] = name
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path, source_count, min_rows


def prepare_datasets(args: argparse.Namespace) -> tuple[dict[str, int],
                                                        dict[str, int],
                                                        dict[str, str]]:
    source_counts: dict[str, int] = {}
    benchmark_counts: dict[str, int] = {}
    dataset_paths: dict[str, str] = {}
    missing = []
    for name in args.datasets:
        path = BASE_DATASETS[name]
        if not path.exists():
            missing.append(str(path))
            continue
        source_count = dataset_count(path)
        DATASETS[name] = path
        dataset_paths[name] = str(path)
        source_counts[name] = source_count
        benchmark_counts[name] = source_count
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): "
            + ", ".join(missing)
            + ". Run the dataset preparation step first.")
    args.dataset_paths = dataset_paths
    args.source_dataset_counts = source_counts
    return benchmark_counts, source_counts, dataset_paths


def build_vllm_command(args: argparse.Namespace, method: str, port: int,
                       batch_size: int) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "spec",
        "vllm",
        "serve",
        str(BASE_MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--seed",
        "42",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(batch_size),
        "--generation-config",
        "vllm",
    ]
    smurfs_max_k = smurfs_dynamic_max_k_for_batch(args, batch_size)
    if method == "vllm_fastdraft":
        command.extend([
            "--speculative-config",
            json.dumps({
                "model": str(FASTDRAFT_MODEL),
                "num_speculative_tokens": args.fastdraft_k,
                "method": "draft_model",
                "draft_tensor_parallel_size": 1,
                "max_model_len": args.max_model_len,
            }),
        ])
    elif method == "smurfs_fastdraft":
        command.extend([
            "--speculative-config",
            json.dumps({
                "model": str(FASTDRAFT_MODEL),
                "num_speculative_tokens": smurfs_max_k,
                "method": "draft_model",
                "draft_tensor_parallel_size": 1,
                "max_model_len": args.max_model_len,
            }),
        ])
    elif method == "vllm_eagle3":
        command.extend([
            "--speculative-config",
            json.dumps({
                "model": str(EAGLE3_SPECULATOR_MODEL),
                "num_speculative_tokens": args.eagle3_k,
                "method": "eagle3",
                "max_model_len": args.max_model_len,
            }),
        ])
    return command


def smurfs_dynamic_max_k_for_batch(args: argparse.Namespace,
                                   batch_size: int) -> int:
    override = int(args.smurfs_dynamic_max_k or 0)
    if override > 0:
        return override
    if batch_size >= args.smurfs_dynamic_high_batch_threshold:
        return int(args.smurfs_dynamic_high_batch_max_k)
    return int(args.smurfs_dynamic_low_batch_max_k)


def start_server(args: argparse.Namespace, method: str, batch_size: int,
                 run_dir: Path, env: dict[str, str]) -> tuple[
                     subprocess.Popen[Any], int
                 ]:
    port = find_free_port(args.port_base)
    command = build_vllm_command(args, method, port, batch_size)
    command_meta: dict[str, Any] = {"server": command}
    cwd = EVAL_ROOT
    server_env = dict(env)
    for key in list(server_env):
        if key.startswith(SMURFS_DYNAMIC_ENV_PREFIX):
            server_env.pop(key, None)
    if method == "smurfs_fastdraft":
        smurfs_max_k = smurfs_dynamic_max_k_for_batch(args, batch_size)
        server_env.update({
            "SPECLINK_SMURFS_DYNAMIC_ENABLE":
            "1",
            "SPECLINK_SMURFS_DYNAMIC_METHODS":
            "draft_model",
            "SPECLINK_SMURFS_DYNAMIC_INITIAL_K":
            str(args.smurfs_initial_k),
            "SPECLINK_SMURFS_DYNAMIC_MIN_K":
            str(args.scheduler_min_step),
            "SPECLINK_SMURFS_DYNAMIC_MAX_K":
            str(smurfs_max_k),
            "SPECLINK_SMURFS_DYNAMIC_UPDATE_DRAFT_TOKENS":
            str(args.smurfs_dynamic_update_draft_tokens),
            "SPECLINK_SMURFS_DYNAMIC_UP_ACCEPTANCE":
            str(args.smurfs_dynamic_up_acceptance),
            "SPECLINK_SMURFS_DYNAMIC_DOWN_ACCEPTANCE":
            str(args.smurfs_dynamic_down_acceptance),
            "SPECLINK_SMURFS_DYNAMIC_UP_FULL_PREFIX":
            str(args.smurfs_dynamic_up_full_prefix),
            "SPECLINK_SMURFS_DYNAMIC_DOWN_AVG_ACCEPT":
            str(args.smurfs_dynamic_down_avg_accept),
            "SPECLINK_SMURFS_DYNAMIC_MIN_FEEDBACK_BEFORE_DOWN":
            str(args.smurfs_dynamic_min_feedback_before_down),
            "SPECLINK_SMURFS_DYNAMIC_OUT":
            str(run_dir / "smurfs_dynamic_k.jsonl"),
        })
        command_meta["smurfs_dynamic_k_log"] = str(
            run_dir / "smurfs_dynamic_k.jsonl")
        command_meta["smurfs_dynamic_max_k_for_batch"] = smurfs_max_k
    else:
        server_env["SPECLINK_SMURFS_DYNAMIC_ENABLE"] = "0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(command_meta, indent=2) + "\n", encoding="utf-8")
    process = start_logged(command,
                           cwd=cwd,
                           log_path=run_dir / "server.log",
                           env=server_env)
    try:
        wait_for_health(port, process, args.health_timeout_s)
    except Exception:
        stop_process(process)
        raise
    return process, port


def request_count_for_batch(args: argparse.Namespace, batch_size: int) -> int:
    return max(args.min_requests_per_run, batch_size)


def run_guidellm_case(args: argparse.Namespace, *, target: str,
                      dataset_name: str, batch_size: int, run_dir: Path,
                      env: dict[str, str]) -> dict[str, Any]:
    output_path = run_dir / "guidellm_results.json"
    body = {
        "extras": {
            "body": {
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
            }
        }
    }
    max_requests = request_count_for_batch(args, batch_size)
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "spec",
        "guidellm",
        "benchmark",
        "run",
        "--target",
        target,
        "--data",
        str(DATASETS[dataset_name]),
        "--profile",
        "throughput",
        "--rate",
        str(batch_size),
        "--backend",
        "openai_http",
        "--request-type",
        "/v1/completions",
        "--model",
        str(BASE_MODEL),
        "--processor",
        str(BASE_MODEL),
        "--max-requests",
        str(max_requests),
        "--output-path",
        str(output_path),
        "--disable-console-interactive",
        "--backend-args",
        json.dumps(body),
    ]
    run_logged(command,
               cwd=EVAL_ROOT,
               log_path=run_dir / "guidellm_output.log",
               env=env)
    metrics = parse_guidellm(output_path)
    if int(metrics.get("errored_requests") or 0) != 0:
        raise RuntimeError(f"{dataset_name} bs{batch_size} had errors: {metrics}")
    if int(metrics.get("successful_requests") or 0) != max_requests:
        raise RuntimeError(
            f"{dataset_name} bs{batch_size} completed "
            f"{metrics.get('successful_requests')} requests, expected "
            f"{max_requests}: {metrics}")
    return metrics


def summarize_draft_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    total_slots = sum(int(item["scheduled_requests"]) for item in history)
    total_step_slots = sum(
        int(item["scheduler_step"]) * int(item["scheduled_requests"])
        for item in history)
    total_drafted_tokens = sum(int(item["drafted_tokens"]) for item in history)
    draft_dist: Counter[int] = Counter()
    step_round_dist: Counter[int] = Counter()
    step_request_dist: Counter[int] = Counter()
    changes = []
    previous_step = None
    for item in history:
        step = int(item["scheduler_step"])
        scheduled_requests = int(item["scheduled_requests"])
        step_round_dist[step] += 1
        step_request_dist[step] += scheduled_requests
        if previous_step is None or step != previous_step:
            changes.append({
                "global_round": int(item.get("global_round", item["round"])),
                "scheduler_step": step,
            })
            previous_step = step
        for length, count in item["draft_length_counts"].items():
            draft_dist[int(length)] += int(count)
    scheduler_steps = [int(item["scheduler_step"]) for item in history]
    average_scheduler_step = (
        total_step_slots / total_slots if total_slots else None)
    average_actual_draft_length = (
        total_drafted_tokens / total_slots if total_slots else None)
    return {
        "average_scheduler_step": average_scheduler_step,
        "average_scheduler_k": average_scheduler_step,
        "average_actual_draft_length": average_actual_draft_length,
        "average_actual_draft_len": average_actual_draft_length,
        "min_scheduler_step": min(scheduler_steps) if scheduler_steps else None,
        "max_scheduler_step": max(scheduler_steps) if scheduler_steps else None,
        "k_change_count": max(0, len(changes) - 1),
        "draft_length_distribution": dict(sorted(draft_dist.items())),
        "scheduler_step_round_distribution": dict(sorted(step_round_dist.items())),
        "scheduler_step_request_distribution":
        dict(sorted(step_request_dist.items())),
        "scheduler_step_changes": changes,
    }


def read_jsonl_from_offset(path: Path, offset: int) -> tuple[list[dict[str, Any]],
                                                            int]:
    if not path.exists():
        return [], offset
    with path.open("rb") as f:
        f.seek(offset)
        payload = f.read()
        new_offset = f.tell()
    events = []
    for line in payload.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, new_offset


def smurfs_dynamic_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    feedback_events = []
    for event in events:
        if event.get("event") == "proposal":
            k = int(event.get("effective_k") or 0)
            active_requests = int(event.get("active_requests") or 0)
            if k <= 0 or active_requests <= 0:
                continue
            history.append({
                "global_round":
                int(event.get("proposal_index") or len(history) + 1),
                "round":
                int(event.get("proposal_index") or len(history) + 1),
                "scheduler_step":
                k,
                "scheduled_requests":
                active_requests,
                "drafted_tokens":
                int(event.get("draft_tokens") or k * active_requests),
                "draft_length_counts": {
                    str(k): active_requests
                },
            })
        elif event.get("event") == "feedback":
            feedback_events.append(event)
    summary = summarize_draft_history(history)
    summary["draft_length_history_count"] = len(history)
    summary["draft_length_history"] = history
    summary["feedback_event_count"] = len(feedback_events)
    if feedback_events:
        total_drafted = sum(
            int(event.get("window_draft_tokens") or 0)
            for event in feedback_events)
        total_accepted = sum(
            int(event.get("window_accepted_tokens") or 0)
            for event in feedback_events)
        total_requests = sum(
            int(event.get("window_requests") or 0) for event in feedback_events)
        summary["dynamic_feedback_acceptance_rate"] = (
            total_accepted / total_drafted if total_drafted else None)
        summary["dynamic_feedback_avg_accepted_tokens"] = (
            total_accepted / total_requests if total_requests else None)
        summary["dynamic_feedback_events"] = feedback_events
    return summary


def row_for_case(method: str, dataset_name: str, batch_size: int,
                 metrics: dict[str, Any], run_dir: Path,
                 args: argparse.Namespace,
                 smurfs_summary: dict[str, Any] | None) -> dict[str, Any]:
    if method == "vllm_ar":
        k_value: Any = 0
    elif method == "vllm_fastdraft":
        k_value = args.fastdraft_k
    elif method == "vllm_eagle3":
        k_value = args.eagle3_k
    else:
        k_value = (
            f"dynamic:init{args.smurfs_initial_k}:"
            f"max{smurfs_dynamic_max_k_for_batch(args, batch_size)}"
        )
    smurfs_summary = smurfs_summary or {}
    draft_distribution = ""
    step_round_distribution = ""
    step_changes = ""
    smurfs_history_file = ""
    if smurfs_summary:
        history = list(smurfs_summary.get("draft_length_history") or [])
        if history:
            history_path = run_dir / "smurfs_draft_history_delta.jsonl"
            with history_path.open("w", encoding="utf-8") as f:
                for item in history:
                    f.write(json.dumps(item, sort_keys=True) + "\n")
            smurfs_history_file = str(history_path)
        draft_distribution = json.dumps(
            smurfs_summary.get("draft_length_distribution", {}),
            sort_keys=True)
        step_round_distribution = json.dumps(
            smurfs_summary.get("scheduler_step_round_distribution", {}),
            sort_keys=True)
        step_changes = json.dumps(
            smurfs_summary.get("scheduler_step_changes", []), sort_keys=True)
    return {
        "method": method,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "K": k_value,
        "initial_k": (
            args.fastdraft_k if method == "vllm_fastdraft" else
            args.eagle3_k if method == "vllm_eagle3" else
            args.smurfs_initial_k if method == "smurfs_fastdraft" else 0),
        "max_new_tokens": args.max_tokens,
        "max_requests": "",
        "warmup_s": args.warmup_s,
        "measurement_s": args.measurement_s,
        "cooldown_s": args.cooldown_s,
        **metrics,
        "average_actual_draft_length":
        smurfs_summary.get("average_actual_draft_length", ""),
        "average_actual_draft_len":
        smurfs_summary.get("average_actual_draft_len", ""),
        "average_scheduler_step":
        smurfs_summary.get("average_scheduler_step", ""),
        "average_scheduler_k": smurfs_summary.get("average_scheduler_k", ""),
        "min_scheduler_step": smurfs_summary.get("min_scheduler_step", ""),
        "max_scheduler_step": smurfs_summary.get("max_scheduler_step", ""),
        "k_change_count": smurfs_summary.get("k_change_count", ""),
        "dynamic_feedback_acceptance_rate":
        smurfs_summary.get("dynamic_feedback_acceptance_rate", ""),
        "dynamic_feedback_avg_accepted_tokens":
        smurfs_summary.get("dynamic_feedback_avg_accepted_tokens", ""),
        "draft_length_distribution": draft_distribution,
        "scheduler_step_round_distribution": step_round_distribution,
        "scheduler_step_changes": step_changes,
        "smurfs_history_file": smurfs_history_file,
        "work_dir": str(run_dir),
    }


def failure_row(method: str, dataset_name: str, batch_size: int,
                run_dir: Path, args: argparse.Namespace,
                reason: str) -> dict[str, Any]:
    metrics = {
        "status": "failed",
        "successful_requests": 0,
        "errored_requests": 1,
        "duration": None,
        "total_elapsed_s": None,
        "total_output_tokens_per_second": None,
        "steady_state_output_tokens_per_second": None,
        "steady_state_output_tokens_prorated": None,
        "steady_state_output_tokens_completed": None,
        "steady_state_completed_requests": None,
        "output_tokens_total": 0,
        "request_latency_mean": None,
        "ttft_ms_mean": None,
        "tpot_ms_mean": None,
        "ttft_ms_p50": None,
        "ttft_ms_p90": None,
        "tpot_ms_p50": None,
        "tpot_ms_p90": None,
        "spec_acceptance_rate": None,
        "spec_accepted_tokens": None,
        "spec_draft_tokens": None,
        "first_error": reason,
    }
    return row_for_case(method, dataset_name, batch_size, metrics, run_dir,
                        args, None)


def build_resume_command(args: argparse.Namespace) -> str:
    parts = [
        "conda",
        "run",
        "-n",
        "spec",
        "python",
        "examples/evaluate/eval-guidellm/scripts/"
        "run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py",
        "--final-root",
        str(args.final_root),
        "--work-root",
        str(args.work_root),
        "--methods",
        ",".join(args.methods),
        "--datasets",
        ",".join(args.datasets),
        "--batch-sizes",
        ",".join(str(item) for item in args.batch_sizes),
        "--min-requests-per-run",
        str(args.min_requests_per_run),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--fastdraft-k",
        str(args.fastdraft_k),
        "--eagle3-k",
        str(args.eagle3_k),
        "--smurfs-initial-k",
        str(args.smurfs_initial_k),
        "--smurfs-dynamic-max-k",
        str(args.smurfs_dynamic_max_k),
        "--smurfs-dynamic-low-batch-max-k",
        str(args.smurfs_dynamic_low_batch_max_k),
        "--smurfs-dynamic-high-batch-threshold",
        str(args.smurfs_dynamic_high_batch_threshold),
        "--smurfs-dynamic-high-batch-max-k",
        str(args.smurfs_dynamic_high_batch_max_k),
        "--smurfs-dynamic-update-draft-tokens",
        str(args.smurfs_dynamic_update_draft_tokens),
        "--smurfs-dynamic-up-acceptance",
        str(args.smurfs_dynamic_up_acceptance),
        "--smurfs-dynamic-down-acceptance",
        str(args.smurfs_dynamic_down_acceptance),
        "--smurfs-dynamic-up-full-prefix",
        str(args.smurfs_dynamic_up_full_prefix),
        "--smurfs-dynamic-down-avg-accept",
        str(args.smurfs_dynamic_down_avg_accept),
        "--smurfs-dynamic-min-feedback-before-down",
        str(args.smurfs_dynamic_min_feedback_before_down),
        "--scheduler-min-step",
        str(args.scheduler_min_step),
        "--port-base",
        str(args.port_base),
        "--health-timeout-s",
        str(args.health_timeout_s),
        "--warmup-s",
        str(args.warmup_s),
        "--measurement-s",
        str(args.measurement_s),
        "--cooldown-s",
        str(args.cooldown_s),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--resume",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def write_outputs(final_root: Path, rows: list[dict[str, Any]],
                  args: argparse.Namespace,
                  dataset_counts: dict[str, int]) -> None:
    final_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "dataset",
        "batch_size",
        "status",
        "K",
        "initial_k",
        "max_new_tokens",
        "warmup_s",
        "measurement_s",
        "cooldown_s",
        "max_requests",
        "successful_requests",
        "errored_requests",
        "duration",
        "total_elapsed_s",
        "total_output_tokens_per_second",
        "steady_state_output_tokens_per_second",
        "steady_state_output_tokens_prorated",
        "steady_state_output_tokens_completed",
        "steady_state_completed_requests",
        "output_tokens_total",
        "request_latency_mean",
        "ttft_ms_mean",
        "tpot_ms_mean",
        "ttft_ms_p50",
        "ttft_ms_p90",
        "tpot_ms_p50",
        "tpot_ms_p90",
        "spec_acceptance_rate",
        "spec_accepted_tokens",
        "spec_draft_tokens",
        "average_scheduler_k",
        "average_actual_draft_len",
        "average_actual_draft_length",
        "average_scheduler_step",
        "min_scheduler_step",
        "max_scheduler_step",
        "k_change_count",
        "dynamic_feedback_acceptance_rate",
        "dynamic_feedback_avg_accepted_tokens",
        "draft_length_distribution",
        "scheduler_step_round_distribution",
        "scheduler_step_changes",
        "first_error",
        "smurfs_history_file",
        "work_dir",
    ]
    with (final_root / "summary.csv").open("w",
                                           newline="",
                                           encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column, "")) for column in columns})
    (final_root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    smurfs_columns = [
        "dataset",
        "batch_size",
        "initial_k",
        "average_scheduler_k",
        "average_actual_draft_len",
        "min_scheduler_step",
        "max_scheduler_step",
        "k_change_count",
        "dynamic_feedback_acceptance_rate",
        "dynamic_feedback_avg_accepted_tokens",
        "draft_length_distribution",
        "scheduler_step_round_distribution",
        "scheduler_step_changes",
        "smurfs_history_file",
    ]
    smurfs_rows = [
        row for row in rows if row.get("method") == "smurfs_fastdraft"
    ]
    with (final_root / "smurfs_k_summary.csv").open("w",
                                                    newline="",
                                                    encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=smurfs_columns)
        writer.writeheader()
        for row in smurfs_rows:
            writer.writerow({
                column: fmt(row.get(column, "")) for column in smurfs_columns
            })
    with (final_root / "smurfs_k_timeseries.jsonl").open("w",
                                                         encoding="utf-8") as f:
        for row in smurfs_rows:
            history_file = row.get("smurfs_history_file")
            if not history_file:
                continue
            path = Path(str(history_file))
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    item.update({
                        "dataset": row.get("dataset"),
                        "batch_size": row.get("batch_size"),
                        "method": row.get("method"),
                        "initial_k": row.get("initial_k"),
                    })
                    f.write(json.dumps(item, sort_keys=True) + "\n")

    config = vars(args).copy()
    config.update({
        "base_model": str(BASE_MODEL),
        "fastdraft_model": str(FASTDRAFT_MODEL),
        "eagle3_speculator_model": str(EAGLE3_SPECULATOR_MODEL),
        "dataset_counts": dataset_counts,
        "source_dataset_counts": getattr(args, "source_dataset_counts", {}),
        "dataset_paths": getattr(args, "dataset_paths", {}),
    })
    (final_root / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    invoked = " ".join(shlex.quote(arg) for arg in sys.argv)
    resume_command = build_resume_command(args)
    (final_root / "commands.sh").write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(SPECULATORS_ROOT))}",
            "",
            "# Original invocation as seen by the script.",
            f"# {invoked}",
            "",
            "# Resume this run.",
            resume_command,
            "",
        ]),
        encoding="utf-8",
    )

    if int(args.smurfs_dynamic_max_k or 0) > 0:
        smurfs_max_policy = f"override {args.smurfs_dynamic_max_k}"
    else:
        smurfs_max_policy = (
            f"{args.smurfs_dynamic_low_batch_max_k} below bs"
            f"{args.smurfs_dynamic_high_batch_threshold}, "
            f"{args.smurfs_dynamic_high_batch_max_k} otherwise"
        )
    lines = [
        "# Llama-3.1 Throughput Matrix",
        "",
        "- methods: vLLM autoregressive, vLLM+FastDraft K=4, "
        "vLLM+FastDraft+Smurfs dynamic K "
        f"init={args.smurfs_initial_k}/max policy={smurfs_max_policy}, "
        "vLLM+EAGLE3 K=4",
        "- batch size means client-side streaming concurrency",
        f"- batch sizes: {', '.join(map(str, args.batch_sizes))}",
        f"- datasets: {', '.join(args.datasets)}",
        f"- max output tokens: {args.max_tokens}",
        "- steady-state window: "
        f"{args.warmup_s}s warmup, {args.measurement_s}s measurement, "
        f"{args.cooldown_s}s cooldown",
        f"- work/intermediate root: {args.work_root}",
        "",
        "## Dataset Counts",
        "",
        "| dataset | rows |",
        "|---|---:|",
    ]
    for name in args.datasets:
        lines.append(f"| {name} | {dataset_counts[name]} |")
    lines.extend([
        "",
        "## Total Output Tokens/s",
        "",
        "| dataset | batch_size | vllm_ar | vllm_fastdraft | smurfs_fastdraft | vllm_eagle3 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_key = {(row["dataset"], int(row["batch_size"]), row["method"]): row
              for row in rows}
    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            values = []
            for method in METHODS:
                row = by_key.get((dataset, batch_size, method), {})
                value = row.get("total_output_tokens_per_second", "")
                values.append(f"{float(value):.3f}" if value not in {"", None} else "")
            lines.append(
                f"| {dataset} | {batch_size} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |"
            )
    lines.extend([
        "",
        "## Steady-State Output Tokens/s",
        "",
        "| dataset | batch_size | vllm_ar | vllm_fastdraft | smurfs_fastdraft | vllm_eagle3 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            values = []
            for method in METHODS:
                row = by_key.get((dataset, batch_size, method), {})
                value = row.get("steady_state_output_tokens_per_second", "")
                values.append(f"{float(value):.3f}" if value not in {"", None} else "")
            lines.append(
                f"| {dataset} | {batch_size} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |"
            )
    lines.extend([
        "",
        "## Smurfs Dynamic Draft Length",
        "",
        "| dataset | batch_size | avg K | avg actual draft length | min K | max K | K changes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            row = by_key.get((dataset, batch_size, "smurfs_fastdraft"), {})
            avg_step = row.get("average_scheduler_k", "")
            avg_draft = row.get("average_actual_draft_len", "")
            min_step = row.get("min_scheduler_step", "")
            max_step = row.get("max_scheduler_step", "")
            changes = row.get("k_change_count", "")
            lines.append(
                "| {dataset} | {batch_size} | {avg_step} | {avg_draft} | "
                "{min_step} | {max_step} | {changes} |".
                format(dataset=dataset,
                       batch_size=batch_size,
                       avg_step=(f"{float(avg_step):.3f}" if avg_step not in {
                           "", None
                       } else ""),
                       avg_draft=(f"{float(avg_draft):.3f}"
                                  if avg_draft not in {"", None} else ""),
                       min_step=fmt(min_step),
                       max_step=fmt(max_step),
                       changes=fmt(changes)))
    failures = [
        row for row in rows
        if row.get("status") not in {"", None, "ok"}
        or int(row.get("errored_requests") or 0) > 0
    ]
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(
                "- {method} {dataset} bs={batch_size}: {reason}".format(
                    method=row.get("method"),
                    dataset=row.get("dataset"),
                    batch_size=row.get("batch_size"),
                    reason=row.get("first_error") or row.get("status"),
                ))
    else:
        lines.append("- None")
    (final_root / "report.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


def append_status(work_root: Path, fields: list[Any]) -> None:
    with (work_root / "status.tsv").open("a", encoding="utf-8") as f:
        f.write("\t".join(str(field) for field in fields) + "\n")


def read_existing_rows(final_root: Path) -> list[dict[str, Any]]:
    path = final_root / "summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_csv_list(value: str, *, valid: set[str] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if valid is not None:
        unknown = [item for item in items if item not in valid]
        if unknown:
            raise ValueError(f"Unknown value(s): {unknown}. Valid: {sorted(valid)}")
    return items


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--methods",
                        default=",".join(METHODS),
                        help=f"comma list from {METHODS}")
    parser.add_argument("--datasets",
                        default="math_reasoning,mtbench,gsm8k,humaneval",
                        help=f"comma list from {tuple(DATASETS)}")
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--min-requests-per-run", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--fastdraft-k", type=int, default=4)
    parser.add_argument("--eagle3-k", type=int, default=4)
    parser.add_argument("--smurfs-initial-k", type=int, default=4)
    parser.add_argument(
        "--smurfs-dynamic-max-k",
        type=int,
        default=0,
        help=(
            "Override Smurfs dynamic max K for all batch sizes. Default 0 "
            "uses the low/high batch policy below."
        ),
    )
    parser.add_argument("--smurfs-dynamic-low-batch-max-k",
                        type=int,
                        default=12)
    parser.add_argument("--smurfs-dynamic-high-batch-threshold",
                        type=int,
                        default=32)
    parser.add_argument("--smurfs-dynamic-high-batch-max-k",
                        type=int,
                        default=8)
    parser.add_argument("--smurfs-dynamic-update-draft-tokens",
                        type=int,
                        default=256)
    parser.add_argument("--smurfs-dynamic-up-acceptance",
                        type=float,
                        default=0.58)
    parser.add_argument("--smurfs-dynamic-down-acceptance",
                        type=float,
                        default=0.38)
    parser.add_argument("--smurfs-dynamic-up-full-prefix",
                        type=float,
                        default=0.12)
    parser.add_argument("--smurfs-dynamic-down-avg-accept",
                        type=float,
                        default=1.20)
    parser.add_argument("--smurfs-dynamic-min-feedback-before-down",
                        type=int,
                        default=4)
    parser.add_argument("--scheduler-min-step", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=8078)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--warmup-s", type=float, default=10.0)
    parser.add_argument("--measurement-s", type=float, default=60.0)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    args.methods = parse_csv_list(args.methods, valid=set(METHODS))
    args.datasets = parse_csv_list(args.datasets, valid=set(DATASETS))
    args.batch_sizes = parse_int_list(args.batch_sizes)
    if args.smoke_only:
        args.methods = [
            "vllm_ar",
            "vllm_fastdraft",
            "smurfs_fastdraft",
            "vllm_eagle3",
        ]
        args.datasets = ["math_reasoning"]
        args.batch_sizes = [1]
        args.min_requests_per_run = 0
        args.max_tokens = 64
        args.max_model_len = 512
        args.max_num_batched_tokens = 1024
        args.gpu_memory_utilization = 0.72
        args.warmup_s = 2.0
        args.measurement_s = 5.0
        args.cooldown_s = 1.0
        args.request_timeout_s = 600.0

    run_label = "smoke" if args.smoke_only else "full"
    stamp = timestamp()
    if args.final_root is None:
        result_parent = EVAL_ROOT / ("temp" if args.smoke_only else "results_final")
        args.final_root = (
            result_parent /
            f"llama31_vllm_fastdraft_smurfs_eagle3_matrix_2048_{run_label}_{stamp}")
    if args.work_root is None:
        args.work_root = (
            EVAL_ROOT / "temp" /
            f"llama31_vllm_fastdraft_smurfs_eagle3_matrix_2048_{run_label}_{stamp}")
    args.final_root = args.final_root.resolve()
    args.work_root = args.work_root.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.final_root.mkdir(parents=True, exist_ok=True)
    dataset_counts, _, _ = prepare_datasets(args)
    env = add_local_no_proxy(os.environ)
    os.environ["NO_PROXY"] = env["NO_PROXY"]
    os.environ["no_proxy"] = env["NO_PROXY"]

    status_path = args.work_root / "status.tsv"
    if not status_path.exists():
        append_status(args.work_root, [
            "time", "status", "method", "batch_size", "dataset", "message"
        ])
    rows = read_existing_rows(args.final_root) if args.resume else []
    completed = {
        (row["method"], row["dataset"], int(row["batch_size"]))
        for row in rows
        if row.get("method") and row.get("dataset") and row.get("batch_size")
    }
    if rows:
        write_outputs(args.final_root, rows, args, dataset_counts)
    try:
        for method in args.methods:
            for batch_size in args.batch_sizes:
                pending_datasets = [
                    dataset for dataset in args.datasets
                    if (method, dataset, batch_size) not in completed
                ]
                if not pending_datasets:
                    append_status(args.work_root, [
                        timestamp(), "server_skip_completed", method,
                        batch_size, "", ""
                    ])
                    continue
                server_dir = args.work_root / method / f"bs{batch_size}"
                append_status(args.work_root, [
                    timestamp(), "server_start", method, batch_size, "", ""
                ])
                process: subprocess.Popen[Any] | None = None
                try:
                    process, port = start_server(args, method, batch_size,
                                                 server_dir, env)
                except Exception as exc:  # noqa: BLE001
                    append_status(args.work_root, [
                        timestamp(), "server_failed", method, batch_size, "",
                        repr(exc)
                    ])
                    for dataset_name in pending_datasets:
                        case_dir = server_dir / dataset_name
                        case_dir.mkdir(parents=True, exist_ok=True)
                        rows.append(
                            failure_row(method, dataset_name, batch_size,
                                        case_dir, args, repr(exc)))
                        completed.add((method, dataset_name, batch_size))
                    write_outputs(args.final_root, rows, args, dataset_counts)
                    continue
                target = f"http://127.0.0.1:{port}"
                try:
                    for dataset_name in pending_datasets:
                        case_dir = server_dir / dataset_name
                        case_dir.mkdir(parents=True, exist_ok=True)
                        smurfs_dynamic_log = (
                            server_dir / "smurfs_dynamic_k.jsonl"
                            if method == "smurfs_fastdraft" else None)
                        smurfs_dynamic_offset = (
                            smurfs_dynamic_log.stat().st_size
                            if smurfs_dynamic_log is not None
                            and smurfs_dynamic_log.exists() else 0)
                        append_status(args.work_root, [
                            timestamp(), "run_start", method, batch_size,
                            dataset_name, ""
                        ])
                        smurfs_summary = None
                        try:
                            metrics = run_streaming_case(
                                args,
                                target=target,
                                method=method,
                                dataset_name=dataset_name,
                                batch_size=batch_size,
                                run_dir=case_dir,
                            )
                            if method == "smurfs_fastdraft":
                                assert smurfs_dynamic_log is not None
                                events, smurfs_dynamic_offset = (
                                    read_jsonl_from_offset(
                                        smurfs_dynamic_log,
                                        smurfs_dynamic_offset,
                                    ))
                                (case_dir / "smurfs_dynamic_k_delta.jsonl"
                                 ).write_text(
                                     "".join(
                                         json.dumps(event, sort_keys=True) + "\n"
                                         for event in events),
                                     encoding="utf-8",
                                 )
                                smurfs_summary = smurfs_dynamic_summary(events)
                            row = row_for_case(method, dataset_name,
                                               batch_size, metrics, case_dir,
                                               args, smurfs_summary)
                            run_status = "run_ok"
                            if metrics.get("status") != "ok":
                                run_status = "run_error"
                            message = metrics.get(
                                "steady_state_output_tokens_per_second")
                        except Exception as exc:  # noqa: BLE001
                            row = failure_row(method, dataset_name, batch_size,
                                              case_dir, args, repr(exc))
                            run_status = "run_failed"
                            message = repr(exc)
                        rows.append(row)
                        completed.add((method, dataset_name, batch_size))
                        write_outputs(args.final_root, rows, args,
                                      dataset_counts)
                        append_status(args.work_root, [
                            timestamp(), run_status, method, batch_size,
                            dataset_name, message
                        ])
                finally:
                    stop_process(process)
                    append_status(args.work_root, [
                        timestamp(), "server_stop", method, batch_size, "", ""
                    ])
    finally:
        write_outputs(args.final_root, rows, args, dataset_counts)
    print(f"final_root={args.final_root}")
    print(f"work_root={args.work_root}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
