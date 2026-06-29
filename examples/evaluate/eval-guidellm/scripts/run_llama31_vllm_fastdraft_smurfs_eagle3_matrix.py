#!/usr/bin/env python3
"""Run Llama-3.1 AR/FastDraft/Smurfs/EAGLE3 throughput matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
SR24_DEFAULT_MASK = (
    EVAL_ROOT / "data/c4_calibration/sr24_masks/"
    "llama3_1_8b_activation_aware_24.pt"
)
SMURFS_DYNAMIC_ENV_PREFIX = "SPECLINK_SMURFS_DYNAMIC_"
SR24_ENV_PREFIX = "SPECLINK_SR24_"
CUDAGRAPH_STATS_ENV_PREFIX = "SPECLINK_CUDAGRAPH_STATS_"

BASE_DATASETS = {
    "math_reasoning": EVAL_ROOT / "data/math_reasoning.jsonl",
    "mtbench": EVAL_ROOT / "data/mt_bench.jsonl",
    "gsm8k": EVAL_ROOT / "data/gsm8k.jsonl",
    "humaneval": EVAL_ROOT / "data/humaneval.jsonl",
}
DATASETS = BASE_DATASETS.copy()
DEFAULT_METHODS = ("vllm_ar", "vllm_fastdraft", "smurfs_fastdraft", "vllm_eagle3")
SR24_METHODS = ("base_only_24", "all_corrected_24", "speclink_t08")
ALL_METHODS = DEFAULT_METHODS + ("dense_baseline",) + SR24_METHODS
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


def load_prompts(path: Path, limit: int = 0) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("question") or row.get("text")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
                if limit > 0 and len(prompts) >= limit:
                    break
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


def read_gpu_sample() -> dict[str, float] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    memory_values: list[float] = []
    util_values: list[float] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            memory_values.append(float(parts[0]))
            util_values.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    if not memory_values and not util_values:
        return None
    return {
        "memory_mib": max(memory_values) if memory_values else 0.0,
        "gpu_util_pct": max(util_values) if util_values else 0.0,
    }


def read_peak_gpu_memory_mib() -> float | None:
    sample = read_gpu_sample()
    return None if sample is None else sample["memory_mib"]


def start_gpu_memory_sampler(stop_event: threading.Event,
                             interval_s: float = 0.5) -> tuple[dict[str, Any],
                                                               threading.Thread]:
    state: dict[str, Any] = {
        "peak_gpu_memory_mib": None,
        "peak_gpu_util_pct": None,
        "avg_gpu_util_pct": None,
        "gpu_util_sample_count": 0,
        "_gpu_util_sum": 0.0,
    }

    def sample() -> None:
        while not stop_event.is_set():
            value = read_gpu_sample()
            if value is not None:
                memory_mib = float(value["memory_mib"])
                current = state.get("peak_gpu_memory_mib")
                state["peak_gpu_memory_mib"] = (
                    memory_mib
                    if current is None else max(float(current), memory_mib))
                util = float(value["gpu_util_pct"])
                current_util = state.get("peak_gpu_util_pct")
                state["peak_gpu_util_pct"] = (
                    util if current_util is None else max(float(current_util), util)
                )
                state["_gpu_util_sum"] = float(state["_gpu_util_sum"]) + util
                state["gpu_util_sample_count"] = (
                    int(state["gpu_util_sample_count"]) + 1
                )
                state["avg_gpu_util_pct"] = (
                    float(state["_gpu_util_sum"])
                    / int(state["gpu_util_sample_count"])
                )
            stop_event.wait(interval_s)

    thread = threading.Thread(target=sample, name="gpu-memory-sampler", daemon=True)
    thread.start()
    return state, thread


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = sorted((float(start), float(end)) for start, end in intervals
                        if float(end) > float(start))
    if not normalized:
        return []
    merged = [normalized[0]]
    for start, end in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_overlap(start: float, end: float,
                     intervals: list[tuple[float, float]]) -> float:
    if end <= start or not intervals:
        return 0.0
    total = 0.0
    for interval_start, interval_end in intervals:
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        total += max(0.0, min(end, interval_end) - max(start, interval_start))
    return total


def point_in_intervals(value: float, intervals: list[tuple[float, float]]) -> bool:
    for interval_start, interval_end in intervals:
        if value < interval_start:
            return False
        if interval_start <= value <= interval_end:
            return True
    return False


def intersect_intervals(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    left = merge_intervals(left)
    right = merge_intervals(right)
    intersections: list[tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            intersections.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return intersections


def full_batch_active_intervals(
    records: list[dict[str, Any]],
    threshold: int,
) -> list[tuple[float, float]]:
    events: list[tuple[float, int]] = []
    for record in records:
        if not record.get("ok"):
            continue
        sent = record.get("sent_s")
        ended = record.get("ended_s")
        if sent is None or ended is None:
            continue
        if float(ended) <= float(sent):
            continue
        events.append((float(sent), 1))
        events.append((float(ended), -1))
    if not events or threshold <= 0:
        return []
    events.sort(key=lambda item: (item[0], -item[1]))
    intervals: list[tuple[float, float]] = []
    active = 0
    previous_t = events[0][0]
    idx = 0
    while idx < len(events):
        current_t = events[idx][0]
        if current_t > previous_t and active >= threshold:
            intervals.append((previous_t, current_t))
        while idx < len(events) and events[idx][0] == current_t:
            active += events[idx][1]
            idx += 1
        previous_t = current_t
    return intervals


def full_batch_generation_metrics(
    records: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    active_intervals = full_batch_active_intervals(records, batch_size)
    generation_intervals: list[tuple[float, float]] = []
    for record in records:
        if not record.get("ok"):
            continue
        generation_start = record.get("first_token_s")
        if generation_start is None:
            generation_start = record.get("sent_s")
        generation_end = record.get("ended_s")
        if generation_start is None or generation_end is None:
            continue
        if float(generation_end) > float(generation_start):
            generation_intervals.append(
                (float(generation_start), float(generation_end))
            )
    full_generation_intervals = intersect_intervals(
        active_intervals,
        merge_intervals(generation_intervals),
    )
    window_s = sum(end - start for start, end in full_generation_intervals)
    prorated_tokens = 0.0
    completed_requests = 0
    for record in records:
        if not record.get("ok"):
            continue
        generation_start = record.get("first_token_s")
        if generation_start is None:
            generation_start = record.get("sent_s")
        generation_end = record.get("ended_s")
        if generation_start is None or generation_end is None:
            continue
        generation_start = float(generation_start)
        generation_end = float(generation_end)
        if generation_end <= generation_start:
            continue
        overlap = interval_overlap(
            generation_start,
            generation_end,
            full_generation_intervals,
        )
        if overlap > 0:
            prorated_tokens += (
                int(record["output_tokens"])
                * (overlap / (generation_end - generation_start))
            )
        if point_in_intervals(generation_end, full_generation_intervals):
            completed_requests += 1
    return {
        "full_batch_active_threshold": batch_size,
        "full_batch_window_s": window_s,
        "full_batch_output_tokens_prorated": prorated_tokens,
        "full_batch_completed_requests": completed_requests,
        "full_batch_output_tokens_per_second": (
            prorated_tokens / window_s if window_s > 0 else None
        ),
    }


def run_streaming_case(
    args: argparse.Namespace,
    *,
    target: str,
    method: str,
    dataset_name: str,
    batch_size: int,
    run_dir: Path,
) -> dict[str, Any]:
    prompts = load_prompts(DATASETS[dataset_name], limit=args.prompt_limit)
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
    gpu_mem_state, gpu_mem_thread = start_gpu_memory_sampler(stop_event)

    def next_prompt() -> tuple[int, str] | None:
        nonlocal prompt_index
        with prompt_lock:
            if (
                args.fixed_total_requests > 0
                and prompt_index >= args.fixed_total_requests
            ):
                return None
            index = prompt_index
            prompt_index += 1
        return index, prompts[index % len(prompts)]

    def worker(worker_id: int) -> None:
        fixed_request_mode = args.fixed_total_requests > 0
        while (
            not stop_event.is_set()
            and (fixed_request_mode or time.perf_counter() < cooldown_end)
        ):
            next_item = next_prompt()
            if next_item is None:
                return
            dataset_index, prompt = next_item
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
        if args.fixed_total_requests > 0:
            for future in futures:
                future.result()
            stop_event.set()
        else:
            time.sleep(max(args.warmup_s + args.measurement_s +
                           args.cooldown_s, 0.0))
            stop_event.set()
            for future in futures:
                future.result()
    gpu_mem_thread.join(timeout=2.0)
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

    if args.fixed_total_requests > 0:
        measured_completed = ok_records
        steady_output_tokens_completed = output_tokens_total
        steady_output_tokens_prorated = float(output_tokens_total)
        steady_denominator_s = total_elapsed_s
    else:
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
        steady_denominator_s = args.measurement_s

    ttfts = [float(record["ttft_s"]) for record in ok_records
             if record["ttft_s"] is not None]
    tpots = [float(record["tpot_s"]) for record in ok_records
             if record["tpot_s"] is not None]
    full_batch_metrics = full_batch_generation_metrics(ok_records, batch_size)
    accepted = metric_delta(metrics_before, metrics_after,
                            "vllm:spec_decode_num_accepted_tokens")
    drafted = metric_delta(metrics_before, metrics_after,
                           "vllm:spec_decode_num_draft_tokens")
    acceptance_rate = (
        accepted / drafted if accepted is not None and drafted and drafted > 0
        else None)
    fixed_spec_k = fixed_spec_k_for_method(args, method)
    spec_estimated_steps = (
        drafted / fixed_spec_k
        if drafted is not None and fixed_spec_k and fixed_spec_k > 0
        else None
    )
    spec_avg_selected_draft_tokens_per_step = (
        drafted / spec_estimated_steps
        if drafted is not None and spec_estimated_steps
        else None
    )
    spec_avg_accepted_draft_tokens_per_step = (
        accepted / spec_estimated_steps
        if accepted is not None and spec_estimated_steps
        else None
    )
    return {
        "status": "ok" if not errored_records else "error",
        "successful_requests": len(ok_records),
        "errored_requests": len(errored_records),
        "duration": total_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "max_requests": (
            args.fixed_total_requests if args.fixed_total_requests > 0 else ""
        ),
        "total_output_tokens_per_second":
        (output_tokens_total / total_elapsed_s if total_elapsed_s > 0 else None),
        "steady_state_output_tokens_per_second":
        (steady_output_tokens_prorated / steady_denominator_s
         if steady_denominator_s > 0 else None),
        "steady_state_output_tokens_prorated": steady_output_tokens_prorated,
        "steady_state_output_tokens_completed": steady_output_tokens_completed,
        "steady_state_completed_requests": len(measured_completed),
        **full_batch_metrics,
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
        "spec_acceptance_rate_pct": (
            acceptance_rate * 100.0 if acceptance_rate is not None else None
        ),
        "spec_accepted_tokens": accepted,
        "spec_draft_tokens": drafted,
        "spec_estimated_steps": spec_estimated_steps,
        "spec_avg_selected_draft_tokens_per_step":
        spec_avg_selected_draft_tokens_per_step,
        "spec_avg_accepted_draft_tokens_per_step":
        spec_avg_accepted_draft_tokens_per_step,
        "peak_gpu_memory_mib": gpu_mem_state.get("peak_gpu_memory_mib"),
        "avg_gpu_util_pct": gpu_mem_state.get("avg_gpu_util_pct"),
        "peak_gpu_util_pct": gpu_mem_state.get("peak_gpu_util_pct"),
        "gpu_util_sample_count": gpu_mem_state.get("gpu_util_sample_count"),
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


def method_uses_eagle3(method: str) -> bool:
    return method in {"vllm_eagle3", "dense_baseline", *SR24_METHODS}


def fixed_spec_k_for_method(args: argparse.Namespace, method: str) -> int | None:
    if method == "vllm_fastdraft":
        return int(args.fastdraft_k)
    if method_uses_eagle3(method):
        return int(args.eagle3_k)
    return None


def sr24_runtime_mode(method: str) -> str:
    if method == "base_only_24":
        return "base_only"
    if method == "all_corrected_24":
        return "all_corrected"
    if method == "speclink_t08":
        return "selective"
    raise ValueError(f"{method} is not an SR24 method")


def sr24_effective_residual_backend(args: argparse.Namespace, method: str) -> str:
    """Return the per-method residual backend actually used for SR24.

    Keep this as an explicit configuration, not a hidden per-method default.
    `torch_sparse` is useful for all-corrected ablations, but it can add enough
    startup memory pressure to OOM on 32GB GPUs. Experiments that need it should
    pass `--sr24-residual-backend torch_sparse` directly.
    """
    return str(args.sr24_residual_backend)


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
        "sr24_residual_layer_ids_by_leaf",
    )
    return any(str(getattr(args, field, "") or "").strip() for field in scoped_fields)


def sr24_compressed_residual_runtime_settings(
    args: argparse.Namespace,
    method: str,
) -> tuple[int, bool, bool, bool]:
    """Return effective compressed-residual settings for one SR24 method.

    `all_corrected_24` with `compressed_dense` is an operator ablation. The
    useful fast path is cache+prewarm+full materialization, which keeps the
    residual weight GPU-resident before decode and allows CUDA Graph capture.
    Leaving the generic chunked runtime materialization path enabled measures
    repeated residual-weight rebuild overhead instead of the best current
    all-corrected operator shape.
    """
    residual_out_chunk = int(args.sr24_residual_out_chunk)
    cache_weight = bool(args.sr24_cache_compressed_residual_weight)
    prewarm_weight = bool(args.sr24_prewarm_compressed_residual_weight)
    auto_fastpath = (
        bool(getattr(args, "sr24_auto_compressed_residual_fastpath", True))
        and method == "all_corrected_24"
        and args.sr24_backend == "torch_sparse"
        and sr24_effective_residual_backend(args, method) == "compressed_dense"
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
    method: str,
) -> bool:
    if bool(args.sr24_direct_cslt_linear):
        return True
    return (
        bool(getattr(args, "sr24_auto_direct_cslt_base_only", True))
        and method == "base_only_24"
        and args.sr24_backend == "torch_sparse"
        and args.sr24_gate_up_split == "none"
    )


def sr24_cudagraph_bucket_active_hint(
    args: argparse.Namespace,
    method: str,
    batch_size: int,
) -> int:
    if (
        method == "speclink_t08"
        and args.sr24_residual_bucket_scale_by_active
        and args.sr24_cudagraph_bucket
        and args.sr24_allow_cudagraph
        and not args.sr24_force_cudagraph_none_for_mixed
    ):
        return max(1, int(batch_size))
    return 0


def sr24_compile_cache_root(
    args: argparse.Namespace,
    method: str,
    batch_size: int = 0,
) -> Path:
    (
        residual_out_chunk,
        cache_compressed_residual_weight,
        prewarm_compressed_residual_weight,
        auto_compressed_residual_fastpath,
    ) = sr24_compressed_residual_runtime_settings(args, method)
    fingerprint = {
        "method": method,
        "mode": sr24_runtime_mode(method),
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
        "direct_cslt_linear": sr24_effective_direct_cslt_linear(args, method),
        "auto_direct_cslt_base_only": args.sr24_auto_direct_cslt_base_only,
        "base_only_allow_compile": args.sr24_base_only_allow_compile,
        "base_only_dense_nonverify": args.sr24_base_only_dense_nonverify,
        "base_only_dense_verify_max_rows":
        args.sr24_base_only_dense_verify_max_rows,
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
        "row_routed_mlp_reuse_base_output":
        args.sr24_row_routed_mlp_reuse_base_output,
        "row_routed_mlp_min_dense_rows": args.sr24_row_routed_mlp_min_dense_rows,
        "row_routed_mlp_max_dense_rows": args.sr24_row_routed_mlp_max_dense_rows,
        "row_routed_mlp_max_base_rows": args.sr24_row_routed_mlp_max_base_rows,
        "dense_fallback_nonuniform": args.sr24_dense_fallback_nonuniform,
        "selective_correct_non_draft": args.sr24_selective_correct_non_draft,
        "selective_non_draft_policy": args.sr24_selective_non_draft_policy,
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
        "compressed_residual_block_m": args.sr24_compressed_residual_block_m,
        "compressed_residual_block_n": args.sr24_compressed_residual_block_n,
        "compressed_residual_block_g": args.sr24_compressed_residual_block_g,
        "extract_chunk_rows": args.sr24_extract_chunk_rows,
        "residual_bucket_size": args.sr24_residual_bucket_size,
        "residual_bucket_scale_by_active":
        args.sr24_residual_bucket_scale_by_active,
        "residual_bucket_priority": args.sr24_residual_bucket_priority,
        "direct_position_bucket": args.sr24_direct_position_bucket,
        "bonus_priority": args.sr24_bonus_priority,
        "draft_position_priority_scale":
        args.sr24_draft_position_priority_scale,
        "route_bucket_rows": args.sr24_route_bucket_rows,
        "route_all_residual_rows": args.sr24_route_all_residual_rows,
        "route_all_skip_bucket": args.sr24_route_all_skip_bucket,
        "direct_cpu_route_rows": args.sr24_direct_cpu_route_rows,
        "route_reuse_base_output": args.sr24_route_reuse_base_output,
        "route_contiguous_fastpath": args.sr24_route_contiguous_fastpath,
        "route_dense_fallback_fraction":
        args.sr24_route_dense_fallback_fraction,
        "route_min_dense_rows": args.sr24_route_min_dense_rows,
        "route_min_base_rows": args.sr24_route_min_base_rows,
        "route_max_dense_fraction": args.sr24_route_max_dense_fraction,
        "adaptive_dense_fallback": args.sr24_adaptive_dense_fallback,
        "adaptive_dense_fallback_small_rows":
        args.sr24_adaptive_dense_fallback_small_rows,
        "adaptive_dense_fallback_gate_up_fraction":
        args.sr24_adaptive_dense_fallback_gate_up_fraction,
        "adaptive_dense_fallback_down_fraction":
        args.sr24_adaptive_dense_fallback_down_fraction,
        "adaptive_dense_fallback_small_down_no_residual":
        args.sr24_adaptive_dense_fallback_small_down_no_residual,
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
        "sort_bucket_rows": args.sr24_sort_bucket_rows,
        "disable_runtime_stats": args.sr24_disable_runtime_stats,
        "reduce_cpu_sync": args.sr24_reduce_cpu_sync,
        "sync_mask_state": args.sr24_sync_mask_state,
        "breakdown": args.sr24_breakdown,
        "breakdown_linear": args.sr24_breakdown_linear,
        "breakdown_exact_routing": args.sr24_breakdown_exact_routing,
        "breakdown_gpu_counts": args.sr24_breakdown_gpu_counts,
        "cudagraph_stats": args.sr24_cudagraph_stats,
        "force_cudagraph_none_for_mixed":
        args.sr24_force_cudagraph_none_for_mixed,
        "dynamic_auto_cudagraph": args.sr24_dynamic_auto_cudagraph,
        "cudagraph_bucket_active_hint":
        sr24_cudagraph_bucket_active_hint(args, method, batch_size),
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return EVAL_ROOT / "temp/vllm_compile_cache" / f"sr24_{digest}"


def sr24_selective_policy_forces_all_residual(args: argparse.Namespace) -> bool:
    if args.sr24_selective_non_draft_policy != "all":
        return False
    if int(args.sr24_selective_max_residual_draft_rows) > 0:
        return False
    if float(args.sr24_threshold) < 1.0:
        return False
    return args.sr24_selective_residual_policy in {
        "all_if_any_low",
        "batch_all_if_any_low",
        "low_confidence",
    }


def speclink_t08_allows_cudagraph(args: argparse.Namespace) -> bool:
    # compressed_dense materializes a residual weight with advanced indexing in
    # the forward path; CUDA Graph capture rejects that operation. Keep graph
    # ablations on residual backends that avoid rebuilding dense residual
    # weights during replay.
    #
    # Dynamic selective masks are updated outside the model call and are read
    # through SR24 global/context state. GSM8K replays showed graph-mode output
    # drift for dynamic auto/mixed masks, so only static mask-state ablations
    # are allowed to use the graph path by default. The exception is the
    # conservative dense-fallback path: with an exact per-step mask-state sync,
    # high-residual steps can be promoted to an all_residual dense fastpath.
    # Mixed steps are still protected by SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED
    # inside gpu_model_runner, so launching the server without --enforce-eager
    # only gives graph coverage to no/all-residual steps.
    allowed_states = {"all_residual", "no_residual"}
    if not args.sr24_force_cudagraph_none_for_mixed:
        # Experimental graph-safety ablation: static mixed keeps the SR24
        # residual-mask pointer stable and only updates its contents before the
        # target forward. Dynamic "auto" remains eager.
        allowed_states.add("mixed")
    scaled_bucket_can_use_graph = (
        not args.sr24_residual_bucket_scale_by_active
        or args.sr24_cudagraph_bucket
    )
    dynamic_auto_can_use_graph = (
        args.sr24_dynamic_auto_cudagraph
        and args.sr24_static_mask_state == "auto"
        and not args.sr24_force_cudagraph_none_for_mixed
        and scaled_bucket_can_use_graph
        and not args.sr24_route_all_residual_rows
        and not args.sr24_route_reuse_base_output
        and (
            int(args.sr24_residual_bucket_size) <= 0
            or args.sr24_cudagraph_bucket
        )
    )
    dense_fallback_can_use_graph = (
        args.sr24_static_mask_state == "auto"
        and args.sr24_sync_mask_state
        and 0.0 <= float(args.sr24_route_dense_fallback_fraction) <= 1.0
    )
    inferred_all_residual_can_use_graph = (
        args.sr24_static_mask_state == "auto"
        and args.sr24_reduce_cpu_sync
        and not args.sr24_sync_mask_state
        and sr24_selective_policy_forces_all_residual(args)
    )
    if (
        args.sr24_static_mask_state not in allowed_states
        and not dynamic_auto_can_use_graph
        and not dense_fallback_can_use_graph
        and not inferred_all_residual_can_use_graph
    ):
        return False
    return (
        args.sr24_allow_cudagraph
        and args.sr24_static_mask_buffer
        and args.sr24_reduce_cpu_sync
        and args.sr24_residual_backend in {"torch_sparse", "dense_rows"}
    )


def all_corrected_allows_cudagraph(args: argparse.Namespace) -> bool:
    effective_residual_backend = sr24_effective_residual_backend(
        args, "all_corrected_24"
    )
    if not args.sr24_allow_cudagraph:
        return False
    if args.sr24_all_corrected_dense_fastpath:
        return True
    if (
        args.sr24_full_residual_early_dense
        and effective_residual_backend == "dense_rows"
        and (
            args.sr24_static_mask_state == "all_residual"
            or args.sr24_disable_runtime_stats
        )
        and not args.sr24_breakdown
    ):
        # The SR24 hooks remain attached for accounting, but every corrected
        # Linear returns the dense GEMM before sparse-base dispatch. In
        # stats-off all_corrected runs, the scheduler returns the same
        # all-residual plan state without materializing a mask, so this is
        # graph-safe even when the user leaves static_mask_state=auto. Keep
        # breakdown rows eager because Linear CUDA-event timing is diagnostic
        # and should not be captured.
        return True
    if args.sr24_backend != "torch_sparse":
        return False
    if effective_residual_backend == "torch_sparse":
        # The full-model PyTorch SparseSemiStructuredTensor dispatch path can
        # inflate CUDA Graph memory enough to leave too little KV cache for the
        # Llama3.1+EAGLE3 bs64 all-corrected run at max_model_len=4096. Narrow
        # operator ablations such as gate_up=16-31 are bounded enough to test
        # the regular sparse graph path, which is the fastest exact graph path
        # in the current microbench. Full-scope runs still require the explicit
        # direct cuSPARSELt escape hatch.
        return bool(args.sr24_direct_cslt_linear) or sr24_has_narrow_residual_scope(args)
    (
        residual_out_chunk,
        cache_compressed_residual_weight,
        prewarm_compressed_residual_weight,
        _,
    ) = sr24_compressed_residual_runtime_settings(args, "all_corrected_24")
    return (
        effective_residual_backend == "compressed_dense"
        and cache_compressed_residual_weight
        and prewarm_compressed_residual_weight
        and int(residual_out_chunk) <= 0
        and not args.sr24_compressed_residual_triton
    )


def _csv_items(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def speclink_t08_static_densefastpath(args: argparse.Namespace) -> bool:
    if (
        args.sr24_static_mask_state != "all_residual"
        or not args.sr24_static_all_residual_dense_fastpath
    ):
        return False
    target_leafs = _csv_items(args.sr24_target_leafs)
    residual_leafs = _csv_items(args.sr24_residual_target_leafs)
    # Empty residual leafs means "same as target leafs" inside SR24. Only that
    # all-target-residual case is fully dense and safe for default Inductor
    # compile. If residual_leafs is a proper subset, some modules are still
    # sparse base-only and must use the graph-only, no-Inductor path.
    return not residual_leafs or residual_leafs == target_leafs


def sr24_method_uses_default_vllm_compile(
    args: argparse.Namespace,
    method: str,
) -> bool:
    if method == "base_only_24" and args.sr24_backend == "torch_sparse":
        # Do not let base-only use vLLM's default Inductor compile path. The
        # semi-structured sparse verifier path can still fail there during
        # startup profiling with lazy custom-kernel storage. When the explicit
        # --sr24-base-only-allow-compile ablation is set, the runner uses the
        # SR24 graph-only compilation config from effective_vllm_compilation_config()
        # instead of the default compile path.
        return False
    if (
        method == "speclink_t08"
        and args.sr24_direct_cpu_route_rows
        and args.sr24_route_all_residual_rows
    ):
        # Direct CPU row-list routing uses per-step row tensors. GSM8K gates
        # showed it is correct in eager mode but can diverge under vLLM
        # compile/CUDA-Graph replay, even when mixed steps are forced to
        # CUDAGraphMode.NONE. Keep it out of the default compile path.
        return False
    if args.sr24_default_vllm_compile:
        return True
    if method == "all_corrected_24" and args.sr24_all_corrected_dense_fastpath:
        # all_corrected densefastpath is a dense-equivalent/no-op control.
        # Using the SR24 graph-only cache root/compile profile can make this
        # control look artificially slow even though no SR24 Linear hook runs.
        return True
    if (
        method == "all_corrected_24"
        and args.sr24_full_residual_early_dense
        and sr24_effective_residual_backend(args, method) == "dense_rows"
        and (
            args.sr24_static_mask_state == "all_residual"
            or args.sr24_disable_runtime_stats
        )
        and not args.sr24_breakdown
    ):
        # This is also dense-equivalent: SR24 hooks remain attached, but each
        # hooked Linear returns the original dense GEMM before sparse-base
        # dispatch. The default vLLM compile path is faster than the SR24
        # graph-only compile profile for this exact control.
        return True
    if method == "speclink_t08" and speclink_t08_static_densefastpath(args):
        return True
    return False


def sr24_requires_enforce_eager(args: argparse.Namespace, method: str) -> bool:
    if method not in SR24_METHODS:
        return False
    if (
        method == "base_only_24"
        and args.sr24_backend == "torch_sparse"
        and not args.sr24_base_only_allow_compile
    ):
        return True
    if (
        method == "speclink_t08"
        and args.sr24_direct_cpu_route_rows
        and args.sr24_route_all_residual_rows
    ):
        return True
    if sr24_method_uses_default_vllm_compile(args, method):
        return False
    if args.sr24_allow_cudagraph and method == "base_only_24":
        return False
    if method == "all_corrected_24" and all_corrected_allows_cudagraph(args):
        return False
    if method == "speclink_t08" and speclink_t08_allows_cudagraph(args):
        return False
    if method == "all_corrected_24" and args.sr24_all_corrected_dense_fastpath:
        return False
    if (
        method == "base_only_24"
        and args.sr24_backend in {"dense_zero", "prototype"}
    ):
        return False
    return True


def effective_vllm_compilation_config(
    args: argparse.Namespace,
    method: str,
    batch_size: int,
) -> str:
    if args.vllm_compilation_config:
        return args.vllm_compilation_config
    if sr24_method_uses_default_vllm_compile(args, method):
        return ""
    if (
        not args.sr24_allow_cudagraph
        or method not in {"base_only_24", "all_corrected_24", "speclink_t08"}
        or args.sr24_backend != "torch_sparse"
    ):
        return ""
    if method == "all_corrected_24":
        effective_residual_backend = sr24_effective_residual_backend(
            args, method
        )
        if args.sr24_all_corrected_dense_fastpath:
            return ""
        if (
            args.sr24_full_residual_early_dense
            and effective_residual_backend == "dense_rows"
            and all_corrected_allows_cudagraph(args)
        ):
            pass
        elif effective_residual_backend == "compressed_dense":
            if not all_corrected_allows_cudagraph(args):
                return ""
        elif effective_residual_backend != "torch_sparse":
            return ""
    if method == "speclink_t08" and not speclink_t08_allows_cudagraph(args):
        return ""
    verifier_tokens = batch_size * (args.eagle3_k + 1)
    capture_size = max(1024, int(math.ceil(verifier_tokens / 16.0) * 16))
    return json.dumps(
        {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_cudagraph_capture_size": capture_size,
        },
        separators=(",", ":"),
    )


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
    if args.vllm_dtype:
        command.extend(["--dtype", args.vllm_dtype])
    compilation_config = effective_vllm_compilation_config(args, method, batch_size)
    if compilation_config:
        command.extend(["--compilation-config", compilation_config])
    if args.disable_chunked_prefill:
        command.append("--no-enable-chunked-prefill")
    smurfs_max_k = smurfs_dynamic_max_k_for_batch(args, batch_size)
    if sr24_requires_enforce_eager(args, method):
        command.append("--enforce-eager")
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
    elif method_uses_eagle3(method):
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
        if key.startswith(SMURFS_DYNAMIC_ENV_PREFIX) or key.startswith(
                SR24_ENV_PREFIX) or key.startswith(CUDAGRAPH_STATS_ENV_PREFIX):
            server_env.pop(key, None)
    if args.sr24_breakdown or args.sr24_cudagraph_stats:
        cudagraph_stats = run_dir / "cudagraph_stats.jsonl"
        server_env["SPECLINK_CUDAGRAPH_STATS_PATH"] = str(cudagraph_stats)
        server_env["SPECLINK_CUDAGRAPH_STATS_INTERVAL"] = str(
            max(1, int(args.sr24_stats_interval))
        )
        command_meta["cudagraph_stats"] = str(cudagraph_stats)
    else:
        server_env.pop("SPECLINK_CUDAGRAPH_STATS_PATH", None)
        server_env.pop("SPECLINK_CUDAGRAPH_STATS_INTERVAL", None)
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
    if method in SR24_METHODS:
        sr24_log = run_dir / "speclink_sr24_events.jsonl"
        sr24_stats = run_dir / "speclink_sr24_stats.json"
        sr24_breakdown = run_dir / "speclink_sr24_breakdown.json"
        sr24_effective_backend = sr24_effective_residual_backend(args, method)
        (
            sr24_residual_out_chunk,
            sr24_cache_compressed_residual_weight,
            sr24_prewarm_compressed_residual_weight,
            sr24_auto_compressed_residual_fastpath,
        ) = sr24_compressed_residual_runtime_settings(args, method)
        server_env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")
        server_env.update({
            "SPECLINK_SR24_ENABLE": "1",
            "SPECLINK_SR24_MODE": sr24_runtime_mode(method),
            "SPECLINK_SR24_BACKEND": args.sr24_backend,
            "SPECLINK_SR24_RESIDUAL_BACKEND": sr24_effective_backend,
            "SPECLINK_SR24_RESIDUAL_BACKEND_BY_LEAF":
            args.sr24_residual_backend_by_leaf,
            "SPECLINK_SR24_RESIDUAL_DEVICE": args.sr24_residual_device,
            "SPECLINK_SR24_REQUIRE_GPU_RESIDUAL":
            "1" if args.sr24_require_gpu_residual else "0",
            "SPECLINK_SR24_THRESHOLD": str(args.sr24_threshold),
            "SPECLINK_SR24_MASK_PATH": str(args.sr24_mask_path),
            "SPECLINK_SR24_LOG": str(sr24_log),
            "SPECLINK_SR24_STATS_PATH": str(sr24_stats),
            "SPECLINK_SR24_STATS_INTERVAL": str(args.sr24_stats_interval),
            "SPECLINK_SR24_BREAKDOWN": "1" if args.sr24_breakdown else "0",
            "SPECLINK_SR24_BREAKDOWN_PATH": str(sr24_breakdown),
            "SPECLINK_SR24_BREAKDOWN_INTERVAL":
            str(args.sr24_breakdown_interval),
            "SPECLINK_SR24_BREAKDOWN_LINEAR":
            "1" if args.sr24_breakdown_linear else "0",
            "SPECLINK_SR24_BREAKDOWN_EXACT_ROUTING":
            "1" if args.sr24_breakdown_exact_routing else "0",
            "SPECLINK_SR24_BREAKDOWN_SYNC_COUNTS":
            "1" if args.sr24_breakdown_exact_routing else "0",
            "SPECLINK_SR24_BREAKDOWN_GPU_COUNTS":
            "1" if args.sr24_breakdown_gpu_counts else "0",
            "SPECLINK_SR24_REDUCE_CPU_SYNC":
            "1" if args.sr24_reduce_cpu_sync else "0",
            "SPECLINK_SR24_SYNC_MASK_STATE":
            "1" if args.sr24_sync_mask_state else "0",
            "SPECLINK_SR24_STATIC_MASK_STATE": args.sr24_static_mask_state,
            "SPECLINK_SR24_STATIC_ALL_RESIDUAL_DENSE_FASTPATH":
            "1" if args.sr24_static_all_residual_dense_fastpath else "0",
            "SPECLINK_SR24_DIRECT_CSLT_LINEAR":
            "1" if sr24_effective_direct_cslt_linear(args, method) else "0",
            "SPECLINK_SR24_BASE_ONLY_DENSE_NONVERIFY":
            "1" if args.sr24_base_only_dense_nonverify else "0",
            "SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_MAX_ROWS":
            str(args.sr24_base_only_dense_verify_max_rows),
            "SPECLINK_SR24_STATIC_MASK_BUFFER":
            "1" if args.sr24_static_mask_buffer else "0",
            "SPECLINK_SR24_BATCHED_MASK_BUILDER":
            "1" if args.sr24_batched_mask_builder else "0",
            "SPECLINK_SR24_BATCHED_UNIFORM_DIRECT":
            "1" if args.sr24_batched_uniform_direct else "0",
            "SPECLINK_SR24_GPU_COUNT_MASK_BUILDER":
            "1" if args.sr24_gpu_count_mask_builder else "0",
            "SPECLINK_SR24_GATE_UP_SPLIT":
            args.sr24_gate_up_split,
            "SPECLINK_SR24_GATE_UP_CHANNEL_DENSE_FRACTION":
            str(args.sr24_gate_up_channel_dense_fraction),
            "SPECLINK_SR24_GATE_UP_CHANNEL_STRATEGY":
            args.sr24_gate_up_channel_strategy,
            "SPECLINK_SR24_GATE_UP_CHANNEL_FUSED_ACT":
            "1" if args.sr24_gate_up_channel_fused_act else "0",
            "SPECLINK_SR24_ROW_ROUTED_MLP":
            "1" if args.sr24_row_routed_mlp else "0",
            "SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR":
            "1" if args.sr24_row_routed_down_linear else "0",
            "SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT":
            "1" if args.sr24_row_routed_mlp_reuse_base_output else "0",
            "SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS":
            str(args.sr24_row_routed_mlp_min_dense_rows),
            "SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS":
            str(args.sr24_row_routed_mlp_max_dense_rows),
            "SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS":
            str(args.sr24_row_routed_mlp_max_base_rows),
            "SPECLINK_SR24_DENSE_FALLBACK_NONUNIFORM":
            "1" if args.sr24_dense_fallback_nonuniform else "0",
            "SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED":
            "1" if args.sr24_force_cudagraph_none_for_mixed else "0",
            "SPECLINK_SR24_MASK_BUFFER_CAPACITY":
            str(args.sr24_mask_buffer_capacity),
            "SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH":
            "1" if args.sr24_all_corrected_dense_fastpath else "0",
            "SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE":
            "1" if args.sr24_full_residual_early_dense else "0",
            "SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT":
            "1" if args.sr24_selective_correct_non_draft else "0",
            "SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY":
            args.sr24_selective_non_draft_policy,
            "SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY":
            args.sr24_selective_residual_policy,
            "SPECLINK_SR24_PREFIX_THRESHOLD":
            (
                ""
                if args.sr24_prefix_threshold < 0
                else str(args.sr24_prefix_threshold)
            ),
            "SPECLINK_SR24_SELECTIVE_EXTRA_AFTER_LOW":
            str(args.sr24_selective_extra_after_low),
            "SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL":
            str(args.sr24_selective_min_prefix_residual),
            "SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS":
            str(args.sr24_selective_max_residual_draft_rows),
            "SPECLINK_SR24_LOW_CONFIDENCE_CAP_BY_RISK":
            "1" if args.sr24_low_confidence_cap_by_risk else "0",
            "SPECLINK_SR24_EARLY_DENSE_TOKENS":
            str(args.sr24_early_dense_tokens),
            "SPECLINK_SR24_RESIDUAL_BUCKET_SIZE":
            str(args.sr24_residual_bucket_size),
            "SPECLINK_SR24_RESIDUAL_BUCKET_SCALE_BY_ACTIVE":
            "1" if args.sr24_residual_bucket_scale_by_active else "0",
            "SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY":
            "1" if args.sr24_residual_bucket_priority else "0",
            "SPECLINK_SR24_DIRECT_POSITION_BUCKET":
            "1" if args.sr24_direct_position_bucket else "0",
            "SPECLINK_SR24_BONUS_PRIORITY": str(args.sr24_bonus_priority),
            "SPECLINK_SR24_DRAFT_POSITION_PRIORITY_SCALE":
            str(args.sr24_draft_position_priority_scale),
            "SPECLINK_SR24_CUDAGRAPH_BUCKET":
            "1" if args.sr24_cudagraph_bucket else "0",
            "SPECLINK_SR24_ROUTE_BUCKET_ROWS":
            "1" if args.sr24_route_bucket_rows else "0",
            "SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS":
            "1" if args.sr24_route_all_residual_rows else "0",
            "SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET":
            "1" if args.sr24_route_all_skip_bucket else "0",
            "SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS":
            "1" if args.sr24_direct_cpu_route_rows else "0",
        "SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT":
        "1" if args.sr24_route_reuse_base_output else "0",
        "SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH":
        "1" if args.sr24_route_contiguous_fastpath else "0",
        "SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION":
        str(args.sr24_route_dense_fallback_fraction),
        "SPECLINK_SR24_ROUTE_MIN_DENSE_ROWS":
        str(args.sr24_route_min_dense_rows),
        "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS":
        str(args.sr24_route_min_base_rows),
        "SPECLINK_SR24_ROUTE_MAX_DENSE_FRACTION":
        str(args.sr24_route_max_dense_fraction),
        "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK":
        "1" if args.sr24_adaptive_dense_fallback else "0",
            "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS":
            str(args.sr24_adaptive_dense_fallback_small_rows),
            "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_GATE_UP_FRACTION":
            str(args.sr24_adaptive_dense_fallback_gate_up_fraction),
            "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_DOWN_FRACTION":
            str(args.sr24_adaptive_dense_fallback_down_fraction),
            "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL":
            "1" if args.sr24_adaptive_dense_fallback_small_down_no_residual else "0",
            "SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY":
            "1" if args.sr24_triton_route_assembly else "0",
            "SPECLINK_SR24_TRITON_BUCKET_OVERRIDE":
            "1" if args.sr24_triton_bucket_override else "0",
            "SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM":
            "1" if args.sr24_triton_bucket_dense_gemm else "0",
            "SPECLINK_SR24_TRITON_BUCKET_SCATTER":
            "1" if args.sr24_triton_bucket_scatter else "0",
            "SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_M":
            str(args.sr24_triton_bucket_dense_block_m),
            "SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_N":
            str(args.sr24_triton_bucket_dense_block_n),
            "SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_K":
            str(args.sr24_triton_bucket_dense_block_k),
            "SPECLINK_SR24_BUCKET_DENSE_COPY":
            "1" if args.sr24_bucket_dense_copy else "0",
            "SPECLINK_SR24_BUCKET_DENSE_COPY_ACTIVE_ONLY":
            "1" if args.sr24_bucket_dense_copy_active_only else "0",
            "SPECLINK_SR24_SORT_BUCKET_ROWS":
            "1" if args.sr24_sort_bucket_rows else "0",
            "SPECLINK_SR24_DISABLE_RUNTIME_STATS":
            "1" if args.sr24_disable_runtime_stats else "0",
        })
        method_uses_default_compile = sr24_method_uses_default_vllm_compile(
            args, method)
        command_meta["sr24_effective_default_vllm_compile"] = (
            method_uses_default_compile)
        command_meta["sr24_effective_residual_backend"] = (
            sr24_effective_backend
        )
        if args.sr24_allow_cudagraph and not method_uses_default_compile:
            # SR24 behavior depends on env-gated Python branches that vLLM's
            # default compile-cache key does not currently include. Use a
            # separate cache root per SR24 env fingerprint so static/dynamic
            # mask-state runs cannot reuse a graph from a different SR24 mode,
            # while still allowing same-config reruns to benefit from cache.
            # Do not set this for --sr24-default-vllm-compile; dense/no-op
            # correctness checks must use the normal vLLM cache path.
            sr24_cache_root = sr24_compile_cache_root(args, method, batch_size)
            server_env["VLLM_CACHE_ROOT"] = str(sr24_cache_root)
            command_meta["sr24_compile_cache_root"] = str(sr24_cache_root)
        sr24_mode_name = sr24_runtime_mode(method)
        sr24_is_all_corrected = sr24_mode_name == "all_corrected"
        if args.sr24_target_leafs:
            server_env["SPECLINK_SR24_TARGET_LEAFS"] = args.sr24_target_leafs
        else:
            server_env.pop("SPECLINK_SR24_TARGET_LEAFS", None)
        if sr24_is_all_corrected and args.sr24_target_leafs:
            server_env["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = (
                args.sr24_target_leafs
            )
        elif sr24_is_all_corrected:
            server_env.pop("SPECLINK_SR24_RESIDUAL_TARGET_LEAFS", None)
        elif args.sr24_residual_target_leafs:
            server_env["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = (
                args.sr24_residual_target_leafs
            )
        else:
            server_env.pop("SPECLINK_SR24_RESIDUAL_TARGET_LEAFS", None)
        if sr24_is_all_corrected:
            server_env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS", None)
            server_env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", None)
            if args.sr24_residual_layer_ids_by_leaf:
                server_env["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = (
                    args.sr24_residual_layer_ids_by_leaf
                )
            else:
                server_env.pop("SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF", None)
        elif args.sr24_base_only_layer_ids:
            server_env["SPECLINK_SR24_BASE_ONLY_LAYER_IDS"] = (
                args.sr24_base_only_layer_ids
            )
            if args.sr24_base_only_layer_ids_by_leaf:
                server_env["SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF"] = (
                    args.sr24_base_only_layer_ids_by_leaf
                )
            else:
                server_env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", None)
            if args.sr24_residual_layer_ids_by_leaf:
                server_env["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = (
                    args.sr24_residual_layer_ids_by_leaf
                )
            else:
                server_env.pop("SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF", None)
        else:
            server_env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS", None)
            if args.sr24_base_only_layer_ids_by_leaf:
                server_env["SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF"] = (
                    args.sr24_base_only_layer_ids_by_leaf
                )
            else:
                server_env.pop("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", None)
            if args.sr24_residual_layer_ids_by_leaf:
                server_env["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = (
                    args.sr24_residual_layer_ids_by_leaf
                )
            else:
                server_env.pop("SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF", None)
        server_env["SPECLINK_SR24_RESIDUAL_OUT_CHUNK"] = str(
            sr24_residual_out_chunk)
        server_env["SPECLINK_SR24_CACHE_COMPRESSED_RESIDUAL_WEIGHT"] = (
            "1" if sr24_cache_compressed_residual_weight else "0"
        )
        server_env["SPECLINK_SR24_PREWARM_COMPRESSED_RESIDUAL_WEIGHT"] = (
            "1" if sr24_prewarm_compressed_residual_weight else "0"
        )
        server_env["SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON"] = (
            "1" if args.sr24_compressed_residual_triton else "0"
        )
        server_env["SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_M"] = str(
            args.sr24_compressed_residual_block_m)
        server_env["SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_N"] = str(
            args.sr24_compressed_residual_block_n)
        server_env["SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_G"] = str(
            args.sr24_compressed_residual_block_g)
        server_env["SPECLINK_SR24_EXTRACT_CHUNK_ROWS"] = str(
            args.sr24_extract_chunk_rows)
        sr24_active_hint = sr24_cudagraph_bucket_active_hint(
            args, method, batch_size)
        server_env["SPECLINK_SR24_CUDAGRAPH_BUCKET_ACTIVE_HINT"] = str(
            sr24_active_hint)
        command_meta["sr24_cudagraph_bucket_active_hint"] = sr24_active_hint
        command_meta["speclink_sr24_log"] = str(sr24_log)
        command_meta["speclink_sr24_stats"] = str(sr24_stats)
        command_meta["sr24_auto_compressed_residual_fastpath"] = (
            sr24_auto_compressed_residual_fastpath
        )
        command_meta["sr24_effective_residual_out_chunk"] = (
            sr24_residual_out_chunk
        )
        command_meta["sr24_effective_cache_compressed_residual_weight"] = (
            sr24_cache_compressed_residual_weight
        )
        command_meta["sr24_effective_prewarm_compressed_residual_weight"] = (
            sr24_prewarm_compressed_residual_weight
        )
        if args.sr24_breakdown:
            command_meta["speclink_sr24_breakdown"] = str(sr24_breakdown)
            command_meta["speclink_sr24_breakdown_linear"] = (
                args.sr24_breakdown_linear
            )
            command_meta["speclink_sr24_breakdown_exact_routing"] = (
                args.sr24_breakdown_exact_routing
            )
            command_meta["speclink_sr24_breakdown_gpu_counts"] = (
                args.sr24_breakdown_gpu_counts
            )
    else:
        server_env["SPECLINK_SR24_ENABLE"] = "0"
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


def latest_sr24_summary(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    latest_cudagraph: dict[str, Any] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") in {"sr24_verify_summary", "sr24_verify_mask"}:
                latest = record
            elif record.get("event") == "sr24_cudagraph_summary":
                latest_cudagraph = record
    if latest_cudagraph:
        if not latest:
            latest = {}
        latest["cudagraph_mode_counts"] = latest_cudagraph.get(
            "cudagraph_mode_counts") or {}
        latest["cudagraph_steps"] = latest_cudagraph.get("cudagraph_steps")
    return latest


def latest_cudagraph_summary(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "speclink_cudagraph_summary":
                latest = record
    return latest


def server_cudagraph_profile_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest_counts: dict[str, int] = {}
    pattern = re.compile(r"\b([A-Z_]+)=([0-9]+)\b")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Profiling CUDA graph memory:" not in line:
                continue
            counts = {
                key: int(value)
                for key, value in pattern.findall(line)
                if key in {"FULL", "PIECEWISE", "FULL_DECODE_ONLY", "NONE"}
            }
            if counts:
                latest_counts = counts
    if not latest_counts:
        return {}
    return {
        "server_cudagraph_profile_counts":
        json.dumps(latest_counts, sort_keys=True)
    }


def cudagraph_delta_summary(before: dict[str, Any],
                            after: dict[str, Any]) -> dict[str, Any]:
    if not after:
        return {}
    before_cg = before.get("cudagraph_mode_counts") or {}
    after_cg = after.get("cudagraph_mode_counts") or {}
    cg_delta: dict[str, int] = {}
    for key in sorted(set(before_cg) | set(after_cg)):
        try:
            value = int(after_cg.get(key) or 0) - int(before_cg.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value:
            cg_delta[key] = value
    if not cg_delta:
        return {}
    return {"sr24_cudagraph_mode_counts": json.dumps(cg_delta, sort_keys=True)}


def sr24_delta_summary(before: dict[str, Any],
                       after: dict[str, Any]) -> dict[str, Any]:
    if not after:
        return {}
    stats_interval = int(after.get("stats_interval") or 1)
    stats_exact = bool(after.get("stats_exact"))
    # With interval > 1 the event counters are still cumulative, but this
    # runner samples "latest event before/after measurement"; the window delta
    # can lag by up to interval-1 decode steps.
    window_stats_exact = stats_exact and stats_interval == 1
    summary: dict[str, Any] = {
        "sr24_sync_reduced_stats": after.get("sync_reduced_stats"),
        "sr24_stats_exact": window_stats_exact,
        "sr24_stats_interval": stats_interval,
        "sr24_selective_correct_non_draft":
        after.get("selective_correct_non_draft"),
        "sr24_selective_non_draft_policy":
        after.get("selective_non_draft_policy"),
        "sr24_selective_residual_policy":
        after.get("selective_residual_policy"),
        "sr24_runtime_prefix_threshold": after.get("prefix_threshold"),
        "sr24_selective_extra_after_low":
        after.get("selective_extra_after_low"),
        "sr24_selective_min_prefix_residual":
        after.get("selective_min_prefix_residual"),
        "sr24_selective_max_residual_draft_rows":
        after.get("selective_max_residual_draft_rows"),
        "sr24_low_confidence_cap_by_risk":
        after.get("low_confidence_cap_by_risk"),
        "sr24_early_dense_tokens": after.get("early_dense_tokens"),
        "sr24_sync_mask_state": after.get("sync_mask_state"),
        "sr24_static_mask_state": after.get("static_mask_state"),
        "sr24_mask_state": after.get("mask_state"),
        "sr24_mask_state_exact": after.get("mask_state_exact"),
    }
    before_cg = before.get("cudagraph_mode_counts") or {}
    after_cg = after.get("cudagraph_mode_counts") or {}
    cg_delta: dict[str, int] = {}
    for key in sorted(set(before_cg) | set(after_cg)):
        try:
            value = int(after_cg.get(key) or 0) - int(before_cg.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value:
            cg_delta[key] = value
    if cg_delta:
        summary["sr24_cudagraph_mode_counts"] = json.dumps(
            cg_delta, sort_keys=True)
    delta_keys = {
        "steps": "sr24_steps",
        "total_scheduled_tokens": "sr24_total_scheduled_tokens",
        "total_draft_tokens": "sr24_total_draft_tokens",
        "total_valid_draft_tokens": "sr24_total_valid_draft_tokens",
        "non_draft_tokens": "sr24_non_draft_tokens",
        "residual_draft_tokens": "sr24_residual_draft_tokens",
        "base_only_draft_tokens": "sr24_base_only_draft_tokens",
        "residual_non_draft_tokens": "sr24_residual_non_draft_tokens",
        "base_only_non_draft_tokens": "sr24_base_only_non_draft_tokens",
        "early_residual_draft_tokens": "sr24_early_residual_draft_tokens",
        "early_residual_non_draft_tokens":
        "sr24_early_residual_non_draft_tokens",
        "missing_score_tokens": "sr24_missing_score_tokens",
        "bucket_calls": "sr24_bucket_calls",
        "bucket_candidate_rows": "sr24_bucket_candidate_rows",
        "bucket_active_rows": "sr24_bucket_active_rows",
        "bucket_total_rows": "sr24_bucket_total_rows",
        "bucket_residual_requested_rows":
        "sr24_bucket_residual_requested_rows",
        "dense_fallback_nonuniform_steps":
        "sr24_dense_fallback_nonuniform_steps",
        "adaptive_dense_fallback_calls":
        "sr24_adaptive_dense_fallback_calls",
        "adaptive_dense_fallback_rows":
        "sr24_adaptive_dense_fallback_rows",
        "adaptive_dense_fallback_candidate_rows":
        "sr24_adaptive_dense_fallback_candidate_rows",
    }
    for source, target in delta_keys.items():
        after_value = after.get(source)
        if after_value is None:
            continue
        before_value = before.get(source) or 0
        try:
            summary[target] = int(after_value) - int(before_value)
        except (TypeError, ValueError):
            continue
    # Some low-sync SR24 routes deliberately avoid exact mixed-mask counters.
    # If the actual execution plan is all/no-residual, the effective routing is
    # still known exactly even when the raw pre-fallback mask counters are not.
    if summary.get("sr24_mask_state") == "all_residual":
        summary["sr24_residual_draft_tokens"] = int(
            summary.get("sr24_total_valid_draft_tokens") or 0
        )
        summary["sr24_base_only_draft_tokens"] = 0
        summary["sr24_residual_non_draft_tokens"] = int(
            summary.get("sr24_non_draft_tokens") or 0
        )
        summary["sr24_base_only_non_draft_tokens"] = 0
        summary["sr24_stats_exact"] = True
    elif summary.get("sr24_mask_state") == "no_residual":
        summary["sr24_residual_draft_tokens"] = 0
        summary["sr24_base_only_draft_tokens"] = int(
            summary.get("sr24_total_valid_draft_tokens") or 0
        )
        summary["sr24_residual_non_draft_tokens"] = 0
        summary["sr24_base_only_non_draft_tokens"] = int(
            summary.get("sr24_non_draft_tokens") or 0
        )
        summary["sr24_stats_exact"] = True
    timing_delta_keys = {
        "scheduler_mask_wall_cpu_ms": "sr24_scheduler_mask_wall_cpu_ms",
        "scheduler_materialize_counts_wall_cpu_ms":
        "sr24_scheduler_materialize_counts_wall_cpu_ms",
        "scheduler_pending_scores_pop_wall_cpu_ms":
        "sr24_scheduler_pending_scores_pop_wall_cpu_ms",
        "scheduler_batched_mask_builder_wall_cpu_ms":
        "sr24_scheduler_batched_mask_builder_wall_cpu_ms",
        "scheduler_request_routing_loop_wall_cpu_ms":
        "sr24_scheduler_request_routing_loop_wall_cpu_ms",
        "scheduler_batch_all_apply_wall_cpu_ms":
        "sr24_scheduler_batch_all_apply_wall_cpu_ms",
        "scheduler_mask_state_wall_cpu_ms":
        "sr24_scheduler_mask_state_wall_cpu_ms",
        "scheduler_static_mask_copy_wall_cpu_ms":
        "sr24_scheduler_static_mask_copy_wall_cpu_ms",
        "scheduler_row_index_bucket_wall_cpu_ms":
        "sr24_scheduler_row_index_bucket_wall_cpu_ms",
        "scheduler_residual_bucket_wall_cpu_ms":
        "sr24_scheduler_residual_bucket_wall_cpu_ms",
        "scheduler_mixed_row_indices_wall_cpu_ms":
        "sr24_scheduler_mixed_row_indices_wall_cpu_ms",
        "scheduler_direct_cpu_route_rows_wall_cpu_ms":
        "sr24_scheduler_direct_cpu_route_rows_wall_cpu_ms",
    }
    for source, target in timing_delta_keys.items():
        after_value = after.get(source)
        if after_value is None:
            continue
        before_value = before.get(source) or 0.0
        try:
            summary[target] = float(after_value) - float(before_value)
        except (TypeError, ValueError):
            continue
    steps_for_timing = int(summary.get("sr24_steps") or 0)
    for target in timing_delta_keys.values():
        value = summary.get(target)
        summary[f"{target}_per_step"] = (
            float(value) / steps_for_timing
            if value is not None and steps_for_timing else None
        )
    total_draft = int(summary.get("sr24_total_draft_tokens") or 0)
    residual = summary.get("sr24_residual_draft_tokens")
    base_only = summary.get("sr24_base_only_draft_tokens")
    summary["sr24_residual_draft_fraction"] = (
        int(residual) / total_draft
        if residual is not None and total_draft else None)
    summary["sr24_base_only_draft_fraction"] = (
        int(base_only) / total_draft
        if base_only is not None and total_draft else None)
    total_non_draft = int(summary.get("sr24_non_draft_tokens") or 0)
    residual_non_draft = summary.get("sr24_residual_non_draft_tokens")
    summary["sr24_residual_non_draft_fraction"] = (
        int(residual_non_draft) / total_non_draft
        if residual_non_draft is not None and total_non_draft else None)
    bucket_calls = int(summary.get("sr24_bucket_calls") or 0)
    bucket_candidate = int(summary.get("sr24_bucket_candidate_rows") or 0)
    bucket_active = summary.get("sr24_bucket_active_rows")
    bucket_requested = summary.get("sr24_bucket_residual_requested_rows")
    summary["sr24_bucket_candidate_rows_per_call"] = (
        bucket_candidate / bucket_calls if bucket_calls else None)
    summary["sr24_bucket_active_rows_per_call"] = (
        int(bucket_active) / bucket_calls
        if bucket_calls and bucket_active is not None else None)
    summary["sr24_bucket_active_fraction_of_requested"] = (
        int(bucket_active) / int(bucket_requested)
        if (
            bucket_active is not None and bucket_requested is not None
            and int(bucket_requested) > 0
        ) else None)
    return summary


def sr24_static_summary(run_dir: Path,
                        args: argparse.Namespace,
                        method: str,
                        batch_size: int = 0) -> dict[str, Any]:
    path = run_dir / "speclink_sr24_stats.json"
    if not path.exists():
        return {}
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "sr24_preset": args.sr24_preset,
        "sr24_mode": stats.get("mode"),
        "sr24_backend": stats.get("backend"),
        "sr24_residual_backend": stats.get("residual_backend"),
        "sr24_residual_device": stats.get("residual_device"),
        "sr24_require_gpu_residual": stats.get("require_gpu_residual"),
        "sr24_threshold": stats.get("threshold"),
        "sr24_prefix_threshold": stats.get("selective_prefix_threshold"),
        "sr24_all_corrected_dense_fastpath":
        stats.get("all_corrected_dense_fastpath"),
        "sr24_full_residual_early_dense":
        stats.get("full_residual_early_dense"),
        "sr24_dense_fastpath_noop": stats.get("dense_fastpath_noop"),
        "sr24_selective_correct_non_draft_static":
        stats.get("selective_correct_non_draft"),
        "sr24_selective_non_draft_policy_static":
        stats.get("selective_non_draft_policy"),
        "sr24_selective_residual_policy_static":
        stats.get("selective_residual_policy"),
        "sr24_prefix_threshold_static":
        stats.get("selective_prefix_threshold"),
        "sr24_selective_extra_after_low_static":
        stats.get("selective_extra_after_low"),
        "sr24_selective_min_prefix_residual_static":
        stats.get("selective_min_prefix_residual"),
        "sr24_selective_max_residual_draft_rows_static":
        stats.get("selective_max_residual_draft_rows"),
        "sr24_low_confidence_cap_by_risk_static":
        stats.get("low_confidence_cap_by_risk"),
        "sr24_early_dense_tokens_static": stats.get("early_dense_tokens"),
        "sr24_sync_mask_state_static": stats.get("sync_mask_state"),
        "sr24_static_mask_state_static": stats.get("static_mask_state"),
        "sr24_static_all_residual_dense_fastpath_static":
        stats.get("static_all_residual_dense_fastpath"),
        "sr24_default_vllm_compile_static": args.sr24_default_vllm_compile,
        "sr24_disable_compile_cache_static":
        False,
        "sr24_allow_cudagraph": args.sr24_allow_cudagraph,
        "sr24_dynamic_auto_cudagraph": args.sr24_dynamic_auto_cudagraph,
        "sr24_compile_cache_root_static": (
            str(sr24_compile_cache_root(args, method, batch_size))
            if (
                args.sr24_allow_cudagraph
                and not args.sr24_default_vllm_compile
                and method in SR24_METHODS
            )
            else ""
        ),
        "sr24_direct_cslt_linear_static": stats.get("direct_cslt_linear"),
        "sr24_base_only_dense_nonverify_static":
        stats.get("base_only_dense_nonverify"),
        "sr24_base_only_allow_compile_static":
        args.sr24_base_only_allow_compile,
        "sr24_base_only_dense_verify_max_rows_static":
        stats.get("base_only_dense_verify_max_rows"),
        "sr24_static_mask_buffer": stats.get("static_mask_buffer"),
        "sr24_batched_mask_builder": stats.get("batched_mask_builder"),
        "sr24_batched_uniform_direct": stats.get("batched_uniform_direct"),
        "sr24_gpu_count_mask_builder": stats.get("gpu_count_mask_builder"),
        "sr24_gate_up_split": stats.get("gate_up_split"),
        "sr24_gate_up_channel_dense_fraction":
        stats.get("gate_up_channel_dense_fraction"),
        "sr24_gate_up_channel_strategy":
        stats.get("gate_up_channel_strategy"),
        "sr24_gate_up_channel_fused_act":
        stats.get("gate_up_channel_fused_act"),
        "sr24_row_routed_mlp": stats.get("row_routed_mlp"),
        "sr24_row_routed_down_linear": stats.get("row_routed_down_linear"),
        "sr24_row_routed_mlp_reuse_base_output":
        stats.get("row_routed_mlp_reuse_base_output"),
        "sr24_row_routed_mlp_min_dense_rows":
        stats.get("row_routed_mlp_min_dense_rows"),
        "sr24_row_routed_mlp_max_dense_rows":
        stats.get("row_routed_mlp_max_dense_rows"),
        "sr24_row_routed_mlp_max_base_rows":
        stats.get("row_routed_mlp_max_base_rows"),
        "sr24_dense_fallback_nonuniform":
        stats.get("dense_fallback_nonuniform"),
        "sr24_force_cudagraph_none_for_mixed":
        stats.get("force_cudagraph_none_for_mixed"),
        "sr24_static_mask_buffer_capacity":
        stats.get("static_mask_buffer_capacity"),
        "sr24_residual_bucket_size": stats.get("residual_bucket_size"),
        "sr24_residual_bucket_scale_by_active":
        stats.get("residual_bucket_scale_by_active"),
        "sr24_cudagraph_bucket": stats.get("cudagraph_bucket"),
        "sr24_cudagraph_bucket_active_hint": (
            stats.get("cudagraph_bucket_active_hint")
            or sr24_cudagraph_bucket_active_hint(args, method, batch_size)
        ),
        "sr24_residual_bucket_priority": stats.get("residual_bucket_priority"),
        "sr24_direct_position_bucket": stats.get("direct_position_bucket"),
        "sr24_bonus_priority": stats.get("bonus_priority"),
        "sr24_draft_position_priority_scale":
        stats.get("draft_position_priority_scale"),
        "sr24_route_bucket_rows": stats.get("route_bucket_rows"),
        "sr24_route_bucket_rows_graph_static_unsafe":
        stats.get("route_bucket_rows_graph_static_unsafe"),
        "sr24_route_all_residual_rows":
        stats.get("route_all_residual_rows"),
        "sr24_route_all_skip_bucket":
        stats.get("route_all_skip_bucket"),
        "sr24_direct_cpu_route_rows":
        stats.get("direct_cpu_route_rows"),
        "sr24_route_reuse_base_output":
        stats.get("route_reuse_base_output"),
        "sr24_route_contiguous_fastpath":
        stats.get("route_contiguous_fastpath"),
        "sr24_fixed_prefix_route_fastpath":
        stats.get("fixed_prefix_route_fastpath"),
        "sr24_route_dense_fallback_fraction":
        stats.get("route_dense_fallback_fraction"),
        "sr24_route_min_dense_rows": stats.get("route_min_dense_rows"),
        "sr24_route_min_base_rows": stats.get("route_min_base_rows"),
        "sr24_route_max_dense_fraction":
        stats.get("route_max_dense_fraction"),
        "sr24_adaptive_dense_fallback":
        stats.get("adaptive_dense_fallback"),
        "sr24_adaptive_dense_fallback_small_rows":
        stats.get("adaptive_dense_fallback_small_rows"),
        "sr24_adaptive_dense_fallback_gate_up_fraction":
        stats.get("adaptive_dense_fallback_gate_up_fraction"),
        "sr24_adaptive_dense_fallback_down_fraction":
        stats.get("adaptive_dense_fallback_down_fraction"),
        "sr24_adaptive_dense_fallback_small_down_no_residual":
        stats.get("adaptive_dense_fallback_small_down_no_residual"),
        "sr24_adaptive_dense_fallback_calls":
        stats.get("adaptive_dense_fallback_calls"),
        "sr24_adaptive_dense_fallback_rows":
        stats.get("adaptive_dense_fallback_rows"),
        "sr24_adaptive_dense_fallback_candidate_rows":
        stats.get("adaptive_dense_fallback_candidate_rows"),
        "sr24_triton_route_assembly": stats.get("triton_route_assembly"),
        "sr24_triton_bucket_override": stats.get("triton_bucket_override"),
        "sr24_triton_bucket_dense_gemm":
        stats.get("triton_bucket_dense_gemm"),
        "sr24_triton_bucket_scatter":
        stats.get("triton_bucket_scatter"),
        "sr24_triton_bucket_dense_block_m":
        stats.get("triton_bucket_dense_block_m"),
        "sr24_triton_bucket_dense_block_n":
        stats.get("triton_bucket_dense_block_n"),
        "sr24_triton_bucket_dense_block_k":
        stats.get("triton_bucket_dense_block_k"),
        "sr24_bucket_dense_copy": stats.get("bucket_dense_copy"),
        "sr24_bucket_dense_copy_active_only":
        stats.get("bucket_dense_copy_active_only"),
        "sr24_sort_bucket_rows": stats.get("sort_bucket_rows"),
        "sr24_linear_hooks_enabled": stats.get("linear_hooks_enabled"),
        "sr24_draft_scores_enabled": stats.get("draft_scores_enabled"),
        "sr24_runtime_stats_enabled": stats.get("runtime_stats_enabled"),
        "sr24_runtime_timing_enabled_static": stats.get("runtime_timing_enabled"),
        "sr24_target_leafs": ",".join(stats.get("target_leafs") or []),
        "sr24_residual_target_leafs":
        ",".join(stats.get("residual_target_leafs") or []),
        "sr24_base_only_layer_ids":
        ",".join(str(item) for item in (stats.get("base_only_layer_ids") or [])),
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
        "sr24_cache_compressed_residual_weight":
        stats.get("cache_compressed_residual_weight"),
        "sr24_prewarm_compressed_residual_weight":
        stats.get("prewarm_compressed_residual_weight"),
        "sr24_compressed_residual_triton":
        stats.get("compressed_residual_triton"),
        "sr24_compressed_residual_block_m":
        stats.get("compressed_residual_block_m"),
        "sr24_compressed_residual_block_n":
        stats.get("compressed_residual_block_n"),
        "sr24_compressed_residual_block_g":
        stats.get("compressed_residual_block_g"),
        "sr24_extract_chunk_rows": stats.get("residual_extract_chunk_rows"),
        "sr24_residual_extract_cpu_fallback_chunks":
        stats.get("residual_extract_cpu_fallback_chunks"),
        "sr24_residual_extract_cpu_fallback_module_count":
        stats.get("residual_extract_cpu_fallback_module_count"),
        "sr24_residual_backend_counts":
        json.dumps(stats.get("residual_backend_counts") or {}, sort_keys=True),
        "sr24_residual_device_counts":
        json.dumps(stats.get("residual_device_counts") or {}, sort_keys=True),
        "sr24_residual_cpu_module_count":
        stats.get("residual_cpu_module_count"),
        "sr24_residual_cuda_module_count":
        stats.get("residual_cuda_module_count"),
        "sr24_compressed_residual_runtime_on_gpu":
        stats.get("compressed_residual_runtime_on_gpu"),
        "sr24_compressed_residual_non_gpu_modules":
        ",".join(stats.get("compressed_residual_non_gpu_modules") or []),
        "sr24_module_count_attached": stats.get("module_count_attached"),
        "sr24_storage_over_dense": stats.get("storage_over_dense"),
        "sr24_actual_weight_storage_bytes":
        stats.get("actual_weight_storage_bytes"),
        "sr24_sparse_metadata_bytes": stats.get("sparse_metadata_bytes"),
        "sr24_mask_metadata_bytes": stats.get("mask_metadata_bytes"),
        "sr24_mask_cache_method": stats.get("mask_cache_method"),
        "sr24_mask_path": stats.get("mask_path"),
    }


def row_for_case(method: str, dataset_name: str, batch_size: int,
                 metrics: dict[str, Any], run_dir: Path,
                 args: argparse.Namespace,
                 smurfs_summary: dict[str, Any] | None,
                 sr24_summary: dict[str, Any] | None = None,
                 repeat_index: int = 1) -> dict[str, Any]:
    if method == "vllm_ar":
        k_value: Any = 0
    elif method == "vllm_fastdraft":
        k_value = args.fastdraft_k
    elif method_uses_eagle3(method):
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
        "repeat_index": repeat_index,
        "repeat_count": args.repeats,
        "K": k_value,
        "initial_k": (
            args.fastdraft_k if method == "vllm_fastdraft" else
            args.eagle3_k if method_uses_eagle3(method) else
            args.smurfs_initial_k if method == "smurfs_fastdraft" else 0),
        "max_new_tokens": args.max_tokens,
        "max_requests": metrics.get("max_requests", ""),
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
        **(sr24_summary or {}),
        "work_dir": str(run_dir),
    }


def failure_row(method: str, dataset_name: str, batch_size: int,
                run_dir: Path, args: argparse.Namespace,
                reason: str, repeat_index: int = 1) -> dict[str, Any]:
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
        "full_batch_output_tokens_per_second": None,
        "full_batch_output_tokens_prorated": None,
        "full_batch_window_s": None,
        "full_batch_completed_requests": None,
        "full_batch_active_threshold": batch_size,
        "output_tokens_total": 0,
        "request_latency_mean": None,
        "ttft_ms_mean": None,
        "tpot_ms_mean": None,
        "ttft_ms_p50": None,
        "ttft_ms_p90": None,
        "tpot_ms_p50": None,
        "tpot_ms_p90": None,
        "spec_acceptance_rate": None,
        "spec_acceptance_rate_pct": None,
        "spec_accepted_tokens": None,
        "spec_draft_tokens": None,
        "spec_estimated_steps": None,
        "spec_avg_selected_draft_tokens_per_step": None,
        "spec_avg_accepted_draft_tokens_per_step": None,
        "peak_gpu_memory_mib": None,
        "avg_gpu_util_pct": None,
        "peak_gpu_util_pct": None,
        "gpu_util_sample_count": 0,
        "first_error": reason,
    }
    return row_for_case(method, dataset_name, batch_size, metrics, run_dir,
                        args, None, repeat_index=repeat_index)


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
        "--repeats",
        str(args.repeats),
        "--prompt-limit",
        str(args.prompt_limit),
        "--min-requests-per-run",
        str(args.min_requests_per_run),
        "--fixed-total-requests",
        str(args.fixed_total_requests),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        *(
            ["--vllm-compilation-config", str(args.vllm_compilation_config)]
            if args.vllm_compilation_config
            else []
        ),
        *(["--disable-chunked-prefill"] if args.disable_chunked_prefill else []),
        "--fastdraft-k",
        str(args.fastdraft_k),
        "--eagle3-k",
        str(args.eagle3_k),
        "--sr24-preset",
        str(args.sr24_preset),
        "--sr24-threshold",
        str(args.sr24_threshold),
        "--sr24-backend",
        str(args.sr24_backend),
        "--sr24-residual-backend",
        str(args.sr24_residual_backend),
        "--sr24-residual-backend-by-leaf",
        str(args.sr24_residual_backend_by_leaf),
        "--sr24-residual-device",
        str(args.sr24_residual_device),
        *(["--sr24-require-gpu-residual"] if args.sr24_require_gpu_residual else []),
        "--sr24-mask-path",
        str(args.sr24_mask_path),
        "--sr24-stats-interval",
        str(args.sr24_stats_interval),
        *(["--sr24-breakdown"] if args.sr24_breakdown else []),
        *(["--sr24-breakdown-linear"] if args.sr24_breakdown_linear else []),
        *(
            ["--sr24-breakdown-exact-routing"]
            if args.sr24_breakdown_exact_routing
            else []
        ),
        *(["--sr24-breakdown-gpu-counts"] if args.sr24_breakdown_gpu_counts else []),
        *(["--sr24-cudagraph-stats"] if args.sr24_cudagraph_stats else []),
        "--sr24-breakdown-interval",
        str(args.sr24_breakdown_interval),
        *(["--sr24-reduce-cpu-sync"] if args.sr24_reduce_cpu_sync else []),
        *(
            []
            if args.sr24_sync_mask_state
            else ["--no-sr24-sync-mask-state"]
        ),
        "--sr24-static-mask-state",
        str(args.sr24_static_mask_state),
        *(
            ["--sr24-static-all-residual-dense-fastpath"]
            if args.sr24_static_all_residual_dense_fastpath
            else []
        ),
        *(
            ["--sr24-default-vllm-compile"]
            if args.sr24_default_vllm_compile
            else []
        ),
        *(
            ["--sr24-force-eager-after-preset"]
            if args.sr24_force_eager_after_preset
            else []
        ),
        *(
            ["--sr24-direct-cslt-linear"]
            if args.sr24_direct_cslt_linear
            else []
        ),
        *(
            []
            if args.sr24_auto_direct_cslt_base_only
            else ["--no-sr24-auto-direct-cslt-base-only"]
        ),
        *(
            ["--sr24-base-only-allow-compile"]
            if args.sr24_base_only_allow_compile
            else ["--no-sr24-base-only-allow-compile"]
        ),
        *(
            ["--sr24-base-only-dense-nonverify"]
            if args.sr24_base_only_dense_nonverify
            else []
        ),
        "--sr24-base-only-dense-verify-max-rows",
        str(args.sr24_base_only_dense_verify_max_rows),
        *(["--sr24-static-mask-buffer"] if args.sr24_static_mask_buffer else []),
        *(
            ["--sr24-batched-mask-builder"]
            if args.sr24_batched_mask_builder
            else []
        ),
        *(
            ["--sr24-batched-uniform-direct"]
            if args.sr24_batched_uniform_direct
            else []
        ),
        *(
            ["--sr24-gpu-count-mask-builder"]
            if args.sr24_gpu_count_mask_builder
            else []
        ),
        "--sr24-gate-up-split",
        str(args.sr24_gate_up_split),
        "--sr24-gate-up-channel-dense-fraction",
        str(args.sr24_gate_up_channel_dense_fraction),
        "--sr24-gate-up-channel-strategy",
        str(args.sr24_gate_up_channel_strategy),
        *(["--sr24-gate-up-channel-fused-act"]
          if args.sr24_gate_up_channel_fused_act else []),
        *(["--sr24-row-routed-mlp"] if args.sr24_row_routed_mlp else []),
        *(["--sr24-row-routed-down-linear"]
          if args.sr24_row_routed_down_linear else []),
        *(["--sr24-row-routed-mlp-reuse-base-output"]
          if args.sr24_row_routed_mlp_reuse_base_output else []),
        "--sr24-row-routed-mlp-min-dense-rows",
        str(args.sr24_row_routed_mlp_min_dense_rows),
        "--sr24-row-routed-mlp-max-dense-rows",
        str(args.sr24_row_routed_mlp_max_dense_rows),
        "--sr24-row-routed-mlp-max-base-rows",
        str(args.sr24_row_routed_mlp_max_base_rows),
        *(
            ["--no-sr24-force-cudagraph-none-for-mixed"]
            if not args.sr24_force_cudagraph_none_for_mixed else []
        ),
        *(
            ["--sr24-dynamic-auto-cudagraph"]
            if args.sr24_dynamic_auto_cudagraph else []
        ),
        "--sr24-mask-buffer-capacity",
        str(args.sr24_mask_buffer_capacity),
        *(["--sr24-allow-cudagraph"] if args.sr24_allow_cudagraph else []),
        *(
            []
            if args.sr24_all_corrected_dense_fastpath
            else ["--no-sr24-all-corrected-dense-fastpath"]
        ),
        *(
            ["--sr24-full-residual-early-dense"]
            if args.sr24_full_residual_early_dense
            else []
        ),
        *(
            []
            if args.sr24_selective_correct_non_draft
            else ["--no-sr24-selective-correct-non-draft"]
        ),
        "--sr24-selective-non-draft-policy",
        str(args.sr24_selective_non_draft_policy),
        "--sr24-selective-residual-policy",
        str(args.sr24_selective_residual_policy),
        "--sr24-prefix-threshold",
        str(args.sr24_prefix_threshold),
        "--sr24-selective-extra-after-low",
        str(args.sr24_selective_extra_after_low),
        "--sr24-selective-min-prefix-residual",
        str(args.sr24_selective_min_prefix_residual),
        "--sr24-selective-max-residual-draft-rows",
        str(args.sr24_selective_max_residual_draft_rows),
        *(
            ["--sr24-low-confidence-cap-by-risk"]
            if args.sr24_low_confidence_cap_by_risk
            else []
        ),
        "--sr24-early-dense-tokens",
        str(args.sr24_early_dense_tokens),
        "--sr24-target-leafs",
        str(args.sr24_target_leafs),
        "--sr24-residual-target-leafs",
        str(args.sr24_residual_target_leafs),
        "--sr24-base-only-layer-ids",
        str(args.sr24_base_only_layer_ids),
        "--sr24-base-only-layer-ids-by-leaf",
        str(args.sr24_base_only_layer_ids_by_leaf),
        "--sr24-residual-layer-ids-by-leaf",
        str(args.sr24_residual_layer_ids_by_leaf),
        "--sr24-residual-out-chunk",
        str(args.sr24_residual_out_chunk),
        *(
            ["--sr24-cache-compressed-residual-weight"]
            if args.sr24_cache_compressed_residual_weight
            else []
        ),
        *(
            ["--sr24-prewarm-compressed-residual-weight"]
            if args.sr24_prewarm_compressed_residual_weight
            else []
        ),
        *(
            []
            if args.sr24_auto_compressed_residual_fastpath
            else ["--no-sr24-auto-compressed-residual-fastpath"]
        ),
        *(
            ["--sr24-compressed-residual-triton"]
            if args.sr24_compressed_residual_triton
            else []
        ),
        "--sr24-compressed-residual-block-m",
        str(args.sr24_compressed_residual_block_m),
        "--sr24-compressed-residual-block-n",
        str(args.sr24_compressed_residual_block_n),
        "--sr24-compressed-residual-block-g",
        str(args.sr24_compressed_residual_block_g),
        "--sr24-extract-chunk-rows",
        str(args.sr24_extract_chunk_rows),
        "--sr24-residual-bucket-size",
        str(args.sr24_residual_bucket_size),
        *(
            ["--sr24-residual-bucket-scale-by-active"]
            if args.sr24_residual_bucket_scale_by_active
            else []
        ),
        "--sr24-bonus-priority",
        str(args.sr24_bonus_priority),
        "--sr24-draft-position-priority-scale",
        str(args.sr24_draft_position_priority_scale),
        *(
            ["--sr24-residual-bucket-priority"]
            if args.sr24_residual_bucket_priority
            else []
        ),
        *(
            ["--sr24-direct-position-bucket"]
            if args.sr24_direct_position_bucket
            else []
        ),
        *(["--sr24-cudagraph-bucket"] if args.sr24_cudagraph_bucket else []),
        *(["--sr24-route-bucket-rows"] if args.sr24_route_bucket_rows else []),
        *(["--sr24-route-all-residual-rows"]
          if args.sr24_route_all_residual_rows else []),
        *(["--sr24-route-all-skip-bucket"]
          if args.sr24_route_all_skip_bucket else []),
        *(["--sr24-direct-cpu-route-rows"]
          if args.sr24_direct_cpu_route_rows else []),
        *(["--sr24-route-reuse-base-output"]
          if args.sr24_route_reuse_base_output else []),
        *(["--sr24-route-contiguous-fastpath"]
          if args.sr24_route_contiguous_fastpath else []),
        "--sr24-route-dense-fallback-fraction",
        str(args.sr24_route_dense_fallback_fraction),
        *(
            ["--sr24-triton-route-assembly"]
            if args.sr24_triton_route_assembly
            else []
        ),
        *(
            ["--sr24-triton-bucket-override"]
            if args.sr24_triton_bucket_override
            else []
        ),
        *(
            ["--sr24-triton-bucket-dense-gemm"]
            if args.sr24_triton_bucket_dense_gemm
            else []
        ),
        *(
            ["--sr24-triton-bucket-scatter"]
            if args.sr24_triton_bucket_scatter
            else []
        ),
        "--sr24-triton-bucket-dense-block-m",
        str(args.sr24_triton_bucket_dense_block_m),
        "--sr24-triton-bucket-dense-block-n",
        str(args.sr24_triton_bucket_dense_block_n),
        "--sr24-triton-bucket-dense-block-k",
        str(args.sr24_triton_bucket_dense_block_k),
        *(
            ["--sr24-bucket-dense-copy"]
            if args.sr24_bucket_dense_copy
            else []
        ),
        *(
            ["--sr24-bucket-dense-copy-active-only"]
            if args.sr24_bucket_dense_copy_active_only
            else []
        ),
        *(
            ["--sr24-sort-bucket-rows"]
            if args.sr24_sort_bucket_rows
            else []
        ),
        *(
            ["--sr24-disable-runtime-stats"]
            if args.sr24_disable_runtime_stats
            else []
        ),
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
        "repeat_index",
        "repeat_count",
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
        "full_batch_output_tokens_per_second",
        "full_batch_output_tokens_prorated",
        "full_batch_window_s",
        "full_batch_completed_requests",
        "full_batch_active_threshold",
        "output_tokens_total",
        "request_latency_mean",
        "ttft_ms_mean",
        "tpot_ms_mean",
        "ttft_ms_p50",
        "ttft_ms_p90",
        "tpot_ms_p50",
        "tpot_ms_p90",
        "spec_acceptance_rate",
        "spec_acceptance_rate_pct",
        "spec_accepted_tokens",
        "spec_draft_tokens",
        "spec_estimated_steps",
        "spec_avg_selected_draft_tokens_per_step",
        "spec_avg_accepted_draft_tokens_per_step",
        "peak_gpu_memory_mib",
        "avg_gpu_util_pct",
        "peak_gpu_util_pct",
        "gpu_util_sample_count",
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
        "sr24_preset",
        "sr24_mode",
        "sr24_backend",
        "sr24_residual_backend",
        "sr24_residual_device",
        "sr24_require_gpu_residual",
        "sr24_threshold",
        "sr24_prefix_threshold",
        "sr24_all_corrected_dense_fastpath",
        "sr24_full_residual_early_dense",
        "sr24_dense_fastpath_noop",
        "sr24_selective_correct_non_draft_static",
        "sr24_selective_non_draft_policy_static",
        "sr24_selective_residual_policy_static",
        "sr24_prefix_threshold_static",
        "sr24_selective_extra_after_low_static",
        "sr24_selective_min_prefix_residual_static",
        "sr24_selective_max_residual_draft_rows_static",
        "sr24_low_confidence_cap_by_risk_static",
        "sr24_early_dense_tokens_static",
        "sr24_sync_mask_state_static",
        "sr24_static_mask_state_static",
        "sr24_static_all_residual_dense_fastpath_static",
        "sr24_default_vllm_compile_static",
        "sr24_disable_compile_cache_static",
        "sr24_compile_cache_root_static",
        "sr24_direct_cslt_linear_static",
        "sr24_base_only_dense_nonverify_static",
        "sr24_base_only_allow_compile_static",
        "sr24_base_only_dense_verify_max_rows_static",
        "sr24_static_mask_buffer",
        "sr24_batched_mask_builder",
        "sr24_batched_uniform_direct",
        "sr24_gpu_count_mask_builder",
        "sr24_gate_up_split",
        "sr24_gate_up_channel_dense_fraction",
        "sr24_gate_up_channel_strategy",
        "sr24_gate_up_channel_fused_act",
        "sr24_row_routed_mlp",
        "sr24_row_routed_down_linear",
        "sr24_row_routed_mlp_reuse_base_output",
        "sr24_row_routed_mlp_min_dense_rows",
        "sr24_row_routed_mlp_max_dense_rows",
        "sr24_row_routed_mlp_max_base_rows",
        "sr24_bonus_priority",
        "sr24_draft_position_priority_scale",
        "sr24_force_cudagraph_none_for_mixed",
        "sr24_allow_cudagraph",
        "sr24_dynamic_auto_cudagraph",
        "sr24_cudagraph_bucket",
        "sr24_cudagraph_bucket_active_hint",
        "sr24_static_mask_buffer_capacity",
        "sr24_residual_bucket_size",
        "sr24_residual_bucket_scale_by_active",
        "sr24_residual_bucket_priority",
        "sr24_direct_position_bucket",
        "sr24_route_bucket_rows",
        "sr24_route_bucket_rows_graph_static_unsafe",
        "sr24_route_all_residual_rows",
        "sr24_route_all_skip_bucket",
        "sr24_route_reuse_base_output",
        "sr24_route_contiguous_fastpath",
        "sr24_fixed_prefix_route_fastpath",
        "sr24_route_dense_fallback_fraction",
        "sr24_route_min_dense_rows",
        "sr24_route_min_base_rows",
        "sr24_route_max_dense_fraction",
        "sr24_adaptive_dense_fallback",
        "sr24_adaptive_dense_fallback_small_rows",
        "sr24_adaptive_dense_fallback_gate_up_fraction",
        "sr24_adaptive_dense_fallback_down_fraction",
        "sr24_adaptive_dense_fallback_small_down_no_residual",
        "sr24_adaptive_dense_fallback_calls",
        "sr24_adaptive_dense_fallback_rows",
        "sr24_adaptive_dense_fallback_candidate_rows",
        "sr24_triton_route_assembly",
        "sr24_triton_bucket_override",
        "sr24_triton_bucket_dense_gemm",
        "sr24_triton_bucket_scatter",
        "sr24_triton_bucket_dense_block_m",
        "sr24_triton_bucket_dense_block_n",
        "sr24_triton_bucket_dense_block_k",
        "sr24_bucket_dense_copy",
        "sr24_bucket_dense_copy_active_only",
        "sr24_sort_bucket_rows",
        "sr24_linear_hooks_enabled",
        "sr24_draft_scores_enabled",
        "sr24_runtime_stats_enabled",
        "sr24_runtime_timing_enabled_static",
        "sr24_target_leafs",
        "sr24_residual_target_leafs",
        "sr24_base_only_layer_ids",
        "sr24_base_only_layer_ids_by_leaf",
        "sr24_residual_layer_ids_by_leaf",
        "sr24_residual_out_chunk",
        "sr24_cache_compressed_residual_weight",
        "sr24_prewarm_compressed_residual_weight",
        "sr24_compressed_residual_triton",
        "sr24_compressed_residual_block_m",
        "sr24_compressed_residual_block_n",
        "sr24_compressed_residual_block_g",
        "sr24_extract_chunk_rows",
        "sr24_residual_extract_cpu_fallback_chunks",
        "sr24_residual_extract_cpu_fallback_module_count",
        "sr24_residual_backend_counts",
        "sr24_residual_device_counts",
        "sr24_residual_cpu_module_count",
        "sr24_residual_cuda_module_count",
        "sr24_compressed_residual_runtime_on_gpu",
        "sr24_compressed_residual_non_gpu_modules",
        "sr24_sync_reduced_stats",
        "sr24_sync_mask_state",
        "sr24_static_mask_state",
        "sr24_mask_state",
        "sr24_mask_state_exact",
        "sr24_stats_exact",
        "sr24_stats_interval",
        "sr24_cudagraph_mode_counts",
        "server_cudagraph_profile_counts",
        "sr24_scheduler_mask_wall_cpu_ms",
        "sr24_scheduler_mask_wall_cpu_ms_per_step",
        "sr24_scheduler_materialize_counts_wall_cpu_ms_per_step",
        "sr24_scheduler_pending_scores_pop_wall_cpu_ms_per_step",
        "sr24_scheduler_batched_mask_builder_wall_cpu_ms_per_step",
        "sr24_scheduler_request_routing_loop_wall_cpu_ms_per_step",
        "sr24_scheduler_batch_all_apply_wall_cpu_ms_per_step",
        "sr24_scheduler_mask_state_wall_cpu_ms_per_step",
        "sr24_scheduler_static_mask_copy_wall_cpu_ms_per_step",
        "sr24_scheduler_row_index_bucket_wall_cpu_ms_per_step",
        "sr24_scheduler_residual_bucket_wall_cpu_ms_per_step",
        "sr24_scheduler_mixed_row_indices_wall_cpu_ms_per_step",
        "sr24_scheduler_direct_cpu_route_rows_wall_cpu_ms_per_step",
        "sr24_selective_correct_non_draft",
        "sr24_selective_non_draft_policy",
        "sr24_selective_residual_policy",
        "sr24_runtime_prefix_threshold",
        "sr24_selective_extra_after_low",
        "sr24_selective_min_prefix_residual",
        "sr24_selective_max_residual_draft_rows",
        "sr24_low_confidence_cap_by_risk",
        "sr24_early_dense_tokens",
        "sr24_residual_draft_fraction",
        "sr24_base_only_draft_fraction",
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
        "sr24_total_scheduled_tokens",
        "sr24_missing_score_tokens",
        "sr24_steps",
        "sr24_storage_over_dense",
        "sr24_actual_weight_storage_bytes",
        "sr24_sparse_metadata_bytes",
        "sr24_mask_metadata_bytes",
        "sr24_module_count_attached",
        "sr24_mask_cache_method",
        "sr24_mask_path",
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
    median_rows = build_median_rows(rows)
    with (final_root / "median_summary.csv").open("w",
                                                  newline="",
                                                  encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in median_rows:
            writer.writerow({column: fmt(row.get(column, "")) for column in columns})
    (final_root / "median_summary.json").write_text(
        json.dumps(median_rows, indent=2) + "\n", encoding="utf-8")

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
    write_iteration_logs(final_root, args, median_rows, resume_command)

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
        f"- methods: {', '.join(args.methods)}",
        "- method notes: vLLM autoregressive, vLLM+FastDraft, "
        "vLLM+FastDraft+Smurfs dynamic K "
        f"init={args.smurfs_initial_k}/max policy={smurfs_max_policy}, "
        "vLLM+EAGLE3, and optional SR24 EAGLE3 variants",
        "- batch size means client-side streaming concurrency",
        f"- batch sizes: {', '.join(map(str, args.batch_sizes))}",
        f"- repeats per config: {args.repeats}; tables below use medians over successful repeats",
        f"- datasets: {', '.join(args.datasets)}",
        f"- prompt limit per dataset: {args.prompt_limit or 'all'}",
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
        "## Median Total Output Tokens/s",
        "",
        "| dataset | batch_size | " + " | ".join(args.methods) + " |",
        "|---|---:|" + "|".join("---:" for _ in args.methods) + "|",
    ])
    by_key = {(row["dataset"], int(row["batch_size"]), row["method"]): row
              for row in median_rows}
    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            values = []
            for method in args.methods:
                row = by_key.get((dataset, batch_size, method), {})
                value = row.get("total_output_tokens_per_second", "")
                values.append(f"{float(value):.3f}" if value not in {"", None} else "")
            lines.append(
                f"| {dataset} | {batch_size} | " + " | ".join(values) + " |"
            )
    lines.extend([
        "",
        "## Median Steady-State Output Tokens/s",
        "",
        "| dataset | batch_size | " + " | ".join(args.methods) + " |",
        "|---|---:|" + "|".join("---:" for _ in args.methods) + "|",
    ])
    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            values = []
            for method in args.methods:
                row = by_key.get((dataset, batch_size, method), {})
                value = row.get("steady_state_output_tokens_per_second", "")
                values.append(f"{float(value):.3f}" if value not in {"", None} else "")
            lines.append(
                f"| {dataset} | {batch_size} | " + " | ".join(values) + " |"
            )
    lines.extend([
        "",
        "## Median Full-Batch Output Tokens/s",
        "",
        "This table uses client-side full-concurrency generation windows only; "
        "it is an output-token overlap estimate intended to remove request-drain "
        "tail effects from fixed-total-request runs.",
        "",
        "| dataset | batch_size | " + " | ".join(args.methods) + " |",
        "|---|---:|" + "|".join("---:" for _ in args.methods) + "|",
    ])
    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            values = []
            for method in args.methods:
                row = by_key.get((dataset, batch_size, method), {})
                value = row.get("full_batch_output_tokens_per_second", "")
                values.append(f"{float(value):.3f}" if value not in {"", None} else "")
            lines.append(
                f"| {dataset} | {batch_size} | " + " | ".join(values) + " |"
            )
    speculative_rows = [
        row for row in median_rows
        if row.get("spec_acceptance_rate") not in {"", None}
    ]
    def display_cudagraph(row: dict[str, Any]) -> str:
        runtime_counts = row.get("sr24_cudagraph_mode_counts") or ""
        if runtime_counts:
            return str(runtime_counts)
        profile_counts = row.get("server_cudagraph_profile_counts") or ""
        if profile_counts:
            return f"profile:{profile_counts}"
        return ""

    if speculative_rows:
        lines.extend([
            "",
            "## Median Speculative Acceptance",
            "",
            "Accepted/selected lengths use vLLM's draft-token Prometheus "
            "counters. They do not include the verifier bonus token.",
            "",
            "| method | dataset | batch_size | acceptance % | accepted draft tokens/step | selected draft tokens/step | estimated spec steps | avg GPU util % | cudagraph modes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in speculative_rows:
            lines.append(
                "| {method} | {dataset} | {batch_size} | {accept_pct} | "
                "{accepted_per_step} | {selected_per_step} | {steps} | "
                "{gpu_util} | {cg_modes} |".format(
                    method=row.get("method", ""),
                    dataset=row.get("dataset", ""),
                    batch_size=row.get("batch_size", ""),
                    accept_pct=fmt(row.get("spec_acceptance_rate_pct")),
                    accepted_per_step=fmt(
                        row.get("spec_avg_accepted_draft_tokens_per_step")
                    ),
                    selected_per_step=fmt(
                        row.get("spec_avg_selected_draft_tokens_per_step")
                    ),
                    steps=fmt(row.get("spec_estimated_steps")),
                    gpu_util=fmt(row.get("avg_gpu_util_pct")),
                    cg_modes=display_cudagraph(row),
                )
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
    sr24_rows = [row for row in median_rows if row.get("method") in SR24_METHODS]
    if sr24_rows:
        lines.extend([
            "",
            "## Median SR24 Residual Summary",
            "",
            "| method | dataset | batch_size | residual draft fraction | residual non-draft fraction | residual draft tokens | residual non-draft tokens | bucket active/requested | bucket rows/call | total draft tokens | non-draft tokens | missing score tokens | cudagraph modes | target leafs | residual leafs | backend | storage/dense |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|",
        ])
        for row in sr24_rows:
            backend_label = (
                f"{row.get('sr24_backend')}/"
                f"{row.get('sr24_residual_backend')}"
                f"@{row.get('sr24_residual_device')}"
            )
            lines.append(
                "| {method} | {dataset} | {batch_size} | {draft_fraction} | "
                "{non_draft_fraction} | {residual_draft_tokens} | "
                "{residual_non_draft_tokens} | {bucket_actual} | "
                "{bucket_rows_per_call} | {total_draft_tokens} | "
                "{non_draft_tokens} | {missing} | {cg_modes} | "
                "{target_leafs} | {residual_leafs} | {backend} | {storage} |".format(
                    method=row.get("method", ""),
                    dataset=row.get("dataset", ""),
                    batch_size=row.get("batch_size", ""),
                    draft_fraction=fmt(row.get("sr24_residual_draft_fraction")),
                    non_draft_fraction=fmt(
                        row.get("sr24_residual_non_draft_fraction")),
                    residual_draft_tokens=fmt(
                        row.get("sr24_residual_draft_tokens")),
                    residual_non_draft_tokens=fmt(
                        row.get("sr24_residual_non_draft_tokens")),
                    bucket_actual=(
                        f"{fmt(row.get('sr24_bucket_active_rows'))}/"
                        f"{fmt(row.get('sr24_bucket_residual_requested_rows'))}"
                    ),
                    bucket_rows_per_call=(
                        f"{fmt(row.get('sr24_bucket_active_rows_per_call'))}/"
                        f"{fmt(row.get('sr24_bucket_candidate_rows_per_call'))}"
                    ),
                    total_draft_tokens=fmt(row.get("sr24_total_draft_tokens")),
                    non_draft_tokens=fmt(row.get("sr24_non_draft_tokens")),
                    missing=fmt(row.get("sr24_missing_score_tokens")),
                    cg_modes=display_cudagraph(row),
                    target_leafs=row.get("sr24_target_leafs") or "",
                    residual_leafs=row.get("sr24_residual_target_leafs") or "",
                    backend=backend_label,
                    storage=fmt(row.get("sr24_storage_over_dense")),
                ))
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


def _as_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def build_median_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("method") or not row.get("dataset") or not row.get(
                "batch_size"):
            continue
        grouped.setdefault(
            (str(row["method"]), str(row["dataset"]), int(row["batch_size"])),
            [],
        ).append(row)

    identity_keys = {
        "method",
        "dataset",
        "batch_size",
        "repeat_index",
        "repeat_count",
        "status",
        "first_error",
        "work_dir",
        "smurfs_history_file",
        "draft_length_distribution",
        "scheduler_step_round_distribution",
        "scheduler_step_changes",
        "sr24_mask_path",
    }
    median_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        ok_rows = [
            row for row in group
            if row.get("status") == "ok"
            and int(float(row.get("errored_requests") or 0)) == 0
        ]
        candidates = ok_rows or group
        tps_sorted = sorted(
            candidates,
            key=lambda row: (
                _as_float(row.get("steady_state_output_tokens_per_second"))
                if _as_float(row.get("steady_state_output_tokens_per_second"))
                is not None else -1.0
            ),
        )
        representative = tps_sorted[len(tps_sorted) // 2]
        out = dict(representative)
        out["method"], out["dataset"], out["batch_size"] = key
        out["repeat_index"] = "median"
        out["repeat_count"] = len(candidates)
        if not ok_rows:
            out["status"] = "failed"
        elif len(ok_rows) == len(group):
            out["status"] = "ok"
        else:
            out["status"] = "partial"
        if out["status"] == "ok":
            out["first_error"] = ""
        for column in set().union(*(row.keys() for row in candidates)):
            if column in identity_keys:
                continue
            values = [
                number for number in (_as_float(row.get(column))
                                      for row in candidates)
                if number is not None
            ]
            if values:
                out[column] = _median(values)
        median_rows.append(out)
    return median_rows


def git_label() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(SPECULATORS_ROOT),
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(SPECULATORS_ROOT),
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        return f"{rev}-dirty" if status else rev
    except Exception:
        return "unknown"


def write_iteration_logs(final_root: Path, args: argparse.Namespace,
                         median_rows: list[dict[str, Any]],
                         resume_command: str) -> None:
    dense_by_key = {
        (row.get("dataset"), int(row.get("batch_size"))):
        _as_float(row.get("steady_state_output_tokens_per_second"))
        for row in median_rows
        if row.get("method") == "dense_baseline"
    }
    fields = [
        "version",
        "git_diff_or_commit",
        "change_description",
        "hypothesis",
        "exact_command",
        "method",
        "dataset",
        "batch_size",
        "accuracy",
        "tps",
        "speedup",
        "residual_ratio",
        "peak_vram",
        "avg_gpu_util",
        "peak_gpu_util",
        "profiling_observation",
        "kept_or_reverted",
        "reason",
    ]
    rows: list[dict[str, Any]] = []
    current_git = git_label()
    for row in median_rows:
        batch_size = int(row.get("batch_size"))
        tps = _as_float(row.get("steady_state_output_tokens_per_second"))
        dense_tps = dense_by_key.get((row.get("dataset"), batch_size))
        speedup = (
            tps / dense_tps if tps is not None and dense_tps and dense_tps > 0
            else None)
        method = str(row.get("method") or "")
        if method in SR24_METHODS:
            observation = (
                "PyTorch SparseSemiStructuredTensorCUSPARSELT backend; "
                f"{row.get('sr24_backend')}/"
                f"{row.get('sr24_residual_backend')}"
                f"@{row.get('sr24_residual_device')}; "
                "current microbenchmarks show sparse slower than dense for "
                "tested verifier-row shapes."
            )
            hypothesis = (
                "DLM-selected draft rows above threshold use residual correction "
                "while other rows use 2:4 base path."
            )
        else:
            observation = "Baseline row for throughput comparison."
            hypothesis = "Baseline for SR24 speedup calculation."
        rows.append({
            "version": "sr24_t08_torch_sparse",
            "git_diff_or_commit": current_git,
            "change_description":
            "DLM-guided selective residual 2:4 throughput run",
            "hypothesis": hypothesis,
            "exact_command": resume_command,
            "method": method,
            "dataset": row.get("dataset"),
            "batch_size": batch_size,
            "accuracy": "",
            "tps": tps,
            "speedup": speedup,
            "residual_ratio": row.get("sr24_residual_draft_fraction", ""),
            "peak_vram": row.get("peak_gpu_memory_mib", ""),
            "avg_gpu_util": row.get("avg_gpu_util_pct", ""),
            "peak_gpu_util": row.get("peak_gpu_util_pct", ""),
            "profiling_observation": observation,
            "kept_or_reverted": "kept_for_evidence",
            "reason": row.get("first_error") or row.get("status") or "",
        })

    with (final_root / "iteration_log.csv").open("w",
                                                 newline="",
                                                 encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})

    lines = [
        "# SR24 Iteration Log",
        "",
        "| method | dataset | batch_size | TPS | speedup | residual ratio | avg GPU util | peak GPU util | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {dataset} | {batch_size} | {tps} | {speedup} | "
            "{residual} | {avg_gpu_util} | {peak_gpu_util} | {status} |".format(
                method=row.get("method", ""),
                dataset=row.get("dataset", ""),
                batch_size=row.get("batch_size", ""),
                tps=fmt(row.get("tps")),
                speedup=fmt(row.get("speedup")),
                residual=fmt(row.get("residual_ratio")),
                avg_gpu_util=fmt(row.get("avg_gpu_util")),
                peak_gpu_util=fmt(row.get("peak_gpu_util")),
                status=row.get("reason", ""),
            ))
    (final_root / "iteration_log.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")


def parse_csv_list(value: str, *, valid: set[str] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if valid is not None:
        unknown = [item for item in items if item not in valid]
        if unknown:
            raise ValueError(f"Unknown value(s): {unknown}. Valid: {sorted(valid)}")
    return items


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
        args.sr24_bucket_dense_copy = True
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
        args.sr24_bucket_dense_copy = True
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
        args.sr24_bucket_dense_copy = True
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
        # Explicit opt-in candidate for the current speed work: use the same
        # accuracy guard as gateup_cap0_dense_guard, but exercise the actual
        # bucket-copy mixed operator instead of immediately falling back to
        # dense. This remains a diagnostic probe because GSM8K-10 found a
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
        args.sr24_bucket_dense_copy = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_adaptive_dense_fallback = False
        args.sr24_adaptive_dense_fallback_gate_up_fraction = 0.05
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "lowresidual_gateup_riskcap2":
        # Current best measured gate-up-only low-residual speed/quality probe.
        # It is deliberately gate_up_proj only: adding down_proj residual
        # correction erased the speed benefit in the 2026-06-29 bucket
        # follow-up. Bucket8 was a tiny throughput improvement over bucket16,
        # while bucket4 reduced full-batch throughput; callers can still
        # override the bucket with --sr24-residual-bucket-size.
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
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = False
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "mlpall_lowconf_prefix5_tritonoverride":
        # Current all-MLP speed-target probe. This is the only measured route
        # that reached about 1.2x dense full-batch throughput, but its quality
        # evidence is still limited to GSM8K-50 plus dense-repeat stability
        # checks. Keep it separate from the safer gate-up-only preset.
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
        args.sr24_bucket_dense_copy = True
        args.sr24_adaptive_dense_fallback = True
        args.sr24_allow_cudagraph = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "criticalprefix4_bucket16_directcslt":
        # Quality/throughput candidate from the SR24 slowdown pass.  The
        # extra post-low-confidence row is part of the current paired GSM8K-50
        # clean candidate; without it the selector regresses to dense-like
        # accepted length and loses the base-only speed headroom.
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
        args.sr24_default_vllm_compile = False
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "criticalprefix_extra2_gateup_scaledbucket":
        # Trace-driven quality-first candidate from the 2026-06-29 acceptance
        # analysis. Protect a mandatory four-row accepted-prefix guard plus two
        # rows after the first low-confidence draft token. Scale the bucket by
        # active request count, and keep the per-request budget large enough
        # for the prefix guard plus bonus row.
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
    if preset == "down0_15_fixedprefix4_directcslt":
        # Current doc2 precision/speed candidate from the 2026-06-29 slowdown
        # pass. It protects the first four speculative verifier rows in early
        # down_proj layers only, leaving gate_up dense after gate_up tail
        # sparsity was shown to break the same GSM8K regression.
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
    if preset == "fixedprefix4_bucket16_directcslt":
        # Low-sync/low-score-overhead ablation for the same bucket16/direct-
        # cuSPARSELt operator shape. fixed_prefix does not consume DLM selected
        # token probabilities, so the proposer avoids the extra full-vocab
        # logsumexp path used by critical_prefix.
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
        args.sr24_route_all_skip_bucket = True
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
    elif preset == "throughput_aggressive":
        base_only_by_leaf = "gate_up_proj=31;down_proj=30-31"
    else:
        raise ValueError(f"Unknown SR24 preset: {preset}")

    # Presets are intentionally explicit. Use --sr24-preset manual for older
    # ablations that need full target leafs, exact stats, or forced eager.
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
    "sr24_selective_extra_after_low": "--sr24-selective-extra-after-low",
    "sr24_selective_min_prefix_residual": "--sr24-selective-min-prefix-residual",
    "sr24_selective_max_residual_draft_rows":
    "--sr24-selective-max-residual-draft-rows",
    "sr24_residual_bucket_size": "--sr24-residual-bucket-size",
    "sr24_residual_bucket_scale_by_active":
    "--sr24-residual-bucket-scale-by-active",
    "sr24_residual_bucket_priority": "--sr24-residual-bucket-priority",
    "sr24_bonus_priority": "--sr24-bonus-priority",
    "sr24_draft_position_priority_scale": "--sr24-draft-position-priority-scale",
    "sr24_row_routed_mlp": "--sr24-row-routed-mlp",
    "sr24_row_routed_down_linear": "--sr24-row-routed-down-linear",
    "sr24_row_routed_mlp_reuse_base_output":
    "--sr24-row-routed-mlp-reuse-base-output",
    "sr24_row_routed_mlp_min_dense_rows":
    "--sr24-row-routed-mlp-min-dense-rows",
    "sr24_row_routed_mlp_max_dense_rows":
    "--sr24-row-routed-mlp-max-dense-rows",
    "sr24_row_routed_mlp_max_base_rows":
    "--sr24-row-routed-mlp-max-base-rows",
    "sr24_bucket_dense_copy": "--sr24-bucket-dense-copy",
    "sr24_bucket_dense_copy_active_only": "--sr24-bucket-dense-copy-active-only",
    "sr24_base_only_allow_compile": "--sr24-base-only-allow-compile",
    "sr24_route_all_residual_rows": "--sr24-route-all-residual-rows",
    "sr24_route_all_skip_bucket": "--sr24-route-all-skip-bucket",
    "sr24_route_reuse_base_output": "--sr24-route-reuse-base-output",
    "sr24_route_contiguous_fastpath": "--sr24-route-contiguous-fastpath",
    "sr24_route_dense_fallback_fraction":
    "--sr24-route-dense-fallback-fraction",
    "sr24_route_min_dense_rows": "--sr24-route-min-dense-rows",
    "sr24_route_min_base_rows": "--sr24-route-min-base-rows",
    "sr24_route_max_dense_fraction": "--sr24-route-max-dense-fraction",
    "sr24_adaptive_dense_fallback": "--sr24-adaptive-dense-fallback",
    "sr24_adaptive_dense_fallback_small_rows":
    "--sr24-adaptive-dense-fallback-small-rows",
    "sr24_adaptive_dense_fallback_gate_up_fraction":
    "--sr24-adaptive-dense-fallback-gate-up-fraction",
    "sr24_adaptive_dense_fallback_down_fraction":
    "--sr24-adaptive-dense-fallback-down-fraction",
    "sr24_adaptive_dense_fallback_small_down_no_residual":
    "--sr24-adaptive-dense-fallback-small-down-no-residual",
    "sr24_triton_route_assembly": "--sr24-triton-route-assembly",
    "sr24_triton_bucket_override": "--sr24-triton-bucket-override",
    "sr24_triton_bucket_dense_gemm": "--sr24-triton-bucket-dense-gemm",
    "sr24_triton_bucket_scatter": "--sr24-triton-bucket-scatter",
    "sr24_direct_cslt_linear": "--sr24-direct-cslt-linear",
    "sr24_default_vllm_compile": "--sr24-default-vllm-compile",
    "sr24_cudagraph_bucket": "--sr24-cudagraph-bucket",
    "sr24_force_cudagraph_none_for_mixed":
    "--sr24-force-cudagraph-none-for-mixed",
    "sr24_dynamic_auto_cudagraph": "--sr24-dynamic-auto-cudagraph",
    "sr24_allow_cudagraph": "--sr24-allow-cudagraph",
    "sr24_force_eager_after_preset": "--sr24-force-eager-after-preset",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--methods",
                        default=",".join(DEFAULT_METHODS),
                        help=f"comma list from {ALL_METHODS}")
    parser.add_argument("--datasets",
                        default="math_reasoning,mtbench,gsm8k,humaneval",
                        help=f"comma list from {tuple(DATASETS)}")
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=0,
        help="Use only the first N prompts from each JSONL dataset for streaming runs.",
    )
    parser.add_argument("--min-requests-per-run", type=int, default=0)
    parser.add_argument(
        "--fixed-total-requests",
        type=int,
        default=0,
        help=(
            "Streaming-client ablation: send exactly this many requests, then "
            "stop replenishing. The steady-state tokens/s field is reported as "
            "total output tokens divided by actual fixed-batch elapsed time. "
            "Default 0 keeps the normal time-window replenishing load."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
            "Optional JSON string passed through to vLLM --compilation-config. "
            "Used for CUDA Graph/compile ablations, e.g. "
            "'{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'."
        ),
    )
    parser.add_argument(
        "--disable-chunked-prefill",
        action="store_true",
        help=(
            "Pass --no-enable-chunked-prefill to vLLM. This is a CUDA Graph "
            "coverage ablation for continuous streaming runs; default keeps "
            "vLLM's normal chunked-prefill behavior."
        ),
    )
    parser.add_argument("--fastdraft-k", type=int, default=4)
    parser.add_argument("--eagle3-k", type=int, default=4)
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
            "lowresidual_gateup_riskcap2",
            "mlpall_lowconf_prefix5_tritonoverride",
            "speed_tradeoff_down16_base",
            "criticalprefix4_bucket16_directcslt",
            "criticalprefix_extra2_gateup_scaledbucket",
            "down0_15_fixedprefix4_directcslt",
            "fixedprefix4_bucket16_directcslt",
            "accuracy_first",
            "accuracy_gate_only",
            "accuracy_down_only",
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
            "enables dynamic-auto CUDA Graph with a stable bucket; use it as "
            "a graph precision probe, not as a quality-safe path. "
            "lowresidual_gateup_riskcap2 is the current best measured "
            "gate-up-only low-residual candidate: low_confidence@0.8, "
            "prefix2, riskcap2, direct-cuSPARSELt, dynamic CUDA Graph, and "
            "bucket8 by default. "
            "mlpall_lowconf_prefix5_tritonoverride is the all-MLP speed "
            "target probe: gate_up/down all layers, low_confidence@0.6, "
            "prefix5, bucket32, dynamic CUDA Graph, and Triton bucket "
            "override; it reached about 1.2x dense full-batch throughput in "
            "the bs64/math/max128 probe but still needs broader quality gates. "
            "speed_tradeoff_down16_base is the current best speed/quality "
            "tradeoff probe: gate_up=16-31 cap1/bucket32 residual and "
            "down_proj=16-31 base-only; it is not paired-accuracy stable. "
            "criticalprefix4_bucket16_directcslt is the current "
            "compile-aligned quality-safe bucket16/direct-cuSPARSELt shape: "
            "critical_prefix@0.6, prefix4, no extra row after first low token. "
            "down0_15_fixedprefix4_directcslt protects down_proj=0-15 with "
            "fixed_prefix=4 and keeps gate_up dense; it fixed the doc2 GSM8K "
            "regression while staying close to dense throughput in the "
            "bs64/math/max128 probe. "
            "fixedprefix4_bucket16_directcslt uses the same operator scope "
            "but fixed_prefix=4, avoiding DLM selected-probability score "
            "collection and serving as the low-sync/low-score overhead "
            "ablation. "
            "accuracy_first is now the conservative static tail candidate: "
            "qkv/o exact densefastpath plus base-only gate_up=31 with "
            "--sr24-gate-up-split up_sparse. accuracy_gate_only now attaches "
            "only the fully fused gate_up=31 sparse tail; accuracy_down_only "
            "keeps only down=31 as a negative/diagnostic ablation. "
            "throughput_aggressive uses gate_up=31,down=30-31."
        ),
    )
    parser.add_argument("--sr24-threshold", type=float, default=0.8)
    parser.add_argument("--sr24-backend",
                        choices=["dense_zero", "prototype", "torch_sparse"],
                        default="torch_sparse")
    parser.add_argument("--sr24-residual-backend",
                        choices=["compressed_dense", "torch_sparse", "dense_rows"],
                        default="dense_rows",
                        help=(
                            "Residual correction backend for SR24. dense_rows "
                            "is the default quality-safe backend because it "
                            "uses exact dense Linear outputs for corrected "
                            "rows. compressed_dense is GPU-resident but split "
                            "sparse-base/residual bf16 GEMMs are not currently "
                            "dense-equivalent; use it only for storage/operator "
                            "ablations."
                        ))
    parser.add_argument(
        "--sr24-residual-backend-by-leaf",
        default="",
        help=(
            "Optional SR24 per-leaf residual backend override, for example "
            "'gate_up_proj=torch_sparse;down_proj=dense_rows'. Leaves not "
            "listed use --sr24-residual-backend."
        ),
    )
    parser.add_argument("--sr24-residual-device",
                        choices=["auto", "cpu", "cuda"],
                        default="auto",
                        help=(
                            "Storage device for compressed SR24 residual "
                            "values. auto uses GPU-resident residual values; "
                            "use cpu explicitly only as a memory fallback or "
                            "CPU-transfer ablation."
                        ))
    parser.add_argument(
        "--sr24-require-gpu-residual",
        action="store_true",
        help=(
            "Fail SR24 model attach if compressed_dense residual values are "
            "not GPU-resident. Use this for all_corrected_24/speclink_t08 "
            "performance diagnostics."
        ))
    parser.add_argument("--sr24-mask-path",
                        type=Path,
                        default=SR24_DEFAULT_MASK)
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
            "Write an SR24 component breakdown JSON with scheduler mask-build "
            "time, bucket/topk timing, reduced routing counts, and CUDA graph "
            "mode counts. Profiling-only; it adds overhead and may reduce "
            "throughput. Use --sr24-breakdown-linear and "
            "--sr24-breakdown-exact-routing only for focused ablations."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-linear",
        action="store_true",
        help=(
            "Also time SR24 Linear internals such as sparse base, dense "
            "correction, gather, and scatter. Use with eager/no-compile "
            "profiles; this is disabled by default so torch.compile can still "
            "capture graph-on serving paths."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-exact-routing",
        action="store_true",
        help=(
            "Synchronize GPU scalars to report exact residual/base routing "
            "counts and bucket fill ratio. This intentionally measures the "
            "CPU-sync overhead as a separate ablation."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-gpu-counts",
        action="store_true",
        help=(
            "Accumulate residual/base routing counts and bucket active rows on "
            "GPU, then read them only when the breakdown flushes. This is a "
            "low-CPU-sync routing diagnostic for clean-ish ablations."
        ),
    )
    parser.add_argument(
        "--sr24-cudagraph-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record generic CUDA Graph FULL/NONE mode counts without enabling "
            "SR24 breakdown or Linear timing. Use this for clean serving rows "
            "that also use --sr24-disable-runtime-stats."
        ),
    )
    parser.add_argument(
        "--sr24-breakdown-interval",
        type=int,
        default=2000,
        help="Flush SR24 breakdown after roughly this many CUDA timing events.",
    )
    parser.add_argument(
        "--sr24-reduce-cpu-sync",
        action="store_true",
        help=(
            "Enable the SR24 CPU-sync reduction ablation. This skips exact "
            "hot-path GPU scalar stats and uses a masked full-row residual "
            "path to reduce host synchronization."
        ),
    )
    parser.add_argument(
        "--sr24-sync-mask-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --sr24-reduce-cpu-sync, synchronize once per verify step to "
            "classify the residual mask as all_residual/no_residual/mixed. "
            "This avoids per-Linear mask.all()/mask.any() synchronizations "
            "while preserving all-residual fastpaths. Use "
            "--no-sr24-sync-mask-state for the no-sync ablation."
        ),
    )
    parser.add_argument(
        "--sr24-static-mask-state",
        choices=["auto", "all_residual", "no_residual", "mixed"],
        default="auto",
        help=(
            "Static SR24 residual-mask state override for CPU-sync ablations. "
            "auto preserves normal behavior. all_residual/no_residual skip "
            "runtime mask-state reduction and force that state. mixed skips "
            "the reduction but keeps the mixed-mask path."
        ),
    )
    parser.add_argument(
        "--sr24-static-all-residual-dense-fastpath",
        action="store_true",
        help=(
            "When --sr24-static-mask-state=all_residual, keep the original "
            "dense Linear weights and bypass SR24 weight rewriting. This is a "
            "diagnostic upper-bound ablation for SR24 dispatch/storage "
            "overhead, not a sparse speedup mode."
        ),
    )
    parser.add_argument(
        "--sr24-default-vllm-compile",
        action="store_true",
        help=(
            "Do not force eager or an SR24-specific compilation config for "
            "SR24 methods. This opt-in ablation lets vLLM use its default "
            "compile/CUDA-graph settings and is expected to fail for some "
            "real sparse paths; keep it off for normal runs."
        ),
    )
    parser.add_argument(
        "--sr24-force-eager-after-preset",
        action="store_true",
        help=(
            "Diagnostic override applied after --sr24-preset. It disables "
            "SR24 CUDA Graph launch and forces eager execution even when the "
            "selected preset normally enables graph-safe serving. Use this "
            "for SR24 CUDA-event breakdown runs, where timing events cannot "
            "be captured safely inside CUDA Graph replay/capture."
        ),
    )
    parser.add_argument(
        "--sr24-direct-cslt-linear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For torch_sparse SR24 weights, call torch._cslt_sparse_mm on the "
            "packed semi-structured tensor directly instead of going through "
            "F.linear/__torch_dispatch__. Default is off because the current "
            "direct call is slower in the bs64 attention-only ablation."
        ),
    )
    parser.add_argument(
        "--sr24-auto-direct-cslt-base-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically use the direct cuSPARSELt Linear path for "
            "base_only_24 torch_sparse runs. The current bs64/math/K8 probe "
            "shows this helps base-only throughput, while speclink_t08 and "
            "all_corrected_24 remain controlled by --sr24-direct-cslt-linear."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-allow-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow base_only_24 torch_sparse runs to use the SR24 graph-only "
            "compilation config. This avoids vLLM's default Inductor compile "
            "path while recovering CUDA Graph coverage for the sparse base "
            "upper-bound row. Use --no-sr24-base-only-allow-compile for the "
            "older eager-safe diagnostic."
        ),
    )
    parser.add_argument(
        "--sr24-base-only-dense-nonverify",
        action="store_true",
        help=(
            "Base-only diagnostic ablation: keep a dense copy and use it for "
            "non-speculative/non-verify forwards, so base_only_24 sparse work "
            "is isolated to speculative verifier rows. This costs dense-copy "
            "memory and is not a deployable storage-saving mode."
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
        "--sr24-static-mask-buffer",
        action="store_true",
        help=(
            "Use a reusable GPU bool buffer for the selective residual mask. "
            "This is required for the experimental speclink_t08 CUDA Graph "
            "ablation and should be paired with --sr24-reduce-cpu-sync."
        ),
    )
    parser.add_argument(
        "--sr24-batched-mask-builder",
        action="store_true",
        help=(
            "Experimental selective critical_prefix+bonus path: build draft "
            "residual mask rows with one batched Triton kernel instead of "
            "launching per-request cumsum/where/min fragments. This is a "
            "diagnostic speed ablation and falls back unless the SR24 policy "
            "matches the supported shape. Supported policies are "
            "critical_prefix, all_if_any_low, low_confidence, and "
            "high_confidence when early_dense_tokens is disabled; "
            "min_prefix_residual is supported by the batched path."
        ),
    )
    parser.add_argument(
        "--sr24-batched-uniform-direct",
        action="store_true",
        help=(
            "SR24 CPU-sync ablation: when every request has the same K and "
            "the score rows map directly to request rows, launch the batched "
            "mask builder without copying per-request starts/counts through "
            "CPU-side staging tensors. Requires --sr24-batched-mask-builder."
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
        "--sr24-gpu-count-mask-builder",
        action="store_true",
        help=(
            "SR24 CPU-sync ablation: let the batched mask builder consume "
            "scheduler count tensors already on GPU when the score layout is "
            "compatible. This is off by default because older diagnostics "
            "showed mixed results."
        ),
    )
    parser.add_argument(
        "--sr24-mask-buffer-capacity",
        type=int,
        default=16384,
        help="Capacity in tokens for the reusable SR24 residual-mask buffer.",
    )
    parser.add_argument(
        "--sr24-gate-up-split",
        choices=["none", "up_sparse", "gate_sparse", "channel_pair"],
        default="none",
        help=(
            "SR24 base-only gate_up_proj ablation. up_sparse keeps the gate "
            "half dense and sparsifies only the up half; gate_sparse does the "
            "opposite. channel_pair keeps selected intermediate gate/up channel "
            "pairs dense, sparsifies the remaining pairs, and requires dense "
            "down_proj. Default none preserves the fused gate_up_proj behavior."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-channel-dense-fraction",
        type=float,
        default=0.125,
        help=(
            "For --sr24-gate-up-split channel_pair, fraction of intermediate "
            "gate/up channel pairs kept dense."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-channel-strategy",
        choices=["norm", "front"],
        default="norm",
        help=(
            "For --sr24-gate-up-split channel_pair, choose dense channel pairs "
            "by paired row norm or by the front contiguous channel block."
        ),
    )
    parser.add_argument(
        "--sr24-gate-up-channel-fused-act",
        action="store_true",
        help=(
            "For --sr24-gate-up-split channel_pair, reassemble grouped "
            "[gate, up] activations and use vLLM's fused SiluAndMul. This is "
            "an explicit ablation because it adds extra concat work."
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
            "Experimental row-routed MLP variant: compute the sparse-base MLP "
            "for all rows and overwrite the selected dense rows, avoiding "
            "per-step base-row complement construction. Default off."
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
        "--sr24-dynamic-auto-cudagraph",
        action="store_true",
        help=(
            "Experimental SR24 ablation. When paired with "
            "--sr24-allow-cudagraph, --no-sr24-force-cudagraph-none-for-mixed, "
            "--sr24-static-mask-state=auto, and graph-safe static mask/bucket "
            "buffers, launch dynamic auto/mixed speclink_t08 without "
            "--enforce-eager. Default is off because dynamic graph correctness "
            "must be verified by replay/quality checks before final claims."
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
            "auto/mixed masks stay eager for correctness. all_corrected_24 "
            "compressed_dense can use CUDA Graph only when the GPU residual "
            "weight is cached, prewarmed, and materialized as one full tensor."
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
            "Keep SR24 Linear hooks attached, but when a step/module is known "
            "to be all-residual and a dense weight is available, run the dense "
            "Linear directly instead of sparse base plus dense correction. "
            "This is the optimized all-residual hook path; the default stays "
            "off so --no-sr24-all-corrected-dense-fastpath remains a true "
            "operator-ablation of sparse base plus correction."
        ),
    )
    parser.add_argument(
        "--sr24-selective-correct-non-draft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For speclink_t08/selective SR24, correct non-draft scheduled "
            "tokens with the residual path and gate only draft-token rows by "
            "DLM confidence. Disabling this reproduces the older base-only "
            "prefill/non-draft behavior."
        ),
    )
    parser.add_argument(
        "--sr24-selective-non-draft-policy",
        choices=["auto", "all", "none", "bonus", "predicted_full_accept"],
        default="auto",
        help=(
            "Selective SR24 non-draft/bonus-row residual policy. auto preserves "
            "--sr24-selective-correct-non-draft behavior; all corrects every "
            "non-draft row; none corrects none; bonus corrects only the "
            "speculative bonus row; predicted_full_accept corrects the bonus "
            "row only when all draft scores are present and above the "
            "threshold."
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
            "--sr24-selective-min-prefix-residual draft rows plus the bonus row "
            "and does not require draft confidence scores."
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
            "Use this for attention-only or MLP-only SR24 ablations."
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
            "Diagnostic compressed_dense optimization: after the first GPU "
            "materialization of a compressed residual weight, cache the dense "
            "GPU residual tensor and reuse it on later Linear calls. This "
            "trades memory for avoiding repeated materialization."
        ),
    )
    parser.add_argument(
        "--sr24-prewarm-compressed-residual-weight",
        action="store_true",
        help=(
            "When compressed residual weight caching is enabled, materialize "
            "the dense GPU residual tensor during model attach instead of the "
            "first Linear call. This is a CUDA Graph ablation for "
            "all_corrected_24 compressed_dense and increases peak load-time "
            "memory pressure."
        ),
    )
    parser.add_argument(
        "--sr24-auto-compressed-residual-fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For all_corrected_24 with torch_sparse/compressed_dense and the "
            "dense fastpath disabled, automatically use the best current "
            "compressed_dense residual path: cache the GPU residual weight, "
            "prewarm it at model attach, and set residual_out_chunk=0 so CUDA "
            "Graph capture can be used. This is not the best overall exact "
            "all_corrected_24 path; current measurements favor "
            "torch_sparse residual with direct cuSPARSELt. Use "
            "--no-sr24-auto-compressed-residual-fastpath to reproduce the "
            "older chunked materialization ablation."
        ),
    )
    parser.add_argument(
        "--sr24-compressed-residual-triton",
        action="store_true",
        help=(
            "Experimental compressed_dense residual path: compute the residual "
            "matmul directly from GPU-resident compressed 2:4 values with a "
            "Triton kernel instead of materializing a dense residual weight."
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
            "residual values during model loading. Smaller values reduce "
            "transient GPU memory; runtime residual storage still follows "
            "--sr24-residual-device."
        ),
    )
    parser.add_argument(
        "--sr24-residual-bucket-size",
        type=int,
        default=0,
        help=(
            "Reduce-sync mixed-mask ablation for torch_sparse residuals. When "
            ">0 and the linear input has more rows than this value, compute "
            "residual correction for a fixed-size GPU top-k bucket from the "
            "residual mask instead of all rows. Default 0 disables it. If true "
            "residual rows exceed the bucket, this is approximate and must be "
            "paired with quality checks."
        ),
    )
    parser.add_argument(
        "--sr24-residual-bucket-scale-by-active",
        action="store_true",
        help=(
            "Treat --sr24-residual-bucket-size as a per-active-request "
            "budget when the scheduler builds residual buckets. Default keeps "
            "the historical global bucket size."
        ),
    )
    parser.add_argument(
        "--sr24-residual-bucket-priority",
        action="store_true",
        help=(
            "When residual bucket size is positive, choose capped residual rows "
            "by SR24 priority scores instead of the bool mask's arbitrary top-k."
        ),
    )
    parser.add_argument(
        "--sr24-bonus-priority",
        type=float,
        default=4.0,
        help=(
            "Priority score assigned to the speculative bonus/non-draft row "
            "when --sr24-residual-bucket-priority is active. Default 4.0 "
            "preserves the old behavior; lower values reserve capped buckets "
            "for draft verification rows first."
        ),
    )
    parser.add_argument(
        "--sr24-draft-position-priority-scale",
        type=float,
        default=0.0,
        help=(
            "Additional per-draft-position priority scale for capped residual "
            "buckets. Default 0 preserves old behavior; positive values make "
            "earlier draft positions dominate global top-k selection."
        ),
    )
    parser.add_argument(
        "--sr24-direct-position-bucket",
        action="store_true",
        help=(
            "Build capped SR24 residual buckets directly in draft-position "
            "order instead of using the global priority top-k path. This is an "
            "experimental scheduler-overhead ablation and is off by default."
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
            "For route_all_residual_rows, skip residual bucket construction "
            "when the route-all Linear path does not consume the bucket. This "
            "is an opt-in scheduler-overhead ablation."
        ),
    )
    parser.add_argument(
        "--sr24-direct-cpu-route-rows",
        action="store_true",
        help=(
            "For prefix_confidence + bonus + route_all_residual_rows, build "
            "exact residual/base row lists from the small draft-score tensor on "
            "CPU instead of materializing them with GPU nonzero. Experimental "
            "scheduler-overhead ablation; off by default."
        ),
    )
    parser.add_argument(
        "--sr24-route-reuse-base-output",
        action="store_true",
        help=(
            "For torch_sparse + dense_rows mixed-mask ablations, reuse the "
            "already-computed full sparse base output and run dense Linear only "
            "for residual rows. This avoids full dense+where without splitting "
            "the base sparse GEMM."
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
        "--sr24-adaptive-dense-fallback-small-rows",
        type=int,
        default=0,
        help=(
            "Row-count cutoff for SR24 adaptive dense fallback small-shape "
            "rules. Default 0 disables the small-row rule; use 128 only to "
            "reproduce the older diagnostic ablation."
        ),
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
        "--sr24-dense-fallback-nonuniform",
        action="store_true",
        help=(
            "For SR24 selective runs, use the all-residual dense verifier "
            "fastpath on non-uniform scheduled-token steps. These refill/mixed "
            "steps cannot use FULL CUDA Graphs anyway, so this is a conservative "
            "throughput ablation that corrects extra rows instead of leaving "
            "them base-only."
        ),
    )
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
            "For torch_sparse + dense_rows residual bucket ablations, run the "
            "sparse base on all rows and use a Triton kernel to overwrite only "
            "active bucket rows with dense output. This avoids base-row gather, "
            "delta compute, and index_add_ in the bucket correction path."
        ),
    )
    parser.add_argument(
        "--sr24-triton-bucket-dense-gemm",
        action="store_true",
        help=(
            "Experimental fused correction prototype for torch_sparse + "
            "dense_rows residual buckets: compute active bucket-row dense "
            "GEMM in Triton and scatter directly into the sparse base output, "
            "skipping the intermediate dense_output tensor. Off by default."
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
            "GEMM, overwrite every bucket row with the dense output via "
            "index_copy_ instead of gathering base rows, computing "
            "dense-minus-base, and index_add_. Padded bucket rows become dense, "
            "which is quality-conservative but may change acceptance."
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
        "--sr24-sort-bucket-rows",
        action="store_true",
        help=(
            "Sort capped SR24 residual bucket rows by row id before dense "
            "gather/index_copy correction. This is an off-by-default memory "
            "coalescing ablation for graph-safe bucketed dense_rows paths."
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
        "--sr24-disable-runtime-stats",
        action="store_true",
        help=(
            "CPU/Python overhead ablation: disable SR24 runtime verify summary "
            "and SR24 verify counter updates. Static model-attach stats are still "
            "written; pass --sr24-cudagraph-stats if clean CUDA Graph mode "
            "counts are still needed."
        ),
    )
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
    sr24_preset_overrides = capture_sr24_preset_overrides(args, sys.argv[1:])

    args.methods = parse_csv_list(args.methods, valid=set(ALL_METHODS))
    args.datasets = parse_csv_list(args.datasets, valid=set(DATASETS))
    args.batch_sizes = parse_int_list(args.batch_sizes)
    args.sr24_gate_up_channel_dense_fraction = min(
        max(float(args.sr24_gate_up_channel_dense_fraction), 0.0), 1.0)
    args.sr24_route_min_dense_rows = max(0, int(args.sr24_route_min_dense_rows))
    args.sr24_route_min_base_rows = max(0, int(args.sr24_route_min_base_rows))
    if args.smoke_only:
        args.methods = [
            "vllm_ar",
            "vllm_fastdraft",
            "smurfs_fastdraft",
            "vllm_eagle3",
        ]
        args.datasets = ["math_reasoning"]
        args.batch_sizes = [1]
        args.repeats = 1
        args.prompt_limit = 8
        args.min_requests_per_run = 0
        args.fixed_total_requests = 0
        args.max_tokens = 64
        args.max_model_len = 512
        args.max_num_batched_tokens = 1024
        args.gpu_memory_utilization = 0.72
        args.warmup_s = 2.0
        args.measurement_s = 5.0
        args.cooldown_s = 1.0
        args.request_timeout_s = 600.0
    apply_sr24_preset(args)
    restore_sr24_preset_overrides(args, sr24_preset_overrides)
    if args.sr24_force_eager_after_preset:
        args.sr24_default_vllm_compile = False
        args.sr24_allow_cudagraph = False
        args.sr24_dynamic_auto_cudagraph = False
        args.sr24_force_cudagraph_none_for_mixed = True
        args.sr24_cudagraph_bucket = False
        args.vllm_compilation_config = ""

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
    args.sr24_mask_path = args.sr24_mask_path.resolve()
    args.repeats = max(1, int(args.repeats))
    args.prompt_limit = max(0, int(args.prompt_limit))
    args.fixed_total_requests = max(0, int(args.fixed_total_requests))
    args.sr24_mask_buffer_capacity = max(
        0, int(args.sr24_mask_buffer_capacity))
    args.sr24_extract_chunk_rows = max(1, int(args.sr24_extract_chunk_rows))
    args.sr24_compressed_residual_block_m = max(
        1, int(args.sr24_compressed_residual_block_m))
    args.sr24_compressed_residual_block_n = max(
        1, int(args.sr24_compressed_residual_block_n))
    args.sr24_compressed_residual_block_g = max(
        1, int(args.sr24_compressed_residual_block_g))
    args.sr24_selective_max_residual_draft_rows = max(
        0, int(args.sr24_selective_max_residual_draft_rows))
    args.sr24_triton_bucket_dense_block_m = max(
        1, int(args.sr24_triton_bucket_dense_block_m))
    args.sr24_triton_bucket_dense_block_n = max(
        1, int(args.sr24_triton_bucket_dense_block_n))
    args.sr24_triton_bucket_dense_block_k = max(
        1, int(args.sr24_triton_bucket_dense_block_k))
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
            "time", "status", "method", "batch_size", "repeat", "dataset",
            "message"
        ])
    rows = read_existing_rows(args.final_root) if args.resume else []
    completed = {
        (
            row["method"],
            row["dataset"],
            int(row["batch_size"]),
            int(row.get("repeat_index") or 1),
        )
        for row in rows
        if row.get("method") and row.get("dataset") and row.get("batch_size")
        and str(row.get("repeat_index", "1")).isdigit()
    }
    if rows:
        write_outputs(args.final_root, rows, args, dataset_counts)
    try:
        for method in args.methods:
            for batch_size in args.batch_sizes:
                for repeat_index in range(1, args.repeats + 1):
                    pending_datasets = [
                        dataset for dataset in args.datasets
                        if (method, dataset, batch_size, repeat_index)
                        not in completed
                    ]
                    if not pending_datasets:
                        append_status(args.work_root, [
                            timestamp(), "server_skip_completed", method,
                            batch_size, repeat_index, "", ""
                        ])
                        continue
                    server_dir = (
                        args.work_root / method / f"bs{batch_size}" /
                        f"rep{repeat_index}")
                    append_status(args.work_root, [
                        timestamp(), "server_start", method, batch_size,
                        repeat_index, "", ""
                    ])
                    process: subprocess.Popen[Any] | None = None
                    try:
                        process, port = start_server(args, method, batch_size,
                                                     server_dir, env)
                    except Exception as exc:  # noqa: BLE001
                        append_status(args.work_root, [
                            timestamp(), "server_failed", method, batch_size,
                            repeat_index, "", repr(exc)
                        ])
                        for dataset_name in pending_datasets:
                            case_dir = server_dir / dataset_name
                            case_dir.mkdir(parents=True, exist_ok=True)
                            rows.append(
                                failure_row(method, dataset_name, batch_size,
                                            case_dir, args, repr(exc),
                                            repeat_index))
                            completed.add(
                                (method, dataset_name, batch_size,
                                 repeat_index))
                        write_outputs(args.final_root, rows, args,
                                      dataset_counts)
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
                            sr24_log = (
                                server_dir / "speclink_sr24_events.jsonl"
                                if method in SR24_METHODS else None)
                            sr24_before = (
                                latest_sr24_summary(sr24_log)
                                if sr24_log is not None else {})
                            cudagraph_stats = (
                                server_dir / "cudagraph_stats.jsonl"
                                if (
                                    args.sr24_breakdown
                                    or args.sr24_cudagraph_stats
                                ) else None)
                            cudagraph_before = (
                                latest_cudagraph_summary(cudagraph_stats)
                                if cudagraph_stats is not None else {})
                            append_status(args.work_root, [
                                timestamp(), "run_start", method, batch_size,
                                repeat_index, dataset_name, ""
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
                                             json.dumps(event, sort_keys=True) +
                                             "\n" for event in events),
                                         encoding="utf-8",
                                     )
                                    smurfs_summary = smurfs_dynamic_summary(
                                        events)
                                sr24_summary = None
                                if method in SR24_METHODS and sr24_log is not None:
                                    sr24_after = latest_sr24_summary(sr24_log)
                                    sr24_summary = {
                                        **sr24_static_summary(
                                            server_dir, args, method, batch_size),
                                        **sr24_delta_summary(
                                            sr24_before, sr24_after),
                                    }
                                    sr24_summary = {
                                        **sr24_summary,
                                        **server_cudagraph_profile_summary(
                                            server_dir / "server.log"),
                                    }
                                if cudagraph_stats is not None:
                                    cudagraph_after = latest_cudagraph_summary(
                                        cudagraph_stats)
                                    graph_summary = cudagraph_delta_summary(
                                        cudagraph_before, cudagraph_after)
                                    if graph_summary:
                                        sr24_summary = {
                                            **(sr24_summary or {}),
                                            **graph_summary,
                                        }
                                row = row_for_case(method, dataset_name,
                                                   batch_size, metrics,
                                                   case_dir, args,
                                                   smurfs_summary,
                                                   sr24_summary, repeat_index)
                                run_status = "run_ok"
                                if metrics.get("status") != "ok":
                                    run_status = "run_error"
                                message = metrics.get(
                                    "steady_state_output_tokens_per_second")
                            except Exception as exc:  # noqa: BLE001
                                row = failure_row(
                                    method, dataset_name, batch_size, case_dir,
                                    args, repr(exc), repeat_index)
                                run_status = "run_failed"
                                message = repr(exc)
                            rows.append(row)
                            completed.add(
                                (method, dataset_name, batch_size,
                                 repeat_index))
                            write_outputs(args.final_root, rows, args,
                                          dataset_counts)
                            append_status(args.work_root, [
                                timestamp(), run_status, method, batch_size,
                                repeat_index, dataset_name, message
                            ])
                    finally:
                        stop_process(process)
                        append_status(args.work_root, [
                            timestamp(), "server_stop", method, batch_size,
                            repeat_index, "", ""
                        ])
    finally:
        write_outputs(args.final_root, rows, args, dataset_counts)
    print(f"final_root={args.final_root}")
    print(f"work_root={args.work_root}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
