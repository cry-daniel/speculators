#!/usr/bin/env python3
"""Run the focused SpecLink-CV Qwen3 math throughput check.

This is intentionally narrow: Qwen3-8B and math_reasoning, with configurable
K, prefix length, and closed-loop serving concurrency. It compares normal
EAGLE3 one-shot verification with the current
`cv_half_async_staged_simple` path, the prefix-FULL-cudagraph variant, and the
throughput-oriented `cv_wavefront_staged` scheduler using steady-state
saturated output tokens/s.
Intermediate outputs are written under eval-guidellm/temp by default.
The CV methods intentionally keep vLLM's own async scheduling disabled; the
`async` in the method names refers to SpecLink-CV's prefix queue, not vLLM's
fixed-K async scheduler.

Typical use from the repo root:

    conda run -n spec python examples/evaluate/eval-guidellm/scripts/run_speclink_cv_qwen_math.py \
      --python-bin /ACALAB/stu1/miniconda3/envs/spec/bin/python \
      --methods eagle3_oneshot,cv_half_async_staged_fullgraph \
      --k 16 --force-prefix-len 8 --batch-size 64 --server-max-num-seqs 64

The script starts/stops one vLLM server per method, then writes:

- `summary.csv`: machine-readable throughput and CV counters.
- `summary.md`: short human-readable comparison.
- `runs/*/`: server logs, steady-state client logs, and raw JSON results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parents[2]
MODELS_ROOT = REPO_ROOT.parent / "models"

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_SPECULATOR = MODELS_ROOT / "qwen3-8b-eagle3-speculator"
DEFAULT_DATASET = EVAL_DIR / "data/math_reasoning.jsonl"
DEFAULT_METHODS = (
    "eagle3_oneshot,cv_half_async_staged_simple,"
    "cv_half_async_staged_fullgraph,cv_wavefront_staged"
)
CV_METHODS = {
    "cv_half_async_staged_simple",
    "cv_half_async_staged_fullgraph",
    "cv_wavefront_staged",
}


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def local_no_proxy_value(env: dict[str, str]) -> str:
    values = [
        "localhost",
        "127.0.0.1",
        "::1",
    ]
    existing = env.get("NO_PROXY") or env.get("no_proxy") or ""
    for item in existing.split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return ",".join(values)


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def wait_for_health(
    *, port: int, proc: subprocess.Popen[Any], log_path: Path, timeout_s: float
) -> None:
    url = f"http://127.0.0.1:{port}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited with code {proc.returncode}\n"
                f"--- server log tail ---\n{tail(log_path)}"
            )
        try:
            with opener.open(url, timeout=2.0) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = repr(exc)
        time.sleep(1.0)
    raise RuntimeError(
        f"timed out waiting for {url}: {last_error}\n"
        f"--- server log tail ---\n{tail(log_path)}"
    )


def post_vllm_profile_endpoint(port: int, endpoint: str) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}/{endpoint.lstrip('/')}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, method="POST")
    try:
        with opener.open(request, timeout=10.0) as response:
            return response.status < 500, f"HTTP {response.status}"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def terminate_server(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def prepend_pythonpath(env: dict[str, str]) -> None:
    entries = [str(REPO_ROOT / "vllm"), str(REPO_ROOT)]
    old = env.get("PYTHONPATH")
    if old:
        entries.append(old)
    env["PYTHONPATH"] = os.pathsep.join(entries)


def method_env(method: str, args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    no_proxy = local_no_proxy_value(env)
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy
    prepend_pythonpath(env)

    is_cv = method in CV_METHODS
    is_wavefront = method == "cv_wavefront_staged"
    # Wavefront batching is only useful when the verifier chunks can enter the
    # h+1 FULL CUDA graph path; otherwise it trades queue wait for PIECEWISE
    # forwards and can look artificially bad.
    use_prefix_full_cudagraph = (
        method
        in {
            "cv_half_async_staged_fullgraph",
            "cv_wavefront_staged",
        }
        or (is_cv and args.prefix_full_cudagraph)
    )
    cv_log_jsonl = "" if args.disable_cv_jsonl else str(
        run_dir / "speclink_cv_events.jsonl"
    )
    cv_profile_jsonl = "" if args.disable_cv_jsonl else str(
        run_dir / "speclink_cv_profile.jsonl"
    )
    env.update(
        {
            "SPECLINK_CV_ENABLE": "1" if is_cv else "0",
            "SPECLINK_CV_ASYNC_QUEUE": "1" if is_cv else "0",
            "SPECLINK_CV_STAGED_DRAFTING": "1" if is_cv else "0",
            "SPECLINK_CV_ALLOW_BATCHED_PREFIX": (
                "1" if is_cv and args.allow_batched_prefix else "0"
            ),
            "SPECLINK_CV_ALLOW_BATCHED_SUFFIX": (
                "1" if is_cv and args.allow_batched_suffix else "0"
            ),
            "SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING": "1",
            "SPECLINK_CV_FORCE_PREFIX_LEN": str(args.force_prefix_len),
            "SPECLINK_CV_PREFIX_WAVEFRONT": "1" if is_wavefront else "0",
            "SPECLINK_CV_PREFIX_WAVE_WAIT_FOR_MIN": (
                "1" if is_wavefront and args.prefix_wave_wait_for_min else "0"
            ),
            "SPECLINK_CV_PREFIX_WAVE_EXCLUSIVE": (
                "1" if is_wavefront and args.prefix_wave_exclusive else "0"
            ),
            "SPECLINK_CV_PREFIX_FULL_CUDAGRAPH": (
                "1" if use_prefix_full_cudagraph else "0"
            ),
            "SPECLINK_CV_PREFIX_WAVE_MIN_SEQS": str(
                args.prefix_wave_min_seqs
            ),
            "SPECLINK_CV_PREFIX_WAVE_MAX_WAIT_MS": str(
                args.prefix_wave_max_wait_ms
            ),
            "SPECLINK_CV_LOG_JSONL": cv_log_jsonl,
            "SPECLINK_CV_PROFILE_JSONL": cv_profile_jsonl,
            "SPECLINK_CV_LOG_MAX_EVENTS": str(args.log_max_events),
            "SPECLINK_CV_PROFILE_MAX_EVENTS": str(args.profile_max_events),
        }
    )
    return env


def server_command(args: argparse.Namespace, method: str) -> list[str]:
    cmd = [
        args.python_bin,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.base_model,
        "--seed",
        str(args.seed),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(args.port),
        "--max-num-seqs",
        str(args.server_max_num_seqs or args.batch_size),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    if args.disable_uvicorn_access_log:
        cmd.append("--disable-uvicorn-access-log")
    if args.disable_log_stats:
        cmd.append("--disable-log-stats")
    if args.disable_vllm_async_scheduling:
        cmd.append("--no-async-scheduling")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.nsys_profile:
        cmd.append("--profiler-config.profiler")
        cmd.append("cuda")
    if method in {"eagle3_oneshot", *CV_METHODS}:
        spec_config = {
            "model": args.speculator_model,
            "num_speculative_tokens": args.k,
            "method": "eagle3",
            "max_model_len": args.max_model_len,
        }
        cmd.extend(["--speculative-config", json.dumps(spec_config)])
    else:
        raise ValueError(f"unsupported method: {method}")
    return cmd


def maybe_wrap_nsys(
    args: argparse.Namespace, method: str, run_dir: Path, server_cmd: list[str]
) -> list[str]:
    if not args.nsys_profile:
        return server_cmd
    nsys_dir = run_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    output_base = nsys_dir / f"vllm_guidellm_profile_{method}"
    return [
        args.nsys_bin,
        "profile",
        "-o",
        str(output_base),
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt,cublas",
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node",
        "--gpu-metrics-devices=cuda-visible",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat",
        "--sample=none",
        "--stats=false",
        *server_cmd,
    ]


def benchmark_command(
    args: argparse.Namespace, method: str, run_dir: Path
) -> list[str]:
    run_label = f"qwen3_8b_math_k{args.k}_bs{args.batch_size}_{method}"
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "tools/speclink_cv/steady_state_openai_benchmark.py"),
        "--target",
        f"http://127.0.0.1:{args.port}",
        "--model",
        args.base_model,
        "--dataset",
        str(args.dataset),
        "--run-label",
        run_label,
        "--request-type",
        args.request_type,
        "--concurrency",
        str(args.batch_size),
        "--warmup-s",
        str(args.warmup_s),
        "--measurement-s",
        str(args.measurement_s),
        "--cooldown-s",
        str(args.cooldown_s),
        "--bucket-s",
        str(args.bucket_s),
        "--max-prompts",
        str(args.max_prompts),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.client_timeout_s),
        "--max-errors",
        str(args.max_errors),
        "--output-dir",
        str(run_dir),
    ]
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if not args.ignore_eos:
        cmd.append("--no-ignore-eos")
    return cmd


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def profile_summary(path: Path) -> dict[str, Any]:
    rows = parse_jsonl(path)
    # The profile stream is intentionally capped. These counters are quick
    # diagnostics for the sampled prefix/suffix decisions, while throughput is
    # read from the steady-state benchmark JSON.
    prefix_results = [
        row
        for row in rows
        if row.get("event") == "verify_chunk_result"
        and row.get("phase") == "prefix"
    ]
    suffix_results = [
        row
        for row in rows
        if row.get("event") == "verify_chunk_result"
        and row.get("phase") == "suffix"
    ]
    prefix_scheduled = [
        row
        for row in rows
        if row.get("event") in {"verify_chunk_scheduled", "verify_chunk_queued"}
        and row.get("phase") == "prefix"
    ]
    selected_h = [
        fnum(row.get("chunk_len", row.get("scheduled_chunk_len")))
        for row in prefix_scheduled
    ]
    skipped_suffix = sum(fnum(row.get("skipped_suffix_tokens")) for row in prefix_results)
    suffix_len_total = sum(fnum(row.get("suffix_len")) for row in prefix_results)
    accepted_prefix = [
        fnum(row.get("num_accepted"))
        for row in prefix_results
        if row.get("num_accepted") is not None
    ]
    async_steps = [row for row in rows if row.get("event") == "async_queue_step"]
    dispatch_counts = [
        fnum(row.get("dispatch_count"))
        for row in async_steps
        if fnum(row.get("dispatch_count")) > 0
    ]
    wavefront_waits = [
        row for row in async_steps if row.get("reason") == "wavefront_wait"
    ]
    same_shape_fill = [
        fnum(row.get("ready_same_shape_seqs"))
        for row in async_steps
        if row.get("ready_same_shape_seqs") is not None
    ]
    dispatch_tokens = [
        fnum(row.get("dispatch_tokens"))
        for row in async_steps
        if fnum(row.get("dispatch_tokens")) > 0
    ]
    prefix_wait_ms = [
        fnum(row.get("queue_wait_ms"))
        for row in rows
        if row.get("event") == "verify_chunk_dequeued"
        and row.get("phase") == "prefix"
    ]
    forward_plans = [
        row for row in rows if row.get("event") == "model_forward_plan"
    ]
    phase_forward_counts: dict[str, int] = {}
    phase_cudagraph_counts: dict[str, int] = {}
    phase_uniform_query_lens: dict[str, int] = {}
    for row in forward_plans:
        phase = str(row.get("phase") or "unknown")
        mode = str(row.get("cudagraph_mode") or "unknown")
        phase_forward_counts[phase] = phase_forward_counts.get(phase, 0) + 1
        key = f"{phase}:{mode}"
        phase_cudagraph_counts[key] = phase_cudagraph_counts.get(key, 0) + 1
        uniform_query_len = row.get("uniform_decode_query_len")
        if uniform_query_len is not None:
            query_key = f"{phase}:{uniform_query_len}"
            phase_uniform_query_lens[query_key] = (
                phase_uniform_query_lens.get(query_key, 0) + 1
            )
    prefix_forward_tokens = [
        fnum(row.get("num_tokens_unpadded"))
        for row in forward_plans
        if row.get("phase") == "prefix" and fnum(row.get("num_tokens_unpadded")) > 0
    ]
    prefix_full_forward_tokens = [
        fnum(row.get("num_tokens_unpadded"))
        for row in forward_plans
        if row.get("phase") == "prefix"
        and row.get("cudagraph_mode") == "FULL"
        and fnum(row.get("num_tokens_unpadded")) > 0
    ]
    sorted_dispatch_counts = sorted(dispatch_counts)
    mid = len(sorted_dispatch_counts) // 2
    if not sorted_dispatch_counts:
        prefix_wave_p50 = ""
    elif len(sorted_dispatch_counts) % 2:
        prefix_wave_p50 = sorted_dispatch_counts[mid]
    else:
        prefix_wave_p50 = (
            sorted_dispatch_counts[mid - 1] + sorted_dispatch_counts[mid]
        ) / 2
    seq_budgets = [
        fnum(row.get("seq_budget"))
        for row in async_steps
        if fnum(row.get("dispatch_count")) > 0 and fnum(row.get("seq_budget")) > 0
    ]
    full_waves = 0
    for row in async_steps:
        dispatch_count = fnum(row.get("dispatch_count"))
        seq_budget = fnum(row.get("seq_budget"))
        if dispatch_count > 0 and seq_budget > 0 and dispatch_count >= seq_budget:
            full_waves += 1
    return {
        "profile_events": len(rows),
        "prefix_result_count": len(prefix_results),
        "suffix_result_count": len(suffix_results),
        "prefix_scheduled_count": len(prefix_scheduled),
        "selected_h_avg": (
            sum(selected_h) / len(selected_h) if selected_h else ""
        ),
        "prefix_accepted_tokens_avg": (
            sum(accepted_prefix) / len(accepted_prefix) if accepted_prefix else ""
        ),
        "skipped_suffix_tokens": skipped_suffix,
        "skipped_suffix_ratio": (
            skipped_suffix / suffix_len_total if suffix_len_total else ""
        ),
        "suffix_scheduled_ratio": (
            len(suffix_results) / len(prefix_results) if prefix_results else ""
        ),
        "prefix_wave_avg_seqs": (
            sum(dispatch_counts) / len(dispatch_counts) if dispatch_counts else ""
        ),
        "prefix_wave_avg_draft_tokens": (
            sum(dispatch_tokens) / len(dispatch_tokens) if dispatch_tokens else ""
        ),
        "prefix_forward_avg_total_tokens": (
            sum(prefix_forward_tokens) / len(prefix_forward_tokens)
            if prefix_forward_tokens
            else ""
        ),
        "prefix_full_forward_avg_total_tokens": (
            sum(prefix_full_forward_tokens) / len(prefix_full_forward_tokens)
            if prefix_full_forward_tokens
            else ""
        ),
        "prefix_wave_p50_seqs": prefix_wave_p50,
        "prefix_wave_full_ratio": (
            full_waves / len(dispatch_counts) if dispatch_counts else ""
        ),
        "prefix_wavefront_wait_count": len(wavefront_waits),
        "prefix_wave_ready_same_shape_avg": (
            sum(same_shape_fill) / len(same_shape_fill)
            if same_shape_fill
            else ""
        ),
        "prefix_wait_ms_avg": (
            sum(prefix_wait_ms) / len(prefix_wait_ms) if prefix_wait_ms else ""
        ),
        "phase_forward_counts": json.dumps(
            phase_forward_counts, sort_keys=True
        ),
        "phase_cudagraph_counts": json.dumps(
            phase_cudagraph_counts, sort_keys=True
        ),
        "phase_uniform_query_lens": json.dumps(
            phase_uniform_query_lens, sort_keys=True
        ),
    }


def run_parse_logs(args: argparse.Namespace, run_dir: Path) -> None:
    parser = EVAL_DIR / "scripts/parse_logs.py"
    if not parser.exists():
        return
    with (run_dir / "acceptance_analysis.txt").open("w", encoding="utf-8") as out:
        subprocess.run(
            [args.python_bin, str(parser), str(run_dir / "vllm_server.log")],
            cwd=EVAL_DIR,
            text=True,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )


def read_result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "steady_state_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_command(path: Path, env: dict[str, str], commands: list[list[str]]) -> None:
    cv_env = {
        key: value
        for key, value in env.items()
        if key.startswith("SPECLINK_CV_") or key in {"PYTHONPATH", "NO_PROXY"}
    }
    if "VLLM_WORKER_MULTIPROC_METHOD" in env:
        cv_env["VLLM_WORKER_MULTIPROC_METHOD"] = env[
            "VLLM_WORKER_MULTIPROC_METHOD"
        ]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {EVAL_DIR}"]
    for key, value in sorted(cv_env.items()):
        lines.append(f"export {key}={json.dumps(value)}")
    for command in commands:
        lines.append(subprocess.list2cmdline(command))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def run_case(args: argparse.Namespace, method: str, output_root: Path) -> dict[str, Any]:
    run_name = f"qwen3_8b_math_k{args.k}_bs{args.batch_size}_{method}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = method_env(method, args, run_dir)
    if args.nsys_profile:
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    server_cmd = server_command(args, method)
    launch_cmd = maybe_wrap_nsys(args, method, run_dir, server_cmd)
    bench_cmd = benchmark_command(args, method, run_dir)
    write_run_command(run_dir / "run_command.sh", env, [launch_cmd, bench_cmd])
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["method"] = method
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.resume and (run_dir / "steady_state_results.json").exists():
        result = read_result(run_dir)
        return summarize_run(method, run_dir, result, "ok_resumed")

    if args.dry_run:
        return {
            "method": method,
            "status": "planned",
            "output_dir": str(run_dir),
            "server_command": subprocess.list2cmdline(server_cmd),
            "benchmark_command": subprocess.list2cmdline(bench_cmd),
        }

    if port_is_open(args.port):
        return {
            "method": method,
            "status": "failed",
            "error": f"port {args.port} is already in use",
            "output_dir": str(run_dir),
        }

    proc: subprocess.Popen[Any] | None = None
    server_log = run_dir / "vllm_server.log"
    benchmark_log = run_dir / "steady_state_output.log"
    try:
        with server_log.open("w", encoding="utf-8") as server_out:
            proc = subprocess.Popen(
                launch_cmd,
                cwd=EVAL_DIR,
                env=env,
                stdout=server_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        wait_for_health(
            port=args.port,
            proc=proc,
            log_path=server_log,
            timeout_s=args.health_timeout_s,
        )
        if args.nsys_profile:
            ok, message = post_vllm_profile_endpoint(args.port, "start_profile")
            (run_dir / "nsys_start_profile.txt").write_text(
                f"{ok} {message}\n", encoding="utf-8"
            )
            if not ok:
                return {
                    "method": method,
                    "status": "failed",
                    "error": f"start_profile failed: {message}",
                    "output_dir": str(run_dir),
                }
        with benchmark_log.open("w", encoding="utf-8") as benchmark_out:
            benchmark_rc = subprocess.run(
                bench_cmd,
                cwd=EVAL_DIR,
                env=env,
                stdout=benchmark_out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            ).returncode
        if args.nsys_profile:
            ok, message = post_vllm_profile_endpoint(args.port, "stop_profile")
            (run_dir / "nsys_stop_profile.txt").write_text(
                f"{ok} {message}\n", encoding="utf-8"
            )
        if benchmark_rc != 0:
            return {
                "method": method,
                "status": "failed",
                "returncode": benchmark_rc,
                "error": tail(benchmark_log),
                "output_dir": str(run_dir),
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "method": method,
            "status": "failed",
            "error": repr(exc),
            "output_dir": str(run_dir),
        }
    finally:
        terminate_server(proc)
        time.sleep(2.0)

    run_parse_logs(args, run_dir)
    if args.nsys_profile:
        run_nsys_stats(args, run_dir, method)
    result = read_result(run_dir)
    return summarize_run(method, run_dir, result, "ok")


def run_nsys_stats(args: argparse.Namespace, run_dir: Path, method: str) -> None:
    nsys_dir = run_dir / "nsys"
    rep_path = nsys_dir / f"vllm_guidellm_profile_{method}.nsys-rep"
    if not rep_path.exists():
        matches = sorted(nsys_dir.glob(f"vllm_guidellm_profile_{method}*.nsys-rep"))
        if not matches:
            return
        rep_path = matches[-1]
    reports = (
        "cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,osrt_sum,"
        "nvtx_gpu_proj_sum"
    )
    with (nsys_dir / "nsys_stats.txt").open("w", encoding="utf-8") as out:
        subprocess.run(
            [
                args.nsys_bin,
                "stats",
                "--force-export=true",
                "--force-overwrite=true",
                "--report",
                reports,
                str(rep_path),
            ],
            cwd=nsys_dir,
            text=True,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )


def summarize_run(
    method: str, run_dir: Path, result: dict[str, Any], status: str
) -> dict[str, Any]:
    row = {
        "method": method,
        "status": status,
        "output_dir": str(run_dir),
        "throughput": result.get("output_tokens_per_second", ""),
        "measurement_output_tokens": result.get("measurement_output_tokens", ""),
        "counting_mode": result.get("counting_mode", ""),
        "requests_completed": result.get("requests_completed", ""),
        "requests_errored": result.get("requests_errored", ""),
        "unfinished_worker_threads": result.get("unfinished_worker_threads", ""),
        "workload_hash": result.get("workload_hash", ""),
    }
    row.update(profile_summary(run_dir / "speclink_cv_profile.jsonl"))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotate_relative_metrics(rows: list[dict[str, Any]]) -> None:
    baseline = next(
        (row for row in rows if row.get("method") == "eagle3_oneshot"),
        None,
    )
    base_tps = fnum(baseline.get("throughput")) if baseline else 0.0
    for row in rows:
        tps = fnum(row.get("throughput"))
        speedup = tps / base_tps if base_tps and tps else 0.0
        row["speedup_vs_eagle3"] = speedup if speedup else ""
        skipped = fnum(row.get("skipped_suffix_ratio"))
        row["realized_skip_efficiency"] = (
            (speedup - 1.0) / skipped if speedup and skipped else ""
        )


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# SpecLink-CV Focused Throughput",
        "",
        f"- Model: `{args.base_model}`",
        f"- Dataset: `{args.dataset}`",
        f"- K: `{args.k}`",
        f"- Concurrency: `{args.batch_size}`",
        f"- Measurement: steady-state saturated output tokens/s",
        f"- Warmup / measurement / cooldown: `{args.warmup_s}s / {args.measurement_s}s / {args.cooldown_s}s`",
        "",
        "| Method | Status | Output tok/s | Speedup | Skip ratio | Wave seqs | Wave draft toks | FULL forward toks | Prefix wait ms | Realized skip eff. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {tps} | {speedup} | {skip} | {wave} | {draft_toks} | {forward_toks} | {wait} | {eff} |".format(
                method=row.get("method", ""),
                status=row.get("status", ""),
                tps=row.get("throughput", ""),
                speedup=row.get("speedup_vs_eagle3", ""),
                skip=row.get("skipped_suffix_ratio", ""),
                wave=row.get("prefix_wave_avg_seqs", ""),
                draft_toks=row.get("prefix_wave_avg_draft_tokens", ""),
                forward_toks=row.get("prefix_full_forward_avg_total_tokens", ""),
                wait=row.get("prefix_wait_ms_avg", ""),
                eff=row.get("realized_skip_efficiency", ""),
            )
        )
    for row in rows:
        if row.get("method") == "eagle3_oneshot":
            continue
        speedup = fnum(row.get("speedup_vs_eagle3"))
        if speedup:
            lines.append(
                f"`{row.get('method')} / eagle3_oneshot = {speedup:.3f}x`."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EVAL_DIR / f"temp/speclink_cv_qwen_math_{timestamp}",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--speculator-model", default=str(DEFAULT_SPECULATOR))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--server-max-num-seqs",
        type=int,
        default=0,
        help=(
            "Override vLLM --max-num-seqs. Defaults to --batch-size. "
            "Use a larger value only for padding/occupancy experiments."
        ),
    )
    parser.add_argument("--force-prefix-len", type=int, default=0)
    parser.add_argument("--prefix-wave-min-seqs", type=int, default=0)
    parser.add_argument("--prefix-wave-max-wait-ms", type=float, default=2.0)
    parser.add_argument(
        "--prefix-wave-wait-for-min",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prefix-wave-exclusive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prefix-full-cudagraph", action="store_true")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--measurement-s", type=float, default=20.0)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--bucket-s", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--request-type", default="chat_completions")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--health-timeout-s", type=float, default=600.0)
    parser.add_argument("--client-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-errors", type=int, default=64)
    parser.add_argument(
        "--disable-uvicorn-access-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable per-request uvicorn access logs during saturated "
            "throughput runs. This reduces host-side IO noise; use "
            "--no-disable-uvicorn-access-log when debugging HTTP traffic."
        ),
    )
    parser.add_argument(
        "--disable-log-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable periodic vLLM stats logs in focused throughput probes. "
            "Use --no-disable-log-stats if the server-side metrics lines are "
            "needed for diagnosis."
        ),
    )
    parser.add_argument("--profile-max-events", type=int, default=400)
    parser.add_argument("--log-max-events", type=int, default=400)
    parser.add_argument(
        "--disable-cv-jsonl",
        action="store_true",
        help=(
            "Disable SpecLink-CV event/profile JSONL in the vLLM hot path. "
            "Use for headline throughput; summary CV counters will be blank."
        ),
    )
    parser.add_argument("--allow-batched-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-batched-suffix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep vLLM's fixed-K async scheduler disabled. Required for the "
            "current CV prefix/suffix scheduler path."
        ),
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--nsys-profile", action="store_true")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.dataset = args.dataset.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    methods = split_csv(args.methods)
    if not args.disable_vllm_async_scheduling and any(
        method in CV_METHODS for method in methods
    ):
        raise SystemExit(
            "CV methods require --disable-vllm-async-scheduling. "
            "SPECLINK_CV_ASYNC_QUEUE is the CV prefix queue; vLLM async "
            "scheduling does not currently run the CV prefix/suffix state "
            "machine."
        )
    rows: list[dict[str, Any]] = []
    for method in methods:
        print(f"[INFO] Running {method}")
        row = run_case(args, method, args.output_root)
        rows.append(row)
        print(
            "[INFO] {method}: status={status} throughput={throughput}".format(
                method=method,
                status=row.get("status"),
                throughput=row.get("throughput", ""),
            )
        )
    annotate_relative_metrics(rows)
    write_csv(args.output_root / "summary.csv", rows)
    write_report(args.output_root / "summary.md", rows, args)
    if args.nsys_profile and not args.dry_run:
        summary_tool = REPO_ROOT / "tools/speclink_cv/nsys_profile_summary.py"
        subprocess.run(
            [
                args.python_bin,
                str(summary_tool),
                "--run-root",
                str(args.output_root),
            ],
            cwd=REPO_ROOT,
            text=True,
            check=False,
        )
    print(f"[INFO] Output root: {args.output_root}")
    if args.dry_run:
        return 0
    return 0 if all(str(row.get("status", "")).startswith("ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
