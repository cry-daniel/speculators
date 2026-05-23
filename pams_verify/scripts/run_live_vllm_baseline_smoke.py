#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, register_experiment, write_json


SPEC_BIN = Path("/ACALAB/stu1/miniconda3/envs/spec/bin")


def run_text(cmd: list[str], env: dict[str, str], timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def wait_health(port: int, proc: subprocess.Popen, env: dict[str, str], timeout: int) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{port}/health"
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}"
        code, out = run_text(["curl", "--noproxy", "*", "-sf", health_url], env, timeout=5)
        if code == 0:
            return True, "ready"
        last_error = out.strip()
        time.sleep(5)
    return False, f"health timeout: {last_error}"


def stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def gpu_snapshot(env: dict[str, str]) -> dict[str, Any]:
    code, out = run_text(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        env,
        timeout=10,
    )
    return {"exit_code": code, "raw": out.strip()}


def run_one(
    args: argparse.Namespace,
    method: str,
    server_env: dict[str, str],
    client_env: dict[str, str],
) -> dict[str, Any]:
    exp = EXPERIMENTS / "01_dense_baselines"
    run_dir = exp / "raw" / f"live_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log = run_dir / "vllm_server.log"
    bench_log = run_dir / "vllm_bench_serve.log"
    result_file = run_dir / f"{method}.json"

    serve_cmd = [
        "vllm",
        "serve",
        args.target_model,
        "--served-model-name",
        args.target_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--trust-remote-code",
        "--enforce-eager",
    ]
    if method == "ngram_4":
        serve_cmd.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "ngram",
                        "num_speculative_tokens": 4,
                        "prompt_lookup_min": 4,
                        "prompt_lookup_max": 4,
                    }
                ),
            ]
        )
    elif method == "ngram_8":
        serve_cmd.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "ngram",
                        "num_speculative_tokens": 8,
                        "prompt_lookup_min": 8,
                        "prompt_lookup_max": 8,
                    }
                ),
            ]
        )

    bench_cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        args.target_model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        "inf",
        "--num-warmups",
        str(args.num_warmups),
        "--save-result",
        "--result-dir",
        str(run_dir),
        "--result-filename",
        result_file.name,
        "--percentile-metrics",
        "ttft,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--disable-tqdm",
        "--trust-remote-code",
    ]

    result: dict[str, Any] = {
        "method": method,
        "serve_cmd": serve_cmd,
        "bench_cmd": bench_cmd,
        "server_log": str(server_log),
        "bench_log": str(bench_log),
        "result_file": str(result_file),
        "status": "unknown",
        "gpu_before": gpu_snapshot(client_env),
    }

    with server_log.open("w", encoding="utf-8") as server_handle:
        proc = subprocess.Popen(
            serve_cmd,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
            env=server_env,
            cwd=str(ROOT),
            text=True,
        )
        result["server_pid"] = proc.pid
        try:
            ready, ready_msg = wait_health(args.port, proc, client_env, args.server_timeout)
            result["server_ready"] = ready
            result["server_ready_message"] = ready_msg
            if not ready:
                result["status"] = "server_start_failed"
                return result

            code, bench_out = run_text(bench_cmd, client_env, timeout=args.bench_timeout)
            bench_log.write_text(bench_out, encoding="utf-8")
            result["bench_exit_code"] = code
            result["bench_output_tail"] = "\n".join(bench_out.splitlines()[-80:])
            result["gpu_after_bench"] = gpu_snapshot(client_env)
            if code == 0 and result_file.exists():
                result["status"] = "completed"
                try:
                    result["benchmark_json"] = json.loads(result_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    result["benchmark_json_parse_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result["status"] = "bench_failed"
            return result
        finally:
            stop_proc(proc)
            result["gpu_after_stop"] = gpu_snapshot(client_env)


def load_completed_raw_runs() -> list[dict[str, Any]]:
    exp = EXPERIMENTS / "01_dense_baselines"
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((exp / "raw").glob("live_*")):
        method = run_dir.name.removeprefix("live_")
        result_file = run_dir / f"{method}.json"
        if not result_file.exists():
            continue
        try:
            bench = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            bench = {"parse_error": f"{type(exc).__name__}: {exc}"}
        runs.append(
            {
                "method": method,
                "status": "completed",
                "result_file": str(result_file),
                "server_log": str(run_dir / "vllm_server.log"),
                "bench_log": str(run_dir / "vllm_bench_serve.log"),
                "benchmark_json": bench,
            }
        )
    return runs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--methods", nargs="+", default=["dense_no_spec"])
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--bench-timeout", type=int, default=300)
    parser.add_argument("--refresh-only", action="store_true")
    args = parser.parse_args()

    server_env = os.environ.copy()
    server_env["PATH"] = f"{SPEC_BIN}:{server_env.get('PATH', '')}"
    server_env["NO_PROXY"] = f"{server_env.get('NO_PROXY', '')},localhost,127.0.0.1"
    server_env["no_proxy"] = f"{server_env.get('no_proxy', '')},localhost,127.0.0.1"

    client_env = server_env.copy()

    results = []
    if not args.refresh_only:
        for offset, method in enumerate(args.methods):
            args.port = args.port + offset
            results.append(run_one(args, method, server_env, client_env))

    merged_by_method = {run["method"]: run for run in load_completed_raw_runs()}
    for run in results:
        merged_by_method[run["method"]] = run
    merged_results = list(merged_by_method.values())

    exp = EXPERIMENTS / "01_dense_baselines"
    write_json(exp / "parsed" / "live_vllm_baseline_smoke.json", {"runs": merged_results})
    completed = [run for run in merged_results if run["status"] == "completed"]
    failed = [run for run in merged_results if run["status"] != "completed"]
    summary = [
        "# Live vLLM Baseline Smoke",
        "",
        "This smoke run starts a real vLLM OpenAI-compatible server and measures it with `vllm bench serve` on a short random workload.",
        "",
        f"- Target model: `{args.target_model}`",
        f"- max_model_len: `{args.max_model_len}`",
        f"- max_num_seqs: `{args.max_num_seqs}`",
        f"- prompts: `{args.num_prompts}`",
        f"- completed methods: `{[run['method'] for run in completed]}`",
        f"- failed methods: `{[(run['method'], run['status']) for run in failed]}`",
    ]
    for run in completed:
        bench = run.get("benchmark_json", {})
        summary.extend(
            [
                "",
                f"## {run['method']}",
                "",
                f"- request throughput: `{bench.get('request_throughput', 'unknown')}`",
                f"- output throughput: `{bench.get('output_throughput', 'unknown')}`",
                f"- mean TTFT ms: `{bench.get('mean_ttft_ms', 'unknown')}`",
                f"- mean ITL ms: `{bench.get('mean_itl_ms', 'unknown')}`",
                f"- result file: `{run['result_file']}`",
            ]
        )
    register_experiment(
        "01_dense_baselines",
        config=vars(args),
        command="python scripts/run_live_vllm_baseline_smoke.py --target-model Qwen/Qwen3-8B --methods dense_no_spec --max-model-len 2048 --num-prompts 4",
        status="completed_live_smoke_partial" if completed else "live_smoke_failed",
        summary="\n".join(summary),
        stdout=json.dumps({"runs": merged_results}, indent=2, sort_keys=True),
        stderr="",
        metadata_extra={"live_smoke_runs": merged_results},
        model_name=args.target_model,
        model_dtype=args.dtype,
        max_model_len=args.max_model_len,
        vllm_launch_args=next((run.get("serve_cmd", []) for run in merged_results if run.get("serve_cmd")), []),
    )
    print(json.dumps({"completed": len(completed), "failed": len(failed), "results": merged_results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
