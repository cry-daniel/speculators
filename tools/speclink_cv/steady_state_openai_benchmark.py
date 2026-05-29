#!/usr/bin/env python3
"""Closed-loop steady-state OpenAI-compatible serving benchmark.

This client is for saturated throughput measurement.  It keeps exactly
``concurrency`` request workers active during warmup and the fixed measurement
window, stops launching new work at the end of the window, and excludes drain
tokens from throughput.

The preferred counting path is streaming ``usage.completion_tokens`` with
``continuous_usage_stats``.  That lets the client count token deltas by the time
the server emits each stream chunk instead of assigning a whole request to its
completion time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


PromptValue = str | list[int]


@dataclass(frozen=True)
class WorkItem:
    sequence_index: int
    dataset_index: int
    row: dict[str, Any]
    prompt: PromptValue


class CounterState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.next_sequence = 0
        self.measurement_tokens = 0
        self.warmup_tokens = 0
        self.cooldown_tokens = 0
        self.total_stream_tokens = 0
        self.requests_started = 0
        self.requests_completed = 0
        self.requests_errored = 0
        self.continuous_usage_chunks = 0
        self.final_usage_fallback_requests = 0
        self.bucket_tokens: dict[int, int] = {}
        self.request_rows: list[dict[str, Any]] = []

    def next_work(self, rows: list[dict[str, Any]]) -> WorkItem:
        with self.lock:
            sequence_index = self.next_sequence
            self.next_sequence += 1
            self.requests_started += 1
        dataset_index = sequence_index % len(rows)
        row = rows[dataset_index]
        return WorkItem(
            sequence_index=sequence_index,
            dataset_index=dataset_index,
            row=row,
            prompt=build_prompt(row, dataset_index),
        )

    def add_tokens(
        self,
        *,
        tokens: int,
        now: float,
        warmup_end: float,
        measurement_end: float,
        measurement_s: float,
        bucket_s: float,
    ) -> None:
        if tokens <= 0:
            return
        with self.lock:
            self.total_stream_tokens += tokens
            if now < warmup_end:
                self.warmup_tokens += tokens
            elif now < measurement_end:
                self.measurement_tokens += tokens
                bucket = int(min(measurement_s - 1e-9, now - warmup_end) // bucket_s)
                self.bucket_tokens[bucket] = self.bucket_tokens.get(bucket, 0) + tokens
            else:
                self.cooldown_tokens += tokens

    def add_request_row(self, row: dict[str, Any]) -> None:
        with self.lock:
            if row.get("ok"):
                self.requests_completed += 1
            else:
                self.requests_errored += 1
            self.request_rows.append(row)


def read_jsonl(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_prompts > 0 and len(rows) >= max_prompts:
                break
    if not rows:
        raise SystemExit(f"no rows loaded from {path}")
    return rows


def build_prompt(row: dict[str, Any], dataset_index: int) -> PromptValue:
    token_ids = row.get("prompt_token_ids")
    if isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids):
        return [int(item) for item in token_ids]
    prompt = row.get("prompt") or row.get("question") or row.get("text")
    if isinstance(prompt, str):
        return prompt
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        parts: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            if isinstance(turn, str):
                text = turn
            elif isinstance(turn, dict):
                text = str(turn.get("content") or turn.get("text") or "")
            else:
                text = str(turn)
            parts.append(f"User turn {idx}:\n{text}")
        return "\n\n".join(parts) + "\n\nAssistant:"
    raise ValueError(f"dataset row {dataset_index} has no supported prompt field")


def workload_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for idx, row in enumerate(rows):
        prompt = build_prompt(row, idx)
        digest.update(str(idx).encode())
        digest.update(b"\0")
        if isinstance(prompt, list):
            digest.update(json.dumps(prompt, separators=(",", ":")).encode())
        else:
            digest.update(prompt.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def make_payload(
    *,
    request_type: str,
    model: str,
    prompt: PromptValue,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int,
    ignore_eos: bool,
    request_id: str,
    continuous_usage: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "request_id": request_id,
        "stream": True,
    }
    if top_k is not None:
        body["top_k"] = top_k
    if continuous_usage:
        body["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }
    if request_type == "completions":
        body["prompt"] = prompt
    elif request_type == "chat_completions":
        if isinstance(prompt, list):
            raise ValueError("chat_completions does not support prompt_token_ids")
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"unsupported request type: {request_type}")
    return body


def endpoint_url(target: str, request_type: str) -> str:
    base = target.rstrip("/")
    if request_type == "completions":
        return f"{base}/v1/completions"
    if request_type == "chat_completions":
        return f"{base}/v1/chat/completions"
    raise ValueError(f"unsupported request type: {request_type}")


def iter_sse_json(response: requests.Response) -> Iterable[dict[str, Any]]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        if not payload:
            continue
        yield json.loads(payload)


def stream_request(
    *,
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    state: CounterState,
    warmup_end: float,
    measurement_end: float,
    measurement_s: float,
    bucket_s: float,
    allow_final_usage_fallback: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    previous_completion_tokens = 0
    final_completion_tokens: int | None = None
    continuous_chunks = 0
    status_code = 0
    error = ""
    output_parts: list[str] = []
    try:
        with session.post(url, json=payload, stream=True, timeout=timeout) as resp:
            status_code = resp.status_code
            resp.raise_for_status()
            for chunk in iter_sse_json(resp):
                now = time.perf_counter()
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    current = usage.get("completion_tokens")
                    if isinstance(current, int):
                        final_completion_tokens = current
                        delta = max(0, current - previous_completion_tokens)
                        previous_completion_tokens = max(
                            previous_completion_tokens, current
                        )
                        if delta:
                            continuous_chunks += 1
                            state.add_tokens(
                                tokens=delta,
                                now=now,
                                warmup_end=warmup_end,
                                measurement_end=measurement_end,
                                measurement_s=measurement_s,
                                bucket_s=bucket_s,
                            )
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    text_delta = choice.get("text")
                    if text_delta is None:
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            text_delta = delta.get("content")
                    if text_delta:
                        output_parts.append(str(text_delta))
                    finish = choice.get("finish_reason")
                    if finish is not None and final_completion_tokens is None:
                        final_completion_tokens = previous_completion_tokens
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        error = repr(exc)

    ended = time.perf_counter()
    if ok and continuous_chunks == 0 and allow_final_usage_fallback:
        tokens = final_completion_tokens or 0
        if warmup_end <= ended < measurement_end:
            state.add_tokens(
                tokens=tokens,
                now=ended,
                warmup_end=warmup_end,
                measurement_end=measurement_end,
                measurement_s=measurement_s,
                bucket_s=bucket_s,
            )
            with state.lock:
                state.final_usage_fallback_requests += 1

    with state.lock:
        state.continuous_usage_chunks += continuous_chunks

    return {
        "ok": ok,
        "request_id": payload.get("request_id", ""),
        "status_code": status_code,
        "error": error,
        "start_perf_s": started,
        "end_perf_s": ended,
        "latency_s": ended - started,
        "completion_tokens": final_completion_tokens,
        "continuous_usage_chunks": continuous_chunks,
        "output": "".join(output_parts),
        "request_args": json.dumps(
            {"body": payload}, sort_keys=True, ensure_ascii=False
        ),
    }


def worker(
    *,
    worker_id: int,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    state: CounterState,
    start_perf: float,
    warmup_end: float,
    measurement_end: float,
    url: str,
    stop_queue: queue.Queue[str],
) -> None:
    session = requests.Session()
    while time.perf_counter() < measurement_end:
        item = state.next_work(rows)
        request_id = (
            f"steady-{args.run_label}-w{worker_id}-s{item.sequence_index:08d}"
        )
        seed = args.seed + item.sequence_index
        try:
            payload = make_payload(
                request_type=args.request_type,
                model=args.model,
                prompt=item.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=seed,
                ignore_eos=args.ignore_eos,
                request_id=request_id,
                continuous_usage=not args.no_continuous_usage,
            )
            result = stream_request(
                session=session,
                url=url,
                payload=payload,
                timeout=args.timeout,
                state=state,
                warmup_end=warmup_end,
                measurement_end=measurement_end,
                measurement_s=args.measurement_s,
                bucket_s=args.bucket_s,
                allow_final_usage_fallback=args.allow_final_usage_fallback,
            )
            result.update(
                {
                    "worker_id": worker_id,
                    "sequence_index": item.sequence_index,
                    "dataset_index": item.dataset_index,
                    "question_id": item.row.get("question_id", ""),
                    "phase_at_start": (
                        "warmup"
                        if result["start_perf_s"] < warmup_end
                        else "measurement"
                        if result["start_perf_s"] < measurement_end
                        else "cooldown"
                    ),
                    "phase_at_end": (
                        "warmup"
                        if result["end_perf_s"] < warmup_end
                        else "measurement"
                        if result["end_perf_s"] < measurement_end
                        else "cooldown"
                    ),
                    "relative_start_s": result["start_perf_s"] - start_perf,
                    "relative_end_s": result["end_perf_s"] - start_perf,
                }
            )
            state.add_request_row(result)
        except Exception as exc:  # noqa: BLE001
            state.add_request_row(
                {
                    "ok": False,
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "sequence_index": item.sequence_index,
                    "dataset_index": item.dataset_index,
                    "question_id": item.row.get("question_id", ""),
                    "error": repr(exc),
                }
            )
            stop_queue.put(repr(exc))
            return


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(state: CounterState, args: argparse.Namespace, rows: list[dict[str, Any]], start_perf: float, start_epoch: float) -> dict[str, Any]:
    measurement_tps = state.measurement_tokens / args.measurement_s
    latencies = [
        float(row["latency_s"])
        for row in state.request_rows
        if row.get("ok") and row.get("latency_s") is not None
    ]
    counting_mode = (
        "streaming_continuous_usage"
        if state.continuous_usage_chunks > 0
        else "final_usage_fallback"
        if state.final_usage_fallback_requests > 0
        else "no_tokens_counted"
    )
    return {
        "benchmark_type": "closed_loop_steady_state",
        "throughput_name": "saturated output tokens/s",
        "run_label": args.run_label,
        "target": args.target,
        "model": args.model,
        "request_type": args.request_type,
        "dataset": str(args.dataset),
        "workload_hash": workload_hash(rows),
        "concurrency": args.concurrency,
        "warmup_s": args.warmup_s,
        "measurement_s": args.measurement_s,
        "cooldown_s": args.cooldown_s,
        "measurement_start_epoch": start_epoch + args.warmup_s,
        "measurement_end_epoch": start_epoch + args.warmup_s + args.measurement_s,
        "measurement_output_tokens": state.measurement_tokens,
        "output_tokens_per_second": measurement_tps,
        "warmup_output_tokens": state.warmup_tokens,
        "cooldown_or_drain_output_tokens": state.cooldown_tokens,
        "total_stream_output_tokens": state.total_stream_tokens,
        "requests_started": state.requests_started,
        "requests_completed": state.requests_completed,
        "requests_errored": state.requests_errored,
        "continuous_usage_chunks": state.continuous_usage_chunks,
        "final_usage_fallback_requests": state.final_usage_fallback_requests,
        "counting_mode": counting_mode,
        "ignore_eos": args.ignore_eos,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "prompt_rows": len(rows),
        "latency_mean_s": statistics.mean(latencies) if latencies else "",
        "latency_p50_s": statistics.median(latencies) if latencies else "",
        "latency_max_s": max(latencies) if latencies else "",
        "start_perf_s": start_perf,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-label", default="run")
    parser.add_argument(
        "--request-type",
        choices=["completions", "chat_completions"],
        default="completions",
    )
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--measurement-s", type=float, default=120.0)
    parser.add_argument("--cooldown-s", type=float, default=30.0)
    parser.add_argument("--bucket-s", type=float, default=1.0)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--no-continuous-usage", action="store_true")
    parser.add_argument("--allow-final-usage-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.measurement_s <= 0:
        raise SystemExit("--measurement-s must be positive")
    if args.warmup_s < 0 or args.cooldown_s < 0:
        raise SystemExit("warmup/cooldown must be non-negative")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive for steady-state runs")

    rows = read_jsonl(args.dataset, args.max_prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = CounterState()
    stop_queue: queue.Queue[str] = queue.Queue()
    start_perf = time.perf_counter()
    start_epoch = time.time()
    warmup_end = start_perf + args.warmup_s
    measurement_end = warmup_end + args.measurement_s
    url = endpoint_url(args.target, args.request_type)

    threads = [
        threading.Thread(
            target=worker,
            kwargs={
                "worker_id": worker_id,
                "rows": rows,
                "args": args,
                "state": state,
                "start_perf": start_perf,
                "warmup_end": warmup_end,
                "measurement_end": measurement_end,
                "url": url,
                "stop_queue": stop_queue,
            },
            daemon=True,
        )
        for worker_id in range(args.concurrency)
    ]
    for thread in threads:
        thread.start()

    join_deadline = measurement_end + args.cooldown_s
    for thread in threads:
        remaining = max(0.1, join_deadline - time.perf_counter())
        thread.join(timeout=remaining)

    request_rows = sorted(
        state.request_rows,
        key=lambda row: int(row.get("sequence_index", 0)),
    )
    summary = summarize(state, args, rows, start_perf, start_epoch)
    summary["unfinished_worker_threads"] = sum(1 for thread in threads if thread.is_alive())
    summary["worker_error"] = stop_queue.get() if not stop_queue.empty() else ""

    bucket_rows = []
    num_buckets = math.ceil(args.measurement_s / args.bucket_s)
    for bucket in range(num_buckets):
        tokens = state.bucket_tokens.get(bucket, 0)
        window_start = bucket * args.bucket_s
        window_end = min(args.measurement_s, (bucket + 1) * args.bucket_s)
        width = max(1e-9, window_end - window_start)
        bucket_rows.append(
            {
                "bucket_index": bucket,
                "window_start_s": window_start,
                "window_end_s": window_end,
                "output_tokens": tokens,
                "output_tokens_per_second": tokens / width,
            }
        )

    write_json(args.output_dir / "steady_state_results.json", summary)
    write_csv(args.output_dir / "steady_state_summary.csv", [summary])
    write_csv(args.output_dir / "steady_state_buckets.csv", bucket_rows)
    write_csv(args.output_dir / "steady_state_requests.csv", request_rows)
    with (args.output_dir / "steady_state_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if (
        summary["counting_mode"] != "streaming_continuous_usage"
        and not args.allow_final_usage_fallback
    ):
        print(
            "[ERROR] no streaming continuous usage tokens were counted; "
            "do not use this run for throughput",
        )
        return 2

    print(
        "[INFO] saturated output tokens/s at concurrency "
        f"{args.concurrency}: {summary['output_tokens_per_second']:.3f}"
    )
    print(f"[INFO] measurement tokens: {summary['measurement_output_tokens']}")
    print(f"[INFO] output dir: {args.output_dir}")
    return 0 if not summary["worker_error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
