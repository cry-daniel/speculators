#!/usr/bin/env python3
"""Compare dense EAGLE3 and quota-routed residual complement on GSM8K.

The two servers use the same Llama-3.1-8B target, EAGLE3 K=7 drafter,
128-example deterministic GSM8K subset, greedy decoding, request concurrency,
and execution mode.  Quota Dd/8 keeps the current verifier row plus d-1
highest-prefix-confidence draft rows dense.  ``--compile-graph`` enables
vLLM torch.compile plus exact-size CUDA Graph buckets for dynamic K=7 batches.

Acceptance length follows vLLM's convention and includes the bonus/current
token: ``1 + accepted_draft_tokens / verification_steps``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parent.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
MODELS_ROOT = SPECULATORS_ROOT.parent / "models"

sys.path.insert(0, str(SCRIPT.parent))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    extract_final_answer,
    load_gsm8k,
)
from run_structured_24_spec_quality import (  # noqa: E402
    add_local_no_proxy,
    start_vllm_server,
    stop_process,
)


MODEL_LABEL = "llama3_1_8b"
DEFAULT_MODEL = MODELS_ROOT / "llama-3.1-8b-instruct"
DEFAULT_SPECULATOR = MODELS_ROOT / "llama-3.1-8b-eagle3-speculator"
SPEC_COUNTER = re.compile(
    r"^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens))"
    r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def scrape_acceptance_counters(port: int) -> dict[str, float]:
    counters: dict[str, float] = {}
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        match = SPEC_COUNTER.match(line.strip())
        if match:
            key = match.group(1)
            counters[key] = counters.get(key, 0.0) + float(match.group(2))
    return counters


def counter_delta(before: dict[str, float], after: dict[str, float], suffix: str) -> float:
    key = f"vllm:spec_decode_num_{suffix}"
    return float(after.get(key, 0.0) - before.get(key, 0.0))


def make_env(args: argparse.Namespace, method: str, case_dir: Path) -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    env.update(
        {
            "SPECLINK_TRACE_CONFIDENCE": "0",
            "SPECLINK_STRUCTURED_24_STATS_PATH": str(
                (case_dir / "vllm_structured_24_stats.json").resolve()
            ),
        }
    )
    if not args.disable_step_stats:
        env["SPECLINK_TOKEN_DENSE_STATS_PATH"] = str(
            (case_dir / "token_dense_stats.jsonl").resolve()
        )
    if method == "eagle3_dense":
        env.update(
            {
                "SPECLINK_STRUCTURED_24_ENABLE": "0",
                "SPECLINK_TOKEN_DENSE_ENABLE": "0",
                "SPECLINK_TOKEN_DENSE_GRAPH_ROUTE": "0",
            }
        )
        return env

    graph_rows = ",".join(str(rows) for rows in args.cudagraph_capture_sizes)
    env.update(
        {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": MODEL_LABEL,
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(
                args.calibration_cache_root.resolve()
            ),
            "SPECLINK_STRUCTURED_24_POLICY": "all_sparse",
            "SPECLINK_TOKEN_DENSE_ENABLE": "1",
            "SPECLINK_TOKEN_DENSE_MODE": "high_confidence_dense",
            "SPECLINK_TOKEN_DENSE_BACKEND": "residual_complement_splitk2",
            "SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS": str(args.dense_eighths),
            "SPECLINK_TOKEN_DENSE_ROUTING_SCOPE": "global",
            "SPECLINK_TOKEN_DENSE_SCORE_MODE": getattr(
                args, "score_mode", "prefix_product"
            ),
            "SPECLINK_TOKEN_DENSE_EXPECTED_ROWS": str(args.max_num_seqs * 8),
            "SPECLINK_TOKEN_DENSE_STATS_INTERVAL": (
                "0" if args.disable_step_stats else "1"
            ),
            "SPECLINK_TOKEN_DENSE_STATS_DETAIL": "0",
            "SPECLINK_TOKEN_DENSE_PREFILL_FUSED": (
                "1" if args.prefill_fused else "0"
            ),
            # A single persistent GPU workspace is updated before every replay.
            # Exact K=7 buckets avoid padding a route for M rows into a graph
            # whose linear input has a larger M.
            "SPECLINK_TOKEN_DENSE_GRAPH_ROUTE": "1"
            if args.compile_graph
            else "0",
            "SPECLINK_TOKEN_DENSE_GRAPH_ROWS": graph_rows,
        }
    )
    return env


def post_completion(
    *,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    min_tokens: int | None = None,
    ignore_eos: bool = False,
    request_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "request_id": request_id,
    }
    if min_tokens is not None:
        body["min_tokens"] = min_tokens
    if ignore_eos:
        body["ignore_eos"] = True
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices") or []
        text = str(choices[0].get("text", "")) if choices else ""
        usage = payload.get("usage") or {}
        return {
            "generation": text,
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "latency_sec": time.perf_counter() - started,
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    return {
        "generation": "",
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "latency_sec": time.perf_counter() - started,
        "error": error,
    }


def warmup(port: int, model: str, args: argparse.Namespace) -> None:
    prompts = [
        "Question:\nWhat is 17 plus 25?\n\nAnswer:",
        "Question:\nA box has 6 rows of 7 balls. How many balls?\n\nAnswer:",
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                post_completion,
                port=port,
                model=model,
                prompt=prompt,
                max_tokens=32,
                request_id=f"gsm8k-warmup-{index}",
                timeout_s=args.request_timeout_s,
            )
            for index, prompt in enumerate(prompts)
        ]
        for future in futures:
            result = future.result()
            if result["error"]:
                raise RuntimeError(f"warmup request failed: {result['error']}")


def run_dataset(
    *,
    port: int,
    model: str,
    method: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    details: list[dict[str, Any]] = []

    def evaluate_one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        response = post_completion(
            port=port,
            model=model,
            prompt=str(row["prompt"]),
            max_tokens=args.max_tokens,
            request_id=f"gsm8k-{method}-p{index:05d}",
            timeout_s=args.request_timeout_s,
        )
        generation = str(response["generation"])
        prediction = extract_final_answer(generation)
        gold = row.get("gold_answer")
        counted = not response["error"] and gold is not None
        return {
            "index": index,
            "id": row.get("id"),
            "method": method,
            "prompt": row.get("prompt"),
            "generation": generation,
            "gold_answer": gold,
            "pred_answer": prediction,
            "correct": bool(counted and prediction is not None and prediction == gold)
            if counted
            else None,
            "counted": counted,
            **response,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(evaluate_one, index, row) for index, row in enumerate(rows)]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            details.append(result)
            if completed % 16 == 0 or completed == len(rows):
                print(f"[{method}] completed {completed}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    details.sort(key=lambda row: int(row["index"]))
    return details, elapsed


def latest_token_dense_summary(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "verify_token_mask_summary":
            latest = record
    return latest


def summarize_case(
    *,
    method: str,
    details: list[dict[str, Any]],
    elapsed: float,
    before: dict[str, float],
    after: dict[str, float],
    case_dir: Path,
    dense_eighths: int,
) -> dict[str, Any]:
    counted = [row for row in details if row["counted"]]
    correct = sum(bool(row["correct"]) for row in counted)
    completion_tokens = sum(int(row["completion_tokens"]) for row in details)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in details)
    errors = sum(bool(row["error"]) for row in details)
    verification_steps = counter_delta(before, after, "drafts")
    drafted_tokens = counter_delta(before, after, "draft_tokens")
    accepted_tokens = counter_delta(before, after, "accepted_tokens")
    route = latest_token_dense_summary(case_dir / "token_dense_stats.jsonl")
    if method.startswith("residual_complement_d") and not route:
        dense_fraction = (dense_eighths - 1) / 7
        route = {
            "dense_draft_fraction": dense_fraction,
            "sparse_draft_fraction": 1.0 - dense_fraction,
            "missing_score_tokens": None,
        }
    latencies = [float(row["latency_sec"]) for row in details]
    return {
        "method": method,
        "questions": len(details),
        "counted": len(counted),
        "correct": correct,
        "accuracy": correct / len(counted) if counted else None,
        "errors": errors,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_tokens_per_sec": completion_tokens / elapsed if elapsed else None,
        "total_tokens_per_sec": (prompt_tokens + completion_tokens) / elapsed
        if elapsed
        else None,
        "requests_per_sec": len(details) / elapsed if elapsed else None,
        "mean_request_latency_sec": statistics.mean(latencies),
        "median_request_latency_sec": statistics.median(latencies),
        "verification_steps": verification_steps,
        "draft_tokens": drafted_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "draft_acceptance_rate": accepted_tokens / drafted_tokens
        if drafted_tokens
        else None,
        "mean_acceptance_length": 1.0 + accepted_tokens / verification_steps
        if verification_steps
        else None,
        "token_dense_dense_draft_fraction": route.get("dense_draft_fraction"),
        "token_dense_sparse_draft_fraction": route.get("sparse_draft_fraction"),
        "token_dense_missing_score_tokens": route.get("missing_score_tokens"),
    }


def run_case(
    *,
    method: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = args.output_root / method
    case_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("token_dense_stats.jsonl", "vllm_structured_24_stats.json"):
        path = case_dir / stale
        if path.exists():
            path.unlink()
    env = make_env(args, method, case_dir)
    process = None
    try:
        process, port = start_vllm_server(
            args,
            base_model=str(args.model.resolve()),
            speculator_model=str(args.speculator.resolve()),
            case_dir=case_dir,
            env=env,
        )
        warmup(port, str(args.model.resolve()), args)
        before = scrape_acceptance_counters(port)
        details, elapsed = run_dataset(
            port=port,
            model=str(args.model.resolve()),
            method=method,
            rows=rows,
            args=args,
        )
        after = scrape_acceptance_counters(port)
        write_jsonl(case_dir / "generations_gsm8k.jsonl", details)
        summary = summarize_case(
            method=method,
            details=details,
            elapsed=elapsed,
            before=before,
            after=after,
            case_dir=case_dir,
            dense_eighths=args.dense_eighths,
        )
        write_json(case_dir / "summary.json", summary)
        return summary
    finally:
        stop_process(process)
        time.sleep(args.server_shutdown_settle_s)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = list(summaries[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def compare_outputs(root: Path, residual_method: str) -> dict[str, Any]:
    def load(method: str) -> list[dict[str, Any]]:
        path = root / method / "generations_gsm8k.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    dense = load("eagle3_dense")
    residual = load(residual_method)
    rows: list[dict[str, Any]] = []
    for left, right in zip(dense, residual, strict=True):
        rows.append(
            {
                "index": left["index"],
                "id": left["id"],
                "gold_answer": left["gold_answer"],
                "dense_pred_answer": left["pred_answer"],
                "residual_pred_answer": right["pred_answer"],
                "dense_correct": left["correct"],
                "residual_correct": right["correct"],
                "same_prediction": left["pred_answer"] == right["pred_answer"],
                "same_generation": left["generation"] == right["generation"],
            }
        )
    with (root / "output_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "same_prediction": sum(bool(row["same_prediction"]) for row in rows),
        "same_generation": sum(bool(row["same_generation"]) for row in rows),
        "dense_only_correct": sum(
            bool(row["dense_correct"] and not row["residual_correct"]) for row in rows
        ),
        "residual_only_correct": sum(
            bool(row["residual_correct"] and not row["dense_correct"]) for row in rows
        ),
        "questions": len(rows),
    }


def write_report(
    root: Path,
    summaries: list[dict[str, Any]],
    comparison: dict[str, Any],
    args: argparse.Namespace,
    residual_method: str,
) -> None:
    by_method = {row["method"]: row for row in summaries}
    dense = by_method["eagle3_dense"]
    residual = by_method[residual_method]
    speedup = residual["output_tokens_per_sec"] / dense["output_tokens_per_sec"]
    accuracy_delta = residual["accuracy"] - dense["accuracy"]
    lines = [
        f"# Llama-3.1-8B GSM8K: EAGLE3 dense vs residual-complement D{args.dense_eighths}/8",
        "",
        f"- Questions: {args.num_questions}; EAGLE3 K={args.num_spec_tokens}.",
        f"- Concurrency: {args.concurrency}; max output tokens: {args.max_tokens}; greedy decoding.",
        (
            "- Both methods use vLLM torch.compile with exact-size "
            "FULL_DECODE_ONLY CUDA Graph buckets. Mixed prefill runs eagerly. "
            if args.compile_graph
            else "- Both methods use eager execution. "
        )
        + "Timed throughput includes prompt prefill and generation, but excludes server startup and warmup.",
        "- Mean acceptance length follows vLLM and includes the bonus/current token.",
        "",
        "| Method | Accuracy | Correct | Mean acceptance length | Draft acceptance | Output tok/s | Total tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['accuracy']:.4f} | {row['correct']}/{row['counted']} | "
            f"{row['mean_acceptance_length']:.4f} | {row['draft_acceptance_rate']:.4f} | "
            f"{row['output_tokens_per_sec']:.2f} | {row['total_tokens_per_sec']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- Residual-complement throughput relative to dense EAGLE3: **{speedup:.4f}x**.",
            f"- Accuracy delta (residual - dense): **{accuracy_delta:+.4f}**.",
            f"- Same extracted answer: {comparison['same_prediction']}/{comparison['questions']}.",
            f"- Byte-identical full generation: {comparison['same_generation']}/{comparison['questions']}.",
            f"- Dense-only correct / residual-only correct: {comparison['dense_only_correct']} / {comparison['residual_only_correct']}.",
            "",
            "## Outputs",
            "",
            "- `eagle3_dense/generations_gsm8k.jsonl`: all dense EAGLE3 generations.",
            f"- `{residual_method}/generations_gsm8k.jsonl`: all D{args.dense_eighths}/8 generations.",
            "- `output_comparison.csv`: paired answer and correctness comparison.",
            "- Per-method `summary.json`, server log, server command, and sparse-route statistics.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--speculator", type=Path, default=DEFAULT_SPECULATOR)
    parser.add_argument(
        "--calibration-cache-root",
        type=Path,
        default=DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    )
    parser.add_argument("--num-questions", type=int, default=128)
    parser.add_argument("--num-spec-tokens", type=int, default=7)
    parser.add_argument("--dense-eighths", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--port-base", type=int, default=8050)
    parser.add_argument("--health-timeout-s", type=float, default=1200.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable vLLM torch.compile and FULL_DECODE_ONLY CUDA Graphs with "
            "one exact bucket per possible K=7 verifier batch."
        ),
    )
    parser.add_argument(
        "--disable-step-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable token-route aggregation, locking, and per-step JSONL writes.",
    )
    parser.add_argument(
        "--prefill-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the single-kernel dual-2:4 base+complement path for prefill "
            "(default); --no-prefill-fused retains the decode-oriented "
            "indexed Split-K2 ablation."
        ),
    )
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = EVAL_ROOT / (
            f"results/llama_gsm8k_eagle3_residual_d{args.dense_eighths}_{timestamp()}"
        )
    if args.num_spec_tokens != 7:
        parser.error("eighths-based verifier quotas require --num-spec-tokens 7")
    if args.max_num_seqs < args.concurrency:
        parser.error("--max-num-seqs must be >= --concurrency")
    args.cudagraph_capture_sizes = list(
        range(args.num_spec_tokens + 1, (args.num_spec_tokens + 1) * args.max_num_seqs + 1, args.num_spec_tokens + 1)
    )
    if args.compile_graph:
        if args.enforce_eager:
            parser.error("--compile-graph requires --no-enforce-eager")
        args.compilation_config = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
            "max_cudagraph_capture_size": args.cudagraph_capture_sizes[-1],
            "cudagraph_num_of_warmups": 1,
        }
    else:
        args.compilation_config = None
    return args


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.model.exists() or not args.speculator.exists():
        raise SystemExit(f"missing model/speculator: {args.model}, {args.speculator}")
    pack = load_gsm8k(args.num_questions, args.seed)
    if pack.error or len(pack.rows) != args.num_questions:
        raise SystemExit(
            f"failed to load {args.num_questions} GSM8K rows: {pack.error or len(pack.rows)}"
        )
    write_json(
        args.output_root / "run_config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    residual_method = f"residual_complement_d{args.dense_eighths}"
    summaries = []
    for method in ("eagle3_dense", residual_method):
        print(f"[run] {method}", flush=True)
        summaries.append(run_case(method=method, rows=pack.rows, args=args))
    write_summary_csv(args.output_root / "summary.csv", summaries)
    comparison = compare_outputs(args.output_root, residual_method)
    write_json(args.output_root / "comparison.json", comparison)
    write_report(args.output_root, summaries, comparison, args, residual_method)
    print(args.output_root)
    print(json.dumps({"summaries": summaries, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
