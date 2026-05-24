#!/usr/bin/env python3
"""Evaluate final-answer accuracy through an OpenAI-compatible vLLM API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from speculators.speclink.math_eval import (  # noqa: E402
    build_math_prompt,
    extract_final_answer,
    flexible_answer_equal,
    load_dataset,
    output_equivalence,
    strict_answer_equal,
    write_jsonl,
)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def call_chat(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[str | None, dict[str, Any] | None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    response = post_json(
        base_url.rstrip("/") + "/v1/chat/completions",
        payload,
        timeout=timeout,
    )
    content = response["choices"][0]["message"]["content"]
    return content, response.get("usage")


def load_dense_reference(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    refs: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            refs[str(row["id"])] = row
    return refs


def evaluate_one(
    index: int,
    record: Any,
    args: argparse.Namespace,
    dense_refs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt = build_math_prompt(record.prompt)
    reference_answer = extract_final_answer(record.reference_raw)
    started = time.perf_counter()
    output = None
    usage = None
    error = None
    try:
        output, usage = call_chat(
            args.base_url,
            args.model,
            prompt,
            args.temperature,
            args.max_tokens,
            args.timeout,
        )
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        error = repr(exc)
    latency = time.perf_counter() - started
    extracted = extract_final_answer(output)
    dense_ref = dense_refs.get(record.id)
    exact_equiv = None
    normalized_equiv = None
    if dense_ref:
        exact_equiv, normalized_equiv = output_equivalence(
            output,
            dense_ref.get("model_output"),
        )
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    output_tokens = usage.get("completion_tokens") if usage else None
    return {
        "id": record.id,
        "index": index,
        "prompt": record.prompt,
        "reference_raw": record.reference_raw,
        "reference_answer": reference_answer,
        "model_output": output,
        "extracted_answer": extracted,
        "strict_correct": strict_answer_equal(extracted, reference_answer),
        "flexible_correct": flexible_answer_equal(extracted, reference_answer),
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "dense_output_exact_equivalence": exact_equiv,
        "dense_output_normalized_equivalence": normalized_equiv,
        "error": error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    latencies = [row["latency"] for row in rows if row.get("error") is None]
    output_tokens = [
        row["output_tokens"] for row in rows if isinstance(row.get("output_tokens"), int)
    ]
    dense_equiv_values = [
        row["dense_output_normalized_equivalence"]
        for row in rows
        if row["dense_output_normalized_equivalence"] is not None
    ]
    invalid = [row for row in rows if not row.get("extracted_answer")]
    return {
        "n": n,
        "errors": sum(1 for row in rows if row.get("error")),
        "strict_em": sum(bool(row["strict_correct"]) for row in rows) / n if n else None,
        "flexible_em": sum(bool(row["flexible_correct"]) for row in rows) / n if n else None,
        "pass_at_1": sum(bool(row["flexible_correct"]) for row in rows) / n if n else None,
        "invalid_extract_rate": len(invalid) / n if n else None,
        "avg_latency": statistics.mean(latencies) if latencies else None,
        "p50_latency": percentile(latencies, 50),
        "p95_latency": percentile(latencies, 95),
        "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
        "dense_output_equivalence_rate": (
            sum(bool(value) for value in dense_equiv_values) / len(dense_equiv_values)
            if dense_equiv_values
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--dense-reference-jsonl")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, limit=args.limit)
    dense_refs = load_dense_reference(args.dense_reference_jsonl)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(evaluate_one, index, record, args, dense_refs)
            for index, record in enumerate(dataset)
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["index"])
    write_jsonl(args.out_jsonl, rows)
    summary = summarize(rows)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out_jsonl}")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
