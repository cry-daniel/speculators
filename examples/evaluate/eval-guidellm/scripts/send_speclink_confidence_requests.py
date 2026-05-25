#!/usr/bin/env python3
"""Send deterministic OpenAI-compatible requests for confidence tracing."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if 0 < max_prompts <= len(rows):
                break
    return rows


def build_prompt(row: dict[str, Any], dataset_index: int) -> str:
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
    raise ValueError(f"Dataset row {dataset_index} has no text prompt")


def post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(
    *,
    target: str,
    model: str,
    model_label: str,
    dataset_label: str,
    method: str,
    num_spec_tokens: int,
    row: dict[str, Any],
    dataset_index: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    prompt = build_prompt(row, dataset_index)
    effective_dataset_label = dataset_label or str(row.get("dataset_label") or "")
    label_prefix = f"{model_label}-" if model_label else ""
    dataset_prefix = f"{effective_dataset_label}-" if effective_dataset_label else ""
    request_id = (
        f"speclink-{dataset_prefix}{label_prefix}{method}-k{num_spec_tokens}-p{dataset_index:06d}"
    )
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "ignore_eos": True,
        "request_id": request_id,
    }
    started = time.perf_counter()
    try:
        response = post_json(f"{target.rstrip('/')}/v1/completions", body, timeout)
        ok = True
        error = None
    except urllib.error.HTTPError as exc:
        ok = False
        error = exc.read().decode("utf-8", errors="replace")
        response = {}
    except Exception as exc:  # noqa: BLE001
        ok = False
        error = repr(exc)
        response = {}
    elapsed = time.perf_counter() - started
    return {
        "ok": ok,
        "error": error,
        "request_id": request_id,
        "dataset_index": dataset_index,
        "question_id": row.get("question_id"),
        "category": row.get("category"),
        "dataset_label": effective_dataset_label,
        "method": method,
        "model_label": model_label,
        "num_spec_tokens": num_spec_tokens,
        "latency_s": elapsed,
        "prompt_chars": len(prompt),
        "response_id": response.get("id"),
        "usage": response.get("usage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", required=True, choices=["eagle3", "peagle"])
    parser.add_argument("--num-spec-tokens", type=int, required=True)
    parser.add_argument("--max-prompts", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset, args.max_prompts)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                run_one,
                target=args.target,
                model=args.model,
                model_label=args.model_label,
                dataset_label=args.dataset_label,
                method=args.method,
                num_spec_tokens=args.num_spec_tokens,
                row=row,
                dataset_index=i,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + i,
                timeout=args.timeout,
            )
            for i, row in enumerate(rows)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok" if result["ok"] else "error"
            print(
                f"[{status}] {result['request_id']} "
                f"latency={result['latency_s']:.3f}s"
            )

    results.sort(key=lambda item: item["dataset_index"])
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")

    failures = [item for item in results if not item["ok"]]
    if failures:
        raise SystemExit(f"{len(failures)} request(s) failed; see {args.output_jsonl}")


if __name__ == "__main__":
    main()
