#!/usr/bin/env python3
"""Replay lm-eval samples through a running vLLM OpenAI completions server.

This is a focused diagnostic for SR24 serving-only quality regressions.  It
mirrors lm-eval local-completions tokenized requests by sending prompt token ids
to `/v1/completions`, optionally replaying earlier docs before recording the
target docs.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                row["_sample_index"] = index
                rows.append(row)
    return rows


def sample_prompt(sample: dict[str, Any]) -> str:
    arguments = sample.get("arguments")
    if isinstance(arguments, dict):
        gen_args = arguments.get("gen_args_0")
        if isinstance(gen_args, dict) and "arg_0" in gen_args:
            return str(gen_args["arg_0"])
    prompt = sample.get("prompt")
    if prompt is not None:
        return str(prompt)
    raise ValueError("sample has no arguments.gen_args_0.arg_0 or prompt")


def sample_gen_args(sample: dict[str, Any]) -> dict[str, Any]:
    arguments = sample.get("arguments")
    if isinstance(arguments, dict):
        gen_args = arguments.get("gen_args_0")
        if isinstance(gen_args, dict) and isinstance(gen_args.get("arg_1"), dict):
            return dict(gen_args["arg_1"])
    return {}


def parse_doc_ids(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def parse_indices(raw: str) -> set[int]:
    if not raw:
        return set()
    return parse_doc_ids(raw)


def request_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    request_id: str,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def truncate_prompt_tokens(
    token_ids: list[int],
    *,
    max_model_len: int,
    max_tokens: int,
) -> tuple[list[int], bool]:
    # Mirrors lm-eval TemplateAPI tokenized local-completions behavior:
    # max_length is effectively max_model_len - 1 before reserving gen tokens.
    max_context_len = max(1, int(max_model_len) - 1 - int(max_tokens))
    if len(token_ids) <= max_context_len:
        return token_ids, False
    return token_ids[-max_context_len:], True


def build_payload(
    *,
    model: str,
    prompt_token_ids: list[int],
    gen_args: dict[str, Any],
    default_max_tokens: int,
    request_id: str,
    return_token_ids: bool,
) -> dict[str, Any]:
    max_tokens = int(gen_args.get("max_gen_toks") or default_max_tokens)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": float(gen_args.get("temperature", 0)),
        "top_p": float(gen_args.get("top_p", 1)),
        "request_id": request_id,
    }
    if return_token_ids:
        payload["return_token_ids"] = True
    until = gen_args.get("until")
    if until:
        payload["stop"] = until
    if not bool(gen_args.get("do_sample", False)):
        payload["temperature"] = 0
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:8152/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-doc-ids", default="")
    parser.add_argument(
        "--target-row-indices",
        default="",
        help=(
            "Comma-separated zero-based sample file row indices. Prefer this "
            "when an lm-eval samples file contains duplicate doc_id values."
        ),
    )
    parser.add_argument(
        "--replay-through-doc-id",
        type=int,
        default=None,
        help=(
            "Replay every sample with doc_id <= this value. Only target docs "
            "are written with full text; earlier docs are sent to reproduce "
            "serving state."
        ),
    )
    parser.add_argument(
        "--replay-through-row-index",
        type=int,
        default=None,
        help="Replay every sample row up to and including this zero-based index.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--add-bos-token", action="store_true")
    parser.add_argument(
        "--request-id-prefix",
        default="sr24-replay",
        help=(
            "Prefix for stable per-sample request ids. vLLM completions will "
            "expose these as cmpl-{prefix}-doc{doc_id}-row{row}-0 in trace."
        ),
    )
    parser.add_argument(
        "--return-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ask vLLM to include generated token_ids in the completion "
            "response. Enabled by default for trace alignment diagnostics."
        ),
    )
    args = parser.parse_args()

    samples = load_samples(args.samples_path)
    targets = parse_doc_ids(args.target_doc_ids)
    target_rows = parse_indices(args.target_row_indices)
    replay_limit = args.replay_through_doc_id
    replay_row_limit = args.replay_through_row_index
    selected = [
        sample for sample in samples
        if (
            int(sample.get("doc_id", -1)) in targets
            or int(sample.get("_sample_index", -1)) in target_rows
            or (
                replay_limit is not None
                and int(sample.get("doc_id", -1)) <= int(replay_limit)
            )
            or (
                replay_row_limit is not None
                and int(sample.get("_sample_index", -1)) <= int(replay_row_limit)
            )
        )
    ]
    if not selected:
        raise SystemExit("no selected samples")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows: list[dict[str, Any]] = []
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    for sample in selected:
        doc_id = int(sample.get("doc_id", -1))
        sample_index = int(sample.get("_sample_index", -1))
        gen_args = sample_gen_args(sample)
        max_tokens = int(gen_args.get("max_gen_toks") or args.default_max_tokens)
        token_ids = tokenizer.encode(
            sample_prompt(sample),
            add_special_tokens=args.add_bos_token,
        )
        token_ids, truncated = truncate_prompt_tokens(
            token_ids,
            max_model_len=args.max_model_len,
            max_tokens=max_tokens,
        )
        request_id = f"{args.request_id_prefix}-doc{doc_id}-row{sample_index}"
        payload = build_payload(
            model=args.model,
            prompt_token_ids=token_ids,
            gen_args=gen_args,
            default_max_tokens=args.default_max_tokens,
            request_id=request_id,
            return_token_ids=args.return_token_ids,
        )
        start = time.perf_counter()
        response = request_json(
            args.server_url,
            payload,
            timeout=args.timeout,
            request_id=request_id,
        )
        elapsed = time.perf_counter() - start
        choice = (response.get("choices") or [{}])[0]
        text = str(choice.get("text") or "")
        row = {
            "request_id": request_id,
            "expected_trace_request_id": f"cmpl-{request_id}-0",
            "doc_id": doc_id,
            "sample_index": sample_index,
            "target_doc": doc_id in targets or sample_index in target_rows,
            "elapsed_seconds": elapsed,
            "prompt_tokens": len(token_ids),
            "prompt_truncated": truncated,
            "finish_reason": choice.get("finish_reason"),
            "text": text if doc_id in targets or sample_index in target_rows else "",
            "text_head": text[:240],
            "token_ids": choice.get("token_ids"),
            "prompt_token_ids": choice.get("prompt_token_ids"),
            "response_usage": response.get("usage"),
            "reference_filtered_resps": sample.get("filtered_resps"),
            "reference_exact_match": sample.get("exact_match"),
        }
        rows.append(row)
        args.output_path.write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in rows) + "\n",
            encoding="utf-8",
        )
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)
    print(args.output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
