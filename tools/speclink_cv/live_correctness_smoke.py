#!/usr/bin/env python3
"""Compare live SpecLink-CV output against one-shot EAGLE3 output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument(
        "--speculator-model",
        default=os.environ.get("EAGLE3_SPECULATOR_MODEL", ""),
        help="EAGLE3 speculator path. Defaults to EAGLE3_SPECULATOR_MODEL.",
    )
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Solve step by step: What is 17 plus 25?",
    )
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file with prompts. When set, runs a batched smoke.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Prompt field to read from --prompts-jsonl.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to read from --prompts-jsonl.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--chat-completions",
        action="store_true",
        help=(
            "Render prompts as OpenAI-style user chat messages before "
            "generation, matching the GuideLLM chat_completions path."
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=root / "examples/evaluate/eval-guidellm",
        help="Run child vLLM imports from here to avoid repo-root import shadowing.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root
        / "examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke.json",
    )
    parser.add_argument(
        "--event-jsonl",
        type=Path,
        default=root
            / "examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_events.jsonl",
    )
    parser.add_argument(
        "--baseline-event-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional baseline one-shot debug event JSONL. When omitted and "
            "--debug-dump is set, a sibling *_baseline_events.jsonl file is used."
        ),
    )
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=None,
        help="Optional live scheduler/chunk profile JSONL for the CV run.",
    )
    parser.add_argument(
        "--profile-max-events",
        type=int,
        default=500,
        help=(
            "Maximum regular profile JSONL rows for the CV child. One limit "
            "marker may be added. Use 0 to disable the cap for a short "
            "trusted run."
        ),
    )
    parser.add_argument(
        "--log-max-events",
        type=int,
        default=1_000,
        help=(
            "Maximum regular verbose event JSONL rows for each child. One "
            "limit marker may be added. Use 0 to disable the cap for a short "
            "trusted run."
        ),
    )
    parser.add_argument(
        "--confidence-sizing",
        action="store_true",
        help="Enable live uncalibrated DLM-confidence chunk sizing for the CV run.",
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=None,
        help="Optional binning calibration_model.json for live confidence sizing.",
    )
    parser.add_argument(
        "--candidate-chunks",
        default="",
        help=(
            "Optional SPECLINK_CV_CANDIDATE_CHUNKS override, e.g. '1,8'. "
            "Useful for diagnosing whether smaller chunks are exact."
        ),
    )
    parser.add_argument(
        "--force-prefix-len",
        type=int,
        default=0,
        help="Optional SPECLINK_CV_FORCE_PREFIX_LEN diagnostic override.",
    )
    parser.add_argument(
        "--dense-realign-steps",
        type=int,
        default=None,
        help=(
            "Optional SPECLINK_CV_DENSE_REALIGN_STEPS diagnostic override. "
            "Use 0 only to isolate suffix-realignment effects."
        ),
    )
    parser.add_argument(
        "--prefix-reject-dense-realign-steps",
        type=int,
        default=None,
        help=(
            "Optional SPECLINK_CV_PREFIX_REJECT_DENSE_REALIGN_STEPS "
            "diagnostic override. Positive values skip immediate drafting "
            "after prefix rejection and run dense TLM realignment steps."
        ),
    )
    parser.add_argument(
        "--roofline-packing",
        action="store_true",
        help=(
            "Enable the live roofline-aware gate. Underfilled prefix chunks "
            "fall back to one-shot verification."
        ),
    )
    parser.add_argument(
        "--async-queue",
        action="store_true",
        help="Enable the live SpecLink-CV prefix verification queue.",
    )
    parser.add_argument(
        "--allow-batched-prefix-verification",
        action="store_true",
        help=(
            "Experimental: let several prefix chunks share one TLM step. "
            "Default conservative mode caps live prefix verification at one."
        ),
    )
    parser.add_argument(
        "--global-batch-barrier",
        action="store_true",
        help=(
            "Diagnostic: in async queue mode, dispatch h<K prefix chunks only "
            "when every running request can enter the same prefix batch."
        ),
    )
    parser.add_argument(
        "--allow-shape-drift-chunking",
        action="store_true",
        help=(
            "Experimental: allow h<K live chunking despite verifier-shape "
            "argmax drift evidence. Default exact-safe mode falls back to "
            "one-shot verification."
        ),
    )
    parser.add_argument(
        "--confirm-prefix-reject-one-shot",
        action="store_true",
        help=(
            "Diagnostic: when a prefix chunk rejects, discard that prefix "
            "output and requeue the original full-K draft for one-shot "
            "confirmation before committing tokens."
        ),
    )
    parser.add_argument(
        "--confirm-prefix-accept-one-shot",
        action="store_true",
        help=(
            "Diagnostic: when a prefix chunk is fully accepted, discard that "
            "prefix output and requeue the original full-K draft for one-shot "
            "confirmation before committing tokens."
        ),
    )
    parser.add_argument(
        "--prefix-low-margin-fallback-threshold",
        type=float,
        default=None,
        help=(
            "Diagnostic: if any prefix verifier top-1/top-2 logit margin is "
            "at or below this threshold, discard the prefix output and requeue "
            "the original full-K draft for one-shot confirmation."
        ),
    )
    parser.add_argument(
        "--batch-wide-low-margin-fallback",
        action="store_true",
        help=(
            "Diagnostic: when any request in a prefix verifier batch is below "
            "the low-margin threshold, requeue every request in that prefix "
            "batch for full-K confirmation to preserve the one-shot batch "
            "shape more closely."
        ),
    )
    parser.add_argument(
        "--batch-wide-prefix-reject-fallback",
        action="store_true",
        help=(
            "Diagnostic: when any request in a prefix verifier batch rejects, "
            "discard every prefix row in that batch and requeue the original "
            "full-K drafts for one-shot confirmation."
        ),
    )
    parser.add_argument(
        "--recompute-committed-prefix",
        action="store_true",
        help=(
            "Diagnostic: after a prefix chunk commits tokens, roll back the "
            "computed-token cursor so the next TLM step recomputes committed "
            "tokens and overwrites h<K verifier KV state."
        ),
    )
    parser.add_argument(
        "--allow-batched-dense-realign",
        action="store_true",
        help=(
            "Diagnostic: do not isolate dense realignment/recompute steps; "
            "allow them to batch with other running requests to test whether "
            "active-batch shape explains h<K drift."
        ),
    )
    parser.add_argument(
        "--prefix-no-kv-write",
        action="store_true",
        help=(
            "Diagnostic: run h<K prefix probes without writing KV cache. "
            "Accepted-prefix rows are forced to full-K confirmation, while "
            "prefix rejects rely on committed-prefix recompute before drafting."
        ),
    )
    parser.add_argument(
        "--confirmation-full-active-set",
        action="store_true",
        help=(
            "Diagnostic: when any request is requeued for full-K confirmation, "
            "also force every currently running non-suffix draft request into "
            "the same full-K confirmation batch."
        ),
    )
    parser.add_argument(
        "--lockstep-iteration-barrier",
        action="store_true",
        help=(
            "Diagnostic: hold requests that have resolved a prefix batch until "
            "every request from that same speculative iteration has resolved."
        ),
    )
    parser.add_argument(
        "--prefix-probe-block-rollback",
        action="store_true",
        help=(
            "Diagnostic: before requeueing a discarded h<K prefix probe for "
            "full-K confirmation, truncate the request KV block table back to "
            "the committed num_computed_tokens length."
        ),
    )
    parser.add_argument(
        "--draft-accept-eps",
        type=float,
        default=None,
        help=(
            "Diagnostic: set SPECLINK_CV_DRAFT_ACCEPT_EPS for both baseline "
            "and CV child runs. In VLLM_BATCH_INVARIANT mode, greedy "
            "rejection sampling accepts a draft token whose verifier logit is "
            "within eps of the target max logit."
        ),
    )
    parser.add_argument(
        "--greedy-eps",
        type=float,
        default=None,
        help=(
            "Diagnostic: set SPECLINK_CV_GREEDY_EPS for both baseline and CV "
            "child runs. In VLLM_BATCH_INVARIANT mode, greedy sampling picks "
            "the smallest token id within eps of the row max."
        ),
    )
    parser.add_argument(
        "--util-threshold",
        type=float,
        default=None,
        help="Override SPECLINK_CV_UTIL_THRESHOLD for roofline-packing smoke.",
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        help="Enable verbose SpecLink-CV event fields for the CV child run.",
    )
    parser.add_argument(
        "--kv-debug-tail-tokens",
        type=int,
        default=0,
        help=(
            "Debug only: dump KV checksums for this many logical tokens at "
            "each verifier row. The vLLM debug event records both the legacy "
            "seq-tail window and the history window before the first verifier "
            "target position. Keep this small, e.g. 4."
        ),
    )
    parser.add_argument(
        "--kv-debug-max-layers",
        type=int,
        default=0,
        help=(
            "Debug only: maximum attention layers for KV checksum dumps. "
            "Keep this small, e.g. 2."
        ),
    )
    parser.add_argument(
        "--kv-debug-row-index",
        type=int,
        default=-1,
        help=(
            "Debug only: restrict KV checksum dumps to one active-batch row. "
            "Use -1 to dump every row."
        ),
    )
    parser.add_argument(
        "--kv-debug-min-output-tokens",
        type=int,
        default=-1,
        help=(
            "Debug only: only dump KV checksums when the request already has "
            "at least this many output tokens. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--kv-debug-max-output-tokens",
        type=int,
        default=-1,
        help=(
            "Debug only: only dump KV checksums when the request already has "
            "at most this many output tokens. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra environment variable for both baseline and CV child runs. "
            "Repeatable; useful for backend diagnostics such as "
            "VLLM_ATTENTION_BACKEND=TRITON_ATTN."
        ),
    )
    return parser.parse_args()


def run_child(
    args: argparse.Namespace, *, enable_cv: bool, label: str
) -> dict[str, Any]:
    if not args.speculator_model:
        raise SystemExit(
            "--speculator-model is required, or set EAGLE3_SPECULATOR_MODEL"
        )

    prompts = load_prompts(args)
    code = r"""
from vllm import LLM, SamplingParams
import json
import os

prompts = json.loads(os.environ["SMOKE_PROMPTS_JSON"])
sampling_params = SamplingParams(
    max_tokens=int(os.environ["SMOKE_MAX_TOKENS"]),
    temperature=0,
    top_p=1,
)
llm = LLM(
    model=os.environ["SMOKE_MODEL"],
    speculative_config={
        "model": os.environ["SMOKE_SPECULATOR_MODEL"],
        "num_speculative_tokens": int(os.environ["SMOKE_NUM_SPEC_TOKENS"]),
        "method": "eagle3",
        "max_model_len": 4096,
    },
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    max_num_batched_tokens=1024,
    max_num_seqs=int(os.environ["SMOKE_MAX_NUM_SEQS"]),
    enforce_eager=True,
    async_scheduling=False,
)
if os.environ.get("SMOKE_USE_CHAT_COMPLETIONS") == "1":
    conversations = [
        [{"role": "user", "content": prompt}]
        for prompt in prompts
    ]
    out = llm.chat(conversations, sampling_params)
else:
    out = llm.generate(prompts, sampling_params)
items = []
for prompt, result in zip(prompts, out):
    completion = result.outputs[0]
    items.append({
        "prompt": prompt,
        "text": completion.text,
        "token_ids": list(completion.token_ids),
    })
print(json.dumps({
    "text": items[0]["text"],
    "token_ids": items[0]["token_ids"],
    "outputs": items,
}, ensure_ascii=False))
"""
    env = os.environ.copy()
    env.update(
        {
            "SMOKE_MODEL": args.model,
            "SMOKE_SPECULATOR_MODEL": args.speculator_model,
            "SMOKE_NUM_SPEC_TOKENS": str(args.num_spec_tokens),
            "SMOKE_MAX_TOKENS": str(args.max_tokens),
            "SMOKE_MAX_NUM_SEQS": str(args.max_num_seqs),
            "SMOKE_PROMPTS_JSON": json.dumps(prompts, ensure_ascii=False),
            "SMOKE_USE_CHAT_COMPLETIONS": "1"
            if args.chat_completions
            else "0",
            "SPECLINK_CV_ENABLE": "1" if enable_cv else "0",
        }
    )
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env expects a non-empty KEY, got {item!r}")
        env[key] = value
    if args.log_max_events is not None:
        env["SPECLINK_CV_LOG_MAX_EVENTS"] = str(args.log_max_events)
    if args.profile_max_events is not None:
        env["SPECLINK_CV_PROFILE_MAX_EVENTS"] = str(args.profile_max_events)
    if args.kv_debug_tail_tokens:
        env["SPECLINK_CV_KV_DEBUG_TAIL_TOKENS"] = str(
            args.kv_debug_tail_tokens
        )
    if args.kv_debug_max_layers:
        env["SPECLINK_CV_KV_DEBUG_MAX_LAYERS"] = str(
            args.kv_debug_max_layers
        )
    if args.kv_debug_row_index >= 0:
        env["SPECLINK_CV_KV_DEBUG_ROW_INDEX"] = str(args.kv_debug_row_index)
    if args.kv_debug_min_output_tokens >= 0:
        env["SPECLINK_CV_KV_DEBUG_MIN_OUTPUT_TOKENS"] = str(
            args.kv_debug_min_output_tokens
        )
    if args.kv_debug_max_output_tokens >= 0:
        env["SPECLINK_CV_KV_DEBUG_MAX_OUTPUT_TOKENS"] = str(
            args.kv_debug_max_output_tokens
        )
    if args.draft_accept_eps is not None:
        env["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = str(args.draft_accept_eps)
    if args.greedy_eps is not None:
        env["SPECLINK_CV_GREEDY_EPS"] = str(args.greedy_eps)
    if enable_cv:
        env["SPECLINK_CV_LOG_JSONL"] = str(args.event_jsonl)
        if args.confidence_sizing:
            env["SPECLINK_CV_CONFIDENCE_SIZING"] = "1"
        if args.candidate_chunks:
            env["SPECLINK_CV_CANDIDATE_CHUNKS"] = args.candidate_chunks
        if args.force_prefix_len:
            env["SPECLINK_CV_FORCE_PREFIX_LEN"] = str(args.force_prefix_len)
        if args.dense_realign_steps is not None:
            env["SPECLINK_CV_DENSE_REALIGN_STEPS"] = str(
                args.dense_realign_steps
            )
        if args.prefix_reject_dense_realign_steps is not None:
            env["SPECLINK_CV_PREFIX_REJECT_DENSE_REALIGN_STEPS"] = str(
                args.prefix_reject_dense_realign_steps
            )
        if args.calibration_path is not None:
            env["SPECLINK_CV_CALIBRATION_PATH"] = str(args.calibration_path)
        if args.roofline_packing:
            env["SPECLINK_CV_ROOFLINE_PACKING"] = "1"
        if args.async_queue:
            env["SPECLINK_CV_ASYNC_QUEUE"] = "1"
        if args.allow_batched_prefix_verification:
            env["SPECLINK_CV_ALLOW_BATCHED_PREFIX"] = "1"
        if args.global_batch_barrier:
            env["SPECLINK_CV_GLOBAL_BATCH_BARRIER"] = "1"
        if args.allow_shape_drift_chunking:
            env["SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING"] = "1"
        if args.confirm_prefix_reject_one_shot:
            env["SPECLINK_CV_CONFIRM_PREFIX_REJECT_ONE_SHOT"] = "1"
        if args.confirm_prefix_accept_one_shot:
            env["SPECLINK_CV_CONFIRM_PREFIX_ACCEPT_ONE_SHOT"] = "1"
        if args.prefix_low_margin_fallback_threshold is not None:
            env["SPECLINK_CV_PREFIX_LOW_MARGIN_FALLBACK_THRESHOLD"] = str(
                args.prefix_low_margin_fallback_threshold
            )
        if args.batch_wide_low_margin_fallback:
            env["SPECLINK_CV_BATCH_WIDE_LOW_MARGIN_FALLBACK"] = "1"
        if args.batch_wide_prefix_reject_fallback:
            env["SPECLINK_CV_BATCH_WIDE_PREFIX_REJECT_FALLBACK"] = "1"
        if args.recompute_committed_prefix:
            env["SPECLINK_CV_RECOMPUTE_COMMITTED_PREFIX"] = "1"
        if args.allow_batched_dense_realign:
            env["SPECLINK_CV_ALLOW_BATCHED_DENSE_REALIGN"] = "1"
        if args.prefix_no_kv_write:
            env["SPECLINK_CV_PREFIX_NO_KV_WRITE"] = "1"
        if args.confirmation_full_active_set:
            env["SPECLINK_CV_CONFIRMATION_FULL_ACTIVE_SET"] = "1"
        if args.lockstep_iteration_barrier:
            env["SPECLINK_CV_LOCKSTEP_ITERATION_BARRIER"] = "1"
        if args.prefix_probe_block_rollback:
            env["SPECLINK_CV_PREFIX_PROBE_BLOCK_ROLLBACK"] = "1"
        if args.util_threshold is not None:
            env["SPECLINK_CV_UTIL_THRESHOLD"] = str(args.util_threshold)
        if args.profile_jsonl is not None:
            env["SPECLINK_CV_PROFILE_JSONL"] = str(args.profile_jsonl)
        if args.debug_dump:
            env["SPECLINK_CV_DEBUG_DUMP"] = "1"
    elif args.debug_dump:
        env["SPECLINK_CV_LOG_JSONL"] = str(args.baseline_event_jsonl)
        env["SPECLINK_CV_DEBUG_DUMP"] = "1"

    started_at = time.monotonic()
    print(f"[live_correctness_smoke] start {label}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=args.workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started_at
    print(
        f"[live_correctness_smoke] end {label} "
        f"returncode={proc.returncode} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout)

    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    search_start = 0
    while True:
        json_start = proc.stdout.find("{", search_start)
        if json_start < 0:
            break
        try:
            candidate, _ = decoder.raw_decode(proc.stdout[json_start:])
        except json.JSONDecodeError:
            search_start = json_start + 1
            continue
        if isinstance(candidate, dict) and "outputs" in candidate:
            result = candidate
        search_start = json_start + 1
    if result is not None:
        result["combined_log_tail"] = proc.stdout[-4000:]
        return result
    raise SystemExit(f"child run did not emit JSON:\n{proc.stdout[-4000:]}")


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_jsonl is None:
        return [args.prompt]
    path = args.prompts_jsonl
    if not path.is_absolute():
        path = repo_root() / path
    prompts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get(args.prompt_field)
            if prompt is None:
                raise SystemExit(
                    f"missing prompt field {args.prompt_field!r} in {path}"
                )
            prompts.append(str(prompt))
            if len(prompts) >= args.num_prompts:
                break
    if not prompts:
        raise SystemExit(f"no prompts loaded from {path}")
    return prompts


def main() -> int:
    args = parse_args()
    root = repo_root()
    if not args.workdir.is_absolute():
        args.workdir = root / args.workdir
    if not args.output_json.is_absolute():
        args.output_json = root / args.output_json
    if not args.event_jsonl.is_absolute():
        args.event_jsonl = root / args.event_jsonl
    if args.baseline_event_jsonl is None:
        args.baseline_event_jsonl = args.event_jsonl.with_name(
            f"{args.event_jsonl.stem}_baseline{args.event_jsonl.suffix}"
        )
    elif not args.baseline_event_jsonl.is_absolute():
        args.baseline_event_jsonl = root / args.baseline_event_jsonl
    if args.calibration_path is not None and not args.calibration_path.is_absolute():
        args.calibration_path = root / args.calibration_path
    if args.prompts_jsonl is not None and not args.prompts_jsonl.is_absolute():
        args.prompts_jsonl = root / args.prompts_jsonl
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.event_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.event_jsonl.exists():
        args.event_jsonl.unlink()
    args.baseline_event_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline_event_jsonl.exists():
        args.baseline_event_jsonl.unlink()
    if args.profile_jsonl is not None:
        if not args.profile_jsonl.is_absolute():
            args.profile_jsonl = root / args.profile_jsonl
        args.profile_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.profile_jsonl.exists():
            args.profile_jsonl.unlink()

    baseline = run_child(args, enable_cv=False, label="baseline_eagle3")
    speclink_cv = run_child(args, enable_cv=True, label="speclink_cv")
    baseline_outputs = baseline.get("outputs") or []
    cv_outputs = speclink_cv.get("outputs") or []
    matched_items = [
        left.get("token_ids") == right.get("token_ids")
        for left, right in zip(baseline_outputs, cv_outputs)
    ]
    mismatches: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(baseline_outputs, cv_outputs)):
        left_ids = list(left.get("token_ids") or [])
        right_ids = list(right.get("token_ids") or [])
        if left_ids == right_ids:
            continue
        first_diff_index = None
        for token_index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
            if left_id != right_id:
                first_diff_index = token_index
                break
        if first_diff_index is None and len(left_ids) != len(right_ids):
            first_diff_index = min(len(left_ids), len(right_ids))
        mismatches.append(
            {
                "index": index,
                "prompt": left.get("prompt", ""),
                "first_diff_index": first_diff_index,
                "baseline_token_ids": left_ids,
                "speclink_cv_token_ids": right_ids,
                "baseline_text": left.get("text", ""),
                "speclink_cv_text": right.get("text", ""),
            }
        )
    matched = (
        len(baseline_outputs) == len(cv_outputs)
        and bool(matched_items)
        and all(matched_items)
    )
    result = {
        "matched": matched,
        "matched_items": matched_items,
        "model": args.model,
        "speculator_model": args.speculator_model,
        "num_spec_tokens": args.num_spec_tokens,
        "confidence_sizing": args.confidence_sizing,
        "async_queue": args.async_queue,
        "allow_batched_prefix_verification": (
            args.allow_batched_prefix_verification
        ),
        "global_batch_barrier": args.global_batch_barrier,
        "allow_shape_drift_chunking": args.allow_shape_drift_chunking,
        "roofline_packing": args.roofline_packing,
        "util_threshold": args.util_threshold,
        "calibration_path": str(args.calibration_path)
        if args.calibration_path is not None
        else "",
        "candidate_chunks": args.candidate_chunks,
        "force_prefix_len": args.force_prefix_len,
        "dense_realign_steps": args.dense_realign_steps,
        "prefix_reject_dense_realign_steps": (
            args.prefix_reject_dense_realign_steps
        ),
        "prefix_no_kv_write": args.prefix_no_kv_write,
        "confirmation_full_active_set": args.confirmation_full_active_set,
        "lockstep_iteration_barrier": args.lockstep_iteration_barrier,
        "prefix_probe_block_rollback": args.prefix_probe_block_rollback,
        "draft_accept_eps": args.draft_accept_eps,
        "greedy_eps": args.greedy_eps,
        "confirm_prefix_reject_one_shot": args.confirm_prefix_reject_one_shot,
        "confirm_prefix_accept_one_shot": args.confirm_prefix_accept_one_shot,
        "prefix_low_margin_fallback_threshold": (
            args.prefix_low_margin_fallback_threshold
        ),
        "batch_wide_low_margin_fallback": args.batch_wide_low_margin_fallback,
        "batch_wide_prefix_reject_fallback": (
            args.batch_wide_prefix_reject_fallback
        ),
        "debug_dump": args.debug_dump,
        "max_tokens": args.max_tokens,
        "max_num_seqs": args.max_num_seqs,
        "chat_completions": args.chat_completions,
        "extra_env": args.env,
        "prompt": args.prompt,
        "prompts_jsonl": str(args.prompts_jsonl) if args.prompts_jsonl else "",
        "num_prompts": len(baseline_outputs),
        "mismatches": mismatches,
        "baseline": baseline,
        "speclink_cv": speclink_cv,
        "event_jsonl": str(args.event_jsonl),
        "baseline_event_jsonl": str(args.baseline_event_jsonl),
        "profile_jsonl": str(args.profile_jsonl) if args.profile_jsonl else "",
        "output_json": str(args.output_json),
    }
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ("matched", "output_json")}))
    print(f"matched={matched} output_json={args.output_json}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
