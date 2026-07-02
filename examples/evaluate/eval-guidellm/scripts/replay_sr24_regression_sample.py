#!/usr/bin/env python3
"""Replay one lm-eval sample with dense EAGLE3 and SR24 variants.

This diagnostic reads the exact prompt stored in an lm-eval `samples_*.jsonl`
file and runs one vLLM offline generation mode per process.  Use separate
processes for dense, all-corrected, and selective SR24 so environment-gated
model rewriting cannot leak across modes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from run_structured_24_spec_quality import (  # noqa: E402
    DEFAULT_BASE_MODELS,
    EAGLE3_SPECULATORS,
)


MODE_CHOICES = {
    "dense_baseline",
    "all_corrected_fastpath",
    "all_corrected_real",
    "selective_prefix4",
}
SR24_PRESETS = {
    "manual",
    "criticalprefix4_bucket16_directcslt",
    "mlpall_lowconf_prefix5_tritonoverride",
}


TASK_STOPS = {
    "gsm8k_cot": ["Q:", "</s>", "<|im_end|>"],
    "minerva_math500": ["Problem:"],
}


def _bool_env(value: bool) -> str:
    return "1" if value else "0"


def load_sample(samples_path: Path, doc_id: int) -> dict[str, Any]:
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("doc_id", -1)) == int(doc_id):
                return row
    raise SystemExit(f"doc_id={doc_id} not found in {samples_path}")


def parse_doc_ids(args: argparse.Namespace) -> list[int]:
    if not args.doc_ids:
        return [int(args.doc_id)]
    out: list[int] = []
    for part in args.doc_ids.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("--doc-ids did not contain any ids")
    return out


def sample_prompt(sample: dict[str, Any]) -> str:
    args = sample.get("arguments")
    if isinstance(args, dict):
        gen_args = args.get("gen_args_0")
        if isinstance(gen_args, dict) and "arg_0" in gen_args:
            return str(gen_args["arg_0"])
    prompt = sample.get("prompt")
    if prompt is not None:
        return str(prompt)
    raise SystemExit("sample does not contain arguments.gen_args_0.arg_0 or prompt")


def clear_sr24_env() -> None:
    prefixes = (
        "SPECLINK_SR24_",
        "SPECLINK_STRUCTURED_24_",
        "SPECLINK_TOKEN_DENSE_",
        "SPECLINK_TRACE_",
    )
    for key in list(os.environ):
        if key.startswith(prefixes):
            os.environ.pop(key, None)
    os.environ["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
    os.environ["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    os.environ["SPECLINK_SR24_ENABLE"] = "0"


def configure_sr24_env(args: argparse.Namespace, output_dir: Path) -> None:
    clear_sr24_env()
    if args.enable_confidence_trace:
        os.environ["SPECLINK_TRACE_CONFIDENCE"] = "1"
        os.environ["SPECLINK_TRACE_OUTPUT"] = str(
            (output_dir / "speclink_confidence_trace.jsonl").resolve()
        )
        os.environ["SPECLINK_TRACE_RUN_ID"] = (
            f"{args.model_label}_{args.task}_doc{args.doc_id}_{args.mode}"
        )
        os.environ["SPECLINK_TRACE_MODEL_LABEL"] = args.model_label
        os.environ["SPECLINK_TRACE_DATASET_LABEL"] = args.task
        os.environ["SPECLINK_TRACE_METHOD"] = "eagle3"
    mode = args.mode
    if mode == "dense_baseline":
        return

    sr24_mode = "all_corrected" if mode.startswith("all_corrected") else "selective"
    os.environ["SPECLINK_SR24_ENABLE"] = "1"
    os.environ["SPECLINK_SR24_MODE"] = sr24_mode
    os.environ["SPECLINK_SR24_BACKEND"] = args.sr24_backend
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = args.sr24_residual_backend
    os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = args.sr24_residual_device
    os.environ["SPECLINK_SR24_THRESHOLD"] = str(args.sr24_threshold)
    dense_fastpath = args.sr24_all_corrected_dense_fastpath
    if dense_fastpath is None:
        dense_fastpath = (
            mode == "all_corrected_fastpath"
            or args.sr24_static_all_residual_dense_fastpath
        )
    os.environ["SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH"] = _bool_env(
        dense_fastpath
    )
    os.environ["SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE"] = _bool_env(
        args.sr24_full_residual_early_dense
    )
    os.environ["SPECLINK_SR24_STATIC_ALL_RESIDUAL_DENSE_FASTPATH"] = _bool_env(
        args.sr24_static_all_residual_dense_fastpath
    )
    os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "1"
    os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = (
        args.sr24_selective_non_draft_policy
    )
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = (
        args.sr24_selective_residual_policy
    )
    os.environ["SPECLINK_SR24_SELECTIVE_EXTRA_AFTER_LOW"] = str(
        args.sr24_selective_extra_after_low
    )
    os.environ["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = str(
        args.sr24_selective_min_prefix_residual
    )
    os.environ["SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS"] = str(
        args.sr24_selective_max_residual_draft_rows
    )
    os.environ["SPECLINK_SR24_LOW_CONFIDENCE_CAP_BY_RISK"] = _bool_env(
        args.sr24_low_confidence_cap_by_risk
    )
    os.environ["SPECLINK_SR24_EARLY_DENSE_TOKENS"] = str(
        args.sr24_early_dense_tokens
    )
    os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = _bool_env(args.sr24_reduce_cpu_sync)
    os.environ["SPECLINK_SR24_SYNC_MASK_STATE"] = _bool_env(args.sr24_sync_mask_state)
    os.environ["SPECLINK_SR24_STATIC_MASK_STATE"] = args.sr24_static_mask_state
    os.environ["SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED"] = _bool_env(
        args.sr24_force_cudagraph_none_for_mixed
    )
    os.environ["SPECLINK_SR24_STATIC_MASK_BUFFER"] = _bool_env(
        args.sr24_static_mask_buffer
    )
    os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = _bool_env(
        args.sr24_batched_mask_builder
    )
    os.environ["SPECLINK_SR24_RESIDUAL_BUCKET_SIZE"] = str(
        args.sr24_residual_bucket_size
    )
    os.environ["SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY"] = _bool_env(
        args.sr24_residual_bucket_priority
    )
    os.environ["SPECLINK_SR24_BONUS_PRIORITY"] = str(args.sr24_bonus_priority)
    os.environ["SPECLINK_SR24_DRAFT_POSITION_PRIORITY_SCALE"] = str(
        args.sr24_draft_position_priority_scale
    )
    os.environ["SPECLINK_SR24_CUDAGRAPH_BUCKET"] = _bool_env(
        args.sr24_cudagraph_bucket
    )
    os.environ["SPECLINK_SR24_DIRECT_CSLT_LINEAR"] = _bool_env(
        args.sr24_direct_cslt_linear
    )
    os.environ["SPECLINK_SR24_BUCKET_DENSE_COPY"] = _bool_env(
        args.sr24_bucket_dense_copy
    )
    os.environ["SPECLINK_SR24_BUCKET_DENSE_COPY_ACTIVE_ONLY"] = _bool_env(
        args.sr24_bucket_dense_copy_active_only
    )
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP"] = _bool_env(args.sr24_row_routed_mlp)
    os.environ["SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR"] = _bool_env(
        args.sr24_row_routed_down_linear
    )
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT"] = _bool_env(
        args.sr24_row_routed_mlp_reuse_base_output
    )
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS"] = str(
        args.sr24_row_routed_mlp_min_dense_rows
    )
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS"] = str(
        args.sr24_row_routed_mlp_max_dense_rows
    )
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS"] = str(
        args.sr24_row_routed_mlp_max_base_rows
    )
    os.environ["SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS"] = _bool_env(
        args.sr24_route_all_residual_rows
    )
    os.environ["SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT"] = _bool_env(
        args.sr24_route_reuse_base_output
    )
    os.environ["SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION"] = str(
        args.sr24_route_dense_fallback_fraction
    )
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK"] = _bool_env(
        args.sr24_adaptive_dense_fallback
    )
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS"] = str(
        args.sr24_adaptive_dense_fallback_small_rows
    )
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_GATE_UP_FRACTION"] = str(
        args.sr24_adaptive_dense_fallback_gate_up_fraction
    )
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_DOWN_FRACTION"] = str(
        args.sr24_adaptive_dense_fallback_down_fraction
    )
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL"] = (
        _bool_env(args.sr24_adaptive_dense_fallback_small_down_no_residual)
    )
    os.environ["SPECLINK_SR24_TRITON_BUCKET_OVERRIDE"] = _bool_env(
        args.sr24_triton_bucket_override
    )
    os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM"] = _bool_env(
        args.sr24_triton_bucket_dense_gemm
    )
    os.environ["SPECLINK_SR24_TRITON_BUCKET_SCATTER"] = _bool_env(
        args.sr24_triton_bucket_scatter
    )
    os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_M"] = str(
        args.sr24_triton_bucket_dense_block_m
    )
    os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_N"] = str(
        args.sr24_triton_bucket_dense_block_n
    )
    os.environ["SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_K"] = str(
        args.sr24_triton_bucket_dense_block_k
    )
    os.environ["SPECLINK_SR24_REQUIRE_GPU_RESIDUAL"] = _bool_env(
        args.sr24_require_gpu_residual
    )
    os.environ["SPECLINK_SR24_DISABLE_RUNTIME_STATS"] = _bool_env(
        args.sr24_disable_runtime_stats
    )
    os.environ["SPECLINK_SR24_STATS_INTERVAL"] = str(args.sr24_stats_interval)
    os.environ["SPECLINK_SR24_TARGET_LEAFS"] = args.sr24_target_leafs
    os.environ["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = args.sr24_residual_target_leafs
    if args.sr24_base_only_layer_ids_by_leaf:
        os.environ["SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF"] = (
            args.sr24_base_only_layer_ids_by_leaf
        )
    os.environ["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = (
        args.sr24_residual_layer_ids_by_leaf
    )
    os.environ["SPECLINK_SR24_RESIDUAL_OUT_CHUNK"] = str(args.sr24_residual_out_chunk)
    os.environ["SPECLINK_SR24_EXTRACT_CHUNK_ROWS"] = str(args.sr24_extract_chunk_rows)
    os.environ["SPECLINK_SR24_LOG"] = str((output_dir / "speclink_sr24_events.jsonl").resolve())
    os.environ["SPECLINK_SR24_STATS_PATH"] = str(
        (output_dir / "speclink_sr24_stats.json").resolve()
    )
    if args.sr24_debug_trace:
        os.environ["SPECLINK_SR24_DEBUG_TRACE"] = "1"
        os.environ["SPECLINK_SR24_DEBUG_TRACE_PATH"] = str(
            (output_dir / "speclink_sr24_debug_trace.jsonl").resolve()
        )
        if args.sr24_debug_req_substr:
            os.environ["SPECLINK_SR24_DEBUG_REQ_SUBSTR"] = (
                args.sr24_debug_req_substr
            )
    if args.sr24_mask_path:
        os.environ["SPECLINK_SR24_MASK_PATH"] = str(args.sr24_mask_path.resolve())


def _logprob_obj_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    out: dict[str, Any] = {}
    for attr in ("logprob", "rank", "decoded_token"):
        if hasattr(value, attr):
            item = getattr(value, attr)
            if hasattr(item, "item"):
                item = item.item()
            out[attr] = item
    if not out:
        try:
            out["logprob"] = float(value)
        except (TypeError, ValueError):
            out["repr"] = repr(value)
    return out


def serialize_logprobs(logprobs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not logprobs:
        return rows
    for step_idx, step in enumerate(logprobs):
        step_row: dict[str, Any] = {"position": step_idx, "tokens": {}}
        if isinstance(step, dict):
            for token_id, value in step.items():
                step_row["tokens"][str(int(token_id))] = _logprob_obj_to_dict(value)
        else:
            step_row["repr"] = repr(step)
        rows.append(step_row)
    return rows


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_compilation_config(value: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--compilation-config is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--compilation-config must decode to a JSON object")
    return parsed


def apply_sr24_preset(args: argparse.Namespace) -> None:
    """Expand SR24 presets used by the lm-eval runners.

    Keep this replay script self-contained: it should reproduce a problematic
    lm-eval sample without importing the full runner and inheriting unrelated
    process-management behavior.
    """
    preset = str(getattr(args, "sr24_preset", "manual"))
    if preset == "manual":
        return
    if preset == "criticalprefix4_bucket16_directcslt":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = (
            "gate_up_proj=16-31;down_proj=8-15"
        )
        args.sr24_selective_residual_policy = "critical_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.6
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_extra_after_low = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 16
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        if args.sr24_default_vllm_compile is None:
            args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.sr24_stats_interval = max(int(args.sr24_stats_interval), 32)
        return
    if preset == "mlpall_lowconf_prefix5_tritonoverride":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = ""
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.6
        args.sr24_selective_min_prefix_residual = 5
        args.sr24_selective_extra_after_low = 0
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
    raise SystemExit(f"unknown --sr24-preset: {preset}")


def speclink_t08_allows_cudagraph(args: argparse.Namespace) -> bool:
    allowed_states = {"all_residual", "no_residual"}
    if not args.sr24_force_cudagraph_none_for_mixed:
        allowed_states.add("mixed")
    dynamic_auto_can_use_graph = (
        args.sr24_dynamic_auto_cudagraph
        and args.sr24_static_mask_state == "auto"
        and not args.sr24_force_cudagraph_none_for_mixed
        and (
            int(args.sr24_residual_bucket_size) <= 0
            or args.sr24_cudagraph_bucket
        )
    )
    if (
        args.sr24_static_mask_state not in allowed_states
        and not dynamic_auto_can_use_graph
    ):
        return False
    return (
        args.sr24_allow_cudagraph
        and args.sr24_static_mask_buffer
        and args.sr24_reduce_cpu_sync
        and args.sr24_residual_backend in {"torch_sparse", "dense_rows"}
    )


def effective_compilation_config(args: argparse.Namespace) -> dict[str, Any] | None:
    explicit = parse_compilation_config(args.compilation_config)
    if explicit is not None:
        return explicit
    if args.sr24_default_vllm_compile:
        return None
    if args.mode != "selective_prefix4":
        return None
    if args.sr24_backend != "torch_sparse" or not speclink_t08_allows_cudagraph(args):
        return None
    verifier_tokens = int(args.max_num_seqs) * (int(args.num_spec_tokens) + 1)
    capture_size = max(1024, ((verifier_tokens + 15) // 16) * 16)
    return {
        "mode": "NONE",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "max_cudagraph_capture_size": capture_size,
    }


def run_replay(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Import vLLM only after SR24 env is configured.
    from vllm import LLM, SamplingParams, TokensPrompt

    doc_ids = parse_doc_ids(args)
    samples = [load_sample(args.samples_path, doc_id) for doc_id in doc_ids]
    prompts = [sample_prompt(sample) for sample in samples]
    stop = args.stop if args.stop else TASK_STOPS.get(args.task, [])
    model_path = args.model_path or DEFAULT_BASE_MODELS[args.model_label]
    tokenizer_path = args.tokenizer_path or model_path
    speculator_path = args.speculator_model_path or EAGLE3_SPECULATORS[args.model_label]
    speculative_config = {
        "model": speculator_path,
        "num_speculative_tokens": args.num_spec_tokens,
        "method": "eagle3",
        "max_model_len": args.max_model_len,
    }

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
        stop=stop,
        logprobs=args.logprobs,
        logprob_token_ids=parse_int_list(args.logprob_token_ids),
    )
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        generation_config="vllm",
        speculative_config=speculative_config,
        compilation_config=effective_compilation_config(args),
    )
    llm_prompts: list[str | TokensPrompt]
    prompt_token_counts: list[int | None]
    if args.tokenized_requests:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )
        max_context_len = max(1, int(args.max_model_len) - 1 - int(args.max_tokens))
        tokenized_prompts: list[TokensPrompt] = []
        prompt_token_counts = []
        for prompt in prompts:
            ids = tokenizer(
                prompt,
                add_special_tokens=bool(args.add_bos_token),
                truncation=False,
                return_attention_mask=False,
            ).input_ids
            ids = [int(token_id) for token_id in ids]
            if len(ids) > max_context_len:
                ids = ids[-max_context_len:]
            prompt_token_counts.append(len(ids))
            tokenized_prompts.append(TokensPrompt(prompt_token_ids=ids))
        llm_prompts = tokenized_prompts
    else:
        llm_prompts = prompts
        prompt_token_counts = [None] * len(prompts)
    outputs = llm.generate(llm_prompts, sampling_params)
    if not outputs:
        raise RuntimeError("vLLM returned no outputs")
    results: list[dict[str, Any]] = []
    for doc_id, sample, prompt, request_output in zip(
        doc_ids, samples, prompts, outputs, strict=True
    ):
        if not request_output.outputs:
            raise RuntimeError(f"vLLM returned no outputs for doc_id={doc_id}")
        output = request_output.outputs[0]
        token_ids = [int(tok) for tok in output.token_ids]
        results.append({
            "mode": args.mode,
            "model_label": args.model_label,
            "task": args.task,
            "doc_id": doc_id,
            "target": sample.get("target"),
            "prompt_hash": sample.get("prompt_hash"),
            "doc_hash": sample.get("doc_hash"),
            "prompt_chars": len(prompt),
            "text": output.text,
            "token_ids": token_ids,
            "finish_reason": getattr(output, "finish_reason", None),
            "stop_reason": getattr(output, "stop_reason", None),
            "cumulative_logprob": getattr(output, "cumulative_logprob", None),
            "logprobs": serialize_logprobs(getattr(output, "logprobs", None)),
            "prompt_token_count": prompt_token_counts[len(results)]
            if len(results) < len(prompt_token_counts)
            else None,
            "config": {
                "max_tokens": args.max_tokens,
                "num_spec_tokens": args.num_spec_tokens,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "vllm_cache_root": str(args.vllm_cache_root.resolve())
                if args.vllm_cache_root
                else "",
                "stop": stop,
                "tokenized_requests": args.tokenized_requests,
                "add_bos_token": args.add_bos_token,
                "sr24_backend": args.sr24_backend,
                "sr24_residual_backend": args.sr24_residual_backend,
                "sr24_base_only_layer_ids_by_leaf":
                args.sr24_base_only_layer_ids_by_leaf,
                "sr24_residual_layer_ids_by_leaf":
                args.sr24_residual_layer_ids_by_leaf,
                "sr24_selective_residual_policy":
                args.sr24_selective_residual_policy,
                "sr24_threshold": args.sr24_threshold,
                "sr24_selective_min_prefix_residual":
                args.sr24_selective_min_prefix_residual,
                "sr24_selective_max_residual_draft_rows":
                args.sr24_selective_max_residual_draft_rows,
                "sr24_low_confidence_cap_by_risk":
                args.sr24_low_confidence_cap_by_risk,
                "sr24_early_dense_tokens": args.sr24_early_dense_tokens,
                "sr24_residual_bucket_size": args.sr24_residual_bucket_size,
                "sr24_cudagraph_bucket": args.sr24_cudagraph_bucket,
                "sr24_direct_cslt_linear": args.sr24_direct_cslt_linear,
                "sr24_bucket_dense_copy": args.sr24_bucket_dense_copy,
                "sr24_bucket_dense_copy_active_only":
                args.sr24_bucket_dense_copy_active_only,
                "sr24_bonus_priority": args.sr24_bonus_priority,
                "sr24_draft_position_priority_scale":
                args.sr24_draft_position_priority_scale,
                "sr24_row_routed_mlp": args.sr24_row_routed_mlp,
                "sr24_row_routed_down_linear": args.sr24_row_routed_down_linear,
                "sr24_row_routed_mlp_reuse_base_output":
                args.sr24_row_routed_mlp_reuse_base_output,
                "sr24_row_routed_mlp_max_dense_rows":
                args.sr24_row_routed_mlp_max_dense_rows,
                "sr24_row_routed_mlp_max_base_rows":
                args.sr24_row_routed_mlp_max_base_rows,
                "sr24_adaptive_dense_fallback":
                args.sr24_adaptive_dense_fallback,
                "sr24_adaptive_dense_fallback_small_rows":
                args.sr24_adaptive_dense_fallback_small_rows,
                "sr24_adaptive_dense_fallback_gate_up_fraction":
                args.sr24_adaptive_dense_fallback_gate_up_fraction,
                "sr24_adaptive_dense_fallback_down_fraction":
                args.sr24_adaptive_dense_fallback_down_fraction,
                "sr24_adaptive_dense_fallback_small_down_no_residual":
                args.sr24_adaptive_dense_fallback_small_down_no_residual,
                "sr24_triton_bucket_override":
                args.sr24_triton_bucket_override,
                "sr24_triton_bucket_dense_gemm":
                args.sr24_triton_bucket_dense_gemm,
                "sr24_triton_bucket_scatter":
                args.sr24_triton_bucket_scatter,
                "sr24_all_corrected_dense_fastpath":
                args.sr24_all_corrected_dense_fastpath,
                "sr24_full_residual_early_dense":
                args.sr24_full_residual_early_dense,
                "sr24_static_all_residual_dense_fastpath":
                args.sr24_static_all_residual_dense_fastpath,
                "sr24_default_vllm_compile": args.sr24_default_vllm_compile,
                "sr24_dynamic_auto_cudagraph": args.sr24_dynamic_auto_cudagraph,
                "sr24_disable_runtime_stats": args.sr24_disable_runtime_stats,
                "effective_compilation_config": effective_compilation_config(args),
                "enforce_eager": args.enforce_eager,
            },
        })
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--doc-id", type=int, required=True)
    parser.add_argument(
        "--doc-ids",
        default="",
        help=(
            "Optional comma-separated doc ids to replay in one vLLM process. "
            "--doc-id is still used for backward-compatible output naming when "
            "this is empty."
        ),
    )
    parser.add_argument("--task", default="gsm8k_cot")
    parser.add_argument("--mode", choices=sorted(MODE_CHOICES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="llama3_1_8b")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--speculator-model-path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help=(
            "vLLM max_num_seqs for offline replay. Set this to the serving "
            "runner's max_num_seqs when checking scheduler-shape effects."
        ),
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--logprob-token-ids", default="")
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument(
        "--tokenized-requests",
        action="store_true",
        help=(
            "Mirror lm-eval local-completions tokenized_requests=True: encode "
            "the prompt with the HF tokenizer, left-truncate to "
            "max_model_len - 1 - max_tokens, and pass TokensPrompt to vLLM."
        ),
    )
    parser.add_argument(
        "--add-bos-token",
        action="store_true",
        help="Add special/BOS tokens when --tokenized-requests is enabled.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--compilation-config", default="")
    parser.add_argument(
        "--vllm-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional per-replay VLLM_CACHE_ROOT. Use this to isolate "
            "torch.compile/CUDA Graph artifacts when testing SR24 graph "
            "correctness."
        ),
    )
    parser.add_argument("--sr24-backend", default="torch_sparse")
    parser.add_argument(
        "--sr24-preset",
        choices=sorted(SR24_PRESETS),
        default="manual",
        help="Expand one of the lm-eval SR24 presets before replaying.",
    )
    parser.add_argument("--sr24-residual-backend", default="dense_rows")
    parser.add_argument("--sr24-residual-device", default="auto")
    parser.add_argument("--sr24-threshold", type=float, default=0.4)
    parser.add_argument("--sr24-target-leafs", default="gate_up_proj")
    parser.add_argument("--sr24-residual-target-leafs", default="gate_up_proj")
    parser.add_argument(
        "--sr24-base-only-layer-ids-by-leaf",
        default="",
    )
    parser.add_argument(
        "--sr24-residual-layer-ids-by-leaf",
        default="gate_up_proj=16-31",
    )
    parser.add_argument("--sr24-residual-out-chunk", type=int, default=4096)
    parser.add_argument("--sr24-extract-chunk-rows", type=int, default=128)
    parser.add_argument("--sr24-selective-non-draft-policy", default="bonus")
    parser.add_argument("--sr24-selective-residual-policy", default="all_if_any_low")
    parser.add_argument("--sr24-selective-extra-after-low", type=int, default=0)
    parser.add_argument("--sr24-selective-min-prefix-residual", type=int, default=4)
    parser.add_argument("--sr24-selective-max-residual-draft-rows", type=int, default=0)
    parser.add_argument("--sr24-low-confidence-cap-by-risk", action="store_true")
    parser.add_argument("--sr24-early-dense-tokens", type=int, default=0)
    parser.add_argument("--sr24-reduce-cpu-sync", action="store_true")
    parser.add_argument(
        "--sr24-sync-mask-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sr24-static-mask-state", default="auto")
    parser.add_argument(
        "--sr24-force-cudagraph-none-for-mixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep mixed SR24 verify plans on eager CUDA Graph NONE by default. "
            "Use --no-sr24-force-cudagraph-none-for-mixed only for controlled "
            "graph-correctness replay."
        ),
    )
    parser.add_argument(
        "--sr24-all-corrected-dense-fastpath",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH. By default "
            "the replay enables it for all_corrected_fastpath and for the "
            "static-all-residual dense-fastpath diagnostic."
        ),
    )
    parser.add_argument(
        "--sr24-static-all-residual-dense-fastpath",
        action="store_true",
        help=(
            "Diagnostic for selective + static all_residual: keep attached "
            "modules on their original dense vLLM Linear path instead of "
            "rewriting them to sparse+residual hooks."
        ),
    )
    parser.add_argument(
        "--sr24-full-residual-early-dense",
        action="store_true",
        help=(
            "When a dynamic selective step is all-residual, run dense target "
            "Linear directly instead of sparse base plus dense-row correction."
        ),
    )
    parser.add_argument("--sr24-static-mask-buffer", action="store_true")
    parser.add_argument("--sr24-batched-mask-builder", action="store_true")
    parser.add_argument("--sr24-residual-bucket-size", type=int, default=0)
    parser.add_argument("--sr24-residual-bucket-priority", action="store_true")
    parser.add_argument("--sr24-direct-cslt-linear", action="store_true")
    parser.add_argument("--sr24-bucket-dense-copy", action="store_true")
    parser.add_argument("--sr24-bucket-dense-copy-active-only", action="store_true")
    parser.add_argument("--sr24-bonus-priority", type=float, default=4.0)
    parser.add_argument("--sr24-draft-position-priority-scale",
                        type=float,
                        default=0.0)
    parser.add_argument(
        "--sr24-cudagraph-bucket",
        action="store_true",
        help=(
            "Experimental graph-correctness diagnostic. Allow selective SR24 "
            "residual buckets to use persistent CUDA Graph buffers. This is "
            "off by default because GSM8K quality probes currently show "
            "regressions."
        ),
    )
    parser.add_argument("--sr24-row-routed-mlp", action="store_true")
    parser.add_argument("--sr24-row-routed-down-linear", action="store_true")
    parser.add_argument("--sr24-row-routed-mlp-reuse-base-output", action="store_true")
    parser.add_argument("--sr24-row-routed-mlp-min-dense-rows", type=int, default=128)
    parser.add_argument("--sr24-row-routed-mlp-max-dense-rows", type=int, default=0)
    parser.add_argument("--sr24-row-routed-mlp-max-base-rows", type=int, default=0)
    parser.add_argument("--sr24-route-all-residual-rows", action="store_true")
    parser.add_argument("--sr24-route-reuse-base-output", action="store_true")
    parser.add_argument("--sr24-route-dense-fallback-fraction", type=float, default=1.1)
    parser.add_argument("--sr24-adaptive-dense-fallback", action="store_true")
    parser.add_argument("--sr24-adaptive-dense-fallback-small-rows", type=int, default=128)
    parser.add_argument("--sr24-adaptive-dense-fallback-gate-up-fraction", type=float, default=0.10)
    parser.add_argument("--sr24-adaptive-dense-fallback-down-fraction", type=float, default=0.25)
    parser.add_argument(
        "--no-sr24-adaptive-dense-fallback-small-down-no-residual",
        dest="sr24_adaptive_dense_fallback_small_down_no_residual",
        action="store_false",
    )
    parser.set_defaults(sr24_adaptive_dense_fallback_small_down_no_residual=True)
    parser.add_argument("--sr24-triton-bucket-override", action="store_true")
    parser.add_argument("--sr24-triton-bucket-dense-gemm", action="store_true")
    parser.add_argument("--sr24-triton-bucket-scatter", action="store_true")
    parser.add_argument("--sr24-triton-bucket-dense-block-m", type=int, default=16)
    parser.add_argument("--sr24-triton-bucket-dense-block-n", type=int, default=32)
    parser.add_argument("--sr24-triton-bucket-dense-block-k", type=int, default=128)
    parser.add_argument("--sr24-require-gpu-residual", action="store_true")
    parser.add_argument("--sr24-allow-cudagraph", action="store_true")
    parser.add_argument(
        "--sr24-default-vllm-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--sr24-dynamic-auto-cudagraph", action="store_true")
    parser.add_argument("--sr24-disable-runtime-stats", action="store_true")
    parser.add_argument("--sr24-stats-interval", type=int, default=1)
    parser.add_argument("--sr24-mask-path", type=Path, default=None)
    parser.add_argument(
        "--sr24-debug-trace",
        action="store_true",
        help=(
            "Write speclink_sr24_debug_trace.jsonl with per-request draft "
            "scores and residual/base routing decisions."
        ),
    )
    parser.add_argument(
        "--sr24-debug-req-substr",
        default="",
        help=(
            "Optional substring filter for SR24 debug trace request ids. "
            "Leave empty to trace every request in this single-sample replay."
        ),
    )
    parser.add_argument(
        "--enable-confidence-trace",
        action="store_true",
        help=(
            "Also enable SPECLINK_TRACE_CONFIDENCE and write token-level "
            "acceptance/confidence records when rejection-sampler labels are "
            "available."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_sr24_preset(args)
    if args.sr24_default_vllm_compile is None:
        args.sr24_default_vllm_compile = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.vllm_cache_root:
        args.vllm_cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = str(args.vllm_cache_root.resolve())
    configure_sr24_env(args, args.output_dir)
    results = run_replay(args)
    if len(results) == 1:
        result = results[0]
        out_path = args.output_dir / (
            f"{args.task}_doc{result['doc_id']}_{args.mode}.json"
        )
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(out_path.resolve())
        return
    written: list[str] = []
    for result in results:
        out_path = args.output_dir / (
            f"{args.task}_doc{result['doc_id']}_{args.mode}.json"
        )
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(str(out_path.resolve()))
    summary_path = args.output_dir / f"{args.task}_docs_{args.mode}.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for path in written:
        print(path)
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
