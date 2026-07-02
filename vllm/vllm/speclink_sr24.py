# SPDX-License-Identifier: Apache-2.0
"""SpecLink selective residual 2:4 support.

This is an env-gated correctness-first implementation for the local SpecLink
experiments. It rewrites target-model linear weights into the 2:4 base part at
model-load time and keeps only the complementary residual values plus the base
mask for optional per-token correction.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import threading
import time
import atexit
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.speclink_confidence_trace import (
    enabled as _confidence_trace_enabled,
    record_sr24_verify_mask as _trace_record_sr24_verify_mask,
)


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)
_BIT_COUNTS = torch.tensor(
    [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4],
    dtype=torch.int16,
)

TARGET_LEAFS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
}

SR24_BACKENDS = {"dense_zero", "prototype", "torch_sparse"}
DENSE_ZERO_BACKENDS = {"dense_zero", "prototype"}

_TARGET_LEAFS_CSV = ",".join(sorted(TARGET_LEAFS))

FUSED_CACHE_LEAFS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}

_propose_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "speclink_sr24_propose_context", default=None
)
_verify_residual_mask: ContextVar[torch.Tensor | None] = ContextVar(
    "speclink_sr24_verify_residual_mask", default=None
)
_verify_residual_state: ContextVar[str | None] = ContextVar(
    "speclink_sr24_verify_residual_state", default=None
)
_verify_residual_priority: ContextVar[torch.Tensor | None] = ContextVar(
    "speclink_sr24_verify_residual_priority", default=None
)
_verify_residual_bucket: ContextVar[tuple[torch.Tensor, torch.Tensor] | None] = (
    ContextVar("speclink_sr24_verify_residual_bucket", default=None)
)
_verify_residual_rows: ContextVar[torch.Tensor | None] = ContextVar(
    "speclink_sr24_verify_residual_rows", default=None
)
_verify_base_rows: ContextVar[torch.Tensor | None] = ContextVar(
    "speclink_sr24_verify_base_rows", default=None
)
_verify_fixed_prefix_route: ContextVar["FixedPrefixRouteDescriptor | None"] = (
    ContextVar("speclink_sr24_fixed_prefix_route", default=None)
)
_fast_verify_residual_active = False
_fast_verify_residual_mask: torch.Tensor | None = None
_fast_verify_residual_state: str | None = None
_fast_verify_residual_priority: torch.Tensor | None = None
_fast_verify_residual_bucket: tuple[torch.Tensor, torch.Tensor] | None = None
_fast_verify_residual_rows: torch.Tensor | None = None
_fast_verify_base_rows: torch.Tensor | None = None
_fast_verify_fixed_prefix_route: "FixedPrefixRouteDescriptor | None" = None
_fixed_block_input_buffer_cache: dict[
    tuple[str, str, int, torch.dtype, int, int], torch.Tensor
] = {}


@dataclass(frozen=True)
class FixedPrefixRouteDescriptor:
    active_count: int
    scheduled_width: int
    valid_width: int
    prefix: int
    dense_width: int
    base_width: int


@dataclass(frozen=True)
class VerifyResidualPlan:
    mask: torch.Tensor | None
    state: str
    priority: torch.Tensor | None = None
    bucket: tuple[torch.Tensor, torch.Tensor] | None = None
    residual_rows: torch.Tensor | None = None
    base_rows: torch.Tensor | None = None
    fixed_prefix_route: FixedPrefixRouteDescriptor | None = None

_pending_scores: defaultdict[str, deque[torch.Tensor]] = defaultdict(deque)
_pending_generated_lens: defaultdict[str, deque[int | None]] = defaultdict(deque)
_device_constant_cache: dict[tuple[str, str], torch.Tensor] = {}
_device_arange_cache: dict[tuple[str, str], torch.Tensor] = {}
_static_mask_buffers: dict[str, torch.Tensor] = {}
_static_priority_buffers: dict[str, torch.Tensor] = {}
_static_long_buffers: dict[tuple[str, str], torch.Tensor] = {}
_static_float_buffers: dict[tuple[str, str], torch.Tensor] = {}
_static_int32_buffers: dict[tuple[str, str], torch.Tensor] = {}
_static_int32_cpu_buffers: dict[str, torch.Tensor] = {}
_static_long_cpu_buffers: dict[str, torch.Tensor] = {}
_route_overlap_stream_cache: dict[str, tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}
_lock = threading.Lock()
_breakdown_lock = threading.Lock()
_grouping_trace_lock = threading.Lock()
_grouped_queue_shadow_lock = threading.RLock()
_grouped_queue_shadow_next_index = 0
_grouped_queue_shadow_by_key: defaultdict[
    tuple[Any, ...], deque[dict[str, Any]]
] = defaultdict(deque)
_grouped_queue_shadow_signal_handlers_installed = False
_grouped_queue_shadow_previous_signal_handlers: dict[int, Any] = {}
_stats_accum: dict[str, Any] = {
    "steps": 0,
    "total_scheduled_tokens": 0,
    "total_draft_tokens": 0,
    "total_valid_draft_tokens": 0,
    "non_draft_tokens": 0,
    "residual_draft_tokens": 0,
    "base_only_draft_tokens": 0,
    "residual_non_draft_tokens": 0,
    "base_only_non_draft_tokens": 0,
    "early_residual_draft_tokens": 0,
    "early_residual_non_draft_tokens": 0,
    "missing_score_tokens": 0,
    "bucket_calls": 0,
    "bucket_candidate_rows": 0,
    "bucket_active_rows": 0,
    "bucket_total_rows": 0,
    "bucket_residual_requested_rows": 0,
    "adaptive_dense_fallback_calls": 0,
    "adaptive_dense_fallback_rows": 0,
    "adaptive_dense_fallback_candidate_rows": 0,
    "stats_exact": True,
    "sync_reduced_stats": False,
    "last_flush_steps": 0,
    "cudagraph_mode_counts": {},
    "cudagraph_steps": 0,
    "last_cudagraph_flush_steps": 0,
    "scheduler_mask_wall_cpu_ms": 0.0,
    "scheduler_materialize_counts_wall_cpu_ms": 0.0,
    "scheduler_pending_scores_pop_wall_cpu_ms": 0.0,
    "scheduler_batched_mask_builder_wall_cpu_ms": 0.0,
    "scheduler_request_routing_loop_wall_cpu_ms": 0.0,
    "scheduler_batch_all_apply_wall_cpu_ms": 0.0,
    "scheduler_mask_state_wall_cpu_ms": 0.0,
    "scheduler_static_mask_copy_wall_cpu_ms": 0.0,
    "scheduler_row_index_bucket_wall_cpu_ms": 0.0,
    "scheduler_residual_bucket_wall_cpu_ms": 0.0,
    "scheduler_mixed_row_indices_wall_cpu_ms": 0.0,
    "scheduler_direct_cpu_route_rows_wall_cpu_ms": 0.0,
}
_RUNTIME_TIMING_KEYS = (
    "scheduler_mask_wall_cpu_ms",
    "scheduler_materialize_counts_wall_cpu_ms",
    "scheduler_pending_scores_pop_wall_cpu_ms",
    "scheduler_batched_mask_builder_wall_cpu_ms",
    "scheduler_request_routing_loop_wall_cpu_ms",
    "scheduler_batch_all_apply_wall_cpu_ms",
    "scheduler_mask_state_wall_cpu_ms",
    "scheduler_static_mask_copy_wall_cpu_ms",
    "scheduler_row_index_bucket_wall_cpu_ms",
    "scheduler_residual_bucket_wall_cpu_ms",
    "scheduler_mixed_row_indices_wall_cpu_ms",
    "scheduler_direct_cpu_route_rows_wall_cpu_ms",
)
_generic_cudagraph_accum: dict[str, Any] = {
    "cudagraph_mode_counts": {},
    "cudagraph_shape_counts": {},
    "cudagraph_steps": 0,
    "last_flush_steps": 0,
}
_breakdown_accum: dict[str, Any] = {
    "cpu_ms": defaultdict(float),
    "cpu_calls": defaultdict(int),
    "cuda_ms": defaultdict(float),
    "cuda_calls": defaultdict(int),
    "cuda_events": defaultdict(int),
    "counts": defaultdict(float),
    "pending_cuda": [],
    "pending_cuda_events": 0,
    "flushes": 0,
    "last_snapshot_steps": 0,
}
_breakdown_gpu_counts: dict[tuple[str, str], torch.Tensor] = {}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in _TRUTHY


def _generic_cudagraph_stats_path() -> str:
    return os.getenv("SPECLINK_CUDAGRAPH_STATS_PATH", "").strip()


def _generic_cudagraph_stats_interval() -> int:
    raw = os.getenv("SPECLINK_CUDAGRAPH_STATS_INTERVAL", "32").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 32


def _write_generic_cudagraph_event(event: dict[str, Any]) -> None:
    path = _generic_cudagraph_stats_path()
    if not path:
        return
    try:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        pass


def _route_overlap_streams_for_device(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    key = str(device)
    streams = _route_overlap_stream_cache.get(key)
    if streams is not None:
        return streams
    with _lock:
        streams = _route_overlap_stream_cache.get(key)
        if streams is None:
            streams = (
                torch.cuda.Stream(device=device),
                torch.cuda.Stream(device=device),
            )
            _route_overlap_stream_cache[key] = streams
    return streams


def _record_generic_cudagraph_mode(key: str, shape_key: str | None = None) -> None:
    if not _generic_cudagraph_stats_path():
        return
    event: dict[str, Any] | None = None
    with _lock:
        counts = _generic_cudagraph_accum.setdefault("cudagraph_mode_counts", {})
        counts[key] = int(counts.get(key, 0)) + 1
        if shape_key:
            shape_counts = _generic_cudagraph_accum.setdefault(
                "cudagraph_shape_counts", {}
            )
            shape_counts[shape_key] = int(shape_counts.get(shape_key, 0)) + 1
        _generic_cudagraph_accum["cudagraph_steps"] = int(
            _generic_cudagraph_accum.get("cudagraph_steps") or 0
        ) + 1
        steps = int(_generic_cudagraph_accum["cudagraph_steps"])
        last_flush = int(_generic_cudagraph_accum.get("last_flush_steps") or 0)
        if steps - last_flush >= _generic_cudagraph_stats_interval():
            _generic_cudagraph_accum["last_flush_steps"] = steps
            event = {
                "timestamp": time.time(),
                "event": "speclink_cudagraph_summary",
                "cudagraph_steps": steps,
                "cudagraph_mode_counts": dict(counts),
                "cudagraph_shape_counts": dict(
                    _generic_cudagraph_accum.get("cudagraph_shape_counts") or {}
                ),
            }
    if event is not None:
        _write_generic_cudagraph_event(event)


def _flush_generic_cudagraph_stats() -> None:
    if not _generic_cudagraph_stats_path():
        return
    with _lock:
        steps = int(_generic_cudagraph_accum.get("cudagraph_steps") or 0)
        if steps <= int(_generic_cudagraph_accum.get("last_flush_steps") or 0):
            return
        _generic_cudagraph_accum["last_flush_steps"] = steps
        event = {
            "timestamp": time.time(),
            "event": "speclink_cudagraph_summary",
            "cudagraph_steps": steps,
            "cudagraph_mode_counts": dict(
                _generic_cudagraph_accum.get("cudagraph_mode_counts") or {}
            ),
            "cudagraph_shape_counts": dict(
                _generic_cudagraph_accum.get("cudagraph_shape_counts") or {}
            ),
        }
    _write_generic_cudagraph_event(event)


atexit.register(_flush_generic_cudagraph_stats)


def enabled() -> bool:
    return _env_flag("SPECLINK_SR24_ENABLE")


def _base_only_scope_configured() -> bool:
    return bool(
        os.getenv("SPECLINK_SR24_BASE_ONLY_LAYER_IDS", "").strip()
        or os.getenv("SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF", "").strip()
    )


def _base_only_dense_verify_scope_configured() -> bool:
    return bool(
        os.getenv("SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_LAYER_IDS", "").strip()
        or os.getenv(
            "SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_LAYER_IDS_BY_LEAF", ""
        ).strip()
    )


def linear_hooks_enabled() -> bool:
    if not enabled():
        return False
    if (
        mode() == "selective"
        and static_mask_state() == "all_residual"
        and static_all_residual_dense_fastpath()
        and all_corrected_dense_fastpath()
        and not _base_only_scope_configured()
    ):
        # Diagnostic dense no-op path: keep original vLLM Linear execution and
        # skip SR24 verify masks/contexts entirely.
        return False
    return not (
        mode() == "all_corrected"
        and backend() == "torch_sparse"
        and all_corrected_dense_fastpath()
    )


def draft_scores_enabled() -> bool:
    """Whether SR24 needs DLM token-confidence scores in the propose pass."""
    if not linear_hooks_enabled():
        return False
    # Static all/no-residual modes do not inspect per-token DLM confidence in
    # build_verify_residual_mask(), so avoid the extra draft-logit log_softmax
    # and Python queueing in the proposer hot path.
    if static_mask_state() in {"all_residual", "no_residual"}:
        return False
    if selective_residual_policy() == "fixed_prefix":
        return False
    return mode() == "selective"


def early_dense_tokens() -> int:
    raw = os.getenv("SPECLINK_SR24_EARLY_DENSE_TOKENS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def dense_fallback_nonuniform() -> bool:
    return _env_flag("SPECLINK_SR24_DENSE_FALLBACK_NONUNIFORM", "0")


def needs_length_context() -> bool:
    return enabled() and linear_hooks_enabled() and (
        early_dense_tokens() > 0 or debug_trace_enabled()
    )


def mode() -> str:
    return os.getenv("SPECLINK_SR24_MODE", "selective").strip()


def threshold() -> float:
    return float(os.getenv("SPECLINK_SR24_THRESHOLD", "0.8"))


def backend() -> str:
    return os.getenv("SPECLINK_SR24_BACKEND", "prototype").strip().lower()


def residual_backend() -> str:
    return os.getenv(
        "SPECLINK_SR24_RESIDUAL_BACKEND", "compressed_dense"
    ).strip().lower()


def residual_backend_by_leaf() -> dict[str, str]:
    raw = os.getenv("SPECLINK_SR24_RESIDUAL_BACKEND_BY_LEAF", "").strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for item in raw.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(
                "SPECLINK_SR24_RESIDUAL_BACKEND_BY_LEAF entries must be "
                f"leaf=backend; got {item!r}"
            )
        leaf, backend_value = item.split("=", 1)
        leaf = leaf.strip()
        backend_value = backend_value.strip().lower()
        if backend_value not in {"compressed_dense", "torch_sparse", "dense_rows"}:
            raise RuntimeError(
                "unsupported per-leaf SR24 residual backend "
                f"{backend_value!r} for {leaf!r}"
            )
        out[leaf] = backend_value
    return out


def residual_device() -> str:
    raw = os.getenv("SPECLINK_SR24_RESIDUAL_DEVICE", "auto").strip().lower()
    if raw != "auto":
        return raw
    return "cuda"


def require_gpu_residual() -> bool:
    return _env_flag("SPECLINK_SR24_REQUIRE_GPU_RESIDUAL", "0")


def reduce_cpu_sync() -> bool:
    return _env_flag("SPECLINK_SR24_REDUCE_CPU_SYNC")


def sync_mask_state() -> bool:
    return _env_flag("SPECLINK_SR24_SYNC_MASK_STATE", "1")


def static_mask_state() -> str:
    raw = os.getenv("SPECLINK_SR24_STATIC_MASK_STATE", "auto").strip().lower()
    if raw == "":
        raw = "auto"
    if raw not in {"auto", "all_residual", "no_residual", "mixed"}:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_STATIC_MASK_STATE="
            f"{raw}; expected auto, all_residual, no_residual, or mixed"
        )
    return raw


def static_all_residual_dense_fastpath() -> bool:
    return _env_flag("SPECLINK_SR24_STATIC_ALL_RESIDUAL_DENSE_FASTPATH", "0")


def direct_cslt_linear() -> bool:
    return _env_flag("SPECLINK_SR24_DIRECT_CSLT_LINEAR", "0")


def cslt_small_m_alg_id_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_CSLT_SMALL_M_ALG_ID_ENABLE", "0")


def cslt_small_m_threshold() -> int:
    raw = os.getenv("SPECLINK_SR24_CSLT_SMALL_M_THRESHOLD", "96").strip()
    return max(0, int(raw or "0"))


def cslt_small_m_alg_id() -> int:
    raw = os.getenv("SPECLINK_SR24_CSLT_SMALL_M_ALG_ID", "1").strip()
    return max(0, int(raw or "0"))


def cslt_small_m_threshold_by_leaf() -> dict[str, int]:
    return _leaf_int_map_from_env("SPECLINK_SR24_CSLT_SMALL_M_THRESHOLD_BY_LEAF")


def cslt_small_m_alg_id_by_leaf() -> dict[str, int]:
    return _leaf_int_map_from_env("SPECLINK_SR24_CSLT_SMALL_M_ALG_ID_BY_LEAF")


def _cslt_alg_id_for_rows(weight: Any, rows: int) -> int:
    default_alg_id = int(getattr(weight, "alg_id_cusparselt", 0))
    leaf = str(getattr(weight, "_speclink_sr24_profile_leaf", "") or "")
    if leaf:
        leaf_algs = cslt_small_m_alg_id_by_leaf()
        if leaf in leaf_algs:
            threshold = cslt_small_m_threshold_by_leaf().get(
                leaf,
                cslt_small_m_threshold(),
            )
            if int(rows) <= max(0, int(threshold)):
                return max(0, int(leaf_algs[leaf]))
    if (
        cslt_small_m_alg_id_enabled()
        and int(rows) <= cslt_small_m_threshold()
    ):
        return cslt_small_m_alg_id()
    return default_alg_id


def gate_up_split() -> str:
    raw = os.getenv("SPECLINK_SR24_GATE_UP_SPLIT", "none").strip().lower()
    aliases = {
        "": "none",
        "0": "none",
        "false": "none",
        "none": "none",
        "off": "none",
        "up": "up_sparse",
        "sparse_up": "up_sparse",
        "up_sparse": "up_sparse",
        "gate": "gate_sparse",
        "sparse_gate": "gate_sparse",
        "gate_sparse": "gate_sparse",
        "channel": "channel_pair",
        "channel_pair": "channel_pair",
        "paired_channel": "channel_pair",
        "paired_channels": "channel_pair",
    }
    if raw not in aliases:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_GATE_UP_SPLIT="
            f"{raw}; expected none, up_sparse, gate_sparse, or channel_pair"
        )
    return aliases[raw]


def gate_up_channel_dense_fraction() -> float:
    raw = os.getenv("SPECLINK_SR24_GATE_UP_CHANNEL_DENSE_FRACTION", "0.125")
    try:
        value = float(raw)
    except ValueError:
        value = 0.125
    return min(max(value, 0.0), 1.0)


def gate_up_channel_strategy() -> str:
    raw = os.getenv("SPECLINK_SR24_GATE_UP_CHANNEL_STRATEGY", "norm").strip().lower()
    if raw not in {"norm", "front"}:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_GATE_UP_CHANNEL_STRATEGY="
            f"{raw}; expected norm or front"
        )
    return raw


def gate_up_channel_fused_act() -> bool:
    return _env_flag("SPECLINK_SR24_GATE_UP_CHANNEL_FUSED_ACT", "0")


def row_routed_mlp() -> bool:
    return _env_flag("SPECLINK_SR24_ROW_ROUTED_MLP", "0")


def row_routed_down_linear() -> bool:
    return _env_flag("SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR", "0")


def row_routed_down_fixed_block_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_ROW_ROUTED_DOWN_FIXED_BLOCK", "0")


def row_routed_mlp_reuse_base_output() -> bool:
    return _env_flag("SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT", "0")


def row_routed_mlp_fixed_block_dense_fill() -> bool:
    return _env_flag("SPECLINK_SR24_ROW_ROUTED_MLP_FIXED_BLOCK_DENSE_FILL", "0")


def fixed_block_input_buffer_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_FIXED_BLOCK_INPUT_BUFFER", "0")


def fixed_block_output_buffer_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER", "0")


def scheduler_policy_path() -> str:
    return os.getenv("SPECLINK_SR24_SCHEDULER_POLICY_PATH", "").strip()


def scheduler_policy_allow_legacy_mixed() -> bool:
    return _env_flag("SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_LEGACY_MIXED", "0")


def scheduler_policy_dense_bypass() -> bool:
    return _env_flag("SPECLINK_SR24_SCHEDULER_POLICY_DENSE_BYPASS", "0")


def scheduler_policy_allow_single_block_packed_parallel() -> bool:
    return _env_flag(
        "SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_SINGLE_BLOCK_PACKED_PARALLEL",
        "1",
    )


def scheduler_policy_allow_serial_packed_parallel() -> bool:
    return _env_flag(
        "SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_SERIAL_PACKED_PARALLEL",
        "0",
    )


def scheduler_policy_near_full_tolerance() -> int:
    raw = os.getenv("SPECLINK_SR24_SCHEDULER_POLICY_NEAR_FULL_TOLERANCE", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def fixed_block_capacity_padding_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_FIXED_BLOCK_CAPACITY_PADDING", "0")


def fixed_block_capacity_zero_dummy_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_FIXED_BLOCK_CAPACITY_ZERO_DUMMY", "0")


def grouped_queue_shadow_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_GROUPED_QUEUE_SHADOW", "0")


def grouped_queue_shadow_max_wait_blocks() -> int:
    raw = os.getenv("SPECLINK_SR24_GROUPED_QUEUE_MAX_WAIT_BLOCKS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def grouping_trace_enabled() -> bool:
    return (
        _env_flag("SPECLINK_SR24_GROUPING_TRACE", "0")
        or grouped_queue_shadow_enabled()
        or bool(os.getenv("SPECLINK_SR24_GROUPING_TRACE_PATH", "").strip())
    )


def grouping_trace_path() -> Path | None:
    raw = os.getenv("SPECLINK_SR24_GROUPING_TRACE_PATH", "").strip()
    if raw:
        return Path(raw)
    stats_path = os.getenv("SPECLINK_SR24_STATS_PATH", "").strip()
    if stats_path:
        return Path(stats_path).with_name("speclink_sr24_grouping_trace.jsonl")
    return None


def _write_grouping_trace(event: dict[str, Any]) -> None:
    if not grouping_trace_enabled():
        return
    path = grouping_trace_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _grouping_trace_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        pass


def _grouped_queue_shadow_emit(
    *,
    action: str,
    reason: str,
    key: tuple[Any, ...],
    entries: list[dict[str, Any]],
    max_wait_blocks: int,
) -> None:
    if not entries:
        return
    dense_rows = sum(int(entry["dense_rows"]) for entry in entries)
    base_rows = sum(int(entry["base_rows"]) for entry in entries)
    target_dense_rows = int(entries[0]["target_dense_rows"])
    target_base_rows = int(entries[0]["target_base_rows"])
    speedups = [
        float(entry["mixed_local_speedup_vs_dense"])
        for entry in entries
        if entry.get("mixed_local_speedup_vs_dense") is not None
    ]
    local_speedup = (
        sum(speedups) / max(len(speedups), 1)
        if speedups else 1.0
    )
    block_indices = [int(entry["shadow_block_index"]) for entry in entries]
    _write_grouping_trace(
        {
            "timestamp": time.time(),
            "event": "sr24_grouped_queue_shadow_decision",
            "action": action,
            "reason": reason,
            "key": list(key),
            "max_wait_blocks": int(max_wait_blocks),
            "start_block_index": block_indices[0],
            "end_block_index": block_indices[-1],
            "block_count": len(entries),
            "block_indices": block_indices,
            "dense_rows": dense_rows,
            "base_rows": base_rows,
            "target_dense_rows": target_dense_rows,
            "target_base_rows": target_base_rows,
            "dense_fill": dense_rows / max(target_dense_rows, 1),
            "base_fill": base_rows / max(target_base_rows, 1),
            "wait_blocks": max(0, len(entries) - 1),
            "local_speedup": local_speedup,
            "dense_time_units": float(len(entries)),
            "mixed_time_units": (
                float(len(entries)) / max(local_speedup, 1.0)
                if action == "group" else float(len(entries))
            ),
        }
    )


def _grouped_queue_shadow_entry_from_event(
    event: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    if not grouped_queue_shadow_enabled():
        return None
    if event.get("event") != "sr24_grouping_opportunity":
        return None
    if not bool(event.get("compact_spec_batch")):
        return None
    if not bool(event.get("policy_compatible")):
        return None
    if not bool(event.get("descriptor_available")):
        return None
    scheduled_width = event.get("scheduled_width")
    valid_width = event.get("valid_width")
    prefix = event.get("prefix")
    policy_batch = event.get("policy_batch")
    dense_rows = event.get("dense_rows")
    base_rows = event.get("base_rows")
    target_dense_rows = event.get("grouped_dense_rows")
    target_base_rows = event.get("grouped_base_rows")
    speedup = event.get("mixed_local_speedup_vs_dense")
    try:
        scheduled_width = int(scheduled_width)
        valid_width = int(valid_width)
        prefix = int(prefix)
        policy_batch = int(policy_batch)
        dense_rows = int(dense_rows)
        base_rows = int(base_rows)
        target_dense_rows = int(target_dense_rows)
        target_base_rows = int(target_base_rows)
        speedup = float(speedup)
    except (TypeError, ValueError):
        return None
    if (
        scheduled_width <= 0
        or valid_width <= 0
        or policy_batch <= 0
        or dense_rows <= 0
        or base_rows <= 0
        or target_dense_rows <= 0
        or target_base_rows <= 0
        or speedup <= 1.0
    ):
        return None
    policy_k = event.get("policy_k")
    if policy_k not in (None, ""):
        try:
            if int(policy_k) != valid_width:
                return None
        except (TypeError, ValueError):
            return None
    policy_prefix = event.get("policy_prefix")
    if policy_prefix not in (None, ""):
        try:
            if int(policy_prefix) != prefix:
                return None
        except (TypeError, ValueError):
            return None
    key = (
        policy_batch,
        scheduled_width,
        valid_width,
        prefix,
        target_dense_rows,
        target_base_rows,
    )
    return key, {
        "shadow_block_index": -1,
        "dense_rows": dense_rows,
        "base_rows": base_rows,
        "target_dense_rows": target_dense_rows,
        "target_base_rows": target_base_rows,
        "mixed_local_speedup_vs_dense": speedup,
    }


def _grouped_queue_shadow_drain_key(
    key: tuple[Any, ...],
    queue: deque[dict[str, Any]],
    *,
    max_wait_blocks: int,
    flush_tail: bool = False,
) -> None:
    while queue:
        first = queue[0]
        dense_sum = 0
        base_sum = 0
        found_index: int | None = None
        max_index = min(len(queue) - 1, max_wait_blocks)
        for index in range(max_index + 1):
            dense_sum += int(queue[index]["dense_rows"])
            base_sum += int(queue[index]["base_rows"])
            if (
                dense_sum >= int(first["target_dense_rows"])
                and base_sum >= int(first["target_base_rows"])
            ):
                found_index = index
                break
        if found_index is not None:
            entries = [queue.popleft() for _ in range(found_index + 1)]
            _grouped_queue_shadow_emit(
                action="group",
                reason="target_reached",
                key=key,
                entries=entries,
                max_wait_blocks=max_wait_blocks,
            )
            continue
        if flush_tail:
            entries = [queue.popleft() for _ in range(len(queue))]
            _grouped_queue_shadow_emit(
                action="fallback",
                reason="tail_underfilled",
                key=key,
                entries=entries,
                max_wait_blocks=max_wait_blocks,
            )
            continue
        if len(queue) > max_wait_blocks:
            entry = queue.popleft()
            _grouped_queue_shadow_emit(
                action="fallback",
                reason="timeout_underfilled",
                key=key,
                entries=[entry],
                max_wait_blocks=max_wait_blocks,
            )
            continue
        break


def _record_grouped_queue_shadow_event(event: dict[str, Any]) -> None:
    parsed = _grouped_queue_shadow_entry_from_event(event)
    if parsed is None:
        return
    _install_grouped_queue_shadow_signal_handlers()
    key, entry = parsed
    max_wait_blocks = grouped_queue_shadow_max_wait_blocks()
    global _grouped_queue_shadow_next_index
    with _grouped_queue_shadow_lock:
        entry["shadow_block_index"] = _grouped_queue_shadow_next_index
        _grouped_queue_shadow_next_index += 1
        queue = _grouped_queue_shadow_by_key[key]
        queue.append(entry)
        _grouped_queue_shadow_drain_key(
            key,
            queue,
            max_wait_blocks=max_wait_blocks,
        )


def _flush_grouped_queue_shadow_tail() -> None:
    if not grouped_queue_shadow_enabled():
        return
    max_wait_blocks = grouped_queue_shadow_max_wait_blocks()
    with _grouped_queue_shadow_lock:
        for key, queue in list(_grouped_queue_shadow_by_key.items()):
            _grouped_queue_shadow_drain_key(
                key,
                queue,
                max_wait_blocks=max_wait_blocks,
                flush_tail=True,
            )


def _handle_grouped_queue_shadow_signal(signum: int, frame: Any) -> None:
    _flush_grouped_queue_shadow_tail()
    previous = _grouped_queue_shadow_previous_signal_handlers.get(signum)
    if callable(previous) and previous is not _handle_grouped_queue_shadow_signal:
        previous(signum, frame)
        return
    if previous == signal.SIG_IGN:
        return
    raise SystemExit(128 + int(signum))


def _install_grouped_queue_shadow_signal_handlers() -> None:
    global _grouped_queue_shadow_signal_handlers_installed
    if _grouped_queue_shadow_signal_handlers_installed:
        return
    with _grouped_queue_shadow_lock:
        if _grouped_queue_shadow_signal_handlers_installed:
            return
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                _grouped_queue_shadow_previous_signal_handlers[signum] = (
                    signal.getsignal(signum)
                )
                signal.signal(signum, _handle_grouped_queue_shadow_signal)
            except Exception:
                pass
        _grouped_queue_shadow_signal_handlers_installed = True


atexit.register(_flush_grouped_queue_shadow_tail)


@lru_cache(maxsize=8)
def _scheduler_policy_by_batch(path: str) -> dict[int, dict[str, Any]]:
    if not path:
        return {}
    policy_path = Path(path).expanduser()
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "failed to load SPECLINK_SR24_SCHEDULER_POLICY_PATH="
            f"{path!r}"
        ) from exc
    policies = payload.get("policies")
    if not isinstance(policies, list):
        raise RuntimeError(
            "SR24 scheduler policy JSON must contain a 'policies' list"
        )
    by_batch: dict[int, dict[str, Any]] = {}
    for item in policies:
        if not isinstance(item, dict):
            continue
        try:
            batch = int(item.get("batch_size"))
        except (TypeError, ValueError):
            continue
        if batch > 0:
            by_batch[batch] = item
    return by_batch


def _scheduler_policy_for_active_count(
    active_count: int,
) -> tuple[int, dict[str, Any]] | None:
    path = scheduler_policy_path()
    if not path:
        return None
    policies = _scheduler_policy_by_batch(path)
    if not policies:
        return None
    active_count = max(1, int(active_count))
    if active_count in policies:
        return active_count, policies[active_count]
    tolerance = scheduler_policy_near_full_tolerance()
    if tolerance > 0:
        larger = [
            batch
            for batch in policies
            if batch > active_count and batch - active_count <= tolerance
        ]
        if larger:
            batch = min(larger)
            return batch, policies[batch]
    smaller = [batch for batch in policies if batch <= active_count]
    if smaller:
        batch = max(smaller)
        return batch, policies[batch]
    batch = min(policies)
    return batch, policies[batch]


def _scheduler_policy_int(policy: dict[str, Any], key: str) -> int | None:
    value = policy.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scheduler_policy_operator_supported(policy: dict[str, Any]) -> bool:
    """Return whether this live path implements the policy's mixed operator.

    The current fixed-block serving path is still the legacy split
    dense/sparse MLP.  It can approximate the planner's packed_parallel row
    only for a single ready verifier block; rows that require grouped verifier
    blocks still need a real grouped operator or queue.
    """
    if scheduler_policy_allow_legacy_mixed():
        return True
    operator = str(policy.get("mixed_operator") or "").strip()
    if not operator:
        return True
    if operator == "packed_parallel":
        min_grouped = _scheduler_policy_int(policy, "min_grouped_verifier_blocks")
        action = str(policy.get("planner_action") or "").strip()
        single_block_ready = (
            action == "use_mixed_single_block"
            and min_grouped is not None
            and int(min_grouped) <= 1
            and fixed_block_input_buffer_enabled()
        )
        return (
            scheduler_policy_allow_single_block_packed_parallel()
            and single_block_ready
            and (
                route_overlap_streams()
                or scheduler_policy_allow_serial_packed_parallel()
            )
        )
    return operator in {"legacy_fixed_block"}


def _scheduler_policy_requires_dense_fallback(
    policy: dict[str, Any],
    *,
    prefix: int,
    valid_width: int,
) -> tuple[bool, str, int | None]:
    policy_prefix = _scheduler_policy_int(policy, "prefix")
    policy_k = _scheduler_policy_int(policy, "k")
    policy_action = str(policy.get("planner_action") or "").strip()
    min_grouped = _scheduler_policy_int(policy, "min_grouped_verifier_blocks")
    policy_compatible = (
        policy_prefix in (None, prefix)
        and policy_k in (None, valid_width)
    )
    if not policy_compatible:
        return False, "incompatible", min_grouped
    if not _scheduler_policy_operator_supported(policy):
        return True, "operator_unimplemented", min_grouped
    if (
        policy_action in {"dense_fallback", "dense_fallback_until_grouped"}
        and (min_grouped is None or min_grouped > 1)
    ):
        return True, "underfilled", min_grouped
    return False, "", min_grouped


def _trace_grouping_opportunity(
    *,
    reason: str,
    sr_mode: str,
    residual_policy: str,
    non_draft_policy: str,
    mask_state: str | None,
    compact_spec_batch: bool,
    compact_width: int | None,
    draft_counts: list[int],
    scheduled_counts: list[int],
    total_num_scheduled_tokens: int,
    min_prefix_residual: int,
    fixed_prefix_route: FixedPrefixRouteDescriptor | None,
    batched_mask_applied: bool,
) -> None:
    if not grouping_trace_enabled():
        return
    request_count = len(draft_counts)
    active_count = sum(1 for count in draft_counts if int(count) > 0)
    scheduled_width: int | None = None
    valid_width: int | None = None
    prefix = max(0, int(min_prefix_residual))
    dense_width: int | None = None
    base_width: int | None = None
    descriptor_available = fixed_prefix_route is not None
    if fixed_prefix_route is not None:
        scheduled_width = int(fixed_prefix_route.scheduled_width)
        valid_width = int(fixed_prefix_route.valid_width)
        prefix = int(fixed_prefix_route.prefix)
        dense_width = int(fixed_prefix_route.dense_width)
        base_width = int(fixed_prefix_route.base_width)
    elif compact_spec_batch and compact_width is not None:
        scheduled_width = int(compact_width)
        valid_width = max(0, scheduled_width - 1)
        if valid_width >= prefix:
            dense_width = prefix + 1
            base_width = max(0, valid_width - prefix)

    dense_rows = (
        int(active_count) * int(dense_width)
        if dense_width is not None
        else None
    )
    base_rows = (
        int(active_count) * int(base_width)
        if base_width is not None
        else None
    )
    policy_batch: int | None = None
    policy: dict[str, Any] | None = None
    policy_action = ""
    policy_prefix: int | None = None
    policy_k: int | None = None
    policy_compatible = False
    operator_supported = False
    single_block_fallback = False
    single_block_fallback_reason = ""
    min_grouped: int | None = None
    target_effective_batch_size: int | None = None
    grouped_dense_rows: int | None = None
    grouped_base_rows: int | None = None
    mixed_operator = ""
    mixed_local_speedup: float | None = None
    policy_match = _scheduler_policy_for_active_count(active_count)
    if policy_match is not None:
        policy_batch, policy = policy_match
        policy_action = str(policy.get("planner_action") or "").strip()
        policy_prefix = _scheduler_policy_int(policy, "prefix")
        policy_k = _scheduler_policy_int(policy, "k")
        min_grouped = _scheduler_policy_int(policy, "min_grouped_verifier_blocks")
        target_effective_batch_size = _scheduler_policy_int(
            policy, "target_effective_batch_size"
        )
        grouped_dense_rows = _scheduler_policy_int(policy, "grouped_dense_rows")
        grouped_base_rows = _scheduler_policy_int(policy, "grouped_base_rows")
        mixed_operator = str(policy.get("mixed_operator") or "").strip()
        try:
            raw_speedup = policy.get("mixed_local_speedup_vs_dense")
            mixed_local_speedup = (
                None if raw_speedup in (None, "") else float(raw_speedup)
            )
        except (TypeError, ValueError):
            mixed_local_speedup = None
        if valid_width is not None:
            policy_compatible = (
                policy_prefix in (None, prefix)
                and policy_k in (None, valid_width)
            )
            (
                single_block_fallback,
                single_block_fallback_reason,
                _min_grouped_from_policy,
            ) = _scheduler_policy_requires_dense_fallback(
                policy,
                prefix=prefix,
                valid_width=valid_width,
            )
            if min_grouped is None:
                min_grouped = _min_grouped_from_policy
        operator_supported = _scheduler_policy_operator_supported(policy)

    current_verifier_blocks = 1 if compact_spec_batch and active_count > 0 else 0
    if min_grouped is not None and min_grouped > 0:
        missing_grouped_blocks = max(0, int(min_grouped) - current_verifier_blocks)
        block_group_fill_ratio = current_verifier_blocks / max(int(min_grouped), 1)
    else:
        missing_grouped_blocks = 0
        block_group_fill_ratio = None
    if target_effective_batch_size is not None and target_effective_batch_size > 0:
        missing_effective_requests = max(
            0, int(target_effective_batch_size) - int(active_count)
        )
        effective_batch_fill_ratio = active_count / max(
            int(target_effective_batch_size), 1
        )
        effective_batch_ready = active_count >= int(target_effective_batch_size)
        effective_batch_near_full_ready = (
            missing_effective_requests <= scheduler_policy_near_full_tolerance()
        )
    else:
        missing_effective_requests = 0
        effective_batch_fill_ratio = None
        effective_batch_ready = False
        effective_batch_near_full_ready = False
    policy_batch_fill_ratio = (
        active_count / max(int(policy_batch), 1)
        if policy_batch is not None
        else None
    )
    single_block_group_ready = (
        compact_spec_batch
        and policy_compatible
        and operator_supported
        and min_grouped is not None
        and int(min_grouped) > 0
        and current_verifier_blocks >= int(min_grouped)
    )
    count_only_request_fill_ready = (
        compact_spec_batch
        and policy_compatible
        and min_grouped is not None
        and int(min_grouped) > 0
        and active_count >= int(min_grouped)
    )
    use_mixed_single_block = (
        compact_spec_batch
        and policy_compatible
        and operator_supported
        and not single_block_fallback
        and policy_action == "use_mixed_single_block"
    )
    event = {
        "timestamp": time.time(),
        "event": "sr24_grouping_opportunity",
        "reason": reason,
        "mode": sr_mode,
        "residual_policy": residual_policy,
        "non_draft_policy": non_draft_policy,
        "mask_state": mask_state,
        "batched_mask_applied": bool(batched_mask_applied),
        "compact_spec_batch": bool(compact_spec_batch),
        "descriptor_available": bool(descriptor_available),
        "request_count": int(request_count),
        "active_requests": int(active_count),
        "compact_tensor_blocks": int(current_verifier_blocks),
        "active_verifier_blocks": int(active_count),
        "active_request_verifier_blocks": int(active_count),
        "scheduled_counts_min": (
            int(min(scheduled_counts)) if scheduled_counts else None
        ),
        "scheduled_counts_max": (
            int(max(scheduled_counts)) if scheduled_counts else None
        ),
        "scheduled_width": scheduled_width,
        "valid_width": valid_width,
        "prefix": int(prefix),
        "dense_rows_per_block": dense_width,
        "base_rows_per_block": base_width,
        "dense_rows": dense_rows,
        "base_rows": base_rows,
        "total_scheduled_tokens": int(total_num_scheduled_tokens),
        "policy_batch": policy_batch,
        "policy_action": policy_action,
        "policy_prefix": policy_prefix,
        "policy_k": policy_k,
        "policy_compatible": bool(policy_compatible),
        "mixed_operator": mixed_operator,
        "operator_supported_live": bool(operator_supported),
        "single_block_policy_fallback": bool(single_block_fallback),
        "single_block_fallback_reason": single_block_fallback_reason,
        "min_grouped_verifier_blocks": min_grouped,
        "target_effective_batch_size": target_effective_batch_size,
        "grouped_dense_rows": grouped_dense_rows,
        "grouped_base_rows": grouped_base_rows,
        "mixed_local_speedup_vs_dense": mixed_local_speedup,
        "current_verifier_blocks": int(current_verifier_blocks),
        "missing_grouped_verifier_blocks": int(missing_grouped_blocks),
        "block_group_fill_ratio": block_group_fill_ratio,
        "policy_batch_fill_ratio": policy_batch_fill_ratio,
        "target_effective_missing_requests": int(missing_effective_requests),
        "target_effective_batch_fill_ratio": effective_batch_fill_ratio,
        "target_effective_batch_ready": bool(effective_batch_ready),
        "target_effective_near_full_ready": bool(
            effective_batch_near_full_ready
        ),
        "count_only_request_fill_ready": bool(count_only_request_fill_ready),
        "meets_min_grouped_by_count": bool(count_only_request_fill_ready),
        "grouping_ready_now": bool(single_block_group_ready),
        "would_use_mixed_single_block": bool(use_mixed_single_block),
        "would_use_mixed_if_grouped": bool(single_block_group_ready),
        "would_feed_grouped_operator_if_implemented": bool(
            single_block_group_ready
        ),
        "grouping_requires_cross_step_queue": bool(
            compact_spec_batch
            and policy_compatible
            and min_grouped is not None
            and int(min_grouped) > current_verifier_blocks
        ),
        "grouping_requires_cross_step_queue_by_effective_batch": bool(
            compact_spec_batch
            and policy_compatible
            and target_effective_batch_size is not None
            and active_count < int(target_effective_batch_size)
        ),
    }
    _write_grouping_trace(event)
    _record_grouped_queue_shadow_event(event)


def row_routed_mlp_min_dense_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS", "128").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 128


@lru_cache(maxsize=16)
def _leaf_int_map_from_env(name: str) -> dict[str, int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    out: dict[str, int] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"{name} entries must be leaf=value; got {item!r}")
        leaf, value = item.split("=", 1)
        leaf = leaf.strip()
        if leaf not in TARGET_LEAFS:
            raise RuntimeError(
                f"{name} has unsupported leaf {leaf!r}; supported={_TARGET_LEAFS_CSV}"
            )
        try:
            out[leaf] = int(value.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"{name} value for {leaf!r} must be an integer; got {value!r}"
            ) from exc
    return out


def _row_routed_leaf_threshold(
    *,
    name: str,
    module_leaf: str | None,
    default: int,
    minimum: int,
) -> int:
    if not module_leaf:
        return default
    value = _leaf_int_map_from_env(name).get(module_leaf)
    if value is None:
        return default
    return max(minimum, int(value))


def row_routed_mlp_min_dense_rows_for_leaf(module_leaf: str | None) -> int:
    return _row_routed_leaf_threshold(
        name="SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS_BY_LEAF",
        module_leaf=module_leaf,
        default=row_routed_mlp_min_dense_rows(),
        minimum=1,
    )


def row_routed_mlp_max_dense_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def row_routed_mlp_max_dense_rows_for_leaf(module_leaf: str | None) -> int:
    return _row_routed_leaf_threshold(
        name="SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS_BY_LEAF",
        module_leaf=module_leaf,
        default=row_routed_mlp_max_dense_rows(),
        minimum=0,
    )


def row_routed_mlp_max_base_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def row_routed_mlp_max_base_rows_for_leaf(module_leaf: str | None) -> int:
    return _row_routed_leaf_threshold(
        name="SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS_BY_LEAF",
        module_leaf=module_leaf,
        default=row_routed_mlp_max_base_rows(),
        minimum=0,
    )


def base_only_dense_nonverify() -> bool:
    return _env_flag("SPECLINK_SR24_BASE_ONLY_DENSE_NONVERIFY", "0")


def base_only_dense_verify_max_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_MAX_ROWS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _layer_in_scope(
    layer_index: int | None,
    module_leaf: str,
    layer_ids: set[int] | None,
    layer_ids_by_leaf: dict[str, set[int]],
) -> bool:
    if layer_ids_by_leaf:
        leaf_ids = layer_ids_by_leaf.get(module_leaf)
        if leaf_ids is None:
            return False
        return layer_index in leaf_ids
    if layer_ids is not None:
        return layer_index in layer_ids
    return True


def _selective_dense_nonverify_scope_raw() -> str:
    return os.getenv(
        "SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_LAYER_IDS_BY_LEAF", ""
    ).strip()


@lru_cache(maxsize=4)
def _selective_dense_nonverify_layer_ids_by_leaf() -> dict[str, set[int]]:
    raw = _selective_dense_nonverify_scope_raw().lower()
    if raw in {"", "none", "base_only", "sparse", "all", "*"}:
        return {}
    return _parse_layer_ids_by_leaf_env(
        "SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_LAYER_IDS_BY_LEAF"
    )


def selective_dense_nonverify_max_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_MAX_ROWS", "0")
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return 0


def selective_dense_nonverify_static_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_STATIC_ROWS", "0")
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return 0


def _selective_dense_nonverify_for_module(module: Any) -> bool:
    """Whether a no-mask selective forward should keep dense non-draft work.

    Empty scope preserves the old behavior where
    `SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT=1` makes no-verify/non-draft
    selective forwards dense.  A non-empty per-leaf layer scope lets experiments
    keep that quality guard only in sensitive layers while allowing other
    no-verify rows to use the 2:4 sparse base.
    """
    if not selective_correct_non_draft():
        return False
    raw = _selective_dense_nonverify_scope_raw()
    if not raw:
        return True
    if raw.lower() in {"none", "base_only", "sparse"}:
        return False
    if raw.lower() in {"all", "*"}:
        return True
    module_leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    layer_index = getattr(module, "_speclink_sr24_profile_layer", None)
    layer_ids_by_leaf = _selective_dense_nonverify_layer_ids_by_leaf()
    if not layer_ids_by_leaf:
        return False
    return _layer_in_scope(
        int(layer_index) if layer_index is not None else None,
        module_leaf,
        None,
        layer_ids_by_leaf,
    )


def _selective_dense_nonverify_for_module_rows(
    module: Any,
    rows: int | None,
) -> bool:
    if _selective_dense_nonverify_for_module(module):
        return True
    if not selective_correct_non_draft():
        return False
    max_rows = selective_dense_nonverify_max_rows()
    if max_rows <= 0 or rows is None:
        return False
    if _torch_is_compiling():
        rows_hint = selective_dense_nonverify_static_rows()
        return rows_hint > 0 and rows_hint <= max_rows
    # During torch.compile graph capture, shape-derived row counts can be
    # symbolic.  Keep this guard Python-static so Inductor never sees a
    # SymPy/Torch symbolic LessThan as a graph input.
    if type(rows) is not int:
        rows_hint = selective_dense_nonverify_static_rows()
        if rows_hint > 0:
            return rows_hint <= max_rows
        return False
    return 0 < rows <= max_rows


def _runtime_base_only_scope_raw() -> str:
    return os.getenv(
        "SPECLINK_SR24_RUNTIME_BASE_ONLY_LAYER_IDS_BY_LEAF", ""
    ).strip()


@lru_cache(maxsize=4)
def _runtime_base_only_layer_ids_by_leaf() -> dict[str, set[int]]:
    raw = _runtime_base_only_scope_raw().lower()
    if raw in {"", "none"}:
        return {}
    return _parse_layer_ids_by_leaf_env(
        "SPECLINK_SR24_RUNTIME_BASE_ONLY_LAYER_IDS_BY_LEAF"
    )


def _runtime_base_only_for_module(module: Any) -> bool:
    """Use base-only compute for selected layers while keeping uniform storage.

    This is different from attaching those layers as true no-residual modules:
    the module still has the same dense_rows SR24 attributes as other layers, so
    default vLLM compile sees a uniform data format across decoder layers.  The
    routing decision is made only at runtime inside the Linear/MLP hooks.
    """
    layer_ids_by_leaf = _runtime_base_only_layer_ids_by_leaf()
    if not layer_ids_by_leaf:
        return False
    module_leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    layer_index = getattr(module, "_speclink_sr24_profile_layer", None)
    return _layer_in_scope(
        int(layer_index) if layer_index is not None else None,
        module_leaf,
        None,
        layer_ids_by_leaf,
    )


def _selective_no_context_uses_sparse_base(
    module: Any,
    rows: int | None = None,
) -> bool:
    return (
        mode() == "selective"
        and _current_residual_state() is None
        and _current_residual_mask() is None
        and not _selective_dense_nonverify_for_module_rows(module, rows)
    )


def static_mask_buffer_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_STATIC_MASK_BUFFER")


def cudagraph_bucket_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_CUDAGRAPH_BUCKET", "0")


def static_mask_buffer_capacity() -> int:
    try:
        return max(0, int(os.getenv("SPECLINK_SR24_MASK_BUFFER_CAPACITY", "16384")))
    except ValueError:
        return 16384


def all_corrected_dense_fastpath() -> bool:
    return _env_flag("SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH", "1")


def full_residual_early_dense() -> bool:
    return _env_flag("SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE", "0")


def noverify_dense_mlp_fastpath() -> bool:
    return _env_flag("SPECLINK_SR24_NOVERIFY_DENSE_MLP_FASTPATH", "1")


def _all_residual_dense_shortcut_enabled() -> bool:
    """Whether all-residual verify states may collapse to dense Linear.

    `all_corrected_24` normally uses the algebraically equivalent dense Linear
    as a no-op control. When that fastpath is explicitly disabled, the point is
    to measure the real sparse-base plus residual-correction operator, so the
    all-residual state must not silently short-circuit back to dense. Set
    SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE=1 to measure the optimized hook
    path that keeps SR24 attached but avoids doing sparse base work before a
    known full dense correction.
    """
    return full_residual_early_dense() or not (
        mode() == "all_corrected" and not all_corrected_dense_fastpath()
    )


def residual_out_chunk() -> int:
    raw = os.getenv("SPECLINK_SR24_RESIDUAL_OUT_CHUNK", "4096").strip()
    try:
        return int(raw)
    except ValueError:
        return 4096


def cache_compressed_residual_weight() -> bool:
    return _env_flag("SPECLINK_SR24_CACHE_COMPRESSED_RESIDUAL_WEIGHT", "0")


def prewarm_compressed_residual_weight() -> bool:
    return _env_flag("SPECLINK_SR24_PREWARM_COMPRESSED_RESIDUAL_WEIGHT", "0")


def compressed_residual_triton() -> bool:
    return _env_flag("SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON", "0")


def compressed_residual_block_m() -> int:
    raw = os.getenv("SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_M", "32").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def compressed_residual_block_n() -> int:
    raw = os.getenv("SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_N", "128").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def compressed_residual_block_g() -> int:
    raw = os.getenv("SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_G", "16").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 32


def residual_extract_chunk_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_EXTRACT_CHUNK_ROWS", "128").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 128


def residual_bucket_size() -> int:
    raw = os.getenv("SPECLINK_SR24_RESIDUAL_BUCKET_SIZE", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def residual_bucket_scale_by_active() -> bool:
    return _env_flag("SPECLINK_SR24_RESIDUAL_BUCKET_SCALE_BY_ACTIVE", "0")


def cudagraph_bucket_active_hint() -> int:
    raw = os.getenv("SPECLINK_SR24_CUDAGRAPH_BUCKET_ACTIVE_HINT", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _effective_residual_bucket_size(active_count: int | None = None) -> int:
    bucket_size = residual_bucket_size()
    if bucket_size <= 0:
        return 0
    if not residual_bucket_scale_by_active():
        return bucket_size
    if active_count is None:
        return bucket_size
    return bucket_size * max(1, int(active_count))


def residual_bucket_priority() -> bool:
    return _env_flag("SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY", "0")


def direct_position_bucket_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_DIRECT_POSITION_BUCKET", "0")


def bonus_priority() -> float:
    raw = os.getenv("SPECLINK_SR24_BONUS_PRIORITY", "4.0").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 4.0
    return max(0.0, value)


def draft_position_priority_scale() -> float:
    raw = os.getenv("SPECLINK_SR24_DRAFT_POSITION_PRIORITY_SCALE", "0.0").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    return max(0.0, value)


def bucket_dense_copy() -> bool:
    return _env_flag("SPECLINK_SR24_BUCKET_DENSE_COPY", "0")


def bucket_dense_copy_active_only() -> bool:
    return _env_flag("SPECLINK_SR24_BUCKET_DENSE_COPY_ACTIVE_ONLY", "0")


def bucket_dense_compute_active_only() -> bool:
    return _env_flag("SPECLINK_SR24_BUCKET_DENSE_COMPUTE_ACTIVE_ONLY", "0")


def bucket_dense_active_mask_fused() -> bool:
    return _env_flag("SPECLINK_SR24_BUCKET_DENSE_ACTIVE_MASK_FUSED", "0")


def bucket_dense_active_mask_fused_graph_safe() -> bool:
    return bucket_dense_active_mask_fused() and triton_bucket_dense_gemm()


def bucket_dense_delta_add() -> bool:
    return _env_flag("SPECLINK_SR24_BUCKET_DENSE_DELTA_ADD", "0")


def sort_bucket_rows() -> bool:
    return _env_flag("SPECLINK_SR24_SORT_BUCKET_ROWS", "0")


def route_bucket_rows() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_BUCKET_ROWS", "0")


def route_bucket_rows_graph_static_unsafe() -> bool:
    """Whether route-bucket cached row plans are unsafe for CUDA Graph replay.

    Graph-static buckets keep a fixed bucket length and pad inactive tail
    entries through `bucket_values`. The route-bucket cached row plan only
    carries row ids, so it cannot distinguish active selected rows from padded
    rows. Even when the eager active-only path consumes `bucket_values`, the
    captured/replayed residual/base row plan does not. Keep this ablation eager
    until the route plan itself is represented as fixed-shape rows plus an
    active mask.
    """
    return route_bucket_rows() and cudagraph_bucket_enabled()


def route_all_residual_rows() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS", "0")


def route_all_skip_bucket() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET", "0")


def direct_cpu_route_rows_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS", "0")


def fixed_prefix_route_fastpath_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_FIXED_PREFIX_ROUTE_FASTPATH", "1")


def fixed_prefix_route_descriptor_only() -> bool:
    return _env_flag("SPECLINK_SR24_FIXED_PREFIX_ROUTE_DESCRIPTOR_ONLY", "0")


def route_reuse_base_output() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT", "0")


def route_contiguous_fastpath() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH", "0")


def route_overlap_streams() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_OVERLAP_STREAMS", "0")


def route_overlap_allow_cudagraph() -> bool:
    return _env_flag("SPECLINK_SR24_ROUTE_OVERLAP_ALLOW_CUDAGRAPH", "0")


def route_dense_fallback_fraction() -> float:
    raw = os.getenv("SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION", "1.1").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.1


def route_min_dense_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_ROUTE_MIN_DENSE_ROWS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def route_min_base_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_ROUTE_MIN_BASE_ROWS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def route_min_base_rows_for_leaf(module_leaf: str | None) -> int:
    return _row_routed_leaf_threshold(
        name="SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF",
        module_leaf=module_leaf,
        default=route_min_base_rows(),
        minimum=0,
    )


def route_max_dense_fraction() -> float:
    raw = os.getenv("SPECLINK_SR24_ROUTE_MAX_DENSE_FRACTION", "1.1").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.1


def adaptive_dense_fallback_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK", "0")


def adaptive_dense_fallback_no_residual_only() -> bool:
    return _env_flag("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_NO_RESIDUAL_ONLY", "0")


def adaptive_dense_fallback_small_rows() -> int:
    raw = os.getenv("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS", "0")
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return 0


def adaptive_dense_fallback_gate_up_fraction() -> float:
    raw = os.getenv("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_GATE_UP_FRACTION", "0.10")
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return 0.10


def adaptive_dense_fallback_down_fraction() -> float:
    raw = os.getenv("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_DOWN_FRACTION", "0.25")
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return 0.25


def adaptive_dense_fallback_small_down_no_residual() -> bool:
    return _env_flag("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL", "1")


def adaptive_dense_fallback_small_gate_up_no_residual() -> bool:
    return _env_flag(
        "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_GATE_UP_NO_RESIDUAL", "0"
    )


def triton_route_assembly() -> bool:
    return _env_flag("SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY", "0")


def triton_bucket_override() -> bool:
    return _env_flag("SPECLINK_SR24_TRITON_BUCKET_OVERRIDE", "0")


def triton_bucket_dense_gemm() -> bool:
    return _env_flag("SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM", "0")


def triton_bucket_scatter() -> bool:
    return _env_flag("SPECLINK_SR24_TRITON_BUCKET_SCATTER", "0")


def triton_bucket_dense_block_m() -> int:
    raw = os.getenv("SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_M", "16").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 16


def triton_bucket_dense_block_n() -> int:
    raw = os.getenv("SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_N", "32").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 32


def triton_bucket_dense_block_k() -> int:
    raw = os.getenv("SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_K", "128").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 128


def batched_mask_builder_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_BATCHED_MASK_BUILDER", "0")


def batched_uniform_direct_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_BATCHED_UNIFORM_DIRECT", "0")


def gpu_count_mask_builder_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_GPU_COUNT_MASK_BUILDER", "0")


def selective_correct_non_draft() -> bool:
    return _env_flag("SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT", "1")


def selective_non_draft_policy() -> str:
    raw = os.getenv(
        "SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY", "auto"
    ).strip().lower()
    if raw == "auto":
        return "all" if selective_correct_non_draft() else "none"
    return raw


def selective_residual_policy() -> str:
    return os.getenv(
        "SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY", "critical_prefix"
    ).strip().lower()


def selective_prefix_threshold() -> float:
    raw = os.getenv("SPECLINK_SR24_PREFIX_THRESHOLD", "").strip()
    if not raw:
        return threshold()
    try:
        value = float(raw)
    except ValueError:
        return threshold()
    return min(max(value, 0.0), 1.0)


def selective_extra_after_low() -> int:
    raw = os.getenv("SPECLINK_SR24_SELECTIVE_EXTRA_AFTER_LOW", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def selective_min_prefix_residual() -> int:
    raw = os.getenv("SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def selective_max_residual_draft_rows() -> int:
    raw = os.getenv(
        "SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS", "0"
    ).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _selective_policy_forces_all_residual(
    *,
    sr_mode: str,
    correct_non_draft: bool,
    non_draft_policy: str,
    residual_policy: str,
    cutoff: float,
    max_residual_draft_rows: int,
) -> bool:
    """Return whether the selective policy is conservatively all-residual.

    This is intentionally narrow: it covers the diagnostic high-threshold
    policies where every draft row should be corrected and every non-draft row
    is explicitly corrected.  In this case a dynamic mixed mask only adds CPU
    synchronization and blocks CUDA Graph replay; the Linear hooks can use the
    same all-residual plan state as all_corrected/static all_residual.
    """
    if sr_mode != "selective":
        return False
    if not correct_non_draft or non_draft_policy != "all":
        return False
    if max_residual_draft_rows > 0:
        return False
    if cutoff < 1.0:
        return False
    return residual_policy in {
        "all_if_any_low",
        "batch_all_if_any_low",
        "low_confidence",
    }


def low_confidence_cap_by_risk() -> bool:
    return _env_flag("SPECLINK_SR24_LOW_CONFIDENCE_CAP_BY_RISK", "0")


def log_path() -> Path | None:
    raw = os.getenv("SPECLINK_SR24_LOG", "").strip()
    return Path(raw) if raw else None


def debug_trace_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_DEBUG_TRACE", "0")


def debug_trace_path() -> Path | None:
    raw = os.getenv("SPECLINK_SR24_DEBUG_TRACE_PATH", "").strip()
    if raw:
        return Path(raw)
    stats_path = os.getenv("SPECLINK_SR24_STATS_PATH", "").strip()
    if stats_path:
        return Path(stats_path).with_name("speclink_sr24_debug_trace.jsonl")
    return None


def debug_trace_req_substr() -> str:
    return os.getenv("SPECLINK_SR24_DEBUG_REQ_SUBSTR", "").strip()


def _debug_trace_matches(req_id: str) -> bool:
    if not debug_trace_enabled():
        return False
    needle = debug_trace_req_substr()
    return not needle or needle in req_id


def _write_debug_trace(event: dict[str, Any]) -> None:
    path = debug_trace_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        pass


def _debug_trace_verify_request(
    *,
    req_id: str,
    req_idx: int,
    sr_mode: str,
    residual_policy: str,
    non_draft_policy: str,
    cutoff: float,
    draft_count: int,
    scheduled_count: int,
    cu_scheduled_count: int,
    total_num_scheduled_tokens: int,
    scores: torch.Tensor | None,
    residual_mask: torch.Tensor | None,
    mask_state: str,
    generated_len: int | None,
    batched_mask_applied: bool,
) -> None:
    if not _debug_trace_matches(req_id):
        return
    end = int(cu_scheduled_count)
    sched = int(scheduled_count)
    start = end - sched
    valid_rows = max(0, min(int(draft_count), int(total_num_scheduled_tokens) - start))
    bonus_row = start + valid_rows
    has_bonus_row = bonus_row < min(end, int(total_num_scheduled_tokens))
    score_values: list[float] | None = None
    if scores is not None:
        try:
            score_values = [
                float(value)
                for value in scores[:valid_rows].detach().to("cpu").tolist()
            ]
        except Exception:
            score_values = None
    mask_values: list[bool] | None = None
    bonus_uses_residual: bool | None = None
    if residual_mask is not None and valid_rows > 0:
        try:
            mask_values = [
                bool(value)
                for value in residual_mask[
                    start : start + valid_rows
                ].detach().to("cpu").tolist()
            ]
        except Exception:
            mask_values = None
    if residual_mask is not None and has_bonus_row:
        try:
            bonus_uses_residual = bool(
                residual_mask[bonus_row].detach().to("cpu").item()
            )
        except Exception:
            bonus_uses_residual = None
    _write_debug_trace(
        {
            "timestamp": time.time(),
            "event": "verify_mask_request",
            "req_id": req_id,
            "req_idx": int(req_idx),
            "mode": sr_mode,
            "residual_policy": residual_policy,
            "non_draft_policy": non_draft_policy,
            "threshold": float(cutoff),
            "draft_count": int(draft_count),
            "scheduled_count": int(scheduled_count),
            "cu_scheduled_count": int(cu_scheduled_count),
            "row_start": int(start),
            "valid_rows": int(valid_rows),
            "bonus_row": int(bonus_row) if has_bonus_row else None,
            "generated_len": (
                int(generated_len) if generated_len is not None else None
            ),
            "mask_state": mask_state,
            "batched_mask_applied": bool(batched_mask_applied),
            "scores": score_values,
            "residual_mask": mask_values,
            "bonus_uses_residual": bonus_uses_residual,
        }
    )


def _stats_interval() -> int:
    try:
        return max(0, int(os.getenv("SPECLINK_SR24_STATS_INTERVAL", "1")))
    except ValueError:
        return 1


def runtime_stats_enabled() -> bool:
    return not _env_flag("SPECLINK_SR24_DISABLE_RUNTIME_STATS")


def runtime_timing_enabled() -> bool:
    return runtime_stats_enabled() and _env_flag(
        "SPECLINK_SR24_RUNTIME_TIMING", "1"
    )


def breakdown_enabled() -> bool:
    return _env_flag("SPECLINK_SR24_BREAKDOWN")


def breakdown_linear_enabled() -> bool:
    return breakdown_enabled() and _env_flag("SPECLINK_SR24_BREAKDOWN_LINEAR")


def breakdown_exact_routing() -> bool:
    return breakdown_enabled() and _env_flag("SPECLINK_SR24_BREAKDOWN_EXACT_ROUTING")


def breakdown_sync_counts() -> bool:
    return breakdown_enabled() and (
        breakdown_exact_routing()
        or _env_flag("SPECLINK_SR24_BREAKDOWN_SYNC_COUNTS")
    )


def breakdown_gpu_counts_enabled() -> bool:
    return breakdown_enabled() and _env_flag("SPECLINK_SR24_BREAKDOWN_GPU_COUNTS")


def breakdown_interval() -> int:
    raw = os.getenv("SPECLINK_SR24_BREAKDOWN_INTERVAL", "2000").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2000


def breakdown_snapshot_interval() -> int:
    raw = os.getenv("SPECLINK_SR24_BREAKDOWN_SNAPSHOT_INTERVAL", "20").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 20


def breakdown_path() -> Path | None:
    raw = os.getenv("SPECLINK_SR24_BREAKDOWN_PATH", "").strip()
    if raw:
        return Path(raw)
    stats_path = os.getenv("SPECLINK_SR24_STATS_PATH", "").strip()
    if stats_path:
        return Path(stats_path).with_name("speclink_sr24_breakdown.json")
    return None


def _module_leaf(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def _set_module_profile_metadata(module: Any, module_name: str) -> None:
    module._speclink_sr24_profile_module = module_name
    module._speclink_sr24_profile_leaf = _module_leaf(module_name)
    module._speclink_sr24_profile_layer = _layer_index(module_name)


def _set_sparse_weight_profile_metadata(weight: Any, module: Any) -> None:
    try:
        weight._speclink_sr24_profile_module = getattr(
            module, "_speclink_sr24_profile_module", ""
        )
        weight._speclink_sr24_profile_leaf = getattr(
            module, "_speclink_sr24_profile_leaf", ""
        )
        weight._speclink_sr24_profile_layer = getattr(
            module, "_speclink_sr24_profile_layer", None
        )
    except Exception:
        pass


def _device_constant(
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    key = (name, str(device))
    cached = _device_constant_cache.get(key)
    if cached is None:
        cached = tensor.to(device=device, non_blocking=True)
        _device_constant_cache[key] = cached
    return cached


def _torch_is_compiling() -> bool:
    try:
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if is_compiling is not None and bool(is_compiling()):
            return True
    except Exception:
        pass
    try:
        dynamo = getattr(torch, "_dynamo", None)
        is_compiling = getattr(dynamo, "is_compiling", None)
        return bool(is_compiling()) if is_compiling is not None else False
    except Exception:
        return False


def _device_arange(
    length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    if _torch_is_compiling():
        return torch.arange(length, dtype=dtype, device=device)
    key = (str(device), str(dtype))
    cached = _device_arange_cache.get(key)
    if cached is None or int(cached.numel()) < length:
        capacity = 1 << (length - 1).bit_length()
        cached = torch.arange(capacity, dtype=dtype, device=device)
        _device_arange_cache[key] = cached
    return cached[:length]


def _mask_bits_for(device: torch.device) -> torch.Tensor:
    return _device_constant("mask_bits", _MASK_BITS, device)


def _bit_counts_for(device: torch.device) -> torch.Tensor:
    return _device_constant("bit_counts", _BIT_COUNTS, device)


def _target_leafs() -> set[str]:
    raw = os.getenv("SPECLINK_SR24_TARGET_LEAFS", "").strip()
    if not raw:
        return TARGET_LEAFS
    leafs = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(leafs - TARGET_LEAFS)
    if unknown:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_TARGET_LEAFS entries="
            f"{unknown}; supported={_TARGET_LEAFS_CSV}"
        )
    if not leafs:
        raise RuntimeError("SPECLINK_SR24_TARGET_LEAFS did not select any modules")
    return leafs


def _residual_target_leafs(target_leafs: set[str] | None = None) -> set[str]:
    raw = os.getenv("SPECLINK_SR24_RESIDUAL_TARGET_LEAFS", "").strip()
    if not raw:
        return set(target_leafs) if target_leafs is not None else _target_leafs()
    if raw.lower() in {"none", "base_only"}:
        return set()
    leafs = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(leafs - TARGET_LEAFS)
    if unknown:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_RESIDUAL_TARGET_LEAFS entries="
            f"{unknown}; supported={_TARGET_LEAFS_CSV}"
        )
    if target_leafs is not None:
        outside = sorted(leafs - target_leafs)
        if outside:
            raise RuntimeError(
                "SPECLINK_SR24_RESIDUAL_TARGET_LEAFS must be a subset of "
                f"SPECLINK_SR24_TARGET_LEAFS; outside={outside}"
            )
    return leafs


def _parse_layer_ids_env(name: str) -> set[int] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    layer_ids: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise RuntimeError(
                    f"{name} has descending range {item}; expected start<=end"
                )
            layer_ids.update(range(start, end + 1))
        else:
            layer_ids.add(int(item))
    if any(layer < 0 for layer in layer_ids):
        raise RuntimeError(f"{name} cannot contain negative layer ids")
    return layer_ids


def _parse_layer_ids_by_leaf_env(name: str) -> dict[str, set[int]]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    result: dict[str, set[int]] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            leaf, spec = entry.split("=", 1)
        elif ":" in entry:
            leaf, spec = entry.split(":", 1)
        else:
            raise RuntimeError(
                f"{name} entry {entry!r} must use leaf=ids or leaf:ids"
            )
        leaf = leaf.strip()
        if leaf not in TARGET_LEAFS:
            raise RuntimeError(
                f"{name} has unsupported leaf {leaf!r}; supported={_TARGET_LEAFS_CSV}"
            )
        layer_ids: set[int] = set()
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_raw, end_raw = item.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    raise RuntimeError(
                        f"{name} has descending range {item}; expected start<=end"
                    )
                layer_ids.update(range(start, end + 1))
            else:
                layer_ids.add(int(item))
        if any(layer < 0 for layer in layer_ids):
            raise RuntimeError(f"{name} cannot contain negative layer ids")
        result[leaf] = layer_ids
    return result


def _layer_index(module_name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def _module_is_skipped(module_name: str) -> bool:
    lowered = module_name.lower()
    return (
        module_name == "lm_head"
        or module_name.endswith(".lm_head")
        or "embed_tokens" in lowered
        or "embedding" in lowered
        or ".wte" in lowered
    )


def _iter_target_modules(model: Any) -> list[tuple[str, Any, torch.Tensor]]:
    modules: list[tuple[str, Any, torch.Tensor]] = []
    target_leafs = _target_leafs()
    for name, module in model.named_modules():
        if _module_is_skipped(name) or _module_leaf(name) not in target_leafs:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        usable_in = (int(weight.shape[1]) // 4) * 4
        if usable_in > 0:
            modules.append((name, module, weight))
    return modules


def _compute_keep_mask_24(weight: torch.Tensor) -> torch.Tensor:
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    score = weight[:, :usable_in].detach().abs().float().view(
        out_features, usable_in // 4, 4
    )
    keep_idx = score.topk(k=2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    return keep


def _pack_group_bytes(group_bytes: torch.Tensor) -> torch.Tensor:
    group_bytes = group_bytes.to(dtype=torch.uint8)
    groups = int(group_bytes.shape[1])
    if groups % 2:
        pad = torch.zeros(
            (int(group_bytes.shape[0]), 1),
            dtype=torch.uint8,
            device=group_bytes.device,
        )
        group_bytes = torch.cat([group_bytes, pad], dim=1)
    return (group_bytes[:, 0::2] | (group_bytes[:, 1::2] << 4)).cpu()


def _pack_keep_mask(keep: torch.Tensor) -> torch.Tensor:
    bits = _mask_bits_for(keep.device)
    group_bytes = (
        keep.to(torch.uint8) * bits.view(1, 1, 4)
    ).sum(dim=-1).to(torch.uint8)
    return _pack_group_bytes(group_bytes)


def _expand_mask_bytes(
    mask_bytes: torch.Tensor,
    *,
    out_features: int,
    groups: int,
    device: torch.device,
) -> torch.Tensor:
    if tuple(mask_bytes.shape) == (out_features, groups):
        return mask_bytes.to(device=device, dtype=torch.uint8, non_blocking=True)
    packed_expected = (out_features, (groups + 1) // 2)
    if tuple(mask_bytes.shape) != packed_expected:
        raise RuntimeError(
            f"SpecLink SR24 mask shape {tuple(mask_bytes.shape)} does not match "
            f"{(out_features, groups)} or packed {packed_expected}"
        )
    packed = mask_bytes.to(device=device, dtype=torch.uint8, non_blocking=True)
    unpacked = torch.empty(
        (out_features, packed_expected[1] * 2),
        dtype=torch.uint8,
        device=device,
    )
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    return unpacked[:, :groups]


def _load_mask_cache(mask_path: str) -> dict[str, Any]:
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"missing SR24 mask cache: {path}")
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict) or "masks" not in cache:
        raise RuntimeError(f"invalid SR24 mask cache: {path}")
    return cache


def _cache_mask_for_module(
    module_name: str,
    cache: dict[str, Any],
) -> torch.Tensor | None:
    masks = cache.get("masks", {})
    if module_name in masks:
        return masks[module_name]
    leaf = _module_leaf(module_name)
    candidate_leafs = FUSED_CACHE_LEAFS.get(leaf)
    if not candidate_leafs:
        return None
    prefix = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    parts = []
    for candidate_leaf in candidate_leafs:
        candidate = f"{prefix}.{candidate_leaf}" if prefix else candidate_leaf
        mask = masks.get(candidate)
        if mask is None:
            return None
        parts.append(mask)
    return torch.cat(parts, dim=0)


def _unpacked_group_bytes_to_keep(
    group_bytes: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    bits = _mask_bits_for(device)
    return (
        group_bytes.to(device=device, dtype=torch.uint8).unsqueeze(-1)
        & bits.view(1, 1, 4)
    ).ne(0)


def _mask_is_24(group_bytes: torch.Tensor) -> bool:
    counts = _bit_counts_for(group_bytes.device)
    low = counts[group_bytes.to(torch.long) & 0x0F]
    return bool((low == 2).all().item())


def _keep_mask_is_24(keep: torch.Tensor, *, chunk_rows: int = 512) -> bool:
    # Avoid allocating a full [out_features, groups] reduction tensor on GPU.
    # With GPU-resident residual values, model-load memory is tight enough that
    # even a modest CUDA reduction temporary can OOM on large fused Llama
    # projections. This is an attach-time validation path, so copying small
    # chunks to CPU is preferable to spending scarce GPU memory.
    rows = int(keep.shape[0])
    if keep.is_cuda:
        chunk_rows = min(max(1, int(chunk_rows)), 32)
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            chunk = keep[start:end].detach().to(device="cpu", non_blocking=False)
            if not bool((chunk.sum(dim=-1) == 2).all().item()):
                return False
        return True
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        if not bool((keep[start:end].sum(dim=-1) == 2).all().item()):
            return False
    return True


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_log(record: dict[str, Any]) -> None:
    path = log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _breakdown_cpu_start() -> float:
    return time.perf_counter() if breakdown_enabled() else 0.0


def _breakdown_record_cpu(name: str, start: float) -> None:
    if not start or not breakdown_enabled():
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with _breakdown_lock:
        _breakdown_accum["cpu_ms"][name] += elapsed_ms
        _breakdown_accum["cpu_calls"][name] += 1


def _breakdown_cuda_start() -> torch.cuda.Event | None:
    if not breakdown_enabled() or not torch.cuda.is_available():
        return None
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    return start


def _breakdown_record_cuda(
    name: str,
    start: torch.cuda.Event | None,
    *,
    calls: int = 1,
) -> None:
    _breakdown_record_cuda_many([name], start, calls=calls)


def _breakdown_record_cuda_many(
    names: list[str],
    start: torch.cuda.Event | None,
    *,
    calls: int = 1,
) -> None:
    if start is None or not breakdown_enabled() or not torch.cuda.is_available():
        return
    names = [name for name in names if name]
    if not names:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    with _breakdown_lock:
        for name in names:
            _breakdown_accum["pending_cuda"].append((name, start, end, int(calls)))
        _breakdown_accum["pending_cuda_events"] += len(names)


def _breakdown_count(name: str, value: float | int = 1) -> None:
    if not breakdown_enabled():
        return
    with _breakdown_lock:
        _breakdown_accum["counts"][name] += float(value)


def _breakdown_count_gpu(name: str, value: torch.Tensor | float | int) -> None:
    if not breakdown_gpu_counts_enabled():
        return
    if not isinstance(value, torch.Tensor):
        _breakdown_count(name, value)
        return
    if value.numel() != 1:
        value = value.sum()
    if not value.is_cuda:
        _breakdown_count(name, float(value.detach().item()))
        return
    value = value.detach().to(dtype=torch.float32)
    device = value.device
    device_key = f"{device.type}:{device.index if device.index is not None else 0}"
    key = (device_key, name)
    with _breakdown_lock:
        counter = _breakdown_gpu_counts.get(key)
        if counter is None or counter.device != device:
            counter = torch.zeros((), device=device, dtype=torch.float32)
            _breakdown_gpu_counts[key] = counter
        counter.add_(value)


def _breakdown_gpu_counts_snapshot_locked() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for (_device_key, name), counter in list(_breakdown_gpu_counts.items()):
        try:
            value = float(counter.detach().cpu().item())
        except Exception:
            continue
        snapshot[name] = snapshot.get(name, 0.0) + value
    return snapshot


def _module_profile_suffixes(module: Any) -> list[str]:
    if not breakdown_linear_enabled():
        return []
    leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    layer = getattr(module, "_speclink_sr24_profile_layer", None)
    suffixes: list[str] = []
    if leaf:
        suffixes.append(f"by_leaf_{leaf}")
    if leaf and isinstance(layer, int):
        if layer < 8:
            bucket = "0_7"
        elif layer < 16:
            bucket = "8_15"
        else:
            bucket = "16_31"
        suffixes.append(f"by_leaf_{leaf}__layers_{bucket}")
    return suffixes


def _breakdown_record_cuda_module(
    module: Any,
    name: str,
    start: torch.cuda.Event | None,
    *,
    calls: int = 1,
) -> None:
    aliases = [name, *(f"{name}__{suffix}" for suffix in _module_profile_suffixes(module))]
    _breakdown_record_cuda_many(aliases, start, calls=calls)


def _breakdown_count_module(
    module: Any,
    name: str,
    value: float | int = 1,
) -> None:
    _breakdown_count(name, value)
    for suffix in _module_profile_suffixes(module):
        _breakdown_count(f"{name}__{suffix}", value)


def _breakdown_count_bucket(
    *,
    rows: int,
    bucket_rows: torch.Tensor | None,
    bucket_values: torch.Tensor | None,
) -> None:
    if not breakdown_enabled() or bucket_rows is None or bucket_values is None:
        return
    bucket_count = int(bucket_rows.numel())
    _breakdown_count("bucket_calls", 1)
    _breakdown_count("bucket_candidate_rows", bucket_count)
    _breakdown_count("bucket_total_rows", rows)
    if bucket_count > 0 and breakdown_gpu_counts_enabled():
        _breakdown_count_gpu(
            "gpu_bucket_active_rows",
            bucket_values.to(dtype=torch.int32).sum(),
        )
    if bucket_count <= 0 or not breakdown_sync_counts():
        return
    try:
        active = int(bucket_values.to(dtype=torch.int32).sum().item())
    except Exception:
        active = 0
    _breakdown_count("bucket_active_rows", active)


def _breakdown_count_routing_gpu(
    *,
    residual_mask: torch.Tensor | None,
    rows: int,
    total_valid_draft_tokens: int,
    non_draft_tokens: int,
    residual_non_draft_tokens: int,
) -> None:
    if not breakdown_gpu_counts_enabled() or residual_mask is None:
        return
    rows = int(rows)
    if rows <= 0:
        return
    mask = residual_mask[:rows].to(dtype=torch.int32)
    residual_total = mask.sum()
    device = residual_total.device
    residual_non_draft = torch.as_tensor(
        float(max(0, int(residual_non_draft_tokens))),
        dtype=torch.float32,
        device=device,
    )
    draft_total = torch.as_tensor(
        float(max(0, int(total_valid_draft_tokens))),
        dtype=torch.float32,
        device=device,
    )
    non_draft_total = torch.as_tensor(
        float(max(0, int(non_draft_tokens))),
        dtype=torch.float32,
        device=device,
    )
    residual_draft = (residual_total.to(dtype=torch.float32) -
                      residual_non_draft).clamp(min=0.0)
    base_draft = (draft_total - residual_draft).clamp(min=0.0)
    base_non_draft = (non_draft_total - residual_non_draft).clamp(min=0.0)
    _breakdown_count_gpu("gpu_residual_tokens", residual_total)
    _breakdown_count_gpu("gpu_base_only_tokens", float(rows) - residual_total)
    _breakdown_count_gpu("gpu_residual_draft_tokens", residual_draft)
    _breakdown_count_gpu("gpu_base_only_draft_tokens", base_draft)
    _breakdown_count_gpu("gpu_residual_non_draft_tokens", residual_non_draft)
    _breakdown_count_gpu("gpu_base_only_non_draft_tokens", base_non_draft)


def _breakdown_snapshot_locked(*, include_gpu_counts: bool = False) -> dict[str, Any]:
    cpu_ms = dict(_breakdown_accum["cpu_ms"])
    cpu_calls = dict(_breakdown_accum["cpu_calls"])
    cuda_ms = dict(_breakdown_accum["cuda_ms"])
    cuda_calls = dict(_breakdown_accum["cuda_calls"])
    cuda_events = dict(_breakdown_accum["cuda_events"])
    counts = dict(_breakdown_accum["counts"])
    if include_gpu_counts:
        for key, value in _breakdown_gpu_counts_snapshot_locked().items():
            counts[key] = counts.get(key, 0.0) + value
    bucket_active_rows = counts.get(
        "bucket_active_rows",
        counts.get("gpu_bucket_active_rows"),
    )
    return {
        "timestamp": time.time(),
        "cpu_ms": cpu_ms,
        "cpu_calls": cpu_calls,
        "cpu_avg_ms": {
            key: cpu_ms[key] / cpu_calls[key]
            for key in cpu_ms
            if cpu_calls.get(key)
        },
        "cuda_ms": cuda_ms,
        "cuda_calls": cuda_calls,
        "cuda_events": cuda_events,
        "cuda_avg_ms": {
            key: cuda_ms[key] / cuda_calls[key]
            for key in cuda_ms
            if cuda_calls.get(key)
        },
        "counts": counts,
        "derived": {
            "bucket_fill_ratio": (
                bucket_active_rows
                / counts.get("bucket_candidate_rows", 1.0)
                if (
                    counts.get("bucket_candidate_rows", 0.0)
                    and bucket_active_rows is not None
                )
                else None
            ),
            "avg_bucket_candidate_rows": (
                counts.get("bucket_candidate_rows", 0.0)
                / counts.get("bucket_calls", 1.0)
                if counts.get("bucket_calls", 0.0)
                else None
            ),
            "avg_bucket_active_rows": (
                bucket_active_rows
                / counts.get("bucket_calls", 1.0)
                if counts.get("bucket_calls", 0.0)
                and bucket_active_rows is not None
                else None
            ),
            "avg_scheduled_tokens_per_step": (
                counts.get("scheduled_tokens", 0.0)
                / counts.get("verify_steps", 1.0)
                if counts.get("verify_steps", 0.0)
                else None
            ),
        },
        "flushes": int(_breakdown_accum.get("flushes") or 0),
    }


def _flush_breakdown(*, force: bool = False) -> None:
    if not breakdown_enabled():
        return
    path = breakdown_path()
    if path is None:
        return
    snapshot_payload: dict[str, Any] | None = None
    with _breakdown_lock:
        pending = list(_breakdown_accum["pending_cuda"])
        if (
            not force
            and len(pending) < breakdown_interval()
            and int(_breakdown_accum.get("pending_cuda_events") or 0)
            < breakdown_interval()
        ):
            steps = int(_breakdown_accum["counts"].get("verify_steps", 0.0))
            last_snapshot = int(_breakdown_accum.get("last_snapshot_steps") or 0)
            if (
                steps <= 0
                or (path.exists() and steps - last_snapshot < breakdown_snapshot_interval())
            ):
                return
            _breakdown_accum["last_snapshot_steps"] = steps
            snapshot_payload = _breakdown_snapshot_locked(
                include_gpu_counts=breakdown_gpu_counts_enabled()
            )
            if pending and breakdown_linear_enabled():
                # Linear component profiling is explicitly sync-heavy. vLLM's
                # server is normally terminated by the benchmark harness, so
                # relying on the atexit force-flush can lose pending CUDA
                # events. Drain them during periodic snapshots for diagnostic
                # runs, while keeping normal breakdown snapshots low-overhead.
                _breakdown_accum["pending_cuda"] = []
                _breakdown_accum["pending_cuda_events"] = 0
                snapshot_payload = None
        else:
            _breakdown_accum["pending_cuda"] = []
            _breakdown_accum["pending_cuda_events"] = 0
    if snapshot_payload is not None:
        try:
            _write_json(path, snapshot_payload)
        except Exception:
            pass
        return
    if pending and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pending = []
    with _breakdown_lock:
        for name, start, end, calls in pending:
            try:
                elapsed_ms = float(start.elapsed_time(end))
            except Exception:
                continue
            _breakdown_accum["cuda_ms"][name] += elapsed_ms
            _breakdown_accum["cuda_calls"][name] += max(1, int(calls))
            _breakdown_accum["cuda_events"][name] += 1
        _breakdown_accum["flushes"] = int(_breakdown_accum.get("flushes") or 0) + 1
        payload = _breakdown_snapshot_locked(include_gpu_counts=True)
    if (
        not payload["counts"]
        and not payload["cpu_ms"]
        and not payload["cuda_ms"]
        and path.exists()
    ):
        return
    try:
        _write_json(path, payload)
    except Exception:
        pass


def _flush_breakdown_at_exit() -> None:
    try:
        _flush_breakdown(force=True)
    except Exception:
        pass


atexit.register(_flush_breakdown_at_exit)


def _static_residual_mask_view(
    rows: int,
    device: torch.device,
    *,
    fill_value: bool | None = None,
) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    key = str(device)
    with _lock:
        buffer = _static_mask_buffers.get(key)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows:
            buffer = torch.empty(capacity, dtype=torch.bool, device=device)
            if fill_value is None:
                buffer.fill_(True)
            _static_mask_buffers[key] = buffer
    view = buffer[:rows]
    if fill_value is not None:
        view.fill_(fill_value)
    return view


def _static_residual_priority_view(
    rows: int,
    device: torch.device,
    *,
    fill_value: float | None = None,
) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    key = str(device)
    with _lock:
        buffer = _static_priority_buffers.get(key)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows:
            buffer = torch.empty(capacity, dtype=torch.float32, device=device)
            if fill_value is None:
                buffer.zero_()
            _static_priority_buffers[key] = buffer
    view = buffer[:rows]
    if fill_value is not None:
        view.fill_(float(fill_value))
    return view


def _static_int32_view(name: str, rows: int, device: torch.device) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    key = (str(device), str(name))
    with _lock:
        buffer = _static_int32_buffers.get(key)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows:
            buffer = torch.empty(capacity, dtype=torch.int32, device=device)
            _static_int32_buffers[key] = buffer
    return buffer[:rows]


def _static_long_view(name: str, rows: int, device: torch.device) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    key = (str(device), str(name))
    with _lock:
        buffer = _static_long_buffers.get(key)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows:
            buffer = torch.empty(capacity, dtype=torch.long, device=device)
            _static_long_buffers[key] = buffer
    return buffer[:rows]


def _static_float_view(
    name: str,
    rows: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    key = (str(device), f"{name}:{str(dtype)}")
    with _lock:
        buffer = _static_float_buffers.get(key)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows or buffer.dtype != dtype:
            buffer = torch.empty(capacity, dtype=dtype, device=device)
            _static_float_buffers[key] = buffer
    return buffer[:rows]


def _static_residual_bucket_capture_view(
    *,
    rows: int,
    device: torch.device,
    value_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    rows = int(rows)
    active_hint = cudagraph_bucket_active_hint()
    bucket_size = _effective_residual_bucket_size(
        active_count=active_hint if active_hint > 0 else None
    )
    if bucket_size <= 0 or rows <= 0:
        return None
    k = min(rows, bucket_size)
    bucket_rows = _static_long_view("bucket_rows", k, device)
    bucket_values = _static_float_view(
        "bucket_values",
        k,
        device,
        dtype=value_dtype,
    )
    bucket_rows.copy_(
        torch.arange(k, dtype=torch.long, device=device),
        non_blocking=False,
    )
    bucket_values.fill_(1.0)
    return bucket_rows, bucket_values


def _static_bucket_complement_capture_view(
    *,
    rows: int,
    bucket_rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    rows = int(rows)
    bucket_count = int(bucket_rows.numel())
    if rows <= 0:
        return None
    if bucket_count <= 0:
        base_rows = _static_long_view("bucket_complement_base_rows", rows, device)
        base_rows.copy_(
            torch.arange(rows, dtype=torch.long, device=device),
            non_blocking=False,
        )
        return base_rows
    if bucket_count >= rows:
        return torch.empty(0, dtype=torch.long, device=device)
    base_rows = _static_long_view(
        "bucket_complement_base_rows",
        rows - bucket_count,
        device,
    )
    base_rows.copy_(
        torch.arange(bucket_count, rows, dtype=torch.long, device=device),
        non_blocking=False,
    )
    return base_rows


@triton.jit
def _bucket_complement_rows_kernel(
    sorted_bucket_rows,
    base_rows,
    output_count,
    bucket_count: tl.constexpr,
    block_m: tl.constexpr,
) -> None:
    offs = tl.program_id(0) * block_m + tl.arange(0, block_m)
    valid = offs < output_count
    row = offs
    for bucket_idx in tl.static_range(0, bucket_count):
        bucket_row = tl.load(sorted_bucket_rows + bucket_idx)
        row += tl.where(row >= bucket_row, 1, 0)
    tl.store(base_rows + offs, row, mask=valid)


def _triton_bucket_complement_rows(
    *,
    rows: int,
    device: torch.device,
    bucket_rows: torch.Tensor,
    static_output: bool,
) -> torch.Tensor | None:
    rows = int(rows)
    bucket_count = int(bucket_rows.numel())
    if rows <= 0:
        return None
    if bucket_count <= 0:
        base_rows = (
            _static_long_view("bucket_complement_base_rows", rows, device)
            if static_output
            else torch.empty(rows, dtype=torch.long, device=device)
        )
        base_rows.copy_(
            torch.arange(rows, dtype=torch.long, device=device),
            non_blocking=False,
        )
        return base_rows
    if bucket_count >= rows:
        return torch.empty(0, dtype=torch.long, device=device)
    if not bucket_rows.is_cuda:
        return None
    output_count = rows - bucket_count
    sorted_rows = torch.sort(
        bucket_rows.to(device=device, dtype=torch.long),
        stable=True,
    ).values
    base_rows = (
        _static_long_view("bucket_complement_base_rows", output_count, device)
        if static_output
        else torch.empty(output_count, dtype=torch.long, device=device)
    )
    block_m = 256
    grid = (triton.cdiv(output_count, block_m),)
    _bucket_complement_rows_kernel[grid](
        sorted_rows,
        base_rows,
        output_count=output_count,
        bucket_count=bucket_count,
        block_m=block_m,
    )
    _breakdown_count("scheduler_bucket_base_rows_triton_builds", 1)
    _breakdown_count("scheduler_bucket_base_rows_triton_rows", output_count)
    return base_rows


def _static_int32_cpu_view(name: str, rows: int) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=torch.int32, device="cpu")
    with _lock:
        buffer = _static_int32_cpu_buffers.get(name)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows:
            try:
                buffer = torch.empty(
                    capacity,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=torch.cuda.is_available(),
                )
            except Exception:
                buffer = torch.empty(capacity, dtype=torch.int32, device="cpu")
            _static_int32_cpu_buffers[name] = buffer
    return buffer[:rows]


def _static_long_cpu_view(name: str, rows: int) -> torch.Tensor:
    rows = int(rows)
    if rows <= 0:
        return torch.empty(0, dtype=torch.long, device="cpu")
    with _lock:
        buffer = _static_long_cpu_buffers.get(name)
        capacity = max(static_mask_buffer_capacity(), rows)
        if buffer is None or int(buffer.numel()) < rows:
            try:
                buffer = torch.empty(
                    capacity,
                    dtype=torch.long,
                    device="cpu",
                    pin_memory=torch.cuda.is_available(),
                )
            except Exception:
                buffer = torch.empty(capacity, dtype=torch.long, device="cpu")
            _static_long_cpu_buffers[name] = buffer
    return buffer[:rows]


def _copy_int32_values_to_device(
    *,
    name: str,
    dst: torch.Tensor,
    values: list[int],
) -> None:
    rows = len(values)
    if rows <= 0:
        return
    src = _static_int32_cpu_view(name, rows)
    src.numpy()[:rows] = values
    dst.copy_(src, non_blocking=True)


def _copy_long_values_to_device(
    *,
    name: str,
    dst: torch.Tensor,
    values: list[int],
) -> None:
    rows = len(values)
    if rows <= 0:
        return
    src = _static_long_cpu_view(name, rows)
    src.numpy()[:rows] = values
    dst.copy_(src, non_blocking=True)


def _semi_structured_storage_bytes(
    tensor: Any,
    *,
    logical_value_bytes: int | None = None,
) -> tuple[int, int]:
    value_bytes = 0
    meta_bytes = 0
    seen: set[int] = set()

    def add_tensor(item: Any, *, metadata: bool) -> None:
        nonlocal value_bytes, meta_bytes
        if not isinstance(item, torch.Tensor):
            return
        ptr = int(item.untyped_storage().data_ptr())
        if ptr in seen:
            return
        seen.add(ptr)
        size = int(item.untyped_storage().nbytes())
        if metadata:
            meta_bytes += size
        else:
            value_bytes += size

    for attr in ("values", "packed", "packed_t"):
        try:
            item = getattr(tensor, attr)
            add_tensor(item() if callable(item) else item, metadata=False)
        except Exception:
            pass
    for attr in ("indices", "meta", "meta_t"):
        try:
            item = getattr(tensor, attr)
            add_tensor(item() if callable(item) else item, metadata=True)
        except Exception:
            pass
    if (
        logical_value_bytes is not None
        and meta_bytes == 0
        and value_bytes > logical_value_bytes
    ):
        meta_bytes = value_bytes - logical_value_bytes
        value_bytes = logical_value_bytes
    return value_bytes, meta_bytes


def _to_sparse_semi_structured(weight: torch.Tensor) -> Any:
    from torch.sparse import to_sparse_semi_structured

    return to_sparse_semi_structured(weight.contiguous())


def _replace_module_weight_with_sparse_base(module: Any, sparse_base: Any) -> None:
    # The SR24 sparse Llama path bypasses the module's regular forward, so the
    # dense Parameter is replaced after model loading to avoid keeping a second
    # dense copy. Keep the `weight` key in `_parameters` when it exists because
    # vLLM's compile path expects the module parameter layout to stay stable.
    if hasattr(module, "_parameters") and "weight" in module._parameters:
        module._parameters["weight"] = sparse_base
    else:
        module.weight = sparse_base


def _sparse_base_weight(module: Any) -> Any:
    return getattr(module, "_speclink_sr24_sparse_base_weight", module.weight)


def _extract_residual_values_chunked(
    weight_view: torch.Tensor,
    keep: torch.Tensor,
    *,
    storage_device: torch.device,
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, int]:
    out_features, groups, _ = weight_view.shape
    values_per_row = groups * 2
    chunk_rows = residual_extract_chunk_rows() if chunk_rows is None else chunk_rows
    chunk_rows = max(1, int(chunk_rows))
    residual_values = torch.empty(
        out_features * values_per_row,
        device=storage_device,
        dtype=weight_view.dtype,
    )
    offset = 0
    cpu_fallback_chunks = 0
    for start in range(0, out_features, chunk_rows):
        end = min(out_features, start + chunk_rows)
        try:
            chunk_values = weight_view[start:end][~keep[start:end]].contiguous()
        except torch.OutOfMemoryError:
            if weight_view.device.type == "cuda":
                torch.cuda.empty_cache()
            cpu_fallback_chunks += 1
            chunk_weight = weight_view[start:end].detach().cpu()
            chunk_keep = keep[start:end].detach().cpu()
            chunk_values = chunk_weight[~chunk_keep].contiguous()
        next_offset = offset + int(chunk_values.numel())
        if chunk_values.device != storage_device:
            chunk_values = chunk_values.to(device=storage_device, non_blocking=False)
        residual_values[offset:next_offset].copy_(
            chunk_values,
            non_blocking=False,
        )
        offset = next_offset
    if offset != int(residual_values.numel()):
        raise RuntimeError(
            "SR24 residual extraction size mismatch: "
            f"filled={offset}, expected={int(residual_values.numel())}"
        )
    return residual_values.contiguous(), cpu_fallback_chunks


def _attach_sr24_module(
    *,
    module_name: str,
    module: Any,
    weight: torch.Tensor,
    keep: torch.Tensor,
    method: str,
    force_no_residual: bool = False,
    force_dense_fastpath: bool = False,
    base_only_dense_verify_layer_ids: set[int] | None = None,
    base_only_dense_verify_layer_ids_by_leaf: dict[str, set[int]] | None = None,
) -> dict[str, Any]:
    _set_module_profile_metadata(module, module_name)
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    param_device = weight.device
    param_dtype = weight.dtype
    param_element_size = weight.element_size()
    keep = keep.to(device=param_device, dtype=torch.bool)
    sr_backend = backend()
    if sr_backend not in SR24_BACKENDS:
        raise RuntimeError(f"unsupported SPECLINK_SR24_BACKEND={sr_backend}")
    module_leaf = _module_leaf(module_name)
    layer_index = _layer_index(module_name)
    dense_verify_layer_ids_by_leaf = (
        base_only_dense_verify_layer_ids_by_leaf or {}
    )
    dense_verify_scope_match = _layer_in_scope(
        layer_index,
        module_leaf,
        base_only_dense_verify_layer_ids,
        dense_verify_layer_ids_by_leaf,
    )
    sr_residual_backend = residual_backend_by_leaf().get(
        module_leaf,
        residual_backend(),
    )
    if sr_residual_backend not in {"compressed_dense", "torch_sparse", "dense_rows"}:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_RESIDUAL_BACKEND="
            f"{sr_residual_backend}"
        )
    if sr_backend in DENSE_ZERO_BACKENDS and sr_residual_backend == "torch_sparse":
        raise RuntimeError(
            "SPECLINK_SR24_RESIDUAL_BACKEND=torch_sparse is only supported "
            "with SPECLINK_SR24_BACKEND=torch_sparse"
        )
    sr_residual_device = residual_device()
    if sr_residual_device not in {"cpu", "cuda"}:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_RESIDUAL_DEVICE="
            f"{sr_residual_device}"
        )
    sr_selective_policy = selective_residual_policy()
    if sr_selective_policy not in {
        "critical_prefix",
        "all_if_any_low",
        "batch_all_if_any_low",
        "low_confidence",
        "high_confidence",
        "prefix_confidence",
        "fixed_prefix",
    }:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY="
            f"{sr_selective_policy}"
        )
    sr_non_draft_policy = selective_non_draft_policy()
    sr_bonus_priority = bonus_priority()
    if sr_non_draft_policy not in {
        "all",
        "none",
        "bonus",
        "predicted_full_accept",
    }:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY="
            f"{sr_non_draft_policy}"
        )
    if sr_backend == "torch_sparse":
        if usable_in != in_features:
            raise RuntimeError(
                f"SR24 torch_sparse backend requires full in_features to be 2:4 "
                f"groupable for {module_name}; got in_features={in_features}"
            )
        if param_dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                f"SR24 torch_sparse backend requires fp16/bf16 weights for "
                f"{module_name}; got {param_dtype}"
            )
    if tuple(keep.shape) != (out_features, groups, 4):
        raise RuntimeError(
            f"SR24 keep mask for {module_name} has {tuple(keep.shape)}, "
            f"expected {(out_features, groups, 4)}"
        )
    _mask_bits_for(param_device)
    _bit_counts_for(param_device)
    if not _keep_mask_is_24(keep):
        raise RuntimeError(f"SR24 base mask for {module_name} is not 2:4")

    dense_bytes = out_features * in_features * param_element_size
    need_residual = mode() != "base_only" and not force_no_residual
    dense_fastpath = (
        sr_backend == "torch_sparse"
        and need_residual
        and (
            mode() == "all_corrected"
            or force_dense_fastpath
            or (
                mode() == "selective"
                and static_mask_state() == "all_residual"
                and static_all_residual_dense_fastpath()
            )
        )
        and all_corrected_dense_fastpath()
    )
    if dense_fastpath:
        module._speclink_sr24_enabled = True
        module._speclink_sr24_backend = sr_backend
        module._speclink_sr24_dense_fastpath = True
        module._speclink_sr24_weight_shape = (out_features, in_features)
        module._speclink_sr24_weight_dtype = param_dtype
        module._speclink_sr24_usable_in = usable_in
        module._speclink_sr24_mask_method = method
        module._speclink_sr24_residual_backend = "dense_fastpath"
        module._speclink_sr24_residual_device = "none"
        module._speclink_sr24_no_residual = False
        return {
            "module": module_name,
            "leaf": module_leaf,
            "layer": _layer_index(module_name),
            "shape": [out_features, in_features],
            "usable_in": usable_in,
            "base_mask_is_24": True,
            "residual_mask_is_24": True,
            "reconstructs_dense": True,
            "all_corrected_dense_fastpath": True,
            "mask_method": method,
            "backend": sr_backend,
            "residual_backend": "dense_fastpath",
            "residual_device": "none",
            "dense_weight_bytes": dense_bytes,
            "base_weight_storage_bytes": dense_bytes,
            "residual_value_storage_bytes": 0,
            "residual_extract_chunk_rows": residual_extract_chunk_rows(),
            "residual_bucket_size": residual_bucket_size(),
            "residual_bucket_scale_by_active": residual_bucket_scale_by_active(),
            "residual_bucket_priority": residual_bucket_priority(),
            "direct_position_bucket": direct_position_bucket_enabled(),
            "bucket_dense_copy": bucket_dense_copy(),
            "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
            "bucket_dense_compute_active_only":
            bucket_dense_compute_active_only(),
            "bucket_dense_active_mask_fused":
            bucket_dense_active_mask_fused(),
            "sort_bucket_rows": sort_bucket_rows(),
            "route_bucket_rows": route_bucket_rows(),
            "route_bucket_rows_graph_static_unsafe":
            route_bucket_rows_graph_static_unsafe(),
            "route_all_residual_rows": route_all_residual_rows(),
            "route_all_skip_bucket": route_all_skip_bucket(),
            "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
            "route_reuse_base_output": route_reuse_base_output(),
            "route_contiguous_fastpath": route_contiguous_fastpath(),
            "route_overlap_streams": route_overlap_streams(),
            "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
            "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
            "fixed_prefix_route_descriptor_only":
            fixed_prefix_route_descriptor_only(),
            "fixed_block_input_buffer": fixed_block_input_buffer_enabled(),
            "fixed_block_output_buffer": fixed_block_output_buffer_enabled(),
            "scheduler_policy_path": scheduler_policy_path(),
            "scheduler_policy_allow_legacy_mixed":
            scheduler_policy_allow_legacy_mixed(),
            "scheduler_policy_allow_single_block_packed_parallel":
            scheduler_policy_allow_single_block_packed_parallel(),
            "scheduler_policy_allow_serial_packed_parallel":
            scheduler_policy_allow_serial_packed_parallel(),
            "scheduler_policy_near_full_tolerance":
            scheduler_policy_near_full_tolerance(),
            "fixed_block_capacity_padding":
            fixed_block_capacity_padding_enabled(),
            "fixed_block_capacity_zero_dummy":
            fixed_block_capacity_zero_dummy_enabled(),
            "route_dense_fallback_fraction": route_dense_fallback_fraction(),
            "triton_route_assembly": triton_route_assembly(),
            "triton_bucket_override": triton_bucket_override(),
            "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
            "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
            "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
            "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
            "residual_extract_cpu_fallback_chunks": 0,
            "sparse_metadata_bytes": 0,
            "mask_metadata_bytes": 0,
            "actual_weight_storage_bytes": dense_bytes,
        }

    with torch.no_grad():
        if sr_backend in DENSE_ZERO_BACKENDS:
            keep_dense_for_no_residual = (
                not need_residual
                and (
                    base_only_dense_nonverify()
                    or (mode() == "base_only"
                        and base_only_dense_verify_max_rows() > 0
                        and dense_verify_scope_match)
                )
            )
            dense_weight = (
                weight.detach().clone()
                if (
                    (need_residual and sr_residual_backend == "dense_rows")
                    or keep_dense_for_no_residual
                )
                else None
            )
            original = weight[:, :usable_in].detach().clone()
            dense_tail = (
                weight[:, usable_in:].detach().clone()
                if usable_in < in_features
                else None
            )
            residual_extract_cpu_fallback_chunks = 0
            residual_values = (
                None
                if sr_residual_backend == "dense_rows"
                else original.view(out_features, groups, 4)[~keep].contiguous()
            )
            view = weight[:, :usable_in].view(out_features, groups, 4)
            view.masked_fill_(~keep, 0)
        else:
            dense_tail = None
            # Avoid holding original/base/residual dense clones at once. Save
            # only the complementary values, mask the loaded Parameter in
            # place into W_base, convert it, then optionally build W_residual.
            weight_view = weight[:, :usable_in].view(out_features, groups, 4)
            residual_storage_device = (
                torch.device("cpu")
                if (
                    sr_residual_backend == "compressed_dense"
                    and sr_residual_device == "cpu"
                )
                else param_device
            )
            keep_dense_for_no_residual = (
                not need_residual
                and (
                    base_only_dense_nonverify()
                    or (mode() == "base_only"
                        and base_only_dense_verify_max_rows() > 0
                        and dense_verify_scope_match)
                )
            )
            preserve_dense_rows_weight = (
                need_residual and sr_residual_backend == "dense_rows"
            )
            dense_weight = (
                weight
                if preserve_dense_rows_weight
                else (
                    weight.detach().clone()
                    if keep_dense_for_no_residual
                    else None
                )
            )
            residual_extract_cpu_fallback_chunks = 0
            if (
                sr_residual_backend in {"dense_rows", "torch_sparse"}
                or not need_residual
            ):
                residual_values = None
            else:
                (
                    residual_values,
                    residual_extract_cpu_fallback_chunks,
                ) = _extract_residual_values_chunked(
                    weight_view,
                    keep,
                    storage_device=residual_storage_device,
                )
            residual_dense = None
            residual_view = None
            if need_residual and sr_residual_backend == "torch_sparse":
                # Clone the original dense weight before masking `weight` into
                # W_base.  The residual tensor keeps the complementary 2:4
                # entries and is converted to a second semi-structured tensor
                # after the dense Parameter has been replaced.
                residual_dense = weight.detach().clone()
                residual_view = residual_dense[:, :usable_in].view(
                    out_features, groups, 4
                )
                residual_view.masked_fill_(keep, 0)
            if preserve_dense_rows_weight:
                sparse_dense = weight.detach().clone()
                sparse_view = sparse_dense[:, :usable_in].view(
                    out_features, groups, 4
                )
                sparse_view.masked_fill_(~keep, 0)
                sparse_base = _to_sparse_semi_structured(sparse_dense)
                _set_sparse_weight_profile_metadata(sparse_base, module)
                del sparse_view
                del sparse_dense
            else:
                weight_view.masked_fill_(~keep, 0)
                sparse_base = _to_sparse_semi_structured(weight.detach())
                _set_sparse_weight_profile_metadata(sparse_base, module)
            base_weight_replaced = False
            if need_residual and sr_residual_backend == "torch_sparse":
                _replace_module_weight_with_sparse_base(module, sparse_base)
                base_weight_replaced = True
                del weight_view
                del weight
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if residual_dense is None:
                    raise RuntimeError("SR24 torch_sparse residual staging is missing")
                sparse_residual = _to_sparse_semi_structured(residual_dense)
                _set_sparse_weight_profile_metadata(sparse_residual, module)
                if residual_view is not None:
                    del residual_view
                del residual_dense
            else:
                sparse_residual = None

    mask_bytes = _pack_keep_mask(keep.detach().cpu())
    if sr_backend in DENSE_ZERO_BACKENDS or (
        sr_backend == "torch_sparse"
        and sr_residual_backend == "compressed_dense"
        and sr_residual_device == "cuda"
        and (mode() != "base_only")
    ):
        mask_bytes = mask_bytes.to(device=param_device, non_blocking=True)
    module._speclink_sr24_enabled = True
    module._speclink_sr24_backend = sr_backend
    module._speclink_sr24_base_mask_bytes = mask_bytes
    module._speclink_sr24_weight_shape = (out_features, in_features)
    module._speclink_sr24_weight_dtype = param_dtype
    module._speclink_sr24_usable_in = usable_in
    module._speclink_sr24_mask_method = method
    module._speclink_sr24_residual_backend = sr_residual_backend
    module._speclink_sr24_no_residual = not need_residual
    module._speclink_sr24_residual_device = (
        str(residual_values.device)
        if "residual_values" in locals() and residual_values is not None
        else (
            str(dense_weight.device)
            if "dense_weight" in locals() and dense_weight is not None
            else "none"
        )
    )
    if dense_tail is not None:
        module._speclink_sr24_dense_tail = dense_tail
    if sr_backend in DENSE_ZERO_BACKENDS:
        if "dense_weight" in locals() and dense_weight is not None:
            module._speclink_sr24_dense_weight = dense_weight
        elif residual_values is not None:
            module._speclink_sr24_residual_values = residual_values
    else:
        logical_sparse_value_bytes = (
            out_features * in_features // 2 * param_element_size
        )
        base_value_bytes, base_meta_bytes = _semi_structured_storage_bytes(
            sparse_base,
            logical_value_bytes=logical_sparse_value_bytes,
        )
        if residual_values is not None:
            module._speclink_sr24_residual_values = residual_values
        if dense_weight is not None:
            module._speclink_sr24_dense_weight = dense_weight
        if sparse_residual is not None:
            module._speclink_sr24_residual_sparse = sparse_residual
            residual_value_bytes, residual_meta_bytes = _semi_structured_storage_bytes(
                sparse_residual,
                logical_value_bytes=logical_sparse_value_bytes,
            )
            if hasattr(module, "_speclink_sr24_residual_values"):
                delattr(module, "_speclink_sr24_residual_values")
        else:
            residual_value_bytes = (
                dense_weight.numel() * dense_weight.element_size()
                if dense_weight is not None
                else (
                    residual_values.numel() * residual_values.element_size()
                    if residual_values is not None
                    else 0
                )
            )
            residual_meta_bytes = 0
        if preserve_dense_rows_weight:
            module._speclink_sr24_sparse_base_weight = sparse_base
        elif not base_weight_replaced:
            _replace_module_weight_with_sparse_base(module, sparse_base)
        if (
            need_residual
            and sr_residual_backend == "compressed_dense"
            and cache_compressed_residual_weight()
            and prewarm_compressed_residual_weight()
            and hasattr(module, "_speclink_sr24_residual_values")
            and _is_cuda_device_label(getattr(module, "_speclink_sr24_residual_device", ""))
        ):
            _compressed_residual_weight(
                module,
                dtype=param_dtype,
                device=param_device,
            )
        if (
            need_residual
            and sr_residual_backend == "compressed_dense"
            and compressed_residual_triton()
            and hasattr(module, "_speclink_sr24_residual_values")
            and _is_cuda_device_label(getattr(module, "_speclink_sr24_residual_device", ""))
        ):
            _compressed_residual_cached_tensor(
                module,
                source_attr="_speclink_sr24_residual_values",
                cache_attr="_speclink_sr24_triton_residual_values",
                dtype=param_dtype,
                device=param_device,
                counter_prefix="compressed_residual_triton_values",
            )
            _compressed_residual_cached_tensor(
                module,
                source_attr="_speclink_sr24_base_mask_bytes",
                cache_attr="_speclink_sr24_triton_mask_bytes",
                dtype=torch.uint8,
                device=param_device,
                counter_prefix="compressed_residual_triton_mask",
            )
        if sparse_residual is not None and residual_values is not None:
            del residual_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mask_metadata_bytes = mask_bytes.numel() * mask_bytes.element_size()
    if sr_backend in DENSE_ZERO_BACKENDS:
        base_weight_bytes = weight.numel() * weight.element_size()
        residual_bytes = (
            dense_weight.numel() * dense_weight.element_size()
            if "dense_weight" in locals() and dense_weight is not None
            else (
                residual_values.numel() * residual_values.element_size()
                if residual_values is not None
                else 0
            )
        )
        metadata_bytes = mask_metadata_bytes
        actual_weight_storage_bytes = base_weight_bytes + residual_bytes + metadata_bytes
    else:
        base_weight_bytes = base_value_bytes + base_meta_bytes
        residual_bytes = residual_value_bytes
        metadata_bytes = base_meta_bytes + residual_meta_bytes + mask_metadata_bytes
        actual_weight_storage_bytes = (
            base_value_bytes
            + residual_value_bytes
            + base_meta_bytes
            + residual_meta_bytes
            + mask_metadata_bytes
        )
    return {
        "module": module_name,
        "leaf": module_leaf,
        "layer": layer_index,
        "shape": [out_features, in_features],
        "usable_in": usable_in,
        "base_mask_is_24": True,
        "residual_mask_is_24": True,
        "reconstructs_dense": True,
        "mask_method": method,
        "backend": sr_backend,
        "residual_backend": "none" if not need_residual else sr_residual_backend,
        "residual_device": getattr(module, "_speclink_sr24_residual_device", "none"),
        "dense_weight_bytes": dense_bytes,
        "base_weight_storage_bytes": base_weight_bytes,
        "residual_value_storage_bytes": residual_bytes,
        "residual_extract_chunk_rows": residual_extract_chunk_rows(),
        "residual_bucket_size": residual_bucket_size(),
        "residual_bucket_scale_by_active": residual_bucket_scale_by_active(),
        "cudagraph_bucket": cudagraph_bucket_enabled(),
        "cudagraph_bucket_active_hint": cudagraph_bucket_active_hint(),
        "residual_bucket_priority": residual_bucket_priority(),
        "direct_position_bucket": direct_position_bucket_enabled(),
        "bonus_priority": bonus_priority(),
        "draft_position_priority_scale": draft_position_priority_scale(),
        "bucket_dense_copy": bucket_dense_copy(),
        "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
        "bucket_dense_compute_active_only": bucket_dense_compute_active_only(),
        "bucket_dense_active_mask_fused": bucket_dense_active_mask_fused(),
        "sort_bucket_rows": sort_bucket_rows(),
        "route_bucket_rows": route_bucket_rows(),
        "route_bucket_rows_graph_static_unsafe":
        route_bucket_rows_graph_static_unsafe(),
        "route_all_residual_rows": route_all_residual_rows(),
        "route_all_skip_bucket": route_all_skip_bucket(),
        "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
        "route_reuse_base_output": route_reuse_base_output(),
        "route_contiguous_fastpath": route_contiguous_fastpath(),
        "route_overlap_streams": route_overlap_streams(),
        "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
        "base_only_dense_verify_scope_match": dense_verify_scope_match,
        "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
        "fixed_prefix_route_descriptor_only": fixed_prefix_route_descriptor_only(),
        "fixed_block_input_buffer": fixed_block_input_buffer_enabled(),
        "fixed_block_output_buffer": fixed_block_output_buffer_enabled(),
        "route_dense_fallback_fraction": route_dense_fallback_fraction(),
        "route_min_dense_rows": route_min_dense_rows(),
        "route_min_base_rows": route_min_base_rows(),
        "route_min_base_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
        ),
        "route_max_dense_fraction": route_max_dense_fraction(),
        "adaptive_dense_fallback": adaptive_dense_fallback_enabled(),
        "adaptive_dense_fallback_no_residual_only":
        adaptive_dense_fallback_no_residual_only(),
        "adaptive_dense_fallback_small_rows":
        adaptive_dense_fallback_small_rows(),
        "adaptive_dense_fallback_gate_up_fraction":
        adaptive_dense_fallback_gate_up_fraction(),
        "adaptive_dense_fallback_down_fraction":
        adaptive_dense_fallback_down_fraction(),
        "adaptive_dense_fallback_small_down_no_residual":
        adaptive_dense_fallback_small_down_no_residual(),
        "adaptive_dense_fallback_small_gate_up_no_residual":
        adaptive_dense_fallback_small_gate_up_no_residual(),
        "dense_fallback_nonuniform": dense_fallback_nonuniform(),
        "triton_route_assembly": triton_route_assembly(),
        "triton_bucket_override": triton_bucket_override(),
        "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
        "triton_bucket_scatter": triton_bucket_scatter(),
        "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
        "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
        "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
        "residual_extract_cpu_fallback_chunks":
        residual_extract_cpu_fallback_chunks,
        "sparse_metadata_bytes": metadata_bytes,
        "mask_metadata_bytes": mask_metadata_bytes,
        "actual_weight_storage_bytes": actual_weight_storage_bytes,
    }


def _attach_gate_up_split_module(
    *,
    module_name: str,
    module: Any,
    weight: torch.Tensor,
    split_mode: str,
    method: str,
) -> dict[str, Any]:
    _set_module_profile_metadata(module, module_name)
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    param_device = weight.device
    param_dtype = weight.dtype
    param_element_size = weight.element_size()
    if split_mode not in {"up_sparse", "gate_sparse"}:
        raise RuntimeError(f"unsupported gate_up split mode {split_mode}")
    if out_features % 2 != 0:
        raise RuntimeError(
            f"SR24 gate_up split requires even out_features for {module_name}; "
            f"got {out_features}"
        )
    if backend() != "torch_sparse":
        raise RuntimeError("SR24 gate_up split currently requires torch_sparse backend")
    if usable_in != in_features:
        raise RuntimeError(
            f"SR24 gate_up split requires full in_features to be 2:4 groupable "
            f"for {module_name}; got in_features={in_features}"
        )
    if param_dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            f"SR24 gate_up split requires fp16/bf16 weights for {module_name}; "
            f"got {param_dtype}"
        )

    half = out_features // 2
    if split_mode == "up_sparse":
        dense_name = "gate_proj"
        dense_start = 0
        sparse_name = "up_proj"
        sparse_start = half
    else:
        dense_name = "up_proj"
        dense_start = half
        sparse_name = "gate_proj"
        sparse_start = 0
    dense_end = dense_start + half
    sparse_end = sparse_start + half
    dense_bytes = out_features * in_features * param_element_size

    with torch.no_grad():
        dense_half = weight[dense_start:dense_end].detach().clone()
        sparse_dense = weight[sparse_start:sparse_end].detach().clone()
        keep = _compute_keep_mask_24(sparse_dense).to(
            device=param_device, dtype=torch.bool
        )
        if tuple(keep.shape) != (half, groups, 4):
            raise RuntimeError(
                f"SR24 gate_up split mask for {module_name} has {tuple(keep.shape)}, "
                f"expected {(half, groups, 4)}"
            )
        if not _keep_mask_is_24(keep):
            raise RuntimeError(f"SR24 gate_up split mask for {module_name} is not 2:4")
        sparse_view = sparse_dense[:, :usable_in].view(half, groups, 4)
        sparse_view.masked_fill_(~keep, 0)
        sparse_base = _to_sparse_semi_structured(sparse_dense)
        _set_sparse_weight_profile_metadata(sparse_base, module)

    logical_sparse_value_bytes = half * in_features // 2 * param_element_size
    sparse_value_bytes, sparse_meta_bytes = _semi_structured_storage_bytes(
        sparse_base,
        logical_value_bytes=logical_sparse_value_bytes,
    )
    dense_half_bytes = dense_half.numel() * dense_half.element_size()
    actual_weight_storage_bytes = dense_half_bytes + sparse_value_bytes + sparse_meta_bytes

    module._speclink_sr24_enabled = True
    module._speclink_sr24_backend = "torch_sparse"
    module._speclink_sr24_dense_fastpath = False
    module._speclink_sr24_gate_up_split = split_mode
    module._speclink_sr24_gate_up_dense_weight = dense_half
    module._speclink_sr24_gate_up_sparse_weight = sparse_base
    module._speclink_sr24_gate_up_dense_name = dense_name
    module._speclink_sr24_gate_up_sparse_name = sparse_name
    module._speclink_sr24_gate_up_half = half
    module._speclink_sr24_weight_shape = (out_features, in_features)
    module._speclink_sr24_weight_dtype = param_dtype
    module._speclink_sr24_usable_in = usable_in
    module._speclink_sr24_mask_method = method
    module._speclink_sr24_residual_backend = "none"
    module._speclink_sr24_residual_device = "none"
    module._speclink_sr24_no_residual = True
    _replace_module_weight_with_sparse_base(module, sparse_base)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "module": module_name,
        "leaf": _module_leaf(module_name),
        "layer": _layer_index(module_name),
        "shape": [out_features, in_features],
        "usable_in": usable_in,
        "base_mask_is_24": True,
        "residual_mask_is_24": True,
        "reconstructs_dense": False,
        "mask_method": method,
        "backend": "torch_sparse",
        "residual_backend": "none",
        "residual_device": "none",
        "gate_up_split": split_mode,
        "gate_up_split_dense": dense_name,
        "gate_up_split_sparse": sparse_name,
        "dense_weight_bytes": dense_bytes,
        "base_weight_storage_bytes": sparse_value_bytes + sparse_meta_bytes,
        "residual_value_storage_bytes": 0,
        "residual_extract_chunk_rows": residual_extract_chunk_rows(),
        "residual_bucket_size": residual_bucket_size(),
        "residual_bucket_scale_by_active": residual_bucket_scale_by_active(),
        "residual_bucket_priority": residual_bucket_priority(),
        "direct_position_bucket": direct_position_bucket_enabled(),
        "bonus_priority": bonus_priority(),
        "draft_position_priority_scale": draft_position_priority_scale(),
        "bucket_dense_copy": bucket_dense_copy(),
        "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
        "bucket_dense_compute_active_only": bucket_dense_compute_active_only(),
        "bucket_dense_active_mask_fused": bucket_dense_active_mask_fused(),
        "sort_bucket_rows": sort_bucket_rows(),
        "route_bucket_rows": route_bucket_rows(),
        "route_bucket_rows_graph_static_unsafe":
        route_bucket_rows_graph_static_unsafe(),
        "route_all_residual_rows": route_all_residual_rows(),
        "route_all_skip_bucket": route_all_skip_bucket(),
        "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
        "route_reuse_base_output": route_reuse_base_output(),
        "route_contiguous_fastpath": route_contiguous_fastpath(),
        "route_overlap_streams": route_overlap_streams(),
        "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
        "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
        "fixed_prefix_route_descriptor_only": fixed_prefix_route_descriptor_only(),
        "fixed_block_input_buffer": fixed_block_input_buffer_enabled(),
        "fixed_block_output_buffer": fixed_block_output_buffer_enabled(),
        "route_dense_fallback_fraction": route_dense_fallback_fraction(),
        "route_min_dense_rows": route_min_dense_rows(),
        "route_min_base_rows": route_min_base_rows(),
        "route_min_base_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
        ),
        "route_max_dense_fraction": route_max_dense_fraction(),
        "adaptive_dense_fallback": adaptive_dense_fallback_enabled(),
        "adaptive_dense_fallback_no_residual_only":
        adaptive_dense_fallback_no_residual_only(),
        "adaptive_dense_fallback_small_rows":
        adaptive_dense_fallback_small_rows(),
        "adaptive_dense_fallback_gate_up_fraction":
        adaptive_dense_fallback_gate_up_fraction(),
        "adaptive_dense_fallback_down_fraction":
        adaptive_dense_fallback_down_fraction(),
        "adaptive_dense_fallback_small_down_no_residual":
        adaptive_dense_fallback_small_down_no_residual(),
        "adaptive_dense_fallback_small_gate_up_no_residual":
        adaptive_dense_fallback_small_gate_up_no_residual(),
        "dense_fallback_nonuniform": dense_fallback_nonuniform(),
        "triton_route_assembly": triton_route_assembly(),
        "triton_bucket_override": triton_bucket_override(),
        "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
        "triton_bucket_scatter": triton_bucket_scatter(),
        "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
        "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
        "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
        "residual_extract_cpu_fallback_chunks": 0,
        "sparse_metadata_bytes": sparse_meta_bytes,
        "mask_metadata_bytes": 0,
        "actual_weight_storage_bytes": actual_weight_storage_bytes,
    }


def _select_gate_up_channel_rows(
    weight: torch.Tensor,
    *,
    half: int,
    dense_fraction: float,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_count = int(round(half * dense_fraction))
    dense_count = min(max(dense_count, 0), half)
    channel_ids = torch.arange(half, device=weight.device, dtype=torch.long)
    if dense_count == 0:
        dense_channels = channel_ids[:0]
    elif dense_count == half:
        dense_channels = channel_ids
    elif strategy == "front":
        dense_channels = channel_ids[:dense_count]
    elif strategy == "norm":
        gate_score = weight[:half].float().pow(2).mean(dim=1)
        up_score = weight[half:].float().pow(2).mean(dim=1)
        dense_channels = (
            gate_score + up_score
        ).topk(dense_count, largest=True, sorted=True).indices
    else:
        raise RuntimeError(f"unsupported gate_up channel strategy: {strategy}")
    dense_mask = torch.zeros(half, device=weight.device, dtype=torch.bool)
    dense_mask[dense_channels] = True
    sparse_channels = channel_ids[~dense_mask]
    dense_rows = torch.cat([dense_channels, dense_channels + half]).contiguous()
    sparse_rows = torch.cat([sparse_channels, sparse_channels + half]).contiguous()
    return (
        dense_channels.contiguous(),
        sparse_channels.contiguous(),
        dense_rows,
        sparse_rows,
    )


def _attach_gate_up_channel_pair_module(
    *,
    module_name: str,
    module: Any,
    weight: torch.Tensor,
    method: str,
) -> dict[str, Any]:
    _set_module_profile_metadata(module, module_name)
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    param_device = weight.device
    param_dtype = weight.dtype
    param_element_size = weight.element_size()
    if out_features % 2 != 0:
        raise RuntimeError(
            "SR24 gate_up channel-pair split requires even out_features for "
            f"{module_name}; got {out_features}"
        )
    if backend() != "torch_sparse":
        raise RuntimeError(
            "SR24 gate_up channel-pair split requires torch_sparse backend"
        )
    if usable_in != in_features:
        raise RuntimeError(
            "SR24 gate_up channel-pair split requires full in_features to be "
            f"2:4 groupable for {module_name}; got in_features={in_features}"
        )
    if param_dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            f"SR24 gate_up channel-pair split requires fp16/bf16 weights for "
            f"{module_name}; got {param_dtype}"
        )

    half = out_features // 2
    dense_fraction = gate_up_channel_dense_fraction()
    channel_strategy = gate_up_channel_strategy()
    dense_bytes = out_features * in_features * param_element_size
    with torch.no_grad():
        (
            dense_channels,
            sparse_channels,
            dense_rows,
            sparse_rows,
        ) = _select_gate_up_channel_rows(
            weight,
            half=half,
            dense_fraction=dense_fraction,
            strategy=channel_strategy,
        )
        if int(dense_channels.numel()) <= 0 or int(sparse_channels.numel()) <= 0:
            raise RuntimeError(
                "SR24 gate_up channel-pair split requires both dense and sparse "
                f"channels; dense_fraction={dense_fraction}"
            )
        dense_weight = weight.index_select(0, dense_rows).detach().clone()
        sparse_dense = weight.index_select(0, sparse_rows).detach().clone()
        keep = _compute_keep_mask_24(sparse_dense).to(
            device=param_device, dtype=torch.bool
        )
        expected_shape = (int(sparse_rows.numel()), groups, 4)
        if tuple(keep.shape) != expected_shape:
            raise RuntimeError(
                "SR24 gate_up channel-pair split mask for "
                f"{module_name} has {tuple(keep.shape)}, expected {expected_shape}"
            )
        if not _keep_mask_is_24(keep):
            raise RuntimeError(
                f"SR24 gate_up channel-pair split mask for {module_name} is not 2:4"
            )
        sparse_view = sparse_dense[:, :usable_in].view(
            int(sparse_rows.numel()), groups, 4
        )
        sparse_view.masked_fill_(~keep, 0)
        sparse_base = _to_sparse_semi_structured(sparse_dense)
        _set_sparse_weight_profile_metadata(sparse_base, module)

    logical_sparse_value_bytes = (
        int(sparse_rows.numel()) * in_features // 2 * param_element_size
    )
    sparse_value_bytes, sparse_meta_bytes = _semi_structured_storage_bytes(
        sparse_base,
        logical_value_bytes=logical_sparse_value_bytes,
    )
    dense_weight_bytes = dense_weight.numel() * dense_weight.element_size()
    actual_weight_storage_bytes = (
        dense_weight_bytes + sparse_value_bytes + sparse_meta_bytes
    )

    module._speclink_sr24_enabled = True
    module._speclink_sr24_backend = "torch_sparse"
    module._speclink_sr24_dense_fastpath = False
    module._speclink_sr24_gate_up_split = "channel_pair"
    module._speclink_sr24_gate_up_dense_weight = dense_weight
    module._speclink_sr24_gate_up_sparse_weight = sparse_base
    module._speclink_sr24_gate_up_dense_channels = dense_channels
    module._speclink_sr24_gate_up_sparse_channels = sparse_channels
    module._speclink_sr24_gate_up_grouped_channels = torch.cat(
        [dense_channels, sparse_channels]
    ).contiguous()
    module._speclink_sr24_gate_up_dense_channel_count = int(
        dense_channels.numel()
    )
    module._speclink_sr24_gate_up_sparse_channel_count = int(
        sparse_channels.numel()
    )
    module._speclink_sr24_gate_up_dense_fraction = dense_fraction
    module._speclink_sr24_gate_up_channel_strategy = channel_strategy
    module._speclink_sr24_gate_up_half = half
    module._speclink_sr24_weight_shape = (out_features, in_features)
    module._speclink_sr24_weight_dtype = param_dtype
    module._speclink_sr24_usable_in = usable_in
    module._speclink_sr24_mask_method = method
    module._speclink_sr24_residual_backend = "none"
    module._speclink_sr24_residual_device = "none"
    module._speclink_sr24_no_residual = True
    _replace_module_weight_with_sparse_base(module, sparse_base)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "module": module_name,
        "leaf": _module_leaf(module_name),
        "layer": _layer_index(module_name),
        "shape": [out_features, in_features],
        "usable_in": usable_in,
        "base_mask_is_24": True,
        "residual_mask_is_24": True,
        "reconstructs_dense": False,
        "mask_method": method,
        "backend": "torch_sparse",
        "residual_backend": "none",
        "residual_device": "none",
        "gate_up_split": "channel_pair",
        "gate_up_channel_dense_fraction": dense_fraction,
        "gate_up_channel_strategy": channel_strategy,
        "gate_up_dense_channels": int(dense_channels.numel()),
        "gate_up_sparse_channels": int(sparse_channels.numel()),
        "dense_weight_bytes": dense_bytes,
        "base_weight_storage_bytes": sparse_value_bytes + sparse_meta_bytes,
        "residual_value_storage_bytes": 0,
        "residual_extract_chunk_rows": residual_extract_chunk_rows(),
        "residual_bucket_size": residual_bucket_size(),
        "residual_bucket_scale_by_active": residual_bucket_scale_by_active(),
        "residual_bucket_priority": residual_bucket_priority(),
        "direct_position_bucket": direct_position_bucket_enabled(),
        "bonus_priority": bonus_priority(),
        "draft_position_priority_scale": draft_position_priority_scale(),
        "bucket_dense_copy": bucket_dense_copy(),
        "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
        "bucket_dense_compute_active_only": bucket_dense_compute_active_only(),
        "bucket_dense_active_mask_fused": bucket_dense_active_mask_fused(),
        "sort_bucket_rows": sort_bucket_rows(),
        "route_bucket_rows": route_bucket_rows(),
        "route_bucket_rows_graph_static_unsafe":
        route_bucket_rows_graph_static_unsafe(),
        "route_all_residual_rows": route_all_residual_rows(),
        "route_all_skip_bucket": route_all_skip_bucket(),
        "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
        "route_reuse_base_output": route_reuse_base_output(),
        "route_contiguous_fastpath": route_contiguous_fastpath(),
        "route_overlap_streams": route_overlap_streams(),
        "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
        "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
        "scheduler_policy_path": scheduler_policy_path(),
        "scheduler_policy_allow_legacy_mixed":
        scheduler_policy_allow_legacy_mixed(),
        "scheduler_policy_allow_single_block_packed_parallel":
        scheduler_policy_allow_single_block_packed_parallel(),
        "scheduler_policy_allow_serial_packed_parallel":
        scheduler_policy_allow_serial_packed_parallel(),
        "route_dense_fallback_fraction": route_dense_fallback_fraction(),
        "route_min_dense_rows": route_min_dense_rows(),
        "route_min_base_rows": route_min_base_rows(),
        "route_min_base_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
        ),
        "route_max_dense_fraction": route_max_dense_fraction(),
        "adaptive_dense_fallback": adaptive_dense_fallback_enabled(),
        "adaptive_dense_fallback_no_residual_only":
        adaptive_dense_fallback_no_residual_only(),
        "adaptive_dense_fallback_small_rows":
        adaptive_dense_fallback_small_rows(),
        "adaptive_dense_fallback_gate_up_fraction":
        adaptive_dense_fallback_gate_up_fraction(),
        "adaptive_dense_fallback_down_fraction":
        adaptive_dense_fallback_down_fraction(),
        "adaptive_dense_fallback_small_down_no_residual":
        adaptive_dense_fallback_small_down_no_residual(),
        "adaptive_dense_fallback_small_gate_up_no_residual":
        adaptive_dense_fallback_small_gate_up_no_residual(),
        "triton_route_assembly": triton_route_assembly(),
        "triton_bucket_override": triton_bucket_override(),
        "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
        "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
        "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
        "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
        "residual_extract_cpu_fallback_chunks": 0,
        "sparse_metadata_bytes": sparse_meta_bytes,
        "mask_metadata_bytes": 0,
        "actual_weight_storage_bytes": actual_weight_storage_bytes,
    }


def _is_cuda_device_label(value: Any) -> bool:
    return str(value or "").startswith("cuda")


def _finalize_attach_stats(stats: dict[str, Any]) -> None:
    dense_bytes = int(stats.get("dense_weight_bytes") or 0)
    stats["storage_over_dense"] = (
        int(stats.get("actual_weight_storage_bytes") or 0) / dense_bytes
        if dense_bytes
        else 0.0
    )

    residual_device_counts: dict[str, int] = {}
    residual_backend_counts: dict[str, int] = {}
    compressed_non_gpu_modules: list[str] = []
    residual_cpu_modules: list[str] = []
    residual_cuda_modules: list[str] = []
    extract_fallback_modules: list[str] = []
    for row in stats.get("per_module") or []:
        if not isinstance(row, dict):
            continue
        module_name = str(row.get("module") or "")
        residual_backend_value = str(row.get("residual_backend") or "unknown")
        residual_device_value = str(row.get("residual_device") or "unknown")
        residual_backend_counts[residual_backend_value] = (
            residual_backend_counts.get(residual_backend_value, 0) + 1
        )
        residual_device_counts[residual_device_value] = (
            residual_device_counts.get(residual_device_value, 0) + 1
        )
        if residual_device_value.startswith("cpu"):
            residual_cpu_modules.append(module_name)
        elif _is_cuda_device_label(residual_device_value):
            residual_cuda_modules.append(module_name)
        if int(row.get("residual_extract_cpu_fallback_chunks") or 0) > 0:
            extract_fallback_modules.append(module_name)
        if (
            residual_backend_value == "compressed_dense"
            and not _is_cuda_device_label(residual_device_value)
        ):
            compressed_non_gpu_modules.append(module_name)

    stats["residual_backend_counts"] = residual_backend_counts
    stats["residual_device_counts"] = residual_device_counts
    stats["residual_cpu_module_count"] = len(residual_cpu_modules)
    stats["residual_cuda_module_count"] = len(residual_cuda_modules)
    stats["residual_extract_cpu_fallback_module_count"] = len(
        extract_fallback_modules
    )
    stats["residual_cpu_modules"] = residual_cpu_modules
    stats["residual_extract_cpu_fallback_modules"] = extract_fallback_modules
    stats["compressed_residual_runtime_on_gpu"] = not compressed_non_gpu_modules
    stats["compressed_residual_non_gpu_modules"] = compressed_non_gpu_modules
    if require_gpu_residual() and compressed_non_gpu_modules:
        preview = ", ".join(compressed_non_gpu_modules[:8])
        if len(compressed_non_gpu_modules) > 8:
            preview += ", ..."
        raise RuntimeError(
            "SPECLINK_SR24_REQUIRE_GPU_RESIDUAL=1 but compressed_dense "
            f"residual values are not GPU-resident for {len(compressed_non_gpu_modules)} "
            f"module(s): {preview}"
        )
    if require_gpu_residual() and extract_fallback_modules:
        preview = ", ".join(extract_fallback_modules[:8])
        if len(extract_fallback_modules) > 8:
            preview += ", ..."
        raise RuntimeError(
            "SPECLINK_SR24_REQUIRE_GPU_RESIDUAL=1 but residual extraction "
            "used CPU fallback chunks for "
            f"{len(extract_fallback_modules)} module(s): {preview}"
        )


def _prewarm_bucket_complement_kernel(stats: dict[str, Any]) -> None:
    if not (
        row_routed_mlp()
        and residual_bucket_size() > 0
        and torch.cuda.is_available()
    ):
        stats["bucket_complement_triton_prewarmed"] = False
        return
    bucket_count = residual_bucket_size()
    rows = max(bucket_count + 256, 1024)
    try:
        device = torch.device("cuda")
        bucket_rows = torch.arange(bucket_count, dtype=torch.long, device=device)
        _triton_bucket_complement_rows(
            rows=rows,
            device=device,
            bucket_rows=bucket_rows,
            static_output=False,
        )
        torch.cuda.synchronize(device)
        stats["bucket_complement_triton_prewarmed"] = True
    except Exception as exc:
        stats["bucket_complement_triton_prewarmed"] = False
        stats["bucket_complement_triton_prewarm_error"] = str(exc)


def apply_sr24_from_env(
    model: Any,
    *,
    logger: Any | None = None,
    context: str = "target_model",
) -> dict[str, Any] | None:
    if not enabled():
        return None
    sr_mode = mode()
    if sr_mode not in {"base_only", "all_corrected", "selective"}:
        raise RuntimeError(f"unsupported SPECLINK_SR24_MODE={sr_mode}")
    sr_backend = backend()
    if sr_backend not in SR24_BACKENDS:
        raise RuntimeError(f"unsupported SPECLINK_SR24_BACKEND={sr_backend}")
    sr_residual_backend = residual_backend()
    if sr_residual_backend not in {"compressed_dense", "torch_sparse", "dense_rows"}:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_RESIDUAL_BACKEND="
            f"{sr_residual_backend}"
        )
    sr_residual_device = residual_device()
    if sr_residual_device not in {"cpu", "cuda"}:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_RESIDUAL_DEVICE="
            f"{sr_residual_device}"
        )
    sr_selective_policy = selective_residual_policy()
    if sr_selective_policy not in {
        "critical_prefix",
        "all_if_any_low",
        "batch_all_if_any_low",
        "low_confidence",
        "high_confidence",
        "prefix_confidence",
        "fixed_prefix",
    }:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY="
            f"{sr_selective_policy}"
        )
    sr_non_draft_policy = selective_non_draft_policy()
    sr_bonus_priority = bonus_priority()
    if sr_non_draft_policy not in {
        "all",
        "none",
        "bonus",
        "predicted_full_accept",
    }:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY="
            f"{sr_non_draft_policy}"
        )
    sr_all_corrected_dense_fastpath = (
        sr_backend == "torch_sparse"
        and sr_mode == "all_corrected"
        and all_corrected_dense_fastpath()
    )
    sr_gate_up_split = gate_up_split()
    sr_static_all_residual_dense_fastpath = (
        sr_backend == "torch_sparse"
        and sr_mode == "selective"
        and static_mask_state() == "all_residual"
        and static_all_residual_dense_fastpath()
        and all_corrected_dense_fastpath()
    )
    if _env_flag("SPECLINK_STRUCTURED_24_ENABLE") or _env_flag(
        "SPECLINK_TOKEN_DENSE_ENABLE"
    ):
        raise RuntimeError(
            "SPECLINK_SR24_ENABLE cannot be combined with "
            "SPECLINK_STRUCTURED_24_ENABLE or SPECLINK_TOKEN_DENSE_ENABLE"
        )

    modules = _iter_target_modules(model)
    target_leafs = sorted(_target_leafs())
    residual_target_leafs_set = _residual_target_leafs(set(target_leafs))
    residual_target_leafs = sorted(residual_target_leafs_set)
    base_only_layer_ids = _parse_layer_ids_env(
        "SPECLINK_SR24_BASE_ONLY_LAYER_IDS"
    )
    base_only_layer_ids_by_leaf = _parse_layer_ids_by_leaf_env(
        "SPECLINK_SR24_BASE_ONLY_LAYER_IDS_BY_LEAF"
    )
    base_only_dense_verify_layer_ids = _parse_layer_ids_env(
        "SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_LAYER_IDS"
    )
    base_only_dense_verify_layer_ids_by_leaf = _parse_layer_ids_by_leaf_env(
        "SPECLINK_SR24_BASE_ONLY_DENSE_VERIFY_LAYER_IDS_BY_LEAF"
    )
    residual_layer_ids_by_leaf = _parse_layer_ids_by_leaf_env(
        "SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"
    )
    if sr_mode == "all_corrected":
        # all_corrected is the dense-equivalent control for every attached
        # SR24 module. Base-only filters are ignored, but residual layer
        # filters may narrow the attach scope so memory-heavy compressed
        # residual ablations can be run without rewriting every target layer.
        # Filtered-out modules stay dense; they are not converted to base-only.
        residual_target_leafs_set = set(target_leafs)
        residual_target_leafs = sorted(residual_target_leafs_set)
        base_only_layer_ids = None
        base_only_layer_ids_by_leaf = {}
    mask_path = os.getenv("SPECLINK_SR24_MASK_PATH", "").strip()
    mask_cache = None
    stats: dict[str, Any] = {
        "enabled": True,
        "context": context,
        "mode": sr_mode,
        "backend": sr_backend,
        "target_leafs": target_leafs,
        "residual_target_leafs": residual_target_leafs,
        "base_only_layer_ids": (
            sorted(base_only_layer_ids) if base_only_layer_ids is not None else []
        ),
        "base_only_layer_ids_by_leaf": {
            leaf: sorted(layer_ids)
            for leaf, layer_ids in sorted(base_only_layer_ids_by_leaf.items())
        },
        "residual_layer_ids_by_leaf": {
            leaf: sorted(layer_ids)
            for leaf, layer_ids in sorted(residual_layer_ids_by_leaf.items())
        },
        "runtime_base_only_scope":
        _runtime_base_only_scope_raw(),
        "runtime_base_only_layer_ids_by_leaf": {
            leaf: sorted(layer_ids)
            for leaf, layer_ids in sorted(
                _runtime_base_only_layer_ids_by_leaf().items()
            )
        },
        "residual_backend": (
            "dense_fastpath"
            if sr_all_corrected_dense_fastpath
            or sr_static_all_residual_dense_fastpath
            else sr_residual_backend
        ),
        "residual_backend_by_leaf": residual_backend_by_leaf(),
        "residual_device": (
            "none"
            if sr_all_corrected_dense_fastpath
            or sr_static_all_residual_dense_fastpath
            else sr_residual_device
        ),
        "direct_cslt_linear": direct_cslt_linear(),
        "cslt_small_m_alg_id_enabled": cslt_small_m_alg_id_enabled(),
        "cslt_small_m_threshold": cslt_small_m_threshold(),
        "cslt_small_m_alg_id": cslt_small_m_alg_id(),
        "cslt_small_m_threshold_by_leaf": cslt_small_m_threshold_by_leaf(),
        "cslt_small_m_alg_id_by_leaf": cslt_small_m_alg_id_by_leaf(),
        "threshold": threshold(),
        "all_corrected_dense_fastpath": all_corrected_dense_fastpath(),
        "full_residual_early_dense": full_residual_early_dense(),
        "noverify_dense_mlp_fastpath": noverify_dense_mlp_fastpath(),
        "residual_out_chunk": residual_out_chunk(),
        "cache_compressed_residual_weight": cache_compressed_residual_weight(),
        "prewarm_compressed_residual_weight":
        prewarm_compressed_residual_weight(),
        "compressed_residual_triton": compressed_residual_triton(),
        "compressed_residual_block_m": compressed_residual_block_m(),
        "compressed_residual_block_n": compressed_residual_block_n(),
        "compressed_residual_block_g": compressed_residual_block_g(),
        "require_gpu_residual": require_gpu_residual(),
        "residual_extract_chunk_rows": residual_extract_chunk_rows(),
        "residual_bucket_size": residual_bucket_size(),
        "residual_bucket_scale_by_active": residual_bucket_scale_by_active(),
        "cudagraph_bucket": cudagraph_bucket_enabled(),
        "cudagraph_bucket_active_hint": cudagraph_bucket_active_hint(),
        "residual_bucket_priority": residual_bucket_priority(),
        "direct_position_bucket": direct_position_bucket_enabled(),
        "bonus_priority": sr_bonus_priority,
        "draft_position_priority_scale": draft_position_priority_scale(),
        "bucket_dense_copy": bucket_dense_copy(),
        "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
        "bucket_dense_compute_active_only":
        bucket_dense_compute_active_only(),
        "bucket_dense_active_mask_fused": bucket_dense_active_mask_fused(),
        "sort_bucket_rows": sort_bucket_rows(),
        "route_bucket_rows": route_bucket_rows(),
        "route_bucket_rows_graph_static_unsafe":
        route_bucket_rows_graph_static_unsafe(),
        "route_all_residual_rows": route_all_residual_rows(),
        "route_all_skip_bucket": route_all_skip_bucket(),
        "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
        "route_reuse_base_output": route_reuse_base_output(),
        "route_contiguous_fastpath": route_contiguous_fastpath(),
        "route_overlap_streams": route_overlap_streams(),
        "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
        "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
        "route_dense_fallback_fraction": route_dense_fallback_fraction(),
        "route_min_dense_rows": route_min_dense_rows(),
        "route_min_base_rows": route_min_base_rows(),
        "route_min_base_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
        ),
        "route_max_dense_fraction": route_max_dense_fraction(),
        "adaptive_dense_fallback": adaptive_dense_fallback_enabled(),
        "adaptive_dense_fallback_no_residual_only":
        adaptive_dense_fallback_no_residual_only(),
        "adaptive_dense_fallback_small_rows":
        adaptive_dense_fallback_small_rows(),
        "adaptive_dense_fallback_gate_up_fraction":
        adaptive_dense_fallback_gate_up_fraction(),
        "adaptive_dense_fallback_down_fraction":
        adaptive_dense_fallback_down_fraction(),
        "adaptive_dense_fallback_small_down_no_residual":
        adaptive_dense_fallback_small_down_no_residual(),
        "adaptive_dense_fallback_small_gate_up_no_residual":
        adaptive_dense_fallback_small_gate_up_no_residual(),
        "triton_route_assembly": triton_route_assembly(),
        "triton_bucket_override": triton_bucket_override(),
        "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
        "triton_bucket_scatter": triton_bucket_scatter(),
        "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
        "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
        "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
        "selective_correct_non_draft": selective_correct_non_draft(),
        "selective_residual_policy": sr_selective_policy,
        "selective_non_draft_policy": sr_non_draft_policy,
        "selective_dense_nonverify_scope":
        _selective_dense_nonverify_scope_raw(),
        "selective_dense_nonverify_max_rows":
        selective_dense_nonverify_max_rows(),
        "selective_dense_nonverify_static_rows":
        selective_dense_nonverify_static_rows(),
        "selective_dense_nonverify_layer_ids_by_leaf": {
            leaf: sorted(layer_ids)
            for leaf, layer_ids in sorted(
                _selective_dense_nonverify_layer_ids_by_leaf().items()
            )
        },
        "selective_prefix_threshold": selective_prefix_threshold(),
        "selective_extra_after_low": selective_extra_after_low(),
        "selective_min_prefix_residual": selective_min_prefix_residual(),
        "selective_max_residual_draft_rows":
        selective_max_residual_draft_rows(),
        "low_confidence_cap_by_risk": low_confidence_cap_by_risk(),
        "early_dense_tokens": early_dense_tokens(),
        "sync_mask_state": sync_mask_state(),
        "static_mask_state": static_mask_state(),
        "static_all_residual_dense_fastpath":
        static_all_residual_dense_fastpath(),
        "linear_hooks_enabled": linear_hooks_enabled(),
        "draft_scores_enabled": draft_scores_enabled(),
        "runtime_stats_enabled": runtime_stats_enabled(),
        "runtime_timing_enabled": runtime_timing_enabled(),
        "direct_cslt_linear": direct_cslt_linear(),
        "gate_up_split": sr_gate_up_split,
        "gate_up_channel_dense_fraction": gate_up_channel_dense_fraction(),
        "gate_up_channel_strategy": gate_up_channel_strategy(),
        "gate_up_channel_fused_act": gate_up_channel_fused_act(),
        "row_routed_mlp": row_routed_mlp(),
        "row_routed_down_linear": row_routed_down_linear(),
        "row_routed_down_fixed_block": row_routed_down_fixed_block_enabled(),
        "row_routed_mlp_reuse_base_output": row_routed_mlp_reuse_base_output(),
        "row_routed_mlp_fixed_block_dense_fill":
        row_routed_mlp_fixed_block_dense_fill(),
        "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
        "fixed_prefix_route_descriptor_only":
        fixed_prefix_route_descriptor_only(),
        "fixed_block_input_buffer": fixed_block_input_buffer_enabled(),
        "fixed_block_output_buffer": fixed_block_output_buffer_enabled(),
        "scheduler_policy_path": scheduler_policy_path(),
        "scheduler_policy_allow_legacy_mixed":
        scheduler_policy_allow_legacy_mixed(),
        "scheduler_policy_allow_single_block_packed_parallel":
        scheduler_policy_allow_single_block_packed_parallel(),
        "scheduler_policy_allow_serial_packed_parallel":
        scheduler_policy_allow_serial_packed_parallel(),
        "scheduler_policy_near_full_tolerance":
        scheduler_policy_near_full_tolerance(),
        "fixed_block_capacity_padding":
        fixed_block_capacity_padding_enabled(),
        "fixed_block_capacity_zero_dummy":
        fixed_block_capacity_zero_dummy_enabled(),
        "scheduler_policy_dense_bypass": scheduler_policy_dense_bypass(),
        "grouped_queue_shadow": grouped_queue_shadow_enabled(),
        "grouped_queue_max_wait_blocks":
        grouped_queue_shadow_max_wait_blocks(),
        "grouping_trace_enabled": grouping_trace_enabled(),
        "grouping_trace_path": (
            str(grouping_trace_path()) if grouping_trace_path() is not None else ""
        ),
        "row_routed_mlp_min_dense_rows": row_routed_mlp_min_dense_rows(),
        "row_routed_mlp_min_dense_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS_BY_LEAF"
        ),
        "row_routed_mlp_max_dense_rows": row_routed_mlp_max_dense_rows(),
        "row_routed_mlp_max_dense_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROW_ROUTED_MLP_MAX_DENSE_ROWS_BY_LEAF"
        ),
        "row_routed_mlp_max_base_rows": row_routed_mlp_max_base_rows(),
        "row_routed_mlp_max_base_rows_by_leaf": _leaf_int_map_from_env(
            "SPECLINK_SR24_ROW_ROUTED_MLP_MAX_BASE_ROWS_BY_LEAF"
        ),
        "force_cudagraph_none_for_mixed":
        force_cudagraph_none_for_mixed_enabled(),
        "base_only_dense_nonverify": base_only_dense_nonverify(),
        "base_only_dense_verify_max_rows": base_only_dense_verify_max_rows(),
        "base_only_dense_verify_layer_ids": (
            sorted(base_only_dense_verify_layer_ids)
            if base_only_dense_verify_layer_ids is not None
            else []
        ),
        "base_only_dense_verify_layer_ids_by_leaf": {
            leaf: sorted(layer_ids)
            for leaf, layer_ids in sorted(
                base_only_dense_verify_layer_ids_by_leaf.items()
            )
        },
        "static_mask_buffer": static_mask_buffer_enabled(),
        "static_mask_buffer_capacity": static_mask_buffer_capacity(),
        "batched_mask_builder": batched_mask_builder_enabled(),
        "batched_uniform_direct": batched_uniform_direct_enabled(),
        "gpu_count_mask_builder": gpu_count_mask_builder_enabled(),
        "mask_path": str(Path(mask_path).resolve()) if mask_path else "",
        "mask_cache_method": "",
        "dense_fastpath_noop": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "module_count_seen": len(modules),
        "module_count_attached": 0,
        "dense_weight_bytes": 0,
        "actual_weight_storage_bytes": 0,
        "base_weight_storage_bytes": 0,
        "residual_value_storage_bytes": 0,
        "residual_extract_cpu_fallback_chunks": 0,
        "residual_extract_cpu_fallback_module_count": 0,
        "residual_backend_counts": {},
        "residual_device_counts": {},
        "residual_cpu_module_count": 0,
        "residual_cuda_module_count": 0,
        "residual_cpu_modules": [],
        "residual_extract_cpu_fallback_modules": [],
        "compressed_residual_runtime_on_gpu": True,
        "compressed_residual_non_gpu_modules": [],
        "sparse_metadata_bytes": 0,
        "mask_metadata_bytes": 0,
        "storage_over_dense": 0.0,
        "missing_cached_mask_modules": [],
        "per_module": [],
        "dense_zero_backend_note": (
            "dense_zero/prototype backend: module.weight is dense-shaped base 2:4; "
            "residual is a compressed value stream and is materialized for "
            "masked dense matmul at runtime."
        ),
        "sparse_backend_note": (
            "torch_sparse backend: module dense weight Parameter is replaced "
            "with PyTorch SparseSemiStructuredTensorCUSPARSELT base weight; "
            "residual is either a compressed value stream materialized only "
            "for corrected rows or, with SPECLINK_SR24_RESIDUAL_BACKEND="
            "torch_sparse, a second semi-structured sparse tensor."
        ),
    }
    full_all_corrected_dense_fastpath = (
        sr_all_corrected_dense_fastpath
        and set(residual_target_leafs) == set(target_leafs)
        and base_only_layer_ids is None
        and not base_only_layer_ids_by_leaf
        and not residual_layer_ids_by_leaf
        and sr_gate_up_split == "none"
    )
    if full_all_corrected_dense_fastpath:
        per_module: list[dict[str, Any]] = []
        for name, _module, weight in modules:
            out_features = int(weight.shape[0])
            in_features = int(weight.shape[1])
            usable_in = (in_features // 4) * 4
            dense_bytes = out_features * in_features * int(weight.element_size())
            row = {
                "module": name,
                "leaf": _module_leaf(name),
                "layer": _layer_index(name),
                "shape": [out_features, in_features],
                "usable_in": usable_in,
                "base_mask_is_24": True,
                "residual_mask_is_24": True,
                "reconstructs_dense": True,
                "all_corrected_dense_fastpath": True,
                "mask_method": "dense_fastpath_noop",
                "backend": sr_backend,
                "residual_backend": "dense_fastpath",
                "residual_device": "none",
                "dense_weight_bytes": dense_bytes,
                "base_weight_storage_bytes": dense_bytes,
                "residual_value_storage_bytes": 0,
                "residual_extract_chunk_rows": residual_extract_chunk_rows(),
                "residual_bucket_size": residual_bucket_size(),
                "residual_bucket_scale_by_active":
                residual_bucket_scale_by_active(),
                "residual_bucket_priority": residual_bucket_priority(),
                "direct_position_bucket": direct_position_bucket_enabled(),
                "bucket_dense_copy": bucket_dense_copy(),
                "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
                "bucket_dense_compute_active_only":
                bucket_dense_compute_active_only(),
                "bucket_dense_active_mask_fused":
                bucket_dense_active_mask_fused(),
                "sort_bucket_rows": sort_bucket_rows(),
                "route_bucket_rows": route_bucket_rows(),
                "route_bucket_rows_graph_static_unsafe":
                route_bucket_rows_graph_static_unsafe(),
                "route_all_residual_rows": route_all_residual_rows(),
                "route_all_skip_bucket": route_all_skip_bucket(),
                "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
                "route_reuse_base_output": route_reuse_base_output(),
                "route_contiguous_fastpath": route_contiguous_fastpath(),
                "route_overlap_streams": route_overlap_streams(),
                "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
                "fixed_prefix_route_fastpath": fixed_prefix_route_fastpath_enabled(),
                "route_dense_fallback_fraction": route_dense_fallback_fraction(),
                "route_min_dense_rows": route_min_dense_rows(),
                "route_min_base_rows": route_min_base_rows(),
                "route_min_base_rows_by_leaf": _leaf_int_map_from_env(
                    "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
                ),
                "route_max_dense_fraction": route_max_dense_fraction(),
                "triton_route_assembly": triton_route_assembly(),
                "triton_bucket_override": triton_bucket_override(),
                "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
                "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
                "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
                "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
                "residual_extract_cpu_fallback_chunks": 0,
                "sparse_metadata_bytes": 0,
                "mask_metadata_bytes": 0,
                "actual_weight_storage_bytes": dense_bytes,
            }
            per_module.append(row)
            stats["module_count_attached"] += 1
            for key in (
                "dense_weight_bytes",
                "actual_weight_storage_bytes",
                "base_weight_storage_bytes",
                "residual_value_storage_bytes",
                "residual_extract_cpu_fallback_chunks",
                "sparse_metadata_bytes",
                "mask_metadata_bytes",
            ):
                stats[key] += int(row[key])
        stats["per_module"] = per_module
        stats["dense_fastpath_noop"] = True
        _finalize_attach_stats(stats)
        stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        stats_path = os.getenv("SPECLINK_SR24_STATS_PATH", "").strip()
        if stats_path:
            _write_json(Path(stats_path), stats)
        _write_log({
            "timestamp": time.time(),
            "event": "sr24_model_attached",
            **stats,
        })
        if logger is not None:
            logger.info(
                "Applied SpecLink SR24 dense no-op: mode=%s modules=%d "
                "backend=%s residual_backend=%s storage_over_dense=%.4f",
                sr_mode,
                stats["module_count_attached"],
                stats["backend"],
                stats["residual_backend"],
                stats["storage_over_dense"],
            )
        return stats

    mask_cache = _load_mask_cache(mask_path) if mask_path else None
    stats["mask_cache_method"] = (
        (mask_cache.get("metadata", {}) or {}).get("method", "")
        if mask_cache
        else ""
    )
    for name, module, weight in modules:
        module_leaf = _module_leaf(name)
        force_no_residual = module_leaf not in residual_target_leafs_set
        layer_index = _layer_index(name)
        force_dense_fastpath = False
        if force_no_residual:
            leaf_layer_ids = base_only_layer_ids_by_leaf.get(module_leaf)
            allowed_layer_ids = (
                leaf_layer_ids
                if leaf_layer_ids is not None
                else base_only_layer_ids
            )
            if allowed_layer_ids is not None and layer_index not in allowed_layer_ids:
                continue
            if allowed_layer_ids is None and base_only_layer_ids_by_leaf:
                continue
        elif residual_layer_ids_by_leaf:
            leaf_layer_ids = residual_layer_ids_by_leaf.get(module_leaf)
            if leaf_layer_ids is not None:
                if layer_index not in leaf_layer_ids:
                    leaf_base_only_layer_ids = base_only_layer_ids_by_leaf.get(
                        module_leaf
                    )
                    allowed_base_only_layer_ids = (
                        leaf_base_only_layer_ids
                        if leaf_base_only_layer_ids is not None
                        else base_only_layer_ids
                    )
                    if (
                        allowed_base_only_layer_ids is not None
                        and layer_index in allowed_base_only_layer_ids
                    ):
                        force_no_residual = True
                    else:
                        continue
            else:
                if sr_mode == "all_corrected":
                    continue
                force_dense_fastpath = (
                    sr_mode == "selective"
                    and sr_backend == "torch_sparse"
                    and static_all_residual_dense_fastpath()
                    and all_corrected_dense_fastpath()
                )
        if sr_static_all_residual_dense_fastpath and not force_no_residual:
            # Diagnostic dense no-op path for filtered selective runs.  Keep
            # the original vLLM Linear module untouched so this mode can be
            # used as a strict dense-equivalence sanity check.
            out_features = int(weight.shape[0])
            in_features = int(weight.shape[1])
            usable_in = (in_features // 4) * 4
            dense_bytes = out_features * in_features * int(weight.element_size())
            row = {
                "module": name,
                "leaf": module_leaf,
                "layer": layer_index,
                "shape": [out_features, in_features],
                "usable_in": usable_in,
                "base_mask_is_24": True,
                "residual_mask_is_24": True,
                "reconstructs_dense": True,
                "all_corrected_dense_fastpath": True,
                "mask_method": "static_all_residual_dense_fastpath_noop",
                "backend": sr_backend,
                "residual_backend": "dense_fastpath",
                "residual_device": "none",
                "dense_weight_bytes": dense_bytes,
                "base_weight_storage_bytes": dense_bytes,
                "residual_value_storage_bytes": 0,
                "residual_extract_chunk_rows": residual_extract_chunk_rows(),
                "residual_bucket_size": residual_bucket_size(),
                "residual_bucket_scale_by_active":
                residual_bucket_scale_by_active(),
                "residual_bucket_priority": residual_bucket_priority(),
                "direct_position_bucket": direct_position_bucket_enabled(),
                "bucket_dense_copy": bucket_dense_copy(),
                "bucket_dense_copy_active_only": bucket_dense_copy_active_only(),
                "bucket_dense_compute_active_only":
                bucket_dense_compute_active_only(),
                "bucket_dense_active_mask_fused":
                bucket_dense_active_mask_fused(),
                "sort_bucket_rows": sort_bucket_rows(),
                "route_bucket_rows": route_bucket_rows(),
                "route_bucket_rows_graph_static_unsafe":
                route_bucket_rows_graph_static_unsafe(),
                "route_all_residual_rows": route_all_residual_rows(),
                "route_all_skip_bucket": route_all_skip_bucket(),
                "direct_cpu_route_rows": direct_cpu_route_rows_enabled(),
                "route_reuse_base_output": route_reuse_base_output(),
                "route_overlap_streams": route_overlap_streams(),
                "route_overlap_allow_cudagraph": route_overlap_allow_cudagraph(),
                "route_dense_fallback_fraction": route_dense_fallback_fraction(),
                "route_min_dense_rows": route_min_dense_rows(),
                "route_min_base_rows": route_min_base_rows(),
                "route_min_base_rows_by_leaf": _leaf_int_map_from_env(
                    "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
                ),
                "route_max_dense_fraction": route_max_dense_fraction(),
                "triton_route_assembly": triton_route_assembly(),
                "triton_bucket_override": triton_bucket_override(),
                "triton_bucket_dense_gemm": triton_bucket_dense_gemm(),
                "triton_bucket_dense_block_m": triton_bucket_dense_block_m(),
                "triton_bucket_dense_block_n": triton_bucket_dense_block_n(),
                "triton_bucket_dense_block_k": triton_bucket_dense_block_k(),
                "residual_extract_cpu_fallback_chunks": 0,
                "sparse_metadata_bytes": 0,
                "mask_metadata_bytes": 0,
                "actual_weight_storage_bytes": dense_bytes,
            }
            stats["per_module"].append(row)
            stats["module_count_attached"] += 1
            stats["dense_fastpath_noop"] = True
            for key in (
                "dense_weight_bytes",
                "actual_weight_storage_bytes",
                "base_weight_storage_bytes",
                "residual_value_storage_bytes",
                "residual_extract_cpu_fallback_chunks",
                "sparse_metadata_bytes",
                "mask_metadata_bytes",
            ):
                stats[key] += int(row[key])
            continue
        if module_leaf == "gate_up_proj" and sr_gate_up_split != "none":
            if not force_no_residual:
                raise RuntimeError(
                    "SPECLINK_SR24_GATE_UP_SPLIT is only supported for "
                    "base-only gate_up_proj targets. Remove gate_up_proj from "
                    "SPECLINK_SR24_RESIDUAL_TARGET_LEAFS for this ablation."
                )
            if sr_gate_up_split == "channel_pair":
                row = _attach_gate_up_channel_pair_module(
                    module_name=name,
                    module=module,
                    weight=weight,
                    method="magnitude_channel_pair",
                )
            else:
                row = _attach_gate_up_split_module(
                    module_name=name,
                    module=module,
                    weight=weight,
                    split_mode=sr_gate_up_split,
                    method="magnitude_split",
                )
            stats["per_module"].append(row)
            stats["module_count_attached"] += 1
            for key in (
                "dense_weight_bytes",
                "actual_weight_storage_bytes",
                "base_weight_storage_bytes",
                "residual_value_storage_bytes",
                "residual_extract_cpu_fallback_chunks",
                "sparse_metadata_bytes",
                "mask_metadata_bytes",
            ):
                stats[key] += int(row[key])
            continue
        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])
        usable_in = (in_features // 4) * 4
        groups = usable_in // 4
        method = "magnitude"
        if mask_cache is not None:
            cached = _cache_mask_for_module(name, mask_cache)
            if cached is None:
                stats["missing_cached_mask_modules"].append(name)
                keep = _compute_keep_mask_24(weight)
                method = "magnitude_missing_cache_fallback"
            else:
                group_bytes = _expand_mask_bytes(
                    cached,
                    out_features=out_features,
                    groups=groups,
                    device=torch.device("cpu"),
                )
                if not _mask_is_24(group_bytes):
                    raise RuntimeError(f"cached SR24 mask for {name} is not 2:4")
                keep = _unpacked_group_bytes_to_keep(
                    group_bytes, device=torch.device("cpu")
                )
                method = "cached_mask"
        else:
            keep = _compute_keep_mask_24(weight)
        row = _attach_sr24_module(
            module_name=name,
            module=module,
            weight=weight,
            keep=keep,
            method=method,
            force_no_residual=force_no_residual,
            force_dense_fastpath=force_dense_fastpath,
            base_only_dense_verify_layer_ids=base_only_dense_verify_layer_ids,
            base_only_dense_verify_layer_ids_by_leaf=(
                base_only_dense_verify_layer_ids_by_leaf
            ),
        )
        stats["per_module"].append(row)
        stats["module_count_attached"] += 1
        for key in (
            "dense_weight_bytes",
            "actual_weight_storage_bytes",
            "base_weight_storage_bytes",
            "residual_value_storage_bytes",
            "residual_extract_cpu_fallback_chunks",
            "sparse_metadata_bytes",
            "mask_metadata_bytes",
        ):
            stats[key] += int(row[key])
    _finalize_attach_stats(stats)
    _prewarm_bucket_complement_kernel(stats)
    stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    stats_path = os.getenv("SPECLINK_SR24_STATS_PATH", "").strip()
    if stats_path:
        _write_json(Path(stats_path), stats)
    _write_log({"timestamp": time.time(), "event": "sr24_model_attached", **stats})
    if logger is not None:
        if stats.get("dense_fastpath_noop"):
            logger.info(
                "Applied SpecLink SR24 dense no-op: mode=%s modules=%d "
                "backend=%s residual_backend=%s storage_over_dense=%.4f",
                sr_mode,
                stats["module_count_attached"],
                stats["backend"],
                stats["residual_backend"],
                stats["storage_over_dense"],
            )
        else:
            logger.info(
                "Applied SpecLink SR24: mode=%s modules=%d backend=%s "
                "residual_backend=%s storage_over_dense=%.4f",
                sr_mode,
                stats["module_count_attached"],
                stats["backend"],
                stats["residual_backend"],
                stats["storage_over_dense"],
            )
    return stats


def begin_propose_context(
    *,
    req_ids: list[str],
    prompt_lens: list[int],
    generated_lens: list[int],
    active_requests: int,
    batch_size: int,
    num_spec_tokens: int,
    method: str = "",
) -> Any:
    if not draft_scores_enabled():
        return None
    return _propose_context.set(
        {
            "req_ids": req_ids,
            "prompt_lens": prompt_lens,
            "generated_lens": generated_lens,
            "active_requests": active_requests,
            "batch_size": batch_size,
            "num_spec_tokens": num_spec_tokens,
            "method": method or "unknown",
        }
    )


def end_propose_context(token: Any) -> None:
    if token is not None:
        _propose_context.reset(token)


@torch.inference_mode()
def record_draft_scores(
    *,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    temperature: torch.Tensor | None = None,
    method: str = "",
) -> None:
    if not draft_scores_enabled():
        return
    ctx = _propose_context.get()
    if ctx is None or not logits_by_position:
        return
    batch_size = min(int(draft_token_ids.shape[0]), len(ctx["req_ids"]))
    num_spec_tokens = min(int(draft_token_ids.shape[1]), len(logits_by_position))
    if batch_size <= 0 or num_spec_tokens <= 0:
        return
    draft_token_ids = draft_token_ids[:batch_size, :num_spec_tokens]
    score_device = logits_by_position[0].device
    per_req_scores = torch.full(
        (batch_size, num_spec_tokens),
        float("nan"),
        device=score_device,
        dtype=torch.float32,
    )
    for pos, logits in enumerate(logits_by_position[:num_spec_tokens]):
        logits = logits[:batch_size].detach().float()
        selected = draft_token_ids[:batch_size, pos].to(
            device=logits.device, dtype=torch.long
        )
        valid = (selected >= 0) & (selected < logits.shape[-1])
        safe_selected = selected.clamp(min=0, max=logits.shape[-1] - 1)
        selected_logit = logits.gather(
            1,
            safe_selected.view(-1, 1),
        ).squeeze(1)
        selected_logprob = selected_logit - torch.logsumexp(logits, dim=-1)
        selected_score = selected_logprob.exp()
        per_req_scores[:, pos] = torch.where(
            valid,
            selected_score,
            torch.full_like(selected_score, float("nan")),
        )
    req_ids = ctx["req_ids"]
    needs_generated_lens = early_dense_tokens() > 0 or debug_trace_enabled()
    generated_lens = ctx.get("generated_lens") or []
    with _lock:
        for req_idx in range(batch_size):
            # Keep a row view into the small batched score tensor instead of
            # launching one clone/copy per request. The tensor is immutable
            # after this point, and the view keeps its backing storage alive
            # until the verifier pops it from the per-request queue.
            scores = per_req_scores[req_idx].detach()
            scores._speclink_sr24_score_base = per_req_scores
            scores._speclink_sr24_score_row = req_idx
            _pending_scores[req_ids[req_idx]].append(scores)
            if needs_generated_lens:
                generated_len = (
                    int(generated_lens[req_idx])
                    if req_idx < len(generated_lens)
                    else None
                )
                _pending_generated_lens[req_ids[req_idx]].append(generated_len)
            if _debug_trace_matches(req_ids[req_idx]):
                selected_ids = draft_token_ids[req_idx].detach().to("cpu").tolist()
                selected_scores = scores.detach().to("cpu").tolist()
                _write_debug_trace(
                    {
                        "timestamp": time.time(),
                        "event": "draft_scores",
                        "req_id": req_ids[req_idx],
                        "method": method or ctx.get("method") or "unknown",
                        "batch_index": req_idx,
                        "prompt_len": (
                            int(ctx["prompt_lens"][req_idx])
                            if req_idx < len(ctx.get("prompt_lens") or [])
                            else None
                        ),
                        "generated_len": (
                            int(generated_lens[req_idx])
                            if req_idx < len(generated_lens)
                            else None
                        ),
                        "num_spec_tokens": int(num_spec_tokens),
                        "draft_token_ids": selected_ids,
                        "draft_selected_prob": selected_scores,
                    }
                )


def begin_verify_context(residual_plan: VerifyResidualPlan | torch.Tensor | None) -> Any:
    global _fast_verify_residual_active
    global _fast_verify_residual_mask
    global _fast_verify_residual_state
    global _fast_verify_residual_priority
    global _fast_verify_residual_bucket
    global _fast_verify_residual_rows
    global _fast_verify_base_rows
    global _fast_verify_fixed_prefix_route
    if not enabled() or residual_plan is None:
        return None
    if isinstance(residual_plan, VerifyResidualPlan):
        mask = residual_plan.mask
        state = residual_plan.state
        priority = residual_plan.priority
        bucket = residual_plan.bucket
        residual_rows = residual_plan.residual_rows
        base_rows = residual_plan.base_rows
        fixed_prefix_route = residual_plan.fixed_prefix_route
    else:
        mask = residual_plan
        state = "mixed"
        priority = None
        bucket = None
        residual_rows = None
        base_rows = None
        fixed_prefix_route = None
    mask_token = _verify_residual_mask.set(mask)
    state_token = _verify_residual_state.set(state)
    priority_token = _verify_residual_priority.set(priority)
    bucket_token = _verify_residual_bucket.set(bucket)
    residual_rows_token = _verify_residual_rows.set(residual_rows)
    base_rows_token = _verify_base_rows.set(base_rows)
    fixed_prefix_route_token = _verify_fixed_prefix_route.set(fixed_prefix_route)
    previous_fast = (
        _fast_verify_residual_active,
        _fast_verify_residual_mask,
        _fast_verify_residual_state,
        _fast_verify_residual_priority,
        _fast_verify_residual_bucket,
        _fast_verify_residual_rows,
        _fast_verify_base_rows,
        _fast_verify_fixed_prefix_route,
    )
    _fast_verify_residual_active = True
    _fast_verify_residual_mask = mask
    _fast_verify_residual_state = state
    _fast_verify_residual_priority = priority
    _fast_verify_residual_bucket = bucket
    _fast_verify_residual_rows = residual_rows
    _fast_verify_base_rows = base_rows
    _fast_verify_fixed_prefix_route = fixed_prefix_route
    return (
        mask_token,
        state_token,
        priority_token,
        bucket_token,
        residual_rows_token,
        base_rows_token,
        fixed_prefix_route_token,
        previous_fast,
    )


def end_verify_context(token: Any) -> None:
    global _fast_verify_residual_active
    global _fast_verify_residual_mask
    global _fast_verify_residual_state
    global _fast_verify_residual_priority
    global _fast_verify_residual_bucket
    global _fast_verify_residual_rows
    global _fast_verify_base_rows
    global _fast_verify_fixed_prefix_route
    if token is not None:
        (
            mask_token,
            state_token,
            priority_token,
            bucket_token,
            residual_rows_token,
            base_rows_token,
            fixed_prefix_route_token,
            previous_fast,
        ) = token
        _verify_residual_mask.reset(mask_token)
        _verify_residual_state.reset(state_token)
        _verify_residual_priority.reset(priority_token)
        _verify_residual_bucket.reset(bucket_token)
        _verify_residual_rows.reset(residual_rows_token)
        _verify_base_rows.reset(base_rows_token)
        _verify_fixed_prefix_route.reset(fixed_prefix_route_token)
        (
            _fast_verify_residual_active,
            _fast_verify_residual_mask,
            _fast_verify_residual_state,
            _fast_verify_residual_priority,
            _fast_verify_residual_bucket,
            _fast_verify_residual_rows,
            _fast_verify_base_rows,
            _fast_verify_fixed_prefix_route,
        ) = previous_fast


def begin_cudagraph_capture_verify_context(
    *,
    rows: int,
    device: torch.device,
    active_requests: int | None = None,
    min_scheduled_tokens: int | None = None,
    max_scheduled_tokens: int | None = None,
    use_spec_decode: bool | None = None,
) -> Any:
    """Use the runtime static mask buffer while capturing mixed SR24 graphs.

    CUDA Graph replay does not rerun the Python Linear hooks, so mixed SR24 is
    only graph-safe when the captured selector tensors are persistent buffers
    whose contents are updated before replay. Bucketed dense-row correction and
    the row-routed MLP path are allowed only through persistent bucket/base-row
    buffers.
    """
    if not enabled() or not linear_hooks_enabled():
        return None
    if force_cudagraph_none_for_mixed_enabled():
        return None
    if not static_mask_buffer_enabled():
        return None
    if rows <= 0:
        return None
    fixed_prefix_capture_plan = _fixed_prefix_row_routed_cudagraph_plan(
        rows=rows,
        device=device,
        active_requests=active_requests,
        min_scheduled_tokens=min_scheduled_tokens,
        max_scheduled_tokens=max_scheduled_tokens,
        use_spec_decode=use_spec_decode,
    )
    if fixed_prefix_capture_plan is not None:
        _breakdown_count("cudagraph_capture_fixed_prefix_row_routed_context", 1)
        return begin_verify_context(fixed_prefix_capture_plan)
    scaled_bucket_without_hint = (
        residual_bucket_scale_by_active()
        and (not cudagraph_bucket_enabled() or cudagraph_bucket_active_hint() <= 0)
    )
    active_only_dense_compute = (
        bucket_dense_compute_active_only() and bucket_dense_copy_active_only()
    )
    active_only_dense_compute_unsafe = (
        active_only_dense_compute
        and not bucket_dense_active_mask_fused_graph_safe()
    )
    if (
        route_all_residual_rows()
        or route_reuse_base_output()
        or route_bucket_rows_graph_static_unsafe()
        or route_overlap_streams()
        or active_only_dense_compute_unsafe
        or scaled_bucket_without_hint
        or (residual_bucket_size() > 0 and not cudagraph_bucket_enabled())
    ):
        _breakdown_count("cudagraph_capture_static_mixed_context_unsafe_route", 1)
        return None
    mask = _static_residual_mask_view(rows, device, fill_value=True)
    bucket = (
        _static_residual_bucket_capture_view(rows=rows, device=device)
        if cudagraph_bucket_enabled()
        else None
    )
    _breakdown_count("cudagraph_capture_static_mixed_context", 1)
    if bucket is not None:
        _breakdown_count("cudagraph_capture_static_mixed_bucket_context", 1)
    residual_rows = None
    base_rows = None
    if route_bucket_rows() and bucket is not None:
        bucket_rows, _ = bucket
        residual_rows = bucket_rows
        base_rows = _static_bucket_complement_capture_view(
            rows=rows,
            bucket_rows=bucket_rows,
            device=device,
        )
        _breakdown_count(
            "cudagraph_capture_route_bucket_rows_context",
            1,
        )
    elif row_routed_mlp() and bucket is not None:
        bucket_rows, _ = bucket
        residual_rows = bucket_rows
        if row_routed_mlp_reuse_base_output():
            _breakdown_count(
                "cudagraph_capture_row_routed_reuse_base_rows_context",
                1,
            )
        else:
            base_rows = _static_bucket_complement_capture_view(
                rows=rows,
                bucket_rows=bucket_rows,
                device=device,
            )
            _breakdown_count(
                "cudagraph_capture_row_routed_packed_rows_context",
                1,
            )
    return begin_verify_context(
        VerifyResidualPlan(
            mask=mask,
            state="mixed",
            bucket=bucket,
            residual_rows=residual_rows,
            base_rows=base_rows,
        )
    )


def _fixed_prefix_row_routed_cudagraph_plan(
    *,
    rows: int,
    device: torch.device,
    active_requests: int | None,
    min_scheduled_tokens: int | None,
    max_scheduled_tokens: int | None,
    use_spec_decode: bool | None,
) -> VerifyResidualPlan | None:
    """Build a graph-capture route table for uniform fixed-prefix decode.

    The normal scheduler can update route rows at runtime, but CUDA Graph
    capture must see stable tensor addresses. For the score-free
    fixed_prefix+row-routed path the route is determined only by
    (active_requests, K): first prefix draft rows are dense, middle draft rows
    are sparse-only, and the bonus/non-draft row is dense. This makes the common
    full-batch decode graph safe without scanning a dynamic mask in Python.
    """
    rows = int(rows)
    if rows <= 0:
        return None
    if not (
        row_routed_mlp()
        and route_all_residual_rows()
        and route_all_skip_bucket()
        and fixed_prefix_route_fastpath_enabled()
        and selective_residual_policy() == "fixed_prefix"
        and selective_non_draft_policy() in {"all", "bonus"}
    ):
        return None
    if use_spec_decode is False:
        return None
    if active_requests is None or min_scheduled_tokens is None or max_scheduled_tokens is None:
        return None
    active_count = int(active_requests)
    scheduled_width = int(max_scheduled_tokens)
    if active_count <= 0 or scheduled_width <= 1:
        return None
    if int(min_scheduled_tokens) != scheduled_width:
        return None
    if rows != active_count * scheduled_width:
        # Padded tail graph shapes need a mask-aware route table; keep them on
        # the existing conservative capture path for now.
        return None
    prefix = max(0, int(selective_min_prefix_residual()))
    valid_width = scheduled_width - 1
    if valid_width < prefix:
        return None
    base_width = valid_width - prefix
    if base_width <= 0:
        return None
    group_width = prefix + 1
    total_residual = active_count * group_width
    total_base = active_count * base_width
    if total_residual <= 0 or total_base <= 0:
        return None

    mask = _static_residual_mask_view(rows, device, fill_value=False)
    offsets = _device_arange(total_residual, dtype=torch.long, device=device)
    req_offsets = torch.div(offsets, group_width, rounding_mode="floor")
    pos_offsets = offsets.remainder(group_width)
    row_offsets = torch.where(
        pos_offsets < prefix,
        pos_offsets,
        torch.full_like(pos_offsets, valid_width),
    )
    residual_rows = _static_long_view(
        "cudagraph_fixed_prefix_route_residual_rows",
        total_residual,
        device,
    )
    residual_rows.copy_(req_offsets * scheduled_width + row_offsets)
    mask.index_fill_(0, residual_rows, True)

    base_offsets = _device_arange(total_base, dtype=torch.long, device=device)
    base_req_offsets = torch.div(base_offsets, base_width, rounding_mode="floor")
    base_pos_offsets = base_offsets.remainder(base_width) + prefix
    base_rows = _static_long_view(
        "cudagraph_fixed_prefix_route_base_rows",
        total_base,
        device,
    )
    base_rows.copy_(base_req_offsets * scheduled_width + base_pos_offsets)
    _breakdown_count("cudagraph_capture_fixed_prefix_row_routed_rows", rows)
    _breakdown_count(
        "cudagraph_capture_fixed_prefix_row_routed_residual_rows",
        total_residual,
    )
    _breakdown_count(
        "cudagraph_capture_fixed_prefix_row_routed_base_rows",
        total_base,
    )
    plan_residual_rows = residual_rows
    plan_base_rows = base_rows
    if fixed_prefix_route_descriptor_only() and row_routed_mlp():
        plan_residual_rows = None
        plan_base_rows = None
        _breakdown_count(
            "cudagraph_capture_fixed_prefix_descriptor_only_rows_skipped",
            1,
        )
    return VerifyResidualPlan(
        mask=mask,
        state="mixed",
        residual_rows=plan_residual_rows,
        base_rows=plan_base_rows,
        fixed_prefix_route=FixedPrefixRouteDescriptor(
            active_count=active_count,
            scheduled_width=scheduled_width,
            valid_width=valid_width,
            prefix=prefix,
            dense_width=group_width,
            base_width=base_width,
        ),
    )


def _cudagraph_shape_key(
    *,
    mode_key: str,
    phase: str | None = None,
    active_requests: int | None = None,
    num_tokens_unpadded: int | None = None,
    num_tokens_padded: int | None = None,
    min_scheduled_tokens: int | None = None,
    max_scheduled_tokens: int | None = None,
    uniform_scheduled: bool | None = None,
    use_spec_decode: bool | None = None,
) -> str:
    parts = [f"mode={mode_key}"]
    if phase is not None:
        parts.append(f"phase={phase}")
    if active_requests is not None:
        parts.append(f"reqs={int(active_requests)}")
    if num_tokens_unpadded is not None or num_tokens_padded is not None:
        parts.append(
            "tokens="
            f"{'' if num_tokens_unpadded is None else int(num_tokens_unpadded)}"
            "->"
            f"{'' if num_tokens_padded is None else int(num_tokens_padded)}"
        )
    if min_scheduled_tokens is not None or max_scheduled_tokens is not None:
        parts.append(
            "sched="
            f"{'' if min_scheduled_tokens is None else int(min_scheduled_tokens)}"
            "-"
            f"{'' if max_scheduled_tokens is None else int(max_scheduled_tokens)}"
        )
    if uniform_scheduled is not None:
        parts.append(f"uniform={int(bool(uniform_scheduled))}")
    if use_spec_decode is not None:
        parts.append(f"spec={int(bool(use_spec_decode))}")
    return "|".join(parts)


def record_cudagraph_mode(
    runtime_mode: Any,
    *,
    phase: str | None = None,
    active_requests: int | None = None,
    num_tokens_unpadded: int | None = None,
    num_tokens_padded: int | None = None,
    min_scheduled_tokens: int | None = None,
    max_scheduled_tokens: int | None = None,
    uniform_scheduled: bool | None = None,
    use_spec_decode: bool | None = None,
) -> None:
    key = getattr(runtime_mode, "name", None) or str(runtime_mode)
    shape_key = _cudagraph_shape_key(
        mode_key=key,
        phase=phase,
        active_requests=active_requests,
        num_tokens_unpadded=num_tokens_unpadded,
        num_tokens_padded=num_tokens_padded,
        min_scheduled_tokens=min_scheduled_tokens,
        max_scheduled_tokens=max_scheduled_tokens,
        uniform_scheduled=uniform_scheduled,
        use_spec_decode=use_spec_decode,
    )
    _record_generic_cudagraph_mode(key, shape_key=shape_key)
    if not linear_hooks_enabled() or (
        not runtime_stats_enabled() and not breakdown_enabled()
    ):
        return
    _breakdown_count(f"cudagraph_mode_{key}", 1)
    event: dict[str, Any] | None = None
    with _lock:
        counts = _stats_accum.setdefault("cudagraph_mode_counts", {})
        counts[key] = int(counts.get(key, 0)) + 1
        shape_counts = _stats_accum.setdefault("cudagraph_shape_counts", {})
        shape_counts[shape_key] = int(shape_counts.get(shape_key, 0)) + 1
        _stats_accum["cudagraph_steps"] = int(
            _stats_accum.get("cudagraph_steps") or 0
        ) + 1
        interval = _stats_interval()
        if interval <= 0:
            return
        steps = int(_stats_accum["cudagraph_steps"])
        if (
            steps - int(_stats_accum.get("last_cudagraph_flush_steps") or 0)
            >= interval
        ):
            _stats_accum["last_cudagraph_flush_steps"] = steps
            event = {
                "timestamp": time.time(),
                "event": "sr24_cudagraph_summary",
                "stats_interval": interval,
                "cudagraph_steps": steps,
                "cudagraph_mode_counts": dict(counts),
                "cudagraph_shape_counts": dict(shape_counts),
            }
    if event is not None:
        _write_log(event)


def force_cudagraph_none_for_verify_plan(
    residual_plan: VerifyResidualPlan | torch.Tensor | None,
) -> bool:
    """Return whether this SR24 verify step should avoid CUDA Graph replay.

    Dynamic selective SR24 reads a per-step residual mask through SR24 context
    state inside Linear hooks. A replayed CUDA Graph can otherwise observe stale
    routing state. Keep the guard scoped to dynamic mixed plans so dense,
    base-only, all-corrected, and static all/no-residual ablations still use
    normal vLLM graph dispatch.
    """
    if not enabled() or not linear_hooks_enabled() or residual_plan is None:
        return False
    if (
        bucket_dense_compute_active_only()
        and bucket_dense_copy_active_only()
        and not bucket_dense_active_mask_fused_graph_safe()
    ):
        if isinstance(residual_plan, VerifyResidualPlan):
            force_none = residual_plan.state == "mixed"
        else:
            force_none = True
        if force_none:
            _breakdown_count(
                "cudagraph_forced_none_bucket_dense_compute_active_only", 1
            )
        return force_none
    if direct_cpu_route_rows_enabled() and route_all_residual_rows():
        if isinstance(residual_plan, VerifyResidualPlan):
            force_none = residual_plan.state == "mixed"
        else:
            force_none = True
        if force_none:
            _breakdown_count("cudagraph_forced_none_direct_cpu_route_rows", 1)
        return force_none
    if route_bucket_rows_graph_static_unsafe():
        if isinstance(residual_plan, VerifyResidualPlan):
            force_none = residual_plan.state == "mixed"
        else:
            force_none = True
        if force_none:
            _breakdown_count("cudagraph_forced_none_route_bucket_static", 1)
        return force_none
    if route_overlap_streams():
        if isinstance(residual_plan, VerifyResidualPlan):
            force_none = residual_plan.state == "mixed"
            if (
                force_none
                and route_overlap_allow_cudagraph()
                and residual_plan.fixed_prefix_route is not None
            ):
                _breakdown_count(
                    "cudagraph_allow_fixed_prefix_route_overlap_streams", 1
                )
                return False
        else:
            force_none = True
        if force_none:
            _breakdown_count("cudagraph_forced_none_route_overlap_streams", 1)
        return force_none
    if not force_cudagraph_none_for_mixed_enabled():
        return False
    if isinstance(residual_plan, VerifyResidualPlan):
        force_none = residual_plan.state == "mixed"
    else:
        force_none = True
    if force_none:
        _breakdown_count("cudagraph_forced_none_dynamic_mixed", 1)
    return force_none


def force_cudagraph_none_for_mixed_enabled() -> bool:
    raw = os.getenv("SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _accumulate_stats(record: dict[str, Any]) -> dict[str, Any] | None:
    if not runtime_stats_enabled():
        return None
    with _lock:
        record_exact = bool(record.get("stats_exact", True))
        if not record_exact:
            _stats_accum["stats_exact"] = False
        if bool(record.get("sync_reduced_stats", False)):
            _stats_accum["sync_reduced_stats"] = True
        if bool(record.get("dense_fallback_nonuniform", False)):
            _stats_accum["dense_fallback_nonuniform_steps"] = int(
                _stats_accum.get("dense_fallback_nonuniform_steps") or 0
            ) + 1
        _stats_accum["steps"] += 1
        _stats_accum["total_scheduled_tokens"] += int(
            record.get("total_scheduled_tokens") or 0
        )
        _stats_accum["total_draft_tokens"] += int(record.get("total_draft_tokens") or 0)
        _stats_accum["total_valid_draft_tokens"] += int(
            record.get("total_valid_draft_tokens") or 0
        )
        _stats_accum["non_draft_tokens"] += int(record.get("non_draft_tokens") or 0)
        _stats_accum["residual_draft_tokens"] += int(
            record.get("residual_draft_tokens") or 0
        )
        _stats_accum["base_only_draft_tokens"] += int(
            record.get("base_only_draft_tokens") or 0
        )
        _stats_accum["missing_score_tokens"] += int(
            record.get("missing_score_tokens") or 0
        )
        _stats_accum["residual_non_draft_tokens"] = int(
            _stats_accum.get("residual_non_draft_tokens") or 0
        ) + int(record.get("residual_non_draft_tokens") or 0)
        _stats_accum["base_only_non_draft_tokens"] = int(
            _stats_accum.get("base_only_non_draft_tokens") or 0
        ) + int(record.get("base_only_non_draft_tokens") or 0)
        _stats_accum["early_residual_draft_tokens"] = int(
            _stats_accum.get("early_residual_draft_tokens") or 0
        ) + int(record.get("early_residual_draft_tokens") or 0)
        _stats_accum["early_residual_non_draft_tokens"] = int(
            _stats_accum.get("early_residual_non_draft_tokens") or 0
        ) + int(record.get("early_residual_non_draft_tokens") or 0)
        _stats_accum["bucket_calls"] = int(
            _stats_accum.get("bucket_calls") or 0
        ) + int(record.get("bucket_calls") or 0)
        _stats_accum["bucket_candidate_rows"] = int(
            _stats_accum.get("bucket_candidate_rows") or 0
        ) + int(record.get("bucket_candidate_rows") or 0)
        _stats_accum["bucket_total_rows"] = int(
            _stats_accum.get("bucket_total_rows") or 0
        ) + int(record.get("bucket_total_rows") or 0)
        _stats_accum["bucket_residual_requested_rows"] = int(
            _stats_accum.get("bucket_residual_requested_rows") or 0
        ) + int(record.get("bucket_residual_requested_rows") or 0)
        if record.get("bucket_active_rows") is not None:
            _stats_accum["bucket_active_rows"] = int(
                _stats_accum.get("bucket_active_rows") or 0
            ) + int(record.get("bucket_active_rows") or 0)
        for timing_key in _RUNTIME_TIMING_KEYS:
            timing_value = record.get(timing_key)
            if timing_value is None:
                continue
            try:
                timing_float = float(timing_value)
            except (TypeError, ValueError):
                continue
            _stats_accum[timing_key] = float(
                _stats_accum.get(timing_key) or 0.0
            ) + timing_float
        interval = _stats_interval()
        if interval <= 0:
            return None
        steps = int(_stats_accum["steps"])
        if steps - int(_stats_accum["last_flush_steps"]) < interval:
            return None
        _stats_accum["last_flush_steps"] = steps
        total_draft = int(_stats_accum["total_draft_tokens"])
        residual = int(_stats_accum["residual_draft_tokens"])
        base_only = int(_stats_accum["base_only_draft_tokens"])
        bucket_calls = int(_stats_accum.get("bucket_calls") or 0)
        bucket_candidate_rows = int(
            _stats_accum.get("bucket_candidate_rows") or 0
        )
        bucket_active_rows = int(_stats_accum.get("bucket_active_rows") or 0)
        bucket_residual_requested_rows = int(
            _stats_accum.get("bucket_residual_requested_rows") or 0
        )
        stats_exact = bool(_stats_accum.get("stats_exact", True))
        summary = {
            "timestamp": time.time(),
            "event": "sr24_verify_summary",
            "mode": record.get("mode"),
            "threshold": record.get("threshold"),
            "prefix_threshold": record.get("prefix_threshold"),
            "stats_interval": interval,
            "selective_correct_non_draft": record.get("selective_correct_non_draft"),
            "selective_non_draft_policy": record.get("selective_non_draft_policy"),
            "selective_residual_policy": record.get("selective_residual_policy"),
            "selective_extra_after_low": record.get("selective_extra_after_low"),
            "selective_min_prefix_residual":
            record.get("selective_min_prefix_residual"),
            "selective_max_residual_draft_rows":
            record.get("selective_max_residual_draft_rows"),
            "low_confidence_cap_by_risk":
            record.get("low_confidence_cap_by_risk"),
            "early_dense_tokens": record.get("early_dense_tokens"),
            "sync_reduced_stats": bool(_stats_accum.get("sync_reduced_stats", False)),
            "sync_mask_state": record.get("sync_mask_state"),
            "static_mask_state": record.get("static_mask_state"),
            "adaptive_dense_fallback": adaptive_dense_fallback_enabled(),
            "adaptive_dense_fallback_no_residual_only":
            adaptive_dense_fallback_no_residual_only(),
            "adaptive_dense_fallback_small_rows":
            adaptive_dense_fallback_small_rows(),
            "adaptive_dense_fallback_gate_up_fraction":
            adaptive_dense_fallback_gate_up_fraction(),
            "adaptive_dense_fallback_down_fraction":
            adaptive_dense_fallback_down_fraction(),
            "adaptive_dense_fallback_small_down_no_residual":
            adaptive_dense_fallback_small_down_no_residual(),
            "adaptive_dense_fallback_small_gate_up_no_residual":
            adaptive_dense_fallback_small_gate_up_no_residual(),
            "dense_fallback_nonuniform": record.get(
                "dense_fallback_nonuniform",
                dense_fallback_nonuniform(),
            ),
            "dense_fallback_nonuniform_steps": int(
                _stats_accum.get("dense_fallback_nonuniform_steps") or 0
            ),
            "adaptive_dense_fallback_calls": int(
                _stats_accum.get("adaptive_dense_fallback_calls") or 0
            ),
            "adaptive_dense_fallback_rows": int(
                _stats_accum.get("adaptive_dense_fallback_rows") or 0
            ),
            "adaptive_dense_fallback_candidate_rows": int(
                _stats_accum.get("adaptive_dense_fallback_candidate_rows") or 0
            ),
            "mask_state": record.get("mask_state"),
            "mask_state_exact": record.get("mask_state_exact"),
            "stats_exact": stats_exact,
            "cudagraph_mode_counts": dict(
                _stats_accum.get("cudagraph_mode_counts") or {}
            ),
            "cudagraph_shape_counts": dict(
                _stats_accum.get("cudagraph_shape_counts") or {}
            ),
            "cudagraph_steps": int(_stats_accum.get("cudagraph_steps") or 0),
            "steps": steps,
            "total_scheduled_tokens": int(_stats_accum["total_scheduled_tokens"]),
            "total_draft_tokens": total_draft,
            "total_valid_draft_tokens": int(_stats_accum["total_valid_draft_tokens"]),
            "non_draft_tokens": int(_stats_accum["non_draft_tokens"]),
            "residual_draft_tokens": residual if stats_exact else None,
            "base_only_draft_tokens": base_only if stats_exact else None,
            "missing_score_tokens": (
                int(_stats_accum["missing_score_tokens"]) if stats_exact else None
            ),
            "residual_non_draft_tokens": int(
                _stats_accum.get("residual_non_draft_tokens") or 0
            ),
            "base_only_non_draft_tokens": int(
                _stats_accum.get("base_only_non_draft_tokens") or 0
            ),
            "early_residual_draft_tokens": int(
                _stats_accum.get("early_residual_draft_tokens") or 0
            ),
            "early_residual_non_draft_tokens": int(
                _stats_accum.get("early_residual_non_draft_tokens") or 0
            ),
            "residual_draft_fraction": (
                residual / total_draft if stats_exact and total_draft else None
            ),
            "base_only_draft_fraction": (
                base_only / total_draft if stats_exact and total_draft else None
            ),
            "bucket_calls": bucket_calls,
            "bucket_candidate_rows": bucket_candidate_rows,
            "bucket_active_rows": bucket_active_rows if bucket_calls else None,
            "bucket_total_rows": int(_stats_accum.get("bucket_total_rows") or 0),
            "bucket_residual_requested_rows": (
                bucket_residual_requested_rows if stats_exact else None
            ),
            "bucket_candidate_rows_per_call": (
                bucket_candidate_rows / bucket_calls if bucket_calls else None
            ),
            "bucket_active_rows_per_call": (
                bucket_active_rows / bucket_calls
                if bucket_calls and bucket_active_rows
                else None
            ),
            "bucket_active_fraction_of_requested": (
                bucket_active_rows / bucket_residual_requested_rows
                if (
                    stats_exact
                    and bucket_calls
                    and bucket_active_rows
                    and bucket_residual_requested_rows
                )
                else None
            ),
        }
        for timing_key in _RUNTIME_TIMING_KEYS:
            timing_sum = float(_stats_accum.get(timing_key) or 0.0)
            summary[timing_key] = timing_sum
            summary[f"{timing_key}_per_step"] = (
                timing_sum / steps if steps else None
            )
        return summary


def _record_adaptive_dense_fallback_runtime_stats(
    *,
    rows: int,
    dense_candidate_rows: int,
) -> None:
    if not runtime_stats_enabled():
        return
    with _lock:
        _stats_accum["adaptive_dense_fallback_calls"] = int(
            _stats_accum.get("adaptive_dense_fallback_calls") or 0
        ) + 1
        _stats_accum["adaptive_dense_fallback_rows"] = int(
            _stats_accum.get("adaptive_dense_fallback_rows") or 0
        ) + int(rows)
        _stats_accum["adaptive_dense_fallback_candidate_rows"] = int(
            _stats_accum.get("adaptive_dense_fallback_candidate_rows") or 0
        ) + int(dense_candidate_rows)


def _write_verify_record(
    record: dict[str, Any],
    summary_record: dict[str, Any] | None,
) -> None:
    if not runtime_stats_enabled():
        return
    if summary_record is not None:
        _write_log(summary_record)
    elif _stats_interval() <= 1:
        _write_log(record)


@triton.jit
def _sr24_priority_from_score(
    score,
    present,
    use_residual,
    valid_count,
    offs,
    position_priority_scale,
    min_prefix_residual: tl.constexpr,
    cutoff: tl.constexpr,
    policy_id: tl.constexpr,
    block_k: tl.constexpr,
):
    valid_float = valid_count + 0.0
    safe_valid = tl.maximum(valid_float, 1.0)
    position_rank = valid_float - offs.to(tl.float32)
    position_bonus = (position_rank / safe_valid) * 0.05
    position_bonus += position_rank * position_priority_scale
    safe_score = tl.minimum(tl.maximum(score, 0.0), 1.0)
    low_severity = tl.minimum(tl.maximum(cutoff - score, 0.0), 1.0)
    score_priority = low_severity
    if policy_id == 1:
        # high_confidence: capped buckets should keep the highest DLM-selected
        # probabilities, not just the earliest positions.
        score_priority = safe_score
    elif policy_id == 4:
        # prefix_confidence: speculative acceptance is a prefix event, so use
        # cumulative selected-token probability as the bucket value signal.
        running = 1.0
        prefix_priority = tl.zeros((block_k,), dtype=tl.float32)
        for pos in tl.static_range(0, block_k):
            value = tl.sum(tl.where(offs == pos, safe_score, 0.0), axis=0)
            running = running * value
            prefix_priority = tl.where(offs == pos, running, prefix_priority)
        score_priority = prefix_priority
    elif policy_id == 5:
        score_priority = 1.0
    priority = tl.where(
        present,
        2.0 + score_priority + position_bonus,
        3.5,
    )
    forced_prefix = offs < min_prefix_residual
    priority = tl.where(
        use_residual & forced_prefix,
        tl.maximum(priority, 5.0 + position_bonus),
        priority,
    )
    return tl.where(use_residual, priority, 0.0)


@triton.jit
def _sr24_capped_low_confidence_mask(
    risky,
    score,
    present,
    in_valid,
    offs,
    cutoff: tl.constexpr,
    residual_budget: tl.constexpr,
    min_prefix_residual: tl.constexpr,
    block_k: tl.constexpr,
):
    prefix = in_valid & (offs < min_prefix_residual)
    if residual_budget <= 0:
        return risky | prefix
    # Pick the most risky non-prefix low-confidence rows, not merely the first
    # rows that happened to cross the threshold. Missing scores are treated as
    # high risk. A tiny position bonus gives deterministic earliest-row ties.
    risk = tl.where(
        risky & (~prefix),
        tl.where(present, tl.maximum(cutoff - score, 0.0), 1.0),
        float("-inf"),
    )
    risk = risk + (block_k - offs).to(tl.float32) * 0.000001
    selected = tl.full((block_k,), False, tl.int1)
    for _ in tl.static_range(0, residual_budget):
        best = tl.max(risk, axis=0)
        take = risk == best
        take = take & (best > float("-inf"))
        selected = selected | take
        risk = tl.where(take, float("-inf"), risk)
    return prefix | selected


@triton.jit
def _sr24_cap_selected_residual_mask(
    use_residual,
    in_valid,
    offs,
    residual_budget: tl.constexpr,
    min_prefix_residual: tl.constexpr,
):
    if residual_budget <= 0:
        return use_residual
    forced_prefix = in_valid & (offs < min_prefix_residual)
    extra_budget = residual_budget - min_prefix_residual
    if extra_budget <= 0:
        return forced_prefix
    candidates = use_residual & (~forced_prefix)
    candidate_rank = tl.cumsum(candidates.to(tl.int32), 0)
    return forced_prefix | (candidates & (candidate_rank <= extra_budget))


@triton.jit
def _sr24_prefix_confidence_mask(
    score,
    present,
    in_valid,
    offs,
    prefix_cutoff: tl.constexpr,
    block_k: tl.constexpr,
):
    safe_score = tl.where(
        in_valid & present,
        tl.minimum(tl.maximum(score, 0.0), 1.0),
        0.0,
    )
    running = tl.full((), 1.0, tl.float32)
    selected = tl.full((block_k,), False, tl.int1)
    for pos in tl.static_range(0, block_k):
        value = tl.sum(tl.where(offs == pos, safe_score, 0.0), axis=0)
        running = running * value
        selected = selected | (
            (offs == pos) & in_valid & (running >= prefix_cutoff)
        )
    return in_valid & ((~present) | selected)


@triton.jit
def _critical_prefix_bonus_mask_uniform_direct_kernel(
    scores,
    generated_lens,
    residual_mask,
    residual_priority,
    bonus_priority_value,
    position_priority_scale,
    bonus_policy_id: tl.constexpr,
    cutoff: tl.constexpr,
    prefix_cutoff: tl.constexpr,
    extra_after_low: tl.constexpr,
    min_prefix_residual: tl.constexpr,
    policy_id: tl.constexpr,
    residual_budget: tl.constexpr,
    cap_by_risk: tl.constexpr,
    early_dense_tokens: tl.constexpr,
    valid_count: tl.constexpr,
    scheduled_stride: tl.constexpr,
    score_stride: tl.constexpr,
    write_priority: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    req = tl.program_id(0)
    offs = tl.arange(0, block_k)
    row_start = req * scheduled_stride
    generated_len = tl.load(generated_lens + req)
    in_valid = offs < valid_count
    score = tl.load(
        scores + req * score_stride + offs,
        mask=in_valid,
        other=float("nan"),
    )
    present = score == score
    if policy_id == 1:
        use_residual = in_valid & ((~present) | (score > cutoff))
    elif policy_id == 2:
        risky = in_valid & ((~present) | (score <= cutoff))
        any_risky = tl.max(tl.where(risky, 1, 0), axis=0) != 0
        use_residual = in_valid & any_risky
    elif policy_id == 3:
        risky = in_valid & ((~present) | (score <= cutoff))
        if residual_budget > 0:
            if cap_by_risk:
                use_residual = _sr24_capped_low_confidence_mask(
                    risky,
                    score,
                    present,
                    in_valid,
                    offs,
                    cutoff,
                    residual_budget,
                    min_prefix_residual,
                    block_k,
                )
            else:
                risky_rank = tl.cumsum(risky.to(tl.int32), 0)
                use_residual = risky & (risky_rank <= residual_budget)
        else:
            use_residual = risky
    elif policy_id == 4:
        use_residual = _sr24_prefix_confidence_mask(
            score,
            present,
            in_valid,
            offs,
            prefix_cutoff,
            block_k,
        )
    elif policy_id == 5:
        use_residual = in_valid & (offs < min_prefix_residual)
    else:
        low = in_valid & present & (score <= cutoff)
        first_low = tl.min(tl.where(low, offs, block_k), axis=0)
        use_residual = in_valid & ((~present) | (offs <= first_low + extra_after_low))
    if min_prefix_residual > 0:
        use_residual = use_residual | (in_valid & (offs < min_prefix_residual))
    if residual_budget > 0 and policy_id != 3:
        use_residual = _sr24_cap_selected_residual_mask(
            use_residual,
            in_valid,
            offs,
            residual_budget,
            min_prefix_residual,
        )
    early_use_residual = in_valid & ((offs + generated_len) < early_dense_tokens)
    if early_dense_tokens > 0:
        use_residual = use_residual | early_use_residual
    tl.store(residual_mask + row_start + offs, use_residual, mask=in_valid)
    if write_priority:
        priority = _sr24_priority_from_score(
            score,
            present,
            use_residual,
            valid_count,
            offs,
            position_priority_scale,
            min_prefix_residual,
            cutoff,
            policy_id,
            block_k,
        )
        if early_dense_tokens > 0:
            priority = tl.where(
                early_use_residual,
                tl.maximum(priority, 5.0),
                priority,
            )
        tl.store(residual_priority + row_start + offs, priority, mask=in_valid)
    bonus_use_residual = True
    if bonus_policy_id == 2:
        bonus_bad = in_valid & ((~present) | (score <= cutoff))
        bonus_use_residual = tl.max(tl.where(bonus_bad, 1, 0), axis=0) == 0
        if early_dense_tokens > 0:
            bonus_use_residual = bonus_use_residual | (
                (generated_len + valid_count) < early_dense_tokens
            )
    tl.store(residual_mask + row_start + valid_count, bonus_use_residual)
    if write_priority:
        tl.store(
            residual_priority + row_start + valid_count,
            tl.where(bonus_use_residual, bonus_priority_value, 0.0),
        )


@triton.jit
def _critical_prefix_bonus_mask_kernel(
    scores,
    generated_lens,
    starts,
    valid_rows,
    score_lens,
    has_bonus,
    residual_mask,
    residual_priority,
    bonus_priority_value,
    position_priority_scale,
    bonus_policy_id: tl.constexpr,
    cutoff: tl.constexpr,
    prefix_cutoff: tl.constexpr,
    extra_after_low: tl.constexpr,
    min_prefix_residual: tl.constexpr,
    policy_id: tl.constexpr,
    residual_budget: tl.constexpr,
    cap_by_risk: tl.constexpr,
    early_dense_tokens: tl.constexpr,
    max_k: tl.constexpr,
    write_priority: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    req = tl.program_id(0)
    offs = tl.arange(0, block_k)
    valid_count = tl.load(valid_rows + req)
    score_count = tl.load(score_lens + req)
    row_start = tl.load(starts + req)
    generated_len = tl.load(generated_lens + req)
    in_valid = offs < valid_count
    in_score = offs < score_count
    score = tl.load(
        scores + req * max_k + offs,
        mask=offs < max_k,
        other=float("nan"),
    )
    present = score == score
    missing_full = score_count < valid_count
    if policy_id == 1:
        use_residual = in_valid & ((~present) | (score > cutoff))
    elif policy_id == 2:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        any_risky = tl.max(tl.where(risky, 1, 0), axis=0) != 0
        use_residual = in_valid & any_risky
    elif policy_id == 3:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        if residual_budget > 0:
            if cap_by_risk:
                use_residual = _sr24_capped_low_confidence_mask(
                    risky,
                    score,
                    present,
                    in_valid,
                    offs,
                    cutoff,
                    residual_budget,
                    min_prefix_residual,
                    block_k,
                )
            else:
                risky_rank = tl.cumsum(risky.to(tl.int32), 0)
                use_residual = risky & (risky_rank <= residual_budget)
        else:
            use_residual = risky
    elif policy_id == 4:
        use_residual = _sr24_prefix_confidence_mask(
            score,
            present,
            in_valid,
            offs,
            prefix_cutoff,
            block_k,
        )
    elif policy_id == 5:
        use_residual = in_valid & (offs < min_prefix_residual)
    else:
        low = in_valid & in_score & present & (score <= cutoff)
        first_low = tl.min(tl.where(low, offs, block_k), axis=0)
        use_residual = in_valid & (
            missing_full | (~present) | (offs <= first_low + extra_after_low)
        )
    if min_prefix_residual > 0:
        use_residual = use_residual | (in_valid & (offs < min_prefix_residual))
    if residual_budget > 0 and policy_id != 3:
        use_residual = _sr24_cap_selected_residual_mask(
            use_residual,
            in_valid,
            offs,
            residual_budget,
            min_prefix_residual,
        )
    early_use_residual = in_valid & ((offs + generated_len) < early_dense_tokens)
    if early_dense_tokens > 0:
        use_residual = use_residual | early_use_residual
    tl.store(residual_mask + row_start + offs, use_residual, mask=in_valid)
    if write_priority:
        priority = _sr24_priority_from_score(
            score,
            present,
            use_residual,
            valid_count,
            offs,
            position_priority_scale,
            min_prefix_residual,
            cutoff,
            policy_id,
            block_k,
        )
        if early_dense_tokens > 0:
            priority = tl.where(
                early_use_residual,
                tl.maximum(priority, 5.0),
                priority,
            )
        tl.store(residual_priority + row_start + offs, priority, mask=in_valid)
    bonus = tl.load(has_bonus + req) != 0
    bonus_use_residual = True
    if bonus_policy_id == 2:
        bonus_bad = in_valid & (missing_full | (~present) | (score <= cutoff))
        bonus_use_residual = tl.max(tl.where(bonus_bad, 1, 0), axis=0) == 0
        if early_dense_tokens > 0:
            bonus_use_residual = bonus_use_residual | (
                (generated_len + valid_count) < early_dense_tokens
            )
    tl.store(
        residual_mask + row_start + valid_count,
        bonus_use_residual,
        mask=bonus,
    )
    if write_priority:
        tl.store(
            residual_priority + row_start + valid_count,
            tl.where(bonus_use_residual, bonus_priority_value, 0.0),
            mask=bonus,
        )


@triton.jit
def _critical_prefix_bonus_mask_indexed_kernel(
    scores,
    score_rows,
    generated_lens,
    starts,
    valid_rows,
    score_lens,
    has_bonus,
    residual_mask,
    residual_priority,
    bonus_priority_value,
    position_priority_scale,
    bonus_policy_id: tl.constexpr,
    cutoff: tl.constexpr,
    prefix_cutoff: tl.constexpr,
    extra_after_low: tl.constexpr,
    min_prefix_residual: tl.constexpr,
    policy_id: tl.constexpr,
    residual_budget: tl.constexpr,
    cap_by_risk: tl.constexpr,
    early_dense_tokens: tl.constexpr,
    score_stride: tl.constexpr,
    write_priority: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    req = tl.program_id(0)
    offs = tl.arange(0, block_k)
    valid_count = tl.load(valid_rows + req)
    score_count = tl.load(score_lens + req)
    score_row = tl.load(score_rows + req)
    row_start = tl.load(starts + req)
    generated_len = tl.load(generated_lens + req)
    in_valid = offs < valid_count
    in_score = (score_row >= 0) & (offs < score_count)
    score = tl.load(
        scores + score_row * score_stride + offs,
        mask=in_score,
        other=float("nan"),
    )
    present = score == score
    missing_full = score_count < valid_count
    if policy_id == 1:
        use_residual = in_valid & ((~present) | (score > cutoff))
    elif policy_id == 2:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        any_risky = tl.max(tl.where(risky, 1, 0), axis=0) != 0
        use_residual = in_valid & any_risky
    elif policy_id == 3:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        if residual_budget > 0:
            if cap_by_risk:
                use_residual = _sr24_capped_low_confidence_mask(
                    risky,
                    score,
                    present,
                    in_valid,
                    offs,
                    cutoff,
                    residual_budget,
                    min_prefix_residual,
                    block_k,
                )
            else:
                risky_rank = tl.cumsum(risky.to(tl.int32), 0)
                use_residual = risky & (risky_rank <= residual_budget)
        else:
            use_residual = risky
    elif policy_id == 4:
        use_residual = _sr24_prefix_confidence_mask(
            score,
            present,
            in_valid,
            offs,
            prefix_cutoff,
            block_k,
        )
    elif policy_id == 5:
        use_residual = in_valid & (offs < min_prefix_residual)
    else:
        low = in_valid & in_score & present & (score <= cutoff)
        first_low = tl.min(tl.where(low, offs, block_k), axis=0)
        use_residual = in_valid & (
            missing_full | (~present) | (offs <= first_low + extra_after_low)
        )
    if min_prefix_residual > 0:
        use_residual = use_residual | (in_valid & (offs < min_prefix_residual))
    if residual_budget > 0 and policy_id != 3:
        use_residual = _sr24_cap_selected_residual_mask(
            use_residual,
            in_valid,
            offs,
            residual_budget,
            min_prefix_residual,
        )
    early_use_residual = in_valid & ((offs + generated_len) < early_dense_tokens)
    if early_dense_tokens > 0:
        use_residual = use_residual | early_use_residual
    tl.store(residual_mask + row_start + offs, use_residual, mask=in_valid)
    if write_priority:
        priority = _sr24_priority_from_score(
            score,
            present,
            use_residual,
            valid_count,
            offs,
            position_priority_scale,
            min_prefix_residual,
            cutoff,
            policy_id,
            block_k,
        )
        if early_dense_tokens > 0:
            priority = tl.where(
                early_use_residual,
                tl.maximum(priority, 5.0),
                priority,
            )
        tl.store(residual_priority + row_start + offs, priority, mask=in_valid)
    bonus = tl.load(has_bonus + req) != 0
    bonus_use_residual = True
    if bonus_policy_id == 2:
        bonus_bad = in_valid & (missing_full | (~present) | (score <= cutoff))
        bonus_use_residual = tl.max(tl.where(bonus_bad, 1, 0), axis=0) == 0
        if early_dense_tokens > 0:
            bonus_use_residual = bonus_use_residual | (
                (generated_len + valid_count) < early_dense_tokens
            )
    tl.store(
        residual_mask + row_start + valid_count,
        bonus_use_residual,
        mask=bonus,
    )
    if write_priority:
        tl.store(
            residual_priority + row_start + valid_count,
            tl.where(bonus_use_residual, bonus_priority_value, 0.0),
            mask=bonus,
        )


@triton.jit
def _critical_prefix_bonus_mask_req_kernel(
    scores,
    generated_lens,
    num_draft_tokens,
    num_scheduled_tokens,
    cu_scheduled_tokens,
    residual_mask,
    residual_priority,
    bonus_priority_value,
    position_priority_scale,
    bonus_policy_id: tl.constexpr,
    total_num_scheduled_tokens: tl.constexpr,
    cutoff: tl.constexpr,
    prefix_cutoff: tl.constexpr,
    extra_after_low: tl.constexpr,
    min_prefix_residual: tl.constexpr,
    policy_id: tl.constexpr,
    residual_budget: tl.constexpr,
    cap_by_risk: tl.constexpr,
    early_dense_tokens: tl.constexpr,
    score_cols: tl.constexpr,
    score_stride: tl.constexpr,
    write_priority: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    req = tl.program_id(0)
    offs = tl.arange(0, block_k)
    draft_count = tl.load(num_draft_tokens + req)
    sched_count = tl.load(num_scheduled_tokens + req)
    row_end = tl.load(cu_scheduled_tokens + req)
    row_start = row_end - sched_count
    generated_len = tl.load(generated_lens + req)
    remaining = total_num_scheduled_tokens - row_start
    valid_count = tl.maximum(0, tl.minimum(draft_count, remaining))
    in_valid = (draft_count > 0) & (offs < valid_count)
    score_count = tl.minimum(score_cols, valid_count)
    in_score = offs < score_count
    score = tl.load(
        scores + req * score_stride + offs,
        mask=(draft_count > 0) & in_score,
        other=float("nan"),
    )
    present = score == score
    missing_full = score_count < valid_count
    if policy_id == 1:
        use_residual = in_valid & ((~present) | (score > cutoff))
    elif policy_id == 2:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        any_risky = tl.max(tl.where(risky, 1, 0), axis=0) != 0
        use_residual = in_valid & any_risky
    elif policy_id == 3:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        if residual_budget > 0:
            if cap_by_risk:
                use_residual = _sr24_capped_low_confidence_mask(
                    risky,
                    score,
                    present,
                    in_valid,
                    offs,
                    cutoff,
                    residual_budget,
                    min_prefix_residual,
                    block_k,
                )
            else:
                risky_rank = tl.cumsum(risky.to(tl.int32), 0)
                use_residual = risky & (risky_rank <= residual_budget)
        else:
            use_residual = risky
    elif policy_id == 4:
        use_residual = _sr24_prefix_confidence_mask(
            score,
            present,
            in_valid,
            offs,
            prefix_cutoff,
            block_k,
        )
    elif policy_id == 5:
        use_residual = in_valid & (offs < min_prefix_residual)
    else:
        low = in_valid & in_score & present & (score <= cutoff)
        first_low = tl.min(tl.where(low, offs, block_k), axis=0)
        use_residual = in_valid & (
            missing_full | (~present) | (offs <= first_low + extra_after_low)
        )
    if min_prefix_residual > 0:
        use_residual = use_residual | (in_valid & (offs < min_prefix_residual))
    if residual_budget > 0 and policy_id != 3:
        use_residual = _sr24_cap_selected_residual_mask(
            use_residual,
            in_valid,
            offs,
            residual_budget,
            min_prefix_residual,
        )
    early_use_residual = in_valid & ((offs + generated_len) < early_dense_tokens)
    if early_dense_tokens > 0:
        use_residual = use_residual | early_use_residual
    tl.store(residual_mask + row_start + offs, use_residual, mask=in_valid)
    if write_priority:
        priority = _sr24_priority_from_score(
            score,
            present,
            use_residual,
            valid_count,
            offs,
            position_priority_scale,
            min_prefix_residual,
            cutoff,
            policy_id,
            block_k,
        )
        if early_dense_tokens > 0:
            priority = tl.where(
                early_use_residual,
                tl.maximum(priority, 5.0),
                priority,
            )
        tl.store(residual_priority + row_start + offs, priority, mask=in_valid)
    bonus_row = row_start + valid_count
    bonus_limit = tl.minimum(row_end, total_num_scheduled_tokens)
    bonus_mask = (draft_count > 0) & (bonus_row < bonus_limit)
    bonus_use_residual = True
    if bonus_policy_id == 2:
        bonus_bad = in_valid & (missing_full | (~present) | (score <= cutoff))
        bonus_use_residual = tl.max(tl.where(bonus_bad, 1, 0), axis=0) == 0
        if early_dense_tokens > 0:
            bonus_use_residual = bonus_use_residual | (
                (generated_len + valid_count) < early_dense_tokens
            )
    tl.store(
        residual_mask + bonus_row,
        bonus_use_residual,
        mask=bonus_mask,
    )
    if write_priority:
        tl.store(
            residual_priority + bonus_row,
            tl.where(bonus_use_residual, bonus_priority_value, 0.0),
            mask=bonus_mask,
        )


@triton.jit
def _critical_prefix_bonus_mask_req_indexed_kernel(
    scores,
    score_rows,
    generated_lens,
    num_draft_tokens,
    num_scheduled_tokens,
    cu_scheduled_tokens,
    residual_mask,
    residual_priority,
    bonus_priority_value,
    position_priority_scale,
    bonus_policy_id: tl.constexpr,
    total_num_scheduled_tokens: tl.constexpr,
    cutoff: tl.constexpr,
    prefix_cutoff: tl.constexpr,
    extra_after_low: tl.constexpr,
    min_prefix_residual: tl.constexpr,
    policy_id: tl.constexpr,
    residual_budget: tl.constexpr,
    cap_by_risk: tl.constexpr,
    early_dense_tokens: tl.constexpr,
    score_cols: tl.constexpr,
    score_stride: tl.constexpr,
    write_priority: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    req = tl.program_id(0)
    offs = tl.arange(0, block_k)
    draft_count = tl.load(num_draft_tokens + req)
    sched_count = tl.load(num_scheduled_tokens + req)
    row_end = tl.load(cu_scheduled_tokens + req)
    row_start = row_end - sched_count
    remaining = total_num_scheduled_tokens - row_start
    valid_count = tl.maximum(0, tl.minimum(draft_count, remaining))
    score_row = tl.load(score_rows + req)
    generated_len = tl.load(generated_lens + req)
    in_valid = (draft_count > 0) & (offs < valid_count)
    score_count = tl.minimum(score_cols, valid_count)
    in_score = (score_row >= 0) & (offs < score_count)
    score = tl.load(
        scores + score_row * score_stride + offs,
        mask=(draft_count > 0) & in_score,
        other=float("nan"),
    )
    present = score == score
    missing_full = score_count < valid_count
    if policy_id == 1:
        use_residual = in_valid & ((~present) | (score > cutoff))
    elif policy_id == 2:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        any_risky = tl.max(tl.where(risky, 1, 0), axis=0) != 0
        use_residual = in_valid & any_risky
    elif policy_id == 3:
        risky = in_valid & (missing_full | (~present) | (score <= cutoff))
        if residual_budget > 0:
            if cap_by_risk:
                use_residual = _sr24_capped_low_confidence_mask(
                    risky,
                    score,
                    present,
                    in_valid,
                    offs,
                    cutoff,
                    residual_budget,
                    min_prefix_residual,
                    block_k,
                )
            else:
                risky_rank = tl.cumsum(risky.to(tl.int32), 0)
                use_residual = risky & (risky_rank <= residual_budget)
        else:
            use_residual = risky
    elif policy_id == 4:
        use_residual = _sr24_prefix_confidence_mask(
            score,
            present,
            in_valid,
            offs,
            prefix_cutoff,
            block_k,
        )
    elif policy_id == 5:
        use_residual = in_valid & (offs < min_prefix_residual)
    else:
        low = in_valid & in_score & present & (score <= cutoff)
        first_low = tl.min(tl.where(low, offs, block_k), axis=0)
        use_residual = in_valid & (
            missing_full | (~present) | (offs <= first_low + extra_after_low)
        )
    if min_prefix_residual > 0:
        use_residual = use_residual | (in_valid & (offs < min_prefix_residual))
    if residual_budget > 0 and policy_id != 3:
        use_residual = _sr24_cap_selected_residual_mask(
            use_residual,
            in_valid,
            offs,
            residual_budget,
            min_prefix_residual,
        )
    early_use_residual = in_valid & ((offs + generated_len) < early_dense_tokens)
    if early_dense_tokens > 0:
        use_residual = use_residual | early_use_residual
    tl.store(residual_mask + row_start + offs, use_residual, mask=in_valid)
    if write_priority:
        priority = _sr24_priority_from_score(
            score,
            present,
            use_residual,
            valid_count,
            offs,
            position_priority_scale,
            min_prefix_residual,
            cutoff,
            policy_id,
            block_k,
        )
        if early_dense_tokens > 0:
            priority = tl.where(
                early_use_residual,
                tl.maximum(priority, 5.0),
                priority,
            )
        tl.store(residual_priority + row_start + offs, priority, mask=in_valid)
    bonus_row = row_start + valid_count
    bonus_limit = tl.minimum(row_end, total_num_scheduled_tokens)
    bonus_mask = (draft_count > 0) & (bonus_row < bonus_limit)
    bonus_use_residual = True
    if bonus_policy_id == 2:
        bonus_bad = in_valid & (missing_full | (~present) | (score <= cutoff))
        bonus_use_residual = tl.max(tl.where(bonus_bad, 1, 0), axis=0) == 0
        if early_dense_tokens > 0:
            bonus_use_residual = bonus_use_residual | (
                (generated_len + valid_count) < early_dense_tokens
            )
    tl.store(
        residual_mask + bonus_row,
        bonus_use_residual,
        mask=bonus_mask,
    )
    if write_priority:
        tl.store(
            residual_priority + bonus_row,
            tl.where(bonus_use_residual, bonus_priority_value, 0.0),
            mask=bonus_mask,
        )


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _graph_static_bucket_enabled() -> bool:
    return (
        static_mask_buffer_enabled()
        and not force_cudagraph_none_for_mixed_enabled()
        and cudagraph_bucket_enabled()
    )


def _build_direct_position_bucket_from_active(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    rows: int,
    device: torch.device,
    value_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not direct_position_bucket_enabled():
        return None
    bucket_size = _effective_residual_bucket_size(len(active))
    if bucket_size <= 0:
        return None
    rows = int(rows)
    graph_static_bucket = _graph_static_bucket_enabled()
    if rows <= bucket_size and not graph_static_bucket:
        return None
    min_valid = min((int(item[2]) for item in active), default=0)
    if min_valid > 0 and len(active) > 0 and bucket_size <= len(active) * min_valid:
        # Fast path for the common full-batch speculative decode shape:
        # position-major draft rows only, e.g. all request pos0, then all
        # request pos1. This avoids Python construction of bucket_size row ids
        # and avoids a global top-k/sort. It intentionally falls back whenever
        # the fixed bucket would need padding or bonus rows, because downstream
        # row-routed paths treat bucket row ids as dense rows.
        direct_cpu_start = _breakdown_cpu_start()
        direct_cuda_start = _breakdown_cuda_start()
        active_count = len(active)
        start_values = [int(item[1]) for item in active]
        starts = _static_long_view(
            "direct_position_bucket_starts",
            active_count,
            device,
        )
        _copy_long_values_to_device(
            name="direct_position_bucket_starts",
            dst=starts,
            values=start_values,
        )
        offsets = _device_arange(bucket_size, dtype=torch.long, device=device)
        req_offsets = offsets.remainder(active_count)
        pos_offsets = torch.div(offsets, active_count, rounding_mode="floor")
        selected_rows = starts.index_select(0, req_offsets) + pos_offsets
        if graph_static_bucket:
            bucket_rows = _static_long_view("bucket_rows", bucket_size, device)
            bucket_values = _static_float_view(
                "bucket_values",
                bucket_size,
                device,
                dtype=value_dtype,
            )
            bucket_rows.copy_(selected_rows, non_blocking=False)
            bucket_values.fill_(1.0)
            _breakdown_count("scheduler_static_bucket_copy_for_graph", 1)
        else:
            bucket_rows = selected_rows.contiguous()
            bucket_values = torch.ones(
                (bucket_size,),
                dtype=value_dtype,
                device=device,
            )
        _breakdown_record_cuda(
            "scheduler_direct_position_bucket_cuda_ms",
            direct_cuda_start,
        )
        _breakdown_record_cpu(
            "scheduler_direct_position_bucket_cpu_ms",
            direct_cpu_start,
        )
        _breakdown_count("scheduler_direct_position_bucket_vector_builds", 1)
        _breakdown_count("scheduler_direct_position_bucket_builds", 1)
        _breakdown_count("scheduler_direct_position_bucket_rows", bucket_size)
        return bucket_rows, bucket_values

    direct_cpu_start = _breakdown_cpu_start()
    direct_cuda_start = _breakdown_cuda_start()
    selected: list[int] = []
    max_valid = 0
    for _req_idx, _start, valid, _score_len, _n, _has_bonus, _scores in active:
        max_valid = max(max_valid, int(valid))
    for pos in range(max_valid):
        for _req_idx, start, valid, _score_len, _n, _has_bonus, _scores in active:
            if len(selected) >= bucket_size:
                break
            if pos < int(valid):
                row = int(start) + pos
                if 0 <= row < rows:
                    selected.append(row)
        if len(selected) >= bucket_size:
            break
    if len(selected) < bucket_size:
        for _req_idx, start, valid, _score_len, _n, has_bonus, _scores in active:
            if len(selected) >= bucket_size:
                break
            if has_bonus:
                row = int(start) + int(valid)
                if 0 <= row < rows:
                    selected.append(row)
    if len(selected) < bucket_size:
        return None

    selected = selected[:bucket_size]
    if graph_static_bucket:
        bucket_rows = _static_long_view("bucket_rows", bucket_size, device)
        bucket_values = _static_float_view(
            "bucket_values",
            bucket_size,
            device,
            dtype=value_dtype,
        )
        _copy_long_values_to_device(
            name="direct_position_bucket_rows",
            dst=bucket_rows,
            values=selected,
        )
        bucket_values.fill_(1.0)
        _breakdown_count("scheduler_static_bucket_copy_for_graph", 1)
    else:
        bucket_rows = torch.empty((bucket_size,), dtype=torch.long, device=device)
        _copy_long_values_to_device(
            name="direct_position_bucket_rows",
            dst=bucket_rows,
            values=selected,
        )
        bucket_values = torch.ones(
            (bucket_size,),
            dtype=value_dtype,
            device=device,
        )
    _breakdown_record_cuda(
        "scheduler_direct_position_bucket_cuda_ms",
        direct_cuda_start,
    )
    _breakdown_record_cpu(
        "scheduler_direct_position_bucket_cpu_ms",
        direct_cpu_start,
    )
    _breakdown_count("scheduler_direct_position_bucket_builds", 1)
    _breakdown_count("scheduler_direct_position_bucket_rows", bucket_size)
    return bucket_rows, bucket_values


def _fixed_prefix_requested_residual_rows(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    min_prefix_residual: int,
    non_draft_policy: str,
    rows: int,
) -> int:
    rows = int(rows)
    if rows <= 0:
        return 0
    prefix = max(0, int(min_prefix_residual))
    if non_draft_policy == "all":
        base_only_draft_rows = 0
        for _req_idx, start, valid, _score_len, _n, _has_bonus, _scores in active:
            start = int(start)
            valid = max(0, int(valid))
            if valid <= 0:
                continue
            for pos in range(prefix, valid):
                row = start + pos
                if 0 <= row < rows:
                    base_only_draft_rows += 1
        return max(0, rows - base_only_draft_rows)
    requested = 0
    for _req_idx, start, valid, _score_len, _n, has_bonus, _scores in active:
        start = int(start)
        valid = max(0, int(valid))
        if valid <= 0:
            continue
        for pos in range(min(prefix, valid)):
            row = start + pos
            if 0 <= row < rows:
                requested += 1
        if non_draft_policy == "bonus" and has_bonus:
            bonus_row = start + valid
            if 0 <= bonus_row < rows:
                requested += 1
    return requested


def _score_tensor_cpu_view(
    scores: torch.Tensor,
    cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    base = getattr(scores, "_speclink_sr24_score_base", None)
    row = getattr(scores, "_speclink_sr24_score_row", None)
    if isinstance(base, torch.Tensor) and row is not None:
        key = id(base)
        cpu_base = cache.get(key)
        if cpu_base is None:
            cpu_base = base.detach().to(device="cpu")
            cache[key] = cpu_base
            _breakdown_count("scheduler_direct_cpu_route_score_base_copies", 1)
        return cpu_base[int(row)]
    _breakdown_count("scheduler_direct_cpu_route_score_row_copies", 1)
    return scores.detach().to(device="cpu")


def _direct_cpu_bonus_uses_residual(
    *,
    non_draft_policy: str,
    scores: torch.Tensor | None,
    score_len: int,
    valid: int,
    cutoff: float,
    score_cache: dict[int, torch.Tensor],
) -> bool:
    if non_draft_policy == "bonus":
        return True
    if non_draft_policy != "predicted_full_accept":
        return False
    if scores is None or int(score_len) < int(valid):
        return False
    score_cpu = _score_tensor_cpu_view(scores, score_cache)
    for value in score_cpu[: int(valid)].tolist():
        score = float(value)
        if math.isnan(score) or score <= float(cutoff):
            return False
    return True


def _build_direct_cpu_prefix_confidence_route_rows(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    rows: int,
    device: torch.device,
    prefix_cutoff: float,
    min_prefix_residual: int,
    non_draft_policy: str,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not direct_cpu_route_rows_enabled():
        return None
    rows = int(rows)
    if rows <= 0 or not active:
        return None
    cpu_start = _breakdown_cpu_start()
    residual_rows: list[int] = []
    residual_seen: set[int] = set()
    score_cache: dict[int, torch.Tensor] = {}
    cutoff = float(prefix_cutoff)
    min_prefix = max(0, int(min_prefix_residual))
    for _req_idx, start, valid, score_len, _n, has_bonus, scores in active:
        start = int(start)
        valid = max(0, int(valid))
        if valid <= 0:
            continue
        score_values: list[float] | None = None
        if scores is not None and int(score_len) > 0:
            score_cpu = _score_tensor_cpu_view(scores, score_cache)
            score_values = score_cpu[: min(int(score_len), valid)].tolist()
        running = 1.0
        for pos in range(valid):
            row = start + pos
            if row < 0 or row >= rows:
                continue
            present = (
                score_values is not None
                and pos < len(score_values)
                and not math.isnan(float(score_values[pos]))
            )
            if present:
                score = min(max(float(score_values[pos]), 0.0), 1.0)
                running *= score
                use_residual = running >= cutoff
            else:
                running = 0.0
                use_residual = True
            if min_prefix > 0 and pos < min_prefix:
                use_residual = True
            if use_residual and row not in residual_seen:
                residual_seen.add(row)
                residual_rows.append(row)
        if has_bonus and _direct_cpu_bonus_uses_residual(
            non_draft_policy=non_draft_policy,
            scores=scores,
            score_len=score_len,
            valid=valid,
            cutoff=cutoff,
            score_cache=score_cache,
        ):
            bonus_row = start + valid
            if 0 <= bonus_row < rows and bonus_row not in residual_seen:
                residual_seen.add(bonus_row)
                residual_rows.append(bonus_row)
    if not residual_rows:
        residual_tensor = torch.empty(0, dtype=torch.long, device=device)
    else:
        residual_tensor = _static_long_view(
            "direct_cpu_route_residual_rows",
            len(residual_rows),
            device,
        )
        _copy_long_values_to_device(
            name="direct_cpu_route_residual_rows",
            dst=residual_tensor,
            values=residual_rows,
        )
    if len(residual_seen) >= rows:
        base_rows: list[int] = []
    else:
        base_rows = [row for row in range(rows) if row not in residual_seen]
    if not base_rows:
        base_tensor = torch.empty(0, dtype=torch.long, device=device)
    else:
        base_tensor = _static_long_view(
            "direct_cpu_route_base_rows",
            len(base_rows),
            device,
        )
        _copy_long_values_to_device(
            name="direct_cpu_route_base_rows",
            dst=base_tensor,
            values=base_rows,
        )
    _breakdown_record_cpu("scheduler_direct_cpu_route_rows_cpu_ms", cpu_start)
    _breakdown_count("scheduler_direct_cpu_route_rows_builds", 1)
    _breakdown_count("scheduler_direct_cpu_route_rows_total_rows", rows)
    _breakdown_count("scheduler_direct_cpu_route_rows_residual", len(residual_rows))
    _breakdown_count("scheduler_direct_cpu_route_rows_base", len(base_rows))
    return residual_tensor, base_tensor


def _build_direct_cpu_fixed_prefix_route_rows(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    rows: int,
    device: torch.device,
    min_prefix_residual: int,
    non_draft_policy: str,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not direct_cpu_route_rows_enabled():
        return None
    rows = int(rows)
    if rows <= 0 or not active:
        return None
    cpu_start = _breakdown_cpu_start()
    residual_rows: list[int] = []
    residual_seen: set[int] = set()
    prefix = max(0, int(min_prefix_residual))
    score_cache: dict[int, torch.Tensor] = {}
    if non_draft_policy == "all":
        base_seen: set[int] = set()
        base_rows: list[int] = []
        for _req_idx, start, valid, _score_len, _n, _has_bonus, _scores in active:
            start = int(start)
            valid = max(0, int(valid))
            if valid <= prefix:
                continue
            for pos in range(prefix, valid):
                row = start + pos
                if 0 <= row < rows and row not in base_seen:
                    base_seen.add(row)
                    base_rows.append(row)
        residual_rows = [row for row in range(rows) if row not in base_seen]
        if residual_rows:
            residual_tensor = _static_long_view(
                "direct_cpu_route_residual_rows",
                len(residual_rows),
                device,
            )
            _copy_long_values_to_device(
                name="direct_cpu_route_residual_rows",
                dst=residual_tensor,
                values=residual_rows,
            )
        else:
            residual_tensor = torch.empty(0, dtype=torch.long, device=device)
        if base_rows:
            base_tensor = _static_long_view(
                "direct_cpu_route_base_rows",
                len(base_rows),
                device,
            )
            _copy_long_values_to_device(
                name="direct_cpu_route_base_rows",
                dst=base_tensor,
                values=base_rows,
            )
        else:
            base_tensor = torch.empty(0, dtype=torch.long, device=device)
        _breakdown_record_cpu("scheduler_direct_cpu_route_rows_cpu_ms", cpu_start)
        _breakdown_count("scheduler_direct_cpu_fixed_prefix_route_rows_builds", 1)
        _breakdown_count("scheduler_direct_cpu_fixed_prefix_all_rows_builds", 1)
        _breakdown_count("scheduler_direct_cpu_route_rows_total_rows", rows)
        _breakdown_count("scheduler_direct_cpu_route_rows_residual", len(residual_rows))
        _breakdown_count("scheduler_direct_cpu_route_rows_base", len(base_rows))
        return residual_tensor, base_tensor
    for _req_idx, start, valid, score_len, _n, has_bonus, scores in active:
        start = int(start)
        valid = max(0, int(valid))
        if valid <= 0:
            continue
        for pos in range(min(prefix, valid)):
            row = start + pos
            if 0 <= row < rows and row not in residual_seen:
                residual_seen.add(row)
                residual_rows.append(row)
        if has_bonus and _direct_cpu_bonus_uses_residual(
            non_draft_policy=non_draft_policy,
            scores=scores,
            score_len=score_len,
            valid=valid,
            cutoff=cutoff,
            score_cache=score_cache,
        ):
            bonus_row = start + valid
            if 0 <= bonus_row < rows and bonus_row not in residual_seen:
                residual_seen.add(bonus_row)
                residual_rows.append(bonus_row)
    if residual_rows:
        residual_tensor = _static_long_view(
            "direct_cpu_route_residual_rows",
            len(residual_rows),
            device,
        )
        _copy_long_values_to_device(
            name="direct_cpu_route_residual_rows",
            dst=residual_tensor,
            values=residual_rows,
        )
    else:
        residual_tensor = torch.empty(0, dtype=torch.long, device=device)
    base_rows = [row for row in range(rows) if row not in residual_seen]
    if base_rows:
        base_tensor = _static_long_view(
            "direct_cpu_route_base_rows",
            len(base_rows),
            device,
        )
        _copy_long_values_to_device(
            name="direct_cpu_route_base_rows",
            dst=base_tensor,
            values=base_rows,
        )
    else:
        base_tensor = torch.empty(0, dtype=torch.long, device=device)
    _breakdown_record_cpu("scheduler_direct_cpu_route_rows_cpu_ms", cpu_start)
    _breakdown_count("scheduler_direct_cpu_fixed_prefix_route_rows_builds", 1)
    _breakdown_count("scheduler_direct_cpu_route_rows_total_rows", rows)
    _breakdown_count("scheduler_direct_cpu_route_rows_residual", len(residual_rows))
    _breakdown_count("scheduler_direct_cpu_route_rows_base", len(base_rows))
    return residual_tensor, base_tensor


def _try_build_fixed_prefix_bonus_route_rows_vectorized(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    rows: int,
    device: torch.device,
    min_prefix_residual: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    FixedPrefixRouteDescriptor | None,
] | None:
    """Build fixed-prefix+bonus route rows without per-row Python lists.

    This handles the common full decode shape used by fixed-prefix SR24:
    each active request has the same number of valid draft rows and the verifier
    bonus row is present. The route is graph-friendly because the returned row
    tensors use the static SR24 buffers.
    """
    rows = int(rows)
    prefix = max(0, int(min_prefix_residual))
    active_count = len(active)
    if rows <= 0 or active_count <= 0 or prefix < 0:
        return None
    if prefix == 0:
        group_width = 1
    else:
        group_width = prefix + 1

    start_values: list[int] = []
    valid_value: int | None = None
    for _req_idx, start, valid, _score_len, _n, has_bonus, _scores in active:
        start = int(start)
        valid = int(valid)
        if start < 0 or valid <= 0 or not has_bonus:
            return None
        if valid < prefix:
            return None
        if valid_value is None:
            valid_value = valid
        elif valid != valid_value:
            return None
        bonus_row = start + valid
        if bonus_row < 0 or bonus_row >= rows:
            return None
        start_values.append(start)
    if valid_value is None:
        return None

    total_residual = active_count * group_width
    if total_residual <= 0 or total_residual >= rows:
        return None

    cpu_start = _breakdown_cpu_start()
    cuda_start = _breakdown_cuda_start()
    valid_width = int(valid_value)
    scheduled_width = valid_width + 1
    contiguous_blocks = (
        scheduled_width > 0
        and rows == active_count * scheduled_width
        and all(
            int(start) == req_idx * scheduled_width
            for req_idx, start in enumerate(start_values)
        )
    )
    if contiguous_blocks:
        descriptor_base_width = max(0, valid_width - prefix)
        fixed_prefix_route = FixedPrefixRouteDescriptor(
            active_count=active_count,
            scheduled_width=scheduled_width,
            valid_width=valid_width,
            prefix=prefix,
            dense_width=group_width,
            base_width=descriptor_base_width,
        )
        if fixed_prefix_route_descriptor_only():
            _breakdown_record_cuda(
                "scheduler_fixed_prefix_route_descriptor_only_cuda_ms",
                cuda_start,
            )
            _breakdown_record_cpu(
                "scheduler_fixed_prefix_route_descriptor_only_cpu_ms",
                cpu_start,
            )
            _breakdown_count("scheduler_fixed_prefix_route_descriptor_only_builds", 1)
            _breakdown_count("scheduler_fixed_prefix_route_contiguous_builds", 1)
            _breakdown_count("scheduler_fixed_prefix_route_rows_total_rows", rows)
            _breakdown_count(
                "scheduler_fixed_prefix_route_rows_residual",
                active_count * group_width,
            )
            _breakdown_count(
                "scheduler_fixed_prefix_route_rows_base",
                active_count * descriptor_base_width,
            )
            return None, None, fixed_prefix_route
    starts = _static_long_view(
        "fixed_prefix_route_vector_starts",
        active_count,
        device,
    )
    _copy_long_values_to_device(
        name="fixed_prefix_route_vector_starts",
        dst=starts,
        values=start_values,
    )
    offsets = _device_arange(total_residual, dtype=torch.long, device=device)
    req_offsets = torch.div(offsets, group_width, rounding_mode="floor")
    pos_offsets = offsets.remainder(group_width)
    row_offsets = torch.where(
        pos_offsets < prefix,
        pos_offsets,
        torch.full_like(pos_offsets, int(valid_value)),
    )
    selected_rows = starts.index_select(0, req_offsets) + row_offsets
    residual_tensor = _static_long_view(
        "fixed_prefix_route_residual_rows",
        total_residual,
        device,
    )
    residual_tensor.copy_(selected_rows, non_blocking=False)
    fixed_prefix_route: FixedPrefixRouteDescriptor | None = None
    if contiguous_blocks:
        base_width = max(0, valid_width - prefix)
        total_base = active_count * base_width
        if total_base > 0:
            base_offsets = _device_arange(total_base, dtype=torch.long, device=device)
            base_req_offsets = torch.div(
                base_offsets,
                base_width,
                rounding_mode="floor",
            )
            base_pos_offsets = base_offsets.remainder(base_width) + prefix
            selected_base_rows = (
                starts.index_select(0, base_req_offsets) + base_pos_offsets
            )
            base_tensor = _static_long_view(
                "fixed_prefix_route_base_rows",
                total_base,
                device,
            )
            base_tensor.copy_(selected_base_rows, non_blocking=False)
        else:
            base_tensor = torch.empty(0, dtype=torch.long, device=device)
        fixed_prefix_route = FixedPrefixRouteDescriptor(
            active_count=active_count,
            scheduled_width=scheduled_width,
            valid_width=valid_width,
            prefix=prefix,
            dense_width=group_width,
            base_width=base_width,
        )
        _breakdown_count("scheduler_fixed_prefix_route_contiguous_builds", 1)
    else:
        base_tensor = _compute_bucket_complement_rows(
            rows=rows,
            device=device,
            bucket_rows=residual_tensor,
        )
        if base_tensor is None:
            return None
    _breakdown_record_cuda(
        "scheduler_fixed_prefix_route_rows_cuda_ms",
        cuda_start,
    )
    _breakdown_record_cpu(
        "scheduler_fixed_prefix_route_rows_cpu_ms",
        cpu_start,
    )
    _breakdown_count("scheduler_fixed_prefix_route_vectorized_builds", 1)
    _breakdown_count("scheduler_fixed_prefix_route_bonus_builds", 1)
    _breakdown_count("scheduler_fixed_prefix_route_rows_total_rows", rows)
    _breakdown_count("scheduler_fixed_prefix_route_rows_residual", total_residual)
    _breakdown_count("scheduler_fixed_prefix_route_rows_base", int(base_tensor.numel()))
    return residual_tensor, base_tensor, fixed_prefix_route


def _build_fixed_prefix_all_route_rows_fastpath(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    rows: int,
    device: torch.device,
    min_prefix_residual: int,
    non_draft_policy: str,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    FixedPrefixRouteDescriptor | None,
] | None:
    if not fixed_prefix_route_fastpath_enabled():
        return None
    rows = int(rows)
    if rows <= 0 or not active:
        return None
    cpu_start = _breakdown_cpu_start()
    cuda_start = _breakdown_cuda_start()
    prefix = max(0, int(min_prefix_residual))
    if non_draft_policy == "all":
        vectorized = _try_build_fixed_prefix_bonus_route_rows_vectorized(
            active=active,
            rows=rows,
            device=device,
            min_prefix_residual=min_prefix_residual,
        )
        if vectorized is not None:
            _breakdown_count("scheduler_fixed_prefix_route_all_vectorized_builds", 1)
            return vectorized
        base_rows: list[int] = []
        base_seen: set[int] = set()
        for _req_idx, start, valid, _score_len, _n, _has_bonus, _scores in active:
            start = int(start)
            valid = max(0, int(valid))
            if valid <= prefix:
                continue
            for pos in range(prefix, valid):
                row = start + pos
                if 0 <= row < rows:
                    base_rows.append(row)
                    base_seen.add(row)
        residual_rows = (
            []
            if len(base_seen) >= rows
            else [row for row in range(rows) if row not in base_seen]
        )
    elif non_draft_policy == "bonus":
        vectorized = _try_build_fixed_prefix_bonus_route_rows_vectorized(
            active=active,
            rows=rows,
            device=device,
            min_prefix_residual=min_prefix_residual,
        )
        if vectorized is not None:
            return vectorized
        residual_rows = []
        residual_seen: set[int] = set()
        for _req_idx, start, valid, _score_len, _n, has_bonus, _scores in active:
            start = int(start)
            valid = max(0, int(valid))
            for pos in range(min(prefix, valid)):
                row = start + pos
                if 0 <= row < rows and row not in residual_seen:
                    residual_seen.add(row)
                    residual_rows.append(row)
            if has_bonus:
                bonus_row = start + valid
                if 0 <= bonus_row < rows and bonus_row not in residual_seen:
                    residual_seen.add(bonus_row)
                    residual_rows.append(bonus_row)
        base_rows = [row for row in range(rows) if row not in residual_seen]
    else:
        return None

    if base_rows:
        base_tensor = _static_long_view(
            "fixed_prefix_route_base_rows",
            len(base_rows),
            device,
        )
        _copy_long_values_to_device(
            name="fixed_prefix_route_base_rows",
            dst=base_tensor,
            values=base_rows,
        )
    else:
        base_tensor = torch.empty(0, dtype=torch.long, device=device)
    if not residual_rows:
        residual_tensor = torch.empty(0, dtype=torch.long, device=device)
    else:
        residual_tensor = _static_long_view(
            "fixed_prefix_route_residual_rows",
            len(residual_rows),
            device,
        )
        _copy_long_values_to_device(
            name="fixed_prefix_route_residual_rows",
            dst=residual_tensor,
            values=residual_rows,
        )
    _breakdown_record_cuda(
        "scheduler_fixed_prefix_route_rows_cuda_ms",
        cuda_start,
    )
    _breakdown_record_cpu(
        "scheduler_fixed_prefix_route_rows_cpu_ms",
        cpu_start,
    )
    _breakdown_count("scheduler_fixed_prefix_route_fastpath_builds", 1)
    _breakdown_count(f"scheduler_fixed_prefix_route_{non_draft_policy}_builds", 1)
    _breakdown_count("scheduler_fixed_prefix_route_rows_total_rows", rows)
    _breakdown_count(
        "scheduler_fixed_prefix_route_rows_residual",
        int(residual_tensor.numel()),
    )
    _breakdown_count("scheduler_fixed_prefix_route_rows_base", int(base_tensor.numel()))
    return residual_tensor, base_tensor, None


def _build_direct_cpu_critical_prefix_route_rows(
    *,
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]],
    active_generated_lens: list[int],
    rows: int,
    device: torch.device,
    cutoff: float,
    extra_after_low: int,
    min_prefix_residual: int,
    early_tokens: int,
    non_draft_policy: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not direct_cpu_route_rows_enabled():
        return None
    rows = int(rows)
    if rows <= 0 or not active:
        return None
    cpu_start = _breakdown_cpu_start()
    residual_rows: list[int] = []
    residual_seen: set[int] = set()
    score_cache: dict[int, torch.Tensor] = {}
    cutoff = float(cutoff)
    min_prefix = max(0, int(min_prefix_residual))
    extra = max(0, int(extra_after_low))
    early = max(0, int(early_tokens))
    generated_len_sentinel = 1_000_000_000
    for active_idx, (_req_idx, start, valid, score_len, _n, has_bonus, scores) in enumerate(active):
        start = int(start)
        valid = max(0, int(valid))
        if valid <= 0:
            continue
        generated_len = (
            int(active_generated_lens[active_idx])
            if active_idx < len(active_generated_lens)
            else generated_len_sentinel
        )
        score_values: list[float] | None = None
        if scores is not None and int(score_len) >= valid:
            score_cpu = _score_tensor_cpu_view(scores, score_cache)
            score_values = score_cpu[:valid].tolist()
        if score_values is None:
            use_residual = [True] * valid
        else:
            use_residual = [False] * valid
            low_seen = 0
            first_low = valid
            for pos in range(valid):
                score = float(score_values[pos])
                present = not math.isnan(score)
                low = present and score <= cutoff
                if low:
                    low_seen += 1
                    first_low = min(first_low, pos)
                use_residual[pos] = (low_seen == 0) or (low and low_seen == 1)
            if first_low < valid and extra > 0:
                last_extra = min(valid - 1, first_low + extra)
                for pos in range(last_extra + 1):
                    use_residual[pos] = True
        if min_prefix > 0:
            for pos in range(min(min_prefix, valid)):
                use_residual[pos] = True
        if early > 0 and generated_len != generated_len_sentinel:
            early_count = min(valid, max(0, early - generated_len))
            for pos in range(early_count):
                use_residual[pos] = True
        for pos, selected in enumerate(use_residual):
            if not selected:
                continue
            row = start + pos
            if 0 <= row < rows and row not in residual_seen:
                residual_seen.add(row)
                residual_rows.append(row)
        bonus_early = (
            early > 0
            and generated_len != generated_len_sentinel
            and generated_len + valid < early
        )
        if has_bonus and (
            bonus_early
            or _direct_cpu_bonus_uses_residual(
                non_draft_policy=non_draft_policy,
                scores=scores,
                score_len=score_len,
                valid=valid,
                cutoff=cutoff,
                score_cache=score_cache,
            )
        ):
            bonus_row = start + valid
            if 0 <= bonus_row < rows and bonus_row not in residual_seen:
                residual_seen.add(bonus_row)
                residual_rows.append(bonus_row)
    if residual_rows:
        residual_tensor = _static_long_view(
            "direct_cpu_route_residual_rows",
            len(residual_rows),
            device,
        )
        _copy_long_values_to_device(
            name="direct_cpu_route_residual_rows",
            dst=residual_tensor,
            values=residual_rows,
        )
    else:
        residual_tensor = torch.empty(0, dtype=torch.long, device=device)
    base_rows = [row for row in range(rows) if row not in residual_seen]
    if base_rows:
        base_tensor = _static_long_view(
            "direct_cpu_route_base_rows",
            len(base_rows),
            device,
        )
        _copy_long_values_to_device(
            name="direct_cpu_route_base_rows",
            dst=base_tensor,
            values=base_rows,
        )
    else:
        base_tensor = torch.empty(0, dtype=torch.long, device=device)
    _breakdown_record_cpu("scheduler_direct_cpu_route_rows_cpu_ms", cpu_start)
    _breakdown_count("scheduler_direct_cpu_critical_prefix_route_rows_builds", 1)
    _breakdown_count("scheduler_direct_cpu_route_rows_total_rows", rows)
    _breakdown_count("scheduler_direct_cpu_route_rows_residual", len(residual_rows))
    _breakdown_count("scheduler_direct_cpu_route_rows_base", len(base_rows))
    return residual_tensor, base_tensor


def _try_build_critical_prefix_bonus_mask_batched(
    *,
    residual_mask: torch.Tensor,
    residual_priority: torch.Tensor | None = None,
    scores_by_req: list[torch.Tensor | None],
    generated_lens_by_req: list[int | None] | None = None,
    draft_counts: list[int],
    scheduled_counts: list[int],
    cu_scheduled_counts: list[int],
    num_draft_tokens_gpu: torch.Tensor | None = None,
    num_scheduled_tokens_gpu: torch.Tensor | None = None,
    cu_num_scheduled_tokens_gpu: torch.Tensor | None = None,
    total_num_scheduled_tokens: int,
    device: torch.device,
    cutoff: float,
    prefix_cutoff: float,
    extra_after_low: int,
    min_prefix_residual: int,
    max_residual_draft_rows: int,
    early_tokens: int = 0,
    residual_policy: str,
    non_draft_policy: str,
    track_basic_stats: bool,
    build_direct_route_rows: bool = False,
    build_fixed_prefix_all_route_rows: bool = False,
) -> tuple[
    bool,
    dict[str, int],
    tuple[torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor] | None,
    FixedPrefixRouteDescriptor | None,
]:
    if not batched_mask_builder_enabled():
        return False, {}, None, None, None
    if residual_policy == "high_confidence":
        policy_id = 1
    elif residual_policy == "all_if_any_low":
        policy_id = 2
    elif residual_policy == "low_confidence":
        policy_id = 3
    elif residual_policy == "prefix_confidence":
        policy_id = 4
    elif residual_policy == "fixed_prefix":
        policy_id = 5
    else:
        policy_id = 0
    setup_cpu_start = _breakdown_cpu_start()
    priority_arg = residual_priority if residual_priority is not None else residual_mask
    write_priority = residual_priority is not None
    bonus_priority_value = float(bonus_priority())
    bonus_policy_id = 2 if non_draft_policy == "predicted_full_accept" else 1
    position_priority_scale = float(draft_position_priority_scale())
    active: list[tuple[int, int, int, int, int, bool, torch.Tensor | None]] = []
    generated_len_sentinel = 1_000_000_000
    req_generated_len_values = [generated_len_sentinel] * len(draft_counts)
    if early_tokens > 0 and generated_lens_by_req is not None:
        for req_idx, generated_len in enumerate(
            generated_lens_by_req[: len(draft_counts)]
        ):
            if generated_len is not None:
                req_generated_len_values[req_idx] = max(0, int(generated_len))
    active_generated_len_values: list[int] = []
    total_draft_tokens = 0
    total_valid_draft_tokens = 0
    bonus_residual_tokens = 0
    forced_residual_draft_tokens = 0
    forced_residual_non_draft_tokens = 0
    forced_early_draft_tokens = 0
    forced_early_non_draft_tokens = 0
    max_valid = 0
    for req_idx, n in enumerate(draft_counts):
        n = int(n)
        if n <= 0:
            continue
        if track_basic_stats:
            total_draft_tokens += n
        end = int(cu_scheduled_counts[req_idx])
        sched = int(scheduled_counts[req_idx])
        start = end - sched
        valid = max(0, min(n, int(total_num_scheduled_tokens) - start))
        if valid <= 0:
            continue
        if track_basic_stats:
            total_valid_draft_tokens += valid
        bonus_row = start + valid
        has_bonus_row = bonus_row < min(end, int(total_num_scheduled_tokens))
        if has_bonus_row:
            bonus_residual_tokens += 1
            if non_draft_policy == "bonus":
                forced_residual_non_draft_tokens += 1
        forced_prefix = min(valid, max(0, int(min_prefix_residual)))
        forced_early = 0
        if early_tokens > 0:
            generated_len = req_generated_len_values[req_idx]
            if generated_len != generated_len_sentinel:
                forced_early = min(valid, max(0, int(early_tokens) - generated_len))
                forced_early_draft_tokens += forced_early
                if has_bonus_row and generated_len + valid < int(early_tokens):
                    forced_early_non_draft_tokens += 1
        forced_residual_draft_tokens += max(forced_prefix, forced_early)
        scores = scores_by_req[req_idx]
        score_len = 0
        if scores is not None:
            score_len = min(int(scores.numel()), valid)
        active.append((req_idx, start, valid, score_len, n, has_bonus_row, scores))
        active_generated_len_values.append(req_generated_len_values[req_idx])
        max_valid = max(max_valid, valid)
    if not active or max_valid <= 0:
        return False, {}, None, None, None
    active_count = len(active)

    def _success_result() -> tuple[
        bool,
        dict[str, int],
        tuple[torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor, torch.Tensor] | None,
        FixedPrefixRouteDescriptor | None,
    ]:
        direct_bucket = None
        direct_route_rows = None
        fixed_prefix_route = None
        if build_direct_route_rows and residual_policy == "prefix_confidence":
            direct_route_rows = _build_direct_cpu_prefix_confidence_route_rows(
                active=active,
                rows=total_num_scheduled_tokens,
                device=device,
                prefix_cutoff=prefix_cutoff,
                min_prefix_residual=min_prefix_residual,
                non_draft_policy=non_draft_policy,
                cutoff=cutoff,
            )
        elif (
            build_fixed_prefix_all_route_rows
            and residual_policy == "fixed_prefix"
            and non_draft_policy in {"all", "bonus"}
        ):
            direct_route_rows = _build_fixed_prefix_all_route_rows_fastpath(
                active=active,
                rows=total_num_scheduled_tokens,
                device=device,
                min_prefix_residual=min_prefix_residual,
                non_draft_policy=non_draft_policy,
            )
            if direct_route_rows is not None:
                residual_rows, base_rows, fixed_prefix_route = direct_route_rows
            direct_route_rows = (
                (residual_rows, base_rows)
                if residual_rows is not None and base_rows is not None
                else None
            )
        elif build_direct_route_rows and residual_policy == "fixed_prefix":
            direct_route_rows = _build_direct_cpu_fixed_prefix_route_rows(
                active=active,
                rows=total_num_scheduled_tokens,
                device=device,
                min_prefix_residual=min_prefix_residual,
                non_draft_policy=non_draft_policy,
                cutoff=cutoff,
            )
        elif build_direct_route_rows and residual_policy == "critical_prefix":
            direct_route_rows = _build_direct_cpu_critical_prefix_route_rows(
                active=active,
                active_generated_lens=active_generated_len_values,
                rows=total_num_scheduled_tokens,
                device=device,
                cutoff=cutoff,
                extra_after_low=extra_after_low,
                min_prefix_residual=min_prefix_residual,
                early_tokens=early_tokens,
                non_draft_policy=non_draft_policy,
            )
        if (
            fixed_prefix_route is not None
            and fixed_prefix_route_descriptor_only()
            and row_routed_mlp()
            and route_all_residual_rows()
        ):
            _breakdown_count("scheduler_fixed_prefix_route_descriptor_only_skip_bucket", 1)
        else:
            direct_bucket = _build_direct_position_bucket_from_active(
                active=active,
                rows=total_num_scheduled_tokens,
                device=device,
                value_dtype=torch.float32,
            )
        return True, {
            "total_draft_tokens": total_draft_tokens,
            "total_valid_draft_tokens": total_valid_draft_tokens,
            "bonus_residual_tokens": bonus_residual_tokens,
            "forced_residual_draft_tokens": forced_residual_draft_tokens,
            "forced_residual_non_draft_tokens": forced_residual_non_draft_tokens,
            "forced_early_draft_tokens": forced_early_draft_tokens,
            "forced_early_non_draft_tokens": forced_early_non_draft_tokens,
        }, direct_bucket, direct_route_rows, fixed_prefix_route

    cap_by_risk = low_confidence_cap_by_risk()
    start_values: list[int] = []
    valid_values: list[int] = []
    score_len_values: list[int] = []
    has_bonus_values: list[int] = []
    score_row_values: list[int] = []
    indexed_score_base: torch.Tensor | None = None
    indexed_scores_possible = True
    for active_idx, (_, start, valid, score_len, _, has_bonus_row, scores) in enumerate(
        active
    ):
        start_values.append(start)
        valid_values.append(valid)
        score_len_values.append(score_len)
        has_bonus_values.append(1 if has_bonus_row else 0)
        if score_len <= 0 or scores is None:
            score_row_values.append(-1)
            continue
        score_base = getattr(scores, "_speclink_sr24_score_base", None)
        score_row = getattr(scores, "_speclink_sr24_score_row", None)
        if (
            not isinstance(score_base, torch.Tensor)
            or score_base.ndim != 2
            or score_base.device != device
            or score_base.dtype != torch.float32
            or int(score_base.shape[1]) < score_len
            or not isinstance(score_row, int)
        ):
            indexed_scores_possible = False
        elif indexed_score_base is None:
            indexed_score_base = score_base
            score_row_values.append(int(score_row))
        elif (
            score_base.untyped_storage().data_ptr()
            == indexed_score_base.untyped_storage().data_ptr()
            and tuple(score_base.shape) == tuple(indexed_score_base.shape)
            and tuple(score_base.stride()) == tuple(indexed_score_base.stride())
        ):
            score_row_values.append(int(score_row))
        else:
            indexed_scores_possible = False
    tensor_setup_timer = _breakdown_cuda_start()
    block_k = _next_power_of_2(max_valid)
    generated_lens_req = _static_int32_view(
        "batched_generated_lens_req",
        max(1, len(draft_counts)),
        device,
    )
    generated_lens_active = _static_int32_view(
        "batched_generated_lens_active",
        max(1, active_count),
        device,
    )
    if early_tokens > 0:
        _copy_int32_values_to_device(
            name="batched_generated_lens_req",
            dst=generated_lens_req[: len(draft_counts)],
            values=req_generated_len_values,
        )
        _copy_int32_values_to_device(
            name="batched_generated_lens_active",
            dst=generated_lens_active[:active_count],
            values=active_generated_len_values,
        )
    uniform_direct_possible = (
        batched_uniform_direct_enabled()
        and not gpu_count_mask_builder_enabled()
        and indexed_scores_possible
        and indexed_score_base is not None
        and active_count == len(draft_counts)
        and max_valid > 0
    )
    if uniform_direct_possible:
        scheduled_stride = max_valid + 1
        expected_total = len(draft_counts) * scheduled_stride
        if int(total_num_scheduled_tokens) != expected_total:
            uniform_direct_possible = False
        else:
            for active_idx, (
                req_idx,
                start,
                valid,
                score_len,
                n,
                has_bonus_row,
                _scores,
            ) in enumerate(active):
                if (
                    req_idx != active_idx
                    or int(n) != max_valid
                    or int(valid) != max_valid
                    or int(score_len) != max_valid
                    or not has_bonus_row
                    or int(scheduled_counts[req_idx]) != scheduled_stride
                    or int(cu_scheduled_counts[req_idx])
                    != (req_idx + 1) * scheduled_stride
                    or int(start) != req_idx * scheduled_stride
                    or int(score_row_values[active_idx]) != req_idx
                ):
                    uniform_direct_possible = False
                    break
    if uniform_direct_possible and indexed_score_base is not None:
        _breakdown_record_cuda(
            "scheduler_batched_mask_tensor_setup_cuda_ms",
            tensor_setup_timer,
        )
        _breakdown_record_cpu(
            "scheduler_batched_mask_setup_cpu_ms",
            setup_cpu_start,
        )
        timer = _breakdown_cuda_start()
        _critical_prefix_bonus_mask_uniform_direct_kernel[(len(draft_counts),)](
            indexed_score_base,
            generated_lens_req,
            residual_mask,
            priority_arg,
            bonus_priority_value,
            position_priority_scale,
            int(bonus_policy_id),
            float(cutoff),
            float(prefix_cutoff),
            int(extra_after_low),
            int(min_prefix_residual),
            int(policy_id),
            int(max_residual_draft_rows),
            bool(cap_by_risk),
            int(early_tokens),
            int(max_valid),
            int(max_valid + 1),
            int(indexed_score_base.stride(0)),
            bool(write_priority),
            block_k,
        )
        _breakdown_record_cuda(
            "scheduler_batched_mask_uniform_direct_kernel_cuda_ms",
            timer,
        )
        _breakdown_count("batched_mask_builder_uniform_direct_steps", 1)
        _breakdown_count("batched_mask_builder_steps", 1)
        _breakdown_count("batched_mask_builder_active_requests", active_count)
        return _success_result()
    gpu_counts_available = (
        gpu_count_mask_builder_enabled()
        and indexed_scores_possible
        and indexed_score_base is not None
        and isinstance(num_draft_tokens_gpu, torch.Tensor)
        and isinstance(num_scheduled_tokens_gpu, torch.Tensor)
        and isinstance(cu_num_scheduled_tokens_gpu, torch.Tensor)
        and num_draft_tokens_gpu.device == device
        and num_scheduled_tokens_gpu.device == device
        and cu_num_scheduled_tokens_gpu.device == device
        and int(num_draft_tokens_gpu.numel()) >= len(draft_counts)
        and int(num_scheduled_tokens_gpu.numel()) >= len(draft_counts)
        and int(cu_num_scheduled_tokens_gpu.numel()) >= len(draft_counts)
    )
    if gpu_counts_available and indexed_score_base is not None:
        direct_score_rows = all(
            score_len > 0 and score_row_values[active_idx] == req_idx
            for active_idx, (req_idx, _, _, score_len, _, _, _) in enumerate(active)
        )
        _breakdown_record_cuda(
            "scheduler_batched_mask_tensor_setup_cuda_ms",
            tensor_setup_timer,
        )
        if direct_score_rows:
            _breakdown_record_cpu("scheduler_batched_mask_setup_cpu_ms", setup_cpu_start)
            timer = _breakdown_cuda_start()
            _critical_prefix_bonus_mask_req_kernel[(len(draft_counts),)](
                indexed_score_base,
                generated_lens_req,
                num_draft_tokens_gpu,
                num_scheduled_tokens_gpu,
                cu_num_scheduled_tokens_gpu,
                residual_mask,
                priority_arg,
                bonus_priority_value,
                position_priority_scale,
                int(bonus_policy_id),
                int(total_num_scheduled_tokens),
                float(cutoff),
                float(prefix_cutoff),
                int(extra_after_low),
                int(min_prefix_residual),
                int(policy_id),
                int(max_residual_draft_rows),
                bool(cap_by_risk),
                int(early_tokens),
                int(indexed_score_base.shape[1]),
                int(indexed_score_base.stride(0)),
                bool(write_priority),
                block_k,
            )
            _breakdown_record_cuda(
                "scheduler_batched_mask_req_kernel_cuda_ms",
                timer,
            )
            _breakdown_count("batched_mask_builder_gpu_count_steps", 1)
            _breakdown_count("batched_mask_builder_direct_score_rows_steps", 1)
            _breakdown_count("batched_mask_builder_steps", 1)
            _breakdown_count("batched_mask_builder_active_requests", active_count)
            return _success_result()

        # The request-indexed GPU-count kernel was measured much slower than
        # the compact indexed kernel in bs64/K8 serving diagnostics. If score
        # rows are not direct request rows, keep the old compact path below
        # instead of launching the full-request indexed kernel.
        _breakdown_count("batched_mask_builder_gpu_count_indexed_fallback_steps", 1)

    starts = _static_int32_view("batched_starts", active_count, device)
    valid_rows = _static_int32_view("batched_valid_rows", active_count, device)
    score_lens = _static_int32_view("batched_score_lens", active_count, device)
    has_bonus = _static_int32_view("batched_has_bonus", active_count, device)
    _copy_int32_values_to_device(
        name="batched_starts",
        dst=starts,
        values=start_values,
    )
    _copy_int32_values_to_device(
        name="batched_valid_rows",
        dst=valid_rows,
        values=valid_values,
    )
    _copy_int32_values_to_device(
        name="batched_score_lens",
        dst=score_lens,
        values=score_len_values,
    )
    _copy_int32_values_to_device(
        name="batched_has_bonus",
        dst=has_bonus,
        values=has_bonus_values,
    )
    _breakdown_record_cuda("scheduler_batched_mask_tensor_setup_cuda_ms", tensor_setup_timer)
    if indexed_scores_possible and indexed_score_base is not None:
        score_row_setup_timer = _breakdown_cuda_start()
        score_rows = _static_int32_view("batched_score_rows", active_count, device)
        _copy_int32_values_to_device(
            name="batched_score_rows",
            dst=score_rows,
            values=score_row_values,
        )
        _breakdown_record_cuda(
            "scheduler_batched_mask_score_rows_setup_cuda_ms",
            score_row_setup_timer,
        )
        _breakdown_record_cpu("scheduler_batched_mask_setup_cpu_ms", setup_cpu_start)
        timer = _breakdown_cuda_start()
        _critical_prefix_bonus_mask_indexed_kernel[(active_count,)](
            indexed_score_base,
            score_rows,
            generated_lens_active,
            starts,
            valid_rows,
            score_lens,
            has_bonus,
            residual_mask,
            priority_arg,
            bonus_priority_value,
            position_priority_scale,
            int(bonus_policy_id),
            float(cutoff),
            float(prefix_cutoff),
            int(extra_after_low),
            int(min_prefix_residual),
            int(policy_id),
            int(max_residual_draft_rows),
            bool(cap_by_risk),
            int(early_tokens),
            int(indexed_score_base.stride(0)),
            bool(write_priority),
            block_k,
        )
        _breakdown_record_cuda("scheduler_batched_mask_indexed_kernel_cuda_ms", timer)
        _breakdown_count("batched_mask_builder_indexed_steps", 1)
        _breakdown_count("batched_mask_builder_steps", 1)
        _breakdown_count("batched_mask_builder_active_requests", active_count)
        return _success_result()

    score_matrix_setup_timer = _breakdown_cuda_start()
    score_matrix = torch.empty(
        (active_count, max_valid),
        dtype=torch.float32,
        device=device,
    )
    score_matrix.fill_(float("nan"))
    for active_idx, (_, _, _, score_len, _, _, scores) in enumerate(active):
        if score_len > 0 and scores is not None:
            score_matrix[active_idx, :score_len].copy_(
                scores[:score_len].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
    _breakdown_record_cuda(
        "scheduler_batched_mask_score_matrix_setup_cuda_ms",
        score_matrix_setup_timer,
    )
    _breakdown_record_cpu("scheduler_batched_mask_setup_cpu_ms", setup_cpu_start)
    timer = _breakdown_cuda_start()
    _critical_prefix_bonus_mask_kernel[(active_count,)](
        score_matrix,
        generated_lens_active,
        starts,
        valid_rows,
        score_lens,
        has_bonus,
        residual_mask,
        priority_arg,
        bonus_priority_value,
        position_priority_scale,
        int(bonus_policy_id),
        float(cutoff),
        float(prefix_cutoff),
        int(extra_after_low),
        int(min_prefix_residual),
        int(policy_id),
        int(max_residual_draft_rows),
        bool(cap_by_risk),
        int(early_tokens),
        int(max_valid),
        bool(write_priority),
        block_k,
    )
    _breakdown_record_cuda("scheduler_batched_mask_kernel_cuda_ms", timer)
    _breakdown_count("batched_mask_builder_steps", 1)
    _breakdown_count("batched_mask_builder_active_requests", active_count)
    return _success_result()


def _finalize_verify_plan_breakdown(
    plan: VerifyResidualPlan,
    *,
    cpu_start: float,
    total_num_scheduled_tokens: int,
    total_draft_tokens: int | None = None,
    total_valid_draft_tokens: int | None = None,
    non_draft_tokens: int | None = None,
    residual_draft_tokens: int | None = None,
    residual_non_draft_tokens: int | None = None,
    base_only_draft_tokens: int | None = None,
    base_only_non_draft_tokens: int | None = None,
    early_residual_draft_tokens: int | None = None,
    early_residual_non_draft_tokens: int | None = None,
    missing_score_tokens: int | None = None,
    stats_exact: bool | None = None,
) -> VerifyResidualPlan:
    if not breakdown_enabled():
        return plan
    _breakdown_record_cpu("scheduler_mask_build_cpu_ms", cpu_start)
    _breakdown_count("verify_steps", 1)
    _breakdown_count("scheduled_tokens", total_num_scheduled_tokens)
    if total_draft_tokens is not None:
        _breakdown_count("draft_tokens", total_draft_tokens)
    if total_valid_draft_tokens is not None:
        _breakdown_count("valid_draft_tokens", total_valid_draft_tokens)
    if non_draft_tokens is not None:
        _breakdown_count("non_draft_tokens", non_draft_tokens)
    if residual_draft_tokens is not None:
        _breakdown_count("residual_draft_tokens", residual_draft_tokens)
    if residual_non_draft_tokens is not None:
        _breakdown_count("residual_non_draft_tokens", residual_non_draft_tokens)
    if base_only_draft_tokens is not None:
        _breakdown_count("base_only_draft_tokens", base_only_draft_tokens)
    if base_only_non_draft_tokens is not None:
        _breakdown_count("base_only_non_draft_tokens", base_only_non_draft_tokens)
    if early_residual_draft_tokens is not None:
        _breakdown_count("early_residual_draft_tokens", early_residual_draft_tokens)
    if early_residual_non_draft_tokens is not None:
        _breakdown_count(
            "early_residual_non_draft_tokens", early_residual_non_draft_tokens
        )
    if missing_score_tokens is not None:
        _breakdown_count("missing_score_tokens", missing_score_tokens)
    if stats_exact is not None:
        _breakdown_count("verify_steps_exact" if stats_exact else "verify_steps_approx", 1)
    _breakdown_count(f"mask_state_{plan.state}", 1)
    if plan.mask is not None:
        _breakdown_count("mixed_mask_rows", int(plan.mask.numel()))
    if plan.bucket is not None:
        bucket_rows, bucket_values = plan.bucket
        _breakdown_count_bucket(
            rows=total_num_scheduled_tokens,
            bucket_rows=bucket_rows,
            bucket_values=bucket_values,
        )
    if plan.residual_rows is not None:
        _breakdown_count("cached_residual_rows_available_steps", 1)
        _breakdown_count(
            "cached_residual_rows_available",
            int(plan.residual_rows.numel()),
        )
    if plan.base_rows is not None:
        _breakdown_count("cached_base_rows_available_steps", 1)
        _breakdown_count("cached_base_rows_available", int(plan.base_rows.numel()))
    _flush_breakdown(force=False)
    return plan


def build_verify_residual_mask(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    cu_num_scheduled_tokens: Any,
    total_num_scheduled_tokens: int,
    device: torch.device,
    num_scheduled_tokens_gpu: torch.Tensor | None = None,
    num_draft_tokens_gpu: torch.Tensor | None = None,
    cu_num_scheduled_tokens_gpu: torch.Tensor | None = None,
) -> VerifyResidualPlan | None:
    if not enabled() or total_num_scheduled_tokens <= 0:
        return None
    if not linear_hooks_enabled():
        return None
    breakdown_cpu_start = _breakdown_cpu_start()
    runtime_timing = runtime_timing_enabled()
    runtime_wall_start = time.perf_counter() if runtime_timing else 0.0
    runtime_timings: dict[str, float] = {}

    def _runtime_wall_start() -> float:
        return time.perf_counter() if runtime_timing else 0.0

    def _runtime_wall_record(name: str, start: float) -> None:
        if start:
            runtime_timings[name] = (time.perf_counter() - start) * 1000.0

    def _attach_runtime_timings(record: dict[str, Any]) -> None:
        if not runtime_timing:
            return
        record.update(runtime_timings)
        record["scheduler_mask_wall_cpu_ms"] = (
            time.perf_counter() - runtime_wall_start
        ) * 1000.0

    sr_mode = mode()
    correct_non_draft = selective_correct_non_draft()
    non_draft_policy = selective_non_draft_policy()
    residual_policy = selective_residual_policy()
    prefix_cutoff = selective_prefix_threshold()
    extra_after_low = selective_extra_after_low()
    min_prefix_residual = selective_min_prefix_residual()
    max_residual_draft_rows = selective_max_residual_draft_rows()
    early_tokens = early_dense_tokens()
    static_state = static_mask_state()
    cutoff = threshold()
    if residual_policy not in {
        "critical_prefix",
        "all_if_any_low",
        "batch_all_if_any_low",
        "low_confidence",
        "high_confidence",
        "prefix_confidence",
        "fixed_prefix",
    }:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY="
            f"{residual_policy}"
        )
    if non_draft_policy not in {
        "all",
        "none",
        "bonus",
        "predicted_full_accept",
    }:
        raise RuntimeError(
            "unsupported SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY="
            f"{non_draft_policy}"
        )
    policy_forces_all_residual = _selective_policy_forces_all_residual(
        sr_mode=sr_mode,
        correct_non_draft=correct_non_draft,
        non_draft_policy=non_draft_policy,
        residual_policy=residual_policy,
        cutoff=cutoff,
        max_residual_draft_rows=max_residual_draft_rows,
    )
    if sr_mode == "base_only" and not runtime_stats_enabled() and not breakdown_enabled():
        # Base-only SR24 never needs per-request draft scores or a residual
        # mask: every verifier row in the selected modules uses W_base. Skip
        # scheduler count materialization and mask allocation in stats-off
        # throughput runs.
        return VerifyResidualPlan(mask=None, state="no_residual")
    if sr_mode == "all_corrected" and not runtime_stats_enabled() and not breakdown_enabled():
        # All-corrected SR24 is logically all-residual for every scheduled row.
        # Do not allocate a mask or scan request scores just to rediscover that
        # fact; the Linear hook can use the dense/all-residual fast path from
        # the plan state alone.
        return VerifyResidualPlan(mask=None, state="all_residual")
    if static_state in {"all_residual", "no_residual"} and not runtime_stats_enabled():
        # Static all/no-residual states do not need DLM scores or per-step
        # accounting. Skip the request scan entirely in stats-off throughput
        # runs; the returned state is the only information used by SR24 Linear
        # hooks.
        return _finalize_verify_plan_breakdown(
            VerifyResidualPlan(mask=None, state=static_state),
            cpu_start=breakdown_cpu_start,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            stats_exact=False,
        )
    if (
        policy_forces_all_residual
        and not runtime_stats_enabled()
        and not breakdown_enabled()
    ):
        return _finalize_verify_plan_breakdown(
            VerifyResidualPlan(mask=None, state="all_residual"),
            cpu_start=breakdown_cpu_start,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            stats_exact=False,
        )
    materialize_counts_start = _breakdown_cpu_start()
    materialize_counts_wall_start = _runtime_wall_start()
    draft_counts = [int(num_draft_tokens[idx]) for idx in range(len(req_ids))]
    scheduled_counts = [
        int(num_scheduled_tokens[idx]) for idx in range(len(req_ids))
    ]
    cu_scheduled_counts = [
        int(cu_num_scheduled_tokens[idx]) for idx in range(len(req_ids))
    ]
    compact_width: int | None = None
    compact_spec_batch = False
    if sr_mode == "selective" and any(n > 0 for n in draft_counts):
        compact_spec_batch = True
        for draft_count, scheduled_count in zip(draft_counts, scheduled_counts):
            draft_count = int(draft_count)
            scheduled_count = int(scheduled_count)
            if draft_count <= 0 or scheduled_count != draft_count + 1:
                compact_spec_batch = False
                break
            if compact_width is None:
                compact_width = scheduled_count
            elif scheduled_count != compact_width:
                compact_spec_batch = False
                break
        if not compact_spec_batch:
            # A mixed step can contain prompt chunks, normal decode rows, and
            # speculative verifier rows together.  The fixed-prefix row-routed
            # path assumes a compact K+1 verifier block per request; applying
            # the sparse base to the rest of a mixed step is both inaccurate
            # and slower for long prompt chunks.  Keep the entire non-compact
            # step dense and reserve SR24 routing for uniform verifier blocks.
            _breakdown_count("scheduler_selective_noncompact_spec_all_residual", 1)
            _trace_grouping_opportunity(
                reason="noncompact_spec_all_residual",
                sr_mode=sr_mode,
                residual_policy=residual_policy,
                non_draft_policy=non_draft_policy,
                mask_state="all_residual",
                compact_spec_batch=False,
                compact_width=compact_width,
                draft_counts=draft_counts,
                scheduled_counts=scheduled_counts,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                min_prefix_residual=min_prefix_residual,
                fixed_prefix_route=None,
                batched_mask_applied=False,
            )
            return _finalize_verify_plan_breakdown(
                VerifyResidualPlan(mask=None, state="all_residual"),
                cpu_start=breakdown_cpu_start,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                total_draft_tokens=sum(max(0, int(n)) for n in draft_counts),
                stats_exact=False,
            )
    nonuniform_dense_fallback = (
        sr_mode == "selective"
        and dense_fallback_nonuniform()
        and len(scheduled_counts) > 1
        and min(scheduled_counts) != max(scheduled_counts)
    )
    _breakdown_record_cpu(
        "scheduler_materialize_counts_cpu_ms",
        materialize_counts_start,
    )
    _runtime_wall_record(
        "scheduler_materialize_counts_wall_cpu_ms",
        materialize_counts_wall_start,
    )
    if (
        sr_mode == "all_corrected"
        or static_state in {"all_residual", "no_residual"}
        or policy_forces_all_residual
        or nonuniform_dense_fallback
    ):
        if nonuniform_dense_fallback:
            _breakdown_count("scheduler_dense_fallback_nonuniform_steps", 1)
        total_draft_tokens = 0
        total_valid_draft_tokens = 0
        missing_score_tokens = 0
        with _lock:
            for req_idx, req_id in enumerate(req_ids):
                n = draft_counts[req_idx]
                if n <= 0:
                    continue
                total_draft_tokens += n
                pending = _pending_scores.get(req_id)
                if pending:
                    pending.popleft()
                if early_tokens > 0:
                    pending_lens = _pending_generated_lens.get(req_id)
                    if pending_lens:
                        pending_lens.popleft()
                end = cu_scheduled_counts[req_idx]
                sched = scheduled_counts[req_idx]
                start = end - sched
                valid_rows = max(
                    0, min(n, total_num_scheduled_tokens - start)
                )
                invalid_rows = n - valid_rows
                total_valid_draft_tokens += valid_rows
                missing_score_tokens += max(0, invalid_rows)
        non_draft_tokens = max(
            0, total_num_scheduled_tokens - total_valid_draft_tokens
        )
        plan_state = (
            "all_residual"
            if (
                sr_mode == "all_corrected"
                or policy_forces_all_residual
                or nonuniform_dense_fallback
            )
            else static_state
        )
        residual_all = plan_state == "all_residual"
        if runtime_stats_enabled():
            record = {
                "timestamp": time.time(),
                "event": "sr24_verify_mask",
                "mode": sr_mode,
                "threshold": threshold(),
                "prefix_threshold": prefix_cutoff,
                "stats_interval": _stats_interval(),
                "selective_correct_non_draft": correct_non_draft,
                "selective_non_draft_policy": non_draft_policy,
                "selective_residual_policy": residual_policy,
                "selective_extra_after_low": extra_after_low,
                "selective_min_prefix_residual": min_prefix_residual,
                "selective_max_residual_draft_rows":
                max_residual_draft_rows,
                "early_dense_tokens": early_tokens,
                "dense_fallback_nonuniform": nonuniform_dense_fallback,
                "sync_reduced_stats": reduce_cpu_sync(),
                "sync_mask_state": sync_mask_state(),
                "static_mask_state": static_state,
                "stats_exact": True,
                "mask_state": plan_state,
                "mask_state_exact": True,
                "request_count": len(req_ids),
                "total_scheduled_tokens": total_num_scheduled_tokens,
                "total_draft_tokens": total_draft_tokens,
                "total_valid_draft_tokens": total_valid_draft_tokens,
                "non_draft_tokens": non_draft_tokens,
                "residual_draft_tokens": (
                    total_valid_draft_tokens if residual_all else 0
                ),
                "base_only_draft_tokens": (
                    0 if residual_all else total_valid_draft_tokens
                ),
                "residual_non_draft_tokens": non_draft_tokens if residual_all else 0,
                "base_only_non_draft_tokens": (
                    0 if residual_all else non_draft_tokens
                ),
                "early_residual_draft_tokens": 0,
                "early_residual_non_draft_tokens": 0,
                "missing_score_tokens": missing_score_tokens,
                "residual_draft_fraction": (
                    total_valid_draft_tokens / total_draft_tokens
                    if residual_all and total_draft_tokens else 0.0
                ),
            }
            _attach_runtime_timings(record)
            summary_record = _accumulate_stats(record)
            _write_verify_record(record, summary_record)
        return _finalize_verify_plan_breakdown(
            VerifyResidualPlan(mask=None, state=plan_state),
            cpu_start=breakdown_cpu_start,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            total_draft_tokens=total_draft_tokens,
            total_valid_draft_tokens=total_valid_draft_tokens,
            non_draft_tokens=non_draft_tokens,
            residual_draft_tokens=total_valid_draft_tokens if residual_all else 0,
            base_only_draft_tokens=0 if residual_all else total_valid_draft_tokens,
            residual_non_draft_tokens=non_draft_tokens if residual_all else 0,
            base_only_non_draft_tokens=0 if residual_all else non_draft_tokens,
            missing_score_tokens=missing_score_tokens,
            stats_exact=True,
        )
    selective_correct_all_non_draft = (
        sr_mode == "selective" and non_draft_policy == "all"
    )
    mask_init_cpu_start = _breakdown_cpu_start()
    mask_init_cuda_start = _breakdown_cuda_start()
    if sr_mode == "base_only" or (
        sr_mode == "selective" and not selective_correct_all_non_draft
    ):
        residual_mask = torch.zeros(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device=device,
        )
        residual_priority = (
            torch.zeros(
                total_num_scheduled_tokens,
                dtype=torch.float32,
                device=device,
            )
            if residual_bucket_priority() and residual_bucket_size() > 0
            else None
        )
    else:
        residual_mask = torch.ones(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device=device,
        )
        residual_priority = (
            torch.full(
                (total_num_scheduled_tokens,),
                4.0,
                dtype=torch.float32,
                device=device,
            )
            if residual_bucket_priority() and residual_bucket_size() > 0
            else None
        )
    _breakdown_record_cuda("scheduler_mask_init_cuda_ms", mask_init_cuda_start)
    _breakdown_record_cpu("scheduler_mask_init_cpu_ms", mask_init_cpu_start)

    sync_reduced = reduce_cpu_sync()
    stats_enabled = runtime_stats_enabled()
    exact_breakdown = breakdown_exact_routing()
    # These counters are diagnostic only. Keep them for runtime stats and
    # breakdown runs, but skip them in stats-off throughput paths so disabling
    # diagnostics also removes the Python integer bookkeeping in the scheduler
    # hot path. The verify plan itself depends on the mask/state/bucket tensors,
    # not on these scalar counters.
    track_basic_stats = stats_enabled or breakdown_enabled()
    track_exact_stats = not sync_reduced or exact_breakdown
    stats_exact = not sync_reduced or exact_breakdown
    total_draft_tokens = 0
    total_valid_draft_tokens = 0
    residual_draft_tokens = 0
    base_only_draft_tokens = 0
    missing_score_tokens = 0
    predicted_residual_non_draft_tokens = 0
    conservative_residual_draft_tokens = 0
    conservative_residual_non_draft_tokens = 0
    early_residual_draft_tokens = 0
    early_residual_non_draft_tokens = 0
    scores_by_req: list[torch.Tensor | None] = [None] * len(req_ids)
    wants_generated_lens = early_tokens > 0 or debug_trace_enabled()
    generated_lens_by_req: list[int | None] = (
        [None] * len(req_ids) if wants_generated_lens else []
    )
    batch_risky: torch.Tensor | None = None
    pending_pop_start = _breakdown_cpu_start()
    pending_pop_wall_start = _runtime_wall_start()
    with _lock:
        for req_idx, req_id in enumerate(req_ids):
            n = draft_counts[req_idx]
            if n <= 0:
                continue
            pending = _pending_scores.get(req_id)
            if pending:
                scores_by_req[req_idx] = pending.popleft()
            if wants_generated_lens:
                pending_lens = _pending_generated_lens.get(req_id)
                if pending_lens:
                    generated_lens_by_req[req_idx] = pending_lens.popleft()
    _breakdown_record_cpu("scheduler_pending_scores_pop_cpu_ms", pending_pop_start)
    _runtime_wall_record(
        "scheduler_pending_scores_pop_wall_cpu_ms",
        pending_pop_wall_start,
    )

    batched_mask_applied = False
    direct_position_bucket: tuple[torch.Tensor, torch.Tensor] | None = None
    direct_cpu_route_rows: tuple[torch.Tensor, torch.Tensor] | None = None
    fixed_prefix_route_descriptor: FixedPrefixRouteDescriptor | None = None
    skip_residual_bucket = False
    if (
        sr_mode == "selective"
        and residual_policy in {
            "critical_prefix",
            "all_if_any_low",
            "low_confidence",
            "high_confidence",
            "prefix_confidence",
            "fixed_prefix",
        }
        and non_draft_policy in {"all", "bonus", "predicted_full_accept"}
        and not track_exact_stats
    ):
        # The batched kernel overwrites draft rows according to the selected
        # confidence policy and writes the speculative bonus row according to
        # the non-draft policy. `bonus` keeps the old always-correct behavior;
        # `predicted_full_accept` corrects the bonus only when every draft
        # score is present and above the same threshold.
        # `early_dense_tokens` is handled by passing per-request generated
        # lengths into the same batched kernel, so the early quality guard does
        # not force a Python per-request routing loop.
        # `low_confidence` matches the slow path by marking missing or
        # low-confidence draft rows residual and high-confidence rows base-only.
        batched_builder_wall_start = _runtime_wall_start()
        direct_route_wall_start = _runtime_wall_start()
        build_fixed_prefix_all_route_rows = (
            fixed_prefix_route_fastpath_enabled()
            and route_all_residual_rows()
            and residual_policy == "fixed_prefix"
            and non_draft_policy in {"all", "bonus"}
            and early_tokens <= 0
            and (
                max_residual_draft_rows <= 0
                or max_residual_draft_rows >= min_prefix_residual
            )
        )
        build_direct_route_rows = (
            direct_cpu_route_rows_enabled()
            and (route_all_residual_rows() or route_reuse_base_output())
            and residual_policy in {
                "critical_prefix",
                "prefix_confidence",
                "fixed_prefix",
            }
            and non_draft_policy in {"bonus", "predicted_full_accept"}
        )
        (
            batched_mask_applied,
            batched_stats,
            direct_position_bucket,
            direct_cpu_route_rows,
            fixed_prefix_route_descriptor,
        ) = (
            _try_build_critical_prefix_bonus_mask_batched(
                residual_mask=residual_mask,
                residual_priority=residual_priority,
                scores_by_req=scores_by_req,
                generated_lens_by_req=(
                    generated_lens_by_req if wants_generated_lens else None
                ),
                draft_counts=draft_counts,
                scheduled_counts=scheduled_counts,
                cu_scheduled_counts=cu_scheduled_counts,
                num_draft_tokens_gpu=num_draft_tokens_gpu,
                num_scheduled_tokens_gpu=num_scheduled_tokens_gpu,
                cu_num_scheduled_tokens_gpu=cu_num_scheduled_tokens_gpu,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                device=device,
                cutoff=cutoff,
                prefix_cutoff=prefix_cutoff,
                extra_after_low=extra_after_low,
                min_prefix_residual=min_prefix_residual,
                max_residual_draft_rows=max_residual_draft_rows,
                early_tokens=early_tokens,
                residual_policy=residual_policy,
                non_draft_policy=non_draft_policy,
                track_basic_stats=track_basic_stats,
                build_direct_route_rows=build_direct_route_rows,
                build_fixed_prefix_all_route_rows=(
                    build_fixed_prefix_all_route_rows
                ),
            )
        )
        if direct_cpu_route_rows is not None:
            _runtime_wall_record(
                "scheduler_direct_cpu_route_rows_wall_cpu_ms",
                direct_route_wall_start,
            )
        _runtime_wall_record(
            "scheduler_batched_mask_builder_wall_cpu_ms",
            batched_builder_wall_start,
        )
        if batched_mask_applied:
            total_draft_tokens = batched_stats["total_draft_tokens"]
            total_valid_draft_tokens = batched_stats["total_valid_draft_tokens"]
            conservative_residual_draft_tokens = int(
                batched_stats.get("forced_residual_draft_tokens") or 0
            )
            conservative_residual_non_draft_tokens = int(
                batched_stats.get("forced_residual_non_draft_tokens") or 0
            )
            early_residual_draft_tokens = int(
                batched_stats.get("forced_early_draft_tokens") or 0
            )
            early_residual_non_draft_tokens = int(
                batched_stats.get("forced_early_non_draft_tokens") or 0
            )
            if non_draft_policy == "bonus":
                predicted_residual_non_draft_tokens = (
                    conservative_residual_non_draft_tokens
                )
            effective_bucket_size = _effective_residual_bucket_size(len(req_ids))
            if residual_policy == "fixed_prefix" and effective_bucket_size > 0:
                fixed_prefix_requested_rows = (
                    conservative_residual_draft_tokens
                    + conservative_residual_non_draft_tokens
                )
                if fixed_prefix_requested_rows > effective_bucket_size:
                    # fixed_prefix is a quality contract: these rows must be
                    # corrected exactly. A capped bucket is only safe when it
                    # covers every requested prefix/bonus row; otherwise fall
                    # back to the exact residual-mask or routed-row path.
                    direct_position_bucket = None
                    skip_residual_bucket = True
                    _breakdown_count("scheduler_fixed_prefix_bucket_overflow_steps", 1)
                    _breakdown_count(
                        "scheduler_fixed_prefix_bucket_overflow_requested_rows",
                        fixed_prefix_requested_rows,
                    )
                    _breakdown_count(
                        "scheduler_fixed_prefix_bucket_overflow_bucket_size",
                        effective_bucket_size,
                    )

    if not batched_mask_applied:
        routing_loop_start = _breakdown_cpu_start()
        routing_loop_wall_start = _runtime_wall_start()
        for req_idx in range(len(req_ids)):
            n = draft_counts[req_idx]
            if n <= 0:
                continue
            if track_basic_stats:
                total_draft_tokens += n
            end = cu_scheduled_counts[req_idx]
            sched = scheduled_counts[req_idx]
            start = end - sched
            scores = scores_by_req[req_idx]
            # vLLM samples logits from model rows [start, start + n].
            # Rows [start, start + n - 1] verify draft tokens 0..n-1, and
            # row start + n is the bonus-token row. Keep the bonus row governed by
            # the non-draft/default mask; do not shift draft scores onto it.
            first_row = start
            valid_rows = max(0, min(n, total_num_scheduled_tokens - first_row))
            invalid_rows = n - valid_rows
            if invalid_rows > 0 and track_exact_stats:
                missing_score_tokens += invalid_rows
                residual_draft_tokens += invalid_rows
            if valid_rows <= 0:
                continue
            if track_basic_stats:
                total_valid_draft_tokens += valid_rows

            row_slice = slice(first_row, first_row + valid_rows)
            bonus_row = first_row + valid_rows
            has_bonus_row = bonus_row < min(end, total_num_scheduled_tokens)
            bonus_use_residual: torch.Tensor | None = None
            score_priority: torch.Tensor | None = None
            generated_len = (
                generated_lens_by_req[req_idx] if early_tokens > 0 else None
            )
            if sr_mode == "base_only":
                residual_mask[row_slice].fill_(False)
                if track_exact_stats:
                    base_only_draft_tokens += valid_rows
            elif sr_mode == "all_corrected":
                residual_mask[row_slice].fill_(True)
                if track_exact_stats:
                    residual_draft_tokens += valid_rows
            else:
                if scores is None or int(scores.numel()) < valid_rows:
                    if residual_policy == "fixed_prefix":
                        int_positions = _device_arange(
                            valid_rows,
                            dtype=torch.int32,
                            device=device,
                        )
                        use_residual = int_positions < min_prefix_residual
                    elif residual_policy == "batch_all_if_any_low":
                        batch_risky = torch.ones((), dtype=torch.bool, device=device)
                        use_residual = torch.zeros(
                            valid_rows,
                            dtype=torch.bool,
                            device=device,
                        )
                    else:
                        use_residual = torch.ones(
                            valid_rows,
                            dtype=torch.bool,
                            device=device,
                        )
                    score_priority = (
                        torch.full(
                            (valid_rows,),
                            3.5,
                            dtype=torch.float32,
                            device=device,
                        )
                        if residual_priority is not None
                        else None
                    )
                    if residual_policy == "fixed_prefix" and score_priority is not None:
                        score_priority = torch.where(
                            use_residual,
                            score_priority,
                            torch.zeros_like(score_priority),
                        )
                    if track_exact_stats and residual_policy != "fixed_prefix":
                        missing_score_tokens += valid_rows
                    if non_draft_policy == "bonus" and has_bonus_row:
                        bonus_use_residual = torch.ones(
                            (), dtype=torch.bool, device=device
                        )
                    elif (
                        non_draft_policy == "predicted_full_accept"
                        and has_bonus_row
                    ):
                        # Missing DLM scores are rare; keep the bonus row exact
                        # rather than risking a base-only sampled token.
                        bonus_use_residual = torch.ones(
                            (), dtype=torch.bool, device=device
                        )
                else:
                    score_policy_cuda_start = _breakdown_cuda_start()
                    score_slice = scores[:valid_rows].to(
                        device=device,
                        non_blocking=True,
                    )
                    score_present = ~torch.isnan(score_slice)
                    positions = (
                        _device_arange(
                            valid_rows,
                            dtype=torch.float32,
                            device=device,
                        )
                        if residual_priority is not None
                        else None
                    )
                    position_bonus = (
                        (valid_rows - positions) / max(valid_rows, 1) * 0.05
                        + (valid_rows - positions)
                        * float(draft_position_priority_scale())
                        if positions is not None
                        else None
                    )
                    priority_signal = (cutoff - score_slice).clamp(
                        min=0.0, max=1.0
                    )
                    if residual_policy == "critical_prefix":
                        low_confidence = score_present & (score_slice <= cutoff)
                        low_count = torch.cumsum(
                            low_confidence.to(torch.int32), dim=0
                        )
                        score_selected = (low_count == 0) | (
                            low_confidence & (low_count == 1)
                        )
                        if extra_after_low > 0:
                            int_positions = _device_arange(
                                valid_rows,
                                dtype=torch.int32,
                                device=device,
                            )
                            first_low_pos = torch.where(
                                low_confidence,
                                int_positions,
                                torch.full_like(int_positions, valid_rows),
                            ).min()
                            score_selected = score_selected | (
                                int_positions <= first_low_pos + extra_after_low
                            )
                    elif residual_policy == "all_if_any_low":
                        # Conservative request-level ablation: avoid mixing
                        # base-only and corrected draft rows inside one speculative
                        # verify step whenever that step contains any likely reject.
                        risky = (~score_present) | (score_slice <= cutoff)
                        score_selected = torch.zeros_like(score_present) | risky.any()
                    elif residual_policy == "batch_all_if_any_low":
                        # Coarser graph-friendly ablation: avoid row-level mixed
                        # correction entirely. If any draft token in this verify
                        # step is risky, correct every scheduled row; otherwise
                        # leave the whole step base-only.
                        risky = (~score_present) | (score_slice <= cutoff)
                        step_risky = risky.any()
                        batch_risky = (
                            step_risky
                            if batch_risky is None
                            else (batch_risky | step_risky)
                        )
                        score_selected = torch.zeros_like(score_present)
                    elif residual_policy == "prefix_confidence":
                        # Speculative acceptance is a prefix event. Correct
                        # rows whose DLM-selected-token prefix probability is
                        # still high, because accepted base-only prefix rows
                        # can write hidden/KV state that later rows depend on.
                        safe_scores = torch.where(
                            score_present,
                            score_slice.clamp(min=0.0, max=1.0),
                            torch.zeros_like(score_slice),
                        )
                        prefix_confidence = torch.cumprod(safe_scores, dim=0)
                        score_selected = prefix_confidence >= prefix_cutoff
                        priority_signal = prefix_confidence
                    elif residual_policy == "fixed_prefix":
                        int_positions = _device_arange(
                            valid_rows,
                            dtype=torch.int32,
                            device=device,
                        )
                        score_selected = int_positions < min_prefix_residual
                        priority_signal = torch.ones_like(score_slice)
                    else:
                        score_selected = (
                            score_slice <= cutoff
                            if residual_policy == "low_confidence"
                            else score_slice > cutoff
                        )
                        if residual_policy == "high_confidence":
                            priority_signal = score_slice.clamp(min=0.0, max=1.0)
                    if non_draft_policy == "bonus" and has_bonus_row:
                        bonus_use_residual = torch.ones(
                            (), dtype=torch.bool, device=device
                        )
                    elif (
                        non_draft_policy == "predicted_full_accept"
                        and has_bonus_row
                    ):
                        # The bonus row is used only when the draft prefix is fully
                        # accepted. Use DLM confidence as a cheap GPU-side proxy:
                        # correct the bonus when every draft score is present and
                        # above the same threshold used for draft routing.
                        bonus_use_residual = score_present.all() & (
                            score_slice > cutoff
                        ).all()
                    use_residual = torch.where(
                        score_present,
                        score_selected,
                        torch.ones_like(score_present),
                    )
                    if (
                        residual_policy == "low_confidence"
                        and max_residual_draft_rows > 0
                    ):
                        if low_confidence_cap_by_risk():
                            prefix_positions = _device_arange(
                                valid_rows,
                                dtype=torch.int32,
                                device=device,
                            )
                            prefix_use_residual = (
                                prefix_positions < min_prefix_residual
                                if min_prefix_residual > 0
                                else torch.zeros_like(use_residual)
                            )
                            capped_candidates = use_residual & ~prefix_use_residual
                            capped_count = int(max_residual_draft_rows)
                            if capped_count > 0 and int(valid_rows) > 0:
                                risk_scores = torch.where(
                                    capped_candidates,
                                    (cutoff - score_slice).clamp(min=0.0),
                                    torch.full_like(score_slice, float("-inf")),
                                )
                                k = min(capped_count, int(valid_rows))
                                selected = torch.zeros_like(use_residual)
                                topk_values, topk_idx = torch.topk(
                                    risk_scores,
                                    k=k,
                                    largest=True,
                                    sorted=False,
                                )
                                selected.scatter_(
                                    0,
                                    topk_idx,
                                    topk_values > float("-inf"),
                                )
                                use_residual = prefix_use_residual | selected
                            else:
                                use_residual = prefix_use_residual
                        else:
                            residual_rank = torch.cumsum(
                                use_residual.to(torch.int32), dim=0
                            )
                            use_residual = use_residual & (
                                residual_rank <= max_residual_draft_rows
                            )
                    if min_prefix_residual > 0:
                        prefix_positions = _device_arange(
                            valid_rows,
                            dtype=torch.int32,
                            device=device,
                        )
                        use_residual = use_residual | (
                            prefix_positions < min_prefix_residual
                        )
                    if residual_priority is not None:
                        if position_bonus is None:
                            position_bonus = torch.zeros_like(score_slice)
                        safe_priority_signal = torch.where(
                            score_present,
                            priority_signal,
                            torch.zeros_like(priority_signal),
                        )
                        score_priority = torch.where(
                            score_present,
                            2.0 + safe_priority_signal + position_bonus,
                            torch.full_like(score_slice, 3.5),
                        )
                        score_priority = torch.where(
                            use_residual,
                            score_priority,
                            torch.zeros_like(score_priority),
                        )
                    _breakdown_record_cuda(
                        "scheduler_score_policy_cuda_ms",
                        score_policy_cuda_start,
                    )
                    if track_exact_stats:
                        missing_score_tokens += int((~score_present).sum().item())
                if min_prefix_residual > 0:
                    prefix_positions = _device_arange(
                        valid_rows,
                        dtype=torch.int32,
                        device=device,
                    )
                    prefix_use_residual = prefix_positions < min_prefix_residual
                    use_residual = use_residual | prefix_use_residual
                    if residual_priority is not None and score_priority is not None:
                        score_priority = torch.where(
                            prefix_use_residual,
                            torch.maximum(
                                score_priority,
                                torch.full_like(score_priority, 5.0),
                            ),
                            score_priority,
                        )
                if (
                    residual_policy != "low_confidence"
                    and max_residual_draft_rows > 0
                ):
                    cap_positions = _device_arange(
                        valid_rows,
                        dtype=torch.int32,
                        device=device,
                    )
                    forced_prefix = (
                        cap_positions < min_prefix_residual
                        if min_prefix_residual > 0
                        else torch.zeros_like(use_residual)
                    )
                    extra_budget = max(
                        0,
                        int(max_residual_draft_rows)
                        - int(min_prefix_residual),
                    )
                    if extra_budget <= 0:
                        use_residual = forced_prefix
                    else:
                        capped_candidates = use_residual & ~forced_prefix
                        capped_rank = torch.cumsum(
                            capped_candidates.to(torch.int32),
                            dim=0,
                        )
                        use_residual = forced_prefix | (
                            capped_candidates & (capped_rank <= extra_budget)
                        )
                    if residual_priority is not None and score_priority is not None:
                        score_priority = torch.where(
                            use_residual,
                            score_priority,
                            torch.zeros_like(score_priority),
                        )
                if early_tokens > 0 and generated_len is not None:
                    early_positions = _device_arange(
                        valid_rows,
                        dtype=torch.int32,
                        device=device,
                    )
                    early_use_residual = early_positions + int(generated_len) < early_tokens
                    use_residual = use_residual | early_use_residual
                    if residual_priority is not None and score_priority is not None:
                        score_priority = torch.where(
                            early_use_residual,
                            torch.maximum(
                                score_priority,
                                torch.full_like(score_priority, 5.0),
                            ),
                            score_priority,
                        )
                    if has_bonus_row and int(generated_len) + valid_rows < early_tokens:
                        bonus_use_residual = torch.ones(
                            (), dtype=torch.bool, device=device
                        )
                        if track_exact_stats:
                            early_residual_non_draft_tokens += 1
                    if track_exact_stats:
                        early_residual_draft_tokens += int(
                            early_use_residual.sum().item()
                        )
                mask_write_cuda_start = _breakdown_cuda_start()
                residual_mask[row_slice] = use_residual
                if residual_priority is not None and score_priority is not None:
                    residual_priority[row_slice] = score_priority
                if bonus_use_residual is not None:
                    residual_mask[bonus_row] = bonus_use_residual
                    if residual_priority is not None:
                        residual_priority[bonus_row] = bonus_use_residual.to(
                            dtype=torch.float32
                        ) * float(bonus_priority())
                    if track_exact_stats:
                        predicted_residual_non_draft_tokens += int(
                            bonus_use_residual.item()
                        )
                    elif non_draft_policy == "bonus":
                        predicted_residual_non_draft_tokens += 1
                _breakdown_record_cuda(
                    "scheduler_mask_write_cuda_ms",
                    mask_write_cuda_start,
                )
                if track_exact_stats:
                    residual_count = int(use_residual.sum().item())
                    residual_draft_tokens += residual_count
                    base_only_draft_tokens += valid_rows - residual_count
        _breakdown_record_cpu(
            "scheduler_request_routing_loop_cpu_ms",
            routing_loop_start,
        )
        _runtime_wall_record(
            "scheduler_request_routing_loop_wall_cpu_ms",
            routing_loop_wall_start,
        )

    batch_all_risky_cpu: bool | None = None
    if residual_policy == "batch_all_if_any_low":
        batch_all_apply_cpu_start = _breakdown_cpu_start()
        batch_all_apply_cuda_start = _breakdown_cuda_start()
        batch_all_apply_wall_start = _runtime_wall_start()
        if batch_risky is not None:
            residual_mask.copy_(batch_risky.expand_as(residual_mask))
        else:
            residual_mask.fill_(False)
        _breakdown_record_cuda(
            "scheduler_batch_all_mask_apply_cuda_ms",
            batch_all_apply_cuda_start,
        )
        _breakdown_record_cpu(
            "scheduler_batch_all_mask_apply_cpu_ms",
            batch_all_apply_cpu_start,
        )
        _runtime_wall_record(
            "scheduler_batch_all_apply_wall_cpu_ms",
            batch_all_apply_wall_start,
        )
        if track_exact_stats:
            batch_all_risky_cpu = bool(
                batch_risky.item() if batch_risky is not None else False
            )
            residual_draft_tokens = (
                total_valid_draft_tokens if batch_all_risky_cpu else 0
            )
            base_only_draft_tokens = (
                0 if batch_all_risky_cpu else total_valid_draft_tokens
            )

    non_draft_tokens = max(0, total_num_scheduled_tokens - total_valid_draft_tokens)
    if residual_policy == "batch_all_if_any_low":
        if track_exact_stats:
            residual_non_draft_tokens = (
                non_draft_tokens if bool(batch_all_risky_cpu) else 0
            )
            base_only_non_draft_tokens = (
                0 if bool(batch_all_risky_cpu) else non_draft_tokens
            )
        else:
            residual_non_draft_tokens = 0
            base_only_non_draft_tokens = non_draft_tokens
    elif sr_mode == "base_only":
        residual_non_draft_tokens = 0
        base_only_non_draft_tokens = non_draft_tokens
    elif sr_mode == "selective" and non_draft_policy in {
        "bonus",
        "predicted_full_accept",
    }:
        if track_exact_stats:
            residual_non_draft_tokens = predicted_residual_non_draft_tokens
            base_only_non_draft_tokens = max(
                0, non_draft_tokens - residual_non_draft_tokens
            )
        else:
            residual_non_draft_tokens = predicted_residual_non_draft_tokens
            base_only_non_draft_tokens = max(
                0, non_draft_tokens - residual_non_draft_tokens
            )
    elif sr_mode == "all_corrected" or (
        sr_mode == "selective" and non_draft_policy == "all"
    ):
        residual_non_draft_tokens = non_draft_tokens
        base_only_non_draft_tokens = 0
    else:
        residual_non_draft_tokens = 0
        base_only_non_draft_tokens = non_draft_tokens

    if not stats_exact:
        _breakdown_count_routing_gpu(
            residual_mask=residual_mask,
            rows=total_num_scheduled_tokens,
            total_valid_draft_tokens=total_valid_draft_tokens,
            non_draft_tokens=non_draft_tokens,
            residual_non_draft_tokens=residual_non_draft_tokens,
        )

    mask_state_exact = True
    if sr_mode == "base_only":
        mask_state = "no_residual"
    elif stats_exact:
        residual_total = residual_draft_tokens + residual_non_draft_tokens
        if residual_total <= 0:
            mask_state = "no_residual"
        elif residual_total >= total_num_scheduled_tokens:
            mask_state = "all_residual"
        else:
            mask_state = "mixed"
    elif static_state in {"all_residual", "no_residual", "mixed"}:
        mask_state = static_state
        mask_state_exact = False
    elif (
        sr_mode == "selective"
        and sync_reduced
        and batched_mask_applied
        and (
            conservative_residual_draft_tokens
            + conservative_residual_non_draft_tokens
        )
        >= total_num_scheduled_tokens
    ):
        # The batched mask builder returns a no-sync lower bound for rows that
        # are forced residual independent of runtime confidence values. When
        # that bound already covers every scheduled row (commonly from the
        # early-dense guard plus the bonus-row policy), the mixed mask is
        # logically all-residual. Use the dense shortcut instead of running the
        # slower sparse-base plus dense-correction path.
        mask_state = "all_residual"
        mask_state_exact = False
        _breakdown_count("scheduler_conservative_all_residual_steps", 1)
    elif (
        sr_mode == "selective"
        and sync_reduced
        and batched_mask_applied
        and 0.0 <= route_dense_fallback_fraction() <= 1.0
    ):
        fallback_fraction = route_dense_fallback_fraction()
        conservative_total = (
            conservative_residual_draft_tokens
            + conservative_residual_non_draft_tokens
        )
        if (
            conservative_total / max(total_num_scheduled_tokens, 1)
            >= fallback_fraction
        ):
            # Avoid the GPU->CPU mask sum used by the exact fallback path.
            # This conservative count includes only rows that the routing policy
            # forces residual independent of runtime confidence values. Promoting
            # the step to all-residual can only correct extra rows, so it keeps
            # correctness while preserving the dense/all-residual fast path.
            mask_state = "all_residual"
            _breakdown_count("scheduler_dense_fallback_predicted_steps", 1)
        else:
            mask_state = "mixed"
            _breakdown_count("scheduler_dense_fallback_predicted_mixed_steps", 1)
        mask_state_exact = False
    elif sync_mask_state():
        # One decode-step synchronization is much cheaper than checking
        # mask.any()/mask.all() inside every target Linear. This keeps the
        # all-residual dense fastpath available while avoiding per-layer CPU
        # syncs in the hot path.
        mask_state_sync_cpu_start = _breakdown_cpu_start()
        mask_state_sum_cuda_start = _breakdown_cuda_start()
        mask_state_wall_start = _runtime_wall_start()
        residual_total = int(residual_mask.to(torch.int32).sum().item())
        _breakdown_record_cuda(
            "scheduler_mask_state_sum_cuda_ms",
            mask_state_sum_cuda_start,
        )
        _breakdown_record_cpu(
            "scheduler_mask_state_sync_cpu_ms",
            mask_state_sync_cpu_start,
        )
        _runtime_wall_record(
            "scheduler_mask_state_wall_cpu_ms",
            mask_state_wall_start,
        )
        if residual_total <= 0:
            mask_state = "no_residual"
        elif residual_total >= total_num_scheduled_tokens:
            mask_state = "all_residual"
        else:
            fallback_fraction = route_dense_fallback_fraction()
            if (
                sr_mode == "selective"
                and 0.0 <= fallback_fraction <= 1.0
                and residual_total / max(total_num_scheduled_tokens, 1)
                >= fallback_fraction
            ):
                # Conservative mixed-mask fallback: when most rows would need
                # residual correction, use the dense all-residual fastpath for
                # the whole step. This corrects extra rows instead of leaving
                # them base-only, and avoids paying sparse-base plus a large
                # dense-row correction.
                mask_state = "all_residual"
                mask_state_exact = False
                _breakdown_count("scheduler_dense_fallback_steps", 1)
            else:
                mask_state = "mixed"
    else:
        mask_state = "mixed"
        mask_state_exact = False

    trace_confidence = _confidence_trace_enabled()
    if trace_confidence or debug_trace_enabled():
        for req_idx, req_id in enumerate(req_ids):
            if req_idx >= len(draft_counts) or draft_counts[req_idx] <= 0:
                continue
            end = cu_scheduled_counts[req_idx]
            sched = scheduled_counts[req_idx]
            start = end - sched
            valid_rows = max(0, min(draft_counts[req_idx],
                                    total_num_scheduled_tokens - start))
            if valid_rows <= 0:
                continue
            generated_len = (
                generated_lens_by_req[req_idx]
                if req_idx < len(generated_lens_by_req)
                else None
            )
            if trace_confidence:
                if mask_state == "all_residual":
                    trace_mask = torch.ones(
                        valid_rows,
                        dtype=torch.bool,
                        device=device,
                    )
                elif mask_state == "no_residual":
                    trace_mask = torch.zeros(
                        valid_rows,
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    trace_mask = residual_mask[start:start + valid_rows]
                _trace_record_sr24_verify_mask(
                    req_id=req_id,
                    residual_mask=trace_mask,
                    scores=scores_by_req[req_idx],
                    valid_rows=valid_rows,
                    residual_policy=residual_policy,
                    non_draft_policy=non_draft_policy,
                    threshold=cutoff,
                    mask_state=mask_state,
                    generated_len=generated_len,
                )
            if debug_trace_enabled():
                _debug_trace_verify_request(
                    req_id=req_id,
                    req_idx=req_idx,
                    sr_mode=sr_mode,
                    residual_policy=residual_policy,
                    non_draft_policy=non_draft_policy,
                    cutoff=cutoff,
                    draft_count=draft_counts[req_idx],
                    scheduled_count=scheduled_counts[req_idx],
                    cu_scheduled_count=cu_scheduled_counts[req_idx],
                    total_num_scheduled_tokens=total_num_scheduled_tokens,
                    scores=scores_by_req[req_idx],
                    residual_mask=residual_mask,
                    mask_state=mask_state,
                    generated_len=generated_len,
                    batched_mask_applied=batched_mask_applied,
                )

    runtime_record: dict[str, Any] | None = None
    effective_stats_exact = stats_exact
    record_residual_draft_tokens: int | None = (
        residual_draft_tokens if stats_exact else None
    )
    record_base_only_draft_tokens: int | None = (
        base_only_draft_tokens if stats_exact else None
    )
    record_residual_non_draft_tokens = residual_non_draft_tokens
    record_base_only_non_draft_tokens = base_only_non_draft_tokens
    record_missing_score_tokens: int | None = (
        missing_score_tokens if stats_exact else None
    )
    if mask_state == "all_residual":
        effective_stats_exact = True
        record_residual_draft_tokens = total_valid_draft_tokens
        record_base_only_draft_tokens = 0
        record_residual_non_draft_tokens = non_draft_tokens
        record_base_only_non_draft_tokens = 0
        record_missing_score_tokens = missing_score_tokens if stats_exact else None
    elif mask_state == "no_residual":
        effective_stats_exact = True
        record_residual_draft_tokens = 0
        record_base_only_draft_tokens = total_valid_draft_tokens
        record_residual_non_draft_tokens = 0
        record_base_only_non_draft_tokens = non_draft_tokens
        record_missing_score_tokens = missing_score_tokens if stats_exact else None
    runtime_residual_requested_rows = (
        (record_residual_draft_tokens or 0) + record_residual_non_draft_tokens
        if effective_stats_exact
        else None
    )
    if runtime_stats_enabled():
        runtime_record = {
            "timestamp": time.time(),
            "event": "sr24_verify_mask",
            "mode": sr_mode,
            "threshold": cutoff,
            "prefix_threshold": prefix_cutoff,
            "stats_interval": _stats_interval(),
            "selective_correct_non_draft": correct_non_draft,
            "selective_non_draft_policy": non_draft_policy,
            "selective_residual_policy": residual_policy,
            "selective_extra_after_low": extra_after_low,
            "selective_min_prefix_residual": min_prefix_residual,
            "selective_max_residual_draft_rows": max_residual_draft_rows,
            "low_confidence_cap_by_risk": low_confidence_cap_by_risk(),
            "early_dense_tokens": early_tokens,
            "dense_fallback_nonuniform": nonuniform_dense_fallback,
            "sync_reduced_stats": sync_reduced,
            "sync_mask_state": sync_mask_state(),
            "static_mask_state": static_state,
            "stats_exact": effective_stats_exact,
            "routing_stats_exact": stats_exact,
            "mask_state": mask_state,
            "mask_state_exact": mask_state_exact,
            "request_count": len(req_ids),
            "total_scheduled_tokens": total_num_scheduled_tokens,
            "total_draft_tokens": total_draft_tokens,
            "total_valid_draft_tokens": total_valid_draft_tokens,
            "non_draft_tokens": non_draft_tokens,
            "residual_draft_tokens": record_residual_draft_tokens,
            "base_only_draft_tokens": record_base_only_draft_tokens,
            "residual_non_draft_tokens": record_residual_non_draft_tokens,
            "base_only_non_draft_tokens": record_base_only_non_draft_tokens,
            "conservative_residual_draft_tokens":
            conservative_residual_draft_tokens,
            "conservative_residual_non_draft_tokens":
            conservative_residual_non_draft_tokens,
            "early_residual_draft_tokens": early_residual_draft_tokens,
            "early_residual_non_draft_tokens": early_residual_non_draft_tokens,
            "missing_score_tokens": record_missing_score_tokens,
            "residual_draft_fraction": (
                record_residual_draft_tokens / total_draft_tokens
                if record_residual_draft_tokens is not None and total_draft_tokens
                else None
            ),
        }

    _trace_grouping_opportunity(
        reason="post_mask_state",
        sr_mode=sr_mode,
        residual_policy=residual_policy,
        non_draft_policy=non_draft_policy,
        mask_state=mask_state,
        compact_spec_batch=compact_spec_batch,
        compact_width=compact_width,
        draft_counts=draft_counts,
        scheduled_counts=scheduled_counts,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        min_prefix_residual=min_prefix_residual,
        fixed_prefix_route=fixed_prefix_route_descriptor,
        batched_mask_applied=batched_mask_applied,
    )

    if mask_state in {"all_residual", "no_residual"}:
        if runtime_record is not None:
            _attach_runtime_bucket_stats(
                runtime_record,
                rows=total_num_scheduled_tokens,
                bucket=None,
                residual_requested_rows=runtime_residual_requested_rows,
                sync_active=False,
            )
            _attach_runtime_timings(runtime_record)
            summary_record = _accumulate_stats(runtime_record)
            _write_verify_record(runtime_record, summary_record)
        return _finalize_verify_plan_breakdown(
            VerifyResidualPlan(mask=None, state=mask_state),
            cpu_start=breakdown_cpu_start,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            total_draft_tokens=total_draft_tokens,
            total_valid_draft_tokens=total_valid_draft_tokens,
            non_draft_tokens=non_draft_tokens,
            residual_draft_tokens=record_residual_draft_tokens,
            base_only_draft_tokens=record_base_only_draft_tokens,
            residual_non_draft_tokens=record_residual_non_draft_tokens,
            base_only_non_draft_tokens=record_base_only_non_draft_tokens,
            early_residual_draft_tokens=(
                early_residual_draft_tokens if stats_exact else None
            ),
            early_residual_non_draft_tokens=early_residual_non_draft_tokens,
            missing_score_tokens=record_missing_score_tokens,
            stats_exact=effective_stats_exact,
        )
    if static_mask_buffer_enabled():
        static_copy_cpu_start = _breakdown_cpu_start()
        static_copy_cuda_start = _breakdown_cuda_start()
        static_copy_wall_start = _runtime_wall_start()
        if force_cudagraph_none_for_mixed_enabled():
            static_mask = _static_residual_mask_view(
                total_num_scheduled_tokens,
                device,
            )
        else:
            # Mixed CUDA Graph ablations capture padded decode shapes. The
            # runtime scheduler only materializes real scheduled rows, so keep
            # the unused tail conservative and deterministic instead of
            # replaying stale base-only bits from an earlier shorter step.
            static_mask_full = _static_residual_mask_view(
                max(static_mask_buffer_capacity(), total_num_scheduled_tokens),
                device,
                fill_value=True,
            )
            static_mask = static_mask_full[:total_num_scheduled_tokens]
            _breakdown_count("scheduler_static_mask_fill_tail_for_graph", 1)
        static_mask.copy_(residual_mask, non_blocking=False)
        static_priority = None
        if residual_priority is not None:
            static_priority = _static_residual_priority_view(
                total_num_scheduled_tokens,
                device,
            )
            static_priority.copy_(residual_priority, non_blocking=False)
        _breakdown_record_cuda(
            "scheduler_static_mask_copy_cuda_ms",
            static_copy_cuda_start,
        )
        _breakdown_record_cpu(
            "scheduler_static_mask_copy_cpu_ms",
            static_copy_cpu_start,
        )
        _runtime_wall_record(
            "scheduler_static_mask_copy_wall_cpu_ms",
            static_copy_wall_start,
        )
        row_index_bucket_wall_start = _runtime_wall_start()
        residual_bucket_wall_start = _runtime_wall_start()
        static_bucket = direct_position_bucket
        if static_bucket is not None:
            _breakdown_count("scheduler_direct_position_bucket_used_steps", 1)
        elif (
            direct_cpu_route_rows is not None
            and route_all_residual_rows()
            and not row_routed_mlp()
        ):
            static_bucket = None
            _breakdown_count("scheduler_direct_cpu_route_rows_skip_bucket", 1)
        elif (
            route_all_skip_bucket()
            and route_all_residual_rows()
            and not row_routed_mlp()
        ):
            static_bucket = None
            _breakdown_count("scheduler_route_all_bucket_skipped", 1)
        elif skip_residual_bucket:
            static_bucket = None
            _breakdown_count("scheduler_residual_bucket_skipped_for_quality", 1)
        else:
            static_bucket = _compute_residual_bucket(
                rows=total_num_scheduled_tokens,
                device=device,
                residual_mask=static_mask,
                residual_priority=static_priority,
                active_count=len(req_ids),
            )
        _runtime_wall_record(
            "scheduler_residual_bucket_wall_cpu_ms",
            residual_bucket_wall_start,
        )
        row_indices_wall_start = _runtime_wall_start()
        if (
            route_bucket_rows()
            and static_bucket is not None
            and not route_all_residual_rows()
        ):
            static_residual_rows = static_bucket[0]
            static_base_rows = _compute_bucket_complement_rows(
                rows=total_num_scheduled_tokens,
                device=device,
                bucket_rows=static_bucket[0],
            )
            _breakdown_count("scheduler_route_bucket_rows_plan_hits", 1)
        elif direct_cpu_route_rows is not None:
            static_residual_rows, static_base_rows = direct_cpu_route_rows
            _breakdown_count("scheduler_direct_cpu_route_rows_plan_hits", 1)
        elif (
            fixed_prefix_route_descriptor is not None
            and row_routed_mlp()
            and route_all_residual_rows()
        ):
            static_residual_rows = None
            static_base_rows = None
            _breakdown_count(
                "scheduler_fixed_prefix_route_descriptor_only_plan_hits", 1
            )
        else:
            static_residual_rows, static_base_rows = _compute_mixed_row_indices(
                rows=total_num_scheduled_tokens,
                device=device,
                residual_mask=static_mask,
            )
        _runtime_wall_record(
            "scheduler_mixed_row_indices_wall_cpu_ms",
            row_indices_wall_start,
        )
        if (
            row_routed_mlp()
            and static_bucket is not None
            and fixed_prefix_route_descriptor is None
        ):
            static_residual_rows = static_bucket[0]
            if row_routed_mlp_reuse_base_output():
                static_base_rows = None
                _breakdown_count("scheduler_bucket_base_rows_reuse_base_skips", 1)
            else:
                static_base_rows = _compute_bucket_complement_rows(
                    rows=total_num_scheduled_tokens,
                    device=device,
                    bucket_rows=static_bucket[0],
                )
        _runtime_wall_record(
            "scheduler_row_index_bucket_wall_cpu_ms",
            row_index_bucket_wall_start,
        )
        if runtime_record is not None:
            _attach_runtime_bucket_stats(
                runtime_record,
                rows=total_num_scheduled_tokens,
                bucket=static_bucket,
                residual_requested_rows=runtime_residual_requested_rows,
                sync_active=stats_exact,
            )
            _attach_runtime_timings(runtime_record)
            summary_record = _accumulate_stats(runtime_record)
            _write_verify_record(runtime_record, summary_record)
        return _finalize_verify_plan_breakdown(
            VerifyResidualPlan(
                mask=static_mask,
                state=mask_state,
                priority=static_priority,
                bucket=static_bucket,
                residual_rows=static_residual_rows,
                base_rows=static_base_rows,
                fixed_prefix_route=fixed_prefix_route_descriptor,
            ),
            cpu_start=breakdown_cpu_start,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            total_draft_tokens=total_draft_tokens,
            total_valid_draft_tokens=total_valid_draft_tokens,
            non_draft_tokens=non_draft_tokens,
            residual_draft_tokens=residual_draft_tokens if stats_exact else None,
            base_only_draft_tokens=base_only_draft_tokens if stats_exact else None,
            residual_non_draft_tokens=residual_non_draft_tokens,
            base_only_non_draft_tokens=base_only_non_draft_tokens,
            early_residual_draft_tokens=(
                early_residual_draft_tokens if stats_exact else None
            ),
            early_residual_non_draft_tokens=early_residual_non_draft_tokens,
            missing_score_tokens=missing_score_tokens if stats_exact else None,
            stats_exact=stats_exact,
        )
    row_index_bucket_wall_start = _runtime_wall_start()
    residual_bucket_wall_start = _runtime_wall_start()
    residual_bucket = direct_position_bucket
    if residual_bucket is not None:
        _breakdown_count("scheduler_direct_position_bucket_used_steps", 1)
    elif (
        direct_cpu_route_rows is not None
        and route_all_residual_rows()
        and not row_routed_mlp()
    ):
        residual_bucket = None
        _breakdown_count("scheduler_direct_cpu_route_rows_skip_bucket", 1)
    elif (
        route_all_skip_bucket()
        and route_all_residual_rows()
        and not row_routed_mlp()
    ):
        residual_bucket = None
        _breakdown_count("scheduler_route_all_bucket_skipped", 1)
    elif skip_residual_bucket:
        residual_bucket = None
        _breakdown_count("scheduler_residual_bucket_skipped_for_quality", 1)
    else:
        residual_bucket = _compute_residual_bucket(
            rows=total_num_scheduled_tokens,
            device=device,
            residual_mask=residual_mask,
            residual_priority=residual_priority,
            active_count=len(req_ids),
        )
    _runtime_wall_record(
        "scheduler_residual_bucket_wall_cpu_ms",
        residual_bucket_wall_start,
    )
    row_indices_wall_start = _runtime_wall_start()
    if (
        route_bucket_rows()
        and residual_bucket is not None
        and not route_all_residual_rows()
    ):
        residual_rows = residual_bucket[0]
        base_rows = _compute_bucket_complement_rows(
            rows=total_num_scheduled_tokens,
            device=device,
            bucket_rows=residual_bucket[0],
        )
        _breakdown_count("scheduler_route_bucket_rows_plan_hits", 1)
    elif direct_cpu_route_rows is not None:
        residual_rows, base_rows = direct_cpu_route_rows
        _breakdown_count("scheduler_direct_cpu_route_rows_plan_hits", 1)
    elif (
        fixed_prefix_route_descriptor is not None
        and row_routed_mlp()
        and route_all_residual_rows()
    ):
        residual_rows = None
        base_rows = None
        _breakdown_count(
            "scheduler_fixed_prefix_route_descriptor_only_plan_hits", 1
        )
    else:
        residual_rows, base_rows = _compute_mixed_row_indices(
            rows=total_num_scheduled_tokens,
            device=device,
            residual_mask=residual_mask,
        )
    _runtime_wall_record(
        "scheduler_mixed_row_indices_wall_cpu_ms",
        row_indices_wall_start,
    )
    if (
        row_routed_mlp()
        and residual_bucket is not None
        and fixed_prefix_route_descriptor is None
    ):
        residual_rows = residual_bucket[0]
        if row_routed_mlp_reuse_base_output():
            base_rows = None
            _breakdown_count("scheduler_bucket_base_rows_reuse_base_skips", 1)
        else:
            base_rows = _compute_bucket_complement_rows(
                rows=total_num_scheduled_tokens,
                device=device,
                bucket_rows=residual_bucket[0],
            )
    _runtime_wall_record(
        "scheduler_row_index_bucket_wall_cpu_ms",
        row_index_bucket_wall_start,
    )
    if runtime_record is not None:
        _attach_runtime_bucket_stats(
            runtime_record,
            rows=total_num_scheduled_tokens,
            bucket=residual_bucket,
            residual_requested_rows=runtime_residual_requested_rows,
            sync_active=stats_exact,
        )
        _attach_runtime_timings(runtime_record)
        summary_record = _accumulate_stats(runtime_record)
        _write_verify_record(runtime_record, summary_record)
    return _finalize_verify_plan_breakdown(
        VerifyResidualPlan(
            mask=residual_mask,
            state=mask_state,
            priority=residual_priority,
            bucket=residual_bucket,
            residual_rows=residual_rows,
            base_rows=base_rows,
            fixed_prefix_route=fixed_prefix_route_descriptor,
        ),
        cpu_start=breakdown_cpu_start,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        total_draft_tokens=total_draft_tokens,
        total_valid_draft_tokens=total_valid_draft_tokens,
        non_draft_tokens=non_draft_tokens,
        residual_draft_tokens=residual_draft_tokens if stats_exact else None,
        base_only_draft_tokens=base_only_draft_tokens if stats_exact else None,
        residual_non_draft_tokens=residual_non_draft_tokens,
        base_only_non_draft_tokens=base_only_non_draft_tokens,
        early_residual_draft_tokens=(
            early_residual_draft_tokens if stats_exact else None
        ),
        early_residual_non_draft_tokens=early_residual_non_draft_tokens,
        missing_score_tokens=missing_score_tokens if stats_exact else None,
        stats_exact=stats_exact,
    )


def sparse_backend_active(module: Any) -> bool:
    return (
        enabled()
        and getattr(module, "_speclink_sr24_enabled", False)
        and getattr(module, "_speclink_sr24_backend", "") == "torch_sparse"
        and not getattr(module, "_speclink_sr24_dense_fastpath", False)
    )


def dense_zero_dense_rows_active(module: Any) -> bool:
    return (
        enabled()
        and getattr(module, "_speclink_sr24_enabled", False)
        and getattr(module, "_speclink_sr24_backend", "") in DENSE_ZERO_BACKENDS
        and getattr(module, "_speclink_sr24_dense_weight", None) is not None
    )


def _current_residual_state() -> str | None:
    return _fast_verify_residual_state


def _current_residual_mask() -> torch.Tensor | None:
    return _fast_verify_residual_mask


def _current_residual_priority() -> torch.Tensor | None:
    return _fast_verify_residual_priority


def _current_residual_bucket() -> tuple[torch.Tensor, torch.Tensor] | None:
    return _fast_verify_residual_bucket


def _current_residual_rows() -> torch.Tensor | None:
    return _fast_verify_residual_rows


def _current_base_rows() -> torch.Tensor | None:
    return _fast_verify_base_rows


def _current_fixed_prefix_route() -> FixedPrefixRouteDescriptor | None:
    return _fast_verify_fixed_prefix_route


def _residual_rows_for_input(input_tensor: torch.Tensor) -> torch.Tensor | None:
    rows = int(input_tensor.shape[0])
    if rows <= 0:
        return None
    sr_mode = mode()
    if sr_mode == "base_only":
        return None
    if sr_mode == "all_corrected":
        return _device_arange(rows, dtype=torch.long, device=input_tensor.device)
    residual_state = _current_residual_state()
    if residual_state == "no_residual":
        return None
    if residual_state == "all_residual":
        return _device_arange(rows, dtype=torch.long, device=input_tensor.device)
    residual_mask = _current_residual_mask()
    if residual_mask is None:
        if sr_mode == "base_only":
            return None
        if sr_mode == "selective" and not selective_correct_non_draft():
            return None
        return _device_arange(rows, dtype=torch.long, device=input_tensor.device)
    if residual_mask.numel() < rows:
        raise RuntimeError(
            f"SpecLink SR24 mask has {residual_mask.numel()} rows, "
            f"but linear input has {rows}"
        )
    cached_rows = _current_residual_rows()
    if cached_rows is not None and residual_mask.numel() == rows:
        _breakdown_count("cached_residual_rows_hits", 1)
        return cached_rows.to(device=input_tensor.device, dtype=torch.long)
    row_uses_residual = residual_mask[:rows].to(device=input_tensor.device)
    return row_uses_residual.nonzero(as_tuple=False).squeeze(1)


def _residual_mask_for_input(input_tensor: torch.Tensor) -> torch.Tensor | None:
    rows = int(input_tensor.shape[0])
    if rows <= 0 or mode() == "base_only":
        return None
    residual_state = _current_residual_state()
    if residual_state == "no_residual":
        return None
    if residual_state == "all_residual":
        if static_mask_buffer_enabled():
            return _static_residual_mask_view(
                rows,
                input_tensor.device,
                fill_value=True,
            )
        return torch.ones(rows, dtype=torch.bool, device=input_tensor.device)
    residual_mask = _current_residual_mask()
    if residual_mask is None:
        sr_mode = mode()
        if sr_mode == "base_only":
            return None
        if sr_mode == "selective" and not selective_correct_non_draft():
            return None
        if static_mask_buffer_enabled():
            return _static_residual_mask_view(
                rows,
                input_tensor.device,
                fill_value=True,
            )
        return torch.ones(rows, dtype=torch.bool, device=input_tensor.device)
    if residual_mask.numel() < rows:
        raise RuntimeError(
            f"SpecLink SR24 mask has {residual_mask.numel()} rows, "
            f"but linear input has {rows}"
        )
    return residual_mask[:rows].to(device=input_tensor.device)


def _residual_priority_for_input(input_tensor: torch.Tensor) -> torch.Tensor | None:
    rows = int(input_tensor.shape[0])
    if rows <= 0 or not residual_bucket_priority():
        return None
    residual_priority = _current_residual_priority()
    if residual_priority is None:
        return None
    if residual_priority.numel() < rows:
        raise RuntimeError(
            f"SpecLink SR24 priority has {residual_priority.numel()} rows, "
            f"but linear input has {rows}"
        )
    return residual_priority[:rows].to(device=input_tensor.device)


def _cache_mixed_row_indices_enabled() -> bool:
    return route_all_residual_rows() or route_reuse_base_output()


def _compute_mixed_row_indices(
    *,
    rows: int,
    device: torch.device,
    residual_mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not _cache_mixed_row_indices_enabled() or rows <= 0:
        return None, None
    if residual_mask.numel() < rows:
        return None, None
    timer = _breakdown_cuda_start()
    row_mask = residual_mask[:rows].to(device=device, dtype=torch.bool)
    residual_rows = row_mask.nonzero(as_tuple=False).squeeze(1)
    base_rows = (~row_mask).nonzero(as_tuple=False).squeeze(1)
    _breakdown_record_cuda("scheduler_row_indices_cuda_ms", timer)
    _breakdown_count("scheduler_row_indices_builds", 1)
    _breakdown_count("scheduler_row_indices_rows", rows)
    _breakdown_count("scheduler_row_indices_residual_rows", int(residual_rows.numel()))
    _breakdown_count("scheduler_row_indices_base_rows", int(base_rows.numel()))
    return residual_rows, base_rows


def _compute_bucket_complement_rows(
    *,
    rows: int,
    device: torch.device,
    bucket_rows: torch.Tensor,
) -> torch.Tensor | None:
    rows = int(rows)
    if rows <= 0:
        return None
    bucket_count = int(bucket_rows.numel())
    if bucket_count <= 0 or bucket_count >= rows:
        return torch.empty(0, dtype=torch.long, device=device)
    static_output = (
        static_mask_buffer_enabled()
        and not force_cudagraph_none_for_mixed_enabled()
        and cudagraph_bucket_enabled()
    )
    timer = _breakdown_cuda_start()
    try:
        base_rows = _triton_bucket_complement_rows(
            rows=rows,
            device=device,
            bucket_rows=bucket_rows,
            static_output=static_output,
        )
    except Exception:
        base_rows = None
        _breakdown_count("scheduler_bucket_base_rows_triton_failures", 1)
    if base_rows is not None:
        _breakdown_record_cuda("scheduler_bucket_base_rows_cuda_ms", timer)
        _breakdown_count("scheduler_bucket_base_rows_builds", 1)
        _breakdown_count("scheduler_bucket_base_rows", int(base_rows.numel()))
        return base_rows

    route_dense = torch.zeros(rows, dtype=torch.bool, device=device)
    route_dense.index_fill_(0, bucket_rows.to(device=device, dtype=torch.long), True)
    base_rows = (~route_dense).nonzero(as_tuple=False).squeeze(1)
    if static_output:
        static_base_rows = _static_long_view(
            "bucket_complement_base_rows",
            int(base_rows.numel()),
            device,
        )
        static_base_rows.copy_(base_rows, non_blocking=False)
        base_rows = static_base_rows
        _breakdown_count("scheduler_static_bucket_base_rows_copy_for_graph", 1)
    _breakdown_record_cuda("scheduler_bucket_base_rows_cuda_ms", timer)
    _breakdown_count("scheduler_bucket_base_rows_builds", 1)
    _breakdown_count("scheduler_bucket_base_rows", int(base_rows.numel()))
    return base_rows


def _compute_residual_bucket(
    *,
    rows: int,
    device: torch.device,
    residual_mask: torch.Tensor,
    residual_priority: torch.Tensor | None = None,
    value_dtype: torch.dtype = torch.float32,
    active_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    bucket_cpu_start = _breakdown_cpu_start()
    rows = int(rows)
    runtime_bucket_size = _effective_residual_bucket_size(active_count)
    if runtime_bucket_size <= 0:
        return None
    graph_static_bucket = (
        static_mask_buffer_enabled()
        and not force_cudagraph_none_for_mixed_enabled()
        and cudagraph_bucket_enabled()
    )
    if rows <= runtime_bucket_size and not graph_static_bucket:
        return None
    k = min(rows, runtime_bucket_size)
    if graph_static_bucket:
        # CUDA Graph capture records the bucket tensor pointers and the static
        # bucket length. Keep that length stable across replay.  For
        # scale-by-active policies, use the launch-time active-request hint
        # (normally the benchmark batch size) and pad inactive tail entries
        # with zero values during drain steps.
        active_hint = cudagraph_bucket_active_hint()
        graph_bucket_size = _effective_residual_bucket_size(
            active_count=active_hint if active_hint > 0 else active_count
        )
        k = min(rows, graph_bucket_size)
    bucket_timer = _breakdown_cuda_start()
    mask_values = residual_mask[:rows].to(
        device=device,
        dtype=value_dtype,
    )
    selection_k = min(rows, runtime_bucket_size, k)
    if rows > runtime_bucket_size and residual_priority is not None:
        priority = residual_priority[:rows].to(
            device=device,
            dtype=torch.float32,
        )
        scores = torch.where(
            mask_values.to(dtype=torch.bool),
            priority,
            torch.full_like(priority, -1.0),
        )
        _, bucket_rows = torch.topk(scores, k=selection_k, largest=True, sorted=False)
        values = mask_values.index_select(0, bucket_rows)
    elif rows > runtime_bucket_size:
        scores = mask_values.to(dtype=torch.float32)
        _, bucket_rows = torch.topk(scores, k=selection_k, largest=True, sorted=False)
        values = mask_values.index_select(0, bucket_rows)
    else:
        bucket_rows = torch.arange(selection_k, dtype=torch.long, device=device)
        # For graph-static buckets the bucket tensor can be longer than the
        # current real input rows. Keep the padded tail inactive; the output
        # tensor is not guaranteed to have valid storage for padded row ids.
        fill_value = 0.0 if graph_static_bucket and selection_k > rows else 1.0
        values = torch.full(
            (selection_k,), fill_value, dtype=value_dtype, device=device
        )
        if rows > 0:
            values[:rows].copy_(mask_values)
    if sort_bucket_rows() and int(bucket_rows.numel()) > 1:
        sort_timer = _breakdown_cuda_start()
        bucket_rows, order = torch.sort(bucket_rows)
        values = values.index_select(0, order)
        _breakdown_record_cuda("scheduler_bucket_sort_cuda_ms", sort_timer)
        _breakdown_count("scheduler_bucket_sort_calls", 1)
        _breakdown_count("scheduler_bucket_sort_rows", int(bucket_rows.numel()))
    if graph_static_bucket:
        static_rows = _static_long_view("bucket_rows", k, device)
        static_values = _static_float_view(
            "bucket_values",
            k,
            device,
            dtype=value_dtype,
        )
        static_values.zero_()
        copy_count = min(int(bucket_rows.numel()), k)
        if copy_count > 0:
            static_rows[:copy_count].copy_(bucket_rows[:copy_count], non_blocking=False)
            static_values[:copy_count].copy_(
                values[:copy_count].to(dtype=value_dtype),
                non_blocking=False,
            )
        if k > copy_count:
            static_rows[copy_count:k].fill_(0)
        bucket_rows = static_rows
        values = static_values
        _breakdown_count("scheduler_static_bucket_copy_for_graph", 1)
    _breakdown_record_cuda("scheduler_bucket_topk_cuda_ms", bucket_timer)
    _breakdown_record_cpu("scheduler_bucket_build_cpu_ms", bucket_cpu_start)
    _breakdown_count_bucket(
        rows=rows,
        bucket_rows=bucket_rows,
        bucket_values=values,
    )
    return bucket_rows, values


def _attach_runtime_bucket_stats(
    record: dict[str, Any],
    *,
    rows: int,
    bucket: tuple[torch.Tensor, torch.Tensor] | None,
    residual_requested_rows: int | None,
    sync_active: bool,
) -> dict[str, Any]:
    if bucket is None:
        record.update(
            {
                "bucket_calls": 0,
                "bucket_candidate_rows": 0,
                "bucket_active_rows": None,
                "bucket_total_rows": int(rows),
                "bucket_residual_requested_rows": residual_requested_rows,
            }
        )
        return record
    bucket_rows, bucket_values = bucket
    record["bucket_calls"] = 1
    record["bucket_candidate_rows"] = int(bucket_rows.numel())
    record["bucket_total_rows"] = int(rows)
    record["bucket_residual_requested_rows"] = residual_requested_rows
    if sync_active:
        try:
            record["bucket_active_rows"] = int(
                bucket_values.to(dtype=torch.int32).sum().item()
            )
        except Exception:
            record["bucket_active_rows"] = None
    else:
        record["bucket_active_rows"] = None
    return record


def _residual_bucket_for_mask(
    input_tensor: torch.Tensor,
    residual_mask: torch.Tensor,
    residual_priority: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if _current_residual_state() != "mixed":
        return None
    rows = int(input_tensor.shape[0])
    cached = _current_residual_bucket()
    if cached is not None and int(residual_mask.numel()) == rows:
        bucket_rows, bucket_values = cached
        if int(bucket_rows.numel()) > rows:
            bucket_rows = bucket_rows[:rows]
            bucket_values = bucket_values[:rows]
        return (
            bucket_rows.to(device=input_tensor.device, non_blocking=True),
            bucket_values.to(
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                non_blocking=True,
            ),
        )
    return _compute_residual_bucket(
        rows=rows,
        device=input_tensor.device,
        residual_mask=residual_mask,
        residual_priority=residual_priority,
        value_dtype=input_tensor.dtype,
    )


@torch.inference_mode()
def _dense_zero_dense_rows_output(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    bias = getattr(module, "bias", None)
    if getattr(module, "skip_bias_add", False):
        bias = None
    dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
    if dense_weight is None:
        raise RuntimeError("SR24 dense_rows weight is missing")
    if mode() == "base_only":
        return F.linear(input_tensor, module.weight, bias).contiguous()
    residual_state = _current_residual_state()
    if residual_state == "no_residual":
        return F.linear(input_tensor, module.weight, bias).contiguous()
    if residual_state == "all_residual":
        return _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()

    row_uses_dense = _residual_mask_for_input(input_tensor)
    if row_uses_dense is None:
        return F.linear(input_tensor, module.weight, bias).contiguous()
    row_uses_dense = row_uses_dense.to(device=input_tensor.device, dtype=torch.bool)

    dense_rows = row_uses_dense.nonzero(as_tuple=False).squeeze(1)
    base_rows = (~row_uses_dense).nonzero(as_tuple=False).squeeze(1)
    out_features = int(getattr(module, "_speclink_sr24_weight_shape")[0])
    output = torch.empty(
        (int(input_tensor.shape[0]), out_features),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    if int(base_rows.numel()) > 0:
        base_input = input_tensor.index_select(0, base_rows)
        output.index_copy_(0, base_rows, F.linear(base_input, module.weight, bias))
    if int(dense_rows.numel()) > 0:
        dense_input = input_tensor.index_select(0, dense_rows)
        output.index_copy_(
            0,
            dense_rows,
            _sr24_dense_linear(dense_input, dense_weight, bias),
        )
    return output.contiguous()


@torch.library.custom_op("speclink_sr24::cslt_linear", mutates_args=())
def _opaque_cslt_linear(
    packed: torch.Tensor,
    input_tensor: torch.Tensor,
    transpose_result: bool,
    alg_id: int,
) -> torch.Tensor:
    del transpose_result
    rows = int(input_tensor.shape[0])
    dense_input = input_tensor
    row_multiple = 8
    to_pad_m = (
        -rows % row_multiple if rows < row_multiple or rows % row_multiple else 0
    )
    if to_pad_m:
        dense_input = F.pad(dense_input, (0, 0, 0, to_pad_m))
    if not dense_input.is_contiguous():
        dense_input = dense_input.contiguous()
    result = torch._cslt_sparse_mm(
        packed,
        dense_input.t().contiguous(),
        transpose_result=True,
        alg_id=alg_id,
    )
    return result[:rows].contiguous()


@_opaque_cslt_linear.register_fake
def _opaque_cslt_linear_fake(
    packed: torch.Tensor,
    input_tensor: torch.Tensor,
    transpose_result: bool,
    alg_id: int,
) -> torch.Tensor:
    del transpose_result, alg_id
    return input_tensor.new_empty((input_tensor.shape[0], packed.shape[0]))


@torch.library.custom_op("speclink_sr24::dense_linear", mutates_args=())
def _opaque_dense_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return F.linear(input_tensor, weight, bias=None).contiguous()


@_opaque_dense_linear.register_fake
def _opaque_dense_linear_fake(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return input_tensor.new_empty((input_tensor.shape[0], weight.shape[0]))


def _sr24_dense_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is None and direct_cslt_linear():
        return _opaque_dense_linear(input_tensor, weight)
    return F.linear(input_tensor, weight, bias)


def _mark_exact_dense_output(output: torch.Tensor) -> torch.Tensor:
    try:
        output._speclink_sr24_exact_dense = True
    except Exception:
        pass
    return output


def _is_exact_dense_output(output: torch.Tensor) -> bool:
    return bool(getattr(output, "_speclink_sr24_exact_dense", False))


@torch.inference_mode()
def _sr24_exact_dense_linear(
    module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the original vLLM Linear when SR24 needs a full dense row set.

    `dense_rows` keeps the original module weight resident and stores the 2:4
    base separately. In that case, calling the module's own forward matches the
    dense baseline's Linear implementation more closely than an out-of-band
    `F.linear` call, and lets residual postprocess skip a second dense GEMM.
    """
    module_weight = getattr(module, "weight", None)
    if (
        getattr(module, "_speclink_sr24_residual_backend", "") == "dense_rows"
        and isinstance(module_weight, torch.Tensor)
        and module_weight is dense_weight
    ):
        try:
            module_output = module(input_tensor)
            output = module_output[0] if isinstance(module_output, tuple) else module_output
            return _mark_exact_dense_output(output.contiguous())
        except Exception:
            pass
    return _mark_exact_dense_output(
        _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
    )


@torch.inference_mode()
def _semi_structured_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not direct_cslt_linear():
        return F.linear(input_tensor, weight, bias)
    packed = getattr(weight, "packed", None)
    if packed is None:
        raise RuntimeError(
            "SPECLINK_SR24_DIRECT_CSLT_LINEAR requires a cuSPARSELt "
            "semi-structured weight with a packed representation; refusing "
            "to fall back to dense F.linear on a sparse weight."
        )
    if bias is None:
        return _opaque_cslt_linear(
            packed,
            input_tensor,
            bool(getattr(weight, "fuse_transpose_cusparselt", False)),
            _cslt_alg_id_for_rows(weight, int(input_tensor.shape[0])),
        )
    rows = int(input_tensor.shape[0])
    dense_input = input_tensor
    row_multiple = 8
    to_pad_m = (
        -rows % row_multiple if rows < row_multiple or rows % row_multiple else 0
    )
    if to_pad_m:
        dense_input = F.pad(dense_input, (0, 0, 0, to_pad_m))
    if not dense_input.is_contiguous():
        dense_input = dense_input.contiguous()
    fuse_transpose = bool(getattr(weight, "fuse_transpose_cusparselt", False))
    result = torch._cslt_sparse_mm(
        packed,
        dense_input.t().contiguous(),
        bias=bias,
        transpose_result=fuse_transpose,
        alg_id=_cslt_alg_id_for_rows(weight, int(input_tensor.shape[0])),
    )
    output = result if fuse_transpose else result.t()
    if int(output.shape[0]) != rows:
        output = output[:rows]
    return output


@triton.jit
def _routed_assemble_kernel(
    dense_output,
    dense_rows,
    base_output,
    base_rows,
    output,
    dense_count: tl.constexpr,
    total_rows: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    col_mask = cols < out_features
    is_dense = row < dense_count
    base_row = row - dense_count
    dense_dst_row = tl.load(dense_rows + row, mask=is_dense, other=0)
    base_dst_row = tl.load(base_rows + base_row, mask=~is_dense, other=0)
    dst_row = tl.where(is_dense, dense_dst_row, base_dst_row)
    dense_vals = tl.load(
        dense_output + row * out_features + cols,
        mask=is_dense & col_mask,
        other=0.0,
    )
    base_vals = tl.load(
        base_output + base_row * out_features + cols,
        mask=(~is_dense) & col_mask,
        other=0.0,
    )
    tl.store(
        output + dst_row * out_features + cols,
        dense_vals + base_vals,
        mask=col_mask,
    )


@torch.inference_mode()
def _triton_routed_assemble(
    dense_output: torch.Tensor,
    dense_rows: torch.Tensor,
    base_output: torch.Tensor,
    base_rows: torch.Tensor,
    *,
    total_rows: int,
    out_features: int,
) -> torch.Tensor:
    dense_output = dense_output.contiguous()
    base_output = base_output.contiguous()
    dense_rows = dense_rows.contiguous()
    base_rows = base_rows.contiguous()
    output = torch.empty(
        (total_rows, out_features),
        dtype=dense_output.dtype,
        device=dense_output.device,
    )
    dense_count = int(dense_output.shape[0])
    base_count = int(base_output.shape[0])
    if dense_count + base_count != int(total_rows):
        raise RuntimeError(
            "SR24 routed assembly row mismatch: "
            f"dense={dense_count}, base={base_count}, total={total_rows}"
        )
    block_n = 1024
    grid = (int(total_rows), triton.cdiv(int(out_features), block_n))
    _routed_assemble_kernel[grid](
        dense_output,
        dense_rows,
        base_output,
        base_rows,
        output,
        dense_count,
        int(total_rows),
        int(out_features),
        block_n,
        num_warps=8,
    )
    return output


@triton.jit
def _fixed_block_assemble_kernel(
    dense_output,
    base_output,
    output,
    active_count: tl.constexpr,
    scheduled_width: tl.constexpr,
    prefix: tl.constexpr,
    valid_width: tl.constexpr,
    base_width: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    col_mask = cols < out_features

    request_idx = row // scheduled_width
    pos = row - request_idx * scheduled_width
    is_prefix = pos < prefix
    is_bonus = pos == valid_width
    dense_prefix_offset = request_idx * prefix + pos
    dense_bonus_offset = active_count * prefix + request_idx
    dense_offset = tl.where(is_prefix, dense_prefix_offset, dense_bonus_offset)
    base_offset = request_idx * base_width + (pos - prefix)

    dense_vals = tl.load(
        dense_output + dense_offset * out_features + cols,
        mask=(is_prefix | is_bonus) & col_mask,
        other=0.0,
    )
    base_vals = tl.load(
        base_output + base_offset * out_features + cols,
        mask=(~(is_prefix | is_bonus)) & col_mask,
        other=0.0,
    )
    tl.store(
        output + row * out_features + cols,
        dense_vals + base_vals,
        mask=col_mask,
    )


@torch.inference_mode()
def _triton_fixed_block_assemble(
    dense_down: torch.Tensor,
    base_down: torch.Tensor,
    *,
    active_count: int,
    scheduled_width: int,
    prefix: int,
    valid_width: int,
    base_width: int,
    out_features: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    dense_down = dense_down.contiguous()
    base_down = base_down.contiguous()
    active_count = int(active_count)
    scheduled_width = int(scheduled_width)
    prefix = int(prefix)
    valid_width = int(valid_width)
    base_width = int(base_width)
    out_features = int(out_features)
    total_rows = active_count * scheduled_width
    expected_dense = active_count * (prefix + 1)
    expected_base = active_count * base_width
    if int(dense_down.shape[0]) != expected_dense or int(
            base_down.shape[0]) != expected_base:
        raise RuntimeError(
            "SR24 fixed-block assembly row mismatch: "
            f"dense={int(dense_down.shape[0])}/{expected_dense}, "
            f"base={int(base_down.shape[0])}/{expected_base}"
        )
    if output is None:
        output = torch.empty(
            (total_rows, out_features),
            dtype=dense_down.dtype,
            device=dense_down.device,
        )
    elif (
        tuple(output.shape) != (total_rows, out_features)
        or output.dtype != dense_down.dtype
        or output.device != dense_down.device
    ):
        raise RuntimeError(
            "SR24 fixed-block assembly output mismatch: "
            f"shape={tuple(output.shape)}/{(total_rows, out_features)}, "
            f"dtype={output.dtype}/{dense_down.dtype}, "
            f"device={output.device}/{dense_down.device}"
        )
    block_n = 1024
    grid = (total_rows, triton.cdiv(out_features, block_n))
    _fixed_block_assemble_kernel[grid](
        dense_down,
        base_down,
        output,
        active_count,
        scheduled_width,
        prefix,
        valid_width,
        base_width,
        out_features,
        block_n,
        num_warps=8,
    )
    return output


@triton.jit
def _bucket_override_kernel(
    dense_output,
    bucket_rows,
    bucket_values,
    output,
    bucket_count: tl.constexpr,
    out_features: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    col_mask = cols < out_features
    use_dense = tl.load(bucket_values + row, mask=row < bucket_count, other=0) != 0
    dst_row = tl.load(bucket_rows + row, mask=row < bucket_count, other=0)
    vals = tl.load(
        dense_output + row * out_features + cols,
        mask=(row < bucket_count) & use_dense & col_mask,
        other=0.0,
    )
    tl.store(
        output + dst_row * out_features + cols,
        vals,
        mask=(row < bucket_count) & use_dense & col_mask,
    )


@torch.inference_mode()
def _triton_bucket_override_inplace(
    base_output: torch.Tensor,
    dense_output: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
) -> torch.Tensor:
    base_output = base_output.contiguous()
    dense_output = dense_output.contiguous()
    bucket_rows = bucket_rows.contiguous()
    bucket_values = bucket_values.to(dtype=torch.bool).contiguous()
    out_features = int(base_output.shape[1])
    bucket_count = int(bucket_rows.numel())
    if bucket_count <= 0:
        return base_output
    block_n = 1024
    grid = (bucket_count, triton.cdiv(out_features, block_n))
    _bucket_override_kernel[grid](
        dense_output,
        bucket_rows,
        bucket_values,
        base_output,
        bucket_count,
        out_features,
        block_n,
        num_warps=8,
    )
    return base_output


@triton.jit
def _bucket_dense_gemm_scatter_kernel(
    input_ptr,
    weight_ptr,
    bucket_rows_ptr,
    bucket_values_ptr,
    output_ptr,
    bucket_count: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    input_stride_m: tl.constexpr,
    input_stride_k: tl.constexpr,
    weight_stride_n: tl.constexpr,
    weight_stride_k: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    bucket_valid = offs_m < bucket_count
    bucket_rows = tl.load(bucket_rows_ptr + offs_m, mask=bucket_valid, other=0)
    bucket_values = tl.load(
        bucket_values_ptr + offs_m,
        mask=bucket_valid,
        other=0,
    )
    row_valid = bucket_valid & (bucket_values != 0)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in range(0, in_features, block_k):
        offs_k = k_start + tl.arange(0, block_k)
        k_mask = offs_k < in_features
        x = tl.load(
            input_ptr
            + bucket_rows[:, None] * input_stride_m
            + offs_k[None, :] * input_stride_k,
            mask=row_valid[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + offs_n[:, None] * weight_stride_n
            + offs_k[None, :] * weight_stride_k,
            mask=(offs_n[:, None] < out_features) & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        output_ptr
        + bucket_rows[:, None] * output_stride_m
        + offs_n[None, :] * output_stride_n,
        acc,
        mask=row_valid[:, None] & (offs_n[None, :] < out_features),
    )


@triton.jit
def _bucket_dense_scatter_kernel(
    dense_output_ptr,
    bucket_rows_ptr,
    bucket_values_ptr,
    output_ptr,
    bucket_count: tl.constexpr,
    out_features: tl.constexpr,
    dense_stride_m: tl.constexpr,
    dense_stride_n: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    bucket_valid = offs_m < bucket_count
    bucket_rows = tl.load(bucket_rows_ptr + offs_m, mask=bucket_valid, other=0)
    bucket_values = tl.load(
        bucket_values_ptr + offs_m,
        mask=bucket_valid,
        other=0,
    )
    row_valid = bucket_valid & (bucket_values != 0)
    values = tl.load(
        dense_output_ptr
        + offs_m[:, None] * dense_stride_m
        + offs_n[None, :] * dense_stride_n,
        mask=bucket_valid[:, None] & (offs_n[None, :] < out_features),
        other=0.0,
    )
    tl.store(
        output_ptr
        + bucket_rows[:, None] * output_stride_m
        + offs_n[None, :] * output_stride_n,
        values,
        mask=row_valid[:, None] & (offs_n[None, :] < out_features),
    )


@torch.inference_mode()
def _triton_bucket_dense_scatter_inplace(
    dense_output: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
    base_output: torch.Tensor,
) -> bool:
    if not triton_bucket_scatter():
        return False
    if not (
        dense_output.is_cuda
        and bucket_rows.is_cuda
        and base_output.is_cuda
    ):
        return False
    if dense_output.ndim != 2 or base_output.ndim != 2:
        return False
    if dense_output.dtype != base_output.dtype:
        return False
    bucket_count = int(bucket_rows.numel())
    if bucket_count <= 0:
        return True
    if int(dense_output.shape[0]) != bucket_count:
        return False
    out_features = int(base_output.shape[1])
    if int(dense_output.shape[1]) != out_features:
        return False
    dense_contig = dense_output if dense_output.is_contiguous() else dense_output.contiguous()
    output_contig = base_output if base_output.is_contiguous() else base_output.contiguous()
    rows_contig = bucket_rows.to(
        device=base_output.device,
        dtype=torch.long,
        non_blocking=True,
    ).contiguous()
    values_contig = bucket_values.to(
        device=base_output.device,
        dtype=torch.bool,
        non_blocking=True,
    ).contiguous()
    block_m = triton_bucket_dense_block_m()
    block_n = triton_bucket_dense_block_n()
    grid = (
        triton.cdiv(bucket_count, block_m),
        triton.cdiv(out_features, block_n),
    )
    _bucket_dense_scatter_kernel[grid](
        dense_contig,
        rows_contig,
        values_contig,
        output_contig,
        bucket_count,
        out_features,
        int(dense_contig.stride(0)),
        int(dense_contig.stride(1)),
        int(output_contig.stride(0)),
        int(output_contig.stride(1)),
        block_m,
        block_n,
        num_warps=4,
        num_stages=3,
    )
    if output_contig.data_ptr() != base_output.data_ptr():
        base_output.copy_(output_contig)
    return True


@torch.inference_mode()
def _triton_bucket_dense_gemm_scatter_inplace(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
    base_output: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    force_all_bucket_rows: bool = False,
) -> bool:
    if not triton_bucket_dense_gemm():
        return False
    if bias is not None:
        return False
    if not (
        input_tensor.is_cuda
        and dense_weight.is_cuda
        and bucket_rows.is_cuda
        and base_output.is_cuda
    ):
        return False
    if input_tensor.ndim != 2 or dense_weight.ndim != 2 or base_output.ndim != 2:
        return False
    if input_tensor.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if dense_weight.dtype != input_tensor.dtype or base_output.dtype != input_tensor.dtype:
        return False
    bucket_count = int(bucket_rows.numel())
    if bucket_count <= 0:
        return True
    in_features = int(input_tensor.shape[1])
    out_features = int(dense_weight.shape[0])
    if int(dense_weight.shape[1]) != in_features:
        return False
    if int(base_output.shape[1]) != out_features:
        return False
    input_contig = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
    weight_contig = dense_weight if dense_weight.is_contiguous() else dense_weight.contiguous()
    output_contig = base_output if base_output.is_contiguous() else base_output.contiguous()
    bucket_rows_contig = bucket_rows.to(
        device=input_tensor.device,
        dtype=torch.long,
        non_blocking=True,
    ).contiguous()
    bucket_values_contig = bucket_values.to(
        device=input_tensor.device,
        dtype=torch.bool,
        non_blocking=True,
    ).contiguous()
    if force_all_bucket_rows:
        # Match the quality-safe bucket_dense_copy path: every selected bucket
        # row is overwritten with the dense result, including conservative
        # bucket padding rows whose original residual-mask value is false.
        bucket_values_contig = torch.ones_like(bucket_values_contig)
    block_m = triton_bucket_dense_block_m()
    block_n = triton_bucket_dense_block_n()
    block_k = triton_bucket_dense_block_k()
    grid = (
        triton.cdiv(bucket_count, block_m),
        triton.cdiv(out_features, block_n),
    )
    _bucket_dense_gemm_scatter_kernel[grid](
        input_contig,
        weight_contig,
        bucket_rows_contig,
        bucket_values_contig,
        output_contig,
        bucket_count,
        in_features,
        out_features,
        int(input_contig.stride(0)),
        int(input_contig.stride(1)),
        int(weight_contig.stride(0)),
        int(weight_contig.stride(1)),
        int(output_contig.stride(0)),
        int(output_contig.stride(1)),
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )
    if output_contig.data_ptr() != base_output.data_ptr():
        base_output.copy_(output_contig)
    return True


@torch.inference_mode()
def _bucket_dense_overwrite_inplace(
    base_output: torch.Tensor,
    dense_output: torch.Tensor,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
) -> torch.Tensor:
    """Overwrite corrected bucket rows with exact dense output.

    The older delta-add path computes `base + (dense - base)` for corrected
    rows. In bf16/fp16 that is not bitwise equivalent to the dense verifier
    row and can perturb reasoning outputs when a selective policy requests all
    rows as residual. This path preserves inactive bucket rows with a vectorized
    select and writes active rows as the already-computed dense output.
    """
    active = bucket_values.to(
        device=base_output.device,
        dtype=torch.bool,
        non_blocking=True,
    ).unsqueeze(1)
    local_bucket_rows = bucket_rows.to(
        device=base_output.device,
        dtype=torch.long,
        non_blocking=True,
    )
    base_rows = base_output.index_select(0, local_bucket_rows)
    overwrite = torch.where(active, dense_output, base_rows)
    base_output.index_copy_(0, local_bucket_rows, overwrite)
    return base_output


@torch.inference_mode()
def _routed_dense_rows_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bias: torch.Tensor | None,
    dense_rows: torch.Tensor,
    *,
    route_label: str = "routed_dense_rows",
    base_rows: torch.Tensor | None = None,
) -> torch.Tensor | None:
    rows = int(input_tensor.shape[0])
    if rows <= 0:
        return input_tensor.new_empty((0, int(dense_weight.shape[0])))
    dense_rows = dense_rows.to(device=input_tensor.device, dtype=torch.long)
    dense_count = int(dense_rows.numel())
    if dense_count <= 0:
        return None
    if dense_count >= rows:
        dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
        _breakdown_record_cuda_module(
            module, f"{route_label}_all_dense_gemm_cuda_ms", dense_timer
        )
        if breakdown_linear_enabled():
            _breakdown_count_module(module, f"{route_label}_calls", 1)
            _breakdown_count_module(module, f"{route_label}_rows", rows)
            _breakdown_count_module(module, f"{route_label}_dense_rows", rows)
            _breakdown_count_module(module, f"{route_label}_base_rows", 0)
        return output

    linear_breakdown = breakdown_linear_enabled()
    base_count = rows - dense_count
    min_dense = route_min_dense_rows()
    if dense_count < min_dense:
        if linear_breakdown:
            _breakdown_count_module(
                module, f"{route_label}_skipped_small_dense_rows", 1
            )
            _breakdown_count_module(
                module, f"{route_label}_skipped_dense_rows", dense_count
            )
            _breakdown_count_module(
                module, f"{route_label}_min_dense_rows", min_dense
            )
        return None

    dense_fraction = dense_count / max(rows, 1)
    max_dense_fraction = route_max_dense_fraction()
    if 0.0 <= max_dense_fraction <= 1.0 and dense_fraction > max_dense_fraction:
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
        _breakdown_record_cuda_module(
            module,
            f"{route_label}_fallback_max_dense_fraction_gemm_cuda_ms",
            dense_timer,
        )
        if linear_breakdown:
            _breakdown_count_module(
                module, f"{route_label}_fallback_max_dense_fraction_calls", 1
            )
            _breakdown_count_module(module, f"{route_label}_calls", 1)
            _breakdown_count_module(module, f"{route_label}_rows", rows)
            _breakdown_count_module(module, f"{route_label}_dense_rows", dense_count)
            _breakdown_count_module(module, f"{route_label}_base_rows", base_count)
        return output

    module_leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    min_base = route_min_base_rows_for_leaf(module_leaf)
    if min_base > 0 and base_count < min_base:
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
        _breakdown_record_cuda_module(
            module,
            f"{route_label}_fallback_small_base_gemm_cuda_ms",
            dense_timer,
        )
        if linear_breakdown:
            _breakdown_count_module(
                module, f"{route_label}_fallback_small_base_calls", 1
            )
            _breakdown_count_module(module, f"{route_label}_calls", 1)
            _breakdown_count_module(module, f"{route_label}_rows", rows)
            _breakdown_count_module(module, f"{route_label}_dense_rows", dense_count)
            _breakdown_count_module(module, f"{route_label}_base_rows", base_count)
            _breakdown_count_module(module, f"{route_label}_min_base_rows", min_base)
            _breakdown_count_module(
                module,
                f"{route_label}_min_base_rows_by_leaf_enabled",
                int(module_leaf in _leaf_int_map_from_env(
                    "SPECLINK_SR24_ROUTE_MIN_BASE_ROWS_BY_LEAF"
                )),
            )
        return output

    fallback_fraction = route_dense_fallback_fraction()
    if 0.0 <= fallback_fraction <= 1.0:
        if dense_fraction >= fallback_fraction:
            dense_timer = _breakdown_cuda_start() if linear_breakdown else None
            output = _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
            _breakdown_record_cuda_module(
                module,
                f"{route_label}_fallback_dense_gemm_cuda_ms",
                dense_timer,
            )
            if linear_breakdown:
                _breakdown_count_module(
                    module, f"{route_label}_fallback_dense_calls", 1
                )
                _breakdown_count_module(module, f"{route_label}_calls", 1)
                _breakdown_count_module(module, f"{route_label}_rows", rows)
                _breakdown_count_module(
                    module,
                    f"{route_label}_dense_rows",
                    dense_count,
                )
                _breakdown_count_module(
                    module,
                    f"{route_label}_base_rows",
                    base_count,
                )
            return output

    if linear_breakdown:
        _breakdown_count_module(module, f"{route_label}_calls", 1)
        _breakdown_count_module(module, f"{route_label}_rows", rows)
        _breakdown_count_module(
            module, f"{route_label}_dense_rows", dense_count
        )

    if base_rows is not None:
        base_rows = base_rows.to(device=input_tensor.device, dtype=torch.long)
        if int(base_rows.numel()) + dense_count != rows:
            base_rows = None
    if base_rows is None:
        route_timer = _breakdown_cuda_start() if linear_breakdown else None
        route_dense = torch.zeros(rows, dtype=torch.bool, device=input_tensor.device)
        route_dense.index_fill_(0, dense_rows, True)
        base_rows = (~route_dense).nonzero(as_tuple=False).squeeze(1)
        _breakdown_record_cuda_module(
            module, f"{route_label}_route_build_cuda_ms", route_timer
        )
    elif linear_breakdown:
        _breakdown_count_module(module, f"{route_label}_cached_base_rows_hits", 1)
    if linear_breakdown:
        _breakdown_count_module(
            module, f"{route_label}_base_rows", int(base_rows.numel())
        )
    out_features = int(dense_weight.shape[0])
    contiguous_output = _routed_contiguous_rows_output(
        module,
        input_tensor,
        dense_weight,
        bias,
        dense_rows,
        route_label=route_label,
    )
    if contiguous_output is not None:
        return contiguous_output
    if (
        route_overlap_streams()
        and input_tensor.is_cuda
        and int(base_rows.numel()) > 0
        and int(dense_rows.numel()) > 0
    ):
        output = torch.empty(
            (rows, out_features),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        current_stream = torch.cuda.current_stream(input_tensor.device)
        base_stream, dense_stream = _route_overlap_streams_for_device(
            input_tensor.device
        )
        base_stream.wait_stream(current_stream)
        dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(base_stream):
            base_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
            base_input = input_tensor.index_select(0, base_rows)
            _breakdown_record_cuda(
                f"{route_label}_overlap_base_gather_cuda_ms",
                base_gather_timer,
            )
            base_timer = _breakdown_cuda_start() if linear_breakdown else None
            base_output = _semi_structured_linear(
                base_input, _sparse_base_weight(module), bias
            )
            _breakdown_record_cuda_module(
                module, f"{route_label}_overlap_base_sparse_gemm_cuda_ms", base_timer
            )
            scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
            output.index_copy_(0, base_rows, base_output)
            _breakdown_record_cuda(
                f"{route_label}_overlap_base_index_copy_cuda_ms",
                scatter_timer,
            )
        with torch.cuda.stream(dense_stream):
            dense_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
            dense_input = input_tensor.index_select(0, dense_rows)
            _breakdown_record_cuda(
                f"{route_label}_overlap_dense_gather_cuda_ms",
                dense_gather_timer,
            )
            dense_timer = _breakdown_cuda_start() if linear_breakdown else None
            dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
            _breakdown_record_cuda_module(
                module, f"{route_label}_overlap_dense_gemm_cuda_ms", dense_timer
            )
            scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
            output.index_copy_(0, dense_rows, dense_output)
            _breakdown_record_cuda(
                f"{route_label}_overlap_dense_index_copy_cuda_ms",
                scatter_timer,
            )
        current_stream.wait_stream(base_stream)
        current_stream.wait_stream(dense_stream)
        if linear_breakdown:
            _breakdown_count_module(module, f"{route_label}_overlap_stream_calls", 1)
            _breakdown_count_module(
                module, f"{route_label}_overlap_cached_stream_calls", 1
            )
        return output.contiguous()
    if triton_route_assembly():
        dense_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_input = input_tensor.index_select(0, dense_rows)
        _breakdown_record_cuda(
            f"{route_label}_dense_gather_cuda_ms",
            dense_gather_timer,
        )
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
        _breakdown_record_cuda_module(
            module, f"{route_label}_dense_gemm_cuda_ms", dense_timer
        )
        if int(base_rows.numel()) > 0:
            base_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
            base_input = input_tensor.index_select(0, base_rows)
            _breakdown_record_cuda(
                f"{route_label}_base_gather_cuda_ms",
                base_gather_timer,
            )
            base_timer = _breakdown_cuda_start() if linear_breakdown else None
            base_output = _semi_structured_linear(
                base_input, _sparse_base_weight(module), bias
            )
            _breakdown_record_cuda_module(
                module,
                f"{route_label}_base_sparse_gemm_cuda_ms",
                base_timer,
            )
        else:
            base_output = torch.empty(
                (0, out_features),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
        assemble_timer = _breakdown_cuda_start() if linear_breakdown else None
        output = _triton_routed_assemble(
            dense_output,
            dense_rows,
            base_output,
            base_rows,
            total_rows=rows,
            out_features=out_features,
        )
        _breakdown_record_cuda(f"{route_label}_triton_assemble_cuda_ms", assemble_timer)
        return output

    output = torch.empty(
        (rows, out_features),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    if int(base_rows.numel()) > 0:
        base_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_input = input_tensor.index_select(0, base_rows)
        _breakdown_record_cuda(
            f"{route_label}_base_gather_cuda_ms",
            base_gather_timer,
        )
        base_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_output = _semi_structured_linear(
            base_input, _sparse_base_weight(module), bias
        )
        _breakdown_record_cuda_module(
            module, f"{route_label}_base_sparse_gemm_cuda_ms", base_timer
        )
        scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
        output.index_copy_(0, base_rows, base_output)
        _breakdown_record_cuda(f"{route_label}_base_index_copy_cuda_ms", scatter_timer)
    if int(dense_rows.numel()) > 0:
        dense_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_input = input_tensor.index_select(0, dense_rows)
        _breakdown_record_cuda(
            f"{route_label}_dense_gather_cuda_ms",
            dense_gather_timer,
        )
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
        _breakdown_record_cuda_module(
            module, f"{route_label}_dense_gemm_cuda_ms", dense_timer
        )
        scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
        output.index_copy_(0, dense_rows, dense_output)
        _breakdown_record_cuda(f"{route_label}_dense_index_copy_cuda_ms", scatter_timer)
    return output.contiguous()


@torch.inference_mode()
def _routed_bucket_dense_rows_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bias: torch.Tensor | None,
    bucket_rows: torch.Tensor,
    bucket_values: torch.Tensor,
) -> torch.Tensor | None:
    if bucket_dense_copy() and not bucket_dense_copy_active_only():
        # Match the quality-safe bucket dense-copy semantics: all selected
        # bucket rows are computed by the dense path, including conservative
        # inactive bucket entries. This keeps route-bucket routing comparable
        # to the normal bucket_dense_copy correction path.
        dense_rows = bucket_rows
    else:
        dense_row_values = bucket_values.to(dtype=torch.bool)
        dense_rows = bucket_rows[dense_row_values]
    return _routed_dense_rows_output(
        module,
        input_tensor,
        dense_weight,
        bias,
        dense_rows,
        route_label="route_bucket_dense_rows",
    )


def _rows_equal_arange(
    rows_tensor: torch.Tensor,
    *,
    start: int,
    length: int,
    device: torch.device,
) -> bool:
    if length <= 0:
        return int(rows_tensor.numel()) == 0
    if int(rows_tensor.numel()) != length:
        return False
    expected = _device_arange(length, dtype=torch.long, device=device)
    if start:
        expected = expected + int(start)
    try:
        return bool(torch.equal(rows_tensor, expected))
    except Exception:
        return False


@torch.inference_mode()
def _routed_contiguous_rows_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bias: torch.Tensor | None,
    dense_rows: torch.Tensor,
    *,
    route_label: str,
) -> torch.Tensor | None:
    if not route_contiguous_fastpath():
        return None
    rows = int(input_tensor.shape[0])
    dense_count = int(dense_rows.numel())
    if rows <= 0 or dense_count <= 0 or dense_count >= rows:
        return None
    device = input_tensor.device
    linear_breakdown = breakdown_linear_enabled()
    dense_is_prefix = _rows_equal_arange(
        dense_rows,
        start=0,
        length=dense_count,
        device=device,
    )
    dense_is_suffix = False
    if not dense_is_prefix:
        dense_is_suffix = _rows_equal_arange(
            dense_rows,
            start=rows - dense_count,
            length=dense_count,
            device=device,
        )
    if not dense_is_prefix and not dense_is_suffix:
        if linear_breakdown:
            _breakdown_count_module(
                module, f"{route_label}_contiguous_fastpath_miss", 1
            )
        return None

    if linear_breakdown:
        _breakdown_count_module(module, f"{route_label}_contiguous_fastpath_hits", 1)
        _breakdown_count_module(module, f"{route_label}_contiguous_dense_rows", dense_count)
        _breakdown_count_module(
            module, f"{route_label}_contiguous_base_rows", rows - dense_count
        )
        if dense_is_prefix:
            _breakdown_count_module(module, f"{route_label}_contiguous_prefix_hits", 1)
        else:
            _breakdown_count_module(module, f"{route_label}_contiguous_suffix_hits", 1)

    dense_timer = _breakdown_cuda_start() if linear_breakdown else None
    if dense_is_prefix:
        dense_output = _sr24_dense_linear(
            input_tensor[:dense_count],
            dense_weight,
            bias,
        )
    else:
        dense_output = _sr24_dense_linear(
            input_tensor[rows - dense_count:],
            dense_weight,
            bias,
        )
    _breakdown_record_cuda_module(
        module, f"{route_label}_contiguous_dense_gemm_cuda_ms", dense_timer
    )

    base_timer = _breakdown_cuda_start() if linear_breakdown else None
    if dense_is_prefix:
        base_output = _semi_structured_linear(
            input_tensor[dense_count:],
            _sparse_base_weight(module),
            bias,
        )
    else:
        base_output = _semi_structured_linear(
            input_tensor[:rows - dense_count],
            _sparse_base_weight(module),
            bias,
        )
    _breakdown_record_cuda_module(
        module, f"{route_label}_contiguous_base_sparse_gemm_cuda_ms", base_timer
    )

    cat_timer = _breakdown_cuda_start() if linear_breakdown else None
    if dense_is_prefix:
        output = torch.cat([dense_output, base_output], dim=0)
    else:
        output = torch.cat([base_output, dense_output], dim=0)
    _breakdown_record_cuda(
        f"{route_label}_contiguous_cat_cuda_ms",
        cat_timer,
    )
    return output.contiguous()


@torch.inference_mode()
def _row_routed_mlp_contiguous_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    down_dense_weight: torch.Tensor | None,
    dense_rows: torch.Tensor | None,
    base_rows: torch.Tensor | None,
    *,
    down_uses_exact_dense_rows: bool,
    act_fn: Any | None,
) -> torch.Tensor | None:
    if not route_contiguous_fastpath():
        return None
    if input_tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        # The contiguous fastpath deliberately ignores row-index tensors after
        # it has proven a prefix/suffix split. During CUDA Graph replay those
        # Python checks do not run again, while the persistent bucket rows may
        # contain a different dynamic route. Use the generic row-indexed graph
        # path unless a future scheduler can provide a static prefix contract.
        return None
    rows = int(input_tensor.shape[0])
    dense_count = int(dense_rows.numel())
    if rows <= 0 or dense_count <= 0 or dense_count >= rows:
        return None
    if base_rows is None or int(base_rows.numel()) + dense_count != rows:
        return None

    device = input_tensor.device
    base_count = rows - dense_count
    dense_is_prefix = _rows_equal_arange(
        dense_rows,
        start=0,
        length=dense_count,
        device=device,
    )
    base_is_suffix = False
    dense_is_suffix = False
    base_is_prefix = False
    if dense_is_prefix:
        base_is_suffix = _rows_equal_arange(
            base_rows,
            start=dense_count,
            length=base_count,
            device=device,
        )
    else:
        dense_is_suffix = _rows_equal_arange(
            dense_rows,
            start=base_count,
            length=dense_count,
            device=device,
        )
        if dense_is_suffix:
            base_is_prefix = _rows_equal_arange(
                base_rows,
                start=0,
                length=base_count,
                device=device,
            )
    if not ((dense_is_prefix and base_is_suffix)
            or (dense_is_suffix and base_is_prefix)):
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_mlp_contiguous_fastpath_miss", 1)
        return None

    linear_breakdown = breakdown_linear_enabled()
    if linear_breakdown:
        _breakdown_count("row_routed_mlp_contiguous_fastpath_hits", 1)
        _breakdown_count("row_routed_mlp_contiguous_dense_rows", dense_count)
        _breakdown_count("row_routed_mlp_contiguous_base_rows", base_count)
        _breakdown_count(
            "row_routed_mlp_contiguous_prefix_hits"
            if dense_is_prefix else "row_routed_mlp_contiguous_suffix_hits",
            1,
        )

    if dense_is_prefix:
        dense_input = input_tensor[:dense_count]
        base_input = input_tensor[dense_count:]
    else:
        dense_input = input_tensor[base_count:]
        base_input = input_tensor[:base_count]

    dense_timer = _breakdown_cuda_start() if linear_breakdown else None
    dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
    _breakdown_record_cuda("row_routed_mlp_contiguous_dense_gate_up_cuda_ms",
                           dense_timer)

    base_timer = _breakdown_cuda_start() if linear_breakdown else None
    base_gate_up = _semi_structured_linear(
        base_input, _sparse_base_weight(gate_up_module), None
    )
    _breakdown_record_cuda("row_routed_mlp_contiguous_base_gate_up_cuda_ms",
                           base_timer)

    intermediate_size = int(getattr(down_module, "_speclink_sr24_weight_shape")[1])
    dense_act_timer = _breakdown_cuda_start() if linear_breakdown else None
    dense_act = act_fn(dense_gate_up) if act_fn is not None else _silu_and_mul_local(
        dense_gate_up,
        intermediate_size,
    )
    _breakdown_record_cuda("row_routed_mlp_contiguous_dense_act_cuda_ms",
                           dense_act_timer)

    base_act_timer = _breakdown_cuda_start() if linear_breakdown else None
    base_act = act_fn(base_gate_up) if act_fn is not None else _silu_and_mul_local(
        base_gate_up,
        intermediate_size,
    )
    _breakdown_record_cuda("row_routed_mlp_contiguous_base_act_cuda_ms",
                           base_act_timer)

    dense_down_timer = _breakdown_cuda_start() if linear_breakdown else None
    if down_uses_exact_dense_rows:
        if down_dense_weight is None:
            return None
        dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
    else:
        dense_down = _semi_structured_linear(
            dense_act, _sparse_base_weight(down_module), None
        )
    _breakdown_record_cuda("row_routed_mlp_contiguous_dense_down_cuda_ms",
                           dense_down_timer)

    base_down_timer = _breakdown_cuda_start() if linear_breakdown else None
    base_down = _semi_structured_linear(
        base_act, _sparse_base_weight(down_module), None
    )
    _breakdown_record_cuda("row_routed_mlp_contiguous_base_down_cuda_ms",
                           base_down_timer)

    cat_timer = _breakdown_cuda_start() if linear_breakdown else None
    if dense_is_prefix:
        output = torch.cat([dense_down, base_down], dim=0)
    else:
        output = torch.cat([base_down, dense_down], dim=0)
    _breakdown_record_cuda("row_routed_mlp_contiguous_cat_cuda_ms", cat_timer)
    return output.contiguous()


@torch.inference_mode()
def _row_routed_mlp_overlap_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    down_dense_weight: torch.Tensor | None,
    dense_rows: torch.Tensor,
    base_rows: torch.Tensor,
    *,
    down_uses_exact_dense_rows: bool,
    act_fn: Any | None,
) -> torch.Tensor | None:
    """Run row-routed dense and sparse MLP branches on separate streams.

    This is an explicit SR24 ablation path. The branch shapes are independent:
    dense rows use dense gate_up/down, while base rows use 2:4 sparse
    gate_up/down. The only synchronization point is before assembling the final
    row order. It is disabled during CUDA Graph capture because auxiliary
    streams are not represented in the current graph-safe verify plan.
    """
    if not route_overlap_streams():
        return None
    if not input_tensor.is_cuda or torch.cuda.is_current_stream_capturing():
        return None
    dense_count = int(dense_rows.numel())
    base_count = int(base_rows.numel())
    if dense_count <= 0 or base_count <= 0:
        return None

    rows = int(input_tensor.shape[0])
    linear_breakdown = breakdown_linear_enabled()
    current_stream = torch.cuda.current_stream(input_tensor.device)
    base_stream, dense_stream = _route_overlap_streams_for_device(
        input_tensor.device
    )
    base_stream.wait_stream(current_stream)
    dense_stream.wait_stream(current_stream)

    dense_down: torch.Tensor | None = None
    base_down: torch.Tensor | None = None
    intermediate_size = int(getattr(down_module, "_speclink_sr24_weight_shape")[1])

    with torch.cuda.stream(dense_stream):
        dense_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_input = input_tensor.index_select(0, dense_rows)
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_dense_gather_cuda_ms",
            dense_gather_timer,
        )
        dense_gate_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_dense_gate_up_cuda_ms",
            dense_gate_timer,
        )
        dense_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_act = act_fn(dense_gate_up) if act_fn is not None else _silu_and_mul_local(
            dense_gate_up,
            intermediate_size,
        )
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_dense_act_cuda_ms",
            dense_act_timer,
        )
        dense_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        if down_uses_exact_dense_rows:
            if down_dense_weight is None:
                return None
            dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
        else:
            dense_down = _semi_structured_linear(
                dense_act, _sparse_base_weight(down_module), None
            )
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_dense_down_cuda_ms",
            dense_down_timer,
        )

    with torch.cuda.stream(base_stream):
        base_gather_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_input = input_tensor.index_select(0, base_rows)
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_base_gather_cuda_ms",
            base_gather_timer,
        )
        base_gate_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_gate_up = _semi_structured_linear(
            base_input, _sparse_base_weight(gate_up_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_base_gate_up_cuda_ms",
            base_gate_timer,
        )
        base_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_act = act_fn(base_gate_up) if act_fn is not None else _silu_and_mul_local(
            base_gate_up,
            intermediate_size,
        )
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_base_act_cuda_ms",
            base_act_timer,
        )
        base_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_down = _semi_structured_linear(
            base_act, _sparse_base_weight(down_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_base_down_sparse_cuda_ms",
            base_down_timer,
        )

    current_stream.wait_stream(dense_stream)
    current_stream.wait_stream(base_stream)
    if dense_down is None or base_down is None:
        return None

    out_features = int(base_down.shape[1])
    assemble_timer = _breakdown_cuda_start() if linear_breakdown else None
    if triton_route_assembly():
        output = _triton_routed_assemble(
            dense_down,
            dense_rows,
            base_down,
            base_rows,
            total_rows=rows,
            out_features=out_features,
        )
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_triton_assemble_cuda_ms",
            assemble_timer,
        )
    else:
        output = torch.empty(
            (rows, out_features),
            dtype=base_down.dtype,
            device=base_down.device,
        )
        output.index_copy_(0, dense_rows, dense_down)
        output.index_copy_(0, base_rows, base_down)
        _breakdown_record_cuda(
            "row_routed_mlp_overlap_index_copy_cuda_ms",
            assemble_timer,
        )
    if linear_breakdown:
        _breakdown_count("row_routed_mlp_overlap_stream_calls", 1)
        _breakdown_count("row_routed_mlp_overlap_dense_rows", dense_count)
        _breakdown_count("row_routed_mlp_overlap_base_rows", base_count)
    return output.contiguous()


def _adaptive_dense_fallback_decision(
    module: Any,
    *,
    rows: int,
    dense_candidate_rows: int,
    allow_zero_residual: bool = False,
) -> tuple[bool, str]:
    if not adaptive_dense_fallback_enabled():
        return False, "disabled"
    if rows <= 0:
        return False, "empty"
    if mode() not in {"selective", "all_corrected"}:
        return False, "mode"
    if adaptive_dense_fallback_no_residual_only() and not allow_zero_residual:
        return False, "no_residual_only"
    if getattr(module, "_speclink_sr24_residual_backend", "") != "dense_rows":
        return False, "backend"
    leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    if leaf not in {"gate_up_proj", "down_proj"}:
        return False, "leaf"
    dense_candidate_rows = max(0, min(int(dense_candidate_rows), int(rows)))
    small_rows = adaptive_dense_fallback_small_rows()
    if dense_candidate_rows <= 0:
        if (
            allow_zero_residual
            and leaf == "down_proj"
            and rows <= small_rows
            and adaptive_dense_fallback_small_down_no_residual()
        ):
            return True, "small_down_no_residual"
        if (
            allow_zero_residual
            and leaf == "gate_up_proj"
            and rows <= small_rows
            and adaptive_dense_fallback_small_gate_up_no_residual()
        ):
            return True, "small_gate_up_no_residual"
        return False, "no_residual_rows"
    if rows <= small_rows:
        return True, f"small_{leaf}"
    fraction = dense_candidate_rows / max(rows, 1)
    if leaf == "gate_up_proj":
        threshold = adaptive_dense_fallback_gate_up_fraction()
    else:
        threshold = adaptive_dense_fallback_down_fraction()
    if fraction >= threshold:
        return True, f"{leaf}_fraction_ge_{threshold:g}"
    return False, "below_threshold"


def _adaptive_dense_candidate_rows_for_bucket(
    *,
    rows: int,
    bucket_rows: torch.Tensor,
) -> int:
    bucket_count = int(bucket_rows.numel())
    if bucket_count <= 0:
        return 0
    # Use the rows that this execution plan will actually correct. Earlier
    # versions treated a full capped bucket as "all rows might need residual",
    # which made bucket16 look like a 100% residual step and caused adaptive
    # dense fallback to fire on nearly every mixed Linear call. That is safe for
    # accuracy but is not a useful speed planner: it replaces the intended
    # sparse-base plus small bucket correction with dense Linear everywhere.
    return bucket_count


@torch.inference_mode()
def _adaptive_dense_fallback_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    dense_candidate_rows: int,
    reason: str,
) -> torch.Tensor:
    linear_breakdown = breakdown_linear_enabled()
    rows = int(input_tensor.shape[0])
    _record_adaptive_dense_fallback_runtime_stats(
        rows=rows,
        dense_candidate_rows=dense_candidate_rows,
    )
    dense_timer = _breakdown_cuda_start() if linear_breakdown else None
    output = _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
    _breakdown_record_cuda_module(
        module,
        "adaptive_dense_fallback_gemm_cuda_ms",
        dense_timer,
    )
    if linear_breakdown:
        _breakdown_count_module(module, "adaptive_dense_fallback_calls", 1)
        _breakdown_count_module(module, "adaptive_dense_fallback_rows", rows)
        _breakdown_count_module(
            module,
            "adaptive_dense_fallback_dense_candidate_rows",
            dense_candidate_rows,
        )
        _breakdown_count(f"adaptive_dense_fallback_reason_{reason}", 1)
    return output


@torch.inference_mode()
def sparse_linear_output(module: Any, input_tensor: torch.Tensor) -> torch.Tensor | None:
    if dense_zero_dense_rows_active(module):
        if input_tensor.ndim != 2:
            raise RuntimeError("SpecLink SR24 dense_rows Llama path expects 2D tensors")
        return _dense_zero_dense_rows_output(module, input_tensor)
    if not sparse_backend_active(module):
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 sparse Llama path expects 2D tensors")
    rows = int(input_tensor.shape[0])

    bias = getattr(module, "bias", None)
    if getattr(module, "skip_bias_add", False):
        bias = None
    if getattr(module, "_speclink_sr24_dense_fastpath", False):
        return _sr24_dense_linear(input_tensor, module.weight, bias).contiguous()

    dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
    if getattr(module, "_speclink_sr24_gate_up_split", "none") == "channel_pair":
        raise RuntimeError(
            "SR24 gate_up channel_pair is supported only through the LlamaMLP "
            "grouped MLP path; down_proj must remain dense for this ablation."
        )
    if _runtime_base_only_for_module(module):
        base_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = _semi_structured_linear(
            input_tensor, _sparse_base_weight(module), bias
        )
        _breakdown_record_cuda_module(
            module, "runtime_base_only_sparse_linear_cuda_ms", base_timer
        )
        if breakdown_linear_enabled():
            _breakdown_count_module(module, "runtime_base_only_calls", 1)
            _breakdown_count_module(
                module, "runtime_base_only_rows", int(input_tensor.shape[0])
            )
        return output.contiguous()
    if (
        dense_weight is not None
        and mode() != "base_only"
        and static_mask_state() == "all_residual"
        and _all_residual_dense_shortcut_enabled()
    ):
        if breakdown_linear_enabled():
            _breakdown_count_module(module, "full_residual_early_dense_calls", 1)
            _breakdown_count_module(
                module, "full_residual_early_dense_rows", int(input_tensor.shape[0])
            )
        return _sr24_exact_dense_linear(module, input_tensor, dense_weight, bias)
    if dense_weight is not None and mode() != "base_only":
        residual_state = _current_residual_state()
        if residual_state == "all_residual" and _all_residual_dense_shortcut_enabled():
            if breakdown_linear_enabled():
                _breakdown_count_module(module, "full_residual_early_dense_calls", 1)
                _breakdown_count_module(
                    module,
                    "full_residual_early_dense_rows",
                    int(input_tensor.shape[0]),
                )
            return _sr24_exact_dense_linear(module, input_tensor, dense_weight, bias)
        if residual_state is None and _current_residual_mask() is None:
            sr_mode = mode()
            if (
                sr_mode == "all_corrected"
                and _all_residual_dense_shortcut_enabled()
            ) or (
                sr_mode == "selective"
                and _selective_dense_nonverify_for_module_rows(module, rows)
            ):
                if breakdown_linear_enabled():
                    _breakdown_count_module(
                        module, "full_residual_early_dense_calls", 1
                    )
                    _breakdown_count_module(
                        module,
                        "full_residual_early_dense_rows",
                        int(input_tensor.shape[0]),
                    )
                return _sr24_exact_dense_linear(module, input_tensor, dense_weight, bias)
        if reduce_cpu_sync():
            if residual_state == "no_residual":
                should_fallback, reason = _adaptive_dense_fallback_decision(
                    module,
                    rows=rows,
                    dense_candidate_rows=0,
                    allow_zero_residual=True,
                )
                if should_fallback:
                    return _adaptive_dense_fallback_output(
                        module,
                        input_tensor,
                        dense_weight,
                        bias,
                        dense_candidate_rows=0,
                        reason=reason,
                    )
                return _semi_structured_linear(
                    input_tensor, _sparse_base_weight(module), bias
                ).contiguous()
            if residual_state == "mixed":
                residual_mask = _residual_mask_for_input(input_tensor)
                if residual_mask is not None:
                    residual_priority = _residual_priority_for_input(input_tensor)
                    residual_bucket = _residual_bucket_for_mask(
                        input_tensor,
                        residual_mask,
                        residual_priority,
                    )
                    if residual_bucket is not None:
                        bucket_rows, _ = residual_bucket
                        dense_candidate_rows = (
                            _adaptive_dense_candidate_rows_for_bucket(
                                rows=rows,
                                bucket_rows=bucket_rows,
                            )
                        )
                    else:
                        # Without a bucket, the existing dense_rows mixed path
                        # computes a full dense GEMM plus sparse base and then
                        # selects rows. Dense fallback avoids the duplicate
                        # verifier work and is conservative for accuracy.
                        dense_candidate_rows = rows
                    should_fallback, reason = _adaptive_dense_fallback_decision(
                        module,
                        rows=rows,
                        dense_candidate_rows=dense_candidate_rows,
                    )
                    if should_fallback:
                        return _adaptive_dense_fallback_output(
                            module,
                            input_tensor,
                            dense_weight,
                            bias,
                            dense_candidate_rows=dense_candidate_rows,
                            reason=reason,
                        )
        else:
            residual_rows = _residual_rows_for_input(input_tensor)
            if residual_rows is not None and int(residual_rows.numel()) == rows:
                return _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
    elif (
        dense_weight is not None
        and mode() == "base_only"
        and base_only_dense_nonverify()
        and _current_residual_state() is None
        and _current_residual_mask() is None
    ):
        return _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()

    if dense_weight is not None and mode() == "base_only":
        max_dense_rows = base_only_dense_verify_max_rows()
        if (
            max_dense_rows > 0
            and _current_residual_state() == "no_residual"
            and int(input_tensor.shape[0]) <= max_dense_rows
        ):
            dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
            output = _sr24_dense_linear(input_tensor, dense_weight, bias).contiguous()
            _breakdown_record_cuda_module(
                module,
                "base_only_dense_verify_fallback_cuda_ms",
                dense_timer,
            )
            if breakdown_linear_enabled():
                _breakdown_count_module(
                    module, "base_only_dense_verify_fallback_calls", 1
                )
                _breakdown_count_module(
                    module,
                    "base_only_dense_verify_fallback_rows",
                    int(input_tensor.shape[0]),
                )
            return output

    if (
        route_bucket_rows()
        and not route_all_residual_rows()
        and dense_weight is not None
        and getattr(module, "_speclink_sr24_residual_backend", "") == "dense_rows"
        and mode() != "base_only"
        and reduce_cpu_sync()
        and _current_residual_state() == "mixed"
    ):
        residual_mask = _residual_mask_for_input(input_tensor)
        if residual_mask is not None:
            residual_priority = _residual_priority_for_input(input_tensor)
            residual_bucket = _residual_bucket_for_mask(
                input_tensor,
                residual_mask,
                residual_priority,
            )
            if residual_bucket is not None:
                bucket_rows, bucket_values = residual_bucket
                cached_dense_rows = _current_residual_rows()
                cached_base_rows = _current_base_rows()
                if (
                    cached_dense_rows is not None
                    and cached_base_rows is not None
                    and int(cached_dense_rows.numel()) == int(bucket_rows.numel())
                    and int(cached_dense_rows.numel())
                    + int(cached_base_rows.numel())
                    == int(input_tensor.shape[0])
                ):
                    _breakdown_count("route_bucket_rows_cached_plan_hits", 1)
                    routed_output = _routed_dense_rows_output(
                        module,
                        input_tensor,
                        dense_weight,
                        bias,
                        cached_dense_rows,
                        route_label="route_bucket_dense_rows",
                        base_rows=cached_base_rows,
                    )
                else:
                    routed_output = _routed_bucket_dense_rows_output(
                        module,
                        input_tensor,
                        dense_weight,
                        bias,
                        bucket_rows,
                        bucket_values,
                    )
                if routed_output is not None:
                    return routed_output
    if (
        route_all_residual_rows()
        and not route_reuse_base_output()
        and dense_weight is not None
        and getattr(module, "_speclink_sr24_residual_backend", "") == "dense_rows"
        and mode() != "base_only"
        and reduce_cpu_sync()
        and _current_residual_state() == "mixed"
    ):
        residual_mask = _residual_mask_for_input(input_tensor)
        if residual_mask is not None:
            residual_mask = residual_mask.to(
                device=input_tensor.device,
                dtype=torch.bool,
            )
            dense_rows = None
            base_rows = None
            cached_dense_rows = _current_residual_rows()
            cached_base_rows = _current_base_rows()
            if (
                cached_dense_rows is not None
                and cached_base_rows is not None
                and residual_mask.numel() == int(input_tensor.shape[0])
            ):
                dense_rows = cached_dense_rows
                base_rows = cached_base_rows
                _breakdown_count("route_all_residual_rows_cached_plan_hits", 1)
            if dense_rows is None:
                dense_rows = residual_mask.nonzero(as_tuple=False).squeeze(1)
            routed_output = _routed_dense_rows_output(
                module,
                input_tensor,
                dense_weight,
                bias,
                dense_rows,
                route_label="route_all_residual_rows",
                base_rows=base_rows,
            )
            if routed_output is not None:
                return routed_output

    linear_breakdown = breakdown_linear_enabled()
    base_timer = _breakdown_cuda_start() if linear_breakdown else None
    base_output = _semi_structured_linear(
        input_tensor, _sparse_base_weight(module), bias
    )
    _breakdown_record_cuda_module(module, "base_sparse_linear_cuda_ms", base_timer)
    if linear_breakdown:
        _breakdown_count_module(module, "base_sparse_linear_calls", 1)
        _breakdown_count_module(module, "base_sparse_linear_rows", rows)
        _breakdown_count_module(
            module, "base_sparse_linear_output_elements", base_output.numel()
        )
    if mode() == "base_only" or getattr(
        module, "_speclink_sr24_no_residual", False
    ):
        return base_output.contiguous()
    if _selective_no_context_uses_sparse_base(module, rows):
        if linear_breakdown:
            _breakdown_count_module(
                module, "selective_noverify_sparse_base_rows", rows
            )
        return base_output.contiguous()
    if reduce_cpu_sync():
        residual_mask = _residual_mask_for_input(input_tensor)
        if residual_mask is None:
            return base_output.contiguous()
        residual_state = _current_residual_state()
        full_residual = residual_state == "all_residual" or (
            residual_state is None
            and _current_residual_mask() is None
            and mode() == "all_corrected"
        )
        residual_priority = _residual_priority_for_input(input_tensor)
        residual_bucket = _residual_bucket_for_mask(
            input_tensor,
            residual_mask,
            residual_priority,
        )
        if dense_weight is not None:
            if (
                route_reuse_base_output()
                and getattr(module, "_speclink_sr24_residual_backend", "") == "dense_rows"
            ):
                cached_residual_rows = _current_residual_rows()
                if (
                    cached_residual_rows is not None
                    and residual_mask.numel() == int(input_tensor.shape[0])
                ):
                    residual_rows = cached_residual_rows
                    _breakdown_count("route_reuse_base_output_cached_plan_hits", 1)
                else:
                    residual_rows = residual_mask.to(
                        device=input_tensor.device,
                        dtype=torch.bool,
                    ).nonzero(as_tuple=False).squeeze(1)
                return _dense_rows_linear_output(
                    module,
                    input_tensor,
                    base_output,
                    residual_rows,
                    bias,
                ).contiguous()
            if residual_bucket is not None:
                bucket_rows, bucket_values = residual_bucket
                if linear_breakdown:
                    _breakdown_count_module(module, "dense_rows_bucket_linear_calls", 1)
                    _breakdown_count_module(
                        module, "dense_rows_bucket_rows", int(bucket_rows.numel())
                    )
                    if breakdown_sync_counts():
                        try:
                            _breakdown_count(
                                "dense_rows_bucket_active_rows",
                                int(bucket_values.to(dtype=torch.int32).sum().item()),
                            )
                        except Exception:
                            pass
                compute_active_only = (
                    bucket_dense_compute_active_only()
                    and bucket_dense_copy_active_only()
                )
                if (
                    compute_active_only
                    and bucket_dense_active_mask_fused()
                    and triton_bucket_dense_gemm()
                    and base_output.is_cuda
                ):
                    fused_timer = (
                        _breakdown_cuda_start() if linear_breakdown else None
                    )
                    fused_ok = False
                    try:
                        fused_ok = _triton_bucket_dense_gemm_scatter_inplace(
                            input_tensor,
                            dense_weight,
                            bucket_rows,
                            bucket_values,
                            base_output,
                            bias,
                            force_all_bucket_rows=False,
                        )
                    except Exception:
                        fused_ok = False
                    _breakdown_record_cuda_module(
                        module,
                        "bucket_active_mask_fused_dense_gemm_scatter_cuda_ms",
                        fused_timer,
                    )
                    if fused_ok:
                        if linear_breakdown:
                            _breakdown_count_module(
                                module,
                                "bucket_active_mask_fused_dense_gemm_scatter_calls",
                                1,
                            )
                            _breakdown_count_module(
                                module,
                                "bucket_active_mask_fused_dense_gemm_scatter_rows",
                                int(bucket_rows.numel()),
                            )
                        return base_output.contiguous()
                active_bucket_rows = bucket_rows
                active_bucket_values = bucket_values
                active_bucket_indices = None
                if compute_active_only:
                    active_bucket_indices = bucket_values.to(
                        dtype=torch.bool
                    ).nonzero(as_tuple=False).squeeze(1)
                    active_bucket_rows = bucket_rows.index_select(
                        0, active_bucket_indices
                    )
                    active_bucket_values = bucket_values.index_select(
                        0, active_bucket_indices
                    )
                    _breakdown_count_module(
                        module,
                        "bucket_dense_compute_active_only_calls",
                        1,
                    )
                    _breakdown_count_module(
                        module,
                        "bucket_dense_compute_active_only_rows",
                        int(active_bucket_rows.numel()),
                    )
                    if int(active_bucket_rows.numel()) <= 0:
                        return base_output.contiguous()
                if (
                    triton_bucket_dense_gemm()
                    and base_output.is_cuda
                    and not compute_active_only
                ):
                    fused_timer = (
                        _breakdown_cuda_start() if linear_breakdown else None
                    )
                    fused_ok = False
                    try:
                        fused_ok = _triton_bucket_dense_gemm_scatter_inplace(
                            input_tensor,
                            dense_weight,
                            bucket_rows,
                            bucket_values,
                            base_output,
                            bias,
                            force_all_bucket_rows=(
                                bucket_dense_copy()
                                and not bucket_dense_copy_active_only()
                            ),
                        )
                    except Exception:
                        fused_ok = False
                    _breakdown_record_cuda_module(
                        module,
                        "bucket_triton_dense_gemm_scatter_cuda_ms",
                        fused_timer,
                    )
                    if fused_ok:
                        if linear_breakdown:
                            _breakdown_count_module(
                                module,
                                "bucket_triton_dense_gemm_scatter_calls",
                                1,
                            )
                            _breakdown_count_module(
                                module,
                                "bucket_triton_dense_gemm_scatter_rows",
                                int(bucket_rows.numel()),
                            )
                        return base_output.contiguous()
                gather_input_timer = (
                    _breakdown_cuda_start() if linear_breakdown else None
                )
                dense_input = input_tensor.index_select(0, active_bucket_rows)
                _breakdown_record_cuda(
                    "gather_input_index_select_cuda_ms",
                    gather_input_timer,
                )
                dense_timer = _breakdown_cuda_start() if linear_breakdown else None
                dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
                _breakdown_record_cuda_module(
                    module, "residual_dense_gemm_cuda_ms", dense_timer
                )
                if bucket_dense_copy():
                    copy_timer = (
                        _breakdown_cuda_start() if linear_breakdown else None
                    )
                    if bucket_dense_copy_active_only():
                        scatter_ok = False
                        try:
                            scatter_ok = _triton_bucket_dense_scatter_inplace(
                                dense_output,
                                active_bucket_rows,
                                active_bucket_values,
                                base_output,
                            )
                        except Exception:
                            scatter_ok = False
                        if not scatter_ok:
                            _bucket_dense_overwrite_inplace(
                                base_output,
                                dense_output,
                                active_bucket_rows,
                                active_bucket_values,
                            )
                    else:
                        base_output.index_copy_(0, active_bucket_rows, dense_output)
                    _breakdown_record_cuda(
                        "bucket_dense_copy_index_copy_cuda_ms",
                        copy_timer,
                    )
                    if linear_breakdown:
                        _breakdown_count_module(
                            module, "bucket_dense_copy_calls", 1
                        )
                        _breakdown_count_module(
                            module,
                            "bucket_dense_copy_rows",
                            int(bucket_rows.numel()),
                        )
                    return base_output.contiguous()
                if triton_bucket_override() and base_output.is_cuda:
                    override_timer = (
                        _breakdown_cuda_start() if linear_breakdown else None
                    )
                    output = _triton_bucket_override_inplace(
                        base_output,
                        dense_output,
                        bucket_rows,
                        bucket_values,
                    )
                    _breakdown_record_cuda(
                        "bucket_triton_override_cuda_ms",
                        override_timer,
                    )
                    return output.contiguous()
                if not bucket_dense_delta_add():
                    overwrite_timer = (
                        _breakdown_cuda_start() if linear_breakdown else None
                    )
                    output = _bucket_dense_overwrite_inplace(
                        base_output,
                        dense_output,
                        bucket_rows,
                        bucket_values,
                    )
                    _breakdown_record_cuda(
                        "bucket_dense_overwrite_index_copy_cuda_ms",
                        overwrite_timer,
                    )
                    if linear_breakdown:
                        _breakdown_count_module(
                            module, "bucket_dense_overwrite_calls", 1
                        )
                        _breakdown_count_module(
                            module,
                            "bucket_dense_overwrite_rows",
                            int(bucket_rows.numel()),
                        )
                    return output.contiguous()
                gather_base_timer = (
                    _breakdown_cuda_start() if linear_breakdown else None
                )
                base_rows = base_output.index_select(0, bucket_rows)
                _breakdown_record_cuda(
                    "gather_base_index_select_cuda_ms",
                    gather_base_timer,
                )
                delta_timer = _breakdown_cuda_start() if linear_breakdown else None
                delta = (dense_output - base_rows) * bucket_values.unsqueeze(1)
                _breakdown_record_cuda("bucket_delta_compute_cuda_ms", delta_timer)
                scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
                base_output.index_add_(0, bucket_rows, delta)
                _breakdown_record_cuda("scatter_index_add_cuda_ms", scatter_timer)
                return base_output.contiguous()
            dense_timer = _breakdown_cuda_start() if linear_breakdown else None
            dense_output = _sr24_dense_linear(input_tensor, dense_weight, bias)
            _breakdown_record_cuda_module(
                module, "residual_dense_full_gemm_cuda_ms", dense_timer
            )
            select_timer = _breakdown_cuda_start() if linear_breakdown else None
            selector = residual_mask.to(dtype=torch.bool).unsqueeze(1)
            output = torch.where(selector, dense_output, base_output)
            _breakdown_record_cuda("residual_dense_full_select_cuda_ms", select_timer)
            return output.contiguous()
        residual_weight = getattr(module, "_speclink_sr24_residual_sparse", None)
        if residual_weight is not None:
            if full_residual:
                residual_timer = _breakdown_cuda_start() if linear_breakdown else None
                residual_output = _semi_structured_linear(
                    input_tensor,
                    residual_weight,
                    None,
                )
                _breakdown_record_cuda_module(
                    module,
                    "residual_sparse_full_gemm_cuda_ms",
                    residual_timer,
                )
                add_timer = _breakdown_cuda_start() if linear_breakdown else None
                output = base_output + residual_output
                _breakdown_record_cuda(
                    "residual_sparse_full_add_cuda_ms",
                    add_timer,
                )
                return output.contiguous()
            if residual_bucket is not None:
                bucket_rows, bucket_values = residual_bucket
                gather_input_timer = (
                    _breakdown_cuda_start() if linear_breakdown else None
                )
                residual_input = input_tensor.index_select(0, bucket_rows)
                _breakdown_record_cuda(
                    "gather_input_index_select_cuda_ms",
                    gather_input_timer,
                )
                residual_timer = _breakdown_cuda_start() if linear_breakdown else None
                residual_output = _semi_structured_linear(
                    residual_input,
                    residual_weight,
                    None,
                )
                _breakdown_record_cuda_module(
                    module, "residual_sparse_gemm_cuda_ms", residual_timer
                )
                delta_timer = _breakdown_cuda_start() if linear_breakdown else None
                residual_output = residual_output * bucket_values.unsqueeze(1)
                _breakdown_record_cuda("bucket_delta_compute_cuda_ms", delta_timer)
                scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
                base_output.index_add_(0, bucket_rows, residual_output)
                _breakdown_record_cuda("scatter_index_add_cuda_ms", scatter_timer)
                return base_output.contiguous()
            residual_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_output = _semi_structured_linear(
                input_tensor,
                residual_weight,
                None,
            )
            _breakdown_record_cuda_module(
                module, "residual_sparse_full_gemm_cuda_ms", residual_timer
            )
            select_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_output = residual_output * residual_mask.to(
                dtype=residual_output.dtype
            ).unsqueeze(1)
            output = base_output + residual_output
            _breakdown_record_cuda("residual_sparse_full_select_cuda_ms", select_timer)
            return output.contiguous()
        residual_values = getattr(module, "_speclink_sr24_residual_values", None)
        if residual_values is None:
            raise RuntimeError("SR24 residual weight is missing")
        if full_residual:
            residual_rows = _device_arange(
                rows,
                dtype=torch.long,
                device=input_tensor.device,
            )
            return _compressed_residual_linear_output(
                module,
                input_tensor,
                base_output,
                residual_rows,
            ).contiguous()
        return _compressed_residual_linear_output_masked(
            module,
            input_tensor,
            base_output,
            residual_mask,
        ).contiguous()
    residual_rows = _residual_rows_for_input(input_tensor)
    if residual_rows is None:
        return base_output.contiguous()

    if dense_weight is not None:
        return _dense_rows_linear_output(
            module,
            input_tensor,
            base_output,
            residual_rows,
            bias,
        ).contiguous()

    residual_weight = getattr(module, "_speclink_sr24_residual_sparse", None)
    if residual_weight is None:
        if mode() == "base_only":
            return base_output
        residual_values = getattr(module, "_speclink_sr24_residual_values", None)
        if residual_values is None:
            raise RuntimeError("SR24 residual weight is missing")
        return _compressed_residual_linear_output(
            module,
            input_tensor,
            base_output,
            residual_rows,
        ).contiguous()
    rows = int(input_tensor.shape[0])
    if int(residual_rows.numel()) == rows:
        return (
            base_output + _semi_structured_linear(input_tensor, residual_weight, None)
        ).contiguous()

    residual_input = input_tensor.index_select(0, residual_rows)
    residual_output = _semi_structured_linear(residual_input, residual_weight, None)
    output = base_output.clone()
    output.index_add_(0, residual_rows, residual_output)
    return output.contiguous()


@torch.inference_mode()
def gate_up_split_linear_output(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    split_mode = getattr(module, "_speclink_sr24_gate_up_split", "none")
    if not enabled() or split_mode not in {"up_sparse", "gate_sparse"}:
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 gate_up split path expects 2D tensors")
    dense_weight = getattr(module, "_speclink_sr24_gate_up_dense_weight", None)
    sparse_weight = getattr(module, "_speclink_sr24_gate_up_sparse_weight", None)
    if dense_weight is None or sparse_weight is None:
        raise RuntimeError("SR24 gate_up split weights are missing")
    dense_output = _sr24_dense_linear(input_tensor, dense_weight, bias=None)
    sparse_output = _semi_structured_linear(input_tensor, sparse_weight, bias=None)
    if split_mode == "up_sparse":
        return torch.cat((dense_output, sparse_output), dim=-1).contiguous()
    return torch.cat((sparse_output, dense_output), dim=-1).contiguous()


@torch.inference_mode()
def gate_up_split_act_output(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    split_mode = getattr(module, "_speclink_sr24_gate_up_split", "none")
    if not enabled() or split_mode not in {"up_sparse", "gate_sparse"}:
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 gate_up split path expects 2D tensors")
    dense_weight = getattr(module, "_speclink_sr24_gate_up_dense_weight", None)
    sparse_weight = getattr(module, "_speclink_sr24_gate_up_sparse_weight", None)
    if dense_weight is None or sparse_weight is None:
        raise RuntimeError("SR24 gate_up split weights are missing")

    linear_breakdown = breakdown_linear_enabled()
    dense_timer = _breakdown_cuda_start() if linear_breakdown else None
    dense_output = _sr24_dense_linear(input_tensor, dense_weight, bias=None)
    _breakdown_record_cuda_module(
        module,
        "gate_up_split_dense_half_gemm_cuda_ms",
        dense_timer,
    )
    sparse_timer = _breakdown_cuda_start() if linear_breakdown else None
    sparse_output = _semi_structured_linear(input_tensor, sparse_weight, bias=None)
    _breakdown_record_cuda_module(
        module,
        "gate_up_split_sparse_half_gemm_cuda_ms",
        sparse_timer,
    )
    act_timer = _breakdown_cuda_start() if linear_breakdown else None
    if split_mode == "up_sparse":
        activated = F.silu(dense_output) * sparse_output
    else:
        activated = F.silu(sparse_output) * dense_output
    _breakdown_record_cuda_module(
        module,
        "gate_up_split_direct_act_cuda_ms",
        act_timer,
    )
    if linear_breakdown:
        _breakdown_count_module(module, "gate_up_split_direct_act_calls", 1)
        _breakdown_count_module(
            module,
            "gate_up_split_direct_act_rows",
            int(input_tensor.shape[0]),
        )
    return activated.contiguous()


@torch.inference_mode()
def gate_up_channel_split_mlp_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    act_fn: Any | None = None,
) -> torch.Tensor | None:
    split_mode = getattr(gate_up_module, "_speclink_sr24_gate_up_split", "none")
    if not enabled() or split_mode != "channel_pair":
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError(
            "SpecLink SR24 gate_up channel-pair MLP path expects 2D tensors"
        )
    if getattr(down_module, "_speclink_sr24_enabled", False):
        raise RuntimeError(
            "SR24 gate_up channel_pair currently requires a dense down_proj. "
            "Remove down_proj from SPECLINK_SR24_TARGET_LEAFS for this ablation."
        )
    dense_weight = getattr(
        gate_up_module, "_speclink_sr24_gate_up_dense_weight", None
    )
    sparse_weight = getattr(
        gate_up_module, "_speclink_sr24_gate_up_sparse_weight", None
    )
    dense_count = int(
        getattr(gate_up_module, "_speclink_sr24_gate_up_dense_channel_count", 0)
    )
    sparse_count = int(
        getattr(gate_up_module, "_speclink_sr24_gate_up_sparse_channel_count", 0)
    )
    grouped_channels = getattr(
        gate_up_module, "_speclink_sr24_gate_up_grouped_channels", None
    )
    if (
        dense_weight is None
        or sparse_weight is None
        or grouped_channels is None
        or dense_count <= 0
        or sparse_count <= 0
    ):
        raise RuntimeError("SR24 gate_up channel-pair weights are missing")

    dense_output = _sr24_dense_linear(input_tensor, dense_weight, bias=None)
    sparse_output = _semi_structured_linear(input_tensor, sparse_weight, bias=None)
    dense_gate = dense_output[:, :dense_count]
    dense_up = dense_output[:, dense_count:]
    sparse_gate = sparse_output[:, :sparse_count]
    sparse_up = sparse_output[:, sparse_count:]
    if gate_up_channel_fused_act() and act_fn is not None:
        grouped_gate = torch.cat([dense_gate, sparse_gate], dim=-1)
        grouped_up = torch.cat([dense_up, sparse_up], dim=-1)
        grouped_gate_up = torch.cat([grouped_gate, grouped_up], dim=-1)
        grouped_act = act_fn(grouped_gate_up)
    else:
        dense_act = F.silu(dense_gate) * dense_up
        sparse_act = F.silu(sparse_gate) * sparse_up
        grouped_act = torch.cat([dense_act, sparse_act], dim=-1)

    down_weight = getattr(down_module, "weight", None)
    if down_weight is None:
        raise RuntimeError("SR24 gate_up channel-pair path needs down_proj.weight")
    grouped_down_weight = getattr(
        down_module, "_speclink_sr24_gate_up_grouped_down_weight", None
    )
    if (
        grouped_down_weight is None
        or grouped_down_weight.device != down_weight.device
        or grouped_down_weight.dtype != down_weight.dtype
        or tuple(grouped_down_weight.shape) != tuple(down_weight.shape)
    ):
        with torch.no_grad():
            grouped_down_weight = down_weight.index_select(
                1, grouped_channels.to(device=down_weight.device)
            ).contiguous()
        down_module._speclink_sr24_gate_up_grouped_down_weight = grouped_down_weight

    bias = getattr(down_module, "bias", None)
    if getattr(down_module, "skip_bias_add", False):
        bias = None
    return F.linear(grouped_act, grouped_down_weight, bias).contiguous()


@torch.inference_mode()
def row_routed_gate_up_output(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Route only gate_up rows for mixed SR24 plans.

    The full row-routed MLP path requires gate_up and down to be SR24 in the
    same layer. The current quality-safe scope intentionally uses different
    layers for gate_up and down, so this gate_up-only path removes the largest
    duplicate work without changing the downstream dense/down path.
    """
    if not enabled() or not row_routed_mlp():
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 row-routed gate_up path expects 2D tensors")
    if mode() == "base_only" or _current_residual_state() != "mixed":
        return None
    module_leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    if module_leaf != "gate_up_proj":
        return None
    if not sparse_backend_active(module):
        return None
    if getattr(module, "_speclink_sr24_residual_backend", "") != "dense_rows":
        return None
    dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
    if dense_weight is None:
        return None

    residual_mask = _residual_mask_for_input(input_tensor)
    if residual_mask is None:
        return None
    rows = int(input_tensor.shape[0])
    dense_rows = _current_residual_rows()
    base_rows = _current_base_rows()
    if (
        dense_rows is not None
        and base_rows is not None
        and int(dense_rows.numel()) + int(base_rows.numel()) == rows
    ):
        dense_rows = dense_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        base_rows = base_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        _breakdown_count("row_routed_gate_up_cached_plan_hits", 1)
    else:
        residual_priority = _residual_priority_for_input(input_tensor)
        residual_bucket = _residual_bucket_for_mask(
            input_tensor,
            residual_mask,
            residual_priority,
        )
        if residual_bucket is not None:
            bucket_rows, bucket_values = residual_bucket
            if bucket_dense_copy_active_only():
                active = bucket_values.to(dtype=torch.bool).nonzero(
                    as_tuple=False
                ).squeeze(1)
                dense_rows = bucket_rows.index_select(0, active)
            else:
                dense_rows = bucket_rows
        else:
            dense_rows = residual_mask.to(
                device=input_tensor.device,
                dtype=torch.bool,
            ).nonzero(as_tuple=False).squeeze(1)
        dense_rows = dense_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        route_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        route_dense = torch.zeros(rows, dtype=torch.bool, device=input_tensor.device)
        if int(dense_rows.numel()) > 0:
            route_dense.index_fill_(0, dense_rows, True)
        base_rows = (~route_dense).nonzero(as_tuple=False).squeeze(1)
        _breakdown_record_cuda("row_routed_gate_up_route_build_cuda_ms", route_timer)

    dense_count = int(dense_rows.numel())
    if dense_count <= 0:
        return None
    min_dense_rows = row_routed_mlp_min_dense_rows_for_leaf(module_leaf)
    if dense_count < min_dense_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_gate_up_skipped_small_dense_rows", 1)
            _breakdown_count("row_routed_gate_up_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_gate_up_skipped_total_rows", rows)
            _breakdown_count("row_routed_gate_up_min_dense_rows", min_dense_rows)
        return None
    max_dense_rows = row_routed_mlp_max_dense_rows_for_leaf(module_leaf)
    if max_dense_rows > 0 and dense_count > max_dense_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_gate_up_skipped_large_dense_rows", 1)
            _breakdown_count("row_routed_gate_up_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_gate_up_skipped_total_rows", rows)
            _breakdown_count("row_routed_gate_up_max_dense_rows", max_dense_rows)
        return None
    bias = getattr(module, "bias", None)
    if getattr(module, "skip_bias_add", False):
        bias = None
    base_count = int(base_rows.numel()) if base_rows is not None else 0
    min_base_rows = route_min_base_rows_for_leaf(module_leaf)
    if min_base_rows > 0 and 0 < base_count < min_base_rows:
        dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias)
        _breakdown_record_cuda_module(
            module,
            "row_routed_gate_up_fallback_small_base_cuda_ms",
            dense_timer,
        )
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_gate_up_fallback_small_base_calls", 1)
            _breakdown_count("row_routed_gate_up_fallback_small_base_rows", base_count)
            _breakdown_count("row_routed_gate_up_min_base_rows", min_base_rows)
        return output.contiguous()
    max_base_rows = row_routed_mlp_max_base_rows_for_leaf(module_leaf)
    if max_base_rows > 0 and base_count > max_base_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_gate_up_skipped_large_base_rows", 1)
            _breakdown_count("row_routed_gate_up_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_gate_up_skipped_base_rows", base_count)
            _breakdown_count("row_routed_gate_up_skipped_total_rows", rows)
            _breakdown_count("row_routed_gate_up_max_base_rows", max_base_rows)
        return None

    if dense_count >= rows:
        dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias)
        _breakdown_record_cuda_module(
            module,
            "row_routed_gate_up_full_dense_cuda_ms",
            dense_timer,
        )
        if breakdown_linear_enabled():
            _breakdown_count_module(module, "row_routed_gate_up_full_dense_calls", 1)
            _breakdown_count_module(module, "row_routed_gate_up_rows", rows)
            _breakdown_count_module(module, "row_routed_gate_up_dense_rows", rows)
            _breakdown_count_module(module, "row_routed_gate_up_base_rows", 0)
        return output.contiguous()

    if breakdown_linear_enabled():
        _breakdown_count_module(module, "row_routed_gate_up_calls", 1)
        _breakdown_count_module(module, "row_routed_gate_up_rows", rows)
        _breakdown_count_module(module, "row_routed_gate_up_dense_rows", dense_count)
        _breakdown_count_module(module, "row_routed_gate_up_base_rows", base_count)

    contiguous_output = _routed_contiguous_rows_output(
        module,
        input_tensor,
        dense_weight,
        bias,
        dense_rows,
        route_label="row_routed_gate_up",
    )
    if contiguous_output is not None:
        return contiguous_output

    dense_gather_timer = (
        _breakdown_cuda_start() if breakdown_linear_enabled() else None
    )
    dense_input = input_tensor.index_select(0, dense_rows)
    _breakdown_record_cuda(
        "row_routed_gate_up_dense_gather_cuda_ms", dense_gather_timer
    )
    dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
    _breakdown_record_cuda_module(
        module,
        "row_routed_gate_up_dense_gemm_cuda_ms",
        dense_timer,
    )

    out_features = int(dense_output.shape[1])
    output = torch.empty(
        (rows, out_features),
        dtype=dense_output.dtype,
        device=dense_output.device,
    )
    base_rows = (
        base_rows.to(device=input_tensor.device, dtype=torch.long, non_blocking=True)
        if base_rows is not None
        else torch.empty(0, dtype=torch.long, device=input_tensor.device)
    )
    if int(base_rows.numel()) > 0:
        base_gather_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_input = input_tensor.index_select(0, base_rows)
        _breakdown_record_cuda(
            "row_routed_gate_up_base_gather_cuda_ms", base_gather_timer
        )
        base_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        base_output = _semi_structured_linear(
            base_input, _sparse_base_weight(module), bias
        )
        _breakdown_record_cuda_module(
            module,
            "row_routed_gate_up_base_sparse_cuda_ms",
            base_timer,
        )
        assemble_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output.index_copy_(0, base_rows, base_output)
        output.index_copy_(0, dense_rows, dense_output)
        _breakdown_record_cuda(
            "row_routed_gate_up_index_copy_cuda_ms", assemble_timer
        )
    else:
        assemble_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output.index_copy_(0, dense_rows, dense_output)
        _breakdown_record_cuda(
            "row_routed_gate_up_index_copy_cuda_ms", assemble_timer
        )
    return output.contiguous()


@torch.inference_mode()
def _row_routed_down_fixed_block_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    bias: torch.Tensor | None,
    dense_rows: torch.Tensor,
    base_rows: torch.Tensor | None,
) -> torch.Tensor | None:
    """Route fixed-prefix down_proj rows without dynamic row gathers.

    This is the down-proj-only counterpart of the fixed-block MLP route.  It
    keeps important rows dense-only and unimportant rows 2:4 sparse-only, while
    replacing flat `index_select`/`index_copy` routing with stable
    `[active, K + 1, intermediate]` slices.
    """
    if (
        not row_routed_down_fixed_block_enabled()
        or not route_contiguous_fastpath()
        or not fixed_prefix_route_fastpath_enabled()
    ):
        return None
    fixed_route = _current_fixed_prefix_route()
    if fixed_route is None:
        return None
    rows = int(input_tensor.shape[0])
    active_count = int(fixed_route.active_count)
    scheduled_width = int(fixed_route.scheduled_width)
    valid_width = int(fixed_route.valid_width)
    prefix = int(fixed_route.prefix)
    dense_width = int(fixed_route.dense_width)
    base_width = int(fixed_route.base_width)
    if (
        active_count <= 0
        or scheduled_width <= 1
        or valid_width < prefix
        or dense_width != prefix + 1
        or base_width != max(0, valid_width - prefix)
        or active_count * scheduled_width != rows
        or base_width <= 0
    ):
        return None

    dense_count = active_count * dense_width
    base_count = active_count * base_width
    if int(dense_rows.numel()) != dense_count:
        return None
    if base_rows is not None and int(base_rows.numel()) != base_count:
        return None

    module_leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "down_proj")
    max_dense_fraction = route_max_dense_fraction()
    dense_fraction = dense_count / max(rows, 1)
    if 0.0 <= max_dense_fraction <= 1.0 and dense_fraction > max_dense_fraction:
        return None
    min_base_rows = route_min_base_rows_for_leaf(module_leaf)
    if min_base_rows > 0 and base_count < min_base_rows:
        return None

    linear_breakdown = breakdown_linear_enabled()
    policy_match = _scheduler_policy_for_active_count(active_count)
    matched_batch: int | None = None
    policy: dict[str, Any] | None = None
    if policy_match is not None:
        matched_batch, policy = policy_match
        dense_fallback, fallback_reason, min_grouped = (
            _scheduler_policy_requires_dense_fallback(
                policy,
                prefix=prefix,
                valid_width=valid_width,
            )
        )
        if linear_breakdown:
            _breakdown_count("row_routed_down_fixed_block_scheduler_policy_hits", 1)
            _breakdown_count(
                "row_routed_down_fixed_block_scheduler_policy_active_requests",
                active_count,
            )
            _breakdown_count(
                "row_routed_down_fixed_block_scheduler_policy_matched_batch",
                matched_batch,
            )
        if fallback_reason == "incompatible":
            if linear_breakdown:
                _breakdown_count(
                    "row_routed_down_fixed_block_scheduler_policy_incompatible",
                    1,
                )
        elif dense_fallback:
            if linear_breakdown:
                if fallback_reason == "operator_unimplemented":
                    _breakdown_count(
                        "row_routed_down_fixed_block_scheduler_policy_operator_unimplemented",
                        1,
                    )
                if min_grouped is not None:
                    _breakdown_count(
                        "row_routed_down_fixed_block_scheduler_policy_min_grouped",
                        min_grouped,
                    )
                _breakdown_count(
                    "row_routed_down_fixed_block_scheduler_policy_dense_fallback",
                    1,
                )
            return None
        elif linear_breakdown:
            _breakdown_count(
                "row_routed_down_fixed_block_scheduler_policy_mixed_allowed",
                1,
            )

    dense_compute_count = dense_count
    base_compute_count = base_count
    capacity_padding = False
    if (
        fixed_block_capacity_padding_enabled()
        and policy is not None
        and matched_batch is not None
        and matched_batch > active_count
    ):
        dense_capacity = (
            _scheduler_policy_int(policy, "dense_capacity")
            or _scheduler_policy_int(policy, "grouped_dense_rows")
        )
        base_capacity = (
            _scheduler_policy_int(policy, "base_capacity")
            or _scheduler_policy_int(policy, "grouped_base_rows")
        )
        if (
            dense_capacity is not None
            and base_capacity is not None
            and dense_capacity >= dense_count
            and base_capacity >= base_count
        ):
            dense_compute_count = int(dense_capacity)
            base_compute_count = int(base_capacity)
            capacity_padding = (
                dense_compute_count > dense_count
                or base_compute_count > base_count
            )
            if linear_breakdown and capacity_padding:
                _breakdown_count("row_routed_down_fixed_block_capacity_padding_calls", 1)
                _breakdown_count(
                    "row_routed_down_fixed_block_capacity_padding_active_requests",
                    active_count,
                )
                _breakdown_count(
                    "row_routed_down_fixed_block_capacity_padding_matched_batch",
                    matched_batch,
                )
                _breakdown_count(
                    "row_routed_down_fixed_block_capacity_padding_dense_rows",
                    dense_count,
                )
                _breakdown_count(
                    "row_routed_down_fixed_block_capacity_padding_dense_capacity",
                    dense_compute_count,
                )
                _breakdown_count(
                    "row_routed_down_fixed_block_capacity_padding_base_rows",
                    base_count,
                )
                _breakdown_count(
                    "row_routed_down_fixed_block_capacity_padding_base_capacity",
                    base_compute_count,
                )

    input_features = int(input_tensor.shape[1])
    view_timer = _breakdown_cuda_start() if linear_breakdown else None
    input_blocks = input_tensor.reshape(active_count, scheduled_width, input_features)
    if fixed_block_input_buffer_enabled():
        dense_input = _fixed_block_input_buffer(
            f"down_dense:{id(module)}",
            dense_compute_count,
            input_features,
            input_tensor,
        )
        if prefix > 0:
            dense_input[:active_count * prefix].view(
                active_count, prefix, input_features
            ).copy_(input_blocks[:, :prefix, :])
        dense_input[
            active_count * prefix:dense_count
        ].view(active_count, 1, input_features).copy_(
            input_blocks[:, valid_width:valid_width + 1, :]
        )
        if (
            dense_compute_count > dense_count
            and fixed_block_capacity_zero_dummy_enabled()
        ):
            dense_input[dense_count:dense_compute_count].zero_()

        base_input = _fixed_block_input_buffer(
            f"down_base:{id(module)}",
            base_compute_count,
            input_features,
            input_tensor,
        )
        base_input[:base_count].view(active_count, base_width, input_features).copy_(
            input_blocks[:, prefix:valid_width, :]
        )
        if (
            base_compute_count > base_count
            and fixed_block_capacity_zero_dummy_enabled()
        ):
            base_input[base_count:base_compute_count].zero_()
        if linear_breakdown:
            _breakdown_count("row_routed_down_fixed_block_input_buffer_calls", 1)
    else:
        dense_parts: list[torch.Tensor] = []
        if prefix > 0:
            dense_parts.append(input_blocks[:, :prefix, :].reshape(-1, input_features))
        dense_parts.append(
            input_blocks[:, valid_width:valid_width + 1, :].reshape(
                -1, input_features
            )
        )
        dense_input = (
            dense_parts[0] if len(dense_parts) == 1 else torch.cat(dense_parts, dim=0)
        )
        base_input = input_blocks[:, prefix:valid_width, :].reshape(
            -1, input_features
        )
        if capacity_padding:
            if fixed_block_capacity_zero_dummy_enabled():
                dense_padding = dense_input.new_zeros(
                    (dense_compute_count - dense_count, input_features)
                )
                base_padding = base_input.new_zeros(
                    (base_compute_count - base_count, input_features)
                )
            else:
                dense_padding = dense_input.new_empty(
                    (dense_compute_count - dense_count, input_features)
                )
                base_padding = base_input.new_empty(
                    (base_compute_count - base_count, input_features)
                )
            dense_input = torch.cat([dense_input, dense_padding], dim=0)
            base_input = torch.cat([base_input, base_padding], dim=0)
    _breakdown_record_cuda("row_routed_down_fixed_block_view_cuda_ms", view_timer)

    if linear_breakdown:
        _breakdown_count("row_routed_down_fixed_block_calls", 1)
        _breakdown_count("row_routed_down_fixed_block_rows", rows)
        _breakdown_count("row_routed_down_fixed_block_active_requests", active_count)
        _breakdown_count("row_routed_down_fixed_block_scheduled_width", scheduled_width)
        _breakdown_count("row_routed_down_fixed_block_prefix", prefix)
        _breakdown_count("row_routed_down_fixed_block_dense_rows", dense_count)
        _breakdown_count("row_routed_down_fixed_block_base_rows", base_count)
        _breakdown_count(
            "row_routed_down_fixed_block_dense_compute_rows",
            dense_compute_count,
        )
        _breakdown_count(
            "row_routed_down_fixed_block_base_compute_rows",
            base_compute_count,
        )

    dense_output: torch.Tensor | None = None
    if (
        route_overlap_streams()
        and input_tensor.is_cuda
        and torch.cuda.is_available()
    ):
        current_stream = torch.cuda.current_stream(input_tensor.device)
        _, dense_stream = _route_overlap_streams_for_device(input_tensor.device)
        with torch.cuda.stream(dense_stream):
            dense_stream.wait_stream(current_stream)
            dense_timer = _breakdown_cuda_start() if linear_breakdown else None
            dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
            _breakdown_record_cuda_module(
                module,
                "row_routed_down_fixed_block_dense_gemm_cuda_ms",
                dense_timer,
            )
        base_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_output = _semi_structured_linear(
            base_input,
            _sparse_base_weight(module),
            bias,
        )
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_fixed_block_base_sparse_cuda_ms",
            base_timer,
        )
        current_stream.wait_stream(dense_stream)
        if linear_breakdown:
            _breakdown_count("row_routed_down_fixed_block_overlap_stream_calls", 1)
    else:
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_fixed_block_dense_gemm_cuda_ms",
            dense_timer,
        )

        base_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_output = _semi_structured_linear(
            base_input,
            _sparse_base_weight(module),
            bias,
        )
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_fixed_block_base_sparse_cuda_ms",
            base_timer,
        )

    if dense_output is None:
        return None

    dense_output_active = dense_output[:dense_count]
    base_output_active = base_output[:base_count]
    out_features = int(dense_output_active.shape[1])
    assemble_timer = _breakdown_cuda_start() if linear_breakdown else None
    if triton_route_assembly():
        output = (
            _fixed_block_input_buffer(
                f"down_output:{id(module)}",
                rows,
                out_features,
                dense_output_active,
            )
            if fixed_block_output_buffer_enabled()
            else None
        )
        output = _triton_fixed_block_assemble(
            dense_output_active,
            base_output_active,
            active_count=active_count,
            scheduled_width=scheduled_width,
            prefix=prefix,
            valid_width=valid_width,
            base_width=base_width,
            out_features=out_features,
            output=output,
        )
        _breakdown_record_cuda(
            "row_routed_down_fixed_block_triton_assemble_cuda_ms",
            assemble_timer,
        )
        if linear_breakdown:
            _breakdown_count("row_routed_down_fixed_block_triton_assemble_calls", 1)
            if fixed_block_output_buffer_enabled():
                _breakdown_count(
                    "row_routed_down_fixed_block_output_buffer_calls", 1
                )
        return output
    if fixed_block_output_buffer_enabled():
        output = _fixed_block_input_buffer(
            f"down_output:{id(module)}",
            rows,
            out_features,
            dense_output_active,
        )
        output_blocks = output.view(active_count, scheduled_width, out_features)
        if linear_breakdown:
            _breakdown_count("row_routed_down_fixed_block_output_buffer_calls", 1)
    else:
        output_blocks = torch.empty(
            (active_count, scheduled_width, out_features),
            dtype=dense_output_active.dtype,
            device=dense_output_active.device,
        )
    dense_prefix_rows = active_count * prefix
    if prefix > 0:
        output_blocks[:, :prefix, :].copy_(
            dense_output_active[:dense_prefix_rows].reshape(
                active_count, prefix, out_features
            )
        )
    output_blocks[:, valid_width, :].copy_(
        dense_output_active[dense_prefix_rows:dense_count].reshape(
            active_count, out_features
        )
    )
    output_blocks[:, prefix:valid_width, :].copy_(
        base_output_active.reshape(active_count, base_width, out_features)
    )
    _breakdown_record_cuda(
        "row_routed_down_fixed_block_assemble_cuda_ms",
        assemble_timer,
    )
    return output_blocks.reshape(rows, out_features)


@torch.inference_mode()
def row_routed_down_output(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Route only down_proj rows for mixed SR24 plans.

    This is narrower than row-routed MLP: it keeps the normal activation path
    intact and only avoids computing sparse down_proj for rows that will be
    overwritten by dense residual correction.
    """
    linear_breakdown = breakdown_linear_enabled()
    if linear_breakdown:
        _breakdown_count("row_routed_down_entered", 1)
    if not enabled() or not row_routed_down_linear():
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_disabled", 1)
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 row-routed down path expects 2D tensors")
    if mode() == "base_only" or _current_residual_state() != "mixed":
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_non_mixed_state", 1)
        return None
    module_leaf = str(getattr(module, "_speclink_sr24_profile_leaf", "") or "")
    if module_leaf != "down_proj":
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_wrong_leaf", 1)
        return None
    if not sparse_backend_active(module):
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_inactive_sparse_backend", 1)
        return None
    if getattr(module, "_speclink_sr24_residual_backend", "") != "dense_rows":
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_non_dense_rows_residual", 1)
        return None
    dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
    if dense_weight is None:
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_missing_dense_weight", 1)
        return None

    residual_mask = _residual_mask_for_input(input_tensor)
    if residual_mask is None:
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_missing_residual_mask", 1)
        return None
    rows = int(input_tensor.shape[0])
    dense_rows = _current_residual_rows()
    base_rows = _current_base_rows()
    if (
        dense_rows is not None
        and base_rows is not None
        and int(dense_rows.numel()) + int(base_rows.numel()) == rows
    ):
        dense_rows = dense_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        base_rows = base_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        _breakdown_count("row_routed_down_cached_plan_hits", 1)
    else:
        residual_priority = _residual_priority_for_input(input_tensor)
        residual_bucket = _residual_bucket_for_mask(
            input_tensor,
            residual_mask,
            residual_priority,
        )
        if residual_bucket is not None:
            bucket_rows, bucket_values = residual_bucket
            if bucket_dense_copy_active_only():
                active = bucket_values.to(dtype=torch.bool).nonzero(
                    as_tuple=False
                ).squeeze(1)
                dense_rows = bucket_rows.index_select(0, active)
            else:
                dense_rows = bucket_rows
        else:
            dense_rows = residual_mask.to(
                device=input_tensor.device,
                dtype=torch.bool,
            ).nonzero(as_tuple=False).squeeze(1)
        dense_rows = dense_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        route_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        route_dense = torch.zeros(rows, dtype=torch.bool, device=input_tensor.device)
        if int(dense_rows.numel()) > 0:
            route_dense.index_fill_(0, dense_rows, True)
        base_rows = (~route_dense).nonzero(as_tuple=False).squeeze(1)
        _breakdown_record_cuda("row_routed_down_route_build_cuda_ms", route_timer)

    dense_count = int(dense_rows.numel())
    if dense_count <= 0:
        if linear_breakdown:
            _breakdown_count("row_routed_down_skip_empty_dense_rows", 1)
        return None
    min_dense_rows = row_routed_mlp_min_dense_rows_for_leaf(module_leaf)
    if dense_count < min_dense_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_down_skipped_small_dense_rows", 1)
            _breakdown_count("row_routed_down_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_down_skipped_total_rows", rows)
            _breakdown_count("row_routed_down_min_dense_rows", min_dense_rows)
        return None
    max_dense_rows = row_routed_mlp_max_dense_rows_for_leaf(module_leaf)
    if max_dense_rows > 0 and dense_count > max_dense_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_down_skipped_large_dense_rows", 1)
            _breakdown_count("row_routed_down_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_down_skipped_total_rows", rows)
            _breakdown_count("row_routed_down_max_dense_rows", max_dense_rows)
        return None
    bias = getattr(module, "bias", None)
    if getattr(module, "skip_bias_add", False):
        bias = None
    base_count = int(base_rows.numel()) if base_rows is not None else 0
    min_base_rows = route_min_base_rows_for_leaf(module_leaf)
    if min_base_rows > 0 and 0 < base_count < min_base_rows:
        dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias)
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_fallback_small_base_cuda_ms",
            dense_timer,
        )
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_down_fallback_small_base_calls", 1)
            _breakdown_count("row_routed_down_fallback_small_base_rows", base_count)
            _breakdown_count("row_routed_down_min_base_rows", min_base_rows)
        return output.contiguous()
    max_base_rows = row_routed_mlp_max_base_rows_for_leaf(module_leaf)
    if max_base_rows > 0 and base_count > max_base_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_down_skipped_large_base_rows", 1)
            _breakdown_count("row_routed_down_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_down_skipped_base_rows", base_count)
            _breakdown_count("row_routed_down_skipped_total_rows", rows)
            _breakdown_count("row_routed_down_max_base_rows", max_base_rows)
        return None

    if dense_count >= rows:
        dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias)
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_full_dense_cuda_ms",
            dense_timer,
        )
        if breakdown_linear_enabled():
            _breakdown_count_module(module, "row_routed_down_full_dense_calls", 1)
            _breakdown_count_module(module, "row_routed_down_rows", rows)
            _breakdown_count_module(module, "row_routed_down_dense_rows", rows)
            _breakdown_count_module(module, "row_routed_down_base_rows", 0)
        return output.contiguous()

    if breakdown_linear_enabled():
        _breakdown_count_module(module, "row_routed_down_calls", 1)
        _breakdown_count_module(module, "row_routed_down_rows", rows)
        _breakdown_count_module(module, "row_routed_down_dense_rows", dense_count)
        _breakdown_count_module(module, "row_routed_down_base_rows", base_count)
        _breakdown_count("row_routed_down_executed", 1)

    fixed_block_output = _row_routed_down_fixed_block_output(
        module,
        input_tensor,
        dense_weight,
        bias,
        dense_rows,
        base_rows,
    )
    if fixed_block_output is not None:
        return fixed_block_output

    contiguous_output = _routed_contiguous_rows_output(
        module,
        input_tensor,
        dense_weight,
        bias,
        dense_rows,
        route_label="row_routed_down",
    )
    if contiguous_output is not None:
        return contiguous_output

    base_rows = (
        base_rows.to(device=input_tensor.device, dtype=torch.long, non_blocking=True)
        if base_rows is not None
        else torch.empty(0, dtype=torch.long, device=input_tensor.device)
    )
    if (
        int(base_rows.numel()) > 0
        and route_overlap_streams()
        and input_tensor.is_cuda
        and torch.cuda.is_available()
    ):
        current_stream = torch.cuda.current_stream(input_tensor.device)
        _, dense_stream = _route_overlap_streams_for_device(input_tensor.device)
        dense_output: torch.Tensor | None = None

        with torch.cuda.stream(dense_stream):
            dense_stream.wait_stream(current_stream)
            dense_gather_timer = (
                _breakdown_cuda_start() if breakdown_linear_enabled() else None
            )
            dense_input = input_tensor.index_select(0, dense_rows)
            _breakdown_record_cuda(
                "row_routed_down_dense_gather_cuda_ms", dense_gather_timer
            )
            dense_timer = (
                _breakdown_cuda_start() if breakdown_linear_enabled() else None
            )
            dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
            _breakdown_record_cuda_module(
                module,
                "row_routed_down_dense_gemm_cuda_ms",
                dense_timer,
            )

        base_gather_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_input = input_tensor.index_select(0, base_rows)
        _breakdown_record_cuda(
            "row_routed_down_base_gather_cuda_ms", base_gather_timer
        )
        base_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        base_output = _semi_structured_linear(
            base_input, _sparse_base_weight(module), bias
        )
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_base_sparse_cuda_ms",
            base_timer,
        )

        current_stream.wait_stream(dense_stream)
        if dense_output is None:
            return None
        out_features = int(dense_output.shape[1])
        assemble_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        if triton_route_assembly():
            output = _triton_routed_assemble(
                dense_output,
                dense_rows,
                base_output,
                base_rows,
                total_rows=rows,
                out_features=out_features,
            )
            _breakdown_record_cuda(
                "row_routed_down_triton_assemble_cuda_ms", assemble_timer
            )
        else:
            output = torch.empty(
                (rows, out_features),
                dtype=dense_output.dtype,
                device=dense_output.device,
            )
            output.index_copy_(0, base_rows, base_output)
            output.index_copy_(0, dense_rows, dense_output)
            _breakdown_record_cuda(
                "row_routed_down_index_copy_cuda_ms", assemble_timer
            )
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_down_overlap_stream_calls", 1)
        return output.contiguous()

    dense_gather_timer = (
        _breakdown_cuda_start() if breakdown_linear_enabled() else None
    )
    dense_input = input_tensor.index_select(0, dense_rows)
    _breakdown_record_cuda(
        "row_routed_down_dense_gather_cuda_ms", dense_gather_timer
    )
    dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
    _breakdown_record_cuda_module(
        module,
        "row_routed_down_dense_gemm_cuda_ms",
        dense_timer,
    )

    out_features = int(dense_output.shape[1])
    if int(base_rows.numel()) > 0:
        base_gather_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_input = input_tensor.index_select(0, base_rows)
        _breakdown_record_cuda(
            "row_routed_down_base_gather_cuda_ms", base_gather_timer
        )
        base_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        base_output = _semi_structured_linear(
            base_input, _sparse_base_weight(module), bias
        )
        _breakdown_record_cuda_module(
            module,
            "row_routed_down_base_sparse_cuda_ms",
            base_timer,
        )
        assemble_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        if triton_route_assembly():
            output = _triton_routed_assemble(
                dense_output,
                dense_rows,
                base_output,
                base_rows,
                total_rows=rows,
                out_features=out_features,
            )
            _breakdown_record_cuda(
                "row_routed_down_triton_assemble_cuda_ms", assemble_timer
            )
        else:
            output = torch.empty(
                (rows, out_features),
                dtype=dense_output.dtype,
                device=dense_output.device,
            )
            output.index_copy_(0, base_rows, base_output)
            output.index_copy_(0, dense_rows, dense_output)
            _breakdown_record_cuda(
                "row_routed_down_index_copy_cuda_ms", assemble_timer
            )
    else:
        assemble_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        output = torch.empty(
            (rows, out_features),
            dtype=dense_output.dtype,
            device=dense_output.device,
        )
        output.index_copy_(0, dense_rows, dense_output)
        _breakdown_record_cuda("row_routed_down_index_copy_cuda_ms", assemble_timer)
    return output.contiguous()


@torch.inference_mode()
def noverify_dense_mlp_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    act_fn: Any | None = None,
) -> torch.Tensor | None:
    linear_breakdown = breakdown_linear_enabled()
    if linear_breakdown:
        _breakdown_count("noverify_dense_mlp_entered", 1)
    if not enabled() or not noverify_dense_mlp_fastpath():
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_disabled", 1)
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 no-verify MLP path expects 2D tensors")
    rows = int(input_tensor.shape[0])
    if mode() == "base_only":
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_base_only", 1)
        return None
    if _current_residual_state() is not None or _current_residual_mask() is not None:
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_active_residual_context", 1)
        return None
    sr_mode = mode()
    if not (
        (
            sr_mode == "all_corrected"
            and _all_residual_dense_shortcut_enabled()
        )
        or (
            sr_mode == "selective"
            and _selective_dense_nonverify_for_module_rows(gate_up_module, rows)
        )
    ):
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_mode_policy", 1)
        return None
    if not (
        sparse_backend_active(gate_up_module)
        and sparse_backend_active(down_module)
    ):
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_inactive_sparse_backend", 1)
        return None
    dense_weight = getattr(gate_up_module, "_speclink_sr24_dense_weight", None)
    down_dense_weight = getattr(down_module, "_speclink_sr24_dense_weight", None)
    if dense_weight is None or down_dense_weight is None:
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_missing_dense_weight", 1)
        return None

    down_module_weight = getattr(down_module, "weight", None)
    down_needs_reduce = (
        bool(getattr(down_module, "reduce_results", False))
        and int(getattr(down_module, "tp_size", 1) or 1) > 1
    )
    if down_needs_reduce and down_module_weight is not down_dense_weight:
        if linear_breakdown:
            _breakdown_count("noverify_dense_mlp_skip_reduce_mismatch", 1)
        return None

    gate_bias = getattr(gate_up_module, "bias", None)
    if getattr(gate_up_module, "skip_bias_add", False):
        gate_bias = None
    down_bias = getattr(down_module, "bias", None)
    if getattr(down_module, "skip_bias_add", False):
        down_bias = None

    gate_timer = _breakdown_cuda_start() if linear_breakdown else None
    gate_up = _sr24_exact_dense_linear(
        gate_up_module,
        input_tensor,
        dense_weight,
        gate_bias,
    )
    _breakdown_record_cuda_module(
        gate_up_module,
        "noverify_dense_mlp_gate_up_cuda_ms",
        gate_timer,
    )

    act_timer = _breakdown_cuda_start() if linear_breakdown else None
    routed_act = act_fn(gate_up) if act_fn is not None else _silu_and_mul_local(
        gate_up,
        int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
    )
    _breakdown_record_cuda("noverify_dense_mlp_act_cuda_ms", act_timer)

    down_timer = _breakdown_cuda_start() if linear_breakdown else None
    output = _sr24_exact_dense_linear(
        down_module,
        routed_act,
        down_dense_weight,
        down_bias,
    )
    _breakdown_record_cuda_module(
        down_module,
        "noverify_dense_mlp_down_cuda_ms",
        down_timer,
    )
    if linear_breakdown:
        _breakdown_count("noverify_dense_mlp_calls", 1)
        _breakdown_count("noverify_dense_mlp_rows", rows)
        _breakdown_count_module(
            gate_up_module,
            "noverify_dense_mlp_gate_up_calls",
            1,
        )
        _breakdown_count_module(
            gate_up_module,
            "noverify_dense_mlp_gate_up_rows",
            rows,
        )
        _breakdown_count_module(
            down_module,
            "noverify_dense_mlp_down_calls",
            1,
        )
        _breakdown_count_module(
            down_module,
            "noverify_dense_mlp_down_rows",
            rows,
        )
    return output.contiguous()


def _fixed_block_input_buffer(
    name: str,
    rows: int,
    hidden_size: int,
    like: torch.Tensor,
) -> torch.Tensor:
    rows = int(rows)
    hidden_size = int(hidden_size)
    device = like.device
    key = (
        name,
        device.type,
        -1 if device.index is None else int(device.index),
        like.dtype,
        rows,
        hidden_size,
    )
    cached = _fixed_block_input_buffer_cache.get(key)
    if cached is None or cached.device != device or cached.dtype != like.dtype:
        cached = torch.empty((rows, hidden_size), dtype=like.dtype, device=device)
        _fixed_block_input_buffer_cache[key] = cached
    return cached


_ROW_ROUTED_MLP_DENSE_BYPASS = object()


@torch.inference_mode()
def _row_routed_mlp_fixed_block_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    down_dense_weight: torch.Tensor | None,
    dense_rows: torch.Tensor,
    base_rows: torch.Tensor | None,
    *,
    down_uses_exact_dense_rows: bool,
    act_fn: Any | None,
) -> torch.Tensor | None:
    """Route fixed-prefix request blocks without dynamic row gather/scatter.

    The scheduler's fixed-prefix+bonus route emits one regular block per
    request: first `prefix` draft rows are dense, middle draft rows are sparse,
    and the verifier bonus row is dense.  The generic row-routed MLP consumes
    flat row-id tensors and therefore pays index_select/index_copy overhead.
    This path keeps the same disjoint dense/sparse semantics but uses stable
    request-block slices, which is closer to the desired graph-friendly route
    table operator.
    """
    if not route_contiguous_fastpath() or not fixed_prefix_route_fastpath_enabled():
        return None
    rows = int(input_tensor.shape[0])
    fixed_route = _current_fixed_prefix_route()
    if (
        fixed_route is not None
        and int(fixed_route.active_count) > 0
        and int(fixed_route.scheduled_width) > 1
        and int(fixed_route.active_count) * int(fixed_route.scheduled_width) == rows
    ):
        dense_count = int(fixed_route.active_count) * int(fixed_route.dense_width)
        base_count = int(fixed_route.active_count) * int(fixed_route.base_width)
        if (
            dense_rows is not None
            and base_rows is not None
            and (
                int(dense_rows.numel()) != dense_count
                or int(base_rows.numel()) != base_count
            )
        ):
            fixed_route = None
    else:
        fixed_route = None
    if fixed_route is None:
        if dense_rows is None or base_rows is None:
            return None
        dense_count = int(dense_rows.numel())
        base_count = int(base_rows.numel())
    module_leaf = str(
        getattr(gate_up_module, "_speclink_sr24_profile_leaf", "") or "gate_up_proj"
    )
    dense_fraction = dense_count / max(rows, 1)
    max_dense_fraction = route_max_dense_fraction()
    if 0.0 <= max_dense_fraction <= 1.0 and dense_fraction > max_dense_fraction:
        return None
    linear_breakdown = breakdown_linear_enabled()
    min_base_rows = route_min_base_rows_for_leaf(module_leaf)
    if min_base_rows > 0 and base_count < min_base_rows:
        dense_output = _row_routed_mlp_full_dense_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
            reason="fixed_block_small_base",
        )
        if dense_output is not None:
            if linear_breakdown:
                _breakdown_count(
                    "row_routed_mlp_fixed_block_dense_fallback_small_base",
                    1,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_small_base_rows",
                    base_count,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_small_base_min_rows",
                    min_base_rows,
                )
            return dense_output
        return None
    prefix = max(0, int(selective_min_prefix_residual()))
    if fixed_route is not None:
        descriptor_matches = (
            int(fixed_route.active_count) > 0
            and int(fixed_route.scheduled_width) > 1
            and int(fixed_route.prefix) == prefix
            and int(fixed_route.active_count) * int(fixed_route.scheduled_width)
            == rows
            and int(fixed_route.active_count) * int(fixed_route.dense_width)
            == dense_count
            and int(fixed_route.active_count) * int(fixed_route.base_width)
            == base_count
        )
        if descriptor_matches:
            active_count = int(fixed_route.active_count)
            scheduled_width = int(fixed_route.scheduled_width)
            valid_width = int(fixed_route.valid_width)
            group_width = int(fixed_route.dense_width)
            base_width = int(fixed_route.base_width)
            if linear_breakdown:
                _breakdown_count("row_routed_mlp_fixed_block_descriptor_hits", 1)
        else:
            if linear_breakdown:
                _breakdown_count(
                    "row_routed_mlp_fixed_block_descriptor_misses", 1
                )
            fixed_route = None
    if fixed_route is None:
        group_width = prefix + 1
        if (
            rows <= 0
            or dense_count <= 0
            or base_count <= 0
            or group_width <= 0
            or dense_count % group_width != 0
        ):
            return None
        active_count = dense_count // group_width
        if active_count <= 0 or rows % active_count != 0:
            return None
        scheduled_width = rows // active_count
        valid_width = scheduled_width - 1
        if valid_width < prefix:
            return None
        base_width = valid_width - prefix
        if base_width <= 0 or base_count != active_count * base_width:
            return None

    policy_match = _scheduler_policy_for_active_count(active_count)
    matched_batch: int | None = None
    policy: dict[str, Any] | None = None
    if policy_match is not None:
        matched_batch, policy = policy_match
        dense_fallback, fallback_reason, min_grouped = (
            _scheduler_policy_requires_dense_fallback(
                policy,
                prefix=prefix,
                valid_width=valid_width,
            )
        )
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_fixed_block_scheduler_policy_hits", 1)
            _breakdown_count(
                "row_routed_mlp_fixed_block_scheduler_policy_active_requests",
                active_count,
            )
            _breakdown_count(
                "row_routed_mlp_fixed_block_scheduler_policy_matched_batch",
                matched_batch,
            )
        if fallback_reason == "incompatible":
            if linear_breakdown:
                _breakdown_count(
                    "row_routed_mlp_fixed_block_scheduler_policy_incompatible",
                    1,
                )
        elif dense_fallback:
            if scheduler_policy_dense_bypass():
                if linear_breakdown:
                    _breakdown_count(
                        "row_routed_mlp_fixed_block_scheduler_policy_dense_bypass",
                        1,
                    )
                    if fallback_reason == "operator_unimplemented":
                        _breakdown_count(
                            "row_routed_mlp_fixed_block_scheduler_policy_operator_unimplemented",
                            1,
                        )
                    if min_grouped is not None:
                        _breakdown_count(
                            "row_routed_mlp_fixed_block_scheduler_policy_min_grouped",
                            min_grouped,
                        )
                dense_output = _row_routed_mlp_full_dense_output(
                    gate_up_module,
                    down_module,
                    input_tensor,
                    dense_weight,
                    down_dense_weight,
                    down_uses_exact_dense_rows=down_uses_exact_dense_rows,
                    act_fn=act_fn,
                    reason=f"scheduler_policy_dense_bypass_{fallback_reason}",
                )
                if dense_output is not None:
                    if linear_breakdown:
                        _breakdown_count(
                            "row_routed_mlp_fixed_block_scheduler_policy_"
                            "dense_bypass_full_dense",
                            1,
                        )
                    return dense_output
                return _ROW_ROUTED_MLP_DENSE_BYPASS
            dense_output = _row_routed_mlp_full_dense_output(
                gate_up_module,
                down_module,
                input_tensor,
                dense_weight,
                down_dense_weight,
                down_uses_exact_dense_rows=down_uses_exact_dense_rows,
                act_fn=act_fn,
                reason=f"scheduler_policy_{fallback_reason}",
            )
            if dense_output is not None:
                if linear_breakdown:
                    if fallback_reason == "operator_unimplemented":
                        _breakdown_count(
                            "row_routed_mlp_fixed_block_scheduler_policy_operator_unimplemented",
                            1,
                        )
                    _breakdown_count(
                        "row_routed_mlp_fixed_block_scheduler_policy_dense_fallback",
                        1,
                    )
                    if min_grouped is not None:
                        _breakdown_count(
                            "row_routed_mlp_fixed_block_scheduler_policy_min_grouped",
                            min_grouped,
                        )
                return dense_output
            return None
        elif linear_breakdown:
            _breakdown_count(
                "row_routed_mlp_fixed_block_scheduler_policy_mixed_allowed",
                1,
            )

    promoted_width = 0
    min_dense_rows = row_routed_mlp_min_dense_rows_for_leaf(module_leaf)
    if (
        row_routed_mlp_fixed_block_dense_fill()
        and dense_count < min_dense_rows
        and base_width > 0
    ):
        need_rows = min_dense_rows - dense_count
        promoted_width = min(
            base_width, (need_rows + active_count - 1) // active_count
        )
        if promoted_width > 0:
            promoted_rows = promoted_width * active_count
            dense_count += promoted_rows
            base_width -= promoted_width
            base_count -= promoted_rows
            if linear_breakdown:
                _breakdown_count("row_routed_mlp_fixed_block_fill_dense_calls", 1)
                _breakdown_count(
                    "row_routed_mlp_fixed_block_fill_dense_rows", promoted_rows
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_fill_dense_width", promoted_width
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_fill_dense_target", min_dense_rows
                )
    if base_width <= 0 or base_count <= 0:
        dense_output = _row_routed_mlp_full_dense_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
            reason="fixed_block_filled_all_rows",
        )
        if dense_output is not None:
            return dense_output
        return None

    hidden_size = int(input_tensor.shape[1])
    dense_compute_count = dense_count
    base_compute_count = base_count
    capacity_padding = False
    if (
        fixed_block_capacity_padding_enabled()
        and policy is not None
        and matched_batch is not None
        and matched_batch > active_count
        and promoted_width == 0
    ):
        dense_capacity = (
            _scheduler_policy_int(policy, "dense_capacity")
            or _scheduler_policy_int(policy, "grouped_dense_rows")
        )
        base_capacity = (
            _scheduler_policy_int(policy, "base_capacity")
            or _scheduler_policy_int(policy, "grouped_base_rows")
        )
        if (
            dense_capacity is not None
            and base_capacity is not None
            and dense_capacity >= dense_count
            and base_capacity >= base_count
        ):
            dense_compute_count = int(dense_capacity)
            base_compute_count = int(base_capacity)
            capacity_padding = (
                dense_compute_count > dense_count
                or base_compute_count > base_count
            )
            if linear_breakdown and capacity_padding:
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_calls", 1
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_active_requests",
                    active_count,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_matched_batch",
                    matched_batch,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_dense_rows",
                    dense_count,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_dense_capacity",
                    dense_compute_count,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_base_rows",
                    base_count,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_base_capacity",
                    base_compute_count,
                )
                _breakdown_count(
                    "row_routed_mlp_fixed_block_capacity_padding_zero_dummy",
                    1 if fixed_block_capacity_zero_dummy_enabled() else 0,
                )
    if linear_breakdown:
        _breakdown_count("row_routed_mlp_fixed_block_calls", 1)
        _breakdown_count("row_routed_mlp_fixed_block_rows", rows)
        _breakdown_count("row_routed_mlp_fixed_block_active_requests", active_count)
        _breakdown_count("row_routed_mlp_fixed_block_scheduled_width", scheduled_width)
        _breakdown_count("row_routed_mlp_fixed_block_prefix", prefix)
        _breakdown_count("row_routed_mlp_fixed_block_dense_rows", dense_count)
        _breakdown_count("row_routed_mlp_fixed_block_base_rows", base_count)
        _breakdown_count(
            "row_routed_mlp_fixed_block_dense_compute_rows", dense_compute_count
        )
        _breakdown_count(
            "row_routed_mlp_fixed_block_base_compute_rows", base_compute_count
        )

    view_timer = _breakdown_cuda_start() if linear_breakdown else None
    input_blocks = input_tensor.reshape(active_count, scheduled_width, hidden_size)
    base_start = prefix + promoted_width
    dense_prefix_rows = active_count * prefix
    dense_promoted_rows = active_count * promoted_width
    dense_bonus_start = dense_prefix_rows + dense_promoted_rows
    if fixed_block_input_buffer_enabled():
        dense_input = _fixed_block_input_buffer(
            "dense",
            dense_compute_count,
            hidden_size,
            input_tensor,
        )
        if prefix > 0:
            dense_input[:dense_prefix_rows].view(
                active_count, prefix, hidden_size
            ).copy_(input_blocks[:, :prefix, :])
        if promoted_width > 0:
            dense_input[
                dense_prefix_rows:dense_bonus_start
            ].view(active_count, promoted_width, hidden_size).copy_(
                input_blocks[:, prefix:prefix + promoted_width, :]
            )
        dense_input[
            dense_bonus_start:dense_bonus_start + active_count
        ].view(active_count, 1, hidden_size).copy_(
            input_blocks[:, valid_width:valid_width + 1, :]
        )
        if (
            dense_compute_count > dense_count
            and fixed_block_capacity_zero_dummy_enabled()
        ):
            dense_input[dense_count:dense_compute_count].zero_()
        base_input = _fixed_block_input_buffer(
            "base",
            base_compute_count,
            hidden_size,
            input_tensor,
        )
        base_input[:base_count].view(active_count, base_width, hidden_size).copy_(
            input_blocks[:, base_start:valid_width, :]
        )
        if (
            base_compute_count > base_count
            and fixed_block_capacity_zero_dummy_enabled()
        ):
            base_input[base_count:base_compute_count].zero_()
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_fixed_block_input_buffer_calls", 1)
            _breakdown_count(
                "row_routed_mlp_fixed_block_input_buffer_dense_rows",
                dense_compute_count,
            )
            _breakdown_count(
                "row_routed_mlp_fixed_block_input_buffer_base_rows",
                base_compute_count,
            )
    else:
        dense_parts: list[torch.Tensor] = []
        if prefix > 0:
            dense_parts.append(input_blocks[:, :prefix, :].reshape(-1, hidden_size))
        if promoted_width > 0:
            dense_parts.append(
                input_blocks[
                    :, prefix:prefix + promoted_width, :
                ].reshape(-1, hidden_size)
            )
        dense_parts.append(
            input_blocks[:, valid_width:valid_width + 1, :].reshape(-1, hidden_size)
        )
        dense_input = (
            dense_parts[0] if len(dense_parts) == 1 else torch.cat(dense_parts, dim=0)
        )
        base_input = input_blocks[:, base_start:valid_width, :].reshape(
            -1, hidden_size
        )
        if capacity_padding:
            if fixed_block_capacity_zero_dummy_enabled():
                dense_padding = dense_input.new_zeros(
                    (dense_compute_count - dense_count, hidden_size)
                )
                base_padding = base_input.new_zeros(
                    (base_compute_count - base_count, hidden_size)
                )
            else:
                dense_padding = dense_input.new_empty(
                    (dense_compute_count - dense_count, hidden_size)
                )
                base_padding = base_input.new_empty(
                    (base_compute_count - base_count, hidden_size)
                )
            dense_input = torch.cat([dense_input, dense_padding], dim=0)
            base_input = torch.cat([base_input, base_padding], dim=0)
    _breakdown_record_cuda("row_routed_mlp_fixed_block_view_cuda_ms", view_timer)

    intermediate_size = int(getattr(down_module, "_speclink_sr24_weight_shape")[1])
    if row_routed_mlp_reuse_base_output():
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_reuse_base_output_calls", 1)
            _breakdown_count("row_routed_mlp_reuse_base_output_rows", rows)
            _breakdown_count(
                "row_routed_mlp_reuse_base_output_dense_rows", dense_count
            )
            _breakdown_count(
                "row_routed_mlp_fixed_block_reuse_base_output_calls", 1
            )

        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
        _breakdown_record_cuda("row_routed_mlp_dense_gate_up_cuda_ms", dense_timer)

        base_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_gate_up = _semi_structured_linear(
            input_tensor, _sparse_base_weight(gate_up_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_gate_up_cuda_ms", base_timer
        )

        dense_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_act = (
            act_fn(dense_gate_up)
            if act_fn is not None
            else _silu_and_mul_local(dense_gate_up, intermediate_size)
        )
        _breakdown_record_cuda("row_routed_mlp_dense_act_cuda_ms", dense_act_timer)

        base_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_act = (
            act_fn(base_gate_up)
            if act_fn is not None
            else _silu_and_mul_local(base_gate_up, intermediate_size)
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_act_cuda_ms", base_act_timer
        )

        dense_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        if down_uses_exact_dense_rows:
            if down_dense_weight is None:
                return None
            dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
        else:
            dense_down = _semi_structured_linear(
                dense_act, _sparse_base_weight(down_module), None
            )
        _breakdown_record_cuda("row_routed_mlp_dense_down_cuda_ms", dense_down_timer)

        base_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_down = _semi_structured_linear(
            base_act, _sparse_base_weight(down_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_down_sparse_cuda_ms", base_down_timer
        )

        out_features = int(base_down.shape[1])
        assemble_timer = _breakdown_cuda_start() if linear_breakdown else None
        output_blocks = base_down.reshape(active_count, scheduled_width, out_features)
        dense_prefix_rows = active_count * prefix
        dense_promoted_rows = active_count * promoted_width
        dense_bonus_start = dense_prefix_rows + dense_promoted_rows
        dense_down_active = dense_down[:dense_count]
        if prefix > 0:
            output_blocks[:, :prefix, :].copy_(
                dense_down_active[:dense_prefix_rows].reshape(
                    active_count, prefix, out_features
                )
            )
        if promoted_width > 0:
            output_blocks[:, prefix:prefix + promoted_width, :].copy_(
                dense_down_active[dense_prefix_rows:dense_bonus_start].reshape(
                    active_count, promoted_width, out_features
                )
            )
        output_blocks[:, valid_width, :].copy_(
            dense_down_active[dense_bonus_start:dense_count].reshape(
                active_count, out_features
            )
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_index_copy_cuda_ms", assemble_timer
        )
        return base_down.contiguous()

    dense_down = None
    if (
        route_overlap_streams()
        and input_tensor.is_cuda
        and torch.cuda.is_available()
    ):
        current_stream = torch.cuda.current_stream(input_tensor.device)
        _, dense_stream = _route_overlap_streams_for_device(input_tensor.device)

        with torch.cuda.stream(dense_stream):
            dense_stream.wait_stream(current_stream)
            dense_timer = _breakdown_cuda_start() if linear_breakdown else None
            dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
            _breakdown_record_cuda(
                "row_routed_mlp_fixed_block_dense_gate_up_cuda_ms", dense_timer
            )

            dense_act_timer = _breakdown_cuda_start() if linear_breakdown else None
            dense_act = (
                act_fn(dense_gate_up)
                if act_fn is not None
                else _silu_and_mul_local(dense_gate_up, intermediate_size)
            )
            _breakdown_record_cuda(
                "row_routed_mlp_fixed_block_dense_act_cuda_ms", dense_act_timer
            )

            dense_down_timer = _breakdown_cuda_start() if linear_breakdown else None
            if down_uses_exact_dense_rows:
                if down_dense_weight is None:
                    return None
                dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
            else:
                dense_down = _semi_structured_linear(
                    dense_act, _sparse_base_weight(down_module), None
                )
            _breakdown_record_cuda(
                "row_routed_mlp_fixed_block_dense_down_cuda_ms", dense_down_timer
            )

        base_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_gate_up = _semi_structured_linear(
            base_input, _sparse_base_weight(gate_up_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_base_gate_up_cuda_ms", base_timer
        )

        base_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_act = (
            act_fn(base_gate_up)
            if act_fn is not None
            else _silu_and_mul_local(base_gate_up, intermediate_size)
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_base_act_cuda_ms", base_act_timer
        )

        base_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_down = _semi_structured_linear(
            base_act, _sparse_base_weight(down_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_base_down_sparse_cuda_ms", base_down_timer
        )

        current_stream.wait_stream(dense_stream)
        _breakdown_count("row_routed_mlp_fixed_block_overlap_stream_calls", 1)
    else:
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_dense_gate_up_cuda_ms", dense_timer
        )

        base_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_gate_up = _semi_structured_linear(
            base_input, _sparse_base_weight(gate_up_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_base_gate_up_cuda_ms", base_timer
        )

        dense_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        dense_act = (
            act_fn(dense_gate_up)
            if act_fn is not None
            else _silu_and_mul_local(dense_gate_up, intermediate_size)
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_dense_act_cuda_ms", dense_act_timer
        )

        base_act_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_act = (
            act_fn(base_gate_up)
            if act_fn is not None
            else _silu_and_mul_local(base_gate_up, intermediate_size)
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_base_act_cuda_ms", base_act_timer
        )

        dense_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        if down_uses_exact_dense_rows:
            if down_dense_weight is None:
                return None
            dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
        else:
            dense_down = _semi_structured_linear(
                dense_act, _sparse_base_weight(down_module), None
            )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_dense_down_cuda_ms", dense_down_timer
        )

        base_down_timer = _breakdown_cuda_start() if linear_breakdown else None
        base_down = _semi_structured_linear(
            base_act, _sparse_base_weight(down_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_base_down_sparse_cuda_ms", base_down_timer
        )

    dense_down_active = dense_down[:dense_count]
    base_down_active = base_down[:base_count]
    out_features = int(base_down_active.shape[1])
    assemble_timer = _breakdown_cuda_start() if linear_breakdown else None
    if triton_route_assembly():
        output = (
            _fixed_block_input_buffer(
                f"output:{id(gate_up_module)}",
                rows,
                out_features,
                base_down,
            )
            if fixed_block_output_buffer_enabled()
            else None
        )
        output = _triton_fixed_block_assemble(
            dense_down_active,
            base_down_active,
            active_count=active_count,
            scheduled_width=scheduled_width,
            prefix=prefix,
            valid_width=valid_width,
            base_width=base_width,
            out_features=out_features,
            output=output,
        )
        _breakdown_record_cuda(
            "row_routed_mlp_fixed_block_triton_assemble_cuda_ms",
            assemble_timer,
        )
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_fixed_block_triton_assemble_calls", 1)
            if fixed_block_output_buffer_enabled():
                _breakdown_count(
                    "row_routed_mlp_fixed_block_output_buffer_calls", 1
                )
        return output
    if fixed_block_output_buffer_enabled():
        output = _fixed_block_input_buffer(
            f"output:{id(gate_up_module)}",
            rows,
            out_features,
            base_down,
        )
        output_blocks = output.view(active_count, scheduled_width, out_features)
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_fixed_block_output_buffer_calls", 1)
            _breakdown_count("row_routed_mlp_fixed_block_output_buffer_rows", rows)
    else:
        output_blocks = torch.empty(
            (active_count, scheduled_width, out_features),
            dtype=base_down.dtype,
            device=base_down.device,
        )
    if prefix > 0:
        output_blocks[:, :prefix, :].copy_(
            dense_down_active[:dense_prefix_rows].reshape(
                active_count, prefix, out_features
            )
        )
    if promoted_width > 0:
        output_blocks[:, prefix:prefix + promoted_width, :].copy_(
            dense_down_active[dense_prefix_rows:dense_bonus_start].reshape(
                active_count, promoted_width, out_features
            )
        )
    output_blocks[:, valid_width, :].copy_(
        dense_down_active[dense_bonus_start:dense_count].reshape(
            active_count, out_features
        )
    )
    output_blocks[:, base_start:valid_width, :].copy_(
        base_down_active.reshape(active_count, base_width, out_features)
    )
    _breakdown_record_cuda(
        "row_routed_mlp_fixed_block_assemble_cuda_ms", assemble_timer
    )
    return output_blocks.reshape(rows, out_features)


@torch.inference_mode()
def row_routed_mlp_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    act_fn: Any | None = None,
) -> torch.Tensor | None:
    linear_breakdown = breakdown_linear_enabled()
    if linear_breakdown:
        _breakdown_count("row_routed_mlp_entered", 1)
    if not enabled() or not row_routed_mlp():
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_disabled", 1)
        return None
    if input_tensor.ndim != 2:
        raise RuntimeError("SpecLink SR24 row-routed MLP path expects 2D tensors")
    if mode() == "base_only" or _current_residual_state() != "mixed":
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_non_mixed_state", 1)
        return None
    if not (
        sparse_backend_active(gate_up_module)
        and sparse_backend_active(down_module)
        and getattr(gate_up_module, "_speclink_sr24_residual_backend", "") == "dense_rows"
    ):
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_backend_or_residual", 1)
        return None
    if _runtime_base_only_for_module(gate_up_module) or _runtime_base_only_for_module(
        down_module
    ):
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_runtime_base_only", 1)
        return None
    dense_weight = getattr(gate_up_module, "_speclink_sr24_dense_weight", None)
    if dense_weight is None:
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_missing_dense_weight", 1)
        return None
    down_dense_weight = getattr(down_module, "_speclink_sr24_dense_weight", None)
    down_uses_exact_dense_rows = (
        down_dense_weight is not None
        and getattr(down_module, "_speclink_sr24_residual_backend", "") == "dense_rows"
        and not getattr(down_module, "_speclink_sr24_no_residual", False)
    )
    down_is_base_only = getattr(down_module, "_speclink_sr24_no_residual", False)
    if not down_uses_exact_dense_rows and not down_is_base_only:
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_down_policy", 1)
        return None
    module_leaf = str(
        getattr(gate_up_module, "_speclink_sr24_profile_leaf", "") or "gate_up_proj"
    )

    residual_mask = _residual_mask_for_input(input_tensor)
    if residual_mask is None:
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_missing_residual_mask", 1)
        return None
    rows = int(input_tensor.shape[0])
    dense_rows = _current_residual_rows()
    base_rows = _current_base_rows()
    if _current_fixed_prefix_route() is not None:
        fixed_block_output = _row_routed_mlp_fixed_block_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            dense_rows,
            base_rows,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
        )
        if fixed_block_output is _ROW_ROUTED_MLP_DENSE_BYPASS:
            return None
        if fixed_block_output is not None:
            return fixed_block_output
    reuse_base_output = row_routed_mlp_reuse_base_output()
    if (
        dense_rows is not None
        and (
            reuse_base_output
            or (
                base_rows is not None
                and int(dense_rows.numel()) + int(base_rows.numel()) == rows
            )
        )
    ):
        dense_rows = dense_rows.to(
            device=input_tensor.device,
            dtype=torch.long,
            non_blocking=True,
        )
        if base_rows is not None:
            base_rows = base_rows.to(
                device=input_tensor.device,
                dtype=torch.long,
                non_blocking=True,
        )
        _breakdown_count("row_routed_mlp_cached_plan_hits", 1)
    if reuse_base_output and _current_fixed_prefix_route() is not None:
        fixed_block_output = _row_routed_mlp_fixed_block_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            dense_rows,
            base_rows,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
        )
        if fixed_block_output is _ROW_ROUTED_MLP_DENSE_BYPASS:
            return None
        if fixed_block_output is not None:
            return fixed_block_output
        if dense_rows is None:
            return None
    else:
        residual_priority = _residual_priority_for_input(input_tensor)
        residual_bucket = _residual_bucket_for_mask(
            input_tensor,
            residual_mask,
            residual_priority,
        )
        if residual_bucket is None:
            dense_rows = residual_mask.to(
                device=input_tensor.device,
                dtype=torch.bool,
            ).nonzero(as_tuple=False).squeeze(1)
        else:
            bucket_rows, bucket_values = residual_bucket
            if bucket_dense_copy() and not bucket_dense_copy_active_only():
                # Keep row-routed MLP fallback semantics aligned with the
                # conservative bucket dense-copy path: all bucket rows are
                # recomputed densely, including inactive padding/conservative
                # entries selected by the scheduler bucket.
                dense_rows = bucket_rows.to(
                    device=input_tensor.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            else:
                active = bucket_values.to(dtype=torch.bool).nonzero(
                    as_tuple=False).squeeze(1)
                dense_rows = bucket_rows.index_select(0, active).to(
                    device=input_tensor.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
        base_rows = None
    dense_count = int(dense_rows.numel())
    if dense_count <= 0:
        if linear_breakdown:
            _breakdown_count("row_routed_mlp_skip_empty_dense_rows", 1)
        return None
    max_dense_rows = row_routed_mlp_max_dense_rows_for_leaf(module_leaf)
    if max_dense_rows > 0 and dense_count > max_dense_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_mlp_skipped_large_dense_rows", 1)
            _breakdown_count("row_routed_mlp_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_mlp_skipped_total_rows", rows)
            _breakdown_count("row_routed_mlp_max_dense_rows", max_dense_rows)
        return None
    if dense_count >= rows:
        dense_output = _row_routed_mlp_full_dense_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
            reason="all_rows",
        )
        if dense_output is not None:
            return dense_output
        gate_up = _sr24_dense_linear(input_tensor, dense_weight, bias=None)
        routed_act = act_fn(gate_up) if act_fn is not None else _silu_and_mul_local(
            gate_up,
            int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
        )
        return _semi_structured_linear(
            routed_act, _sparse_base_weight(down_module), None
        ).contiguous()

    if reuse_base_output:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_mlp_reuse_base_output_calls", 1)
            _breakdown_count("row_routed_mlp_reuse_base_output_rows", rows)
            _breakdown_count(
                "row_routed_mlp_reuse_base_output_dense_rows", dense_count
            )

        dense_gather_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        dense_input = input_tensor.index_select(0, dense_rows)
        _breakdown_record_cuda(
            "row_routed_mlp_dense_gather_cuda_ms", dense_gather_timer
        )
        dense_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
        _breakdown_record_cuda("row_routed_mlp_dense_gate_up_cuda_ms", dense_timer)

        base_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_gate_up = _semi_structured_linear(
            input_tensor, _sparse_base_weight(gate_up_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_gate_up_cuda_ms", base_timer
        )

        dense_act_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        dense_act = act_fn(dense_gate_up) if act_fn is not None else _silu_and_mul_local(
            dense_gate_up,
            int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
        )
        _breakdown_record_cuda("row_routed_mlp_dense_act_cuda_ms", dense_act_timer)

        base_act_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_act = act_fn(base_gate_up) if act_fn is not None else _silu_and_mul_local(
            base_gate_up,
            int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_act_cuda_ms", base_act_timer
        )

        dense_down_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        if down_uses_exact_dense_rows:
            dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
        else:
            dense_down = _semi_structured_linear(
                dense_act, _sparse_base_weight(down_module), None
            )
        _breakdown_record_cuda("row_routed_mlp_dense_down_cuda_ms", dense_down_timer)

        base_down_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_down = _semi_structured_linear(
            base_act, _sparse_base_weight(down_module), None
        )
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_down_sparse_cuda_ms", base_down_timer
        )

        assemble_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_down.index_copy_(0, dense_rows, dense_down)
        _breakdown_record_cuda(
            "row_routed_mlp_reuse_base_index_copy_cuda_ms", assemble_timer
        )
        return base_down.contiguous()

    fixed_block_output = _row_routed_mlp_fixed_block_output(
        gate_up_module,
        down_module,
        input_tensor,
        dense_weight,
        down_dense_weight,
        dense_rows,
        base_rows,
        down_uses_exact_dense_rows=down_uses_exact_dense_rows,
        act_fn=act_fn,
    )
    if fixed_block_output is not None:
        return fixed_block_output

    if base_rows is None:
        route_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
        route_dense = torch.zeros(rows, dtype=torch.bool, device=input_tensor.device)
        route_dense.index_fill_(0, dense_rows, True)
        base_rows = (~route_dense).nonzero(as_tuple=False).squeeze(1)
        _breakdown_record_cuda("row_routed_mlp_route_build_cuda_ms", route_timer)
    min_dense_rows = row_routed_mlp_min_dense_rows_for_leaf(module_leaf)
    if dense_count < min_dense_rows and int(base_rows.numel()) > 0:
        fill_count = min(min_dense_rows - dense_count, int(base_rows.numel()))
        if fill_count > 0:
            extra_dense_rows = base_rows[:fill_count]
            dense_rows = torch.cat([dense_rows, extra_dense_rows], dim=0)
            base_rows = base_rows[fill_count:]
            dense_count = int(dense_rows.numel())
            if breakdown_linear_enabled():
                _breakdown_count("row_routed_mlp_fill_min_dense_calls", 1)
                _breakdown_count("row_routed_mlp_fill_min_dense_rows", fill_count)
                _breakdown_count("row_routed_mlp_fill_min_dense_target", min_dense_rows)
    if dense_count < min_dense_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_mlp_skipped_small_dense_rows", 1)
            _breakdown_count("row_routed_mlp_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_mlp_skipped_total_rows", rows)
            _breakdown_count("row_routed_mlp_min_dense_rows", min_dense_rows)
        return None
    if dense_count >= rows:
        dense_output = _row_routed_mlp_full_dense_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
            reason="filled_all_rows",
        )
        if dense_output is not None:
            return dense_output
    max_base_rows = row_routed_mlp_max_base_rows_for_leaf(module_leaf)
    if max_base_rows > 0 and int(base_rows.numel()) > max_base_rows:
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_mlp_skipped_large_base_rows", 1)
            _breakdown_count("row_routed_mlp_skipped_dense_rows", dense_count)
            _breakdown_count("row_routed_mlp_skipped_base_rows", int(base_rows.numel()))
            _breakdown_count("row_routed_mlp_skipped_total_rows", rows)
            _breakdown_count("row_routed_mlp_max_base_rows", max_base_rows)
        return None
    base_count = int(base_rows.numel())
    dense_fraction = dense_count / max(rows, 1)
    max_dense_fraction = route_max_dense_fraction()
    if 0.0 <= max_dense_fraction <= 1.0 and dense_fraction > max_dense_fraction:
        dense_output = _row_routed_mlp_full_dense_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
            reason="max_dense_fraction",
        )
        if dense_output is not None:
            if breakdown_linear_enabled():
                _breakdown_count("row_routed_mlp_fallback_dense_fraction", dense_fraction)
            return dense_output
    min_base_rows = route_min_base_rows_for_leaf(module_leaf)
    if min_base_rows > 0 and base_count < min_base_rows:
        dense_output = _row_routed_mlp_full_dense_output(
            gate_up_module,
            down_module,
            input_tensor,
            dense_weight,
            down_dense_weight,
            down_uses_exact_dense_rows=down_uses_exact_dense_rows,
            act_fn=act_fn,
            reason="small_base",
        )
        if dense_output is not None:
            if breakdown_linear_enabled():
                _breakdown_count("row_routed_mlp_fallback_small_base_rows", base_count)
                _breakdown_count("row_routed_mlp_fallback_min_base_rows", min_base_rows)
            return dense_output
    if breakdown_linear_enabled():
        _breakdown_count("row_routed_mlp_calls", 1)
        _breakdown_count("row_routed_mlp_rows", rows)
        _breakdown_count("row_routed_mlp_dense_rows", dense_count)
        _breakdown_count("row_routed_mlp_base_rows", base_count)

    contiguous_output = _row_routed_mlp_contiguous_output(
        gate_up_module,
        down_module,
        input_tensor,
        dense_weight,
        down_dense_weight,
        dense_rows,
        base_rows,
        down_uses_exact_dense_rows=down_uses_exact_dense_rows,
        act_fn=act_fn,
    )
    if contiguous_output is not None:
        return contiguous_output

    overlap_output = _row_routed_mlp_overlap_output(
        gate_up_module,
        down_module,
        input_tensor,
        dense_weight,
        down_dense_weight,
        dense_rows,
        base_rows,
        down_uses_exact_dense_rows=down_uses_exact_dense_rows,
        act_fn=act_fn,
    )
    if overlap_output is not None:
        return overlap_output

    dense_gather_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    dense_input = input_tensor.index_select(0, dense_rows)
    _breakdown_record_cuda("row_routed_mlp_dense_gather_cuda_ms", dense_gather_timer)
    dense_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    dense_gate_up = _sr24_dense_linear(dense_input, dense_weight, bias=None)
    _breakdown_record_cuda("row_routed_mlp_dense_gate_up_cuda_ms", dense_timer)

    base_gather_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    base_input = input_tensor.index_select(0, base_rows)
    _breakdown_record_cuda("row_routed_mlp_base_gather_cuda_ms", base_gather_timer)
    base_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    base_gate_up = _semi_structured_linear(
        base_input, _sparse_base_weight(gate_up_module), None
    )
    _breakdown_record_cuda("row_routed_mlp_base_gate_up_cuda_ms", base_timer)

    if down_uses_exact_dense_rows:
        dense_act_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        dense_act = act_fn(dense_gate_up) if act_fn is not None else _silu_and_mul_local(
            dense_gate_up,
            int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
        )
        _breakdown_record_cuda("row_routed_mlp_dense_act_cuda_ms", dense_act_timer)

        dense_down_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        dense_down = _sr24_dense_linear(dense_act, down_dense_weight, None)
        _breakdown_record_cuda("row_routed_mlp_dense_down_cuda_ms", dense_down_timer)

        base_act_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_act = act_fn(base_gate_up) if act_fn is not None else _silu_and_mul_local(
            base_gate_up,
            int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
        )
        _breakdown_record_cuda("row_routed_mlp_base_act_cuda_ms", base_act_timer)

        base_down_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        base_down = _semi_structured_linear(
            base_act, _sparse_base_weight(down_module), None
        )
        _breakdown_record_cuda("row_routed_mlp_base_down_sparse_cuda_ms", base_down_timer)

        out_features = int(base_down.shape[1])
        assemble_timer = (
            _breakdown_cuda_start() if breakdown_linear_enabled() else None
        )
        if triton_route_assembly():
            output = _triton_routed_assemble(
                dense_down,
                dense_rows,
                base_down,
                base_rows,
                total_rows=rows,
                out_features=out_features,
            )
            _breakdown_record_cuda(
                "row_routed_mlp_triton_assemble_cuda_ms",
                assemble_timer,
            )
        else:
            output = torch.empty(
                (rows, out_features),
                dtype=base_down.dtype,
                device=base_down.device,
            )
            output.index_copy_(0, dense_rows, dense_down)
            output.index_copy_(0, base_rows, base_down)
            _breakdown_record_cuda(
                "row_routed_mlp_index_copy_cuda_ms",
                assemble_timer,
            )
        if breakdown_linear_enabled():
            _breakdown_count("row_routed_mlp_exact_dense_rows_calls", 1)
        return output.contiguous()

    cat_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    routed_gate_up = torch.cat([dense_gate_up, base_gate_up], dim=0)
    _breakdown_record_cuda("row_routed_mlp_gate_up_cat_cuda_ms", cat_timer)
    act_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    routed_act = act_fn(routed_gate_up) if act_fn is not None else _silu_and_mul_local(
        routed_gate_up,
        int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
    )
    _breakdown_record_cuda("row_routed_mlp_act_cuda_ms", act_timer)
    down_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    routed_down = _semi_structured_linear(
        routed_act, _sparse_base_weight(down_module), None
    )
    _breakdown_record_cuda("row_routed_mlp_down_sparse_cuda_ms", down_timer)

    out_features = int(routed_down.shape[1])
    assemble_timer = _breakdown_cuda_start() if breakdown_linear_enabled() else None
    if triton_route_assembly():
        output = _triton_routed_assemble(
            routed_down[:dense_count],
            dense_rows,
            routed_down[dense_count:],
            base_rows,
            total_rows=rows,
            out_features=out_features,
        )
        _breakdown_record_cuda(
            "row_routed_mlp_triton_assemble_cuda_ms",
            assemble_timer,
        )
    else:
        output = torch.empty(
            (rows, out_features),
            dtype=routed_down.dtype,
            device=routed_down.device,
        )
        output.index_copy_(0, dense_rows, routed_down[:dense_count])
        output.index_copy_(0, base_rows, routed_down[dense_count:])
        _breakdown_record_cuda("row_routed_mlp_index_copy_cuda_ms", assemble_timer)
    return output.contiguous()


def _silu_and_mul_local(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    return F.silu(gate) * up


@torch.inference_mode()
def _row_routed_mlp_full_dense_output(
    gate_up_module: Any,
    down_module: Any,
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    down_dense_weight: torch.Tensor | None,
    *,
    down_uses_exact_dense_rows: bool,
    act_fn: Any | None,
    reason: str,
) -> torch.Tensor | None:
    if not down_uses_exact_dense_rows or down_dense_weight is None:
        return None
    linear_breakdown = breakdown_linear_enabled()
    rows = int(input_tensor.shape[0])
    gate_timer = _breakdown_cuda_start() if linear_breakdown else None
    gate_up = _sr24_dense_linear(input_tensor, dense_weight, bias=None)
    _breakdown_record_cuda(
        f"row_routed_mlp_{reason}_dense_gate_up_cuda_ms",
        gate_timer,
    )
    act_timer = _breakdown_cuda_start() if linear_breakdown else None
    routed_act = act_fn(gate_up) if act_fn is not None else _silu_and_mul_local(
        gate_up,
        int(getattr(down_module, "_speclink_sr24_weight_shape")[1]),
    )
    _breakdown_record_cuda(
        f"row_routed_mlp_{reason}_dense_act_cuda_ms",
        act_timer,
    )
    down_timer = _breakdown_cuda_start() if linear_breakdown else None
    output = _sr24_dense_linear(routed_act, down_dense_weight, None).contiguous()
    _breakdown_record_cuda(
        f"row_routed_mlp_{reason}_dense_down_cuda_ms",
        down_timer,
    )
    if linear_breakdown:
        _breakdown_count("row_routed_mlp_full_dense_fallback_calls", 1)
        _breakdown_count(f"row_routed_mlp_full_dense_fallback_{reason}", 1)
        _breakdown_count("row_routed_mlp_full_dense_fallback_rows", rows)
    return output


@torch.inference_mode()
def _dense_rows_linear_output(
    module: Any,
    input_tensor: torch.Tensor,
    base_output: torch.Tensor,
    residual_rows: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
    if dense_weight is None or int(residual_rows.numel()) <= 0:
        return base_output
    rows = int(input_tensor.shape[0])
    linear_breakdown = breakdown_linear_enabled()
    if linear_breakdown:
        _breakdown_count_module(module, "residual_dense_rows_calls", 1)
        _breakdown_count_module(module, "residual_dense_rows_total_rows", rows)
        _breakdown_count_module(
            module, "residual_dense_rows", int(residual_rows.numel())
        )
    if int(residual_rows.numel()) == rows:
        dense_timer = _breakdown_cuda_start() if linear_breakdown else None
        output = _sr24_dense_linear(input_tensor, dense_weight, bias)
        _breakdown_record_cuda_module(
            module,
            "residual_dense_rows_full_gemm_cuda_ms",
            dense_timer,
        )
        if linear_breakdown:
            _breakdown_count_module(module, "residual_dense_rows_full_calls", 1)
        return output
    gather_timer = _breakdown_cuda_start() if linear_breakdown else None
    dense_input = input_tensor.index_select(0, residual_rows)
    _breakdown_record_cuda("residual_dense_rows_gather_cuda_ms", gather_timer)
    dense_timer = _breakdown_cuda_start() if linear_breakdown else None
    dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
    _breakdown_record_cuda_module(
        module, "residual_dense_rows_gemm_cuda_ms", dense_timer
    )
    scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
    base_output.index_copy_(0, residual_rows, dense_output)
    _breakdown_record_cuda("residual_dense_rows_index_copy_cuda_ms", scatter_timer)
    return base_output


def _compressed_residual_weight(
    module: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out_features, in_features = getattr(module, "_speclink_sr24_weight_shape")
    cache_enabled = cache_compressed_residual_weight()
    if cache_enabled:
        cached = getattr(module, "_speclink_sr24_cached_residual_weight", None)
        if (
            isinstance(cached, torch.Tensor)
            and cached.device == device
            and cached.dtype == dtype
            and tuple(cached.shape) == (out_features, in_features)
        ):
            _breakdown_count("compressed_residual_cached_weight_hits", 1)
            return cached
    usable_in = int(getattr(module, "_speclink_sr24_usable_in", 0))
    groups = usable_in // 4
    mask_bytes = getattr(module, "_speclink_sr24_base_mask_bytes")
    group_bytes = _expand_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=groups,
        device=device,
    )
    keep = _unpacked_group_bytes_to_keep(group_bytes, device=device)
    residual_values = getattr(module, "_speclink_sr24_residual_values").to(
        device=device, dtype=dtype, non_blocking=True
    )

    residual_weight = torch.zeros(
        (out_features, in_features),
        device=device,
        dtype=dtype,
    )
    residual_view = residual_weight[:, :usable_in].view(out_features, groups, 4)
    residual_view.masked_scatter_(torch.logical_not(keep), residual_values)
    if cache_enabled:
        module._speclink_sr24_cached_residual_weight = residual_weight
        _breakdown_count("compressed_residual_cached_weight_misses", 1)
    return residual_weight


def _compressed_residual_cached_tensor(
    module: Any,
    *,
    source_attr: str,
    cache_attr: str,
    dtype: torch.dtype,
    device: torch.device,
    counter_prefix: str,
) -> torch.Tensor | None:
    source = getattr(module, source_attr, None)
    if source is None:
        return None
    cached = getattr(module, cache_attr, None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.device == device
        and cached.dtype == dtype
        and tuple(cached.shape) == tuple(source.shape)
        and cached.is_contiguous()
    ):
        _breakdown_count(f"{counter_prefix}_cached_hits", 1)
        return cached
    tensor = source.to(device=device, dtype=dtype, non_blocking=True)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    setattr(module, cache_attr, tensor)
    _breakdown_count(f"{counter_prefix}_cached_misses", 1)
    return tensor


def _compressed_residual_weight_slice(
    module: Any,
    *,
    row_start: int,
    row_end: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out_features, in_features = getattr(module, "_speclink_sr24_weight_shape")
    usable_in = int(getattr(module, "_speclink_sr24_usable_in", 0))
    groups = usable_in // 4
    row_start = max(0, int(row_start))
    row_end = min(int(out_features), int(row_end))
    if row_end <= row_start:
        raise RuntimeError("empty SR24 residual weight slice")
    if cache_compressed_residual_weight():
        residual_weight = _compressed_residual_weight(
            module,
            dtype=dtype,
            device=device,
        )
        return residual_weight[row_start:row_end]
    mask_bytes = getattr(module, "_speclink_sr24_base_mask_bytes")
    mask_slice = mask_bytes[row_start:row_end]
    group_bytes = _expand_mask_bytes(
        mask_slice,
        out_features=row_end - row_start,
        groups=groups,
        device=device,
    )
    keep = _unpacked_group_bytes_to_keep(group_bytes, device=device)
    residual_values = getattr(module, "_speclink_sr24_residual_values").to(
        device=device, dtype=dtype, non_blocking=True
    )
    residual_slice = residual_values.view(out_features, groups, 2)[
        row_start:row_end
    ].contiguous()

    residual_weight = torch.zeros(
        (row_end - row_start, in_features),
        device=device,
        dtype=dtype,
    )
    residual_view = residual_weight[:, :usable_in].view(
        row_end - row_start, groups, 4
    )
    residual_view.masked_scatter_(torch.logical_not(keep), residual_slice.view(-1))
    return residual_weight


@triton.jit
def _compressed_residual_matmul_kernel(
    input_ptr,
    residual_values_ptr,
    mask_bytes_ptr,
    output_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    groups: tl.constexpr,
    input_stride_m: tl.constexpr,
    input_stride_k: tl.constexpr,
    mask_stride_n: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for group_start in range(0, groups, block_g):
        offs_g = group_start + tl.arange(0, block_g)
        group_mask = offs_g < groups
        packed = tl.load(
            mask_bytes_ptr
            + offs_n[:, None] * mask_stride_n
            + (offs_g[None, :] // 2),
            mask=(offs_n[:, None] < out_features) & group_mask[None, :],
            other=0,
        )
        group_bits = tl.where(
            (offs_g[None, :] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        )
        miss0 = (group_bits & 0x1) == 0
        miss1 = (group_bits & 0x2) == 0
        miss2 = (group_bits & 0x4) == 0
        miss3 = (group_bits & 0x8) == 0
        first_missing = tl.minimum(
            tl.minimum(tl.where(miss0, 0, 4), tl.where(miss1, 1, 4)),
            tl.minimum(tl.where(miss2, 2, 4), tl.where(miss3, 3, 4)),
        )
        second_missing = tl.maximum(
            tl.maximum(tl.where(miss0, 0, -1), tl.where(miss1, 1, -1)),
            tl.maximum(tl.where(miss2, 2, -1), tl.where(miss3, 3, -1)),
        )
        residual_base = (offs_n[:, None] * groups + offs_g[None, :]) * 2
        w0 = tl.load(
            residual_values_ptr + residual_base,
            mask=(offs_n[:, None] < out_features) & group_mask[None, :],
            other=0.0,
        )
        w1 = tl.load(
            residual_values_ptr + residual_base + 1,
            mask=(offs_n[:, None] < out_features) & group_mask[None, :],
            other=0.0,
        )
        for pos in tl.static_range(0, 4):
            x = tl.load(
                input_ptr
                + offs_m[:, None] * input_stride_m
                + (offs_g[None, :] * 4 + pos) * input_stride_k,
                mask=(offs_m[:, None] < rows) & group_mask[None, :],
                other=0.0,
            )
            w = tl.where(
                first_missing == pos,
                w0,
                tl.where(second_missing == pos, w1, 0.0),
            )
            acc += tl.dot(x, tl.trans(w))

    tl.store(
        output_ptr
        + offs_m[:, None] * output_stride_m
        + offs_n[None, :] * output_stride_n,
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


def _compressed_residual_triton_linear(
    module: Any,
    input_tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if not compressed_residual_triton():
        return None
    if device.type != "cuda" or input_tensor.ndim != 2:
        return None
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    out_features, in_features = getattr(module, "_speclink_sr24_weight_shape")
    usable_in = int(getattr(module, "_speclink_sr24_usable_in", 0))
    if usable_in != int(in_features) or usable_in <= 0:
        return None
    rows = int(input_tensor.shape[0])
    if rows <= 0:
        return input_tensor.new_empty((0, int(out_features)))
    groups = usable_in // 4
    residual_values = _compressed_residual_cached_tensor(
        module,
        source_attr="_speclink_sr24_residual_values",
        cache_attr="_speclink_sr24_triton_residual_values",
        dtype=dtype,
        device=device,
        counter_prefix="compressed_residual_triton_values",
    )
    if residual_values is None:
        return None
    mask_bytes = _compressed_residual_cached_tensor(
        module,
        source_attr="_speclink_sr24_base_mask_bytes",
        cache_attr="_speclink_sr24_triton_mask_bytes",
        dtype=torch.uint8,
        device=device,
        counter_prefix="compressed_residual_triton_mask",
    )
    if mask_bytes is None:
        return None
    expected_mask_cols = (groups + 1) // 2
    if tuple(mask_bytes.shape) != (int(out_features), expected_mask_cols):
        return None
    input_contig = input_tensor.to(dtype=dtype)
    if not input_contig.is_contiguous():
        input_contig = input_contig.contiguous()
    output = torch.empty((rows, int(out_features)), device=device, dtype=dtype)
    block_m = compressed_residual_block_m()
    block_n = compressed_residual_block_n()
    block_g = compressed_residual_block_g()
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(int(out_features), block_n),
    )
    _compressed_residual_matmul_kernel[grid](
        input_contig,
        residual_values,
        mask_bytes,
        output,
        rows,
        int(out_features),
        int(groups),
        int(input_contig.stride(0)),
        int(input_contig.stride(1)),
        int(mask_bytes.stride(0)),
        int(output.stride(0)),
        int(output.stride(1)),
        block_m,
        block_n,
        block_g,
        num_warps=4,
        num_stages=3,
    )
    return output


@torch.inference_mode()
def _compressed_residual_linear_output(
    module: Any,
    input_tensor: torch.Tensor,
    base_output: torch.Tensor,
    residual_rows: torch.Tensor,
) -> torch.Tensor:
    usable_in = int(getattr(module, "_speclink_sr24_usable_in", 0))
    if usable_in <= 0 or int(residual_rows.numel()) <= 0:
        return base_output

    dtype = getattr(module, "_speclink_sr24_weight_dtype", input_tensor.dtype)
    device = input_tensor.device

    full_rows = int(residual_rows.numel()) == int(input_tensor.shape[0])
    linear_breakdown = breakdown_linear_enabled()
    gather_timer = None if full_rows or not linear_breakdown else _breakdown_cuda_start()
    residual_input = input_tensor if full_rows else input_tensor.index_select(
        0, residual_rows
    )
    _breakdown_record_cuda("compressed_residual_gather_cuda_ms", gather_timer)
    if linear_breakdown:
        _breakdown_count_module(module, "compressed_residual_calls", 1)
        _breakdown_count_module(
            module, "compressed_residual_rows", int(residual_input.shape[0])
        )
    out_features, _ = getattr(module, "_speclink_sr24_weight_shape")
    chunk = residual_out_chunk()
    triton_timer = _breakdown_cuda_start() if linear_breakdown else None
    try:
        residual_output = _compressed_residual_triton_linear(
            module,
            residual_input,
            dtype=dtype,
            device=device,
        )
    except Exception:
        residual_output = None
    _breakdown_record_cuda("compressed_residual_triton_cuda_ms", triton_timer)
    if residual_output is not None:
        if linear_breakdown:
            _breakdown_count_module(module, "compressed_residual_triton_calls", 1)
            _breakdown_count_module(
                module,
                "compressed_residual_triton_rows",
                int(residual_input.shape[0]),
            )
    elif chunk <= 0 or chunk >= int(out_features):
        materialize_timer = _breakdown_cuda_start() if linear_breakdown else None
        residual_weight = _compressed_residual_weight(
            module, dtype=dtype, device=device
        )
        _breakdown_record_cuda_module(
            module,
            "compressed_residual_materialize_cuda_ms",
            materialize_timer,
        )
        gemm_timer = _breakdown_cuda_start() if linear_breakdown else None
        residual_output = F.linear(residual_input, residual_weight, bias=None)
        _breakdown_record_cuda_module(
            module, "compressed_residual_gemm_cuda_ms", gemm_timer
        )
    else:
        residual_output = torch.empty(
            (int(residual_input.shape[0]), int(out_features)),
            device=device,
            dtype=dtype,
        )
        for start in range(0, int(out_features), chunk):
            end = min(int(out_features), start + chunk)
            materialize_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_weight = _compressed_residual_weight_slice(
                module,
                row_start=start,
                row_end=end,
                dtype=dtype,
                device=device,
            )
            _breakdown_record_cuda_module(
                module,
                "compressed_residual_materialize_cuda_ms",
                materialize_timer,
            )
            gemm_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_slice = F.linear(
                residual_input, residual_weight, bias=None
            )
            _breakdown_record_cuda_module(
                module, "compressed_residual_gemm_cuda_ms", gemm_timer
            )
            copy_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_output[:, start:end] = residual_slice
            _breakdown_record_cuda(
                "compressed_residual_slice_copy_cuda_ms",
                copy_timer,
            )
    if full_rows:
        add_timer = _breakdown_cuda_start() if linear_breakdown else None
        output = base_output + residual_output
        _breakdown_record_cuda("compressed_residual_add_cuda_ms", add_timer)
        return output
    scatter_timer = _breakdown_cuda_start() if linear_breakdown else None
    output = base_output.clone()
    output.index_add_(0, residual_rows, residual_output)
    _breakdown_record_cuda("compressed_residual_scatter_cuda_ms", scatter_timer)
    return output


@torch.inference_mode()
def _compressed_residual_linear_output_masked(
    module: Any,
    input_tensor: torch.Tensor,
    base_output: torch.Tensor,
    residual_mask: torch.Tensor,
) -> torch.Tensor:
    usable_in = int(getattr(module, "_speclink_sr24_usable_in", 0))
    if usable_in <= 0:
        return base_output

    dtype = getattr(module, "_speclink_sr24_weight_dtype", input_tensor.dtype)
    device = input_tensor.device
    out_features, _ = getattr(module, "_speclink_sr24_weight_shape")
    chunk = residual_out_chunk()
    linear_breakdown = breakdown_linear_enabled()
    if linear_breakdown:
        _breakdown_count_module(module, "compressed_residual_masked_calls", 1)
        _breakdown_count_module(
            module, "compressed_residual_masked_rows", int(input_tensor.shape[0])
        )
    triton_timer = _breakdown_cuda_start() if linear_breakdown else None
    try:
        residual_output = _compressed_residual_triton_linear(
            module,
            input_tensor,
            dtype=dtype,
            device=device,
        )
    except Exception:
        residual_output = None
    _breakdown_record_cuda("compressed_residual_triton_cuda_ms", triton_timer)
    if residual_output is not None:
        if linear_breakdown:
            _breakdown_count_module(module, "compressed_residual_triton_calls", 1)
            _breakdown_count_module(
                module,
                "compressed_residual_triton_rows",
                int(input_tensor.shape[0]),
            )
    elif chunk <= 0 or chunk >= int(out_features):
        materialize_timer = _breakdown_cuda_start() if linear_breakdown else None
        residual_weight = _compressed_residual_weight(
            module,
            dtype=dtype,
            device=device,
        )
        _breakdown_record_cuda_module(
            module,
            "compressed_residual_materialize_cuda_ms",
            materialize_timer,
        )
        gemm_timer = _breakdown_cuda_start() if linear_breakdown else None
        residual_output = F.linear(input_tensor, residual_weight, bias=None)
        _breakdown_record_cuda_module(
            module, "compressed_residual_gemm_cuda_ms", gemm_timer
        )
    else:
        residual_output = torch.empty(
            (int(input_tensor.shape[0]), int(out_features)),
            device=device,
            dtype=dtype,
        )
        for start in range(0, int(out_features), chunk):
            end = min(int(out_features), start + chunk)
            materialize_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_weight = _compressed_residual_weight_slice(
                module,
                row_start=start,
                row_end=end,
                dtype=dtype,
                device=device,
            )
            _breakdown_record_cuda_module(
                module,
                "compressed_residual_materialize_cuda_ms",
                materialize_timer,
            )
            gemm_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_slice = F.linear(
                input_tensor, residual_weight, bias=None
            )
            _breakdown_record_cuda_module(
                module, "compressed_residual_gemm_cuda_ms", gemm_timer
            )
            copy_timer = _breakdown_cuda_start() if linear_breakdown else None
            residual_output[:, start:end] = residual_slice
            _breakdown_record_cuda(
                "compressed_residual_slice_copy_cuda_ms",
                copy_timer,
            )
    add_timer = _breakdown_cuda_start() if linear_breakdown else None
    residual_output = residual_output * residual_mask.to(
        dtype=residual_output.dtype
    ).unsqueeze(1)
    output = base_output + residual_output
    _breakdown_record_cuda("compressed_residual_mask_add_cuda_ms", add_timer)
    return output


@torch.inference_mode()
def residual_linear_output(
    module: Any,
    input_tensor: torch.Tensor,
    base_output: torch.Tensor,
) -> torch.Tensor:
    if not enabled() or not getattr(module, "_speclink_sr24_enabled", False):
        return base_output
    if _is_exact_dense_output(base_output):
        if breakdown_linear_enabled():
            _breakdown_count_module(module, "exact_dense_postprocess_skips", 1)
            _breakdown_count_module(
                module,
                "exact_dense_postprocess_skip_rows",
                int(input_tensor.shape[0]),
            )
        return base_output
    if getattr(module, "_speclink_sr24_dense_fastpath", False):
        return base_output
    if input_tensor.ndim != 2 or base_output.ndim != 2:
        raise RuntimeError("SpecLink SR24 Llama path expects 2D tensors")

    rows = int(input_tensor.shape[0])
    if rows <= 0:
        return base_output
    if (
        mode() == "base_only"
        or getattr(module, "_speclink_sr24_no_residual", False)
        or _runtime_base_only_for_module(module)
    ):
        return base_output
    residual_state = _current_residual_state()
    if residual_state == "no_residual":
        return base_output
    if _selective_no_context_uses_sparse_base(module, rows):
        if breakdown_linear_enabled():
            _breakdown_count_module(
                module, "selective_noverify_sparse_base_rows", rows
            )
        return base_output
    if reduce_cpu_sync():
        row_uses_residual = _residual_mask_for_input(input_tensor)
        if row_uses_residual is None:
            return base_output
        full_residual = residual_state == "all_residual" or (
            residual_state is None
            and _current_residual_mask() is None
            and mode() == "all_corrected"
        )
        if full_residual:
            dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
            if dense_weight is not None:
                bias = getattr(module, "bias", None)
                if getattr(module, "skip_bias_add", False):
                    bias = None
                residual_rows = _device_arange(
                    rows,
                    dtype=torch.long,
                    device=input_tensor.device,
                )
                return _dense_rows_linear_output(
                    module,
                    input_tensor,
                    base_output,
                    residual_rows,
                    bias,
                ).contiguous()
            residual_weight = getattr(module, "_speclink_sr24_residual_sparse", None)
            if residual_weight is not None:
                residual_timer = (
                    _breakdown_cuda_start() if breakdown_linear_enabled() else None
                )
                residual_output = _semi_structured_linear(
                    input_tensor, residual_weight, None
                )
                _breakdown_record_cuda_module(
                    module, "residual_sparse_full_gemm_cuda_ms", residual_timer
                )
                return (
                    base_output
                    + residual_output
                ).contiguous()
            residual_rows = _device_arange(
                rows,
                dtype=torch.long,
                device=input_tensor.device,
            )
            return _compressed_residual_linear_output(
                module,
                input_tensor,
                base_output,
                residual_rows,
            ).contiguous()
        residual_priority = _residual_priority_for_input(input_tensor)
        residual_bucket = _residual_bucket_for_mask(
            input_tensor,
            row_uses_residual,
            residual_priority,
        )
        dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
        if dense_weight is not None:
            bias = getattr(module, "bias", None)
            if getattr(module, "skip_bias_add", False):
                bias = None
            if (
                route_reuse_base_output()
                and getattr(module, "_speclink_sr24_residual_backend", "") == "dense_rows"
            ):
                residual_rows = _current_residual_rows()
                if (
                    residual_rows is not None
                    and row_uses_residual.numel() == int(input_tensor.shape[0])
                ):
                    _breakdown_count("route_reuse_base_output_cached_rows_hits", 1)
                    residual_rows = residual_rows.to(
                        device=input_tensor.device,
                        dtype=torch.long,
                    )
                else:
                    residual_rows = row_uses_residual.to(
                        device=input_tensor.device,
                        dtype=torch.bool,
                    ).nonzero(as_tuple=False).squeeze(1)
                return _dense_rows_linear_output(
                    module,
                    input_tensor,
                    base_output,
                    residual_rows,
                    bias,
                ).contiguous()
            if residual_bucket is not None:
                bucket_rows, bucket_values = residual_bucket
                if breakdown_linear_enabled():
                    _breakdown_count_module(module, "dense_rows_bucket_linear_calls", 1)
                    _breakdown_count_module(
                        module, "dense_rows_bucket_rows", int(bucket_rows.numel())
                    )
                compute_active_only = (
                    bucket_dense_compute_active_only()
                    and bucket_dense_copy_active_only()
                )
                if (
                    compute_active_only
                    and bucket_dense_active_mask_fused()
                    and triton_bucket_dense_gemm()
                    and base_output.is_cuda
                ):
                    fused_timer = (
                        _breakdown_cuda_start()
                        if breakdown_linear_enabled()
                        else None
                    )
                    try:
                        fused_ok = _triton_bucket_dense_gemm_scatter_inplace(
                            input_tensor,
                            dense_weight,
                            bucket_rows,
                            bucket_values,
                            base_output,
                            bias,
                            force_all_bucket_rows=False,
                        )
                    except Exception:
                        fused_ok = False
                    _breakdown_record_cuda_module(
                        module,
                        "bucket_active_mask_fused_dense_gemm_scatter_cuda_ms",
                        fused_timer,
                    )
                    if fused_ok:
                        if breakdown_linear_enabled():
                            _breakdown_count_module(
                                module,
                                "bucket_active_mask_fused_dense_gemm_scatter_calls",
                                1,
                            )
                            _breakdown_count_module(
                                module,
                                "bucket_active_mask_fused_dense_gemm_scatter_rows",
                                int(bucket_rows.numel()),
                            )
                        return base_output.contiguous()
                active_bucket_rows = bucket_rows
                active_bucket_values = bucket_values
                active_bucket_indices = None
                if compute_active_only:
                    active_bucket_indices = bucket_values.to(
                        dtype=torch.bool
                    ).nonzero(as_tuple=False).squeeze(1)
                    active_bucket_rows = bucket_rows.index_select(
                        0, active_bucket_indices
                    )
                    active_bucket_values = bucket_values.index_select(
                        0, active_bucket_indices
                    )
                    _breakdown_count_module(
                        module,
                        "bucket_dense_compute_active_only_calls",
                        1,
                    )
                    _breakdown_count_module(
                        module,
                        "bucket_dense_compute_active_only_rows",
                        int(active_bucket_rows.numel()),
                    )
                    if int(active_bucket_rows.numel()) <= 0:
                        return base_output.contiguous()
                if (
                    triton_bucket_dense_gemm()
                    and base_output.is_cuda
                    and not compute_active_only
                ):
                    fused_timer = (
                        _breakdown_cuda_start()
                        if breakdown_linear_enabled()
                        else None
                    )
                    try:
                        fused_ok = _triton_bucket_dense_gemm_scatter_inplace(
                            input_tensor,
                            dense_weight,
                            bucket_rows,
                            bucket_values,
                            base_output,
                            bias,
                            force_all_bucket_rows=(
                                bucket_dense_copy()
                                and not bucket_dense_copy_active_only()
                            ),
                        )
                    except Exception:
                        fused_ok = False
                    _breakdown_record_cuda_module(
                        module,
                        "bucket_triton_dense_gemm_scatter_cuda_ms",
                        fused_timer,
                    )
                    if fused_ok:
                        return base_output.contiguous()
                gather_input_timer = (
                    _breakdown_cuda_start() if breakdown_linear_enabled() else None
                )
                dense_input = input_tensor.index_select(0, active_bucket_rows)
                _breakdown_record_cuda(
                    "gather_input_index_select_cuda_ms", gather_input_timer
                )
                dense_timer = (
                    _breakdown_cuda_start() if breakdown_linear_enabled() else None
                )
                dense_output = _sr24_dense_linear(dense_input, dense_weight, bias)
                _breakdown_record_cuda_module(
                    module, "residual_dense_gemm_cuda_ms", dense_timer
                )
                if bucket_dense_copy():
                    copy_timer = (
                        _breakdown_cuda_start() if breakdown_linear_enabled() else None
                    )
                    if bucket_dense_copy_active_only():
                        scatter_ok = False
                        try:
                            scatter_ok = _triton_bucket_dense_scatter_inplace(
                                dense_output,
                                active_bucket_rows,
                                active_bucket_values,
                                base_output,
                            )
                        except Exception:
                            scatter_ok = False
                        if not scatter_ok:
                            _bucket_dense_overwrite_inplace(
                                base_output,
                                dense_output,
                                active_bucket_rows,
                                active_bucket_values,
                            )
                    else:
                        base_output.index_copy_(0, active_bucket_rows, dense_output)
                    _breakdown_record_cuda(
                        "bucket_dense_copy_index_copy_cuda_ms",
                        copy_timer,
                    )
                    if breakdown_linear_enabled():
                        _breakdown_count_module(
                            module, "bucket_dense_copy_calls", 1
                        )
                        _breakdown_count_module(
                            module,
                            "bucket_dense_copy_rows",
                            int(bucket_rows.numel()),
                        )
                    return base_output.contiguous()
                if not bucket_dense_delta_add():
                    overwrite_timer = (
                        _breakdown_cuda_start()
                        if breakdown_linear_enabled()
                        else None
                    )
                    output = _bucket_dense_overwrite_inplace(
                        base_output,
                        dense_output,
                        bucket_rows,
                        bucket_values,
                    )
                    _breakdown_record_cuda(
                        "bucket_dense_overwrite_index_copy_cuda_ms",
                        overwrite_timer,
                    )
                    if breakdown_linear_enabled():
                        _breakdown_count_module(
                            module, "bucket_dense_overwrite_calls", 1
                        )
                        _breakdown_count_module(
                            module,
                            "bucket_dense_overwrite_rows",
                            int(bucket_rows.numel()),
                        )
                    return output.contiguous()
                gather_base_timer = (
                    _breakdown_cuda_start() if breakdown_linear_enabled() else None
                )
                base_rows = base_output.index_select(0, bucket_rows)
                _breakdown_record_cuda(
                    "gather_base_index_select_cuda_ms", gather_base_timer
                )
                delta_timer = (
                    _breakdown_cuda_start() if breakdown_linear_enabled() else None
                )
                delta = (dense_output - base_rows) * bucket_values.unsqueeze(1)
                _breakdown_record_cuda("bucket_delta_compute_cuda_ms", delta_timer)
                scatter_timer = (
                    _breakdown_cuda_start() if breakdown_linear_enabled() else None
                )
                base_output.index_add_(0, bucket_rows, delta)
                _breakdown_record_cuda("scatter_index_add_cuda_ms", scatter_timer)
                return base_output.contiguous()
            dense_timer = (
                _breakdown_cuda_start() if breakdown_linear_enabled() else None
            )
            dense_output = _sr24_dense_linear(input_tensor, dense_weight, bias)
            _breakdown_record_cuda_module(
                module, "residual_dense_full_gemm_cuda_ms", dense_timer
            )
            select_timer = (
                _breakdown_cuda_start() if breakdown_linear_enabled() else None
            )
            selector = row_uses_residual.to(dtype=torch.bool).unsqueeze(1)
            output = torch.where(selector, dense_output, base_output)
            _breakdown_record_cuda("residual_dense_full_select_cuda_ms", select_timer)
            return output.contiguous()
        return _compressed_residual_linear_output_masked(
            module,
            input_tensor,
            base_output,
            row_uses_residual,
        )
    if residual_state == "all_residual":
        row_uses_residual = torch.ones(rows, dtype=torch.bool, device=input_tensor.device)
    else:
        residual_mask = _current_residual_mask()
        if residual_mask is None:
            sr_mode = mode()
            if sr_mode == "base_only":
                return base_output
            if sr_mode == "selective" and not (
                _selective_dense_nonverify_for_module_rows(module, rows)
            ):
                return base_output
            row_uses_residual = torch.ones(
                rows,
                dtype=torch.bool,
                device=input_tensor.device,
            )
        else:
            if residual_mask.numel() < rows:
                raise RuntimeError(
                    f"SpecLink SR24 mask has {residual_mask.numel()} rows, "
                    f"but linear input has {rows}"
                )
            row_uses_residual = residual_mask[:rows].to(device=input_tensor.device)

    residual_rows = row_uses_residual.nonzero(as_tuple=False).squeeze(1)
    dense_weight = getattr(module, "_speclink_sr24_dense_weight", None)
    if dense_weight is not None:
        bias = getattr(module, "bias", None)
        if getattr(module, "skip_bias_add", False):
            bias = None
        return _dense_rows_linear_output(
            module,
            input_tensor,
            base_output,
            residual_rows,
            bias,
        )
    return _compressed_residual_linear_output(
        module,
        input_tensor,
        base_output,
        residual_rows,
    )
