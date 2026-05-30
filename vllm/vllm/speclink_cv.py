# SPDX-License-Identifier: Apache-2.0
"""Small runtime helpers for the SpecLink-CV prototype.

The current experiment keeps one narrow policy: verify a fixed prefix first
(`h=K/2`, or `SPECLINK_CV_FORCE_PREFIX_LEN`), then draft/verify the suffix only
when the prefix is fully accepted.  The helpers here are intentionally
environment-variable driven so the vLLM CLI surface stays unchanged.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class SpecLinkCVRuntimeConfig:
    enable: bool = False
    async_queue: bool = False
    allow_batched_prefix: bool = False
    allow_batched_suffix: bool = False
    allow_shape_drift_chunking: bool = True
    staged_drafting: bool = False
    default_half_policy: str = "floor"
    force_prefix_len: int = 0
    max_verify_tokens_per_step: int = 0
    max_verify_seqs_per_step: int = 0
    max_queue_wait_ms: float = 2.0
    prefix_wavefront: bool = False
    prefix_wave_wait_for_min: bool = False
    prefix_wave_exclusive: bool = False
    prefix_wave_min_seqs: int = 0
    prefix_wave_max_wait_ms: float = 2.0
    prefix_full_cudagraph: bool = False
    log_jsonl: str = ""
    profile_jsonl: str = ""
    log_max_events: int = 0
    profile_max_events: int = 0
    profile_copy_shapes: bool = False

    # The scheduler/worker still carry a few correctness diagnostic branches.
    # They default off and are not part of the focused performance path.
    global_batch_barrier: bool = False
    suffix_replay_one_shot_shape: bool = False
    confirm_prefix_reject_one_shot: bool = False
    confirm_prefix_accept_one_shot: bool = False
    confirmation_full_active_set: bool = False
    lockstep_iteration_barrier: bool = False
    prefix_low_margin_fallback_threshold: float = 0.0
    batch_wide_low_margin_fallback: bool = False
    batch_wide_prefix_reject_fallback: bool = False
    recompute_committed_prefix: bool = False
    allow_batched_dense_realign: bool = False
    prefix_no_kv_write: bool = False
    prefix_probe_block_rollback: bool = False
    force_decode_isolation: bool = False
    dense_realign_steps: int = 0
    prefix_reject_dense_realign_steps: int = 0
    debug_dump: bool = False
    kv_debug_tail_tokens: int = 0
    kv_debug_max_layers: int = 0
    kv_debug_row_index: int = -1
    kv_debug_min_output_tokens: int = -1
    kv_debug_max_output_tokens: int = -1

    @classmethod
    def from_env(cls) -> "SpecLinkCVRuntimeConfig":
        return cls(
            enable=_env_bool("SPECLINK_CV_ENABLE"),
            async_queue=_env_bool("SPECLINK_CV_ASYNC_QUEUE"),
            allow_batched_prefix=_env_bool("SPECLINK_CV_ALLOW_BATCHED_PREFIX"),
            allow_batched_suffix=_env_bool(
                "SPECLINK_CV_ALLOW_BATCHED_SUFFIX",
                _env_bool("SPECLINK_CV_ALLOW_BATCHED_PREFIX"),
            ),
            allow_shape_drift_chunking=_env_bool(
                "SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING", True
            ),
            staged_drafting=_env_bool("SPECLINK_CV_STAGED_DRAFTING"),
            default_half_policy=os.environ.get(
                "SPECLINK_CV_DEFAULT_HALF_POLICY", "floor"
            ).strip(),
            force_prefix_len=_env_int("SPECLINK_CV_FORCE_PREFIX_LEN", 0),
            max_verify_tokens_per_step=_env_int(
                "SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP", 0
            ),
            max_verify_seqs_per_step=_env_int(
                "SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP", 0
            ),
            max_queue_wait_ms=_env_float("SPECLINK_CV_MAX_QUEUE_WAIT_MS", 2.0),
            prefix_wavefront=_env_bool("SPECLINK_CV_PREFIX_WAVEFRONT"),
            prefix_wave_wait_for_min=_env_bool(
                "SPECLINK_CV_PREFIX_WAVE_WAIT_FOR_MIN"
            ),
            prefix_wave_exclusive=_env_bool("SPECLINK_CV_PREFIX_WAVE_EXCLUSIVE"),
            prefix_wave_min_seqs=_env_int("SPECLINK_CV_PREFIX_WAVE_MIN_SEQS", 0),
            prefix_wave_max_wait_ms=_env_float(
                "SPECLINK_CV_PREFIX_WAVE_MAX_WAIT_MS", 2.0
            ),
            prefix_full_cudagraph=_env_bool(
                "SPECLINK_CV_PREFIX_FULL_CUDAGRAPH"
            ),
            log_jsonl=os.environ.get("SPECLINK_CV_LOG_JSONL", "").strip(),
            profile_jsonl=os.environ.get("SPECLINK_CV_PROFILE_JSONL", "").strip(),
            log_max_events=_env_int("SPECLINK_CV_LOG_MAX_EVENTS", 1_000),
            profile_max_events=_env_int("SPECLINK_CV_PROFILE_MAX_EVENTS", 500),
            profile_copy_shapes=_env_bool("SPECLINK_CV_PROFILE_COPY_SHAPES"),
        )

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_seq_budget(self, scheduler_seq_budget: int) -> int:
        if not self.allow_batched_prefix:
            return 1
        return max(1, self.max_verify_seqs_per_step or scheduler_seq_budget or 1)

    def effective_prefix_wave_min_seqs(self, scheduler_seq_budget: int) -> int:
        seq_budget = self.effective_seq_budget(scheduler_seq_budget)
        if self.prefix_wave_min_seqs > 0:
            return max(1, min(self.prefix_wave_min_seqs, seq_budget))
        return max(1, min(seq_budget, (3 * seq_budget + 3) // 4))

    def effective_dense_realign_steps(self, num_spec_tokens: int) -> int:
        if self.dense_realign_steps >= 0:
            return self.dense_realign_steps
        return max(1, num_spec_tokens)

    def effective_prefix_reject_dense_realign_steps(
        self, num_spec_tokens: int
    ) -> int:
        if self.prefix_reject_dense_realign_steps <= 0:
            return 0
        return self.prefix_reject_dense_realign_steps


@dataclass(frozen=True)
class AsyncPrefixCandidate:
    request_id: str
    selected_h: int
    k: int
    queue_enter_time: float = 0.0

    def age_ms(self, now: float) -> float:
        return max(0.0, (now - self.queue_enter_time) * 1000.0)


def half_chunk(k: int, policy: str = "floor") -> int:
    if k <= 1:
        return k
    if policy == "ceil":
        return (k + 1) // 2
    if policy != "floor":
        raise ValueError(f"unknown SpecLink-CV half policy: {policy}")
    return max(1, k // 2)


def choose_prefix_len(
    k: int,
    config: SpecLinkCVRuntimeConfig,
) -> tuple[int, str, dict[str, Any]]:
    if k <= 1:
        return k, "one_token", {}
    if config.force_prefix_len > 0:
        forced = min(max(1, config.force_prefix_len), k)
        return forced, "forced_prefix_len", {"forced_prefix_len": forced}
    h = half_chunk(k, config.default_half_policy)
    return h, "fixed_half", {}


def should_wait_for_global_batch_fill(
    *, waiting_reqs: int, running_reqs: int, max_running_reqs: int
) -> bool:
    return waiting_reqs > 0 and running_reqs < max(1, max_running_reqs)


def select_async_prefix_dispatch(
    candidates: list[AsyncPrefixCandidate],
    *,
    config: SpecLinkCVRuntimeConfig,
    token_budget: int,
    seq_budget: int,
    now: float,
    has_other_ready_work: bool,
    ready_same_shape_seqs: int = 0,
) -> tuple[set[str], dict[str, Any]]:
    if not candidates:
        return set(), {"reason": "empty"}
    effective_token_budget = max(
        1, config.max_verify_tokens_per_step or token_budget or 1
    )
    effective_seq_budget = config.effective_seq_budget(seq_budget)
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -int(candidate.age_ms(now) >= config.max_queue_wait_ms),
            candidate.request_id,
        ),
    )

    max_age_ms = max(candidate.age_ms(now) for candidate in ordered)
    min_wave_seqs = 1
    if config.prefix_wavefront and config.allow_batched_prefix:
        min_wave_seqs = config.effective_prefix_wave_min_seqs(seq_budget)
        same_shape_fill_seqs = max(0, int(ready_same_shape_seqs))
        wave_ready = len(ordered) + same_shape_fill_seqs >= min_wave_seqs
        wave_expired = max_age_ms >= config.prefix_wave_max_wait_ms
        if (
            config.prefix_wave_wait_for_min
            and not wave_ready
            and not wave_expired
            and has_other_ready_work
        ):
            return set(), {
                "reason": "wavefront_wait",
                "candidate_count": len(candidates),
                "selected_count": 0,
                "dispatch_count": 0,
                "selected_tokens": 0,
                "dispatch_tokens": 0,
                "token_budget": effective_token_budget,
                "seq_budget": effective_seq_budget,
                "allow_batched_prefix": config.allow_batched_prefix,
                "prefix_wave_min_seqs": min_wave_seqs,
                "ready_same_shape_seqs": same_shape_fill_seqs,
                "prefix_wave_max_wait_ms": config.prefix_wave_max_wait_ms,
                "max_candidate_age_ms": max_age_ms,
                "has_other_ready_work": has_other_ready_work,
            }

    selected: list[AsyncPrefixCandidate] = []
    selected_tokens = 0
    for candidate in ordered:
        if len(selected) + 1 > effective_seq_budget:
            continue
        if selected_tokens + candidate.selected_h > effective_token_budget:
            continue
        selected.append(candidate)
        selected_tokens += candidate.selected_h

    if not selected:
        return set(), {
            "reason": "budget_exhausted",
            "candidate_count": len(candidates),
            "token_budget": effective_token_budget,
            "seq_budget": effective_seq_budget,
        }

    reason = "simple_priority"
    if config.prefix_wavefront and config.allow_batched_prefix:
        if len(selected) >= min_wave_seqs:
            reason = "wavefront_min_seqs"
        elif max_age_ms >= config.prefix_wave_max_wait_ms:
            reason = "wavefront_timeout"
        elif has_other_ready_work:
            reason = "wavefront_mixed_fill"
        else:
            reason = "wavefront_no_other_work"

    detail = {
        "reason": reason,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "dispatch_count": len(selected),
        "selected_tokens": selected_tokens,
        "dispatch_tokens": selected_tokens,
        "token_budget": effective_token_budget,
        "seq_budget": effective_seq_budget,
        "allow_batched_prefix": config.allow_batched_prefix,
        "token_budget_utilization": selected_tokens / effective_token_budget,
        "seq_budget_utilization": len(selected) / effective_seq_budget,
        "max_queue_wait_ms": config.max_queue_wait_ms,
        "prefix_wavefront": config.prefix_wavefront,
        "prefix_wave_min_seqs": min_wave_seqs,
        "ready_same_shape_seqs": (
            max(0, int(ready_same_shape_seqs))
            if config.prefix_wavefront and config.allow_batched_prefix
            else 0
        ),
        "prefix_wave_max_wait_ms": config.prefix_wave_max_wait_ms,
        "max_candidate_age_ms": max_age_ms,
        "has_other_ready_work": has_other_ready_work,
    }
    return {candidate.request_id for candidate in selected}, detail


_JSONL_EVENT_COUNTS: dict[str, int] = {}
_JSONL_LIMIT_REPORTED: set[str] = set()


def append_event(config: SpecLinkCVRuntimeConfig, event: dict[str, Any]) -> None:
    _append_jsonl(config.log_jsonl, event, max_events=config.log_max_events)


def append_profile(config: SpecLinkCVRuntimeConfig, event: dict[str, Any]) -> None:
    _append_jsonl(config.profile_jsonl, event, max_events=config.profile_max_events)


def _append_jsonl(
    path_str: str, event: dict[str, Any], *, max_events: int = 0
) -> None:
    if not path_str:
        return
    count = _JSONL_EVENT_COUNTS.get(path_str, 0)
    if max_events > 0 and count >= max_events:
        if path_str in _JSONL_LIMIT_REPORTED:
            return
        event = {
            "event": "jsonl_limit_reached",
            "max_events": max_events,
            "path": path_str,
        }
        _JSONL_LIMIT_REPORTED.add(path_str)
    payload = {"ts": time.time(), **event}
    path = Path(path_str)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        _JSONL_EVENT_COUNTS[path_str] = count + 1
    except Exception:
        return
