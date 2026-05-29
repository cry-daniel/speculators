# SPDX-License-Identifier: Apache-2.0
"""Runtime helpers for SpecLink-CV experiments.

This module is intentionally small and environment-variable driven. The
current vLLM patch uses it to enable a minimal exact prefix/suffix verification
path without adding new public CLI arguments yet.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
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


def _parse_candidates(value: str) -> tuple[int | str, ...]:
    candidates: list[int | str] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item == "full":
            candidates.append("full")
        else:
            candidates.append(int(item))
    return tuple(candidates or (1, 2, 4, 6, 8, "full"))


@dataclass(frozen=True)
class SpecLinkCVRuntimeConfig:
    enable: bool = False
    confidence_sizing: bool = False
    async_queue: bool = False
    roofline_packing: bool = False
    allow_batched_prefix: bool = False
    allow_batched_suffix: bool = False
    global_batch_barrier: bool = False
    allow_shape_drift_chunking: bool = True
    suffix_replay_one_shot_shape: bool = False
    confirm_prefix_reject_one_shot: bool = False
    confirm_prefix_accept_one_shot: bool = False
    confirmation_full_active_set: bool = False
    lockstep_iteration_barrier: bool = False
    prefix_probe_block_rollback: bool = False
    prefix_low_margin_fallback_threshold: float = 0.0
    batch_wide_low_margin_fallback: bool = False
    batch_wide_prefix_reject_fallback: bool = False
    recompute_committed_prefix: bool = False
    allow_batched_dense_realign: bool = False
    prefix_no_kv_write: bool = False
    prefix_full_cudagraph: bool = False
    staged_drafting: bool = False
    force_decode_isolation: bool = False
    dense_realign_steps: int = 0
    prefix_reject_dense_realign_steps: int = 0
    candidate_chunks: tuple[int | str, ...] = (1, 2, 4, 6, 8, "full")
    default_half_policy: str = "floor"
    min_benefit: float = 0.0
    force_prefix_len: int = 0
    max_verify_tokens_per_step: int = 0
    max_verify_seqs_per_step: int = 0
    max_queue_wait_ms: float = 2.0
    util_threshold: float = 0.6
    calibration_path: str = ""
    log_jsonl: str = ""
    profile_jsonl: str = ""
    debug_dump: bool = False
    kv_debug_tail_tokens: int = 0
    kv_debug_max_layers: int = 0
    kv_debug_row_index: int = -1
    kv_debug_min_output_tokens: int = -1
    kv_debug_max_output_tokens: int = -1
    log_max_events: int = 0
    profile_max_events: int = 0

    @classmethod
    def from_env(cls) -> "SpecLinkCVRuntimeConfig":
        return cls(
            enable=_env_bool("SPECLINK_CV_ENABLE"),
            confidence_sizing=False,
            async_queue=_env_bool("SPECLINK_CV_ASYNC_QUEUE"),
            roofline_packing=_env_bool("SPECLINK_CV_ROOFLINE_PACKING")
            or _env_bool("SPECLINK_CV_ROOFLLINE_PACKING"),
            allow_batched_prefix=_env_bool("SPECLINK_CV_ALLOW_BATCHED_PREFIX"),
            allow_batched_suffix=_env_bool(
                "SPECLINK_CV_ALLOW_BATCHED_SUFFIX",
                _env_bool("SPECLINK_CV_ALLOW_BATCHED_PREFIX"),
            ),
            global_batch_barrier=False,
            allow_shape_drift_chunking=_env_bool(
                "SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING", True
            ),
            suffix_replay_one_shot_shape=False,
            confirm_prefix_reject_one_shot=False,
            confirm_prefix_accept_one_shot=False,
            confirmation_full_active_set=False,
            lockstep_iteration_barrier=False,
            prefix_probe_block_rollback=False,
            prefix_low_margin_fallback_threshold=0.0,
            batch_wide_low_margin_fallback=False,
            batch_wide_prefix_reject_fallback=False,
            recompute_committed_prefix=False,
            allow_batched_dense_realign=False,
            prefix_no_kv_write=False,
            prefix_full_cudagraph=False,
            staged_drafting=_env_bool("SPECLINK_CV_STAGED_DRAFTING"),
            force_decode_isolation=False,
            dense_realign_steps=0,
            prefix_reject_dense_realign_steps=0,
            candidate_chunks=_parse_candidates(
                os.environ.get(
                    "SPECLINK_CV_CANDIDATE_CHUNKS", "1,2,4,6,8,full"
                )
            ),
            default_half_policy=os.environ.get(
                "SPECLINK_CV_DEFAULT_HALF_POLICY", "floor"
            ).strip(),
            min_benefit=_env_float("SPECLINK_CV_MIN_BENEFIT", 0.0),
            force_prefix_len=_env_int("SPECLINK_CV_FORCE_PREFIX_LEN", 0),
            max_verify_tokens_per_step=_env_int(
                "SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP", 0
            ),
            max_verify_seqs_per_step=_env_int(
                "SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP", 0
            ),
            max_queue_wait_ms=_env_float("SPECLINK_CV_MAX_QUEUE_WAIT_MS", 2.0),
            util_threshold=_env_float("SPECLINK_CV_UTIL_THRESHOLD", 0.6),
            calibration_path="",
            log_jsonl=os.environ.get("SPECLINK_CV_LOG_JSONL", "").strip(),
            profile_jsonl=os.environ.get("SPECLINK_CV_PROFILE_JSONL", "").strip(),
            debug_dump=False,
            kv_debug_tail_tokens=0,
            kv_debug_max_layers=0,
            kv_debug_row_index=-1,
            kv_debug_min_output_tokens=-1,
            kv_debug_max_output_tokens=-1,
            log_max_events=_env_int("SPECLINK_CV_LOG_MAX_EVENTS", 1_000),
            profile_max_events=_env_int("SPECLINK_CV_PROFILE_MAX_EVENTS", 500),
        )

    def to_log_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_chunks"] = list(self.candidate_chunks)
        return data

    def effective_seq_budget(self, scheduler_seq_budget: int) -> int:
        """Return the live prefix-verification sequence budget.

        Multi-request prefix verification is currently an experimental mode.
        The conservative default schedules one prefix chunk per target-model
        step because live token-id smoke tests found batch-shape dependent
        mismatches when several prefix chunks were verified together.
        """
        if not self.allow_batched_prefix:
            return 1
        return max(1, self.max_verify_seqs_per_step or scheduler_seq_budget or 1)

    def effective_dense_realign_steps(self, num_spec_tokens: int) -> int:
        """Return dense TLM steps after a rejected suffix verifier chunk.

        The default is 0 because the performance path relies on discarding
        unverified suffix state instead of forcing extra dense verifier steps.
        """
        if self.dense_realign_steps >= 0:
            return self.dense_realign_steps
        return max(1, num_spec_tokens)

    def effective_prefix_reject_dense_realign_steps(
        self, num_spec_tokens: int
    ) -> int:
        """Return dense TLM steps after a prefix rejection.

        This is disabled by default because prefix rejection is the intended
        suffix-pruning fast path. Positive values are diagnostic and test
        whether letting the EAGLE drafter run immediately after a shorter
        prefix verifier forward causes later token drift.
        """
        if self.prefix_reject_dense_realign_steps <= 0:
            return 0
        return self.prefix_reject_dense_realign_steps


@dataclass(frozen=True)
class AsyncPrefixCandidate:
    request_id: str
    selected_h: int
    k: int
    selected_benefit: float = 0.0
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


def normalize_candidates(
    k: int, candidates: tuple[int | str, ...]
) -> list[int]:
    values: set[int] = {k}
    for candidate in candidates:
        if isinstance(candidate, str):
            item = candidate.strip()
            if item == "full":
                values.add(k)
            elif item:
                values.add(int(item))
        else:
            values.add(int(candidate))
    return sorted(value for value in values if 1 <= value <= k)


@lru_cache(maxsize=8)
def _load_calibration_model(path: str) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        model = json.load(f)
    if model.get("model_type") != "binning":
        raise ValueError(
            "unsupported SpecLink-CV calibration type: "
            f"{model.get('model_type')}"
        )
    return model


def _apply_binning_calibration(
    probs: list[float], config: SpecLinkCVRuntimeConfig
) -> tuple[list[float], str, str | None]:
    if not config.calibration_path:
        return probs, "draft_selected_prob_uncalibrated", None
    try:
        model = _load_calibration_model(config.calibration_path)
        bins = list(model.get("bins") or [])
        if not bins:
            raise ValueError("calibration model has no bins")
        global_rate = float(model.get("global_acceptance_rate", 0.5))
        calibrated: list[float] = []
        for prob in probs:
            value = min(max(float(prob), 0.0), 1.0)
            matched = None
            for item in bins:
                left = float(item.get("left", 0.0))
                right = float(item.get("right", 1.0))
                if left <= value < right or (value == 1.0 and right == 1.0):
                    matched = item
                    break
            rate = global_rate if matched is None else float(
                matched.get("acceptance_rate", global_rate)
            )
            calibrated.append(min(max(rate, 1e-6), 1.0))
        return calibrated, "calibrated_binning", None
    except Exception as exc:  # noqa: BLE001
        return probs, "draft_selected_prob_uncalibrated", str(exc)


def choose_prefix_len(
    k: int,
    config: SpecLinkCVRuntimeConfig,
    a_hat: list[float] | None = None,
) -> tuple[int, str, dict[str, Any]]:
    """Choose the prefix length to verify first.

    When a calibration model is configured, draft selected probabilities are
    mapped to empirical local acceptance estimates before scoring chunks.
    """
    if k <= 1:
        return k, "one_token", {}
    if config.force_prefix_len > 0:
        forced = min(max(1, config.force_prefix_len), k)
        return forced, "forced_prefix_len", {"forced_prefix_len": forced}
    if config.confidence_sizing:
        if not a_hat:
            return k, "confidence_unavailable_one_shot", {}
        probs = [min(max(float(value), 1e-6), 1.0) for value in a_hat[:k]]
        if len(probs) < k:
            probs.extend([0.5] * (k - len(probs)))
        raw_probs = list(probs)
        probs, confidence_source, calibration_error = _apply_binning_calibration(
            probs, config
        )
        candidates = normalize_candidates(k, config.candidate_chunks)
        prefix_survival: dict[int, float] = {}
        reject_probability: dict[int, float] = {}
        expected_benefit: dict[int, float] = {}
        # A small fixed extra forward cost keeps high-confidence drafts on the
        # one-shot path unless the expected skipped suffix is meaningful.
        extra_forward_cost = 1.0
        for h in candidates:
            survival = math.prod(probs[:h])
            reject_prob = 1.0 - survival
            suffix_len = max(k - h, 0)
            benefit = reject_prob * suffix_len - extra_forward_cost
            prefix_survival[h] = survival
            reject_probability[h] = reject_prob
            expected_benefit[h] = benefit
        selected_h = max(candidates, key=lambda h: (expected_benefit[h], h))
        reason = (
            "confidence_calibrated"
            if confidence_source == "calibrated_binning"
            else "confidence_uncalibrated"
        )
        if expected_benefit[selected_h] <= config.min_benefit:
            selected_h = k
            reason = "one_shot_min_benefit"
        details = {
            "candidate_chunks": candidates,
            "a_hat": probs,
            "raw_draft_selected_prob": raw_probs,
            "prefix_survival": prefix_survival,
            "reject_probability": reject_probability,
            "expected_benefit": expected_benefit,
            "selected_benefit": expected_benefit.get(selected_h, 0.0),
            "confidence_source": confidence_source,
            "calibration_path": config.calibration_path,
        }
        if calibration_error is not None:
            details["calibration_error"] = calibration_error
        return selected_h, reason, details
    h = half_chunk(k, config.default_half_policy)
    return h, "fixed_half", {"candidate_chunks": normalize_candidates(k, config.candidate_chunks)}


def apply_roofline_packing_policy(
    *,
    k: int,
    selected_h: int,
    reason: str,
    decision: dict[str, Any],
    config: SpecLinkCVRuntimeConfig,
    candidate_seq_count: int,
    token_budget: int,
    seq_budget: int,
) -> tuple[int, str, dict[str, Any]]:
    """Apply a lightweight live packing gate for small prefix chunks.

    The current live implementation does not have a full async verification
    queue. When roofline packing is enabled, this gate estimates whether the
    current scheduler step has enough ready prefix work to avoid an underfilled
    verifier launch. If not, it falls back to exact one-shot verification.
    """
    if not config.roofline_packing or selected_h >= k:
        return selected_h, reason, decision
    seqs = max(1, int(candidate_seq_count))
    effective_token_budget = max(1, config.max_verify_tokens_per_step or token_budget or k)
    effective_seq_budget = config.effective_seq_budget(seq_budget or seqs)
    prefix_tokens = max(1, selected_h) * seqs
    token_util = min(1.0, prefix_tokens / effective_token_budget)
    seq_util = min(1.0, seqs / effective_seq_budget)
    predicted_utilization = max(token_util, seq_util)
    roofline = {
        "roofline_packing": True,
        "candidate_seq_count": seqs,
        "prefix_tokens": prefix_tokens,
        "token_budget": effective_token_budget,
        "seq_budget": effective_seq_budget,
        "token_budget_utilization": token_util,
        "seq_budget_utilization": seq_util,
        "predicted_utilization": predicted_utilization,
        "util_threshold": config.util_threshold,
    }
    updated = dict(decision)
    updated["roofline"] = roofline
    if predicted_utilization < config.util_threshold:
        updated["roofline_fallback_reason"] = "underfilled_prefix_batch"
        return k, "roofline_fallback_one_shot", updated
    return selected_h, reason, updated


def should_wait_for_global_batch_fill(
    *, waiting_reqs: int, running_reqs: int, max_running_reqs: int
) -> bool:
    """Return whether a global barrier should wait for more requests.

    Waiting while the running batch is already full deadlocks generation: queued
    requests cannot be admitted until the current running requests make
    progress, but the barrier would also prevent those requests from verifying
    their prefix chunks.
    """
    return waiting_reqs > 0 and running_reqs < max(1, max_running_reqs)


def select_async_prefix_dispatch(
    candidates: list[AsyncPrefixCandidate],
    *,
    config: SpecLinkCVRuntimeConfig,
    token_budget: int,
    seq_budget: int,
    now: float,
    has_other_ready_work: bool,
) -> tuple[set[str], dict[str, Any]]:
    if not candidates:
        return set(), {"reason": "empty"}
    effective_token_budget = max(
        1, config.max_verify_tokens_per_step or token_budget or 1
    )
    effective_seq_budget = config.effective_seq_budget(seq_budget)
    scored: list[tuple[bool, float, str, AsyncPrefixCandidate]] = []
    for candidate in candidates:
        age_ms = candidate.age_ms(now)
        urgent = age_ms >= config.max_queue_wait_ms
        priority = (
            float(candidate.selected_benefit) / max(candidate.selected_h, 1)
            + 0.001 * age_ms
        )
        scored.append((urgent, priority, candidate.request_id, candidate))
    scored.sort(key=lambda item: (not item[0], -item[1], item[2]))

    selected: list[AsyncPrefixCandidate] = []
    selected_tokens = 0
    for _, _, _, candidate in scored:
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

    token_util = min(1.0, selected_tokens / effective_token_budget)
    seq_util = min(1.0, len(selected) / effective_seq_budget)
    predicted_utilization = max(token_util, seq_util)
    urgent_selected = [candidate for candidate in selected if candidate.age_ms(now) >= config.max_queue_wait_ms]

    reason = "simple_priority"
    dispatch = selected
    if config.roofline_packing:
        if predicted_utilization >= config.util_threshold:
            reason = "roofline_packed"
        elif urgent_selected:
            reason = "age_timeout"
            dispatch = urgent_selected
        elif not has_other_ready_work:
            # Avoid a scheduler spin when every runnable request is waiting for
            # better packing. This is still exact prefix verification; it only
            # gives up waiting for more chunks in the current step.
            reason = "no_other_ready_work"
        else:
            reason = "wait_for_utilization"
            dispatch = []

    detail = {
        "reason": reason,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "dispatch_count": len(dispatch),
        "selected_tokens": selected_tokens,
        "dispatch_tokens": sum(candidate.selected_h for candidate in dispatch),
        "token_budget": effective_token_budget,
        "seq_budget": effective_seq_budget,
        "allow_batched_prefix": config.allow_batched_prefix,
        "token_budget_utilization": token_util,
        "seq_budget_utilization": seq_util,
        "predicted_utilization": predicted_utilization,
        "util_threshold": config.util_threshold,
        "max_queue_wait_ms": config.max_queue_wait_ms,
        "candidate_ages_ms": {
            candidate.request_id: candidate.age_ms(now) for candidate in candidates
        },
    }
    return {candidate.request_id for candidate in dispatch}, detail


_JSONL_EVENT_COUNTS: dict[str, int] = {}
_JSONL_LIMIT_REPORTED: set[str] = set()


def append_event(config: SpecLinkCVRuntimeConfig, event: dict[str, Any]) -> None:
    _append_jsonl(config.log_jsonl, event, max_events=config.log_max_events)


def append_profile(config: SpecLinkCVRuntimeConfig, event: dict[str, Any]) -> None:
    _append_jsonl(
        config.profile_jsonl, event, max_events=config.profile_max_events
    )


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
        # Logging must never affect request execution.
        return
