# SPDX-License-Identifier: Apache-2.0
"""Environment-gated dynamic draft length control for Smurfs/FastDraft runs."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_FALSEY = {"0", "false", "FALSE", "no", "NO", "off", "OFF"}
_DEFAULT_METHODS = {"draft_model"}
_VLLM_SPEC_MARKERS = ("spec-webui-vllm-spec-",)
_SMURFS_MARKERS = ("spec-webui-smurfs-",)
_DEFAULT_OUTPUT = (
    "/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/"
    "examples/evaluate/eval-guidellm/temp/smurfs_ui_backend/"
    "smurfs_dynamic_k.jsonl"
)
_STATE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _methods_env() -> set[str]:
    value = os.getenv("SPECLINK_SMURFS_DYNAMIC_METHODS", "")
    if not value:
        return set(_DEFAULT_METHODS)
    return {item.strip() for item in value.split(",") if item.strip()}


def enabled(method: str | None = None) -> bool:
    value = os.getenv("SPECLINK_SMURFS_DYNAMIC_ENABLE")
    if value in _FALSEY:
        return False
    if value not in _TRUTHY and not (
        os.getenv("SPECLINK_SMURFS_DYNAMIC_INITIAL_K")
        or os.getenv("SPECLINK_SMURFS_DYNAMIC_UPDATE_DRAFT_TOKENS")
    ):
        return False
    if method is None:
        return True
    return method in _methods_env()


def _log_event(event: dict[str, Any]) -> None:
    output = os.getenv("SPECLINK_SMURFS_DYNAMIC_OUT", "") or _DEFAULT_OUTPUT
    if not output:
        return
    event = dict(event)
    event.setdefault("time_s", time.time())
    try:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        # Never let diagnostics perturb serving.
        return


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def is_vllm_spec_request(request_id: str | None) -> bool:
    return _contains_any(str(request_id or ""), _VLLM_SPEC_MARKERS)


def is_smurfs_request(request_id: str | None) -> bool:
    return _contains_any(str(request_id or ""), _SMURFS_MARKERS)


def fixed_draft_limit_for_request(
    request_id: str | None,
    configured_max_k: int,
) -> int | None:
    if not is_vllm_spec_request(request_id):
        return None
    fixed_k = _int_env("SPECLINK_VLLM_SPEC_FIXED_K", 12)
    return max(1, min(max(1, int(configured_max_k)), fixed_k))


def log_fixed_proposal(
    *,
    configured_max_k: int,
    active_requests: int,
    effective_k: int,
    method: str,
) -> None:
    _log_event(
        {
            "event": "proposal",
            "mode": "vllm_spec",
            "method": method,
            "active_requests": int(active_requests),
            "effective_k": int(effective_k),
            "configured_max_k": int(configured_max_k),
            "draft_tokens": int(effective_k) * max(0, int(active_requests)),
        }
    )


@dataclass
class _DynamicState:
    min_k: int
    max_k: int
    current_k: int
    update_draft_tokens: int
    up_acceptance: float
    down_acceptance: float
    up_full_prefix: float
    down_avg_accept: float
    min_feedback_before_down: int
    proposal_index: int = 0
    feedback_index: int = 0
    window_drafted: int = 0
    window_accepted: int = 0
    window_requests: int = 0
    window_full_prefix: int = 0

    @classmethod
    def from_env(cls, configured_max_k: int) -> "_DynamicState":
        max_k = max(1, _int_env("SPECLINK_SMURFS_DYNAMIC_MAX_K", configured_max_k))
        max_k = min(max(1, configured_max_k), max_k)
        min_k = max(1, min(max_k, _int_env("SPECLINK_SMURFS_DYNAMIC_MIN_K", 1)))
        initial_k = _int_env("SPECLINK_SMURFS_DYNAMIC_INITIAL_K", min(4, max_k))
        initial_k = max(min_k, min(max_k, initial_k))
        return cls(
            min_k=min_k,
            max_k=max_k,
            current_k=initial_k,
            update_draft_tokens=max(
                1, _int_env("SPECLINK_SMURFS_DYNAMIC_UPDATE_DRAFT_TOKENS", 256)
            ),
            up_acceptance=_float_env("SPECLINK_SMURFS_DYNAMIC_UP_ACCEPTANCE", 0.58),
            down_acceptance=_float_env(
                "SPECLINK_SMURFS_DYNAMIC_DOWN_ACCEPTANCE", 0.38
            ),
            up_full_prefix=_float_env("SPECLINK_SMURFS_DYNAMIC_UP_FULL_PREFIX", 0.12),
            down_avg_accept=_float_env(
                "SPECLINK_SMURFS_DYNAMIC_DOWN_AVG_ACCEPT", 1.20
            ),
            min_feedback_before_down=max(
                0,
                _int_env("SPECLINK_SMURFS_DYNAMIC_MIN_FEEDBACK_BEFORE_DOWN", 4),
            ),
        )


_STATE: _DynamicState | None = None


def _state(configured_max_k: int) -> _DynamicState:
    global _STATE
    configured_max_k = max(1, int(configured_max_k))
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = _DynamicState.from_env(configured_max_k)
            _log_event(
                {
                    "event": "init",
                    "mode": "smurfs",
                    "min_k": _STATE.min_k,
                    "max_k": _STATE.max_k,
                    "current_k": _STATE.current_k,
                    "update_draft_tokens": _STATE.update_draft_tokens,
                    "up_acceptance": _STATE.up_acceptance,
                    "down_acceptance": _STATE.down_acceptance,
                    "up_full_prefix": _STATE.up_full_prefix,
                    "down_avg_accept": _STATE.down_avg_accept,
                    "min_feedback_before_down": _STATE.min_feedback_before_down,
                }
            )
        return _STATE


def current_draft_limit(
    configured_max_k: int,
    *,
    active_requests: int,
    method: str,
) -> int:
    if not enabled(method):
        return max(1, int(configured_max_k))
    state = _state(configured_max_k)
    with _STATE_LOCK:
        k = max(state.min_k, min(state.max_k, state.current_k))
        state.proposal_index += 1
        proposal_index = state.proposal_index
    _log_event(
        {
            "event": "proposal",
            "mode": "smurfs",
            "proposal_index": proposal_index,
            "method": method,
            "active_requests": int(active_requests),
            "effective_k": int(k),
            "configured_max_k": int(configured_max_k),
            "draft_tokens": int(k) * max(0, int(active_requests)),
        }
    )
    return k


def record_verify(
    *,
    num_draft_tokens: int,
    num_accepted_tokens: int,
    method: str | None = None,
    request_id: str | None = None,
) -> None:
    if not enabled(method):
        return
    if is_vllm_spec_request(request_id):
        return
    num_draft_tokens = max(0, int(num_draft_tokens))
    if num_draft_tokens <= 0:
        return
    num_accepted_tokens = max(0, min(num_draft_tokens, int(num_accepted_tokens)))
    state = _state(_int_env("SPECLINK_SMURFS_DYNAMIC_MAX_K", num_draft_tokens))
    event: dict[str, Any] | None = None
    with _STATE_LOCK:
        state.window_drafted += num_draft_tokens
        state.window_accepted += num_accepted_tokens
        state.window_requests += 1
        if num_accepted_tokens >= num_draft_tokens:
            state.window_full_prefix += 1
        if state.window_drafted < state.update_draft_tokens:
            return

        drafted = state.window_drafted
        accepted = state.window_accepted
        requests = state.window_requests
        full_prefix = state.window_full_prefix
        accept_rate = accepted / drafted if drafted else 0.0
        avg_accept = accepted / requests if requests else 0.0
        full_prefix_rate = full_prefix / requests if requests else 0.0

        old_k = state.current_k
        new_k = old_k
        next_feedback_index = state.feedback_index + 1
        should_increase = (
            accept_rate >= state.up_acceptance
            or full_prefix_rate >= state.up_full_prefix
        )
        should_decrease = (
            accept_rate <= state.down_acceptance
            or avg_accept <= state.down_avg_accept
        )
        if should_increase:
            new_k += 1
        elif (
            should_decrease
            and next_feedback_index > state.min_feedback_before_down
        ):
            new_k -= 1
        new_k = max(state.min_k, min(state.max_k, new_k))
        state.current_k = new_k
        state.feedback_index = next_feedback_index
        event = {
            "event": "feedback",
            "mode": "smurfs",
            "feedback_index": state.feedback_index,
            "method": method,
            "old_k": old_k,
            "new_k": new_k,
            "window_draft_tokens": drafted,
            "window_accepted_tokens": accepted,
            "window_requests": requests,
            "window_full_prefix": full_prefix,
            "acceptance_rate": accept_rate,
            "avg_accepted_tokens": avg_accept,
            "full_prefix_rate": full_prefix_rate,
        }
        state.window_drafted = 0
        state.window_accepted = 0
        state.window_requests = 0
        state.window_full_prefix = 0
    _log_event(event)
