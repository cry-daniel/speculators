# SPDX-License-Identifier: Apache-2.0
"""Token-level dense/sparse routing for SpecLink structured 2:4 studies.

This module records DLM selected-token probabilities and builds the per-target
verification row mask used by :mod:`vllm.speclink_linear`. Prefill rows stay
dense. Scored draft rows are ranked globally by per-request cumulative
confidence, and only the Top-N rows stay dense. Verifier bonus, non-draft
decode, and missing-score rows can explicitly use 2:4 because they have no
drafter confidence.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
CONFIDENCE_SEMANTICS = "prefix_product"
ROUTING_MODE = "topk_cumulative_confidence"
VALID_DENSE_TOKEN_BUDGETS = {0, 8, 16, 32, 64, 128, 256}
VALID_LINEAR_STRATEGIES = {
    "auto",
    "full_sparse_residual",
    "full_sparse_dense_override",
    "split_dense_sparse",
    "sparse_only_decode",
}
VALID_MLP_STRATEGIES = {
    "auto",
    "gate_only",
    "linear",
}
VALID_SCORE_BACKENDS = {"torch_softmax", "triton_selected", "triton_fused"}
PURE_BATCH_DENSE_SELECTIONS = frozenset(
    {"batch_adaptive", "batch_alternating", "batch_confidence"}
)
VALID_DENSE_SELECTIONS = {
    "batch_adaptive",
    "balanced_confidence",
    "balanced_low_confidence",
    "balanced_prefix",
    "batch_alternating",
    "batch_confidence",
    "highest",
    "lowest",
    "request_highest",
    "request_lowest",
    "request_contiguous",
}
VALID_DENSE_BUDGET_MODES = {"fixed", "dynamic"}
DEFAULT_DENSE_TOKEN_BUDGET = 32
DEFAULT_DENSE_TOKEN_RATIO = 0.125
DEFAULT_DENSE_MIN_PER_REQUEST = 1
DEFAULT_DENSE_TOKEN_CAP = -1
DEFAULT_LINEAR_STRATEGY = "auto"
DEFAULT_MLP_STRATEGY = "auto"
DEFAULT_SCORE_BACKEND = "triton_fused"

_propose_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "speclink_token_dense_propose_context", default=None
)


@dataclass
class TokenDensePlan:
    dense_mask: torch.Tensor
    dense_rows: torch.Tensor
    sparse_rows: torch.Tensor
    dense_count: int
    sparse_count: int
    total_rows: int
    dense_slots: torch.Tensor | None = None
    all_sparse: bool = False
    contiguous_dense_prefix: bool = False
    has_prefill_rows: bool = False
    prefill_rows: torch.Tensor | None = None
    decode_rows: torch.Tensor | None = None
    prefill_count: int = 0
    decode_count: int = 0
    contiguous_prefill_prefix: bool = False
    contiguous_prefill_suffix: bool = False


@dataclass(frozen=True)
class _CumulativeLogScores:
    values: torch.Tensor


_verify_dense_mask: ContextVar[TokenDensePlan | torch.Tensor | None] = ContextVar(
    "speclink_token_dense_verify_mask", default=None
)
_verify_dense_mask_summary_cache: ContextVar[
    dict[
        tuple[int, str, int | None],
        tuple[torch.Tensor, int, torch.Tensor, torch.Tensor],
    ]
    | None
] = ContextVar("speclink_token_dense_verify_mask_summary_cache", default=None)

_pending_scores: defaultdict[str, deque[Any]] = defaultdict(deque)
_plan_buffers: dict[tuple[str, int | None, int], dict[str, torch.Tensor]] = {}
_lock = threading.Lock()
_stats_accum: dict[str, Any] = {
    "steps": 0,
    "total_scheduled_tokens": 0,
    "total_draft_tokens": 0,
    "scored_draft_tokens": 0,
    "forced_dense_tokens": 0,
    "dense_draft_tokens": 0,
    "sparse_draft_tokens": 0,
    "sparse_unscored_decode_tokens": 0,
    "missing_score_tokens": 0,
    "effective_dense_budget_sum": 0,
    "effective_dense_budget_min": None,
    "effective_dense_budget_max": None,
    "scored_request_count": 0,
    "batch_dense_steps": 0,
    "batch_sparse_steps": 0,
    "batch_confidence_count": 0,
    "batch_confidence_sum": 0.0,
    "batch_confidence_min": None,
    "batch_confidence_max": None,
    "batch_confidence_invalid_count": 0,
    "batch_confidence_histogram": [0] * 20,
    "last_flush_steps": 0,
}
_STATIC_ENABLED = os.getenv("SPECLINK_TOKEN_DENSE_ENABLE", "0") in _TRUTHY
_PRODUCTION_FAST = os.getenv("SPECLINK_PRODUCTION_FAST", "0") in _TRUTHY
_STATIC_MODE = os.getenv("SPECLINK_TOKEN_DENSE_MODE", ROUTING_MODE).strip()
_STATIC_DENSE_TOKEN_BUDGET_RAW = os.getenv(
    "SPECLINK_TOKEN_DENSE_DENSE_TOKENS", str(DEFAULT_DENSE_TOKEN_BUDGET)
).strip()
_STATIC_DENSE_BUDGET_MODE = os.getenv(
    "SPECLINK_TOKEN_DENSE_BUDGET_MODE", "fixed"
).strip()
_STATIC_DENSE_TOKEN_RATIO_RAW = os.getenv(
    "SPECLINK_TOKEN_DENSE_DENSE_RATIO", str(DEFAULT_DENSE_TOKEN_RATIO)
).strip()
_STATIC_DENSE_MIN_PER_REQUEST_RAW = os.getenv(
    "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST",
    str(DEFAULT_DENSE_MIN_PER_REQUEST),
).strip()
_STATIC_DENSE_TOKEN_CAP_RAW = os.getenv(
    "SPECLINK_TOKEN_DENSE_DENSE_CAP", str(DEFAULT_DENSE_TOKEN_CAP)
).strip()
_STATIC_LINEAR_STRATEGY = os.getenv(
    "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY", DEFAULT_LINEAR_STRATEGY
).strip()
_STATIC_MLP_STRATEGY = os.getenv(
    "SPECLINK_TOKEN_DENSE_MLP_STRATEGY", DEFAULT_MLP_STRATEGY
).strip()
_STATIC_SCORE_BACKEND = os.getenv(
    "SPECLINK_TOKEN_DENSE_SCORE_BACKEND", DEFAULT_SCORE_BACKEND
).strip()
_STATIC_DENSE_SELECTION = os.getenv(
    "SPECLINK_TOKEN_DENSE_DENSE_SELECTION", "highest"
).strip()
_STATIC_BATCH_CONFIDENCE_THRESHOLD = float(
    os.getenv("SPECLINK_TOKEN_DENSE_BATCH_CONFIDENCE_THRESHOLD", "0.364")
)
_STATIC_ADAPTIVE_DENSE_MAX_REQUESTS = int(
    os.getenv("SPECLINK_TOKEN_DENSE_ADAPTIVE_DENSE_MAX_REQUESTS", "1")
)
_STATIC_BATCH_ROUTE_BLOCK_STEPS = int(
    os.getenv("SPECLINK_TOKEN_DENSE_BATCH_ROUTE_BLOCK_STEPS", "1")
)
_STATIC_BATCH_ROUTE_INITIAL_CREDIT = float(
    os.getenv("SPECLINK_TOKEN_DENSE_BATCH_ROUTE_INITIAL_CREDIT", "0.0")
)
_STATIC_BALANCED_START_POSITION_RAW = os.getenv(
    "SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION", "0"
).strip()
_STATIC_SPARSE_BONUS = (
    os.getenv("SPECLINK_TOKEN_DENSE_SPARSE_BONUS", "0") in _TRUTHY
)
_STATIC_SPARSE_UNSCORED_DECODE = (
    os.getenv("SPECLINK_TOKEN_DENSE_SPARSE_UNSCORED_DECODE", "0") in _TRUTHY
)
_STATIC_FAST_PLAN = os.getenv("SPECLINK_TOKEN_DENSE_FAST_PLAN", "1") in _TRUTHY
_STATIC_GRAPH_ROUTING = (
    os.getenv("SPECLINK_TOKEN_DENSE_GRAPH_ROUTING", "0") in _TRUTHY
)
_STATIC_NUM_SPEC_TOKENS = int(
    os.getenv("SPECLINK_TOKEN_DENSE_NUM_SPEC_TOKENS", "8")
)
_graph_plan_buffers: dict[
    tuple[str, int | None, int], dict[str, torch.Tensor]
] = {}
_graph_capture_plans: dict[
    tuple[str, int | None, int, int], TokenDensePlan
] = {}
_batch_route_credit = _STATIC_BATCH_ROUTE_INITIAL_CREDIT
_batch_route_remaining = 0
_batch_route_sparse = False


def enabled() -> bool:
    return _STATIC_ENABLED


def production_fast_enabled() -> bool:
    return _PRODUCTION_FAST


def current_verify_dense_mask_summary(
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, int] | None:
    summary = current_verify_dense_row_summary(rows, device)
    if summary is None:
        return None
    row_is_dense, dense_count, _dense_rows, _sparse_rows = summary
    return row_is_dense, dense_count


def current_verify_sparse_only_all_sparse(rows: int) -> bool:
    plan_or_mask = _verify_dense_mask.get()
    return (
        rows > 0
        and isinstance(plan_or_mask, TokenDensePlan)
        and plan_or_mask.all_sparse
        and rows == plan_or_mask.total_rows
    )


def current_verify_has_prefill_rows(rows: int) -> bool:
    plan = _verify_dense_mask.get()
    return (
        isinstance(plan, TokenDensePlan)
        and plan.total_rows == rows
        and plan.has_prefill_rows
    )


def current_verify_prefill_row_summary(
    rows: int,
) -> tuple[int, torch.Tensor, torch.Tensor, str] | None:
    """Return the prefill/decode partition for a mixed verification batch."""

    plan = _verify_dense_mask.get()
    if (
        not isinstance(plan, TokenDensePlan)
        or plan.total_rows != rows
        or not plan.has_prefill_rows
    ):
        return None
    if plan.prefill_rows is None or plan.decode_rows is None:
        return None
    layout = (
        "prefix"
        if plan.contiguous_prefill_prefix
        else "suffix" if plan.contiguous_prefill_suffix else "indexed"
    )
    return (
        plan.prefill_count,
        plan.prefill_rows[: plan.prefill_count],
        plan.decode_rows[: plan.decode_count],
        layout,
    )


def current_verify_contiguous_dense_prefix(rows: int) -> int | None:
    plan = _verify_dense_mask.get()
    if (
        isinstance(plan, TokenDensePlan)
        and plan.contiguous_dense_prefix
        and plan.total_rows == rows
    ):
        return plan.dense_count
    return None


def cudagraph_route(plan: TokenDensePlan | None) -> int:
    if (
        dense_selection() in PURE_BATCH_DENSE_SELECTIONS
        and plan is not None
        and plan.dense_count == 0
        and plan.sparse_count == plan.total_rows
    ):
        return 1
    return 0


def current_verify_dense_row_summary(
    rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor] | None:
    plan_or_mask = _verify_dense_mask.get()
    if plan_or_mask is None:
        plan_or_mask = _graph_capture_plan(rows, device)
        if plan_or_mask is None:
            return None
    if rows <= 0:
        if isinstance(plan_or_mask, TokenDensePlan):
            empty = plan_or_mask.dense_mask[:0].to(device=device)
        else:
            empty = plan_or_mask[:0].to(device=device)
        empty_rows = torch.empty(0, device=device, dtype=torch.int32)
        return empty, 0, empty_rows, empty_rows

    if isinstance(plan_or_mask, TokenDensePlan) and rows == plan_or_mask.total_rows:
        if plan_or_mask.all_sparse:
            empty_mask = plan_or_mask.dense_mask[:0].to(device=device)
            empty_rows = plan_or_mask.dense_rows[:0]
            return empty_mask, 0, empty_rows, empty_rows
        return (
            plan_or_mask.dense_mask[:rows].to(device=device),
            plan_or_mask.dense_count,
            plan_or_mask.dense_rows[: plan_or_mask.dense_count],
            plan_or_mask.sparse_rows[: plan_or_mask.sparse_count],
        )

    dense_mask = (
        plan_or_mask.dense_mask if isinstance(plan_or_mask, TokenDensePlan) else plan_or_mask
    )
    if dense_mask.numel() < rows:
        padding_rows = rows - int(dense_mask.numel())
        if rows % 8 != 0 or padding_rows >= 16:
            raise RuntimeError(
                f"SpecLink token-dense mask has {dense_mask.numel()} rows, "
                f"but linear input has {rows}"
            )
        dense_mask = torch.cat(
            (
                dense_mask.to(device=device),
                torch.zeros(padding_rows, device=device, dtype=torch.bool),
            )
        )
    cache = _verify_dense_mask_summary_cache.get()
    key = (rows, device.type, device.index)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    row_is_dense = dense_mask[:rows].to(device=device)
    dense_count = int(row_is_dense.sum().item())
    dense_rows = row_is_dense.nonzero(as_tuple=False).squeeze(1)
    sparse_rows = (~row_is_dense).nonzero(as_tuple=False).squeeze(1)
    result = (
        row_is_dense,
        dense_count,
        dense_rows.to(dtype=torch.int32).contiguous(),
        sparse_rows.to(dtype=torch.int32).contiguous(),
    )
    if cache is not None:
        cache[key] = result
    return result


def current_verify_dense_slots(
    rows: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the per-row compact dense slot prepared with the route plan."""

    plan = _verify_dense_mask.get()
    if plan is None:
        plan = _graph_capture_plan(rows, device)
    if (
        not isinstance(plan, TokenDensePlan)
        or plan.total_rows != rows
        or plan.dense_slots is None
    ):
        return None
    slots = plan.dense_slots[:rows]
    return slots if slots.device == device else slots.to(device=device)


def dense_token_budget() -> int:
    raw = (
        _STATIC_DENSE_TOKEN_BUDGET_RAW
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_DENSE_TOKENS", str(DEFAULT_DENSE_TOKEN_BUDGET)
        ).strip()
    )
    try:
        budget = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_TOKENS must be one of "
            f"{sorted(VALID_DENSE_TOKEN_BUDGETS)}, got {raw!r}"
        ) from exc
    if budget not in VALID_DENSE_TOKEN_BUDGETS:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_TOKENS must be one of "
            f"{sorted(VALID_DENSE_TOKEN_BUDGETS)}, got {budget}"
        )
    return budget


def dense_budget_mode() -> str:
    value = (
        _STATIC_DENSE_BUDGET_MODE
        if production_fast_enabled()
        else os.getenv("SPECLINK_TOKEN_DENSE_BUDGET_MODE", "fixed").strip()
    )
    if value not in VALID_DENSE_BUDGET_MODES:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_BUDGET_MODE must be one of "
            f"{sorted(VALID_DENSE_BUDGET_MODES)}, got {value!r}"
        )
    return value


def dense_token_ratio() -> float:
    raw = (
        _STATIC_DENSE_TOKEN_RATIO_RAW
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_DENSE_RATIO",
            str(DEFAULT_DENSE_TOKEN_RATIO),
        ).strip()
    )
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_RATIO must be a float in [0, 1], "
            f"got {raw!r}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_RATIO must be a float in [0, 1], "
            f"got {value}"
        )
    return value


def dense_min_per_request() -> int:
    raw = (
        _STATIC_DENSE_MIN_PER_REQUEST_RAW
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST",
            str(DEFAULT_DENSE_MIN_PER_REQUEST),
        ).strip()
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST must be a "
            f"non-negative integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST must be a "
            f"non-negative integer, got {value}"
        )
    return value


def dense_token_cap() -> int:
    raw = (
        _STATIC_DENSE_TOKEN_CAP_RAW
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_DENSE_CAP",
            str(DEFAULT_DENSE_TOKEN_CAP),
        ).strip()
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_CAP must be -1 or a non-negative "
            f"integer, got {raw!r}"
        ) from exc
    if value < -1:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_CAP must be -1 or a non-negative "
            f"integer, got {value}"
        )
    return value


def effective_dense_token_budget(
    scored_rows: int,
    active_requests: int,
) -> int:
    """Return the dense budget for scored draft-verification rows only."""

    scored_rows = max(0, int(scored_rows))
    active_requests = max(0, int(active_requests))
    if dense_budget_mode() == "fixed":
        return min(scored_rows, dense_token_budget())

    ratio_budget = int(dense_token_ratio() * scored_rows + 0.5)
    request_floor = min(
        scored_rows,
        dense_min_per_request() * active_requests,
    )
    cap = dense_token_cap()
    upper_bound = scored_rows if cap < 0 else min(scored_rows, cap)
    if upper_bound < request_floor:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_CAP is smaller than the active "
            "per-request floor: "
            f"cap={upper_bound}, floor={request_floor}, "
            f"active_requests={active_requests}"
        )
    return min(max(ratio_budget, request_floor), upper_bound)


def mode() -> str:
    if production_fast_enabled():
        return _STATIC_MODE
    return os.getenv("SPECLINK_TOKEN_DENSE_MODE", ROUTING_MODE).strip()


def dense_selection() -> str:
    selection = (
        _STATIC_DENSE_SELECTION
        if production_fast_enabled()
        else os.getenv("SPECLINK_TOKEN_DENSE_DENSE_SELECTION", "highest").strip()
    )
    if selection not in VALID_DENSE_SELECTIONS:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_DENSE_SELECTION must be one of "
            f"{sorted(VALID_DENSE_SELECTIONS)}, got {selection!r}"
        )
    return selection


def batch_confidence_threshold() -> float:
    threshold = (
        _STATIC_BATCH_CONFIDENCE_THRESHOLD
        if production_fast_enabled()
        else float(
            os.getenv(
                "SPECLINK_TOKEN_DENSE_BATCH_CONFIDENCE_THRESHOLD",
                "0.364",
            )
        )
    )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_BATCH_CONFIDENCE_THRESHOLD must be in "
            f"[0, 1], got {threshold}"
        )
    return threshold


def adaptive_dense_max_requests() -> int:
    value = (
        _STATIC_ADAPTIVE_DENSE_MAX_REQUESTS
        if production_fast_enabled()
        else int(
            os.getenv(
                "SPECLINK_TOKEN_DENSE_ADAPTIVE_DENSE_MAX_REQUESTS",
                "1",
            )
        )
    )
    if value < 0:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_ADAPTIVE_DENSE_MAX_REQUESTS must be "
            f"non-negative, got {value}"
        )
    return value


def batch_route_block_steps() -> int:
    value = (
        _STATIC_BATCH_ROUTE_BLOCK_STEPS
        if production_fast_enabled()
        else int(os.getenv("SPECLINK_TOKEN_DENSE_BATCH_ROUTE_BLOCK_STEPS", "1"))
    )
    if value <= 0:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_BATCH_ROUTE_BLOCK_STEPS must be positive"
        )
    return value


def balanced_start_position() -> int:
    raw = (
        _STATIC_BALANCED_START_POSITION_RAW
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION",
            "0",
        ).strip()
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION must be a "
            f"non-negative integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION must be a "
            f"non-negative integer, got {value}"
        )
    return value


def sparse_bonus_enabled() -> bool:
    if production_fast_enabled():
        return _STATIC_SPARSE_BONUS
    return os.getenv("SPECLINK_TOKEN_DENSE_SPARSE_BONUS", "0") in _TRUTHY


def sparse_unscored_decode_enabled() -> bool:
    if production_fast_enabled():
        return _STATIC_SPARSE_UNSCORED_DECODE
    return (
        os.getenv("SPECLINK_TOKEN_DENSE_SPARSE_UNSCORED_DECODE", "0")
        in _TRUTHY
    )


def linear_strategy() -> str:
    strategy = (
        _STATIC_LINEAR_STRATEGY
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY", DEFAULT_LINEAR_STRATEGY
        ).strip()
    )
    if strategy not in VALID_LINEAR_STRATEGIES:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY must be one of "
            f"{sorted(VALID_LINEAR_STRATEGIES)}, got {strategy!r}"
        )
    return strategy


def sparse_only_decode_enabled() -> bool:
    return enabled() and linear_strategy() == "sparse_only_decode"


def draft_scores_required() -> bool:
    """Whether token routing consumes drafter confidence scores."""

    if not enabled():
        return False
    selection = dense_selection()
    if selection == "batch_confidence":
        return True
    return (
        linear_strategy() != "sparse_only_decode"
        and selection
        not in {
            "balanced_prefix",
            "batch_adaptive",
            "batch_alternating",
            "request_contiguous",
        }
    )


def mlp_strategy() -> str:
    strategy = (
        _STATIC_MLP_STRATEGY
        if production_fast_enabled()
        else os.getenv("SPECLINK_TOKEN_DENSE_MLP_STRATEGY", DEFAULT_MLP_STRATEGY).strip()
    )
    if strategy not in VALID_MLP_STRATEGIES:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_MLP_STRATEGY must be one of "
            f"{sorted(VALID_MLP_STRATEGIES)}, got {strategy!r}"
        )
    return strategy


def score_backend() -> str:
    backend = (
        _STATIC_SCORE_BACKEND
        if production_fast_enabled()
        else os.getenv(
            "SPECLINK_TOKEN_DENSE_SCORE_BACKEND", DEFAULT_SCORE_BACKEND
        ).strip()
    )
    if backend not in VALID_SCORE_BACKENDS:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_SCORE_BACKEND must be one of "
            f"{sorted(VALID_SCORE_BACKENDS)}, got {backend!r}"
        )
    return backend


def fast_plan_enabled() -> bool:
    if production_fast_enabled():
        return _STATIC_FAST_PLAN
    return os.getenv("SPECLINK_TOKEN_DENSE_FAST_PLAN", "1") in _TRUTHY


def stats_path() -> Path | None:
    if production_fast_enabled():
        return None
    value = os.getenv("SPECLINK_TOKEN_DENSE_STATS_PATH", "").strip()
    return Path(value) if value else None


def stats_detail_enabled() -> bool:
    if production_fast_enabled():
        return False
    return os.getenv("SPECLINK_TOKEN_DENSE_STATS_DETAIL", "0") in _TRUTHY


def stats_interval() -> int:
    if production_fast_enabled():
        return 0
    try:
        return max(0, int(os.getenv("SPECLINK_TOKEN_DENSE_STATS_INTERVAL", "1000")))
    except ValueError:
        return 1000


def _accumulate_stats(record: dict[str, Any]) -> dict[str, Any] | None:
    if production_fast_enabled():
        return None
    with _lock:
        _stats_accum["steps"] += 1
        _stats_accum["total_scheduled_tokens"] += int(
            record.get("total_scheduled_tokens") or 0
        )
        _stats_accum["total_draft_tokens"] += int(record.get("total_draft_tokens") or 0)
        _stats_accum["scored_draft_tokens"] += int(
            record.get("scored_draft_tokens") or 0
        )
        _stats_accum["forced_dense_tokens"] += int(
            record.get("forced_dense_tokens") or 0
        )
        _stats_accum["dense_draft_tokens"] += int(record.get("dense_draft_tokens") or 0)
        _stats_accum["sparse_draft_tokens"] += int(
            record.get("sparse_draft_tokens") or 0
        )
        _stats_accum["sparse_unscored_decode_tokens"] += int(
            record.get("sparse_unscored_decode_tokens") or 0
        )
        _stats_accum["missing_score_tokens"] += int(
            record.get("missing_score_tokens") or 0
        )
        effective_budget = int(record.get("effective_dense_token_budget") or 0)
        _stats_accum["effective_dense_budget_sum"] += effective_budget
        current_min = _stats_accum["effective_dense_budget_min"]
        current_max = _stats_accum["effective_dense_budget_max"]
        _stats_accum["effective_dense_budget_min"] = (
            effective_budget if current_min is None else min(current_min, effective_budget)
        )
        _stats_accum["effective_dense_budget_max"] = (
            effective_budget if current_max is None else max(current_max, effective_budget)
        )
        _stats_accum["scored_request_count"] += int(
            record.get("scored_request_count") or 0
        )
        batch_route = record.get("batch_route")
        if batch_route == "dense":
            _stats_accum["batch_dense_steps"] += 1
        elif batch_route == "sparse":
            _stats_accum["batch_sparse_steps"] += 1
        batch_confidence = record.get("batch_confidence")
        if batch_confidence is not None:
            batch_confidence = float(batch_confidence)
            if math.isfinite(batch_confidence):
                _stats_accum["batch_confidence_count"] += 1
                _stats_accum["batch_confidence_sum"] += batch_confidence
                current_confidence_min = _stats_accum["batch_confidence_min"]
                current_confidence_max = _stats_accum["batch_confidence_max"]
                _stats_accum["batch_confidence_min"] = (
                    batch_confidence
                    if current_confidence_min is None
                    else min(current_confidence_min, batch_confidence)
                )
                _stats_accum["batch_confidence_max"] = (
                    batch_confidence
                    if current_confidence_max is None
                    else max(current_confidence_max, batch_confidence)
                )
                confidence_bin = min(19, max(0, int(batch_confidence * 20)))
                _stats_accum["batch_confidence_histogram"][confidence_bin] += 1
            else:
                _stats_accum["batch_confidence_invalid_count"] += 1
        interval = stats_interval()
        if interval <= 0:
            return None
        steps = int(_stats_accum["steps"])
        if steps - int(_stats_accum["last_flush_steps"]) < interval:
            return None
        _stats_accum["last_flush_steps"] = steps
        total_draft = int(_stats_accum["total_draft_tokens"])
        dense_draft = int(_stats_accum["dense_draft_tokens"])
        sparse_draft = int(_stats_accum["sparse_draft_tokens"])
        return {
            "timestamp": time.time(),
            "event": "verify_token_mask_summary",
            "mode": record.get("mode"),
            "dense_token_budget": record.get("dense_token_budget"),
            "dense_budget_mode": record.get("dense_budget_mode"),
            "dense_token_ratio": record.get("dense_token_ratio"),
            "dense_min_per_request": record.get("dense_min_per_request"),
            "dense_token_cap": record.get("dense_token_cap"),
            "balanced_start_position": record.get("balanced_start_position"),
            "linear_strategy": record.get("linear_strategy"),
            "mlp_strategy": record.get("mlp_strategy"),
            "score_backend": record.get("score_backend"),
            "fast_plan": record.get("fast_plan"),
            "confidence_semantics": record.get("confidence_semantics"),
            "steps": steps,
            "total_scheduled_tokens": int(_stats_accum["total_scheduled_tokens"]),
            "total_draft_tokens": total_draft,
            "scored_draft_tokens": int(_stats_accum["scored_draft_tokens"]),
            "forced_dense_tokens": int(_stats_accum["forced_dense_tokens"]),
            "dense_draft_tokens": dense_draft,
            "sparse_draft_tokens": sparse_draft,
            "sparse_unscored_decode_tokens": int(
                _stats_accum["sparse_unscored_decode_tokens"]
            ),
            "missing_score_tokens": int(_stats_accum["missing_score_tokens"]),
            "scored_request_count": int(_stats_accum["scored_request_count"]),
            "batch_dense_steps": int(_stats_accum["batch_dense_steps"]),
            "batch_sparse_steps": int(_stats_accum["batch_sparse_steps"]),
            "batch_confidence_mean": (
                float(_stats_accum["batch_confidence_sum"])
                / int(_stats_accum["batch_confidence_count"])
                if _stats_accum["batch_confidence_count"]
                else None
            ),
            "batch_confidence_min": _stats_accum["batch_confidence_min"],
            "batch_confidence_max": _stats_accum["batch_confidence_max"],
            "batch_confidence_count": int(
                _stats_accum["batch_confidence_count"]
            ),
            "batch_confidence_invalid_count": int(
                _stats_accum["batch_confidence_invalid_count"]
            ),
            "batch_confidence_histogram": list(
                _stats_accum["batch_confidence_histogram"]
            ),
            "effective_dense_budget_mean": (
                float(_stats_accum["effective_dense_budget_sum"]) / steps
                if steps
                else None
            ),
            "effective_dense_budget_min": _stats_accum[
                "effective_dense_budget_min"
            ],
            "effective_dense_budget_max": _stats_accum[
                "effective_dense_budget_max"
            ],
            "dense_draft_fraction": dense_draft / total_draft if total_draft else None,
            "sparse_draft_fraction": sparse_draft / total_draft if total_draft else None,
            "gpu_topk": record.get("gpu_topk"),
            "fixed_row_buffers": record.get("fixed_row_buffers"),
        }


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
    if not enabled():
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


def begin_verify_context(dense_mask: TokenDensePlan | torch.Tensor | None) -> Any:
    if not enabled() or dense_mask is None:
        return None
    mask_token = _verify_dense_mask.set(dense_mask)
    cache_token = _verify_dense_mask_summary_cache.set({})
    return mask_token, cache_token


def end_verify_context(token: Any) -> None:
    if token is not None:
        if isinstance(token, tuple) and len(token) == 2:
            mask_token, cache_token = token
            _verify_dense_mask_summary_cache.reset(cache_token)
            _verify_dense_mask.reset(mask_token)
        else:
            _verify_dense_mask.reset(token)


def _compute_selected_logprobs(
    logits: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    return compute_token_logprobs(logits, selected.view(-1, 1)).squeeze(1)


def compute_greedy_token_ids_and_logprobs(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.worker.gpu.sample.logprob import (
        compute_greedy_token_ids_and_logprobs as compute_greedy,
    )

    return compute_greedy(logits)


@torch.inference_mode()
def record_draft_scores(
    *,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    selected_logprobs_by_position: list[torch.Tensor] | None = None,
    temperature: torch.Tensor | None = None,
    method: str = "",
) -> None:
    """Record selected-token confidence for prefix-confidence routing."""
    if not enabled():
        return
    if linear_strategy() == "sparse_only_decode":
        return
    ctx = _propose_context.get()
    source_count = (
        len(selected_logprobs_by_position)
        if selected_logprobs_by_position is not None
        else len(logits_by_position)
    )
    if ctx is None or source_count <= 0:
        return

    batch_size = min(int(draft_token_ids.shape[0]), len(ctx["req_ids"]))
    num_spec_tokens = min(int(draft_token_ids.shape[1]), source_count)
    if batch_size <= 0 or num_spec_tokens <= 0:
        return

    draft_token_ids = draft_token_ids[:batch_size, :num_spec_tokens]
    backend = score_backend()
    score_columns: list[torch.Tensor] = []
    if selected_logprobs_by_position is not None:
        if backend != "triton_fused":
            raise RuntimeError(
                "precomputed draft logprobs require the triton_fused backend"
            )
        score_columns.extend(
            scores[:batch_size].detach()
            for scores in selected_logprobs_by_position[:num_spec_tokens]
        )
    else:
        for pos, logits in enumerate(logits_by_position[:num_spec_tokens]):
            logits = logits[:batch_size].detach()
            selected = draft_token_ids[:batch_size, pos].to(
                device=logits.device, dtype=torch.long
            )
            if backend in {"triton_selected", "triton_fused"}:
                selected_logprob = _compute_selected_logprobs(logits, selected)
                score_columns.append(selected_logprob.detach())
            else:
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                selected_logprob = log_probs.gather(
                    1, selected.view(-1, 1)
                ).squeeze(1)
                score_columns.append(selected_logprob.exp().detach())

    if not score_columns:
        return
    per_req_scores = torch.stack(score_columns, dim=1).contiguous()
    if backend != "torch_softmax":
        per_req_scores = per_req_scores.cumsum(dim=1)

    req_ids = ctx["req_ids"]
    with _lock:
        for req_idx in range(batch_size):
            scores = per_req_scores[req_idx]
            if backend != "torch_softmax":
                _pending_scores[req_ids[req_idx]].append(
                    _CumulativeLogScores(scores)
                )
            else:
                _pending_scores[req_ids[req_idx]].append(scores.clone())


def _write_stats(record: dict[str, Any]) -> None:
    if production_fast_enabled():
        return
    path = stats_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _fixed_capacity(total_rows: int) -> int:
    return max(1, _next_power_of_two(total_rows))


def _get_plan_buffers(device: torch.device, total_rows: int) -> dict[str, torch.Tensor]:
    capacity = _fixed_capacity(total_rows)
    key = (device.type, device.index, capacity)
    buffers = _plan_buffers.get(key)
    if buffers is None:
        buffers = {
            "dense_mask": torch.empty(capacity, device=device, dtype=torch.bool),
            "dense_rows": torch.empty(capacity, device=device, dtype=torch.int32),
            "sparse_rows": torch.empty(capacity, device=device, dtype=torch.int32),
            "dense_slots": torch.empty(capacity, device=device, dtype=torch.int32),
            "prefill_rows": torch.empty(capacity, device=device, dtype=torch.int32),
            "decode_rows": torch.empty(capacity, device=device, dtype=torch.int32),
            "row_ids_int32": torch.arange(
                capacity,
                device=device,
                dtype=torch.int32,
            ),
            "row_ids_long": torch.arange(
                capacity,
                device=device,
                dtype=torch.long,
            ),
        }
        _plan_buffers[key] = buffers
    return buffers


def _get_graph_plan_buffers(
    device: torch.device,
    total_rows: int,
) -> dict[str, torch.Tensor]:
    key = (device.type, device.index, total_rows)
    buffers = _graph_plan_buffers.get(key)
    if buffers is None:
        buffers = {
            "dense_mask": torch.empty(
                total_rows,
                device=device,
                dtype=torch.bool,
            ),
            "dense_rows": torch.empty(
                total_rows,
                device=device,
                dtype=torch.int32,
            ),
            "sparse_rows": torch.empty(
                total_rows,
                device=device,
                dtype=torch.int32,
            ),
            "dense_slots": torch.empty(
                total_rows,
                device=device,
                dtype=torch.int32,
            ),
            "prefill_rows": torch.empty(
                total_rows,
                device=device,
                dtype=torch.int32,
            ),
            "decode_rows": torch.empty(
                total_rows,
                device=device,
                dtype=torch.int32,
            ),
            "row_ids_int32": torch.arange(
                total_rows,
                device=device,
                dtype=torch.int32,
            ),
            "row_ids_long": torch.arange(
                total_rows,
                device=device,
                dtype=torch.long,
            ),
        }
        _graph_plan_buffers[key] = buffers
    return buffers


def _graph_sparse_count(total_rows: int) -> int:
    if _STATIC_LINEAR_STRATEGY == "sparse_only_decode":
        return total_rows
    group_size = _STATIC_NUM_SPEC_TOKENS + 1
    max_requests = total_rows // group_size
    scored_rows = max_requests * max(0, _STATIC_NUM_SPEC_TOKENS)
    budget = effective_dense_token_budget(scored_rows, max_requests)
    if _STATIC_DENSE_SELECTION == "request_contiguous":
        dense_requests = min(
            max_requests,
            int(budget / max(1, _STATIC_NUM_SPEC_TOKENS) + 0.5),
        )
        return (max_requests - dense_requests) * group_size
    sparse_rows = max(0, scored_rows - budget)
    if _STATIC_SPARSE_BONUS:
        sparse_rows += max_requests
    return min(total_rows, sparse_rows)


def _full_cudagraph_context_active() -> bool:
    if not _STATIC_GRAPH_ROUTING:
        return False
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return False
        context = get_forward_context()
        return (
            context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and context.batch_descriptor is not None
            and context.batch_descriptor.uniform
        )
    except Exception:
        return False


def _get_or_create_graph_plan(
    rows: int,
    device: torch.device,
    route: int = 0,
) -> TokenDensePlan | None:
    if rows <= 0 or not _STATIC_GRAPH_ROUTING:
        return None
    route = 1 if route else 0
    key = (device.type, device.index, rows, route)
    cached = _graph_capture_plans.get(key)
    if cached is not None:
        return cached
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SpecLink graph routing plan was not initialized during CUDA graph "
            f"warmup for {rows} rows"
        )

    buffers = _get_graph_plan_buffers(device, rows)
    dense_mask = buffers["dense_mask"]
    dense_mask.fill_(True)
    sparse_count = (
        rows
        if _STATIC_DENSE_SELECTION in PURE_BATCH_DENSE_SELECTIONS
        and route == 1
        else (
            0
            if _STATIC_DENSE_SELECTION
            in PURE_BATCH_DENSE_SELECTIONS
            else _graph_sparse_count(rows)
        )
    )
    if sparse_count > 0:
        dense_mask[rows - sparse_count :] = False
    plan = _make_plan(
        dense_mask,
        dense_count=rows - sparse_count,
        sparse_count=sparse_count,
        total_rows=rows,
        buffers=buffers,
        contiguous_dense_prefix=(
            _STATIC_DENSE_SELECTION == "request_contiguous"
        ),
    )
    _graph_capture_plans[key] = plan
    return plan


def _graph_capture_plan(
    rows: int,
    device: torch.device,
) -> TokenDensePlan | None:
    if not _full_cudagraph_context_active():
        return None
    route = 0
    if _STATIC_DENSE_SELECTION in PURE_BATCH_DENSE_SELECTIONS:
        from vllm.forward_context import get_forward_context

        batch_descriptor = get_forward_context().batch_descriptor
        if batch_descriptor is not None:
            route = batch_descriptor.speclink_route
    return _get_or_create_graph_plan(rows, device, route)


def prepare_cudagraph_plan(
    rows: int,
    device: torch.device,
    *,
    uniform_decode: bool,
    route: int = 0,
) -> TokenDensePlan | None:
    if not uniform_decode:
        return None
    return _get_or_create_graph_plan(rows, device, route)


def verify_plan_fits_cudagraph(
    plan: TokenDensePlan | None,
    *,
    actual_rows: int,
    padded_rows: int,
) -> bool:
    if not _STATIC_GRAPH_ROUTING:
        return True
    if padded_rows < actual_rows:
        return False
    if plan is not None and plan.has_prefill_rows:
        return False
    if _STATIC_DENSE_SELECTION in PURE_BATCH_DENSE_SELECTIONS:
        return (
            actual_rows == padded_rows
            and (
                plan is None
                or (
                plan.dense_count == plan.total_rows
                or plan.sparse_count == plan.total_rows
                )
            )
        )
    actual_sparse_count = plan.sparse_count if plan is not None else 0
    graph_sparse_count = _graph_sparse_count(padded_rows)
    return (
        graph_sparse_count >= actual_sparse_count
        and graph_sparse_count - actual_sparse_count
        <= padded_rows - actual_rows
    )


def pad_verify_plan_for_cudagraph(
    plan: TokenDensePlan | None,
    *,
    actual_rows: int,
    padded_rows: int,
    device: torch.device,
) -> TokenDensePlan | None:
    """Move a dynamic plan into fixed per-capture-size graph buffers."""
    if not _STATIC_GRAPH_ROUTING:
        return plan
    if padded_rows < actual_rows:
        raise RuntimeError(
            f"CUDA graph rows {padded_rows} are smaller than actual rows {actual_rows}"
        )
    if _STATIC_DENSE_SELECTION in PURE_BATCH_DENSE_SELECTIONS:
        if actual_rows != padded_rows:
            raise RuntimeError(
                "SpecLink dual-route CUDA graphs require an unpadded pure plan"
            )
        return plan

    actual_sparse_count = plan.sparse_count if plan is not None else 0
    graph_sparse_count = _graph_sparse_count(padded_rows)
    if graph_sparse_count < actual_sparse_count:
        raise RuntimeError(
            "SpecLink graph plan cannot represent the runtime sparse rows: "
            f"graph={graph_sparse_count}, runtime={actual_sparse_count}"
        )
    padding_sparse_count = graph_sparse_count - actual_sparse_count
    padding_rows = padded_rows - actual_rows
    if padding_sparse_count > padding_rows:
        raise RuntimeError(
            "SpecLink graph plan lacks padding rows for fixed sparse count: "
            f"need={padding_sparse_count}, available={padding_rows}"
        )

    buffers = _get_graph_plan_buffers(device, padded_rows)
    dense_mask = buffers["dense_mask"]
    dense_mask.fill_(True)
    if actual_sparse_count > 0:
        assert plan is not None
        if plan.all_sparse:
            dense_mask[:actual_rows] = False
        else:
            sparse_rows = plan.sparse_rows[:actual_sparse_count]
            dense_mask[sparse_rows.to(dtype=torch.long)] = False
    if padding_sparse_count > 0:
        dense_mask[
            actual_rows : actual_rows + padding_sparse_count
        ] = False
    return _make_plan(
        dense_mask,
        dense_count=padded_rows - graph_sparse_count,
        sparse_count=graph_sparse_count,
        total_rows=padded_rows,
        buffers=buffers,
        contiguous_dense_prefix=(
            plan is not None
            and plan.contiguous_dense_prefix
            and actual_rows == padded_rows
        ),
        has_prefill_rows=(
            plan is not None and plan.has_prefill_rows
        ),
    )


def _as_score_tensor(scores: Any, device: torch.device) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        return scores.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(scores, device=device, dtype=torch.float32)


def _make_plan(
    dense_mask: torch.Tensor,
    *,
    dense_count: int,
    sparse_count: int,
    total_rows: int,
    all_sparse: bool = False,
    contiguous_dense_prefix: bool = False,
    has_prefill_rows: bool = False,
    prefill_mask: torch.Tensor | None = None,
    contiguous_prefill_prefix: bool = False,
    contiguous_prefill_suffix: bool = False,
    buffers: dict[str, torch.Tensor] | None = None,
) -> TokenDensePlan:
    if all_sparse:
        buffers = _get_plan_buffers(dense_mask.device, 1)
        return TokenDensePlan(
            dense_mask=dense_mask[:0],
            dense_rows=buffers["dense_rows"][:0],
            sparse_rows=buffers["sparse_rows"][:0],
            dense_count=0,
            sparse_count=total_rows,
            total_rows=total_rows,
            dense_slots=buffers["dense_slots"][:0],
            all_sparse=True,
            has_prefill_rows=has_prefill_rows,
        )

    if buffers is None:
        buffers = _get_plan_buffers(dense_mask.device, total_rows)
    target_dense_mask = buffers["dense_mask"][:total_rows]
    if target_dense_mask.data_ptr() != dense_mask.data_ptr():
        target_dense_mask.copy_(dense_mask)

    if contiguous_dense_prefix:
        if dense_count:
            buffers["dense_rows"][:dense_count].copy_(
                torch.arange(
                    dense_count,
                    device=dense_mask.device,
                    dtype=torch.int32,
                )
            )
        if sparse_count:
            buffers["sparse_rows"][:sparse_count].copy_(
                torch.arange(
                    dense_count,
                    total_rows,
                    device=dense_mask.device,
                    dtype=torch.int32,
                )
            )
    elif dense_count == 0 and sparse_count == total_rows:
        buffers["sparse_rows"][:total_rows].copy_(
            torch.arange(total_rows, device=dense_mask.device, dtype=torch.int32)
        )
    else:
        dense_rows = dense_mask.nonzero(as_tuple=False).squeeze(1).to(dtype=torch.int32)
        sparse_rows = (~dense_mask).nonzero(as_tuple=False).squeeze(1).to(dtype=torch.int32)
        if dense_count:
            buffers["dense_rows"][:dense_count].copy_(dense_rows)
        if sparse_count:
            buffers["sparse_rows"][:sparse_count].copy_(sparse_rows)
    target_dense_slots = buffers["dense_slots"][:total_rows]
    target_dense_slots.fill_(-1)
    if dense_count:
        target_dense_slots.masked_scatter_(
            target_dense_mask,
            buffers["row_ids_int32"][:dense_count],
        )
    prefill_count = 0
    decode_count = total_rows
    if has_prefill_rows:
        if prefill_mask is None:
            prefill_mask = target_dense_mask
        else:
            prefill_mask = prefill_mask[:total_rows].to(
                device=dense_mask.device,
                dtype=torch.bool,
            )
        prefill_count = int(prefill_mask.sum().item())
        decode_count = total_rows - prefill_count
        if prefill_count:
            buffers["prefill_rows"][:prefill_count].copy_(
                prefill_mask.nonzero(as_tuple=False).squeeze(1).to(torch.int32)
            )
        if decode_count:
            buffers["decode_rows"][:decode_count].copy_(
                (~prefill_mask).nonzero(as_tuple=False).squeeze(1).to(torch.int32)
            )
    return TokenDensePlan(
        dense_mask=target_dense_mask,
        dense_rows=buffers["dense_rows"],
        sparse_rows=buffers["sparse_rows"],
        dense_count=dense_count,
        sparse_count=sparse_count,
        total_rows=total_rows,
        dense_slots=target_dense_slots,
        all_sparse=False,
        contiguous_dense_prefix=contiguous_dense_prefix,
        has_prefill_rows=has_prefill_rows,
        prefill_rows=buffers["prefill_rows"] if has_prefill_rows else None,
        decode_rows=buffers["decode_rows"] if has_prefill_rows else None,
        prefill_count=prefill_count,
        decode_count=decode_count,
        contiguous_prefill_prefix=contiguous_prefill_prefix,
        contiguous_prefill_suffix=contiguous_prefill_suffix,
    )


def _build_contiguous_request_plan(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    num_decode_draft_tokens: Any | None,
    cu_num_scheduled_tokens: Any,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> TokenDensePlan:
    """Route a trailing set of complete decode requests through 2:4."""

    dense_mask = torch.ones(
        total_num_scheduled_tokens,
        device=device,
        dtype=torch.bool,
    )
    candidates: list[tuple[int, int, int]] = []
    scored_rows = 0
    for req_idx, _req_id in enumerate(req_ids):
        draft_rows = int(num_draft_tokens[req_idx])
        if draft_rows <= 0:
            continue
        decode_rows = (
            int(num_decode_draft_tokens[req_idx])
            if num_decode_draft_tokens is not None
            else draft_rows
        )
        if decode_rows < 0:
            continue
        end = int(cu_num_scheduled_tokens[req_idx])
        scheduled_rows = int(num_scheduled_tokens[req_idx])
        start = end - scheduled_rows
        candidates.append((start, end, draft_rows))
        scored_rows += draft_rows

    budget = effective_dense_token_budget(scored_rows, len(candidates))
    cumulative = [0]
    for _start, _end, draft_rows in candidates:
        cumulative.append(cumulative[-1] + draft_rows)
    dense_request_count = min(
        range(len(cumulative)),
        key=lambda index: (abs(cumulative[index] - budget), -index),
    )

    sparse_candidates = candidates[dense_request_count:]
    sparse_count = 0
    for start, end, _draft_rows in sparse_candidates:
        dense_mask[start:end] = False
        sparse_count += end - start

    dense_count = total_num_scheduled_tokens - sparse_count
    contiguous_dense_prefix = (
        sparse_count > 0
        and sparse_candidates[0][0] == dense_count
        and sparse_candidates[-1][1] == total_num_scheduled_tokens
        and all(
            left[1] == right[0]
            for left, right in zip(
                sparse_candidates,
                sparse_candidates[1:],
                strict=False,
            )
        )
    )
    return _make_plan(
        dense_mask,
        dense_count=dense_count,
        sparse_count=sparse_count,
        total_rows=total_num_scheduled_tokens,
        contiguous_dense_prefix=contiguous_dense_prefix,
    )


def _build_alternating_batch_plan(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    num_decode_draft_tokens: Any | None,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> TokenDensePlan | None:
    """Choose a pure dense or pure sparse decode batch by a fixed duty cycle."""

    for req_idx, _req_id in enumerate(req_ids):
        scheduled_rows = int(num_scheduled_tokens[req_idx])
        if scheduled_rows <= 0:
            continue
        draft_rows = int(num_draft_tokens[req_idx])
        decode_rows = (
            int(num_decode_draft_tokens[req_idx])
            if num_decode_draft_tokens is not None
            else draft_rows
        )
        if decode_rows < 0 or draft_rows <= 0 or scheduled_rows != draft_rows + 1:
            return None

    global _batch_route_credit, _batch_route_remaining, _batch_route_sparse
    sparse_fraction = 1.0 - dense_token_ratio()
    with _lock:
        if _batch_route_remaining <= 0:
            _batch_route_credit += sparse_fraction
            _batch_route_sparse = _batch_route_credit >= 1.0
            if _batch_route_sparse:
                _batch_route_credit -= 1.0
            _batch_route_remaining = batch_route_block_steps()
        use_sparse = _batch_route_sparse
        _batch_route_remaining -= 1
    if not use_sparse:
        dense_mask = torch.ones(
            total_num_scheduled_tokens,
            device=device,
            dtype=torch.bool,
        )
        return _make_plan(
            dense_mask,
            dense_count=total_num_scheduled_tokens,
            sparse_count=0,
            total_rows=total_num_scheduled_tokens,
            contiguous_dense_prefix=True,
        )

    empty_mask = torch.empty(0, device=device, dtype=torch.bool)
    return _make_plan(
        empty_mask,
        dense_count=0,
        sparse_count=total_num_scheduled_tokens,
        total_rows=total_num_scheduled_tokens,
        all_sparse=True,
    )


def _build_adaptive_batch_plan(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    num_decode_draft_tokens: Any | None,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> TokenDensePlan | None:
    """Route small verification batches dense and larger batches sparse."""

    active_requests = 0
    total_draft_tokens = 0
    for req_idx, _req_id in enumerate(req_ids):
        scheduled_rows = int(num_scheduled_tokens[req_idx])
        if scheduled_rows <= 0:
            continue
        draft_rows = int(num_draft_tokens[req_idx])
        decode_rows = (
            int(num_decode_draft_tokens[req_idx])
            if num_decode_draft_tokens is not None
            else draft_rows
        )
        if decode_rows < 0 or draft_rows <= 0 or scheduled_rows != draft_rows + 1:
            return None
        active_requests += 1
        total_draft_tokens += draft_rows

    use_sparse = active_requests > adaptive_dense_max_requests()
    if not production_fast_enabled():
        record = {
            "timestamp": time.time(),
            "event": "verify_token_mask",
            "mode": mode(),
            "dense_token_budget": 0,
            "effective_dense_token_budget": 0,
            "dense_budget_mode": dense_budget_mode(),
            "dense_token_ratio": dense_token_ratio(),
            "dense_min_per_request": dense_min_per_request(),
            "dense_token_cap": dense_token_cap(),
            "linear_strategy": linear_strategy(),
            "mlp_strategy": mlp_strategy(),
            "confidence_semantics": "active_request_threshold",
            "request_count": active_requests,
            "total_scheduled_tokens": total_num_scheduled_tokens,
            "total_draft_tokens": total_draft_tokens,
            "scored_draft_tokens": total_draft_tokens,
            "scored_request_count": active_requests,
            "forced_dense_tokens": 0 if use_sparse else total_num_scheduled_tokens,
            "dense_draft_tokens": 0 if use_sparse else total_draft_tokens,
            "sparse_draft_tokens": total_draft_tokens if use_sparse else 0,
            "missing_score_tokens": 0,
            "batch_route": "sparse" if use_sparse else "dense",
            "adaptive_dense_max_requests": adaptive_dense_max_requests(),
            "gpu_topk": False,
            "fixed_row_buffers": True,
        }
        summary_record = _accumulate_stats(record)
        if summary_record is not None:
            _write_stats(summary_record)

    if not use_sparse:
        dense_mask = torch.ones(
            total_num_scheduled_tokens,
            device=device,
            dtype=torch.bool,
        )
        return _make_plan(
            dense_mask,
            dense_count=total_num_scheduled_tokens,
            sparse_count=0,
            total_rows=total_num_scheduled_tokens,
            contiguous_dense_prefix=True,
        )

    empty_mask = torch.empty(0, device=device, dtype=torch.bool)
    return _make_plan(
        empty_mask,
        dense_count=0,
        sparse_count=total_num_scheduled_tokens,
        total_rows=total_num_scheduled_tokens,
        all_sparse=True,
    )


def _build_confidence_batch_plan(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    num_decode_draft_tokens: Any | None,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> TokenDensePlan | None:
    """Route a complete batch sparse only when normalized confidence is high."""

    normalized_confidences: list[torch.Tensor] = []
    eligible = True
    total_draft_tokens = 0
    scored_draft_tokens = 0
    with _lock:
        for req_idx, req_id in enumerate(req_ids):
            scheduled_rows = int(num_scheduled_tokens[req_idx])
            if scheduled_rows <= 0:
                continue
            draft_rows = int(num_draft_tokens[req_idx])
            total_draft_tokens += max(0, draft_rows)
            decode_rows = (
                int(num_decode_draft_tokens[req_idx])
                if num_decode_draft_tokens is not None
                else draft_rows
            )
            pending = _pending_scores.get(req_id)
            scores_raw = pending.popleft() if pending else None
            if (
                decode_rows < 0
                or draft_rows <= 0
                or scheduled_rows != draft_rows + 1
                or scores_raw is None
            ):
                eligible = False
                continue
            cumulative_log_scores = isinstance(
                scores_raw, _CumulativeLogScores
            )
            scores_value = (
                scores_raw.values if cumulative_log_scores else scores_raw
            )
            scores = _as_score_tensor(scores_value, device)
            valid_rows = min(draft_rows, int(scores.numel()))
            if valid_rows <= 0:
                eligible = False
                continue
            scored_draft_tokens += valid_rows
            if cumulative_log_scores:
                log_geometric_mean = scores[valid_rows - 1] / valid_rows
            else:
                log_geometric_mean = (
                    scores[:valid_rows].clamp_min(1e-30).log().mean()
                )
            normalized_confidences.append(log_geometric_mean.exp())

    batch_confidence = (
        float(torch.stack(normalized_confidences).mean().item())
        if normalized_confidences
        else None
    )
    confidence_allows_sparse = (
        eligible
        and batch_confidence is not None
        and batch_confidence >= batch_confidence_threshold()
    )
    global _batch_route_credit
    sparse_fraction = 1.0 - dense_token_ratio()
    with _lock:
        _batch_route_credit += sparse_fraction
        if confidence_allows_sparse and _batch_route_credit >= 1.0:
            use_sparse = True
            _batch_route_credit -= 1.0
        else:
            use_sparse = False
            if not confidence_allows_sparse:
                _batch_route_credit = min(_batch_route_credit, 1.0)
    if not production_fast_enabled():
        record = {
            "timestamp": time.time(),
            "event": "verify_token_mask",
            "mode": mode(),
            "dense_token_budget": 0,
            "effective_dense_token_budget": 0,
            "dense_budget_mode": dense_budget_mode(),
            "dense_token_ratio": dense_token_ratio(),
            "dense_min_per_request": dense_min_per_request(),
            "dense_token_cap": dense_token_cap(),
            "linear_strategy": linear_strategy(),
            "mlp_strategy": mlp_strategy(),
            "confidence_semantics": "batch_geometric_mean",
            "request_count": len(req_ids),
            "total_scheduled_tokens": total_num_scheduled_tokens,
            "total_draft_tokens": total_draft_tokens,
            "scored_draft_tokens": scored_draft_tokens,
            "scored_request_count": len(normalized_confidences),
            "forced_dense_tokens": 0 if use_sparse else total_num_scheduled_tokens,
            "dense_draft_tokens": 0 if use_sparse else total_draft_tokens,
            "sparse_draft_tokens": total_draft_tokens if use_sparse else 0,
            "missing_score_tokens": max(0, total_draft_tokens - scored_draft_tokens),
            "batch_route": "sparse" if use_sparse else "dense",
            "batch_confidence": batch_confidence,
            "gpu_topk": False,
            "fixed_row_buffers": True,
        }
        summary_record = _accumulate_stats(record)
        if summary_record is not None:
            _write_stats(summary_record)
    if not use_sparse:
        dense_mask = torch.ones(
            total_num_scheduled_tokens,
            device=device,
            dtype=torch.bool,
        )
        return _make_plan(
            dense_mask,
            dense_count=total_num_scheduled_tokens,
            sparse_count=0,
            total_rows=total_num_scheduled_tokens,
            contiguous_dense_prefix=True,
        )

    empty_mask = torch.empty(0, device=device, dtype=torch.bool)
    return _make_plan(
        empty_mask,
        dense_count=0,
        sparse_count=total_num_scheduled_tokens,
        total_rows=total_num_scheduled_tokens,
        all_sparse=True,
    )


def build_verify_dense_mask(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    num_decode_draft_tokens: Any | None = None,
    cu_num_scheduled_tokens: Any,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> TokenDensePlan | None:
    """Build a per-token dense mask for the current target verification pass.

    The final sampled position is the verifier bonus token. It remains dense by
    default and uses 2:4 when ``SPECLINK_TOKEN_DENSE_SPARSE_BONUS=1``. The first
    target-logit row participates in the scored budget, so a balanced
    per-request floor can protect it without charging it twice. For each
    request, draft position ``h`` has confidence
    ``product(selected_prob[0:h + 1])``. The selected candidate rows remain
    dense; the remaining scored draft rows use 2:4. ``balanced_confidence``
    first protects the configured leading rows of every active request, then
    spends the remaining budget globally on the highest-confidence rows.
    """
    if not enabled() or total_num_scheduled_tokens <= 0:
        return None
    routing_mode = mode()
    if routing_mode != ROUTING_MODE:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_MODE currently supports only " f"{ROUTING_MODE}"
        )

    strategy = linear_strategy()
    selection = dense_selection()
    prefill_mask: torch.Tensor | None = None
    prefill_ranges: list[tuple[int, int]] = []
    for req_idx in range(len(req_ids)):
        scheduled = int(num_scheduled_tokens[req_idx])
        decode_rows = (
            int(num_decode_draft_tokens[req_idx])
            if num_decode_draft_tokens is not None
            else int(num_draft_tokens[req_idx])
        )
        if scheduled <= 0 or decode_rows >= 0:
            continue
        if prefill_mask is None:
            prefill_mask = torch.zeros(
                total_num_scheduled_tokens,
                device=device,
                dtype=torch.bool,
            )
        end = int(cu_num_scheduled_tokens[req_idx])
        start = end - scheduled
        prefill_mask[start:end] = True
        prefill_ranges.append((start, end))
    has_prefill_rows = prefill_mask is not None
    contiguous_prefill_prefix = bool(
        prefill_ranges
        and prefill_ranges[0][0] == 0
        and all(
            left[1] == right[0]
            for left, right in zip(prefill_ranges, prefill_ranges[1:])
        )
    )
    contiguous_prefill_suffix = bool(
        prefill_ranges
        and prefill_ranges[-1][1] == total_num_scheduled_tokens
        and all(
            left[1] == right[0]
            for left, right in zip(prefill_ranges, prefill_ranges[1:])
        )
    )

    if strategy == "sparse_only_decode":
        dense_tokens = 0
        sparse_tokens = 0
        dense_ranges: list[tuple[int, int]] = []
        for req_idx, _req_id in enumerate(req_ids):
            sched = int(num_scheduled_tokens[req_idx])
            if sched <= 0:
                continue
            end = int(cu_num_scheduled_tokens[req_idx])
            start = end - sched
            decode_n = (
                int(num_decode_draft_tokens[req_idx])
                if num_decode_draft_tokens is not None
                else int(num_draft_tokens[req_idx])
            )
            if decode_n < 0:
                dense_ranges.append((start, end))
                dense_tokens += sched
            else:
                sparse_tokens += sched
        if not production_fast_enabled():
            budget = effective_dense_token_budget(0, 0)
            block_strategy = mlp_strategy()
            record = {
                "timestamp": time.time(),
                "event": "verify_token_mask",
                "mode": routing_mode,
                "dense_token_budget": budget,
                "effective_dense_token_budget": budget,
                "dense_budget_mode": dense_budget_mode(),
                "dense_token_ratio": dense_token_ratio(),
                "dense_min_per_request": dense_min_per_request(),
                "dense_token_cap": dense_token_cap(),
                "linear_strategy": strategy,
                "mlp_strategy": block_strategy,
                "confidence_semantics": "not_used_sparse_only",
                "request_count": len(req_ids),
                "total_scheduled_tokens": total_num_scheduled_tokens,
                "total_draft_tokens": sum(int(v) for v in num_draft_tokens),
                "scored_draft_tokens": 0,
                "scored_request_count": 0,
                "dense_scored_draft_tokens": 0,
                "sparse_scored_draft_tokens": sum(int(v) for v in num_draft_tokens),
                "forced_dense_tokens": dense_tokens,
                "dense_draft_tokens": 0,
                "sparse_draft_tokens": sparse_tokens,
                "missing_score_tokens": 0,
                "gpu_topk": False,
                "fixed_row_buffers": True,
                "sparse_only_fast_path": True,
                "dense_draft_fraction": 0.0,
                "sparse_draft_fraction": (
                    sparse_tokens / total_num_scheduled_tokens
                    if total_num_scheduled_tokens
                    else None
                ),
            }
            summary_record = _accumulate_stats(record)
            if summary_record is not None:
                _write_stats(summary_record)
        if dense_tokens == 0:
            empty_mask = torch.empty(0, device=device, dtype=torch.bool)
            return _make_plan(
                empty_mask,
                dense_count=0,
                sparse_count=total_num_scheduled_tokens,
                total_rows=total_num_scheduled_tokens,
                all_sparse=True,
            )
        dense_mask = torch.zeros(
            total_num_scheduled_tokens,
            device=device,
            dtype=torch.bool,
        )
        for start, end in dense_ranges:
            dense_mask[start:end] = True
        return _make_plan(
            dense_mask,
            dense_count=dense_tokens,
            sparse_count=total_num_scheduled_tokens - dense_tokens,
            total_rows=total_num_scheduled_tokens,
            has_prefill_rows=has_prefill_rows,
            prefill_mask=prefill_mask,
            contiguous_prefill_prefix=contiguous_prefill_prefix,
            contiguous_prefill_suffix=contiguous_prefill_suffix,
        )

    if selection == "request_contiguous":
        return _build_contiguous_request_plan(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_draft_tokens=num_draft_tokens,
            num_decode_draft_tokens=num_decode_draft_tokens,
            cu_num_scheduled_tokens=cu_num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            device=device,
        )

    if selection == "batch_adaptive":
        return _build_adaptive_batch_plan(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_draft_tokens=num_draft_tokens,
            num_decode_draft_tokens=num_decode_draft_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            device=device,
        )

    if selection == "batch_alternating":
        return _build_alternating_batch_plan(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_draft_tokens=num_draft_tokens,
            num_decode_draft_tokens=num_decode_draft_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            device=device,
        )

    if selection == "batch_confidence":
        return _build_confidence_batch_plan(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_draft_tokens=num_draft_tokens,
            num_decode_draft_tokens=num_decode_draft_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            device=device,
        )

    block_strategy = mlp_strategy()
    use_fast_plan = fast_plan_enabled()
    plan_buffers = (
        _get_plan_buffers(device, total_num_scheduled_tokens)
        if use_fast_plan
        else None
    )
    if plan_buffers is not None:
        dense_mask = plan_buffers["dense_mask"][:total_num_scheduled_tokens]
        dense_mask.fill_(True)
        row_ids = plan_buffers["row_ids_long"]
    else:
        dense_mask = torch.ones(
            total_num_scheduled_tokens,
            device=device,
            dtype=torch.bool,
        )
        row_ids = None
    candidate_scores: list[torch.Tensor] = []
    candidate_rows: list[torch.Tensor] = []
    candidate_req_indices: list[int] = []
    dense_draft_tokens = 0
    sparse_draft_tokens = 0
    sparse_bonus_tokens = 0
    sparse_unscored_decode_tokens = 0
    missing_score_tokens = 0
    total_draft_tokens = 0
    scored_draft_tokens = 0
    forced_dense_tokens = 0
    include_request_summaries = stats_detail_enabled()
    request_summaries: list[dict[str, Any]] = []

    with _lock:
        for req_idx, req_id in enumerate(req_ids):
            n = int(num_draft_tokens[req_idx])
            decode_n = (
                int(num_decode_draft_tokens[req_idx])
                if num_decode_draft_tokens is not None
                else n
            )
            end = int(cu_num_scheduled_tokens[req_idx])
            sched = int(num_scheduled_tokens[req_idx])
            start = end - sched
            pending = _pending_scores.get(req_id)
            if decode_n < 0:
                if n > 0 and pending:
                    pending.popleft()
                forced_dense_tokens += n
                total_draft_tokens += n
                if include_request_summaries:
                    request_summaries.append(
                        {
                            "_req_idx": req_idx,
                            "_forced_dense_tokens": n,
                            "_scored_draft_tokens": 0,
                            "request_id": req_id,
                            "draft_tokens": n,
                            "dense_draft_tokens": n,
                            "sparse_draft_tokens": 0,
                            "score_count": 0,
                            "forced_dense": True,
                            "confidence_semantics": CONFIDENCE_SEMANTICS,
                            "final_prefix_confidence": None,
                        }
                    )
                continue
            if n <= 0:
                if sparse_unscored_decode_enabled() and sched > 0:
                    dense_mask[start:end] = False
                    sparse_unscored_decode_tokens += sched
                continue
            total_draft_tokens += n
            scores_raw = (
                pending.popleft()
                if selection != "balanced_prefix" and pending
                else None
            )
            cumulative_log_scores = isinstance(scores_raw, _CumulativeLogScores)
            scores_value = (
                scores_raw.values if cumulative_log_scores else scores_raw
            )
            scores = (
                _as_score_tensor(scores_value, device)
                if scores_raw is not None
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            sparse_bonus = sparse_bonus_enabled()
            per_req_dense = 0 if sparse_bonus else 1
            per_req_sparse = 0
            if sparse_bonus:
                dense_mask[end - 1] = False
                sparse_bonus_tokens += 1
            else:
                forced_dense_tokens += 1
            # SpecDecodeMetadata samples the final n + 1 scheduled rows. The
            # final row is the verifier bonus; the preceding n rows are scored.
            first_row = end - n
            candidate_start = max(start, first_row - 1)
            eligible_rows = max(0, n)
            valid_rows = max(
                0,
                min(eligible_rows, end - 1 - candidate_start),
            )
            available_scores = max(0, int(scores.numel()))
            valid_scores = (
                valid_rows
                if selection == "balanced_prefix"
                else min(valid_rows, available_scores)
            )
            missing = eligible_rows - valid_scores
            if missing > 0:
                missing_score_tokens += missing
                sparse_missing = 0
                if sparse_unscored_decode_enabled():
                    missing_start = candidate_start + valid_scores
                    missing_end = min(
                        end - 1,
                        candidate_start + eligible_rows,
                    )
                    sparse_missing = max(0, missing_end - missing_start)
                    if sparse_missing:
                        dense_mask[missing_start:missing_end] = False
                        sparse_unscored_decode_tokens += sparse_missing
                forced_missing = missing - sparse_missing
                forced_dense_tokens += forced_missing
                per_req_dense += forced_missing
            prefix_confidence_value: float | None = None
            if valid_scores > 0:
                if selection == "balanced_prefix":
                    prefix_ranking_score = torch.empty(
                        0, device=device, dtype=torch.float32
                    )
                elif cumulative_log_scores:
                    prefix_ranking_score = scores[:valid_scores]
                else:
                    score_vec = scores[:valid_scores].clamp_min(0.0)
                    prefix_ranking_score = score_vec.cumprod(dim=0)
                if row_ids is not None:
                    rows = row_ids[
                        candidate_start : candidate_start + valid_scores
                    ]
                else:
                    rows = torch.arange(
                        candidate_start,
                        candidate_start + valid_scores,
                        device=device,
                        dtype=torch.int32,
                    )
                if selection != "balanced_prefix":
                    candidate_scores.append(prefix_ranking_score)
                candidate_rows.append(rows)
                candidate_req_indices.append(req_idx)
                scored_draft_tokens += valid_scores
                per_req_sparse += valid_scores
                if include_request_summaries and selection != "balanced_prefix":
                    final_score = prefix_ranking_score[-1]
                    if cumulative_log_scores:
                        final_score = final_score.exp()
                    prefix_confidence_value = float(final_score.item())
            if include_request_summaries:
                request_summaries.append(
                    {
                        "_req_idx": req_idx,
                        "_forced_dense_tokens": per_req_dense,
                        "_scored_draft_tokens": per_req_sparse,
                        "request_id": req_id,
                        "draft_tokens": n,
                        "dense_draft_tokens": per_req_dense,
                        "sparse_draft_tokens": per_req_sparse,
                        "score_count": int(scores.numel()),
                        "forced_dense": False,
                        "confidence_semantics": CONFIDENCE_SEMANTICS,
                        "final_prefix_confidence": prefix_confidence_value,
                    }
                )

    scored_request_count = len(candidate_rows)
    budget = effective_dense_token_budget(
        scored_draft_tokens,
        scored_request_count,
    )
    dense_scored_count = 0
    scored_by_req: defaultdict[int, int] = defaultdict(int)
    dense_scored_by_req: defaultdict[int, int] = defaultdict(int)
    if scored_draft_tokens > 0:
        rows_all = torch.cat(candidate_rows)
        dense_mask[
            rows_all if rows_all.dtype == torch.long else rows_all.to(dtype=torch.long)
        ] = False
        dense_scored_count = min(budget, scored_draft_tokens)
        if dense_scored_count > 0:
            if selection == "balanced_prefix":
                offsets: list[int] = []
                offset = 0
                max_candidates = 0
                for rows in candidate_rows:
                    offsets.append(offset)
                    count = int(rows.numel())
                    offset += count
                    max_candidates = max(max_candidates, count)
                balanced_positions: list[int] = []
                start_position = min(
                    balanced_start_position(),
                    max(0, max_candidates - 1),
                )
                position_order = [
                    *range(start_position, max_candidates),
                    *range(0, start_position),
                ]
                for position in position_order:
                    for offset, rows in zip(offsets, candidate_rows):
                        if position < int(rows.numel()):
                            balanced_positions.append(offset + position)
                            if len(balanced_positions) == dense_scored_count:
                                break
                    if len(balanced_positions) == dense_scored_count:
                        break
                dense_positions = torch.tensor(
                    balanced_positions,
                    device=device,
                    dtype=torch.long,
                )
            elif selection in {
                "balanced_confidence",
                "balanced_low_confidence",
            }:
                offsets: list[int] = []
                offset = 0
                max_candidates = 0
                for rows in candidate_rows:
                    offsets.append(offset)
                    count = int(rows.numel())
                    offset += count
                    max_candidates = max(max_candidates, count)

                protected_positions: list[int] = []
                protected_prefix = max(1, dense_min_per_request())
                for position in range(min(protected_prefix, max_candidates)):
                    for row_offset, rows in zip(offsets, candidate_rows):
                        if position < int(rows.numel()):
                            protected_positions.append(row_offset + position)
                            if len(protected_positions) == dense_scored_count:
                                break
                    if len(protected_positions) == dense_scored_count:
                        break

                dense_positions = torch.tensor(
                    protected_positions,
                    device=device,
                    dtype=torch.long,
                )
                remaining_budget = dense_scored_count - len(protected_positions)
                if remaining_budget > 0:
                    if selection == "balanced_confidence":
                        scores_all = torch.cat(candidate_scores)
                        eligible = torch.ones(
                            scored_draft_tokens,
                            device=device,
                            dtype=torch.bool,
                        )
                        if dense_positions.numel():
                            eligible[dense_positions] = False
                        eligible_positions = torch.nonzero(
                            eligible,
                            as_tuple=False,
                        ).flatten()
                        eligible_scores = scores_all.index_select(
                            0,
                            eligible_positions,
                        )
                        _values, ranked_indices = torch.topk(
                            eligible_scores,
                            k=remaining_budget,
                            largest=True,
                            sorted=False,
                        )
                        ranked_positions = eligible_positions.index_select(
                            0,
                            ranked_indices,
                        )
                        dense_positions = torch.cat(
                            (dense_positions, ranked_positions)
                        )
                    else:
                        ranked_chunks: list[torch.Tensor] = []
                        position = protected_prefix
                        while remaining_budget > 0:
                            frontier_positions: list[int] = []
                            frontier_scores: list[torch.Tensor] = []
                            for row_offset, rows, scores in zip(
                                offsets,
                                candidate_rows,
                                candidate_scores,
                            ):
                                if position < int(rows.numel()):
                                    frontier_positions.append(
                                        row_offset + position
                                    )
                                    frontier_scores.append(scores[position])
                            if not frontier_positions:
                                break
                            positions_tensor = torch.tensor(
                                frontier_positions,
                                device=device,
                                dtype=torch.long,
                            )
                            scores_tensor = torch.stack(frontier_scores)
                            take = min(
                                remaining_budget,
                                len(frontier_positions),
                            )
                            _values, ranked_indices = torch.topk(
                                scores_tensor,
                                k=take,
                                largest=False,
                                sorted=False,
                            )
                            ranked_chunks.append(
                                positions_tensor.index_select(
                                    0,
                                    ranked_indices,
                                )
                            )
                            remaining_budget -= take
                            if take < len(frontier_positions):
                                break
                            position += 1
                        if remaining_budget:
                            raise RuntimeError(
                                "balanced low-confidence routing could not "
                                "spend its remaining dense budget: "
                                f"{remaining_budget}"
                            )
                        dense_positions = torch.cat(
                            (dense_positions, *ranked_chunks)
                        )
            elif selection in {"request_highest", "request_lowest"}:
                offsets: list[int] = []
                counts: list[int] = []
                offset = 0
                for rows in candidate_rows:
                    offsets.append(offset)
                    count = int(rows.numel())
                    counts.append(count)
                    offset += count
                offsets_tensor = torch.tensor(
                    offsets,
                    device=device,
                    dtype=torch.long,
                )
                counts_tensor = torch.tensor(
                    counts,
                    device=device,
                    dtype=torch.long,
                )
                first_scores = torch.stack([scores[0] for scores in candidate_scores])
                request_order = torch.argsort(
                    first_scores,
                    descending=selection == "request_highest",
                )
                ranked_positions: list[torch.Tensor] = []
                max_candidates = max(counts, default=0)
                ordered_offsets = offsets_tensor.index_select(0, request_order)
                ordered_counts = counts_tensor.index_select(0, request_order)
                for chunk_start in range(0, max_candidates, 2):
                    positions = torch.arange(
                        chunk_start,
                        min(chunk_start + 2, max_candidates),
                        device=device,
                        dtype=torch.long,
                    )
                    position_grid = ordered_offsets[:, None] + positions[None, :]
                    valid = ordered_counts[:, None] > positions[None, :]
                    ranked_positions.append(position_grid[valid])
                dense_positions = torch.cat(ranked_positions)[:dense_scored_count]
            else:
                scores_all = torch.cat(candidate_scores)
                _values, dense_positions = torch.topk(
                    scores_all,
                    k=dense_scored_count,
                    largest=selection == "highest",
                    sorted=False,
                )
            selected_rows = rows_all.index_select(0, dense_positions)
            dense_mask[
                selected_rows
                if selected_rows.dtype == torch.long
                else selected_rows.to(dtype=torch.long)
            ] = True
        dense_draft_tokens += dense_scored_count
        sparse_draft_tokens = scored_draft_tokens - dense_scored_count

        if include_request_summaries:
            offset = 0
            selected_mask = torch.zeros(
                scored_draft_tokens, device=device, dtype=torch.bool
            )
            if dense_scored_count > 0:
                selected_mask[dense_positions] = True
            for req_idx, rows in zip(candidate_req_indices, candidate_rows):
                count = int(rows.numel())
                if count == 0:
                    continue
                scored_by_req[req_idx] += count
                dense_count = int(selected_mask[offset : offset + count].sum().item())
                dense_scored_by_req[req_idx] += dense_count
                offset += count

    dense_draft_tokens += forced_dense_tokens
    if include_request_summaries:
        for summary in request_summaries:
            req_idx = summary.pop("_req_idx", None)
            forced = int(summary.pop("_forced_dense_tokens", 0) or 0)
            scored = int(summary.pop("_scored_draft_tokens", 0) or 0)
            if req_idx is None:
                continue
            dense_scored = int(dense_scored_by_req.get(int(req_idx), 0))
            sparse_scored = int(scored_by_req.get(int(req_idx), scored) - dense_scored)
            summary["dense_scored_draft_tokens"] = dense_scored
            summary["sparse_scored_draft_tokens"] = sparse_scored
            summary["dense_draft_tokens"] = forced + dense_scored
            summary["sparse_draft_tokens"] = sparse_scored
    record = {
        "timestamp": time.time(),
        "event": "verify_token_mask",
        "mode": routing_mode,
        "dense_token_budget": budget,
        "effective_dense_token_budget": budget,
        "dense_budget_mode": dense_budget_mode(),
        "dense_token_ratio": dense_token_ratio(),
        "dense_min_per_request": dense_min_per_request(),
        "dense_token_cap": dense_token_cap(),
        "dense_selection": selection,
        "balanced_start_position": balanced_start_position(),
        "linear_strategy": strategy,
        "mlp_strategy": block_strategy,
        "score_backend": score_backend(),
        "fast_plan": use_fast_plan,
        "confidence_semantics": (
            "fixed_request_prefix"
            if selection == "balanced_prefix"
            else (
                "request_prefix_then_reach_probability"
                if selection == "balanced_confidence"
                else (
                    "request_prefix_then_low_confidence"
                    if selection == "balanced_low_confidence"
                    else CONFIDENCE_SEMANTICS
                )
            )
        ),
        "request_count": len(req_ids),
        "total_scheduled_tokens": total_num_scheduled_tokens,
        "total_draft_tokens": total_draft_tokens,
        "scored_draft_tokens": scored_draft_tokens,
        "scored_request_count": scored_request_count,
        "dense_scored_draft_tokens": dense_scored_count,
        "sparse_scored_draft_tokens": sparse_draft_tokens,
        "forced_dense_tokens": forced_dense_tokens,
        "dense_draft_tokens": dense_draft_tokens,
        "sparse_draft_tokens": sparse_draft_tokens,
        "sparse_bonus_tokens": sparse_bonus_tokens,
        "sparse_bonus": sparse_bonus_enabled(),
        "sparse_unscored_decode_tokens": sparse_unscored_decode_tokens,
        "sparse_unscored_decode": sparse_unscored_decode_enabled(),
        "missing_score_tokens": missing_score_tokens,
        "gpu_topk": selection != "balanced_prefix",
        "fixed_row_buffers": use_fast_plan,
        "dense_draft_fraction": (
            dense_draft_tokens / total_draft_tokens if total_draft_tokens else None
        ),
        "sparse_draft_fraction": (
            sparse_draft_tokens / total_draft_tokens if total_draft_tokens else None
        ),
    }
    if include_request_summaries:
        record["requests"] = request_summaries
        _write_stats(record)
    else:
        summary_record = _accumulate_stats(record)
        if summary_record is not None:
            _write_stats(summary_record)
    total_sparse_tokens = (
        sparse_draft_tokens
        + sparse_bonus_tokens
        + sparse_unscored_decode_tokens
    )
    if total_sparse_tokens == 0:
        return None
    dense_count = total_num_scheduled_tokens - total_sparse_tokens
    return _make_plan(
        dense_mask,
        dense_count=dense_count,
        sparse_count=total_sparse_tokens,
        total_rows=total_num_scheduled_tokens,
        has_prefill_rows=has_prefill_rows,
        prefill_mask=prefill_mask,
        contiguous_prefill_prefix=contiguous_prefill_prefix,
        contiguous_prefill_suffix=contiguous_prefill_suffix,
    )
