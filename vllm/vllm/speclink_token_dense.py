# SPDX-License-Identifier: Apache-2.0
"""Token-level dense/sparse routing for SpecLink structured 2:4 studies.

The normal structured-2:4 path edits target-model weights in-place at load
time. This module implements the experimental alternative used for the
accuracy-debug pass: keep the original dense weights, attach 2:4 masks to target
linear modules, and at verification time run high-confidence draft-token rows
dense while low-confidence rows use the masked activation-aware weight.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)

_propose_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "speclink_token_dense_propose_context", default=None
)
_verify_dense_mask: ContextVar[torch.Tensor | None] = ContextVar(
    "speclink_token_dense_verify_mask", default=None
)

_pending_scores: defaultdict[str, deque[list[float]]] = defaultdict(deque)
_lock = threading.Lock()
_stats_accum: dict[str, Any] = {
    "steps": 0,
    "total_scheduled_tokens": 0,
    "total_draft_tokens": 0,
    "dense_draft_tokens": 0,
    "sparse_draft_tokens": 0,
    "missing_score_tokens": 0,
    "last_flush_steps": 0,
}
_STATIC_ENABLED = os.getenv("SPECLINK_TOKEN_DENSE_ENABLE", "0") in _TRUTHY


def enabled() -> bool:
    return _STATIC_ENABLED


def threshold() -> float:
    return float(os.getenv("SPECLINK_TOKEN_DENSE_THRESHOLD", "0.7"))


def mode() -> str:
    return os.getenv("SPECLINK_TOKEN_DENSE_MODE", "high_confidence_dense").strip()


def stats_path() -> Path | None:
    value = os.getenv("SPECLINK_TOKEN_DENSE_STATS_PATH", "").strip()
    return Path(value) if value else None


def stats_detail_enabled() -> bool:
    return os.getenv("SPECLINK_TOKEN_DENSE_STATS_DETAIL", "0") in _TRUTHY


def stats_interval() -> int:
    try:
        return max(0, int(os.getenv("SPECLINK_TOKEN_DENSE_STATS_INTERVAL", "1000")))
    except ValueError:
        return 1000


def _accumulate_stats(record: dict[str, Any]) -> dict[str, Any] | None:
    with _lock:
        _stats_accum["steps"] += 1
        _stats_accum["total_scheduled_tokens"] += int(
            record.get("total_scheduled_tokens") or 0
        )
        _stats_accum["total_draft_tokens"] += int(record.get("total_draft_tokens") or 0)
        _stats_accum["dense_draft_tokens"] += int(record.get("dense_draft_tokens") or 0)
        _stats_accum["sparse_draft_tokens"] += int(
            record.get("sparse_draft_tokens") or 0
        )
        _stats_accum["missing_score_tokens"] += int(
            record.get("missing_score_tokens") or 0
        )
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
            "threshold": record.get("threshold"),
            "steps": steps,
            "total_scheduled_tokens": int(_stats_accum["total_scheduled_tokens"]),
            "total_draft_tokens": total_draft,
            "dense_draft_tokens": dense_draft,
            "sparse_draft_tokens": sparse_draft,
            "missing_score_tokens": int(_stats_accum["missing_score_tokens"]),
            "dense_draft_fraction": dense_draft / total_draft if total_draft else None,
            "sparse_draft_fraction": sparse_draft / total_draft if total_draft else None,
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


def begin_verify_context(dense_mask: torch.Tensor | None) -> Any:
    if not enabled() or dense_mask is None:
        return None
    return _verify_dense_mask.set(dense_mask)


def end_verify_context(token: Any) -> None:
    if token is not None:
        _verify_dense_mask.reset(token)


@torch.inference_mode()
def record_draft_scores(
    *,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    temperature: torch.Tensor | None = None,
    method: str = "",
) -> None:
    """Record selected-token probabilities for the next verifier step."""
    if not enabled():
        return
    ctx = _propose_context.get()
    if ctx is None or not logits_by_position:
        return

    batch_size = min(int(draft_token_ids.shape[0]), len(ctx["req_ids"]))
    num_spec_tokens = min(int(draft_token_ids.shape[1]), len(logits_by_position))
    if batch_size <= 0 or num_spec_tokens <= 0:
        return

    draft_token_ids = draft_token_ids[:batch_size, :num_spec_tokens]
    per_req_scores = [[1.0] * num_spec_tokens for _ in range(batch_size)]
    for pos, logits in enumerate(logits_by_position[:num_spec_tokens]):
        logits = logits[:batch_size].detach().float()
        selected = draft_token_ids[:batch_size, pos].to(
            device=logits.device, dtype=torch.long
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_logprob = log_probs.gather(1, selected.view(-1, 1)).squeeze(1)
        selected_prob = selected_logprob.exp().detach().cpu().tolist()
        for req_idx, score in enumerate(selected_prob):
            per_req_scores[req_idx][pos] = float(score)

    req_ids = ctx["req_ids"]
    with _lock:
        for req_idx in range(batch_size):
            _pending_scores[req_ids[req_idx]].append(per_req_scores[req_idx])


def _write_stats(record: dict[str, Any]) -> None:
    path = stats_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_verify_dense_mask(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    cu_num_scheduled_tokens: Any,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a per-token dense mask for the current target verification pass.

    The first sampled token per request is the already-known next token and the
    last sampled position is the verifier bonus token, so only scheduled draft
    rows ``start + 1 ... start + K`` are eligible for sparse routing.
    """
    if not enabled() or total_num_scheduled_tokens <= 0:
        return None
    routing_mode = mode()
    if routing_mode != "high_confidence_dense":
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_MODE currently supports only "
            "high_confidence_dense"
        )

    dense_mask_cpu = torch.ones(total_num_scheduled_tokens, dtype=torch.bool)
    cutoff = threshold()
    dense_draft_tokens = 0
    sparse_draft_tokens = 0
    missing_score_tokens = 0
    total_draft_tokens = 0
    include_request_summaries = stats_detail_enabled()
    request_summaries: list[dict[str, Any]] = []

    with _lock:
        for req_idx, req_id in enumerate(req_ids):
            n = int(num_draft_tokens[req_idx])
            if n <= 0:
                continue
            total_draft_tokens += n
            end = int(cu_num_scheduled_tokens[req_idx])
            sched = int(num_scheduled_tokens[req_idx])
            start = end - sched
            pending = _pending_scores.get(req_id)
            scores = pending.popleft() if pending else []
            per_req_dense = 0
            per_req_sparse = 0
            for pos in range(n):
                row = start + 1 + pos
                if row >= total_num_scheduled_tokens:
                    missing_score_tokens += 1
                    dense_draft_tokens += 1
                    per_req_dense += 1
                    continue
                score = float(scores[pos]) if pos < len(scores) else None
                if score is None:
                    missing_score_tokens += 1
                    use_dense = True
                else:
                    use_dense = score >= cutoff
                dense_mask_cpu[row] = use_dense
                if use_dense:
                    dense_draft_tokens += 1
                    per_req_dense += 1
                else:
                    sparse_draft_tokens += 1
                    per_req_sparse += 1
            if include_request_summaries:
                request_summaries.append(
                    {
                        "request_id": req_id,
                        "draft_tokens": n,
                        "dense_draft_tokens": per_req_dense,
                        "sparse_draft_tokens": per_req_sparse,
                        "score_count": len(scores),
                    }
                )

    record = {
        "timestamp": time.time(),
        "event": "verify_token_mask",
        "mode": routing_mode,
        "threshold": cutoff,
        "request_count": len(req_ids),
        "total_scheduled_tokens": total_num_scheduled_tokens,
        "total_draft_tokens": total_draft_tokens,
        "dense_draft_tokens": dense_draft_tokens,
        "sparse_draft_tokens": sparse_draft_tokens,
        "missing_score_tokens": missing_score_tokens,
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
    if sparse_draft_tokens == 0:
        return None
    return dense_mask_cpu.to(device=device, non_blocking=True)


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
            f"SpecLink token-dense mask shape {tuple(mask_bytes.shape)} does "
            f"not match {(out_features, groups)} or {packed_expected}"
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


@torch.inference_mode()
def mixed_sparse_linear_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_output: torch.Tensor,
) -> torch.Tensor:
    """Replace low-confidence token rows with the attached 2:4 linear output."""
    if not _STATIC_ENABLED:
        return dense_output
    dense_mask = _verify_dense_mask.get()
    mask_bytes = getattr(module, "_speclink_24_mask_bytes", None)
    if dense_mask is None or mask_bytes is None:
        return dense_output
    if input_tensor.ndim != 2 or dense_output.ndim != 2:
        raise RuntimeError("SpecLink token-dense Llama path expects 2D tensors")

    rows = int(input_tensor.shape[0])
    if rows <= 0:
        return dense_output
    if dense_mask.numel() < rows:
        raise RuntimeError(
            f"SpecLink token-dense mask has {dense_mask.numel()} rows, "
            f"but linear input has {rows}"
        )
    row_is_dense = dense_mask[:rows].to(device=input_tensor.device)
    if bool(row_is_dense.all().item()):
        return dense_output
    sparse_rows = (~row_is_dense).nonzero(as_tuple=False).squeeze(1)
    if sparse_rows.numel() == 0:
        return dense_output

    weight = module.weight
    bias = getattr(module, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        bias = None
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    if usable_in <= 0:
        return dense_output

    group_mask = _expand_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=groups,
        device=weight.device,
    )
    bits = _MASK_BITS.to(device=weight.device)
    keep = (group_mask.unsqueeze(-1) & bits.view(1, 1, 4)).ne(0)
    sparse_weight = weight[:, :usable_in] * keep.view(
        out_features, usable_in
    ).to(dtype=weight.dtype)
    row_scale = getattr(module, "_speclink_24_row_scale", None)
    if row_scale is not None:
        scale = row_scale.to(device=weight.device, dtype=weight.dtype, non_blocking=True)
        sparse_weight = sparse_weight * scale.view(-1, 1)
    if usable_in < in_features:
        sparse_weight = torch.cat([sparse_weight, weight[:, usable_in:]], dim=1)

    sparse_input = input_tensor.index_select(0, sparse_rows)
    sparse_output = F.linear(sparse_input, sparse_weight, bias)
    output = dense_output.clone()
    output.index_copy_(0, sparse_rows, sparse_output)
    return output
