# SPDX-License-Identifier: Apache-2.0
"""SpecLink-only confidence trace helpers.

This module is intentionally inactive unless SPECLINK_TRACE_CONFIDENCE=1.
It records draft-model confidence features at proposal time and writes the
records only after the corresponding verifier/rejection-sampler labels are
available.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_propose_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "speclink_trace_propose_context", default=None
)
_verify_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "speclink_trace_verify_context", default=None
)

_pending: defaultdict[str, deque[list[dict[str, Any]]]] = defaultdict(deque)
_step_ids: defaultdict[str, int] = defaultdict(int)
_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("SPECLINK_TRACE_CONFIDENCE", "0") in _TRUTHY


def _method(default: str = "") -> str:
    return os.getenv("SPECLINK_TRACE_METHOD") or default or "unknown"


def _run_id() -> str:
    return os.getenv("SPECLINK_TRACE_RUN_ID", "unknown")


def _model_label() -> str:
    return os.getenv("SPECLINK_TRACE_MODEL_LABEL", "unknown")


def _dataset_label() -> str:
    return os.getenv("SPECLINK_TRACE_DATASET_LABEL", "unknown")


def _output_path() -> Path | None:
    value = os.getenv("SPECLINK_TRACE_OUTPUT")
    return Path(value) if value else None


def _parse_prompt_id(request_id: str) -> int | None:
    match = re.search(r"(?:^|[-_])p(\d+)(?:$|[-_])", request_id)
    if match:
        return int(match.group(1))
    return None


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
            "method": _method(method),
        }
    )


def begin_verify_context(
    *,
    req_ids: list[str],
    active_requests: int,
    batch_size: int,
    num_spec_tokens: int,
    method: str = "",
) -> Any:
    if not enabled():
        return None
    return _verify_context.set(
        {
            "req_ids": req_ids,
            "active_requests": active_requests,
            "batch_size": batch_size,
            "num_spec_tokens": num_spec_tokens,
            "method": _method(method),
        }
    )


def end_propose_context(token: Any) -> None:
    if token is not None:
        _propose_context.reset(token)


def end_verify_context(token: Any) -> None:
    if token is not None:
        _verify_context.reset(token)


def _temperature_values(
    temperature: torch.Tensor | None,
    batch_size: int,
) -> list[float | None]:
    if temperature is None:
        return [None] * batch_size
    values = temperature[:batch_size].detach().float().cpu().tolist()
    return [float(v) for v in values]


@torch.inference_mode()
def record_draft_features(
    *,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    temperature: torch.Tensor | None = None,
    method: str = "",
) -> None:
    """Store draft-token confidence features until labels are available."""
    if not enabled():
        return
    ctx = _propose_context.get()
    if ctx is None or not logits_by_position:
        return

    batch_size = min(int(draft_token_ids.shape[0]), len(ctx["req_ids"]))
    num_spec_tokens = int(draft_token_ids.shape[1])
    draft_token_ids = draft_token_ids[:batch_size, :num_spec_tokens]
    token_ids_cpu = draft_token_ids.detach().cpu().tolist()
    temperatures = _temperature_values(temperature, batch_size)

    per_position: list[dict[str, list[Any]]] = []
    for pos, logits in enumerate(logits_by_position[:num_spec_tokens]):
        logits = logits[:batch_size].detach().float()
        selected = draft_token_ids[:batch_size, pos].to(
            device=logits.device, dtype=torch.long
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(log_probs, k=2, dim=-1)
        selected_logprob = log_probs.gather(1, selected.view(-1, 1)).squeeze(1)
        selected_prob = selected_logprob.exp()
        probs = log_probs.exp()
        entropy_terms = torch.where(
            probs > 0,
            probs * log_probs,
            torch.zeros_like(log_probs),
        )
        entropy = -entropy_terms.sum(dim=-1)
        rank = torch.ones_like(selected, dtype=torch.int64)
        not_top1 = selected != top_indices[:, 0]
        if bool(not_top1.any().item()):
            rank = (log_probs > selected_logprob.unsqueeze(1)).sum(dim=-1) + 1
        per_position.append(
            {
                "selected_logprob": selected_logprob.cpu().tolist(),
                "selected_prob": selected_prob.cpu().tolist(),
                "top1_logprob": top_values[:, 0].cpu().tolist(),
                "top2_logprob": top_values[:, 1].cpu().tolist(),
                "margin_logprob": (top_values[:, 0] - top_values[:, 1])
                .cpu()
                .tolist(),
                "entropy": entropy.cpu().tolist(),
                "rank": rank.cpu().tolist(),
            }
        )

    now = time.time()
    req_ids = ctx["req_ids"]
    prompt_lens = ctx["prompt_lens"]
    generated_lens = ctx["generated_lens"]
    active_requests = int(ctx["active_requests"])
    trace_method = _method(method or ctx.get("method", ""))

    with _lock:
        for req_idx in range(batch_size):
            req_id = req_ids[req_idx]
            _step_ids[req_id] += 1
            step_id = _step_ids[req_id]
            prompt_id = _parse_prompt_id(req_id)
            context_len = int(prompt_lens[req_idx] + generated_lens[req_idx])
            generated_len = int(generated_lens[req_idx])
            records: list[dict[str, Any]] = []
            for pos in range(num_spec_tokens):
                pos_features = per_position[pos]
                records.append(
                    {
                        "run_id": _run_id(),
                        "dataset_label": _dataset_label(),
                        "model_label": _model_label(),
                        "method": trace_method,
                        "request_id": req_id,
                        "sequence_id": req_id,
                        "step_id": step_id,
                        "draft_position": pos + 1,
                        "num_spec_tokens": num_spec_tokens,
                        "token_id": int(token_ids_cpu[req_idx][pos]),
                        "token_text": None,
                        "context_len": context_len,
                        "generated_len_so_far": generated_len,
                        "prompt_id": prompt_id,
                        "dataset_index": prompt_id,
                        "draft_selected_logprob": float(
                            pos_features["selected_logprob"][req_idx]
                        ),
                        "draft_selected_prob": float(
                            pos_features["selected_prob"][req_idx]
                        ),
                        "draft_top1_logprob": float(
                            pos_features["top1_logprob"][req_idx]
                        ),
                        "draft_top2_logprob": float(
                            pos_features["top2_logprob"][req_idx]
                        ),
                        "draft_margin_logprob": float(
                            pos_features["margin_logprob"][req_idx]
                        ),
                        "draft_entropy": float(pos_features["entropy"][req_idx]),
                        "draft_rank_of_selected": int(pos_features["rank"][req_idx]),
                        "draft_temperature": temperatures[req_idx],
                        "batch_size": batch_size,
                        "active_requests": active_requests,
                        "timestamp": now,
                    }
                )
            _pending[req_id].append(records)


def _write_records(records: list[dict[str, Any]]) -> None:
    path = _output_path()
    if path is None or not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


@torch.inference_mode()
def label_verified_tokens(
    *,
    metadata: Any,
    output_token_ids: torch.Tensor,
    target_logits: torch.Tensor | None = None,
    method: str = "",
) -> None:
    """Attach verifier labels to the oldest pending draft records."""
    if not enabled():
        return
    ctx = _verify_context.get()
    if ctx is None:
        return

    req_ids = ctx["req_ids"]
    batch_size = min(len(req_ids), int(output_token_ids.shape[0]))
    num_draft_tokens = list(metadata.num_draft_tokens)
    output_cpu = output_token_ids[:batch_size].detach().cpu()
    scheduled_token_ids = metadata.draft_token_ids.detach().cpu().tolist()

    target_features: dict[str, list[Any]] = {}
    if target_logits is not None and metadata.draft_token_ids.numel() > 0:
        logits = target_logits.detach().float()
        draft_ids = metadata.draft_token_ids.to(device=logits.device, dtype=torch.long)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, draft_ids.view(-1, 1)).squeeze(1)
        top_values, top_indices = torch.topk(log_probs, k=2, dim=-1)
        rank = torch.ones_like(draft_ids, dtype=torch.int64)
        not_top1 = draft_ids != top_indices[:, 0]
        if bool(not_top1.any().item()):
            rank = (log_probs > selected.unsqueeze(1)).sum(dim=-1) + 1
        target_features = {
            "target_selected_logprob": selected.cpu().tolist(),
            "target_selected_prob": selected.exp().cpu().tolist(),
            "target_top1_token_id": top_indices[:, 0].cpu().tolist(),
            "target_top1_logprob": top_values[:, 0].cpu().tolist(),
            "target_top2_logprob": top_values[:, 1].cpu().tolist(),
            "target_margin_logprob": (top_values[:, 0] - top_values[:, 1])
            .cpu()
            .tolist(),
            "target_rank_of_draft_token": rank.cpu().tolist(),
        }

    flattened_offset = 0
    labeled: list[dict[str, Any]] = []
    trace_method = _method(method or ctx.get("method", ""))
    with _lock:
        for req_idx in range(batch_size):
            req_id = req_ids[req_idx]
            n = int(num_draft_tokens[req_idx]) if req_idx < len(num_draft_tokens) else 0
            if n <= 0:
                continue
            pending = _pending.get(req_id)
            if not pending:
                flattened_offset += n
                continue
            records = pending.popleft()
            valid_count = int((output_cpu[req_idx] != -1).sum().item())
            num_accepted = max(0, min(valid_count - 1, n))
            first_reject = None if num_accepted >= n else num_accepted + 1
            scheduled = scheduled_token_ids[flattened_offset : flattened_offset + n]
            for pos in range(min(n, len(records))):
                record = dict(records[pos])
                draft_position = pos + 1
                reached = first_reject is None or draft_position <= first_reject
                accepted_local: int | None
                if not reached:
                    accepted_local = None
                elif first_reject is not None and draft_position == first_reject:
                    accepted_local = 0
                else:
                    accepted_local = 1
                scheduled_id = int(scheduled[pos]) if pos < len(scheduled) else None
                record.update(
                    {
                        "method": trace_method,
                        "reached": int(reached),
                        "accepted_local": accepted_local,
                        "accepted_global": int(accepted_local == 1),
                        "first_reject_position": first_reject,
                        "num_accepted_in_step": num_accepted,
                        "num_scheduled_draft_tokens": n,
                        "trace_pending_tokens": len(records),
                        "scheduled_token_id": scheduled_id,
                        "token_id_match": scheduled_id == record.get("token_id"),
                    }
                )
                flat_idx = flattened_offset + pos
                if target_features and flat_idx < len(
                    target_features["target_selected_logprob"]
                ):
                    target_top1 = int(target_features["target_top1_token_id"][flat_idx])
                    record.update(
                        {
                            "target_selected_logprob": float(
                                target_features["target_selected_logprob"][flat_idx]
                            ),
                            "target_selected_prob": float(
                                target_features["target_selected_prob"][flat_idx]
                            ),
                            "target_top1_token_id": target_top1,
                            "target_top1_logprob": float(
                                target_features["target_top1_logprob"][flat_idx]
                            ),
                            "target_top2_logprob": float(
                                target_features["target_top2_logprob"][flat_idx]
                            ),
                            "target_margin_logprob": float(
                                target_features["target_margin_logprob"][flat_idx]
                            ),
                            "target_rank_of_draft_token": int(
                                target_features["target_rank_of_draft_token"][flat_idx]
                            ),
                            "target_agrees_with_draft_top1": int(
                                target_top1 == record.get("token_id")
                            ),
                        }
                    )
                labeled.append(record)
            flattened_offset += n
    _write_records(labeled)
