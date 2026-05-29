#!/usr/bin/env python3
"""Core SpecLink-CV decision, queue, and trace utilities."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class CVState(str, Enum):
    NORMAL = "NORMAL"
    DRAFTING = "DRAFTING"
    VERIFY_READY = "VERIFY_READY"
    VERIFY_PREFIX = "VERIFY_PREFIX"
    PREFIX_REJECTED = "PREFIX_REJECTED"
    PREFIX_ACCEPTED = "PREFIX_ACCEPTED"
    VERIFY_SUFFIX_READY = "VERIFY_SUFFIX_READY"
    VERIFY_SUFFIX = "VERIFY_SUFFIX"
    COMMIT_READY = "COMMIT_READY"
    DONE_OR_NEXT_STEP = "DONE_OR_NEXT_STEP"


@dataclass(frozen=True)
class ChunkDecision:
    k: int
    candidate_chunks: list[int]
    selected_h: int
    reason: str
    confidence_sizing: bool
    prefix_survival: dict[int, float]
    reject_probability: dict[int, float]
    expected_benefit: dict[int, float]
    a_hat: list[float]


@dataclass
class SpecLinkCVConfig:
    enable: bool = False
    confidence_sizing: bool = False
    async_queue: bool = False
    roofline_packing: bool = False
    candidate_chunks: tuple[int | str, ...] = (1, 2, 4, 6, 8, "full")
    default_half_policy: str = "floor"
    min_benefit: float = 0.0
    max_verify_tokens_per_step: int = 8192
    max_verify_seqs_per_step: int = 32
    max_queue_wait_ms: float = 2.0
    util_threshold: float = 0.6
    age_weight: float = 0.001


@dataclass(frozen=True)
class DecodeStepTrace:
    workload: str
    model_label: str
    method: str
    request_id: str
    step_id: int
    k: int
    actual_accept_tokens: int
    draft_selected_prob: tuple[float, ...]
    draft_margin_logprob: tuple[float, ...] = ()
    draft_entropy: tuple[float, ...] = ()


@dataclass
class SpecLinkCVState:
    request_id: str
    model_id: str = ""
    dataset_id: str = ""
    k: int = 0
    draft_tokens: list[int] = field(default_factory=list)
    a_hat: list[float] = field(default_factory=list)
    selected_h: int = 0
    verified_prefix_len: int = 0
    accepted_prefix_len: int = 0
    first_reject_pos: int | None = None
    suffix_pending: bool = False
    chunk_phase: str = ""
    queue_enter_time: float = 0.0
    queue_wait_ms: float = 0.0
    num_extra_tlm_forwards: int = 0
    skipped_suffix_tokens: int = 0
    fallback_reason: str = ""
    state: CVState = CVState.NORMAL
    correctness_debug_info: dict[str, Any] = field(default_factory=dict)

    def on_draft(self, *, k: int, a_hat: list[float], selected_h: int) -> None:
        self.k = k
        self.a_hat = a_hat
        self.selected_h = selected_h
        self.state = CVState.VERIFY_READY

    def begin_prefix(self, now: float | None = None) -> None:
        self.chunk_phase = "prefix"
        self.queue_enter_time = now if now is not None else time.time()
        self.state = CVState.VERIFY_PREFIX

    def finish_prefix(self, actual_accept_tokens: int) -> None:
        self.verified_prefix_len = self.selected_h
        self.accepted_prefix_len = min(actual_accept_tokens, self.selected_h)
        if actual_accept_tokens < self.selected_h:
            self.first_reject_pos = actual_accept_tokens + 1
            self.suffix_pending = False
            self.skipped_suffix_tokens = max(self.k - self.selected_h, 0)
            self.state = CVState.PREFIX_REJECTED
        elif self.selected_h < self.k:
            self.suffix_pending = True
            self.state = CVState.PREFIX_ACCEPTED
        else:
            self.suffix_pending = False
            self.state = CVState.COMMIT_READY

    def begin_suffix(self) -> None:
        if not self.suffix_pending:
            raise ValueError("suffix is not pending")
        self.chunk_phase = "suffix"
        self.state = CVState.VERIFY_SUFFIX
        self.num_extra_tlm_forwards += 1

    def finish_suffix(self, actual_accept_tokens: int) -> None:
        self.first_reject_pos = None if actual_accept_tokens >= self.k else actual_accept_tokens + 1
        self.suffix_pending = False
        self.state = CVState.COMMIT_READY

    def commit(self) -> None:
        self.state = CVState.DONE_OR_NEXT_STEP


@dataclass(frozen=True)
class VerifyChunk:
    request_id: str
    chunk_id: str
    phase: str
    start_draft_pos: int
    chunk_len: int
    k: int
    selected_h: int
    survival_prob: float
    reject_prob: float
    expected_benefit: float
    context_len: int = 0
    arrival_time: float = 0.0
    deadline_or_max_wait: float = 2.0
    model_id: str = ""
    dataset_id: str = ""

    def age_ms(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, (now - self.arrival_time) * 1000.0)


class VerificationQueue:
    def __init__(self, config: SpecLinkCVConfig):
        self.config = config
        self._chunks: list[VerifyChunk] = []

    def push(self, chunk: VerifyChunk) -> None:
        self._chunks.append(chunk)

    def __len__(self) -> int:
        return len(self._chunks)

    def pop_ready(self, now: float | None = None) -> list[VerifyChunk]:
        now = now if now is not None else time.time()
        scored: list[tuple[float, int, VerifyChunk]] = []
        for idx, chunk in enumerate(self._chunks):
            age = chunk.age_ms(now)
            priority = chunk.expected_benefit / max(chunk.chunk_len, 1) + self.config.age_weight * age
            if age >= self.config.max_queue_wait_ms:
                priority += 1e6
            scored.append((priority, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[VerifyChunk] = []
        token_budget = 0
        seq_budget = 0
        selected_indices: set[int] = set()
        for _, idx, chunk in scored:
            if seq_budget + 1 > self.config.max_verify_seqs_per_step:
                continue
            if token_budget + chunk.chunk_len > self.config.max_verify_tokens_per_step:
                continue
            selected.append(chunk)
            selected_indices.add(idx)
            seq_budget += 1
            token_budget += chunk.chunk_len
        self._chunks = [chunk for idx, chunk in enumerate(self._chunks) if idx not in selected_indices]
        return selected


@dataclass(frozen=True)
class CostEstimate:
    chunk_len: int
    batch_size: int
    predicted_time_ms: float
    predicted_utilization: float
    token_budget_utilization: float


class RooflinePacker:
    def __init__(self, config: SpecLinkCVConfig):
        self.config = config

    def estimate(self, chunks: Iterable[VerifyChunk]) -> CostEstimate:
        group = list(chunks)
        total_tokens = sum(chunk.chunk_len for chunk in group)
        batch_size = len(group)
        token_budget = max(self.config.max_verify_tokens_per_step, 1)
        seq_budget = max(self.config.max_verify_seqs_per_step, 1)
        token_util = total_tokens / token_budget
        seq_util = batch_size / seq_budget
        predicted_util = min(1.0, max(token_util, seq_util))
        # Lightweight lookup-free proxy. It is for packing tests and reports, not
        # for claiming hardware-level roofline accuracy.
        predicted_time = 0.03 * total_tokens + 0.08 * batch_size + 0.25
        return CostEstimate(
            chunk_len=total_tokens,
            batch_size=batch_size,
            predicted_time_ms=predicted_time,
            predicted_utilization=predicted_util,
            token_budget_utilization=token_util,
        )

    def select(self, chunks: list[VerifyChunk]) -> tuple[list[VerifyChunk], CostEstimate, str]:
        if not chunks:
            return [], self.estimate([]), "empty"
        fifo = chunks[: self.config.max_verify_seqs_per_step]
        within_budget: list[VerifyChunk] = []
        total_tokens = 0
        for chunk in fifo:
            if total_tokens + chunk.chunk_len > self.config.max_verify_tokens_per_step:
                continue
            within_budget.append(chunk)
            total_tokens += chunk.chunk_len
        estimate = self.estimate(within_budget)
        if not self.config.roofline_packing:
            return within_budget, estimate, "simple"
        if estimate.predicted_utilization >= self.config.util_threshold:
            return within_budget, estimate, "roofline"
        urgent = [
            chunk
            for chunk in within_budget
            if chunk.age_ms() >= self.config.max_queue_wait_ms
        ]
        if urgent:
            return urgent, self.estimate(urgent), "age_timeout"
        return [], estimate, "wait_for_utilization"


def normalize_candidates(k: int, candidates: Iterable[int | str]) -> list[int]:
    values = set()
    for candidate in candidates:
        if isinstance(candidate, str):
            if candidate == "full":
                values.add(k)
            elif candidate.strip():
                values.add(int(candidate))
        else:
            values.add(int(candidate))
    values.add(k)
    return sorted(value for value in values if 1 <= value <= k)


def half_chunk(k: int, policy: str = "floor") -> int:
    if k <= 1:
        return k
    if policy == "ceil":
        return int(math.ceil(k / 2))
    if policy != "floor":
        raise ValueError(f"unknown half policy: {policy}")
    return max(1, k // 2)


def choose_chunk_size(
    *,
    k: int,
    a_hat: list[float],
    config: SpecLinkCVConfig,
    suffix_cost_per_token: float = 1.0,
    extra_forward_cost: float = 0.0,
    underutilization_cost: float = 0.0,
) -> ChunkDecision:
    candidates = normalize_candidates(k, config.candidate_chunks)
    probs = [min(max(float(value), 1e-6), 1.0) for value in a_hat[:k]]
    if len(probs) < k:
        probs.extend([0.5] * (k - len(probs)))
    if not config.confidence_sizing:
        h = half_chunk(k, config.default_half_policy)
        return ChunkDecision(
            k=k,
            candidate_chunks=candidates,
            selected_h=h,
            reason="fixed_half",
            confidence_sizing=False,
            prefix_survival={h: math.prod(probs[:h])},
            reject_probability={h: 1.0 - math.prod(probs[:h])},
            expected_benefit={h: 0.0},
            a_hat=probs,
        )

    prefix_survival: dict[int, float] = {}
    reject_probability: dict[int, float] = {}
    expected_benefit: dict[int, float] = {}
    for h in candidates:
        survival = math.prod(probs[:h])
        reject_prob = 1.0 - survival
        suffix_len = max(k - h, 0)
        suffix_cost = suffix_len * suffix_cost_per_token
        extra_cost = extra_forward_cost + underutilization_cost
        benefit = reject_prob * suffix_cost - extra_cost
        prefix_survival[h] = survival
        reject_probability[h] = reject_prob
        expected_benefit[h] = benefit
    selected_h = max(candidates, key=lambda h: (expected_benefit[h], h))
    reason = "confidence"
    if expected_benefit[selected_h] <= config.min_benefit:
        selected_h = k
        reason = "one_shot_min_benefit"
    return ChunkDecision(
        k=k,
        candidate_chunks=candidates,
        selected_h=selected_h,
        reason=reason,
        confidence_sizing=True,
        prefix_survival=prefix_survival,
        reject_probability=reject_probability,
        expected_benefit=expected_benefit,
        a_hat=probs,
    )


def find_trace_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*.jsonl")
        if path.parent.name == "trace" or path.name.endswith("_trace.jsonl")
    )


def load_trace_steps(root: Path, workloads: set[str] | None = None) -> list[DecodeStepTrace]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in find_trace_files(root):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                workload = str(row.get("dataset_label") or "")
                if workloads is not None and workload not in workloads:
                    continue
                key = (
                    workload,
                    str(row.get("model_label") or "unknown"),
                    str(row.get("method") or "unknown"),
                    str(row.get("request_id") or row.get("sequence_id") or ""),
                    int(row.get("step_id") or 0),
                    int(row.get("num_spec_tokens") or row.get("num_scheduled_draft_tokens") or 0),
                )
                group = groups.setdefault(
                    key,
                    {
                        "positions": [],
                        "probs": [],
                        "margins": [],
                        "entropies": [],
                        "accepted": int(row.get("num_accepted_in_step") or 0),
                    },
                )
                group["positions"].append(int(row.get("draft_position") or 0))
                group["probs"].append(float(row.get("draft_selected_prob") or 0.0))
                group["margins"].append(float(row.get("draft_margin_logprob") or 0.0))
                group["entropies"].append(float(row.get("draft_entropy") or 0.0))
    steps: list[DecodeStepTrace] = []
    for key, value in groups.items():
        workload, model_label, method, request_id, step_id, k = key
        ordering = sorted(range(len(value["positions"])), key=lambda idx: value["positions"][idx])
        steps.append(
            DecodeStepTrace(
                workload=workload,
                model_label=model_label,
                method=method,
                request_id=request_id,
                step_id=step_id,
                k=k,
                actual_accept_tokens=max(0, min(int(value["accepted"]), k)),
                draft_selected_prob=tuple(value["probs"][idx] for idx in ordering),
                draft_margin_logprob=tuple(value["margins"][idx] for idx in ordering),
                draft_entropy=tuple(value["entropies"][idx] for idx in ordering),
            )
        )
    return sorted(
        steps,
        key=lambda item: (
            item.workload,
            item.model_label,
            item.method,
            item.request_id,
            item.step_id,
        ),
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        if fieldnames is None:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_snapshot(config: SpecLinkCVConfig) -> dict[str, Any]:
    data = asdict(config)
    data["candidate_chunks"] = list(config.candidate_chunks)
    return data
