from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Method = Literal[
    "dense_all_blocks",
    "independent_topk",
    "shared_only",
    "shared_fixed_residual",
    "pams",
    "pams_fallback",
    "oracle_shared_residual",
]


@dataclass
class TokenCandidates:
    token_index: int
    candidate_scores: dict[int, float]
    acceptance_prior: float
    prefix_reach_probability: float
    risk_score: float
    target_top_blocks: set[int] | None = None


@dataclass
class PlanResult:
    method: str
    shared_blocks: set[int]
    token_blocks: dict[int, set[int]]
    fallback_tokens: set[int]

    @property
    def union_blocks(self) -> set[int]:
        out: set[int] = set()
        for blocks in self.token_blocks.values():
            out |= blocks
        return out


def _top(scores: dict[int, float], budget: int) -> set[int]:
    if budget <= 0:
        return set()
    return {block for block, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:budget]}


def _aggregate(tokens: list[TokenCandidates], weights: list[float] | None = None) -> dict[int, float]:
    agg: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(tokens)
    for token, weight in zip(tokens, weights):
        for block, score in token.candidate_scores.items():
            agg[block] = agg.get(block, 0.0) + weight * score
    return agg


def plan_masks(
    tokens: list[TokenCandidates],
    *,
    method: Method,
    dense_total_blocks: int,
    topk: int = 8,
    shared_budget_ratio: float = 0.5,
    residual_budget: int = 4,
    lambda_risk: float = 1.0,
    alpha_reach: float = 1.0,
    beta_risk: float = 1.0,
    fallback_threshold: float = 0.65,
    low_uncertain: float = 0.35,
    high_uncertain: float = 0.75,
    b_min: int = 1,
) -> PlanResult:
    if not tokens:
        return PlanResult(method, set(), {}, set())

    if method == "dense_all_blocks":
        dense = set(range(dense_total_blocks))
        return PlanResult(method, dense, {t.token_index: set(dense) for t in tokens}, set())

    if method == "oracle_shared_residual":
        oracle_tokens = []
        for token in tokens:
            scores = {b: 1.0 / (rank + 1) for rank, b in enumerate(sorted(token.target_top_blocks or []))}
            oracle_tokens.append(TokenCandidates(token.token_index, scores, token.acceptance_prior, token.prefix_reach_probability, token.risk_score))
        tokens_for_plan = oracle_tokens
    else:
        tokens_for_plan = tokens

    if method == "independent_topk":
        return PlanResult(
            method,
            set(),
            {token.token_index: _top(token.candidate_scores, topk) for token in tokens_for_plan},
            set(),
        )

    shared_budget = max(1, int(round(topk * shared_budget_ratio)))
    if method == "shared_only":
        shared = _top(_aggregate(tokens_for_plan), topk)
        return PlanResult(method, shared, {token.token_index: set(shared) for token in tokens_for_plan}, set())

    if method in {"shared_fixed_residual", "oracle_shared_residual"}:
        shared = _top(_aggregate(tokens_for_plan), shared_budget)
        token_blocks = {}
        for token in tokens_for_plan:
            residual_scores = {b: s for b, s in token.candidate_scores.items() if b not in shared}
            token_blocks[token.token_index] = set(shared) | _top(residual_scores, residual_budget)
        return PlanResult(method, shared, token_blocks, set())

    weights = [
        token.prefix_reach_probability * token.acceptance_prior + lambda_risk * token.risk_score
        for token in tokens_for_plan
    ]
    shared = _top(_aggregate(tokens_for_plan, weights), shared_budget)
    token_blocks = {}
    fallback_tokens: set[int] = set()
    for token in tokens_for_plan:
        continuous_budget = (
            b_min
            + alpha_reach * token.prefix_reach_probability * token.acceptance_prior
            + beta_risk * token.risk_score
        )
        budget = max(b_min, int(round(continuous_budget)))
        residual_scores = {b: s for b, s in token.candidate_scores.items() if b not in shared}
        token_blocks[token.token_index] = set(shared) | _top(residual_scores, budget)
        if method == "pams_fallback":
            uncertain = low_uncertain <= token.acceptance_prior <= high_uncertain
            if token.risk_score > fallback_threshold or (token.token_index <= 1 and uncertain):
                fallback_tokens.add(token.token_index)
    return PlanResult(method, shared, token_blocks, fallback_tokens)

