"""SpecLink sparse verification planner.

This module is intentionally independent of vLLM so the planner can be tested
and used by offline simulators before a serving-time sparse kernel exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BlockScore:
    block: int
    score: float


@dataclass(frozen=True)
class SpeclinkConfig:
    layout: str = "speclink_prob"
    topk_per_token: int = 32
    shared_budget: int = 32
    private_min: int = 0
    private_max: int = 16
    alpha: float = 8.0
    beta: float = 8.0
    lambda_risk: float = 1.0
    risk_threshold: float = 0.35
    fallback_enabled: bool = False


@dataclass(frozen=True)
class SpeclinkPlan:
    shared_blocks: list[int]
    residual_blocks_per_token: list[list[int]]
    final_blocks_per_token: list[list[int]]
    union_blocks: list[int]
    rho: list[float]
    risk: list[float]
    per_token_budget: list[int]
    fallback_tokens: list[int]
    stats: dict[str, Any] = field(default_factory=dict)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _as_block_scores(items: Iterable[Any]) -> list[BlockScore]:
    out: list[BlockScore] = []
    for item in items:
        if isinstance(item, BlockScore):
            out.append(item)
        elif isinstance(item, dict):
            out.append(BlockScore(block=int(item["block"]), score=float(item["score"])))
        else:
            block, score = item
            out.append(BlockScore(block=int(block), score=float(score)))
    out.sort(key=lambda x: (-x.score, x.block))
    return out


def _top_blocks(items: Sequence[BlockScore], budget: int, banned: set[int] | None = None) -> list[int]:
    if budget <= 0:
        return []
    banned = banned or set()
    selected: list[int] = []
    for item in items:
        if item.block in banned or item.block in selected:
            continue
        selected.append(item.block)
        if len(selected) >= budget:
            break
    return selected


def _rho_and_risk(accept_probs: Sequence[float]) -> tuple[list[float], list[float]]:
    rho: list[float] = []
    risk: list[float] = []
    prefix = 1.0
    for prob in accept_probs:
        a_i = _clamp(float(prob), 0.0, 1.0)
        rho.append(prefix)
        risk.append(prefix * 4.0 * a_i * (1.0 - a_i))
        prefix *= a_i
    return rho, risk


class SpeclinkPlanner:
    """Build shared and per-token sparse KV plans for speculative verification."""

    def __init__(self, config: SpeclinkConfig | None = None) -> None:
        self.config = config or SpeclinkConfig()

    def plan(
        self,
        candidates: Sequence[Iterable[Any]],
        accept_probs: Sequence[float],
        config: SpeclinkConfig | None = None,
    ) -> SpeclinkPlan:
        cfg = config or self.config
        normalized = [_as_block_scores(items) for items in candidates]
        if len(normalized) != len(accept_probs):
            raise ValueError("candidates and accept_probs must have the same length")
        rho, risk = _rho_and_risk(accept_probs)

        if cfg.layout == "independent_topk":
            return self._independent_plan(normalized, rho, risk, cfg)

        shared_blocks = self._shared_blocks(normalized, accept_probs, rho, risk, cfg)
        shared_set = set(shared_blocks)
        residuals: list[list[int]] = []
        final: list[list[int]] = []
        budgets: list[int] = []
        fallback_tokens: list[int] = []

        for idx, items in enumerate(normalized):
            budget = self._private_budget(idx, accept_probs, rho, risk, cfg)
            budgets.append(budget)
            residual = _top_blocks(items, budget, banned=shared_set)
            residuals.append(residual)
            final_blocks = sorted(shared_set.union(residual))
            final.append(final_blocks)
            if cfg.fallback_enabled and risk[idx] > cfg.risk_threshold:
                fallback_tokens.append(idx)

        union_blocks = sorted({block for blocks in final for block in blocks})
        stats = self._stats(final, residuals, shared_blocks, union_blocks)
        stats["layout"] = cfg.layout
        return SpeclinkPlan(
            shared_blocks=shared_blocks,
            residual_blocks_per_token=residuals,
            final_blocks_per_token=final,
            union_blocks=union_blocks,
            rho=rho,
            risk=risk,
            per_token_budget=budgets,
            fallback_tokens=fallback_tokens,
            stats=stats,
        )

    def _independent_plan(
        self,
        candidates: Sequence[Sequence[BlockScore]],
        rho: Sequence[float],
        risk: Sequence[float],
        cfg: SpeclinkConfig,
    ) -> SpeclinkPlan:
        final = [_top_blocks(items, cfg.topk_per_token) for items in candidates]
        union_blocks = sorted({block for blocks in final for block in blocks})
        stats = self._stats(final, final, [], union_blocks)
        stats["layout"] = cfg.layout
        return SpeclinkPlan(
            shared_blocks=[],
            residual_blocks_per_token=final,
            final_blocks_per_token=final,
            union_blocks=union_blocks,
            rho=list(rho),
            risk=list(risk),
            per_token_budget=[cfg.topk_per_token for _ in final],
            fallback_tokens=[],
            stats=stats,
        )

    def _shared_blocks(
        self,
        candidates: Sequence[Sequence[BlockScore]],
        accept_probs: Sequence[float],
        rho: Sequence[float],
        risk: Sequence[float],
        cfg: SpeclinkConfig,
    ) -> list[int]:
        if cfg.shared_budget <= 0:
            return []
        utilities: dict[int, float] = {}
        if cfg.layout == "snapkv_static":
            for item in candidates[0] if candidates else []:
                utilities[item.block] = utilities.get(item.block, 0.0) + item.score
        else:
            for idx, items in enumerate(candidates):
                prob = _clamp(float(accept_probs[idx]), 0.0, 1.0)
                if cfg.layout == "shared_only":
                    weight = 1.0
                elif cfg.layout in {"speclink_fixed", "speclink_prob", "speclink_prob_fallback"}:
                    weight = rho[idx] * prob + cfg.lambda_risk * risk[idx]
                else:
                    raise ValueError(f"unsupported Speclink layout: {cfg.layout}")
                for item in items:
                    utilities[item.block] = utilities.get(item.block, 0.0) + weight * item.score
        ranked = sorted(utilities.items(), key=lambda x: (-x[1], x[0]))
        return [block for block, _score in ranked[: cfg.shared_budget]]

    def _private_budget(
        self,
        idx: int,
        accept_probs: Sequence[float],
        rho: Sequence[float],
        risk: Sequence[float],
        cfg: SpeclinkConfig,
    ) -> int:
        if cfg.layout in {"snapkv_static", "shared_only"}:
            return 0
        if cfg.layout == "speclink_fixed":
            return max(0, min(cfg.private_max, cfg.topk_per_token - cfg.shared_budget))
        prob = _clamp(float(accept_probs[idx]), 0.0, 1.0)
        raw = cfg.private_min + round(cfg.alpha * rho[idx] * prob + cfg.beta * risk[idx])
        return int(_clamp(raw, 0, cfg.private_max))

    @staticmethod
    def _stats(
        final: Sequence[Sequence[int]],
        residuals: Sequence[Sequence[int]],
        shared: Sequence[int],
        union_blocks: Sequence[int],
    ) -> dict[str, Any]:
        per_token = [len(blocks) for blocks in final]
        residual_counts = [len(blocks) for blocks in residuals]
        mean_blocks = sum(per_token) / len(per_token) if per_token else 0.0
        return {
            "num_tokens": len(final),
            "shared_blocks": len(shared),
            "mean_blocks_per_token": mean_blocks,
            "union_blocks": len(union_blocks),
            "residual_counts": residual_counts,
            "private_unique_blocks": len(
                {block for blocks in residuals for block in blocks}
            ),
        }
