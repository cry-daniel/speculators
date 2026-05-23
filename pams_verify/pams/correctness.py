from __future__ import annotations

from typing import Any


def verifier_decision_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "decision_match_rate": 0.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "dense_fallback_rate": 0.0,
        }
    matches = 0
    false_accept = 0
    false_reject = 0
    fallback = 0
    for row in rows:
        dense = bool(row.get("dense_accept"))
        sparse = bool(row.get("sparse_accept"))
        if row.get("dense_fallback", False):
            fallback += 1
        if dense == sparse:
            matches += 1
        elif sparse and not dense:
            false_accept += 1
        elif dense and not sparse:
            false_reject += 1
    return {
        "decision_match_rate": matches / total,
        "false_accept_rate": false_accept / total,
        "false_reject_rate": false_reject / total,
        "dense_fallback_rate": fallback / total,
    }


def token_id_exact_match(reference: list[int], candidate: list[int]) -> bool:
    return list(reference) == list(candidate)

