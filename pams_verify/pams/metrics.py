from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def safe_percentile(values: Iterable[float], pct: float) -> float:
    vals = list(values)
    return float(np.percentile(vals, pct)) if vals else 0.0


def brier_score(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probs: list[float], labels: list[int], bins: int = 10) -> float:
    if not probs:
        return 0.0
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (p >= lo) & (p <= hi if idx == bins - 1 else p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(conf - acc)
    return float(ece)


def auroc(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    pos = [(s, y) for s, y in zip(scores, labels) if y == 1]
    neg = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = len(pos) * len(neg)
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1.0
            elif math.isclose(ps, ns):
                wins += 0.5
    return float(wins / total)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

