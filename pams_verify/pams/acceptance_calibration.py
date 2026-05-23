from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import auroc, brier_score, expected_calibration_error


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TemperatureCalibrator:
    temperature: float = 1.0
    bias: float = 0.0

    def predict(self, raw_scores: list[float] | np.ndarray) -> list[float]:
        raw = np.asarray(raw_scores, dtype=np.float64)
        return sigmoid(raw / max(self.temperature, 1e-6) + self.bias).clip(1e-4, 1 - 1e-4).tolist()


def _log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = probs.clip(1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def fit_temperature(raw_scores: list[float], labels: list[int]) -> TemperatureCalibrator:
    if not raw_scores:
        return TemperatureCalibrator()
    raw = np.asarray(raw_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    best = (float("inf"), 1.0, 0.0)
    for temp in np.linspace(0.25, 4.0, 32):
        for bias in np.linspace(-2.0, 2.0, 41):
            probs = sigmoid(raw / temp + bias)
            loss = _log_loss(probs, y)
            if loss < best[0]:
                best = (loss, float(temp), float(bias))
    return TemperatureCalibrator(temperature=best[1], bias=best[2])


def calibration_report(calibrator: TemperatureCalibrator, rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(row["draft_logit_margin"]) for row in rows]
    labels = [int(row["accepted"]) for row in rows]
    probs = calibrator.predict(scores)
    return {
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
        "auroc_accept": auroc(probs, labels),
        "temperature": calibrator.temperature,
        "bias": calibrator.bias,
    }


def add_acceptance_priors(rows: list[dict[str, Any]], calibrator: TemperatureCalibrator) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_block.setdefault(str(row["block_id"]), []).append(dict(row))
    out: list[dict[str, Any]] = []
    for block_rows in by_block.values():
        block_rows.sort(key=lambda item: int(item["token_index"]))
        scores = [float(row["draft_logit_margin"]) for row in block_rows]
        probs = calibrator.predict(scores)
        rho = 1.0
        for row, prob in zip(block_rows, probs):
            risk = rho * 4.0 * prob * (1.0 - prob)
            row["acceptance_prior"] = prob
            row["prefix_reach_probability"] = rho
            row["risk_score"] = risk
            rho *= prob
            out.append(row)
    return out

