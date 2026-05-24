#!/usr/bin/env python3
"""Fit a lightweight acceptance-probability calibrator from live traces."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FEATURE_NAMES = [
    "position",
    "position_frac",
    "accept_prob",
    "rho",
    "risk",
    "prompt_len_log",
    "decode_len_log",
]


@dataclass
class Example:
    features: list[float]
    label: int
    source: str


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(value: float) -> float:
    value = clamp(value, -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-value))


def read_examples(paths: list[Path], limit: int | None) -> list[Example]:
    examples: list[Example] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and len(examples) >= limit:
                    return examples
                row = json.loads(line)
                accept_probs = row.get("accept_probs") or row.get("a_i")
                accepted_prefix_len = row.get("accepted_prefix_len")
                if accept_probs is None or accepted_prefix_len is None:
                    continue
                try:
                    accepted_prefix_len = int(accepted_prefix_len)
                except (TypeError, ValueError):
                    continue
                rho = row.get("rho") or compute_rho([float(x) for x in accept_probs])
                risk = row.get("risk") or [
                    rho_i * 4.0 * float(prob) * (1.0 - float(prob))
                    for rho_i, prob in zip(rho, accept_probs, strict=False)
                ]
                k = max(1, int(row.get("num_spec_tokens") or len(accept_probs)))
                prompt_len = max(0.0, float(row.get("prompt_len") or 0.0))
                decode_len = max(0.0, float(row.get("decode_len") or 0.0))
                for pos, prob in enumerate(accept_probs[:k]):
                    features = [
                        float(pos),
                        float(pos) / max(k - 1, 1),
                        clamp(float(prob), 0.0, 1.0),
                        clamp(float(rho[pos]) if pos < len(rho) else 0.0, 0.0, 1.0),
                        clamp(float(risk[pos]) if pos < len(risk) else 0.0, 0.0, 1.0),
                        math.log1p(prompt_len),
                        math.log1p(decode_len),
                    ]
                    examples.append(
                        Example(
                            features=features,
                            label=1 if pos < accepted_prefix_len else 0,
                            source=str(path),
                        )
                    )
    return examples


def compute_rho(accept_probs: list[float]) -> list[float]:
    rho: list[float] = []
    reach = 1.0
    for prob in accept_probs:
        rho.append(reach)
        reach *= clamp(float(prob), 0.0, 1.0)
    return rho


def split_examples(examples: list[Example], eval_fraction: float) -> tuple[list[Example], list[Example]]:
    if not examples:
        return [], []
    eval_size = max(1, round(len(examples) * eval_fraction))
    split = max(1, len(examples) - eval_size)
    return examples[:split], examples[split:]


def standardize(train: list[Example], examples: list[Example]) -> tuple[list[list[float]], list[float], list[float]]:
    cols = len(FEATURE_NAMES)
    means = [0.0] * cols
    scales = [1.0] * cols
    for idx in range(cols):
        values = [ex.features[idx] for ex in train]
        means[idx] = sum(values) / len(values)
        var = sum((value - means[idx]) ** 2 for value in values) / len(values)
        scales[idx] = math.sqrt(var) or 1.0
    transformed = [
        [(ex.features[idx] - means[idx]) / scales[idx] for idx in range(cols)]
        for ex in examples
    ]
    return transformed, means, scales


def fit_logistic(
    train: list[Example],
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> dict[str, Any]:
    x_train, means, scales = standardize(train, train)
    y_train = [ex.label for ex in train]
    weights = [0.0] * (len(FEATURE_NAMES) + 1)
    n = float(len(x_train))
    for _ in range(steps):
        grads = [0.0] * len(weights)
        for features, label in zip(x_train, y_train, strict=True):
            pred = sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], features, strict=True)))
            err = pred - label
            grads[0] += err
            for idx, value in enumerate(features, start=1):
                grads[idx] += err * value
        for idx in range(len(weights)):
            penalty = l2 * weights[idx] if idx else 0.0
            weights[idx] -= learning_rate * ((grads[idx] / n) + penalty)
    return {
        "type": "logistic",
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "means": means,
        "scales": scales,
    }


def predict_logistic(model: dict[str, Any], features: list[float]) -> float:
    xs = [
        (features[idx] - model["means"][idx]) / model["scales"][idx]
        for idx in range(len(FEATURE_NAMES))
    ]
    weights = model["weights"]
    return sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], xs, strict=True)))


def fit_isotonic(train: list[Example]) -> dict[str, Any]:
    pairs = sorted((ex.features[2], float(ex.label)) for ex in train)
    blocks: list[dict[str, float]] = []
    for score, label in pairs:
        blocks.append({"lo": score, "hi": score, "sum": label, "weight": 1.0})
        while len(blocks) >= 2:
            prev = blocks[-2]["sum"] / blocks[-2]["weight"]
            cur = blocks[-1]["sum"] / blocks[-1]["weight"]
            if prev <= cur:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "lo": left["lo"],
                    "hi": right["hi"],
                    "sum": left["sum"] + right["sum"],
                    "weight": left["weight"] + right["weight"],
                }
            )
    thresholds = [block["hi"] for block in blocks]
    values = [block["sum"] / block["weight"] for block in blocks]
    return {"type": "isotonic", "input_feature": "accept_prob", "thresholds": thresholds, "values": values}


def predict_isotonic(model: dict[str, Any], features: list[float]) -> float:
    score = features[2]
    thresholds = model["thresholds"]
    values = model["values"]
    for threshold, value in zip(thresholds, values, strict=True):
        if score <= threshold:
            return clamp(float(value), 0.0, 1.0)
    return clamp(float(values[-1]) if values else score, 0.0, 1.0)


def predict(model: dict[str, Any], features: list[float]) -> float:
    if model["type"] == "logistic":
        return predict_logistic(model, features)
    if model["type"] == "isotonic":
        return predict_isotonic(model, features)
    raise ValueError(f"unsupported model type: {model['type']}")


def metrics(model: dict[str, Any], examples: list[Example]) -> dict[str, float]:
    if not examples:
        return {"n": 0.0}
    preds = [clamp(predict(model, ex.features), 1e-6, 1 - 1e-6) for ex in examples]
    labels = [ex.label for ex in examples]
    brier = sum((pred - label) ** 2 for pred, label in zip(preds, labels, strict=True)) / len(labels)
    log_loss = -sum(
        label * math.log(pred) + (1 - label) * math.log(1 - pred)
        for pred, label in zip(preds, labels, strict=True)
    ) / len(labels)
    acc = sum((pred >= 0.5) == bool(label) for pred, label in zip(preds, labels, strict=True)) / len(labels)
    return {
        "n": float(len(labels)),
        "positive_rate": sum(labels) / len(labels),
        "pred_mean": sum(preds) / len(preds),
        "brier": brier,
        "log_loss": log_loss,
        "accuracy_at_0p5": acc,
        "ece_10bin": expected_calibration_error(preds, labels, bins=10),
    }


def expected_calibration_error(preds: list[float], labels: list[int], bins: int) -> float:
    total = len(preds)
    error = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        bucket = [
            (pred, label)
            for pred, label in zip(preds, labels, strict=True)
            if (lo <= pred < hi) or (idx == bins - 1 and pred == 1.0)
        ]
        if not bucket:
            continue
        conf = sum(pred for pred, _ in bucket) / len(bucket)
        acc = sum(label for _, label in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(conf - acc)
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="Pickle path for the fitted calibrator")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--method", choices=["logistic", "isotonic"], default="logistic")
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    trace_paths = [Path(path) for path in args.traces]
    examples = read_examples(trace_paths, args.limit_examples)
    if not examples:
        raise SystemExit("no calibration examples found")
    train, eval_rows = split_examples(examples, args.eval_fraction)
    if args.method == "logistic":
        model = fit_logistic(train, steps=args.steps, learning_rate=args.learning_rate, l2=args.l2)
    else:
        model = fit_isotonic(train)
    model.update(
        {
            "source_traces": [str(path) for path in trace_paths],
            "num_examples": len(examples),
            "num_train_examples": len(train),
            "num_eval_examples": len(eval_rows),
            "label_definition": "label=1 when position < accepted_prefix_len",
            "feature_source": "live trace moving-average accept_probs plus position/rho/risk/prompt/decode length",
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(model, handle)

    summary = {
        "method": args.method,
        "out": str(out_path),
        "source_traces": [str(path) for path in trace_paths],
        "num_examples": len(examples),
        "num_train_examples": len(train),
        "num_eval_examples": len(eval_rows),
        "feature_names": FEATURE_NAMES if args.method == "logistic" else ["accept_prob"],
        "train_metrics": metrics(model, train),
        "eval_metrics": metrics(model, eval_rows),
        "limitations": [
            "Uses accepted_prefix_len from plan-only live traces as the acceptance label.",
            "Uses moving-average accept_probs and metadata available in current traces; draft logprob/entropy is not yet available.",
            "This calibrates planner probabilities only; it is not sparse verifier quality evidence.",
        ],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
