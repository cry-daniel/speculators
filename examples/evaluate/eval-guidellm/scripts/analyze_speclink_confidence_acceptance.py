#!/usr/bin/env python3
"""Analyze SpecLink DLM-confidence vs TLM-acceptance traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


FEATURES_FULL = [
    "draft_selected_logprob",
    "draft_margin_logprob",
    "draft_entropy",
    "draft_position",
    "context_len",
    "generated_len_so_far",
    "num_spec_tokens",
]
PREDICTORS = {
    "position_only": [],
    "logprob": ["draft_selected_logprob"],
    "margin": ["draft_margin_logprob"],
    "entropy": ["draft_entropy"],
    "full": FEATURES_FULL,
}
COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
]
LEGACY_MODEL_LABEL = "qwen3_8b"
LEGACY_DATASET_LABEL = "math"


def normalize_model_label(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return LEGACY_MODEL_LABEL


def normalize_dataset_label(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return LEGACY_DATASET_LABEL


def group_name(frame: pd.DataFrame | pd.Series | dict[str, Any]) -> str:
    if isinstance(frame, pd.DataFrame):
        dataset = normalize_dataset_label(frame["dataset_label"].iloc[0])
        model = normalize_model_label(frame["model_label"].iloc[0])
        method = str(frame["method"].iloc[0])
        k = int(frame["num_spec_tokens"].iloc[0])
    elif isinstance(frame, pd.Series):
        dataset = normalize_dataset_label(frame.get("dataset_label"))
        model = normalize_model_label(frame.get("model_label"))
        method = str(frame.get("method"))
        k = int(frame.get("num_spec_tokens"))
    else:
        dataset = normalize_dataset_label(frame.get("dataset_label"))
        model = normalize_model_label(frame.get("model_label"))
        method = str(frame.get("method"))
        k = int(frame.get("num_spec_tokens"))
    return f"{dataset}/{model}/{method}/K={k}"


def file_prefix(dataset_label: str, model_label: str, method: str) -> str:
    clean_dataset = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_label)
    clean_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_label)
    clean_method = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in method)
    return f"{clean_dataset}_{clean_model}_{clean_method}"


@dataclass
class FittedLogistic:
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    constant: float | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(frame), self.constant, dtype=np.float64)
        x = frame[self.columns].astype(float).to_numpy()
        x = np.nan_to_num(x, nan=self.mean)
        x = (x - self.mean) / self.std
        x = np.column_stack([np.ones(len(x)), x])
        return sigmoid(x @ self.weights)

    def to_json(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "constant": self.constant,
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_logistic(frame: pd.DataFrame, columns: list[str]) -> FittedLogistic:
    y = frame["accepted_local"].astype(float).to_numpy()
    if len(frame) == 0:
        return FittedLogistic(columns, np.zeros(len(columns)), np.ones(len(columns)), np.zeros(len(columns) + 1), 0.5)
    if len(np.unique(y)) < 2:
        return FittedLogistic(columns, np.zeros(len(columns)), np.ones(len(columns)), np.zeros(len(columns) + 1), float(np.mean(y)))
    x = frame[columns].astype(float).to_numpy()
    mean = np.zeros(x.shape[1], dtype=np.float64)
    for col_idx in range(x.shape[1]):
        finite = np.isfinite(x[:, col_idx])
        mean[col_idx] = float(x[finite, col_idx].mean()) if finite.any() else 0.0
    x = np.nan_to_num(x, nan=mean)
    std = np.nanstd(x, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    x = (x - mean) / std
    x = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(x.shape[1], dtype=np.float64)
    lr = 0.15
    l2 = 1e-3
    for _ in range(900):
        p = sigmoid(x @ w)
        grad = (x.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return FittedLogistic(columns, mean, std, w)


def stable_split(request_id: str) -> str:
    value = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16) % 10
    if value < 6:
        return "train"
    if value < 8:
        return "val"
    return "test"


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - y.sum())
    rank_sum = float(np.where(y == 1)[0].sum() + y.sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float(y.sum()), 1.0)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, float]:
    ece = 0.0
    mce = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any():
            continue
        gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
        ece += gap * float(mask.mean())
        mce = max(mce, gap)
    return ece, mce


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) <= 1 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    y = y.astype(float)
    pred = (p >= 0.5).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ece, mce = calibration_error(y, p)
    pearson = correlation(y, p)
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    p_rank = pd.Series(p).rank(method="average").to_numpy()
    spearman = correlation(y_rank, p_rank)
    return {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else float("nan"),
        "brier": float(((p - y) ** 2).mean()) if len(y) else float("nan"),
        "ece": ece,
        "mce": mce,
        "pearson": pearson,
        "spearman": spearman,
    }


def evidence_level(full: dict[str, float], position: dict[str, float], benefit_gain: float) -> str:
    au = full.get("auroc", float("nan"))
    ece = full.get("ece", float("nan"))
    brier = full.get("brier", float("nan"))
    base_brier = position.get("brier", float("nan"))
    brier_better = np.isfinite(brier) and np.isfinite(base_brier) and brier < base_brier
    if au >= 0.75 and ece <= 0.08 and brier_better and benefit_gain > 0:
        return "Strong"
    if au >= 0.65 and ece <= 0.15 and brier_better:
        return "Moderate"
    return "Weak"


def load_trace(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "trace").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No trace rows found under {root / 'trace'}")
    frame = pd.DataFrame(rows)
    if "dataset_label" not in frame:
        frame["dataset_label"] = LEGACY_DATASET_LABEL
    frame["dataset_label"] = frame["dataset_label"].map(normalize_dataset_label)
    if "model_label" not in frame:
        frame["model_label"] = LEGACY_MODEL_LABEL
    frame["model_label"] = frame["model_label"].map(normalize_model_label)
    for col in ["accepted_local", "first_reject_position", "prompt_id", "dataset_index"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def position_prior(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    global_mean = float(train["accepted_local"].mean()) if len(train) else 0.5
    by_pos = train.groupby("draft_position")["accepted_local"].mean().to_dict()
    return frame["draft_position"].map(by_pos).fillna(global_mean).astype(float).to_numpy()


def calibrate_group(frame: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    frame = frame.copy()
    frame["split"] = frame["request_id"].astype(str).map(stable_split)
    usable = frame[(frame["reached"] == 1) & frame["accepted_local"].notna()].copy()
    train = usable[usable["split"] == "train"]
    test = usable[usable["split"] == "test"]
    model_params: dict[str, Any] = {}
    frame["pred_accept_prob_position_only"] = position_prior(train, frame)
    for name, cols in PREDICTORS.items():
        if name == "position_only":
            continue
        model = fit_logistic(train, cols)
        frame[f"pred_accept_prob_{name}"] = model.predict(frame)
        model_params[name] = model.to_json()

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = normalize_dataset_label(frame["dataset_label"].iloc[0])
    model_label = normalize_model_label(frame["model_label"].iloc[0])
    method = str(frame["method"].iloc[0])
    k = int(frame["num_spec_tokens"].iloc[0])
    prefix = file_prefix(dataset_label, model_label, method)
    (out_dir / f"{prefix}_k{k}_model_params.json").write_text(
        json.dumps(model_params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    predictor_metrics: dict[str, dict[str, float]] = {}
    for name in PREDICTORS:
        pred_col = f"pred_accept_prob_{name}"
        if len(test):
            y = test["accepted_local"].astype(float).to_numpy()
            p = frame.loc[test.index, pred_col].astype(float).to_numpy()
            predictor_metrics[name] = metrics(y, p)
        else:
            predictor_metrics[name] = {}

    prefix_rows = prefix_metrics(frame)
    benefit_top = top_benefit(prefix_rows)
    pos_metrics = predictor_metrics.get("position_only", {})
    full_metrics = predictor_metrics.get("full", {})
    best = best_predictor(predictor_metrics)
    summary = {
        "dataset_label": dataset_label,
        "model_label": model_label,
        "method": method,
        "num_spec_tokens": k,
        "num_prompts": int(frame["request_id"].nunique()),
        "num_steps": int(frame[["request_id", "step_id"]].drop_duplicates().shape[0]),
        "num_draft_tokens": int(len(frame)),
        "num_reached_tokens": int(len(usable)),
        "acceptance_rate_overall_reached": float(usable["accepted_local"].mean()) if len(usable) else float("nan"),
        "best_predictor": best,
        "predicted_benefit_top_quantile_actual_skipped_tokens": benefit_top,
        "evidence_level": evidence_level(full_metrics, pos_metrics, benefit_top),
    }
    for pos in range(1, max(4, k) + 1):
        pos_df = usable[usable["draft_position"] == pos]
        summary[f"acceptance_rate_pos{pos}"] = float(pos_df["accepted_local"].mean()) if len(pos_df) else float("nan")
    for key, value in full_metrics.items():
        summary[key] = value
    for h in [1, 2, 4, k]:
        h_rows = [row for row in prefix_rows if row["h"] == h and row["split"] == "test"]
        if h_rows:
            y = np.array([row["actual_reject_within_h"] for row in h_rows], dtype=float)
            p = np.array([row["pred_reject_within_h"] for row in h_rows], dtype=float)
            summary[f"reject_within_{h}_auroc"] = auroc(y, p)
            summary[f"reject_within_{h}_brier"] = float(((p - y) ** 2).mean())
            summary[f"reject_within_{h}_ece"] = calibration_error(y, p)[0]
    summary["predictors"] = predictor_metrics
    return frame, summary, prefix_rows


def best_predictor(all_metrics: dict[str, dict[str, float]]) -> str:
    best = "none"
    best_score = -float("inf")
    for name, vals in all_metrics.items():
        score = vals.get("auroc", float("nan"))
        if not np.isfinite(score):
            score = -vals.get("brier", float("inf"))
        if score > best_score:
            best = name
            best_score = score
    return best


def prefix_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (request_id, step_id), group in frame.groupby(["request_id", "step_id"]):
        group = group.sort_values("draft_position")
        k = int(group["num_spec_tokens"].iloc[0])
        first_reject = group["first_reject_position"].dropna()
        first_reject_pos = int(first_reject.iloc[0]) if len(first_reject) else None
        split = str(group["split"].iloc[0])
        dataset_label = normalize_dataset_label(group["dataset_label"].iloc[0])
        model_label = normalize_model_label(group["model_label"].iloc[0])
        method = str(group["method"].iloc[0])
        for h in sorted({1, 2, 4, k}):
            if h > k:
                continue
            prefix = group[group["draft_position"] <= h]
            if len(prefix) < h:
                continue
            probs = prefix["pred_accept_prob_full"].astype(float).clip(1e-6, 1 - 1e-6)
            pred_reject = 1.0 - float(np.prod(probs))
            actual_reject = int(first_reject_pos is not None and first_reject_pos <= h)
            skipped = (k - h) if actual_reject else 0
            rows.append(
                {
                    "method": method,
                    "dataset_label": dataset_label,
                    "model_label": model_label,
                    "num_spec_tokens": k,
                    "request_id": request_id,
                    "step_id": int(step_id),
                    "split": split,
                    "h": h,
                    "pred_reject_within_h": pred_reject,
                    "actual_reject_within_h": actual_reject,
                    "predicted_benefit": pred_reject * (k - h),
                    "actual_skip_tokens_if_chunk_h": skipped,
                }
            )
    return rows


def top_benefit(rows: list[dict[str, Any]]) -> float:
    useful = [row for row in rows if row["split"] == "test" and row["h"] < row["num_spec_tokens"]]
    if not useful:
        return float("nan")
    useful = sorted(useful, key=lambda row: row["predicted_benefit"], reverse=True)
    n = max(1, math.ceil(0.1 * len(useful)))
    return float(np.mean([row["actual_skip_tokens_if_chunk_h"] for row in useful[:n]]))


def write_sanity(method_frame: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# Sanity Checks: {normalize_dataset_label(method_frame['dataset_label'].iloc[0])}/{normalize_model_label(method_frame['model_label'].iloc[0])}/{method_frame['method'].iloc[0]}",
        "",
    ]
    required = [
        "run_id", "dataset_label", "model_label", "method", "request_id", "step_id", "draft_position",
        "num_spec_tokens", "token_id", "draft_selected_prob",
        "draft_selected_logprob", "draft_margin_logprob", "draft_entropy",
        "accepted_local", "reached",
    ]
    missing = [col for col in required if col not in method_frame.columns]
    lines.append(f"- rows: {len(method_frame)}")
    lines.append(f"- requests: {method_frame['request_id'].nunique()}")
    lines.append(f"- missing required columns: {missing or 'none'}")
    if "token_id_match" in method_frame:
        mismatch = int((method_frame["token_id_match"] == False).sum())  # noqa: E712
    else:
        mismatch = 0
    lines.append(f"- token_id mismatches: {mismatch}")
    invalid_prob = int(((method_frame["draft_selected_prob"] < 0) | (method_frame["draft_selected_prob"] > 1)).sum())
    invalid_logprob = int((method_frame["draft_selected_logprob"] > 1e-8).sum())
    invalid_margin = int((method_frame["draft_margin_logprob"] < -1e-8).sum())
    invalid_entropy = int((method_frame["draft_entropy"] < -1e-8).sum())
    lines.append(f"- invalid probabilities: {invalid_prob}")
    lines.append(f"- positive logprobs: {invalid_logprob}")
    lines.append(f"- negative margins: {invalid_margin}")
    lines.append(f"- negative entropies: {invalid_entropy}")
    usable = method_frame[(method_frame["reached"] == 1) & method_frame["accepted_local"].notna()]
    lines.append("")
    lines.append("## Acceptance By Position")
    for (k, pos), group in usable.groupby(["num_spec_tokens", "draft_position"]):
        lines.append(f"- K={int(k)} pos={int(pos)} rate={group['accepted_local'].mean():.4f} n={len(group)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _font():
    return ImageFont.load_default()


def draw_lines(path: Path, title: str, x_label: str, y_label: str, series: list[tuple[str, list[float], list[float]]], y_min: float = 0.0, y_max: float = 1.0) -> None:
    width, height = 1100, 680
    margin_l, margin_r, margin_t, margin_b = 90, 190, 70, 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((margin_l, 25), title, fill="black", font=font)
    xs_all = [x for _, xs, _ in series for x in xs]
    if not xs_all:
        image.save(path)
        return
    x_min, x_max = min(xs_all), max(xs_all)
    if x_min == x_max:
        x_max = x_min + 1
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def sx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_t + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#222222")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="#222222")
    for i in range(6):
        y = y_min + i * (y_max - y_min) / 5
        py = sy(y)
        draw.line((margin_l, py, margin_l + plot_w, py), fill="#eeeeee")
        draw.text((25, py - 6), f"{y:.2f}", fill="#333333", font=font)
    draw.text((margin_l + plot_w // 2 - 30, height - 40), x_label, fill="black", font=font)
    draw.text((15, 35), y_label, fill="black", font=font)
    label_items: list[tuple[float, float, str, str]] = []
    for idx, (label, xs, ys) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        points = [(sx(float(x)), sy(float(y))) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        if points:
            label_items.append((points[-1][0], points[-1][1], label, color))
    if label_items:
        label_items.sort(key=lambda item: item[1])
        min_gap = 16
        adjusted: list[list[float | str]] = []
        for x, y, label, color in label_items:
            new_y = y if not adjusted else max(y, float(adjusted[-1][1]) + min_gap)
            adjusted.append([x, new_y, label, color])
        overflow = float(adjusted[-1][1]) - (margin_t + plot_h)
        if overflow > 0:
            for item in adjusted:
                item[1] = float(item[1]) - overflow
        label_x = margin_l + plot_w + 14
        for x, y, label, color in adjusted:
            y = float(y)
            draw.line((float(x) + 5, y, label_x - 4, y), fill=str(color))
            draw.text((label_x, y - 6), str(label), fill=str(color), font=font)
    image.save(path)


def draw_confidence_fit_facets(path: Path, rows: pd.DataFrame) -> None:
    width, height = 1200, 820
    margin_l, margin_r, margin_t, margin_b = 80, 35, 85, 70
    gap_x, gap_y = 55, 70
    cols = 2
    groups = list(rows.groupby(["dataset_label", "model_label", "method", "num_spec_tokens"]))
    panel_count = max(1, len(groups))
    panel_rows = int(np.ceil(panel_count / cols))
    plot_w = (width - margin_l - margin_r - gap_x * (cols - 1)) / cols
    plot_h = (height - margin_t - margin_b - gap_y * (panel_rows - 1)) / panel_rows
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    draw.text((margin_l, 25), "Confidence Fit Curve", fill="black", font=font)
    draw.text(
        (margin_l, 45),
        "blue points: binned actual acceptance; orange line: calibrated fit",
        fill="#333333",
        font=font,
    )

    def dashed_line(points: list[tuple[float, float]], color: str) -> None:
        if len(points) < 2:
            return
        for a, b in zip(points, points[1:]):
            x1, y1 = a
            x2, y2 = b
            length = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
            dash = 8.0
            steps = int(length // dash)
            for i in range(0, steps + 1, 2):
                start = i * dash / length
                end = min((i + 1) * dash / length, 1.0)
                draw.line(
                    (
                        x1 + (x2 - x1) * start,
                        y1 + (y2 - y1) * start,
                        x1 + (x2 - x1) * end,
                        y1 + (y2 - y1) * end,
                    ),
                    fill=color,
                    width=3,
                )

    for idx, ((dataset_label, model_label, method, k), group) in enumerate(groups):
        col = idx % cols
        row = idx // cols
        x0 = margin_l + col * (plot_w + gap_x)
        y0 = margin_t + row * (plot_h + gap_y)
        x1 = x0 + plot_w
        y1 = y0 + plot_h

        def sx(x: float) -> float:
            return x0 + min(max(x, 0.0), 1.0) * plot_w

        def sy(y: float) -> float:
            return y0 + (1.0 - min(max(y, 0.0), 1.0)) * plot_h

        draw.rectangle((x0, y0, x1, y1), outline="#222222")
        for i in range(6):
            val = i / 5
            px = sx(val)
            py = sy(val)
            draw.line((x0, py, x1, py), fill="#eeeeee")
            draw.line((px, y0, px, y1), fill="#f4f4f4")
            draw.text((x0 - 34, py - 6), f"{val:.1f}", fill="#444444", font=font)
            draw.text((px - 8, y1 + 8), f"{val:.1f}", fill="#444444", font=font)

        title = f"{dataset_label}/{model_label}/{method}/K={int(k)}"
        draw.text((x0, y0 - 20), title, fill="#111111", font=font)
        group = group.sort_values("confidence")
        actual = [
            (sx(float(x)), sy(float(y)))
            for x, y in zip(group["confidence"], group["actual_acceptance"])
            if np.isfinite(x) and np.isfinite(y)
        ]
        fitted = [
            (sx(float(x)), sy(float(y)))
            for x, y in zip(group["confidence"], group["fitted_acceptance"])
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(actual) >= 2:
            draw.line(actual, fill="#1f77b4", width=3)
        for point in actual:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#1f77b4")
        dashed_line(fitted, "#ff7f0e")

    draw.text((width // 2 - 85, height - 35), "DLM selected-token probability", fill="black", font=font)
    image.save(path)


def draw_all_figures(root: Path, token_df: pd.DataFrame, calibrated: pd.DataFrame, prefix_rows: list[dict[str, Any]]) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    usable = calibrated[(calibrated["reached"] == 1) & calibrated["accepted_local"].notna()]
    acc_rows = []
    for (dataset_label, model_label, method, k, pos), group in usable.groupby(["dataset_label", "model_label", "method", "num_spec_tokens", "draft_position"]):
        acc_rows.append({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k, "draft_position": pos, "acceptance": group["accepted_local"].mean(), "n": len(group)})
    save_csv(fig_dir / "acceptance_by_position_data.csv", acc_rows)
    series = []
    for (dataset_label, model_label, method, k), group in pd.DataFrame(acc_rows).groupby(["dataset_label", "model_label", "method", "num_spec_tokens"]):
        group = group.sort_values("draft_position")
        series.append((group_name({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k}), group["draft_position"].tolist(), group["acceptance"].tolist()))
    draw_lines(fig_dir / "acceptance_by_position.png", "Local Acceptance By Draft Position", "draft position", "acceptance", series)

    bin_rows = []
    for feature in ["draft_selected_prob", "draft_margin_logprob", "draft_entropy"]:
        values = usable[feature].astype(float)
        if values.nunique() < 3:
            continue
        bins = pd.qcut(values, q=min(10, values.nunique()), duplicates="drop")
        tmp = usable.assign(_bin=bins)
        for (dataset_label, model_label, method, k, interval), group in tmp.groupby(["dataset_label", "model_label", "method", "num_spec_tokens", "_bin"], observed=True):
            bin_rows.append({"feature": feature, "dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k, "bin_mid": float(interval.mid), "acceptance": group["accepted_local"].mean(), "n": len(group)})
    save_csv(fig_dir / "confidence_bins_data.csv", bin_rows)
    conf_series = []
    for (feature, dataset_label, model_label, method, k), group in pd.DataFrame(bin_rows).groupby(["feature", "dataset_label", "model_label", "method", "num_spec_tokens"]):
        group = group.sort_values("bin_mid")
        label = f"{group_name({'dataset_label': dataset_label, 'model_label': model_label, 'method': method, 'num_spec_tokens': k})} {feature.replace('draft_', '')}"
        conf_series.append((label, group["bin_mid"].tolist(), group["acceptance"].tolist()))
    draw_lines(fig_dir / "confidence_bins.png", "Confidence Feature Bins", "feature bin midpoint", "acceptance", conf_series)

    rel_rows = []
    test = usable[usable["split"] == "test"]
    if len(test):
        bins = pd.cut(test["pred_accept_prob_full"].astype(float), bins=np.linspace(0, 1, 11), include_lowest=True)
        tmp = test.assign(_bin=bins)
        for (dataset_label, model_label, method, k, interval), group in tmp.groupby(["dataset_label", "model_label", "method", "num_spec_tokens", "_bin"], observed=True):
            rel_rows.append({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k, "pred_mean": group["pred_accept_prob_full"].mean(), "actual": group["accepted_local"].mean(), "n": len(group), "bin_mid": float(interval.mid)})
    save_csv(fig_dir / "reliability_data.csv", rel_rows)
    save_csv(fig_dir / "calibration_curve_data.csv", rel_rows)
    rel_series = [("ideal", [0, 1], [0, 1])]
    if rel_rows:
        for (dataset_label, model_label, method, k), group in pd.DataFrame(rel_rows).groupby(["dataset_label", "model_label", "method", "num_spec_tokens"]):
            group = group.sort_values("pred_mean")
            rel_series.append((group_name({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k}), group["pred_mean"].tolist(), group["actual"].tolist()))
    draw_lines(fig_dir / "reliability.png", "Reliability Diagram", "predicted acceptance", "actual acceptance", rel_series)
    draw_lines(fig_dir / "calibration_curve.png", "Calibration Curve", "predicted acceptance", "actual acceptance", rel_series)

    fit_rows = []
    for (dataset_label, model_label, method, k), group in usable.groupby(["dataset_label", "model_label", "method", "num_spec_tokens"]):
        values = group["draft_selected_prob"].astype(float)
        if values.nunique() < 3:
            continue
        bins = pd.qcut(values, q=min(12, values.nunique()), duplicates="drop")
        tmp = group.assign(_bin=bins)
        for interval, bin_group in tmp.groupby("_bin", observed=True):
            fit_rows.append({
                "model_label": model_label,
                "dataset_label": dataset_label,
                "method": method,
                "num_spec_tokens": k,
                "confidence": bin_group["draft_selected_prob"].mean(),
                "actual_acceptance": bin_group["accepted_local"].mean(),
                "fitted_acceptance": bin_group["pred_accept_prob_full"].mean(),
                "n": len(bin_group),
                "bin_mid": float(interval.mid),
            })
    save_csv(fig_dir / "confidence_fit_curve_data.csv", fit_rows)
    if fit_rows:
        draw_confidence_fit_facets(fig_dir / "confidence_fit_curve.png", pd.DataFrame(fit_rows))
    else:
        draw_lines(fig_dir / "confidence_fit_curve.png", "Confidence Fit Curve", "DLM selected-token probability", "local acceptance", [])

    save_csv(fig_dir / "reject_within_h_data.csv", prefix_rows)
    reject_series = []
    if prefix_rows:
        prefix_df = pd.DataFrame(prefix_rows)
        bins = pd.cut(prefix_df["pred_reject_within_h"].astype(float), bins=np.linspace(0, 1, 11), include_lowest=True)
        tmp = prefix_df.assign(_bin=bins)
        reject_plot_rows = []
        for (dataset_label, model_label, method, k, h, interval), group in tmp.groupby(["dataset_label", "model_label", "method", "num_spec_tokens", "h", "_bin"], observed=True):
            reject_plot_rows.append({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k, "h": h, "pred": group["pred_reject_within_h"].mean(), "actual": group["actual_reject_within_h"].mean(), "n": len(group)})
        save_csv(fig_dir / "reject_within_h_plot_data.csv", reject_plot_rows)
        for (dataset_label, model_label, method, k, h), group in pd.DataFrame(reject_plot_rows).groupby(["dataset_label", "model_label", "method", "num_spec_tokens", "h"]):
            group = group.sort_values("pred")
            reject_series.append((f"{group_name({'dataset_label': dataset_label, 'model_label': model_label, 'method': method, 'num_spec_tokens': k})} h{h}", group["pred"].tolist(), group["actual"].tolist()))
    draw_lines(fig_dir / "reject_within_h.png", "Reject Within h Calibration", "predicted reject probability", "actual reject frequency", reject_series)

    benefit_rows = []
    if prefix_rows:
        prefix_df = pd.DataFrame(prefix_rows)
        prefix_df = prefix_df[prefix_df["h"] < prefix_df["num_spec_tokens"]]
        if len(prefix_df):
            bins = pd.qcut(prefix_df["predicted_benefit"], q=min(10, prefix_df["predicted_benefit"].nunique()), duplicates="drop")
            tmp = prefix_df.assign(_bin=bins)
            for (dataset_label, model_label, method, k, interval), group in tmp.groupby(["dataset_label", "model_label", "method", "num_spec_tokens", "_bin"], observed=True):
                benefit_rows.append({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k, "predicted_benefit": group["predicted_benefit"].mean(), "actual_skipped": group["actual_skip_tokens_if_chunk_h"].mean(), "n": len(group)})
    save_csv(fig_dir / "chunk_benefit_data.csv", benefit_rows)
    benefit_series = []
    if benefit_rows:
        for (dataset_label, model_label, method, k), group in pd.DataFrame(benefit_rows).groupby(["dataset_label", "model_label", "method", "num_spec_tokens"]):
            group = group.sort_values("predicted_benefit")
            benefit_series.append((group_name({"dataset_label": dataset_label, "model_label": model_label, "method": method, "num_spec_tokens": k}), group["predicted_benefit"].tolist(), group["actual_skipped"].tolist()))
    draw_lines(fig_dir / "chunk_benefit.png", "Predicted Chunk Benefit", "predicted benefit", "actual skipped tokens", benefit_series, y_min=0.0, y_max=max(1.0, max([row["actual_skipped"] for row in benefit_rows], default=1.0)))


def write_report(root: Path, summaries: list[dict[str, Any]], token_df: pd.DataFrame) -> None:
    lines = [
        "# SpecLink Confidence Acceptance Experiment",
        "",
        "## Goal",
        "This experiment tests whether DLM draft-token confidence predicts TLM local acceptance. It does not implement chunked verification scheduling.",
        "",
        "## Setup",
        f"- trace rows: {len(token_df)}",
        f"- datasets: {', '.join(sorted(map(str, token_df['dataset_label'].unique())))}",
        f"- models: {', '.join(sorted(map(str, token_df['model_label'].unique())))}",
        f"- methods: {', '.join(sorted(map(str, token_df['method'].unique())))}",
        f"- num_spec_tokens: {', '.join(map(str, sorted(token_df['num_spec_tokens'].unique())))}",
        "",
        "## Instrumentation",
        "- DLM confidence is collected in `SpecDecodeBaseProposer.propose()` from proposer logits before the draft token ids are returned.",
        "- TLM acceptance labels are collected in `vllm.v1.sample.rejection_sampler.RejectionSampler.forward()` from rejection-sampler output rows.",
        "- Records are aligned by vLLM request id and a per-request speculative step counter. Draft records are buffered until their verifier step is sampled.",
        "- `token_text` is intentionally left null to avoid tokenizer overhead in the hot path.",
        "",
        "## Main Results",
    ]
    for row in summaries:
        lines.extend(
            [
                f"### {group_name(row)}",
                f"- prompts: {row['num_prompts']}",
                f"- steps: {row['num_steps']}",
                f"- reached tokens: {row['num_reached_tokens']}",
                f"- overall local acceptance: {row.get('acceptance_rate_overall_reached', float('nan')):.4f}",
                f"- best predictor: {row.get('best_predictor')}",
                f"- full AUROC/ECE/Brier: {row.get('auroc', float('nan')):.4f} / {row.get('ece', float('nan')):.4f} / {row.get('brier', float('nan')):.4f}",
                f"- evidence level: {row.get('evidence_level')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Implication For SpecLink",
            "Use the evidence level in `summary.csv` as the decision gate. Strong or Moderate supports using DLM confidence as a chunk-size feature; Weak means the scheduler should fall back to position priors or online target-side acceptance statistics.",
            "",
            "## Visualizations",
            "- `figures/calibration_curve.png` plots predicted local acceptance against empirical acceptance with an ideal diagonal.",
            "- `figures/confidence_fit_curve.png` plots DLM selected-token probability against empirical and fitted local acceptance curves.",
            "- Figure labels are placed outside the plot area instead of using an overlapping legend.",
            "",
            "## Files",
            "- `trace/`: raw token-level JSONL",
            "- `parsed/`: cleaned token-level CSV and sanity checks",
            "- `calibration/`: predictions, model parameters, metric JSON",
            "- `figures/`: PNG figures and their source CSV files",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(root: Path) -> None:
    parsed_dir = root / "parsed"
    cal_dir = root / "calibration"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    token_df = load_trace(root)
    summaries: list[dict[str, Any]] = []
    calibrated_parts: list[pd.DataFrame] = []
    prefix_rows_all: list[dict[str, Any]] = []

    for (dataset_label, model_label, method), method_frame in token_df.groupby(["dataset_label", "model_label", "method"]):
        prefix = file_prefix(
            normalize_dataset_label(dataset_label),
            normalize_model_label(model_label),
            str(method),
        )
        method_frame.to_csv(parsed_dir / f"{prefix}_token_level.csv", index=False)
        write_sanity(method_frame, parsed_dir / f"{prefix}_sanity.md")
        method_summaries = []
        method_parts: list[pd.DataFrame] = []
        for (_, _, _, k), group in method_frame.groupby(["dataset_label", "model_label", "method", "num_spec_tokens"]):
            calibrated, summary, prefix_rows = calibrate_group(group, cal_dir)
            calibrated.to_csv(cal_dir / f"{prefix}_k{int(k)}_calibrated.csv", index=False)
            calibrated_parts.append(calibrated)
            method_parts.append(calibrated)
            prefix_rows_all.extend(prefix_rows)
            summaries.append(summary)
            method_summaries.append(summary)
        pd.concat(method_parts).to_csv(
            cal_dir / f"{prefix}_calibrated.csv",
            index=False,
        )
        (cal_dir / f"{prefix}_summary.json").write_text(
            json.dumps(method_summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    calibrated_all = pd.concat(calibrated_parts, ignore_index=True)
    summary_flat = []
    for row in summaries:
        flat = {k: v for k, v in row.items() if k != "predictors"}
        summary_flat.append(flat)
    pd.DataFrame(summary_flat).to_csv(root / "summary.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    draw_all_figures(root, token_df, calibrated_all, prefix_rows_all)
    write_report(root, summaries, token_df)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "trace").mkdir()
        rows = []
        for method in ["eagle3", "peagle"]:
            for req in range(8):
                first_reject = 3 if req % 2 else None
                for pos in range(1, 5):
                    reached = first_reject is None or pos <= first_reject
                    accepted = None if not reached else int(first_reject is None or pos < first_reject)
                    rows.append({
                        "run_id": "selftest",
                        "dataset_label": "selftest_dataset",
                        "model_label": "selftest_model",
                        "method": method,
                        "request_id": f"speclink-{method}-k4-p{req:06d}",
                        "step_id": 1,
                        "draft_position": pos,
                        "num_spec_tokens": 4,
                        "token_id": pos,
                        "context_len": 10,
                        "generated_len_so_far": req,
                        "draft_selected_logprob": -0.1 * pos,
                        "draft_selected_prob": math.exp(-0.1 * pos),
                        "draft_top1_logprob": -0.1 * pos,
                        "draft_top2_logprob": -1.0,
                        "draft_margin_logprob": 1.0 - 0.1 * pos,
                        "draft_entropy": 1.0 + pos,
                        "reached": int(reached),
                        "accepted_local": accepted,
                        "accepted_global": int(accepted == 1),
                        "first_reject_position": first_reject,
                        "num_accepted_in_step": 4 if first_reject is None else first_reject - 1,
                    })
        with (root / "trace" / "self_trace.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        analyze(root)
        assert (root / "summary.csv").exists()
        assert (root / "figures" / "acceptance_by_position.png").exists()
    print("[INFO] self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_root is None:
        raise SystemExit("output_root is required unless --self-test is used")
    analyze(args.output_root)


if __name__ == "__main__":
    main()
