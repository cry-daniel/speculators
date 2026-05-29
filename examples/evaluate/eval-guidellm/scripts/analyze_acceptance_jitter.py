#!/usr/bin/env python3
"""Summarize and plot per-step accepted draft-token counts."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CASE_ORDER = [
    ("qwen3_8b", "peagle", "Qwen3 P-EAGLE"),
    ("qwen3_8b", "eagle3", "Qwen3 EAGLE3"),
    ("llama3_1_8b", "eagle3", "Llama3 EAGLE3"),
]
K_ORDER = [8, 12, 16]
WORKLOAD_ORDER = ["math", "mtbench", "synthetic_1000x1000"]
WORKLOAD_TITLES = {
    "math": "Math",
    "mtbench": "MTBench",
    "synthetic_1000x1000": "Synthetic 1000/1000",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_trace_files(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(
        path
        for path in input_root.rglob("*.jsonl")
        if path.parent.name == "trace" or path.name.endswith("_trace.jsonl")
    )


def first_non_null(values: pd.Series) -> Any:
    usable = values.dropna()
    if len(usable):
        return usable.iloc[0]
    return None


def compute_predicted_useful_accepted_from_prefix_probs(
    probs: pd.Series,
    max_tokens: int,
) -> float:
    if max_tokens <= 0:
        return 0.0
    if probs is None:
        return float("nan")
    values = probs.tolist()[:max_tokens]
    if not values:
        return float("nan")

    expected_accepted = 0.0
    survival = 1.0
    for value in values:
        if value is None or not np.isfinite(float(value)):
            return float("nan")
        p = float(value)
        if p <= 0.0:
            break
        if p >= 1.0:
            survival *= 1.0
        else:
            survival *= max(0.0, min(1.0, p))
        expected_accepted += survival

    useful = 1.0 + expected_accepted
    return min(max(useful, 0.0), float(max_tokens))


def normalized_dataset(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return "math"


def normalized_model(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return "qwen3_8b"


def load_trace(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in find_trace_files(input_root):
        rows.extend(load_jsonl(path))
    if not rows:
        raise SystemExit(f"No trace JSONL rows found under {input_root}")
    frame = pd.DataFrame(rows)
    for col, default in [
        ("dataset_label", "math"),
        ("model_label", "qwen3_8b"),
        ("method", "eagle3"),
    ]:
        if col not in frame:
            frame[col] = default
    frame["dataset_label"] = frame["dataset_label"].map(normalized_dataset)
    frame["model_label"] = frame["model_label"].map(normalized_model)
    for col in [
        "step_id",
        "draft_position",
        "num_spec_tokens",
        "num_accepted_in_step",
        "num_scheduled_draft_tokens",
        "draft_selected_prob",
        "first_reject_position",
        "dataset_index",
        "prompt_id",
        "context_len",
        "generated_len_so_far",
    ]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def make_step_frame(token_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "dataset_label",
        "model_label",
        "method",
        "num_spec_tokens",
        "request_id",
        "step_id",
    ]
    for key, group in token_df.groupby(group_cols, dropna=False):
        dataset_label, model_label, method, k, request_id, step_id = key
        group = group.sort_values("draft_position")
        scheduled = first_non_null(group.get("num_scheduled_draft_tokens", pd.Series(dtype=float)))
        if scheduled is None or not np.isfinite(float(scheduled)):
            scheduled = int(group["draft_position"].max())
        scheduled = int(scheduled)
        accepted = first_non_null(group.get("num_accepted_in_step", pd.Series(dtype=float)))
        if accepted is None or not np.isfinite(float(accepted)):
            first_reject = first_non_null(group.get("first_reject_position", pd.Series(dtype=float)))
            accepted = scheduled if first_reject is None else max(int(first_reject) - 1, 0)
        accepted = max(0, min(int(accepted), scheduled))
        first_reject = first_non_null(group.get("first_reject_position", pd.Series(dtype=float)))
        first_reject_int = None if first_reject is None or not np.isfinite(float(first_reject)) else int(first_reject)
        request_order = first_non_null(group.get("dataset_index", pd.Series(dtype=float)))
        if request_order is None or not np.isfinite(float(request_order)):
            request_order = first_non_null(group.get("prompt_id", pd.Series(dtype=float)))

        prob_series = group["draft_selected_prob"] if "draft_selected_prob" in group else pd.Series(dtype=float)
        scheduled_limit = int(max(scheduled, 0))
        token_limit = min(scheduled_limit, int(k))
        pred_useful_accepted = compute_predicted_useful_accepted_from_prefix_probs(
            prob_series.head(token_limit),
            token_limit,
        )
        if np.isfinite(pred_useful_accepted):
            pred_num_accepted = pred_useful_accepted
            pred_redundancy = (int(k) - pred_useful_accepted) / int(k) if int(k) > 0 else float("nan")
        else:
            pred_num_accepted = float("nan")
            pred_redundancy = float("nan")

        rows.append(
            {
                "dataset_label": str(dataset_label),
                "model_label": str(model_label),
                "method": str(method),
                "case_label": case_label(str(model_label), str(method)),
                "num_spec_tokens": int(k),
                "request_id": str(request_id),
                "request_order": int(request_order) if request_order is not None and np.isfinite(float(request_order)) else -1,
                "step_id": int(step_id),
                "num_scheduled_draft_tokens": scheduled,
                "num_accepted": accepted,
                "accept_fraction": accepted / scheduled if scheduled else float("nan"),
                "first_reject_position": first_reject_int,
                "pred_num_accepted": pred_num_accepted,
                "pred_redundancy": pred_redundancy,
                "context_len": first_non_null(group.get("context_len", pd.Series(dtype=float))),
                "generated_len_so_far": first_non_null(group.get("generated_len_so_far", pd.Series(dtype=float))),
            }
        )
    step_df = pd.DataFrame(rows)
    if len(step_df):
        step_df = step_df.sort_values(
            [
                "dataset_label",
                "model_label",
                "method",
                "num_spec_tokens",
                "request_order",
                "request_id",
                "step_id",
            ],
            kind="stable",
        ).reset_index(drop=True)
        step_df["global_step_index"] = (
            step_df.groupby(["dataset_label", "model_label", "method", "num_spec_tokens"])
            .cumcount()
            .astype(int)
        )
    return step_df


def case_label(model_label: str, method: str) -> str:
    for model, case_method, label in CASE_ORDER:
        if model == model_label and case_method == method:
            return label
    return f"{model_label} {method}".strip()


def summarize(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_label, model_label, method, k), group in step_df.groupby(
        ["dataset_label", "model_label", "method", "num_spec_tokens"]
    ):
        accepted = group["num_accepted"].astype(float).to_numpy()
        scheduled = group["num_scheduled_draft_tokens"].astype(float).to_numpy()
        diffs = []
        for _, req_group in group.groupby("request_id"):
            vals = req_group.sort_values("step_id")["num_accepted"].astype(float).to_numpy()
            if len(vals) > 1:
                diffs.extend(np.abs(np.diff(vals)).tolist())
        row: dict[str, Any] = {
            "dataset_label": dataset_label,
            "model_label": model_label,
            "method": method,
            "case_label": case_label(model_label, method),
            "num_spec_tokens": int(k),
            "num_requests": int(group["request_id"].nunique()),
            "num_decode_steps": int(len(group)),
            "avg_scheduled_draft_tokens": float(np.mean(scheduled)) if len(scheduled) else float("nan"),
            "mean_accepted": float(np.mean(accepted)) if len(accepted) else float("nan"),
            "std_accepted": float(np.std(accepted, ddof=0)) if len(accepted) else float("nan"),
            "median_accepted": float(np.median(accepted)) if len(accepted) else float("nan"),
            "p10_accepted": float(np.quantile(accepted, 0.10)) if len(accepted) else float("nan"),
            "p90_accepted": float(np.quantile(accepted, 0.90)) if len(accepted) else float("nan"),
            "p_zero_accepted": float(np.mean(accepted == 0)) if len(accepted) else float("nan"),
            "p_full_prefix_accepted": float(np.mean(accepted >= scheduled)) if len(accepted) else float("nan"),
            "mean_abs_step_delta": float(np.mean(diffs)) if diffs else float("nan"),
        }
        for h in [2, 4]:
            if h <= int(k):
                fail = float(np.mean(accepted < h)) if len(accepted) else float("nan")
                row[f"p_accepted_lt_{h}"] = fail
                row[f"p_accepted_ge_{h}"] = 1.0 - fail if np.isfinite(fail) else float("nan")
                row[f"fixed{h}_suffix_forward_possible_rate"] = row[f"p_accepted_ge_{h}"]
            else:
                row[f"p_accepted_lt_{h}"] = float("nan")
                row[f"p_accepted_ge_{h}"] = float("nan")
                row[f"fixed{h}_suffix_forward_possible_rate"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["dataset_label", "model_label", "method", "num_spec_tokens"],
        kind="stable",
    )


def distribution(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_label, model_label, method, k), group in step_df.groupby(
        ["dataset_label", "model_label", "method", "num_spec_tokens"]
    ):
        counts = Counter(int(v) for v in group["num_accepted"])
        total = max(int(sum(counts.values())), 1)
        for accepted in range(int(k) + 1):
            count = counts.get(accepted, 0)
            rows.append(
                {
                    "dataset_label": dataset_label,
                    "model_label": model_label,
                    "method": method,
                    "case_label": case_label(model_label, method),
                    "num_spec_tokens": int(k),
                    "num_accepted": accepted,
                    "count": count,
                    "fraction": count / total,
                }
            )
    return pd.DataFrame(rows)


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], text: str, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font())
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.text((x, y), text, fill=fill, font=font())


def draw_dashed_hline(
    draw: ImageDraw.ImageDraw,
    x0: float,
    x1: float,
    y: float,
    color: str,
    dash: int = 8,
) -> None:
    x = x0
    while x < x1:
        draw.line((x, y, min(x + dash, x1), y), fill=color, width=2)
        x += dash * 2


def draw_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[float, float, float, float],
    group: pd.DataFrame,
    title: str,
    k: int,
    show_y_label: bool,
    show_x_label: bool,
) -> None:
    x0, y0, x1, y1 = panel
    title_h = 30
    margin_l = 44
    margin_r = 16
    margin_t = title_h + 12
    margin_b = 34
    px0, py0 = x0 + margin_l, y0 + margin_t
    px1, py1 = x1 - margin_r, y1 - margin_b
    draw.text((x0, y0 + 4), title, fill="#111111", font=font())
    draw.rectangle((px0, py0, px1, py1), outline="#222222")

    if group.empty:
        text_center(draw, (px0, py0, px1, py1), "missing", "#777777")
        return

    def sx(idx: float) -> float:
        max_x = max(float(len(group) - 1), 1.0)
        return px0 + idx / max_x * (px1 - px0)

    def sy(value: float) -> float:
        return py1 - min(max(value, 0.0), float(k)) / max(float(k), 1.0) * (py1 - py0)

    ticks = sorted(set([0, 2, 4, k]))
    for tick in ticks:
        if tick > k:
            continue
        y = sy(float(tick))
        draw.line((px0, y, px1, y), fill="#eeeeee")
        draw.text((x0 + 8, y - 6), str(tick), fill="#444444", font=font())

    thresholds = [(2, "#b23a48"), (4, "#2f6f9f")]
    for h, color in thresholds:
        if h <= k:
            y = sy(float(h))
            draw_dashed_hline(draw, px0, px1, y, color)
            draw.text((px1 - 28, y - 13), f"h={h}", fill=color, font=font())

    values = group["num_accepted"].astype(float).tolist()
    points = [(sx(float(idx)), sy(value)) for idx, value in enumerate(values)]
    if len(points) >= 2:
        draw.line(points, fill="#0b7285", width=2)
    step = max(1, len(points) // 250)
    for point in points[::step]:
        draw.ellipse(
            (point[0] - 1.6, point[1] - 1.6, point[0] + 1.6, point[1] + 1.6),
            fill="#0b7285",
        )

    if show_y_label:
        draw.text((x0 + 2, py0 - 18), "accepted", fill="#111111", font=font())
    if show_x_label:
        draw.text((px0 + 70, y1 - 22), "decode step", fill="#111111", font=font())


def draw_workload_grid(path: Path, step_df: pd.DataFrame, workload: str) -> None:
    width, height = 1800, 1080
    margin_l, margin_r, margin_t, margin_b = 80, 40, 86, 40
    gap_x, gap_y = 42, 48
    rows, cols = len(CASE_ORDER), len(K_ORDER)
    panel_w = (width - margin_l - margin_r - gap_x * (cols - 1)) / cols
    panel_h = (height - margin_t - margin_b - gap_y * (rows - 1)) / rows
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = f"Accepted draft tokens per decode step: {WORKLOAD_TITLES.get(workload, workload)}"
    draw.text((margin_l, 24), title, fill="#111111", font=font())
    draw.text(
        (margin_l, 48),
        "Dashed lines mark fixed h=2 and h=4 boundaries. Lower crossings mean a fixed boundary would often hit rejection before the suffix.",
        fill="#333333",
        font=font(),
    )

    workload_df = step_df[step_df["dataset_label"] == workload]
    for row_idx, (model_label, method, label) in enumerate(CASE_ORDER):
        for col_idx, k in enumerate(K_ORDER):
            x0 = margin_l + col_idx * (panel_w + gap_x)
            y0 = margin_t + row_idx * (panel_h + gap_y)
            panel = (x0, y0, x0 + panel_w, y0 + panel_h)
            group = workload_df[
                (workload_df["model_label"] == model_label)
                & (workload_df["method"] == method)
                & (workload_df["num_spec_tokens"] == k)
            ].sort_values(["request_order", "request_id", "step_id"], kind="stable")
            draw_panel(
                draw,
                panel,
                group,
                f"{label} | K={k}",
                k,
                show_y_label=col_idx == 0,
                show_x_label=row_idx == rows - 1,
            )
    image.save(path)


def write_report(root: Path, summary: pd.DataFrame, step_df: pd.DataFrame) -> None:
    lines = [
        "# Acceptance Jitter Report",
        "",
        "This report isolates per-decode-step accepted draft-token counts. It is meant to answer whether a fixed boundary such as 2 or 4 is stable enough before considering end-to-end latency.",
        "",
        f"- decode steps: {len(step_df)}",
        f"- workloads: {', '.join(sorted(step_df['dataset_label'].unique()))}",
        f"- cases: {', '.join(sorted(step_df['case_label'].unique()))}",
        f"- K values: {', '.join(map(str, sorted(step_df['num_spec_tokens'].unique())))}",
        "",
        "## Key Columns",
        "",
        "- `p_accepted_lt_2`: fraction of decode steps where fewer than 2 draft tokens were accepted.",
        "- `p_accepted_lt_4`: fraction of decode steps where fewer than 4 draft tokens were accepted.",
        "- `mean_abs_step_delta`: average absolute change in accepted count between adjacent decode steps within the same request.",
        "",
        "## Fixed Boundary Reading",
        "",
        "A fixed h is theoretically fragile when `p_accepted_lt_h` is high: the verifier would often discover rejection before the suffix boundary, so a planned suffix forward would not be useful on those steps.",
        "",
        "## Compact Summary",
        "",
    ]
    compact_cols = [
        "dataset_label",
        "case_label",
        "num_spec_tokens",
        "num_decode_steps",
        "mean_accepted",
        "std_accepted",
        "p_accepted_lt_2",
        "p_accepted_lt_4",
        "p_full_prefix_accepted",
    ]
    for _, row in summary[compact_cols].iterrows():
        lines.append(
            "- "
            f"{row['dataset_label']} / {row['case_label']} / K={int(row['num_spec_tokens'])}: "
            f"mean={row['mean_accepted']:.2f}, std={row['std_accepted']:.2f}, "
            f"P(<2)={row['p_accepted_lt_2']:.2f}, P(<4)={row['p_accepted_lt_4']:.2f}, "
            f"P(full)={row['p_full_prefix_accepted']:.2f}, steps={int(row['num_decode_steps'])}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `step_level_acceptance.csv`: one row per decode step.",
            "- `summary.csv`: fixed-boundary and jitter metrics by workload/case/K.",
            "- `accepted_count_distribution.csv`: empirical distribution over accepted count.",
            "- `figures/*_accepted_count_jitter.png`: workload-specific decode-step curves.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(input_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    token_df = load_trace(input_root)
    step_df = make_step_frame(token_df)
    if step_df.empty:
        raise SystemExit("No decode-step rows could be derived from traces")
    summary = summarize(step_df)
    dist = distribution(step_df)
    step_df.to_csv(output_root / "step_level_acceptance.csv", index=False)
    summary.to_csv(output_root / "summary.csv", index=False)
    dist.to_csv(output_root / "accepted_count_distribution.csv", index=False)
    for workload in WORKLOAD_ORDER:
        if (step_df["dataset_label"] == workload).any():
            draw_workload_grid(
                fig_dir / f"{workload}_accepted_count_jitter.png",
                step_df,
                workload,
            )
    write_report(output_root, summary, step_df)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        trace_dir = root / "case" / "trace"
        trace_dir.mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        for dataset in WORKLOAD_ORDER:
            for model, method, _ in CASE_ORDER:
                for k in K_ORDER:
                    for req in range(3):
                        for step in range(1, 8):
                            accepted = (req + step) % (k + 1)
                            first_reject = None if accepted >= k else accepted + 1
                            for pos in range(1, k + 1):
                                reached = first_reject is None or pos <= first_reject
                                rows.append(
                                    {
                                        "dataset_label": dataset,
                                        "model_label": model,
                                        "method": method,
                                        "request_id": f"speclink-{dataset}-{model}-{method}-k{k}-p{req:06d}",
                                        "step_id": step,
                                        "draft_position": pos,
                                        "num_spec_tokens": k,
                                        "num_scheduled_draft_tokens": k,
                                        "dataset_index": req,
                                        "context_len": 100 + step,
                                        "generated_len_so_far": step,
                                        "reached": int(reached),
                                        "accepted_local": int(pos <= accepted) if reached else None,
                                        "first_reject_position": first_reject,
                                        "num_accepted_in_step": accepted,
                                    }
                                )
        with (trace_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = root / "out"
        analyze(root, out)
        assert (out / "summary.csv").exists()
        assert (out / "step_level_acceptance.csv").exists()
        assert (out / "figures" / "math_accepted_count_jitter.png").exists()
    print("[INFO] self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", nargs="?", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.input_root is None:
        raise SystemExit("input_root is required unless --self-test is used")
    output_root = args.output_root or args.input_root
    analyze(args.input_root, output_root)


if __name__ == "__main__":
    main()
