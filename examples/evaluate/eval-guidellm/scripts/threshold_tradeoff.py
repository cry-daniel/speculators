#!/usr/bin/env python3
"""Analyze confidence-threshold tradeoffs from speculative decoding traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60"
DEFAULT_WORKLOADS = "math,mtbench"
DEFAULT_MODELS = ""
DEFAULT_METHODS = ""
DEFAULT_NUM_SPEC_TOKENS = ""
CASE_ORDER = [
    ("qwen3_8b", "peagle", "Qwen3 P-EAGLE"),
    ("qwen3_8b", "eagle3", "Qwen3 EAGLE3"),
    ("llama3_1_8b", "eagle3", "Llama3 EAGLE3"),
]
WORKLOAD_TITLES = {"math": "Math", "mtbench": "MTBench"}
K_COLORS = {8: "#0b7285", 12: "#9a6700", 16: "#7b2cbf"}


@dataclass(frozen=True)
class DecodeStep:
    workload: str
    model_label: str
    method: str
    case_label: str
    num_spec_tokens: int
    request_id: str
    step_id: int
    actual_accept_tokens: int
    draft_probs: tuple[float, ...]


def case_label(model_label: str, method: str) -> str:
    for model, case_method, label in CASE_ORDER:
        if model == model_label and case_method == method:
            return label
    return f"{model_label} {method}".strip()


def parse_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item) for item in parse_list(value)})
    if not thresholds:
        raise SystemExit("At least one threshold is required")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise SystemExit(f"Threshold must be in [0, 1]: {threshold}")
    return thresholds


def parse_ints(value: str) -> list[int]:
    ints: list[int] = []
    for item in parse_list(value):
        try:
            ints.append(int(item))
        except ValueError as exc:
            raise SystemExit(f"Invalid integer in list: {item}") from exc
    return ints


def dataset_from_run_path(path: Path) -> str:
    parts = path.parts
    if "runs" not in parts:
        return ""
    idx = parts.index("runs") + 1
    if idx >= len(parts):
        return ""
    run_name = parts[idx]
    return run_name.split("_", 1)[0]


def trace_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [
        path
        for path in sorted(root.rglob("*.jsonl"))
        if path.parent.name == "trace" or path.name.endswith("_trace.jsonl")
    ]


def load_steps(
    root: Path,
    workloads: set[str],
    models: set[str],
    methods: set[str],
    num_spec_tokens: set[int],
) -> list[DecodeStep]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    required = {"draft_selected_prob", "num_accepted_in_step", "draft_position"}
    missing: set[str] = set()
    for path in trace_files(root):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                workload = str(row.get("dataset_label") or "")
                if not workload:
                    workload = dataset_from_run_path(path)
                if workload not in workloads:
                    continue
                for field in required:
                    if field not in row:
                        missing.add(field)
                if missing:
                    continue
                model_label = str(row.get("model_label") or "qwen3_8b")
                method = str(row.get("method") or "eagle3")
                if models and model_label not in models:
                    continue
                if methods and method not in methods:
                    continue
                k = int(row["num_spec_tokens"])
                if num_spec_tokens and k not in num_spec_tokens:
                    continue
                key = (
                    workload,
                    model_label,
                    method,
                    k,
                    str(row.get("request_id") or row.get("sequence_id") or ""),
                    int(row["step_id"]),
                )
                group = grouped.setdefault(
                    key,
                    {
                        "positions": [],
                        "probs": [],
                        "actual_accept_tokens": int(row["num_accepted_in_step"]),
                    },
                )
                prob = float(row["draft_selected_prob"])
                if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
                    raise SystemExit(f"Invalid draft_selected_prob={prob} in {path}")
                group["positions"].append(int(row["draft_position"]))
                group["probs"].append(prob)
    if missing:
        raise SystemExit(f"Trace rows are missing required fields: {sorted(missing)}")
    if not grouped:
        raise SystemExit(f"No matching trace rows found under {root}")

    steps: list[DecodeStep] = []
    for key, group in grouped.items():
        workload, model_label, method, k, request_id, step_id = key
        ordered_probs = tuple(
            prob for _, prob in sorted(zip(group["positions"], group["probs"]))
        )
        if not ordered_probs:
            continue
        actual_accept = max(0, min(int(group["actual_accept_tokens"]), int(k)))
        steps.append(
            DecodeStep(
                workload=str(workload),
                model_label=str(model_label),
                method=str(method),
                case_label=case_label(str(model_label), str(method)),
                num_spec_tokens=int(k),
                request_id=str(request_id),
                step_id=int(step_id),
                actual_accept_tokens=actual_accept,
                draft_probs=ordered_probs,
            )
        )
    return sorted(
        steps,
        key=lambda step: (
            step.workload,
            step.model_label,
            step.method,
            step.num_spec_tokens,
            step.request_id,
            step.step_id,
        ),
    )


def predicted_tokens(draft_probs: tuple[float, ...], threshold: float) -> int:
    confidence = 1.0
    predicted = 0
    for position, prob in enumerate(draft_probs, start=1):
        confidence *= prob
        if confidence < threshold:
            break
        predicted = position
    return predicted


def summarize_group(steps: list[DecodeStep], threshold: float) -> dict[str, Any]:
    n = len(steps)
    if n == 0:
        raise ValueError("Cannot summarize an empty group")
    error_count = 0
    mismatch_count = 0
    pred_sum = 0.0
    efficiency_sum = 0.0
    actual_sum = 0.0
    for step in steps:
        pred = predicted_tokens(step.draft_probs, threshold)
        pred_sum += pred
        efficiency_sum += pred / max(step.num_spec_tokens, 1)
        actual_sum += step.actual_accept_tokens
        if pred != step.actual_accept_tokens:
            mismatch_count += 1
        if pred > step.actual_accept_tokens:
            error_count += 1
    first = steps[0]
    return {
        "workload": first.workload,
        "case_label": first.case_label,
        "model_label": first.model_label,
        "method": first.method,
        "num_spec_tokens": first.num_spec_tokens,
        "threshold": threshold,
        "num_decode_steps": n,
        "error_probability": error_count / n,
        "mismatch_probability": mismatch_count / n,
        "compute_efficiency": efficiency_sum / n,
        "mean_pred_tokens": pred_sum / n,
        "mean_actual_accept_tokens": actual_sum / n,
    }


def pareto_flags(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["workload"],
            row["model_label"],
            row["method"],
            row["num_spec_tokens"],
        )
        grouped.setdefault(key, []).append(row)
    for group in grouped.values():
        for row in group:
            dominated = False
            for other in group:
                if other is row:
                    continue
                no_worse = (
                    other["error_probability"] <= row["error_probability"]
                    and other["compute_efficiency"] >= row["compute_efficiency"]
                )
                strictly_better = (
                    other["error_probability"] < row["error_probability"]
                    or other["compute_efficiency"] > row["compute_efficiency"]
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            row["is_pareto_optimal"] = not dominated


def analyze_steps(steps: list[DecodeStep], thresholds: list[float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[DecodeStep]] = {}
    for step in steps:
        key = (step.workload, step.model_label, step.method, step.num_spec_tokens)
        grouped.setdefault(key, []).append(step)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_steps = grouped[key]
        for threshold in thresholds:
            rows.append(summarize_group(group_steps, threshold))
    pareto_flags(rows)
    return sorted(
        rows,
        key=lambda row: (
            row["workload"],
            row["case_label"],
            row["num_spec_tokens"],
            row["threshold"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "workload",
        "case_label",
        "model_label",
        "method",
        "num_spec_tokens",
        "threshold",
        "num_decode_steps",
        "error_probability",
        "mismatch_probability",
        "compute_efficiency",
        "mean_pred_tokens",
        "mean_actual_accept_tokens",
        "is_pareto_optimal",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font())
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.text((x, y), text, fill=fill, font=font())


def draw_tradeoff(path: Path, rows: list[dict[str, Any]], workloads: list[str]) -> None:
    width, height = 1700, 780
    margin_l, margin_r, margin_t, margin_b = 70, 40, 80, 50
    gap_x, gap_y = 40, 56
    cols, panel_rows = len(CASE_ORDER), len(workloads)
    panel_w = (width - margin_l - margin_r - gap_x * (cols - 1)) / cols
    panel_h = (height - margin_t - margin_b - gap_y * (panel_rows - 1)) / panel_rows
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 24), "DLM confidence threshold tradeoff", fill="#111111", font=font())
    draw.text(
        (margin_l, 48),
        "x: overrun risk P(pred > actual), y: compute efficiency E(pred / K). "
        "Hollow points are Pareto-optimal thresholds.",
        fill="#333333",
        font=font(),
    )

    def sx(x0: float, x1: float, value: float) -> float:
        return x0 + min(max(value, 0.0), 1.0) * (x1 - x0)

    def sy(y0: float, y1: float, value: float) -> float:
        return y1 - min(max(value, 0.0), 1.0) * (y1 - y0)

    for row_idx, workload in enumerate(workloads):
        for col_idx, (model_label, method, label) in enumerate(CASE_ORDER):
            x0 = margin_l + col_idx * (panel_w + gap_x)
            y0 = margin_t + row_idx * (panel_h + gap_y)
            x1, y1 = x0 + panel_w, y0 + panel_h
            px0, py0 = x0 + 48, y0 + 34
            px1, py1 = x1 - 16, y1 - 34
            title = f"{WORKLOAD_TITLES.get(workload, workload)} / {label}"
            draw.text((x0, y0 + 4), title, fill="#111111", font=font())
            draw.rectangle((px0, py0, px1, py1), outline="#222222")
            for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x = sx(px0, px1, tick)
                y = sy(py0, py1, tick)
                draw.line((x, py0, x, py1), fill="#eeeeee")
                draw.line((px0, y, px1, y), fill="#eeeeee")
                draw.text((x - 8, py1 + 8), f"{tick:.2g}", fill="#555555", font=font())
                draw.text((x0 + 8, y - 6), f"{tick:.2g}", fill="#555555", font=font())
            subset = [
                row
                for row in rows
                if row["workload"] == workload
                and row["model_label"] == model_label
                and row["method"] == method
            ]
            if not subset:
                draw_text_center(draw, (px0, py0, px1, py1), "missing", "#777777")
                continue
            for k in sorted({int(row["num_spec_tokens"]) for row in subset}):
                group = sorted(
                    [row for row in subset if int(row["num_spec_tokens"]) == k],
                    key=lambda row: float(row["error_probability"]),
                )
                color = K_COLORS.get(k, "#0b7285")
                points = [
                    (
                        sx(px0, px1, float(row["error_probability"])),
                        sy(py0, py1, float(row["compute_efficiency"])),
                    )
                    for row in group
                ]
                if len(points) >= 2:
                    draw.line(points, fill=color, width=2)
                for point, row in zip(points, group):
                    radius = 4 if row["is_pareto_optimal"] else 2
                    fill = "white" if row["is_pareto_optimal"] else color
                    draw.ellipse(
                        (
                            point[0] - radius,
                            point[1] - radius,
                            point[0] + radius,
                            point[1] + radius,
                        ),
                        outline=color,
                        fill=fill,
                        width=2 if row["is_pareto_optimal"] else 1,
                    )
                label_point = points[-1]
                draw.text(
                    (min(label_point[0] + 4, px1 - 34), label_point[1] - 7),
                    f"K={k}",
                    fill=color,
                    font=font(),
                )
            if row_idx == panel_rows - 1:
                draw.text((px0 + 80, y1 - 18), "overrun risk", fill="#111111", font=font())
            if col_idx == 0:
                draw.text((x0 + 2, py0 - 18), "compute efficiency", fill="#111111", font=font())
    image.save(path)


def write_report(root: Path, rows: list[dict[str, Any]], thresholds: list[float]) -> None:
    lines = [
        "# Threshold Tradeoff Report",
        "",
        "This report analyzes DLM confidence as an empirical threshold parameter for speculative decoding.",
        "",
        "Definitions:",
        "",
        "- `prediction overrun`: `P(pred_tokens > actual_accept_tokens)`.",
        "- `prediction mismatch`: `P(pred_tokens != actual_accept_tokens)`.",
        "- `compute efficiency`: `E[pred_tokens / K]`.",
        "- Pareto optimal means no other threshold has both lower-or-equal error and higher-or-equal efficiency with one strict improvement.",
        "",
        f"Thresholds: {', '.join(f'{threshold:g}' for threshold in thresholds)}",
        "",
        "## Pareto Thresholds",
        "",
    ]
    pareto_rows = [row for row in rows if row["is_pareto_optimal"]]
    for workload in sorted({row["workload"] for row in pareto_rows}):
        lines.append(f"### {WORKLOAD_TITLES.get(workload, workload)}")
        for _, _, label in CASE_ORDER:
            case_rows = [row for row in pareto_rows if row["workload"] == workload and row["case_label"] == label]
            if not case_rows:
                continue
            for k in sorted({int(row["num_spec_tokens"]) for row in case_rows}):
                group = sorted(
                    [row for row in case_rows if int(row["num_spec_tokens"]) == k],
                    key=lambda row: (float(row["error_probability"]), -float(row["compute_efficiency"])),
                )
                parts = [
                    (
                        f"t={row['threshold']:g} "
                        f"over={row['error_probability']:.3f} "
                        f"mismatch={row['mismatch_probability']:.3f} "
                        f"eff={row['compute_efficiency']:.3f}"
                    )
                    for row in group
                ]
                lines.append(f"- {label} K={k}: " + "; ".join(parts))
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `threshold_tradeoff.csv`: all threshold points.",
            "- `pareto_thresholds.csv`: non-dominated threshold points.",
            "- `figures/threshold_tradeoff.png`: risk/efficiency tradeoff curves.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    trace_root: Path,
    output_root: Path,
    thresholds: list[float],
    workloads: list[str],
    models: list[str],
    methods: list[str],
    num_spec_tokens: list[int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = analyze_steps(
        load_steps(
            trace_root,
            set(workloads),
            set(models),
            set(methods),
            set(num_spec_tokens),
        ),
        thresholds,
    )
    write_csv(output_root / "threshold_tradeoff.csv", rows)
    write_csv(output_root / "pareto_thresholds.csv", [row for row in rows if row["is_pareto_optimal"]])
    draw_tradeoff(fig_dir / "threshold_tradeoff.png", rows, workloads)
    write_report(output_root, rows, thresholds)
    print(f"[INFO] Wrote threshold tradeoff: {output_root / 'threshold_tradeoff.csv'}")
    print(f"[INFO] Wrote Pareto thresholds:  {output_root / 'pareto_thresholds.csv'}")
    print(f"[INFO] Wrote figure:             {fig_dir / 'threshold_tradeoff.png'}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        trace_dir = root / "runs" / "math_qwen3_8b_eagle3_k4" / "trace"
        trace_dir.mkdir(parents=True)
        rows = []
        for step_id, (probs, accepted) in enumerate(
            [([0.9, 0.9, 0.1, 0.9], 2), ([0.8, 0.8, 0.8, 0.8], 4)],
            start=1,
        ):
            for pos, prob in enumerate(probs, start=1):
                rows.append(
                    {
                        "dataset_label": "math",
                        "model_label": "qwen3_8b",
                        "method": "eagle3",
                        "request_id": "selftest",
                        "step_id": step_id,
                        "draft_position": pos,
                        "num_spec_tokens": 4,
                        "draft_selected_prob": prob,
                        "num_accepted_in_step": accepted,
                    }
                )
        with (trace_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = root / "out"
        analyze(root, out, [0.5, 0.05], ["math"], ["qwen3_8b"], ["eagle3"], [4])
        tradeoff = list(csv.DictReader((out / "threshold_tradeoff.csv").open()))
        by_threshold = {float(row["threshold"]): row for row in tradeoff}
        assert abs(float(by_threshold[0.5]["error_probability"]) - 0.0) < 1e-9
        assert abs(float(by_threshold[0.5]["compute_efficiency"]) - 0.625) < 1e-9
        assert abs(float(by_threshold[0.05]["error_probability"]) - 0.5) < 1e-9
        assert (out / "pareto_thresholds.csv").exists()
        assert (out / "figures" / "threshold_tradeoff.png").exists()
    print("[INFO] self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", nargs="?", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--num-spec-tokens", default=DEFAULT_NUM_SPEC_TOKENS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.trace_root is None:
        raise SystemExit("trace_root is required unless --self-test is used")
    if args.output_root is None:
        raise SystemExit("--output-root is required unless --self-test is used")
    analyze(
        trace_root=args.trace_root,
        output_root=args.output_root,
        thresholds=parse_thresholds(args.thresholds),
        workloads=parse_list(args.workloads),
        models=parse_list(args.models),
        methods=parse_list(args.methods),
        num_spec_tokens=parse_ints(args.num_spec_tokens),
    )


if __name__ == "__main__":
    main()
