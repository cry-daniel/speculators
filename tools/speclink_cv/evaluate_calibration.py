#!/usr/bin/env python3
"""Evaluate a SpecLink-CV confidence calibration model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.speclink_cv.calibrate_acceptance import bin_index, collect_rows
from tools.speclink_cv.core import write_csv, write_json


def predict(model: dict, confidence: float) -> float:
    idx = bin_index(confidence, int(model["num_bins"]))
    return float(model["bins"][idx]["acceptance_rate"])


def draw_reliability(path: Path, rows: list[dict]) -> None:
    width, height = 720, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    px0, py0, px1, py1 = 70, 60, 660, 450
    draw.text((70, 24), "Confidence calibration reliability", fill="#111", font=font)
    draw.rectangle((px0, py0, px1, py1), outline="#222")
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = px0 + tick * (px1 - px0)
        y = py1 - tick * (py1 - py0)
        draw.line((x, py0, x, py1), fill="#eee")
        draw.line((px0, y, px1, y), fill="#eee")
        draw.text((x - 8, py1 + 8), f"{tick:g}", fill="#555", font=font)
        draw.text((px0 - 36, y - 6), f"{tick:g}", fill="#555", font=font)
    draw.line((px0, py1, px1, py0), fill="#999", width=1)
    points = []
    for row in rows:
        if int(row["count"]) <= 0:
            continue
        x = px0 + float(row["mean_confidence"]) * (px1 - px0)
        y = py1 - float(row["actual_acceptance"]) * (py1 - py0)
        points.append((x, y))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#0b7285")
    if len(points) >= 2:
        draw.line(points, fill="#0b7285", width=2)
    draw.text((px0 + 190, py1 + 34), "mean DLM confidence", fill="#111", font=font)
    draw.text((8, py0 - 24), "actual acceptance", fill="#111", font=font)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--calibration-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", default="math,mtbench")
    parser.add_argument("--split", choices=["all", "even", "odd"], default="odd")
    args = parser.parse_args()
    model = json.loads(args.calibration_model.read_text(encoding="utf-8"))
    workloads = {item.strip() for item in args.workloads.split(",") if item.strip()}
    rows = collect_rows(args.trace_root, workloads)
    if args.split == "even":
        rows = [row for row in rows if int(row["dataset_index"]) % 2 == 0]
    elif args.split == "odd":
        rows = [row for row in rows if int(row["dataset_index"]) % 2 == 1]
    if not rows:
        raise SystemExit("No evaluation rows found")

    bin_rows: list[dict] = []
    total_abs_gap = 0.0
    total_brier = 0.0
    for item in model["bins"]:
        subset = [
            row
            for row in rows
            if int(item["bin"]) == bin_index(float(row["confidence"]), int(model["num_bins"]))
        ]
        if subset:
            actual = sum(int(row["accepted"]) for row in subset) / len(subset)
            conf = sum(float(row["confidence"]) for row in subset) / len(subset)
        else:
            actual = float(item["acceptance_rate"])
            conf = (float(item["left"]) + float(item["right"])) / 2
        pred = float(item["acceptance_rate"])
        count = len(subset)
        total_abs_gap += count * abs(pred - actual)
        for row in subset:
            total_brier += (predict(model, float(row["confidence"])) - int(row["accepted"])) ** 2
        bin_rows.append(
            {
                "bin": item["bin"],
                "count": count,
                "mean_confidence": conf,
                "predicted_acceptance": pred,
                "actual_acceptance": actual,
                "abs_gap": abs(pred - actual),
            }
        )
    metrics = {
        "split": args.split,
        "rows": len(rows),
        "ece": total_abs_gap / len(rows),
        "brier": total_brier / len(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "calibration_metrics.json", metrics)
    write_csv(args.output_dir / "reliability_bins.csv", bin_rows)
    draw_reliability(args.output_dir / "reliability_diagram.png", bin_rows)
    (args.output_dir / "calibration_eval_report.md").write_text(
        "# Calibration Evaluation\n\n"
        f"- split: {args.split}\n"
        f"- rows: {len(rows)}\n"
        f"- ECE: {metrics['ece']:.4f}\n"
        f"- Brier: {metrics['brier']:.4f}\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote {args.output_dir / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()
