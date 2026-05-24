#!/usr/bin/env python3
"""Write dependency-light PNG figures for SpecLink reports."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canvas(width: int = 1100, height: int = 620) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "white")
    return img, ImageDraw.Draw(img)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill: str = "#222222") -> None:
    draw.text(xy, value, fill=fill, font=ImageFont.load_default())


def write_bar(path: Path, title: str, rows: list[tuple[str, float]], ylabel: str) -> None:
    img, draw = canvas()
    text(draw, (30, 20), title)
    text(draw, (30, 45), ylabel)
    if not rows:
        text(draw, (30, 90), "No rows")
        save(img, path)
        return
    left, top, right, bottom = 80, 80, 1060, 520
    max_value = max(value for _label, value in rows) or 1.0
    draw.rectangle((left, top, right, bottom), outline="#CCCCCC")
    bar_w = max(8, (right - left) // max(1, len(rows)) - 8)
    for idx, (label, value) in enumerate(rows):
        x0 = left + idx * ((right - left) / max(1, len(rows))) + 4
        x1 = x0 + bar_w
        y0 = bottom - (value / max_value) * (bottom - top)
        draw.rectangle((x0, y0, x1, bottom), fill=COLORS[idx % len(COLORS)])
        text(draw, (int(x0), int(y0) - 14), f"{value:.3g}")
        text(draw, (int(x0), bottom + 8), label[:16])
    save(img, path)


def write_lines(
    path: Path,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    xlabel: str,
    ylabel: str,
) -> None:
    img, draw = canvas()
    text(draw, (30, 20), title)
    text(draw, (30, 45), f"{xlabel} / {ylabel}")
    left, top, right, bottom = 80, 80, 1060, 520
    draw.rectangle((left, top, right, bottom), outline="#CCCCCC")
    points = [point for values in series.values() for point in values]
    if not points:
        text(draw, (30, 90), "No rows")
        save(img, path)
        return
    min_x = min(x for x, _y in points)
    max_x = max(x for x, _y in points)
    min_y = min(0.0, min(y for _x, y in points))
    max_y = max(y for _x, y in points) or 1.0
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0
    for idx, (name, values) in enumerate(sorted(series.items())):
        color = COLORS[idx % len(COLORS)]
        xy = []
        for x, y in sorted(values):
            px = left + (x - min_x) / (max_x - min_x) * (right - left)
            py = bottom - (y - min_y) / (max_y - min_y) * (bottom - top)
            xy.append((px, py))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        if len(xy) > 1:
            draw.line(xy, fill=color, width=2)
        text(draw, (840, 85 + idx * 16), name[:28], fill=color)
    save(img, path)


def write_stacked(path: Path, title: str, rows: list[dict[str, str]], fields: list[str]) -> None:
    img, draw = canvas()
    text(draw, (30, 20), title)
    left, top, right, bottom = 80, 80, 1060, 520
    draw.rectangle((left, top, right, bottom), outline="#CCCCCC")
    if not rows:
        text(draw, (30, 90), "No rows")
        save(img, path)
        return
    bar_w = max(8, (right - left) // max(1, len(rows)) - 10)
    for idx, row in enumerate(rows):
        x0 = left + idx * ((right - left) / max(1, len(rows))) + 5
        y = bottom
        for fidx, field in enumerate(fields):
            value = fnum(row.get(field))
            h = value / 100.0 * (bottom - top)
            draw.rectangle((x0, y - h, x0 + bar_w, y), fill=COLORS[fidx % len(COLORS)])
            y -= h
        label = f"{row.get('method', '')}-k{row.get('num_spec_tokens', '')}"
        text(draw, (int(x0), bottom + 8), label[:16])
    for fidx, field in enumerate(fields):
        text(draw, (835, 85 + fidx * 16), field.replace("_pct", ""), fill=COLORS[fidx % len(COLORS)])
    save(img, path)


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    tables = root / "tables"
    figures = root / "figures"
    baseline = read_csv(tables / "baseline_summary.csv")
    breakdown = read_csv(tables / "breakdown_summary.csv")
    sparse = read_csv(tables / "sparse_layout_summary.csv")
    micro = read_csv(tables / "sparse_microbench.csv")

    write_bar(
        figures / "baseline_tokens_per_s.png",
        "Output Tokens per Second",
        [(row.get("run") or row.get("method", ""), fnum(row.get("output_tokens_per_s"))) for row in baseline],
        "generated tok/s",
    )
    write_bar(
        figures / "baseline_accuracy.png",
        "Flexible Final-Answer EM",
        [(row.get("run") or row.get("method", ""), fnum(row.get("flexible_em"))) for row in baseline],
        "EM",
    )
    acceptance_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in baseline:
        label = row.get("run") or row.get("method", "")
        for key, value in row.items():
            if key.startswith("acceptance_rate_pos_") and value:
                acceptance_series[label].append((fnum(key.rsplit("_", 1)[1]), fnum(value)))
    write_lines(figures / "acceptance_by_position.png", "Acceptance by Position", acceptance_series, "position", "acceptance")
    write_stacked(
        figures / "breakdown_stacked_bar.png",
        "Engine Step Breakdown",
        breakdown,
        [
            "draft_forward_pct",
            "target_verify_forward_pct",
            "accept_reject_sampler_pct",
            "speclink_planner_pct",
            "scheduler_step_pct",
            "engine_update_pct",
            "other_pct",
        ],
    )
    union_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    jaccard_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    loaded_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    wasted_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in sparse:
        layout = row.get("layout", "")
        k = fnum(row.get("num_spec_tokens"))
        union_series[layout].append((k, fnum(row.get("union_blocks"))))
        jaccard_series[layout].append((k, fnum(row.get("jaccard_mean"))))
        loaded_series[layout].append((k, fnum(row.get("accepted_tokens_per_loaded_kv_block"))))
        wasted_series[layout].append((fnum(row.get("weighted_wasted_private_blocks")), fnum(row.get("weighted_coverage"))))
    write_lines(figures / "union_blocks_vs_k.png", "Union Blocks vs K", union_series, "K", "union blocks")
    write_lines(figures / "jaccard_overlap_heatmap.png", "Jaccard Overlap by K", jaccard_series, "K", "Jaccard")
    write_lines(
        figures / "accepted_tokens_per_loaded_kv_block.png",
        "Accepted Tokens per Loaded KV Block",
        loaded_series,
        "K",
        "accepted / block",
    )
    write_lines(
        figures / "reach_probability_vs_wasted_blocks.png",
        "Weighted Coverage vs Wasted Private Blocks",
        wasted_series,
        "wasted private blocks",
        "weighted coverage",
    )
    pareto: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in micro:
        pareto[row.get("layout", "")].append(
            (
                fnum(row.get("kernel_or_proxy_time_ms_mean")),
                fnum(row.get("bytes_estimate")),
            )
        )
    write_lines(
        figures / "sparse_quality_speed_pareto.png",
        "Proxy Sparse Time vs Estimated Bytes",
        pareto,
        "proxy ms",
        "estimated bytes",
    )
    print(f"wrote figures to {figures}")


if __name__ == "__main__":
    main()
