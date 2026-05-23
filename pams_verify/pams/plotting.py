from __future__ import annotations

from pathlib import Path
from typing import Iterable


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        _save_bar_pil(path, labels, values, title, ylabel)


def save_line(path: Path, xs: Iterable[float], ys: Iterable[float], title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(xs), list(ys), marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        _save_line_pil(path, list(xs), list(ys), title, xlabel, ylabel)


def _save_bar_pil(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    from PIL import Image, ImageDraw

    width, height = 900, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 18), title, fill="black")
    draw.text((24, 42), ylabel, fill="black")
    if not values:
        values = [0.0]
        labels = ["empty"]
    min_v = min(0.0, min(values))
    max_v = max(1e-9, max(values))
    chart_left, chart_top, chart_right, chart_bottom = 70, 80, 870, 430
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="black")
    n = len(values)
    bar_w = max(4, (chart_right - chart_left) // max(1, n) - 4)
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = chart_left + idx * ((chart_right - chart_left) / max(1, n)) + 2
        norm = (value - min_v) / max(1e-9, max_v - min_v)
        y0 = chart_bottom - norm * (chart_bottom - chart_top)
        draw.rectangle((x0, y0, x0 + bar_w, chart_bottom), fill=(80, 120, 200), outline="black")
        draw.text((x0, chart_bottom + 8), str(label)[:12], fill="black")
    img.save(path)


def _save_line_pil(path: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str) -> None:
    from PIL import Image, ImageDraw

    width, height = 900, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 18), title, fill="black")
    draw.text((24, 42), f"{xlabel} / {ylabel}", fill="black")
    chart_left, chart_top, chart_right, chart_bottom = 70, 80, 870, 430
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="black")
    if len(xs) >= 2 and len(ys) >= 2:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        points = []
        for x, y in zip(xs, ys):
            px = chart_left + (x - min_x) / max(1e-9, max_x - min_x) * (chart_right - chart_left)
            py = chart_bottom - (y - min_y) / max(1e-9, max_y - min_y) * (chart_bottom - chart_top)
            points.append((px, py))
        draw.line(points, fill=(80, 120, 200), width=2)
        for px, py in points[:: max(1, len(points) // 40)]:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(20, 60, 160))
    img.save(path)
