#!/usr/bin/env python3
"""Analyze rolling vLLM throughput samples before final request drain.

GuideLLM's reported output tokens/s is end-to-end: successful output tokens
divided by benchmark wall time.  With a fixed number of requests, the final
drain can run with far fewer active requests than the configured batch size.
This script parses vLLM's periodic logger rows only as a diagnostic for older
finite-request runs.  It is not the official saturated-throughput measurement.
Use tools/speclink_cv/steady_state_openai_benchmark.py, or the matrix runner's
--benchmark-mode steady_state, for fixed-window closed-loop throughput claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


LOGGER_RE = re.compile(
    r"Avg prompt throughput:\s*(?P<prompt>[0-9.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*(?P<generation>[0-9.]+)\s*tokens/s,\s*"
    r"Running:\s*(?P<running>[0-9]+)\s*reqs,\s*"
    r"Waiting:\s*(?P<waiting>[0-9]+)\s*reqs,\s*"
    r"GPU KV cache usage:\s*(?P<kv>[0-9.]+)%"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="GuideLLM matrix result roots.")
    parser.add_argument("--output-dir", help="Directory for report files.")
    parser.add_argument(
        "--running-fraction",
        type=float,
        default=0.8,
        help="A sample is steady-state when Running >= ceil(batch_size * fraction).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def f(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def ratio(numer: float | str, denom: float | str | None) -> float | str:
    if numer == "" or denom in ("", None, 0):
        return ""
    try:
        return float(numer) / float(denom)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def parse_vllm_logger(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    if not path.exists():
        return samples
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOGGER_RE.search(line)
        if not match:
            continue
        samples.append(
            {
                "prompt_tps": float(match.group("prompt")),
                "generation_tps": float(match.group("generation")),
                "running": float(match.group("running")),
                "waiting": float(match.group("waiting")),
                "kv_usage_pct": float(match.group("kv")),
            }
        )
    return samples


def summary_path(root: Path) -> Path | None:
    for candidate in (root / "09_reports/summary_metrics.csv", root / "summary_metrics.csv"):
        if candidate.exists():
            return candidate
    return None


def load_matrix_rows(root: Path) -> list[dict[str, str]]:
    path = summary_path(root)
    if path is None:
        return []
    rows = read_csv(path)
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("measurement_type") not in ("", "guidellm_end_to_end"):
            continue
        if row.get("status") not in ("", "ok"):
            continue
        if not row.get("output_dir"):
            continue
        out.append(row)
    return out


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("dataset", "")),
        str(row.get("K", "")),
        str(row.get("batch_size", "")),
    )


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("dataset", "")),
        str(row.get("K", "")),
        str(row.get("batch_size", "")),
        str(row.get("source_group", "")),
        str(row.get("method", "")),
    )


def analyze_row(row: dict[str, str], *, root: Path, running_fraction: float) -> dict[str, Any]:
    run_dir = Path(row["output_dir"])
    config = read_json(run_dir / "config.json")
    batch_size = int(config.get("batch_size") or row.get("batch_size") or 0)
    threshold = max(1, math.ceil(batch_size * running_fraction)) if batch_size else 1
    samples = parse_vllm_logger(run_dir / "vllm_server.log")
    gen = [sample["generation_tps"] for sample in samples]
    steady = [
        sample["generation_tps"]
        for sample in samples
        if sample["running"] >= threshold
    ]
    full = [
        sample["generation_tps"]
        for sample in samples
        if batch_size and sample["running"] >= batch_size
    ]
    tail = [
        sample["generation_tps"]
        for sample in samples
        if sample["running"] < threshold
    ]
    running_values = [sample["running"] for sample in samples]
    e2e = f(row.get("throughput"))
    steady_mean = mean(steady)
    full_mean = mean(full)
    return {
        "source_root": str(root),
        "source_group": root.name,
        "model": row.get("model", ""),
        "dataset": row.get("dataset", ""),
        "K": row.get("K", ""),
        "batch_size": row.get("batch_size", ""),
        "method": row.get("method", ""),
        "run_dir": str(run_dir),
        "end_to_end_tps": e2e if e2e is not None else "",
        "vllm_sample_count": len(samples),
        "steady_running_threshold": threshold,
        "running_mean": mean(running_values),
        "running_p50": percentile(running_values, 50),
        "running_p95": percentile(running_values, 95),
        "rolling_gen_tps_mean": mean(gen),
        "rolling_gen_tps_p50": percentile(gen, 50),
        "rolling_gen_tps_p95": percentile(gen, 95),
        "steady_sample_count": len(steady),
        "steady_sample_fraction": len(steady) / len(samples) if samples else "",
        "steady_gen_tps_mean": steady_mean,
        "steady_gen_tps_p50": percentile(steady, 50),
        "steady_gen_tps_p95": percentile(steady, 95),
        "full_batch_sample_count": len(full),
        "full_batch_gen_tps_mean": full_mean,
        "tail_sample_count": len(tail),
        "tail_gen_tps_mean": mean(tail),
        "steady_vs_end_to_end": ratio(steady_mean, e2e),
        "full_batch_vs_end_to_end": ratio(full_mean, e2e),
    }


def add_baseline_ratios(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("method") == "eagle3_oneshot":
            baselines[row_key(row)] = row
    for row in rows:
        base = baselines.get(row_key(row))
        if not base or row.get("method") == "eagle3_oneshot":
            row["steady_speedup_vs_eagle3"] = ""
            row["end_to_end_speedup_vs_eagle3"] = ""
            continue
        row["steady_speedup_vs_eagle3"] = ratio(
            row.get("steady_gen_tps_mean", ""), base.get("steady_gen_tps_mean", "")
        )
        row["end_to_end_speedup_vs_eagle3"] = ratio(
            row.get("end_to_end_tps", ""), base.get("end_to_end_tps", "")
        )


def fmt(value: Any, digits: int = 2) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Steady-State Throughput Analysis",
        "",
        "GuideLLM throughput is generated output tokens divided by benchmark wall time. The diagnostic columns below use vLLM rolling logger samples and exclude samples whose active `Running` request count is below the configured threshold, so they reduce fixed-request drain effects. This is not the official saturated-throughput metric.",
        "",
        "| group | model | dataset | K | bs | method | E2E tok/s | steady tok/s | steady/E2E | E2E speedup | steady speedup | steady samples | tail samples | running mean |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=row_sort_key):
        lines.append(
            "| "
            f"{row.get('source_group', '')} | "
            f"{row.get('model', '')} | {row.get('dataset', '')} | "
            f"{row.get('K', '')} | {row.get('batch_size', '')} | "
            f"{row.get('method', '')} | {fmt(row.get('end_to_end_tps'))} | "
            f"{fmt(row.get('steady_gen_tps_mean'))} | "
            f"{fmt(row.get('steady_vs_end_to_end'), 3)} | "
            f"{fmt(row.get('end_to_end_speedup_vs_eagle3'), 3)} | "
            f"{fmt(row.get('steady_speedup_vs_eagle3'), 3)} | "
            f"{row.get('steady_sample_count', '')} | "
            f"{row.get('tail_sample_count', '')} | "
            f"{fmt(row.get('running_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `steady tok/s` here is parsed from periodic vLLM log samples. It is a diagnostic only, not a replacement for the closed-loop fixed-window saturated throughput client.",
            "- Final serving throughput claims should use `tools/speclink_cv/steady_state_openai_benchmark.py` or `run_speclink_cv_guidellm_matrix.py --benchmark-mode steady_state`.",
            "- A large `steady/E2E` ratio means fixed-request drain or setup/cooldown inside the benchmark window is materially lowering the end-to-end number.",
            "- If steady speedup is high but E2E speedup is low, increase `MAX_REQUESTS` or use a longer load phase before making an implementation conclusion.",
            "- Blank steady-state fields mean the run finished before vLLM emitted enough periodic logger samples for that threshold.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    roots = [Path(item).resolve() for item in args.roots]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else roots[0] / "09_reports"
    rows: list[dict[str, Any]] = []
    for root in roots:
        for matrix_row in load_matrix_rows(root):
            rows.append(
                analyze_row(
                    matrix_row,
                    root=root,
                    running_fraction=args.running_fraction,
                )
            )
    add_baseline_ratios(rows)
    write_csv(output_dir / "steady_state_throughput.csv", rows)
    write_markdown(output_dir / "steady_state_throughput.md", rows)
    print(f"[INFO] wrote {output_dir / 'steady_state_throughput.csv'}")
    print(f"[INFO] wrote {output_dir / 'steady_state_throughput.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
