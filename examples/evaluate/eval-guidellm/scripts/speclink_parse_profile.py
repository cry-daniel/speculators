#!/usr/bin/env python3
"""Aggregate SPECLINK_PROFILE JSONL events into coarse breakdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TIME_FIELDS = [
    "draft_forward_ms",
    "target_verify_forward_ms",
    "accept_reject_sampler_ms",
    "scheduler_step_ms",
    "engine_update_ms",
]
SUBCOMPONENT_FIELDS = ["speclink_planner_ms"]


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def iter_profile_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.rglob("profile_events.jsonl")))
        elif path.exists():
            out.append(path)
    return out


def read_events(
    paths: list[Path],
    exclude_run_substrings: list[str] | None = None,
) -> list[dict[str, Any]]:
    profile_paths = iter_profile_paths(paths)
    if exclude_run_substrings:
        profile_paths = [
            path
            for path in profile_paths
            if not any(marker in str(path.parent) for marker in exclude_run_substrings)
        ]
    events: list[dict[str, Any]] = []
    for path in profile_paths:
        events.extend(read_profile_events(path))
    return events


def read_profile_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    env = read_env(path.parent / "env.txt")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                event.setdefault("method", env.get("method", "unknown"))
                event.setdefault("num_spec_tokens", _int_or_original(env.get("num_spec_tokens")))
                event.setdefault("rate", _int_or_original(env.get("guidellm_rate")))
                event["run_dir"] = str(path.parent)
                events.append(event)
    return events


def _int_or_original(value: Any) -> Any:
    if value is None or value == "":
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def aggregate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            event.get("method", "unknown"),
            event.get("num_spec_tokens"),
            event.get("guidellm_rate") or event.get("rate"),
        )
        grouped.setdefault(key, []).append(event)

    rows: list[dict[str, Any]] = []
    for (method, k, rate), group in sorted(grouped.items(), key=str):
        has_worker_profile = any("worker_profile" in str(item.get("run_dir", "")) for item in group)
        if has_worker_profile:
            group = [item for item in group if "worker_profile" in str(item.get("run_dir", ""))]

        engine_events = [item for item in group if item.get("step") == "engine_step"]
        n_engine_steps = len(engine_events)
        total_engine_ms = sum_float(engine_events, "total_engine_step_ms")
        scheduler_ms = sum_float(engine_events, "scheduler_step_ms")
        model_wait_ms = sum_float(engine_events, "target_verify_forward_ms")
        update_total_ms = sum_float(engine_events, "engine_update_ms")
        draft_ms = sum_float(group, "draft_forward_ms")
        sampler_ms = sum_float(group, "accept_reject_sampler_ms")
        planner_ms = sum_float(group, "speclink_planner_ms")

        if model_wait_ms > 0.0:
            draft_ms = min(draft_ms, model_wait_ms)
            sampler_ms = min(sampler_ms, max(model_wait_ms - draft_ms, 0.0))
        target_residual_ms = max(model_wait_ms - draft_ms - sampler_ms, 0.0)

        if update_total_ms > 0.0:
            planner_ms = min(planner_ms, update_total_ms)
        update_residual_ms = max(update_total_ms - planner_ms, 0.0)

        known_additive_ms = (
            scheduler_ms
            + draft_ms
            + target_residual_ms
            + sampler_ms
            + update_residual_ms
            + planner_ms
        )
        other_ms = max(total_engine_ms - known_additive_ms, 0.0) if total_engine_ms else None
        row: dict[str, Any] = {
            "method": method,
            "num_spec_tokens": k,
            "rate": rate,
            "profile_scope": "worker_profile" if has_worker_profile else "all_profiles",
            "run_count": len({item.get("run_dir") for item in group if item.get("run_dir")}),
            "n_events": len(group),
            "n_engine_steps": n_engine_steps,
            "n_plan_steps": sum(1 for item in group if item.get("step") == "speclink_plan"),
            "draft_forward_ms": mean_per_engine_step(draft_ms, n_engine_steps),
            "target_verify_forward_ms": mean_per_engine_step(target_residual_ms, n_engine_steps),
            "accept_reject_sampler_ms": mean_per_engine_step(sampler_ms, n_engine_steps),
            "speclink_planner_ms": mean_per_engine_step(planner_ms, n_engine_steps),
            "scheduler_step_ms": mean_per_engine_step(scheduler_ms, n_engine_steps),
            "engine_update_ms": mean_per_engine_step(update_residual_ms, n_engine_steps),
            "model_wait_total_ms": mean_per_engine_step(model_wait_ms, n_engine_steps),
            "engine_update_total_ms": mean_per_engine_step(update_total_ms, n_engine_steps),
            "total_engine_step_ms": mean_per_engine_step(total_engine_ms, n_engine_steps),
            "other_ms": mean_per_engine_step(other_ms or 0.0, n_engine_steps) if other_ms is not None else None,
        }
        row["draft_forward_pct"] = pct(draft_ms, total_engine_ms)
        row["target_verify_forward_pct"] = pct(target_residual_ms, total_engine_ms)
        row["accept_reject_sampler_pct"] = pct(sampler_ms, total_engine_ms)
        row["speclink_planner_pct"] = pct(planner_ms, total_engine_ms)
        row["scheduler_step_pct"] = pct(scheduler_ms, total_engine_ms)
        row["engine_update_pct"] = pct(update_residual_ms, total_engine_ms)
        row["model_wait_total_pct"] = pct(model_wait_ms, total_engine_ms)
        row["engine_update_total_pct"] = pct(update_total_ms, total_engine_ms)
        row["other_pct"] = pct(other_ms, total_engine_ms) if other_ms is not None else None
        row["unknown_pct"] = None if total_engine_ms else 100.0
        rows.append(row)
    return rows


def sum_float(events: list[dict[str, Any]], field: str) -> float:
    return sum(float(item[field]) for item in events if item.get(field) is not None)


def mean_per_engine_step(total_ms: float, n_engine_steps: int) -> float | None:
    return total_ms / n_engine_steps if n_engine_steps else None


def pct(value: float | None, total: float) -> float | None:
    if value is None or not total:
        return None
    return value / total * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "method",
        "num_spec_tokens",
        "rate",
        "profile_scope",
        "run_count",
        "n_events",
        "n_engine_steps",
        "n_plan_steps",
        "draft_forward_ms",
        "target_verify_forward_ms",
        "accept_reject_sampler_ms",
        "speclink_planner_ms",
        "scheduler_step_ms",
        "engine_update_ms",
        "model_wait_total_ms",
        "engine_update_total_ms",
        "total_engine_step_ms",
        "draft_forward_pct",
        "target_verify_forward_pct",
        "accept_reject_sampler_pct",
        "speclink_planner_pct",
        "scheduler_step_pct",
        "engine_update_pct",
        "model_wait_total_pct",
        "engine_update_total_pct",
        "other_pct",
        "unknown_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "method",
        "num_spec_tokens",
        "rate",
        "profile_scope",
        "run_count",
        "n_events",
        "n_engine_steps",
        "n_plan_steps",
        "draft_forward_pct",
        "target_verify_forward_pct",
        "accept_reject_sampler_pct",
        "speclink_planner_pct",
        "scheduler_step_pct",
        "engine_update_pct",
        "other_pct",
        "unknown_pct",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) < 1e-12:
            return "0"
        return f"{value:.3g}"
    return str(value)


def maybe_write_figure(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None or not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    labels = [f"{row['method']}-k{row['num_spec_tokens']}" for row in rows]
    fields = [
        "draft_forward_pct",
        "target_verify_forward_pct",
        "accept_reject_sampler_pct",
        "speclink_planner_pct",
        "scheduler_step_pct",
        "engine_update_pct",
        "other_pct",
    ]
    bottoms = [0.0 for _ in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 4))
    for field in fields:
        values = [float(row.get(field) or 0.0) for row in rows]
        ax.bar(labels, values, bottom=bottoms, label=field.replace("_pct", ""))
        bottoms = [left + value for left, value in zip(bottoms, values, strict=True)]
    ax.set_ylabel("Percent of engine step")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-jsonl",
        nargs="+",
        required=True,
        help="One or more profile_events.jsonl files, or directories to scan recursively.",
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--figure")
    parser.add_argument(
        "--exclude-run-substring",
        action="append",
        default=[],
        help="Skip profile files whose run directory contains this substring.",
    )
    args = parser.parse_args()

    rows = aggregate(
        read_events(
            [Path(item) for item in args.profile_jsonl],
            exclude_run_substrings=args.exclude_run_substring,
        )
    )
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    maybe_write_figure(Path(args.figure) if args.figure else None, rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
