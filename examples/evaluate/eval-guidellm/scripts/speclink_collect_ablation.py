#!/usr/bin/env python3
"""Collect speclink plan-only ablation metrics from existing run dirs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_nested(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_nested(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_nested(value, key)
            if found is not None:
                return found
    return None


def stat_mean(obj: Any) -> float | None:
    if isinstance(obj, int | float):
        return float(obj)
    if isinstance(obj, dict):
        if "successful" in obj:
            return stat_mean(obj["successful"])
        value = obj.get("mean")
        if isinstance(value, int | float):
            return float(value)
    return None


def request_total(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("successful", "total"):
            if isinstance(value.get(key), int):
                return int(value[key])
    if isinstance(value, int):
        return value
    return None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


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


def unslug(value: str) -> str:
    return value.replace("p", ".").replace("m", "-")


def parse_run_name(name: str) -> dict[str, str]:
    tokens = name.split("_")
    out = {
        "draft_method": "",
        "layout": name,
        "num_spec_tokens": "",
        "rate": "",
        "repeat_id": "",
        "block_size": "",
        "shared_budget": "",
        "private_min": "",
        "private_max": "",
        "lambda_risk": "",
        "alpha": "",
        "beta": "",
        "fallback": "",
    }
    keyed: dict[str, str] = {}
    layout_tokens: list[str] = []
    for token in tokens:
        if token == "plan":
            continue
        if token.startswith("rate"):
            out["rate"] = token[4:]
        elif token.startswith("r") and token[1:].isdigit():
            out["repeat_id"] = token[1:]
        elif token.startswith("k") and token[1:].isdigit():
            out["num_spec_tokens"] = token[1:]
        elif token.startswith("bs") and token[2:].isdigit():
            keyed["block_size"] = token[2:]
        elif token.startswith("sb") and token[2:].isdigit():
            keyed["shared_budget"] = token[2:]
        elif token.startswith("pmin") and token[4:].isdigit():
            keyed["private_min"] = token[4:]
        elif token.startswith("pmax") and token[4:].isdigit():
            keyed["private_max"] = token[4:]
        elif token.startswith("lam"):
            keyed["lambda_risk"] = unslug(token[3:])
        elif token.startswith("a") and len(token) > 1:
            keyed["alpha"] = unslug(token[1:])
        elif token.startswith("b") and len(token) > 1:
            keyed["beta"] = unslug(token[1:])
        elif token.startswith("fb"):
            keyed["fallback"] = token[2:]
        else:
            layout_tokens.append(token)
    if layout_tokens and layout_tokens[0] in {"eagle3", "peagle"}:
        out["draft_method"] = layout_tokens[0]
        layout_tokens = layout_tokens[1:]
    out["layout"] = "_".join(layout_tokens) if layout_tokens else name
    out.update(keyed)
    return out


def trace_stats(path: Path) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "union_blocks": [],
        "mean_blocks_per_token": [],
        "estimated_hbm_bytes": [],
        "planner_ms": [],
        "fallback_tokens": [],
    }
    candidate_sources: set[str] = set()
    block_sizes: set[int] = set()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            candidate_source = event.get("candidate_source")
            if candidate_source:
                candidate_sources.add(str(candidate_source))
            block_size = event.get("block_size")
            if isinstance(block_size, int):
                block_sizes.add(block_size)
            for key in ("union_blocks", "mean_blocks_per_token", "estimated_hbm_bytes", "planner_ms"):
                value = event.get(key)
                if isinstance(value, int | float):
                    values[key].append(float(value))
            fallback = event.get("fallback_tokens")
            if isinstance(fallback, list):
                values["fallback_tokens"].append(float(len(fallback)))
    return {
        "trace_steps": len(values["union_blocks"]),
        "candidate_source": ",".join(sorted(candidate_sources)),
        "block_size": ",".join(str(item) for item in sorted(block_sizes)),
        "mean_union_blocks": mean(values["union_blocks"]),
        "p95_union_blocks": quantile(values["union_blocks"], 0.95),
        "mean_blocks_per_token": mean(values["mean_blocks_per_token"]),
        "mean_estimated_hbm_bytes": mean(values["estimated_hbm_bytes"]),
        "mean_planner_ms": mean(values["planner_ms"]),
        "p95_planner_ms": quantile(values["planner_ms"], 0.95),
        "mean_fallback_tokens": mean(values["fallback_tokens"]),
    }


def collect_run(run_dir: Path) -> dict[str, Any]:
    guidellm = read_json(run_dir / "guidellm_results.json")
    benchmark = (guidellm.get("benchmarks") or [{}])[0]
    metrics = benchmark.get("metrics", {})
    accuracy = read_json(run_dir / "accuracy_summary.json")
    env = read_env(run_dir / "env.txt")
    parsed = parse_run_name(run_dir.name)
    row = {
        "run": run_dir.name,
        "run_group": run_dir.parent.name,
        "draft_method": parsed["draft_method"] or env.get("speclink_method", "").replace("_speclink", ""),
        "layout": env.get("speclink_layout") or parsed["layout"],
        "num_spec_tokens": parsed["num_spec_tokens"],
        "rate": parsed["rate"],
        "repeat_id": env.get("repeat_id") or parsed["repeat_id"],
        "block_size_config": env.get("speclink_block_size") or parsed["block_size"],
        "shared_budget": env.get("speclink_shared_budget") or parsed["shared_budget"],
        "private_min": env.get("speclink_private_min") or parsed["private_min"],
        "private_max": env.get("speclink_private_max") or parsed["private_max"],
        "lambda_risk": env.get("speclink_lambda_risk") or parsed["lambda_risk"],
        "alpha": env.get("speclink_alpha") or parsed["alpha"],
        "beta": env.get("speclink_beta") or parsed["beta"],
        "fallback": parsed["fallback"],
        "n_requests": request_total(find_nested(metrics, "request_totals")),
        "output_tokens_per_s": stat_mean(find_nested(metrics, "output_tokens_per_second")),
        "total_tokens_per_s": stat_mean(find_nested(metrics, "tokens_per_second")),
        "strict_em": accuracy.get("strict_em"),
        "flexible_em": accuracy.get("flexible_em"),
        "accuracy_n": accuracy.get("n"),
        "accuracy_errors": accuracy.get("errors"),
    }
    row.update(trace_stats(run_dir / "live_sparse_trace.jsonl"))
    return row


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "run_group",
        "draft_method",
        "layout",
        "num_spec_tokens",
        "rate",
        "repeat_id",
        "block_size_config",
        "shared_budget",
        "private_min",
        "private_max",
        "lambda_risk",
        "alpha",
        "beta",
        "fallback",
        "n_requests",
        "accuracy_n",
        "accuracy_errors",
        "output_tokens_per_s",
        "flexible_em",
        "trace_steps",
        "candidate_source",
        "block_size",
        "mean_union_blocks",
        "p95_union_blocks",
        "mean_blocks_per_token",
        "mean_estimated_hbm_bytes",
        "mean_planner_ms",
        "p95_planner_ms",
        "mean_fallback_tokens",
        "run",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "run_group",
        "draft_method",
        "layout",
        "num_spec_tokens",
        "rate",
        "repeat_id",
        "block_size_config",
        "shared_budget",
        "private_max",
        "lambda_risk",
        "fallback",
        "n_requests",
        "accuracy_n",
        "output_tokens_per_s",
        "flexible_em",
        "trace_steps",
        "candidate_source",
        "mean_union_blocks",
        "p95_union_blocks",
        "mean_estimated_hbm_bytes",
        "mean_planner_ms",
    ]
    lines = [
        "# speclink Plan-Only Ablation",
        "",
        "These rows summarize existing plan-only serving runs. They are not sparse-kernel speedup results.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="SpecLink result root")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    run_dirs = sorted((root / "05_speclink").glob("*_rate1_plan"))
    run_dirs.extend(sorted((root / "05_speclink_g2").glob("*_rate1_plan")))
    run_dirs.extend(sorted((root / "06_serving_rates").glob("*_plan")))
    rows = [collect_run(run_dir) for run_dir in run_dirs]
    rows.sort(
        key=lambda row: (
            row["run_group"],
            row["draft_method"],
            row["layout"],
            int(row["num_spec_tokens"] or 0),
            int(row["rate"] or 0),
            int(row["repeat_id"] or 0),
            int(row["block_size_config"] or 0),
            int(row["shared_budget"] or 0),
            int(row["private_max"] or 0),
            row["lambda_risk"],
            row["fallback"],
        )
    )
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"collected {len(rows)} speclink plan-only ablation rows")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
