#!/usr/bin/env python3
"""Simulate sparse KV layout metrics from real or proxy SpecLink traces."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import copy
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from speculators.speclink.planner import SpeclinkConfig, SpeclinkPlanner  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def estimate_hbm_bytes(
    union_blocks: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int,
) -> int:
    return num_layers * num_kv_heads * union_blocks * block_size * head_dim * bytes_per_elem * 2


def mean_jaccard(blocks: list[list[int]]) -> float | None:
    values: list[float] = []
    for i, left in enumerate(blocks):
        left_set = set(left)
        for right in blocks[i + 1 :]:
            right_set = set(right)
            union = left_set.union(right_set)
            if union:
                values.append(len(left_set.intersection(right_set)) / len(union))
    return statistics.mean(values) if values else None


def expected_accepted_tokens(plan: Any, accept_probs: list[float]) -> float:
    return sum(rho * prob for rho, prob in zip(plan.rho, accept_probs, strict=True))


def coverage_metrics(
    candidates: list[list[dict[str, Any]]],
    final_blocks_per_token: list[list[int]],
    rho: list[float],
) -> tuple[float | None, float | None]:
    if not candidates or not final_blocks_per_token:
        return None, None
    coverages: list[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    for idx, (items, blocks) in enumerate(zip(candidates, final_blocks_per_token, strict=True)):
        selected = set(blocks)
        score_sum = sum(float(item.get("score", 0.0)) for item in items if int(item["block"]) in selected)
        coverages.append(score_sum)
        weight = rho[idx] if idx < len(rho) else 1.0
        weighted_num += weight * score_sum
        weighted_den += weight
    return statistics.mean(coverages), (weighted_num / weighted_den if weighted_den else None)


def wasted_private_blocks(plan: Any) -> float:
    total = 0.0
    for rho, residual in zip(plan.rho, plan.residual_blocks_per_token, strict=True):
        total += (1.0 - rho) * len(residual)
    return total


def simulate_trace(trace: dict[str, Any], layout: str, args: argparse.Namespace) -> dict[str, Any]:
    candidates = trace["candidates"]
    accept_probs = trace.get("accept_probs") or trace.get("a_i")
    if accept_probs is None:
        raise ValueError("trace is missing accept_probs or a_i")
    block_size = int(trace.get("block_size") or args.block_size)
    if layout == "dense_verifier":
        decode_tokens = int(trace.get("decode_len") or trace.get("generated_len") or 0)
        context_tokens = max(
            int(trace.get("prompt_len") or 0) + decode_tokens,
            block_size,
        )
        union_blocks = max(1, (context_tokens + block_size - 1) // block_size)
        bytes_estimate = estimate_hbm_bytes(
            union_blocks,
            block_size,
            args.num_layers,
            args.num_kv_heads,
            args.head_dim,
            args.bytes_per_elem,
        )
        accepted = sum(
            rho * prob
            for rho, prob in zip(
                _rho([float(x) for x in accept_probs]),
                [float(x) for x in accept_probs],
                strict=True,
            )
        )
        return {
            "context_label": trace.get("context_label", ""),
            "request_id": trace.get("request_id"),
            "step": trace.get("step"),
            "layout": layout,
            "num_spec_tokens": len(candidates),
            "block_size": block_size,
            "mean_blocks_per_token": union_blocks,
            "union_blocks": union_blocks,
            "jaccard_mean": 1.0,
            "coverage_mean": 1.0,
            "weighted_coverage": 1.0,
            "private_unique_blocks": 0,
            "weighted_wasted_private_blocks": 0.0,
            "estimated_hbm_bytes_per_step": bytes_estimate,
            "accepted_tokens_per_loaded_kv_block": (
                accepted / union_blocks if union_blocks else None
            ),
            "accepted_tokens_per_estimated_hbm_byte": (
                accepted / bytes_estimate if bytes_estimate else None
            ),
            "fallback_tokens": 0,
            "candidate_source": trace.get("candidate_source", "unknown"),
        }
    cfg = SpeclinkConfig(
        layout=layout,
        topk_per_token=args.topk_per_token,
        shared_budget=args.shared_budget,
        private_min=args.private_min,
        private_max=args.private_max,
        alpha=args.alpha,
        beta=args.beta,
        lambda_risk=args.lambda_risk,
        risk_threshold=args.risk_threshold,
        fallback_enabled=layout.endswith("fallback"),
    )
    plan = SpeclinkPlanner(cfg).plan(candidates, accept_probs)
    bytes_estimate = estimate_hbm_bytes(
        len(plan.union_blocks),
        block_size,
        args.num_layers,
        args.num_kv_heads,
        args.head_dim,
        args.bytes_per_elem,
    )
    accepted = expected_accepted_tokens(plan, [float(x) for x in accept_probs])
    coverage_mean, weighted_coverage = coverage_metrics(candidates, plan.final_blocks_per_token, plan.rho)
    return {
        "context_label": trace.get("context_label", ""),
        "request_id": trace.get("request_id"),
        "step": trace.get("step"),
        "layout": layout,
        "num_spec_tokens": len(candidates),
        "block_size": block_size,
        "mean_blocks_per_token": plan.stats["mean_blocks_per_token"],
        "union_blocks": len(plan.union_blocks),
        "jaccard_mean": mean_jaccard(plan.final_blocks_per_token),
        "coverage_mean": coverage_mean,
        "weighted_coverage": weighted_coverage,
        "private_unique_blocks": plan.stats["private_unique_blocks"],
        "weighted_wasted_private_blocks": wasted_private_blocks(plan),
        "estimated_hbm_bytes_per_step": bytes_estimate,
        "accepted_tokens_per_loaded_kv_block": (
            accepted / len(plan.union_blocks) if plan.union_blocks else None
        ),
        "accepted_tokens_per_estimated_hbm_byte": (
            accepted / bytes_estimate if bytes_estimate else None
        ),
        "fallback_tokens": len(plan.fallback_tokens),
        "candidate_source": trace.get("candidate_source", "unknown"),
    }


def _rho(accept_probs: list[float]) -> list[float]:
    out: list[float] = []
    prefix = 1.0
    for prob in accept_probs:
        out.append(prefix)
        prefix *= max(0.0, min(1.0, prob))
    return out


def expand_k_slices(traces: list[dict[str, Any]], k_slices: list[int]) -> list[dict[str, Any]]:
    if not k_slices:
        return traces
    out: list[dict[str, Any]] = []
    for trace in traces:
        candidates = trace.get("candidates") or []
        accept_probs = trace.get("accept_probs") or trace.get("a_i") or []
        for k in k_slices:
            if k <= 0 or len(candidates) < k or len(accept_probs) < k:
                continue
            sliced = copy.deepcopy(trace)
            sliced["candidates"] = candidates[:k]
            sliced["accept_probs"] = accept_probs[:k]
            sliced["num_spec_tokens"] = k
            for key in ("draft_tokens", "generated_token_ids", "rho", "risk", "residual_counts"):
                if isinstance(sliced.get(key), list):
                    sliced[key] = sliced[key][:k]
            out.append(sliced)
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("context_label", ""),
            row["layout"],
            row["num_spec_tokens"],
            row["block_size"],
            row["candidate_source"],
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "mean_blocks_per_token",
        "union_blocks",
        "jaccard_mean",
        "coverage_mean",
        "weighted_coverage",
        "private_unique_blocks",
        "weighted_wasted_private_blocks",
        "estimated_hbm_bytes_per_step",
        "accepted_tokens_per_loaded_kv_block",
        "accepted_tokens_per_estimated_hbm_byte",
        "fallback_tokens",
    ]
    for key, group in sorted(grouped.items()):
        context_label, layout, k, block_size, source = key
        row: dict[str, Any] = {
            "context_label": context_label,
            "layout": layout,
            "num_spec_tokens": k,
            "block_size": block_size,
            "candidate_source": source,
            "n_steps": len(group),
        }
        for metric in metrics:
            values = [item[metric] for item in group if item.get(metric) is not None]
            row[metric] = statistics.mean(values) if values else None
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No sparse layout rows.\n", encoding="utf-8")
        return
    cols = [
        "context_label",
        "layout",
        "num_spec_tokens",
        "block_size",
        "candidate_source",
        "n_steps",
        "union_blocks",
        "jaccard_mean",
        "coverage_mean",
        "weighted_coverage",
        "estimated_hbm_bytes_per_step",
        "accepted_tokens_per_loaded_kv_block",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--layouts",
        default="dense_verifier,independent_topk,snapkv_static,shared_only,speclink_fixed,speclink_prob,speclink_prob_fallback",
    )
    parser.add_argument(
        "--k-slices",
        default="",
        help="Optional comma-separated speculative lengths to simulate by truncating each trace, for example 2,4,8.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--topk-per-token", type=int, default=32)
    parser.add_argument("--shared-budget", type=int, default=32)
    parser.add_argument("--private-min", type=int, default=0)
    parser.add_argument("--private-max", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bytes-per-elem", type=int, default=2)
    args = parser.parse_args()

    traces = expand_k_slices(
        read_jsonl(Path(args.traces)),
        [int(item) for item in args.k_slices.split(",") if item.strip()],
    )
    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    per_trace_rows = [
        simulate_trace(trace, layout, args)
        for trace in traces
        for layout in layouts
    ]
    rows = aggregate(per_trace_rows)
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print(f"simulated {len(traces)} trace steps across {len(layouts)} layouts")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
