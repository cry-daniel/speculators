#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.acceptance_calibration import add_acceptance_priors, calibration_report, fit_temperature
from pams.configs import EXPERIMENTS, read_jsonl, register_experiment, write_json, write_jsonl
from pams.correctness import verifier_decision_metrics
from pams.mask_planner import TokenCandidates, plan_masks
from pams.metrics import auroc, jaccard, pearson, safe_mean
from pams.plotting import save_bar, save_line


METHODS = [
    "dense_all_blocks",
    "independent_topk",
    "shared_only",
    "shared_fixed_residual",
    "pams",
    "pams_fallback",
    "oracle_shared_residual",
]


def load_rows(trace_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "calibration": read_jsonl(trace_dir / "trace_calibration.jsonl"),
        "validation": read_jsonl(trace_dir / "trace_validation.jsonl"),
        "test": read_jsonl(trace_dir / "trace_test.jsonl"),
    }


def candidate_from_row(row: dict[str, Any]) -> TokenCandidates:
    return TokenCandidates(
        token_index=int(row["token_index"]),
        candidate_scores={int(k): float(v) for k, v in row["candidate_blocks"].items()},
        acceptance_prior=float(row.get("acceptance_prior", row.get("draft_probability", 0.5))),
        prefix_reach_probability=float(row.get("prefix_reach_probability", 1.0)),
        risk_score=float(row.get("risk_score", 0.0)),
        target_top_blocks={int(x) for x in row["target_dense_top_blocks"]},
    )


def group_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["block_id"])].append(row)
    for block_rows in grouped.values():
        block_rows.sort(key=lambda item: int(item["token_index"]))
    return grouped


def dense_total_blocks(rows: list[dict[str, Any]]) -> int:
    max_block = 0
    for row in rows:
        block_ids = [int(x) for x in row["candidate_blocks"].keys()] + [int(x) for x in row["target_dense_top_blocks"]]
        if block_ids:
            max_block = max(max_block, max(block_ids))
    return max_block + 1


def evaluate_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    grouped = group_by_block(rows)
    per_block = []
    decision_rows = []
    for block_rows in grouped.values():
        tokens = [candidate_from_row(row) for row in block_rows]
        total_blocks = dense_total_blocks(block_rows)
        plan = plan_masks(tokens, method=method, dense_total_blocks=total_blocks, topk=8, residual_budget=4)
        token_lens = []
        recalls = []
        accepted_weighted_recalls = []
        jaccards = []
        for left_idx, left in enumerate(tokens):
            selected = plan.token_blocks.get(left.token_index, set())
            token_lens.append(len(selected))
            target = left.target_top_blocks or set()
            recall = len(selected & target) / max(1, len(target))
            recalls.append(recall)
            accepted_weighted_recalls.append(recall * float(block_rows[left_idx]["accepted"]) * left.prefix_reach_probability)
            for right in tokens[left_idx + 1 :]:
                jaccards.append(jaccard(selected, plan.token_blocks.get(right.token_index, set())))
            dense_accept = bool(block_rows[left_idx]["accepted"])
            dense_fallback = left.token_index in plan.fallback_tokens
            sparse_accept = dense_accept if dense_fallback or method == "dense_all_blocks" else recall >= 0.50
            decision_rows.append({"dense_accept": dense_accept, "sparse_accept": sparse_accept, "dense_fallback": dense_fallback})
        union_len = len(plan.union_blocks)
        accepted_tokens = sum(int(row["accepted"]) for row in block_rows)
        per_block.append(
            {
                "mean_token_blocks": safe_mean(token_lens),
                "union_blocks": union_len,
                "union_growth_ratio": union_len / max(1e-9, safe_mean(token_lens)),
                "mask_jaccard_overlap": safe_mean(jaccards),
                "target_attention_top_block_recall": safe_mean(recalls),
                "accepted_token_weighted_recall": safe_mean(accepted_weighted_recalls),
                "accepted_tokens_per_loaded_block": accepted_tokens / max(1, union_len),
                "estimated_hbm_bytes_per_block": union_len * int(block_rows[0]["block_size"]) * 36 * 2 * 8 * 128 * 2,
                "dense_fallback_tokens": len(plan.fallback_tokens),
            }
        )
    correctness = verifier_decision_metrics(decision_rows)
    return {
        "method": method,
        "num_speculative_blocks": len(per_block),
        "average_mean_token_blocks": safe_mean([item["mean_token_blocks"] for item in per_block]),
        "average_union_blocks": safe_mean([item["union_blocks"] for item in per_block]),
        "union_growth_ratio": safe_mean([item["union_growth_ratio"] for item in per_block]),
        "mask_jaccard_overlap": safe_mean([item["mask_jaccard_overlap"] for item in per_block]),
        "target_attention_top_block_recall": safe_mean([item["target_attention_top_block_recall"] for item in per_block]),
        "accepted_token_weighted_recall": safe_mean([item["accepted_token_weighted_recall"] for item in per_block]),
        "estimated_hbm_bytes_per_speculative_block": safe_mean([item["estimated_hbm_bytes_per_block"] for item in per_block]),
        "accepted_tokens_per_loaded_block": safe_mean([item["accepted_tokens_per_loaded_block"] for item in per_block]),
        "dense_fallback_ratio": safe_mean([item["dense_fallback_tokens"] for item in per_block]) / 4.0,
        **correctness,
    }


def write_union_outputs(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [evaluate_method(test_rows, method) for method in METHODS if method != "pams_fallback"]
    exp = EXPERIMENTS / "03_union_problem"
    write_json(exp / "parsed" / "union_metrics.json", {"results": results})
    save_bar(
        exp / "figures" / "union_growth_vs_draft_len.png",
        [r["method"] for r in results],
        [r["union_growth_ratio"] for r in results],
        "Union growth by method",
        "Union growth ratio",
    )
    save_bar(
        exp / "figures" / "jaccard_overlap.png",
        [r["method"] for r in results],
        [r["mask_jaccard_overlap"] for r in results],
        "Mask Jaccard overlap",
        "Jaccard",
    )
    save_bar(
        exp / "figures" / "coverage_vs_union_blocks.png",
        [r["method"] for r in results],
        [r["target_attention_top_block_recall"] for r in results],
        "Target top-block recall",
        "Recall",
    )
    independent = next(r for r in results if r["method"] == "independent_topk")
    pams = next(r for r in results if r["method"] == "pams")
    register_experiment(
        "03_union_problem",
        config={"methods": [r["method"] for r in results], "block_sizes": [16, 32, 64], "label": "offline_simulation"},
        command="python scripts/run_offline_mask_analysis.py --trace-dir experiments/02_trace_collection/raw --output-dir experiments/05_mask_planner_offline",
        status="completed_offline_simulation",
        summary=(
            "# Phase 5 Union Problem\n\n"
            "This is an offline simulation over deterministic synthetic traces. It is motivation only and is not an end-to-end speed claim.\n\n"
            f"- Independent union growth ratio: `{independent['union_growth_ratio']:.3f}`\n"
            f"- PAMS union growth ratio: `{pams['union_growth_ratio']:.3f}`\n"
            f"- Independent accepted tokens per loaded block: `{independent['accepted_tokens_per_loaded_block']:.3f}`\n"
            f"- PAMS accepted tokens per loaded block: `{pams['accepted_tokens_per_loaded_block']:.3f}`\n"
        ),
        metadata_extra={"label": "offline_simulation", "metrics_json": str(exp / "parsed" / "union_metrics.json")},
    )
    return {"results": results}


def write_acceptance_outputs(rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    calibrator = fit_temperature(
        [float(row["draft_logit_margin"]) for row in rows_by_split["calibration"]],
        [int(row["accepted"]) for row in rows_by_split["calibration"]],
    )
    with_priors = {split: add_acceptance_priors(rows, calibrator) for split, rows in rows_by_split.items()}
    reports = {split: calibration_report(calibrator, rows) for split, rows in rows_by_split.items()}
    test = with_priors["test"]
    useful = [int(row["accepted"]) * int(row["reached"]) for row in test]
    rho = [float(row["prefix_reach_probability"]) for row in test]
    probs = [float(row["acceptance_prior"]) for row in test]
    reports["test"]["auroc_useful_from_rho"] = auroc(rho, useful)
    reports["test"]["rho_useful_correlation"] = pearson(rho, useful)
    reports["test"]["auroc_accept_from_prior"] = auroc(probs, [int(row["accepted"]) for row in test])
    exp = EXPERIMENTS / "04_acceptance_prior"
    write_json(exp / "parsed" / "acceptance_prior_metrics.json", reports)
    save_bar(
        exp / "figures" / "calibration_curve.png",
        ["ece", "brier", "auroc"],
        [reports["test"]["ece"], reports["test"]["brier"], reports["test"]["auroc_accept"]],
        "Acceptance prior quality",
        "Metric",
    )
    by_depth = defaultdict(list)
    for row in test:
        by_depth[int(row["token_index"])].append((1 - int(row["reached"])) * 4)
    save_bar(
        exp / "figures" / "wasted_blocks_by_depth.png",
        [str(k) for k in sorted(by_depth)],
        [safe_mean(v) for k, v in sorted(by_depth.items())],
        "Wasted private blocks by depth",
        "Estimated blocks",
    )
    sorted_pairs = sorted((r, u) for r, u in zip(rho, useful))
    save_line(
        exp / "figures" / "reach_probability_vs_usefulness.png",
        [p[0] for p in sorted_pairs[:: max(1, len(sorted_pairs) // 100)]],
        [p[1] for p in sorted_pairs[:: max(1, len(sorted_pairs) // 100)]],
        "Reach probability vs usefulness",
        "rho_i",
        "useful_i",
    )
    risk_pairs = sorted((float(row["risk_score"]), int(row["accepted"]) == 0) for row in test)
    save_line(
        exp / "figures" / "risk_vs_fallback.png",
        [p[0] for p in risk_pairs[:: max(1, len(risk_pairs) // 100)]],
        [float(p[1]) for p in risk_pairs[:: max(1, len(risk_pairs) // 100)]],
        "Risk score vs fallback need proxy",
        "risk_i",
        "not accepted",
    )
    register_experiment(
        "04_acceptance_prior",
        config={"calibration": "temperature_grid", "features": ["draft_logit_margin"], "splits": list(rows_by_split)},
        command="python scripts/run_offline_mask_analysis.py --trace-dir experiments/02_trace_collection/raw --output-dir experiments/05_mask_planner_offline",
        status="completed_offline_simulation",
        summary=(
            "# Phase 6 Acceptance Prior\n\n"
            "Temperature scaling is trained on calibration, selected once, and evaluated on validation/test. The current run uses synthetic fallback traces.\n\n"
            f"- Test ECE: `{reports['test']['ece']:.4f}`\n"
            f"- Test Brier: `{reports['test']['brier']:.4f}`\n"
            f"- AUROC accepted: `{reports['test']['auroc_accept']:.4f}`\n"
            f"- AUROC useful from rho_i: `{reports['test']['auroc_useful_from_rho']:.4f}`\n"
        ),
        metadata_extra={"label": "offline_simulation", "metrics_json": str(exp / "parsed" / "acceptance_prior_metrics.json")},
    )
    return reports, with_priors


def write_mask_outputs(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [evaluate_method(test_rows, method) for method in METHODS]
    exp = EXPERIMENTS / "05_mask_planner_offline"
    write_json(exp / "parsed" / "mask_planner_metrics.json", {"results": results})
    write_jsonl(exp / "raw" / "test_with_priors.jsonl", test_rows)
    labels = [r["method"] for r in results]
    save_bar(
        exp / "figures" / "pareto_quality_vs_blocks.png",
        labels,
        [r["target_attention_top_block_recall"] / max(r["average_union_blocks"], 1e-6) for r in results],
        "Quality per loaded block",
        "Recall/block",
    )
    save_bar(
        exp / "figures" / "pareto_quality_vs_loaded_blocks.png",
        labels,
        [r["accepted_tokens_per_loaded_block"] for r in results],
        "Accepted tokens per loaded block",
        "Accepted/block",
    )
    save_bar(
        exp / "figures" / "false_accept_reject.png",
        labels,
        [r["false_accept_rate"] + r["false_reject_rate"] for r in results],
        "Sparse decision mismatch",
        "False accept + false reject",
    )
    save_bar(
        exp / "figures" / "accepted_tokens_per_loaded_block.png",
        labels,
        [r["accepted_tokens_per_loaded_block"] for r in results],
        "Accepted tokens per loaded block",
        "Accepted/block",
    )
    save_bar(
        exp / "figures" / "fallback_tradeoff.png",
        labels,
        [r["dense_fallback_ratio"] for r in results],
        "Dense fallback ratio",
        "Fallback ratio",
    )
    pams = next(r for r in results if r["method"] == "pams")
    pams_fb = next(r for r in results if r["method"] == "pams_fallback")
    register_experiment(
        "05_mask_planner_offline",
        config={"methods": labels, "hyperparameter_selection": "fixed_smoke_values_validation_ready"},
        command="python scripts/run_offline_mask_analysis.py --trace-dir experiments/02_trace_collection/raw --output-dir experiments/05_mask_planner_offline",
        status="completed_offline_simulation",
        summary=(
            "# Phase 8 Offline Mask Planner\n\n"
            "Offline sparse-verifier quality is evaluated on the test split after calibration. This is not an end-to-end speed claim.\n\n"
            f"- PAMS accepted tokens per loaded block: `{pams['accepted_tokens_per_loaded_block']:.3f}`\n"
            f"- PAMS false accept rate: `{pams['false_accept_rate']:.4f}`\n"
            f"- PAMS_Fallback false accept rate: `{pams_fb['false_accept_rate']:.4f}`\n"
            f"- PAMS_Fallback dense fallback ratio: `{pams_fb['dense_fallback_ratio']:.4f}`\n"
        ),
        metadata_extra={"label": "offline_simulation", "metrics_json": str(exp / "parsed" / "mask_planner_metrics.json")},
    )
    return {"results": results}


def write_ablations(test_rows: list[dict[str, Any]]) -> None:
    # The smoke ablation uses planner variants that correspond to the requested knobs.
    variants = {
        "no_acceptance_prior": dict(lambda_risk=0.0, alpha_reach=0.0, beta_risk=1.0),
        "no_reach_probability": dict(lambda_risk=1.0, alpha_reach=0.0, beta_risk=1.0),
        "no_risk_term": dict(lambda_risk=0.0, alpha_reach=1.0, beta_risk=0.0),
        "shared_only": None,
        "independent_topk": None,
        "fixed_residual": None,
        "no_fallback": None,
        "dense_fallback_all_early": dict(fallback_threshold=-1.0),
    }
    rows = []
    grouped = group_by_block(test_rows)
    for name, params in variants.items():
        method = {
            "shared_only": "shared_only",
            "independent_topk": "independent_topk",
            "fixed_residual": "shared_fixed_residual",
            "no_fallback": "pams",
        }.get(name, "pams_fallback")
        decision_rows = []
        loaded_blocks = []
        for block_rows in grouped.values():
            tokens = [candidate_from_row(row) for row in block_rows]
            kwargs = params or {}
            plan = plan_masks(tokens, method=method, dense_total_blocks=dense_total_blocks(block_rows), topk=8, residual_budget=4, **kwargs)
            loaded_blocks.append(len(plan.union_blocks))
            for idx, token in enumerate(tokens):
                selected = plan.token_blocks.get(token.token_index, set())
                target = token.target_top_blocks or set()
                recall = len(selected & target) / max(1, len(target))
                dense = bool(block_rows[idx]["accepted"])
                fallback = token.token_index in plan.fallback_tokens
                sparse = dense if fallback else recall >= 0.5
                decision_rows.append({"dense_accept": dense, "sparse_accept": sparse, "dense_fallback": fallback})
        metrics = verifier_decision_metrics(decision_rows)
        rows.append({"ablation": name, "avg_loaded_blocks": safe_mean(loaded_blocks), **metrics})
    exp = EXPERIMENTS / "12_ablations"
    write_json(exp / "parsed" / "ablation_metrics.json", {"results": rows})
    save_bar(
        exp / "figures" / "ablation_summary.png",
        [r["ablation"] for r in rows],
        [r["false_accept_rate"] + r["false_reject_rate"] for r in rows],
        "Ablation mismatch summary",
        "False accept + false reject",
    )
    register_experiment(
        "12_ablations",
        config={"label": "offline_simulation", "variants": list(variants)},
        command="python scripts/run_offline_mask_analysis.py --trace-dir experiments/02_trace_collection/raw --output-dir experiments/05_mask_planner_offline",
        status="completed_offline_simulation",
        summary="# Phase 13 Ablations\n\nOffline ablations were run on the synthetic test trace. End-to-end ablations remain unavailable until vLLM integration succeeds.",
        metadata_extra={"label": "offline_simulation", "metrics_json": str(exp / "parsed" / "ablation_metrics.json")},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=EXPERIMENTS / "02_trace_collection" / "raw")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "05_mask_planner_offline")
    args = parser.parse_args()
    rows_by_split = load_rows(args.trace_dir)
    _, with_priors = write_acceptance_outputs(rows_by_split)
    write_union_outputs(with_priors["test"])
    write_mask_outputs(with_priors["test"])
    write_ablations(with_priors["test"])
    print(json.dumps({"status": "completed_offline_simulation", "test_rows": len(with_priors["test"])}, indent=2))


if __name__ == "__main__":
    main()

