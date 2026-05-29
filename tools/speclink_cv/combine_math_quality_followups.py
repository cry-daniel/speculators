#!/usr/bin/env python3
"""Combine staged-CV math-quality follow-up roots into one summary.

The staged-CV math runs were produced in a few bounded slices while debugging
performance.  This utility makes the current evidence easier to audit by
pairing every CV row with its EAGLE3 baseline and writing one all-batch table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "examples/evaluate/eval-guidellm"
DEFAULT_ROOTS = [
    EVAL_ROOT / "temp/speclink_cv_math_pure_vllm_baseline_20260528",
    EVAL_ROOT / "temp/speclink_cv_math_k8_k12_bs8_bs32_staged_quality_20260528",
    EVAL_ROOT / "temp/speclink_cv_math_k8_k12_bs16_staged_quality_req64_20260528",
    EVAL_ROOT / "temp/speclink_cv_math_k8_k12_bs16_staged_quality_20260528",
]
DEFAULT_OUTPUT = (
    EVAL_ROOT / "temp/speclink_cv_math_k8_k12_all_bs_staged_quality_combined_20260528"
)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def ratio(numer: Any, denom: Any) -> str:
    numer_f = safe_float(numer)
    denom_f = safe_float(denom)
    if numer_f is None or denom_f in (None, 0.0):
        return ""
    return str(numer_f / denom_f)


def prefer_row(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    """Prefer fuller reruns when the same scenario appears in multiple roots."""
    if existing is None:
        return candidate
    existing_key = (
        safe_int(existing.get("successful_requests")) or 0,
        safe_int(existing.get("quality_evaluable")) or 0,
        safe_float(existing.get("quality_max_tokens")) or 0.0,
    )
    candidate_key = (
        safe_int(candidate.get("successful_requests")) or 0,
        safe_int(candidate.get("quality_evaluable")) or 0,
        safe_float(candidate.get("quality_max_tokens")) or 0.0,
    )
    return candidate if candidate_key > existing_key else existing


def build_by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("measurement_type") != "guidellm_end_to_end":
            continue
        key = (
            row.get("model"),
            row.get("dataset"),
            row.get("K"),
            row.get("batch_size"),
            row.get("method"),
        )
        by_key[key] = prefer_row(by_key.get(key), row)
    return by_key


def collect_rows(roots: list[Path]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    steady_rows: list[dict[str, Any]] = []
    for root in roots:
        for row in read_csv(root / "summary_metrics.csv"):
            row = dict(row)
            row["source_root"] = str(root)
            summary_rows.append(row)
        for row in read_csv(root / "09_reports/steady_state_throughput.csv"):
            row = dict(row)
            row["source_root"] = str(root)
            steady_rows.append(row)

    by_key = build_by_key(summary_rows)
    steady_by_key = build_by_key(steady_rows)

    rows: list[dict[str, Any]] = []
    for key, cv in by_key.items():
        model, dataset, k, batch_size, method = key
        if dataset != "math" or method != "cv_half_async_staged_simple":
            continue
        baseline = by_key.get((model, dataset, k, batch_size, "eagle3_oneshot"), {})
        pure = by_key.get((model, dataset, k, batch_size, "pure_vllm"), {})
        if not pure:
            # pure_vllm does not depend on K, so allow a same-model/dataset/batch
            # row from any K slice to serve both K=8 and K=12 summaries.
            pure_candidates = [
                row
                for p_key, row in by_key.items()
                if p_key[0] == model
                and p_key[1] == dataset
                and p_key[3] == batch_size
                and p_key[4] == "pure_vllm"
            ]
            if pure_candidates:
                pure = sorted(
                    pure_candidates,
                    key=lambda row: (
                        safe_int(row.get("successful_requests")) or 0,
                        safe_int(row.get("quality_evaluable")) or 0,
                    ),
                    reverse=True,
                )[0]
        steady = steady_by_key.get(key, {})
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "K": k,
                "batch_size": batch_size,
                "method": method,
                "requests": cv.get("successful_requests", ""),
                "max_tokens": cv.get("quality_max_tokens", ""),
                "pure_vllm_tps": pure.get("throughput", ""),
                "eagle3_tps": baseline.get("throughput", ""),
                "cv_tps": cv.get("throughput", ""),
                "eagle3_speedup_vs_pure": ratio(
                    baseline.get("throughput", ""), pure.get("throughput", "")
                ),
                "cv_speedup_vs_pure": ratio(
                    cv.get("throughput", ""), pure.get("throughput", "")
                ),
                "e2e_speedup_vs_eagle3": cv.get("speedup_vs_eagle3", ""),
                "steady_speedup_vs_eagle3": steady.get("steady_speedup_vs_eagle3", ""),
                "eagle3_actual_average_batch_size": baseline.get(
                    "actual_average_batch_size", ""
                ),
                "cv_actual_average_batch_size": cv.get("actual_average_batch_size", ""),
                "eagle3_actual_scheduled_seqs_per_step": baseline.get(
                    "actual_scheduled_seqs_per_step", ""
                ),
                "cv_actual_scheduled_seqs_per_step": cv.get(
                    "actual_scheduled_seqs_per_step", ""
                ),
                "eagle3_actual_scheduled_tokens_per_step": baseline.get(
                    "actual_scheduled_tokens_per_step", ""
                ),
                "cv_actual_scheduled_tokens_per_step": cv.get(
                    "actual_scheduled_tokens_per_step", ""
                ),
                "eagle3_itl_p95": baseline.get("itl_p95", ""),
                "cv_itl_p95": cv.get("itl_p95", ""),
                "eagle3_e2e_p95": baseline.get("e2e_p95", ""),
                "cv_e2e_p95": cv.get("e2e_p95", ""),
                "quality_score_eagle3": baseline.get("quality_score", ""),
                "quality_score_cv": cv.get("quality_score", ""),
                "quality_correct_pure": pure.get("quality_correct", ""),
                "quality_correct_eagle3": baseline.get("quality_correct", ""),
                "quality_correct_cv": cv.get("quality_correct", ""),
                "quality_evaluable": cv.get("quality_evaluable", ""),
                "quality_delta_vs_eagle3": cv.get("quality_delta_vs_eagle3", ""),
                "quality_gate_status": cv.get("quality_gate_status", ""),
                "skipped_suffix_ratio": cv.get("skipped_suffix_ratio", ""),
                "verify_target_token_ratio_vs_oneshot_est": cv.get(
                    "verify_target_token_ratio_vs_oneshot_est", ""
                ),
                "draft_tokens_saved_by_staging_est": cv.get(
                    "draft_tokens_saved_by_staging_est", ""
                ),
                "draft_discard_ratio_est": cv.get("draft_discard_ratio_est", ""),
                "prefix_full_accept_ratio": cv.get("prefix_accepted_ratio", ""),
                "prefix_dispatch_seq_util_avg": cv.get("prefix_dispatch_seq_util_avg", ""),
                "prefix_underfilled_dispatch_ratio": cv.get(
                    "prefix_underfilled_dispatch_ratio", ""
                ),
                "gpu_active_util_eagle3": baseline.get("gpu_active_util", ""),
                "gpu_active_util_cv": cv.get("gpu_active_util", ""),
                "pure_source_root": pure.get("source_root", ""),
                "source_root": cv.get("source_root", ""),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("model", "")),
            safe_int(row.get("K")) or 0,
            safe_int(row.get("batch_size")) or 0,
        ),
    )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Staged CV Math-Quality All-Batch Summary",
        "",
        "This report combines the existing staged-CV math follow-up slices. "
        "Rows are quality-gated by math answer exact match, not by byte-for-byte "
        "EAGLE3 token identity.",
        "",
        "Scope boundary: this is the relaxed math-quality performance view. It "
        "does not close the original TODO.md strict greedy token-id correctness "
        "gate for live h<K chunking.",
        "",
        "| model | K | batch | requests | pure tok/s | EAGLE3 tok/s | CV tok/s | CV/EAGLE3 | CV/pure | EAGLE3 EM | CV EM | quality gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('model', '')} | {row.get('K', '')} | "
            f"{row.get('batch_size', '')} | {row.get('requests', '')} | "
            f"{fmt(row.get('pure_vllm_tps'), 1)} | "
            f"{fmt(row.get('eagle3_tps'), 1)} | {fmt(row.get('cv_tps'), 1)} | "
            f"{fmt(row.get('e2e_speedup_vs_eagle3'))} | "
            f"{fmt(row.get('cv_speedup_vs_pure'))} | "
            f"{row.get('quality_correct_eagle3', '')}/{row.get('quality_evaluable', '')} | "
            f"{row.get('quality_correct_cv', '')}/{row.get('quality_evaluable', '')} | "
            f"{row.get('quality_gate_status', '')} |"
        )
    valid = [
        row for row in rows
        if row.get("quality_gate_status") == "math_quality_preserved"
    ]
    speedups = [
        safe_float(row.get("e2e_speedup_vs_eagle3"))
        for row in valid
        if safe_float(row.get("e2e_speedup_vs_eagle3")) is not None
    ]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- rows: `{len(rows)}`",
            f"- quality-preserving rows: `{len(valid)}`",
            f"- rows with end-to-end speedup > 1: `{sum(1 for value in speedups if value and value > 1.0)}`",
            f"- mean speedup over quality-preserving rows: `{sum(speedups) / len(speedups):.3f}`"
            if speedups
            else "- mean speedup over quality-preserving rows: ``",
            "",
            "Notes:",
            "",
            "- If both 32-request and 64-request bs=16 slices are present, this "
            "summary keeps the fuller 64-request row for that scenario.",
            "- `pure tok/s` is blank until a matching pure-vLLM result root is present.",
            "- `pure_vllm` does not use K; when only one K slice is available for a "
            "model/batch pair, the combiner reuses it for the other K row.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-speclink-cv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = {
        ("qwen3_8b", "8"): "#1f77b4",
        ("qwen3_8b", "12"): "#ff7f0e",
        ("llama3_1_8b", "8"): "#2ca02c",
        ("llama3_1_8b", "12"): "#d62728",
    }
    markers = {"qwen3_8b": "o", "llama3_1_8b": "s"}
    for model in sorted({str(row["model"]) for row in rows}):
        for k in sorted({str(row["K"]) for row in rows if row["model"] == model}, key=int):
            subset = [
                row for row in rows
                if row.get("model") == model and str(row.get("K")) == k
            ]
            subset.sort(key=lambda row: safe_int(row.get("batch_size")) or 0)
            x = [safe_int(row.get("batch_size")) for row in subset]
            y = [safe_float(row.get("e2e_speedup_vs_eagle3")) for row in subset]
            label = f"{model.replace('_1_', ' ').replace('_', '-')} K={k}"
            ax.plot(
                x,
                y,
                marker=markers.get(model, "o"),
                linewidth=2,
                color=colors.get((model, k)),
                label=label,
            )
    ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("CV / EAGLE3 generated token throughput")
    ax.set_title("Staged CV Math-Quality End-to-End Speedup")
    ax.set_xticks([8, 16, 32])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_absolute_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-speclink-cv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({str(row["model"]) for row in rows})
    ks = sorted({str(row["K"]) for row in rows}, key=int)
    fig, axes = plt.subplots(
        len(models),
        len(ks),
        figsize=(13, 6.8),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    colors = {
        "pure_vllm_tps": "#7f7f7f",
        "eagle3_tps": "#1f77b4",
        "cv_tps": "#ff7f0e",
    }
    labels = {
        "pure_vllm_tps": "pure vLLM",
        "eagle3_tps": "EAGLE3",
        "cv_tps": "staged CV",
    }
    for row_i, model in enumerate(models):
        for col_i, k in enumerate(ks):
            ax = axes[row_i][col_i]
            subset = [
                row for row in rows
                if str(row.get("model")) == model and str(row.get("K")) == k
            ]
            subset.sort(key=lambda row: safe_int(row.get("batch_size")) or 0)
            x = [safe_int(row.get("batch_size")) for row in subset]
            for field in ("pure_vllm_tps", "eagle3_tps", "cv_tps"):
                y = [safe_float(row.get(field)) for row in subset]
                if not any(value is not None for value in y):
                    continue
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    label=labels[field],
                    color=colors[field],
                )
            ax.set_title(f"{model.replace('_1_', ' ').replace('_', '-')} K={k}")
            ax.set_xticks([8, 16, 32])
            ax.grid(True, axis="y", alpha=0.3)
            if col_i == 0:
                ax.set_ylabel("Generated tokens/s")
            if row_i == len(models) - 1:
                ax.set_xlabel("Batch size")
    handles, labels_seen = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_seen, loc="upper center", ncol=3, frameon=True)
    fig.suptitle("Math-Quality Absolute Throughput", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def classify_speedup(row: dict[str, Any]) -> str:
    speedup = safe_float(row.get("e2e_speedup_vs_eagle3"))
    if speedup is None:
        return "missing"
    if speedup < 1.0:
        return "regression"
    if speedup < 1.1:
        return "low_gain"
    if speedup < 1.3:
        return "moderate_gain"
    return "high_gain"


def diagnosis_for(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    speedup = safe_float(row.get("e2e_speedup_vs_eagle3"))
    k = safe_int(row.get("K"))
    dispatch_util = safe_float(row.get("prefix_dispatch_seq_util_avg"))
    underfilled = safe_float(row.get("prefix_underfilled_dispatch_ratio"))
    cv_batch = safe_float(row.get("cv_actual_average_batch_size"))
    eagle_batch = safe_float(row.get("eagle3_actual_average_batch_size"))
    skipped = safe_float(row.get("skipped_suffix_ratio"))
    gpu_cv = safe_float(row.get("gpu_active_util_cv"))
    gpu_eagle = safe_float(row.get("gpu_active_util_eagle3"))

    if k == 8:
        reasons.append("K=8 has less suffix work to remove than K=12")
    if dispatch_util is not None and dispatch_util < 0.55:
        reasons.append("prefix dispatch is underfilled")
    if underfilled is not None and underfilled >= 0.8:
        reasons.append("most prefix dispatches are below the target batch")
    if cv_batch is not None and eagle_batch is not None and cv_batch < 0.8 * eagle_batch:
        reasons.append("CV lowers effective average batch size")
    if skipped is not None and skipped < 0.4:
        reasons.append("skipped suffix ratio is modest")
    if gpu_cv is not None and gpu_eagle is not None and gpu_cv < gpu_eagle - 5.0:
        reasons.append("CV active GPU utilization is lower than EAGLE3")
    if speedup is not None and speedup >= 1.3:
        reasons.append("staged drafting and verifier skipping both materialize")
    if not reasons:
        reasons.append("gain is mostly bounded by serving overhead and tail drain")
    return "; ".join(reasons)


def write_diagnosis(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "model": row.get("model", ""),
                "K": row.get("K", ""),
                "batch_size": row.get("batch_size", ""),
                "requests": row.get("requests", ""),
                "e2e_speedup_vs_eagle3": row.get("e2e_speedup_vs_eagle3", ""),
                "eagle3_speedup_vs_pure": row.get("eagle3_speedup_vs_pure", ""),
                "cv_speedup_vs_pure": row.get("cv_speedup_vs_pure", ""),
                "steady_speedup_vs_eagle3": row.get("steady_speedup_vs_eagle3", ""),
                "class": classify_speedup(row),
                "skipped_suffix_ratio": row.get("skipped_suffix_ratio", ""),
                "verify_target_token_ratio_vs_oneshot_est": row.get(
                    "verify_target_token_ratio_vs_oneshot_est", ""
                ),
                "draft_tokens_saved_by_staging_est": row.get(
                    "draft_tokens_saved_by_staging_est", ""
                ),
                "draft_discard_ratio_est": row.get("draft_discard_ratio_est", ""),
                "eagle3_actual_average_batch_size": row.get(
                    "eagle3_actual_average_batch_size", ""
                ),
                "cv_actual_average_batch_size": row.get(
                    "cv_actual_average_batch_size", ""
                ),
                "prefix_dispatch_seq_util_avg": row.get(
                    "prefix_dispatch_seq_util_avg", ""
                ),
                "prefix_underfilled_dispatch_ratio": row.get(
                    "prefix_underfilled_dispatch_ratio", ""
                ),
                "gpu_active_util_eagle3": row.get("gpu_active_util_eagle3", ""),
                "gpu_active_util_cv": row.get("gpu_active_util_cv", ""),
                "diagnosis": diagnosis_for(row),
            }
        )

    write_csv(report_dir / "math_quality_performance_diagnosis.csv", records)

    lines = [
        "# Math-Quality Performance Diagnosis",
        "",
        "This report explains the relaxed math-quality speedup rows. It should "
        "not be read as proof of the original strict token-id TODO gate.",
        "",
        "| model | K | batch | speedup | skipped suffix | CV avg batch | EAGLE3 avg batch | prefix util | diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        lines.append(
            f"| {row['model']} | {row['K']} | {row['batch_size']} | "
            f"{fmt(row['e2e_speedup_vs_eagle3'])} | "
            f"{fmt(row['skipped_suffix_ratio'])} | "
            f"{fmt(row['cv_actual_average_batch_size'], 2)} | "
            f"{fmt(row['eagle3_actual_average_batch_size'], 2)} | "
            f"{fmt(row['prefix_dispatch_seq_util_avg'])} | "
            f"{row['diagnosis']} |"
        )

    low = [row for row in records if row["class"] in {"regression", "low_gain"}]
    high = [row for row in records if row["class"] == "high_gain"]
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            f"- Low-gain or regression rows: `{len(low)}`.",
            f"- High-gain rows (>=1.3x): `{len(high)}`.",
            "- The strongest gains require both verifier suffix skipping and staged "
            "draft suffix saving; prefix-only verifier savings are often hidden by "
            "small-batch dispatch, scheduling overhead, or fixed-request tail drain.",
            "- K=8 rows are expected to be weaker because the removable suffix is "
            "shorter while the extra prefix pass overhead remains.",
            "- Rows where CV average batch size drops substantially versus EAGLE3 "
            "should be treated as scheduler-utilization problems, not as evidence "
            "that the pruning opportunity is absent.",
        ]
    )
    (report_dir / "math_quality_performance_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Result roots containing summary_metrics.csv.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [path.resolve() for path in args.roots]
    output_root = args.output_root.resolve()
    rows = collect_rows(roots)
    report_dir = output_root / "09_reports"
    figure_dir = output_root / "08_figures"
    write_csv(report_dir / "math_quality_all_batch_summary.csv", rows)
    write_json(report_dir / "math_quality_all_batch_summary.json", {"roots": roots, "rows": rows})
    write_markdown(report_dir / "math_quality_all_batch_summary.md", rows)
    write_diagnosis(report_dir, rows)
    write_plot(figure_dir / "math_quality_speedup_curve.png", rows)
    write_absolute_plot(figure_dir / "math_quality_absolute_throughput.png", rows)
    print(f"[INFO] wrote combined math-quality summary: {output_root}")
    print(f"[INFO] rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
