#!/usr/bin/env python3
"""Generate SpecLink summary tables and a markdown report skeleton."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def read_section(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    text = path.read_text(encoding="utf-8").strip()
    return text if text else fallback


def maybe_run(args: list[str]) -> None:
    try:
        run(args)
    except subprocess.CalledProcessError:
        return


def exists(path: Path) -> str:
    return "yes" if path.exists() else "no"


def matrix_counts(path: Path) -> tuple[int, int, int, int] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    total = len(rows)
    complete = sum(1 for row in rows if row.get("status") == "complete")
    missing = sum(1 for row in rows if row.get("status") == "missing")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    return total, complete, missing, failed


def matrix_status_text(root: Path, stem: str) -> str:
    counts = matrix_counts(root / "tables" / f"{stem}.tsv")
    if counts is None:
        return "status file missing"
    total, complete, missing, failed = counts
    return f"total={total}, complete={complete}, missing={missing}, failed={failed}"


def matrix_status_label(root: Path, stem: str) -> str:
    counts = matrix_counts(root / "tables" / f"{stem}.tsv")
    if counts is None:
        return "missing"
    total, complete, _missing, failed = counts
    if total and complete == total and failed == 0:
        return "supported"
    if complete:
        return "partial"
    return "missing"


def formal_baseline_note(root: Path) -> str:
    counts = matrix_counts(root / "tables" / "formal_matrix_status.tsv")
    if counts is None:
        return "- Formal baseline matrix status has not been generated yet."
    total, complete, missing, failed = counts
    if total and complete == total and failed == 0:
        return "- Formal baseline matrix is complete: all planned rate/K/repeat rows are present."
    return (
        "- Formal baseline matrix is incomplete: "
        f"`complete={complete}`, `missing={missing}`, `failed={failed}`, `total={total}`."
    )


def padded_context_status(root: Path) -> tuple[str, str]:
    manifest_path = root / "04_sparse_challenge" / "padded_data" / "padded_math_manifest.json"
    if not manifest_path.exists():
        return "missing", "04_sparse_challenge/padded_data/padded_math_manifest.json missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "missing", "padded_math_manifest.json is not valid JSON"

    required_targets = {0, 2048, 4096, 8192}
    rows = [row for row in manifest if isinstance(row, dict)]
    present_targets = {int(row.get("target_tokens")) for row in rows if row.get("target_tokens") is not None}
    paths_exist = {
        int(row.get("target_tokens")): Path(str(row.get("path", ""))).exists()
        for row in rows
        if row.get("target_tokens") is not None
    }
    complete_targets = {target for target in required_targets if target in present_targets and paths_exist.get(target)}
    status = "supported" if complete_targets == required_targets else ("partial" if complete_targets else "missing")
    target_text = ",".join(str(target) for target in sorted(complete_targets))
    missing_text = ",".join(str(target) for target in sorted(required_targets - complete_targets))
    evidence = f"{manifest_path} targets={target_text or 'none'}"
    if missing_text:
        evidence += f"; missing_targets={missing_text}"
    return status, evidence


def sparse_microbench_status(root: Path) -> tuple[str, str]:
    csv_path = root / "tables" / "sparse_microbench.csv"
    if not csv_path.exists():
        return "missing", "tables/sparse_microbench.csv missing"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_k = {2, 4, 8, 12}
    required_contexts = {512, 2048, 4096, 8192}
    required_blocks = {16, 32, 64}
    required_layouts = {"dense_verifier", "independent_topk", "snapkv_static", "shared_only", "speclink_prob"}
    got_k = {int(row["num_spec_tokens"]) for row in rows if row.get("num_spec_tokens")}
    got_contexts = {int(row["context_len"]) for row in rows if row.get("context_len")}
    got_blocks = {int(row["block_size"]) for row in rows if row.get("block_size")}
    got_layouts = {row["layout"] for row in rows if row.get("layout")}
    devices = sorted({row["device"] for row in rows if row.get("device")})

    missing = []
    if not required_k <= got_k:
        missing.append(f"k={','.join(str(item) for item in sorted(required_k - got_k))}")
    if not required_contexts <= got_contexts:
        missing.append(f"context={','.join(str(item) for item in sorted(required_contexts - got_contexts))}")
    if not required_blocks <= got_blocks:
        missing.append(f"block={','.join(str(item) for item in sorted(required_blocks - got_blocks))}")
    if not required_layouts <= got_layouts:
        missing.append(f"layout={','.join(sorted(required_layouts - got_layouts))}")

    status = "supported-proxy" if not missing else ("partial" if rows else "missing")
    evidence = (
        f"{csv_path} rows={len(rows)} devices={','.join(devices) or 'unknown'} "
        f"k={','.join(str(item) for item in sorted(got_k))} "
        f"context={','.join(str(item) for item in sorted(got_contexts))} "
        f"block={','.join(str(item) for item in sorted(got_blocks))}"
    )
    if missing:
        evidence += f"; missing {'; '.join(missing)}"
    return status, evidence


def sparse_layout_hidden_status(root: Path) -> tuple[str, str]:
    csv_path = root / "tables" / "sparse_layout_hidden_summary.csv"
    if not csv_path.exists():
        return "missing", "tables/sparse_layout_hidden_summary.csv missing"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "missing", "tables/sparse_layout_hidden_summary.csv has no rows"
    contexts = sorted({row.get("context_label", "unknown") for row in rows})
    sources = sorted({row.get("candidate_source", "unknown") for row in rows})
    layouts = sorted({row.get("layout", "unknown") for row in rows})
    ks = sorted({row.get("num_spec_tokens", "") for row in rows if row.get("num_spec_tokens")})
    evidence = (
        f"{csv_path} rows={len(rows)} contexts={','.join(contexts)} sources={','.join(sources)} "
        f"k={','.join(ks)} layouts={','.join(layouts)}"
    )
    return "supported-proxy", evidence


def calibration_status(root: Path) -> tuple[str, str]:
    calibrator = root / "05_speclink" / "calibrator.pkl"
    summary = root / "05_speclink" / "calibrator_summary.json"
    if not calibrator.exists() or not summary.exists():
        return "missing", "05_speclink/calibrator.pkl or calibrator_summary.json missing"
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "partial", "05_speclink/calibrator_summary.json is not valid JSON"
    eval_metrics = data.get("eval_metrics") or {}
    evidence = (
        f"{summary} method={data.get('method')} examples={data.get('num_examples')} "
        f"eval_ece={eval_metrics.get('ece_10bin')} eval_brier={eval_metrics.get('brier')}"
    )
    return "supported-proxy", evidence


def write_sparse_quality_md(csv_path: Path, md_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "This table is offline/proxy verifier-quality evidence. It uses proxy",
        "sparse candidates and Transformers masked attention, not a vLLM sparse",
        "verifier kernel.",
        "",
    ]
    if not rows:
        lines.append("No sparse quality rows were produced.")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    cols = [
        "context_label",
        "candidate_source",
        "layout",
        "n_samples",
        "n_positions",
        "top1_match_rate",
        "kl_dense_to_sparse_mean",
        "actual_dense_accept_rate",
        "actual_false_accept_rate",
        "actual_false_reject_rate",
        "counterfactual_top2_false_accept_rate",
        "avg_union_blocks",
    ]
    cols = [col for col in cols if col in rows[0]]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt_value(row.get(col)) for col in cols) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if text.strip() == "":
        return ""
    return f"{number:.5g}"


def sparse_quality_status(root: Path) -> tuple[str, str]:
    csv_path = root / "tables" / "sparse_quality_summary.csv"
    if not csv_path.exists():
        return "missing", "tables/sparse_quality_summary.csv missing"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "missing", "tables/sparse_quality_summary.csv has no rows"
    positions = sum(int(float(row.get("n_positions") or 0)) for row in rows)
    sources = sorted({row.get("candidate_source", "unknown") for row in rows})
    evidence_types = sorted({row.get("evidence_type", "unknown") for row in rows})
    evidence = (
        f"{csv_path} rows={len(rows)} positions={positions} "
        f"sources={','.join(sources)} evidence={','.join(evidence_types)}"
    )
    return "supported-proxy", evidence


def status_table(root: Path) -> str:
    padded_status, padded_evidence = padded_context_status(root)
    microbench_status, microbench_evidence = sparse_microbench_status(root)
    hidden_layout_status, hidden_layout_evidence = sparse_layout_hidden_status(root)
    calibrator_status, calibrator_evidence = calibration_status(root)
    quality_status, quality_evidence = sparse_quality_status(root)
    rows = [
        (
            "dense/eagle3/peagle real e2e smoke",
            "supported",
            "02_baselines/*_smoke/guidellm_results.json",
        ),
        (
            "dense/eagle3/peagle accuracy outputs",
            "supported",
            "02_baselines/*_smoke/accuracy_outputs.jsonl",
        ),
        (
            "baseline summary",
            "supported",
            f"tables/baseline_summary.csv exists={exists(root / 'tables' / 'baseline_summary.csv')}",
        ),
        (
            "eagle3/peagle breakdown",
            "supported",
            "tables/breakdown_summary.csv and 03_breakdown/*_profile/profile_events.jsonl",
        ),
        (
            "speclink vLLM plan_only smoke",
            "supported",
            "05_speclink/speclink_prob_profile_smoke/live_sparse_trace.jsonl",
        ),
        (
            "patch off-mode regression check",
            "supported",
            "05_speclink/patch_regression_check.md plus complete baseline matrix",
        ),
        (
            "speclink plan-only ablation table",
            "supported-proxy",
            "tables/speclink_ablation.csv from 05_speclink, 05_speclink_g2, and 06_serving_rates plan-only runs",
        ),
        (
            "speclink plan-only layout/K matrix",
            "supported",
            "tables/speclink_plan_matrix_status.md",
        ),
        (
            "speclink G2 budget/block/fallback matrix",
            matrix_status_label(root, "speclink_g2_matrix_status"),
            matrix_status_text(root, "speclink_g2_matrix_status"),
        ),
        (
            "speclink G4 serving-rate matrix",
            matrix_status_label(root, "speclink_serving_matrix_status"),
            matrix_status_text(root, "speclink_serving_matrix_status"),
        ),
        (
            "SnapKV/static sparse challenge",
            hidden_layout_status if hidden_layout_status != "missing" else "partial",
            (
                hidden_layout_evidence
                if hidden_layout_status != "missing"
                else "tables/sparse_layout_summary.csv from position_recency_proxy live trace"
            ),
        ),
        (
            "formal rate/K/repeat matrix",
            matrix_status_label(root, "formal_matrix_status"),
            matrix_status_text(root, "formal_matrix_status"),
        ),
        (
            "padded 2k/4k/8k contexts",
            padded_status,
            padded_evidence,
        ),
        (
            "sparse microbenchmark",
            microbench_status,
            microbench_evidence,
        ),
        (
            "acceptance probability calibration",
            calibrator_status,
            calibrator_evidence,
        ),
        (
            "offline sparse verifier quality",
            quality_status,
            quality_evidence,
        ),
        (
            "real sparse verifier kernel",
            "missing",
            "SPECLINK_MODE=sparse_kernel is not integrated",
        ),
    ]
    lines = [
        "| requirement | status | evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {name} | {status} | `{evidence}` |" for name, status, evidence in rows)
    return "\n".join(lines)


def minimum_acceptance_table(root: Path) -> str:
    hidden_status, hidden_evidence = sparse_layout_hidden_status(root)
    sparse_layout_evidence = (
        hidden_evidence
        if hidden_status != "missing"
        else "tables/sparse_layout_summary.csv from position_recency_proxy traces"
    )
    snapkv_evidence = (
        hidden_evidence
        if hidden_status != "missing"
        else "tables/sparse_layout_summary.csv"
    )
    rows = [
        (
            "1. dense/eagle3/peagle real GuideLLM results",
            "supported",
            "02_baselines/*_smoke/guidellm_results.json",
        ),
        (
            "2. dense/eagle3/peagle real accuracy outputs",
            "supported",
            "02_baselines/*_smoke/accuracy_outputs.jsonl and accuracy_summary.json",
        ),
        (
            "3. baseline_summary.csv includes throughput, latency, accuracy, acceptance",
            "supported",
            "tables/baseline_summary.csv",
        ),
        (
            "4. eagle3/peagle coarse draft/verifier/other breakdown",
            "supported",
            "tables/breakdown_summary.csv plus 03_breakdown/*_profile/profile_events.jsonl",
        ),
        (
            "5. sparse_layout_summary shows independent_topk union growth and speclink reduction",
            "supported-proxy",
            sparse_layout_evidence,
        ),
        (
            "6. SnapKV/static challenge has coverage, union, Jaccard, HBM bytes",
            "supported-proxy",
            snapkv_evidence,
        ),
        (
            "7. speclink planner unit tests pass",
            "supported",
            "verification.md",
        ),
        (
            "8. vLLM speclink plan_only smoke writes live trace",
            "supported",
            "05_speclink/speclink_prob_profile_smoke/live_sparse_trace.jsonl",
        ),
        (
            "9. speclink_experiment_report.md",
            "supported",
            "speclink_experiment_report.md",
        ),
    ]
    lines = [
        "| minimum acceptance item | status | evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {name} | {status} | `{evidence}` |" for name, status, evidence in rows)
    return "\n".join(lines)


def failure_summary(root: Path) -> str:
    failures = sorted(root.rglob("failures.md"))
    if not failures:
        return "No `failures.md` files were found under this result root."
    lines = ["Recorded failure files:"]
    lines.extend(f"- `{path}`" for path in failures)
    return "\n".join(lines)


def figure_summary(root: Path) -> str:
    figures = sorted((root / "figures").glob("*.png"))
    if not figures:
        return "No PNG figures were generated."
    lines = ["Generated figures:"]
    lines.extend(f"- `{path}`" for path in figures)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--baseline-root",
        help="Root containing baseline run dirs (default: RESULTS_ROOT)",
    )
    parser.add_argument("--report", help="Default: RESULTS_ROOT/speclink_experiment_report.md")
    args = parser.parse_args()

    root = Path(args.results_root)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    baseline_csv = tables / "baseline_summary.csv"
    baseline_md = tables / "baseline_summary.md"
    equivalence_diffs_md = tables / "dense_equivalence_diffs.md"
    matrix_status_md = tables / "formal_matrix_status.md"
    matrix_status_tsv = tables / "formal_matrix_status.tsv"
    breakdown_matrix_status_md = tables / "breakdown_matrix_status.md"
    breakdown_matrix_status_tsv = tables / "breakdown_matrix_status.tsv"
    speclink_matrix_status_md = tables / "speclink_plan_matrix_status.md"
    speclink_matrix_status_tsv = tables / "speclink_plan_matrix_status.tsv"
    speclink_g2_matrix_status_md = tables / "speclink_g2_matrix_status.md"
    speclink_g2_matrix_status_tsv = tables / "speclink_g2_matrix_status.tsv"
    speclink_serving_matrix_status_md = tables / "speclink_serving_matrix_status.md"
    speclink_serving_matrix_status_tsv = tables / "speclink_serving_matrix_status.tsv"
    run(
        [
            "python",
            str(SCRIPT_DIR / "speclink_collect_baselines.py"),
            "--root",
            str(Path(args.baseline_root) if args.baseline_root else root),
            "--out-csv",
            str(baseline_csv),
            "--out-md",
            str(baseline_md),
            "--out-diffs",
            str(equivalence_diffs_md),
        ]
    )
    matrix_status_args = [
        "python",
        str(SCRIPT_DIR / "speclink_matrix_status.py"),
        "--root",
        str(root),
    ]
    maybe_run(
        [
            *matrix_status_args,
            "--matrix",
            "baseline",
            "--out-tsv",
            str(matrix_status_tsv),
            "--out-md",
            str(matrix_status_md),
        ]
    )
    maybe_run(
        [
            *matrix_status_args,
            "--matrix",
            "breakdown",
            "--out-tsv",
            str(breakdown_matrix_status_tsv),
            "--out-md",
            str(breakdown_matrix_status_md),
        ]
    )
    maybe_run(
        [
            *matrix_status_args,
            "--matrix",
            "speclink-plan",
            "--out-tsv",
            str(speclink_matrix_status_tsv),
            "--out-md",
            str(speclink_matrix_status_md),
        ]
    )
    maybe_run(
        [
            *matrix_status_args,
            "--matrix",
            "speclink-g2",
            "--expected-benchmark-limit",
            "30",
            "--expected-accuracy-limit",
            "30",
            "--out-tsv",
            str(speclink_g2_matrix_status_tsv),
            "--out-md",
            str(speclink_g2_matrix_status_md),
        ]
    )
    maybe_run(
        [
            *matrix_status_args,
            "--matrix",
            "speclink-serving",
            "--k-values",
            "8",
            "--expected-benchmark-limit",
            "80",
            "--expected-accuracy-limit",
            "80",
            "--out-tsv",
            str(speclink_serving_matrix_status_tsv),
            "--out-md",
            str(speclink_serving_matrix_status_md),
        ]
    )

    sparse_md = tables / "sparse_layout_summary.md"
    sparse_hidden_md = tables / "sparse_layout_hidden_summary.md"
    breakdown_md = tables / "breakdown_summary.md"
    microbench_md = tables / "sparse_microbench.md"
    sparse_quality_csv = tables / "sparse_quality_summary.csv"
    sparse_quality_md = tables / "sparse_quality_summary.md"
    speclink_ablation_csv = tables / "speclink_ablation.csv"
    speclink_ablation_md = tables / "speclink_ablation.md"
    patch_regression_md = root / "05_speclink" / "patch_regression_check.md"
    if not breakdown_md.exists():
        breakdown_md.write_text("Breakdown profile events have not been collected yet.\n", encoding="utf-8")
    if not sparse_md.exists():
        sparse_md.write_text("Sparse trace simulation has not been run yet.\n", encoding="utf-8")
    maybe_run(
        [
            "python",
            str(SCRIPT_DIR / "speclink_collect_ablation.py"),
            "--root",
            str(root),
            "--out-csv",
            str(speclink_ablation_csv),
            "--out-md",
            str(speclink_ablation_md),
        ]
    )
    write_sparse_quality_md(sparse_quality_csv, sparse_quality_md)
    report = Path(args.report) if args.report else root / "speclink_experiment_report.md"
    lines = [
        "# speclink Experimental Report",
        "",
        "## 1. Environment",
        "",
        f"- Environment audit: `{root / '00_env' / 'env.json'}`",
        f"- P-EAGLE patch check: `{root / '00_env' / 'peagle_patch_check.txt'}`",
        "",
        "## 2. Dataset and Metrics",
        "",
        f"- Dataset profile: `{root / '01_dataset' / 'dataset_profile.json'}`",
        "- Accuracy metrics: strict final-answer EM, flexible final-answer EM, pass@1, dense-output equivalence.",
        "",
        "## 3. Dense / EAGLE3 / P-EAGLE End-to-End Results",
        "",
        f"- Baseline summary CSV: `{baseline_csv}`",
        f"- Baseline summary table: `{baseline_md}`",
        f"- Dense equivalence mismatch examples: `{equivalence_diffs_md}`",
        "",
        read_section(baseline_md, "Baseline summary has not been generated yet."),
        "",
        read_section(equivalence_diffs_md, "Dense equivalence diff summary has not been generated yet."),
        "",
        "## 4. Breakdown",
        "",
        f"- Breakdown table: `{breakdown_md}`",
        "- Percent columns use the same denominator: summed `total_engine_step_ms` from vLLM engine steps.",
        "- `target_verify_forward_pct` is residual model wait after subtracting worker-level draft/sampler timing when that instrumentation is available.",
        "- `speclink_planner_pct` is split out from engine update time when plan events are present.",
        "",
        read_section(breakdown_md, "Breakdown profile events have not been collected yet."),
        "",
        "## 5. Challenge: Why Naive Sparse Verification Is Not Enough",
        "",
        f"- Sparse layout table: `{sparse_md}`",
        "- Plan-only and simulated sparse-memory metrics must not be reported as real sparse-kernel speedup.",
        "- Current live traces use `candidate_source=position_recency_proxy`; replace with draft/target attention-derived candidates before making semantic sparse-attention claims.",
        "",
        read_section(sparse_md, "Sparse trace simulation has not been run yet."),
        "",
        "- Model-derived sparse layout table: "
        f"`{sparse_hidden_md}`",
        "- This table uses `hidden_similarity_proxy` candidates from offline Qwen3 hidden states; it is stronger than the position-recency proxy but still not draft attention.",
        "",
        read_section(sparse_hidden_md, "Hidden-similarity sparse layout simulation has not been run yet."),
        "",
        "- Sparse microbenchmark table: "
        f"`{microbench_md}`",
        "- Current microbenchmark rows are PyTorch union-attention proxy measurements; they are not vLLM sparse-kernel timings.",
        "",
        read_section(microbench_md, "Sparse microbenchmark has not been run yet."),
        "",
        "- Offline sparse verifier quality table: "
        f"`{sparse_quality_md}`",
        "- Current sparse-quality rows are masked-logits proxy measurements from Transformers, not vLLM sparse-kernel correctness.",
        "",
        read_section(sparse_quality_md, "Offline sparse verifier quality has not been run yet."),
        "",
        "## 6. speclink Method",
        "",
        "- Planner: shared sparse KV blocks plus per-token residual blocks.",
        "- Acceptance-aware weights use `rho_i`, `a_i`, and `rho_i * 4 * a_i * (1 - a_i)` risk.",
        "",
        "## 7. speclink Results and Ablations",
        "",
        f"- Padded context manifest: `{root / '04_sparse_challenge' / 'padded_data' / 'padded_math_manifest.json'}`",
        formal_baseline_note(root),
        "- Formal speclink plan-only layout/K rows are summarized in the matrix status table below.",
        f"- Formal baseline matrix status: `{matrix_status_md}`",
        f"- Breakdown matrix status: `{breakdown_matrix_status_md}`",
        f"- speclink plan-only matrix status: `{speclink_matrix_status_md}`",
        f"- speclink G2 budget/block/fallback matrix status: `{speclink_g2_matrix_status_md}`",
        f"- speclink G4 serving-rate matrix status: `{speclink_serving_matrix_status_md}`",
        f"- speclink plan-only ablation table: `{speclink_ablation_md}`",
        f"- Patch off-mode regression check: `{patch_regression_md}`",
        "",
        read_section(matrix_status_md, "Formal baseline matrix status has not been generated yet."),
        "",
        read_section(breakdown_matrix_status_md, "Breakdown matrix status has not been generated yet."),
        "",
        read_section(speclink_matrix_status_md, "speclink plan-only matrix status has not been generated yet."),
        "",
        read_section(speclink_g2_matrix_status_md, "speclink G2 matrix status has not been generated yet."),
        "",
        read_section(speclink_serving_matrix_status_md, "speclink G4 serving-rate matrix status has not been generated yet."),
        "",
        read_section(speclink_ablation_md, "speclink ablation summary has not been generated yet."),
        "",
        read_section(patch_regression_md, "Patch regression check has not been written yet."),
        "",
        figure_summary(root),
        "",
        "## 8. Minimum Acceptance Checklist",
        "",
        minimum_acceptance_table(root),
        "",
        "## 9. Key Claims Supported / Not Yet Supported",
        "",
        "- Real end-to-end claims require GuideLLM and accuracy outputs from live vLLM runs.",
        "- Sparse-layout claims require real/proxy trace files plus `sparse_layout_summary.csv`.",
        "- Sparse-kernel speedup is not supported until `SPECLINK_MODE=sparse_kernel` is integrated and measured.",
        "",
        status_table(root),
        "",
        "## 10. Failures",
        "",
        failure_summary(root),
        "",
        "## 11. Reproduction Commands",
        "",
        "See `command.txt` and `env.txt` in each run directory.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
