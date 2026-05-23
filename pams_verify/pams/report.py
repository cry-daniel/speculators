from __future__ import annotations

import argparse
import json
from pathlib import Path

from .configs import EXPERIMENTS, REPORTS, ROOT


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summary_for(exp_name: str) -> str:
    path = EXPERIMENTS / exp_name / "summary.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else f"# {exp_name}\n\nNo summary found."


def decide() -> tuple[str, list[str]]:
    end2end = load(EXPERIMENTS / "10_end2end" / "parsed" / "end2end_matrix.json")
    correctness = load(EXPERIMENTS / "11_correctness_quality" / "parsed" / "correctness_metrics.json")
    hardware = load(EXPERIMENTS / "00_env" / "hardware.json")
    memory = load(EXPERIMENTS / "00_env" / "memory_estimate.json")
    mask = load(EXPERIMENTS / "05_mask_planner_offline" / "parsed" / "mask_planner_metrics.json")
    reasons = []
    if not end2end.get("pams_end2end_available", False):
        reasons.append("No real patched-vLLM PAMS end-to-end result is available.")
    reasons.append("Integration B did not compile or run a sparse verifier attention patch.")
    approx_false_accept = correctness.get("approximate_sparse", {}).get("false_accept_rate")
    if approx_false_accept is not None and approx_false_accept >= 0.001:
        reasons.append(f"Approximate sparse false accept rate in the offline audit is {approx_false_accept:.4f}, above the 0.1% GO threshold.")
    if mask:
        reasons.append("Offline and microbenchmark evidence cannot satisfy GO without live vLLM speedup.")
    return "NO-GO", reasons[:3]


def write_report(root: Path, output: Path) -> tuple[str, list[str]]:
    decision, reasons = decide()
    union = load(EXPERIMENTS / "03_union_problem" / "parsed" / "union_metrics.json").get("results", [])
    acceptance = load(EXPERIMENTS / "04_acceptance_prior" / "parsed" / "acceptance_prior_metrics.json")
    mask = load(EXPERIMENTS / "05_mask_planner_offline" / "parsed" / "mask_planner_metrics.json").get("results", [])
    kernel = load(EXPERIMENTS / "06_sparse_kernel_microbench" / "parsed" / "kernel_summary.json")
    integration_b = load(EXPERIMENTS / "08_vllm_integration_b_attention_patch" / "parsed" / "integration_b_result.json")
    end2end = load(EXPERIMENTS / "10_end2end" / "parsed" / "end2end_matrix.json")
    correctness = load(EXPERIMENTS / "11_correctness_quality" / "parsed" / "correctness_metrics.json")
    hardware = load(EXPERIMENTS / "00_env" / "hardware.json")
    memory = load(EXPERIMENTS / "00_env" / "memory_estimate.json")
    live_baselines = load(EXPERIMENTS / "01_dense_baselines" / "parsed" / "live_vllm_baseline_smoke.json").get("runs", [])

    strongest = "Offline PAMS mask planning improves the synthetic accepted-token-per-loaded-block proxy." if mask else "No positive PAMS result."
    weakest = "No live patched-vLLM sparse verifier path compiled or ran."
    lines = [
        "# PAMS-Verify Final Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Decision: **{decision}**",
        "",
        f"Strongest result: {strongest}",
        f"Weakest result: {weakest}",
        "End-to-end vLLM speedup achieved: no.",
        "",
        "## 2. Hardware and Model Setup",
        "",
        f"GPU: `{hardware.get('gpu_name', 'unknown')}`",
        f"VRAM GB: `{hardware.get('total_vram_gb', 'unknown')}`",
        f"Torch CUDA available in escalated preflight: `{hardware.get('cuda_available', 'unknown')}`",
        f"vLLM version: `{hardware.get('vllm_version', 'unknown')}`",
        f"Recommended max_model_len: `{memory.get('recommended_max_model_len', 'unknown')}`",
        "",
        summary_for("00_env"),
        "",
        "## 3. Correctness Policy",
        "",
        "Exact modes must compare token IDs or dense verifier decisions. Approximate sparse modes report false accept and false reject separately. The current exactness evidence is offline synthetic only.",
        "",
        f"Correctness metrics: `{json.dumps(correctness, sort_keys=True)[:1200]}`",
        "",
        "## 4. Experiment 1: Union Problem",
        "",
        f"Union metrics: `{json.dumps(union[:4], sort_keys=True)[:1200]}`",
        "",
        "## 5. Experiment 2: Acceptance Prior",
        "",
        f"Acceptance metrics: `{json.dumps(acceptance, sort_keys=True)[:1200]}`",
        "",
        "## 6. Experiment 3: Offline PAMS Mask Planning",
        "",
        f"Offline mask metrics: `{json.dumps(mask[:4], sort_keys=True)[:1200]}`",
        "",
        "## 7. Experiment 4: Sparse Kernel Microbenchmark",
        "",
        f"Kernel metrics: `{json.dumps(kernel, sort_keys=True)[:1200]}`",
        "",
        "## 8. Experiment 5: vLLM Integration Attempts",
        "",
        "Integration A registered an exact scheduler policy offline but did not apply a live vLLM hook.",
        "Integration B inspected vLLM and wrote a proposed diff, but arbitrary verifier block masks were unsupported and no patch was applied.",
        "Integration C evaluated fallback policies offline only.",
        "",
        f"Integration B evidence: `{json.dumps(integration_b, sort_keys=True)[:1200]}`",
        "",
        "## 9. End-to-End Results",
        "",
        "Live standard-baseline smoke results were collected for Qwen3-8B at `max_model_len=2048`, `max_num_seqs=4`, four random short prompts, and concurrency 1. These are smoke measurements, not the full matrix.",
        "",
        f"Live baseline smoke: `{json.dumps(live_baselines, sort_keys=True)[:1600]}`",
        "",
        f"End-to-end matrix: `{json.dumps(end2end, sort_keys=True)[:1200]}`",
        "",
        "No PAMS end-to-end throughput or ITL claim is made.",
        "",
        "## 10. Ablations",
        "",
        summary_for("12_ablations"),
        "",
        "## 11. Failure Log",
        "",
        (REPORTS / "failure_log.md").read_text(encoding="utf-8") if (REPORTS / "failure_log.md").exists() else "Failure log not generated.",
        "",
        "## 12. Final Judgment",
        "",
        "This is currently a NO-GO for a systems paper claim. The cleanest claim is an offline research prototype showing the union-growth motivation and a concrete implementation plan for shared+residual mask planning. The next engineering step is an editable vLLM source checkout with a minimal exact scheduler hook first, then an attention backend prototype that can consume verifier block masks.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    go_text = [
        "# GO / NO-GO",
        "",
        f"Decision: {decision}",
        "",
        "Top reasons:",
    ] + [f"{idx}. {reason}" for idx, reason in enumerate(reasons, start=1)]
    (ROOT / "GO_NO_GO.md").write_text("\n".join(go_text) + "\n", encoding="utf-8")
    (REPORTS / "go_no_go.md").write_text("\n".join(go_text) + "\n", encoding="utf-8")
    return decision, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=REPORTS / "final_report.md")
    args = parser.parse_args()
    decision, reasons = write_report(args.root, args.output)
    while len(reasons) < 3:
        reasons.append("No additional evidence available.")
    print(f"PAMS_VERIFY_FINAL_DECISION: {decision}")
    print("Top 3 reasons:")
    for idx, reason in enumerate(reasons[:3], start=1):
        print(f"{idx}. {reason}")
    print("Strongest end-to-end figure: experiments/10_end2end/figures/end2end_throughput.png (unavailable placeholder)")
    print("Strongest correctness figure: experiments/11_correctness_quality/figures/false_accept_false_reject.png")
    print("Strongest ablation: experiments/12_ablations/figures/ablation_summary.png")
    print("Main failure if NO-GO: no live patched-vLLM sparse verifier end-to-end result")
    print("Recommended next engineering step: create an editable vLLM checkout and implement Integration A as a minimal exact hook before attempting arbitrary attention masks")


if __name__ == "__main__":
    main()
