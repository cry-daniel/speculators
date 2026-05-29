#!/usr/bin/env python3
"""Run the trace-based SpecLink-CV milestone experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tools.speclink_cv.core import (
    SpecLinkCVConfig,
    choose_chunk_size,
    config_snapshot,
    load_trace_steps,
    write_csv,
    write_json,
)
from tools.speclink_cv.env_check import collect as collect_env
from tools.speclink_cv.env_check import write_markdown as write_env_markdown


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS = [
    "test_chunk_decision",
    "test_state_machine",
    "test_async_queue",
    "test_roofline_packing",
    "test_correctness_smoke",
    "test_vllm_runtime_config",
]


def run_cmd(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        result = subprocess.run(cmd, cwd=cwd, text=True, stdout=out, stderr=err, check=False)
    return result.returncode


def create_tree(root: Path) -> None:
    for name in [
        "00_env",
        "01_impl_notes",
        "02_unit_tests",
        "03_confidence_calibration",
        "04_baselines",
        "05_cv_ablation",
        "06_scheduler_queue",
        "07_roofline_packing",
        "08_figures",
        "09_reports",
        "logs",
        "patches",
        "scripts",
    ]:
        (root / name).mkdir(parents=True, exist_ok=True)


def write_run_command(root: Path, argv: list[str]) -> None:
    script = root / "scripts" / "run_command.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        + " ".join(subprocess.list2cmdline([item]) for item in argv)
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def run_env(root: Path) -> None:
    report = collect_env()
    write_json(root / "00_env" / "env_report.json", report)
    write_env_markdown(report, root / "00_env" / "env_report.md")


def run_tests(root: Path) -> dict[str, Any]:
    rows = []
    for test in DEFAULT_TESTS:
        cmd = [sys.executable, "-m", f"tools.speclink_cv.{test}"]
        out = root / "02_unit_tests" / f"{test}.stdout.log"
        err = root / "02_unit_tests" / f"{test}.stderr.log"
        rc = run_cmd(cmd, REPO_ROOT, out, err)
        rows.append({"test": test, "returncode": rc, "status": "pass" if rc == 0 else "fail"})
    write_csv(root / "02_unit_tests" / "unit_test_summary.csv", rows)
    write_json(root / "02_unit_tests" / "unit_test_summary.json", {"tests": rows})
    lines = ["# Unit Test Summary", ""]
    for row in rows:
        lines.append(f"- {row['test']}: {row['status']} rc={row['returncode']}")
    (root / "02_unit_tests" / "unit_test_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"tests": rows, "passed": all(row["returncode"] == 0 for row in rows)}


def run_auxiliary_tools(root: Path, trace_root: Path, workloads: str) -> dict[str, Any]:
    jobs = [
        (
            "collect_confidence_traces",
            [
                sys.executable,
                "-m",
                "tools.speclink_cv.collect_confidence_traces",
                str(trace_root),
                "--output-dir",
                str(root / "03_confidence_calibration" / "trace_collection"),
                "--workloads",
                workloads,
            ],
        ),
        (
            "calibrate_acceptance",
            [
                sys.executable,
                "-m",
                "tools.speclink_cv.calibrate_acceptance",
                str(trace_root),
                "--output-dir",
                str(root / "03_confidence_calibration" / "fit"),
                "--workloads",
                workloads,
                "--split",
                "even",
            ],
        ),
        (
            "evaluate_calibration",
            [
                sys.executable,
                "-m",
                "tools.speclink_cv.evaluate_calibration",
                str(trace_root),
                "--calibration-model",
                str(root / "03_confidence_calibration" / "fit" / "calibration_model.json"),
                "--output-dir",
                str(root / "03_confidence_calibration" / "eval"),
                "--workloads",
                workloads,
                "--split",
                "odd",
            ],
        ),
        (
            "profile_verify_cost",
            [
                sys.executable,
                "-m",
                "tools.speclink_cv.profile_verify_cost",
                str(trace_root),
                "--output-dir",
                str(root / "07_roofline_packing" / "verify_cost_proxy"),
                "--workloads",
                workloads,
            ],
        ),
    ]
    rows = []
    for name, cmd in jobs:
        stdout = root / "logs" / f"{name}.stdout.log"
        stderr = root / "logs" / f"{name}.stderr.log"
        rc = run_cmd(cmd, REPO_ROOT, stdout, stderr)
        rows.append({"tool": name, "returncode": rc, "status": "pass" if rc == 0 else "fail"})
        if rc != 0:
            break
    write_csv(root / "logs" / "auxiliary_tool_summary.csv", rows)
    write_json(root / "logs" / "auxiliary_tool_summary.json", {"tools": rows})
    return {"tools": rows, "passed": all(row["returncode"] == 0 for row in rows)}


def method_config(method_name: str, k: int) -> SpecLinkCVConfig:
    confidence = "_conf_" in method_name
    async_queue = "_async_" in method_name
    roofline = method_name.endswith("_roofline")
    return SpecLinkCVConfig(
        enable=True,
        confidence_sizing=confidence,
        async_queue=async_queue,
        roofline_packing=roofline,
        candidate_chunks=(1, 2, 4, 6, 8, "full"),
        min_benefit=0.0,
        max_verify_tokens_per_step=8192,
        max_verify_seqs_per_step=32,
    )


def simulate_cv(root: Path, trace_root: Path, workloads: set[str]) -> list[dict[str, Any]]:
    steps = load_trace_steps(trace_root, workloads)
    methods = [
        "cv_half_sync_simple",
        "cv_half_sync_roofline",
        "cv_half_async_simple",
        "cv_half_async_roofline",
        "cv_conf_sync_simple",
        "cv_conf_sync_roofline",
        "cv_conf_async_simple",
        "cv_conf_async_roofline",
    ]
    grouped: dict[tuple[Any, ...], list[Any]] = {}
    for step in steps:
        grouped.setdefault((step.workload, step.model_label, step.method, step.k), []).append(step)

    rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        workload, model_label, source_method, k = key
        for method_name in methods:
            cfg = method_config(method_name, int(k))
            selected_sum = 0
            skipped_sum = 0
            skip_possible_sum = 0
            extra_forwards = 0
            prefix_rejected = 0
            prefix_accepted = 0
            full_count = 0
            for step in group:
                decision = choose_chunk_size(
                    k=step.k,
                    a_hat=list(step.draft_selected_prob),
                    config=cfg,
                    suffix_cost_per_token=1.0,
                    extra_forward_cost=0.25,
                    underutilization_cost=0.0 if cfg.roofline_packing else 0.1,
                )
                h = decision.selected_h
                selected_sum += h
                full_count += int(h == step.k)
                if step.actual_accept_tokens < h:
                    skipped = max(step.k - h, 0)
                    prefix_rejected += 1
                else:
                    skipped = 0
                    prefix_accepted += 1
                    if h < step.k:
                        extra_forwards += 1
                skipped_sum += skipped
                skip_possible_sum += max(step.k - h, 0)
                decision_rows.append(
                    {
                        "workload": workload,
                        "model_label": model_label,
                        "source_method": source_method,
                        "num_spec_tokens": k,
                        "method": method_name,
                        "request_id": step.request_id,
                        "step_id": step.step_id,
                        "actual_accept_tokens": step.actual_accept_tokens,
                        "selected_h": h,
                        "reason": decision.reason,
                        "skipped_suffix_tokens": skipped,
                    }
                )
            n = len(group)
            rows.append(
                {
                    "measurement_type": "trace_exact_simulation_not_end_to_end",
                    "model": model_label,
                    "source_speculator": source_method,
                    "dataset": workload,
                    "K": k,
                    "batch_size": "trace",
                    "method": method_name,
                    "throughput": "",
                    "speedup_vs_eagle3": "",
                    "ttft_p95": "",
                    "itl_p95": "",
                    "e2e_p95": "",
                    "exact_match_vs_eagle3": 1.0,
                    "selected_h_avg": selected_sum / n,
                    "skipped_suffix_ratio": skipped_sum / max(sum(step.k for step in group), 1),
                    "extra_tlm_forwards_per_request": extra_forwards / n,
                    "queue_wait_p95": 0.0 if cfg.async_queue else "",
                    "gpu_util": "",
                    "fallback_ratio": full_count / n,
                    "prefix_rejected_ratio": prefix_rejected / n,
                    "prefix_accepted_ratio": prefix_accepted / n,
                    "skip_possible_ratio": skipped_sum / max(skip_possible_sum, 1),
                    "num_decode_steps": n,
                }
            )
            write_json(
                root / "05_cv_ablation" / f"{method_name}_config.json",
                config_snapshot(cfg),
            )
    write_csv(root / "05_cv_ablation" / "cv_trace_decisions.csv", decision_rows)
    write_csv(root / "05_cv_ablation" / "cv_ablation_summary.csv", rows)
    return rows


def write_baseline_notes(root: Path) -> None:
    rows = [
        {
            "baseline": "eagle3_one_shot",
            "status": "source_trace_only",
            "meaning": "Existing trace labels are produced by one-shot EAGLE3 verification.",
        },
        {
            "baseline": "greedy_non_spec",
            "status": "not_run",
            "meaning": "Requires live vLLM/GuideLLM execution and is intentionally not inferred from traces.",
        },
    ]
    write_csv(root / "04_baselines" / "baseline_status.csv", rows)
    (root / "04_baselines" / "baseline_notes.md").write_text(
        "# Baseline Notes\n\n"
        "- This milestone reuses existing EAGLE3 one-shot verifier traces as the correctness baseline.\n"
        "- It does not run new GuideLLM baselines, because the target verifier token schedule is not changed yet.\n"
        "- End-to-end latency and throughput should be measured only after the vLLM scheduler/model-runner path performs real prefix verification.\n",
        encoding="utf-8",
    )


def write_queue_notes(root: Path) -> None:
    write_json(
        root / "06_scheduler_queue" / "queue_config.json",
        {
            "queue": "VerificationQueue",
            "priority": "shorter prefix chunks first, with age-based promotion",
            "max_wait_steps": 4,
            "status": "unit_tested_policy_live_scheduler_has_conservative_async_prefix_queue",
        },
    )
    (root / "06_scheduler_queue" / "queue_design.md").write_text(
        "# Scheduler Queue Design\n\n"
        "The reusable queue policy prioritizes smaller prefix chunks to reduce wasted suffix "
        "verification, then promotes older items to bound starvation. The live vLLM scheduler "
        "contains a conservative async prefix queue with priority dispatch, age timeout, "
        "queue/dequeue/profile events, and exact prefix verification. The current live queue "
        "is not yet a full cross-request roofline optimizer; it is the first scheduler-integrated "
        "version used by the GuideLLM matrix runner.\n",
        encoding="utf-8",
    )


def draw_skipped_suffix_figure(root: Path, summary_rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for row in summary_rows:
        method = str(row["method"])
        grouped.setdefault(method, {"skip": 0.0, "extra": 0.0, "h": 0.0})
        grouped[method]["skip"] += float(row["skipped_suffix_ratio"])
        grouped[method]["extra"] += float(row["extra_tlm_forwards_per_request"])
        grouped[method]["h"] += float(row["selected_h_avg"])
        counts[method] = counts.get(method, 0) + 1
    points = []
    for method, values in sorted(grouped.items()):
        count = counts[method]
        points.append(
            {
                "method": method,
                "skipped_suffix_ratio": values["skip"] / count,
                "extra_tlm_forwards_per_decode_step": values["extra"] / count,
                "selected_h_avg": values["h"] / count,
            }
        )
    write_csv(root / "08_figures" / "cv_trace_tradeoff_points.csv", points)

    width, height = 1120, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    px0, py0, px1, py1 = 90, 80, 1010, 590
    draw.text((90, 34), "SpecLink-CV trace tradeoff: skipped verifier suffix vs extra TLM forwards", fill="#111", font=title_font)
    draw.rectangle((px0, py0, px1, py1), outline="#222")
    max_x = max([point["extra_tlm_forwards_per_decode_step"] for point in points] + [0.1])
    max_y = max([point["skipped_suffix_ratio"] for point in points] + [0.1])
    max_x = max_x * 1.15
    max_y = max_y * 1.15
    for tick in range(6):
        frac = tick / 5
        x = px0 + frac * (px1 - px0)
        y = py1 - frac * (py1 - py0)
        draw.line((x, py0, x, py1), fill="#eeeeee")
        draw.line((px0, y, px1, y), fill="#eeeeee")
        draw.text((x - 18, py1 + 10), f"{frac * max_x:.2f}", fill="#555", font=font)
        draw.text((px0 - 58, y - 6), f"{frac * max_y:.2f}", fill="#555", font=font)
    draw.text((px0 + 310, py1 + 44), "extra TLM forwards per decode step", fill="#111", font=font)
    draw.text((12, py0 - 28), "skipped suffix ratio", fill="#111", font=font)

    colors = ["#0b7285", "#5f3dc4", "#2b8a3e", "#c92a2a", "#e67700", "#364fc7", "#087f5b", "#862e9c"]
    for idx, point in enumerate(points):
        x = px0 + point["extra_tlm_forwards_per_decode_step"] / max_x * (px1 - px0)
        y = py1 - point["skipped_suffix_ratio"] / max_y * (py1 - py0)
        color = colors[idx % len(colors)]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        label = point["method"].replace("cv_", "").replace("_", " ")
        label_x = min(x + 10, px1 - 230)
        label_y = max(py0 + 8, min(y - 8, py1 - 22))
        draw.text((label_x, label_y), label, fill=color, font=font)
    image.save(root / "08_figures" / "cv_trace_tradeoff.png")


def write_impl_notes(root: Path) -> None:
    (root / "01_impl_notes" / "implementation_notes.md").write_text(
        "# SpecLink-CV Implementation Notes\n\n"
        "- This milestone implements reusable decision, state-machine, queue, "
        "packing, calibration, and trace-experiment tooling.\n"
        "- The current experiment is trace-based exact simulation over existing "
        "one-shot EAGLE3 verifier labels. It proves decision logic and estimates "
        "skippable suffix work, but it is not an end-to-end vLLM speedup claim.\n"
        "- The repo also contains a gated live vLLM slice for non-async EAGLE3 "
        "prefix/suffix verification. It supports fixed-half, uncalibrated "
        "draft-confidence prefix sizing, and calibrated binning via "
        "SPECLINK_CV_CALIBRATION_PATH. It also has a conservative live async "
        "prefix queue with priority/age dispatch, a lightweight roofline-aware "
        "fallback that avoids launching underfilled prefix chunks by using "
        "one-shot verification, plus SPECLINK_CV_PROFILE_JSONL scheduler/chunk "
        "profile logging. Multi-request roofline packing is still limited; "
        "the current live queue is a first scheduler-integrated version.\n",
        encoding="utf-8",
    )


def write_patches(root: Path) -> None:
    diff = subprocess.run(
        ["git", "diff", "--", "vllm"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (root / "patches" / "vllm_speclink_cv.diff").write_text(diff.stdout, encoding="utf-8")


def write_report(root: Path, summary_rows: list[dict[str, Any]], tests: dict[str, Any]) -> None:
    best = sorted(
        summary_rows,
        key=lambda row: (
            -float(row["skipped_suffix_ratio"]),
            float(row["extra_tlm_forwards_per_request"]),
            row["method"],
        ),
    )[:10]
    lines = [
        "# SPECLINK_CV_REPORT",
        "",
        "This run is primarily a trace-based SpecLink-CV milestone, not a full serving benchmark.",
        "It uses existing EAGLE3 one-shot verification traces to evaluate exact prefix-gated chunk decisions.",
        "",
        "## Status",
        "",
        f"- unit tests passed: {tests['passed']}",
        "- live vLLM chunked verification patch: implemented for non-async EAGLE3 fixed-half, uncalibrated confidence, and calibrated-binning confidence prefix sizing",
        "- live profile JSONL: implemented for schedule_step, verify_chunk_scheduled, verify_chunk_result, and roofline fallback decision events",
        "- live async queue: conservative prefix queue implemented with priority dispatch, age timeout, queue/dequeue/profile events, and exact prefix verification",
        "- live roofline packing: lightweight utilization gate implemented as one-shot fallback in sync mode and queue utilization dispatch in async mode; full multi-request roofline cost-model packing remains incomplete",
        "- correctness claim in this report: trace-level exact equivalence only; run `tools/speclink_cv/live_correctness_smoke.py` for GPU token-match smoke",
        "- throughput/latency claim: not reported, because this trace run does not execute a serving benchmark",
        "",
        "## Best Trace-Simulation Rows",
        "",
    ]
    for row in best:
        lines.append(
            "- "
            f"{row['dataset']} {row['model']} {row['source_speculator']} K={row['K']} {row['method']}: "
            f"skipped_suffix_ratio={float(row['skipped_suffix_ratio']):.4f}, "
            f"extra_tlm_forwards_per_request={float(row['extra_tlm_forwards_per_request']):.4f}, "
            f"selected_h_avg={float(row['selected_h_avg']):.2f}"
        )
    lines.extend(
        [
            "",
            "## Required Next Work",
            "",
            "- Resolve batched live correctness mismatches before using batched `cv_*` rows as speedup claims.",
            "- Expand the conservative live async queue into a richer cross-request packing policy.",
            "- Replace the current live roofline heuristics with a measured queue-level cost model.",
            "- Run the full GuideLLM matrix after the batched correctness gate is satisfied; until then only rows with `exact_match_vs_eagle3 == 1.0` are valid performance evidence.",
        ]
    )
    (root / "09_reports" / "SPECLINK_CV_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_csv(root / "09_reports" / "summary_metrics.csv", summary_rows)
    write_json(root / "09_reports" / "summary_metrics.json", {"rows": summary_rows})
    write_csv(root / "summary_metrics.csv", summary_rows)
    write_json(root / "summary_metrics.json", {"rows": summary_rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workloads", default="math,mtbench")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        REPO_ROOT
        / "examples"
        / "evaluate"
        / "eval-guidellm"
        / "results"
        / f"speclink_cv_{timestamp}"
    )
    create_tree(output_root)
    write_run_command(output_root, sys.argv)
    run_env(output_root)
    write_impl_notes(output_root)
    write_patches(output_root)
    tests = run_tests(output_root)
    auxiliary = run_auxiliary_tools(output_root, args.trace_root, args.workloads)
    summary_rows = simulate_cv(
        output_root,
        args.trace_root,
        {item.strip() for item in args.workloads.split(",") if item.strip()},
    )
    write_baseline_notes(output_root)
    write_queue_notes(output_root)
    draw_skipped_suffix_figure(output_root, summary_rows)
    write_report(output_root, summary_rows, tests)
    write_json(
        output_root / "logs" / "run_status.json",
        {"unit_tests": tests, "auxiliary_tools": auxiliary},
    )
    print(f"[INFO] Wrote SpecLink-CV trace experiment: {output_root}")


if __name__ == "__main__":
    main()
