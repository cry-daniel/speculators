#!/usr/bin/env python3
"""Explain why SpecLink-CV verifier savings did or did not become speedup.

The script consumes existing GuideLLM matrix artifacts. It does not start vLLM
and it does not read unbounded debug dumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.speclink_cv.core import write_csv, write_json


SPEC_RE = re.compile(
    r"Mean acceptance length: (?P<mean>[0-9.]+), "
    r"Accepted throughput: (?P<accepted_tps>[0-9.]+) tokens/s, "
    r"Drafted throughput: (?P<drafted_tps>[0-9.]+) tokens/s, "
    r"Accepted: (?P<accepted_tokens>[0-9]+) tokens, "
    r"Drafted: (?P<drafted_tokens>[0-9]+) tokens, "
    r"Per-position acceptance rate: (?P<rates>[0-9., ]+), "
    r"Avg Draft acceptance rate: (?P<draft_accept>[0-9.]+)%"
)
GEN_RE = re.compile(
    r"Avg generation throughput: (?P<gen_tps>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs"
)


def safe_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def trimmed(values: list[float]) -> list[float]:
    if len(values) > 4:
        return values[1:-1]
    return values


def mean(values: list[float]) -> float | str:
    values = trimmed(values)
    return statistics.mean(values) if values else ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summary_path(root: Path) -> Path:
    candidates = [
        root / "09_reports" / "summary_metrics.csv",
        root / "summary_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _metric_mean(metrics: dict[str, Any], name: str) -> str:
    value = (
        metrics.get(name, {})
        .get("successful", {})
        .get("mean", "")
    )
    return str(value) if value != "" else ""


def parse_guidellm_metrics(run_dir: Path) -> dict[str, str]:
    path = run_dir / "guidellm_results.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        benchmark = (data.get("benchmarks") or [{}])[0]
        metrics = benchmark.get("metrics") or {}
    except Exception:
        return {}
    return {
        "throughput": _metric_mean(metrics, "output_tokens_per_second"),
        "total_tokens_per_second": _metric_mean(metrics, "tokens_per_second"),
        "requests_per_second": _metric_mean(metrics, "requests_per_second"),
        "actual_average_batch_size": _metric_mean(
            metrics, "actual_average_batch_size"
        ),
        "output_tokens_per_iteration": _metric_mean(
            metrics, "output_tokens_per_iteration"
        ),
    }


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def parse_guidellm_window(run_dir: Path) -> tuple[float | None, float | None]:
    path = run_dir / "guidellm_results.json"
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    benchmarks = data.get("benchmarks") or []
    if not benchmarks:
        return None, None
    bench = benchmarks[0]
    start = safe_float(bench.get("start_time"))
    end = safe_float(bench.get("end_time"))
    return start, end


def gpu_util_metrics(values: list[float], prefix: str) -> dict[str, str]:
    if not values:
        return {f"{prefix}sample_count": "0"}
    return {
        f"{prefix}util": str(statistics.mean(values)),
        f"{prefix}util_p50": str(percentile(values, 0.50)),
        f"{prefix}util_p95": str(percentile(values, 0.95)),
        f"{prefix}busy_ratio_50": str(
            sum(1 for item in values if item >= 50.0) / len(values)
        ),
        f"{prefix}busy_ratio_80": str(
            sum(1 for item in values if item >= 80.0) / len(values)
        ),
        f"{prefix}sample_count": str(len(values)),
    }


def parse_gpu_util_metrics(run_dir: Path) -> dict[str, str]:
    path = run_dir / "gpu_util.csv"
    if not path.exists():
        return {}
    values: list[float] = []
    active_values: list[float] = []
    powers: list[float] = []
    active_powers: list[float] = []
    active_start, active_end = parse_guidellm_window(run_dir)
    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith("timestamp"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            sample_time: float | None = None
            try:
                sample_time = datetime.strptime(
                    parts[0], "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp()
            except ValueError:
                pass
            active = (
                sample_time is not None
                and active_start is not None
                and active_end is not None
                and active_start <= sample_time <= active_end
            )
            util: float | None = None
            power: float | None = None
            try:
                util = float(parts[1])
                values.append(util)
            except ValueError:
                pass
            try:
                power = float(parts[2])
                powers.append(power)
            except ValueError:
                pass
            if active:
                if util is not None:
                    active_values.append(util)
                if power is not None:
                    active_powers.append(power)
    if not values:
        return {"gpu_sample_count": "0"}
    metrics = {
        **gpu_util_metrics(values, "gpu_"),
        "gpu_power_avg": str(statistics.mean(powers)) if powers else "",
    }
    if active_start is not None and active_end is not None:
        metrics.update(gpu_util_metrics(active_values, "gpu_active_"))
        metrics["gpu_active_power_avg"] = (
            str(statistics.mean(active_powers)) if active_powers else ""
        )
    return metrics


def discover_run_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return rows
    for config_path in sorted(runs_dir.glob("*/config.json")):
        run_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row: dict[str, str] = {
            "measurement_type": "guidellm_end_to_end",
            "model": str(config.get("model_label", "")),
            "dataset": str(config.get("dataset_label", "")),
            "K": str(config.get("K", "")),
            "batch_size": str(config.get("batch_size", "")),
            "method": str(config.get("method", "")),
            "output_dir": str(run_dir),
            "status": "ok" if (run_dir / "guidellm_results.json").exists() else "",
        }
        row.update(parse_guidellm_metrics(run_dir))
        row.update(parse_gpu_util_metrics(run_dir))
        rows.append(row)
    return rows


def load_summary_rows(root: Path) -> list[dict[str, str]]:
    rows = read_csv(summary_path(root))
    discovered = discover_run_rows(root)
    by_scenario = {
        (
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("K", ""),
            row.get("batch_size", ""),
            row.get("method", ""),
        ): row
        for row in rows
    }
    for row in discovered:
        key = (
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("K", ""),
            row.get("batch_size", ""),
            row.get("method", ""),
        )
        existing = by_scenario.get(key)
        if existing is None:
            rows.append(row)
            by_scenario[key] = row
        else:
            for field, value in row.items():
                if value != "" and not existing.get(field):
                    existing[field] = value
    return rows


def parse_vllm_spec_metrics(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "vllm_server.log"
    if not log_path.exists():
        return {}
    gen_tps: list[float] = []
    mean_accept: list[float] = []
    accepted_tps: list[float] = []
    drafted_tps: list[float] = []
    draft_accept_rate: list[float] = []
    per_position: list[list[float]] = []
    accepted_tokens_last = ""
    drafted_tokens_last = ""
    enforce_eager = False
    cudagraph_none = False
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Enforce eager set" in line:
            enforce_eager = True
        if "cudagraph_mode=<CUDAGraphMode.NONE" in line:
            cudagraph_none = True
        gen_match = GEN_RE.search(line)
        if gen_match and int(gen_match.group("running")) > 0:
            gen_tps.append(float(gen_match.group("gen_tps")))
        spec_match = SPEC_RE.search(line)
        if spec_match:
            mean_accept.append(float(spec_match.group("mean")))
            accepted_tps.append(float(spec_match.group("accepted_tps")))
            drafted_tps.append(float(spec_match.group("drafted_tps")))
            accepted_tokens_last = spec_match.group("accepted_tokens")
            drafted_tokens_last = spec_match.group("drafted_tokens")
            draft_accept_rate.append(float(spec_match.group("draft_accept")))
            per_position.append(
                [float(item.strip()) for item in spec_match.group("rates").split(",")]
            )
    metrics: dict[str, Any] = {
        "vllm_generation_tps_mid_avg": mean(gen_tps),
        "spec_mean_acceptance_len_mid_avg": mean(mean_accept),
        "spec_accepted_tps_mid_avg": mean(accepted_tps),
        "spec_drafted_tps_mid_avg": mean(drafted_tps),
        "spec_draft_accept_rate_mid_avg": mean(draft_accept_rate),
        "spec_accepted_tokens_last": accepted_tokens_last,
        "spec_drafted_tokens_last": drafted_tokens_last,
        "vllm_enforce_eager": int(enforce_eager),
        "vllm_cudagraph_none": int(cudagraph_none),
    }
    usable_positions = trimmed(per_position)
    if usable_positions:
        width = max(len(row) for row in usable_positions)
        for idx in range(width):
            values = [row[idx] for row in usable_positions if idx < len(row)]
            metrics[f"spec_accept_pos{idx + 1}_mid_avg"] = (
                statistics.mean(values) if values else ""
            )
    return metrics


def parse_cv_prefix_profile(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "speclink_cv_profile.jsonl"
    if not path.exists():
        return {}
    prefix_results: list[dict[str, Any]] = []
    suffix_results: list[dict[str, Any]] = []
    prefix_scheduled: list[dict[str, Any]] = []
    suffix_scheduled: list[dict[str, Any]] = []
    async_steps: list[dict[str, Any]] = []
    forward_plans: list[dict[str, Any]] = []
    staged_suffix_registered: list[dict[str, Any]] = []
    log_limit_reached = False
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "jsonl_limit_reached":
                log_limit_reached = True
            if row.get("event") == "async_queue_step":
                async_steps.append(row)
            if row.get("event") == "model_forward_plan":
                forward_plans.append(row)
            if row.get("event") == "staged_suffix_draft_registered":
                staged_suffix_registered.append(row)
            if row.get("event") == "verify_chunk_scheduled":
                if row.get("phase") == "prefix":
                    prefix_scheduled.append(row)
                elif row.get("phase") == "suffix":
                    suffix_scheduled.append(row)
            if row.get("event") == "verify_chunk_result":
                if row.get("phase") == "prefix":
                    prefix_results.append(row)
                elif row.get("phase") == "suffix":
                    suffix_results.append(row)
    if not prefix_results and not prefix_scheduled and not async_steps:
        return {"cv_prefix_profile_log_limit_reached": int(log_limit_reached)}
    selected_h_values = [
        float(row.get("chunk_len", 0) or 0)
        for row in prefix_scheduled
        if row.get("chunk_len") not in ("", None)
    ]
    prefix_accepted_values = [
        float(row.get("num_accepted", 0) or 0)
        for row in prefix_results
        if row.get("num_accepted") not in ("", None)
    ]
    suffix_candidate_tokens = sum(
        int(float(row.get("suffix_len", 0) or 0)) for row in prefix_results
    )
    skipped_suffix_tokens = sum(
        int(float(row.get("skipped_suffix_tokens", 0) or 0))
        for row in prefix_results
    )
    extra_tlm_forward = sum(
        int(float(row.get("extra_tlm_forward", 0) or 0))
        for row in prefix_results
    )
    baseline_target_tokens = sum(
        int(float(row.get("k", 0) or 0)) + 1
        for row in prefix_scheduled
        if row.get("k") not in ("", None)
    )
    draft_tokens_generated_est = sum(
        int(
            float(
                row.get("chunk_len", 0)
                if row.get("staged_drafting")
                else row.get("k", 0)
            )
            or 0
        )
        for row in prefix_scheduled
    ) + sum(
        int(float(row.get("registered_suffix_len", row.get("new_draft_len", 0)) or 0))
        for row in staged_suffix_registered
    )
    draft_tokens_full_k_est = sum(
        int(float(row.get("k", 0) or 0))
        for row in prefix_scheduled
        if row.get("k") not in ("", None)
    )
    prefix_tokens_accepted = sum(
        int(float(row.get("num_accepted", 0) or 0))
        for row in prefix_results
    )
    suffix_tokens_accepted = sum(
        int(float(row.get("num_accepted", 0) or 0))
        for row in suffix_results
    )
    draft_tokens_accepted_est = prefix_tokens_accepted + suffix_tokens_accepted
    draft_tokens_discarded_est = max(
        0, draft_tokens_generated_est - draft_tokens_accepted_est
    )
    prefix_target_tokens = sum(
        int(float(row.get("chunk_len", 0) or 0)) + 1
        for row in prefix_scheduled
    )
    suffix_target_tokens = sum(
        int(
            float(
                row.get("effective_verify_tokens", row.get("chunk_len", 0))
                or 0
            )
        )
        + 1
        for row in suffix_scheduled
    )
    dispatched = [
        row for row in async_steps if int(float(row.get("dispatch_count", 0) or 0)) > 0
    ]
    dispatch_count_values = [
        float(row.get("dispatch_count", 0) or 0) for row in dispatched
    ]
    dispatch_util_values = [
        float(
            row.get("predicted_utilization")
            or max(
                float(row.get("token_budget_utilization", 0) or 0),
                float(row.get("seq_budget_utilization", 0) or 0),
            )
        )
        for row in dispatched
    ]
    util_threshold_values = [
        float(row.get("util_threshold", 0.6) or 0.6) for row in dispatched
    ]
    underfilled = [
        util < threshold
        for util, threshold in zip(dispatch_util_values, util_threshold_values)
    ]
    forward_mode_hist = Counter(
        f"{row.get('phase')}:{row.get('cudagraph_mode')}"
        for row in forward_plans
    )
    hist = Counter(int(float(row.get("num_accepted", 0))) for row in prefix_results)
    result_total = len(prefix_results)
    total = result_total or 1
    full_accept = sum(
        1
        for row in prefix_results
        if int(float(row.get("num_accepted", 0)))
        >= int(float(row.get("chunk_len", 0)))
    )
    zero_accept = hist.get(0, 0)
    reject = sum(1 for row in prefix_results if row.get("result") == "rejected_skip_suffix")
    return {
        "cv_prefix_scheduled_count_profile": len(prefix_scheduled),
        "cv_suffix_scheduled_count_profile": len(suffix_scheduled),
        "cv_selected_h_avg_profile": (
            statistics.mean(selected_h_values) if selected_h_values else ""
        ),
        "cv_prefix_accepted_tokens_avg_profile": (
            statistics.mean(prefix_accepted_values)
            if prefix_accepted_values
            else ""
        ),
        "cv_skipped_suffix_ratio_profile": (
            skipped_suffix_tokens / suffix_candidate_tokens
            if suffix_candidate_tokens
            else ""
        ),
        "cv_verify_target_token_ratio_vs_oneshot_est_profile": (
            (prefix_target_tokens + suffix_target_tokens) / baseline_target_tokens
            if baseline_target_tokens
            else ""
        ),
        "cv_draft_tokens_generated_est_profile": draft_tokens_generated_est,
        "cv_draft_tokens_full_k_est_profile": draft_tokens_full_k_est,
        "cv_draft_tokens_saved_by_staging_est_profile": max(
            0, draft_tokens_full_k_est - draft_tokens_generated_est
        ),
        "cv_draft_tokens_accepted_est_profile": draft_tokens_accepted_est,
        "cv_draft_tokens_discarded_est_profile": draft_tokens_discarded_est,
        "cv_draft_discard_ratio_est_profile": (
            draft_tokens_discarded_est / draft_tokens_generated_est
            if draft_tokens_generated_est
            else ""
        ),
        "cv_draft_unverified_suffix_ratio_est_profile": (
            skipped_suffix_tokens / draft_tokens_generated_est
            if draft_tokens_generated_est
            else ""
        ),
        "cv_staged_drafting_prefix_count_profile": sum(
            1 for row in prefix_scheduled if row.get("staged_drafting")
        ),
        "cv_staged_suffix_registered_count_profile": len(
            staged_suffix_registered
        ),
        "cv_extra_tlm_forwards_per_prefix_profile": (
            extra_tlm_forward / len(prefix_results) if prefix_results else ""
        ),
        "cv_prefix_dispatch_count_avg_profile": (
            statistics.mean(dispatch_count_values) if dispatch_count_values else ""
        ),
        "cv_prefix_dispatch_util_avg_profile": (
            statistics.mean(dispatch_util_values) if dispatch_util_values else ""
        ),
        "cv_prefix_underfilled_dispatch_ratio_profile": (
            sum(1 for item in underfilled if item) / len(underfilled)
            if underfilled
            else ""
        ),
        "cv_forward_plan_modes": ",".join(
            f"{mode}:{count}" for mode, count in sorted(forward_mode_hist.items())
        ),
        "cv_prefix_profile_results": result_total,
        "cv_prefix_profile_log_limit_reached": int(log_limit_reached),
        "cv_prefix_accept_histogram": ",".join(
            f"{accepted}:{hist[accepted]}" for accepted in sorted(hist)
        ),
        "cv_prefix_full_accept_ratio": full_accept / total,
        "cv_prefix_zero_accept_ratio": zero_accept / total,
        "cv_prefix_reject_ratio_profile": reject / total,
    }


def scenario_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("dataset", "")),
        str(row.get("K", "")),
        str(row.get("batch_size", "")),
    )


def method_is_cv(method: str, prefixes: tuple[str, ...]) -> bool:
    return any(method.startswith(prefix) for prefix in prefixes)


def diagnose(row: dict[str, Any]) -> tuple[str, str]:
    reasons: list[str] = []
    speedup = safe_float(row.get("speedup_vs_baseline"))
    skipped = safe_float(row.get("cv_skipped_suffix_ratio"))
    verify_ratio = safe_float(row.get("cv_verify_target_token_ratio_vs_oneshot_est"))
    base_accept = safe_float(row.get("baseline_spec_mean_acceptance_len_mid_avg"))
    cv_accept = safe_float(row.get("cv_spec_mean_acceptance_len_mid_avg"))
    dispatch_util = safe_float(row.get("cv_prefix_dispatch_util_avg"))
    underfilled = safe_float(row.get("cv_prefix_underfilled_dispatch_ratio"))
    selected_h = safe_float(row.get("cv_selected_h_avg"))
    accepted_prefix = safe_float(row.get("cv_prefix_accepted_tokens_avg"))
    full_accept_ratio = safe_float(row.get("cv_prefix_full_accept_ratio"))
    quality_gate = str(row.get("cv_quality_gate_status", ""))
    cv_enforce_eager = safe_float(row.get("cv_vllm_enforce_eager"))
    cv_cudagraph_none = safe_float(row.get("cv_vllm_cudagraph_none"))
    draft_discard = safe_float(row.get("cv_draft_discard_ratio_est_profile"))
    unverified_suffix = safe_float(
        row.get("cv_draft_unverified_suffix_ratio_est_profile")
    )
    staging_saved = safe_float(
        row.get("cv_draft_tokens_saved_by_staging_est_profile")
    )
    cv_gpu_util = safe_float(
        row.get("cv_gpu_active_util") or row.get("cv_gpu_util")
    )
    cv_gpu_busy = safe_float(
        row.get("cv_gpu_active_busy_ratio_50") or row.get("cv_gpu_busy_ratio_50")
    )

    if skipped is not None and skipped >= 0.4 and verify_ratio is not None and verify_ratio <= 0.6:
        reasons.append("suffix_verifier_tokens_were_skipped")
    if base_accept is not None and cv_accept is not None and cv_accept < base_accept * 0.95:
        reasons.append("acceptance_length_dropped")
    if dispatch_util is not None and dispatch_util < 0.6:
        reasons.append("prefix_dispatch_underfilled")
    if underfilled is not None and underfilled >= 0.5:
        reasons.append("most_prefix_dispatches_below_util_threshold")
    if selected_h is not None and accepted_prefix is not None and accepted_prefix < selected_h * 0.5:
        reasons.append("fixed_prefix_too_long_for_first_reject_distribution")
    if full_accept_ratio is not None and full_accept_ratio < 0.25:
        reasons.append("prefix_rarely_survives_to_suffix")
    if cv_enforce_eager == 1 or cv_cudagraph_none == 1:
        reasons.append("cv_ran_without_cudagraphs")
    if cv_gpu_util is not None and cv_gpu_util < 50.0:
        reasons.append("gpu_underutilized")
    if cv_gpu_busy is not None and cv_gpu_busy < 0.5:
        reasons.append("gpu_rarely_above_50pct")
    if "drop" in quality_gate:
        reasons.append("math_quality_drop")
    if unverified_suffix is not None and unverified_suffix > 0:
        reasons.append("suffix_tlm_skipped_after_dlm_already_drafted_it")
    if draft_discard is not None and draft_discard >= 0.5:
        reasons.append("most_drafted_tokens_are_not_accepted")
    if staging_saved is not None and staging_saved > 0:
        reasons.append("staged_drafting_saved_dlm_suffix_tokens")
    else:
        reasons.append("drafter_still_generates_full_k_before_prefix_result")

    if speedup is not None and speedup >= 1.0:
        label = "speedup_observed"
    elif skipped is not None and skipped > 0 and (
        verify_ratio is None or verify_ratio < 1.0
    ):
        label = "verifier_savings_hidden_by_overheads"
    else:
        label = "no_verifier_savings_observed"
    return label, ";".join(dict.fromkeys(reasons))


def build_rows(
    roots: list[Path],
    *,
    baseline_method: str,
    cv_prefixes: tuple[str, ...],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for root in roots:
        summary_rows = load_summary_rows(root)
        rows_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in summary_rows:
            if row.get("status") not in ("", "ok"):
                continue
            rows_by_key.setdefault(scenario_key(row), []).append(row)
        for key, rows in sorted(rows_by_key.items()):
            baseline = next(
                (row for row in rows if row.get("method") == baseline_method),
                None,
            )
            if baseline is None:
                continue
            baseline_dir = Path(str(baseline.get("output_dir", "")))
            baseline_log = parse_vllm_spec_metrics(baseline_dir)
            for cv in rows:
                method = str(cv.get("method", ""))
                if not method_is_cv(method, cv_prefixes):
                    continue
                cv_dir = Path(str(cv.get("output_dir", "")))
                cv_log = parse_vllm_spec_metrics(cv_dir)
                cv_profile = parse_cv_prefix_profile(cv_dir)
                base_throughput = safe_float(baseline.get("throughput"))
                cv_throughput = safe_float(cv.get("throughput"))
                speedup = (
                    cv_throughput / base_throughput
                    if base_throughput and cv_throughput is not None
                    else ""
                )
                row: dict[str, Any] = {
                    "root": str(root),
                    "model": key[0],
                    "dataset": key[1],
                    "K": key[2],
                    "batch_size": key[3],
                    "baseline_method": baseline_method,
                    "cv_method": method,
                    "baseline_throughput": baseline.get("throughput", ""),
                    "cv_throughput": cv.get("throughput", ""),
                    "speedup_vs_baseline": speedup,
                    "baseline_actual_average_batch_size": baseline.get(
                        "actual_average_batch_size", ""
                    ),
                    "cv_actual_average_batch_size": cv.get(
                        "actual_average_batch_size", ""
                    ),
                    "baseline_quality_score": baseline.get("quality_score", ""),
                    "cv_quality_score": cv.get("quality_score", ""),
                    "cv_quality_gate_status": cv.get("quality_gate_status", ""),
                    "baseline_gpu_util": baseline.get("gpu_util", ""),
                    "baseline_gpu_util_p95": baseline.get("gpu_util_p95", ""),
                    "baseline_gpu_busy_ratio_50": baseline.get(
                        "gpu_busy_ratio_50", ""
                    ),
                    "baseline_gpu_power_avg": baseline.get("gpu_power_avg", ""),
                    "baseline_gpu_active_util": baseline.get(
                        "gpu_active_util", ""
                    ),
                    "baseline_gpu_active_util_p95": baseline.get(
                        "gpu_active_util_p95", ""
                    ),
                    "baseline_gpu_active_busy_ratio_50": baseline.get(
                        "gpu_active_busy_ratio_50", ""
                    ),
                    "baseline_gpu_active_busy_ratio_80": baseline.get(
                        "gpu_active_busy_ratio_80", ""
                    ),
                    "baseline_gpu_active_power_avg": baseline.get(
                        "gpu_active_power_avg", ""
                    ),
                    "cv_gpu_util": cv.get("gpu_util", ""),
                    "cv_gpu_util_p95": cv.get("gpu_util_p95", ""),
                    "cv_gpu_busy_ratio_50": cv.get("gpu_busy_ratio_50", ""),
                    "cv_gpu_power_avg": cv.get("gpu_power_avg", ""),
                    "cv_gpu_active_util": cv.get("gpu_active_util", ""),
                    "cv_gpu_active_util_p95": cv.get("gpu_active_util_p95", ""),
                    "cv_gpu_active_busy_ratio_50": cv.get(
                        "gpu_active_busy_ratio_50", ""
                    ),
                    "cv_gpu_active_busy_ratio_80": cv.get(
                        "gpu_active_busy_ratio_80", ""
                    ),
                    "cv_gpu_active_power_avg": cv.get("gpu_active_power_avg", ""),
                    "cv_selected_h_avg": cv.get("selected_h_avg", ""),
                    "cv_prefix_rejected_ratio": cv.get("prefix_rejected_ratio", ""),
                    "cv_skipped_suffix_ratio": cv.get("skipped_suffix_ratio", ""),
                    "cv_verify_target_token_ratio_vs_oneshot_est": cv.get(
                        "verify_target_token_ratio_vs_oneshot_est", ""
                    ),
                    "cv_prefix_accepted_tokens_avg": cv.get(
                        "prefix_accepted_tokens_avg", ""
                    ),
                    "cv_prefix_dispatch_count_avg": cv.get(
                        "prefix_dispatch_count_avg", ""
                    ),
                    "cv_prefix_dispatch_util_avg": cv.get(
                        "prefix_dispatch_util_avg", ""
                    ),
                    "cv_prefix_underfilled_dispatch_ratio": cv.get(
                        "prefix_underfilled_dispatch_ratio", ""
                    ),
                    "cv_prefix_singleton_dispatch_ratio": cv.get(
                        "prefix_singleton_dispatch_ratio", ""
                    ),
                    "cv_profile_log_limit_reached": cv.get(
                        "profile_log_limit_reached", ""
                    ),
                    "baseline_profile_log_limit_reached": baseline.get(
                        "profile_log_limit_reached", ""
                    ),
                }
                for name, value in baseline_log.items():
                    row[f"baseline_{name}"] = value
                for name, value in cv_log.items():
                    row[f"cv_{name}"] = value
                row.update(cv_profile)
                fallback_fields = {
                    "cv_selected_h_avg": "cv_selected_h_avg_profile",
                    "cv_prefix_accepted_tokens_avg": (
                        "cv_prefix_accepted_tokens_avg_profile"
                    ),
                    "cv_skipped_suffix_ratio": "cv_skipped_suffix_ratio_profile",
                    "cv_verify_target_token_ratio_vs_oneshot_est": (
                        "cv_verify_target_token_ratio_vs_oneshot_est_profile"
                    ),
                    "cv_prefix_dispatch_count_avg": (
                        "cv_prefix_dispatch_count_avg_profile"
                    ),
                    "cv_prefix_dispatch_util_avg": (
                        "cv_prefix_dispatch_util_avg_profile"
                    ),
                    "cv_prefix_underfilled_dispatch_ratio": (
                        "cv_prefix_underfilled_dispatch_ratio_profile"
                    ),
                }
                for target, source in fallback_fields.items():
                    if row.get(target) in ("", None) and row.get(source) not in (
                        "",
                        None,
                    ):
                        row[target] = row[source]
                base_accept = safe_float(row.get("baseline_spec_mean_acceptance_len_mid_avg"))
                cv_accept = safe_float(row.get("cv_spec_mean_acceptance_len_mid_avg"))
                row["mean_acceptance_len_delta"] = (
                    cv_accept - base_accept
                    if base_accept is not None and cv_accept is not None
                    else ""
                )
                base_draft_tps = safe_float(row.get("baseline_spec_drafted_tps_mid_avg"))
                cv_draft_tps = safe_float(row.get("cv_spec_drafted_tps_mid_avg"))
                row["drafted_tps_ratio_cv_over_baseline"] = (
                    cv_draft_tps / base_draft_tps
                    if base_draft_tps and cv_draft_tps is not None
                    else ""
                )
                for pos in range(1, 9):
                    base_pos = safe_float(row.get(f"baseline_spec_accept_pos{pos}_mid_avg"))
                    cv_pos = safe_float(row.get(f"cv_spec_accept_pos{pos}_mid_avg"))
                    row[f"spec_accept_pos{pos}_delta"] = (
                        cv_pos - base_pos
                        if base_pos is not None and cv_pos is not None
                        else ""
                    )
                label, reasons = diagnose(row)
                row["diagnosis"] = label
                row["diagnosis_reasons"] = reasons
                comparisons.append(row)
    return comparisons


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SpecLink-CV Performance Gap Analysis",
        "",
        "This report explains why observed verifier-token savings may not become end-to-end throughput gains.",
        "All rows are derived from existing GuideLLM/vLLM artifacts; no new serving run is launched.",
        "",
    ]
    if not rows:
        lines.append("No comparable baseline/CV rows were found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            "## Summary",
            "",
            "| model | dataset | K | bs | method | CV throughput | EAGLE3 throughput | speedup | skipped suffix | TLM token ratio | accept len EAGLE3 -> CV | dispatch util | eager | diagnosis |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('model')} | {row.get('dataset')} | {row.get('K')} | "
            f"{row.get('batch_size')} | {row.get('cv_method')} | "
            f"{fmt(row.get('cv_throughput'), 2)} | "
            f"{fmt(row.get('baseline_throughput'), 2)} | "
            f"{fmt(row.get('speedup_vs_baseline'), 3)} | "
            f"{fmt(row.get('cv_skipped_suffix_ratio'), 3)} | "
            f"{fmt(row.get('cv_verify_target_token_ratio_vs_oneshot_est'), 3)} | "
            f"{fmt(row.get('baseline_spec_mean_acceptance_len_mid_avg'), 2)} -> "
            f"{fmt(row.get('cv_spec_mean_acceptance_len_mid_avg'), 2)} | "
            f"{fmt(row.get('cv_prefix_dispatch_util_avg'), 3)} | "
            f"{fmt(row.get('cv_vllm_enforce_eager'), 0)} | "
            f"{row.get('diagnosis')} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance Drift",
            "",
            "| model | dataset | K | bs | h | prefix full-accept | prefix zero-accept | prefix accept histogram | pos1 delta | pos2 delta | pos3 delta | pos4 delta |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('model')} | {row.get('dataset')} | {row.get('K')} | "
            f"{row.get('batch_size')} | {fmt(row.get('cv_selected_h_avg'), 1)} | "
            f"{fmt(row.get('cv_prefix_full_accept_ratio'), 3)} | "
            f"{fmt(row.get('cv_prefix_zero_accept_ratio'), 3)} | "
            f"{row.get('cv_prefix_accept_histogram', '')} | "
            f"{fmt(row.get('spec_accept_pos1_delta'), 3)} | "
            f"{fmt(row.get('spec_accept_pos2_delta'), 3)} | "
            f"{fmt(row.get('spec_accept_pos3_delta'), 3)} | "
            f"{fmt(row.get('spec_accept_pos4_delta'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Forward Plans",
            "",
            "| model | dataset | K | bs | method | forward phase:mode counts |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('model')} | {row.get('dataset')} | {row.get('K')} | "
            f"{row.get('batch_size')} | {row.get('cv_method')} | "
            f"{row.get('cv_forward_plan_modes', '')} |"
        )
    lines.extend(
        [
            "",
            "## Draft-Side Cost",
            "",
            "| model | dataset | K | bs | method | draft generated est. | full-K draft est. | staging saved est. | accepted draft est. | discarded draft ratio |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('model')} | {row.get('dataset')} | {row.get('K')} | "
            f"{row.get('batch_size')} | {row.get('cv_method')} | "
            f"{fmt(row.get('cv_draft_tokens_generated_est_profile'), 0)} | "
            f"{fmt(row.get('cv_draft_tokens_full_k_est_profile'), 0)} | "
            f"{fmt(row.get('cv_draft_tokens_saved_by_staging_est_profile'), 0)} | "
            f"{fmt(row.get('cv_draft_tokens_accepted_est_profile'), 0)} | "
            f"{fmt(row.get('cv_draft_discard_ratio_est_profile'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## GPU Utilization",
            "",
            "| model | dataset | K | bs | method | EAGLE3 active util avg | CV active util avg | CV active p95 | CV active busy >=80% | CV total util avg |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('model')} | {row.get('dataset')} | {row.get('K')} | "
            f"{row.get('batch_size')} | {row.get('cv_method')} | "
            f"{fmt(row.get('baseline_gpu_active_util') or row.get('baseline_gpu_util'), 1)} | "
            f"{fmt(row.get('cv_gpu_active_util') or row.get('cv_gpu_util'), 1)} | "
            f"{fmt(row.get('cv_gpu_active_util_p95') or row.get('cv_gpu_util_p95'), 1)} | "
            f"{fmt(row.get('cv_gpu_active_busy_ratio_80'), 3)} | "
            f"{fmt(row.get('cv_gpu_util'), 1)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `skipped suffix` measures how many draft suffix tokens were not verified after a prefix reject.",
            "- `unverified suffix ratio` estimates the share of DLM-generated draft tokens that were never verified by the TLM because the prefix rejected. This is useful because prefix reject correctly skips suffix verification, but the current DLM path has already paid to draft those suffix tokens.",
            "- `staging saved est.` estimates draft-model suffix tokens avoided by staged drafting. Non-staged CV still drafts full K before knowing whether the prefix rejects.",
            "- `discarded draft ratio` estimates the share of DLM-generated draft tokens that did not become accepted output tokens. It includes both verified rejects and suffix tokens discarded after prefix reject.",
            "- GPU utilization is sampled with `nvidia-smi` when `--gpu-util-sampling-ms` is enabled. The active GPU columns filter samples to GuideLLM's benchmark start/end timestamps, while total GPU util still includes GuideLLM setup/validation idle time.",
            "- `TLM token ratio` estimates target-model verification tokens relative to one-shot EAGLE3. A prefix-only verify roughly schedules `h + 1` target tokens instead of `K + 1`, before accounting for suffix replays or dense realignment.",
            "- `prefix full-accept` is the fraction of prefix probes that survive to suffix verification. If it is low, the run saves suffix verification but also mostly terminates speculative steps inside the prefix.",
            "- `posN delta` compares CV rolling per-position acceptance against the EAGLE3 baseline. Negative early-position deltas mean the chunked path is not only truncating late suffix tokens; it is changing early accept/reject behavior.",
            "- A speedup can disappear when the drafter still generates full K before the prefix result is known, prefix verifier batches are underfilled, or chunked verification lowers the average accepted length and increases decode steps.",
            "- Rows with `profile_log_limit_reached=1` are still useful for representative diagnostics, but not full-run distributions.",
            "",
            "## Detailed Reasons",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- {row.get('model')} {row.get('dataset')} K={row.get('K')} "
            f"bs={row.get('batch_size')} {row.get('cv_method')}: "
            f"{row.get('diagnosis_reasons')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-method", default="eagle3_oneshot")
    parser.add_argument("--cv-prefixes", default="cv_")
    args = parser.parse_args()

    roots = [root.resolve() for root in args.matrix_roots]
    output_dir = args.output_dir.resolve()
    cv_prefixes = tuple(item.strip() for item in args.cv_prefixes.split(",") if item.strip())
    rows = build_rows(
        roots,
        baseline_method=args.baseline_method,
        cv_prefixes=cv_prefixes,
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(output_dir / "performance_gap.csv", rows, fieldnames or None)
    write_json(output_dir / "performance_gap.json", {"rows": rows})
    write_markdown(output_dir / "performance_gap.md", rows)
    print(f"[INFO] Wrote {output_dir / 'performance_gap.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
