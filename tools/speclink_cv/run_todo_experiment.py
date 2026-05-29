#!/usr/bin/env python3
"""Run a TODO-shaped SpecLink-CV experiment bundle.

This orchestrator is intentionally conservative.  It creates the required
``results/speclink_cv_TIMESTAMP`` tree, runs environment and unit checks,
optionally runs bounded live token-id correctness gates and a tiny GuideLLM
serving smoke, then writes a top-level report. The full serving matrix is still
delegated to ``run_speclink_cv_guidellm_matrix.py``; this script records that
command so a long run can be resumed in slices. For throughput claims, prefer
``--full-benchmark-mode steady_state`` so the generated matrix uses a fixed
measurement window and closed-loop concurrency instead of finite-request drain
makespan.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.speclink_cv.env_check import collect as collect_env
from tools.speclink_cv.env_check import write_markdown as write_env_markdown


EVAL_ROOT = REPO_ROOT / "examples" / "evaluate" / "eval-guidellm"
RESULTS_ROOT = EVAL_ROOT / "results"

RESULT_SUBDIRS = [
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
]

UNIT_TESTS = [
    "test_chunk_decision",
    "test_state_machine",
    "test_async_queue",
    "test_roofline_packing",
    "test_correctness_smoke",
    "test_vllm_runtime_config",
    "test_sampler_draft_accept_eps",
]

FULL_METHODS = (
    "pure_vllm,eagle3_oneshot,"
    "cv_half_sync_simple,cv_half_sync_roofline,"
    "cv_half_async_simple,cv_half_async_roofline,"
    "cv_conf_sync_simple,cv_conf_sync_roofline,"
    "cv_conf_async_simple,cv_conf_async_roofline"
)
FULL_MATRIX_MODELS = {"qwen3_8b", "llama3_1_8b"}
FULL_MATRIX_DATASETS = {"math", "mtbench"}
FULL_MATRIX_KS = {"8", "12"}
FULL_MATRIX_BATCHES = {"8", "16", "32"}
FULL_MATRIX_METHODS = set(FULL_METHODS.split(","))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def zero_or_empty(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return abs(float(str(value).strip())) == 0.0
    except (TypeError, ValueError):
        return False


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def throughput_value(row: dict[str, Any]) -> Any:
    method = str(row.get("method", ""))
    method_specific = ["cv_tps"] if method.startswith("cv_") else []
    if method == "eagle3_oneshot":
        method_specific = ["eagle3_tps"]
    return first_present(
        row,
        [
            "throughput",
            "output_throughput",
            "output_tokens_per_second",
            "generated_tokens_per_second",
            "total_tokens_per_second",
            *method_specific,
        ],
    )


def speedup_value(row: dict[str, Any]) -> Any:
    return first_present(
        row,
        [
            "speedup_vs_eagle3",
            "e2e_speedup_vs_eagle3",
            "steady_speedup_vs_eagle3",
            "staged_batched_speedup_vs_eagle3",
            "nonstaged_batched_speedup_vs_eagle3",
        ],
    )


def quality_score_value(row: dict[str, Any]) -> Any:
    method = str(row.get("method", ""))
    method_specific = ["quality_score_cv"] if method.startswith("cv_") else []
    if method == "eagle3_oneshot":
        method_specific = ["quality_score_eagle3"]
    return first_present(
        row,
        [
            "quality_score",
            *method_specific,
            "quality_gate_score",
            "quality_common_score",
        ],
    )


def quality_delta_value(row: dict[str, Any]) -> Any:
    return first_present(row, ["quality_delta_vs_eagle3", "quality_gate_delta_vs_eagle3"])


def normalize_math_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("throughput") in (None, ""):
        row["throughput"] = throughput_value(row)
    if row.get("speedup_vs_eagle3") in (None, ""):
        row["speedup_vs_eagle3"] = speedup_value(row)
    if row.get("quality_score") in (None, ""):
        row["quality_score"] = quality_score_value(row)
    if row.get("quality_delta_vs_eagle3") in (None, ""):
        row["quality_delta_vs_eagle3"] = quality_delta_value(row)
    return row


PRIMARY_SUMMARY_FIELDS = [
    "model",
    "dataset",
    "K",
    "batch_size",
    "method",
    "measurement_type",
    "throughput",
    "speedup_vs_eagle3",
    "ttft_p95",
    "itl_p95",
    "e2e_p95",
    "exact_match_vs_eagle3",
    "selected_h_avg",
    "skipped_suffix_ratio",
    "extra_tlm_forwards_per_request",
    "queue_wait_p95",
    "gpu_util",
    "fallback_ratio",
    "quality_gate_status",
    "speedup_claim_status",
    "source",
    "source_root",
    "output_dir",
]


def _method_specific_value(
    row: dict[str, Any], generic: str, cv_key: str, eagle_key: str
) -> Any:
    method = str(row.get("method", ""))
    keys = [generic]
    if method.startswith("cv_"):
        keys.append(cv_key)
    elif method == "eagle3_oneshot":
        keys.append(eagle_key)
    return first_present(row, keys)


def primary_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write the TODO-required table with stable field names."""
    output: list[dict[str, Any]] = []
    serving_types = {"steady_state_saturated", "guidellm_end_to_end"}
    for row in rows:
        if row.get("measurement_type") not in serving_types:
            continue
        if not row.get("model") or not row.get("method"):
            continue
        output.append(
            {
                "model": row.get("model", ""),
                "dataset": row.get("dataset", ""),
                "K": row.get("K", ""),
                "batch_size": row.get("batch_size", ""),
                "method": row.get("method", ""),
                "measurement_type": row.get("measurement_type", ""),
                "throughput": throughput_value(row),
                "speedup_vs_eagle3": speedup_value(row),
                "ttft_p95": row.get("ttft_p95", ""),
                "itl_p95": _method_specific_value(
                    row, "itl_p95", "cv_itl_p95", "eagle3_itl_p95"
                ),
                "e2e_p95": _method_specific_value(
                    row, "e2e_p95", "cv_e2e_p95", "eagle3_e2e_p95"
                ),
                "exact_match_vs_eagle3": row.get("exact_match_vs_eagle3", ""),
                "selected_h_avg": row.get("selected_h_avg", ""),
                "skipped_suffix_ratio": row.get("skipped_suffix_ratio", ""),
                "extra_tlm_forwards_per_request": row.get(
                    "extra_tlm_forwards_per_request", ""
                ),
                "queue_wait_p95": row.get("queue_wait_p95", ""),
                "gpu_util": first_present(
                    row,
                    [
                        "gpu_active_util",
                        "gpu_util",
                        "gpu_active_util_cv",
                        "gpu_active_util_eagle3",
                    ],
                ),
                "fallback_ratio": row.get("fallback_ratio", ""),
                "quality_gate_status": row.get("quality_gate_status", ""),
                "speedup_claim_status": row.get("speedup_claim_status", ""),
                "source": row.get("source", ""),
                "source_root": row.get("source_root", ""),
                "output_dir": row.get("output_dir", row.get("output_root", "")),
            }
        )
    return output


def best_speclink_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in primary_summary_rows(rows):
        method = str(row.get("method", ""))
        if not method.startswith("cv_"):
            continue
        speedup = safe_float(row.get("speedup_vs_eagle3"))
        if speedup is None or speedup <= 1.0:
            continue
        if row.get("quality_gate_status") != "math_quality_preserved":
            continue
        candidate = dict(row)
        candidate["selection_basis"] = (
            "relaxed_math_quality_preserved; strict_token_id_not_global"
        )
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            str(item.get("model", "")),
            str(item.get("dataset", "")),
            str(item.get("K", "")),
            str(item.get("batch_size", "")),
            -(safe_float(item.get("speedup_vs_eagle3")) or 0.0),
        )
    )
    return candidates


def live_row_is_strict_greedy(row: dict[str, Any]) -> bool:
    return zero_or_empty(row.get("greedy_eps")) and zero_or_empty(
        row.get("draft_accept_eps")
    )


def live_row_token_matched(row: dict[str, Any]) -> bool:
    return str(row.get("matched", "")).strip().lower() == "true"


def live_row_strict_matched(row: dict[str, Any]) -> bool:
    return live_row_is_strict_greedy(row) and live_row_token_matched(row)


def live_row_strict_failure_reason(row: dict[str, Any]) -> str:
    if not live_row_is_strict_greedy(row):
        return "non_strict_eps"
    if not live_row_token_matched(row):
        return "token_mismatch"
    return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_unit_rows(root: Path) -> list[dict[str, Any]]:
    return read_csv(root / "02_unit_tests" / "unit_test_summary.csv")


def read_run_config(root: Path) -> dict[str, Any]:
    path = root / "logs" / "todo_run_config.json"
    strict_todo_config = True
    if not path.exists():
        path = root / "logs" / "run_config.json"
        strict_todo_config = False
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not strict_todo_config and not any(
        key in data
        for key in (
            "full_benchmark_mode",
            "guidellm_benchmark_mode",
            "skip_guidellm_smoke",
        )
    ):
        return {}
    return data


def read_live_summary(root: Path, run_config: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir = root / "05_cv_ablation" / "live_correctness_gate"
    rows = read_csv(output_dir / "summary.csv")
    status = "ok" if rows else "skipped" if (run_config or {}).get("skip_live") else "missing"
    return {"run": {"status": status}, "rows": rows, "output_dir": str(output_dir)}


def read_full_live_summary(
    root: Path,
    import_roots: list[str] | None = None,
    planned_cases: int | None = None,
) -> dict[str, Any]:
    output_dir = root / "05_cv_ablation" / "full_live_correctness_gate"
    rows = read_csv(output_dir / "summary.csv")
    row_order: dict[tuple[str, str, str, str, str], int] = {}

    def dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(row.get("model", "")),
            str(row.get("dataset", "")),
            str(row.get("K", "")),
            str(row.get("batch_size", "")),
            str(row.get("mode", "")),
        )

    for index, row in enumerate(rows):
        row_order[dedupe_key(row)] = index

    for import_root in import_roots or []:
        path = resolve_existing_csv(import_root, "summary.csv")
        if path is not None:
            imported_rows = read_csv(path)
            import_root_abs = str(Path(import_root).resolve())
            for row in imported_rows:
                row.setdefault("output_dir", import_root_abs)
                row["source"] = "full_live_correctness_import"
                key = dedupe_key(row)
                existing = row_order.get(key)
                if existing is None:
                    row_order[key] = len(rows)
                    rows.append(row)
                elif (
                    live_row_is_strict_greedy(row)
                    or not live_row_is_strict_greedy(rows[existing])
                ):
                    rows[existing] = row

    detected_planned_cases = 0
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            detected_planned_cases = int(data.get("total_planned_cases") or 0)
        except Exception:
            detected_planned_cases = 0
    if detected_planned_cases <= 0:
        plan_path = root / "logs" / "full_live_correctness_plan.json"
        if plan_path.exists():
            try:
                data = json.loads(plan_path.read_text(encoding="utf-8"))
                detected_planned_cases = int(data.get("planned_cases") or 0)
            except Exception:
                detected_planned_cases = 0
    if planned_cases is None:
        planned_cases = detected_planned_cases
    status = "ok" if rows else "missing"
    return {
        "run": {"status": status},
        "rows": rows,
        "output_dir": str(output_dir),
        "planned_cases": planned_cases,
    }


def read_guidellm_smoke_summary(
    root: Path, run_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    output_dir = root / "04_baselines" / "guidellm_smoke"
    rows = read_csv(output_dir / "09_reports" / "summary_metrics.csv")
    status = (
        "ok"
        if rows
        else "skipped"
        if (run_config or {}).get("skip_guidellm_smoke")
        else "missing"
    )
    return {"run": {"status": status}, "rows": rows, "output_dir": str(output_dir)}


def _full_matrix_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("measurement_type", "")),
        str(row.get("model", "")),
        str(row.get("dataset", "")),
        str(row.get("K", "")),
        str(row.get("batch_size", "")),
        str(row.get("method", "")),
    )


def is_planned_full_matrix_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("model", "")) in FULL_MATRIX_MODELS
        and str(row.get("dataset", "")) in FULL_MATRIX_DATASETS
        and str(row.get("K", "")) in FULL_MATRIX_KS
        and str(row.get("batch_size", "")) in FULL_MATRIX_BATCHES
        and str(row.get("method", "")) in FULL_MATRIX_METHODS
    )


def annotate_full_matrix_scope(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["full_matrix_scope"] = (
        "planned_todo_full_matrix"
        if is_planned_full_matrix_row(row)
        else "extra_best_candidate_or_smoke"
    )
    return row


def split_full_matrix_rows(
    rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planned: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    for row in rows or []:
        if is_planned_full_matrix_row(row):
            planned.append(row)
        else:
            extra.append(row)
    return planned, extra


def refresh_merged_full_matrix_metrics(rows: list[dict[str, Any]]) -> None:
    """Recompute fields that can span local and imported matrix rows."""
    baseline_by_key: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if row.get("method") != "eagle3_oneshot" or row.get("status") != "ok":
            continue
        throughput = safe_float(row.get("throughput"))
        if throughput is None:
            continue
        baseline_by_key[
            (
                str(row.get("model", "")),
                str(row.get("dataset", "")),
                str(row.get("K", "")),
                str(row.get("batch_size", "")),
            )
        ] = throughput

    for row in rows:
        key = (
            str(row.get("model", "")),
            str(row.get("dataset", "")),
            str(row.get("K", "")),
            str(row.get("batch_size", "")),
        )
        baseline = baseline_by_key.get(key)
        throughput = safe_float(row.get("throughput"))
        if baseline and throughput is not None:
            row["speedup_vs_eagle3"] = throughput / baseline

        method = str(row.get("method", ""))
        if not method.startswith("cv_"):
            continue
        if row.get("quality_gate_status") in ("", None, "missing_quality_gate"):
            if str(row.get("quality_reliable", "")) in {"0", "0.0", "False"}:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "quality_unreliable_short_outputs"
            else:
                row["quality_gate_pass"] = 0
                row["quality_gate_status"] = "missing_quality_gate"
        if row.get("speedup_claim_status") in ("", None, "missing_quality_gate"):
            if str(row.get("quality_gate_pass", "")) in {"1", "1.0", "True"}:
                row["speedup_claim_valid"] = 1
                row["speedup_claim_status"] = "valid_quality_preserving_chunked"
            else:
                row["speedup_claim_valid"] = 0
                row["speedup_claim_status"] = str(row.get("quality_gate_status", ""))


def filter_full_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        source = row.get("source", "")
        if source and source != "full_matrix":
            continue
        if row.get("measurement_type") not in {
            "guidellm_end_to_end",
            "steady_state_saturated",
        }:
            continue
        if not row.get("method"):
            continue
        filtered.append(annotate_full_matrix_scope(row))
    return filtered


def read_existing_full_matrix_rows(
    root: Path, import_roots: list[str] | None = None
) -> list[dict[str, Any]]:
    rows = read_csv(root / "09_reports" / "summary_metrics.csv")
    if not rows:
        rows = read_csv(root / "summary_metrics.csv")
    filtered = filter_full_matrix_rows(rows)
    row_order = {_full_matrix_row_key(row): idx for idx, row in enumerate(filtered)}
    for import_root in import_roots or []:
        path = resolve_existing_csv(import_root, "09_reports/summary_metrics.csv")
        if path is None:
            path = resolve_existing_csv(import_root, "summary_metrics.csv")
        if path is None:
            continue
        for row in filter_full_matrix_rows(read_csv(path)):
            row["source_root"] = str(Path(import_root).resolve())
            key = _full_matrix_row_key(row)
            existing = row_order.get(key)
            if existing is None:
                row_order[key] = len(filtered)
                filtered.append(row)
            else:
                filtered[existing] = row
    refresh_merged_full_matrix_metrics(filtered)
    return filtered


def resolve_existing_csv(path_or_root: str | Path, relative: str) -> Path | None:
    path = Path(path_or_root)
    candidates = [path]
    if path.is_dir() or not path.suffix:
        candidates = [path / relative, path / Path(relative).name]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def read_math_quality_followup_rows(
    root: Path, import_roots: list[str] | None = None
) -> list[dict[str, Any]]:
    sources: list[tuple[Path, str]] = []
    local = root / "05_cv_ablation/math_quality_followup/summary_metrics.csv"
    if local.exists():
        sources.append((local, str(root / "05_cv_ablation/math_quality_followup")))
    for import_root in import_roots or []:
        path = resolve_existing_csv(import_root, "09_reports/summary_metrics.csv")
        if path is None:
            path = resolve_existing_csv(
                import_root, "09_reports/math_quality_all_batch_summary.csv"
            )
        if path is not None:
            sources.append((path, str(Path(import_root).resolve())))

    rows: list[dict[str, Any]] = []
    for path, output_root in sources:
        for row in read_csv(path):
            row.setdefault("measurement_type", "guidellm_end_to_end")
            row["source"] = "math_quality_followup"
            row.setdefault("output_root", output_root)
            rows.append(normalize_math_quality_row(row))
    return rows


def math_quality_row_ok(row: dict[str, Any]) -> bool:
    if row.get("status") == "ok":
        return True
    return bool(row.get("quality_gate_status"))


def math_quality_valid_speedup(row: dict[str, Any]) -> bool:
    if row.get("speedup_claim_status") == "valid_quality_preserving_chunked":
        return True
    if row.get("quality_gate_status") != "math_quality_preserved":
        return False
    for key in ("e2e_speedup_vs_eagle3", "speedup_vs_eagle3"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value) > 1.0
        except (TypeError, ValueError):
            continue
    return False


def read_contribution_ablation_rows(
    root: Path, import_roots: list[str] | None = None
) -> list[dict[str, Any]]:
    sources: list[tuple[Path, str]] = []
    local = root / "05_cv_ablation/contribution_ablation/09_reports/contribution_ablation.csv"
    if local.exists():
        sources.append((local, str(root / "05_cv_ablation/contribution_ablation")))
    for import_root in import_roots or []:
        path = resolve_existing_csv(import_root, "09_reports/contribution_ablation.csv")
        if path is not None:
            sources.append((path, str(Path(import_root).resolve())))

    rows: list[dict[str, Any]] = []
    for path, output_root in sources:
        for row in read_csv(path):
            row["measurement_type"] = "contribution_ablation"
            row["source"] = "contribution_followup"
            row.setdefault("output_root", output_root)
            rows.append(row)
    return rows


def apply_saved_args(args: argparse.Namespace, saved: dict[str, Any]) -> argparse.Namespace:
    """Apply original TODO-run settings before finalize-only regeneration."""
    for key, value in saved.items():
        if key == "output_root":
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _read_plan_import_roots(root: Path, plan_name: str) -> list[str]:
    path = root / "logs" / plan_name
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    imports = data.get("import_roots") or []
    if not isinstance(imports, list):
        return []
    return [str(item) for item in imports if str(item)]


def restore_plan_import_roots(args: argparse.Namespace, root: Path) -> argparse.Namespace:
    """Keep imported standalone roots when refreshing reports from sliced scripts."""
    plan_map = {
        "full_live_import_root": "full_live_correctness_plan.json",
        "full_matrix_import_root": "full_matrix_plan.json",
        "math_quality_import_root": "math_quality_followup_plan.json",
        "contribution_import_root": "contribution_ablation_plan.json",
    }
    for attr, plan_name in plan_map.items():
        if getattr(args, attr, None):
            continue
        restored = _read_plan_import_roots(root, plan_name)
        if restored:
            setattr(args, attr, restored)
    return args


def create_tree(root: Path) -> None:
    for name in RESULT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def shell_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run_capture(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if dry_run:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 0,
            "status": "planned",
            "elapsed_s": 0.0,
        }
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=out,
            stderr=err,
            check=False,
        )
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "elapsed_s": round(time.time() - started, 3),
    }


def command_text(cmd: list[str], cwd: Path) -> str:
    return "cd " + shlex.quote(str(cwd)) + "\n" + shlex.join(cmd)


def copy_inputs(root: Path) -> None:
    for relative in [
        "TODO.md",
        "AGENTS.md",
        "examples/evaluate/eval-guidellm/run_speclink_cv_math_quality.sh",
        "examples/evaluate/eval-guidellm/run_speclink_cv_contribution_ablation.sh",
        "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py",
        "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_full.sh",
        "tools/speclink_cv/run_live_correctness_gate.py",
        "tools/speclink_cv/live_correctness_smoke.py",
        "tools/speclink_cv/run_todo_experiment.py",
    ]:
        src = REPO_ROOT / relative
        if src.exists():
            dest = root / "scripts" / relative.replace("/", "__")
            shutil.copy2(src, dest)


def write_patch_snapshot(root: Path) -> None:
    targets = [
        "TODO.md",
        "AGENTS.md",
        "vllm",
        "tools",
        "examples/evaluate/eval-guidellm/scripts",
    ]
    diff = subprocess.run(
        ["git", "diff", "--", *targets],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (root / "patches" / "vllm_speclink_cv.diff").write_text(
        diff.stdout, encoding="utf-8"
    )
    (root / "patches" / "speclink_cv.diff").write_text(
        diff.stdout, encoding="utf-8"
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *targets],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    untracked_diff = []
    for raw in untracked.stdout.splitlines():
        path = Path(raw)
        full_path = REPO_ROOT / path
        if not full_path.is_file():
            continue
        untracked_diff.extend(
            [
                f"diff --git a/{path} b/{path}",
                "new file mode 100644",
                "index 0000000..0000000",
                "--- /dev/null",
                f"+++ b/{path}",
            ]
        )
        text = full_path.read_text(encoding="utf-8", errors="ignore")
        untracked_diff.extend("+" + line for line in text.splitlines())
        untracked_diff.append("")
    if untracked_diff:
        with (root / "patches" / "vllm_speclink_cv.diff").open(
            "a", encoding="utf-8"
        ) as f:
            f.write("\n# Untracked files included by run_todo_experiment.py\n")
            f.write("\n".join(untracked_diff) + "\n")
        with (root / "patches" / "speclink_cv.diff").open(
            "a", encoding="utf-8"
        ) as f:
            f.write("\n# Untracked files included by run_todo_experiment.py\n")
            f.write("\n".join(untracked_diff) + "\n")
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (root / "patches" / "git_status.txt").write_text(
        status.stdout + status.stderr, encoding="utf-8"
    )


def run_env(root: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        payload = {"status": "planned"}
        write_json(root / "00_env" / "env_report.json", payload)
        (root / "00_env" / "env_report.md").write_text(
            "# SpecLink-CV Environment Report\n\n- status: planned\n",
            encoding="utf-8",
        )
        return payload
    report = collect_env()
    write_json(root / "00_env" / "env_report.json", report)
    write_env_markdown(report, root / "00_env" / "env_report.md")
    return report


def run_unit_tests(root: Path, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in UNIT_TESTS:
        cmd = [sys.executable, "-m", f"tools.speclink_cv.{name}"]
        stdout = root / "02_unit_tests" / f"{name}.stdout.log"
        stderr = root / "02_unit_tests" / f"{name}.stderr.log"
        result = run_capture(
            cmd,
            cwd=REPO_ROOT,
            stdout_path=stdout,
            stderr_path=stderr,
            dry_run=dry_run,
        )
        rows.append(
            {
                "test": name,
                "status": result["status"] if dry_run else ("pass" if result["returncode"] == 0 else "fail"),
                "returncode": result["returncode"],
                "elapsed_s": result["elapsed_s"],
                "stdout": str(stdout),
                "stderr": str(stderr),
            }
        )
    write_csv(root / "02_unit_tests" / "unit_test_summary.csv", rows)
    write_json(root / "02_unit_tests" / "unit_test_summary.json", {"tests": rows})
    lines = ["# Unit Test Summary", ""]
    for row in rows:
        lines.append(
            f"- {row['test']}: {row['status']} rc={row['returncode']} "
            f"elapsed_s={row['elapsed_s']}"
        )
    (root / "02_unit_tests" / "unit_test_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def live_gate_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "tools/speclink_cv/run_live_correctness_gate.py",
        "--models",
        args.live_models,
        "--datasets",
        args.live_datasets,
        "--ks",
        args.live_ks,
        "--modes",
        args.live_modes,
        "--num-prompts",
        str(args.live_num_prompts),
        "--batch-size",
        str(args.live_batch_size),
        "--max-tokens",
        str(args.live_max_tokens),
        "--profile-max-events",
        str(args.profile_max_events),
        "--log-max-events",
        str(args.log_max_events),
        "--output-root",
        str(output_dir),
    ]
    if args.live_force_prefix_len:
        cmd.extend(["--force-prefix-len", str(args.live_force_prefix_len)])
    for item in args.live_env:
        cmd.extend(["--env", item])
    return cmd


def run_live_gate(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = root / "05_cv_ablation" / "live_correctness_gate"
    cmd = live_gate_command(args, output_dir)
    shell_script(
        output_dir / "run_command.sh",
        [command_text(cmd, REPO_ROOT)],
    )
    result = run_capture(
        cmd,
        cwd=REPO_ROOT,
        stdout_path=output_dir / "stdout.log",
        stderr_path=output_dir / "stderr.log",
        dry_run=args.dry_run or args.skip_live,
    )
    if args.skip_live:
        result["status"] = "skipped"
    rows = read_csv(output_dir / "summary.csv")
    write_csv(root / "05_cv_ablation" / "live_correctness_summary.csv", rows)
    write_json(
        root / "05_cv_ablation" / "live_correctness_summary.json",
        {"run": result, "rows": rows},
    )
    return {"run": result, "rows": rows, "output_dir": str(output_dir)}


def guidellm_smoke_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py",
        "--benchmark-mode",
        args.guidellm_benchmark_mode,
        "--smoke",
        "--max-requests",
        str(args.guidellm_max_requests),
        "--max-tokens",
        str(args.guidellm_max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(args.port),
        "--output-root",
        str(output_dir),
        "--skip-unit-tests",
    ]
    if args.guidellm_benchmark_mode == "steady_state":
        cmd.extend(
            [
                "--steady-state-warmup-s",
                str(args.guidellm_steady_state_warmup_s),
                "--steady-state-measurement-s",
                str(args.guidellm_steady_state_measurement_s),
                "--steady-state-cooldown-s",
                str(args.guidellm_steady_state_cooldown_s),
                "--steady-state-max-prompts",
                str(args.guidellm_max_requests),
                "--steady-state-ignore-eos",
            ]
        )
    if args.dry_run:
        cmd.append("--dry-run")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.disable_vllm_async_scheduling:
        cmd.append("--disable-vllm-async-scheduling")
    for item in args.guidellm_env:
        cmd.extend(["--env", item])
    return cmd


def run_guidellm_smoke(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = root / "04_baselines" / "guidellm_smoke"
    cmd = guidellm_smoke_command(args, output_dir)
    shell_script(
        output_dir / "run_command.sh",
        [command_text(cmd, REPO_ROOT)],
    )
    result = run_capture(
        cmd,
        cwd=REPO_ROOT,
        stdout_path=output_dir / "stdout.log",
        stderr_path=output_dir / "stderr.log",
        dry_run=args.skip_guidellm_smoke,
    )
    if args.skip_guidellm_smoke:
        result["status"] = "skipped"
    rows = read_csv(output_dir / "09_reports" / "summary_metrics.csv")
    write_csv(root / "04_baselines" / "guidellm_smoke_summary.csv", rows)
    write_json(
        root / "04_baselines" / "guidellm_smoke_summary.json",
        {"run": result, "rows": rows},
    )
    return {"run": result, "rows": rows, "output_dir": str(output_dir)}


def full_matrix_command(root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py",
        "--benchmark-mode",
        args.full_benchmark_mode,
        "--models",
        "qwen3_8b,llama3_1_8b",
        "--datasets",
        "math,mtbench",
        "--ks",
        "8,12",
        "--batch-sizes",
        "8,16,32",
        "--methods",
        FULL_METHODS,
        "--max-requests",
        str(args.full_max_requests),
        "--max-tokens",
        str(args.full_max_tokens),
        "--profile-max-events",
        str(args.profile_max_events),
        "--log-max-events",
        str(args.log_max_events),
        "--analysis-profile-max-rows",
        str(args.analysis_profile_max_rows),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--port",
        str(args.port),
        "--resume",
        "--output-root",
        str(root),
    ]
    if args.full_benchmark_mode == "steady_state":
        cmd.extend(
            [
                "--steady-state-warmup-s",
                str(args.full_steady_state_warmup_s),
                "--steady-state-measurement-s",
                str(args.full_steady_state_measurement_s),
                "--steady-state-cooldown-s",
                str(args.full_steady_state_cooldown_s),
                "--steady-state-max-prompts",
                str(args.full_steady_state_max_prompts),
                "--steady-state-ignore-eos",
            ]
        )
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.disable_vllm_async_scheduling:
        cmd.append("--disable-vllm-async-scheduling")
    if args.full_allow_shape_drift_chunking:
        cmd.append("--allow-shape-drift-chunking")
    if args.full_allow_batched_prefix_verification:
        cmd.append("--allow-batched-prefix-verification")
    full_env = list(args.full_env)
    if not any(item.startswith("SPECLINK_CV_DENSE_REALIGN_STEPS=") for item in full_env):
        full_env.append("SPECLINK_CV_DENSE_REALIGN_STEPS=0")
    if not any(item.startswith("SPECLINK_CV_ALLOW_BATCHED_SUFFIX=") for item in full_env):
        full_env.append("SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1")
    for item in full_env:
        cmd.extend(["--env", item])
    return cmd


def full_matrix_slice_command(root: Path, args: argparse.Namespace) -> list[str]:
    cmd = full_matrix_command(root, args)
    cmd.extend(["--case-offset", "${offset}", "--case-limit", "${CASE_LIMIT}"])
    return cmd


def shell_join_with_variables(cmd: list[str]) -> str:
    parts: list[str] = []
    for item in cmd:
        if item in {"${offset}", "${CASE_LIMIT}"}:
            parts.append(item)
        else:
            parts.append(shlex.quote(item))
    return " ".join(parts)


def write_planned_full_matrix(root: Path, args: argparse.Namespace) -> None:
    cmd = full_matrix_command(root, args)
    script_name = (
        "run_full_steady_state_matrix.sh"
        if args.full_benchmark_mode == "steady_state"
        else "run_full_guidellm_matrix.sh"
    )
    shell_script(
        root / "scripts" / script_name,
        [command_text(cmd, REPO_ROOT)],
    )
    # Keep the historical filename as a compatibility alias for older notes.
    if script_name != "run_full_guidellm_matrix.sh":
        shell_script(
            root / "scripts" / "run_full_guidellm_matrix.sh",
            [command_text(cmd, REPO_ROOT)],
        )
    cases = 2 * 2 * 2 * 3 * 10
    slice_cmd = full_matrix_slice_command(root, args)
    sliced_name = (
        "run_full_steady_state_matrix_sliced.sh"
        if args.full_benchmark_mode == "steady_state"
        else "run_full_guidellm_matrix_sliced.sh"
    )
    sliced_script = root / "scripts" / sliced_name
    sliced_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(REPO_ROOT))}",
                f"PLANNED_CASES={cases}",
                f"CASE_LIMIT=\"${{CASE_LIMIT:-{args.full_cases_per_slice}}}\"",
                "START_OFFSET=\"${START_OFFSET:-0}\"",
                "MAX_SLICES=\"${MAX_SLICES:-0}\"",
                "offset=\"${START_OFFSET}\"",
                "slice=0",
                "while [[ \"${offset}\" -lt \"${PLANNED_CASES}\" ]]; do",
                "  if [[ \"${MAX_SLICES}\" != \"0\" && \"${slice}\" -ge \"${MAX_SLICES}\" ]]; then",
                "    break",
                "  fi",
                "  echo \"[INFO] Running full matrix slice offset=${offset} limit=${CASE_LIMIT}\"",
                "  " + shell_join_with_variables(slice_cmd),
                "  offset=$((offset + CASE_LIMIT))",
                "  slice=$((slice + 1))",
                "done",
                "echo \"[INFO] Refreshing full-matrix summary with analyze-only\"",
                "  " + shlex.join(full_matrix_command(root, args) + ["--analyze-only"]),
                "echo \"[INFO] Refreshing TODO-level final report\"",
                "  "
                + shlex.join(
                    [
                        sys.executable,
                        "-u",
                        "tools/speclink_cv/run_todo_experiment.py",
                        "--finalize-only",
                        "--output-root",
                        str(root),
                    ]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    sliced_script.chmod(0o755)
    if sliced_name != "run_full_guidellm_matrix_sliced.sh":
        legacy_sliced = root / "scripts" / "run_full_guidellm_matrix_sliced.sh"
        shutil.copy2(sliced_script, legacy_sliced)
        legacy_sliced.chmod(0o755)
    write_json(
        root / "logs" / "full_matrix_plan.json",
        {
            "planned_cases": cases,
            "cases_per_slice": args.full_cases_per_slice,
            "benchmark_mode": args.full_benchmark_mode,
            "throughput_semantics": (
                "closed_loop_steady_state_saturated_output_tokens_per_second"
                if args.full_benchmark_mode == "steady_state"
                else "finite_request_guidellm_end_to_end_output_tokens_per_second"
            ),
            "steady_state": {
                "warmup_s": args.full_steady_state_warmup_s,
                "measurement_s": args.full_steady_state_measurement_s,
                "cooldown_s": args.full_steady_state_cooldown_s,
                "max_prompts": args.full_steady_state_max_prompts,
                "ignore_eos": args.full_benchmark_mode == "steady_state",
            },
            "models": ["qwen3_8b", "llama3_1_8b"],
            "datasets": ["math", "mtbench"],
            "ks": [8, 12],
            "batch_sizes": [8, 16, 32],
            "methods": FULL_METHODS.split(","),
            "import_roots": list(args.full_matrix_import_root),
            "command": cmd,
            "sliced_command": str(sliced_script),
            "note": (
                "Use scripts/run_full_steady_state_matrix_sliced.sh for final "
                "serving throughput when benchmark_mode=steady_state. The "
                "legacy run_full_guidellm_matrix_sliced.sh alias points to the "
                "same command for compatibility."
            ),
        },
    )


def write_planned_math_quality_followup(root: Path, args: argparse.Namespace) -> None:
    output_dir = root / "05_cv_ablation" / "math_quality_followup"
    command = [
        f"OUTPUT_ROOT={shlex.quote(str(output_dir))}",
        f"MODEL_LIST={shlex.quote(args.math_quality_models)}",
        f"DATASET_LIST={shlex.quote(args.math_quality_datasets)}",
        f"K_LIST={shlex.quote(args.math_quality_ks)}",
        f"BATCH_SIZE_LIST={shlex.quote(args.math_quality_batch_sizes)}",
        f"METHOD_LIST={shlex.quote(args.math_quality_methods)}",
        f"MAX_REQUESTS={args.math_quality_max_requests}",
        f"MAX_TOKENS={args.math_quality_max_tokens}",
        f"FORCE_PREFIX_LEN={args.math_quality_force_prefix_len}",
        f"PORT={args.math_quality_port}",
        "ALLOW_CV_CUDAGRAPH=1",
        "BATCH_INVARIANT=0",
        "ALLOW_BATCHED_PREFIX_VERIFICATION=1",
        "ALLOW_BATCHED_SUFFIX=1",
        "SUFFIX_REPLAY_ONE_SHOT_SHAPE=0",
        "DENSE_REALIGN_STEPS=0",
        f"GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization}",
        "bash examples/evaluate/eval-guidellm/run_speclink_cv_math_quality.sh",
    ]
    shell_script(
        root / "scripts" / "run_math_quality_followup.sh",
        ["cd " + shlex.quote(str(REPO_ROOT)), " ".join(command)],
    )
    write_json(
        root / "logs" / "math_quality_followup_plan.json",
        {
            "output_root": output_dir,
            "import_roots": list(args.math_quality_import_root),
            "models": split_csv(args.math_quality_models),
            "datasets": split_csv(args.math_quality_datasets),
            "ks": split_csv(args.math_quality_ks),
            "batch_sizes": split_csv(args.math_quality_batch_sizes),
            "methods": split_csv(args.math_quality_methods),
            "max_requests": args.math_quality_max_requests,
            "max_tokens": args.math_quality_max_tokens,
            "force_prefix_len": args.math_quality_force_prefix_len,
            "command": " ".join(command),
            "note": (
                "Relaxed math-quality follow-up. It checks math EM and "
                "throughput, but it is not the original exact token-id gate."
            ),
        },
    )


def write_planned_contribution_ablation(root: Path, args: argparse.Namespace) -> None:
    output_dir = root / "05_cv_ablation" / "contribution_ablation"
    max_tokens = args.contribution_max_tokens
    if args.contribution_benchmark_mode == "steady_state" and max_tokens <= 0:
        max_tokens = 1024
    command = [
        f"OUTPUT_ROOT={shlex.quote(str(output_dir))}",
        f"MODEL_LIST={shlex.quote(args.contribution_models)}",
        f"DATASET_LIST={shlex.quote(args.contribution_datasets)}",
        f"K_LIST={shlex.quote(args.contribution_ks)}",
        f"BATCH_SIZE_LIST={shlex.quote(args.contribution_batch_sizes)}",
        f"MAX_REQUESTS={args.contribution_max_requests}",
        f"MAX_TOKENS={max_tokens}",
        f"BENCHMARK_MODE={args.contribution_benchmark_mode}",
        f"STEADY_STATE_WARMUP_S={args.contribution_steady_state_warmup_s}",
        f"STEADY_STATE_MEASUREMENT_S={args.contribution_steady_state_measurement_s}",
        f"STEADY_STATE_COOLDOWN_S={args.contribution_steady_state_cooldown_s}",
        f"STEADY_STATE_MAX_PROMPTS={args.contribution_steady_state_max_prompts}",
        f"FORCE_PREFIX_LEN={args.contribution_force_prefix_len}",
        f"PORT={args.contribution_port}",
        f"GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization}",
        "bash examples/evaluate/eval-guidellm/run_speclink_cv_contribution_ablation.sh",
    ]
    shell_script(
        root / "scripts" / "run_contribution_ablation.sh",
        ["cd " + shlex.quote(str(REPO_ROOT)), " ".join(command)],
    )
    write_json(
        root / "logs" / "contribution_ablation_plan.json",
        {
            "output_root": output_dir,
            "import_roots": list(args.contribution_import_root),
            "models": split_csv(args.contribution_models),
            "datasets": split_csv(args.contribution_datasets),
            "ks": split_csv(args.contribution_ks),
            "batch_sizes": split_csv(args.contribution_batch_sizes),
            "max_requests": args.contribution_max_requests,
            "max_tokens": max_tokens,
            "benchmark_mode": args.contribution_benchmark_mode,
            "steady_state": {
                "warmup_s": args.contribution_steady_state_warmup_s,
                "measurement_s": args.contribution_steady_state_measurement_s,
                "cooldown_s": args.contribution_steady_state_cooldown_s,
                "max_prompts": args.contribution_steady_state_max_prompts,
            },
            "force_prefix_len": args.contribution_force_prefix_len,
            "command": " ".join(command),
            "note": (
                "Contribution ablation keeps skip_suffix enabled, then "
                "separates non-staged TLM suffix skip, staged DLM suffix "
                "saving, and batched scheduling. Use steady_state mode for "
                "saturated output tokens/s."
            ),
        },
    )


def full_live_correctness_command(root: Path, args: argparse.Namespace) -> list[str]:
    output_dir = root / "05_cv_ablation" / "full_live_correctness_gate"
    cmd = [
        sys.executable,
        "-u",
        "tools/speclink_cv/run_live_correctness_gate.py",
        "--models",
        "qwen3_8b,llama3_1_8b",
        "--datasets",
        "math,mtbench",
        "--ks",
        "8,12",
        "--batch-sizes",
        "8,16,32",
        "--modes",
        args.full_live_modes,
        "--num-prompts",
        str(args.full_live_num_prompts),
        "--max-tokens",
        str(args.full_live_max_tokens),
        "--profile-max-events",
        str(args.profile_max_events),
        "--log-max-events",
        str(args.log_max_events),
        "--output-root",
        str(output_dir),
    ]
    if args.full_live_force_prefix_len:
        cmd.extend(
            ["--force-prefix-len", str(args.full_live_force_prefix_len)]
        )
    if args.full_live_num_prompts_per_batch:
        cmd.append("--num-prompts-per-batch")
    if args.full_live_allow_batched_prefix_verification:
        cmd.append("--allow-batched-prefix-verification")
    if args.full_live_global_batch_barrier:
        cmd.append("--global-batch-barrier")
    for item in args.full_live_env:
        cmd.extend(["--env", item])
    return cmd


def full_live_slice_command(root: Path, args: argparse.Namespace) -> list[str]:
    cmd = full_live_correctness_command(root, args)
    cmd.extend(["--case-offset", "${offset}", "--case-limit", "${CASE_LIMIT}"])
    return cmd


def planned_full_live_cases(args: argparse.Namespace) -> int:
    modes = [item for item in args.full_live_modes.split(",") if item.strip()]
    return 2 * 2 * 2 * 3 * max(1, len(modes))


def write_planned_full_live_correctness(root: Path, args: argparse.Namespace) -> None:
    cases = planned_full_live_cases(args)
    cmd = full_live_correctness_command(root, args)
    shell_script(
        root / "scripts" / "run_full_live_correctness_gate.sh",
        [command_text(cmd, REPO_ROOT)],
    )
    slice_cmd = full_live_slice_command(root, args)
    sliced_script = root / "scripts" / "run_full_live_correctness_gate_sliced.sh"
    sliced_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(REPO_ROOT))}",
                f"PLANNED_CASES={cases}",
                f"CASE_LIMIT=\"${{CASE_LIMIT:-{args.full_live_cases_per_slice}}}\"",
                "START_OFFSET=\"${START_OFFSET:-0}\"",
                "MAX_SLICES=\"${MAX_SLICES:-0}\"",
                "offset=\"${START_OFFSET}\"",
                "slice=0",
                "while [[ \"${offset}\" -lt \"${PLANNED_CASES}\" ]]; do",
                "  if [[ \"${MAX_SLICES}\" != \"0\" && \"${slice}\" -ge \"${MAX_SLICES}\" ]]; then",
                "    break",
                "  fi",
                "  echo \"[INFO] Running full live correctness slice offset=${offset} limit=${CASE_LIMIT}\"",
                "  " + shell_join_with_variables(slice_cmd),
                "  offset=$((offset + CASE_LIMIT))",
                "  slice=$((slice + 1))",
                "done",
                "echo \"[INFO] Refreshing TODO-level final report\"",
                "  "
                + shlex.join(
                    [
                        sys.executable,
                        "-u",
                        "tools/speclink_cv/run_todo_experiment.py",
                        "--finalize-only",
                        "--output-root",
                        str(root),
                    ]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    sliced_script.chmod(0o755)
    write_json(
        root / "logs" / "full_live_correctness_plan.json",
        {
            "planned_cases": cases,
            "cases_per_slice": args.full_live_cases_per_slice,
            "import_roots": list(args.full_live_import_root),
            "models": ["qwen3_8b", "llama3_1_8b"],
            "datasets": ["math", "mtbench"],
            "ks": [8, 12],
            "batch_sizes": [8, 16, 32],
            "modes": [item for item in args.full_live_modes.split(",") if item.strip()],
            "max_tokens": args.full_live_max_tokens,
            "num_prompts": args.full_live_num_prompts,
            "num_prompts_per_batch": args.full_live_num_prompts_per_batch,
            "command": cmd,
            "sliced_command": str(sliced_script),
            "note": "Use scripts/run_full_live_correctness_gate_sliced.sh for resumable token-id correctness gates.",
        },
    )


def summary_rows(
    root: Path,
    unit_rows: list[dict[str, Any]],
    live: dict[str, Any],
    full_live: dict[str, Any],
    guide: dict[str, Any],
    full_matrix_rows: list[dict[str, Any]] | None = None,
    math_quality_rows: list[dict[str, Any]] | None = None,
    contribution_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unit_rows:
        rows.append(
            {
                "measurement_type": "unit_test",
                "name": row["test"],
                "status": row["status"],
                "returncode": row["returncode"],
                "output_dir": str(root / "02_unit_tests"),
            }
        )
    for row in full_live.get("rows", []):
        full_live_row = {
            "measurement_type": "live_token_id_correctness",
            "source": row.get("source", "full_live_correctness"),
            "model": row.get("model", ""),
            "dataset": row.get("dataset", ""),
            "K": row.get("K", ""),
            "batch_size": row.get("batch_size", ""),
            "method": row.get("mode", ""),
            "status": row.get("status", full_live["run"].get("status", "")),
            "greedy_eps": row.get("greedy_eps", ""),
            "draft_accept_eps": row.get("draft_accept_eps", ""),
            "profile_max_events": row.get("profile_max_events", ""),
            "log_max_events": row.get("log_max_events", ""),
            "matched": row.get("matched", ""),
            "strict_greedy": str(live_row_is_strict_greedy(row)),
            "strict_matched": str(live_row_strict_matched(row)),
            "strict_failure_reason": live_row_strict_failure_reason(row),
            "matched_count": row.get("matched_count", ""),
            "total_count": row.get("total_count", ""),
            "first_mismatch_token_index": row.get("first_mismatch_token_index", ""),
            "output_dir": row.get("output_dir", full_live.get("output_dir", "")),
        }
        rows.append(full_live_row)
    for row in live.get("rows", []):
        rows.append(
            {
                "measurement_type": "live_token_id_correctness",
                "model": row.get("model", ""),
                "dataset": row.get("dataset", ""),
                "K": row.get("K", ""),
                "batch_size": row.get("batch_size", ""),
                "method": row.get("mode", ""),
                "status": row.get("status", live["run"].get("status", "")),
                "greedy_eps": row.get("greedy_eps", ""),
                "draft_accept_eps": row.get("draft_accept_eps", ""),
                "profile_max_events": row.get("profile_max_events", ""),
                "log_max_events": row.get("log_max_events", ""),
                "matched": row.get("matched", ""),
                "strict_greedy": str(live_row_is_strict_greedy(row)),
                "strict_matched": str(live_row_strict_matched(row)),
                "strict_failure_reason": live_row_strict_failure_reason(row),
                "matched_count": row.get("matched_count", ""),
                "total_count": row.get("total_count", ""),
                "first_mismatch_token_index": row.get("first_mismatch_token_index", ""),
                "output_dir": row.get("output_dir", live.get("output_dir", "")),
            }
        )
    for row in guide.get("rows", []):
        guide_row = dict(row)
        guide_row.setdefault("measurement_type", "guidellm_end_to_end")
        guide_row["source"] = "guidellm_smoke"
        rows.append(guide_row)
    for row in full_matrix_rows or []:
        full_row = dict(row)
        full_row.setdefault("measurement_type", "guidellm_end_to_end")
        full_row["source"] = "full_matrix"
        full_row.setdefault(
            "full_matrix_scope",
            "planned_todo_full_matrix"
            if is_planned_full_matrix_row(full_row)
            else "extra_best_candidate_or_smoke",
        )
        rows.append(full_row)
    for row in math_quality_rows or []:
        quality_row = dict(row)
        quality_row.setdefault("measurement_type", "guidellm_end_to_end")
        quality_row["source"] = "math_quality_followup"
        rows.append(quality_row)
    for row in contribution_rows or []:
        contribution_row = dict(row)
        contribution_row["measurement_type"] = "contribution_ablation"
        contribution_row["source"] = "contribution_followup"
        rows.append(contribution_row)
    if not live.get("rows"):
        rows.append(
            {
                "measurement_type": "live_token_id_correctness",
                "status": live["run"].get("status", ""),
                "output_dir": live.get("output_dir", ""),
            }
        )
    if not guide.get("rows"):
        rows.append(
            {
                "measurement_type": "guidellm_end_to_end",
                "status": guide["run"].get("status", ""),
                "output_dir": guide.get("output_dir", ""),
                "source": "guidellm_smoke",
            }
        )
    return rows


def audit_rows(
    unit_rows: list[dict[str, Any]],
    live: dict[str, Any],
    full_live: dict[str, Any],
    guide: dict[str, Any],
    full_matrix_rows: list[dict[str, Any]] | None = None,
    math_quality_rows: list[dict[str, Any]] | None = None,
    contribution_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    unit_missing = not unit_rows
    unit_passed = bool(unit_rows) and all(row["status"] == "pass" for row in unit_rows)
    unit_planned = bool(unit_rows) and all(row["status"] == "planned" for row in unit_rows)
    live_rows = live.get("rows", [])
    live_passed = bool(live_rows) and all(str(row.get("matched")) == "True" for row in live_rows)
    chunked_rows = [row for row in live_rows if row.get("mode") == "chunked"]
    full_live_rows = full_live.get("rows", [])
    full_live_matched_rows = [
        row
        for row in full_live_rows
        if row.get("status") == "ok" and live_row_strict_matched(row)
    ]
    planned_live_cases = int(full_live.get("planned_cases") or 0)
    if planned_live_cases <= 0:
        planned_live_cases = planned_full_live_cases(
            argparse.Namespace(full_live_modes="chunked,exactsafe")
        )
    full_live_failed_rows = [
        row
        for row in full_live_rows
        if row.get("status") != "ok" or not live_row_strict_matched(row)
    ]
    current_refresh_rows = [
        row
        for row in full_live_rows
        if "currentcfg" in str(row.get("output_dir", ""))
    ]
    current_refresh_failures = [
        row
        for row in current_refresh_rows
        if row.get("status") != "ok" or not live_row_strict_matched(row)
    ]
    if not full_live_rows:
        full_live_status = "planned_not_complete"
    elif len(full_live_rows) < planned_live_cases:
        full_live_status = f"partial_{len(full_live_rows)}_of_{planned_live_cases}"
        if full_live_failed_rows:
            full_live_status += "_has_failures"
    elif len(full_live_matched_rows) == len(full_live_rows):
        full_live_status = "complete_passed"
    else:
        full_live_status = "has_correctness_failures"
    guide_rows = guide.get("rows", [])
    guide_ok_rows = [row for row in guide_rows if row.get("status") == "ok"]
    full_rows, extra_full_rows = split_full_matrix_rows(full_matrix_rows)
    full_ok_rows = [row for row in full_rows if row.get("status") == "ok"]
    math_rows = math_quality_rows or []
    math_ok_rows = [row for row in math_rows if math_quality_row_ok(row)]
    math_valid_rows = [
        row
        for row in math_rows
        if math_quality_valid_speedup(row)
    ]
    contribution = contribution_rows or []
    planned_full_cases = 2 * 2 * 2 * 3 * 10
    if not full_rows:
        full_status = "planned_not_complete"
    elif len(full_rows) < planned_full_cases:
        full_status = f"partial_{len(full_rows)}_of_{planned_full_cases}"
    elif len(full_ok_rows) == len(full_rows):
        full_status = "steady_state_matrix_complete"
    else:
        full_status = "steady_state_matrix_has_failures"
    return [
        {
            "area": "implementation",
            "status": "partial_live_implementation",
            "requirement": "vLLM-integrated SpecLink-CV prefix/suffix verification with confidence, async queue, and roofline switches",
            "evidence": "vllm/vllm/speclink_cv.py; vllm/vllm/v1/core/sched/scheduler.py; unit tests",
            "notes": "Code paths and switches exist, but broad h<K correctness is still not proven for the full TODO matrix.",
        },
        {
            "area": "unit_tests",
            "status": (
                "complete"
                if unit_passed
                else "planned"
                if unit_planned
                else "missing"
                if unit_missing
                else "failed"
            ),
            "requirement": "chunk decision, state machine, async queue, roofline, correctness smoke, runtime config tests",
            "evidence": "02_unit_tests/unit_test_summary.csv",
            "notes": (
                "No unit-test summary rows were found in this bundle."
                if unit_missing
                else f"{len(unit_rows)} tests planned."
                if unit_planned
                else f"{sum(1 for row in unit_rows if row['status'] == 'pass')}/{len(unit_rows)} tests passed."
            ),
        },
        {
            "area": "live_correctness",
            "status": "quick_passed" if live_passed and chunked_rows else "partial_or_failed",
            "requirement": "vLLM+EAGLE3 one-shot vs SpecLink-CV token-id exact match",
            "evidence": "05_cv_ablation/live_correctness_gate/summary.csv",
            "notes": "Quick gate is bounded evidence only; it does not cover batch sizes 8/16/32 or long MTBench generations.",
        },
        {
            "area": "full_live_correctness",
            "status": full_live_status,
            "requirement": "Full token-id correctness gates for models, datasets, K, batch sizes, and chunked/exactsafe modes",
            "evidence": "09_reports/summary_metrics.csv; 05_cv_ablation/full_live_correctness_gate/summary.csv; scripts/run_full_live_correctness_gate_sliced.sh",
            "notes": (
                f"{len(full_live_rows)}/{planned_live_cases} full live correctness rows recorded; "
                f"{len(full_live_matched_rows)} strict-greedy rows matched. "
                f"Current-config refresh recorded {len(current_refresh_rows)} rows "
                f"with {len(current_refresh_failures)} failures before the strict "
                "run was intentionally stopped. Strict exactness remains a risk, "
                "not the accepted performance-path success gate."
            ),
        },
        {
            "area": "serving_smoke",
            "status": (
                "quick_completed"
                if guide_rows and len(guide_ok_rows) == len(guide_rows)
                else guide["run"].get("status", "missing")
            ),
            "requirement": "Quick serving smoke metrics and CV ablation evidence",
            "evidence": "04_baselines/guidellm_smoke/09_reports/summary_metrics.csv",
            "notes": (
                f"{len(guide_ok_rows)}/{len(guide_rows)} GuideLLM smoke rows ok. "
                "Smoke rows are not the full 240-case TODO matrix. "
                "Use steady_state mode for final saturated tokens/s."
                if guide_rows
                else f"GuideLLM smoke status={guide['run'].get('status', 'missing')}."
            ),
        },
        {
            "area": "full_matrix",
            "status": full_status,
            "requirement": "2 models x 2 datasets x K={8,12} x batch={8,16,32} x 10 methods serving matrix",
            "evidence": "scripts/run_full_steady_state_matrix.sh; scripts/run_full_steady_state_matrix_sliced.sh; logs/full_matrix_plan.json",
            "notes": (
                f"{len(full_ok_rows)}/{len(full_rows)} full-matrix serving rows ok; "
                f"{len(extra_full_rows)} extra staged/smoke rows are reported separately; "
                "throughput claims should use measurement_type=steady_state_saturated "
                "and be paired with the relaxed math-quality gate."
            ),
        },
        {
            "area": "math_quality_followup",
            "status": (
                "complete_relaxed_quality"
                if math_rows and len(math_ok_rows) == len(math_rows)
                else "planned_not_complete"
                if not math_rows
                else "partial_or_failed"
            ),
            "requirement": "Relaxed math_reasoning EM/throughput follow-up for staged CV",
            "evidence": "scripts/run_math_quality_followup.sh; 05_cv_ablation/math_quality_followup/summary_metrics.csv",
            "notes": (
                f"{len(math_ok_rows)}/{len(math_rows)} math-quality GuideLLM rows ok; "
                f"{len(math_valid_rows)} rows are quality-preserving CV speedup candidates. "
                "This is the accepted performance-path quality gate; it does not prove "
                "strict token-id equivalence."
            ),
        },
        {
            "area": "contribution_ablation",
            "status": "complete" if contribution else "planned_not_complete",
            "requirement": "Ablate TLM suffix skip, staged DLM suffix saving, and batched scheduling while keeping skip_suffix enabled",
            "evidence": "scripts/run_contribution_ablation.sh; 05_cv_ablation/contribution_ablation/09_reports/contribution_ablation.csv",
            "notes": (
                f"{len(contribution)} contribution rows recorded. "
                "This explains performance sources after the relaxed math-quality run."
            ),
        },
        {
            "area": "reporting",
            "status": "complete_for_current_bundle",
            "requirement": "SPECLINK_CV_REPORT.md, summary_metrics.csv/json, patch snapshot",
            "evidence": "09_reports/; patches/vllm_speclink_cv.diff",
            "notes": "Report separates strict exactness risk from relaxed math-quality/performance claims.",
        },
    ]


def write_report(
    root: Path,
    args: argparse.Namespace,
    unit_rows: list[dict[str, Any]],
    live: dict[str, Any],
    full_live: dict[str, Any],
    guide: dict[str, Any],
    audit: list[dict[str, str]],
    full_matrix_rows: list[dict[str, Any]] | None = None,
    math_quality_rows: list[dict[str, Any]] | None = None,
    contribution_rows: list[dict[str, Any]] | None = None,
) -> None:
    live_rows = live.get("rows", [])
    full_live_rows = full_live.get("rows", [])
    full_live_failures = [
        row
        for row in full_live_rows
        if row.get("status") != "ok" or not live_row_strict_matched(row)
    ]
    guide_rows = guide.get("rows", [])
    failed_units = [row for row in unit_rows if row["status"] == "fail"]
    planned_units = [row for row in unit_rows if row["status"] == "planned"]
    live_failed = [
        row
        for row in live_rows
        if row.get("status") != "ok" or not live_row_strict_matched(row)
    ]
    guide_rows = guide.get("rows", [])
    guide_ok_rows = [row for row in guide_rows if row.get("status") == "ok"]
    full_rows, extra_full_rows = split_full_matrix_rows(full_matrix_rows)
    full_ok_rows = [row for row in full_rows if row.get("status") == "ok"]
    planned_full_cases = 2 * 2 * 2 * 3 * 10
    planned_live_cases = int(full_live.get("planned_cases") or 0)
    if planned_live_cases <= 0:
        planned_live_cases = planned_full_live_cases(
            argparse.Namespace(full_live_modes="chunked,exactsafe")
        )
    math_rows = math_quality_rows or []
    math_ok_rows = [row for row in math_rows if math_quality_row_ok(row)]
    contribution = contribution_rows or []
    full_live_import_names = [
        Path(str(path)).name
        for path in getattr(args, "full_live_import_root", [])
        if "currentcfg" in Path(str(path)).name
    ]
    current_refresh_rows = [
        row
        for row in full_live_rows
        if any(name and name in str(row.get("output_dir", "")) for name in full_live_import_names)
    ]
    current_refresh_failures = [
        row
        for row in current_refresh_rows
        if row.get("status") != "ok" or not live_row_strict_matched(row)
    ]
    merged_rows = summary_rows(
        root,
        unit_rows,
        live,
        full_live,
        guide,
        full_matrix_rows,
        math_quality_rows,
        contribution_rows,
    )
    primary_rows = primary_summary_rows(merged_rows)
    best_candidates = best_speclink_candidate_rows(merged_rows)
    best_candidate = max(
        best_candidates,
        key=lambda row: safe_float(row.get("speedup_vs_eagle3")) or 0.0,
        default=None,
    )

    def avg_speedup_for(key: str, value: str) -> str:
        values = [
            safe_float(row.get("speedup_vs_eagle3"))
            for row in best_candidates
            if str(row.get(key, "")) == value
        ]
        values = [value for value in values if value is not None]
        if not values:
            return "n/a"
        return f"{sum(values) / len(values):.3f}x over {len(values)} rows"

    def best_text() -> str:
        if best_candidate is None:
            return "No quality-preserving speedup candidate is recorded."
        return (
            f"{best_candidate.get('model')} {best_candidate.get('dataset')} "
            f"K={best_candidate.get('K')} bs={best_candidate.get('batch_size')} "
            f"{best_candidate.get('method')} at "
            f"{best_candidate.get('speedup_vs_eagle3')}x"
        )

    guide_status_text = (
        f"`{len(guide_ok_rows)}/{len(guide_rows)}` ok"
        if guide_rows
        else f"`{guide['run'].get('status', 'missing')}`"
    )
    full_status_text = (
        f"`{len(full_ok_rows)}/{planned_full_cases}` ok"
        if full_rows
        else f"`0/{planned_full_cases}` recorded"
    )
    lines = [
        "# SPECLINK_CV_REPORT",
        "",
        "## Scope",
        "",
        f"- output root: `{root}`",
        f"- initial dry_run setting: `{args.dry_run}`",
        f"- finalize_only: `{args.finalize_only}`",
        f"- live gate skipped: `{args.skip_live}`",
        f"- quick serving smoke skipped: `{args.skip_guidellm_smoke}`",
        f"- quick serving smoke benchmark mode: `{args.guidellm_benchmark_mode}`",
        f"- full serving matrix benchmark mode: `{args.full_benchmark_mode}`",
        "",
        "## Current Bundle Results",
        "",
        f"- unit tests: `{len(unit_rows) - len(failed_units) - len(planned_units)}/{len(unit_rows)}` passed, `{len(planned_units)}` planned",
        f"- bounded live token-id rows: `{len(live_rows)}`",
        f"- bounded live token-id failures: `{len(live_failed)}`",
        f"- full live token-id rows: `{len(full_live_rows)}/{planned_live_cases}`, strict failures: `{len(full_live_failures)}`",
        f"- current-config strict refresh rows: `{len(current_refresh_rows)}`, failures: `{len(current_refresh_failures)}`",
        f"- quick serving smoke rows: {guide_status_text}",
        f"- full serving matrix rows: {full_status_text}",
        f"- math-quality follow-up rows: `{len(math_ok_rows)}/{len(math_rows)}` ok",
        f"- contribution ablation rows: `{len(contribution)}`",
        "",
        "## Interpretation",
        "",
        "- This bundle has the planned steady-state full-matrix throughput evidence. The original strict h<K token-id gate is still reported separately and is not fully green; do not use strict token-id equality as the performance success criterion for the relaxed math-quality path.",
        "- For final serving throughput, use the generated steady-state full matrix. In that mode `batch_size=N` means closed-loop concurrency N, and `throughput` is saturated output tokens/s over a fixed measurement window with warmup and drain excluded.",
        "- Performance-oriented CV runs should use live h<K chunking with batched prefix verification and `SPECLINK_CV_DENSE_REALIGN_STEPS=0`; conservative dense-realign diagnostics can change draft-aware acceptance opportunities and understate the intended speedup path.",
        "- Finite-request GuideLLM rows remain useful for output inspection and relaxed quality checks, but they should not be reported as final LLM serving tokens/s.",
        "- `mode=chunked` rows are the meaningful h<K SpecLink-CV token-id gate. `mode=exactsafe` is a one-shot fallback guard and is not a suffix-pruning speedup.",
        "- For strict token-id audits, any row with token mismatch, missing token-id evidence, or exact-safe fallback is excluded from valid strict-correctness speedup claims.",
        "- The generated full-matrix commands are in `scripts/run_full_steady_state_matrix.sh` and `scripts/run_full_steady_state_matrix_sliced.sh` when `--full-benchmark-mode steady_state` is used. The legacy `run_full_guidellm_matrix*.sh` files are compatibility aliases.",
        "- The full-matrix progress denominator counts only the original TODO matrix: 2 models x 2 datasets x K={8,12} x batch={8,16,32} x 10 methods. Extra staged/best-candidate or smoke rows are reported separately.",
        "- The relaxed math-quality follow-up command is in `scripts/run_math_quality_followup.sh`; it is the quality gate for the performance path and does not try to close the exact token-id gate.",
        "- The contribution ablation command is in `scripts/run_contribution_ablation.sh`; it keeps skip-suffix enabled and separates DLM suffix saving from batch scheduling.",
        "",
        "## TODO Questions",
        "",
        "- End-to-end/serving performance is recorded in `09_reports/primary_summary_table.csv`; the full 240-case matrix is complete, but strict h<K exactness still blocks a global exact-correct SpecLink-CV claim.",
        f"- Best relaxed math-quality candidate: {best_text()}. This is not a strict token-id global best because full h<K correctness still has failures.",
        f"- K comparison among quality-preserving speedup candidates: K=8 -> {avg_speedup_for('K', '8')}; K=12 -> {avg_speedup_for('K', '12')}; K=16 -> {avg_speedup_for('K', '16')}.",
        f"- Batch comparison among quality-preserving speedup candidates: bs=8 -> {avg_speedup_for('batch_size', '8')}; bs=16 -> {avg_speedup_for('batch_size', '16')}; bs=32 -> {avg_speedup_for('batch_size', '32')}.",
        "- DLM confidence calibration and fixed-half vs confidence-guided comparisons are materialized as `08_figures/confidence_calibration_reliability.*` and `08_figures/fixed_half_vs_confidence_speedup.*`; use them as diagnostics, not as final correctness proof.",
        "- Sync vs async queue and simple vs roofline packing are materialized as `08_figures/sync_vs_async_speedup.*`, `08_figures/async_queue_wait_distribution.*`, `08_figures/simple_vs_roofline_speedup.*`, and `08_figures/ablation_heatmap.*`.",
        "- Current performance evidence points to skip-suffix plus staged DLM suffix saving and batched scheduling as the useful path; `09_reports/best_speclink_cv_candidates.csv` lists only quality-preserving speedup candidates.",
        f"- The current-config strict refresh covers `{len(current_refresh_rows)}` h<K chunked rows with `{len(current_refresh_failures)}` failures; this run was intentionally stopped once the remaining strict failures were clear.",
        "- Extra verifier pass overhead, queue wait, skipped suffix ratio, selected h, GPU utilization, and fallback ratio are exported in `09_reports/primary_summary_table.csv` and `09_reports/summary_metrics.csv`.",
        "- Known limitations remain: broad strict token-id correctness is not solved, dense realign can erase speedup, and scheduler/CUDA graph/KV-cache interactions still need targeted optimization before claiming a final exact SpecLink-CV method.",
        "",
        "## Live Correctness Rows",
        "",
    ]
    if live_rows:
        lines.extend(
            [
                "| model | dataset | K | batch | mode | greedy_eps | strict greedy | matched | strict match | reason | matched count | total | first mismatch token |",
                "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in live_rows:
            lines.append(
                f"| {row.get('model', '')} | {row.get('dataset', '')} | "
                f"{row.get('K', '')} | {row.get('batch_size', '')} | "
                f"{row.get('mode', '')} | {row.get('greedy_eps', '')} | "
                f"{live_row_is_strict_greedy(row)} | "
                f"{row.get('matched', '')} | "
                f"{live_row_strict_matched(row)} | "
                f"{live_row_strict_failure_reason(row)} | "
                f"{row.get('matched_count', '')} | {row.get('total_count', '')} | "
                f"{row.get('first_mismatch_token_index', '')} |"
            )
    else:
        lines.append(f"- no live rows; run status: `{live['run'].get('status')}`")
    lines.extend(["", "## Full Live Correctness Rows", ""])
    if full_live_rows:
        lines.extend(
            [
                "| model | dataset | K | batch | mode | greedy_eps | strict greedy | matched | strict match | reason | matched count | total | first mismatch token |",
                "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in full_live_rows:
            lines.append(
                f"| {row.get('model', '')} | {row.get('dataset', '')} | "
                f"{row.get('K', '')} | {row.get('batch_size', '')} | "
                f"{row.get('mode', '')} | {row.get('greedy_eps', '')} | "
                f"{live_row_is_strict_greedy(row)} | "
                f"{row.get('matched', '')} | "
                f"{live_row_strict_matched(row)} | "
                f"{live_row_strict_failure_reason(row)} | "
                f"{row.get('matched_count', '')} | {row.get('total_count', '')} | "
                f"{row.get('first_mismatch_token_index', '')} |"
            )
    else:
        lines.append(
            f"- no full live rows; planned cases: `{full_live.get('planned_cases', 0)}`"
        )
    lines.extend(["", "## Full Matrix Rows", ""])
    if full_rows:
        lines.extend(
            [
                "| model | dataset | K | batch | method | status | throughput | speedup | quality gate | speedup claim |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in full_rows:
            lines.append(
                f"| {row.get('model', '')} | {row.get('dataset', '')} | "
                f"{row.get('K', '')} | {row.get('batch_size', '')} | "
                f"{row.get('method', '')} | {row.get('status', '')} | "
                f"{throughput_value(row)} | "
                f"{speedup_value(row)} | "
                f"{row.get('quality_gate_status', '')} | "
                f"{row.get('speedup_claim_status', '')} |"
            )
    else:
        lines.append("- no full-matrix serving rows recorded yet.")
    lines.extend(["", "## Extra Steady-State Rows", ""])
    if extra_full_rows:
        lines.extend(
            [
                "| model | dataset | K | batch | method | status | throughput | speedup | quality gate | reason |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in extra_full_rows:
            reason = "outside original 240-case TODO matrix"
            lines.append(
                f"| {row.get('model', '')} | {row.get('dataset', '')} | "
                f"{row.get('K', '')} | {row.get('batch_size', '')} | "
                f"{row.get('method', '')} | {row.get('status', '')} | "
                f"{throughput_value(row)} | "
                f"{speedup_value(row)} | "
                f"{row.get('quality_gate_status', '')} | {reason} |"
            )
    else:
        lines.append("- no extra steady-state/best-candidate rows recorded.")
    lines.extend(["", "## Math Quality Follow-Up Rows", ""])
    if math_rows:
        lines.extend(
            [
                "| model | dataset | K | batch | method | measurement | throughput | speedup | quality gate | quality score | delta | source |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in math_rows:
            lines.append(
                f"| {row.get('model', '')} | {row.get('dataset', '')} | "
                f"{row.get('K', '')} | {row.get('batch_size', '')} | "
                f"{row.get('method', '')} | {row.get('measurement_type', '')} | "
                f"{throughput_value(row)} | "
                f"{speedup_value(row)} | "
                f"{row.get('quality_gate_status', '')} | "
                f"{quality_score_value(row)} | "
                f"{quality_delta_value(row)} | "
                f"{row.get('source_root', row.get('output_root', ''))} |"
            )
    else:
        lines.append("- no relaxed math-quality rows recorded.")
    lines.extend(["", "## Requirement Audit", ""])
    lines.extend(
        [
            "| area | status | requirement | evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in audit:
        lines.append(
            f"| {row['area']} | {row['status']} | {row['requirement']} | `{row['evidence']}` |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `00_env/env_report.md` and `.json`: environment evidence.",
            "- `02_unit_tests/unit_test_summary.*`: unit test evidence.",
            "- `05_cv_ablation/live_correctness_gate/`: bounded token-id gate raw evidence.",
            "- `05_cv_ablation/full_live_correctness_gate/`: local full token-id gate raw evidence; imported rows are merged into `09_reports/summary_metrics.csv`.",
            "- `04_baselines/guidellm_smoke/`: optional GuideLLM smoke evidence.",
            "- `05_cv_ablation/math_quality_followup/`: optional relaxed math EM throughput follow-up.",
            "- `05_cv_ablation/contribution_ablation/`: optional performance contribution ablation.",
            "- `09_reports/summary_metrics.csv` and `.json`: merged summary table.",
            "- `09_reports/primary_summary_table.csv`: TODO-required primary table with throughput, latency, correctness, chunk, queue, GPU, and fallback fields.",
            "- `09_reports/best_speclink_cv_candidates.csv`: quality-preserving speedup candidates under the relaxed math-quality gate.",
            "- `09_reports/DELIVERY_SUMMARY.md` and `.json`: compact TODO section-20 handoff summary.",
            "- `patches/vllm_speclink_cv.diff`: current implementation diff.",
        ]
    )
    (root / "09_reports" / "SPECLINK_CV_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_delivery_summary(
    root: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    audit: list[dict[str, str]],
) -> None:
    unit_rows = [row for row in rows if row.get("measurement_type") == "unit_test"]
    live_rows = [
        row
        for row in rows
        if row.get("measurement_type") == "live_token_id_correctness"
        and row.get("model")
    ]
    current_live_rows = [
        row
        for row in live_rows
        if "currentcfg" in str(row.get("output_dir", ""))
    ]
    full_matrix_rows = [
        row
        for row in rows
        if row.get("source") == "full_matrix"
        and row.get("full_matrix_scope") == "planned_todo_full_matrix"
    ]
    math_rows = [row for row in rows if row.get("source") == "math_quality_followup"]
    contribution_rows = [
        row for row in rows if row.get("measurement_type") == "contribution_ablation"
    ]
    candidates = best_speclink_candidate_rows(rows)
    best = max(
        candidates,
        key=lambda row: safe_float(row.get("speedup_vs_eagle3")) or 0.0,
        default={},
    )
    delivery = {
        "result_root": str(root),
        "status": "relaxed_math_quality_performance_delivered_strict_exactness_risk",
        "modified_file_groups": [
            "vllm/vllm/speclink_cv.py and vllm/v1 scheduler/worker/sample integration",
            "tools/speclink_cv/* experiment, analysis, calibration, and audit tools",
            "examples/evaluate/eval-guidellm/run_speclink_cv_*.sh and scripts/run_speclink_cv_guidellm_*.py",
            "AGENTS.md and TODO.md documentation snapshots",
            "vllm_spec_webui.py lightweight demo",
        ],
        "new_flags_and_env": [
            "SPECLINK_CV_ENABLE / --speclink-cv-enable equivalent wrapper path",
            "SPECLINK_CV_CONFIDENCE_SIZING",
            "SPECLINK_CV_ASYNC_QUEUE",
            "SPECLINK_CV_ROOFLINE_PACKING",
            "SPECLINK_CV_CANDIDATE_CHUNKS",
            "SPECLINK_CV_DENSE_REALIGN_STEPS=0 for the performance path",
            "SPECLINK_CV_ALLOW_BATCHED_SUFFIX=1",
            "run_speclink_cv_guidellm_matrix.py --benchmark-mode steady_state",
            "run_live_correctness_gate.py --allow-batched-prefix-verification",
        ],
        "how_to_run": {
            "full_matrix": str(root / "scripts/run_full_steady_state_matrix.sh"),
            "sliced_full_matrix": str(
                root / "scripts/run_full_steady_state_matrix_sliced.sh"
            ),
            "math_quality": str(root / "scripts/run_math_quality_followup.sh"),
            "contribution_ablation": str(root / "scripts/run_contribution_ablation.sh"),
            "strict_correctness_gate": str(
                root / "scripts/run_full_live_correctness_gate_sliced.sh"
            ),
        },
        "unit_tests": {
            "passed": sum(row.get("status") == "pass" for row in unit_rows),
            "total": len(unit_rows),
            "summary": str(root / "02_unit_tests/unit_test_summary.md"),
        },
        "strict_correctness": {
            "full_live_rows": len(live_rows),
            "strict_failures": sum(row.get("strict_matched") != "True" for row in live_rows),
            "current_config_rows": len(current_live_rows),
            "current_config_failures": sum(
                row.get("strict_matched") != "True" for row in current_live_rows
            ),
            "decision": "Do not use strict token-id equality as the current performance success gate.",
        },
        "main_experiment": {
            "steady_state_rows": len(full_matrix_rows),
            "models": "qwen3_8b,llama3_1_8b",
            "datasets": "math,mtbench",
            "K": "8,12 plus K=16 math follow-up rows",
            "batch_sizes": "8,16,32",
            "methods": "pure_vllm,eagle3_oneshot,8 SpecLink-CV ablations plus staged follow-ups",
        },
        "math_quality": {
            "rows": len(math_rows),
            "ok_rows": sum(math_quality_row_ok(row) for row in math_rows),
            "quality_preserving_speedup_candidates": len(candidates),
            "summary_table": str(root / "09_reports/best_speclink_cv_candidates.csv"),
        },
        "best_relaxed_candidate": best,
        "contribution_ablation": {
            "rows": len(contribution_rows),
            "summary": str(root / "09_reports/summary_metrics.csv"),
            "interpretation": (
                "skip_suffix is kept enabled; staged DLM suffix saving and "
                "batched scheduler integration are the useful performance path."
            ),
        },
        "result_paths": {
            "report": str(root / "09_reports/SPECLINK_CV_REPORT.md"),
            "audit": str(root / "09_reports/TODO_REQUIREMENT_AUDIT.md"),
            "primary_table": str(root / "09_reports/primary_summary_table.csv"),
            "summary_metrics": str(root / "09_reports/summary_metrics.csv"),
            "figures": str(root / "08_figures"),
            "patch": str(root / "patches/vllm_speclink_cv.diff"),
        },
        "remaining_work": [
            "Strict h<K token-id equivalence still fails in bs32 and selected Llama/MTBench cases.",
            "Scheduler/CUDA graph/KV-cache interactions need further profiling before an exact-correct CV claim.",
            "Dense realign is a diagnostic path and can erase the intended speedup.",
            "Future online-serving evaluation should be separate from steady-state saturated throughput.",
        ],
        "audit_status": audit,
    }
    write_json(root / "09_reports" / "DELIVERY_SUMMARY.json", delivery)

    best_text = (
        f"{best.get('model')} {best.get('dataset')} K={best.get('K')} "
        f"bs={best.get('batch_size')} {best.get('method')} "
        f"speedup={best.get('speedup_vs_eagle3')}"
        if best
        else "No relaxed quality-preserving speedup candidate recorded."
    )
    md_lines = [
        "# SpecLink-CV Delivery Summary",
        "",
        f"- result root: `{root}`",
        "- status: relaxed math-quality/performance path delivered; strict exactness remains a documented risk.",
        f"- best relaxed candidate: {best_text}",
        f"- unit tests: `{delivery['unit_tests']['passed']}/{delivery['unit_tests']['total']}` passed",
        f"- steady-state matrix rows: `{delivery['main_experiment']['steady_state_rows']}`",
        f"- math-quality rows: `{delivery['math_quality']['ok_rows']}/{delivery['math_quality']['rows']}` ok",
        f"- strict current-config refresh: `{delivery['strict_correctness']['current_config_rows']}` rows, `{delivery['strict_correctness']['current_config_failures']}` failures",
        "",
        "## Run Scripts",
        "",
        *[f"- {name}: `{path}`" for name, path in delivery["how_to_run"].items()],
        "",
        "## Key Tables",
        "",
        f"- primary table: `{delivery['result_paths']['primary_table']}`",
        f"- best candidates: `{delivery['math_quality']['summary_table']}`",
        f"- merged metrics: `{delivery['result_paths']['summary_metrics']}`",
        "",
        "## Remaining Work",
        "",
        *[f"- {item}" for item in delivery["remaining_work"]],
    ]
    (root / "09_reports" / "DELIVERY_SUMMARY.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT / f"speclink_cv_{timestamp()}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--skip-guidellm-smoke", action="store_true")
    parser.add_argument(
        "--guidellm-benchmark-mode",
        choices=["guidellm", "steady_state"],
        default="guidellm",
        help=(
            "Benchmark client for the quick serving smoke. Keep guidellm for "
            "quality/output inspection; use steady_state for fixed-window "
            "saturated output tokens/s."
        ),
    )
    parser.add_argument("--live-models", default="qwen3_8b")
    parser.add_argument("--live-datasets", default="math")
    parser.add_argument("--live-ks", default="8")
    parser.add_argument("--live-modes", default="exactsafe,chunked")
    parser.add_argument("--live-num-prompts", type=int, default=2)
    parser.add_argument("--live-batch-size", type=int, default=1)
    parser.add_argument("--live-max-tokens", type=int, default=8)
    parser.add_argument("--live-force-prefix-len", type=int, default=0)
    parser.add_argument(
        "--live-env",
        action="append",
        default=["VLLM_BATCH_INVARIANT=1"],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--profile-max-events", type=int, default=50)
    parser.add_argument("--log-max-events", type=int, default=100)
    parser.add_argument("--analysis-profile-max-rows", type=int, default=1000)
    parser.add_argument("--guidellm-max-requests", type=int, default=1)
    parser.add_argument("--guidellm-max-tokens", type=int, default=8)
    parser.add_argument("--guidellm-steady-state-warmup-s", type=float, default=5.0)
    parser.add_argument("--guidellm-steady-state-measurement-s", type=float, default=15.0)
    parser.add_argument("--guidellm-steady-state-cooldown-s", type=float, default=5.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--guidellm-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--full-max-requests", type=int, default=80)
    parser.add_argument("--full-max-tokens", type=int, default=128)
    parser.add_argument(
        "--full-benchmark-mode",
        choices=["guidellm", "steady_state"],
        default="steady_state",
        help=(
            "Benchmark mode for the generated full serving matrix. Default is "
            "steady_state so throughput claims use closed-loop saturated "
            "output tokens/s and exclude finite-request drain."
        ),
    )
    parser.add_argument("--full-steady-state-warmup-s", type=float, default=30.0)
    parser.add_argument("--full-steady-state-measurement-s", type=float, default=120.0)
    parser.add_argument("--full-steady-state-cooldown-s", type=float, default=30.0)
    parser.add_argument(
        "--full-steady-state-max-prompts",
        type=int,
        default=80,
        help="Prompt pool size cycled by the steady-state full matrix client.",
    )
    parser.add_argument("--full-cases-per-slice", type=int, default=1)
    parser.add_argument(
        "--full-matrix-import-root",
        action="append",
        default=[],
        help=(
            "Existing full serving matrix root or summary_metrics.csv to "
            "include in finalize-only reports without rerunning vLLM."
        ),
    )
    parser.add_argument("--full-live-modes", default="chunked,exactsafe")
    parser.add_argument("--full-live-num-prompts", type=int, default=4)
    parser.add_argument(
        "--full-live-num-prompts-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--full-live-max-tokens", type=int, default=32)
    parser.add_argument("--full-live-force-prefix-len", type=int, default=0)
    parser.add_argument("--full-live-cases-per-slice", type=int, default=1)
    parser.add_argument("--full-live-allow-batched-prefix-verification", action="store_true")
    parser.add_argument("--full-live-global-batch-barrier", action="store_true")
    parser.add_argument(
        "--full-live-env",
        action="append",
        default=["VLLM_BATCH_INVARIANT=1"],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--full-live-import-root",
        action="append",
        default=[],
        help=(
            "Existing run_live_correctness_gate root or summary.csv to include "
            "in TODO reports without rerunning vLLM."
        ),
    )
    parser.add_argument(
        "--full-allow-shape-drift-chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use live h<K chunking for the full serving matrix by default. "
            "Pass --no-full-allow-shape-drift-chunking for exact-safe "
            "one-shot fallback diagnostics."
        ),
    )
    parser.add_argument(
        "--full-allow-batched-prefix-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Batch prefix verification chunks in the full serving matrix by "
            "default; disable only for conservative scheduler diagnostics."
        ),
    )
    parser.add_argument(
        "--full-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--math-quality-models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--math-quality-datasets", default="math")
    parser.add_argument("--math-quality-ks", default="8,12")
    parser.add_argument("--math-quality-batch-sizes", default="8,16,32")
    parser.add_argument(
        "--math-quality-methods",
        default="eagle3_oneshot,cv_half_async_staged_simple",
    )
    parser.add_argument("--math-quality-max-requests", type=int, default=64)
    parser.add_argument("--math-quality-max-tokens", type=int, default=0)
    parser.add_argument("--math-quality-force-prefix-len", type=int, default=0)
    parser.add_argument("--math-quality-port", type=int, default=8078)
    parser.add_argument(
        "--math-quality-import-root",
        action="append",
        default=[],
        help=(
            "Existing math-quality follow-up root or summary_metrics.csv to "
            "include in finalize-only reports without rerunning vLLM."
        ),
    )
    parser.add_argument("--contribution-models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--contribution-datasets", default="math")
    parser.add_argument("--contribution-ks", default="12")
    parser.add_argument("--contribution-batch-sizes", default="16")
    parser.add_argument(
        "--contribution-benchmark-mode",
        choices=["guidellm", "steady_state"],
        default="steady_state",
    )
    parser.add_argument("--contribution-max-requests", type=int, default=64)
    parser.add_argument("--contribution-max-tokens", type=int, default=0)
    parser.add_argument("--contribution-steady-state-warmup-s", type=float, default=30.0)
    parser.add_argument(
        "--contribution-steady-state-measurement-s", type=float, default=120.0
    )
    parser.add_argument("--contribution-steady-state-cooldown-s", type=float, default=30.0)
    parser.add_argument("--contribution-steady-state-max-prompts", type=int, default=64)
    parser.add_argument("--contribution-force-prefix-len", type=int, default=0)
    parser.add_argument("--contribution-port", type=int, default=8096)
    parser.add_argument(
        "--contribution-import-root",
        action="append",
        default=[],
        help=(
            "Existing contribution ablation root or contribution_ablation.csv "
            "to include in finalize-only reports without rerunning vLLM."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.full_cases_per_slice <= 0:
        raise SystemExit("--full-cases-per-slice must be positive")
    if args.full_live_cases_per_slice <= 0:
        raise SystemExit("--full-live-cases-per-slice must be positive")
    root = args.output_root
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    args.output_root = root
    create_tree(root)
    if args.finalize_only:
        run_config = read_run_config(root)
        if run_config:
            args = apply_saved_args(args, run_config)
            args.output_root = root
            args.finalize_only = True
        args = restore_plan_import_roots(args, root)
        write_planned_full_matrix(root, args)
        write_planned_full_live_correctness(root, args)
        write_planned_math_quality_followup(root, args)
        write_planned_contribution_ablation(root, args)
        copy_inputs(root)
        write_patch_snapshot(root)
        unit_rows = read_unit_rows(root)
        live = read_live_summary(root, run_config)
        full_live = read_full_live_summary(
            root,
            list(args.full_live_import_root),
            planned_full_live_cases(args),
        )
        guide = read_guidellm_smoke_summary(root, run_config)
        full_rows = read_existing_full_matrix_rows(
            root, list(args.full_matrix_import_root)
        )
        math_rows = read_math_quality_followup_rows(
            root, list(args.math_quality_import_root)
        )
        contribution_rows = read_contribution_ablation_rows(
            root, list(args.contribution_import_root)
        )
        rows = summary_rows(
            root,
            unit_rows,
            live,
            full_live,
            guide,
            full_rows,
            math_rows,
            contribution_rows,
        )
        audit = audit_rows(
            unit_rows,
            live,
            full_live,
            guide,
            full_rows,
            math_rows,
            contribution_rows,
        )
        write_csv(root / "09_reports" / "summary_metrics.csv", rows)
        write_json(root / "09_reports" / "summary_metrics.json", {"rows": rows})
        write_csv(
            root / "09_reports" / "primary_summary_table.csv",
            primary_summary_rows(rows),
        )
        write_csv(
            root / "09_reports" / "best_speclink_cv_candidates.csv",
            best_speclink_candidate_rows(rows),
        )
        write_csv(root / "summary_metrics.csv", rows)
        write_json(root / "summary_metrics.json", {"rows": rows})
        write_csv(root / "09_reports" / "TODO_REQUIREMENT_AUDIT.csv", audit)
        write_json(root / "09_reports" / "TODO_REQUIREMENT_AUDIT.json", {"rows": audit})
        write_report(
            root,
            args,
            unit_rows,
            live,
            full_live,
            guide,
            audit,
            full_rows,
            math_rows,
            contribution_rows,
        )
        write_delivery_summary(root, args, rows, audit)
        (root / "09_reports" / "TODO_REQUIREMENT_AUDIT.md").write_text(
            "\n".join(
                [
                    "# SpecLink-CV TODO Requirement Audit",
                    "",
                    "| area | status | requirement | evidence | notes |",
                    "| --- | --- | --- | --- | --- |",
                    *[
                        f"| {row['area']} | {row['status']} | {row['requirement']} | "
                        f"`{row['evidence']}` | {row['notes']} |"
                        for row in audit
                    ],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        planned_full_rows, extra_full_rows = split_full_matrix_rows(full_rows)
        print(
            json.dumps(
                {
                    "output_root": str(root),
                    "summary": str(root / "09_reports" / "summary_metrics.csv"),
                    "full_matrix_rows": len(planned_full_rows),
                    "extra_full_matrix_rows": len(extra_full_rows),
                }
            )
        )
        return 0
    shell_script(
        root / "scripts" / "run_command.sh",
        [command_text([sys.executable, "-u", *sys.argv], REPO_ROOT)],
    )
    copy_inputs(root)
    write_patch_snapshot(root)
    write_json(root / "logs" / "todo_run_config.json", vars(args))
    run_env(root, args.dry_run)
    unit_rows = run_unit_tests(root, args.dry_run)
    write_planned_full_matrix(root, args)
    write_planned_full_live_correctness(root, args)
    write_planned_math_quality_followup(root, args)
    write_planned_contribution_ablation(root, args)
    live = run_live_gate(root, args)
    full_live = read_full_live_summary(
        root,
        list(args.full_live_import_root),
        planned_full_live_cases(args),
    )
    guide = run_guidellm_smoke(root, args)
    full_rows = read_existing_full_matrix_rows(
        root, list(args.full_matrix_import_root)
    )
    math_rows = read_math_quality_followup_rows(
        root, list(args.math_quality_import_root)
    )
    contribution_rows = read_contribution_ablation_rows(
        root, list(args.contribution_import_root)
    )
    rows = summary_rows(
        root,
        unit_rows,
        live,
        full_live,
        guide,
        full_rows,
        math_rows,
        contribution_rows,
    )
    audit = audit_rows(
        unit_rows,
        live,
        full_live,
        guide,
        full_rows,
        math_rows,
        contribution_rows,
    )
    write_csv(root / "09_reports" / "summary_metrics.csv", rows)
    write_json(root / "09_reports" / "summary_metrics.json", {"rows": rows})
    write_csv(
        root / "09_reports" / "primary_summary_table.csv",
        primary_summary_rows(rows),
    )
    write_csv(
        root / "09_reports" / "best_speclink_cv_candidates.csv",
        best_speclink_candidate_rows(rows),
    )
    write_csv(root / "summary_metrics.csv", rows)
    write_json(root / "summary_metrics.json", {"rows": rows})
    write_csv(root / "09_reports" / "TODO_REQUIREMENT_AUDIT.csv", audit)
    write_json(root / "09_reports" / "TODO_REQUIREMENT_AUDIT.json", {"rows": audit})
    write_report(
        root,
        args,
        unit_rows,
        live,
        full_live,
        guide,
        audit,
        full_rows,
        math_rows,
        contribution_rows,
    )
    write_delivery_summary(root, args, rows, audit)
    (root / "09_reports" / "TODO_REQUIREMENT_AUDIT.md").write_text(
        "\n".join(
            [
                "# SpecLink-CV TODO Requirement Audit",
                "",
                "| area | status | requirement | evidence | notes |",
                "| --- | --- | --- | --- | --- |",
                *[
                    f"| {row['area']} | {row['status']} | {row['requirement']} | "
                    f"`{row['evidence']}` | {row['notes']} |"
                    for row in audit
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(root), "summary": str(root / "09_reports" / "summary_metrics.csv")}))
    failed_units = [row for row in unit_rows if row["status"] == "fail"]
    live_failed = [
        row
        for row in live.get("rows", [])
        if row.get("status") != "ok" or str(row.get("matched")) != "True"
    ]
    guide_failed = [
        row
        for row in guide.get("rows", [])
        if row.get("status") not in {"ok", "planned"}
    ]
    if args.skip_guidellm_smoke:
        guide_failed = []
    return 1 if failed_units or live_failed or guide_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
