#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

METRIC_PRIORITY = [
    "exact_match,get_response",
    "exact_match,none",
    "exact_match",
    "acc,none",
    "acc",
    "acc_norm,none",
    "pass@1,create_test",
    "pass@1",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_result_json(run_dir: Path) -> Path | None:
    candidates = [
        path
        for path in (run_dir / "lm_eval_output").rglob("*.json")
        if path.name != "run_meta.json"
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_samples_jsonl(run_dir: Path, task_name: str) -> Path | None:
    candidates = list((run_dir / "lm_eval_output").rglob(f"samples_{task_name}_*.jsonl"))
    if not candidates:
        candidates = list((run_dir / "lm_eval_output").rglob("samples_*.jsonl"))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def choose_metric(task_result: dict[str, Any]) -> tuple[str, float | None]:
    for key in METRIC_PRIORITY:
        if key in task_result and not key.endswith("_stderr"):
            try:
                return key, float(task_result[key])
            except (TypeError, ValueError):
                return key, None
    for key, value in task_result.items():
        if key.endswith("_stderr") or key in {"alias", "samples"}:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def sample_count(data: dict[str, Any], task: str) -> int | None:
    for section in ("n-samples", "n_samples"):
        values = data.get(section)
        if isinstance(values, dict):
            raw = values.get(task)
            if isinstance(raw, dict):
                raw = raw.get("effective") or raw.get("original")
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return None


def sample_key(sample: dict[str, Any]) -> str:
    for key in ("doc_id", "doc_hash", "prompt_hash"):
        if key in sample:
            return f"{key}:{sample[key]}"
    return json.dumps(sample.get("doc", sample), sort_keys=True, ensure_ascii=False)


def sample_correct(sample: dict[str, Any], metric: str) -> bool | None:
    metric_name = metric.split(",", 1)[0] if metric else ""
    candidates = [metric_name, "exact_match", "acc", "acc_norm", "pass@1"]
    for key in candidates:
        if key and key in sample:
            value = sample[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value) > 0.5
    return None


def load_sample_map(run_dir: Path, task_name: str, metric: str) -> dict[str, dict[str, Any]]:
    sample_path = find_samples_jsonl(run_dir, task_name)
    if sample_path is None:
        return {}
    samples: dict[str, dict[str, Any]] = {}
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            correct = sample_correct(sample, metric)
            samples[sample_key(sample)] = {
                "correct": correct,
                "doc_id": sample.get("doc_id"),
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
                "target": sample.get("target"),
                "filtered_resps": sample.get("filtered_resps"),
                "resps": sample.get("resps"),
                "sample_path": str(sample_path),
            }
    return samples


def rows_from_runs(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(output_dir.rglob("run_meta.json")):
        run_dir = meta_path.parent
        meta = load_json(meta_path)
        result_path = find_result_json(run_dir)
        status = meta.get("status", "")
        if not result_path:
            rows.append({**meta, "metric": "", "score": "", "samples": "", "result_path": "", "status": status or "missing_result"})
            continue
        data = load_json(result_path)
        results = data.get("results", {})
        for task_name, task_result in results.items():
            if not isinstance(task_result, dict):
                continue
            metric, score = choose_metric(task_result)
            rows.append(
                {
                    **meta,
                    "run_dir": str(run_dir),
                    "task_result_name": task_name,
                    "metric": metric,
                    "score": score,
                    "samples": sample_count(data, task_name),
                    "result_path": str(result_path),
                    "status": status,
                }
            )
    for skip_path in sorted(output_dir.rglob("skip.json")):
        skip = load_json(skip_path)
        rows.append(
            {
                "model_label": skip.get("model_label"),
                "mode": skip.get("mode"),
                "mode_group": "",
                "task": skip.get("task"),
                "task_result_name": skip.get("task"),
                "metric": "",
                "score": "",
                "samples": "",
                "status": "skipped",
                "error": skip.get("reason", ""),
                "result_path": "",
            }
        )
    return rows


def add_dense_comparisons(rows: list[dict[str, Any]]) -> None:
    dense_scores: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("mode") != "dense_ar":
            continue
        score = row.get("score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            dense_scores[(str(row.get("model_label")), str(row.get("task_result_name")))] = float(score)
    for row in rows:
        score = row.get("score")
        dense = dense_scores.get((str(row.get("model_label")), str(row.get("task_result_name"))))
        row["dense_score"] = dense
        if isinstance(score, (int, float)) and dense is not None:
            row["delta_pp_vs_dense"] = (float(score) - dense) * 100.0
        else:
            row["delta_pp_vs_dense"] = ""


def add_paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dense_rows: dict[tuple[str, str], dict[str, Any]] = {}
    sample_maps: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}

    for row in rows:
        if row.get("mode") == "dense_ar" and row.get("status") == "ok":
            key = (str(row.get("model_label")), str(row.get("task_result_name")))
            dense_rows[key] = row

    paired_events: list[dict[str, Any]] = []
    for row in rows:
        task_name = str(row.get("task_result_name") or row.get("task") or "")
        model_label = str(row.get("model_label") or "")
        mode = str(row.get("mode") or "")
        run_dir = Path(str(row.get("run_dir") or ""))
        metric = str(row.get("metric") or "")
        row_key = (model_label, task_name)
        sample_key_tuple = (model_label, mode, task_name)
        if row.get("status") != "ok" or not run_dir:
            continue
        sample_maps[sample_key_tuple] = load_sample_map(run_dir, task_name, metric)
        dense_row = dense_rows.get(row_key)
        if dense_row is None:
            continue
        dense_mode = str(dense_row.get("mode") or "")
        dense_tuple = (model_label, dense_mode, task_name)
        if dense_tuple not in sample_maps:
            dense_maps_run_dir = Path(str(dense_row.get("run_dir") or ""))
            sample_maps[dense_tuple] = load_sample_map(
                dense_maps_run_dir, task_name, str(dense_row.get("metric") or "")
            )
        dense_samples = sample_maps.get(dense_tuple, {})
        experiment_samples = sample_maps.get(sample_key_tuple, {})
        common_keys = sorted(set(dense_samples) & set(experiment_samples))

        counts = {
            "paired_samples": 0,
            "dense_correct": 0,
            "experimental_correct": 0,
            "both_correct": 0,
            "dense_correct_experimental_wrong": 0,
            "dense_wrong_experimental_correct": 0,
            "both_wrong": 0,
        }
        for key in common_keys:
            dense_correct = dense_samples[key].get("correct")
            experiment_correct = experiment_samples[key].get("correct")
            if dense_correct is None or experiment_correct is None:
                continue
            counts["paired_samples"] += 1
            counts["dense_correct"] += int(bool(dense_correct))
            counts["experimental_correct"] += int(bool(experiment_correct))
            if dense_correct and experiment_correct:
                counts["both_correct"] += 1
            elif dense_correct and not experiment_correct:
                counts["dense_correct_experimental_wrong"] += 1
                paired_events.append(
                    {
                        "event": "dense_correct_experimental_wrong",
                        "model_label": model_label,
                        "task": task_name,
                        "mode": mode,
                        "sample_key": key,
                        "dense": dense_samples[key],
                        "experimental": experiment_samples[key],
                    }
                )
            elif not dense_correct and experiment_correct:
                counts["dense_wrong_experimental_correct"] += 1
                paired_events.append(
                    {
                        "event": "dense_wrong_experimental_correct",
                        "model_label": model_label,
                        "task": task_name,
                        "mode": mode,
                        "sample_key": key,
                        "dense": dense_samples[key],
                        "experimental": experiment_samples[key],
                    }
                )
            else:
                counts["both_wrong"] += 1

        row.update(counts)
        dense_correct_count = counts["dense_correct"]
        if dense_correct_count:
            row["dense_correct_retention"] = counts["both_correct"] / dense_correct_count
        else:
            row["dense_correct_retention"] = ""

    return paired_events


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_label",
        "mode",
        "mode_group",
        "task",
        "task_result_name",
        "metric",
        "dense_score",
        "score",
        "delta_pp_vs_dense",
        "samples",
        "paired_samples",
        "dense_correct",
        "experimental_correct",
        "dense_correct_experimental_wrong",
        "dense_wrong_experimental_correct",
        "both_wrong",
        "dense_correct_retention",
        "status",
        "spec_acceptance_rate",
        "spec_accepted_tokens",
        "spec_draft_tokens",
        "result_path",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return ""
        return str(value)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# lm-eval Accuracy Report\n\n")
        handle.write("Modes are dense target-only, dense EAGLE3, activation-aware, and token_dense_t00-t10.\n\n")
        handle.write("| Task | Protocol | Model | Mode | Dense | Experimental | Delta pp | Dense->Wrong | Retention | Samples | Status |\n")
        handle.write("|------|----------|-------|------|------:|-------------:|---------:|------------:|----------:|--------:|--------|\n")
        for row in sorted(rows, key=lambda item: (str(item.get("task_result_name")), str(item.get("model_label")), str(item.get("mode")))):
            protocol = "generative" if "generative" in str(row.get("task")) or str(row.get("metric")).startswith("exact_match") else "official"
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row.get("task_result_name") or row.get("task") or ""),
                        protocol,
                        str(row.get("model_label") or ""),
                        str(row.get("mode") or ""),
                        fmt(row.get("dense_score")),
                        fmt(row.get("score")),
                        fmt(row.get("delta_pp_vs_dense")),
                        fmt(row.get("dense_correct_experimental_wrong")),
                        fmt(row.get("dense_correct_retention")),
                        fmt(row.get("samples")),
                        str(row.get("status") or ""),
                    ]
                )
                + " |\n"
            )
        skipped = [row for row in rows if row.get("status") == "skipped"]
        if skipped:
            handle.write("\n## Skipped\n\n")
            for row in skipped:
                handle.write(f"- {row.get('model_label')} / {row.get('mode')} / {row.get('task')}: {row.get('error')}\n")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    rows = rows_from_runs(output_dir)
    add_dense_comparisons(rows)
    paired_events = add_paired_comparisons(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "paired_regressions.jsonl").open("w", encoding="utf-8") as handle:
        for event in paired_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_report(output_dir / "report.md", rows)
    print(output_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
