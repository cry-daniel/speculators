#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

METRIC_PRIORITY = [
    "exact_match,get_response",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match,none",
    "exact_match",
    "acc,none",
    "acc",
    "acc_norm,none",
    "pass@1,create_test",
    "pass@1",
]
REQUEST_PROGRESS_RE = re.compile(
    r"Requesting API:\s+100%.*?\|\s*(\d+)/(\d+)\s*"
    r"\[[^\]]*,\s*([0-9.]+)(s/it|it/s)\]"
)
_TOKENIZER_CACHE: dict[str, Any] = {}
COMPARISON_CONFIG_FIELDS = (
    "num_spec_tokens",
    "batch_size",
    "num_concurrent",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_context_length",
    "max_new_tokens",
    "dtype",
    "seed",
    "limit",
)


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


def request_elapsed_seconds(run_dir: Path) -> float | None:
    log_path = run_dir / "lm_eval.log"
    if not log_path.is_file():
        return None
    matches = REQUEST_PROGRESS_RE.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        return None
    completed, total, rate, unit = matches[-1]
    iterations = int(completed)
    if iterations != int(total) or iterations <= 0:
        return None
    rate_value = float(rate)
    if rate_value <= 0:
        return None
    return iterations * rate_value if unit == "s/it" else iterations / rate_value


def _sample_response_text(sample: dict[str, Any]) -> str | None:
    resps = sample.get("resps")
    if not isinstance(resps, list) or not resps:
        return None
    first = resps[0]
    if isinstance(first, list) and first:
        first = first[0]
    return first if isinstance(first, str) else None


def output_token_count(
    run_dir: Path,
    task_name: str,
    tokenizer_path: str,
) -> int | None:
    sample_path = find_samples_jsonl(run_dir, task_name)
    if sample_path is None or not tokenizer_path:
        return None
    texts: list[str] = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = _sample_response_text(json.loads(line))
            if text is not None:
                texts.append(text)
    if not texts:
        return 0
    try:
        tokenizer = _TOKENIZER_CACHE.get(tokenizer_path)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            _TOKENIZER_CACHE[tokenizer_path] = tokenizer
        tokenized = tokenizer(texts, add_special_tokens=False)
        return sum(len(token_ids) for token_ids in tokenized["input_ids"])
    except Exception:
        return None


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
    run_config_path = output_dir / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.is_file() else {}
    shared_config = {
        key: run_config[key]
        for key in (
            "token_dense_linear_strategy",
            "token_dense_mlp_strategy",
            "token_dense_projection_policy",
            "token_dense_mixed_projection_policy",
            "token_dense_mixed_layers",
            "token_dense_mlp_static_layers",
            "token_dense_gate_up_dense_layers",
            "token_dense_down_dense_layers",
            "token_dense_attention_dense_layers",
            "token_dense_dense_layers",
            "token_dense_dense_ratio",
            "token_dense_dense_min_per_request",
            "token_dense_dense_cap",
            "token_dense_dense_selection",
            "token_dense_balanced_start_position",
            "token_dense_sparse_value_scale",
            "token_dense_gate_up_value_scale",
            "token_dense_gate_up_hybrid",
            "token_dense_group_reconstruction",
            "token_dense_row_scale_mode",
            "token_dense_row_scale_max",
            "token_dense_sparse_output_mode",
            "token_dense_sparse_accumulator",
            "token_dense_mask_method",
            *COMPARISON_CONFIG_FIELDS,
        )
        if key in run_config
    }
    for meta_path in sorted(output_dir.rglob("run_meta.json")):
        run_dir = meta_path.parent
        meta = {**shared_config, **load_json(meta_path)}
        result_path = find_result_json(run_dir)
        status = meta.get("status", "")
        if not result_path:
            rows.append({**meta, "metric": "", "score": "", "samples": "", "result_path": "", "status": status or "missing_result"})
            continue
        data = load_json(result_path)
        results = data.get("results", {})
        elapsed_seconds = request_elapsed_seconds(run_dir)
        for task_name, task_result in results.items():
            if not isinstance(task_result, dict):
                continue
            metric, score = choose_metric(task_result)
            output_tokens = output_token_count(
                run_dir,
                task_name,
                str(meta.get("tokenizer_path") or meta.get("model_path") or ""),
            )
            request_tok_s = (
                output_tokens / elapsed_seconds
                if output_tokens is not None
                and elapsed_seconds is not None
                and elapsed_seconds > 0
                else None
            )
            rows.append(
                {
                    **meta,
                    "run_dir": str(run_dir),
                    "task_result_name": task_name,
                    "metric": metric,
                    "score": score,
                    "samples": sample_count(data, task_name),
                    "request_elapsed_seconds": elapsed_seconds,
                    "output_tokens": output_tokens,
                    "request_output_tokens_per_second": request_tok_s,
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


def add_external_baselines(
    rows: list[dict[str, Any]],
    baseline_dir: Path | None,
) -> None:
    if baseline_dir is None:
        return
    existing = {
        comparison_key(row)
        for row in rows
        if row.get("mode") == "eagle3_dense"
    }
    for row in rows_from_runs(baseline_dir.resolve()):
        key = comparison_key(row)
        if row.get("mode") == "eagle3_dense" and key not in existing:
            rows.append(row)
            existing.add(key)


def comparison_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Identify an apples-to-apples dense EAGLE3 comparison point."""

    return (
        str(row.get("model_label", "")),
        str(row.get("task_result_name", "")),
        *(str(row.get(field, "")) for field in COMPARISON_CONFIG_FIELDS),
    )


def add_accuracy_comparisons(rows: list[dict[str, Any]]) -> None:
    baseline_scores: dict[tuple[str, ...], float] = {}
    for row in rows:
        if row.get("mode") != "eagle3_dense":
            continue
        score = row.get("score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            baseline_scores[comparison_key(row)] = float(score)
    for row in rows:
        score = row.get("score")
        baseline = baseline_scores.get(comparison_key(row))
        row["eagle3_dense_score"] = baseline
        if isinstance(score, (int, float)) and baseline is not None:
            row["delta_pp_vs_eagle3_dense"] = (
                float(score) - baseline
            ) * 100.0
        else:
            row["delta_pp_vs_eagle3_dense"] = ""


def add_throughput_comparisons(rows: list[dict[str, Any]]) -> None:
    dense_throughput: dict[tuple[str, ...], float] = {}
    for row in rows:
        if row.get("mode") != "eagle3_dense":
            continue
        throughput = row.get("request_output_tokens_per_second")
        if isinstance(throughput, (int, float)) and math.isfinite(float(throughput)):
            dense_throughput[comparison_key(row)] = float(throughput)
    for row in rows:
        baseline = dense_throughput.get(comparison_key(row))
        throughput = row.get("request_output_tokens_per_second")
        row["eagle3_dense_request_tok_s"] = baseline
        if (
            baseline is not None
            and baseline > 0
            and isinstance(throughput, (int, float))
        ):
            row["speedup_vs_eagle3_dense"] = float(throughput) / baseline
        else:
            row["speedup_vs_eagle3_dense"] = ""


def add_goal_checks(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        delta_pp = row.get("delta_pp_vs_eagle3_dense")
        speedup = row.get("speedup_vs_eagle3_dense")
        accuracy_ok = (
            float(delta_pp) >= -5.0
            if isinstance(delta_pp, (int, float))
            else ""
        )
        speedup_13 = (
            float(speedup) >= 1.3
            if isinstance(speedup, (int, float))
            else ""
        )
        speedup_14 = (
            float(speedup) >= 1.4
            if isinstance(speedup, (int, float))
            else ""
        )
        row["accuracy_within_5pp"] = accuracy_ok
        row["speedup_at_least_1_3x"] = speedup_13
        row["speedup_at_least_1_4x"] = speedup_14
        row["meets_hard_goal"] = (
            bool(accuracy_ok and speedup_13)
            if accuracy_ok != "" and speedup_13 != ""
            else ""
        )
        row["meets_target_goal"] = (
            bool(accuracy_ok and speedup_14)
            if accuracy_ok != "" and speedup_14 != ""
            else ""
        )


def add_paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_rows: dict[tuple[str, str], dict[str, Any]] = {}
    sample_maps: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}

    for row in rows:
        if row.get("mode") == "eagle3_dense" and row.get("status") == "ok":
            key = (str(row.get("model_label")), str(row.get("task_result_name")))
            baseline_rows[key] = row

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
        baseline_row = baseline_rows.get(row_key)
        if baseline_row is None:
            continue
        baseline_mode = str(baseline_row.get("mode") or "")
        baseline_tuple = (model_label, baseline_mode, task_name)
        if baseline_tuple not in sample_maps:
            baseline_run_dir = Path(str(baseline_row.get("run_dir") or ""))
            sample_maps[baseline_tuple] = load_sample_map(
                baseline_run_dir,
                task_name,
                str(baseline_row.get("metric") or ""),
            )
        baseline_samples = sample_maps.get(baseline_tuple, {})
        experiment_samples = sample_maps.get(sample_key_tuple, {})
        common_keys = sorted(set(baseline_samples) & set(experiment_samples))

        counts = {
            "paired_samples": 0,
            "eagle3_dense_correct": 0,
            "experimental_correct": 0,
            "both_correct": 0,
            "eagle3_dense_correct_experimental_wrong": 0,
            "eagle3_dense_wrong_experimental_correct": 0,
            "both_wrong": 0,
        }
        for key in common_keys:
            baseline_correct = baseline_samples[key].get("correct")
            experiment_correct = experiment_samples[key].get("correct")
            if baseline_correct is None or experiment_correct is None:
                continue
            counts["paired_samples"] += 1
            counts["eagle3_dense_correct"] += int(bool(baseline_correct))
            counts["experimental_correct"] += int(bool(experiment_correct))
            if baseline_correct and experiment_correct:
                counts["both_correct"] += 1
            elif baseline_correct and not experiment_correct:
                counts["eagle3_dense_correct_experimental_wrong"] += 1
                paired_events.append(
                    {
                        "event": "eagle3_dense_correct_experimental_wrong",
                        "model_label": model_label,
                        "task": task_name,
                        "mode": mode,
                        "sample_key": key,
                        "eagle3_dense": baseline_samples[key],
                        "experimental": experiment_samples[key],
                    }
                )
            elif not baseline_correct and experiment_correct:
                counts["eagle3_dense_wrong_experimental_correct"] += 1
                paired_events.append(
                    {
                        "event": "eagle3_dense_wrong_experimental_correct",
                        "model_label": model_label,
                        "task": task_name,
                        "mode": mode,
                        "sample_key": key,
                        "eagle3_dense": baseline_samples[key],
                        "experimental": experiment_samples[key],
                    }
                )
            else:
                counts["both_wrong"] += 1

        row.update(counts)
        baseline_correct_count = counts["eagle3_dense_correct"]
        if baseline_correct_count:
            row["eagle3_dense_correct_retention"] = (
                counts["both_correct"] / baseline_correct_count
            )
        else:
            row["eagle3_dense_correct_retention"] = ""

    return paired_events


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_label",
        "mode",
        "mode_group",
        "task",
        "task_result_name",
        "metric",
        "token_dense_linear_strategy",
        "token_dense_mlp_strategy",
        "token_dense_projection_policy",
        "token_dense_gate_up_dense_layers",
        "token_dense_down_dense_layers",
        "token_dense_attention_dense_layers",
        "token_dense_dense_layers",
        "token_dense_sparse_value_scale",
        "token_dense_gate_up_value_scale",
        "token_dense_gate_up_hybrid",
        "token_dense_group_reconstruction",
        "token_dense_row_scale_mode",
        "token_dense_row_scale_max",
        "token_dense_sparse_output_mode",
        "token_dense_sparse_accumulator",
        "token_dense_mask_method",
        "eagle3_dense_score",
        "score",
        "delta_pp_vs_eagle3_dense",
        "samples",
        "request_elapsed_seconds",
        "output_tokens",
        "request_output_tokens_per_second",
        "eagle3_dense_request_tok_s",
        "speedup_vs_eagle3_dense",
        "accuracy_within_5pp",
        "speedup_at_least_1_3x",
        "speedup_at_least_1_4x",
        "meets_hard_goal",
        "meets_target_goal",
        "paired_samples",
        "eagle3_dense_correct",
        "experimental_correct",
        "eagle3_dense_correct_experimental_wrong",
        "eagle3_dense_wrong_experimental_correct",
        "both_wrong",
        "eagle3_dense_correct_retention",
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
        handle.write(
            "Modes are dense target-only, dense EAGLE3, activation-aware, and "
            "fixed-budget token_dense_dN. Accuracy deltas and throughput "
            "speedups use dense EAGLE3 as the baseline.\n\n"
        )
        handle.write("| Task | Protocol | Model | Mode | Score | Delta pp | Request tok/s | Speedup | >=1.3x and <=5pp | >=1.4x and <=5pp | EAGLE3->Wrong | Retention | Samples | Status |\n")
        handle.write("|------|----------|-------|------|------:|---------:|--------------:|--------:|----------------:|----------------:|------------:|----------:|--------:|--------|\n")
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
                        fmt(row.get("score")),
                        fmt(row.get("delta_pp_vs_eagle3_dense")),
                        fmt(row.get("request_output_tokens_per_second")),
                        fmt(row.get("speedup_vs_eagle3_dense")),
                        fmt(row.get("meets_hard_goal")),
                        fmt(row.get("meets_target_goal")),
                        fmt(
                            row.get(
                                "eagle3_dense_correct_experimental_wrong"
                            )
                        ),
                        fmt(row.get("eagle3_dense_correct_retention")),
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


def token_dense_budget(mode: str) -> int | None:
    prefix = "token_dense_d"
    if not mode.startswith(prefix):
        return None
    raw = mode.removeprefix(prefix)
    if not raw.isdigit():
        return None
    value = int(raw)
    return value if value in {0, 8, 16, 32, 64, 128, 256} else None


def write_token_dense_accuracy_plot(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    plot_rows: list[tuple[str, int, float]] = []
    for row in rows:
        task_name = str(row.get("task_result_name") or row.get("task") or "")
        if task_name != "gsm8k_cot" or row.get("status") != "ok":
            continue
        budget = token_dense_budget(str(row.get("mode") or ""))
        score = row.get("score")
        if budget is None or not isinstance(score, (int, float)):
            continue
        score_value = float(score)
        if not math.isfinite(score_value):
            continue
        plot_rows.append((str(row.get("model_label") or ""), budget, score_value))
    if not plot_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for model_label in sorted({item[0] for item in plot_rows}):
        points = sorted(
            (item for item in plot_rows if item[0] == model_label),
            key=lambda item: item[1],
        )
        ax.plot(
            [item[1] for item in points],
            [item[2] for item in points],
            marker="o",
            linewidth=1.8,
            label=model_label,
        )
    ax.set_title("GSM8K CoT token_dense accuracy")
    ax.set_xlabel("Dense draft-token budget")
    ax.set_ylabel("Exact match")
    ax.set_xticks(sorted({item[1] for item in plot_rows}))
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "gsm8k_cot_token_dense_accuracy.png", dpi=200)
    plt.close(fig)


def write_quality_throughput_plot(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    plot_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and isinstance(row.get("delta_pp_vs_eagle3_dense"), (int, float))
        and isinstance(row.get("speedup_vs_eagle3_dense"), (int, float))
    ]
    tasks = sorted(
        {
            str(row.get("task_result_name") or row.get("task") or "")
            for row in plot_rows
        }
    )
    if not tasks:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(6.0 * len(tasks), 4.8),
        squeeze=False,
    )
    for task, ax in zip(tasks, axes[0]):
        task_rows = [
            row
            for row in plot_rows
            if str(row.get("task_result_name") or row.get("task") or "") == task
        ]
        for row in sorted(
            task_rows,
            key=lambda item: (
                str(item.get("model_label") or ""),
                str(item.get("mode") or ""),
            ),
        ):
            ax.scatter(
                float(row["speedup_vs_eagle3_dense"]),
                float(row["delta_pp_vs_eagle3_dense"]),
                s=42,
                label=(
                    f"{row.get('model_label', '')} / "
                    f"{row.get('mode', '')}"
                ),
            )
        ax.axhline(-5.0, color="#c43c39", linestyle="--", linewidth=1.2)
        ax.axvline(1.3, color="#555555", linestyle="--", linewidth=1.2)
        ax.axvline(1.4, color="#2f6f44", linestyle=":", linewidth=1.4)
        ax.set_title(task)
        ax.set_xlabel("Speedup vs dense EAGLE3")
        ax.set_ylabel("Accuracy change vs dense EAGLE3 (pp)")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("SpecLink quality-throughput tradeoff")
    fig.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "quality_throughput_tradeoff.png", dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    rows = rows_from_runs(output_dir)
    add_external_baselines(rows, args.baseline_dir)
    add_accuracy_comparisons(rows)
    add_throughput_comparisons(rows)
    add_goal_checks(rows)
    paired_events = add_paired_comparisons(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "paired_regressions.jsonl").open("w", encoding="utf-8") as handle:
        for event in paired_events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    write_report(output_dir / "report.md", rows)
    write_token_dense_accuracy_plot(output_dir, rows)
    write_quality_throughput_plot(output_dir, rows)
    print(output_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
