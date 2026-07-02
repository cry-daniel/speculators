#!/usr/bin/env python3
"""Build a stability-aware SR24 accuracy gate from lm-eval sample files.

The normal aggregate report compares an experiment with one dense reference.
For SR24 debugging that is too weak: dense EAGLE3 speculative serving can vary
across runs on reasoning samples even with greedy decoding.  This script reads
multiple result roots, treats dense EAGLE3 runs as repeats, and reports only
regressions that survive the repeat/AR reference checks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DENSE_SPEC_MODES = {"dense_baseline", "eagle3_dense"}
AR_MODES = {"dense_ar"}
EXPERIMENT_MODES = {"base_only_24", "all_corrected_24", "speclink_t08"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_key(sample: dict[str, Any]) -> str:
    for key in ("doc_id", "doc_hash", "prompt_hash"):
        if key in sample and sample[key] is not None:
            return f"{key}:{sample[key]}"
    return json.dumps(sample.get("doc", sample), sort_keys=True, ensure_ascii=False)


def _correct(sample: dict[str, Any]) -> bool | None:
    for key in ("exact_match", "acc", "acc_norm", "pass@1", "score"):
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) > 0.5
    return None


def _filtered(sample: dict[str, Any]) -> str:
    value = sample.get("filtered_resps")
    if isinstance(value, list) and value:
        return str(value[0])
    value = sample.get("resps")
    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], list)
        and value[0]
    ):
        return str(value[0][0])
    return ""


def _text(sample: dict[str, Any]) -> str:
    value = sample.get("resps")
    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], list)
        and value[0]
    ):
        return str(value[0][0])
    return _filtered(sample)


def _sample_paths(run_dir: Path, task: str) -> list[Path]:
    output = run_dir / "lm_eval_output"
    paths = list(output.rglob(f"samples_{task}_*.jsonl"))
    if not paths:
        paths = list(output.rglob("samples_*.jsonl"))
    return sorted(paths)


def _read_samples(run_dir: Path, task: str) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for path in _sample_paths(run_dir, task):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                samples[_sample_key(sample)] = {
                    "correct": _correct(sample),
                    "doc_id": sample.get("doc_id"),
                    "doc_hash": sample.get("doc_hash"),
                    "prompt_hash": sample.get("prompt_hash"),
                    "target": sample.get("target"),
                    "filtered": _filtered(sample),
                    "text": _text(sample),
                    "sample_path": str(path.resolve()),
                }
    return samples


def _collect_runs(roots: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for root in roots:
        for meta_path in sorted(root.rglob("run_meta.json")):
            meta = _load_json(meta_path)
            if meta.get("status") != "ok":
                continue
            mode = str(meta.get("mode") or "")
            if mode not in DENSE_SPEC_MODES | AR_MODES | EXPERIMENT_MODES:
                continue
            task = str(meta.get("task") or "")
            run_dir = meta_path.parent
            samples = _read_samples(run_dir, task)
            if not samples:
                continue
            runs.append(
                {
                    "root": str(root.resolve()),
                    "run_dir": str(run_dir.resolve()),
                    "model_label": str(meta.get("model_label") or ""),
                    "task": task,
                    "mode": mode,
                    "samples": samples,
                    "spec_acceptance_rate": meta.get("spec_acceptance_rate"),
                    "started_at": meta.get("started_at"),
                }
            )
    return runs


def _truth_values(records: list[dict[str, Any]]) -> list[bool]:
    values: list[bool] = []
    for record in records:
        value = record.get("correct")
        if value is not None:
            values.append(bool(value))
    return values


def _stable_state(values: list[bool]) -> str:
    if not values:
        return "missing"
    if all(values):
        return "all_correct"
    if not any(values):
        return "all_wrong"
    return "unstable"


def _first_char_diff(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit if len(left) != len(right) else -1


def _pick_reference(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return records[0] if records else None


def _build_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["model_label"], run["task"])].append(run)

    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for (model_label, task), group_runs in sorted(grouped.items()):
        ar_runs = [run for run in group_runs if run["mode"] in AR_MODES]
        dense_runs = [run for run in group_runs if run["mode"] in DENSE_SPEC_MODES]
        experiment_runs = [
            run for run in group_runs if run["mode"] in EXPERIMENT_MODES
        ]
        dense_sample_keys = set().union(
            *(set(run["samples"]) for run in dense_runs)
        ) if dense_runs else set()
        ar_sample_keys = set().union(
            *(set(run["samples"]) for run in ar_runs)
        ) if ar_runs else set()
        for exp_run in experiment_runs:
            exp_samples = exp_run["samples"]
            common_with_dense = set(exp_samples) & dense_sample_keys
            common_with_ar = set(exp_samples) & ar_sample_keys
            stable_dense_regressions = 0
            stable_ar_regressions = 0
            stable_both_regressions = 0
            dense_unstable_samples = 0
            dense_repeat_all_correct = 0
            dense_repeat_all_wrong = 0
            dense_repeat_missing = 0
            exp_correct_count = 0
            paired_count = 0
            for key, exp_sample in sorted(exp_samples.items()):
                exp_correct = exp_sample.get("correct")
                dense_records = [
                    run["samples"][key]
                    for run in dense_runs
                    if key in run["samples"]
                ]
                ar_records = [
                    run["samples"][key]
                    for run in ar_runs
                    if key in run["samples"]
                ]
                dense_values = _truth_values(dense_records)
                ar_values = _truth_values(ar_records)
                dense_state = _stable_state(dense_values)
                ar_state = _stable_state(ar_values)
                if dense_state == "all_correct":
                    dense_repeat_all_correct += 1
                elif dense_state == "all_wrong":
                    dense_repeat_all_wrong += 1
                elif dense_state == "unstable":
                    dense_unstable_samples += 1
                else:
                    dense_repeat_missing += 1
                if exp_correct is not None:
                    exp_correct_count += int(bool(exp_correct))
                if dense_values or ar_values:
                    paired_count += 1

                is_exp_wrong = exp_correct is False
                reg_dense = dense_state == "all_correct" and is_exp_wrong
                reg_ar = ar_state == "all_correct" and is_exp_wrong
                reg_both = reg_dense and (
                    ar_state in {"all_correct", "missing"}
                )
                stable_dense_regressions += int(reg_dense)
                stable_ar_regressions += int(reg_ar)
                stable_both_regressions += int(reg_both)
                if reg_dense or reg_ar or dense_state == "unstable":
                    dense_ref = _pick_reference(dense_records)
                    first_diff = (
                        _first_char_diff(dense_ref["text"], exp_sample["text"])
                        if dense_ref is not None
                        else ""
                    )
                    event_rows.append(
                        {
                            "model_label": model_label,
                            "task": task,
                            "mode": exp_run["mode"],
                            "sample_key": key,
                            "doc_id": exp_sample.get("doc_id"),
                            "target": exp_sample.get("target"),
                            "event": (
                                "stable_regression"
                                if reg_dense or reg_ar
                                else "dense_repeat_unstable"
                            ),
                            "exp_correct": exp_correct,
                            "dense_repeat_state": dense_state,
                            "dense_repeat_values": json.dumps(dense_values),
                            "ar_state": ar_state,
                            "ar_values": json.dumps(ar_values),
                            "exp_filtered": exp_sample.get("filtered"),
                            "dense_filtered": (
                                dense_ref.get("filtered")
                                if dense_ref is not None
                                else ""
                            ),
                            "first_char_diff_vs_first_dense": first_diff,
                            "experiment_sample_path": exp_sample.get("sample_path"),
                            "experiment_run_dir": exp_run["run_dir"],
                            "first_dense_sample_path": (
                                dense_ref.get("sample_path")
                                if dense_ref is not None
                                else ""
                            ),
                        }
                    )
            total = len(exp_samples)
            summary_rows.append(
                {
                    "model_label": model_label,
                    "task": task,
                    "mode": exp_run["mode"],
                    "experiment_run_dir": exp_run["run_dir"],
                    "experiment_root": exp_run["root"],
                    "samples": total,
                    "paired_samples": paired_count,
                    "dense_repeat_count": len(dense_runs),
                    "ar_repeat_count": len(ar_runs),
                    "common_with_dense": len(common_with_dense),
                    "common_with_ar": len(common_with_ar),
                    "experiment_correct": exp_correct_count,
                    "experiment_accuracy": (
                        exp_correct_count / total if total else ""
                    ),
                    "dense_repeat_all_correct": dense_repeat_all_correct,
                    "dense_repeat_all_wrong": dense_repeat_all_wrong,
                    "dense_repeat_unstable": dense_unstable_samples,
                    "dense_repeat_missing": dense_repeat_missing,
                    "stable_regressions_vs_dense_repeat":
                    stable_dense_regressions,
                    "stable_regressions_vs_ar": stable_ar_regressions,
                    "stable_regressions_vs_both_or_dense_only":
                    stable_both_regressions,
                    "spec_acceptance_rate": exp_run.get("spec_acceptance_rate"),
                }
            )
    return summary_rows, event_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def _write_report(
    path: Path,
    *,
    roots: list[Path],
    runs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SR24 Stability Gate",
        "",
        "This report compares SR24 samples against repeated dense EAGLE3",
        "references and optional dense AR references. A stable regression is",
        "counted only when every available dense speculative repeat for that",
        "sample is correct and the SR24 sample is wrong. `dense_repeat_unstable`",
        "means the dense speculative repeats disagree, so that sample should not",
        "be used as direct SR24 quality evidence.",
        "",
        "## Summary",
        "",
        "| model | task | mode | samples | dense repeats | AR repeats | exp acc | dense repeat unstable | stable reg vs dense repeats | stable reg vs AR | stable reg vs both/dense-only |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {task} | `{mode}` | {samples} | {dense_repeats} | {ar_repeats} | {acc} | {unstable} | {reg_dense} | {reg_ar} | {reg_both} |".format(
                model=row.get("model_label", ""),
                task=row.get("task", ""),
                mode=row.get("mode", ""),
                samples=_fmt(row.get("samples")),
                dense_repeats=_fmt(row.get("dense_repeat_count")),
                ar_repeats=_fmt(row.get("ar_repeat_count")),
                acc=_fmt(row.get("experiment_accuracy")),
                unstable=_fmt(row.get("dense_repeat_unstable")),
                reg_dense=_fmt(row.get("stable_regressions_vs_dense_repeat")),
                reg_ar=_fmt(row.get("stable_regressions_vs_ar")),
                reg_both=_fmt(
                    row.get("stable_regressions_vs_both_or_dense_only")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Runs Used",
            "",
            "| role | model | task | mode | samples | run dir |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for run in runs:
        role = (
            "ar"
            if run["mode"] in AR_MODES
            else "dense_repeat"
            if run["mode"] in DENSE_SPEC_MODES
            else "experiment"
        )
        lines.append(
            "| {role} | {model} | {task} | `{mode}` | {samples} | `{run_dir}` |".format(
                role=role,
                model=run["model_label"],
                task=run["task"],
                mode=run["mode"],
                samples=len(run["samples"]),
                run_dir=run["run_dir"],
            )
        )
    lines.extend(["", "## Inputs", ""])
    for root in roots:
        lines.append(f"- `{root.resolve()}`")
    lines.extend(
        [
            "",
            "## Event Files",
            "",
            f"- summary CSV: `{(path.parent / 'stability_summary.csv').resolve()}`",
            f"- event JSONL: `{(path.parent / 'stability_events.jsonl').resolve()}`",
            f"- event CSV: `{(path.parent / 'stability_events.csv').resolve()}`",
        ]
    )
    stable_events = [
        row for row in event_rows if row.get("event") == "stable_regression"
    ]
    if stable_events:
        lines.extend(["", "## Stable Regression Samples", ""])
        for row in stable_events[:20]:
            lines.append(
                "- doc `{doc}` mode `{mode}` target `{target}` dense `{dense}` exp `{exp}`".format(
                    doc=row.get("doc_id"),
                    mode=row.get("mode"),
                    target=row.get("target"),
                    dense=row.get("dense_filtered"),
                    exp=row.get("exp_filtered"),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        type=Path,
        help="One or more run_lm_eval_accuracy output roots.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(args.roots)
    summary_rows, event_rows = _build_rows(runs)
    _write_csv(args.output_root / "stability_summary.csv", summary_rows)
    _write_csv(args.output_root / "stability_events.csv", event_rows)
    _write_jsonl(args.output_root / "stability_events.jsonl", event_rows)
    _write_report(
        args.output_root / "report.md",
        roots=args.roots,
        runs=runs,
        summary_rows=summary_rows,
        event_rows=event_rows,
    )
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
