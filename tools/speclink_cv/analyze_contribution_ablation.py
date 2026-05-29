#!/usr/bin/env python3
"""Summarize SpecLink-CV contribution ablations.

This analysis keeps the core skip-suffix mechanism enabled for every CV row.
It compares:

* non-staged CV vs staged CV, with the same batched scheduler setting, to
  estimate the contribution from avoiding draft-model suffix generation;
* singleton-live CV vs batched-live CV, both with h<K prefix verification, to
  estimate the contribution from batching/scheduling.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


KEY_FIELDS = ("model", "dataset", "K", "batch_size")
DEFAULT_BASELINE = "eagle3_oneshot"
DEFAULT_NONSTAGED = "cv_half_async_simple"
DEFAULT_STAGED = "cv_half_async_staged_simple"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batched-root",
        action="append",
        default=[],
        help="Result root with normal batched live prefix/suffix CV.",
    )
    parser.add_argument(
        "--singleton-root",
        action="append",
        default=[],
        help=(
            "Result root with live h<K CV but singleton scheduling, e.g. "
            "MAX_VERIFY_SEQS_PER_STEP=1 and SPECLINK_CV_ALLOW_BATCHED_SUFFIX=0."
        ),
    )
    parser.add_argument("--output-dir", help="Directory for CSV/Markdown reports.")
    parser.add_argument("--baseline-method", default=DEFAULT_BASELINE)
    parser.add_argument("--nonstaged-method", default=DEFAULT_NONSTAGED)
    parser.add_argument("--staged-method", default=DEFAULT_STAGED)
    return parser.parse_args()


def read_summary(root: Path) -> list[dict[str, str]]:
    candidates = [
        root / "09_reports" / "summary_metrics.csv",
        root / "summary_metrics.csv",
    ]
    summary_path = next((path for path in candidates if path.exists()), None)
    if summary_path is None:
        raise FileNotFoundError(f"summary_metrics.csv not found under {root}")
    with summary_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def iter_rows(roots: Iterable[str], label: str) -> Iterable[dict[str, str]]:
    for root_name in roots:
        root = Path(root_name).resolve()
        for row in read_summary(root):
            row = dict(row)
            row["_ablation_group"] = label
            row["_source_root"] = str(root)
            if row.get("measurement_type") not in (
                "",
                "guidellm_end_to_end",
                "steady_state_saturated",
            ):
                continue
            if row.get("status") not in ("", "ok"):
                continue
            yield row


def key_for(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def f(row: dict[str, str] | None, field: str) -> float | None:
    if row is None:
        return None
    value = row.get(field, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom in (None, 0.0):
        return None
    return numer / denom


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def choose(rows: dict[tuple[str, str], dict[str, str]], group: str, method: str):
    return rows.get((group, method))


def build_records(
    rows: list[dict[str, str]],
    *,
    baseline_method: str,
    nonstaged_method: str,
    staged_method: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[tuple[str, str], dict[str, str]]] = (
        defaultdict(dict)
    )
    for row in rows:
        grouped[key_for(row)][(row["_ablation_group"], row.get("method", ""))] = row

    records: list[dict[str, object]] = []
    for key, case_rows in sorted(grouped.items()):
        baseline = choose(case_rows, "batched", baseline_method) or choose(
            case_rows, "singleton", baseline_method
        )
        nonstaged_batched = choose(case_rows, "batched", nonstaged_method)
        staged_batched = choose(case_rows, "batched", staged_method)
        staged_singleton = choose(case_rows, "singleton", staged_method)
        nonstaged_singleton = choose(case_rows, "singleton", nonstaged_method)

        staged_reference = staged_batched or staged_singleton
        nonstaged_reference = nonstaged_batched or nonstaged_singleton
        if baseline is None and staged_reference is None and nonstaged_reference is None:
            continue

        base_tps = f(baseline, "throughput")
        nonstaged_batched_tps = f(nonstaged_batched, "throughput")
        staged_batched_tps = f(staged_batched, "throughput")
        staged_singleton_tps = f(staged_singleton, "throughput")

        record: dict[str, object] = {
            "model": key[0],
            "dataset": key[1],
            "K": key[2],
            "batch_size": key[3],
            "baseline_tps": base_tps,
            "nonstaged_batched_tps": nonstaged_batched_tps,
            "staged_batched_tps": staged_batched_tps,
            "staged_singleton_tps": staged_singleton_tps,
            "nonstaged_batched_speedup_vs_eagle3": ratio(
                nonstaged_batched_tps, base_tps
            ),
            "staged_batched_speedup_vs_eagle3": ratio(staged_batched_tps, base_tps),
            "dlm_suffix_saving_speedup": ratio(
                staged_batched_tps, nonstaged_batched_tps
            ),
            "batch_scheduling_speedup": ratio(
                staged_batched_tps, staged_singleton_tps
            ),
            "skip_suffix_ratio_nonstaged": f(
                nonstaged_reference, "skipped_suffix_ratio"
            ),
            "skip_suffix_ratio_staged": f(staged_reference, "skipped_suffix_ratio"),
            "tlm_token_ratio_staged": f(
                staged_reference, "verify_target_token_ratio_vs_oneshot_est"
            ),
            "draft_tokens_full_k_est": f(staged_reference, "draft_tokens_full_k_est"),
            "draft_tokens_generated_est": f(
                staged_reference, "draft_tokens_generated_est"
            ),
            "draft_tokens_saved_by_staging_est": f(
                staged_reference, "draft_tokens_saved_by_staging_est"
            ),
            "draft_discard_ratio_est": f(staged_reference, "draft_discard_ratio_est"),
            "prefix_full_accept_ratio": f(staged_reference, "suffix_scheduled_ratio"),
            "prefix_accepted_tokens_avg": f(
                staged_reference, "prefix_accepted_tokens_avg"
            ),
            "prefix_dispatch_seq_util_avg": f(
                staged_reference, "prefix_dispatch_seq_util_avg"
            ),
            "actual_scheduled_seqs_per_step_eagle3": f(
                baseline, "actual_scheduled_seqs_per_step"
            ),
            "actual_scheduled_seqs_per_step_staged": f(
                staged_reference, "actual_scheduled_seqs_per_step"
            ),
            "actual_scheduled_tokens_per_step_eagle3": f(
                baseline, "actual_scheduled_tokens_per_step"
            ),
            "actual_scheduled_tokens_per_step_staged": f(
                staged_reference, "actual_scheduled_tokens_per_step"
            ),
            "scheduled_seqs_per_step_gain": ratio(
                f(staged_reference, "actual_scheduled_seqs_per_step"),
                f(baseline, "actual_scheduled_seqs_per_step"),
            ),
            "scheduled_tokens_per_step_gain": ratio(
                f(staged_reference, "actual_scheduled_tokens_per_step"),
                f(baseline, "actual_scheduled_tokens_per_step"),
            ),
            "gpu_active_util_eagle3": f(baseline, "gpu_active_util"),
            "gpu_active_util_staged": f(staged_reference, "gpu_active_util"),
            "quality_score_eagle3": f(baseline, "quality_score"),
            "quality_score_staged": f(staged_reference, "quality_score"),
            "quality_gate_staged": (staged_reference or {}).get(
                "speedup_claim_status", ""
            ),
        }
        records.append(record)
    return records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        path.write_text("")
        return
    fields = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def pairwise_model_notes(records: list[dict[str, object]]) -> list[str]:
    by_case: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_case[(str(record["dataset"]), str(record["K"]), str(record["batch_size"]))].append(
            record
        )

    notes: list[str] = []
    for (dataset, k, batch_size), case_records in sorted(by_case.items()):
        qwen = next(
            (row for row in case_records if str(row["model"]).startswith("qwen3")),
            None,
        )
        llama = next(
            (row for row in case_records if str(row["model"]).startswith("llama3")),
            None,
        )
        if qwen is None or llama is None:
            continue
        q_speed = qwen.get("staged_batched_speedup_vs_eagle3")
        l_speed = llama.get("staged_batched_speedup_vs_eagle3")
        if not isinstance(q_speed, float) or not isinstance(l_speed, float):
            continue
        q_base_util = qwen.get("gpu_active_util_eagle3")
        l_base_util = llama.get("gpu_active_util_eagle3")
        q_skip = qwen.get("skip_suffix_ratio_staged")
        l_skip = llama.get("skip_suffix_ratio_staged")
        q_seq_gain = ratio(
            qwen.get("actual_scheduled_seqs_per_step_staged"),  # type: ignore[arg-type]
            qwen.get("actual_scheduled_seqs_per_step_eagle3"),  # type: ignore[arg-type]
        )
        l_seq_gain = ratio(
            llama.get("actual_scheduled_seqs_per_step_staged"),  # type: ignore[arg-type]
            llama.get("actual_scheduled_seqs_per_step_eagle3"),  # type: ignore[arg-type]
        )
        notes.append(
            "- "
            f"{dataset} K={k} bs={batch_size}: Llama speedup {fmt(l_speed)} vs "
            f"Qwen {fmt(q_speed)}. Baseline active GPU util was "
            f"Llama {fmt(l_base_util, 1)} vs Qwen {fmt(q_base_util, 1)}, "
            f"skip-suffix ratios were Llama {fmt(l_skip)} vs Qwen {fmt(q_skip)}, "
            f"and scheduled-seq gain was Llama {fmt(l_seq_gain)}x vs "
            f"Qwen {fmt(q_seq_gain)}x."
        )
    return notes


def write_markdown(path: Path, records: list[dict[str, object]]) -> None:
    lines = [
        "# SpecLink-CV Contribution Ablation",
        "",
        "All CV rows keep h<K prefix verification enabled, so skip-suffix remains the core mechanism. This report does not treat one-shot EAGLE3 as a skip-suffix ablation; EAGLE3 is only the throughput baseline.",
        "",
        "## Summary",
        "",
        "| model | dataset | K | bs | EAGLE3 tok/s | non-staged batched tok/s | staged batched tok/s | staged singleton tok/s | staged speedup | DLM suffix speedup | batch sched speedup | sched seq gain | sched token gain | skip suffix | TLM token ratio | DLM saved | quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record['model']} | {record['dataset']} | {record['K']} | "
            f"{record['batch_size']} | {fmt(record['baseline_tps'], 2)} | "
            f"{fmt(record['nonstaged_batched_tps'], 2)} | "
            f"{fmt(record['staged_batched_tps'], 2)} | "
            f"{fmt(record['staged_singleton_tps'], 2)} | "
            f"{fmt(record['staged_batched_speedup_vs_eagle3'])} | "
            f"{fmt(record['dlm_suffix_saving_speedup'])} | "
            f"{fmt(record['batch_scheduling_speedup'])} | "
            f"{fmt(record['scheduled_seqs_per_step_gain'])} | "
            f"{fmt(record['scheduled_tokens_per_step_gain'])} | "
            f"{fmt(record['skip_suffix_ratio_staged'])} | "
            f"{fmt(record['tlm_token_ratio_staged'])} | "
            f"{fmt(record['draft_tokens_saved_by_staging_est'], 0)} | "
            f"{record['quality_gate_staged']} |"
        )

    lines.extend(
        [
            "",
            "## Reading The Columns",
            "",
            "- `skip suffix` is the verifier suffix that was not sent to the target model after a prefix reject. This is the core contribution and is kept enabled in all CV variants.",
            "- `DLM suffix speedup` is staged batched throughput divided by non-staged batched throughput. It estimates the benefit of not drafting suffix tokens before the prefix result is known.",
            "- `batch sched speedup` is staged batched throughput divided by staged singleton-live throughput. It estimates the benefit of batching prefix/suffix verifier work while still using h<K chunked verification.",
            "- `sched seq/token gain` are active vLLM scheduling proxies. They are useful when fixed-request GuideLLM throughput is distorted by the final drain/tail.",
            "- `TLM token ratio` estimates target-model verification tokens relative to one-shot EAGLE3; lower is better when quality stays stable.",
            "",
            "## Llama vs Qwen",
            "",
        ]
    )
    notes = pairwise_model_notes(records)
    if notes:
        lines.extend(notes)
    else:
        lines.append("- Need paired Qwen and Llama rows with the same dataset/K/batch size.")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if not args.batched_root and not args.singleton_root:
        raise SystemExit("provide at least one --batched-root or --singleton-root")
    output_dir = Path(args.output_dir or args.batched_root[0] or args.singleton_root[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_rows(args.batched_root, "batched"))
    rows.extend(iter_rows(args.singleton_root, "singleton"))
    records = build_records(
        rows,
        baseline_method=args.baseline_method,
        nonstaged_method=args.nonstaged_method,
        staged_method=args.staged_method,
    )
    write_csv(output_dir / "contribution_ablation.csv", records)
    write_markdown(output_dir / "contribution_ablation.md", records)
    print(f"[INFO] wrote {output_dir / 'contribution_ablation.csv'}")
    print(f"[INFO] wrote {output_dir / 'contribution_ablation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
