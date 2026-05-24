#!/usr/bin/env python3
"""Audit SpecLink formal experiment matrix completion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slug(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def base_row(
    matrix: str,
    method: str,
    num_spec_tokens: Any,
    rate: str,
    repeat_id: Any,
    layout: str,
    output_dir: Path,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "matrix": matrix,
        "method": method,
        "draft_method": "",
        "num_spec_tokens": num_spec_tokens,
        "rate": rate,
        "repeat_id": repeat_id,
        "layout": layout,
        "block_size": "",
        "shared_budget": "",
        "private_min": "",
        "private_max": "",
        "lambda_risk": "",
        "alpha": "",
        "beta": "",
        "fallback": "",
        "output_dir": output_dir,
    }
    row.update(extra)
    return row


def planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    if args.matrix == "baseline":
        for repeat in csv_items(args.repeats):
            for rate in csv_items(args.rates):
                rows.append(
                    base_row(
                        "baseline",
                        "dense",
                        0,
                        rate,
                        repeat,
                        "",
                        root / "02_baselines" / f"dense_rate{rate}_r{repeat}",
                    )
                )
                for method in ("eagle3", "peagle"):
                    for k in csv_items(args.k_values):
                        rows.append(
                            base_row(
                                "baseline",
                                method,
                                k,
                                rate,
                                repeat,
                                "",
                                root / "02_baselines" / f"{method}_k{k}_rate{rate}_r{repeat}",
                            )
                        )
    elif args.matrix == "breakdown":
        for method in ("eagle3", "peagle"):
            for k in csv_items(args.k_values):
                for rate in ("1", "4"):
                    rows.append(
                        base_row(
                            "breakdown",
                            method,
                            k,
                            rate,
                            0,
                            "",
                            root / "03_breakdown" / f"{method}_k{k}_rate{rate}_profile",
                        )
                    )
    elif args.matrix == "speclink-plan":
        for layout in csv_items(args.speclink_layouts):
            for k in csv_items(args.k_values):
                rows.append(
                    base_row(
                        "speclink-plan",
                        "speclink",
                        k,
                        "1",
                        0,
                        layout,
                        root / "05_speclink" / f"{layout}_k{k}_rate1_plan",
                    )
                )
    elif args.matrix == "speclink-g2":
        for draft_method in csv_items(args.speclink_draft_methods):
            for k in csv_items(args.k_values):
                for block_size in csv_items(args.block_sizes):
                    for shared_budget in csv_items(args.shared_budgets):
                        for private_min in csv_items(args.private_min_values):
                            for private_max in csv_items(args.private_max_values):
                                for lambda_risk in csv_items(args.lambda_risk_values):
                                    for fallback in csv_items(args.fallback_values):
                                        for alpha in csv_items(args.alpha_values):
                                            for beta in csv_items(args.beta_values):
                                                if fallback in {"1", "true", "on"}:
                                                    layout = "speclink_prob_fallback"
                                                    fallback_label = "1"
                                                elif fallback in {"0", "false", "off"}:
                                                    layout = "speclink_prob"
                                                    fallback_label = "0"
                                                else:
                                                    raise ValueError(f"unsupported fallback value: {fallback}")
                                                run_name = (
                                                    f"{draft_method}_{layout}_k{k}_bs{block_size}_"
                                                    f"sb{shared_budget}_pmin{private_min}_pmax{private_max}_"
                                                    f"lam{slug(lambda_risk)}_a{slug(alpha)}_b{slug(beta)}_"
                                                    f"fb{fallback_label}_rate1_plan"
                                                )
                                                rows.append(
                                                    base_row(
                                                        "speclink-g2",
                                                        "speclink",
                                                        k,
                                                        "1",
                                                        0,
                                                        layout,
                                                        root / "05_speclink_g2" / run_name,
                                                        draft_method=draft_method,
                                                        block_size=block_size,
                                                        shared_budget=shared_budget,
                                                        private_min=private_min,
                                                        private_max=private_max,
                                                        lambda_risk=lambda_risk,
                                                        alpha=alpha,
                                                        beta=beta,
                                                        fallback=fallback_label,
                                                    )
                                                )
    elif args.matrix == "speclink-serving":
        layout = "speclink_prob"
        block_size = "32"
        shared_budget = "16"
        private_min = "0"
        private_max = "16"
        lambda_risk = "0"
        alpha = "8"
        beta = "8"
        fallback = "0"
        for draft_method in csv_items(args.speclink_draft_methods):
            for k in csv_items(args.k_values):
                for rate in csv_items(args.rates):
                    for repeat in csv_items(args.repeats):
                        run_name = (
                            f"{draft_method}_{layout}_k{k}_bs{block_size}_"
                            f"sb{shared_budget}_pmin{private_min}_pmax{private_max}_"
                            f"lam{slug(lambda_risk)}_a{slug(alpha)}_b{slug(beta)}_"
                            f"fb{fallback}_rate{rate}_r{repeat}_plan"
                        )
                        rows.append(
                            base_row(
                                "speclink-serving",
                                "speclink",
                                k,
                                rate,
                                repeat,
                                layout,
                                root / "06_serving_rates" / run_name,
                                draft_method=draft_method,
                                block_size=block_size,
                                shared_budget=shared_budget,
                                private_min=private_min,
                                private_max=private_max,
                                lambda_risk=lambda_risk,
                                alpha=alpha,
                                beta=beta,
                                fallback=fallback,
                            )
                        )
    else:
        raise ValueError(f"unsupported matrix: {args.matrix}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def find_nested(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_nested(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_nested(value, key)
            if found is not None:
                return found
    return None


def stat_mean(obj: Any) -> float | None:
    if isinstance(obj, int | float):
        return float(obj)
    if isinstance(obj, dict):
        if "successful" in obj:
            return stat_mean(obj["successful"])
        value = obj.get("mean")
        if isinstance(value, int | float):
            return float(value)
    return None


def request_total(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("successful", "total"):
            if isinstance(value.get(key), int):
                return int(value[key])
    if isinstance(value, int):
        return value
    return None


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(row["output_dir"])
    required = ["guidellm_results.json", "accuracy_summary.json", "env.txt", "command.txt"]
    if row["method"] != "dense":
        required.append("acceptance_analysis.txt")
    if row["matrix"] in {"breakdown", "speclink-plan", "speclink-g2", "speclink-serving"}:
        required.append("profile_events.jsonl")
    if row["matrix"] in {"speclink-plan", "speclink-g2", "speclink-serving"}:
        required.append("live_sparse_trace.jsonl")

    missing = [name for name in required if not (out_dir / name).exists()]
    guidellm = read_json(out_dir / "guidellm_results.json")
    benchmark = (guidellm.get("benchmarks") or [{}])[0]
    metrics = benchmark.get("metrics", {})
    accuracy = read_json(out_dir / "accuracy_summary.json")
    env = read_env(out_dir / "env.txt")
    accuracy_errors = int_value(accuracy.get("errors"))

    failure_file = out_dir / "failures.md"
    status = "complete" if not missing and not failure_file.exists() else "missing"
    if failure_file.exists() or (accuracy_errors is not None and accuracy_errors > 0):
        status = "failed"

    row = dict(row)
    row["output_dir"] = str(out_dir)
    row["status"] = status
    row["missing_files"] = ",".join(missing)
    row["failure_file"] = str(failure_file) if failure_file.exists() else ""
    row["benchmark_limit"] = env.get("benchmark_limit", "")
    row["accuracy_limit"] = env.get("accuracy_limit", "")
    row["n_requests"] = request_total(find_nested(metrics, "request_totals"))
    row["accuracy_n"] = accuracy.get("n")
    row["accuracy_errors"] = accuracy_errors
    row["output_tokens_per_s"] = stat_mean(find_nested(metrics, "output_tokens_per_second"))
    row["total_tokens_per_s"] = stat_mean(find_nested(metrics, "tokens_per_second"))
    row["flexible_em"] = accuracy.get("flexible_em")
    row["strict_em"] = accuracy.get("strict_em")
    limit_mismatches: list[str] = []
    expected_benchmark_limit = row.get("expected_benchmark_limit")
    expected_accuracy_limit = row.get("expected_accuracy_limit")
    if expected_benchmark_limit and row["benchmark_limit"] != expected_benchmark_limit:
        limit_mismatches.append(
            f"benchmark_limit expected {expected_benchmark_limit} got {row['benchmark_limit'] or 'missing'}"
        )
    if expected_accuracy_limit and row["accuracy_limit"] != expected_accuracy_limit:
        limit_mismatches.append(
            f"accuracy_limit expected {expected_accuracy_limit} got {row['accuracy_limit'] or 'missing'}"
        )
    if status == "complete" and limit_mismatches:
        status = "missing"
        row["status"] = status
        row["missing_files"] = "; ".join(limit_mismatches)
    if (
        status == "complete"
        and row["n_requests"] is not None
        and row["accuracy_n"] is not None
        and int(row["n_requests"]) != int(row["accuracy_n"])
    ):
        status = "failed"
        row["status"] = status
        row["failure_file"] = "n_requests != accuracy_n"
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "matrix",
        "method",
        "draft_method",
        "num_spec_tokens",
        "rate",
        "repeat_id",
        "layout",
        "block_size",
        "shared_budget",
        "private_min",
        "private_max",
        "lambda_risk",
        "alpha",
        "beta",
        "fallback",
        "status",
        "missing_files",
        "benchmark_limit",
        "accuracy_limit",
        "n_requests",
        "accuracy_n",
        "accuracy_errors",
        "output_tokens_per_s",
        "total_tokens_per_s",
        "strict_em",
        "flexible_em",
        "failure_file",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    complete = sum(1 for row in rows if row["status"] == "complete")
    failed = sum(1 for row in rows if row["status"] == "failed")
    missing = sum(1 for row in rows if row["status"] == "missing")
    cols = [
        "matrix",
        "method",
        "draft_method",
        "num_spec_tokens",
        "rate",
        "repeat_id",
        "layout",
        "block_size",
        "shared_budget",
        "private_max",
        "lambda_risk",
        "fallback",
        "status",
        "benchmark_limit",
        "accuracy_limit",
        "n_requests",
        "accuracy_n",
        "accuracy_errors",
        "output_tokens_per_s",
        "flexible_em",
        "missing_files",
    ]
    lines = [
        "# SpecLink Matrix Status",
        "",
        f"- total: {total}",
        f"- complete: {complete}",
        f"- missing: {missing}",
        f"- failed: {failed}",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--matrix",
        choices=["baseline", "breakdown", "speclink-plan", "speclink-g2", "speclink-serving"],
        default="baseline",
    )
    parser.add_argument("--rates", default="1,2,4")
    parser.add_argument("--repeats", default="0,1,2")
    parser.add_argument("--k-values", default="2,4,8")
    parser.add_argument(
        "--speclink-layouts",
        default="independent_topk,snapkv_static,shared_only,speclink_fixed,speclink_prob,speclink_prob_fallback",
    )
    parser.add_argument("--speclink-draft-methods", default="eagle3,peagle")
    parser.add_argument("--block-sizes", default="32")
    parser.add_argument("--shared-budgets", default="16,32,64")
    parser.add_argument("--private-min-values", default="0")
    parser.add_argument("--private-max-values", default="8,16")
    parser.add_argument("--lambda-risk-values", default="0,1")
    parser.add_argument("--fallback-values", default="0,1")
    parser.add_argument("--alpha-values", default="8")
    parser.add_argument("--beta-values", default="8")
    parser.add_argument(
        "--expected-benchmark-limit",
        help="If set, rows with a different env benchmark_limit are not formal-complete.",
    )
    parser.add_argument(
        "--expected-accuracy-limit",
        help="If set, rows with a different env accuracy_limit are not formal-complete.",
    )
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows = []
    for row in planned_rows(args):
        row["expected_benchmark_limit"] = args.expected_benchmark_limit
        row["expected_accuracy_limit"] = args.expected_accuracy_limit
        rows.append(audit_row(row))
    write_tsv(Path(args.out_tsv), rows)
    write_md(Path(args.out_md), rows)
    print(f"audited {len(rows)} {args.matrix} matrix rows")
    print(f"wrote {args.out_tsv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
