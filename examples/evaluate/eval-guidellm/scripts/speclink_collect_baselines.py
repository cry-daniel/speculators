#!/usr/bin/env python3
"""Collect dense/EAGLE3/P-EAGLE run directories into baseline summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from pathlib import Path
from typing import Any


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
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


def stat_value(obj: Any, preferred: str = "mean") -> float | None:
    if obj is None:
        return None
    if isinstance(obj, int | float):
        return float(obj)
    if isinstance(obj, dict):
        if "successful" in obj:
            return stat_value(obj["successful"], preferred)
        if preferred in obj and isinstance(obj[preferred], int | float):
            return float(obj[preferred])
        percentiles = obj.get("percentiles")
        if isinstance(percentiles, dict) and preferred in percentiles:
            value = percentiles[preferred]
            return float(value) if isinstance(value, int | float) else None
    return None


def parse_guidellm(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = (data.get("benchmarks") or [{}])[0]
    metrics = benchmark.get("metrics", {})
    args = data.get("args", {})
    return {
        "n_requests": find_nested(metrics, "request_totals") or find_nested(
            benchmark,
            "requests_made",
        ),
        "output_tokens_per_s": stat_value(find_nested(metrics, "output_tokens_per_second")),
        "total_tokens_per_s": stat_value(find_nested(metrics, "tokens_per_second")),
        "requests_per_s": stat_value(find_nested(metrics, "requests_per_second")),
        "mean_latency_ms": _seconds_to_ms(stat_value(find_nested(metrics, "request_latency"))),
        "p50_latency_ms": _seconds_to_ms(
            stat_value(find_nested(metrics, "request_latency"), "p50")
        ),
        "p95_latency_ms": _seconds_to_ms(
            stat_value(find_nested(metrics, "request_latency"), "p95")
        ),
        "ttft_ms": stat_value(find_nested(metrics, "time_to_first_token_ms")),
        "itl_ms": stat_value(find_nested(metrics, "inter_token_latency_ms")),
        "tpot_ms": stat_value(find_nested(metrics, "time_per_output_token_ms")),
        "rate_from_guidellm": _first_rate(args.get("rate")),
    }


def _seconds_to_ms(value: float | None) -> float | None:
    return value * 1000.0 if value is not None else None


def _first_rate(value: Any) -> float | None:
    if isinstance(value, list) and value:
        return float(value[0])
    if isinstance(value, int | float):
        return float(value)
    return None


def parse_request_total(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("successful", "total"):
            if key in value and isinstance(value[key], int):
                return int(value[key])
    if isinstance(value, int):
        return value
    return None


def parse_acceptance(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Weighted per-position acceptance rates:\s*\n\[([^\]]*)\]", text)
    if not match:
        return {}
    values = [float(item) for item in match.group(1).split() if item.strip()]
    result: dict[str, Any] = {"mean_accepted_tokens": sum(values)}
    for idx, value in enumerate(values, start=1):
        result[f"acceptance_rate_pos_{idx}"] = value
    return result


def parse_accuracy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "strict_em": data.get("strict_em"),
        "flexible_em": data.get("flexible_em"),
        "dense_output_equiv": data.get("dense_output_equivalence_rate"),
        "avg_output_tokens": data.get("avg_output_tokens"),
        "accuracy_errors": data.get("errors"),
    }
    result.update(parse_equivalence_outputs(path.with_name("accuracy_outputs.jsonl")))
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_equivalence_outputs(path: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("dense_output_exact_equivalence") is not None
        or row.get("dense_output_normalized_equivalence") is not None
    ]
    if not rows:
        return {}

    exact_values = [bool(row.get("dense_output_exact_equivalence")) for row in rows]
    normalized_values = [
        bool(row.get("dense_output_normalized_equivalence")) for row in rows
    ]
    normalized_mismatches = [
        row for row in rows if not bool(row.get("dense_output_normalized_equivalence"))
    ]
    exact_rate = sum(exact_values) / len(exact_values)
    normalized_rate = sum(normalized_values) / len(normalized_values)
    return {
        "dense_output_exact_equiv": exact_rate,
        "dense_output_exact_mismatch_rate": 1.0 - exact_rate,
        "dense_output_normalized_mismatch_rate": 1.0 - normalized_rate,
        "dense_equiv_compared_n": len(rows),
        "final_answer_correct_when_dense_mismatch": (
            sum(bool(row.get("flexible_correct")) for row in normalized_mismatches)
            / len(normalized_mismatches)
            if normalized_mismatches
            else None
        ),
        "_equivalence_examples": normalized_mismatches[:10],
    }


def infer_method(run_dir: Path, env: dict[str, str]) -> str:
    if env.get("method"):
        return env["method"]
    name = run_dir.name.lower()
    for method in ("speclink", "peagle", "eagle3", "dense", "base"):
        if method in name:
            return "dense" if method == "base" else method
    return "unknown"


def infer_k(run_dir: Path, env: dict[str, str], method: str) -> int | None:
    if env.get("num_spec_tokens"):
        return int(env["num_spec_tokens"])
    match = re.search(r"k(\d+)", run_dir.name.lower())
    if match:
        return int(match.group(1))
    if method == "dense":
        return 0
    return None


def server_error_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "ERROR" in line or "Traceback" in line:
                count += 1
    return count


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for guidellm_path in sorted(root.rglob("guidellm_results.json")):
        run_dir = guidellm_path.parent
        env = read_env(run_dir / "env.txt")
        method = infer_method(run_dir, env)
        guidellm = parse_guidellm(guidellm_path)
        row: dict[str, Any] = {
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "method": method,
            "num_spec_tokens": infer_k(run_dir, env, method),
            "rate": env.get("guidellm_rate") or guidellm.get("rate_from_guidellm"),
            "repeat_id": env.get("repeat_id"),
            "n_requests": parse_request_total(guidellm.get("n_requests")),
            "server_errors": server_error_count(run_dir / "vllm_server.log"),
        }
        for key, value in guidellm.items():
            if key not in {"n_requests", "rate_from_guidellm"}:
                row[key] = value
        row.update(parse_accuracy(run_dir / "accuracy_summary.json"))
        row.update(parse_acceptance(run_dir / "acceptance_analysis.txt"))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "method",
        "run",
        "num_spec_tokens",
        "rate",
        "repeat_id",
        "n_requests",
        "output_tokens_per_s",
        "total_tokens_per_s",
        "requests_per_s",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "ttft_ms",
        "itl_ms",
        "tpot_ms",
        "strict_em",
        "flexible_em",
        "dense_output_equiv",
        "dense_output_exact_equiv",
        "dense_output_exact_mismatch_rate",
        "dense_output_normalized_mismatch_rate",
        "dense_equiv_compared_n",
        "final_answer_correct_when_dense_mismatch",
        "avg_output_tokens",
        "mean_accepted_tokens",
        "server_errors",
        "run_dir",
    ]
    acceptance_keys = sorted(
        {key for row in rows for key in row if key.startswith("acceptance_rate_pos_")},
        key=lambda item: int(item.rsplit("_", 1)[1]),
    )
    keys[-2:-2] = acceptance_keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_cols = sorted(
        {key for row in rows for key in row if key.startswith("acceptance_rate_pos_")},
        key=lambda item: int(item.rsplit("_", 1)[1]),
    )
    sections = [
        (
            "Throughput",
            [
                "method",
                "run",
                "num_spec_tokens",
                "rate",
                "repeat_id",
                "n_requests",
                "output_tokens_per_s",
                "total_tokens_per_s",
                "requests_per_s",
                "server_errors",
            ],
        ),
        (
            "Latency",
            [
                "method",
                "run",
                "mean_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "ttft_ms",
                "itl_ms",
                "tpot_ms",
            ],
        ),
        (
            "Accuracy and Dense Equivalence",
            [
                "method",
                "run",
                "strict_em",
                "flexible_em",
                "dense_output_equiv",
                "dense_output_exact_equiv",
                "dense_output_exact_mismatch_rate",
                "dense_output_normalized_mismatch_rate",
                "final_answer_correct_when_dense_mismatch",
                "avg_output_tokens",
            ],
        ),
        (
            "Speculative Acceptance",
            [
                "method",
                "run",
                "mean_accepted_tokens",
                *acceptance_cols,
            ],
        ),
    ]
    lines: list[str] = []
    for title, cols in sections:
        lines.extend(
            [
                f"### {title}",
                "",
                "| " + " | ".join(cols) + " |",
                "| " + " | ".join(["---"] * len(cols)) + " |",
            ]
        )
        for row in rows:
            values = [_fmt(row.get(col)) for col in cols]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def attach_dense_outputs(rows: list[dict[str, Any]]) -> None:
    dense_by_parent: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("method") != "dense":
            continue
        run_dir = Path(str(row.get("run_dir", "")))
        dense_by_parent[run_dir.parent] = {
            str(item.get("id")): item for item in read_jsonl(run_dir / "accuracy_outputs.jsonl")
        }
    for row in rows:
        examples = row.get("_equivalence_examples") or []
        if not examples:
            continue
        run_dir = Path(str(row.get("run_dir", "")))
        dense_rows = read_dense_reference_from_command(run_dir)
        if not dense_rows:
            dense_rows = dense_by_parent.get(run_dir.parent, {})
        for example in examples:
            dense = dense_rows.get(str(example.get("id")))
            example["dense_model_output"] = (
                dense.get("model_output")
                if dense
                else "<missing dense output for this id>"
            )


def read_dense_reference_from_command(run_dir: Path) -> dict[str, dict[str, Any]]:
    command_path = run_dir / "command.txt"
    if not command_path.exists():
        return {}
    try:
        parts = shlex.split(command_path.read_text(encoding="utf-8", errors="ignore"))
    except ValueError:
        return {}
    if "--dense-reference-jsonl" not in parts:
        return {}
    idx = parts.index("--dense-reference-jsonl")
    if idx + 1 >= len(parts):
        return {}
    ref_path = Path(parts[idx + 1])
    env = read_env(run_dir / "env.txt")
    if not ref_path.is_absolute():
        cwd = Path(env.get("cwd", "."))
        ref_path = cwd / ref_path
    return {str(item.get("id")): item for item in read_jsonl(ref_path)}


def write_equivalence_diffs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dense Output Equivalence Diffs",
        "",
        "Only runs with a dense reference are listed. Examples are normalized-output mismatches.",
        "",
    ]
    any_examples = False
    for row in rows:
        examples = row.get("_equivalence_examples") or []
        if not examples:
            continue
        any_examples = True
        lines.extend(
            [
                f"## {row.get('method')} / {row.get('run')}",
                "",
                f"- compared_n: {_fmt(row.get('dense_equiv_compared_n'))}",
                f"- exact_mismatch_rate: {_fmt(row.get('dense_output_exact_mismatch_rate'))}",
                f"- normalized_mismatch_rate: {_fmt(row.get('dense_output_normalized_mismatch_rate'))}",
                f"- final_answer_correct_when_dense_mismatch: {_fmt(row.get('final_answer_correct_when_dense_mismatch'))}",
                "",
            ]
        )
        for idx, example in enumerate(examples, start=1):
            lines.extend(
                [
                    f"### Example {idx}: id={example.get('id')}",
                    "",
                    f"- reference_answer: `{_fmt(example.get('reference_answer'))}`",
                    f"- extracted_answer: `{_fmt(example.get('extracted_answer'))}`",
                    f"- flexible_correct: `{_fmt(example.get('flexible_correct'))}`",
                    f"- exact_equivalence: `{_fmt(example.get('dense_output_exact_equivalence'))}`",
                    f"- normalized_equivalence: `{_fmt(example.get('dense_output_normalized_equivalence'))}`",
                    "",
                    "method output:",
                    "",
                    "```text",
                    truncate(example.get("model_output")),
                    "```",
                    "",
                    "dense output:",
                    "",
                    "```text",
                    truncate(example.get("dense_model_output")),
                    "```",
                    "",
                ]
            )
    if not any_examples:
        lines.append("No dense-reference mismatches were found in the collected runs.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def truncate(value: Any, limit: int = 700) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated]"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root containing run directories")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--out-diffs",
        help="Dense equivalence mismatch examples markdown. Default: sibling of --out-md.",
    )
    args = parser.parse_args()

    rows = collect(Path(args.root))
    attach_dense_outputs(rows)
    write_csv(Path(args.out_csv), rows)
    write_markdown(Path(args.out_md), rows)
    diff_path = Path(args.out_diffs) if args.out_diffs else Path(args.out_md).with_name(
        "dense_equivalence_diffs.md"
    )
    write_equivalence_diffs(diff_path, rows)
    print(f"collected {len(rows)} runs")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    print(f"wrote {diff_path}")


if __name__ == "__main__":
    main()
