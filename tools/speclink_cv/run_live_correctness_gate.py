#!/usr/bin/env python3
"""Run live SpecLink-CV token-id correctness gates.

This is a thin orchestrator around ``live_correctness_smoke.py``.  It is
intended for the TODO correctness gate: run vLLM+EAGLE3 one-shot and live
SpecLink-CV for the same prompts, then record whether token IDs match.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = repo_root()
EVAL_DIR = REPO_ROOT / "examples/evaluate/eval-guidellm"
MODELS_ROOT = REPO_ROOT.parent / "models"


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def zero_or_empty(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return abs(float(str(value).strip())) == 0.0
    except (TypeError, ValueError):
        return False


def strict_greedy_row(row: dict[str, Any]) -> bool:
    return zero_or_empty(row.get("greedy_eps")) and zero_or_empty(
        row.get("draft_accept_eps")
    )


def strict_matched_row(row: dict[str, Any]) -> bool:
    return strict_greedy_row(row) and str(row.get("matched")) == "True"


def strict_failure_reason(row: dict[str, Any]) -> str:
    if not strict_greedy_row(row):
        return "non_strict_eps"
    if str(row.get("matched")) != "True":
        return "token_mismatch"
    return ""


def model_registry() -> dict[str, dict[str, str]]:
    return {
        "qwen3_8b": {
            "base": env_default("QWEN3_8B_MODEL", "Qwen/Qwen3-8B"),
            "speculator": env_default(
                "QWEN3_8B_EAGLE3_SPECULATOR_MODEL",
                env_default(
                    "EAGLE3_SPECULATOR_MODEL",
                    str(MODELS_ROOT / "qwen3-8b-eagle3-speculator"),
                ),
            ),
        },
        "llama3_1_8b": {
            "base": env_default(
                "LLAMA3_1_8B_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
            ),
            "speculator": env_default(
                "LLAMA3_1_8B_EAGLE3_SPECULATOR_MODEL",
                env_default(
                    "LLAMA_EAGLE3_SPECULATOR_MODEL",
                    str(MODELS_ROOT / "llama-3.1-8b-eagle3-speculator"),
                ),
            ),
        },
    }


def dataset_registry() -> dict[str, Path]:
    return {
        "math": Path(env_default("MATH_DATASET", str(EVAL_DIR / "data/math_reasoning.jsonl"))),
        "mtbench": Path(env_default("MTBENCH_DATASET", str(EVAL_DIR / "data/mt_bench.jsonl"))),
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--datasets", default="math,mtbench")
    parser.add_argument("--ks", default="8")
    parser.add_argument(
        "--modes",
        default="chunked,exactsafe",
        help=(
            "Comma-separated gate modes. chunked enables h<K live chunking; "
            "chunked_confirm also confirms prefix rejects with full-K one-shot; "
            "chunked_confirm_all confirms both prefix accepts and rejects; "
            "chunked_recompute rolls back h<K committed-token KV so the "
            "next TLM step recomputes committed tokens; "
            "chunked_low_margin adds a verifier margin fallback; "
            "chunked_reject_confirm_low_margin combines reject confirmation "
            "with the margin fallback; "
            "chunked_reject_confirm_low_margin_isolated also forces "
            "conservative non-batched prefix scheduling; "
            "chunked_confirm_all_barrier confirms both accepts/rejects and "
            "requires all running requests to enter the same prefix batch; "
            "chunked_grouped_batchwide_prefixreject_barrier falls back every "
            "row in a rejecting prefix batch to grouped full-K confirmation; "
            "chunked_confirm_all_rollback_barrier combines confirm-all, "
            "global barrier, block rollback, full-active-set confirmation, "
            "and lockstep iteration barriers for a strict diagnostic; "
            "exactsafe uses the default shape-drift guard."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--batch-sizes",
        default="",
        help=(
            "Optional comma-separated max_num_seqs sweep. When set, this "
            "overrides --batch-size, e.g. '8,16,32'."
        ),
    )
    parser.add_argument(
        "--num-prompts-per-batch",
        action="store_true",
        help=(
            "Use num_prompts=batch_size for each case. This helps make the "
            "actual request pressure match a batch-size sweep."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--chat-completions", action="store_true", default=True)
    parser.add_argument("--no-chat-completions", dest="chat_completions", action="store_false")
    parser.add_argument("--async-queue", action="store_true", default=True)
    parser.add_argument("--sync-mode", dest="async_queue", action="store_false")
    parser.add_argument(
        "--allow-batched-prefix-verification",
        action="store_true",
        help=(
            "Forward live_correctness_smoke.py's experimental batched-prefix "
            "mode instead of the conservative one-prefix cap."
        ),
    )
    parser.add_argument(
        "--global-batch-barrier",
        action="store_true",
        help="Forward live_correctness_smoke.py's global batch barrier flag.",
    )
    parser.add_argument(
        "--prefix-low-margin-threshold",
        type=float,
        default=0.5,
        help="Threshold used by chunked_*_low_margin modes.",
    )
    parser.add_argument(
        "--dense-realign-steps",
        type=int,
        default=None,
        help=(
            "Forward live_correctness_smoke.py's "
            "--dense-realign-steps diagnostic override."
        ),
    )
    parser.add_argument(
        "--prefix-reject-dense-realign-steps",
        type=int,
        default=None,
        help=(
            "Forward live_correctness_smoke.py's "
            "--prefix-reject-dense-realign-steps diagnostic override."
        ),
    )
    parser.add_argument(
        "--prefix-no-kv-write",
        action="store_true",
        help=(
            "Forward live_correctness_smoke.py's prefix probe no-KV-write "
            "diagnostic override."
        ),
    )
    parser.add_argument(
        "--confirmation-full-active-set",
        action="store_true",
        help=(
            "Forward live_correctness_smoke.py's full-active-set "
            "confirmation diagnostic override."
        ),
    )
    parser.add_argument(
        "--lockstep-iteration-barrier",
        action="store_true",
        help=(
            "Forward live_correctness_smoke.py's lockstep speculative "
            "iteration barrier diagnostic override."
        ),
    )
    parser.add_argument(
        "--prefix-probe-block-rollback",
        action="store_true",
        help=(
            "Forward live_correctness_smoke.py's prefix-probe block-table "
            "rollback diagnostic override."
        ),
    )
    parser.add_argument(
        "--draft-accept-eps",
        type=float,
        default=None,
        help=(
            "Forward live_correctness_smoke.py's "
            "--draft-accept-eps diagnostic override."
        ),
    )
    parser.add_argument(
        "--greedy-eps",
        type=float,
        default=None,
        help=(
            "Forward live_correctness_smoke.py's --greedy-eps diagnostic "
            "override."
        ),
    )
    parser.add_argument(
        "--candidate-chunks",
        default="",
        help="Optional candidate chunk override forwarded to the live smoke.",
    )
    parser.add_argument(
        "--force-prefix-len",
        type=int,
        default=0,
        help="Optional forced prefix length forwarded to the live smoke.",
    )
    parser.add_argument("--debug-dump", action="store_true")
    parser.add_argument(
        "--profile-max-events",
        type=int,
        default=500,
        help=(
            "Maximum regular profile JSONL rows forwarded to each live "
            "smoke. One limit marker may be added. Use 0 only for a short "
            "trusted diagnostic."
        ),
    )
    parser.add_argument(
        "--log-max-events",
        type=int,
        default=1_000,
        help=(
            "Maximum regular verbose event JSONL rows forwarded to each live "
            "smoke. One limit marker may be added. Use 0 only for a short "
            "trusted diagnostic."
        ),
    )
    parser.add_argument(
        "--kv-debug-tail-tokens",
        type=int,
        default=0,
        help=(
            "Forward bounded KV checksum window dumps to the live smoke. "
            "Current vLLM debug events include both seq-tail and pre-target "
            "history windows."
        ),
    )
    parser.add_argument(
        "--kv-debug-max-layers",
        type=int,
        default=0,
        help="Forward bounded KV checksum layer dumps to the live smoke.",
    )
    parser.add_argument(
        "--kv-debug-row-index",
        type=int,
        default=-1,
        help="Forward a row-index filter for KV checksum dumps.",
    )
    parser.add_argument(
        "--kv-debug-min-output-tokens",
        type=int,
        default=-1,
        help="Forward a lower output-token-count filter for KV checksum dumps.",
    )
    parser.add_argument(
        "--kv-debug-max-output-tokens",
        type=int,
        default=-1,
        help="Forward an upper output-token-count filter for KV checksum dumps.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help=(
            "Extra KEY=VALUE environment variable forwarded to "
            "live_correctness_smoke.py. Repeatable. Use this for exact-mode "
            "toggles such as VLLM_BATCH_INVARIANT=1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write per-case configs/commands and summary rows without running vLLM.",
    )
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Skip this many planned cases before running/analyzing.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Run/analyze at most this many cases after --case-offset; 0 means no limit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EVAL_DIR / "temp" / f"speclink_cv_live_correctness_gate_{timestamp()}",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Record failed cases and continue. Enabled by default.",
    )
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help=(
            "Return non-zero when any recorded row fails. By default this "
            "gate records failures but exits zero so sliced long runs can "
            "continue collecting later cases."
        ),
    )
    return parser.parse_args()


def make_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    models = model_registry()
    datasets = dataset_registry()
    batch_sizes = (
        [int(item) for item in split_csv(args.batch_sizes)]
        if args.batch_sizes
        else [int(args.batch_size)]
    )
    cases: list[dict[str, Any]] = []
    for model_label in split_csv(args.models):
        if model_label not in models:
            raise SystemExit(f"unknown model label: {model_label}")
        for dataset_label in split_csv(args.datasets):
            if dataset_label not in datasets:
                raise SystemExit(f"unknown dataset label: {dataset_label}")
            for k_raw in split_csv(args.ks):
                k = int(k_raw)
                for batch_size in batch_sizes:
                    num_prompts = (
                        batch_size
                        if args.num_prompts_per_batch
                        else int(args.num_prompts)
                    )
                    for mode in split_csv(args.modes):
                        if mode not in {
                            "chunked",
                            "chunked_confirm",
                            "chunked_confirm_all",
                            "chunked_recompute",
                            "chunked_low_margin",
                            "chunked_reject_confirm_low_margin",
                            "chunked_reject_confirm_low_margin_isolated",
                            "chunked_confirm_all_barrier",
                            "chunked_grouped_batchwide_prefixreject_barrier",
                            "chunked_confirm_all_rollback_barrier",
                            "exactsafe",
                        }:
                            raise SystemExit(f"unknown gate mode: {mode}")
                        cases.append(
                            {
                                "model_label": model_label,
                                "dataset_label": dataset_label,
                                "K": k,
                                "mode": mode,
                                "base_model": models[model_label]["base"],
                                "speculator_model": models[model_label]["speculator"],
                                "dataset_path": datasets[dataset_label],
                                "batch_size": batch_size,
                                "num_prompts": num_prompts,
                                "max_tokens": args.max_tokens,
                            }
                        )
    return cases


def run_name_for_case(index: int, case: dict[str, Any]) -> str:
    return (
        f"{index:03d}_{case['model_label']}_{case['dataset_label']}_"
        f"k{case['K']}_bs{case['batch_size']}_{case['mode']}"
    )


def command_for_case(case: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/speclink_cv/live_correctness_smoke.py"),
        "--model",
        case["base_model"],
        "--speculator-model",
        case["speculator_model"],
        "--num-spec-tokens",
        str(case["K"]),
        "--prompts-jsonl",
        str(case["dataset_path"]),
        "--prompt-field",
        args.prompt_field,
        "--num-prompts",
        str(case["num_prompts"]),
        "--max-num-seqs",
        str(case["batch_size"]),
        "--max-tokens",
        str(args.max_tokens),
        "--output-json",
        str(run_dir / "correctness.json"),
        "--event-jsonl",
        str(run_dir / "speclink_cv_events.jsonl"),
        "--baseline-event-jsonl",
        str(run_dir / "baseline_events.jsonl"),
        "--profile-jsonl",
        str(run_dir / "speclink_cv_profile.jsonl"),
        "--profile-max-events",
        str(args.profile_max_events),
        "--log-max-events",
        str(args.log_max_events),
    ]
    if args.chat_completions:
        cmd.append("--chat-completions")
    if args.async_queue:
        cmd.append("--async-queue")
    if (
        args.allow_batched_prefix_verification
        and case["mode"] != "chunked_reject_confirm_low_margin_isolated"
    ):
        cmd.append("--allow-batched-prefix-verification")
    if args.global_batch_barrier:
        cmd.append("--global-batch-barrier")
    if args.debug_dump:
        cmd.append("--debug-dump")
    if args.kv_debug_tail_tokens:
        cmd.extend(["--kv-debug-tail-tokens", str(args.kv_debug_tail_tokens)])
    if args.kv_debug_max_layers:
        cmd.extend(["--kv-debug-max-layers", str(args.kv_debug_max_layers)])
    if args.kv_debug_row_index >= 0:
        cmd.extend(["--kv-debug-row-index", str(args.kv_debug_row_index)])
    if args.kv_debug_min_output_tokens >= 0:
        cmd.extend(
            [
                "--kv-debug-min-output-tokens",
                str(args.kv_debug_min_output_tokens),
            ]
        )
    if args.kv_debug_max_output_tokens >= 0:
        cmd.extend(
            [
                "--kv-debug-max-output-tokens",
                str(args.kv_debug_max_output_tokens),
            ]
        )
    if args.candidate_chunks:
        cmd.extend(["--candidate-chunks", args.candidate_chunks])
    if args.force_prefix_len:
        cmd.extend(["--force-prefix-len", str(args.force_prefix_len)])
    if args.dense_realign_steps is not None:
        cmd.extend(["--dense-realign-steps", str(args.dense_realign_steps)])
    if args.prefix_reject_dense_realign_steps is not None:
        cmd.extend(
            [
                "--prefix-reject-dense-realign-steps",
                str(args.prefix_reject_dense_realign_steps),
            ]
        )
    if args.prefix_no_kv_write:
        cmd.append("--prefix-no-kv-write")
    if args.confirmation_full_active_set:
        cmd.append("--confirmation-full-active-set")
    if args.lockstep_iteration_barrier:
        cmd.append("--lockstep-iteration-barrier")
    if args.prefix_probe_block_rollback:
        cmd.append("--prefix-probe-block-rollback")
    if args.draft_accept_eps is not None:
        cmd.extend(["--draft-accept-eps", str(args.draft_accept_eps)])
    if args.greedy_eps is not None:
        cmd.extend(["--greedy-eps", str(args.greedy_eps)])
    if case["mode"].startswith("chunked"):
        cmd.append("--allow-shape-drift-chunking")
    if case["mode"] in {
        "chunked_confirm",
        "chunked_confirm_all",
        "chunked_reject_confirm_low_margin",
        "chunked_reject_confirm_low_margin_isolated",
    }:
        cmd.append("--confirm-prefix-reject-one-shot")
    if case["mode"] == "chunked_confirm_all":
        cmd.append("--confirm-prefix-accept-one-shot")
    if case["mode"] == "chunked_recompute":
        cmd.append("--recompute-committed-prefix")
    if case["mode"] == "chunked_confirm_all_barrier":
        cmd.extend(
            [
                "--allow-batched-prefix-verification",
                "--global-batch-barrier",
                "--confirm-prefix-reject-one-shot",
                "--confirm-prefix-accept-one-shot",
            ]
        )
    if case["mode"] in {
        "chunked_grouped_batchwide_prefixreject_barrier",
    }:
        cmd.extend(
            [
                "--allow-batched-prefix-verification",
                "--global-batch-barrier",
                "--batch-wide-prefix-reject-fallback",
            ]
        )
    if case["mode"] == "chunked_confirm_all_rollback_barrier":
        cmd.extend(
            [
                "--allow-batched-prefix-verification",
                "--global-batch-barrier",
                "--confirm-prefix-reject-one-shot",
                "--confirm-prefix-accept-one-shot",
                "--prefix-probe-block-rollback",
                "--confirmation-full-active-set",
                "--lockstep-iteration-barrier",
            ]
        )
    if "low_margin" in case["mode"]:
        cmd.extend(
            [
                "--prefix-low-margin-fallback-threshold",
                str(args.prefix_low_margin_threshold),
            ]
        )
    for item in args.env:
        cmd.extend(["--env", item])
    return cmd


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
        for row in rows:
            writer.writerow(row)


def run_command_with_progress(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    label: str,
) -> int:
    start = time.monotonic()
    print(f"[INFO] start {label}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            ready, _, _ = select.select([fd], [], [], 30.0)
            if ready:
                line = proc.stdout.readline()
                if line:
                    out.write(line)
                    out.flush()
                    if (
                        line.startswith("[live_correctness_smoke]")
                        or line.startswith("[INFO]")
                        or line.startswith("[ERROR]")
                    ):
                        print(line.rstrip(), flush=True)
                    continue
            if proc.poll() is not None:
                for line in proc.stdout:
                    out.write(line)
                    if (
                        line.startswith("[live_correctness_smoke]")
                        or line.startswith("[INFO]")
                        or line.startswith("[ERROR]")
                    ):
                        print(line.rstrip(), flush=True)
                break
            elapsed = time.monotonic() - start
            print(f"[INFO] still running {label} elapsed_s={elapsed:.1f}", flush=True)
        returncode = proc.wait()
    elapsed = time.monotonic() - start
    print(
        f"[INFO] end {label} returncode={returncode} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return returncode


def summarize_result(case: dict[str, Any], run_dir: Path, returncode: int) -> dict[str, Any]:
    result_path = run_dir / "correctness.json"
    row: dict[str, Any] = {
        "model": case["model_label"],
        "dataset": case["dataset_label"],
        "K": case["K"],
        "mode": case["mode"],
        "batch_size": case.get("batch_size", ""),
        "num_prompts": case.get("num_prompts", ""),
        "max_tokens": case.get("max_tokens", ""),
        "greedy_eps": case.get("greedy_eps", ""),
        "draft_accept_eps": case.get("draft_accept_eps", ""),
        "profile_max_events": case.get("profile_max_events", ""),
        "log_max_events": case.get("log_max_events", ""),
        "status": "ok" if returncode == 0 else "fail",
        "returncode": returncode,
        "matched": "",
        "matched_count": "",
        "total_count": "",
        "first_mismatch_index": "",
        "first_mismatch_token_index": "",
        "output_json": str(result_path),
        "output_dir": str(run_dir),
    }
    if not result_path.exists():
        row["status"] = "crash"
        return row
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        row["status"] = "bad_json"
        row["error"] = repr(exc)
        return row

    matched_items = list(data.get("matched_items") or [])
    mismatches = list(data.get("mismatches") or [])
    row.update(
        {
            "batch_size": case.get("batch_size", ""),
            "num_prompts": data.get("num_prompts", case.get("num_prompts", "")),
            "max_tokens": data.get("max_tokens", case.get("max_tokens", "")),
            "greedy_eps": data.get("greedy_eps", case.get("greedy_eps", "")),
            "draft_accept_eps": data.get(
                "draft_accept_eps", case.get("draft_accept_eps", "")
            ),
            "matched": bool(data.get("matched")),
            "matched_count": sum(1 for item in matched_items if item),
            "total_count": len(matched_items),
        }
    )
    row["strict_greedy"] = strict_greedy_row(row)
    row["strict_matched"] = strict_matched_row(row)
    row["strict_failure_reason"] = strict_failure_reason(row)
    if mismatches:
        first = mismatches[0]
        row["first_mismatch_index"] = first.get("index", "")
        row["first_mismatch_token_index"] = first.get("first_diff_index", "")
    return row


def collect_existing_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "model_label" not in config or "dataset_label" not in config:
            continue
        returncode = 0 if (run_dir / "correctness.json").exists() else 1
        row = summarize_result(config, run_dir, returncode)
        rows.append(row)
    return rows


def write_report(
    root: Path,
    rows: list[dict[str, Any]],
    extra_env: list[str],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# SpecLink-CV Live Correctness Gate",
        "",
        "This report compares live SpecLink-CV output token IDs against a",
        "vLLM+EAGLE3 one-shot baseline using `live_correctness_smoke.py`.",
        "",
    ]
    if extra_env:
        lines.extend(
            [
                "Extra environment:",
                "",
                *[f"- `{item}`" for item in extra_env],
                "",
            ]
        )
    diagnostic_items = [
        ("greedy_eps", args.greedy_eps),
        ("draft_accept_eps", args.draft_accept_eps),
        ("profile_max_events", args.profile_max_events),
        ("log_max_events", args.log_max_events),
        ("kv_debug_tail_tokens", args.kv_debug_tail_tokens),
        ("kv_debug_max_layers", args.kv_debug_max_layers),
        ("kv_debug_row_index", args.kv_debug_row_index),
        ("kv_debug_min_output_tokens", args.kv_debug_min_output_tokens),
        ("kv_debug_max_output_tokens", args.kv_debug_max_output_tokens),
        ("prefix_probe_block_rollback", args.prefix_probe_block_rollback),
    ]
    lines.extend(
        [
            "Diagnostic arguments:",
            "",
            *[f"- `{key}={value}`" for key, value in diagnostic_items],
            "",
        ]
    )
    if any(
        row.get("mode") == "chunked_confirm_all_rollback_barrier"
        for row in rows
    ):
        lines.extend(
            [
                "Mode-specific diagnostics:",
                "",
                "- `chunked_confirm_all_rollback_barrier` implies `--allow-batched-prefix-verification`, `--global-batch-barrier`, `--confirm-prefix-reject-one-shot`, `--confirm-prefix-accept-one-shot`, `--prefix-probe-block-rollback`, `--confirmation-full-active-set`, and `--lockstep-iteration-barrier`.",
                "",
            ]
        )
    lines.extend(
        [
            "| model | dataset | K | batch | prompts | mode | greedy_eps | strict greedy | matched | strict match | reason | matched_count | total | status | first mismatch token |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('model', '')} | {row.get('dataset', '')} | "
            f"{row.get('K', '')} | {row.get('batch_size', '')} | "
            f"{row.get('num_prompts', '')} | {row.get('mode', '')} | "
            f"{row.get('greedy_eps', '')} | {row.get('strict_greedy', '')} | "
            f"{row.get('matched', '')} | {row.get('strict_matched', '')} | "
            f"{row.get('strict_failure_reason', '')} | "
            f"{row.get('matched_count', '')} | {row.get('total_count', '')} | "
            f"{row.get('status', '')} | "
            f"{row.get('first_mismatch_token_index', '')} |"
        )
    lines.extend(
        [
            "",
            "`mode=chunked` enables experimental h<K live chunking and is the",
            "actual SpecLink-CV correctness gate. `mode=chunked_confirm` also",
            "confirms prefix rejects with a full-K one-shot fallback before",
            "committing tokens; it is a correctness diagnostic, not a speedup",
            "claim. `mode=chunked_confirm_all` additionally confirms fully",
            "accepted prefixes with a full-K one-shot fallback before committing",
            "tokens, which isolates accepted-prefix boundary drift. The",
            "`chunked_*_low_margin` modes discard prefix outputs and requeue",
            "full-K confirmation when the verifier top-1/top-2 margin is below",
            "the configured threshold. `mode=chunked_confirm_all_barrier`",
            "only dispatches h<K prefix chunks when every running request can",
            "enter the same prefix batch. "
            "`mode=chunked_grouped_batchwide_prefixreject_barrier` requeues",
            "every row in a prefix batch for grouped full-K confirmation when",
            "any row rejects; it is a correctness-recovery diagnostic for the",
            "batch-wide fallback path. "
            "`mode=chunked_confirm_all_rollback_barrier` is the most",
            "conservative h<K diagnostic: it discards prefix outputs, rolls",
            "back prefix-probe blocks, groups full-active-set confirmations,",
            "and uses lockstep barriers. A failure in this mode is strong",
            "evidence that h<K has already changed the later verifier/KV or",
            "numerical trajectory.",
            "`mode=exactsafe` leaves the",
            "shape-drift guard enabled; it should be exact but is not a chunked",
            "speedup claim because it falls back to one-shot verification.",
            "`greedy_eps` is a diagnostic near-tie guard for",
            "`VLLM_BATCH_INVARIANT=1`; do not treat it as a performance",
            "optimization or a proof of exact chunked-verification semantics.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases_all = make_cases(args)
    if args.case_offset < 0:
        raise SystemExit("--case-offset must be >= 0")
    if args.case_limit < 0:
        raise SystemExit("--case-limit must be >= 0")
    case_end = None if args.case_limit == 0 else args.case_offset + args.case_limit
    cases = cases_all[args.case_offset:case_end]
    rows: list[dict[str, Any]] = []
    command_lines: list[str] = []
    for index, case in enumerate(cases):
        absolute_index = args.case_offset + index
        run_name = run_name_for_case(absolute_index, case)
        run_dir = args.output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = command_for_case(case, args, run_dir)
        command_lines.append(subprocess.list2cmdline(cmd))
        (run_dir / "command.sh").write_text(
            subprocess.list2cmdline(cmd) + "\n", encoding="utf-8"
        )
        config = {
            **case,
            "extra_env": list(args.env),
            "greedy_eps": args.greedy_eps,
            "draft_accept_eps": args.draft_accept_eps,
            "profile_max_events": args.profile_max_events,
            "log_max_events": args.log_max_events,
            "kv_debug_tail_tokens": args.kv_debug_tail_tokens,
            "kv_debug_max_layers": args.kv_debug_max_layers,
            "kv_debug_row_index": args.kv_debug_row_index,
            "kv_debug_min_output_tokens": args.kv_debug_min_output_tokens,
            "kv_debug_max_output_tokens": args.kv_debug_max_output_tokens,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        if args.dry_run:
            (run_dir / "stdout.log").write_text(
                "[DRY RUN] command was not executed.\n", encoding="utf-8"
            )
            row = {
                "model": case["model_label"],
                "dataset": case["dataset_label"],
                "K": case["K"],
                "mode": case["mode"],
                "batch_size": case.get("batch_size", ""),
                "num_prompts": case.get("num_prompts", ""),
                "max_tokens": case.get("max_tokens", ""),
                "greedy_eps": args.greedy_eps,
                "draft_accept_eps": args.draft_accept_eps,
                "profile_max_events": args.profile_max_events,
                "log_max_events": args.log_max_events,
                "status": "dry_run",
                "returncode": "",
                "matched": "",
                "strict_greedy": "",
                "strict_matched": "",
                "strict_failure_reason": "",
                "matched_count": "",
                "total_count": "",
                "first_mismatch_index": "",
                "first_mismatch_token_index": "",
                "output_json": str(run_dir / "correctness.json"),
                "output_dir": str(run_dir),
            }
        else:
            returncode = run_command_with_progress(
                cmd,
                cwd=REPO_ROOT,
                stdout_path=run_dir / "stdout.log",
                label=run_name,
            )
            summary_case = {
                **case,
                "greedy_eps": args.greedy_eps,
                "draft_accept_eps": args.draft_accept_eps,
                "profile_max_events": args.profile_max_events,
                "log_max_events": args.log_max_events,
            }
            row = summarize_result(summary_case, run_dir, returncode)
        rows.append(row)
        write_csv(args.output_root / "summary_current_slice.csv", rows)
        all_rows = rows if args.dry_run else collect_existing_rows(args.output_root)
        write_csv(args.output_root / "summary.csv", all_rows)
        (args.output_root / "summary.json").write_text(
            json.dumps(
                {
                    "case_offset": args.case_offset,
                    "case_limit": args.case_limit,
                    "selected_cases": len(cases),
                    "total_planned_cases": len(cases_all),
                    "rows": all_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_report(args.output_root, all_rows, args.env, args)
        if (
            not args.dry_run
            and row.get("returncode") != 0
            and not args.continue_on_error
        ):
            break

    (args.output_root / "commands.sh").write_text(
        "\n".join(command_lines) + "\n", encoding="utf-8"
    )
    all_rows = rows if args.dry_run else collect_existing_rows(args.output_root)
    write_csv(args.output_root / "summary_current_slice.csv", rows)
    write_csv(args.output_root / "summary.csv", all_rows)
    (args.output_root / "summary.json").write_text(
        json.dumps(
            {
                "case_offset": args.case_offset,
                "case_limit": args.case_limit,
                "selected_cases": len(cases),
                "total_planned_cases": len(cases_all),
                "rows": all_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(args.output_root, all_rows, args.env, args)
    failures = [
        row
        for row in all_rows
        if row.get("status") != "dry_run" and row.get("matched") is not True
    ]
    print(f"[INFO] wrote correctness gate results to {args.output_root}")
    print(
        f"[INFO] cases={len(all_rows)} selected_cases={len(cases)} "
        f"total_planned_cases={len(cases_all)} failures={len(failures)}"
    )
    return 1 if failures and args.strict_exit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
