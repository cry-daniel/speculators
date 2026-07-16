#!/usr/bin/env python3
"""Token-dense fixed-budget accuracy sweep under vLLM EAGLE3 serving."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = EVAL_ROOT / "results"
RESULTS_BAK_ROOT = EVAL_ROOT / "results.bak"

sys.path.insert(0, str(SCRIPT_DIR))
from residual_24_feasibility import (  # noqa: E402
    DEFAULT_C4_CALIBRATION_CACHE_ROOT,
    LAYER_SENSITIVITY_DEFAULT_MODELS,
    ensure_quality_dependencies,
    load_datasets,
    metric_value,
    parse_csv_list,
    parse_model_id_overrides,
    set_seed,
    write_json,
)
from run_structured_24_spec_quality import (  # noqa: E402
    ACCURACY_DATASETS,
    DEFAULT_BASE_MODELS,
    EAGLE3_SPECULATORS,
    SparseCase,
    configure_local_no_proxy,
    metric_delta,
    run_accuracy_dataset,
    scrape_spec_metrics,
    start_vllm_server,
    stop_process,
)
from token_dense_methods import (  # noqa: E402
    DEFAULT_TOKEN_DENSE_METHODS,
    DEFAULT_TOKEN_DENSE_MASK_ROOT,
    MethodConfig,
    method_env,
    parse_method_config,
    timestamp,
)


CSV_FIELDS = [
    "model_label",
    "model_id",
    "method",
    "dataset",
    "metric_name",
    "metric_type",
    "dense_metric_value",
    "sparse_metric_value",
    "delta_vs_dense",
    "accuracy_drop",
    "num_examples",
    "output_tokens",
    "spec_acceptance_rate_case",
    "spec_accepted_tokens_case",
    "spec_draft_tokens_case",
    "effective_sparse_fraction",
    "token_dense_budget",
    "token_dense_linear_strategy",
    "token_dense_mlp_strategy",
    "token_dense_sparse_value_scale",
    "token_dense_row_scale_mode",
    "token_dense_row_scale_max",
    "token_dense_sparse_output_mode",
    "token_dense_sparse_accumulator",
    "token_dense_dense_draft_fraction",
    "token_dense_sparse_draft_fraction",
    "token_dense_scored_draft_tokens",
    "token_dense_forced_dense_tokens",
    "token_dense_missing_score_tokens",
    "cutlass_sparse24_dynamic_backend_requested",
    "cutlass_sparse24_dynamic_backend_enabled",
    "cutlass_sparse24_dynamic_prepack_module_count",
    "cutlass_sparse24_skipped_module_count",
    "failed",
    "error",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_selected_datasets(args: argparse.Namespace) -> dict[str, Any]:
    selected = [name for name in parse_csv_list(args.datasets) if name in ACCURACY_DATASETS]
    if not selected:
        raise RuntimeError("select at least one accuracy dataset: gsm8k, math_reasoning")
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.datasets = ",".join(selected)
    return load_datasets(dataset_args)


def metrics_complete(case_dir: Path, dataset_names: set[str]) -> bool:
    path = case_dir / "metrics.json"
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    datasets = data.get("datasets")
    return isinstance(datasets, dict) and dataset_names.issubset(datasets.keys())


def summarize_token_dense_stats(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "verify_token_mask_summary":
                latest = record
    return latest


def run_method(
    args: argparse.Namespace,
    *,
    model_label: str,
    model_id: str,
    speculator_model: str,
    method: MethodConfig,
    datasets: dict[str, Any],
    case_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_names = set(datasets)
    if args.resume and metrics_complete(case_dir, dataset_names):
        data = load_json(case_dir / "metrics.json")
        mask_stats = data.get("mask_stats", {})
        if isinstance(mask_stats, dict):
            mask_stats["token_dense_stats"] = data.get("token_dense_stats", {})
        return data.get("datasets", {}), mask_stats

    case_dir.mkdir(parents=True, exist_ok=True)
    stats_path = case_dir / "vllm_structured_24_stats.json"
    env = method_env(args, model_label=model_label, method=method, stats_path=stats_path)
    previous_enforce_eager = args.enforce_eager
    previous_token_dense_active = getattr(args, "token_dense_active", False)
    args.token_dense_active = method.base_method == "token_dense"
    if method.base_method == "token_dense" and args.token_dense_enforce_eager:
        args.enforce_eager = True

    process = None
    metrics: dict[str, Any] = {}
    mask_stats: dict[str, Any] = {}
    try:
        process, port = start_vllm_server(
            args,
            base_model=model_id,
            speculator_model=speculator_model,
            case_dir=case_dir,
            env=env,
        )
        before = scrape_spec_metrics(port)
        sparse_case = SparseCase(method.label, "token_dense_accuracy", method.policy, keep_n=method.keep_n)
        for dataset_name, pack in datasets.items():
            metrics[dataset_name] = run_accuracy_dataset(
                args,
                port=port,
                model_id=model_id,
                model_label=model_label,
                case=sparse_case,
                dataset_name=dataset_name,
                pack=pack,
                case_dir=case_dir,
            )
            write_json(case_dir / "accuracy_metrics.partial.json", metrics)
        after = scrape_spec_metrics(port)
        accepted = metric_delta(before, after, "vllm:spec_decode_num_accepted_tokens")
        drafted = metric_delta(before, after, "vllm:spec_decode_num_draft_tokens")
        for metric in metrics.values():
            metric["spec_accepted_tokens_case"] = accepted
            metric["spec_draft_tokens_case"] = drafted
            metric["spec_acceptance_rate_case"] = accepted / drafted if accepted is not None and drafted else None
    finally:
        args.enforce_eager = previous_enforce_eager
        args.token_dense_active = previous_token_dense_active
        stop_process(process)
        if args.server_shutdown_settle_s > 0:
            time.sleep(args.server_shutdown_settle_s)

    if stats_path.exists():
        try:
            mask_stats = load_json(stats_path)
        except Exception:
            mask_stats = {}
    token_dense_stats = summarize_token_dense_stats(case_dir / "token_dense_stats.jsonl")
    write_json(
        case_dir / "metrics.json",
        {
            "method": method.label,
            "datasets": metrics,
            "mask_stats": mask_stats,
            "token_dense_stats": token_dense_stats,
        },
    )
    mask_stats["token_dense_stats"] = token_dense_stats
    return metrics, mask_stats


def make_rows(
    *,
    args: argparse.Namespace,
    model_label: str,
    model_id: str,
    method: MethodConfig,
    dense_metrics: dict[str, Any],
    method_metrics: dict[str, Any],
    mask_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_dense_stats = mask_stats.get("token_dense_stats", {})
    for dataset_name, sparse_metric in method_metrics.items():
        dense_metric = dense_metrics.get(dataset_name, {})
        dense_value = metric_value(dense_metric)
        sparse_value = metric_value(sparse_metric)
        delta = sparse_value - dense_value if dense_value is not None and sparse_value is not None else None
        accuracy_drop = dense_value - sparse_value if dense_value is not None and sparse_value is not None else None
        rows.append(
            {
                "model_label": model_label,
                "model_id": model_id,
                "method": method.label,
                "dataset": dataset_name,
                "metric_name": sparse_metric.get("metric_name", dense_metric.get("metric_name", "")),
                "metric_type": sparse_metric.get("metric_type", dense_metric.get("metric_type", "")),
                "dense_metric_value": dense_value,
                "sparse_metric_value": sparse_value,
                "delta_vs_dense": delta,
                "accuracy_drop": accuracy_drop,
                "num_examples": sparse_metric.get("num_examples", dense_metric.get("num_examples", 0)),
                "output_tokens": sparse_metric.get("output_tokens", 0),
                "spec_acceptance_rate_case": sparse_metric.get("spec_acceptance_rate_case"),
                "spec_accepted_tokens_case": sparse_metric.get("spec_accepted_tokens_case"),
                "spec_draft_tokens_case": sparse_metric.get("spec_draft_tokens_case"),
                "effective_sparse_fraction": mask_stats.get("effective_sparse_fraction"),
                "token_dense_budget": method.token_dense_budget,
                "token_dense_linear_strategy": token_dense_stats.get(
                    "linear_strategy",
                    mask_stats.get("speclink_kernel_linear_strategy"),
                ),
                "token_dense_mlp_strategy": token_dense_stats.get("mlp_strategy"),
                "token_dense_sparse_value_scale": (
                    args.token_dense_sparse_value_scale
                ),
                "token_dense_row_scale_mode": args.token_dense_row_scale_mode,
                "token_dense_row_scale_max": args.token_dense_row_scale_max,
                "token_dense_sparse_output_mode": (
                    args.token_dense_sparse_output_mode
                ),
                "token_dense_sparse_accumulator": (
                    args.token_dense_sparse_accumulator
                ),
                "token_dense_dense_draft_fraction": token_dense_stats.get("dense_draft_fraction"),
                "token_dense_sparse_draft_fraction": token_dense_stats.get("sparse_draft_fraction"),
                "token_dense_scored_draft_tokens": token_dense_stats.get(
                    "scored_draft_tokens"
                ),
                "token_dense_forced_dense_tokens": token_dense_stats.get(
                    "forced_dense_tokens"
                ),
                "token_dense_missing_score_tokens": token_dense_stats.get("missing_score_tokens"),
                "cutlass_sparse24_dynamic_backend_requested": mask_stats.get(
                    "cutlass_sparse24_dynamic_backend_requested"
                ),
                "cutlass_sparse24_dynamic_backend_enabled": mask_stats.get(
                    "cutlass_sparse24_dynamic_backend_enabled"
                ),
                "cutlass_sparse24_dynamic_prepack_module_count": mask_stats.get(
                    "cutlass_sparse24_dynamic_prepack_module_count"
                ),
                "cutlass_sparse24_skipped_module_count": len(
                    mask_stats.get("cutlass_sparse24_skipped_modules") or []
                ),
                "failed": bool(sparse_metric.get("failed", False)),
                "error": sparse_metric.get("error", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    report = output_root / "summary.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Token-Dense Accuracy Sweep\n\n")
        handle.write(f"Output root: `{output_root}`\n\n")
        handle.write("## Method\n\n")
        handle.write("- Serving uses vLLM + EAGLE3 speculative decoding.\n")
        handle.write("- Only the target/base model is masked; the EAGLE3 drafter remains dense.\n")
        handle.write("- `activation_aware` is the C4 activation-RMS 2:4 baseline.\n")
        handle.write("- `token_dense_dN` keeps the global Top-N scored draft verifier rows dense by per-request cumulative DLM confidence.\n")
        handle.write(f"- Token-dense linear strategy: `{args.token_dense_linear_strategy}`.\n")
        handle.write(f"- Token-dense MLP strategy: `{args.token_dense_mlp_strategy}`.\n")
        handle.write(
            "- Retained sparse value scale: "
            f"`{args.token_dense_sparse_value_scale}`.\n"
        )
        handle.write(
            "- Per-row sparse scaling: "
            f"`{args.token_dense_row_scale_mode}` "
            f"(maximum `{args.token_dense_row_scale_max}`).\n"
        )
        handle.write(
            "- Sparse output mode: "
            f"`{args.token_dense_sparse_output_mode}`.\n"
        )
        handle.write("- Sparse rows use the strict `vllm.speclink_kernel` CUTLASS MMA.SP backend; missing masks or unsupported shapes fail at load time.\n")
        handle.write("- Accuracy is measured on GSM8K and/or math_reasoning only.\n\n")
        handle.write("## Inputs\n\n")
        handle.write(f"- models: `{args.models}`\n")
        handle.write(f"- methods: `{args.methods}`\n")
        handle.write(f"- datasets: `{args.datasets}`\n")
        handle.write(f"- dtype: `{args.dtype}`\n")
        handle.write(f"- calibration RMS root: `{args.calibration_cache_root}`\n\n")
        handle.write("## Accuracy Summary\n\n")
        handle.write("| model | dataset | method | dense | sparse | delta | drop | examples |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {model_label} | {dataset} | {method} | {dense_metric_value} | "
                "{sparse_metric_value} | {delta_vs_dense} | {accuracy_drop} | {num_examples} |\n".format(
                    **row
                )
            )
        handle.write("\n## Files\n\n")
        handle.write("- `token_dense_budget_quality.csv`: unified accuracy comparison.\n")
        handle.write("- `token_dense_accuracy.csv`: accuracy-only comparison.\n")
        handle.write("- `runs/*/*/vllm_structured_24_stats.json`: vLLM-side mask proof and cache metadata.\n")


def write_outputs(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    write_csv(output_root / "token_dense_budget_quality.csv", rows)
    write_csv(output_root / "token_dense_accuracy.csv", rows)
    write_summary(output_root, rows, args)


def configure_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.models = "qwen3_8b"
    args.methods = "activation_aware,token_dense_d16,token_dense_d32"
    args.datasets = "gsm8k"
    args.gsm8k_num_examples = 1
    args.math_num_examples = 0
    args.accuracy_max_tokens = 32
    args.accuracy_concurrency = 1
    args.max_num_seqs = 1
    args.gpu_memory_utilization = max(args.gpu_memory_utilization, 0.95)
    args.output_root = args.output_root or (RESULTS_BAK_ROOT / f"token_dense_accuracy_smoke_{timestamp()}")


def run(args: argparse.Namespace) -> None:
    configure_local_no_proxy()
    configure_smoke(args)
    ensure_quality_dependencies()
    set_seed(args.seed)
    output_root = args.output_root or (RESULTS_ROOT / f"token_dense_accuracy_{timestamp()}")
    args.output_root = output_root
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = load_selected_datasets(args)

    base_models = dict(DEFAULT_BASE_MODELS)
    base_models.update(LAYER_SENSITIVITY_DEFAULT_MODELS)
    base_models.update(parse_model_id_overrides(args.model_id))
    speculators = dict(EAGLE3_SPECULATORS)
    speculators.update(parse_model_id_overrides(args.speculator_model))
    selected_models = parse_csv_list(args.models)
    methods = parse_csv_list(args.methods)
    if "activation_aware" not in methods:
        methods = ["activation_aware", *methods]
    method_configs = [parse_method_config(method) for method in methods]
    if any(method.base_method == "token_dense" for method in method_configs) and args.dtype != "fp16":
        raise ValueError("token_dense_d* methods require --dtype fp16")
    if (
        not math.isfinite(args.token_dense_sparse_value_scale)
        or args.token_dense_sparse_value_scale <= 0.0
    ):
        raise ValueError(
            "--token-dense-sparse-value-scale must be finite and positive"
        )
    if (
        not math.isfinite(args.token_dense_row_scale_max)
        or args.token_dense_row_scale_max < 1.0
    ):
        raise ValueError(
            "--token-dense-row-scale-max must be finite and at least 1.0"
        )
    if (
        any(method.base_method == "token_dense" for method in method_configs)
        and args.token_dense_linear_strategy == "full_sparse_residual"
        and (
            args.token_dense_sparse_value_scale != 1.0
            or args.token_dense_row_scale_mode == "variance"
        )
    ):
        raise ValueError(
            "sparse value scaling is incompatible with exact full_sparse_residual"
        )
    dense_config = MethodConfig(label="dense", base_method="dense", policy="dense")

    write_json(
        output_root / "run_config.json",
        {
            "argv": sys.argv,
            "models": selected_models,
            "methods": methods,
            "datasets": list(datasets),
            "base_models": base_models,
            "speculators": speculators,
            "calibration_cache_root": str(args.calibration_cache_root.resolve()),
            "num_spec_tokens": args.num_spec_tokens,
            "dtype": args.dtype,
            "token_dense_linear_strategy": args.token_dense_linear_strategy,
            "token_dense_mlp_strategy": args.token_dense_mlp_strategy,
            "token_dense_score_backend": args.token_dense_score_backend,
            "token_dense_fast_plan": args.token_dense_fast_plan,
            "token_dense_reuse_buffers": args.token_dense_reuse_buffers,
            "token_dense_contiguous_scatter": (
                args.token_dense_contiguous_scatter
            ),
            "token_dense_sparse_value_scale": (
                args.token_dense_sparse_value_scale
            ),
            "token_dense_row_scale_mode": args.token_dense_row_scale_mode,
            "token_dense_row_scale_max": args.token_dense_row_scale_max,
            "token_dense_sparse_output_mode": (
                args.token_dense_sparse_output_mode
            ),
            "token_dense_sparse_accumulator": (
                args.token_dense_sparse_accumulator
            ),
            "token_dense_cudagraph_mode": args.token_dense_cudagraph_mode,
            "token_dense_mask_root": str(args.token_dense_mask_root),
            "token_dense_mask_method": args.token_dense_mask_method,
            "created_at": timestamp(),
            "smoke": args.smoke,
        },
    )

    rows: list[dict[str, Any]] = []
    for model_label in selected_models:
        model_id = base_models.get(model_label)
        speculator_model = speculators.get(model_label)
        if not model_id:
            raise ValueError(f"unknown model label: {model_label}")
        if not speculator_model:
            raise ValueError(f"missing EAGLE3 speculator for {model_label}")

        dense_metrics, _dense_stats = run_method(
            args,
            model_label=model_label,
            model_id=model_id,
            speculator_model=speculator_model,
            method=dense_config,
            datasets=datasets,
            case_dir=output_root / "runs" / model_label / "dense",
        )

        for method_config in method_configs:
            method_metrics, mask_stats = run_method(
                args,
                model_label=model_label,
                model_id=model_id,
                speculator_model=speculator_model,
                method=method_config,
                datasets=datasets,
                case_dir=output_root / "runs" / model_label / method_config.label,
            )
            rows.extend(
                make_rows(
                    args=args,
                    model_label=model_label,
                    model_id=model_id,
                    method=method_config,
                    dense_metrics=dense_metrics,
                    method_metrics=method_metrics,
                    mask_stats=mask_stats,
                )
            )
            write_outputs(output_root, rows, args)

    write_outputs(output_root, rows, args)
    print(output_root.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token-dense fixed-budget accuracy benchmark under EAGLE3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--model-id", action="append", default=[], help="Override base model as LABEL=PATH_OR_ID.")
    parser.add_argument("--speculator-model", action="append", default=[], help="Override EAGLE3 model as LABEL=PATH_OR_ID.")
    parser.add_argument("--methods", default=DEFAULT_TOKEN_DENSE_METHODS)
    parser.add_argument("--datasets", default="gsm8k,math_reasoning")
    parser.add_argument("--gsm8k-num-examples", type=int, default=0)
    parser.add_argument("--math-num-examples", type=int, default=80)
    parser.add_argument("--accuracy-max-tokens", type=int, default=512)
    parser.add_argument("--accuracy-concurrency", type=int, default=8)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="fp16")
    parser.add_argument("--port-base", type=int, default=8170)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=5.0)
    parser.add_argument("--calibration-cache-root", type=Path, default=DEFAULT_C4_CALIBRATION_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--token-dense-enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-token-dense-enforce-eager", dest="token_dense_enforce_eager", action="store_false")
    parser.add_argument(
        "--token-dense-cudagraph-mode",
        choices=("none", "full", "full_decode_only"),
        default="none",
        help=(
            "CUDA graph mode for token-dense routing. Dynamic routing only "
            "supports none; other values fail before server startup."
        ),
    )
    parser.add_argument(
        "--token-dense-linear-strategy",
        choices=(
            "auto",
            "full_sparse_residual",
            "full_sparse_dense_override",
            "split_dense_sparse",
            "sparse_only_decode",
        ),
        default="auto",
    )
    parser.add_argument(
        "--token-dense-mlp-strategy",
        choices=("auto", "gate_only", "linear"),
        default="auto",
    )
    parser.add_argument(
        "--token-dense-score-backend",
        choices=("torch_softmax", "triton_selected", "triton_fused"),
        default="triton_fused",
    )
    parser.add_argument(
        "--token-dense-fast-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-reuse-buffers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-contiguous-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--token-dense-sparse-value-scale",
        type=float,
        default=1.0,
        help="Scale retained 2:4 values at prepack time without runtime overhead.",
    )
    parser.add_argument(
        "--token-dense-row-scale-mode",
        choices=("none", "cache", "variance"),
        default="cache",
        help=(
            "Use cached per-row scales or compute activation-weighted "
            "variance-preserving scales during prepack."
        ),
    )
    parser.add_argument(
        "--token-dense-row-scale-max",
        type=float,
        default=1.25,
        help="Upper clamp for variance-preserving per-row scales.",
    )
    parser.add_argument(
        "--token-dense-sparse-output-mode",
        choices=("contiguous", "fused_mlp", "view_mlp", "view_mlp_o"),
        default="contiguous",
        help=(
            "Keep every sparse output row-major or preserve CUTLASS views for "
            "MLP and attention output projections."
        ),
    )
    parser.add_argument(
        "--token-dense-sparse-accumulator",
        choices=(
            "fp32",
            "fp16",
            "fp16_gate",
            "fp16_gate_down",
            "fp16_qkv_gate",
        ),
        default="fp32",
        help=(
            "Accumulator policy for supported CUTLASS sparse decode tiles; "
            "fp16_gate keeps QKV/output/down on FP32; fp16_gate_down keeps "
            "QKV/output on FP32; fp16_qkv_gate keeps output/down on FP32."
        ),
    )
    parser.add_argument("--token-dense-mask-root", type=Path, default=DEFAULT_TOKEN_DENSE_MASK_ROOT)
    parser.add_argument(
        "--token-dense-mask-method",
        choices=(
            "wanda",
            "covwanda",
            "proxsparse",
            "maskllm",
            "inherit",
            "none",
        ),
        default="wanda",
    )
    parser.add_argument(
        "--production-fast",
        action="store_true",
        help="Disable token-dense debug stats and use static hot-path env reads.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
