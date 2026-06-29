#!/usr/bin/env python3
"""Run the SR24 slowdown breakdown protocol.

Usage:
  cd examples/evaluate/eval-guidellm
  conda run -n spec python scripts/run_sr24_slowdown_breakdown.py

This is an orchestrator around the existing serving and microbenchmark tools.
It intentionally separates clean throughput rows from instrumented diagnostic
rows:

* clean_serving: low-overhead vLLM/GuideLLM run for tok/s, GPU util, acceptance,
  and CUDA Graph mode counts.
* instrumented_serving: SR24 linear/routing CUDA event timing. Its tok/s is a
  diagnostic value because the event timing and exact routing counters add
  synchronization overhead.
* --sr24-breakdown-gpu-counts: optional lower-CPU-sync routing counts. It
  accumulates residual/base rows and bucket active rows on GPU and reads them
  during breakdown snapshots.
* component_microbench: isolated Linear-shape timing for dense, sparse base,
  dense-row correction, gather/scatter, and routed assembly candidates.

The final seven-part report joins the serving artifacts into one table that
matches the slowdown questions in SR24_SLOWDOWN_BREAKDOWN.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parents[2]
CPU_SYNC_VARIANT_NAMES = [
    "low_sync_stats_on",
    "low_sync_stats_off",
    "sync_mask_state",
    "sync_heavy",
    "low_sync_gpu_counts",
]
_SR24_PRESET_OVERRIDE_FLAGS: dict[str, tuple[str, ...]] = {
    "sr24_allow_cudagraph": (
        "--sr24-allow-cudagraph",
        "--no-sr24-allow-cudagraph",
    ),
    "sr24_default_vllm_compile": (
        "--sr24-default-vllm-compile",
        "--no-sr24-default-vllm-compile",
    ),
    "sr24_dynamic_auto_cudagraph": (
        "--sr24-dynamic-auto-cudagraph",
        "--no-sr24-dynamic-auto-cudagraph",
    ),
    "sr24_force_cudagraph_none_for_mixed": (
        "--sr24-force-cudagraph-none-for-mixed",
        "--no-sr24-force-cudagraph-none-for-mixed",
    ),
    "sr24_cudagraph_bucket": (
        "--sr24-cudagraph-bucket",
        "--no-sr24-cudagraph-bucket",
    ),
    "sr24_disable_runtime_stats": (
        "--sr24-disable-runtime-stats",
        "--no-sr24-disable-runtime-stats",
    ),
}


def _argv_has_flag(argv: list[str], flags: tuple[str, ...]) -> bool:
    for item in argv:
        for flag in flags:
            if item == flag or item.startswith(f"{flag}="):
                return True
    return False


def capture_explicit_sr24_overrides(
    args: argparse.Namespace,
    argv: list[str],
) -> dict[str, object]:
    return {
        attr: getattr(args, attr)
        for attr, flags in _SR24_PRESET_OVERRIDE_FLAGS.items()
        if _argv_has_flag(argv, flags)
    }


def restore_explicit_sr24_overrides(args: argparse.Namespace) -> None:
    overrides = getattr(args, "_explicit_sr24_overrides", {})
    if not isinstance(overrides, dict):
        return
    for attr, value in overrides.items():
        setattr(args, attr, value)


def expand_breakdown_sr24_preset(args: argparse.Namespace) -> None:
    """Expand known matrix-runner presets before composing child commands.

    This wrapper passes many SR24 arguments explicitly to the matrix runner.
    If it also forwards a non-manual preset, the matrix runner treats those
    explicit wrapper defaults as user overrides and can silently undo the
    intended preset.  Expand the current breakdown preset locally, then forward
    it as a manual configuration so the generated command is self-contained.
    """
    source = str(getattr(args, "sr24_preset", "manual"))
    args.sr24_preset_source = source
    if source == "manual":
        return
    explicit_residual_bucket_size = bool(
        getattr(args, "_explicit_sr24_residual_bucket_size", False)
    )
    explicit_residual_bucket_scale_by_active = bool(
        getattr(args, "_explicit_sr24_residual_bucket_scale_by_active", False)
    )
    explicit_residual_bucket_priority = bool(
        getattr(args, "_explicit_sr24_residual_bucket_priority", False)
    )
    residual_bucket_size = args.sr24_residual_bucket_size
    residual_bucket_scale_by_active = args.sr24_residual_bucket_scale_by_active
    residual_bucket_priority = args.sr24_residual_bucket_priority
    if source == "mlpall_lowconf_prefix5_tritonoverride":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj,down_proj"
        args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = ""
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.6
        args.sr24_selective_min_prefix_residual = 5
        args.sr24_selective_extra_after_low = 0
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 32
        args.sr24_residual_bucket_priority = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_allow_cudagraph = True
        args.sr24_triton_bucket_override = True
        args.sr24_disable_runtime_stats = True
        args.clean_stats_interval = max(int(args.clean_stats_interval), 32)
        args.sr24_preset = "manual"
        if explicit_residual_bucket_size:
            args.sr24_residual_bucket_size = residual_bucket_size
        if explicit_residual_bucket_scale_by_active:
            args.sr24_residual_bucket_scale_by_active = (
                residual_bucket_scale_by_active
            )
        if explicit_residual_bucket_priority:
            args.sr24_residual_bucket_priority = residual_bucket_priority
        return
    if source == "down0_15_fixedprefix4_directcslt":
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "down_proj"
        args.sr24_residual_target_leafs = "down_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "down_proj=0-15"
        args.sr24_selective_residual_policy = "fixed_prefix"
        args.sr24_selective_non_draft_policy = "all"
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_max_residual_draft_rows = 4
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 0
        args.sr24_residual_bucket_priority = False
        args.sr24_bucket_dense_copy = True
        args.sr24_route_all_residual_rows = True
        args.sr24_route_all_skip_bucket = True
        args.sr24_direct_cpu_route_rows = False
        args.sr24_route_dense_fallback_fraction = 0.9
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.clean_stats_interval = max(int(args.clean_stats_interval), 32)
        args.sr24_preset = "manual"
        if explicit_residual_bucket_size:
            args.sr24_residual_bucket_size = residual_bucket_size
        if explicit_residual_bucket_scale_by_active:
            args.sr24_residual_bucket_scale_by_active = (
                residual_bucket_scale_by_active
            )
        if explicit_residual_bucket_priority:
            args.sr24_residual_bucket_priority = residual_bucket_priority
        return
    if source == "lowresidual_gateup_riskcap2":
        # Keep the breakdown wrapper in lockstep with the matrix runner's
        # current best low-residual gate_up-only candidate. Without local
        # expansion, wrapper defaults are forwarded as explicit overrides and
        # silently replace the intended preset in the child runner.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "low_confidence"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_selective_min_prefix_residual = 2
        args.sr24_selective_max_residual_draft_rows = 2
        args.sr24_low_confidence_cap_by_risk = True
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 8
        args.sr24_residual_bucket_priority = True
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = True
        args.sr24_default_vllm_compile = True
        args.sr24_cudagraph_bucket = True
        args.sr24_force_cudagraph_none_for_mixed = False
        args.sr24_dynamic_auto_cudagraph = True
        args.sr24_disable_runtime_stats = True
        args.clean_stats_interval = max(int(args.clean_stats_interval), 32)
        args.sr24_preset = "manual"
        if explicit_residual_bucket_size:
            args.sr24_residual_bucket_size = residual_bucket_size
        if explicit_residual_bucket_scale_by_active:
            args.sr24_residual_bucket_scale_by_active = (
                residual_bucket_scale_by_active
            )
        if explicit_residual_bucket_priority:
            args.sr24_residual_bucket_priority = residual_bucket_priority
        return
    if source == "criticalprefix_extra2_gateup_scaledbucket":
        # Current quality-oriented gate_up-only candidate. Keep this expansion
        # in lockstep with the matrix runner so slowdown breakdown rows profile
        # the same route that serving uses. The mandatory prefix guard and
        # per-active bucket size match the paired GSM8K regression fix.
        args.sr24_backend = "torch_sparse"
        args.sr24_residual_backend = "dense_rows"
        args.sr24_residual_device = "cuda"
        args.sr24_require_gpu_residual = True
        args.sr24_target_leafs = "gate_up_proj"
        args.sr24_residual_target_leafs = "gate_up_proj"
        args.sr24_base_only_layer_ids = ""
        args.sr24_base_only_layer_ids_by_leaf = ""
        args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31"
        args.sr24_selective_residual_policy = "critical_prefix"
        args.sr24_selective_non_draft_policy = "bonus"
        args.sr24_threshold = 0.8
        args.sr24_prefix_threshold = 0.8
        args.sr24_selective_extra_after_low = 2
        args.sr24_selective_min_prefix_residual = 4
        args.sr24_selective_max_residual_draft_rows = 0
        args.sr24_low_confidence_cap_by_risk = False
        args.sr24_reduce_cpu_sync = True
        args.sr24_sync_mask_state = False
        args.sr24_static_mask_state = "auto"
        args.sr24_static_mask_buffer = True
        args.sr24_batched_mask_builder = True
        args.sr24_residual_bucket_size = 5
        args.sr24_residual_bucket_scale_by_active = True
        args.sr24_residual_bucket_priority = True
        args.sr24_direct_position_bucket = True
        args.sr24_bonus_priority = 0.5
        args.sr24_draft_position_priority_scale = 0.0
        args.sr24_bucket_dense_copy = True
        args.sr24_direct_cslt_linear = True
        args.sr24_allow_cudagraph = False
        args.sr24_default_vllm_compile = False
        args.sr24_cudagraph_bucket = False
        args.sr24_force_cudagraph_none_for_mixed = True
        args.sr24_dynamic_auto_cudagraph = False
        args.sr24_disable_runtime_stats = False
        args.sr24_preset = "manual"
        if explicit_residual_bucket_size:
            args.sr24_residual_bucket_size = residual_bucket_size
        if explicit_residual_bucket_scale_by_active:
            args.sr24_residual_bucket_scale_by_active = (
                residual_bucket_scale_by_active
            )
        if explicit_residual_bucket_priority:
            args.sr24_residual_bucket_priority = residual_bucket_priority
        return
    if source != "criticalprefix4_bucket16_directcslt":
        return

    args.sr24_backend = "torch_sparse"
    args.sr24_residual_backend = "dense_rows"
    args.sr24_residual_device = "cuda"
    args.sr24_require_gpu_residual = True
    args.sr24_target_leafs = "gate_up_proj,down_proj"
    args.sr24_residual_target_leafs = "gate_up_proj,down_proj"
    args.sr24_base_only_layer_ids = ""
    args.sr24_base_only_layer_ids_by_leaf = ""
    args.sr24_residual_layer_ids_by_leaf = "gate_up_proj=16-31;down_proj=8-15"
    args.sr24_selective_residual_policy = "critical_prefix"
    args.sr24_selective_non_draft_policy = "bonus"
    args.sr24_threshold = 0.6
    args.sr24_selective_min_prefix_residual = 4
    args.sr24_selective_extra_after_low = 1
    args.sr24_selective_max_residual_draft_rows = 0
    args.sr24_reduce_cpu_sync = True
    args.sr24_sync_mask_state = False
    args.sr24_static_mask_state = "auto"
    args.sr24_static_mask_buffer = True
    args.sr24_batched_mask_builder = True
    args.sr24_residual_bucket_size = 16
    args.sr24_residual_bucket_priority = False
    args.sr24_bucket_dense_copy = True
    args.sr24_direct_cslt_linear = True
    args.sr24_allow_cudagraph = True
    args.sr24_default_vllm_compile = False
    args.sr24_cudagraph_bucket = True
    args.sr24_force_cudagraph_none_for_mixed = False
    args.sr24_dynamic_auto_cudagraph = True
    args.sr24_disable_runtime_stats = True
    args.sr24_preset = "manual"
    if explicit_residual_bucket_size:
        args.sr24_residual_bucket_size = residual_bucket_size
    if explicit_residual_bucket_scale_by_active:
        args.sr24_residual_bucket_scale_by_active = (
            residual_bucket_scale_by_active
        )
    if explicit_residual_bucket_priority:
        args.sr24_residual_bucket_priority = residual_bucket_priority


def request_coverage_warnings(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    checks: list[tuple[str, int, bool]] = [
        ("clean_serving", args.fixed_total_requests, not args.skip_clean_serving),
        (
            "instrumented_serving",
            args.instrumented_requests,
            not args.skip_instrumented_serving,
        ),
        (
            "cpu_sync_ablation",
            args.cpu_sync_requests,
            args.include_cpu_sync_ablation,
        ),
    ]
    for label, requests, enabled in checks:
        if enabled and requests < args.batch_size:
            warnings.append(
                f"{label}: requests={requests} < batch_size={args.batch_size}; "
                "full-batch throughput fields will be absent or unusable. "
                "Treat this row as diagnostic-only."
            )
    return warnings


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def quote_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    commands: list[str],
) -> None:
    rendered = f"(cd {shlex.quote(str(cwd))} && {quote_cmd(command)})"
    commands.append(rendered)
    print(rendered, flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def common_matrix_args(
    args: argparse.Namespace,
    final_root: Path,
    work_root: Path,
    *,
    fixed_total_requests: int,
    max_tokens: int,
    stats_interval: int,
) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_ROOT / "scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py"),
        "--datasets",
        args.dataset,
        "--batch-sizes",
        str(args.batch_size),
        "--repeats",
        "1",
        "--fixed-total-requests",
        str(fixed_total_requests),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--warmup-s",
        str(args.warmup_s),
        "--measurement-s",
        str(args.measurement_s),
        "--cooldown-s",
        str(args.cooldown_s),
        "--health-timeout-s",
        str(args.health_timeout_s),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--port-base",
        str(args.port_base),
        "--final-root",
        str(final_root),
        "--work-root",
        str(work_root),
        "--eagle3-k",
        str(args.eagle3_k),
        "--sr24-preset",
        args.sr24_preset,
        "--sr24-backend",
        args.sr24_backend,
        "--sr24-residual-backend",
        args.sr24_residual_backend,
        "--sr24-residual-device",
        args.sr24_residual_device,
        "--sr24-residual-out-chunk",
        str(args.sr24_residual_out_chunk),
        "--sr24-threshold",
        str(args.sr24_threshold),
        "--sr24-target-leafs",
        args.sr24_target_leafs,
        "--sr24-residual-target-leafs",
        args.sr24_residual_target_leafs,
        "--sr24-residual-layer-ids-by-leaf",
        args.sr24_residual_layer_ids_by_leaf,
        "--sr24-base-only-layer-ids",
        args.sr24_base_only_layer_ids,
        "--sr24-base-only-layer-ids-by-leaf",
        args.sr24_base_only_layer_ids_by_leaf,
        "--sr24-static-mask-state",
        args.sr24_static_mask_state,
        "--sr24-selective-residual-policy",
        args.sr24_selective_residual_policy,
        "--sr24-selective-non-draft-policy",
        args.sr24_selective_non_draft_policy,
        "--sr24-prefix-threshold",
        str(args.sr24_prefix_threshold),
        "--sr24-selective-extra-after-low",
        str(args.sr24_selective_extra_after_low),
        "--sr24-selective-min-prefix-residual",
        str(args.sr24_selective_min_prefix_residual),
        "--sr24-selective-max-residual-draft-rows",
        str(args.sr24_selective_max_residual_draft_rows),
        "--sr24-early-dense-tokens",
        str(args.sr24_early_dense_tokens),
        "--sr24-stats-interval",
        str(stats_interval),
        "--sr24-residual-bucket-size",
        str(args.sr24_residual_bucket_size),
        "--sr24-bonus-priority",
        str(args.sr24_bonus_priority),
        "--sr24-draft-position-priority-scale",
        str(args.sr24_draft_position_priority_scale),
        "--sr24-route-dense-fallback-fraction",
        str(args.sr24_route_dense_fallback_fraction),
        "--sr24-static-mask-buffer",
        "--sr24-batched-mask-builder",
    ]
    if args.sr24_low_confidence_cap_by_risk:
        command.append("--sr24-low-confidence-cap-by-risk")
    if args.sr24_cache_compressed_residual_weight:
        command.append("--sr24-cache-compressed-residual-weight")
    if args.sr24_prewarm_compressed_residual_weight:
        command.append("--sr24-prewarm-compressed-residual-weight")
    if not args.sr24_auto_compressed_residual_fastpath:
        command.append("--no-sr24-auto-compressed-residual-fastpath")
    if args.sr24_compressed_residual_triton:
        command.append("--sr24-compressed-residual-triton")
    command.extend([
        "--sr24-compressed-residual-block-m",
        str(args.sr24_compressed_residual_block_m),
        "--sr24-compressed-residual-block-n",
        str(args.sr24_compressed_residual_block_n),
        "--sr24-compressed-residual-block-g",
        str(args.sr24_compressed_residual_block_g),
        "--sr24-extract-chunk-rows",
        str(args.sr24_extract_chunk_rows),
    ])
    if args.sr24_direct_cslt_linear:
        command.append("--sr24-direct-cslt-linear")
    if not args.sr24_auto_direct_cslt_base_only:
        command.append("--no-sr24-auto-direct-cslt-base-only")
    if args.sr24_adaptive_dense_fallback:
        command.append("--sr24-adaptive-dense-fallback")
    command.extend([
        "--sr24-adaptive-dense-fallback-small-rows",
        str(args.sr24_adaptive_dense_fallback_small_rows),
        "--sr24-adaptive-dense-fallback-gate-up-fraction",
        str(args.sr24_adaptive_dense_fallback_gate_up_fraction),
        "--sr24-adaptive-dense-fallback-down-fraction",
        str(args.sr24_adaptive_dense_fallback_down_fraction),
    ])
    if not args.sr24_adaptive_dense_fallback_small_down_no_residual:
        command.append("--no-sr24-adaptive-dense-fallback-small-down-no-residual")
    if args.sr24_residual_bucket_priority:
        command.append("--sr24-residual-bucket-priority")
    if args.sr24_residual_bucket_scale_by_active:
        command.append("--sr24-residual-bucket-scale-by-active")
    if args.sr24_direct_position_bucket:
        command.append("--sr24-direct-position-bucket")
    if args.sr24_route_bucket_rows:
        command.append("--sr24-route-bucket-rows")
    if args.sr24_default_vllm_compile:
        command.append("--sr24-default-vllm-compile")
    if args.sr24_static_all_residual_dense_fastpath:
        command.append("--sr24-static-all-residual-dense-fastpath")
    if args.sr24_dynamic_auto_cudagraph:
        command.append("--sr24-dynamic-auto-cudagraph")
    command.append(
        "--sr24-force-cudagraph-none-for-mixed"
        if args.sr24_force_cudagraph_none_for_mixed
        else "--no-sr24-force-cudagraph-none-for-mixed"
    )
    if args.sr24_cudagraph_bucket:
        command.append("--sr24-cudagraph-bucket")
    if args.sr24_route_all_residual_rows:
        command.append("--sr24-route-all-residual-rows")
    if args.sr24_route_all_skip_bucket:
        command.append("--sr24-route-all-skip-bucket")
    if args.sr24_direct_cpu_route_rows:
        command.append("--sr24-direct-cpu-route-rows")
    if args.sr24_route_reuse_base_output:
        command.append("--sr24-route-reuse-base-output")
    if args.sr24_route_contiguous_fastpath:
        command.append("--sr24-route-contiguous-fastpath")
    if args.sr24_triton_route_assembly:
        command.append("--sr24-triton-route-assembly")
    if args.sr24_triton_bucket_override:
        command.append("--sr24-triton-bucket-override")
    if args.sr24_triton_bucket_dense_gemm:
        command.append("--sr24-triton-bucket-dense-gemm")
    command.extend([
        "--sr24-triton-bucket-dense-block-m",
        str(args.sr24_triton_bucket_dense_block_m),
        "--sr24-triton-bucket-dense-block-n",
        str(args.sr24_triton_bucket_dense_block_n),
        "--sr24-triton-bucket-dense-block-k",
        str(args.sr24_triton_bucket_dense_block_k),
    ])
    if args.sr24_row_routed_mlp:
        command.append("--sr24-row-routed-mlp")
    if args.sr24_row_routed_down_linear:
        command.append("--sr24-row-routed-down-linear")
    if args.sr24_row_routed_mlp_reuse_base_output:
        command.append("--sr24-row-routed-mlp-reuse-base-output")
    if args.sr24_bucket_dense_copy:
        command.append("--sr24-bucket-dense-copy")
    if args.sr24_sort_bucket_rows:
        command.append("--sr24-sort-bucket-rows")
    command.extend([
        "--sr24-row-routed-mlp-min-dense-rows",
        str(args.sr24_row_routed_mlp_min_dense_rows),
        "--sr24-row-routed-mlp-max-dense-rows",
        str(args.sr24_row_routed_mlp_max_dense_rows),
        "--sr24-row-routed-mlp-max-base-rows",
        str(args.sr24_row_routed_mlp_max_base_rows),
        "--sr24-route-min-dense-rows",
        str(args.sr24_route_min_dense_rows),
        "--sr24-route-min-base-rows",
        str(args.sr24_route_min_base_rows),
        "--sr24-route-max-dense-fraction",
        str(args.sr24_route_max_dense_fraction),
    ])
    if args.sr24_disable_runtime_stats:
        command.append("--sr24-disable-runtime-stats")
    if args.sr24_reduce_cpu_sync:
        command.append("--sr24-reduce-cpu-sync")
    if args.sr24_full_residual_early_dense:
        command.append("--sr24-full-residual-early-dense")
    if args.sr24_breakdown_gpu_counts:
        command.append("--sr24-breakdown-gpu-counts")
    if args.sr24_cudagraph_stats:
        command.append("--sr24-cudagraph-stats")
    command.append(
        "--sr24-all-corrected-dense-fastpath"
        if args.sr24_all_corrected_dense_fastpath
        else "--no-sr24-all-corrected-dense-fastpath"
    )
    command.append(
        "--sr24-sync-mask-state"
        if args.sr24_sync_mask_state
        else "--no-sr24-sync-mask-state"
    )
    return command


def maybe_bool_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.sr24_require_gpu_residual:
        flags.append("--sr24-require-gpu-residual")
    if args.sr24_allow_cudagraph:
        flags.append("--sr24-allow-cudagraph")
    if args.disable_chunked_prefill:
        flags.append("--disable-chunked-prefill")
    return flags


def run_clean_serving_named(
    args: argparse.Namespace,
    output_root: Path,
    work_root: Path,
    commands: list[str],
    *,
    name: str,
    methods: str,
    fixed_total_requests: int,
    max_tokens: int,
    stats_interval: int,
    extra_flags: list[str] | None = None,
) -> Path:
    final_root = output_root / name
    clean_work = work_root / name
    command = common_matrix_args(
        args,
        final_root,
        clean_work,
        fixed_total_requests=fixed_total_requests,
        max_tokens=max_tokens,
        stats_interval=stats_interval,
    )
    command.extend(["--methods", methods])
    command.extend(maybe_bool_flags(args))
    if extra_flags:
        command.extend(extra_flags)
    run_command(command, cwd=EVAL_ROOT, dry_run=args.dry_run, commands=commands)
    return final_root


def run_clean_serving(args: argparse.Namespace, output_root: Path,
                      work_root: Path, commands: list[str]) -> Path:
    return run_clean_serving_named(
        args,
        output_root,
        work_root,
        commands,
        name="clean_serving",
        methods=args.clean_methods,
        fixed_total_requests=args.fixed_total_requests,
        max_tokens=args.max_tokens,
        stats_interval=args.clean_stats_interval,
    )


def _args_with(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    variant = copy.copy(args)
    for key, value in updates.items():
        setattr(variant, key, value)
    return variant


def run_cpu_sync_ablation(
    args: argparse.Namespace,
    output_root: Path,
    work_root: Path,
    commands: list[str],
) -> list[Path]:
    """Run clean serving variants that isolate SR24 CPU-sync overhead.

    These rows intentionally do not enable Linear CUDA-event timing. The only
    diagnostic row here is ``low_sync_gpu_counts``, which keeps routing counts on
    GPU and reads them during breakdown snapshots.
    """
    variants: list[tuple[str, argparse.Namespace, list[str]]] = [
        (
            "low_sync_stats_on",
            _args_with(
                args,
                sr24_reduce_cpu_sync=True,
                sr24_sync_mask_state=False,
                sr24_disable_runtime_stats=False,
            ),
            [],
        ),
        (
            "low_sync_stats_off",
            _args_with(
                args,
                sr24_reduce_cpu_sync=True,
                sr24_sync_mask_state=False,
                sr24_disable_runtime_stats=True,
            ),
            [],
        ),
        (
            "sync_mask_state",
            _args_with(
                args,
                sr24_reduce_cpu_sync=True,
                sr24_sync_mask_state=True,
                sr24_disable_runtime_stats=False,
            ),
            [],
        ),
        (
            "sync_heavy",
            _args_with(
                args,
                sr24_reduce_cpu_sync=False,
                sr24_sync_mask_state=True,
                sr24_disable_runtime_stats=False,
            ),
            [],
        ),
        (
            "low_sync_gpu_counts",
            _args_with(
                args,
                sr24_reduce_cpu_sync=True,
                sr24_sync_mask_state=False,
                sr24_disable_runtime_stats=False,
            ),
            [
                "--sr24-breakdown",
                "--sr24-breakdown-gpu-counts",
                "--sr24-breakdown-interval",
                str(args.sr24_breakdown_interval),
            ],
        ),
    ]
    roots: list[Path] = []
    for name, variant_args, extra_flags in variants:
        roots.append(
            run_clean_serving_named(
                variant_args,
                output_root,
                work_root,
                commands,
                name=f"cpu_sync_ablation/{name}",
                methods=args.cpu_sync_methods,
                fixed_total_requests=args.cpu_sync_requests,
                max_tokens=args.cpu_sync_max_tokens,
                stats_interval=args.clean_stats_interval,
                extra_flags=extra_flags,
            )
        )
    return roots


def run_instrumented_serving(args: argparse.Namespace, output_root: Path,
                             work_root: Path, commands: list[str]) -> Path:
    # CUDA-event instrumentation is not capture-safe. Keep instrumented rows
    # eager-only and use clean_serving rows for CUDA Graph FULL/NONE coverage.
    # Do not inherit --sr24-default-vllm-compile here: the default compile/cache
    # path can absorb SR24 Linear hooks and leave breakdown_linear fields empty.
    args = _args_with(
        args,
        sr24_allow_cudagraph=False,
        sr24_default_vllm_compile=False,
        sr24_dynamic_auto_cudagraph=False,
        sr24_force_cudagraph_none_for_mixed=True,
        sr24_cudagraph_bucket=False,
    )
    final_root = output_root / "instrumented_serving"
    instr_work = work_root / "instrumented_serving"
    command = common_matrix_args(
        args,
        final_root,
        instr_work,
        fixed_total_requests=args.instrumented_requests,
        max_tokens=args.instrumented_max_tokens,
        stats_interval=1,
    )
    command.extend(["--methods", args.instrumented_methods])
    command.extend(maybe_bool_flags(args))
    command.extend([
        "--sr24-breakdown",
        "--sr24-breakdown-linear",
        "--sr24-breakdown-exact-routing",
        "--sr24-breakdown-interval",
        str(args.sr24_breakdown_interval),
        "--sr24-force-eager-after-preset",
        "--no-sr24-all-corrected-dense-fastpath",
    ])
    run_command(command, cwd=EVAL_ROOT, dry_run=args.dry_run, commands=commands)
    return final_root


def run_component_microbench(args: argparse.Namespace, output_root: Path,
                             commands: list[str]) -> Path:
    micro_root = output_root / "component_microbench"
    command = [
        sys.executable,
        str(EVAL_ROOT / "scripts/profile_speclink_sr24_component_breakdown.py"),
        "--shape",
        args.gate_up_shape,
        "--shape",
        args.down_shape,
        "--residual-fractions",
        args.microbench_residual_fractions,
        "--bucket-size",
        str(args.microbench_bucket_size),
        "--warmup",
        str(args.microbench_warmup),
        "--repeats",
        str(args.microbench_repeats),
        "--output-root",
        str(micro_root),
    ]
    run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run, commands=commands)
    return micro_root


def run_reports(args: argparse.Namespace, output_root: Path,
                roots: list[Path], commands: list[str]) -> None:
    summary_root = output_root / "component_summary"
    seven_root = output_root / "seven_part_report"
    root_args = [str(root) for root in roots]
    run_command(
        [
            sys.executable,
            str(EVAL_ROOT / "scripts/summarize_sr24_breakdown.py"),
            "--roots",
            *root_args,
            "--output-root",
            str(summary_root),
        ],
        cwd=EVAL_ROOT,
        dry_run=args.dry_run,
        commands=commands,
    )
    run_command(
        [
            sys.executable,
            str(EVAL_ROOT / "scripts/make_sr24_seven_part_breakdown.py"),
            "--roots",
            *root_args,
            "--output-root",
            str(seven_root),
        ],
        cwd=EVAL_ROOT,
        dry_run=args.dry_run,
        commands=commands,
    )


def write_readme(output_root: Path, work_root: Path, commands: list[str],
                 args: argparse.Namespace) -> None:
    warnings = request_coverage_warnings(args)
    config = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "clean_methods": parse_csv(args.clean_methods),
        "instrumented_methods": parse_csv(args.instrumented_methods),
        "max_tokens": args.max_tokens,
        "instrumented_max_tokens": args.instrumented_max_tokens,
        "fixed_total_requests": args.fixed_total_requests,
        "instrumented_requests": args.instrumented_requests,
        "sr24_policy": args.sr24_selective_residual_policy,
        "sr24_threshold": args.sr24_threshold,
        "sr24_preset": args.sr24_preset,
        "sr24_preset_source": getattr(args, "sr24_preset_source", args.sr24_preset),
        "sr24_prefix_threshold": args.sr24_prefix_threshold,
        "sr24_extra_after_low": args.sr24_selective_extra_after_low,
        "sr24_min_prefix_residual": args.sr24_selective_min_prefix_residual,
        "sr24_max_residual_draft_rows": (
            args.sr24_selective_max_residual_draft_rows
        ),
        "sr24_direct_cslt_linear": args.sr24_direct_cslt_linear,
        "sr24_auto_direct_cslt_base_only": args.sr24_auto_direct_cslt_base_only,
        "sr24_residual_bucket_size": args.sr24_residual_bucket_size,
        "sr24_residual_bucket_scale_by_active":
        args.sr24_residual_bucket_scale_by_active,
        "sr24_residual_bucket_priority": args.sr24_residual_bucket_priority,
        "sr24_direct_position_bucket": args.sr24_direct_position_bucket,
        "sr24_bonus_priority": args.sr24_bonus_priority,
        "sr24_draft_position_priority_scale":
        args.sr24_draft_position_priority_scale,
        "sr24_default_vllm_compile": args.sr24_default_vllm_compile,
        "sr24_dynamic_auto_cudagraph": args.sr24_dynamic_auto_cudagraph,
        "sr24_force_cudagraph_none_for_mixed": (
            args.sr24_force_cudagraph_none_for_mixed
        ),
        "sr24_cudagraph_bucket": args.sr24_cudagraph_bucket,
        "sr24_cudagraph_stats": args.sr24_cudagraph_stats,
        "sr24_route_bucket_rows": args.sr24_route_bucket_rows,
        "sr24_route_all_residual_rows": args.sr24_route_all_residual_rows,
        "sr24_route_all_skip_bucket": args.sr24_route_all_skip_bucket,
        "sr24_direct_cpu_route_rows": args.sr24_direct_cpu_route_rows,
        "sr24_route_reuse_base_output": args.sr24_route_reuse_base_output,
        "sr24_route_contiguous_fastpath": args.sr24_route_contiguous_fastpath,
        "sr24_route_min_dense_rows": args.sr24_route_min_dense_rows,
        "sr24_route_min_base_rows": args.sr24_route_min_base_rows,
        "sr24_route_max_dense_fraction": args.sr24_route_max_dense_fraction,
        "sr24_triton_route_assembly": args.sr24_triton_route_assembly,
        "sr24_triton_bucket_override": args.sr24_triton_bucket_override,
        "sr24_triton_bucket_dense_gemm": args.sr24_triton_bucket_dense_gemm,
        "sr24_triton_bucket_dense_block_m":
        args.sr24_triton_bucket_dense_block_m,
        "sr24_triton_bucket_dense_block_n":
        args.sr24_triton_bucket_dense_block_n,
        "sr24_triton_bucket_dense_block_k":
        args.sr24_triton_bucket_dense_block_k,
        "sr24_bucket_dense_copy": args.sr24_bucket_dense_copy,
        "sr24_sort_bucket_rows": args.sr24_sort_bucket_rows,
        "sr24_row_routed_mlp": args.sr24_row_routed_mlp,
        "sr24_row_routed_mlp_reuse_base_output": (
            args.sr24_row_routed_mlp_reuse_base_output
        ),
        "sr24_row_routed_mlp_min_dense_rows": (
            args.sr24_row_routed_mlp_min_dense_rows
        ),
        "sr24_row_routed_mlp_max_dense_rows": (
            args.sr24_row_routed_mlp_max_dense_rows
        ),
        "sr24_row_routed_mlp_max_base_rows": (
            args.sr24_row_routed_mlp_max_base_rows
        ),
        "sr24_adaptive_dense_fallback": args.sr24_adaptive_dense_fallback,
        "sr24_adaptive_dense_fallback_small_rows": (
            args.sr24_adaptive_dense_fallback_small_rows
        ),
        "sr24_adaptive_dense_fallback_gate_up_fraction": (
            args.sr24_adaptive_dense_fallback_gate_up_fraction
        ),
        "sr24_adaptive_dense_fallback_down_fraction": (
            args.sr24_adaptive_dense_fallback_down_fraction
        ),
        "sr24_adaptive_dense_fallback_small_down_no_residual": (
            args.sr24_adaptive_dense_fallback_small_down_no_residual
        ),
        "sr24_target_leafs": args.sr24_target_leafs,
        "sr24_base_only_layer_ids": args.sr24_base_only_layer_ids,
        "sr24_base_only_layer_ids_by_leaf": args.sr24_base_only_layer_ids_by_leaf,
        "sr24_static_mask_state": args.sr24_static_mask_state,
        "sr24_static_all_residual_dense_fastpath": (
            args.sr24_static_all_residual_dense_fastpath
        ),
        "sr24_residual_layer_ids_by_leaf": args.sr24_residual_layer_ids_by_leaf,
        "sr24_residual_out_chunk": args.sr24_residual_out_chunk,
        "sr24_cache_compressed_residual_weight": (
            args.sr24_cache_compressed_residual_weight
        ),
        "sr24_prewarm_compressed_residual_weight": (
            args.sr24_prewarm_compressed_residual_weight
        ),
        "sr24_auto_compressed_residual_fastpath": (
            args.sr24_auto_compressed_residual_fastpath
        ),
        "sr24_compressed_residual_triton": args.sr24_compressed_residual_triton,
        "sr24_compressed_residual_block_m": args.sr24_compressed_residual_block_m,
        "sr24_compressed_residual_block_n": args.sr24_compressed_residual_block_n,
        "sr24_compressed_residual_block_g": args.sr24_compressed_residual_block_g,
        "sr24_extract_chunk_rows": args.sr24_extract_chunk_rows,
        "include_cpu_sync_ablation": args.include_cpu_sync_ablation,
        "cpu_sync_methods": parse_csv(args.cpu_sync_methods),
        "cpu_sync_requests": args.cpu_sync_requests,
        "cpu_sync_max_tokens": args.cpu_sync_max_tokens,
        "warnings": warnings,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n\n".join(commands)
        + "\n",
        encoding="utf-8",
    )
    primary_outputs: list[Path] = []
    if not args.skip_clean_serving:
        primary_outputs.append(output_root / "clean_serving" / "report.md")
    if args.include_cpu_sync_ablation:
        primary_outputs.extend(
            output_root / "cpu_sync_ablation" / name / "report.md"
            for name in CPU_SYNC_VARIANT_NAMES
        )
    if not args.skip_instrumented_serving:
        primary_outputs.append(output_root / "instrumented_serving" / "report.md")
    if not args.skip_microbench:
        primary_outputs.append(output_root / "component_microbench" / "summary.md")
    if not args.skip_clean_serving or not args.skip_instrumented_serving:
        primary_outputs.extend(
            [
                output_root / "component_summary" / "report.md",
                output_root / "seven_part_report" / "report.md",
                output_root / "seven_part_report" / "seven_part_breakdown.csv",
            ]
        )
    lines = [
        "# SR24 Slowdown Breakdown Run",
        "",
        f"- final root: `{output_root.resolve()}`",
        f"- work root: `{work_root.resolve()}`",
        "- clean serving rows are the throughput reference.",
        "- instrumented serving rows are for component timing only.",
        "- component microbench rows are isolated operator upper/lower bounds.",
        "",
    ]
    if warnings:
        lines.extend(
            [
                "Warnings:",
                "",
                *[f"- {warning}" for warning in warnings],
                "",
            ]
        )
    lines.extend([
        "Primary outputs:",
        "",
        *[f"- `{path.resolve()}`" for path in primary_outputs],
        "",
    ])
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ts = timestamp()
    default_output = EVAL_ROOT / "results.bak" / f"sr24_slowdown_breakdown_{ts}"
    default_work = EVAL_ROOT / "temp" / f"sr24_slowdown_breakdown_{ts}_work"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--dataset", default="math_reasoning")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clean-methods",
                        default="dense_baseline,base_only_24,speclink_t08")
    parser.add_argument("--instrumented-methods",
                        default="speclink_t08,all_corrected_24")
    parser.add_argument("--fixed-total-requests", type=int, default=64)
    parser.add_argument("--instrumented-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--instrumented-max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--measurement-s", type=float, default=10.0)
    parser.add_argument("--cooldown-s", type=float, default=1.0)
    parser.add_argument("--health-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--port-base", type=int, default=8660)
    parser.add_argument("--eagle3-k", type=int, default=8)
    parser.add_argument("--sr24-preset", default="manual")
    parser.add_argument("--sr24-backend", default="torch_sparse")
    parser.add_argument("--sr24-residual-backend", default="dense_rows")
    parser.add_argument("--sr24-residual-device", default="cuda")
    parser.add_argument("--sr24-residual-out-chunk", type=int, default=4096)
    parser.add_argument("--sr24-cache-compressed-residual-weight",
                        action="store_true")
    parser.add_argument("--sr24-prewarm-compressed-residual-weight",
                        action="store_true")
    parser.add_argument("--sr24-auto-compressed-residual-fastpath",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--sr24-compressed-residual-triton",
                        action="store_true")
    parser.add_argument("--sr24-compressed-residual-block-m",
                        type=int,
                        default=32)
    parser.add_argument("--sr24-compressed-residual-block-n",
                        type=int,
                        default=128)
    parser.add_argument("--sr24-compressed-residual-block-g",
                        type=int,
                        default=16)
    parser.add_argument("--sr24-extract-chunk-rows", type=int, default=128)
    parser.add_argument("--sr24-require-gpu-residual",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--sr24-threshold", type=float, default=0.3)
    parser.add_argument("--sr24-prefix-threshold", type=float, default=-1.0)
    parser.add_argument("--sr24-selective-residual-policy",
                        default="high_confidence")
    parser.add_argument("--sr24-selective-non-draft-policy", default="bonus")
    parser.add_argument("--sr24-selective-extra-after-low", type=int, default=0)
    parser.add_argument("--sr24-selective-min-prefix-residual",
                        type=int,
                        default=0)
    parser.add_argument("--sr24-selective-max-residual-draft-rows",
                        type=int,
                        default=0)
    parser.add_argument("--sr24-low-confidence-cap-by-risk",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--sr24-early-dense-tokens", type=int, default=0)
    parser.add_argument(
        "--sr24-reduce-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reduce scheduler and Linear-hook CPU synchronization in SR24 "
            "mixed paths. Use --no-sr24-reduce-cpu-sync for the sync-heavy "
            "ablation."
        ),
    )
    parser.add_argument(
        "--sr24-sync-mask-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize once per verify step to compute exact all/no/mixed "
            "mask state. Default is off for the low-CPU-sync ablation."
        ),
    )
    parser.add_argument("--sr24-target-leafs", default="gate_up_proj")
    parser.add_argument("--sr24-residual-target-leafs", default="gate_up_proj")
    parser.add_argument("--sr24-base-only-layer-ids", default="")
    parser.add_argument("--sr24-base-only-layer-ids-by-leaf", default="")
    parser.add_argument("--sr24-residual-layer-ids-by-leaf",
                        default="gate_up_proj=16-31")
    parser.add_argument("--sr24-static-mask-state",
                        choices=["auto", "all_residual", "no_residual", "mixed"],
                        default="auto")
    parser.add_argument("--sr24-static-all-residual-dense-fastpath",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument(
        "--sr24-direct-cslt-linear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward SPECLINK_SR24_DIRECT_CSLT_LINEAR for the current "
            "cuSPARSELt-backed sparse-base candidate."
        ),
    )
    parser.add_argument(
        "--sr24-auto-direct-cslt-base-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve the matrix runner's automatic direct cuSPARSELt "
            "base-only fast path unless explicitly disabled."
        ),
    )
    parser.add_argument(
        "--sr24-route-dense-fallback-fraction",
        type=float,
        default=1.1,
        help=(
            "If sync-mask-state is enabled, promote high-residual selective "
            "steps to the all-residual dense fastpath at this fraction. Values "
            "outside [0,1] disable the fallback."
        ),
    )
    parser.add_argument("--sr24-residual-bucket-size", type=int, default=0)
    parser.add_argument(
        "--sr24-residual-bucket-scale-by-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scale residual bucket capacity by active request count. This is "
            "needed to profile the current scaled-bucket SR24 candidate."
        ),
    )
    parser.add_argument("--sr24-residual-bucket-priority",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--sr24-direct-position-bucket",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument(
        "--sr24-bonus-priority",
        type=float,
        default=4.0,
        help=(
            "Priority score assigned to the speculative bonus/non-draft row "
            "when --sr24-residual-bucket-priority is active. Default 4.0 "
            "preserves the old behavior; lower values reserve capped buckets "
            "for draft verification rows first."
        ),
    )
    parser.add_argument(
        "--sr24-draft-position-priority-scale",
        type=float,
        default=0.0,
        help=(
            "Additional per-draft-position priority scale for capped residual "
            "buckets. Default 0 preserves old behavior; positive values make "
            "earlier draft positions dominate global top-k selection."
        ),
    )
    parser.add_argument("--sr24-route-bucket-rows",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument(
        "--sr24-default-vllm-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Do not force SR24-specific compilation config or VLLM_CACHE_ROOT. "
            "Use this for dense-equivalent/no-op controls such as "
            "all_corrected_24 densefastpath."
        ),
    )
    parser.add_argument(
        "--sr24-dynamic-auto-cudagraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow dynamic auto/mixed speclink_t08 runs to launch without "
            "--enforce-eager when the matrix runner's graph-safety guards pass."
        ),
    )
    parser.add_argument(
        "--sr24-force-cudagraph-none-for-mixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep mixed SR24 verification steps eager inside vLLM by default. "
            "Use --no-sr24-force-cudagraph-none-for-mixed only for graph-safety "
            "diagnostic runs with stable preallocated mask/bucket buffers."
        ),
    )
    parser.add_argument(
        "--sr24-cudagraph-bucket",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a stable fixed-length residual bucket tensor for graph-enabled "
            "dynamic auto/mixed SR24 runs."
        ),
    )
    parser.add_argument(
        "--sr24-route-all-residual-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route residual rows directly to dense Linear and only base rows "
            "to sparse Linear for dense_rows residual ablations. This avoids "
            "computing sparse base for rows that will be overwritten."
        ),
    )
    parser.add_argument(
        "--sr24-route-all-skip-bucket",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For route_all_residual_rows, skip residual bucket construction "
            "when the route-all Linear path does not consume the bucket."
        ),
    )
    parser.add_argument(
        "--sr24-direct-cpu-route-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For prefix_confidence + bonus + route_all_residual_rows, build "
            "exact route rows from the small draft-score tensor on CPU instead "
            "of using GPU nonzero over the full residual mask."
        ),
    )
    parser.add_argument(
        "--sr24-route-reuse-base-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the normal sparse base output and overwrite only residual "
            "rows with dense_rows Linear. This localizes correction cost while "
            "preserving the current full sparse-base pass."
        ),
    )
    parser.add_argument(
        "--sr24-route-contiguous-fastpath",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH to route-all or "
            "route-bucket rows when residual rows form a contiguous prefix or "
            "suffix. This is a routed-row ablation flag, not a default path."
        ),
    )
    parser.add_argument("--sr24-route-min-dense-rows", type=int, default=0)
    parser.add_argument("--sr24-route-min-base-rows", type=int, default=0)
    parser.add_argument(
        "--sr24-route-max-dense-fraction",
        type=float,
        default=1.1,
        help=(
            "For route_all_residual_rows, fall back to a full dense Linear when "
            "the routed dense fraction is above this value. Values outside "
            "[0,1] disable the guard."
        ),
    )
    parser.add_argument(
        "--sr24-triton-route-assembly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the Triton routed-output assembly kernel when available.",
    )
    parser.add_argument(
        "--sr24-triton-bucket-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the Triton bucket override kernel for bucketed dense rows.",
    )
    parser.add_argument(
        "--sr24-triton-bucket-dense-gemm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the prototype fused bucket dense GEMM/scatter Triton kernel.",
    )
    parser.add_argument("--sr24-triton-bucket-dense-block-m",
                        type=int,
                        default=16)
    parser.add_argument("--sr24-triton-bucket-dense-block-n",
                        type=int,
                        default=32)
    parser.add_argument("--sr24-triton-bucket-dense-block-k",
                        type=int,
                        default=128)
    parser.add_argument(
        "--sr24-bucket-dense-copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Overwrite bucket rows with dense output instead of delta add. "
            "This is the current quality-safe bucketed correction candidate."
        ),
    )
    parser.add_argument(
        "--sr24-sort-bucket-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sort selected bucket rows before dense correction. Default off; "
            "the current measured result showed extra scheduler overhead."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the experimental MLP-level routing path for mixed masks. "
            "This is a diagnostic candidate for avoiding Linear-level double "
            "work on residual rows."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-down-linear",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the experimental Linear-level down_proj routing path. "
            "It computes dense down only for residual rows and sparse down "
            "only for base rows. Default off."
        ),
    )
    parser.add_argument(
        "--sr24-row-routed-mlp-reuse-base-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward the row-routed MLP reuse-base-output variant. This keeps "
            "the full sparse base MLP output and overwrites dense rows, so the "
            "breakdown can separate scheduler savings from the extra full "
            "base sparse gate/up and down work."
        ),
    )
    parser.add_argument("--sr24-row-routed-mlp-min-dense-rows",
                        type=int,
                        default=128)
    parser.add_argument("--sr24-row-routed-mlp-max-dense-rows",
                        type=int,
                        default=0)
    parser.add_argument("--sr24-row-routed-mlp-max-base-rows",
                        type=int,
                        default=0)
    parser.add_argument(
        "--sr24-adaptive-dense-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward the matrix runner's shape-aware SR24 dense fallback. "
            "This is off by default so older slowdown breakdowns remain "
            "comparable; enable it for the current guarded t08 candidate."
        ),
    )
    parser.add_argument("--sr24-adaptive-dense-fallback-small-rows",
                        type=int,
                        default=0,
                        help=(
                            "Row-count cutoff for adaptive dense fallback "
                            "small-shape rules. Default 0 disables the "
                            "small-row rule; the older value 128 is a "
                            "diagnostic ablation and was negative for the "
                            "current bs64/K8 critical-prefix candidate."
                        ))
    parser.add_argument("--sr24-adaptive-dense-fallback-gate-up-fraction",
                        type=float,
                        default=0.10)
    parser.add_argument("--sr24-adaptive-dense-fallback-down-fraction",
                        type=float,
                        default=0.25)
    parser.add_argument(
        "--sr24-adaptive-dense-fallback-small-down-no-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sr24-disable-runtime-stats",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument(
        "--sr24-all-corrected-dense-fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward the all_corrected_24 dense-fastpath switch to clean "
            "serving rows. Use --no-sr24-all-corrected-dense-fastpath when "
            "profiling the real sparse-base plus residual-correction path."
        ),
    )
    parser.add_argument(
        "--sr24-full-residual-early-dense",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forward SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE to the matrix "
            "runner. Use this for quality-safe selective runs where fully "
            "protected verify steps should bypass sparse base work."
        ),
    )
    parser.add_argument(
        "--sr24-allow-cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow safe SR24 CUDA Graph paths in clean serving rows. This lets "
            "base_only_24/all-residual static paths report their real graph-on "
            "throughput, while dynamic mixed speclink_t08 still stays eager "
            "under the correctness guard unless explicitly made graph-safe."
        ),
    )
    parser.add_argument(
        "--sr24-cudagraph-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record generic CUDA Graph FULL/NONE counts in clean serving rows "
            "without enabling SR24 Linear timing. Enabled by default for the "
            "slowdown breakdown protocol."
        ),
    )
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--clean-stats-interval", type=int, default=32)
    parser.add_argument("--sr24-breakdown-interval", type=int, default=2000)
    parser.add_argument(
        "--sr24-breakdown-gpu-counts",
        action="store_true",
        help=(
            "Use GPU-accumulated routing counts in breakdown rows instead of "
            "per-step CPU scalar sync."
        ),
    )
    parser.add_argument("--gate-up-shape", default="512,28672,4096")
    parser.add_argument("--down-shape", default="512,4096,14336")
    parser.add_argument("--microbench-residual-fractions",
                        default="0.0625,0.125,0.25,0.5,0.875")
    parser.add_argument("--microbench-bucket-size", type=int, default=64)
    parser.add_argument("--microbench-warmup", type=int, default=20)
    parser.add_argument("--microbench-repeats", type=int, default=80)
    parser.add_argument("--skip-clean-serving", action="store_true")
    parser.add_argument(
        "--include-cpu-sync-ablation",
        action="store_true",
        help=(
            "Run additional clean serving variants for CPU-sync attribution: "
            "low-sync stats on/off, mask-state sync on, sync-heavy, and "
            "low-sync GPU-count breakdown."
        ),
    )
    parser.add_argument("--cpu-sync-methods", default="dense_baseline,speclink_t08")
    parser.add_argument("--cpu-sync-requests", type=int, default=64)
    parser.add_argument("--cpu-sync-max-tokens", type=int, default=256)
    parser.add_argument("--skip-instrumented-serving", action="store_true")
    parser.add_argument("--skip-microbench", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args._explicit_sr24_residual_bucket_size = any(
        item == "--sr24-residual-bucket-size"
        or item.startswith("--sr24-residual-bucket-size=")
        for item in sys.argv[1:]
    )
    args._explicit_sr24_residual_bucket_scale_by_active = any(
        item in {
            "--sr24-residual-bucket-scale-by-active",
            "--no-sr24-residual-bucket-scale-by-active",
        }
        for item in sys.argv[1:]
    )
    args._explicit_sr24_residual_bucket_priority = any(
        item in {
            "--sr24-residual-bucket-priority",
            "--no-sr24-residual-bucket-priority",
        }
        for item in sys.argv[1:]
    )
    args._explicit_sr24_overrides = capture_explicit_sr24_overrides(
        args, sys.argv[1:]
    )
    return args


def main() -> int:
    args = parse_args()
    expand_breakdown_sr24_preset(args)
    restore_explicit_sr24_overrides(args)
    args.sr24_triton_bucket_dense_block_m = max(
        1, int(args.sr24_triton_bucket_dense_block_m))
    args.sr24_triton_bucket_dense_block_n = max(
        1, int(args.sr24_triton_bucket_dense_block_n))
    args.sr24_triton_bucket_dense_block_k = max(
        1, int(args.sr24_triton_bucket_dense_block_k))
    for warning in request_coverage_warnings(args):
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    report_roots: list[Path] = []
    if not args.skip_clean_serving:
        report_roots.append(run_clean_serving(args, output_root, work_root, commands))
    if args.include_cpu_sync_ablation:
        report_roots.extend(
            run_cpu_sync_ablation(args, output_root, work_root, commands)
        )
    if not args.skip_instrumented_serving:
        report_roots.append(
            run_instrumented_serving(args, output_root, work_root, commands)
        )
    if not args.skip_microbench:
        report_roots.append(run_component_microbench(args, output_root, commands))
    if report_roots:
        run_reports(args, output_root, report_roots, commands)
    write_readme(output_root, work_root, commands, args)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
