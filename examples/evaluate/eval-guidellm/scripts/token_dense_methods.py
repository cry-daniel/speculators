#!/usr/bin/env python3
"""Shared method parsing and vLLM env setup for token-dense accuracy runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_structured_24_spec_quality import add_local_no_proxy


EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_DENSE_MASK_ROOT = (
    EVAL_ROOT
    / "data"
    / "c4_calibration"
    / "offline_24_masks"
    / "c4_512_seed42_bf16_max512"
)
TOKEN_DENSE_BUDGETS = (16, 32, 64, 128)
SUPPORTED_TOKEN_DENSE_BUDGETS = (0, 8, *TOKEN_DENSE_BUDGETS, 256)
TOKEN_DENSE_METHODS = [f"token_dense_d{budget}" for budget in TOKEN_DENSE_BUDGETS]
TOKEN_DENSE_DYNAMIC_METHOD = "token_dense_dynamic"
DEFAULT_TOKEN_DENSE_METHODS = ",".join(TOKEN_DENSE_METHODS)


@dataclass(frozen=True)
class MethodConfig:
    label: str
    base_method: str
    policy: str = "all_sparse"
    keep_n: int = 0
    token_dense_budget: int | None = None
    token_dense_budget_mode: str = "fixed"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_method_config(label: str) -> MethodConfig:
    if label == "activation_aware":
        return MethodConfig(label=label, base_method="activation_aware")
    match = re.fullmatch(r"activation_aware_(keep_first_last|keep_first|keep_last)_(\d+)", label)
    if match:
        return MethodConfig(
            label=label,
            base_method="activation_aware",
            policy=match.group(1),
            keep_n=int(match.group(2)),
        )
    if label == TOKEN_DENSE_DYNAMIC_METHOD:
        return MethodConfig(
            label=label,
            base_method="token_dense",
            token_dense_budget_mode="dynamic",
        )
    token_dense_match = re.fullmatch(r"token_dense_d(\d{1,3})", label)
    if token_dense_match:
        budget = int(token_dense_match.group(1))
        if budget not in SUPPORTED_TOKEN_DENSE_BUDGETS:
            raise ValueError(
                f"unsupported token-dense budget in {label!r}; "
                f"use one of {SUPPORTED_TOKEN_DENSE_BUDGETS}"
            )
        return MethodConfig(
            label=label,
            base_method="token_dense",
            token_dense_budget=budget,
        )
    raise ValueError(
        f"unsupported method label {label!r}; use activation_aware or "
        "token_dense_dynamic/token_dense_d0/d8/d16/d32/d64/d128/d256"
    )


def method_env(
    args: Any,
    *,
    model_label: str,
    method: MethodConfig,
    stats_path: Path,
) -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    env["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"
    env["SPECLINK_STRUCTURED_24_DYNAMIC_CUTLASS_BACKEND"] = "0"
    if method.label == "dense":
        env["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
        return env

    env.update(
        {
            "SPECLINK_STRUCTURED_24_ENABLE": "1",
            "SPECLINK_STRUCTURED_24_MODEL_LABEL": model_label,
            "SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT": str(
                args.calibration_cache_root.resolve()
            ),
            "SPECLINK_STRUCTURED_24_POLICY": method.policy,
            "SPECLINK_STRUCTURED_24_KEEP_N": str(method.keep_n),
            "SPECLINK_STRUCTURED_24_STATS_PATH": str(stats_path.resolve()),
        }
    )
    if method.base_method == "token_dense":
        mask_method = str(getattr(args, "token_dense_mask_method", "wanda") or "").strip()
        sparse_output_mode = str(
            getattr(args, "token_dense_sparse_output_mode", "contiguous")
        )
        sparse_output_policies = {
            "contiguous": "0",
            "fused_mlp": "gate_up",
            "view_mlp": "mlp",
            "view_mlp_o": "layerwise",
        }
        if sparse_output_mode not in sparse_output_policies:
            raise ValueError(
                f"unsupported token_dense_sparse_output_mode={sparse_output_mode!r}"
            )
        mask_root = Path(
            getattr(args, "token_dense_mask_root", DEFAULT_TOKEN_DENSE_MASK_ROOT)
            or DEFAULT_TOKEN_DENSE_MASK_ROOT
        )
        if mask_method == "inherit":
            if not env.get("SPECLINK_STRUCTURED_24_MASK_CACHE", "").strip():
                raise ValueError(
                    "token_dense_mask_method=inherit requires "
                    "SPECLINK_STRUCTURED_24_MASK_CACHE"
                )
        elif mask_method == "none":
            env.pop("SPECLINK_STRUCTURED_24_MASK_CACHE", None)
            env.pop("SPECLINK_STRUCTURED_24_CACHE_STRICT", None)
        elif mask_method:
            env["SPECLINK_STRUCTURED_24_MASK_CACHE"] = str(
                (mask_root / model_label / f"{mask_method}.pt").resolve()
            )
            env["SPECLINK_STRUCTURED_24_CACHE_STRICT"] = "1"
        env.update(
            {
                "SPECLINK_TOKEN_DENSE_ENABLE": "1",
                "SPECLINK_TOKEN_DENSE_MODE": "topk_cumulative_confidence",
                "SPECLINK_TOKEN_DENSE_DENSE_TOKENS": str(
                    method.token_dense_budget
                    if method.token_dense_budget is not None
                    else 32
                ),
                "SPECLINK_TOKEN_DENSE_BUDGET_MODE": (
                    method.token_dense_budget_mode
                ),
                "SPECLINK_TOKEN_DENSE_DENSE_RATIO": str(
                    getattr(args, "token_dense_dense_ratio", 0.125)
                ),
                "SPECLINK_TOKEN_DENSE_DENSE_MIN_PER_REQUEST": str(
                    getattr(args, "token_dense_dense_min_per_request", 1)
                ),
                "SPECLINK_TOKEN_DENSE_DENSE_CAP": str(
                    getattr(args, "token_dense_dense_cap", -1)
                ),
                "SPECLINK_TOKEN_DENSE_DENSE_SELECTION": str(
                    getattr(args, "token_dense_dense_selection", "highest")
                ),
                "SPECLINK_TOKEN_DENSE_BATCH_CONFIDENCE_THRESHOLD": str(
                    getattr(
                        args,
                        "token_dense_batch_confidence_threshold",
                        0.364,
                    )
                ),
                "SPECLINK_TOKEN_DENSE_ADAPTIVE_DENSE_MAX_REQUESTS": str(
                    getattr(
                        args,
                        "token_dense_adaptive_dense_max_requests",
                        1,
                    )
                ),
                "SPECLINK_TOKEN_DENSE_BATCH_ROUTE_BLOCK_STEPS": str(
                    getattr(args, "token_dense_batch_route_block_steps", 1)
                ),
                "SPECLINK_TOKEN_DENSE_BATCH_ROUTE_INITIAL_CREDIT": str(
                    getattr(args, "token_dense_batch_route_initial_credit", 0.0)
                ),
                "SPECLINK_TOKEN_DENSE_BALANCED_START_POSITION": str(
                    getattr(args, "token_dense_balanced_start_position", 0)
                ),
                "SPECLINK_TOKEN_DENSE_SPARSE_BONUS": (
                    "1" if getattr(args, "token_dense_sparse_bonus", False) else "0"
                ),
                "SPECLINK_TOKEN_DENSE_SPARSE_UNSCORED_DECODE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_sparse_unscored_decode",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_LINEAR_STRATEGY": str(
                    getattr(
                        args,
                        "token_dense_linear_strategy",
                        "auto",
                    )
                ),
                "SPECLINK_TOKEN_DENSE_MLP_STRATEGY": str(
                    getattr(args, "token_dense_mlp_strategy", "auto")
                ),
                "SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP": (
                    "1"
                    if getattr(args, "token_dense_fused_batch_mlp", False)
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_FULL_SPARSE_OVERRIDE_MLP": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_full_sparse_override_mlp",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_FULL_SPARSE_OVERRIDE_MLP_MIN_ROWS": str(
                    getattr(
                        args,
                        "token_dense_full_sparse_override_mlp_min_rows",
                        224,
                    )
                ),
                "SPECLINK_TOKEN_DENSE_INLINE_SWIGLU_MLP": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_inline_swiglu_mlp",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_ROUTED_SWIGLU_MLP": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_routed_swiglu_mlp",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_SPARSE_GATE_DENSE_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_sparse_gate_dense_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_DIRECT_GATE_RESIDUAL_EPILOGUE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_direct_gate_residual_epilogue",
                        True,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_GATE_DOWN_GATHER_EPILOGUE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_gate_down_gather_epilogue",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_CONTIGUOUS_DOWN_INPUT": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_contiguous_down_input",
                        True,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_UNINITIALIZED_ROUTED_WORKSPACE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_uninitialized_routed_workspace",
                        True,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_PERSISTENT_GATE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_persistent_gate",
                        True,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_GATE_SHAPE_TUNING": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_gate_shape_tuning",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_FUSED_GATE_EPILOGUE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_fused_gate_epilogue",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_PERSISTENT_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_persistent_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_GATHER_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_gather_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_INPLACE_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_inplace_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PAIRED_INPLACE_O": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_paired_inplace_o",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PARALLEL_SPLITK_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_parallel_splitk_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_SINGLE_LAUNCH_SPLITK_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_single_launch_splitk_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_TILED_INDEXED_SPLITK_DOWN": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_tiled_indexed_splitk_down",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_INDEXED_DOWN_EPILOGUE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_indexed_down_epilogue",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_INDEXED_OUTPUT_EPILOGUE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_indexed_output_epilogue",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_QKV_HETEROGENEOUS_ROUTING": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_qkv_heterogeneous_routing",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_QKV_HETEROGENEOUS_MAX_ROWS": str(
                    getattr(
                        args,
                        "token_dense_qkv_heterogeneous_max_rows",
                        704,
                    )
                ),
                "SPECLINK_SPARSE24_QKV_PAIRED_ROUTING": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_qkv_paired_routing",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_QKV_PAIRED_MAX_ROWS": str(
                    getattr(
                        args,
                        "token_dense_qkv_paired_max_rows",
                        704,
                    )
                ),
                "SPECLINK_SPARSE24_QKV_ACTIVE_WAVE_C12": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_qkv_active_wave_c12",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_QKV_VEC4_POSTOP": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_qkv_vec4_postop",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_QKV_FUSED_EPILOGUE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_qkv_fused_epilogue",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_QKV_DIRECT_CACHE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_qkv_direct_cache",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_PROJECTION_POLICY": str(
                    getattr(args, "token_dense_projection_policy", "all")
                ),
                "SPECLINK_TOKEN_DENSE_MIXED_PROJECTION_POLICY": str(
                    getattr(
                        args,
                        "token_dense_mixed_projection_policy",
                        "all",
                    )
                ),
                "SPECLINK_TOKEN_DENSE_MIXED_LAYERS": str(
                    getattr(args, "token_dense_mixed_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_MLP_STATIC_LAYERS": str(
                    getattr(args, "token_dense_mlp_static_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_O_SPARSE_LAYERS": str(
                    getattr(args, "token_dense_o_sparse_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_GATE_UP_DENSE_LAYERS": str(
                    getattr(args, "token_dense_gate_up_dense_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_DOWN_DENSE_LAYERS": str(
                    getattr(args, "token_dense_down_dense_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_ATTENTION_DENSE_LAYERS": str(
                    getattr(args, "token_dense_attention_dense_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_DENSE_LAYERS": str(
                    getattr(args, "token_dense_dense_layers", "")
                ),
                "SPECLINK_TOKEN_DENSE_SCORE_BACKEND": str(
                    getattr(
                        args,
                        "token_dense_score_backend",
                        "triton_fused",
                    )
                ),
                "SPECLINK_TOKEN_DENSE_FAST_PLAN": (
                    "1" if getattr(args, "token_dense_fast_plan", True) else "0"
                ),
                "SPECLINK_TOKEN_DENSE_GRAPH_ROUTING": (
                    "1"
                    if str(
                        getattr(args, "token_dense_cudagraph_mode", "none")
                    ).lower()
                    in {"full", "full_decode_only"}
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_NUM_SPEC_TOKENS": str(
                    getattr(args, "num_spec_tokens", 8)
                ),
                "SPECLINK_DECODE_ONLY_ISOLATE_BATCHES": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_isolate_decode_batches",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_REUSE_BUFFERS": (
                    "1" if getattr(args, "token_dense_reuse_buffers", True) else "0"
                ),
                "SPECLINK_SPARSE24_CONTIGUOUS_SCATTER": (
                    "1"
                    if getattr(args, "token_dense_contiguous_scatter", True)
                    else "0"
                ),
                "SPECLINK_SPARSE24_PARALLEL_MIXED_OVERRIDE": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_parallel_mixed_override",
                        True,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_VALUE_SCALE": str(
                    getattr(args, "token_dense_sparse_value_scale", 1.0)
                ),
                "SPECLINK_SPARSE24_GATE_UP_VALUE_SCALE": str(
                    getattr(args, "token_dense_gate_up_value_scale", 1.0)
                ),
                "SPECLINK_SPARSE24_GATE_UP_HYBRID": str(
                    getattr(args, "token_dense_gate_up_hybrid", "none")
                ),
                "SPECLINK_SPARSE24_GROUP_RECONSTRUCTION": (
                    "1"
                    if getattr(args, "token_dense_group_reconstruction", False)
                    else "0"
                ),
                "SPECLINK_SPARSE24_GROUP_COVARIANCE_CACHE": str(
                    (
                        Path(args.token_dense_group_covariance_cache)
                        if getattr(
                            args,
                            "token_dense_group_covariance_cache",
                            None,
                        )
                        else mask_root
                        / model_label
                        / "gate_group_covariances.pt"
                    ).resolve()
                ),
                "SPECLINK_SPARSE24_ROW_SCALE_MODE": str(
                    getattr(args, "token_dense_row_scale_mode", "cache")
                ),
                "SPECLINK_SPARSE24_VARIANCE_SCALE_PROJECTION_POLICY": str(
                    getattr(
                        args,
                        "token_dense_variance_scale_projection_policy",
                        "all",
                    )
                ),
                "SPECLINK_SPARSE24_ROW_SCALE_MAX": str(
                    getattr(args, "token_dense_row_scale_max", 1.25)
                ),
                "SPECLINK_SPARSE24_ACCUMULATOR": str(
                    getattr(args, "token_dense_sparse_accumulator", "fp32")
                ),
                "SPECLINK_SPARSE24_DIRECT_STORE_GATE_UP": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_direct_store_gate_up",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_TOKEN_DENSE_CUTLASS_DOWN_FP16": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_cutlass_down_fp16",
                        False,
                    )
                    else "0"
                ),
                # Module-level sparse/dense bypass attributes are constants in
                # the AOT graph but are not part of vLLM's cross-process cache
                # key. Reusing a graph across layer/projection policies is
                # incorrect; CUDA graph capture remains enabled independently.
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "SPECLINK_SPARSE24_SKIP_TRANSPOSE": (
                    sparse_output_policies[sparse_output_mode]
                ),
                "SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT": (
                    "1"
                    if sparse_output_mode != "contiguous"
                    else "0"
                ),
                "SPECLINK_SPARSE24_TRANSPOSED_MLP_FUSION": (
                    "1" if sparse_output_mode == "fused_mlp" else "0"
                ),
                "SPECLINK_SPARSE24_RELEASE_DENSE_WEIGHT": (
                    "1"
                    if getattr(
                        args,
                        "token_dense_release_dense_weights",
                        False,
                    )
                    else "0"
                ),
                "SPECLINK_SPARSE24_RETAIN_DENSE_WEIGHT": str(
                    getattr(
                        args,
                        "token_dense_retain_dense_weight",
                        "none",
                    )
                ),
                "SPECLINK_TOKEN_DENSE_STATS_PATH": str(
                    (stats_path.parent / "token_dense_stats.jsonl").resolve()
                ),
                "SPECLINK_TOKEN_DENSE_STATS_DETAIL": "0",
                "SPECLINK_STRUCTURED_24_DYNAMIC_CUTLASS_BACKEND": "1",
            }
        )
        if getattr(args, "production_fast", False):
            env["SPECLINK_PRODUCTION_FAST"] = "1"
            env["SPECLINK_TOKEN_DENSE_STATS_INTERVAL"] = "0"
    return env
