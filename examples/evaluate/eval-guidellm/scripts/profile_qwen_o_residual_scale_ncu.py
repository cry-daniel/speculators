#!/usr/bin/env python3
"""Profile the small-M Qwen o_proj residual-complement components with NCU."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

from sparse24_benchmark_common import (
    ShapeCase,
    capture_graph,
    generate_routes,
    idle_state,
    make_input,
    make_synthetic_weight,
    route_from_record,
    route_key,
)
from speculators.speclink import (
    SPARSE_RESIDUAL_SMEM,
    TP1_FUSED_WEIGHT_SHAPES,
    cusparselt_sparse_residual_kernel_attributes,
    cusparselt_sparse_residual_residual_linear,
    cusparselt_sparse_residual_sparse_linear,
    prepare_cusparselt_sparse_residual_weight,
    prepare_online_sparse24_weight,
    prepare_sparse24_weight,
    select_cusparselt_algorithm,
    sparse24_linear,
)


MODELS = ("qwen3_8b", "qwen3_14b", "qwen3_32b")
M_VALUES = (512, 1024)
DENSE_FRACTION = Fraction(1, 8)
SEED = 20260720
WARMUP_REPLAYS = 100

CUBLAS_FULL = "cublas_full"
CUSPARSELT_BASE_FULL = "cusparselt_base_full"
COMPLEMENT_DENSE_FRACTION = "complement_dense_fraction"
METHODS = (CUBLAS_FULL, CUSPARSELT_BASE_FULL, COMPLEMENT_DENSE_FRACTION)


def check(
    actual: torch.Tensor, expected: torch.Tensor, *, context: str
) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    result = {
        "allclose": bool(
            torch.allclose(
                actual.float(), expected.float(), rtol=5e-2, atol=5e-2
            )
        ),
        "max_abs_error": float(difference.max().item()),
    }
    if not result["allclose"]:
        raise RuntimeError(f"{context} correctness failed: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with real GPU access")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    n, k = TP1_FUSED_WEIGHT_SHAPES[args.model]["o"]
    case = ShapeCase(args.model, "o", args.m, k, n)
    route = route_from_record(
        generate_routes([args.m], [DENSE_FRACTION], args.seed)["routes"]
        [route_key(args.m, DENSE_FRACTION)],
        device,
    )
    dense_rows = args.m // 8
    if route.dense_count != dense_rows:
        raise RuntimeError("route dense count changed")

    weight, weight24 = make_synthetic_weight(case, args.seed, device)
    x = make_input(case, args.seed, device, purpose="qwen_o_scale_ncu")
    x_dense = x.index_select(0, route.dense_indices).contiguous()
    algorithm_id: int | None = None
    attributes: dict[str, Any] | None = None

    if args.profile_method == CUBLAS_FULL:
        call = lambda: F.linear(x, weight).contiguous()
        eager = call()
        correctness = {"finite": bool(torch.isfinite(eager).all().item())}
    elif args.profile_method == CUSPARSELT_BASE_FULL:
        sparse = prepare_sparse24_weight(weight24)
        algorithm_id = select_cusparselt_algorithm(sparse, x)
        call = lambda: sparse24_linear(x, sparse)
        eager = call()
        correctness = {
            "base_reference": check(
                eager, F.linear(x, weight24), context="cuSPARSELt base"
            )
        }
    else:
        canonical = prepare_online_sparse24_weight(
            weight, weight24, variant=SPARSE_RESIDUAL_SMEM
        )
        runtime = prepare_cusparselt_sparse_residual_weight(
            canonical, sparse_weight=weight24
        )
        call = lambda: cusparselt_sparse_residual_residual_linear(
            x_dense, runtime
        )
        correction = call()
        base_dense = cusparselt_sparse_residual_sparse_linear(x_dense, runtime)
        correctness = {
            "base_plus_complement_reference": check(
                base_dense + correction,
                F.linear(x_dense, weight),
                context="base plus complement",
            )
        }
        attributes = cusparselt_sparse_residual_kernel_attributes(
            dense_rows, n, residual_only=True
        )

    graph = capture_graph(call, warmup=args.capture_warmup, unroll=1)
    if graph.unroll != 1:
        raise RuntimeError("profile graph must contain one call")
    for _ in range(WARMUP_REPLAYS):
        graph.graph.replay()
    torch.cuda.synchronize(device)

    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(
        f"profile_{args.model}_o_m{args.m}_{args.profile_method}"
    )
    graph.graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.profiler.stop()

    payload: dict[str, Any] = {
        "case": case.key,
        "model": args.model,
        "projection": "o",
        "profile_method": args.profile_method,
        "M_original": args.m,
        "M_executed": (
            dense_rows
            if args.profile_method == COMPLEMENT_DENSE_FRACTION
            else args.m
        ),
        "N": n,
        "K": k,
        "dense_rows": dense_rows,
        "sparse_rows": args.m - dense_rows,
        "weight_elements": n * k,
        "graph_unroll": 1,
        "warmup_replays": WARMUP_REPLAYS,
        "ncu_expected_cache_control": "all",
        "cusparselt_algorithm_id": algorithm_id,
        "correctness": correctness,
    }
    if attributes is not None:
        payload["kernel_attributes"] = attributes
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--m", choices=M_VALUES, type=int, required=True)
    parser.add_argument("--profile-method", choices=METHODS, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--busy-util-threshold", type=int, default=5)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-sample-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if not args.profile_only:
        parser.error("this harness requires --profile-only")
    return args


def main() -> int:
    args = parse_args()
    state = idle_state(args.device_index)
    if state["compute_processes"] and not args.allow_busy_gpu:
        raise RuntimeError(f"GPU is busy: {state['compute_processes']}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
