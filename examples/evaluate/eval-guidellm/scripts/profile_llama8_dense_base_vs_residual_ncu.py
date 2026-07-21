#!/usr/bin/env python3
"""Expose matched Llama3.1-8B GEMM roles to Nsight Compute.

The old dense-base one-weight implementation is profiled as two isolated,
full-SM pure-role BM64 kernels.  This deliberately removes the concurrent
quota as a confounder and permits equal-work comparisons against cuBLAS M=256
and prepared cuSPARSELt M=1792.  The current sparse-base representation is
represented by prepared cuSPARSELt M=2048 plus its M=256 complement-HMMA.SP
correction.  Formal graph E2E latency comes from the separate benchmark; NCU
kernel replay is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any, Callable

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
    cusparselt_sparse_residual_residual_linear,
    cusparselt_sparse_residual_kernel_attributes,
    cusparselt_sparse_residual_sparse_linear,
    old_concurrent_branch_linear_out,
    old_concurrent_kernel_attributes,
    prepare_cusparselt_sparse_residual_weight,
    prepare_old_concurrent_weight,
    prepare_online_sparse24_weight,
    prepare_sparse24_weight,
    select_cusparselt_algorithm,
    sparse24_linear,
)


MODEL = "llama3_1_8b"
M = 2048
DENSE_FRACTION = Fraction(1, 8)
DENSE_ROWS = 256
SPARSE_ROWS = 1792
SEED = 20260720
WARMUP_REPLAYS = 100
FULL_SM_PERSISTENT_BLOCKS = 170
PROJECTIONS = ("qkv", "o", "gate_up", "down")

OLD_DENSE_M256 = "old_dense_m256"
CUBLAS_M256 = "cublas_m256"
OLD_SPARSE_M1792 = "old_sparse_m1792"
CUSPARSELT_M1792 = "cusparselt_m1792"
CUBLAS_M2048 = "cublas_m2048"
CUSPARSELT_M2048 = "cusparselt_m2048"
COMPLEMENT_M256 = "complement_m256"
METHODS = (
    OLD_DENSE_M256,
    CUBLAS_M256,
    OLD_SPARSE_M1792,
    CUSPARSELT_M1792,
    CUBLAS_M2048,
    CUSPARSELT_M2048,
    COMPLEMENT_M256,
)


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

    n, k = TP1_FUSED_WEIGHT_SHAPES[MODEL][args.projection]
    case = ShapeCase(MODEL, args.projection, M, k, n)
    routes = generate_routes([M], [DENSE_FRACTION], args.seed)
    route = route_from_record(
        routes["routes"][route_key(M, DENSE_FRACTION)], device
    )
    if (route.dense_count, route.sparse_count) != (DENSE_ROWS, SPARSE_ROWS):
        raise RuntimeError("route must be exactly 256 dense + 1792 sparse")

    weight, weight24 = make_synthetic_weight(case, args.seed, device)
    x = make_input(case, args.seed, device, purpose="llama8_dense_base_ncu")
    x_dense = x.index_select(0, route.dense_indices).contiguous()
    x_sparse = x.index_select(0, route.sparse_indices).contiguous()

    selected = args.profile_method
    sparse1792_algorithm: int | None = None
    sparse2048_algorithm: int | None = None
    kernel_attributes: dict[str, Any] | None = None

    if selected in {OLD_DENSE_M256, OLD_SPARSE_M1792}:
        old_runtime = prepare_old_concurrent_weight(weight, weight24)
        old_output = torch.zeros((M, n), dtype=torch.bfloat16, device=device)
        branch = "dense" if selected == OLD_DENSE_M256 else "sparse"

        def call() -> torch.Tensor:
            return old_concurrent_branch_linear_out(
                x,
                old_runtime,
                route.dense_indices,
                route.sparse_indices,
                old_output,
                branch=branch,
                persistent_blocks=FULL_SM_PERSISTENT_BLOCKS,
            )

        eager = call().index_select(
            0,
            route.dense_indices
            if selected == OLD_DENSE_M256
            else route.sparse_indices,
        )
        if selected == OLD_DENSE_M256:
            expected = F.linear(x_dense, weight)
            kernel_attributes = old_concurrent_kernel_attributes("dense")
        else:
            sparse1792 = prepare_sparse24_weight(weight24)
            sparse1792_algorithm = select_cusparselt_algorithm(
                sparse1792, x_sparse
            )
            expected = sparse24_linear(x_sparse, sparse1792)
            kernel_attributes = old_concurrent_kernel_attributes("sparse")
        checks = {
            f"{selected}_equal_work": check(
                eager, expected, context=f"{selected} equal-work correctness"
            )
        }
    elif selected == CUBLAS_M256:
        call = lambda: F.linear(x_dense, weight).contiguous()
        eager = call()
        checks = {"finite": bool(torch.isfinite(eager).all().item())}
    elif selected == CUSPARSELT_M1792:
        sparse1792 = prepare_sparse24_weight(weight24)
        sparse1792_algorithm = select_cusparselt_algorithm(sparse1792, x_sparse)
        call = lambda: sparse24_linear(x_sparse, sparse1792)
        eager = call()
        checks = {
            "cusparselt_m1792_reference": check(
                eager,
                F.linear(x_sparse, weight24),
                context="cuSPARSELt M1792",
            )
        }
    elif selected == CUBLAS_M2048:
        call = lambda: F.linear(x, weight).contiguous()
        eager = call()
        checks = {"finite": bool(torch.isfinite(eager).all().item())}
    elif selected == CUSPARSELT_M2048:
        sparse2048 = prepare_sparse24_weight(weight24)
        sparse2048_algorithm = select_cusparselt_algorithm(sparse2048, x)
        call = lambda: sparse24_linear(x, sparse2048)
        eager = call()
        checks = {
            "cusparselt_m2048_reference": check(
                eager,
                F.linear(x, weight24),
                context="cuSPARSELt M2048",
            )
        }
    elif selected == COMPLEMENT_M256:
        canonical = prepare_online_sparse24_weight(
            weight, weight24, variant=SPARSE_RESIDUAL_SMEM
        )
        current_runtime = prepare_cusparselt_sparse_residual_weight(
            canonical, sparse_weight=weight24
        )
        call = lambda: cusparselt_sparse_residual_residual_linear(
            x_dense, current_runtime
        )
        eager = call()
        base_dense = cusparselt_sparse_residual_sparse_linear(
            x_dense, current_runtime
        )
        checks = {
            "base_plus_complement_vs_dense_m256": check(
                base_dense + eager,
                F.linear(x_dense, weight),
                context="base plus complement versus dense M256",
            )
        }
        kernel_attributes = cusparselt_sparse_residual_kernel_attributes(
            DENSE_ROWS, n, residual_only=True
        )
    else:
        raise AssertionError(selected)

    graph = capture_graph(call, warmup=args.capture_warmup, unroll=1)
    if graph.unroll != 1:
        raise RuntimeError(f"{selected}: graph must contain one call")
    for _ in range(WARMUP_REPLAYS):
        graph.graph.replay()
    torch.cuda.synchronize(device)

    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"profile_{args.projection}_{args.profile_method}")
    graph.graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.profiler.stop()

    payload: dict[str, Any] = {
        "case": case.key,
        "projection": args.projection,
        "profile_method": args.profile_method,
        "M_original": M,
        "M_executed": {
            OLD_DENSE_M256: DENSE_ROWS,
            CUBLAS_M256: DENSE_ROWS,
            OLD_SPARSE_M1792: SPARSE_ROWS,
            CUSPARSELT_M1792: SPARSE_ROWS,
            CUBLAS_M2048: M,
            CUSPARSELT_M2048: M,
            COMPLEMENT_M256: DENSE_ROWS,
        }[args.profile_method],
        "N": n,
        "K": k,
        "dense_rows": DENSE_ROWS,
        "sparse_rows": SPARSE_ROWS,
        "old_branch_persistent_blocks": FULL_SM_PERSISTENT_BLOCKS,
        "old_branch_profile_mode": "isolated_full_sm_equal_work",
        "graph_unroll": 1,
        "warmup_replays_per_method": WARMUP_REPLAYS,
        "cache_state_before_ncu_override": "natural_steady_state",
        "ncu_expected_cache_control": "all",
        "explicit_activation_compaction_in_profiled_call": False,
        "subset_activation_prepared_before_capture": args.profile_method
        in {CUBLAS_M256, CUSPARSELT_M1792, COMPLEMENT_M256},
        "cusparselt_m1792_algorithm_id": sparse1792_algorithm,
        "cusparselt_m2048_algorithm_id": sparse2048_algorithm,
        "correctness": checks,
    }
    if kernel_attributes is not None:
        payload["kernel_attributes"] = kernel_attributes
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--projection", choices=PROJECTIONS, required=True)
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
    if args.capture_warmup <= 0:
        parser.error("capture-warmup must be positive")
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
