#!/usr/bin/env python3
"""Profile old Qwen3-8B qkv one-weight concurrent separate vs cuBLAS.

The old graph executes the final BM64 prebroadcast-full-tiles dense M=256 and
online-pack sparse M=1792 branches on distinct CUDA streams.  D29:S141 is the
shape-screened quota for the [6144,4096] qkv weight and exactly partitions the
170 SMs on the RTX 5090.  The comparison graph is one cuBLAS BF16 dense M=2048
GEMM.  The two methods have the same input/output shape but intentionally have
different numerical semantics on the 1792 sparse rows.

Use ``--mode profile`` under Nsight Compute whole-graph profiling.  Use
``--mode formal`` for the clean 100-warmup, 10x1000 CUDA Event latency pair.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sparse24_benchmark_common import (
    CapturedGraph,
    ShapeCase,
    capture_graph,
    capture_multistream_graph,
    create_multistream_resources,
    generate_routes,
    gpu_identity,
    idle_state,
    launch_two_branch_concurrent,
    make_input,
    make_synthetic_weight,
    route_from_record,
    route_key,
    steady_graph_sample,
    summarize,
    write_csv,
    write_json,
)
from speculators.speclink import (
    OLD_CONCURRENT_DENSE_BRANCH,
    OLD_CONCURRENT_SPARSE_BRANCH,
    TP1_FUSED_WEIGHT_SHAPES,
    old_concurrent_branch_linear_out,
    old_concurrent_kernel_attributes,
    prepare_old_concurrent_weight,
)


MODEL = "qwen3_8b"
PROJECTION = "qkv"
M = 2048
DENSE_FRACTION = Fraction(1, 8)
DENSE_ROWS = 256
SPARSE_ROWS = 1792
DENSE_BLOCKS = 29
SPARSE_BLOCKS = 141
WARMUP_REPLAYS = 100
TRIALS = 10
REPLAYS_PER_SAMPLE = 1000
SEED = 20260721

OLD_CONCURRENT = "old_dense_base_concurrent_d29_s141"
CUBLAS_DENSE = "cublas_dense_m2048"
METHODS = (OLD_CONCURRENT, CUBLAS_DENSE)
LABELS = {
    OLD_CONCURRENT: "Old one-weight concurrent: dense M256 + sparse M1792",
    CUBLAS_DENSE: "cuBLAS dense M2048",
}


@dataclass(slots=True)
class Workloads:
    device: torch.device
    graphs: dict[str, CapturedGraph]
    checks: dict[str, dict[str, Any]]
    attributes: dict[str, Any]
    stream_contract: dict[str, Any]


def check(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    context: str,
) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    result = {
        "allclose": bool(
            torch.allclose(
                actual.float(), expected.float(), rtol=5e-2, atol=5e-2
            )
        ),
        "max_abs_error": float(difference.max().item()),
        "output_contiguous": bool(actual.is_contiguous()),
    }
    if not result["allclose"] or not result["output_contiguous"]:
        raise RuntimeError(f"{context} correctness failed: {result}")
    return result


def prepare_workloads(args: argparse.Namespace) -> Workloads:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with real GPU access")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    n, k = TP1_FUSED_WEIGHT_SHAPES[MODEL][PROJECTION]
    if (n, k) != (6144, 4096):
        raise RuntimeError(f"Qwen3-8B qkv shape changed: {(n, k)}")
    case = ShapeCase(MODEL, PROJECTION, M, k, n)
    route = route_from_record(
        generate_routes([M], [DENSE_FRACTION], args.seed)["routes"]
        [route_key(M, DENSE_FRACTION)],
        device,
    )
    if (route.dense_count, route.sparse_count) != (DENSE_ROWS, SPARSE_ROWS):
        raise RuntimeError("route must be exactly 256 dense + 1792 sparse rows")

    weight, weight24 = make_synthetic_weight(case, args.seed, device)
    x = make_input(case, args.seed, device, purpose="qwen8_qkv_old_ncu")
    runtime = prepare_old_concurrent_weight(weight, weight24)
    old_output = torch.zeros((M, n), dtype=torch.bfloat16, device=device)
    resources = create_multistream_resources(device)

    def branch_call(branch: str, persistent_blocks: int) -> torch.Tensor:
        return old_concurrent_branch_linear_out(
            x,
            runtime,
            route.dense_indices,
            route.sparse_indices,
            old_output,
            branch=branch,
            persistent_blocks=persistent_blocks,
        )

    def old_call() -> torch.Tensor:
        return launch_two_branch_concurrent(
            lambda: branch_call(OLD_CONCURRENT_DENSE_BRANCH, DENSE_BLOCKS),
            lambda: branch_call(OLD_CONCURRENT_SPARSE_BRANCH, SPARSE_BLOCKS),
            old_output,
            resources,
            device=device,
        )

    def cublas_call() -> torch.Tensor:
        return F.linear(x, weight).contiguous()

    dense_x = x.index_select(0, route.dense_indices).contiguous()
    sparse_x = x.index_select(0, route.sparse_indices).contiguous()
    hybrid_reference = torch.empty_like(old_output)
    hybrid_reference.index_copy_(
        0, route.dense_indices, F.linear(dense_x, weight)
    )
    hybrid_reference.index_copy_(
        0, route.sparse_indices, F.linear(sparse_x, weight24)
    )
    dense_reference = F.linear(x, weight).contiguous()

    old_eager = old_call()
    cublas_eager = cublas_call()
    torch.cuda.synchronize(device)
    checks = {
        "old_eager_vs_routed_reference": check(
            old_eager, hybrid_reference, context="old eager"
        ),
        "cublas_eager_vs_dense_reference": check(
            cublas_eager, dense_reference, context="cuBLAS eager"
        ),
    }

    graphs = {
        OLD_CONCURRENT: capture_multistream_graph(
            old_call,
            resources,
            warmup=args.capture_warmup,
            device=device,
        ),
        CUBLAS_DENSE: capture_graph(
            cublas_call, warmup=args.capture_warmup, unroll=1
        ),
    }
    checks.update(
        {
            "old_graph_vs_routed_reference": check(
                graphs[OLD_CONCURRENT].output,
                hybrid_reference,
                context="old graph",
            ),
            "cublas_graph_vs_dense_reference": check(
                graphs[CUBLAS_DENSE].output,
                dense_reference,
                context="cuBLAS graph",
            ),
        }
    )

    dense_attributes = old_concurrent_kernel_attributes(
        OLD_CONCURRENT_DENSE_BRANCH
    )
    sparse_attributes = old_concurrent_kernel_attributes(
        OLD_CONCURRENT_SPARSE_BRANCH
    )
    for name, raw in (("dense", dense_attributes), ("sparse", sparse_attributes)):
        if int(raw.get("local_bytes", -1)) != 0:
            raise RuntimeError(f"{name} branch spill contract changed: {raw}")
    handles = {
        "capture_stream": int(resources.capture_stream.cuda_stream),
        "dense_stream": int(resources.dense_stream.cuda_stream),
        "sparse_stream": int(resources.sparse_stream.cuda_stream),
    }
    if len(set(handles.values())) != len(handles):
        raise RuntimeError(f"worker streams are not distinct: {handles}")
    return Workloads(
        device=device,
        graphs=graphs,
        checks=checks,
        attributes={"dense": dense_attributes, "sparse": sparse_attributes},
        stream_contract={
            **handles,
            "worker_streams_distinct": True,
            "explicit_fork_join_events": True,
            "graph_nodes": {OLD_CONCURRENT: 2, CUBLAS_DENSE: 1},
        },
    )


def warmup(selected: dict[str, CapturedGraph], device: torch.device) -> None:
    for method in METHODS:
        if selected[method].unroll != 1:
            raise RuntimeError(f"{method}: graph must contain one logical call")
        for _ in range(WARMUP_REPLAYS):
            selected[method].graph.replay()
    torch.cuda.synchronize(device)


def run_formal(args: argparse.Namespace, workloads: Workloads) -> dict[str, Any]:
    if args.output_root is None:
        raise ValueError("--output-root is required in formal mode")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    warmup(workloads.graphs, workloads.device)

    values: dict[str, list[float]] = {method: [] for method in METHODS}
    raw_rows: list[dict[str, Any]] = []
    orders = [(OLD_CONCURRENT, CUBLAS_DENSE), (CUBLAS_DENSE, OLD_CONCURRENT)] * 5
    for trial, order in enumerate(orders):
        for order_index, method in enumerate(order):
            host_start = time.perf_counter()
            latency_us = steady_graph_sample(
                workloads.graphs[method], REPLAYS_PER_SAMPLE
            )
            enqueue_ms = (time.perf_counter() - host_start) * 1000.0
            total_ms = latency_us * REPLAYS_PER_SAMPLE / 1000.0
            enqueue_ratio = enqueue_ms / total_ms
            values[method].append(latency_us)
            raw_rows.append(
                {
                    "trial": trial,
                    "method": method,
                    "method_label": LABELS[method],
                    "order": ",".join(order),
                    "order_index": order_index,
                    "latency_us": latency_us,
                    "interval_total_ms": total_ms,
                    "host_enqueue_ms": enqueue_ms,
                    "enqueue_gpu_ratio": enqueue_ratio,
                    "warmup_replays": WARMUP_REPLAYS,
                    "replays_per_sample": REPLAYS_PER_SAMPLE,
                    "graph_unroll": 1,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        stats = summarize(values[method])
        summary_rows.append(
            {
                "method": method,
                "method_label": LABELS[method],
                "median_us": stats["median_us"],
                "p10_us": stats["p10_us"],
                "p90_us": stats["p90_us"],
                "mean_us": stats["mean_us"],
            }
        )
    medians = {
        row["method"]: float(row["median_us"]) for row in summary_rows
    }
    comparison = {
        "old_over_cublas_latency_ratio": (
            medians[OLD_CONCURRENT] / medians[CUBLAS_DENSE]
        ),
        "old_speedup_vs_cublas": (
            medians[CUBLAS_DENSE] / medians[OLD_CONCURRENT]
        ),
    }
    write_csv(root / "formal_raw.csv", raw_rows)
    write_csv(root / "formal_summary.csv", summary_rows)
    payload = {
        "case": "qwen3_8b__qkv__m2048",
        "M": M,
        "N": 6144,
        "K": 4096,
        "dense_rows": DENSE_ROWS,
        "sparse_rows": SPARSE_ROWS,
        "dense_blocks": DENSE_BLOCKS,
        "sparse_blocks": SPARSE_BLOCKS,
        "methods": list(METHODS),
        "semantic_warning": (
            "same input/output shape, but hybrid sparse rows use the 2:4 weight"
        ),
        "protocol": {
            "warmup_replays": WARMUP_REPLAYS,
            "trials": TRIALS,
            "replays_per_sample": REPLAYS_PER_SAMPLE,
            "order": "exact 5:5 pairwise balanced",
            "cache_state": "natural steady state; NCU separately uses cache-control all",
        },
        "checks": workloads.checks,
        "kernel_attributes": workloads.attributes,
        "stream_contract": workloads.stream_contract,
        "comparison": comparison,
    }
    write_json(root / "formal_metadata.json", payload)
    print(json.dumps({**comparison, "output_root": str(root)}, sort_keys=True))
    return payload


def run_profile(args: argparse.Namespace, workloads: Workloads) -> dict[str, Any]:
    if args.profile_method is None:
        raise ValueError("--profile-method is required in profile mode")
    graph = workloads.graphs[args.profile_method]
    for _ in range(WARMUP_REPLAYS):
        graph.graph.replay()
    torch.cuda.synchronize(workloads.device)

    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"profile_{args.profile_method}")
    graph.graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(workloads.device)
    torch.cuda.profiler.stop()

    payload = {
        "case": "qwen3_8b__qkv__m2048",
        "profile_method": args.profile_method,
        "method_label": LABELS[args.profile_method],
        "M": M,
        "N": 6144,
        "K": 4096,
        "dense_rows": DENSE_ROWS,
        "sparse_rows": SPARSE_ROWS,
        "dense_blocks": DENSE_BLOCKS if args.profile_method == OLD_CONCURRENT else None,
        "sparse_blocks": SPARSE_BLOCKS if args.profile_method == OLD_CONCURRENT else None,
        "graph_nodes": workloads.stream_contract["graph_nodes"][args.profile_method],
        "graph_unroll": graph.unroll,
        "warmup_replays": WARMUP_REPLAYS,
        "ncu_required_mode": "kernel replay + graph-profiling graph",
        "ncu_cache_control": "all",
        "semantic_warning": (
            "hybrid and dense have the same shape but different sparse-row semantics"
        ),
        "checks": workloads.checks,
        "kernel_attributes": workloads.attributes,
        "stream_contract": workloads.stream_contract,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "profile"), required=True)
    parser.add_argument("--profile-method", choices=METHODS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--busy-util-threshold", type=int, default=5)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-sample-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if args.mode == "profile" and args.profile_method is None:
        parser.error("--profile-method is required in profile mode")
    if args.mode == "formal" and args.output_root is None:
        parser.error("--output-root is required in formal mode")
    return args


def main() -> int:
    args = parse_args()
    before = idle_state(args.device_index)
    if before["compute_processes"] and not args.allow_busy_gpu:
        raise RuntimeError(f"GPU is busy: {before['compute_processes']}")
    workloads = prepare_workloads(args)
    if args.mode == "formal":
        payload = run_formal(args, workloads)
        payload["gpu"] = gpu_identity(args.device_index)
        payload["idle_before"] = before
        assert args.output_root is not None
        write_json(args.output_root.resolve() / "measurement_provenance.json", payload)
    else:
        run_profile(args, workloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
