"""The two residual-complement execution topologies used by benchmarks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from sparse24_benchmark_common import (
    MultiStreamResources,
    launch_two_branch_concurrent,
)
from speculators.speclink import (
    cusparselt_sparse_residual_fused_dense_linear,
    cusparselt_sparse_residual_residual_linear,
    cusparselt_sparse_residual_indexed_add_,
    cusparselt_sparse_residual_indexed_copy_,
    cusparselt_sparse_residual_indexed_gather,
    cusparselt_sparse_residual_residual_linear_splitk2,
    cusparselt_sparse_residual_splitk2_indexed_add_,
    cusparselt_sparse_residual_sparse_linear,
)


WAVE_AWARE_COMPLEMENT_DISPATCH = {
    # M, dense_rows, N, K -> CUTLASS complement variant.  Every unlisted
    # shape retains the stable 128x64/S4 kernel.
    (512, 128, 4096, 4096): "feature256_token64_s3",
    # M=512, 1/8-dense out/down cases that remained below 1.1x cuBLAS.
    # These choices are made from complete two-stream E2E screens, not from
    # isolated complement latency: the smaller grids alter how cuSPARSELt
    # base CTAs and complement CTAs share the 170 SMs.  Qwen3-14B uses the
    # low-SMEM form because it wins the required HBM-cold protocol; the longer
    # Qwen3-32B K dimension favors the higher-throughput S4 form in both cache
    # states.
    (512, 64, 5120, 5120): "b_resident_feature64_token64_b2a1_p40",
    (512, 64, 5120, 8192): "feature64_token64_s4_p40",
}

# Split-K is selected only where the 1/8-dense complement grid has exactly 80
# CTAs on the 170-SM RTX 5090.  Two local K halves expose 160 otherwise
# identical low-SMEM CTAs and remove the severe last-wave underfill.  Other
# shapes retain the simpler one-grid complement until independently measured.
WAVE_AWARE_SPLITK2_KEYS = {
    (512, 64, 5120, 5120),
}

# Formal M=512, D=128 gate_up winners under the HBM-cold 10x1000 protocol.
# The fallback remains the stable N128/S3 kernel, so the optimized row-routing
# dataflow is usable by gate_up shapes beyond this measured set as well.
FUSED_GATEUP_VARIANT_DISPATCH = {
    (24576, 4096): "n128_s3",
    (28672, 4096): "f64_n128_s4",
    (34816, 5120): "f64_n128_s4",
    (51200, 5120): "n128_s3",
    (57344, 8192): "n128_s3",
}

# Fused token partitioning is beneficial only for the formally measured
# M=512, D=128 gate_up routes.  At D=64, the dense branch is too small to
# amortize partition/gather/scatter, so the concurrent separate path remains
# faster.  Keep the topology decision explicit instead of applying the D=128
# winner to every dense-token fraction.
FUSED_GATEUP_ROUTE_KEYS = {
    (512, 128, n, k) for n, k in FUSED_GATEUP_VARIANT_DISPATCH
}


def select_fused_gateup_variant(runtime: Any) -> str:
    return FUSED_GATEUP_VARIANT_DISPATCH.get(
        (int(runtime.n), int(runtime.k)), "n128_s3"
    )


def should_use_fused_gateup(
    x: torch.Tensor, dense_indices: torch.Tensor, runtime: Any
) -> bool:
    return (
        int(x.shape[0]),
        int(dense_indices.numel()),
        int(runtime.n),
        int(runtime.k),
    ) in FUSED_GATEUP_ROUTE_KEYS


def select_wave_aware_complement_variant(
    x: torch.Tensor, dense_indices: torch.Tensor, runtime: Any
) -> str:
    key = (
        int(x.shape[0]),
        int(dense_indices.numel()),
        int(runtime.n),
        int(runtime.k),
    )
    return WAVE_AWARE_COMPLEMENT_DISPATCH.get(
        key, "feature128_token64_s4"
    )


def select_wave_aware_complement_splitk2(
    x: torch.Tensor, dense_indices: torch.Tensor, runtime: Any
) -> bool:
    key = (
        int(x.shape[0]),
        int(dense_indices.numel()),
        int(runtime.n),
        int(runtime.k),
    )
    return key in WAVE_AWARE_SPLITK2_KEYS


def branch_to_output(
    branch: Callable[[], torch.Tensor],
    output: torch.Tensor,
    indices: torch.Tensor,
    *,
    optimized_copy: bool = False,
) -> torch.Tensor:
    result = branch()
    if optimized_copy:
        cusparselt_sparse_residual_indexed_copy_(output, result, indices)
    else:
        output.index_copy_(0, indices, result)
    return output


def launch_separate(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    runtime: Any,
    resources: MultiStreamResources,
    *,
    complement_variant: str = "auto",
    complement_first: bool = False,
    optimized_gather: bool = False,
    optimized_merge: bool = False,
    splitk2_complement: bool | None = None,
    splitk2_variant: str = "auto",
) -> torch.Tensor:
    """All-row prepared base || dense-row complement, then add/scatter."""

    wave_aware = complement_variant == "wave_aware"
    if splitk2_complement is None:
        splitk2_complement = wave_aware and select_wave_aware_complement_splitk2(
            x, dense_indices, runtime
        )
    if wave_aware:
        complement_variant = select_wave_aware_complement_variant(
            x, dense_indices, runtime
        )
    origin = torch.cuda.current_stream(x.device)
    resources.fork_event.record(origin)
    resources.dense_stream.wait_event(resources.fork_event)
    resources.sparse_stream.wait_event(resources.fork_event)

    def launch_complement() -> torch.Tensor:
        dense_x = (
            cusparselt_sparse_residual_indexed_gather(x, dense_indices)
            if optimized_gather
            else x.index_select(0, dense_indices)
        )
        if splitk2_complement:
            return cusparselt_sparse_residual_residual_linear_splitk2(
                dense_x, runtime, variant=splitk2_variant
            )
        return cusparselt_sparse_residual_residual_linear(
            dense_x, runtime, variant=complement_variant
        )

    # CUDA can admit most or all of the all-row base grid before it observes
    # the short complement grid.  Submitting the complement first lets its
    # dense-row CTAs claim a bounded slice of the SMs; cuSPARSELt then fills
    # the remaining SMs.  The streams remain independent and no dependency is
    # introduced between the two GEMMs.
    if complement_first:
        with torch.cuda.stream(resources.dense_stream):
            correction = launch_complement()
        resources.dense_done_event.record(resources.dense_stream)
        with torch.cuda.stream(resources.sparse_stream):
            base = cusparselt_sparse_residual_sparse_linear(x, runtime)
        resources.sparse_done_event.record(resources.sparse_stream)
    else:
        with torch.cuda.stream(resources.sparse_stream):
            base = cusparselt_sparse_residual_sparse_linear(x, runtime)
        resources.sparse_done_event.record(resources.sparse_stream)
        with torch.cuda.stream(resources.dense_stream):
            correction = launch_complement()
        resources.dense_done_event.record(resources.dense_stream)
    origin.wait_event(resources.sparse_done_event)
    origin.wait_event(resources.dense_done_event)
    if splitk2_complement:
        cusparselt_sparse_residual_splitk2_indexed_add_(
            base, correction, dense_indices
        )
    elif optimized_merge:
        cusparselt_sparse_residual_indexed_add_(
            base, correction, dense_indices
        )
    else:
        corrected = base.index_select(0, dense_indices).add_(correction)
        base.index_copy_(0, dense_indices, corrected)
    return base


def launch_fused(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    runtime: Any,
    output: torch.Tensor,
    resources: MultiStreamResources,
    *,
    variant: str = "auto",
    optimized_routes: bool = True,
) -> torch.Tensor:
    """Sparse-row base || dense-row fused base+complement HMMA.SP."""

    if variant == "auto":
        variant = select_fused_gateup_variant(runtime)

    return launch_two_branch_concurrent(
        lambda: branch_to_output(
            lambda: cusparselt_sparse_residual_fused_dense_linear(
                (
                    cusparselt_sparse_residual_indexed_gather(x, dense_indices)
                    if optimized_routes
                    else x.index_select(0, dense_indices)
                ),
                runtime,
                variant=variant,
            ),
            output,
            dense_indices,
            optimized_copy=optimized_routes,
        ),
        lambda: branch_to_output(
            lambda: cusparselt_sparse_residual_sparse_linear(
                (
                    cusparselt_sparse_residual_indexed_gather(x, sparse_indices)
                    if optimized_routes
                    else x.index_select(0, sparse_indices)
                ),
                runtime,
            ),
            output,
            sparse_indices,
            optimized_copy=optimized_routes,
        ),
        output,
        resources,
        device=x.device,
    )


__all__ = [
    "branch_to_output",
    "launch_fused",
    "launch_separate",
    "select_fused_gateup_variant",
    "should_use_fused_gateup",
    "select_wave_aware_complement_splitk2",
    "select_wave_aware_complement_variant",
]
