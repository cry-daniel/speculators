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
    cusparselt_sparse_residual_sparse_linear,
)


def branch_to_output(
    branch: Callable[[], torch.Tensor],
    output: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    output.index_copy_(0, indices, branch())
    return output


def launch_separate(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    runtime: Any,
    resources: MultiStreamResources,
) -> torch.Tensor:
    """All-row prepared base || dense-row complement, then add/scatter."""

    origin = torch.cuda.current_stream(x.device)
    resources.fork_event.record(origin)
    resources.dense_stream.wait_event(resources.fork_event)
    resources.sparse_stream.wait_event(resources.fork_event)
    with torch.cuda.stream(resources.sparse_stream):
        base = cusparselt_sparse_residual_sparse_linear(x, runtime)
    resources.sparse_done_event.record(resources.sparse_stream)
    with torch.cuda.stream(resources.dense_stream):
        dense_x = x.index_select(0, dense_indices)
        correction = cusparselt_sparse_residual_residual_linear(dense_x, runtime)
    resources.dense_done_event.record(resources.dense_stream)
    origin.wait_event(resources.sparse_done_event)
    origin.wait_event(resources.dense_done_event)
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
    variant: str,
) -> torch.Tensor:
    """Sparse-row base || dense-row fused base+complement HMMA.SP."""

    return launch_two_branch_concurrent(
        lambda: branch_to_output(
            lambda: cusparselt_sparse_residual_fused_dense_linear(
                x.index_select(0, dense_indices), runtime, variant=variant
            ),
            output,
            dense_indices,
        ),
        lambda: branch_to_output(
            lambda: cusparselt_sparse_residual_sparse_linear(
                x.index_select(0, sparse_indices), runtime
            ),
            output,
            sparse_indices,
        ),
        output,
        resources,
        device=x.device,
    )


__all__ = ["branch_to_output", "launch_fused", "launch_separate"]
