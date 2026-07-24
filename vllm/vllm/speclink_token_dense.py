# SPDX-License-Identifier: Apache-2.0
"""Token-level dense/sparse routing for SpecLink structured 2:4 studies.

The normal structured-2:4 path edits target-model weights in-place at load
time. This module implements the experimental alternative used for the
accuracy-debug pass: keep the original dense weights, attach 2:4 masks to target
linear modules, and at verification time run high-confidence draft-token rows
dense while low-confidence rows use the masked activation-aware weight.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_MASK_BITS = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)

_propose_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "speclink_token_dense_propose_context", default=None
)
_verify_route: ContextVar[VerifyTokenRoute | torch.Tensor | None] = ContextVar(
    "speclink_token_dense_verify_route", default=None
)

_pending_scores: defaultdict[str, deque[torch.Tensor]] = defaultdict(deque)
_lock = threading.Lock()
_stats_accum: dict[str, Any] = {
    "steps": 0,
    "total_scheduled_tokens": 0,
    "total_draft_tokens": 0,
    "dense_draft_tokens": 0,
    "sparse_draft_tokens": 0,
    "missing_score_tokens": 0,
    "last_flush_steps": 0,
}
_STATIC_ENABLED = os.getenv("SPECLINK_TOKEN_DENSE_ENABLE", "0") in _TRUTHY
_STATIC_PREFILL_FUSED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PREFILL_FUSED", "1") in _TRUTHY
)
_STATIC_GATEUP_FUSED = (
    os.getenv("SPECLINK_TOKEN_DENSE_GATEUP_FUSED", "1") in _TRUTHY
)
# Formal HBM-cold M=512, D=128 winners.  Keep this deliberately narrow: the
# same token-partitioned fused topology loses at D=64, where gather/scatter is
# not amortized, and the other projections have not shown a fused win.
_FUSED_GATEUP_VARIANT_IDS = {
    (512, 128, 24576, 4096): 2,  # Qwen3-8B gate_up: n128_s3
    (512, 128, 28672, 4096): 4,  # Llama3.1-8B gate_up: f64_n128_s4
    (512, 128, 34816, 5120): 4,
    (512, 128, 51200, 5120): 2,
    (512, 128, 57344, 8192): 2,
}
# Mirrored from begin/end_verify_context immediately outside model forward.
# Dynamo guards this ordinary Python boolean and builds separate full-dense
# prefill and routed-verifier graphs; it never has to trace ContextVar.get().
_compile_full_dense_call = False


@dataclass(slots=True)
class VerifyTokenRoute:
    """Prepared verifier routing reused by every target-model linear.

    ``dense_indices`` and ``sparse_indices`` are constructed once per verifier
    step. Reusing them avoids repeating selector/top-k/nonzero work in every
    qkv/o/gate_up/down module.
    """

    dense_mask: torch.Tensor
    dense_indices: torch.Tensor
    sparse_indices: torch.Tensor
    routing_scope: str
    fraction_eighths: int
    total_draft_tokens: int
    dense_draft_tokens: int


@dataclass(slots=True)
class _ResidualStreams:
    dense: torch.cuda.Stream
    sparse: torch.cuda.Stream
    fork: torch.cuda.Event
    dense_done: torch.cuda.Event
    sparse_done: torch.cuda.Event


_residual_streams: dict[tuple[str, int | None], _ResidualStreams] = {}
_graph_route_workspaces: dict[tuple[str, int | None, int], torch.Tensor] = {}
_graph_sparse_route_workspaces: dict[
    tuple[str, int | None, int], torch.Tensor
] = {}
_confidence_reduction_workspaces: dict[
    tuple[str, int | None, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _fused_gateup_variant_id(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    packed: torch.Tensor,
) -> int | None:
    """Return the measured fused token-partition variant, if applicable."""

    if not _STATIC_GATEUP_FUSED:
        return None
    return _FUSED_GATEUP_VARIANT_IDS.get(
        (
            int(x.shape[0]),
            int(dense_indices.numel()),
            int(packed.shape[0]),
            int(x.shape[1]),
        )
    )


def _splitk2_variant_id_for_shape(
    *, output_features: int, dense_rows: int, fallback: int
) -> int:
    """Mirror the measured Split-K2 selector using the active graph shape."""

    if os.getenv("SPECLINK_TOKEN_DENSE_SPLITK2_VARIANT", "auto").strip() != "auto":
        return fallback
    if dense_rows < 256:
        return 3 if output_features == 5120 else 0
    return {
        4096: 4,
        6144: 4,
        5120: 3,
        24576: 5,
        28672: 6,
    }.get(output_features, 0)


@triton.jit
def _greedy_token_logprob_kernel(
    token_ids_ptr,
    selected_logprobs_ptr,
    logits_ptr,
    logits_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """One-pass stable argmax and selected-token log-softmax."""

    row = tl.program_id(0)
    row_ptr = logits_ptr + row * logits_stride
    running_max = float("-inf")
    running_sum = 0.0
    running_id = 0
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            row_ptr + offsets,
            mask=offsets < vocab_size,
            other=float("-inf"),
        ).to(tl.float32)
        block_max = tl.max(values, axis=0)
        block_id = start + tl.argmax(values, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(values - new_max), axis=0
        )
        replace = (block_max > running_max) | (
            (block_max == running_max) & (block_id < running_id)
        )
        running_id = tl.where(replace, block_id, running_id)
        running_max = new_max
    tl.store(token_ids_ptr + row, running_id)
    # The chosen token is the global maximum, so log(p) = -log(sum(exp(x-max))).
    tl.store(selected_logprobs_ptr + row, -tl.log(running_sum))


@triton.jit
def _greedy_token_logprob_partials_kernel(
    logits_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_id_ptr,
    logits_stride,
    vocab_size: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Parallel first-stage max/logsumexp reduction over vocabulary tiles."""

    row = tl.program_id(0)
    split = tl.program_id(1)
    offsets = split * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=offsets < vocab_size,
        other=float("-inf"),
    ).to(tl.float32)
    block_max = tl.max(values, axis=0)
    block_argmax = tl.argmax(values, axis=0)
    slot = row * NUM_SPLITS + split
    tl.store(partial_max_ptr + slot, block_max)
    tl.store(partial_sum_ptr + slot, tl.sum(tl.exp(values - block_max), axis=0))
    tl.store(partial_id_ptr + slot, split * BLOCK_SIZE + block_argmax)


@triton.jit
def _greedy_token_logprob_finalize_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_id_ptr,
    token_ids_ptr,
    selected_logprobs_ptr,
    NUM_SPLITS: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    """Combine vocabulary-tile reductions without materializing log-softmax."""

    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_SIZE)
    mask = offsets < NUM_SPLITS
    base = row * NUM_SPLITS
    maxima = tl.load(
        partial_max_ptr + base + offsets,
        mask=mask,
        other=float("-inf"),
    )
    global_max = tl.max(maxima, axis=0)
    max_split = tl.argmax(maxima, axis=0)
    partial_sums = tl.load(
        partial_sum_ptr + base + offsets,
        mask=mask,
        other=0.0,
    )
    total_sum = tl.sum(partial_sums * tl.exp(maxima - global_max), axis=0)
    token_id = tl.load(partial_id_ptr + base + max_split)
    tl.store(token_ids_ptr + row, token_id)
    tl.store(selected_logprobs_ptr + row, -tl.log(total_sum))


def _confidence_reduction_workspace(
    logits: torch.Tensor,
    *,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (logits.device.type, logits.device.index, int(logits.shape[0]), splits)
    workspace = _confidence_reduction_workspaces.get(key)
    if workspace is None:
        shape = (int(logits.shape[0]), splits)
        workspace = (
            torch.empty(shape, device=logits.device, dtype=torch.float32),
            torch.empty(shape, device=logits.device, dtype=torch.float32),
            torch.empty(shape, device=logits.device, dtype=torch.int32),
        )
        _confidence_reduction_workspaces[key] = workspace
    return workspace


def greedy_sample_with_logprob(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy-sample ids and their log-probabilities without a second scan."""

    if logits.ndim != 2 or not logits.is_cuda:
        raise ValueError("logits must be a CUDA [rows,vocab] tensor")
    rows, vocab_size = logits.shape
    token_ids = torch.empty(rows, device=logits.device, dtype=torch.int64)
    selected_logprobs = torch.empty(
        rows, device=logits.device, dtype=torch.float32
    )
    # A single CTA per row serially scans ~150 vocabulary tiles and remains
    # inefficient across the serving B64--B256 range. Use a two-stage exact
    # reduction there: many CTAs stream disjoint vocabulary tiles, followed by
    # one tiny per-row partial reduction. The selected ID and log-probability
    # are mathematically identical to the one-CTA path.
    if rows <= 256 and vocab_size >= 65536:
        block_size = 4096
        splits = triton.cdiv(vocab_size, block_size)
        partial_max, partial_sum, partial_id = _confidence_reduction_workspace(
            logits, splits=splits
        )
        _greedy_token_logprob_partials_kernel[(rows, splits)](
            logits,
            partial_max,
            partial_sum,
            partial_id,
            logits.stride(0),
            vocab_size,
            NUM_SPLITS=splits,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        _greedy_token_logprob_finalize_kernel[(rows,)](
            partial_max,
            partial_sum,
            partial_id,
            token_ids,
            selected_logprobs,
            NUM_SPLITS=splits,
            REDUCE_SIZE=triton.next_power_of_2(splits),
            num_warps=1,
        )
    else:
        _greedy_token_logprob_kernel[(rows,)](
            token_ids,
            selected_logprobs,
            logits,
            logits.stride(0),
            vocab_size,
            BLOCK_SIZE=1024,
        )
    return token_ids, selected_logprobs


@torch.library.custom_op(
    "speclink::residual_complement_splitk2",
    mutates_args=(),
)
def _compiled_residual_complement_splitk2(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    packed: torch.Tensor,
    residual: torch.Tensor,
    algorithm_id: int,
    variant_id: int,
) -> torch.Tensor:
    """Opaque compiled boundary for the concurrent residual dataflow.

    Dynamo/Inductor sees one shape-defined operator rather than Python
    ContextVars, lazy extension loading, streams, and events.  Full CUDA Graph
    capture still records every kernel and dependency launched below.
    """

    from speculators.speclink.sparse_residual_cutlass_sparse24 import _extension

    extension = _extension()
    if _compile_full_dense_call:
        # The AOT graph is shared with exact verifier buckets, but this Python
        # custom-op implementation runs only for non-captured execution.  A
        # pure prefill therefore reuses the compiled surrounding model while
        # replacing the routed branch with one exact-dense dual-HMMA.SP launch.
        if prefill_fused_enabled():
            fused_variant = 2 if int(x.shape[0]) >= 128 else 0
            return extension.cusparselt_fused_base_complement_forward(
                x, packed, residual, fused_variant
            )
        dense_indices = torch.arange(
            x.shape[0], device=x.device, dtype=torch.int32
        )
    if dense_indices.numel() == 0:
        # Dynamo can instantiate an exact graph bucket whose verifier route has
        # no dense rows (D1, plus zero-sized compile/profile placeholders).  Do
        # not launch the indexed complement kernel: its public CUDA contract
        # intentionally requires at least one index.  Keeping this fast path
        # inside the opaque op also makes the captured graph genuinely base-only.
        padded_rows = (int(x.shape[0]) + 15) // 16 * 16
        dense_b = x.t()
        if padded_rows != int(x.shape[0]):
            dense_b = F.pad(dense_b, (0, padded_rows - int(x.shape[0])))
        base = torch._cslt_sparse_mm(
            packed,
            dense_b,
            transpose_result=True,
            alg_id=algorithm_id,
        )
        return base[: x.shape[0]]

    fused_variant = _fused_gateup_variant_id(x, dense_indices, packed)
    if fused_variant is not None:
        if sparse_indices.numel() == 0:
            raise RuntimeError("fused token partition requires non-empty sparse rows")
        # D=128 gate_up: sparse rows use the cuSPARSELt base directly, while
        # dense rows use the one-kernel base+complement HMMA.SP path.  Both
        # compact row groups write into one output and execute concurrently.
        streams = _get_residual_streams(x.device)
        origin = torch.cuda.current_stream(x.device)
        output = torch.empty(
            (x.shape[0], packed.shape[0]), device=x.device, dtype=x.dtype
        )
        streams.fork.record(origin)
        streams.dense.wait_event(streams.fork)
        streams.sparse.wait_event(streams.fork)
        with torch.cuda.stream(streams.dense):
            dense_x = extension.cusparselt_indexed_gather(x, dense_indices)
            dense_output = extension.cusparselt_fused_base_complement_forward(
                dense_x, packed, residual, fused_variant
            )
            extension.cusparselt_indexed_copy_inplace(
                output, dense_output, dense_indices
            )
        streams.dense_done.record(streams.dense)
        with torch.cuda.stream(streams.sparse):
            sparse_x = extension.cusparselt_indexed_gather(x, sparse_indices)
            padded_rows = (int(sparse_x.shape[0]) + 15) // 16 * 16
            dense_b = sparse_x.t()
            if padded_rows != int(sparse_x.shape[0]):
                dense_b = F.pad(
                    dense_b, (0, padded_rows - int(sparse_x.shape[0]))
                )
            sparse_output = torch._cslt_sparse_mm(
                packed,
                dense_b,
                transpose_result=True,
                alg_id=algorithm_id,
            )[: sparse_indices.numel()]
            extension.cusparselt_indexed_copy_inplace(
                output, sparse_output, sparse_indices
            )
        streams.sparse_done.record(streams.sparse)
        origin.wait_event(streams.dense_done)
        origin.wait_event(streams.sparse_done)
        return output

    variant_id = _splitk2_variant_id_for_shape(
        output_features=int(packed.shape[0]),
        dense_rows=int(dense_indices.numel()),
        fallback=variant_id,
    )
    streams = _get_residual_streams(x.device)
    origin = torch.cuda.current_stream(x.device)
    streams.fork.record(origin)
    streams.dense.wait_event(streams.fork)
    streams.sparse.wait_event(streams.fork)
    with torch.cuda.stream(streams.dense):
        partials = (
            extension.cusparselt_complement_sparse_splitk2_indexed_forward(
                x, dense_indices, packed, residual, variant_id
            )
        )
    streams.dense_done.record(streams.dense)
    with torch.cuda.stream(streams.sparse):
        # cuSPARSELt BF16 requires the dense operand's token dimension to be a
        # multiple of 16. The public semi-structured wrapper pads M<16 too;
        # reproduce that contract here because K=7 Graph buckets include M=8.
        padded_rows = (int(x.shape[0]) + 15) // 16 * 16
        dense_b = x.t()
        if padded_rows != int(x.shape[0]):
            dense_b = F.pad(dense_b, (0, padded_rows - int(x.shape[0])))
        base = torch._cslt_sparse_mm(
            packed,
            dense_b,
            transpose_result=True,
            alg_id=algorithm_id,
        )
        base = base[: x.shape[0]]
    streams.sparse_done.record(streams.sparse)
    origin.wait_event(streams.dense_done)
    origin.wait_event(streams.sparse_done)
    return extension.cusparselt_splitk2_indexed_add_inplace(
        base, partials, dense_indices
    )


@_compiled_residual_complement_splitk2.register_fake
def _compiled_residual_complement_splitk2_fake(
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    packed: torch.Tensor,
    residual: torch.Tensor,
    algorithm_id: int,
    variant_id: int,
) -> torch.Tensor:
    del dense_indices, sparse_indices, residual, algorithm_id, variant_id
    return x.new_empty((x.shape[0], packed.shape[0]))


def enabled() -> bool:
    return _STATIC_ENABLED


def threshold() -> float:
    return float(os.getenv("SPECLINK_TOKEN_DENSE_THRESHOLD", "0.7"))


def mode() -> str:
    return os.getenv("SPECLINK_TOKEN_DENSE_MODE", "high_confidence_dense").strip()


def fraction_eighths() -> int | None:
    """Return the fixed total-token dense quota, or None for legacy threshold."""

    raw = os.getenv("SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS", "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 0 or value > 8:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_FRACTION_EIGHTHS must be an integer in [0,8]"
        )
    return value


def draft_scores_required() -> bool:
    """Whether routing needs normalized DLM confidence scores.

    D1 selects no draft rows, so confidence cannot affect its fixed route and
    the proposer should stay on its ordinary greedy path.
    """

    quota = fraction_eighths()
    return enabled() and (quota is None or quota > 1)


def routing_scope() -> str:
    value = os.getenv("SPECLINK_TOKEN_DENSE_ROUTING_SCOPE", "global").strip()
    if value not in {"global", "per_request"}:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_ROUTING_SCOPE must be global or per_request"
        )
    return value


def score_mode() -> str:
    """Return the DLM score used to rank fixed-quota dense draft rows."""

    value = os.getenv(
        "SPECLINK_TOKEN_DENSE_SCORE_MODE", "prefix_product"
    ).strip()
    if value not in {"prefix_product", "token_probability"}:
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_SCORE_MODE must be prefix_product or "
            "token_probability"
        )
    return value


def stats_path() -> Path | None:
    value = os.getenv("SPECLINK_TOKEN_DENSE_STATS_PATH", "").strip()
    return Path(value) if value else None


def stats_detail_enabled() -> bool:
    return os.getenv("SPECLINK_TOKEN_DENSE_STATS_DETAIL", "0") in _TRUTHY


def stats_interval() -> int:
    try:
        return max(0, int(os.getenv("SPECLINK_TOKEN_DENSE_STATS_INTERVAL", "1000")))
    except ValueError:
        return 1000


def reset_runtime_state() -> None:
    """Clear completed-request confidence queues and aggregate counters."""

    with _lock:
        _pending_scores.clear()
        _stats_accum.update(
            {
                "steps": 0,
                "total_scheduled_tokens": 0,
                "total_draft_tokens": 0,
                "dense_draft_tokens": 0,
                "sparse_draft_tokens": 0,
                "missing_score_tokens": 0,
                "last_flush_steps": 0,
            }
        )


def graph_route_enabled() -> bool:
    return os.getenv("SPECLINK_TOKEN_DENSE_GRAPH_ROUTE", "0") in _TRUTHY


def prefill_fused_enabled() -> bool:
    # A prefill row always needs exact dense semantics.  With the persistent
    # base-2:4 + complement-2:4 representation, the decode-oriented path would
    # launch an all-row cuSPARSELt GEMM, an indexed Split-K=2 complement GEMM,
    # and an indexed reduce/add.  The dual-HMMA.SP kernel shares activation and
    # metadata stages and writes the sum once, which is the appropriate default
    # for this all-dense phase.  Keep an environment opt-out for ablations.
    return _STATIC_PREFILL_FUSED


def _graph_route_workspace(
    eighths: int,
    device: torch.device,
    *,
    min_rows: int = 0,
) -> torch.Tensor:
    configured_rows = [
        int(text.strip())
        for text in os.getenv(
            "SPECLINK_TOKEN_DENSE_GRAPH_ROWS", "512,1024,2048"
        ).split(",")
        if text.strip()
    ]
    capacity_rows = max([min_rows, *configured_rows])
    if capacity_rows <= 0 or capacity_rows % 8:
        raise RuntimeError(
            "graph-routed verification workspace requires a positive M "
            "capacity divisible by 8"
        )
    key = (device.type, device.index, eighths)
    workspace = _graph_route_workspaces.get(key)
    required_count = capacity_rows * eighths // 8
    if workspace is None or int(workspace.numel()) < required_count:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA graph route workspace must be allocated before capture")
        batch = capacity_rows // 8
        current = torch.arange(batch, device=device, dtype=torch.int32) * 8
        if eighths == 1:
            initial = current
        else:
            offsets = torch.arange(
                eighths, device=device, dtype=torch.int32
            )
            initial = (current[:, None] + offsets[None, :]).flatten().sort().values
        if int(initial.numel()) != required_count:
            raise RuntimeError("invalid graph-route placeholder size")
        workspace = initial.contiguous()
        _graph_route_workspaces[key] = workspace
    return workspace


def _graph_route_bucket_rows(rows: int) -> int:
    configured_rows = sorted(
        {
            int(text.strip())
            for text in os.getenv(
                "SPECLINK_TOKEN_DENSE_GRAPH_ROWS", "512,1024,2048"
            ).split(",")
            if text.strip()
        }
    )
    for bucket in configured_rows:
        if bucket >= rows:
            return bucket
    raise RuntimeError(
        f"no CUDA graph route bucket can hold {rows} rows; "
        f"configured={configured_rows}"
    )


def _graph_route_buffer(
    rows: int,
    eighths: int,
    device: torch.device,
) -> torch.Tensor:
    if rows <= 0 or rows % 8:
        raise RuntimeError("graph-routed verification requires M divisible by 8")
    dense_count = rows * eighths // 8
    workspace = _graph_route_workspace(eighths, device, min_rows=rows)
    return workspace[:dense_count]


def _graph_sparse_route_workspace(
    eighths: int,
    device: torch.device,
    *,
    min_rows: int = 0,
) -> torch.Tensor:
    """Persistent sparse-row indices updated outside CUDA Graph replay."""

    configured_rows = [
        int(text.strip())
        for text in os.getenv(
            "SPECLINK_TOKEN_DENSE_GRAPH_ROWS", "512,1024,2048"
        ).split(",")
        if text.strip()
    ]
    capacity_rows = max([min_rows, *configured_rows])
    if capacity_rows <= 0 or capacity_rows % 8:
        raise RuntimeError(
            "graph-routed verification workspace requires a positive M "
            "capacity divisible by 8"
        )
    key = (device.type, device.index, eighths)
    workspace = _graph_sparse_route_workspaces.get(key)
    required_count = capacity_rows * (8 - eighths) // 8
    if workspace is None or int(workspace.numel()) < required_count:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph sparse-route workspace must be allocated before capture"
            )
        batch = capacity_rows // 8
        current = torch.arange(batch, device=device, dtype=torch.int32) * 8
        offsets = torch.arange(
            eighths, 8, device=device, dtype=torch.int32
        )
        initial = (current[:, None] + offsets[None, :]).flatten()
        if int(initial.numel()) != required_count:
            raise RuntimeError("invalid graph sparse-route placeholder size")
        workspace = initial.contiguous()
        _graph_sparse_route_workspaces[key] = workspace
    return workspace


def _graph_sparse_route_buffer(
    rows: int,
    eighths: int,
    device: torch.device,
) -> torch.Tensor:
    if rows <= 0 or rows % 8:
        raise RuntimeError("graph-routed verification requires M divisible by 8")
    sparse_count = rows * (8 - eighths) // 8
    workspace = _graph_sparse_route_workspace(eighths, device, min_rows=rows)
    return workspace[:sparse_count]


def _accumulate_stats(record: dict[str, Any]) -> dict[str, Any] | None:
    with _lock:
        _stats_accum["steps"] += 1
        _stats_accum["total_scheduled_tokens"] += int(
            record.get("total_scheduled_tokens") or 0
        )
        _stats_accum["total_draft_tokens"] += int(record.get("total_draft_tokens") or 0)
        _stats_accum["dense_draft_tokens"] += int(record.get("dense_draft_tokens") or 0)
        _stats_accum["sparse_draft_tokens"] += int(
            record.get("sparse_draft_tokens") or 0
        )
        _stats_accum["missing_score_tokens"] += int(
            record.get("missing_score_tokens") or 0
        )
        interval = stats_interval()
        if interval <= 0:
            return None
        steps = int(_stats_accum["steps"])
        if steps - int(_stats_accum["last_flush_steps"]) < interval:
            return None
        _stats_accum["last_flush_steps"] = steps
        total_draft = int(_stats_accum["total_draft_tokens"])
        dense_draft = int(_stats_accum["dense_draft_tokens"])
        sparse_draft = int(_stats_accum["sparse_draft_tokens"])
        return {
            "timestamp": time.time(),
            "event": "verify_token_mask_summary",
            "mode": record.get("mode"),
            "threshold": record.get("threshold"),
            "steps": steps,
            "total_scheduled_tokens": int(_stats_accum["total_scheduled_tokens"]),
            "total_draft_tokens": total_draft,
            "dense_draft_tokens": dense_draft,
            "sparse_draft_tokens": sparse_draft,
            "missing_score_tokens": int(_stats_accum["missing_score_tokens"]),
            "dense_draft_fraction": dense_draft / total_draft if total_draft else None,
            "sparse_draft_fraction": sparse_draft / total_draft if total_draft else None,
        }


def begin_propose_context(
    *,
    req_ids: list[str],
    prompt_lens: list[int],
    generated_lens: list[int],
    active_requests: int,
    batch_size: int,
    num_spec_tokens: int,
    method: str = "",
) -> Any:
    if not enabled():
        return None
    return _propose_context.set(
        {
            "req_ids": req_ids,
            "prompt_lens": prompt_lens,
            "generated_lens": generated_lens,
            "active_requests": active_requests,
            "batch_size": batch_size,
            "num_spec_tokens": num_spec_tokens,
            "method": method or "unknown",
        }
    )


def end_propose_context(token: Any) -> None:
    if token is not None:
        _propose_context.reset(token)


def begin_verify_context(route: VerifyTokenRoute | torch.Tensor | None) -> Any:
    global _compile_full_dense_call
    previous_full_dense = _compile_full_dense_call
    _compile_full_dense_call = bool(enabled() and route is None)
    route_token = _verify_route.set(route) if enabled() and route is not None else None
    return route_token, previous_full_dense


def end_verify_context(token: Any) -> None:
    global _compile_full_dense_call
    if token is None:
        return
    route_token, previous_full_dense = token
    if route_token is not None:
        _verify_route.reset(route_token)
    _compile_full_dense_call = bool(previous_full_dense)


@torch.inference_mode()
def record_draft_scores(
    *,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    selected_logprobs: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    method: str = "",
) -> None:
    """Record selected-token probabilities for the next verifier step."""
    if not enabled():
        return
    if not draft_scores_required():
        return
    ctx = _propose_context.get()
    if ctx is None or (selected_logprobs is None and not logits_by_position):
        return

    batch_size = min(int(draft_token_ids.shape[0]), len(ctx["req_ids"]))
    available_positions = (
        int(selected_logprobs.shape[1])
        if selected_logprobs is not None
        else len(logits_by_position)
    )
    num_spec_tokens = min(int(draft_token_ids.shape[1]), available_positions)
    if batch_size <= 0 or num_spec_tokens <= 0:
        return

    draft_token_ids = draft_token_ids[:batch_size, :num_spec_tokens]
    if selected_logprobs is None:
        selected_logprob_parts: list[torch.Tensor] = []
        for pos, logits in enumerate(logits_by_position[:num_spec_tokens]):
            logits = logits[:batch_size].detach().float()
            selected = draft_token_ids[:batch_size, pos].to(
                device=logits.device, dtype=torch.long
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_logprob = log_probs.gather(1, selected.view(-1, 1)).squeeze(1)
            selected_logprob_parts.append(selected_logprob)
        selected_logprobs = torch.stack(selected_logprob_parts, dim=1)
    else:
        selected_logprobs = selected_logprobs[
            :batch_size, :num_spec_tokens
        ].detach().float()

    # Both alternatives stay entirely on GPU.  Prefix confidence is
    # accumulated in log space for numerical stability; the ablation ranks by
    # each position's selected-token probability without prefix accumulation.
    if score_mode() == "prefix_product":
        per_req_scores = selected_logprobs.cumsum(dim=1).exp()
    else:
        per_req_scores = selected_logprobs.exp()

    req_ids = ctx["req_ids"]
    with _lock:
        for req_idx in range(batch_size):
            _pending_scores[req_ids[req_idx]].append(
                per_req_scores[req_idx].detach().contiguous()
            )


def _write_stats(record: dict[str, Any]) -> None:
    path = stats_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_verify_dense_mask(
    *,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_draft_tokens: Any,
    cu_num_scheduled_tokens: Any,
    total_num_scheduled_tokens: int,
    device: torch.device,
) -> VerifyTokenRoute | None:
    """Build one quota-controlled route for the target verification pass.

    The first sampled token per request is the already-known next token and the
    following scheduled rows are draft tokens.  D0 is the pure-2:4 verifier
    endpoint: every scheduled verifier row uses only the sparse base.  D1--D8
    keep the current token dense and select exactly ``d-1`` of seven draft rows
    per request, preserving a total verifier M of ``8 * request_count``.
    """
    if not enabled() or total_num_scheduled_tokens <= 0:
        return None
    routing_mode = mode()
    if routing_mode != "high_confidence_dense":
        raise RuntimeError(
            "SPECLINK_TOKEN_DENSE_MODE currently supports only "
            "high_confidence_dense"
        )

    cutoff = threshold()
    quota = fraction_eighths()
    dense_mask = torch.full(
        (total_num_scheduled_tokens,),
        quota != 0,
        dtype=torch.bool,
        device=device,
    )
    scores_required = quota is None or quota > 1
    scope = routing_scope()
    missing_score_tokens = 0
    total_draft_tokens = 0
    stats_enabled = stats_path() is not None
    include_request_summaries = stats_enabled and stats_detail_enabled()
    request_summaries: list[dict[str, Any]] = []
    # Entries retain every draft row.  ``scores`` may be shorter when the DLM
    # result for an asynchronously admitted request has not arrived yet.
    candidates: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    with _lock:
        for req_idx, req_id in enumerate(req_ids):
            n = int(num_draft_tokens[req_idx])
            if n <= 0:
                continue
            total_draft_tokens += n
            end = int(cu_num_scheduled_tokens[req_idx])
            sched = int(num_scheduled_tokens[req_idx])
            start = end - sched
            pending = _pending_scores.get(req_id) if scores_required else None
            scores = pending.popleft() if pending else None
            rows = torch.arange(
                start + 1,
                min(start + 1 + n, total_num_scheduled_tokens),
                dtype=torch.int64,
                device=device,
            )
            if rows.numel() != n:
                missing_score_tokens += n - int(rows.numel())
                n = int(rows.numel())
            if n <= 0:
                continue
            dense_mask[rows] = False
            available = 0 if scores is None else min(n, int(scores.numel()))
            known_scores = (
                scores[:available].to(device=device, dtype=torch.float32)
                if available
                else torch.empty(0, device=device, dtype=torch.float32)
            )
            candidates.append((req_id, rows, known_scores))
            if scores_required and available < n:
                # Threshold routing remains fail-safe dense.  A fixed d/8
                # quota instead ranks missing scores last so asynchronous
                # admission cannot change the graph shape or requested ratio.
                if quota is None:
                    dense_mask[rows[available:]] = True
                missing_score_tokens += n - available

    selected_known_count = 0
    if quota is None:
        for _, rows, scores in candidates:
            dense_mask[rows] = scores >= cutoff
    elif scope == "global":
        if candidates:
            all_rows = torch.cat([entry[1] for entry in candidates])
            score_parts = []
            for _, rows, known_scores in candidates:
                if known_scores.numel() == rows.numel():
                    score_parts.append(known_scores)
                else:
                    padded = torch.full(
                        (rows.numel(),),
                        -torch.inf,
                        device=device,
                        dtype=torch.float32,
                    )
                    padded[: known_scores.numel()] = known_scores
                    score_parts.append(padded)
            all_scores = torch.cat(score_parts)
            # Generalize gracefully when a request has fewer than seven draft
            # tokens; the formal K=7 case remains exact.
            target = round(total_draft_tokens * (quota - 1) / 7)
            select_count = min(int(all_rows.numel()), max(0, target))
            if select_count:
                order = torch.argsort(
                    all_scores, descending=True, stable=True
                )[:select_count]
                dense_mask[all_rows.index_select(0, order)] = True
            selected_known_count = select_count
    else:
        for req_id, rows, known_scores in candidates:
            n = int(rows.numel())
            target = round(n * (quota - 1) / 7)
            select_count = min(n, max(0, target))
            if select_count:
                scores = torch.full(
                    (n,), -torch.inf, device=device, dtype=torch.float32
                )
                scores[: known_scores.numel()] = known_scores
                order = torch.argsort(
                    scores, descending=True, stable=True
                )[:select_count]
                dense_mask[rows.index_select(0, order)] = True
            selected_known_count += select_count

    candidate_rows = (
        torch.cat([entry[1] for entry in candidates])
        if candidates
        else torch.empty(0, device=device, dtype=torch.int64)
    )
    known_dense = (
        int(dense_mask[candidate_rows].sum().item())
        if quota is None and candidates
        else selected_known_count
    )
    dense_draft_tokens = (
        known_dense + missing_score_tokens if quota is None else known_dense
    )
    sparse_draft_tokens = max(0, total_draft_tokens - dense_draft_tokens)
    if include_request_summaries:
        for req_id, rows, scores in candidates:
            per_req_dense = int(dense_mask[rows].sum().item())
            per_req_sparse = int(rows.numel()) - per_req_dense
            request_summaries.append(
                {
                    "request_id": req_id,
                    "draft_tokens": int(rows.numel()),
                    "dense_draft_tokens": per_req_dense,
                    "sparse_draft_tokens": per_req_sparse,
                    "score_count": int(scores.numel()),
                }
            )

    # The serving fast path can omit stats entirely. This bypasses not only
    # file I/O but also per-step dictionaries and the aggregate mutex.
    if stats_enabled:
        record = {
            "timestamp": time.time(),
            "event": "verify_token_mask",
            "mode": routing_mode,
            "threshold": cutoff,
            "fraction_eighths": quota,
            "routing_scope": scope,
            "request_count": len(req_ids),
            "total_scheduled_tokens": total_num_scheduled_tokens,
            "total_draft_tokens": total_draft_tokens,
            "dense_draft_tokens": dense_draft_tokens,
            "sparse_draft_tokens": sparse_draft_tokens,
            "missing_score_tokens": missing_score_tokens,
            "dense_draft_fraction": (
                dense_draft_tokens / total_draft_tokens
                if total_draft_tokens
                else None
            ),
            "sparse_draft_fraction": (
                sparse_draft_tokens / total_draft_tokens
                if total_draft_tokens
                else None
            ),
        }
        if include_request_summaries:
            record["requests"] = request_summaries
            _write_stats(record)
        else:
            summary_record = _accumulate_stats(record)
            if summary_record is not None:
                _write_stats(summary_record)
    dense_indices = dense_mask.nonzero(as_tuple=False).flatten().to(torch.int32)
    sparse_indices = (~dense_mask).nonzero(as_tuple=False).flatten().to(torch.int32)
    pure_k7_verify = total_draft_tokens * 8 == total_num_scheduled_tokens * 7
    if graph_route_enabled() and quota is not None and pure_k7_verify:
        # CUDA Graph pads a shrinking active batch to its next captured bucket.
        # Fused token partitioning needs a total, disjoint partition of that
        # whole bucket; stale tail indices from a previous larger step can be
        # out of range for a smaller graph and are not safe for indexed copy.
        graph_rows = _graph_route_bucket_rows(total_num_scheduled_tokens)
        persistent_dense = _graph_route_buffer(
            graph_rows, quota, device
        )
        persistent_sparse = _graph_sparse_route_buffer(
            graph_rows, quota, device
        )
        padding_rows = graph_rows - total_num_scheduled_tokens
        dense_padding_rows = padding_rows * quota // 8
        padding = torch.arange(
            total_num_scheduled_tokens,
            graph_rows,
            device=device,
            dtype=torch.int32,
        )
        full_dense_indices = torch.cat(
            (dense_indices, padding[:dense_padding_rows])
        )
        full_sparse_indices = torch.cat(
            (sparse_indices, padding[dense_padding_rows:])
        )
        if persistent_dense.numel() != full_dense_indices.numel():
            raise RuntimeError(
                "quota route count changed; CUDA graph requires fixed d/8 rows"
            )
        if persistent_sparse.numel() != full_sparse_indices.numel():
            raise RuntimeError(
                "quota sparse-route count changed; CUDA graph requires fixed rows"
            )
        persistent_dense.copy_(full_dense_indices)
        persistent_sparse.copy_(full_sparse_indices)
        dense_indices = persistent_dense
        sparse_indices = persistent_sparse
    return VerifyTokenRoute(
        dense_mask=dense_mask.contiguous(),
        dense_indices=dense_indices.contiguous(),
        sparse_indices=sparse_indices.contiguous(),
        routing_scope=scope,
        fraction_eighths=quota or 0,
        total_draft_tokens=total_draft_tokens,
        dense_draft_tokens=dense_draft_tokens,
    )


def _expand_mask_bytes(
    mask_bytes: torch.Tensor,
    *,
    out_features: int,
    groups: int,
    device: torch.device,
) -> torch.Tensor:
    if tuple(mask_bytes.shape) == (out_features, groups):
        return mask_bytes.to(device=device, dtype=torch.uint8, non_blocking=True)

    packed_expected = (out_features, (groups + 1) // 2)
    if tuple(mask_bytes.shape) != packed_expected:
        raise RuntimeError(
            f"SpecLink token-dense mask shape {tuple(mask_bytes.shape)} does "
            f"not match {(out_features, groups)} or {packed_expected}"
        )
    packed = mask_bytes.to(device=device, dtype=torch.uint8, non_blocking=True)
    unpacked = torch.empty(
        (out_features, packed_expected[1] * 2),
        dtype=torch.uint8,
        device=device,
    )
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    return unpacked[:, :groups]


def _dense_mask_from_context(
    rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    route = _verify_route.get()
    if route is None:
        return None
    if isinstance(route, VerifyTokenRoute):
        dense_mask = route.dense_mask
        dense_indices = route.dense_indices
        sparse_indices = route.sparse_indices
    else:
        dense_mask = route
        dense_indices = dense_mask.nonzero(as_tuple=False).flatten().to(torch.int32)
        sparse_indices = (~dense_mask).nonzero(as_tuple=False).flatten().to(
            torch.int32
        )
    if dense_mask.numel() < rows:
        raise RuntimeError(
            f"SpecLink token route has {dense_mask.numel()} rows, but linear input "
            f"has {rows}; run this backend in eager/unpadded verification mode"
        )
    if dense_mask.device != device:
        dense_mask = dense_mask.to(device=device, non_blocking=True)
        dense_indices = dense_indices.to(device=device, non_blocking=True)
        sparse_indices = sparse_indices.to(device=device, non_blocking=True)
    if dense_mask.numel() == rows:
        return (
            dense_mask,
            dense_indices[dense_indices < rows].contiguous(),
            sparse_indices[sparse_indices < rows].contiguous(),
        )
    return (
        dense_mask[:rows],
        dense_indices[dense_indices < rows].contiguous(),
        sparse_indices[sparse_indices < rows].contiguous(),
    )


def _get_residual_streams(device: torch.device) -> _ResidualStreams:
    key = (device.type, device.index)
    streams = _residual_streams.get(key)
    if streams is None:
        streams = _ResidualStreams(
            dense=torch.cuda.Stream(device=device),
            sparse=torch.cuda.Stream(device=device),
            fork=torch.cuda.Event(enable_timing=False, external=False),
            dense_done=torch.cuda.Event(enable_timing=False, external=False),
            sparse_done=torch.cuda.Event(enable_timing=False, external=False),
        )
        _residual_streams[key] = streams
    return streams


def _splitk2_variant(runtime: Any, dense_rows: int | None = None) -> str:
    override = os.getenv(
        "SPECLINK_TOKEN_DENSE_SPLITK2_VARIANT", "auto"
    ).strip()
    if override != "auto":
        return override
    if dense_rows is None:
        expected_rows = int(
            os.getenv("SPECLINK_TOKEN_DENSE_EXPECTED_ROWS", "0") or "0"
        )
        quota = fraction_eighths()
        dense_rows = expected_rows * quota // 8 if quota is not None else 0
    if dense_rows < 256:
        return (
            "b_resident_feature64_token64_b2a1_p40"
            if int(runtime.n) == 5120
            else "b_resident_feature64_token64_b2a1"
        )
    return {
        4096: "b_resident_feature128_token64_b2a1",
        6144: "b_resident_feature128_token64_b2a1",
        5120: "b_resident_feature64_token64_b2a1_p40",
        24576: "b_resident_feature128_token64_b2a1_p192",
        28672: "b_resident_feature128_token64_b2a1_p224",
    }.get(int(runtime.n), "b_resident_feature64_token64_b2a1")


@torch.inference_mode()
def residual_complement_linear(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Run the prepared one-weight base+Split-K2 complement linear.

    Returns ``None`` for ordinary modules so callers can retain their normal
    vLLM linear path.  Prepared modules never read a dense weight: all rows use
    the cuSPARSELt base and selected dense rows receive the complementary 2:4
    correction on a concurrent stream.
    """

    runtime = getattr(module, "_speclink_residual_complement_runtime", None)
    if runtime is None:
        return None
    if input_tensor.ndim != 2 or not input_tensor.is_cuda:
        raise RuntimeError("SpecLink residual-complement expects CUDA [M,K] input")
    if int(getattr(module, "tp_size", 1)) != 1:
        raise RuntimeError("SpecLink residual-complement currently requires TP=1")
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        raise RuntimeError("SpecLink residual-complement does not support linear bias")

    from speculators.speclink import (
        cusparselt_sparse_residual_fused_dense_linear,
        cusparselt_sparse_residual_residual_linear_splitk2_indexed,
        cusparselt_sparse_residual_sparse_linear,
        cusparselt_sparse_residual_splitk2_indexed_add_,
    )

    x = input_tensor.contiguous()
    graph_workspace = getattr(module, "_speclink_graph_dense_indices", None)
    graph_sparse_workspace = getattr(
        module, "_speclink_graph_sparse_indices", None
    )
    graph_eighths = int(getattr(module, "_speclink_graph_eighths", 0))
    if (
        torch.compiler.is_compiling()
        and isinstance(graph_workspace, torch.Tensor)
        and isinstance(graph_sparse_workspace, torch.Tensor)
    ):
        # ContextVar remains outside the AOT graph.  Every exact verifier graph
        # captures a prefix of this workspace; the opaque op itself switches to
        # full-dense fused execution for compiled-but-not-captured prefill.
        dense_indices = graph_workspace[: x.shape[0] * graph_eighths // 8]
        sparse_indices = graph_sparse_workspace[
            : x.shape[0] * (8 - graph_eighths) // 8
        ]
        return torch.ops.speclink.residual_complement_splitk2.default(
            x,
            dense_indices,
            sparse_indices,
            module._speclink_residual_packed,
            module._speclink_residual_values,
            module._speclink_residual_complement_algorithm_id,
            module._speclink_residual_splitk2_variant_id,
        )
    else:
        route = _dense_mask_from_context(int(x.shape[0]), x.device)
        if (
            route is None
            and isinstance(graph_workspace, torch.Tensor)
            and isinstance(graph_sparse_workspace, torch.Tensor)
            and torch.cuda.is_current_stream_capturing()
        ):
            dense_indices = graph_workspace[
                : int(x.shape[0]) * graph_eighths // 8
            ]
            sparse_indices = graph_sparse_workspace[
                : int(x.shape[0]) * (8 - graph_eighths) // 8
            ]
        elif route is None:
            # Prefill and non-speculative calls have no confidence route. They
            # are exact dense by construction.  Do not construct an all-row
            # index tensor for the fused path: besides being unnecessary, that
            # allocation used to repeat in every qkv/o/gate_up/down module.
            if prefill_fused_enabled():
                variant = "n128_s3" if int(x.shape[0]) >= 128 else "n64_s3"
                return cusparselt_sparse_residual_fused_dense_linear(
                    x, runtime, variant=variant
                )
            dense_indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int32)
            sparse_indices = torch.empty(
                0, device=x.device, dtype=torch.int32
            )
        else:
            _, dense_indices, sparse_indices = route

    if dense_indices.numel() == 0:
        # D0 is a real base-only endpoint.  Some decode calls execute outside
        # the compiled/captured graph (for example graph-bucket transitions),
        # so mirror the opaque op's zero-dense fast path here rather than
        # invoking an indexed complement kernel with an empty index set.
        return cusparselt_sparse_residual_sparse_linear(x, runtime)

    if _fused_gateup_variant_id(
        x, dense_indices, module._speclink_residual_packed
    ) is not None:
        return torch.ops.speclink.residual_complement_splitk2.default(
            x,
            dense_indices,
            sparse_indices,
            module._speclink_residual_packed,
            module._speclink_residual_values,
            module._speclink_residual_complement_algorithm_id,
            module._speclink_residual_splitk2_variant_id,
        )

    streams = _get_residual_streams(x.device)
    origin = torch.cuda.current_stream(x.device)
    streams.fork.record(origin)
    streams.dense.wait_event(streams.fork)
    streams.sparse.wait_event(streams.fork)
    with torch.cuda.stream(streams.dense):
        partials = cusparselt_sparse_residual_residual_linear_splitk2_indexed(
            x,
            dense_indices,
            runtime,
            variant=_splitk2_variant(runtime, int(dense_indices.numel())),
        )
    streams.dense_done.record(streams.dense)
    with torch.cuda.stream(streams.sparse):
        base = cusparselt_sparse_residual_sparse_linear(x, runtime)
    streams.sparse_done.record(streams.sparse)
    origin.wait_event(streams.dense_done)
    origin.wait_event(streams.sparse_done)
    return cusparselt_sparse_residual_splitk2_indexed_add_(
        base, partials, dense_indices
    )


def _sparse_weight_from_attached_mask(
    weight: torch.Tensor,
    mask_bytes: torch.Tensor,
    *,
    chunk_rows: int = 256,
) -> torch.Tensor:
    """Materialize the setup-only 2:4 base without a full keep-mask tensor."""

    n, k = map(int, weight.shape)
    groups = k // 4
    packed_expected = (n, (groups + 1) // 2)
    if tuple(mask_bytes.shape) not in {(n, groups), packed_expected}:
        raise RuntimeError(
            f"attached mask {tuple(mask_bytes.shape)} does not match weight {(n, k)}"
        )
    sparse = torch.zeros_like(weight)
    bits = _MASK_BITS.to(device=weight.device)
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        chunk = mask_bytes[start:stop].to(
            device=weight.device, dtype=torch.uint8, non_blocking=True
        )
        if tuple(mask_bytes.shape) == packed_expected:
            unpacked = torch.empty(
                (stop - start, packed_expected[1] * 2),
                device=weight.device,
                dtype=torch.uint8,
            )
            unpacked[:, 0::2] = chunk & 0x0F
            unpacked[:, 1::2] = (chunk >> 4) & 0x0F
            chunk = unpacked[:, :groups]
        keep = (chunk.unsqueeze(-1) & bits.view(1, 1, 4)).ne(0)
        sparse[start:stop].view(stop - start, groups, 4).copy_(
            weight[start:stop].view(stop - start, groups, 4)
            * keep.to(dtype=weight.dtype)
        )
    return sparse.contiguous()


@torch.inference_mode()
def prepare_residual_complement_module(module: Any) -> dict[str, Any]:
    """Convert one loaded BF16 vLLM linear to the persistent one-weight form."""

    weight = getattr(module, "weight", None)
    mask_bytes = getattr(module, "_speclink_24_mask_bytes", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16:
        raise RuntimeError("residual-complement preparation requires BF16 weight")
    if not weight.is_cuda or weight.ndim != 2 or not weight.is_contiguous():
        raise RuntimeError("residual-complement weight must be contiguous CUDA [N,K]")
    if mask_bytes is None:
        raise RuntimeError("residual-complement preparation requires an attached 2:4 mask")
    if getattr(module, "_speclink_24_row_scale", None) is not None:
        # A row-rescaled base is not complementary to the original dense
        # weight.  The one-weight exact decomposition uses mask positions but
        # always retains the checkpoint values themselves.
        module._speclink_24_row_scale = None

    from speculators.speclink import (
        SPARSE_RESIDUAL_SMEM,
        prepare_cusparselt_sparse_residual_weight,
        prepare_online_sparse24_weight,
        tune_cusparselt_sparse_residual_algorithm,
    )

    sparse_weight = _sparse_weight_from_attached_mask(weight, mask_bytes)
    canonical = prepare_online_sparse24_weight(
        weight, sparse_weight, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical, sparse_weight=sparse_weight
    )
    tune_rows = int(os.getenv("SPECLINK_TOKEN_DENSE_EXPECTED_ROWS", "0") or "0")
    algorithm_id = int(runtime.cusparselt.algorithm_id)
    if tune_rows > 0:
        sample = torch.empty(
            (tune_rows, int(weight.shape[1])),
            device=weight.device,
            dtype=torch.bfloat16,
        )
        sample.normal_(mean=0.0, std=0.02)
        algorithm_id = tune_cusparselt_sparse_residual_algorithm(runtime, sample)
        del sample

    original_shape = tuple(map(int, weight.shape))
    original_bytes = int(weight.untyped_storage().nbytes())
    module._speclink_residual_complement_runtime = runtime
    module._speclink_residual_complement_shape = original_shape
    module._speclink_residual_complement_algorithm_id = algorithm_id
    module.register_buffer(
        "_speclink_residual_packed", runtime.packed, persistent=False
    )
    module.register_buffer(
        "_speclink_residual_values", runtime.residual, persistent=False
    )
    from speculators.speclink.cusparselt_sparse_residual import (
        COMPLEMENT_SPLITK2_VARIANTS,
    )

    module._speclink_residual_splitk2_variant_id = COMPLEMENT_SPLITK2_VARIANTS[
        _splitk2_variant(runtime)
    ]
    if graph_route_enabled():
        quota = fraction_eighths()
        if quota is None:
            raise RuntimeError("graph route requires fixed fraction eighths")
        module._speclink_graph_dense_indices = _graph_route_workspace(
            quota, runtime.device
        )
        module._speclink_graph_sparse_indices = _graph_sparse_route_workspace(
            quota, runtime.device
        )
        module._speclink_graph_eighths = quota
    # Drop duplicate selector storage and the dense parameter.  The runtime now
    # owns exactly packed base values+cuSPARSELt metadata and complement values.
    module._speclink_24_mask_bytes = None
    module._speclink_24_row_scale = None
    module.register_parameter("weight", None)
    del canonical, sparse_weight, weight
    return {
        "shape": list(original_shape),
        "algorithm_id": algorithm_id,
        "released_dense_bytes": original_bytes,
        "persistent_bytes": int(runtime.persistent_bytes()),
        "storage": "cusparselt_base_plus_complement_no_duplicate_metadata",
    }


@torch.inference_mode()
def mixed_sparse_linear_output(
    module: Any,
    input_tensor: torch.Tensor,
    dense_output: torch.Tensor,
) -> torch.Tensor:
    """Replace low-confidence token rows with the attached 2:4 linear output."""
    if not _STATIC_ENABLED:
        return dense_output
    mask_bytes = getattr(module, "_speclink_24_mask_bytes", None)
    # The EAGLE drafter is intentionally unmasked. Check the module-local
    # state before touching ContextVar so its torch.compile trace remains a
    # plain dense model even while target token routing is enabled.
    if mask_bytes is None:
        return dense_output
    route = _verify_route.get()
    if route is None:
        return dense_output
    if input_tensor.ndim != 2 or dense_output.ndim != 2:
        raise RuntimeError("SpecLink token-dense Llama path expects 2D tensors")

    rows = int(input_tensor.shape[0])
    if rows <= 0:
        return dense_output
    dense_mask = route.dense_mask if isinstance(route, VerifyTokenRoute) else route
    if dense_mask.numel() < rows:
        raise RuntimeError(
            f"SpecLink token-dense mask has {dense_mask.numel()} rows, "
            f"but linear input has {rows}"
        )
    row_is_dense = dense_mask[:rows].to(device=input_tensor.device)
    if bool(row_is_dense.all().item()):
        return dense_output
    sparse_rows = (~row_is_dense).nonzero(as_tuple=False).squeeze(1)
    if sparse_rows.numel() == 0:
        return dense_output

    weight = module.weight
    bias = getattr(module, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        bias = None
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    usable_in = (in_features // 4) * 4
    groups = usable_in // 4
    if usable_in <= 0:
        return dense_output

    group_mask = _expand_mask_bytes(
        mask_bytes,
        out_features=out_features,
        groups=groups,
        device=weight.device,
    )
    bits = _MASK_BITS.to(device=weight.device)
    keep = (group_mask.unsqueeze(-1) & bits.view(1, 1, 4)).ne(0)
    sparse_weight = weight[:, :usable_in] * keep.view(
        out_features, usable_in
    ).to(dtype=weight.dtype)
    row_scale = getattr(module, "_speclink_24_row_scale", None)
    if row_scale is not None:
        scale = row_scale.to(device=weight.device, dtype=weight.dtype, non_blocking=True)
        sparse_weight = sparse_weight * scale.view(-1, 1)
    if usable_in < in_features:
        sparse_weight = torch.cat([sparse_weight, weight[:, usable_in:]], dim=1)

    sparse_input = input_tensor.index_select(0, sparse_rows)
    sparse_output = F.linear(sparse_input, sparse_weight, bias)
    output = dense_output.clone()
    output.index_copy_(0, sparse_rows, sparse_output)
    return output
