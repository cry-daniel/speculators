# SPDX-License-Identifier: Apache-2.0
"""SpecLink selective dense + structured 2:4 linear wrappers.

The model files call :func:`speclink_linear_forward` instead of invoking vLLM
linears directly. Outside token-dense target verification it delegates to the
original module. During verification it requires prepacked 2:4 masks and uses
one of these runtime strategies:

* ``auto``: prefer the exact residual sparse path when prepacked; otherwise
  choose between the non-residual strategies from layer shape and dense count.
* ``full_sparse_residual``: all rows use ``W24``; dense rows add
  ``W - W24``.
* ``full_sparse_dense_override``: all rows use ``W24``; dense rows overwrite
  with dense ``W``.
* ``split_dense_sparse``: sparse rows use ``W24``; dense rows use dense ``W``.
* ``sparse_only_decode``: all verification rows use ``W24``.
"""

from __future__ import annotations

from typing import Any

import os
import torch
from torch.library import Library

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.speclink_kernel import (
    sparse24_add_indexed_rows_contiguous_,
    sparse24_add_indexed_rows_strided_,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_heterogeneous_linear_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_indexed_output_gemm_prepacked,
    sparse24_cutlass_paired_gather_residual_prepacked,
    sparse24_cutlass_paired_gather_residual_qkv_prepacked,
    sparse24_cutlass_paired_inplace_residual_prepacked,
    sparse24_gather_rows_,
    sparse24_merge_rows_,
    sparse24_mixed_dense_override_prepacked,
    sparse24_partition_rows_,
    sparse24_qkv_add_routed_residual_postop_cache_inplace_,
    sparse24_qkv_add_routed_residual_postop_inplace_,
    sparse24_transpose_add_routed_residual,
)
from vllm.speclink_token_dense import (
    current_verify_contiguous_dense_prefix,
    current_verify_dense_slots,
    current_verify_dense_row_summary,
    current_verify_prefill_row_summary,
    enabled as token_dense_enabled,
    fast_plan_enabled,
    linear_strategy,
)
from vllm.utils.torch_utils import direct_register_custom_op


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_FALSY = {"0", "false", "FALSE", "no", "NO", "off", "OFF"}
_CUSTOM_CONTIGUOUS_SCATTER = (
    os.getenv("SPECLINK_SPARSE24_CONTIGUOUS_SCATTER", "1") in _TRUTHY
)
_REUSE_SPARSE_BUFFERS = (
    os.getenv("SPECLINK_SPARSE24_REUSE_BUFFERS", "1") in _TRUTHY
)
_GRAPH_ROUTING_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_GRAPH_ROUTING", "0") in _TRUTHY
)
_CACHE_GRAPH_EAGER_FALLBACK = (
    os.getenv("SPECLINK_SPARSE24_CACHE_GRAPH_EAGER_FALLBACK", "0") in _TRUTHY
)
_SPARSE24_PAD_M_MULTIPLE = int(os.getenv("SPECLINK_SPARSE24_PAD_M_MULTIPLE", "8"))
_SPARSE24_OUTPUT_RING_SIZE = max(
    1,
    int(os.getenv("SPECLINK_SPARSE24_OUTPUT_RING", "2")),
)
_QKV_PARALLEL_RESIDUAL_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_PARALLEL_RESIDUAL", "1") in _TRUTHY
)
_QKV_CUSPARSELT_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_CUSPARSELT", "0") in _TRUTHY
)
_QKV_CUSPARSELT_MIN_ROWS = int(
    os.getenv("SPECLINK_SPARSE24_QKV_CUSPARSELT_MIN_ROWS", "576")
)
_QKV_CUSPARSELT_ALG_ID = int(
    os.getenv("SPECLINK_SPARSE24_QKV_CUSPARSELT_ALG_ID", "1")
)
_QKV_HETEROGENEOUS_ROUTING_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_HETEROGENEOUS_ROUTING", "0") in _TRUTHY
)
_QKV_HETEROGENEOUS_MAX_ROWS = int(
    os.getenv("SPECLINK_SPARSE24_QKV_HETEROGENEOUS_MAX_ROWS", "704")
)
_QKV_PAIRED_ROUTING_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_PAIRED_ROUTING", "0") in _TRUTHY
)
_QKV_PAIRED_MAX_ROWS = int(
    os.getenv("SPECLINK_SPARSE24_QKV_PAIRED_MAX_ROWS", "704")
)
_QKV_ACTIVE_WAVE_C12_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_ACTIVE_WAVE_C12", "0") in _TRUTHY
)
_QKV_VEC4_POSTOP_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_VEC4_POSTOP", "0") in _TRUTHY
)
_QKV_DIRECT_CACHE_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_DIRECT_CACHE", "0") in _TRUTHY
)
_QKV_FUSED_EPILOGUE_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_QKV_FUSED_EPILOGUE", "0") in _TRUTHY
)
_PAIRED_INPLACE_O_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_INPLACE_O", "0") in _TRUTHY
)
_SPARSE24_ACCUMULATOR = os.getenv(
    "SPECLINK_SPARSE24_ACCUMULATOR", "fp32"
).strip().lower()
_PARALLEL_SPLIT_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_PARALLEL_SPLIT", "1") in _TRUTHY
)
_PARALLEL_MIXED_OVERRIDE_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_PARALLEL_MIXED_OVERRIDE", "1") in _TRUTHY
)
_INDEXED_OUTPUT_EPILOGUE_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_INDEXED_OUTPUT_EPILOGUE", "0") in _TRUTHY
)
_DIRECT_STORE_GATE_UP_ENABLED = (
    os.getenv("SPECLINK_SPARSE24_DIRECT_STORE_GATE_UP", "0") in _TRUTHY
)
_LOG_KERNEL_DISPATCH = (
    os.getenv("SPECLINK_SPARSE24_LOG_DISPATCH", "0") in _TRUTHY
)
_QKV_PARALLEL_RESIDUAL_MIN_ROWS = int(
    os.getenv("SPECLINK_SPARSE24_QKV_PARALLEL_RESIDUAL_MIN_ROWS", "64")
)
_QKV_PARALLEL_RESIDUAL_MAX_DENSE_ROWS = 96
_SPARSE24_OUTPUT_BUFFERS: dict[tuple[str, int | None, int, int], dict[str, Any]] = {}
_SPARSE24_VIEW_OUTPUT_BUFFERS: dict[
    tuple[str, int | None, int, int], dict[str, Any]
] = {}
_SPARSE24_WORKSPACE_BUFFERS: dict[
    tuple[str, int | None, int, int], torch.Tensor
] = {}
_SPARSE24_GATHER_BUFFERS: dict[
    tuple[str, int | None, int, int], torch.Tensor
] = {}
_MIXED_OVERRIDE_BUFFERS: dict[tuple[str, int | None, int, int, int, int], dict[str, Any]] = {}
_MIXED_OVERRIDE_CACHE_BYTES = 0
_QKV_PARALLEL_RESIDUAL_STREAMS: dict[
    tuple[str, int], tuple[torch.cuda.Stream, torch.cuda.Stream]
] = {}
_QKV_PAIRED_RESIDUAL_BUFFERS: dict[
    tuple[str, int | None, int, int], dict[str, Any]
] = {}
_QKV_FUSED_EPILOGUE_BARRIERS: dict[
    tuple[str, int | None, int, int], torch.Tensor
] = {}
_PAIRED_INPLACE_O_COUNTERS: dict[
    tuple[str, int | None, int, int, int, int, str], torch.Tensor
] = {}
_MIXED_LINEAR_STREAMS: dict[
    tuple[str, int], tuple[torch.cuda.Stream, torch.cuda.Stream]
] = {}
_GATE_UP_HYBRID_STREAMS: dict[
    tuple[str, int], tuple[torch.cuda.Stream, torch.cuda.Stream]
] = {}
_SPECLINK_OP_LIB = Library("speclink", "FRAGMENT")
_DISPATCH_LOGGED: set[tuple[str, int, int, int]] = set()
logger = init_logger(__name__)


def _log_kernel_dispatch(
    backend: str,
    rows: int,
    dense_count: int,
    out_features: int,
) -> None:
    if not _LOG_KERNEL_DISPATCH:
        return
    key = (backend, rows, dense_count, out_features)
    if key in _DISPATCH_LOGGED:
        return
    _DISPATCH_LOGGED.add(key)
    logger.info(
        "SpecLink sparse24 dispatch backend=%s rows=%d dense_rows=%d "
        "out_features=%d capturing=%s",
        backend,
        rows,
        dense_count,
        out_features,
        _cuda_graph_capturing(),
    )


def _should_direct_store_gate_up(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    *,
    contiguous_output: bool,
) -> bool:
    """Select the direct row-major epilogue only for profiled Gate/Up shapes."""

    return (
        _DIRECT_STORE_GATE_UP_ENABLED
        and contiguous_output
        and input_tensor.ndim == 2
        and int(input_tensor.shape[0]) % 8 == 0
        and int(input_tensor.shape[1]) == 4096
        and int(dense_weight.shape[0]) in {24576, 28672}
    )


def _direct_store_gate_up(
    input_tensor: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    return sparse24_cutlass_inline_transpose_gemm_prepacked(
        input_tensor.contiguous(),
        full_a_values,
        full_a_meta_e,
        config="auto",
        store_mode="vector",
    )


def prepare_mixed_linear_streams(device: torch.device) -> bool:
    """Create row-routing side streams before CUDA graph capture."""

    if (
        not (_PARALLEL_SPLIT_ENABLED or _PARALLEL_MIXED_OVERRIDE_ENABLED)
        or device.type != "cuda"
        or not torch.cuda.is_available()
    ):
        return False
    key = (device.type, int(device.index or 0))
    if key not in _MIXED_LINEAR_STREAMS:
        _MIXED_LINEAR_STREAMS[key] = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
        )
    return True


def _mixed_linear_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream] | None:
    return _MIXED_LINEAR_STREAMS.get((device.type, int(device.index or 0)))


def _sparse_only_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    contiguous_output: bool,
) -> torch.Tensor:
    rows = int(input_tensor.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is not None and summary[1] == 0:
        if _should_direct_store_gate_up(
            input_tensor,
            dense_weight,
            contiguous_output=contiguous_output,
        ):
            return _direct_store_gate_up(
                input_tensor,
                full_a_values,
                full_a_meta_e,
            )
        input_transposed = (
            _use_transposed_sparse_inputs_enabled()
            and _is_transposed_sparse_input(input_tensor)
        )
        return sparse24_cutlass_device_gemm_prepacked(
            input_tensor if input_transposed else input_tensor.contiguous(),
            full_a_values,
            full_a_meta_e,
            contiguous_output=contiguous_output,
            input_transposed=input_transposed,
        )
    if contiguous_output:
        return torch.mm(input_tensor, dense_weight.t())
    padded_rows = ((rows + _SPARSE24_PAD_M_MULTIPLE - 1) // _SPARSE24_PAD_M_MULTIPLE) * (
        _SPARSE24_PAD_M_MULTIPLE
    )
    output = torch.empty_strided(
        (rows, int(dense_weight.shape[0])),
        (1, padded_rows),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    output.copy_(torch.mm(input_tensor, dense_weight.t()))
    return output


def _sparse_only_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    contiguous_output: bool,
) -> torch.Tensor:
    del dense_weight, full_a_meta_e
    shape = (input_tensor.shape[0], full_a_values.shape[0])
    if contiguous_output:
        return input_tensor.new_empty(shape)
    padded_rows = (
        (input_tensor.shape[0] + _SPARSE24_PAD_M_MULTIPLE - 1)
        // _SPARSE24_PAD_M_MULTIPLE
        * _SPARSE24_PAD_M_MULTIPLE
    )
    return torch.empty_strided(
        shape,
        (1, padded_rows),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )


def _hybrid_gate_up_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    sparse_a_values: torch.Tensor,
    sparse_a_meta_e: torch.Tensor,
    sparse_first: bool,
) -> torch.Tensor:
    """Keep one fused gate/up half dense and execute the other as 2:4."""

    rows = int(input_tensor.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None or summary[1] != 0:
        return torch.mm(input_tensor, dense_weight.t())
    out_features = int(dense_weight.shape[0])
    if out_features % 2:
        raise RuntimeError(
            "SpecLink gate/up hybrid requires an even output dimension"
        )
    split = out_features // 2
    dense_slice = slice(split, None) if sparse_first else slice(0, split)
    streams = _gate_up_hybrid_streams(input_tensor.device)
    if streams is None:
        dense_output = torch.mm(input_tensor, dense_weight[dense_slice].t())
        sparse_output = sparse24_cutlass_device_gemm_prepacked(
            input_tensor.contiguous(),
            sparse_a_values,
            sparse_a_meta_e,
            contiguous_output=True,
        )
    else:
        dense_stream, sparse_stream = streams
        current_stream = torch.cuda.current_stream(input_tensor.device)
        dense_stream.wait_stream(current_stream)
        sparse_stream.wait_stream(current_stream)
        with torch.cuda.stream(dense_stream):
            dense_output = torch.mm(input_tensor, dense_weight[dense_slice].t())
        with torch.cuda.stream(sparse_stream):
            sparse_output = sparse24_cutlass_device_gemm_prepacked(
                input_tensor.contiguous(),
                sparse_a_values,
                sparse_a_meta_e,
                contiguous_output=True,
            )
        current_stream.wait_stream(dense_stream)
        current_stream.wait_stream(sparse_stream)
    if sparse_first:
        return torch.cat((sparse_output, dense_output), dim=1)
    return torch.cat((dense_output, sparse_output), dim=1)


def _hybrid_gate_up_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    sparse_a_values: torch.Tensor,
    sparse_a_meta_e: torch.Tensor,
    sparse_first: bool,
) -> torch.Tensor:
    del sparse_a_values, sparse_a_meta_e, sparse_first
    return input_tensor.new_empty(
        (input_tensor.shape[0], dense_weight.shape[0])
    )


def _force_sparse_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    contiguous_output: bool,
) -> torch.Tensor:
    """Use 2:4 for decode rows while preserving mixed prefill rows exactly."""

    rows = int(input_tensor.shape[0])
    if _prefer_dense_static_projection(
        rows,
        in_features=int(dense_weight.shape[1]),
        out_features=int(dense_weight.shape[0]),
    ):
        return _dense_linear_with_layout(
            input_tensor,
            dense_weight,
            contiguous_output=contiguous_output,
        )
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None:
        return _dense_linear_with_layout(
            input_tensor,
            dense_weight,
            contiguous_output=contiguous_output,
        )

    prefill_summary = current_verify_prefill_row_summary(rows)
    if prefill_summary is not None:
        if _prefer_dense_mixed_prefill_projection(
            in_features=int(dense_weight.shape[1]),
            out_features=int(dense_weight.shape[0]),
        ):
            return _dense_linear_with_layout(
                input_tensor,
                dense_weight,
                contiguous_output=contiguous_output,
            )
        prefill_count, prefill_rows, decode_rows, prefill_layout = prefill_summary
        mixed_output = _split_dense_sparse_rows_impl(
            input_tensor,
            dense_weight,
            full_a_values,
            full_a_meta_e,
            dense_count=prefill_count,
            dense_rows=prefill_rows,
            sparse_rows=decode_rows,
            contiguous_dense_prefix=prefill_layout == "prefix",
            contiguous_dense_suffix=prefill_layout == "suffix",
        )
        if contiguous_output:
            return mixed_output
        padded_rows = _sparse24_m_run(rows)
        output = torch.empty_strided(
            (rows, int(dense_weight.shape[0])),
            (1, padded_rows),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        output.copy_(mixed_output)
        return output

    input_transposed = (
        _use_transposed_sparse_inputs_enabled()
        and _is_transposed_sparse_input(input_tensor)
    )
    if _should_direct_store_gate_up(
        input_tensor,
        dense_weight,
        contiguous_output=contiguous_output,
    ):
        return _direct_store_gate_up(
            input_tensor,
            full_a_values,
            full_a_meta_e,
        )
    return sparse24_cutlass_device_gemm_prepacked(
        input_tensor if input_transposed else input_tensor.contiguous(),
        full_a_values,
        full_a_meta_e,
        contiguous_output=contiguous_output,
        input_transposed=input_transposed,
    )


def _released_force_sparse_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    contiguous_output: bool,
) -> torch.Tensor:
    """Run static W24 in verify and reconstruct exact W elsewhere."""

    rows = int(input_tensor.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None:
        output = _full_sparse_residual_linear_core(
            input_tensor,
            dense_weight,
            full_a_values,
            full_a_meta_e,
            residual_a_values,
            residual_a_meta_e,
            None,
        )
    else:
        prefill_summary = current_verify_prefill_row_summary(rows)
        if prefill_summary is None:
            return sparse24_cutlass_device_gemm_prepacked(
                input_tensor.contiguous(),
                full_a_values,
                full_a_meta_e,
                contiguous_output=contiguous_output,
            )

        prefill_count, prefill_rows, _decode_rows, _prefill_layout = (
            prefill_summary
        )
        output = sparse24_cutlass_device_gemm_prepacked(
            input_tensor.contiguous(),
            full_a_values,
            full_a_meta_e,
            contiguous_output=True,
        )
        if prefill_count > 0:
            prefill_input = _gather_rows(input_tensor, prefill_rows)
            residual_output = sparse24_cutlass_device_gemm_prepacked(
                prefill_input.contiguous(),
                residual_a_values,
                residual_a_meta_e,
                contiguous_output=True,
            )
            _add_indexed_rows_(output, residual_output, prefill_rows)

    if contiguous_output:
        return output
    padded_rows = _sparse24_m_run(rows)
    strided_output = torch.empty_strided(
        (rows, int(full_a_values.shape[0])),
        (1, padded_rows),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    strided_output.copy_(output)
    return strided_output


def _prefer_dense_static_projection(
    rows: int,
    *,
    in_features: int,
    out_features: int,
) -> bool:
    # Qwen3-8B down_proj SparseGemm crosses dense cuBLAS at 48 rows on
    # Blackwell. Continuous batches frequently visit this tail region.
    return rows < 48 and in_features == 12288 and out_features == 4096


def _prefer_dense_mixed_prefill_projection(
    *,
    in_features: int,
    out_features: int,
) -> bool:
    # Partitioning a mixed batch around Qwen's wide down_proj costs more than
    # keeping the full projection dense. Pure decode batches remain 2:4.
    return in_features == 12288 and out_features == 4096


def _dense_linear_with_layout(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    *,
    contiguous_output: bool,
) -> torch.Tensor:
    dense_output = torch.mm(input_tensor, dense_weight.t())
    if contiguous_output:
        return dense_output
    rows = int(input_tensor.shape[0])
    output = torch.empty_strided(
        (rows, int(dense_weight.shape[0])),
        (1, _sparse24_m_run(rows)),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    output.copy_(dense_output)
    return output


def _force_sparse_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    contiguous_output: bool,
) -> torch.Tensor:
    return _sparse_only_linear_fake(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        contiguous_output,
    )


def _mixed_dense_override_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    """Graph-safe dynamic routing behind an opaque custom-op boundary."""

    rows = int(input_tensor.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None or summary[1] == rows:
        return torch.mm(input_tensor, dense_weight.t())

    _row_is_dense, dense_count, dense_rows, sparse_rows = summary
    if dense_count == 0:
        return sparse24_cutlass_device_gemm_prepacked(
            input_tensor.contiguous(),
            full_a_values,
            full_a_meta_e,
            contiguous_output=True,
        )
    heterogeneous_output = _qkv_heterogeneous_exact(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        dense_rows,
        sparse_rows,
    )
    if heterogeneous_output is not None:
        return heterogeneous_output
    if current_verify_contiguous_dense_prefix(rows) == dense_count:
        output = sparse24_cutlass_device_gemm_prepacked(
            input_tensor.contiguous(),
            full_a_values,
            full_a_meta_e,
            contiguous_output=True,
        )
        torch.mm(
            input_tensor[:dense_count],
            dense_weight.t(),
            out=output[:dense_count],
        )
        return output
    streams = _mixed_linear_streams(input_tensor.device)
    if _PARALLEL_MIXED_OVERRIDE_ENABLED and streams is not None:
        dense_rows = dense_rows[:dense_count]
        input_tensor = input_tensor.contiguous()
        sparse_stream, dense_stream = streams
        current_stream = torch.cuda.current_stream(input_tensor.device)
        sparse_stream.wait_stream(current_stream)
        dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(sparse_stream):
            output = sparse24_cutlass_device_gemm_prepacked(
                input_tensor,
                full_a_values,
                full_a_meta_e,
                contiguous_output=True,
            )
        with torch.cuda.stream(dense_stream):
            dense_input = _gather_rows(
                input_tensor,
                dense_rows,
                reuse_during_capture=True,
            )
            dense_output = torch.mm(dense_input, dense_weight.t())
        current_stream.wait_stream(sparse_stream)
        current_stream.wait_stream(dense_stream)
        sparse24_copy_indexed_rows_contiguous_(
            output,
            dense_output,
            dense_rows,
        )
        return output
    return sparse24_mixed_dense_override_prepacked(
        input_tensor.contiguous(),
        dense_weight,
        full_a_values,
        full_a_meta_e,
        dense_rows[:dense_count],
    )


def _mixed_dense_override_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    del dense_weight, full_a_meta_e
    return input_tensor.new_empty(
        (input_tensor.shape[0], full_a_values.shape[0])
    )


def _split_dense_sparse_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    """Graph-safe disjoint dense/sparse row GEMMs."""

    rows = int(input_tensor.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None or summary[1] == rows:
        return torch.mm(input_tensor, dense_weight.t())
    _row_is_dense, dense_count, dense_rows, sparse_rows = summary
    return _split_dense_sparse_rows_impl(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        dense_count=dense_count,
        dense_rows=dense_rows,
        sparse_rows=sparse_rows,
        contiguous_dense_prefix=(
            current_verify_contiguous_dense_prefix(rows) == dense_count
        ),
        contiguous_dense_suffix=False,
    )


def _split_dense_sparse_rows_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    *,
    dense_count: int,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    contiguous_dense_prefix: bool,
    contiguous_dense_suffix: bool,
) -> torch.Tensor:
    """Run disjoint dense and 2:4 GEMMs for an explicit row partition."""

    rows = int(input_tensor.shape[0])
    if dense_count == 0:
        return sparse24_cutlass_device_gemm_prepacked(
            input_tensor.contiguous(),
            full_a_values,
            full_a_meta_e,
            contiguous_output=True,
        )
    if dense_count == rows:
        return torch.mm(input_tensor, dense_weight.t())

    sparse_count = rows - dense_count
    streams = _mixed_linear_streams(input_tensor.device)
    if contiguous_dense_prefix or contiguous_dense_suffix:
        input_tensor = input_tensor.contiguous()
        if contiguous_dense_prefix:
            dense_input = input_tensor[:dense_count]
            sparse_input = input_tensor[dense_count:]
        else:
            sparse_input = input_tensor[:sparse_count]
            dense_input = input_tensor[sparse_count:]

        if streams is not None:
            dense_stream, sparse_stream = streams
            current_stream = torch.cuda.current_stream(input_tensor.device)
            dense_stream.wait_stream(current_stream)
            sparse_stream.wait_stream(current_stream)
            with torch.cuda.stream(dense_stream):
                dense_output = torch.mm(dense_input, dense_weight.t())
            with torch.cuda.stream(sparse_stream):
                sparse_output = sparse24_cutlass_device_gemm_prepacked(
                    sparse_input,
                    full_a_values,
                    full_a_meta_e,
                    contiguous_output=True,
                )
            current_stream.wait_stream(dense_stream)
            current_stream.wait_stream(sparse_stream)
        else:
            dense_output = torch.mm(dense_input, dense_weight.t())
            sparse_output = sparse24_cutlass_device_gemm_prepacked(
                sparse_input,
                full_a_values,
                full_a_meta_e,
                contiguous_output=True,
            )
        return torch.cat(
            (dense_output, sparse_output)
            if contiguous_dense_prefix
            else (sparse_output, dense_output),
            dim=0,
        )

    if streams is not None and sparse_count > 0:
        dense_rows = dense_rows[:dense_count]
        sparse_rows = sparse_rows[:sparse_count]
        input_tensor = input_tensor.contiguous()
        dense_input = torch.empty(
            (dense_count, input_tensor.shape[1]),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        sparse_rows_padded = (sparse_count + 7) // 8 * 8
        sparse_input_padded = torch.empty(
            (sparse_rows_padded, input_tensor.shape[1]),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        sparse24_partition_rows_(
            input_tensor,
            dense_rows,
            sparse_rows,
            dense_input,
            sparse_input_padded[:sparse_count],
        )
        if sparse_rows_padded != sparse_count:
            sparse_input_padded[sparse_count:].zero_()

        output = (
            torch.empty(
                (rows, dense_weight.shape[0]),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            if _INDEXED_OUTPUT_EPILOGUE_ENABLED
            else None
        )
        dense_stream, sparse_stream = streams
        current_stream = torch.cuda.current_stream(input_tensor.device)
        dense_stream.wait_stream(current_stream)
        sparse_stream.wait_stream(current_stream)
        with torch.cuda.stream(dense_stream):
            dense_output = torch.mm(dense_input, dense_weight.t())
        with torch.cuda.stream(sparse_stream):
            if _INDEXED_OUTPUT_EPILOGUE_ENABLED:
                assert output is not None
                sparse24_cutlass_indexed_output_gemm_prepacked(
                    sparse_input_padded,
                    full_a_values,
                    full_a_meta_e,
                    sparse_rows,
                    output_rows=rows,
                    out=output,
                    config="auto",
                )
                sparse_output_padded = None
            else:
                sparse_output_padded = sparse24_cutlass_device_gemm_prepacked(
                    sparse_input_padded,
                    full_a_values,
                    full_a_meta_e,
                    contiguous_output=True,
                )
        current_stream.wait_stream(dense_stream)
        current_stream.wait_stream(sparse_stream)

        if _INDEXED_OUTPUT_EPILOGUE_ENABLED:
            assert output is not None
            sparse24_copy_indexed_rows_contiguous_(
                output,
                dense_output,
                dense_rows,
            )
            return output
        assert sparse_output_padded is not None
        output = torch.empty(
            (rows, dense_weight.shape[0]),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        sparse24_merge_rows_(
            output,
            dense_output,
            sparse_output_padded[:sparse_count],
            dense_rows,
            sparse_rows,
        )
        return output

    out_features = int(dense_weight.shape[0])
    output = torch.empty(
        (rows, out_features),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    sparse_input = _gather_rows(input_tensor, sparse_rows[:sparse_count])
    sparse_output = sparse24_cutlass_device_gemm_prepacked(
        sparse_input,
        full_a_values,
        full_a_meta_e,
        contiguous_output=True,
    )
    dense_input = _gather_rows(input_tensor, dense_rows[:dense_count])
    dense_output = torch.mm(dense_input, dense_weight.t())
    sparse24_copy_indexed_rows_contiguous_(
        output,
        sparse_output,
        sparse_rows[:sparse_count],
    )
    sparse24_copy_indexed_rows_contiguous_(
        output,
        dense_output,
        dense_rows[:dense_count],
    )
    return output


def _split_dense_sparse_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    del dense_weight, full_a_meta_e
    return input_tensor.new_empty(
        (input_tensor.shape[0], full_a_values.shape[0])
    )


def _qkv_cusparselt_parallel_residual(
    input_tensor: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor | None,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_count: int,
) -> torch.Tensor | None:
    """Run the large-M QKV sparse body and compact residual concurrently."""

    rows = int(input_tensor.shape[0])
    if (
        not _QKV_CUSPARSELT_ENABLED
        or not isinstance(qkv_cusparselt_packed, torch.Tensor)
        or int(qkv_cusparselt_packed.numel()) == 0
        or rows < _QKV_CUSPARSELT_MIN_ROWS
        or int(input_tensor.shape[1]) != 4096
        or int(residual_a_values.shape[0]) != 6144
        or dense_count <= 0
        or dense_count > _QKV_PARALLEL_RESIDUAL_MAX_DENSE_ROWS
        or dense_count >= rows
    ):
        return None
    streams = _qkv_parallel_residual_streams(input_tensor.device)
    dense_slots = current_verify_dense_slots(rows, input_tensor.device)
    if streams is None or dense_slots is None:
        return None

    full_stream, residual_stream = streams
    current_stream = torch.cuda.current_stream(input_tensor.device)
    full_stream.wait_stream(current_stream)
    residual_stream.wait_stream(current_stream)
    with torch.cuda.stream(full_stream):
        full_output = torch._cslt_sparse_mm(
            qkv_cusparselt_packed,
            input_tensor.contiguous().t(),
            transpose_result=False,
            alg_id=_QKV_CUSPARSELT_ALG_ID,
        ).t()
    with torch.cuda.stream(residual_stream):
        dense_input = _gather_rows(
            input_tensor,
            dense_rows,
            reuse_during_capture=True,
        )
        residual_output = sparse24_cutlass_device_gemm_prepacked(
            dense_input.contiguous(),
            residual_a_values,
            residual_a_meta_e,
            contiguous_output=False,
        )
    current_stream.wait_stream(full_stream)
    current_stream.wait_stream(residual_stream)
    return sparse24_transpose_add_routed_residual(
        full_output,
        residual_output,
        dense_slots,
        dense_count=dense_count,
    )


def _cached_qkv_paired_residual(
    *,
    rows: int,
    out_features: int,
    device: torch.device,
    create: bool,
) -> torch.Tensor | None:
    key = (device.type, device.index, rows, out_features)
    ring_size = _output_ring_size()
    entry = _QKV_PAIRED_RESIDUAL_BUFFERS.get(key)
    if entry is None or len(entry["buffers"]) != ring_size:
        if not create:
            return None
        entry = {
            "buffers": [
                torch.empty(
                    (rows, out_features),
                    device=device,
                    dtype=torch.float16,
                )
                for _ in range(ring_size)
            ],
            "next": 0,
        }
        _QKV_PAIRED_RESIDUAL_BUFFERS[key] = entry
    buffers = entry["buffers"]
    index = int(entry["next"])
    entry["next"] = (index + 1) % len(buffers)
    return buffers[index]


def _cached_qkv_fused_epilogue_barrier(
    *,
    rows: int,
    dense_count: int,
    device: torch.device,
    create: bool,
) -> torch.Tensor | None:
    key = (device.type, device.index, rows, dense_count)
    barrier = _QKV_FUSED_EPILOGUE_BARRIERS.get(key)
    if barrier is None and create:
        # The persistent kernel resets arrivals and toggles the sense word, so
        # one stable buffer per graph shape is reusable across graph replays.
        barrier = torch.zeros((2,), device=device, dtype=torch.int32)
        _QKV_FUSED_EPILOGUE_BARRIERS[key] = barrier
    return barrier


_QKV_PAIRED_C9 = "256x32_full_256x32_residual_contiguous"
_QKV_PAIRED_C12 = "256x64_full_256x64_residual_contiguous"
_QKV_PAIRED_C13 = "256x64_full_256x32_residual_contiguous"


def _qkv_paired_config(
    rows: int,
    dense_count: int,
    out_features: int,
) -> str | None:
    feature_tiles = (out_features + 255) // 256
    c9_full_row_tiles = (rows + 31) // 32
    c9_residual_row_tiles = (dense_count + 31) // 32
    if feature_tiles * (c9_full_row_tiles + c9_residual_row_tiles) <= 168:
        return _QKV_PAIRED_C9

    # C9 incurs an extra full-row tile above 192 Qwen rows. A layer-cold sweep
    # over every K=8 active wave and the bs/K endpoints found C12 fastest from
    # that boundary through the largest supported verifier graph. This range
    # also avoids dropping tail waves into the generic multi-launch fallback.
    if (
        _QKV_ACTIVE_WAVE_C12_ENABLED
        and out_features == 6144
        and 193 <= rows <= 704
        and dense_count <= 64
    ):
        return _QKV_PAIRED_C12
    if out_features == 6144 and rows in {224, 288, 352}:
        if dense_count <= 32:
            return _QKV_PAIRED_C13
        if dense_count <= 64:
            return _QKV_PAIRED_C12
    if (
        out_features == 6144
        and rows in {576, 704}
        and dense_count <= 64
    ):
        return _QKV_PAIRED_C13
    return None


def _qkv_use_paired_backend(
    rows: int,
    dense_count: int,
    out_features: int,
) -> bool:
    return _qkv_paired_config(rows, dense_count, out_features) is not None


def _qkv_paired_exact(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
) -> torch.Tensor | None:
    """Run contiguous W24 plus gathered complementary R24 in one grid."""

    rows = int(input_tensor.shape[0])
    dense_count = int(dense_rows.numel())
    out_features = int(full_a_values.shape[0])
    paired_config = _qkv_paired_config(rows, dense_count, out_features)
    dense_weight_released = int(dense_weight.numel()) == 0
    dense_weight_shape_valid = dense_weight_released or (
        tuple(dense_weight.shape) == (out_features, 4096)
    )
    if (
        not _QKV_PAIRED_ROUTING_ENABLED
        or paired_config is None
        or not input_tensor.is_cuda
        or input_tensor.dtype != torch.float16
        or not input_tensor.is_contiguous()
        or rows <= 0
        or rows > _QKV_PAIRED_MAX_ROWS
        or int(input_tensor.shape[1]) != 4096
        or out_features not in {5120, 6144}
        or not dense_weight_shape_valid
        or dense_count <= 0
        or dense_count >= rows
        or current_verify_prefill_row_summary(rows) is not None
    ):
        return None

    _log_kernel_dispatch("qkv_paired", rows, dense_count, out_features)

    capturing = _cuda_graph_capturing()
    full_cached = _cached_sparse_buffers(
        rows=rows,
        out_features=out_features,
        device=input_tensor.device,
        create=not capturing,
    )
    residual_out = _cached_qkv_paired_residual(
        rows=dense_count,
        out_features=out_features,
        device=input_tensor.device,
        create=not capturing,
    )
    if full_cached is None or residual_out is None:
        return None
    full_out = full_cached[0][:rows]
    sparse24_cutlass_paired_gather_residual_prepacked(
        input_tensor,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        dense_rows,
        full_out=full_out,
        residual_out=residual_out,
        schedule="interleaved",
        config=paired_config,
    )
    sparse24_add_indexed_rows_contiguous_(
        full_out,
        residual_out,
        dense_rows,
    )
    return full_out


_PAIRED_INPLACE_O_M128N32 = "128x32_full_128x32_residual_inplace"
_PAIRED_INPLACE_O_M256N32 = "256x32_full_256x32_residual_inplace"
_PAIRED_INPLACE_O_M256N64 = "256x64_full_256x32_residual_inplace"


def _paired_inplace_o_config(rows: int, dense_count: int) -> str | None:
    """Select the measured-best exact O kernel for bs>=16 decode waves."""

    if (
        not _PAIRED_INPLACE_O_ENABLED
        or not 0 < dense_count <= 64
        or not 112 <= rows <= 704
    ):
        return None

    # These ranges reproduce the best tile at every active request count from
    # 16 through 64 for K={6,8,10}. The split at 32 dense rows also separates
    # the fixed Qwen and Llama routing budgets without model-specific branches.
    if rows <= 126 and dense_count <= 32:
        return _PAIRED_INPLACE_O_M128N32
    if dense_count <= 32:
        if rows <= 288 or rows >= 577:
            return _PAIRED_INPLACE_O_M256N32
        return _PAIRED_INPLACE_O_M256N64
    if rows <= 256 or 513 <= rows <= 640:
        return _PAIRED_INPLACE_O_M256N32
    return _PAIRED_INPLACE_O_M256N64


def _paired_inplace_o_counters(
    device: torch.device,
    rows: int,
    dense_count: int,
    out_features: int,
    packed_weight_ptr: int,
    config: str,
) -> torch.Tensor | None:
    key = (
        device.type,
        device.index,
        rows,
        dense_count,
        out_features,
        packed_weight_ptr,
        config,
    )
    counters = _PAIRED_INPLACE_O_COUNTERS.get(key)
    if counters is None:
        if _cuda_graph_capturing():
            return None
        feature_columns = 128 if config.startswith("128x") else 256
        counters = torch.zeros(
            (out_features + feature_columns - 1) // feature_columns,
            device=device,
            dtype=torch.int32,
        )
        _PAIRED_INPLACE_O_COUNTERS[key] = counters
    return counters


def _paired_inplace_o_exact(
    input_tensor: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
) -> torch.Tensor | None:
    """Run square 4096-wide O projection with an exact residual epilogue."""

    rows = int(input_tensor.shape[0])
    dense_count = int(dense_rows.numel())
    out_features = int(full_a_values.shape[0])
    config = _paired_inplace_o_config(rows, dense_count)
    if (
        config is None
        or not input_tensor.is_cuda
        or input_tensor.dtype != torch.float16
        or not input_tensor.is_contiguous()
        or int(input_tensor.shape[1]) != 4096
        or out_features != 4096
        or dense_count >= rows
        or current_verify_prefill_row_summary(rows) is not None
    ):
        return None

    counters = _paired_inplace_o_counters(
        input_tensor.device,
        rows,
        dense_count,
        out_features,
        full_a_values.data_ptr(),
        config,
    )
    if counters is None:
        return None

    _log_kernel_dispatch(
        f"o_paired_inplace_{config}",
        rows,
        dense_count,
        out_features,
    )
    output = torch.empty(
        (rows, out_features),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    sparse24_cutlass_paired_inplace_residual_prepacked(
        input_tensor,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        dense_rows,
        out=output,
        feature_counters=counters,
        config=config,
        schedule="partitioned",
    )
    return output


def _qkv_heterogeneous_config(
    rows: int,
    dense_count: int,
    out_features: int = 6144,
) -> str:
    """Choose the profiled QKV route from live verifier row counts."""

    fp16_accumulator = _SPARSE24_ACCUMULATOR in {
        "fp16",
        "fp16_qkv_gate",
    }
    # Qwen3-8B and Llama-3.1-8B both have a 6144-wide fused QKV. Distinguish
    # their fixed routing budgets by the live dense fraction instead of shape.
    high_dense_budget = dense_count * 5 >= rows
    if high_dense_budget:
        if rows < 144:
            family = "n32_n32"
        elif rows < 256:
            family = "n64_n32"
        elif rows < 512:
            family = "n128_n64"
        elif rows < 640:
            family = "n64_n64"
        else:
            family = "n128_n64"
        high_budget_configs = {
            "n32_n32": (
                "256x32x64_s3_sw4_f16"
                if fp16_accumulator
                else "256x32x64_s3_sw4"
            ),
            "n64_n32": (
                "256x64_sparse_128x32_dense_s3_f16"
                if fp16_accumulator
                else "256x64_sparse_128x32_dense_s3"
            ),
            "n64_n64": (
                "256x64_sparse_128x64_dense_s3_f16"
                if fp16_accumulator
                else "256x64x64_s3"
            ),
            "n128_n64": (
                "256x128_sparse_128x64_dense_s2_f16"
                if fp16_accumulator
                else "256x128_sparse_128x64_dense_s2"
            ),
        }
        return high_budget_configs[family]
    if rows < 144:
        return (
            "256x32x64_s3_sw4_f16"
            if fp16_accumulator
            else "256x32x64_s3_sw4"
        )
    if 352 <= rows < 576:
        return (
            "256x128_sparse_128x64_dense_s2_f16"
            if fp16_accumulator
            else "256x128_sparse_128x64_dense_s2"
        )
    if dense_count >= 64 and dense_count % 64 == 0:
        return (
            "256x64_sparse_128x64_dense_s3_f16"
            if fp16_accumulator
            else "256x64x64_s3"
        )
    return (
        "256x64_sparse_128x32_dense_s3_f16"
        if fp16_accumulator
        else "256x64_sparse_128x32_dense_s3"
    )


def _qkv_heterogeneous_exact(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
) -> torch.Tensor | None:
    """Run profiled Qwen/Llama QKV dense/2:4 rows in one indexed launch."""

    rows = int(input_tensor.shape[0])
    out_features = int(dense_weight.shape[0])
    if (
        not _QKV_HETEROGENEOUS_ROUTING_ENABLED
        or not input_tensor.is_cuda
        or input_tensor.dtype != torch.float16
        or not input_tensor.is_contiguous()
        or rows <= 0
        or rows > _QKV_HETEROGENEOUS_MAX_ROWS
        or int(input_tensor.shape[1]) != 4096
        or out_features not in {5120, 6144}
        or tuple(dense_weight.shape) != (out_features, 4096)
        or dense_weight.dtype != torch.float16
        or not dense_weight.is_contiguous()
        or int(dense_rows.numel()) <= 0
        or int(sparse_rows.numel()) <= 0
        or int(dense_rows.numel()) + int(sparse_rows.numel()) != rows
    ):
        return None

    _log_kernel_dispatch("qkv_heterogeneous", rows, int(dense_rows.numel()), out_features)

    out = None
    if _reuse_sparse_buffers_enabled():
        cached = _cached_sparse_buffers(
            rows=rows,
            out_features=out_features,
            device=input_tensor.device,
            create=not _cuda_graph_capturing(),
        )
        if cached is not None and tuple(cached[0].shape) == (
            rows,
            out_features,
        ):
            out = cached[0]
    return sparse24_cutlass_heterogeneous_linear_prepacked(
        input_tensor,
        full_a_values,
        full_a_meta_e,
        dense_weight,
        dense_rows,
        sparse_rows,
        out=out,
        config=_qkv_heterogeneous_config(
            rows,
            int(dense_rows.numel()),
            out_features,
        ),
    )


def _full_sparse_residual_linear_core(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor | None,
) -> torch.Tensor:
    rows = int(input_tensor.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None:
        if int(dense_weight.numel()) > 0:
            return torch.mm(input_tensor, dense_weight.t())
        dense_count = rows
        dense_rows = torch.arange(
            rows,
            device=input_tensor.device,
            dtype=torch.int32,
        )
        sparse_rows = dense_rows[:0]
    else:
        _row_is_dense, dense_count, dense_rows, sparse_rows = summary

    if dense_count == rows and int(dense_weight.numel()) > 0:
        return torch.mm(input_tensor, dense_weight.t())

    inplace_o_output = _paired_inplace_o_exact(
        input_tensor,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        dense_rows,
    )
    if inplace_o_output is not None:
        return inplace_o_output

    paired_output = _qkv_paired_exact(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        dense_rows,
    )
    if paired_output is not None:
        return paired_output

    heterogeneous_output = _qkv_heterogeneous_exact(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        dense_rows,
        sparse_rows,
    )
    if heterogeneous_output is not None:
        return heterogeneous_output

    qkv_output = _qkv_cusparselt_parallel_residual(
        input_tensor,
        qkv_cusparselt_packed,
        residual_a_values,
        residual_a_meta_e,
        dense_rows,
        dense_count,
    )
    if qkv_output is not None:
        return qkv_output

    def sparse_gemm(
        x: torch.Tensor,
        values: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            x.contiguous(),
            values,
            meta,
            contiguous_output=True,
        )

    if (
        rows >= _QKV_PARALLEL_RESIDUAL_MIN_ROWS
        and (
            (
                int(input_tensor.shape[1]) == 4096
                and int(full_a_values.shape[0]) == 6144
            )
            or (
                int(input_tensor.shape[1]) == 12288
                and int(full_a_values.shape[0]) == 4096
            )
        )
        and 0 < dense_count <= _QKV_PARALLEL_RESIDUAL_MAX_DENSE_ROWS
        and dense_count < rows
    ):
        streams = _qkv_parallel_residual_streams(input_tensor.device)
        if streams is not None:
            full_stream, residual_stream = streams
            current_stream = torch.cuda.current_stream(input_tensor.device)
            full_stream.wait_stream(current_stream)
            residual_stream.wait_stream(current_stream)
            with torch.cuda.stream(full_stream):
                output = sparse_gemm(
                    input_tensor,
                    full_a_values,
                    full_a_meta_e,
                )
            with torch.cuda.stream(residual_stream):
                dense_input = _gather_rows(
                    input_tensor,
                    dense_rows,
                    reuse_during_capture=True,
                )
                residual_output = sparse_gemm(
                    dense_input,
                    residual_a_values,
                    residual_a_meta_e,
                )
            current_stream.wait_stream(full_stream)
            current_stream.wait_stream(residual_stream)
            return _add_indexed_rows_(output, residual_output, dense_rows)

    output = sparse_gemm(input_tensor, full_a_values, full_a_meta_e)
    if dense_count > 0:
        dense_input = _gather_rows(input_tensor, dense_rows)
        residual_output = sparse_gemm(
            dense_input,
            residual_a_values,
            residual_a_meta_e,
        )
        _add_indexed_rows_(output, residual_output, dense_rows)
    return output


def _full_sparse_residual_linear_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    return _full_sparse_residual_linear_core(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        None,
    )


def _full_sparse_residual_linear_v3_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    use_qkv_cusparselt: bool,
) -> torch.Tensor:
    return _full_sparse_residual_linear_core(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        qkv_cusparselt_packed if use_qkv_cusparselt else None,
    )


def _apply_standard_qkv_postop(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_size: int,
    kv_size: int,
    epsilon: float,
    normalize_qk: bool,
    is_neox: bool,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    head_dim = 128
    if normalize_qk:
        ops.fused_qk_norm_rope(
            qkv,
            q_size // head_dim,
            kv_size // head_dim,
            kv_size // head_dim,
            head_dim,
            epsilon,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            positions,
            -1,
        )
    else:
        q, k, _v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        ops.rotary_embedding(
            positions,
            q,
            k,
            head_dim,
            cos_sin_cache,
            is_neox,
        )
    return qkv


def _qkv_routed_postop_core(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    use_qkv_cusparselt: bool,
    q_size: int,
    kv_size: int,
    epsilon: float,
    normalize_qk: bool,
    is_neox: bool,
    cache_context: tuple[Any, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    rows = int(input_tensor.shape[0])
    out_features = int(full_a_values.shape[0])
    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is not None:
        _row_is_dense, dense_count, dense_rows, _sparse_rows = summary
        dense_slots = current_verify_dense_slots(rows, input_tensor.device)
        paired_config = _qkv_paired_config(rows, dense_count, out_features)
        fused_epilogue = (
            _QKV_FUSED_EPILOGUE_ENABLED
            and cache_context is None
            and paired_config == _QKV_PAIRED_C13
            and dense_slots is not None
            and 0 < dense_count < rows
            and out_features == q_size + 2 * kv_size
        )
        if fused_epilogue:
            capturing = _cuda_graph_capturing()
            full_cached = _cached_sparse_buffers(
                rows=rows,
                out_features=out_features,
                device=input_tensor.device,
                create=not capturing,
            )
            residual_out = _cached_qkv_paired_residual(
                rows=dense_count,
                out_features=out_features,
                device=input_tensor.device,
                create=not capturing,
            )
            grid_barrier = _cached_qkv_fused_epilogue_barrier(
                rows=rows,
                dense_count=dense_count,
                device=input_tensor.device,
                create=not capturing,
            )
            if (
                full_cached is not None
                and residual_out is not None
                and grid_barrier is not None
            ):
                _log_kernel_dispatch(
                    "qkv_paired_fused_epilogue",
                    rows,
                    dense_count,
                    out_features,
                )
                full_out = full_cached[0][:rows]
                sparse24_cutlass_paired_gather_residual_qkv_prepacked(
                    input_tensor,
                    full_a_values,
                    full_a_meta_e,
                    residual_a_values,
                    residual_a_meta_e,
                    dense_rows,
                    dense_slots,
                    cos_sin_cache,
                    positions,
                    grid_barrier,
                    q_size=q_size,
                    kv_size=kv_size,
                    head_dim=128,
                    epsilon=epsilon,
                    is_neox=is_neox,
                    q_weight=q_weight if normalize_qk else None,
                    k_weight=k_weight if normalize_qk else None,
                    full_out=full_out,
                    residual_out=residual_out,
                    schedule="interleaved",
                    config=paired_config,
                )
                return full_out
        if (
            _QKV_VEC4_POSTOP_ENABLED
            and paired_config is not None
            and dense_slots is not None
            and 0 < dense_count < rows
            and out_features == q_size + 2 * kv_size
        ):
            dispatch = (
                "qkv_paired_vec4_postop_cache"
                if cache_context is not None
                else "qkv_paired_vec4_postop"
            )
            _log_kernel_dispatch(dispatch, rows, dense_count, out_features)
            capturing = _cuda_graph_capturing()
            full_cached = _cached_sparse_buffers(
                rows=rows,
                out_features=out_features,
                device=input_tensor.device,
                create=not capturing,
            )
            residual_out = _cached_qkv_paired_residual(
                rows=dense_count,
                out_features=out_features,
                device=input_tensor.device,
                create=not capturing,
            )
            if full_cached is not None and residual_out is not None:
                full_out = full_cached[0][:rows]
                sparse24_cutlass_paired_gather_residual_prepacked(
                    input_tensor,
                    full_a_values,
                    full_a_meta_e,
                    residual_a_values,
                    residual_a_meta_e,
                    dense_rows,
                    full_out=full_out,
                    residual_out=residual_out,
                    schedule="interleaved",
                    config=paired_config,
                )
                if cache_context is None:
                    sparse24_qkv_add_routed_residual_postop_inplace_(
                        full_out,
                        residual_out,
                        dense_slots,
                        cos_sin_cache,
                        positions,
                        q_size=q_size,
                        kv_size=kv_size,
                        head_dim=128,
                        epsilon=epsilon,
                        is_neox=is_neox,
                        q_weight=q_weight if normalize_qk else None,
                        k_weight=k_weight if normalize_qk else None,
                        postop_config="vec8",
                    )
                else:
                    _attn_layer, kv_cache, slot_mapping = cache_context
                    key_cache, value_cache = kv_cache.unbind(0)
                    sparse24_qkv_add_routed_residual_postop_cache_inplace_(
                        full_out,
                        residual_out,
                        dense_slots,
                        cos_sin_cache,
                        positions,
                        slot_mapping,
                        key_cache,
                        value_cache,
                        q_size=q_size,
                        kv_size=kv_size,
                        head_dim=128,
                        epsilon=epsilon,
                        is_neox=is_neox,
                        q_weight=q_weight if normalize_qk else None,
                        k_weight=k_weight if normalize_qk else None,
                        postop_config="vec8",
                    )
                return full_out

    qkv = _full_sparse_residual_linear_core(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        qkv_cusparselt_packed if use_qkv_cusparselt else None,
    )
    qkv = _apply_standard_qkv_postop(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_size,
        kv_size,
        epsilon,
        normalize_qk,
        is_neox,
    )
    if cache_context is not None:
        attn_layer, kv_cache, slot_mapping = cache_context
        _query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        kv_heads = kv_size // 128
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key.view(-1, kv_heads, 128),
            value.view(-1, kv_heads, 128),
            kv_cache,
            slot_mapping,
        )
    return qkv


def _qkv_routed_postop_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    use_qkv_cusparselt: bool,
    q_size: int,
    kv_size: int,
    epsilon: float,
    normalize_qk: bool,
    is_neox: bool,
) -> torch.Tensor:
    return _qkv_routed_postop_core(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        qkv_cusparselt_packed,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        use_qkv_cusparselt,
        q_size,
        kv_size,
        epsilon,
        normalize_qk,
        is_neox,
    )


def _qkv_routed_postop_cache_impl(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    use_qkv_cusparselt: bool,
    q_size: int,
    kv_size: int,
    epsilon: float,
    normalize_qk: bool,
    is_neox: bool,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
    )

    _metadata, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    if slot_mapping is None:
        qkv = _qkv_routed_postop_core(
            input_tensor,
            dense_weight,
            full_a_values,
            full_a_meta_e,
            residual_a_values,
            residual_a_meta_e,
            qkv_cusparselt_packed,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            use_qkv_cusparselt,
            q_size,
            kv_size,
            epsilon,
            normalize_qk,
            is_neox,
        )
        return qkv, qkv.new_empty((0,))
    if (
        kv_cache.dtype != torch.float16
        or kv_cache.ndim != 5
        or int(kv_cache.shape[0]) != 2
    ):
        raise RuntimeError(
            "SpecLink direct QKV cache path requires a rank-5 FP16 KV cache, "
            f"got dtype={kv_cache.dtype} shape={tuple(kv_cache.shape)}"
        )
    qkv = _qkv_routed_postop_core(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        qkv_cusparselt_packed,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        use_qkv_cusparselt,
        q_size,
        kv_size,
        epsilon,
        normalize_qk,
        is_neox,
        cache_context=(attn_layer, kv_cache, slot_mapping),
    )
    return qkv, qkv.new_empty((0,))


def _qkv_routed_postop_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    use_qkv_cusparselt: bool,
    q_size: int,
    kv_size: int,
    epsilon: float,
    normalize_qk: bool,
    is_neox: bool,
) -> torch.Tensor:
    del (
        dense_weight,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        qkv_cusparselt_packed,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        use_qkv_cusparselt,
        q_size,
        kv_size,
        epsilon,
        normalize_qk,
        is_neox,
    )
    return input_tensor.new_empty((input_tensor.shape[0], full_a_values.shape[0]))


def _qkv_routed_postop_cache_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    use_qkv_cusparselt: bool,
    q_size: int,
    kv_size: int,
    epsilon: float,
    normalize_qk: bool,
    is_neox: bool,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del layer_name
    qkv = _qkv_routed_postop_fake(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
        qkv_cusparselt_packed,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        use_qkv_cusparselt,
        q_size,
        kv_size,
        epsilon,
        normalize_qk,
        is_neox,
    )
    return qkv, input_tensor.new_empty((0,))


def _full_sparse_residual_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
) -> torch.Tensor:
    del dense_weight, full_a_meta_e, residual_a_values, residual_a_meta_e
    return input_tensor.new_empty((input_tensor.shape[0], full_a_values.shape[0]))


def _released_force_sparse_linear_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    contiguous_output: bool,
) -> torch.Tensor:
    del residual_a_values, residual_a_meta_e
    return _sparse_only_linear_fake(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        contiguous_output,
    )


def _full_sparse_residual_linear_v3_fake(
    input_tensor: torch.Tensor,
    dense_weight: torch.Tensor,
    full_a_values: torch.Tensor,
    full_a_meta_e: torch.Tensor,
    residual_a_values: torch.Tensor,
    residual_a_meta_e: torch.Tensor,
    qkv_cusparselt_packed: torch.Tensor,
    use_qkv_cusparselt: bool,
) -> torch.Tensor:
    del qkv_cusparselt_packed, use_qkv_cusparselt
    return _full_sparse_residual_linear_fake(
        input_tensor,
        dense_weight,
        full_a_values,
        full_a_meta_e,
        residual_a_values,
        residual_a_meta_e,
    )


direct_register_custom_op(
    op_name="sparse_only_linear",
    op_func=_sparse_only_linear_impl,
    fake_impl=_sparse_only_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="hybrid_gate_up_linear",
    op_func=_hybrid_gate_up_linear_impl,
    fake_impl=_hybrid_gate_up_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="force_sparse_linear",
    op_func=_force_sparse_linear_impl,
    fake_impl=_force_sparse_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="released_force_sparse_linear",
    op_func=_released_force_sparse_linear_impl,
    fake_impl=_released_force_sparse_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="mixed_dense_override_linear",
    op_func=_mixed_dense_override_linear_impl,
    fake_impl=_mixed_dense_override_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="split_dense_sparse_linear",
    op_func=_split_dense_sparse_linear_impl,
    fake_impl=_split_dense_sparse_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="full_sparse_residual_linear_v2",
    op_func=_full_sparse_residual_linear_impl,
    fake_impl=_full_sparse_residual_linear_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="full_sparse_residual_linear_v3",
    op_func=_full_sparse_residual_linear_v3_impl,
    fake_impl=_full_sparse_residual_linear_v3_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="qkv_routed_postop",
    op_func=_qkv_routed_postop_impl,
    fake_impl=_qkv_routed_postop_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="qkv_routed_postop_cache",
    op_func=_qkv_routed_postop_cache_impl,
    fake_impl=_qkv_routed_postop_cache_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
def _return_output(module: Any, output: torch.Tensor) -> Any:
    if not getattr(module, "return_bias", True):
        return output
    bias = getattr(module, "bias", None)
    output_bias = bias if getattr(module, "skip_bias_add", False) else None
    return output, output_bias


def _runtime_bias(module: Any) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    if not isinstance(bias, torch.Tensor):
        return None
    if getattr(module, "skip_bias_add", False):
        return None
    if hasattr(module, "reduce_results") and int(getattr(module, "tp_rank", 0)) > 0:
        return None
    return bias


def _finish_parallel(module: Any, output_parallel: torch.Tensor) -> torch.Tensor:
    output = output_parallel
    if (
        hasattr(module, "gather_output")
        and bool(getattr(module, "gather_output"))
        and int(getattr(module, "tp_size", 1)) > 1
    ):
        output = tensor_model_parallel_all_gather(output_parallel)
    if (
        hasattr(module, "reduce_results")
        and bool(getattr(module, "reduce_results"))
        and int(getattr(module, "tp_size", 1)) > 1
    ):
        output = tensor_model_parallel_all_reduce(output_parallel)
    return output


def _validate_module(
    module: Any,
    input_tensor: torch.Tensor,
    requested_strategy: str,
    runtime_strategy: str,
) -> None:
    cache_key = (
        requested_strategy,
        runtime_strategy,
        getattr(module, "_speclink_sparse24_backend", "cutlass"),
        input_tensor.device.type,
        input_tensor.device.index,
        input_tensor.dtype,
        int(input_tensor.shape[1]) if input_tensor.ndim == 2 else -1,
    )
    if getattr(module, "_speclink_sparse24_validation_cache", None) == cache_key:
        return
    module_name = str(
        getattr(
            module,
            "_speclink_sparse24_module_name",
            getattr(module, "prefix", module.__class__.__name__),
        )
    )
    if getattr(module, "_speclink_selective_dense_bypass", False):
        raise RuntimeError(
            f"SpecLink token-dense reached dense-bypass module {module_name} "
            "with sparse-routed rows"
        )
    if not getattr(module, "_speclink_selective_dense_enabled", False):
        raise RuntimeError(
            f"SpecLink token-dense requires a prepacked 2:4 mask for "
            f"{module_name}; none was attached"
        )
    if input_tensor.ndim != 2:
        raise RuntimeError(
            f"SpecLink token-dense linear expects a rank-2 input for {module_name}, "
            f"got {tuple(input_tensor.shape)}"
        )
    if not input_tensor.is_cuda or input_tensor.dtype != torch.float16:
        raise RuntimeError(
            "SpecLink token-dense fused sparse kernels require CUDA fp16 input "
            f"for {module_name}, got device={input_tensor.device}, "
            f"dtype={input_tensor.dtype}"
        )
    weight = getattr(module, "weight", None)
    released_dense_weight = _dense_weight_released(module)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"SpecLink token-dense cannot find weight for {module_name}")
    if released_dense_weight:
        if int(weight.numel()) != 0:
            raise RuntimeError(
                f"SpecLink released dense weight for {module_name} is not empty"
            )
    elif not weight.is_cuda or weight.dtype != torch.float16:
        raise RuntimeError(
            "SpecLink token-dense requires CUDA fp16 dense weights for "
            f"{module_name}, got device={weight.device}, dtype={weight.dtype}"
        )
    expected_in_features = _module_in_features(module)
    if int(input_tensor.shape[1]) != expected_in_features:
        raise RuntimeError(
            f"SpecLink token-dense input K mismatch for {module_name}: "
            f"input={int(input_tensor.shape[1])}, weight={expected_in_features}"
        )
    if hasattr(module, "input_is_parallel") and not bool(
        getattr(module, "input_is_parallel")
    ):
        raise RuntimeError(
            f"SpecLink token-dense does not support non-parallel row input for "
            f"{module_name}"
        )
    attached_strategy = getattr(
        module, "_speclink_sparse24_linear_strategy", requested_strategy
    )
    if attached_strategy != requested_strategy:
        raise RuntimeError(
            f"SpecLink token-dense strategy mismatch for {module_name}: "
            f"attached={attached_strategy}, requested={requested_strategy}"
        )
    sparse_backend = getattr(module, "_speclink_sparse24_backend", "cutlass")
    if sparse_backend != "cutlass":
        raise RuntimeError(
            f"SpecLink unsupported sparse backend {sparse_backend!r} for "
            f"{module_name}"
        )
    required = [
        "_speclink_sparse24_full_a_values",
        "_speclink_sparse24_full_a_meta_e",
    ]
    if runtime_strategy == "full_sparse_residual":
        required.extend(
            [
                "_speclink_sparse24_residual_a_values",
                "_speclink_sparse24_residual_a_meta_e",
            ]
        )
    missing = [name for name in required if not isinstance(getattr(module, name, None), torch.Tensor)]
    if missing:
        raise RuntimeError(
            f"SpecLink token-dense missing prepacked sparse tensors for "
            f"{module_name}: {', '.join(missing)}"
        )
    module._speclink_sparse24_validation_cache = cache_key


def _gather_rows(
    input_tensor: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    reuse_during_capture: bool = False,
) -> torch.Tensor:
    dense_rows = int(row_indices.numel())
    in_features = int(input_tensor.shape[1])
    out = None
    capturing = _cuda_graph_capturing()
    if (
        input_tensor.is_cuda
        and _reuse_sparse_buffers_enabled()
        and (not capturing or reuse_during_capture)
    ):
        key = (
            input_tensor.device.type,
            input_tensor.device.index,
            dense_rows,
            in_features,
        )
        out = _SPARSE24_GATHER_BUFFERS.get(key)
        if out is None and not capturing:
            out = torch.empty(
                (dense_rows, in_features),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            _SPARSE24_GATHER_BUFFERS[key] = out
    if out is None:
        out = torch.empty(
            (dense_rows, in_features),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
    sparse24_gather_rows_(input_tensor.contiguous(), row_indices, out)
    return out


def _add_indexed_rows_(
    full_out: torch.Tensor,
    row_add: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    full_m = int(full_out.shape[0])
    if tuple(full_out.stride()) == (1, full_m):
        return sparse24_add_indexed_rows_strided_(full_out, row_add, row_indices)
    if full_out.is_contiguous():
        if (
            _CUSTOM_CONTIGUOUS_SCATTER
            and full_out.is_cuda
            and row_add.is_contiguous()
        ):
            return sparse24_add_indexed_rows_contiguous_(
                full_out,
                row_add,
                row_indices,
            )
        full_out.index_add_(0, row_indices.to(dtype=torch.long), row_add.contiguous())
        return full_out
    raise RuntimeError(
        "SpecLink residual scatter-add requires either a contiguous output or "
        f"a CUTLASS sparse view, got stride={tuple(full_out.stride())}"
    )


def _reuse_sparse_buffers_enabled() -> bool:
    if not _REUSE_SPARSE_BUFFERS:
        return False
    if not _GRAPH_ROUTING_ENABLED or _CACHE_GRAPH_EAGER_FALLBACK:
        return True
    return _cuda_graph_capturing()


def _module_name(module: Any) -> str:
    return str(
        getattr(
            module,
            "_speclink_sparse24_module_name",
            getattr(module, "prefix", module.__class__.__name__),
        )
    )


def _dense_weight_released(module: Any) -> bool:
    return bool(
        getattr(module, "_speclink_sparse24_dense_weight_released", False)
    )


def _module_in_features(module: Any) -> int:
    stored = getattr(module, "_speclink_sparse24_in_features", None)
    if stored is not None:
        return int(stored)
    return int(module.weight.shape[1])


def _module_out_features(module: Any) -> int:
    stored = getattr(module, "_speclink_sparse24_out_features", None)
    if stored is not None:
        return int(stored)
    return int(module.weight.shape[0])


def _is_attention_input_projection(module: Any) -> bool:
    name = _module_name(module).lower()
    return name.endswith((".qkv_proj", ".q_proj", ".k_proj", ".v_proj"))


def _is_mlp_projection(module: Any) -> bool:
    name = _module_name(module).lower()
    return name.endswith(
        (".gate_up_proj", ".gate_proj", ".up_proj", ".down_proj")
    )


def _is_qkv_parallel_residual_shape(
    module_name: str,
    in_features: int,
    out_features: int,
) -> bool:
    lowered_name = module_name.lower()
    qkv = (
        lowered_name.endswith(".qkv_proj")
        and in_features == 4096
        and out_features == 6144
    )
    qwen_down = (
        lowered_name.endswith(".down_proj")
        and in_features == 12288
        and out_features == 4096
    )
    return _QKV_PARALLEL_RESIDUAL_ENABLED and (qkv or qwen_down)


def prepare_qkv_parallel_residual_streams(
    module_name: str,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> bool:
    """Create the shared side streams before any CUDA graph capture."""

    if not _is_qkv_parallel_residual_shape(
        module_name,
        in_features,
        out_features,
    ) or device.type != "cuda":
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = ("cuda", int(device_index))
    if key in _QKV_PARALLEL_RESIDUAL_STREAMS:
        return True
    if _cuda_graph_capturing():
        raise RuntimeError(
            "SpecLink QKV residual streams must be created before CUDA graph capture"
        )
    normalized_device = torch.device("cuda", device_index)
    with torch.cuda.device(normalized_device):
        _QKV_PARALLEL_RESIDUAL_STREAMS[key] = (
            torch.cuda.Stream(device=normalized_device),
            torch.cuda.Stream(device=normalized_device),
        )
    return True


def prepare_gate_up_hybrid_streams(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device.type, int(device_index))
    if key not in _GATE_UP_HYBRID_STREAMS:
        with torch.cuda.device(device_index):
            _GATE_UP_HYBRID_STREAMS[key] = (
                torch.cuda.Stream(device=device_index),
                torch.cuda.Stream(device=device_index),
            )
    return True


def _gate_up_hybrid_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream] | None:
    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    return _GATE_UP_HYBRID_STREAMS.get((device.type, int(device_index or 0)))


def _qkv_parallel_residual_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream] | None:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _QKV_PARALLEL_RESIDUAL_STREAMS.get(("cuda", int(device_index)))


def _should_parallel_qkv_residual(
    module: Any,
    input_tensor: torch.Tensor,
    dense_count: int,
) -> bool:
    rows = int(input_tensor.shape[0])
    return (
        bool(getattr(module, "_speclink_qkv_parallel_residual", False))
        and input_tensor.is_cuda
        and rows >= _QKV_PARALLEL_RESIDUAL_MIN_ROWS
        and 0 < dense_count <= _QKV_PARALLEL_RESIDUAL_MAX_DENSE_ROWS
        and dense_count < rows
    )


def _should_use_qkv_cusparselt(
    module: Any,
    input_tensor: torch.Tensor,
    dense_count: int,
) -> bool:
    packed = getattr(module, "_speclink_sparse24_qkv_cusparselt_packed", None)
    return (
        _QKV_CUSPARSELT_ENABLED
        and isinstance(packed, torch.Tensor)
        and input_tensor.is_cuda
        and input_tensor.dtype == torch.float16
        and int(input_tensor.shape[0]) >= _QKV_CUSPARSELT_MIN_ROWS
        and int(input_tensor.shape[1]) == 4096
        and _module_out_features(module) == 6144
        and 0 < dense_count < int(input_tensor.shape[0])
    )


def _skip_sparse_transpose_enabled(module: Any) -> bool:
    policy = os.getenv("SPECLINK_SPARSE24_SKIP_TRANSPOSE", "0")
    if policy in _TRUTHY:
        return True
    if policy in _FALSY:
        return False
    if policy in {"layerwise", "attention_contiguous", "non_attention"}:
        return not _is_attention_input_projection(module)
    if policy == "attention":
        return _is_attention_input_projection(module)
    if policy == "mlp":
        return _is_mlp_projection(module)
    if policy == "gate_up":
        name = _module_name(module).lower()
        return name.endswith(
            (".gate_up_proj", ".gate_proj", ".up_proj")
        )
    raise RuntimeError(
        "SPECLINK_SPARSE24_SKIP_TRANSPOSE must be 0, 1, attention, "
        "layerwise, mlp, or gate_up"
    )


def _reuse_mixed_buffers_enabled() -> bool:
    return os.getenv("SPECLINK_SPARSE24_REUSE_MIXED_BUFFERS", "0") in _TRUTHY


def _use_transposed_sparse_inputs_enabled() -> bool:
    return os.getenv("SPECLINK_SPARSE24_USE_TRANSPOSED_INPUT", "0") in _TRUTHY


def _is_transposed_sparse_input(input_tensor: torch.Tensor) -> bool:
    if input_tensor.ndim != 2:
        return False
    rows = int(input_tensor.shape[0])
    return rows % 8 == 0 and tuple(input_tensor.stride()) == (1, rows)


def _mixed_cache_max_bytes() -> int:
    mb = int(os.getenv("SPECLINK_SPARSE24_MIXED_CACHE_MAX_MB", "128"))
    return max(0, mb) * 1024 * 1024


def _cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _sparse24_m_run(rows: int) -> int:
    return (
        (rows + _SPARSE24_PAD_M_MULTIPLE - 1)
        // _SPARSE24_PAD_M_MULTIPLE
        * _SPARSE24_PAD_M_MULTIPLE
    )


def _output_ring_size() -> int:
    return _SPARSE24_OUTPUT_RING_SIZE


def _cached_sparse_buffers(
    *,
    rows: int,
    out_features: int,
    device: torch.device,
    create: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    m_run = _sparse24_m_run(rows)
    key = (device.type, device.index, m_run, out_features)
    ring_size = _output_ring_size()
    entry = _SPARSE24_OUTPUT_BUFFERS.get(key)
    if entry is None or len(entry["buffers"]) != ring_size:
        if not create:
            return None
        entry = {
            "buffers": [
                torch.empty((m_run, out_features), device=device, dtype=torch.float16)
                for _ in range(ring_size)
            ],
            "next": 0,
        }
        _SPARSE24_OUTPUT_BUFFERS[key] = entry

    buffers = entry["buffers"]
    index = int(entry["next"])
    entry["next"] = (index + 1) % len(buffers)
    out = buffers[index]

    workspace = _SPARSE24_WORKSPACE_BUFFERS.get(key)
    if workspace is None:
        workspace = torch.empty((out_features, m_run), device=device, dtype=torch.float16)
        _SPARSE24_WORKSPACE_BUFFERS[key] = workspace
    return out, workspace


def _cached_sparse_view_buffer(
    *,
    rows: int,
    out_features: int,
    device: torch.device,
) -> torch.Tensor:
    m_run = _sparse24_m_run(rows)
    key = (device.type, device.index, m_run, out_features)
    ring_size = _output_ring_size()
    entry = _SPARSE24_VIEW_OUTPUT_BUFFERS.get(key)
    if entry is None or len(entry["buffers"]) != ring_size:
        entry = {
            "buffers": [
                torch.empty_strided(
                    (m_run, out_features),
                    (1, m_run),
                    device=device,
                    dtype=torch.float16,
                )
                for _ in range(ring_size)
            ],
            "next": 0,
        }
        _SPARSE24_VIEW_OUTPUT_BUFFERS[key] = entry

    buffers = entry["buffers"]
    index = int(entry["next"])
    entry["next"] = (index + 1) % len(buffers)
    return buffers[index]


def _cached_mixed_override_buffers(
    *,
    rows: int,
    dense_rows: int,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    global _MIXED_OVERRIDE_CACHE_BYTES
    m_run = _sparse24_m_run(rows)
    dense_capacity = _sparse24_m_run(max(1, dense_rows))
    key = (
        device.type,
        device.index,
        m_run,
        in_features,
        out_features,
        dense_capacity,
    )
    ring_size = _output_ring_size()
    entry = _MIXED_OVERRIDE_BUFFERS.get(key)
    if entry is None or len(entry["outputs"]) != ring_size:
        bytes_needed = 2 * (
            dense_capacity * in_features
            + dense_capacity * out_features
            + out_features * m_run
            + ring_size * m_run * out_features
        )
        max_bytes = _mixed_cache_max_bytes()
        if max_bytes <= 0 or _MIXED_OVERRIDE_CACHE_BYTES + bytes_needed > max_bytes:
            return None
        entry = {
            "dense_x": torch.empty(
                (dense_capacity, in_features), device=device, dtype=torch.float16
            ),
            "dense_y": torch.empty(
                (dense_capacity, out_features), device=device, dtype=torch.float16
            ),
            "workspace": torch.empty(
                (out_features, m_run), device=device, dtype=torch.float16
            ),
            "outputs": [
                torch.empty((m_run, out_features), device=device, dtype=torch.float16)
                for _ in range(ring_size)
            ],
            "next": 0,
        }
        _MIXED_OVERRIDE_BUFFERS[key] = entry
        _MIXED_OVERRIDE_CACHE_BYTES += bytes_needed

    outputs = entry["outputs"]
    index = int(entry["next"])
    entry["next"] = (index + 1) % len(outputs)
    return (
        outputs[index],
        entry["dense_x"][:dense_rows],
        entry["dense_y"][:dense_rows],
        entry["workspace"],
    )


def _sparse_gemm(
    module: Any,
    input_tensor: torch.Tensor,
    *,
    residual: bool,
    out: torch.Tensor | None = None,
    contiguous_output: bool = False,
    reuse_buffers: bool = False,
    reuse_during_capture: bool = False,
) -> torch.Tensor:
    prefix = "residual" if residual else "full"
    device_config = getattr(
        module,
        f"_speclink_sparse24_{prefix}_device_config",
        getattr(module, "_speclink_sparse24_device_config", None),
    )
    workspace = None
    input_transposed = (
        _use_transposed_sparse_inputs_enabled()
        and _is_transposed_sparse_input(input_tensor)
    )
    input_arg = input_tensor if input_transposed else input_tensor.contiguous()
    capturing = _cuda_graph_capturing()
    if reuse_buffers and out is None and _reuse_sparse_buffers_enabled() and (
        not capturing or reuse_during_capture
    ):
        out_features = int(
            getattr(module, f"_speclink_sparse24_{prefix}_a_values").shape[0]
        )
        if contiguous_output:
            cached = _cached_sparse_buffers(
                rows=int(input_tensor.shape[0]),
                out_features=out_features,
                device=input_tensor.device,
                create=not capturing,
            )
            if cached is not None:
                out, workspace = cached
        else:
            out = _cached_sparse_view_buffer(
                rows=int(input_tensor.shape[0]),
                out_features=out_features,
                device=input_tensor.device,
            )
    return sparse24_cutlass_device_gemm_prepacked(
        input_arg,
        getattr(module, f"_speclink_sparse24_{prefix}_a_values"),
        getattr(module, f"_speclink_sparse24_{prefix}_a_meta_e"),
        contiguous_output=contiguous_output,
        input_transposed=input_transposed,
        out=out,
        workspace=workspace,
        device_config=device_config,
    )


def _apply_runtime_bias(module: Any, output: torch.Tensor) -> torch.Tensor:
    bias = _runtime_bias(module)
    return output if bias is None else output + bias


def _full_sparse_residual(
    module: Any,
    input_tensor: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    dense_count: int,
) -> torch.Tensor:
    paired_output = _qkv_paired_exact(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        getattr(module, "_speclink_sparse24_residual_a_values"),
        getattr(module, "_speclink_sparse24_residual_a_meta_e"),
        dense_rows,
    )
    if paired_output is not None:
        return paired_output

    heterogeneous_output = _qkv_heterogeneous_exact(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        dense_rows,
        sparse_rows,
    )
    if heterogeneous_output is not None:
        return heterogeneous_output

    if (
        _should_parallel_qkv_residual(module, input_tensor, dense_count)
        and _should_use_qkv_cusparselt(module, input_tensor, dense_count)
    ):
        qkv_output = _qkv_cusparselt_parallel_residual(
            input_tensor,
            module._speclink_sparse24_qkv_cusparselt_packed,
            getattr(module, "_speclink_sparse24_residual_a_values"),
            getattr(module, "_speclink_sparse24_residual_a_meta_e"),
            dense_rows,
            dense_count,
        )
        if qkv_output is not None:
            return qkv_output

    if _should_parallel_qkv_residual(module, input_tensor, dense_count):
        streams = _qkv_parallel_residual_streams(input_tensor.device)
        if streams is None:
            raise RuntimeError(
                "SpecLink QKV residual streams were not prepared before execution"
            )
        full_stream, residual_stream = streams
        current_stream = torch.cuda.current_stream(input_tensor.device)
        full_stream.wait_stream(current_stream)
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            output = _sparse_gemm(
                module,
                input_tensor,
                residual=False,
                contiguous_output=True,
                reuse_buffers=True,
                reuse_during_capture=True,
            )
        with torch.cuda.stream(residual_stream):
            dense_input = _gather_rows(
                input_tensor,
                dense_rows,
                reuse_during_capture=True,
            )
            residual_output = _sparse_gemm(
                module,
                dense_input,
                residual=True,
                contiguous_output=_CUSTOM_CONTIGUOUS_SCATTER,
                reuse_buffers=True,
                reuse_during_capture=True,
            )
        current_stream.wait_stream(full_stream)
        current_stream.wait_stream(residual_stream)
        _add_indexed_rows_(output, residual_output, dense_rows)
        return output

    output = _sparse_gemm(
        module,
        input_tensor,
        residual=False,
        contiguous_output=True,
        reuse_buffers=True,
    )
    if dense_count > 0:
        dense_input = _gather_rows(input_tensor, dense_rows)
        residual_output = _sparse_gemm(
            module,
            dense_input,
            residual=True,
            contiguous_output=_CUSTOM_CONTIGUOUS_SCATTER,
            reuse_buffers=True,
        )
        _add_indexed_rows_(output, residual_output, dense_rows)
    return output


def _full_sparse_pair_dense(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    output = _sparse_gemm(
        module,
        input_tensor,
        residual=False,
        contiguous_output=True,
        reuse_buffers=True,
    )
    residual_output = _sparse_gemm(
        module,
        input_tensor,
        residual=True,
        contiguous_output=True,
        reuse_buffers=True,
    )
    output.add_(residual_output)
    return output


def _full_sparse_dense_override(
    module: Any,
    input_tensor: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    dense_count: int,
) -> torch.Tensor:
    if dense_count <= 0:
        return _sparse_gemm(
            module,
            input_tensor,
            residual=False,
            contiguous_output=True,
        )
    residual_values = getattr(
        module, "_speclink_sparse24_residual_a_values", None
    )
    residual_meta = getattr(
        module, "_speclink_sparse24_residual_a_meta_e", None
    )
    if isinstance(residual_values, torch.Tensor) and isinstance(
        residual_meta, torch.Tensor
    ):
        paired_output = _qkv_paired_exact(
            input_tensor,
            module.weight,
            getattr(module, "_speclink_sparse24_full_a_values"),
            getattr(module, "_speclink_sparse24_full_a_meta_e"),
            residual_values,
            residual_meta,
            dense_rows,
        )
        if paired_output is not None:
            return paired_output
    heterogeneous_output = _qkv_heterogeneous_exact(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        dense_rows,
        sparse_rows,
    )
    if heterogeneous_output is not None:
        return heterogeneous_output
    if (
        current_verify_contiguous_dense_prefix(int(input_tensor.shape[0]))
        == dense_count
    ):
        output = _sparse_gemm(
            module,
            input_tensor,
            residual=False,
            contiguous_output=True,
            reuse_buffers=True,
        )
        torch.mm(
            input_tensor[:dense_count],
            module.weight.t(),
            out=output[:dense_count],
        )
        return output
    device_config = getattr(module, "_speclink_sparse24_device_config", None)
    out = dense_x = dense_y = workspace = None
    if _reuse_mixed_buffers_enabled() and not _cuda_graph_capturing():
        cached = _cached_mixed_override_buffers(
            rows=int(input_tensor.shape[0]),
            dense_rows=dense_count,
            in_features=int(input_tensor.shape[1]),
            out_features=_module_out_features(module),
            device=input_tensor.device,
        )
        if cached is not None:
            out, dense_x, dense_y, workspace = cached
    return sparse24_mixed_dense_override_prepacked(
        input_tensor.contiguous(),
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        dense_rows,
        out=out,
        dense_x=dense_x,
        dense_y=dense_y,
        workspace=workspace,
        device_config=device_config,
    )


def _full_sparse_dense_override_compile_safe(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    if getattr(module, "_speclink_sparse24_qkv_paired", False):
        return _full_sparse_residual_compile_safe(module, input_tensor)
    return torch.ops.speclink.mixed_dense_override_linear.default(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
    )


def _split_dense_sparse_compile_safe(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.speclink.split_dense_sparse_linear.default(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
    )


def _sparse_only_decode(module: Any, input_tensor: torch.Tensor) -> torch.Tensor:
    skip_transpose = _skip_sparse_transpose_enabled(module)
    return _sparse_gemm(
        module,
        input_tensor,
        residual=False,
        contiguous_output=not skip_transpose,
        reuse_buffers=True,
    )


def _sparse_only_decode_compile_safe(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    gate_up_hybrid = getattr(module, "_speclink_gate_up_hybrid", "none")
    if gate_up_hybrid != "none":
        return torch.ops.speclink.hybrid_gate_up_linear.default(
            input_tensor,
            module.weight,
            getattr(module, "_speclink_sparse24_full_a_values"),
            getattr(module, "_speclink_sparse24_full_a_meta_e"),
            bool(getattr(module, "_speclink_gate_up_hybrid_sparse_first")),
        )
    contiguous_output = not _skip_sparse_transpose_enabled(module)
    return torch.ops.speclink.sparse_only_linear.default(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        contiguous_output,
    )


def _force_sparse_decode_compile_safe(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    if _dense_weight_released(module):
        contiguous_output = not _skip_sparse_transpose_enabled(module)
        return torch.ops.speclink.released_force_sparse_linear.default(
            input_tensor,
            module.weight,
            getattr(module, "_speclink_sparse24_full_a_values"),
            getattr(module, "_speclink_sparse24_full_a_meta_e"),
            getattr(module, "_speclink_sparse24_residual_a_values"),
            getattr(module, "_speclink_sparse24_residual_a_meta_e"),
            contiguous_output,
        )
    contiguous_output = not _skip_sparse_transpose_enabled(module)
    return torch.ops.speclink.force_sparse_linear.default(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        contiguous_output,
    )


def _full_sparse_residual_compile_safe(
    module: Any,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    qkv_cusparselt_packed = getattr(
        module,
        "_speclink_sparse24_qkv_cusparselt_packed",
        None,
    )
    use_qkv_cusparselt = _QKV_CUSPARSELT_ENABLED and isinstance(
        qkv_cusparselt_packed,
        torch.Tensor,
    )
    if not use_qkv_cusparselt:
        qkv_cusparselt_packed = getattr(
            module,
            "_speclink_sparse24_full_a_meta_e",
        )
    return torch.ops.speclink.full_sparse_residual_linear_v3.default(
        input_tensor,
        module.weight,
        getattr(module, "_speclink_sparse24_full_a_values"),
        getattr(module, "_speclink_sparse24_full_a_meta_e"),
        getattr(module, "_speclink_sparse24_residual_a_values"),
        getattr(module, "_speclink_sparse24_residual_a_meta_e"),
        qkv_cusparselt_packed,
        use_qkv_cusparselt,
    )


def _split_dense_sparse(
    module: Any,
    input_tensor: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    dense_count: int,
) -> torch.Tensor:
    rows = int(input_tensor.shape[0])
    if dense_count <= 0:
        return _sparse_gemm(
            module,
            input_tensor,
            residual=False,
            contiguous_output=True,
            reuse_buffers=True,
        )
    contiguous_dense_count = current_verify_contiguous_dense_prefix(rows)
    if contiguous_dense_count == dense_count:
        sparse_count = rows - dense_count
        if _sparse24_m_run(sparse_count) == sparse_count:
            output = torch.empty(
                (rows, _module_out_features(module)),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            torch.mm(
                input_tensor[:dense_count],
                module.weight.t(),
                out=output[:dense_count],
            )
            _sparse_gemm(
                module,
                input_tensor[dense_count:],
                residual=False,
                out=output[dense_count:],
                contiguous_output=True,
            )
            return output
        dense_output = torch.mm(input_tensor[:dense_count], module.weight.t())
        sparse_output = _sparse_gemm(
            module,
            input_tensor[dense_count:],
            residual=False,
            contiguous_output=True,
            reuse_buffers=True,
        )
        return torch.cat((dense_output, sparse_output), dim=0)
    out_features = _module_out_features(module)
    output = torch.empty(
        (rows, out_features),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    sparse_count = rows - dense_count
    if sparse_count > 0:
        sparse_input = _gather_rows(input_tensor, sparse_rows)
        sparse_output = _sparse_gemm(
            module,
            sparse_input,
            residual=False,
            contiguous_output=True,
            reuse_buffers=True,
        )
        sparse24_copy_indexed_rows_contiguous_(
            output,
            sparse_output,
            sparse_rows,
        )

    if dense_count > 0:
        dense_input = _gather_rows(input_tensor, dense_rows)
        dense_output = torch.empty(
            (dense_count, out_features),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        torch.mm(dense_input, module.weight.t(), out=dense_output)
        sparse24_copy_indexed_rows_contiguous_(output, dense_output, dense_rows)

    return output


def _choose_strategy(
    strategy: str,
    module: Any,
    input_tensor: torch.Tensor,
    dense_count: int,
) -> str:
    if strategy != "auto":
        return strategy
    if isinstance(
        getattr(module, "_speclink_sparse24_residual_a_values", None), torch.Tensor
    ) and isinstance(
        getattr(module, "_speclink_sparse24_residual_a_meta_e", None), torch.Tensor
    ):
        return "full_sparse_residual"
    in_features = int(input_tensor.shape[1])
    out_features = _module_out_features(module)
    if dense_count <= 16:
        return "full_sparse_dense_override"
    if in_features <= out_features and (
        dense_count < 128 or out_features < 4 * in_features
    ):
        return "full_sparse_dense_override"
    return "split_dense_sparse"


def _released_dense_forward(
    module: Any,
    input_tensor: torch.Tensor,
    requested_strategy: str,
) -> Any:
    strategy = "full_sparse_residual"
    _validate_module(module, input_tensor, requested_strategy, strategy)
    output_parallel = _full_sparse_pair_dense(module, input_tensor)
    output_parallel = _apply_runtime_bias(module, output_parallel)
    output = _finish_parallel(module, output_parallel)
    return _return_output(module, output)


@torch.inference_mode()
def speclink_qkv_forward(
    module: Any,
    input_tensor: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: Any,
    attention_layer: Any | None = None,
    *,
    q_size: int,
    kv_size: int,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    epsilon: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Run exact routed QKV plus fused Q/K post-op when supported."""

    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        return None
    if (
        not (_QKV_VEC4_POSTOP_ENABLED or _QKV_FUSED_EPILOGUE_ENABLED)
        or not token_dense_enabled()
        or not getattr(module, "_speclink_selective_dense_enabled", False)
        or getattr(module, "_speclink_selective_dense_bypass", False)
        or not getattr(module, "_speclink_selective_mixed_rows", True)
        or linear_strategy() not in {"auto", "full_sparse_residual"}
        or int(getattr(module, "tp_size", 1)) != 1
        or _runtime_bias(module) is not None
        or input_tensor.ndim != 2
        or input_tensor.dtype != torch.float16
        or not input_tensor.is_cuda
        or int(input_tensor.shape[1]) != 4096
        or int(q_size + 2 * kv_size) != 6144
        or positions.ndim != 1
        or positions.dtype != torch.int64
        or not positions.is_cuda
        or int(positions.numel()) != int(input_tensor.shape[0])
        or int(getattr(rotary_emb, "head_size", -1)) != 128
        or int(getattr(rotary_emb, "rotary_dim", -1)) != 128
        or not bool(getattr(rotary_emb, "is_neox_style", False))
    ):
        return None
    required = (
        "_speclink_sparse24_full_a_values",
        "_speclink_sparse24_full_a_meta_e",
        "_speclink_sparse24_residual_a_values",
        "_speclink_sparse24_residual_a_meta_e",
    )
    if any(
        not isinstance(getattr(module, name, None), torch.Tensor)
        for name in required
    ):
        return None
    match_cache = getattr(rotary_emb, "_match_cos_sin_cache_dtype", None)
    if match_cache is None:
        return None
    cos_sin_cache = match_cache(input_tensor)
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.device != input_tensor.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[1]) != 128
        or not cos_sin_cache.is_contiguous()
    ):
        return None
    if normalize_qk and (
        q_weight.device != input_tensor.device
        or k_weight.device != input_tensor.device
        or q_weight.dtype != torch.float16
        or k_weight.dtype != torch.float16
        or tuple(q_weight.shape) != (128,)
        or tuple(k_weight.shape) != (128,)
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    ):
        return None

    full_values = getattr(module, "_speclink_sparse24_full_a_values")
    full_meta = getattr(module, "_speclink_sparse24_full_a_meta_e")
    residual_values = getattr(module, "_speclink_sparse24_residual_a_values")
    residual_meta = getattr(module, "_speclink_sparse24_residual_a_meta_e")
    qkv_cusparselt_packed = getattr(
        module, "_speclink_sparse24_qkv_cusparselt_packed", full_meta
    )
    use_qkv_cusparselt = _QKV_CUSPARSELT_ENABLED and isinstance(
        qkv_cusparselt_packed, torch.Tensor
    )
    if not use_qkv_cusparselt:
        qkv_cusparselt_packed = full_meta
    placeholder = cos_sin_cache
    op_args = (
        input_tensor,
        module.weight,
        full_values,
        full_meta,
        residual_values,
        residual_meta,
        qkv_cusparselt_packed,
        q_weight if normalize_qk else placeholder,
        k_weight if normalize_qk else placeholder,
        cos_sin_cache,
        positions.contiguous(),
        use_qkv_cusparselt,
        q_size,
        kv_size,
        float(epsilon),
        normalize_qk,
        True,
    )
    direct_cache = (
        _QKV_DIRECT_CACHE_ENABLED
        and attention_layer is not None
        and getattr(attention_layer, "kv_cache_torch_dtype", None)
        == torch.float16
        and not bool(getattr(attention_layer, "calculate_kv_scales", False))
        and getattr(attention_layer, "query_quant", None) is None
        and getattr(attention_layer, "kv_sharing_target_layer_name", None) is None
        and getattr(attention_layer, "attn_type", "") == "decoder"
        and int(getattr(attention_layer, "num_heads", -1)) == q_size // 128
        and int(getattr(attention_layer, "num_kv_heads", -1)) == kv_size // 128
        and int(getattr(attention_layer, "head_size", -1)) == 128
        and getattr(attention_layer.attn_backend, "get_name", lambda: "")()
        == "FLASH_ATTN"
        and not bool(
            getattr(
                attention_layer.attn_backend,
                "forward_includes_kv_cache_update",
                True,
            )
        )
        and isinstance(getattr(attention_layer, "layer_name", None), str)
    )
    if direct_cache:
        return torch.ops.speclink.qkv_routed_postop_cache.default(
            *op_args,
            attention_layer.layer_name,
        )
    return torch.ops.speclink.qkv_routed_postop.default(*op_args)


@torch.inference_mode()
def speclink_linear_forward(module: Any, input_tensor: torch.Tensor) -> Any:
    """Run the optional SpecLink selective-dense linear replacement."""

    # The model implementations are shared by the TLM and EAGLE3 drafter. Only
    # modules explicitly prepared by the TLM load hook may inspect routing
    # state; keeping this branch first also leaves ordinary dense modules
    # traceable by torch.compile.
    if not getattr(module, "_speclink_selective_dense_enabled", False):
        return module(input_tensor)
    if not token_dense_enabled():
        return module(input_tensor)
    if getattr(module, "_speclink_selective_dense_bypass", False):
        return module(input_tensor)
    requested_strategy = linear_strategy()
    rows = int(input_tensor.shape[0]) if input_tensor.ndim >= 1 else 0
    if (
        requested_strategy != "sparse_only_decode"
        and not bool(getattr(module, "_speclink_selective_mixed_rows", True))
    ):
        strategy = "sparse_only_decode"
        if not torch.compiler.is_compiling():
            _validate_module(module, input_tensor, requested_strategy, strategy)
        output_parallel = _force_sparse_decode_compile_safe(module, input_tensor)
        output_parallel = _apply_runtime_bias(module, output_parallel)
        output = _finish_parallel(module, output_parallel)
        return _return_output(module, output)
    if requested_strategy == "sparse_only_decode":
        strategy = "sparse_only_decode"
        if not torch.compiler.is_compiling():
            _validate_module(module, input_tensor, requested_strategy, strategy)
        output_parallel = _sparse_only_decode_compile_safe(module, input_tensor)
        output_parallel = _apply_runtime_bias(module, output_parallel)
        output = _finish_parallel(module, output_parallel)
        return _return_output(module, output)
    if requested_strategy == "full_sparse_residual":
        strategy = "full_sparse_residual"
        if not torch.compiler.is_compiling():
            _validate_module(module, input_tensor, requested_strategy, strategy)
        output_parallel = _full_sparse_residual_compile_safe(module, input_tensor)
        output_parallel = _apply_runtime_bias(module, output_parallel)
        output = _finish_parallel(module, output_parallel)
        return _return_output(module, output)
    if requested_strategy == "split_dense_sparse" and torch.compiler.is_compiling():
        output_parallel = _split_dense_sparse_compile_safe(module, input_tensor)
        output_parallel = _apply_runtime_bias(module, output_parallel)
        output = _finish_parallel(module, output_parallel)
        return _return_output(module, output)
    if (
        requested_strategy == "full_sparse_dense_override"
        and torch.compiler.is_compiling()
    ):
        output_parallel = _full_sparse_dense_override_compile_safe(
            module, input_tensor
        )
        output_parallel = _apply_runtime_bias(module, output_parallel)
        output = _finish_parallel(module, output_parallel)
        return _return_output(module, output)
    if requested_strategy == "auto" and torch.compiler.is_compiling():
        has_residual = isinstance(
            getattr(module, "_speclink_sparse24_residual_a_values", None),
            torch.Tensor,
        ) and isinstance(
            getattr(module, "_speclink_sparse24_residual_a_meta_e", None),
            torch.Tensor,
        )
        output_parallel = (
            _full_sparse_residual_compile_safe(module, input_tensor)
            if has_residual
            else _full_sparse_dense_override_compile_safe(
                module, input_tensor
            )
        )
        output_parallel = _apply_runtime_bias(module, output_parallel)
        output = _finish_parallel(module, output_parallel)
        return _return_output(module, output)

    summary = current_verify_dense_row_summary(rows, input_tensor.device)
    if summary is None:
        if _dense_weight_released(module):
            return _released_dense_forward(
                module,
                input_tensor,
                requested_strategy,
            )
        return module(input_tensor)
    _row_is_dense, dense_count, dense_rows, sparse_rows = summary
    if not fast_plan_enabled():
        if dense_count:
            dense_rows = dense_rows[dense_rows < rows].contiguous()
            dense_count = int(dense_rows.numel())
        if int(sparse_rows.numel()):
            sparse_rows = sparse_rows[sparse_rows < rows].contiguous()
    if dense_count == rows:
        if _dense_weight_released(module):
            return _released_dense_forward(
                module,
                input_tensor,
                requested_strategy,
            )
        return module(input_tensor)
    strategy = _choose_strategy(
        requested_strategy, module, input_tensor, dense_count
    )
    _validate_module(module, input_tensor, requested_strategy, strategy)
    if strategy == "sparse_only_decode":
        output_parallel = _sparse_only_decode(module, input_tensor)
    elif strategy == "full_sparse_residual":
        output_parallel = _full_sparse_residual(
            module, input_tensor, dense_rows, sparse_rows, dense_count
        )
    elif strategy == "full_sparse_dense_override":
        output_parallel = _full_sparse_dense_override(
            module, input_tensor, dense_rows, sparse_rows, dense_count
        )
    elif strategy == "split_dense_sparse":
        output_parallel = _split_dense_sparse(
            module, input_tensor, dense_rows, sparse_rows, dense_count
        )
    else:
        raise RuntimeError(f"unsupported SpecLink token-dense linear strategy: {strategy}")

    output_parallel = _apply_runtime_bias(module, output_parallel)
    output = _finish_parallel(module, output_parallel)
    return _return_output(module, output)
