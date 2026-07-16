# SPDX-License-Identifier: Apache-2.0
"""MLP dispatch for SpecLink selective structured 2:4 linears."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.library import Library

from vllm.speclink_breakdown import verify_detail_enabled, verify_timer
from vllm.speclink_linear import (
    _log_kernel_dispatch,
    speclink_linear_forward,
)
from vllm.speclink_kernel import (
    dense_cutlass_weight_t_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_add_indexed_rows_strided_,
    sparse24_add_indexed_rows_transposed_to_contiguous_,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_device_splitk_gemm_prepacked,
    sparse24_cutlass_device_strided_input_gemm_prepacked,
    sparse24_cutlass_full_sparse_dense_override_linear_prepacked,
    sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
    sparse24_cutlass_heterogeneous_linear_prepacked,
    sparse24_cutlass_heterogeneous_swiglu_prepacked,
    sparse24_cutlass_indexed_output_gemm_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_paired_gather_residual_prepacked,
    sparse24_cutlass_paired_fused_routed_swiglu_prepacked,
    sparse24_cutlass_paired_inplace_residual_prepacked,
    sparse24_cutlass_paired_persistent_gemm_prepacked,
    sparse24_cutlass_paired_persistent_routed_swiglu_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_,
    sparse24_gather_rows_strided_,
    sparse24_merge_rows_,
    sparse24_partition_rows_,
    sparse24_routed_swiglu_correction_,
    sparse24_routed_swiglu_correction_gather_,
    sparse24_silu_and_mul_transposed,
    sparse24_silu_and_mul_transposed_to_contiguous,
    sparse24_transpose_add_routed_residual,
    sparse24_transpose_add_routed_splitk_residual,
    sparse24_transpose_input_to_strided,
    sparse24_transpose_output_contiguous,
)
from vllm.speclink_token_dense import (
    current_verify_dense_slots,
    current_verify_prefill_row_summary,
    current_verify_dense_row_summary,
    enabled as token_dense_enabled,
    linear_strategy,
    mlp_strategy,
)
from vllm.utils.torch_utils import direct_register_custom_op


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}
_FUSED_BATCH_MLP_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_FUSED_BATCH_MLP", "0") in _TRUTHY
)
_INLINE_SWIGLU_MLP_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_INLINE_SWIGLU_MLP", "0") in _TRUTHY
)
_ROUTED_SWIGLU_MLP_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_ROUTED_SWIGLU_MLP", "0") in _TRUTHY
)
_DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_DIRECT_GATE_RESIDUAL_EPILOGUE", "1")
    in _TRUTHY
)
_GATE_DOWN_GATHER_EPILOGUE_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_GATE_DOWN_GATHER_EPILOGUE", "0")
    in _TRUTHY
)
_CONTIGUOUS_DOWN_INPUT_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_CONTIGUOUS_DOWN_INPUT", "1") in _TRUTHY
)
_UNINITIALIZED_ROUTED_WORKSPACE_ENABLED = (
    os.getenv(
        "SPECLINK_TOKEN_DENSE_UNINITIALIZED_ROUTED_WORKSPACE",
        "1",
    )
    in _TRUTHY
)
_PAIRED_PERSISTENT_GATE_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_PERSISTENT_GATE", "1")
    in _TRUTHY
)
_PAIRED_GATE_SHAPE_TUNING_ENABLED = (
    os.getenv(
        "SPECLINK_TOKEN_DENSE_PAIRED_GATE_SHAPE_TUNING",
        os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_GATE_WORKER_TUNING", "0"),
    )
    in _TRUTHY
)
_PAIRED_FUSED_GATE_EPILOGUE_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_FUSED_GATE_EPILOGUE", "0")
    in _TRUTHY
)
_PAIRED_PERSISTENT_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_PERSISTENT_DOWN", "0")
    in _TRUTHY
)
_PAIRED_GATHER_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_GATHER_DOWN", "0") in _TRUTHY
)
_PAIRED_INPLACE_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PAIRED_INPLACE_DOWN", "0") in _TRUTHY
)
_PARALLEL_SPLITK_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_PARALLEL_SPLITK_DOWN", "0") in _TRUTHY
)
_SINGLE_LAUNCH_SPLITK_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_SINGLE_LAUNCH_SPLITK_DOWN", "0")
    in _TRUTHY
)
_TILED_INDEXED_SPLITK_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_TILED_INDEXED_SPLITK_DOWN", "0")
    in _TRUTHY
)
_SPARSE_GATE_DENSE_DOWN_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_SPARSE_GATE_DENSE_DOWN", "0")
    in _TRUTHY
)
_FULL_SPARSE_OVERRIDE_MLP_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_FULL_SPARSE_OVERRIDE_MLP", "0")
    in _TRUTHY
)
_FULL_SPARSE_OVERRIDE_MLP_MIN_ROWS = int(
    os.getenv("SPECLINK_TOKEN_DENSE_FULL_SPARSE_OVERRIDE_MLP_MIN_ROWS", "224")
)
_SPARSE_GATE_SWIGLU_FP16_ENABLED = os.getenv(
    "SPECLINK_SPARSE24_ACCUMULATOR", "fp32"
) in {"fp16", "fp16_gate", "fp16_gate_down", "fp16_qkv_gate"}
_INDEXED_DOWN_EPILOGUE_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_INDEXED_DOWN_EPILOGUE", "0") in _TRUTHY
)
_TRANSPOSED_RESIDUAL_MLP_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_TRANSPOSED_RESIDUAL_MLP", "1") in _TRUTHY
)
_CUTLASS_DOWN_FP16_ENABLED = (
    os.getenv("SPECLINK_TOKEN_DENSE_CUTLASS_DOWN_FP16", "0") in _TRUTHY
)
_SPECLINK_OP_LIB = Library("speclink", "FRAGMENT")
_MIXED_MLP_STREAMS: dict[
    tuple[str, int], tuple[torch.cuda.Stream, torch.cuda.Stream]
] = {}
_PARALLEL_SPLITK_DOWN_STREAMS: dict[
    tuple[str, int], tuple[torch.cuda.Stream, ...]
] = {}
_SINGLE_LAUNCH_SPLITK_DOWN_WORKSPACES: dict[
    tuple[str, int | None, int, int], torch.Tensor
] = {}
_ALL_DENSE_ROW_IDS: dict[tuple[str, int | None, int], torch.Tensor] = {}
_GATE_UP_DENSE_WEIGHT_ROWS: dict[
    tuple[str, int | None, int], torch.Tensor
] = {}
_PAIRED_FUSED_GATE_COUNTERS: dict[
    tuple[str, int | None, int, int, int, int], torch.Tensor
] = {}
_PAIRED_INPLACE_DOWN_COUNTERS: dict[
    tuple[str, int | None, int, int, int, int, str], torch.Tensor
] = {}
_ROUTED_SWIGLU_CONFIG = "256x64x64_s3_sw4"
_SPECLINK_VERIFY_DETAIL = verify_detail_enabled()


def _paired_fused_gate_counters(
    device: torch.device,
    run_rows: int,
    dense_count: int,
    gate_up_size: int,
    packed_weight_ptr: int,
) -> torch.Tensor | None:
    key = (
        device.type,
        device.index,
        run_rows,
        dense_count,
        gate_up_size,
        packed_weight_ptr,
    )
    counters = _PAIRED_FUSED_GATE_COUNTERS.get(key)
    if counters is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        counters = torch.zeros(
            gate_up_size // 256,
            device=device,
            dtype=torch.int32,
        )
        _PAIRED_FUSED_GATE_COUNTERS[key] = counters
    return counters


def _paired_inplace_down_counters(
    device: torch.device,
    run_rows: int,
    dense_count: int,
    hidden_size: int,
    packed_weight_ptr: int,
    config: str,
) -> torch.Tensor | None:
    key = (
        device.type,
        device.index,
        run_rows,
        dense_count,
        hidden_size,
        packed_weight_ptr,
        config,
    )
    counters = _PAIRED_INPLACE_DOWN_COUNTERS.get(key)
    if counters is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        feature_columns = 128 if config.startswith("128x") else 256
        counters = torch.zeros(
            (hidden_size + feature_columns - 1) // feature_columns,
            device=device,
            dtype=torch.int32,
        )
        _PAIRED_INPLACE_DOWN_COUNTERS[key] = counters
    return counters


_CUTLASS_DOWN_FP16_CONFIGS = {
    112: "64x64x64_s4",
    144: "64x128x64_s3",
    176: "64x128x64_s3",
    224: "64x128x64_s3",
    288: "64x128x64_s3",
    352: "128x128x64_s3",
    448: "128x128x64_s3",
    576: "128x128x64_s3",
    704: "128x128x64_s3",
}


def _cutlass_down_fp16_config(rows: int) -> str | None:
    return _CUTLASS_DOWN_FP16_CONFIGS.get(rows)


def _dense_down_fp16_impl(
    hidden: torch.Tensor, weight_t: torch.Tensor
) -> torch.Tensor:
    config = _cutlass_down_fp16_config(int(hidden.shape[0]))
    if config is None:
        return torch.mm(hidden, weight_t.t())
    return dense_cutlass_weight_t_gemm(
        hidden,
        weight_t,
        accumulator="fp16",
        device_config=config,
    )


def _dense_down_fp16_fake(
    hidden: torch.Tensor, weight_t: torch.Tensor
) -> torch.Tensor:
    return hidden.new_empty((hidden.shape[0], weight_t.shape[0]))


direct_register_custom_op(
    op_name="dense_down_fp16",
    op_func=_dense_down_fp16_impl,
    fake_impl=_dense_down_fp16_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)


def _use_paired_persistent_gate(
    run_rows: int,
    dense_count: int,
    gate_up_size: int,
) -> bool:
    if not _PAIRED_PERSISTENT_GATE_ENABLED:
        return False
    if gate_up_size == 24576:
        return (
            run_rows == 112
            or run_rows >= 192
            or (run_rows == 144 and dense_count >= 32)
            or (run_rows == 176 and dense_count >= 36)
        )
    if gate_up_size == 28672:
        return run_rows >= 112
    return False


def _paired_gate_worker_blocks(
    run_rows: int,
    dense_count: int,
    gate_up_size: int,
) -> int:
    if not _PAIRED_GATE_SHAPE_TUNING_ENABLED:
        return 0
    dense_cap = {24576: 32, 28672: 64}.get(gate_up_size)
    if dense_cap is None or dense_count > dense_cap:
        return 0
    if gate_up_size == 24576 and run_rows in {144, 176}:
        return 128
    if (run_rows, gate_up_size) == (288, 24576):
        return 144
    if (run_rows, gate_up_size) == (224, 28672):
        return 144
    return 0


def _paired_gate_schedule(
    run_rows: int,
    dense_count: int,
    gate_up_size: int,
) -> str:
    if not _PAIRED_GATE_SHAPE_TUNING_ENABLED:
        return "interleaved"
    dense_cap = {24576: 32, 28672: 64}.get(gate_up_size)
    tuned_rows = {144, 176, 224, 288, 352, 448, 576, 704}
    if (
        dense_cap is not None
        and dense_count <= dense_cap
        and run_rows in tuned_rows
    ):
        return "partitioned"
    return "interleaved"


def _use_paired_persistent_down(
    run_rows: int,
    dense_count: int,
    intermediate_size: int,
) -> bool:
    if not _PAIRED_PERSISTENT_DOWN_ENABLED:
        return False
    key = (run_rows, dense_count)
    if intermediate_size == 12288:
        return key in {(352, 72), (448, 112), (704, 144)}
    if intermediate_size == 14336:
        return key in {
            (288, 64),
            (352, 72),
            (448, 112),
            (704, 144),
        }
    return False


def _paired_gather_down_config(
    run_rows: int,
    dense_count: int,
    intermediate_size: int,
) -> str | None:
    """Select only paired-gather Down shapes with measured net benefit."""

    if not _PAIRED_GATHER_DOWN_ENABLED:
        return None
    key = (run_rows, dense_count)
    if intermediate_size == 12288:
        if key in {(144, 16), (288, 32)}:
            return "256x32_full_256x32_residual_contiguous"
        if key in {(352, 32), (448, 32), (576, 32)}:
            return "256x64_full_256x64_residual_contiguous"
    if intermediate_size == 14336:
        if key == (144, 40):
            return "256x32_full_256x32_residual_contiguous"
        if key in {(352, 64), (448, 64), (576, 64)}:
            return "256x64_full_256x64_residual_contiguous"
    return None


def _paired_inplace_down_config(
    run_rows: int,
    dense_count: int,
    intermediate_size: int,
) -> str | None:
    """Select the best measured exact mixed-Down kernel for verifier waves."""

    if not _PAIRED_INPLACE_DOWN_ENABLED:
        return None
    if not 112 <= run_rows <= 704:
        return None
    dense_cap = {12288: 32, 14336: 64}.get(intermediate_size)
    if dense_cap is None or dense_count > dense_cap:
        return None
    # Exhaustive CUDA Graph measurements for every active request count from
    # 16 through 64 and K={6,8,10} reduce to these tile boundaries. They cover
    # tail waves that previously fell back to paired GEMM plus indexed add.
    if run_rows <= 126 and dense_count <= 32:
        return "128x32_full_128x32_residual_inplace"
    if dense_count <= 32:
        if run_rows <= 288 or 577 <= run_rows <= 640:
            return "256x32_full_256x32_residual_inplace"
        return "256x64_full_256x32_residual_inplace"
    if run_rows <= 256 or 513 <= run_rows <= 640:
        return "256x32_full_256x32_residual_inplace"
    return "256x64_full_256x32_residual_inplace"


def _parallel_splitk_down_slices(
    run_rows: int,
    dense_count: int,
    intermediate_size: int,
) -> int:
    """Select only zero-copy split-K Down shapes with measured net benefit."""

    if not _PARALLEL_SPLITK_DOWN_ENABLED:
        return 0
    key = (run_rows, dense_count)
    if intermediate_size == 12288:
        return {
            (112, 16): 4,
            (144, 16): 4,
            (288, 32): 4,
        }.get(key, 0)
    if intermediate_size == 14336:
        return {
            (112, 30): 4,
            (144, 40): 4,
            (288, 64): 4,
            (576, 64): 4,
        }.get(key, 0)
    return 0


def _single_launch_splitk_down_slices(
    run_rows: int,
    dense_run: int,
    intermediate_size: int,
) -> int:
    """Select serial split-K only for exact positive verifier shapes."""

    if not _SINGLE_LAUNCH_SPLITK_DOWN_ENABLED:
        return 0
    key = (run_rows, dense_run)
    if intermediate_size == 12288:
        return {(288, 32): 4}.get(key, 0)
    if intermediate_size == 14336:
        return {(288, 64): 4}.get(key, 0)
    return 0


def _tiled_indexed_splitk_down_config(
    run_rows: int,
    dense_run: int,
    intermediate_size: int,
) -> tuple[int, str] | None:
    """Select measured-positive serial split-K plus dense-only epilogues."""

    if not _TILED_INDEXED_SPLITK_DOWN_ENABLED:
        return None
    key = (run_rows, dense_run)
    if intermediate_size == 12288:
        split_k = {
            (144, 16): 4,
            (176, 24): 2,
            (224, 32): 2,
            (288, 32): 4,
            (352, 32): 2,
            (448, 32): 2,
            (576, 32): 2,
            (704, 32): 2,
        }.get(key)
    elif intermediate_size == 14336:
        split_k = {
            (112, 32): 4,
            (144, 40): 8,
            (176, 56): 4,
            (288, 64): 8,
            (352, 64): 2,
            (448, 64): 4,
            (576, 64): 2,
            (704, 64): 2,
        }.get(key)
    else:
        split_k = None
    if split_k is None:
        return None
    if run_rows <= 144:
        full_config = "128x32x64_s4_sw4"
    elif run_rows <= 288:
        full_config = "128x64x64_s5"
    elif run_rows <= 576:
        full_config = "256x64x64_s3"
    else:
        full_config = "128x64x64_s5"
    return split_k, full_config


def _single_launch_splitk_down_workspace(
    device: torch.device,
    dense_run: int,
    output_size: int,
) -> torch.Tensor:
    key = (device.type, device.index, dense_run, output_size)
    workspace = _SINGLE_LAUNCH_SPLITK_DOWN_WORKSPACES.get(key)
    if workspace is None:
        workspace_ints = ((output_size + 255) // 256) * (
            (dense_run + 31) // 32
        )
        workspace = torch.zeros(
            workspace_ints,
            device=device,
            dtype=torch.int32,
        )
        _SINGLE_LAUNCH_SPLITK_DOWN_WORKSPACES[key] = workspace
    return workspace


def _sparse_gate_dense_down_config(
    run_rows: int,
    gate_up_size: int,
) -> str:
    if (
        _SPARSE_GATE_SWIGLU_FP16_ENABLED
        and run_rows >= 112
        and gate_up_size in {24576, 28672}
    ):
        return "256x64x64_s3_sw4_f16"
    if (
        72 <= run_rows <= 96
        and gate_up_size in {24576, 28672}
    ):
        return "256x32x64_s3_sw4"
    return "auto"


def prepare_mixed_mlp_streams(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    key = (device.type, device.index if device.index is not None else 0)
    if key not in _MIXED_MLP_STREAMS:
        _MIXED_MLP_STREAMS[key] = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
        )
    if _PARALLEL_SPLITK_DOWN_ENABLED and key not in _PARALLEL_SPLITK_DOWN_STREAMS:
        _PARALLEL_SPLITK_DOWN_STREAMS[key] = tuple(
            torch.cuda.Stream(device=device) for _ in range(4)
        )
    return True


def _mixed_mlp_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream] | None:
    key = (device.type, device.index if device.index is not None else 0)
    return _MIXED_MLP_STREAMS.get(key)


def _parallel_splitk_down_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, ...] | None:
    key = (device.type, device.index if device.index is not None else 0)
    return _PARALLEL_SPLITK_DOWN_STREAMS.get(key)


def _splitk_down_weight_views(
    values: torch.Tensor,
    metadata: torch.Tensor,
    split_k_slices: int,
) -> tuple[tuple[int, int, torch.Tensor, torch.Tensor], ...]:
    output_size, sparse_k = map(int, values.shape)
    full_k = sparse_k * 2
    if full_k % split_k_slices:
        raise RuntimeError(
            f"Down K={full_k} is not divisible by split={split_k_slices}"
        )
    slice_k = full_k // split_k_slices
    slices: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for split in range(split_k_slices):
        start = split * slice_k
        end = start + slice_k
        values_view = values[:, start // 2 : end // 2]
        metadata_start = (start // 32) * output_size * 2
        metadata_end = metadata_start + (slice_k // 32) * output_size * 2
        slices.append(
            (
                start,
                end,
                values_view,
                metadata[metadata_start:metadata_end],
            )
        )
    return tuple(slices)


def _all_dense_row_ids(rows: int, device: torch.device) -> torch.Tensor:
    key = (device.type, device.index, rows)
    row_ids = _ALL_DENSE_ROW_IDS.get(key)
    if row_ids is None:
        row_ids = torch.arange(rows, device=device, dtype=torch.int32)
        _ALL_DENSE_ROW_IDS[key] = row_ids
    return row_ids


def _gate_up_dense_weight_rows(
    out_features: int, device: torch.device
) -> torch.Tensor | None:
    """Map interleaved 64-channel Gate/Up tiles to dense weight rows."""

    intermediate = out_features // 2
    pair_channels = 64
    if out_features % 2 or intermediate % pair_channels:
        return None
    key = (device.type, device.index, out_features)
    row_ids = _GATE_UP_DENSE_WEIGHT_ROWS.get(key)
    if row_ids is None:
        gate_rows = torch.arange(
            intermediate, device=device, dtype=torch.int32
        ).reshape(-1, pair_channels)
        row_ids = torch.cat(
            (gate_rows, gate_rows + intermediate), dim=1
        ).flatten().contiguous()
        _GATE_UP_DENSE_WEIGHT_ROWS[key] = row_ids
    return row_ids


def _silu_and_mul_contiguous(gate_up: torch.Tensor) -> torch.Tensor:
    hidden = torch.empty(
        (gate_up.shape[0], gate_up.shape[1] // 2),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    torch.ops._C.silu_and_mul(hidden, gate_up)
    return hidden


def _sparse_gate_up_hidden(
    x: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
) -> torch.Tensor:
    if _INLINE_SWIGLU_MLP_ENABLED or _ROUTED_SWIGLU_MLP_ENABLED:
        return sparse24_cutlass_gate_up_swiglu_prepacked(
            x,
            gate_up_values,
            gate_up_meta,
            config="auto",
            output_transposed=True,
        )
    gate_up = sparse24_cutlass_device_gemm_prepacked(
        x,
        gate_up_values,
        gate_up_meta,
        contiguous_output=False,
    )
    return sparse24_silu_and_mul_transposed(gate_up)


def _sparse_gate_swiglu_impl(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
) -> torch.Tensor:
    """Fuse all-sparse Gate/Up + SwiGLU and keep Down outside the op."""

    rows = int(x.shape[0])
    summary = current_verify_dense_row_summary(rows, x.device)
    if summary is None:
        gate_up = torch.mm(x, gate_up_weight.t())
        return _silu_and_mul_contiguous(gate_up)

    def sparse_hidden(value: torch.Tensor) -> torch.Tensor:
        value_rows = int(value.shape[0])
        run_rows = (value_rows + 7) // 8 * 8
        if run_rows == value_rows:
            run_x = value.contiguous()
        else:
            run_x = torch.zeros(
                (run_rows, value.shape[1]),
                device=value.device,
                dtype=value.dtype,
            )
            run_x[:value_rows].copy_(value)
        return sparse24_cutlass_gate_up_swiglu_prepacked(
            run_x,
            gate_up_values,
            gate_up_meta,
            config=_sparse_gate_dense_down_config(
                run_rows,
                int(gate_up_weight.shape[0]),
            ),
            output_transposed=False,
        )[:value_rows]

    prefill_summary = current_verify_prefill_row_summary(rows)
    if prefill_summary is None:
        return sparse_hidden(x)

    prefill_count, prefill_rows, decode_rows, layout = prefill_summary
    decode_count = rows - prefill_count
    if decode_count == 0:
        gate_up = torch.mm(x, gate_up_weight.t())
        return _silu_and_mul_contiguous(gate_up)

    streams = _mixed_mlp_streams(x.device)
    if layout in {"prefix", "suffix"}:
        x = x.contiguous()
        if layout == "prefix":
            dense_x = x[:prefill_count]
            sparse_x = x[prefill_count:]
        else:
            sparse_x = x[:decode_count]
            dense_x = x[decode_count:]
        if streams is None:
            dense_gate_up = torch.mm(dense_x, gate_up_weight.t())
            dense_hidden = _silu_and_mul_contiguous(dense_gate_up)
            sparse_output = sparse_hidden(sparse_x)
        else:
            dense_stream, sparse_stream = streams
            current_stream = torch.cuda.current_stream(x.device)
            dense_stream.wait_stream(current_stream)
            sparse_stream.wait_stream(current_stream)
            with torch.cuda.stream(dense_stream):
                dense_gate_up = torch.mm(dense_x, gate_up_weight.t())
                dense_hidden = _silu_and_mul_contiguous(dense_gate_up)
            with torch.cuda.stream(sparse_stream):
                sparse_output = sparse_hidden(sparse_x)
            current_stream.wait_stream(dense_stream)
            current_stream.wait_stream(sparse_stream)
        return torch.cat(
            (dense_hidden, sparse_output)
            if layout == "prefix"
            else (sparse_output, dense_hidden)
        )

    x = x.contiguous()
    dense_x = torch.empty(
        (prefill_count, x.shape[1]), device=x.device, dtype=x.dtype
    )
    sparse_x = torch.empty(
        (decode_count, x.shape[1]), device=x.device, dtype=x.dtype
    )
    sparse24_partition_rows_(
        x,
        prefill_rows,
        decode_rows,
        dense_x,
        sparse_x,
    )
    if streams is None:
        dense_gate_up = torch.mm(dense_x, gate_up_weight.t())
        dense_hidden = _silu_and_mul_contiguous(dense_gate_up)
        sparse_output = sparse_hidden(sparse_x)
    else:
        dense_stream, sparse_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        dense_stream.wait_stream(current_stream)
        sparse_stream.wait_stream(current_stream)
        with torch.cuda.stream(dense_stream):
            dense_gate_up = torch.mm(dense_x, gate_up_weight.t())
            dense_hidden = _silu_and_mul_contiguous(dense_gate_up)
        with torch.cuda.stream(sparse_stream):
            sparse_output = sparse_hidden(sparse_x)
        current_stream.wait_stream(dense_stream)
        current_stream.wait_stream(sparse_stream)
    output = torch.empty(
        (rows, gate_up_weight.shape[0] // 2),
        device=x.device,
        dtype=x.dtype,
    )
    sparse24_merge_rows_(
        output,
        dense_hidden,
        sparse_output,
        prefill_rows,
        decode_rows,
    )
    return output


def _sparse_gate_swiglu_fake(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
) -> torch.Tensor:
    del gate_up_values, gate_up_meta
    return x.new_empty((x.shape[0], gate_up_weight.shape[0] // 2))


def _sparse_mixed_down(
    hidden: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    sparse_rows: torch.Tensor,
    output_rows: int,
    output: torch.Tensor | None,
) -> torch.Tensor | None:
    if _INDEXED_DOWN_EPILOGUE_ENABLED:
        assert output is not None
        sparse24_cutlass_indexed_output_gemm_prepacked(
            hidden,
            down_values,
            down_meta,
            sparse_rows,
            output_rows=output_rows,
            out=output,
            config="auto",
            input_transposed=True,
        )
        return None
    return sparse24_cutlass_device_gemm_prepacked(
        hidden,
        down_values,
        down_meta,
        contiguous_output=True,
        input_transposed=True,
    )


def _is_cutlass_transposed_gate_up(gate_up: torch.Tensor) -> bool:
    if (
        not gate_up.is_cuda
        or gate_up.dtype != torch.float16
        or gate_up.ndim != 2
    ):
        return False
    rows = int(gate_up.shape[0])
    return (
        int(gate_up.stride(0)) == 1
        and int(gate_up.stride(1)) >= rows
    )


def _transposed_silu_and_mul_impl(gate_up: torch.Tensor) -> torch.Tensor:
    return sparse24_silu_and_mul_transposed(gate_up)


def _transposed_silu_and_mul_fake(gate_up: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        (gate_up.shape[0], gate_up.shape[1] // 2),
        (1, gate_up.stride(1)),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )


def _transposed_silu_and_mul_contiguous_impl(
    gate_up: torch.Tensor,
) -> torch.Tensor:
    return sparse24_silu_and_mul_transposed_to_contiguous(gate_up)


def _transposed_silu_and_mul_contiguous_fake(
    gate_up: torch.Tensor,
) -> torch.Tensor:
    return gate_up.new_empty(
        (gate_up.shape[0], gate_up.shape[1] // 2),
    )


direct_register_custom_op(
    op_name="transposed_silu_and_mul",
    op_func=_transposed_silu_and_mul_impl,
    fake_impl=_transposed_silu_and_mul_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="transposed_silu_and_mul_contiguous",
    op_func=_transposed_silu_and_mul_contiguous_impl,
    fake_impl=_transposed_silu_and_mul_contiguous_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="sparse_gate_swiglu",
    op_func=_sparse_gate_swiglu_impl,
    fake_impl=_sparse_gate_swiglu_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)


def _apply_gate_up_activation(mlp: Any, gate_up: torch.Tensor) -> torch.Tensor:
    if _is_cutlass_transposed_gate_up(gate_up):
        down_proj = mlp.down_proj
        down_uses_sparse_layout = getattr(
            down_proj,
            "_speclink_selective_dense_enabled",
            False,
        ) and not getattr(
            down_proj,
            "_speclink_selective_dense_bypass",
            False,
        )
        if down_uses_sparse_layout:
            return torch.ops.speclink.transposed_silu_and_mul.default(gate_up)
        return torch.ops.speclink.transposed_silu_and_mul_contiguous.default(
            gate_up
        )
    return mlp.act_fn(gate_up)


def _fused_override_backend(
    rows: int,
    dense_count: int,
    gate_up_size: int,
) -> str | None:
    """Choose the exact mixed MLP backend from active-shape profiles."""

    if rows % 8:
        return None
    if gate_up_size == 24576:
        if rows < 256:
            return "heterogeneous"
        if dense_count <= 64:
            return "persistent_gate"
        return "parallel_gate"
    if gate_up_size == 28672:
        if rows < 272:
            return "heterogeneous"
        if rows < 384:
            return "parallel_gate"
        if rows < 480:
            return "persistent_gate"
        return "parallel_gate"
    return None


def _fused_override_mlp(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
) -> torch.Tensor | None:
    """Select the profiled exact mixed-row Gate/SwiGLU/Down pipeline."""

    rows, hidden_size = map(int, x.shape)
    gate_up_size = int(gate_up_weight.shape[0])
    intermediate_size = gate_up_size // 2
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    prefill_summary = current_verify_prefill_row_summary(rows)
    if (
        not _FULL_SPARSE_OVERRIDE_MLP_ENABLED
        or linear_strategy() != "full_sparse_dense_override"
        or rows < _FULL_SPARSE_OVERRIDE_MLP_MIN_ROWS
        or dense_count <= 0
        or dense_count >= rows
        or dense_count + sparse_count != rows
        or prefill_summary is not None
        or hidden_size != 4096
        or gate_up_size not in {24576, 28672}
        or tuple(gate_up_weight.shape) != (gate_up_size, hidden_size)
        or tuple(down_weight.shape) != (hidden_size, intermediate_size)
        or not gate_up_weight.is_contiguous()
        or not down_weight.is_contiguous()
    ):
        return None
    backend = _fused_override_backend(rows, dense_count, gate_up_size)
    if backend is None:
        return None
    dense_slots = current_verify_dense_slots(rows, x.device)
    dense_weight_rows = _gate_up_dense_weight_rows(
        gate_up_size, x.device
    )
    if dense_slots is None or dense_weight_rows is None:
        return None

    _log_kernel_dispatch(
        f"mlp_{backend}", rows, dense_count, hidden_size
    )
    if backend == "heterogeneous":
        hidden = sparse24_cutlass_heterogeneous_swiglu_prepacked(
            x,
            gate_up_values,
            gate_up_meta,
            gate_up_weight,
            dense_weight_rows,
            dense_rows,
            sparse_rows,
            config="256x32x64_s3_sw4_f16",
        )
        return sparse24_cutlass_heterogeneous_linear_prepacked(
            hidden,
            down_values,
            down_meta,
            down_weight,
            dense_rows,
            sparse_rows,
            config="auto",
        )

    if backend == "persistent_gate":
        hidden = sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked(
            x,
            gate_up_values,
            gate_up_meta,
            gate_up_weight,
            dense_weight_rows,
            dense_rows,
            dense_slots,
            config="256x64_sparse_128x64_dense_f16",
        )
        return sparse24_cutlass_heterogeneous_linear_prepacked(
            hidden,
            down_values,
            down_meta,
            down_weight,
            dense_rows,
            sparse_rows,
            config="auto",
        )

    hidden = None
    streams = _mixed_mlp_streams(x.device)
    if streams is not None and rows % 8 == 0:
        sparse_stream, dense_stream = streams
        sparse_hidden = torch.empty(
            (rows, intermediate_size), device=x.device, dtype=x.dtype
        )
        dense_x = torch.empty(
            (dense_count, hidden_size), device=x.device, dtype=x.dtype
        )
        dense_gate_up = torch.empty(
            (dense_count, gate_up_size), device=x.device, dtype=x.dtype
        )
        dense_hidden = torch.empty(
            (dense_count, intermediate_size), device=x.device, dtype=x.dtype
        )
        current_stream = torch.cuda.current_stream(x.device)
        sparse_stream.wait_stream(current_stream)
        dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(sparse_stream):
            sparse24_cutlass_gate_up_swiglu_prepacked(
                x,
                gate_up_values,
                gate_up_meta,
                out=sparse_hidden,
                config="256x64x64_s3_sw4_f16",
            )
        with torch.cuda.stream(dense_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x)
            torch.mm(dense_x, gate_up_weight.t(), out=dense_gate_up)
            torch.ops._C.silu_and_mul(dense_hidden, dense_gate_up)
        current_stream.wait_stream(sparse_stream)
        current_stream.wait_stream(dense_stream)
        sparse24_copy_indexed_rows_contiguous_(
            sparse_hidden, dense_hidden, dense_rows
        )
        hidden = sparse_hidden
    if hidden is None:
        hidden = sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked(
            x,
            gate_up_values,
            gate_up_meta,
            gate_up_weight,
            dense_weight_rows,
            dense_rows,
            dense_slots,
            config="256x64_sparse_128x64_dense_f16",
        )
    return sparse24_cutlass_full_sparse_dense_override_linear_prepacked(
        hidden,
        down_values,
        down_meta,
        down_weight,
        dense_rows,
        dense_slots,
        config="256x64_sparse_128x64_dense_f16",
    )


def _batch_routed_mlp_impl(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
) -> torch.Tensor:
    """Execute dense, sparse, or mixed rows behind one graph boundary."""

    rows = int(x.shape[0])
    summary = current_verify_dense_row_summary(rows, x.device)
    if summary is None or summary[1] == rows:
        gate_up = torch.mm(x, gate_up_weight.t())
        hidden = _silu_and_mul_contiguous(gate_up)
        return torch.mm(hidden, down_weight.t())

    _row_is_dense, dense_count, dense_rows, sparse_rows = summary
    if dense_count == 0:
        if rows % 8:
            if _INLINE_SWIGLU_MLP_ENABLED:
                padded_rows = (rows + 7) // 8 * 8
                x_padded = torch.empty(
                    (padded_rows, x.shape[1]), device=x.device, dtype=x.dtype
                )
                x_padded[:rows].copy_(x)
                x_padded[rows:].zero_()
                hidden_padded = _sparse_gate_up_hidden(
                    x_padded, gate_up_values, gate_up_meta
                )
                output_padded = sparse24_cutlass_device_gemm_prepacked(
                    hidden_padded,
                    down_values,
                    down_meta,
                    contiguous_output=True,
                    input_transposed=True,
                )
                return output_padded[:rows]
            gate_up = sparse24_cutlass_device_gemm_prepacked(
                x.contiguous(),
                gate_up_values,
                gate_up_meta,
                contiguous_output=True,
            )
            hidden = _silu_and_mul_contiguous(gate_up)
            return sparse24_cutlass_device_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                contiguous_output=True,
            )
        hidden = _sparse_gate_up_hidden(
            x.contiguous(), gate_up_values, gate_up_meta
        )
        return sparse24_cutlass_device_gemm_prepacked(
            hidden,
            down_values,
            down_meta,
            contiguous_output=True,
            input_transposed=True,
        )

    sparse_count = rows - dense_count
    dense_rows = dense_rows[:dense_count]
    sparse_rows = sparse_rows[:sparse_count]
    x = x.contiguous()

    full_override_output = _fused_override_mlp(
        x,
        gate_up_weight,
        down_weight,
        gate_up_values,
        gate_up_meta,
        down_values,
        down_meta,
        dense_rows,
        sparse_rows,
    )
    if full_override_output is not None:
        return full_override_output

    dense_x = torch.empty(
        (dense_count, x.shape[1]),
        device=x.device,
        dtype=x.dtype,
    )
    sparse_rows_padded = (sparse_count + 7) // 8 * 8
    sparse_x_padded = torch.empty(
        (sparse_rows_padded, x.shape[1]),
        device=x.device,
        dtype=x.dtype,
    )
    sparse_x = sparse_x_padded[:sparse_count]
    sparse24_partition_rows_(
        x,
        dense_rows,
        sparse_rows,
        dense_x,
        sparse_x,
    )
    if sparse_rows_padded != sparse_count:
        sparse_x_padded[sparse_count:].zero_()

    output = (
        torch.empty(
            (rows, down_weight.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        if _INDEXED_DOWN_EPILOGUE_ENABLED
        else None
    )
    streams = _mixed_mlp_streams(x.device)
    if streams is None:
        dense_gate_up = torch.mm(dense_x, gate_up_weight.t())
        dense_hidden = _silu_and_mul_contiguous(dense_gate_up)
        dense_output = torch.mm(dense_hidden, down_weight.t())

        sparse_hidden = _sparse_gate_up_hidden(
            sparse_x_padded, gate_up_values, gate_up_meta
        )
        sparse_output_padded = _sparse_mixed_down(
            sparse_hidden,
            down_values,
            down_meta,
            sparse_rows,
            rows,
            output,
        )
    else:
        dense_stream, sparse_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        dense_stream.wait_stream(current_stream)
        sparse_stream.wait_stream(current_stream)
        with torch.cuda.stream(dense_stream):
            dense_gate_up = torch.mm(dense_x, gate_up_weight.t())
            dense_hidden = _silu_and_mul_contiguous(dense_gate_up)
            dense_output = torch.mm(dense_hidden, down_weight.t())
        with torch.cuda.stream(sparse_stream):
            sparse_hidden = _sparse_gate_up_hidden(
                sparse_x_padded, gate_up_values, gate_up_meta
            )
            sparse_output_padded = _sparse_mixed_down(
                sparse_hidden,
                down_values,
                down_meta,
                sparse_rows,
                rows,
                output,
            )
        current_stream.wait_stream(dense_stream)
        current_stream.wait_stream(sparse_stream)
    if _INDEXED_DOWN_EPILOGUE_ENABLED:
        assert output is not None
        sparse24_copy_indexed_rows_contiguous_(
            output,
            dense_output,
            dense_rows,
        )
        return output
    assert sparse_output_padded is not None
    output = torch.empty(
        (rows, down_weight.shape[0]), device=x.device, dtype=x.dtype
    )
    sparse24_merge_rows_(
        output,
        dense_output,
        sparse_output_padded[:sparse_count],
        dense_rows,
        sparse_rows,
    )
    return output


def _batch_routed_mlp_fake(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
) -> torch.Tensor:
    del gate_up_weight, gate_up_values, gate_up_meta, down_values, down_meta
    return x.new_empty((x.shape[0], down_weight.shape[0]))


def _full_plus_residual_projection(
    x: torch.Tensor,
    full_values: torch.Tensor,
    full_meta: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta: torch.Tensor,
    dense_count: int,
    dense_rows: torch.Tensor,
) -> torch.Tensor:
    rows = int(x.shape[0])
    all_dense = dense_count == rows
    residual_x = x
    if not all_dense:
        residual_x = torch.empty(
            (dense_count, x.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )

    streams = _mixed_mlp_streams(x.device)
    if streams is None:
        full_output = sparse24_cutlass_device_gemm_prepacked(
            x,
            full_values,
            full_meta,
            contiguous_output=True,
        )
        if not all_dense:
            sparse24_gather_rows_(x, dense_rows, residual_x)
        residual_output = sparse24_cutlass_device_gemm_prepacked(
            residual_x,
            residual_values,
            residual_meta,
            contiguous_output=True,
        )
    else:
        full_stream, residual_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        full_stream.wait_stream(current_stream)
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            full_output = sparse24_cutlass_device_gemm_prepacked(
                x,
                full_values,
                full_meta,
                contiguous_output=True,
            )
        with torch.cuda.stream(residual_stream):
            if not all_dense:
                sparse24_gather_rows_(x, dense_rows, residual_x)
            residual_output = sparse24_cutlass_device_gemm_prepacked(
                residual_x,
                residual_values,
                residual_meta,
                contiguous_output=True,
            )
        current_stream.wait_stream(full_stream)
        current_stream.wait_stream(residual_stream)

    if all_dense:
        full_output.add_(residual_output)
    else:
        sparse24_add_indexed_rows_contiguous_(
            full_output,
            residual_output,
            dense_rows,
        )
    return full_output


def _batch_routed_residual_mlp_impl(
    x: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    down_residual_values: torch.Tensor,
    down_residual_meta: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct selected dense MLP rows from complementary 2:4 GEMMs."""

    rows = int(x.shape[0])
    summary = current_verify_dense_row_summary(rows, x.device)
    if summary is None:
        dense_count = rows
        dense_rows = (
            _all_dense_row_ids(rows, x.device)
            if _ROUTED_SWIGLU_MLP_ENABLED
            else torch.empty(0, device=x.device, dtype=torch.int32)
        )
        row_is_dense = None
    else:
        row_is_dense, dense_count, dense_rows, _sparse_rows = summary

    if dense_count == 0:
        if _ROUTED_SWIGLU_MLP_ENABLED:
            x = x.contiguous()
            padded_rows = (rows + 7) // 8 * 8
            if padded_rows != rows:
                padded_x = torch.zeros(
                    (padded_rows, x.shape[1]), device=x.device, dtype=x.dtype
                )
                padded_x[:rows].copy_(x)
                x = padded_x
            hidden = _sparse_gate_up_hidden(x, gate_up_values, gate_up_meta)
            output = sparse24_cutlass_device_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                contiguous_output=True,
                input_transposed=True,
            )
            return output[:rows]
        if rows % 8:
            gate_up = sparse24_cutlass_device_gemm_prepacked(
                x.contiguous(),
                gate_up_values,
                gate_up_meta,
                contiguous_output=True,
            )
            hidden = _silu_and_mul_contiguous(gate_up)
            return sparse24_cutlass_device_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                contiguous_output=True,
            )
        gate_up = sparse24_cutlass_device_gemm_prepacked(
            x.contiguous(),
            gate_up_values,
            gate_up_meta,
            contiguous_output=False,
        )
        hidden = sparse24_silu_and_mul_transposed(gate_up)
        return sparse24_cutlass_device_gemm_prepacked(
            hidden,
            down_values,
            down_meta,
            contiguous_output=True,
            input_transposed=True,
        )

    x = x.contiguous()
    dense_rows = dense_rows[:dense_count]
    if _ROUTED_SWIGLU_MLP_ENABLED:
        if not x.is_cuda or x.dtype != torch.float16:
            raise RuntimeError(
                "routed sparse SwiGLU MLP requires a CUDA fp16 input"
            )
        dense_slots = current_verify_dense_slots(rows, x.device)
        if dense_slots is None:
            if row_is_dense is None:
                dense_slots = dense_rows
            else:
                dense_slots = torch.full(
                    (rows,), -1, device=x.device, dtype=torch.int32
                )
                dense_slots.masked_scatter_(
                    row_is_dense,
                    _all_dense_row_ids(dense_count, x.device),
                )
        return _batch_routed_residual_mlp_routed_swiglu(
            x,
            dense_rows,
            dense_slots,
            gate_up_values,
            gate_up_meta,
            gate_up_residual_values,
            gate_up_residual_meta,
            down_values,
            down_meta,
            down_residual_values,
            down_residual_meta,
        )
    if (
        _TRANSPOSED_RESIDUAL_MLP_ENABLED
        and x.is_cuda
        and dense_count < rows
        and rows % 8 == 0
        and dense_count % 8 == 0
    ):
        return _batch_routed_residual_mlp_transposed(
            x,
            dense_rows,
            gate_up_values,
            gate_up_meta,
            gate_up_residual_values,
            gate_up_residual_meta,
            down_values,
            down_meta,
            down_residual_values,
            down_residual_meta,
        )

    gate_up = _full_plus_residual_projection(
        x,
        gate_up_values,
        gate_up_meta,
        gate_up_residual_values,
        gate_up_residual_meta,
        dense_count,
        dense_rows,
    )
    hidden = _silu_and_mul_contiguous(gate_up)
    return _full_plus_residual_projection(
        hidden,
        down_values,
        down_meta,
        down_residual_values,
        down_residual_meta,
        dense_count,
        dense_rows,
    )


def _routed_residual_gate_swiglu_hidden(
    x: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slots: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return exact dense-row and 2:4 sparse-row SwiGLU hidden states."""

    rows, hidden_size = map(int, x.shape)
    dense_count = int(dense_rows.numel())
    run_rows = (rows + 7) // 8 * 8
    dense_run = (dense_count + 7) // 8 * 8
    gate_up_size = int(gate_up_values.shape[0])
    intermediate_size = gate_up_size // 2

    if run_rows == rows:
        run_x = x
        run_dense_slots = dense_slots
    else:
        run_x = torch.zeros(
            (run_rows, hidden_size), device=x.device, dtype=x.dtype
        )
        run_x[:rows].copy_(x)
        run_dense_slots = torch.full(
            (run_rows,), -1, device=x.device, dtype=torch.int32
        )
        run_dense_slots[:rows].copy_(dense_slots)

    dense_x = (
        torch.empty(
            (dense_run, hidden_size), device=x.device, dtype=x.dtype
        )
        if _UNINITIALIZED_ROUTED_WORKSPACE_ENABLED
        else torch.zeros(
            (dense_run, hidden_size), device=x.device, dtype=x.dtype
        )
    )
    routed_hidden = torch.empty(
        (run_rows, intermediate_size), device=x.device, dtype=x.dtype
    )
    dense_base = torch.empty(
        (dense_count, gate_up_size), device=x.device, dtype=x.dtype
    )
    gate_residual = torch.empty(
        (dense_run, gate_up_size), device=x.device, dtype=x.dtype
    )
    gate_workspace = (
        None
        if _DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED
        else torch.empty(
            (gate_up_size, dense_run), device=x.device, dtype=x.dtype
        )
    )
    streams = _mixed_mlp_streams(x.device)
    use_paired_gate = _use_paired_persistent_gate(
        run_rows, dense_count, gate_up_size
    )

    if (
        use_paired_gate
        and _PAIRED_FUSED_GATE_EPILOGUE_ENABLED
        and not _GATE_DOWN_GATHER_EPILOGUE_ENABLED
        and 0 < dense_count < run_rows
    ):
        feature_counters = _paired_fused_gate_counters(
            x.device,
            run_rows,
            dense_count,
            gate_up_size,
            gate_up_values.data_ptr(),
        )
        if feature_counters is not None:
            _log_kernel_dispatch(
                "mlp_paired_fused_gate_epilogue",
                run_rows,
                dense_count,
                intermediate_size,
            )
            sparse24_cutlass_paired_fused_routed_swiglu_prepacked(
                run_x,
                gate_up_values,
                gate_up_meta,
                gate_up_residual_values,
                gate_up_residual_meta,
                dense_rows,
                run_dense_slots,
                out=routed_hidden,
                dense_base=dense_base,
                feature_counters=feature_counters,
                config=_ROUTED_SWIGLU_CONFIG,
                worker_blocks=_paired_gate_worker_blocks(
                    run_rows, dense_count, gate_up_size
                ),
                schedule="partitioned",
            )
            return routed_hidden, run_dense_slots, None

    if use_paired_gate:
        sparse24_gather_rows_(
            run_x, dense_rows, dense_x[:dense_count]
        )
        sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
            run_x,
            gate_up_values,
            gate_up_meta,
            run_dense_slots,
            dense_x,
            gate_up_residual_values,
            gate_up_residual_meta,
            dense_count=dense_count,
            full_out=routed_hidden,
            dense_base=dense_base,
            residual_out=gate_residual,
            schedule=_paired_gate_schedule(
                run_rows,
                dense_count,
                gate_up_size,
            ),
            worker_blocks=_paired_gate_worker_blocks(
                run_rows, dense_count, gate_up_size
            ),
        )
    elif streams is None:
        sparse24_cutlass_routed_swiglu_prepacked(
            run_x,
            gate_up_values,
            gate_up_meta,
            run_dense_slots,
            dense_count=dense_count,
            out=routed_hidden,
            dense_base=dense_base,
            config=_ROUTED_SWIGLU_CONFIG,
        )
        sparse24_gather_rows_(run_x, dense_rows, dense_x[:dense_count])
        if _DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED:
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                dense_x,
                gate_up_residual_values,
                gate_up_residual_meta,
                out=gate_residual,
                config="auto",
                store_mode="vector",
            )
        else:
            sparse24_cutlass_device_gemm_prepacked(
                dense_x,
                gate_up_residual_values,
                gate_up_residual_meta,
                contiguous_output=True,
                out=gate_residual,
                workspace=gate_workspace,
                device_config="auto",
            )
    else:
        full_stream, residual_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        full_stream.wait_stream(current_stream)
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                run_x,
                gate_up_values,
                gate_up_meta,
                run_dense_slots,
                dense_count=dense_count,
                out=routed_hidden,
                dense_base=dense_base,
                config=_ROUTED_SWIGLU_CONFIG,
            )
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_(run_x, dense_rows, dense_x[:dense_count])
            if _DIRECT_GATE_RESIDUAL_EPILOGUE_ENABLED:
                sparse24_cutlass_inline_transpose_gemm_prepacked(
                    dense_x,
                    gate_up_residual_values,
                    gate_up_residual_meta,
                    out=gate_residual,
                    config="auto",
                    store_mode="vector",
                )
            else:
                sparse24_cutlass_device_gemm_prepacked(
                    dense_x,
                    gate_up_residual_values,
                    gate_up_residual_meta,
                    contiguous_output=True,
                    out=gate_residual,
                    workspace=gate_workspace,
                    device_config="auto",
                )
        current_stream.wait_stream(full_stream)
        current_stream.wait_stream(residual_stream)

    compact_dense_hidden = None
    if _GATE_DOWN_GATHER_EPILOGUE_ENABLED:
        _log_kernel_dispatch(
            "mlp_gate_down_gather_epilogue",
            run_rows,
            dense_count,
            intermediate_size,
        )
        compact_dense_hidden = torch.empty(
            (dense_run, intermediate_size),
            device=x.device,
            dtype=x.dtype,
        )
        sparse24_routed_swiglu_correction_gather_(
            dense_base,
            gate_residual[:dense_count],
            dense_rows,
            routed_hidden,
            compact_dense_hidden,
        )
    else:
        sparse24_routed_swiglu_correction_(
            dense_base,
            gate_residual[:dense_count],
            dense_rows,
            routed_hidden,
        )
    return routed_hidden, run_dense_slots, compact_dense_hidden


def _batch_routed_residual_mlp_routed_swiglu(
    x: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slots: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    down_residual_values: torch.Tensor,
    down_residual_meta: torch.Tensor,
) -> torch.Tensor:
    """Fuse row routing, residual layout conversion, SwiGLU, and Down input."""

    rows, hidden_size = map(int, x.shape)
    dense_count = int(dense_rows.numel())
    routed_hidden, run_dense_slots, compact_dense_hidden = (
        _routed_residual_gate_swiglu_hidden(
            x,
            dense_rows,
            dense_slots,
            gate_up_values,
            gate_up_meta,
            gate_up_residual_values,
            gate_up_residual_meta,
        )
    )
    run_rows = int(routed_hidden.shape[0])
    dense_run = (dense_count + 7) // 8 * 8
    intermediate_size = int(routed_hidden.shape[1])
    paired_inplace_config = _paired_inplace_down_config(
        run_rows,
        dense_count,
        intermediate_size,
    )
    if paired_inplace_config is not None and 0 < dense_count < run_rows:
        feature_counters = _paired_inplace_down_counters(
            x.device,
            run_rows,
            dense_count,
            hidden_size,
            down_values.data_ptr(),
            paired_inplace_config,
        )
        if feature_counters is not None:
            _log_kernel_dispatch(
                f"mlp_paired_inplace_down_{paired_inplace_config}",
                run_rows,
                dense_count,
                hidden_size,
            )
            output = torch.empty(
                (run_rows, hidden_size),
                device=x.device,
                dtype=x.dtype,
            )
            sparse24_cutlass_paired_inplace_residual_prepacked(
                routed_hidden,
                down_values,
                down_meta,
                down_residual_values,
                down_residual_meta,
                dense_rows,
                out=output,
                feature_counters=feature_counters,
                config=paired_inplace_config,
                schedule="partitioned",
            )
            return output[:rows]

    paired_gather_config = _paired_gather_down_config(
        run_rows,
        dense_count,
        intermediate_size,
    )
    if paired_gather_config is not None:
        _log_kernel_dispatch(
            f"mlp_paired_gather_down_{paired_gather_config}",
            run_rows,
            dense_count,
            hidden_size,
        )
        full_down = torch.empty(
            (run_rows, hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        residual_down = torch.empty(
            (dense_count, hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        sparse24_cutlass_paired_gather_residual_prepacked(
            routed_hidden,
            down_values,
            down_meta,
            down_residual_values,
            down_residual_meta,
            dense_rows,
            full_out=full_down,
            residual_out=residual_down,
            schedule="interleaved",
            config=paired_gather_config,
        )
        sparse24_add_indexed_rows_contiguous_(
            full_down,
            residual_down,
            dense_rows,
        )
        return full_down[:rows]

    streams = _mixed_mlp_streams(x.device)
    split_streams = _parallel_splitk_down_streams(x.device)
    split_k_slices = _parallel_splitk_down_slices(
        run_rows,
        dense_count,
        intermediate_size,
    )
    if (
        streams is None
        or split_streams is None
        or len(split_streams) < split_k_slices
    ):
        split_k_slices = 0
    single_launch_split_k_slices = _single_launch_splitk_down_slices(
        run_rows,
        dense_run,
        intermediate_size,
    )
    tiled_indexed_config = _tiled_indexed_splitk_down_config(
        run_rows,
        dense_run,
        intermediate_size,
    )
    if tiled_indexed_config is not None:
        split_k_slices = 0
        single_launch_split_k_slices = 0
    elif split_k_slices:
        single_launch_split_k_slices = 0
    use_paired_down = (
        split_k_slices == 0
        and single_launch_split_k_slices == 0
        and tiled_indexed_config is None
        and _use_paired_persistent_down(
            run_rows,
            dense_count,
            intermediate_size,
        )
    )
    use_contiguous_down = (
        split_k_slices > 0
        or single_launch_split_k_slices > 0
        or tiled_indexed_config is not None
        or use_paired_down
        or (
            _CONTIGUOUS_DOWN_INPUT_ENABLED
            and not (
                hidden_size == 4096
                and intermediate_size == 12288
                and run_rows >= 704
            )
        )
    )
    if use_contiguous_down:
        hidden = routed_hidden
        if compact_dense_hidden is not None:
            dense_hidden = compact_dense_hidden
            gather_hidden_rows = None
        else:
            dense_hidden = (
                torch.empty(
                    (dense_run, intermediate_size),
                    device=x.device,
                    dtype=x.dtype,
                )
                if _UNINITIALIZED_ROUTED_WORKSPACE_ENABLED
                else torch.zeros(
                    (dense_run, intermediate_size),
                    device=x.device,
                    dtype=x.dtype,
                )
            )
            gather_hidden_rows = sparse24_gather_rows_
    else:
        hidden = sparse24_transpose_input_to_strided(routed_hidden)
        if _UNINITIALIZED_ROUTED_WORKSPACE_ENABLED:
            dense_hidden = torch.empty_strided(
                (dense_run, intermediate_size),
                (1, dense_run),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            dense_hidden = torch.zeros(
                (dense_run, intermediate_size),
                device=x.device,
                dtype=x.dtype,
            ).as_strided(
                (dense_run, intermediate_size), (1, dense_run)
            )
        gather_hidden_rows = sparse24_gather_rows_strided_
    if tiled_indexed_config is not None:
        full_down = torch.empty(
            (run_rows, hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        full_down = torch.empty_strided(
            (run_rows, hidden_size),
            (1, run_rows),
            device=x.device,
            dtype=x.dtype,
        )
    if split_k_slices:
        residual_down = None
        residual_partials = torch.empty_strided(
            (split_k_slices, dense_run, hidden_size),
            (hidden_size * dense_run, 1, dense_run),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        residual_down = torch.empty_strided(
            (dense_run, hidden_size),
            (1, dense_run),
            device=x.device,
            dtype=x.dtype,
        )
        residual_partials = None

    single_launch_workspace = None
    if single_launch_split_k_slices:
        _log_kernel_dispatch(
            f"mlp_single_launch_splitk_down_s{single_launch_split_k_slices}",
            run_rows,
            dense_count,
            hidden_size,
        )
        single_launch_workspace = _single_launch_splitk_down_workspace(
            x.device,
            dense_run,
            hidden_size,
        )
    tiled_indexed_workspace = None
    if tiled_indexed_config is not None:
        tiled_split_k_slices, tiled_full_config = tiled_indexed_config
        _log_kernel_dispatch(
            f"mlp_tiled_indexed_splitk_down_s{tiled_split_k_slices}",
            run_rows,
            dense_count,
            hidden_size,
        )
        tiled_indexed_workspace = _single_launch_splitk_down_workspace(
            x.device,
            dense_run,
            hidden_size,
        )

    if split_k_slices:
        assert streams is not None
        assert split_streams is not None
        assert residual_partials is not None
        _log_kernel_dispatch(
            f"mlp_parallel_splitk_down_s{split_k_slices}",
            run_rows,
            dense_count,
            hidden_size,
        )
        weight_slices = _splitk_down_weight_views(
            down_residual_values,
            down_residual_meta,
            split_k_slices,
        )
        full_stream, gather_stream = streams
        active_split_streams = split_streams[:split_k_slices]
        current_stream = torch.cuda.current_stream(x.device)
        full_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                contiguous_output=False,
                out=full_down,
                device_config="auto",
            )
        if gather_hidden_rows is not None:
            gather_stream.wait_stream(current_stream)
            with torch.cuda.stream(gather_stream):
                gather_hidden_rows(
                    hidden, dense_rows, dense_hidden[:dense_count]
                )
            for stream in active_split_streams:
                stream.wait_stream(gather_stream)
        else:
            for stream in active_split_streams:
                stream.wait_stream(current_stream)
        for split, (stream, packed_slice) in enumerate(
            zip(active_split_streams, weight_slices, strict=True)
        ):
            start, end, values, metadata = packed_slice
            with torch.cuda.stream(stream):
                sparse24_cutlass_device_strided_input_gemm_prepacked(
                    dense_hidden[:, start:end],
                    values,
                    metadata,
                    out=residual_partials[split],
                )
        current_stream.wait_stream(full_stream)
        for stream in active_split_streams:
            current_stream.wait_stream(stream)
        output = sparse24_transpose_add_routed_splitk_residual(
            full_down,
            residual_partials,
            run_dense_slots,
            dense_count=dense_count,
        )
        return output[:rows]

    assert residual_down is not None
    if use_paired_down:
        if gather_hidden_rows is not None:
            gather_hidden_rows(
                hidden, dense_rows, dense_hidden[:dense_count]
            )
        sparse24_cutlass_paired_persistent_gemm_prepacked(
            hidden,
            down_values,
            down_meta,
            dense_hidden,
            down_residual_values,
            down_residual_meta,
            full_out=full_down,
            residual_out=residual_down,
            schedule="interleaved",
        )
    elif streams is None:
        if tiled_indexed_config is not None:
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                out=full_down,
                config=tiled_full_config,
                store_mode="vector",
            )
        else:
            sparse24_cutlass_device_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                contiguous_output=False,
                input_transposed=not use_contiguous_down,
                out=full_down,
                device_config="auto",
            )
        if gather_hidden_rows is not None:
            gather_hidden_rows(
                hidden, dense_rows, dense_hidden[:dense_count]
            )
        if tiled_indexed_config is not None:
            assert tiled_indexed_workspace is not None
            sparse24_cutlass_device_splitk_gemm_prepacked(
                dense_hidden,
                down_residual_values,
                down_residual_meta,
                split_k_slices=tiled_split_k_slices,
                out=residual_down,
                workspace=tiled_indexed_workspace,
            )
        elif single_launch_split_k_slices:
            assert single_launch_workspace is not None
            sparse24_cutlass_device_splitk_gemm_prepacked(
                dense_hidden,
                down_residual_values,
                down_residual_meta,
                split_k_slices=single_launch_split_k_slices,
                out=residual_down,
                workspace=single_launch_workspace,
            )
        else:
            sparse24_cutlass_device_gemm_prepacked(
                dense_hidden,
                down_residual_values,
                down_residual_meta,
                contiguous_output=False,
                input_transposed=not use_contiguous_down,
                out=residual_down,
                device_config="auto",
            )
    else:
        full_stream, residual_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        full_stream.wait_stream(current_stream)
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            if tiled_indexed_config is not None:
                sparse24_cutlass_inline_transpose_gemm_prepacked(
                    hidden,
                    down_values,
                    down_meta,
                    out=full_down,
                    config=tiled_full_config,
                    store_mode="vector",
                )
            else:
                sparse24_cutlass_device_gemm_prepacked(
                    hidden,
                    down_values,
                    down_meta,
                    contiguous_output=False,
                    input_transposed=not use_contiguous_down,
                    out=full_down,
                    device_config="auto",
                )
        with torch.cuda.stream(residual_stream):
            if gather_hidden_rows is not None:
                gather_hidden_rows(
                    hidden, dense_rows, dense_hidden[:dense_count]
                )
            if tiled_indexed_config is not None:
                assert tiled_indexed_workspace is not None
                sparse24_cutlass_device_splitk_gemm_prepacked(
                    dense_hidden,
                    down_residual_values,
                    down_residual_meta,
                    split_k_slices=tiled_split_k_slices,
                    out=residual_down,
                    workspace=tiled_indexed_workspace,
                )
            elif single_launch_split_k_slices:
                assert single_launch_workspace is not None
                sparse24_cutlass_device_splitk_gemm_prepacked(
                    dense_hidden,
                    down_residual_values,
                    down_residual_meta,
                    split_k_slices=single_launch_split_k_slices,
                    out=residual_down,
                    workspace=single_launch_workspace,
                )
            else:
                sparse24_cutlass_device_gemm_prepacked(
                    dense_hidden,
                    down_residual_values,
                    down_residual_meta,
                    contiguous_output=False,
                    input_transposed=not use_contiguous_down,
                    out=residual_down,
                    device_config="auto",
                )
        current_stream.wait_stream(full_stream)
        current_stream.wait_stream(residual_stream)

    if tiled_indexed_config is not None:
        return sparse24_add_indexed_rows_transposed_to_contiguous_(
            full_down,
            residual_down,
            dense_rows,
        )[:rows]

    output = sparse24_transpose_add_routed_residual(
        full_down,
        residual_down,
        run_dense_slots,
        dense_count=dense_count,
    )
    return output[:rows]


def _batch_routed_residual_mlp_transposed(
    x: torch.Tensor,
    dense_rows: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    down_residual_values: torch.Tensor,
    down_residual_meta: torch.Tensor,
) -> torch.Tensor:
    """Keep the mixed residual MLP in CUTLASS layout until final output."""

    rows, hidden_size = map(int, x.shape)
    dense_count = int(dense_rows.numel())
    residual_x = torch.empty(
        (dense_count, hidden_size),
        device=x.device,
        dtype=x.dtype,
    )
    streams = _mixed_mlp_streams(x.device)

    if streams is None:
        full_gate_up = sparse24_cutlass_device_gemm_prepacked(
            x,
            gate_up_values,
            gate_up_meta,
            contiguous_output=False,
        )
        sparse24_gather_rows_(x, dense_rows, residual_x)
        residual_gate_up = sparse24_cutlass_device_gemm_prepacked(
            residual_x,
            gate_up_residual_values,
            gate_up_residual_meta,
            contiguous_output=False,
        )
    else:
        full_stream, residual_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        full_stream.wait_stream(current_stream)
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            full_gate_up = sparse24_cutlass_device_gemm_prepacked(
                x,
                gate_up_values,
                gate_up_meta,
                contiguous_output=False,
            )
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_(x, dense_rows, residual_x)
            residual_gate_up = sparse24_cutlass_device_gemm_prepacked(
                residual_x,
                gate_up_residual_values,
                gate_up_residual_meta,
                contiguous_output=False,
            )
        current_stream.wait_stream(full_stream)
        current_stream.wait_stream(residual_stream)

    sparse24_add_indexed_rows_strided_(
        full_gate_up,
        residual_gate_up,
        dense_rows,
    )
    hidden = sparse24_silu_and_mul_transposed(full_gate_up)
    intermediate_size = int(hidden.shape[1])
    residual_hidden = torch.empty_strided(
        (dense_count, intermediate_size),
        (1, dense_count),
        device=x.device,
        dtype=x.dtype,
    )

    if streams is None:
        full_down = sparse24_cutlass_device_gemm_prepacked(
            hidden,
            down_values,
            down_meta,
            contiguous_output=False,
            input_transposed=True,
        )
        sparse24_gather_rows_strided_(hidden, dense_rows, residual_hidden)
        residual_down = sparse24_cutlass_device_gemm_prepacked(
            residual_hidden,
            down_residual_values,
            down_residual_meta,
            contiguous_output=False,
            input_transposed=True,
        )
    else:
        full_stream, residual_stream = streams
        current_stream = torch.cuda.current_stream(x.device)
        full_stream.wait_stream(current_stream)
        residual_stream.wait_stream(current_stream)
        with torch.cuda.stream(full_stream):
            full_down = sparse24_cutlass_device_gemm_prepacked(
                hidden,
                down_values,
                down_meta,
                contiguous_output=False,
                input_transposed=True,
            )
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_strided_(hidden, dense_rows, residual_hidden)
            residual_down = sparse24_cutlass_device_gemm_prepacked(
                residual_hidden,
                down_residual_values,
                down_residual_meta,
                contiguous_output=False,
                input_transposed=True,
            )
        current_stream.wait_stream(full_stream)
        current_stream.wait_stream(residual_stream)

    sparse24_add_indexed_rows_strided_(full_down, residual_down, dense_rows)
    return sparse24_transpose_output_contiguous(full_down)


def _batch_routed_residual_mlp_fake(
    x: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    down_residual_values: torch.Tensor,
    down_residual_meta: torch.Tensor,
) -> torch.Tensor:
    del (
        gate_up_values,
        gate_up_meta,
        gate_up_residual_values,
        gate_up_residual_meta,
        down_meta,
        down_residual_values,
        down_residual_meta,
    )
    return x.new_empty((x.shape[0], down_values.shape[0]))


def _routed_residual_gate_swiglu_impl(
    x: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
) -> torch.Tensor:
    """Return mixed exact-dense/2:4 Gate+SwiGLU rows for a dense Down."""

    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
        raise RuntimeError(
            "routed residual Gate+SwiGLU requires a CUDA fp16 matrix"
        )

    rows = int(x.shape[0])
    summary = current_verify_dense_row_summary(rows, x.device)
    if summary is None:
        row_is_dense = None
        dense_count = rows
        dense_rows = _all_dense_row_ids(rows, x.device)
    else:
        row_is_dense, dense_count, dense_rows, _sparse_rows = summary

    x = x.contiguous()
    if dense_count == 0:
        run_rows = (rows + 7) // 8 * 8
        if run_rows == rows:
            run_x = x
        else:
            run_x = torch.zeros(
                (run_rows, x.shape[1]), device=x.device, dtype=x.dtype
            )
            run_x[:rows].copy_(x)
        hidden = sparse24_cutlass_gate_up_swiglu_prepacked(
            run_x,
            gate_up_values,
            gate_up_meta,
            config=_sparse_gate_dense_down_config(
                run_rows, int(gate_up_values.shape[0])
            ),
            output_transposed=False,
        )
        return hidden[:rows]

    dense_rows = dense_rows[:dense_count]
    dense_slots = current_verify_dense_slots(rows, x.device)
    if dense_slots is None:
        if row_is_dense is None:
            dense_slots = dense_rows
        else:
            dense_slots = torch.full(
                (rows,), -1, device=x.device, dtype=torch.int32
            )
            dense_slots.masked_scatter_(
                row_is_dense,
                _all_dense_row_ids(dense_count, x.device),
            )
    hidden, _run_dense_slots, _compact_dense_hidden = (
        _routed_residual_gate_swiglu_hidden(
            x,
            dense_rows,
            dense_slots,
            gate_up_values,
            gate_up_meta,
            gate_up_residual_values,
            gate_up_residual_meta,
        )
    )
    return hidden[:rows]


def _routed_residual_gate_swiglu_fake(
    x: torch.Tensor,
    gate_up_values: torch.Tensor,
    gate_up_meta: torch.Tensor,
    gate_up_residual_values: torch.Tensor,
    gate_up_residual_meta: torch.Tensor,
) -> torch.Tensor:
    del (
        gate_up_meta,
        gate_up_residual_values,
        gate_up_residual_meta,
    )
    return x.new_empty((x.shape[0], gate_up_values.shape[0] // 2))


direct_register_custom_op(
    op_name="batch_routed_mlp",
    op_func=_batch_routed_mlp_impl,
    fake_impl=_batch_routed_mlp_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="batch_routed_residual_mlp",
    op_func=_batch_routed_residual_mlp_impl,
    fake_impl=_batch_routed_residual_mlp_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)
direct_register_custom_op(
    op_name="routed_residual_gate_swiglu",
    op_func=_routed_residual_gate_swiglu_impl,
    fake_impl=_routed_residual_gate_swiglu_fake,
    target_lib=_SPECLINK_OP_LIB,
    dispatch_key="CUDA",
)


def _dense_mlp_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up_proj = mlp.gate_up_proj
    down_proj = mlp.down_proj
    if _SPECLINK_VERIFY_DETAIL:
        with verify_timer("gate_up_proj"):
            if getattr(
                gate_up_proj,
                "_speclink_sparse24_dense_weight_released",
                False,
            ):
                gate_up, _ = speclink_linear_forward(gate_up_proj, x)
            else:
                gate_up, _ = gate_up_proj(x)
        with verify_timer("swiglu"):
            hidden = _apply_gate_up_activation(mlp, gate_up)
        with verify_timer("down_proj"):
            if getattr(
                down_proj,
                "_speclink_sparse24_dense_weight_released",
                False,
            ):
                down, _ = speclink_linear_forward(down_proj, hidden)
            else:
                down, _ = down_proj(hidden)
        return down
    if getattr(gate_up_proj, "_speclink_sparse24_dense_weight_released", False):
        gate_up, _ = speclink_linear_forward(gate_up_proj, x)
    else:
        gate_up, _ = gate_up_proj(x)
    hidden = _apply_gate_up_activation(mlp, gate_up)
    if getattr(down_proj, "_speclink_sparse24_dense_weight_released", False):
        down, _ = speclink_linear_forward(down_proj, hidden)
    else:
        down, _ = down_proj(hidden)
    return down


def _can_use_cutlass_down_fp16(down_proj: Any, hidden: torch.Tensor) -> bool:
    if not _CUTLASS_DOWN_FP16_ENABLED:
        return False
    if not hidden.is_cuda or hidden.dtype != torch.float16 or hidden.ndim != 2:
        return False
    if int(getattr(down_proj, "tp_size", 1)) != 1:
        return False
    if getattr(down_proj, "bias", None) is not None:
        return False
    if getattr(down_proj, "_speclink_sparse24_dense_weight_released", False):
        return False
    if (
        getattr(down_proj, "_speclink_selective_dense_enabled", False)
        and not getattr(down_proj, "_speclink_selective_dense_bypass", False)
    ):
        return False
    weight = getattr(down_proj, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if not weight.is_cuda or weight.dtype != torch.float16 or weight.ndim != 2:
        return False
    return tuple(weight.shape) in {(4096, 12288), (4096, 14336)}


def _dense_down_projection(down_proj: Any, hidden: torch.Tensor) -> torch.Tensor:
    if _can_use_cutlass_down_fp16(down_proj, hidden):
        return torch.ops.speclink.dense_down_fp16.default(
            hidden, down_proj.weight
        )
    down, _ = speclink_linear_forward(down_proj, hidden)
    return down


def _linear_mlp_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    if _SPECLINK_VERIFY_DETAIL:
        with verify_timer("gate_up_proj"):
            gate_up, _ = speclink_linear_forward(mlp.gate_up_proj, x)
        with verify_timer("swiglu"):
            hidden = _apply_gate_up_activation(mlp, gate_up)
        with verify_timer("down_proj"):
            return _dense_down_projection(mlp.down_proj, hidden)
    hidden = _linear_gate_swiglu_hidden(mlp, x)
    return _dense_down_projection(mlp.down_proj, hidden)


def _linear_gate_swiglu_hidden(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = speclink_linear_forward(mlp.gate_up_proj, x)
    return _apply_gate_up_activation(mlp, gate_up)


def _gate_only_mlp_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up, _ = speclink_linear_forward(mlp.gate_up_proj, x)
    hidden = _apply_gate_up_activation(mlp, gate_up)
    if _can_use_cutlass_down_fp16(mlp.down_proj, hidden):
        return torch.ops.speclink.dense_down_fp16.default(
            hidden, mlp.down_proj.weight
        )
    down, _ = mlp.down_proj(hidden)
    return down


def _can_use_sparse_gate_dense_down(mlp: Any, x: torch.Tensor) -> bool:
    if not _SPARSE_GATE_DENSE_DOWN_ENABLED:
        return False
    if mlp_strategy() != "linear":
        return False
    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
        return False

    gate_up = mlp.gate_up_proj
    down = mlp.down_proj
    if not getattr(
        gate_up, "_speclink_sparse24_sparse_gate_dense_down", False
    ):
        return False
    if not getattr(gate_up, "_speclink_selective_dense_enabled", False):
        return False
    if getattr(gate_up, "_speclink_selective_dense_bypass", False):
        return False
    if getattr(gate_up, "_speclink_selective_mixed_rows", True):
        return False
    if not (
        not getattr(down, "_speclink_selective_dense_enabled", False)
        or getattr(down, "_speclink_selective_dense_bypass", False)
    ):
        return False
    if getattr(down, "_speclink_sparse24_dense_weight_released", False):
        return False
    if int(getattr(gate_up, "tp_size", 1)) != 1:
        return False
    if int(getattr(down, "tp_size", 1)) != 1:
        return False
    if getattr(gate_up, "bias", None) is not None:
        return False
    if getattr(down, "bias", None) is not None:
        return False
    if getattr(gate_up, "_speclink_gate_up_hybrid", "none") != "none":
        return False
    if mlp.act_fn.__class__.__name__ != "SiluAndMul":
        return False
    return isinstance(
        getattr(gate_up, "_speclink_sparse24_full_a_values", None),
        torch.Tensor,
    ) and isinstance(
        getattr(gate_up, "_speclink_sparse24_full_a_meta_e", None),
        torch.Tensor,
    )


def _sparse_gate_dense_down_forward(
    mlp: Any, x: torch.Tensor
) -> torch.Tensor:
    hidden = _sparse_gate_swiglu_hidden(mlp, x)
    if _can_use_cutlass_down_fp16(mlp.down_proj, hidden):
        return torch.ops.speclink.dense_down_fp16.default(
            hidden, mlp.down_proj.weight
        )
    down, _ = mlp.down_proj(hidden)
    return down


def _sparse_gate_swiglu_hidden(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up = mlp.gate_up_proj
    return torch.ops.speclink.sparse_gate_swiglu.default(
        x,
        gate_up.weight,
        gate_up._speclink_sparse24_full_a_values,
        gate_up._speclink_sparse24_full_a_meta_e,
    )


def _can_use_routed_gate_dense_down(mlp: Any, x: torch.Tensor) -> bool:
    if not (_FUSED_BATCH_MLP_ENABLED and _ROUTED_SWIGLU_MLP_ENABLED):
        return False
    if linear_strategy() != "full_sparse_residual":
        return False
    if mlp_strategy() != "linear":
        return False
    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
        return False

    gate_up = mlp.gate_up_proj
    down = mlp.down_proj
    if not getattr(gate_up, "_speclink_sparse24_routed_swiglu", False):
        return False
    if not getattr(
        gate_up, "_speclink_sparse24_gate_up_interleaved", False
    ):
        return False
    if not getattr(gate_up, "_speclink_selective_dense_enabled", False):
        return False
    if getattr(gate_up, "_speclink_selective_dense_bypass", False):
        return False
    if not getattr(gate_up, "_speclink_selective_mixed_rows", True):
        return False
    if not (
        not getattr(down, "_speclink_selective_dense_enabled", False)
        or getattr(down, "_speclink_selective_dense_bypass", False)
    ):
        return False
    if getattr(down, "_speclink_sparse24_dense_weight_released", False):
        return False
    if int(getattr(gate_up, "tp_size", 1)) != 1:
        return False
    if int(getattr(down, "tp_size", 1)) != 1:
        return False
    if getattr(gate_up, "bias", None) is not None:
        return False
    if getattr(down, "bias", None) is not None:
        return False
    if getattr(gate_up, "_speclink_gate_up_hybrid", "none") != "none":
        return False
    if mlp.act_fn.__class__.__name__ != "SiluAndMul":
        return False
    for name in (
        "_speclink_sparse24_full_a_values",
        "_speclink_sparse24_full_a_meta_e",
        "_speclink_sparse24_residual_a_values",
        "_speclink_sparse24_residual_a_meta_e",
    ):
        if not isinstance(getattr(gate_up, name, None), torch.Tensor):
            return False
    return True


def _routed_gate_dense_down_forward(
    mlp: Any, x: torch.Tensor
) -> torch.Tensor:
    hidden = _routed_gate_swiglu_hidden(mlp, x)
    return _dense_down_projection(mlp.down_proj, hidden)


def _routed_gate_swiglu_hidden(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up = mlp.gate_up_proj
    return torch.ops.speclink.routed_residual_gate_swiglu.default(
        x,
        gate_up._speclink_sparse24_full_a_values,
        gate_up._speclink_sparse24_full_a_meta_e,
        gate_up._speclink_sparse24_residual_a_values,
        gate_up._speclink_sparse24_residual_a_meta_e,
    )


def _can_use_static_sparse_mlp(mlp: Any, x: torch.Tensor) -> bool:
    """Use the inline Gate+SwiGLU epilogue when both MLP projections are static 2:4."""

    if not (_FUSED_BATCH_MLP_ENABLED and _INLINE_SWIGLU_MLP_ENABLED):
        return False
    if _ROUTED_SWIGLU_MLP_ENABLED or mlp_strategy() != "linear":
        return False
    if linear_strategy() not in {
        "full_sparse_dense_override",
        "split_dense_sparse",
    }:
        return False
    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
        return False

    gate_up = mlp.gate_up_proj
    down = mlp.down_proj
    if not getattr(
        gate_up, "_speclink_sparse24_gate_up_interleaved", False
    ):
        return False
    if getattr(gate_up, "_speclink_sparse24_routed_swiglu", False):
        return False
    for module in (gate_up, down):
        if not getattr(module, "_speclink_selective_dense_enabled", False):
            return False
        if getattr(module, "_speclink_selective_dense_bypass", False):
            return False
        if getattr(module, "_speclink_selective_mixed_rows", True):
            return False
        if getattr(module, "_speclink_sparse24_dense_weight_released", False):
            return False
        if int(getattr(module, "tp_size", 1)) != 1:
            return False
        if getattr(module, "bias", None) is not None:
            return False
        if not isinstance(
            getattr(module, "_speclink_sparse24_full_a_values", None),
            torch.Tensor,
        ):
            return False
        if not isinstance(
            getattr(module, "_speclink_sparse24_full_a_meta_e", None),
            torch.Tensor,
        ):
            return False
    if getattr(gate_up, "_speclink_gate_up_hybrid", "none") != "none":
        return False
    return mlp.act_fn.__class__.__name__ == "SiluAndMul"


def _static_sparse_mlp_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    hidden = _sparse_gate_swiglu_hidden(mlp, x)
    return _dense_down_projection(mlp.down_proj, hidden)


def _can_use_fused_batch_mlp(mlp: Any) -> bool:
    if not _FUSED_BATCH_MLP_ENABLED:
        return False
    strategy = linear_strategy()
    if strategy not in {
        "full_sparse_dense_override",
        "full_sparse_residual",
        "split_dense_sparse",
    }:
        return False
    if mlp_strategy() != "linear":
        return False

    gate_up = mlp.gate_up_proj
    down = mlp.down_proj
    gate_up_interleaved = bool(
        getattr(gate_up, "_speclink_sparse24_gate_up_interleaved", False)
    )
    routed_swiglu = bool(
        getattr(gate_up, "_speclink_sparse24_routed_swiglu", False)
    )
    if routed_swiglu != _ROUTED_SWIGLU_MLP_ENABLED:
        return False
    if _ROUTED_SWIGLU_MLP_ENABLED and strategy != "full_sparse_residual":
        return False
    if gate_up_interleaved != (
        _INLINE_SWIGLU_MLP_ENABLED or _ROUTED_SWIGLU_MLP_ENABLED
    ):
        return False
    for module in (gate_up, down):
        if not getattr(module, "_speclink_selective_dense_enabled", False):
            return False
        if getattr(module, "_speclink_selective_dense_bypass", False):
            return False
        if not getattr(module, "_speclink_selective_mixed_rows", True):
            return False
        if int(getattr(module, "tp_size", 1)) != 1:
            return False
        if getattr(module, "bias", None) is not None:
            return False
        if not isinstance(
            getattr(module, "_speclink_sparse24_full_a_values", None),
            torch.Tensor,
        ):
            return False
        if not isinstance(
            getattr(module, "_speclink_sparse24_full_a_meta_e", None),
            torch.Tensor,
        ):
            return False
        if strategy == "full_sparse_residual":
            if not isinstance(
                getattr(module, "_speclink_sparse24_residual_a_values", None),
                torch.Tensor,
            ):
                return False
            if not isinstance(
                getattr(module, "_speclink_sparse24_residual_a_meta_e", None),
                torch.Tensor,
            ):
                return False
        elif getattr(
            module,
            "_speclink_sparse24_dense_weight_released",
            False,
        ):
            return False
    return getattr(gate_up, "_speclink_gate_up_hybrid", "none") == "none"


def _fused_batch_mlp_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up = mlp.gate_up_proj
    down = mlp.down_proj
    if linear_strategy() == "full_sparse_residual":
        return torch.ops.speclink.batch_routed_residual_mlp.default(
            x,
            gate_up._speclink_sparse24_full_a_values,
            gate_up._speclink_sparse24_full_a_meta_e,
            gate_up._speclink_sparse24_residual_a_values,
            gate_up._speclink_sparse24_residual_a_meta_e,
            down._speclink_sparse24_full_a_values,
            down._speclink_sparse24_full_a_meta_e,
            down._speclink_sparse24_residual_a_values,
            down._speclink_sparse24_residual_a_meta_e,
        )
    return torch.ops.speclink.batch_routed_mlp.default(
        x,
        gate_up.weight,
        down.weight,
        gate_up._speclink_sparse24_full_a_values,
        gate_up._speclink_sparse24_full_a_meta_e,
        down._speclink_sparse24_full_a_values,
        down._speclink_sparse24_full_a_meta_e,
    )


@torch.inference_mode()
def speclink_mlp_forward(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    gate_up_proj = mlp.gate_up_proj
    down_proj = mlp.down_proj
    prepared = (
        getattr(gate_up_proj, "_speclink_selective_dense_enabled", False)
        or getattr(down_proj, "_speclink_selective_dense_enabled", False)
    )
    bypassed = (
        getattr(gate_up_proj, "_speclink_selective_dense_bypass", False)
        and getattr(down_proj, "_speclink_selective_dense_bypass", False)
    )
    if not prepared or bypassed or not token_dense_enabled():
        return _dense_mlp_forward(mlp, x)

    if _can_use_sparse_gate_dense_down(mlp, x):
        return _sparse_gate_dense_down_forward(mlp, x)
    if _can_use_routed_gate_dense_down(mlp, x):
        return _routed_gate_dense_down_forward(mlp, x)
    if _can_use_static_sparse_mlp(mlp, x):
        return _static_sparse_mlp_forward(mlp, x)
    if _can_use_fused_batch_mlp(mlp):
        return _fused_batch_mlp_forward(mlp, x)
    if getattr(
        gate_up_proj, "_speclink_sparse24_gate_up_interleaved", False
    ):
        raise RuntimeError(
            "interleaved sparse gate/up weights require the fused batch MLP path"
        )
    if mlp_strategy() == "gate_only":
        return _gate_only_mlp_forward(mlp, x)
    return _linear_mlp_forward(mlp, x)
