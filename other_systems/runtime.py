"""Lazy PyTorch bindings for the BF16 external-system ports."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import torch
from torch.sparse.semi_structured import SparseSemiStructuredTensorCUSPARSELT

from speculators.speclink._cuda_extension import load_cuda_extension
from speculators.speclink.sparse24_gemm import (
    PreparedSparse24Weight,
    sparse24_linear,
)

from .nm import NMFormat, parse_nm, split_sparta_base_residual


_ROOT = Path(__file__).resolve().parent
_BUILD_ROOT = Path(
    os.environ.get("SPECLINK_OTHER_SYSTEMS_BUILD_DIR", "temp/other_systems_build")
)


def _load(name: str, subdir: str) -> Any:
    root = _ROOT / subdir
    return load_cuda_extension(
        name=name,
        sources=(root / "torch_extension.cu",),
        required=tuple((root / "csrc").glob("*")),
        build_dir=_BUILD_ROOT / name,
        include_cutlass=False,
        verbose_env="SPECLINK_OTHER_SYSTEMS_VERBOSE_BUILD",
    )


@lru_cache(maxsize=1)
def _flash_extension() -> Any:
    return _load("speclink_flash_llm_bf16_sm120", "flash_llm_bf16")


@lru_cache(maxsize=1)
def _spinfer_extension() -> Any:
    return _load("speclink_spinfer_bf16_sm120", "spinfer_bf16")


def _validate_weight(weight: torch.Tensor, *, alignment: int) -> None:
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or weight.shape[0] % alignment
        or weight.shape[1] % 64
    ):
        raise ValueError(
            f"weight must be contiguous BF16 [N,K], N divisible by {alignment}, "
            "and K divisible by 64"
        )


def _target_device(weight: torch.Tensor, device: torch.device | str | None) -> torch.device:
    if device is not None:
        target = torch.device(device)
    elif weight.is_cuda:
        target = weight.device
    else:
        target = torch.device("cuda", torch.cuda.current_device())
    if target.type != "cuda":
        raise ValueError("external kernels require a CUDA target device")
    return target


def _check_input(x: torch.Tensor, *, n: int, k: int, device: torch.device) -> None:
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_cuda
        or not x.is_contiguous()
        or x.device != device
        or x.shape[1] != k
    ):
        raise ValueError("x must be contiguous CUDA BF16 [M,K] on the weight device")
    if x.shape[0] not in {8, 16, 32, 64, 128} and x.shape[0] % 128:
        raise ValueError("M must be 8/16/32/64/128 or a multiple of 128")
    if n <= 0:
        raise AssertionError("invalid prepared weight")


@dataclass(frozen=True)
class FlashLLMWeight:
    compressed: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, int]

    @property
    def device(self) -> torch.device:
        return self.compressed.device


@dataclass(frozen=True)
class SpInferWeight:
    values: torch.Tensor
    global_offsets: torch.Tensor
    median_offsets: torch.Tensor
    bitmap: torch.Tensor
    max_nnz: torch.Tensor
    shape: tuple[int, int]

    @property
    def device(self) -> torch.device:
        return self.values.device


@dataclass(frozen=True)
class SparTAWeight:
    base: PreparedSparse24Weight
    residual: SpInferWeight | None
    shape: tuple[int, int]
    nm_format: NMFormat

    @property
    def device(self) -> torch.device:
        return self.base.device


def prepare_flash_llm(
    weight: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> FlashLLMWeight:
    _validate_weight(weight, alignment=128)
    target = _target_device(weight, device)
    cpu_weight = weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    compressed, offsets = _flash_extension().prepare_cpu(cpu_weight)
    return FlashLLMWeight(
        compressed=compressed.to(target),
        offsets=offsets.to(target),
        shape=(weight.shape[0], weight.shape[1]),
    )


def flash_llm_linear(
    x: torch.Tensor,
    prepared: FlashLLMWeight,
    *,
    split_k: int = 1,
) -> torch.Tensor:
    if not isinstance(prepared, FlashLLMWeight):
        raise TypeError("prepared must be FlashLLMWeight")
    _check_input(x, n=prepared.shape[0], k=prepared.shape[1], device=prepared.device)
    tile_n = 256 if x.shape[0] in {8, 16, 32} else 128
    if prepared.shape[0] % tile_n:
        raise ValueError(
            f"Flash-LLM N must be divisible by {tile_n} for M={x.shape[0]}"
        )
    return _flash_extension().forward(
        x, prepared.compressed, prepared.offsets, prepared.shape[0], split_k
    )


def prepare_spinfer(
    weight: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> SpInferWeight:
    _validate_weight(weight, alignment=64)
    target = _target_device(weight, device)
    cpu_weight = weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    items = _spinfer_extension().prepare_cpu(cpu_weight)
    values, global_offsets, median_offsets, bitmap, max_nnz = (
        item.to(target) for item in items
    )
    return SpInferWeight(
        values=values,
        global_offsets=global_offsets,
        median_offsets=median_offsets,
        bitmap=bitmap,
        max_nnz=max_nnz,
        shape=(weight.shape[0], weight.shape[1]),
    )


def spinfer_linear(
    x: torch.Tensor,
    prepared: SpInferWeight,
    *,
    split_k: int = 1,
) -> torch.Tensor:
    if not isinstance(prepared, SpInferWeight):
        raise TypeError("prepared must be SpInferWeight")
    _check_input(x, n=prepared.shape[0], k=prepared.shape[1], device=prepared.device)
    return _spinfer_extension().forward(
        x,
        prepared.values,
        prepared.global_offsets,
        prepared.median_offsets,
        prepared.bitmap,
        prepared.max_nnz,
        prepared.shape[0],
        split_k,
    )


def spinfer_linear_add(
    x: torch.Tensor,
    prepared: SpInferWeight,
    output: torch.Tensor,
    *,
    split_k: int = 1,
) -> torch.Tensor:
    if not isinstance(prepared, SpInferWeight):
        raise TypeError("prepared must be SpInferWeight")
    _check_input(x, n=prepared.shape[0], k=prepared.shape[1], device=prepared.device)
    if (
        not isinstance(output, torch.Tensor)
        or output.dtype != torch.bfloat16
        or not output.is_cuda
        or not output.is_contiguous()
        or output.device != x.device
        or tuple(output.shape) != (x.shape[0], prepared.shape[0])
    ):
        raise ValueError("output must be contiguous CUDA BF16 [M,N]")
    return _spinfer_extension().forward_add(
        x,
        prepared.values,
        prepared.global_offsets,
        prepared.median_offsets,
        prepared.bitmap,
        prepared.max_nnz,
        output,
        split_k,
    )


def prepare_sparta(
    weight_nm: torch.Tensor,
    fmt: str | NMFormat,
    *,
    device: torch.device | str | None = None,
    algorithm_id: int = 0,
) -> SparTAWeight:
    fmt = parse_nm(fmt)
    _validate_weight(weight_nm, alignment=64)
    target = _target_device(weight_nm, device)
    base, residual = split_sparta_base_residual(weight_nm, fmt)
    base_cuda = base.to(target)
    sparse_base = SparseSemiStructuredTensorCUSPARSELT.from_dense(
        base_cuda.contiguous()
    )
    sparse_base.fuse_transpose_cusparselt = True
    sparse_base.alg_id_cusparselt = algorithm_id
    prepared_base = PreparedSparse24Weight(
        sparse_weight=sparse_base,
        shape=(base_cuda.shape[0], base_cuda.shape[1]),
        algorithm_id=algorithm_id,
    )
    return SparTAWeight(
        # The artifact accepts at most two nonzeros per K4, not necessarily
        # exactly two.  Construct cuSPARSELt directly instead of using
        # SpecLink's stricter exact-2:4 helper.
        base=prepared_base,
        residual=(
            None
            if int(torch.count_nonzero(residual).item()) == 0
            else prepare_spinfer(residual, device=target)
        ),
        shape=(weight_nm.shape[0], weight_nm.shape[1]),
        nm_format=fmt,
    )


def sparta_linear(
    x: torch.Tensor,
    prepared: SparTAWeight,
    *,
    residual_split_k: int = 1,
) -> torch.Tensor:
    if not isinstance(prepared, SparTAWeight):
        raise TypeError("prepared must be SparTAWeight")
    _check_input(x, n=prepared.shape[0], k=prepared.shape[1], device=prepared.device)
    # The SpInfer artifact runs the residual Sputnik kernel first and invokes
    # cuSPARSELt with beta=1.  PyTorch's BF16 cuSPARSELt binding does not expose
    # matrix beta accumulation, so the port reverses the independent operands:
    # cuSPARSELt writes first and SpInfer accumulates in its epilogue.
    output = sparse24_linear(x, prepared.base)
    if prepared.residual is None:
        return output
    return spinfer_linear_add(
        x, prepared.residual, output, split_k=residual_split_k
    )
