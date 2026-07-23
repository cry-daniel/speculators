"""Loader for the residual-complement CUTLASS HMMA.SP kernels."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

from ._cuda_extension import load_cuda_extension


@functools.lru_cache(maxsize=1)
def _extension() -> Any:
    root = Path(__file__).resolve().parents[3]
    csrc = Path(__file__).resolve().parent / "csrc"
    build_dir = Path(
        os.environ.get(
            "SPECLINK_SPARSE_RESIDUAL_CUTLASS_BUILD_DIR",
            root / "temp/torch_extensions/sparse_residual_cutlass_cuda130",
        )
    )
    return load_cuda_extension(
        name="speclink_sparse_residual_cutlass_cuda",
        sources=(
            csrc / "sparse_residual_cutlass_sparse24.cpp",
            csrc / "sparse_residual_cutlass_sparse24.cu",
        ),
        required=(
            csrc / "cutlass_dual_sparse_gemm_with_visitor.h",
            csrc / "cutlass_dual_sparse_mma_multistage.h",
            csrc / "cutlass_sparse_mma_activation_stationary.h",
            csrc / "cutlass_sparse_mma_single_smem.h",
            csrc / "cutlass_transpose_epilogue_visitor.h",
        ),
        build_dir=build_dir,
        verbose_env="SPECLINK_SPARSE_RESIDUAL_CUTLASS_VERBOSE_BUILD",
    )


__all__: list[str] = []
