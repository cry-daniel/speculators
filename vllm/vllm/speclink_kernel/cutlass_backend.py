"""Python bridge for the final CUTLASS sparse24 Tensor Core backend."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .pack import (
    Sparse24Layout,
    pack_mma_sp_operand_a,
    pattern_meta_to_ordered_nibbles,
)


_LIB: ctypes.CDLL | None = None


@contextmanager
def _temporary_sparse24_device_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_SPARSE24_DEVICE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_inline_epilogue_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_SPARSE24_INLINE_EPILOGUE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_inline_epilogue_store(store_mode: str):
    key = "SPECLINK_SPARSE24_INLINE_EPILOGUE_STORE"
    old_value = os.environ.get(key)
    os.environ[key] = store_mode
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_routed_transpose_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_SPARSE24_ROUTED_TRANSPOSE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_swiglu_epilogue_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_SPARSE24_SWIGLU_EPILOGUE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_pair_add_epilogue_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_SPARSE24_PAIR_ADD_EPILOGUE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_inline_qkv_epilogue_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_SPARSE24_QKV_EPILOGUE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_qkv_postop_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_QKV_POSTOP_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_mlp_epilogue_config(config: str | None):
    if config is None:
        yield
        return
    key = "SPECLINK_MLP_EPILOGUE_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def _temporary_dense_gemm_config(config: str | None):
    if config is None or config == "auto":
        yield
        return
    key = "SPECLINK_DENSE_GEMM_CONFIG"
    old_value = os.environ.get(key)
    os.environ[key] = config
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_dir() -> Path:
    return _repo_root() / "examples/evaluate/eval-guidellm/temp/speclink_kernel_cache"


def _source_path() -> Path:
    return Path(__file__).resolve().with_name("cutlass_sparse24_linear.cu")


def _library_path() -> Path:
    return _build_dir() / "libspeclink_sparse24_linear.so"


def _cutlass_include() -> Path:
    override = os.environ.get("SPECLINK_CUTLASS_INCLUDE")
    if override:
        return Path(override)

    spec = importlib.util.find_spec("flashinfer")
    if spec is not None and spec.origin:
        candidate = Path(spec.origin).resolve().parent / "data/cutlass/include"
        if candidate.exists():
            return candidate

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidate = (
            Path(conda_prefix)
            / "lib"
            / version
            / "site-packages"
            / "flashinfer/data/cutlass/include"
        )
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "CUTLASS headers not found. Set SPECLINK_CUTLASS_INCLUDE to "
        "flashinfer/data/cutlass/include."
    )


def _nvcc() -> Path:
    for key in ("SPECLINK_NVCC", "CUDACXX"):
        value = os.environ.get(key)
        if value:
            return Path(value)

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidate = Path(cuda_home) / "bin/nvcc"
        if candidate.exists():
            return candidate

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidate = (
            Path(conda_prefix)
            / "lib"
            / version
            / "site-packages"
            / "nvidia/cu13/bin/nvcc"
        )
        if candidate.exists():
            return candidate

    found = shutil.which("nvcc")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "nvcc not found. Set SPECLINK_NVCC, CUDACXX, or CUDA_HOME to the "
        "conda CUDA 13.0 toolchain."
    )


def _cuda_host_cxx() -> Path | None:
    for key in ("SPECLINK_CUDAHOSTCXX", "CUDAHOSTCXX"):
        value = os.environ.get(key)
        if value:
            return Path(value)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin/x86_64-conda-linux-gnu-g++"
        if candidate.exists():
            return candidate
    return None


def _ensure_library() -> Path:
    source = _source_path()
    library = _library_path()
    if library.exists() and library.stat().st_mtime >= source.stat().st_mtime:
        return library

    _build_dir().mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_nvcc()),
        "-std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-arch=sm_120",
        f"-I{_cutlass_include()}",
        str(source),
        "-o",
        str(library),
    ]
    host_cxx = _cuda_host_cxx()
    if host_cxx is not None:
        cmd[1:1] = ["-ccbin", str(host_cxx)]
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (_build_dir() / "compile_output.txt").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build CUTLASS sparse24 backend:\n" + result.stdout
        )
    return library


def _load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(str(_ensure_library()))
    device_fn = lib.sparse24_cutlass_device_gemm_f16_stream
    device_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    device_fn.restype = ctypes.c_int
    device_strided_input_fn = (
        lib.sparse24_cutlass_device_strided_input_gemm_f16_stream
    )
    device_strided_input_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    device_strided_input_fn.restype = ctypes.c_int
    device_splitk_fn = lib.sparse24_cutlass_device_splitk_gemm_f16_stream
    device_splitk_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    device_splitk_fn.restype = ctypes.c_int
    device_splitk_indexed_add_fn = (
        lib.sparse24_cutlass_device_splitk_indexed_add_gemm_f16_stream
    )
    device_splitk_indexed_add_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    device_splitk_indexed_add_fn.restype = ctypes.c_int
    signal_ready_fn = lib.sparse24_cutlass_signal_ready_f16_stream
    signal_ready_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    signal_ready_fn.restype = ctypes.c_int
    gather_gemm_fn = lib.sparse24_cutlass_gather_gemm_f16_stream
    gather_gemm_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    gather_gemm_fn.restype = ctypes.c_int
    paired_persistent_fn = (
        lib.sparse24_cutlass_paired_persistent_gemm_f16_stream
    )
    paired_persistent_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_persistent_fn.restype = ctypes.c_int
    paired_gather_residual_fn = (
        lib.sparse24_cutlass_paired_gather_residual_gemm_f16_stream
    )
    paired_gather_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_gather_residual_fn.restype = ctypes.c_int
    paired_gather_residual_qkv_fn = (
        lib.sparse24_cutlass_paired_gather_residual_qkv_gemm_f16_stream
    )
    paired_gather_residual_qkv_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_gather_residual_qkv_fn.restype = ctypes.c_int
    paired_fused_routed_qkv_epilogue_fn = (
        lib.sparse24_cutlass_paired_fused_routed_qkv_epilogue_gemm_f16_stream
    )
    paired_fused_routed_qkv_epilogue_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_fused_routed_qkv_epilogue_fn.restype = ctypes.c_int
    paired_finalize_residual_fn = (
        lib.sparse24_cutlass_paired_finalize_residual_gemm_f16_stream
    )
    paired_finalize_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_finalize_residual_fn.restype = ctypes.c_int
    paired_finalize_qkv_fn = (
        lib.sparse24_cutlass_paired_finalize_qkv_gemm_f16_stream
    )
    paired_finalize_qkv_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_finalize_qkv_fn.restype = ctypes.c_int
    paired_inplace_residual_fn = (
        lib.sparse24_cutlass_paired_inplace_residual_gemm_f16_stream
    )
    paired_inplace_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_inplace_residual_fn.restype = ctypes.c_int
    paired_routed_swiglu_fn = (
        lib.sparse24_cutlass_paired_persistent_routed_swiglu_f16_stream
    )
    paired_routed_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_routed_swiglu_fn.restype = ctypes.c_int
    paired_gather_routed_swiglu_fn = (
        lib.sparse24_cutlass_paired_gather_routed_swiglu_f16_stream
    )
    paired_gather_routed_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_gather_routed_swiglu_fn.restype = ctypes.c_int
    paired_fused_routed_swiglu_fn = (
        lib.sparse24_cutlass_paired_fused_routed_swiglu_f16_stream
    )
    paired_fused_routed_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_fused_routed_swiglu_fn.restype = ctypes.c_int
    fused_mixed_mlp_fn = lib.sparse24_cutlass_fused_mixed_mlp_f16_stream
    fused_mixed_mlp_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fused_mixed_mlp_fn.restype = ctypes.c_int
    paired_self_contained_routed_swiglu_fn = (
        lib.sparse24_cutlass_paired_self_contained_routed_swiglu_f16_stream
    )
    paired_self_contained_routed_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_self_contained_routed_swiglu_fn.restype = ctypes.c_int
    paired_self_contained_exact_down_fn = (
        lib.sparse24_cutlass_paired_self_contained_exact_down_f16_stream
    )
    paired_self_contained_exact_down_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    paired_self_contained_exact_down_fn.restype = ctypes.c_int
    gate_dense_down_pipeline_fn = (
        lib.sparse24_cutlass_gate_dense_down_pipeline_f16_stream
    )
    gate_dense_down_pipeline_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    gate_dense_down_pipeline_fn.restype = ctypes.c_int
    gate_sparse_down_pipeline_fn = (
        lib.sparse24_cutlass_gate_sparse_down_pipeline_f16_stream
    )
    gate_sparse_down_pipeline_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    gate_sparse_down_pipeline_fn.restype = ctypes.c_int
    inline_transpose_fn = lib.sparse24_cutlass_inline_transpose_gemm_f16_stream
    inline_transpose_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_transpose_fn.restype = ctypes.c_int
    inline_routed_transpose_fn = (
        lib.sparse24_cutlass_inline_routed_transpose_gemm_f16_stream
    )
    inline_routed_transpose_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_routed_transpose_fn.restype = ctypes.c_int
    routed_residual_epilogue_fn = (
        lib.sparse24_cutlass_routed_residual_epilogue_gemm_f16_stream
    )
    routed_residual_epilogue_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_residual_epilogue_fn.restype = ctypes.c_int
    inline_indexed_transpose_fn = (
        lib.sparse24_cutlass_inline_indexed_transpose_gemm_f16_stream
    )
    inline_indexed_transpose_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_indexed_transpose_fn.restype = ctypes.c_int
    routed_exact_linear_fn = (
        lib.sparse24_cutlass_routed_exact_linear_f16_stream
    )
    routed_exact_linear_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_exact_linear_fn.restype = ctypes.c_int
    heterogeneous_linear_fn = (
        lib.sparse24_cutlass_heterogeneous_linear_f16_stream
    )
    heterogeneous_linear_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    heterogeneous_linear_fn.restype = ctypes.c_int
    heterogeneous_swiglu_fn = (
        lib.sparse24_cutlass_heterogeneous_swiglu_f16_stream
    )
    heterogeneous_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    heterogeneous_swiglu_fn.restype = ctypes.c_int
    full_sparse_dense_override_swiglu_fn = (
        lib.sparse24_cutlass_full_sparse_dense_override_swiglu_f16_stream
    )
    full_sparse_dense_override_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    full_sparse_dense_override_swiglu_fn.restype = ctypes.c_int
    full_sparse_dense_override_linear_fn = (
        lib.sparse24_cutlass_full_sparse_dense_override_linear_f16_stream
    )
    full_sparse_dense_override_linear_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    full_sparse_dense_override_linear_fn.restype = ctypes.c_int
    heterogeneous_component_fn = (
        lib.sparse24_cutlass_heterogeneous_component_f16_stream
    )
    heterogeneous_component_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    heterogeneous_component_fn.restype = ctypes.c_int
    routed_exact_swiglu_fn = (
        lib.sparse24_cutlass_routed_exact_swiglu_f16_stream
    )
    routed_exact_swiglu_fn.argtypes = routed_exact_linear_fn.argtypes
    routed_exact_swiglu_fn.restype = ctypes.c_int
    grouped_owner_linear_fn = (
        lib.sparse24_cutlass_grouped_owner_linear_f16_stream
    )
    grouped_owner_linear_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    grouped_owner_linear_fn.restype = ctypes.c_int
    grouped_owner_swiglu_fn = (
        lib.sparse24_cutlass_grouped_owner_swiglu_f16_stream
    )
    grouped_owner_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    grouped_owner_swiglu_fn.restype = ctypes.c_int
    grouped_owner_qkv_fn = (
        lib.sparse24_cutlass_grouped_owner_qkv_f16_stream
    )
    grouped_owner_qkv_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    grouped_owner_qkv_fn.restype = ctypes.c_int
    inline_swiglu_fn = lib.sparse24_cutlass_inline_swiglu_gemm_f16_stream
    inline_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_swiglu_fn.restype = ctypes.c_int
    inline_routed_swiglu_fn = (
        lib.sparse24_cutlass_inline_routed_swiglu_gemm_f16_stream
    )
    inline_routed_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_routed_swiglu_fn.restype = ctypes.c_int
    inline_routed_approx_swiglu_fn = (
        lib.sparse24_cutlass_inline_routed_approx_swiglu_gemm_f16_stream
    )
    inline_routed_approx_swiglu_fn.argtypes = inline_routed_swiglu_fn.argtypes
    inline_routed_approx_swiglu_fn.restype = ctypes.c_int
    residual_correction_swiglu_fn = (
        lib.sparse24_cutlass_residual_correction_swiglu_gemm_f16_stream
    )
    residual_correction_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    residual_correction_swiglu_fn.restype = ctypes.c_int
    residual_delta_swiglu_fn = (
        lib.sparse24_cutlass_residual_delta_swiglu_gemm_f16_stream
    )
    residual_delta_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    residual_delta_swiglu_fn.restype = ctypes.c_int
    indexed_swiglu_fn = lib.sparse24_cutlass_indexed_swiglu_gemm_f16_stream
    indexed_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    indexed_swiglu_fn.restype = ctypes.c_int
    dual_swiglu_fn = lib.sparse24_cutlass_dual_swiglu_gemm_f16_stream
    dual_swiglu_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    dual_swiglu_fn.restype = ctypes.c_int
    inline_pair_add_fn = lib.sparse24_cutlass_inline_pair_add_gemm_f16_stream
    inline_pair_add_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_pair_add_fn.restype = ctypes.c_int
    inline_swiglu_transposed_fn = (
        lib.sparse24_cutlass_inline_swiglu_transposed_gemm_f16_stream
    )
    inline_swiglu_transposed_fn.argtypes = inline_swiglu_fn.argtypes
    inline_swiglu_transposed_fn.restype = ctypes.c_int
    inline_qkv_fn = lib.sparse24_cutlass_inline_qkv_postop_gemm_f16_stream
    inline_qkv_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    inline_qkv_fn.restype = ctypes.c_int
    device_b_row_fn = lib.sparse24_cutlass_device_gemm_b_row_f16_stream
    device_b_row_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    device_b_row_fn.restype = ctypes.c_int
    dense_fn = lib.dense_cutlass_device_gemm_f16_stream
    dense_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    dense_fn.restype = ctypes.c_int
    dense_f16_accum_fn = lib.dense_cutlass_device_gemm_f16_accum_f16_stream
    dense_f16_accum_fn.argtypes = dense_fn.argtypes
    dense_f16_accum_fn.restype = ctypes.c_int
    dense_weight_t_fn = lib.dense_cutlass_weight_t_gemm_f16_stream
    dense_weight_t_fn.argtypes = dense_fn.argtypes
    dense_weight_t_fn.restype = ctypes.c_int
    dense_weight_t_f16_accum_fn = (
        lib.dense_cutlass_weight_t_gemm_f16_accum_f16_stream
    )
    dense_weight_t_f16_accum_fn.argtypes = dense_fn.argtypes
    dense_weight_t_f16_accum_fn.restype = ctypes.c_int
    dense_weight_t_add_fn = lib.dense_cutlass_weight_t_gemm_add_f16_stream
    dense_weight_t_add_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    dense_weight_t_add_fn.restype = ctypes.c_int
    dense_weight_t_add_f16_accum_fn = (
        lib.dense_cutlass_weight_t_gemm_add_f16_accum_f16_stream
    )
    dense_weight_t_add_f16_accum_fn.argtypes = dense_weight_t_add_fn.argtypes
    dense_weight_t_add_f16_accum_fn.restype = ctypes.c_int
    dense_simt_fn = lib.dense_cutlass_simt_weight_t_gemm_f16_stream
    dense_simt_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    dense_simt_fn.restype = ctypes.c_int
    silu_and_mul_transposed_fn = (
        lib.sparse24_cutlass_silu_and_mul_transposed_f16_stream
    )
    silu_and_mul_transposed_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    silu_and_mul_transposed_fn.restype = ctypes.c_int
    silu_and_mul_contiguous_fn = (
        lib.sparse24_cutlass_silu_and_mul_transposed_to_contiguous_f16_stream
    )
    silu_and_mul_contiguous_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    silu_and_mul_contiguous_fn.restype = ctypes.c_int
    routed_swiglu_correction_fn = (
        lib.sparse24_cutlass_routed_swiglu_correction_f16_stream
    )
    routed_swiglu_correction_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_swiglu_correction_fn.restype = ctypes.c_int
    routed_swiglu_correction_gather_fn = (
        lib.sparse24_cutlass_routed_swiglu_correction_gather_f16_stream
    )
    routed_swiglu_correction_gather_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_swiglu_correction_gather_fn.restype = ctypes.c_int
    routed_swiglu_delta_fn = (
        lib.sparse24_cutlass_routed_swiglu_delta_f16_stream
    )
    routed_swiglu_delta_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_swiglu_delta_fn.restype = ctypes.c_int
    routed_linear_correction_fn = (
        lib.sparse24_cutlass_routed_linear_correction_f16_stream
    )
    routed_linear_correction_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_linear_correction_fn.restype = ctypes.c_int
    routed_swiglu_correction_transposed_fn = (
        lib.sparse24_cutlass_routed_swiglu_correction_transposed_f16_stream
    )
    routed_swiglu_correction_transposed_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_swiglu_correction_transposed_fn.restype = ctypes.c_int
    routed_swiglu_correction_transpose_tiled_fn = (
        lib.sparse24_cutlass_routed_swiglu_correction_transpose_tiled_f16_stream
    )
    routed_swiglu_correction_transpose_tiled_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    routed_swiglu_correction_transpose_tiled_fn.restype = ctypes.c_int
    transpose_output_fn = lib.sparse24_cutlass_transpose_output_f16_stream
    transpose_output_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    transpose_output_fn.restype = ctypes.c_int
    transpose_add_routed_residual_fn = (
        lib.sparse24_cutlass_transpose_add_routed_residual_f16_stream
    )
    transpose_add_routed_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    transpose_add_routed_residual_fn.restype = ctypes.c_int
    transpose_add_routed_residual_to_residual_fn = (
        lib.sparse24_cutlass_transpose_add_routed_residual_to_residual_f16_stream
    )
    transpose_add_routed_residual_to_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    transpose_add_routed_residual_to_residual_fn.restype = ctypes.c_int
    transpose_add_routed_residual_rmsnorm_fn = (
        lib.sparse24_cutlass_transpose_add_routed_residual_rmsnorm_f16_stream
    )
    transpose_add_routed_residual_rmsnorm_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    transpose_add_routed_residual_rmsnorm_fn.restype = ctypes.c_int
    transpose_add_routed_splitk_residual_fn = (
        lib.sparse24_cutlass_transpose_add_routed_splitk_residual_f16_stream
    )
    transpose_add_routed_splitk_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    transpose_add_routed_splitk_residual_fn.restype = ctypes.c_int
    transpose_add_residual_fn = (
        lib.sparse24_cutlass_transpose_add_residual_f16_stream
    )
    transpose_add_residual_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    transpose_add_residual_fn.restype = ctypes.c_int
    transpose_add_rmsnorm_fn = (
        lib.sparse24_cutlass_transpose_add_rmsnorm_f16_stream
    )
    transpose_add_rmsnorm_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    transpose_add_rmsnorm_fn.restype = ctypes.c_int
    qkv_transpose_rmsnorm_fn = (
        lib.sparse24_cutlass_qkv_transpose_rmsnorm_f16_stream
    )
    qkv_transpose_rmsnorm_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    qkv_transpose_rmsnorm_fn.restype = ctypes.c_int
    qkv_transpose_postop_fn = (
        lib.sparse24_cutlass_qkv_transpose_postop_f16_stream
    )
    qkv_transpose_postop_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    qkv_transpose_postop_fn.restype = ctypes.c_int
    qkv_routed_residual_postop_fn = (
        lib.sparse24_cutlass_qkv_transpose_add_routed_residual_postop_f16_stream
    )
    qkv_routed_residual_postop_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    qkv_routed_residual_postop_fn.restype = ctypes.c_int
    qkv_rowmajor_routed_residual_postop_fn = (
        lib.sparse24_cutlass_qkv_add_routed_residual_postop_inplace_f16_stream
    )
    qkv_rowmajor_routed_residual_postop_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    qkv_rowmajor_routed_residual_postop_fn.restype = ctypes.c_int
    qkv_rowmajor_routed_residual_cache_postop_fn = (
        lib.sparse24_cutlass_qkv_add_routed_residual_postop_cache_inplace_f16_stream
    )
    qkv_rowmajor_routed_residual_cache_postop_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    qkv_rowmajor_routed_residual_cache_postop_fn.restype = ctypes.c_int
    qkv_rmsnorm_inplace_fn = (
        lib.sparse24_cutlass_qkv_rmsnorm_inplace_f16_stream
    )
    qkv_rmsnorm_inplace_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    qkv_rmsnorm_inplace_fn.restype = ctypes.c_int
    mixed_override_fn = lib.sparse24_cutlass_mixed_dense_override_f16_stream
    mixed_override_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    mixed_override_fn.restype = ctypes.c_int
    add_prefix_fn = lib.sparse24_cutlass_add_prefix_strided_f16_stream
    add_prefix_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    add_prefix_fn.restype = ctypes.c_int
    add_indexed_fn = lib.sparse24_cutlass_add_indexed_rows_strided_f16_stream
    add_indexed_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    add_indexed_fn.restype = ctypes.c_int
    add_indexed_contiguous_fn = (
        lib.sparse24_cutlass_add_indexed_rows_contiguous_f16_stream
    )
    add_indexed_contiguous_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    add_indexed_contiguous_fn.restype = ctypes.c_int
    add_indexed_transposed_to_contiguous_fn = (
        lib.sparse24_cutlass_add_indexed_rows_transposed_to_contiguous_f16_stream
    )
    add_indexed_transposed_to_contiguous_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    add_indexed_transposed_to_contiguous_fn.restype = ctypes.c_int
    sub_indexed_contiguous_fn = (
        lib.sparse24_cutlass_sub_indexed_rows_contiguous_f16_stream
    )
    sub_indexed_contiguous_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    sub_indexed_contiguous_fn.restype = ctypes.c_int
    gather_rows_fn = lib.sparse24_cutlass_gather_rows_f16_stream
    gather_rows_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    gather_rows_fn.restype = ctypes.c_int
    gather_rows_strided_fn = lib.sparse24_cutlass_gather_rows_strided_f16_stream
    gather_rows_strided_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    gather_rows_strided_fn.restype = ctypes.c_int
    partition_rows_fn = lib.sparse24_cutlass_partition_rows_f16_stream
    partition_rows_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    partition_rows_fn.restype = ctypes.c_int
    merge_rows_fn = lib.sparse24_cutlass_merge_rows_f16_stream
    merge_rows_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    merge_rows_fn.restype = ctypes.c_int
    copy_indexed_fn = lib.sparse24_cutlass_copy_indexed_rows_strided_f16_stream
    copy_indexed_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    copy_indexed_fn.restype = ctypes.c_int
    copy_indexed_contiguous_fn = (
        lib.sparse24_cutlass_copy_indexed_rows_contiguous_f16_stream
    )
    copy_indexed_contiguous_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    copy_indexed_contiguous_fn.restype = ctypes.c_int
    copy_indexed_rowmajor_fn = (
        lib.sparse24_cutlass_copy_indexed_rows_rowmajor_f16_stream
    )
    copy_indexed_rowmajor_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    copy_indexed_rowmajor_fn.restype = ctypes.c_int
    _LIB = lib
    return lib


def _to_n_major(
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    layout: Sparse24Layout,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = K // 4
    if layout == "k_major":
        if values.ndim != 3 or values.shape[-1] != 2 or values.shape[0] != groups:
            raise ValueError("k_major values must have shape [K / 4, N, 2]")
        if meta.shape != values.shape[:2]:
            raise ValueError("k_major values/meta shape mismatch")
        return values.permute(1, 0, 2).contiguous(), meta.t().contiguous()
    if layout == "n_major":
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError("n_major values must have shape [N, K / 4, 2]")
        if values.shape[1] != groups or meta.shape != values.shape[:2]:
            raise ValueError("n_major values/meta shape mismatch")
        return values.contiguous(), meta.contiguous()
    raise ValueError(f"unsupported layout {layout!r}")


def _pack_device_gemm_e(meta_n: torch.Tensor) -> torch.Tensor:
    """Pack n-major pattern IDs into CUTLASS device SparseGemm E layout."""

    if meta_n.dtype != torch.uint8:
        raise ValueError(f"meta dtype must be torch.uint8, got {meta_n.dtype}")
    if meta_n.ndim != 2:
        raise ValueError(f"meta_n must be [N, K / 4], got {tuple(meta_n.shape)}")
    N, groups = meta_n.shape
    if N % 32 != 0:
        raise ValueError(f"CUTLASS SparseGemm requires N divisible by 32, got {N}")
    if groups % 8 != 0:
        raise ValueError(f"K / 4 must be divisible by 8, got {groups}")

    words = groups // 8
    ordered = pattern_meta_to_ordered_nibbles(meta_n)
    chunks = ordered.reshape(N, words, 8).to(torch.int64)
    shifts = torch.arange(8, device=meta_n.device, dtype=torch.int64) * 4
    packed = (chunks << shifts).sum(dim=-1)
    low = (packed & 0xFFFF).to(torch.int16)
    high = ((packed >> 16) & 0xFFFF).to(torch.int16)

    rows = torch.arange(N, device=meta_n.device, dtype=torch.int64)
    row_in_tile = rows % 32
    row_block = row_in_tile // 8
    base = (
        (rows // 32) * 64
        + (row_in_tile % 8) * 8
        + (row_block % 2)
        + (row_block // 2) * 4
    )
    word_offsets = torch.arange(words, device=meta_n.device, dtype=torch.int64) * (
        N * 2
    )
    low_offsets = base[:, None] + word_offsets[None, :]
    high_offsets = low_offsets + 2

    meta_e_i16 = torch.empty(N * words * 2, device=meta_n.device, dtype=torch.int16)
    meta_e_i16[low_offsets.reshape(-1)] = low.reshape(-1)
    meta_e_i16[high_offsets.reshape(-1)] = high.reshape(-1)
    return meta_e_i16.view(torch.uint16).contiguous()


def prepare_cutlass_sparse24_device_gemm(
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    layout: Sparse24Layout = "k_major",
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare compressed weights for the final CUTLASS device SparseGemm path."""

    if not values.is_cuda or not meta.is_cuda:
        raise ValueError("values and meta must be CUDA tensors")
    if values.dtype != torch.float16:
        raise ValueError("CUTLASS SparseGemm currently supports fp16 values only")
    if meta.dtype != torch.uint8:
        raise ValueError(f"meta dtype must be torch.uint8, got {meta.dtype}")
    if K % 64 != 0:
        raise ValueError(f"CUTLASS SparseGemm requires K divisible by 64, got {K}")

    values_n, meta_n = _to_n_major(values, meta, layout=layout, K=K)
    a_values, _a_meta_words = pack_mma_sp_operand_a(values_n, meta_n)
    a_meta_e = _pack_device_gemm_e(meta_n)
    return a_values.contiguous(), a_meta_e


def prepare_cutlass_sparse24_gate_up_swiglu(
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    layout: Sparse24Layout = "k_major",
    K: int,
    channels_per_half_tile: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave gate/up output tiles before CUTLASS sparse prepacking."""

    values_n, meta_n = _to_n_major(values, meta, layout=layout, K=K)
    output_size = int(values_n.shape[0])
    channels_per_half_tile = int(channels_per_half_tile)
    if channels_per_half_tile not in (64, 128):
        raise ValueError("channels_per_half_tile must be 64 or 128")
    tile_channels = 2 * channels_per_half_tile
    if output_size % tile_channels != 0:
        raise ValueError(
            "fused SwiGLU sparse epilogue requires output size divisible by "
            f"{tile_channels}"
        )
    hidden_size = output_size // 2
    if hidden_size % channels_per_half_tile != 0:
        raise ValueError(
            "fused SwiGLU sparse epilogue requires hidden size divisible by "
            f"{channels_per_half_tile}"
        )
    gate_indices = torch.arange(
        hidden_size, device=values_n.device, dtype=torch.int64
    ).reshape(-1, channels_per_half_tile)
    output_order = torch.cat(
        (gate_indices, gate_indices + hidden_size), dim=1
    ).reshape(-1)
    return prepare_cutlass_sparse24_device_gemm(
        values_n.index_select(0, output_order),
        meta_n.index_select(0, output_order),
        layout="n_major",
        K=K,
    )


def prepare_cutlass_sparse24_pair_add(
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    layout: Sparse24Layout = "k_major",
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave full/residual 128-channel tiles for pair-add epilogues."""

    return prepare_cutlass_sparse24_gate_up_swiglu(
        values,
        meta,
        layout=layout,
        K=K,
    )


def sparse24_cutlass_device_gemm(
    X: torch.Tensor,
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    layout: Sparse24Layout = "k_major",
    contiguous_output: bool = False,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    pad_m_multiple: int | None = None,
    device_config: str | None = None,
) -> torch.Tensor:
    """Compute ``Y = X @ W24`` through the final CUTLASS SparseGemm backend."""

    a_values, a_meta_e = prepare_cutlass_sparse24_device_gemm(
        values, meta, layout=layout, K=X.shape[1]
    )
    return sparse24_cutlass_device_gemm_prepacked(
        X,
        a_values,
        a_meta_e,
        contiguous_output=contiguous_output,
        out=out,
        workspace=workspace,
        pad_m_multiple=pad_m_multiple,
        device_config=device_config,
    )


def sparse24_cutlass_device_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    *,
    contiguous_output: bool = False,
    input_transposed: bool = False,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    pad_m_multiple: int | None = None,
    device_config: str | None = None,
) -> torch.Tensor:
    """Compute ``Y = X @ W24`` with prepacked CUTLASS SparseGemm inputs."""

    if not X.is_cuda or not a_values.is_cuda or not a_meta_e.is_cuda:
        raise ValueError("X, a_values, and a_meta_e must be CUDA tensors")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("CUTLASS SparseGemm currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = X.shape
    if K % 64 != 0:
        raise ValueError(f"CUTLASS SparseGemm requires K divisible by 64, got {K}")
    N = a_values.shape[0]
    if N % 32 != 0:
        raise ValueError(f"CUTLASS SparseGemm requires N divisible by 32, got {N}")
    if a_values.shape != (N, K // 2):
        raise ValueError(f"a_values must have shape [N, K / 2], got {tuple(a_values.shape)}")
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(
            f"a_meta_e must have {N * (K // 16)} elements, got {a_meta_e.numel()}"
        )
    if out is not None:
        if not out.is_cuda or out.device != X.device or out.dtype != torch.float16:
            raise ValueError("out must be a CUDA fp16 tensor on the same device as X")
        if out.ndim != 2:
            raise ValueError(f"out must be rank-2, got {tuple(out.shape)}")
    if workspace is not None:
        if not workspace.is_cuda or workspace.device != X.device or workspace.dtype != torch.float16:
            raise ValueError(
                "workspace must be a CUDA fp16 tensor on the same device as X"
            )
        if workspace.ndim != 2:
            raise ValueError(f"workspace must be rank-2, got {tuple(workspace.shape)}")

    M_orig = M
    if pad_m_multiple is None:
        pad_m_multiple = int(os.environ.get("SPECLINK_SPARSE24_PAD_M_MULTIPLE", "8"))
    if pad_m_multiple < 8 or pad_m_multiple % 8 != 0:
        raise ValueError(
            f"pad_m_multiple must be a positive multiple of 8, got {pad_m_multiple}"
        )
    M_run = ((M + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple
    input_ld = 0
    if input_transposed:
        if M_run != M:
            raise ValueError("transposed sparse input requires M to be unpadded")
        if tuple(X.stride()) != (1, M):
            raise ValueError(
                f"transposed sparse input must have stride {(1, M)}, got {tuple(X.stride())}"
            )
        Xc = X
        input_ld = M
    elif M_run != M:
        Xc = torch.empty((M_run, K), device=X.device, dtype=torch.float16)
        Xc[:M].copy_(X)
        Xc[M:].zero_()
    else:
        Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()

    if contiguous_output:
        if out is not None:
            if tuple(out.shape) != (M_run, N):
                raise ValueError(f"out must have shape {(M_run, N)}, got {tuple(out.shape)}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous for contiguous output")
            Y_run = out
        else:
            Y_run = torch.empty((M_run, N), device=X.device, dtype=torch.float16)
        if workspace is not None:
            if tuple(workspace.shape) != (N, M_run):
                raise ValueError(
                    f"workspace must have shape {(N, M_run)}, got {tuple(workspace.shape)}"
                )
            if not workspace.is_contiguous():
                raise ValueError("workspace must be contiguous")
            C_tmp = workspace
        else:
            C_tmp = torch.empty((N, M_run), device=X.device, dtype=torch.float16)
        y_ptr = ctypes.c_void_p(Y_run.data_ptr())
    else:
        if workspace is not None:
            raise ValueError("workspace is not used for non-contiguous view output")
        if out is not None:
            if tuple(out.shape) != (M_run, N):
                raise ValueError(f"out must have shape {(M_run, N)}, got {tuple(out.shape)}")
            if tuple(out.stride()) != (1, M_run):
                raise ValueError(
                    f"out must have stride {(1, M_run)}, got {tuple(out.stride())}"
                )
            Y_run = out
        else:
            Y_run = torch.empty_strided(
                (M_run, N), (1, M_run), device=X.device, dtype=torch.float16
            )
        C_tmp = Y_run
        y_ptr = ctypes.c_void_p()

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_sparse24_device_config(device_config):
        if input_transposed:
            ret = lib.sparse24_cutlass_device_gemm_b_row_f16_stream(
                ctypes.c_void_p(Xc.data_ptr()),
                ctypes.c_void_p(a_values.data_ptr()),
                ctypes.c_void_p(a_meta_e.data_ptr()),
                ctypes.c_void_p(C_tmp.data_ptr()),
                y_ptr,
                ctypes.c_int(M_run),
                ctypes.c_int(K),
                ctypes.c_int(N),
                ctypes.c_int(input_ld),
                ctypes.c_void_p(stream),
            )
        else:
            ret = lib.sparse24_cutlass_device_gemm_f16_stream(
                ctypes.c_void_p(Xc.data_ptr()),
                ctypes.c_void_p(a_values.data_ptr()),
                ctypes.c_void_p(a_meta_e.data_ptr()),
                ctypes.c_void_p(C_tmp.data_ptr()),
                y_ptr,
                ctypes.c_int(M_run),
                ctypes.c_int(K),
                ctypes.c_int(N),
                ctypes.c_void_p(stream),
            )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24_cutlass_device_gemm_f16 failed with code {int(ret)}"
        )
    if M_run == M_orig:
        return Y_run
    return Y_run[:M_orig]


def sparse24_cutlass_device_strided_input_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GEMM on a contiguous K slice with a larger row stride."""

    if any(not tensor.is_cuda for tensor in (X, a_values, a_meta_e)):
        raise ValueError("strided-input sparse GEMM inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in (a_values, a_meta_e)):
        raise ValueError("strided-input sparse GEMM inputs must share one device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("strided-input sparse GEMM currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("strided-input sparse metadata must be uint16/int16")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("strided-input sparse GEMM expects rank-2 X/values")
    rows, K = map(int, X.shape)
    N = int(a_values.shape[0])
    ldb = int(X.stride(0))
    if rows <= 0 or rows % 8 or X.stride(1) != 1 or ldb < K:
        raise ValueError(
            "X must have positive rows divisible by 8, stride(1)=1, and "
            "stride(0)>=K"
        )
    if K % 64 or N % 256:
        raise ValueError("strided-input sparse GEMM requires K % 64 and N % 256")
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(f"a_values must have shape {(N, K // 2)}")
    if int(a_values.stride(1)) != 1 or int(a_values.stride(0)) < K // 2:
        raise ValueError(
            "a_values must have stride(1)=1 and stride(0)>=K/2"
        )
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError("a_meta_e has an invalid number of elements")
    if not a_meta_e.is_contiguous():
        raise ValueError("a_meta_e must be contiguous")
    if out is None:
        output = torch.empty_strided(
            (rows, N), (1, rows), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, N)
            or tuple(output.stride()) != (1, rows)
        ):
            raise ValueError(
                f"out must be CUDA fp16 shape {(rows, N)} with stride {(1, rows)}"
            )

    values = a_values
    metadata = a_meta_e
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_device_strided_input_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(values.data_ptr()),
        ctypes.c_void_p(metadata.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(ldb),
        ctypes.c_int(values.stride(0)),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 CUTLASS strided-input GEMM failed with code " f"{int(ret)}"
        )
    return output


def sparse24_cutlass_device_splitk_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    *,
    split_k_slices: int,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one split-K sparse GEMM launch for a small row count.

    A supplied integer workspace must be zero on its first use. The kernel
    resets every tile counter before returning, including under graph replay.
    """

    tensors = (X, a_values, a_meta_e)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("split-K sparse GEMM inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("split-K sparse GEMM inputs must share one device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("split-K sparse GEMM currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("split-K sparse metadata must be uint16/int16")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("split-K sparse GEMM expects rank-2 X/values")
    rows, K = map(int, X.shape)
    N = int(a_values.shape[0])
    split_k_slices = int(split_k_slices)
    if split_k_slices not in (2, 4, 8):
        raise ValueError("split_k_slices must be 2, 4, or 8")
    if (
        rows <= 0
        or rows % 8
        or K % (64 * split_k_slices)
        or N % 256
    ):
        raise ValueError(
            "split-K sparse GEMM requires rows % 8, N % 256, and "
            "K % (64 * split_k_slices) == 0"
        )
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(f"a_values must have shape {(N, K // 2)}")
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError("a_meta_e has an invalid number of elements")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("split-K sparse GEMM inputs must be contiguous")

    if out is None:
        output = torch.empty_strided(
            (rows, N), (1, rows), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, N)
            or tuple(output.stride()) != (1, rows)
        ):
            raise ValueError(
                f"out must be CUDA fp16 shape {(rows, N)} with "
                f"stride {(1, rows)}"
            )

    required_workspace_ints = ((N + 255) // 256) * ((rows + 31) // 32)
    if workspace is None:
        split_workspace = torch.zeros(
            required_workspace_ints,
            device=X.device,
            dtype=torch.int32,
        )
    else:
        split_workspace = workspace
        if (
            not split_workspace.is_cuda
            or split_workspace.device != X.device
            or split_workspace.dtype != torch.int32
            or split_workspace.ndim != 1
            or not split_workspace.is_contiguous()
            or split_workspace.numel() < required_workspace_ints
        ):
            raise ValueError(
                "workspace must be contiguous CUDA int32 with at least "
                f"{required_workspace_ints} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_device_splitk_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(split_workspace.data_ptr()),
        ctypes.c_int(split_workspace.numel()),
        ctypes.c_int(rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(split_k_slices),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 CUTLASS split-K GEMM failed with code " f"{int(ret)}"
        )
    return output


def sparse24_cutlass_signal_ready_(ready_state: torch.Tensor) -> torch.Tensor:
    """Signal that a concurrent full-output producer has completed."""

    if (
        not ready_state.is_cuda
        or ready_state.dtype != torch.int32
        or ready_state.ndim != 1
        or not ready_state.is_contiguous()
        or ready_state.numel() < 2
    ):
        raise ValueError(
            "ready_state must be contiguous rank-1 CUDA int32 with at least "
            "two elements"
        )
    lib = _load_library()
    stream = torch.cuda.current_stream(ready_state.device).cuda_stream
    ret = lib.sparse24_cutlass_signal_ready_f16_stream(
        ctypes.c_void_p(ready_state.data_ptr()),
        ctypes.c_int(ready_state.numel()),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 CUTLASS ready signal failed with code " f"{int(ret)}"
        )
    return ready_state


def sparse24_cutlass_device_splitk_indexed_add_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    full_output: torch.Tensor,
    *,
    split_k_slices: int,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    ready_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run split-K residual Down and add final rows into a full output.

    The full-output stream must call :func:`sparse24_cutlass_signal_ready_`
    after producing ``full_output``. Supplied workspaces must be zero on first
    use; the residual kernel resets them before returning.
    """

    tensors = (X, a_values, a_meta_e, dense_rows, full_output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("split-K indexed-add inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("split-K indexed-add inputs must share one device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("split-K indexed-add GEMM currently supports fp16")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("split-K indexed-add metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 tensor")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("split-K indexed-add expects rank-2 X/values")
    rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(a_values.shape[0])
    split_k_slices = int(split_k_slices)
    if split_k_slices not in (2, 4, 8):
        raise ValueError("split_k_slices must be 2, 4, or 8")
    if (
        rows <= 0
        or dense_count <= 0
        or dense_count > rows
        or rows % 8
        or K % (64 * split_k_slices)
        or N % 256
    ):
        raise ValueError(
            "split-K indexed-add requires 0 < dense_count <= rows, "
            "rows % 8, N % 256, and K % (64 * split_k_slices) == 0"
        )
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(f"a_values must have shape {(N, K // 2)}")
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError("a_meta_e has an invalid number of elements")
    if any(not tensor.is_contiguous() for tensor in tensors[:-1]):
        raise ValueError("split-K indexed-add inputs must be contiguous")
    if (
        full_output.dtype != torch.float16
        or full_output.ndim != 2
        or int(full_output.shape[1]) != N
        or not full_output.is_contiguous()
    ):
        raise ValueError(
            "full_output must be contiguous CUDA fp16 with N output columns"
        )
    full_rows = int(full_output.shape[0])

    if out is None:
        residual_output = torch.empty_strided(
            (rows, N), (1, rows), device=X.device, dtype=torch.float16
        )
    else:
        residual_output = out
        if (
            not residual_output.is_cuda
            or residual_output.device != X.device
            or residual_output.dtype != torch.float16
            or tuple(residual_output.shape) != (rows, N)
            or tuple(residual_output.stride()) != (1, rows)
        ):
            raise ValueError(
                f"out must be CUDA fp16 shape {(rows, N)} with "
                f"stride {(1, rows)}"
            )

    required_workspace_ints = ((N + 255) // 256) * ((rows + 31) // 32)
    if workspace is None:
        split_workspace = torch.zeros(
            required_workspace_ints, device=X.device, dtype=torch.int32
        )
    else:
        split_workspace = workspace
        if (
            not split_workspace.is_cuda
            or split_workspace.device != X.device
            or split_workspace.dtype != torch.int32
            or split_workspace.ndim != 1
            or not split_workspace.is_contiguous()
            or split_workspace.numel() < required_workspace_ints
        ):
            raise ValueError(
                "workspace must be contiguous CUDA int32 with at least "
                f"{required_workspace_ints} elements"
            )
    if ready_state is None:
        sync_state = torch.zeros(2, device=X.device, dtype=torch.int32)
    else:
        sync_state = ready_state
        if (
            not sync_state.is_cuda
            or sync_state.device != X.device
            or sync_state.dtype != torch.int32
            or sync_state.ndim != 1
            or not sync_state.is_contiguous()
            or sync_state.numel() < 2
        ):
            raise ValueError(
                "ready_state must be contiguous CUDA int32 with at least "
                "two elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_device_splitk_indexed_add_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(residual_output.data_ptr()),
        ctypes.c_void_p(split_workspace.data_ptr()),
        ctypes.c_int(split_workspace.numel()),
        ctypes.c_void_p(full_output.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(sync_state.data_ptr()),
        ctypes.c_int(sync_state.numel()),
        ctypes.c_int(rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(full_rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(split_k_slices),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 CUTLASS split-K indexed-add GEMM failed with code "
            f"{int(ret)}"
        )
    return residual_output


def sparse24_cutlass_gather_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Run sparse GEMM while gathering selected rows in the B iterator."""

    tensors = (X, a_values, a_meta_e, row_indices)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("gather GEMM inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("gather GEMM inputs must share one device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("gather sparse GEMM currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("a_meta_e must have dtype torch.uint16/int16")
    if row_indices.dtype != torch.int32 or row_indices.ndim != 1:
        raise ValueError("row_indices must be a rank-1 CUDA int32 tensor")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, values rank-2, and metadata rank-1")
    source_rows, K = map(int, X.shape)
    rows = int(row_indices.numel())
    N = int(a_values.shape[0])
    if rows <= 0 or rows > source_rows:
        raise ValueError("selected row count must be in [1, X.shape[0]]")
    if K % 64 or N % 32:
        raise ValueError("gather sparse GEMM requires K % 64 and N % 32 == 0")
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(f"a_values must have shape {(N, K // 2)}")
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(f"a_meta_e must have {N * (K // 16)} elements")
    configs = {
        "auto": 0,
        "256x32x64_s3_sw4": 1,
        "256x64x64_s3_sw4": 2,
        "128x32x64_s4_sw4": 3,
    }
    if config not in configs:
        raise ValueError(f"unsupported gather sparse GEMM config {config!r}")
    if not row_indices.is_contiguous():
        row_indices = row_indices.contiguous()

    output_leading_rows = (rows + 7) // 8 * 8
    output_stride = (1, output_leading_rows)
    if out is None:
        output = torch.empty_strided(
            (rows, N), output_stride, device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, N)
            or tuple(output.stride()) != output_stride
        ):
            raise ValueError(
                f"out must have shape/stride {(rows, N)}/{output_stride}"
            )

    X = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_gather_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_gather_gemm_f16 failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_paired_persistent_gemm_prepacked(
    full_x: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_x: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    *,
    full_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    schedule: str = "partitioned",
    config: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run full and dense-row residual 2:4 GEMMs in one persistent grid.

    Both outputs are transposed-stride views over the CUTLASS ``C[N, M]``
    storage. This low-level prototype intentionally requires row counts that
    are multiples of eight so graph callers can own all padding and buffers.
    """

    schedule_ids = {"partitioned": 0, "interleaved": 1}
    config_ids = {
        "auto": 0,
        "256x64_full_256x64_residual": 1,
        "256x128_full_256x64_residual": 2,
        "256x128_full_256x128_residual": 3,
    }
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )

    tensors = (
        full_x,
        full_values,
        full_meta_e,
        residual_x,
        residual_values,
        residual_meta_e,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired persistent GEMM inputs must be CUDA tensors")
    if any(tensor.device != full_x.device for tensor in tensors[1:]):
        raise ValueError("paired persistent GEMM inputs must share one device")
    if full_x.dtype != torch.float16 or residual_x.dtype != torch.float16:
        raise ValueError("paired persistent GEMM currently supports fp16 only")
    if full_values.dtype != torch.float16 or residual_values.dtype != torch.float16:
        raise ValueError("paired persistent GEMM values must be fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("full_meta_e must have dtype torch.uint16/int16")
    if residual_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("residual_meta_e must have dtype torch.uint16/int16")
    if full_x.ndim != 2 or residual_x.ndim != 2:
        raise ValueError("full_x and residual_x must be rank-2")
    if full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("full_values and residual_values must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("full_meta_e and residual_meta_e must be rank-1")

    full_rows, K = full_x.shape
    residual_rows, residual_k = residual_x.shape
    if residual_k != K:
        raise ValueError("full and residual inputs must have the same K")
    if residual_rows <= 0 or residual_rows > full_rows:
        raise ValueError("residual row count must be in [1, full_rows]")
    if full_rows % 8 != 0 or residual_rows % 8 != 0:
        raise ValueError("paired persistent GEMM row counts must be divisible by 8")
    if K % 64 != 0:
        raise ValueError(f"paired persistent GEMM requires K % 64 == 0, got {K}")
    N = full_values.shape[0]
    if N % 256 != 0:
        raise ValueError(f"paired persistent GEMM requires N % 256 == 0, got {N}")
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values:
        raise ValueError(
            f"full_values must have shape {expected_values}, got {tuple(full_values.shape)}"
        )
    if tuple(residual_values.shape) != expected_values:
        raise ValueError(
            "residual_values must match full_values shape, got "
            f"{tuple(residual_values.shape)}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta:
        raise ValueError(
            f"full_meta_e must have {expected_meta} elements, got {full_meta_e.numel()}"
        )
    if residual_meta_e.numel() != expected_meta:
        raise ValueError(
            "residual_meta_e must have "
            f"{expected_meta} elements, got {residual_meta_e.numel()}"
        )

    def prepare_output(
        output: torch.Tensor | None,
        rows: int,
        label: str,
    ) -> torch.Tensor:
        if output is None:
            return torch.empty_strided(
                (rows, N), (1, rows), device=full_x.device, dtype=torch.float16
            )
        if not output.is_cuda or output.device != full_x.device:
            raise ValueError(f"{label} must be a CUDA tensor on the input device")
        if output.dtype != torch.float16:
            raise ValueError(f"{label} must have dtype torch.float16")
        if tuple(output.shape) != (rows, N) or tuple(output.stride()) != (1, rows):
            raise ValueError(
                f"{label} must have shape/stride {(rows, N)}/{(1, rows)}, "
                f"got {tuple(output.shape)}/{tuple(output.stride())}"
            )
        return output

    full_output = prepare_output(full_out, full_rows, "full_out")
    residual_output = prepare_output(
        residual_out, residual_rows, "residual_out"
    )
    full_x = full_x.contiguous()
    residual_x = residual_x.contiguous()
    full_values = full_values.contiguous()
    residual_values = residual_values.contiguous()
    full_meta_e = full_meta_e.contiguous()
    residual_meta_e = residual_meta_e.contiguous()

    lib = _load_library()
    stream = torch.cuda.current_stream(full_x.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_persistent_gemm_f16_stream(
        ctypes.c_void_p(full_x.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(full_output.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_void_p(residual_x.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(residual_output.data_ptr()),
        ctypes.c_int(residual_rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_paired_persistent_gemm_f16 failed with code "
            f"{int(ret)}"
        )
    return full_output, residual_output


def sparse24_cutlass_paired_gather_residual_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    *,
    full_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    schedule: str = "partitioned",
    config: str = "auto",
    worker_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run W24 over all rows and gather-routed R24 in one persistent grid.

    ``residual_out`` stores compact dense-route rows in the same order as
    ``dense_rows``. Configurations ending in ``_contiguous`` write ordinary
    row-major outputs; the remaining configurations use CUTLASS's transposed
    stride for a following fused routed-residual epilogue.
    """

    schedule_ids = {"partitioned": 0, "interleaved": 1}
    config_ids = {
        "auto": 0,
        "256x64_full_256x64_residual": 1,
        "128x64_full_128x64_residual": 2,
        "256x32_full_256x32_residual": 3,
        "128x32_full_128x32_residual": 4,
        "64x64_full_64x64_residual": 5,
        "256x64_full_128x64_residual": 6,
        "128x64_full_256x64_residual": 7,
        "128x64_full_128x64_residual_contiguous": 8,
        "256x32_full_256x32_residual_contiguous": 9,
        "64x64_full_64x64_residual_contiguous": 10,
        "256x32_full_128x32_residual_contiguous": 11,
        "256x64_full_256x64_residual_contiguous": 12,
        "256x64_full_256x32_residual_contiguous": 13,
    }
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired gather residual inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("paired gather residual inputs must share one device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("paired gather residual GEMM currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("paired gather residual metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("paired gather residual matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("paired gather residual metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paired gather residual inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count > full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] rows")
    if K % 64 or N % 64:
        raise ValueError(
            f"paired gather residual requires K and N divisible by 64, got {K}/{N}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )
    contiguous_output = config.endswith("_contiguous")

    def prepare_output(
        output: torch.Tensor | None,
        rows: int,
        label: str,
    ) -> torch.Tensor:
        if output is None:
            if contiguous_output:
                return torch.empty(
                    (rows, N), device=X.device, dtype=torch.float16
                )
            return torch.empty_strided(
                (rows, N), (1, rows), device=X.device, dtype=torch.float16
            )
        expected_stride = (N, 1) if contiguous_output else (1, rows)
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, N)
            or tuple(output.stride()) != expected_stride
        ):
            raise ValueError(
                f"{label} must have CUDA fp16 shape/stride "
                f"{(rows, N)}/{expected_stride}"
            )
        return output

    full_output = prepare_output(full_out, full_rows, "full_out")
    residual_output = prepare_output(
        residual_out, dense_count, "residual_out"
    )
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_gather_residual_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(full_output.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(residual_output.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 paired gather residual GEMM failed with code "
            f"{int(ret)}"
        )
    return full_output, residual_output


def sparse24_cutlass_paired_gather_residual_qkv_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    grid_barrier: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    full_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    schedule: str = "interleaved",
    config: str = "256x64_full_256x32_residual_contiguous",
    worker_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run paired W24/R24 and QKV post-processing in one persistent launch."""

    schedule_ids = {"partitioned": 0, "interleaved": 1}
    config_ids = {"256x64_full_256x32_residual_contiguous": 13}
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    if config not in config_ids:
        raise ValueError(
            "fused paired QKV currently supports only "
            "256x64_full_256x32_residual_contiguous"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if head_dim != 128:
        raise ValueError(f"fused paired QKV requires head_dim=128, got {head_dim}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        dense_slot_by_row,
        cos_sin_cache,
        position_ids,
        grid_barrier,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused paired QKV inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("fused paired QKV inputs must share one device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("fused paired QKV GEMM currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("fused paired QKV metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("fused paired QKV matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("fused paired QKV metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused paired QKV inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count > full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] rows")
    if N != q_size + 2 * kv_size:
        raise ValueError(
            f"QKV output size mismatch: N={N}, q_size={q_size}, kv_size={kv_size}"
        )
    if K % 64 or N % 256 or q_size % 256 or kv_size % 256:
        raise ValueError(
            "fused paired QKV requires K divisible by 64 and N/Q/KV "
            "dimensions divisible by 256"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )
    if (
        dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (full_rows,)
    ):
        raise ValueError(
            "dense_slot_by_row must be contiguous CUDA int32 with shape "
            f"{(full_rows,)}"
        )
    if (
        cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
    ):
        raise ValueError("cos_sin_cache must be contiguous rank-2 CUDA fp16")
    rotary_dim = int(cos_sin_cache.shape[1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}, got {rotary_dim}"
        )
    if position_ids.dtype != torch.int64 or position_ids.numel() != full_rows:
        raise ValueError(
            f"position_ids must be contiguous CUDA int64 with {full_rows} elements"
        )
    if grid_barrier.dtype != torch.int32 or tuple(grid_barrier.shape) != (2,):
        raise ValueError("grid_barrier must be contiguous CUDA int32 shape (2,)")
    if normalize_qk:
        for label, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
            assert weight is not None
            if (
                not weight.is_cuda
                or weight.device != X.device
                or weight.dtype != torch.float16
                or tuple(weight.shape) != (head_dim,)
                or not weight.is_contiguous()
            ):
                raise ValueError(
                    f"{label} must be contiguous CUDA fp16 shape {(head_dim,)}"
                )

    def prepare_output(
        output: torch.Tensor | None,
        rows: int,
        label: str,
    ) -> torch.Tensor:
        if output is None:
            return torch.empty((rows, N), device=X.device, dtype=torch.float16)
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 shape {(rows, N)}"
            )
        return output

    full_output = prepare_output(full_out, full_rows, "full_out")
    residual_output = prepare_output(residual_out, dense_count, "residual_out")
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_gather_residual_qkv_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(full_output.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(residual_output.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_void_p(q_weight.data_ptr()) if q_weight is not None else None,
        ctypes.c_void_p(k_weight.data_ptr()) if k_weight is not None else None,
        ctypes.c_void_p(cos_sin_cache.data_ptr()),
        ctypes.c_void_p(position_ids.data_ptr()),
        ctypes.c_int(q_size),
        ctypes.c_int(kv_size),
        ctypes.c_int(rotary_dim),
        ctypes.c_float(epsilon),
        ctypes.c_int(int(is_neox)),
        ctypes.c_int(int(normalize_qk)),
        ctypes.c_void_p(grid_barrier.data_ptr()),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused paired QKV GEMM failed with code "
            f"{int(ret)}"
        )
    return full_output, residual_output


def sparse24_cutlass_paired_fused_routed_qkv_epilogue_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    dense_base: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    feature_counters: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    config: str = "256x64_full_256x32_residual_epilogue",
    worker_blocks: int = 0,
    residual_worker_blocks: int = 0,
) -> torch.Tensor:
    """Run barrier-free mixed-row W24/R24 QKV epilogues."""

    config_ids = {"256x64_full_256x32_residual_epilogue": 14}
    if config not in config_ids:
        raise ValueError(
            "routed QKV epilogue currently supports only "
            "256x64_full_256x32_residual_epilogue"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    residual_worker_blocks = int(residual_worker_blocks)
    if residual_worker_blocks < 0:
        raise ValueError("residual_worker_blocks must be non-negative")
    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if head_dim != 128 or not is_neox:
        raise ValueError(
            "routed QKV epilogue requires Neox RoPE with head_dim=128"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        dense_slot_by_row,
        dense_base,
        cos_sin_cache,
        position_ids,
        feature_counters,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("routed QKV epilogue inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("routed QKV epilogue inputs must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("routed QKV epilogue inputs must be contiguous")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
        or dense_base.dtype != torch.float16
        or cos_sin_cache.dtype != torch.float16
    ):
        raise ValueError("routed QKV epilogue currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("routed QKV epilogue metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("routed QKV epilogue matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("routed QKV epilogue metadata must be rank-1")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count > full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] rows")
    if N != q_size + 2 * kv_size:
        raise ValueError(
            f"QKV output size mismatch: N={N}, q_size={q_size}, kv_size={kv_size}"
        )
    if K % 64 or N % 256 or q_size % 256 or kv_size % 256:
        raise ValueError(
            "routed QKV epilogue requires K divisible by 64 and N/Q/KV "
            "dimensions divisible by 256"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )
    if (
        dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (full_rows,)
    ):
        raise ValueError(
            "dense_slot_by_row must be contiguous CUDA int32 with shape "
            f"{(full_rows,)}"
        )
    if tuple(dense_base.shape) != (dense_count, N):
        raise ValueError(
            f"dense_base must be contiguous CUDA fp16 shape {(dense_count, N)}"
        )
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != head_dim:
        raise ValueError(
            f"cos_sin_cache must be rank-2 fp16 with width {head_dim}"
        )
    rotary_dim = int(cos_sin_cache.shape[1])
    if position_ids.dtype != torch.int64 or position_ids.numel() != full_rows:
        raise ValueError(
            f"position_ids must be contiguous CUDA int64 with {full_rows} elements"
        )
    if feature_counters.dtype != torch.int32 or tuple(
        feature_counters.shape
    ) != (N // 256,):
        raise ValueError(
            f"feature_counters must be contiguous CUDA int32 shape {(N // 256,)}"
        )
    if normalize_qk:
        for label, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
            assert weight is not None
            if (
                not weight.is_cuda
                or weight.device != X.device
                or weight.dtype != torch.float16
                or tuple(weight.shape) != (head_dim,)
                or not weight.is_contiguous()
            ):
                raise ValueError(
                    f"{label} must be contiguous CUDA fp16 shape {(head_dim,)}"
                )

    if out is None:
        output = torch.empty(
            (full_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        if (
            not out.is_cuda
            or out.device != X.device
            or out.dtype != torch.float16
            or tuple(out.shape) != (full_rows, N)
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous CUDA fp16 shape {(full_rows, N)}"
            )
        output = out

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = (
        lib.sparse24_cutlass_paired_fused_routed_qkv_epilogue_gemm_f16_stream(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(full_values.data_ptr()),
            ctypes.c_void_p(full_meta_e.data_ptr()),
            ctypes.c_void_p(residual_values.data_ptr()),
            ctypes.c_void_p(residual_meta_e.data_ptr()),
            ctypes.c_void_p(dense_rows.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(dense_base.data_ptr()),
            ctypes.c_void_p(q_weight.data_ptr()) if q_weight is not None else None,
            ctypes.c_void_p(k_weight.data_ptr()) if k_weight is not None else None,
            ctypes.c_void_p(cos_sin_cache.data_ptr()),
            ctypes.c_void_p(position_ids.data_ptr()),
            ctypes.c_void_p(feature_counters.data_ptr()),
            ctypes.c_int(full_rows),
            ctypes.c_int(dense_count),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_int(q_size),
            ctypes.c_int(kv_size),
            ctypes.c_int(rotary_dim),
            ctypes.c_float(epsilon),
            ctypes.c_int(int(is_neox)),
            ctypes.c_int(int(normalize_qk)),
            ctypes.c_int(config_ids[config]),
            ctypes.c_int(worker_blocks),
            ctypes.c_int(residual_worker_blocks),
            ctypes.c_void_p(stream),
        )
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused routed QKV epilogue failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_paired_finalize_residual_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    feature_counters: torch.Tensor | None = None,
    config: str = "auto",
    worker_blocks: int = 0,
    schedule: str = "partitioned",
) -> torch.Tensor:
    """Overlap W24/R24 and let each feature's last CTA add the residual.

    ``feature_counters`` is persistent replay state. It must be zero on the
    first invocation; the kernel resets every feature counter before returning.
    """

    config_ids = {
        "auto": 0,
        "256x32_full_256x32_residual_finalize": 1,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    schedule_ids = {"partitioned": 0, "interleaved": 1}
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired residual finalizer inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("paired residual finalizer inputs must share one device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("paired residual finalizer currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("paired residual metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("paired residual matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("paired residual metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paired residual finalizer inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count >= full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] - 1 rows")
    if K % 64 or N % 256:
        raise ValueError(
            "paired residual finalizer requires K divisible by 64 and N "
            f"divisible by 256, got {K}/{N}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )

    def prepare_output(
        output: torch.Tensor | None,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if output is None:
            return torch.empty(shape, device=X.device, dtype=torch.float16)
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != shape
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 with shape {shape}"
            )
        return output

    output = prepare_output(out, (full_rows, N), "out")
    compact_output = prepare_output(
        residual_out, (dense_count, N), "residual_out"
    )
    feature_tiles = N // 256
    if feature_counters is None:
        counters = torch.zeros(
            feature_tiles, device=X.device, dtype=torch.int32
        )
    else:
        counters = feature_counters
        if (
            not counters.is_cuda
            or counters.device != X.device
            or counters.dtype != torch.int32
            or counters.ndim != 1
            or counters.numel() < feature_tiles
            or not counters.is_contiguous()
        ):
            raise ValueError(
                "feature_counters must be contiguous CUDA int32 with at least "
                f"{feature_tiles} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_finalize_residual_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(compact_output.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 paired residual finalizer failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_paired_finalize_qkv_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    feature_counters: torch.Tensor | None = None,
    config: str = "256x64_full_256x32_residual_finalize_qkv",
    worker_blocks: int = 0,
    schedule: str = "partitioned",
) -> torch.Tensor:
    """Finalize W24/R24 QKV and run feature-local Q/K norm plus RoPE."""

    config_ids = {
        "256x32_full_256x32_residual_finalize_qkv": 1,
        "256x64_full_256x64_residual_finalize_qkv": 2,
        "256x64_full_256x32_residual_finalize_qkv": 3,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    schedule_ids = {"partitioned": 0, "interleaved": 1}
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if head_dim != 128 or not is_neox:
        raise ValueError(
            "feature-local QKV finalizer requires Neox RoPE with head_dim=128"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        cos_sin_cache,
        position_ids,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("feature-local QKV finalizer inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("feature-local QKV finalizer inputs must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("feature-local QKV finalizer inputs must be contiguous")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
        or cos_sin_cache.dtype != torch.float16
    ):
        raise ValueError("feature-local QKV finalizer currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("feature-local QKV metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if position_ids.dtype != torch.int64 or position_ids.ndim != 1:
        raise ValueError("position_ids must be a rank-1 int64 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("feature-local QKV matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("feature-local QKV metadata must be rank-1")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count >= full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] - 1 rows")
    if N != q_size + 2 * kv_size:
        raise ValueError(
            f"QKV output size mismatch: N={N}, q_size={q_size}, kv_size={kv_size}"
        )
    if K % 64 or N % 256 or q_size % 256 or kv_size % 256:
        raise ValueError(
            "feature-local QKV finalizer requires K/N/Q/KV dimensions "
            "divisible by the CUTLASS tile"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )
    if position_ids.numel() != full_rows:
        raise ValueError(f"position_ids must contain {full_rows} entries")
    if cos_sin_cache.ndim != 2 or int(cos_sin_cache.shape[1]) != head_dim:
        raise ValueError(
            f"cos_sin_cache must be rank-2 fp16 with width {head_dim}"
        )
    if normalize_qk:
        for label, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
            assert weight is not None
            if (
                not weight.is_cuda
                or weight.device != X.device
                or weight.dtype != torch.float16
                or tuple(weight.shape) != (head_dim,)
                or not weight.is_contiguous()
            ):
                raise ValueError(
                    f"{label} must be contiguous CUDA fp16 shape {(head_dim,)}"
                )

    def prepare_output(
        value: torch.Tensor | None,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.empty(shape, device=X.device, dtype=torch.float16)
        if (
            not value.is_cuda
            or value.device != X.device
            or value.dtype != torch.float16
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 with shape {shape}"
            )
        return value

    output = prepare_output(out, (full_rows, N), "out")
    compact_output = prepare_output(
        residual_out, (dense_count, N), "residual_out"
    )
    feature_tiles = N // 256
    if feature_counters is None:
        counters = torch.zeros(
            feature_tiles, device=X.device, dtype=torch.int32
        )
    else:
        counters = feature_counters
        if (
            not counters.is_cuda
            or counters.device != X.device
            or counters.dtype != torch.int32
            or counters.ndim != 1
            or counters.numel() < feature_tiles
            or not counters.is_contiguous()
        ):
            raise ValueError(
                "feature_counters must be contiguous CUDA int32 with at least "
                f"{feature_tiles} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_finalize_qkv_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(compact_output.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_void_p(q_weight.data_ptr()) if q_weight is not None else None,
        ctypes.c_void_p(k_weight.data_ptr()) if k_weight is not None else None,
        ctypes.c_void_p(cos_sin_cache.data_ptr()),
        ctypes.c_void_p(position_ids.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(q_size),
        ctypes.c_int(kv_size),
        ctypes.c_int(head_dim),
        ctypes.c_float(epsilon),
        ctypes.c_int(int(is_neox)),
        ctypes.c_int(int(normalize_qk)),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 feature-local QKV finalizer failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_paired_inplace_residual_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    feature_counters: torch.Tensor | None = None,
    config: str = "auto",
    worker_blocks: int = 0,
    schedule: str = "partitioned",
) -> torch.Tensor:
    """Run concurrent W24/R24 and indexed-add R24 into the final output.

    ``feature_counters`` is persistent replay state. It must be zero on the
    first invocation; the kernel resets every feature counter before returning.
    """

    config_ids = {
        "auto": 0,
        "128x64_full_128x64_residual_inplace": 1,
        "256x32_full_256x32_residual_inplace": 2,
        "256x64_full_256x32_residual_inplace": 3,
        "256x64w64_full_256x32_residual_inplace": 4,
        "128x32_full_128x32_residual_inplace": 5,
        "64x64w32x64_full_64x32_residual_inplace": 6,
        "256x32_full_last_owner_256x32_residual_inplace": 7,
        "256x32_full_256x32_residual_f16_inplace": 8,
        "256x32_full_256x32_residual_all_f16_inplace": 9,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    schedule_ids = {
        "partitioned": 0,
        "interleaved": 1,
        "shared_phased": 2,
    }
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired in-place residual inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("paired in-place residual inputs must share one device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("paired in-place residual GEMM currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("paired in-place residual metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("paired in-place residual matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("paired in-place residual metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paired in-place residual inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count >= full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] - 1 rows")
    if K % 64 or N % 256:
        raise ValueError(
            "paired in-place residual requires K divisible by 64 and N "
            f"divisible by 256, got {K}/{N}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )

    if out is None:
        output = torch.empty(
            (full_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (full_rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous CUDA fp16 with shape {(full_rows, N)}"
            )

    effective_config = config
    if effective_config == "auto":
        effective_config = "256x32_full_256x32_residual_inplace"
    feature_columns = int(effective_config.split("x", 1)[0])
    feature_tiles = (N + feature_columns - 1) // feature_columns
    if feature_counters is None:
        counters = torch.zeros(
            feature_tiles, device=X.device, dtype=torch.int32
        )
    else:
        counters = feature_counters
        if (
            not counters.is_cuda
            or counters.device != X.device
            or counters.dtype != torch.int32
            or counters.ndim != 1
            or counters.numel() < feature_tiles
            or not counters.is_contiguous()
        ):
            raise ValueError(
                "feature_counters must be contiguous CUDA int32 with at least "
                f"{feature_tiles} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_inplace_residual_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[effective_config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 paired in-place residual GEMM failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
    full_x: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    residual_x: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    *,
    dense_count: int,
    full_out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    schedule: str = "partitioned",
    config: str = "auto",
    worker_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Schedule routed gate/up and dense-row residual in one CTA grid."""

    schedule_ids = {"partitioned": 0, "interleaved": 1}
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    config_ids = {
        "auto": 0,
        "256x64x64_s3": 1,
        "256x64x64_s3_sw4": 2,
        "256x32x64_s3_sw4": 3,
        "256x64x64_s2_sw4": 4,
        "256x64_full_256x32_residual_s3_sw4": 5,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks == 1 or worker_blocks < 0:
        raise ValueError("worker_blocks must be 0 (auto) or at least 2")

    tensors = (
        full_x,
        full_values,
        full_meta_e,
        dense_slot_by_row,
        residual_x,
        residual_values,
        residual_meta_e,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired routed SwiGLU inputs must be CUDA tensors")
    if any(tensor.device != full_x.device for tensor in tensors[1:]):
        raise ValueError("paired routed SwiGLU inputs must share one device")
    if full_x.dtype != torch.float16 or residual_x.dtype != torch.float16:
        raise ValueError("paired routed SwiGLU currently supports fp16 only")
    if full_values.dtype != torch.float16 or residual_values.dtype != torch.float16:
        raise ValueError("paired routed SwiGLU values must be fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("full_meta_e must have dtype torch.uint16/int16")
    if residual_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("residual_meta_e must have dtype torch.uint16/int16")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be a rank-1 int32 tensor")
    if full_x.ndim != 2 or residual_x.ndim != 2:
        raise ValueError("full_x and residual_x must be rank-2")
    if full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("full_values and residual_values must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("full_meta_e and residual_meta_e must be rank-1")

    full_rows, K = (int(value) for value in full_x.shape)
    residual_rows, residual_k = (int(value) for value in residual_x.shape)
    dense_count = int(dense_count)
    N = int(full_values.shape[0])
    hidden_size = N // 2
    if residual_k != K:
        raise ValueError("full and residual inputs must have the same K")
    if dense_count <= 0 or dense_count > full_rows:
        raise ValueError("dense_count must be in [1, full_rows]")
    if residual_rows < dense_count or residual_rows > full_rows:
        raise ValueError("residual rows must cover the padded dense rows")
    if full_rows % 8 != 0 or residual_rows % 8 != 0:
        raise ValueError("paired routed SwiGLU rows must be divisible by 8")
    if K % 64 != 0 or N % 256 != 0:
        raise ValueError("paired routed SwiGLU requires K % 64 and N % 256")
    if tuple(dense_slot_by_row.shape) != (full_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(full_rows,)}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values:
        raise ValueError(f"full_values must have shape {expected_values}")
    if tuple(residual_values.shape) != expected_values:
        raise ValueError(f"residual_values must have shape {expected_values}")
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta:
        raise ValueError(f"full_meta_e must have {expected_meta} elements")
    if residual_meta_e.numel() != expected_meta:
        raise ValueError(
            f"residual_meta_e must have {expected_meta} elements"
        )

    def prepare_output(
        output: torch.Tensor | None,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if output is None:
            return torch.empty(shape, device=full_x.device, dtype=torch.float16)
        if (
            not output.is_cuda
            or output.device != full_x.device
            or output.dtype != torch.float16
            or tuple(output.shape) != shape
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 with shape {shape}"
            )
        return output

    full_output = prepare_output(
        full_out, (full_rows, hidden_size), "full_out"
    )
    compact_base = prepare_output(
        dense_base, (dense_count, N), "dense_base"
    )
    residual_output = prepare_output(
        residual_out, (residual_rows, N), "residual_out"
    )

    full_x = full_x.contiguous()
    full_values = full_values.contiguous()
    full_meta_e = full_meta_e.contiguous()
    dense_slot_by_row = dense_slot_by_row.contiguous()
    residual_x = residual_x.contiguous()
    residual_values = residual_values.contiguous()
    residual_meta_e = residual_meta_e.contiguous()

    lib = _load_library()
    stream = torch.cuda.current_stream(full_x.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_persistent_routed_swiglu_f16_stream(
        ctypes.c_void_p(full_x.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(full_output.data_ptr()),
        ctypes.c_void_p(compact_base.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_void_p(residual_x.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(residual_output.data_ptr()),
        ctypes.c_int(residual_rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "paired persistent routed SwiGLU failed with code " f"{int(ret)}"
        )
    return full_output, compact_base, residual_output


def sparse24_cutlass_paired_gather_routed_swiglu_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    dense_rows: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    *,
    full_out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    schedule: str = "partitioned",
    config: str = "auto",
    worker_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual-row gather into the paired routed SwiGLU GEMM."""

    schedule_ids = {"partitioned": 0, "interleaved": 1}
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    config_ids = {
        "auto": 0,
        "256x64x64_s3": 1,
        "256x64x64_s3_sw4": 2,
        "256x32x64_s3_sw4": 3,
        "256x64x64_s2_sw4": 4,
        "256x64_full_256x32_residual_s3_sw4": 5,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be 0 (auto) or at least 2")

    tensors = (
        X,
        full_values,
        full_meta_e,
        dense_slot_by_row,
        dense_rows,
        residual_values,
        residual_meta_e,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired gather SwiGLU inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("paired gather SwiGLU inputs must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paired gather SwiGLU inputs must be contiguous")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("paired gather SwiGLU currently supports fp16 only")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("paired gather SwiGLU metadata must be uint16/int16")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be a rank-1 int32 tensor")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("paired gather SwiGLU matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("paired gather SwiGLU metadata must be rank-1")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    hidden_size = N // 2
    if dense_count <= 0 or dense_count > full_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] rows")
    if full_rows % 8 or K % 64 or N % 256:
        raise ValueError(
            "paired gather SwiGLU requires rows % 8, K % 64, and N % 256"
        )
    if tuple(dense_slot_by_row.shape) != (full_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(full_rows,)}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )

    def prepare_output(
        output: torch.Tensor | None,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if output is None:
            return torch.empty(shape, device=X.device, dtype=torch.float16)
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != shape
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 with shape {shape}"
            )
        return output

    full_output = prepare_output(
        full_out, (full_rows, hidden_size), "full_out"
    )
    compact_base = prepare_output(
        dense_base, (dense_count, N), "dense_base"
    )
    residual_output = prepare_output(
        residual_out, (dense_count, N), "residual_out"
    )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_gather_routed_swiglu_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(full_output.data_ptr()),
        ctypes.c_void_p(compact_base.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(residual_output.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"paired gather routed SwiGLU failed with code {int(ret)}"
        )
    return full_output, compact_base, residual_output


def sparse24_cutlass_paired_fused_routed_swiglu_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    compact_residual_x: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    feature_counters: torch.Tensor | None = None,
    config: str = "auto",
    worker_blocks: int = 0,
    schedule: str = "partitioned",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse gathered R24 Gate/Up correction into the paired epilogue.

    The W24 epilogue writes sparse-row SwiGLU results and compact raw Gate/Up
    values for routed dense rows. R24 workers either gather those rows from
    ``X`` or consume ``compact_residual_x``, then wait only for the matching
    feature tile before applying exact SwiGLU and scattering into ``out``.
    ``feature_counters`` is persistent CUDA Graph replay state and is reset to
    zero before each invocation returns. Residual values must use the same
    Gate/Up interleaved pack as the full epilogue.
    """

    config_ids = {
        "auto": 0,
        "256x64x64_s3": 1,
        "256x64x64_s3_sw4": 2,
        "256x32x64_s3_sw4": 3,
        "256x64x64_s2_sw4": 4,
        "256x64x64_s3_sw4_fast_silu": 5,
        "256x64_full_256x32_residual_s3_sw4": 6,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    schedule_ids = {"partitioned": 0, "shared_phased": 2}
    if schedule not in schedule_ids:
        raise ValueError(
            f"schedule must be one of {sorted(schedule_ids)}, got {schedule!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        dense_slot_by_row,
    )
    if compact_residual_x is not None:
        tensors = (*tensors, compact_residual_x)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paired fused routed SwiGLU inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError(
            "paired fused routed SwiGLU inputs must share one device"
        )
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
        or (
            compact_residual_x is not None
            and compact_residual_x.dtype != torch.float16
        )
    ):
        raise ValueError("paired fused routed SwiGLU currently supports fp16")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError(
            "paired fused routed SwiGLU metadata must be uint16/int16"
        )
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError(
            "dense_slot_by_row must be a rank-1 int32 CUDA tensor"
        )
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("paired fused routed SwiGLU matrices must be rank-2")
    if compact_residual_x is not None and compact_residual_x.ndim != 2:
        raise ValueError("compact_residual_x must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("paired fused routed SwiGLU metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("paired fused routed SwiGLU inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    hidden_size = N // 2
    if dense_count <= 0 or dense_count >= full_rows:
        raise ValueError(
            "dense_rows must select between 1 and X.shape[0] - 1 rows"
        )
    if full_rows % 8 or K % 64 or N % 256:
        raise ValueError(
            "paired fused routed SwiGLU requires rows % 8, K % 64, and "
            f"N % 256 == 0, got {full_rows}/{K}/{N}"
        )
    if tuple(dense_slot_by_row.shape) != (full_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(full_rows,)}"
        )
    if compact_residual_x is None:
        residual_input = X
        residual_rows = dense_count
        gather_residual_rows = True
    else:
        residual_input = compact_residual_x
        residual_rows, residual_k = map(int, residual_input.shape)
        gather_residual_rows = False
        if residual_k != K or residual_rows < dense_count:
            raise ValueError(
                "compact_residual_x must have shape [rows, K] with rows >= "
                f"dense_count; got {tuple(residual_input.shape)} for "
                f"dense_count={dense_count}, K={K}"
            )
        if residual_rows > full_rows or residual_rows % 8:
            raise ValueError(
                "compact_residual_x rows must be divisible by 8 and no "
                f"larger than X rows; got {residual_rows}/{full_rows}"
            )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )

    if out is None:
        output = torch.empty(
            (full_rows, hidden_size), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (full_rows, hidden_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(full_rows, hidden_size)}"
            )

    if dense_base is None:
        compact_base = torch.empty(
            (dense_count, N), device=X.device, dtype=torch.float16
        )
    else:
        compact_base = dense_base
        if (
            not compact_base.is_cuda
            or compact_base.device != X.device
            or compact_base.dtype != torch.float16
            or tuple(compact_base.shape) != (dense_count, N)
            or not compact_base.is_contiguous()
        ):
            raise ValueError(
                "dense_base must be contiguous CUDA fp16 with shape "
                f"{(dense_count, N)}"
            )

    feature_tiles = N // 256
    if feature_counters is None:
        counters = torch.zeros(
            feature_tiles, device=X.device, dtype=torch.int32
        )
    else:
        counters = feature_counters
        if (
            not counters.is_cuda
            or counters.device != X.device
            or counters.dtype != torch.int32
            or counters.ndim != 1
            or counters.numel() < feature_tiles
            or not counters.is_contiguous()
        ):
            raise ValueError(
                "feature_counters must be contiguous CUDA int32 with at least "
                f"{feature_tiles} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_fused_routed_swiglu_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_input.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(compact_base.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(residual_rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_int(gather_residual_rows),
        ctypes.c_int(schedule_ids[schedule]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 paired fused routed SwiGLU failed with code "
            f"{int(ret)}"
        )
    return output, compact_base, counters


def sparse24_cutlass_fused_mixed_mlp_prepacked(
    X: torch.Tensor,
    gate_full_values: torch.Tensor,
    gate_full_meta_e: torch.Tensor,
    gate_residual_values: torch.Tensor,
    gate_residual_meta_e: torch.Tensor,
    down_full_values: torch.Tensor,
    down_full_meta_e: torch.Tensor,
    down_residual_values: torch.Tensor,
    down_residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    hidden: torch.Tensor | None = None,
    gate_dense_base: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    gate_feature_counters: torch.Tensor | None = None,
    down_feature_counters: torch.Tensor | None = None,
    grid_barrier: torch.Tensor | None = None,
    config: str = "auto",
    worker_blocks: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run exact mixed Gate/SwiGLU and Down in one resident CUDA grid.

    The caller owns all scratch and synchronization tensors when CUDA Graph
    replay is required. Both feature-counter arrays are reset to zero by each
    invocation; ``grid_barrier[0]`` is reset while its sense may toggle.
    """

    config_ids = {"auto": 0, "256x64_gate_256x64_down": 1}
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")

    tensors = (
        X,
        gate_full_values,
        gate_full_meta_e,
        gate_residual_values,
        gate_residual_meta_e,
        down_full_values,
        down_full_meta_e,
        down_residual_values,
        down_residual_meta_e,
        dense_rows,
        dense_slot_by_row,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused mixed MLP inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("fused mixed MLP inputs must share one device")
    value_tensors = (
        X,
        gate_full_values,
        gate_residual_values,
        down_full_values,
        down_residual_values,
    )
    if any(tensor.dtype != torch.float16 for tensor in value_tensors):
        raise ValueError("fused mixed MLP currently supports fp16 values")
    meta_tensors = (
        gate_full_meta_e,
        gate_residual_meta_e,
        down_full_meta_e,
        down_residual_meta_e,
    )
    if any(
        tensor.dtype not in (torch.uint16, torch.int16)
        for tensor in meta_tensors
    ):
        raise ValueError("fused mixed MLP metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError(
            "dense_slot_by_row must be a rank-1 int32 CUDA tensor"
        )
    matrix_tensors = (
        X,
        gate_full_values,
        gate_residual_values,
        down_full_values,
        down_residual_values,
    )
    if any(tensor.ndim != 2 for tensor in matrix_tensors):
        raise ValueError("fused mixed MLP matrices must be rank-2")
    if any(tensor.ndim != 1 for tensor in meta_tensors):
        raise ValueError("fused mixed MLP metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused mixed MLP inputs must be contiguous")

    rows, model_width = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    gate_output_size = int(gate_full_values.shape[0])
    intermediate_size = gate_output_size // 2
    if dense_count <= 0 or dense_count >= rows:
        raise ValueError("dense_rows must select between 1 and rows - 1 rows")
    if tuple(dense_slot_by_row.shape) != (rows,):
        raise ValueError(f"dense_slot_by_row must have shape {(rows,)}")
    if rows % 8 or model_width % 256 or gate_output_size % 256:
        raise ValueError(
            "fused mixed MLP requires rows % 8, model_width % 256, and "
            f"gate_output_size % 256 == 0, got {rows}/{model_width}/"
            f"{gate_output_size}"
        )
    gate_values_shape = (gate_output_size, model_width // 2)
    if tuple(gate_full_values.shape) != gate_values_shape or tuple(
        gate_residual_values.shape
    ) != gate_values_shape:
        raise ValueError(
            f"gate values must both have shape {gate_values_shape}"
        )
    down_values_shape = (model_width, intermediate_size // 2)
    if tuple(down_full_values.shape) != down_values_shape or tuple(
        down_residual_values.shape
    ) != down_values_shape:
        raise ValueError(
            f"down values must both have shape {down_values_shape}"
        )
    gate_meta_elements = gate_output_size * (model_width // 16)
    down_meta_elements = model_width * (intermediate_size // 16)
    if any(tensor.numel() != gate_meta_elements for tensor in meta_tensors[:2]):
        raise ValueError(
            f"gate metadata must each have {gate_meta_elements} elements"
        )
    if any(tensor.numel() != down_meta_elements for tensor in meta_tensors[2:]):
        raise ValueError(
            f"down metadata must each have {down_meta_elements} elements"
        )

    def prepare_buffer(
        value: torch.Tensor | None,
        shape: tuple[int, ...],
        label: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.empty(shape, device=X.device, dtype=torch.float16)
        if (
            not value.is_cuda
            or value.device != X.device
            or value.dtype != torch.float16
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 with shape {shape}"
            )
        return value

    hidden_output = prepare_buffer(
        hidden, (rows, intermediate_size), "hidden"
    )
    compact_base = prepare_buffer(
        gate_dense_base,
        (dense_count, gate_output_size),
        "gate_dense_base",
    )
    output = prepare_buffer(out, (rows, model_width), "out")

    def prepare_state(
        value: torch.Tensor | None,
        elements: int,
        label: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(elements, device=X.device, dtype=torch.int32)
        if (
            not value.is_cuda
            or value.device != X.device
            or value.dtype != torch.int32
            or value.ndim != 1
            or value.numel() < elements
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA int32 with at least "
                f"{elements} elements"
            )
        return value

    gate_counters = prepare_state(
        gate_feature_counters, gate_output_size // 256, "gate_feature_counters"
    )
    down_counters = prepare_state(
        down_feature_counters, model_width // 256, "down_feature_counters"
    )
    barrier = prepare_state(grid_barrier, 2, "grid_barrier")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_fused_mixed_mlp_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(gate_full_values.data_ptr()),
        ctypes.c_void_p(gate_full_meta_e.data_ptr()),
        ctypes.c_void_p(gate_residual_values.data_ptr()),
        ctypes.c_void_p(gate_residual_meta_e.data_ptr()),
        ctypes.c_void_p(down_full_values.data_ptr()),
        ctypes.c_void_p(down_full_meta_e.data_ptr()),
        ctypes.c_void_p(down_residual_values.data_ptr()),
        ctypes.c_void_p(down_residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(hidden_output.data_ptr()),
        ctypes.c_void_p(compact_base.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(gate_counters.data_ptr()),
        ctypes.c_void_p(down_counters.data_ptr()),
        ctypes.c_void_p(barrier.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(model_width),
        ctypes.c_int(intermediate_size),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 fused mixed MLP failed with code {int(ret)}"
        )
    return (
        output,
        hidden_output,
        compact_base,
        gate_counters,
        down_counters,
        barrier,
    )


def sparse24_cutlass_paired_self_contained_routed_swiglu_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    config: str = "256x64x64_s3_sw4",
    worker_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact dense-row SwiGLU in self-contained correction CTAs."""

    config_ids = {
        "256x64x64_s3_sw4": 0,
        "256x32x64_s3_sw4": 1,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        dense_slot_by_row,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("self-contained routed SwiGLU inputs must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError(
            "self-contained routed SwiGLU inputs must share one device"
        )
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("self-contained routed SwiGLU supports fp16 only")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("self-contained routed SwiGLU metadata must be int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be rank-1 CUDA int32")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("self-contained routed SwiGLU matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("self-contained routed SwiGLU metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("self-contained routed SwiGLU inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    hidden_size = N // 2
    if dense_count <= 0 or dense_count >= full_rows:
        raise ValueError("dense_rows must select 1 through X.shape[0] - 1 rows")
    if full_rows % 8 or K % 64 or N % 256:
        raise ValueError(
            "self-contained routed SwiGLU requires rows % 8, K % 64, and "
            "N % 256 == 0"
        )
    if tuple(dense_slot_by_row.shape) != (full_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(full_rows,)}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )

    if out is None:
        output = torch.empty(
            (full_rows, hidden_size), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (full_rows, hidden_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(full_rows, hidden_size)}"
            )
    if dense_base is None:
        compact_base = torch.empty(
            (dense_count, N), device=X.device, dtype=torch.float16
        )
    else:
        compact_base = dense_base
        if (
            not compact_base.is_cuda
            or compact_base.device != X.device
            or compact_base.dtype != torch.float16
            or tuple(compact_base.shape) != (dense_count, N)
            or not compact_base.is_contiguous()
        ):
            raise ValueError(
                "dense_base must be contiguous CUDA fp16 with shape "
                f"{(dense_count, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = (
        lib.sparse24_cutlass_paired_self_contained_routed_swiglu_f16_stream(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(full_values.data_ptr()),
            ctypes.c_void_p(full_meta_e.data_ptr()),
            ctypes.c_void_p(residual_values.data_ptr()),
            ctypes.c_void_p(residual_meta_e.data_ptr()),
            ctypes.c_void_p(dense_rows.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(compact_base.data_ptr()),
            ctypes.c_int(full_rows),
            ctypes.c_int(dense_count),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_int(config_ids[config]),
            ctypes.c_int(worker_blocks),
            ctypes.c_void_p(stream),
        )
    )
    if int(ret) != 0:
        raise RuntimeError(
            "self-contained routed SwiGLU failed with code " f"{int(ret)}"
        )
    return output, compact_base


def sparse24_cutlass_paired_self_contained_exact_down_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    config: str = "256x32_full_256x32_exact_down",
    worker_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skip full-route dense stores and scatter exact W24+R24 once."""

    config_ids = {"256x32_full_256x32_exact_down": 0}
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")

    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        dense_slot_by_row,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("self-contained exact Down inputs must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("self-contained exact Down inputs must share a device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("self-contained exact Down supports fp16 only")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("self-contained exact Down metadata must be int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be rank-1 CUDA int32")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("self-contained exact Down matrices must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("self-contained exact Down metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("self-contained exact Down inputs must be contiguous")

    full_rows, K = map(int, X.shape)
    dense_count = int(dense_rows.numel())
    N = int(full_values.shape[0])
    if dense_count <= 0 or dense_count >= full_rows:
        raise ValueError("dense_rows must select 1 through X.shape[0] - 1 rows")
    if full_rows % 8 or K % 64 or N % 256:
        raise ValueError(
            "self-contained exact Down requires rows % 8, K % 64, and "
            "N % 256 == 0"
        )
    if tuple(dense_slot_by_row.shape) != (full_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(full_rows,)}"
        )
    expected_values = (N, K // 2)
    if tuple(full_values.shape) != expected_values or tuple(
        residual_values.shape
    ) != expected_values:
        raise ValueError(
            f"full/residual values must both have shape {expected_values}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"full/residual metadata must each have {expected_meta} elements"
        )

    if out is None:
        output = torch.empty(
            (full_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (full_rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(full_rows, N)}"
            )
    if dense_base is None:
        compact_base = torch.empty(
            (dense_count, N), device=X.device, dtype=torch.float16
        )
    else:
        compact_base = dense_base
        if (
            not compact_base.is_cuda
            or compact_base.device != X.device
            or compact_base.dtype != torch.float16
            or tuple(compact_base.shape) != (dense_count, N)
            or not compact_base.is_contiguous()
        ):
            raise ValueError(
                "dense_base must be contiguous CUDA fp16 with shape "
                f"{(dense_count, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_paired_self_contained_exact_down_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(compact_base.data_ptr()),
        ctypes.c_int(full_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "self-contained exact Down failed with code " f"{int(ret)}"
        )
    return output, compact_base


def sparse24_cutlass_gate_dense_down_pipeline_prepacked(
    X: torch.Tensor,
    gate_values: torch.Tensor,
    gate_meta_e: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    hidden: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    row_counters: torch.Tensor | None = None,
    config: str = "256x64_gate_64x64_down",
    worker_blocks: int = 0,
    stage: str = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pipeline static W24 Gate/SwiGLU into dense Down in one grid."""

    config_ids = {
        "256x64_gate_64x64_down": 1,
        "256x64_gate_64x128_down": 2,
        "256x64_gate_128x128_down": 3,
        "128x64_gate_64x128_down_w32x64": 4,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    stage_ids = {"full": 0, "gate_only": 1, "down_only": 2}
    if stage not in stage_ids:
        raise ValueError(
            f"stage must be one of {sorted(stage_ids)}, got {stage!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    tensors = (X, gate_values, gate_meta_e, down_weight)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Gate/Down pipeline inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("Gate/Down pipeline inputs must share one device")
    if (
        X.dtype != torch.float16
        or gate_values.dtype != torch.float16
        or down_weight.dtype != torch.float16
    ):
        raise ValueError("Gate/Down pipeline currently supports fp16")
    if gate_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("gate_meta_e must have dtype torch.uint16/int16")
    if X.ndim != 2 or gate_values.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("Gate/Down pipeline matrices must be rank-2")
    if gate_meta_e.ndim != 1:
        raise ValueError("gate_meta_e must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Gate/Down pipeline inputs must be contiguous")

    rows, model_width = map(int, X.shape)
    down_rows, intermediate_size = map(int, down_weight.shape)
    gate_output_size = int(gate_values.shape[0])
    if down_rows != model_width:
        raise ValueError(
            "down_weight must have shape [model_width, intermediate_size]"
        )
    if gate_output_size != 2 * intermediate_size:
        raise ValueError(
            "gate_values first dimension must equal 2 * intermediate_size"
        )
    if rows % 8 or model_width % 64 or intermediate_size % 128:
        raise ValueError(
            "Gate/Down pipeline requires rows % 8, model_width % 64, and "
            "intermediate_size % 128 == 0"
        )
    if tuple(gate_values.shape) != (gate_output_size, model_width // 2):
        raise ValueError(
            "gate_values must have shape "
            f"{(gate_output_size, model_width // 2)}"
        )
    expected_meta = gate_output_size * (model_width // 16)
    if gate_meta_e.numel() != expected_meta:
        raise ValueError(
            f"gate_meta_e must contain {expected_meta} elements"
        )

    def prepare_output(
        value: torch.Tensor | None,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.empty(shape, device=X.device, dtype=torch.float16)
        if (
            not value.is_cuda
            or value.device != X.device
            or value.dtype != torch.float16
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 shape {shape}"
            )
        return value

    hidden_output = prepare_output(
        hidden, (rows, intermediate_size), "hidden"
    )
    output = prepare_output(out, (rows, model_width), "out")
    down_row_tile = 64 if config_ids[config] == 1 else 128
    row_tiles = (rows + down_row_tile - 1) // down_row_tile
    if row_counters is None:
        counters = torch.zeros(
            row_tiles, device=X.device, dtype=torch.int32
        )
    else:
        counters = row_counters
        if (
            not counters.is_cuda
            or counters.device != X.device
            or counters.dtype != torch.int32
            or counters.ndim != 1
            or counters.numel() < row_tiles
            or not counters.is_contiguous()
        ):
            raise ValueError(
                "row_counters must be contiguous CUDA int32 with at least "
                f"{row_tiles} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_gate_dense_down_pipeline_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(gate_values.data_ptr()),
        ctypes.c_void_p(gate_meta_e.data_ptr()),
        ctypes.c_void_p(hidden_output.data_ptr()),
        ctypes.c_void_p(down_weight.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(model_width),
        ctypes.c_int(intermediate_size),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_int(stage_ids[stage]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 Gate/Down pipeline failed with code " f"{int(ret)}"
        )
    return hidden_output, output, counters


def sparse24_cutlass_gate_sparse_down_pipeline_prepacked(
    X: torch.Tensor,
    gate_values: torch.Tensor,
    gate_meta_e: torch.Tensor,
    down_values: torch.Tensor,
    down_meta_e: torch.Tensor,
    *,
    hidden: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    row_counters: torch.Tensor | None = None,
    config: str = "256x64_gate_256x64_sparse_down",
    worker_blocks: int = 0,
    stage: str = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pipeline static W24 Gate/SwiGLU into W24 Down in one grid."""

    config_ids = {
        "256x64_gate_256x64_sparse_down": 1,
        "128x64_gate_128x64_sparse_down": 2,
        "256x32_gate_256x32_sparse_down": 3,
        "256x64_gate_256x64_sparse_down_dynamic_owners": 4,
        "128x64_gate_128x64_sparse_down_dynamic_owners": 5,
        "256x32_gate_256x32_sparse_down_dynamic_owners": 6,
        "128x64_gate_128x64_sparse_down_grid_barrier": 7,
    }
    if config not in config_ids:
        raise ValueError(
            f"config must be one of {sorted(config_ids)}, got {config!r}"
        )
    stage_ids = {"full": 0, "gate_only": 1, "down_only": 2}
    if stage not in stage_ids:
        raise ValueError(
            f"stage must be one of {sorted(stage_ids)}, got {stage!r}"
        )
    worker_blocks = int(worker_blocks)
    if worker_blocks < 0 or worker_blocks == 1:
        raise ValueError("worker_blocks must be zero (auto) or at least two")
    tensors = (
        X,
        gate_values,
        gate_meta_e,
        down_values,
        down_meta_e,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("sparse Gate/Down pipeline inputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("sparse Gate/Down pipeline inputs must share one device")
    if (
        X.dtype != torch.float16
        or gate_values.dtype != torch.float16
        or down_values.dtype != torch.float16
    ):
        raise ValueError("sparse Gate/Down pipeline currently supports fp16")
    if gate_meta_e.dtype not in (torch.uint16, torch.int16) or (
        down_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("Gate/Down metadata must have dtype torch.uint16/int16")
    if X.ndim != 2 or gate_values.ndim != 2 or down_values.ndim != 2:
        raise ValueError("sparse Gate/Down pipeline matrices must be rank-2")
    if gate_meta_e.ndim != 1 or down_meta_e.ndim != 1:
        raise ValueError("Gate/Down metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("sparse Gate/Down pipeline inputs must be contiguous")

    rows, model_width = map(int, X.shape)
    gate_output_size = int(gate_values.shape[0])
    if gate_output_size % 2:
        raise ValueError("gate_values first dimension must be even")
    intermediate_size = gate_output_size // 2
    if rows % 8 or model_width % 128 or intermediate_size % 128:
        raise ValueError(
            "sparse Gate/Down pipeline requires rows % 8, model_width % 128, "
            "and intermediate_size % 128 == 0"
        )
    if tuple(gate_values.shape) != (gate_output_size, model_width // 2):
        raise ValueError(
            "gate_values must have shape "
            f"{(gate_output_size, model_width // 2)}"
        )
    if tuple(down_values.shape) != (model_width, intermediate_size // 2):
        raise ValueError(
            "down_values must have shape "
            f"{(model_width, intermediate_size // 2)}"
        )
    expected_gate_meta = gate_output_size * (model_width // 16)
    expected_down_meta = model_width * (intermediate_size // 16)
    if gate_meta_e.numel() != expected_gate_meta:
        raise ValueError(
            f"gate_meta_e must contain {expected_gate_meta} elements"
        )
    if down_meta_e.numel() != expected_down_meta:
        raise ValueError(
            f"down_meta_e must contain {expected_down_meta} elements"
        )

    def prepare_output(
        value: torch.Tensor | None,
        shape: tuple[int, int],
        label: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.empty(shape, device=X.device, dtype=torch.float16)
        if (
            not value.is_cuda
            or value.device != X.device
            or value.dtype != torch.float16
            or tuple(value.shape) != shape
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 shape {shape}"
            )
        return value

    hidden_output = prepare_output(
        hidden, (rows, intermediate_size), "hidden"
    )
    output = prepare_output(out, (rows, model_width), "out")
    down_row_tile = 32 if config_ids[config] == 3 else 64
    row_tiles = (rows + down_row_tile - 1) // down_row_tile
    counter_elements = (
        max(row_tiles, 2) if config_ids[config] == 7 else row_tiles
    )
    if row_counters is None:
        counters = torch.zeros(
            counter_elements, device=X.device, dtype=torch.int32
        )
    else:
        counters = row_counters
        if (
            not counters.is_cuda
            or counters.device != X.device
            or counters.dtype != torch.int32
            or counters.ndim != 1
            or counters.numel() < counter_elements
            or not counters.is_contiguous()
        ):
            raise ValueError(
                "row_counters must be contiguous CUDA int32 with at least "
                f"{counter_elements} elements"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_gate_sparse_down_pipeline_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(gate_values.data_ptr()),
        ctypes.c_void_p(gate_meta_e.data_ptr()),
        ctypes.c_void_p(hidden_output.data_ptr()),
        ctypes.c_void_p(down_values.data_ptr()),
        ctypes.c_void_p(down_meta_e.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(counters.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(model_width),
        ctypes.c_int(intermediate_size),
        ctypes.c_int(config_ids[config]),
        ctypes.c_int(worker_blocks),
        ctypes.c_int(stage_ids[stage]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 Gate/sparse-Down pipeline failed with code "
            f"{int(ret)}"
        )
    return hidden_output, output, counters


def sparse24_cutlass_inline_transpose_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str | None = None,
    store_mode: str = "scalar",
) -> torch.Tensor:
    """Experimental SparseGemm visitor that writes contiguous output directly."""

    tensors = (X, a_values, a_meta_e)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("X, a_values, and a_meta_e must be CUDA tensors")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("inline sparse epilogue currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = (int(value) for value in X.shape)
    N = int(a_values.shape[0])
    if M % 8 != 0 or K % 64 != 0 or N % 32 != 0:
        raise ValueError("inline sparse epilogue requires M % 8, K % 64, N % 32 == 0")
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(
            f"a_values must have shape {(N, K // 2)}, got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(
            f"a_meta_e must have {N * (K // 16)} elements, got {a_meta_e.numel()}"
        )
    allowed_configs = {
        None,
        "auto",
        "64x32x64_s3",
        "64x64x64_s5",
        "64x64x64_s6",
        "128x32x64_s4_sw4",
        "128x64x64_s5",
        "256x32x64_s3_sw4",
        "256x64x64_s2_sw4",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported inline sparse epilogue config {config!r}")
    if store_mode not in {"scalar", "vector"}:
        raise ValueError(f"unsupported inline sparse epilogue store {store_mode!r}")

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    if out is None:
        output = torch.empty((M, N), device=X.device, dtype=torch.float16)
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous CUDA fp16 with shape {(M, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_inline_epilogue_config(config):
        with _temporary_inline_epilogue_store(store_mode):
            ret = lib.sparse24_cutlass_inline_transpose_gemm_f16_stream(
                ctypes.c_void_p(Xc.data_ptr()),
                ctypes.c_void_p(a_values.data_ptr()),
                ctypes.c_void_p(a_meta_e.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_int(M),
                ctypes.c_int(K),
                ctypes.c_int(N),
                ctypes.c_void_p(stream),
            )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_inline_transpose_gemm_f16 failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_routed_output_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    dense_count: int,
    out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    config: str | None = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write sparse rows directly and retain dense-row base values compactly."""

    tensors = (X, a_values, a_meta_e, dense_slot_by_row)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed sparse output tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("routed sparse output tensors must share a device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("routed sparse output currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = map(int, X.shape)
    N = int(a_values.shape[0])
    dense_count = int(dense_count)
    if dense_count <= 0 or dense_count > M:
        raise ValueError("routed sparse output requires 0 < dense_count <= M")
    if tuple(dense_slot_by_row.shape) != (M,):
        raise ValueError(f"dense_slot_by_row must have shape {(M,)}")
    if M % 8 != 0 or K % 64 != 0 or N % 32 != 0:
        raise ValueError(
            "routed sparse output requires M % 8, K % 64, N % 32 == 0"
        )
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(
            f"a_values must have shape {(N, K // 2)}, got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(
            f"a_meta_e must have {N * (K // 16)} elements, got {a_meta_e.numel()}"
        )
    allowed_configs = {
        None,
        "auto",
        "64x64x64_s6",
        "128x32x64_s4_sw4",
        "128x64x64_s5",
        "256x32x64_s3_sw4",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported routed sparse output config {config!r}")

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    dense_slot_by_row = dense_slot_by_row.contiguous()
    if out is None:
        output = torch.empty((M, N), device=X.device, dtype=torch.float16)
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or not output.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous CUDA fp16 shape {(M, N)}")
    if dense_base is None:
        compact_base = torch.empty(
            (dense_count, N), device=X.device, dtype=torch.float16
        )
    else:
        compact_base = dense_base
        if (
            not compact_base.is_cuda
            or compact_base.device != X.device
            or compact_base.dtype != torch.float16
            or tuple(compact_base.shape) != (dense_count, N)
            or not compact_base.is_contiguous()
        ):
            raise ValueError(
                "dense_base must be contiguous CUDA fp16 shape "
                f"{(dense_count, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_routed_transpose_config(config):
        ret = lib.sparse24_cutlass_inline_routed_transpose_gemm_f16_stream(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(compact_base.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(dense_count),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed output GEMM failed with code " f"{int(ret)}"
        )
    return output, compact_base


def sparse24_cutlass_routed_residual_epilogue_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    routed_residual: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "256x64x64_s3",
) -> torch.Tensor:
    """Add compact routed residual rows while storing a full sparse GEMM."""

    config_ids = {
        "128x32x64_s4_sw4": 0,
        "128x64x64_s5": 1,
        "256x32x64_s3_sw4": 2,
        "256x64x64_s3": 3,
        "256x64x64_s3_sw4": 4,
    }
    if config not in config_ids:
        raise ValueError(
            f"unsupported routed residual epilogue config {config!r}"
        )
    tensors = (
        X,
        a_values,
        a_meta_e,
        routed_residual,
        dense_slot_by_row,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed residual epilogue tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError(
            "routed residual epilogue tensors must share a device"
        )
    if (
        X.dtype != torch.float16
        or a_values.dtype != torch.float16
        or routed_residual.dtype != torch.float16
    ):
        raise ValueError("routed residual epilogue currently supports fp16")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if (
        X.ndim != 2
        or a_values.ndim != 2
        or a_meta_e.ndim != 1
        or routed_residual.ndim != 2
    ):
        raise ValueError("routed residual epilogue expects rank-2 matrices")

    M, K = map(int, X.shape)
    N = int(a_values.shape[0])
    dense_count = int(routed_residual.shape[0])
    if dense_count <= 0 or dense_count > M:
        raise ValueError("routed residual epilogue requires 0 < dense rows <= M")
    if tuple(routed_residual.shape) != (dense_count, N):
        raise ValueError(
            f"routed_residual must have shape {(dense_count, N)}"
        )
    if tuple(dense_slot_by_row.shape) != (M,):
        raise ValueError(f"dense_slot_by_row must have shape {(M,)}")
    if M % 8 != 0 or K % 64 != 0 or N % 32 != 0:
        raise ValueError(
            "routed residual epilogue requires M % 8, K % 64, N % 32 == 0"
        )
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(
            f"a_values must have shape {(N, K // 2)}, "
            f"got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(
            f"a_meta_e must have {N * (K // 16)} elements, "
            f"got {a_meta_e.numel()}"
        )

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    routed_residual = routed_residual.contiguous()
    dense_slot_by_row = dense_slot_by_row.contiguous()
    if out is None:
        output = torch.empty((M, N), device=X.device, dtype=torch.float16)
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or not output.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous CUDA fp16 shape {(M, N)}")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_routed_residual_epilogue_gemm_f16_stream(
        ctypes.c_void_p(Xc.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(routed_residual.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed residual epilogue GEMM failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_indexed_output_gemm_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    output_rows: int,
    out: torch.Tensor | None = None,
    config: str | None = "auto",
    input_transposed: bool = False,
) -> torch.Tensor:
    """Scatter a sparse GEMM's logical rows from its CUTLASS epilogue."""

    tensors = (X, a_values, a_meta_e, row_indices)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all indexed sparse epilogue tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("all indexed sparse epilogue tensors must share a device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("indexed sparse epilogue currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if row_indices.dtype != torch.int32 or row_indices.ndim != 1:
        raise ValueError("row_indices must be a rank-1 CUDA int32 tensor")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = (int(value) for value in X.shape)
    logical_rows = int(row_indices.numel())
    N = int(a_values.shape[0])
    output_rows = int(output_rows)
    if logical_rows <= 0 or logical_rows > M or output_rows <= 0:
        raise ValueError(
            "indexed sparse epilogue requires 0 < logical rows <= X rows "
            "and output_rows > 0"
        )
    if M % 8 != 0 or K % 64 != 0 or N % 32 != 0:
        raise ValueError(
            "indexed sparse epilogue requires M % 8, K % 64, N % 32 == 0"
        )
    if tuple(a_values.shape) != (N, K // 2):
        raise ValueError(
            f"a_values must have shape {(N, K // 2)}, got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(
            f"a_meta_e must have {N * (K // 16)} elements, got {a_meta_e.numel()}"
        )
    allowed_configs = {
        None,
        "auto",
        "64x64x64_s6",
        "64x64x64_s7",
        "128x32x64_s4_sw4",
        "128x64x64_s3",
        "128x64x64_s4",
        "128x64x64_s5",
        "128x128x64_s3",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported indexed sparse epilogue config {config!r}")

    if input_transposed:
        if int(X.stride(0)) != 1 or int(X.stride(1)) < M:
            raise ValueError(
                "transposed indexed input must have stride (1, ld) with ld >= M"
            )
        Xc = X
        input_ld = int(X.stride(1))
    else:
        Xc = X.contiguous()
        input_ld = K
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    row_indices = row_indices.contiguous()
    if out is None:
        output = torch.empty(
            (output_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_inline_epilogue_config(config):
        ret = lib.sparse24_cutlass_inline_indexed_transpose_gemm_f16_stream(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(row_indices.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(logical_rows),
            ctypes.c_int(output_rows),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_int(bool(input_transposed)),
            ctypes.c_int(input_ld),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_inline_indexed_transpose_gemm_f16 failed "
            f"with code {int(ret)}"
        )
    return output


def _validate_routed_exact_tensors(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    out: torch.Tensor | None,
    *,
    swiglu: bool,
) -> tuple[torch.Tensor, int, int, int, int, int]:
    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        sparse_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("routed exact sparse tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("routed exact sparse tensors must share a device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("routed exact sparse kernels currently support fp16 only")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("routed exact metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or sparse_rows.dtype != torch.int32:
        raise ValueError("routed exact row indices must be int32")
    if dense_rows.ndim != 1 or sparse_rows.ndim != 1:
        raise ValueError("routed exact row indices must be rank-1")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("routed exact activations and values must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("routed exact metadata must be rank-1")

    output_rows, K = map(int, X.shape)
    N = int(full_values.shape[0])
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    output_columns = N // 2 if swiglu else N
    if dense_count <= 0 or sparse_count <= 0:
        raise ValueError("routed exact kernels require both dense and sparse rows")
    if dense_count + sparse_count != output_rows:
        raise ValueError(
            "dense and sparse route sizes must partition X rows exactly"
        )
    required_n_multiple = 256 if swiglu else 128
    if K % 64 or N % required_n_multiple:
        raise ValueError(
            f"routed exact kernel requires K % 64 and N % "
            f"{required_n_multiple} == 0"
        )
    if tuple(full_values.shape) != (N, K // 2) or tuple(
        residual_values.shape
    ) != (N, K // 2):
        raise ValueError(
            f"routed exact values must have shape {(N, K // 2)}"
        )
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"routed exact metadata must have {expected_meta} elements"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("routed exact tensors must be contiguous")

    if out is None:
        output = torch.empty(
            (output_rows, output_columns),
            device=X.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, output_columns)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, output_columns)}"
            )
    return output, output_rows, dense_count, sparse_count, K, N


def sparse24_cutlass_routed_exact_linear_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Execute exact dense/sparse rows without compact route buffers."""

    configs = {
        "auto": 0,
        "128x32x64_s4_sw4": 1,
        "128x64x64_s5": 2,
        "256x64x64_s3": 3,
        "256x32_sparse_256x32_dense_s3_f16": 4,
        "256x64_sparse_256x32_dense_s3_f16": 5,
        "256x128_sparse_256x32_dense_s2_f16": 6,
        "256x64_sparse_256x64_dense_s3_f16": 7,
        "256x128_sparse_256x64_dense_s2_f16": 8,
    }
    if config not in configs:
        raise ValueError(f"unsupported routed exact linear config {config!r}")
    output, output_rows, dense_count, sparse_count, K, N = (
        _validate_routed_exact_tensors(
            X,
            full_values,
            full_meta_e,
            residual_values,
            residual_meta_e,
            dense_rows,
            sparse_rows,
            out,
            swiglu=False,
        )
    )
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_routed_exact_linear_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(sparse_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(sparse_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 routed exact linear failed with code {int(ret)}"
        )
    return output


def sparse24_cutlass_heterogeneous_linear_prepacked(
    X: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_meta_e: torch.Tensor,
    dense_weight: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Route exact dense rows and 2:4 rows in one gather-free launch."""

    configs = {
        "auto": 0,
        "128x32x64_s4_sw4": 1,
        "128x64x64_s5": 2,
        "256x64x64_s3": 3,
        "128x32x64_s4_sw4_direct": 4,
        "128x64x64_s5_direct": 5,
        "256x64x64_s3_direct": 6,
        "256x32x64_s3_sw4": 7,
        "256x32x64_s3_sw4_direct": 8,
        "256x64x64_s2_sw4": 9,
        "256x64x64_s2_sw4_direct": 10,
        "256x32x64_s3_sw4_f16": 11,
        "256x64_sparse_128x32_dense_s3": 12,
        "256x64_sparse_128x32_dense_s3_f16": 13,
        "256x64_sparse_128x64_dense_s3_f16": 16,
        "256x128_sparse_128x64_dense_s2": 17,
        "256x128_sparse_128x64_dense_s2_f16": 18,
    }
    if config not in configs:
        raise ValueError(
            f"unsupported heterogeneous routed linear config {config!r}"
        )
    tensors = (
        X,
        sparse_values,
        sparse_meta_e,
        dense_weight,
        dense_rows,
        sparse_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("heterogeneous routed tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("heterogeneous routed tensors must share a device")
    if (
        X.dtype != torch.float16
        or sparse_values.dtype != torch.float16
        or dense_weight.dtype != torch.float16
    ):
        raise ValueError(
            "heterogeneous routed kernels currently support fp16 only"
        )
    if sparse_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("heterogeneous routed metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or sparse_rows.dtype != torch.int32:
        raise ValueError("heterogeneous routed row indices must be int32")
    if X.ndim != 2 or sparse_values.ndim != 2 or dense_weight.ndim != 2:
        raise ValueError("heterogeneous routed matrices must be rank-2")
    if (
        sparse_meta_e.ndim != 1
        or dense_rows.ndim != 1
        or sparse_rows.ndim != 1
    ):
        raise ValueError(
            "heterogeneous metadata and row indices must be rank-1"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("heterogeneous routed tensors must be contiguous")

    output_rows, K = map(int, X.shape)
    N = int(dense_weight.shape[0])
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    if dense_count <= 0 or sparse_count <= 0:
        raise ValueError(
            "heterogeneous routed kernels require both dense and sparse rows"
        )
    if dense_count + sparse_count != output_rows:
        raise ValueError(
            "dense and sparse route sizes must partition X rows exactly"
        )
    if K % 64 or N % 128:
        raise ValueError(
            "heterogeneous routed kernel requires K % 64 and N % 128 == 0"
        )
    if tuple(dense_weight.shape) != (N, K):
        raise ValueError(f"dense_weight must have shape {(N, K)}")
    if tuple(sparse_values.shape) != (N, K // 2):
        raise ValueError(f"sparse_values must have shape {(N, K // 2)}")
    expected_meta = N * (K // 16)
    if sparse_meta_e.numel() != expected_meta:
        raise ValueError(
            f"heterogeneous metadata must have {expected_meta} elements"
        )

    if out is None:
        output = torch.empty(
            (output_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_heterogeneous_linear_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(sparse_values.data_ptr()),
        ctypes.c_void_p(sparse_meta_e.data_ptr()),
        ctypes.c_void_p(dense_weight.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(sparse_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(sparse_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 heterogeneous routed linear failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_heterogeneous_swiglu_prepacked(
    X: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_meta_e: torch.Tensor,
    dense_weight: torch.Tensor,
    dense_weight_rows: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse heterogeneous Gate/Up routing and indexed SwiGLU output."""

    configs = {
        "auto": 0,
        "256x32x64_s3_sw4_f16": 1,
    }
    if config not in configs:
        raise ValueError(
            f"unsupported heterogeneous SwiGLU config {config!r}"
        )
    tensors = (
        X,
        sparse_values,
        sparse_meta_e,
        dense_weight,
        dense_weight_rows,
        dense_rows,
        sparse_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("heterogeneous SwiGLU tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError(
            "heterogeneous SwiGLU tensors must share a device"
        )
    if (
        X.dtype != torch.float16
        or sparse_values.dtype != torch.float16
        or dense_weight.dtype != torch.float16
    ):
        raise ValueError("heterogeneous SwiGLU supports fp16 only")
    if sparse_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("heterogeneous SwiGLU metadata must be uint16/int16")
    if any(
        tensor.dtype != torch.int32
        for tensor in (dense_weight_rows, dense_rows, sparse_rows)
    ):
        raise ValueError("heterogeneous SwiGLU indices must be int32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("heterogeneous SwiGLU tensors must be contiguous")

    output_rows, K = map(int, X.shape)
    N = int(dense_weight.shape[0])
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    if dense_count <= 0 or sparse_count <= 0:
        raise ValueError(
            "heterogeneous SwiGLU requires dense and sparse rows"
        )
    if dense_count + sparse_count != output_rows:
        raise ValueError(
            "dense and sparse route sizes must partition X rows exactly"
        )
    if K % 64 or N % 256:
        raise ValueError(
            "heterogeneous SwiGLU requires K % 64 and N % 256 == 0"
        )
    if tuple(dense_weight.shape) != (N, K):
        raise ValueError(f"dense_weight must have shape {(N, K)}")
    if tuple(sparse_values.shape) != (N, K // 2):
        raise ValueError(f"sparse_values must have shape {(N, K // 2)}")
    if tuple(dense_weight_rows.shape) != (N,):
        raise ValueError(f"dense_weight_rows must have shape {(N,)}")
    expected_meta = N * (K // 16)
    if sparse_meta_e.numel() != expected_meta:
        raise ValueError(
            f"heterogeneous metadata must have {expected_meta} elements"
        )

    hidden_size = N // 2
    if out is None:
        output = torch.empty(
            (output_rows, hidden_size),
            device=X.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, hidden_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, hidden_size)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_heterogeneous_swiglu_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(sparse_values.data_ptr()),
        ctypes.c_void_p(sparse_meta_e.data_ptr()),
        ctypes.c_void_p(dense_weight.data_ptr()),
        ctypes.c_void_p(dense_weight_rows.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(sparse_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(sparse_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 heterogeneous SwiGLU failed with code {int(ret)}"
        )
    return output


def sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked(
    X: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_meta_e: torch.Tensor,
    dense_weight: torch.Tensor,
    dense_weight_rows: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse contiguous all-row W24 SwiGLU with exact dense-row overrides."""

    configs = {
        "auto": 0,
        "256x32_sparse_128x32_dense_f16": 1,
        "256x64_sparse_128x64_dense_f16": 2,
        "256x64_sparse_128x64_dense_f16_w1": 3,
        "256x64_sparse_128x64_dense_f16_w3": 4,
        "256x64_sparse_128x64_dense_f16_w4": 5,
    }
    if config not in configs:
        raise ValueError(
            f"unsupported full-sparse dense-override SwiGLU config {config!r}"
        )
    tensors = (
        X,
        sparse_values,
        sparse_meta_e,
        dense_weight,
        dense_weight_rows,
        dense_rows,
        dense_slot_by_row,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("dense-override SwiGLU tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("dense-override SwiGLU tensors must share a device")
    if (
        X.dtype != torch.float16
        or sparse_values.dtype != torch.float16
        or dense_weight.dtype != torch.float16
    ):
        raise ValueError("dense-override SwiGLU supports fp16 only")
    if sparse_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("dense-override metadata must be uint16/int16")
    if any(
        tensor.dtype != torch.int32
        for tensor in (dense_weight_rows, dense_rows, dense_slot_by_row)
    ):
        raise ValueError("dense-override indices must be int32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("dense-override SwiGLU tensors must be contiguous")

    output_rows, K = map(int, X.shape)
    N = int(dense_weight.shape[0])
    dense_count = int(dense_rows.numel())
    if dense_count <= 0 or dense_count > output_rows:
        raise ValueError("dense row count must be in [1, output_rows]")
    if K % 64 or N % 256:
        raise ValueError(
            "dense-override SwiGLU requires K % 64 and N % 256 == 0"
        )
    if tuple(dense_weight.shape) != (N, K):
        raise ValueError(f"dense_weight must have shape {(N, K)}")
    if tuple(sparse_values.shape) != (N, K // 2):
        raise ValueError(f"sparse_values must have shape {(N, K // 2)}")
    if tuple(dense_weight_rows.shape) != (N,):
        raise ValueError(f"dense_weight_rows must have shape {(N,)}")
    if tuple(dense_slot_by_row.shape) != (output_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(output_rows,)}"
        )
    expected_meta = N * (K // 16)
    if sparse_meta_e.numel() != expected_meta:
        raise ValueError(
            f"dense-override metadata must have {expected_meta} elements"
        )

    hidden_size = N // 2
    if out is None:
        output = torch.empty(
            (output_rows, hidden_size),
            device=X.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, hidden_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, hidden_size)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_full_sparse_dense_override_swiglu_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(sparse_values.data_ptr()),
        ctypes.c_void_p(sparse_meta_e.data_ptr()),
        ctypes.c_void_p(dense_weight.data_ptr()),
        ctypes.c_void_p(dense_weight_rows.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 full-sparse dense-override SwiGLU failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_cutlass_full_sparse_dense_override_linear_prepacked(
    X: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_meta_e: torch.Tensor,
    dense_weight: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse contiguous all-row W24 with exact selected-row dense stores."""

    configs = {
        "auto": 0,
        "256x32_sparse_128x32_dense_f16": 1,
        "256x64_sparse_128x64_dense_f16": 2,
    }
    if config not in configs:
        raise ValueError(
            "unsupported full-sparse dense-override linear config "
            f"{config!r}"
        )
    tensors = (
        X,
        sparse_values,
        sparse_meta_e,
        dense_weight,
        dense_rows,
        dense_slot_by_row,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("dense-override linear tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("dense-override linear tensors must share a device")
    if (
        X.dtype != torch.float16
        or sparse_values.dtype != torch.float16
        or dense_weight.dtype != torch.float16
    ):
        raise ValueError("dense-override linear supports fp16 only")
    if sparse_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("dense-override metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_slot_by_row.dtype != torch.int32:
        raise ValueError("dense-override row indices must be int32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("dense-override linear tensors must be contiguous")

    output_rows, K = map(int, X.shape)
    N = int(dense_weight.shape[0])
    dense_count = int(dense_rows.numel())
    if dense_count <= 0 or dense_count > output_rows:
        raise ValueError("dense row count must be in [1, output_rows]")
    if K % 64 or N % 256:
        raise ValueError(
            "dense-override linear requires K % 64 and N % 256 == 0"
        )
    if tuple(dense_weight.shape) != (N, K):
        raise ValueError(f"dense_weight must have shape {(N, K)}")
    if tuple(sparse_values.shape) != (N, K // 2):
        raise ValueError(f"sparse_values must have shape {(N, K // 2)}")
    if tuple(dense_rows.shape) != (dense_count,):
        raise ValueError("dense_rows must be rank-1")
    if tuple(dense_slot_by_row.shape) != (output_rows,):
        raise ValueError(
            f"dense_slot_by_row must have shape {(output_rows,)}"
        )
    expected_meta = N * (K // 16)
    if sparse_meta_e.numel() != expected_meta:
        raise ValueError(
            f"dense-override metadata must have {expected_meta} elements"
        )

    if out is None:
        output = torch.empty(
            (output_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_full_sparse_dense_override_linear_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(sparse_values.data_ptr()),
        ctypes.c_void_p(sparse_meta_e.data_ptr()),
        ctypes.c_void_p(dense_weight.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 full-sparse dense-override linear failed with code "
            f"{int(ret)}"
        )
    return output


def _heterogeneous_component_output(
    X: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    N: int,
) -> tuple[int, int, int]:
    if not X.is_cuda or not row_indices.is_cuda or not out.is_cuda:
        raise ValueError("routed component tensors must be CUDA")
    if row_indices.device != X.device or out.device != X.device:
        raise ValueError("routed component tensors must share a device")
    if X.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("routed component matrices must be fp16")
    if row_indices.dtype != torch.int32 or row_indices.ndim != 1:
        raise ValueError("routed component row indices must be rank-1 int32")
    if X.ndim != 2 or out.ndim != 2:
        raise ValueError("routed component matrices must be rank-2")
    if not X.is_contiguous() or not row_indices.is_contiguous() or not out.is_contiguous():
        raise ValueError("routed component tensors must be contiguous")
    output_rows, K = map(int, X.shape)
    route_count = int(row_indices.numel())
    if route_count <= 0 or route_count > output_rows:
        raise ValueError("routed component row count must be in [1, M]")
    if tuple(out.shape) != (output_rows, N):
        raise ValueError(f"out must have shape {(output_rows, N)}")
    if K % 64 or N % 128:
        raise ValueError("routed component requires K % 64 and N % 128 == 0")
    return output_rows, route_count, K


def _run_heterogeneous_component(
    X: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    sparse_values: torch.Tensor | None,
    sparse_meta_e: torch.Tensor | None,
    dense_weight: torch.Tensor | None,
    N: int,
    config: str,
    dense_component: bool,
) -> torch.Tensor:
    configs = {
        "auto": 0,
        "128x32x64_s4_sw4": 1,
        "128x64x64_s5": 2,
        "256x64x64_s3": 3,
        "256x32x64_s3_sw4": 7,
        "256x64x64_s2_sw4": 9,
    }
    if config not in configs:
        raise ValueError(f"unsupported routed component config {config!r}")
    output_rows, route_count, K = _heterogeneous_component_output(
        X, row_indices, out, N=N
    )
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_heterogeneous_component_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(
            0 if sparse_values is None else sparse_values.data_ptr()
        ),
        ctypes.c_void_p(
            0 if sparse_meta_e is None else sparse_meta_e.data_ptr()
        ),
        ctypes.c_void_p(
            0 if dense_weight is None else dense_weight.data_ptr()
        ),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(route_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_int(int(dense_component)),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 heterogeneous routed component failed with code "
            f"{int(ret)}"
        )
    return out


def sparse24_cutlass_routed_sparse_rows_prepacked(
    X: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_meta_e: torch.Tensor,
    sparse_rows: torch.Tensor,
    *,
    out: torch.Tensor,
    config: str = "auto",
) -> torch.Tensor:
    """Write selected 2:4 rows directly into their full-output positions."""

    if not sparse_values.is_cuda or not sparse_meta_e.is_cuda:
        raise ValueError("routed sparse operands must be CUDA")
    if sparse_values.device != X.device or sparse_meta_e.device != X.device:
        raise ValueError("routed sparse operands must share X's device")
    if sparse_values.dtype != torch.float16 or sparse_values.ndim != 2:
        raise ValueError("routed sparse values must be rank-2 fp16")
    if sparse_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("routed sparse metadata must be uint16/int16")
    if sparse_meta_e.ndim != 1:
        raise ValueError("routed sparse metadata must be rank-1")
    if not sparse_values.is_contiguous() or not sparse_meta_e.is_contiguous():
        raise ValueError("routed sparse operands must be contiguous")
    K = int(X.shape[1])
    N = int(sparse_values.shape[0])
    if tuple(sparse_values.shape) != (N, K // 2):
        raise ValueError(f"sparse_values must have shape {(N, K // 2)}")
    if sparse_meta_e.numel() != N * (K // 16):
        raise ValueError("routed sparse metadata has the wrong size")
    return _run_heterogeneous_component(
        X,
        sparse_rows,
        out,
        sparse_values=sparse_values,
        sparse_meta_e=sparse_meta_e,
        dense_weight=None,
        N=N,
        config=config,
        dense_component=False,
    )


def dense_cutlass_routed_rows_weight_t(
    X: torch.Tensor,
    dense_weight: torch.Tensor,
    dense_rows: torch.Tensor,
    *,
    out: torch.Tensor,
    config: str = "auto",
) -> torch.Tensor:
    """Write selected exact dense rows directly into a full output tensor."""

    if not dense_weight.is_cuda or dense_weight.device != X.device:
        raise ValueError("routed dense weight must be CUDA on X's device")
    if (
        dense_weight.dtype != torch.float16
        or dense_weight.ndim != 2
        or not dense_weight.is_contiguous()
    ):
        raise ValueError("routed dense weight must be contiguous rank-2 fp16")
    N, K = map(int, dense_weight.shape)
    if int(X.shape[1]) != K:
        raise ValueError(f"dense_weight must have shape {(N, int(X.shape[1]))}")
    return _run_heterogeneous_component(
        X,
        dense_rows,
        out,
        sparse_values=None,
        sparse_meta_e=None,
        dense_weight=dense_weight,
        N=N,
        config=config,
        dense_component=True,
    )


def sparse24_cutlass_routed_exact_swiglu_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse exact row routing, sparse Gate/Up GEMMs, and SwiGLU."""

    configs = {
        "auto": 0,
        "256x32x64_s3_sw4": 1,
        "256x64x64_s3_sw4": 2,
    }
    if config not in configs:
        raise ValueError(f"unsupported routed exact SwiGLU config {config!r}")
    output, output_rows, dense_count, sparse_count, K, N = (
        _validate_routed_exact_tensors(
            X,
            full_values,
            full_meta_e,
            residual_values,
            residual_meta_e,
            dense_rows,
            sparse_rows,
            out,
            swiglu=True,
        )
    )
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_routed_exact_swiglu_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(sparse_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(sparse_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 routed exact SwiGLU failed with code {int(ret)}"
        )
    return output


def _validate_grouped_owner_tensors(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    out: torch.Tensor | None,
    *,
    swiglu: bool,
) -> tuple[torch.Tensor, int, int, int, int]:
    tensors = (
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("grouped owner sparse tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("grouped owner sparse tensors must share a device")
    if (
        X.dtype != torch.float16
        or full_values.dtype != torch.float16
        or residual_values.dtype != torch.float16
    ):
        raise ValueError("grouped owner sparse kernels currently support fp16 only")
    if full_meta_e.dtype not in (torch.uint16, torch.int16) or (
        residual_meta_e.dtype not in (torch.uint16, torch.int16)
    ):
        raise ValueError("grouped owner metadata must be uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 int32 CUDA tensor")
    if X.ndim != 2 or full_values.ndim != 2 or residual_values.ndim != 2:
        raise ValueError("grouped owner activations and values must be rank-2")
    if full_meta_e.ndim != 1 or residual_meta_e.ndim != 1:
        raise ValueError("grouped owner metadata must be rank-1")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped owner tensors must be contiguous")

    output_rows, K = map(int, X.shape)
    N = int(full_values.shape[0])
    dense_count = int(dense_rows.numel())
    output_columns = N // 2 if swiglu else N
    if dense_count <= 0 or dense_count > output_rows:
        raise ValueError("dense_rows must select between 1 and X.shape[0] rows")
    required_n_multiple = 256 if swiglu else 128
    if K % 64 or N % required_n_multiple:
        raise ValueError(
            f"grouped owner kernel requires K % 64 and N % "
            f"{required_n_multiple} == 0"
        )
    if tuple(full_values.shape) != (N, K // 2) or tuple(
        residual_values.shape
    ) != (N, K // 2):
        raise ValueError(f"grouped owner values must have shape {(N, K // 2)}")
    expected_meta = N * (K // 16)
    if full_meta_e.numel() != expected_meta or (
        residual_meta_e.numel() != expected_meta
    ):
        raise ValueError(
            f"grouped owner metadata must have {expected_meta} elements"
        )
    if out is None:
        output = torch.empty(
            (output_rows, output_columns),
            device=X.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, output_columns)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(output_rows, output_columns)}"
            )
    return output, output_rows, dense_count, K, N


def sparse24_cutlass_grouped_owner_linear_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    group_tiles: int = 2,
    config: str = "auto",
) -> torch.Tensor:
    """Run contiguous full sparse GEMM plus race-free indexed residual add."""

    configs = {"auto": 0, "64x32x64_s3": 1, "128x32x64_s4_sw4": 2}
    if config not in configs:
        raise ValueError(f"unsupported grouped owner linear config {config!r}")
    group_tiles = int(group_tiles)
    if group_tiles not in (1, 2, 3, 4):
        raise ValueError("group_tiles must be one of 1, 2, 3, or 4")
    output, output_rows, dense_count, K, N = _validate_grouped_owner_tensors(
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        out,
        swiglu=False,
    )
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_grouped_owner_linear_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(group_tiles),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 grouped owner linear failed with code {int(ret)}"
        )
    return output


def sparse24_cutlass_grouped_owner_swiglu_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    group_tiles: int = 2,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse grouped full Gate/Up, residual correction, and indexed SwiGLU."""

    configs = {"auto": 0, "256x32x64_s3_sw4": 1}
    if config not in configs:
        raise ValueError(f"unsupported grouped owner SwiGLU config {config!r}")
    group_tiles = int(group_tiles)
    if group_tiles not in (1, 2, 3, 4):
        raise ValueError("group_tiles must be one of 1, 2, 3, or 4")
    output, output_rows, dense_count, K, N = _validate_grouped_owner_tensors(
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        out,
        swiglu=True,
    )
    if (
        not dense_slot_by_row.is_cuda
        or dense_slot_by_row.device != X.device
        or dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (output_rows,)
        or not dense_slot_by_row.is_contiguous()
    ):
        raise ValueError(
            "dense_slot_by_row must be contiguous CUDA int32 with shape "
            f"{(output_rows,)}"
        )
    if dense_base is None:
        dense_base_output = torch.empty(
            (dense_count, N), device=X.device, dtype=torch.float16
        )
    else:
        dense_base_output = dense_base
        if (
            not dense_base_output.is_cuda
            or dense_base_output.device != X.device
            or dense_base_output.dtype != torch.float16
            or tuple(dense_base_output.shape) != (dense_count, N)
            or not dense_base_output.is_contiguous()
        ):
            raise ValueError(
                "dense_base must be contiguous CUDA fp16 with shape "
                f"{(dense_count, N)}"
            )
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_grouped_owner_swiglu_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(dense_base_output.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(group_tiles),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 grouped owner SwiGLU failed with code {int(ret)}"
        )
    return output


def sparse24_cutlass_grouped_owner_qkv_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    dense_rows: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    dense_base: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    group_tiles: int = 2,
    config: str = "256x32x64_s3_w64x32",
) -> torch.Tensor:
    """Fuse grouped W24/R24 QKV epilogues without global counters."""

    configs = {
        "256x32x64_s3_w64x32": 1,
        "256x64x64_s3_w64x32": 2,
    }
    if config not in configs:
        raise ValueError(f"unsupported grouped owner QKV config {config!r}")
    group_tiles = int(group_tiles)
    if group_tiles < 1 or group_tiles > 16:
        raise ValueError("group_tiles must be between 1 and 16")
    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if head_dim != 128 or not is_neox:
        raise ValueError(
            "grouped owner QKV requires Neox RoPE with head_dim=128"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    output, output_rows, dense_count, K, N = _validate_grouped_owner_tensors(
        X,
        full_values,
        full_meta_e,
        residual_values,
        residual_meta_e,
        dense_rows,
        out,
        swiglu=False,
    )
    if N != q_size + 2 * kv_size:
        raise ValueError(
            f"QKV output size mismatch: N={N}, q_size={q_size}, kv_size={kv_size}"
        )
    if N % 256 or q_size % 256 or kv_size % 256:
        raise ValueError("grouped owner QKV requires N/Q/KV divisible by 256")
    if (
        not dense_slot_by_row.is_cuda
        or dense_slot_by_row.device != X.device
        or dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (output_rows,)
        or not dense_slot_by_row.is_contiguous()
    ):
        raise ValueError(
            "dense_slot_by_row must be contiguous CUDA int32 with shape "
            f"{(output_rows,)}"
        )
    if (
        not dense_base.is_cuda
        or dense_base.device != X.device
        or dense_base.dtype != torch.float16
        or tuple(dense_base.shape) != (dense_count, N)
        or not dense_base.is_contiguous()
    ):
        raise ValueError(
            "dense_base must be contiguous CUDA fp16 with shape "
            f"{(dense_count, N)}"
        )
    if (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != X.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[1]) != head_dim
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError(
            f"cos_sin_cache must be contiguous CUDA fp16 with width {head_dim}"
        )
    if (
        not position_ids.is_cuda
        or position_ids.device != X.device
        or position_ids.dtype != torch.int64
        or position_ids.numel() != output_rows
        or not position_ids.is_contiguous()
    ):
        raise ValueError(
            "position_ids must be contiguous CUDA int64 with "
            f"{output_rows} elements"
        )
    if normalize_qk:
        for label, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
            assert weight is not None
            if (
                not weight.is_cuda
                or weight.device != X.device
                or weight.dtype != torch.float16
                or tuple(weight.shape) != (head_dim,)
                or not weight.is_contiguous()
            ):
                raise ValueError(
                    f"{label} must be contiguous CUDA fp16 shape {(head_dim,)}"
                )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_grouped_owner_qkv_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(q_weight.data_ptr()) if q_weight is not None else None,
        ctypes.c_void_p(k_weight.data_ptr()) if k_weight is not None else None,
        ctypes.c_void_p(cos_sin_cache.data_ptr()),
        ctypes.c_void_p(position_ids.data_ptr()),
        ctypes.c_int(output_rows),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(q_size),
        ctypes.c_int(kv_size),
        ctypes.c_int(head_dim),
        ctypes.c_float(epsilon),
        ctypes.c_int(int(is_neox)),
        ctypes.c_int(int(normalize_qk)),
        ctypes.c_int(group_tiles),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 grouped owner QKV failed with code {int(ret)}"
        )
    return output


def sparse24_cutlass_gate_up_swiglu_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str | None = None,
    output_transposed: bool = False,
) -> torch.Tensor:
    """Run sparse gate/up GEMM with an inline SwiGLU epilogue."""

    tensors = (X, a_values, a_meta_e)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("X, a_values, and a_meta_e must be CUDA tensors")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("inline sparse SwiGLU epilogue supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = (int(value) for value in X.shape)
    output_size = int(a_values.shape[0])
    hidden_size = output_size // 2
    if M % 8 != 0 or K % 64 != 0 or output_size % 256 != 0:
        raise ValueError(
            "inline sparse SwiGLU requires M % 8, K % 64, "
            "and output_size % 256 == 0"
        )
    if tuple(a_values.shape) != (output_size, K // 2):
        raise ValueError(
            f"a_values must have shape {(output_size, K // 2)}, "
            f"got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != output_size * (K // 16):
        raise ValueError(
            f"a_meta_e must have {output_size * (K // 16)} elements, "
            f"got {a_meta_e.numel()}"
        )
    allowed_configs = {
        None,
        "auto",
        "256x32x64_s3",
        "256x32x64_s3_sw4",
        "256x32x64_s3_sw4_f16",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
        "256x64x64_s3_sw4_f16",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported inline sparse SwiGLU config {config!r}")

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    if out is None:
        if output_transposed:
            output = torch.empty_strided(
                (M, hidden_size),
                (1, M),
                device=X.device,
                dtype=torch.float16,
            )
        else:
            output = torch.empty(
                (M, hidden_size), device=X.device, dtype=torch.float16
            )
    else:
        output = out
        expected_stride = (1, M) if output_transposed else (hidden_size, 1)
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, hidden_size)
            or tuple(output.stride()) != expected_stride
        ):
            raise ValueError(
                "out must be CUDA fp16 with shape/stride "
                f"{(M, hidden_size)}/{expected_stride}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_swiglu_epilogue_config(config):
        fn = (
            lib.sparse24_cutlass_inline_swiglu_transposed_gemm_f16_stream
            if output_transposed
            else lib.sparse24_cutlass_inline_swiglu_gemm_f16_stream
        )
        ret = fn(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(output_size),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 inline SwiGLU GEMM failed with code " f"{int(ret)}"
        )
    return output


def sparse24_cutlass_routed_swiglu_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    dense_count: int,
    out: torch.Tensor | None = None,
    dense_base: torch.Tensor | None = None,
    config: str | None = None,
    output_transposed: bool = False,
    write_dense_approx: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse sparse-row SwiGLU while retaining compact dense-row gate/up."""

    tensors = (X, a_values, a_meta_e, dense_slot_by_row)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed SwiGLU tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("routed SwiGLU tensors must share a device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("routed sparse SwiGLU currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be a rank-1 CUDA int32 tensor")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = (int(value) for value in X.shape)
    output_size = int(a_values.shape[0])
    hidden_size = output_size // 2
    dense_count = int(dense_count)
    if dense_count <= 0 or dense_count > M:
        raise ValueError("routed SwiGLU requires 0 < dense_count <= M")
    if tuple(dense_slot_by_row.shape) != (M,):
        raise ValueError(f"dense_slot_by_row must have shape {(M,)}")
    if M % 8 != 0 or K % 64 != 0 or output_size % 256 != 0:
        raise ValueError(
            "routed sparse SwiGLU requires M % 8, K % 64, "
            "and output_size % 256 == 0"
        )
    if tuple(a_values.shape) != (output_size, K // 2):
        raise ValueError(
            f"a_values must have shape {(output_size, K // 2)}, "
            f"got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != output_size * (K // 16):
        raise ValueError(
            f"a_meta_e must have {output_size * (K // 16)} elements, "
            f"got {a_meta_e.numel()}"
        )
    allowed_configs = {
        None,
        "auto",
        "256x32x64_s3_sw4",
        "256x64x64_s2_sw4",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported routed sparse SwiGLU config {config!r}")
    approx_configs = {
        None: 0,
        "auto": 0,
        "256x32x64_s3_sw4": 1,
        "256x64x64_s3_sw4": 2,
    }
    if write_dense_approx and output_transposed:
        raise ValueError("write_dense_approx requires contiguous output")
    if write_dense_approx and config not in approx_configs:
        raise ValueError(
            f"unsupported routed approximate SwiGLU config {config!r}"
        )

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    dense_slot_by_row = dense_slot_by_row.contiguous()
    if out is None:
        if output_transposed:
            output = torch.empty_strided(
                (M, hidden_size),
                (1, M),
                device=X.device,
                dtype=torch.float16,
            )
        else:
            output = torch.empty(
                (M, hidden_size), device=X.device, dtype=torch.float16
            )
    else:
        output = out
        output_stride = (1, M) if output_transposed else (hidden_size, 1)
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, hidden_size)
            or tuple(output.stride()) != output_stride
        ):
            raise ValueError(
                "out must be CUDA fp16 with shape/stride "
                f"{(M, hidden_size)}/{output_stride}"
            )
    if dense_base is None:
        if output_transposed:
            compact_base = torch.empty_strided(
                (dense_count, output_size),
                (1, dense_count),
                device=X.device,
                dtype=torch.float16,
            )
        else:
            compact_base = torch.empty(
                (dense_count, output_size), device=X.device, dtype=torch.float16
            )
    else:
        compact_base = dense_base
        dense_base_stride = (
            (1, dense_count)
            if output_transposed
            else (output_size, 1)
        )
        if (
            not compact_base.is_cuda
            or compact_base.device != X.device
            or compact_base.dtype != torch.float16
            or tuple(compact_base.shape) != (dense_count, output_size)
            or tuple(compact_base.stride()) != dense_base_stride
        ):
            raise ValueError(
                "dense_base must be CUDA fp16 with shape/stride "
                f"{(dense_count, output_size)}/{dense_base_stride}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    if write_dense_approx:
        ret = lib.sparse24_cutlass_inline_routed_approx_swiglu_gemm_f16_stream(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(compact_base.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(dense_count),
            ctypes.c_int(K),
            ctypes.c_int(output_size),
            ctypes.c_int(approx_configs[config]),
            ctypes.c_void_p(stream),
        )
    else:
        with _temporary_swiglu_epilogue_config(config):
            ret = lib.sparse24_cutlass_inline_routed_swiglu_gemm_f16_stream(
                ctypes.c_void_p(Xc.data_ptr()),
                ctypes.c_void_p(a_values.data_ptr()),
                ctypes.c_void_p(a_meta_e.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_void_p(compact_base.data_ptr()),
                ctypes.c_void_p(dense_slot_by_row.data_ptr()),
                ctypes.c_int(M),
                ctypes.c_int(dense_count),
                ctypes.c_int(K),
                ctypes.c_int(output_size),
                ctypes.c_int(bool(output_transposed)),
                ctypes.c_void_p(stream),
            )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed SwiGLU GEMM failed with code " f"{int(ret)}"
        )
    return output, compact_base


def _validate_indexed_swiglu_tensors(
    X: torch.Tensor,
    values: tuple[torch.Tensor, ...],
    metas: tuple[torch.Tensor, ...],
    row_indices: torch.Tensor,
    out: torch.Tensor,
    config: str,
) -> tuple[int, int, int, int, int]:
    tensors = (X, *values, *metas, row_indices, out)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("indexed SwiGLU tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("indexed SwiGLU tensors must share a device")
    if X.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("indexed SwiGLU currently supports fp16 only")
    if any(value.dtype != torch.float16 for value in values):
        raise ValueError("indexed SwiGLU values must be fp16")
    if any(meta.dtype not in (torch.uint16, torch.int16) for meta in metas):
        raise ValueError("indexed SwiGLU metadata must be uint16/int16")
    if row_indices.dtype != torch.int32 or row_indices.ndim != 1:
        raise ValueError("row_indices must be a rank-1 int32 tensor")
    if X.ndim != 2 or out.ndim != 2:
        raise ValueError("X and out must be rank-2")
    if any(value.ndim != 2 for value in values):
        raise ValueError("indexed SwiGLU values must be rank-2")
    if any(meta.ndim != 1 for meta in metas):
        raise ValueError("indexed SwiGLU metadata must be rank-1")
    M, K = map(int, X.shape)
    logical_rows = int(row_indices.numel())
    N = int(values[0].shape[0])
    hidden_size = N // 2
    output_rows = int(out.shape[0])
    if M % 8 or K % 64 or N % 256:
        raise ValueError(
            "indexed SwiGLU requires M % 8, K % 64, and N % 256 == 0"
        )
    if logical_rows <= 0 or logical_rows > M:
        raise ValueError("logical row count must be in [1, X.shape[0]]")
    if tuple(out.shape) != (output_rows, hidden_size) or not out.is_contiguous():
        raise ValueError(
            f"out must be contiguous with second dimension {hidden_size}"
        )
    for value in values:
        if tuple(value.shape) != (N, K // 2) or not value.is_contiguous():
            raise ValueError(f"each values tensor must have shape {(N, K // 2)}")
    expected_meta = N * (K // 16)
    if any(meta.numel() != expected_meta or not meta.is_contiguous() for meta in metas):
        raise ValueError(f"each metadata tensor must have {expected_meta} elements")
    if not X.is_contiguous() or not row_indices.is_contiguous():
        raise ValueError("X and row_indices must be contiguous")
    configs = {
        "auto": 0,
        "256x32x64_s3_sw4": 1,
        "256x64x64_s3_sw4": 2,
    }
    if config not in configs:
        raise ValueError(f"unsupported indexed SwiGLU config {config!r}")
    return M, logical_rows, output_rows, K, configs[config]


def sparse24_cutlass_indexed_swiglu_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    config: str = "auto",
) -> torch.Tensor:
    """Apply sparse gate/up + SwiGLU and scatter compact rows into ``out``."""

    M, logical_rows, output_rows, K, config_id = (
        _validate_indexed_swiglu_tensors(
            X, (a_values,), (a_meta_e,), row_indices, out, config
        )
    )
    N = int(a_values.shape[0])
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_indexed_swiglu_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(logical_rows),
        ctypes.c_int(output_rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_id),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(f"sparse24 indexed SwiGLU failed with code {int(ret)}")
    return out


def sparse24_cutlass_dual_swiglu_prepacked(
    X: torch.Tensor,
    full_values: torch.Tensor,
    full_meta_e: torch.Tensor,
    residual_values: torch.Tensor,
    residual_meta_e: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    config: str = "auto",
) -> torch.Tensor:
    """Accumulate complementary sparse gate/up weights and scatter SwiGLU."""

    M, logical_rows, output_rows, K, config_id = (
        _validate_indexed_swiglu_tensors(
            X,
            (full_values, residual_values),
            (full_meta_e, residual_meta_e),
            row_indices,
            out,
            config,
        )
    )
    N = int(full_values.shape[0])
    if tuple(residual_values.shape) != tuple(full_values.shape):
        raise ValueError("full and residual values must have the same shape")
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_dual_swiglu_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(full_values.data_ptr()),
        ctypes.c_void_p(full_meta_e.data_ptr()),
        ctypes.c_void_p(residual_values.data_ptr()),
        ctypes.c_void_p(residual_meta_e.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(logical_rows),
        ctypes.c_int(output_rows),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_id),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(f"sparse24 dual SwiGLU failed with code {int(ret)}")
    return out


def sparse24_cutlass_residual_correction_swiglu_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    dense_base: torch.Tensor,
    dense_rows: torch.Tensor,
    out: torch.Tensor,
    *,
    dense_hidden: torch.Tensor | None = None,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse residual GEMM, correction, SwiGLU, and optional compact store."""

    tensors = (X, a_values, a_meta_e, dense_base, dense_rows, out)
    if dense_hidden is not None:
        tensors = (*tensors, dense_hidden)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("residual correction SwiGLU tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("residual correction SwiGLU tensors must share a device")
    if any(
        tensor.dtype != torch.float16
        for tensor in (X, a_values, dense_base, out)
    ):
        raise ValueError("residual correction SwiGLU currently supports fp16 only")
    if dense_hidden is not None and dense_hidden.dtype != torch.float16:
        raise ValueError("dense_hidden must have dtype torch.float16")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("a_meta_e must have dtype torch.uint16/int16")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 CUDA int32 tensor")
    if any(tensor.ndim != 2 for tensor in (X, a_values, dense_base, out)):
        raise ValueError("X, a_values, dense_base, and out must be rank-2")
    if a_meta_e.ndim != 1:
        raise ValueError("a_meta_e must be rank-1")

    M, K = map(int, X.shape)
    output_size = int(a_values.shape[0])
    hidden_size = output_size // 2
    dense_count = int(dense_rows.numel())
    output_rows = int(out.shape[0])
    if M % 8 or K % 64 or output_size % 256:
        raise ValueError(
            "residual correction SwiGLU requires M % 8, K % 64, "
            "and output_size % 256 == 0"
        )
    if dense_count <= 0 or dense_count > M:
        raise ValueError("dense row count must be in [1, X.shape[0]]")
    if tuple(dense_base.shape) != (dense_count, output_size):
        raise ValueError(
            f"dense_base must have shape {(dense_count, output_size)}"
        )
    if tuple(out.shape) != (output_rows, hidden_size):
        raise ValueError(f"out must have second dimension {hidden_size}")
    if dense_hidden is not None and tuple(dense_hidden.shape) != (
        M,
        hidden_size,
    ):
        raise ValueError(
            f"dense_hidden must have shape {(M, hidden_size)}, "
            f"got {tuple(dense_hidden.shape)}"
        )
    if tuple(a_values.shape) != (output_size, K // 2):
        raise ValueError(f"a_values must have shape {(output_size, K // 2)}")
    if a_meta_e.numel() != output_size * (K // 16):
        raise ValueError(f"a_meta_e must have {output_size * (K // 16)} elements")
    if not X.is_contiguous() or not a_values.is_contiguous():
        raise ValueError("X and a_values must be contiguous")
    if not a_meta_e.is_contiguous() or not dense_base.is_contiguous():
        raise ValueError("a_meta_e and dense_base must be contiguous")
    if not dense_rows.is_contiguous() or not out.is_contiguous():
        raise ValueError("dense_rows and out must be contiguous")
    if dense_hidden is not None and not dense_hidden.is_contiguous():
        raise ValueError("dense_hidden must be contiguous")
    configs = {
        "auto": 0,
        "256x32x64_s3_sw4": 1,
        "256x64x64_s3_sw4": 2,
    }
    if config not in configs:
        raise ValueError(f"unsupported residual correction config {config!r}")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_residual_correction_swiglu_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        (
            ctypes.c_void_p(dense_hidden.data_ptr())
            if dense_hidden is not None
            else None
        ),
        ctypes.c_int(M),
        ctypes.c_int(dense_count),
        ctypes.c_int(output_rows),
        ctypes.c_int(K),
        ctypes.c_int(output_size),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 residual correction SwiGLU failed with code "
            f"{int(ret)}"
        )
    return out


def sparse24_cutlass_residual_delta_swiglu_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    dense_base: torch.Tensor,
    dense_delta: torch.Tensor,
    *,
    config: str = "auto",
) -> torch.Tensor:
    """Fuse the residual Gate GEMM with compact SwiGLU delta output."""

    tensors = (X, a_values, a_meta_e, dense_base, dense_delta)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("residual delta SwiGLU tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("residual delta SwiGLU tensors must share a device")
    if any(
        tensor.dtype != torch.float16
        for tensor in (X, a_values, dense_base, dense_delta)
    ):
        raise ValueError("residual delta SwiGLU currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError("a_meta_e must have dtype torch.uint16/int16")
    if any(tensor.ndim != 2 for tensor in (X, a_values, dense_base, dense_delta)):
        raise ValueError("residual delta SwiGLU tensors must be rank-2")
    if a_meta_e.ndim != 1:
        raise ValueError("a_meta_e must be rank-1")

    M, K = map(int, X.shape)
    output_size = int(a_values.shape[0])
    hidden_size = output_size // 2
    dense_count = int(dense_base.shape[0])
    if M % 8 or K % 64 or output_size % 256:
        raise ValueError(
            "residual delta SwiGLU requires M % 8, K % 64, "
            "and output_size % 256 == 0"
        )
    if dense_count <= 0 or dense_count > M:
        raise ValueError("dense row count must be in [1, X.shape[0]]")
    if tuple(dense_base.shape) != (dense_count, output_size):
        raise ValueError(
            f"dense_base must have shape {(dense_count, output_size)}"
        )
    if tuple(dense_delta.shape) != (M, hidden_size):
        raise ValueError(
            f"dense_delta must have shape {(M, hidden_size)}, "
            f"got {tuple(dense_delta.shape)}"
        )
    if tuple(a_values.shape) != (output_size, K // 2):
        raise ValueError(f"a_values must have shape {(output_size, K // 2)}")
    if a_meta_e.numel() != output_size * (K // 16):
        raise ValueError(f"a_meta_e must have {output_size * (K // 16)} elements")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("residual delta SwiGLU tensors must be contiguous")
    configs = {
        "auto": 0,
        "256x32x64_s3_sw4": 1,
        "256x64x64_s3_sw4": 2,
    }
    if config not in configs:
        raise ValueError(f"unsupported residual delta config {config!r}")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_residual_delta_swiglu_gemm_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(a_values.data_ptr()),
        ctypes.c_void_p(a_meta_e.data_ptr()),
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(dense_delta.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(dense_count),
        ctypes.c_int(K),
        ctypes.c_int(output_size),
        ctypes.c_int(configs[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 residual delta SwiGLU failed with code {int(ret)}"
        )
    return dense_delta


def sparse24_routed_swiglu_correction_(
    dense_base: torch.Tensor,
    dense_residual: torch.Tensor,
    dense_rows: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Add dense-row residuals, apply SwiGLU, and scatter into ``out``."""

    tensors = (dense_base, dense_residual, dense_rows, out)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed SwiGLU correction tensors must be CUDA")
    if any(tensor.device != out.device for tensor in tensors[:-1]):
        raise ValueError("routed SwiGLU correction tensors must share a device")
    if (
        dense_base.dtype != torch.float16
        or dense_residual.dtype != torch.float16
        or out.dtype != torch.float16
    ):
        raise ValueError("routed SwiGLU correction currently supports fp16 only")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 CUDA int32 tensor")
    if dense_base.ndim != 2 or dense_residual.ndim != 2 or out.ndim != 2:
        raise ValueError("routed SwiGLU correction tensors must be rank-2")
    dense_count, output_size = map(int, dense_base.shape)
    output_rows, hidden_size = map(int, out.shape)
    if dense_count <= 0 or output_size != 2 * hidden_size:
        raise ValueError("dense gate/up width must be twice the output width")
    if tuple(dense_residual.shape) != (dense_count, output_size):
        raise ValueError("dense_residual must match dense_base")
    if tuple(dense_rows.shape) != (dense_count,):
        raise ValueError(f"dense_rows must have shape {(dense_count,)}")
    if hidden_size % 2 != 0:
        raise ValueError("routed SwiGLU correction requires even hidden size")
    if not dense_rows.is_contiguous():
        raise ValueError("dense_rows must be contiguous")
    contiguous_layout = (
        dense_base.is_contiguous()
        and dense_residual.is_contiguous()
        and out.is_contiguous()
    )
    transposed_layout = (
        dense_base.stride(0) == 1
        and dense_base.stride(1) >= dense_count
        and dense_residual.stride(0) == 1
        and dense_residual.stride(1) >= dense_count
        and out.stride(0) == 1
        and out.stride(1) >= output_rows
    )
    if not contiguous_layout and not transposed_layout:
        raise ValueError(
            "routed SwiGLU correction requires all-contiguous or "
            "all-transposed tensors"
        )

    lib = _load_library()
    stream = torch.cuda.current_stream(out.device).cuda_stream
    common_args = (
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(dense_residual.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(dense_count),
        ctypes.c_int(output_rows),
        ctypes.c_int(hidden_size),
    )
    if contiguous_layout:
        ret = lib.sparse24_cutlass_routed_swiglu_correction_f16_stream(
            *common_args,
            ctypes.c_void_p(stream),
        )
    else:
        ret = (
            lib.sparse24_cutlass_routed_swiglu_correction_transposed_f16_stream(
                *common_args,
                ctypes.c_int(dense_base.stride(1)),
                ctypes.c_int(dense_residual.stride(1)),
                ctypes.c_int(out.stride(1)),
                ctypes.c_void_p(stream),
            )
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed SwiGLU correction failed with code " f"{int(ret)}"
        )
    return out


def sparse24_routed_swiglu_correction_gather_(
    dense_base: torch.Tensor,
    dense_residual: torch.Tensor,
    dense_rows: torch.Tensor,
    out: torch.Tensor,
    dense_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct routed SwiGLU rows and materialize compact Down input."""

    tensors = (dense_base, dense_residual, dense_rows, out, dense_hidden)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed SwiGLU correction tensors must be CUDA")
    if any(tensor.device != out.device for tensor in tensors):
        raise ValueError("routed SwiGLU correction tensors must share a device")
    if any(
        tensor.dtype != torch.float16
        for tensor in (dense_base, dense_residual, out, dense_hidden)
    ):
        raise ValueError("routed SwiGLU correction currently supports fp16 only")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be a rank-1 CUDA int32 tensor")
    if any(
        tensor.ndim != 2
        for tensor in (dense_base, dense_residual, out, dense_hidden)
    ):
        raise ValueError("routed SwiGLU correction tensors must be rank-2")
    dense_count, output_size = map(int, dense_base.shape)
    output_rows, hidden_size = map(int, out.shape)
    dense_run, compact_hidden = map(int, dense_hidden.shape)
    if dense_count <= 0 or output_size != 2 * hidden_size:
        raise ValueError("dense gate/up width must be twice the output width")
    if tuple(dense_residual.shape) != (dense_count, output_size):
        raise ValueError("dense_residual must match dense_base")
    if tuple(dense_rows.shape) != (dense_count,):
        raise ValueError(f"dense_rows must have shape {(dense_count,)}")
    if dense_run < dense_count or compact_hidden != hidden_size:
        raise ValueError(
            "dense_hidden must have shape [dense_run >= dense_count, hidden_size]"
        )
    if hidden_size % 2 != 0:
        raise ValueError("routed SwiGLU correction requires even hidden size")
    if not all(
        tensor.is_contiguous()
        for tensor in (dense_base, dense_residual, dense_rows, out, dense_hidden)
    ):
        raise ValueError("fused correction/gather requires contiguous tensors")

    lib = _load_library()
    stream = torch.cuda.current_stream(out.device).cuda_stream
    ret = lib.sparse24_cutlass_routed_swiglu_correction_gather_f16_stream(
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(dense_residual.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(dense_hidden.data_ptr()),
        ctypes.c_int(dense_count),
        ctypes.c_int(dense_run),
        ctypes.c_int(output_rows),
        ctypes.c_int(hidden_size),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed SwiGLU correction/gather failed with code "
            f"{int(ret)}"
        )
    return out, dense_hidden


def sparse24_routed_swiglu_delta_(
    dense_base: torch.Tensor,
    dense_residual: torch.Tensor,
    dense_delta: torch.Tensor,
) -> torch.Tensor:
    """Materialize compact ``exact SwiGLU - W24 SwiGLU`` rows."""

    tensors = (dense_base, dense_residual, dense_delta)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("routed SwiGLU delta tensors must be CUDA")
    if any(tensor.device != dense_base.device for tensor in tensors[1:]):
        raise ValueError("routed SwiGLU delta tensors must share a device")
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("routed SwiGLU delta currently supports fp16 only")
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("routed SwiGLU delta tensors must be rank-2")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("routed SwiGLU delta tensors must be contiguous")
    dense_count, output_size = map(int, dense_base.shape)
    dense_run, residual_size = map(int, dense_residual.shape)
    delta_run, hidden_size = map(int, dense_delta.shape)
    if dense_count <= 0 or output_size != 2 * hidden_size:
        raise ValueError("dense gate/up width must be twice the delta width")
    if dense_run < dense_count or residual_size != output_size:
        raise ValueError(
            "dense_residual must cover dense_count rows at the gate/up width"
        )
    if delta_run != dense_run or hidden_size % 2:
        raise ValueError(
            "dense_delta must match residual rows with an even hidden width"
        )

    lib = _load_library()
    stream = torch.cuda.current_stream(dense_base.device).cuda_stream
    ret = lib.sparse24_cutlass_routed_swiglu_delta_f16_stream(
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(dense_residual.data_ptr()),
        ctypes.c_void_p(dense_delta.data_ptr()),
        ctypes.c_int(dense_count),
        ctypes.c_int(dense_run),
        ctypes.c_int(hidden_size),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 routed SwiGLU delta failed with code {int(ret)}"
        )
    return dense_delta


def sparse24_routed_swiglu_correction_transpose(
    sparse_hidden: torch.Tensor,
    dense_base: torch.Tensor,
    dense_residual: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse dense-row SwiGLU correction with contiguous-to-strided transpose."""

    tensors = (
        sparse_hidden,
        dense_base,
        dense_residual,
        dense_slot_by_row,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed SwiGLU transpose tensors must be CUDA")
    if any(tensor.device != sparse_hidden.device for tensor in tensors[1:]):
        raise ValueError("routed SwiGLU transpose tensors must share a device")
    if (
        sparse_hidden.dtype != torch.float16
        or dense_base.dtype != torch.float16
        or dense_residual.dtype != torch.float16
    ):
        raise ValueError("routed SwiGLU transpose currently supports fp16 only")
    if (
        sparse_hidden.ndim != 2
        or dense_base.ndim != 2
        or dense_residual.ndim != 2
    ):
        raise ValueError("routed SwiGLU transpose tensors must be rank-2")
    output_rows, hidden_size = map(int, sparse_hidden.shape)
    dense_count, gate_up_size = map(int, dense_base.shape)
    if output_rows <= 0 or hidden_size <= 0 or dense_count <= 0:
        raise ValueError("routed SwiGLU transpose shapes must be non-empty")
    if gate_up_size != 2 * hidden_size:
        raise ValueError("dense gate/up width must be twice the hidden width")
    if tuple(dense_residual.shape) != (dense_count, gate_up_size):
        raise ValueError("dense_residual must match dense_base")
    if (
        dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (output_rows,)
        or not dense_slot_by_row.is_contiguous()
    ):
        raise ValueError(
            "dense_slot_by_row must be contiguous int32 with one slot per row"
        )
    if (
        not sparse_hidden.is_contiguous()
        or not dense_base.is_contiguous()
        or not dense_residual.is_contiguous()
    ):
        raise ValueError("routed SwiGLU transpose inputs must be contiguous")
    if out is None:
        output = torch.empty_strided(
            (output_rows, hidden_size),
            (1, output_rows),
            device=sparse_hidden.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != sparse_hidden.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, hidden_size)
            or tuple(output.stride()) != (1, output_rows)
        ):
            raise ValueError(
                "out must be CUDA fp16 with shape/stride "
                f"{(output_rows, hidden_size)}/{(1, output_rows)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(sparse_hidden.device).cuda_stream
    ret = (
        lib.sparse24_cutlass_routed_swiglu_correction_transpose_tiled_f16_stream(
            ctypes.c_void_p(sparse_hidden.data_ptr()),
            ctypes.c_void_p(dense_base.data_ptr()),
            ctypes.c_void_p(dense_residual.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(output_rows),
            ctypes.c_int(dense_count),
            ctypes.c_int(hidden_size),
            ctypes.c_void_p(stream),
        )
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed SwiGLU correction/transpose failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_routed_linear_correction_(
    dense_base: torch.Tensor,
    dense_residual: torch.Tensor,
    dense_rows: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Add compact dense residuals and scatter corrected rows into ``out``."""

    tensors = (dense_base, dense_residual, dense_rows, out)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed linear correction tensors must be CUDA")
    if any(tensor.device != out.device for tensor in tensors[:-1]):
        raise ValueError("routed linear correction tensors must share a device")
    if (
        dense_base.dtype != torch.float16
        or dense_residual.dtype != torch.float16
        or out.dtype != torch.float16
    ):
        raise ValueError("routed linear correction currently supports fp16 only")
    if dense_rows.dtype != torch.int32 or dense_rows.ndim != 1:
        raise ValueError("dense_rows must be rank-1 CUDA int32")
    if dense_base.ndim != 2 or dense_residual.ndim != 2 or out.ndim != 2:
        raise ValueError("routed linear correction tensors must be rank-2")
    dense_count, output_columns = map(int, dense_base.shape)
    output_rows = int(out.shape[0])
    if dense_count <= 0 or output_columns <= 0 or output_columns % 2:
        raise ValueError("dense correction shape must be non-empty with even width")
    if tuple(dense_residual.shape) != (dense_count, output_columns):
        raise ValueError("dense_residual must match dense_base")
    if tuple(dense_rows.shape) != (dense_count,):
        raise ValueError(f"dense_rows must have shape {(dense_count,)}")
    if tuple(out.shape) != (output_rows, output_columns) or not out.is_contiguous():
        raise ValueError("out must be contiguous and match the correction width")
    if not dense_base.is_contiguous() or not dense_residual.is_contiguous():
        raise ValueError("dense correction inputs must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(out.device).cuda_stream
    ret = lib.sparse24_cutlass_routed_linear_correction_f16_stream(
        ctypes.c_void_p(dense_base.data_ptr()),
        ctypes.c_void_p(dense_residual.data_ptr()),
        ctypes.c_void_p(dense_rows.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(dense_count),
        ctypes.c_int(output_rows),
        ctypes.c_int(output_columns),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed linear correction failed with code " f"{int(ret)}"
        )
    return out


def sparse24_cutlass_pair_add_indexed_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    output_rows: int,
    out: torch.Tensor | None = None,
    config: str | None = "auto",
) -> torch.Tensor:
    """Sum paired sparse projections and scatter their logical rows."""

    tensors = (X, a_values, a_meta_e, row_indices)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all pair-add epilogue tensors must be CUDA")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("all pair-add epilogue tensors must share a device")
    if X.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("pair-add sparse epilogue currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(
            f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}"
        )
    if row_indices.dtype != torch.int32 or row_indices.ndim != 1:
        raise ValueError("row_indices must be a rank-1 CUDA int32 tensor")
    if X.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X rank-2, a_values rank-2, and a_meta_e rank-1")
    M, K = (int(value) for value in X.shape)
    logical_rows = int(row_indices.numel())
    packed_n = int(a_values.shape[0])
    N = packed_n // 2
    output_rows = int(output_rows)
    if (
        logical_rows <= 0
        or logical_rows > M
        or output_rows <= 0
        or packed_n % 256 != 0
    ):
        raise ValueError(
            "pair-add sparse epilogue requires valid logical/output rows and "
            "a packed output size divisible by 256"
        )
    if M % 8 != 0 or K % 64 != 0 or N % 128 != 0:
        raise ValueError(
            "pair-add sparse epilogue requires M % 8, K % 64, N % 128 == 0"
        )
    if tuple(a_values.shape) != (packed_n, K // 2):
        raise ValueError(
            f"a_values must have shape {(packed_n, K // 2)}, "
            f"got {tuple(a_values.shape)}"
        )
    if a_meta_e.numel() != packed_n * (K // 16):
        raise ValueError(
            f"a_meta_e must have {packed_n * (K // 16)} elements, "
            f"got {a_meta_e.numel()}"
        )
    allowed_configs = {
        None,
        "auto",
        "256x32x64_s3_sw4",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported pair-add sparse epilogue config {config!r}")

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    row_indices = row_indices.contiguous()
    if out is None:
        output = torch.empty(
            (output_rows, N), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (output_rows, N)
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous CUDA fp16 with shape {(output_rows, N)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_pair_add_epilogue_config(config):
        ret = lib.sparse24_cutlass_inline_pair_add_gemm_f16_stream(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(row_indices.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(logical_rows),
            ctypes.c_int(output_rows),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 inline pair-add GEMM failed with code " f"{int(ret)}"
        )
    return output


def sparse24_cutlass_qkv_postop_prepacked(
    X: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int = 128,
    rotary_dim: int = 128,
    epsilon: float = 1e-6,
    is_neox: bool = True,
    normalize_qk: bool = True,
    out: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    """Run sparse QKV GEMM with inline Q/K norm, RoPE, and materialization."""

    if not X.is_cuda or X.dtype != torch.float16 or X.ndim != 2:
        raise ValueError("X must be a rank-2 CUDA fp16 tensor")
    if head_dim != 128 or q_size % head_dim or kv_size % head_dim:
        raise ValueError("inline QKV epilogue requires 128-dimensional heads")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even value <= head_dim")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    M, K = map(int, X.shape)
    output_size = q_size + 2 * kv_size
    if M % 8 or K % 64 or output_size % 128:
        raise ValueError(
            "inline QKV epilogue requires M % 8, K % 64, output_size % 128 == 0"
        )
    if (
        not a_values.is_cuda
        or a_values.device != X.device
        or a_values.dtype != torch.float16
        or tuple(a_values.shape) != (output_size, K // 2)
    ):
        raise ValueError(
            "a_values must be CUDA fp16 with shape "
            f"{(output_size, K // 2)}"
        )
    if (
        not a_meta_e.is_cuda
        or a_meta_e.device != X.device
        or a_meta_e.dtype not in (torch.uint16, torch.int16)
        or a_meta_e.ndim != 1
        or a_meta_e.numel() != output_size * (K // 16)
    ):
        raise ValueError("a_meta_e has an invalid device, dtype, shape, or size")
    for label, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if (
            not weight.is_cuda
            or weight.device != X.device
            or weight.dtype != torch.float16
            or tuple(weight.shape) != (head_dim,)
            or not weight.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 shape {(head_dim,)}"
            )
    if (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != X.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[1]) != rotary_dim
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError(
            "cos_sin_cache must be contiguous CUDA fp16 with second dimension "
            f"{rotary_dim}"
        )
    if (
        not position_ids.is_cuda
        or position_ids.device != X.device
        or position_ids.dtype != torch.int64
        or position_ids.ndim != 1
        or position_ids.numel() < M
        or not position_ids.is_contiguous()
    ):
        raise ValueError("position_ids must be contiguous CUDA int64 with length >= M")
    allowed_configs = {
        None,
        "auto",
        "128x32x64_s4",
        "128x32x64_s4_sw2",
        "128x32x64_s4_sw4",
        "128x64x64_s5",
        "256x32x64_s3_sw4",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    if config not in allowed_configs:
        raise ValueError(f"unsupported inline sparse QKV config {config!r}")

    if out is None:
        output = torch.empty(
            (M, output_size), device=X.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != X.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, output_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(M, output_size)}"
            )

    Xc = X.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_inline_qkv_epilogue_config(config):
        ret = lib.sparse24_cutlass_inline_qkv_postop_gemm_f16_stream(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(q_weight.data_ptr()),
            ctypes.c_void_p(k_weight.data_ptr()),
            ctypes.c_void_p(cos_sin_cache.data_ptr()),
            ctypes.c_void_p(position_ids.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(q_size),
            ctypes.c_int(kv_size),
            ctypes.c_int(head_dim),
            ctypes.c_int(rotary_dim),
            ctypes.c_float(epsilon),
            ctypes.c_int(int(is_neox)),
            ctypes.c_int(int(normalize_qk)),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 inline QKV post-op GEMM failed with code {int(ret)}"
        )
    return output


def sparse24_silu_and_mul_transposed(
    gate_up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SwiGLU while preserving CUTLASS's transposed output layout."""

    if not gate_up.is_cuda or gate_up.dtype != torch.float16:
        raise ValueError("gate_up must be a CUDA fp16 tensor")
    if gate_up.ndim != 2 or int(gate_up.shape[1]) % 2:
        raise ValueError(
            f"gate_up must have shape [M, 2H], got {tuple(gate_up.shape)}"
        )
    rows = int(gate_up.shape[0])
    hidden_size = int(gate_up.shape[1]) // 2
    leading_dim = int(gate_up.stride(1))
    if int(gate_up.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "gate_up must use transposed CUTLASS layout with stride "
            f"(1, >=M), got {tuple(gate_up.stride())}"
        )
    if out is None:
        output = torch.empty_strided(
            (rows, hidden_size),
            (1, leading_dim),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != gate_up.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, hidden_size)
            or tuple(output.stride()) != (1, leading_dim)
        ):
            raise ValueError(
                "out must be CUDA fp16 with shape/stride "
                f"{(rows, hidden_size)}/{(1, leading_dim)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(gate_up.device).cuda_stream
    ret = lib.sparse24_cutlass_silu_and_mul_transposed_f16_stream(
        ctypes.c_void_p(gate_up.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(hidden_size),
        ctypes.c_int(leading_dim),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_silu_and_mul_transposed failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_silu_and_mul_transposed_to_contiguous(
    gate_up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse SwiGLU and materialization of a CUTLASS-layout gate-up."""

    if not gate_up.is_cuda or gate_up.dtype != torch.float16:
        raise ValueError("gate_up must be a CUDA fp16 tensor")
    if gate_up.ndim != 2 or int(gate_up.shape[1]) % 2:
        raise ValueError(
            f"gate_up must have shape [M, 2H], got {tuple(gate_up.shape)}"
        )
    rows = int(gate_up.shape[0])
    hidden_size = int(gate_up.shape[1]) // 2
    leading_dim = int(gate_up.stride(1))
    if int(gate_up.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "gate_up must use transposed CUTLASS layout with stride "
            f"(1, >=M), got {tuple(gate_up.stride())}"
        )
    if out is None:
        output = torch.empty(
            (rows, hidden_size),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != gate_up.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, hidden_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(rows, hidden_size)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(gate_up.device).cuda_stream
    ret = (
        lib.sparse24_cutlass_silu_and_mul_transposed_to_contiguous_f16_stream(
            ctypes.c_void_p(gate_up.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(hidden_size),
            ctypes.c_int(leading_dim),
            ctypes.c_void_p(stream),
        )
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused transposed-to-contiguous SwiGLU failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_transpose_output_contiguous(
    transposed: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize a CUTLASS transposed view as contiguous ``[M, N]``."""

    if not transposed.is_cuda or transposed.dtype != torch.float16:
        raise ValueError("transposed output must be a CUDA fp16 tensor")
    if transposed.ndim != 2:
        raise ValueError("transposed output must be rank-2")
    M, N = map(int, transposed.shape)
    if tuple(transposed.stride()) != (1, M):
        raise ValueError(
            f"transposed output must have stride {(1, M)}, "
            f"got {tuple(transposed.stride())}"
        )
    if out is None:
        output = torch.empty((M, N), device=transposed.device, dtype=torch.float16)
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != transposed.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or not output.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous CUDA fp16 shape {(M, N)}")

    lib = _load_library()
    stream = torch.cuda.current_stream(transposed.device).cuda_stream
    ret = lib.sparse24_cutlass_transpose_output_f16_stream(
        ctypes.c_void_p(transposed.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24_cutlass_transpose_output_f16 failed with code {int(ret)}"
        )
    return output


def sparse24_transpose_add_routed_residual(
    full_transposed: torch.Tensor,
    residual_transposed: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    dense_count: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transpose full output and add compact residual rows in one tiled pass."""

    tensors = (full_transposed, residual_transposed, dense_slot_by_row)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed transpose-add tensors must be CUDA")
    if any(tensor.device != full_transposed.device for tensor in tensors[1:]):
        raise ValueError("routed transpose-add tensors must share a device")
    if (
        full_transposed.dtype != torch.float16
        or residual_transposed.dtype != torch.float16
    ):
        raise ValueError("routed transpose-add currently supports fp16 only")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if full_transposed.ndim != 2 or residual_transposed.ndim != 2:
        raise ValueError("routed transpose-add inputs must be rank-2")
    M, N = map(int, full_transposed.shape)
    residual_rows, residual_columns = map(int, residual_transposed.shape)
    dense_count = int(dense_count)
    if residual_columns != N or dense_count <= 0 or dense_count > residual_rows:
        raise ValueError(
            "residual width must match full output and dense_count must fit rows"
        )
    if tuple(dense_slot_by_row.shape) != (M,):
        raise ValueError(f"dense_slot_by_row must have shape {(M,)}")
    if int(full_transposed.stride(0)) != 1 or int(
        full_transposed.stride(1)
    ) < M:
        raise ValueError("full output must have stride (1, ld) with ld >= M")
    if int(residual_transposed.stride(0)) != 1 or int(
        residual_transposed.stride(1)
    ) < dense_count:
        raise ValueError(
            "residual output must have stride (1, ld) with ld >= dense_count"
        )
    if out is None:
        output = torch.empty(
            (M, N), device=full_transposed.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != full_transposed.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or not output.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous CUDA fp16 shape {(M, N)}")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_transposed.device).cuda_stream
    ret = lib.sparse24_cutlass_transpose_add_routed_residual_f16_stream(
        ctypes.c_void_p(full_transposed.data_ptr()),
        ctypes.c_void_p(residual_transposed.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(full_transposed.stride(1)),
        ctypes.c_int(residual_transposed.stride(1)),
        ctypes.c_int(dense_count),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed transpose-add failed with code " f"{int(ret)}"
        )
    return output


def sparse24_transpose_add_routed_residual_to_residual_(
    full_transposed: torch.Tensor,
    routed_residual_transposed: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    model_residual: torch.Tensor,
    *,
    dense_count: int,
) -> torch.Tensor:
    """Merge routed Down output directly into the model residual in-place."""

    tensors = (
        full_transposed,
        routed_residual_transposed,
        dense_slot_by_row,
        model_residual,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed residual epilogue tensors must be CUDA")
    if any(tensor.device != full_transposed.device for tensor in tensors[1:]):
        raise ValueError("routed residual epilogue tensors must share a device")
    if (
        full_transposed.dtype != torch.float16
        or routed_residual_transposed.dtype != torch.float16
        or model_residual.dtype != torch.float16
    ):
        raise ValueError("routed residual epilogue currently supports fp16 only")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if (
        full_transposed.ndim != 2
        or routed_residual_transposed.ndim != 2
        or model_residual.ndim != 2
    ):
        raise ValueError("routed residual epilogue tensors must be rank-2")
    M, N = map(int, full_transposed.shape)
    routed_rows, routed_columns = map(
        int, routed_residual_transposed.shape
    )
    dense_count = int(dense_count)
    if routed_columns != N or dense_count <= 0 or dense_count > routed_rows:
        raise ValueError(
            "routed residual width must match and dense_count must fit rows"
        )
    if tuple(dense_slot_by_row.shape) != (M,):
        raise ValueError(f"dense_slot_by_row must have shape {(M,)}")
    if tuple(model_residual.shape) != (M, N) or not model_residual.is_contiguous():
        raise ValueError(
            f"model_residual must be contiguous CUDA fp16 shape {(M, N)}"
        )
    if int(full_transposed.stride(0)) != 1 or int(
        full_transposed.stride(1)
    ) < M:
        raise ValueError("full output must have stride (1, ld) with ld >= M")
    if int(routed_residual_transposed.stride(0)) != 1 or int(
        routed_residual_transposed.stride(1)
    ) < dense_count:
        raise ValueError(
            "routed residual output must have stride (1, ld) with ld >= dense_count"
        )

    lib = _load_library()
    stream = torch.cuda.current_stream(full_transposed.device).cuda_stream
    ret = (
        lib.sparse24_cutlass_transpose_add_routed_residual_to_residual_f16_stream(
            ctypes.c_void_p(full_transposed.data_ptr()),
            ctypes.c_void_p(routed_residual_transposed.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_void_p(model_residual.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(full_transposed.stride(1)),
            ctypes.c_int(routed_residual_transposed.stride(1)),
            ctypes.c_int(dense_count),
            ctypes.c_void_p(stream),
        )
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed residual epilogue failed with code " f"{int(ret)}"
        )
    return model_residual


def sparse24_transpose_add_routed_residual_rmsnorm(
    full_transposed: torch.Tensor,
    routed_residual_transposed: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    model_residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    dense_count: int,
    epsilon: float,
    out: torch.Tensor | None = None,
    square_partials: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge routed Down into residual and normalize with tiled partials."""

    tensors = (
        full_transposed,
        routed_residual_transposed,
        dense_slot_by_row,
        model_residual,
        weight,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("routed residual RMSNorm tensors must be CUDA")
    if any(tensor.device != full_transposed.device for tensor in tensors[1:]):
        raise ValueError("routed residual RMSNorm tensors must share a device")
    if (
        full_transposed.dtype != torch.float16
        or routed_residual_transposed.dtype != torch.float16
        or model_residual.dtype != torch.float16
        or weight.dtype != torch.float16
    ):
        raise ValueError("routed residual RMSNorm currently supports fp16 only")
    if dense_slot_by_row.dtype != torch.int32 or dense_slot_by_row.ndim != 1:
        raise ValueError("dense_slot_by_row must be rank-1 CUDA int32")
    if (
        full_transposed.ndim != 2
        or routed_residual_transposed.ndim != 2
        or model_residual.ndim != 2
        or weight.ndim != 1
    ):
        raise ValueError("routed residual RMSNorm tensor ranks are invalid")
    M, N = map(int, full_transposed.shape)
    routed_rows, routed_columns = map(
        int, routed_residual_transposed.shape
    )
    dense_count = int(dense_count)
    if N % 32 or tuple(weight.shape) != (N,) or not weight.is_contiguous():
        raise ValueError("hidden size must be divisible by 32 with contiguous weight")
    if routed_columns != N or dense_count <= 0 or dense_count > routed_rows:
        raise ValueError(
            "routed residual width must match and dense_count must fit rows"
        )
    if tuple(dense_slot_by_row.shape) != (M,):
        raise ValueError(f"dense_slot_by_row must have shape {(M,)}")
    if tuple(model_residual.shape) != (M, N) or not model_residual.is_contiguous():
        raise ValueError(f"model_residual must be contiguous shape {(M, N)}")
    if int(full_transposed.stride(0)) != 1 or int(
        full_transposed.stride(1)
    ) < M:
        raise ValueError("full output must have stride (1, ld) with ld >= M")
    if int(routed_residual_transposed.stride(0)) != 1 or int(
        routed_residual_transposed.stride(1)
    ) < dense_count:
        raise ValueError(
            "routed residual output must have stride (1, ld) with ld >= dense_count"
        )

    normalized = out
    if normalized is None:
        normalized = torch.empty_like(model_residual)
    elif (
        not normalized.is_cuda
        or normalized.device != full_transposed.device
        or normalized.dtype != torch.float16
        or tuple(normalized.shape) != (M, N)
        or not normalized.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous CUDA fp16 shape {(M, N)}")
    partials = square_partials
    partial_shape = (M, N // 32)
    if partials is None:
        partials = torch.empty(
            partial_shape,
            device=full_transposed.device,
            dtype=torch.float32,
        )
    elif (
        not partials.is_cuda
        or partials.device != full_transposed.device
        or partials.dtype != torch.float32
        or tuple(partials.shape) != partial_shape
        or not partials.is_contiguous()
    ):
        raise ValueError(
            f"square_partials must be contiguous CUDA fp32 shape {partial_shape}"
        )

    lib = _load_library()
    stream = torch.cuda.current_stream(full_transposed.device).cuda_stream
    ret = lib.sparse24_cutlass_transpose_add_routed_residual_rmsnorm_f16_stream(
        ctypes.c_void_p(full_transposed.data_ptr()),
        ctypes.c_void_p(routed_residual_transposed.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(model_residual.data_ptr()),
        ctypes.c_void_p(normalized.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(partials.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(full_transposed.stride(1)),
        ctypes.c_int(routed_residual_transposed.stride(1)),
        ctypes.c_int(dense_count),
        ctypes.c_float(float(epsilon)),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 routed residual RMSNorm failed with code " f"{int(ret)}"
        )
    return normalized


def sparse24_transpose_add_routed_splitk_residual(
    full_transposed: torch.Tensor,
    residual_partials: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    *,
    dense_count: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce split-K residual partials inside the routed transpose-add."""

    tensors = (full_transposed, residual_partials, dense_slot_by_row)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("split-K routed transpose-add tensors must be CUDA")
    if any(tensor.device != full_transposed.device for tensor in tensors[1:]):
        raise ValueError("split-K routed transpose-add tensors must share a device")
    if (
        full_transposed.dtype != torch.float16
        or residual_partials.dtype != torch.float16
        or dense_slot_by_row.dtype != torch.int32
    ):
        raise ValueError("split-K routed transpose-add requires fp16 outputs and int32 slots")
    if full_transposed.ndim != 2 or residual_partials.ndim != 3:
        raise ValueError("expected rank-2 full output and rank-3 residual partials")
    M, N = map(int, full_transposed.shape)
    split_k_slices, residual_rows, residual_columns = map(
        int, residual_partials.shape
    )
    dense_count = int(dense_count)
    if split_k_slices < 2 or split_k_slices > 8:
        raise ValueError("split-K routed transpose-add supports 2 to 8 slices")
    if residual_columns != N or dense_count <= 0 or dense_count > residual_rows:
        raise ValueError("residual partial shape or dense_count is invalid")
    if tuple(dense_slot_by_row.shape) != (M,) or not dense_slot_by_row.is_contiguous():
        raise ValueError(f"dense_slot_by_row must be contiguous shape {(M,)}")
    full_ld = int(full_transposed.stride(1))
    residual_ld = int(residual_partials.stride(2))
    expected_partial_stride = residual_ld * N
    if int(full_transposed.stride(0)) != 1 or full_ld < M:
        raise ValueError("full output must have stride (1, ld) with ld >= M")
    if (
        int(residual_partials.stride(1)) != 1
        or residual_ld < dense_count
        or int(residual_partials.stride(0)) != expected_partial_stride
    ):
        raise ValueError(
            "residual partials must have strides (N*ld, 1, ld) with ld >= dense_count"
        )
    if out is None:
        output = torch.empty((M, N), device=full_transposed.device, dtype=torch.float16)
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != full_transposed.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or not output.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous CUDA fp16 shape {(M, N)}")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_transposed.device).cuda_stream
    ret = lib.sparse24_cutlass_transpose_add_routed_splitk_residual_f16_stream(
        ctypes.c_void_p(full_transposed.data_ptr()),
        ctypes.c_void_p(residual_partials.data_ptr()),
        ctypes.c_void_p(dense_slot_by_row.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(full_ld),
        ctypes.c_int(residual_ld),
        ctypes.c_int(dense_count),
        ctypes.c_int(split_k_slices),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 split-K routed transpose-add failed with code " f"{int(ret)}"
        )
    return output


def sparse24_transpose_input_to_strided(
    contiguous: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transpose contiguous storage into a logical ``[M, N]`` strided view."""

    if (
        not contiguous.is_cuda
        or contiguous.dtype != torch.float16
        or contiguous.ndim != 2
        or not contiguous.is_contiguous()
    ):
        raise ValueError("input must be a contiguous rank-2 CUDA fp16 tensor")
    M, N = map(int, contiguous.shape)
    if out is None:
        output = torch.empty_strided(
            (M, N), (1, M), device=contiguous.device, dtype=torch.float16
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != contiguous.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (M, N)
            or tuple(output.stride()) != (1, M)
        ):
            raise ValueError(
                f"out must be CUDA fp16 shape/stride {(M, N)}/{(1, M)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(contiguous.device).cuda_stream
    ret = lib.sparse24_cutlass_transpose_output_f16_stream(
        ctypes.c_void_p(contiguous.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(N),
        ctypes.c_int(M),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 contiguous-to-strided transpose failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_transpose_add_residual_(
    transposed: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    """Fuse CUTLASS-output materialization into an in-place residual add."""

    if not transposed.is_cuda or transposed.dtype != torch.float16:
        raise ValueError("transposed must be a CUDA fp16 tensor")
    if transposed.ndim != 2:
        raise ValueError("transposed must be rank-2")
    rows, columns = map(int, transposed.shape)
    leading_dim = int(transposed.stride(1))
    if int(transposed.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "transposed must use CUTLASS layout with stride "
            f"(1, >=M), got {tuple(transposed.stride())}"
        )
    if (
        not residual.is_cuda
        or residual.device != transposed.device
        or residual.dtype != torch.float16
        or tuple(residual.shape) != (rows, columns)
        or not residual.is_contiguous()
    ):
        raise ValueError(
            "residual must be contiguous CUDA fp16 with shape "
            f"{(rows, columns)}"
        )

    lib = _load_library()
    stream = torch.cuda.current_stream(transposed.device).cuda_stream
    ret = lib.sparse24_cutlass_transpose_add_residual_f16_stream(
        ctypes.c_void_p(transposed.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(columns),
        ctypes.c_int(leading_dim),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 fused transpose/residual add failed with code {int(ret)}"
        )


def sparse24_transpose_add_rmsnorm(
    transposed: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float,
    out: torch.Tensor | None = None,
    epilogue_config: str | None = None,
) -> torch.Tensor:
    """Fuse sparse-output materialization, residual add, and RMSNorm."""

    if not transposed.is_cuda or transposed.dtype != torch.float16:
        raise ValueError("transposed must be a CUDA fp16 tensor")
    if transposed.ndim != 2:
        raise ValueError("transposed must be rank-2")
    rows, columns = map(int, transposed.shape)
    leading_dim = int(transposed.stride(1))
    if int(transposed.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "transposed must use CUTLASS layout with stride "
            f"(1, >=M), got {tuple(transposed.stride())}"
        )
    if columns % 32:
        raise ValueError(f"columns must be divisible by 32, got {columns}")
    if (
        not residual.is_cuda
        or residual.device != transposed.device
        or residual.dtype != torch.float16
        or tuple(residual.shape) != (rows, columns)
        or not residual.is_contiguous()
    ):
        raise ValueError(
            "residual must be contiguous CUDA fp16 with shape "
            f"{(rows, columns)}"
        )
    if (
        not weight.is_cuda
        or weight.device != transposed.device
        or weight.dtype != torch.float16
        or tuple(weight.shape) != (columns,)
        or not weight.is_contiguous()
    ):
        raise ValueError(
            f"weight must be contiguous CUDA fp16 with shape {(columns,)}"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if out is None:
        normalized = torch.empty_like(residual)
    else:
        normalized = out
        if (
            not normalized.is_cuda
            or normalized.device != transposed.device
            or normalized.dtype != torch.float16
            or tuple(normalized.shape) != (rows, columns)
            or not normalized.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(rows, columns)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(transposed.device).cuda_stream
    with _temporary_mlp_epilogue_config(epilogue_config):
        ret = lib.sparse24_cutlass_transpose_add_rmsnorm_f16_stream(
            ctypes.c_void_p(transposed.data_ptr()),
            ctypes.c_void_p(residual.data_ptr()),
            ctypes.c_void_p(normalized.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(columns),
            ctypes.c_int(leading_dim),
            ctypes.c_float(epsilon),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 fused transpose/add/RMSNorm failed with code {int(ret)}"
        )
    return normalized


def _validate_qkv_rmsnorm_inputs(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
) -> tuple[int, int]:
    if not qkv.is_cuda or qkv.dtype != torch.float16 or qkv.ndim != 2:
        raise ValueError("qkv must be a rank-2 CUDA fp16 tensor")
    if head_dim != 128:
        raise ValueError(f"fused QKV RMSNorm requires head_dim=128, got {head_dim}")
    if q_size <= 0 or kv_size <= 0:
        raise ValueError("q_size and kv_size must be positive")
    if q_size % head_dim or kv_size % head_dim:
        raise ValueError("q_size and kv_size must be divisible by head_dim")
    rows, output_size = map(int, qkv.shape)
    expected_output_size = q_size + 2 * kv_size
    if output_size != expected_output_size:
        raise ValueError(
            f"qkv must have shape [M, {expected_output_size}], "
            f"got {tuple(qkv.shape)}"
        )
    for label, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if (
            not weight.is_cuda
            or weight.device != qkv.device
            or weight.dtype != torch.float16
            or tuple(weight.shape) != (head_dim,)
            or not weight.is_contiguous()
        ):
            raise ValueError(
                f"{label} must be contiguous CUDA fp16 shape {(head_dim,)}"
            )
    return rows, output_size


def sparse24_qkv_transpose_rmsnorm(
    qkv_transposed: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse CUTLASS-output materialization with per-head Q/K RMSNorm."""

    rows, output_size = _validate_qkv_rmsnorm_inputs(
        qkv_transposed,
        q_weight,
        k_weight,
        q_size=q_size,
        kv_size=kv_size,
        head_dim=head_dim,
    )
    leading_dim = int(qkv_transposed.stride(1))
    if int(qkv_transposed.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "qkv_transposed must use CUTLASS layout with stride "
            f"(1, >=M), got {tuple(qkv_transposed.stride())}"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if out is None:
        output = torch.empty(
            (rows, output_size),
            device=qkv_transposed.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != qkv_transposed.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, output_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(rows, output_size)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(qkv_transposed.device).cuda_stream
    ret = lib.sparse24_cutlass_qkv_transpose_rmsnorm_f16_stream(
        ctypes.c_void_p(qkv_transposed.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(q_weight.data_ptr()),
        ctypes.c_void_p(k_weight.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(q_size),
        ctypes.c_int(kv_size),
        ctypes.c_int(head_dim),
        ctypes.c_int(leading_dim),
        ctypes.c_float(epsilon),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused QKV transpose/RMSNorm failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_qkv_transpose_postop(
    qkv_transposed: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    postop_config: str | None = None,
) -> torch.Tensor:
    """Fuse sparse-output materialization, optional Q/K norm, and RoPE."""

    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if normalize_qk:
        rows, output_size = _validate_qkv_rmsnorm_inputs(
            qkv_transposed,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
        )
    else:
        if (
            not qkv_transposed.is_cuda
            or qkv_transposed.dtype != torch.float16
            or qkv_transposed.ndim != 2
        ):
            raise ValueError("qkv_transposed must be a rank-2 CUDA fp16 tensor")
        rows, output_size = map(int, qkv_transposed.shape)
        if head_dim != 128:
            raise ValueError(
                f"fused QKV post-op requires head_dim=128, got {head_dim}"
            )
        if output_size != q_size + 2 * kv_size:
            raise ValueError(
                f"qkv_transposed must have {q_size + 2 * kv_size} columns"
            )
        if q_size % head_dim or kv_size % head_dim:
            raise ValueError("q_size and kv_size must be divisible by head_dim")
    leading_dim = int(qkv_transposed.stride(1))
    if int(qkv_transposed.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "qkv_transposed must use CUTLASS layout with stride "
            f"(1, >=M), got {tuple(qkv_transposed.stride())}"
        )
    if (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != qkv_transposed.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("cos_sin_cache must be contiguous rank-2 CUDA fp16")
    rotary_dim = int(cos_sin_cache.shape[1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}, got {rotary_dim}"
        )
    if (
        not position_ids.is_cuda
        or position_ids.device != qkv_transposed.device
        or position_ids.dtype != torch.int64
        or position_ids.numel() != rows
        or not position_ids.is_contiguous()
    ):
        raise ValueError(
            f"position_ids must be contiguous CUDA int64 with {rows} elements"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if out is None:
        output = torch.empty(
            (rows, output_size),
            device=qkv_transposed.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != qkv_transposed.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, output_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(rows, output_size)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(qkv_transposed.device).cuda_stream
    with _temporary_qkv_postop_config(postop_config):
        ret = lib.sparse24_cutlass_qkv_transpose_postop_f16_stream(
            ctypes.c_void_p(qkv_transposed.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(q_weight.data_ptr()) if q_weight is not None else None,
            ctypes.c_void_p(k_weight.data_ptr()) if k_weight is not None else None,
            ctypes.c_void_p(cos_sin_cache.data_ptr()),
            ctypes.c_void_p(position_ids.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(q_size),
            ctypes.c_int(kv_size),
            ctypes.c_int(head_dim),
            ctypes.c_int(leading_dim),
            ctypes.c_int(rotary_dim),
            ctypes.c_float(epsilon),
            ctypes.c_int(int(is_neox)),
            ctypes.c_int(int(normalize_qk)),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 fused QKV post-op failed with code {int(ret)}"
        )
    return output


def sparse24_qkv_transpose_add_routed_residual_postop(
    qkv_transposed: torch.Tensor,
    residual_transposed: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    postop_config: str | None = None,
) -> torch.Tensor:
    """Fuse routed residual addition, QKV materialization, norm, and RoPE."""

    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if normalize_qk:
        rows, output_size = _validate_qkv_rmsnorm_inputs(
            qkv_transposed,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
        )
    else:
        if (
            not qkv_transposed.is_cuda
            or qkv_transposed.dtype != torch.float16
            or qkv_transposed.ndim != 2
        ):
            raise ValueError("qkv_transposed must be a rank-2 CUDA fp16 tensor")
        rows, output_size = map(int, qkv_transposed.shape)
        if head_dim != 128:
            raise ValueError(
                f"fused QKV post-op requires head_dim=128, got {head_dim}"
            )
        if output_size != q_size + 2 * kv_size:
            raise ValueError(
                f"qkv_transposed must have {q_size + 2 * kv_size} columns"
            )
        if q_size % head_dim or kv_size % head_dim:
            raise ValueError("q_size and kv_size must be divisible by head_dim")
    leading_dim = int(qkv_transposed.stride(1))
    if int(qkv_transposed.stride(0)) != 1 or leading_dim < rows:
        raise ValueError(
            "qkv_transposed must use CUTLASS layout with stride "
            f"(1, >=M), got {tuple(qkv_transposed.stride())}"
        )
    if (
        not residual_transposed.is_cuda
        or residual_transposed.device != qkv_transposed.device
        or residual_transposed.dtype != torch.float16
        or residual_transposed.ndim != 2
        or int(residual_transposed.shape[1]) != output_size
    ):
        raise ValueError(
            "residual_transposed must be rank-2 CUDA fp16 on the same device "
            f"with {output_size} columns"
        )
    dense_count = int(residual_transposed.shape[0])
    residual_row_major = residual_transposed.is_contiguous()
    residual_leading_dim = int(
        residual_transposed.shape[1]
        if residual_row_major
        else residual_transposed.stride(1)
    )
    valid_transposed = (
        int(residual_transposed.stride(0)) == 1
        and residual_leading_dim >= dense_count
    )
    if dense_count <= 0 or dense_count > rows or not (
        residual_row_major or valid_transposed
    ):
        raise ValueError(
            "residual_transposed must be contiguous or use CUTLASS layout "
            f"with stride (1, >=D), got shape={tuple(residual_transposed.shape)} "
            f"stride={tuple(residual_transposed.stride())}"
        )
    if (
        not dense_slot_by_row.is_cuda
        or dense_slot_by_row.device != qkv_transposed.device
        or dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (rows,)
        or not dense_slot_by_row.is_contiguous()
    ):
        raise ValueError(
            f"dense_slot_by_row must be contiguous CUDA int32 shape {(rows,)}"
        )
    if (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != qkv_transposed.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("cos_sin_cache must be contiguous rank-2 CUDA fp16")
    rotary_dim = int(cos_sin_cache.shape[1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}, got {rotary_dim}"
        )
    if (
        not position_ids.is_cuda
        or position_ids.device != qkv_transposed.device
        or position_ids.dtype != torch.int64
        or position_ids.numel() != rows
        or not position_ids.is_contiguous()
    ):
        raise ValueError(
            f"position_ids must be contiguous CUDA int64 with {rows} elements"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if out is None:
        output = torch.empty(
            (rows, output_size),
            device=qkv_transposed.device,
            dtype=torch.float16,
        )
    else:
        output = out
        if (
            not output.is_cuda
            or output.device != qkv_transposed.device
            or output.dtype != torch.float16
            or tuple(output.shape) != (rows, output_size)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous CUDA fp16 with shape "
                f"{(rows, output_size)}"
            )

    lib = _load_library()
    stream = torch.cuda.current_stream(qkv_transposed.device).cuda_stream
    with _temporary_qkv_postop_config(postop_config):
        ret = (
            lib.sparse24_cutlass_qkv_transpose_add_routed_residual_postop_f16_stream(
                ctypes.c_void_p(qkv_transposed.data_ptr()),
                ctypes.c_void_p(residual_transposed.data_ptr()),
                ctypes.c_void_p(dense_slot_by_row.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_void_p(q_weight.data_ptr())
                if q_weight is not None
                else None,
                ctypes.c_void_p(k_weight.data_ptr())
                if k_weight is not None
                else None,
                ctypes.c_void_p(cos_sin_cache.data_ptr()),
                ctypes.c_void_p(position_ids.data_ptr()),
                ctypes.c_int(rows),
                ctypes.c_int(dense_count),
                ctypes.c_int(q_size),
                ctypes.c_int(kv_size),
                ctypes.c_int(head_dim),
                ctypes.c_int(leading_dim),
                ctypes.c_int(residual_leading_dim),
                ctypes.c_int(rotary_dim),
                ctypes.c_float(epsilon),
                ctypes.c_int(int(is_neox)),
                ctypes.c_int(int(normalize_qk)),
                ctypes.c_int(int(residual_row_major)),
                ctypes.c_void_p(stream),
            )
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused QKV routed-residual post-op failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_qkv_rmsnorm_inplace_(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
) -> None:
    """Apply Q/K per-head RMSNorm in-place to contiguous QKV output."""

    rows, _output_size = _validate_qkv_rmsnorm_inputs(
        qkv,
        q_weight,
        k_weight,
        q_size=q_size,
        kv_size=kv_size,
        head_dim=head_dim,
    )
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous for in-place QKV RMSNorm")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    lib = _load_library()
    stream = torch.cuda.current_stream(qkv.device).cuda_stream
    ret = lib.sparse24_cutlass_qkv_rmsnorm_inplace_f16_stream(
        ctypes.c_void_p(qkv.data_ptr()),
        ctypes.c_void_p(q_weight.data_ptr()),
        ctypes.c_void_p(k_weight.data_ptr()),
        ctypes.c_int(rows),
        ctypes.c_int(q_size),
        ctypes.c_int(kv_size),
        ctypes.c_int(head_dim),
        ctypes.c_float(epsilon),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24 fused in-place QKV RMSNorm failed with code {int(ret)}"
        )


def sparse24_qkv_add_routed_residual_postop_inplace_(
    qkv: torch.Tensor,
    residual: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    postop_config: str | None = None,
) -> torch.Tensor:
    """Add routed residuals, normalize Q/K, and apply RoPE in-place."""

    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if normalize_qk:
        rows, output_size = _validate_qkv_rmsnorm_inputs(
            qkv,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
        )
    else:
        if not qkv.is_cuda or qkv.dtype != torch.float16 or qkv.ndim != 2:
            raise ValueError("qkv must be a rank-2 CUDA fp16 tensor")
        rows, output_size = map(int, qkv.shape)
        if head_dim != 128:
            raise ValueError(
                f"fused QKV post-op requires head_dim=128, got {head_dim}"
            )
        if output_size != q_size + 2 * kv_size:
            raise ValueError(f"qkv must have {q_size + 2 * kv_size} columns")
        if q_size % head_dim or kv_size % head_dim:
            raise ValueError("q_size and kv_size must be divisible by head_dim")
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous for the in-place QKV post-op")
    if (
        not residual.is_cuda
        or residual.device != qkv.device
        or residual.dtype != torch.float16
        or residual.ndim != 2
        or int(residual.shape[1]) != output_size
        or not residual.is_contiguous()
    ):
        raise ValueError(
            "residual must be contiguous rank-2 CUDA fp16 on the same device "
            f"with {output_size} columns"
        )
    dense_count = int(residual.shape[0])
    if dense_count <= 0 or dense_count > rows:
        raise ValueError(
            f"residual row count must be in [1, {rows}], got {dense_count}"
        )
    if (
        not dense_slot_by_row.is_cuda
        or dense_slot_by_row.device != qkv.device
        or dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (rows,)
        or not dense_slot_by_row.is_contiguous()
    ):
        raise ValueError(
            f"dense_slot_by_row must be contiguous CUDA int32 shape {(rows,)}"
        )
    if (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != qkv.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("cos_sin_cache must be contiguous rank-2 CUDA fp16")
    rotary_dim = int(cos_sin_cache.shape[1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}, got {rotary_dim}"
        )
    if (
        not position_ids.is_cuda
        or position_ids.device != qkv.device
        or position_ids.dtype != torch.int64
        or position_ids.numel() != rows
        or not position_ids.is_contiguous()
    ):
        raise ValueError(
            f"position_ids must be contiguous CUDA int64 with {rows} elements"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    lib = _load_library()
    stream = torch.cuda.current_stream(qkv.device).cuda_stream
    with _temporary_qkv_postop_config(postop_config):
        ret = (
            lib.sparse24_cutlass_qkv_add_routed_residual_postop_inplace_f16_stream(
                ctypes.c_void_p(qkv.data_ptr()),
                ctypes.c_void_p(residual.data_ptr()),
                ctypes.c_void_p(dense_slot_by_row.data_ptr()),
                ctypes.c_void_p(q_weight.data_ptr())
                if q_weight is not None
                else None,
                ctypes.c_void_p(k_weight.data_ptr())
                if k_weight is not None
                else None,
                ctypes.c_void_p(cos_sin_cache.data_ptr()),
                ctypes.c_void_p(position_ids.data_ptr()),
                ctypes.c_int(rows),
                ctypes.c_int(dense_count),
                ctypes.c_int(q_size),
                ctypes.c_int(kv_size),
                ctypes.c_int(head_dim),
                ctypes.c_int(rotary_dim),
                ctypes.c_float(epsilon),
                ctypes.c_int(int(is_neox)),
                ctypes.c_int(int(normalize_qk)),
                ctypes.c_void_p(stream),
            )
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused row-major QKV routed-residual post-op failed "
            f"with code {int(ret)}"
        )
    return qkv


def sparse24_qkv_add_routed_residual_postop_cache_inplace_(
    qkv: torch.Tensor,
    residual: torch.Tensor,
    dense_slot_by_row: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
    is_neox: bool,
    q_weight: torch.Tensor | None = None,
    k_weight: torch.Tensor | None = None,
    postop_config: str | None = None,
) -> torch.Tensor:
    """Run routed QKV post-op and store K/V directly into the paged cache."""

    normalize_qk = q_weight is not None or k_weight is not None
    if normalize_qk and (q_weight is None or k_weight is None):
        raise ValueError("q_weight and k_weight must be provided together")
    if normalize_qk:
        rows, output_size = _validate_qkv_rmsnorm_inputs(
            qkv,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
        )
    else:
        if not qkv.is_cuda or qkv.dtype != torch.float16 or qkv.ndim != 2:
            raise ValueError("qkv must be a rank-2 CUDA fp16 tensor")
        rows, output_size = map(int, qkv.shape)
        if output_size != q_size + 2 * kv_size:
            raise ValueError(f"qkv must have {q_size + 2 * kv_size} columns")
    if head_dim != 128 or not is_neox:
        raise ValueError("direct KV-cache post-op requires Neox head_dim=128")
    if not qkv.is_contiguous() or q_size % head_dim or kv_size % head_dim:
        raise ValueError("qkv must be contiguous with valid Q/K/V head sizes")
    if (
        not residual.is_cuda
        or residual.device != qkv.device
        or residual.dtype != torch.float16
        or residual.ndim != 2
        or int(residual.shape[1]) != output_size
        or not residual.is_contiguous()
    ):
        raise ValueError(
            "residual must be contiguous rank-2 CUDA fp16 with matching width"
        )
    dense_count = int(residual.shape[0])
    if dense_count <= 0 or dense_count > rows:
        raise ValueError("residual row count must fit qkv rows")
    if (
        not dense_slot_by_row.is_cuda
        or dense_slot_by_row.device != qkv.device
        or dense_slot_by_row.dtype != torch.int32
        or tuple(dense_slot_by_row.shape) != (rows,)
        or not dense_slot_by_row.is_contiguous()
    ):
        raise ValueError(f"dense_slot_by_row must be CUDA int32 shape {(rows,)}")
    if (
        not cos_sin_cache.is_cuda
        or cos_sin_cache.device != qkv.device
        or cos_sin_cache.dtype != torch.float16
        or cos_sin_cache.ndim != 2
        or int(cos_sin_cache.shape[1]) != head_dim
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("cos_sin_cache must be contiguous CUDA fp16 [P, 128]")
    if (
        not position_ids.is_cuda
        or position_ids.device != qkv.device
        or position_ids.dtype != torch.int64
        or tuple(position_ids.shape) != (rows,)
        or not position_ids.is_contiguous()
    ):
        raise ValueError(
            f"position_ids must be contiguous CUDA int64 shape {(rows,)}"
        )
    if (
        not slot_mapping.is_cuda
        or slot_mapping.device != qkv.device
        or slot_mapping.dtype != torch.int64
        or slot_mapping.ndim != 1
        or int(slot_mapping.numel()) > rows
        or not slot_mapping.is_contiguous()
    ):
        raise ValueError(
            f"slot_mapping must be contiguous CUDA int64 with at most {rows} values"
        )
    kv_heads = kv_size // head_dim
    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        if (
            not cache.is_cuda
            or cache.device != qkv.device
            or cache.dtype != torch.float16
            or cache.ndim != 4
            or int(cache.shape[-1]) != head_dim
            or int(cache.stride(3)) != 1
        ):
            raise ValueError(
                f"{name} must be rank-4 CUDA fp16 with head_dim={head_dim} last"
            )
    if tuple(key_cache.shape) != tuple(value_cache.shape) or (
        tuple(key_cache.stride()) != tuple(value_cache.stride())
    ):
        raise ValueError("key_cache and value_cache shapes/strides must match")
    if int(key_cache.shape[2]) == kv_heads:
        block_size = int(key_cache.shape[1])
        cache_page_stride = int(key_cache.stride(1))
        cache_head_stride = int(key_cache.stride(2))
    elif int(key_cache.shape[1]) == kv_heads:
        block_size = int(key_cache.shape[2])
        cache_page_stride = int(key_cache.stride(2))
        cache_head_stride = int(key_cache.stride(1))
    else:
        raise ValueError(
            "KV cache must use NHD [blocks, block, heads, dim] or "
            "HND [blocks, heads, block, dim] layout"
        )
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    lib = _load_library()
    stream = torch.cuda.current_stream(qkv.device).cuda_stream
    with _temporary_qkv_postop_config(postop_config):
        ret = lib.sparse24_cutlass_qkv_add_routed_residual_postop_cache_inplace_f16_stream(
            ctypes.c_void_p(qkv.data_ptr()),
            ctypes.c_void_p(residual.data_ptr()),
            ctypes.c_void_p(dense_slot_by_row.data_ptr()),
            ctypes.c_void_p(q_weight.data_ptr()) if q_weight is not None else None,
            ctypes.c_void_p(k_weight.data_ptr()) if k_weight is not None else None,
            ctypes.c_void_p(cos_sin_cache.data_ptr()),
            ctypes.c_void_p(position_ids.data_ptr()),
            ctypes.c_void_p(slot_mapping.data_ptr()),
            ctypes.c_void_p(key_cache.data_ptr()),
            ctypes.c_void_p(value_cache.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(dense_count),
            ctypes.c_int(int(slot_mapping.numel())),
            ctypes.c_int(q_size),
            ctypes.c_int(kv_size),
            ctypes.c_int(head_dim),
            ctypes.c_int(head_dim),
            ctypes.c_int(block_size),
            ctypes.c_int64(int(key_cache.stride(0))),
            ctypes.c_int64(cache_page_stride),
            ctypes.c_int64(cache_head_stride),
            ctypes.c_float(epsilon),
            ctypes.c_int(1),
            ctypes.c_int(int(normalize_qk)),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24 fused QKV direct-cache post-op failed with code "
            f"{int(ret)}"
        )
    return qkv


def sparse24_mixed_dense_override_prepacked(
    X: torch.Tensor,
    W_nk: torch.Tensor,
    a_values: torch.Tensor,
    a_meta_e: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    dense_x: torch.Tensor | None = None,
    dense_y: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    pad_m_multiple: int | None = None,
    device_config: str | None = None,
) -> torch.Tensor:
    """Compute ``X @ W24`` and overwrite selected rows with dense ``X @ W``.

    ``W_nk`` is the original Linear weight with shape ``[N, K]``.  The C++
    bridge passes it to CUTLASS as a column-major ``[K, N]`` matrix, so the hot
    path does not materialize ``W.t().contiguous()``.
    """

    if (
        not X.is_cuda
        or not W_nk.is_cuda
        or not a_values.is_cuda
        or not a_meta_e.is_cuda
        or not row_indices.is_cuda
    ):
        raise ValueError("all mixed override inputs must be CUDA tensors")
    if X.dtype != torch.float16 or W_nk.dtype != torch.float16 or a_values.dtype != torch.float16:
        raise ValueError("mixed override currently supports fp16 only")
    if a_meta_e.dtype not in (torch.uint16, torch.int16):
        raise ValueError(f"a_meta_e dtype must be torch.uint16/int16, got {a_meta_e.dtype}")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if X.ndim != 2 or W_nk.ndim != 2 or a_values.ndim != 2 or a_meta_e.ndim != 1:
        raise ValueError("expected X/W/a_values rank-2 and a_meta_e rank-1")
    if row_indices.ndim != 1 or not row_indices.is_contiguous():
        raise ValueError("row_indices must be a contiguous rank-1 tensor")

    M, K = X.shape
    N = W_nk.shape[0]
    dense_rows = int(row_indices.numel())
    if dense_rows <= 0 or dense_rows > M:
        raise ValueError(f"row_indices length must be in [1, {M}], got {dense_rows}")
    if tuple(W_nk.shape) != (N, K):
        raise ValueError(f"W_nk must have shape [N, K] with K={K}, got {tuple(W_nk.shape)}")
    if K % 64 != 0:
        raise ValueError(f"CUTLASS mixed override requires K divisible by 64, got {K}")
    if N % 32 != 0:
        raise ValueError(f"CUTLASS mixed override requires N divisible by 32, got {N}")
    if a_values.shape != (N, K // 2):
        raise ValueError(f"a_values must have shape {(N, K // 2)}, got {tuple(a_values.shape)}")
    if a_meta_e.numel() != N * (K // 16):
        raise ValueError(
            f"a_meta_e must have {N * (K // 16)} elements, got {a_meta_e.numel()}"
        )

    M_orig = M
    if pad_m_multiple is None:
        pad_m_multiple = int(os.environ.get("SPECLINK_SPARSE24_PAD_M_MULTIPLE", "8"))
    if pad_m_multiple < 8 or pad_m_multiple % 8 != 0:
        raise ValueError(
            f"pad_m_multiple must be a positive multiple of 8, got {pad_m_multiple}"
        )
    M_run = ((M + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple
    if M_run != M:
        Xc = torch.empty((M_run, K), device=X.device, dtype=torch.float16)
        Xc[:M].copy_(X)
        Xc[M:].zero_()
    else:
        Xc = X.contiguous()

    W_nk = W_nk.contiguous()
    a_values = a_values.contiguous()
    a_meta_e = a_meta_e.contiguous()
    if dense_x is None:
        dense_x = torch.empty((dense_rows, K), device=X.device, dtype=torch.float16)
    elif (
        not dense_x.is_cuda
        or dense_x.device != X.device
        or dense_x.dtype != torch.float16
        or tuple(dense_x.shape) != (dense_rows, K)
        or not dense_x.is_contiguous()
    ):
        raise ValueError(
            f"dense_x must be contiguous CUDA fp16 shape {(dense_rows, K)}, "
            f"got shape={tuple(dense_x.shape)} device={dense_x.device} "
            f"dtype={dense_x.dtype}"
        )
    if dense_y is None:
        dense_y = torch.empty((dense_rows, N), device=X.device, dtype=torch.float16)
    elif (
        not dense_y.is_cuda
        or dense_y.device != X.device
        or dense_y.dtype != torch.float16
        or tuple(dense_y.shape) != (dense_rows, N)
        or not dense_y.is_contiguous()
    ):
        raise ValueError(
            f"dense_y must be contiguous CUDA fp16 shape {(dense_rows, N)}, "
            f"got shape={tuple(dense_y.shape)} device={dense_y.device} "
            f"dtype={dense_y.dtype}"
        )
    if workspace is None:
        sparse_tmp = torch.empty((N, M_run), device=X.device, dtype=torch.float16)
    elif (
        not workspace.is_cuda
        or workspace.device != X.device
        or workspace.dtype != torch.float16
        or tuple(workspace.shape) != (N, M_run)
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            f"workspace must be contiguous CUDA fp16 shape {(N, M_run)}, "
            f"got shape={tuple(workspace.shape)} device={workspace.device} "
            f"dtype={workspace.dtype}"
        )
    else:
        sparse_tmp = workspace
    if out is None:
        Y_run = torch.empty((M_run, N), device=X.device, dtype=torch.float16)
    else:
        if not out.is_cuda or out.device != X.device or out.dtype != torch.float16:
            raise ValueError("out must be a CUDA fp16 tensor on the same device as X")
        if tuple(out.shape) != (M_run, N):
            raise ValueError(f"out must have shape {(M_run, N)}, got {tuple(out.shape)}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        Y_run = out

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_sparse24_device_config(device_config):
        ret = lib.sparse24_cutlass_mixed_dense_override_f16_stream(
            ctypes.c_void_p(Xc.data_ptr()),
            ctypes.c_void_p(W_nk.data_ptr()),
            ctypes.c_void_p(a_values.data_ptr()),
            ctypes.c_void_p(a_meta_e.data_ptr()),
            ctypes.c_void_p(row_indices.data_ptr()),
            ctypes.c_void_p(dense_x.data_ptr()),
            ctypes.c_void_p(dense_y.data_ptr()),
            ctypes.c_void_p(sparse_tmp.data_ptr()),
            ctypes.c_void_p(Y_run.data_ptr()),
            ctypes.c_int(M_run),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_int(dense_rows),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_mixed_dense_override_f16 failed "
            f"with code {int(ret)}"
        )
    if M_run == M_orig:
        return Y_run
    return Y_run[:M_orig]


def dense_cutlass_device_gemm(
    X: torch.Tensor,
    W: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulator: str = "fp32",
    device_config: str | None = None,
) -> torch.Tensor:
    """Compute ``Y = X @ W`` through the local CUTLASS dense GEMM path."""

    if not X.is_cuda or not W.is_cuda:
        raise ValueError("X and W must be CUDA tensors")
    if X.dtype != torch.float16 or W.dtype != torch.float16:
        raise ValueError("CUTLASS dense GEMM currently supports fp16 only")
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("expected X and W to be rank-2 tensors")
    M, K = X.shape
    if tuple(W.shape)[0] != K:
        raise ValueError(f"W must have shape [K, N] with K={K}, got {tuple(W.shape)}")
    N = W.shape[1]
    if K % 64 != 0:
        raise ValueError(f"CUTLASS dense GEMM requires K divisible by 64, got {K}")
    if N % 8 != 0:
        raise ValueError(f"CUTLASS dense GEMM requires N divisible by 8, got {N}")
    X = X.contiguous()
    W = W.contiguous()
    if out is None:
        Y = torch.empty((M, N), device=X.device, dtype=torch.float16)
    else:
        if not out.is_cuda or out.device != X.device or out.dtype != torch.float16:
            raise ValueError("out must be a CUDA fp16 tensor on the same device as X")
        if tuple(out.shape) != (M, N):
            raise ValueError(f"out must have shape {(M, N)}, got {tuple(out.shape)}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        Y = out

    if accumulator not in {"fp16", "fp32"}:
        raise ValueError("accumulator must be 'fp16' or 'fp32'")
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    fn = (
        lib.dense_cutlass_device_gemm_f16_accum_f16_stream
        if accumulator == "fp16"
        else lib.dense_cutlass_device_gemm_f16_stream
    )
    with _temporary_dense_gemm_config(device_config):
        ret = fn(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_void_p(Y.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(f"dense_cutlass_device_gemm_f16 failed with code {int(ret)}")
    return Y


def dense_cutlass_weight_t_gemm(
    X: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulator: str = "fp32",
    device_config: str | None = None,
) -> torch.Tensor:
    """Compute ``X @ weight_t.T`` without transposing vLLM ``[N, K]`` weights."""

    if not X.is_cuda or not weight_t.is_cuda:
        raise ValueError("X and weight_t must be CUDA tensors")
    if X.device != weight_t.device:
        raise ValueError("X and weight_t must be on the same device")
    if X.dtype != torch.float16 or weight_t.dtype != torch.float16:
        raise ValueError("CUTLASS dense GEMM currently supports fp16 only")
    if X.ndim != 2 or weight_t.ndim != 2:
        raise ValueError("expected X and weight_t to be rank-2 tensors")
    M, K = X.shape
    N, weight_k = weight_t.shape
    if weight_k != K:
        raise ValueError(
            f"weight_t must have shape [N, {K}], got {tuple(weight_t.shape)}"
        )
    if K % 64 != 0 or N % 8 != 0:
        raise ValueError("CUTLASS dense GEMM requires K % 64 == 0 and N % 8 == 0")
    if accumulator not in {"fp16", "fp32"}:
        raise ValueError("accumulator must be 'fp16' or 'fp32'")
    X = X.contiguous()
    weight_t = weight_t.contiguous()
    if out is None:
        Y = torch.empty((M, N), device=X.device, dtype=torch.float16)
    else:
        if not out.is_cuda or out.device != X.device or out.dtype != torch.float16:
            raise ValueError("out must be a CUDA fp16 tensor on the same device as X")
        if tuple(out.shape) != (M, N) or not out.is_contiguous():
            raise ValueError(f"out must be contiguous with shape {(M, N)}")
        Y = out

    lib = _load_library()
    fn = (
        lib.dense_cutlass_weight_t_gemm_f16_accum_f16_stream
        if accumulator == "fp16"
        else lib.dense_cutlass_weight_t_gemm_f16_stream
    )
    stream = torch.cuda.current_stream(X.device).cuda_stream
    with _temporary_dense_gemm_config(device_config):
        ret = fn(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(weight_t.data_ptr()),
            ctypes.c_void_p(Y.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            f"dense_cutlass_weight_t_gemm_f16 failed with code {int(ret)}"
        )
    return Y


def dense_cutlass_weight_t_gemm_add(
    X: torch.Tensor,
    weight_t: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulator: str = "fp16",
    device_config: str | None = None,
) -> torch.Tensor:
    """Compute ``X @ weight_t.T + residual`` in one CUTLASS epilogue."""

    if not X.is_cuda or not weight_t.is_cuda or not residual.is_cuda:
        raise ValueError("X, weight_t, and residual must be CUDA tensors")
    if X.device != weight_t.device or X.device != residual.device:
        raise ValueError("all tensors must be on the same device")
    if any(tensor.dtype != torch.float16 for tensor in (X, weight_t, residual)):
        raise ValueError("CUTLASS dense GEMM currently supports fp16 only")
    if X.ndim != 2 or weight_t.ndim != 2 or residual.ndim != 2:
        raise ValueError("expected rank-2 tensors")
    M, K = X.shape
    N, weight_k = weight_t.shape
    if weight_k != K:
        raise ValueError(
            f"weight_t must have shape [N, {K}], got {tuple(weight_t.shape)}"
        )
    if tuple(residual.shape) != (M, N):
        raise ValueError(
            f"residual must have shape {(M, N)}, got {tuple(residual.shape)}"
        )
    if K % 64 != 0 or N % 8 != 0:
        raise ValueError("CUTLASS dense GEMM requires K % 64 == 0 and N % 8 == 0")
    if accumulator not in {"fp16", "fp32"}:
        raise ValueError("accumulator must be 'fp16' or 'fp32'")
    X = X.contiguous()
    weight_t = weight_t.contiguous()
    residual = residual.contiguous()
    if out is None:
        Y = torch.empty_like(residual)
    else:
        if not out.is_cuda or out.device != X.device or out.dtype != torch.float16:
            raise ValueError("out must be a CUDA fp16 tensor on the same device as X")
        if tuple(out.shape) != (M, N) or not out.is_contiguous():
            raise ValueError(f"out must be contiguous with shape {(M, N)}")
        Y = out

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    fn = (
        lib.dense_cutlass_weight_t_gemm_add_f16_accum_f16_stream
        if accumulator == "fp16"
        else lib.dense_cutlass_weight_t_gemm_add_f16_stream
    )
    with _temporary_dense_gemm_config(device_config):
        ret = fn(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(weight_t.data_ptr()),
            ctypes.c_void_p(residual.data_ptr()),
            ctypes.c_void_p(Y.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(N),
            ctypes.c_void_p(stream),
        )
    if int(ret) != 0:
        raise RuntimeError(
            "dense_cutlass_weight_t_gemm_add_f16 failed "
            f"with code {int(ret)}"
        )
    return Y


def dense_cutlass_simt_weight_t_gemm(
    X: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    config: str = "64x64x8",
) -> torch.Tensor:
    """Compute ``X @ weight_t.T`` on CUDA cores into sparse-view layout."""

    config_ids = {"64x64x8": 0, "128x64x8": 1}
    if config not in config_ids:
        raise ValueError(f"unsupported SIMT config: {config}")
    if not X.is_cuda or not weight_t.is_cuda:
        raise ValueError("X and weight_t must be CUDA tensors")
    if X.device != weight_t.device:
        raise ValueError("X and weight_t must be on the same device")
    if X.dtype != torch.float16 or weight_t.dtype != torch.float16:
        raise ValueError("SIMT residual GEMM currently supports fp16 only")
    if X.ndim != 2 or weight_t.ndim != 2:
        raise ValueError("X and weight_t must be rank-2")
    M, K = X.shape
    N, weight_k = weight_t.shape
    if weight_k != K:
        raise ValueError(
            f"weight_t must have shape [N, {K}], got {tuple(weight_t.shape)}"
        )
    if M % 8 != 0 or K % 8 != 0 or N % 8 != 0:
        raise ValueError("SIMT residual GEMM requires M, K, and N divisible by 8")
    X = X.contiguous()
    weight_t = weight_t.contiguous()
    if out is None:
        output = torch.empty_strided(
            (M, N), (1, M), device=X.device, dtype=torch.float16
        )
    else:
        if not out.is_cuda or out.device != X.device:
            raise ValueError("out must be a CUDA tensor on the input device")
        if out.dtype != torch.float16:
            raise ValueError("out must have dtype torch.float16")
        if tuple(out.shape) != (M, N) or tuple(out.stride()) != (1, M):
            raise ValueError(
                f"out must have shape/stride {(M, N)}/{(1, M)}, "
                f"got {tuple(out.shape)}/{tuple(out.stride())}"
            )
        output = out
    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.dense_cutlass_simt_weight_t_gemm_f16_stream(
        ctypes.c_void_p(weight_t.data_ptr()),
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(K),
        ctypes.c_int(N),
        ctypes.c_int(config_ids[config]),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "dense_cutlass_simt_weight_t_gemm_f16 failed with code "
            f"{int(ret)}"
        )
    return output


def sparse24_add_prefix_strided_(
    full_out: torch.Tensor,
    prefix_add: torch.Tensor,
    *,
    dense_rows: int,
) -> torch.Tensor:
    """In-place add a strided [dense_rows, N] prefix into a full sparse view."""

    if not full_out.is_cuda or not prefix_add.is_cuda:
        raise ValueError("full_out and prefix_add must be CUDA tensors")
    if full_out.device != prefix_add.device:
        raise ValueError("full_out and prefix_add must be on the same device")
    if full_out.dtype != torch.float16 or prefix_add.dtype != torch.float16:
        raise ValueError("prefix add currently supports fp16 only")
    if full_out.ndim != 2 or prefix_add.ndim != 2:
        raise ValueError("full_out and prefix_add must be rank-2 tensors")
    full_m, N = full_out.shape
    if dense_rows <= 0 or dense_rows > full_m:
        raise ValueError(f"dense_rows must be in [1, {full_m}], got {dense_rows}")
    if tuple(full_out.stride()) != (1, full_m):
        raise ValueError(
            f"full_out must have stride {(1, full_m)}, got {tuple(full_out.stride())}"
        )
    if tuple(prefix_add.shape) != (dense_rows, N):
        raise ValueError(
            f"prefix_add must have shape {(dense_rows, N)}, got {tuple(prefix_add.shape)}"
        )
    prefix_stride = prefix_add.stride()
    if prefix_stride[0] != 1 or prefix_stride[1] < dense_rows:
        raise ValueError(
            "prefix_add must be the non-contiguous sparse view with stride "
            f"(1, >= {dense_rows}), got {tuple(prefix_stride)}"
        )
    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    ret = lib.sparse24_cutlass_add_prefix_strided_f16_stream(
        ctypes.c_void_p(full_out.data_ptr()),
        ctypes.c_void_p(prefix_add.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(full_m),
        ctypes.c_int(prefix_stride[1]),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24_cutlass_add_prefix_strided_f16 failed with code {int(ret)}"
        )
    return full_out


def sparse24_add_indexed_rows_strided_(
    full_out: torch.Tensor,
    row_add: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """In-place scatter-add random selected rows into a full sparse view."""

    if not full_out.is_cuda or not row_add.is_cuda or not row_indices.is_cuda:
        raise ValueError("full_out, row_add, and row_indices must be CUDA tensors")
    if full_out.device != row_add.device or full_out.device != row_indices.device:
        raise ValueError("full_out, row_add, and row_indices must be on the same device")
    if full_out.dtype != torch.float16 or row_add.dtype != torch.float16:
        raise ValueError("indexed row add currently supports fp16 outputs only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if full_out.ndim != 2 or row_add.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("full_out and row_add must be rank-2; row_indices must be rank-1")
    full_m, N = full_out.shape
    dense_rows = int(row_indices.numel())
    if dense_rows <= 0 or dense_rows > full_m:
        raise ValueError(f"row_indices length must be in [1, {full_m}], got {dense_rows}")
    if tuple(full_out.stride()) != (1, full_m):
        raise ValueError(
            f"full_out must have stride {(1, full_m)}, got {tuple(full_out.stride())}"
        )
    if tuple(row_add.shape) != (dense_rows, N):
        raise ValueError(
            f"row_add must have shape {(dense_rows, N)}, got {tuple(row_add.shape)}"
        )
    row_stride = row_add.stride()
    if row_stride[0] != 1 or row_stride[1] < dense_rows:
        raise ValueError(
            "row_add must be the non-contiguous sparse view with stride "
            f"(1, >= {dense_rows}), got {tuple(row_stride)}"
        )
    if not row_indices.is_contiguous():
        raise ValueError("row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    ret = lib.sparse24_cutlass_add_indexed_rows_strided_f16_stream(
        ctypes.c_void_p(full_out.data_ptr()),
        ctypes.c_void_p(row_add.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(full_m),
        ctypes.c_int(row_stride[1]),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_add_indexed_rows_strided_f16 failed "
            f"with code {int(ret)}"
        )
    return full_out


def sparse24_add_indexed_rows_contiguous_(
    full_out: torch.Tensor,
    row_add: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """In-place add unique selected rows into a contiguous full output."""

    if not full_out.is_cuda or not row_add.is_cuda or not row_indices.is_cuda:
        raise ValueError("full_out, row_add, and row_indices must be CUDA tensors")
    if full_out.device != row_add.device or full_out.device != row_indices.device:
        raise ValueError("full_out, row_add, and row_indices must be on the same device")
    if full_out.dtype != torch.float16 or row_add.dtype != torch.float16:
        raise ValueError("indexed row add currently supports fp16 outputs only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if full_out.ndim != 2 or row_add.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("full_out and row_add must be rank-2; row_indices must be rank-1")
    full_m, N = full_out.shape
    dense_rows = int(row_indices.numel())
    if dense_rows <= 0 or dense_rows > full_m:
        raise ValueError(f"row_indices length must be in [1, {full_m}], got {dense_rows}")
    if not full_out.is_contiguous():
        raise ValueError("full_out must be contiguous")
    if tuple(row_add.shape) != (dense_rows, N):
        raise ValueError(
            f"row_add must have shape {(dense_rows, N)}, got {tuple(row_add.shape)}"
        )
    if not row_add.is_contiguous():
        raise ValueError("row_add must be contiguous")
    if not row_indices.is_contiguous():
        raise ValueError("row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    ret = lib.sparse24_cutlass_add_indexed_rows_contiguous_f16_stream(
        ctypes.c_void_p(full_out.data_ptr()),
        ctypes.c_void_p(row_add.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(full_m),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_add_indexed_rows_contiguous_f16 failed "
            f"with code {int(ret)}"
        )
    return full_out


def sparse24_add_indexed_rows_transposed_to_contiguous_(
    full_out: torch.Tensor,
    row_add: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Transpose strided correction rows and add them to contiguous output."""

    if not full_out.is_cuda or not row_add.is_cuda or not row_indices.is_cuda:
        raise ValueError("full_out, row_add, and row_indices must be CUDA tensors")
    if full_out.device != row_add.device or full_out.device != row_indices.device:
        raise ValueError("full_out, row_add, and row_indices must be on the same device")
    if full_out.dtype != torch.float16 or row_add.dtype != torch.float16:
        raise ValueError("indexed row add currently supports fp16 outputs only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if full_out.ndim != 2 or row_add.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("full_out and row_add must be rank-2; row_indices must be rank-1")
    full_m, N = full_out.shape
    dense_rows = int(row_indices.numel())
    if dense_rows <= 0 or dense_rows > full_m:
        raise ValueError(f"row_indices length must be in [1, {full_m}], got {dense_rows}")
    if not full_out.is_contiguous():
        raise ValueError("full_out must be contiguous")
    row_m, row_n = row_add.shape
    if row_m < dense_rows or row_n != N:
        raise ValueError(
            f"row_add must have shape [>= {dense_rows}, {N}], got {tuple(row_add.shape)}"
        )
    if tuple(row_add.stride()) != (1, row_m):
        raise ValueError(
            f"row_add must have stride {(1, row_m)}, got {tuple(row_add.stride())}"
        )
    if not row_indices.is_contiguous():
        raise ValueError("row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    ret = (
        lib.sparse24_cutlass_add_indexed_rows_transposed_to_contiguous_f16_stream(
            ctypes.c_void_p(full_out.data_ptr()),
            ctypes.c_void_p(row_add.data_ptr()),
            ctypes.c_void_p(row_indices.data_ptr()),
            ctypes.c_int(dense_rows),
            ctypes.c_int(full_m),
            ctypes.c_int(row_m),
            ctypes.c_int(N),
            ctypes.c_void_p(stream),
        )
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_add_indexed_rows_transposed_to_contiguous_f16 "
            f"failed with code {int(ret)}"
        )
    return full_out


def sparse24_sub_indexed_rows_contiguous_(
    full_out: torch.Tensor,
    row_sub: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """In-place subtract unique selected rows from a contiguous full output."""

    if not full_out.is_cuda or not row_sub.is_cuda or not row_indices.is_cuda:
        raise ValueError("full_out, row_sub, and row_indices must be CUDA tensors")
    if full_out.device != row_sub.device or full_out.device != row_indices.device:
        raise ValueError("full_out, row_sub, and row_indices must be on the same device")
    if full_out.dtype != torch.float16 or row_sub.dtype != torch.float16:
        raise ValueError("indexed row subtract currently supports fp16 outputs only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if full_out.ndim != 2 or row_sub.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("full_out and row_sub must be rank-2; row_indices must be rank-1")
    full_m, N = full_out.shape
    sparse_rows = int(row_indices.numel())
    if sparse_rows <= 0 or sparse_rows > full_m:
        raise ValueError(f"row_indices length must be in [1, {full_m}], got {sparse_rows}")
    if not full_out.is_contiguous():
        raise ValueError("full_out must be contiguous")
    if tuple(row_sub.shape) != (sparse_rows, N):
        raise ValueError(
            f"row_sub must have shape {(sparse_rows, N)}, got {tuple(row_sub.shape)}"
        )
    if not row_sub.is_contiguous():
        raise ValueError("row_sub must be contiguous")
    if not row_indices.is_contiguous():
        raise ValueError("row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    ret = lib.sparse24_cutlass_sub_indexed_rows_contiguous_f16_stream(
        ctypes.c_void_p(full_out.data_ptr()),
        ctypes.c_void_p(row_sub.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(sparse_rows),
        ctypes.c_int(full_m),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_sub_indexed_rows_contiguous_f16 failed "
            f"with code {int(ret)}"
        )
    return full_out


def sparse24_gather_rows_(
    X: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather selected rows from ``X`` into contiguous ``out``."""

    if not X.is_cuda or not row_indices.is_cuda or not out.is_cuda:
        raise ValueError("X, row_indices, and out must be CUDA tensors")
    if X.device != row_indices.device or X.device != out.device:
        raise ValueError("X, row_indices, and out must be on the same device")
    if X.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("row gather currently supports fp16 tensors only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if X.ndim != 2 or out.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("X and out must be rank-2; row_indices must be rank-1")
    dense_rows = int(row_indices.numel())
    M, K = X.shape
    if dense_rows <= 0 or dense_rows > M:
        raise ValueError(f"row_indices length must be in [1, {M}], got {dense_rows}")
    if tuple(out.shape) != (dense_rows, K):
        raise ValueError(f"out must have shape {(dense_rows, K)}, got {tuple(out.shape)}")
    if not X.is_contiguous() or not out.is_contiguous() or not row_indices.is_contiguous():
        raise ValueError("X, out, and row_indices must be contiguous")
    if K % 8 != 0:
        raise ValueError(f"row gather requires K divisible by 8, got {K}")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_gather_rows_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(K),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(f"sparse24_cutlass_gather_rows_f16 failed with code {int(ret)}")
    return out


def sparse24_gather_rows_strided_(
    X: torch.Tensor,
    row_indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather rows while preserving CUTLASS's transposed view layout."""

    if not X.is_cuda or not row_indices.is_cuda or not out.is_cuda:
        raise ValueError("X, row_indices, and out must be CUDA tensors")
    if X.device != row_indices.device or X.device != out.device:
        raise ValueError("X, row_indices, and out must be on the same device")
    if X.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("strided row gather currently supports fp16 tensors only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if X.ndim != 2 or out.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("X and out must be rank-2; row_indices must be rank-1")
    dense_rows = int(row_indices.numel())
    M, K = map(int, X.shape)
    if dense_rows <= 0 or dense_rows > M:
        raise ValueError(f"row_indices length must be in [1, {M}], got {dense_rows}")
    if tuple(out.shape) != (dense_rows, K):
        raise ValueError(f"out must have shape {(dense_rows, K)}, got {tuple(out.shape)}")
    if X.stride(0) != 1 or X.stride(1) != M:
        raise ValueError(f"X must have stride {(1, M)}, got {tuple(X.stride())}")
    if out.stride(0) != 1 or out.stride(1) < dense_rows:
        raise ValueError(
            "out must use transposed layout with stride "
            f"(1, >= {dense_rows}), got {tuple(out.stride())}"
        )
    if not row_indices.is_contiguous():
        raise ValueError("row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_gather_rows_strided_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(M),
        ctypes.c_int(out.stride(1)),
        ctypes.c_int(K),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            f"sparse24_cutlass_gather_rows_strided_f16 failed with code {int(ret)}"
        )
    return out


def sparse24_partition_rows_(
    X: torch.Tensor,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    dense_out: torch.Tensor,
    sparse_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition all rows of ``X`` into two contiguous outputs."""

    tensors = (X, dense_indices, sparse_indices, dense_out, sparse_out)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("partition inputs and outputs must be CUDA tensors")
    if any(tensor.device != X.device for tensor in tensors[1:]):
        raise ValueError("partition inputs and outputs must be on the same device")
    if (
        X.dtype != torch.float16
        or dense_out.dtype != torch.float16
        or sparse_out.dtype != torch.float16
    ):
        raise ValueError("row partition currently supports fp16 tensors only")
    if dense_indices.dtype != torch.int32 or sparse_indices.dtype != torch.int32:
        raise ValueError("partition row indices must be torch.int32")
    if X.ndim != 2 or dense_out.ndim != 2 or sparse_out.ndim != 2:
        raise ValueError("partition inputs and outputs must be rank-2")
    if dense_indices.ndim != 1 or sparse_indices.ndim != 1:
        raise ValueError("partition row indices must be rank-1")
    M, K = X.shape
    dense_rows = int(dense_indices.numel())
    sparse_rows = int(sparse_indices.numel())
    if dense_rows <= 0 or sparse_rows <= 0 or dense_rows + sparse_rows != M:
        raise ValueError("dense and sparse indices must form a non-empty row partition")
    if tuple(dense_out.shape) != (dense_rows, K):
        raise ValueError(f"dense_out must have shape {(dense_rows, K)}")
    if tuple(sparse_out.shape) != (sparse_rows, K):
        raise ValueError(f"sparse_out must have shape {(sparse_rows, K)}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("partition inputs and outputs must be contiguous")
    if K % 8 != 0:
        raise ValueError(f"row partition requires K divisible by 8, got {K}")

    lib = _load_library()
    stream = torch.cuda.current_stream(X.device).cuda_stream
    ret = lib.sparse24_cutlass_partition_rows_f16_stream(
        ctypes.c_void_p(X.data_ptr()),
        ctypes.c_void_p(dense_out.data_ptr()),
        ctypes.c_void_p(sparse_out.data_ptr()),
        ctypes.c_void_p(dense_indices.data_ptr()),
        ctypes.c_void_p(sparse_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(sparse_rows),
        ctypes.c_int(M),
        ctypes.c_int(K),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(f"sparse24 row partition failed with code {int(ret)}")
    return dense_out, sparse_out


def sparse24_merge_rows_(
    out: torch.Tensor,
    dense_values: torch.Tensor,
    sparse_values: torch.Tensor,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
) -> torch.Tensor:
    """Merge two contiguous row partitions into row-major ``out``."""

    tensors = (out, dense_values, sparse_values, dense_indices, sparse_indices)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("merge inputs and output must be CUDA tensors")
    if any(tensor.device != out.device for tensor in tensors[1:]):
        raise ValueError("merge inputs and output must be on the same device")
    if (
        out.dtype != torch.float16
        or dense_values.dtype != torch.float16
        or sparse_values.dtype != torch.float16
    ):
        raise ValueError("row merge currently supports fp16 tensors only")
    if dense_indices.dtype != torch.int32 or sparse_indices.dtype != torch.int32:
        raise ValueError("merge row indices must be torch.int32")
    if out.ndim != 2 or dense_values.ndim != 2 or sparse_values.ndim != 2:
        raise ValueError("merge inputs and output must be rank-2")
    if dense_indices.ndim != 1 or sparse_indices.ndim != 1:
        raise ValueError("merge row indices must be rank-1")
    M, N = out.shape
    dense_rows = int(dense_indices.numel())
    sparse_rows = int(sparse_indices.numel())
    if dense_rows <= 0 or sparse_rows <= 0 or dense_rows + sparse_rows != M:
        raise ValueError("dense and sparse indices must form a non-empty row partition")
    if tuple(dense_values.shape) != (dense_rows, N):
        raise ValueError(f"dense_values must have shape {(dense_rows, N)}")
    if tuple(sparse_values.shape) != (sparse_rows, N):
        raise ValueError(f"sparse_values must have shape {(sparse_rows, N)}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("merge inputs and output must be contiguous")
    if N % 8 != 0:
        raise ValueError(f"row merge requires N divisible by 8, got {N}")

    lib = _load_library()
    stream = torch.cuda.current_stream(out.device).cuda_stream
    ret = lib.sparse24_cutlass_merge_rows_f16_stream(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(dense_values.data_ptr()),
        ctypes.c_void_p(sparse_values.data_ptr()),
        ctypes.c_void_p(dense_indices.data_ptr()),
        ctypes.c_void_p(sparse_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(sparse_rows),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(f"sparse24 row merge failed with code {int(ret)}")
    return out


def sparse24_copy_indexed_rows_strided_(
    full_out: torch.Tensor,
    row_values: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """In-place scatter-copy selected rows into a full sparse view."""

    if not full_out.is_cuda or not row_values.is_cuda or not row_indices.is_cuda:
        raise ValueError("full_out, row_values, and row_indices must be CUDA tensors")
    if full_out.device != row_values.device or full_out.device != row_indices.device:
        raise ValueError("full_out, row_values, and row_indices must be on the same device")
    if full_out.dtype != torch.float16 or row_values.dtype != torch.float16:
        raise ValueError("indexed row copy currently supports fp16 outputs only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if full_out.ndim != 2 or row_values.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("full_out and row_values must be rank-2; row_indices must be rank-1")
    full_m, N = full_out.shape
    dense_rows = int(row_indices.numel())
    if dense_rows <= 0 or dense_rows > full_m:
        raise ValueError(f"row_indices length must be in [1, {full_m}], got {dense_rows}")
    if tuple(full_out.stride()) != (1, full_m):
        raise ValueError(
            f"full_out must have stride {(1, full_m)}, got {tuple(full_out.stride())}"
        )
    if tuple(row_values.shape) != (dense_rows, N):
        raise ValueError(
            f"row_values must have shape {(dense_rows, N)}, got {tuple(row_values.shape)}"
        )
    row_stride = row_values.stride()
    if row_stride[0] != 1 or row_stride[1] < dense_rows:
        raise ValueError(
            "row_values must be a non-contiguous sparse view with stride "
            f"(1, >= {dense_rows}), got {tuple(row_stride)}"
        )
    if not row_indices.is_contiguous():
        raise ValueError("row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    ret = lib.sparse24_cutlass_copy_indexed_rows_strided_f16_stream(
        ctypes.c_void_p(full_out.data_ptr()),
        ctypes.c_void_p(row_values.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(full_m),
        ctypes.c_int(row_stride[1]),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_copy_indexed_rows_strided_f16 failed "
            f"with code {int(ret)}"
        )
    return full_out


def sparse24_copy_indexed_rows_contiguous_(
    full_out: torch.Tensor,
    row_values: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """In-place scatter-copy contiguous rows into strided or row-major output."""

    if not full_out.is_cuda or not row_values.is_cuda or not row_indices.is_cuda:
        raise ValueError("full_out, row_values, and row_indices must be CUDA tensors")
    if full_out.device != row_values.device or full_out.device != row_indices.device:
        raise ValueError("full_out, row_values, and row_indices must be on the same device")
    if full_out.dtype != torch.float16 or row_values.dtype != torch.float16:
        raise ValueError("indexed row copy currently supports fp16 outputs only")
    if row_indices.dtype != torch.int32:
        raise ValueError(f"row_indices must be torch.int32, got {row_indices.dtype}")
    if full_out.ndim != 2 or row_values.ndim != 2 or row_indices.ndim != 1:
        raise ValueError("full_out and row_values must be rank-2; row_indices must be rank-1")
    full_m, N = full_out.shape
    dense_rows = int(row_indices.numel())
    if dense_rows <= 0 or dense_rows > full_m:
        raise ValueError(f"row_indices length must be in [1, {full_m}], got {dense_rows}")
    rowmajor_output = full_out.is_contiguous()
    if not rowmajor_output and tuple(full_out.stride()) != (1, full_m):
        raise ValueError(
            "full_out must be contiguous or have sparse-view stride "
            f"{(1, full_m)}, got {tuple(full_out.stride())}"
        )
    if tuple(row_values.shape) != (dense_rows, N):
        raise ValueError(
            f"row_values must have shape {(dense_rows, N)}, got {tuple(row_values.shape)}"
        )
    if not row_values.is_contiguous() or not row_indices.is_contiguous():
        raise ValueError("row_values and row_indices must be contiguous")

    lib = _load_library()
    stream = torch.cuda.current_stream(full_out.device).cuda_stream
    copy_fn = (
        lib.sparse24_cutlass_copy_indexed_rows_rowmajor_f16_stream
        if rowmajor_output
        else lib.sparse24_cutlass_copy_indexed_rows_contiguous_f16_stream
    )
    ret = copy_fn(
        ctypes.c_void_p(full_out.data_ptr()),
        ctypes.c_void_p(row_values.data_ptr()),
        ctypes.c_void_p(row_indices.data_ptr()),
        ctypes.c_int(dense_rows),
        ctypes.c_int(full_m),
        ctypes.c_int(N),
        ctypes.c_void_p(stream),
    )
    if int(ret) != 0:
        raise RuntimeError(
            "sparse24_cutlass_copy_indexed_rows_contiguous_f16 failed "
            f"with code {int(ret)}"
        )
    return full_out
