"""Shared CUDA-extension toolchain and loader helpers.

The ``spec`` environment uses PyTorch built against CUDA 13.0 while this host
also has a newer system toolkit.  Every SpecLink extension must therefore use
the CUDA compiler shipped in the active conda environment and fail closed on
contradictory explicit settings.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Sequence

import torch


_CUDA_RELEASE_PATTERN = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")


def cuda_release(value: object) -> tuple[int, int] | None:
    match = _CUDA_RELEASE_PATTERN.search(str(value or ""))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def nvcc_release(nvcc: Path) -> tuple[int, int]:
    try:
        completed = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot query CUDA compiler {nvcc}: {error}") from error
    match = re.search(
        r"\brelease\s+(\d+)\.(\d+)\b",
        f"{completed.stdout}\n{completed.stderr}",
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(
            f"cannot determine CUDA release from `{nvcc} --version`"
        )
    return int(match.group(1)), int(match.group(2))


def _cuda_executable(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.parent != Path(".") or expanded.is_absolute():
        return expanded.resolve()
    discovered = shutil.which(value)
    return (Path(discovered) if discovered else expanded).resolve()


def _conda_cuda_home(release: tuple[int, int]) -> Path:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return (
        Path(sys.prefix)
        / "lib"
        / python_dir
        / "site-packages"
        / "nvidia"
        / f"cu{release[0]}"
    ).resolve()


def configure_cuda_toolchain(cpp_extension: Any) -> tuple[Path, Path] | None:
    """Pin ``cpp_extension`` to PyTorch's CUDA release.

    CUDA 13.0 is the supported local stack.  For other PyTorch builds the
    caller retains the normal ``cpp_extension`` discovery behavior.
    """

    torch_release = cuda_release(torch.version.cuda)
    if torch_release != (13, 0):
        return None

    home_text = os.environ.get("CUDA_HOME")
    compiler_text = os.environ.get("CUDACXX")
    explicit_home = Path(home_text).expanduser().resolve() if home_text else None
    explicit_compiler = _cuda_executable(compiler_text) if compiler_text else None

    if explicit_home is not None and explicit_compiler is not None:
        home_nvcc = (explicit_home / "bin" / "nvcc").resolve()
        if explicit_compiler != home_nvcc:
            raise RuntimeError(
                "conflicting CUDA toolchains: CUDA_HOME selects "
                f"{home_nvcc}, but CUDACXX selects {explicit_compiler}"
            )

    if explicit_home is not None:
        cuda_home = explicit_home
        nvcc = explicit_compiler or (cuda_home / "bin" / "nvcc").resolve()
    elif explicit_compiler is not None:
        nvcc = explicit_compiler
        if nvcc.name != "nvcc" or nvcc.parent.name != "bin":
            raise RuntimeError(
                f"cannot infer CUDA_HOME from CUDACXX={nvcc}; set CUDA_HOME"
            )
        cuda_home = nvcc.parent.parent.resolve()
    else:
        cuda_home = _conda_cuda_home(torch_release)
        nvcc = (cuda_home / "bin" / "nvcc").resolve()

    if not nvcc.is_file():
        raise RuntimeError(
            f"CUDA 13.0 compiler {nvcc} does not exist; install or select the "
            "matching conda nvidia/cu13 stack"
        )
    compiler_release = nvcc_release(nvcc)
    if compiler_release != torch_release:
        raise RuntimeError(
            "CUDA compiler mismatch: PyTorch uses "
            f"{torch_release[0]}.{torch_release[1]}, but {nvcc} reports "
            f"{compiler_release[0]}.{compiler_release[1]}"
        )

    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CUDACXX"] = str(nvcc)
    cpp_extension.CUDA_HOME = str(cuda_home)
    return cuda_home, nvcc


def load_cuda_extension(
    *,
    name: str,
    sources: Sequence[Path],
    required: Sequence[Path] = (),
    build_dir: Path,
    include_cutlass: bool = True,
    verbose_env: str,
) -> Any:
    """Build one SM120 extension with the shared fail-closed toolchain."""

    from torch.utils import cpp_extension

    configure_cuda_toolchain(cpp_extension)
    source_paths = tuple(Path(path).resolve() for path in sources)
    required_paths = (*source_paths, *(Path(path).resolve() for path in required))
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError("CUDA extension sources are missing: " + ", ".join(missing))

    repo_root = Path(__file__).resolve().parents[3]
    csrc = Path(__file__).resolve().parent / "csrc"
    include_paths = [str(csrc)]
    if include_cutlass:
        cutlass = repo_root / "vllm/.deps/cutlass-src/include"
        if not cutlass.is_dir():
            raise RuntimeError(f"vendored CUTLASS include tree is missing: {cutlass}")
        include_paths.insert(0, str(cutlass))

    build_dir = Path(build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(
        name=name,
        sources=[str(path) for path in source_paths],
        extra_include_paths=include_paths,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-arch=sm_120",
            "-diag-suppress=177",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        is_python_module=True,
        verbose=os.environ.get(verbose_env, "0") == "1",
    )


__all__ = [
    "configure_cuda_toolchain",
    "cuda_release",
    "load_cuda_extension",
    "nvcc_release",
]
