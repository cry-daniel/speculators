#!/usr/bin/env python3
"""Compile the SM120 ports and check representative arbitrary N:M formats."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from other_systems import (
    apply_nm_mask,
    flash_llm_linear,
    prepare_flash_llm,
    prepare_sparta,
    prepare_spinfer,
    sparta_linear,
    spinfer_linear,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(20260723)
    dense = torch.randn((256, 256), dtype=torch.bfloat16)
    x = torch.randn((128, 256), dtype=torch.bfloat16, device="cuda")
    methods = (
        ("flash_llm", prepare_flash_llm, flash_llm_linear),
        ("spinfer", prepare_spinfer, spinfer_linear),
    )
    for fmt in ("1:4", "5:8", "3:4", "4:4"):
        weight = apply_nm_mask(dense, fmt)
        reference = x @ weight.cuda().t()
        cases = list(methods)
        cases.append(
            (
                "sparta",
                lambda value, fmt=fmt: prepare_sparta(value, fmt),
                sparta_linear,
            )
        )
        for name, prepare, linear in cases:
            output = linear(x, prepare(weight))
            difference = (output.float() - reference.float()).abs()
            if not torch.allclose(output, reference, atol=0.5, rtol=0.1):
                raise RuntimeError(
                    f"{name}/{fmt} failed: max={difference.max().item()}, "
                    f"mean={difference.mean().item()}"
                )
            print(
                f"{name:10s} {fmt:4s} max={difference.max().item():.6f} "
                f"mean={difference.mean().item():.6f}"
            )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
