#!/usr/bin/env python3
"""Focused SR24 fixed-block output-buffer equivalence check.

The full correctness script covers many historical SR24 branches and can fail
when an old assertion no longer matches the current implementation.  This small
check targets one current systems question: enabling
SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER should not change the local fixed-prefix
row-routed MLP result for one verifier block.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = REPO_ROOT / "vllm"
sys.path.insert(0, str(VLLM_ROOT))

from vllm.speclink_sr24 import (  # noqa: E402
    FixedPrefixRouteDescriptor,
    VerifyResidualPlan,
    _semi_structured_linear,
    _sparse_base_weight,
    apply_sr24_from_env,
    begin_verify_context,
    end_verify_context,
    row_routed_mlp_output,
    sparse_backend_active,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(128, 256, bias=False)
        self.down_proj = nn.Linear(128, 128, bias=False)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock()])


def _clear_sr24_env() -> None:
    for key in list(os.environ):
        if (
            key.startswith("SPECLINK_SR24_")
            or key.startswith("SPECLINK_STRUCTURED_24_")
            or key.startswith("SPECLINK_TOKEN_DENSE_")
        ):
            os.environ.pop(key, None)


def _configure_env(*, output_buffer: bool) -> None:
    os.environ["SPECLINK_SR24_ENABLE"] = "1"
    os.environ["SPECLINK_SR24_MODE"] = "selective"
    os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "dense_rows"
    os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = "cuda"
    os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1"
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP"] = "1"
    os.environ["SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS"] = "1"
    os.environ["SPECLINK_SR24_TARGET_LEAFS"] = "gate_up_proj,down_proj"
    os.environ["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = "gate_up_proj,down_proj"
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "fixed_prefix"
    os.environ["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = "2"
    os.environ["SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH"] = "1"
    os.environ["SPECLINK_SR24_FIXED_BLOCK_INPUT_BUFFER"] = "1"
    os.environ["SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER"] = (
        "1" if output_buffer else "0"
    )
    os.environ["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
    os.environ["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"


def _silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate_up[:, :128]) * gate_up[:, 128:]


def _run_once(*, output_buffer: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    _clear_sr24_env()
    _configure_env(output_buffer=output_buffer)
    torch.manual_seed(seed)

    model = TinyModel().cuda().to(dtype=torch.float16)
    originals = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor)
    }
    stats = apply_sr24_from_env(model, context="unit_fixed_block_output_buffer")
    if stats is None:
        raise RuntimeError("SR24 did not attach to tiny model")
    gate_module = model.layers[0].gate_up_proj
    down_module = model.layers[0].down_proj
    if not sparse_backend_active(gate_module) or not sparse_backend_active(down_module):
        raise RuntimeError(f"SR24 sparse backend inactive: {stats}")

    active = 2
    valid_width = 4
    scheduled_width = valid_width + 1
    prefix = 2
    rows = active * scheduled_width
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    mask = torch.tensor(
        [True, True, False, False, True,
         True, True, False, False, True],
        device="cuda",
    )
    plan = VerifyResidualPlan(
        mask=mask,
        state="mixed",
        residual_rows=mask.nonzero(as_tuple=False).squeeze(1),
        base_rows=(~mask).nonzero(as_tuple=False).squeeze(1),
        fixed_prefix_route=FixedPrefixRouteDescriptor(
            active_count=active,
            scheduled_width=scheduled_width,
            valid_width=valid_width,
            prefix=prefix,
            dense_width=prefix + 1,
            base_width=valid_width - prefix,
        ),
    )

    gate_base = _semi_structured_linear(x, _sparse_base_weight(gate_module))
    gate_dense = F.linear(x, originals["layers.0.gate_up_proj"])
    gate_expected = gate_base.clone()
    gate_expected[mask] = gate_dense[mask]
    act_expected = _silu_and_mul(gate_expected)

    down_base = _semi_structured_linear(act_expected, _sparse_base_weight(down_module))
    down_dense = F.linear(act_expected, originals["layers.0.down_proj"])
    expected = down_base.clone()
    expected[mask] = down_dense[mask]

    token = begin_verify_context(plan)
    try:
        actual = row_routed_mlp_output(
            gate_module,
            down_module,
            x,
            _silu_and_mul,
        )
    finally:
        end_verify_context(token)
    if actual is None:
        raise RuntimeError("row_routed_mlp_output returned None")
    return actual.detach().clone(), expected.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    no_buffer, expected = _run_once(output_buffer=False, seed=args.seed)
    with_buffer, expected_with_buffer = _run_once(output_buffer=True, seed=args.seed)
    torch.cuda.synchronize()

    no_buffer_diff = float((no_buffer - expected).abs().max().item())
    with_buffer_diff = float((with_buffer - expected_with_buffer).abs().max().item())
    cross_diff = float((no_buffer - with_buffer).abs().max().item())
    result = {
        "no_buffer_max_abs_diff": no_buffer_diff,
        "with_buffer_max_abs_diff": with_buffer_diff,
        "cross_max_abs_diff": cross_diff,
        "passed": (
            no_buffer_diff <= 5e-2
            and with_buffer_diff <= 5e-2
            and cross_diff <= 5e-2
        ),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
    print("speclink_sr24_fixed_block_output_buffer=ok")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
