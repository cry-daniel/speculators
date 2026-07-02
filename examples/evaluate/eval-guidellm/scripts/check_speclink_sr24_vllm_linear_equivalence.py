#!/usr/bin/env python3
"""Check SR24 all-corrected equivalence on vLLM linear modules.

This is a focused diagnostic for the Llama fused MLP path.  It does not load a
full model; instead it constructs a small vLLM MergedColumnParallelLinear under
a module name that matches the real Llama layer naming scheme, applies SR24, and
compares the all-corrected output against the original dense vLLM forward.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = REPO_ROOT / "vllm"
sys.path.insert(0, str(VLLM_ROOT))

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
import vllm.model_executor.models.llama as llama_module  # noqa: E402
from vllm.model_executor.models.llama import LlamaMLP  # noqa: E402
from vllm.speclink_sr24 import apply_sr24_from_env, sparse_linear_output  # noqa: E402


class LayerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            bias=False,
            disable_tp=True,
            prefix="model.layers.16.mlp",
        )


class WrappedModel(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.model = nn.Module()
        layers = []
        for _idx in range(17):
            layers.append(nn.Identity())
        layers[16] = LayerBlock(hidden_size, intermediate_size)
        self.model.layers = nn.ModuleList(layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--rows", type=int, default=37)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Activation and weight dtype for the diagnostic module.",
    )
    parser.add_argument(
        "--llama-gate-up-shape",
        action="store_true",
        help=(
            "Use Llama-3.1-8B MLP shape: hidden=4096, "
            "intermediate=14336. The selected --leaf controls whether the "
            "checked Linear is gate_up_proj or down_proj."
        ),
    )
    parser.add_argument(
        "--leaf",
        choices=["gate_up_proj", "down_proj"],
        default="gate_up_proj",
        help="Which Llama MLP Linear module to check.",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help=(
            "Compare the full LlamaMLP forward with both gate_up_proj and "
            "down_proj rewritten. This exercises the real vLLM hook path."
        ),
    )
    parser.add_argument(
        "--mlp-target-leafs",
        default="gate_up_proj,down_proj",
        help=(
            "Comma-separated LlamaMLP leafs to rewrite when --mlp is set. "
            "Use gate_up_proj to match the current gate-up-only SR24 serving "
            "quality diagnostics."
        ),
    )
    parser.add_argument(
        "--residual-backends",
        default="dense_rows,compressed_dense",
        help=(
            "Comma-separated residual backends to test. Use dense_rows as the "
            "one-GEMM correctness reference and compressed_dense for the "
            "sparse-base plus residual reconstruction path."
        ),
    )
    parser.add_argument(
        "--loose-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for the loose equivalence check.",
    )
    parser.add_argument(
        "--loose-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the loose equivalence check.",
    )
    parser.add_argument(
        "--reduce-cpu-sync",
        action="store_true",
        help=(
            "Exercise the SR24 reduce-CPU-sync Linear path. This is useful for "
            "checking all-corrected compressed residual fast paths used by "
            "serving diagnostics."
        ),
    )
    parser.add_argument(
        "--compressed-residual-triton",
        action="store_true",
        help="Use the Triton compressed-residual matmul path when available.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable equivalence results.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Print and write failed equivalence rows without raising.",
    )
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _clear_sr24_env() -> None:
    for key in list(os.environ):
        if (
            key.startswith("SPECLINK_SR24_")
            or key.startswith("SPECLINK_STRUCTURED_24_")
            or key.startswith("SPECLINK_TOKEN_DENSE_")
        ):
            os.environ.pop(key, None)


def _configure_sr24_env(
    residual_backend: str,
    *,
    leaf: str,
    reduce_cpu_sync: bool,
    compressed_residual_triton: bool,
) -> None:
    _clear_sr24_env()
    os.environ["SPECLINK_SR24_ENABLE"] = "1"
    os.environ["SPECLINK_SR24_MODE"] = "all_corrected"
    os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = residual_backend
    os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = "cuda"
    os.environ["SPECLINK_SR24_TARGET_LEAFS"] = leaf
    os.environ["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = leaf
    os.environ["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = f"{leaf}=16"
    os.environ["SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH"] = "0"
    os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1" if reduce_cpu_sync else "0"
    os.environ["SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON"] = (
        "1" if compressed_residual_triton else "0"
    )
    os.environ["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
    os.environ["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"


def _configure_sr24_mlp_env(
    residual_backend: str,
    *,
    target_leafs: str,
    reduce_cpu_sync: bool,
    compressed_residual_triton: bool,
) -> None:
    _clear_sr24_env()
    os.environ["SPECLINK_SR24_ENABLE"] = "1"
    os.environ["SPECLINK_SR24_MODE"] = "all_corrected"
    os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = residual_backend
    os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = "cuda"
    leafs = _parse_csv(target_leafs)
    if not leafs:
        raise RuntimeError("--mlp-target-leafs must select at least one leaf")
    os.environ["SPECLINK_SR24_TARGET_LEAFS"] = ",".join(leafs)
    os.environ["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = ",".join(leafs)
    os.environ["SPECLINK_SR24_RESIDUAL_LAYER_IDS_BY_LEAF"] = ";".join(
        f"{leaf}=16" for leaf in leafs
    )
    os.environ["SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH"] = "0"
    os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1" if reduce_cpu_sync else "0"
    os.environ["SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON"] = (
        "1" if compressed_residual_triton else "0"
    )
    os.environ["SPECLINK_STRUCTURED_24_ENABLE"] = "0"
    os.environ["SPECLINK_TOKEN_DENSE_ENABLE"] = "0"


def _set_llama_sr24_hooks(enabled: bool) -> None:
    llama_module._SPECLINK_SR24_ENABLED = enabled
    llama_module._SPECLINK_SR24_LINEAR_HOOKS_ENABLED = enabled


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch_sparse SR24 equivalence check")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    hidden_size = int(args.hidden_size)
    intermediate_size = int(args.intermediate_size)
    rows = int(args.rows)
    if args.llama_gate_up_shape:
        hidden_size = 4096
        intermediate_size = 14336

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])

    initialized = False
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment()
            initialize_model_parallel(tensor_model_parallel_size=1)
            initialized = True

            dense_model = WrappedModel(hidden_size, intermediate_size).to(
                device=device, dtype=dtype
            )
            dense_mlp = dense_model.model.layers[16].mlp
            dense_linear = getattr(dense_mlp, args.leaf)
            with torch.no_grad():
                dense_mlp.gate_up_proj.weight.normal_(mean=0.0, std=0.02)
                dense_mlp.down_proj.weight.normal_(mean=0.0, std=0.02)
                original_gate_up = dense_mlp.gate_up_proj.weight.detach().clone()
                original_down = dense_mlp.down_proj.weight.detach().clone()
                original_weight = dense_linear.weight.detach().clone()
            input_size = (
                hidden_size
                if args.mlp or args.leaf == "gate_up_proj"
                else intermediate_size
            )
            x = torch.randn(rows, input_size, device=device, dtype=dtype)

            _set_llama_sr24_hooks(False)
            if args.mlp:
                dense_output = dense_mlp(x)
            else:
                dense_output, _ = dense_linear(x)
            dense_output = dense_output.contiguous()
            del dense_model

            results = []
            for residual_backend in _parse_csv(args.residual_backends):
                model = WrappedModel(hidden_size, intermediate_size).to(
                    device=device, dtype=dtype
                )
                mlp = model.model.layers[16].mlp
                linear = getattr(mlp, args.leaf)
                with torch.no_grad():
                    mlp.gate_up_proj.weight.copy_(original_gate_up)
                    mlp.down_proj.weight.copy_(original_down)
                    linear.weight.copy_(original_weight)

                if args.mlp:
                    _configure_sr24_mlp_env(
                        residual_backend,
                        target_leafs=args.mlp_target_leafs,
                        reduce_cpu_sync=bool(args.reduce_cpu_sync),
                        compressed_residual_triton=bool(
                            args.compressed_residual_triton
                        ),
                    )
                    _set_llama_sr24_hooks(True)
                else:
                    _configure_sr24_env(
                        residual_backend,
                        leaf=args.leaf,
                        reduce_cpu_sync=bool(args.reduce_cpu_sync),
                        compressed_residual_triton=bool(
                            args.compressed_residual_triton
                        ),
                    )
                stats = apply_sr24_from_env(
                    model,
                    context=f"vllm_linear_equivalence_{residual_backend}",
                )
                expected_modules = (
                    len(_parse_csv(args.mlp_target_leafs)) if args.mlp else 1
                )
                if (
                    not stats
                    or int(stats.get("module_count_attached") or 0)
                    != expected_modules
                ):
                    raise RuntimeError(
                        f"expected {expected_modules} attached SR24 module(s) for "
                        f"{residual_backend}, got {stats}"
                    )
                if args.mlp:
                    sr24_output = mlp(x)
                else:
                    sr24_output = sparse_linear_output(linear, x)
                if sr24_output is None:
                    raise RuntimeError(
                        f"SR24 sparse_linear_output returned None for "
                        f"{residual_backend}"
                    )
                torch.cuda.synchronize()
                diff = (dense_output - sr24_output).abs()
                max_abs = float(diff.max().item())
                mean_abs = float(diff.mean().item())
                exact = bool(torch.equal(dense_output, sr24_output))
                close = bool(torch.allclose(dense_output, sr24_output, atol=0, rtol=0))
                loose_close = bool(
                    torch.allclose(
                        dense_output,
                        sr24_output,
                        atol=args.loose_atol,
                        rtol=args.loose_rtol,
                    )
                )
                results.append(
                    {
                        "backend": residual_backend,
                        "exact": exact,
                        "close": close,
                        "loose_close": loose_close,
                        "max_abs": max_abs,
                        "mean_abs": mean_abs,
                        "shape": tuple(dense_output.shape),
                        "reduce_cpu_sync": bool(args.reduce_cpu_sync),
                        "compressed_residual_triton": bool(
                            args.compressed_residual_triton
                        ),
                        "dtype": str(dtype),
                        "scope": "mlp" if args.mlp else args.leaf,
                    }
                )
                del model
                del sr24_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if initialized:
            destroy_model_parallel()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    failed = False
    for result in results:
        print(
            "sr24_vllm_linear_equivalence "
            f"backend={result['backend']} exact={result['exact']} "
            f"close={result['close']} loose_close={result['loose_close']} "
            f"max_abs={result['max_abs']:.8f} "
            f"mean_abs={result['mean_abs']:.8f} "
            f"reduce_cpu_sync={result['reduce_cpu_sync']} "
            f"compressed_residual_triton={result['compressed_residual_triton']} "
            f"shape={result['shape']} dtype={result['dtype']} "
            f"scope={result['scope']}"
        )
        failed = failed or not result["loose_close"]
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "failed": failed,
                    "args": vars(args) | {
                        "output_json": str(args.output_json.resolve())
                        if args.output_json
                        else None,
                    },
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if failed and not args.allow_failure:
        raise RuntimeError(
            "one or more SR24 all-corrected outputs are not numerically close "
            "to dense vLLM output"
        )


if __name__ == "__main__":
    main()
