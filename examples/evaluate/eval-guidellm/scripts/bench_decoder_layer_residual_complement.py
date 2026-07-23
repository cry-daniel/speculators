#!/usr/bin/env python3
"""One-decoder-layer decoding benchmark for residual-complement routing.

The benchmark models one real TP=1 Qwen/Llama decoder layer with synthetic
BF16 weights: RMSNorm, fused qkv, optional Qwen q/k RMSNorm, RoPE, explicit
non-fused GQA attention (QK / softmax / AV), o projection, residual RMSNorm,
gate+up, SiLU*up, down, and the final residual add.

Each request contributes one current token and seven draft tokens.  Thus
``M = 8 * batch_size`` remains 512/1024/2048 for B=64/128/256.  Dense quotas
1/8..8/8 are evaluated with both global-batch and per-request confidence
ranking.  Optimized E2E uses concurrent cuSPARSELt base + Split-K2 complement.
For the measured M=512, 1/4-dense gate_up case, it instead uses the selected
fused token-partition path (sparse-row cuSPARSELt plus dense-row dual-HMMA.SP).
Pure 2:4 applies the prepared cuSPARSELt base to all four linears.  The
diagnostic breakdown deliberately serializes the sparse stages so GEMM,
gather, reduce and scatter are additive and interpretable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sparse24_benchmark_common as common
from residual_complement_runtime import (
    launch_fused,
    launch_separate,
    should_use_fused_gateup,
)
from speculators.speclink import (
    SPARSE_RESIDUAL_SMEM,
    TP1_FUSED_WEIGHT_SHAPES,
    cusparselt_sparse_residual_indexed_add_,
    cusparselt_sparse_residual_indexed_gather,
    cusparselt_sparse_residual_residual_linear_splitk2,
    cusparselt_sparse_residual_sparse_linear,
    prepare_cusparselt_sparse_residual_weight,
    prepare_online_sparse24_weight,
    select_cusparselt_algorithm,
)


SCRIPT = Path(__file__).resolve()
EVAL_ROOT = SCRIPT.parent.parent
MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
BATCH_SIZES = (64, 128, 256)
SCOPES = ("global", "per_request")
EIGHTHS = tuple(range(1, 9))
CATEGORIES = ("GEMM", "Attention", "Softmax", "Gather", "Scatter", "Reduce", "Others")
SEED = 20260721


@dataclass(frozen=True, slots=True)
class ModelSpec:
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    qwen_qk_norm: bool
    rope_theta: float
    rms_eps: float = 1e-6


MODEL_SPECS = {
    "qwen3_8b": ModelSpec(4096, 32, 8, 128, 12288, True, 1_000_000.0),
    "llama3_1_8b": ModelSpec(4096, 32, 8, 128, 14336, False, 500_000.0),
    "qwen3_14b": ModelSpec(5120, 40, 8, 128, 17408, True, 1_000_000.0),
    "qwen3_32b": ModelSpec(5120, 64, 8, 128, 25600, True, 1_000_000.0),
    "llama3_70b": ModelSpec(8192, 64, 8, 128, 28672, False, 500_000.0),
}
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama3-70B",
}


@dataclass(slots=True)
class Projection:
    dense: torch.Tensor
    runtime: Any
    algorithm_id: int


@dataclass(slots=True)
class LayerState:
    spec: ModelSpec
    projections: dict[str, Projection]
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor | None
    k_norm: torch.Tensor | None
    past_k: torch.Tensor
    past_v: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    causal_mask: torch.Tensor
    resources: Any
    gate_up_fused_output: torch.Tensor


class PhaseEvents:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def run(self, category: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        self.pairs.append((category, start, end))
        return output

    def finish(self) -> dict[str, float]:
        torch.cuda.synchronize()
        totals = {category: 0.0 for category in CATEGORIES}
        for category, start, end in self.pairs:
            totals[category] += 1000.0 * float(start.elapsed_time(end))
        return totals


def parse_csv(value: str, allowed: Sequence[str], name: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(f"invalid {name}: {unknown or value}")
    return values


def parse_int_csv(value: str, allowed: Sequence[int], name: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part.strip())
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(f"invalid {name}: {unknown or value}")
    return values


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate(x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        c = cos.view(1, cos.shape[0], 1, cos.shape[1])
        s = sin.view(1, sin.shape[0], 1, sin.shape[1])
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def attention_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    past_k: torch.Tensor,
    scale: float,
    causal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [B,Q,H,D] -> [B,KV,G,Q,D], avoiding materialized repeat_kv.
    batch, query, heads, dim = q.shape
    kv_heads = k.shape[2]
    groups = heads // kv_heads
    qg = q.view(batch, query, kv_heads, groups, dim).permute(0, 2, 3, 1, 4)
    new_k = k.permute(0, 2, 1, 3)
    all_k = torch.cat((past_k, new_k), dim=2)
    scores = torch.einsum("bhgqd,bhsd->bhgqs", qg, all_k).mul_(scale)
    scores.masked_fill_(causal_mask.view(1, 1, 1, query, -1), -float("inf"))
    return scores, qg


def attention_av(
    probs: torch.Tensor,
    qg: torch.Tensor,
    v: torch.Tensor,
    past_v: torch.Tensor,
) -> torch.Tensor:
    new_v = v.permute(0, 2, 1, 3)
    all_v = torch.cat((past_v, new_v), dim=2)
    context = torch.einsum("bhgqs,bhsd->bhgqd", probs, all_v)
    batch, kv_heads, groups, query, dim = context.shape
    return context.permute(0, 3, 1, 2, 4).reshape(batch * query, kv_heads * groups * dim)


def make_confidence(batch: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(common.stable_seed(seed, "confidence", batch))
    selected_prob = torch.rand(
        (batch, 7), device=device, generator=generator, dtype=torch.float32
    ).mul_(0.45).add_(0.5)
    return selected_prob.cumprod(dim=1)


def dense_indices_from_confidence(
    confidence: torch.Tensor,
    eighths: int,
    scope: str,
) -> torch.Tensor:
    batch = int(confidence.shape[0])
    current = torch.arange(batch, device=confidence.device, dtype=torch.int64) * 8
    if eighths == 1:
        return current
    draft_rows = current[:, None] + torch.arange(
        1, 8, device=confidence.device, dtype=torch.int64
    )[None, :]
    if scope == "per_request":
        order = torch.argsort(confidence, dim=1, descending=True, stable=True)
        selected = draft_rows.gather(1, order[:, : eighths - 1]).flatten()
    elif scope == "global":
        count = (eighths - 1) * batch
        order = torch.argsort(confidence.flatten(), descending=True, stable=True)[:count]
        selected = draft_rows.flatten().index_select(0, order)
    else:
        raise ValueError(scope)
    return torch.cat((current, selected)).sort().values.contiguous()


def prepare_projection(
    model: str,
    name: str,
    max_m: int,
    device: torch.device,
    seed: int,
) -> Projection:
    n, k = TP1_FUSED_WEIGHT_SHAPES[model][name]
    case = common.ShapeCase(model, name, max_m, k, n)
    dense, sparse = common.make_synthetic_weight(case, seed, device)
    canonical = prepare_online_sparse24_weight(
        dense, sparse, variant=SPARSE_RESIDUAL_SMEM
    )
    runtime = prepare_cusparselt_sparse_residual_weight(
        canonical, sparse_weight=sparse
    )
    sample = common.make_input(case, seed, device, purpose="decoder_layer_tune")
    algorithm_id = select_cusparselt_algorithm(runtime.cusparselt, sample)
    del sparse, canonical, sample
    gc.collect()
    torch.cuda.empty_cache()
    return Projection(dense=dense, runtime=runtime, algorithm_id=algorithm_id)


def prepare_layer(model: str, max_batch: int, device: torch.device, seed: int) -> LayerState:
    spec = MODEL_SPECS[model]
    max_m = max_batch * 8
    projections = {
        name: prepare_projection(model, name, max_m, device, seed)
        for name in ("qkv", "o", "gate_up", "down")
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(common.stable_seed(seed, "layer_aux", model, max_batch))
    norm = lambda size: torch.ones(size, dtype=torch.bfloat16, device=device)
    past = 127
    past_k = torch.randn(
        (max_batch, spec.kv_heads, past, spec.head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    past_v = torch.randn(
        past_k.shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    positions = torch.arange(127, 135, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        spec.rope_theta
        ** (torch.arange(0, spec.head_dim, 2, device=device).float() / spec.head_dim)
    )
    angles = positions[:, None] * inv_freq[None, :]
    total_keys = past + 8
    key_positions = torch.arange(total_keys, device=device)
    query_positions = torch.arange(past, past + 8, device=device)
    causal_mask = key_positions[None, :] > query_positions[:, None]
    return LayerState(
        spec=spec,
        projections=projections,
        input_norm=norm(spec.hidden),
        post_norm=norm(spec.hidden),
        q_norm=norm(spec.head_dim) if spec.qwen_qk_norm else None,
        k_norm=norm(spec.head_dim) if spec.qwen_qk_norm else None,
        past_k=past_k,
        past_v=past_v,
        rope_cos=angles.cos().to(torch.bfloat16),
        rope_sin=angles.sin().to(torch.bfloat16),
        causal_mask=causal_mask,
        resources=common.create_multistream_resources(device),
        gate_up_fused_output=torch.empty(
            (max_m, projections["gate_up"].runtime.n),
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def optimized_linear(
    name: str,
    projection: Projection,
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    resources: Any,
    gate_up_fused_output: torch.Tensor,
) -> torch.Tensor:
    if name == "gate_up" and should_use_fused_gateup(
        x, dense_indices, projection.runtime
    ):
        return launch_fused(
            x.contiguous(),
            dense_indices,
            sparse_indices,
            projection.runtime,
            gate_up_fused_output[: x.shape[0]],
            resources,
            variant="auto",
            optimized_routes=True,
        )
    return launch_separate(
        x.contiguous(),
        dense_indices,
        projection.runtime,
        resources,
        complement_variant="feature128_token64_s4",
        complement_first=True,
        optimized_gather=True,
        optimized_merge=True,
        splitk2_complement=True,
        splitk2_variant="auto",
    )


def diagnostic_linear(
    projection: Projection,
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    timer: PhaseEvents,
) -> torch.Tensor:
    dense_x = timer.run(
        "Gather",
        lambda: cusparselt_sparse_residual_indexed_gather(
            x.contiguous(), dense_indices
        ),
    )
    base = timer.run(
        "GEMM",
        lambda: cusparselt_sparse_residual_sparse_linear(
            x.contiguous(), projection.runtime
        ),
    )
    partials = timer.run(
        "GEMM",
        lambda: cusparselt_sparse_residual_residual_linear_splitk2(
            dense_x, projection.runtime, variant="auto"
        ),
    )
    correction = timer.run("Reduce", lambda: partials[0].add(partials[1]))
    return timer.run(
        "Scatter",
        lambda: cusparselt_sparse_residual_indexed_add_(
            base, correction, dense_indices
        ),
    )


def layer_forward(
    hidden: torch.Tensor,
    state: LayerState,
    *,
    method: str,
    dense_indices: torch.Tensor | None = None,
    sparse_indices: torch.Tensor | None = None,
    timer: PhaseEvents | None = None,
) -> torch.Tensor:
    batch = hidden.shape[0] // 8
    spec = state.spec

    def phase(category: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        return timer.run(category, fn) if timer is not None else fn()

    def linear(name: str, x: torch.Tensor) -> torch.Tensor:
        projection = state.projections[name]
        if method == "dense":
            return phase("GEMM", lambda: F.linear(x, projection.dense))
        if method == "pure_sparse":
            return phase(
                "GEMM",
                lambda: cusparselt_sparse_residual_sparse_linear(
                    x.contiguous(), projection.runtime
                ),
            )
        assert dense_indices is not None
        if timer is not None:
            return diagnostic_linear(projection, x, dense_indices, timer)
        assert sparse_indices is not None
        return optimized_linear(
            name,
            projection,
            x,
            dense_indices,
            sparse_indices,
            state.resources,
            state.gate_up_fused_output,
        )

    residual = hidden
    x = phase("Others", lambda: rms_norm(hidden, state.input_norm, spec.rms_eps))
    qkv = linear("qkv", x)
    q_size = spec.q_heads * spec.head_dim
    kv_size = spec.kv_heads * spec.head_dim
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(batch, 8, spec.q_heads, spec.head_dim)
    k = k.view(batch, 8, spec.kv_heads, spec.head_dim)
    v = v.view(batch, 8, spec.kv_heads, spec.head_dim)
    if spec.qwen_qk_norm:
        assert state.q_norm is not None and state.k_norm is not None
        q = phase("Others", lambda: rms_norm(q, state.q_norm, spec.rms_eps))
        k = phase("Others", lambda: rms_norm(k, state.k_norm, spec.rms_eps))
    q, k = phase(
        "Others", lambda: apply_rope(q, k, state.rope_cos, state.rope_sin)
    )
    scores, qg = phase(
        "Attention",
        lambda: attention_qk(
            q,
            k,
            state.past_k[:batch],
            spec.head_dim**-0.5,
            state.causal_mask,
        ),
    )
    probs = phase("Softmax", lambda: torch.softmax(scores.float(), dim=-1).to(q.dtype))
    attn = phase(
        "Attention",
        lambda: attention_av(probs, qg, v, state.past_v[:batch]),
    )
    attn_out = linear("o", attn)
    residual = phase("Others", lambda: residual.add(attn_out))
    x = phase("Others", lambda: rms_norm(residual, state.post_norm, spec.rms_eps))
    gate_up = linear("gate_up", x)
    gate, up = gate_up.chunk(2, dim=-1)
    x = phase("Others", lambda: F.silu(gate).mul_(up))
    down = linear("down", x)
    return phase("Others", lambda: residual.add(down))


def time_once(fn: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return 1000.0 * float(start.elapsed_time(end))


def choose_iterations(fn: Callable[[], torch.Tensor], target_us: float, maximum: int) -> int:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    latency = max(1.0, time_once(fn))
    return max(1, min(maximum, int(math.ceil(target_us / latency))))


def measure(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    trials: int,
    target_us: float,
    max_iterations: int,
    eviction: torch.Tensor,
) -> tuple[dict[str, float], list[float], int]:
    iterations = choose_iterations(fn, target_us, max_iterations)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        eviction.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(1000.0 * float(start.elapsed_time(end)) / iterations)
    return common.summarize(samples), samples, iterations


def measure_breakdown(
    fn: Callable[[PhaseEvents], torch.Tensor], trials: int
) -> dict[str, dict[str, float]]:
    for _ in range(3):
        fn(PhaseEvents())
    torch.cuda.synchronize()
    samples = {category: [] for category in CATEGORIES}
    for _ in range(trials):
        timer = PhaseEvents()
        fn(timer)
        totals = timer.finish()
        for category in CATEGORIES:
            samples[category].append(totals[category])
    return {category: common.summarize(values) for category, values in samples.items()}


def make_hidden(model: str, batch: int, device: torch.device, seed: int) -> torch.Tensor:
    spec = MODEL_SPECS[model]
    generator = torch.Generator(device=device)
    generator.manual_seed(common.stable_seed(seed, "hidden", model, batch))
    return torch.randn(
        (batch * 8, spec.hidden),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = args.worker_model
    state = prepare_layer(model, max(args.batch_sizes), device, args.seed)
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024, dtype=torch.uint8, device=device
    )
    eviction.zero_()
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []

    for batch in args.batch_sizes:
        hidden = make_hidden(model, batch, device, args.seed)
        confidence = make_confidence(batch, device, args.seed)
        dense_eager = lambda: layer_forward(hidden, state, method="dense")
        dense_graph = common.capture_graph(dense_eager, warmup=3)
        dense_fn = dense_graph.graph.replay
        dense_summary, dense_samples, dense_iterations = measure(
            dense_fn,
            warmup=args.warmup,
            trials=args.trials,
            target_us=args.target_interval_us,
            max_iterations=args.max_iterations,
            eviction=eviction,
        )
        pure_eager = lambda: layer_forward(hidden, state, method="pure_sparse")
        pure_graph = common.capture_graph(pure_eager, warmup=3)
        pure_fn = pure_graph.graph.replay
        pure_summary, pure_samples, pure_iterations = measure(
            pure_fn,
            warmup=args.warmup,
            trials=args.trials,
            target_us=args.target_interval_us,
            max_iterations=args.max_iterations,
            eviction=eviction,
        )
        dense_breakdown = measure_breakdown(
            lambda timer: layer_forward(hidden, state, method="dense", timer=timer),
            args.breakdown_trials,
        )
        pure_breakdown = measure_breakdown(
            lambda timer: layer_forward(
                hidden, state, method="pure_sparse", timer=timer
            ),
            args.breakdown_trials,
        )
        for category, summary in dense_breakdown.items():
            breakdown_rows.append(
                {
                    "model": model,
                    "batch_size": batch,
                    "M": batch * 8,
                    "routing_scope": "baseline",
                    "dense_fraction": "8/8",
                    "method": "dense_cublas",
                    "category": category,
                    **summary,
                }
            )
        for category, summary in pure_breakdown.items():
            breakdown_rows.append(
                {
                    "model": model,
                    "batch_size": batch,
                    "M": batch * 8,
                    "routing_scope": "baseline",
                    "dense_fraction": "0/8",
                    "method": "pure_24_cusparselt",
                    "category": category,
                    **summary,
                }
            )
        for trial, value in enumerate(dense_samples):
            raw.append(
                {
                    "model": model,
                    "batch_size": batch,
                    "M": batch * 8,
                    "routing_scope": "baseline",
                    "dense_fraction": "8/8",
                    "method": "dense_cublas",
                    "trial": trial,
                    "iterations": dense_iterations,
                    "latency_us": value,
                }
            )
        for trial, value in enumerate(pure_samples):
            raw.append(
                {
                    "model": model,
                    "batch_size": batch,
                    "M": batch * 8,
                    "routing_scope": "baseline",
                    "dense_fraction": "0/8",
                    "method": "pure_24_cusparselt",
                    "trial": trial,
                    "iterations": pure_iterations,
                    "latency_us": value,
                }
            )

        for scope in args.scopes:
            for eighths in args.eighths:
                indices = dense_indices_from_confidence(confidence, eighths, scope)
                route_mask = torch.ones(
                    batch * 8, dtype=torch.bool, device=device
                )
                route_mask[indices] = False
                sparse_indices = torch.nonzero(route_mask, as_tuple=False).flatten().contiguous()
                uses_fused_gate_up = batch == 64 and int(indices.numel()) == 128
                method_variant = (
                    "mixed_fused_gate_up"
                    if uses_fused_gate_up
                    else "separate_splitk2"
                )
                method_eager = lambda indices=indices, sparse_indices=sparse_indices: layer_forward(
                    hidden,
                    state,
                    method="residual_complement",
                    dense_indices=indices,
                    sparse_indices=sparse_indices,
                )
                method_graph = common.capture_multistream_graph(
                    method_eager,
                    state.resources,
                    warmup=3,
                    device=device,
                )
                method_fn = method_graph.graph.replay
                check_timer = PhaseEvents()
                reference = layer_forward(
                    hidden,
                    state,
                    method="residual_complement",
                    dense_indices=indices,
                    timer=check_timer,
                )
                check_timer.finish()
                difference = (method_graph.output.float() - reference.float()).abs()
                correctness = {
                    "correct": bool(
                        torch.allclose(
                            method_graph.output,
                            reference,
                            rtol=1e-1,
                            atol=2e-1,
                        )
                    ),
                    "max_abs_error": float(difference.max().item()),
                    "mean_abs_error": float(difference.mean().item()),
                }
                if not correctness["correct"]:
                    raise RuntimeError(
                        f"optimized/diagnostic mismatch for {model} B={batch} "
                        f"{scope} {eighths}/8: {correctness}"
                    )
                method_summary, samples, iterations = measure(
                    method_fn,
                    warmup=args.warmup,
                    trials=args.trials,
                    target_us=args.target_interval_us,
                    max_iterations=args.max_iterations,
                    eviction=eviction,
                )
                method_breakdown = measure_breakdown(
                    lambda timer, indices=indices: layer_forward(
                        hidden,
                        state,
                        method="residual_complement",
                        dense_indices=indices,
                        timer=timer,
                    ),
                    args.breakdown_trials,
                )
                fraction = f"{eighths}/8"
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "batch_size": batch,
                        "M": batch * 8,
                        "draft_tokens_per_request": 7,
                        "context_length_including_current": 128,
                        "max_visible_length": 135,
                        "routing_scope": scope,
                        "dense_fraction": fraction,
                        "dense_rows": int(indices.numel()),
                        "sparse_rows": batch * 8 - int(indices.numel()),
                        "dense_cublas_median_us": dense_summary["median_us"],
                        "dense_cublas_p10_us": dense_summary["p10_us"],
                        "dense_cublas_p90_us": dense_summary["p90_us"],
                        "cusparselt_pure_24_median_us": pure_summary["median_us"],
                        "cusparselt_pure_24_p10_us": pure_summary["p10_us"],
                        "cusparselt_pure_24_p90_us": pure_summary["p90_us"],
                        "method_median_us": method_summary["median_us"],
                        "method_p10_us": method_summary["p10_us"],
                        "method_p90_us": method_summary["p90_us"],
                        "speedup_vs_dense": dense_summary["median_us"]
                        / method_summary["median_us"],
                        "pure_24_speedup_vs_dense": (
                            dense_summary["median_us"] / pure_summary["median_us"]
                        ),
                        "hybrid_speedup_vs_pure_24": (
                            pure_summary["median_us"] / method_summary["median_us"]
                        ),
                        "method_iterations": iterations,
                        "dense_iterations": dense_iterations,
                        "method_variant": method_variant,
                        "correctness": correctness,
                    }
                )
                for category, summary in method_breakdown.items():
                    breakdown_rows.append(
                        {
                            "model": model,
                            "batch_size": batch,
                            "M": batch * 8,
                            "routing_scope": scope,
                            "dense_fraction": fraction,
                            "method": "residual_complement_diagnostic_serial",
                            "category": category,
                            **summary,
                        }
                    )
                for trial, value in enumerate(samples):
                    raw.append(
                        {
                            "model": model,
                            "batch_size": batch,
                            "M": batch * 8,
                            "routing_scope": scope,
                            "dense_fraction": fraction,
                            "method": f"residual_complement_{method_variant}",
                            "trial": trial,
                            "iterations": iterations,
                            "latency_us": value,
                        }
                    )
                print(
                    f"{model} B={batch} {scope} D={fraction}: "
                    f"hybrid={method_summary['median_us']:.3f} us, "
                    f"dense={dense_summary['median_us']:.3f} us, "
                    f"pure24={pure_summary['median_us']:.3f} us, "
                    f"hybrid/dense="
                    f"{dense_summary['median_us'] / method_summary['median_us']:.3f}x",
                    flush=True,
                )
                del indices, sparse_indices, route_mask, method_graph, reference, difference
                gc.collect()
        del hidden, confidence, dense_graph, pure_graph
        torch.cuda.empty_cache()

    payload = {"model": model, "rows": rows, "raw": raw, "breakdown": breakdown_rows}
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def plot_speedup(path: Path, rows: list[dict[str, Any]]) -> None:
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    batches = sorted({int(row["batch_size"]) for row in rows})
    figure, axes = plt.subplots(
        len(models),
        len(batches),
        figsize=(5 * len(batches), 3.2 * len(models)),
        squeeze=False,
    )
    for ri, model in enumerate(models):
        for ci, batch in enumerate(batches):
            axis = axes[ri][ci]
            for scope, marker in (("global", "o"), ("per_request", "s")):
                selected = sorted(
                    (
                        row for row in rows
                        if row["model"] == model
                        and int(row["batch_size"]) == batch
                        and row["routing_scope"] == scope
                    ),
                    key=lambda row: int(row["dense_fraction"].split("/")[0]),
                )
                if selected:
                    axis.plot(
                        [int(row["dense_fraction"].split("/")[0]) / 8 for row in selected],
                        [float(row["speedup_vs_dense"]) for row in selected],
                        marker=marker,
                        label=scope,
                    )
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(f"{MODEL_LABELS[model]}, B={batch}, M={batch * 8}")
            axis.set_xlabel("Dense-token fraction")
            axis.set_ylabel("Speedup vs dense cuBLAS")
            axis.grid(alpha=0.25)
            if ri == 0 and ci == len(batches) - 1:
                axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_absolute_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    figure, axes = plt.subplots(
        1, len(models), figsize=(7.0 * len(models), 4.2), squeeze=False
    )
    colors = ("#4c78a8", "#59a14f", "#f28e2b")
    for ci, model in enumerate(models):
        axis = axes[0][ci]
        scopes = {str(row["routing_scope"]) for row in rows if row["model"] == model}
        preferred_scope = "global" if "global" in scopes else sorted(scopes)[0]
        selected = sorted(
            (
                row for row in rows
                if row["model"] == model
                and row["routing_scope"] == preferred_scope
            ),
            key=lambda row: (
                int(row["M"]),
                int(str(row["dense_fraction"]).split("/")[0]),
            ),
        )
        labels = [f"M{row['M']}\nD={row['dense_fraction']}" for row in selected]
        positions = np.arange(len(selected), dtype=np.float64)
        width = 0.25
        series = (
            [float(row["dense_cublas_median_us"]) for row in selected],
            [float(row["cusparselt_pure_24_median_us"]) for row in selected],
            [float(row["method_median_us"]) for row in selected],
        )
        for offset, values, label, color in zip(
            (-width, 0.0, width),
            series,
            ("Dense cuBLAS", "Pure 2:4 cuSPARSELt", "Hybrid"),
            colors,
            strict=True,
        ):
            axis.bar(positions + offset, values, width, label=label, color=color)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylabel("Median layer latency (us)")
        axis.set_title(f"{MODEL_LABELS[model]} ({preferred_scope})")
        axis.grid(axis="y", alpha=0.25)
        if ci == len(models) - 1:
            axis.legend(fontsize=9)
    figure.suptitle("One decoder layer: absolute latency", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_breakdown(path: Path, rows: list[dict[str, Any]]) -> None:
    # Keep the plot readable: show global routing at 1/8, 1/2, and 8/8 for all
    # models/B.  The CSV retains every fraction and both routing scopes.
    selected_fractions = ("1/8", "4/8", "8/8")
    models = [model for model in MODELS if any(row["model"] == model for row in rows)]
    batches = sorted({int(row["batch_size"]) for row in rows})
    figure, axes = plt.subplots(
        len(models),
        len(batches),
        figsize=(5.3 * len(batches), 3.5 * len(models)),
        squeeze=False,
    )
    colors = dict(zip(CATEGORIES, plt.cm.tab10.colors, strict=False))
    for ri, model in enumerate(models):
        for ci, batch in enumerate(batches):
            axis = axes[ri][ci]
            available_fractions = [
                fraction
                for fraction in selected_fractions
                if any(
                    row["model"] == model
                    and int(row["batch_size"]) == batch
                    and row["routing_scope"] == "global"
                    and row["dense_fraction"] == fraction
                    for row in rows
                )
            ]
            labels = ["dense", *available_fractions]
            bottoms = np.zeros(len(labels))
            for category in CATEGORIES:
                values = []
                dense = next(
                    row for row in rows
                    if row["model"] == model
                    and int(row["batch_size"]) == batch
                    and row["method"] == "dense_cublas"
                    and row["category"] == category
                )
                values.append(float(dense["median_us"]))
                for fraction in available_fractions:
                    match = next(
                        row for row in rows
                        if row["model"] == model
                        and int(row["batch_size"]) == batch
                        and row["routing_scope"] == "global"
                        and row["dense_fraction"] == fraction
                        and row["category"] == category
                    )
                    values.append(float(match["median_us"]))
                axis.bar(labels, values, bottom=bottoms, color=colors[category], label=category)
                bottoms += np.asarray(values)
            axis.set_title(f"{MODEL_LABELS[model]}, B={batch}")
            axis.set_ylabel("Diagnostic GPU work (us)")
            axis.tick_params(axis="x", rotation=25)
            if ri == 0 and ci == len(batches) - 1:
                axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    files = []
    for model in args.models:
        common.require_idle_gpu(args.device_index)
        worker_output = work / f"{model}.json"
        command = [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--worker-model", model,
            "--worker-output", str(worker_output),
            "--batch-sizes", ",".join(map(str, args.batch_sizes)),
            "--scopes", ",".join(args.scopes),
            "--eighths", ",".join(map(str, args.eighths)),
            "--device-index", str(args.device_index),
            "--seed", str(args.seed),
            "--warmup", str(args.warmup),
            "--trials", str(args.trials),
            "--breakdown-trials", str(args.breakdown_trials),
            "--target-interval-us", str(args.target_interval_us),
            "--max-iterations", str(args.max_iterations),
            "--eviction-mib", str(args.eviction_mib),
        ]
        subprocess.run(command, cwd=EVAL_ROOT, check=True)
        files.append(worker_output)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        raw.extend(payload["raw"])
        breakdown.extend(payload["breakdown"])
    write_csv(output / "summary.csv", rows)
    write_csv(output / "raw.csv", raw)
    write_csv(output / "breakdown.csv", breakdown)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plot_absolute_comparison(figures / "one_layer_absolute_comparison.png", rows)
    plot_speedup(figures / "one_layer_speedup.png", rows)
    plot_breakdown(figures / "one_layer_breakdown.png", breakdown)
    speedups = [float(row["speedup_vs_dense"]) for row in rows]
    pure_speedups = [float(row["pure_24_speedup_vs_dense"]) for row in rows]
    hybrid_vs_pure = [float(row["hybrid_speedup_vs_pure_24"]) for row in rows]
    analysis = {
        "cases": len(rows),
        "geomean_speedup": geometric_mean(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "pure_24_geomean_speedup_vs_dense": geometric_mean(pure_speedups),
        "hybrid_geomean_speedup_vs_pure_24": geometric_mean(hybrid_vs_pure),
        "by_scope": {
            scope: geometric_mean(
                [float(row["speedup_vs_dense"]) for row in rows if row["routing_scope"] == scope]
            )
            for scope in args.scopes
        },
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    report = [
        "# One-layer decoding: dense cuBLAS vs residual-complement",
        "",
        f"- Cases: {len(rows)}; geometric-mean speedup: {analysis['geomean_speedup']:.4f}x.",
        f"- Pure 2:4 cuSPARSELt geometric-mean speedup vs dense: "
        f"{analysis['pure_24_geomean_speedup_vs_dense']:.4f}x.",
        f"- Hybrid geometric mean relative to pure 2:4: "
        f"{analysis['hybrid_geomean_speedup_vs_pure_24']:.4f}x.",
        "- Every request contributes one current token plus seven draft tokens; M=8B.",
        "- Current token is always dense. Quota d/8 adds d-1 selected draft rows per request on average.",
        "- E2E uses concurrent cuSPARSELt base plus Split-K2 complement.",
        "- Breakdown uses explicit QK / softmax / AV and a serialized sparse diagnostic, so its categories are additive but are not the optimized E2E critical path.",
        "",
        "| Routing | Geomean speedup |",
        "|---|---:|",
    ]
    for scope, value in analysis["by_scope"].items():
        report.append(f"| {scope} | {value:.4f}x |")
    report.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `summary.csv`: all model/B/scope/fraction E2E medians and P10/P90.",
            "- `breakdown.csv`: GEMM/Attention/Softmax/Gather/Scatter/Reduce/Others.",
            "- `raw.csv`: ten E2E samples per method/case.",
            "- `figures/one_layer_absolute_comparison.png`: dense, pure 2:4, and hybrid latency.",
            "- `figures/one_layer_speedup.png` and `figures/one_layer_breakdown.png`.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(analysis, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)))
    parser.add_argument("--scopes", default=",".join(SCOPES))
    parser.add_argument("--eighths", default=",".join(map(str, EIGHTHS)))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--work-root", type=Path, default=EVAL_ROOT / "temp/decoder_layer_residual_complement")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--breakdown-trials", type=int, default=10)
    parser.add_argument("--target-interval-us", type=float, default=200_000.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--eviction-mib", type=int, default=256)
    args = parser.parse_args()
    args.models = parse_csv(args.models, MODELS, "models")
    args.batch_sizes = parse_int_csv(args.batch_sizes, BATCH_SIZES, "batch sizes")
    args.scopes = parse_csv(args.scopes, SCOPES, "scopes")
    args.eighths = parse_int_csv(args.eighths, EIGHTHS, "eighths")
    if args.worker:
        if args.worker_model is None or args.worker_output is None:
            parser.error("worker requires --worker-model and --worker-output")
    elif args.output_root is None:
        parser.error("coordinator requires --output-root")
    if args.trials != 10 or args.breakdown_trials != 10:
        parser.error("formal protocol requires ten E2E and ten breakdown trials")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        run_coordinator(parsed)
