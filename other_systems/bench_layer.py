#!/usr/bin/env python3
"""One-decoder-layer BF16 N:M benchmark for the three external systems."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "examples/evaluate/eval-guidellm/scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from other_systems import benchmark_common as common
from other_systems import parse_nm
from bench_decoder_layer_residual_complement import (
    MODEL_SPECS,
    ModelSpec,
    apply_rope,
    attention_av,
    attention_qk,
    rms_norm,
)


@dataclass
class Projection:
    dense: torch.Tensor
    prepared: common.PreparedSystem
    split_by_m: dict[int, int]


@dataclass
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


def prepare_layer(
    model: str,
    system: str,
    fmt: str,
    max_batch: int,
    device: torch.device,
    seed: int,
) -> LayerState:
    spec = MODEL_SPECS[model]
    projections: dict[str, Projection] = {}
    for name in common.PROJECTIONS:
        weight = common.make_nm_weight(
            model, name, fmt, device=device, seed=seed
        )
        projections[name] = Projection(
            dense=weight,
            prepared=common.prepare_system(system, weight, fmt),
            split_by_m={},
        )

    norm = lambda size: torch.ones(size, dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(common._stable_seed(seed, model, "layer_aux"))
    past = 127
    past_k = torch.randn(
        (max_batch, spec.kv_heads, past, spec.head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    past_v = torch.randn(
        past_k.shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    positions = torch.arange(127, 135, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        spec.rope_theta
        ** (
            torch.arange(0, spec.head_dim, 2, device=device).float()
            / spec.head_dim
        )
    )
    angles = positions[:, None] * inv_freq[None, :]
    total_keys = past + 8
    causal_mask = (
        torch.arange(total_keys, device=device)[None, :]
        > torch.arange(past, past + 8, device=device)[:, None]
    )
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
    )


def tune_layer_splits(
    state: LayerState,
    model: str,
    m: int,
    device: torch.device,
    seed: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, projection in state.projections.items():
        sample = common.make_input(model, name, m, device=device, seed=seed)
        selected, samples = common.select_split(
            projection.prepared, sample, repeats=repeats
        )
        projection.split_by_m[m] = selected.split_k
        for item in samples:
            rows.append({"projection": name, "M": m, **item})
        del sample, selected
    return rows


def make_hidden(
    model: str, batch: int, device: torch.device, seed: int
) -> torch.Tensor:
    spec = MODEL_SPECS[model]
    generator = torch.Generator(device=device)
    generator.manual_seed(common._stable_seed(seed, model, batch, "hidden"))
    return torch.randn(
        (batch * 8, spec.hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).contiguous()


def layer_forward(
    hidden: torch.Tensor,
    state: LayerState,
    *,
    external: bool,
) -> torch.Tensor:
    batch = hidden.shape[0] // 8
    m = int(hidden.shape[0])
    spec = state.spec

    def linear(name: str, x: torch.Tensor) -> torch.Tensor:
        projection = state.projections[name]
        if not external:
            return F.linear(x, projection.dense)
        return projection.prepared.with_split(
            projection.split_by_m[m]
        ).linear(x.contiguous())

    residual = hidden
    x = rms_norm(hidden, state.input_norm, spec.rms_eps)
    qkv = linear("qkv", x)
    q_size = spec.q_heads * spec.head_dim
    kv_size = spec.kv_heads * spec.head_dim
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(batch, 8, spec.q_heads, spec.head_dim)
    k = k.view(batch, 8, spec.kv_heads, spec.head_dim)
    v = v.view(batch, 8, spec.kv_heads, spec.head_dim)
    if spec.qwen_qk_norm:
        assert state.q_norm is not None and state.k_norm is not None
        q = rms_norm(q, state.q_norm, spec.rms_eps)
        k = rms_norm(k, state.k_norm, spec.rms_eps)
    q, k = apply_rope(q, k, state.rope_cos, state.rope_sin)
    scores, qg = attention_qk(
        q,
        k,
        state.past_k[:batch],
        spec.head_dim**-0.5,
        state.causal_mask,
    )
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    attn = attention_av(probs, qg, v, state.past_v[:batch])
    attn_out = linear("o", attn)
    residual = residual.add(attn_out)
    x = rms_norm(residual, state.post_norm, spec.rms_eps)
    gate_up = linear("gate_up", x)
    gate, up = gate_up.chunk(2, dim=-1)
    x = F.silu(gate).mul_(up)
    down = linear("down", x)
    return residual.add(down)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(
        len(tuple(dict.fromkeys(str(row["model"]) for row in rows))),
        len(tuple(dict.fromkeys(str(row["nm_format"]) for row in rows))),
        figsize=(10, 6),
        squeeze=False,
        sharey=True,
    )
    colors = {
        "flash_llm": "#4c78a8",
        "spinfer": "#f58518",
        "sparta": "#54a24b",
    }
    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    formats = tuple(dict.fromkeys(str(row["nm_format"]) for row in rows))
    for i, model in enumerate(models):
        for j, fmt in enumerate(formats):
            axis = axes[i][j]
            selected = [
                row
                for row in rows
                if row["model"] == model and row["nm_format"] == fmt
            ]
            labels = [
                f"B{row['batch_size']}\n{row['method']}" for row in selected
            ]
            x = list(range(len(selected)))
            axis.bar(
                x,
                [float(row["speedup_vs_dense"]) for row in selected],
                color=[colors[str(row["method"])] for row in selected],
            )
            axis.axhline(1.0, color="black", linewidth=1)
            axis.set_xticks(x, labels, rotation=90, fontsize=7)
            axis.set_title(f"{model} {fmt}")
            axis.set_ylabel("one-layer speedup vs BF16 cuBLAS")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "External N:M one-layer decoding, context=128, draft=7 "
        "(median of 10 x 1000)"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(common.DEFAULT_MODELS))
    parser.add_argument("--batch-sizes", default=",".join(map(str, common.BATCH_SIZES)))
    parser.add_argument("--formats", default=",".join(common.FORMATS))
    parser.add_argument("--systems", default=",".join(common.SYSTEMS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--split-screen-repeats", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "examples/evaluate/eval-guidellm/results/"
            "other_systems_nm_layer_8b"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        args.models = common.parse_csv(
            args.models, tuple(common.TP1_FUSED_WEIGHT_SHAPES), "models"
        )
        args.formats = tuple(
            parse_nm(value).label
            for value in args.formats.split(",")
            if value.strip()
        )
        if not args.formats:
            raise ValueError("at least one N:M format is required")
        args.systems = common.parse_csv(args.systems, common.SYSTEMS, "systems")
        args.batch_sizes = tuple(
            int(value) for value in args.batch_sizes.split(",")
        )
        if not args.batch_sizes or any(
            value not in common.BATCH_SIZES for value in args.batch_sizes
        ):
            raise ValueError(
                f"batch sizes must be selected from {common.BATCH_SIZES}"
            )
    except ValueError as error:
        parser.error(str(error))
    if args.smoke:
        args.models = args.models[:1]
        args.batch_sizes = args.batch_sizes[:1]
        args.formats = args.formats[:1]
        args.warmup = 3
        args.trials = 2
        args.replays = 5
        args.split_screen_repeats = 2
        if args.output_root == Path(
            "examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b"
        ):
            args.output_root = Path(
                "examples/evaluate/eval-guidellm/temp/other_systems_nm_layer_smoke"
            )
    elif args.trials != 10:
        parser.error("formal protocol requires exactly 10 trials")
    return args


def main() -> None:
    args = parse_args()
    common.assert_gpu_idle(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024, dtype=torch.uint8, device=device
    )
    eviction.zero_()

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    dense_cache: dict[
        tuple[str, str, int],
        tuple[common.TimingSummary, list[float]],
    ] = {}
    for model in args.models:
        for fmt in args.formats:
            for system in args.systems:
                state = prepare_layer(
                    model,
                    system,
                    fmt,
                    max(args.batch_sizes),
                    device,
                    args.seed,
                )
                for batch in args.batch_sizes:
                    m = batch * 8
                    for screen in tune_layer_splits(
                        state,
                        model,
                        m,
                        device,
                        args.seed,
                        args.split_screen_repeats,
                    ):
                        screens.append(
                            {
                                "model": model,
                                "nm_format": fmt,
                                "method": system,
                                "batch_size": batch,
                                **screen,
                            }
                        )
                    hidden = make_hidden(model, batch, device, args.seed)
                    dense_graph = common.capture(
                        lambda hidden=hidden, state=state: layer_forward(
                            hidden, state, external=False
                        ),
                        warmup=3,
                    )
                    external_graph = common.capture(
                        lambda hidden=hidden, state=state: layer_forward(
                            hidden, state, external=True
                        ),
                        warmup=3,
                    )
                    check = common.correctness(
                        external_graph.output,
                        dense_graph.output,
                        atol=0.5,
                        rtol=0.2,
                    )
                    if not check["correct"]:
                        raise RuntimeError(
                            f"layer correctness failed for {model}/{fmt}/"
                            f"{system}/B{batch}: {check}"
                        )
                    cache_key = (model, fmt, batch)
                    if cache_key not in dense_cache:
                        dense_cache[cache_key] = common.formal_measure(
                            dense_graph,
                            eviction,
                            warmup=args.warmup,
                            trials=args.trials,
                            replays=args.replays,
                        )
                    dense_summary, dense_samples = dense_cache[cache_key]
                    method_summary, method_samples = common.formal_measure(
                        external_graph,
                        eviction,
                        warmup=args.warmup,
                        trials=args.trials,
                        replays=args.replays,
                    )
                    rows.append(
                        {
                            "model": model,
                            "nm_format": fmt,
                            "method": system,
                            "batch_size": batch,
                            "M": m,
                            "draft_tokens_per_request": 7,
                            "context_length_including_current": 128,
                            "dense_cublas_median_us": dense_summary.median_us,
                            "dense_cublas_p10_us": dense_summary.p10_us,
                            "dense_cublas_p90_us": dense_summary.p90_us,
                            **method_summary.as_dict(),
                            "speedup_vs_dense": (
                                dense_summary.median_us
                                / method_summary.median_us
                            ),
                            "projection_splits": json.dumps(
                                {
                                    name: projection.split_by_m[m]
                                    for name, projection in state.projections.items()
                                },
                                sort_keys=True,
                            ),
                            **check,
                        }
                    )
                    timed_methods = [(system, method_samples)]
                    if system == args.systems[0]:
                        timed_methods.insert(0, ("dense_cublas", dense_samples))
                    for method, values in timed_methods:
                        for trial, latency in enumerate(values):
                            raw.append(
                                {
                                    "model": model,
                                    "nm_format": fmt,
                                    "method": method,
                                    "batch_size": batch,
                                    "M": m,
                                    "trial": trial,
                                    "latency_us": latency,
                                }
                            )
                    common.write_csv(output / "layer_results.csv", rows)
                    common.write_csv(output / "layer_raw_trials.csv", raw)
                    common.write_csv(output / "split_screen.csv", screens)
                    print(
                        f"[layer] {model} {fmt} {system} B={batch} "
                        f"dense={dense_summary.median_us:.3f}us "
                        f"method={method_summary.median_us:.3f}us "
                        f"speedup={dense_summary.median_us / method_summary.median_us:.4f}x",
                        flush=True,
                    )
                    del hidden, dense_graph, external_graph
                del state
                torch.cuda.empty_cache()

    common.write_csv(output / "layer_results.csv", rows)
    common.write_csv(output / "layer_raw_trials.csv", raw)
    common.write_csv(output / "split_screen.csv", screens)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    (output / "environment.json").write_text(
        json.dumps(common.environment_report(device), indent=2) + "\n"
    )
    plot(rows, output / "figures/layer_speedup.png")
    print(output)


if __name__ == "__main__":
    main()
