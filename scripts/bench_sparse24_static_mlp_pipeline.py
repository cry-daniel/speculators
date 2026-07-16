#!/usr/bin/env python3
"""Benchmark the static 2:4 Gate+SwiGLU -> Down layout pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm import _custom_ops as _vllm_custom_ops  # noqa: E402,F401
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
)


MODELS = {
    "qwen3_8b": {"hidden": 4096, "intermediate": 12288},
    "llama3_1_8b": {"hidden": 4096, "intermediate": 14336},
}
DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def _capture_unrolled(
    fn: Callable[[], torch.Tensor], unroll: int
) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            captured = fn()
    torch.cuda.synchronize()
    del captured
    return graph


def _sample_ms(
    graph: torch.cuda.CUDAGraph, *, unroll: int, replays: int
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (unroll * replays)


def _interleaved_medians(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    unroll: int,
    replays: int,
    trials: int,
) -> dict[str, float]:
    graphs = {
        name: _capture_unrolled(fn, unroll) for name, fn in functions.items()
    }
    for _ in range(3):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    names = tuple(functions)
    for trial in range(trials):
        order = names if trial % 2 == 0 else tuple(reversed(names))
        for name in order:
            samples[name].append(
                _sample_ms(graphs[name], unroll=unroll, replays=replays)
            )
    return {name: statistics.median(values) for name, values in samples.items()}


def _prepare_weight(
    in_features: int,
    out_features: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (in_features, out_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    sparse_weight, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(sparse_weight, layout="n_major")
    values, meta = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=in_features,
    )
    del sparse_weight, packed
    return weight, values, meta


def _run_case(
    *,
    model: str,
    batch_size: int,
    k: int,
    gate_weight: torch.Tensor,
    gate_values: torch.Tensor,
    gate_meta: torch.Tensor,
    down_weight: torch.Tensor,
    down_values: torch.Tensor,
    down_meta: torch.Tensor,
    generator: torch.Generator,
    gate_config: str,
    down_config: str,
    unroll: int,
    replays: int,
    trials: int,
) -> dict[str, object]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    rows = batch_size * (k + 1)
    if rows % 8:
        raise ValueError(f"rows must be divisible by 8, got {rows}")
    x = torch.randn(
        (rows, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)

    dense_gate = torch.empty(
        (rows, 2 * intermediate), device="cuda", dtype=torch.float16
    )
    dense_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    dense_output = torch.empty((rows, hidden), device="cuda", dtype=torch.float16)
    contiguous_hidden = torch.empty_like(dense_hidden)
    contiguous_output = torch.empty_like(dense_output)
    contiguous_workspace = torch.empty(
        (hidden, rows), device="cuda", dtype=torch.float16
    )
    transposed_hidden = torch.empty_strided(
        (rows, intermediate),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )
    transposed_output = torch.empty_like(dense_output)
    transposed_workspace = torch.empty_like(contiguous_workspace)

    def dense() -> torch.Tensor:
        torch.mm(x, gate_weight, out=dense_gate)
        torch.ops._C.silu_and_mul(dense_hidden, dense_gate)
        return torch.mm(dense_hidden, down_weight, out=dense_output)

    def contiguous() -> torch.Tensor:
        sparse24_cutlass_gate_up_swiglu_prepacked(
            x,
            gate_values,
            gate_meta,
            out=contiguous_hidden,
            config="auto",
            output_transposed=False,
        )
        return sparse24_cutlass_device_gemm_prepacked(
            contiguous_hidden,
            down_values,
            down_meta,
            contiguous_output=True,
            out=contiguous_output,
            workspace=contiguous_workspace,
            device_config="auto",
        )

    def transposed() -> torch.Tensor:
        sparse24_cutlass_gate_up_swiglu_prepacked(
            x,
            gate_values,
            gate_meta,
            out=transposed_hidden,
            config=gate_config,
            output_transposed=True,
        )
        return sparse24_cutlass_device_gemm_prepacked(
            transposed_hidden,
            down_values,
            down_meta,
            contiguous_output=True,
            input_transposed=True,
            out=transposed_output,
            workspace=transposed_workspace,
            device_config=down_config,
        )

    expected = contiguous().clone()
    actual = transposed().clone()
    torch.cuda.synchronize()
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    if not torch.allclose(actual, expected, rtol=2e-3, atol=2e-3):
        raise RuntimeError(
            f"layout pipeline mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={max_abs_diff}"
        )

    timings = _interleaved_medians(
        {"dense": dense, "contiguous": contiguous, "transposed": transposed},
        unroll=unroll,
        replays=replays,
        trials=trials,
    )
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "gate_config": gate_config,
        "down_config": down_config,
        "dense_ms": timings["dense"],
        "contiguous_ms": timings["contiguous"],
        "transposed_ms": timings["transposed"],
        "contiguous_speedup_vs_dense": timings["dense"] / timings["contiguous"],
        "transposed_speedup_vs_dense": timings["dense"] / timings["transposed"],
        "transposed_speedup_vs_contiguous": (
            timings["contiguous"] / timings["transposed"]
        ),
        "max_abs_diff": max_abs_diff,
    }


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    figure, axes = plt.subplots(
        len(models), 2, figsize=(11.2, 3.9 * len(models)), squeeze=False
    )
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    for model_index, model in enumerate(models):
        selected_model = [row for row in rows if row["model"] == model]
        variants = sorted(
            {
                (int(row["K"]), str(row["gate_config"]), str(row["down_config"]))
                for row in selected_model
            }
        )
        for k, gate_config, down_config in variants:
            selected = sorted(
                (
                    row
                    for row in selected_model
                    if int(row["K"]) == k
                    and row["gate_config"] == gate_config
                    and row["down_config"] == down_config
                ),
                key=lambda row: int(row["batch_size"]),
            )
            x = [int(row["batch_size"]) for row in selected]
            label = f"K={k} {gate_config}/{down_config}"
            axes[model_index][0].plot(
                x,
                [float(row["transposed_speedup_vs_contiguous"]) for row in selected],
                marker="o",
                color=colors.get(k),
                label=label,
            )
            axes[model_index][1].plot(
                x,
                [float(row["transposed_speedup_vs_dense"]) for row in selected],
                marker="s",
                color=colors.get(k),
                label=label,
            )
        for axis in axes[model_index]:
            axis.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
            axis.set_xlabel("Batch size")
            axis.set_xticks(sorted({int(row["batch_size"]) for row in selected_model}))
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
        axes[model_index][0].set_title(f"{model}: layout pipeline gain")
        axes[model_index][0].set_ylabel("Transposed / contiguous speedup")
        axes[model_index][1].set_title(f"{model}: sparse pipeline vs dense MLP")
        axes[model_index][1].set_ylabel("Dense / transposed latency")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(rows: list[dict[str, object]], output: Path) -> None:
    gains = [float(row["transposed_speedup_vs_contiguous"]) for row in rows]
    sparse_gains = [float(row["transposed_speedup_vs_dense"]) for row in rows]
    lines = [
        "# Static Sparse MLP Layout Pipeline",
        "",
        f"- points: {len(rows)}",
        f"- transposed/contiguous median: {statistics.median(gains):.4f}x",
        f"- transposed/contiguous minimum: {min(gains):.4f}x",
        f"- sparse/dense median: {statistics.median(sparse_gains):.4f}x",
        f"- maximum absolute difference: {max(float(row['max_abs_diff']) for row in rows):.6f}",
        "",
        "The transposed path fuses SwiGLU into the sparse Gate/Up epilogue and",
        "keeps the resulting activation in the layout consumed directly by sparse Down.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=_csv_strings, default=("qwen3_8b",))
    parser.add_argument("--batch-sizes", type=_csv_ints, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--k-values", type=_csv_ints, default=DEFAULT_K_VALUES)
    parser.add_argument("--gate-configs", type=_csv_strings, default=("auto",))
    parser.add_argument("--down-configs", type=_csv_strings, default=("auto",))
    parser.add_argument("--accumulator", default="fp16_qkv_gate")
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_static_mlp_pipeline_{datetime.now():%Y%m%d_%H%M%S}",
    )
    args = parser.parse_args()
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f"unsupported models: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["SPECLINK_SPARSE24_ACCUMULATOR"] = args.accumulator
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(args.models):
        shape = MODELS[model]
        hidden = int(shape["hidden"])
        intermediate = int(shape["intermediate"])
        generator = torch.Generator(device="cuda").manual_seed(
            args.seed + model_index
        )
        gate_weight, gate_values, gate_meta = _prepare_weight(
            hidden, 2 * intermediate, generator
        )
        down_weight, down_values, down_meta = _prepare_weight(
            intermediate, hidden, generator
        )
        for batch_size in args.batch_sizes:
            for k in args.k_values:
                for gate_config in args.gate_configs:
                    for down_config in args.down_configs:
                        row = _run_case(
                            model=model,
                            batch_size=batch_size,
                            k=k,
                            gate_weight=gate_weight,
                            gate_values=gate_values,
                            gate_meta=gate_meta,
                            down_weight=down_weight,
                            down_values=down_values,
                            down_meta=down_meta,
                            generator=generator,
                            gate_config=gate_config,
                            down_config=down_config,
                            unroll=args.unroll,
                            replays=args.replays,
                            trials=args.trials,
                        )
                        rows.append(row)
                        print(
                            f"{model} bs={batch_size} K={k} rows={row['rows']} "
                            f"tiles={gate_config}/{down_config} "
                            f"dense={float(row['dense_ms']):.4f} ms "
                            f"contiguous={float(row['contiguous_ms']):.4f} ms "
                            f"transposed={float(row['transposed_ms']):.4f} ms "
                            f"layout_gain={float(row['transposed_speedup_vs_contiguous']):.3f}x",
                            flush=True,
                        )
        del gate_weight, gate_values, gate_meta
        del down_weight, down_values, down_meta
        torch.cuda.empty_cache()

    csv_path = args.output_root / "static_mlp_pipeline.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output_root / "static_mlp_pipeline.png")
    _write_report(rows, args.output_root / "report.md")
    print(csv_path)


if __name__ == "__main__":
    main()
