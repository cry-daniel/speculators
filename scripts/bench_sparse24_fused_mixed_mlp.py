#!/usr/bin/env python3
"""Benchmark one-launch exact mixed-row Gate/SwiGLU and Down."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
from typing import Callable

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    paired_graph_median_ms,
    parse_csv_ints,
)
from bench_sparse24_paired_residual import prepare_exact_weights  # noqa: E402
from bench_sparse24_routed_swiglu import prepare_gate_up_weights  # noqa: E402
from vllm.speclink_kernel import (  # noqa: E402
    sparse24_cutlass_fused_mixed_mlp_prepacked,
    sparse24_cutlass_paired_fused_routed_swiglu_prepacked,
    sparse24_cutlass_paired_inplace_residual_prepacked,
)


def paired_eager_median_ms(
    control: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    unroll: int,
    trials: int,
) -> tuple[float, float]:
    for _ in range(3):
        control()
        candidate()
    torch.cuda.synchronize()

    def sample(fn: Callable[[], torch.Tensor]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(unroll):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / unroll

    control_samples: list[float] = []
    candidate_samples: list[float] = []
    for trial in range(trials):
        if trial % 2:
            candidate_samples.append(sample(candidate))
            control_samples.append(sample(control))
        else:
            control_samples.append(sample(control))
            candidate_samples.append(sample(candidate))
    return statistics.median(control_samples), statistics.median(
        candidate_samples
    )


def confidence_route(
    rows: int,
    dense_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    confidence = torch.rand(rows, device="cuda", generator=generator)
    dense_rows = torch.topk(
        confidence, k=dense_count, largest=False, sorted=False
    ).indices
    dense_rows = dense_rows.sort().values.to(torch.int32).contiguous()
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows.long()] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    return dense_rows, dense_slots


def run_case(
    *,
    model: str,
    batch_size: int,
    k: int,
    dense_count: int,
    worker_blocks: int,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> dict[str, object]:
    model_width = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    rows = batch_size * (k + 1)
    if not 0 < dense_count < rows:
        raise ValueError(f"dense_count must be in [1, {rows - 1}]")

    gate_weights = prepare_gate_up_weights(
        model_width, intermediate, generator
    )
    (
        _gate_dense_weight,
        _gate_full_values,
        _gate_full_meta,
        gate_routed_values,
        gate_routed_meta,
        _gate_residual_values,
        _gate_residual_meta,
        gate_residual_routed_values,
        gate_residual_routed_meta,
        _gate_residual_fp8,
        _gate_residual_fp8_scale,
    ) = gate_weights
    (
        _down_dense_weight,
        down_full_values,
        down_full_meta,
        down_residual_values,
        down_residual_meta,
        _down_pair_values,
        _down_pair_meta,
    ) = prepare_exact_weights(intermediate, model_width, generator)

    dense_rows, dense_slots = confidence_route(
        rows, dense_count, generator
    )
    x = torch.randn(
        (rows, model_width),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)

    control_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    control_gate_base = torch.empty(
        (dense_count, 2 * intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    control_output = torch.empty(
        (rows, model_width), device="cuda", dtype=torch.float16
    )
    control_gate_counters = torch.zeros(
        2 * intermediate // 256, device="cuda", dtype=torch.int32
    )
    control_down_counters = torch.zeros(
        model_width // 256, device="cuda", dtype=torch.int32
    )

    candidate_hidden = torch.empty_like(control_hidden)
    candidate_gate_base = torch.empty_like(control_gate_base)
    candidate_output = torch.empty_like(control_output)
    candidate_gate_counters = torch.zeros_like(control_gate_counters)
    candidate_down_counters = torch.zeros_like(control_down_counters)
    candidate_barrier = torch.zeros(2, device="cuda", dtype=torch.int32)

    def control() -> torch.Tensor:
        sparse24_cutlass_paired_fused_routed_swiglu_prepacked(
            x,
            gate_routed_values,
            gate_routed_meta,
            gate_residual_routed_values,
            gate_residual_routed_meta,
            dense_rows,
            dense_slots,
            out=control_hidden,
            dense_base=control_gate_base,
            feature_counters=control_gate_counters,
            config="256x64_full_256x32_residual_s3_sw4",
            worker_blocks=worker_blocks,
        )
        return sparse24_cutlass_paired_inplace_residual_prepacked(
            control_hidden,
            down_full_values,
            down_full_meta,
            down_residual_values,
            down_residual_meta,
            dense_rows,
            out=control_output,
            feature_counters=control_down_counters,
            config="256x64_full_256x32_residual_inplace",
            worker_blocks=worker_blocks,
        )

    def candidate() -> torch.Tensor:
        sparse24_cutlass_fused_mixed_mlp_prepacked(
            x,
            gate_routed_values,
            gate_routed_meta,
            gate_residual_routed_values,
            gate_residual_routed_meta,
            down_full_values,
            down_full_meta,
            down_residual_values,
            down_residual_meta,
            dense_rows,
            dense_slots,
            hidden=candidate_hidden,
            gate_dense_base=candidate_gate_base,
            out=candidate_output,
            gate_feature_counters=candidate_gate_counters,
            down_feature_counters=candidate_down_counters,
            grid_barrier=candidate_barrier,
            worker_blocks=worker_blocks,
        )
        return candidate_output

    control_actual = control().clone()
    candidate_actual = candidate().clone()
    torch.cuda.synchronize()
    output_max_abs_diff = float(
        (control_actual.float() - candidate_actual.float()).abs().max().item()
    )
    hidden_max_abs_diff = float(
        (control_hidden.float() - candidate_hidden.float()).abs().max().item()
    )
    if not torch.allclose(
        control_actual, candidate_actual, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"final output mismatch: max_abs_diff={output_max_abs_diff}"
        )
    if not torch.allclose(
        control_hidden, candidate_hidden, rtol=3e-2, atol=1e-1
    ):
        raise RuntimeError(
            f"hidden output mismatch: max_abs_diff={hidden_max_abs_diff}"
        )

    control_ms, candidate_ms = paired_graph_median_ms(
        control,
        candidate,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    eager_control_ms, eager_candidate_ms = paired_eager_median_ms(
        control,
        candidate,
        unroll=unroll,
        trials=trials,
    )
    candidate()
    torch.cuda.synchronize()

    control_gate_counter_max = int(control_gate_counters.abs().max().item())
    control_down_counter_max = int(control_down_counters.abs().max().item())
    candidate_gate_counter_max = int(
        candidate_gate_counters.abs().max().item()
    )
    candidate_down_counter_max = int(
        candidate_down_counters.abs().max().item()
    )
    barrier_arrivals = int(candidate_barrier[0].item())
    barrier_sense = int(candidate_barrier[1].item())
    state_values = (
        control_gate_counter_max,
        control_down_counter_max,
        candidate_gate_counter_max,
        candidate_down_counter_max,
        barrier_arrivals,
    )
    if any(state_values):
        raise RuntimeError(f"persistent replay state was not reset: {state_values}")

    return {
        "model": model,
        "batch_size": batch_size,
        "k": k,
        "rows": rows,
        "dense_count": dense_count,
        "worker_blocks": worker_blocks,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "speedup": control_ms / candidate_ms,
        "eager_control_ms": eager_control_ms,
        "eager_candidate_ms": eager_candidate_ms,
        "eager_speedup": eager_control_ms / eager_candidate_ms,
        "output_max_abs_diff": output_max_abs_diff,
        "hidden_max_abs_diff": hidden_max_abs_diff,
        "control_gate_counter_max": control_gate_counter_max,
        "control_down_counter_max": control_down_counter_max,
        "candidate_gate_counter_max": candidate_gate_counter_max,
        "candidate_down_counter_max": candidate_down_counter_max,
        "barrier_arrivals": barrier_arrivals,
        "barrier_sense": barrier_sense,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [f"workers={int(row['worker_blocks'])}" for row in rows]
    graph_speedups = [float(row["speedup"]) for row in rows]
    eager_speedups = [float(row["eager_speedup"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = list(range(len(labels)))
    width = 0.38
    axis.bar(
        [position - width / 2 for position in positions],
        graph_speedups,
        width=width,
        color="#2b6f8a",
        label="CUDA Graph",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        eager_speedups,
        width=width,
        color="#d17a38",
        label="Eager",
    )
    axis.set_xticks(positions, labels)
    axis.axhline(1.0, color="#333333", linewidth=1.0, label="two-launch control")
    axis.set_ylabel("Speedup")
    axis.set_title("Exact mixed-row Gate/SwiGLU + Down")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen3_8b")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dense-count", type=int, default=32)
    parser.add_argument("--worker-blocks", default="0,120,144,160")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=3)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "examples/evaluate/eval-guidellm/temp/"
            "sparse24_fused_mixed_mlp_qwen_bs32_k10"
        ),
    )
    args = parser.parse_args()

    if args.batch_size < 16:
        raise ValueError("this fused MLP prototype is restricted to batch >= 16")
    workers = parse_csv_ints(args.worker_blocks)
    if not workers:
        raise ValueError("worker_blocks cannot be empty")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for index, worker_blocks in enumerate(workers):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + index)
        result = run_case(
            model=args.model,
            batch_size=args.batch_size,
            k=args.k,
            dense_count=args.dense_count,
            worker_blocks=worker_blocks,
            generator=generator,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        results.append(result)
        print(
            f"workers={worker_blocks}: control={float(result['control_ms']):.5f} ms "
            f"candidate={float(result['candidate_ms']):.5f} ms "
            f"speedup={float(result['speedup']):.4f}x "
            f"eager={float(result['eager_speedup']):.4f}x "
            f"diff={float(result['output_max_abs_diff']):.5f}",
            flush=True,
        )

    csv_path = args.output_root / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "speedup.png", results)


if __name__ == "__main__":
    main()
