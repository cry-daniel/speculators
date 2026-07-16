#!/usr/bin/env python3
"""Benchmark exact row-routed Gate correction pipelined through dense Down."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    capture_unrolled,
    graph_sample_ms,
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from vllm import _custom_ops as _vllm_ops  # noqa: E402,F401
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_residual_delta_swiglu_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_,
    sparse24_merge_rows_,
    sparse24_routed_swiglu_correction_,
    sparse24_routed_swiglu_correction_gather_,
    sparse24_routed_swiglu_delta_,
)


def padded_rows(rows: int) -> int:
    return (rows + 7) // 8 * 8


def graph_median_ms(
    fn,
    *,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> float:
    graph = capture_unrolled(fn, unroll=unroll)
    for _ in range(graph_warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    return statistics.median(
        graph_sample_ms(graph, unroll=unroll, replays=replays)
        for _ in range(trials)
    )


def make_route(
    batch_size: int,
    k: int,
    *,
    dense_ratio: float,
    min_dense_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = batch_size * (k + 1)
    by_request = torch.arange(
        rows, device="cuda", dtype=torch.int32
    ).reshape(batch_size, k + 1)
    mandatory = by_request[:, :min_dense_per_request].flatten()
    scored_rows = batch_size * k
    dense_count = max(
        int(mandatory.numel()),
        int(scored_rows * dense_ratio + 0.5),
    )
    dense_count = min(scored_rows, dense_count)
    candidates = by_request[:, min_dense_per_request:k].flatten()
    extra_count = dense_count - int(mandatory.numel())
    permutation = torch.randperm(
        int(candidates.numel()), device="cuda", generator=generator
    )
    dense_rows = torch.cat(
        (mandatory, candidates[permutation[:extra_count]])
    ).sort().values.contiguous()
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows.long()] = torch.arange(
        dense_rows.numel(), device="cuda", dtype=torch.int32
    )
    return dense_rows, dense_slots


def silu_and_mul(out: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
    torch.ops._C.silu_and_mul(out, gate_up)
    return out


def prepare_gate_weight(
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    dense = torch.randn(
        (hidden, 2 * intermediate),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense.mul_(0.02)
    dense.masked_fill_(dense == 0, torch.finfo(dense.dtype).tiny)
    sparse, _ = apply_random_24_mask(dense, generator=generator)

    def prepack(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_24(matrix, layout="n_major")
        return prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=hidden,
        )

    full_values, full_meta = prepack(sparse)
    residual_values, residual_meta = prepack(dense - sparse)
    return dense, sparse, full_values, full_meta, residual_values, residual_meta


def run_case(
    model: str,
    batch_size: int,
    k: int,
    *,
    dense_ratio: float,
    min_dense_per_request: int,
    gate_config: str,
    residual_gate_config: str,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
    breakdown: bool,
) -> dict[str, object]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    rows = batch_size * (k + 1)
    dense_rows, dense_slots = make_route(
        batch_size,
        k,
        dense_ratio=dense_ratio,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
    )
    dense_count = int(dense_rows.numel())
    sparse_mask = torch.ones(rows, device="cuda", dtype=torch.bool)
    sparse_mask[dense_rows.long()] = False
    sparse_rows = (
        sparse_mask.nonzero().flatten().to(torch.int32).contiguous()
    )
    sparse_count = int(sparse_rows.numel())
    dense_run = padded_rows(dense_count)
    x = torch.randn(
        (rows, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)
    gate = prepare_gate_weight(hidden, intermediate, generator)
    down_weight = torch.randn(
        (intermediate, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    down_weight.mul_(0.02)

    dense_gate_up = torch.empty(
        (rows, 2 * intermediate), device="cuda", dtype=torch.float16
    )
    dense_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    dense_output = torch.empty(
        (rows, hidden), device="cuda", dtype=torch.float16
    )

    dense_x = torch.zeros(
        (dense_run, hidden), device="cuda", dtype=torch.float16
    )
    gate_residual = torch.empty(
        (dense_run, 2 * intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    pipeline_hidden = torch.empty_like(baseline_hidden)
    baseline_dense_base = torch.empty(
        (dense_count, 2 * intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    pipeline_dense_base = torch.empty_like(baseline_dense_base)
    dense_delta = torch.empty(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    baseline_output = torch.empty(
        (rows, hidden), device="cuda", dtype=torch.float16
    )
    pipeline_output = torch.empty_like(baseline_output)
    fused_hidden = torch.empty_like(baseline_hidden)
    fused_dense_base = torch.empty_like(baseline_dense_base)
    fused_dense_delta = torch.empty_like(dense_delta)
    fused_output = torch.empty_like(baseline_output)
    delta_output = torch.empty(
        (dense_run, hidden), device="cuda", dtype=torch.float16
    )
    fused_delta_output = torch.empty_like(delta_output)
    partition_hidden = torch.empty_like(baseline_hidden)
    partition_dense_base = torch.empty_like(baseline_dense_base)
    partition_dense_hidden = torch.empty(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    partition_sparse_hidden = torch.empty(
        (sparse_count, intermediate), device="cuda", dtype=torch.float16
    )
    partition_dense_output = torch.empty(
        (dense_count, hidden), device="cuda", dtype=torch.float16
    )
    partition_sparse_output = torch.empty(
        (sparse_count, hidden), device="cuda", dtype=torch.float16
    )
    partition_output = torch.empty_like(baseline_output)

    full_stream = torch.cuda.Stream()
    residual_stream = torch.cuda.Stream()
    gate_done = torch.cuda.Event()

    def dense_mlp() -> torch.Tensor:
        torch.mm(x, gate[0], out=dense_gate_up)
        silu_and_mul(dense_hidden, dense_gate_up)
        return torch.mm(dense_hidden, down_weight, out=dense_output)

    def residual_gate() -> torch.Tensor:
        sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
        return sparse24_cutlass_inline_transpose_gemm_prepacked(
            dense_x,
            gate[4],
            gate[5],
            out=gate_residual,
            config=residual_gate_config,
            store_mode="vector",
        )

    def baseline() -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slots,
                dense_count=dense_count,
                out=baseline_hidden,
                dense_base=baseline_dense_base,
                config=gate_config,
            )
        with torch.cuda.stream(residual_stream):
            residual_gate()
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        sparse24_routed_swiglu_correction_(
            baseline_dense_base,
            gate_residual[:dense_count],
            dense_rows,
            baseline_hidden,
        )
        return torch.mm(baseline_hidden, down_weight, out=baseline_output)

    def pipeline() -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slots,
                dense_count=dense_count,
                out=pipeline_hidden,
                dense_base=pipeline_dense_base,
                config=gate_config,
                write_dense_approx=True,
            )
            gate_done.record(full_stream)
            torch.mm(pipeline_hidden, down_weight, out=pipeline_output)
        with torch.cuda.stream(residual_stream):
            residual_gate()
            residual_stream.wait_event(gate_done)
            sparse24_routed_swiglu_delta_(
                pipeline_dense_base, gate_residual, dense_delta
            )
            torch.mm(dense_delta, down_weight, out=delta_output)
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return sparse24_add_indexed_rows_contiguous_(
            pipeline_output,
            delta_output[:dense_count],
            dense_rows,
        )

    def fused_pipeline() -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slots,
                dense_count=dense_count,
                out=fused_hidden,
                dense_base=fused_dense_base,
                config=gate_config,
                write_dense_approx=True,
            )
            gate_done.record(full_stream)
            torch.mm(fused_hidden, down_weight, out=fused_output)
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            residual_stream.wait_event(gate_done)
            sparse24_cutlass_residual_delta_swiglu_prepacked(
                dense_x,
                gate[4],
                gate[5],
                fused_dense_base,
                fused_dense_delta,
                config=residual_gate_config,
            )
            torch.mm(
                fused_dense_delta,
                down_weight,
                out=fused_delta_output,
            )
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return sparse24_add_indexed_rows_contiguous_(
            fused_output,
            fused_delta_output[:dense_count],
            dense_rows,
        )

    def partition_down() -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slots,
                dense_count=dense_count,
                out=partition_hidden,
                dense_base=partition_dense_base,
                config=gate_config,
                write_dense_approx=True,
            )
        with torch.cuda.stream(residual_stream):
            residual_gate()
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        sparse24_routed_swiglu_correction_gather_(
            partition_dense_base,
            gate_residual[:dense_count],
            dense_rows,
            partition_hidden,
            partition_dense_hidden,
        )
        sparse24_gather_rows_(
            partition_hidden,
            sparse_rows,
            partition_sparse_hidden,
        )
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            torch.mm(
                partition_sparse_hidden,
                down_weight,
                out=partition_sparse_output,
            )
        with torch.cuda.stream(residual_stream):
            torch.mm(
                partition_dense_hidden[:dense_count],
                down_weight,
                out=partition_dense_output,
            )
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return sparse24_merge_rows_(
            partition_output,
            partition_dense_output,
            partition_sparse_output,
            dense_rows,
            sparse_rows,
        )

    baseline_value = baseline().clone()
    pipeline_value = pipeline().clone()
    fused_value = fused_pipeline().clone()
    partition_value = partition_down().clone()
    torch.cuda.synchronize()
    max_abs_diff = float(
        (baseline_value.float() - pipeline_value.float()).abs().max().item()
    )
    if not torch.allclose(
        baseline_value, pipeline_value, rtol=6e-2, atol=4e-1
    ):
        raise RuntimeError(f"pipelined MLP mismatch: max_abs_diff={max_abs_diff}")
    fused_max_abs_diff = float(
        (baseline_value.float() - fused_value.float()).abs().max().item()
    )
    if not torch.allclose(
        baseline_value, fused_value, rtol=6e-2, atol=4e-1
    ):
        raise RuntimeError(
            f"fused pipelined MLP mismatch: max_abs_diff={fused_max_abs_diff}"
        )
    partition_max_abs_diff = float(
        (baseline_value.float() - partition_value.float()).abs().max().item()
    )
    if not torch.allclose(
        baseline_value, partition_value, rtol=6e-2, atol=4e-1
    ):
        raise RuntimeError(
            "partitioned Down MLP mismatch: "
            f"max_abs_diff={partition_max_abs_diff}"
        )
    baseline_ms, pipeline_ms = paired_graph_median_ms(
        baseline,
        pipeline,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    fused_baseline_ms, fused_pipeline_ms = paired_graph_median_ms(
        baseline,
        fused_pipeline,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    partition_baseline_ms, partition_down_ms = paired_graph_median_ms(
        baseline,
        partition_down,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    dense_exact_ms, exact_vs_dense_ms = paired_graph_median_ms(
        dense_mlp,
        baseline,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    dense_fused_ms, fused_vs_dense_ms = paired_graph_median_ms(
        dense_mlp,
        fused_pipeline,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    dense_partition_ms, partition_vs_dense_ms = paired_graph_median_ms(
        dense_mlp,
        partition_down,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    result: dict[str, object] = {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": dense_count,
        "dense_ratio": dense_ratio,
        "gate_config": gate_config,
        "residual_gate_config": residual_gate_config,
        "baseline_ms": baseline_ms,
        "pipeline_ms": pipeline_ms,
        "speedup": baseline_ms / pipeline_ms,
        "max_abs_diff": max_abs_diff,
        "fused_baseline_ms": fused_baseline_ms,
        "fused_pipeline_ms": fused_pipeline_ms,
        "fused_speedup": fused_baseline_ms / fused_pipeline_ms,
        "fused_max_abs_diff": fused_max_abs_diff,
        "partition_baseline_ms": partition_baseline_ms,
        "partition_down_ms": partition_down_ms,
        "partition_speedup": partition_baseline_ms / partition_down_ms,
        "partition_max_abs_diff": partition_max_abs_diff,
        "dense_exact_pair_ms": dense_exact_ms,
        "exact_vs_dense_pair_ms": exact_vs_dense_ms,
        "exact_speedup_vs_dense": dense_exact_ms / exact_vs_dense_ms,
        "dense_fused_pair_ms": dense_fused_ms,
        "fused_vs_dense_pair_ms": fused_vs_dense_ms,
        "fused_speedup_vs_dense": dense_fused_ms / fused_vs_dense_ms,
        "dense_partition_pair_ms": dense_partition_ms,
        "partition_vs_dense_pair_ms": partition_vs_dense_ms,
        "partition_speedup_vs_dense": (
            dense_partition_ms / partition_vs_dense_ms
        ),
    }
    if breakdown:
        def full_gate_only() -> torch.Tensor:
            return sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slots,
                dense_count=dense_count,
                out=pipeline_hidden,
                dense_base=pipeline_dense_base,
                config=gate_config,
                write_dense_approx=True,
            )[0]

        def correction_only() -> torch.Tensor:
            return sparse24_routed_swiglu_correction_(
                baseline_dense_base,
                gate_residual[:dense_count],
                dense_rows,
                baseline_hidden,
            )

        def delta_only() -> torch.Tensor:
            return sparse24_routed_swiglu_delta_(
                pipeline_dense_base,
                gate_residual,
                dense_delta,
            )

        def full_down_only() -> torch.Tensor:
            return torch.mm(
                pipeline_hidden,
                down_weight,
                out=pipeline_output,
            )

        def compact_down_only() -> torch.Tensor:
            return torch.mm(
                dense_delta,
                down_weight,
                out=delta_output,
            )

        def indexed_add_only() -> torch.Tensor:
            return sparse24_add_indexed_rows_contiguous_(
                pipeline_output,
                delta_output[:dense_count],
                dense_rows,
            )

        components = {
            "full_gate_ms": full_gate_only,
            "residual_gate_ms": residual_gate,
            "correction_ms": correction_only,
            "delta_ms": delta_only,
            "full_down_ms": full_down_only,
            "compact_down_ms": compact_down_only,
            "indexed_add_ms": indexed_add_only,
        }
        for name, component in components.items():
            result[name] = graph_median_ms(
                component,
                unroll=unroll,
                replays=replays,
                trials=trials,
                graph_warmup_replays=graph_warmup_replays,
            )
    return result


def write_plot(output_root: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"{row['model']}\nbs{row['batch_size']}/K{row['K']}" for row in rows
    ]
    figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(rows)), 4.5))
    positions = list(range(len(rows)))
    axis.bar(
        [position - 0.28 for position in positions],
        [float(row["speedup"]) for row in rows],
        width=0.28,
        color="#176B87",
        label="Separate delta kernel",
    )
    axis.bar(
        positions,
        [float(row["fused_speedup"]) for row in rows],
        width=0.28,
        color="#D1495B",
        label="Residual delta epilogue",
    )
    axis.bar(
        [position + 0.28 for position in positions],
        [float(row["partition_speedup"]) for row in rows],
        width=0.28,
        color="#4F772D",
        label="Partitioned dense Down",
    )
    axis.axhline(1.0, color="#222222", linewidth=1)
    axis.set_ylabel("Speedup vs correction-then-Down")
    axis.set_xticks(positions, labels)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "routed_mlp_pipeline.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--batch-sizes", default="16,32,64")
    parser.add_argument("--k-values", default="6,8,10")
    parser.add_argument("--dense-ratio", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument("--gate-config", default="256x64x64_s3_sw4")
    parser.add_argument("--residual-gate-config", default="auto")
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=5)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    models = parse_csv_strings(args.models)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    k_values = parse_csv_ints(args.k_values)
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results: list[dict[str, object]] = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r}")
        for batch_size in batch_sizes:
            for k in k_values:
                row = run_case(
                    model,
                    batch_size,
                    k,
                    dense_ratio=args.dense_ratio,
                    min_dense_per_request=args.min_dense_per_request,
                    gate_config=args.gate_config,
                    residual_gate_config=args.residual_gate_config,
                    generator=generator,
                    unroll=args.unroll,
                    replays=args.replays,
                    trials=args.trials,
                    graph_warmup_replays=args.graph_warmup_replays,
                    breakdown=args.breakdown,
                )
                results.append(row)
                print(row, flush=True)

    csv_path = args.output_root / "routed_mlp_pipeline.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root, results)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
