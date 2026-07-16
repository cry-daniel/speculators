#!/usr/bin/env python3
"""Benchmark direct indexed-row stores from the sparse Down epilogue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_indexed_output_gemm_prepacked,
    sparse24_merge_rows_,
)


MODELS = {
    "qwen3_8b": {"hidden": 4096, "intermediate": 12288},
    "llama3_1_8b": {"hidden": 4096, "intermediate": 14336},
}
DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def capture_unrolled(
    fn: Callable[[], torch.Tensor],
    *,
    unroll: int,
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


def graph_sample_ms(
    graph: torch.cuda.CUDAGraph,
    *,
    unroll: int,
    replays: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (replays * unroll)


def paired_graph_median_ms(
    baseline_fn: Callable[[], torch.Tensor],
    indexed_fn: Callable[[], torch.Tensor],
    *,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> tuple[float, float]:
    baseline_graph = capture_unrolled(baseline_fn, unroll=unroll)
    indexed_graph = capture_unrolled(indexed_fn, unroll=unroll)
    for _ in range(graph_warmup_replays):
        baseline_graph.replay()
        indexed_graph.replay()
    torch.cuda.synchronize()
    baseline_samples: list[float] = []
    indexed_samples: list[float] = []
    for trial in range(trials):
        if trial % 2 == 0:
            baseline_samples.append(
                graph_sample_ms(
                    baseline_graph, unroll=unroll, replays=replays
                )
            )
            indexed_samples.append(
                graph_sample_ms(
                    indexed_graph, unroll=unroll, replays=replays
                )
            )
        else:
            indexed_samples.append(
                graph_sample_ms(
                    indexed_graph, unroll=unroll, replays=replays
                )
            )
            baseline_samples.append(
                graph_sample_ms(
                    baseline_graph, unroll=unroll, replays=replays
                )
            )
    return statistics.median(baseline_samples), statistics.median(
        indexed_samples
    )


def prepare_down_weight(
    intermediate: int,
    hidden: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (intermediate, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    values, meta = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=intermediate,
    )
    return values, meta


def route_indices(
    batch_size: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_per_request = k + 1
    indices = torch.arange(
        batch_size * rows_per_request,
        device="cuda",
        dtype=torch.int32,
    ).reshape(batch_size, rows_per_request)
    dense_rows = indices[:, :2].reshape(-1).contiguous()
    sparse_rows = indices[:, 2:].reshape(-1).contiguous()
    return dense_rows, sparse_rows


def run_case(
    model: str,
    batch_size: int,
    k: int,
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> dict[str, object]:
    total_rows = batch_size * (k + 1)
    dense_rows, sparse_rows = route_indices(batch_size, k)
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    sparse_rows_padded = (sparse_count + 7) // 8 * 8

    hidden_transposed = torch.empty_strided(
        (sparse_rows_padded, intermediate),
        (1, sparse_rows_padded),
        device="cuda",
        dtype=torch.float16,
    )
    hidden_transposed.copy_(
        torch.randn(
            hidden_transposed.shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 0.1
    )
    if sparse_rows_padded != sparse_count:
        hidden_transposed[sparse_count:].zero_()
    dense_output = torch.randn(
        (dense_count, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )

    compact_output = torch.empty(
        (sparse_rows_padded, hidden),
        device="cuda",
        dtype=torch.float16,
    )
    compact_workspace = torch.empty(
        (hidden, sparse_rows_padded),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_output = torch.empty(
        (total_rows, hidden), device="cuda", dtype=torch.float16
    )
    indexed_output = torch.empty_like(baseline_output)

    def baseline_down() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            hidden_transposed,
            values,
            meta,
            contiguous_output=True,
            input_transposed=True,
            out=compact_output,
            workspace=compact_workspace,
            device_config="auto",
        )

    def indexed_down() -> torch.Tensor:
        return sparse24_cutlass_indexed_output_gemm_prepacked(
            hidden_transposed,
            values,
            meta,
            sparse_rows,
            output_rows=total_rows,
            out=indexed_output,
            config="auto",
            input_transposed=True,
        )

    def baseline_route() -> torch.Tensor:
        baseline_down()
        return sparse24_merge_rows_(
            baseline_output,
            dense_output,
            compact_output[:sparse_count],
            dense_rows,
            sparse_rows,
        )

    def indexed_route() -> torch.Tensor:
        indexed_down()
        return sparse24_copy_indexed_rows_contiguous_(
            indexed_output,
            dense_output,
            dense_rows,
        )

    expected = baseline_route().clone()
    actual = indexed_route().clone()
    torch.cuda.synchronize()
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    if not torch.allclose(actual, expected, rtol=2e-2, atol=8e-2):
        raise RuntimeError(
            f"indexed Down mismatch for {model} bs={batch_size} K={k}: "
            f"max_abs_diff={max_abs_diff}"
        )

    baseline_down_ms, indexed_down_ms = paired_graph_median_ms(
        baseline_down,
        indexed_down,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    baseline_route_ms, indexed_route_ms = paired_graph_median_ms(
        baseline_route,
        indexed_route,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "total_rows": total_rows,
        "dense_rows": dense_count,
        "sparse_rows": sparse_count,
        "sparse_rows_padded": sparse_rows_padded,
        "baseline_down_ms": baseline_down_ms,
        "indexed_down_ms": indexed_down_ms,
        "down_speedup": baseline_down_ms / indexed_down_ms,
        "baseline_route_ms": baseline_route_ms,
        "indexed_route_ms": indexed_route_ms,
        "route_speedup": baseline_route_ms / indexed_route_ms,
        "saved_intermediate_bytes": sparse_count * hidden * 4,
        "max_abs_diff": max_abs_diff,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.2 * len(models), 4.2),
        squeeze=False,
    )
    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    for axis, model in zip(axes[0], models, strict=True):
        selected = [row for row in rows if row["model"] == model]
        for k in sorted({int(row["K"]) for row in selected}):
            by_k = [row for row in selected if int(row["K"]) == k]
            axis.plot(
                [int(row["batch_size"]) for row in by_k],
                [float(row["route_speedup"]) for row in by_k],
                marker="o",
                color=colors.get(k),
                label=f"K={k}, Down + routing",
            )
            axis.plot(
                [int(row["batch_size"]) for row in by_k],
                [float(row["down_speedup"]) for row in by_k],
                marker="s",
                linestyle="--",
                color=colors.get(k),
                alpha=0.75,
                label=f"K={k}, Down only",
            )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model} indexed sparse Down epilogue")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Speedup")
        axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=parse_csv_strings,
        default=tuple(MODELS),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_csv_ints,
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument(
        "--k-values",
        type=parse_csv_ints,
        default=DEFAULT_K_VALUES,
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=50)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_models = [model for model in args.models if model not in MODELS]
    if invalid_models:
        raise ValueError(f"unsupported models: {invalid_models}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    results: list[dict[str, object]] = []
    for model in args.models:
        shape = MODELS[model]
        values, meta = prepare_down_weight(
            shape["intermediate"], shape["hidden"], generator
        )
        for batch_size in args.batch_sizes:
            for k in args.k_values:
                result = run_case(
                    model,
                    batch_size,
                    k,
                    values,
                    meta,
                    hidden=shape["hidden"],
                    intermediate=shape["intermediate"],
                    generator=generator,
                    unroll=args.unroll,
                    replays=args.replays,
                    trials=args.trials,
                    graph_warmup_replays=args.graph_warmup_replays,
                )
                results.append(result)
                print(
                    f"{model} bs={batch_size} K={k} "
                    f"down={float(result['down_speedup']):.3f}x "
                    f"route={float(result['route_speedup']):.3f}x",
                    flush=True,
                )
        del values, meta
        torch.cuda.empty_cache()

    csv_path = args.output_root / "indexed_down_epilogue_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_path = args.output_root / "indexed_down_epilogue_speedup.png"
    write_plot(plot_path, results)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
