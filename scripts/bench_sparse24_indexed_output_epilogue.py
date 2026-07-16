#!/usr/bin/env python3
"""Benchmark indexed sparse epilogues for mixed-row QKV, O, and Down."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
    prepare_down_weight,
    route_indices,
)
from vllm.speclink_kernel import (  # noqa: E402
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_indexed_output_gemm_prepacked,
    sparse24_merge_rows_,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
DEFAULT_PROJECTIONS = ("qkv", "o", "down")


def projection_shape(model: str, projection: str) -> tuple[int, int, bool]:
    hidden = int(MODELS[model]["hidden"])
    if projection == "qkv":
        return hidden, 6144, False
    if projection == "o":
        return hidden, hidden, False
    if projection == "down":
        return int(MODELS[model]["intermediate"]), hidden, True
    raise ValueError(f"unsupported projection {projection!r}")


def make_input(
    rows: int,
    features: int,
    *,
    transposed: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    if transposed:
        tensor = torch.empty_strided(
            (rows, features),
            (1, rows),
            device="cuda",
            dtype=torch.float16,
        )
        tensor.copy_(
            torch.randn(
                tensor.shape,
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            * 0.1
        )
        return tensor
    return torch.randn(
        (rows, features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ) * 0.1


def run_projection_case(
    model: str,
    projection: str,
    batch_size: int,
    k: int,
    values: torch.Tensor,
    meta: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    input_transposed: bool,
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

    sparse_input = make_input(
        sparse_rows_padded,
        in_features,
        transposed=input_transposed,
        generator=generator,
    )
    if sparse_rows_padded != sparse_count:
        sparse_input[sparse_count:].zero_()
    dense_output = torch.randn(
        (dense_count, out_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    compact_output = torch.empty(
        (sparse_rows_padded, out_features),
        device="cuda",
        dtype=torch.float16,
    )
    compact_workspace = torch.empty(
        (out_features, sparse_rows_padded),
        device="cuda",
        dtype=torch.float16,
    )
    baseline_output = torch.empty(
        (total_rows, out_features), device="cuda", dtype=torch.float16
    )
    indexed_output = torch.empty_like(baseline_output)

    def baseline_gemm() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            sparse_input,
            values,
            meta,
            contiguous_output=True,
            input_transposed=input_transposed,
            out=compact_output,
            workspace=compact_workspace,
            device_config="auto",
        )

    def indexed_gemm() -> torch.Tensor:
        return sparse24_cutlass_indexed_output_gemm_prepacked(
            sparse_input,
            values,
            meta,
            sparse_rows,
            output_rows=total_rows,
            out=indexed_output,
            config="auto",
            input_transposed=input_transposed,
        )

    def baseline_route() -> torch.Tensor:
        baseline_gemm()
        return sparse24_merge_rows_(
            baseline_output,
            dense_output,
            compact_output[:sparse_count],
            dense_rows,
            sparse_rows,
        )

    def indexed_route() -> torch.Tensor:
        indexed_gemm()
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
            f"indexed output mismatch for {model}/{projection} "
            f"bs={batch_size} K={k}: max_abs_diff={max_abs_diff}"
        )

    baseline_ms, indexed_ms = paired_graph_median_ms(
        baseline_route,
        indexed_route,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    return {
        "model": model,
        "projection": projection,
        "batch_size": batch_size,
        "K": k,
        "total_rows": total_rows,
        "dense_rows": dense_count,
        "sparse_rows": sparse_count,
        "in_features": in_features,
        "out_features": out_features,
        "input_transposed": input_transposed,
        "baseline_route_ms": baseline_ms,
        "indexed_route_ms": indexed_ms,
        "route_speedup": baseline_ms / indexed_ms,
        "saved_intermediate_bytes": sparse_rows_padded * out_features * 4,
        "max_abs_diff": max_abs_diff,
    }


def combined_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["batch_size"]), int(row["K"]))
        grouped.setdefault(key, []).append(row)
    combined: list[dict[str, object]] = []
    for (model, batch_size, k), selected in sorted(grouped.items()):
        baseline_ms = sum(float(row["baseline_route_ms"]) for row in selected)
        indexed_ms = sum(float(row["indexed_route_ms"]) for row in selected)
        combined.append(
            {
                "model": model,
                "batch_size": batch_size,
                "K": k,
                "projection_count": len(selected),
                "projections": ",".join(str(row["projection"]) for row in selected),
                "baseline_route_ms": baseline_ms,
                "indexed_route_ms": indexed_ms,
                "route_speedup": baseline_ms / indexed_ms,
                "saved_intermediate_bytes": sum(
                    int(row["saved_intermediate_bytes"]) for row in selected
                ),
            }
        )
    return combined


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    output_root: Path,
    rows: list[dict[str, object]],
    combined: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {6: "#176B87", 8: "#B33F40", 10: "#2A9D8F"}
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    projections = list(dict.fromkeys(str(row["projection"]) for row in rows))

    figure, axes = plt.subplots(
        len(projections),
        len(models),
        figsize=(6.0 * len(models), 3.5 * len(projections)),
        squeeze=False,
    )
    for row_index, projection in enumerate(projections):
        for column_index, model in enumerate(models):
            axis = axes[row_index][column_index]
            selected = [
                row
                for row in rows
                if row["model"] == model and row["projection"] == projection
            ]
            for k in sorted({int(row["K"]) for row in selected}):
                by_k = [row for row in selected if int(row["K"]) == k]
                axis.plot(
                    [int(row["batch_size"]) for row in by_k],
                    [float(row["route_speedup"]) for row in by_k],
                    marker="o",
                    color=colors[k],
                    label=f"K={k}",
                )
            axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
            axis.set_title(f"{model} {projection}")
            axis.set_xlabel("Batch size")
            axis.set_ylabel("Down + routing speedup" if projection == "down" else "GEMM + routing speedup")
            axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "indexed_output_epilogue_by_projection.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(models), figsize=(6.0 * len(models), 4.0), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        selected = [row for row in combined if row["model"] == model]
        for k in sorted({int(row["K"]) for row in selected}):
            by_k = [row for row in selected if int(row["K"]) == k]
            axis.plot(
                [int(row["batch_size"]) for row in by_k],
                [float(row["route_speedup"]) for row in by_k],
                marker="o",
                color=colors[k],
                label=f"K={k}",
            )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model} QKV + O + Down")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Combined epilogue-stage speedup")
        axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "indexed_output_epilogue_combined.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_csv_strings, default=tuple(MODELS))
    parser.add_argument(
        "--projections", type=parse_csv_strings, default=DEFAULT_PROJECTIONS
    )
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--k-values", type=parse_csv_ints, default=DEFAULT_K_VALUES)
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
    invalid_projections = [
        projection
        for projection in args.projections
        if projection not in DEFAULT_PROJECTIONS
    ]
    if invalid_models or invalid_projections:
        raise ValueError(
            f"unsupported models={invalid_models}, projections={invalid_projections}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    results: list[dict[str, object]] = []
    for model in args.models:
        for projection in args.projections:
            in_features, out_features, input_transposed = projection_shape(
                model, projection
            )
            values, meta = prepare_down_weight(
                in_features, out_features, generator
            )
            for batch_size in args.batch_sizes:
                for k in args.k_values:
                    result = run_projection_case(
                        model,
                        projection,
                        batch_size,
                        k,
                        values,
                        meta,
                        in_features=in_features,
                        out_features=out_features,
                        input_transposed=input_transposed,
                        generator=generator,
                        unroll=args.unroll,
                        replays=args.replays,
                        trials=args.trials,
                        graph_warmup_replays=args.graph_warmup_replays,
                    )
                    results.append(result)
                    print(
                        f"{model} {projection} bs={batch_size} K={k} "
                        f"route={float(result['route_speedup']):.3f}x",
                        flush=True,
                    )
            del values, meta
            torch.cuda.empty_cache()

    combined = combined_rows(results)
    write_csv(args.output_root / "indexed_output_epilogue_benchmark.csv", results)
    write_csv(args.output_root / "indexed_output_epilogue_combined.csv", combined)
    write_plots(args.output_root, results, combined)
    print(args.output_root)


if __name__ == "__main__":
    main()
