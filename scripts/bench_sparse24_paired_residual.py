#!/usr/bin/env python3
"""Benchmark exact paired W24/residual epilogues for mixed-row routing."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    assert_24_weight,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    prepare_cutlass_sparse24_pair_add,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_indexed_output_gemm_prepacked,
    sparse24_cutlass_pair_add_indexed_prepacked,
    sparse24_gather_rows_,
    sparse24_partition_rows_,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)
DEFAULT_PROJECTIONS = ("qkv", "o", "gate_up", "down")


def projection_shape(model: str, projection: str) -> tuple[int, int]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    if projection == "qkv":
        return hidden, 6144
    if projection == "o":
        return hidden, hidden
    if projection == "gate_up":
        return hidden, 2 * intermediate
    if projection == "down":
        return intermediate, hidden
    raise ValueError(f"unsupported projection {projection!r}")


def padded_rows(rows: int) -> int:
    return (rows + 7) // 8 * 8


def route_indices(
    batch_size: int,
    k: int,
    *,
    dense_fraction: float,
    min_dense_per_request: int,
    generator: torch.Generator,
    route_mode: str = "ratio_total",
    dense_cap: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = batch_size * (k + 1)
    mandatory_count = batch_size * min_dense_per_request
    if route_mode == "ratio_total":
        dense_count = max(
            mandatory_count,
            math.ceil(total_rows * dense_fraction),
        )
    elif route_mode == "bonus_dense":
        dense_count = mandatory_count + math.ceil(
            batch_size * k * dense_fraction
        )
    elif route_mode == "draft_ratio_cap":
        dense_count = max(
            mandatory_count,
            int(batch_size * k * dense_fraction + 0.5),
        )
        if dense_cap >= 0:
            if dense_cap < mandatory_count:
                raise ValueError(
                    "dense_cap cannot be smaller than the per-request floor"
                )
            dense_count = min(dense_count, dense_cap)
    else:
        raise ValueError(f"unsupported route_mode: {route_mode!r}")
    dense_count = min(total_rows - 1, dense_count)

    by_request = torch.arange(
        total_rows, device="cuda", dtype=torch.int32
    ).reshape(batch_size, k + 1)
    mandatory = by_request[:, :min_dense_per_request].reshape(-1)
    remaining = by_request[:, min_dense_per_request:].reshape(-1)
    extra_count = dense_count - int(mandatory.numel())
    if extra_count > 0:
        order = torch.randperm(
            int(remaining.numel()), device="cuda", generator=generator
        )
        dense_rows = torch.cat((mandatory, remaining[order[:extra_count]]))
    else:
        dense_rows = mandatory
    dense_rows = dense_rows.sort().values.contiguous()
    sparse_mask = torch.ones(total_rows, device="cuda", dtype=torch.bool)
    sparse_mask[dense_rows.long()] = False
    sparse_rows = sparse_mask.nonzero().flatten().to(torch.int32).contiguous()
    return dense_rows, sparse_rows


def prepare_exact_weights(
    in_features: int,
    out_features: int,
    generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    weight = torch.randn(
        (in_features, out_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    residual24 = weight - weight24
    assert_24_weight(weight24)
    assert_24_weight(residual24)

    full_packed = pack_24(weight24, layout="n_major")
    full_values, full_meta = prepare_cutlass_sparse24_device_gemm(
        full_packed.values,
        full_packed.meta,
        layout=full_packed.layout,
        K=in_features,
    )
    residual_packed = pack_24(residual24, layout="n_major")
    residual_values, residual_meta = prepare_cutlass_sparse24_device_gemm(
        residual_packed.values,
        residual_packed.meta,
        layout=residual_packed.layout,
        K=in_features,
    )
    pair_packed = pack_24(
        torch.cat((weight24, residual24), dim=1), layout="n_major"
    )
    pair_values, pair_meta = prepare_cutlass_sparse24_pair_add(
        pair_packed.values,
        pair_packed.meta,
        layout=pair_packed.layout,
        K=in_features,
    )
    return (
        weight,
        full_values,
        full_meta,
        residual_values,
        residual_meta,
        pair_values,
        pair_meta,
    )


def run_case(
    model: str,
    projection: str,
    batch_size: int,
    k: int,
    weights: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    in_features: int,
    out_features: int,
    dense_fraction: float,
    min_dense_per_request: int,
    pair_config: str,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> dict[str, object]:
    (
        dense_weight,
        full_values,
        full_meta,
        residual_values,
        residual_meta,
        pair_values,
        pair_meta,
    ) = weights
    total_rows = batch_size * (k + 1)
    dense_rows, sparse_rows = route_indices(
        batch_size,
        k,
        dense_fraction=dense_fraction,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
    )
    dense_count = int(dense_rows.numel())
    sparse_count = int(sparse_rows.numel())
    dense_run = padded_rows(dense_count)
    sparse_run = padded_rows(sparse_count)

    x = torch.randn(
        (total_rows, in_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)
    dense_input = torch.zeros(
        (dense_run, in_features), device="cuda", dtype=torch.float16
    )
    sparse_input = torch.zeros(
        (sparse_run, in_features), device="cuda", dtype=torch.float16
    )

    baseline_output = torch.empty(
        (total_rows, out_features), device="cuda", dtype=torch.float16
    )
    baseline_workspace = torch.empty(
        (out_features, total_rows), device="cuda", dtype=torch.float16
    )
    residual_output = torch.empty(
        (dense_run, out_features), device="cuda", dtype=torch.float16
    )
    residual_workspace = torch.empty(
        (out_features, dense_run), device="cuda", dtype=torch.float16
    )
    paired_output = torch.empty_like(baseline_output)
    dense_output = torch.empty_like(baseline_output)

    baseline_full_stream = torch.cuda.Stream()
    baseline_residual_stream = torch.cuda.Stream()
    paired_sparse_stream = torch.cuda.Stream()
    paired_dense_stream = torch.cuda.Stream()

    def dense_reference() -> torch.Tensor:
        return torch.mm(x, dense_weight, out=dense_output)

    def baseline_exact() -> torch.Tensor:
        current = torch.cuda.current_stream()
        baseline_full_stream.wait_stream(current)
        baseline_residual_stream.wait_stream(current)
        with torch.cuda.stream(baseline_full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                full_values,
                full_meta,
                contiguous_output=True,
                out=baseline_output,
                workspace=baseline_workspace,
                device_config="auto",
            )
        with torch.cuda.stream(baseline_residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_input[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                dense_input,
                residual_values,
                residual_meta,
                contiguous_output=True,
                out=residual_output,
                workspace=residual_workspace,
                device_config="auto",
            )
        current.wait_stream(baseline_full_stream)
        current.wait_stream(baseline_residual_stream)
        return sparse24_add_indexed_rows_contiguous_(
            baseline_output, residual_output[:dense_count], dense_rows
        )

    def paired_exact() -> torch.Tensor:
        sparse24_partition_rows_(
            x,
            dense_rows,
            sparse_rows,
            dense_input[:dense_count],
            sparse_input[:sparse_count],
        )
        current = torch.cuda.current_stream()
        paired_sparse_stream.wait_stream(current)
        paired_dense_stream.wait_stream(current)
        with torch.cuda.stream(paired_sparse_stream):
            sparse24_cutlass_indexed_output_gemm_prepacked(
                sparse_input,
                full_values,
                full_meta,
                sparse_rows,
                output_rows=total_rows,
                out=paired_output,
                config="auto",
            )
        with torch.cuda.stream(paired_dense_stream):
            sparse24_cutlass_pair_add_indexed_prepacked(
                dense_input,
                pair_values,
                pair_meta,
                dense_rows,
                output_rows=total_rows,
                out=paired_output,
                config=pair_config,
            )
        current.wait_stream(paired_sparse_stream)
        current.wait_stream(paired_dense_stream)
        return paired_output

    dense_expected = dense_reference().clone()
    expected = baseline_exact().clone()
    actual = paired_exact().clone()
    torch.cuda.synchronize()
    dense_selected_max_abs_diff = float(
        (
            dense_expected[dense_rows.long()].float()
            - expected[dense_rows.long()].float()
        )
        .abs()
        .max()
        .item()
    )
    sparse_vs_dense_max_abs_diff = float(
        (
            dense_expected[sparse_rows.long()].float()
            - expected[sparse_rows.long()].float()
        )
        .abs()
        .max()
        .item()
    )
    max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
    if not torch.allclose(actual, expected, rtol=3e-2, atol=1e-1):
        raise RuntimeError(
            f"paired residual mismatch for {model}/{projection} "
            f"bs={batch_size} K={k}: max_abs_diff={max_abs_diff}"
        )
    if not torch.allclose(
        dense_expected[dense_rows.long()],
        expected[dense_rows.long()],
        rtol=3e-2,
        atol=1e-1,
    ):
        raise RuntimeError(
            f"selected dense rows mismatch for "
            f"{model}/{projection} bs={batch_size} K={k}: "
            f"max_abs_diff={dense_selected_max_abs_diff}"
        )

    dense_ms, baseline_control_ms = paired_graph_median_ms(
        dense_reference,
        baseline_exact,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )

    baseline_ms, paired_ms = paired_graph_median_ms(
        baseline_exact,
        paired_exact,
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
        "dense_fraction_actual": dense_count / total_rows,
        "dense_rows_padded": dense_run,
        "sparse_rows_padded": sparse_run,
        "in_features": in_features,
        "out_features": out_features,
        "pair_config": pair_config,
        "baseline_exact_ms": baseline_ms,
        "baseline_exact_dense_control_ms": baseline_control_ms,
        "paired_exact_ms": paired_ms,
        "paired_speedup": baseline_ms / paired_ms,
        "dense_ms": dense_ms,
        "baseline_exact_speedup_vs_dense": dense_ms / baseline_control_ms,
        "paired_exact_speedup_vs_dense": dense_ms / paired_ms,
        "dense_selected_max_abs_diff": dense_selected_max_abs_diff,
        "sparse_vs_dense_max_abs_diff": sparse_vs_dense_max_abs_diff,
        "max_abs_diff": max_abs_diff,
    }


def combined_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            int(row["batch_size"]),
            int(row["K"]),
            str(row["pair_config"]),
        )
        grouped.setdefault(key, []).append(row)
    combined: list[dict[str, object]] = []
    for (model, batch_size, k, pair_config), selected in sorted(grouped.items()):
        baseline_ms = sum(float(row["baseline_exact_ms"]) for row in selected)
        paired_ms = sum(float(row["paired_exact_ms"]) for row in selected)
        dense_ms = sum(float(row["dense_ms"]) for row in selected)
        combined.append(
            {
                "model": model,
                "batch_size": batch_size,
                "K": k,
                "pair_config": pair_config,
                "projection_count": len(selected),
                "projections": ",".join(str(row["projection"]) for row in selected),
                "baseline_exact_ms": baseline_ms,
                "paired_exact_ms": paired_ms,
                "paired_speedup": baseline_ms / paired_ms,
                "dense_ms": dense_ms,
                "baseline_exact_speedup_vs_dense": dense_ms / baseline_ms,
                "paired_exact_speedup_vs_dense": dense_ms / paired_ms,
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
    configs = list(dict.fromkeys(str(row["pair_config"]) for row in rows))
    line_styles = ["-", "--", "-."]

    figure, axes = plt.subplots(
        len(projections),
        len(models),
        figsize=(6.2 * len(models), 3.5 * len(projections)),
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
            for config_index, config in enumerate(configs):
                for k in sorted({int(row["K"]) for row in selected}):
                    by_key = [
                        row
                        for row in selected
                        if int(row["K"]) == k and row["pair_config"] == config
                    ]
                    if not by_key:
                        continue
                    axis.plot(
                        [int(row["batch_size"]) for row in by_key],
                        [float(row["paired_speedup"]) for row in by_key],
                        marker="o",
                        color=colors[k],
                        linestyle=line_styles[config_index % len(line_styles)],
                        label=f"K={k}, {config}",
                    )
            axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
            axis.set_title(f"{model} {projection}")
            axis.set_xlabel("Batch size")
            axis.set_ylabel("Exact mixed-row speedup")
            axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_root / "paired_residual_by_projection.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False
    )
    for axis, model in zip(axes[0], models, strict=True):
        selected = [row for row in combined if row["model"] == model]
        for config_index, config in enumerate(configs):
            for k in sorted({int(row["K"]) for row in selected}):
                by_key = [
                    row
                    for row in selected
                    if int(row["K"]) == k and row["pair_config"] == config
                ]
                if not by_key:
                    continue
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [float(row["paired_speedup"]) for row in by_key],
                    marker="o",
                    color=colors[k],
                    linestyle=line_styles[config_index % len(line_styles)],
                    label=f"K={k}, {config}",
                )
        axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
        axis.set_title(f"{model}, summed projections")
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Combined exact mixed-row speedup")
        axis.set_xticks(sorted({int(row["batch_size"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_root / "paired_residual_combined.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(projections),
        len(models),
        figsize=(6.2 * len(models), 3.5 * len(projections)),
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
                by_key = [row for row in selected if int(row["K"]) == k]
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["baseline_exact_speedup_vs_dense"])
                        for row in by_key
                    ],
                    marker="o",
                    color=colors[k],
                    linestyle="--",
                    label=f"K={k}, full+residual",
                )
                axis.plot(
                    [int(row["batch_size"]) for row in by_key],
                    [
                        float(row["paired_exact_speedup_vs_dense"])
                        for row in by_key
                    ],
                    marker="s",
                    color=colors[k],
                    linestyle="-",
                    label=f"K={k}, partition+pair",
                )
            axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
            axis.set_title(f"{model} {projection}")
            axis.set_xlabel("Batch size")
            axis.set_ylabel("Exact mixed-row speedup vs dense")
            axis.set_xticks(
                sorted({int(row["batch_size"]) for row in selected})
            )
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_root / "exact_mixed_vs_dense.png", dpi=200)
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
    parser.add_argument("--pair-configs", type=parse_csv_strings, default=("auto",))
    parser.add_argument("--dense-fraction", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=30)
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
    allowed_configs = {
        "auto",
        "256x32x64_s3_sw4",
        "256x64x64_s3",
        "256x64x64_s3_sw4",
    }
    invalid_configs = [
        config for config in args.pair_configs if config not in allowed_configs
    ]
    if invalid_models or invalid_projections or invalid_configs:
        raise ValueError(
            f"unsupported models={invalid_models}, projections={invalid_projections}, "
            f"pair_configs={invalid_configs}"
        )
    if not 0.0 < args.dense_fraction < 1.0:
        raise ValueError("--dense-fraction must be in (0, 1)")
    if args.min_dense_per_request < 1:
        raise ValueError("--min-dense-per-request must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results: list[dict[str, object]] = []
    for model in args.models:
        for projection in args.projections:
            in_features, out_features = projection_shape(model, projection)
            weights = prepare_exact_weights(
                in_features, out_features, generator
            )
            for batch_size in args.batch_sizes:
                for k in args.k_values:
                    for pair_config in args.pair_configs:
                        result = run_case(
                            model,
                            projection,
                            batch_size,
                            k,
                            weights,
                            in_features=in_features,
                            out_features=out_features,
                            dense_fraction=args.dense_fraction,
                            min_dense_per_request=args.min_dense_per_request,
                            pair_config=pair_config,
                            generator=generator,
                            unroll=args.unroll,
                            replays=args.replays,
                            trials=args.trials,
                            graph_warmup_replays=args.graph_warmup_replays,
                        )
                        results.append(result)
                        print(
                            f"{model} {projection} bs={batch_size} K={k} "
                            f"dense={int(result['dense_rows'])}/{int(result['total_rows'])} "
                            f"config={pair_config} "
                            f"speedup={float(result['paired_speedup']):.3f}x",
                            flush=True,
                        )
            del weights
            torch.cuda.empty_cache()

    combined = combined_rows(results)
    write_csv(args.output_root / "paired_residual_benchmark.csv", results)
    write_csv(args.output_root / "paired_residual_combined.csv", combined)
    write_plots(args.output_root, results, combined)
    print(args.output_root)


if __name__ == "__main__":
    main()
