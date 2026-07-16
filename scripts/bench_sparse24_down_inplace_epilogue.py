#!/usr/bin/env python3
"""Benchmark mixed-row sparse projections with an in-place residual epilogue."""

from __future__ import annotations

import argparse
import csv
import math
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
    parse_csv_ints,
    parse_csv_strings,
)
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    assert_24_weight,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_paired_finalize_residual_prepacked,
    sparse24_cutlass_paired_gather_residual_prepacked,
    sparse24_cutlass_paired_inplace_residual_prepacked,
    sparse24_cutlass_paired_self_contained_exact_down_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_routed_residual_epilogue_prepacked,
    sparse24_gather_rows_,
)


ROUTING = {
    "qwen3_8b": {"ratio": 0.125, "cap": 32},
    "llama3_1_8b": {"ratio": 0.3125, "cap": 64},
}


def graph_samples_ms(
    fn,
    *,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> list[float]:
    graph = capture_unrolled(fn, unroll=unroll)
    for _ in range(graph_warmup_replays):
        graph.replay()
    torch.cuda.synchronize()
    return [
        graph_sample_ms(graph, unroll=unroll, replays=replays)
        for _ in range(trials)
    ]


def parse_explicit_cases(value: str) -> tuple[tuple[int, int], ...]:
    cases: list[tuple[int, int]] = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "explicit cases must use ROWS:DENSE_ROWS"
            )
        rows, dense_count = map(int, fields)
        if rows <= 0 or not 0 < dense_count < rows:
            raise argparse.ArgumentTypeError(
                "explicit cases require ROWS > DENSE_ROWS > 0"
            )
        cases.append((rows, dense_count))
    return tuple(cases)


def make_dense_rows(
    batch_size: int,
    k: int,
    *,
    ratio: float,
    cap: int,
    generator: torch.Generator,
) -> torch.Tensor:
    rows = batch_size * (k + 1)
    dense_count = min(cap, int(batch_size * k * ratio + 0.5))
    dense_count = max(1, min(rows - 1, dense_count))
    return torch.randperm(
        rows, device="cuda", dtype=torch.int64, generator=generator
    )[:dense_count].sort().values.to(torch.int32).contiguous()


def prepare_weight(
    intermediate: int,
    hidden: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dense = torch.randn(
        (intermediate, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.02)
    dense.add_(torch.where(dense >= 0, 0.005, -0.005))
    sparse, _ = apply_random_24_mask(dense, generator=generator)
    residual = dense - sparse
    assert_24_weight(sparse)
    assert_24_weight(residual)

    def prepack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_24(weight, layout="n_major")
        return prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=intermediate,
        )

    sparse_values, sparse_meta = prepack(sparse)
    residual_values, residual_meta = prepack(residual)
    return dense, sparse_values, sparse_meta, residual_values, residual_meta


def run_case(
    model: str,
    batch_size: int,
    k: int,
    weights: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    generator: torch.Generator,
    args: argparse.Namespace,
    *,
    explicit_rows: int | None = None,
    explicit_dense_count: int | None = None,
) -> list[dict[str, object]]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    in_features = hidden if args.projection == "o" else intermediate
    out_features = hidden
    rows = (
        int(explicit_rows)
        if explicit_rows is not None
        else batch_size * (k + 1)
    )
    route = ROUTING[model]
    routing_ratio = (
        float(args.dense_ratio)
        if args.dense_ratio is not None
        else float(route["ratio"])
    )
    routing_cap = (
        int(args.dense_cap)
        if args.dense_cap is not None
        else int(route["cap"])
    )
    if explicit_dense_count is None:
        dense_rows = make_dense_rows(
            batch_size,
            k,
            ratio=routing_ratio,
            cap=routing_cap,
            generator=generator,
        )
    else:
        dense_rows = torch.randperm(
            rows,
            device="cuda",
            dtype=torch.int64,
            generator=generator,
        )[: int(explicit_dense_count)].sort().values.to(torch.int32).contiguous()
    dense_count = int(dense_rows.numel())
    dense_run = (dense_count + 7) // 8 * 8
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows.long()] = torch.arange(
        dense_count, device="cuda", dtype=torch.int32
    )
    dense_weight, full_values, full_meta, residual_values, residual_meta = weights
    x = torch.randn(
        (rows, in_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.1)
    dense_output = torch.empty(
        (rows, out_features), device="cuda", dtype=torch.float16
    )
    paired_output = torch.empty_like(dense_output)
    paired_residual = torch.empty(
        (dense_count, out_features), device="cuda", dtype=torch.float16
    )
    exact_output = torch.empty_like(dense_output)
    exact_base = torch.empty(
        (dense_count, out_features), device="cuda", dtype=torch.float16
    )

    def dense_fn() -> torch.Tensor:
        return torch.mm(x, dense_weight, out=dense_output)

    def paired_fn() -> torch.Tensor:
        sparse24_cutlass_paired_gather_residual_prepacked(
            x,
            full_values,
            full_meta,
            residual_values,
            residual_meta,
            dense_rows,
            full_out=paired_output,
            residual_out=paired_residual,
            schedule="interleaved",
            config=args.paired_config,
        )
        return sparse24_add_indexed_rows_contiguous_(
            paired_output, paired_residual, dense_rows
        )

    def exact_fn() -> torch.Tensor:
        output, _ = sparse24_cutlass_paired_self_contained_exact_down_prepacked(
            x,
            full_values,
            full_meta,
            residual_values,
            residual_meta,
            dense_rows,
            dense_slots,
            out=exact_output,
            dense_base=exact_base,
        )
        return output

    residual_first_dense_x = torch.zeros(
        (dense_run, in_features), device="cuda", dtype=torch.float16
    )
    residual_first_residual = torch.empty(
        (dense_run, out_features), device="cuda", dtype=torch.float16
    )
    residual_first_output = torch.empty_like(dense_output)

    def residual_first_fn() -> torch.Tensor:
        sparse24_gather_rows_(
            x, dense_rows, residual_first_dense_x[:dense_count]
        )
        sparse24_cutlass_inline_transpose_gemm_prepacked(
            residual_first_dense_x,
            residual_values,
            residual_meta,
            out=residual_first_residual,
            config=args.residual_first_residual_config,
            store_mode="vector",
        )
        return sparse24_cutlass_routed_residual_epilogue_prepacked(
            x,
            full_values,
            full_meta,
            residual_first_residual[:dense_count],
            dense_slots,
            out=residual_first_output,
            config=args.residual_first_full_config,
        )

    expected = paired_fn().clone()
    sparse24_cutlass_paired_gather_residual_prepacked(
        x,
        full_values,
        full_meta,
        residual_values,
        residual_meta,
        dense_rows,
        full_out=paired_output,
        residual_out=paired_residual,
        schedule="interleaved",
        config=args.paired_config,
    )
    full_reference = paired_output.clone()
    dense_expected = dense_fn().clone()
    torch.cuda.synchronize()
    dense_selected_diff = float(
        (
            expected[dense_rows.long()].float()
            - dense_expected[dense_rows.long()].float()
        ).abs().max().item()
    )
    if not torch.allclose(
        expected[dense_rows.long()],
        dense_expected[dense_rows.long()],
        rtol=3e-2,
        atol=1e-1,
    ):
        raise RuntimeError(
            f"dense route mismatch for {model} bs={batch_size} K={k}: "
            f"{dense_selected_diff:.6f}"
        )
    dense_samples = graph_samples_ms(
        dense_fn,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    paired_samples = graph_samples_ms(
        paired_fn,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    dense_ms = statistics.median(dense_samples)
    paired_ms = statistics.median(paired_samples)
    residual_first_metrics: dict[str, object] = {
        "residual_first_full_config": "",
        "residual_first_residual_config": "",
        "residual_first_ms": "",
        "residual_first_vs_paired_speedup": "",
        "residual_first_vs_dense_speedup": "",
        "residual_first_max_abs_diff": "",
        "self_contained_exact_ms": "",
        "self_contained_exact_vs_paired_speedup": "",
        "self_contained_exact_vs_dense_speedup": "",
        "self_contained_exact_max_abs_diff": "",
    }
    run_auxiliary = not args.skip_auxiliary and rows % 8 == 0
    if run_auxiliary:
        exact_actual = exact_fn().clone()
        residual_first_actual = residual_first_fn().clone()
        torch.cuda.synchronize()
        exact_max_abs_diff = float(
            (exact_actual.float() - expected.float()).abs().max().item()
        )
        if not torch.allclose(exact_actual, expected, rtol=3e-2, atol=1e-1):
            raise RuntimeError(
                f"self-contained exact projection mismatch for {model} "
                f"bs={batch_size} K={k}: max_abs_diff={exact_max_abs_diff:.6f}"
            )
        residual_first_max_abs_diff = float(
            (residual_first_actual.float() - expected.float()).abs().max().item()
        )
        if not torch.allclose(
            residual_first_actual, expected, rtol=3e-2, atol=1e-1
        ):
            raise RuntimeError(
                f"residual-first epilogue mismatch for {model} "
                f"bs={batch_size} K={k}: "
                f"max_abs_diff={residual_first_max_abs_diff:.6f}"
            )
        exact_samples = graph_samples_ms(
            exact_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        exact_ms = statistics.median(exact_samples)
        residual_first_samples = graph_samples_ms(
            residual_first_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        residual_first_ms = statistics.median(residual_first_samples)
        residual_first_metrics.update(
            {
                "residual_first_full_config": args.residual_first_full_config,
                "residual_first_residual_config": (
                    args.residual_first_residual_config
                ),
                "residual_first_ms": residual_first_ms,
                "residual_first_vs_paired_speedup": (
                    paired_ms / residual_first_ms
                ),
                "residual_first_vs_dense_speedup": dense_ms / residual_first_ms,
                "residual_first_max_abs_diff": residual_first_max_abs_diff,
                "self_contained_exact_ms": exact_ms,
                "self_contained_exact_vs_paired_speedup": paired_ms / exact_ms,
                "self_contained_exact_vs_dense_speedup": dense_ms / exact_ms,
                "self_contained_exact_max_abs_diff": exact_max_abs_diff,
            }
        )
    finalizer_metrics: dict[str, object] = {
        "finalizer_ms": "",
        "finalizer_vs_paired_speedup": "",
        "finalizer_vs_dense_speedup": "",
        "finalizer_max_abs_diff": "",
        "finalizer_counter_max": "",
        "finalizer_counter_max_after_graph": "",
    }
    if args.include_finalizer:
        finalizer_output = torch.empty_like(dense_output)
        finalizer_residual = torch.empty_like(paired_residual)
        finalizer_counters = torch.zeros(
            out_features // 256, device="cuda", dtype=torch.int32
        )

        def finalizer_fn() -> torch.Tensor:
            return sparse24_cutlass_paired_finalize_residual_prepacked(
                x,
                full_values,
                full_meta,
                residual_values,
                residual_meta,
                dense_rows,
                out=finalizer_output,
                residual_out=finalizer_residual,
                feature_counters=finalizer_counters,
                config=args.finalizer_config,
                schedule=args.finalizer_schedule,
            )

        finalizer_actual = finalizer_fn().clone()
        torch.cuda.synchronize()
        finalizer_max_abs_diff = float(
            (finalizer_actual.float() - expected.float()).abs().max().item()
        )
        finalizer_counter_max = int(
            finalizer_counters.abs().max().item()
        )
        if not torch.allclose(
            finalizer_actual, expected, rtol=3e-2, atol=1e-1
        ):
            raise RuntimeError(
                f"last-CTA finalizer mismatch for {model} bs={batch_size} "
                f"K={k}: max={finalizer_max_abs_diff:.6f}"
            )
        if finalizer_counter_max != 0:
            raise RuntimeError(
                f"last-CTA finalizer counter leak for {model} "
                f"bs={batch_size} K={k}: {finalizer_counter_max}"
            )
        finalizer_samples = graph_samples_ms(
            finalizer_fn,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        finalizer_counter_max_after_graph = int(
            finalizer_counters.abs().max().item()
        )
        if finalizer_counter_max_after_graph != 0:
            raise RuntimeError(
                f"last-CTA finalizer counter leak after graph for {model} "
                f"bs={batch_size} K={k}: "
                f"{finalizer_counter_max_after_graph}"
            )
        finalizer_ms = statistics.median(finalizer_samples)
        finalizer_metrics = {
            "finalizer_ms": finalizer_ms,
            "finalizer_vs_paired_speedup": paired_ms / finalizer_ms,
            "finalizer_vs_dense_speedup": dense_ms / finalizer_ms,
            "finalizer_max_abs_diff": finalizer_max_abs_diff,
            "finalizer_counter_max": finalizer_counter_max,
            "finalizer_counter_max_after_graph": (
                finalizer_counter_max_after_graph
            ),
        }
    output: list[dict[str, object]] = []
    for config in args.inplace_configs:
        for schedule in args.schedules:
            candidate_output = torch.empty_like(dense_output)
            feature_columns = int(config.split("x", 1)[0])
            counters = torch.zeros(
                hidden // feature_columns, device="cuda", dtype=torch.int32
            )

            def candidate_fn() -> torch.Tensor:
                return sparse24_cutlass_paired_inplace_residual_prepacked(
                    x,
                    full_values,
                    full_meta,
                    residual_values,
                    residual_meta,
                    dense_rows,
                    out=candidate_output,
                    feature_counters=counters,
                    config=config,
                    schedule=schedule,
                )

            actual = candidate_fn().clone()
            torch.cuda.synchronize()
            absolute_diff = (actual.float() - expected.float()).abs()
            max_abs_diff = float(absolute_diff.max().item())
            dense_max_abs_diff = float(
                absolute_diff[dense_rows.long()].max().item()
            )
            sparse_mask = torch.ones(
                rows, device="cuda", dtype=torch.bool
            )
            sparse_mask[dense_rows.long()] = False
            sparse_max_abs_diff = float(
                absolute_diff[sparse_mask].max().item()
            )
            counter_max = int(counters.abs().max().item())
            if not torch.allclose(actual, expected, rtol=3e-2, atol=1e-1):
                first_bad_row = int(
                    (absolute_diff.max(dim=1).values > 1e-1)
                    .nonzero()[0]
                    .item()
                )
                candidate_delta = (
                    actual[dense_rows.long()].float()
                    - full_reference[dense_rows.long()].float()
                )
                expected_delta = (
                    expected[dense_rows.long()].float()
                    - full_reference[dense_rows.long()].float()
                )
                raise RuntimeError(
                    f"in-place mismatch for {model} bs={batch_size} K={k} "
                    f"{config}/{schedule}: max={max_abs_diff:.6f}, "
                    f"dense={dense_max_abs_diff:.6f}, "
                    f"sparse={sparse_max_abs_diff:.6f}, "
                    f"first_bad_row={first_bad_row}, "
                    f"candidate_delta_max={candidate_delta.abs().max().item():.6f}, "
                    f"expected_delta_max={expected_delta.abs().max().item():.6f}, "
                    f"candidate_delta_l1={candidate_delta.abs().mean().item():.6f}, "
                    f"expected_delta_l1={expected_delta.abs().mean().item():.6f}"
                )
            if counter_max != 0:
                raise RuntimeError(
                    f"counter leak for {model} bs={batch_size} K={k} "
                    f"{config}/{schedule}: {counter_max}"
                )
            samples = graph_samples_ms(
                candidate_fn,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
                graph_warmup_replays=args.graph_warmup_replays,
            )
            counter_max_after_graph = int(counters.abs().max().item())
            if counter_max_after_graph != 0:
                raise RuntimeError(
                    f"counter leak after graph for {model} bs={batch_size} "
                    f"K={k} {config}/{schedule}: {counter_max_after_graph}"
                )
            candidate_ms = statistics.median(samples)
            output.append(
                {
                    "model": model,
                    "case": f"m{rows}_d{dense_count}",
                    "projection": args.projection,
                    "batch_size": batch_size,
                    "K": k,
                    "rows": rows,
                    "dense_rows": dense_count,
                    "dense_ratio": dense_count / rows,
                    "routing_ratio": routing_ratio,
                    "routing_cap": routing_cap,
                    "in_features": in_features,
                    "out_features": out_features,
                    "paired_config": args.paired_config,
                    "inplace_config": config,
                    "schedule": schedule,
                    "dense_down_ms": dense_ms,
                    "paired_add_ms": paired_ms,
                    "inplace_ms": candidate_ms,
                    "inplace_vs_paired_speedup": paired_ms / candidate_ms,
                    "inplace_vs_dense_speedup": dense_ms / candidate_ms,
                    "paired_vs_dense_speedup": dense_ms / paired_ms,
                    "max_abs_diff": max_abs_diff,
                    "dense_max_abs_diff": dense_max_abs_diff,
                    "sparse_max_abs_diff": sparse_max_abs_diff,
                    "dense_selected_max_abs_diff": dense_selected_diff,
                    "counter_max": counter_max,
                    "counter_max_after_graph": counter_max_after_graph,
                    **residual_first_metrics,
                    **finalizer_metrics,
                    "scope": "token_mixed_projection_kernel_only",
                }
            )
    return output


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    cases = tuple(
        dict.fromkeys(
            (str(row["model"]), int(row["batch_size"])) for row in rows
        )
    )
    columns = min(3, len(cases))
    panel_rows = math.ceil(len(cases) / columns)
    figure, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(5.6 * columns, 4.0 * panel_rows),
        squeeze=False,
    )
    colors = {
        "partitioned": "#B33F40",
        "shared_phased": "#176B87",
        "interleaved": "#4F772D",
    }
    markers = {
        "256x32_full_256x32_residual_inplace": "o",
        "256x64_full_256x32_residual_inplace": "s",
        "256x64w64_full_256x32_residual_inplace": "^",
        "128x32_full_128x32_residual_inplace": "P",
        "64x64w32x64_full_64x32_residual_inplace": "v",
        "256x32_full_last_owner_256x32_residual_inplace": "X",
        "256x32_full_256x32_residual_f16_inplace": "D",
        "256x32_full_256x32_residual_all_f16_inplace": "*",
    }
    flat_axes = list(axes.flat)
    for axis, (model, batch_size) in zip(flat_axes, cases, strict=False):
        selected = [
            row
            for row in rows
            if row["model"] == model
            and int(row["batch_size"]) == batch_size
        ]
        for config in dict.fromkeys(str(row["inplace_config"]) for row in selected):
            for schedule in dict.fromkeys(str(row["schedule"]) for row in selected):
                series = sorted(
                    (
                        row
                        for row in selected
                        if row["inplace_config"] == config
                        and row["schedule"] == schedule
                    ),
                    key=lambda row: int(row["K"]),
                )
                if "last_owner" in config:
                    short_config = "M256N32 last-owner"
                elif "residual_f16" in config:
                    short_config = "M256N32 R24-FP16"
                elif "all_f16" in config:
                    short_config = "M256N32 W24/R24-FP16"
                elif config.startswith("64x64"):
                    short_config = "M64N64/M64N32"
                elif config.startswith("256x64w64"):
                    short_config = "N64w64/N32"
                elif config.startswith("256x64"):
                    short_config = "N64/N32"
                elif config.startswith("128x32"):
                    short_config = "M128N32/M128N32"
                else:
                    short_config = "N32/N32"
                axis.plot(
                    [int(row["K"]) for row in series],
                    [float(row["inplace_vs_paired_speedup"]) for row in series],
                    color=colors[schedule],
                    marker=markers[config],
                    linestyle="-",
                    label=f"{short_config} {schedule} vs paired+add",
                )
                axis.plot(
                    [int(row["K"]) for row in series],
                    [float(row["inplace_vs_dense_speedup"]) for row in series],
                    color=colors[schedule],
                    marker=markers[config],
                    linestyle="--",
                    label=f"{short_config} {schedule} vs dense",
                )
        finalizer_by_k = {
            int(row["K"]): row
            for row in selected
            if str(row.get("finalizer_ms", ""))
        }
        if finalizer_by_k:
            finalizer_series = [
                finalizer_by_k[k] for k in sorted(finalizer_by_k)
            ]
            axis.plot(
                [int(row["K"]) for row in finalizer_series],
                [
                    float(row["finalizer_vs_paired_speedup"])
                    for row in finalizer_series
                ],
                color="#7A5195",
                marker="D",
                linestyle="-",
                label="last-CTA finalizer vs paired+add",
            )
            axis.plot(
                [int(row["K"]) for row in finalizer_series],
                [
                    float(row["finalizer_vs_dense_speedup"])
                    for row in finalizer_series
                ],
                color="#7A5195",
                marker="D",
                linestyle="--",
                label="last-CTA finalizer vs dense",
            )
        residual_first_by_k = {
            int(row["K"]): row
            for row in selected
            if str(row.get("residual_first_ms", ""))
        }
        residual_first_series = [
            residual_first_by_k[k] for k in sorted(residual_first_by_k)
        ]
        if residual_first_series:
            axis.plot(
                [int(row["K"]) for row in residual_first_series],
                [
                    float(row["residual_first_vs_paired_speedup"])
                    for row in residual_first_series
                ],
                color="#D97706",
                marker="X",
                linestyle="-",
                label="residual-first sparse epilogue vs paired+add",
            )
            axis.plot(
                [int(row["K"]) for row in residual_first_series],
                [
                    float(row["residual_first_vs_dense_speedup"])
                    for row in residual_first_series
                ],
                color="#D97706",
                marker="X",
                linestyle="--",
                label="residual-first sparse epilogue vs dense",
            )
        self_contained_by_k = {
            int(row["K"]): row
            for row in selected
            if str(row.get("self_contained_exact_ms", ""))
        }
        self_contained_series = [
            self_contained_by_k[k] for k in sorted(self_contained_by_k)
        ]
        if self_contained_series:
            axis.plot(
                [int(row["K"]) for row in self_contained_series],
                [
                    float(row["self_contained_exact_vs_paired_speedup"])
                    for row in self_contained_series
                ],
                color="#111827",
                marker="H",
                linestyle="-",
                label="self-contained exact vs paired+add",
            )
            axis.plot(
                [int(row["K"]) for row in self_contained_series],
                [
                    float(row["self_contained_exact_vs_dense_speedup"])
                    for row in self_contained_series
                ],
                color="#111827",
                marker="H",
                linestyle="--",
                label="self-contained exact vs dense",
            )
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
        projection = str(selected[0]["projection"])
        axis.set_title(f"{model} bs={batch_size} mixed {projection}")
        axis.set_xlabel("K")
        axis.set_ylabel("Reference / candidate speedup")
        axis.set_xticks(sorted({int(row["K"]) for row in selected}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7.5)
    for axis in flat_axes[len(cases) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Token-mixed projection kernel only; not end-to-end throughput",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    def optional_metric(row: dict[str, object], key: str, suffix: str = "") -> str:
        value = row.get(key, "")
        if str(value) == "":
            return "-"
        return f"{float(value):.5f}{suffix}"

    lines = [
        "# Mixed-row projection in-place epilogue",
        "",
        "Kernel-only ablation. This is not end-to-end SpecLink throughput.",
        "",
        "| Model | Projection | bs | K | Dense rows | Config | Schedule | Dense ms | Paired+add ms | Self-contained ms | Residual-first ms | In-place ms | Finalizer ms | Self-contained vs paired | Self-contained vs dense | Residual-first vs paired | Residual-first vs dense | In-place vs dense | Finalizer vs dense | Self-contained max diff | Residual-first max diff | In-place max diff | Finalizer max diff |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['projection']} | "
            f"{row['batch_size']} | {row['K']} | "
            f"{row['dense_rows']} | {row['inplace_config']} | "
            f"{row['schedule']} | "
            f"{float(row['dense_down_ms']):.5f} | "
            f"{float(row['paired_add_ms']):.5f} | "
            f"{optional_metric(row, 'self_contained_exact_ms')} | "
            f"{optional_metric(row, 'residual_first_ms')} | "
            f"{float(row['inplace_ms']):.5f} | "
            f"{optional_metric(row, 'finalizer_ms')} | "
            f"{optional_metric(row, 'self_contained_exact_vs_paired_speedup', 'x')} | "
            f"{optional_metric(row, 'self_contained_exact_vs_dense_speedup', 'x')} | "
            f"{optional_metric(row, 'residual_first_vs_paired_speedup', 'x')} | "
            f"{optional_metric(row, 'residual_first_vs_dense_speedup', 'x')} | "
            f"{float(row['inplace_vs_dense_speedup']):.3f}x | "
            f"{optional_metric(row, 'finalizer_vs_dense_speedup', 'x')} | "
            f"{optional_metric(row, 'self_contained_exact_max_abs_diff')} | "
            f"{optional_metric(row, 'residual_first_max_abs_diff')} | "
            f"{float(row['max_abs_diff']):.5f} | "
            f"{optional_metric(row, 'finalizer_max_abs_diff')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_csv_strings, default=tuple(MODELS))
    parser.add_argument("--projection", choices=("down", "o"), default="down")
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=(16,))
    parser.add_argument("--k-values", type=parse_csv_ints, default=(6, 8, 10))
    parser.add_argument(
        "--explicit-cases",
        type=parse_explicit_cases,
        default=(),
        help=(
            "Comma-separated ROWS:DENSE_ROWS shapes; skips the batch/K "
            "matrix and measures the exact production graph shapes."
        ),
    )
    parser.add_argument(
        "--dense-ratio",
        type=float,
        default=None,
        help="optional routing-ratio override for every selected model",
    )
    parser.add_argument(
        "--dense-cap",
        type=int,
        default=None,
        help="optional dense-row cap override for every selected model",
    )
    parser.add_argument(
        "--paired-config",
        default="256x64_full_256x32_residual_contiguous",
    )
    parser.add_argument(
        "--inplace-configs",
        type=parse_csv_strings,
        default=(
            "256x32_full_256x32_residual_inplace",
            "256x64_full_256x32_residual_inplace",
        ),
    )
    parser.add_argument(
        "--schedules",
        type=parse_csv_strings,
        default=("partitioned", "shared_phased"),
    )
    parser.add_argument(
        "--residual-first-full-config",
        choices=(
            "128x32x64_s4_sw4",
            "128x64x64_s5",
            "256x32x64_s3_sw4",
            "256x64x64_s3",
            "256x64x64_s3_sw4",
        ),
        default="256x64x64_s3",
    )
    parser.add_argument(
        "--residual-first-residual-config",
        choices=(
            "128x32x64_s4_sw4",
            "128x64x64_s5",
            "256x32x64_s3_sw4",
            "256x64x64_s3",
            "256x64x64_s3_sw4",
        ),
        default="256x32x64_s3_sw4",
    )
    parser.add_argument(
        "--include-finalizer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-auxiliary",
        action="store_true",
        help=(
            "Skip self-contained and residual-first auxiliary timing for "
            "arbitrary-M wave sweeps."
        ),
    )
    parser.add_argument(
        "--finalizer-config",
        default="256x32_full_256x32_residual_finalize",
    )
    parser.add_argument(
        "--finalizer-schedule",
        choices=("partitioned", "interleaved"),
        default="partitioned",
    )
    parser.add_argument("--unroll", type=int, default=2)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=3)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the projection speedup figure.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-candidate dictionaries while retaining artifacts.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.dense_ratio is not None and not 0.0 < args.dense_ratio < 1.0:
        parser.error("--dense-ratio must be between zero and one")
    if args.dense_cap is not None and args.dense_cap <= 0:
        parser.error("--dense-cap must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    invalid_models = [model for model in args.models if model not in MODELS]
    if invalid_models:
        raise ValueError(f"unsupported models: {invalid_models}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results: list[dict[str, object]] = []
    for model in args.models:
        in_features = (
            int(MODELS[model]["hidden"])
            if args.projection == "o"
            else int(MODELS[model]["intermediate"])
        )
        weights = prepare_weight(
            in_features,
            int(MODELS[model]["hidden"]),
            generator,
        )
        cases = (
            [(0, 0, rows, dense_count) for rows, dense_count in args.explicit_cases]
            if args.explicit_cases
            else [
                (batch_size, k, None, None)
                for batch_size in args.batch_sizes
                for k in args.k_values
            ]
        )
        for batch_size, k, explicit_rows, explicit_dense_count in cases:
            case_rows = run_case(
                model,
                batch_size,
                k,
                weights,
                generator,
                args,
                explicit_rows=explicit_rows,
                explicit_dense_count=explicit_dense_count,
            )
            results.extend(case_rows)
            if not args.quiet:
                for row in case_rows:
                    print(row, flush=True)
        del weights
        torch.cuda.empty_cache()

    csv_path = args.output_root / "down_inplace_epilogue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    if args.plot:
        write_plot(args.output_root / "down_inplace_epilogue.png", results)
    write_report(args.output_root / "report.md", results)
    print(csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
