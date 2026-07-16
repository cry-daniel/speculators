#!/usr/bin/env python3
"""Benchmark exact dense+2:4 row routing without compact route buffers."""

from __future__ import annotations

import argparse
import csv
import math
import os
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
    dense_cutlass_routed_rows_weight_t,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_heterogeneous_linear_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_routed_exact_linear_prepacked,
    sparse24_cutlass_routed_sparse_rows_prepacked,
    sparse24_mixed_dense_override_prepacked,
)


QKV_OUTPUTS = {"qwen3_8b": 6144, "llama3_1_8b": 6144}


def make_route(
    batch_size: int,
    k: int,
    *,
    dense_ratio: float,
    min_dense_per_request: int,
    generator: torch.Generator,
    dense_cap: int = -1,
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
    if dense_cap >= 0:
        dense_count = min(dense_count, dense_cap)
        mandatory = mandatory[:dense_count]
    candidates = by_request[:, min_dense_per_request:k].flatten()
    extra_count = dense_count - int(mandatory.numel())
    permutation = torch.randperm(
        int(candidates.numel()), device="cuda", generator=generator
    )
    dense_rows = (
        torch.cat((mandatory, candidates[permutation[:extra_count]]))
        .sort()
        .values.contiguous()
    )
    sparse_mask = torch.ones(rows, device="cuda", dtype=torch.bool)
    sparse_mask[dense_rows.long()] = False
    sparse_rows = sparse_mask.nonzero().flatten().to(torch.int32).contiguous()
    return dense_rows, sparse_rows


def prepare_weight(
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

    def prepack(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_24(matrix, layout="n_major")
        return prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=in_features,
        )

    sparse_values, sparse_meta = prepack(weight24)
    residual_values, residual_meta = prepack(weight - weight24)
    return (
        weight,
        weight24,
        sparse_values,
        sparse_meta,
        residual_values,
        residual_meta,
    )


def projection_shapes(model: str) -> dict[str, tuple[int, int]]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    return {
        "qkv": (hidden, QKV_OUTPUTS[model]),
        "o": (hidden, hidden),
        "gate_up": (hidden, 2 * intermediate),
        "down": (intermediate, hidden),
    }


def measure_pair(
    baseline,
    candidate,
    args: argparse.Namespace,
) -> tuple[float, float]:
    return paired_graph_median_ms(
        baseline,
        candidate,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )


def run_projection(
    *,
    model: str,
    batch_size: int,
    k: int,
    projection: str,
    dense_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    rows = batch_size * (k + 1)
    in_features, out_features = projection_shapes(model)[projection]
    x = torch.randn(
        (rows, in_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.1)
    (
        weight,
        weight24,
        sparse_values,
        sparse_meta,
        residual_values,
        residual_meta,
    ) = prepare_weight(in_features, out_features, generator)
    dense_weight = weight.t().contiguous()
    dense_out = torch.empty(
        (rows, out_features), device="cuda", dtype=torch.float16
    )
    sparse_out = torch.empty_like(dense_out)
    old_out = torch.empty_like(dense_out)
    fused_override_out = torch.empty_like(dense_out)
    parallel_override_out = torch.empty_like(dense_out)
    new_out = torch.empty_like(dense_out)
    dual_stream_out = torch.empty_like(dense_out)
    dense_rows_long = dense_rows.long()
    fused_dense_x = torch.empty(
        (int(dense_rows.numel()), in_features),
        device="cuda",
        dtype=torch.float16,
    )
    fused_dense_y = torch.empty(
        (int(dense_rows.numel()), out_features),
        device="cuda",
        dtype=torch.float16,
    )
    fused_workspace = torch.empty(
        (out_features, rows), device="cuda", dtype=torch.float16
    )
    parallel_dense_x = torch.empty_like(fused_dense_x)
    parallel_dense_y = torch.empty_like(fused_dense_y)
    sparse_stream = torch.cuda.Stream()
    dense_stream = torch.cuda.Stream()
    override_sparse_stream = torch.cuda.Stream()
    override_dense_stream = torch.cuda.Stream()
    heterogeneous_config = args.heterogeneous_config
    if heterogeneous_config == "production":
        from vllm.speclink_linear import _qkv_heterogeneous_config

        heterogeneous_config = _qkv_heterogeneous_config(
            rows,
            int(dense_rows.numel()),
            out_features,
        )

    def dense_fn() -> torch.Tensor:
        return torch.mm(x, weight, out=dense_out)

    def sparse_fn() -> torch.Tensor:
        return sparse24_cutlass_inline_transpose_gemm_prepacked(
            x,
            sparse_values,
            sparse_meta,
            out=sparse_out,
            config=args.static_config,
            store_mode="vector",
        )

    def old_exact_fn() -> torch.Tensor:
        return sparse24_cutlass_routed_exact_linear_prepacked(
            x,
            sparse_values,
            sparse_meta,
            residual_values,
            residual_meta,
            dense_rows,
            sparse_rows,
            out=old_out,
            config=args.old_config,
        )

    def fused_override_fn() -> torch.Tensor:
        return sparse24_mixed_dense_override_prepacked(
            x,
            dense_weight,
            sparse_values,
            sparse_meta,
            dense_rows,
            out=fused_override_out,
            dense_x=fused_dense_x,
            dense_y=fused_dense_y,
            workspace=fused_workspace,
            device_config=args.static_config,
        )

    def parallel_override_fn() -> torch.Tensor:
        current = torch.cuda.current_stream()
        override_sparse_stream.wait_stream(current)
        override_dense_stream.wait_stream(current)
        with torch.cuda.stream(override_sparse_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                sparse_values,
                sparse_meta,
                contiguous_output=True,
                out=parallel_override_out,
                device_config=args.static_config,
            )
        with torch.cuda.stream(override_dense_stream):
            torch.index_select(
                x,
                0,
                dense_rows_long,
                out=parallel_dense_x,
            )
            torch.mm(parallel_dense_x, weight, out=parallel_dense_y)
        current.wait_stream(override_sparse_stream)
        current.wait_stream(override_dense_stream)
        sparse24_copy_indexed_rows_contiguous_(
            parallel_override_out,
            parallel_dense_y,
            dense_rows,
        )
        return parallel_override_out

    def heterogeneous_fn() -> torch.Tensor:
        return sparse24_cutlass_heterogeneous_linear_prepacked(
            x,
            sparse_values,
            sparse_meta,
            dense_weight,
            dense_rows,
            sparse_rows,
            out=new_out,
            config=heterogeneous_config,
        )

    def dual_stream_fn() -> torch.Tensor:
        current = torch.cuda.current_stream()
        sparse_stream.wait_stream(current)
        dense_stream.wait_stream(current)
        with torch.cuda.stream(sparse_stream):
            sparse24_cutlass_routed_sparse_rows_prepacked(
                x,
                sparse_values,
                sparse_meta,
                sparse_rows,
                out=dual_stream_out,
                config=args.sparse_component_config,
            )
        with torch.cuda.stream(dense_stream):
            dense_cutlass_routed_rows_weight_t(
                x,
                dense_weight,
                dense_rows,
                out=dual_stream_out,
                config=args.dense_component_config,
            )
        current.wait_stream(sparse_stream)
        current.wait_stream(dense_stream)
        return dual_stream_out

    dense_ms, sparse_ms = measure_pair(dense_fn, sparse_fn, args)
    dense_old_ms, old_ms = measure_pair(dense_fn, old_exact_fn, args)
    dense_fused_override_ms, fused_override_ms = measure_pair(
        dense_fn, fused_override_fn, args
    )
    dense_parallel_override_ms, parallel_override_ms = measure_pair(
        dense_fn, parallel_override_fn, args
    )
    dense_new_ms, new_ms = measure_pair(dense_fn, heterogeneous_fn, args)
    dense_dual_ms, dual_ms = measure_pair(dense_fn, dual_stream_fn, args)

    def sparse_component_fn() -> torch.Tensor:
        return sparse24_cutlass_routed_sparse_rows_prepacked(
            x,
            sparse_values,
            sparse_meta,
            sparse_rows,
            out=dual_stream_out,
            config=args.sparse_component_config,
        )

    def dense_component_fn() -> torch.Tensor:
        return dense_cutlass_routed_rows_weight_t(
            x,
            dense_weight,
            dense_rows,
            out=dual_stream_out,
            config=args.dense_component_config,
        )

    sparse_component_ms, dense_component_ms = measure_pair(
        sparse_component_fn, dense_component_fn, args
    )

    heterogeneous_fn()
    old_exact_fn()
    fused_override_fn()
    parallel_override_fn()
    dual_stream_fn()
    expected = x @ weight24
    expected[dense_rows.long()] = x[dense_rows.long()] @ weight
    torch.cuda.synchronize()
    old_diff = float((old_out - expected).abs().max().item())
    fused_override_diff = float(
        (fused_override_out - expected).abs().max().item()
    )
    parallel_override_diff = float(
        (parallel_override_out - expected).abs().max().item()
    )
    heterogeneous_vs_parallel_diff = float(
        (new_out - parallel_override_out).abs().max().item()
    )
    new_diff = float((new_out - expected).abs().max().item())
    dual_diff = float((dual_stream_out - expected).abs().max().item())
    new_dense_diff = float(
        (new_out[dense_rows.long()] - expected[dense_rows.long()])
        .abs()
        .max()
        .item()
    )
    new_sparse_diff = float(
        (new_out[sparse_rows.long()] - expected[sparse_rows.long()])
        .abs()
        .max()
        .item()
    )
    dual_dense_diff = float(
        (dual_stream_out[dense_rows.long()] - expected[dense_rows.long()])
        .abs()
        .max()
        .item()
    )
    dual_sparse_diff = float(
        (dual_stream_out[sparse_rows.long()] - expected[sparse_rows.long()])
        .abs()
        .max()
        .item()
    )
    if not torch.allclose(new_out, expected, rtol=3e-2, atol=8e-2):
        raise AssertionError(
            f"heterogeneous output mismatch for {model}/{projection}: "
            f"max_abs_diff={new_diff}, dense_diff={new_dense_diff}, "
            f"sparse_diff={new_sparse_diff}, "
            f"dual_dense_diff={dual_dense_diff}, "
            f"dual_sparse_diff={dual_sparse_diff}"
        )
    if not torch.allclose(
        fused_override_out, expected, rtol=3e-2, atol=8e-2
    ):
        raise AssertionError(
            f"fused override output mismatch for {model}/{projection}: "
            f"max_abs_diff={fused_override_diff}"
        )
    if not torch.allclose(
        parallel_override_out, expected, rtol=3e-2, atol=8e-2
    ):
        raise AssertionError(
            f"parallel override output mismatch for {model}/{projection}: "
            f"max_abs_diff={parallel_override_diff}"
        )
    if not torch.allclose(
        dual_stream_out, expected, rtol=3e-2, atol=8e-2
    ):
        raise AssertionError(
            f"dual-stream output mismatch for {model}/{projection}: "
            f"max_abs_diff={dual_diff}"
        )

    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": int(dense_rows.numel()),
        "sparse_rows": int(sparse_rows.numel()),
        "dense_fraction": int(dense_rows.numel()) / rows,
        "projection": projection,
        "in_features": in_features,
        "out_features": out_features,
        "old_config": args.old_config,
        "static_config": args.static_config,
        "heterogeneous_config": heterogeneous_config,
        "sparse_component_config": args.sparse_component_config,
        "dense_component_config": args.dense_component_config,
        "dense_static_pair_ms": dense_ms,
        "dense_old_pair_ms": dense_old_ms,
        "dense_fused_override_pair_ms": dense_fused_override_ms,
        "dense_parallel_override_pair_ms": dense_parallel_override_ms,
        "dense_new_pair_ms": dense_new_ms,
        "dense_dual_pair_ms": dense_dual_ms,
        "static_sparse_ms": sparse_ms,
        "old_dual_sparse_ms": old_ms,
        "fused_override_ms": fused_override_ms,
        "parallel_override_ms": parallel_override_ms,
        "heterogeneous_ms": new_ms,
        "dual_stream_ms": dual_ms,
        "sparse_component_ms": sparse_component_ms,
        "dense_component_ms": dense_component_ms,
        "static_sparse_speedup_vs_dense": dense_ms / sparse_ms,
        "old_speedup_vs_dense": dense_old_ms / old_ms,
        "fused_override_speedup_vs_dense": (
            dense_fused_override_ms / fused_override_ms
        ),
        "parallel_override_speedup_vs_dense": (
            dense_parallel_override_ms / parallel_override_ms
        ),
        "heterogeneous_speedup_vs_dense": dense_new_ms / new_ms,
        "dual_stream_speedup_vs_dense": dense_dual_ms / dual_ms,
        "heterogeneous_speedup_vs_old": old_ms / new_ms,
        "heterogeneous_speedup_vs_parallel_override": (
            parallel_override_ms / new_ms
        ),
        "dual_stream_speedup_vs_old": old_ms / dual_ms,
        "old_max_abs_diff": old_diff,
        "fused_override_max_abs_diff": fused_override_diff,
        "parallel_override_max_abs_diff": parallel_override_diff,
        "heterogeneous_vs_parallel_max_abs_diff": (
            heterogeneous_vs_parallel_diff
        ),
        "heterogeneous_max_abs_diff": new_diff,
        "dual_stream_max_abs_diff": dual_diff,
    }


def write_report(output_root: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Dense + 2:4 Heterogeneous Routing",
        "",
        "The static sparse row is an accuracy-free upper bound. The two exact "
        "rows preserve confidence-selected dense rows.",
        "",
        "| Model | bs/K | Projection | Dense rows | Static 2:4 | Parallel override | One-launch dense+2:4 | New/override |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['batch_size']}/{row['K']} | "
            f"{row['projection']} | {row['dense_rows']} | "
            f"{float(row['static_sparse_speedup_vs_dense']):.3f}x | "
            f"{float(row['parallel_override_speedup_vs_dense']):.3f}x | "
            f"{float(row['heterogeneous_speedup_vs_dense']):.3f}x | "
            f"{float(row['heterogeneous_speedup_vs_parallel_override']):.3f}x |"
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n")


def write_plot(output_root: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"{row['model'].replace('_8b', '')}\nbs{row['batch_size']}/K{row['K']} {row['projection']}"
        for row in rows
    ]
    positions = list(range(len(rows)))
    width = 0.15
    figure, axis = plt.subplots(
        figsize=(max(9.0, 1.25 * len(rows)), 5.0)
    )
    for offset, field, label, color in (
        (-2.5 * width, "static_sparse_speedup_vs_dense", "Whole-batch 2:4 upper", "#457B9D"),
        (-1.5 * width, "old_speedup_vs_dense", "Dual 2:4 exact", "#E9C46A"),
        (-0.5 * width, "fused_override_speedup_vs_dense", "Serial dense override", "#F4A261"),
        (0.5 * width, "parallel_override_speedup_vs_dense", "Parallel dense override", "#D45D79"),
        (1.5 * width, "heterogeneous_speedup_vs_dense", "One-launch dense + 2:4", "#2A9D8F"),
        (2.5 * width, "dual_stream_speedup_vs_dense", "Dual-stream dense + 2:4", "#7A5195"),
    ):
        axis.bar(
            [position + offset for position in positions],
            [float(row[field]) for row in rows],
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(1.0, color="#222222", linewidth=1)
    axis.axhline(1.4, color="#B33F40", linewidth=1, linestyle="--")
    axis.set_ylabel("Kernel speedup vs dense GEMM")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_root / "heterogeneous_routing.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--batch-sizes", default="16")
    parser.add_argument("--k-values", default="6")
    parser.add_argument("--projections", default="qkv,o,gate_up,down")
    parser.add_argument("--dense-ratio", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument(
        "--sparse-accumulator",
        choices=("fp32", "fp16", "fp16_qkv_gate"),
        default="fp32",
    )
    parser.add_argument("--old-config", default="128x32x64_s4_sw4")
    parser.add_argument("--static-config", default="256x64x64_s3")
    parser.add_argument(
        "--heterogeneous-config", default="128x32x64_s4_sw4"
    )
    parser.add_argument(
        "--sparse-component-config", default="128x32x64_s4_sw4"
    )
    parser.add_argument(
        "--dense-component-config", default="128x32x64_s4_sw4"
    )
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["SPECLINK_SPARSE24_ACCUMULATOR"] = args.sparse_accumulator
    models = parse_csv_strings(args.models)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    k_values = parse_csv_ints(args.k_values)
    projections = parse_csv_strings(args.projections)
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result_rows: list[dict[str, object]] = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r}")
        shapes = projection_shapes(model)
        for batch_size in batch_sizes:
            for k in k_values:
                dense_rows, sparse_rows = make_route(
                    batch_size,
                    k,
                    dense_ratio=args.dense_ratio,
                    min_dense_per_request=args.min_dense_per_request,
                    generator=generator,
                )
                for projection in projections:
                    if projection not in shapes:
                        raise ValueError(
                            f"unsupported projection {projection!r}"
                        )
                    row = run_projection(
                        model=model,
                        batch_size=batch_size,
                        k=k,
                        projection=projection,
                        dense_rows=dense_rows,
                        sparse_rows=sparse_rows,
                        generator=generator,
                        args=args,
                    )
                    result_rows.append(row)
                    print(row, flush=True)
                    torch.cuda.empty_cache()

    csv_path = args.output_root / "heterogeneous_routing.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    write_report(args.output_root, result_rows)
    write_plot(args.output_root, result_rows)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
