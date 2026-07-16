#!/usr/bin/env python3
"""Benchmark one-launch heterogeneous Gate/Up + SwiGLU routing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_heterogeneous_routing import make_route  # noqa: E402
from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    MODELS,
    paired_graph_median_ms,
    parse_csv_ints,
    parse_csv_strings,
)
from vllm import _custom_ops as _vllm_ops  # noqa: E402,F401
from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_gate_up_swiglu,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
    sparse24_cutlass_heterogeneous_swiglu_prepacked,
)


def prepare_gate_up(
    model: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    output = 2 * intermediate
    weight = torch.randn(
        (hidden, output),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    values, meta = prepare_cutlass_sparse24_gate_up_swiglu(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=hidden,
    )
    pair_channels = 64
    gate_rows = torch.arange(
        intermediate, device="cuda", dtype=torch.int32
    ).reshape(-1, pair_channels)
    dense_weight_rows = torch.cat(
        (gate_rows, gate_rows + intermediate), dim=1
    ).flatten().contiguous()
    return (
        weight,
        weight24,
        weight.t().contiguous(),
        values,
        meta,
        dense_weight_rows,
    )


def silu_and_mul(out: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
    torch.ops._C.silu_and_mul(out, gate_up)
    return out


def run_case(
    *,
    model: str,
    batch_size: int,
    k: int,
    weight: torch.Tensor,
    weight24: torch.Tensor,
    dense_weight: torch.Tensor,
    values: torch.Tensor,
    meta: torch.Tensor,
    dense_weight_rows: torch.Tensor,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    rows = batch_size * (k + 1)
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    output = 2 * intermediate
    dense_rows, sparse_rows = make_route(
        batch_size,
        k,
        dense_ratio=args.dense_ratio,
        min_dense_per_request=args.min_dense_per_request,
        generator=generator,
    )
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows.long()] = torch.arange(
        dense_rows.numel(), device="cuda", dtype=torch.int32
    )
    dense_rows_long = dense_rows.long()
    x = torch.randn(
        (rows, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.1)

    dense_gate_up = torch.empty(
        (rows, output), device="cuda", dtype=torch.float16
    )
    dense_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    static_hidden = torch.empty_like(dense_hidden)
    parallel_hidden = torch.empty_like(dense_hidden)
    heterogeneous_hidden = torch.empty_like(dense_hidden)
    override_hidden = torch.empty_like(dense_hidden)
    dense_x = torch.empty(
        (int(dense_rows.numel()), hidden),
        device="cuda",
        dtype=torch.float16,
    )
    routed_gate_up = torch.empty(
        (int(dense_rows.numel()), output),
        device="cuda",
        dtype=torch.float16,
    )
    routed_hidden = torch.empty(
        (int(dense_rows.numel()), intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    sparse_stream = torch.cuda.Stream()
    dense_stream = torch.cuda.Stream()

    def dense_fn() -> torch.Tensor:
        torch.mm(x, weight, out=dense_gate_up)
        return silu_and_mul(dense_hidden, dense_gate_up)

    def static_fn() -> torch.Tensor:
        return sparse24_cutlass_gate_up_swiglu_prepacked(
            x,
            values,
            meta,
            out=static_hidden,
            config=args.static_config,
        )

    def parallel_override_fn() -> torch.Tensor:
        current = torch.cuda.current_stream()
        sparse_stream.wait_stream(current)
        dense_stream.wait_stream(current)
        with torch.cuda.stream(sparse_stream):
            sparse24_cutlass_gate_up_swiglu_prepacked(
                x,
                values,
                meta,
                out=parallel_hidden,
                config=args.static_config,
            )
        with torch.cuda.stream(dense_stream):
            torch.index_select(x, 0, dense_rows_long, out=dense_x)
            torch.mm(dense_x, weight, out=routed_gate_up)
            silu_and_mul(routed_hidden, routed_gate_up)
        current.wait_stream(sparse_stream)
        current.wait_stream(dense_stream)
        sparse24_copy_indexed_rows_contiguous_(
            parallel_hidden,
            routed_hidden,
            dense_rows,
        )
        return parallel_hidden

    def heterogeneous_fn() -> torch.Tensor:
        return sparse24_cutlass_heterogeneous_swiglu_prepacked(
            x,
            values,
            meta,
            dense_weight,
            dense_weight_rows,
            dense_rows,
            sparse_rows,
            out=heterogeneous_hidden,
            config=args.heterogeneous_config,
        )

    def full_sparse_dense_override_fn() -> torch.Tensor:
        return sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked(
            x,
            values,
            meta,
            dense_weight,
            dense_weight_rows,
            dense_rows,
            dense_slots,
            out=override_hidden,
            config=args.override_config,
        )

    measure = lambda baseline, candidate: paired_graph_median_ms(
        baseline,
        candidate,
        unroll=args.unroll,
        replays=args.replays,
        trials=args.trials,
        graph_warmup_replays=args.graph_warmup_replays,
    )
    dense_static_ms, static_ms = measure(dense_fn, static_fn)
    dense_parallel_ms, parallel_ms = measure(dense_fn, parallel_override_fn)
    dense_heterogeneous_ms, heterogeneous_ms = measure(
        dense_fn, heterogeneous_fn
    )
    dense_override_ms, override_ms = measure(
        dense_fn, full_sparse_dense_override_fn
    )

    static_fn()
    parallel_override_fn()
    heterogeneous_fn()
    full_sparse_dense_override_fn()
    expected_gate_up = x @ weight24
    expected_gate_up[dense_rows_long] = x[dense_rows_long] @ weight
    gate, up = expected_gate_up.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.float()).half() * up).half()
    torch.cuda.synchronize()
    parallel_diff = float((parallel_hidden - expected).abs().max().item())
    heterogeneous_error = (heterogeneous_hidden - expected).abs()
    heterogeneous_diff = float(heterogeneous_error.max().item())
    heterogeneous_dense_diff = float(
        heterogeneous_error.index_select(0, dense_rows_long).max().item()
    )
    heterogeneous_sparse_diff = float(
        heterogeneous_error.index_select(0, sparse_rows.long()).max().item()
    )
    override_error = (override_hidden - expected).abs()
    override_diff = float(override_error.max().item())
    override_dense_diff = float(
        override_error.index_select(0, dense_rows_long).max().item()
    )
    override_sparse_diff = float(
        override_error.index_select(0, sparse_rows.long()).max().item()
    )
    if not torch.allclose(
        heterogeneous_hidden, expected, rtol=5e-2, atol=1e-1
    ):
        raise AssertionError(
            "heterogeneous SwiGLU mismatch: "
            f"max={heterogeneous_diff} "
            f"dense={heterogeneous_dense_diff} "
            f"sparse={heterogeneous_sparse_diff}"
        )
    if not torch.allclose(override_hidden, expected, rtol=5e-2, atol=1e-1):
        raise AssertionError(
            "full-sparse dense-override SwiGLU mismatch: "
            f"max={override_diff} "
            f"dense={override_dense_diff} "
            f"sparse={override_sparse_diff}"
        )
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": int(dense_rows.numel()),
        "sparse_rows": int(sparse_rows.numel()),
        "static_config": args.static_config,
        "heterogeneous_config": args.heterogeneous_config,
        "override_config": args.override_config,
        "dense_static_pair_ms": dense_static_ms,
        "dense_parallel_pair_ms": dense_parallel_ms,
        "dense_heterogeneous_pair_ms": dense_heterogeneous_ms,
        "dense_override_pair_ms": dense_override_ms,
        "static_sparse_swiglu_ms": static_ms,
        "parallel_override_swiglu_ms": parallel_ms,
        "heterogeneous_swiglu_ms": heterogeneous_ms,
        "full_sparse_dense_override_swiglu_ms": override_ms,
        "static_speedup_vs_dense": dense_static_ms / static_ms,
        "parallel_speedup_vs_dense": dense_parallel_ms / parallel_ms,
        "heterogeneous_speedup_vs_dense": (
            dense_heterogeneous_ms / heterogeneous_ms
        ),
        "heterogeneous_speedup_vs_parallel": parallel_ms / heterogeneous_ms,
        "override_speedup_vs_dense": dense_override_ms / override_ms,
        "override_speedup_vs_parallel": parallel_ms / override_ms,
        "parallel_max_abs_diff": parallel_diff,
        "heterogeneous_max_abs_diff": heterogeneous_diff,
        "heterogeneous_dense_max_abs_diff": heterogeneous_dense_diff,
        "heterogeneous_sparse_max_abs_diff": heterogeneous_sparse_diff,
        "override_max_abs_diff": override_diff,
        "override_dense_max_abs_diff": override_dense_diff,
        "override_sparse_max_abs_diff": override_sparse_diff,
    }


def write_outputs(root: Path, rows: list[dict[str, object]]) -> None:
    csv_path = root / "heterogeneous_swiglu.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Heterogeneous Gate/Up + SwiGLU",
        "",
        "| Model | BS | K | Rows | Dense rows | Static vs dense | "
        "Parallel vs dense | Heterogeneous vs dense | Override vs dense | "
        "Override vs parallel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['batch_size']} | {row['K']} | "
            f"{row['rows']} | {row['dense_rows']} | "
            f"{float(row['static_speedup_vs_dense']):.3f}x | "
            f"{float(row['parallel_speedup_vs_dense']):.3f}x | "
            f"{float(row['heterogeneous_speedup_vs_dense']):.3f}x | "
            f"{float(row['override_speedup_vs_dense']):.3f}x | "
            f"{float(row['override_speedup_vs_parallel']):.3f}x |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"bs{row['batch_size']}/K{row['K']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.2
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 1.25), 4.8))
    for offset, key, label, color in (
        (-1.5 * width, "static_speedup_vs_dense", "Whole-batch 2:4", "#457B9D"),
        (-0.5 * width, "parallel_speedup_vs_dense", "Parallel override", "#E9C46A"),
        (0.5 * width, "heterogeneous_speedup_vs_dense", "Heterogeneous fused", "#2A9D8F"),
        (1.5 * width, "override_speedup_vs_dense", "Persistent override", "#B33F40"),
    ):
        axis.bar(
            x + offset,
            [float(row[key]) for row in rows],
            width,
            label=label,
            color=color,
        )
    axis.axhline(1.0, color="#333333", linewidth=1)
    axis.set_ylabel("Speedup vs dense Gate/Up + SwiGLU")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(root / "heterogeneous_swiglu.png", dpi=180)
    plt.close(figure)
    print(f"wrote {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument("--batch-sizes", default="16")
    parser.add_argument("--k-values", default="6")
    parser.add_argument("--dense-ratio", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument(
        "--static-config", default="256x64x64_s3_sw4_f16"
    )
    parser.add_argument(
        "--heterogeneous-config", default="256x32x64_s3_sw4_f16"
    )
    parser.add_argument(
        "--override-config",
        default="auto",
        choices=(
            "auto",
            "256x32_sparse_128x32_dense_f16",
            "256x64_sparse_128x64_dense_f16",
            "256x64_sparse_128x64_dense_f16_w1",
            "256x64_sparse_128x64_dense_f16_w3",
            "256x64_sparse_128x64_dense_f16_w4",
        ),
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

    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    results: list[dict[str, object]] = []
    for model in parse_csv_strings(args.models):
        prepared = prepare_gate_up(model, generator)
        for batch_size in parse_csv_ints(args.batch_sizes):
            for k in parse_csv_ints(args.k_values):
                row = run_case(
                    model=model,
                    batch_size=batch_size,
                    k=k,
                    weight=prepared[0],
                    weight24=prepared[1],
                    dense_weight=prepared[2],
                    values=prepared[3],
                    meta=prepared[4],
                    dense_weight_rows=prepared[5],
                    generator=generator,
                    args=args,
                )
                results.append(row)
                print(row, flush=True)
    write_outputs(args.output_root, results)


if __name__ == "__main__":
    main()
