#!/usr/bin/env python3
"""Benchmark exact mixed-row Gate+SwiGLU -> Down pipelines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_heterogeneous_routing import make_route  # noqa: E402
from bench_sparse24_heterogeneous_swiglu import (  # noqa: E402
    prepare_gate_up,
    silu_and_mul,
)
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
    prepare_cutlass_sparse24_device_gemm,
    sparse24_copy_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_full_sparse_dense_override_linear_prepacked,
    sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
    sparse24_cutlass_heterogeneous_linear_prepacked,
    sparse24_cutlass_heterogeneous_swiglu_prepacked,
)


def prepare_down(
    model: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    weight = torch.randn(
        (intermediate, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    values, meta = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=intermediate,
    )
    return weight, weight24, weight.t().contiguous(), values, meta


def run_case(
    *,
    model: str,
    batch_size: int,
    k: int,
    gate: tuple[torch.Tensor, ...],
    down: tuple[torch.Tensor, ...],
    generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, object]:
    rows = batch_size * (k + 1)
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    dense_rows, sparse_rows = make_route(
        batch_size,
        k,
        dense_ratio=args.dense_ratio,
        min_dense_per_request=args.min_dense_per_request,
        generator=generator,
    )
    dense_rows_long = dense_rows.long()
    sparse_rows_long = sparse_rows.long()
    dense_slots = torch.full(
        (rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows_long] = torch.arange(
        dense_rows.numel(), device="cuda", dtype=torch.int32
    )
    x = torch.randn(
        (rows, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.1)

    dense_gate_up = torch.empty(
        (rows, 2 * intermediate), device="cuda", dtype=torch.float16
    )
    dense_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    dense_output = torch.empty(
        (rows, hidden), device="cuda", dtype=torch.float16
    )
    static_hidden = torch.empty_like(dense_hidden)
    static_output = torch.empty_like(dense_output)
    static_workspace = torch.empty(
        (hidden, rows), device="cuda", dtype=torch.float16
    )
    heterogeneous_hidden = torch.empty_like(dense_hidden)
    heterogeneous_output = torch.empty_like(dense_output)
    override_hidden = torch.empty_like(dense_hidden)
    override_output = torch.empty_like(dense_output)
    full_override_output = torch.empty_like(dense_output)
    parallel_hidden = torch.empty_like(dense_hidden)
    parallel_output = torch.empty_like(dense_output)

    dense_x = torch.empty(
        (int(dense_rows.numel()), hidden),
        device="cuda",
        dtype=torch.float16,
    )
    routed_gate_up = torch.empty(
        (int(dense_rows.numel()), 2 * intermediate),
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

    def dense_gate() -> torch.Tensor:
        torch.mm(x, gate[0], out=dense_gate_up)
        return silu_and_mul(dense_hidden, dense_gate_up)

    def static_gate() -> torch.Tensor:
        return sparse24_cutlass_gate_up_swiglu_prepacked(
            x,
            gate[3],
            gate[4],
            out=static_hidden,
            config=args.static_gate_config,
        )

    def heterogeneous_gate() -> torch.Tensor:
        return sparse24_cutlass_heterogeneous_swiglu_prepacked(
            x,
            gate[3],
            gate[4],
            gate[2],
            gate[5],
            dense_rows,
            sparse_rows,
            out=heterogeneous_hidden,
            config=args.heterogeneous_gate_config,
        )

    def override_gate() -> torch.Tensor:
        return sparse24_cutlass_full_sparse_dense_override_swiglu_prepacked(
            x,
            gate[3],
            gate[4],
            gate[2],
            gate[5],
            dense_rows,
            dense_slots,
            out=override_hidden,
            config=args.override_gate_config,
        )

    def parallel_gate() -> torch.Tensor:
        current = torch.cuda.current_stream()
        sparse_stream.wait_stream(current)
        dense_stream.wait_stream(current)
        with torch.cuda.stream(sparse_stream):
            sparse24_cutlass_gate_up_swiglu_prepacked(
                x,
                gate[3],
                gate[4],
                out=parallel_hidden,
                config=args.static_gate_config,
            )
        with torch.cuda.stream(dense_stream):
            torch.index_select(x, 0, dense_rows_long, out=dense_x)
            torch.mm(dense_x, gate[0], out=routed_gate_up)
            silu_and_mul(routed_hidden, routed_gate_up)
        current.wait_stream(sparse_stream)
        current.wait_stream(dense_stream)
        return sparse24_copy_indexed_rows_contiguous_(
            parallel_hidden,
            routed_hidden,
            dense_rows,
        )

    def mixed_down(
        hidden_input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return sparse24_cutlass_heterogeneous_linear_prepacked(
            hidden_input,
            down[3],
            down[4],
            down[2],
            dense_rows,
            sparse_rows,
            out=output,
            config=args.down_config,
        )

    def override_down(
        hidden_input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return sparse24_cutlass_full_sparse_dense_override_linear_prepacked(
            hidden_input,
            down[3],
            down[4],
            down[2],
            dense_rows,
            dense_slots,
            out=output,
            config=args.override_down_config,
        )

    def dense_mlp() -> torch.Tensor:
        dense_gate()
        return torch.mm(dense_hidden, down[0], out=dense_output)

    def static_mlp() -> torch.Tensor:
        static_gate()
        return sparse24_cutlass_device_gemm_prepacked(
            static_hidden,
            down[3],
            down[4],
            contiguous_output=True,
            out=static_output,
            workspace=static_workspace,
            device_config=args.static_down_config,
        )

    def heterogeneous_mlp() -> torch.Tensor:
        heterogeneous_gate()
        return mixed_down(heterogeneous_hidden, heterogeneous_output)

    def override_mlp() -> torch.Tensor:
        override_gate()
        return mixed_down(override_hidden, override_output)

    def full_override_mlp() -> torch.Tensor:
        override_gate()
        return override_down(override_hidden, full_override_output)

    def parallel_mlp() -> torch.Tensor:
        parallel_gate()
        return override_down(parallel_hidden, parallel_output)

    expected_gate_up = x @ gate[1]
    expected_gate_up[dense_rows_long] = x[dense_rows_long] @ gate[0]
    expected_gate, expected_up = expected_gate_up.chunk(2, dim=-1)
    expected_hidden = (
        torch.nn.functional.silu(expected_gate.float()).half() * expected_up
    ).half()
    expected_output = expected_hidden @ down[1]
    expected_output[dense_rows_long] = (
        expected_hidden[dense_rows_long] @ down[0]
    )
    static_expected_gate_up = x @ gate[1]
    static_gate_part, static_up_part = static_expected_gate_up.chunk(2, dim=-1)
    static_expected_hidden = (
        torch.nn.functional.silu(static_gate_part.float()).half()
        * static_up_part
    ).half()
    static_expected_output = static_expected_hidden @ down[1]

    candidates = {
        "static": (static_mlp, static_output, static_expected_output),
        "heterogeneous": (
            heterogeneous_mlp,
            heterogeneous_output,
            expected_output,
        ),
        "persistent_override": (override_mlp, override_output, expected_output),
        "full_override": (
            full_override_mlp,
            full_override_output,
            expected_output,
        ),
        "parallel_override": (parallel_mlp, parallel_output, expected_output),
    }
    errors: dict[str, float] = {}
    dense_errors: dict[str, float] = {}
    sparse_errors: dict[str, float] = {}
    for name, (candidate, output, expected) in candidates.items():
        candidate()
        torch.cuda.synchronize()
        error = (output.float() - expected.float()).abs()
        errors[name] = float(error.max().item())
        dense_errors[name] = float(
            error.index_select(0, dense_rows_long).max().item()
        )
        sparse_errors[name] = float(
            error.index_select(0, sparse_rows_long).max().item()
        )
        if not torch.allclose(output, expected, rtol=6e-2, atol=3e-2):
            raise RuntimeError(
                f"{name} MLP mismatch: max={errors[name]:.6f} "
                f"dense={dense_errors[name]:.6f} "
                f"sparse={sparse_errors[name]:.6f}"
            )

    timings: dict[str, float] = {}
    for name, (candidate, _, _) in candidates.items():
        dense_ms, candidate_ms = paired_graph_median_ms(
            dense_mlp,
            candidate,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
            graph_warmup_replays=args.graph_warmup_replays,
        )
        timings[f"dense_for_{name}_ms"] = dense_ms
        timings[f"{name}_ms"] = candidate_ms
        timings[f"{name}_speedup_vs_dense"] = dense_ms / candidate_ms

    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": int(dense_rows.numel()),
        "sparse_rows": int(sparse_rows.numel()),
        "static_gate_config": args.static_gate_config,
        "static_down_config": args.static_down_config,
        "heterogeneous_gate_config": args.heterogeneous_gate_config,
        "override_gate_config": args.override_gate_config,
        "override_down_config": args.override_down_config,
        "down_config": args.down_config,
        **timings,
        **{f"{name}_max_abs_diff": value for name, value in errors.items()},
        **{
            f"{name}_dense_max_abs_diff": value
            for name, value in dense_errors.items()
        },
        **{
            f"{name}_sparse_max_abs_diff": value
            for name, value in sparse_errors.items()
        },
    }


def write_outputs(root: Path, rows: list[dict[str, object]]) -> None:
    csv_path = root / "mixed_mlp_pipeline.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Exact Mixed-Row MLP Pipeline",
        "",
        "| Model | BS | K | Rows | Dense | Static | Heterogeneous | "
        "Persistent gate + heterogeneous Down | Full override | "
        "Parallel gate + override Down |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['batch_size']} | {row['K']} | "
            f"{row['rows']} | {row['dense_rows']} | "
            f"{float(row['static_speedup_vs_dense']):.3f}x | "
            f"{float(row['heterogeneous_speedup_vs_dense']):.3f}x | "
            f"{float(row['persistent_override_speedup_vs_dense']):.3f}x | "
            f"{float(row['full_override_speedup_vs_dense']):.3f}x | "
            f"{float(row['parallel_override_speedup_vs_dense']):.3f}x |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt
    import numpy as np

    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    figure, axes = plt.subplots(
        len(models),
        1,
        figsize=(max(10, len(rows) * 0.75), 4.4 * len(models)),
        squeeze=False,
    )
    colors = ("#457B9D", "#2A9D8F", "#B33F40", "#6A4C93", "#E9C46A")
    fields = (
        ("static_speedup_vs_dense", "Whole-batch 2:4"),
        ("heterogeneous_speedup_vs_dense", "Heterogeneous"),
        (
            "persistent_override_speedup_vs_dense",
            "Persistent gate + heterogeneous Down",
        ),
        ("full_override_speedup_vs_dense", "Full override"),
        (
            "parallel_override_speedup_vs_dense",
            "Parallel gate + override Down",
        ),
    )
    for model_index, model in enumerate(models):
        selected = [row for row in rows if row["model"] == model]
        labels = [f"bs{row['batch_size']}/K{row['K']}" for row in selected]
        x = np.arange(len(selected))
        width = 0.16
        for index, ((field, label), color) in enumerate(zip(fields, colors)):
            axes[model_index][0].bar(
                x + (index - 2.0) * width,
                [float(row[field]) for row in selected],
                width,
                label=label,
                color=color,
            )
        axes[model_index][0].axhline(1.0, color="#333333", linewidth=1)
        axes[model_index][0].set_title(model)
        axes[model_index][0].set_ylabel("Speedup vs dense MLP")
        axes[model_index][0].set_xticks(x, labels, rotation=30, ha="right")
        axes[model_index][0].grid(axis="y", alpha=0.2)
        axes[model_index][0].legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "mixed_mlp_pipeline.png", dpi=180)
    plt.close(figure)
    print(f"wrote {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument("--batch-sizes", default="16")
    parser.add_argument("--k-values", default="6")
    parser.add_argument("--dense-ratio", type=float, default=0.16)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument(
        "--static-gate-config", default="256x64x64_s3_sw4_f16"
    )
    parser.add_argument("--static-down-config", default="auto")
    parser.add_argument(
        "--heterogeneous-gate-config", default="256x32x64_s3_sw4_f16"
    )
    parser.add_argument(
        "--override-gate-config", default="256x64_sparse_128x64_dense_f16"
    )
    parser.add_argument(
        "--override-down-config", default="256x64_sparse_128x64_dense_f16"
    )
    parser.add_argument("--down-config", default="auto")
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--graph-warmup-replays", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for model_index, model in enumerate(parse_csv_strings(args.models)):
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r}")
        generator = torch.Generator(device="cuda").manual_seed(
            args.seed + model_index
        )
        gate = prepare_gate_up(model, generator)
        down = prepare_down(model, generator)
        for batch_size in parse_csv_ints(args.batch_sizes):
            for k in parse_csv_ints(args.k_values):
                row = run_case(
                    model=model,
                    batch_size=batch_size,
                    k=k,
                    gate=gate,
                    down=down,
                    generator=generator,
                    args=args,
                )
                results.append(row)
                print(row, flush=True)
        del gate, down
        torch.cuda.empty_cache()
    write_outputs(args.output_root, results)


if __name__ == "__main__":
    main()
