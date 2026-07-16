#!/usr/bin/env python3
"""Benchmark sparse gate/up GEMM with an inline SwiGLU visitor epilogue."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    prepare_cutlass_sparse24_gate_up_swiglu,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
    sparse24_silu_and_mul_transposed,
    sparse24_silu_and_mul_transposed_to_contiguous,
)


MODELS = {
    "qwen3_8b": {"model_width": 4096, "intermediate": 12288},
    "llama3_1_8b": {"model_width": 4096, "intermediate": 14336},
}
DEFAULT_ROWS = "112,144,176,224,288,352,448,576,704"
DEFAULT_B_ROW_CONFIGS = (
    "64x64x64,64x64x64_s4,64x64x64_s5,64x64x64_s6,64x64x64_s7,"
    "128x32x64_s4,128x32x64_s4_sw2,128x32x64_s4_sw4,"
    "128x64x64_s4,128x64x64_s5,256x64x64_s3"
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def capture_unrolled(fn: Callable[[], object], unroll: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    torch.cuda.synchronize()
    return graph


def graph_milliseconds(
    graph: torch.cuda.CUDAGraph,
    *,
    unroll: int,
    replays: int,
    trials: int,
) -> float:
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (replays * unroll))
    return statistics.median(samples)


def make_sparse_prepacked(
    rows: int,
    columns: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (rows, columns),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight.mul_(0.02)
    weight.add_(torch.where(weight >= 0, 0.005, -0.005))
    weight24, _ = apply_random_24_mask(weight, generator=generator)
    packed = pack_24(weight24, layout="n_major")
    return weight24, packed.values, packed.meta


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = MODELS[args.model]
    model_width = shape["model_width"]
    intermediate = shape["intermediate"]
    gate_up_size = 2 * intermediate
    transposed_fused_config = (
        args.fused_config
        if args.fused_config
        in {"auto", "256x64x64_s3", "256x64x64_s3_sw4"}
        else "auto"
    )

    gate_up_weight, gate_up_values_raw, gate_up_meta_raw = make_sparse_prepacked(
        model_width, gate_up_size, generator
    )
    gate_up_values, gate_up_meta = prepare_cutlass_sparse24_device_gemm(
        gate_up_values_raw,
        gate_up_meta_raw,
        layout="n_major",
        K=model_width,
    )
    fused_values, fused_meta = prepare_cutlass_sparse24_gate_up_swiglu(
        gate_up_values_raw,
        gate_up_meta_raw,
        layout="n_major",
        K=model_width,
    )
    del gate_up_weight, gate_up_values_raw, gate_up_meta_raw

    down_weight = down_values = down_meta = None
    if args.with_down:
        if args.down_kind == "sparse":
            down_weight, down_values_raw, down_meta_raw = make_sparse_prepacked(
                intermediate, model_width, generator
            )
            down_values, down_meta = prepare_cutlass_sparse24_device_gemm(
                down_values_raw,
                down_meta_raw,
                layout="n_major",
                K=intermediate,
            )
            del down_values_raw, down_meta_raw
        else:
            down_weight = torch.randn(
                (model_width, intermediate),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            down_weight.mul_(0.02)

    results: list[dict[str, object]] = []
    for rows in args.rows:
        x = torch.randn(
            (rows, model_width),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        gate_up_transposed = torch.empty_strided(
            (rows, gate_up_size),
            (1, rows),
            device="cuda",
            dtype=torch.float16,
        )
        hidden_transposed = torch.empty_strided(
            (rows, intermediate),
            (1, rows),
            device="cuda",
            dtype=torch.float16,
        )
        hidden_contiguous = torch.empty(
            (rows, intermediate), device="cuda", dtype=torch.float16
        )
        hidden_fused = torch.empty_like(hidden_contiguous)
        hidden_fused_transposed = torch.empty_strided(
            (rows, intermediate),
            (1, rows),
            device="cuda",
            dtype=torch.float16,
        )

        def baseline_transposed_fn() -> torch.Tensor:
            sparse24_cutlass_device_gemm_prepacked(
                x,
                gate_up_values,
                gate_up_meta,
                out=gate_up_transposed,
                device_config="auto",
            )
            return sparse24_silu_and_mul_transposed(
                gate_up_transposed, out=hidden_transposed
            )

        def baseline_contiguous_fn() -> torch.Tensor:
            sparse24_cutlass_device_gemm_prepacked(
                x,
                gate_up_values,
                gate_up_meta,
                out=gate_up_transposed,
                device_config="auto",
            )
            return sparse24_silu_and_mul_transposed_to_contiguous(
                gate_up_transposed, out=hidden_contiguous
            )

        def fused_fn() -> torch.Tensor:
            return sparse24_cutlass_gate_up_swiglu_prepacked(
                x,
                fused_values,
                fused_meta,
                out=hidden_fused,
                config=args.fused_config,
            )

        def fused_transposed_fn() -> torch.Tensor:
            return sparse24_cutlass_gate_up_swiglu_prepacked(
                x,
                fused_values,
                fused_meta,
                out=hidden_fused_transposed,
                config=transposed_fused_config,
                output_transposed=True,
            )

        baseline_contiguous_fn()
        fused_fn()
        fused_transposed_fn()
        torch.cuda.synchronize()
        max_abs_diff = float(
            (hidden_contiguous.float() - hidden_fused.float()).abs().max().item()
        )
        if not torch.allclose(
            hidden_contiguous, hidden_fused, rtol=args.rtol, atol=args.atol
        ):
            raise RuntimeError(
                f"fused SwiGLU mismatch for M={rows}: max_abs_diff={max_abs_diff}"
            )
        transposed_diff = float(
            (hidden_contiguous.float() - hidden_fused_transposed.float())
            .abs()
            .max()
            .item()
        )
        if not torch.allclose(
            hidden_contiguous,
            hidden_fused_transposed,
            rtol=args.rtol,
            atol=args.atol,
        ):
            raise RuntimeError(
                f"transposed fused SwiGLU mismatch for M={rows}: "
                f"max_abs_diff={transposed_diff}"
            )

        baseline_transposed_graph = capture_unrolled(
            baseline_transposed_fn, args.unroll
        )
        baseline_contiguous_graph = capture_unrolled(
            baseline_contiguous_fn, args.unroll
        )
        fused_graph = capture_unrolled(fused_fn, args.unroll)
        fused_transposed_graph = capture_unrolled(
            fused_transposed_fn, args.unroll
        )
        baseline_transposed_ms = graph_milliseconds(
            baseline_transposed_graph,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
        )
        baseline_contiguous_ms = graph_milliseconds(
            baseline_contiguous_graph,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
        )
        fused_ms = graph_milliseconds(
            fused_graph,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
        )
        fused_transposed_ms = graph_milliseconds(
            fused_transposed_graph,
            unroll=args.unroll,
            replays=args.replays,
            trials=args.trials,
        )

        full_baseline_ms: float | None = None
        full_fused_ms: float | None = None
        full_speedup: float | None = None
        full_fused_transposed_ms: float | None = None
        full_transposed_speedup: float | None = None
        if args.with_down:
            assert down_weight is not None
            baseline_output = torch.empty(
                (rows, model_width), device="cuda", dtype=torch.float16
            )
            fused_output = torch.empty_like(baseline_output)
            fused_transposed_output = torch.empty_like(baseline_output)
            baseline_workspace = torch.empty(
                (model_width, rows), device="cuda", dtype=torch.float16
            )
            fused_workspace = torch.empty_like(baseline_workspace)
            fused_transposed_workspace = torch.empty_like(baseline_workspace)

            if args.down_kind == "sparse":
                assert down_values is not None and down_meta is not None

                def full_baseline_fn() -> torch.Tensor:
                    baseline_transposed_fn()
                    return sparse24_cutlass_device_gemm_prepacked(
                        hidden_transposed,
                        down_values,
                        down_meta,
                        contiguous_output=True,
                        input_transposed=True,
                        out=baseline_output,
                        workspace=baseline_workspace,
                        device_config="auto",
                    )

                def full_fused_fn() -> torch.Tensor:
                    fused_fn()
                    return sparse24_cutlass_device_gemm_prepacked(
                        hidden_fused,
                        down_values,
                        down_meta,
                        contiguous_output=True,
                        out=fused_output,
                        workspace=fused_workspace,
                        device_config="auto",
                    )

                def full_fused_transposed_fn() -> torch.Tensor:
                    fused_transposed_fn()
                    return sparse24_cutlass_device_gemm_prepacked(
                        hidden_fused_transposed,
                        down_values,
                        down_meta,
                        contiguous_output=True,
                        input_transposed=True,
                        out=fused_transposed_output,
                        workspace=fused_transposed_workspace,
                        device_config="auto",
                    )

            else:
                down_weight_t = down_weight.t()

                def full_baseline_fn() -> torch.Tensor:
                    baseline_contiguous_fn()
                    return torch.mm(
                        hidden_contiguous,
                        down_weight_t,
                        out=baseline_output,
                    )

                def full_fused_fn() -> torch.Tensor:
                    fused_fn()
                    return torch.mm(
                        hidden_fused,
                        down_weight_t,
                        out=fused_output,
                    )

                def full_fused_transposed_fn() -> torch.Tensor:
                    fused_transposed_fn()
                    return torch.mm(
                        hidden_fused_transposed,
                        down_weight_t,
                        out=fused_transposed_output,
                    )

            full_baseline_fn()
            full_fused_fn()
            torch.cuda.synchronize()
            if args.validate_down:
                if args.down_kind == "sparse":
                    baseline_reference = hidden_transposed.contiguous() @ down_weight
                    fused_reference = hidden_fused @ down_weight
                else:
                    baseline_reference = hidden_contiguous @ down_weight.t()
                    fused_reference = hidden_fused @ down_weight.t()
                torch.cuda.synchronize()
                baseline_reference_diff = float(
                    (baseline_output.float() - baseline_reference.float())
                    .abs()
                    .max()
                    .item()
                )
                fused_reference_diff = float(
                    (fused_output.float() - fused_reference.float())
                    .abs()
                    .max()
                    .item()
                )
                print(
                    f"down validation M={rows}: "
                    f"b_row_max_diff={baseline_reference_diff:.6f} "
                    f"normal_max_diff={fused_reference_diff:.6f}",
                    flush=True,
                )
                for config in (
                    args.validate_b_row_configs
                    if args.down_kind == "sparse"
                    else []
                ):
                    candidate_output = torch.empty_like(baseline_output)
                    candidate_workspace = torch.empty_like(baseline_workspace)
                    sparse24_cutlass_device_gemm_prepacked(
                        hidden_transposed,
                        down_values,
                        down_meta,
                        contiguous_output=True,
                        input_transposed=True,
                        out=candidate_output,
                        workspace=candidate_workspace,
                        device_config=config,
                    )
                    torch.cuda.synchronize()
                    candidate_diff = float(
                        (candidate_output.float() - baseline_reference.float())
                        .abs()
                        .max()
                        .item()
                    )
                    print(
                        f"b_row config={config} max_diff={candidate_diff:.6f}",
                        flush=True,
                    )
            if not torch.allclose(
                baseline_output, fused_output, rtol=5e-2, atol=2e-1
            ):
                full_diff = float(
                    (baseline_output.float() - fused_output.float())
                    .abs()
                    .max()
                    .item()
                )
                full_mean_diff = float(
                    (baseline_output.float() - fused_output.float())
                    .abs()
                    .mean()
                    .item()
                )
                baseline_abs_max = float(
                    baseline_output.float().abs().max().item()
                )
                fused_abs_max = float(fused_output.float().abs().max().item())
                raise RuntimeError(
                    f"full MLP mismatch for M={rows}: max_abs_diff={full_diff}, "
                    f"mean_abs_diff={full_mean_diff}, "
                    f"baseline_abs_max={baseline_abs_max}, "
                    f"fused_abs_max={fused_abs_max}, hidden_diff={max_abs_diff}"
                )
            full_fused_transposed_fn()
            torch.cuda.synchronize()
            if not torch.allclose(
                baseline_output,
                fused_transposed_output,
                rtol=5e-2,
                atol=2e-1,
            ):
                full_transposed_diff = float(
                    (baseline_output.float() - fused_transposed_output.float())
                    .abs()
                    .max()
                    .item()
                )
                raise RuntimeError(
                    f"transposed full MLP mismatch for M={rows}: "
                    f"max_abs_diff={full_transposed_diff}"
                )
            full_baseline_graph = capture_unrolled(full_baseline_fn, args.unroll)
            full_fused_graph = capture_unrolled(full_fused_fn, args.unroll)
            full_fused_transposed_graph = capture_unrolled(
                full_fused_transposed_fn, args.unroll
            )
            full_baseline_ms = graph_milliseconds(
                full_baseline_graph,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
            full_fused_ms = graph_milliseconds(
                full_fused_graph,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
            full_speedup = full_baseline_ms / full_fused_ms
            full_fused_transposed_ms = graph_milliseconds(
                full_fused_transposed_graph,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
            )
            full_transposed_speedup = (
                full_baseline_ms / full_fused_transposed_ms
            )

        result = {
            "model": args.model,
            "down_kind": args.down_kind if args.with_down else "none",
            "fused_config": args.fused_config,
            "transposed_fused_config": transposed_fused_config,
            "M": rows,
            "K": model_width,
            "gate_up_N": gate_up_size,
            "baseline_transposed_ms": baseline_transposed_ms,
            "baseline_contiguous_ms": baseline_contiguous_ms,
            "fused_swiglu_ms": fused_ms,
            "fused_swiglu_transposed_ms": fused_transposed_ms,
            "speedup_vs_transposed": baseline_transposed_ms / fused_ms,
            "speedup_vs_contiguous": baseline_contiguous_ms / fused_ms,
            "transposed_fused_speedup": (
                baseline_transposed_ms / fused_transposed_ms
            ),
            "full_mlp_baseline_ms": full_baseline_ms,
            "full_mlp_fused_ms": full_fused_ms,
            "full_mlp_speedup": full_speedup,
            "full_mlp_fused_transposed_ms": full_fused_transposed_ms,
            "full_mlp_transposed_speedup": full_transposed_speedup,
            "max_abs_diff": max_abs_diff,
            "transposed_max_abs_diff": transposed_diff,
        }
        results.append(result)
        full_text = (
            f" full_mlp={full_speedup:.3f}x"
            f" full_mlp_t={full_transposed_speedup:.3f}x"
            if full_speedup is not None and full_transposed_speedup is not None
            else ""
        )
        print(
            f"{args.model} M={rows} baseline_t={baseline_transposed_ms:.4f} ms "
            f"baseline_c={baseline_contiguous_ms:.4f} ms "
            f"fused={fused_ms:.4f} ms "
            f"fused_t={fused_transposed_ms:.4f} ms "
            f"speedup_t={result['speedup_vs_transposed']:.3f}x "
            f"speedup_c={result['speedup_vs_contiguous']:.3f}x "
            f"transposed_fused={result['transposed_fused_speedup']:.3f}x"
            f"{full_text}",
            flush=True,
        )

    return results


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    m_values = [int(row["M"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        m_values,
        [float(row["speedup_vs_transposed"]) for row in rows],
        marker="o",
        label="Inline contiguous vs transposed baseline",
    )
    axis.plot(
        m_values,
        [float(row["speedup_vs_contiguous"]) for row in rows],
        marker="s",
        label="Inline contiguous vs contiguous baseline",
    )
    axis.plot(
        m_values,
        [float(row["transposed_fused_speedup"]) for row in rows],
        marker="D",
        label="Inline transposed vs transposed baseline",
    )
    if all(row["full_mlp_speedup"] is not None for row in rows):
        axis.plot(
            m_values,
            [float(row["full_mlp_speedup"]) for row in rows],
            marker="^",
            label="Full MLP, inline contiguous",
        )
        axis.plot(
            m_values,
            [float(row["full_mlp_transposed_speedup"]) for row in rows],
            marker="v",
            label="Full MLP, inline transposed",
        )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Verification rows (M)")
    axis.set_ylabel("Speedup")
    axis.set_title(
        f"{rows[0]['model']} inline sparse SwiGLU "
        f"({rows[0]['down_kind']} Down)"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen3_8b")
    parser.add_argument(
        "--rows", type=parse_csv_ints, default=parse_csv_ints(DEFAULT_ROWS)
    )
    parser.add_argument(
        "--with-down", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--down-kind", choices=("sparse", "dense"), default="sparse"
    )
    parser.add_argument(
        "--fused-config",
        choices=(
            "auto",
            "256x32x64_s3",
            "256x32x64_s3_sw4",
            "256x32x64_s3_sw4_f16",
            "256x64x64_s3",
            "256x64x64_s3_sw4",
            "256x64x64_s3_sw4_f16",
        ),
        default="auto",
    )
    parser.add_argument(
        "--validate-down", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--validate-b-row-configs",
        type=parse_csv_strings,
        default=parse_csv_strings(DEFAULT_B_ROW_CONFIGS),
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "swiglu_epilogue_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = args.output_root / "swiglu_epilogue_speedup.png"
    write_plot(plot_path, rows)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
