#!/usr/bin/env python3
"""Benchmark split-K for the compact R24 QKV residual projection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_sparse24_heterogeneous_routing import prepare_weight  # noqa: E402
from bench_sparse24_indexed_down_epilogue import (  # noqa: E402
    paired_graph_median_ms,
    parse_csv_ints,
)
from vllm.speclink_kernel import (  # noqa: E402
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_device_splitk_gemm_prepacked,
)


HIDDEN_SIZE = 4096
QKV_SIZE = 6144


def run_case(
    dense_rows: int,
    split_k_slices: int,
    *,
    generator: torch.Generator,
    residual_values: torch.Tensor,
    residual_meta: torch.Tensor,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> dict[str, object]:
    run_rows = (dense_rows + 7) // 8 * 8
    x = torch.randn(
        (run_rows, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    ).mul_(0.1)
    baseline_out = torch.empty_strided(
        (run_rows, QKV_SIZE),
        (1, run_rows),
        device="cuda",
        dtype=torch.float16,
    )
    split_out = torch.empty_strided(
        (run_rows, QKV_SIZE),
        (1, run_rows),
        device="cuda",
        dtype=torch.float16,
    )
    workspace = torch.zeros(
        ((QKV_SIZE + 255) // 256) * ((run_rows + 31) // 32),
        device="cuda",
        dtype=torch.int32,
    )

    def baseline() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            x,
            residual_values,
            residual_meta,
            out=baseline_out,
            device_config="auto",
        )

    def splitk() -> torch.Tensor:
        return sparse24_cutlass_device_splitk_gemm_prepacked(
            x,
            residual_values,
            residual_meta,
            split_k_slices=split_k_slices,
            out=split_out,
            workspace=workspace,
        )

    expected = baseline().clone()
    actual = splitk().clone()
    torch.cuda.synchronize()
    max_abs_diff = float(
        (actual.float() - expected.float()).abs().max().item()
    )
    if not torch.allclose(actual, expected, rtol=3e-2, atol=1e-1):
        raise RuntimeError(
            "split-K QKV residual mismatch: "
            f"dense_rows={dense_rows}, run_rows={run_rows}, "
            f"split_k={split_k_slices}, max_abs_diff={max_abs_diff}"
        )
    if int(workspace.abs().max().item()) != 0:
        raise RuntimeError("split-K workspace was not reset after eager run")

    baseline_ms, splitk_ms = paired_graph_median_ms(
        baseline,
        splitk,
        unroll=unroll,
        replays=replays,
        trials=trials,
        graph_warmup_replays=graph_warmup_replays,
    )
    torch.cuda.synchronize()
    workspace_max_after_graph = int(workspace.abs().max().item())
    if workspace_max_after_graph != 0:
        raise RuntimeError(
            "split-K workspace was not reset after graph replay: "
            f"max={workspace_max_after_graph}"
        )
    return {
        "dense_rows": dense_rows,
        "run_rows": run_rows,
        "split_k_slices": split_k_slices,
        "baseline_ms": baseline_ms,
        "splitk_ms": splitk_ms,
        "splitk_speedup": baseline_ms / splitk_ms,
        "max_abs_diff": max_abs_diff,
        "workspace_max_after_graph": workspace_max_after_graph,
    }


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    colors = {2: "#0072B2", 4: "#D55E00", 8: "#009E73"}
    for split_k in sorted({int(row["split_k_slices"]) for row in rows}):
        selected = [
            row for row in rows if int(row["split_k_slices"]) == split_k
        ]
        axis.plot(
            [int(row["dense_rows"]) for row in selected],
            [float(row["splitk_speedup"]) for row in selected],
            marker="o",
            color=colors[split_k],
            label=f"split-K {split_k}",
        )
    axis.axhline(1.0, color="#555555", linewidth=1, linestyle=":")
    axis.set_xlabel("Selected dense rows")
    axis.set_ylabel("Residual GEMM speedup")
    axis.set_title("QKV R24 split-K, K=4096, N=6144")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-rows",
        type=parse_csv_ints,
        default=(16, 20, 24, 32, 40, 50, 60, 64),
    )
    parser.add_argument(
        "--split-k-slices", type=parse_csv_ints, default=(2, 4, 8)
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--unroll", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=30)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(value <= 0 or value > 64 for value in args.dense_rows):
        raise ValueError("--dense-rows values must be in [1, 64]")
    if any(value not in (2, 4, 8) for value in args.split_k_slices):
        raise ValueError("--split-k-slices values must be 2, 4, or 8")

    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    (
        _weight,
        _weight24,
        _sparse_values,
        _sparse_meta,
        residual_values,
        residual_meta,
    ) = prepare_weight(HIDDEN_SIZE, QKV_SIZE, generator)

    results: list[dict[str, object]] = []
    for dense_rows in args.dense_rows:
        for split_k_slices in args.split_k_slices:
            result = run_case(
                dense_rows,
                split_k_slices,
                generator=generator,
                residual_values=residual_values,
                residual_meta=residual_meta,
                unroll=args.unroll,
                replays=args.replays,
                trials=args.trials,
                graph_warmup_replays=args.graph_warmup_replays,
            )
            results.append(result)
            print(
                f"dense={dense_rows} run={int(result['run_rows'])} "
                f"split_k={split_k_slices} "
                f"baseline={float(result['baseline_ms']):.5f}ms "
                f"split={float(result['splitk_ms']):.5f}ms "
                f"speedup={float(result['splitk_speedup']):.3f}x",
                flush=True,
            )

    csv_path = args.output_root / "qkv_residual_splitk.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_plot(args.output_root / "qkv_residual_splitk.png", results)
    print(args.output_root)


if __name__ == "__main__":
    main()
