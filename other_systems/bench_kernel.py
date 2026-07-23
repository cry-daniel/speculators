#!/usr/bin/env python3
"""Formal BF16 N:M kernel benchmark for Flash-LLM, SparTA, and SpInfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from other_systems import benchmark_common as common
from other_systems import parse_nm


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    methods = ("flash_llm", "spinfer", "sparta")
    formats = tuple(dict.fromkeys(str(row["nm_format"]) for row in rows))
    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    figure, axes = plt.subplots(
        len(models), len(formats), figsize=(13, 7), squeeze=False, sharey=True
    )
    colors = {
        "flash_llm": "#4c78a8",
        "spinfer": "#f58518",
        "sparta": "#54a24b",
    }
    for row_index, model in enumerate(models):
        for column_index, fmt in enumerate(formats):
            axis = axes[row_index][column_index]
            selected = [
                row
                for row in rows
                if row["model"] == model
                and row["nm_format"] == fmt
                and row["method"] in methods
            ]
            labels = [
                f"{row['projection']} M{row['M']}\n{row['method']}"
                for row in selected
            ]
            x = list(range(len(selected)))
            axis.bar(
                x,
                [float(row["speedup_vs_dense"]) for row in selected],
                color=[colors[str(row["method"])] for row in selected],
            )
            axis.axhline(1.0, color="black", linewidth=1)
            axis.set_xticks(x, labels, rotation=90, fontsize=7)
            axis.set_title(f"{model} {fmt}")
            axis.set_ylabel("speedup vs BF16 cuBLAS")
            axis.grid(axis="y", alpha=0.25)
    figure.suptitle("External N:M GEMM on RTX 5090 (median of 10 x 1000)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(common.DEFAULT_MODELS))
    parser.add_argument("--projections", default=",".join(common.PROJECTIONS))
    parser.add_argument("--m-values", default=",".join(map(str, common.M_VALUES)))
    parser.add_argument("--formats", default=",".join(common.FORMATS))
    parser.add_argument("--systems", default=",".join(common.SYSTEMS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--split-screen-repeats", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "examples/evaluate/eval-guidellm/results/"
            "other_systems_nm_kernel_8b"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        args.models = common.parse_csv(
            args.models, tuple(common.TP1_FUSED_WEIGHT_SHAPES), "models"
        )
        args.projections = common.parse_csv(
            args.projections, common.PROJECTIONS, "projections"
        )
        args.formats = tuple(
            parse_nm(value).label
            for value in args.formats.split(",")
            if value.strip()
        )
        if not args.formats:
            raise ValueError("at least one N:M format is required")
        args.systems = common.parse_csv(args.systems, common.SYSTEMS, "systems")
        args.m_values = tuple(int(value) for value in args.m_values.split(","))
        if not args.m_values or any(value not in common.M_VALUES for value in args.m_values):
            raise ValueError(f"M must be selected from {common.M_VALUES}")
    except ValueError as error:
        parser.error(str(error))
    if args.smoke:
        args.models = args.models[:1]
        args.projections = args.projections[:1]
        args.m_values = args.m_values[:1]
        args.formats = args.formats[:1]
        args.warmup = 3
        args.trials = 2
        args.replays = 10
        args.split_screen_repeats = 3
        if args.output_root == Path(
            "examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b"
        ):
            args.output_root = Path(
                "examples/evaluate/eval-guidellm/temp/other_systems_nm_kernel_smoke"
            )
    elif args.trials != 10:
        parser.error("formal protocol requires exactly 10 trials")
    return args


def main() -> None:
    args = parse_args()
    common.assert_gpu_idle(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024, dtype=torch.uint8, device=device
    )
    eviction.zero_()

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    screens: list[dict[str, Any]] = []
    for model in args.models:
        for projection in args.projections:
            n, k = common.TP1_FUSED_WEIGHT_SHAPES[model][projection]
            for fmt in args.formats:
                weight = common.make_nm_weight(
                    model, projection, fmt, device=device, seed=args.seed
                )
                dense_by_m: dict[int, common.TimingSummary] = {}
                for m in args.m_values:
                    x = common.make_input(
                        model, projection, m, device=device, seed=args.seed
                    )
                    reference = common.dense_linear(x, weight)
                    dense_graph = common.capture(
                        lambda x=x, weight=weight: common.dense_linear(x, weight)
                    )
                    dense_summary, dense_raw = common.formal_measure(
                        dense_graph,
                        eviction,
                        warmup=args.warmup,
                        trials=args.trials,
                        replays=args.replays,
                    )
                    dense_by_m[m] = dense_summary
                    rows.append(
                        {
                            "model": model,
                            "projection": projection,
                            "M": m,
                            "N": n,
                            "K": k,
                            "nm_format": fmt,
                            "method": "dense_cublas",
                            "split_k": 0,
                            **dense_summary.as_dict(),
                            "speedup_vs_dense": 1.0,
                            "correct": True,
                            "max_abs_error": 0.0,
                            "mean_abs_error": 0.0,
                        }
                    )
                    for trial, latency in enumerate(dense_raw):
                        raw.append(
                            {
                                "model": model,
                                "projection": projection,
                                "M": m,
                                "nm_format": fmt,
                                "method": "dense_cublas",
                                "trial": trial,
                                "latency_us": latency,
                            }
                        )

                    del reference, dense_graph, x

                # Compression is an offline setup cost.  Prepare each external
                # representation once and reuse it across all activation M.
                for system in args.systems:
                    prepared = common.prepare_system(system, weight, fmt)
                    for m in args.m_values:
                        x = common.make_input(
                            model, projection, m, device=device, seed=args.seed
                        )
                        reference = common.dense_linear(x, weight)
                        selected, split_rows = common.select_split(
                            prepared,
                            x,
                            repeats=args.split_screen_repeats,
                        )
                        for item in split_rows:
                            screens.append(
                                {
                                    "model": model,
                                    "projection": projection,
                                    "M": m,
                                    "N": n,
                                    "K": k,
                                    "nm_format": fmt,
                                    "method": system,
                                    **item,
                                }
                            )
                        graph = common.capture(
                            lambda selected=selected, x=x: selected.linear(x)
                        )
                        check = common.correctness(graph.output, reference)
                        if not check["correct"]:
                            raise RuntimeError(
                                f"{system} correctness failed for "
                                f"{model}/{projection}/M{m}/{fmt}: {check}"
                            )
                        summary, samples = common.formal_measure(
                            graph,
                            eviction,
                            warmup=args.warmup,
                            trials=args.trials,
                            replays=args.replays,
                        )
                        rows.append(
                            {
                                "model": model,
                                "projection": projection,
                                "M": m,
                                "N": n,
                                "K": k,
                                "nm_format": fmt,
                                "method": system,
                                "split_k": selected.split_k,
                                **summary.as_dict(),
                                "speedup_vs_dense": (
                                    dense_by_m[m].median_us / summary.median_us
                                ),
                                **check,
                            }
                        )
                        for trial, latency in enumerate(samples):
                            raw.append(
                                {
                                    "model": model,
                                    "projection": projection,
                                    "M": m,
                                    "nm_format": fmt,
                                    "method": system,
                                    "trial": trial,
                                    "latency_us": latency,
                                }
                            )
                        print(
                            f"[kernel] {model} {projection} M={m} {fmt} "
                            f"{system} split={selected.split_k} "
                            f"speedup={dense_by_m[m].median_us / summary.median_us:.4f}x",
                            flush=True,
                        )
                        del graph, selected, reference, x
                    del prepared
                    torch.cuda.empty_cache()
                del weight
                torch.cuda.empty_cache()

    common.write_csv(output / "kernel_results.csv", rows)
    common.write_csv(output / "kernel_raw_trials.csv", raw)
    common.write_csv(output / "split_screen.csv", screens)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    (output / "environment.json").write_text(
        json.dumps(common.environment_report(device), indent=2) + "\n"
    )
    plot(rows, output / "figures/kernel_speedup.png")
    print(output)


if __name__ == "__main__":
    main()
