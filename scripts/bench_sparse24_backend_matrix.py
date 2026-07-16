#!/usr/bin/env python3
"""Compare dense, local CUTLASS 2:4, and cuSPARSELt on serving shapes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.speclink_kernel import (  # noqa: E402
    apply_random_24_mask,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_cutlass_device_gemm_prepacked,
)


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    qkv_size: int


MODELS = {
    "qwen3_8b": ModelShape(4096, 12288, 6144),
    "llama3_1_8b": ModelShape(4096, 14336, 6144),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def projection_shapes(model: ModelShape) -> dict[str, tuple[int, int]]:
    return {
        "qkv_proj": (model.hidden_size, model.qkv_size),
        "o_proj": (model.hidden_size, model.hidden_size),
        "gate_up_proj": (model.hidden_size, 2 * model.intermediate_size),
        "down_proj": (model.intermediate_size, model.hidden_size),
    }


def event_time_ms(fn: Callable[[], torch.Tensor], *, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeat)


def median_eager_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = [event_time_ms(fn, repeat=repeat) for _ in range(trials)]
    return float(statistics.median(samples)), samples


def median_graph_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    trials: int,
) -> tuple[float | None, list[float], str]:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        sentinel = torch.empty(0, device="cuda")

        def replay() -> torch.Tensor:
            graph.replay()
            return sentinel

        samples = [event_time_ms(replay, repeat=repeat) for _ in range(trials)]
        return float(statistics.median(samples)), samples, ""
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, [], f"{type(exc).__name__}: {exc}"


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def sample_string(values: list[float]) -> str:
    return ";".join(f"{value:.6f}" for value in values)


def make_case_tensors(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (rows, in_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense_weight_kn = torch.randn(
        (in_features, out_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense_weight_kn = torch.where(
        dense_weight_kn == 0,
        torch.full_like(dense_weight_kn, 1.0e-3),
        dense_weight_kn,
    ).contiguous()
    weight24_kn, _ = apply_random_24_mask(
        dense_weight_kn,
        generator=generator,
    )
    packed = pack_24(weight24_kn, layout="n_major")
    values, metadata = prepare_cutlass_sparse24_device_gemm(
        packed.values,
        packed.meta,
        layout=packed.layout,
        K=in_features,
    )
    return (
        x.contiguous(),
        dense_weight_kn,
        weight24_kn.contiguous(),
        values,
        metadata,
    )


def run_case(
    *,
    model_label: str,
    projection: str,
    batch_size: int,
    num_spec_tokens: int,
    in_features: int,
    out_features: int,
    seed: int,
    warmup: int,
    repeat: int,
    trials: int,
) -> list[dict[str, object]]:
    rows = batch_size * (num_spec_tokens + 1)
    x, _dense_weight, weight24_kn, values, metadata = make_case_tensors(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
    )
    reference = x @ weight24_kn
    dense_out = torch.empty_like(reference)
    cutlass_out = torch.empty_like(reference)
    cutlass_workspace = torch.empty(
        (out_features, rows), device="cuda", dtype=torch.float16
    )
    cutlass_view = torch.empty_strided(
        (rows, out_features),
        (1, rows),
        device="cuda",
        dtype=torch.float16,
    )

    def dense_fn() -> torch.Tensor:
        return torch.mm(x, weight24_kn, out=dense_out)

    def cutlass_contiguous_fn() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            x,
            values,
            metadata,
            contiguous_output=True,
            out=cutlass_out,
            workspace=cutlass_workspace,
            device_config="auto",
        )

    def cutlass_view_fn() -> torch.Tensor:
        return sparse24_cutlass_device_gemm_prepacked(
            x,
            values,
            metadata,
            contiguous_output=False,
            out=cutlass_view,
            device_config="auto",
        )

    backend_fns: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("dense", dense_fn),
        ("cutlass_contiguous", cutlass_contiguous_fn),
        ("cutlass_view", cutlass_view_fn),
    ]
    cslt_resources: list[torch.Tensor] = []
    for alg_id in (0, 1):
        previous_force_cutlass = bool(SparseSemiStructuredTensor._FORCE_CUTLASS)
        try:
            SparseSemiStructuredTensor._FORCE_CUTLASS = False
            cslt_weight = to_sparse_semi_structured(
                weight24_kn.t().contiguous()
            )
        finally:
            SparseSemiStructuredTensor._FORCE_CUTLASS = previous_force_cutlass
        setattr(cslt_weight, "alg_id_cusparselt", alg_id)
        cslt_packed = cslt_weight.packed
        cslt_resources.extend((cslt_weight, cslt_packed))

        def cslt_fn(
            sparse_weight: torch.Tensor = cslt_weight,
            selected_alg_id: int = alg_id,
        ) -> torch.Tensor:
            setattr(sparse_weight, "alg_id_cusparselt", selected_alg_id)
            return F.linear(x, sparse_weight)

        backend_fns.append((f"cusparselt_alg{alg_id}", cslt_fn))

        def cslt_direct_fn(
            packed_weight: torch.Tensor = cslt_packed,
            selected_alg_id: int = alg_id,
        ) -> torch.Tensor:
            return torch._cslt_sparse_mm(
                packed_weight,
                x.t(),
                transpose_result=False,
                alg_id=selected_alg_id,
            ).t()

        backend_fns.append((f"cusparselt_direct_alg{alg_id}", cslt_direct_fn))

        def cslt_direct_contiguous_fn(
            packed_weight: torch.Tensor = cslt_packed,
            selected_alg_id: int = alg_id,
        ) -> torch.Tensor:
            return torch._cslt_sparse_mm(
                packed_weight,
                x.t(),
                transpose_result=False,
                alg_id=selected_alg_id,
            ).t().contiguous()

        backend_fns.append(
            (f"cusparselt_direct_contiguous_alg{alg_id}", cslt_direct_contiguous_fn)
        )

    result_rows: list[dict[str, object]] = []
    for backend, fn in backend_fns:
        eager_ms = float("nan")
        eager_samples: list[float] = []
        graph_ms: float | None = None
        graph_samples: list[float] = []
        error = ""
        graph_error = ""
        passed = False
        error_abs = float("nan")
        try:
            actual = fn()
            torch.cuda.synchronize()
            error_abs = max_abs_error(actual, reference)
            passed = bool(torch.allclose(actual, reference, rtol=2.0e-2, atol=2.0e-1))
            eager_ms, eager_samples = median_eager_ms(
                fn,
                warmup=warmup,
                repeat=repeat,
                trials=trials,
            )
            graph_ms, graph_samples, graph_error = median_graph_ms(
                fn,
                warmup=warmup,
                repeat=repeat,
                trials=trials,
            )
        except Exception as exc:  # noqa: BLE001
            torch.cuda.synchronize()
            error = f"{type(exc).__name__}: {exc}"
        result_rows.append(
            {
                "model": model_label,
                "projection": projection,
                "batch_size": batch_size,
                "num_spec_tokens": num_spec_tokens,
                "rows": rows,
                "in_features": in_features,
                "out_features": out_features,
                "backend": backend,
                "eager_ms": eager_ms,
                "eager_samples_ms": sample_string(eager_samples),
                "graph_ms": "" if graph_ms is None else graph_ms,
                "graph_samples_ms": sample_string(graph_samples),
                "max_abs_error": error_abs,
                "pass": passed,
                "error": error,
                "graph_error": graph_error,
            }
        )

    dense_row = next(row for row in result_rows if row["backend"] == "dense")
    dense_eager = float(dense_row["eager_ms"])
    dense_graph = (
        float(dense_row["graph_ms"])
        if dense_row["graph_ms"] != ""
        else float("nan")
    )
    cutlass_view_row = next(
        row for row in result_rows if row["backend"] == "cutlass_view"
    )
    cutlass_view_graph = (
        float(cutlass_view_row["graph_ms"])
        if cutlass_view_row["graph_ms"] != ""
        else float("nan")
    )
    for row in result_rows:
        eager_ms = float(row["eager_ms"])
        graph_ms_value = (
            float(row["graph_ms"])
            if row["graph_ms"] != ""
            else float("nan")
        )
        row["eager_speedup_vs_dense"] = (
            dense_eager / eager_ms if eager_ms > 0 else float("nan")
        )
        row["graph_speedup_vs_dense"] = (
            dense_graph / graph_ms_value
            if dense_graph > 0 and graph_ms_value > 0
            else float("nan")
        )
        row["graph_speedup_vs_cutlass_view"] = (
            cutlass_view_graph / graph_ms_value
            if cutlass_view_graph > 0 and graph_ms_value > 0
            else float("nan")
        )
    del cslt_resources, cslt_packed, cslt_weight
    del cutlass_view, cutlass_workspace, cutlass_out
    del dense_out, reference, metadata, values, weight24_kn, _dense_weight, x
    gc.collect()
    torch.cuda.empty_cache()
    return result_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    candidates = [
        row
        for row in rows
        if row["backend"] != "dense"
        and row["pass"]
        and finite_float(row["graph_speedup_vs_dense"]) is not None
    ]
    grouped: dict[tuple[str, str, int, int], list[dict[str, object]]] = {}
    for row in candidates:
        key = (
            str(row["model"]),
            str(row["projection"]),
            int(row["batch_size"]),
            int(row["num_spec_tokens"]),
        )
        grouped.setdefault(key, []).append(row)
    best_rows = [
        max(items, key=lambda item: float(item["graph_speedup_vs_dense"]))
        for items in grouped.values()
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Sparse24 Backend Matrix\n\n")
        handle.write(
            "CUDA-graph speedups compare the same fp16 2:4 materialized "
            "weight against dense cuBLAS. Packing is excluded.\n\n"
        )
        handle.write(
            "| model | projection | bs | K | rows | best backend | speedup |\n"
        )
        handle.write("|---|---|---:|---:|---:|---|---:|\n")
        for row in sorted(
            best_rows,
            key=lambda item: (
                str(item["model"]),
                str(item["projection"]),
                int(item["batch_size"]),
                int(item["num_spec_tokens"]),
            ),
        ):
            handle.write(
                f"| {row['model']} | {row['projection']} | "
                f"{row['batch_size']} | {row['num_spec_tokens']} | "
                f"{row['rows']} | {row['backend']} | "
                f"{float(row['graph_speedup_vs_dense']):.3f}x |\n"
            )


def write_plots(output_root: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    backends = (
        "cutlass_contiguous",
        "cutlass_view",
        "cusparselt_alg1",
        "cusparselt_direct_alg1",
        "cusparselt_direct_contiguous_alg1",
    )
    colors = {
        "cutlass_contiguous": "#1f77b4",
        "cutlass_view": "#2ca02c",
        "cusparselt_alg1": "#9467bd",
        "cusparselt_direct_alg1": "#d62728",
        "cusparselt_direct_contiguous_alg1": "#ff7f0e",
    }
    for model in sorted({str(row["model"]) for row in rows}):
        projections = [
            name
            for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
            if any(
                row["model"] == model and row["projection"] == name
                for row in rows
            )
        ]
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
        for axis, projection in zip(axes.flat, projections):
            projection_rows = [
                row
                for row in rows
                if row["model"] == model and row["projection"] == projection
            ]
            points = sorted(
                {
                    (int(row["batch_size"]), int(row["num_spec_tokens"]))
                    for row in projection_rows
                }
            )
            x_values = list(range(len(points)))
            for backend in backends:
                values: list[float] = []
                for batch_size, num_spec_tokens in points:
                    match = next(
                        (
                            row
                            for row in projection_rows
                            if row["backend"] == backend
                            and int(row["batch_size"]) == batch_size
                            and int(row["num_spec_tokens"]) == num_spec_tokens
                        ),
                        None,
                    )
                    value = (
                        finite_float(match["graph_speedup_vs_dense"])
                        if match is not None
                        else None
                    )
                    values.append(float("nan") if value is None else value)
                axis.plot(
                    x_values,
                    values,
                    marker="o",
                    linewidth=1.6,
                    label=backend,
                    color=colors[backend],
                )
            axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
            axis.axhline(1.4, color="#ff7f0e", linewidth=1.0, linestyle=":")
            axis.set_title(projection)
            axis.set_xticks(x_values)
            axis.set_xticklabels(
                [f"{batch_size}/{num_spec_tokens}" for batch_size, num_spec_tokens in points],
                rotation=35,
                ha="right",
            )
            axis.set_ylabel("CUDA-graph speedup vs dense")
            axis.grid(axis="y", alpha=0.25)
        for axis in axes.flat[len(projections) :]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        figure.suptitle(f"{model}: sparse backend by bs/K")
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(output_root / f"backend_speedup_{model}.png", dpi=180)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_csv, default=list(MODELS))
    parser.add_argument(
        "--projections",
        type=parse_csv,
        default=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    parser.add_argument("--batch-sizes", type=parse_int_csv, default=[16, 32, 64])
    parser.add_argument("--k-values", type=parse_int_csv, default=[6, 8, 10])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with real GPU access")
    unknown_models = sorted(set(args.models) - set(MODELS))
    if unknown_models:
        raise ValueError(f"unsupported models: {unknown_models}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        REPO_ROOT
        / "examples/evaluate/eval-guidellm/temp"
        / f"sparse24_backend_matrix_{stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    case_index = 0
    for model_label in args.models:
        shapes = projection_shapes(MODELS[model_label])
        for projection in args.projections:
            if projection not in shapes:
                raise ValueError(f"unsupported projection: {projection}")
            in_features, out_features = shapes[projection]
            for batch_size in args.batch_sizes:
                for num_spec_tokens in args.k_values:
                    case_index += 1
                    logical_rows = batch_size * (num_spec_tokens + 1)
                    print(
                        f"[{case_index:02d}] {model_label} {projection} "
                        f"bs={batch_size} K={num_spec_tokens} M={logical_rows}",
                        flush=True,
                    )
                    case_rows = run_case(
                        model_label=model_label,
                        projection=projection,
                        batch_size=batch_size,
                        num_spec_tokens=num_spec_tokens,
                        in_features=in_features,
                        out_features=out_features,
                        seed=args.seed + case_index,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        trials=args.trials,
                    )
                    rows.extend(case_rows)
                    write_csv(output_root / "backend_matrix.csv", rows)
                    for row in case_rows:
                        print(
                            f"  {row['backend']}: graph="
                            f"{row['graph_speedup_vs_dense']:.3f}x "
                            f"pass={row['pass']} error={row['error']}",
                            flush=True,
                        )
    write_report(output_root / "report.md", rows)
    write_plots(output_root, rows)
    metadata = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "models": args.models,
        "projections": args.projections,
        "batch_sizes": args.batch_sizes,
        "k_values": args.k_values,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_root.resolve())


if __name__ == "__main__":
    main()
