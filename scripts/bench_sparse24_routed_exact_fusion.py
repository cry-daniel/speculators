#!/usr/bin/env python3
"""Benchmark gather-free exact row routing for QKV and the full MLP."""

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
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    sparse24_add_indexed_rows_contiguous_,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_grouped_owner_linear_prepacked,
    sparse24_cutlass_grouped_owner_swiglu_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_routed_exact_linear_prepacked,
    sparse24_cutlass_routed_exact_swiglu_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_,
    sparse24_routed_swiglu_correction_,
)


DEFAULT_BATCH_SIZES = (16, 32, 64)
DEFAULT_K_VALUES = (6, 8, 10)


def padded_rows(rows: int) -> int:
    return (rows + 7) // 8 * 8


def make_route(
    batch_size: int,
    k: int,
    *,
    dense_ratio: float,
    min_dense_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = batch_size * (k + 1)
    by_request = torch.arange(
        total_rows, device="cuda", dtype=torch.int32
    ).reshape(batch_size, k + 1)
    mandatory = by_request[:, :min_dense_per_request].flatten()
    scored_rows = batch_size * k
    dense_count = max(
        int(mandatory.numel()),
        int(scored_rows * dense_ratio + 0.5),
    )
    dense_count = min(scored_rows, dense_count)
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
    sparse_mask = torch.ones(total_rows, device="cuda", dtype=torch.bool)
    sparse_mask[dense_rows.long()] = False
    sparse_rows = sparse_mask.nonzero().flatten().to(torch.int32).contiguous()
    dense_slots = torch.full(
        (total_rows,), -1, device="cuda", dtype=torch.int32
    )
    dense_slots[dense_rows.long()] = torch.arange(
        dense_rows.numel(), device="cuda", dtype=torch.int32
    )
    return dense_rows, sparse_rows, dense_slots


def prepare_exact_weight(
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
    dense = torch.randn(
        (in_features, out_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    dense.mul_(0.02)
    dense.add_(torch.where(dense >= 0, 0.005, -0.005))
    sparse, _ = apply_random_24_mask(dense, generator=generator)

    def prepack(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_24(matrix, layout="n_major")
        return prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=in_features,
        )

    full_values, full_meta = prepack(sparse)
    residual_values, residual_meta = prepack(dense - sparse)
    return (
        dense,
        sparse,
        full_values,
        full_meta,
        residual_values,
        residual_meta,
    )


def run_case(
    model: str,
    batch_size: int,
    k: int,
    *,
    dense_ratio: float,
    min_dense_per_request: int,
    linear_config: str,
    swiglu_config: str,
    grouped_linear_config: str,
    grouped_group_tiles: int,
    generator: torch.Generator,
    unroll: int,
    replays: int,
    trials: int,
    graph_warmup_replays: int,
) -> dict[str, object]:
    hidden = int(MODELS[model]["hidden"])
    intermediate = int(MODELS[model]["intermediate"])
    rows = batch_size * (k + 1)
    dense_rows, sparse_rows, dense_slots = make_route(
        batch_size,
        k,
        dense_ratio=dense_ratio,
        min_dense_per_request=min_dense_per_request,
        generator=generator,
    )
    dense_count = int(dense_rows.numel())
    dense_run = padded_rows(dense_count)

    x = torch.randn(
        (rows, hidden),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    x.mul_(0.1)
    gate = prepare_exact_weight(hidden, 2 * intermediate, generator)
    down = prepare_exact_weight(intermediate, hidden, generator)
    qkv = prepare_exact_weight(hidden, 6144, generator)

    dense_x = torch.zeros(
        (dense_run, hidden), device="cuda", dtype=torch.float16
    )
    old_gate_hidden = torch.empty(
        (rows, intermediate), device="cuda", dtype=torch.float16
    )
    new_gate_hidden = torch.empty_like(old_gate_hidden)
    grouped_gate_hidden = torch.empty_like(old_gate_hidden)
    gate_dense_base = torch.empty(
        (dense_count, 2 * intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    gate_residual = torch.empty(
        (dense_run, 2 * intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    grouped_gate_dense_base = torch.empty_like(gate_dense_base)

    full_stream = torch.cuda.Stream()
    residual_stream = torch.cuda.Stream()

    def old_gate() -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_routed_swiglu_prepacked(
                x,
                gate[2],
                gate[3],
                dense_slots,
                dense_count=dense_count,
                out=old_gate_hidden,
                dense_base=gate_dense_base,
                config=swiglu_config,
            )
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_(x, dense_rows, dense_x[:dense_count])
            sparse24_cutlass_inline_transpose_gemm_prepacked(
                dense_x,
                gate[4],
                gate[5],
                out=gate_residual,
                config="auto",
                store_mode="vector",
            )
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return sparse24_routed_swiglu_correction_(
            gate_dense_base,
            gate_residual[:dense_count],
            dense_rows,
            old_gate_hidden,
        )

    def new_gate() -> torch.Tensor:
        return sparse24_cutlass_routed_exact_swiglu_prepacked(
            x,
            gate[2],
            gate[3],
            gate[4],
            gate[5],
            dense_rows,
            sparse_rows,
            out=new_gate_hidden,
            config=swiglu_config,
        )

    def grouped_gate() -> torch.Tensor:
        return sparse24_cutlass_grouped_owner_swiglu_prepacked(
            x,
            gate[2],
            gate[3],
            gate[4],
            gate[5],
            dense_rows,
            dense_slots,
            out=grouped_gate_hidden,
            dense_base=grouped_gate_dense_base,
            group_tiles=grouped_group_tiles,
            config="256x32x64_s3_sw4",
        )

    down_dense_input = torch.zeros(
        (dense_run, intermediate), device="cuda", dtype=torch.float16
    )
    old_down_output = torch.empty(
        (rows, hidden), device="cuda", dtype=torch.float16
    )
    new_down_output = torch.empty_like(old_down_output)
    grouped_down_output = torch.empty_like(old_down_output)
    down_residual = torch.empty(
        (dense_run, hidden), device="cuda", dtype=torch.float16
    )
    down_residual_workspace = torch.empty(
        (hidden, dense_run), device="cuda", dtype=torch.float16
    )

    def old_down_from(hidden_input: torch.Tensor) -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                hidden_input,
                down[2],
                down[3],
                contiguous_output=True,
                out=old_down_output,
                device_config="auto",
            )
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_(
                hidden_input,
                dense_rows,
                down_dense_input[:dense_count],
            )
            sparse24_cutlass_device_gemm_prepacked(
                down_dense_input,
                down[4],
                down[5],
                contiguous_output=True,
                out=down_residual,
                workspace=down_residual_workspace,
                device_config="auto",
            )
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return sparse24_add_indexed_rows_contiguous_(
            old_down_output,
            down_residual[:dense_count],
            dense_rows,
        )

    def old_down() -> torch.Tensor:
        return old_down_from(new_gate_hidden)

    def new_down() -> torch.Tensor:
        return sparse24_cutlass_routed_exact_linear_prepacked(
            new_gate_hidden,
            down[2],
            down[3],
            down[4],
            down[5],
            dense_rows,
            sparse_rows,
            out=new_down_output,
            config=linear_config,
        )

    def grouped_down_from(hidden_input: torch.Tensor) -> torch.Tensor:
        return sparse24_cutlass_grouped_owner_linear_prepacked(
            hidden_input,
            down[2],
            down[3],
            down[4],
            down[5],
            dense_rows,
            out=grouped_down_output,
            group_tiles=grouped_group_tiles,
            config=grouped_linear_config,
        )

    def grouped_down() -> torch.Tensor:
        return grouped_down_from(new_gate_hidden)

    def old_mlp() -> torch.Tensor:
        old_gate()
        return old_down_from(old_gate_hidden)

    def new_mlp() -> torch.Tensor:
        new_gate()
        return new_down()

    def grouped_mlp() -> torch.Tensor:
        grouped_gate()
        return grouped_down_from(grouped_gate_hidden)

    qkv_dense_input = torch.zeros(
        (dense_run, hidden), device="cuda", dtype=torch.float16
    )
    old_qkv_output = torch.empty(
        (rows, 6144), device="cuda", dtype=torch.float16
    )
    new_qkv_output = torch.empty_like(old_qkv_output)
    grouped_qkv_output = torch.empty_like(old_qkv_output)
    qkv_residual = torch.empty(
        (dense_run, 6144), device="cuda", dtype=torch.float16
    )
    qkv_residual_workspace = torch.empty(
        (6144, dense_run), device="cuda", dtype=torch.float16
    )

    def old_qkv() -> torch.Tensor:
        current = torch.cuda.current_stream()
        full_stream.wait_stream(current)
        residual_stream.wait_stream(current)
        with torch.cuda.stream(full_stream):
            sparse24_cutlass_device_gemm_prepacked(
                x,
                qkv[2],
                qkv[3],
                contiguous_output=True,
                out=old_qkv_output,
                device_config="auto",
            )
        with torch.cuda.stream(residual_stream):
            sparse24_gather_rows_(x, dense_rows, qkv_dense_input[:dense_count])
            sparse24_cutlass_device_gemm_prepacked(
                qkv_dense_input,
                qkv[4],
                qkv[5],
                contiguous_output=True,
                out=qkv_residual,
                workspace=qkv_residual_workspace,
                device_config="auto",
            )
        current.wait_stream(full_stream)
        current.wait_stream(residual_stream)
        return sparse24_add_indexed_rows_contiguous_(
            old_qkv_output, qkv_residual[:dense_count], dense_rows
        )

    def new_qkv() -> torch.Tensor:
        return sparse24_cutlass_routed_exact_linear_prepacked(
            x,
            qkv[2],
            qkv[3],
            qkv[4],
            qkv[5],
            dense_rows,
            sparse_rows,
            out=new_qkv_output,
            config=linear_config,
        )

    def grouped_qkv() -> torch.Tensor:
        return sparse24_cutlass_grouped_owner_linear_prepacked(
            x,
            qkv[2],
            qkv[3],
            qkv[4],
            qkv[5],
            dense_rows,
            out=grouped_qkv_output,
            group_tiles=grouped_group_tiles,
            config=grouped_linear_config,
        )

    old_gate_value = old_gate().clone()
    new_gate_value = new_gate().clone()
    grouped_gate_value = grouped_gate().clone()
    old_mlp_value = old_mlp().clone()
    new_mlp_value = new_mlp().clone()
    grouped_mlp_value = grouped_mlp().clone()
    old_qkv_value = old_qkv().clone()
    new_qkv_value = new_qkv().clone()
    grouped_qkv_value = grouped_qkv().clone()
    torch.cuda.synchronize()
    gate_diff = float(
        (old_gate_value.float() - new_gate_value.float()).abs().max().item()
    )
    mlp_diff = float(
        (old_mlp_value.float() - new_mlp_value.float()).abs().max().item()
    )
    qkv_diff = float(
        (old_qkv_value.float() - new_qkv_value.float()).abs().max().item()
    )
    grouped_gate_diff = float(
        (old_gate_value.float() - grouped_gate_value.float()).abs().max().item()
    )
    grouped_mlp_diff = float(
        (old_mlp_value.float() - grouped_mlp_value.float()).abs().max().item()
    )
    grouped_qkv_diff = float(
        (old_qkv_value.float() - grouped_qkv_value.float()).abs().max().item()
    )
    if not torch.allclose(old_gate_value, new_gate_value, rtol=4e-2, atol=2e-1):
        raise RuntimeError(f"Gate+SwiGLU mismatch: max_abs_diff={gate_diff}")
    if not torch.allclose(old_mlp_value, new_mlp_value, rtol=5e-2, atol=3e-1):
        raise RuntimeError(f"MLP mismatch: max_abs_diff={mlp_diff}")
    if not torch.allclose(old_qkv_value, new_qkv_value, rtol=4e-2, atol=2e-1):
        raise RuntimeError(f"QKV mismatch: max_abs_diff={qkv_diff}")
    if not torch.allclose(
        old_gate_value, grouped_gate_value, rtol=4e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"grouped Gate+SwiGLU mismatch: max_abs_diff={grouped_gate_diff}"
        )
    if not torch.allclose(
        old_mlp_value, grouped_mlp_value, rtol=5e-2, atol=3e-1
    ):
        raise RuntimeError(
            f"grouped MLP mismatch: max_abs_diff={grouped_mlp_diff}"
        )
    if not torch.allclose(
        old_qkv_value, grouped_qkv_value, rtol=4e-2, atol=2e-1
    ):
        raise RuntimeError(
            f"grouped QKV mismatch: max_abs_diff={grouped_qkv_diff}"
        )

    def measure(old_fn, new_fn) -> tuple[float, float]:
        return paired_graph_median_ms(
            old_fn,
            new_fn,
            unroll=unroll,
            replays=replays,
            trials=trials,
            graph_warmup_replays=graph_warmup_replays,
        )

    old_gate_ms, new_gate_ms = measure(old_gate, new_gate)
    old_down_ms, new_down_ms = measure(old_down, new_down)
    old_mlp_ms, new_mlp_ms = measure(old_mlp, new_mlp)
    old_qkv_ms, new_qkv_ms = measure(old_qkv, new_qkv)
    grouped_old_gate_ms, grouped_gate_ms = measure(old_gate, grouped_gate)
    grouped_old_down_ms, grouped_down_ms = measure(old_down, grouped_down)
    grouped_old_mlp_ms, grouped_mlp_ms = measure(old_mlp, grouped_mlp)
    grouped_old_qkv_ms, grouped_qkv_ms = measure(old_qkv, grouped_qkv)
    return {
        "model": model,
        "batch_size": batch_size,
        "K": k,
        "rows": rows,
        "dense_rows": dense_count,
        "sparse_rows": int(sparse_rows.numel()),
        "dense_ratio": dense_ratio,
        "linear_config": linear_config,
        "swiglu_config": swiglu_config,
        "grouped_linear_config": grouped_linear_config,
        "grouped_group_tiles": grouped_group_tiles,
        "old_gate_ms": old_gate_ms,
        "new_gate_ms": new_gate_ms,
        "gate_speedup": old_gate_ms / new_gate_ms,
        "old_down_ms": old_down_ms,
        "new_down_ms": new_down_ms,
        "down_speedup": old_down_ms / new_down_ms,
        "old_mlp_ms": old_mlp_ms,
        "new_mlp_ms": new_mlp_ms,
        "mlp_speedup": old_mlp_ms / new_mlp_ms,
        "old_qkv_ms": old_qkv_ms,
        "new_qkv_ms": new_qkv_ms,
        "qkv_speedup": old_qkv_ms / new_qkv_ms,
        "grouped_old_gate_ms": grouped_old_gate_ms,
        "grouped_gate_ms": grouped_gate_ms,
        "grouped_gate_speedup": grouped_old_gate_ms / grouped_gate_ms,
        "grouped_old_down_ms": grouped_old_down_ms,
        "grouped_down_ms": grouped_down_ms,
        "grouped_down_speedup": grouped_old_down_ms / grouped_down_ms,
        "grouped_old_mlp_ms": grouped_old_mlp_ms,
        "grouped_mlp_ms": grouped_mlp_ms,
        "grouped_mlp_speedup": grouped_old_mlp_ms / grouped_mlp_ms,
        "grouped_old_qkv_ms": grouped_old_qkv_ms,
        "grouped_qkv_ms": grouped_qkv_ms,
        "grouped_qkv_speedup": grouped_old_qkv_ms / grouped_qkv_ms,
        "gate_max_abs_diff": gate_diff,
        "mlp_max_abs_diff": mlp_diff,
        "qkv_max_abs_diff": qkv_diff,
        "grouped_gate_max_abs_diff": grouped_gate_diff,
        "grouped_mlp_max_abs_diff": grouped_mlp_diff,
        "grouped_qkv_max_abs_diff": grouped_qkv_diff,
    }


def write_plot(output_root: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"{row['model']}\nbs{row['batch_size']}/K{row['K']}" for row in rows
    ]
    x = list(range(len(rows)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(8, len(rows) * 1.2), 4.5))
    for offset, field, label, color in (
        (-width, "grouped_qkv_speedup", "QKV", "#176B87"),
        (0.0, "grouped_gate_speedup", "Gate+SwiGLU", "#B33F40"),
        (width, "grouped_mlp_speedup", "Full MLP", "#2A9D8F"),
    ):
        axis.bar(
            [value + offset for value in x],
            [float(row[field]) for row in rows],
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(1.0, color="#222222", linewidth=1)
    axis.set_ylabel("Speedup vs current routed residual")
    axis.set_xticks(x, labels)
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "routed_exact_fusion.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--batch-sizes", default="16,32,64")
    parser.add_argument("--k-values", default="6,8,10")
    parser.add_argument("--dense-ratio", type=float, default=0.125)
    parser.add_argument("--min-dense-per-request", type=int, default=1)
    parser.add_argument("--linear-config", default="128x64x64_s5")
    parser.add_argument("--swiglu-config", default="256x64x64_s3_sw4")
    parser.add_argument("--grouped-linear-config", default="64x32x64_s3")
    parser.add_argument("--grouped-group-tiles", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=4)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-warmup-replays", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    models = parse_csv_strings(args.models)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    k_values = parse_csv_ints(args.k_values)
    args.output_root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r}")
        for batch_size in batch_sizes:
            for k in k_values:
                row = run_case(
                    model,
                    batch_size,
                    k,
                    dense_ratio=args.dense_ratio,
                    min_dense_per_request=args.min_dense_per_request,
                    linear_config=args.linear_config,
                    swiglu_config=args.swiglu_config,
                    grouped_linear_config=args.grouped_linear_config,
                    grouped_group_tiles=args.grouped_group_tiles,
                    generator=generator,
                    unroll=args.unroll,
                    replays=args.replays,
                    trials=args.trials,
                    graph_warmup_replays=args.graph_warmup_replays,
                )
                rows.append(row)
                print(row, flush=True)

    csv_path = args.output_root / "routed_exact_fusion.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_plot(args.output_root, rows)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
