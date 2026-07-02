#!/usr/bin/env python3
"""Sweep semi-structured sparse backends for SR24 MLP shapes.

This is a focused operator probe for low-batch SR24 serving.  If a full 2:4
base MLP is slower than dense at the row count implied by batch size and K, no
token-routing policy can deliver a 1.2x end-to-end speedup there.  The script
checks whether PyTorch's chosen cuSPARSELt algorithm or backend is the limiting
factor.  It can compare cuSPARSELt against PyTorch's CUTLASS semi-structured
backend by toggling SparseSemiStructuredTensor._FORCE_CUTLASS during weight
conversion.
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_alg_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            gate_raw, down_raw = item.split(":", 1)
        elif "/" in item:
            gate_raw, down_raw = item.split("/", 1)
        else:
            gate_raw = down_raw = item
        pairs.append((int(gate_raw), int(down_raw)))
    return pairs


def make_base_24(weight: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % 4:
        raise ValueError("in_features must be divisible by 4")
    grouped = weight.view(out_features, in_features // 4, 4)
    keep_idx = grouped.abs().topk(2, dim=-1, largest=True, sorted=False).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool)
    keep.scatter_(-1, keep_idx, True)
    base = torch.zeros_like(grouped)
    base[keep] = grouped[keep]
    return base.view_as(weight).contiguous()


def silu_and_mul(gate_up: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    return F.silu(gate) * up


def set_alg_id(weight: torch.Tensor, alg_id: int) -> None:
    setattr(weight, "alg_id_cusparselt", int(alg_id))


@contextmanager
def sparse_backend(name: str):
    previous_force_cutlass = bool(SparseSemiStructuredTensor._FORCE_CUTLASS)
    if name == "cutlass":
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    elif name == "cusparselt":
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
    else:
        raise ValueError(f"unsupported sparse backend: {name}")
    try:
        yield
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = previous_force_cutlass


def time_call(fn: Callable[[], Any], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def time_graph(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float | None, str | None]:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / repeats), None
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, f"{type(exc).__name__}: {exc}"


def run_case(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    sparse_backend_name: str,
    gate_up_alg_id: int,
    down_alg_id: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    gate_up_weight = torch.randn(
        intermediate_size * 2,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    down_weight = torch.randn(
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=dtype,
    )
    with sparse_backend(sparse_backend_name):
        gate_up_base = to_sparse_semi_structured(make_base_24(gate_up_weight))
        down_base = to_sparse_semi_structured(make_base_24(down_weight))
    if sparse_backend_name == "cusparselt":
        set_alg_id(gate_up_base, gate_up_alg_id)
        set_alg_id(down_base, down_alg_id)

    def dense_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_weight)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_weight)

    def sparse_mlp() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_base)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_base)

    def sparse_gate_up() -> torch.Tensor:
        return F.linear(x, gate_up_base)

    def sparse_down() -> torch.Tensor:
        gate_up = F.linear(x, gate_up_base)
        act = silu_and_mul(gate_up, intermediate_size)
        return F.linear(act, down_base)

    dense_graph_ms, dense_graph_error = time_graph(
        dense_mlp, warmup=warmup, repeats=repeats
    )
    sparse_graph_ms, sparse_graph_error = time_graph(
        sparse_mlp, warmup=warmup, repeats=repeats
    )
    row = {
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype).replace("torch.", ""),
        "sparse_backend": sparse_backend_name,
        "alg_id": (
            str(gate_up_alg_id)
            if gate_up_alg_id == down_alg_id
            else f"{gate_up_alg_id}:{down_alg_id}"
        ),
        "gate_up_alg_id": gate_up_alg_id,
        "down_alg_id": down_alg_id,
        "dense_ms": time_call(dense_mlp, warmup=warmup, repeats=repeats),
        "sparse_ms": time_call(sparse_mlp, warmup=warmup, repeats=repeats),
        "sparse_gate_up_ms": time_call(
            sparse_gate_up, warmup=warmup, repeats=repeats
        ),
        "sparse_down_chain_ms": time_call(
            sparse_down, warmup=warmup, repeats=repeats
        ),
        "dense_graph_ms": dense_graph_ms,
        "dense_graph_error": dense_graph_error,
        "sparse_graph_ms": sparse_graph_ms,
        "sparse_graph_error": sparse_graph_error,
    }
    dense = dense_graph_ms or row["dense_ms"]
    sparse = sparse_graph_ms or row["sparse_ms"]
    row["sparse_over_dense"] = sparse / dense if dense else None
    row["speedup_vs_dense"] = dense / sparse if sparse else None
    row["meets_1p2_speedup"] = (
        row["speedup_vs_dense"] is not None and row["speedup_vs_dense"] >= 1.2
    )
    return row


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# SR24 cuSPARSELt alg_id Sweep\n\n")
        f.write(
            "| rows | backend | alg_id | dense graph ms | sparse graph ms | "
            "sparse/dense | speedup | >=1.2x |\n"
        )
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            dense = row.get("dense_graph_ms")
            sparse = row.get("sparse_graph_ms")
            dense_s = "" if dense is None else f"{float(dense):.4f}"
            sparse_s = "" if sparse is None else f"{float(sparse):.4f}"
            ratio = row.get("sparse_over_dense")
            speedup = row.get("speedup_vs_dense")
            f.write(
                f"| {row['rows']} | {row.get('sparse_backend', '')} | "
                f"{row['alg_id']} | {dense_s} | "
                f"{sparse_s} | "
                f"{'' if ratio is None else f'{float(ratio):.3f}x'} | "
                f"{'' if speedup is None else f'{float(speedup):.3f}x'} | "
                f"{'yes' if row.get('meets_1p2_speedup') else 'no'} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep cuSPARSELt alg_id for SR24 sparse MLP row sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows", type=parse_int_list, default=[72, 144, 288, 576])
    parser.add_argument("--alg-ids", type=parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--alg-pairs",
        type=parse_alg_pairs,
        default=None,
        help=(
            "Optional comma-separated gate_up:down alg pairs for cuSPARSELt. "
            "When unset, --alg-ids applies the same alg to both projections."
        ),
    )
    parser.add_argument(
        "--sparse-backends",
        default="cusparselt",
        help="Comma-separated backends to test: cusparselt,cutlass.",
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the cuSPARSELt alg_id sweep")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    output_root = args.output_root or (
        Path("examples/evaluate/eval-guidellm/results.bak")
        / f"speclink_sr24_cslt_algos_{timestamp()}"
    )
    rows: list[dict[str, Any]] = []
    sparse_backends = [
        item.strip()
        for item in args.sparse_backends.split(",")
        if item.strip()
    ]
    if not sparse_backends:
        raise SystemExit("empty --sparse-backends")
    for row_count in args.rows:
        for backend_name in sparse_backends:
            alg_pairs = (
                args.alg_pairs
                if args.alg_pairs is not None
                else [(alg_id, alg_id) for alg_id in args.alg_ids]
            )
            if backend_name != "cusparselt":
                alg_pairs = [(0, 0)]
            for gate_up_alg_id, down_alg_id in alg_pairs:
                try:
                    rows.append(
                        run_case(
                            rows=row_count,
                            hidden_size=args.hidden_size,
                            intermediate_size=args.intermediate_size,
                            dtype=dtype,
                            sparse_backend_name=backend_name,
                            gate_up_alg_id=gate_up_alg_id,
                            down_alg_id=down_alg_id,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    torch.cuda.synchronize()
                    rows.append({
                        "rows": row_count,
                        "hidden_size": args.hidden_size,
                        "intermediate_size": args.intermediate_size,
                        "dtype": str(dtype).replace("torch.", ""),
                        "sparse_backend": backend_name,
                        "alg_id": (
                            str(gate_up_alg_id)
                            if gate_up_alg_id == down_alg_id
                            else f"{gate_up_alg_id}:{down_alg_id}"
                        ),
                        "gate_up_alg_id": gate_up_alg_id,
                        "down_alg_id": down_alg_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    write_outputs(output_root, rows)
    print(output_root.resolve())


if __name__ == "__main__":
    main()
