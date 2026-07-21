#!/usr/bin/env python3
"""Aggregate the Llama8 dense-base and Qwen o_proj NCU campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from ncu_report_utils import extract


PROJECTIONS = ("qkv", "o", "gate_up", "down")
QWEN_MODELS = ("qwen3_8b", "qwen3_14b", "qwen3_32b")
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
}
SM_COUNT = 170
SHARED_MEMORY_PER_SM = 102400


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    keys: list[str] = []
    for row in materialized:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(materialized)


def one(
    rows: list[dict[str, str]], *, case: str, method: str
) -> dict[str, str]:
    matches = [
        row for row in rows if row["case"] == case and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {case}/{method}: {len(matches)}")
    return matches[0]


def ratio(left: Any, right: Any) -> float | None:
    if left is None or right in {None, 0}:
        return None
    return float(left) / float(right)


def grid_ctas(value: str) -> int:
    numbers = [int(part) for part in re.findall(r"\d+", value)]
    if len(numbers) != 3:
        raise ValueError(f"unexpected grid shape: {value}")
    return math.prod(numbers)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def parse_llama_ncu(root: Path) -> list[dict[str, Any]]:
    pattern = re.compile(r"llama3_1_8b_(qkv|o|gate_up|down)__(.+)\.ncu-rep")
    rows: list[dict[str, Any]] = []
    for report in sorted((root / "ncu").glob("*.ncu-rep")):
        match = pattern.fullmatch(report.name)
        if match is None:
            raise RuntimeError(f"unexpected Llama NCU report name: {report.name}")
        projection, method = match.groups()
        row = extract(report, method)
        row.update({"projection": projection})
        rows.append(row)
    if len(rows) != len(PROJECTIONS) * 7:
        raise RuntimeError(f"incomplete Llama NCU matrix: {len(rows)}")
    return rows


def llama_e2e(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_rows = read_csv(args.llama_d27_root / "summary.csv")
    qkv_rows = read_csv(args.llama_qkv_root / "summary.csv")
    current_rows = read_csv(args.current_m2048_root / "summary.csv")
    selected = {
        "qkv": (qkv_rows, "two_branch_concurrent__d29_s141", "D29:S141"),
        "o": (base_rows, "two_branch_concurrent__d27_s143", "D27:S143"),
        "gate_up": (
            base_rows,
            "two_branch_concurrent__d27_s143",
            "D27:S143",
        ),
        "down": (base_rows, "two_branch_concurrent__d27_s143", "D27:S143"),
    }
    result: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        case = f"llama3_1_8b__{projection}__m2048"
        old_source, old_method, quota = selected[projection]
        old = float(one(old_source, case=case, method=old_method)["median_us"])
        current = float(
            one(
                current_rows,
                case=case,
                method="cusparselt_full24_residual_concurrent",
            )["median_us"]
        )
        cublas = float(
            one(current_rows, case=case, method="cublas_dense")["median_us"]
        )
        pure = float(
            one(current_rows, case=case, method="cusparselt_pure24")["median_us"]
        )
        result.append(
            {
                "projection": projection,
                "old_quota": quota,
                "old_dense_base_us": old,
                "current_complement_us": current,
                "cublas_dense_m2048_us": cublas,
                "cusparselt_pure24_m2048_us": pure,
                "old_speedup_vs_cublas": cublas / old,
                "current_speedup_vs_cublas": cublas / current,
                "current_speedup_vs_old": old / current,
                "current_overhead_vs_pure24_pct": 100.0 * (current / pure - 1.0),
            }
        )
    return result


def llama_pairs(ncu_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["projection"], row["method"]): row for row in ncu_rows}
    pairs = (
        ("dense", "old_dense_m256", "cublas_m256"),
        ("sparse", "old_sparse_m1792", "cusparselt_m1792"),
    )
    result: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        for role, custom_key, baseline_key in pairs:
            custom = by_key[(projection, custom_key)]
            baseline = by_key[(projection, baseline_key)]
            result.append(
                {
                    "projection": projection,
                    "role": role,
                    "custom": custom_key,
                    "baseline": baseline_key,
                    "custom_duration_us": custom["duration_us"],
                    "baseline_duration_us": baseline["duration_us"],
                    "duration_ratio": ratio(
                        custom["duration_us"], baseline["duration_us"]
                    ),
                    "custom_tensor_active_pct": custom["tensor_active_pct"],
                    "baseline_tensor_active_pct": baseline["tensor_active_pct"],
                    "tensor_active_ratio": ratio(
                        custom["tensor_active_pct"], baseline["tensor_active_pct"]
                    ),
                    "custom_dram_read_bytes": custom["dram_read_bytes"],
                    "baseline_dram_read_bytes": baseline["dram_read_bytes"],
                    "dram_read_ratio": ratio(
                        custom["dram_read_bytes"], baseline["dram_read_bytes"]
                    ),
                    "custom_shared_bank_conflicts": custom[
                        "shared_bank_conflicts"
                    ],
                    "baseline_shared_bank_conflicts": baseline[
                        "shared_bank_conflicts"
                    ],
                    "shared_bank_conflict_ratio": ratio(
                        custom["shared_bank_conflicts"],
                        baseline["shared_bank_conflicts"],
                    ),
                    "custom_barrier_stall_per_issue": custom[
                        "stall_barrier_per_issue"
                    ],
                    "baseline_barrier_stall_per_issue": baseline[
                        "stall_barrier_per_issue"
                    ],
                    "barrier_stall_ratio": ratio(
                        custom["stall_barrier_per_issue"],
                        baseline["stall_barrier_per_issue"],
                    ),
                    "custom_long_scoreboard_per_issue": custom[
                        "stall_long_scoreboard_per_issue"
                    ],
                    "baseline_long_scoreboard_per_issue": baseline[
                        "stall_long_scoreboard_per_issue"
                    ],
                    "registers_per_thread": custom["registers_per_thread"],
                    "dynamic_shared_mem_bytes": custom[
                        "dynamic_shared_mem_bytes"
                    ],
                    "local_load_sectors": custom["local_load_sectors"],
                    "local_store_sectors": custom["local_store_sectors"],
                }
            )
    return result


def write_llama_report(
    root: Path,
    e2e: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    lines = [
        "# Llama3.1-8B: why retire dense-base one-weight",
        "",
        "Formal E2E uses 100 warmups and 10 x 1000 CUDA Graph replays. NCU uses ",
        "one isolated full-SM equal-work kernel with `cache-control all`; its duration is diagnostic, not the formal latency authority.",
        "",
        "## Formal E2E",
        "",
        "| Layer | Old dense-base | Current complement | cuBLAS M2048 | pure 2:4 | Current vs old |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in e2e:
        lines.append(
            f"| {row['projection']} ({row['old_quota']}) | {fmt(row['old_dense_base_us'])} | "
            f"{fmt(row['current_complement_us'])} | {fmt(row['cublas_dense_m2048_us'])} | "
            f"{fmt(row['cusparselt_pure24_m2048_us'])} | {fmt(row['current_speedup_vs_old'])}x |"
        )
    lines += [
        "",
        "## NCU equal-work role pairs",
        "",
        "| Layer | Role | Custom/library time | Tensor active custom/library | DRAM ratio | bank-conflict ratio | barrier-stall ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['projection']} | {row['role']} | {fmt(row['custom_duration_us'])}/{fmt(row['baseline_duration_us'])} us "
            f"({fmt(row['duration_ratio'])}x) | {fmt(row['custom_tensor_active_pct'], 1)}/{fmt(row['baseline_tensor_active_pct'], 1)}% | "
            f"{fmt(row['dram_read_ratio'], 2)}x | {fmt(row['shared_bank_conflict_ratio'], 2)}x | {fmt(row['barrier_stall_ratio'], 1)}x |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "The old dense branch reads essentially the same HBM bytes as matched cuBLAS M256, but converts those bytes into much less Tensor-Core activity while paying a large persistent-CTA barrier/control tax. The old sparse branch additionally performs dense-to-2:4 online packing, increasing shared-memory traffic/conflicts and, for several shapes, DRAM traffic. The issue is pipeline efficiency, not densifying sparse rows and not register spill.",
        "",
        "The current representation removes the old dense branch and online dense-to-2:4 pack: all rows use prepared cuSPARSELt base 2:4, while only 256 routed rows execute complement HMMA.SP.",
        "",
        "- [NCU rows](ncu_summary.csv)",
        "- [Equal-work pair ratios](ncu_pair_ratios.csv)",
        "- [Formal E2E rows](e2e_summary.csv)",
        "- [Figure](figures/llama8_dense_base_ncu.png)",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_llama(root: Path, e2e: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = [row["projection"] for row in e2e]
    x = np.arange(len(labels))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - width, [row["old_dense_base_us"] for row in e2e], width, label="Old dense-base")
    axes[0].bar(x, [row["current_complement_us"] for row in e2e], width, label="Current complement")
    axes[0].bar(x + width, [row["cublas_dense_m2048_us"] for row in e2e], width, label="cuBLAS dense")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Formal median (us)")
    axes[0].set_title("M=2048 E2E")
    axes[0].legend()
    dense = {row["projection"]: row for row in pairs if row["role"] == "dense"}
    sparse = {row["projection"]: row for row in pairs if row["role"] == "sparse"}
    axes[1].bar(x - 1.5 * width, [dense[p]["custom_tensor_active_pct"] for p in labels], width, label="Old dense")
    axes[1].bar(x - 0.5 * width, [dense[p]["baseline_tensor_active_pct"] for p in labels], width, label="cuBLAS M256")
    axes[1].bar(x + 0.5 * width, [sparse[p]["custom_tensor_active_pct"] for p in labels], width, label="Old sparse")
    axes[1].bar(x + 1.5 * width, [sparse[p]["baseline_tensor_active_pct"] for p in labels], width, label="cuSPARSELt M1792")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("NCU Tensor active (%)")
    axes[1].set_title("Equal-work isolated kernels")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "llama8_dense_base_ncu.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_qwen_ncu(root: Path) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(qwen3_(?:8b|14b|32b))_o_m(512|1024)__(.+)\.ncu-rep"
    )
    rows: list[dict[str, Any]] = []
    for report in sorted((root / "ncu").glob("*.ncu-rep")):
        match = pattern.fullmatch(report.name)
        if match is None:
            raise RuntimeError(f"unexpected Qwen NCU report name: {report.name}")
        model, m_text, method = match.groups()
        row = extract(report, method)
        row.update({"model": model, "M": int(m_text), "grid_ctas": grid_ctas(row["grid_size"])})
        rows.append(row)
    if len(rows) != len(QWEN_MODELS) * 2 * 3:
        raise RuntimeError(f"incomplete Qwen NCU matrix: {len(rows)}")
    return rows


def qwen_summary(
    args: argparse.Namespace, ncu_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["M"], row["method"]): row for row in ncu_rows}
    formal_by_m = {
        512: read_csv(args.current_m512_root / "summary.csv"),
        1024: read_csv(args.current_m1024_root / "summary.csv"),
    }
    shapes = {
        "qwen3_8b": (4096, 4096),
        "qwen3_14b": (5120, 5120),
        "qwen3_32b": (5120, 8192),
    }
    result: list[dict[str, Any]] = []
    for m in (512, 1024):
        for model in QWEN_MODELS:
            case = f"{model}__o__m{m}"
            formal = formal_by_m[m]
            hybrid = float(one(formal, case=case, method="cusparselt_full24_residual_concurrent")["median_us"])
            cublas = float(one(formal, case=case, method="cublas_dense")["median_us"])
            pure = float(one(formal, case=case, method="cusparselt_pure24")["median_us"])
            base = by_key[(model, m, "cusparselt_base_full")]
            complement = by_key[(model, m, "complement_dense_fraction")]
            cublas_ncu = by_key[(model, m, "cublas_full")]
            base_ctas = int(base["grid_ctas"])
            complement_ctas = int(complement["grid_ctas"])
            base_waves = math.ceil(base_ctas / SM_COUNT)
            available_ctas = base_waves * SM_COUNT
            exposed_ctas = max(0, base_ctas + complement_ctas - available_ctas)
            n, k = shapes[model]
            overhead = hybrid - pure
            result.append(
                {
                    "M": m,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "N": n,
                    "K": k,
                    "weight_elements": n * k,
                    "hybrid_us": hybrid,
                    "cublas_us": cublas,
                    "pure24_us": pure,
                    "speedup_vs_cublas": cublas / hybrid,
                    "pure24_speedup_vs_cublas": cublas / pure,
                    "formal_overhead_vs_pure24_us": overhead,
                    "formal_overhead_vs_pure24_pct": 100.0 * overhead / pure,
                    "ncu_cublas_duration_us": cublas_ncu["duration_us"],
                    "ncu_base_duration_us": base["duration_us"],
                    "ncu_complement_duration_us": complement["duration_us"],
                    "ncu_complement_tensor_active_pct": complement[
                        "tensor_active_pct"
                    ],
                    "ncu_complement_dram_read_bytes": complement[
                        "dram_read_bytes"
                    ],
                    "ncu_complement_regs_per_thread": complement[
                        "registers_per_thread"
                    ],
                    "ncu_complement_smem_bytes": complement[
                        "dynamic_shared_mem_bytes"
                    ],
                    "base_grid_ctas": base_ctas,
                    "complement_grid_ctas": complement_ctas,
                    "combined_grid_ctas": base_ctas + complement_ctas,
                    "base_wave_capacity_ctas": available_ctas,
                    "minimum_combined_waves": math.ceil(
                        (base_ctas + complement_ctas) / SM_COUNT
                    ),
                    "exposed_ctas_beyond_base_waves": exposed_ctas,
                    "exposed_fraction_of_complement_ctas": (
                        exposed_ctas / complement_ctas
                    ),
                    "formal_overhead_over_ncu_complement": overhead
                    / float(complement["duration_us"]),
                }
            )
    return result


def write_qwen_report(root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Why Qwen3-14B/32B o_proj gains less than Qwen3-8B",
        "",
        "Formal latency is steady-state 10 x 1000 CUDA Graph replay. NCU uses cache-control all and is diagnostic. The GPU has 170 SMs and 102,400 B shared memory per SM.",
        "",
        "The cuSPARSELt base CTA uses 76,800 B shared memory and the complement CTA uses 69,632 B. Their sum is 146,432 B, so they cannot co-reside on one SM; one CTA from either grid occupies the SM shared-memory slot.",
        "",
        "## Formal latency and NCU CTA geometry",
        "",
        "| M | Model | Hybrid | cuBLAS | speedup | pure 2:4 | exposed overhead | base+comp CTAs / base-wave capacity | exposed comp CTAs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['M']} | {row['model_label']} | {fmt(row['hybrid_us'])} | {fmt(row['cublas_us'])} | "
            f"{fmt(row['speedup_vs_cublas'])}x | {fmt(row['pure24_us'])} | {fmt(row['formal_overhead_vs_pure24_us'])} | "
            f"{row['combined_grid_ctas']}/{row['base_wave_capacity_ctas']} | {row['exposed_ctas_beyond_base_waves']} |"
        )
    lines += [
        "",
        "## NCU correction kernel",
        "",
        "| M | Model | complement time | Tensor active | DRAM read | grid CTAs |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['M']} | {row['model_label']} | {fmt(row['ncu_complement_duration_us'])} us | "
            f"{fmt(row['ncu_complement_tensor_active_pct'], 1)}% | {fmt(row['ncu_complement_dram_read_bytes'] / 1e6, 1)} MB | "
            f"{row['complement_grid_ctas']} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "Qwen3-8B aligns with the 170-SM wave boundary: at M=512, 128 base CTAs + 32 complement CTAs = 160; at M=1024, 256 + 64 = 320, which fits in two 170-CTA waves. The complement can therefore be hidden in otherwise unused wave slots.",
        "",
        "Qwen3-14B/32B have N=5120: at M=512, 160 + 40 = 200, leaving 30 CTAs beyond one wave; at M=1024, 320 + 80 = 400, leaving 60 beyond two waves. Exactly 75% of the complement grid falls beyond the base-wave capacity. Since the two CTA types cannot share an SM, that correction tail is exposed on the critical path.",
        "",
        "The effect is not primarily a relative cuBLAS change: the pure-2:4 versus cuBLAS ratio stays broadly similar. Qwen3-32B's larger K=8192 lengthens both the base and the exposed correction CTA relative to Qwen3-14B K=5120.",
        "",
        "- [NCU rows](ncu_summary.csv)",
        "- [Combined analysis](scale_summary.csv)",
        "- [Speedup figure](figures/qwen_o_speedup.png)",
        "- [CTA-wave figure](figures/qwen_o_cta_waves.png)",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_qwen(root: Path, rows: list[dict[str, Any]]) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(QWEN_MODELS))
    width = 0.34
    for offset, m in ((-width / 2, 512), (width / 2, 1024)):
        selected = [next(row for row in rows if row["M"] == m and row["model"] == model) for model in QWEN_MODELS]
        ax.bar(x + offset, [row["speedup_vs_cublas"] for row in selected], width, label=f"M={m}")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(x, [MODEL_LABELS[model] for model in QWEN_MODELS])
    ax.set_ylabel("Speedup vs cuBLAS dense")
    ax.set_title("Residual-complement o_proj")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "qwen_o_speedup.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for axis, m in zip(axes, (512, 1024), strict=True):
        selected = [next(row for row in rows if row["M"] == m and row["model"] == model) for model in QWEN_MODELS]
        base = np.array([row["base_grid_ctas"] for row in selected])
        complement = np.array([row["complement_grid_ctas"] for row in selected])
        axis.bar(x, base, label="base CTAs")
        axis.bar(x, complement, bottom=base, label="complement CTAs")
        waves = 1 if m == 512 else 2
        axis.axhline(waves * SM_COUNT, color="red", linestyle="--", label=f"{waves} wave capacity")
        axis.set_xticks(x, ["8B", "14B", "32B"])
        axis.set_ylabel("CTA count")
        axis.set_title(f"M={m}")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "qwen_o_cta_waves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-ncu-root", type=Path, required=True)
    parser.add_argument("--llama-d27-root", type=Path, required=True)
    parser.add_argument("--llama-qkv-root", type=Path, required=True)
    parser.add_argument("--current-m2048-root", type=Path, required=True)
    parser.add_argument("--qwen-ncu-root", type=Path, required=True)
    parser.add_argument("--current-m512-root", type=Path, required=True)
    parser.add_argument("--current-m1024-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llama_ncu = parse_llama_ncu(args.llama_ncu_root)
    llama_e2e_rows = llama_e2e(args)
    llama_pair_rows = llama_pairs(llama_ncu)
    write_csv(args.llama_ncu_root / "ncu_summary.csv", llama_ncu)
    write_csv(args.llama_ncu_root / "ncu_pair_ratios.csv", llama_pair_rows)
    write_csv(args.llama_ncu_root / "e2e_summary.csv", llama_e2e_rows)
    write_llama_report(args.llama_ncu_root, llama_e2e_rows, llama_pair_rows)
    plot_llama(args.llama_ncu_root, llama_e2e_rows, llama_pair_rows)

    qwen_ncu = parse_qwen_ncu(args.qwen_ncu_root)
    qwen_rows = qwen_summary(args, qwen_ncu)
    write_csv(args.qwen_ncu_root / "ncu_summary.csv", qwen_ncu)
    write_csv(args.qwen_ncu_root / "scale_summary.csv", qwen_rows)
    write_qwen_report(args.qwen_ncu_root, qwen_rows)
    plot_qwen(args.qwen_ncu_root, qwen_rows)

    metadata = {
        "llama_ncu_reports": len(llama_ncu),
        "qwen_ncu_reports": len(qwen_ncu),
        "sm_count": SM_COUNT,
        "shared_memory_per_sm": SHARED_MEMORY_PER_SM,
        "ncu_cache_control": "all",
        "formal_latency_protocol": "100 warmup, 10 x 1000 graph replay median",
    }
    for root in (args.llama_ncu_root, args.qwen_ncu_root):
        (root / "analysis_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
