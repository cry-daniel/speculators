#!/usr/bin/env python3
"""Non-additive GPU breakdown of the current SpecLink D1 full layer.

The profiled graph preserves the current complement-first two-stream execution:
the dense-row stream performs indexed gather plus the Split-K2 complement, the
other stream launches the all-row cuSPARSELt base, and the origin stream waits
for both before indexed reduce/add. Timing events are embedded in the CUDA
Graph, so profiling does not serialize the base and complement branches.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
EVAL_ROOT = ROOT / "examples/evaluate/eval-guidellm"
BENCH_SCRIPTS = EVAL_ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))

from other_systems import benchmark_common as timing  # noqa: E402
from other_systems import bench_full_layer_five_models as formal_layer  # noqa: E402
import bench_decoder_layer_residual_complement as hybrid_layer  # noqa: E402
from speculators.speclink import (  # noqa: E402
    cusparselt_sparse_residual_indexed_gather,
    cusparselt_sparse_residual_residual_linear_splitk2,
    cusparselt_sparse_residual_sparse_linear,
    cusparselt_sparse_residual_splitk2_indexed_add_,
)


MODELS = formal_layer.MODELS
BATCH_SIZES = formal_layer.BATCH_SIZES
MODEL_LABELS = formal_layer.MODEL_LABELS
CATEGORIES = ("GEMM", "Attention", "Gather/Scatter", "Others")
CATEGORY_COLORS = {
    "GEMM": "#4c78a8",
    "Attention": "#f58518",
    "Gather/Scatter": "#54a24b",
    "Others": "#b279a2",
}
LINEARS = ("qkv", "o", "gate_up", "down")
SEED = formal_layer.SEED
REFERENCE_ROOT = (
    EVAL_ROOT
    / "results_final/five_model_full_layer_5_8_vs_speclink_d1_20260723"
)


@dataclass(frozen=True, slots=True)
class TimedPair:
    category: str
    component: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class ConcurrentGraphEvents:
    """Stable event handles reused by warmup, capture, and every replay."""

    def __init__(self) -> None:
        self.pairs: dict[tuple[str, str], TimedPair] = {}
        for name in LINEARS:
            self._add("Gather/Scatter", f"{name}.indexed_gather")
            self._add("GEMM", f"{name}.complement_splitk2")
            self._add("GEMM", f"{name}.base_cusparselt")
            self._add("Gather/Scatter", f"{name}.indexed_reduce_add")
        self._add("Attention", "qk_norm_rope_qk_softmax_av")
        for component in (
            "input_rmsnorm",
            "attention_residual_add",
            "post_attention_rmsnorm",
            "silu_times_up",
            "final_residual_add",
        ):
            self._add("Others", component)

    def _add(self, category: str, component: str) -> None:
        key = (category, component)
        if key in self.pairs:
            raise ValueError(f"duplicate timing pair: {key}")
        # external=True materializes graph event-record nodes whose timestamps
        # remain queryable after graph replay.
        self.pairs[key] = TimedPair(
            category,
            component,
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )

    def run(
        self,
        category: str,
        component: str,
        fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    ) -> Any:
        pair = self.pairs[(category, component)]
        pair.start.record()
        output = fn()
        pair.end.record()
        return output

    def component_us(self) -> dict[tuple[str, str], float]:
        return {
            key: 1000.0 * float(pair.start.elapsed_time(pair.end))
            for key, pair in self.pairs.items()
        }

    def category_us(self) -> dict[str, float]:
        totals = {category: 0.0 for category in CATEGORIES}
        for (category, _), duration in self.component_us().items():
            totals[category] += duration
        return totals


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def parse_csv(
    value: str, allowed: Sequence[str], label: str
) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid {label}: {unknown or value!r}"
        )
    return selected


def profiled_separate_linear(
    name: str,
    projection: hybrid_layer.Projection,
    x: torch.Tensor,
    dense_indices: torch.Tensor,
    resources: Any,
    events: ConcurrentGraphEvents,
) -> torch.Tensor:
    """Exact D1 separate path with graph-resident, per-branch timing."""

    x = x.contiguous()
    origin = torch.cuda.current_stream(x.device)
    resources.fork_event.record(origin)
    resources.dense_stream.wait_event(resources.fork_event)
    resources.sparse_stream.wait_event(resources.fork_event)

    # Match optimized_linear(): complement is submitted before the all-row
    # base so the short dense-row grid can claim SMs before cuSPARSELt.
    with torch.cuda.stream(resources.dense_stream):
        dense_x = events.run(
            "Gather/Scatter",
            f"{name}.indexed_gather",
            lambda: cusparselt_sparse_residual_indexed_gather(
                x, dense_indices
            ),
        )
        correction = events.run(
            "GEMM",
            f"{name}.complement_splitk2",
            lambda: cusparselt_sparse_residual_residual_linear_splitk2(
                dense_x, projection.runtime, variant="auto"
            ),
        )
    resources.dense_done_event.record(resources.dense_stream)

    with torch.cuda.stream(resources.sparse_stream):
        base = events.run(
            "GEMM",
            f"{name}.base_cusparselt",
            lambda: cusparselt_sparse_residual_sparse_linear(
                x, projection.runtime
            ),
        )
    resources.sparse_done_event.record(resources.sparse_stream)

    origin.wait_event(resources.sparse_done_event)
    origin.wait_event(resources.dense_done_event)
    return events.run(
        "Gather/Scatter",
        f"{name}.indexed_reduce_add",
        lambda: cusparselt_sparse_residual_splitk2_indexed_add_(
            base, correction, dense_indices
        ),
    )


def profiled_layer_forward(
    hidden: torch.Tensor,
    state: hybrid_layer.LayerState,
    dense_indices: torch.Tensor,
    events: ConcurrentGraphEvents,
) -> torch.Tensor:
    """Current full layer with four requested top-level categories."""

    batch = hidden.shape[0] // 8
    spec = state.spec

    def linear(name: str, x: torch.Tensor) -> torch.Tensor:
        return profiled_separate_linear(
            name,
            state.projections[name],
            x,
            dense_indices,
            state.resources,
            events,
        )

    residual = hidden
    x = events.run(
        "Others",
        "input_rmsnorm",
        lambda: hybrid_layer.rms_norm(
            hidden, state.input_norm, spec.rms_eps
        ),
    )
    qkv = linear("qkv", x)
    q_size = spec.q_heads * spec.head_dim
    kv_size = spec.kv_heads * spec.head_dim
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q = q.view(batch, 8, spec.q_heads, spec.head_dim)
    k = k.view(batch, 8, spec.kv_heads, spec.head_dim)
    v = v.view(batch, 8, spec.kv_heads, spec.head_dim)

    def attention() -> torch.Tensor:
        nonlocal q, k
        if spec.qwen_qk_norm:
            assert state.q_norm is not None and state.k_norm is not None
            q = hybrid_layer.rms_norm(q, state.q_norm, spec.rms_eps)
            k = hybrid_layer.rms_norm(k, state.k_norm, spec.rms_eps)
        q, k = hybrid_layer.apply_rope(
            q, k, state.rope_cos, state.rope_sin
        )
        scores, qg = hybrid_layer.attention_qk(
            q,
            k,
            state.past_k[:batch],
            spec.head_dim**-0.5,
            state.causal_mask,
        )
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return hybrid_layer.attention_av(
            probs, qg, v, state.past_v[:batch]
        )

    attn = events.run(
        "Attention", "qk_norm_rope_qk_softmax_av", attention
    )
    attn_out = linear("o", attn)
    residual = events.run(
        "Others",
        "attention_residual_add",
        lambda: residual.add(attn_out),
    )
    x = events.run(
        "Others",
        "post_attention_rmsnorm",
        lambda: hybrid_layer.rms_norm(
            residual, state.post_norm, spec.rms_eps
        ),
    )
    gate_up = linear("gate_up", x)
    gate, up = gate_up.chunk(2, dim=-1)
    x = events.run(
        "Others", "silu_times_up", lambda: F.silu(gate).mul_(up)
    )
    down = linear("down", x)
    return events.run(
        "Others", "final_residual_add", lambda: residual.add(down)
    )


def load_formal_e2e(
    reference_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    source = reference_root / "full_layer_results.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"missing formal E2E reference {source}; run "
            "other_systems/bench_full_layer_five_models.py first or pass "
            "--reference-root pointing to an existing formal run"
        )
    result: dict[tuple[str, int], dict[str, Any]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "speclink_d1":
                continue
            result[(row["model"], int(row["M"]))] = {
                "median_us": float(row["median_us"]),
                "p10_us": float(row["p10_us"]),
                "p90_us": float(row["p90_us"]),
                "algorithm_ids": json.loads(
                    row["cusparselt_algorithm_ids"]
                ),
            }
    return result


def install_formal_algorithms(
    state: hybrid_layer.LayerState, algorithm_ids: dict[str, int]
) -> None:
    """Use the exact cuSPARSELt choices recorded by the formal E2E run."""

    for name, algorithm_id in algorithm_ids.items():
        projection = state.projections[name]
        prepared = projection.runtime.cusparselt
        prepared.sparse_weight.alg_id_cusparselt = int(algorithm_id)
        prepared.algorithm_id = int(algorithm_id)
        projection.algorithm_id = int(algorithm_id)


def measure_profiled_graph(
    captured: Any,
    events: ConcurrentGraphEvents,
    eviction: torch.Tensor,
    *,
    warmup: int,
    trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for _ in range(warmup):
        captured.graph.replay()
    torch.cuda.synchronize()
    category_trials: list[dict[str, Any]] = []
    component_trials: list[dict[str, Any]] = []
    for trial in range(trials):
        eviction.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        captured.graph.replay()
        end.record()
        end.synchronize()
        e2e_us = 1000.0 * float(start.elapsed_time(end))
        component = events.component_us()
        category = {name: 0.0 for name in CATEGORIES}
        for (category_name, component_name), duration_us in component.items():
            category[category_name] += duration_us
            component_trials.append(
                {
                    "trial": trial,
                    "category": category_name,
                    "component": component_name,
                    "active_us": duration_us,
                    "instrumented_e2e_us": e2e_us,
                }
            )
        for category_name, duration_us in category.items():
            category_trials.append(
                {
                    "trial": trial,
                    "category": category_name,
                    "active_us": duration_us,
                    "instrumented_e2e_us": e2e_us,
                }
            )
    return category_trials, component_trials


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    formal = load_formal_e2e(args.reference_root)
    state = hybrid_layer.prepare_layer(
        args.worker_model, max(args.batch_sizes), device, args.seed
    )
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    eviction.zero_()
    category_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for batch in args.batch_sizes:
        m = batch * 8
        formal_row = formal[(args.worker_model, m)]
        algorithms = formal_row["algorithm_ids"]
        install_formal_algorithms(state, algorithms)
        hidden = hybrid_layer.make_hidden(
            args.worker_model, batch, device, args.seed
        )
        confidence = hybrid_layer.make_confidence(
            batch, device, args.seed
        )
        dense_indices = hybrid_layer.dense_indices_from_confidence(
            confidence, 1, "global"
        )
        if int(dense_indices.numel()) != m // 8:
            raise RuntimeError("D1 routing did not produce exactly M/8 rows")

        events = ConcurrentGraphEvents()
        captured = hybrid_layer.common.capture_multistream_graph(
            lambda: profiled_layer_forward(
                hidden, state, dense_indices, events
            ),
            state.resources,
            warmup=args.capture_warmup,
            device=device,
        )
        reference_timer = hybrid_layer.PhaseEvents()
        reference = hybrid_layer.layer_forward(
            hidden,
            state,
            method="residual_complement",
            dense_indices=dense_indices,
            timer=reference_timer,
        )
        reference_timer.finish()
        check = formal_layer.check_outputs(
            captured.output, reference, atol=0.2, rtol=0.1
        )
        check.update(
            {
                "model": args.worker_model,
                "M": m,
                "dense_rows": int(dense_indices.numel()),
                "sparse_rows": m - int(dense_indices.numel()),
                "cusparselt_algorithm_ids": json.dumps(
                    algorithms, sort_keys=True
                ),
            }
        )
        checks.append(check)

        category, component = measure_profiled_graph(
            captured,
            events,
            eviction,
            warmup=args.warmup,
            trials=args.trials,
        )
        common = {
            "model": args.worker_model,
            "model_label": MODEL_LABELS[args.worker_model],
            "batch_size": batch,
            "M": m,
            "dense_fraction": "1/8",
            "dense_rows": int(dense_indices.numel()),
            "sparse_rows": m - int(dense_indices.numel()),
            "formal_e2e_median_us": formal_row["median_us"],
            "formal_e2e_p10_us": formal_row["p10_us"],
            "formal_e2e_p90_us": formal_row["p90_us"],
        }
        category_rows.extend({**common, **row} for row in category)
        component_rows.extend({**common, **row} for row in component)
        medians = {
            category_name: statistics.median(
                row["active_us"]
                for row in category
                if row["category"] == category_name
            )
            for category_name in CATEGORIES
        }
        print(
            f"[breakdown] {args.worker_model} M={m}: "
            + ", ".join(
                f"{name}={medians[name]:.3f} us" for name in CATEGORIES
            )
            + f", formal E2E={formal_row['median_us']:.3f} us",
            flush=True,
        )
        del (
            hidden,
            confidence,
            dense_indices,
            events,
            captured,
            reference_timer,
            reference,
        )
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "model": args.worker_model,
        "category_rows": category_rows,
        "component_rows": component_rows,
        "checks": checks,
    }
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(values),
        "p10_us": percentile(values, 0.1),
        "p90_us": percentile(values, 0.9),
        "min_us": min(values),
        "max_us": max(values),
        "mean_us": statistics.mean(values),
    }


def aggregate(
    raw: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in raw:
        key = (str(row["model"]), int(row["M"]), str(row["category"]))
        groups.setdefault(key, []).append(row)
    case_e2e = {
        (str(row["model"]), int(row["M"])): float(
            row["formal_e2e_median_us"]
        )
        for row in raw
    }
    instrumented = {
        case: summarize(
            [
                float(row["instrumented_e2e_us"])
                for row in raw
                if (str(row["model"]), int(row["M"])) == case
                and row["category"] == CATEGORIES[0]
            ]
        )
        for case in case_e2e
    }
    medians_by_case: dict[tuple[str, int], dict[str, float]] = {}
    for (model, m, category), rows in groups.items():
        medians_by_case.setdefault((model, m), {})[category] = (
            statistics.median(float(row["active_us"]) for row in rows)
        )
    result: list[dict[str, Any]] = []
    for (model, m, category), rows in sorted(
        groups.items(),
        key=lambda item: (
            MODELS.index(item[0][0]),
            item[0][1],
            CATEGORIES.index(item[0][2]),
        ),
    ):
        stats = summarize([float(row["active_us"]) for row in rows])
        formal_e2e = case_e2e[(model, m)]
        sum_medians = sum(medians_by_case[(model, m)].values())
        result.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "batch_size": m // 8,
                "M": m,
                "dense_fraction": "1/8",
                "dense_rows": m // 8,
                "sparse_rows": m * 7 // 8,
                "category": category,
                **stats,
                "active_pct_of_formal_e2e": (
                    100.0 * stats["median_us"] / formal_e2e
                ),
                "active_pct_of_instrumented_e2e": (
                    100.0
                    * stats["median_us"]
                    / instrumented[(model, m)]["median_us"]
                ),
                "formal_e2e_median_us": formal_e2e,
                "instrumented_e2e_median_us": instrumented[(model, m)][
                    "median_us"
                ],
                "instrumented_e2e_p10_us": instrumented[(model, m)][
                    "p10_us"
                ],
                "instrumented_e2e_p90_us": instrumented[(model, m)][
                    "p90_us"
                ],
                "instrumentation_ratio_vs_formal_e2e": (
                    instrumented[(model, m)]["median_us"] / formal_e2e
                ),
                "sum_category_medians_us": sum_medians,
                "sum_category_medians_pct_of_formal_e2e": (
                    100.0 * sum_medians / formal_e2e
                ),
                "sum_category_medians_pct_of_instrumented_e2e": (
                    100.0
                    * sum_medians
                    / instrumented[(model, m)]["median_us"]
                ),
                "accounting": "non_additive_concurrent_active_time",
            }
        )
    return result


def plot_breakdown(
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    percent: bool,
) -> None:
    indexed = {
        (str(row["model"]), int(row["M"]), str(row["category"])): row
        for row in rows
    }
    figure, axes = plt.subplots(2, 3, figsize=(17.2, 9.6))
    width = 0.19
    for model, axis in zip(MODELS, axes.flat):
        m_values = sorted(
            {
                int(row["M"])
                for row in rows
                if row["model"] == model
            }
        )
        positions = np.arange(len(m_values), dtype=np.float64)
        offsets = (
            np.arange(len(CATEGORIES), dtype=np.float64)
            - (len(CATEGORIES) - 1) / 2
        ) * width
        for index, category in enumerate(CATEGORIES):
            field = (
                "active_pct_of_instrumented_e2e"
                if percent
                else "median_us"
            )
            values = [
                float(indexed[(model, m, category)][field])
                for m in m_values
            ]
            axis.bar(
                positions + offsets[index],
                values,
                width,
                color=CATEGORY_COLORS[category],
                label=category,
            )
        axis.set_xticks(positions, [f"M={m}" for m in m_values])
        axis.set_title(MODEL_LABELS[model])
        axis.grid(axis="y", alpha=0.22)
        if percent:
            axis.set_ylabel("Active time / profiled E2E (%)")
        else:
            axis.set_ylabel("Median active time (μs)")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.suptitle(
        "SpecLink D1 full-layer concurrent GPU breakdown",
        fontsize=16,
        y=0.997,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Current complement-first CUDA Graph; category times are "
            "non-additive because base and complement GEMMs overlap"
        ),
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def geometric_mean(values: Sequence[float]) -> float:
    return math.exp(statistics.mean(math.log(float(value)) for value in values))


def write_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    indexed = {
        (str(row["model"]), int(row["M"]), str(row["category"])): row
        for row in rows
    }
    instrumentation = {
        (str(row["model"]), int(row["M"])): float(
            row["instrumentation_ratio_vs_formal_e2e"]
        )
        for row in rows
    }
    category_ratios = {
        category: geometric_mean(
            [
                float(row["active_pct_of_instrumented_e2e"]) / 100.0
                for row in rows
                if row["category"] == category
            ]
        )
        for category in CATEGORIES
    }
    dominant = max(category_ratios, key=category_ratios.get)
    lines = [
        "# SpecLink D1 complete-layer concurrent breakdown",
        "",
        "This report profiles the same full decoder-layer implementation used "
        "by the five-model formal Layer comparison. D1 means one current "
        "token row per eight-row request group is dense: all rows execute the "
        "2:4 base, while M/8 rows concurrently execute the complementary "
        "2:4 stream.",
        "",
        "## Accounting",
        "",
        "- **GEMM** is the sum of the four cuSPARSELt base durations and four "
        "Split-K2 complement durations. Base and complement remain on separate "
        "streams, so this is active time, not elapsed wall time.",
        "- **Attention** includes Q/K norm when present, RoPE, QK, softmax, and "
        "AV.",
        "- **Gather/Scatter** includes indexed gather plus the fused Split-K "
        "reduce/indexed add for all four linears.",
        "- **Others** includes input/post-attention RMSNorm, residual adds, and "
        "SiLU-times-Up.",
        "",
        "The categories are intentionally **non-additive**. In particular, the "
        "two GEMM streams overlap, and gather may overlap the base. Formal E2E "
        "comes from the existing 10 trials × 1000 CUDA Graph replays result. "
        "Category percentages use the same-run instrumented E2E denominator; "
        "the formal E2E remains the authoritative latency.",
        "",
        "## Protocol",
        "",
        f"- Five model shapes; M={','.join(str(batch * 8) for batch in args.batch_sizes)}; "
        "context length 128; seven draft tokens plus one current token.",
        f"- {args.capture_warmup} capture warmups, {args.warmup} graph replay "
        f"warmups, {args.trials} independent profiled replays, and "
        f"{args.eviction_mib} MiB eviction before each replay.",
        "- CUDA timing events are graph nodes. No synchronization is inserted "
        "between the concurrent base and complement paths.",
        "",
        f"Across the {len(instrumentation)} cases, the largest geometric-mean "
        "active-time/E2E "
        f"ratio is **{dominant} ({category_ratios[dominant]:.2%})**. The "
        "instrumented/formal E2E ratio is "
        f"**{geometric_mean(list(instrumentation.values())):.4f}x**. This is "
        "a sanity comparison that includes both event-node perturbation and "
        "run-to-run GPU clock variation.",
    ]
    for model in args.models:
        lines.extend(
            [
                "",
                f"## {MODEL_LABELS[model]}",
                "",
                "| M | Formal E2E | GEMM | Attention | Gather/Scatter | "
                "Others | Sum active / profiled E2E | Instrumented / formal |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in (batch * 8 for batch in args.batch_sizes):
            selected = {
                category: indexed[(model, m, category)]
                for category in CATEGORIES
            }
            formal_e2e = float(
                selected[CATEGORIES[0]]["formal_e2e_median_us"]
            )
            values = [
                float(selected[category]["median_us"])
                for category in CATEGORIES
            ]
            sum_pct = float(
                selected[CATEGORIES[0]][
                    "sum_category_medians_pct_of_instrumented_e2e"
                ]
            )
            lines.append(
                f"| {m} | {formal_e2e:.3f} μs | "
                + " | ".join(f"{value:.3f} μs" for value in values)
                + f" | {sum_pct:.2f}% | "
                f"{instrumentation[(model, m)]:.4f}x |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `speclink_d1_breakdown.csv`: requested four-category medians, "
            "P10/P90, E2E ratios, and non-additive accounting label.",
            "- `speclink_d1_breakdown_raw_trials.csv`: every category total "
            "from all ten trials.",
            "- `speclink_d1_breakdown_component_trials.csv`: auditable "
            "per-linear/per-stage event durations.",
            "- `correctness.json`: profiled-vs-diagnostic output checks.",
            "- `figures/speclink_d1_breakdown_pct.png`: active-time/E2E view.",
            "- `figures/speclink_d1_breakdown_us.png`: absolute-time view.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_coordinator(args: argparse.Namespace) -> None:
    formal = load_formal_e2e(args.reference_root)
    missing = [
        (model, batch * 8)
        for model in args.models
        for batch in args.batch_sizes
        if (model, batch * 8) not in formal
    ]
    if missing:
        raise RuntimeError(f"formal E2E reference is missing cases: {missing}")
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    worker_files: list[Path] = []
    for model in args.models:
        worker_output = work / f"{model}.json"
        if args.resume and worker_output.exists():
            print(f"[resume] {model}", flush=True)
            worker_files.append(worker_output)
            continue
        timing.assert_gpu_idle(args.device)
        command = [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--worker-model",
            model,
            "--worker-output",
            str(worker_output),
            "--batch-sizes",
            ",".join(map(str, args.batch_sizes)),
            "--reference-root",
            str(args.reference_root),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--capture-warmup",
            str(args.capture_warmup),
            "--warmup",
            str(args.warmup),
            "--trials",
            str(args.trials),
            "--eviction-mib",
            str(args.eviction_mib),
        ]
        if args.smoke:
            command.append("--smoke")
        print(f"[worker] {model}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        worker_files.append(worker_output)

    raw: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for worker_file in worker_files:
        payload = json.loads(worker_file.read_text(encoding="utf-8"))
        raw.extend(payload["category_rows"])
        components.extend(payload["component_rows"])
        checks.extend(payload["checks"])
    expected = len(args.models) * len(args.batch_sizes) * args.trials * len(
        CATEGORIES
    )
    if len(raw) != expected:
        raise RuntimeError(f"expected {expected} category trials, got {len(raw)}")
    summary = aggregate(raw)
    write_csv(output / "speclink_d1_breakdown.csv", summary)
    write_csv(output / "speclink_d1_breakdown_raw_trials.csv", raw)
    write_csv(
        output / "speclink_d1_breakdown_component_trials.csv", components
    )
    (output / "correctness.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "models": list(args.models),
        "M_values": [batch * 8 for batch in args.batch_sizes],
        "method": "SpecLink D1 current concurrent separate",
        "categories": list(CATEGORIES),
        "accounting": "non_additive_concurrent_active_time",
        "formal_e2e_reference": str(args.reference_root.resolve()),
        "protocol": {
            "capture_warmup": args.capture_warmup,
            "profile_warmup": args.warmup,
            "profile_trials": args.trials,
            "profile_replays_per_trial": 1,
            "eviction_mib_before_each_trial": args.eviction_mib,
            "formal_e2e": "10 trials x 1000 CUDA Graph replays",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figures = output / "figures"
    plot_breakdown(
        summary, figures / "speclink_d1_breakdown_pct.png", percent=True
    )
    plot_breakdown(
        summary, figures / "speclink_d1_breakdown_us.png", percent=False
    )
    write_report(output / "report.md", summary, args)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--batch-sizes", default=",".join(map(str, BATCH_SIZES))
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REFERENCE_ROOT / "speclink_d1_breakdown",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "temp/five_model_full_layer_speclink_d1_breakdown_20260723"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        args.models = parse_csv(args.models, MODELS, "models")
        args.batch_sizes = tuple(
            int(item) for item in args.batch_sizes.split(",") if item
        )
        if (
            not args.batch_sizes
            or any(item not in BATCH_SIZES for item in args.batch_sizes)
        ):
            raise argparse.ArgumentTypeError(
                f"batch sizes must be selected from {BATCH_SIZES}"
            )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.worker and (
        args.worker_model is None or args.worker_output is None
    ):
        parser.error(
            "worker mode requires --worker-model and --worker-output"
        )
    if args.smoke:
        args.models = args.models[:1]
        args.batch_sizes = args.batch_sizes[:1]
        args.capture_warmup = 2
        args.warmup = 3
        args.trials = 2
        if not args.worker:
            args.output_root = EVAL_ROOT / "temp/speclink_d1_breakdown_smoke"
            args.work_root = (
                EVAL_ROOT / "temp/speclink_d1_breakdown_smoke_work"
            )
    elif args.trials != 10:
        parser.error("formal breakdown protocol requires exactly 10 trials")
    if any(
        value <= 0
        for value in (
            args.capture_warmup,
            args.warmup,
            args.trials,
            args.eviction_mib,
        )
    ):
        parser.error("all protocol counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
