#!/usr/bin/env python3
"""Measure the shape-matched pure-2:4 cuSPARSELt kernel upper bound.

The benchmark uses the same five model shapes, M values, HBM-cold protocol,
CUDA Graph timing, and synthetic BF16 inputs as ``bench_five_models.py``.
Workers are isolated by (model, projection) so the largest Llama-3-70B weight
does not remain resident while the next shape is prepared.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

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

from other_systems import benchmark_common as common  # noqa: E402
import sparse24_benchmark_common as hybrid_common  # noqa: E402
from speculators.speclink import (  # noqa: E402
    TP1_FUSED_WEIGHT_SHAPES,
    prepare_sparse24_weight,
    select_cusparselt_algorithm,
    sparse24_linear,
)


MODELS = tuple(TP1_FUSED_WEIGHT_SHAPES)
PROJECTIONS = ("qkv", "o", "gate_up", "down")
M_VALUES = (512, 1024, 2048)
MODEL_LABELS = {
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "llama3_70b": "Llama-3-70B",
}
SEED = 20260723


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"invalid {label}: {unknown or value!r}"
        )
    return selected


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


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = args.worker_model
    projection = args.worker_projection
    n, k = TP1_FUSED_WEIGHT_SHAPES[model][projection]
    weight_case = hybrid_common.ShapeCase(
        model, projection, max(args.m_values), k, n
    )
    dense_weight, sparse_weight = hybrid_common.make_synthetic_weight(
        weight_case, args.seed, device
    )
    del dense_weight
    prepared = prepare_sparse24_weight(sparse_weight)
    eviction = torch.empty(
        args.eviction_mib * 1024 * 1024,
        device=device,
        dtype=torch.uint8,
    )
    eviction.zero_()

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for m in args.m_values:
        case = hybrid_common.ShapeCase(model, projection, m, k, n)
        x = hybrid_common.make_input(
            case,
            args.seed,
            device,
            purpose="five_model_comparison",
        )
        algorithm_id = select_cusparselt_algorithm(prepared, x)
        captured = common.capture(
            lambda: sparse24_linear(x, prepared),
            warmup=args.capture_warmup,
        )
        expected = F.linear(x, sparse_weight)
        check = common.correctness(
            captured.output,
            expected,
            atol=0.06,
            rtol=0.06,
        )
        summary, samples = common.formal_measure(
            captured,
            eviction,
            warmup=args.warmup,
            trials=args.trials,
            replays=args.replays,
        )
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "projection": projection,
            "M": m,
            "N": n,
            "K": k,
            "method": "pure_24_upper_bound",
            "method_label": "2:4 Upper Bound",
            "sparsity_format": "2:4",
            "semantic": "all_tokens_pure_2_4",
            "weight_density": 0.5,
            "cusparselt_algorithm_id": algorithm_id,
            **summary.as_dict(),
            **check,
        }
        rows.append(row)
        raw.extend(
            {
                "model": model,
                "projection": projection,
                "M": m,
                "N": n,
                "K": k,
                "method": "pure_24_upper_bound",
                "trial": trial,
                "latency_us": latency,
                "cusparselt_algorithm_id": algorithm_id,
            }
            for trial, latency in enumerate(samples)
        )
        print(
            f"[pure24] {model}/{projection} M={m}: "
            f"{summary.median_us:.3f} us (alg={algorithm_id})",
            flush=True,
        )
        del captured, expected, x
        gc.collect()
        torch.cuda.empty_cache()

    payload = {"rows": rows, "raw": raw}
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    work = args.work_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    worker_files: list[Path] = []
    for model in args.models:
        for projection in args.projections:
            worker_output = work / f"{model}__{projection}.json"
            if args.resume and worker_output.exists():
                print(f"[resume] {model}/{projection}", flush=True)
                worker_files.append(worker_output)
                continue
            common.assert_gpu_idle(args.device)
            command = [
                sys.executable,
                str(SCRIPT),
                "--worker",
                "--worker-model",
                model,
                "--worker-projection",
                projection,
                "--worker-output",
                str(worker_output),
                "--device",
                str(args.device),
                "--m-values",
                ",".join(map(str, args.m_values)),
                "--seed",
                str(args.seed),
                "--capture-warmup",
                str(args.capture_warmup),
                "--warmup",
                str(args.warmup),
                "--trials",
                str(args.trials),
                "--replays",
                str(args.replays),
                "--eviction-mib",
                str(args.eviction_mib),
            ]
            print(f"[worker] {model}/{projection}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            worker_files.append(worker_output)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for path in worker_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        raw.extend(payload["raw"])
    expected = len(args.models) * len(args.projections) * len(args.m_values)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(rows)}")
    rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            PROJECTIONS.index(row["projection"]),
            int(row["M"]),
        )
    )
    write_csv(output / "pure24_upper_bound.csv", rows)
    write_csv(output / "pure24_upper_bound_raw_trials.csv", raw)
    metadata = {
        "models": list(args.models),
        "projections": list(args.projections),
        "M_values": list(args.m_values),
        "method": "pure cuSPARSELt 2:4 over all token rows",
        "plot_role": "2:4 Upper Bound",
        "protocol": {
            "warmups": args.warmup,
            "trials": args.trials,
            "replays_per_trial": args.replays,
            "eviction_mib_before_each_trial": args.eviction_mib,
            "timing": "CUDA Event total interval divided by replay count",
            "synchronization_inside_timed_interval": False,
        },
    }
    (output / "pure24_upper_bound_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output / "pure24_upper_bound.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", choices=MODELS)
    parser.add_argument("--worker-projection", choices=PROJECTIONS)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--projections", default=",".join(PROJECTIONS))
    parser.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "results_final/five_model_nm_vs_speclink_kernel_20260723"
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=(
            EVAL_ROOT
            / "temp/pure24_upper_bound_five_models_20260723"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        args.models = parse_csv(args.models, MODELS, "models")
        args.projections = parse_csv(
            args.projections, PROJECTIONS, "projections"
        )
        args.m_values = tuple(
            int(item) for item in args.m_values.split(",") if item
        )
        if (
            not args.m_values
            or any(item not in M_VALUES for item in args.m_values)
        ):
            raise argparse.ArgumentTypeError(
                f"M must be selected from {M_VALUES}"
            )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.worker and (
        args.worker_model is None
        or args.worker_projection is None
        or args.worker_output is None
    ):
        parser.error(
            "worker mode requires --worker-model, --worker-projection, "
            "and --worker-output"
        )
    if args.trials != 10:
        parser.error("formal protocol requires exactly 10 trials")
    counts = (
        args.capture_warmup,
        args.warmup,
        args.trials,
        args.replays,
        args.eviction_mib,
    )
    if any(value <= 0 for value in counts):
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
