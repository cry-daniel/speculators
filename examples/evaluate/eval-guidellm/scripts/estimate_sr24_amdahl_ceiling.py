#!/usr/bin/env python3
"""Estimate SR24 end-to-end speed ceilings from layer/leaf coverage.

This is an offline guardrail for the current SR24 optimization work. It answers
whether a proposed 2:4 target scope covers enough verifier/TLM compute to ever
reach an end-to-end speed target, before spending time on another live serving
run.

The estimate is intentionally optimistic:

* it assumes every token in the selected layers/leaves benefits from the
  operator speedup;
* it ignores scheduler, DLM, KV-cache, sampling, networking, and graph overhead;
* it treats matrix multiply FLOPs as the dominant TLM verifier cost.

If an optimistic row cannot reach 1.2x, the real serving path cannot reach it
with that scope either.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("empty float list")
    return values


def parse_layer_list(value: str, *, num_layers: int) -> set[int]:
    layers: set[int] = set()
    value = value.strip()
    if not value:
        return layers
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise argparse.ArgumentTypeError(f"bad layer range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    bad = [idx for idx in layers if idx < 0 or idx >= num_layers]
    if bad:
        raise argparse.ArgumentTypeError(f"layers out of range: {bad}")
    return layers


def parse_target_scope(value: str, *, num_layers: int) -> dict[str, set[int]]:
    """Parse `gate_up_proj=16-31;down_proj=14-31` style scopes."""
    scope: dict[str, set[int]] = {}
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                "target scope entries must look like leaf=layers"
            )
        leaf, layers_raw = entry.split("=", 1)
        leaf = leaf.strip()
        if leaf not in {"gate_up_proj", "down_proj", "qkv_proj", "o_proj"}:
            raise argparse.ArgumentTypeError(f"unsupported leaf: {leaf}")
        scope.setdefault(leaf, set()).update(
            parse_layer_list(layers_raw, num_layers=num_layers)
        )
    return scope


def layer_flops(args: argparse.Namespace) -> dict[str, float]:
    hidden = float(args.hidden_size)
    intermediate = float(args.intermediate_size)
    head_dim = float(args.head_dim)
    q_out = float(args.num_attention_heads) * head_dim
    kv_out = float(args.num_key_value_heads) * head_dim
    return {
        "qkv_proj": hidden * (q_out + 2.0 * kv_out),
        "o_proj": q_out * hidden,
        "gate_up_proj": 2.0 * hidden * intermediate,
        "down_proj": hidden * intermediate,
    }


def default_scopes(num_layers: int) -> dict[str, str]:
    last = num_layers - 1
    return {
        "down_front14": f"down_proj=14-{last}",
        "down_front13": f"down_proj=13-{last}",
        "prefix1_front28_mlp_tail": f"gate_up_proj=28-{last};down_proj=28-{last}",
        "gateup16_31": f"gate_up_proj=16-{last}",
        "all_mlp": f"gate_up_proj=0-{last};down_proj=0-{last}",
        "all_down": f"down_proj=0-{last}",
    }


def amdahl_speedup(target_fraction: float, operator_speedup: float) -> float:
    if operator_speedup <= 0.0:
        return math.nan
    return 1.0 / ((1.0 - target_fraction) + target_fraction / operator_speedup)


def required_fraction(global_speedup: float, operator_speedup: float) -> float | None:
    if global_speedup <= 1.0 or operator_speedup <= 1.0:
        return None
    numerator = (1.0 / global_speedup) - 1.0
    denominator = (1.0 / operator_speedup) - 1.0
    if denominator == 0.0:
        return None
    return numerator / denominator


def estimate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    flops = layer_flops(args)
    total_per_layer = sum(flops.values())
    total_model = total_per_layer * float(args.num_layers)
    scopes = default_scopes(args.num_layers)
    for item in args.scope:
        if ":" not in item:
            raise SystemExit("--scope entries must be name:leaf=layers")
        name, scope = item.split(":", 1)
        scopes[name.strip()] = scope.strip()

    rows: list[dict[str, Any]] = []
    for scope_name, scope_raw in scopes.items():
        scope = parse_target_scope(scope_raw, num_layers=args.num_layers)
        target_flops = 0.0
        selected: dict[str, str] = {}
        for leaf, layers in sorted(scope.items()):
            target_flops += flops[leaf] * float(len(layers))
            if layers:
                selected[leaf] = (
                    f"{min(layers)}-{max(layers)}"
                    if len(layers) > 1
                    else str(next(iter(layers)))
                )
        fraction = target_flops / total_model if total_model > 0.0 else math.nan
        max_infinite_speedup = 1.0 / max(1.0 - fraction, 1e-12)
        for op_speedup in args.operator_speedups:
            global_speedup = amdahl_speedup(fraction, op_speedup)
            need_fraction = required_fraction(args.target_global_speedup, op_speedup)
            rows.append({
                "scope": scope_name,
                "scope_raw": scope_raw,
                "selected_layers": json.dumps(selected, sort_keys=True),
                "target_compute_fraction": fraction,
                "operator_speedup": op_speedup,
                "optimistic_global_speedup": global_speedup,
                "max_global_speedup_if_free": max_infinite_speedup,
                "required_compute_fraction_for_target": (
                    "" if need_fraction is None else need_fraction
                ),
                "can_reach_target_with_this_scope": (
                    global_speedup >= args.target_global_speedup
                ),
                "can_reach_target_even_if_free": (
                    max_infinite_speedup >= args.target_global_speedup
                ),
            })
    return rows


def write_outputs(output_root: Path, rows: list[dict[str, Any]],
                  args: argparse.Namespace) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "ceiling.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "ceiling.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# SR24 Amdahl Ceiling\n\n")
        f.write(
            "This is an optimistic compute-coverage estimate. If a row cannot "
            "reach the target here, live vLLM serving cannot reach it with the "
            "same target scope.\n\n"
        )
        f.write("## Model Defaults\n\n")
        f.write(f"- layers: `{args.num_layers}`\n")
        f.write(f"- hidden: `{args.hidden_size}`\n")
        f.write(f"- intermediate: `{args.intermediate_size}`\n")
        f.write(f"- attention heads / KV heads / head dim: "
                f"`{args.num_attention_heads}` / "
                f"`{args.num_key_value_heads}` / `{args.head_dim}`\n")
        f.write(f"- target global speedup: `{args.target_global_speedup:.3f}x`\n\n")
        f.write(
            "| scope | target compute % | operator speedup | optimistic global "
            "speedup | free-op ceiling | can reach target |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['scope']} | "
                f"{100.0 * float(row['target_compute_fraction']):.2f}% | "
                f"{float(row['operator_speedup']):.3f}x | "
                f"{float(row['optimistic_global_speedup']):.3f}x | "
                f"{float(row['max_global_speedup_if_free']):.3f}x | "
                f"{row['can_reach_target_with_this_scope']} |\n"
            )
        f.write("\n## Read\n\n")
        f.write(
            "- `free-op ceiling` is the absolute upper bound if the selected "
            "leaf/layer compute became free.\n"
        )
        f.write(
            "- Down-only tail scopes are useful diagnostics, but they do not "
            "cover enough TLM compute to carry a 1.2x serving claim unless the "
            "rest of the stack also improves.\n"
        )
        f.write(
            "- A real 1.2x path needs either broad MLP coverage with a quality "
            "controller or an operator that reduces a much larger fraction of "
            "the verifier step than down_proj tail layers alone.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--target-global-speedup", type=float, default=1.2)
    parser.add_argument(
        "--operator-speedups",
        type=parse_float_list,
        default=parse_float_list("1.225,1.5,2.0"),
        help=(
            "Comma-separated local operator speedups to test. 1.225 matches the "
            "best down_proj prefix2 effective-bs64 microbench point."
        ),
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Extra scope as name:leaf=layers[;leaf=layers], e.g. my:down_proj=14-31",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results.bak") / f"sr24_amdahl_ceiling_{timestamp()}",
    )
    args = parser.parse_args()

    rows = estimate_rows(args)
    write_outputs(args.output_root, rows, args)
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
