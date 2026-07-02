#!/usr/bin/env python3
"""Estimate whether SR24 routing can meet a speed target with an operator.

This is an offline guardrail. It joins two existing diagnostics:

1. component microbench rows, usually from
   ``profile_speclink_sr24_component_breakdown.py``;
2. accepted-base-only risk projections from
   ``analyze_sr24_acceptance_trace.py``.

The goal is to catch impossible controller sweeps early. If the residual
fraction needed for quality is far above the residual fraction where the mixed
operator can beat dense, the next optimization must be an operator change
rather than another routing-threshold sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return EVAL_ROOT / path


def _resolve_output(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise SystemExit(f"expected JSON list in {path}")
        return [row for row in value if isinstance(row, dict)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _shape_match(row: dict[str, Any], rows: int, out_features: int,
                 in_features: int) -> bool:
    row_count = int(float(row.get("rows") or -1))
    if row_count != rows:
        return False
    # Linear microbench uses out_features/in_features. Whole-MLP microbench
    # uses hidden_size/intermediate_size. Keep the CLI names generic enough for
    # both so the same ceiling gate can compare both operator families.
    row_out = row.get("out_features", row.get("hidden_size"))
    row_in = row.get("in_features", row.get("intermediate_size"))
    return int(float(row_out or -1)) == out_features and int(
        float(row_in or -1)
    ) == in_features


def _component_points(
    rows: list[dict[str, Any]],
    *,
    shape_rows: int,
    out_features: int,
    in_features: int,
    operator_col: str,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        if not _shape_match(row, shape_rows, out_features, in_features):
            continue
        residual_fraction = _float(row.get("residual_fraction"))
        if residual_fraction is None:
            bucket_size = _float(row.get("bucket_size"))
            row_count = _float(row.get("rows"))
            if bucket_size is not None and row_count:
                residual_fraction = bucket_size / row_count
        dense_ms = _float(row.get("dense_graph_ms")) or _float(row.get("dense_ms"))
        operator_ms = _float(row.get(operator_col))
        base_ms = _float(row.get("base_sparse_graph_ms"))
        if residual_fraction is None or dense_ms is None or operator_ms is None:
            continue
        points.append({
            "residual_fraction": residual_fraction,
            "dense_ms": dense_ms,
            "operator_ms": operator_ms,
            "operator_over_dense": operator_ms / dense_ms,
            "operator_speedup_vs_dense": dense_ms / operator_ms,
            "base_sparse_graph_ms": base_ms if base_ms is not None else "",
        })
    points.sort(key=lambda row: row["residual_fraction"])
    return points


def _interpolate(points: list[dict[str, float]], x: float, key: str) -> float | None:
    if not points:
        return None
    if x <= points[0]["residual_fraction"]:
        return points[0][key]
    if x >= points[-1]["residual_fraction"]:
        return points[-1][key]
    for left, right in zip(points, points[1:]):
        lx = left["residual_fraction"]
        rx = right["residual_fraction"]
        if lx <= x <= rx:
            if rx == lx:
                return left[key]
            ratio = (x - lx) / (rx - lx)
            return left[key] + ratio * (right[key] - left[key])
    return None


def _max_fraction_for_speedup(
    points: list[dict[str, float]],
    target_speedup: float,
) -> float | None:
    if not points:
        return None
    target_ratio = 1.0 / target_speedup
    ordered = sorted(points, key=lambda row: row["residual_fraction"])
    last_ok: float | None = None
    for left, right in zip(ordered, ordered[1:]):
        left_ratio = left["operator_over_dense"]
        right_ratio = right["operator_over_dense"]
        left_x = left["residual_fraction"]
        right_x = right["residual_fraction"]
        if left_ratio <= target_ratio:
            last_ok = left_x
        if (left_ratio - target_ratio) * (right_ratio - target_ratio) <= 0:
            if right_ratio == left_ratio:
                candidate = max(left_x, right_x)
            else:
                candidate = left_x + (
                    (target_ratio - left_ratio)
                    * (right_x - left_x)
                    / (right_ratio - left_ratio)
                )
            if left_ratio <= target_ratio or right_ratio <= target_ratio:
                last_ok = max(last_ok if last_ok is not None else candidate, candidate)
    if ordered[-1]["operator_over_dense"] <= target_ratio:
        last_ok = ordered[-1]["residual_fraction"]
    return last_ok


def _best_projection_rows(
    projection_rows: list[dict[str, str]],
    targets: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for target in targets:
        candidates: list[dict[str, str]] = []
        for row in projection_rows:
            accepted_base = _float(row.get("accepted_base_only_fraction"))
            residual_fraction = _float(
                row.get("projected_residual_fraction")
                or row.get("sr24_residual_fraction")
            )
            if accepted_base is None or residual_fraction is None:
                continue
            if accepted_base <= target:
                candidates.append(row)
        if not candidates:
            out.append({
                "accepted_base_target": target,
                "projection_found": False,
            })
            continue
        best = min(
            candidates,
            key=lambda row: (
                _float(row.get("projected_residual_fraction")) or 1.0,
                _float(row.get("projected_prefix_residual_len")) or 0.0,
                _float(row.get("projected_score_threshold")) or 0.0,
            ),
        )
        out.append({
            "accepted_base_target": target,
            "projection_found": True,
            "prefix": best.get("projected_prefix_residual_len", ""),
            "threshold": best.get("projected_score_threshold", ""),
            "projected_residual_fraction": (
                _float(best.get("projected_residual_fraction")) or 0.0
            ),
            "accepted_base_only_fraction": (
                _float(best.get("accepted_base_only_fraction")) or 0.0
            ),
            "steps_with_accepted_base_only_fraction": (
                _float(best.get("steps_with_accepted_base_only_fraction")) or 0.0
            ),
            "mean_projected_residual_rows_per_step": (
                _float(best.get("mean_projected_residual_rows_per_step")) or 0.0
            ),
        })
    return out


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    points: list[dict[str, float]],
    speed_fraction: float | None,
    joined_rows: list[dict[str, Any]],
) -> None:
    lines = [
    "# SR24 Speed/Quality Ceiling",
        "",
        "This is an offline feasibility check for a selected SR24 operator.",
        "It should be read before spending GPU time on another routing sweep.",
        "",
        "## Inputs",
        "",
        f"- component CSV: `{args.component_csv.resolve()}`",
        f"- projection CSV: `{args.projection_csv.resolve()}`",
        f"- shape: rows={args.shape_rows}, out={args.out_features}, in={args.in_features}",
        f"- operator column: `{args.operator_col}`",
        f"- target operator speedup: `{args.target_speedup:.3f}x`",
        "",
        "## Component Points",
        "",
        "| residual frac | dense ms | operator ms | operator/dense | speedup vs dense |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in points:
        lines.append(
            "| {frac} | {dense} | {op} | {ratio} | {speedup} |".format(
                frac=_fmt(row["residual_fraction"]),
                dense=_fmt(row["dense_ms"]),
                op=_fmt(row["operator_ms"]),
                ratio=_fmt(row["operator_over_dense"]),
                speedup=_fmt(row["operator_speedup_vs_dense"]),
            )
        )
    lines.extend([
        "",
        "## Quality Targets",
        "",
        "| accepted-base target | prefix | threshold | projected residual frac | accepted base-only frac | projected operator speedup | meets target speed? |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in joined_rows:
        lines.append(
            "| {target} | {prefix} | {threshold} | {residual} | {accepted} | {speedup} | {meets} |".format(
                target=_fmt(row.get("accepted_base_target"), 2),
                prefix=row.get("prefix", ""),
                threshold=row.get("threshold", ""),
                residual=_fmt(row.get("projected_residual_fraction")),
                accepted=_fmt(row.get("accepted_base_only_fraction")),
                speedup=_fmt(row.get("projected_operator_speedup")),
                meets=row.get("meets_speed_target", ""),
            )
        )
    lines.extend(["", "## Read", ""])
    if speed_fraction is None:
        lines.append(
            f"- The measured operator never reaches `{args.target_speedup:.3f}x` "
            "in the provided component rows."
        )
    else:
        lines.append(
            f"- With the current operator, `{args.target_speedup:.3f}x` requires "
            f"residual fraction <= `{speed_fraction:.4f}` for this shape."
        )
    impossible = [
        row for row in joined_rows
        if row.get("projection_found") and not row.get("meets_speed_target")
    ]
    if impossible:
        lines.append(
            "- The listed quality targets require more residual rows than the "
            "current mixed operator can afford. This points to a fused/packed "
            "operator change, not another threshold-only controller sweep."
        )
    else:
        lines.append(
            "- At least one listed quality target is compatible with the measured "
            "operator speed target; it is worth a live serving smoke."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-csv", type=Path, required=True)
    parser.add_argument("--projection-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shape-rows", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=28672)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--operator-col", default="bucket_delta_inplace_graph_ms")
    parser.add_argument("--target-speedup", type=float, default=1.2)
    parser.add_argument(
        "--accepted-base-targets",
        default="0.20,0.10,0.05,0.02,0.00",
        help="Comma-separated accepted base-only risk targets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.component_csv = _resolve(args.component_csv)
    args.projection_csv = _resolve(args.projection_csv)
    output_root = _resolve_output(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    component_rows = _read_table(args.component_csv)
    projection_rows = _read_table(args.projection_csv)
    points = _component_points(
        component_rows,
        shape_rows=args.shape_rows,
        out_features=args.out_features,
        in_features=args.in_features,
        operator_col=args.operator_col,
    )
    if len(points) < 2:
        raise SystemExit("need at least two component points for the requested shape")

    speed_fraction = _max_fraction_for_speedup(points, args.target_speedup)
    targets = [
        float(part)
        for part in args.accepted_base_targets.split(",")
        if part.strip()
    ]
    quality_rows = _best_projection_rows(projection_rows, targets)
    joined_rows: list[dict[str, Any]] = []
    for row in quality_rows:
        residual_fraction = _float(row.get("projected_residual_fraction"))
        operator_speedup = (
            _interpolate(points, residual_fraction, "operator_speedup_vs_dense")
            if residual_fraction is not None
            else None
        )
        joined = dict(row)
        joined["projected_operator_speedup"] = operator_speedup
        joined["speed_target_residual_fraction_max"] = speed_fraction
        joined["meets_speed_target"] = (
            bool(operator_speedup is not None and operator_speedup >= args.target_speedup)
            if row.get("projection_found")
            else False
        )
        joined_rows.append(joined)

    _write_csv(output_root / "speed_quality_ceiling.csv", joined_rows)
    _write_csv(output_root / "component_points.csv", points)
    _write_report(
        output_root / "report.md",
        args=args,
        points=points,
        speed_fraction=speed_fraction,
        joined_rows=joined_rows,
    )
    print((output_root / "report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
