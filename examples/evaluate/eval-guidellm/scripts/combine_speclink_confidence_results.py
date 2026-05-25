#!/usr/bin/env python3
"""Combine confidence trace result roots and rerun analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_speclink_confidence_acceptance import analyze


def parse_source(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) == 2:
        model_label, root = parts
        dataset_label = "math"
    elif len(parts) == 3:
        model_label, dataset_label, root = parts
    else:
        raise argparse.ArgumentTypeError(
            "source must be MODEL:RESULT_ROOT or MODEL:DATASET:RESULT_ROOT"
        )
    model_label = model_label.strip()
    dataset_label = dataset_label.strip()
    if not model_label or not dataset_label:
        raise argparse.ArgumentTypeError("source model and dataset labels cannot be empty")
    path = Path(root).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"source root does not exist: {path}")
    return model_label, dataset_label, path


def copy_trace(
    model_label: str,
    dataset_label: str,
    source: Path,
    output: Path,
    *,
    method: str | None,
    num_spec_tokens: int | None,
) -> int:
    trace_dir = source / "trace"
    if not trace_dir.exists():
        raise SystemExit(f"missing trace directory: {trace_dir}")
    out_trace = output / "trace"
    out_trace.mkdir(parents=True, exist_ok=True)
    rows = 0
    for path in sorted(trace_dir.glob("*.jsonl")):
        name = path.name
        for prefix in (f"{dataset_label}_{model_label}_", f"{model_label}_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        target = out_trace / f"{dataset_label}_{model_label}_{name}"
        written = 0
        with path.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                if method and row.get("method") != method:
                    continue
                if num_spec_tokens is not None and int(row.get("num_spec_tokens", -1)) != num_spec_tokens:
                    continue
                row["dataset_label"] = dataset_label
                row["model_label"] = model_label
                row["source_root"] = str(source)
                dst.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
                rows += 1
        if written == 0:
            target.unlink()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="MODEL:RESULT_ROOT or MODEL:DATASET:RESULT_ROOT. Can be passed multiple times.",
    )
    parser.add_argument("--method")
    parser.add_argument("--num-spec-tokens", type=int)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.output_root.expanduser()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output root already exists; pass --overwrite: {root}")
        for child in root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    (root / "trace").mkdir(parents=True)

    source_rows: list[dict[str, str | int]] = []
    for model_label, dataset_label, source in args.source:
        rows = copy_trace(
            model_label,
            dataset_label,
            source,
            root,
            method=args.method,
            num_spec_tokens=args.num_spec_tokens,
        )
        source_rows.append({
            "dataset_label": dataset_label,
            "model_label": model_label,
            "source_root": str(source),
            "trace_rows": rows,
        })

    (root / "combined_sources.json").write_text(
        json.dumps(source_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "commands.sh").write_text("", encoding="utf-8")

    if args.analyze:
        analyze(root)

    print(f"[INFO] Combined output root: {root}")
    for row in source_rows:
        print(
            f"[INFO] {row['dataset_label']}/{row['model_label']}: {row['trace_rows']} trace rows "
            f"from {row['source_root']}"
        )


if __name__ == "__main__":
    main()
