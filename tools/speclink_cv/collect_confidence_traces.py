#!/usr/bin/env python3
"""Index existing confidence traces for a SpecLink-CV run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.speclink_cv.core import find_trace_files, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", default="math,mtbench")
    args = parser.parse_args()
    workloads = {item.strip() for item in args.workloads.split(",") if item.strip()}
    rows = []
    for path in find_trace_files(args.trace_root):
        count = 0
        first: dict | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("dataset_label")) not in workloads:
                continue
            count += 1
            if first is None:
                first = item
        if count == 0:
            continue
        assert first is not None
        rows.append(
            {
                "trace_file": str(path),
                "records": count,
                "workload": first.get("dataset_label"),
                "model_label": first.get("model_label"),
                "method": first.get("method"),
                "num_spec_tokens": first.get("num_spec_tokens"),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trace_manifest.csv", rows)
    write_json(
        args.output_dir / "trace_manifest.json",
        {"trace_root": str(args.trace_root), "workloads": sorted(workloads), "files": rows},
    )
    (args.output_dir / "trace_collection_report.md").write_text(
        "# Confidence Trace Collection\n\n"
        f"- trace root: `{args.trace_root}`\n"
        f"- workloads: {', '.join(sorted(workloads))}\n"
        f"- files: {len(rows)}\n"
        f"- records: {sum(int(row['records']) for row in rows)}\n",
        encoding="utf-8",
    )
    print(f"[INFO] Indexed {len(rows)} trace files")


if __name__ == "__main__":
    main()
