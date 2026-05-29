#!/usr/bin/env python3
"""Build a lightweight verifier-cost lookup from trace metadata.

This is a trace-derived proxy for the current TODO milestone. It is not a
hardware profiler and must not be used to claim end-to-end speedup.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from tools.speclink_cv.core import load_trace_steps, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", default="math,mtbench")
    args = parser.parse_args()
    workloads = {item.strip() for item in args.workloads.split(",") if item.strip()}
    steps = load_trace_steps(args.trace_root, workloads)
    grouped = defaultdict(list)
    for step in steps:
        for chunk_len in [1, 2, 4, 6, 8, step.k]:
            if 1 <= chunk_len <= step.k:
                grouped[(step.workload, step.model_label, step.method, step.k, chunk_len)].append(step)
    rows = []
    for key, values in sorted(grouped.items()):
        workload, model_label, method, k, chunk_len = key
        # Synthetic proxy: more context and more tokens cost more, but there is
        # no measured GPU timing here.
        predicted_ms = 0.25 + 0.04 * chunk_len
        rows.append(
            {
                "workload": workload,
                "model_label": model_label,
                "method": method,
                "num_spec_tokens": k,
                "chunk_len": chunk_len,
                "samples": len(values),
                "predicted_verify_ms_proxy": predicted_ms,
                "source": "trace_proxy_not_hardware_profile",
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "verify_cost_lookup.csv", rows)
    write_json(
        args.output_dir / "verify_cost_lookup.json",
        {"source": "trace_proxy_not_hardware_profile", "rows": rows},
    )
    (args.output_dir / "verify_cost_report.md").write_text(
        "# Verify Cost Proxy\n\n"
        "This file is a trace-derived placeholder lookup. It is useful for unit "
        "testing roofline-aware packing code paths, but it is not an Nsight or "
        "vLLM timing profile and should not be reported as hardware speedup.\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote {args.output_dir / 'verify_cost_lookup.csv'}")


if __name__ == "__main__":
    main()
