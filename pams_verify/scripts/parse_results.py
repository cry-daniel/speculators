#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pams.configs import EXPERIMENTS, REPORTS, ensure_all_experiments_scaffolded, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EXPERIMENTS)
    args = parser.parse_args()
    ensure_all_experiments_scaffolded()
    index = []
    failures = []
    for meta_path in sorted(args.root.glob("*/metadata.json")):
        exp_name = meta_path.parent.name
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            meta = {"run_status": "metadata_parse_error", "error": str(exc)}
        status = str(meta.get("run_status", "unknown"))
        index.append({"experiment": exp_name, "status": status, "metadata": str(meta_path)})
        if status not in {
            "completed",
            "completed_synthetic_fallback",
            "completed_offline_simulation",
            "completed_reference_gpu",
            "completed_cpu_reference_scaled_down",
            "completed_offline_synthetic_audit",
            "completed_no_oom_observed",
            "completed_live_smoke_partial",
        }:
            failures.append({"experiment": exp_name, "status": status, "metadata": str(meta_path)})
    write_json(args.root / "parsed_results_index.json", {"experiments": index, "failures": failures})
    REPORTS.mkdir(parents=True, exist_ok=True)
    failure_lines = ["# Failure Log", ""]
    if not failures:
        failure_lines.append("No failures recorded.")
    else:
        for item in failures:
            failure_lines.append(f"- `{item['experiment']}`: `{item['status']}`")
    (REPORTS / "failure_log.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    decision_lines = [
        "# GO / NO-GO",
        "",
        "Decision: NO-GO",
        "",
        "Reason: no live patched-vLLM PAMS sparse verifier end-to-end result is present in the current evidence.",
    ]
    (ROOT / "GO_NO_GO.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    (REPORTS / "go_no_go.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    print(json.dumps({"experiments": len(index), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
