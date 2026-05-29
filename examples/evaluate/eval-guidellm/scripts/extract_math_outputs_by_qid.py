#!/usr/bin/env python3
"""Extract per-question math outputs from GuideLLM result files.

GuideLLM's raw JSON is nested under benchmarks[*].requests.successful. This
helper maps each request back to the math_reasoning dataset question_id and
writes compact Markdown/JSON comparisons between an EAGLE3 run and a CV run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
MATRIX_SCRIPT = SCRIPT_DIR / "run_speclink_cv_guidellm_matrix.py"


def _load_matrix_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("speclink_cv_matrix", MATRIX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helper module: {MATRIX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_matrix_helpers()


@dataclass(frozen=True)
class Case:
    label: str
    eagle3_dir: Path
    cv_dir: Path


def result_path(run_dir: Path) -> Path:
    if run_dir.name == "guidellm_results.json":
        return run_dir
    return run_dir / "guidellm_results.json"


def parse_case(raw: str) -> Case:
    parts = raw.split("::")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--case must have the form label::eagle3_dir::cv_dir"
        )
    label, eagle3_dir, cv_dir = parts
    return Case(label.strip(), Path(eagle3_dir).resolve(), Path(cv_dir).resolve())


def load_requests_by_qid(
    run_dir: Path,
    dataset_by_prompt: dict[str, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    requests = M.load_successful_requests(result_path(run_dir))
    out: dict[int, dict[str, Any]] = {}
    for request in requests:
        prompt = M.request_prompt_text(request)
        item = dataset_by_prompt.get(prompt)
        if not item:
            continue
        qid = int(item["question_id"])
        out[qid] = request
    return out


def request_summary(
    request: dict[str, Any] | None,
    reference_answer: str,
) -> dict[str, Any]:
    if request is None:
        return {
            "present": False,
            "predicted_answer": "",
            "correct": False,
            "output_tokens": "",
            "prompt_tokens": "",
            "output": "",
        }
    predicted = M.extract_predicted_answer(str(request.get("output", "")))
    return {
        "present": True,
        "predicted_answer": predicted,
        "correct": bool(predicted and predicted == reference_answer),
        "output_tokens": request.get("output_tokens", ""),
        "prompt_tokens": request.get("prompt_tokens", ""),
        "request_id": request.get("request_id", ""),
        "response_id": request.get("response_id", ""),
        "output": str(request.get("output", "")),
    }


def include_row(mode: str, eagle3: dict[str, Any], cv: dict[str, Any]) -> bool:
    if mode == "all":
        return True
    if mode == "cv-wrong":
        return bool(cv.get("present")) and not bool(cv.get("correct"))
    if mode == "cv-drop":
        return (
            bool(cv.get("present"))
            and not bool(cv.get("correct"))
            and bool(eagle3.get("correct"))
        )
    if mode == "disagree":
        return eagle3.get("predicted_answer") != cv.get("predicted_answer")
    raise ValueError(f"unknown mode: {mode}")


def build_case_rows(
    case: Case,
    dataset_by_prompt: dict[str, dict[str, Any]],
    qids: set[int] | None,
    mode: str,
) -> dict[str, Any]:
    eagle3_by_qid = load_requests_by_qid(case.eagle3_dir, dataset_by_prompt)
    cv_by_qid = load_requests_by_qid(case.cv_dir, dataset_by_prompt)
    dataset_by_qid = {
        int(item["question_id"]): item for item in dataset_by_prompt.values()
    }
    all_qids = sorted(set(eagle3_by_qid) | set(cv_by_qid))
    if qids is not None:
        all_qids = [qid for qid in all_qids if qid in qids]

    rows: list[dict[str, Any]] = []
    for qid in all_qids:
        item = dataset_by_qid.get(qid, {})
        reference_answer = M.extract_reference_answer(item) if item else ""
        eagle3 = request_summary(eagle3_by_qid.get(qid), reference_answer)
        cv = request_summary(cv_by_qid.get(qid), reference_answer)
        if not include_row(mode, eagle3, cv):
            continue
        rows.append(
            {
                "question_id": qid,
                "prompt": str(item.get("prompt", "")),
                "reference": item.get("reference", []),
                "reference_answer": reference_answer,
                "eagle3": eagle3,
                "cv": cv,
            }
        )

    return {
        "label": case.label,
        "eagle3_dir": str(case.eagle3_dir),
        "cv_dir": str(case.cv_dir),
        "mode": mode,
        "total_eagle3_requests": len(eagle3_by_qid),
        "total_cv_requests": len(cv_by_qid),
        "selected_rows": len(rows),
        "rows": rows,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Math Output Comparison")
    lines.append("")
    lines.append(f"- dataset: `{payload['dataset']}`")
    lines.append(f"- mode: `{payload['mode']}`")
    lines.append("")
    for case in payload["cases"]:
        lines.append(f"## {case['label']}")
        lines.append("")
        lines.append(f"- EAGLE3 dir: `{case['eagle3_dir']}`")
        lines.append(f"- CV dir: `{case['cv_dir']}`")
        lines.append(
            f"- selected rows: {case['selected_rows']} "
            f"(EAGLE3 requests={case['total_eagle3_requests']}, "
            f"CV requests={case['total_cv_requests']})"
        )
        lines.append("")
        for row in case["rows"]:
            eagle3 = row["eagle3"]
            cv = row["cv"]
            lines.append(f"### qid={row['question_id']}")
            lines.append("")
            lines.append(f"- reference answer: `{row['reference_answer']}`")
            lines.append(
                "- EAGLE3: "
                f"pred=`{eagle3['predicted_answer']}`, "
                f"correct={eagle3['correct']}, "
                f"output_tokens={eagle3['output_tokens']}"
            )
            lines.append(
                "- CV: "
                f"pred=`{cv['predicted_answer']}`, "
                f"correct={cv['correct']}, "
                f"output_tokens={cv['output_tokens']}"
            )
            lines.append("")
            lines.append("Prompt:")
            lines.append("")
            lines.append("```text")
            lines.append(row["prompt"])
            lines.append("```")
            lines.append("")
            lines.append("Reference:")
            lines.append("")
            lines.append("```text")
            reference = row["reference"]
            if isinstance(reference, list):
                lines.append("\n\n".join(str(item) for item in reference))
            else:
                lines.append(str(reference))
            lines.append("```")
            lines.append("")
            lines.append("EAGLE3 Output:")
            lines.append("")
            lines.append("```text")
            lines.append(eagle3["output"])
            lines.append("```")
            lines.append("")
            lines.append("CV Output:")
            lines.append("")
            lines.append("```text")
            lines.append(cv["output"])
            lines.append("```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_qids(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    out: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.add(int(item))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=EVAL_DIR / "data" / "math_reasoning.jsonl",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=[],
        help="Repeatable. Format: label::eagle3_run_dir::cv_run_dir",
    )
    parser.add_argument("--label", default="comparison")
    parser.add_argument("--eagle3-dir", type=Path)
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=["cv-wrong", "cv-drop", "disagree", "all"],
        default="cv-wrong",
        help=(
            "Rows to export. cv-wrong exports all CV wrong answers; cv-drop "
            "exports only rows where EAGLE3 is correct and CV is wrong."
        ),
    )
    parser.add_argument(
        "--qids",
        default="",
        help="Optional comma-separated question_id filter, for example 408,402.",
    )
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    cases = list(args.case)
    if args.eagle3_dir or args.cv_dir:
        if not args.eagle3_dir or not args.cv_dir:
            parser.error("--eagle3-dir and --cv-dir must be provided together")
        cases.append(
            Case(
                args.label,
                args.eagle3_dir.resolve(),
                args.cv_dir.resolve(),
            )
        )
    if not cases:
        parser.error("provide at least one --case or --eagle3-dir/--cv-dir pair")

    dataset_path = args.dataset.resolve()
    dataset_by_prompt = M.load_dataset_by_prompt(dataset_path)
    qids = parse_qids(args.qids)
    payload = {
        "dataset": str(dataset_path),
        "mode": args.mode,
        "cases": [
            build_case_rows(case, dataset_by_prompt, qids, args.mode)
            for case in cases
        ],
    }

    out_md = args.out_md.resolve()
    write_markdown(payload, out_md)
    out_json = args.out_json.resolve() if args.out_json else out_md.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] wrote markdown: {out_md}")
    print(f"[INFO] wrote json: {out_json}")
    for case in payload["cases"]:
        print(
            f"[INFO] {case['label']}: selected_rows={case['selected_rows']} "
            f"cv_dir={case['cv_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
