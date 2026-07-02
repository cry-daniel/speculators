#!/usr/bin/env python3
"""Compare dense and SR24 lm-eval sample outputs.

This is an offline diagnostic. It does not rerun vLLM or lm-eval; it reads two
`samples_*.jsonl` files and reports where the generated responses first diverge.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_tokenizer(model: str | None) -> Any | None:
    if not model:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            key = sample_key(sample)
            samples[key] = sample
    return samples


def sample_key(sample: dict[str, Any]) -> str:
    for key in ("doc_id", "doc_hash", "prompt_hash"):
        value = sample.get(key)
        if value is not None:
            return f"{key}:{value}"
    return json.dumps(sample.get("doc", {}), sort_keys=True, ensure_ascii=False)


def response_text(sample: dict[str, Any]) -> str:
    resps = sample.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0])
        if isinstance(first, str):
            return first
    filtered = sample.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    return ""


def filtered_text(sample: dict[str, Any]) -> str:
    filtered = sample.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    return response_text(sample)


def correct(sample: dict[str, Any]) -> bool | None:
    value = sample.get("exact_match")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return None


def first_char_diff(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit if len(left) != len(right) else -1


def token_diff(
    tokenizer: Any | None,
    left: str,
    right: str,
) -> tuple[int | None, str, str]:
    if tokenizer is None:
        return None, "", ""
    left_ids = tokenizer.encode(left, add_special_tokens=False)
    right_ids = tokenizer.encode(right, add_special_tokens=False)
    limit = min(len(left_ids), len(right_ids))
    diff_idx = -1
    for idx in range(limit):
        if left_ids[idx] != right_ids[idx]:
            diff_idx = idx
            break
    if diff_idx < 0:
        diff_idx = limit if len(left_ids) != len(right_ids) else -1
    if diff_idx < 0:
        return -1, "", ""
    left_piece = tokenizer.decode(left_ids[max(0, diff_idx - 4):diff_idx + 5])
    right_piece = tokenizer.decode(right_ids[max(0, diff_idx - 4):diff_idx + 5])
    return diff_idx, left_piece, right_piece


def truncate(text: str, limit: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def analyze(args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(args.tokenizer)
    dense = load_samples(args.dense_samples)
    experiment = load_samples(args.experiment_samples)
    trace_by_doc, trace_mapping = load_trace_by_doc(
        args.confidence_trace,
        experiment,
        tokenizer,
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(set(dense) & set(experiment)):
        dense_sample = dense[key]
        exp_sample = experiment[key]
        doc_id = dense_sample.get("doc_id")
        dense_resp = response_text(dense_sample)
        exp_resp = response_text(exp_sample)
        dense_filtered = filtered_text(dense_sample)
        exp_filtered = filtered_text(exp_sample)
        char_idx = first_char_diff(dense_resp, exp_resp)
        token_idx, dense_token_context, exp_token_context = token_diff(
            tokenizer,
            dense_resp,
            exp_resp,
        )
        dense_correct = correct(dense_sample)
        exp_correct = correct(exp_sample)
        trace_summary = trace_by_doc.get(doc_id, {})
        trace_token_idx = map_sample_token_to_trace_token(
            token_idx,
            trace_summary.get("mapping_sample_offset", ""),
        )
        divergence_step = find_divergence_step(
            trace_summary.get("steps", []),
            trace_token_idx,
        )
        before_divergence = summarize_base_before_step(
            trace_summary.get("steps", []),
            divergence_step.get("step_id"),
        )
        rows.append(
            {
                "sample_key": key,
                "doc_id": doc_id,
                "target": dense_sample.get("target"),
                "dense_correct": dense_correct,
                "experiment_correct": exp_correct,
                "regression": bool(dense_correct and exp_correct is False),
                "improvement": bool(dense_correct is False and exp_correct),
                "same_correctness": dense_correct == exp_correct,
                "first_char_diff": char_idx,
                "first_token_diff": token_idx,
                "first_trace_token_diff": trace_token_idx,
                "dense_response_chars": len(dense_resp),
                "experiment_response_chars": len(exp_resp),
                "dense_filtered": dense_filtered,
                "experiment_filtered": exp_filtered,
                "dense_response": dense_resp,
                "experiment_response": exp_resp,
                "dense_token_context": dense_token_context,
                "experiment_token_context": exp_token_context,
                "dense_sample_path": str(args.dense_samples),
                "experiment_sample_path": str(args.experiment_samples),
                "request_id": trace_summary.get("request_id", ""),
                "trace_mapping_method": trace_summary.get(
                    "mapping_method", ""
                ),
                "trace_mapping_match_tokens": trace_summary.get(
                    "mapping_match_tokens", ""
                ),
                "trace_mapping_second_match_tokens": trace_summary.get(
                    "mapping_second_match_tokens", ""
                ),
                "trace_mapping_sample_offset": trace_summary.get(
                    "mapping_sample_offset", ""
                ),
                "trace_steps": trace_summary.get("step_count", ""),
                "trace_has_sr24_mask": trace_summary.get(
                    "trace_has_sr24_mask", ""
                ),
                "trace_reached_base_only_tokens":
                trace_summary.get("reached_base_only_tokens", ""),
                "trace_accepted_base_only_tokens":
                trace_summary.get("accepted_base_only_tokens", ""),
                "trace_rejected_base_only_tokens":
                trace_summary.get("rejected_base_only_tokens", ""),
                "divergence_step_id": divergence_step.get("step_id", ""),
                "divergence_step_generated_len":
                divergence_step.get("generated_len", ""),
                "divergence_step_output_end":
                divergence_step.get("output_end", ""),
                "divergence_step_trace_output_start":
                divergence_step.get("trace_output_start", ""),
                "divergence_step_trace_output_end":
                divergence_step.get("trace_output_end", ""),
                "divergence_step_accepted_draft_tokens":
                divergence_step.get("accepted_draft_tokens", ""),
                "divergence_step_first_reject_position":
                divergence_step.get("first_reject_position", ""),
                "divergence_step_output_token_ids":
                " ".join(
                    str(token_id)
                    for token_id in divergence_step.get(
                        "output_token_ids_valid", []
                    )
                ),
                "divergence_step_bonus_token_id":
                divergence_step.get("bonus_token_id", ""),
                "divergence_step_mask_state":
                divergence_step.get("mask_state", ""),
                "divergence_step_mask_pattern":
                divergence_step.get("mask_pattern", ""),
                "divergence_step_raw_mask_pattern":
                divergence_step.get("raw_mask_pattern", ""),
                "divergence_step_reached_base_only":
                divergence_step.get("reached_base_only", ""),
                "divergence_step_accepted_base_only":
                divergence_step.get("accepted_base_only", ""),
                "divergence_step_rejected_base_only":
                divergence_step.get("rejected_base_only", ""),
                "before_divergence_reached_base_only":
                before_divergence.get("reached_base_only", ""),
                "before_divergence_accepted_base_only":
                before_divergence.get("accepted_base_only", ""),
                "before_divergence_rejected_base_only":
                before_divergence.get("rejected_base_only", ""),
            }
        )
    setattr(args, "_trace_mapping", trace_mapping)
    return rows


def load_trace_by_doc(
    trace_path: Path | None,
    experiment_samples: dict[str, dict[str, Any]],
    tokenizer: Any | None,
) -> tuple[dict[Any, dict[str, Any]], dict[str, Any]]:
    if trace_path is None or not trace_path.exists():
        return {}, {"method": "none", "mapped_requests": 0, "note": "no trace"}

    doc_ids = sorted(
        {
            sample.get("doc_id")
            for sample in experiment_samples.values()
            if sample.get("doc_id") is not None
        }
    )
    request_order: list[str] = []
    request_seen: set[str] = set()
    raw_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            request_id = str(row.get("request_id") or "")
            if not request_id:
                continue
            if request_id not in request_seen:
                request_seen.add(request_id)
                request_order.append(request_id)
            raw_rows[request_id].append(row)

    request_to_doc, mapping_info = map_requests_to_docs(
        raw_rows,
        request_order,
        doc_ids,
        experiment_samples,
        tokenizer,
    )

    out: dict[Any, dict[str, Any]] = {}
    for request_id, rows in raw_rows.items():
        mapping = request_to_doc.get(request_id, {})
        doc_id = mapping.get("doc_id")
        if doc_id is None:
            continue
        steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
        reached_base = 0
        accepted_base = 0
        rejected_base = 0
        for row in rows:
            step_id = int(row.get("step_id") or -1)
            steps[step_id].append(row)
            is_base = sr24_effective_residual(row) == 0
            is_reached = row.get("reached") == 1
            accepted_local = row.get("accepted_local")
            if is_base and is_reached:
                reached_base += 1
                if accepted_local == 1:
                    accepted_base += 1
                if accepted_local == 0:
                    rejected_base += 1
        step_summaries = [
            summarize_trace_step(step_rows)
            for _, step_rows in sorted(steps.items())
        ]
        attach_trace_output_spans(step_summaries)
        has_sr24_mask = any(
            sr24_effective_residual(row) is not None
            for step_rows in steps.values()
            for row in step_rows
        )
        out[doc_id] = {
            "request_id": request_id,
            "mapping_method": mapping.get("method", ""),
            "mapping_match_tokens": mapping.get("match_tokens", ""),
            "mapping_second_match_tokens": mapping.get(
                "second_match_tokens", ""
            ),
            "mapping_sample_offset": mapping.get("sample_offset", ""),
            "step_count": len(step_summaries),
            "trace_has_sr24_mask": int(has_sr24_mask),
            "reached_base_only_tokens": reached_base,
            "accepted_base_only_tokens": accepted_base,
            "rejected_base_only_tokens": rejected_base,
            "steps": step_summaries,
        }
    return out, mapping_info


def map_requests_to_docs(
    raw_rows: dict[str, list[dict[str, Any]]],
    request_order: list[str],
    doc_ids: list[Any],
    experiment_samples: dict[str, dict[str, Any]],
    tokenizer: Any | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if tokenizer is not None:
        mapped, info = map_requests_by_output_tokens(
            raw_rows,
            experiment_samples,
            tokenizer,
        )
        if info.get("mapped_requests") == len(raw_rows) and not info.get(
            "ambiguous_requests"
        ):
            return mapped, info

    # Fallback kept only for older traces. This is weaker under serving because
    # vLLM request first-seen order does not have to equal lm-eval doc order.
    mapped = {
        request_id: {
            "doc_id": doc_id,
            "method": "request_order_fallback",
            "match_tokens": "",
            "second_match_tokens": "",
            "sample_offset": "",
        }
        for request_id, doc_id in zip(request_order, doc_ids)
    }
    return mapped, {
        "method": "request_order_fallback",
        "mapped_requests": len(mapped),
        "ambiguous_requests": 0,
        "note": (
            "weak mapping: request first-seen order zipped with sorted doc_id"
        ),
    }


def map_requests_by_output_tokens(
    raw_rows: dict[str, list[dict[str, Any]]],
    experiment_samples: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sample_ids: list[tuple[Any, list[int]]] = []
    for sample in experiment_samples.values():
        doc_id = sample.get("doc_id")
        if doc_id is None:
            continue
        sample_ids.append(
            (
                doc_id,
                tokenizer.encode(response_text(sample), add_special_tokens=False),
            )
        )

    mapped: dict[str, dict[str, Any]] = {}
    ambiguous = 0
    low_match = 0
    for request_id, rows in raw_rows.items():
        output_ids = trace_output_token_ids(rows)
        scored: list[tuple[int, int, Any]] = []
        for doc_id, response_ids in sample_ids:
            match_tokens, offset = longest_prefix_match_at_any_offset(
                output_ids,
                response_ids,
            )
            scored.append((match_tokens, offset, doc_id))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        best = scored[0] if scored else (0, -1, None)
        second = scored[1] if len(scored) > 1 else (0, -1, None)
        if best[0] == second[0]:
            ambiguous += 1
        if best[0] <= 0:
            low_match += 1
        mapped[request_id] = {
            "doc_id": best[2],
            "method": "output_token_subsequence",
            "match_tokens": best[0],
            "second_match_tokens": second[0],
            "sample_offset": best[1],
        }
    return mapped, {
        "method": "output_token_subsequence",
        "mapped_requests": len(mapped),
        "ambiguous_requests": ambiguous,
        "low_match_requests": low_match,
    }


def trace_output_token_ids(rows: list[dict[str, Any]]) -> list[int]:
    by_step: dict[int, list[int]] = {}
    for row in rows:
        try:
            step_id = int(row.get("step_id") or -1)
        except (TypeError, ValueError):
            continue
        if step_id in by_step:
            continue
        output_ids = row.get("output_token_ids_valid")
        if not isinstance(output_ids, list):
            continue
        by_step[step_id] = [
            int(token_id)
            for token_id in output_ids
            if isinstance(token_id, int)
        ]
    return [
        token_id
        for _, step_ids in sorted(by_step.items())
        for token_id in step_ids
    ]


def longest_prefix_match_at_any_offset(
    needle: list[int],
    haystack: list[int],
) -> tuple[int, int]:
    best = 0
    best_offset = -1
    if not needle or not haystack:
        return best, best_offset
    for offset in range(len(haystack)):
        matched = 0
        while (
            matched < len(needle)
            and offset + matched < len(haystack)
            and needle[matched] == haystack[offset + matched]
        ):
            matched += 1
        if matched > best:
            best = matched
            best_offset = offset
    return best, best_offset


def sr24_effective_residual(row: dict[str, Any]) -> int | None:
    """Return the residual path actually used for a trace row.

    `sr24_uses_residual` is the requested mask. Bucketed paths can request more
    rows than the fixed correction bucket can actually correct, so prefer
    `sr24_effective_residual` when the trace recorded it.
    """
    for key in ("sr24_effective_residual", "sr24_uses_residual"):
        value = row.get(key)
        if value in (0, 1, False, True):
            return int(bool(value))
        if isinstance(value, str) and value in {"0", "1"}:
            return int(value)
    return None


def summarize_trace_step(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row.get("draft_position") or 0))
    first = rows[0] if rows else {}
    accepted = max(int(row.get("num_accepted_in_step") or 0) for row in rows)
    generated_len = int(first.get("generated_len_so_far") or 0)
    has_sr24_mask = any(sr24_effective_residual(row) is not None for row in rows)
    reached_base = sum(
        1
        for row in rows
        if has_sr24_mask
        and row.get("reached") == 1
        and sr24_effective_residual(row) == 0
    )
    accepted_base = sum(
        1
        for row in rows
        if has_sr24_mask
        and row.get("accepted_local") == 1
        and sr24_effective_residual(row) == 0
    )
    rejected_base = sum(
        1
        for row in rows
        if has_sr24_mask
        and row.get("accepted_local") == 0
        and row.get("reached") == 1
        and sr24_effective_residual(row) == 0
    )
    mask_pattern = (
        "".join(
            "1" if sr24_effective_residual(row) == 1 else "0"
            for row in rows
        )
        if has_sr24_mask
        else "missing"
    )
    raw_mask_pattern = (
        "".join(
            "1" if row.get("sr24_uses_residual") == 1 else "0"
            for row in rows
        )
        if has_sr24_mask
        else "missing"
    )
    # A speculative verifier step emits accepted draft tokens plus one target
    # bonus/replacement token unless generation stops at EOS. This is enough to
    # locate the step containing a response-token divergence.
    output_end = generated_len + accepted + 1
    output_token_ids = first.get("output_token_ids_valid", [])
    if not isinstance(output_token_ids, list):
        output_token_ids = []
    return {
        "step_id": int(first.get("step_id") or -1),
        "generated_len": generated_len,
        "output_end": output_end,
        "accepted_draft_tokens": accepted,
        "first_reject_position": first.get("first_reject_position"),
        "output_token_ids_valid": output_token_ids,
        "bonus_token_id": first.get("bonus_token_id"),
        "mask_state": first.get("sr24_mask_state"),
        "mask_pattern": mask_pattern,
        "raw_mask_pattern": raw_mask_pattern,
        "reached_base_only": reached_base,
        "accepted_base_only": accepted_base,
        "rejected_base_only": rejected_base,
    }


def find_divergence_step(
    steps: list[dict[str, Any]],
    token_idx: int | None,
) -> dict[str, Any]:
    if token_idx is None or token_idx < 0:
        return {}
    for step in steps:
        if "trace_output_start" in step and "trace_output_end" in step:
            start = int(step.get("trace_output_start") or 0)
            end = int(step.get("trace_output_end") or start)
        else:
            start = int(step.get("generated_len") or 0)
            end = int(step.get("output_end") or start)
        if start <= token_idx < end:
            return step
    previous: dict[str, Any] = {}
    for step in steps:
        if "trace_output_start" in step:
            start = int(step.get("trace_output_start") or 0)
        else:
            start = int(step.get("generated_len") or 0)
        if start <= token_idx:
            previous = step
        else:
            break
    return previous


def attach_trace_output_spans(steps: list[dict[str, Any]]) -> None:
    """Attach cumulative trace-output spans to step summaries.

    `generated_len_so_far` can jump or omit tokens depending on where the trace
    starts relative to the lm-eval sample. The trace's own emitted token list is
    the stable coordinate system for mapping a response-token divergence back
    to a speculative verify step.
    """
    cursor = 0
    for step in steps:
        output_token_ids = step.get("output_token_ids_valid", [])
        if not isinstance(output_token_ids, list):
            output_token_ids = []
        step["trace_output_start"] = cursor
        cursor += len(output_token_ids)
        step["trace_output_end"] = cursor


def map_sample_token_to_trace_token(
    sample_token_idx: int | None,
    sample_offset: Any,
) -> int | None:
    """Convert a sample-response token index into the trace-output index.

    Confidence traces can start after an already-emitted token because lm-eval
    samples include the full response while the trace-output subsequence is
    reconstructed from speculative decode steps. The request-to-doc mapper
    records that offset; use it before locating the divergence step.
    """
    if sample_token_idx is None or sample_token_idx < 0:
        return sample_token_idx
    try:
        offset = int(sample_offset)
    except (TypeError, ValueError):
        offset = 0
    return sample_token_idx - max(0, offset)


def summarize_base_before_step(
    steps: list[dict[str, Any]],
    step_id: Any,
) -> dict[str, int] | dict[str, str]:
    try:
        step_id_int = int(step_id)
    except (TypeError, ValueError):
        return {}
    before = [
        step
        for step in steps
        if int(step.get("step_id") or -1) < step_id_int
    ]
    return {
        "reached_base_only": sum(int(step.get("reached_base_only") or 0)
                                 for step in before),
        "accepted_base_only": sum(int(step.get("accepted_base_only") or 0)
                                  for step in before),
        "rejected_base_only": sum(int(step.get("rejected_base_only") or 0)
                                  for step in before),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    paired = len(rows)
    regressions = [row for row in rows if row["regression"]]
    improvements = [row for row in rows if row["improvement"]]
    response_different = [
        row for row in rows if int(row.get("first_char_diff", -1)) >= 0
    ]
    same_correctness_different = [
        row for row in response_different
        if row["same_correctness"] and not row["regression"] and not row["improvement"]
    ]
    dense_correct = sum(1 for row in rows if row["dense_correct"])
    exp_correct = sum(1 for row in rows if row["experiment_correct"])
    same = sum(1 for row in rows if row["same_correctness"])
    lines = [
        "# SR24 Sample Divergence",
        "",
        f"- dense samples: `{args.dense_samples.resolve()}`",
        f"- experiment samples: `{args.experiment_samples.resolve()}`",
        "- trace mapping: `{method}`; mapped requests `{mapped}`; "
        "ambiguous `{ambiguous}`; low-match `{low}`".format(
            method=getattr(args, "_trace_mapping", {}).get("method", ""),
            mapped=getattr(args, "_trace_mapping", {}).get(
                "mapped_requests", ""
            ),
            ambiguous=getattr(args, "_trace_mapping", {}).get(
                "ambiguous_requests", ""
            ),
            low=getattr(args, "_trace_mapping", {}).get(
                "low_match_requests", ""
            ),
        ),
        f"- paired samples: `{paired}`",
        f"- dense correct: `{dense_correct}`",
        f"- experiment correct: `{exp_correct}`",
        f"- same correctness: `{same}`",
        f"- response text differs: `{len(response_different)}`",
        f"- same-correctness text differs: `{len(same_correctness_different)}`",
        f"- dense correct -> experiment wrong: `{len(regressions)}`",
        f"- dense wrong -> experiment correct: `{len(improvements)}`",
        "",
        "## Output Divergence",
        "",
        "| doc_id | target | dense correct | experiment correct | first char diff | first sample token diff | first trace token diff | dense chars | experiment chars | dense token context | experiment token context |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in response_different[: args.max_examples]:
        lines.append(
            "| {doc_id} | `{target}` | {dense_correct} | {exp_correct} | {char} | {tok} | {trace_tok} | {dense_len} | {exp_len} | {dense_ctx} | {exp_ctx} |".format(
                doc_id=row.get("doc_id"),
                target=truncate(str(row.get("target")), 80),
                dense_correct=row.get("dense_correct"),
                exp_correct=row.get("experiment_correct"),
                char=row.get("first_char_diff"),
                tok=row.get("first_token_diff"),
                trace_tok=row.get("first_trace_token_diff"),
                dense_len=row.get("dense_response_chars"),
                exp_len=row.get("experiment_response_chars"),
                dense_ctx=truncate(str(row.get("dense_token_context")), 120),
                exp_ctx=truncate(str(row.get("experiment_token_context")), 120),
            )
        )
    lines.extend(
        [
            "",
        "## Regressions",
        "",
        "| doc_id | target | first sample token diff | first trace token diff | trace span | step gen:end | output ids | bonus | effective mask | raw mask | accepted | reached base | accepted base | rejected base | accepted base before | rejected base before | dense filtered | experiment filtered | dense token context | experiment token context |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in regressions[: args.max_examples]:
        lines.append(
            "| {doc_id} | `{target}` | {tok} | {trace_tok} | {trace_start}:{trace_end} | {gen}:{end} | {out_ids} | {bonus} | {mask} | {raw_mask} | {accepted} | {reached_base} | {accepted_base} | {rejected_base} | {accepted_base_before} | {rejected_base_before} | {dense} | {exp} | {dense_ctx} | {exp_ctx} |".format(
                doc_id=row.get("doc_id"),
                target=truncate(str(row.get("target")), 80),
                tok=row.get("first_token_diff"),
                trace_tok=row.get("first_trace_token_diff"),
                trace_start=row.get("divergence_step_trace_output_start", ""),
                trace_end=row.get("divergence_step_trace_output_end", ""),
                gen=row.get("divergence_step_generated_len", ""),
                end=row.get("divergence_step_output_end", ""),
                out_ids=truncate(
                    str(row.get("divergence_step_output_token_ids", "")),
                    80,
                ),
                bonus=row.get("divergence_step_bonus_token_id", ""),
                mask=row.get("divergence_step_mask_pattern", ""),
                raw_mask=row.get("divergence_step_raw_mask_pattern", ""),
                accepted=row.get("divergence_step_accepted_draft_tokens", ""),
                reached_base=row.get("divergence_step_reached_base_only", ""),
                accepted_base=row.get("divergence_step_accepted_base_only", ""),
                rejected_base=row.get("divergence_step_rejected_base_only", ""),
                accepted_base_before=row.get(
                    "before_divergence_accepted_base_only", ""
                ),
                rejected_base_before=row.get(
                    "before_divergence_rejected_base_only", ""
                ),
                dense=truncate(str(row.get("dense_filtered")), 160),
                exp=truncate(str(row.get("experiment_filtered")), 160),
                dense_ctx=truncate(str(row.get("dense_token_context")), 120),
                exp_ctx=truncate(str(row.get("experiment_token_context")), 120),
            )
        )
    lines.extend(
        [
            "",
            "## Improvements",
            "",
            "| doc_id | target | first char diff | first token diff | dense filtered | experiment filtered |",
            "| ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in improvements[: args.max_examples]:
        lines.append(
            "| {doc_id} | `{target}` | {char} | {tok} | {dense} | {exp} |".format(
                doc_id=row.get("doc_id"),
                target=truncate(str(row.get("target")), 80),
                char=row.get("first_char_diff"),
                tok=row.get("first_token_diff"),
                dense=truncate(str(row.get("dense_filtered")), 160),
                exp=truncate(str(row.get("experiment_filtered")), 160),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-samples", type=Path, required=True)
    parser.add_argument("--experiment-samples", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument(
        "--confidence-trace",
        type=Path,
        help=(
            "Optional speclink_confidence_trace.jsonl. When provided, request "
            "first-seen order is mapped to sorted sample doc_id values."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = analyze(args)
    write_csv(args.output_root / "sample_divergence.csv", rows)
    (args.output_root / "sample_divergence.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(args.output_root / "report.md", rows, args)
    print(args.output_root.resolve())


if __name__ == "__main__":
    main()
