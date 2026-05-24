#!/usr/bin/env python3
"""Offline masked-logits quality check for sparse verifier layouts.

This is not a vLLM sparse-kernel benchmark. It runs the target model through
Transformers with dense causal attention and with per-query sparse attention
masks, then compares verifier logits on a small set of positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from speculators.speclink.math_eval import build_math_prompt, load_dataset  # noqa: E402
from speculators.speclink.planner import SpeclinkConfig, SpeclinkPlanner  # noqa: E402


EVIDENCE_TYPE = "offline_masked_qwen3_logits_proxy"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "This table is offline/proxy verifier-quality evidence. It uses",
        "proxy sparse candidates and Transformers masked",
        "attention, not a vLLM sparse verifier kernel.",
        "",
    ]
    if not rows:
        lines.append("No sparse quality rows were produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    cols = [
        "context_label",
        "candidate_source",
        "layout",
        "n_samples",
        "n_positions",
        "top1_match_rate",
        "kl_dense_to_sparse_mean",
        "actual_dense_accept_rate",
        "actual_false_accept_rate",
        "actual_false_reject_rate",
        "counterfactual_top2_false_accept_rate",
        "avg_union_blocks",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_accept_probs(value: str, k: int) -> list[float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parts:
        parts = [0.7]
    while len(parts) < k:
        parts.append(parts[-1])
    return [max(0.0, min(1.0, item)) for item in parts[:k]]


def make_proxy_candidates(
    *,
    context_tokens: int,
    block_size: int,
    draft_tokens: list[int],
    topk_per_token: int,
    shared_budget: int,
    private_max: int,
) -> list[list[dict[str, float]]]:
    num_blocks = max(1, (context_tokens + block_size - 1) // block_size)
    keep = max(topk_per_token, shared_budget + private_max + 8)
    keep = max(1, min(num_blocks, keep))
    candidates: list[list[dict[str, float]]] = []
    for pos, token_id in enumerate(draft_tokens):
        center = max(0, num_blocks - 1 - (pos % max(num_blocks, 1)))
        token_factor = 1.0 + ((int(token_id) % 17) / 10000.0)
        scored: list[dict[str, float]] = []
        for block in range(num_blocks):
            distance = abs(block - center)
            recency = block / max(num_blocks - 1, 1)
            score = token_factor * ((1.0 / (1.0 + distance)) + 0.001 * recency)
            scored.append({"block": float(block), "score": float(score)})
        scored.sort(key=lambda item: (-item["score"], item["block"]))
        total = sum(item["score"] for item in scored[:keep]) or 1.0
        candidates.append(
            [
                {"block": int(item["block"]), "score": float(item["score"] / total)}
                for item in scored[:keep]
            ]
        )
    return candidates


def make_hidden_similarity_candidates(
    *,
    hidden_states: torch.Tensor,
    context_tokens: int,
    block_size: int,
    query_positions: list[int],
    topk_per_token: int,
    shared_budget: int,
    private_max: int,
) -> list[list[dict[str, float]]]:
    """Score history blocks by cosine similarity to each verifier query state."""

    num_blocks = max(1, (context_tokens + block_size - 1) // block_size)
    keep = max(topk_per_token, shared_budget + private_max + 8)
    keep = max(1, min(num_blocks, keep))

    history = hidden_states[:context_tokens].float()
    history = torch.nn.functional.normalize(history, dim=-1)
    candidates: list[list[dict[str, float]]] = []

    for query_position in query_positions:
        query = hidden_states[query_position].float()
        query = torch.nn.functional.normalize(query, dim=-1)
        token_scores = torch.matmul(history, query)
        block_scores: list[dict[str, float]] = []
        for block in range(num_blocks):
            start = block * block_size
            end = min((block + 1) * block_size, context_tokens)
            if end <= start:
                continue
            score = float(token_scores[start:end].max().item())
            block_scores.append({"block": block, "score": score})
        if not block_scores:
            block_scores = [{"block": 0, "score": 1.0}]
        min_score = min(item["score"] for item in block_scores)
        shifted = [
            {"block": item["block"], "score": item["score"] - min_score + 1e-6}
            for item in block_scores
        ]
        shifted.sort(key=lambda item: (-item["score"], item["block"]))
        total = sum(item["score"] for item in shifted[:keep]) or 1.0
        candidates.append(
            [
                {"block": int(item["block"]), "score": float(item["score"] / total)}
                for item in shifted[:keep]
            ]
        )
    return candidates


def plan_windows(
    generated_ids: list[int],
    prompt_len: int,
    args: argparse.Namespace,
    hidden_states: torch.Tensor | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Return sparse plan metadata keyed by (layout, generated offset)."""

    out: dict[tuple[str, int], dict[str, Any]] = {}
    trace_rows: list[dict[str, Any]] = []
    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    accept_probs = parse_accept_probs(args.accept_probs, args.num_spec_tokens)
    for start in range(0, len(generated_ids), args.num_spec_tokens):
        draft = generated_ids[start : start + args.num_spec_tokens]
        if not draft:
            continue
        context_start = prompt_len + start
        if args.candidate_source == "hidden_similarity_proxy":
            if hidden_states is None:
                raise ValueError("hidden_similarity_proxy requires hidden states")
            candidates = make_hidden_similarity_candidates(
                hidden_states=hidden_states,
                context_tokens=context_start,
                block_size=args.block_size,
                query_positions=[
                    context_start + pos - 1 for pos in range(len(draft))
                ],
                topk_per_token=args.topk_per_token,
                shared_budget=args.shared_budget,
                private_max=args.private_max,
            )
        elif args.candidate_source == "position_recency_proxy":
            candidates = make_proxy_candidates(
                context_tokens=context_start,
                block_size=args.block_size,
                draft_tokens=draft,
                topk_per_token=args.topk_per_token,
                shared_budget=args.shared_budget,
                private_max=args.private_max,
            )
        else:
            raise ValueError(f"unsupported candidate source: {args.candidate_source}")
        probs = accept_probs[: len(draft)]
        trace_rows.append(
            {
                "request_id": None,
                "sample_id": None,
                "sample_index": None,
                "context_label": args.context_label,
                "step": start // args.num_spec_tokens,
                "prompt_len": prompt_len,
                "generated_len": start,
                "num_spec_tokens": len(draft),
                "block_size": args.block_size,
                "draft_tokens": draft,
                "accepted_prefix_len": None,
                "accept_probs": probs,
                "a_i": probs,
                "candidates": candidates,
                "candidate_source": args.candidate_source,
                "trace_source": EVIDENCE_TYPE,
            }
        )
        for layout in layouts:
            cfg = SpeclinkConfig(
                layout=layout,
                topk_per_token=args.topk_per_token,
                shared_budget=args.shared_budget,
                private_min=args.private_min,
                private_max=args.private_max,
                alpha=args.alpha,
                beta=args.beta,
                lambda_risk=args.lambda_risk,
                risk_threshold=args.risk_threshold,
                fallback_enabled=layout.endswith("fallback"),
            )
            plan = SpeclinkPlanner(cfg).plan(candidates, probs)
            for pos, final_blocks in enumerate(plan.final_blocks_per_token):
                out[(layout, start + pos)] = {
                    "context_start": context_start,
                    "position_in_window": pos,
                    "final_blocks": final_blocks,
                    "union_blocks": len(plan.union_blocks),
                    "mean_blocks_per_token": plan.stats.get("mean_blocks_per_token"),
                    "fallback": pos in set(plan.fallback_tokens),
                }
    return out, trace_rows


def causal_mask(length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    neg = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, length, length), dtype=dtype, device=device)
    future = torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1
    )
    mask.masked_fill_(future.view(1, 1, length, length), neg)
    return mask


def sparse_mask_for_layout(
    *,
    layout: str,
    length: int,
    prompt_len: int,
    generated_len: int,
    plans: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask = causal_mask(length, dtype, device)
    neg = torch.finfo(dtype).min
    for offset in range(generated_len):
        query = prompt_len + offset - 1
        if query < 0:
            continue
        plan = plans.get((layout, offset))
        if plan is None:
            continue
        context_start = int(plan["context_start"])
        allowed = torch.zeros(query + 1, dtype=torch.bool, device=device)
        for block in plan["final_blocks"]:
            start = int(block) * args.block_size
            end = min((int(block) + 1) * args.block_size, context_start, query + 1)
            if end > start:
                allowed[start:end] = True
        if args.allow_local_draft and query >= context_start:
            allowed[context_start : query + 1] = True
        disallowed = ~allowed
        mask[0, 0, query, : query + 1].masked_fill_(disallowed, neg)
    return mask


def prompt_to_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": build_math_prompt(prompt)}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = messages[0]["content"]
    return tokenizer(text, add_special_tokens=False).input_ids


def safe_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001
        return str(token_id)


def evaluate_sample(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    prompt_ids = prompt_to_ids(tokenizer, str(row["prompt"]))
    generated_ids = tokenizer(
        str(row.get("model_output") or ""),
        add_special_tokens=False,
    ).input_ids
    generated_ids = generated_ids[: args.max_positions]
    max_generated_by_length = max(0, args.max_seq_len - len(prompt_ids))
    generated_ids = generated_ids[:max_generated_by_length]
    if not generated_ids:
        return [], [], {"skipped": 1}

    input_ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    generated_len = len(generated_ids)
    query_positions = [prompt_len + offset - 1 for offset in range(generated_len)]
    candidate_ids = torch.tensor(generated_ids, dtype=torch.long, device=device)
    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]

    with torch.inference_mode():
        dense_output = model(
            input_ids,
            output_hidden_states=args.candidate_source == "hidden_similarity_proxy",
        )
        dense_logits = dense_output.logits[0, query_positions, :].float()
        hidden_states = (
            dense_output.hidden_states[-1][0].detach()
            if args.candidate_source == "hidden_similarity_proxy"
            else None
        )
        dense_log_probs = torch.log_softmax(dense_logits, dim=-1)
        dense_probs = dense_log_probs.exp()
        dense_top = torch.topk(dense_logits, k=2, dim=-1).indices
        dense_top1 = dense_top[:, 0]
        dense_top2 = dense_top[:, 1]
        dense_accept_actual = dense_top1.eq(candidate_ids)
    plans, trace_rows = plan_windows(
        generated_ids,
        prompt_len,
        args,
        hidden_states=hidden_states,
    )
    for trace in trace_rows:
        trace["request_id"] = row.get("id")
        trace["sample_id"] = row.get("id")
        trace["sample_index"] = row.get("index")

    rows: list[dict[str, Any]] = []
    for layout in layouts:
        mask = sparse_mask_for_layout(
            layout=layout,
            length=input_ids.shape[1],
            prompt_len=prompt_len,
            generated_len=generated_len,
            plans=plans,
            args=args,
            dtype=dtype,
            device=device,
        )
        with torch.inference_mode():
            sparse_logits = model(input_ids, attention_mask=mask).logits[
                0, query_positions, :
            ].float()
            sparse_log_probs = torch.log_softmax(sparse_logits, dim=-1)
            kl = (dense_probs * (dense_log_probs - sparse_log_probs)).sum(dim=-1)
            sparse_top1 = sparse_logits.argmax(dim=-1)
            sparse_accept_actual = sparse_top1.eq(candidate_ids)
            sparse_accept_top2 = sparse_top1.eq(dense_top2)

        for offset in range(generated_len):
            plan = plans[(layout, offset)]
            dense_accept = bool(dense_accept_actual[offset].item())
            sparse_accept = bool(sparse_accept_actual[offset].item())
            row_out = {
                "context_label": args.context_label,
                "sample_id": row.get("id"),
                "sample_index": row.get("index"),
                "layout": layout,
                "evidence_type": EVIDENCE_TYPE,
                "candidate_source": args.candidate_source,
                "prompt_len": prompt_len,
                "generated_offset": offset,
                "query_position": query_positions[offset],
                "candidate_token_id": int(candidate_ids[offset].item()),
                "candidate_token": safe_token(tokenizer, int(candidate_ids[offset].item())),
                "dense_top1_id": int(dense_top1[offset].item()),
                "dense_top1": safe_token(tokenizer, int(dense_top1[offset].item())),
                "sparse_top1_id": int(sparse_top1[offset].item()),
                "sparse_top1": safe_token(tokenizer, int(sparse_top1[offset].item())),
                "top1_match": bool(dense_top1[offset].eq(sparse_top1[offset]).item()),
                "kl_dense_to_sparse": float(kl[offset].item()),
                "actual_dense_accept": dense_accept,
                "actual_sparse_accept": sparse_accept,
                "actual_false_accept": (not dense_accept) and sparse_accept,
                "actual_false_reject": dense_accept and (not sparse_accept),
                "counterfactual_top2_false_accept": bool(
                    sparse_accept_top2[offset].item()
                ),
                "union_blocks": int(plan["union_blocks"]),
                "mean_blocks_per_token": float(plan["mean_blocks_per_token"]),
                "fallback": bool(plan["fallback"]),
            }
            rows.append(row_out)
    return rows, trace_rows, {"skipped": 0}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.get("context_label") or ""),
                str(row.get("candidate_source") or ""),
                str(row["layout"]),
            ),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (context_label, candidate_source, layout), group in sorted(grouped.items()):
        n = len(group)
        samples = {row["sample_id"] for row in group}
        bool_rate = lambda key: sum(1 for row in group if row[key]) / n if n else None
        out.append(
            {
                "context_label": context_label,
                "layout": layout,
                "evidence_type": EVIDENCE_TYPE,
                "candidate_source": candidate_source,
                "n_samples": len(samples),
                "n_positions": n,
                "top1_match_rate": bool_rate("top1_match"),
                "kl_dense_to_sparse_mean": mean(
                    [float(row["kl_dense_to_sparse"]) for row in group]
                ),
                "actual_dense_accept_rate": bool_rate("actual_dense_accept"),
                "actual_sparse_accept_rate": bool_rate("actual_sparse_accept"),
                "actual_false_accept_rate": bool_rate("actual_false_accept"),
                "actual_false_reject_rate": bool_rate("actual_false_reject"),
                "counterfactual_top2_false_accept_rate": bool_rate(
                    "counterfactual_top2_false_accept"
                ),
                "avg_union_blocks": mean([float(row["union_blocks"]) for row in group]),
                "avg_mean_blocks_per_token": mean(
                    [float(row["mean_blocks_per_token"]) for row in group]
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-reference-jsonl", required=True)
    parser.add_argument(
        "--dataset-jsonl",
        help=(
            "Optional dataset JSONL whose prompts replace the dense-reference "
            "prompts by matching record id. Use this for padded-context quality checks."
        ),
    )
    parser.add_argument("--context-label", default="orig")
    parser.add_argument(
        "--candidate-source",
        choices=["position_recency_proxy", "hidden_similarity_proxy"],
        default="position_recency_proxy",
    )
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-summary-md", required=True)
    parser.add_argument(
        "--out-traces-jsonl",
        help="Optional normalized sparse trace JSONL for layout simulation.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-positions", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-spec-tokens", type=int, default=8)
    parser.add_argument(
        "--layouts",
        default="independent_topk,snapkv_static,shared_only,speclink_prob,speclink_prob_fallback",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--topk-per-token", type=int, default=32)
    parser.add_argument("--shared-budget", type=int, default=16)
    parser.add_argument("--private-min", type=int, default=0)
    parser.add_argument("--private-max", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--lambda-risk", type=float, default=0.0)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument(
        "--accept-probs",
        default="0.70,0.55,0.40,0.30,0.22,0.16,0.12,0.09",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-local-draft", action="store_true", default=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-8B offline sparse quality eval")
    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    dense_refs = read_jsonl(Path(args.dense_reference_jsonl))
    if args.dataset_jsonl:
        by_id = {str(row.get("id")): row for row in dense_refs}
        refs = []
        for index, record in enumerate(load_dataset(args.dataset_jsonl, limit=args.limit)):
            dense = by_id.get(record.id)
            if dense is None:
                continue
            refs.append(
                {
                    **dense,
                    "id": record.id,
                    "index": index,
                    "prompt": record.prompt,
                    "reference_raw": record.reference_raw,
                }
            )
    else:
        refs = dense_refs[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    all_rows: list[dict[str, Any]] = []
    all_trace_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in refs:
        sample_rows, trace_rows, stats = evaluate_sample(
            row=row,
            tokenizer=tokenizer,
            model=model,
            args=args,
            device=device,
            dtype=dtype,
        )
        skipped += int(stats["skipped"])
        all_rows.extend(sample_rows)
        all_trace_rows.extend(trace_rows)

    summary = summarize(all_rows)
    write_jsonl(Path(args.out_jsonl), all_rows)
    if args.out_traces_jsonl:
        write_jsonl(Path(args.out_traces_jsonl), all_trace_rows)
    write_csv(Path(args.out_summary_csv), summary)
    write_md(Path(args.out_summary_md), summary)
    print(
        "evaluated "
        f"{len(refs) - skipped}/{len(refs)} samples, {len(all_rows)} layout-position rows"
    )
    print(f"wrote {args.out_jsonl}")
    if args.out_traces_jsonl:
        print(f"wrote {args.out_traces_jsonl}")
    print(f"wrote {args.out_summary_csv}")
    print(f"wrote {args.out_summary_md}")


if __name__ == "__main__":
    main()
