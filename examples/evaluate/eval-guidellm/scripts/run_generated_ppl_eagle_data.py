#!/usr/bin/env python3
"""Generated-PPL sweep on EAGLE Alpaca, SUM, and MT-Bench prompts.

Protocol
--------
For each model and method, vLLM EAGLE3 first generates a complete response.
The exact prompt and output token IDs returned by vLLM are retained.  A second,
plain dense vLLM server then teacher-forces every ``prompt + generated answer``
sequence and reports perplexity over generated answer tokens only:

    Generated PPL = exp(-sum(answer-token log p_dense) / answer-token count)

This is intentionally different from corpus/ground-truth PPL.  D0--D8 affect
generation only; the scorer is always the original dense base model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
EVAL_ROOT = SCRIPT_DIR.parent
SPECULATORS_ROOT = EVAL_ROOT.parents[2]
MODELS_ROOT = SPECULATORS_ROOT.parent / "models"
RESULTS_FINAL_ROOT = EVAL_ROOT / "results_final"
TEMP_ROOT = EVAL_ROOT / "temp"
DATA_ROOT = EVAL_ROOT / "data" / "eagle_generated_ppl"
CALIBRATION_ROOT = (
    EVAL_ROOT
    / "data"
    / "c4_calibration"
    / "activation_rms"
    / "c4_512_seed42_bf16_max512"
)

sys.path.insert(0, str(SCRIPT_DIR))
import run_lm_eval_official_quota_sweep as quota  # noqa: E402
from run_structured_24_spec_quality import (  # noqa: E402
    add_local_no_proxy,
    find_free_port,
    stop_process,
    wait_for_health,
)


EAGLE_COMMIT = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
EAGLE_REPOSITORY = "https://github.com/SafeAILab/EAGLE"
DATASETS = {
    "alpaca": {
        "rows": 80,
        "turns": 80,
        "sha256": "8ad3e5a7fc61b88d8e25edb81a52a2ce00a04494d5694c122cb11a5cc26a00df",
    },
    "sum": {
        "rows": 80,
        "turns": 80,
        "sha256": "916876f0c42f938a944de96eed16e51425f29f27438cca5c9ef4737c0fedf771",
    },
    "mt_bench": {
        "rows": 80,
        "turns": 160,
        "sha256": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    },
}
MODEL_CONFIGS = quota.MODEL_CONFIGS
DENSE_METHOD = "dense_eagle3"
MAX_GEN_TOKS = 256
DEFAULT_LLAMA_DENSE_LEAFS = ("gate_up_proj", "o_proj")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_eighths(value: str) -> list[int]:
    values: list[int] = []
    for item in parse_csv(value):
        parsed = int(item)
        if not 0 <= parsed <= 8:
            raise ValueError(f"dense eighths must be in [0,8], got {parsed}")
        if parsed not in values:
            values.append(parsed)
    return values


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=SPECULATORS_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def prepare_datasets() -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "repository": EAGLE_REPOSITORY,
        "commit": EAGLE_COMMIT,
        "datasets": {},
    }
    for name, spec in DATASETS.items():
        path = DATA_ROOT / name / "question.jsonl"
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/SafeAILab/EAGLE/"
                f"{EAGLE_COMMIT}/eagle/data/{name}/question.jsonl"
            )
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if sha256_bytes(payload) != spec["sha256"]:
                raise RuntimeError(f"downloaded {name} SHA256 mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        actual_hash = sha256_file(path)
        rows = read_jsonl(path)
        actual_turns = sum(len(row.get("turns") or []) for row in rows)
        if (
            actual_hash != spec["sha256"]
            or len(rows) != spec["rows"]
            or actual_turns != spec["turns"]
        ):
            raise RuntimeError(
                f"{name} audit failed: sha={actual_hash}, "
                f"rows={len(rows)}, turns={actual_turns}"
            )
        manifest["datasets"][name] = {
            "path": str(path.relative_to(EVAL_ROOT)),
            "sha256": actual_hash,
            "rows": len(rows),
            "turns": actual_turns,
            "source": (
                f"{EAGLE_REPOSITORY}/blob/{EAGLE_COMMIT}/"
                f"eagle/data/{name}/question.jsonl"
            ),
        }
    write_json(DATA_ROOT / "manifest.json", manifest)
    return manifest


def method_name(eighths: int | None) -> str:
    return DENSE_METHOD if eighths is None else f"d{eighths}"


def method_display(eighths: int | None) -> str:
    return "Dense EAGLE3" if eighths is None else f"D{eighths}/8"


def method_order(method: str) -> int:
    return 9 if method == DENSE_METHOD else int(method.removeprefix("d"))


def dense_leafs_for(model_label: str, args: argparse.Namespace) -> list[str]:
    configured = getattr(args, "dense_leafs_list", [])
    if configured:
        return list(configured)
    if model_label == "llama3_1_8b":
        return list(DEFAULT_LLAMA_DENSE_LEAFS)
    return []


def args_for_model(
    args: argparse.Namespace, model_label: str
) -> argparse.Namespace:
    """Return an isolated per-model config for shared quota helpers.

    ``quota.make_env`` reads ``dense_leafs_list`` from the namespace.  Never
    mutate the sweep-wide namespace here: otherwise the default Llama dense
    leaves can leak into a later Qwen run when model order is changed.
    """

    model_args = argparse.Namespace(**vars(args))
    model_args.dense_leafs_list = dense_leafs_for(model_label, args)
    return model_args


def load_questions(dataset: str, limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl(DATA_ROOT / dataset / "question.jsonl")
    return rows if limit <= 0 else rows[:limit]


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    model_label: str,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
    }
    if model_label == "qwen3_8b":
        kwargs["enable_thinking"] = False
    token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("chat template did not return a flat token-id list")
    return token_ids


def post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_one(
    *,
    port: int,
    served_model: str,
    prompt_token_ids: list[int],
    generation: dict[str, Any],
    request_id: str,
    seed: int,
    timeout_s: float,
    retries: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": served_model,
        "prompt": prompt_token_ids,
        "add_special_tokens": False,
        "max_tokens": MAX_GEN_TOKS,
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "seed": seed,
        "stream": False,
        "request_id": request_id,
        "return_token_ids": True,
    }
    if "top_k" in generation:
        body["top_k"] = generation["top_k"]
        body["min_p"] = generation["min_p"]
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            payload = post_json(
                f"http://127.0.0.1:{port}/v1/completions",
                body,
                timeout_s=timeout_s,
            )
            choice = (payload.get("choices") or [])[0]
            returned_prompt = choice.get("prompt_token_ids")
            output_ids = choice.get("token_ids")
            if returned_prompt != prompt_token_ids:
                raise RuntimeError("vLLM returned prompt token IDs differ")
            if not isinstance(output_ids, list):
                raise RuntimeError("vLLM did not return output token IDs")
            return {
                "status": "ok",
                "response": str(choice.get("text") or ""),
                "prompt_token_ids": returned_prompt,
                "output_token_ids": [int(item) for item in output_ids],
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "prompt_tokens": len(returned_prompt),
                "generated_tokens": len(output_ids),
                "latency_sec": time.perf_counter() - started,
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {detail}"
            else:
                last_error = repr(exc)
            if attempt < retries:
                time.sleep(float(attempt))
    return {
        "status": "failed",
        "response": "",
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": [],
        "finish_reason": None,
        "stop_reason": None,
        "prompt_tokens": len(prompt_token_ids),
        "generated_tokens": 0,
        "latency_sec": None,
        "attempts": retries,
        "error": last_error,
    }


def generate_dataset(
    *,
    args: argparse.Namespace,
    port: int,
    model_label: str,
    served_model: str,
    tokenizer: Any,
    dataset: str,
    method: str,
) -> list[dict[str, Any]]:
    questions = load_questions(dataset, args.limit)
    conversations: dict[int, list[dict[str, str]]] = {
        index: [] for index in range(len(questions))
    }
    records: list[dict[str, Any]] = []
    max_turns = max(len(row["turns"]) for row in questions)
    generation = quota.generation_protocol(
        model_label, llama_qwen_sampling=False
    )

    for turn_index in range(max_turns):
        jobs: list[tuple[int, dict[str, Any], list[int]]] = []
        for index, question in enumerate(questions):
            turns = question["turns"]
            if turn_index >= len(turns):
                continue
            conversations[index].append(
                {"role": "user", "content": str(turns[turn_index])}
            )
            prompt_ids = apply_chat_template(
                tokenizer,
                conversations[index],
                model_label=model_label,
            )
            if len(prompt_ids) + MAX_GEN_TOKS > args.max_model_len:
                raise RuntimeError(
                    f"{dataset} question {question['question_id']} turn "
                    f"{turn_index + 1}: prompt {len(prompt_ids)} + "
                    f"{MAX_GEN_TOKS} exceeds max_model_len={args.max_model_len}"
                )
            jobs.append((index, question, prompt_ids))

        completed: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    generate_one,
                    port=port,
                    served_model=served_model,
                    prompt_token_ids=prompt_ids,
                    generation=generation,
                    request_id=(
                        f"generated-ppl-{model_label}-{method}-{dataset}-"
                        f"{question['question_id']}-t{turn_index + 1}"
                    ),
                    seed=args.seed,
                    timeout_s=args.request_timeout_s,
                    retries=args.request_retries,
                ): (index, question, prompt_ids)
                for index, question, prompt_ids in jobs
            }
            for future in as_completed(futures):
                index, question, _ = futures[future]
                result = future.result()
                result.update(
                    {
                        "model_label": model_label,
                        "method": method,
                        "dataset": dataset,
                        "question_id": question["question_id"],
                        "category": question.get("category", ""),
                        "turn_index": turn_index,
                        "seed": args.seed,
                    }
                )
                completed[index] = result

        for index, question, _ in jobs:
            result = completed[index]
            if result["status"] != "ok":
                raise RuntimeError(
                    f"generation failed for {dataset}/"
                    f"{question['question_id']}/turn{turn_index + 1}: "
                    f"{result.get('error')}"
                )
            conversations[index].append(
                {"role": "assistant", "content": result["response"]}
            )
            records.append(result)
    records.sort(key=lambda row: (int(row["question_id"]), int(row["turn_index"])))
    return records


def generation_file(
    output_root: Path, model_label: str, method: str, dataset: str
) -> Path:
    return output_root / model_label / method / dataset / "generations.jsonl"


def expected_responses(dataset: str, limit: int) -> int:
    return sum(len(row["turns"]) for row in load_questions(dataset, limit))


def existing_generation_complete(
    path: Path, *, dataset: str, limit: int
) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return (
        len(rows) == expected_responses(dataset, limit)
        and all(row.get("status") == "ok" for row in rows)
        and all(len(row.get("output_token_ids") or []) > 0 for row in rows)
    )


def dense_server_env() -> dict[str, str]:
    env = add_local_no_proxy(os.environ.copy())
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    for name in list(env):
        if name.startswith("SPECLINK_STRUCTURED_24_") or name.startswith(
            "SPECLINK_TOKEN_DENSE_"
        ):
            env.pop(name, None)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "SPECLINK_STRUCTURED_24_ENABLE": "0",
            "SPECLINK_TOKEN_DENSE_ENABLE": "0",
        }
    )
    return env


def write_case_runtime_audit(
    *,
    case_dir: Path,
    model_label: str,
    dense_eighths: int | None,
    expected_dense_leafs: list[str],
    calibration_cache_root: Path,
) -> None:
    stats_path = case_dir / "vllm_structured_24_stats.json"
    if dense_eighths is None:
        record = {
            "status": "passed",
            "model_label": model_label,
            "method": DENSE_METHOD,
            "structured_24_enabled": False,
            "token_dense_enabled": False,
            "runtime_sparse_value_loading": False,
            "stats_file_expected": False,
        }
        write_json(case_dir / "runtime_audit.json", record)
        return

    if not stats_path.exists():
        raise RuntimeError(f"missing runtime structured-2:4 stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    sparse_modules = [
        module
        for module in stats.get("per_module") or []
        if not module.get("kept_dense")
    ]
    runtime_storage = [
        str((module.get("residual_complement_runtime") or {}).get("storage"))
        for module in sparse_modules
    ]
    errors: list[str] = []
    checks = {
        "enabled": stats.get("enabled") is True,
        "model_label": stats.get("model_label") == model_label,
        "policy": stats.get("policy") == "all_sparse",
        "token_dense_enabled": stats.get("token_dense_enabled") is True,
        "token_dense_backend": (
            stats.get("token_dense_backend")
            == "residual_complement_splitk2"
        ),
        "actual_sparsity": math.isclose(
            float(stats.get("actual_sparsity") or -1.0), 0.5, abs_tol=1e-12
        ),
        "calibration_cache_root": (
            Path(str(stats.get("calibration_cache_root"))).resolve()
            == calibration_cache_root.resolve()
        ),
        "dense_leafs": sorted(stats.get("dense_leafs") or [])
        == sorted(expected_dense_leafs),
        "sparse_modules_present": bool(sparse_modules),
        "runtime_storage": bool(runtime_storage)
        and all(
            item == "cusparselt_base_plus_complement_no_duplicate_metadata"
            for item in runtime_storage
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    record = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "model_label": model_label,
        "method": f"d{dense_eighths}",
        "dense_eighths": dense_eighths,
        "structured_24_enabled": True,
        "token_dense_enabled": True,
        "runtime_sparse_value_loading": True,
        "nm_structure": "2:4 base + complementary 2:4 residual",
        "mask_method": "C4 activation-aware/Wanda cache",
        "calibration_cache_root": str(calibration_cache_root.resolve()),
        "always_dense_leafs": expected_dense_leafs,
        "masked_module_count": len(sparse_modules),
        "runtime_storage": sorted(set(runtime_storage)),
        "stats_path": str(stats_path.resolve()),
        "stats_sha256": sha256_file(stats_path),
        "checks": checks,
    }
    write_json(case_dir / "runtime_audit.json", record)
    if errors:
        raise RuntimeError(f"runtime audit failed: {', '.join(errors)}")


def start_dense_scorer(
    args: argparse.Namespace,
    *,
    model: Path,
    score_dir: Path,
) -> tuple[subprocess.Popen[Any], int]:
    port = find_free_port(args.port_base + 50)
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "spec",
        "vllm",
        "serve",
        str(model.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--seed",
        str(args.seed),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--generation-config",
        "vllm",
        "--no-enable-prefix-caching",
    ]
    score_dir.mkdir(parents=True, exist_ok=True)
    write_json(score_dir / "server_command.json", {"command": command, "port": port})
    log_handle = (score_dir / "vllm_dense_scorer.log").open(
        "w", encoding="utf-8"
    )
    log_handle.write("$ " + " ".join(command) + "\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=EVAL_ROOT,
        env=dense_server_env(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_health(port, process, args.health_timeout_s)
    except Exception:
        stop_process(process)
        raise
    return process, port


def chosen_prompt_logprob(entry: Any, token_id: int) -> float:
    if not isinstance(entry, dict):
        raise RuntimeError(f"missing prompt logprob for token {token_id}")
    item = entry.get(str(token_id))
    if item is None:
        item = entry.get(token_id)
    if item is None and len(entry) == 1:
        item = next(iter(entry.values()))
    if not isinstance(item, dict) or "logprob" not in item:
        raise RuntimeError(
            f"chosen token {token_id} absent from prompt-logprob entry"
        )
    return float(item["logprob"])


def score_batch(
    *,
    port: int,
    served_model: str,
    rows: list[dict[str, Any]],
    timeout_s: float,
) -> list[dict[str, Any]]:
    sequences = [
        [*row["prompt_token_ids"], *row["output_token_ids"]] for row in rows
    ]
    payload = post_json(
        f"http://127.0.0.1:{port}/v1/completions",
        {
            "model": served_model,
            "prompt": sequences,
            "add_special_tokens": False,
            "max_tokens": 0,
            "echo": True,
            "prompt_logprobs": 0,
            "return_token_ids": True,
            "stream": False,
        },
        timeout_s=timeout_s,
    )
    choices = payload.get("choices") or []
    if len(choices) != len(rows):
        raise RuntimeError(
            f"dense scorer returned {len(choices)} choices for {len(rows)} rows"
        )
    output: list[dict[str, Any]] = []
    for choice in sorted(choices, key=lambda item: int(item["index"])):
        index = int(choice["index"])
        source = rows[index]
        prompt_ids = source["prompt_token_ids"]
        output_ids = source["output_token_ids"]
        returned_ids = choice.get("prompt_token_ids")
        sequence = sequences[index]
        if returned_ids != sequence:
            raise RuntimeError("dense scorer changed the supplied token IDs")
        prompt_logprobs = choice.get("prompt_logprobs")
        if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(
            sequence
        ):
            raise RuntimeError("dense scorer returned incomplete prompt logprobs")
        answer_logprobs = [
            chosen_prompt_logprob(prompt_logprobs[position], token_id)
            for position, token_id in enumerate(
                output_ids, start=len(prompt_ids)
            )
        ]
        if not answer_logprobs:
            raise RuntimeError("generated answer has zero scored tokens")
        output.append(
            {
                **{
                    key: source[key]
                    for key in (
                        "model_label",
                        "method",
                        "dataset",
                        "question_id",
                        "turn_index",
                        "generated_tokens",
                    )
                },
                "answer_nll": -sum(answer_logprobs),
                "answer_mean_nll": -sum(answer_logprobs) / len(answer_logprobs),
                "answer_ppl": math.exp(
                    min(709.0, -sum(answer_logprobs) / len(answer_logprobs))
                ),
                "scored_tokens": len(answer_logprobs),
            }
        )
    return output


def score_model(
    args: argparse.Namespace,
    *,
    output_root: Path,
    model_label: str,
    model: Path,
    methods: list[str],
) -> list[dict[str, Any]]:
    score_dir = output_root / model_label / "dense_scorer"
    all_generations: list[dict[str, Any]] = []
    for dataset in args.datasets_list:
        # Adjacent methods share identical first-turn prompt prefixes.  The
        # scorer disables prefix caching for strict, implementation-agnostic
        # dense likelihoods, but this ordering also makes audits easy to read.
        for method in methods:
            all_generations.extend(
                read_jsonl(
                    generation_file(output_root, model_label, method, dataset)
                )
            )
    all_generations.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["question_id"]),
            int(row["turn_index"]),
            method_order(str(row["method"])),
        )
    )

    process = None
    scored: list[dict[str, Any]] = []
    try:
        for start_attempt in range(1, args.server_start_retries + 1):
            try:
                process, port = start_dense_scorer(
                    args, model=model, score_dir=score_dir
                )
                break
            except RuntimeError:
                failed_log = score_dir / (
                    f"vllm_dense_scorer_start_failure_{start_attempt}.log"
                )
                server_log = score_dir / "vllm_dense_scorer.log"
                if server_log.exists():
                    server_log.replace(failed_log)
                if start_attempt >= args.server_start_retries:
                    raise
                print(
                    f"[scorer-retry] {model_label} attempt {start_attempt}/"
                    f"{args.server_start_retries} failed",
                    flush=True,
                )
                time.sleep(args.server_shutdown_settle_s)
        if process is None:
            raise RuntimeError("dense scorer did not start")
        for offset in range(0, len(all_generations), args.score_batch_size):
            batch = all_generations[offset : offset + args.score_batch_size]
            scored.extend(
                score_batch(
                    port=port,
                    served_model=str(model.resolve()),
                    rows=batch,
                    timeout_s=args.request_timeout_s,
                )
            )
            print(
                f"[score] {model_label} "
                f"{min(offset + len(batch), len(all_generations))}/"
                f"{len(all_generations)}",
                flush=True,
            )
    finally:
        stop_process(process)
        time.sleep(args.server_shutdown_settle_s)
    write_jsonl(score_dir / "answer_scores.jsonl", scored)
    return scored


def aggregate_scores(
    scored: list[dict[str, Any]], *, methods: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in scored:
        key = (row["model_label"], row["method"], row["dataset"])
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (model_label, method, dataset), values in grouped.items():
        total_nll = sum(float(row["answer_nll"]) for row in values)
        tokens = sum(int(row["scored_tokens"]) for row in values)
        rows.append(
            {
                "model_label": model_label,
                "model": MODEL_CONFIGS[model_label]["display"],
                "method": method,
                "dense_eighths": (
                    None if method == DENSE_METHOD else method_order(method)
                ),
                "dataset": dataset,
                "generated_ppl": math.exp(min(709.0, total_nll / tokens)),
                "answer_nll": total_nll,
                "scored_tokens": tokens,
                "responses": len(values),
                "questions": len(
                    {str(row["question_id"]) for row in values}
                ),
                "complete": True,
            }
        )
    for model_label in sorted({row["model_label"] for row in rows}):
        for method in methods:
            values = [
                row
                for row in rows
                if row["model_label"] == model_label
                and row["method"] == method
            ]
            if not values:
                continue
            total_nll = sum(float(row["answer_nll"]) for row in values)
            tokens = sum(int(row["scored_tokens"]) for row in values)
            rows.append(
                {
                    "model_label": model_label,
                    "model": MODEL_CONFIGS[model_label]["display"],
                    "method": method,
                    "dense_eighths": (
                        None if method == DENSE_METHOD else method_order(method)
                    ),
                    "dataset": "all_token_weighted",
                    "generated_ppl": math.exp(
                        min(709.0, total_nll / tokens)
                    ),
                    "answer_nll": total_nll,
                    "scored_tokens": tokens,
                    "responses": sum(int(row["responses"]) for row in values),
                    "questions": sum(int(row["questions"]) for row in values),
                    "complete": True,
                }
            )
    dense_refs = {
        (row["model_label"], row["dataset"]): float(row["generated_ppl"])
        for row in rows
        if row["method"] == DENSE_METHOD
    }
    for row in rows:
        reference = dense_refs.get((row["model_label"], row["dataset"]))
        row["dense_eagle3_generated_ppl"] = reference
        row["relative_ppl_change_pct"] = (
            100.0 * (float(row["generated_ppl"]) / reference - 1.0)
            if reference
            else None
        )
    rows.sort(
        key=lambda row: (
            str(row["model_label"]),
            method_order(str(row["method"])),
            str(row["dataset"]),
        )
    )
    return rows


def plot_summary(
    output_root: Path, summary: list[dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    model_labels = [
        label
        for label in ("qwen3_8b", "llama3_1_8b")
        if any(row["model_label"] == label for row in summary)
    ]
    if not model_labels:
        return
    methods = [*(f"d{index}" for index in range(9)), DENSE_METHOD]
    datasets = ("all_token_weighted", "alpaca", "sum", "mt_bench")
    display = {
        "all_token_weighted": "All (token-weighted)",
        "alpaca": "Alpaca",
        "sum": "SUM",
        "mt_bench": "MT-Bench",
    }
    colors = {
        "all_token_weighted": "#222222",
        "alpaca": "#4c78a8",
        "sum": "#f58518",
        "mt_bench": "#54a24b",
    }
    fig, axes = plt.subplots(
        1,
        len(model_labels),
        figsize=(6.3 * len(model_labels), 4.4),
        squeeze=False,
        constrained_layout=True,
    )
    lookup = {
        (row["model_label"], row["method"], row["dataset"]): float(
            row["generated_ppl"]
        )
        for row in summary
    }
    x = list(range(len(methods)))
    for axis, model_label in zip(axes[0], model_labels):
        for dataset in datasets:
            values = [
                lookup.get((model_label, method, dataset), math.nan)
                for method in methods
            ]
            axis.plot(
                x,
                values,
                marker="o",
                linewidth=2.2 if dataset == "all_token_weighted" else 1.5,
                markersize=5,
                color=colors[dataset],
                label=display[dataset],
            )
        axis.axvline(8.5, color="#999999", linestyle="--", linewidth=1)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [*(f"D{index}" for index in range(9)), "Dense"],
            rotation=35,
            ha="right",
        )
        axis.set_title(str(MODEL_CONFIGS[model_label]["display"]))
        axis.set_xlabel("Dense-token quota")
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Generated PPL (lower is better)")
    axes[0][-1].legend(frameon=False, fontsize=9)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "generated_ppl_by_quota.png", dpi=200)
    plt.close(fig)


def write_report(
    output_root: Path,
    summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Generated PPL: EAGLE3 and SpecLink D0-D8",
        "",
        "Generated PPL is computed by first generating a complete answer with "
        "the evaluated method, then teacher-forcing the exact generated token "
        "IDs with the original dense base model. Only answer tokens contribute "
        "to NLL and token count.",
        "",
        "## Protocol",
        "",
        f"- Data: SafeAILab/EAGLE commit `{EAGLE_COMMIT}`, datasets "
        "`alpaca`, `sum`, and `mt_bench`.",
        f"- Maximum generated tokens per response: {MAX_GEN_TOKS}.",
        "- MT-Bench uses both turns; turn 2 includes the same method's "
        "generated turn-1 assistant response.",
        "- Prompting uses each checkpoint's unmodified "
        "`tokenizer_config.json -> chat_template`.",
        "- Qwen3-8B: thinking disabled; temperature=0.7, top_p=0.8, "
        "top_k=20, min_p=0.",
        "- Llama-3.1-8B: greedy generation. Its `gate_up_proj` and `o_proj` "
        "remain dense; D0-D8 route qkv/down as in the current vLLM policy.",
        "- Seed=42. D0-D8 use EAGLE3 K=7 and the global prefix-product quota.",
        "- The dense scorer is a separate plain vLLM server with all "
        "SpecLink structured-sparsity hooks disabled.",
        "",
        "## Token-weighted result across all three datasets",
        "",
        "| Model | Method | Generated PPL | vs dense EAGLE3 | Tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["dataset"] != "all_token_weighted":
            continue
        relative = row.get("relative_ppl_change_pct")
        relative_text = (
            f"{float(relative):+.3f}%" if relative is not None else "n/a"
        )
        lines.append(
            f"| {row['model']} | {row['method']} | "
            f"{float(row['generated_ppl']):.6f} | "
            f"{relative_text} | {int(row['scored_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: dataset-level and all-dataset token-weighted PPL.",
            "- `figures/generated_ppl_by_quota.png`: all datasets and the "
            "token-weighted aggregate.",
            "- `*/dense_scorer/answer_scores.jsonl`: per-response dense NLL.",
            "- `*/METHOD/DATASET/generations.jsonl`: exact generation text and "
            "prompt/output token IDs.",
            "- `experiment_audit.json`: repository, model, data, mask, and "
            "runtime protocol audit.",
            "",
        ]
    )
    (output_root / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def experiment_audit(
    args: argparse.Namespace, dataset_manifest: dict[str, Any]
) -> dict[str, Any]:
    diff = git_output("diff", "--binary", "HEAD")
    source_files = [
        SCRIPT,
        SCRIPT_DIR / "run_lm_eval_official_quota_sweep.py",
        SCRIPT_DIR / "run_structured_24_spec_quality.py",
        SPECULATORS_ROOT / "vllm" / "vllm" / "speclink_token_dense.py",
        SPECULATORS_ROOT / "vllm" / "vllm" / "speclink_structured_24.py",
    ]
    tokenizer_audits: dict[str, Any] = {}
    model_audits: dict[str, Any] = {}
    for model_label in args.models_list:
        model = Path(MODEL_CONFIGS[model_label]["model"]).resolve()
        speculator = Path(MODEL_CONFIGS[model_label]["speculator"]).resolve()
        tokenizer_config = model / "tokenizer_config.json"
        config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        if not config.get("chat_template"):
            raise RuntimeError(f"{tokenizer_config} has no chat_template")
        tokenizer_audits[model_label] = {
            "tokenizer_config": str(tokenizer_config),
            "tokenizer_config_sha256": sha256_file(tokenizer_config),
            "chat_template_sha256": sha256_bytes(
                str(config["chat_template"]).encode("utf-8")
            ),
            "chat_template_modified": False,
        }
        model_audits[model_label] = {
            "base": quota.checkpoint_audit(model),
            "speculator": quota.checkpoint_audit(speculator),
            "always_dense_leafs": dense_leafs_for(model_label, args),
        }
    calibration_root = args.calibration_cache_root.resolve()
    calibration_manifest = calibration_root / "manifest.json"
    return {
        "created_at": timestamp(),
        "source": {
            "repository": git_output("remote", "get-url", "origin"),
            "speculators_commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "dirty_files": git_output("status", "--short").splitlines(),
            "working_tree_diff_sha256": sha256_bytes(diff.encode("utf-8")),
            "source_file_sha256": {
                str(path.relative_to(SPECULATORS_ROOT)): sha256_file(path)
                for path in source_files
            },
            "vllm_version": importlib.metadata.version("vllm"),
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
        },
        "datasets": dataset_manifest,
        "models": model_audits,
        "tokenizers": tokenizer_audits,
        "calibration": {
            "root": str(calibration_root),
            "manifest": (
                str(calibration_manifest.resolve())
                if calibration_manifest.exists()
                else None
            ),
            "manifest_sha256": (
                sha256_file(calibration_manifest)
                if calibration_manifest.exists()
                else None
            ),
        },
        "protocol": {
            "definition": (
                "generate with evaluated sparse/dense method, then score exact "
                "generated answer tokens under original dense base model"
            ),
            "datasets": args.datasets_list,
            "models": args.models_list,
            "methods": [
                *([] if args.skip_dense_eagle3 else [DENSE_METHOD]),
                *(f"d{item}" for item in args.eighths_list),
            ],
            "seed": args.seed,
            "num_spec_tokens": args.num_spec_tokens,
            "max_gen_toks": MAX_GEN_TOKS,
            "qwen_generation": quota.generation_protocol(
                "qwen3_8b", llama_qwen_sampling=False
            ),
            "llama_generation": quota.generation_protocol(
                "llama3_1_8b", llama_qwen_sampling=False
            ),
            "teacher_forced_scope": "generated_answer_tokens_only",
            "generated_eos_scored": False,
            "text_retokenized_for_scoring": False,
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.seed != 42:
        raise ValueError("formal protocol requires seed=42")
    if args.num_spec_tokens != 7:
        raise ValueError("D0-D8 eighth quotas require EAGLE3 K=7")
    if args.max_model_len < 2048:
        raise ValueError("--max-model-len must be at least 2048")
    if args.request_retries < 1:
        raise ValueError("--request-retries must be positive")
    if args.server_start_retries < 1:
        raise ValueError("--server-start-retries must be positive")
    if args.score_batch_size < 1:
        raise ValueError("--score-batch-size must be positive")
    for model_label in args.models_list:
        if model_label not in MODEL_CONFIGS:
            raise ValueError(f"unknown model {model_label}")
        for key in ("model", "speculator"):
            path = Path(MODEL_CONFIGS[model_label][key])
            if not path.exists():
                raise FileNotFoundError(path)
    for dataset in args.datasets_list:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset}")
    if not args.calibration_cache_root.exists():
        raise FileNotFoundError(args.calibration_cache_root)
    if not args.smoke and args.limit:
        raise ValueError("formal results cannot use --limit")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    dataset_manifest = prepare_datasets()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit = experiment_audit(args, dataset_manifest)
    write_json(output_root / "experiment_audit.json", audit)
    write_json(
        output_root / "run_config.json",
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not key.endswith("_list")
            },
            "models": args.models_list,
            "datasets": args.datasets_list,
            "eighths": args.eighths_list,
            "max_gen_toks": MAX_GEN_TOKS,
        },
    )
    method_values: list[int | None] = [
        *([] if args.skip_dense_eagle3 else [None]),
        *args.eighths_list,
    ]
    methods = [method_name(item) for item in method_values]
    server_args = quota.make_server_args(args)

    for model_label in args.models_list:
        model = Path(MODEL_CONFIGS[model_label]["model"]).resolve()
        speculator = Path(MODEL_CONFIGS[model_label]["speculator"]).resolve()
        tokenizer = AutoTokenizer.from_pretrained(
            model, local_files_only=True, trust_remote_code=False
        )
        model_args = args_for_model(args, model_label)
        model_dense_leafs = model_args.dense_leafs_list
        for dense_eighths in method_values:
            method = method_name(dense_eighths)
            pending = [
                dataset
                for dataset in args.datasets_list
                if not (
                    args.resume
                    and existing_generation_complete(
                        generation_file(
                            output_root, model_label, method, dataset
                        ),
                        dataset=dataset,
                        limit=args.limit,
                    )
                )
            ]
            if not pending:
                continue
            case_dir = output_root / model_label / method
            print(
                f"[server] {model_label} {method_display(dense_eighths)} "
                f"pending={','.join(pending)}",
                flush=True,
            )
            env = quota.make_env(
                model_args,
                model_label=model_label,
                dense_eighths=dense_eighths,
                case_dir=case_dir,
            )
            write_json(
                case_dir / "case_config.json",
                {
                    "model_label": model_label,
                    "method": method,
                    "dense_eighths": dense_eighths,
                    "num_spec_tokens": args.num_spec_tokens,
                    "always_dense_leafs": model_dense_leafs,
                    "generation": quota.generation_protocol(
                        model_label, llama_qwen_sampling=False
                    ),
                    "chat_template_source": (
                        "checkpoint tokenizer_config.json -> chat_template"
                    ),
                    "chat_template_modified": False,
                },
            )
            process = None
            try:
                for start_attempt in range(
                    1, args.server_start_retries + 1
                ):
                    try:
                        process, port = quota.start_vllm_server(
                            server_args,
                            base_model=str(model),
                            speculator_model=str(speculator),
                            case_dir=case_dir,
                            env=env,
                        )
                        break
                    except RuntimeError:
                        failed_log = case_dir / (
                            "vllm_server_start_failure_"
                            f"{start_attempt}.log"
                        )
                        server_log = case_dir / "vllm_server.log"
                        if server_log.exists():
                            server_log.replace(failed_log)
                        if start_attempt >= args.server_start_retries:
                            raise
                        print(
                            f"[server-retry] {model_label} "
                            f"{method_display(dense_eighths)} attempt "
                            f"{start_attempt}/"
                            f"{args.server_start_retries} failed",
                            flush=True,
                        )
                        time.sleep(args.server_shutdown_settle_s)
                if process is None:
                    raise RuntimeError("vLLM server did not start")
                write_case_runtime_audit(
                    case_dir=case_dir,
                    model_label=model_label,
                    dense_eighths=dense_eighths,
                    expected_dense_leafs=model_dense_leafs,
                    calibration_cache_root=args.calibration_cache_root,
                )
                for dataset in pending:
                    print(
                        f"[generate] {model_label} {method} {dataset}",
                        flush=True,
                    )
                    records = generate_dataset(
                        args=args,
                        port=port,
                        model_label=model_label,
                        served_model=str(model),
                        tokenizer=tokenizer,
                        dataset=dataset,
                        method=method,
                    )
                    path = generation_file(
                        output_root, model_label, method, dataset
                    )
                    write_jsonl(path, records)
                    write_json(
                        path.with_name("generation_summary.json"),
                        {
                            "responses": len(records),
                            "questions": len(
                                {str(row["question_id"]) for row in records}
                            ),
                            "generated_tokens": sum(
                                int(row["generated_tokens"]) for row in records
                            ),
                            "mean_latency_sec": sum(
                                float(row["latency_sec"]) for row in records
                            )
                            / len(records),
                            "complete": existing_generation_complete(
                                path, dataset=dataset, limit=args.limit
                            ),
                        },
                    )
            finally:
                stop_process(process)
                time.sleep(args.server_shutdown_settle_s)

        scored_path = (
            output_root
            / model_label
            / "dense_scorer"
            / "answer_scores.jsonl"
        )
        if args.resume and scored_path.exists():
            scored = read_jsonl(scored_path)
            expected = sum(
                expected_responses(dataset, args.limit)
                for dataset in args.datasets_list
            ) * len(methods)
            if len(scored) != expected:
                scored = score_model(
                    args,
                    output_root=output_root,
                    model_label=model_label,
                    model=model,
                    methods=methods,
                )
        else:
            scored = score_model(
                args,
                output_root=output_root,
                model_label=model_label,
                model=model,
                methods=methods,
            )

        existing_scores: list[dict[str, Any]] = []
        for other_model in args.models_list:
            path = (
                output_root
                / other_model
                / "dense_scorer"
                / "answer_scores.jsonl"
            )
            if path.exists():
                existing_scores.extend(read_jsonl(path))
        summary = aggregate_scores(existing_scores, methods=methods)
        write_csv(output_root / "summary.csv", summary)
        write_json(output_root / "summary.json", summary)
        plot_summary(output_root, summary)
        write_report(output_root, summary, args)
        print(
            f"[model-complete] {model_label}: {len(scored)} responses scored",
            flush=True,
        )
    print(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", default="qwen3_8b,llama3_1_8b")
    parser.add_argument("--datasets", default="alpaca,sum,mt_bench")
    parser.add_argument("--eighths", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--skip-dense-eagle3", action="store_true")
    parser.add_argument("--dense-leafs", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-spec-tokens", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=8420)
    parser.add_argument("--health-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--server-shutdown-settle-s", type=float, default=3.0)
    parser.add_argument("--server-start-retries", type=int, default=2)
    parser.add_argument(
        "--calibration-cache-root", type=Path, default=CALIBRATION_ROOT
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.models_list = parse_csv(args.models)
    args.datasets_list = parse_csv(args.datasets)
    args.eighths_list = parse_eighths(args.eighths)
    args.dense_leafs_list = parse_csv(args.dense_leafs)
    if args.smoke:
        args.models_list = args.models_list[:1]
        args.eighths_list = [8]
        args.limit = args.limit or 1
        args.batch_size = min(args.batch_size, 4)
        args.concurrency = min(args.concurrency, 4)
        args.max_num_seqs = min(args.max_num_seqs, 4)
        args.score_batch_size = min(args.score_batch_size, 4)
    if args.output_root is None:
        parent = TEMP_ROOT if args.smoke else RESULTS_FINAL_ROOT
        prefix = (
            "generated_ppl_eagle_data_smoke"
            if args.smoke
            else "generated_ppl_eagle_data"
        )
        args.output_root = parent / f"{prefix}_{timestamp()}"
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
