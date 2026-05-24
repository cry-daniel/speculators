#!/usr/bin/env python3
"""Patch the installed vLLM package with SpecLink plan-only instrumentation."""

from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

import vllm


MARKER = "# SPECLINK_PLAN_ONLY_PATCH"


SCHEDULER_HELPER = r'''

# SPECLINK_PLAN_ONLY_PATCH_BEGIN
class _SpeclinkRuntime:
    """Lightweight SpecLink plan-only tracing for speculative verification.

    This runtime intentionally does not alter token generation. Candidate KV
    blocks are a deterministic position-recency proxy unless a future patch
    wires in draft/target attention scores.
    """

    def __init__(
        self,
        block_size: int,
        num_spec_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        self.enabled = os.getenv("SPECLINK_ENABLE", "0") == "1"
        self.mode = os.getenv("SPECLINK_MODE", "off")
        if not self.enabled or self.mode == "off":
            self.enabled = False
        self.layout = os.getenv("SPECLINK_LAYOUT", "speclink_prob")
        self.method = os.getenv("SPECLINK_METHOD", "speclink")
        self.trace_out = os.getenv("SPECLINK_TRACE_OUT", "")
        self.profile_enabled = os.getenv("SPECLINK_PROFILE", "0") == "1"
        self.profile_out = os.getenv("SPECLINK_PROFILE_OUT", "")
        self.block_size = int(os.getenv("SPECLINK_BLOCK_SIZE", str(block_size)))
        self.topk_per_token = int(os.getenv("SPECLINK_TOPK_PER_TOKEN", "32"))
        self.shared_budget = int(os.getenv("SPECLINK_SHARED_BUDGET", "32"))
        self.private_min = int(os.getenv("SPECLINK_PRIVATE_MIN", "0"))
        self.private_max = int(os.getenv("SPECLINK_PRIVATE_MAX", "16"))
        self.alpha = float(os.getenv("SPECLINK_ALPHA", "8"))
        self.beta = float(os.getenv("SPECLINK_BETA", "8"))
        self.lambda_risk = float(os.getenv("SPECLINK_LAMBDA_RISK", "1"))
        self.risk_threshold = float(os.getenv("SPECLINK_RISK_THRESHOLD", "0.35"))
        self.acceptance_decay = float(os.getenv("SPECLINK_ACCEPTANCE_DECAY", "0.9"))
        self.num_spec_tokens = num_spec_tokens
        self.accept_probs = [0.7] * max(num_spec_tokens, 1)
        self.step_by_req: dict[str, int] = defaultdict(int)

        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        num_heads = getattr(hf_config, "num_attention_heads", 32)
        hidden_size = getattr(hf_config, "hidden_size", 4096)
        default_head_dim = hidden_size // num_heads if num_heads else 128
        self.num_layers = int(
            os.getenv(
                "SPECLINK_NUM_LAYERS",
                str(getattr(hf_config, "num_hidden_layers", 36)),
            )
        )
        self.num_kv_heads = int(
            os.getenv(
                "SPECLINK_NUM_KV_HEADS",
                str(getattr(hf_config, "num_key_value_heads", num_heads)),
            )
        )
        self.head_dim = int(os.getenv("SPECLINK_HEAD_DIM", str(default_head_dim)))
        self.bytes_per_elem = int(os.getenv("SPECLINK_BYTES_PER_ELEM", "2"))

    def observe(
        self,
        request: Request,
        scheduled_spec_token_ids: list[int],
        generated_token_ids: list[int],
        num_accepted: int,
    ) -> None:
        if not self.enabled or self.mode not in {"trace", "plan_only", "sparse_kernel"}:
            return

        planner_start = time.perf_counter()
        num_draft_tokens = len(scheduled_spec_token_ids)
        self._update_accept_probs(num_draft_tokens, num_accepted)
        accept_probs = self.accept_probs[:num_draft_tokens]
        candidates = self._make_proxy_candidates(request, scheduled_spec_token_ids)

        plan = None
        planner_error = None
        try:
            from speculators.speclink.planner import SpeclinkConfig, SpeclinkPlanner

            config = SpeclinkConfig(
                layout=self.layout,
                topk_per_token=self.topk_per_token,
                shared_budget=self.shared_budget,
                private_min=self.private_min,
                private_max=self.private_max,
                alpha=self.alpha,
                beta=self.beta,
                lambda_risk=self.lambda_risk,
                risk_threshold=self.risk_threshold,
                fallback_enabled=self.layout.endswith("fallback"),
            )
            plan = SpeclinkPlanner(config).plan(candidates, accept_probs)
        except Exception as exc:  # noqa: BLE001
            planner_error = repr(exc)

        planner_ms = (time.perf_counter() - planner_start) * 1000.0
        req_id = request.request_id
        step = self.step_by_req[req_id]
        self.step_by_req[req_id] += 1

        prompt_len = request.num_prompt_tokens
        decode_len = request.num_output_tokens
        union_blocks = list(plan.union_blocks) if plan is not None else []
        estimated_hbm_bytes = self._estimate_hbm_bytes(len(union_blocks))
        trace = {
            "ts": time.time(),
            "request_id": req_id,
            "step": step,
            "method": self.method,
            "mode": self.mode,
            "layout": self.layout,
            "num_spec_tokens": num_draft_tokens,
            "block_size": self.block_size,
            "prompt_len": prompt_len,
            "decode_len": decode_len,
            "draft_tokens": list(scheduled_spec_token_ids),
            "generated_token_ids": list(generated_token_ids),
            "accepted_prefix_len": num_accepted,
            "accept_probs": accept_probs,
            "rho": list(plan.rho) if plan is not None else [],
            "risk": list(plan.risk) if plan is not None else [],
            "shared_blocks": list(plan.shared_blocks) if plan is not None else [],
            "residual_counts": (
                [len(blocks) for blocks in plan.residual_blocks_per_token]
                if plan is not None
                else []
            ),
            "union_blocks": len(union_blocks),
            "union_block_ids": union_blocks,
            "mean_blocks_per_token": (
                plan.stats.get("mean_blocks_per_token") if plan is not None else None
            ),
            "estimated_hbm_bytes": estimated_hbm_bytes,
            "fallback_tokens": list(plan.fallback_tokens) if plan is not None else [],
            "planner_ms": planner_ms,
            "candidate_source": "position_recency_proxy",
            "candidates": candidates,
            "planner_error": planner_error,
        }
        self._append_jsonl(self.trace_out, trace)
        if self.profile_enabled:
            self._append_jsonl(
                self.profile_out,
                {
                    "ts": trace["ts"],
                    "request_id": req_id,
                    "method": self.method,
                    "num_spec_tokens": num_draft_tokens,
                    "step": "speclink_plan",
                    "speclink_planner_ms": planner_ms,
                    "num_draft_tokens": num_draft_tokens,
                    "num_accepted_tokens": num_accepted,
                    "prompt_len": prompt_len,
                    "decode_len": decode_len,
                },
            )

    def _update_accept_probs(self, num_draft_tokens: int, num_accepted: int) -> None:
        if num_draft_tokens > len(self.accept_probs):
            self.accept_probs.extend([0.7] * (num_draft_tokens - len(self.accept_probs)))
        decay = self.acceptance_decay
        for pos in range(num_draft_tokens):
            observed = 1.0 if pos < num_accepted else 0.0
            self.accept_probs[pos] = decay * self.accept_probs[pos] + (
                1.0 - decay
            ) * observed

    def _make_proxy_candidates(
        self,
        request: Request,
        scheduled_spec_token_ids: list[int],
    ) -> list[list[dict[str, float]]]:
        context_tokens = max(request.num_tokens, request.num_computed_tokens, 1)
        num_blocks = max(1, (context_tokens + self.block_size - 1) // self.block_size)
        keep = max(self.topk_per_token, self.shared_budget + self.private_max + 8)
        keep = max(1, min(num_blocks, keep))
        candidates: list[list[dict[str, float]]] = []
        for pos, token_id in enumerate(scheduled_spec_token_ids):
            center = max(0, num_blocks - 1 - (pos % max(num_blocks, 1)))
            token_factor = 1.0 + ((int(token_id) % 17) / 10000.0)
            scored = []
            for block in range(num_blocks):
                distance = abs(block - center)
                recency = block / max(num_blocks - 1, 1)
                score = token_factor * ((1.0 / (1.0 + distance)) + 0.001 * recency)
                scored.append({"block": block, "score": score})
            scored.sort(key=lambda item: (-item["score"], item["block"]))
            total = sum(item["score"] for item in scored[:keep]) or 1.0
            candidates.append(
                [
                    {"block": int(item["block"]), "score": float(item["score"] / total)}
                    for item in scored[:keep]
                ]
            )
        return candidates

    def _estimate_hbm_bytes(self, union_blocks: int) -> int:
        return (
            self.num_layers
            * self.num_kv_heads
            * union_blocks
            * self.block_size
            * self.head_dim
            * self.bytes_per_elem
            * 2
        )

    @staticmethod
    def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write SPECLINK trace/profile event to %s", path)


# SPECLINK_PLAN_ONLY_PATCH_END
'''


CORE_HELPER = r'''

# SPECLINK_PLAN_ONLY_PATCH_BEGIN
def _speclink_write_profile_event(
    *,
    vllm_config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_step_ms: float,
    model_wait_ms: float,
    engine_update_ms: float,
    total_engine_step_ms: float,
) -> None:
    if os.getenv("SPECLINK_PROFILE", "0") != "1":
        return
    profile_out = os.getenv("SPECLINK_PROFILE_OUT", "")
    if not profile_out:
        return
    speculative_config = vllm_config.speculative_config
    num_spec_tokens = (
        speculative_config.num_speculative_tokens if speculative_config else 0
    )
    event = {
        "ts": time.time(),
        "method": os.getenv("SPECLINK_METHOD", "speclink"),
        "num_spec_tokens": num_spec_tokens,
        "step": "engine_step",
        "scheduler_step_ms": scheduler_step_ms,
        "target_verify_forward_ms": model_wait_ms,
        "engine_update_ms": engine_update_ms,
        "total_engine_step_ms": total_engine_step_ms,
        "request_ids": list(scheduler_output.num_scheduled_tokens.keys()),
        "num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
    }
    try:
        os.makedirs(os.path.dirname(profile_out), exist_ok=True)
        with open(profile_out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write SPECLINK profile event to %s", profile_out)


# SPECLINK_PLAN_ONLY_PATCH_END
'''


GPU_MODEL_RUNNER_HELPER = r'''

# SPECLINK_PLAN_ONLY_PATCH_BEGIN
def _speclink_write_worker_profile_event(**payload: Any) -> None:
    if os.getenv("SPECLINK_PROFILE", "0") != "1":
        return
    profile_out = os.getenv("SPECLINK_PROFILE_OUT", "")
    if not profile_out:
        return
    payload.setdefault("ts", time.time())
    payload.setdefault("method", os.getenv("SPECLINK_METHOD", "unknown"))
    try:
        payload.setdefault("num_spec_tokens", getattr(getattr(payload.get("_self"), "speculative_config", None), "num_speculative_tokens", None))
        payload.pop("_self", None)
        os.makedirs(os.path.dirname(profile_out), exist_ok=True)
        with open(profile_out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to write SPECLINK worker profile event to %s", profile_out)


# SPECLINK_PLAN_ONLY_PATCH_END
'''


def backup(path: Path, timestamp: str) -> None:
    dst = path.with_name(f"{path.name}.bak-speclink-{timestamp}")
    if not dst.exists():
        shutil.copy2(path, dst)
        print(f"backed up {path} -> {dst}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"cannot find insertion point: {label}")
    return text.replace(old, new, 1)


def patch_scheduler(path: Path, timestamp: str) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{path} already contains {MARKER}")
        return
    backup(path, timestamp)
    text = replace_once(
        text,
        "import itertools\nimport time\n",
        "import itertools\nimport json\nimport os\nimport time\n",
        "scheduler imports",
    )
    text = replace_once(
        text,
        "\n\nclass Scheduler(SchedulerInterface):\n",
        SCHEDULER_HELPER + "\n\nclass Scheduler(SchedulerInterface):\n",
        "scheduler helper",
    )
    text = replace_once(
        text,
        """            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
""",
        """            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        self.speclink_runtime = _SpeclinkRuntime(
            block_size=block_size,
            num_spec_tokens=self.num_spec_tokens,
            vllm_config=vllm_config,
        )

        # Create the KV cache manager.
""",
        "scheduler runtime init",
    )
    text = replace_once(
        text,
        """                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
""",
        """                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )
                self.speclink_runtime.observe(
                    request=request,
                    scheduled_spec_token_ids=scheduled_spec_token_ids,
                    generated_token_ids=generated_token_ids,
                    num_accepted=num_accepted,
                )

            # Free encoder inputs only after the step has actually executed.
""",
        "scheduler observe call",
    )
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def patch_core(path: Path, timestamp: str) -> None:
    text = path.read_text(encoding="utf-8")
    changed = False
    if MARKER not in text:
        backup(path, timestamp)
        text = replace_once(
            text,
            "import gc\nimport os\n",
            "import gc\nimport json\nimport os\n",
            "core imports",
        )
        text = replace_once(
            text,
            "\n\nclass EngineCore:\n",
            CORE_HELPER + "\n\nclass EngineCore:\n",
            "core helper",
        )
        text = replace_once(
            text,
            """        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
""",
            """        speclink_step_start = time.perf_counter()
        speclink_scheduler_start = speclink_step_start
        scheduler_output = self.scheduler.schedule()
        speclink_scheduler_ms = (
            time.perf_counter() - speclink_scheduler_start
        ) * 1000.0
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        speclink_model_wait_start = time.perf_counter()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
        speclink_model_wait_ms = (
            time.perf_counter() - speclink_model_wait_start
        ) * 1000.0

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        speclink_update_start = time.perf_counter()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        speclink_update_ms = (time.perf_counter() - speclink_update_start) * 1000.0
        speclink_total_ms = (time.perf_counter() - speclink_step_start) * 1000.0
        _speclink_write_profile_event(
            vllm_config=self.vllm_config,
            scheduler_output=scheduler_output,
            scheduler_step_ms=speclink_scheduler_ms,
            model_wait_ms=speclink_model_wait_ms,
            engine_update_ms=speclink_update_ms,
            total_engine_step_ms=speclink_total_ms,
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
""",
            "core step timings",
        )
        changed = True
    else:
        print(f"{path} already contains {MARKER}")

    if "speclink_batch_item = batch_queue.pop()" not in text:
        backup(path, timestamp)
        text = replace_once(
            text,
            """        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
""",
            """        if self.scheduler.has_requests():
            speclink_scheduler_start = time.perf_counter()
            scheduler_output = self.scheduler.schedule()
            speclink_scheduler_ms = (
                time.perf_counter() - speclink_scheduler_start
            ) * 1000.0
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
""",
            "core batch queue schedule timings",
        )
        text = replace_once(
            text,
            """            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True
""",
            """            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                speclink_enqueue_ts = time.perf_counter()
                batch_queue.appendleft(
                    (
                        future,
                        scheduler_output,
                        exec_future,
                        speclink_scheduler_ms,
                        speclink_enqueue_ts,
                    )
                )
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True
""",
            "core batch queue first append",
        )
        text = replace_once(
            text,
            """        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
""",
            """        # Block until the next result is available.
        speclink_batch_item = batch_queue.pop()
        future, scheduler_output, exec_model_fut = speclink_batch_item[:3]
        speclink_scheduler_ms = (
            float(speclink_batch_item[3]) if len(speclink_batch_item) > 3 else 0.0
        )
        speclink_model_wait_start = time.perf_counter()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
        speclink_model_wait_ms = (
            time.perf_counter() - speclink_model_wait_start
        ) * 1000.0

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        speclink_update_start = time.perf_counter()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        speclink_update_ms = (time.perf_counter() - speclink_update_start) * 1000.0
        speclink_total_ms = (
            speclink_scheduler_ms + speclink_model_wait_ms + speclink_update_ms
        )
        _speclink_write_profile_event(
            vllm_config=self.vllm_config,
            scheduler_output=scheduler_output,
            scheduler_step_ms=speclink_scheduler_ms,
            model_wait_ms=speclink_model_wait_ms,
            engine_update_ms=speclink_update_ms,
            total_engine_step_ms=speclink_total_ms,
        )
""",
            "core batch queue pop/update timings",
        )
        text = replace_once(
            text,
            """            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
""",
            """            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            speclink_enqueue_ts = time.perf_counter()
            batch_queue.appendleft(
                (
                    future,
                    deferred_scheduler_output,
                    exec_future,
                    speclink_scheduler_ms,
                    speclink_enqueue_ts,
                )
            )
""",
            "core batch queue deferred append",
        )
        changed = True
    else:
        print(f"{path} already contains batch queue SPECLINK timings")

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")


def patch_gpu_model_runner(path: Path, timestamp: str) -> None:
    text = path.read_text(encoding="utf-8")
    changed = False
    if "_speclink_write_worker_profile_event" not in text:
        backup(path, timestamp)
        text = replace_once(
            text,
            "import functools\nimport gc\n",
            "import functools\nimport gc\nimport json\nimport os\n",
            "gpu_model_runner imports",
        )
        text = replace_once(
            text,
            "\n\nclass GPUModelRunner",
            GPU_MODEL_RUNNER_HELPER + "\n\nclass GPUModelRunner",
            "gpu_model_runner helper",
        )
        changed = True
    else:
        print(f"{path} already contains worker SPECLINK helper")

    if "speclink_sampler_start = time.perf_counter()" not in text:
        backup(path, timestamp)
        text = replace_once(
            text,
            """        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
""",
            """        speclink_sampler_start = time.perf_counter()
        if spec_decode_metadata is None:
            output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
            _speclink_write_worker_profile_event(
                _self=self,
                step="sampler",
                accept_reject_sampler_ms=(time.perf_counter() - speclink_sampler_start) * 1000.0,
            )
            return output
""",
            "plain sampler timing",
        )
        text = replace_once(
            text,
            """        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output
""",
            """        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        _speclink_write_worker_profile_event(
            _self=self,
            step="accept_reject_sampler",
            accept_reject_sampler_ms=(time.perf_counter() - speclink_sampler_start) * 1000.0,
            num_draft_tokens=sum(spec_decode_metadata.num_draft_tokens)
            if spec_decode_metadata is not None
            else None,
        )
        return sampler_output
""",
            "rejection sampler timing",
        )
        changed = True
    else:
        print(f"{path} already contains sampler SPECLINK timings")

    if "speclink_draft_start = time.perf_counter()" not in text:
        backup(path, timestamp)
        text = replace_once(
            text,
            """        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)
""",
            """        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            speclink_draft_start = time.perf_counter()
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)
            draft_count = None
            try:
                if hasattr(self._draft_token_ids, "numel"):
                    draft_count = int(self._draft_token_ids.numel())
                elif self._draft_token_ids is not None:
                    draft_count = sum(len(ids) for ids in self._draft_token_ids)
            except Exception:  # noqa: BLE001
                draft_count = None
            _speclink_write_worker_profile_event(
                _self=self,
                step="draft_forward",
                draft_forward_ms=(time.perf_counter() - speclink_draft_start) * 1000.0,
                num_draft_tokens=draft_count,
            )
""",
            "draft proposer timing",
        )
        changed = True
    else:
        print(f"{path} already contains draft SPECLINK timings")

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")


def main() -> None:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    vllm_root = Path(vllm.__file__).resolve().parent
    patch_scheduler(vllm_root / "v1" / "core" / "sched" / "scheduler.py", timestamp)
    patch_core(vllm_root / "v1" / "engine" / "core.py", timestamp)
    patch_gpu_model_runner(vllm_root / "v1" / "worker" / "gpu_model_runner.py", timestamp)


if __name__ == "__main__":
    main()
