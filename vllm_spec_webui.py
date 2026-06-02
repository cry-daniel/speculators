#!/usr/bin/env python3
"""Lightweight vLLM speculative-decoding load visualizer.

Run:

    pip install gradio httpx

    python vllm_spec_webui.py \
      --server-url http://localhost:8000/v1 \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 7860

Smurfs mode connects to an already running vLLM FastDraft server with
SPECLINK_SMURFS_DYNAMIC_ENABLE=1:

    python vllm_spec_webui.py \
      --server-url http://localhost:8078/v1 \
      --backend-mode smurfs \
      --smurfs-k-log /path/to/smurfs_dynamic_k.jsonl

This tool does not start vLLM and does not load local model weights.  It sends
concurrent OpenAI-compatible streaming requests to an already running server and
visualizes the first response plus an approximate demo TPS.  The backend server
must be started with enough draft tokens for both modes.  The UI switches actual
backend behavior by request-id prefix:

- `smurfs`: dynamic K, controlled by `SPECLINK_SMURFS_DYNAMIC_*`.
- `vllm_spec`: fixed K, controlled by `SPECLINK_VLLM_SPEC_FIXED_K`.

Both modes read the `SPECLINK_SMURFS_DYNAMIC_OUT` JSONL file and display the
current K recorded by the backend.  Backend speculative-decoding limits are
configured when the vLLM server starts and cannot be hot-updated from this UI.

Demo TPS is:

    all non-empty streaming deltas from the current batch / elapsed seconds

This is intentionally lightweight.  For paper-quality throughput, use a
tokenizer or vLLM metrics to count exact output tokens and use a fixed-window
serving benchmark instead of this finite batch demo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


DEFAULT_SERVER_URL = "http://localhost:8000/v1"
BACKEND_SMURFS = "smurfs"
BACKEND_VLLM_SPEC = "vllm_spec"
BACKEND_CHOICES = [BACKEND_SMURFS, BACKEND_VLLM_SPEC]
LEGACY_BACKEND_ALIASES = {
    "vLLM Smurfs Dynamic K": BACKEND_SMURFS,
    "vLLM Chat": BACKEND_VLLM_SPEC,
}
SMURFS_METRICS_POLL_SECONDS = 0.5
GPU_NAME = "Unknown GPU"
ACTIVE_STATE: "RunState | None" = None
ACTIVE_LOCK = threading.Lock()


@dataclass
class RunState:
    stop_event: threading.Event = field(default_factory=threading.Event)
    start_time: float = field(default_factory=time.time)
    total_tokens: int = 0
    first_request_output: str = ""
    completed_requests: int = 0
    errors: list[str] = field(default_factory=list)
    current_k: str = "N/A"
    average_k: str = "N/A"
    average_draft_length: str = "N/A"
    k_changes: str = "N/A"
    metrics_error: str = ""
    k_log_offset: int = 0


def clear_active_state(state: RunState) -> None:
    global ACTIVE_STATE
    with ACTIVE_LOCK:
        if ACTIVE_STATE is state:
            ACTIVE_STATE = None


def query_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
            timeout=3,
        )
    except Exception:
        return "Unknown GPU"
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else "Unknown GPU"


def normalize_server_url(server_url: str) -> str:
    return (server_url.strip() or DEFAULT_SERVER_URL).rstrip("/")


def normalize_backend_mode(backend_mode: str) -> str:
    backend_mode = (backend_mode or "").strip()
    backend_mode = LEGACY_BACKEND_ALIASES.get(backend_mode, backend_mode)
    if backend_mode in BACKEND_CHOICES:
        return backend_mode
    return BACKEND_SMURFS


def normalize_smurfs_k_log(path: str) -> str:
    return path.strip()


def request_id_mode_prefix(backend_mode: str) -> str:
    if backend_mode == BACKEND_VLLM_SPEC:
        return "vllm-spec"
    return "smurfs"


async def resolve_model_name(server_url: str, model_name: str) -> str:
    model_name = model_name.strip()
    if model_name:
        return model_name

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{normalize_server_url(server_url)}/models")
        response.raise_for_status()
        data = response.json()
    models = data.get("data") or []
    if models and isinstance(models[0], dict) and models[0].get("id"):
        return str(models[0]["id"])
    raise RuntimeError("model name is empty and /models did not return a model id")


def make_completion_payload(
    *,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    ignore_eos: bool,
    request_id: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "ignore_eos": ignore_eos,
        "stream": True,
        "request_id": request_id,
    }


def extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    text = choice.get("text")
    return text if isinstance(text, str) else ""


async def stream_one_request(
    *,
    request_index: int,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    state: RunState,
    queue: asyncio.Queue[tuple[str, int, str]],
) -> None:
    try:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"HTTP {response.status_code} for request {request_index}: {body[:500]}"
                )
            async for raw_line in response.aiter_lines():
                if state.stop_event.is_set():
                    break
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = extract_delta_text(chunk)
                if text:
                    await queue.put(("delta", request_index, text))
    except Exception as exc:  # noqa: BLE001
        await queue.put(("error", request_index, repr(exc)))
    finally:
        await queue.put(("done", request_index, ""))


def format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "N/A"


def format_k_changes(changes: Any, *, limit: int = 6) -> str:
    if not isinstance(changes, list) or not changes:
        return "N/A"
    tail = changes[-limit:]
    formatted = []
    for item in tail:
        if not isinstance(item, dict):
            continue
        proposal_index = item.get("index", item.get("proposal_index", "?"))
        effective_k = item.get("effective_k", "?")
        formatted.append(f"{proposal_index}:K{effective_k}")
    return ", ".join(formatted) if formatted else "N/A"


def update_smurfs_k_state(
    state: RunState,
    events: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    total_slots = 0
    total_k_slots = 0
    changes = []
    previous_k = None
    current_k = None
    proposal_seen = False
    for event in events:
        event_mode = event.get("mode")
        if event_mode and event_mode != mode:
            continue
        event_type = event.get("event")
        if event_type == "init" and mode == BACKEND_SMURFS:
            k = int(event.get("current_k") or 0)
            if k > 0:
                current_k = k
            continue
        if event_type == "proposal":
            k = int(event.get("effective_k") or 0)
            active_requests = int(event.get("active_requests") or 0)
            if k <= 0:
                continue
            proposal_seen = True
            current_k = k
            if active_requests > 0:
                total_slots += active_requests
                total_k_slots += k * active_requests
            event_index = event.get("proposal_index", event.get("index", "?"))
        elif event_type == "feedback":
            k = int(event.get("new_k") or 0)
            if k <= 0:
                continue
            current_k = k
            event_index = f"fb{event.get('feedback_index', '?')}"
        else:
            continue
        if previous_k is None or previous_k != k:
            changes.append({
                "index": event_index,
                "effective_k": k,
            })
            previous_k = k

    if not proposal_seen:
        state.metrics_error = f"{mode} K log has no proposal events"
        return

    state.current_k = str(current_k) if current_k is not None else "N/A"
    avg_k = total_k_slots / total_slots if total_slots else None
    state.average_k = format_float(avg_k)
    state.average_draft_length = format_float(avg_k)
    state.k_changes = format_k_changes(changes)
    state.metrics_error = ""


def set_k_log_offset(state: RunState, k_log_path: str) -> None:
    if not k_log_path:
        state.k_log_offset = 0
        return
    path = Path(k_log_path)
    state.k_log_offset = path.stat().st_size if path.exists() else 0


async def poll_smurfs_k_log(*, k_log_path: str, state: RunState,
                            mode: str) -> None:
    if not k_log_path:
        state.metrics_error = "Smurfs K log path is empty"
        return
    try:
        path = Path(k_log_path)
        if not path.exists():
            state.metrics_error = f"Smurfs K log not found: {path}"
            return
        events = []
        with path.open("r", encoding="utf-8") as handle:
            if state.k_log_offset:
                handle.seek(state.k_log_offset)
            for line in handle:
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        update_smurfs_k_state(state, events, mode=mode)
    except Exception as exc:  # noqa: BLE001
        state.metrics_error = f"Smurfs K log failed: {exc!r}"


def display_values(
    *,
    model_name: str,
    batch_size: int,
    state: RunState,
    status: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
    elapsed = max(time.time() - state.start_time, 1e-6)
    tokens_per_second = state.total_tokens / elapsed
    if state.errors:
        error_preview = " | ".join(state.errors[:2])
        status = f"{status}; errors: {error_preview}"
    if state.metrics_error:
        status = f"{status}; {state.metrics_error}"
    return (
        model_name,
        GPU_NAME,
        state.current_k,
        state.average_k,
        state.average_draft_length,
        state.k_changes,
        str(batch_size),
        f"{tokens_per_second:.2f}",
        str(state.total_tokens),
        state.first_request_output,
        f"{elapsed:.2f}s",
        status,
    )


async def run_batch_streaming(
    server_url: str,
    backend_mode: str,
    smurfs_k_log: str,
    model_name: str,
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    ignore_eos: bool,
):
    global ACTIVE_STATE
    batch_size = max(1, int(batch_size))
    max_new_tokens = max(1, int(max_new_tokens))
    backend_mode = normalize_backend_mode(backend_mode)
    prompt = prompt or ""

    state = RunState()
    with ACTIVE_LOCK:
        if ACTIVE_STATE is not None:
            ACTIVE_STATE.stop_event.set()
        ACTIVE_STATE = state

    try:
        model_name = await resolve_model_name(server_url, model_name)
    except Exception as exc:  # noqa: BLE001
        state.errors.append(str(exc))
        yield display_values(
            model_name=model_name or "",
            batch_size=batch_size,
            state=state,
            status="Model resolution failed",
        )
        clear_active_state(state)
        return

    base_url = normalize_server_url(server_url)
    url = f"{base_url}/completions"
    k_log_path = normalize_smurfs_k_log(smurfs_k_log)
    if k_log_path:
        set_k_log_offset(state, k_log_path)
    queue: asyncio.Queue[tuple[str, int, str]] = asyncio.Queue()
    run_id = uuid.uuid4().hex[:10]
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

    yield display_values(
        model_name=model_name,
        batch_size=batch_size,
        state=state,
        status="Running",
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        last_metrics_poll = 0.0
        if k_log_path:
            await poll_smurfs_k_log(k_log_path=k_log_path,
                                    state=state,
                                    mode=backend_mode)
        tasks = []
        request_mode_prefix = request_id_mode_prefix(backend_mode)
        for request_index in range(batch_size):
            request_id = (
                f"spec-webui-{request_mode_prefix}-{run_id}-{request_index}"
            )
            payload = make_completion_payload(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
                ignore_eos=bool(ignore_eos),
                request_id=request_id,
            )
            tasks.append(
                asyncio.create_task(
                    stream_one_request(
                        request_index=request_index,
                        client=client,
                        url=url,
                        payload=payload,
                        state=state,
                        queue=queue,
                    )
                )
            )

        while True:
            if k_log_path:
                now = time.monotonic()
                if now - last_metrics_poll >= SMURFS_METRICS_POLL_SECONDS:
                    last_metrics_poll = now
                    await poll_smurfs_k_log(k_log_path=k_log_path,
                                            state=state,
                                            mode=backend_mode)

            drained = 0
            try:
                event_type, request_index, payload = await asyncio.wait_for(
                    queue.get(), timeout=0.08
                )
                events = [(event_type, request_index, payload)]
            except asyncio.TimeoutError:
                events = []

            while True:
                try:
                    events.append(queue.get_nowait())
                    drained += 1
                    if drained >= 256:
                        break
                except asyncio.QueueEmpty:
                    break

            for event_type, request_index, payload in events:
                if event_type == "delta":
                    # Real-time demo counting: every non-empty streaming delta is
                    # treated as one token. Use tokenizer/vLLM metrics for exact
                    # paper-quality token accounting.
                    state.total_tokens += 1
                    if request_index == 0:
                        state.first_request_output += payload
                elif event_type == "error":
                    state.errors.append(payload)
                elif event_type == "done":
                    state.completed_requests += 1

            done = all(task.done() for task in tasks)
            stopped = state.stop_event.is_set()
            status = (
                "Stopped"
                if stopped
                else f"Running ({state.completed_requests}/{batch_size} done)"
            )
            if done:
                status = "Finished" if not stopped else "Stopped"
                if k_log_path:
                    await poll_smurfs_k_log(k_log_path=k_log_path,
                                            state=state,
                                            mode=backend_mode)

            yield display_values(
                model_name=model_name,
                batch_size=batch_size,
                state=state,
                status=status,
            )

            if done and queue.empty():
                break

        await asyncio.gather(*tasks, return_exceptions=True)
        if k_log_path:
            await poll_smurfs_k_log(k_log_path=k_log_path,
                                    state=state,
                                    mode=backend_mode)

    clear_active_state(state)


def stop_current_run() -> str:
    with ACTIVE_LOCK:
        if ACTIVE_STATE is not None:
            ACTIVE_STATE.stop_event.set()
            return "Stop requested"
    return "No active run"


def clear_outputs() -> tuple[str, str, str, str, str, str, str, str, str]:
    with ACTIVE_LOCK:
        if ACTIVE_STATE is not None:
            ACTIVE_STATE.stop_event.set()
    return "N/A", "N/A", "N/A", "N/A", "0.00", "0", "", "0.00s", "Idle"


def build_ui(args: argparse.Namespace) -> gr.Blocks:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gradio is required for the WebUI. Install dependencies with: "
            "pip install gradio httpx"
        ) from exc

    with gr.Blocks(title="vLLM Speculative Decoding Visualizer") as demo:
        gr.Markdown("# vLLM Speculative Decoding Visualizer")

        with gr.Row():
            server_url = gr.Textbox(
                label="Server URL",
                value=args.server_url,
                placeholder="http://localhost:8000/v1",
            )
            backend_mode = gr.Dropdown(
                label="Backend",
                choices=BACKEND_CHOICES,
                value=normalize_backend_mode(args.backend_mode),
            )
            smurfs_k_log = gr.Textbox(
                label="Smurfs K Log",
                value=args.smurfs_k_log,
                placeholder="/path/to/smurfs_dynamic_k.jsonl",
            )
            model_name = gr.Textbox(label="Model Name", value=args.model)

        with gr.Row():
            batch_size = gr.Number(label="Batch Size", value=32, precision=0)
            max_new_tokens = gr.Number(label="Max New Tokens", value=256, precision=0)
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.0, step=0.05)
            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
            ignore_eos = gr.Checkbox(label="ignore_eos", value=False)

        with gr.Row():
            model_display = gr.Textbox(label="Model Name", interactive=False)
            gpu_display = gr.Textbox(label="GPU Name", value=GPU_NAME, interactive=False)

        with gr.Row():
            current_k_display = gr.Textbox(label="Current K", value="N/A", interactive=False)
            average_k_display = gr.Textbox(label="Average K", value="N/A", interactive=False)
            average_draft_display = gr.Textbox(label="Average Draft Length", value="N/A", interactive=False)
            k_changes_display = gr.Textbox(label="K Changes", value="N/A", interactive=False)

        with gr.Row():
            batch_display = gr.Textbox(label="Batch Size", interactive=False)
            tps_display = gr.Textbox(label="Current Tokens/s", value="0.00", interactive=False)
            total_tokens = gr.Textbox(label="Total Generated Tokens", value="0", interactive=False)
            elapsed = gr.Textbox(label="Elapsed Time", value="0.00s", interactive=False)

        prompt = gr.Textbox(label="Prompt", lines=7, value="Explain speculative decoding in one paragraph.")
        first_output = gr.Textbox(label="First Request Output", lines=14)
        status = gr.Textbox(label="Status", value="Idle", interactive=False)

        with gr.Row():
            run_button = gr.Button("Run", variant="primary")
            stop_button = gr.Button("Stop")
            clear_button = gr.Button("Clear")

        run_event = run_button.click(
            run_batch_streaming,
            inputs=[
                server_url,
                backend_mode,
                smurfs_k_log,
                model_name,
                batch_size,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                ignore_eos,
            ],
            outputs=[
                model_display,
                gpu_display,
                current_k_display,
                average_k_display,
                average_draft_display,
                k_changes_display,
                batch_display,
                tps_display,
                total_tokens,
                first_output,
                elapsed,
                status,
            ],
        )
        stop_button.click(stop_current_run, outputs=[status])
        clear_button.click(
            clear_outputs,
            outputs=[
                current_k_display,
                average_k_display,
                average_draft_display,
                k_changes_display,
                tps_display,
                total_tokens,
                first_output,
                elapsed,
                status,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--backend-mode",
                        default=BACKEND_SMURFS)
    parser.add_argument("--smurfs-k-log", default="")
    parser.add_argument("--model", default="")
    # Deprecated no-op arguments kept so older launch commands still work.
    parser.add_argument("--method-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--num-spec-tokens",
                        type=int,
                        default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    global GPU_NAME
    args = parse_args()
    GPU_NAME = query_gpu_name()
    demo = build_ui(args)
    demo.queue().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
