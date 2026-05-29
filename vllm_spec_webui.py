#!/usr/bin/env python3
"""Lightweight vLLM speculative-decoding load visualizer.

Run:

    pip install gradio httpx

    python vllm_spec_webui.py \
      --server-url http://localhost:8000/v1 \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --method-name speclink \
      --num-spec-tokens 8 \
      --host 0.0.0.0 \
      --port 7860

This tool does not start vLLM and does not load local model weights.  It sends
concurrent OpenAI-compatible streaming requests to an already running vLLM
server and visualizes the first response plus an approximate demo TPS.

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
from typing import Any

import httpx


DEFAULT_SERVER_URL = "http://localhost:8000/v1"
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


def make_chat_payload(
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
        "messages": [{"role": "user", "content": prompt}],
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


def display_values(
    *,
    model_name: str,
    method_name: str,
    num_spec_tokens: int,
    batch_size: int,
    state: RunState,
    status: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    elapsed = max(time.time() - state.start_time, 1e-6)
    tokens_per_second = state.total_tokens / elapsed
    if state.errors:
        error_preview = " | ".join(state.errors[:2])
        status = f"{status}; errors: {error_preview}"
    return (
        model_name,
        GPU_NAME,
        method_name,
        str(num_spec_tokens),
        str(batch_size),
        f"{tokens_per_second:.2f}",
        str(state.total_tokens),
        state.first_request_output,
        f"{elapsed:.2f}s",
        status,
    )


async def run_batch_streaming(
    server_url: str,
    model_name: str,
    method_name: str,
    num_spec_tokens: int,
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
    num_spec_tokens = max(0, int(num_spec_tokens))
    method_name = method_name.strip() or "unknown"
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
            method_name=method_name,
            num_spec_tokens=num_spec_tokens,
            batch_size=batch_size,
            state=state,
            status="Model resolution failed",
        )
        clear_active_state(state)
        return

    base_url = normalize_server_url(server_url)
    url = f"{base_url}/chat/completions"
    queue: asyncio.Queue[tuple[str, int, str]] = asyncio.Queue()
    run_id = uuid.uuid4().hex[:10]
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

    yield display_values(
        model_name=model_name,
        method_name=method_name,
        num_spec_tokens=num_spec_tokens,
        batch_size=batch_size,
        state=state,
        status="Running",
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = []
        for request_index in range(batch_size):
            payload = make_chat_payload(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
                ignore_eos=bool(ignore_eos),
                request_id=f"spec-webui-{run_id}-{request_index}",
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

            yield display_values(
                model_name=model_name,
                method_name=method_name,
                num_spec_tokens=num_spec_tokens,
                batch_size=batch_size,
                state=state,
                status=status,
            )

            if done and queue.empty():
                break

        await asyncio.gather(*tasks, return_exceptions=True)

    clear_active_state(state)


def stop_current_run() -> str:
    with ACTIVE_LOCK:
        if ACTIVE_STATE is not None:
            ACTIVE_STATE.stop_event.set()
            return "Stop requested"
    return "No active run"


def clear_outputs() -> tuple[str, str, str, str, str]:
    with ACTIVE_LOCK:
        if ACTIVE_STATE is not None:
            ACTIVE_STATE.stop_event.set()
    return "0.00", "0", "", "0.00s", "Idle"


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
                label="vLLM Server URL",
                value=args.server_url,
                placeholder="http://localhost:8000/v1",
            )
            model_name = gr.Textbox(label="Model Name", value=args.model)
            method_name = gr.Textbox(label="Method Name", value=args.method_name)
            num_spec_tokens = gr.Number(
                label="NUM SPEC TOKENS",
                value=args.num_spec_tokens,
                precision=0,
            )

        with gr.Row():
            batch_size = gr.Number(label="Batch Size", value=32, precision=0)
            max_new_tokens = gr.Number(label="Max New Tokens", value=256, precision=0)
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.0, step=0.05)
            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
            ignore_eos = gr.Checkbox(label="ignore_eos", value=False)

        with gr.Row():
            model_display = gr.Textbox(label="Model Name", interactive=False)
            gpu_display = gr.Textbox(label="GPU Name", value=GPU_NAME, interactive=False)
            method_display = gr.Textbox(label="Method Name", interactive=False)
            spec_display = gr.Textbox(label="NUM SPEC TOKENS", interactive=False)

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
                model_name,
                method_name,
                num_spec_tokens,
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
                method_display,
                spec_display,
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
            outputs=[tps_display, total_tokens, first_output, elapsed, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default="")
    parser.add_argument("--method-name", default="unknown")
    parser.add_argument("--num-spec-tokens", type=int, default=0)
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
