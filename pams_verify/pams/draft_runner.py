from __future__ import annotations


def status() -> dict:
    return {
        "implemented": "synthetic_and_future_hf_runner",
        "online_vllm_draft_logits": False,
        "notes": [
            "The current runnable path uses deterministic synthetic draft traces.",
            "A real HF/vLLM draft runner should replace this when GPU access is available.",
        ],
    }

