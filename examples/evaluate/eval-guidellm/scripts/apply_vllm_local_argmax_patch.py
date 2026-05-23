#!/usr/bin/env python3
"""Apply the local-argmax EAGLE3 vLLM patch used by the benchmarks.

This helper is intentionally version-targeted for the local vLLM 0.20.0
environment used by the Qwen3-8B EAGLE3/P-EAGLE experiments. It makes the
patch reproducible without committing files from site-packages into this repo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


TOP_TOKEN_METHODS = '''\
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy draft tokens without materializing target-vocab logits."""
        logits = self.logits_processor(self.lm_head, hidden_states)
        assert logits is not None
        draft_token_ids = logits.argmax(dim=-1)
        if (
            self.draft_id_to_target_id is None
            or self.draft_id_to_target_id_is_identity
        ):
            return draft_token_ids
        return draft_token_ids + self.draft_id_to_target_id[draft_token_ids]

    def get_top_k_tokens(
        self,
        hidden_states: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Top-k draft tokens without materializing target-vocab logits."""
        logits = self.logits_processor(self.lm_head, hidden_states)
        assert logits is not None
        draft_token_ids = torch.topk(logits, k, dim=-1).indices
        if (
            self.draft_id_to_target_id is None
            or self.draft_id_to_target_id_is_identity
        ):
            return draft_token_ids
        return draft_token_ids + self.draft_id_to_target_id[draft_token_ids]

'''


def vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not import vLLM from the active Python environment.")
    return Path(spec.origin).resolve().parent


def patch_file(path: Path, transform) -> bool:
    old = path.read_text()
    new = transform(old)
    if new == old:
        return False
    path.write_text(new)
    return True


def patch_llama_eagle3(text: str) -> str:
    if "class Eagle3LlamaForCausalLM(LlamaForCausalLM):\n" not in text:
        raise SystemExit("Could not find Eagle3LlamaForCausalLM class.")

    if "supports_mapped_top_tokens = True" not in text:
        text = text.replace(
            "class Eagle3LlamaForCausalLM(LlamaForCausalLM):\n",
            "class Eagle3LlamaForCausalLM(LlamaForCausalLM):\n"
            "    supports_mapped_top_tokens = True\n\n",
            1,
        )

    if "def get_top_tokens(" not in text:
        marker = "    def combine_hidden_states(\n"
        if marker not in text:
            raise SystemExit("Could not find combine_hidden_states insertion point.")
        text = text.replace(marker, TOP_TOKEN_METHODS + marker, 1)

    return text


def patch_llm_base_proposer(text: str) -> str:
    old = '''\
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        return self.model.compute_logits(hidden_states).argmax(dim=-1)
'''
    new = '''\
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return self.model.compute_logits(hidden_states).argmax(dim=-1)
'''
    if old in text and "return self.model.get_top_tokens(hidden_states)" not in text:
        text = text.replace(old, new, 1)

    warning_guard = (
        '                and not getattr(self.model, "supports_mapped_top_tokens", False)\n'
    )
    if warning_guard not in text and "supports_mapped_top_tokens" not in text:
        old_guard = '''\
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
'''
        new_guard = '''\
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
                and not getattr(self.model, "supports_mapped_top_tokens", False)
'''
        if old_guard in text:
            text = text.replace(old_guard, new_guard, 1)

    return text


def main() -> None:
    root = vllm_root()
    targets = [
        (
            root / "model_executor/models/llama_eagle3.py",
            patch_llama_eagle3,
        ),
        (
            root / "v1/spec_decode/llm_base_proposer.py",
            patch_llm_base_proposer,
        ),
    ]

    changed = []
    for path, transform in targets:
        if not path.exists():
            raise SystemExit(f"Missing expected vLLM file: {path}")
        if patch_file(path, transform):
            changed.append(path)

    if changed:
        for path in changed:
            print(f"patched {path}")
    else:
        print("vLLM local-argmax patch already applied")


if __name__ == "__main__":
    main()
