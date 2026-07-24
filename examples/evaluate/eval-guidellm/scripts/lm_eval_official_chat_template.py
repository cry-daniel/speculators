#!/usr/bin/env python3
"""Run lm-eval while forwarding official tokenizer chat-template arguments.

lm-eval 0.4.12's ``local-completions`` adapter applies the tokenizer's official
chat template when ``--apply_chat_template`` is present, but unlike its local
HF/vLLM adapters it does not forward ``chat_template_args``.  This entry point
keeps the checkpoint's ``tokenizer_config.json -> chat_template`` unchanged and
only forwards the JSON object in ``SPECLINK_LMEVAL_CHAT_TEMPLATE_ARGS`` to
``transformers.PreTrainedTokenizer.apply_chat_template``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from lm_eval.models.openai_completions import LocalCompletionsAPI


_CHAT_TEMPLATE_ARGS: dict[str, Any] = json.loads(
    os.environ.get("SPECLINK_LMEVAL_CHAT_TEMPLATE_ARGS", "{}")
)


def _apply_official_chat_template(
    self: LocalCompletionsAPI,
    chat_history: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **_CHAT_TEMPLATE_ARGS,
        )
    # Preserve the upstream adapter behavior for configurations not used by
    # this experiment.
    from lm_eval.models.api_models import JsonChatStr

    if self.tokenizer_backend == "remote" and self.tokenized_requests:
        return chat_history  # type: ignore[return-value]
    return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))


LocalCompletionsAPI.apply_chat_template = _apply_official_chat_template


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
