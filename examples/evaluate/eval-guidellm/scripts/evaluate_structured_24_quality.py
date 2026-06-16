#!/usr/bin/env python3
"""Run the structured 2:4 masked-weight quality experiment.

This is the user-facing entry point requested by TODO.md. The implementation
is shared with ``residual_24_feasibility.py quality`` so there is only one
quality-evaluation code path to maintain.

Smoke:

  cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators
  conda run -n spec python -u \
    examples/evaluate/eval-guidellm/scripts/evaluate_structured_24_quality.py \
    --smoke \
    --output-root examples/evaluate/eval-guidellm/results/structured_24_quality_smoke

Full:

  conda run -n spec python -u \
    examples/evaluate/eval-guidellm/scripts/evaluate_structured_24_quality.py \
    --models qwen3_8b,llama3_1_8b \
    --mask-scopes none,attn,ffn,all \
    --datasets gsm8k,humaneval,math_reasoning,mtbench,dolly \
    --gsm8k-num-examples 128 \
    --humaneval-num-examples 64 \
    --math-num-examples 128 \
    --dolly-num-examples 128 \
    --mtbench-num-examples 80 \
    --dtype bf16 \
    --output-root examples/evaluate/eval-guidellm/results/structured_24_quality_full
"""

from __future__ import annotations

import sys

from residual_24_feasibility import apply_24_mask_to_model, build_parser

__all__ = ["apply_24_mask_to_model", "main"]


def main() -> None:
    # Reuse the consolidated quality subcommand while preserving the TODO.md
    # command shape: this wrapper accepts quality options directly.
    parser = build_parser()
    args = parser.parse_args(["quality", *sys.argv[1:]])
    args.func(args)


if __name__ == "__main__":
    main()
