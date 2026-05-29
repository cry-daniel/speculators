#!/usr/bin/env python3
"""Unit checks for SpecLink-CV draft-preferred greedy tolerance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_SRC = REPO_ROOT / "vllm"
if str(VLLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLLM_SRC))

from vllm.v1.sample.sampler import Sampler


def main() -> None:
    previous_batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT")
    previous_eps = os.environ.get("SPECLINK_CV_DRAFT_ACCEPT_EPS")
    previous_greedy_eps = os.environ.get("SPECLINK_CV_GREEDY_EPS")
    try:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"

        os.environ.pop("SPECLINK_CV_DRAFT_ACCEPT_EPS", None)
        os.environ.pop("SPECLINK_CV_GREEDY_EPS", None)
        logits = torch.tensor([[10.0, 9.9, 0.0]], dtype=torch.float32)
        preferred = torch.tensor([1], dtype=torch.int64)
        sampled = Sampler.greedy_sample_with_preferred_tokens(logits, preferred)
        assert int(sampled.item()) == 0

        os.environ["SPECLINK_CV_GREEDY_EPS"] = "0.2"
        sampled = Sampler.greedy_sample(logits)
        assert int(sampled.item()) == 0

        logits = torch.tensor([[9.8, 10.0, 9.9]], dtype=torch.float32)
        sampled = Sampler.greedy_sample(logits)
        assert int(sampled.item()) == 0

        os.environ["SPECLINK_CV_GREEDY_EPS"] = "0.05"
        sampled = Sampler.greedy_sample(logits)
        assert int(sampled.item()) == 1

        tied_logits = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
        os.environ["SPECLINK_CV_GREEDY_EPS"] = "0.2"
        os.environ["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = "0"
        sampled = Sampler.greedy_sample_with_preferred_tokens(
            tied_logits, preferred
        )
        assert int(sampled.item()) == 1

        os.environ.pop("SPECLINK_CV_GREEDY_EPS", None)

        logits = torch.tensor([[10.0, 9.9, 0.0]], dtype=torch.float32)
        os.environ["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = "0.2"
        sampled = Sampler.greedy_sample_with_preferred_tokens(logits, preferred)
        assert int(sampled.item()) == 1

        os.environ["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = "0.05"
        sampled = Sampler.greedy_sample_with_preferred_tokens(logits, preferred)
        assert int(sampled.item()) == 0

        os.environ["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = "0"
        sampled = Sampler.greedy_sample_with_preferred_tokens(
            tied_logits, preferred
        )
        assert int(sampled.item()) == 1

        non_tied_logits = torch.tensor([[1.0, 0.999, 0.0]], dtype=torch.float32)
        sampled = Sampler.greedy_sample_with_preferred_tokens(
            non_tied_logits, preferred
        )
        assert int(sampled.item()) == 0

        os.environ["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = "0.01"
        sampled = Sampler.greedy_sample_with_preferred_tokens(
            tied_logits, preferred
        )
        assert int(sampled.item()) == 1
    finally:
        if previous_batch_invariant is None:
            os.environ.pop("VLLM_BATCH_INVARIANT", None)
        else:
            os.environ["VLLM_BATCH_INVARIANT"] = previous_batch_invariant
        if previous_eps is None:
            os.environ.pop("SPECLINK_CV_DRAFT_ACCEPT_EPS", None)
        else:
            os.environ["SPECLINK_CV_DRAFT_ACCEPT_EPS"] = previous_eps
        if previous_greedy_eps is None:
            os.environ.pop("SPECLINK_CV_GREEDY_EPS", None)
        else:
            os.environ["SPECLINK_CV_GREEDY_EPS"] = previous_greedy_eps

    print("[PASS] test_sampler_draft_accept_eps")


if __name__ == "__main__":
    main()
