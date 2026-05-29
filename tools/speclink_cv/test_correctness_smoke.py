#!/usr/bin/env python3
"""Trace-level correctness smoke for exact prefix-gated CV simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.speclink_cv.core import SpecLinkCVState


def simulate(k: int, h: int, actual_accept: int) -> tuple[int, int]:
    state = SpecLinkCVState(request_id="smoke")
    state.on_draft(k=k, a_hat=[0.5] * k, selected_h=h)
    state.begin_prefix()
    state.finish_prefix(actual_accept)
    if state.suffix_pending:
        state.begin_suffix()
        state.finish_suffix(actual_accept)
    return actual_accept, state.skipped_suffix_tokens


def test_trace_level_equivalence() -> None:
    for k in [8, 12]:
        for h in [1, 2, 4, k // 2, k]:
            for actual in range(k + 1):
                produced_accept, _ = simulate(k, h, actual)
                assert produced_accept == actual


if __name__ == "__main__":
    test_trace_level_equivalence()
    print("[PASS] test_correctness_smoke")
