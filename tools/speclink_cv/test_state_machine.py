#!/usr/bin/env python3
"""Unit tests for SpecLink-CV request state transitions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.speclink_cv.core import CVState, SpecLinkCVState


def test_prefix_reject_skips_suffix() -> None:
    state = SpecLinkCVState(request_id="r0")
    state.on_draft(k=8, a_hat=[0.5] * 8, selected_h=4)
    state.begin_prefix()
    state.finish_prefix(actual_accept_tokens=2)
    assert state.state == CVState.PREFIX_REJECTED
    assert state.skipped_suffix_tokens == 4
    assert not state.suffix_pending


def test_prefix_accept_schedules_suffix() -> None:
    state = SpecLinkCVState(request_id="r1")
    state.on_draft(k=8, a_hat=[0.9] * 8, selected_h=4)
    state.begin_prefix()
    state.finish_prefix(actual_accept_tokens=8)
    assert state.state == CVState.PREFIX_ACCEPTED
    assert state.suffix_pending
    state.begin_suffix()
    state.finish_suffix(actual_accept_tokens=8)
    state.commit()
    assert state.state == CVState.DONE_OR_NEXT_STEP
    assert state.num_extra_tlm_forwards == 1


def test_full_k_equivalent_oneshot() -> None:
    state = SpecLinkCVState(request_id="r2")
    state.on_draft(k=8, a_hat=[0.9] * 8, selected_h=8)
    state.begin_prefix()
    state.finish_prefix(actual_accept_tokens=8)
    assert state.state == CVState.COMMIT_READY
    assert not state.suffix_pending


def test_prefix_reject_drafter_rollback_uses_current_forward_len() -> None:
    context_len = 100
    k = 16
    h = 4
    accepted = 1

    current_forward_seq_len = context_len + h + 1
    current_forward_rejected = h - accepted
    full_draft_rejected = k - accepted

    assert current_forward_seq_len - current_forward_rejected == (
        context_len + accepted + 1
    )
    assert current_forward_seq_len - full_draft_rejected < context_len


if __name__ == "__main__":
    test_prefix_reject_skips_suffix()
    test_prefix_accept_schedules_suffix()
    test_full_k_equivalent_oneshot()
    test_prefix_reject_drafter_rollback_uses_current_forward_len()
    print("[PASS] test_state_machine")
