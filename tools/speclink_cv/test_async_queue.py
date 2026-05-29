#!/usr/bin/env python3
"""Unit tests for SpecLink-CV async verification queue."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.speclink_cv.core import SpecLinkCVConfig, VerificationQueue, VerifyChunk
from vllm.speclink_cv import should_wait_for_global_batch_fill


def chunk(request_id: str, benefit: float, age_ms: float = 0.0) -> VerifyChunk:
    return VerifyChunk(
        request_id=request_id,
        chunk_id=request_id + "-p",
        phase="prefix",
        start_draft_pos=1,
        chunk_len=4,
        k=8,
        selected_h=4,
        survival_prob=0.5,
        reject_prob=0.5,
        expected_benefit=benefit,
        arrival_time=time.time() - age_ms / 1000.0,
    )


def test_priority_and_budget() -> None:
    queue = VerificationQueue(SpecLinkCVConfig(max_verify_seqs_per_step=2))
    queue.push(chunk("low", 0.1))
    queue.push(chunk("high", 4.0))
    queue.push(chunk("mid", 1.0))
    ready = queue.pop_ready()
    assert [item.request_id for item in ready] == ["high", "mid"]
    assert len(queue) == 1


def test_age_timeout_prevents_starvation() -> None:
    queue = VerificationQueue(SpecLinkCVConfig(max_verify_seqs_per_step=1, max_queue_wait_ms=2))
    queue.push(chunk("young", 100.0))
    queue.push(chunk("old", 0.0, age_ms=10.0))
    ready = queue.pop_ready()
    assert ready[0].request_id == "old"


def test_global_barrier_does_not_wait_when_running_full() -> None:
    assert should_wait_for_global_batch_fill(
        waiting_reqs=16,
        running_reqs=4,
        max_running_reqs=8,
    )
    assert not should_wait_for_global_batch_fill(
        waiting_reqs=16,
        running_reqs=8,
        max_running_reqs=8,
    )
    assert not should_wait_for_global_batch_fill(
        waiting_reqs=0,
        running_reqs=4,
        max_running_reqs=8,
    )


if __name__ == "__main__":
    test_priority_and_budget()
    test_age_timeout_prevents_starvation()
    test_global_barrier_does_not_wait_when_running_full()
    print("[PASS] test_async_queue")
