#!/usr/bin/env python3
"""Unit tests for SpecLink-CV roofline-aware packing proxy."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.speclink_cv.core import RooflinePacker, SpecLinkCVConfig, VerifyChunk


def make_chunk(idx: int, chunk_len: int = 4, age_ms: float = 0.0) -> VerifyChunk:
    return VerifyChunk(
        request_id=f"r{idx}",
        chunk_id=f"c{idx}",
        phase="prefix",
        start_draft_pos=1,
        chunk_len=chunk_len,
        k=8,
        selected_h=chunk_len,
        survival_prob=0.5,
        reject_prob=0.5,
        expected_benefit=1.0,
        arrival_time=time.time() - age_ms / 1000.0,
    )


def test_waits_when_underutilized() -> None:
    cfg = SpecLinkCVConfig(
        roofline_packing=True,
        max_verify_tokens_per_step=64,
        max_verify_seqs_per_step=16,
        util_threshold=0.5,
        max_queue_wait_ms=100,
    )
    selected, _, reason = RooflinePacker(cfg).select([make_chunk(0, 1)])
    assert selected == []
    assert reason == "wait_for_utilization"


def test_age_timeout_schedules_underfilled() -> None:
    cfg = SpecLinkCVConfig(
        roofline_packing=True,
        max_verify_tokens_per_step=64,
        max_verify_seqs_per_step=16,
        util_threshold=0.9,
        max_queue_wait_ms=1,
    )
    selected, _, reason = RooflinePacker(cfg).select([make_chunk(0, 1, age_ms=20)])
    assert len(selected) == 1
    assert reason == "age_timeout"


if __name__ == "__main__":
    test_waits_when_underutilized()
    test_age_timeout_schedules_underfilled()
    print("[PASS] test_roofline_packing")
