#!/usr/bin/env python3
"""Unit tests for SpecLink-CV chunk decisions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.speclink_cv.core import SpecLinkCVConfig, choose_chunk_size


def test_fixed_half() -> None:
    cfg = SpecLinkCVConfig(confidence_sizing=False)
    assert choose_chunk_size(k=8, a_hat=[0.9] * 8, config=cfg).selected_h == 4
    assert choose_chunk_size(k=12, a_hat=[0.9] * 12, config=cfg).selected_h == 6


def test_confidence_high_falls_back_full() -> None:
    cfg = SpecLinkCVConfig(confidence_sizing=True, min_benefit=0.0)
    decision = choose_chunk_size(k=8, a_hat=[0.99] * 8, config=cfg, extra_forward_cost=10.0)
    assert decision.selected_h == 8
    assert decision.reason == "one_shot_min_benefit"


def test_confidence_low_selects_small_prefix() -> None:
    cfg = SpecLinkCVConfig(confidence_sizing=True, min_benefit=0.0)
    decision = choose_chunk_size(k=8, a_hat=[0.2] + [0.9] * 7, config=cfg)
    assert decision.selected_h < 8
    assert decision.reason == "confidence"


if __name__ == "__main__":
    test_fixed_half()
    test_confidence_high_falls_back_full()
    test_confidence_low_selects_small_prefix()
    print("[PASS] test_chunk_decision")
