#!/usr/bin/env python3
"""Unit tests for the vLLM-side SpecLink-CV runtime helpers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vllm.speclink_cv import (
    AsyncPrefixCandidate,
    SpecLinkCVRuntimeConfig,
    append_profile,
    apply_roofline_packing_policy,
    choose_prefix_len,
    select_async_prefix_dispatch,
)


def test_env_parsing() -> None:
    with patch.dict(
        "os.environ",
        {
            "SPECLINK_CV_ENABLE": "1",
            "SPECLINK_CV_CONFIDENCE_SIZING": "0",
            "SPECLINK_CV_CANDIDATE_CHUNKS": "1,2,full",
            "SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP": "4096",
            "SPECLINK_CV_ALLOW_BATCHED_PREFIX": "1",
            "SPECLINK_CV_ALLOW_BATCHED_SUFFIX": "1",
            "SPECLINK_CV_GLOBAL_BATCH_BARRIER": "1",
            "SPECLINK_CV_ALLOW_SHAPE_DRIFT_CHUNKING": "1",
            "SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE": "0",
            "SPECLINK_CV_CONFIRM_PREFIX_REJECT_ONE_SHOT": "1",
            "SPECLINK_CV_CONFIRM_PREFIX_ACCEPT_ONE_SHOT": "1",
            "SPECLINK_CV_CONFIRMATION_FULL_ACTIVE_SET": "1",
            "SPECLINK_CV_LOCKSTEP_ITERATION_BARRIER": "1",
            "SPECLINK_CV_PREFIX_PROBE_BLOCK_ROLLBACK": "1",
            "SPECLINK_CV_PREFIX_LOW_MARGIN_FALLBACK_THRESHOLD": "0.5",
            "SPECLINK_CV_BATCH_WIDE_LOW_MARGIN_FALLBACK": "1",
            "SPECLINK_CV_BATCH_WIDE_PREFIX_REJECT_FALLBACK": "1",
            "SPECLINK_CV_RECOMPUTE_COMMITTED_PREFIX": "1",
            "SPECLINK_CV_ALLOW_BATCHED_DENSE_REALIGN": "1",
            "SPECLINK_CV_PREFIX_NO_KV_WRITE": "1",
            "SPECLINK_CV_FORCE_DECODE_ISOLATION": "1",
            "SPECLINK_CV_KV_DEBUG_TAIL_TOKENS": "4",
            "SPECLINK_CV_KV_DEBUG_MAX_LAYERS": "2",
            "SPECLINK_CV_KV_DEBUG_ROW_INDEX": "3",
            "SPECLINK_CV_KV_DEBUG_MIN_OUTPUT_TOKENS": "10",
            "SPECLINK_CV_KV_DEBUG_MAX_OUTPUT_TOKENS": "20",
            "SPECLINK_CV_LOG_MAX_EVENTS": "123",
            "SPECLINK_CV_PROFILE_MAX_EVENTS": "456",
            "SPECLINK_CV_DENSE_REALIGN_STEPS": "0",
            "SPECLINK_CV_PREFIX_REJECT_DENSE_REALIGN_STEPS": "2",
            "SPECLINK_CV_MAX_QUEUE_WAIT_MS": "3.5",
            "SPECLINK_CV_LOG_JSONL": "/tmp/speclink_cv.jsonl",
            "SPECLINK_CV_PROFILE_JSONL": "/tmp/speclink_cv_profile.jsonl",
        },
        clear=False,
    ):
        cfg = SpecLinkCVRuntimeConfig.from_env()
    assert cfg.enable
    assert not cfg.confidence_sizing
    assert cfg.candidate_chunks == (1, 2, "full")
    assert cfg.max_verify_tokens_per_step == 4096
    assert cfg.allow_batched_prefix
    assert cfg.allow_batched_suffix
    assert cfg.global_batch_barrier
    assert cfg.allow_shape_drift_chunking
    assert not cfg.suffix_replay_one_shot_shape
    assert cfg.confirm_prefix_reject_one_shot
    assert cfg.confirm_prefix_accept_one_shot
    assert cfg.confirmation_full_active_set
    assert cfg.lockstep_iteration_barrier
    assert cfg.prefix_probe_block_rollback
    assert cfg.prefix_low_margin_fallback_threshold == 0.5
    assert cfg.batch_wide_low_margin_fallback
    assert cfg.batch_wide_prefix_reject_fallback
    assert cfg.recompute_committed_prefix
    assert cfg.allow_batched_dense_realign
    assert cfg.prefix_no_kv_write
    assert cfg.force_decode_isolation
    assert cfg.kv_debug_tail_tokens == 4
    assert cfg.kv_debug_max_layers == 2
    assert cfg.kv_debug_row_index == 3
    assert cfg.kv_debug_min_output_tokens == 10
    assert cfg.kv_debug_max_output_tokens == 20
    assert cfg.log_max_events == 123
    assert cfg.profile_max_events == 456
    assert cfg.dense_realign_steps == 0
    assert cfg.effective_dense_realign_steps(8) == 0
    assert cfg.prefix_reject_dense_realign_steps == 2
    assert cfg.effective_prefix_reject_dense_realign_steps(8) == 2
    assert cfg.max_queue_wait_ms == 3.5
    assert cfg.log_jsonl == "/tmp/speclink_cv.jsonl"
    assert cfg.profile_jsonl == "/tmp/speclink_cv_profile.jsonl"


def test_dense_realign_default_uses_num_spec_tokens() -> None:
    cfg = SpecLinkCVRuntimeConfig()
    assert cfg.dense_realign_steps == -1
    assert cfg.effective_dense_realign_steps(8) == 8
    assert cfg.effective_dense_realign_steps(0) == 1
    assert cfg.prefix_reject_dense_realign_steps == 0
    assert cfg.effective_prefix_reject_dense_realign_steps(8) == 0


def test_suffix_replay_defaults_off() -> None:
    with patch.dict("os.environ", {}, clear=True):
        cfg = SpecLinkCVRuntimeConfig.from_env()
    assert not cfg.suffix_replay_one_shot_shape
    assert cfg.log_max_events == 1_000
    assert cfg.profile_max_events == 500


def test_suffix_replay_can_be_enabled() -> None:
    with patch.dict(
        "os.environ",
        {"SPECLINK_CV_SUFFIX_REPLAY_ONE_SHOT_SHAPE": "1"},
        clear=True,
    ):
        cfg = SpecLinkCVRuntimeConfig.from_env()
    assert cfg.suffix_replay_one_shot_shape


def test_batched_suffix_defaults_to_batched_prefix() -> None:
    with patch.dict(
        "os.environ",
        {"SPECLINK_CV_ALLOW_BATCHED_PREFIX": "1"},
        clear=True,
    ):
        cfg = SpecLinkCVRuntimeConfig.from_env()
    assert cfg.allow_batched_prefix
    assert cfg.allow_batched_suffix

    with patch.dict(
        "os.environ",
        {
            "SPECLINK_CV_ALLOW_BATCHED_PREFIX": "1",
            "SPECLINK_CV_ALLOW_BATCHED_SUFFIX": "0",
        },
        clear=True,
    ):
        cfg = SpecLinkCVRuntimeConfig.from_env()
    assert cfg.allow_batched_prefix
    assert not cfg.allow_batched_suffix


def test_fixed_half_prefix_choice() -> None:
    cfg = SpecLinkCVRuntimeConfig(enable=True, confidence_sizing=False)
    assert choose_prefix_len(8, cfg)[:2] == (4, "fixed_half")
    assert choose_prefix_len(12, cfg)[:2] == (6, "fixed_half")


def test_confidence_unavailable_is_conservative() -> None:
    cfg = SpecLinkCVRuntimeConfig(enable=True, confidence_sizing=True)
    assert choose_prefix_len(8, cfg)[:2] == (8, "confidence_unavailable_one_shot")


def test_force_prefix_len_overrides_decision() -> None:
    cfg = SpecLinkCVRuntimeConfig(
        enable=True,
        confidence_sizing=False,
        force_prefix_len=1,
    )
    selected_h, reason, decision = choose_prefix_len(8, cfg)
    assert selected_h == 1
    assert reason == "forced_prefix_len"
    assert decision["forced_prefix_len"] == 1


def test_confidence_selects_small_prefix_when_low_confidence() -> None:
    cfg = SpecLinkCVRuntimeConfig(enable=True, confidence_sizing=True)
    selected_h, reason, decision = choose_prefix_len(8, cfg, [0.35] * 8)
    assert selected_h < 8
    assert reason == "confidence_uncalibrated"
    assert decision["confidence_source"] == "draft_selected_prob_uncalibrated"


def test_confidence_keeps_one_shot_when_high_confidence() -> None:
    cfg = SpecLinkCVRuntimeConfig(enable=True, confidence_sizing=True)
    assert choose_prefix_len(8, cfg, [0.999] * 8)[:2] == (
        8,
        "one_shot_min_benefit",
    )


def test_confidence_uses_binning_calibration() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "calibration_model.json"
        path.write_text(
            json.dumps(
                {
                    "model_type": "binning",
                    "global_acceptance_rate": 0.5,
                    "bins": [
                        {"left": 0.0, "right": 0.5, "acceptance_rate": 0.8},
                        {"left": 0.5, "right": 1.0, "acceptance_rate": 0.2},
                    ],
                }
            ),
            encoding="utf-8",
        )
        cfg = SpecLinkCVRuntimeConfig(
            enable=True,
            confidence_sizing=True,
            calibration_path=str(path),
        )
        selected_h, reason, decision = choose_prefix_len(8, cfg, [0.95] * 8)
    assert selected_h == 2
    assert reason == "confidence_calibrated"
    assert decision["confidence_source"] == "calibrated_binning"
    assert decision["raw_draft_selected_prob"] == [0.95] * 8
    assert decision["a_hat"] == [0.2] * 8


def test_roofline_underfilled_prefix_falls_back() -> None:
    cfg = SpecLinkCVRuntimeConfig(
        enable=True,
        confidence_sizing=False,
        roofline_packing=True,
        allow_batched_prefix=True,
        util_threshold=0.6,
    )
    selected_h, reason, decision = choose_prefix_len(8, cfg)
    selected_h, reason, decision = apply_roofline_packing_policy(
        k=8,
        selected_h=selected_h,
        reason=reason,
        decision=decision,
        config=cfg,
        candidate_seq_count=1,
        token_budget=1024,
        seq_budget=8,
    )
    assert selected_h == 8
    assert reason == "roofline_fallback_one_shot"
    assert decision["roofline"]["predicted_utilization"] < 0.6


def test_roofline_keeps_well_packed_prefix() -> None:
    cfg = SpecLinkCVRuntimeConfig(
        enable=True,
        confidence_sizing=False,
        roofline_packing=True,
        allow_batched_prefix=True,
        util_threshold=0.5,
    )
    selected_h, reason, decision = choose_prefix_len(8, cfg)
    selected_h, reason, decision = apply_roofline_packing_policy(
        k=8,
        selected_h=selected_h,
        reason=reason,
        decision=decision,
        config=cfg,
        candidate_seq_count=8,
        token_budget=64,
        seq_budget=8,
    )
    assert selected_h == 4
    assert reason == "fixed_half"
    assert decision["roofline"]["predicted_utilization"] >= 0.5


def test_append_profile_writes_jsonl() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "profile.jsonl"
        cfg = SpecLinkCVRuntimeConfig(enable=True, profile_jsonl=str(path))
        append_profile(cfg, {"event": "unit_test", "value": 3})
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "unit_test"
    assert rows[0]["value"] == 3
    assert "ts" in rows[0]


def test_append_profile_respects_max_events() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "profile_limited.jsonl"
        cfg = SpecLinkCVRuntimeConfig(
            enable=True,
            profile_jsonl=str(path),
            profile_max_events=2,
        )
        append_profile(cfg, {"event": "first"})
        append_profile(cfg, {"event": "second"})
        append_profile(cfg, {"event": "third"})
        append_profile(cfg, {"event": "fourth"})
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    assert [row["event"] for row in rows] == [
        "first",
        "second",
        "jsonl_limit_reached",
    ]
    assert rows[-1]["max_events"] == 2


def test_async_dispatch_waits_for_roofline_utilization() -> None:
    cfg = SpecLinkCVRuntimeConfig(
        enable=True,
        async_queue=True,
        roofline_packing=True,
        allow_batched_prefix=True,
        util_threshold=0.9,
        max_queue_wait_ms=100.0,
    )
    selected, detail = select_async_prefix_dispatch(
        [AsyncPrefixCandidate("r0", selected_h=4, k=8, queue_enter_time=1.0)],
        config=cfg,
        token_budget=1024,
        seq_budget=8,
        now=1.001,
        has_other_ready_work=True,
    )
    assert selected == set()
    assert detail["reason"] == "wait_for_utilization"


def test_async_dispatch_timeout_prevents_starvation() -> None:
    cfg = SpecLinkCVRuntimeConfig(
        enable=True,
        async_queue=True,
        roofline_packing=True,
        allow_batched_prefix=True,
        util_threshold=0.9,
        max_queue_wait_ms=2.0,
    )
    selected, detail = select_async_prefix_dispatch(
        [AsyncPrefixCandidate("r0", selected_h=4, k=8, queue_enter_time=1.0)],
        config=cfg,
        token_budget=1024,
        seq_budget=8,
        now=1.010,
        has_other_ready_work=True,
    )
    assert selected == {"r0"}
    assert detail["reason"] == "age_timeout"


def test_async_dispatch_conservative_default_caps_one_prefix() -> None:
    cfg = SpecLinkCVRuntimeConfig(
        enable=True,
        async_queue=True,
        roofline_packing=True,
        max_verify_seqs_per_step=8,
    )
    selected, detail = select_async_prefix_dispatch(
        [
            AsyncPrefixCandidate("r0", selected_h=4, k=8, queue_enter_time=1.0),
            AsyncPrefixCandidate("r1", selected_h=4, k=8, queue_enter_time=1.0),
        ],
        config=cfg,
        token_budget=1024,
        seq_budget=8,
        now=1.001,
        has_other_ready_work=False,
    )
    assert len(selected) == 1
    assert detail["seq_budget"] == 1
    assert not detail["allow_batched_prefix"]


if __name__ == "__main__":
    test_env_parsing()
    test_dense_realign_default_uses_num_spec_tokens()
    test_suffix_replay_defaults_off()
    test_suffix_replay_can_be_enabled()
    test_fixed_half_prefix_choice()
    test_confidence_unavailable_is_conservative()
    test_force_prefix_len_overrides_decision()
    test_confidence_selects_small_prefix_when_low_confidence()
    test_confidence_keeps_one_shot_when_high_confidence()
    test_confidence_uses_binning_calibration()
    test_roofline_underfilled_prefix_falls_back()
    test_roofline_keeps_well_packed_prefix()
    test_append_profile_writes_jsonl()
    test_async_dispatch_waits_for_roofline_utilization()
    test_async_dispatch_timeout_prevents_starvation()
    test_async_dispatch_conservative_default_caps_one_prefix()
    print("[PASS] test_vllm_runtime_config")
