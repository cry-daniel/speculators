#!/usr/bin/env python3
"""Lightweight correctness checks for the SpecLink SR24 prototype."""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = REPO_ROOT / "vllm"
sys.path.insert(0, str(VLLM_ROOT))

from vllm.speclink_sr24 import (  # noqa: E402
    FixedPrefixRouteDescriptor,
    VerifyResidualPlan,
    _adaptive_dense_fallback_decision,
    _bucket_dense_overwrite_inplace,
    _semi_structured_linear,
    _sparse_base_weight,
    _triton_bucket_override_inplace,
    apply_sr24_from_env,
    begin_propose_context,
    begin_verify_context,
    build_verify_residual_mask,
    end_propose_context,
    end_verify_context,
    linear_hooks_enabled,
    record_draft_scores,
    residual_linear_output,
    row_routed_down_output,
    row_routed_mlp_output,
    sparse_backend_active,
    sparse_linear_output,
)


def unwrap_residual_mask(plan: object) -> torch.Tensor:
    assert plan is not None
    mask = getattr(plan, "mask", plan)
    assert isinstance(mask, torch.Tensor)
    return mask


def unwrap_residual_priority(plan: object) -> torch.Tensor:
    assert plan is not None
    priority = getattr(plan, "priority", None)
    assert isinstance(priority, torch.Tensor)
    return priority


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(8, 12, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.gate_up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock()])


def _module_originals(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor)
    }


def _reconstruct_residual(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    out_features, in_features = weight.shape
    usable_in = int(module._speclink_sr24_usable_in)
    groups = usable_in // 4
    mask_bytes = module._speclink_sr24_base_mask_bytes
    packed = mask_bytes.to(dtype=torch.uint8)
    group_bytes = torch.empty(out_features, groups, dtype=torch.uint8)
    group_bytes[:, 0::2] = packed[:, : (groups + 1) // 2] & 0x0F
    group_bytes[:, 1::2] = (packed[:, : groups // 2] >> 4) & 0x0F
    bits = torch.tensor([1, 2, 4, 8], dtype=torch.uint8)
    keep = (group_bytes.unsqueeze(-1) & bits.view(1, 1, 4)).ne(0)
    assert bool((keep.sum(dim=-1) == 2).all().item())
    residual = torch.zeros_like(weight)
    residual[:, :usable_in].view(out_features, groups, 4)[~keep] = (
        module._speclink_sr24_residual_values.to(dtype=weight.dtype)
    )
    return residual


def _record_unit_scores(
    *,
    req_ids: list[str],
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    num_spec_tokens: int,
    generated_lens: list[int] | None = None,
) -> None:
    propose_token = begin_propose_context(
        req_ids=req_ids,
        prompt_lens=[1] * len(req_ids),
        generated_lens=generated_lens or [0] * len(req_ids),
        active_requests=len(req_ids),
        batch_size=len(req_ids),
        num_spec_tokens=num_spec_tokens,
        method="unit",
    )
    try:
        record_draft_scores(
            draft_token_ids=draft_token_ids,
            logits_by_position=logits_by_position,
            method="unit",
        )
    finally:
        end_propose_context(propose_token)


def _build_unit_plan(
    *,
    req_ids: list[str],
    score_req_ids: list[str] | None = None,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    scheduled_tokens_per_req: int,
    device: torch.device,
    generated_lens: list[int] | None = None,
    use_gpu_counts: bool = False,
) -> torch.Tensor:
    num_spec_tokens = int(draft_token_ids.shape[1])
    _record_unit_scores(
        req_ids=score_req_ids or req_ids,
        draft_token_ids=draft_token_ids,
        logits_by_position=logits_by_position,
        num_spec_tokens=num_spec_tokens,
        generated_lens=generated_lens,
    )
    scheduled = [scheduled_tokens_per_req] * len(req_ids)
    cu_scheduled = [
        scheduled_tokens_per_req * (idx + 1) for idx in range(len(req_ids))
    ]
    gpu_kwargs = {}
    if use_gpu_counts:
        gpu_kwargs = {
            "num_scheduled_tokens_gpu": torch.tensor(
                scheduled,
                dtype=torch.int32,
                device=device,
            ),
            "num_draft_tokens_gpu": torch.full(
                (len(req_ids),),
                num_spec_tokens,
                dtype=torch.int32,
                device=device,
            ),
            "cu_num_scheduled_tokens_gpu": torch.tensor(
                cu_scheduled,
                dtype=torch.int32,
                device=device,
            ),
        }
    previous_gpu_count_builder = os.environ.get(
        "SPECLINK_SR24_GPU_COUNT_MASK_BUILDER"
    )
    if use_gpu_counts:
        os.environ["SPECLINK_SR24_GPU_COUNT_MASK_BUILDER"] = "1"
    try:
        plan = build_verify_residual_mask(
            req_ids=req_ids,
            num_scheduled_tokens=scheduled,
            num_draft_tokens=[num_spec_tokens] * len(req_ids),
            cu_num_scheduled_tokens=cu_scheduled,
            total_num_scheduled_tokens=scheduled_tokens_per_req * len(req_ids),
            device=device,
            **gpu_kwargs,
        )
    finally:
        if use_gpu_counts:
            if previous_gpu_count_builder is None:
                os.environ.pop("SPECLINK_SR24_GPU_COUNT_MASK_BUILDER", None)
            else:
                os.environ["SPECLINK_SR24_GPU_COUNT_MASK_BUILDER"] = (
                    previous_gpu_count_builder
                )
    return plan


def _build_unit_mask(
    *,
    req_ids: list[str],
    score_req_ids: list[str] | None = None,
    draft_token_ids: torch.Tensor,
    logits_by_position: list[torch.Tensor],
    scheduled_tokens_per_req: int,
    device: torch.device,
    generated_lens: list[int] | None = None,
    use_gpu_counts: bool = False,
) -> torch.Tensor:
    plan = _build_unit_plan(
        req_ids=req_ids,
        score_req_ids=score_req_ids,
        draft_token_ids=draft_token_ids,
        logits_by_position=logits_by_position,
        scheduled_tokens_per_req=scheduled_tokens_per_req,
        device=device,
        generated_lens=generated_lens,
        use_gpu_counts=use_gpu_counts,
    )
    if (
        isinstance(plan, VerifyResidualPlan)
        and plan.mask is None
        and plan.state in {"all_residual", "no_residual"}
    ):
        return torch.full(
            (scheduled_tokens_per_req * len(req_ids),),
            plan.state == "all_residual",
            dtype=torch.bool,
            device=device,
        )
    return unwrap_residual_mask(plan).detach().clone()


def main() -> None:
    torch.manual_seed(0)
    os.environ["SPECLINK_SR24_ENABLE"] = "1"
    os.environ["SPECLINK_SR24_MODE"] = "all_corrected"
    os.environ["SPECLINK_SR24_BACKEND"] = "prototype"
    os.environ["SPECLINK_SR24_THRESHOLD"] = "0.8"
    os.environ.pop("SPECLINK_SR24_REQUIRE_GPU_RESIDUAL", None)
    os.environ.pop("SPECLINK_SR24_RESIDUAL_DEVICE", None)
    os.environ.pop("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK", None)
    os.environ.pop("SPECLINK_SR24_MASK_PATH", None)
    os.environ.pop("SPECLINK_STRUCTURED_24_ENABLE", None)
    os.environ.pop("SPECLINK_TOKEN_DENSE_ENABLE", None)

    fake_gate_up = type("FakeModule", (), {})()
    fake_gate_up._speclink_sr24_profile_leaf = "gate_up_proj"
    fake_gate_up._speclink_sr24_residual_backend = "dense_rows"
    fake_down = type("FakeModule", (), {})()
    fake_down._speclink_sr24_profile_leaf = "down_proj"
    fake_down._speclink_sr24_residual_backend = "dense_rows"
    os.environ["SPECLINK_SR24_MODE"] = "selective"
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK"] = "1"
    should_fallback, reason = _adaptive_dense_fallback_decision(
        fake_gate_up,
        rows=512,
        dense_candidate_rows=64,
    )
    assert should_fallback, reason
    should_fallback, reason = _adaptive_dense_fallback_decision(
        fake_down,
        rows=512,
        dense_candidate_rows=64,
    )
    assert not should_fallback, reason
    should_fallback, reason = _adaptive_dense_fallback_decision(
        fake_down,
        rows=64,
        dense_candidate_rows=0,
        allow_zero_residual=True,
    )
    assert not should_fallback and reason == "no_residual_rows", reason
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS"] = "64"
    os.environ["SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL"] = "1"
    should_fallback, reason = _adaptive_dense_fallback_decision(
        fake_down,
        rows=64,
        dense_candidate_rows=0,
        allow_zero_residual=True,
    )
    assert should_fallback, reason
    os.environ.pop("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS", None)
    os.environ.pop(
        "SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_DOWN_NO_RESIDUAL",
        None,
    )
    os.environ.pop("SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK", None)
    os.environ["SPECLINK_SR24_MODE"] = "all_corrected"

    if torch.cuda.is_available():
        base_output = torch.randn(7, 32, device="cuda", dtype=torch.float16)
        dense_output = torch.randn(4, 32, device="cuda", dtype=torch.float16)
        bucket_rows = torch.tensor([0, 2, 5, 6], device="cuda", dtype=torch.long)
        bucket_values = torch.tensor(
            [True, False, True, True],
            device="cuda",
            dtype=torch.bool,
        )
        expected = base_output.clone()
        active = bucket_values.nonzero(as_tuple=False).squeeze(1)
        expected.index_copy_(
            0,
            bucket_rows.index_select(0, active),
            dense_output.index_select(0, active),
        )
        actual = _triton_bucket_override_inplace(
            base_output.clone(),
            dense_output,
            bucket_rows,
            bucket_values,
        )
        torch.cuda.synchronize()
        assert torch.allclose(actual, expected)
        overwrite_actual = _bucket_dense_overwrite_inplace(
            base_output.clone(),
            dense_output,
            bucket_rows,
            bucket_values,
        )
        torch.cuda.synchronize()
        assert torch.allclose(overwrite_actual, expected)

    model = TinyModel()
    originals = _module_originals(model)
    stats = apply_sr24_from_env(model, context="unit")
    assert stats is not None
    assert stats["module_count_attached"] == 4

    for name, module in model.named_modules():
        if not getattr(module, "_speclink_sr24_enabled", False):
            continue
        original = originals[name]
        residual = _reconstruct_residual(module)
        assert torch.allclose(module.weight + residual, original)

        x = torch.randn(5, original.shape[1])
        dense_out = F.linear(x, original)
        base_out = F.linear(x, module.weight)
        os.environ["SPECLINK_SR24_MODE"] = "all_corrected"
        corrected = residual_linear_output(module, x, base_out)
        assert torch.allclose(corrected, dense_out, atol=1e-5, rtol=1e-5)

        os.environ["SPECLINK_SR24_MODE"] = "selective"
        mask = torch.tensor([True, False, True, False, True])
        token = begin_verify_context(mask)
        try:
            selective = residual_linear_output(module, x, base_out)
        finally:
            end_verify_context(token)
        expected = base_out.clone()
        expected[mask] = dense_out[mask]
        assert torch.allclose(selective, expected, atol=1e-5, rtol=1e-5)

        os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "1"
        no_mask_corrected = residual_linear_output(module, x, base_out)
        assert torch.allclose(no_mask_corrected, dense_out, atol=1e-5, rtol=1e-5)

        os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "0"
        no_mask_legacy = residual_linear_output(module, x, base_out)
        assert torch.allclose(no_mask_legacy, base_out)
        os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "1"

        os.environ["SPECLINK_SR24_MODE"] = "base_only"
        assert torch.allclose(residual_linear_output(module, x, base_out), base_out)

    os.environ["SPECLINK_SR24_MODE"] = "selective"
    os.environ["SPECLINK_SR24_BACKEND"] = "dense_zero"
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "dense_rows"
    os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "1"
    dense_rows_model = TinyModel()
    dense_rows_originals = _module_originals(dense_rows_model)
    dense_rows_stats = apply_sr24_from_env(dense_rows_model, context="dense_rows_unit")
    assert dense_rows_stats is not None
    assert dense_rows_stats["module_count_attached"] == 4
    assert dense_rows_stats["residual_backend"] == "dense_rows"
    for name, module in dense_rows_model.named_modules():
        if not getattr(module, "_speclink_sr24_enabled", False):
            continue
        original = dense_rows_originals[name]
        assert hasattr(module, "_speclink_sr24_dense_weight")
        x = torch.randn(5, original.shape[1])
        dense_out = F.linear(x, original)
        base_out = F.linear(x, module.weight)
        mask = torch.tensor([True, False, True, False, True])
        token = begin_verify_context(mask)
        try:
            selective = residual_linear_output(module, x, base_out)
        finally:
            end_verify_context(token)
        expected = base_out.clone()
        expected[mask] = dense_out[mask]
        assert torch.allclose(selective, expected, atol=1e-5, rtol=1e-5)
        os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1"
        os.environ["SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT"] = "1"
        token = begin_verify_context(mask)
        try:
            reuse_selective = residual_linear_output(module, x, base_out)
        finally:
            end_verify_context(token)
            os.environ.pop("SPECLINK_SR24_REDUCE_CPU_SYNC", None)
            os.environ.pop("SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT", None)
        assert torch.allclose(reuse_selective, expected, atol=1e-5, rtol=1e-5)
        os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1"
        token = begin_verify_context(
            VerifyResidualPlan(mask=None, state="all_residual")
        )
        try:
            all_residual_reduced = residual_linear_output(module, x, base_out)
        finally:
            end_verify_context(token)
            os.environ.pop("SPECLINK_SR24_REDUCE_CPU_SYNC", None)
        assert torch.allclose(
            all_residual_reduced,
            dense_out,
            atol=1e-5,
            rtol=1e-5,
        )
        token = begin_verify_context(mask)
        try:
            prehook_selective = sparse_linear_output(module, x)
        finally:
            end_verify_context(token)
        assert prehook_selective is not None
        assert torch.allclose(prehook_selective, expected, atol=1e-5, rtol=1e-5)
        os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1"
        os.environ["SPECLINK_SR24_RESIDUAL_BUCKET_SIZE"] = "8"
        token = begin_verify_context(mask)
        try:
            bucket_selective = sparse_linear_output(module, x)
        finally:
            end_verify_context(token)
            os.environ.pop("SPECLINK_SR24_REDUCE_CPU_SYNC", None)
            os.environ.pop("SPECLINK_SR24_RESIDUAL_BUCKET_SIZE", None)
        assert bucket_selective is not None
        assert torch.allclose(bucket_selective, expected, atol=1e-5, rtol=1e-5)
        no_mask_corrected = residual_linear_output(module, x, base_out)
        assert torch.allclose(no_mask_corrected, dense_out, atol=1e-5, rtol=1e-5)

    os.environ["SPECLINK_SR24_MODE"] = "all_corrected"
    os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "compressed_dense"
    os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = "auto"
    os.environ.pop("SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH", None)
    fastpath_model = TinyModel()
    fastpath_originals = _module_originals(fastpath_model)
    fastpath_stats = apply_sr24_from_env(
        fastpath_model,
        context="unit_all_corrected_dense_fastpath_default",
    )
    assert fastpath_stats is not None
    assert fastpath_stats["dense_fastpath_noop"] is True
    assert fastpath_stats["residual_backend"] == "dense_fastpath"
    assert fastpath_stats["residual_device"] == "none"
    assert fastpath_stats["module_count_attached"] == 4
    assert linear_hooks_enabled() is False
    for name, module in fastpath_model.named_modules():
        if not getattr(module, "_speclink_sr24_enabled", False):
            continue
        assert getattr(module, "_speclink_sr24_dense_fastpath", False) is True
        assert not hasattr(module, "_speclink_sr24_residual_values")
        assert not hasattr(module, "_speclink_sr24_base_mask_bytes")
        assert torch.equal(module.weight, fastpath_originals[name])
        x = torch.randn(5, fastpath_originals[name].shape[1])
        assert sparse_linear_output(module, x) is None

    os.environ["SPECLINK_SR24_MODE"] = "selective"
    os.environ["SPECLINK_SR24_BACKEND"] = "prototype"
    os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "compressed_dense"
    os.environ["SPECLINK_SR24_THRESHOLD"] = "0.8"
    os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "1"
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "critical_prefix"
    score_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    draft_token_ids = torch.tensor(
        [[1, 1, 1], [1, 1, 1]],
        device=score_device,
        dtype=torch.long,
    )
    logits_by_position = []
    for _ in range(3):
        logits = torch.zeros(2, 5, device=score_device)
        logits[0].fill_(-10.0)
        logits[0, 1] = 10.0
        logits_by_position.append(logits)
    propose_token = begin_propose_context(
        req_ids=["sr24_unit_high", "sr24_unit_low"],
        prompt_lens=[1, 1],
        generated_lens=[0, 0],
        active_requests=2,
        batch_size=2,
        num_spec_tokens=3,
        method="unit",
    )
    try:
        record_draft_scores(
            draft_token_ids=draft_token_ids,
            logits_by_position=logits_by_position,
            method="unit",
        )
    finally:
        end_propose_context(propose_token)
    residual_plan = build_verify_residual_mask(
        req_ids=["sr24_unit_high", "sr24_unit_low"],
        num_scheduled_tokens=[4, 4],
        num_draft_tokens=[3, 3],
        cu_num_scheduled_tokens=[4, 8],
        total_num_scheduled_tokens=8,
        device=score_device,
    )
    residual_mask = unwrap_residual_mask(residual_plan)
    assert residual_mask.device.type == score_device.type
    expected_mask = torch.tensor(
        [True, True, True, True, True, False, False, True],
        device=score_device,
    )
    assert torch.equal(residual_mask, expected_mask)

    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "all_if_any_low"
    mixed_logits_by_position = []
    for pos in range(3):
        logits = torch.zeros(2, 5, device=score_device)
        logits[1].fill_(-10.0)
        logits[1, 1] = 10.0
        if pos != 1:
            logits[0].fill_(-10.0)
            logits[0, 1] = 10.0
        mixed_logits_by_position.append(logits)
    propose_token = begin_propose_context(
        req_ids=["sr24_unit_mixed_all_if_any", "sr24_unit_high_all_if_any"],
        prompt_lens=[1, 1],
        generated_lens=[0, 0],
        active_requests=2,
        batch_size=2,
        num_spec_tokens=3,
        method="unit",
    )
    try:
        record_draft_scores(
            draft_token_ids=draft_token_ids,
            logits_by_position=mixed_logits_by_position,
            method="unit",
        )
    finally:
        end_propose_context(propose_token)
    residual_plan = build_verify_residual_mask(
        req_ids=["sr24_unit_mixed_all_if_any", "sr24_unit_high_all_if_any"],
        num_scheduled_tokens=[4, 4],
        num_draft_tokens=[3, 3],
        cu_num_scheduled_tokens=[4, 8],
        total_num_scheduled_tokens=8,
        device=score_device,
    )
    expected_mask = torch.tensor(
        [True, True, True, True, False, False, False, True],
        device=score_device,
    )
    residual_mask = unwrap_residual_mask(residual_plan)
    assert torch.equal(residual_mask, expected_mask)

    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "prefix_confidence"
    os.environ["SPECLINK_SR24_PREFIX_THRESHOLD"] = "0.5"
    os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "bonus"
    prefix_logits_by_position = []
    prefix_probs = [
        [0.9, 0.9],
        [0.9, 0.4],
        [0.9, 0.9],
    ]
    for probs in prefix_probs:
        logits = torch.empty(2, 5, device=score_device)
        logits.fill_(-100.0)
        logits[:, 0] = 0.0
        for row, prob in enumerate(probs):
            logits[row, 1] = math.log(prob / (1.0 - prob))
        prefix_logits_by_position.append(logits)
    residual_mask = _build_unit_mask(
        req_ids=["sr24_unit_prefix_high", "sr24_unit_prefix_drop"],
        draft_token_ids=draft_token_ids,
        logits_by_position=prefix_logits_by_position,
        scheduled_tokens_per_req=4,
        device=score_device,
    )
    expected_mask = torch.tensor(
        [True, True, True, True, True, False, False, True],
        device=score_device,
    )
    assert torch.equal(residual_mask, expected_mask)
    os.environ.pop("SPECLINK_SR24_PREFIX_THRESHOLD", None)
    os.environ.pop("SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY", None)

    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "low_confidence"
    os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "bonus"
    os.environ["SPECLINK_SR24_EARLY_DENSE_TOKENS"] = "2"
    early_req_ids = ["sr24_unit_early_prefix", "sr24_unit_late_prefix"]
    high_logits_by_position = []
    for _ in range(3):
        logits = torch.empty(2, 5, device=score_device)
        logits.fill_(-10.0)
        logits[:, 1] = 10.0
        high_logits_by_position.append(logits)
    _record_unit_scores(
        req_ids=early_req_ids,
        draft_token_ids=draft_token_ids,
        logits_by_position=high_logits_by_position,
        num_spec_tokens=3,
        generated_lens=[0, 3],
    )
    residual_plan = build_verify_residual_mask(
        req_ids=early_req_ids,
        num_scheduled_tokens=[4, 4],
        num_draft_tokens=[3, 3],
        cu_num_scheduled_tokens=[4, 8],
        total_num_scheduled_tokens=8,
        device=score_device,
    )
    expected_mask = torch.tensor(
        [True, True, False, True, False, False, False, True],
        device=score_device,
    )
    residual_mask = unwrap_residual_mask(residual_plan)
    assert torch.equal(residual_mask, expected_mask)
    os.environ.pop("SPECLINK_SR24_EARLY_DENSE_TOKENS", None)
    os.environ.pop("SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY", None)
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "critical_prefix"

    propose_token = begin_propose_context(
        req_ids=["sr24_unit_high_low_policy", "sr24_unit_low_low_policy"],
        prompt_lens=[1, 1],
        generated_lens=[0, 0],
        active_requests=2,
        batch_size=2,
        num_spec_tokens=3,
        method="unit",
    )
    try:
        record_draft_scores(
            draft_token_ids=draft_token_ids,
            logits_by_position=logits_by_position,
            method="unit",
        )
    finally:
        end_propose_context(propose_token)
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "low_confidence"
    residual_plan = build_verify_residual_mask(
        req_ids=["sr24_unit_high_low_policy", "sr24_unit_low_low_policy"],
        num_scheduled_tokens=[4, 4],
        num_draft_tokens=[3, 3],
        cu_num_scheduled_tokens=[4, 8],
        total_num_scheduled_tokens=8,
        device=score_device,
    )
    expected_mask = torch.tensor(
        [False, False, False, True, True, True, True, True],
        device=score_device,
    )
    residual_mask = unwrap_residual_mask(residual_plan)
    assert torch.equal(residual_mask, expected_mask)

    propose_token = begin_propose_context(
        req_ids=["sr24_unit_high_old", "sr24_unit_low_old"],
        prompt_lens=[1, 1],
        generated_lens=[0, 0],
        active_requests=2,
        batch_size=2,
        num_spec_tokens=3,
        method="unit",
    )
    try:
        record_draft_scores(
            draft_token_ids=draft_token_ids,
            logits_by_position=logits_by_position,
            method="unit",
        )
    finally:
        end_propose_context(propose_token)
    os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "0"
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "high_confidence"
    residual_plan = build_verify_residual_mask(
        req_ids=["sr24_unit_high_old", "sr24_unit_low_old"],
        num_scheduled_tokens=[4, 4],
        num_draft_tokens=[3, 3],
        cu_num_scheduled_tokens=[4, 8],
        total_num_scheduled_tokens=8,
        device=score_device,
    )
    expected_mask = torch.tensor(
        [True, True, True, False, False, False, False, False],
        device=score_device,
    )
    residual_mask = unwrap_residual_mask(residual_plan)
    assert torch.equal(residual_mask, expected_mask)
    os.environ["SPECLINK_SR24_SELECTIVE_CORRECT_NON_DRAFT"] = "1"
    os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "critical_prefix"

    if torch.cuda.is_available():
        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "bonus"
        os.environ["SPECLINK_SR24_SELECTIVE_EXTRA_AFTER_LOW"] = "3"
        os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1"
        os.environ["SPECLINK_SR24_SYNC_MASK_STATE"] = "0"
        os.environ["SPECLINK_SR24_STATIC_MASK_BUFFER"] = "1"
        compare_req_ids = [
            "sr24_unit_batched_high",
            "sr24_unit_batched_low0",
            "sr24_unit_batched_low3",
        ]
        compare_draft_token_ids = torch.ones(
            (3, 5),
            dtype=torch.long,
            device=score_device,
        )
        compare_logits_by_position = []
        for pos in range(5):
            logits = torch.empty(3, 5, device=score_device)
            logits.fill_(-10.0)
            logits[:, 1] = 10.0
            if pos == 0:
                logits[1].fill_(0.0)
            if pos == 3:
                logits[2].fill_(0.0)
            compare_logits_by_position.append(logits)
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        slow_mask = _build_unit_mask(
            req_ids=[f"{req}_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        batched_uniform_mask = _build_unit_mask(
            req_ids=[f"{req}_batched_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        batched_indexed_mask = _build_unit_mask(
            req_ids=[f"{req}_batched_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ.pop("SPECLINK_SR24_BATCHED_UNIFORM_DIRECT", None)
        batched_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_batched_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        assert torch.equal(slow_mask, batched_uniform_mask), (
            "uniform-direct SR24 mask builder diverged from slow path: "
            f"slow={slow_mask.to('cpu').tolist()} "
            f"uniform={batched_uniform_mask.to('cpu').tolist()}"
        )
        assert torch.equal(slow_mask, batched_indexed_mask), (
            "indexed SR24 mask builder diverged from slow path after disabling "
            "uniform direct: "
            f"slow={slow_mask.to('cpu').tolist()} "
            f"indexed={batched_indexed_mask.to('cpu').tolist()}"
        )
        assert torch.equal(slow_mask, batched_gpu_counts_mask), (
            "GPU-count SR24 mask builder diverged from slow path: "
            f"slow={slow_mask.to('cpu').tolist()} "
            f"batched={batched_gpu_counts_mask.to('cpu').tolist()}"
        )
        non_direct_req_ids = [f"{req}_non_direct" for req in reversed(compare_req_ids)]
        non_direct_score_req_ids = [
            f"{req}_non_direct" for req in compare_req_ids
        ]
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        slow_non_direct_mask = _build_unit_mask(
            req_ids=non_direct_req_ids,
            score_req_ids=non_direct_score_req_ids,
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        batched_non_direct_gpu_counts_mask = _build_unit_mask(
            req_ids=non_direct_req_ids,
            score_req_ids=non_direct_score_req_ids,
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        assert torch.equal(slow_non_direct_mask, batched_non_direct_gpu_counts_mask), (
            "GPU-count non-direct score-row fallback diverged from slow path: "
            f"slow={slow_non_direct_mask.to('cpu').tolist()} "
            f"batched={batched_non_direct_gpu_counts_mask.to('cpu').tolist()}"
        )
        expected_bonus_mask = torch.tensor(
            [
                True, True, True, True, True, True,
                True, True, True, True, False, True,
                True, True, True, True, True, True,
            ],
            device=score_device,
        )
        assert torch.equal(batched_uniform_mask, expected_bonus_mask)

        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "critical_prefix"
        os.environ["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = "1"
        os.environ["SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS"] = "2"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        capped_slow_mask = _build_unit_mask(
            req_ids=[f"{req}_capped_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        capped_uniform_mask = _build_unit_mask(
            req_ids=[f"{req}_capped_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        capped_indexed_mask = _build_unit_mask(
            req_ids=[f"{req}_capped_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ.pop("SPECLINK_SR24_BATCHED_UNIFORM_DIRECT", None)
        capped_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_capped_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        expected_capped_mask = torch.tensor(
            [
                True, True, False, False, False, True,
                True, True, False, False, False, True,
                True, True, False, False, False, True,
            ],
            device=score_device,
        )
        assert torch.equal(capped_slow_mask, expected_capped_mask)
        for label, mask in (
            ("uniform-direct", capped_uniform_mask),
            ("indexed", capped_indexed_mask),
            ("gpu-count", capped_gpu_counts_mask),
        ):
            assert torch.equal(capped_slow_mask, mask), (
                f"{label} capped critical-prefix SR24 mask builder diverged "
                "from slow path: "
                f"slow={capped_slow_mask.to('cpu').tolist()} "
                f"batched={mask.to('cpu').tolist()}"
            )
        os.environ.pop("SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL", None)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_MAX_RESIDUAL_DRAFT_ROWS", None)

        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = (
            "predicted_full_accept"
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        predicted_slow_mask = _build_unit_mask(
            req_ids=[f"{req}_predicted_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        predicted_uniform_mask = _build_unit_mask(
            req_ids=[f"{req}_predicted_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        predicted_indexed_mask = _build_unit_mask(
            req_ids=[f"{req}_predicted_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ.pop("SPECLINK_SR24_BATCHED_UNIFORM_DIRECT", None)
        predicted_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_predicted_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        expected_predicted_bonus_mask = torch.tensor(
            [
                True, True, True, True, True, True,
                True, True, True, True, False, False,
                True, True, True, True, True, False,
            ],
            device=score_device,
        )
        assert torch.equal(predicted_slow_mask, expected_predicted_bonus_mask)
        assert torch.equal(predicted_slow_mask, predicted_uniform_mask), (
            "uniform-direct predicted_full_accept SR24 mask builder diverged "
            "from slow path: "
            f"slow={predicted_slow_mask.to('cpu').tolist()} "
            f"uniform={predicted_uniform_mask.to('cpu').tolist()}"
        )
        assert torch.equal(predicted_slow_mask, predicted_indexed_mask), (
            "indexed predicted_full_accept SR24 mask builder diverged from "
            "slow path: "
            f"slow={predicted_slow_mask.to('cpu').tolist()} "
            f"indexed={predicted_indexed_mask.to('cpu').tolist()}"
        )
        assert torch.equal(predicted_slow_mask, predicted_gpu_counts_mask), (
            "GPU-count predicted_full_accept SR24 mask builder diverged from "
            "slow path: "
            f"slow={predicted_slow_mask.to('cpu').tolist()} "
            f"batched={predicted_gpu_counts_mask.to('cpu').tolist()}"
        )
        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "bonus"

        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "high_confidence"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        high_slow_mask = _build_unit_mask(
            req_ids=[f"{req}_high_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        high_batched_uniform_mask = _build_unit_mask(
            req_ids=[f"{req}_high_batched_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        high_batched_indexed_mask = _build_unit_mask(
            req_ids=[f"{req}_high_batched_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        high_batched_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_high_batched_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        assert torch.equal(high_slow_mask, high_batched_uniform_mask), (
            "uniform-direct high-confidence SR24 mask builder diverged from "
            "slow path: "
            f"slow={high_slow_mask.to('cpu').tolist()} "
            f"uniform={high_batched_uniform_mask.to('cpu').tolist()}"
        )
        assert torch.equal(high_slow_mask, high_batched_indexed_mask), (
            "indexed high-confidence SR24 mask builder diverged from slow path: "
            f"slow={high_slow_mask.to('cpu').tolist()} "
            f"indexed={high_batched_indexed_mask.to('cpu').tolist()}"
        )
        assert torch.equal(high_slow_mask, high_batched_gpu_counts_mask), (
            "GPU-count high-confidence SR24 mask builder diverged from slow path: "
            f"slow={high_slow_mask.to('cpu').tolist()} "
            f"batched={high_batched_gpu_counts_mask.to('cpu').tolist()}"
        )
        expected_high_mask = torch.tensor(
            [
                True, True, True, True, True, True,
                False, True, True, True, True, True,
                True, True, True, False, True, True,
            ],
            device=score_device,
        )
        assert torch.equal(high_slow_mask, expected_high_mask)

        os.environ["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = "2"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        min_prefix_slow_mask = _build_unit_mask(
            req_ids=[f"{req}_min_prefix_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        min_prefix_uniform_mask = _build_unit_mask(
            req_ids=[f"{req}_min_prefix_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        min_prefix_indexed_mask = _build_unit_mask(
            req_ids=[f"{req}_min_prefix_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ.pop("SPECLINK_SR24_BATCHED_UNIFORM_DIRECT", None)
        min_prefix_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_min_prefix_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        for label, mask in (
            ("uniform-direct", min_prefix_uniform_mask),
            ("indexed", min_prefix_indexed_mask),
            ("gpu-count", min_prefix_gpu_counts_mask),
        ):
            assert torch.equal(min_prefix_slow_mask, mask), (
                f"{label} min-prefix SR24 mask builder diverged from slow path: "
                f"slow={min_prefix_slow_mask.to('cpu').tolist()} "
                f"batched={mask.to('cpu').tolist()}"
            )
        expected_min_prefix_mask = torch.tensor(
            [
                True, True, True, True, True, True,
                True, True, True, True, True, True,
                True, True, True, False, True, True,
            ],
            device=score_device,
        )
        assert torch.equal(min_prefix_slow_mask, expected_min_prefix_mask)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL", None)

        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "prefix_confidence"
        os.environ["SPECLINK_SR24_PREFIX_THRESHOLD"] = "0.8"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        prefix_conf_slow_mask = _build_unit_mask(
            req_ids=[f"{req}_prefix_conf_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        prefix_conf_uniform_mask = _build_unit_mask(
            req_ids=[f"{req}_prefix_conf_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        prefix_conf_indexed_mask = _build_unit_mask(
            req_ids=[f"{req}_prefix_conf_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ.pop("SPECLINK_SR24_BATCHED_UNIFORM_DIRECT", None)
        prefix_conf_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_prefix_conf_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        for label, mask in (
            ("uniform-direct", prefix_conf_uniform_mask),
            ("indexed", prefix_conf_indexed_mask),
            ("gpu-count", prefix_conf_gpu_counts_mask),
        ):
            assert torch.equal(prefix_conf_slow_mask, mask), (
                f"{label} prefix-confidence SR24 mask builder diverged "
                "from slow path: "
                f"slow={prefix_conf_slow_mask.to('cpu').tolist()} "
                f"batched={mask.to('cpu').tolist()}"
            )
        expected_prefix_conf_mask = torch.tensor(
            [
                True, True, True, True, True, True,
                False, False, False, False, False, True,
                True, True, True, False, False, True,
            ],
            device=score_device,
        )
        assert torch.equal(prefix_conf_slow_mask, expected_prefix_conf_mask)
        os.environ["SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS"] = "1"
        os.environ["SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS"] = "1"
        direct_cpu_plan = _build_unit_plan(
            req_ids=[f"{req}_prefix_conf_direct_cpu" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        assert isinstance(direct_cpu_plan, VerifyResidualPlan)
        assert torch.equal(
            unwrap_residual_mask(direct_cpu_plan),
            expected_prefix_conf_mask,
        )
        expected_residual_rows = expected_prefix_conf_mask.nonzero(
            as_tuple=False
        ).squeeze(1)
        expected_base_rows = (~expected_prefix_conf_mask).nonzero(
            as_tuple=False
        ).squeeze(1)
        assert direct_cpu_plan.residual_rows is not None
        assert direct_cpu_plan.base_rows is not None
        assert torch.equal(
            direct_cpu_plan.residual_rows.to(device=score_device),
            expected_residual_rows,
        )
        assert torch.equal(
            direct_cpu_plan.base_rows.to(device=score_device),
            expected_base_rows,
        )
        os.environ.pop("SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS", None)
        os.environ.pop("SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS", None)
        os.environ.pop("SPECLINK_SR24_PREFIX_THRESHOLD", None)
        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "fixed_prefix"
        os.environ["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = "2"
        os.environ["SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS"] = "1"
        os.environ["SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS"] = "1"
        fixed_prefix_plan = _build_unit_plan(
            req_ids=[f"{req}_fixed_prefix" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        expected_fixed_prefix_mask = torch.tensor(
            [
                True, True, False, False, False, True,
                True, True, False, False, False, True,
                True, True, False, False, False, True,
            ],
            device=score_device,
        )
        assert isinstance(fixed_prefix_plan, VerifyResidualPlan)
        assert torch.equal(
            unwrap_residual_mask(fixed_prefix_plan),
            expected_fixed_prefix_mask,
        )
        expected_fixed_residual_rows = expected_fixed_prefix_mask.nonzero(
            as_tuple=False
        ).squeeze(1)
        expected_fixed_base_rows = (~expected_fixed_prefix_mask).nonzero(
            as_tuple=False
        ).squeeze(1)
        assert fixed_prefix_plan.residual_rows is not None
        assert fixed_prefix_plan.base_rows is not None
        assert torch.equal(
            fixed_prefix_plan.residual_rows.to(device=score_device),
            expected_fixed_residual_rows,
        )
        assert torch.equal(
            fixed_prefix_plan.base_rows.to(device=score_device),
            expected_fixed_base_rows,
        )
        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "all"
        fixed_prefix_all_plan = _build_unit_plan(
            req_ids=[f"{req}_fixed_prefix_all" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=7,
            device=score_device,
        )
        expected_fixed_prefix_all_mask = torch.tensor(
            [
                True, True, False, False, False, True, True,
                True, True, False, False, False, True, True,
                True, True, False, False, False, True, True,
            ],
            device=score_device,
        )
        assert isinstance(fixed_prefix_all_plan, VerifyResidualPlan)
        if fixed_prefix_all_plan.mask is not None:
            assert torch.equal(
                unwrap_residual_mask(fixed_prefix_all_plan),
                expected_fixed_prefix_all_mask,
            )
        expected_fixed_all_residual_rows = (
            expected_fixed_prefix_all_mask.nonzero(as_tuple=False).squeeze(1)
        )
        expected_fixed_all_base_rows = (
            (~expected_fixed_prefix_all_mask).nonzero(as_tuple=False).squeeze(1)
        )
        if fixed_prefix_all_plan.residual_rows is not None:
            assert torch.equal(
                fixed_prefix_all_plan.residual_rows.to(device=score_device),
                expected_fixed_all_residual_rows,
            )
        if fixed_prefix_all_plan.base_rows is not None:
            assert torch.equal(
                fixed_prefix_all_plan.base_rows.to(device=score_device),
                expected_fixed_all_base_rows,
            )
        if (
            fixed_prefix_all_plan.mask is None
            and fixed_prefix_all_plan.residual_rows is None
            and fixed_prefix_all_plan.base_rows is None
        ):
            assert (
                fixed_prefix_all_plan.fixed_prefix_route is not None
                or fixed_prefix_all_plan.state == "all_residual"
            )
        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "bonus"
        os.environ.pop("SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS", None)
        os.environ.pop("SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS", None)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL", None)
        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "high_confidence"

        os.environ["SPECLINK_SR24_RESIDUAL_BUCKET_SIZE"] = "8"
        os.environ["SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        priority_slow_plan = _build_unit_plan(
            req_ids=[f"{req}_priority_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        priority_slow_mask = unwrap_residual_mask(priority_slow_plan).detach().clone()
        priority_slow = unwrap_residual_priority(priority_slow_plan).detach().clone()
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "1"
        priority_uniform_plan = _build_unit_plan(
            req_ids=[f"{req}_priority_uniform" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_UNIFORM_DIRECT"] = "0"
        priority_indexed_plan = _build_unit_plan(
            req_ids=[f"{req}_priority_indexed" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        priority_gpu_count_plan = _build_unit_plan(
            req_ids=[f"{req}_priority_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        for label, plan in (
            ("uniform-direct", priority_uniform_plan),
            ("indexed", priority_indexed_plan),
            ("gpu-count", priority_gpu_count_plan),
        ):
            mask = unwrap_residual_mask(plan).detach()
            priority = unwrap_residual_priority(plan).detach()
            assert torch.equal(priority_slow_mask, mask), (
                f"{label} priority-bucket mask diverged from slow path: "
                f"slow={priority_slow_mask.to('cpu').tolist()} "
                f"batched={mask.to('cpu').tolist()}"
            )
            assert torch.allclose(priority_slow, priority), (
                f"{label} priority-bucket scores diverged from slow path: "
                f"slow={priority_slow.to('cpu').tolist()} "
                f"batched={priority.to('cpu').tolist()}"
            )

        priority_probe_ids = ["sr24_unit_priority_a", "sr24_unit_priority_b"]
        priority_probe_tokens = torch.ones(
            (2, 3),
            dtype=torch.long,
            device=score_device,
        )
        high_priority_logits = []
        for probs in ([0.95, 0.85], [0.90, 0.90], [0.85, 0.95]):
            logits = torch.empty(2, 5, device=score_device)
            logits.fill_(-100.0)
            logits[:, 0] = 0.0
            for row, prob in enumerate(probs):
                logits[row, 1] = math.log(prob / (1.0 - prob))
            high_priority_logits.append(logits)
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "high_confidence"
        high_priority_plan = _build_unit_plan(
            req_ids=priority_probe_ids,
            draft_token_ids=priority_probe_tokens,
            logits_by_position=high_priority_logits,
            scheduled_tokens_per_req=4,
            device=score_device,
        )
        high_priority = unwrap_residual_priority(high_priority_plan).detach()
        assert high_priority[0] > high_priority[4], (
            "high-confidence priority should prefer higher selected-token "
            f"probability at the same draft position: "
            f"{high_priority.to('cpu').tolist()}"
        )

        prefix_priority_logits = []
        for probs in ([0.95, 0.95], [0.95, 0.50], [0.95, 0.95]):
            logits = torch.empty(2, 5, device=score_device)
            logits.fill_(-100.0)
            logits[:, 0] = 0.0
            for row, prob in enumerate(probs):
                logits[row, 1] = math.log(prob / (1.0 - prob))
            prefix_priority_logits.append(logits)
        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = (
            "prefix_confidence"
        )
        os.environ["SPECLINK_SR24_PREFIX_THRESHOLD"] = "0.1"
        prefix_priority_plan = _build_unit_plan(
            req_ids=[f"{req}_prefix_priority" for req in priority_probe_ids],
            draft_token_ids=priority_probe_tokens,
            logits_by_position=prefix_priority_logits,
            scheduled_tokens_per_req=4,
            device=score_device,
        )
        prefix_priority = unwrap_residual_priority(prefix_priority_plan).detach()
        assert prefix_priority[1] > prefix_priority[5], (
            "prefix-confidence priority should use cumulative prefix "
            f"probability, not only local position: "
            f"{prefix_priority.to('cpu').tolist()}"
        )
        os.environ.pop("SPECLINK_SR24_PREFIX_THRESHOLD", None)
        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "high_confidence"
        os.environ.pop("SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY", None)
        os.environ.pop("SPECLINK_SR24_RESIDUAL_BUCKET_SIZE", None)

        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "all_if_any_low"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        all_if_any_slow_mask = _build_unit_mask(
            req_ids=[f"{req}_allifany_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        all_if_any_batched_mask = _build_unit_mask(
            req_ids=[f"{req}_allifany_batched" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
        )
        all_if_any_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_allifany_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            use_gpu_counts=True,
        )
        assert torch.equal(all_if_any_slow_mask, all_if_any_batched_mask), (
            "batched all-if-any-low SR24 mask builder diverged from slow path: "
            f"slow={all_if_any_slow_mask.to('cpu').tolist()} "
            f"batched={all_if_any_batched_mask.to('cpu').tolist()}"
        )
        assert torch.equal(all_if_any_slow_mask, all_if_any_gpu_counts_mask), (
            "GPU-count all-if-any-low SR24 mask builder diverged from slow path: "
            f"slow={all_if_any_slow_mask.to('cpu').tolist()} "
            f"batched={all_if_any_gpu_counts_mask.to('cpu').tolist()}"
        )
        expected_all_if_any_mask = torch.tensor(
            [
                False, False, False, False, False, True,
                True, True, True, True, True, True,
                True, True, True, True, True, True,
            ],
            device=score_device,
        )
        assert torch.equal(all_if_any_slow_mask, expected_all_if_any_mask)

        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "critical_prefix"
        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "all"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        slow_all_mask = _build_unit_mask(
            req_ids=[f"{req}_all_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=7,
            device=score_device,
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        batched_all_mask = _build_unit_mask(
            req_ids=[f"{req}_all_batched" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=7,
            device=score_device,
        )
        batched_all_gpu_counts_mask = _build_unit_mask(
            req_ids=[f"{req}_all_batched_gpu_counts" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=7,
            device=score_device,
            use_gpu_counts=True,
        )
        assert torch.equal(slow_all_mask, batched_all_mask), (
            "batched SR24 all-non-draft mask builder diverged from slow path: "
            f"slow={slow_all_mask.to('cpu').tolist()} "
            f"batched={batched_all_mask.to('cpu').tolist()}"
        )
        assert torch.equal(slow_all_mask, batched_all_gpu_counts_mask), (
            "GPU-count SR24 all-non-draft mask builder diverged from slow path: "
            f"slow={slow_all_mask.to('cpu').tolist()} "
            f"batched={batched_all_gpu_counts_mask.to('cpu').tolist()}"
        )
        expected_all_mask = torch.tensor(
            [
                True, True, True, True, True, True, True,
                True, True, True, True, False, True, True,
                True, True, True, True, True, True, True,
            ],
            device=score_device,
        )
        assert torch.equal(batched_all_mask, expected_all_mask) or bool(
            batched_all_mask.all().item()
        )
        os.environ["SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY"] = "bonus"
        os.environ["SPECLINK_SR24_EARLY_DENSE_TOKENS"] = "4"
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "0"
        slow_early_mask = _build_unit_mask(
            req_ids=[f"{req}_early_slow" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            generated_lens=[0, 2, 5],
        )
        os.environ["SPECLINK_SR24_BATCHED_MASK_BUILDER"] = "1"
        batched_early_mask = _build_unit_mask(
            req_ids=[f"{req}_early_batched" for req in compare_req_ids],
            draft_token_ids=compare_draft_token_ids,
            logits_by_position=compare_logits_by_position,
            scheduled_tokens_per_req=6,
            device=score_device,
            generated_lens=[0, 2, 5],
        )
        assert torch.equal(slow_early_mask, batched_early_mask), (
            "SR24 early-dense fallback diverged with batched builder enabled: "
            f"slow={slow_early_mask.to('cpu').tolist()} "
            f"batched={batched_early_mask.to('cpu').tolist()}"
        )
        os.environ.pop("SPECLINK_SR24_BATCHED_MASK_BUILDER", None)
        os.environ.pop("SPECLINK_SR24_EARLY_DENSE_TOKENS", None)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY", None)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_EXTRA_AFTER_LOW", None)
        os.environ.pop("SPECLINK_SR24_REDUCE_CPU_SYNC", None)
        os.environ.pop("SPECLINK_SR24_SYNC_MASK_STATE", None)
        os.environ.pop("SPECLINK_SR24_STATIC_MASK_BUFFER", None)

        os.environ["SPECLINK_SR24_MODE"] = "all_corrected"
        os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
        os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "compressed_dense"
        os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = "auto"
        os.environ.pop("SPECLINK_SR24_REQUIRE_GPU_RESIDUAL", None)
        os.environ["SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH"] = "0"
        sparse_model = TinyModel().cuda().to(dtype=torch.float16)
        # PyTorch semi-structured tensors require CUDA fp16/bf16 and dimensions
        # that satisfy backend alignment constraints, so replace the tiny layers.
        sparse_model.layers[0].qkv_proj = nn.Linear(128, 384, bias=False).cuda().half()
        sparse_model.layers[0].o_proj = nn.Linear(128, 128, bias=False).cuda().half()
        sparse_model.layers[0].gate_up_proj = nn.Linear(128, 512, bias=False).cuda().half()
        sparse_model.layers[0].down_proj = nn.Linear(128, 128, bias=False).cuda().half()
        sparse_originals = _module_originals(sparse_model)
        sparse_stats = apply_sr24_from_env(sparse_model, context="unit_sparse")
        assert sparse_stats is not None
        assert sparse_stats["backend"] == "torch_sparse"
        assert sparse_stats["compressed_residual_runtime_on_gpu"] is True
        assert sparse_stats["compressed_residual_non_gpu_modules"] == []
        assert sparse_stats["residual_cuda_module_count"] == 4
        assert sparse_stats["residual_cpu_module_count"] == 0
        assert sparse_stats["residual_extract_cpu_fallback_module_count"] == 0
        for name, module in sparse_model.named_modules():
            if not getattr(module, "_speclink_sr24_enabled", False):
                continue
            x = torch.randn(7, sparse_originals[name].shape[1], device="cuda", dtype=torch.float16)
            expected = F.linear(x, sparse_originals[name])
            actual = sparse_linear_output(module, x)
            assert actual is not None
            assert torch.allclose(actual, expected, atol=5e-2, rtol=5e-2), name
        os.environ["SPECLINK_SR24_RESIDUAL_DEVICE"] = "cpu"
        os.environ["SPECLINK_SR24_REQUIRE_GPU_RESIDUAL"] = "1"
        cpu_guard_model = TinyModel().cuda().to(dtype=torch.float16)
        cpu_guard_model.layers[0].qkv_proj = nn.Linear(128, 384, bias=False).cuda().half()
        cpu_guard_model.layers[0].o_proj = nn.Linear(128, 128, bias=False).cuda().half()
        cpu_guard_model.layers[0].gate_up_proj = nn.Linear(128, 512, bias=False).cuda().half()
        cpu_guard_model.layers[0].down_proj = nn.Linear(128, 128, bias=False).cuda().half()
        try:
            apply_sr24_from_env(cpu_guard_model, context="unit_require_gpu")
        except RuntimeError as exc:
            assert "SPECLINK_SR24_REQUIRE_GPU_RESIDUAL=1" in str(exc)
        else:
            raise AssertionError("SR24 require-gpu residual guard did not fire")
        os.environ.pop("SPECLINK_SR24_REQUIRE_GPU_RESIDUAL", None)
        os.environ.pop("SPECLINK_SR24_RESIDUAL_DEVICE", None)
        os.environ["SPECLINK_SR24_ALL_CORRECTED_DENSE_FASTPATH"] = "1"

        os.environ["SPECLINK_SR24_MODE"] = "selective"
        os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
        os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "dense_rows"
        os.environ["SPECLINK_SR24_REDUCE_CPU_SYNC"] = "1"
        os.environ["SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS"] = "1"
        routed_model = TinyModel().cuda().to(dtype=torch.float16)
        routed_model.layers[0].qkv_proj = nn.Linear(128, 384, bias=False).cuda().half()
        routed_model.layers[0].o_proj = nn.Linear(128, 128, bias=False).cuda().half()
        routed_model.layers[0].gate_up_proj = nn.Linear(128, 512, bias=False).cuda().half()
        routed_model.layers[0].down_proj = nn.Linear(128, 128, bias=False).cuda().half()
        routed_originals = _module_originals(routed_model)
        routed_stats = apply_sr24_from_env(routed_model, context="unit_routed_all")
        assert routed_stats is not None
        assert routed_stats["residual_backend"] == "dense_rows"
        assert routed_stats["route_all_residual_rows"] is True
        route_mask = torch.tensor(
            [True, True, True, False, False, False, False],
            device="cuda",
        )
        for name, module in routed_model.named_modules():
            if not getattr(module, "_speclink_sr24_enabled", False):
                continue
            x = torch.randn(
                7,
                routed_originals[name].shape[1],
                device="cuda",
                dtype=torch.float16,
            )
            dense_out = F.linear(x, routed_originals[name])
            base_out = _semi_structured_linear(x, _sparse_base_weight(module))
            expected = base_out.clone()
            expected[route_mask] = dense_out[route_mask]
            token = begin_verify_context(route_mask)
            try:
                actual = sparse_linear_output(module, x)
            finally:
                end_verify_context(token)
            assert actual is not None
            assert torch.allclose(actual, expected, atol=5e-2, rtol=5e-2), name
            if name.rsplit(".", 1)[-1] not in {"gate_up_proj", "down_proj"}:
                continue
            os.environ["SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION"] = "0.4"
            token = begin_verify_context(route_mask)
            try:
                fallback_actual = sparse_linear_output(module, x)
            finally:
                end_verify_context(token)
                os.environ.pop("SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION", None)
            assert fallback_actual is not None
            assert torch.allclose(
                fallback_actual,
                dense_out,
                atol=5e-2,
                rtol=5e-2,
            ), name
        os.environ.pop("SPECLINK_SR24_REDUCE_CPU_SYNC", None)
        os.environ.pop("SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS", None)

        os.environ["SPECLINK_SR24_MODE"] = "selective"
        os.environ["SPECLINK_SR24_BACKEND"] = "torch_sparse"
        os.environ["SPECLINK_SR24_RESIDUAL_BACKEND"] = "dense_rows"
        os.environ["SPECLINK_SR24_ROW_ROUTED_MLP"] = "1"
        os.environ["SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS"] = "1"
        os.environ["SPECLINK_SR24_TARGET_LEAFS"] = "gate_up_proj,down_proj"
        os.environ["SPECLINK_SR24_RESIDUAL_TARGET_LEAFS"] = "gate_up_proj,down_proj"
        mlp_model = TinyModel().cuda().to(dtype=torch.float16)
        mlp_model.layers[0].gate_up_proj = nn.Linear(
            128, 256, bias=False
        ).cuda().half()
        mlp_model.layers[0].down_proj = nn.Linear(
            128, 128, bias=False
        ).cuda().half()
        mlp_originals = _module_originals(mlp_model)
        mlp_stats = apply_sr24_from_env(mlp_model, context="unit_row_routed_mlp")
        assert mlp_stats is not None
        gate_module = mlp_model.layers[0].gate_up_proj
        down_module = mlp_model.layers[0].down_proj
        assert sparse_backend_active(gate_module), mlp_stats
        assert sparse_backend_active(down_module), mlp_stats
        assert getattr(gate_module, "_speclink_sr24_residual_backend", "") == (
            "dense_rows"
        )
        assert getattr(down_module, "_speclink_sr24_residual_backend", "") == (
            "dense_rows"
        )
        mlp_x = torch.randn(7, 128, device="cuda", dtype=torch.float16)
        mlp_mask = torch.tensor(
            [True, True, False, True, False, False, True],
            device="cuda",
        )
        gate_base = _semi_structured_linear(mlp_x, _sparse_base_weight(gate_module))
        gate_dense = F.linear(
            mlp_x,
            mlp_originals["layers.0.gate_up_proj"],
        )
        gate_expected = gate_base.clone()
        gate_expected[mlp_mask] = gate_dense[mlp_mask]

        def _tiny_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
            return F.silu(gate_up[:, :128]) * gate_up[:, 128:]

        act_expected = _tiny_silu_and_mul(gate_expected)
        down_base = _semi_structured_linear(
            act_expected, _sparse_base_weight(down_module)
        )
        down_dense = F.linear(
            act_expected,
            mlp_originals["layers.0.down_proj"],
        )
        mlp_expected = down_base.clone()
        mlp_expected[mlp_mask] = down_dense[mlp_mask]
        token = begin_verify_context(mlp_mask)
        try:
            mlp_actual = row_routed_mlp_output(
                gate_module,
                down_module,
                mlp_x,
                _tiny_silu_and_mul,
            )
        finally:
            end_verify_context(token)
        assert mlp_actual is not None
        assert torch.allclose(
            mlp_actual,
            mlp_expected,
            atol=5e-2,
            rtol=5e-2,
        )

        os.environ["SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY"] = "fixed_prefix"
        os.environ["SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL"] = "2"
        os.environ["SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH"] = "1"
        os.environ["SPECLINK_SR24_FIXED_BLOCK_INPUT_BUFFER"] = "1"
        fixed_active = 2
        fixed_valid_width = 4
        fixed_scheduled_width = fixed_valid_width + 1
        fixed_prefix = 2
        fixed_rows = fixed_active * fixed_scheduled_width
        fixed_x = torch.randn(fixed_rows, 128, device="cuda", dtype=torch.float16)
        fixed_mask = torch.tensor(
            [True, True, False, False, True,
             True, True, False, False, True],
            device="cuda",
        )
        fixed_residual_rows = fixed_mask.nonzero(as_tuple=False).squeeze(1)
        fixed_base_rows = (~fixed_mask).nonzero(as_tuple=False).squeeze(1)
        fixed_plan = VerifyResidualPlan(
            mask=fixed_mask,
            state="mixed",
            residual_rows=fixed_residual_rows,
            base_rows=fixed_base_rows,
            fixed_prefix_route=FixedPrefixRouteDescriptor(
                active_count=fixed_active,
                scheduled_width=fixed_scheduled_width,
                valid_width=fixed_valid_width,
                prefix=fixed_prefix,
                dense_width=fixed_prefix + 1,
                base_width=fixed_valid_width - fixed_prefix,
            ),
        )
        gate_base = _semi_structured_linear(
            fixed_x, _sparse_base_weight(gate_module)
        )
        gate_dense = F.linear(
            fixed_x,
            mlp_originals["layers.0.gate_up_proj"],
        )
        fixed_gate_expected = gate_base.clone()
        fixed_gate_expected[fixed_mask] = gate_dense[fixed_mask]
        fixed_act_expected = _tiny_silu_and_mul(fixed_gate_expected)
        fixed_down_base = _semi_structured_linear(
            fixed_act_expected, _sparse_base_weight(down_module)
        )
        fixed_down_dense = F.linear(
            fixed_act_expected,
            mlp_originals["layers.0.down_proj"],
        )
        fixed_expected = fixed_down_base.clone()
        fixed_expected[fixed_mask] = fixed_down_dense[fixed_mask]

        fixed_outputs: dict[str, torch.Tensor] = {}
        for output_buffer_enabled in (False, True):
            os.environ["SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER"] = (
                "1" if output_buffer_enabled else "0"
            )
            token = begin_verify_context(fixed_plan)
            try:
                fixed_actual = row_routed_mlp_output(
                    gate_module,
                    down_module,
                    fixed_x,
                    _tiny_silu_and_mul,
                )
            finally:
                end_verify_context(token)
            assert fixed_actual is not None
            assert torch.allclose(
                fixed_actual,
                fixed_expected,
                atol=5e-2,
                rtol=5e-2,
            ), f"fixed_block_output_buffer={output_buffer_enabled}"
            fixed_outputs[str(output_buffer_enabled)] = fixed_actual.detach().clone()
        assert torch.allclose(
            fixed_outputs["False"],
            fixed_outputs["True"],
            atol=5e-2,
            rtol=5e-2,
        )
        os.environ.pop("SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER", None)
        os.environ.pop("SPECLINK_SR24_FIXED_BLOCK_INPUT_BUFFER", None)
        os.environ.pop("SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH", None)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_MIN_PREFIX_RESIDUAL", None)
        os.environ.pop("SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY", None)

        os.environ["SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR"] = "1"
        token = begin_verify_context(mlp_mask)
        try:
            down_only_actual = row_routed_down_output(down_module, act_expected)
        finally:
            end_verify_context(token)
        assert down_only_actual is not None
        assert torch.allclose(
            down_only_actual,
            mlp_expected,
            atol=5e-2,
            rtol=5e-2,
        )
        os.environ.pop("SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR", None)
        os.environ.pop("SPECLINK_SR24_ROW_ROUTED_MLP", None)
        os.environ.pop("SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS", None)
        os.environ.pop("SPECLINK_SR24_TARGET_LEAFS", None)
        os.environ.pop("SPECLINK_SR24_RESIDUAL_TARGET_LEAFS", None)

    print("speclink_sr24_correctness=ok")


if __name__ == "__main__":
    main()
