# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _greedy_token_logprob_blocks_kernel(
    local_token_ids_ptr,
    local_max_ptr,
    local_sum_exp_ptr,
    logits_ptr,
    logits_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + req_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    block_max, block_token_idx = tl.max(logits, axis=0, return_indices=True)
    safe_block_max = tl.where(block_max == float("-inf"), 0.0, block_max)
    block_sum_exp = tl.sum(tl.exp(logits - safe_block_max), axis=0)
    offset = req_idx * tl.num_programs(1) + block_idx
    tl.store(local_token_ids_ptr + offset, block_idx * BLOCK_SIZE + block_token_idx)
    tl.store(local_max_ptr + offset, block_max)
    tl.store(local_sum_exp_ptr + offset, block_sum_exp)


@triton.jit
def _greedy_token_logprob_reduce_kernel(
    token_ids_ptr,
    logprobs_ptr,
    local_token_ids_ptr,
    local_max_ptr,
    local_sum_exp_ptr,
    num_blocks,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = block < num_blocks
    offset = req_idx * num_blocks + block
    block_max = tl.load(
        local_max_ptr + offset,
        mask=mask,
        other=float("-inf"),
    )
    max_val, max_block_idx = tl.max(block_max, axis=0, return_indices=True)
    block_sum_exp = tl.load(local_sum_exp_ptr + offset, mask=mask, other=0.0)
    sum_exp = tl.sum(block_sum_exp * tl.exp(block_max - max_val), axis=0)
    token_id = tl.load(
        local_token_ids_ptr + req_idx * num_blocks + max_block_idx
    )
    tl.store(token_ids_ptr + req_idx, token_id)
    tl.store(logprobs_ptr + req_idx, -tl.log(sum_exp))


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        PADDED_TOPK=triton.next_power_of_2(num_logprobs),
    )
    return logprobs


def compute_greedy_token_ids_and_logprobs(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return greedy token IDs and their normalized logprobs without softmax."""
    batch_size, vocab_size = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_token_ids = logits.new_empty(
        (batch_size, num_blocks), dtype=torch.int32
    )
    local_max = logits.new_empty((batch_size, num_blocks), dtype=torch.float32)
    local_sum_exp = logits.new_empty(
        (batch_size, num_blocks), dtype=torch.float32
    )
    token_ids = logits.new_empty(batch_size, dtype=torch.int64)
    logprobs = logits.new_empty(batch_size, dtype=torch.float32)
    _greedy_token_logprob_blocks_kernel[(batch_size, num_blocks)](
        local_token_ids,
        local_max,
        local_sum_exp,
        logits,
        logits.stride(0),
        vocab_size,
        BLOCK_SIZE=block_size,  # type: ignore
    )
    _greedy_token_logprob_reduce_kernel[(batch_size,)](
        token_ids,
        logprobs,
        local_token_ids,
        local_max,
        local_sum_exp,
        num_blocks,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
    )
    return token_ids, logprobs


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    if num_logprobs > 0:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)

    # NOTE(woosuk): Here, to save GPU memory, we do not materialize the full
    # logprobs tensor. Instead, we only compute and return the logprobs of
    # the topk + 1 tokens.
    logprobs = compute_token_logprobs(logits, logprob_token_ids)
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )
