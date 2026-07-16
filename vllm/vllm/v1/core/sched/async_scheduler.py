# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self._pending_new_request_outputs: set[str] = set()
        self._last_scheduled_req_ids: set[str] = set()

    def _defer_running_request(self, request: Request) -> bool:
        req_id = request.request_id
        if req_id in self._pending_new_request_outputs:
            return True

        if request.spec_token_ids and req_id not in self._last_scheduled_req_ids:
            # Async speculative inputs live only in the worker's immediately
            # preceding batch. If a request skipped that batch, its draft
            # placeholders can no longer be resolved there.
            if request.num_output_placeholders > 0:
                return True
            request.spec_token_ids = []
        return False

    def _release_pending_new_request(self, request: Request) -> None:
        self._pending_new_request_outputs.discard(request.request_id)
        request.spec_token_ids = []

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        new_req_ids = {
            request.req_id for request in scheduler_output.scheduled_new_reqs
        }
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders
            if self.num_spec_tokens and req_id in new_req_ids:
                # A refill request can complete prefill while another async batch is
                # still in flight. Do not consume its GPU-only sampled/draft tokens
                # until this prefill output has reached the scheduler.
                self._pending_new_request_outputs.add(req_id)
        self._last_scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.request_id in self._pending_new_request_outputs:
            self._release_pending_new_request(request)

        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
