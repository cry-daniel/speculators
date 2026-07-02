# SR24 Grouped Queue Next Design - 2026-07-02

## Why This Is Needed

Current `lossy_prefix2_rowrouted_mlp_operator_guard` already avoids duplicate
work for the row classes the user cares about:

- important verifier rows go through dense MLP only,
- unimportant verifier rows go through 2:4 sparse MLP only,
- no dense correction is applied to rows that already used the sparse branch.

The remaining slowdown is system-level underfill. A single compact verifier
block at bs8/16/32 does not feed enough dense-important rows or 2:4 sparse rows
to make the split MLP faster than dense. The refreshed offline queue plan is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_queue_live_plan_front28_policyoverride_20260702
```

It says no-wait local MLP speedup is `1.000x/1.000x/1.000x/1.071x` for
bs8/16/32/64. Reaching the `1.2x` local MLP target needs bounded cross-step
waiting: bs8=`15`, bs16=`7`, bs32=`15`, bs64=`15` verifier blocks on the current
math trace.

## Correct Insertion Point

Do not put the real queue only inside `vllm/vllm/speclink_sr24.py` Linear hooks.
At that point the verifier forward is already running; delaying a verifier
block there would either return wrong logits or force dense fallback.

The live queue must be scheduler/model-runner level:

1. Build the fixed-prefix route descriptor in `speclink_sr24.build_verify...`
   as today.
2. Before launching the target verifier forward, decide whether this compact
   verifier block should run now or enter a bounded SR24 queue.
3. Dequeue compatible blocks when their combined dense/base rows reach the
   scheduler policy target.
4. Run one grouped/fused mixed MLP shape for queued blocks, or dense fallback
   on timeout/tail.
5. Preserve per-request dependencies: no request can sample/accept the queued
   draft tokens until its verifier logits have been produced.

## Data Format

Use a fixed-capacity verifier-block descriptor:

```text
[block_id, request_id, scheduled_width=K+1, prefix, valid_width]
dense rows: prefix draft rows + verifier bonus row
base rows: remaining draft rows
```

The operator-facing tensors should be packed by group:

```text
dense_input: [sum dense_rows, hidden]
base_input:  [sum base_rows, hidden]
output:      [sum active_rows, hidden]
route table: block-local output slices back to request rows
```

For the current K=8/prefix2 policy, the target effective bs64 shape is:

```text
grouped_dense_rows = 192
grouped_base_rows  = 384
```

## Fallback Policy

The queue must be bounded, not lossless-at-any-cost:

- If compatible work reaches target rows before `max_wait_blocks`, use grouped
  mixed MLP.
- If it times out, run dense verifier for those blocks.
- If it is a drain/tail block, run dense verifier.
- If the grouped operator is unavailable, run dense verifier.

This is essential for bs8/16 request drain behavior; otherwise queueing can
improve local MLP time while hurting end-to-end latency.

## Immediate Validation

Use the offline plan as the replay oracle:

```text
queue_plan.csv
queue_plan.jsonl
queue_plan_summary.csv
```

Minimum live checks:

1. Default-off correctness: no output/token changes when queue env is off.
2. Queue trace parity: with queue env on but operator still fallback-only,
   emitted live actions should match offline `group/fallback` counts.
3. Dense-fallback safety: timeout/tail fallback must produce the same tokens as
   dense EAGLE3 on a short greedy smoke.
4. Throughput only after the grouped operator is real; queue bookkeeping alone
   is not a speed claim.

## Current Shadow Implementation

`vllm/vllm/speclink_sr24.py` now has a default-off live shadow queue:

```text
SPECLINK_SR24_GROUPED_QUEUE_SHADOW=1
SPECLINK_SR24_GROUPED_QUEUE_MAX_WAIT_BLOCKS=15
SPECLINK_SR24_GROUPING_TRACE_PATH=/path/to/speclink_sr24_grouping_trace.jsonl
```

It consumes the same `sr24_grouping_opportunity` events already emitted by the
SR24 route planner and writes `sr24_grouped_queue_shadow_decision` events with
`action=group|fallback` and reasons `target_reached`, `timeout_underfilled`, or
`tail_underfilled`.

This shadow path does not delay verifier execution and does not change logits.
It is only a live parity tool for the future scheduler queue. The grouped
operator and request-dependency scheduling are still not implemented.

Analyze shadow traces with:

```bash
conda run -n spec python examples/evaluate/eval-guidellm/scripts/analyze_sr24_grouped_queue_shadow.py \
  --trace-glob '/path/to/**/speclink_sr24_grouping_trace.jsonl' \
  --offline-plan-summary-csv /path/to/queue_plan_summary.csv \
  --output-root /path/to/shadow_analysis
```

The script writes `shadow_summary.csv`, `shadow_offline_delta.csv`, and
`shadow_report.md`. It has a no-GPU smoke path and was checked on a synthetic
three-block trace: two compatible blocks grouped, the remaining tail block
fell back dense.
