# SR24 Actionable Slowdown Breakdown

This note answers the current pivot question: before trying more controller
sweeps, where is SR24 slow?

## Direct Answer To The New Breakdown Direction

The right next step is a fixed seven-part breakdown, not another threshold or
controller sweep. The current evidence says:

| part | should measure | current status | current read |
| --- | --- | --- | --- |
| scheduler / mask build | per-step residual mask construction, bucket-row selection, and request routing wall time | clean path is sub-ms: latest prefix5 `speclink_t08` has `0.556ms/step`; sync-heavy diagnostic can show `6.270ms/step` | do not optimize from sync-heavy numbers; clean mask build is not the first bottleneck |
| base sparse linear | 2:4 sparse base Linear time, especially `gate_up_proj=16-31` and `down_proj` scopes | latest diagnostic `speclink_t08` sparse base is `1.208ms/call`; `gate_up_proj=16-31` is `1.239ms/call` | this is the largest GPU-side component in the mixed path |
| residual correction | dense-row correction GEMM or dense overwrite time | latest diagnostic residual dense correction is `0.136ms/call`; `gate_up_proj=16-31` correction is `0.172ms/call` | secondary for bucket32, but it is pure extra work after sparse base has already run |
| gather/scatter | `index_select`, `index_add_`, `index_copy_`, bucket assembly | latest diagnostic `0.027ms/event` | visible but not the current first-order bottleneck |
| routing statistics | draft residual rows, non-draft residual rows, bucket fill ratio | latest diagnostic: draft residual/base `8212/1404`, non-draft residual/base `1202/1823`, bucket fill `31.49/32` | too many draft rows still require residual correction; bucket is full, so this is not empty-bucket waste |
| CUDA Graph | FULL/NONE graph steps for dense/base-only/t08 | clean prefix5: base-only `{"FULL":62,"NONE":2}`, `speclink_t08` `{"FULL":56,"NONE":8}` | graph coverage is healthy enough in the clean row; graph loss is not the main reason here |
| GPU util | whether the GPU is idle or just doing inefficient work | clean prefix5: dense `78.875%`, base-only `79.667%`, `speclink_t08` `79.556%` | GPU is busy; the problem is useful-work efficiency, not lack of occupancy |

The short conclusion is: current SR24 is slow because the mixed operator does

```text
sparse base for all rows
then dense correction / overwrite for selected rows
then row assembly
```

Rows that need dense correction have already paid the sparse-base cost. This is
why `base_only_24` can be fast while `speclink_t08` fails to preserve enough of
that gain.

The immediate optimization target should therefore be an operator/routing
change that avoids duplicate work for corrected rows, not another broad policy
sweep:

1. Dense/residual rows should not first run through sparse base.
2. Base rows should run only the 2:4 sparse path.
3. Assembly should be one low-overhead, graph-compatible path.
4. Any new candidate must report the seven fields above before quality or
   throughput claims.

Primary source:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_prefix5_20260629_023311/seven_part_report/report.md
```

Earlier source:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_currentcandidate_20260629/seven_part_report/report.md
```

Related current summary:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_CURRENT_SLOWDOWN_BREAKDOWN_20260629.md
```

## Short Conclusion

The current SR24 path is not primarily slow because of accepted length, CUDA
Graph misses, GPU underutilization, or CPU-side mask construction in the clean
path. The latest prefix5 run shows the distinction clearly: full-batch decode
has a speedup, but total tok/s is still worse than dense in the one-wave
end-to-end measurement.

The main slowdown is useful-work efficiency inside the mixed sparse plus
residual operator:

1. 2:4 sparse base Linear is faster than dense when run alone.
2. Mixed SR24 still computes sparse base for many rows that later need dense
   residual correction or dense overwrite.
3. Residual correction adds extra dense GEMMs plus gather/scatter/assembly.
4. At realistic residual-row fractions, those extra operations erase most of
   the sparse-base win.

So the next useful direction is not another broad scheduler/controller sweep.
It should be an operator-level change that avoids duplicate work for corrected
rows and reduces row assembly overhead.

## 2026-06-29 Prefix5 Seven-Part Breakdown

Run root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_prefix5_20260629_023311
```

Primary outputs:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_prefix5_20260629_023311/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_prefix5_20260629_023311/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_prefix5_20260629_023311/component_microbench/summary.md
```

Config: Qwen3/EAGLE3, `math_reasoning`, bs=64, K=8, max_tokens=128,
`low_confidence`, threshold 0.6, forced prefix residual length 5, target leafs
`gate_up_proj,down_proj`, residual bucket size 32.

Clean serving:

| method | total tok/s | full-batch tok/s | TPOT ms | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense_baseline | 2237.872 | 3032.665 | 20.703 | 1.429 | 78.875% | n/a |
| base_only_24 | 2703.991 | 5079.227 | 12.802 | 2.409 | 79.667% | `{"FULL": 62, "NONE": 2}` |
| speclink_t08 | 1940.635 | 3669.383 | 18.167 | 2.400 | 79.556% | `{"FULL": 56, "NONE": 8}` |

Read: `base_only_24` is a real upper bound and is not slow here. `speclink_t08`
keeps similar acceptance and GPU utilization, but residual correction makes
TPOT much closer to dense than to base-only. Full-batch decode is faster than
dense, but the one-wave total tok/s is lower, so future reports should keep
both metrics.

Component timing from the eager/instrumented diagnostic row:

| part | speclink_t08 diagnostic value | read |
| --- | ---: | --- |
| clean scheduler/mask wall | 0.556 ms/step | sub-ms; not the main clean-path bottleneck |
| exact diagnostic scheduler/mask wall | 6.270 ms/step | sync-heavy diagnosis only |
| base sparse linear | 1.208 ms/call | the dominant GPU-side cost in mixed SR24 |
| residual dense correction | 0.136 ms/call | secondary for bucket32, but still extra work |
| gather/scatter | 0.027 ms/event | not first-order in this run |
| bucket fill | 31.49 / 32.00 active rows | bucket is full; this is not empty-bucket waste |
| draft residual/base rows | 8212 / 1404 | most draft rows still need residual correction |
| non-draft residual/base rows | 1202 / 1823 | bonus/non-draft correction is also visible |

The earlier clean summary field `bucket_active_rows_per_call=0` was misleading
for low-sync runs. With GPU-side breakdown counts, the bucket is almost fully
active: 31.49 active rows out of 32 candidate rows per bucket call.

Microbench confirms the operator picture:

- Gate/up shape `512 x 28672 x 4096`: sparse base is about `0.65x` dense, but
  the current mixed proxy reaches dense parity by residual fraction 0.125 and
  becomes slower at 0.25+.
- Down shape `512 x 4096 x 14336`: sparse base is about `0.57x` dense; the
  mixed proxy stays useful until about 0.25 residual fraction, then slows down.
- Bucket dense-copy microbench is attractive in isolation, but serving still
  pays sparse base for rows that are later corrected.

## Seven-Part Read

| part | what was checked | current evidence | read |
| --- | --- | --- | --- |
| scheduler / mask build | per-step residual mask, row bucket, bucket build | clean SR24 mask wall about `0.409 ms/step`; diagnostic sync-heavy paths can show `8-44 ms/step` | clean scheduler cost is not the main bottleneck; sync-heavy diagnostics should not be used as throughput evidence |
| base sparse linear | sparse base GEMM, especially gate_up layers | row-routed gate_up sparse base about `0.985 ms/call`; microbench base sparse is about `0.57-0.66x` of dense | sparse base itself has real headroom |
| residual correction | dense rows correction / dense overwrite | row-routed gate_up dense correction about `0.164 ms/call` for 16 dense rows; all-corrected gate_up dense correction about `0.482 ms/call` | correction is small only when residual rows are small; it becomes a large cost at high correction rate |
| gather/scatter | `index_select`, `index_add_`, `index_copy_`, assembly | row-routed gather/scatter about `0.034 ms/call`; all-corrected gather/scatter about `0.021 ms/call` | not the first bottleneck alone, but still part of the duplicated mixed path |
| routing stats | residual/base rows, bucket fill | draft residual fraction around `0.53-0.57`; non-draft residual fraction around `0.49-0.60`; bucket fill about `0.976-0.997` | too many draft rows still need residual correction; bucket is full, so this is not empty-bucket waste |
| CUDA Graph | FULL/NONE graph steps | clean `speclink_t08`: `{"FULL": 94, "NONE": 2}`; FULL fraction about `0.979` | graph coverage is healthy for clean serving |
| GPU util | average/peak utilization | clean `speclink_t08` avg GPU util about `90.17%`, peak `100%` | GPU is busy; the issue is inefficient work, not idle GPU |

## Throughput Evidence

Clean serving, bs=64:

| method | full-batch tok/s | total tok/s | speedup vs dense | avg GPU util | graph read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | `3439.701` | `2591.040` | `1.000x` | `85.385%` | dense graph has many non-FULL steps |
| base_only_24 | `4307.870` | `2972.244` | `1.252x` | `90.182%` | base-only upper bound is real |
| speclink_t08 | `3971.578` | `2645.645` | `1.155x` | `90.167%` | graph coverage is good |

This says base-only sparsity can be fast, but the mixed residual-corrected path
does not preserve enough of that win.

## Microbench Evidence

Representative isolated Linear shapes show the same pattern:

- Gate/up shape `512 x 28672 x 4096`: sparse base is about `0.65x` dense.
- Down shape `512 x 4096 x 14336`: sparse base is about `0.57x` dense.
- Once residual fraction reaches `0.25-0.50`, the current mixed path is often
  near dense or slower than dense.

That means the bottleneck is not that 2:4 sparsity is useless. The bottleneck is
the current two-pass mixed implementation:

```text
sparse base for all rows
then dense correction / overwrite for selected rows
then row assembly
```

## What To Do Next

The next implementation target should be a fused or low-overhead routed
operator, not another policy sweep:

1. For rows selected as dense/residual, avoid doing sparse base first.
2. For base rows, run only 2:4 sparse base.
3. Assemble output with one low-overhead path, ideally avoiding two full
   `index_copy_` operations.
4. Keep CUDA Graph compatibility; graph-none variants were already too slow.
5. Keep CPU-sync reduction as an ablation guardrail, not the main optimization.

The current `row_routed_mlp_output()` is semantically close to this direction,
but earlier probes showed the naive row split is still not enough. The likely
next useful prototype is a fused/packed row-routed MLP path that reduces launch
count and assembly cost rather than simply calling separate dense and sparse
GEMMs with `index_copy_`.

## 2026-06-29 Row-Routed MLP Contiguous Probe

I tested a narrower version of the operator idea: when dense/residual rows are a
contiguous prefix or suffix, avoid `index_select` plus final `index_copy_` and
instead use direct slices plus `torch.cat`.

Microbench output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_contiguous_probe_20260629
```

Serving outputs:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_contiguous_fastpath_captureguard_bs64_math128_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_rowrouted_nocontig_bs64_math128_20260629
```

Microbench read:

| bucket | dense graph ms | exact-down Triton graph ms | contiguous graph ms | contiguous / dense | read |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.8610 | 0.8676 | 0.8553 | 0.99x | tiny improvement, not a 1.2x path |
| 32 | 0.8671 | 0.8164 | 0.8100 | 0.93x | useful but below dense/1.2 target |
| 64 | 0.8678 | 0.8110 | 0.8044 | 0.93x | useful but below dense/1.2 target |
| 128 | 0.8684 | 0.7490 | 0.7402 | 0.85x | closest case, still short of 1.2x target |

Serving read, bs=64, math_reasoning, max_tokens=128, 64 requests:

| variant | total tok/s | full-batch tok/s | avg accepted draft tokens/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| prefix5 baseline, no row-routed MLP | 2014.508 | 3607.854 | 2.355583 | 79.250% | `{"FULL": 49}` |
| row-routed MLP, no contiguous fastpath | 802.499 | 1157.404 | 0.004714 | 91.850% | `{"FULL": 49}` |
| row-routed MLP, contiguous fastpath enabled | 813.361 | 1161.252 | 0.066597 | 86.600% | `{"FULL": 49}` |

Decision:

1. The contiguous assembly optimization is real in isolation, but too small to
   rescue the current row-routed MLP path.
2. The live row-routed MLP path itself is not quality-safe for this candidate:
   accepted draft tokens collapse, so the verifier rejects almost everything.
3. The contiguous check cannot be captured blindly in CUDA Graphs. Graph replay
   does not rerun Python route checks, and dynamic bucket rows are not guaranteed
   to stay a prefix/suffix. The implementation therefore guards this fastpath
   out during CUDA Graph capture.
4. Do not make row-routed MLP the main path. A future fused operator must either
   consume dynamic row indices directly, or the scheduler must provide a static
   prefix/suffix routing contract that remains true across graph replay.

## 2026-06-29 Down-Only Row Routing Probe

I added a default-off experimental switch:

```text
SPECLINK_SR24_ROW_ROUTED_DOWN_LINEAR=1
--sr24-row-routed-down-linear
```

This path is narrower than full row-routed MLP. It keeps the normal activation
path, then routes only `down_proj`: dense down for residual rows, sparse down
for base rows, and output assembly. The goal was to avoid the old
`down_proj` two-pass pattern where residual rows first pay sparse base and are
then overwritten by dense correction.

Correctness smoke:

```text
conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Result:

```text
speclink_sr24_correctness=ok
```

Serving A/B, Llama-3.1-8B, `math_reasoning`, bs64, K=8, max new tokens 128,
same currentcandidate settings except for the new flag:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_downlinear_control_bs64_math128_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_downlinear_currentcandidate_bs64_math128_20260629
```

| variant | method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| control, no down routing | dense_baseline | 2150.945 | 3011.313 | 1.401 | 86.625% | same-run dense |
| control, no down routing | speclink_t08 | 2021.249 | 3171.180 | 1.645 | 86.750% | baseline for this ablation |
| down-only routing | dense_baseline | 1747.760 | 2860.902 | 1.426 | 56.400% | noisy dense row |
| down-only routing | speclink_t08 | 2027.150 | 3159.199 | 1.630 | 86.750% | no improvement over control |

Linear breakdown smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_downlinear_breakdown_smoke_20260629
```

The new branch did execute:

```text
row_routed_down_calls = 160
row_routed_down_dense_rows = 2560
row_routed_down_base_rows = 24032
row_routed_down_base_sparse_cuda_ms = 0.890 ms/call
row_routed_down_dense_gemm_cuda_ms = 0.099 ms/call
row_routed_down_index_copy_cuda_ms = 0.009 ms/call
```

Decision: keep this as a default-off diagnostic, not as the next default speed
path. It proves that simply moving `down_proj` from two-pass correction to
Linear-level row routing is not enough. The base side is still a large sparse
GEMM over many split rows, and the accepted draft length did not improve. The
next speed candidate must reduce/fuse the row-routed sparse-base work itself or
reduce residual-row demand while keeping quality stable.
