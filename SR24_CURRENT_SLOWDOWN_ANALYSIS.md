# SR24 Current Slowdown Analysis

This note summarizes the current SR24/SpecLink slowdown diagnosis from the
latest local artifacts. It intentionally separates clean serving measurements
from diagnostic breakdown rows, because CUDA-event timing and exact routing
counters add synchronization overhead.

## 2026-06-28 CPU-Sync Ablation and Focused Breakdown

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_critical_fallback050_earlydense_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_predicted_fallback050_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_focused_seven_part_breakdown_bs64_math128_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_focused_speclink_eager_breakdown_bs64_math128_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix4_clean_bs64_math256_20260628_goal/report.md
```

The dense-fallback experiment confirmed one real CPU-side issue. The old
fallback path used `residual_mask.sum().item()` once per step to decide
`mask_state`, which forced a GPU-to-CPU synchronization. In the fallback050
clean run this showed up as about `25.68ms/step` in
`scheduler_mask_state_wall_cpu_ms_per_step`, while the actual batched mask
builder was only about `0.31ms/step`.

I changed the selective fallback decision to use a conservative no-sync row
count from the batched mask builder. It counts rows forced residual by policy
such as prefix rows and bonus rows. If that lower bound already exceeds the
dense-fallback threshold, the step is promoted to the exact all-residual
fastpath without reading the GPU mask back to CPU. The same patch also makes
the matrix runner report effective all/no-residual routing for those promoted
steps instead of leaving residual draft tokens blank.

Clean before/after for the same bs64, math, K=4, max-tokens 256 fallback050
shape:

| run | dense full-batch tok/s | SR24 full-batch tok/s | SR24/dense | scheduler mask-state wall | scheduler mask wall | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| old GPU-sum fallback | `4942.642` | `4701.227` | `0.951x` | `~25.68ms/step` | `~25.68ms/step` dominated by sync | bad CPU sync |
| predicted no-sync fallback | `4921.788` | `4747.879` | `0.965x` | `0.000ms/step` | `~0.232ms/step` | sync fixed, still not faster than dense |

This means reducing the CPU synchronization is necessary, but not sufficient.
Once the sync is gone, all-residual fallback is still dense-equivalent for the
touched modules and remains slightly below dense EAGLE3 because it keeps SR24
mask/proposer overhead while giving up sparse-base savings.

Focused seven-part read:

| part | current measurement | read |
| --- | ---: | --- |
| scheduler / mask build | clean predicted fallback `~0.232ms/step`; exact-routing diagnostic inflates CPU routing to `~32ms/step` | clean low-sync path is sub-ms; exact-routing numbers are diagnostic only |
| base sparse Linear | eager diagnostic `2.336ms/call` aggregate; `gate_up_proj=16-31` `2.724ms/call`; `down_proj=8-15` `1.561ms/call` | largest localized GPU-side cost in mixed steps |
| residual correction | eager diagnostic `0.170ms/call` aggregate; gate_up `0.182ms/call`; down `0.146ms/call` | much smaller than sparse base but additive |
| gather/scatter | `gather_input_index_select` `0.018ms/call`; bucket dense copy `0.0048ms/call` | not the main bottleneck in the measured shape |
| routing statistics | eager exact diagnostic: residual draft `13828/13828`, residual non-draft `3457/7244`, one mixed step and sixty all-residual steps | quality-safe critical-prefix routing often becomes very conservative |
| CUDA Graph | clean fallback run `{"FULL":219,"NONE":40}`; eager diagnostic `{"NONE":64}` by design | graph coverage is not the current clean-run blocker |
| GPU util | clean predicted fallback avg `91.29%`; eager diagnostic `72.14%` | clean run is GPU-busy; eager diagnostic underutilization comes from instrumentation/eager mode |

The updated conclusion is narrower than before: ordinary CPU-side sync was a
real bug and is now removed for fallback decisions, but it does not unlock the
target speedup. The remaining speed path is not "promote more steps to dense";
that just becomes dense with overhead. The path with upside is still the mixed
operator path: reduce how often the quality-safe controller marks rows
residual, or make the mixed sparse-base plus dense-row correction operator
cheaper/fused enough that it keeps more of the base-only sparse headroom.

Practical next experiments:

1. Keep exact-routing and CUDA-event breakdown out of clean serving runs.
2. Compare the quality-safe `critical_prefix` controller against `fixed_prefix`
   and lower-score-overhead variants, because DLM selected-probability scoring
   and conservative prefix routing can erase the sparse gain. The first clean
   `fixed_prefix4_bucket16_directcslt` check did not recover speed:
   dense/SR24 full-batch tok/s was `4964.673/4661.306`, so score-path removal
   alone is not enough.
3. Prioritize a fused or MLP-level mixed operator for `gate_up_proj=16-31`.
   The measured sparse base cost is dominated by `gate_up`, while residual
   dense rows and gather/scatter are secondary.
4. Treat all-residual fallback as a correctness/safety guard only, not as the
   performance path.

Follow-up K correction and row-routed result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_k8_clean_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_k8_clean_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_gateup_k8_eager_breakdown_bs64_math128_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_gateup_bucket64_k8_clean_bs64_math256_20260628_goal/report.md
```

The speed target should be read at EAGLE3 K=8. Earlier fallback050 checks that
omitted `--eagle3-k 8` used the runner default K=4; with `prefix4`, K=4
degenerates into all draft rows using residual correction and is not the right
mixed-operator test.

K=8 clean baseline:

| method | full-batch tok/s | total tok/s | read |
| --- | ---: | ---: | --- |
| dense EAGLE3 | `3149.263` | `2599.101` | baseline |
| `base_only_24` | `3749.429` | `3183.587` | `1.191x` full-batch upper bound in this run |
| `speclink_t08` criticalprefix4 bucket16 | `3438.111` | `2832.689` | `1.092x`; below target |

I added a gated gate_up-only row-routed path behind
`--sr24-row-routed-mlp`. This is needed because the quality-safe scope uses
`gate_up_proj=16-31` and `down_proj=8-15`, so the older full MLP row-routed
path never matched a layer where both gate_up and down were SR24. The new path
does match and computes dense gate_up for selected rows plus sparse gate_up for
base rows before assembling the gate_up output.

The result is not the final speed path:

| variant | full-batch tok/s | total tok/s | read |
| --- | ---: | ---: | --- |
| bucket16 gate_up row-routed | `3477.607` | `2944.526` | small improvement over `3438.111`, still only `1.096x` vs dense |
| bucket64 gate_up row-routed | `3444.199` | `2742.896` | more dense rows does not help |

Focused K=8 breakdown with the new gate_up-only path shows why. The path now
hits (`row_routed_gate_up_calls=944`), but bucket16 has only `16` dense rows per
gate_up call out of about `538` total rows. It saves too little sparse work and
adds dense GEMM plus assemble:

| component | avg ms/call | read |
| --- | ---: | --- |
| row-routed gate_up base sparse | `0.950` | still dominates |
| row-routed gate_up dense GEMM | `0.164` | additive |
| row-routed gate_up assemble | `0.043` | nontrivial for small dense bucket |
| residual dense GEMM on remaining down path | `0.133` | still additive |

Therefore the next speed path is not simple row routing over a small bucket.
Either the controller must make fewer rows exact while preserving accuracy, or
the operator has to fuse/pack the mixed gate_up computation so the dense-row
correction and assemble overhead are much lower.

Reporting update: the seven-part reducers now understand the
`row_routed_gate_up_*` counters emitted by the gate-up-only mixed path. This
matters because the previous reducer recognized full row-routed MLP counters
but not the current quality-safe gate-up-only path, so base/correction/assemble
costs could be hidden in the standard report. The refreshed offline report is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_gateup_breakdown_report_refresh_20260628/report.md
```

For the existing row-routed gate-up diagnostic artifact, the corrected report
shows:

| part | value | read |
| --- | ---: | --- |
| row-routed gate_up base sparse | `0.950ms/call`, `521.847` base rows/call out of `537.847` total rows/call | sparse base is still the dominant cost |
| row-routed gate_up dense GEMM | `0.164ms/call`, `16.000` dense rows/call | correction is smaller but additive |
| row-routed gate_up gather/scatter | `0.056ms/call`, including `0.043ms` index-copy | not the first bottleneck, but large enough to erase small bucket wins |
| routing | draft residual/base `13382/11554`, non-draft residual/base `3117/3788`, bucket fill `0.989` | many rows still need residual protection |

This tightens the operator conclusion: bucket16 row routing does skip dense
GEMM for most rows, but it still launches sparse gate_up over about `522` rows
per call and then adds dense GEMM plus assembly for the protected bucket. The
measured clean row remains only `1.096x` full-batch over dense, so a simple
Python/Torch row-routed composition is not the requested `1.2x` path.

Avoid-mixed controller probes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_avoidmixed_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_batchall_avoidmixed_stats_bs64_math128_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_request_allifany_fallback06_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_request_allifany_fallback06_quality_gsm8k20_20260628_goal/report.md
```

| candidate | evidence | read |
| --- | --- | --- |
| `batch_all_if_any_low`, stats off | dense/SR24 full-batch `3189.572/2991.770`, total `2747.177/2365.361`, accepted `1.749/1.753` | slower than dense; no useful speed signal |
| `batch_all_if_any_low`, stats on | SR24 residual draft/non-draft fraction `1.000/1.000`, CUDA Graph `{"NONE":84}`, scheduler mask wall `32.174ms/step` | at bs64 one risky request makes the whole batch all-residual; too coarse |
| request-level `all_if_any_low` + fallback06 | dense/SR24 full-batch `3173.547/3322.561`, total `2796.617/2601.492`, accepted `1.740/2.144` | better than batch-level but only `1.047x` full-batch and total still slower |
| request-level `all_if_any_low` + fallback06 GSM8K-20 | dense/SR24 exact `0.700/0.700`, pair reg/imp `0/0`, SR24 residual draft fraction `1.000` | paired-clean only because it effectively corrects every draft row |

The avoid-mixed probe is useful but not sufficient. Batch-level gating collapses
to all-residual at bs64. Request-level gating can pass a small paired accuracy
gate, but the current fallback setting also promotes the measured quality run
to all-residual (`SR24 draft residual=1.0`), so it is closer to a dense
correctness guard than a sparse speed path. This rules out a coarse all/no
residual controller as the main route to `1.2x`; the remaining viable paths are
a fused/packed mixed operator, or a finer request/token controller that keeps a
large fraction of rows truly base-only while passing paired accuracy.

## Goal Continuation: Base-Only and All-Corrected Read, 2026-06-28

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_slowdown_diagnosis_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_earlydense_enabled_smoke_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_earlydense_clean_smoke_20260628_goal/report.md
```

`base_only_24` is not slow because the model accepts fewer draft tokens or
because the GPU is idle. In the two current clean references it accepts more
draft tokens per step than dense EAGLE3 and keeps GPU utilization high:

| root | base-only full-batch tok/s | speedup vs dense | base accepted/step | dense accepted/step | GPU util | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| current candidate graph-count | `3660.767` | `1.153x` | `2.150` | `1.719` | `93.190%` | `{"FULL":143,"NONE":49}` | acceptance and GPU util are ok |
| corrected fast candidate | `4307.108` | `1.222x` | `2.205` | `1.736` | `89.364%` | `{"FULL":94,"NONE":2}` | acceptance, GPU util, and graph coverage are ok |

The old `all_corrected_24` operator-ablation path with
`--no-sr24-all-corrected-dense-fastpath` was still doing duplicate useful work:
`base_sparse_linear_calls=648` plus `residual_dense_full_gemm_cuda_ms` in the
small bs8 diagnostic smoke. That explains the slow all-corrected rows: they
were measuring sparse base plus full dense correction, not a practical
optimized all-residual implementation.

I added an explicit optimized hook path:

```text
SPECLINK_SR24_FULL_RESIDUAL_EARLY_DENSE=1
```

and the matrix-runner flag:

```text
--sr24-full-residual-early-dense
```

This keeps SR24 Linear hooks attached, but when a step/module is known to be
all-residual and a dense weight is available, it returns the dense Linear before
dispatching sparse base. The diagnostic smoke confirms the path change:
`full_residual_early_dense_calls=576`, with no `base_sparse_linear_calls` in
the extracted breakdown. The clean smoke also shows the path can use CUDA Graph
(`{"FULL":23,"NONE":2}` for the tiny bs8 run). The tiny fixed-request total
tok/s is not a final throughput claim, but full-batch tok/s improved over the
same-run dense smoke (`800.325` diagnostic full-batch with breakdown and
`776.075` clean full-batch vs dense `716.103`/`712.268`).

Current `speclink_t08` quality boundary: `criticalprefix4_bucket16_directcslt`
is paired-clean on the GSM8K-30 gate, while more aggressive bucket-copy/Triton
variants can regress paired accuracy. The next speed work should preserve the
paired-clean critical-prefix semantics, then reduce the mixed-path cost. The
remaining `1.2x` target likely needs a graph-safe fused/packed mixed operator
or a substantially better row-importance signal; simply lowering the global
bucket or using the known-regressing Triton bucket-copy variants is not a safe
path.

## Current Read: Why It Is Slow

Latest current-candidate refresh, 2026-06-28 18:50:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_candidate_breakdown_bs64_math256_20260628_combined/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_candidate_graph_counts_bs64_math256_20260628/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, bs64, EAGLE3 K=8, max new tokens
256, `criticalprefix4_bucket16_directcslt`, `gate_up_proj=16-31;down_proj=8-15`,
`dense_rows@cuda`, bucket16, bucket dense copy, direct cuSPARSELt, and dynamic
auto CUDA Graph.

Clean serving shows that the GPU is busy and sparse headroom exists:

| method | total tok/s | full-batch tok/s | full speedup | accepted draft/step | GPU util |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense EAGLE3 | `2609.686` | `3173.852` | `1.000x` | `1.696` | `94.960%` |
| `base_only_24` | `3109.799` | `3706.963` | `1.168x` | `2.114` | `92.048%` |
| `speclink_t08` | `2861.536` | `3486.572` | `1.099x` | `2.177` | `94.043%` |

The graph-count refresh gives runtime CUDA Graph modes: dense
`{"FULL":57,"NONE":166,"PIECEWISE":1}`, base-only
`{"FULL":143,"NONE":49}`, all-corrected
`{"FULL":57,"NONE":166,"PIECEWISE":1}`, and `speclink_t08`
`{"FULL":150,"NONE":42}`. So `speclink_t08` is not failing because graph
coverage disappears, although the remaining `NONE` steps should stay as a
guardrail.

The eager instrumented row localizes the cost: `speclink_t08` sparse base is
`0.992ms/call` aggregate (`1.101ms/call` for gate/up16-31 and `0.774ms/call`
for down8-15), dense correction is `0.163ms/call`, and gather/scatter is only
`0.017ms/call`. Routing still marks many rows residual:
draft residual/base `13599/11713`, non-draft residual/base `3164/3788`, draft
residual fraction `0.537`, bucket fill `0.982`.

Current conclusion: the path is slow because it recovers only part of the
base-only sparse headroom. The main cost is useful GPU work shape, not idle
GPU: sparse base Linear is paid over hundreds of rows, then dense correction is
paid for a large protected subset. Scheduler exact-routing timing in the
instrumented row is sync-heavy and not a clean-serving CPU number; gather/scatter
is too small to explain the gap. The next useful direction is a fused/packed
mixed operator or a quality-safe reduction in residual rows.

Latest corrected refresh, 2026-06-28:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/seven_part_report/report.md
```

This is the current first reference. It uses Llama-3.1-8B,
`math_reasoning`, bs64, EAGLE3 K=8, max new tokens 256, and the current
graph-capable bucketed `speclink_t08` candidate:
`critical_prefix@0.6,prefix4,extra1`,
`gate_up_proj=16-31;down_proj=8-15`, bucket12, direct cuSPARSELt, and bucket
dense copy. It intentionally does not enable residual-bucket priority or
direct-position bucket routing; the earlier priority/direct-position refresh is
now a superseded diagnostic condition.

| part | latest value | read |
| --- | ---: | --- |
| clean throughput | dense `3524.307` full-batch tok/s, `2599.815` total tok/s; SR24 `3902.787` full-batch tok/s, `2625.737` total tok/s | current `speclink_t08` is `1.107x` full-batch but only `1.010x` total, still below the `1.2x` target |
| base-only upper bound | `base_only_24` `4307.108` full-batch tok/s, `2955.659` total tok/s | sparse headroom exists; mixed correction consumes most of it |
| accepted length | dense `1.736`, `base_only_24` `2.205`, SR24 `2.103` accepted draft tokens/step | not acceptance collapse |
| scheduler / mask | clean `0.289ms/step`; row bucket `0.064ms/step`; bucket build `0.062ms/step` | sub-ms clean scheduler cost |
| base sparse Linear | diagnostic base sparse `1.041ms/call`; `gate_up_proj=16-31` `1.071ms/call`; `down_proj=8-15` `0.980ms/call` | largest localized GPU-side cost |
| residual correction | diagnostic dense correction `0.161ms/call`; bucket rows/call `12` | smaller than sparse base but additive |
| gather/scatter | `0.014ms/call` | secondary in this row |
| routing | draft residual/base `2928/2464`, draft residual fraction `0.543`, bucket fill `0.989` | many draft rows still require residual protection |
| CUDA Graph | `{"FULL":126,"NONE":2}` | graph coverage is healthy |
| GPU util | SR24 avg `91.154%`, peak `99%` | GPU is busy; the problem is useful-work efficiency |

The fresh answer is therefore: current SR24 is not absolutely slower than dense
in this corrected row, but it is far below the available base-only headroom.
The mixed verify operator pays sparse base plus dense-row correction for too
many rows. It is not primarily a long CPU wait, a generated-token acceptance
collapse, CUDA Graph loss, or an idle GPU problem. The next useful work is a
graph-safe fused/packed mixed operator or a controller/routing signal that
reduces residual rows without paired accuracy loss.

Follow-up ablation: `predicted_full_accept` for non-draft/bonus rows is now
supported by the batched GPU mask builder and passed the SR24 correctness test,
but it is not a speed path on the corrected bucket12 shape. Current-code clean
serving measured `bonus` at `2683.694` total / `3917.754` full-batch tok/s with
`2.171` accepted draft tokens/step, versus `predicted_full_accept` at
`2638.930` total / `3903.531` full-batch tok/s with `2.119` accepted
draft tokens/step. Removing bonus-row residual correction reduced the reported
non-draft residual fraction to `0`, but it did not improve throughput.

CPU-sync follow-up: the breakdown orchestrator now forwards the
compressed-residual cache/prewarm/Triton diagnostic flags, and the same
single-entry protocol was used for a same-condition CPU-sync ablation of the
corrected bucket12 fast candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fast_candidate_cpu_sync_ablation_b12_bs64_math256_20260628/
```

The clean `low_sync_stats_on` row measured SR24 `3966.966` full-batch tok/s
and `2705.051` total tok/s versus same-root dense `3435.870` full-batch and
`2321.907` total, with `2.187` accepted draft tokens/step, `90.917%` GPU util,
`0.338ms/step` scheduler-mask wall time, and CUDA Graph
`{"FULL":94,"NONE":2}`. `low_sync_stats_off` was not faster
(`3861.572` full-batch), while `sync_heavy` fell to `1960.373` full-batch with
CUDA Graph `{"NONE":128}` and `59.864%` GPU util. Therefore CPU-sync reduction
is necessary to avoid the bad path, but the remaining gap to `1.2x` is not
ordinary runtime statistics or mask-state synchronization.

Compressed-dense status: code inspection confirms that
`compressed_dense@cuda` keeps the mask bytes, compressed residual values, and
cached/prewarmed residual dense tensors on CUDA. Its negative all-corrected
performance should be read as duplicated GPU work (`sparse base + residual
GEMM/materialization`), not as accidental CPU residual computation.

Superseded refresh kept for history:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_bs64_math_k8_20260628/seven_part_report/report.md
```

That row used priority/direct-position routing and measured dense/SR24
`3422.973/3340.622` full-batch tok/s (`0.976x`); do not use it as the current
fast-candidate reference.

Older read kept for history:

结论先说清楚：当前 `speclink_t08` 慢，不是因为 accepted length 明显掉了，
也不是因为 CUDA Graph 大面积失效，或者 GPU 长时间空闲。clean serving 行里
GPU util 仍然很高，CUDA Graph 覆盖也接近 dense/base-only。真正的问题是
mixed sparse-base + residual correction 的有效计算形态不好：先对一批 row
跑 sparse base Linear，再对相当多 residual row 额外跑 dense correction。
当 residual row 比例不低时，sparse base 的收益被 correction 和额外 kernel
开销吃掉。

用户建议的 breakdown 对应当前证据如下：

| part | what to measure | current value | read |
| --- | --- | ---: | --- |
| scheduler / mask build | 每 step 构造 residual mask、bucket rows 的时间 | clean `speclink_t08` `0.969ms/step`; low-sync CPU-sync ablation `0.959ms/step` | clean path 是 sub-ms 级，不是首要瓶颈；sync-heavy diagnostic 会放大到十几或几十 ms，不能作为 serving 结论 |
| base sparse linear | `gate_up_proj=16-31` sparse base 时间 | diagnostic `1.012ms/call`, about `268` rows/call | 当前最大的局部 GPU-side 成本 |
| residual correction | dense_rows correction 的 dense GEMM 时间 | diagnostic `0.174ms/call`, bucket rows `32/call` | 单次小于 sparse base，但它是额外叠加成本 |
| gather/scatter | `index_select`, `index_add_`, bucket assembly | diagnostic `0.016ms/call` | 目前太小，不足以解释主差距 |
| routing statistics | draft/non-draft residual row、bucket fill ratio | draft residual/base `6376/1544`, draft residual frac `0.805`, bucket fill `0.979` | 过多 draft rows 仍需要 residual 保护，是 controller/quality 侧的限制 |
| CUDA Graph | dense/base_only/t08 的 FULL/NONE graph step | `speclink_t08` `{"FULL":62,"NONE":2}`; `base_only_24` `{"FULL":62,"NONE":2}` | graph 覆盖健康，不是当前 clean run 首因 |
| GPU util | 是否满载或小 kernel underutilization | `speclink_t08` avg `88.111%`, peak `100%`; dense avg `80.625%` | GPU 很忙；慢是 busy 但 useful work 效率低 |

所以后续优化方向应该换成两条主线：

1. **先做明确 breakdown gate**：每个新 candidate 都必须同时报告 clean
   throughput、scheduler/mask、base sparse、residual correction、gather/scatter、
   routing、CUDA Graph、GPU util。不要只看最终 tok/s，也不要把 sync-heavy
   diagnostic 行当成真实 serving。
2. **优先解决 mixed operator / residual row 问题**：如果不能把 residual row
   比例降下来，就需要 fused/packed GPU operator；单纯清理 Python 统计或继续
   sweep controller，不太可能达到 `1.2x`。

## 2026-06-28 Fresh Seven-Part Pivot Read

Use this result first for the current "measure where it is slow before another
controller sweep" read:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_fresh_bs64_math_k8_20260628/seven_part_report/report.md
```

The same artifacts were re-summarized in the explicit user-pivot breakdown
below, using the requested seven buckets: scheduler/mask build, sparse base
Linear, residual correction, gather/scatter, routing statistics, CUDA Graph,
and GPU utilization.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_explicit_slowdown_breakdown_user_pivot_20260628/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size
64, max new tokens 128 for clean serving, SR24 scoped to
`gate_up_proj=16-31`, `critical_prefix`, threshold `0.3`, minimum residual
prefix `6`, extra-after-low `1`, dense-row residual correction on CUDA,
residual bucket size `32`, bucket priority on, bonus priority `0`, draft
position priority scale `1`, and CUDA Graph enabled for clean serving.

Clean serving rows:

| method | full-batch tok/s | total tok/s | same-root speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3023.082` | `2186.780` | `1.000x` | `1.395` | `80.625%` | n/a |
| `base_only_24` | `3244.196` | `2231.409` | `1.073x` | `1.547` | `86.750%` | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | `2903.555` | `1872.259` | `0.960x` | `1.433` | `88.111%` | `{"FULL":62,"NONE":2}` |

Current seven-part diagnosis:

| part | current evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` wall `0.969ms/step`, batched builder `0.790ms/step`, row bucket `0.107ms/step`, bucket build `0.105ms/step`, row indices `0.001ms/step` | sub-ms clean cost; not the first bottleneck |
| base sparse linear | diagnostic `gate_up_proj=16-31` sparse base `1.012ms/call`, about `268` rows/call | largest localized GPU-side cost |
| residual correction | diagnostic dense-row correction `0.174ms/call`, bucket rows `32/call` | smaller than sparse base, but additive |
| gather/scatter | diagnostic `0.016ms/call` | too small to explain the main gap |
| routing statistics | diagnostic draft residual/base `6376/1544`, non-draft residual/base `990/1823`, draft residual fraction `0.805`, bucket fill `0.979`, actual/requested bucket fraction `0.169` | many draft rows still need residual protection; residual rows are the main controller-side limiter |
| CUDA Graph | clean `speclink_t08` `{"FULL":62,"NONE":2}` | graph coverage is healthy in this run |
| GPU util | clean `speclink_t08` avg `88.111%`, peak `100%` | GPU is busy; slowdown is inefficient useful work, not idle GPU |

Operator microbench:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_fresh_bs64_math_k8_20260628/seven_part_report/operator_microbench.csv
```

The gate/up Linear shape (`512/28672/4096`) has real sparse-base headroom:
base sparse is about `0.65-0.66x` dense under CUDA Graph. The current mixed
operator loses that headroom once residual rows are non-trivial: at residual
fraction `0.125`, current mixed is already `1.03x` dense; at `0.5`, it is
`1.52x` dense. The down Linear shape is more forgiving at low residual
fractions, but also loses when residual fraction reaches `0.5`.

Conclusion: the current slow path is not primarily accepted-length collapse,
ordinary CPU mask construction, CUDA Graph loss, or GPU underutilization. The
slow part is the mixed useful-work shape: sparse base work is paid for the
active rows, then dense correction is added for a large residual-row set. The
next implementation work should either reduce residual rows without losing the
paired accuracy gate, or replace the two-pass sparse-base-plus-dense-correction
operator with a fused/packed GPU path. Another scheduler-sync cleanup or
gather/scatter-only rewrite is lower priority unless a new breakdown moves the
numbers.

### Current Optimization Direction

The next pass should be breakdown-first:

| part | current state | next action |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` is about `0.969ms/step`; the old early-dense Python loop has already been removed from the clean path | keep it as a guardrail; do not make generic CPU cleanup the main task unless this returns to multi-ms |
| base sparse Linear | `gate_up_proj=16-31` sparse base is about `1.012ms/call`, the largest localized GPU-side cost | optimize or avoid the two-pass base computation in mixed rows |
| residual correction | dense-row correction is about `0.174ms/call` for 32-row buckets | reduce corrected rows or fuse/pack correction with the base path |
| gather/scatter | about `0.016ms/call` in the current safe row | lower priority than base/correction; revisit after the two-pass cost drops |
| routing statistics | draft residual/base is `6376/1544`, so about `80.5%` of draft rows are still protected | controller quality gate must reduce residual rows, or the operator must tolerate high residual fractions |
| CUDA Graph | clean `speclink_t08` has `62/64` FULL graph steps | graph is currently healthy; keep checking it for every new candidate |
| GPU util | clean `speclink_t08` avg is `88.111%`, peak `100%` | this is not idle GPU; the issue is inefficient busy work |

The important implication is that `base_only_24` proving sparse headroom is not
enough. `speclink_t08` must either keep most rows base-only without quality
loss, or run a fused/packed sparse-plus-residual operator. Otherwise it pays
sparse base work and residual correction on too many rows and can be slower
than dense while the GPU still looks busy.

### CPU-Sync Ablation

Fresh focused CPU-sync ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_speclink_t08_gateup16_bs64_20260628_goal_continue/
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch
64, max new tokens 128, `speclink_t08`, `gate_up_proj=16-31`,
`critical_prefix`, threshold `0.3`, minimum residual prefix `6`,
extra-after-low `1`, dense-row residual correction on CUDA, residual bucket
size `32`, bucket priority on, static mask buffer on, and CUDA Graph allowed.

| variant | total tok/s | full-batch tok/s | accepted draft/step | GPU util avg/peak | scheduler mask ms/step | graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| low-sync stats off | `2035.618` | `2934.282` | `1.535` | `84.5/100.0%` | n/a | `{"FULL":49}` |
| low-sync stats on | `2059.235` | `2984.749` | `1.504` | `84.8/100.0%` | `0.959` | `{"FULL":62,"NONE":2}` |
| low-sync GPU-count breakdown | `1999.945` | `2908.451` | `1.549` | `83.0/99.0%` | diagnostic `43.147` | `{"FULL":62,"NONE":2}` |
| sync mask-state | `2017.916` | `2924.411` | `1.544` | `84.2/100.0%` | diagnostic `42.981` | `{"FULL":62,"NONE":2}` |
| sync-heavy | `1138.599` | `1540.734` | `1.389` | `64.0/92.0%` | `16.777` | `{"NONE":64}` |

Read: the sync-heavy path is clearly invalid for performance work: it loses
CUDA Graph coverage and cuts full-batch throughput roughly in half. The
low-sync variants cluster around `2.9-3.0k` full-batch tok/s, so removing CPU
sync is a required guardrail, but it is not enough to create the requested
`1.2x` speedup over dense. The remaining gap is still the mixed sparse-base
plus residual-correction operator and routing shape, not ordinary CPU-side
statistics.

### Route-All Residual Rows Probe

Existing `route_all_residual_rows` was tested as a direct attempt to avoid
computing sparse base on rows that will be corrected by dense residual rows:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_all_residual_rows_probe_gateup16_bs64_20260628_goal_continue/
```

Run shape matched the CPU-sync ablation except that `speclink_t08` enabled
`--sr24-route-all-residual-rows`, `--sr24-route-all-skip-bucket`,
`--sr24-route-min-dense-rows 32`, `--sr24-route-min-base-rows 32`, and
`--sr24-triton-route-assembly`.

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util avg/peak | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2186.332` | `3021.394` | `1.395` | `80.4/99.0%` | n/a |
| route-all `speclink_t08` | `1549.691` | `2297.109` | `1.416` | `72.6/97.0%` | `{"NONE":84}` |

Read: the current route-all implementation is not a speed path. It preserves a
normal accepted length, but it loses CUDA Graph coverage and lowers GPU
utilization. It is useful as evidence for the implementation direction: the
right target is not another eager routed-row composition, but a graph-safe
packed/fused mixed Linear or a controller that keeps residual rows low enough
for the existing graph-safe bucket path.

Two existing graph/controller presets were also checked:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_cap0_graph_probe_bs64_math128_20260628_goal_continue/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speed_tradeoff_down16_base_bs64_math128_20260628_goal_continue/
```

| candidate | total tok/s | full-batch tok/s | speedup vs same-root dense full-batch | accepted draft/step | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `gateup_cap0_graph_probe` | `2012.119` | `2919.397` | `0.966x` | `1.537` | `{"FULL":62,"NONE":2}` | graph coverage is healthy but scope is too small/inefficient |
| `speed_tradeoff_down16_base` | `1914.760` | `2921.372` | `0.966x` | `1.464` | `{"NONE":64}` | expanded down scope loses graph coverage and is not useful |

Read: simply keeping graph-safe buckets or adding down-proj base-only scope is
not enough. The next useful controller shape should avoid mixed two-pass
correction entirely: run exact dense for residual-bearing steps and sparse base
only for no-residual steps, then measure whether enough no-residual steps exist
to beat dense without quality loss.

### 2026-06-28 Follow-Up Optimization Probes

Two focused follow-up probes were run after the seven-part breakdown to test
whether simple mixed-path mitigations are worth pursuing.

#### Adaptive Dense Fallback, Low-Sync

Artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_adaptive_densefallback_low_sync_probe_bs64_math128_20260628_095747/
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, batch size 64,
max new tokens 128, same `critical_prefix` controller as the current
`speclink_t08` row, with `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK=1`.

| method | full-batch tok/s | total tok/s | speedup vs dense | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3024.928` | `2188.659` | `1.000x` | `83.375%` | n/a |
| adaptive fallback `speclink_t08` | `2884.326` | `1992.439` | `0.954x` | `86.500%` | `{"FULL":62,"NONE":2}` |

Runtime stats show that the fallback was active:
`adaptive_dense_fallback_calls=832`, `adaptive_dense_fallback_rows=278624`.
This is too aggressive for the capped-bucket path: because exact residual row
counts are intentionally not synchronized in low-sync serving, a full bucket is
treated as a high-residual step and many gate/up calls fall back to dense. That
removes the sparse-base headroom instead of improving it.

Read: adaptive dense fallback is not a speed path in the current low-sync
bucketed controller. It is useful only as a conservative diagnostic guard.

#### Triton Bucket Dense GEMM/Scatter

Clean serving artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_dense_gemm_probe_bs64_math128_20260628_095747/
```

Instrumented artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_dense_gemm_instrumented_bs64_math64_20260628_095747/
```

Clean serving result:

| method | full-batch tok/s | total tok/s | speedup vs dense | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3017.798` | `2183.938` | `1.000x` | `82.500%` | n/a |
| Triton bucket `speclink_t08` | `2805.204` | `1959.862` | `0.930x` | `86.333%` | `{"FULL":62,"NONE":2}` |

The short instrumented row localized the problem:

| component | value |
| --- | ---: |
| base sparse Linear | `0.973ms/call` |
| bucket Triton dense GEMM/scatter | `0.316ms/call` for `32` rows/call |
| regular residual dense GEMM diagnostic | `0.153ms/call` |
| gather/scatter event average | `0.007ms/event` |
| draft residual/base rows | `5977/1431` |
| bucket fill | `0.978` |

Read: the current Triton bucket correction kernel is slower than the ordinary
small dense-row correction path. It does not solve the mixed operator problem.
The next viable kernel work is not a small scatter/GEMM wrapper around the
existing path; it needs a fused/packed mixed Linear or a controller that avoids
mixed correction for most rows.

### All-Corrected Backend Probe

Fresh focused all-corrected backend probes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_sparse_backend_probe_large_shapes_20260628_goal_continue/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_triton_tile_sweep_large_shapes_20260628_goal_continue/summary.md
```

Run shape: representative Llama MLP Linear shapes, rows `512`,
`gate_up_proj` shape `512/28672/4096`, and `down_proj` shape
`512/4096/14336`. These are microbenchmarks, not end-to-end serving runs.

Device check:

| tensor | device |
| --- | --- |
| compressed mask CUDA copy | `cuda:0` |
| residual values CUDA copy | `cuda:0` |
| cached dense residual weight | `cuda:0` |
| predecoded residual positions | `cuda:0` |

So the current compressed-dense path is not accidentally CPU-side once the CUDA
variants are selected. The explicit CPU materialization baseline is much slower
and remains only a diagnostic control.

All-corrected graph candidates:

| shape rows/out/in | dense graph ms | base sparse graph ms | best exact all-corrected graph path | best exact ms | current vs dense time | needed reduction for 1.2x |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `512/28672/4096` | `0.5401` | `0.3557` | `all_sparse_graph` | `0.7606` | `1.41x` | `1.69x` |
| `512/4096/14336` | `0.2928` | `0.1679` | `all_sparse_graph` | `0.3375` | `1.15x` | `1.38x` |

The cached compressed-dense residual path is GPU-resident but slower than the
two-sparse-GEMM exact path:

| shape rows/out/in | all sparse graph ms | compressed cached graph ms | read |
| --- | ---: | ---: | --- |
| `512/28672/4096` | `0.7606` | `1.0607` | materialized dense residual is not a speed path |
| `512/4096/14336` | `0.3375` | `0.4835` | materialized dense residual is not a speed path |

The current hand-written residual 2:4 Triton algorithm is also not a speed path.
After sweeping tile sizes, the best large-shape all-corrected Triton rows were:

| shape rows/out/in | best kernel/tile | dense graph ms | best all-corrected Triton graph ms | all/dense | needed reduction for 1.2x |
| --- | --- | ---: | ---: | ---: | ---: |
| `512/28672/4096` | `pos_tiled`, `M=8,N=16,G=32` | `0.5451` | `21.6977` | `39.80x` | `47.76x` |
| `512/4096/14336` | `pos_tiled`, `M=8,N=16,G=32` | `0.2985` | `10.7931` | `36.15x` | `43.38x` |

This rules out "just tune the current Triton block sizes" for
`all_corrected_24`. A useful all-corrected implementation needs a genuinely
fused packed operator that combines base and complementary residual work without
launching two independent GEMMs or doing irregular scalar residual gathers. If
that kernel is not implemented, `all_corrected_24` should remain a negative
control and `speclink_t08` should focus on keeping corrected-row fractions low
enough that the current two-pass path still wins.

### Prefix6 Quality Check

The latest GSM8K-50 paired run for the more conservative prefix6 setting is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_earlydense_prefix6_paired_gsm8k50_20260628/report.md
```

It reports dense EAGLE3 and `speclink_t08` both at `0.7200` exact match, but
the paired rows still contain 2 dense-correct/SR24-wrong and 2
dense-wrong/SR24-correct samples. Treat this as aggregate-score cancellation,
not a clean quality pass. Prefix6 can reduce some replay divergence, but it is
not a final correctness solution by itself.

Follow-up replay for docs 2, 15, and 20:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_exact_route_replay_docs2_15_20_20260628/
```

This replay compared dense EAGLE3, the current prefix6 bucket32 path, and
`--sr24-route-all-residual-rows`. Docs 2 and 20 were token-identical across all
three variants. Doc 15 differed from dense at token 87 for both SR24 variants,
but both SR24 variants produced the same final correct answer (`125`). The
route-all variant did not move the output closer to dense than bucket32, so this
small replay does not support "capped bucket alone causes the quality issue".
The quality gate still needs stable paired evaluation, but the next speed work
should not assume that route-all residual rows is a free correctness fix.

### Rejected Row-Routed MLP Probe

The MLP-level row routing idea was tested as a clean serving probe:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_mlp_clean_probe_bs64_math128_20260628/clean_serving/report.md
```

Configuration: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch
64, max new tokens 128, all MLP leafs (`gate_up_proj,down_proj`),
prefix6/early-dense64/bucket32, `--sr24-row-routed-mlp`, and mixed CUDA Graph
enabled.

| method | full-batch tok/s | total tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2505.594` | `1539.124` | `1.397` | `59.818%` | n/a |
| row-routed `speclink_t08` | `1219.109` | `1103.084` | `0.017` | `92.467%` | `{"FULL":126,"NONE":2}` |

The candidate is not usable in this form. It keeps CUDA Graph coverage and high
GPU utilization, but target/draft agreement collapses, so the throughput loss is
dominated by poor speculative acceptance rather than CPU overhead. Keep
row-routed MLP off the main optimization path unless a later correctness fix
can recover dense-equivalent verifier behavior.

## 2026-06-28 Current Read

### Gate-Up16 Focused Breakdown After Direction Pivot

The latest focused breakdown for the "measure where it is slow first" direction
is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_gateup16_bs64_math_k8_20260628/seven_part_report_direct/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size
64, max new tokens 128 for clean serving, SR24 only on
`gate_up_proj=16-31`, `high_confidence@0.3`, dense-row residual correction on
CUDA, low-sync counters for clean serving. A separate direct instrumented run
uses CUDA events and exact routing counters; its tok/s is diagnostic only.

Clean serving rows:

| method | full-batch tok/s | total tok/s | same-root full speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2763.202` | `2368.916` | `1.000x` | `1.402` | `88.643%` | n/a |
| `base_only_24` | `2937.606` | `2475.326` | `1.063x` | `1.561` | `91.538%` | `{"FULL":97,"NONE":31}` |
| `speclink_t08` | `2294.100` | `1904.859` | `0.830x` | `1.453` | `91.235%` | `{"NONE":128}` |
| `all_corrected_24` dense fastpath | `2780.837` | `2400.316` | `1.006x` | `1.427` | `90.786%` | dense-equivalent control |

Seven-part read from the clean and instrumented rows:

| part | evidence | diagnosis |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` mask wall `0.453ms/step`; batched builder `0.391ms/step`; row bucket `0.004ms/step`; bucket build `0.002ms/step` | clean scheduler/mask work is sub-ms and is not the primary slowdown |
| base sparse linear | instrumented `gate_up_proj=16-31` sparse base `0.570ms/call`, about `532` rows/call | large GPU-side cost on the mixed path |
| residual correction | instrumented dense correction `0.570ms/call` on the same `gate_up_proj=16-31` rows | correction is as large as the sparse base in this diagnostic shape |
| gather/scatter | instrumented gather/select/scatter wrapper `0.083ms/call` | visible, but smaller than base sparse plus correction |
| routing statistics | clean non-draft residual/base `6744/5736`; instrumented draft residual/base `9300/6804`, non-draft residual/base `2013/3735`, draft residual fraction `0.577` | too many rows still take the residual path for a two-pass operator |
| CUDA Graph | clean `speclink_t08` `{"NONE":128}` versus base-only `{"FULL":97,"NONE":31}` | this run is graph-limited for dynamic mixed `speclink_t08` |
| GPU util | clean `speclink_t08` avg `91.235%`, peak `100%`; diagnostic avg `75.714%` with profiling overhead | not an idle-GPU problem; it is inefficient useful work plus graph loss |

Interpretation: for this focused gate-up16 path, the first bottleneck is no
longer CPU mask construction. The clean scheduler cost is below 0.5ms/step.
`speclink_t08` is slow because every mixed step falls to CUDA Graph `NONE`, and
the Linear path pays both sparse-base work and dense residual correction for a
large residual-row fraction. Exact-routing profiling is useful for attribution
but adds a large sync-heavy wall time, so it should remain a diagnostic
ablation, not the clean serving path.

### Early-Dense Batched Builder Fix

The first implementation pass after the user-pivot breakdown removes the large
Python request-routing loop from the all-MLP `critical_prefix` path with
`early_dense_tokens=64`. `vllm/vllm/speclink_sr24.py` now passes per-request
generated lengths into the Triton batched mask-builder kernels, so the early
dense guard is handled in the same fixed-shape builder instead of falling back
to the per-request Python loop. The local correctness check was updated to
cover non-uniform generated lengths.

Verification:

```text
conda run -n spec python -m py_compile vllm/vllm/speclink_sr24.py examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Both checks passed. The same bs64/math/K=8 clean serving shape after this fix
is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_earlydense_batched_builder_bs64_math_k8_cleanstats_20260628/report.md
```

| method | full-batch tok/s | total tok/s | full speedup | total speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3149.236` | `2205.545` | `1.000x` | `1.000x` | `1.406` | `86.375%` | n/a |
| `speclink_t08` | `4307.435` | `2171.978` | `1.368x` | `0.985x` | `2.329` | `83.375%` | `{"FULL":62,"NONE":2}` |

The scheduler-side bottleneck moved as intended:

| counter | before fix | after fix |
| --- | ---: | ---: |
| mask build wall | `40.230ms/step` | `0.978ms/step` |
| request routing loop | `39.994ms/step` | `0.000ms/step` |
| batched mask builder | `0.000ms/step` | `0.792ms/step` |
| row-index bucket | `0.161ms/step` | `0.112ms/step` |
| residual bucket | `0.159ms/step` | `0.110ms/step` |
| mixed row indices | `0.001ms/step` | `0.001ms/step` |

So the prior first bottleneck is resolved. The fixed path improves full-batch
throughput from `3649.985` to `4307.435` tok/s, about `1.18x` over the previous
all-MLP t08 run and `1.368x` over same-root dense. The short 64-request total
throughput is still slightly below dense (`0.985x`), so the next throughput
question should separate full-window throughput from fill/drain/request-window
effects with a longer run before changing the operator again.

The paired GSM8K-50 accuracy gate is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_earlydense_batched_builder_paired_gsm8k50_20260628/report.md
```

It reports dense EAGLE3 exact match `0.7400` and `speclink_t08` `0.7200`.
Paired counts are 2 dense-correct / t08-wrong and 1 dense-wrong / t08-correct
over 50 samples. This is not a clean quality pass yet; it is a small-sample
warning that the all-MLP residual policy still needs a broader accuracy gate or
a stronger row/layer protection rule before it can be treated as final.

Current read after the fix: the old `scheduler / mask build` row is no longer
the dominant slowdown. The remaining high-value questions are (1) whether total
throughput crosses dense on longer/full-window runs, and (2) whether the
operator-side sparse-base plus residual-correction cost and all-MLP quality risk
can be reduced without losing the `1.3x+` full-batch headroom.

### Historical User Pivot Breakdown: All-MLP Triton Override With Early Dense Guard

This older user-requested breakdown for the "measure where it is slow first"
direction is kept as a historical all-MLP/early-dense route, not as the current
fast-candidate reference:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_k8_combined_breakdown_20260628/report.md
```

It combines a clean-stats serving run and a component-timing profile run for
Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size 64, max new
tokens 128, `max_model_len=2048`, all MLP leafs
(`gate_up_proj,down_proj`), `dense_rows` residual correction on CUDA,
`critical_prefix@0.6,prefix4,extra1`, `early_dense_tokens=64`, bucket32,
bucket priority, Triton bucket override, and mixed CUDA Graph enabled.

Clean-stats rows:

| method | full-batch tok/s | total tok/s | full speedup | total speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3025.258` | `2185.404` | `1.000x` | `1.000x` | `1.399` | `80.125%` | n/a |
| `speclink_t08` | `3649.985` | `1976.243` | `1.207x` | `0.904x` | `2.396` | `84.556%` | `{"FULL":62,"NONE":2}` |

Seven-part read:

| part | current evidence | diagnosis |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08`: wall `40.230ms/step`, almost all in request routing loop `39.994ms/step`; row bucket `0.161ms/step`, bucket build `0.159ms/step`, row indices `0.001ms/step` | this is a real clean-path cost, not just CUDA-event profiling overhead |
| base sparse linear | profile: aggregate sparse base `1.933ms/call`, `gate_up_proj=16-31` sparse base `2.595ms/call`, about `335` rows/call | large GPU-side useful-work cost |
| residual correction | profile: dense-row correction `0.120ms/call`, `gate_up_proj=16-31` dense correction `0.154ms/call`, about `31` bucket rows/call | secondary relative to sparse base |
| gather/scatter | profile: `0.287ms/call`; bucket Triton override itself `0.570ms/call` | assembly/correction wrapper is visible and larger than the dense GEMM |
| routing statistics | profile: draft residual/base `16759/3265`, non-draft residual/base `2503/3787`, draft residual fraction `0.837`, bucket fill `0.836` | quality guard still sends most draft rows through residual correction |
| CUDA Graph | clean `speclink_t08`: `{"FULL":62,"NONE":2}` | graph coverage is healthy; do not diagnose this run as graph loss |
| GPU util | clean `speclink_t08` avg `84.556%`, peak `99%`; profile avg `78.556%`, peak `97%` | GPU is busy; slowdown is inefficient useful work plus scheduler routing, not idle GPU |

Interpretation: this path can hit the full-batch `1.2x` target, but only in the
steady/full-batch window. Total output tok/s is still worse than dense because
per-step CPU routing/mask work is large, and the operator still pays sparse
base work for all rows plus correction/assembly for many residual rows. The
next optimization should therefore be a breakdown-driven implementation pass:

1. remove or drastically reduce the Python/request routing loop in
   `build_verify_residual_mask()` for `critical_prefix`/early-dense/all-MLP;
   the current clean cost is about `40ms/step`;
2. avoid paying full sparse base for rows that are going to be dense-corrected,
   or fuse sparse base + dense-row correction + scatter into a packed
   CUDA/Triton path;
3. keep CUDA Graph coverage as a guardrail, because this run already has
   `62/64` FULL graph steps;
4. keep paired accuracy gates: the earlier all-MLP Triton override without the
   early dense guard reached the speed target but failed GSM8K-50 quality.

### Historical User-Requested Seven-Part Refresh At 03:22

This older current-code refresh for the user-requested breakdown table is kept
for comparison; prefer the corrected fast-candidate reference at the top of this
file:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_bs64_math128_20260628_0322/seven_part_report/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size
64, max new tokens 128, SR24 on `gate_up_proj=16-31` and `down_proj=8-15`,
`critical_prefix`, threshold `0.6`, minimum residual prefix `4`,
extra-after-low `1`, dense-row residual bucket size `32`, bucket priority on,
low-sync runtime counters, CUDA Graph enabled.

Clean serving rows:

| method | full-batch tok/s | total tok/s | same-root speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3128.512` | `2184.738` | `1.000x` | `1.399` | `86.250%` | n/a |
| `base_only_24` | `3428.164` | `2238.218` | `1.096x` | `1.635` | `86.500%` | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | `3182.699` | `2004.300` | `1.017x` | `1.606` | `86.625%` | `{"FULL":62,"NONE":2}` |

This refresh reinforces the same conclusion as the earlier 01:22 report:
`speclink_t08` is not slow because the GPU is idle, the accepted draft length
collapses, or CUDA Graph is disabled. The clean path keeps high GPU util and
`62/64` FULL graph steps, while the base-only upper bound is only `1.096x`
dense in this quality-safe scope. That leaves very little end-to-end headroom
for `speclink_t08` before any residual correction is added.

The requested seven components are:

| part | current evidence | diagnosis |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08`: wall `0.949ms/step`, batched builder `0.771ms/step`, row bucket `0.107ms/step`, bucket build `0.105ms/step`, row indices `0.001ms/step` | sub-ms in the clean path; not the first bottleneck |
| base sparse linear | diagnostic `gate_up_proj=16-31`: `1.069ms/call`; aggregate sparse base `1.007ms/call` | the largest localized GPU-side cost |
| residual correction | diagnostic dense-row correction `0.148ms/call`; `gate_up_proj=16-31` dense correction `0.172ms/call`; bucket rows `32/call` | secondary per call, but additive on top of sparse base |
| gather/scatter | diagnostic `0.015ms/call` | too small to explain the end-to-end gap |
| routing statistics | draft residual/base `7263/5921`, non-draft residual/base `1648/1824`, draft residual fraction `0.551`, bucket fill `0.943`, actual/requested bucket fraction `0.229` | many draft rows still need residual protection |
| CUDA Graph | clean `speclink_t08`: `{"FULL":62,"NONE":2}` | graph coverage is healthy for this path |
| GPU util | clean `speclink_t08` avg `86.625%`, peak `100%` | GPU is busy with inefficient useful work, not idle |

Interpretation: the current slow part is still the mixed operator shape. The
base sparse GEMM is useful by itself, but `speclink_t08` pays sparse base work
for all active rows plus dense-row correction for a still-large residual set.
Since gather/scatter and clean scheduler time are smaller, the next speed path
should be a fused/packed sparse-plus-residual operator or a row-selection signal
that sharply reduces residual rows without losing paired accuracy. Another
threshold-only sweep is unlikely to reach `1.2x`.

The latest seven-part breakdown answers the current "where is it slow" question
for the quality-safe `critical_prefix`/bucket32 path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_seven_part_breakdown_bs64_math128_20260628_0122/seven_part_report/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size
64, max new tokens 128, SR24 on `gate_up_proj=16-31` and `down_proj=8-15`,
`critical_prefix`, threshold `0.6`, minimum residual prefix `4`, extra-after-low
`1`, dense-row residual bucket size `32`, low-sync runtime counters, CUDA Graph
enabled.

Clean serving rows:

| method | full-batch tok/s | total tok/s | same-root speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3021.676` | `2184.905` | `1.000x` | `1.395` | `84.625%` | n/a |
| `base_only_24` | `3480.972` | `2285.907` | `1.152x` | `1.574` | `85.000%` | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | `3086.711` | `2065.992` | `1.022x` | `1.632` | `86.875%` | `{"FULL":62,"NONE":2}` |

This rules out the two easiest explanations for the current path:

1. `base_only_24` is not slow in this scoped configuration. It is faster than
   dense and has slightly higher accepted draft length.
2. `speclink_t08` is not slow because the GPU is idle or because CUDA Graph is
   mostly disabled. The clean row keeps high GPU utilization and `62/64` FULL
   graph steps.

The seven requested components now read as follows:

| part | current evidence | diagnosis |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08`: wall `0.380ms/step`, row bucket `0.112ms/step`, bucket build `0.110ms/step`, row indices `0.001ms/step` | sub-ms in the clean path; not the first bottleneck |
| base sparse linear | diagnostic `gate_up_proj=16-31`: `1.023ms/call`; aggregate sparse base `0.937ms/call` | the largest localized GPU-side cost |
| residual correction | diagnostic dense-row correction `0.148ms/call`; `gate_up_proj=16-31` dense correction `0.171ms/call` | secondary per call, but additive on top of base sparse |
| gather/scatter | diagnostic `0.012ms/call` | too small to explain the end-to-end gap |
| routing statistics | draft residual/base `14125/11395`, non-draft residual/base `3190/3787`, draft residual fraction `0.553`, bucket fill `0.978`, actual bucket fraction `0.124` | too many draft rows still need residual protection for a two-pass operator |
| CUDA Graph | clean `speclink_t08`: `{"FULL":62,"NONE":2}` | graph coverage is healthy for this path |
| GPU util | clean `speclink_t08` avg `86.875%`, peak `100%` | GPU is busy with inefficient useful work, not idle |

Follow-up ablations on the same shape rejected several tempting shortcuts:

| ablation | root | result |
| --- | --- | --- |
| bucket dense copy overwrite | `results.bak/sr24_t08_graphon_bucketcopy_bs64_math128_20260628_0130` | only `1.026x` dense full-batch; gather/scatter was not the bottleneck |
| Triton bucket dense GEMM scatter | `results.bak/sr24_t08_graphon_tritonbucket_bs64_math128_20260628_0133` | `0.962x` dense; slower than the PyTorch/cuBLAS bucket correction |
| adaptive dense fallback | `results.bak/sr24_t08_graphon_adaptivefallback_bs64_math128_20260628_0145` | `0.952x` dense and accepted length falls to `1.402` |
| `all_corrected_24` compressed-dense on CUDA | `results.bak/sr24_allcorrected_compresseddense_cached_bs64_math128_20260628_0137` | GPU-resident (`24` CUDA residual modules, `0` CPU), but only `0.780x` dense |
| direct compressed Triton residual | `results.bak/sr24_allcorrected_compresseddense_triton_gmem078_bs64_math128_20260628_0141` | rejected: `0.181x` dense and no CUDA Graph |

`all_corrected_24` with the dense fastpath is the correct dense-equivalent
control. The matrix runner now treats that path as an effective default-vLLM
compile path automatically, instead of assigning an SR24 compile-cache profile:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_densefastpath_auto_defaultcompile_bs64_math128_20260628_0157/clean_serving
```

The current optimization target should therefore shift away from CPU-sync
cleanup and small gather/scatter changes. The required speed path is either:

1. a better sparse scope whose `base_only_24` upper bound is clearly above the
   desired target while preserving quality with very few residual rows, or
2. a fused/packed MLP or Linear operator that avoids doing full sparse base
   work plus a separate residual GEMM for rows that are going to be corrected.

With the current quality-safe scope, `base_only_24` is only `1.152x` dense.
That means `speclink_t08` cannot realistically reach `1.2x` dense unless the
operator structure changes or the corrected-row fraction drops sharply without
hurting accuracy.

### Required Breakdown Before More Tuning

Do not judge the next SR24/SpecLink candidate from throughput alone. Every
candidate that looks slow or promising should first produce the same seven-part
breakdown:

| component | required measurement | current read |
| --- | --- | --- |
| scheduler / mask build | Per-step residual-mask construction, request routing, bucket rows, and row-index construction. | In the current graph-on quality-safe path this is sub-ms (`0.380ms/step` in the seven-part report). Earlier route-all variants exposed a separate dynamic-row-list failure (`42ms/step`), so this must stay in the table. |
| base sparse linear | Sparse base GEMM time, especially `gate_up_proj=16-31` when using the quality-safe scope. | Large GPU-side cost: `gate_up_proj=16-31` sparse base is about `1.023ms/call`; aggregate sparse base is about `0.937ms/call`. |
| residual correction | Dense-row or compressed residual GEMM time. | Secondary per call but additive: dense-row correction is about `0.148ms/call`, `gate_up_proj=16-31` correction about `0.171ms/call`. |
| gather/scatter | `index_select`, `index_add_`, `index_copy_`, bucket assembly, and any Triton assembly path. | Small in the current quality-safe path (`0.012ms/call`), so gather/scatter-only rewrites are not the main route. |
| routing statistics | Draft residual/base rows, non-draft residual/base rows, bucket fill ratio, and effective corrected-row fraction. | Many draft rows still need residual protection: draft residual/base is `14125/11395`, non-draft residual/base is `3190/3787`, draft residual fraction is `0.553`. |
| CUDA Graph | FULL/NONE counts for dense, `base_only_24`, and `speclink_t08`. | Healthy for the current graph-on path (`{"FULL":62,"NONE":2}` in the seven-part report); graph loss remains a hard regression guard. |
| GPU util | Average/peak GPU utilization plus full-batch and total output tok/s. | GPU is busy, not idle: clean `speclink_t08` averages `86.875%` GPU util. The problem is inefficient useful work, not lack of occupancy alone. |

The practical interpretation is: the next speedup cannot come from another
scalar threshold sweep by itself. The current slow part is the GPU useful-work
shape of the mixed operator: sparse base work plus residual correction work are
both being paid, while the residual rows remain too frequent for the two-pass
implementation to beat dense by the required margin. CUDA Graph coverage and
accepted draft length must remain guardrails, but after they are healthy the
main target is a fused/packed mixed sparse-residual operator or a much sharper
row-selection signal that keeps correction rows near zero without accuracy
loss.

### Base-Only Scope Speed Ceiling Sweep

The follow-up scope sweep is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_scope_sweep_bs64_math128_20260628_0203/report.md
```

It runs only `dense_baseline` and `base_only_24` on bs64/math/max128. It is a
speed-ceiling study, not a quality claim.

| scope | base-only full tok/s | dense full tok/s | speedup | accepted draft/step | modules |
| --- | ---: | ---: | ---: | ---: | ---: |
| safe `gate_up=16-31,down=8-15` | `3432.844` | `3018.718` | `1.137x` | `1.629` | 24 |
| `gate_up=16-31` | `3250.484` | `3025.946` | `1.074x` | `1.542` | 16 |
| `down=8-15` | `3096.800` | `3025.584` | `1.024x` | `1.423` | 8 |
| `down=16-31` | `3025.418` | `3137.366` | `0.964x` | `1.469` | 16 |
| `gate_up=16-31,down=16-31` | `3424.279` | `3026.264` | `1.132x` | `1.598` | 32 |
| all `gate_up` | `3536.317` | `3136.770` | `1.127x` | `1.656` | 32 |
| all MLP `gate_up,down` | `5098.544` | `3134.588` | `1.627x` | `2.434` | 64 |
| tail `gate_up=31` with `up_sparse` split | `2938.625` | `3021.371` | `0.973x` | `1.429` | 1 |

Read: the current quality-related small scopes do not have enough base-only
headroom for a `1.2x` final `speclink_t08` target. The only measured scope with
clear speed headroom is all-MLP, which is also the highest quality-risk scope.
Therefore the next `speclink_t08` candidate should either:

1. start from all-MLP and solve quality by identifying which rows/layers need
   dense/residual protection, or
2. abandon controller-only tuning for small scopes and implement a fused/packed
   mixed sparse/residual operator.

During this sweep, the `gate_up_split` tail case exposed a startup bug in
`vllm/speclink_sr24.py`: the split attach stats referenced an undefined
`sr_bonus_priority`. The local fix uses `bonus_priority()` directly in split
and channel-pair attach rows.

The first all-MLP `speclink_t08` throughput point is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bs64_math128_20260628_0220/report.md
```

| method | full-batch tok/s | total tok/s | speedup vs dense full | accepted draft/step | GPU util | graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3025.159` | `2187.155` | `1.000x` | `1.395` | `84.625%` | n/a |
| all-MLP `base_only_24` | `5399.310` | `2739.934` | `1.785x` | `2.448` | `79.167%` | `{"FULL":62,"NONE":2}` |
| all-MLP `speclink_t08` | `3545.363` | `1717.158` | `1.172x` | `2.345` | `80.700%` | `{"FULL":55,"NONE":9}` |

This makes all-MLP the first route with real t08 speed potential, but it is not
yet a solution. It is still below the `1.2x` full-batch target and has no
quality proof. The next experiment should be a paired accuracy gate and
token-level residual/base trace for this all-MLP t08 candidate, then a controller
that protects the rows/layers causing regressions while preserving most of the
`base_only_24` headroom.

The first paired accuracy gate for the all-MLP candidate is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_accuracy_gsm8k20_20260628_0225/report.md
```

Shape: Llama-3.1-8B, GSM8K COT, 20 samples, max new tokens 512, EAGLE3 K=8.

| mode | accuracy | delta vs dense | pair reg | pair imp | avg output tokens | clipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense EAGLE3 | `0.7000` | `0.0000` | 0 | 0 | `87.1` | 0 |
| all-MLP `base_only_24` | `0.1500` | `-55.0pp` | 12 | 1 | `184.0` | 5 |
| all-MLP `speclink_t08` | `0.7000` | `0.0pp` | 1 | 1 | `106.3` | 0 |

Read: all-MLP pure base-only is not viable; it causes real reasoning errors and
long/clipped repetitions. The current all-MLP `speclink_t08` controller repairs
the aggregate GSM8K-20 score, but it is not yet paired-safe because it has one
dense-correct/t08-wrong regression and one dense-wrong/t08-correct improvement.
The next precision step should rerun this candidate with confidence/SR24
residual-mask trace enabled and inspect the `doc_id:2` regression before
loosening residual protection for speed.

The follow-up trace analysis adds an important guardrail for the next
controller iteration:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_doc2_trace_analysis_unreached_20260628_0305/
```

For the `doc_id:2` regression, `critical_prefix@0.4` has zero accepted,
rejected, and reached effective-base rows, but still has `0.5246` unreached
effective-base suffix rows. The `critical_prefix@0.0` trace has no base-only
rows at all, matching the all-corrected trace at the routing level. Therefore
the next correctness/debug step must separate two issues:

1. whether selective routing leaves any base-only suffix rows that can perturb
   verifier/KV behavior even when they are not accepted locally, and
2. whether a selective all-residual verify plan is numerically/control-flow
   equivalent to the `all_corrected_24` no-fastpath control.

Do not use accepted/rejected base-only fractions alone as the all-MLP quality
criterion. Future trace summaries should include unreached effective-base rows
and steps before deciding that a controller is safe.

## 2026-06-27 Current Read

The latest route-all/fixed-bucket evidence changes the slowdown diagnosis from
"mainly CUDA Graph loss" to a two-stage operator problem:

1. `prefix_confidence@0.05 + gate_up_proj + route_all` repairs accepted length
   and keeps CUDA Graph coverage, but exposes a large dynamic row-list window:
   `scheduler_mask_wall_cpu_ms_per_step=42.339`, almost all in
   `scheduler_row_index_bucket_wall_cpu_ms_per_step=42.083`.
2. Skipping the unused residual bucket proves the bucket itself is not the
   issue: bucket build falls to `0.001ms/step`, but
   `scheduler_mixed_row_indices_wall_cpu_ms_per_step` remains `42.113ms`.
3. The fixed bucket512 + Triton dense-correction control removes dynamic row
   list materialization (`mixed_row_indices=0.001ms/step`, clean scheduler mask
   wall `0.400ms/step`) and keeps CUDA Graph coverage, but drops to
   `2029.030` full-batch tok/s versus same-root dense `3020.696` (`0.672x`).
   GPU util is still high (`89.0%`) and accepted draft/step is healthy
   (`1.411`), so this route is slow because it does too much GPU work: sparse
   base `0.577ms/call` plus 512-row dense correction `0.316ms/call`.

So the next optimization should not be another threshold/controller sweep. The
needed path is a mask-aware route-all/fused operator or compact per-request
prefix representation that avoids dynamic `nonzero` row lists without
correcting a large fixed bucket of mostly low-value rows. The authoritative
current seven-part breakdown is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWDOWN_BREAKDOWN.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_bucket_triton_densegemm_bs64_math128_20260627/seven_part_report/report.md
```

Follow-up probes on the same day rejected two tempting shortcuts:

- `SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS=1` for `prefix_confidence` removes GPU
  `nonzero` row-list materialization, but replaces it with draft-score
  GPU-to-CPU synchronization. Clean `scheduler_row_index_bucket` falls to
  `0.002ms/step`, but `scheduler_direct_cpu_route_rows` becomes
  `43.200ms/step`; full-batch throughput is only `0.930x` dense.
- The new `fixed_prefix` policy avoids draft-score reads and protects a fixed
  prefix plus the bonus row. H=2 + `route_reuse_base_output` gets scheduler
  wall down to `0.277ms/step`, but still reaches only `0.954x` dense; H=0 is
  an even looser speed ceiling with only bonus-row correction and still reaches
  only `0.945x` dense. This means the remaining problem is no longer scheduler
  overhead in that shape; the current gate/up sparse base operator is not fast
  enough in serving even when residual correction is sparse.

The operator-only refresh on the same day confirms this ceiling without vLLM
scheduler noise:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_operator_probe_20260627_225539/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_operator_probe_20260627_225539/summary.md
```

For the Llama `gate_up_proj` shape (`out=28672, in=4096`), graph-captured
base sparse is faster than dense by itself:

| rows | dense graph | base sparse graph | base/dense |
| ---: | ---: | ---: | ---: |
| 256 | `0.2892ms` | `0.1668ms` | `0.58x` |
| 512 | `0.5401ms` | `0.3551ms` | `0.66x` |

But the moment exact dense/residual correction is added, the path loses the
sparse win. With only about 10% corrected rows, the serving-like mixed path is
already `1.24x` dense time at 256 rows and `1.02x` dense time at 512 rows. At
25% correction it is `1.31x` and `1.15x` dense time, respectively. The
all-corrected exact backend is worse: graph-captured base sparse plus
complementary sparse residual is `1.34x` dense time at 256 rows and `1.41x`
dense time at 512 rows. The best current exact all-corrected candidates still
need about `1.60-1.69x` additional operator speedup to reach a `1.2x` dense
serving target.

This means `base_only_24` can be fast while `speclink_t08` remains slow:
`base_only_24` exercises only the base sparse win, while `speclink_t08` pays
base sparse plus correction plus routing/assembly. Any future controller-only
change must first prove that the corrected-row fraction is near zero; otherwise
the operator path, not the controller, is the limiter.

Evidence:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_directcpurows_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_h2_reusebase_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_h0_reusebase_bs64_math128_20260627/seven_part_report/report.md
```

## Source Artifacts

Main current-code breakdown:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/component_microbench/summary.md
```

Current seven-part reducer with the later runtime-timing, route, bucket, and
full-bucket probes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_seven_part_breakdown_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_seven_part_breakdown_20260627/seven_part_breakdown.csv
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_seven_part_breakdown_20260627/joined_rows.csv
```

Current operator microbench refresh:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_operator_probe_20260627_225539/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_operator_probe_20260627_225539/summary.md
```

Important control runs:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_static_allres_densefastpath_noop_defaultcompile_bs64_math128_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_cap0_maskstate_densefallback00_defaultcompile_bs64_math128_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_gateup16_31_torchsparse_directcslt_srcompile_gmem075_bs64_math128_20260627/report.md
```

Run shape for the main serving read: Llama-3.1-8B, `math_reasoning`, EAGLE3
K=8, client concurrency batch size 64, max new tokens 128.

## High-Level Read

`base_only_24` is not the current slowdown source. In the clean serving row it
is faster than dense:

| method | full-batch output tok/s | total output tok/s | speedup vs dense | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | 3024.300 | 2186.000 | 1.000x | 82.625% | n/a |
| `base_only_24` | 3346.098 | 2286.592 | 1.106x | 85.000% | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | 2420.452 | 1667.705 | 0.800x | 85.800% | `{"NONE":64}` |

So the main slowdown is not low accepted length or an idle GPU. The clean
`speclink_t08` run keeps the GPU busy but does less useful work per unit time:
it loses CUDA Graph coverage and enters a mixed sparse-base plus residual
correction path with too many corrected rows.

## Requested Breakdown

| part | what is measured | current evidence | diagnosis |
| --- | --- | --- | --- |
| scheduler / mask build | residual mask construction, bucket row selection, request routing loop | clean `speclink_t08` bs64: wall `0.658 ms/step`, batched mask builder `0.591 ms/step`; diagnostic `speclink_t08`: mask `5.950 ms/step`, request loop `5.765 ms/step` | exact diagnostic path is sync-heavy. In clean serving the scheduler/mask path is visible but not the main slowdown: it is much smaller than TPOT and far smaller than the dense-vs-mixed throughput gap. |
| base sparse linear | `gate_up_proj=16-31` sparse-base Linear time | diagnostic `speclink_t08`: `0.632 ms/call`; microbench base sparse is about `0.57-0.66x` dense when graph captured | base sparse is useful by itself, but it only helps if few rows need correction. |
| residual correction | dense/sparse correction for rows routed to dense accuracy | diagnostic `speclink_t08`: `0.336 ms/call`; exact GPU-resident `all_corrected_24` is only `0.876x` dense full-batch | corrected rows pay extra work on top of sparse base. This is the main useful-work loss. |
| gather/scatter | `index_select`, base-row read, delta, `index_add_`/copy assembly | diagnostic `speclink_t08`: `0.036 ms/call`; microbench grows up to `0.281 ms` at high residual fraction | not the first bottleneck in current routing, but it becomes material when residual rows are high. |
| routing statistics | draft residual/base rows, non-draft residual/base rows, bucket fill | diagnostic `speclink_t08`: draft residual/base `5757/2267`, non-draft residual/base `1003/1771`; clean non-draft residual/base about `3425/3787` | too many draft rows still use residual correction. With a two-pass operator this cannot beat dense. |
| CUDA Graph | dense/base-only/t08 FULL/NONE graph steps | `base_only_24`: `{"FULL":62,"NONE":2}`; clean `speclink_t08`: `{"NONE":64}` | graph loss is a real part of the slowdown. However previous graph-recovery probes still stayed below dense, so graph alone is not enough. |
| GPU util | sampled average/peak utilization | dense `82.625%`, clean `speclink_t08` `85.800%`, peak `100%` | not gross GPU underutilization. The GPU is busy with inefficient dynamic/mixed work. |

## Why The Current Path Is Slow

The current mixed SR24 path has three stacked costs:

1. It computes the sparse base output for many rows.
2. For rows that need dense accuracy, it also runs residual/dense correction.
3. It must assemble the corrected rows back into the original output layout,
   while the dynamic shape/mask path loses CUDA Graph capture in clean serving.

The exact GPU-resident all-corrected probe is the clearest negative control:

| run | dense full-batch tok/s | candidate full-batch tok/s | ratio | backend | graph |
| --- | ---: | ---: | ---: | --- | --- |
| exact `all_corrected_24` | 3131.601 | 2743.941 | 0.876x | `torch_sparse/torch_sparse@cuda` | `{"FULL":62,"NONE":2}` |

This rules out the simple explanation that residual correction is accidentally
on CPU. It is on GPU in this run, graph coverage is mostly healthy, and the path
is still slower. The problem is the operator structure: base sparse plus
residual work is more expensive than one dense Linear for the row mix we have.

The dense-fastpath no-op control also matters:

| run | dense full-batch tok/s | candidate full-batch tok/s | ratio | read |
| --- | ---: | ---: | ---: | --- |
| static all-residual dense-fastpath no-op | 3023.230 | 3025.310 | 1.001x | SR24 env/stats are not inherently slow when Linear hooks are bypassed. |
| `fallback00`, default vLLM compile | 3008.751 | 2949.389 | 0.980x | compile cleanup recovers most safe-fallback overhead, but still does not create a speed path. |

So the slowdown is localized to the dynamic SR24 Linear/MLP path, not to the
benchmark harness, EAGLE3 acceptance length, or SR24 environment flags.

## Optimization Direction

Threshold-only tuning should not be the main next step. It can change quality,
but it cannot reach the speed target while most useful steps still enter the
current two-pass mixed operator.

The next useful implementation work should be:

1. Restore CUDA Graph coverage for mixed SR24 steps by fixing mask/state shapes
   and using preallocated buffers where possible.
2. Reduce duplicate compute: do not run sparse base on rows that will be dense
   corrected, or fuse the mixed MLP/Linear route so corrected and base rows are
   packed into larger efficient kernels.
3. Keep routing statistics in every run: draft residual fraction, non-draft
   residual fraction, bucket fill ratio, and CUDA Graph FULL/NONE counts should
   be treated as required metrics, not optional diagnostics.

The current target condition for a real win is stricter than "sparse base is
faster than dense": the corrected-row fraction must be low enough, or the
correction path must be fused enough, that sparse-base plus correction plus
assembly is below one dense Linear end to end.

## Runtime Timing Instrumentation Added

To make the scheduler/mask-build part measurable in clean serving runs, the SR24
runtime stats path now records CPU wall-clock timings without CUDA events or
forced tensor synchronization. The timing is enabled by default when SR24 runtime
stats are enabled and can be disabled with:

```text
SPECLINK_SR24_RUNTIME_TIMING=0
```

New per-window matrix fields include:

```text
sr24_scheduler_mask_wall_cpu_ms_per_step
sr24_scheduler_materialize_counts_wall_cpu_ms_per_step
sr24_scheduler_pending_scores_pop_wall_cpu_ms_per_step
sr24_scheduler_batched_mask_builder_wall_cpu_ms_per_step
sr24_scheduler_request_routing_loop_wall_cpu_ms_per_step
sr24_scheduler_batch_all_apply_wall_cpu_ms_per_step
sr24_scheduler_mask_state_wall_cpu_ms_per_step
sr24_scheduler_static_mask_copy_wall_cpu_ms_per_step
sr24_scheduler_row_index_bucket_wall_cpu_ms_per_step
```

The matrix runner computes these from before/after deltas over the measurement
window, matching the existing SR24 token-counter handling. The breakdown
summarizer now also emits a `Clean Runtime Scheduler Wall Time` table. This
table should be used for clean serving scheduler overhead; the older
`scheduler_*_cuda_ms` and exact-routing CPU fields remain diagnostic-only.

Validation performed after adding the fields:

```text
conda run -n spec python -m py_compile \
  vllm/vllm/speclink_sr24.py \
  examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py \
  examples/evaluate/eval-guidellm/scripts/summarize_sr24_breakdown.py \
  examples/evaluate/eval-guidellm/scripts/make_sr24_seven_part_breakdown.py

conda run -n spec python examples/evaluate/eval-guidellm/scripts/summarize_sr24_breakdown.py \
  --roots examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/clean_serving \
          examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/instrumented_serving \
  --output-root examples/evaluate/eval-guidellm/results.bak/sr24_runtime_timing_reducer_compat_20260627/component_summary

conda run -n spec python examples/evaluate/eval-guidellm/scripts/make_sr24_seven_part_breakdown.py \
  --roots examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/clean_serving \
          examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_current_20260627/instrumented_serving \
  --output-root examples/evaluate/eval-guidellm/results.bak/sr24_runtime_timing_reducer_compat_20260627/seven_part_report
```

The compatibility reducer output is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_timing_reducer_compat_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_timing_reducer_compat_20260627/seven_part_report/report.md
```

Clean bs64 validation run:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_timing_clean_speclink_t08_bs64_20260627/
```

This run used Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency
64, 64 requests, max new tokens 128, `gate_up_proj=16-31`, static mask buffer,
batched mask builder, and `speclink_t08` with the existing high-confidence
route. Key output:

| metric | value |
| --- | ---: |
| full-batch output tok/s | 2416.158 |
| total output tok/s | 1663.699 |
| mean TPOT | 25.717 ms |
| SR24 steps | 64 |
| scheduler/mask wall time | 0.658 ms/step |
| batched mask builder | 0.591 ms/step |
| static mask copy | 0.009 ms/step |
| row-index bucket | 0.003 ms/step |
| CUDA Graph modes | `{"NONE":64}` |
| non-draft residual fraction | 0.475 |
| average GPU util | 85.400% |

The seven-part report is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_runtime_timing_clean_speclink_t08_bs64_20260627/seven_part_report/report.md
```

Conclusion from this validation: the clean scheduler/mask path is about
`0.66 ms/step`, dominated by the batched mask builder. That is worth optimizing
later, but it is not the primary reason `speclink_t08` is slower than dense.
The immediate bottleneck remains CUDA Graph loss plus the mixed sparse-base and
residual-correction operator doing duplicated work for a high corrected-row
fraction.

## Route And Bucket Follow-Up

I tested four bs64/math/K8/max128 follow-up routes against the clean
`speclink_t08` row to separate duplicate GPU work from CPU-side row planning:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_all_residual_t08_bs64_20260627/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_all_sync_fallback045_t08_bs64_20260627/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket64_triton_t08_bs64_20260627/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket64_triton_graphprobe_t08_bs64_20260627/
```

| case | full-batch tok/s | total tok/s | TPOT | CUDA Graph | scheduler/mask | row-index/bucket | read |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| clean `speclink_t08` | 2416.158 | 1663.699 | 25.717 ms | `NONE=64` | 0.658 ms/step | 0.003 ms/step | baseline mixed path: full sparse base plus dense correction/select |
| route all residual rows | 2652.857 | 1833.590 | 23.838 ms | `NONE=64` | 36.011 ms/step | 35.768 ms/step | less duplicate GPU work, but PyTorch `nonzero` row compaction becomes the bottleneck |
| route all + sync fallback 0.45 | 2948.807 | 1957.693 | 22.318 ms | `FULL=62,NONE=2` | 42.287 ms/step | 0.019 ms/step | graph recovers by promoting high-residual steps to dense/all-residual; this is quality-conservative but dense-equivalent |
| bucket64 Triton correction | 2616.221 | 1743.312 | 24.732 ms | `NONE=64` | 0.852 ms/step | 0.102 ms/step | avoids row compaction but stays eager and corrects only a capped bucket |
| bucket64 Triton + mixed graph probe | 2654.743 | 1879.206 | 24.130 ms | `FULL=62,NONE=2` | 0.352 ms/step | 0.106 ms/step | restoring graph helps only slightly; the bucketed dense scatter is still not enough |
| fullbucket576 Triton | 1682.201 | 1179.944 | 37.915 ms | `NONE=64` | 12.993 ms/step | 0.088 ms/step | fixed-shape full bucket removes row compaction, but bucket/values routing becomes a large request-loop cost and GPU util drops |
| fullbucket576 Triton, low-sync | 1596.742 | 1236.113 | 38.809 ms | `NONE=64` | 13.165 ms/step | 0.104 ms/step | disabling mask-state sync does not fix it; the full bucket path is still too heavy |

This adds two concrete findings:

1. A row-routed mixed operator can reduce duplicated GPU work, but the current
   implementation uses dynamic `nonzero` compaction for residual/base rows. On
   bs64 this costs about `35.8 ms/step`, so it cannot be the final route unless
   row compaction is replaced by a graph-safe preallocated GPU path or avoided
   entirely.
2. Fixed bucket correction avoids that compaction cost, but the current Triton
   dense scatter path is still only about `1.10x` over the clean slow
   `speclink_t08` baseline and remains below dense. The performance target
   needs a larger fused/packed operator than "full sparse base plus capped dense
   bucket overwrite".
3. Increasing the bucket to the full bs64/K8 row budget is worse, not better.
   It avoids `nonzero` row compaction (`~0.10 ms/step`) but pushes
   request-loop/bucket construction to about `13 ms/step`, keeps CUDA Graph at
   `NONE=64`, and lowers average GPU utilization to `64-67%`. The padded static
   bucket tail is now marked inactive so oversized graph buckets do not write
   beyond real rows, but a full fixed bucket is not the right optimization path.

The next useful path is therefore not "make the bucket bigger". It should be
one of:

1. reduce residual/corrected rows before the Linear hook sees them;
2. fuse route construction and correction into a graph-safe GPU kernel that
   skips inactive rows instead of launching dense work over a large bucket; or
3. move to a row-routed MLP/Linear operator that packs base and dense rows once,
   avoiding sparse-base work on rows that will be dense-corrected.

The quality evidence from the existing paired lm-eval runs also remains
important: even the narrow one-module pair-aware `gate_up_proj=31` base-only
candidate tied aggregate GSM8K-50 in one run, but GSM8K-100 and Minerva-100
still showed paired regressions. Most regressions diverged in the first few
generated tokens, so this is a reasoning-trajectory change rather than a late
formatting artifact. Any future speed candidate must therefore be gated by
paired regressions/improvements, not aggregate accuracy alone.

Tooling fix after the bucket probe: both the GuideLLM matrix runner and the
lm-eval accuracy runner now expose and record the Triton bucket dense-GEMM block
parameters:

```text
--sr24-triton-bucket-dense-block-m
--sr24-triton-bucket-dense-block-n
--sr24-triton-bucket-dense-block-k
```

These values are passed to `SPECLINK_SR24_TRITON_BUCKET_DENSE_BLOCK_{M,N,K}`,
included in the SR24 compile-cache fingerprint, written to summary metadata,
and preserved in generated resume commands. This makes follow-up bucket kernel
tuning reproducible instead of accidentally reusing a cache from a different
block shape.

## Static Leaf Candidate Recheck, 2026-06-27

I rechecked the current static leaf candidates after the slowdown breakdown to
separate leaf-selection speedups from the dynamic mixed-operator bottleneck.
All runs used Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency
bs64, 64 fixed requests, and max new tokens 512.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_leaf_gate_only_throughput_bs64_math512_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_accuracy_first_throughput_bs64_math512_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_throughput_aggressive_bs64_math512_20260627/report.md
```

| preset | base-only leaf scope | full-batch speedup | total speedup | CUDA Graph | read |
| --- | --- | ---: | ---: | --- | --- |
| `accuracy_gate_only` | `gate_up_proj=31` | `1.013x` | `0.976x` | `{"FULL":222,"NONE":2}` | graph-safe, but not a speed path |
| `accuracy_first` | `gate_up_proj=31`, `up_sparse`, with qkv/o densefastpath | `0.993x` | `0.968x` | `{"FULL":222,"NONE":2}` | conservative quality direction, but slower than dense here |
| `throughput_aggressive` | `gate_up_proj=31;down_proj=30-31` | `1.027x` | `1.224x` | `{"FULL":254,"NONE":2}` | total tok/s improves through step/drain effects, but stable full-batch speed is far below the `1.2x` target |

The small accuracy gate for the aggressive preset was:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_throughput_aggressive_gsm8k20_accuracy_20260627/report.md
```

It reported GSM8K-20 dense `0.6000` and SR24 `0.7000`, with `0` paired
regressions and `2` paired improvements. This is too small to prove quality
safety, but it shows that the aggressive static tail is not immediately broken
on the first 20 samples. The limiting problem is still performance quality:
the only candidate with `>1.2x` total tok/s does not deliver `>1.2x`
full-batch/steady-state throughput.

This recheck narrows the next optimization step. Static leaf selection alone is
not enough. The conservative static presets preserve CUDA Graph but are
near-parity; the aggressive static preset changes request-step behavior enough
to improve total tok/s, but its full-concurrency kernel throughput remains only
about `1.03x`. The next implementation work should therefore target the
operator structure directly: pack or fuse base/dense rows so a corrected row
does not pay both sparse-base and dense/residual work, while keeping the
graph-safe static-buffer behavior from these static runs.

## Route-All Contiguous Clean Check, 2026-06-27

I re-ran the explicit row-routed path with both
`SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS=1` and
`SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH=1` on the clean bs64/K8/math/max256
serving shape:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_all_contiguous_clean_bs64_math256_20260627/report.md
```

| method | full-batch output tok/s | total output tok/s | speedup vs dense | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | 3428.251 | 2318.787 | 1.000x | 88.429% | n/a |
| route-all contiguous `speclink_t08` | 2364.868 | 1722.057 | 0.690x full / 0.743x total | 64.316% | `{"NONE":133}` |

The scheduler/runtime counters make the failure mode clear:

| metric | value |
| --- | ---: |
| scheduler mask wall time | 5.791 ms/step |
| scheduler row-index/bucket wall time | 5.381 ms/step |
| batched mask builder | 0.351 ms/step |
| residual non-draft fraction | 0.611 |
| non-draft residual/base rows | 5958 / 3788 |

This is a negative result for the current route-all implementation. It tries
to avoid duplicated sparse-base work on corrected rows, but the dynamic
row-index construction itself becomes the dominant cost, CUDA Graph coverage is
lost, and GPU utilization drops. The next route-all attempt should not use
per-step PyTorch `nonzero`/dynamic row compaction as the serving path; it needs
preallocated graph-safe row lists or a fused GPU-side routing kernel before it
is worth another accuracy gate.

Small tooling fix from this run: `SPECLINK_SR24_ROUTE_CONTIGUOUS_FASTPATH` was
visible in per-module stats snapshots but missing from some top-level summary
branches. Future stats now record `route_contiguous_fastpath` consistently, and
`run_sr24_slowdown_breakdown.py` exposes `--sr24-route-contiguous-fastpath` so
the route-all/route-bucket ablation is reproducible from the standard
breakdown runner.

## Seven-Part Breakdown Refresh, 2026-06-27

I refreshed the slowdown read in the form that should be used before any new
SR24 optimization:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_ceiling_refresh_20260627/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix4_thr07_trace_rejected_base_refresh_20260627/report.md
```

The seven components currently read as:

| component | current evidence | implication |
| --- | --- | --- |
| scheduler / mask build | route-all-contiguous still spends `5.791 ms/step` in scheduler mask wall time and `5.381 ms/step` in row-index/bucket time; the low-sync batched builder itself is sub-ms in clean rows | dynamic row construction is a bad serving path unless it becomes preallocated or GPU-fused |
| base sparse Linear | rows=512 gate/up base sparse graph is `0.353 ms` versus dense `0.538 ms`; down is `0.166 ms` versus dense `0.292 ms` | base-only sparse work can be faster; the sparse base is not the reason `base_only_24` is slow |
| residual correction | at full residual, gate/up exact sparse+residual paths are `1.45x-1.98x` dense and down paths are `1.19x-1.68x` dense | exact `all_corrected_24` without dense fastpath has no speed headroom in the current two-pass form |
| gather/scatter | microbench gather/scatter grows with residual rows; gate/up residual 25% already has `0.100 ms` gather/scatter around a `0.166 ms` dense-row GEMM | gather/scatter is secondary at low residual rates, but it becomes a real cost once quality protection pushes residual rows high |
| routing statistics | prefix4/t0.7 trace has residual fraction `0.6186`, accepted base-only fraction `0.0133`, but rejected base-only fraction `0.0697` | accepted-only risk understated the quality problem; first rejected base-only tokens also affect recovered-token logits |
| CUDA Graph | `base_only_24` clean rows keep mostly `FULL`, while mixed `speclink_t08` rows are often `NONE` unless promoted to all-residual dense | graph recovery helps but is not sufficient if the mixed operator still pays two passes |
| GPU util | clean mixed rows are usually GPU-busy, while route-all-contiguous drops to `64.316%` avg GPU util | the default problem is inefficient useful work, not idle GPU; dynamic route-all adds underutilized small/dynamic kernels |

The operator ceiling is the strongest negative result. For rows=512,
gate/up dense graph is about `0.538 ms`; a serving-like mixed path is already
`1.03x` dense at 12.5% residual, `1.14x` dense at 25%, and `2.26x` dense at
100%. The cached compressed-dense residual path is CUDA-resident, but still
reaches `1.98x` dense at 100% residual because it does sparse base plus
residual GEMM plus add/scatter. For down, the mixed path only wins at small
residual fractions and reaches `1.77x` dense at 100% residual.

The quality projection also changes the controller target. On the prefix4/t0.7
GSM8K trace, accepted base-only is only `1.33%`, but rejected base-only is
`6.97%`. A score-threshold projection that constrains both accepted and
rejected base-only risk to at most `5%` needs prefix `4`, threshold `0.6`, and
residual fraction `0.6696`; driving both risks below `2%` needs residual
fraction `0.8777`. At those residual fractions, the current mixed operator is
not expected to beat dense.

Conclusion: the next optimization should not be another threshold-only sweep.
The threshold can reduce quality risk only by raising the residual fraction,
and the current operator becomes dense-time or slower in exactly that regime.
The useful path is either:

1. a fused/packed base+residual operator that avoids sparse-base plus residual
   correction on the same rows, or
2. a stronger routing signal that keeps both accepted and rejected base-only
   risk low while leaving residual fraction well below the break-even range.

## Critical-Prefix Candidate Check, 2026-06-27

I extended
`examples/evaluate/eval-guidellm/scripts/analyze_sr24_acceptance_trace.py`
with an offline projection for the runtime `critical_prefix` policy. Unlike the
previous high-confidence projection, this mirrors the runtime rule: correct the
high-confidence prefix through the first low-confidence draft token, plus
optional `extra_after_low` rows. The analysis output is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix4_thr07_trace_policy_projection_refresh_20260627/report.md
```

On the earlier prefix4/t0.7 trace, `critical_prefix` looked better than
`high_confidence+prefix`: for joint accepted/rejected base-only risk `<=5%`,
`high_confidence+prefix` needed residual fraction `0.6696`, while
`critical_prefix` needed only `0.5164` with prefix `4`, extra `0`, threshold
`0.7`.

I then ran that candidate on real serving throughput:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_prefix_t07_prefix4_throughput_bs64_math128_20260627/clean_serving/report.md
```

| method | full-batch output tok/s | total output tok/s | avg accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | `3017.788` | `2182.388` | `1.395` | `80.5%` | n/a |
| `critical_prefix@0.7,prefix4` | `2460.330` | `1744.432` | `1.406` | `85.6%` | `{"NONE":64}` |

This is still only `0.815x` dense full-batch throughput. The candidate is not
slow because it reduces acceptance or leaves the GPU idle; acceptance is
slightly higher than dense and GPU utilization is higher. It is slow because
the mixed operator remains eager and pays sparse-base plus residual correction.

The matching GSM8K-20 quality gate was also negative:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_prefix_t07_prefix4_quality_gsm8k20_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_prefix_t07_prefix4_quality_trace_analysis_20260627/report.md
```

| metric | dense | candidate |
| --- | ---: | ---: |
| GSM8K-20 flexible exact match | `0.7000` | `0.6000` |
| paired regressions / improvements | `0 / 0` | `2 / 0` |
| accepted requested-base fraction | n/a | `0.0351` |
| rejected requested-base fraction | n/a | `0.0508` |
| mean residual rows/step | n/a | `4.1384` |

So this is a rejected policy candidate. It reduces the projected risk versus
simple high-confidence routing, but not enough to preserve quality on the
paired GSM8K smoke, and it is still much slower than dense. If quality is
protected further by adding prefix 5/6 or lowering thresholds, the offline
projection predicts residual fraction `0.63-0.75+`, which is already in the
operator regime where the current implementation cannot beat dense.

## Row-Routed MLP Ceiling Check, 2026-06-27

I added a fresh whole-MLP microbenchmark for the row-routed direction:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_refresh_20260627/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_rows1024_refresh_20260627/summary.md
```

This benchmark compares dense MLP, sparse base MLP, Linear-level replacement,
row-routed MLP, and exact-down row-routed MLP. The exact-down variant is the
quality-conservative one because the selected dense rows stay dense through
both `gate_up_proj` and `down_proj`.

| rows | dense rows | dense graph ms | exact-down graph ms | exact-down / dense | read |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 32 | `0.8660` | `0.8164` | `0.94x` | too small to reach `1.2x` after serving overhead |
| 512 | 128 | `0.8675` | `0.7490` | `0.86x` | best rows=512 point, still only about `1.16x` operator speedup |
| 512 | 256 | `0.8680` | `0.8627` | `0.99x` | residual rows already too high |
| 1024 | 32 | `1.6837` | `1.3314` | `0.79x` | enough operator headroom if routing remains graph-safe |
| 1024 | 128 | `1.6958` | `1.3171` | `0.78x` | best large-row region, around `1.29x` operator speedup |
| 1024 | 256 | `1.6937` | `1.3808` | `0.82x` | still above the `1.2x` target before serving overhead |
| 1024 | 512 | `1.6950` | `1.4570` | `0.86x` | headroom shrinks quickly as dense rows grow |

This changes the implementation direction:

1. `base_only_24` is not fundamentally slow in the scoped MLP/gate-up path.
   It can keep CUDA Graph coverage and sometimes improves accepted draft
   tokens. If a base-only run is slow, first check leaf scope, CUDA Graph mode,
   and GPU util; do not assume accepted-length collapse.
2. `all_corrected_24` without dense fastpath is still not a speed path. It is
   exact, but the current two-pass sparse-base plus residual correction costs
   too much when the residual fraction is high.
3. `speclink_t08` needs a packed or fused row-routed operator only if the
   controller can keep dense/corrected rows in the low-to-moderate range. The
   rows=1024 microbench says a graph-safe exact-down packed path could have
   enough headroom at 32-256 dense rows, but rows=512 and rows near all-dense do
   not have enough margin after scheduler and serving overhead.
4. The next useful speed prototype should therefore combine two requirements:
   keep residual rows bounded by a quality-safe controller, and execute the
   chosen dense/base rows in a graph-safe packed path. Dynamic per-step
   `nonzero`/bucket assembly remains a known bad serving path.

## Row-Routed Reuse-Base Prototype, 2026-06-27

I implemented an explicit runtime switch:

```text
SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT=1
```

Runner flags:

```text
--sr24-row-routed-mlp-reuse-base-output
```

The implementation is intentionally isolated to `SPECLINK_SR24_ROW_ROUTED_MLP`.
It skips scheduler-side bucket-complement construction and, inside the Llama
MLP, computes the sparse-base MLP for all rows before overwriting the selected
dense bucket rows. This trades extra sparse work on overwritten rows for stable
shapes and much lower scheduler row-index overhead.

Validation artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_bs128_bucket64_probe_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_bs128_bucket128_probe_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_quality_gsm8k20_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_bucket128_quality_gsm8k20_20260627/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_reusebase_microbench_20260627/summary.md
```

Serving speed on Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, bs128,
max new tokens 128:

| candidate | full-batch tok/s | total tok/s | CUDA Graph | scheduler row-index/bucket | read |
| --- | ---: | ---: | --- | ---: | --- |
| dense reference from same comparison root | `3363.972` | `2620.309` | n/a | n/a | reference |
| old bucket64 row-routed | `3529.843` | `2545.426` | `{"FULL":62,"NONE":2}` | `75.841 ms/step` | dynamic complement dominates scheduler accounting |
| reuse-base bucket64 | `3568.978` | `2532.668` | `{"FULL":62,"NONE":2}` | `0.118 ms/step` | complement bottleneck fixed; full-batch only `1.061x` dense |
| reuse-base bucket128 | `3514.762` | `2492.769` | `{"FULL":62,"NONE":2}` | `0.117 ms/step` | more correction rows, slower than bucket64 |

Quality gates:

| candidate | dataset | dense | SR24 | paired regressions | paired improvements | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| reuse-base bucket64 | Llama GSM8K-20 | `0.6500` | `0.5500` | `2` | `0` | not quality-safe |
| reuse-base bucket64 | Qwen GSM8K-20 | `0.9000` | `0.8500` | `1` | `0` | not quality-safe |
| reuse-base bucket128 | Llama GSM8K-20 | `0.6500` | `0.6500` | `1` | `1` | aggregate-neutral, but not paired-stable |

The updated microbenchmark explains the mixed result. For rows=1024,
exact-down reuse-base graph time is promising in isolation:

| bucket | dense graph ms | exact-down reuse-base graph ms | reuse-base / dense |
| ---: | ---: | ---: | ---: |
| 64 | `1.6833` | `1.2823` | `0.76x` |
| 128 | `1.6921` | `1.3218` | `0.78x` |

But the serving result remains far below the `1.2x` end-to-end target because
the operator improvement applies only to the targeted MLP leaf/layer subset and
does not remove the rest of the decode work. This prototype is useful because
it eliminates the scheduler complement problem; it is not a final SR24 path
until paired accuracy is stable and the remaining model-level speed gap is
closed.

## Seven-Part Breakdown Refresh With bs128 Reuse-Base Flags, 2026-06-27

Fresh breakdown artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_reusebase_bs128_math_combined_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_reusebase_bs128_math_combined_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_breakdown_reusebase_bs128_math128_20260627_1720/component_microbench/summary.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency
bs128. Clean serving used max new tokens 128; component diagnostics used max
new tokens 64 and forced eager because CUDA-event timing is not CUDA-Graph
capture safe.

Clean serving result:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph | scheduler wall |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| dense | `3364.397` | `2622.774` | `1.423` | `92.0%` | n/a | n/a |
| `base_only_24` | `3867.182` | `2760.388` | `1.608` | `91.4%` | `{"FULL":62,"NONE":2}` | `0.362ms/step` |
| `speclink_t08` | `3585.037` | `2551.122` | `1.630` | `92.1%` | `{"FULL":62,"NONE":2}` | `1.020ms/step` |

This confirms the current slowdown read:

- not accepted length: `speclink_t08` accepts slightly more draft tokens than
  dense;
- not global GPU idle: average GPU util is about `92%`;
- not mainly CUDA Graph loss in this graph-enabled row: `FULL=62,NONE=2`;
- remaining gap is useful-work efficiency: `speclink_t08` is `1.066x` dense
  full-batch but still below dense on total tok/s, and well below the `1.2x`
  target.

Eager diagnostic row for `speclink_t08`:

| part | value | read |
| --- | ---: | --- |
| scheduler / mask build | `64.557ms/step` diagnostic wall, `64.379ms/step` request loop | exact-routing diagnostics are sync-heavy; use clean row for real serving overhead |
| clean scheduler / mask build | `1.020ms/step`, row bucket `0.117ms/step` | visible but not enough to explain the full speed gap |
| base sparse Linear | `1.230ms/call`, `2051.8` rows/call | the largest measured GPU-side component |
| residual correction | `0.150ms/call`, bucket rows/call `64` | secondary in this bucketed row |
| gather/scatter | `0.018ms/call` | not the first bottleneck here |
| routing stats | draft residual/base `30979/45`, non-draft residual/base `3878/5750`, bucket fill `0.828` | nearly all draft rows are requested residual before bucket capping; quality-safe routing is still too dense for a two-pass design |
| CUDA Graph | clean `FULL=62,NONE=2`, diagnostic `NONE=47` | graph data must come from clean rows, because component timing forces eager |
| GPU util | clean `92.1%`, diagnostic `81.0%` | diagnostic sync lowers util; clean run shows the GPU is busy |

Important configuration finding: in this run
`--sr24-row-routed-mlp --sr24-row-routed-mlp-reuse-base-output` were enabled,
but the row-routed MLP path did not execute. The reason is the layer scope:
`gate_up_proj=16-31` and `down_proj=8-15` have no overlapping MLP layers.
`row_routed_mlp_output()` requires both `gate_up_proj` and `down_proj` in the
same layer to be attached to SR24. Therefore the measured `speclink_t08` row is
still a Linear/bucket path, not a real MLP-level reuse-base path.

Follow-up rule: future row-routed experiments must report
`row_routed_mlp_calls` or `row_routed_mlp_reuse_base_output_calls`. Do not
interpret `sr24_row_routed_mlp=True` as proof that row-routed MLP actually ran.

## True Overlap Row-Routed Check, 2026-06-27

Fresh artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_bs128_math_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_bs128_math_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_eagerclean_bs128_math_20260627/clean_serving/report.md
```

This run attached both `gate_up_proj` and `down_proj` to layers `16-31`, so the
row-routed MLP path really executed. The diagnostic run reported
`row_routed_mlp_reuse_base_output_calls=16`.

Clean serving with CUDA Graph enabled did not start. During graph capture,
`row_routed_mlp_output()` fell back to a dynamic bucket path and called:

```text
vllm/vllm/speclink_sr24.py:7872 bucket_values.to(dtype=torch.bool).nonzero(...)
```

CUDA rejected this with `operation not permitted when stream is capturing`.
This means the current true row-routed path is not CUDA-Graph-safe.

Eager clean serving did run:

| method | full-batch tok/s | total tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | `3365.482` | `2624.085` | `1.422` | `91.9%` | n/a |
| true overlap `speclink_t08` | `2159.013` | `1486.537` | `0.331` | `91.2%` | `{"NONE":128}` |

The slowdown here is not GPU idle. GPU util stays high, but the accepted draft
length collapses and the runtime does much more work for fewer useful tokens.

The diagnostic component split explains why:

| part | value |
| --- | ---: |
| reuse-base MLP total | `6.445ms/call` |
| sparse base MLP over all rows | `6.167ms/call` |
| dense correction for selected rows | `0.278ms/call` |
| gather/scatter | `0.014ms/call` |
| rows per call | `5746` |
| dense rows per call | `64` |

So the current true row-routed reuse-base design is dominated by computing the
sparse-base MLP for every row and then overwriting only a small dense bucket.
It is a useful diagnostic but not a final speed path.

Updated seven-part diagnosis:

| part | current read |
| --- | --- |
| scheduler / mask build | clean low-sync timing is small in the eager run: `0.396ms/step`, with row bucket `0.088ms/step`; exact diagnostic timings are sync-heavy and should not drive decisions. |
| base sparse linear / MLP | main bottleneck when true row-routed fires: the all-row sparse-base MLP costs `6.167ms/call`. |
| residual correction | only `0.278ms/call`; it is secondary in this configuration. |
| gather/scatter | `0.014ms/call`; not the current first bottleneck. |
| routing statistics | the diagnostic route had all draft rows residual and only a 64-row dense bucket per MLP call; this is too much residual pressure for a reuse-base design. |
| CUDA Graph | graph-on true row-routed fails at capture because of dynamic `nonzero`; graph-off runs but is much slower. |
| GPU util | high util means the GPU is busy with inefficient useful-work structure, not waiting idle. |

Next implementation direction:

1. Do not continue optimizing the all-row reuse-base MLP path as the final
   `speclink_t08` route.
2. If row-routed MLP remains the direction, replace it with a graph-safe packed
   route: compute dense rows and base rows separately with static bucket shapes,
   avoid capture-time `nonzero`, and avoid running sparse base on rows that are
   later overwritten by dense.
3. Keep paired accuracy gates in the loop. The true overlap run's accepted
   draft length collapse indicates a quality/output-trajectory problem, not
   only an operator-speed problem.

## Packed Row-Routed Graph Slice, 2026-06-27

Implementation change:

```text
vllm/vllm/speclink_sr24.py
```

The CUDA Graph capture context now attaches persistent static bucket row and
base-row buffers to `VerifyResidualPlan`, so `row_routed_mlp_output()` no longer
falls back to capture-time `bucket_values.nonzero()`. The packed row-routed
path should be run with `SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT=0`.

A fixed-shape Triton bucket-complement kernel was also added. It builds
`base_rows` as the complement of the selected bucket rows without dynamic
`nonzero`. The important detail is that `output_count` is a runtime argument,
not a Triton `constexpr`, so the kernel is not recompiled for every scheduled
row count. The kernel is prewarmed at SR24 attach time when row-routed MLP and
residual bucket routing are enabled.

Validation artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_packed_graph_smoke_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_packed_graph_triton_runtime_smoke_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_packed_graph_triton_runtime_bs128_math_gmem098_20260627/clean_serving/report.md
```

Correctness smoke for the complement kernel passed against a CPU reference.
`conda run -n spec python -m py_compile vllm/vllm/speclink_sr24.py` also passed.

Effect on the original Graph blocker:

| run | Graph | scheduler row-index/bucket | full-batch tok/s | read |
| --- | --- | ---: | ---: | --- |
| pre-packed true overlap | startup failed | n/a | n/a | capture-time `nonzero` failed |
| packed, before runtime-count fix | `FULL=62,NONE=2` | `26.122ms/step` | `1421.608` | service starts, but complement JIT/sort path dominates |
| packed + runtime-count + prewarm, bs32 | `FULL=30,NONE=2` | `3.213ms/step` | `1644.876` | Graph blocker fixed; scheduler improved but still visible at small bs |
| packed + runtime-count + prewarm, bs128 | `FULL=126,NONE=2` | `0.140ms/step` | `2180.612` | scheduler/bucket is no longer the main bottleneck |

The bs128 run needed `--gpu-memory-utilization 0.98`. At the default `0.84`,
vLLM failed before serving because CUDA Graph memory profiling estimated
`5.42GiB` graph memory and left `-0.23GiB` available KV cache.

The bs128 comparison against the same-root dense baseline is still negative:

| method | full-batch tok/s | total tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | `3360.792` | `2615.777` | `1.419` | `92.1%` | n/a |
| packed row-routed `speclink_t08` | `2180.612` | `1528.253` | `0.330` | `95.0%` | `FULL=126,NONE=2` |

Updated conclusion: the Graph and scheduler problems are now mostly removed for
the packed row-routed path at bs128, but the method remains slow because the
accepted draft length collapses. The next work should not be another scheduler
or bucket-complement pass. It should fix the routing/quality policy so accepted
tokens are not routed through too much base-only computation, then remeasure
operator speed.

## User-Requested Seven-Part Breakdown Refresh, 2026-06-27

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs128_math_current_20260627_180905/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_instrumented_current_20260627_180905/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs128_math_packed_clean_20260627_180905/component_summary/report.md
```

Important caveat: the first bs128 refresh accidentally set
`--sr24-row-routed-mlp-max-base-rows 256`, so `speclink_t08` skipped the real
packed row-routed MLP path and fell back to the Linear/bucket path. That run is
still useful for fallback-path component attribution:

| part | fallback-path read |
| --- | --- |
| scheduler / mask build | clean low-sync `speclink_t08` is only `0.501ms/step`; exact diagnostic request-loop timing can hit `39.735ms/step`, but that is sync-heavy instrumentation, not clean serving. |
| base sparse Linear | diagnostic bs64 `speclink_t08` reports `1.427ms/call`; per leaf: `gate_up_proj=1.626ms/call`, `down_proj=1.228ms/call`. |
| residual correction | diagnostic dense-row correction is only `0.138ms/call`. |
| gather/scatter | `0.014ms/call`; not the current primary cost. |
| routing stats | draft residual/base rows were `16297/15`; bucket fill `0.785`; this means almost all draft rows still request residual before capping. |
| CUDA Graph | clean fallback path keeps `FULL=62,NONE=2`. |
| GPU util | clean fallback path is busy (`92.1%`), so the GPU is not simply idle. |

The corrected packed clean row uses `--sr24-row-routed-mlp-max-base-rows 0`.
It confirms the current packed-path slowdown:

| method | full-batch tok/s | total tok/s | accepted draft/step | scheduler wall | row bucket | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| packed `speclink_t08` | `2113.161` | `1493.588` | `0.295` | `0.461ms/step` | `0.141ms/step` | `94.4%` | `{"FULL":126,"NONE":2}` |

This separates the current bottleneck:

1. It is not scheduler/mask-build first: clean scheduler wall time is sub-ms.
2. It is not CUDA Graph first: the packed row has only two `NONE` graph steps.
3. It is not gross GPU underutilization: sampled util is about `94%`.
4. It is primarily routing/quality/useful-work: accepted draft tokens collapse
   to about `0.30/step`, so the speculative path loses its main source of
   speedup while still paying sparse/residual machinery overhead.

Next optimization should therefore start with routing quality. In particular,
avoid policies that leave nearly every draft row residual-requested but then
correct only a small global bucket. A useful next ablation is a per-request
fair bucket or priority policy that protects early draft positions before
bonus/non-draft rows, followed by a clean packed-row rerun.

## Bonus-Priority and Bucket Sweep, 2026-06-27

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_priority1_bs128_math_packed_clean_20260627_VALID/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_priority1_bucket128_bs128_math_packed_clean_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_priority1_bucket256_bs128_math_packed_clean_20260627_rerun/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_exact_routing_bonus1_bucket64_bs64_math_stats1_20260627/seven_part_report/report.md
```

Code change: `SPECLINK_SR24_BONUS_PRIORITY` is now configurable and exposed as
`--sr24-bonus-priority` in the matrix and slowdown runners. The default remains
`4.0`, preserving old behavior. The ablations below use `1.0` to keep the
global residual bucket from being dominated by speculative bonus/non-draft rows.

Clean bs128/math/K8/max128 packed-row results:

| config | full-batch tok/s | total tok/s | accepted draft/step | scheduler mask | row bucket | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bucket64, bonus priority old/default | `2113.161` | `1493.588` | `0.295` | `0.461ms/step` | `0.141ms/step` | `94.4%` | `{"FULL":126,"NONE":2}` |
| bucket64, bonus priority `1.0` | `2810.116` | `1849.070` | `0.720` | `0.468ms/step` | `0.152ms/step` | `94.0%` | `{"FULL":94,"NONE":2}` |
| bucket128, bonus priority `1.0` | `2990.636` | `1852.452` | `0.807` | `2.708ms/step` | `2.043ms/step` | `88.3%` | `{"FULL":94,"NONE":2}` |
| bucket256, bonus priority `1.0` | `2743.412` | `1845.582` | `0.777` | `4.868ms/step` | `4.203ms/step` | `88.7%` | `{"FULL":94,"NONE":2}` |

Lowering the bonus priority is a real improvement, but it is not enough. The
accepted draft length rises from `0.295` to about `0.72-0.81`, while still far
below the dense/EAGLE3 reference range from earlier runs. Increasing the bucket
above 64 does not scale cleanly: it raises scheduler row-bucket cost from
`0.152ms/step` to `2.043ms/step` at bucket128 and `4.203ms/step` at bucket256,
without a proportional accepted-length gain.

Exact routing diagnostic, bs64/math/K8/max32 with CPU sync and stats interval 1:

| metric | value |
| --- | ---: |
| draft residual/base tokens | `10472 / 0` |
| non-draft residual/base tokens | `1309 / 3788` |
| bucket active/requested rows | `9 / 11781` |
| bucket active fraction of requested | `0.000764` |
| scheduler mask wall | `36.278ms/step` |
| CUDA Graph | `{"NONE":30}` |

This diagnostic row is not a throughput number because CPU sync disables the
clean graph path and makes scheduler timing huge. It is useful for routing: in
the exact count, every valid draft row requests residual correction, but the
global capped bucket admits almost none of the requested residual rows. That
explains why clean serving can keep the GPU busy yet still accept very few
draft tokens.

Current optimization direction: stop treating the residual bucket as a global
top-k over all rows. The next candidate should allocate the capped budget
fairly across requests and early draft positions, then optionally spend leftover
rows on bonus/non-draft correction. If that recovers accepted length, optimize
the row-bucket implementation; if not, the `all_if_any_low` routing policy
itself needs to be replaced by a stricter per-token confidence gate.

## Draft-Position Priority Sweep, 2026-06-27

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority_smoke_bs16_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority2_bucket64_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority10_bucket64_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_position_priority10_bucket128_bs128_math_20260627/clean_serving/report.md
```

Code change: `SPECLINK_SR24_DRAFT_POSITION_PRIORITY_SCALE` is now exposed as
`--sr24-draft-position-priority-scale`. Default `0.0` preserves old behavior.
Positive values add a draft-position band to residual priority so earlier draft
positions dominate the global bucket selection before later draft rows.

Clean bs128/math/K8/max128 packed-row results with bonus priority `1.0`:

| config | full-batch tok/s | total tok/s | accepted draft/step | scheduler mask | row bucket | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bucket64, scale `0` | `2810.116` | `1849.070` | `0.720` | `0.468ms/step` | `0.152ms/step` | `94.0%` | `{"FULL":94,"NONE":2}` |
| bucket64, scale `2` | `2522.947` | `1962.705` | `0.799` | `0.459ms/step` | `0.145ms/step` | `90.8%` | `{"FULL":94,"NONE":2}` |
| bucket64, scale `10` | `2762.055` | `1915.171` | `0.866` | `0.462ms/step` | `0.148ms/step` | `93.7%` | `{"FULL":94,"NONE":2}` |
| bucket128, scale `0` | `2990.636` | `1852.452` | `0.807` | `2.708ms/step` | `2.043ms/step` | `88.3%` | `{"FULL":94,"NONE":2}` |
| bucket128, scale `10` | `3101.722` | `2143.421` | `1.102` | `2.342ms/step` | `2.040ms/step` | `86.7%` | `{"FULL":94,"NONE":2}` |

This confirms the quality hypothesis: making early draft rows dominate the
bucket raises acceptance, and bucket128 plus scale 10 is the best current
candidate. It is still below the dense baseline from the same development pass
(`3360-3365` full-batch tok/s and about `2615` total tok/s). The remaining
obvious waste is the global top-k bucket path: bucket128 needs about
`2.04ms/step` just for row-bucket scheduling. The next implementation should
build a direct per-request/position bucket, e.g. position 0 for every active
request before position 1, instead of sorting/top-k over every scheduled row.

## Breakdown-First Pivot, 2026-06-27

New combined diagnosis artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_diagnosis_user_table_20260627/report.md
```

Current-parameter instrumented artifact, using the same bucket/priority family
as the best clean row but with bs64 to avoid bs128 exact-routing OOM risk:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_diagnosis_current_best_instrumented_bs64_20260627/component_summary/report.md
```

The updated seven-part table separates clean serving evidence from sync-heavy
diagnostic evidence:

| part | current read |
| --- | --- |
| scheduler / mask build | Best clean bs128 bucket128/position-scale10 run spends `2.342ms/step` in SR24 scheduler mask wall time, including `2.040ms/step` in row-index/bucket construction. This is now a visible clean-serving bottleneck. |
| base sparse linear | Current row-routed diagnostic shows base-side work is dominated by `base_gate_up=1.330ms/call` over about `1521` base rows/call. |
| residual correction | Dense correction is smaller: `0.290ms/call` total, with `0.175ms` dense gate/up and `0.102ms` dense down over `128` dense rows/call. |
| gather/scatter | Row-routed gather/scatter is visible but secondary: `0.144ms/call`, mostly `assemble=0.128ms`. |
| routing statistics | Current exact diagnostic routes nearly all draft rows to residual: draft residual/base `30462/10`; non-draft residual/base `3809/3787`. This protects quality but leaves little sparse-base-only draft benefit. |
| CUDA Graph | Best clean bs128 row has `{"FULL":94,"NONE":2}`. Graph coverage is not the first suspect for that path. |
| GPU util | Best clean bs128 row has `86.7%` average GPU util and `100%` peak. The GPU is busy; the problem is useful-work efficiency, not gross idleness. |

This changes the next optimization target. Do not start with another high-level
controller sweep. First reduce the clean row-bucket build cost and the
row-routed base-side work, then re-check accepted length. A direct
per-request/position bucket builder is only useful if it also avoids the
bucket-complement and row-routed base-side overhead; just replacing global
top-k is not enough.

## Direct Bucket And Reuse-Base Follow-Up, 2026-06-27

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_position_bucket_bs128_math_gpumem095_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_position_bucket_staticcopy_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_bucket128_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_instrumented_bs64_math_20260627/component_summary/report.md
```

The first direct-position bucket attempt was a useful negative diagnostic. It
created a CUDA tensor from a Python list every step, which pushed clean
`scheduler_batched_mask_builder` to `56.090ms/step`. That has been fixed by
copying through a reused pinned CPU int64 buffer into the static GPU bucket.

Clean bs128/math/K8/max128 comparison:

| config | full-batch tok/s | total tok/s | accepted draft/step | scheduler mask | batched builder | row bucket/index | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bucket128, position scale 10 | `3101.722` | `2143.421` | `1.102` | `2.342ms/step` | `0.219ms/step` | `2.040ms/step` | `86.7%` | `{"FULL":94,"NONE":2}` |
| direct bucket, before static copy fix | `3085.288` | `2191.381` | `1.150` | `58.207ms/step` | `56.090ms/step` | `2.033ms/step` | `88.0%` | `{"FULL":94,"NONE":2}` |
| direct bucket, static copy fix | `2994.575` | `2183.009` | `1.077` | `3.883ms/step` | `0.859ms/step` | `2.936ms/step` | `88.7%` | `{"FULL":62,"NONE":2}` |
| bucket128 + reuse base output | `3102.340` | `2221.430` | `1.064` | `0.465ms/step` | `0.273ms/step` | `0.107ms/step` | `92.0%` | `{"FULL":62,"NONE":2}` |

Read:

1. Direct position ordering does not currently improve the best end-to-end
   row. It removes neither the row-routed base cost nor the bucket-complement
   cost, and the static-copy fix only removes the accidental Python/CUDA
   allocation overhead.
2. `row_routed_mlp_reuse_base_output` proves that bucket-complement construction
   is the scheduler-side cost to remove: row bucket/index drops from
   `2.040ms/step` to `0.107ms/step`.
3. Even after scheduler cost drops below `0.5ms/step`, throughput stays around
   `3100` full-batch tok/s, still below the dense target. The remaining
   bottleneck is the row-routed MLP operator itself.

The reuse-base instrumented row localizes the operator cost:

| component | value | read |
| --- | ---: | --- |
| row-routed MLP reuse total | `3.032ms/call` | diagnostic only, but it shows the operator-side scale |
| sparse base side | `2.709ms/call` | dominant cost after scheduler is reduced |
| dense correction side | `0.323ms/call` | secondary, but still additive |
| dense rows/call | `128` | capped correction bucket |
| reuse rows/call | `1833` base / `128` dense | most work is still sparse-base-side rows |

Updated optimization direction: do not keep iterating on direct bucket alone.
The next candidate should combine graph-safe row planning with a row-routed MLP
that avoids expensive sparse-base work for rows that do not provide enough
accepted-token benefit, or fuses the sparse/base and correction assembly so the
reuse-base path is not a separate large operator.

## Same-Condition Operator Follow-Up, 2026-06-27

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_probe_rows1024_buckets_20260627/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_route_assembly_bucket128_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket128_scale10_repeat_gpumem095_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_baseline_bs128_math_gpumem095_20260627/clean_serving/report.md
```

The row-routed MLP microbench showed that the quality-conservative exact-down
path can be faster than a dense MLP at the isolated operator level:

| rows | bucket | dense graph | exact-down Triton graph | exact-down no-assemble graph | reuse-base graph |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 128 | `1.697ms` | `1.314ms` | `1.274ms` | `1.322ms` |

However, the serving result does not turn that isolated operator win into an
end-to-end speedup. Same-condition bs128/math/K8/max128 rows with
`gpu_memory_utilization=0.95`:

| method/config | total tok/s | total vs dense | full-batch tok/s | full vs dense | accepted draft/step | GPU util | scheduler mask | row bucket/index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense baseline | `2620.402` | `1.000x` | `3294.699` | `1.000x` | `1.423` | `92.0%` | n/a | n/a |
| bucket128/scale10 repeat | `2053.933` | `0.784x` | `3005.977` | `0.912x` | `1.050` | `93.3%` | `0.435ms/step` | `0.139ms/step` |
| bucket128/scale10 + Triton assembly | `2082.782` | `0.795x` | `2983.379` | `0.906x` | `1.030` | `93.4%` | `0.443ms/step` | `0.142ms/step` |
| bucket128/scale10 + reuse base output | `2221.430` | `0.848x` | `3102.340` | `0.942x` | `1.064` | `92.0%` | `0.465ms/step` | `0.107ms/step` |

Read:

1. Current code already has low clean scheduler cost for the bucket128/scale10
   family. The old `2.04ms/step` row-bucket number is not the current limiting
   cost after the static-buffer/bucket-complement fixes.
2. Triton final assembly is not the missing end-to-end speed path. It lowers
   neither total latency nor accepted draft length enough to matter.
3. Reuse-base is the best current clean serving variant in this small same-
   condition comparison, but it is still only `0.848x` dense total tok/s and
   `0.942x` dense full-batch tok/s.
4. The remaining first-order problem is now accepted-token value versus
   row-routed sparse-base work: SR24 accepts about `1.06` draft tokens/step
   versus dense EAGLE3's `1.42`, while GPU utilization is already high.

Next constraint: an operator-only microbenchmark improvement is insufficient.
The next SR24 candidate must either recover accepted draft length while keeping
the scheduler low, or use a sharper routing signal so the row-routed MLP work
is spent only on rows that materially increase accepted tokens. Assembly-only
or bucket-selector-only changes should be treated as secondary ablations.

## Direct-Position Vector Bucket Smoke, 2026-06-27

New artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_position_vector_smoke_bs16_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_direct_position_vector_bucket64_smoke_bs16_20260627/clean_serving/report.md
```

Code change: `_build_direct_position_bucket_from_active()` now has a
device-vectorized fast path for full draft-position buckets. When every active
request has enough valid draft rows to fill the bucket, it constructs
position-major rows on GPU from the per-request start rows instead of building a
Python list of every selected row. It falls back to the older selected-list path
when padding or speculative bonus rows would be needed.

Small bs16/math/K8/max32 smoke results:

| config | full-batch tok/s | total tok/s | accepted draft/step | scheduler mask | batched builder | row bucket/index | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| direct position bucket32 | `772.293` | `440.418` | `0.443` | `4.676ms/step` | `2.416ms/step` | `2.191ms/step` | `51.3%` | `{"FULL":26,"NONE":2}` |
| direct position bucket64 | `765.641` | `431.304` | `0.536` | `4.510ms/step` | `0.737ms/step` | `3.704ms/step` | `43.0%` | `{"FULL":26,"NONE":2}` |

Read:

1. The vector path reduces part of the bucket64 builder cost, but it does not
   repair accepted length. Accepted draft tokens/step remain only `0.44-0.54`.
2. The small bs16 smoke is underfilled and should not be used as a throughput
   comparison against bs128 rows; it is only a routing/scheduler diagnostic.
3. Direct early-position allocation by itself is not a sufficient quality
   signal. The next useful policy should be confidence/value-aware and should
   spend row-routed work only where it can increase accepted draft tokens.

This reinforces the current pivot: do not keep optimizing bucket construction
alone. A candidate must first recover accepted draft length, then prove with the
same seven-part breakdown that scheduler/mask, base sparse, residual correction,
gather/scatter, routing stats, CUDA Graph, and GPU util all move in the right
direction.

## Breakdown-First Refresh For Current t08 Path, 2026-06-27

New artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_slow_parts_bs64_math128_20260627/seven_part_report/report.md
```

Run condition: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, GuideLLM
client-side concurrency/batch size 64, max new tokens 128. The run separates
clean serving rows from diagnostic rows with CUDA event timing.

Key clean-serving rows:

| method | full-batch tok/s | total tok/s | vs dense full | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 baseline | `3025.137` | `2186.396` | `1.000x` | `79.4%` | `1.396` | n/a |
| base_only_24 | `3242.405` | `2228.744` | `1.072x` | `83.6%` | `1.547` | `{"FULL":62,"NONE":2}` |
| speclink_t08 | `2468.236` | `1746.067` | `0.816x` | `82.8%` | `1.438` | `{"NONE":64}` |
| all_corrected_24 diagnostic | `2911.797` | `2006.356` | `0.963x` vs dense full | `86.6%` | `1.402` | `{"NONE":84}` |

Seven-part read for `speclink_t08`:

| component | measured value | read |
| --- | ---: | --- |
| scheduler / mask build | clean `0.674ms/step` | not the main clean-path bottleneck in this configuration |
| base sparse linear | `0.486ms/call`, gate_up 16-31 rows/call `558.3` | useful only if correction is much smaller |
| residual correction | `0.591ms/call` | larger than sparse base; first operator-side cost to reduce |
| gather/scatter | `0.078ms/call` | visible but secondary to base+correction GEMMs |
| routing stats | draft residual/base `17060/9484`; non-draft residual/base `3318/3788` | too many draft rows still pay dense correction |
| CUDA Graph | clean `{"NONE":64}` | mixed t08 path is fully graph-off |
| GPU util | clean avg `82.8%`, peak `100%` | GPU is busy; slowdown is inefficient useful work, not idleness |

CPU-sync ablation under the same run:

| variant | total tok/s | read |
| --- | ---: | --- |
| low-sync stats on | `1768.063` | close to clean `speclink_t08` |
| low-sync stats off | `1714.933` | stats overhead is not first-order |
| sync mask state | `1671.435` | explicit mask-state sync costs about 5% versus low-sync |
| sync heavy | `1274.765` | heavy per-step sync is bad, but diagnostic-only |
| low-sync GPU counts | `1734.509` | routing counters can be collected without changing the conclusion |

Current conclusion:

1. `base_only_24` is not the slow path here. It is faster than dense and keeps
   CUDA Graph coverage.
2. The current `speclink_t08` slowdown has two first-order causes:
   `SPECLINK_SR24_FORCE_CUDAGRAPH_NONE_FOR_MIXED` sends the mixed path to
   graph `NONE`, and the mixed sparse+dense operator does too much correction
   work (`0.486ms` sparse base plus `0.591ms` dense correction per measured
   call).
3. Clean scheduler/mask construction is sub-ms in the normal t08 row. CPU
   synchronization can be very expensive in diagnostic modes, but it does not
   explain the main clean-path gap.
4. The next optimization should not be another threshold-only controller
   sweep. It should either make the mixed t08 path graph-safe with static
   tensors, or reduce/fuse the correction work so dense correction is not
   larger than the sparse base. Routing must also reduce draft residual rows
   without collapsing accepted draft length.

## Priority Signal And Row-Routed MLP Follow-Up, 2026-06-27

Code change:

- Capped residual-bucket priority now matches the selected policy:
  `high_confidence` uses DLM selected-token probability,
  `prefix_confidence` uses cumulative prefix probability, and
  `low_confidence` keeps risk severity.
- `check_speclink_sr24_correctness.py` now checks slow/batched priority
  equivalence, priority direction, and row-routed MLP equivalence against the
  linear-level mixed MLP path.

Validation:

```text
conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Result:

```text
speclink_sr24_correctness=ok
```

Serving checks:

| run | full-batch tok/s dense | full-batch tok/s SR24 | SR24 accepted draft/step | avg GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| `sr24_prefixconf_prioritysignal_bucket512_bs64_math128_20260627` | `3032.967` | `2078.159` | `1.416` | `89.0%` | priority signal is neutral for the old non-row-routed bucket512 path |
| `sr24_prefixconf_prioritysignal_rowrouted_bucket512_bs64_math128_20260627` | `3026.375` | `1951.156` | `1.223` | `63.5%` | explicit row-routed MLP is worse and underfilled |

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_prioritysignal_bucket512_bs64_math128_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_prioritysignal_rowrouted_bucket512_bs64_math128_20260627/clean_serving/report.md
```

Read: the row-routed MLP arithmetic path is unit-equivalent, but the serving
configuration loses accepted length and GPU utilization. The next optimization
should not be another scalar priority tweak. It should inspect row-routed
serving shapes and bucket selection directly, then recover accepted draft/step
before using row-routed MLP as the main `speclink_t08` speed path.

## Corrected Overlap Breakdown, 2026-06-28

The next check fixed the residual-layer scope for both leafs:
`gate_up_proj=16-31;down_proj=16-31`. This removes the obvious early-layer
`down_proj` pollution risk from the earlier row-routed run.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/seven_part_report/report.md
```

Clean serving:

| method | full-batch tok/s | total tok/s | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3026.088` | `2185.578` | `84.8%` | `1.403` | n/a |
| corrected row-routed t08 | `2597.598` | `1840.189` | `86.2%` | `1.230` | `{"FULL":62,"NONE":2}` |

Breakdown:

- scheduler/mask clean cost is `1.310ms/step`; row bucket/index is
  `1.040ms/step`, bucket build is `0.097ms/step`, and mixed row indices are
  only `0.001ms/step`.
- row-routed sparse base total is `0.577ms/call`, mostly base gate/up
  `0.572ms/call`.
- row-routed dense correction total is `0.760ms/call`, dominated by dense
  gate/up `0.477ms/call` and dense down `0.257ms/call`.
- gather/scatter is secondary: dense gather `0.007ms`, base gather `0.004ms`,
  assemble `0.022ms`.
- routing still corrects many rows: draft residual/base `12223/18889`,
  non-draft residual/base `3889/3788`, bucket fill `0.483`.

Current read: this is not a CUDA Graph problem and not a simple GPU-idle
problem. The path is slow because it both lowers useful speculative progress
and pays a large dense correction MLP on top of the sparse base. Next work
should stop treating row-routed MLP as the main path unless a new selector
recovers accepted draft length first.

## Base-Only And All-Corrected Refresh, 2026-06-28

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_latemlp_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_default_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_directcslt_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_operator_probe_20260628/summary.md
```

Same late-MLP scope:

| row | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense for base-only row | `3021.525` | `2185.492` | `1.395` | `80.6%` | n/a |
| `base_only_24` | `5389.838` | `2739.866` | `2.444` | `81.8%` | `{"FULL":69,"NONE":2}` |
| dense for all-corrected row | `3016.036` | `2180.028` | `1.396` | `80.6%` | n/a |
| `all_corrected_24`, torch_sparse residual | `2551.468` | `1778.923` | `1.400` | `88.4%` | `{"FULL":77,"NONE":2}` |
| `all_corrected_24`, direct cuSPARSELt | `2515.072` | `1752.827` | `1.400` | `89.0%` | `{"FULL":77,"NONE":2}` |

Read:

- `base_only_24` is not slow in this same-scope test. It has normal graph
  coverage and GPU utilization, and its accepted draft length is higher than
  dense. The current slowdown is not caused by sparse base alone.
- `all_corrected_24` is slow with accepted length intact. This isolates the
  problem to exact residual correction work: sparse base plus complementary
  residual sparse GEMM is still slower than one dense GEMM in serving.
- Direct `_cslt_sparse_mm` is a negative serving ablation here, so dynamic alg
  switching should not be the next main implementation.
- The standalone operator probe confirms the needed direction: best current
  exact graph path is still slower than dense for representative Llama MLP
  shapes. `compressed_dense` tensors are on GPU, but cached compressed residual
  is not the speed path. A useful all-corrected optimization needs a fused
  packed base+residual operator, not another wrapper around two separate GEMMs.

## Current Critical-Prefix Candidate, 2026-06-28

Current quality-safe selector:

```text
critical_prefix, threshold=0.6, min_prefix_residual=4, extra_after_low=1
residual leafs: gate_up_proj=16-31;down_proj=8-15
non-draft policy: bonus
```

Quality gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_gsm8k50_20260628/trace_analysis/report.md
```

It reaches GSM8K-50 score `0.7200`, matching the previous dense/spec-safe
reference. Trace risk is much lower than the failed `high_confidence@0.7`
candidate:

| selector | score | accepted base-only frac | rejected base-only frac | mean residual rows/step |
| --- | ---: | ---: | ---: | ---: |
| `high_confidence@0.7,prefix3` | `0.7000` | `0.0441` | `0.1524` | `4.344` |
| `critical_prefix@0.6,prefix4,extra1` | `0.7200` | `0.0228` | `0.0389` | `4.455` |

Same-condition CPU-sync ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/clean_serving_nosync/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/clean_serving_syncmask/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/clean_serving_uniformdirect/report.md
```

| row | full-batch tok/s | total tok/s | accepted draft/step | scheduler mask ms/step | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense reference | `3014.282` | `2180.503` | `1.395` | n/a | same-root baseline |
| no-sync SR24 | `2807.819` | `1779.830` | `1.632` | `0.343` | best current serving variant |
| mask-state sync SR24 | `2691.279` | `1716.161` | `1.632` | `5.411` | about `5ms/step` sync penalty |
| uniform-direct SR24 | `2769.327` | `1735.971` | `1.580` | `0.774` | negative builder ablation |

Full breakdown:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_breakdown_bs64_math128_20260628/seven_part_report/report.md
```

Clean serving row:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3132.894` | `2183.544` | `1.401` | `86.4%` | n/a |
| `critical_prefix` SR24 | `2804.983` | `1780.379` | `1.635` | `77.2%` | `{"NONE":64}` |

Breakdown read:

- scheduler/mask build is not the clean first-order bottleneck:
  `0.372ms/step` total, with batched builder `0.195ms` and bucket/index
  `0.109ms`.
- CPU sync must still be avoided. Same-condition mask-state sync adds about
  `5.04ms/step`, but the no-sync path is still below dense.
- instrumented Linear timing points to sparse base as the largest measured
  mixed-path cost: `0.985ms/call` base sparse, `0.148ms/call` residual
  correction, and `0.012ms/call` gather/scatter.
- CUDA Graph is still completely missing for the mixed SR24 serving row:
  `{"NONE":64}`.
- GPU util is lower than dense (`77.2%` versus `86.4%`) but not near zero, so
  this is not just an idle-GPU issue.

Current decision: keep the CPU-sync reductions, keep `critical_prefix` as the
quality-safe selector for the next candidate, and stop spending time on
`uniform-direct` or sync-heavy counters. The remaining speed work should target
mixed-path CUDA Graph coverage and a fused/packed sparse-base plus residual
operator. Any further controller sweep must be judged by two gates at once:
preserve accepted draft length and lower residual rows enough to beat the
current mixed operator cost.

## Mixed CUDA Graph Turn, 2026-06-28

The next serving check enabled the dynamic mixed CUDA Graph path:

```text
--sr24-dynamic-auto-cudagraph
--sr24-cudagraph-bucket
--no-sr24-force-cudagraph-none-for-mixed
```

Throughput artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_bs64_math128_20260628/clean_serving/report.md
```

Llama-3.1-8B, `math_reasoning`, client-side concurrency 64, EAGLE3 K=8,
max new tokens 128:

| method | full-batch tok/s | total tok/s | vs dense full | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3026.614` | `2184.300` | `1.000x` | `1.396` | `80.75%` | n/a |
| graph-on `critical_prefix` SR24 | `3118.481` | `2003.767` | `1.030x` | `1.569` | `84.75%` | `{"FULL":78,"NONE":2}` |

Paired quality artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_paired_gsm8k50_20260628/report.md
```

GSM8K-50 paired result:

| mode | score | pair n | pair reg | pair imp | dense retain |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense EAGLE3 | `0.7200` | `50` | `0` | `0` | `1.0000` |
| graph-on SR24 | `0.7400` | `50` | `1` | `2` | `0.9722` |

Read: the graph guard was a major bottleneck. This path is now quality-plausible
and is the first current `speclink_t08` candidate to beat dense on full-batch
serving. It is still only `1.03x` dense, not the requested `1.2x`. The next
work should keep graph-on and reduce residual correction work: either narrow
the residual leaf/layer scope while preserving quality, or replace the
two-pass sparse-base plus residual correction operator with a fused packed
operator.

Negative follow-ups under the same graph-on serving setup:

| variant | full-batch tok/s | vs same-root dense | accepted draft/step | CUDA Graph | read |
| --- | ---: | ---: | ---: | --- | --- |
| full leafs, threshold `0.6`, bucket32 | `3118.481` | `1.030x` | `1.569` | `{"FULL":78,"NONE":2}` | current best |
| `gate_up_proj=16-31` only | `2968.512` | `0.984x` | `1.493` | `{"FULL":70,"NONE":2}` | removing down reduces accepted length and loses speed |
| `down_proj=8-15` only | `2953.232` | `0.979x` | `1.443` | `{"FULL":72,"NONE":2}` | removing gate/up reduces accepted length and loses speed |
| full leafs, threshold `0.7` | `3088.045` | `1.022x` | `1.631` | `{"FULL":74,"NONE":2}` | higher accepted length does not overcome overhead/noise |
| full leafs, bucket16 | `3108.951` | `1.029x` | `1.579` | `{"FULL":81,"NONE":2}` | less correction capacity does not improve serving speed |

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_gateup16_31_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_down8_15_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t07_prefix4_extra1_graphon_full_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_bucket16_bs64_math128_20260628/clean_serving/report.md
```

Read: the current speed ceiling is not fixed by removing one leaf family,
raising the threshold, or shrinking the residual bucket. Those changes either
reduce accepted draft length or leave the same mixed-operator cost. The next
real path is operator work, not another scalar controller sweep.

## All-MLP CPU-Sync And Bucket Follow-Up, 2026-06-28

This follow-up used a more aggressive all-MLP residual scope to see whether
better accepted draft length could compensate for the mixed SR24 operator cost.
All rows below are Llama-3.1-8B, `math_reasoning`, GuideLLM client-side
concurrency 64, EAGLE3 K=8, max new tokens 128, graph-capable mixed SR24, and
all MLP leafs (`gate_up_proj,down_proj`) as residual candidates.

Reference and ablation artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bs64_math128_20260628_0220/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_stats_off_bs64_math128_20260628_0405/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket16_bs64_math128_20260628_0415/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket64_bs64_math128_20260628_0418/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket32_bonus1_bs64_math128_20260628_0425/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket32_pos10_bs64_math128_20260628_0430/report.md
```

| variant | total tok/s | full-batch tok/s | full vs dense | accepted draft/step | avg GPU util | CUDA Graph | bucket | bonus | pos scale |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| bucket16 | `1732.780` | `3394.140` | `1.122x` | `2.397` | `81.90%` | `{"FULL":53,"NONE":11}` | `16` | `4.0` | `0.0` |
| bucket32 default | `1717.158` | `3545.363` | `1.172x` | `2.345` | `80.70%` | `{"FULL":55,"NONE":9}` | `32` | `4.0` | `0.0` |
| bucket32 stats off | `1710.424` | `3462.887` | `1.145x` | `2.385` | `81.50%` | `{"FULL":49}` | `32` | `4.0` | `0.0` |
| bucket32 bonus1 | `1555.200` | `2746.878` | `0.908x` | `1.872` | `84.82%` | `{"FULL":81,"NONE":15}` | `32` | `1.0` | `0.0` |
| bucket32 pos10 | `1530.576` | `2874.849` | `0.950x` | `1.720` | `88.18%` | `{"FULL":85,"NONE":11}` | `32` | `4.0` | `10.0` |
| bucket64 | `1626.912` | `3284.260` | `1.086x` | `2.067` | `83.60%` | `{"FULL":84,"NONE":12}` | `64` | `4.0` | `0.0` |

The same-root dense reference for the default all-MLP run was `3025.159`
full-batch tok/s and `2187.155` total tok/s. Therefore the best all-MLP
setting is still bucket32 default: it reaches `1.172x` full-batch throughput,
but its total tok/s is worse than dense because the run does not stay in the
full-batch window for the whole request lifecycle.

Read: turning off the remaining SR24 runtime stats did not reveal a hidden CPU
synchronization bottleneck. It slightly reduced full-batch throughput in this
setup. Bucket16 and bucket64 were also worse than bucket32, and lowering bonus
priority or adding draft-position priority collapsed accepted draft length. The
CPU-sync reductions should remain enabled, and sync-heavy diagnostics should
stay diagnostic-only, but the path to `1.2x` is not another scalar sweep. The
next useful implementation step is a fused or packed mixed Linear path that
avoids paying sparse base for rows that are immediately corrected by dense
residual work.

## All-MLP Triton Override Speed/Quality Gate, 2026-06-28

I then tested the graph-capable all-MLP path with the fixed-shape Triton bucket
override. This kernel only changes the bucket row write-back implementation; it
does not change the mathematical selector. The purpose was to separate a real
operator-speed upper bound from a quality-safe candidate.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_bs64_math128_20260628_0500/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_paired_gsm8k50_20260628_0510/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix5_bs64_math128_20260628_0540/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_bs64_math128_20260628_0530/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_lowconf_prefix4_bs64_math128_20260628_0550/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_lowconf_prefix4_paired_gsm8k50_20260628_0600/report.md
```

All serving rows are Llama-3.1-8B, `math_reasoning`, GuideLLM client-side
concurrency 64, EAGLE3 K=8, max new tokens 128, all MLP residual leafs, and
graph-capable mixed SR24.

| variant | full-batch tok/s | same-root dense full tok/s | full vs dense | total tok/s | accepted draft/step | quality gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bucket32 default | `3545.363` | `3025.159` | `1.172x` | `1717.158` | `2.345` | not gated here |
| Triton override, `critical_prefix`, prefix4 | `3645.945` | `3025.805` | `1.205x` | `1993.722` | `2.379` | GSM8K-50 `0.7000` vs dense `0.7200`, pair reg `4`, pair imp `3` |
| Triton override, `critical_prefix`, prefix5 | `3560.886` | `3021.592` | `1.178x` | `1903.229` | `2.377` | not gated |
| Triton override, `critical_prefix`, prefix6 | `3579.986` | `3008.070` | `1.190x` | `1903.661` | `2.377` | not gated |
| Triton override, `low_confidence`, prefix4 | `3622.772` | `3023.944` | `1.198x` | `1925.328` | `2.393` | GSM8K-50 `0.6800` vs dense `0.7200`, pair reg `4`, pair imp `2` |

Read:

1. The Triton override is a real speed improvement for the all-MLP route. It
   adds about `100` full-batch tok/s over the bucket32 default and crosses
   `1.2x` against same-root dense in the full-batch window.
2. It is not quality safe. The paired GSM8K-50 gate drops from dense `0.7200`
   to `0.7000` with `4` dense-correct/SR24-wrong regressions. The
   `low_confidence` policy is worse at `0.6800`.
3. Expanding the mandatory prefix to 5 or 6 rows reduces the full-batch speed
   below the `1.2x` target and does not materially change accepted length.
4. Therefore all-MLP Triton override is currently a speed upper bound, not the
   default SR24 path. The quality-safe scoped graph-on path remains the
   correctness candidate, but its speed is only about `1.02-1.03x`.

I also replayed one paired regression sample (`gsm8k_cot` doc id `11`) with
SR24 debug tracing:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_replay_doc11_prefix4_debug_20260628_0520/gsm8k_cot_doc11_selective_prefix4.json
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_replay_doc11_prefix4_debug_20260628_0520/speclink_sr24_debug_trace.jsonl
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_replay_doc11_prefix4_debug_20260628_0520/speclink_confidence_trace.jsonl
```

The offline single-sample replay generated the correct answer `694`, while the
paired serving run had a regression for the same doc id. The debug trace still
shows the policy risk: many steps use masks such as `RRRRBBBB`, so later draft
positions can be base-only even when their confidence is low. Because the
single-sample replay did not reproduce the serving error, the next correctness
debug step should capture serving-shape traces for paired regressions instead
of only tuning the scalar threshold.

Current direction after this gate: use the user-requested seven-part breakdown
as the default entry point for every new candidate. The current bottleneck is
not low accepted length, not global GPU idle, and not plain CPU stats sync. It
is the mixed operator shape: sparse base work plus dense-row correction plus
some remaining scheduler/bucket overhead. The next useful implementation work
should either produce a quality-aware selector that greatly lowers residual-row
fraction without losing accepted draft length, or replace the two-pass mixed
Linear with a fused/packed CUDA/Triton operator.

## Gate-Up16 Graph-Safe Reduced-Sync Check, 2026-06-28

To isolate the user's CPU-sync concern, I reran the narrower
`gate_up_proj=16-31` high-confidence selector with reduced CPU sync, static
mask buffer, batched mask builder, and mixed CUDA Graph enabled.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_graphon_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_graph_correctness_replay_20260628/compare/report.md
```

| variant | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 reference | `2368.916` | `2763.202` | `1.402` | `88.64%` | n/a |
| gate-up16 SR24, graph off | `1904.859` | `2294.100` | `1.453` | `91.24%` | `{"NONE":128}` |
| gate-up16 SR24, graph on | `1992.979` | `2314.728` | `1.476` | `93.13%` | `{"FULL":107,"NONE":32}` |
| gate-up16 SR24, graph on, per-step mask-state sync | `1967.624` | `2282.304` | `1.455` | `92.71%` | `{"FULL":109,"NONE":30}` |

The graph-safe/reduced-sync path is functionally plausible: a single-sample
GSM8K replay matched graph-off exactly, with `65` identical output token ids
and identical cumulative logprob `-13.578305416107469`.

Read: reducing CPU synchronization and allowing CUDA Graph are necessary
hygiene, but not the main speed lever for this path. They recover only about
`4.6%` total tok/s over graph-off and almost no full-batch throughput. Adding
back a per-step mask-state synchronization drops total tok/s by only `1.3%`
relative to the no-sync graph-on run, so that sync is worth avoiding but is
not the large missing speedup. The current slow path is still dominated by
useful-work shape inside the Linear: we compute sparse base rows and then
immediately pay dense residual correction for many of those same rows.

## Route-Reuse Row-Index Check, 2026-06-28

I tested the existing `--sr24-route-reuse-base-output` path on the same
gate-up16 high-confidence selector. This path keeps the full sparse base
output, runs dense Linear only for residual rows, and scatters those rows back.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_route_reuse_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_route_reuse_eager_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_route_reuse_correctness_replay_20260628/compare_three/report.md
```

| variant | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA Graph | replay |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| graph-on, no route-reuse | `1992.979` | `2314.728` | `1.476` | `93.13%` | `{"FULL":107,"NONE":32}` | exact vs eager |
| route-reuse with dynamic graph | `2141.540` | `2503.360` | `1.440` | `91.07%` | `{"FULL":110,"NONE":32}` | token ids same, logprobs drift |
| route-reuse eager-safe | `1871.695` | `2222.398` | `1.466` | `89.88%` | `{"NONE":139}` | exact vs eager |

The graph version looked faster, but it is not safe: selected tokens matched
the eager replay, while cumulative logprob changed from
`-13.578305416107469` to `-16.11578315886254`. The eager route-reuse replay
matched the eager baseline exactly, so the route-reuse math is fine; the
unsafe part is dynamic row-index use under CUDA Graph replay.

I updated the matrix runner to prevent `route_bucket_rows`,
`route_all_residual_rows`, and `route_reuse_base_output` from passing through
the dynamic-auto graph gate. With the safe eager path, route-reuse is slower
than the graph-safe no-route path, so it should remain a diagnostic. The next
real speed path needs graph-stable fixed-shape routing or a fused packed mixed
Linear rather than the current eager index-select/scatter route.

## Bucket Budget and Quality Tradeoff, 2026-06-28

The latest CPU-sync and row-budget experiments make the current bottleneck
more concrete.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fast_candidate_cpu_sync_ablation_b12_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_sweep_b4_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_sweep_b8_b16_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_gemm_b8_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b64_priority_quality_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b64_priority_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b256_quality_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b256_bs64_math256_20260628/
```

Findings:

| variant | quality probe | full-batch tok/s | same-run dense full | speedup | accepted draft/step |
| --- | --- | ---: | ---: | ---: | ---: |
| bucket8 copy | not quality-safe; triton+copy GSM8K-50 was `0.60` vs dense `0.82` | `3980.7` | `3523.6` | `1.130x` | `2.209` |
| bucket64 priority copy | GSM8K-50 `0.76` vs dense `0.80` | `3339.8` | `3432.8` | `0.973x` | `1.818` |
| bucket256 copy | GSM8K-50 `0.78` vs dense `0.80` | `3331.1` | `3433.3` | `0.970x` | `1.885` |
| quality_gateup_only preset | previously safer, but speed-negative here | `3288.3` | `3436.3` | `0.957x` | `1.712` |

Read: `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE` is a global per-step row budget,
not a per-request budget. At bs64/K8, a quality gate such as prefix4 can imply
roughly hundreds of useful residual rows per step. The fast bucket4/8 rows get
their speed by correcting too few rows globally, which changes target logits
and hurts GSM8K. Increasing the budget toward quality-safe behavior removes
the speedup. Therefore the next optimization should not be another scalar
threshold sweep. It needs either:

- a better importance signal that can identify a much smaller set of truly
  necessary corrected rows than prefix4/global top-k, or
- a fused/packed mixed Linear or MLP operator that makes hundreds of corrected
  rows cheaper than the current sparse-base plus dense-row overwrite path.

I also fixed the prototype Triton bucket GEMM so that when
`--sr24-triton-bucket-dense-gemm` and `--sr24-bucket-dense-copy` are both
enabled, it treats every selected bucket row as active, matching the quality
semantics of the existing torch dense-copy path. This is only a semantic fix;
the quality probe still favors the torch-copy path (`0.78` vs `0.74` on the
bucket256 GSM8K-50 probe), so Triton bucket GEMM is not the current quality
path.
