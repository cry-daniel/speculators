# SR24 Current Breakdown, 2026-06-28

This note is the short current-read for why SR24/SpecLink is slow. It separates
clean serving throughput from diagnostic CUDA-event timing, because exact
routing counters and per-linear CUDA events add synchronization overhead.

Scope for the rows below: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8,
client-side batch/concurrency 64. The latest corrected current-candidate rows
use max new tokens 256. The numbers are diagnostic for direction selection, not
the final full benchmark matrix.

Primary artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_candidate_breakdown_bs64_math256_20260628_combined/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_candidate_graph_counts_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_candidate_breakdown_bs64_math256_20260628_instrumented_eager/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_slowdown_breakdown_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_graphon_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_gateup16_bs64_math_k8_20260628/component_summary_direct/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_default_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_adaptive_densefallback_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_paired_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260628/seven_part_breakdown/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_broad_clean_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_mlp_clean_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cuda_chunked_mlp_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_triton_mlp_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix6_clean_bs64_math128_256req_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_operator_probe_current_20260628/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_slowdown_breakdown_current_20260628/report.md
```

## Current Candidate Breakdown Refresh, 2026-06-28 18:50

Use this refresh first for the current bucket16/direct-cuSPARSELt
`criticalprefix4_bucket16_directcslt` candidate. It uses Llama-3.1-8B,
`math_reasoning`, client-side batch size 64, EAGLE3 K=8, max new tokens 256,
`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, residual bucket size
16, residual bucket priority, bucket dense copy, direct cuSPARSELt, dynamic
auto CUDA Graph, and no active-only bucket scatter.

Clean serving throughput:

| method | total tok/s | full-batch tok/s | full speedup vs dense | accepted draft/step | GPU util |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense EAGLE3 | `2609.686` | `3173.852` | `1.000x` | `1.696` | `94.960%` |
| `base_only_24` | `3109.799` | `3706.963` | `1.168x` | `2.114` | `92.048%` |
| `all_corrected_24` | `2776.473` | `3148.825` | `0.992x` | `1.722` | `92.565%` |
| `speclink_t08` | `2861.536` | `3486.572` | `1.099x` | `2.177` | `94.043%` |

Low-overhead graph-count refresh:

| method | total tok/s | full-batch tok/s | CUDA Graph runtime modes |
| --- | ---: | ---: | --- |
| dense EAGLE3 | `2764.868` | `3175.860` | `{"FULL":57,"NONE":166,"PIECEWISE":1}` |
| `base_only_24` | `3137.144` | `3660.767` | `{"FULL":143,"NONE":49}` |
| `all_corrected_24` | `2778.621` | `3138.691` | `{"FULL":57,"NONE":166,"PIECEWISE":1}` |
| `speclink_t08` | `2872.613` | `3441.300` | `{"FULL":150,"NONE":42}` |

Instrumented serving is forced eager after applying the preset via
`--sr24-force-eager-after-preset`, because CUDA-event timing is not
capture-safe. Its tok/s and GPU util are diagnostic only. The localized timing
for `speclink_t08` is:

| part | value | read |
| --- | ---: | --- |
| scheduler / mask build | exact-routing diagnostic `13.714ms/step`; request loop `13.471ms` | sync-heavy attribution row; not clean serving CPU cost |
| base sparse Linear | aggregate `0.992ms/call`; gate/up16-31 `1.101ms/call`; down8-15 `0.774ms/call` | largest measured per-Linear GPU cost |
| residual correction | aggregate dense rows `0.163ms/call`; gate/up16-31 `0.180ms/call`; down8-15 `0.131ms/call`; bucket rows/call `16` | smaller than sparse base, but additive |
| gather/scatter | `0.017ms/call` | not the first bottleneck |
| routing | draft residual/base `13599/11713`, non-draft residual/base `3164/3788`, draft residual fraction `0.537`, non-draft residual fraction `0.455`, bucket fill `0.982` | many rows still require residual protection |
| CUDA Graph | clean graph-count `speclink_t08` `{"FULL":150,"NONE":42}` | graph coverage is not perfect but not the primary issue |
| GPU util | clean `speclink_t08` avg `94.043%`, peak `100%` | GPU is busy; focus on useful-work efficiency |

Current read: `base_only_24` shows real sparse headroom (`1.168x`
full-batch), but `speclink_t08` recovers only `1.099x` because the mixed path
still pays sparse base Linear on hundreds of rows and dense correction for a
large protected subset. The scheduler exact-routing time is a diagnostic
synchronization artifact, while gather/scatter is too small to explain the gap.
The next speed work should target a fused/packed mixed operator or a
quality-safe way to reduce residual rows; another threshold-only or ordinary
CPU-stat cleanup is lower priority unless this breakdown changes.

## Current Pivot

The next pass should be a slowdown breakdown first, not another threshold
sweep. For every candidate, keep these rows separate:

| row type | purpose | do not use for |
| --- | --- | --- |
| clean serving | real tok/s, GPU util, CUDA Graph mode counts, low-overhead scheduler counters | per-kernel attribution |
| instrumented serving | scheduler/mask, sparse base, residual correction, gather/scatter, routing counters | final tok/s claims |
| component microbench | isolated dense/sparse/mixed Linear shapes and graph replay cost | end-to-end serving claims |

The current answer to "why is it slow" is:

| part | current read |
| --- | --- |
| scheduler / mask build | fixed-shape/batched-builder paths are now sub-ms in clean serving. The old multi-ms or 40ms rows came from sync-heavy diagnostics or old Python request-routing paths. |
| base sparse linear | this is the largest localized GPU-side cost in the mixed path; the gate/up sparse base call is expensive enough that it only helps when residual correction stays low. |
| residual correction | dense-row correction is additive work on top of sparse base. When many rows need correction, the two-pass path becomes slower than dense. |
| gather/scatter | small in the graph-safe gate/up path, but visible in broader all-MLP routing. It is not the first bottleneck unless the operator is otherwise fixed. |
| routing statistics | too many draft rows still require residual protection. Bucket fill is usually high, so the issue is not empty buckets; it is residual-row fraction. |
| CUDA Graph | healthy for fixed-shape bucketed paths, but dynamic routed-row variants can lose graph coverage. Treat graph `NONE` fraction as a hard guardrail. |
| GPU util | clean runs are usually busy. The slowdown is inefficient useful work, not a long idle-GPU gap. |

This means the speed target needs either a lower residual-row fraction with a
paired accuracy gate, or a fused/packed mixed Linear that avoids paying sparse
base plus dense correction as two separate useful-work paths.

## Corrected Fast-Candidate Seven-Part Refresh, 2026-06-28

Use this as the current first reference. It reruns the slowdown breakdown for
the actual graph-safe bucketed candidate, without the priority/direct-position
routing switches that made the previous refresh a different and slower
diagnostic condition.

Primary artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/component_microbench/summary.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, bs64, EAGLE3 K=8, max new tokens
256, `critical_prefix@0.6,prefix4,extra1`,
`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, residual bucket size
12, direct cuSPARSELt base Linear, bucket dense copy, low-sync stats, dynamic
auto CUDA Graph, and graph bucket capture. Do not add
`SPECLINK_SR24_RESIDUAL_BUCKET_PRIORITY` or
`SPECLINK_SR24_DIRECT_POSITION_BUCKET` when reproducing this row.

Clean serving result:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2599.815` | `3524.307` | `1.736` | `89.385%` | n/a |
| `base_only_24` | `2955.659` | `4307.108` | `2.205` | `89.364%` | `{"FULL":94,"NONE":2}` |
| `speclink_t08` | `2625.737` | `3902.787` | `2.103` | `91.154%` | `{"FULL":126,"NONE":2}` |

Read: `base_only_24` still proves sparse-headroom exists (`1.222x` full-batch
speedup), but `speclink_t08` is only `1.107x` full-batch and `1.010x` total
against the same dense run. The remaining gap is not GPU idleness or CUDA Graph
coverage: GPU util is high and graph coverage is `126/128` FULL. It is the
mixed useful-work shape.

Seven-part diagnosis:

| part | current evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean wall `0.289ms/step`; row bucket `0.064ms/step`; bucket build `0.062ms/step`; mixed row indices `0.001ms/step` | clean scheduler/bucket construction is sub-ms |
| base sparse Linear | diagnostic sparse base `1.041ms/call`; `gate_up_proj=16-31` `1.071ms/call`; `down_proj=8-15` `0.980ms/call` | largest localized GPU-side cost |
| residual correction | diagnostic dense correction `0.161ms/call`; gate/up `0.180ms/call`; down `0.123ms/call`; bucket rows/call `12` | secondary but additive on top of sparse base |
| gather/scatter | diagnostic gather/scatter `0.014ms/call` | too small to explain the main gap in this row |
| routing statistics | diagnostic draft residual/base `2928/2464`, non-draft residual/base `674/891`, draft residual fraction `0.543`, bucket fill `0.989`, actual/requested bucket fraction `0.174` | many draft rows still require residual protection |
| CUDA Graph | clean `{"FULL":126,"NONE":2}` | graph coverage is healthy |
| GPU util | clean `91.154%` avg, `99%` peak | GPU is busy; optimize useful-work efficiency |

The component microbench confirms the operator-level issue. For gate/up
`512/28672/4096`, sparse base alone is about `0.65x` dense, but current mixed
is already `1.03x` dense at residual fraction `0.125` and `1.53x` dense at
`0.5`. For down `512/4096/14336`, current mixed beats dense only at low
residual fractions and loses by `0.5`. The next useful work is therefore either
to lower residual rows without paired accuracy loss, or to replace the current
two-pass sparse-base plus dense-row correction with a graph-safe fused/packed
operator.

### CPU-Sync Follow-Up Status

The slowdown entrypoint now forwards the compressed-residual diagnostic flags
(`--sr24-residual-out-chunk`, cache/prewarm, auto fastpath, Triton residual
kernel, block sizes, and extract chunk rows), so CPU-sync and
compressed-residual ablations can be run through the same seven-part protocol.

A same-condition CPU-sync ablation command for the corrected bucket12 fast
candidate was run under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fast_candidate_cpu_sync_ablation_b12_bs64_math256_20260628/
```

Key rows:

| variant | SR24 total tok/s | SR24 full-batch tok/s | accepted draft/step | GPU util | scheduler mask wall | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `low_sync_stats_on` | `2705.051` | `3966.966` | `2.187` | `90.917%` | `0.338ms/step` | `{"FULL":94,"NONE":2}` | current clean reference for this ablation |
| `low_sync_stats_off` | `2615.595` | `3861.572` | `2.094` | `91.154%` | n/a | server `{"FULL":49}` | disabling stats did not produce a clear speed win |
| `sync_mask_state` | `2651.364` | `3941.026` | `2.186` | `88.833%` | `43.812ms/step` | `{"FULL":94,"NONE":2}` | wall timer mostly observes synchronization wait; throughput stays close |
| `sync_heavy` | `1442.689` | `1960.373` | `1.918` | `59.864%` | `10.984ms/step` | `{"NONE":128}` | invalid speed path: loses graph coverage and GPU util |
| `low_sync_gpu_counts` | `2656.091` | `3878.992` | `2.176` | `88.083%` | `44.046ms/step` | `{"FULL":94,"NONE":2}` | useful for routing counts, not a clean speed row |

Read: ordinary CPU-sync cleanup is a guardrail, not the remaining first-order
speed knob. The clean low-sync path is already sub-ms per step and reaches
`1.155x` full-batch speedup versus its same-root dense row (`3966.966` vs
`3435.870`). The bad case is the sync-heavy path, which loses CUDA Graph
coverage and drops to `0.571x`. Further speed work should focus on reducing
residual rows or replacing the two-pass sparse-base plus correction operator.

### Score-Free Low-Sync Ablation

The latest follow-up tested the user's CPU-sync concern through a stricter
score-free path. The `fixedprefix4_bucket16_directcslt` preset uses the same
target scope as `criticalprefix4_bucket16_directcslt`
(`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, bucket16,
direct cuSPARSELt, bucket dense copy), but switches the residual policy to
`fixed_prefix` with prefix length 4. That disables SR24 draft-score collection,
so the EAGLE proposer does not run the extra selected-token
`logsumexp(full_vocab)` path.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix4_bucket16_directcslt_defaultcompile_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix4_defaultcompile_throughput_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_bucket16_directcslt_defaultcompile_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_defaultcompile_throughput_bs64_math256_20260628/report.md
```

Quality gate:

| preset | GSM8K-30 exact | Pair reg | Pair imp | avg output tokens | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `0.7333` | `0` | `0` | `82.9` | reference |
| `fixedprefix4_bucket16_directcslt` | `0.7333` | `0` | `0` | `82.9` | paired-clean on this gate |

Throughput, default vLLM compile, bs64, math_reasoning, 128 requests,
max new tokens 256:

| preset | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | accepted draft/step | SR24 score path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `criticalprefix4_bucket16_directcslt` | `2595.090` | `2714.458` | `1.046x` | `3135.954` | `3112.682` | `0.993x` | `1.718` | on |
| `fixedprefix4_bucket16_directcslt` | `2778.779` | `2562.704` | `0.922x` | `3137.580` | `3102.821` | `0.989x` | `1.660` | off |

Read: disabling DLM score collection and the associated full-vocab logsumexp is
not enough to recover speed. The score-free row is quality-clean on the
GSM8K-30 gate, but it lowers accepted draft length and does not improve
full-batch throughput. This makes generic CPU-sync/score cleanup an ablation,
not the current mainline. The next high-leverage work remains either:

- fix the SR24 `FULL_DECODE_ONLY` / dynamic mixed CUDA Graph quality drift so
  the faster compile path can be used safely; or
- implement a graph-safe fused/packed mixed operator that avoids the current
  two useful-work paths, sparse base for all rows plus dense correction for
  protected rows.

### Compressed-Dense GPU-Side Check

Current code inspection confirms that `compressed_dense` is GPU-side when
`SPECLINK_SR24_RESIDUAL_DEVICE=cuda`: model attach stores packed mask bytes and
compressed residual values on the parameter device, and
`_compressed_residual_weight()` expands the dense residual tensor on the
requested CUDA device. Cache+prewarm materializes that tensor at attach time
and stores it on the module. The remaining `all_corrected_24` slowdown is
therefore duplicated GPU work, `sparse base + residual GEMM/materialization`,
not a CPU residual-computation path.

The cached/prewarmed compressed-dense diagnostic also shows that materialization
is no longer the bottleneck:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_dense_prewarm_probe_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_allcorrected_compressed_dense_prewarm_probe_20260628_work/all_corrected_24/bs32/rep1/speclink_sr24_breakdown.json
```

That row keeps `gate_up_proj=16-31` compressed residual tensors on CUDA with
cache+prewarm enabled. It reached only `1011.227` full-batch tok/s versus dense
`1532.123`. The localized Linear timing was: sparse base `0.908ms/call`,
compressed residual GEMM `1.028ms/call`, residual add `0.271ms/call`, and
compressed materialization `0.0004ms/call`, with `384` cached-weight hits and
`16` misses. So the all-corrected path needs a real fused/packed GPU operator
or should be treated as a diagnostic, not as a route to speed through
cache/prewarm alone.

### Predicted-Full-Accept Bonus Row Ablation

I extended the batched mask builder so
`SPECLINK_SR24_SELECTIVE_NON_DRAFT_POLICY=predicted_full_accept` no longer
falls back to the Python request-routing loop. Its Triton kernels now correct
the speculative bonus row only when every draft score is present and above the
threshold. The new path passed the SR24 correctness check, including slow-path,
uniform-direct, indexed, and GPU-count batched-builder equivalence.

Clean serving on the same corrected bucket12 shape:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_predicted_full_accept_batched_b12_bs64_math256_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bonus_currentcode_b12_bs64_math256_20260628/clean_serving/report.md
```

| non-draft policy | total tok/s | full-batch tok/s | accepted draft/step | non-draft residual fraction | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| `bonus` | `2683.694` | `3917.754` | `2.171` | `0.572894` | `{"FULL":94,"NONE":2}` |
| `predicted_full_accept` | `2638.930` | `3903.531` | `2.119` | `0.000000` | `{"FULL":94,"NONE":2}` |

This is negative despite removing bonus-row residual correction from the
low-sync stats. Accepted draft length drops enough that total throughput is
lower and full-batch throughput is essentially unchanged. Do not treat
non-draft/bonus correction removal as the next speed path unless a later
quality-aware policy changes the acceptance behavior.

## Superseded Seven-Part Refresh, 2026-06-28

The refresh below is kept as history, but it should not be treated as the
current fastest-candidate read. It enabled priority/direct-position routing and
therefore measured a different condition from the bucketed path above.

## User-Requested Seven-Part Refresh, 2026-06-28

Fresh run for the requested slowdown breakdown, using the current graph-capable
bucketed `speclink_t08` candidate rather than the older gate-up-only default:
Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency 64, max new
tokens 256, `critical_prefix@0.6,prefix4,extra1`,
`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, residual bucket size
12, direct cuSPARSELt base Linear, bucket dense copy, and CUDA Graph bucket.

Primary artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_bs64_math_k8_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_bs64_math_k8_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_bs64_math_k8_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_bs64_math_k8_20260628/component_microbench/summary.md
```

Clean serving result:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2310.051` | `3422.973` | `1.697` | `89.429%` | n/a |
| `speclink_t08` | `2225.393` | `3340.622` | `1.698` | `92.200%` | `{"FULL":114,"NONE":77,"PIECEWISE":1}` |

This current bucketed `speclink_t08` row is slightly slower than dense in this
single run (`0.976x` full-batch and total). It is not accepted-length-limited:
accepted draft tokens per step are essentially identical to dense EAGLE3. GPU
utilization is also high. The clean serving signal is therefore not idle GPU or
acceptance collapse; it is extra work in the mixed verify path plus partial
CUDA Graph misses.

Seven-part diagnosis:

| part | current evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean wall `0.899ms/step`; row bucket `0.041ms/step`; bucket build `0.035ms/step`; mixed row indices `0.004ms/step` | clean scheduler/bucket construction is sub-ms, not the main bottleneck |
| base sparse Linear | diagnostic sparse base `1.148ms/call`; `gate_up_proj=16-31` `1.189ms/call`; rows/call `141.380` | largest localized GPU-side cost |
| residual correction | diagnostic dense correction `0.166ms/call`; `gate_up_proj=16-31` `0.184ms/call`; bucket rows/call `12` | secondary but additive on top of sparse base |
| gather/scatter | diagnostic gather/scatter `0.016ms/call` | too small to explain the main gap in this row |
| routing statistics | diagnostic draft residual/base `3673/1847`, non-draft residual/base `690/904`, draft residual fraction `0.665`, bucket fill `0.990`, actual/requested bucket fraction `0.149` | many draft rows still require residual protection; the bucket is full, not empty |
| CUDA Graph | clean `{"FULL":114,"NONE":77,"PIECEWISE":1}` | Graph misses may contribute; keep this as a hard guardrail |
| GPU util | clean `92.200%` avg, `100%` peak | GPU is busy; optimize useful-work efficiency rather than generic CPU waits |

The instrumented serving row is intentionally eager/sync-heavy and should not
be used for throughput. Its role is only to localize the cost. It measured
`4.687ms/step` wall in the exact diagnostic scheduler path and `39.429%` GPU
util, which is expected because CUDA events and exact routing counters perturb
serving.

The component microbench confirms the operator-level issue. For gate/up
`512/28672/4096`, sparse base alone is fast (`0.65-0.66x` dense), but the
current mixed operator is already about dense at residual fraction `0.0625`
and `1.03x` dense at residual fraction `0.125`; it becomes much slower as the
residual fraction grows. For down `512/4096/14336`, low residual fractions can
beat dense, but it also loses once residual fraction reaches `0.5`.

Separate base-only reference: the exact current mixed parameters are not a
valid `base_only_24` launch with default compile in this run; vLLM failed during
profile-run compilation with a PyTorch lazy-allocation error. Use the valid
graph-safe base-only run below as the base-only reference for the same
bs64/math/K8/max256 shape:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_working_graph_bs64_math256_20260628/report.md
```

There, `base_only_24` reached `2785.660` total tok/s and `3965.653`
full-batch tok/s versus dense `2317.632` total and `3430.409` full-batch, with
accepted draft/step `2.027`, GPU util `90.750%`, and CUDA Graph
`{"FULL":126,"NONE":2}`. Base-only still proves sparse headroom exists; the
mixed `speclink_t08` problem is paying sparse base plus correction and losing
some graph coverage.

## Bucket-Size Follow-Up, 2026-06-28

After the current best bucketed path, I tested whether the remaining gap to
`1.2x` could be closed by only changing the fixed residual-correction bucket
size. The tested serving shape stayed the same: Llama-3.1-8B,
`math_reasoning`, EAGLE3 K=8, client concurrency 64, max new tokens 256,
`critical_prefix@0.6,prefix4,extra1`, target leafs
`gate_up_proj,down_proj`, residual layers
`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`,
`SPECLINK_SR24_BUCKET_DENSE_COPY=1`, `SPECLINK_SR24_DIRECT_CSLT_LINEAR=1`,
CUDA Graph bucket enabled, and no mixed-Graph force-off.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b8_repeat_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b10_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b12_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b16_graphon_bs64_math256_20260628/report.md
```

| bucket | total speedup | full-batch speedup | dense total tok/s | SR24 total tok/s | dense full tok/s | SR24 full tok/s | accepted draft/step | CUDA Graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 repeat | `1.148x` | `1.156x` | `2323.570` | `2668.197` | `3436.248` | `3972.309` | `2.219` | `{"FULL":94,"NONE":2}` |
| 10 | `1.163x` | `1.154x` | `2323.843` | `2702.617` | `3438.138` | `3968.271` | `2.199` | `{"FULL":94,"NONE":2}` |
| 12 | `1.173x` | `1.156x` | `2318.338` | `2720.414` | `3429.504` | `3964.080` | `2.191` | `{"FULL":94,"NONE":2}` |
| 16 | `1.173x` | `1.146x` | `2319.712` | `2720.019` | `3429.905` | `3930.796` | `2.198` | `{"FULL":94,"NONE":2}` |

The earlier bucket-8 row that looked close to `1.2x` had an unusually weak
dense baseline (`2271.872` total tok/s and `2914.457` full-batch tok/s). The
repeat returned to the same range as the other bucket sizes. Therefore fixed
bucket-size tuning is not enough: the best stable bucketed path remains about
`1.17x` total and about `1.15x` full-batch. The next speed path should not be
another bucket-size sweep; it should reduce residual rows without paired
accuracy loss or replace the two-pass sparse-base plus dense-row correction
with a fused/packed operator.

I also added an off-by-default bucket row sorting ablation:
`SPECLINK_SR24_SORT_BUCKET_ROWS=1` / `--sr24-sort-bucket-rows`. It sorts capped
bucket rows by row id before dense gather/index-copy correction, testing
whether more coalesced row access can recover the last few percent. On the same
bucket12 shape it was negative:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b12_sortrows_graphon_bs64_math256_20260628/report.md
```

| bucket12 variant | total speedup | full-batch speedup | SR24 total tok/s | SR24 full tok/s | accepted draft/step | scheduler/mask wall | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| unsorted | `1.173x` | `1.156x` | `2720.414` | `3964.080` | `2.191` | `0.336ms/step` | `{"FULL":94,"NONE":2}` |
| sorted rows | `1.157x` | `1.154x` | `2677.046` | `3952.422` | `2.194` | `0.913ms/step` | `{"FULL":94,"NONE":2}` |

The sort kernel costs more than any gather/index-copy locality gain, so bucket
row sorting should stay off unless a future fused bucket kernel changes the
tradeoff.

## Dense No-Op Sanity, 2026-06-28

Before using paired accuracy as the quality gate, compile settings must be
aligned. A dense no-op SR24 run without `--sr24-default-vllm-compile` used a
different vLLM `--compilation-config` from dense EAGLE3 and produced aggregate
score cancellation with `Pair reg=2` and `Pair imp=2`.

The compile-aligned sanity below uses SR24 dense no-op plus
`--sr24-default-vllm-compile`, so dense baseline and SR24 no-op launch with the
same default vLLM compile configuration:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_static_all_residual_noop_defaultcompile_gsm8k20_20260628_followup/report.md
```

Result:

| mode | exact match | pair reg | pair imp | avg output tokens | accepted/drafted |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `0.7000` | `0` | `0` | `87.1` | `1054/5760` |
| SR24 dense no-op | `0.7000` | `0` | `0` | `87.1` | `1054/5760` |

The no-op server log confirms `Applied SpecLink SR24 dense no-op`, and both
server commands omit explicit `--compilation-config`. Therefore, future SR24
accuracy debugging should first pass this dense no-op control or use aligned
compile settings before interpreting paired regressions as SR24 logic errors.

## Compressed Residual Kernel Sweep, 2026-06-28

I added and ran an isolated GPU microbenchmark for the
`compressed_dense`/Triton residual kernel:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_compressed_residual_kernel.py
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_gateup_sweep_20260628_followup/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_down_sweep_20260628_followup/summary.md
```

It compares the packed residual Triton matmul against a materialized dense
residual weight plus torch GEMM. Lower `triton/dense` is better.

| shape | old default block | best block | dense residual graph ms | best Triton graph ms | best Triton/dense | old default Triton/dense |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gate/up `512 x 28672 x 4096` | `16/16/32` | `32/128/16` | `0.5365` | `4.0776` | `7.601x` | `17.345x` |
| down `512 x 4096 x 14336` | `16/16/32` | `32/128/16` | `0.2898` | `2.6841` | `9.262x` | `16.321x` |

I changed the diagnostic defaults for `SPECLINK_SR24_COMPRESSED_RESIDUAL_BLOCK_*`
and the matrix/lm-eval runners to `block_m=32`, `block_n=128`,
`block_g=16`. This makes the experimental packed Triton path less bad, but it
does not change the optimization direction: packed `compressed_dense` is still
far slower than materialized residual GEMM on both representative MLP shapes.
Therefore `compressed_dense` is confirmed GPU-resident but not a viable
speed path with the current kernel. Treat it as a diagnostic memory-saving
path; the all-corrected speed path remains either dense fastpath/no-op control
or a different fused/packed operator, not this packed residual matmul.

## Compile-Aligned `speclink_t08` Accuracy Sanity, 2026-06-28

After the dense no-op sanity, I reran the actual gate-up16 `speclink_t08`
candidate with the same default vLLM compile configuration as dense EAGLE3:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_critical_prefix6_defaultcompile_gsm8k20_20260628_followup/report.md
```

Run shape: Llama-3.1-8B, GSM8K CoT, 20 task-manifest samples, EAGLE3 K=8,
`gate_up_proj=16-31`, `critical_prefix`, threshold `0.3`, forced residual
prefix `6`, one extra row after low confidence, bonus-row correction, bucket32,
`dense_rows@cuda`, and `--sr24-default-vllm-compile`.

| mode | exact match | pair reg | pair imp | avg output tokens | accepted/drafted |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `0.7000` | `0` | `0` | `87.1` | `1054/5760` |
| `speclink_t08` | `0.7000` | `0` | `0` | `87.1` | `1054/5760` |

Both server commands omit explicit `--compilation-config`, and both logs show
the same default vLLM compile configuration. This does not prove full-matrix
accuracy, but it proves the earlier `Pair reg=2 / Pair imp=2` small-sample
result was not a reliable SR24-quality signal when SR24 used a different
compile configuration. Future accuracy gates should either use
`--sr24-default-vllm-compile` or otherwise align dense and SR24 compile modes
before treating paired divergences as true residual-mask errors.

## Fresh Breakdown Pivot, 2026-06-28

The current direction should be a slowdown breakdown first, not another
controller sweep. The latest clean serving row and diagnostic rows point to the
same conclusion: SR24 currently gets longer accepted draft spans, but the mixed
verify path spends too much useful work in sparse base plus correction and loses
serving efficiency.

Fresh artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix6_clean_bs64_math128_256req_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_operator_probe_current_20260628/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_slowdown_breakdown_current_20260628/report.md
```

Clean serving row, Llama-3.1-8B, math_reasoning, K=8, client concurrency 64,
256 total requests, max new tokens 128:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2686.241` | `2449.958` | `1.402` | `93.3%` | n/a |
| prefix6 `speclink_t08` | `2415.512` | `2191.759` | `2.400` | `84.2%` | `{"FULL":114,"NONE":78}` |

This row is important because the accepted draft length improves by about
`1.71x`, but both full-batch and total tok/s are about `0.90x` of dense. The
loss is therefore not explained by acceptance collapse. It is either the mixed
operator cost, graph/shape churn, or both.

The actual serving component diagnostic from
`sr24_user_breakdown_math_bs64_k8_20260628` breaks the mixed path down:

| part | measured value | read |
| --- | ---: | --- |
| scheduler / mask build | clean runtime in the 256-request row: `0.719ms/step`; diagnostic exact row: `36.011ms/step` | Low-sync clean scheduler cost is sub-ms. Exact routing diagnostics are sync-heavy and should not be used as serving cost. |
| base sparse linear | aggregate `1.935ms/call`; `gate_up_proj_layers_16_31=2.605ms/call`; rows/call `334.885` | Sparse base is the largest localized GPU-side operator cost. |
| residual correction | aggregate dense correction `0.121ms/call`; `gate_up_proj_layers_16_31=0.154ms/call`; dense rows/call `31.192` | Residual GEMM itself is smaller than sparse base, but still adds a second path. |
| gather/scatter | `0.288ms/event` | Assembly can exceed the residual GEMM; fusion is more useful than tuning scatter alone. |
| routing statistics | draft residual/base `15776/4248`; non-draft residual/base `2503/3788`; bucket fill `0.876` | Too many draft rows still go through residual correction in the quality-safe diagnostic row. |
| CUDA Graph | clean 256-request SR24 row: `FULL=114`, `NONE=78`; short diagnostic row: `FULL=74`, `NONE=2` | Graph coverage is workload/shape sensitive. It is a guardrail; do not judge from one short diagnostic row only. |
| GPU util | clean SR24 `84.2%` avg, `100%` peak; dense `93.3%` avg, `99%` peak | The GPU is not idle, but SR24 does less useful work per busy GPU second. |

The synthetic operator microbenchmark
`sr24_component_operator_probe_current_20260628` isolates the MLP Linear shapes
without vLLM scheduling overhead:

| Linear shape | residual fraction | dense graph | base sparse graph | mixed current graph | bucket dense copy graph | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate_up `512 x 28672 x 4096` | `0.125` | `0.538ms` | `0.353ms` | `0.551ms` | `0.552ms` | At low residual fraction, current mixed gate_up is only about dense, not clearly faster. |
| gate_up `512 x 28672 x 4096` | `0.500` | `0.539ms` | `0.353ms` | `0.823ms` | `0.551ms` | Once correction rows grow, two-pass mixed is much slower than dense. |
| down `512 x 4096 x 14336` | `0.125` | `0.291ms` | `0.166ms` | `0.265ms` | `0.260ms` | Down projection can benefit at low residual fraction. |
| down `512 x 4096 x 14336` | `0.500` | `0.291ms` | `0.166ms` | `0.363ms` | `0.260ms` | Down also loses when correction grows, but bucket dense copy remains promising. |

Immediate decision:

1. Keep a clean serving row and a diagnostic component row for every candidate.
   Never use diagnostic tok/s as final throughput because CUDA events and exact
   counters add synchronization.
2. Optimize the mixed operator before doing more threshold sweeps. In the
   current evidence, the main costs are sparse base `gate_up_proj` plus
   correction assembly, not scheduler Python.
3. Treat CUDA Graph `NONE` fraction as a required guardrail. The same SR24
   policy can look healthy in a short diagnostic row but lose graph coverage in
   a longer serving row.
4. The most promising concrete operator direction is a bucket dense-copy or
   packed/fused route that avoids dense-minus-base gather/scatter and avoids
   sparse-base work on rows that will be overwritten.

## Current Seven-Part Diagnosis

The immediate optimization direction should be driven by this breakdown, not by
another threshold sweep. Use clean serving rows for throughput, CUDA Graph, GPU
utilization, and low-overhead scheduler counters. Use CUDA-event/component
profile rows only to locate relative operator cost; their tok/s is diagnostic
because the timing itself synchronizes the GPU.

| part | what the current runs show | decision |
| --- | --- | --- |
| scheduler / mask build | The fixed-shape/batched-builder paths are now sub-ms in clean serving (`0.38-0.98ms/step` in the recent quality-safe and early-dense rows). The old all-MLP early-dense Python routing loop was about `40ms/step`, but the batched-builder fix removed that path. | Do not make generic CPU-sync cleanup the main path unless a fresh clean run shows scheduler wall time above about `1ms/step` again. Reject dynamic row-routing variants that reintroduce Python loops. |
| base sparse linear | Sparse base is the largest localized GPU-side operator cost in mixed paths. Recent profiles put aggregate sparse base around `0.94-1.93ms/call`, and `gate_up_proj=16-31` can reach `1.02-2.61ms/call` depending on scope. | Sparse base only helps if it replaces enough dense work. A selective path that still corrects many rows cannot rely on base-only speedup. |
| residual correction | Exact all-corrected keeps accepted length near dense but is slower. The best runnable MLP exact path is still below dense (`2229` full-batch tok/s vs `3018` same-root dense). Chunked/compressed residual is GPU-resident but much slower; current compressed Triton residual is worse. | The real bottleneck is the two-pass mixed Linear shape: sparse base plus dense/residual correction. The useful implementation target is a fused or packed operator, not moving residual data from CPU to GPU. |
| gather/scatter | In quality-safe bucketed paths gather/scatter can be small (`0.012-0.015ms/call`), but in the aggressive all-MLP/profile path it becomes visible (`0.287ms/call`, plus bucket Triton override around `0.57ms/call`). | It is not the first bottleneck in the safe path, but it will matter after residual/base costs are reduced. Favor fusion over optimizing scatter alone. |
| routing statistics | Quality protection still sends many draft rows through residual correction. Recent safe rows have draft residual fractions around `0.55`; prefix6/all-MLP diagnostics can be `0.79-0.84`. Bucket fill is usually high, so the issue is not empty buckets but too many protected rows. | The controller must either reduce residual rows with a paired accuracy gate or use a bigger base-only scope plus stronger quality protection. Threshold-only tuning on the small safe scope is unlikely to reach `1.2x`. |
| CUDA Graph | Good fixed-shape paths keep high graph coverage (`FULL=62/64`, `74/76`, etc.). Dynamic route-reuse/eager variants can drop to all `NONE`. | CUDA Graph coverage is a hard guardrail. Any candidate with mostly `NONE` must be treated as diagnostic unless it wins despite that. |
| GPU util | Clean rows are generally busy: average `80-93%`, peak `99-100%`. Slow exact paths are also busy. | This is not primarily idle GPU. The problem is inefficient useful work: extra sparse/residual GEMMs, correction rows, and small assembly kernels. |

Current high-level read: `base_only_24` shows there is sparse headroom, but
`speclink_t08` loses much of it because the quality-safe mixed path still pays
base sparse work for many rows plus residual correction for too many rows. The
next useful pass should either make the mixed Linear path fused/packed, or
prove a residual-row controller that preserves paired accuracy while sharply
reducing correction rows.

## Base-Only Graph Evidence, 2026-06-28

I reran `base_only_24` without default vLLM compile but with the SR24 graph-safe
base-only path enabled, because the earlier clean row failed when PyTorch/vLLM
compile touched sparse semi-structured tensors.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_working_graph_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_why_slow_current_breakdown_20260628/seven_part_report_with_baseonly_graph/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency 64,
64 total requests, max new tokens 256, `gate_up_proj=16-31`.

| method | total tok/s | full-batch tok/s | accepted draft/step | selected draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2317.632` | `3430.409` | `1.697` | `8.000` | `88.36%` | n/a |
| `base_only_24` | `2785.660` | `3965.653` | `2.027` | `8.000` | `90.75%` | `{"FULL": 126, "NONE": 2}` |

This changes the slowdown read:

1. `base_only_24` is not the current slow path when it can keep CUDA Graph
   coverage. It is `1.20x` total tok/s and `1.16x` full-batch tok/s over the
   same-root dense EAGLE3 row.
2. Acceptance does not collapse. The accepted draft length is higher than the
   same-root dense row (`2.027` vs `1.697` draft tokens/step).
3. GPU utilization stays high (`90.75%` avg, `100%` peak), so this is not an
   idle-GPU issue.
4. The slow part is therefore the mixed path used by `speclink_t08` and the
   exact correction path used by `all_corrected_24`: sparse base plus residual
   correction and its shape/Graph cost. The optimization target should be the
   mixed verify operator and residual-row controller, not generic scheduler
   CPU cleanup.

## Same-Root Dense/Base/Exact Correction Probe, 2026-06-28

I also reran a same-root clean serving comparison with real all-corrected
correction enabled. This disables the dense-equivalent all-corrected fastpath,
uses `torch_sparse` residual correction, and keeps the same `gate_up_proj=16-31`
scope as the base-only probe.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_clean_dense_base_allcorrected_real_gateup16_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_why_slow_current_breakdown_20260628/seven_part_report_with_baseonly_and_allcorrected_graph/report.md
```

| method | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA Graph | residual draft fraction |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| dense EAGLE3 | `2314.017` | `3421.941` | `1.697` | `88.00%` | n/a | n/a |
| `base_only_24` | `2789.625` | `3960.683` | `2.027` | `90.75%` | `{"FULL":126,"NONE":2}` | `0.000` |
| real `all_corrected_24` | `2244.585` | `3000.626` | `1.760` | `89.93%` | `{"FULL":126,"NONE":2}` | `1.000` |

This same-root probe removes two possible wrong explanations:

1. It is not primarily a CUDA Graph problem for exact correction. Both
   base-only and all-corrected have the same healthy graph count
   (`126 FULL / 2 NONE`).
2. It is not primarily low GPU utilization. All rows are busy
   (`88-91%` average GPU utilization, `99-100%` peak).

The all-corrected path is slow because every verifier row pays sparse base plus
exact residual correction. The diagnostic row quantifies the local cost as
about `0.937ms/call` sparse base plus `0.444ms/call` residual correction for
`gate_up_proj=16-31`. Since residual fraction is `1.0`, the sparse path cannot
win. For `speclink_t08`, the next useful target is therefore not broad CPU sync
cleanup; it is either:

1. reduce residual draft rows sharply while preserving paired accuracy, or
2. replace the two-pass mixed Linear with a fused/packed operator that does not
   compute sparse base and dense correction as separate expensive paths.

## Paired Accuracy Finding, 2026-06-28

The earlier `speclink_t08` paired regression read was contaminated by dense
EAGLE3 baseline instability across independent runs. With the same
Llama-3.1-8B, GSM8K-50, K=8, prefix6 configuration, the SR24 outputs are
token/text-identical across the graph-on and graph-none probes, while the dense
EAGLE3 baseline itself flips several GSM8K answers across runs.

| comparison | output diff | correctness diff | read |
| --- | ---: | ---: | --- |
| SR24 graph-on vs SR24 graph-none | `0/50` | `0/50` | SR24 output is stable across these graph settings |
| dense run A vs dense run B | `7/50` | `4/50` | dense EAGLE3 baseline is not stable enough for naive same-root paired claims |
| dense run A vs SR24 graph-on | `0/50` | `0/50` | SR24 graph-on matches one dense baseline realization exactly |
| dense run B vs SR24 graph-on | `7/50` | `4/50` | the apparent `Pair reg=2, Pair imp=2` comes from the dense baseline variant |

Artifacts checked:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_paired_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_graphnone_paired_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_smallfull_paired_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseline_stability_analysis_20260628/dense_vs_dense/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseline_stability_analysis_20260628/sr24_graphon_vs_graphnone/report.md
```

Decision: do not use a single same-root dense EAGLE3 run as the only paired
quality oracle for SR24. The next accuracy gate should compare SR24 output
against a fixed sample reference or run dense twice and report dense-baseline
stability. The current evidence no longer proves that dynamic mixed CUDA Graph
causes the quality drop.

## Clean Serving Snapshot

Do not over-interpret speedups across different roots. The rows below are used
to identify bottleneck class: acceptance, scheduler, graph, operator, or GPU
utilization.

| row | full-batch tok/s | total tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3, late-MLP root | `3021.525` | `2185.492` | `1.395` | `80.6%` | n/a |
| late-MLP `base_only_24` | `5389.838` | `2739.866` | `2.444` | `81.8%` | `{"FULL":69,"NONE":2}` |
| graph-on `speclink_t08`, gate_up16 high-conf | `2314.728` | `1992.979` | `1.476` | `93.1%` | `{"FULL":107,"NONE":32}` |
| route-reuse eager `speclink_t08` | `2222.398` | `1871.695` | `1.466` | `89.9%` | `{"NONE":139}` |
| exact late-MLP `all_corrected_24` | `2551.468` | `1778.923` | `1.400` | `88.4%` | `{"FULL":77,"NONE":2}` |
| gate-up16 `speclink_t08` + adaptive dense fallback | `2975.063` | `2122.919` | `1.428` | `86.8%` | `{"FULL":73,"NONE":2}` |
| user-requested breakdown dense | `3019.523` | `2185.504` | `1.395` | `80.4%` | `{"FULL":38,"NONE":43,"PIECEWISE":1}` |
| user-requested breakdown `speclink_t08` prefix6 | `4076.344` | `2044.725` | `2.382` | `79.8%` | `{"FULL":74,"NONE":2}` |
| broad `base_only_24`, all Llama leafs | `5591.919` | `2678.493` | `2.624` | `76.7%` | `{"FULL":62,"NONE":2}` |
| MLP `all_corrected_24`, sparse residual | `2229.174` | `1496.682` | `1.397` | `87.2%` | `{"FULL":62,"NONE":2}` |
| MLP `all_corrected_24`, chunked compressed residual | `837.875` | `562.780` | `1.395` | `82.8%` | `{"NONE":64}` |
| MLP `all_corrected_24`, current Triton compressed residual | `279.057` | `197.491` | `1.389` | `84.6%` | `{"NONE":64}` |

Immediate read:

- `base_only_24` is not the current problem. Sparse base alone has speed
  headroom and normal graph coverage in the late-MLP scope.
- `all_corrected_24` is slow even with accepted length intact and healthy graph
  coverage. That isolates the mixed operator cost: sparse base plus residual
  correction is too much work.
- `speclink_t08` remains below the useful target even when scheduler sync is
  reduced. GPU utilization is high, so this is not an idle-GPU problem.
- Adaptive dense fallback removes the two-pass mixed Linear cost and brings
  `speclink_t08` back near dense (`2975.063` versus `3018.226` full-batch
  tok/s in the same root), but it does not create a sparse speedup. It is a
  diagnostic/safety fallback, not the final 1.2x path.
- In the latest short fixed-total run, `speclink_t08` is fast in the
  full-batch generation window but slower in total tok/s. That is a different
  issue from kernel throughput: it is dominated by the startup/TTFT and tail
  part of a 64-request fixed-total run. Use full-batch tok/s for verifier
  hot-path comparisons and total tok/s for end-to-end serving behavior.
- The new broad `base_only_24` probe is not slow. It increases accepted draft
  tokens from about `1.40` to `2.62`, keeps good graph coverage, and improves
  full-batch throughput by about `1.79x` over the same-root dense baseline.
  Its lower average GPU utilization is not a failure mode in this short run;
  peak utilization still reaches `99%`, and the workload simply finishes in
  fewer verifier steps.
- The new exact `all_corrected_24` probes show that accepted length is not the
  reason it is slow. The accepted draft length stays around dense (`1.39-1.40`),
  while GPU utilization stays high. The cost is the exact correction operator:
  sparse residual is the best currently runnable exact path, but it is still
  much slower than dense; compressed residual is GPU-resident but either OOMs
  when prewarmed or becomes far too slow when materialized/used dynamically.

## Base-Only Probe, 2026-06-28

Artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_broad_clean_bs64_math128_20260628/report.md
```

This run used all supported Llama leafs
`qkv_proj,o_proj,gate_up_proj,down_proj`, `mode=base_only`, K=8,
client concurrency 64, fixed 64 requests, and max new tokens 128.

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | peak GPU util | CUDA Graph | storage/dense |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| dense EAGLE3 | `3126.057` | `2184.128` | `1.400` | `86.3%` | `99.0%` | `{"FULL":19,"NONE":44,"PIECEWISE":1}` | n/a |
| broad `base_only_24` | `5591.919` | `2678.493` | `2.624` | `76.7%` | `99.0%` | `{"FULL":62,"NONE":2}` | `0.625` |

Answer to the current base-only question: in this current broad probe,
`base_only_24` is not slow because accepted length collapsed or because GPU was
idle. It is faster, accepts longer draft spans, and has better CUDA Graph
coverage. If a later `base_only_24` row is slow, it should be treated as a
scope/operator-shape regression, not as the default behavior of base-only.

## All-Corrected Operator Probes, 2026-06-28

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_mlp_clean_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cuda_mlp_clean_bs64_math128_20260628/work/all_corrected_24/bs64/rep1/server.log
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cuda_chunked_mlp_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_triton_mlp_bs64_math128_20260628/report.md
```

Scope: `gate_up_proj,down_proj` only, exact all-corrected mode, K=8,
client concurrency 64, fixed 64 requests, max new tokens 128.

| path | result | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph | storage/dense | read |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| dense same-root reference | ok | `3017.662` | `2184.601` | `1.396` | `80.3%` | `{"FULL":20,"NONE":43,"PIECEWISE":1}` | n/a | baseline |
| sparse base + sparse residual | ok | `2229.174` | `1496.682` | `1.397` | `87.2%` | `{"FULL":62,"NONE":2}` | `1.1875` | best runnable exact all-corrected path here, but still slower than dense |
| compressed residual, prewarm full dense residual | failed | n/a | n/a | n/a | n/a | n/a | n/a | OOM while materializing cached dense residual weight; tried to allocate `896MiB` with only `412MiB` free |
| compressed residual, chunked materialization on GPU | ok | `837.875` | `562.780` | `1.395` | `82.8%` | `{"NONE":64}` | `1.125` | residual values are GPU-resident, but runtime materialization is too expensive |
| compressed residual, current Triton kernel | ok | `279.057` | `197.491` | `1.389` | `84.6%` | `{"NONE":64}` | `1.125` | current direct Triton residual kernel is much slower than chunked materialization |

Important implementation facts:

- `compressed_dense` is on GPU in the runnable chunked and Triton paths:
  `compressed_residual_runtime_on_gpu=True`,
  `compressed_residual_non_gpu_modules=[]`, and
  `residual_device_counts={'cuda:0': 64}`.
- The failed prewarm path is also a GPU path; it fails because caching a full
  dense residual weight for this MLP scope exceeds the 32GB card's memory
  headroom during attach.
- Therefore the next useful optimization is not "move compressed_dense from CPU
  to GPU"; that is already true in the tested paths. The needed optimization is
  a fused operator that avoids repeated dense residual materialization and also
  avoids the two-pass sparse-base plus residual-correction cost.

## User-Requested Seven-Part Run

The newest focused run is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_math_bs64_k8_20260628/seven_part_breakdown/report.md
```

It ran `dense_baseline`, `all_corrected_24`, and prefix6 `speclink_t08` on
Llama-3.1-8B, `math_reasoning`, K=8, client concurrency 64, fixed 64 requests,
and max new tokens 128.

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3019.523` | `2185.504` | `1.395` | `80.4%` | `{"FULL":38,"NONE":43,"PIECEWISE":1}` |
| all-corrected dense-fastpath control | `3015.577` | `2182.534` | `1.395` | `84.3%` | `{"FULL":38,"NONE":43,"PIECEWISE":1}` |
| prefix6 `speclink_t08` | `4076.344` | `2044.725` | `2.382` | `79.8%` | `{"FULL":74,"NONE":2}` |

Important read:

- `speclink_t08` is not slower in the full-batch verifier hot path in this
  short run. It is slower in total tok/s because total includes TTFT and the
  tail after requests start finishing. The recorded first-token range is about
  `0.655-0.719s` for `speclink_t08`, versus `0.415-0.476s` for dense.
- CUDA Graph coverage is healthy for `speclink_t08` in this run
  (`FULL=74`, `NONE=2`), so graph loss is not the explanation here.
- GPU util is still high enough that this is not an idle-GPU problem.

Component breakdown for `speclink_t08`:

| part | measured value | read |
| --- | --- | --- |
| scheduler / mask build | wall `36.011ms/step`, row bucket `0.147ms/step`, bucket build `0.145ms/step`, indexed CUDA mask kernel `0.049ms/step`, bucket top-k `0.108ms/step` | The large wall number includes diagnostic timing overhead and should not be read as clean serving cost; the GPU-side bucket/mask kernels are small. |
| base sparse linear | average sparse base `1.935ms/call`; `gate_up_proj=16-31` `2.605ms/call`; about `334.9` rows/call | This is the largest measured GPU-side operator cost. Sparse base is only worthwhile if it replaces enough dense work. |
| residual correction | dense correction `0.121ms/call`; `gate_up_proj=16-31` `0.154ms/call`; about `31.2` corrected rows/call | Smaller than sparse base in this prefix6 bucketed path. |
| gather/scatter | `0.288ms/call` | Larger than the dense correction itself, so fused assembly still matters. |
| routing statistics | draft residual/base `15776/4248`; non-draft residual/base `2503/3788`; draft residual fraction `0.788`; bucket fill `0.876`, active bucket rows about `28.0/32` | Quality-safe prefix6 sends many draft rows through residual. That preserves accuracy better, but limits sparse savings. |
| CUDA Graph | `{"FULL":74,"NONE":2}` | Graph coverage is healthy for this fixed-shape bucketed path. |
| GPU util | avg `79.8%`, peak `99.0%` | GPU is busy; the question is useful work per step and end-to-end tail, not occupancy. |

## Seven-Part Breakdown

| part | measured value | current read |
| --- | --- | --- |
| scheduler / mask build | graph-on `speclink_t08`: mask wall `0.237ms/step`, batched builder `0.173ms/step`, row bucket `0.004ms/step`, bucket build `0.002ms/step`, row indices `0.001ms/step` | The clean graph-on path has sub-ms scheduler work. CPU-side mask building is not the first bottleneck for this candidate. |
| scheduler / dynamic row lists | route-reuse eager: mask wall `19.810ms/step`, row bucket `19.578ms/step`, mixed row indices `19.575ms/step`, CUDA Graph `NONE` for every step | Variable-length row-list construction via dynamic row indices is a real trap. It may be eager-correct, but it is not a good serving path unless it becomes fixed-shape/GPU-side. |
| base sparse linear | diagnostic `gate_up_proj=16-31` sparse base `0.570ms/call`, about `532` rows/call | Sparse base is not free. It can beat dense only if correction work stays small. |
| residual correction | diagnostic dense residual correction `0.570ms/call` for the same gate-up scope | Correction is as expensive as the sparse base in this path, so the two-pass method loses the base-only gain. |
| gather/scatter | diagnostic gather/select/scatter wrapper `0.083ms/call` | Secondary cost. It matters, but optimizing only gather/scatter cannot recover the current gap. |
| routing statistics | diagnostic draft residual/base `9300/6804`, draft residual fraction `0.577`; non-draft residual/base `2013/3735`. Clean graph-on non-draft residual/base is `6729/5737`. | Too many rows still go through residual correction. The controller is not sparse enough for a two-pass operator. |
| CUDA Graph | base-only late-MLP `{"FULL":69,"NONE":2}`; graph-on `speclink_t08` `{"FULL":107,"NONE":32}`; route-reuse eager `{"NONE":139}` | Graph loss is not the only issue anymore, but every new routing variant must preserve graph coverage. Dynamic row routing regresses this badly. |
| GPU util | graph-on `speclink_t08` avg `93.1%`, exact `all_corrected_24` avg `88.4%`, dense late-MLP avg `80.6%` | The GPU is busy. The slowdown is inefficient useful work and small/extra kernels, not lack of GPU occupancy. |

## Adaptive Dense Fallback Check

I ran a conservative fallback ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_adaptive_densefallback_bs64_math128_20260628/report.md
```

It uses the same gate-up16 high-confidence selector, but when the mixed
`dense_rows` path would otherwise compute full sparse base plus full dense and
then select rows, it falls back to dense-only for that leaf. This is
accuracy-conservative because it corrects extra rows.

| method | full-batch tok/s | total tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3018.226` | `2183.337` | `1.395` | `82.4%` | n/a |
| `speclink_t08` + adaptive fallback | `2975.063` | `2122.919` | `1.428` | `86.8%` | `{"FULL":73,"NONE":2}` |

I also added low-overhead runtime counters for this fallback. The validation
smoke is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_adaptive_fallback_stats_smoke_20260628/report.md
```

The smoke records:

```text
sr24_adaptive_dense_fallback_calls=304
sr24_adaptive_dense_fallback_rows=38432
sr24_adaptive_dense_fallback_candidate_rows=38432
```

Read: this fallback mostly converts the mixed candidate into dense verifier
work. That proves the two-pass operator is the current cost center, but it also
shows why fallback alone cannot reach `1.2x`: it intentionally gives up most of
the sparse-base speedup.

## Prefix-Protection Accuracy Check

The aggressive all-MLP + Triton override route can hit the full-batch speed
target, but it previously failed GSM8K-50 quality. I tested stronger prefix
protection with `critical_prefix`, `min_prefix_residual=6`, and
`extra_after_low=1`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_paired_gsm8k50_20260628/report.md
```

Result:

| mode | GSM8K-50 | pair reg | pair imp | dense retain | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `0.7200` | `0` | `0` | `1.0000` | reference |
| all-MLP prefix6 `speclink_t08` | `0.7200` | `2` | `2` | `0.9444` | aggregate restored, paired regressions remain |

The paired regressions are `doc_id:11` and `doc_id:15`. They are not simple
formatting errors: the SR24 completions take different arithmetic paths. So
prefix expansion alone does not solve the precision problem; it only restores
aggregate score through cancellation.

## Bottleneck Conclusion

The current slow path is not mainly accepted-length collapse, not mainly CPU
mask construction, and not mainly GPU idle time. The bottleneck is the mixed
linear path:

```text
2:4 sparse/base output for many rows
+ dense or residual correction for too many rows
+ gather/scatter or row assembly
+ graph constraints around dynamic row shapes
```

This explains why `base_only_24` can be fast while both exact
`all_corrected_24` and selective `speclink_t08` are slow. The selective path
keeps enough residual rows to preserve quality, but that residual fraction is
too high for the current two-pass implementation.

## Fresh User-Pivot Breakdown Run

I added the requested seven-part breakdown report path so a single run now
joins clean serving, diagnostic Linear/routing timing, and isolated operator
microbench rows. The report generator now includes `component_microbench`
artifacts instead of silently skipping them.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_fresh_bs64_math_k8_20260628/README.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_fresh_bs64_math_k8_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_fresh_bs64_math_k8_20260628/seven_part_report/operator_microbench.csv
```

Scope: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client concurrency 64,
64 total clean requests, max new tokens 128. The instrumented row uses fewer
requests and is diagnostic-only.

| method | row kind | full-batch tok/s | total tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | clean | `3023.082` | `2186.780` | `1.395` | `80.6%` | n/a |
| `base_only_24` | clean | `3244.196` | `2231.409` | `1.547` | `86.8%` | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | clean | `2903.555` | `1872.259` | `1.433` | `88.1%` | `{"FULL":62,"NONE":2}` |

Seven-part read from the same report:

| part | current value | read |
| --- | ---: | --- |
| scheduler / mask build | clean `speclink_t08` wall `0.969ms/step`; diagnostic exact row `8.451ms/step` | Clean scheduler work is still sub-ms; the larger diagnostic number is synchronization/request-loop overhead and should not drive optimization by itself. |
| base sparse linear | diagnostic `gate_up_proj=16-31` sparse base `1.012ms/call`, `267.65` rows/call | This is the largest localized GPU-side operator cost in the mixed path. |
| residual correction | diagnostic dense-row correction `0.174ms/call`, `32` rows/call | Smaller than sparse base here, but it is still an extra pass on top of sparse base. |
| gather/scatter | `0.016ms/call` | Not the first bottleneck in this fixed-bucket low-sync row. |
| routing statistics | draft residual/base `6376/1544`; draft residual fraction `0.805`; bucket fill `0.979`; actual/requested bucket rows `1253/7396` | Too many draft rows still need dense/residual treatment for the current two-pass operator. |
| CUDA Graph | clean `speclink_t08` `{"FULL":62,"NONE":2}` | Graph coverage is not the first suspect for this candidate. |
| GPU util | clean `speclink_t08` avg `88.1%`, peak `100%` | GPU is busy; focus on useful-work efficiency rather than occupancy. |

The operator microbench strengthens the same conclusion. For gate-up
`512 x 28672 x 4096`, base sparse graph is about `0.65x-0.66x` dense, but the
current mixed proxy becomes `1.03x` dense at residual fraction `0.125` and
`1.52x` dense at residual fraction `0.5`. Down projection is better at low
residual fractions, but also loses once residual fraction reaches `0.5`.

Decision from this fresh run: do not make CPU-sync cleanup the main path right
now. The clean candidate already has good graph coverage and high GPU
utilization. The next optimization must either reduce residual rows under a
paired quality gate, or replace the current two-pass sparse-base plus dense-row
correction with a fixed-shape fused/packed GPU operator.

## All-Corrected Real Operator Check

I fixed the diagnostic path so `--no-sr24-all-corrected-dense-fastpath` really
measures the all-corrected sparse/residual operator. Before this fix,
`all_corrected_24` still short-circuited through the all-residual dense Linear
state, so the breakdown had no Linear component timings.

Equivalence smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_real_operator_equiv_20260628/equivalence.json
```

The small-module `dense_rows` all-corrected check matched dense exactly:
`max_abs=0`, `mean_abs=0`.

Serving/operator artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_real_operator_probe_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_real_operator_probe_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_directcslt_probe_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_dense_gpu_probe_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_dense_prewarm_probe_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_operator_compare_with_compressed_20260628/
```

Scope: Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, all-corrected
`gate_up_proj=16-31`. These are short diagnostic serving rows, not final
throughput matrix results.

| path | bs | total tok/s | same-root dense tok/s | component timing | read |
| --- | ---: | ---: | ---: | --- | --- |
| `dense_rows`, no dense fastpath | 64 | `945.367` | `1394.141` | base sparse `0.788ms/call`, full dense residual `0.684ms/call`, select `0.043ms/call` | This is base sparse plus a second full dense GEMM, so it is expectedly slower than dense. |
| `torch_sparse` residual | 64 | `691.447` | `1399.524` | base sparse `1.048ms/call`, residual sparse `1.162ms/call`, add `0.059ms/call` | Two PyTorch semi-structured GEMMs are slower than dense here. |
| `torch_sparse` residual + direct cuSPARSELt | 64 | `711.781` | n/a in same table | base sparse `1.066ms/call`, residual sparse `1.134ms/call`, add `0.058ms/call` | Direct cuSPARSELt only gives a small improvement; it does not fix the operator shape. |
| `compressed_dense`, runtime materialize | 32 | `428.310` | `988.869` | base sparse `0.684ms/call`, materialize `0.293ms/call`, GEMM `0.135ms/call`, add `0.227ms/call` | It is GPU-resident, but runtime materialization/add overhead is large. |
| `compressed_dense`, cache+prewarm | 32 | `641.698` | `1016.814` | base sparse `0.908ms/call`, materialize `0.0004ms/call`, GEMM `1.028ms/call`, add `0.271ms/call` | Prewarm removes materialization, but the residual GEMM becomes the main cost. |

`compressed_dense` is not falling back to CPU in these probes:
`residual_device_counts={"cuda:0":16}`,
`compressed_residual_runtime_on_gpu=True`,
`compressed_residual_non_gpu_modules=[]`, and
`residual_extract_cpu_fallback_chunks=0`.

Current all-corrected conclusion: the exact sparse/residual decomposition is
not faster with the available PyTorch/cuSPARSELt/compressed paths. To make
`all_corrected_24` a real structured-sparse speed path, the next implementation
must avoid the two independent GEMMs plus add/select pattern. The useful kernel
target is a fused exact operator that computes `X @ (W_base + W_residual)` from
the compressed 2:4 pieces without materializing a second dense GEMM or launching
two full semi-structured GEMMs. Until that exists, dense fastpath remains the
correct control path for all-corrected correctness, not a speedup path.

## Next Work

Use this breakdown as the regression gate for every candidate:

1. Keep clean scheduler/mask build below about `1ms/step`.
2. Keep CUDA Graph coverage high; reject dynamic row-routing paths that fall to
   all `NONE` unless they are only diagnostic.
3. Reduce residual rows with a paired accuracy gate, or implement a fixed-shape
   GPU-side/fused sparse-base plus residual-correction operator.
4. Do not spend the next pass on CPU-sync-only cleanup unless a fresh clean row
   shows scheduler/mask time returning to multi-ms scale.
5. Report each new candidate with the same seven fields: scheduler/mask,
   base sparse, residual correction, gather/scatter, routing stats, CUDA Graph,
   and GPU utilization.

## Current Why-Slow Breakdown, 2026-06-28

This is the latest breakdown run aligned with the seven items in the user
request. It uses Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, client
concurrency 64, `gate_up_proj=16-31`, `critical_prefix`, prefix residual
minimum 6, and bucket size 32.

Final readable reports:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_why_slow_current_breakdown_20260628/seven_part_report_with_eager_linear/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_why_slow_current_breakdown_20260628/component_summary_with_eager_linear/report.md
```

Important measurement detail: clean serving uses low-overhead graph-capable
settings for throughput/GPU util/CUDA Graph counts. Linear component timing
uses a separate eager-only diagnostic run with `--enforce-eager`; otherwise
the default vLLM compile/cache path can absorb the Python SR24 Linear hooks and
leave `--sr24-breakdown-linear` fields empty. I updated
`scripts/run_sr24_slowdown_breakdown.py` so instrumented rows no longer inherit
`--sr24-default-vllm-compile`.

Clean serving result:

| method | full-batch tok/s | total tok/s | same-root speedup | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3427.697` | `2319.447` | `1.000x` | `88.643%` | n/a |
| `speclink_t08` | `3377.320` | `2270.292` | `0.985x` | `88.357%` | `{"FULL":114,"NONE":77,"PIECEWISE":1}` |

Seven-part read:

| part | current evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean `0.316ms/step`; exact diagnostic `5.856ms/step`, mostly request-loop sync | Clean scheduler cost is sub-ms; exact diagnostic sync is not the clean throughput bottleneck. |
| base sparse linear | eager diagnostic `gate_up_proj=16-31` sparse base `1.026ms/call`, `129.27` rows/call | This is the largest localized GPU-side operator cost. |
| residual correction | dense-row correction `0.170ms/call`, bucket rows `32/call` | Secondary to sparse base in this candidate, but still an extra path. |
| gather/scatter | `0.015ms/event` | Not the first bottleneck for the current fixed-bucket path. |
| routing stats | draft residual/base `4459/1077`; non-draft residual/base `692/891`; bucket fill `0.985`; actual/requested bucket rows `1671/5151` | Bucket is full, but many draft rows still require residual correction. |
| CUDA Graph | clean `speclink_t08` has `FULL=0.594`, `NONE=0.401` | Graph misses may contribute, though the run remains near dense. |
| GPU util | clean SR24 `88.357%`, dense `88.643%` | GPU is busy; this is useful-work efficiency, not an idle-GPU problem. |

Operator microbench result: the sparse base alone has headroom
(`gate_up` base sparse graph is about `0.65x` dense, down base sparse about
`0.57x` dense), but current mixed sparse-base plus dense-row correction erases
that headroom once residual rows become nontrivial. For `gate_up`, the mixed
proxy is already `1.03x` dense at residual fraction `0.125`; for `down`, it is
good up to about `0.25` but loses at `0.5`.

Current conclusion: do not pivot primarily to CPU-side scheduler cleanup. The
clean path already keeps mask build below `1ms/step` and keeps GPU utilization
near dense. The next useful direction is either:

1. Reduce residual rows while keeping paired accuracy stable, especially draft
   residual rows.
2. Improve graph coverage for dynamic mixed steps.
3. Replace the two-pass sparse-base plus dense-row correction with a fixed
   shape GPU operator/fused path. Gather/scatter is too small to be the first
   target in this configuration.

## Follow-up CPU Sync Ablation, 2026-06-28

Artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_bs64_math256_20260628/
```

Scope: Llama-3.1-8B, `math_reasoning`, bs/client concurrency 64, EAGLE3 K=8,
`speclink_t08`, `gate_up_proj=16-31`, `all_if_any_low`, threshold `0.4`,
minimum residual prefix `4`, max new tokens `256`, 64 requests.

| variant | total tok/s | full-batch tok/s | TPOT ms | GPU util | mask wall ms/step | main read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `low_sync_stats_off` | `2448.586` | `3352.065` | `19.494` | `91.231%` | n/a | Best total throughput; runtime stats removed. |
| `sync_mask_state` | `2430.538` | `3325.289` | `19.638` | `88.929%` | `36.156` | Mask-state sync is expensive in wall timing but did not hurt this short full-batch row much. |
| `low_sync_stats_on` | `2101.068` | `3276.461` | `19.574` | `92.125%` | `0.213` | Low-sync mask build itself is sub-ms; runtime stats still cost total throughput. |
| `low_sync_gpu_counts` | `2086.288` | `3247.368` | `19.752` | `90.938%` | `30.881` | GPU-count diagnostic is useful for routing stats, not for clean throughput. |
| `sync_heavy` | `1342.352` | `1964.831` | `32.521` | `69.917%` | `10.783` | Per-request sync-heavy routing is clearly bad. |

CPU sync conclusion: reducing CPU synchronization is worth keeping. The
`sync_heavy` variant cuts total throughput by about `45%` versus
`low_sync_stats_off` (`1342` vs `2449` tok/s) and lowers GPU utilization to
about `70%`. However, the best low-sync full-batch throughput is still only
about `3352` tok/s, close to the earlier clean SR24 rows and not a clear
speedup over dense EAGLE3. CPU sync is therefore a necessary cleanup, not the
main path to the `1.2x` target.

The low-sync path to keep is:

```text
SPECLINK_SR24_REDUCE_CPU_SYNC=1
SPECLINK_SR24_SYNC_MASK_STATE=0
SPECLINK_SR24_DISABLE_RUNTIME_STATS=1
```

Use runtime stats or GPU-count breakdown only as diagnostics, and keep their
outputs under `results.bak/` or `temp/`.

## Accuracy Diagnostic Caveat, 2026-06-28

Paired GSM8K-20 probes exposed a measurement caveat before the controller can
be trusted:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_quality_gateup_only_gsm8k20_20260628_followup/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_speclink_t08_gateup_cap0_dense_guard_gsm8k20_20260628_followup/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_doc2_modes_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_gsm8k20_batchshape_20260628/
```

The two lm-eval runs used the same fixed manifest
`configs/task_manifests/gsm8k_cot_20.json`, but dense EAGLE3 itself changed
from `0.65` to `0.75` exact match across independent serving runs. The
`all_corrected_fastpath` replay is a dense no-op control, yet independent
offline replay processes still differed from the dense replay on several
documents. Therefore, one-run paired accuracy can overstate SR24 regressions on
reasoning tasks.

Two extra dense-only repeats confirmed this variance:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_repeat_gsm8k20_r1_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_repeat_gsm8k20_r2_20260628/
```

Dense EAGLE3 exact match on the same 20-doc manifest across four independent
runs was:

| run | exact match |
| --- | ---: |
| `quality_dense` | `0.65` |
| `cap0_dense` | `0.75` |
| `repeat_r1` | `0.65` |
| `repeat_r2` | `0.70` |

Variable examples include doc 2 (`70,000` vs `50,000`), doc 10 (`365.4` vs
`366`), and doc 11 (`8328` vs `694`). These are dense EAGLE3 changes without
SR24, so GSM8K-20 is too small and too unstable for a single-run quality gate.

Current accuracy rule: before changing `speclink_t08` thresholds, first
quantify dense EAGLE3 repeat variance on the same manifest and compare SR24
against a repeated dense/no-op control. A single dense-correct/SR24-wrong sample
is not sufficient evidence unless it reproduces under the same serving shape
and a dense no-op control stays token-identical.

## Historical Refreshed Seven-Part Breakdown, 2026-06-28

This older user-requested slowdown read is kept for history. Prefer the
corrected fast-candidate breakdown at the top of this file when deciding the
next optimization direction.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_20260628/README.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_current_breakdown_20260628/component_microbench/summary.md
```

Scope: Llama-3.1-8B, `math_reasoning`, bs/client concurrency 64, EAGLE3 K=8,
`gate_up_proj=16-31`, max new tokens 256 for clean serving, max new tokens 128
for instrumented serving.

Representative clean serving rows:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2320.531` | `3432.423` | `1.697` | `88.357%` | n/a |
| `base_only_24` | `2784.580` | `3959.937` | `2.027` | `90.500%` | `{"FULL":126,"NONE":2}` |
| `speclink_t08` | `2027.013` | `2929.340` | `1.700` | `87.688%` | `{"NONE":128}` |

Current read:

| part | latest evidence | decision |
| --- | --- | --- |
| scheduler / mask build | Clean `speclink_t08` mask build is `0.448ms/step`; low-sync stats-on row is `0.235ms/step`. Sync-heavy and GPU-count rows can show `8-17ms/step`, but those are diagnostic/ablation rows. | The default clean scheduler is not the main bottleneck. Keep low-sync defaults and do not judge clean serving from exact-routing/GPU-count rows. |
| base sparse linear | Instrumented `speclink_t08` localizes `gate_up_proj=16-31` sparse base at `0.568ms/call` for about `275` rows/call. | Sparse base is a large GPU-side cost; it must replace enough dense work to pay off. |
| residual correction | Dense-row residual correction is `0.336ms/call`; `all_corrected_24` is worse at `0.627ms` base plus `0.583ms` correction. | The current exact correction path is additive work, not a speed path. |
| gather/scatter | `index_select`/`index_add_`/assembly is `0.037ms/call` in this gate-up scoped run. | Secondary in this scope; optimize after reducing sparse base plus correction cost. |
| routing statistics | Instrumented `speclink_t08`: draft residual/base `8533/4595`, non-draft residual/base `1641/1823`, draft residual fraction `0.650`. GPU-count diagnostic gives draft residual fraction `0.692`. | Too many draft rows still require residual correction. Controller-only work must prove a lower residual fraction without paired accuracy loss. |
| CUDA Graph | `base_only_24` is almost all FULL graph (`126/128`), while clean `speclink_t08` is all NONE (`128/128`). | Mixed dynamic SR24 loses CUDA Graph coverage. This is a hard guardrail for future candidates. |
| GPU util | Clean dense `88.4%`, base-only `90.5%`, clean `speclink_t08` `87.7%`. | The GPU is busy; the slowdown is inefficient useful work and graph/shape overhead, not an idle-GPU problem. |

Immediate optimization direction: do not spend the next pass on generic
threshold sweeps or CPU cleanup alone. The useful path is either:

1. make mixed `gate_up_proj=16-31` graph-safe and fused/packed so it does not
   pay sparse base plus dense correction as separate work, or
2. reduce draft residual rows sharply with a repeated/no-op paired accuracy
   gate.

I also tightened the offline reducer so `sync_mask_state`, `sync_heavy`, and
`--sr24-breakdown-gpu-counts` rows are classified as diagnostic rows instead of
clean serving rows. This prevents the report from treating diagnostic
`16ms/step` mask timing as the clean path.

## Max-256 Graph-Safe Follow-Up, 2026-06-28

I then reran two clean max-new-tokens-256 serving probes to separate CUDA Graph
loss from operator/useful-work limits.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_graphon_bs64_math256_20260628/report.md
```

The first run keeps the quality-safe scope
`gate_up_proj=16-31;down_proj=8-15`, uses the `critical_prefix` controller with
threshold `0.6`, forced residual prefix `4`, one extra residual row after a low
confidence token, bonus-row correction, bucket size `32`, static mask buffers,
and graph-safe bucket capture.

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2310.628` | `3416.565` | `1.693` | `88.571%` | n/a |
| `base_only_24` | `2873.587` | `4253.457` | `2.143` | `87.917%` | `{"FULL":126,"NONE":2}` |
| `speclink_t08` | `2606.145` | `3830.144` | `2.169` | `91.231%` | `{"FULL":94,"NONE":2}` |

This confirms CUDA Graph was one large bottleneck for the older clean
`speclink_t08` row: after graph-safe capture, `speclink_t08` improves from
below dense to `1.128x` total tok/s and `1.121x` full-batch tok/s. It still
does not reach the `1.2x` target, even though accepted draft length is higher
and GPU utilization is high. The remaining gap is therefore useful-work
efficiency in the mixed sparse-plus-correction path.

The second run broadens SR24 to all MLP `gate_up_proj,down_proj` leaves. This
tests whether a much stronger sparse base upper bound can overcome correction
cost.

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2320.476` | `3432.682` | `1.697` | `88.500%` | n/a |
| `base_only_24` | `3807.640` | `6328.354` | `3.304` | `88.111%` | `{"FULL":94,"NONE":2}` |
| `speclink_t08` | `1839.157` | `3830.994` | `3.158` | `80.333%` | `{"FULL":134,"NONE":26}` |

The all-MLP base-only upper bound is very large (`1.844x` full-batch), but
`speclink_t08` does not convert that into a clean serving win. It gets longer
accepted draft spans, yet total throughput collapses and full-batch throughput
is only `1.116x` over dense. This means simply widening the sparse scope is not
enough: broad residual protection increases correction work and graph/shape
churn faster than accepted-token benefit.

### Current Bottleneck Answer

For the user-requested breakdown table, the current answer is:

| part | current answer after graph-safe probes |
| --- | --- |
| scheduler / mask build | Low-sync fixed-shape mask build is already sub-ms in clean rows. It is not the first bottleneck unless a candidate reintroduces dynamic Python row loops. |
| base sparse linear | Base sparse is fast enough to give a real upper bound (`base_only_24` wins), but it is only useful when it replaces dense work instead of adding another path. |
| residual correction | This is the main remaining issue. Current mixed verification pays sparse base plus dense correction for many rows, so useful work is duplicated. |
| gather/scatter | Secondary for the quality-safe gate-up scope; visible for broader/all-MLP routing. Fusion is better than scatter-only tuning. |
| routing statistics | Accepted draft length is not the issue in the best graph-safe row (`2.169` vs dense `1.693`). The issue is the number/cost of residual-protected rows required to keep quality. |
| CUDA Graph | Graph loss explained the worst clean row. After graph-safe bucket capture, throughput improves to about `1.12x`, but graph alone is not enough for `1.2x`. |
| GPU util | GPU stays busy (`~88-91%` in key rows). This is inefficient useful work, not idle GPU. |

Next implementation work should therefore focus on one of two concrete paths:

1. a graph-safe fused or packed mixed MLP/Linear operator that avoids computing
   sparse base on rows that will be corrected by dense output, or
2. a stricter residual-row controller that sharply lowers corrected rows while
   passing repeated dense/no-op paired accuracy gates.

Existing row-routed MLP code is useful diagnostic evidence but should not be
promoted as the default path yet. It can skip sparse-base work for corrected
rows, but prior serving rows show lower accepted draft length and heavy
base-side/correction work. A new row-routed candidate must first show recovered
accepted draft length, high CUDA Graph coverage, and a clean serving win before
more assembly tuning is worthwhile.

## Focused Follow-Up Ablations, 2026-06-28

I tried three narrow follow-ups on the graph-safe max-new-tokens-256 setup.
These are not final matrix rows; they are direction checks for the current
bottleneck.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_adaptive_densefallback_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_tritonoverride_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_t05_tritonoverride_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix3_t06_tritonoverride_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routebucket_cached_graphon_allow_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_dense_gemm_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_bucket32_current_shape_20260628/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_allcorrected_residual_sparse_20260628/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_leafbackend_gateup_sparse_down_dense_graphon_bs64_math256_20260628/report.md
```

| candidate | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| graph-safe quality-safe baseline | `2606.145` | `3830.144` | `2.169` | `91.231%` | `{"FULL":94,"NONE":2}` | current best correctness-oriented scoped row |
| adaptive dense fallback | `2123.908` | `3291.031` | `1.719` | `90.933%` | `{"FULL":190,"NONE":2}` | negative; conservative dense fallback disrupts speculative progress |
| Triton bucket override | `2638.427` | `3855.145` | `2.163` | `89.846%` | `{"FULL":94,"NONE":2}` | small positive; useful as a low-risk assembly replacement, not enough for `1.2x` |
| threshold `0.5` + Triton override | `2590.764` | `3869.191` | `2.163` | `91.385%` | `{"FULL":126,"NONE":2}` | no useful win; `min_prefix_residual=4` dominates residual-row selection |
| prefix3 + threshold `0.6` + Triton override | `2627.234` | `3833.443` | `2.128` | `90.231%` | `{"FULL":94,"NONE":2}` | no useful win; fewer forced prefix residual rows lower acceptance enough to erase savings |
| graph-safe route bucket rows | `2522.459` | `3671.220` | `2.082` | `91.538%` | `{"FULL":126,"NONE":2}` | negative; split dense/sparse routing preserves graph but small gathers/GEMMs/assembly erase the skipped sparse-base work |
| Triton bucket dense GEMM scatter | `2468.352` | `3581.018` | `2.188` | `91.538%` | `{"FULL":94,"NONE":2}` | negative; removing the intermediate dense-output tensor is not enough and the custom correction matmul is slower than torch dense + Triton override |
| per-leaf backend: gate_up sparse residual, down dense_rows | `2593.116` | `3885.001` | `2.146` | `91.308%` | `{"FULL":126,"NONE":2}` | small full-batch win over dense_rows+Triton override, but total is lower and still below `1.2x` |

The useful outcome is negative but clarifying:

1. Adaptive dense fallback is not the speed path. It keeps CUDA Graph coverage
   but reduces accepted draft length, so the end-to-end speculative loop loses.
2. Triton bucket override is safe to keep as an assembly/correction ablation
   and gives a small improvement, but it cannot close the remaining gap alone.
3. Simple controller changes around threshold or prefix length do not currently
   produce a better speed-quality point. The residual-row controller appears
   coupled to accepted draft length: removing protection can reduce correction
   work but also reduces useful speculative progress.
4. I added a graph-safe cached row-plan path for `--sr24-route-bucket-rows`
   so it can use persistent bucket/complement rows under CUDA Graph. The
   correctness smoke passed, and the serving row reached `{"FULL":126,"NONE":2}`,
   but throughput was worse than the normal bucket correction path. This
   confirms that PyTorch split routing is not the desired fused operator: it
   avoids sparse-base work for bucket rows, but pays gather, small GEMMs, and
   output assembly.
5. The existing `--sr24-triton-bucket-dense-gemm` correction prototype is also
   negative. It keeps CUDA Graph coverage but is slower than torch dense GEMM
   plus Triton overwrite. Therefore the remaining gap is not just the
   correction-output allocation; the missing piece is a fused/packed base plus
   correction operator or a substantially better base sparse kernel.
6. I added `SPECLINK_SR24_RESIDUAL_BACKEND_BY_LEAF` / runner flag
   `--sr24-residual-backend-by-leaf` for mixed residual backends. The useful
   tested split is `gate_up_proj=torch_sparse;down_proj=dense_rows`, motivated
   by the component microbench: at rows=512 and bucket32, gate/up residual
   sparse delta can be faster than dense-row correction, while down residual
   sparse is slower. The serving row improves full-batch slightly to
   `3885.001` tok/s, but total tok/s drops to `2593.116`. This is a small
   optimization/diagnostic, not the final path.
7. The all-corrected microbench with residual fraction `1.0` confirms that
   exact dual-sparse correction is not a throughput answer for `all_corrected_24`:
   gate/up compressed sparse delta is `1.39x` dense and down is `1.19x` dense.
   All-corrected should use the dense fastpath as the correctness control, or a
   new fused/packed operator; two separate sparse GEMMs do not beat dense.

Current next step: a candidate must change the operator cost model rather than
only changing scalar controller knobs. The most direct target is still a
graph-safe packed/fused mixed MLP/Linear path that avoids sparse-base work on
rows that are going to be corrected, while preserving the current
`critical_prefix@0.6,prefix4,extra1` quality gate or an explicitly stronger
paired accuracy gate.

## CPU Sync and Bucket-Budget Refresh, 2026-06-28

The reduced-sync ablation confirms that CPU synchronization is not the primary
remaining cause of slowdown for the current graph-safe bucket path.

Artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fast_candidate_cpu_sync_ablation_b12_bs64_math256_20260628/
```

Key rows:

| variant | total tok/s | full-batch tok/s | accepted draft/step | GPU util | scheduler/mask wall | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| low_sync_stats_on | `2705.1` | `3967.0` | `2.187` | `90.9%` | `0.338ms/step` | `{"FULL":94,"NONE":2}` |
| low_sync_stats_off | `2615.6` | `3861.6` | `2.094` | `91.2%` | n/a | n/a |
| sync_heavy | `1442.7` | `1960.4` | `1.918` | `59.9%` | `10.984ms/step` | `{"NONE":128}` |

The clean low-sync path is already sub-ms in scheduler/mask work and keeps
healthy CUDA Graph coverage. The old sync-heavy path is bad, but disabling
stats or adding GPU count diagnostics does not unlock the missing `1.2x`.

Bucket-size and quality probes then showed the main tension:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_sweep_b4_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_sweep_b8_b16_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b64_priority_quality_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b64_priority_bs64_math256_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b256_quality_gsm8k50_20260628/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket_copy_b256_bs64_math256_20260628/
```

| candidate | quality | total tok/s | full-batch tok/s | full-batch speedup | accepted draft/step |
| --- | --- | ---: | ---: | ---: | ---: |
| bucket8 copy | not quality-safe; triton+copy GSM8K-50 `0.60` vs dense `0.82` | `2699.6` | `3980.7` | `1.130x` | `2.209` |
| bucket64 priority copy | GSM8K-50 `0.76` vs dense `0.80` | `2435.9` | `3339.8` | `0.973x` | `1.818` |
| bucket256 copy | GSM8K-50 `0.78` vs dense `0.80` | `2400.1` | `3331.1` | `0.970x` | `1.885` |
| quality_gateup_only | safer preset, but speed-negative | `2120.0` | `3288.3` | `0.957x` | `1.712` |

Interpretation: `SPECLINK_SR24_RESIDUAL_BUCKET_SIZE` is a global per-step
budget. At bs64/K8, fixing even a small prefix per request requires far more
than 8 corrected rows globally. Small buckets look fast because they leave many
quality-relevant rows base-only. Once the row budget is raised enough to recover
GSM8K, the current two-pass mixed operator loses the speedup.

I added runner support for the priority knobs
`--sr24-bonus-priority` and `--sr24-draft-position-priority-scale` to
`run_lm_eval_accuracy.py`, and made `run_sr24_slowdown_breakdown.py` forward
`--sr24-preset` instead of hardcoding `manual`. The vLLM Triton bucket-GEMM
prototype now matches `bucket_dense_copy` semantics when both flags are set,
but its quality still trails torch dense-copy on the bucket256 probe, so it
remains diagnostic.

## Bucket Copy / Direct cuSPARSELt Follow-Up, 2026-06-28

I tested the smallest remaining hot-path switches on the same Llama-3.1-8B,
`math_reasoning`, bs64, K=8, max-new-tokens 256 setup:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_bucketcopy_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucketcopy_instrumented_bs64_math128_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b16_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b8_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_qualitysafe_directcslt_bucketcopy_b4_graphon_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_directcslt_bucket16_quality_gsm8k20_20260628/report.md
```

Clean serving results:

| candidate | bucket | total tok/s | same-run dense | total speedup | full-batch tok/s | same-run dense full | full-batch speedup | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bucket dense copy, no direct cslt | 32 | `2640.935` | `2315.294` | `1.141x` | `3828.759` | `3420.673` | `1.119x` | `2.150` | `{"FULL":94,"NONE":2}` |
| bucket dense copy + direct cslt | 32 | `2680.897` | `2330.319` | `1.150x` | `3925.396` | `3442.055` | `1.140x` | `2.147` | `{"FULL":94,"NONE":2}` |
| direct cslt + bucket copy | 16 | `2720.019` | `2319.712` | `1.173x` | `3930.796` | `3429.905` | `1.146x` | `2.198` | `{"FULL":94,"NONE":2}` |
| direct cslt + bucket copy | 8 | `2704.736` | `2271.872` | `1.191x` | `3971.446` | `2914.457` | `1.363x` | `2.201` | `{"FULL":94,"NONE":2}` |
| direct cslt + bucket copy | 4 | `2643.398` | `2318.141` | `1.140x` | `3915.667` | `3426.420` | `1.143x` | `2.097` | `{"FULL":126,"NONE":2}` |

The bucket8 full-batch speedup is inflated by a weak same-run dense full-batch
row (`2914.457`); total tok/s is the more stable comparison there. The best
clean total row is bucket16 with direct cuSPARSELt and dense bucket copy. It is
an improvement over the bucket32 graph-safe row, but still below the requested
`1.2x` total and full-batch target.

The bucket-copy instrumentation confirms the bottleneck ordering:

| component | diagnostic value |
| --- | ---: |
| clean/runtime scheduler mask wall | about `0.448ms/step` |
| exact diagnostic scheduler mask | `0.474ms/step` |
| base sparse Linear | `2.128ms/call` |
| residual dense GEMM | `0.132ms/call` |
| gather/scatter event | about `0.004ms/event` |

This rules out bucket writeback as the main remaining cost. Direct cuSPARSELt
helps because base sparse is the dominant Linear-side cost, but the gain is not
large enough when the mixed path still computes sparse base plus dense
correction separately.

The bucket16/direct-cslt candidate passed a small paired GSM8K sanity gate:
dense and `speclink_t08` both scored `0.7000` on 20 samples, with pair reg/imp
`0/0`. Treat this as a smoke-quality signal only; it is not enough to declare
quality solved.

Current conclusion: the best scoped graph-safe candidate is now
`critical_prefix@0.6,prefix4,extra1`, bonus non-draft correction,
`gate_up_proj=16-31;down_proj=8-15`, bucket16, dense bucket copy, and direct
cuSPARSELt. It should be used as the next comparison point, but completing the
goal still requires either a fused/packed mixed operator or a stronger
quality-gated residual-row reduction to move from about `1.17x` to at least
`1.2x`.

## Compile-Aligned Bucket16 Follow-Up, 2026-06-28

I reran the current bucket16/direct-cuSPARSELt candidate with a stricter
compile-aligned quality gate and two throughput checks. This separates the
accuracy question from the speed question.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_directcslt_bucket16_quality_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket16_directcslt_compilealigned_throughput_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket16_directcslt_srcompile_throughput_bs64_math256_20260628/report.md
```

The compile-aligned GSM8K-50 paired gate passed:

| mode | exact match | pair reg | pair imp | avg output tokens |
| --- | ---: | ---: | ---: | ---: |
| dense EAGLE3 | `0.7200` | `0` | `0` | `91.32` |
| `speclink_t08` | `0.7200` | `0` | `0` | `91.32` |

However, the same candidate is still not a `1.2x` speed path:

| compile mode | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full-batch speedup | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default vLLM compile aligned with quality gate | `2787.350` | `2749.232` | `0.986x` | `3140.273` | `3131.924` | `0.997x` | `1.747` | `{"FULL":44,"PIECEWISE":44}` |
| SR24 `FULL_DECODE_ONLY` compile | `2776.368` | `2905.794` | `1.047x` | `3150.364` | `3477.343` | `1.104x` | `2.161` | `{"FULL":49}` |

Read: compile alignment removes the small-sample paired accuracy concern for
this candidate, but it does not solve the throughput target. The default vLLM
compile path is quality-clean but speed-neutral. The SR24-specific
`FULL_DECODE_ONLY` compile path restores part of the speedup, yet remains below
the requested `1.2x`. The remaining work is therefore not another accuracy
sanity or bucket-size sweep; it is reducing residual rows further under a
paired gate, or replacing the two-pass sparse-base plus dense-row correction
with a fused/packed GPU operator.

## Compile / Correction Ablations, 2026-06-28 Follow-Up

I tested two narrower explanations for the slowdown after the bucket16 result:

1. maybe the fast path only needs a larger quality-safe CUDA Graph capture;
2. maybe bucket dense correction is slow because of PyTorch gather/GEMM/scatter
   overhead and should use the Triton bucket kernel.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_vllmcompile_largegraph_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_vllmcompile_largegraph_throughput_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_triton_bucket_dense_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_triton_bucket_dense_throughput_bs64_math256_20260628/report.md
```

Quality checks:

| candidate | dense exact | SR24 exact | pair reg | pair imp | read |
| --- | ---: | ---: | ---: | ---: | --- |
| VLLM_COMPILE large graph | `0.7000` | `0.7667` | `0` | `2` | no regression on GSM8K-30 |
| Triton bucket dense | `0.7333` | `0.7333` | `0` | `0` | exact paired match on GSM8K-30 |

Throughput checks on Llama-3.1-8B, `math_reasoning`, batch/concurrency 64,
128 total requests, max new tokens 256:

| candidate | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | SR24 accepted draft/step | SR24 GPU util | SR24 graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default compile bucket16 reference | `2595.090` | `2714.458` | `1.046x` | `3135.954` | `3112.682` | `0.993x` | `1.718` | `94.71%` | `{"FULL":44,"PIECEWISE":44}` |
| VLLM_COMPILE large graph | `2574.912` | `2425.580` | `0.942x` | `2995.243` | `3036.585` | `1.014x` | `1.742` | `83.63%` | `{"FULL":49,"PIECEWISE":76}` |
| Triton bucket dense | `2790.161` | `2733.488` | `0.980x` | `3150.925` | `3122.393` | `0.991x` | `1.738` | `92.75%` | `{"FULL":44,"PIECEWISE":44}` |
| raw FULL_DECODE_ONLY | `2776.368` | `2905.794` | `1.047x` | `3150.364` | `3477.343` | `1.104x` | `2.161` | `93.91%` | `{"FULL":49}` |

Read:

- `VLLM_COMPILE + FULL_AND_PIECEWISE` is quality-safe, but it is not a speed
  path here. It increases graph shape diversity and dropped SR24 GPU
  utilization in this run.
- Triton bucket dense correction is quality-clean, but it does not improve
  end-to-end throughput. That makes bucket writeback/gather/scatter a secondary
  cost in the current configuration.
- The only row with a noticeable full-batch gain is still raw
  `FULL_DECODE_ONLY`, but that path has shown quality drift in the stricter
  setting and cannot be treated as the valid candidate yet.

The current bottleneck diagnosis is therefore unchanged and sharper: SR24 is
slow because it still launches useful GPU work twice for protected rows. It
computes sparse base output for the full row set and then computes dense
correction for a substantial residual subset. The next useful work is either
debugging the raw full-graph quality drift, or implementing a real packed/fused
mixed operator that avoids doing sparse-base work for rows that will be
overwritten by dense correction.

## Raw Full-Graph Bucket Copy Follow-Up, 2026-06-28

I tested the fastest-looking path more directly: `criticalprefix4_bucket16`
with direct cuSPARSELt, raw `FULL_DECODE_ONLY`, and bucket dense-copy. The
important finding is that the quality-safe variant is not active-only bucket
copy. On GSM8K-30, preserving only `bucket_values != 0` rows caused paired
regressions, while the original dense-copy semantics that overwrites all bucket
rows passed.

Quality artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_quality_gsm8k30_20260628_rerun/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_activeonly_scatter_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_activeonly_triton_quality_gsm8k30_20260628/report.md
```

| variant | dense exact | SR24 exact | pair reg | pair imp | read |
| --- | ---: | ---: | ---: | ---: | --- |
| raw full, dense-copy all bucket rows | `0.7000` | `0.7000` | `0` | `0` | quality-clean smoke |
| active-only bucket scatter | `0.7000` | `0.6667` | `2` | `1` | unsafe |
| Triton bucket dense GEMM | `0.7000` | `0.6667` | `2` | `1` | unsafe, likely custom-GEMM numerical drift |

Throughput artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_throughput_bs64_math256_20260628_rerun/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_rawfull_activeonly_triton_throughput_bs64_math256_20260628/report.md
```

| variant | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw full, dense-copy all bucket rows | `2729.709` | `2967.505` | `1.087x` | `3156.454` | `3501.694` | `1.109x` | clean on GSM8K-30 |
| Triton bucket dense GEMM | `2487.670` | `3021.605` | `1.215x` | `3093.648` | `3528.235` | `1.140x` | unsafe on GSM8K-30 |

Read: overwriting all selected bucket rows with dense output appears to be a
quality guard, not just wasted work. The custom Triton dense GEMM improves
throughput but changes enough numerics to break paired GSM8K. Scatter-only
active-row writeback also breaks the paired gate because it removes that extra
dense protection. The current clean raw-full path improves absolute SR24
throughput to about `2967 output tok/s` on this bs64 math smoke, but the
same-run dense baseline can still be too fast for a stable `1.2x` claim.

The next optimization should therefore not be active-only bucket pruning. The
better target is a quality-preserving packed operator that keeps the dense-copy
all-bucket semantics where needed, or a route policy that reduces bucket rows
only under a paired quality gate.
