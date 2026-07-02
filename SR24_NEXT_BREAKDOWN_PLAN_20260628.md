# SR24 Next Breakdown Plan, 2026-06-28

This is the short current pivot for SR24/SpecLink optimization. It should be
read before running another controller sweep.

Primary current artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_corrected_fast_candidate_breakdown_b12_bs64_math_k8_20260628/seven_part_report/report.md
```

Run shape: Llama-3.1-8B, `math_reasoning`, client concurrency 64, EAGLE3 K=8,
max new tokens 256, `speclink_t08`, target leaves
`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, bucket12, direct
cuSPARSELt, bucket dense copy, dynamic auto CUDA Graph.

## Current Answer

Current SR24 is slow because the mixed verification path has poor useful-work
efficiency. The clean serving path is not primarily blocked by accepted length,
ordinary CPU mask building, CUDA Graph loss, or idle GPU.

The sparse upper bound exists:

| method | full-batch tok/s | total tok/s | accepted draft/step | GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3524.307` | `2599.815` | `1.736` | `89.385%` | baseline |
| `base_only_24` | `4307.108` | `2955.659` | `2.205` | `89.364%` | sparse headroom exists |
| `speclink_t08` | `3902.787` | `2625.737` | `2.103` | `91.154%` | only `1.107x` full-batch and `1.010x` total |

The mixed path pays sparse base Linear over many rows, then pays dense-row
correction for a large protected subset. That recovers only a small part of the
`base_only_24` headroom.

## Seven-Part Breakdown

| part | current measurement | bottleneck read | next measurement rule |
| --- | --- | --- | --- |
| scheduler / mask build | clean `0.289ms/step`; row bucket `0.064ms/step`; bucket build `0.062ms/step` | not the first bottleneck in clean serving | keep reporting clean wall time; ignore sync-heavy diagnostic tok/s |
| base sparse linear | diagnostic base sparse `1.041ms/call`; gate/up16-31 `1.071ms/call`; down8-15 `0.980ms/call` | largest localized GPU-side cost | report per-leaf sparse base time for every candidate |
| residual correction | diagnostic dense correction `0.161ms/call`; gate/up `0.180ms/call`; down `0.123ms/call`; bucket rows/call `12` | additive work that erases sparse savings when residual rows are common | report dense correction time and corrected rows/call |
| gather/scatter | diagnostic `0.014ms/call` in the bucket12 path | secondary in this row | keep as guardrail; do not optimize it first unless it grows |
| routing statistics | draft residual/base `2928/2464`; non-draft residual/base `674/891`; draft residual fraction `0.543`; bucket fill `0.989` | many draft rows still need residual protection | report draft/non-draft residual fractions and bucket fill for each run |
| CUDA Graph | clean `speclink_t08` `{"FULL":126,"NONE":2}` | graph coverage is healthy here | treat high `NONE` fraction as a hard reject for new variants |
| GPU util | clean avg `91.154%`, peak `99%` | GPU is busy; the issue is inefficient busy work | use GPU util to distinguish underfilled-kernel regressions from useful-work regressions |

## Implication

The next optimization should not be a blind threshold sweep. Each new candidate
must pass a breakdown gate:

1. clean serving throughput, accepted length, GPU util, and CUDA Graph modes;
2. instrumented sparse-base, residual-correction, gather/scatter timing;
3. routing stats: draft residual rows, non-draft residual rows, bucket fill;
4. paired accuracy gate before considering throughput wins valid.

The likely speed paths are:

1. reduce residual rows with a quality-safe token/request controller; or
2. replace the current two-pass sparse-base plus dense correction with a
   graph-safe fused/packed mixed operator.

The lower-priority paths are:

1. generic CPU-side cleanup, because the current clean mask path is sub-ms;
2. gather/scatter-only rewrites, because current gather/scatter is much smaller
   than sparse base and correction;
3. coarse all/no-residual batch gating, because bs64 quickly collapses to
   all-residual and loses sparse benefit.

## Route-Bucket Check

I also checked the route-bucket / row-routed direction because it is the most
direct way to avoid sparse-base work on rows that will be dense-overwritten.
The existing PyTorch split route is not yet the fused operator we need: it
reduces some redundant sparse work, but introduces dense/sparse gathers,
separate GEMMs, and assembly overhead. Prior bs64/math gates showed it was
paired-clean but still below the `1.2x` target:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_routebucket_rawfull_quality_gsm8k30_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix4_routebucket_rawfull_throughput_bs64_math256_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routebucket_cached_graphon_allow_bs64_math256_20260628/report.md
```

Representative results:

| route-bucket variant | dense full tok/s | route full tok/s | dense total tok/s | route total tok/s | read |
| --- | ---: | ---: | ---: | ---: | --- |
| rawfull | `3128.402` | `3393.500` | `2482.458` | `2810.343` | paired-clean, `1.085x` full / `1.132x` total |
| cached graph allow | `3525.085` | `3671.220` | `2603.175` | `2522.459` | graph-friendly but total slower |

While auditing this path, I fixed two consistency issues so future route-bucket
runs measure the intended conservative semantics:

1. `_routed_bucket_dense_rows_output()` now treats
   `bucket_dense_copy=True` and `bucket_dense_copy_active_only=False` as
   "dense all selected bucket rows", matching the normal bucket dense-copy
   path.
2. `row_routed_mlp_output()` now applies the same rule in its fallback
   residual-bucket path; before this, it always used only
   `bucket_values=True` rows.

Validation after the patch:

```text
conda run -n spec python -m py_compile \
  vllm/vllm/speclink_sr24.py \
  vllm/vllm/model_executor/models/llama.py

conda run -n spec python \
  examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Result:

```text
speclink_sr24_correctness=ok
```

Conclusion: route-bucket remains a useful operator-design direction, but the
current PyTorch split path is not enough. The next useful operator work is a
graph-safe fused/packed mixed kernel that avoids computing sparse base for
dense-overwritten rows without paying large gather/assembly overhead.

## 2026-06-28 Triton Bucket Semantics Fix

While checking the correction-only Triton bucket path, I found one concrete
inconsistency. `sparse_linear_output()` already called
`_triton_bucket_dense_gemm_scatter_inplace(..., force_all_bucket_rows=True)`
when `bucket_dense_copy=True` and `bucket_dense_copy_active_only=False`, which
matches the quality-safe `index_copy_` fallback: every selected bucket row is
overwritten with the dense result. `residual_linear_output()` used the same
Triton helper without this flag, so that path could silently behave like
active-only correction under the same external flags.

The local fix is in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm/vllm/speclink_sr24.py
```

It makes `residual_linear_output()` pass the same force-all condition:

```text
force_all_bucket_rows=(bucket_dense_copy() and not bucket_dense_copy_active_only())
```

Validation completed:

```text
conda run -n spec python -m py_compile \
  vllm/vllm/speclink_sr24.py \
  vllm/vllm/model_executor/models/llama.py \
  examples/evaluate/eval-guidellm/scripts/summarize_sr24_breakdown.py \
  examples/evaluate/eval-guidellm/scripts/make_sr24_seven_part_breakdown.py

conda run -n spec python \
  examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Result:

```text
speclink_sr24_correctness=ok
```

I also ran a direct GPU semantic check on the Triton helper. Active-only exactly
matched active-row overwrite, and force-all matched conservative full-bucket
overwrite with max fp16 difference `1.52587890625e-05`. This does not prove an
end-to-end speedup; it removes a semantic mismatch so the next Triton bucket
throughput/accuracy run is measuring the intended quality-safe behavior.

Follow-up gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_forceall_quality_gsm8k30_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_forceall_throughput_bs64_math256_20260628_goal/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_forceall_eager_breakdown_summary_20260628_goal/report.md
```

The force-all Triton bucket path is now paired-clean on the GSM8K-30 gate:

| mode | exact | pair reg | pair imp | read |
| --- | ---: | ---: | ---: | --- |
| dense EAGLE3 | `0.7000` | `0` | `0` | reference |
| `speclink_t08` + force-all Triton bucket | `0.7667` | `0` | `2` | no paired regression in this small gate |

The bs64/math throughput gate is still below target:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2806.159` | `3182.718` | `1.754` | `92.783%` | same-run reference |
| `base_only_24` | `3139.812` | `3827.714` | `2.220` | `93.524%` | sparse upper bound still exists |
| `speclink_t08` + force-all Triton bucket | `2924.692` | `3523.724` | `2.195` | `93.818%` | `1.042x` total and `1.107x` full-batch vs dense |

The short eager component breakdown explains why this is not the final speed
path:

| component | value | read |
| --- | ---: | --- |
| base sparse Linear | `1.013ms/call` aggregate | still dominates |
| gate/up16-31 sparse base | `1.123ms/call` | main localized cost |
| down8-15 sparse base | `0.794ms/call` | smaller but still large |
| force-all Triton dense GEMM/scatter | `0.184ms/call`, `16` rows/call | quality-safe correction, but additive |
| routing | draft residual/base `10904/9744`, non-draft `2581/3788`, bucket fill `0.987` | many rows still need correction |

Conclusion: the semantic fix makes the Triton bucket correction path valid to
measure, but correction-only Triton is not enough to reach `1.2x`. The base
sparse pass over about `528` rows/call remains the dominant cost, and the dense
correction still adds work for a high-fill 16-row bucket. Treat this as a
validated negative ablation for the main speed path. The next operator work
should avoid computing sparse base for rows that will be dense-overwritten, or
fuse/pack the mixed base/correction path more deeply than a correction-only
Triton kernel.

## 2026-06-28 Pivot: Diagnose First

I reran a focused slowdown breakdown before doing another controller sweep. The
main artifact is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_pivot_bs64_math_k8_20260628/seven_part_report_with_runtime/report.md
```

Supporting roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_pivot_bs64_math_k8_20260628/clean_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_pivot_bs64_math_k8_20260628/clean_runtime_stats
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_pivot_bs64_math_k8_20260628/instrumented_serving
```

The clean serving row should be used for real throughput, GPU utilization, and
CUDA Graph behavior. The instrumented row should only be used for local
operator timing because CUDA events and exact routing counters add
synchronization overhead.

### Current Slowdown Breakdown

| part | measured value | read |
| --- | --- | --- |
| scheduler / mask build | clean runtime stats: `0.378ms/step`; batched builder `0.197ms/step`; row bucket `0.110ms/step`; residual bucket `0.107ms/step` | not the primary bottleneck in clean serving |
| base sparse linear | instrumented `speclink_t08`: aggregate `0.986ms/call`; `gate_up_proj` layers 16-31 `1.101ms/call`; `down_proj` layers 8-15 `0.756ms/call` | largest localized GPU-side cost |
| residual correction | instrumented `speclink_t08`: aggregate dense correction `0.164ms/call`; `gate_up_proj` `0.180ms/call`; `down_proj` `0.131ms/call`; bucket rows/call `16` | secondary, but additive on top of the sparse base pass |
| gather/scatter | instrumented `speclink_t08`: input gather `0.016ms/call`; bucket dense copy `0.005ms/call` | not the first bottleneck |
| routing stats | instrumented exact counts: draft residual/base `12712/12712`; non-draft residual/base `3178/3801`; bucket fill `0.976` | about half of draft rows still require dense protection; bucket is nearly full |
| CUDA Graph | clean runtime stats: `{"FULL": 78, "NONE": 2}`; clean serving graph profile `{"FULL": 49}` | graph coverage is healthy in the clean path |
| GPU util | clean serving `speclink_t08`: avg `93.727%`, peak `100%`; full-batch `3502.524 tok/s`; total `2915.708 tok/s` | GPU is busy; slowdown is inefficient useful work, not idle GPU |

Clean serving comparison:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2436.970` | `3186.365` | `1.745` | `87.963%` | baseline |
| `base_only_24` | `3177.714` | `3811.771` | `2.212` | `94.000%` | sparse upper bound is real |
| `speclink_t08` | `2915.708` | `3502.524` | `2.200` | `93.727%` | only `1.099x` full-batch vs dense |

### Updated Read

The current path is slow because it is doing too much useful-looking but
duplicated GPU work:

1. `speclink_t08` computes sparse base Linear over the full verification row
   set.
2. It then computes dense correction for a high-fill 16-row bucket.
3. Gather/scatter and clean scheduler mask build are small compared with the
   sparse base pass.
4. CUDA Graph coverage and GPU utilization are good enough that the first
   optimization target should not be Python scheduling or graph capture.

The next optimization should therefore be operator-side:

1. avoid sparse base computation for rows that will be dense-overwritten; or
2. fuse/pack the mixed sparse-base plus dense-correction path so the protected
   rows do not pay both paths.

Any future candidate should be rejected early unless this seven-part breakdown
shows either lower sparse-base work, lower duplicated correction work, or a
clear routing reduction without losing accuracy.

## 2026-06-28 CPU-Sync Ablation

Following the suggestion to check CPU-side synchronization, I ran a focused
`speclink_t08` CPU-sync ablation with the same bs64/math/K8/manual SR24 shape.
The artifact is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_speclink_t08_bs64_math_k8_20260628/seven_part_report/report.md
```

Per-variant roots are under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_speclink_t08_bs64_math_k8_20260628/cpu_sync_ablation/
```

Result:

| variant | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| low-sync stats on | `2926.467` | `3499.992` | `2.206` | `93.727%` | normal low-sync path |
| low-sync stats off | `2888.125` | `3451.313` | `2.156` | `93.818%` | stats collection is not a major cost |
| sync mask state | `2913.795` | `3459.470` | `2.219` | `92.727%` | one coarse mask-state sync does not explain current gap |
| sync-heavy | `1557.782` | `1825.806` | `1.884` | `61.659%` | many CPU/GPU syncs destroy throughput |
| low-sync GPU counts | `2941.868` | `3435.240` | `2.179` | `92.273%` | GPU-side diagnostic counters are acceptable for focused runs |

This answers the CPU-sync question:

1. CPU/GPU synchronization can absolutely make SR24 slow, as shown by the
   `sync_heavy` row.
2. The current clean low-sync `speclink_t08` path is not in that regime:
   throughput stays near `2.9k total tok/s`, GPU utilization stays above `92%`,
   and stats-on/off are within normal run variance.
3. CPU-sync cleanup remains useful as a guardrail, but it is not the path to
   the required `1.2x` over dense baseline.

Updated optimization priority:

1. Keep `reduce_cpu_sync=True`, `sync_mask_state=False`, and static graph-safe
   mask/bucket buffers for clean serving.
2. Use GPU-count diagnostics only for focused breakdown runs, not full
   throughput matrices.
3. Move the main speed work to the mixed Linear operator: avoid sparse-base
   computation on rows that will be dense-overwritten, or fuse/pack sparse base
   plus dense correction.
