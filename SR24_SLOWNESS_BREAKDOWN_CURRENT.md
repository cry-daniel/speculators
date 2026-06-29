# SR24 Slowness Breakdown, Current Read

## 2026-06-29 Adaptive Dense Fallback Probe

Focused root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_adaptive_dense_fallback_probe_bs64_math128_20260629
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 128, current `criticalprefix4_bucket16_directcslt` SR24 preset.
The only intentional change from the current quality-safe candidate was
`--sr24-adaptive-dense-fallback` with gate/up fraction `0.25` and down fraction
`0.50`. This checks whether replacing high-correction mixed steps with dense
Linear can avoid the duplicated sparse-base plus dense-correction work.

Clean serving result:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3040.780 | 2273.383 | 1.426 | 84.5% | `FULL=20,NONE=43,PIECEWISE=1` |
| `base_only_24` | 3435.842 | 2267.445 | 1.630 | 83.4% | `FULL=62,NONE=2` |
| `speclink_t08` + adaptive dense fallback | 2880.598 | 1989.909 | 1.402 | 86.3% | `FULL=62,NONE=2` |

Read: coarse adaptive dense fallback is a negative path for the current
quality-safe preset. It keeps CUDA Graph coverage and GPU utilization healthy,
but lowers accepted draft length and throughput. This means the next useful
operator work is not a broad "fallback high-residual steps to dense" policy.
It should either use a real fused/grouped mixed operator, or a much more
selective planner that only chooses dense for shapes proven faster by
microbench and live serving.

Diagnostic follow-up:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_adaptive_dense_fallback_probe_bs64_math128_20260629_instrumented
```

The instrumented `speclink_t08` row records `adaptive_dense_fallback_calls=960`:
`640` gate/up calls and `320` down calls. Most were triggered by the small-row
guard (`small_gate_up_proj=624`, `small_down_proj=312`), not by a precise
large-residual planner. The same diagnostic row reports `11.334 ms/step`
scheduler/mask time from exact routing instrumentation, so its tok/s is not a
serving metric; it is useful because it proves the coarse fallback is firing
widely. Do not use adaptive dense fallback as the default path unless a future
planner narrows it to measured-profitable shapes and re-passes the clean
serving and accuracy gates.

Code follow-up: the adaptive planner now uses the actual executed bucket-row
count for capped buckets instead of treating a full capped bucket as if every
row needed residual correction. The default
`SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_ROWS` was also changed from `128`
to `0`, so the small-row fallback rule is disabled unless explicitly requested.
Validation roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_adaptive_dense_fallback_bucketcount_probe_bs64_math128_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noadaptive_compare_bs64_math128_20260629
```

With the bucket-count planner, `speclink_t08` recovered from the bad
`2880.598` full-batch tok/s row to `3157.920`, and accepted draft/step recovered
from `1.402` to `1.650`. However, the same-condition no-adaptive control is
still the cleaner default: it reaches `3152.707` full-batch tok/s against
same-root dense `3027.578` (`1.041x`), while the bucket-count fallback row is
effectively dense parity (`3157.920` against same-root dense `3159.269`).
Therefore the code change is a guardrail against pathological fallback, not a
new speed path toward `1.2x`.

## 2026-06-29 Focused Breakdown Requested By User

Combined seven-part report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_combined_bs64_math_20260629/report.md
```

Raw roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_current_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_breakdown_baseonly_eager_bs64_math_20260629
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 256 for clean serving rows, max new tokens 128 for instrumented
diagnostic rows. `speclink_t08` uses `gate_up_proj=16-31`,
`critical_prefix`, bucket size 4 scaled by active requests, `bucket_dense_copy`,
direct cuSPARSELt base, and graph-bucket active hint. Diagnostic rows are
eager-only with CUDA events and exact routing counters, so their tok/s is not
the clean throughput reference.

| method | row kind | full-batch tok/s | total tok/s | avg GPU util | CUDA Graph |
| --- | --- | ---: | ---: | ---: | --- |
| dense baseline | clean | 3482.460 | 2335.593 | 88.429% | `FULL=115,NONE=76,PIECEWISE=1` |
| `base_only_24` | clean eager-safe | 3892.913 | 2578.924 | 84.846% | `NONE=128` |
| `speclink_t08` | clean graph-bucket | 3492.301 | 2287.666 | 91.643% | `FULL=114,NONE=77,PIECEWISE=1` |
| `speclink_t08` | diagnostic | n/a | 933.089 | 60.444% | `NONE=79` |
| `all_corrected_24` | diagnostic | n/a | 1171.217 | 76.143% | `NONE=79` |

Seven-part read:

| part | current measurement | read |
| --- | --- | --- |
| scheduler / mask build | diagnostic `speclink_t08` reports `7.924ms/step`, mostly exact-routing request loop `7.675ms`; `all_corrected_24` mask is `0.028ms/step` | the clean low-sync serving path should not be treated as scheduler-bound; exact routing diagnostics intentionally add sync overhead |
| base sparse linear | `speclink_t08` gate_up layers 16-31 sparse base `1.040ms/call`, about 233 rows/call | largest localized GPU-side component in the mixed path |
| residual correction | `speclink_t08` dense correction `0.191ms/call`, about 94 bucket rows/call; `all_corrected_24` correction `0.581ms/call` | correction is secondary in selective t08, but dominates when every row is corrected |
| gather/scatter | `speclink_t08` `0.015ms/call`; `all_corrected_24` `0.035ms/call` | not the first bottleneck |
| routing statistics | `speclink_t08` draft residual/base `6420/7500`, non-draft residual/base `1740/1824`, bucket fill `0.982`, avg active bucket rows `91.6` | buckets are full; the issue is the amount/shape of mixed operator work, not empty buckets |
| CUDA Graph | dense `FULL=115,NONE=76`; base-only eager-safe `NONE=128`; graph-bucket `speclink_t08` `FULL=114,NONE=77` | graph-bucket avoids all-NONE for t08, but base-only graph/compile path is still unstable and failed under default compile |
| GPU util | clean `speclink_t08` avg `91.643%`, peak `99%`; base-only avg `84.846%` | this is not primarily idle GPU; improve useful-work efficiency |

Operator microbench agrees with the serving read. The isolated sparse base is
substantially faster than dense (`0.57x-0.66x` of dense graph time), but the
current mixed proxy only wins when the residual fraction is very small. For
Llama gate/up shape, residual fraction 0.125 already makes the mixed proxy
`1.03x` of dense; at 0.25 it is `1.15x`, and at 0.5 it is `1.54x`. Therefore
the current path only has a real speed margin if residual rows are kept very
low or if sparse-base and dense correction are fused/grouped so correction does
not erase the 2:4 base advantage.

Current decision:

1. Do not spend the next pass on `index_select`/`index_add_` first; those are
   much smaller than base sparse and dense correction.
2. Do not assume CUDA Graph alone solves the slowdown. `speclink_t08` now gets
   partial FULL coverage, yet clean total tok/s is still below base-only and
   close to dense.
3. The next optimization should target useful work: either reduce protected
   residual rows while preserving accuracy, or implement a fused/grouped
   mixed operator that avoids paying full sparse base plus dense correction as
   separate work.
4. `base_only_24` with default compile/graph failed with a torch compile lazy
   tensor/custom-kernel error; the eager-safe row is usable as an upper-bound
   throughput reference, but graph-safe base-only needs a separate correctness
   fix before it can be used as the main serving path.

### 2026-06-29 all_corrected_24 Backend Ablation

Focused roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_backend_dense_rows_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_backend_torch_sparse_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_backend_compressed_dense_bs64_math_20260629
```

Setup: same Llama-3.1-8B `math_reasoning` bs64/K8/max256 serving setup as the
focused breakdown, but only `dense_baseline` and `all_corrected_24` were run.
`all_corrected_dense_fastpath` was disabled so the rows measure real
base-sparse plus residual correction, not a dense-equivalent no-op control.

| residual backend | dense total tok/s | all_corrected total tok/s | dense full-batch tok/s | all_corrected full-batch tok/s | avg GPU util | CUDA Graph | storage/dense |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `dense_rows` | 2336.592 | 2069.642 | 3485.192 | 2935.591 | 86.313% | `NONE=140` | 1.625 |
| `torch_sparse` | 2355.788 | 2286.517 | 3558.700 | 3055.136 | 91.071% | `FULL=136,NONE=2` | 1.188 |
| `compressed_dense` | 2355.180 | 2255.007 | 3553.472 | 3008.510 | 89.667% | `FULL=131,NONE=2` | 1.125 |

`compressed_dense` was GPU-resident in this run:
`sr24_residual_device_counts={"cuda:0": 16}`,
`sr24_compressed_residual_runtime_on_gpu=True`, and no non-GPU compressed
modules were reported. So the current loss is not a CPU materialization bug in
this narrow setup. The issue is operator shape: even with graph capture and GPU
resident residuals, all-corrected still spends extra work on sparse base plus a
second residual/correction path.

The best current exact all-corrected backend is `torch_sparse` residual. It
improves total tok/s by about `10.5%` over `dense_rows`
(`2286.517/2069.642`) and is slightly faster than GPU `compressed_dense`
(`2286.517/2255.007`), while using less storage than `dense_rows`. However it
still reaches only about `97.1%` of the same-root dense total tok/s and about
`85.8%` of same-root dense full-batch tok/s.

Related component microbench result from the combined breakdown:

- current Triton compressed residual matmul is slower than the GPU cached dense
  residual path for the tested Llama gate/up and down shapes;
- `compressed_delta_sparse` was the best isolated residual-correction direction
  for the gate/up shape, which matches the serving result that
  `torch_sparse` residual is the strongest all-corrected backend currently.

Decision for all-corrected optimization:

1. Use `torch_sparse` residual as the current exact all-corrected operator
   ablation baseline when it is explicitly requested with
   `--sr24-residual-backend torch_sparse`. It should still not be an implicit
   default because it increases model storage to about `1.1875x` dense and is
   not faster than dense in serving, but the attach-time OOM path has now been
   fixed as described below.
2. Do not promote the current compressed-residual Triton kernel; it is a
   negative result until the kernel is rewritten around a better data layout.
3. A real >dense all-corrected speedup likely requires either a fused two-2:4
   operator, grouped sparse base plus residual execution, or an early-dense
   bypass only as a dense-equivalent control. The existing separate base sparse
   GEMM plus residual GEMM path is not enough.

### 2026-06-29 Update: torch_sparse Residual Attach Memory Fix

Validation root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_torch_sparse_residual_attach_fix_20260629
```

The `torch_sparse` residual attach path used to extract half-dense
`residual_values`, then allocate a full `residual_dense` tensor and scatter
those values into it before converting to a second semi-structured tensor. A
bs16 full-model startup gate hit OOM in that staging path on the 32GB GPU.

The code now clones the original dense weight into `residual_dense` before
masking the module weight into the base 2:4 tensor, zeros the kept entries in
that residual clone, then replaces the module weight and converts the residual
clone to `SparseSemiStructuredTensor`. This removes the extra half-dense
staging allocation and keeps residual construction on GPU.

Sanity evidence:

| check | result |
| --- | --- |
| full-model `all_corrected_24`, bs16, max32 | `status=ok`, peak GPU memory `23814 MiB` |
| residual backend | `torch_sparse`, `128` attached modules, dense fastpath `false` |
| storage/dense | `1.1875` |
| graph modes in gate | `FULL=23,NONE=2`; server profile `FULL=16` |
| gate_up real-shape equivalence | fp16 loose close, `max_abs=0.00390625`, `mean_abs=0.00028110` |
| down real-shape equivalence | fp16 loose close, `max_abs=0.00781250`, `mean_abs=0.00053501` |

The equivalence checks are not bitwise exact because the exact all-corrected
path does two sparse GEMMs plus an add, while the reference dense module does
one dense GEMM. The remaining differences are fp16 accumulation-order noise,
not the previous base-only residual construction bug.

This fix improves robustness and makes the exact all-corrected ablation easier
to run. It does not change the throughput conclusion: two separate sparse
GEMMs plus add still do not beat the dense verifier path for the measured
Llama serving shapes.

## 2026-06-29 Update: Graph-Bucket Active-Hint Result

Current consolidated report with the graph-bucket run included:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_with_graphbucket_20260629/report.md
```

Graph-bucket run:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_graphbucket_bs64_math256_20260629/report.md
```

The active-hint graph bucket fixes the earlier all-`NONE` CUDA Graph profile
for the scaled-bucket `speclink_t08` path. For Llama-3.1-8B,
`math_reasoning`, bs64, K=4, max new tokens 256:

| method | full-batch tok/s | total tok/s | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline, same graph-bucket root | 5248.020 | 3754.224 | 82.000% | 1.4940 | `FULL=126,NONE=1,PIECEWISE=1` |
| `speclink_t08`, old graph-off clean row | 3236.359 | 2278.722 | 87.214% | n/a | `NONE=128` |
| `speclink_t08`, graph-bucket active hint | 4537.540 | 3296.182 | 87.700% | 1.5555 | `FULL=126,NONE=2` |

This changes the bottleneck read. CUDA Graph miss was a real loss and the
active-hint bucket recovers a large part of it, but graph-on `speclink_t08` is
still only `0.865x` of the same-run dense baseline. The remaining slowdown is
therefore primarily the mixed sparse-base plus dense-row correction operator
and the amount of corrected-row work, not scheduler/mask Python overhead and
not a completely idle GPU.

The next useful work should target:

1. reducing sparse-base work that will be overwritten by dense correction;
2. fusing/grouping sparse base and residual correction to avoid fragmented
   small kernels;
3. reducing residual rows only when the paired quality check stays clean.

### 2026-06-29 Triton Bucket Dense GEMM Check

Focused ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tritonbucketdense_bs64_math256_20260629/report.md
```

This run keeps the same Llama-3.1-8B `math_reasoning` bs64/K4/max256 setup as
the graph-bucket run, but adds `--sr24-triton-bucket-dense-gemm` for the bucket
dense correction path.

| method | full-batch tok/s | total tok/s | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| graph-bucket `speclink_t08` without Triton bucket GEMM | 4537.540 | 3296.182 | 87.700% | 1.5555 | `FULL=126,NONE=2` |
| graph-bucket `speclink_t08` with Triton bucket GEMM | 3786.125 | 2576.055 | 91.846% | 1.5460 | `FULL=158,NONE=2` |

The custom bucket dense GEMM/scatter path increases GPU utilization but lowers
useful output throughput. Do not promote this path as the main optimization.
The bottleneck is still the amount and shape of mixed sparse-base plus dense
correction work; replacing only the dense correction scatter with this Triton
kernel is not enough.

### 2026-06-29 Route-Bucket Recheck After Graph-Bucket Fix

Focused ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routebucket_graphbucket_bs64_math256_20260629/report.md
```

This run rechecks the earlier `route_bucket_rows` idea after the active-hint
CUDA Graph bucket fix. It avoids doing sparse base on selected bucket rows by
routing bucket rows through dense and the complement through sparse base, with
Triton route assembly enabled.

| method | full-batch tok/s | total tok/s | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| graph-bucket `speclink_t08` without route bucket | 4537.540 | 3296.182 | 87.700% | 1.5555 | `FULL=126,NONE=2` |
| graph-bucket `speclink_t08` with route bucket | 4008.616 | 2952.255 | 72.000% | 1.5022 | `FULL=126,NONE=2` |

The graph issue is fixed for this path too, but route-bucket still loses
throughput and drops GPU utilization. So the simple row-routed form does avoid
some duplicate sparse-base work, but it replaces it with smaller dense/sparse
GEMMs and assembly that underutilize the GPU. This reinforces the fused/grouped
operator requirement: skipping sparse base for corrected rows only helps if the
replacement does not fragment the batch shape.

Safety follow-up after inspecting the code path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routebucket_graph_guard_smoke_bs16_math64_20260629/report.md
```

`route_bucket_rows + cudagraph_bucket` now sets
`sr24_route_bucket_rows_graph_static_unsafe=True` and forces SR24 mixed steps
to CUDA Graph `NONE`. The reason is semantic, not only performance: graph-static
buckets carry inactive padding through `bucket_values`, while cached route-row
plans only carry row ids. Treating the whole padded bucket as route rows can
remove padded duplicate rows from the sparse/base complement or correct extra
rows not selected by the active mask. The smoke confirmed the guard with
`sr24_cudagraph_mode_counts={"NONE": 32}`. This path should stay
diagnostic-only unless a future implementation makes the route plan consume
`bucket_values` or provides a fixed-shape active-row compaction.

## 2026-06-29 Update: User Seven-Part Breakdown With Route-Bucket Ablation

Current consolidated report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_with_routebucket_20260629/report.md
```

This report is the current reference for answering "why is SpecLink/SR24
slow?" before another selector sweep. It joins:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_criticalprefix_scaledbucket_20260629/clean_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_criticalprefix_scaledbucket_20260629/instrumented_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routebucket_criticalprefix_scaledbucket_bs64_math256_20260629
```

### Seven-Part Read

| part | current measurement | read |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` `0.537ms/step`; exact diagnostic `7.951ms/step` due sync-heavy request loop | clean serving path is not primarily scheduler-bound |
| base sparse linear | `gate_up_proj=16-31` sparse base `1.052ms/call`, about `229` rows/call | largest measured GPU-side component |
| residual correction | dense correction `0.192ms/call`, about `93` rows/call | secondary to sparse base, but still extra work |
| gather/scatter | `0.015ms/call` | not the first bottleneck |
| routing statistics | draft residual/base `6348/7404`, non-draft residual/base `1719/1732`, bucket fill `0.983` | buckets are mostly filled; cost is executing the mixed operator |
| CUDA Graph | clean `speclink_t08` `NONE=128`, `FULL=0` | graph miss is a real serving-side loss |
| GPU util | clean `speclink_t08` avg `87.214%`; route-bucket ablation avg `44.000%` | normal path is busy but inefficient; route-bucket fragments into underutilized work |

### Decision

Do not continue optimizing the current `route_bucket_rows` variant as the main
path: it improves accepted draft tokens only slightly (`1.569` vs dense
`1.494` accepted draft tokens/step) but drops same-root full-batch speedup to
`0.678x` and average GPU utilization to `44.000%`.

The next useful optimization should be one of:

1. make the mixed gate/up path graph-stable instead of `NONE=128`;
2. fuse or group sparse-base plus dense-row correction so corrected rows do not
   pay an additional fragmented path;
3. reduce residual rows only if the paired quality gate stays clean, because
   current corrected-row work is already the quality/speed tradeoff boundary.

## 2026-06-29 Update: Critical-Prefix Scaled-Bucket Breakdown

Current focused report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_criticalprefix_scaledbucket_20260629/seven_part_report/report.md
```

Raw roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_criticalprefix_scaledbucket_20260629/clean_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_slowdown_breakdown_criticalprefix_scaledbucket_20260629/instrumented_serving
```

Config: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, K=8,
`criticalprefix_extra2_gateup_scaledbucket`, `gate_up_proj=16-31`,
bucket size 4 scaled by active requests, bonus priority 0.5. Clean serving
uses max new tokens 256; instrumented rows use max new tokens 128 and CUDA
events, so their tok/s is diagnostic only.

### Short Answer

The current `speclink_t08` path is slow mostly because the mixed sparse-base
plus residual-correction operator does not create enough useful-work savings,
and because the clean mixed path has no CUDA Graph coverage. It is not mainly
a gather/scatter problem, and the clean scheduler/mask path is sub-ms.

| method | row kind | full-batch tok/s | total tok/s | avg GPU util | CUDA Graph |
| --- | --- | ---: | ---: | ---: | --- |
| dense baseline | clean | 3485.868 | 2336.624 | 88.857% | `FULL=115,NONE=76,PIECEWISE=1` |
| `base_only_24` | clean/stats | 3887.844 | 2571.626 | 83.385% | `NONE=128` |
| `speclink_t08` | clean | 3236.359 | 2278.722 | 87.214% | `NONE=128` |

Seven-part read:

| part | measured value | read |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` `0.537ms/step`; diagnostic exact routing `7.951ms/step`, mostly request loop `7.581ms` | clean path is not the primary bottleneck; exact diagnostics show sync-heavy routing overhead only when forced |
| base sparse linear | `gate_up_proj` layers 16-31 sparse base `1.052ms/call`, about `229` rows/call | this is the largest measured GPU-side component |
| residual correction | dense correction `0.192ms/call`, about `93` bucket rows/call | secondary to sparse base in this row, but still extra work |
| gather/scatter | `0.015ms/call` | not the current first bottleneck |
| routing statistics | draft residual/base `6348/7404`; non-draft residual/base `1719/1732`; bucket fill `0.983`; bucket actual/requested `6872/8067` | bucket is mostly filled; the issue is the cost of executing corrected rows, not wasted bucket slots |
| CUDA Graph | clean `speclink_t08` `NONE=128`, `FULL=0` | graph coverage is a real serving-side loss relative to dense |
| GPU util | clean `speclink_t08` avg `87.214%`, peak `100%` | not an idle-GPU problem; optimize useful work and graphability |

The next optimization direction should therefore change from broad selector
sweeps to:

1. make the mixed gate/up path graph-stable, or otherwise avoid the all-NONE
   CUDA Graph profile;
2. reduce or eliminate the full sparse-base pass for rows that will be
   overwritten by dense correction;
3. avoid spending time on gather/scatter first unless future runs show it
   growing, because it is currently much smaller than sparse base GEMM.

## 2026-06-29 Update: User-Requested Component Breakdown

Current focused report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_component_breakdown_qualitysafe_20260629_1058/seven_part_report_with_baseonly_safe/report.md
```

Raw roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_component_breakdown_qualitysafe_20260629_1058/clean_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_component_breakdown_qualitysafe_20260629_1058/baseonly_safe
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_component_breakdown_qualitysafe_20260629_1058/instrumented_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_component_breakdown_qualitysafe_20260629_1058/component_microbench
```

Config: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, K=8,
max new tokens 256 for clean serving, quality-safe SR24 scope
`down_proj=0-15`, `fixed_prefix=4`, `non_draft=all`, route-all residual rows.

### Short Answer

The current quality-safe `speclink_t08` row is not mainly slow because the GPU
is idle or because accepted length collapses. It is near parity with dense in
this short run, but it still does not provide the desired material speedup.

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3186.847 | 2609.786 | 1.7278 | 92.160% | `FULL=124,NONE=163,PIECEWISE=1` |
| `base_only_24` safe | 3115.611 | 2731.195 | 1.7666 | 95.083% | `FULL=194,NONE=43` |
| `speclink_t08` | 3209.523 | 2786.939 | 1.7711 | 93.174% | `FULL=62,NONE=161,PIECEWISE=1` |

The important read is:

1. `speclink_t08` has healthy GPU utilization and normal accepted length.
2. Its CUDA Graph coverage is worse than `base_only_24` and worse than the
   intended graph-stable path: `NONE` is about 72% of recorded steps.
3. The quality-safe routing policy corrects too many rows: diagnostic routing
   reports draft residual/base rows `13868/13868`, all non-draft rows residual
   (`7255`), and total correction fraction `0.604`.
4. The two-pass sparse-base plus correction operator only wins in the
   microbench at low residual fractions. Around 50%-60% correction it is not a
   strong speed path.

### 2026-06-29 Continuation: Base-Only And Compressed-Triton Cache Check

Additional roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_diagnosis_current_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressedtriton_cachecheck_bs64_math_20260629
```

The base-only diagnosis now cross-pairs dense baselines across sibling result
roots by `(dataset, batch_size)`, so a safe base-only rerun can still be
compared against the clean dense row. For Llama `math_reasoning` bs64:

| method | full-batch tok/s | speedup vs dense | accepted draft/step | dense accepted/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `base_only_24` safe | 3115.611 | 0.978x | 1.7666 | 1.7278 | 95.083% | `FULL=194,NONE=43` |

This reinforces the base-only answer: it is not slow because accepted length
collapsed, and it is not an idle-GPU issue. The down-proj-only sparse base path
is simply near dense parity for this serving shape.

For `all_corrected_24`, the compressed Triton path now caches the packed
residual value stream and packed mask bytes on the module's GPU tensors. A
small GPU check confirmed `cache_values_device=cuda:0`,
`cache_mask_device=cuda:0`, and identical outputs across repeated calls. The
serving breakdown also shows the cache behaving as expected:

```text
compressed_residual_triton_values_cached_misses=16
compressed_residual_triton_values_cached_hits=1008
compressed_residual_triton_mask_cached_misses=16
compressed_residual_triton_mask_cached_hits=1008
```

However, the end-to-end result does not improve materially:

| method | full-batch tok/s | total tok/s | avg GPU util | read |
| --- | ---: | ---: | ---: | --- |
| dense baseline | 3036.269 | 2268.591 | 78.375% | same short-run control |
| `all_corrected_24` compressed Triton | 1128.898 | 687.932 | 48.958% | still much slower |

The CUDA-event breakdown stays dominated by the custom compressed residual
kernel, not CPU transfer:

| component | avg per call | total |
| --- | ---: | ---: |
| base sparse down-proj | 0.281 ms | 288.194 ms |
| compressed residual Triton | 7.374 ms | 7551.066 ms |
| compressed residual add | 0.034 ms | 34.450 ms |

So the compressed-dense residual data is now demonstrably GPU-resident, but
the current direct compressed Triton algorithm is not a viable speed path. The
next useful `all_corrected_24` work is a real fused/grouped operator, not more
CPU-transfer cleanup.

The initial `base_only_24` row in `clean_serving` failed during vLLM
startup/profile_run because torch.compile traced into the cuSPARSELt sparse
tensor and hit a lazy-allocation error. The `baseonly_safe` rerun disables the
default vLLM compile path and should be used as the base-only reference for
this breakdown.

### 2026-06-29 Continuation: Selector Quality Probe

Additional GSM8K-30 roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k30_predfullaccept_criticalcap3_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k30_predfullaccept_criticalcap4_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k30_prefix4_predfullaccept_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k30_fixedprefix4_nondraftall_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k30_fixedprefix5_nondraftall_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downall_lmeval_gsm8k30_fixedprefix4_nondraftall_20260629
```

The lm-eval runner had a small compatibility bug: it exported
`SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET` but did not define
`--sr24-route-all-skip-bucket` in the parser. That is fixed; manual SR24
lm-eval runs can now use the same route-all skip-bucket flag as the throughput
runner.

The selector probes changed the quality read:

| candidate | GSM8K-30 exact | pair reg | pair imp | read |
| --- | ---: | ---: | ---: | --- |
| `critical_prefix`, cap3, `predicted_full_accept` | 0.6667 | 2 | 2 | aggregate cancels out but paired unsafe; doc2/doc20 regress |
| `critical_prefix`, cap4, `predicted_full_accept` | 0.6667 | 2 | 2 | still unsafe because max cap alone does not force prefix rows |
| prefix4 forced, `predicted_full_accept` | 0.6667 | 2 | 2 | forcing prefix is not enough; missing non-draft correction matters |
| fixed_prefix4, `non_draft=all`, down0-15 | 0.7333 | 1 | 3 | fixes doc2 but leaves doc20 regression |
| fixed_prefix5, `non_draft=all`, down0-15 | 0.7333 | not paired in-run | not paired in-run | doc20 still wrong; adding one more prefix row is not the fix |
| fixed_prefix4, `non_draft=all`, all down_proj | 0.6667 | not paired in-run | not paired in-run | doc20 still wrong; simply widening down_proj residual scope is not enough |

This means the current precision problem is not solved by reducing correction
to a small confidence-selected prefix. For the focused GSM8K examples,
`non_draft=all` is needed to avoid the doc2 regression, but doc20 remains
unstable even when prefix length and down-proj layer scope are increased. The
doc20 dense output is also an extraction-sensitive case: the dense text is not
a clean reasoning solution, but the flexible extractor marks it correct. Treat
this sample as a strict paired-regression gate, not as proof that the dense
reasoning path is semantically better.

Current quality-safe-enough working point for performance debugging remains
the fixed-prefix / non-draft-all family. Further selector work should use a
larger paired gate and should not promote `predicted_full_accept` as a default
unless it clears doc2-like regressions.

### Seven-Part Breakdown

| part | measured value | read |
| --- | --- | --- |
| scheduler / mask build | exact diagnostic `19.307ms/step`, mostly request loop `19.128ms` | this is sync-heavy diagnostic overhead, not clean serving cost; still shows the Python/request routing path is expensive when exact counters are enabled |
| base sparse linear | `down_proj` sparse base `0.794ms/call`, about `167` base rows/call | sparse base is a major GPU-side cost; it is not close to free |
| residual correction | dense-row correction `0.142ms/call`, about `254` corrected rows/call | secondary to sparse base in this row, but high corrected-row fraction makes it unavoidable |
| gather/scatter | route-all gather/scatter `0.021ms/linear`; base/dense gather and index-copy each around `0.004-0.006ms` | not the first bottleneck right now |
| routing statistics | draft residual/base `13868/13868`; non-draft residual `7255`; correction fraction `0.604` | accuracy-safe policy protects many rows, leaving too little low-cost base-only work |
| CUDA Graph | clean `speclink_t08` `FULL=62,NONE=161`; `base_only_24` safe `FULL=194,NONE=43` | mixed route-all path still loses graph coverage relative to base-only |
| GPU util | clean `speclink_t08` avg `93.174%`, peak `99%` | not an idle-GPU problem; the issue is useful-work efficiency and graph/shape overhead |

### Operator Microbench Read

For gate/up-like `512x28672x4096`, base 2:4 alone is about `0.65x` dense, but
the current mixed path is already slower than dense even at 12.5%-25% residual
rows (`1.03x-1.15x` dense time). For down-proj-like `512x4096x14336`, mixed is
competitive only at low residual fractions: `0.91x` dense time at 12.5%,
`0.99x` at 25%, but `1.24x` at 50%.

Because the serving diagnostic correction fraction is about `60%`, the current
quality-safe route-all `speclink_t08` policy is operating outside the positive
microbench region. This is the main reason a controller sweep alone is unlikely
to create a large speedup.

### Next Optimization Direction

The next pass should optimize for these two targets before another full
accuracy/performance matrix:

1. restore graph-stable mixed execution for the quality-safe route-all path, so
   `speclink_t08` is closer to the `base_only_24` CUDA Graph profile;
2. reduce the corrected-row fraction or switch to an execution plan that avoids
   sparse-base work on rows that will be corrected, because the current
   base-all-then-correct operator is not profitable near 60% correction.

If both are insufficient, the remaining likely path is a fused down-proj
operator or a grouped route plan that computes dense rows and sparse rows
without the current two-pass base-all cost.

### Follow-Up: Direct Row Lists And Dense Fallback

Direct CPU route-row construction was corrected for
`fixed_prefix + non_draft=all`: residual rows now mean all rows except the
base-only draft suffix (`pos >= prefix`) instead of only the prefix/bonus rows.
The focused unit smoke passes:

```text
conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
speclink_sr24_correctness=ok
```

However, direct CPU row lists are not a valid graph/compile speed path yet.
Evidence:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k10_fixedprefix4_directcpuall_eager_20260629_1113/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k10_fixedprefix4_directcpuall_guarded_20260629_1117/report.md
```

| variant | GSM8K-10 exact | read |
| --- | ---: | --- |
| direct CPU rows, `--enforce-eager` | 0.8000 | semantically correct |
| direct CPU rows, graph/default compile config | 0.7000 | doc2 regresses again |

The runner now treats `speclink_t08 + direct_cpu_route_rows +
route_all_residual_rows` as enforce-eager only. This prevents future benchmark
rows from silently using an unsafe graph/compile path. It is a correctness
guard, not a speed optimization.

The dense-fallback threshold was also tested because the microbench shows
mixed sparse+residual loses once correction fraction is high:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_fixedprefix4_fallback025_bs64_math_20260629/report.md
```

| variant | full-batch tok/s | total tok/s | CUDA Graph |
| --- | ---: | ---: | --- |
| dense baseline | 3186.391 | 2610.112 | `FULL=134,NONE=160,PIECEWISE=1` |
| `speclink_t08`, route dense fallback 0.25 | 3188.661 | 2768.922 | `FULL=85,NONE=163,PIECEWISE=1` |

This avoids a worse mixed-operator path but does not create a material speedup:
steady full-batch throughput is still effectively dense parity. It confirms
that the next useful performance work is not another fallback threshold sweep;
it is either lowering the correction fraction while preserving accuracy or
replacing the two-pass route-all operator with a fused/grouped implementation.

### Follow-Up: `all_corrected_24` Operator Breakdown

Focused all-corrected roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_densefastpath_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_cached_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_breakdown_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_residual_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_directcslt_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressedtriton_breakdown_bs64_math_20260629
```

Config: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, K=8,
down-proj-only SR24 scope `down_proj=0-15`.

| variant | full-batch tok/s | same-run dense full-batch tok/s | read |
| --- | ---: | ---: | --- |
| default `all_corrected_24` dense-fastpath | 3199.494 | 3189.435 | dense-equivalent control; SR24 attach/hook is not the bottleneck |
| `compressed_dense`, cached/prewarmed GPU residual | 2898.842 | 3192.483 | residual values are GPU-resident, but sparse-base plus dense residual GEMM is slower than dense |
| `torch_sparse` residual | 2976.045 | 3197.024 | two sparse GEMMs are better than compressed dense, but still slower than dense |
| `torch_sparse` residual, direct cuSPARSELt | 2868.634 | 3186.488 | direct dispatch is worse here; PyTorch sparse dispatch is not the main bottleneck |
| compressed residual Triton diagnostic | 1122.825 | not run | custom compressed residual matmul is much too slow in its current form |

The clean `compressed_dense` run confirms that the residual path is not falling
back to CPU:

```text
backend: torch_sparse/compressed_dense@cuda
compressed_residual_runtime_on_gpu: True
residual_device_counts: {"cuda:0": 16}
effective cache/prewarm: True/True
effective residual_out_chunk: 0
CUDA Graph: FULL=206,NONE=42
```

The CUDA-event diagnostic shows why it is still slow:

| component | average per down_proj call | total in diagnostic |
| --- | ---: | ---: |
| scheduler / mask build | 0.035 ms/step | 2.943 ms |
| base sparse linear | 0.449 ms/call | 604.075 ms |
| compressed residual materialize | 0.0004 ms/call | 0.521 ms |
| residual dense GEMM | 0.366 ms/call | 491.917 ms |
| residual add | 0.028 ms/call | 37.097 ms |

So `all_corrected_24` is not slow because compressed residual values live on
CPU, nor because scheduler mask construction dominates. The slow part is the
operator structure: for every corrected down-proj row it computes the sparse
base output and then computes the complementary residual output, so the
effective work is roughly `base 2:4 GEMM + residual GEMM + add`. Even when the
residual dense weight is prewarmed and cached on GPU, that two-pass structure
is slower than the original dense down-proj for this serving shape.

The current compressed-residual Triton kernel is also not a useful optimization
path as written:

```text
compressed_residual_triton_cuda_ms: 7.423 ms/call
base_sparse_linear_cuda_ms: 0.284 ms/call
avg GPU util: 52.667%
```

The implication for `all_corrected_24` is conservative: keep the default
dense-fastpath as the exact dense-equivalent performance control. Treat
no-fastpath `compressed_dense` and `torch_sparse` residual as operator
diagnostics only. A real speed path would need a fused sparse-base/residual
kernel or a grouped route plan that avoids computing sparse base on rows that
will be corrected; simply moving compressed residual work to GPU is already
done and is not enough.

### Follow-Up: Row-Routed Down Is Not The Speed Path

Focused row-routed roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_rowrouteddown_skipbucket_env_llama_gsm8k10_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_rowrouteddown_bs64_math_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_rowrouteddown_trueeager_counterdiag_bs64_math_20260629
```

Two runner bugs were fixed before trusting the diagnostic:

1. `run_lm_eval_accuracy.py` now passes
   `SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET=1` for the
   `down0_15_fixedprefix4_directcslt` preset. Before this, the accuracy runner
   set the Python arg but did not export the environment variable, so it was
   not fully aligned with the throughput runner.
2. `run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py` now preserves
   `--sr24-force-eager-after-preset` across preset application and sets
   `sr24_default_vllm_compile=False` when the flag is used. Before this fix,
   the requested eager linear-breakdown diagnostic still loaded default
   torch.compile graphs, hiding Python-side row-routed counters.

Correctness gate after the first fix:

| model | task | mode | exact-match |
| --- | --- | --- | ---: |
| Llama-3.1-8B | GSM8K-10 | dense baseline | 0.8000 |
| Llama-3.1-8B | GSM8K-10 | `speclink_t08` row-routed down | 0.8000 |

The clean serving throughput does not improve:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3205.208 | 2787.483 | 1.7644 | 92.826% | `FULL=57,NONE=166,PIECEWISE=1` |
| `speclink_t08` row-routed down | 3174.039 | 2597.805 | 1.7341 | 93.720% | `FULL=125,NONE=162,PIECEWISE=1` |

The true eager counter diagnostic confirms that row-routed down executes, but
the execution shape is unfavorable:

| component | average per hooked down-proj call | total in 8-request diagnostic |
| --- | ---: | ---: |
| scheduler batched mask kernel | 0.061 ms/step | 3.727 ms |
| scheduler row-index build | 0.082 ms/step | 5.026 ms |
| row-routed dense gather | 0.004 ms/call | 4.054 ms |
| row-routed dense GEMM | 0.088 ms/call | 84.558 ms |
| row-routed base gather | 0.004 ms/call | 4.180 ms |
| row-routed base sparse GEMM | 1.013 ms/call | 972.198 ms |
| row-routed index-copy assembly | 0.008 ms/call | 7.904 ms |

Routing counters from the same diagnostic:

```text
row_routed_down_entered=2560
row_routed_down_executed=960
row_routed_down_skip_wrong_leaf=960
row_routed_down_skip_non_mixed_state=640
row_routed_down_dense_rows=40640
row_routed_down_base_rows=27136
scheduler_row_indices_residual_rows=2555
scheduler_row_indices_base_rows=1708
```

This is the key read: row-routed down avoids dense correction on base-only rows,
but it turns the base sparse GEMM into many smaller row-batch sparse GEMMs.
Those small sparse GEMMs are very inefficient on this stack:
`row_routed_down_base_sparse_cuda_ms` is about `1.013ms/call`, much larger than
the earlier full-row sparse-base measurements. Dense correction is not the
bottleneck in this route; it is only `0.088ms/call`. Gather/scatter is also not
the bottleneck.

So the useful optimization direction is not row-routed down with the current
PyTorch/cuSPARSELt sparse tensor path. Keep the quality-safe non-row-routed
path as the serving baseline. A profitable route plan would need a custom
grouped/fused kernel that can handle base rows without launching an
underfilled sparse GEMM per layer/step.

## 2026-06-29 Update: Current Best Dynamic-Graph Breakdown

Current authoritative report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_criticalcap3_bucket256_final_breakdown_20260629/seven_part_report/report.md
```

This report combines:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_criticalcap3_bucket256_dynamicgraph_bs64_math_20260629/clean_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_criticalcap3_bucket256_instr_20260629/instrumented_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_compressed_triton_probe_gpu_mask_20260629
```

### Short Answer

The current `speclink_t08` slowdown is no longer primarily a CUDA Graph or
simple GPU-idleness problem. With `--sr24-dynamic-auto-cudagraph`,
`--sr24-cudagraph-bucket`, and bucket size 256, the clean serving path reaches
good graph coverage and high GPU utilization:

| method | full-batch output tok/s | total output tok/s | speedup vs dense full-batch | avg GPU util | CUDA Graph modes |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3485.760 | 2336.021 | 1.000x | 89.214% | `FULL=115,NONE=76,PIECEWISE=1` |
| `speclink_t08` | 3176.908 | 2368.779 | 0.911x | 90.714% | `FULL=126,NONE=2` |

`speclink_t08` is slightly higher on total tok/s in this fixed-request run,
but still lower on the full-batch window. The full-batch number is the cleaner
comparison for whether the steady high-concurrency decode kernel path is
actually faster.

### Seven-Part Read

| part | latest evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean path `0.352ms/step`; bucket build `0.084ms`, row indices `0.002ms`, direct CPU rows `0.000ms` | clean scheduler overhead is already sub-ms and is not the main bottleneck |
| base sparse linear | diagnostic `speclink_t08` down_proj sparse base `0.819ms/call`, about `434.8` base rows/call | sparse base is a large GPU-side cost, not a free 2:4 win |
| residual correction | diagnostic dense-row correction `0.166ms/call`, bucket rows `256/call` | secondary to base sparse for current capped `t08`, but still erases part of sparse gain |
| gather/scatter | diagnostic `0.013ms/call` | not the current first bottleneck |
| routing statistics | draft residual/base `8232/19328`; non-draft residual/base `584/6649`; bucket fill `0.585`; bucket actual/requested `8240/8818=0.934` | the policy still corrects about 30% of draft rows; bucket capacity 256 is often underfilled but kept graph-stable |
| CUDA Graph | clean `speclink_t08` `FULL=126,NONE=2`, FULL fraction `0.984` | graph coverage has been fixed for this candidate |
| GPU util | clean `speclink_t08` avg `90.714%`, peak `100%` | GPU is busy; remaining issue is useful-work efficiency/operator shape, not idle GPU |

### Operator Evidence

The isolated microbench explains why the end-to-end speedup is hard to get:

| shape | residual frac | dense graph ms | sparse base graph ms | current mixed graph ms | read |
| --- | ---: | ---: | ---: | ---: | --- |
| gate_up-like `512x28672x4096` | 0.25 | 0.538 | 0.353 | 0.619 | mixed is already slower than dense |
| gate_up-like `512x28672x4096` | 0.50 | 0.540 | 0.353 | 0.831 | residual correction overwhelms base sparse savings |
| down_proj-like `512x4096x14336` | 0.25 | 0.291 | 0.166 | 0.286 | barely competitive only at low residual fraction |
| down_proj-like `512x4096x14336` | 0.50 | 0.291 | 0.166 | 0.363 | slower once residual fraction rises |

The actual compressed-residual Triton path is much slower than dense in this
microbench, so it should not be the next optimization target unless rewritten
as a real fused sparse-plus-residual operator.

### Current Optimization Direction

Do not spend the next pass on more acceptance-length tuning alone. The clean
serving path already has high GPU util and good graph coverage. The next useful
experiments should be:

1. reduce corrected rows while preserving accuracy, especially by testing
   smaller graph-stable bucket sizes such as 128/160/192 instead of 256;
2. avoid sparse-base work on rows that will be overwritten by dense correction,
   but without reintroducing the CPU score-copy cost seen in direct CPU routing;
3. if those are insufficient, implement a real fused `down_proj` sparse-base
   plus dense correction kernel, because the current two-pass operator cannot
   reliably beat dense at the measured residual fractions.

## 2026-06-29 Update: Base-Only And Bucket Sweep

The latest focused summary is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_bucket_sweep_current_20260629/summary.md
```

### Base-Only Answer

For the down-proj-only current scope, `base_only_24` is not slow because
acceptance collapses or because the GPU is idle:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3486.725 | 2338.210 | 1.7337 | 89.000% | `FULL=115,NONE=76,PIECEWISE=1` |
| `base_only_24` | 3463.090 | 2456.504 | 1.7280 | 91.538% | `FULL=126,NONE=2` |

The base-only accepted length is essentially the same as dense, and GPU util is
higher. The real read is that this down-proj-only 2:4 base path is only
near-parity with dense for the serving shape, not a large sparse speedup.

### Bucket Sweep

No-route bucket shrinking improves throughput but is not quality-safe:

| candidate | full-batch tok/s | same-run dense full-batch tok/s | GSM8K-10 gate |
| --- | ---: | ---: | --- |
| bucket 128, threshold 0.3 | 3419.373 | 3486.725 | fails, paired regression 1 |
| bucket 160, threshold 0.3 | 3242.539 | 3486.725 | not promoted |
| bucket 192, threshold 0.3 | 3245.280 | 3486.725 | fails, paired regression 1 |
| bucket 128, threshold 0.8 | not rerun for throughput |  | fails, paired regression 1 |
| bucket 128, threshold 0.8, eager | not rerun for throughput |  | fails, paired regression 2 |

The repeated regression is GSM8K `doc_id:2`. Since the failure persists in
eager mode, it is not a CUDA Graph capture bug. It is the no-route bucket-only
correction path or policy being insufficient for this math sample.

The current quality-safe reference remains `down0_15_fixedprefix4_directcslt`:

| candidate | full-batch tok/s | dense full-batch tok/s in same run | GSM8K-10 gate |
| --- | ---: | ---: | --- |
| `down0_15_fixedprefix4_directcslt` | 3194.467 | 3188.254 | passes, paired regression 0 |

This confirms the optimization direction: keep the quality-safe route-all /
fixed-prefix semantics, then make that path cheaper. Do not use bucket
128/160/192 as the main speed path until they pass paired accuracy gates.

The sections below are older notes from the same SR24 debugging thread and may
contain pre-dynamic-graph measurements where mixed SR24 still had many
CUDA-Graph `NONE` steps.

---

This is the current root-cause read before the next SR24 optimization pass.
The latest refreshed seven-part report, aligned with the requested
`scheduler / sparse base / residual correction / gather-scatter / routing /
CUDA Graph / GPU util` table, is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_table_breakdown_current_20260629/seven_part_report_with_baseonly/report.md
```

The corresponding raw roots are:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_table_breakdown_current_20260629/clean_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_table_breakdown_current_20260629/baseonly_manual_safe
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_table_breakdown_current_20260629/instrumented_serving
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_table_breakdown_current_20260629/component_microbench
```

## Short Answer

The current slowdown is not primarily an accepted-length problem. It is also
not explained by GPU idleness. The clean bs64/math/max256 rows show:

| method | full-batch output tok/s | total output tok/s | speedup vs dense | avg GPU util | CUDA Graph modes |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3279.950 | 2876.585 | 1.000x | 93.957% | `FULL=61,NONE=162,PIECEWISE=1` |
| base_only_24 | 3724.388 | 3124.546 | 1.086x | 94.286% | `FULL=176,NONE=41` |
| speclink_t08 | 3150.598 | 2569.479 | 0.961x | 93.240% | `FULL=125,NONE=162,PIECEWISE=1` |
| all_corrected_24 | 3187.023 | 2609.749 | 0.972x | 92.640% | `FULL=124,NONE=163,PIECEWISE=1` |

`base_only_24` still has a real sparse upper bound, but `speclink_t08` and
`all_corrected_24` do not reach it. The slow path is the combination of:

1. CUDA Graph coverage misses on guarded mixed SR24 paths.
2. Too much useful work in sparse-base plus residual-correction Linear paths.
3. High residual-row fraction, so the residual correction erases most of the
   structured-sparse base speedup.
4. In continuous/refill serving, higher TTFT/lower effective utilization can
   still matter, but it is not the main fixed-window finding here.

The diagnostic row localizes the extra work for `speclink_t08`:
`gate_up_proj[16-31]` sparse base is `1.078ms/call`, total measured sparse base
is `0.973ms/call`, dense-row correction is `0.163ms/call`, and gather/scatter
is only `0.016ms/call`. Routing still sends many rows to the protected path:
draft residual/base rows are `13577/11719`, non-draft residual/base rows are
`3162/3788`, with bucket fill `0.979`.

The first clean `base_only_24` run failed in the default
compile/direct-cuSPARSELt path with a Torch compile/cache lazy-allocation error.
The `baseonly_manual_safe` rerun disables that path and succeeds with
`3724.388` full-batch tok/s. Treat this as the current broad-scope base-only
upper bound and the startup failure as a separate compile-path bug, not as the
main SR24 slowdown.

## 2026-06-29 Fixed-Prefix Correction

The first batched precision failure was not CUDA Graph related. The same
`down0_15_fixedprefix4_directcslt` candidate matched dense at batch size 1 but
regressed at batch size 8/eager on GSM8K doc2. Two issues were separated:

1. `fixed_prefix` plus a capped residual bucket was quality-unsafe. With
   batch=8, K=8, prefix=4, and bonus correction, the policy requests about
   40 corrected rows, but the old `residual_bucket_size=16` path could correct
   only the first 16. The runtime now disables the bucket when fixed-prefix
   requested rows exceed the bucket, including the `non_draft=all` case where
   non-draft/prefill rows are also required to be dense.
2. More importantly, `non_draft=bonus` made prompt/context and other non-draft
   rows use the sparse base path. That changes prefill/KV state and is not the
   intended "only approximate draft rows" semantics. Switching the down0
   candidate to `non_draft=all` fixes the focused batch=8 GSM8K-10 regression.

Focused accuracy check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k10_preset_nondraftall_20260629/report.md
```

| case | exact | paired regression vs dense | read |
| --- | ---: | ---: | --- |
| dense baseline | 0.8000 | 0 | reference |
| updated `speclink_t08` | 0.8000 | 0 | fixed-prefix4, `non_draft=all`, down_proj 0-15 |

A prefix-length ablation shows why the current quality floor is still prefix 4:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k10_prefix2_nondraftall_20260629/report.md
```

| case | exact | paired regression vs dense | read |
| --- | ---: | ---: | --- |
| prefix2, `non_draft=all` | 0.7000 | 1 | doc2 regresses back to `120000` |

Focused throughput check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_nondraftall_bs64_math_20260629/report.md
```

| method | full-batch output tok/s | total output tok/s | avg GPU util | read |
| --- | ---: | ---: | ---: | --- |
| dense baseline | 3188.254 | 2610.469 | 92.480% | current comparison run |
| updated `speclink_t08` | 3194.467 | 2774.724 | 92.174% | accuracy-safe but only parity-level speed |

This fixes the immediate quality bug without giving the desired speedup.
The remaining performance gap to the `base_only_24` upper bound is now clearer:
accuracy-safe `non_draft=all` leaves only non-prefix draft rows available for
sparse execution, so the mixed operator must become cheaper or the draft-row
selection must become more selective.

## Seven-Part Read

| part | current evidence | read |
| --- | --- | --- |
| scheduler / mask build | diagnostic exact routing shows `mask=13.496ms/step`, `request_loop=13.229ms`, but this is the sync-heavy diagnostic path | do not use diagnostic scheduler timing as serving throughput; use it only to identify CPU-sync overhead |
| base sparse linear | diagnostic `speclink_t08` `gate_up_proj[16-31]=1.078ms/call`, base sparse `0.973ms/call`; all-corrected base sparse `0.695ms/call` | sparse base is a real GPU-side cost, not free sparse speed |
| residual correction | diagnostic `speclink_t08` dense correction `0.163ms/call`; all-corrected residual dense `0.569ms/call` | correction is secondary in capped t08 but large when all rows are corrected |
| gather/scatter | diagnostic `speclink_t08` `0.016ms/call`; all-corrected `0.030ms/call` | not the first bottleneck in current measurements |
| routing statistics | `speclink_t08` draft residual/base `13577/11719`, non-draft `3162/3788`, bucket fill `0.979` | too many rows still take correction for the sparse upper bound to translate into speedup |
| CUDA Graph | dense `FULL=61,NONE=162`; base-only `FULL=176,NONE=41`; `speclink_t08 FULL=125,NONE=162`; `all_corrected FULL=124,NONE=163` | graph coverage is materially worse than base-only and still costs throughput |
| GPU util | clean dense/base-only/t08/all-corrected util `94.0%/94.3%/93.2%/92.6%` | GPU is busy; the problem is useful-work efficiency, not simple underutilization |

## Breakdown Rule Going Forward

Every future candidate should be judged by this order before a full matrix:

1. clean serving: tok/s, GPU util, CUDA Graph FULL/NONE;
2. low-sync routing counts: draft residual rows, non-draft residual rows, bucket
   active/requested rows;
3. instrumented serving: sparse-base, dense correction, gather/scatter CUDA
   event timings;
4. operator microbench: whether the proposed sparse/residual operator can beat
   dense at the observed residual-row fraction.

Do not use instrumented-serving tok/s as the main speed number. Its CUDA event
and exact routing counters add synchronization; use it only to locate the slow
component.

## Current Interpretation

`base_only_24` is not the problem by itself. In the broad-scope safe rerun it
reached `3724.388` full-batch tok/s versus dense `3279.950`, with GPU util
`94.286%` and much better graph coverage than mixed SR24. That means the 2:4
base path still has a real upper bound.

`speclink_t08` is slow because it cannot convert the base-only upper bound into
end-to-end speed. It reaches only `3150.598` full-batch tok/s (`0.961x` dense)
while GPU util remains `93.240%`. The instrumented row points to inefficient
work: sparse base `0.973ms/call`, correction `0.163ms/call`, and many residual
rows. Gather/scatter is not the first bottleneck.

`all_corrected_24` is a useful negative control. It reaches `3187.023`
full-batch tok/s in the current dense-fastpath clean row, but the instrumented
true correction row shows why the real sparse+residual operator is unattractive:
base sparse `0.695ms/call` plus residual dense `0.569ms/call` when every row is
corrected. This confirms that the hook attachment itself is not the root
slowdown; the real sparse+residual operator path is.

## All-Corrected Smoke Check

A focused bs16 math smoke was run to separate three cases:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_current_modes_smoke_20260629
```

The important distinction is that the runner's default
`all_corrected_24` control can use a dense/no-op fast path. That is useful as a
correctness and hook-overhead control, but it is not the sparse+residual
operator path.

| mode | dense full-batch tok/s | all_corrected full-batch tok/s | read |
| --- | ---: | ---: | --- |
| densefastpath | 1229.938 | 1583.232 | cold/small smoke; all_corrected used `dense_fastpath_noop=True` |
| sparse_residual | 1453.822 | 1305.598 | true sparse base plus dense-row correction is slower than dense |
| early_dense_hook | 1447.252 | 1441.183 | exact dense shortcut through SR24 hooks is dense-equivalent |

This confirms the slowdown is not from merely attaching SR24 hooks. It appears
when the true sparse-base plus residual-correction operator is used.

## Precision Probe

The current `speclink_t08` quality issue is also not just a threshold problem.
A stable GSM8K regression was replayed:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_docs2_11_currentcandidate_20260629/dense_baseline/gsm8k_cot_doc2_dense_baseline.json
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_docs2_11_currentcandidate_20260629/sr24compile_speclink_t08/gsm8k_cot_doc2_selective_prefix4.json
```

Dense answers `70000`; current narrow-scope `speclink_t08` answers `50000`.
When the residual scope is widened to both `gate_up_proj` and `down_proj` over
layers `0-31`, the same request answers `70000` again:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_doc2_scope_probe_20260629/full_mlp_residual/gsm8k_cot_doc2_selective_prefix4.json
```

So the next quality fix should not be another plain `t00-t10` sweep. It should
first choose a residual coverage policy that protects the MLP modules/layers
needed for math correctness, then reduce how many token rows enter that
protected path.

### Follow-Up Scope Sweep

A focused scope sweep was run under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_replay_doc2_scope_probe_20260629
```

and the matching bs64/math/max128 throughput candidates are summarized at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fullscope_candidates_bs64_math128_20260629/report.md
```

The key precision result is:

| candidate | doc2 answer | read |
| --- | ---: | --- |
| current narrow `speclink_t08` | 50000 | wrong |
| full MLP fixed_prefix4 | 70000 | fixed |
| gate_up=0-31 only, down dense | 50000 | wrong |
| down_proj=0-31 only, gate_up dense | 70000 | fixed |
| down_proj=0-15 only, gate_up dense | 70000 | fixed |
| down_proj=0-31 plus gate_up=16-31 base-only | 10000 | wrong |

The best focused candidate is now available as:

```text
--sr24-preset down0_15_fixedprefix4_directcslt
```

It protects only `down_proj=0-15` with fixed-prefix residual correction and
keeps `gate_up` dense. On the bs64/math/max128 fixed-batch smoke it reached
`3031.6` full-batch tok/s versus dense `3157.7` (`0.960x`). This is not yet a
speedup, but it is a better precision/scope starting point than the full MLP
candidate (`2942.3`, `0.932x`) and it avoids the gate-up tail base-only
regression (`10000` on doc2).

## Next Optimization Direction

Do not start with another plain threshold sweep. The next pass should keep the
seven-part breakdown as the acceptance criterion for every candidate:

- first, reduce residual rows enough that the mixed operator lands near the
  microbench break-even region;
- second, keep CUDA Graph `FULL` coverage in clean serving;
- third, replace or fuse the sparse-base plus dense-row correction path if
  residual rows cannot be reduced without quality loss;
- fourth, treat CPU-side sync/row-index construction as an ablation, not the
  whole explanation: the updated accuracy-safe path keeps non-draft rows dense,
  so the remaining speed opportunity is mostly in making the draft-tail sparse
  split cheap enough;
- fifth, check continuous/refill serving, not only early full-batch windows.

## User-Requested Seven-Part Breakdown 2026-06-29

The latest focused breakdown was run for Llama-3.1-8B, `math_reasoning`, bs64,
K=8, max tokens 256, using the current accuracy-safe
`down0_15_fixedprefix4_directcslt` path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_20260629
```

Primary reports:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_20260629/seven_part_report_with_base_safe/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_20260629/component_summary_with_base_safe/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_20260629/component_microbench/summary.md
```

Clean serving rows:

| method | full-batch tok/s | total tok/s | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | --- |
| dense_baseline | 3477.817 | 2332.933 | 90.500% | `{"FULL": 115, "NONE": 76, "PIECEWISE": 1}` |
| speclink_t08 | 3455.804 | 2304.712 | 87.929% | `{"FULL": 115, "NONE": 76, "PIECEWISE": 1}` |
| all_corrected_24 densefastpath | 3480.613 | 2335.409 | 88.214% | `{"FULL": 115, "NONE": 76, "PIECEWISE": 1}` |

The clean `speclink_t08` path is effectively dense parity, not a speedup:
`0.994x` by full-batch throughput. GPU utilization stays high enough that the
problem is not simply idle GPU time. It is useful-work efficiency.

The first `base_only_24` row in the clean run failed under direct-cuSPARSELt plus
default compile with:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

A safe control without direct CSLT/default compile ran successfully:

| method | full-batch tok/s | total tok/s | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | --- |
| dense_baseline | 3555.052 | 2352.252 | 90.500% | `{"FULL": 114, "NONE": 77, "PIECEWISE": 1}` |
| base_only_24 safe/eager | 3222.465 | 2173.432 | 84.400% | `{"NONE": 151}` |

This safe control is not a speed upper bound because it is all eager. It only
confirms that base-only sparse execution works when avoiding the compile lazy
allocation failure.

Diagnostic linear timing for `speclink_t08`:

| part | value | read |
| --- | ---: | --- |
| scheduler/mask exact diagnostic | 12.682 ms/step | diagnostic sync path; not clean throughput |
| sparse base Linear | 0.914 ms/call | dominant GPU-side SR24 Linear cost |
| dense residual correction | 0.111 ms/call | smaller than sparse base in this path |
| route-all gather/scatter | 0.017 ms/linear | not the first bottleneck |
| draft residual/base rows | 7068 / 7068 | fixed-prefix sends half of draft rows to dense correction |
| non-draft residual/base rows | 3441 / 0 | all non-draft rows are dense-corrected for quality |
| correction row fraction | 0.598 | too high for a two-pass sparse+residual operator to win reliably |

The component microbench gives the important operator-level boundary. For
`down_proj` shape `512x4096x14336`, base sparse alone is strong
(`0.166ms` vs dense `0.291ms`, `0.57x`), but the current mixed two-pass path
only beats dense when residual rows stay low:

| residual fraction | mixed/dense | read |
| ---: | ---: | --- |
| 0.125 | 0.91x | promising region |
| 0.250 | 0.99x | break-even |
| 0.500 | 1.24x | loses |
| 0.875 | 1.55x | loses badly |
| 1.000 | 1.77x | all-corrected two-pass is not viable |

For `gate_up` shape `512x28672x4096`, even 12.5% residual is already about
dense parity (`1.03x`), and higher residual fractions quickly lose. That
explains why gate-up selective correction was a poor speed target even before
quality issues.

Current conclusion: the slow part is not primarily gather/scatter and not a
completely idle GPU. The bottleneck is that the useful sparse-base saving is
eaten by a two-pass mixed operator while the quality-safe policy still routes
too many rows through exact dense correction. The next optimization should
therefore reduce corrected row fraction below roughly 25% for `down_proj`, keep
`gate_up` dense or use a different fused strategy, and only then worry about
CPU-side sync/mask construction as a secondary ablation.

## Follow-Up Checks 2026-06-29

### Fixed-Prefix Quality Boundary

The focused `fixed_prefix=3` candidate was tested with
`non_draft=predicted_full_accept`, `down_proj=0-15`, K=8, GSM8K-10, batch 8:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k10_predfullaccept_prefix3_20260629/report.md
```

Result:

| mode | exact | paired regressions | read |
| --- | ---: | ---: | --- |
| dense_baseline | 0.8000 | 0 | reference |
| speclink_t08 prefix3 | 0.7000 | 1 | quality unsafe |

The regression is GSM8K `doc_id:2`: dense answers `70000`, while prefix3 SR24
answers `12000`. Prefix2 had already failed, and prefix4 was the focused
quality-safe setting. This means the current fixed-prefix selector cannot push
the draft residual fraction down to the `~25%` microbench break-even region.
Further speed work should not keep reducing a plain fixed prefix; it needs a
better row selector or a cheaper fused operator.

### Actual Compressed Triton Residual Probe

The component microbench now includes the actual direct compressed-residual
Triton path from `vllm.speclink_sr24`. The diagnostic module keeps packed mask
bytes on CUDA, matching serving attach for `compressed_dense` with
`residual_device=cuda`.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_compressed_triton_probe_gpu_mask_20260629/summary.md
```

Key graph ratios versus dense:

| shape | residual frac | base/dense | mixed/dense | cached dense-residual/dense | compressed Triton/dense | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate_up `512x28672x4096` | 0.25 | 0.66x | 1.15x | 1.06x | 2.86x | Triton direct is not viable |
| gate_up `512x28672x4096` | 0.50 | 0.65x | 1.54x | 1.37x | 4.63x | loses badly |
| gate_up `512x28672x4096` | 1.00 | 0.64x | 2.17x | 1.89x | 8.14x | all-corrected cannot win |
| down_proj `512x4096x14336` | 0.25 | 0.57x | 0.98x | 0.95x | 3.51x | break-even only with dense residual |
| down_proj `512x4096x14336` | 0.50 | 0.57x | 1.25x | 1.19x | 5.24x | too many residual rows |
| down_proj `512x4096x14336` | 1.00 | 0.57x | 1.77x | 1.68x | 9.95x | full residual is a negative control |

This confirms `compressed_dense` is not accidentally CPU-side in the intended
serving configuration: residual values and packed masks can be GPU resident.
However, computing the residual as a separate direct compressed Triton matmul is
much slower than dense GEMM. The best current exact all-corrected behavior is
therefore either the dense fastpath/no-op control, or a future fused kernel that
combines base sparse and residual correction without launching a second
expensive residual matmul.

### Critical-Prefix Cap3 Candidate

Because fixed-prefix 3 failed, a confidence-aware selector was tested:
`critical_prefix`, `max_residual_draft_rows=3`,
`non_draft=predicted_full_accept`, `down_proj=0-15`.

Focused GSM8K-10 quality gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down0_15_lmeval_gsm8k10_predfullaccept_criticalcap3_20260629/report.md
```

| mode | exact | paired regressions | read |
| --- | ---: | ---: | --- |
| dense_baseline | 0.7000 | 0 | same-run reference |
| critical_prefix cap3 | 0.7000 | 0 | small quality gate passed |

This is not a full quality proof, but it is the first tested candidate that
keeps the corrected draft budget below fixed-prefix4 while avoiding paired
regression on this small gate.

Throughput/routing checks then showed the next bottleneck:

| candidate | full-batch tok/s | speedup vs dense | GPU util | scheduler/mask | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| critical cap3 + route_all dynamic rows | 2937.411 | 0.843x | 78.882% | row-index/bucket `~20ms/step` | `{"NONE":128}` | dynamic row construction dominates |
| critical cap3 + direct CPU route rows | 2885.222 | 0.827x | 78.588% | direct CPU route `~20ms/step` | `{"NONE":128}` | CPU score copy just moves the cost |
| critical cap3 + fixed GPU bucket256 | 3136.459 | 0.887x | 87.200% | `0.676ms/step` | `{"NONE":128}` | CPU cost mostly fixed, still slower than dense |
| critical cap3 + bucket256 + graph flags | 3119.586 | 0.894x | 85.933% | `0.356ms/step` | `{"NONE":128}` | flags did not restore graph coverage |

The important change is that fixed bucket routing removes the large CPU
row-index cost. However, the clean run still cannot beat dense because mixed
verification remains CUDA-Graph `NONE` and the current operator still performs
extra sparse-base plus dense-correction work. The next implementation target is
therefore not another CPU route-row variant. It is either:

1. make fixed-bucket mixed verification truly CUDA-Graph capturable; or
2. implement a fused `down_proj` sparse-base plus selected dense correction
   kernel that avoids the current two-pass Linear path; or
3. find a stronger selector that uses far fewer corrected rows while preserving
   paired accuracy on a larger gate.

## 2026-06-29 Update: Low-Residual GateUp RiskCap2 Bucket4 Recheck

The slowdown wrapper previously did not locally expand
`lowresidual_gateup_riskcap2`, so its child command could forward wrapper
defaults as explicit matrix-runner overrides. That is now fixed in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_sr24_slowdown_breakdown.py
```

The generated command now uses the intended low-residual configuration:
`gate_up_proj=16-31`, `low_confidence`, threshold `0.8`, min prefix `2`,
max residual draft rows `2`, `low_confidence_cap_by_risk=1`, and the caller's
bucket override.

The old bucket4 clean result showed a positive SR24 total-throughput speedup,
but a same-configuration recheck on the current tree did not reproduce it:

| run | requests | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | SR24 accepted draft/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old bucket4 clean | 64 | 2338.823 | 2638.117 | 1.128x | 3489.747 | 3678.525 | 1.054x | 2.001 |
| current bucket4 clean recheck | 64 | 2342.977 | 2285.840 | 0.976x | 3523.740 | 3426.795 | 0.972x | 1.734 |
| current seven-part clean | 128 | 2781.358 | 2543.906 | 0.915x | 3201.505 | 3130.366 | 0.978x | 1.718 |

Current paths:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_bucket4_clean_verify_bs64_math256_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowresidual_gateup_riskcap2_bucket4_breakdown_rerun_20260629
```

The current seven-part breakdown says:

| part | current evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean low-sync CPU-sync ablation: `0.523ms/step`; diagnostic exact path can show `15-49ms/step` | exact routing/sync counters are expensive, but the clean path is already sub-ms |
| base sparse linear | diagnostic gate_up16-31 sparse base `1.096ms/call`, about `537.5` rows/call | sparse base is the dominant measured Linear cost |
| residual correction | dense-row correction `0.183ms/call`, bucket rows `4/call` | correction is secondary here because bucket4 is tiny |
| gather/scatter | `0.013ms/call` | not the first bottleneck |
| routing statistics | diagnostic draft residual/base `7916/8012`, non-draft `1991/3599`, bucket fill `1.000` | cap keeps correction small, but about half of draft rows are still classified residual before bucket cap |
| CUDA Graph | clean current: `FULL=124,NONE=163,PIECEWISE=1` in the 128-request run; short CPU-sync ablation: `FULL=6,NONE=25,PIECEWISE=1` | many `NONE` steps remain on this path |
| GPU util | clean 128-request run: dense `92.6%`, SR24 `94.9%`; 64-request clean recheck: SR24 `89.4%` | not an idle-GPU-only problem |

CPU-sync ablation path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowresidual_gateup_riskcap2_bucket4_cpu_sync_ablation_20260629
```

Key rows:

| variant | total tok/s | full-batch tok/s | scheduler/mask evidence | read |
| --- | ---: | ---: | --- | --- |
| low-sync stats on | 1738.092 | 2585.752 | `0.523ms/step` | clean wall-clock mask build is small |
| low-sync stats off | 1742.632 | 2588.385 | stats disabled | runtime stats are not the main slowdown |
| sync mask state | 1803.412 | 2766.613 | `39.690ms/step`, mostly mask-state sync | timing is noisy but confirms explicit sync is costly |
| sync heavy | 1518.442 | 2150.669 | `49.254ms/step`, request loop `49.107ms` | disabling reduced CPU sync is a clear negative |
| low-sync GPU counts | 1730.051 | 2557.849 | `33.581ms/step` diagnostic readout | GPU-count diagnostic still adds synchronization overhead |

Current conclusion: reducing CPU synchronization is necessary but no longer
sufficient. The clean path already avoids the worst sync cost. The current
slowdown comes from a combination of:

1. no reliable accepted-length improvement in the current recheck;
2. many CUDA Graph `NONE` steps in SR24 serving;
3. a non-free gate_up sparse-base pass (`~1.1ms/call`) plus a residual path,
   even when the residual bucket itself is tiny.

Next optimization should therefore not be another blind threshold sweep. The
highest-value next steps are:

1. make the current low-residual path graph-stable and verify that the
   accepted draft/step does not regress versus dense;
2. reduce or eliminate base sparse work for rows that are not producing useful
   accepted tokens;
3. if keeping this operator shape, focus on a selector that improves accepted
   draft/step without increasing residual rows, otherwise move to a fused
   sparse-base-plus-correction kernel.

### Follow-Up: Acceptance Trace Says The Bucketed Selector Is Misaligned

Focused trace root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowresidual_gateup_riskcap2_bucket4_trace_gsm8k20_20260629
```

Offline acceptance analysis:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lowresidual_gateup_riskcap2_bucket4_trace_gsm8k20_20260629/acceptance_trace_analysis/report.md
```

This run is a small GSM8K-20 quality/debug trace, not a final accuracy claim.
It is useful because it aligns the SR24 residual route with actual verifier
acceptance labels for each draft position.

The key result is that the requested residual route and the effective route
after bucket capping diverge sharply:

| metric | value | read |
| --- | ---: | --- |
| requested residual fraction | 0.4970 | the low-confidence policy wants about half the draft rows corrected |
| effective residual fraction | 0.3744 | bucket4 drops many requested rows before the operator runs |
| accepted requested-base fraction | 0.2002 | even the requested low-confidence rule leaves 20% of accepted tokens base-only |
| accepted effective-base fraction | 0.6024 | after bucket4, 60% of actually accepted tokens use the base-only SR24 path |
| rejected requested-base fraction | 0.2318 | first-reject logits are also often base-only |
| mean accepted tokens/step | 1.4064 | accepted length is not high enough to hide routing/operator overhead |
| mean requested residual rows/step | 3.9758 | the intended route is already near half of K=8 before bucket truncation |

By position, positions 1 and 2 are always residual because
`min_prefix_residual=2`, but accepted tokens at later positions are mostly
base-only:

| draft position | accepted tokens | accepted base-only fraction | reached base-only fraction |
| ---: | ---: | ---: | ---: |
| 3 | 139 | 0.8777 | 0.7256 |
| 4 | 67 | 0.8806 | 0.7410 |
| 5 | 29 | 0.7241 | 0.5373 |
| 6 | 13 | 0.7692 | 0.4483 |
| 7 | 8 | 0.6250 | 0.6154 |
| 8 | 4 | 1.0000 | 0.5000 |

The prefix projection shows the speed/quality conflict directly:

| projected prefix residual len | residual fraction | accepted base-only fraction | rejected base-only fraction | mean residual rows/step |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.4970 | 0.2002 | 0.2318 | 3.9758 |
| 3 | 0.5869 | 0.0897 | 0.0832 | 4.6955 |
| 4 | 0.6756 | 0.0362 | 0.0269 | 5.4051 |
| 5 | 0.7627 | 0.0172 | 0.0077 | 6.1019 |
| 8 | 1.0000 | 0.0000 | 0.0000 | 8.0000 |

So the current bucket4 low-residual point is fast only by leaving many useful
accepted rows uncorrected. Fixing that with a prefix rule immediately pushes
the residual fraction toward 60%-70%, where the current two-pass
`sparse base + dense correction` operator is already outside the favorable
microbench region.

A critical-prefix offline projection gives a more promising selector shape, but
it still does not solve the operator problem by itself:

| critical-base target | projected selector | residual fraction | accepted base-only fraction | rejected base-only fraction | mean residual rows/step |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.20 | prefix0, extra1, threshold0.7 | 0.3656 | 0.0906 | 0.1844 | 2.9248 |
| 0.10 | prefix0, extra2, threshold0.8 | 0.4604 | 0.0426 | 0.0845 | 3.6828 |
| 0.05 | prefix4, extra0, threshold0.7 | 0.5156 | 0.0299 | 0.0499 | 4.1248 |
| 0.02 | prefix4, extra2, threshold0.5 | 0.6275 | 0.0082 | 0.0179 | 5.0204 |

This changes the next-step diagnosis:

1. The slowdown is not primarily GPU underutilization; clean SR24 runs still
   report roughly 89%-95% GPU util.
2. It is not only CPU scheduling/mask build; exact diagnostics can be very
   expensive, but the clean low-sync path is already sub-ms per step.
3. The current selector is misaligned with useful tokens once bucket capping is
   applied: many accepted tokens are effectively base-only.
4. Making the selector quality-safe increases residual rows into the region
   where the current two-pass operator is not profitable.

The practical optimization path should therefore be:

1. keep using acceptance traces to choose a low-risk selector such as
   critical-prefix rather than blind low-confidence bucket caps;
2. avoid computing sparse base for rows that will be corrected, or fuse the
   sparse-base and dense-correction work for the selected rows;
3. restore CUDA Graph coverage for the fixed-bucket mixed path before another
   full matrix, because many `NONE` graph steps remain even when CPU sync is
   reduced.

### Follow-Up: Active-Scaled Bucket And Critical-Prefix Extra2

Implementation update:

- Added `SPECLINK_SR24_RESIDUAL_BUCKET_SCALE_BY_ACTIVE=1` to
  `vllm/vllm/speclink_sr24.py`.
- Added `--sr24-residual-bucket-scale-by-active` to both runners.
- Added preset `criticalprefix_extra2_gateup_scaledbucket` to both runners.

The new preset is trace-driven:

```text
target leafs: gate_up_proj
residual layers: gate_up_proj=16-31
policy: critical_prefix
threshold: 0.8
extra_after_low: 2
min_prefix_residual: 0
bucket: 4 per active request
bonus priority: 0.5
CUDA Graph: disabled for this first correctness/routing gate
```

Why this was needed: the old bucket was global to the verifier forward, not
per request. At bs64, a bucket such as 4 or 8 is an extremely small global
correction budget and can silently drop useful draft rows. The new flag keeps
the old behavior by default but lets this candidate test a per-active-request
budget.

GSM8K-10 quality/routing gates:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix_extra2_gateup_scaledbucket_gsm8k10_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix_extra2_gateup_scaledbucket_bonus05_gsm8k10_20260629
```

| candidate | GSM8K-10 exact | pair reg | pair imp | requested residual frac | effective residual frac | accepted effective-base frac | rejected requested-base frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| critical-prefix extra2, bonus priority 4.0 | 0.7000 | 0 | 0 | 0.4652 | 0.3750 | 0.3662 | 0.0759 |
| critical-prefix extra2, bonus priority 0.5 | 0.8000 | 0 | 1 | 0.4586 | 0.4217 | 0.1565 | 0.0862 |

Lowering bonus priority improved the actual bucketed draft correction: accepted
effective-base draft tokens dropped from `36.62%` to `15.65%`. This confirms
that the default high bonus priority was consuming capped bucket slots that
should have gone to useful accepted draft rows.

Short bs64 serving check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix_extra2_gateup_scaledbucket_bs64_math256_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix_extra2_gateup_scaledbucket_rowrouted_bs64_math256_20260629
```

Config: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, K=4,
max new tokens 256, fixed 64 requests.

| method | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3756.528 | 5248.800 | 1.4940 | 82.778% | same-run dense reference |
| critical-prefix extra2 scaled bucket | 2900.516 | 4217.977 | 1.5608 | 83.091% | quality/routing safer, but still slower |
| + row-routed gate_up | 1229.248 | 2044.521 | 1.5519 | 26.885% | much slower; small routed GEMMs underutilize GPU |

This iteration narrows the path:

1. The safer critical-prefix selector can reduce accepted base-only risk, and
   lowering bonus priority is important when using capped buckets.
2. The speed problem remains: with enough correction to protect useful draft
   rows, the current two-pass operator is slower than dense.
3. Naively row-routing gate-up rows is worse, not better, because it fragments
   work into small gather/GEMM/scatter kernels and drops GPU utilization.

The next useful speed work is therefore not row-routed Python/Torch assembly.
It is either a graph-stable fused/grouped operator for the mixed gate-up path,
or a selector that gets accepted effective-base risk lower without increasing
corrected rows beyond roughly four per request.

### 2026-06-29 Update: Precision-Safe Prefix4/Bucket5 Graph-On Check

Focused run:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_prefix4_bucket5_graphon_bs64_math256_20260629/clean_serving
```

Command shape: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency,
EAGLE3 K=8, max new tokens 256, 128 fixed requests. The run used the current
precision-safe `criticalprefix_extra2_gateup_scaledbucket` preset with
`min_prefix_residual=4`, bucket size 5 per active request, and explicit
`--sr24-allow-cudagraph --sr24-cudagraph-bucket`.

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3187.056 | 2608.758 | 1.7241 | 92.120% | `FULL=124,NONE=163,PIECEWISE=1` |
| `speclink_t08` prefix4/bucket5 graph-on | 2909.680 | 2527.459 | 1.7997 | 93.346% | `FULL=183,NONE=41` |

This is the cleanest current answer to "where is it slow":

| part | current read |
| --- | --- |
| scheduler / mask build | Not the main bottleneck in clean serving. The earlier exact diagnostic measured up to `7.924ms/step`, but almost all of that was sync-heavy request-loop instrumentation. In the clean prefix4/bucket5 run, runtime stats were disabled and graph capture was restored. |
| base sparse linear | Still the largest localized GPU-side cost in instrumented rows: gate/up layers 16-31 were about `1.040ms/call` for the selective path. Sparse base alone has headroom versus dense, but only before correction is added. |
| residual correction | The quality-safe selector increases corrected rows. In instrumented rows, selective correction was about `0.191ms/call`; all-corrected correction reached `0.581ms/call`. This correction is the main reason the sparse-base saving does not become throughput speedup. |
| gather/scatter | Measured at `0.015ms/call` in the selective path and `0.035ms/call` in all-corrected. It is not the first optimization target. |
| routing statistics | Earlier exact traces showed buckets nearly full (`bucket_fill ~= 0.98`) and many rows corrected. The current precision-safe preset deliberately protects the first four draft rows, which improves quality but pushes residual work into the range where the current two-pass operator is not profitable. |
| CUDA Graph | The graph-on run proves graph coverage can be restored for the safe preset: `speclink_t08` has fewer `NONE` steps than dense (`41` vs `163`). Since it is still slower, CUDA Graph miss is no longer the primary remaining bottleneck. |
| GPU util | Both rows are busy: dense `92.120%`, `speclink_t08` `93.346%`. This is inefficient useful work, not idle GPU. |

Current conclusion: the slow path is the mixed gate/up operator shape. The code
does sparse base for a large batch of rows and then performs dense residual
correction for selected rows. Once the selector is made quality-safe, the
correction fraction is high enough that the extra correction work eats the 2:4
base saving. The next optimization should therefore focus on either reducing
corrected rows without reintroducing accepted base-only errors, or replacing
the two-pass `sparse base + dense correction` implementation with a fused or
grouped operator. More CPU scheduler tuning or isolated gather/scatter cleanup
is unlikely to move the end-to-end result first.

### 2026-06-29 Update: CPU-Sync Ablation And Wrapper Fix

CPU-sync ablation root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_bs64_math256_20260629
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 256, 64 fixed requests. This run used the breakdown wrapper's
CPU-sync variants. Because the wrapper still overwrote explicit graph flags
before the fix below, these rows are useful for CPU-sync attribution but not
for graph-on throughput comparison.

| variant | full-batch tok/s | total tok/s | avg GPU util | scheduler/mask ms/step | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| low-sync stats on | 3094.564 | 1763.156 | 83.222% | 0.373 | `NONE=192` | clean low-sync scheduler path is sub-ms |
| low-sync stats off | 3104.873 | 1771.294 | 84.222% | n/a | `NONE=192` | disabling runtime stats barely changes full-batch tok/s |
| sync mask state | 3050.184 | 1731.516 | 81.053% | 10.317 | `NONE=192` | per-step mask-state sync is visible but not the main clean path |
| sync-heavy exact routing | 2024.319 | 1490.030 | 64.818% | 14.558 | `NONE=128` | exact per-request routing sync badly pollutes diagnostics |
| low-sync GPU-count breakdown | 3127.490 | 2169.875 | 86.067% | 14.288 | `NONE=128` | GPU counters preserve routing statistics with much less util loss than exact CPU routing |

Conclusion from this ablation:

1. Reducing CPU sync is necessary for trustworthy diagnostics, but it does not
   by itself make `speclink_t08` faster than dense.
2. The clean low-sync scheduler path is already small (`0.373ms/step` in this
   run), while sync-heavy exact routing can create a false bottleneck
   (`14.558ms/step`, GPU util only `64.818%`).
3. Future breakdown rows should prefer low-sync stats or GPU-count breakdown
   and should not use exact CPU routing as an end-to-end throughput reference.

Wrapper bug fixed in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_sr24_slowdown_breakdown.py
```

The wrapper now captures explicit preset overrides for graph/runtime flags and
restores them after local preset expansion. Dry-run confirmed the child matrix
command now preserves:

```text
--sr24-dynamic-auto-cudagraph
--no-sr24-force-cudagraph-none-for-mixed
--sr24-cudagraph-bucket
--sr24-allow-cudagraph
```

Wrapper graph-on verification root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_wrapper_graphon_gate_bs64_math256_20260629
```

This short 64-request gate verified that the fixed wrapper can now produce
graph-on `speclink_t08` rows:

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3484.825 | 2337.452 | 1.7337 | 88.714% | `FULL=115,NONE=76,PIECEWISE=1` |
| `speclink_t08` | 2536.547 | 2177.736 | 1.4680 | 84.750% | `FULL=51,NONE=2` |

Use this root as wrapper validation, not as the final throughput comparison:
the short fixed-request run had lower accepted draft length for `speclink_t08`
than the earlier 128-request graph-on run. The important result is that
orchestrated graph-on breakdowns now work and can be used for the next operator
experiments.

### Current Optimization Direction

The direction should switch from controller-only sweeps to operator-focused
breakdown and implementation work.

What is not the main slowdown now:

1. Scheduler/mask build is not the clean serving bottleneck. Low-sync rows are
   sub-ms per step (`0.373ms` to `0.816ms` in the current checks). Exact CPU
   routing can create `10-15ms/step` overhead, but that is diagnostic pollution.
2. Gather/scatter is not the first target. The measured selective path was
   `0.015ms/call`, much smaller than sparse base and residual correction.
3. CUDA Graph miss is no longer sufficient as an explanation. The safe
   prefix4/bucket5 run restored graph coverage for `speclink_t08`
   (`FULL=183,NONE=41`) and still did not beat dense.
4. GPU idle is not the issue in the clean path. Dense and `speclink_t08` both
   run around `92-93%` average utilization in the clean graph-on check.

What is slow:

1. Sparse base gate/up remains a large GPU-side cost. It is faster than dense
   in isolation, but it is still a full extra operator pass over many rows.
2. Dense residual correction erases most of the 2:4 base saving when quality
   protection forces many rows to be corrected. In the current quality-safe
   selector, the protected prefix and bucket rows push the residual fraction
   into the range where the two-pass operator is not profitable.
3. `all_corrected_24` confirms this structurally: even the best measured exact
   backend (`torch_sparse` residual, explicitly requested) remains below dense,
   because it performs sparse base plus residual work instead of replacing dense
   with one cheaper fused path.

Next implementation work:

1. Keep low-sync diagnostics as the default measurement mode; use exact CPU
   routing only for short localization runs.
2. For speed, target the mixed gate/up operator, not the scheduler first:
   implement a graph-stable fused/grouped path that avoids doing sparse base
   work for rows that will be overwritten by dense correction, without
   fragmenting into tiny GEMMs.
3. In parallel, search for a lower residual-row selector that keeps the paired
   GSM8K/math regression gate clean. The safe preset currently protects too
   many rows for the existing two-pass implementation.
4. Treat `torch_sparse` residual for `all_corrected_24` as an explicit ablation
   only. It should not be hidden behind defaults because startup memory can OOM
   on the 32GB GPU.

### 2026-06-29 Update: User-Requested Slowdown Breakdown

Current breakdown root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_slow_breakdown_bs64_math_20260629
```

Main report:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_slow_breakdown_bs64_math_20260629/seven_part_report/report.md
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 256 for clean serving, and the current quality-safe
`down0_15_fixedprefix4_directcslt` preset. Clean rows use graph-capable serving;
instrumented rows use eager CUDA-event timing and are diagnostic only.

| row | full-batch tok/s | total tok/s | avg GPU util | CUDA Graph | read |
| --- | ---: | ---: | ---: | --- | --- |
| dense baseline | 3479.615 | 2335.073 | 89.929% | `FULL=115,NONE=76,PIECEWISE=1` | reference |
| `speclink_t08` | 3460.648 | 2308.471 | 88.071% | `FULL=115,NONE=76,PIECEWISE=1` | essentially tied/slightly slower |
| `all_corrected_24` | 3482.564 | 2336.385 | 89.357% | `FULL=115,NONE=76,PIECEWISE=1` | dense-fastpath control is tied |

CPU-sync ablation in the same root:

| variant | dense full-batch tok/s | `speclink_t08` full-batch tok/s | SR24 / dense | avg GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| low-sync stats on | 3488.198 | 3510.591 | 1.006x | 87.357% | runtime stats do not explain the gap |
| low-sync stats off | 3548.783 | 3455.533 | 0.974x | 87.714% | disabling stats does not unlock speed |
| sync mask state | 3486.658 | 3462.960 | 0.993x | 87.714% | explicit mask-state sync is tolerable here |
| sync-heavy exact routing | 3487.898 | 3415.111 | 0.979x | 88.286% | exact routing hurts and should remain diagnostic |
| low-sync GPU-count breakdown | 3486.322 | 3447.808 | 0.989x | 88.000% | GPU-count stats are safe enough for routing attribution |

The requested component table now reads as follows:

| part | measured result | conclusion |
| --- | --- | --- |
| scheduler / mask build | For fixed-prefix route-all, runtime stats show about `32ms/step`, almost all in `scheduler_mixed_row_indices_wall_cpu_ms`; exact diagnostic shows `12.201ms/step`, mostly request routing. | Current fixed-prefix route-all is not just a controller issue: it pays a real per-step row-index construction cost. Exact routing makes it worse. |
| base sparse linear | Instrumented `speclink_t08`: `0.957ms/call`, about `95` base rows/call. `all_corrected_24`: `0.586ms/call`, about `522` rows/call. | Sparse base remains the largest measured GPU component in the selective path. |
| residual correction | `speclink_t08`: dense correction `0.111ms/call`, about `144` dense rows/call. `all_corrected_24`: full residual dense GEMM `0.297ms/call`. | Correction is secondary for selective `t08`, but large enough to erase sparse-base savings when correction fraction rises. |
| gather/scatter | Route-all gather/scatter is about `0.017ms/linear`; base/dense gathers and copies are each about `0.004-0.005ms`. | Assembly is not the first bottleneck in this configuration. |
| routing statistics | `speclink_t08` diagnostic: draft residual/base `6936/6936`, non-draft residual/base `3558/0`, correction fraction `0.602`. GPU-count row: draft residual/base `24268/24268`, non-draft residual/base `9855/0`. | The quality-safe fixed-prefix/non-draft-all policy corrects too many rows for the current two-pass operator. |
| CUDA Graph | Clean dense, `speclink_t08`, and `all_corrected_24` all show roughly the same `FULL=115,NONE=76,PIECEWISE=1`. | CUDA Graph coverage is not the differentiator in this run. |
| GPU util | Clean dense `89.929%`, `speclink_t08` `88.071%`, `all_corrected_24` `89.357%`. | The GPU is busy; the problem is inefficient useful work, not idle time. |

Operator microbench from the same root confirms the row-fraction problem:

| shape rows/out/in | residual fraction | dense graph ms | sparse base graph ms | current mixed graph ms | read |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/28672/4096 | 0.125 | 0.539 | 0.353 | 0.555 | gate/up-like mixed path already loses to dense |
| 512/28672/4096 | 0.250 | 0.540 | 0.353 | 0.621 | correction erases sparse gain |
| 512/4096/14336 | 0.125 | 0.291 | 0.166 | 0.266 | down-proj-like path can win at low residual fraction |
| 512/4096/14336 | 0.500 | 0.291 | 0.166 | 0.362 | down-proj also loses once residual fraction is high |

Updated conclusion:

1. The current fixed-prefix route-all version is slow for two concrete reasons:
   per-step row-index construction for residual/base rows, and a two-pass
   sparse-base plus dense-correction operator at a high correction fraction.
2. The next optimization should not be another high-level controller sweep.
   It should first remove the `nonzero`-based row-index construction for
   fixed-prefix/non-draft-all and then remeasure.
3. After that, the real speed path is still operator-side: either avoid sparse
   base work on rows that will be overwritten, or reduce the correction
   fraction while preserving the paired accuracy gate.

### 2026-06-29 Update: Fixed-Prefix Route-Row Fastpath

Code changed:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/vllm/vllm/speclink_sr24.py
```

New path:

- `SPECLINK_SR24_FIXED_PREFIX_ROUTE_FASTPATH=1` by default.
- It only applies to SR24 `fixed_prefix + selective_non_draft_policy=all +
  route_all_residual_rows`, with no early-dense override and with the residual
  draft budget not smaller than the fixed prefix.
- It keeps the same residual mask semantics, but avoids
  `_compute_mixed_row_indices(... nonzero(mask) ...)` by directly constructing:
  - base rows = each request's draft positions `[prefix, valid)`
  - residual rows = complement of those base rows

Validation:

```text
python3 -m py_compile vllm/vllm/speclink_sr24.py
conda run --no-capture-output -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

Both passed. A dedicated GPU check also matched the new fixed-prefix route rows
against the old mask/nonzero rows:

```text
fixed_prefix_route_fastpath_equivalence=ok rows 36 residual 24 base 12
```

The first implementation tried to form the residual complement on GPU through a
sort/complement helper. That was wrong for performance: it moved the cost from
`mixed_row_indices` to direct route-row construction and dropped bs64 math
throughput to `1618.316` full-batch tok/s. That root is kept only as a failed
ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_route_fastpath_gate_bs64_math256_20260629
```

The corrected v2 uses a CPU list complement for the small per-step row set and
copies the static row buffers to GPU. Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_route_fastpath_gate_v2_bs64_math256_20260629
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 256, 64 fixed requests, runtime stats enabled so scheduler
timing is visible.

| method | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | scheduler mask wall | mixed row-index wall | route-row wall | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense baseline | 3484.570 | 2336.572 | 1.7337 | 89.286% | n/a | n/a | n/a | `FULL=115,NONE=76,PIECEWISE=1` |
| `speclink_t08` + fastpath v2 | 3478.324 | 2331.059 | 1.7342 | 90.857% | 0.240ms/step | 0.0008ms/step | 0.176ms/step | `FULL=115,NONE=76,PIECEWISE=1` |

What changed relative to the previous fixed-prefix breakdown:

- `scheduler_mixed_row_indices_wall_cpu_ms_per_step`: about `32ms/step` ->
  `0.0008ms/step`.
- Total scheduler mask wall: about `32ms/step` -> `0.240ms/step`.
- End-to-end `speclink_t08` returned to dense parity instead of being
  scheduler-bound.

Updated optimization state:

1. The fixed-prefix route-all scheduler-row-index issue is now mostly removed
   for the current quality-safe path.
2. This does not achieve the final speed goal. `speclink_t08` is still only
   dense parity (`3478` vs `3485` full-batch tok/s in this gate), not `1.2x`.
3. The next bottleneck is again the operator work: `sparse base + dense
   correction` at a high correction fraction. The next implementation should
   avoid sparse base work on rows overwritten by dense correction, or reduce
   corrected rows without reintroducing the paired accuracy regression.

Default clean preset check after the same fix:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_route_fastpath_cleanpreset_bs64_math256_20260629
```

| method | full-batch tok/s | total tok/s | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | --- |
| dense baseline | 3488.701 | 2337.724 | 90.571% | `FULL=115,NONE=76,PIECEWISE=1` |
| `speclink_t08` | 3482.542 | 2334.537 | 89.000% | `FULL=115,NONE=76,PIECEWISE=1` |

Dense-fallback operator ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_route_fastpath_fallback05_bs64_math256_20260629
```

### 2026-06-29 Update: Planner Override and Base-Only Eager-Safe Fix

Code changed:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.py
```

Two runner issues were fixed before trusting the next planner ablations:

1. SR24 preset application did not preserve route/planner overrides such as
   `--sr24-route-dense-fallback-fraction`. A command requesting fallback
   `0.25` still ran with the preset's `0.9`. The override whitelist now
   preserves route-all, route-reuse, route-contiguous, dense-fallback,
   adaptive-dense-fallback, and Triton route-assembly flags in both throughput
   and lm-eval runners.
2. At this point in the iteration, `base_only_24 + torch_sparse` was moved to
   eager-safe execution in both runners. The default vLLM/Inductor compile path
   can trace into the PyTorch semi-structured sparse custom kernel during
   startup profiling and fail with:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

Validation:

```text
python3 -m py_compile \
  examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py \
  examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.py
```

Focused ablation roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_route_assembly_fastpath_bs64_math256_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_dense_fallback025_overridefix_bs64_math256_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_fourmode_bs64_math256_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_eagersafe_bs64_math256_20260629
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, K=8,
max new tokens 256, 64 fixed requests, current
`down0_15_fixedprefix4_directcslt` quality-safe preset unless noted.

| run | method/config | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | CUDA Graph | read |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| clean preset | dense baseline | 3488.701 | 2337.724 | 1.7342 | 90.571% | `FULL=115,NONE=76,PIECEWISE=1` | same-root reference |
| clean preset | `speclink_t08` | 3482.542 | 2334.537 | 1.7337 | 89.000% | `FULL=115,NONE=76,PIECEWISE=1` | dense parity, not speedup |
| route assembly | `speclink_t08 + --sr24-triton-route-assembly` | 3480.290 | 2332.744 | 1.7337 | 91.214% | `FULL=115,NONE=76,PIECEWISE=1` | no useful gain; assembly is not first bottleneck |
| fallback 0.25 after override fix | `speclink_t08`, actual fallback `0.25` | 3476.751 | 2333.544 | 1.7276 | 89.500% | `FULL=115,NONE=76,PIECEWISE=1` | true high-residual dense fallback avoids neither parity nor slowdown |
| four-mode current | dense baseline | 3487.360 | 2338.147 | 1.7342 | 88.643% | `FULL=115,NONE=76,PIECEWISE=1` | reference |
| four-mode current | `all_corrected_24` dense-fastpath | 3485.569 | 2338.829 | 1.7337 | 89.857% | `FULL=115,NONE=76,PIECEWISE=1` | exact dense-equivalent control; throughput tied |
| four-mode current | `speclink_t08` | 3481.874 | 2332.422 | 1.7337 | 88.571% | `FULL=115,NONE=76,PIECEWISE=1` | still slightly below dense |
| base-only eager-safe | `base_only_24`, eager-safe | 3230.384 | 2185.245 | 1.6645 | 83.267% | `NONE=128` | runs stably; slower here because it is down0-15 only, eager, and lowers acceptance |

Current read:

1. `base_only_24` has two separate issues. The default compile path is
   unstable for the PyTorch semi-structured sparse custom kernel, so it must be
   kept eager-safe for now. In the current down0-15-only preset, eager-safe
   base-only is slower than dense and has lower accepted draft tokens
   (`1.6645` vs dense `1.7342`) plus lower GPU util (`83.3%` vs `88.6%`).
   This row is not the full-model sparse upper bound; it is the current
   quality-safe scope's base-only diagnostic.
2. `all_corrected_24` is optimized only as the dense-equivalent fastpath
   control. The real no-fastpath base+residual operator remains a diagnostic,
   not a speed path.
3. `speclink_t08` remains dense parity after scheduler-row fastpath and after
   route-assembly/fallback ablations. The next speed work must either lower
   correction fraction without accuracy regression or implement a genuinely
   fused/grouped sparse-base + residual operator; small route assembly changes
   and high-residual dense fallback are insufficient.

### 2026-06-29 Update: Base-Only Graph-Only Recovery

Follow-up code change:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py
```

The throughput runner now distinguishes two compile paths for
`base_only_24 + torch_sparse`:

- default vLLM/Inductor compile remains disabled because it still fails with
  the lazy custom-kernel storage error above;
- SR24 graph-only compile is enabled by default for base-only and uses:

```json
{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}
```

The old eager-safe diagnostic is still available with:

```text
--no-sr24-base-only-allow-compile
```

Validation:

- a minimal GPU `torch.compile` probe over `_semi_structured_linear()` and the
  `speclink_sr24::cslt_linear` custom op succeeded;
- `sr24_baseonly_graphonly_default_smoke_20260629` confirmed the default
  base-only command now includes the SR24 graph-only `--compilation-config`;
- the same-root dense/base-only run below completed without failures.

Focused roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_allowcompile_graphonly_bs64_math256_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_graphonly_default_smoke_20260629
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_vs_baseonly_graphonly_bs64_math256_20260629
```

Same-root dense vs graph-only base-only result:

| method | full-batch tok/s | total tok/s | accepted draft/step | acceptance % | avg GPU util | CUDA Graph | storage/dense |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| dense baseline | 3487.075 | 2339.254 | 1.7342 | 21.677% | 88.571% | `FULL=115,NONE=76,PIECEWISE=1` | 1.000 |
| `base_only_24`, graph-only | 3307.729 | 2349.743 | 1.6602 | 20.752% | 91.143% | `FULL=126,NONE=2` | 0.625 |

Updated base-only answer:

1. The original base-only problem had a real compile-mode component. Default
   vLLM/Inductor compile is unsafe for the semi-structured sparse path, but the
   SR24 graph-only config works and restores CUDA Graph coverage.
2. With graph-only coverage, `base_only_24` is not GPU-idle. GPU utilization is
   about `91%`, and total tok/s is slightly above dense in this fixed-request
   run.
3. The reason full-batch tok/s remains below dense is accepted length:
   accepted draft tokens/step drops from `1.7342` to `1.6602`. The next
   base-only-related work is therefore quality/acceptance, not GPU
   utilization.
4. This makes `base_only_24` usable again as a sparse upper-bound diagnostic,
   but not as the final method because base-only changes outputs enough to
   lower speculative acceptance and accuracy.

This set `SPECLINK_SR24_ROUTE_DENSE_FALLBACK_FRACTION=0.5`, so when the
corrected/dense row fraction is high the routed path falls back to one full
dense GEMM instead of running a small sparse-base GEMM plus dense correction.

| method | full-batch tok/s | total tok/s | avg GPU util | read |
| --- | ---: | ---: | ---: | --- |
| dense baseline | 3552.259 | 2354.632 | 86.714% | reference |
| `speclink_t08`, fallback=0.5 | 3521.740 | 2339.470 | 87.357% | still below dense |

Read: dense fallback avoids the worst small sparse-base shape, but it only
collapses the path back toward dense. It cannot produce the requested 1.2x by
itself. The remaining speed path needs either a better grouped/fused sparse
operator for the routed base rows or a lower correction fraction that still
passes the paired accuracy gate.

### 2026-06-29 Update: User-Requested Seven-Part Breakdown

Requested breakdown root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_down015_bs64_math256_20260629
```

Primary reports:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_down015_bs64_math256_20260629/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_down015_bs64_math256_20260629/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_down015_bs64_math256_20260629/component_microbench/summary.md
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 256, current `down0_15_fixedprefix4_directcslt` SR24 preset.

Clean serving results:

| method | full-batch tok/s | total tok/s | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | --- |
| dense baseline | 3482.518 | 2336.101 | 88.571% | `FULL=115,NONE=76,PIECEWISE=1` |
| `base_only_24` | 3297.239 | 2376.506 | 92.286% | `FULL=126,NONE=2` |
| `all_corrected_24` | 3489.273 | 2339.537 | 88.500% | `FULL=115,NONE=76,PIECEWISE=1` |
| `speclink_t08` | 3483.950 | 2325.389 | 89.714% | `FULL=115,NONE=76,PIECEWISE=1` |

Seven-part read:

1. Scheduler / mask build is not the clean-path bottleneck. The low-sync row
   reports `0.240 ms/step` and `speclink_t08` is `3550.745` full-batch tok/s
   against same-root dense `3529.892`.
2. The sync-heavy diagnostic path is harmful: scheduler mask wall time rises to
   `33.119 ms/step`, almost all from request routing loop (`32.999 ms/step`),
   and `speclink_t08` drops to `3422.563` full-batch tok/s. This is an
   ablation target, not the serving path.
3. Routing statistics show why the current fixed-prefix route has limited
   upside: draft residual/base rows are `24308/24308`, so half of draft rows
   still need residual correction; non-draft residual/base rows are `9865/0`,
   so all bonus/non-draft rows are corrected.
4. Linear diagnostics localize the GPU-side cost to sparse base rather than
   gather/scatter. In the routed `speclink_t08` diagnostic row, base sparse is
   `0.946 ms/call`, residual dense correction is `0.111 ms/call`, and
   route-all gather/scatter is only `0.017 ms/linear`.
5. CUDA Graph coverage is not worse than dense in clean serving for
   `speclink_t08`; both have `FULL=115,NONE=76,PIECEWISE=1`.
6. GPU utilization is similar to dense (`89.7%` vs `88.6%`), so the current
   problem is inefficient useful work while the GPU is busy, not an idle GPU.
7. Microbenchmarks agree with this: sparse base alone is faster than dense, but
   once residual fraction reaches the current routed row fraction, correction
   and assembly erase the sparse benefit.

Current conclusion: the slow path is not primarily scheduler/mask build,
gather/scatter, or GPU underutilization. The current `speclink_t08` is dense
parity because it routes too many rows through a two-pass sparse-base plus dense
correction operator, and the sparse base GEMM on the remaining base rows is
itself too expensive at the routed shape. The next useful optimization should
target one of two concrete issues:

- reduce the corrected row fraction while preserving paired accuracy, especially
  avoiding dense correction for all non-draft/bonus rows when it is not needed;
- replace the current two-pass routed Linear with a grouped/fused operator that
  avoids paying `0.946 ms/call` sparse base plus separate dense correction for
  the same down-projection layer.

### 2026-06-29 Follow-Up: Exact all-corrected and t08 quality gate

Exact `all_corrected_24` early-dense check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_earlydense_down015_bs64_math256_20260629_2
```

Setup: Llama-3.1-8B, `math_reasoning`, bs64 client concurrency, EAGLE3 K=8,
max new tokens 256, `down_proj=0-15`, `dense_rows@cuda`,
`--no-sr24-all-corrected-dense-fastpath`,
`--sr24-full-residual-early-dense`, static all-residual, default vLLM compile.

| method | full-batch tok/s | total tok/s | avg GPU util |
| --- | ---: | ---: | ---: |
| dense baseline | 3520.661 | 2345.053 | 87.286% |
| `all_corrected_24` early-dense hook | 3479.719 | 2335.954 | 88.286% |

Read: when the exact all-corrected hook returns dense Linear before sparse-base
dispatch, it is dense-equivalent within run noise. The no-fastpath exact sparse
operator remains slow because it pays sparse base plus dense/full residual work.
So the current practical all-corrected optimization is the early-dense/dense
fastpath; a true sparse exact speedup still needs a fused packed operator.

Current `speclink_t08` quality gate for the best speed candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_directcslt_bucket16_quality_gsm8k50_20260629/report.md
```

The candidate uses `critical_prefix@0.6`, prefix residual floor 4, extra-after-low
1, non-draft policy `bonus`, `gate_up_proj=16-31;down_proj=8-15`,
bucket16, `--sr24-bucket-dense-copy`, `--sr24-direct-cslt-linear`, dynamic-auto
CUDA Graph, and default vLLM compile. GSM8K-50 paired accuracy matches dense:

| mode | exact match | samples | pair regressions | pair improvements | avg output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense baseline | 0.7200 | 50 | 0 | 0 | 89.34 |
| `speclink_t08` | 0.7200 | 50 | 0 | 0 | 89.34 |

This makes bucket16/direct-cuSPARSELt the current quality-safe `speclink_t08`
comparison point, but it is still not the requested speed endpoint. The matching
bs64/math/max256 throughput run reaches total `2720.019` tok/s versus dense
`2319.712` (`1.173x`) and full-batch `3930.796` versus dense `3429.905`
(`1.146x`), below the `1.2x` target. The remaining gap is operator-side:
separate sparse-base and dense-correction work still costs too much for the
quality-safe residual row fraction.

Simple non-draft correction reduction is not enough. A same-setup
`predicted_full_accept` non-draft policy run:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_predfullaccept_bucket16_directcslt_bs64_math256_20260629
```

reduced non-draft residual fraction from `0.5719` to `0.0000`, but throughput
fell instead of improving:

| non-draft policy | dense total tok/s | SR24 total tok/s | dense full-batch tok/s | SR24 full-batch tok/s | accepted draft/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bonus` | 2319.712 | 2720.019 | 3429.905 | 3930.796 | 2.198 |
| `predicted_full_accept` | 2333.375 | 2665.145 | 3477.560 | 3916.066 | 2.176 |

Read: always correcting the bonus row has some cost, but removing it changes
the generated stream enough to lower accepted draft length and does not improve
net serving throughput. Do not promote `predicted_full_accept` unless a later
selector or operator change reverses this tradeoff.
