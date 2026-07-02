# SR24 Current Slowness Breakdown Direction

This note resets the next SR24/SpecLink tuning pass around an explicit
slowdown breakdown. The immediate goal is to explain where the current path is
slow before running another selector or throughput sweep.

Current reference artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWNESS_BREAKDOWN_CURRENT.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_sr24_slowdown_breakdown.py
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_20260629/seven_part_report_with_base_safe/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_explicit_breakdown_bs64_math256_20260629/seven_part_report_with_graph/report.md
```

## One-Page Current Diagnosis

The current SR24 slowdown is not mainly a draft-acceptance collapse, normal
scheduler/mask build cost, gather/scatter overhead, or idle GPU. The current
slow path is the mixed sparse-base plus residual-correction operator at the
protected-row fraction needed for quality, with CUDA Graph coverage as a
secondary but real requirement.

## 2026-06-29 Lossy Pivot Check

The latest user direction allows accuracy loss up to about 8 percentage points
and asks whether the current implementation can trade quality for speed. The
short answer is no for the current global-bucket/two-pass design.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_b12_fast_b8_16_32_64_20260629/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_b12_gsm8k50_accuracy_20260629/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_active_only_compute_critical_eager_bs64_20260629/report.md
```

The b12 critical-prefix row uses Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`,
max tokens 256, and the existing graph bucket path. It improves accepted draft
length, but not enough to overcome the mixed-operator cost at small batch:

| bs | dense total tok/s | b12 total tok/s | total speedup | dense full tok/s | b12 full tok/s | full speedup | accepted draft/step dense -> b12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 1142.993 | 859.029 | 0.752x | 1238.929 | 912.073 | 0.736x | 1.754 -> 1.944 |
| 16 | 1832.143 | 1421.647 | 0.776x | 2053.052 | 1621.107 | 0.790x | 1.809 -> 2.023 |
| 32 | 2264.968 | 2090.328 | 0.923x | 2675.556 | 2511.380 | 0.939x | 1.761 -> 2.065 |
| 64 | 2334.837 | 2707.481 | 1.160x | 3484.084 | 3978.413 | 1.142x | 1.734 -> 2.209 |

The same b12 strategy is also outside the allowed quality budget on GSM8K-50:
dense EAGLE3 flexible exact match is `0.7600`, while b12 `speclink_t08` is
`0.5400` (`-22.0 pp`, paired regressions `12`, paired improvements `1`).
Therefore smaller global buckets are not the next path even under lossy
quality constraints.

I also tested the direct "if sparse already computed an unimportant row, do not
also run dense for it" idea via the new active-only dense-compute ablation.
The naive implementation uses dynamic `nonzero()` compaction, so it is
CUDA-Graph unsafe; in eager bs64 it reaches only `2186.705` full-batch tok/s.
The conclusion is not that active-only routing is wrong; it means the routing
has to be encoded as a graph-safe fixed-layout or custom kernel, not as Python
dynamic compaction inside Linear hooks.

Next implementation target:

1. Use a fixed-capacity row-routing data format: per-position/per-request row
   buckets, active bitmasks, and stable device buffers for CUDA Graph replay.
2. Replace all-base-first plus dense overwrite with disjoint execution:
   important rows go dense; unimportant rows go 2:4 sparse; assemble once.
3. Add an occupancy-aware planner. If important rows are too few, either pack
   them across requests/layers or fall back to sparse-only/dense depending on
   measured row-count thresholds.
4. Only after the disjoint row format is in place, test two-stream overlap of
   sparse rows and dense rows. Stream overlap should be kept only if it wins
   end-to-end, because small kernels can underfill the GPU.

Latest explicit Llama-3.1-8B `math_reasoning`, bs64, EAGLE3 K=8,
max-tokens 256 breakdown:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_explicit_breakdown_bs64_math256_20260629/seven_part_report_with_graph/report.md
```

| Part | What To Measure | Current Measurement | Current Read |
| --- | --- | --- | --- |
| scheduler / mask build | residual mask, bucket rows, per-step scheduling state | clean `speclink_t08` mask wall `0.455 ms/step`; GPU-count row `0.356 ms/step`; sync-heavy exact row `6.280 ms/step` | normal serving path is not scheduler-bound; exact counters are useful only to expose sync/routing cost |
| base sparse linear | sparse base GEMM, especially Llama gate/up layers | diagnostic `speclink_t08` gate_up16-31 sparse base `0.605 ms/call`; all-corrected `0.741 ms/call` | largest localized GPU-side component |
| residual correction | dense-row/compressed residual correction GEMM | diagnostic `speclink_t08` residual correction `0.333 ms/call`; all-corrected `0.480 ms/call` | correction is large enough to erase sparse-base wins at current row fractions |
| gather/scatter | index_select/index_add_/bucket assembly | `0.036 ms/event` for `speclink_t08`; `0.028 ms/event` for all-corrected | not the first bottleneck |
| routing statistics | draft residual rows, non-draft residual rows, bucket fill | diagnostic `speclink_t08` draft residual/base `8640/4608`, non-draft residual/base `1656/1635`, draft residual fraction `0.652` | quality-safe routing protects too many rows for the current two-pass operator |
| CUDA Graph | FULL/NONE/PIECEWISE counts for dense/base-only/t08 | dense `FULL=115,NONE=76,PIECEWISE=1`; base-only `FULL=126,NONE=2`; guarded t08 `NONE=128`; dynamic-graph t08 can reach `FULL=126,NONE=2` but still stays below dense | graph coverage is necessary, but fixing it alone is not sufficient |
| GPU util | average/peak GPU utilization in clean serving | dense `88.429%`; base-only `90.833%`; clean t08 `86.938%`; GPU-count t08 `88.188%` | GPU is busy; the issue is inefficient useful work, not idle hardware |

The most important contrast is:

| Method | Full-batch tok/s | Total tok/s | Accepted draft tokens/step | GPU util | CUDA Graph | Read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| dense baseline | 3482.041 | 2334.752 | 1.734 | 88.429% | `FULL=115,NONE=76,PIECEWISE=1` | baseline |
| `base_only_24` | 3961.598 | 2790.475 | 2.027 | 90.833% | `FULL=126,NONE=2` | sparse-base upper bound is real |
| guarded `speclink_t08` | 2847.156 | 2063.403 | 1.703 | 86.938% | `NONE=128` | slow from graph miss plus mixed-operator cost |

Operator microbench explains the gap. On the dominant Llama gate/up shape
`512 x 28672 x 4096`, sparse base alone is about `0.65x` dense, but current
mixed sparse-base plus correction is already dense-parity at residual fraction
`0.0625-0.125` and becomes slower after that. Current routing is far above
that safe region, so threshold/bucket tweaks alone cannot be expected to
produce a robust `1.2x` speedup.

The latest bucket16 Triton dense-GEMM/scatter sweep is also negative for the
dominant gate/up shape:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket16_param_sweep_20260629/summary.md
```

| Shape | Residual Fraction | Best Triton / Dense | Best Triton / Bucket Delta | Read |
| --- | ---: | ---: | ---: | --- |
| `512 x 28672 x 4096` | 0.03125 | 1.424x | 1.415x | negative for gate/up |
| `512 x 28672 x 4096` | 0.0625 | 1.409x | 1.428x | negative for gate/up |
| `512 x 4096 x 14336` | 0.03125 | 0.999x | 1.028x | dense parity, still slower than bucket delta |
| `512 x 4096 x 14336` | 0.0625 | 1.000x | 1.113x | dense parity, still slower than bucket delta |

So the next optimization should not be another scalar threshold sweep or another
plain Triton tile sweep for the current bucket kernel. It should either:

1. reduce corrected rows much more aggressively while preserving paired
   accuracy, or
2. replace the two-pass sparse-base plus dense-correction path with a fused or
   grouped operator that avoids doing sparse base work for rows that will be
   corrected.

## 2026-06-29 Row-Routed Exact-Down Probe

I also checked whether a more direct row-routed MLP path can recover the lost
operator headroom by avoiding sparse-base work on rows that will be corrected.
This is the closest existing path to a fused/grouped mixed operator.

Microbench roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_exactdown_microbench_current_20260629/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_exactdown_bucket_sweep_current_20260629/summary.md
```

Setup: Llama MLP shape `rows=512`, hidden `4096`, intermediate `14336`, bf16,
CUDA Graph timing. The exact-down path computes dense gate/up and dense down
for corrected rows, sparse gate/up and sparse down for base rows, then
assembles the final hidden states.

Best rows:

| corrected rows | dense MLP graph ms | exact-down Triton assemble ms | exact-down / dense | no-final-assemble / dense | read |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.8632 | 0.8665 | 1.004x | 0.976x | too few corrected rows; dense parity |
| 32 | 0.8662 | 0.8159 | 0.942x | 0.919x | useful but far from 1.2x |
| 64 | 0.8665 | 0.8109 | 0.936x | 0.912x | useful but far from 1.2x |
| 128 | 0.8662 | 0.7488 | 0.864x | 0.839x | best measured point, about `1.16x` operator speed |
| 192 | 0.8670 | 0.7900 | 0.911x | 0.891x | starts losing the best region |

Conclusion: row-routed exact-down is a real improvement over the current
two-pass correction, but it still does not reliably reach the `dense/1.2`
target (`0.722 ms` for this shape). Even the no-final-assemble lower bound at
the best bucket is `0.839x dense`, about `1.19x`, and live serving will still
pay routing/assembly and quality-gate overhead. Treat row-routed exact-down as
a useful ablation and an implementation scaffold, not as the final speed path.

Rows=1024 changes the operator-only read, but not the serving conclusion yet:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_exactdown_rows1024_sweep_20260629/summary.md
```

At Llama MLP `rows=1024`, bucket `384` reaches dense graph `1.6939 ms` and
exact-down Triton assemble `1.3063 ms`, or `0.771x` dense. That is about a
`1.30x` operator speedup, so larger full-batch decode shapes can have enough
raw headroom. The isolated subcomponent timing still shows the work is not
free: for bucket `384`, dense-row gate/up plus down is about
`0.3979 + 0.2181 ms`, base-row sparse gate/up plus down is about
`1.0869 + 0.8837 ms`, and final assembly is about `0.0329 ms`. The gain comes
from routing rows into larger profitable dense/sparse groups, not from
gather/scatter being the dominant part.

The first bs128 live serving attempt did not reach throughput measurement:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_mlp_bucket384_bs128_math128_20260629/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_rowrouted_mlp_bucket384_bs128_math128_20260629_work/speclink_t08/bs128/rep1/server.log
```

vLLM loaded the model and attached SR24, then failed KV-cache allocation:
`storage_over_dense=1.625`, model load around `22.7 GiB`, CUDA graph estimate
around `5.44 GiB`, and available KV-cache memory around `-1.91 GiB`. Therefore
the current all-layer `dense_rows` row-routed path is capacity-limited on this
GPU at bs128 before speed can be judged. The next live serving breakdown must
either reduce SR24 storage scope, reduce graph/KV memory demand, or use a
lower-memory residual backend before claiming the rows=1024 operator result
translates to serving.

The actionable next operator direction is narrower: fuse or group the row-routed
exact-down path so it removes final assembly and keeps the base/dense GEMMs in
profitable shapes, but first run a memory-fitting serving breakdown for the
exact candidate. If the candidate cannot keep enough KV-cache memory, it is not
a deployable speed path regardless of microbench speed.

## 2026-06-29 bs128 Memory-Fitting Breakdown

The scoped quality-safe candidate fits bs128 when the all-layer `dense_rows`
path is avoided:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix_b16_bs128_breakdown_20260629/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_exact_clean_bs128_math128_20260629/report.md
```

Setup: Llama-3.1-8B, `math_reasoning`, client-side bs128, EAGLE3 K=8,
max-tokens 128, `criticalprefix4_bucket16_directcslt`,
`gate_up_proj=16-31;down_proj=8-15`, `dense_rows@cuda`, bucket16,
direct cuSPARSELt sparse base, CUDA Graph bucket enabled.

Clean serving:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| dense | 2240.098 | 3147.956 | 1.422 | 78.467% | `FULL=8,NONE=55,PIECEWISE=1` | baseline row in this root |
| `base_only_24` | 2769.278 | 3871.417 | 1.608 | 91.417% | `FULL=62,NONE=2` | base-only is not slow here |
| `speclink_t08` | 2681.520 | 3702.182 | 1.633 | 91.500% | `FULL=62,NONE=2` | close to, but not robustly above, `1.2x` full-batch |

Direct answer for `base_only_24`: in this scope it is not slow because of
accepted length or GPU underutilization. It has higher accepted draft length
than dense, much better CUDA Graph coverage, and higher GPU util. Any earlier
slow `base_only_24` row should be treated as graph/scope/config limited until
a same-shape clean row contradicts this.

The same run localizes the remaining `speclink_t08` gap:

| part | current bs128 evidence | read |
| --- | --- | --- |
| scheduler / mask build | exact diagnostic `6.607 ms/step`, but this includes sync-heavy request-loop accounting | do not use diagnostic tok/s as serving throughput |
| base sparse linear | `1.004 ms/call` overall; `gate_up_proj=1.054`, `down_proj=0.905` | largest localized GPU-side cost |
| residual correction | `0.163 ms/call` overall; `gate_up_proj=0.181`, `down_proj=0.126` | secondary but still part of the gap to base-only |
| gather/scatter | `0.015 ms/call` | not first bottleneck |
| routing statistics | draft residual/base `4134/3650`, non-draft residual/base `973/1823`, bucket fill `0.964` | many draft rows still go through correction, but bucket capacity is well used |
| CUDA Graph | clean `speclink_t08` `FULL=62,NONE=2` | graph coverage is not the current suspect |
| GPU util | clean `speclink_t08` avg `91.5%`, peak `100%` | not idle GPU |

Exact no-fastpath `all_corrected_24` was also measured cleanly:

| method | total tok/s | full-batch tok/s | accepted draft/step | GPU util | storage/dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 2586.544 | 3367.839 | 1.424 | 91.615% | n/a |
| `all_corrected_24` exact no-fastpath | 1994.514 | 2773.572 | 1.392 | 89.438% | 1.625 |

Its diagnostic row reports sparse base `1.162 ms/call` and dense residual
correction `0.908 ms/call` overall. Per leaf, gate/up costs
`1.360 + 1.088 ms/call`; down costs `0.766 + 0.548 ms/call`. The stats confirm
`dense_rows` residual modules are on `cuda:0` and
`compressed_residual_runtime_on_gpu=True`, so this is not a CPU residual
execution bug. It is the expected cost of doing sparse base plus dense
correction as two separate passes.

Next action from this result: a threshold/bucket-only tweak is unlikely to
close the robust `1.2x` full-batch gap. The useful work must either be reduced
by a stronger quality-safe row selector, or the current separate sparse-base
plus residual-correction path must be replaced by a grouped/fused operator.

## What Looks Slow Now

Use clean serving rows for throughput, CUDA Graph coverage, and GPU
utilization. Use diagnostic rows only for component timing and routing counts,
because exact routing counters and CUDA events add synchronization overhead.

| Part | What To Measure | Current Read |
| --- | --- | --- |
| scheduler / mask build | per-step residual mask construction, bucket row construction, row-index plan time | reduced-sync clean paths can be around sub-ms per step; exact diagnostic paths can report tens of ms because they deliberately synchronize. CPU sync must stay reduced, but this is not the only remaining bottleneck. |
| base sparse linear | `gate_up_proj` / `down_proj` sparse base GEMM time, especially selected layers | dominant localized GPU cost in the mixed path. Recent diagnostic rows show roughly `0.9-1.1 ms/call` for the sparse base in active Llama bs64 math runs. |
| residual correction | dense-row correction or sparse residual correction GEMM time | secondary but still real extra work. It becomes expensive as soon as the selector protects many rows; all-corrected variants show the two-pass exact path does not beat dense. |
| gather/scatter | `index_select`, `index_copy_` / `index_add_`, bucket assembly, route assembly | currently small compared with sparse base and correction, usually around hundredths of a ms per Linear call in diagnostic rows. Do not optimize this first unless a fresh breakdown reverses that. |
| routing statistics | draft residual rows, draft base-only rows, non-draft residual rows, bucket fill ratio | bucket fill is usually high, so the issue is not empty buckets. Quality-safe selectors push residual fraction into a range where the current two-pass mixed operator loses its sparse advantage. |
| CUDA Graph | FULL/NONE/PIECEWISE counts for dense, base-only, and SR24 paths | still a real source of loss. Some graph-bucket fixes recover coverage, but graph coverage alone has not been enough to make the mixed operator clearly faster than dense. |
| GPU util | avg/peak util during clean serving | usually high enough that the main issue is inefficient useful work and small/fragmented kernels, not an idle GPU. |

When explaining a slowdown, use this order and do not skip the serving/microbench
split:

| part | source of truth | decision rule |
| --- | --- | --- |
| scheduler / mask build | clean low-sync serving counters plus one diagnostic sync row | if clean mask time is sub-ms, do not optimize CPU counters first |
| base sparse linear | diagnostic Linear timing and isolated component microbench | if sparse base is the largest call time, focus on GEMM shape and graphability |
| residual correction | diagnostic residual timing and residual-row fraction | if correction plus sparse base is dense-parity, selector tweaks alone are insufficient |
| gather/scatter | diagnostic gather/scatter timing and row-routed microbench subcomponents | optimize only if it becomes comparable to base/correction timing |
| routing statistics | residual/base rows, draft/non-draft split, bucket fill ratio | high bucket fill with high residual fraction means useful-work inefficiency, not empty buckets |
| CUDA Graph | clean serving FULL/NONE/PIECEWISE counts | graph loss is a gating issue; graph recovery alone is not a speed proof |
| GPU util | clean serving GPU util and, when useful, nvidia-smi sampling | high util with low tok/s means inefficient kernels/extra work; low util means scheduling or fragmentation |

## Working Diagnosis

The current slowdown is mainly the combination of:

1. sparse base is computed for many rows even when selected rows later need
   dense/residual correction;
2. residual correction is a second separate operator path, so the sparse-base
   win is erased once the protected-row fraction rises;
3. fixed or dynamic residual buckets can improve quality only by increasing
   corrected rows, which moves the operator into an unfavorable region;
4. some SR24 variants still lose CUDA Graph coverage or use graph-unsafe
   row-routing plans;
5. explicit CPU synchronization is costly, but the reduced-sync path already
   avoids the worst of it.

So the next pass should not start with another threshold sweep. It should first
produce a current seven-part breakdown for the exact candidate being tested.

## Next Measurement Protocol

Run the wrapper from:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec python scripts/run_sr24_slowdown_breakdown.py \
  --sr24-preset criticalprefix4_bucket16_directcslt \
  --batch-size 64 \
  --dataset math_reasoning \
  --fixed-total-requests 128 \
  --max-tokens 256 \
  --instrumented-requests 64 \
  --instrumented-max-tokens 128
```

`criticalprefix4_bucket16_directcslt` is the current default quality/speed
candidate and should include `extra_after_low=1`, `min_prefix_residual=4`,
`non_draft=bonus`, `gate_up_proj=16-31;down_proj=8-15`,
`bucket_dense_copy`, direct cuSPARSELt sparse base, no residual-bucket priority,
and the SR24-specific graph/compile path rather than `--sr24-default-vllm-compile`.
If a generated command shows `--sr24-selective-extra-after-low 0`,
`--sr24-residual-bucket-priority`, or `--sr24-default-vllm-compile`, it is not
the current candidate.

The output should stay under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/
```

Use this acceptance criterion for each future candidate:

1. clean serving must report total/full-batch tok/s, accepted draft tokens per
   step, CUDA Graph counts, and GPU utilization;
2. diagnostic serving must report scheduler/mask timing, sparse-base timing,
   residual-correction timing, gather/scatter timing, and routing counts;
3. component microbench must show whether the operator shape has enough
   headroom before it is promoted into serving;
4. a quality gate must be run only after the breakdown shows a plausible speed
   path.

## Optimization Priority

1. Restore or preserve CUDA Graph coverage for any candidate that is otherwise
   promising.
2. Avoid duplicate work on corrected rows: do not compute full sparse base for
   rows that will be overwritten by dense correction unless the breakdown shows
   it is still faster.
3. Implement a packed/fused mixed operator or grouped route that keeps large
   GEMM shapes instead of many small kernels.
4. Only then tune selectors, with the constraint that quality-safe residual
   fractions must remain inside the region where the operator is faster than
   dense.

2026-06-29 follow-up: bucket-only tile filling is not enough. In the latest
bucket64 row-routed diagnostic, important rows no longer pay sparse+dense
duplicate work, but gate_up/down base sparse remains dominant
(`0.982/0.626 ms/call`) while dense selected rows are small
(`0.156/0.086 ms/call`). Bucket64 and bucket128 improve neither total tok/s nor
the required 1.2x robust speedup. Treat fixed route-table plus fused/grouped
mixed operator work as the next engineering target; use the new
`Row-Routed Linear Components` table from `summarize_sr24_breakdown.py` to
confirm whether a candidate actually executes gate_up/down row routing.

2026-06-29 lossy gate follow-up: use
`scripts/run_sr24_lossy_speed_quality_sweep.py` for the next small candidates.
It applies the current `<=8pp` GSM8K-50 quality budget before launching
throughput and writes outputs under `results.bak/`. The first two probes show
why row-routing needs a tile-fill/planner guard:

| candidate | quality | full-batch speedup | total speedup | action |
| --- | ---: | ---: | ---: | --- |
| `lowresidual_gateup_riskcap2` | `0pp` drop | `1.0185x` | `0.9257x` | quality-safe but too slow |
| `lowresidual_gateup_riskcap2_rowrouted` | `0pp` drop | `0.9906x` | `0.9192x` | row split is real but tiny dense rows are slower |

So the next experiment should not unconditionally enable row routing. It should
use a planner: for small important-row counts, keep base-first overwrite; for
large or tile-filled groups, use row-routed/fused execution; for high residual
fraction, fall back to dense. The route-table format should be fixed-size and
graph-stable before claiming systems-level progress.

2026-06-30 base-only lossy gate result: the pure sparse upper bound is real at
the isolated Linear level, but the current serving scopes do not yet hit the
robust 1.2x target under the 8pp quality budget. The gate_up-shaped mixed
bucket microbenchmark at
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/speclink_sr24_mixed_bucket_probe_20260629_235410/`
shows pure sparse base around `0.657x` dense time, while base-first correction
is already `0.935x-1.15x` dense for bucket `1-128` and full routed assembly is
`>1.2x` dense except at very large buckets. The corresponding quality-gated
serving run at
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_scope_sweep_20260629_235858/`
reports:

| scope | quality | bs64 full speedup | bs64 total speedup | decision |
| --- | ---: | ---: | ---: | --- |
| `gateup16_31` | `-8pp`, pass | `1.068x` | `0.982x` | quality budget is spent but speed is insufficient |
| `gateup_all` | `-30pp`, fail | skipped | skipped | too much accuracy loss |
| `gateup31 up_sparse` | `+2pp`, pass | `0.966x` | `0.911x` | scope too small to help |

The next implementation should therefore be treated as a systems operator
problem:

1. Keep a fixed route-table buffer per layer with slots
   `{row_id, route_kind, active}` so CUDA Graph sees stable pointers.
2. Pre-bucket routes into dense-important and sparse-unimportant packed row
   groups; avoid sparse work for dense-important rows and dense work for
   sparse-unimportant rows.
3. Add a latency-table planner from the microbenchmark: small important groups
   should stay base-first or dense-fallback; only route/tile-fill when the
   predicted group size is above the measured crossover.
4. Replace Python `index_select`/small GEMM/scatter chains with either a fused
   packed operator or grouped GEMM plus fused assembly. If streams are used,
   pre-create graph-safe sparse and dense streams and explicitly measure
   overlap; do not assume overlap from launch order.
5. Re-run the GSM8K-50 `<=8pp` quality gate before any throughput matrix.

2026-06-30 operator follow-up: three tempting implementation shortcuts are now
negative under the current environment.

| probe | artifact | result | decision |
| --- | --- | --- | --- |
| direct packed compressed residual Triton | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_profile_20260630_001736/summary.md` | gate/up-shaped packed residual is `7.614x` dense residual time; down-shaped packed residual is `9.212x` dense residual time | reject current Triton packed residual |
| cached-stream row overlap | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_overlap_cached_streams_bs64_math128_20260630/report.md` | bs64 math max128 `speclink_t08` is `0.769x` total and `0.874x` full-batch versus dense | keep only as ablation |
| exact no-fastpath `all_corrected_24` with torch-sparse residual | `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_gateup16_31_bs64_math128_20260630/report.md` | narrow gate/up16-31, direct cuSPARSELt, graph-covered exact row is `0.812x` total and `0.841x` full-batch versus dense | keep as correctness/control path |

This sharpens the next systems direction. The useful target is not a second
standalone residual sparse GEMM, not dynamic Python row overlap, and not exact
two-pass `all_corrected_24`. The next implementation should pack the route
once and execute mutually exclusive dense-important and sparse-unimportant work
through a fused or grouped operator. If overlap is used, streams must be
pre-created and measured end-to-end, but overlap alone is insufficient.
