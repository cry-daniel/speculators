# SR24 Slowdown Breakdown

This note is the current slowdown reference for SpecLink SR24. It uses existing
Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch/concurrency 128
runs unless a row explicitly says it is diagnostic.

## Latest Seven-Part Read

### 2026-06-28 Gate-Up16 Focused Breakdown

Focused result path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_gateup16_bs64_math_k8_20260628/seven_part_report_direct/report.md
```

This run isolates the current gate-up-only path:
Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size 64,
`gate_up_proj=16-31`, `high_confidence@0.3`, dense-row residual correction on
CUDA. Clean serving uses low-sync counters; the direct instrumented row uses
CUDA events and exact routing only for attribution.

Clean serving rows:

| method | full-batch tok/s | total tok/s | full speedup vs dense | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `2763.202` | `2368.916` | `1.000x` | `1.402` | `88.643%` | n/a |
| `base_only_24` | `2937.606` | `2475.326` | `1.063x` | `1.561` | `91.538%` | `{"FULL":97,"NONE":31}` |
| `speclink_t08` | `2294.100` | `1904.859` | `0.830x` | `1.453` | `91.235%` | `{"NONE":128}` |
| `all_corrected_24` dense fastpath | `2780.837` | `2400.316` | `1.006x` | `1.427` | `90.786%` | dense-equivalent control |

Current seven-part read:

| component | latest evidence | read |
| --- | --- | --- |
| scheduler / mask build | clean `speclink_t08` mask wall `0.453ms/step`, batched builder `0.391ms/step`, row bucket `0.004ms/step` | sub-ms clean cost; not the first bottleneck |
| base sparse linear | diagnostic `gate_up_proj=16-31` sparse base `0.570ms/call`, about `532` rows/call | large GPU-side cost |
| residual correction | diagnostic dense correction `0.570ms/call` | as large as sparse base in this mixed path |
| gather/scatter | diagnostic wrapper cost `0.083ms/call` | visible, but smaller than base sparse plus correction |
| routing statistics | draft residual/base `9300/6804`, non-draft residual/base `2013/3735`, draft residual fraction `0.577` | residual-row fraction is still too high for a two-pass operator |
| CUDA Graph | clean `speclink_t08` `{"NONE":128}`; base-only keeps `{"FULL":97,"NONE":31}` | dynamic mixed t08 is graph-limited in this focused run |
| GPU util | clean `speclink_t08` avg `91.235%`, peak `100%` | not idle; the problem is graph loss plus inefficient useful work |

Decision: do not make another ordinary CPU-sync cleanup the main optimization
unless a fresh clean row shows mask build returning to multi-ms scale. The next
useful speed work is either (1) make dynamic mixed verification graph-safe, or
(2) remove the two-pass Linear shape by fusing/packing sparse base plus residual
correction, or by reducing residual rows with a quality-proven controller.

### 2026-06-28 Update: Early Dense No Longer Forces Python Routing

The all-MLP early-dense path below originally exposed a `40ms/step` scheduler
hot path. That part has now been fixed: early-dense handling is part of the
Triton batched mask builder, using per-request generated lengths, instead of
forcing the slow Python request-routing loop.

Updated clean serving root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_earlydense_batched_builder_bs64_math_k8_cleanstats_20260628/report.md
```

| method | full-batch tok/s | total tok/s | vs dense full | vs dense total | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3149.236` | `2205.545` | `1.000x` | `1.000x` | `1.406` | `86.375%` | n/a |
| all-MLP `speclink_t08` | `4307.435` | `2171.978` | `1.368x` | `0.985x` | `2.329` | `83.375%` | `{"FULL":62,"NONE":2}` |

The scheduler breakdown moved from:

```text
mask wall 40.230ms/step, request routing loop 39.994ms/step
```

to:

```text
mask wall 0.978ms/step, batched builder 0.792ms/step,
request routing loop 0.000ms/step, row bucket 0.112ms/step,
residual bucket 0.110ms/step, mixed row indices 0.001ms/step
```

This changes the immediate diagnosis. The former scheduler/mask-build problem
has been reduced by roughly `40x` and is no longer the first bottleneck for this
candidate. The path now has real full-batch speed headroom, but the short-run
total throughput is still slightly under dense, so longer runs are needed to
separate steady-state speed from fill/drain effects. The operator-side cost
from sparse base plus residual correction remains the next implementation
target.

Quality is not fully cleared yet. The paired GSM8K-50 gate at:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_earlydense_batched_builder_paired_gsm8k50_20260628/report.md
```

reports dense EAGLE3 `0.7400` exact match and all-MLP `speclink_t08` `0.7200`,
with 2 paired regressions and 1 paired improvement. Treat the current all-MLP
controller as promising for speed but not final for quality.

The latest user-pivot breakdown for the all-MLP Triton-override speed path with
an early dense guard is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_k8_combined_breakdown_20260628/report.md
```

This is not the same scope as the smaller quality-safe
`gate_up_proj=16-31;down_proj=8-15` path below. It uses all MLP leafs
(`gate_up_proj,down_proj`), `early_dense_tokens=64`, bucket32, Triton bucket
override, and mixed CUDA Graph. It reaches full-batch `3649.985` tok/s versus
dense `3025.258` (`1.207x`) but total tok/s is lower than dense
(`1976.243` versus `2185.404`, `0.904x`). The seven-part diagnosis is:

| part | current read |
| --- | --- |
| scheduler / mask build | clean `40.230ms/step`, almost entirely request routing loop `39.994ms/step` |
| base sparse linear | profile aggregate sparse base `1.933ms/call`, `gate_up_proj=16-31` `2.595ms/call` |
| residual correction | dense-row correction `0.120ms/call`, `gate_up_proj=16-31` `0.154ms/call` |
| gather/scatter | `0.287ms/call`; Triton bucket override `0.570ms/call` |
| routing | draft residual/base `16759/3265`, draft residual fraction `0.837`, bucket fill `0.836` |
| CUDA Graph | healthy: `{"FULL":62,"NONE":2}` in the clean row |
| GPU util | busy: avg `84.556%`, peak `99%` |

Read: this candidate is no longer primarily graph-limited or GPU-idle. The next
work should reduce the `critical_prefix` request-routing/mask loop and fuse or
pack sparse base plus dense-row correction. Do not start with another scalar
threshold/bucket sweep unless a new breakdown shows these two costs have moved.

The latest current-code seven-part breakdown is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_breakdown_bs64_math128_20260628_0322/seven_part_report/report.md
```

It is the refreshed user-requested table for Llama-3.1-8B,
`math_reasoning`, EAGLE3 K=8, client-side batch size 64, max new tokens 128,
CUDA Graph enabled, and the current quality-safe
`critical_prefix@0.6,prefix4,extra1` bucket32 path on
`gate_up_proj=16-31;down_proj=8-15`. Clean serving rows:

| method | full-batch tok/s | total tok/s | same-root speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3128.512` | `2184.738` | `1.000x` | `1.399` | `86.250%` | n/a |
| `base_only_24` | `3428.164` | `2238.218` | `1.096x` | `1.635` | `86.500%` | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | `3182.699` | `2004.300` | `1.017x` | `1.606` | `86.625%` | `{"FULL":62,"NONE":2}` |

The clean path is graph-healthy and GPU-busy. The diagnostic row localizes the
remaining cost to sparse base plus residual correction: aggregate sparse base
`1.007ms/call`, `gate_up_proj=16-31` sparse base `1.069ms/call`, dense-row
correction `0.148ms/call`, gather/scatter `0.015ms/call`, and draft
residual/base rows `7263/5921`. Clean scheduler/mask construction is
`0.949ms/step`, including `0.107ms/step` row-bucket and `0.105ms/step` bucket
build. Therefore the current slowdown is not a simple CUDA Graph miss, accepted
length collapse, or idle-GPU issue; it is the two-pass useful-work shape of the
mixed sparse-base plus dense-row correction path. The small quality-safe
`base_only_24` upper bound is only `1.096x`, so this scope cannot realistically
yield a `1.2x` `speclink_t08` result without a fused/packed operator or much
lower residual-row fraction.

The previous 01:22 report with the same intended shape remains useful for
cross-run noise comparison:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_requested_seven_part_breakdown_bs64_math128_20260628_0122/seven_part_report/report.md
```

It uses Llama-3.1-8B, `math_reasoning`, EAGLE3 K=8, client-side batch size 64,
max new tokens 128, CUDA Graph enabled, and the current quality-safe
`critical_prefix`/bucket32 selective-residual path. The clean serving rows are:

| method | full-batch tok/s | total tok/s | same-root speedup | accepted draft/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3021.676` | `2184.905` | `1.000x` | `1.395` | `84.625%` | n/a |
| `base_only_24` | `3480.972` | `2285.907` | `1.152x` | `1.574` | `85.000%` | `{"FULL":62,"NONE":2}` |
| `speclink_t08` | `3086.711` | `2065.992` | `1.022x` | `1.632` | `86.875%` | `{"FULL":62,"NONE":2}` |

The current slow path is not explained by accepted-length collapse, CUDA Graph
loss, or an idle GPU. Clean `speclink_t08` keeps `62/64` FULL graph steps and
`86.875%` average GPU utilization. Clean scheduler/mask work is also sub-ms:
`0.380ms/step` total, including `0.112ms/step` row-bucket work and
`0.001ms/step` mixed row-index work.

The instrumented row localizes the cost to GPU-side useful-work efficiency:
`gate_up_proj=16-31` sparse base is `1.023ms/call`, aggregate sparse base is
`0.937ms/call`, dense-row correction is `0.148ms/call`, and gather/scatter is
only `0.012ms/call`. Routing still sends many rows through residual correction:
draft residual/base rows are `14125/11395`, non-draft residual/base rows are
`3190/3787`, and the draft residual fraction is `0.553`.

The direct follow-up ablations on this same shape were:

| ablation | full-batch speedup vs dense | read |
| --- | ---: | --- |
| bucket dense copy overwrite | `1.026x` | tiny gain only; gather/scatter is not the bottleneck |
| Triton bucket dense GEMM scatter | `0.962x` | slower than the default PyTorch/cuBLAS bucket correction |
| adaptive dense fallback | `0.952x` | slower and accepted draft length collapses toward dense |
| `all_corrected_24` compressed-dense on CUDA | `0.780x` | residual modules are GPU-resident, but sparse base plus residual GEMM is extra work |
| direct compressed Triton residual | `0.181x` | rejected diagnostic path |

The active optimization decision is therefore: do not spend the next pass on
ordinary CPU-sync cleanup, fixed-bucket mechanics, or gather/scatter-only
rewrites. With the current quality-safe scope, `base_only_24` is only `1.152x`
dense, so `speclink_t08` cannot reach a `1.2x` dense target unless residual rows
drop sharply without an accuracy regression or the mixed sparse/residual
operator is fused/packed enough to remove the two-pass cost.

The base-only scope speed-ceiling sweep under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_scope_sweep_bs64_math128_20260628_0203/report.md`
makes this stricter. On bs64/math/max128, all small or quality-related scopes
remain below `1.2x` base-only full-batch speedup: safe
`gate_up=16-31,down=8-15` is `1.137x`, `gate_up=16-31` is `1.074x`,
`down=8-15` is `1.024x`, `down=16-31` is `0.964x`,
`gate_up=16-31,down=16-31` is `1.132x`, all `gate_up` is `1.127x`, and
tail `gate_up=31` with `up_sparse` is `0.973x`. Only all-MLP
`gate_up,down=0-31` reaches clear headroom: `5098.544` full-batch tok/s versus
dense `3134.588` (`1.627x`) with accepted draft/step `2.434`. Thus the next
`speclink_t08` path is all-MLP plus a quality-protection controller, or a new
fused/packed mixed operator; threshold-only tuning on the smaller scopes cannot
meet the `1.2x` target.

The first all-MLP `speclink_t08` check under
`/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bs64_math128_20260628_0220/report.md`
keeps part of that headroom but not enough yet: dense is `3025.159` full-batch
tok/s, all-MLP `base_only_24` is `5399.310` (`1.785x`), and all-MLP
`speclink_t08` is `3545.363` (`1.172x`) with accepted draft/step `2.345` and
CUDA Graph `{"FULL":55,"NONE":9}`. This is the best current t08 speed candidate,
but it still lacks a paired quality gate and remains below the `1.2x` target.

Absolute evidence roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dense_baseline_bs128_math_gpumem095_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_bucket128_scale10_repeat_gpumem095_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_route_assembly_bucket128_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_bucket128_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_reuse_base_output_instrumented_bs64_math_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_probe_rows1024_buckets_20260627/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_same_condition_bs128_math_gpumem095_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_trace_current_reusebase_bucket128_bs128_math_20260627/trace_analysis/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_criticalprefix_t04_bucket512_reusebase_bs128_math_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_allcorrected_compressed_nofastpath_clean_bs64_math_20260627/summary.csv
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_allcorrected_compressed_final_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_bucket512_graph_bs64_math128_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_gateup_graph_bs64_math128_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_gateup_routeall_graph_bs64_math128_20260627/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cpu_sync_ablation_prefixconf_routeall_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_gateup_routeall_instrumented_bs64_math128_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_routeall_bs64_math128_20260627/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_routeall_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_skipbucket_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_skipbucket_eager_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_t005_bucket_triton_densegemm_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_routeall_directcpurows_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_h2_routeall_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_h2_reusebase_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix_h0_reusebase_bs64_math128_20260627/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_operator_probe_20260627_225539/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_operator_probe_20260627_225539/summary.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_requested_slow_parts_bs64_math128_20260627/seven_part_report/report.md
```

## Current End-To-End Read

| config | total tok/s | vs dense total | full-batch tok/s | vs dense full | accepted draft/step | GPU util | CUDA Graph | scheduler mask | row bucket/index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| dense EAGLE3 baseline | `2620.402` | `1.000x` | `3294.699` | `1.000x` | `1.423` | `92.0%` | n/a | n/a | n/a |
| base_only_24 same-condition refresh | `2759.003` | `1.053x` | `3879.184` | `1.177x` | `1.599` | `91.5%` | `{"FULL":62,"NONE":2}` | `0.346ms/step` | `0.000ms/step` |
| SR24 bucket128, position scale 10 | `2053.933` | `0.784x` | `3005.977` | `0.912x` | `1.050` | `93.3%` | `{"FULL":94,"NONE":2}` | `0.435ms/step` | `0.139ms/step` |
| SR24 bucket128 + Triton assembly | `2082.782` | `0.795x` | `2983.379` | `0.906x` | `1.030` | `93.4%` | `{"FULL":94,"NONE":2}` | `0.443ms/step` | `0.142ms/step` |
| SR24 bucket128 + reuse base output | `2221.430` | `0.848x` | `3102.340` | `0.942x` | `1.064` | `92.0%` | `{"FULL":62,"NONE":2}` | `0.465ms/step` | `0.107ms/step` |
| SR24 critical_prefix@0.4 + bucket512 + reuse base | `1878.287` | `0.717x` | `2646.401` | `0.803x` | `1.052` | `93.9%` | `{"FULL":94,"NONE":2}` | `0.745ms/step` | `0.089ms/step` |
| SR24 prefix_conf@0.05 + gate_up + route_all | `1960.692` | n/a | `2961.618` | `0.981x` vs same-root dense | `1.399` | `81.0%` | `{"FULL":49}` | stats off | stats off |
| SR24 prefix_conf@0.05 + gate_up + route_all, no contiguous fastpath | `1970.441` | n/a | `2949.227` | `0.974x` vs same-root dense | `1.404` | `86.4%` | `{"FULL":49}` | stats off | stats off |
| SR24 user seven-part refresh, dense same-root | `2180.865` | `1.000x` | `3014.943` | `1.000x` | `1.396` | `80.0%` | n/a | n/a | n/a |
| SR24 user seven-part refresh, base_only_24 | `2226.338` | `1.021x` | `3235.705` | `1.073x` | `1.547` | `82.5%` | `{"FULL":62,"NONE":2}` | `0.207ms/step` | `0.000ms/step` |
| SR24 user seven-part refresh, speclink_t08 route_all | `1970.598` | `0.904x` | `2949.810` | `0.978x` | `1.404` | `87.7%` | `{"FULL":62,"NONE":2}` | `42.339ms/step` | `42.083ms/step` |
| SR24 route_all skip unused bucket | `1948.593` | `0.893x` | `2809.138` | `0.931x` | `1.385` | `82.8%` | `{"FULL":62,"NONE":2}` | `43.172ms/step` | `42.115ms/step` |
| SR24 route_all skip bucket, force graph NONE | `1790.669` | `0.819x` | `2611.693` | `0.863x` | `1.393` | `82.2%` | `{"NONE":64}` | `13.775ms/step` | `13.501ms/step` |
| SR24 fixed bucket512 + Triton dense correction | `1488.014` | `0.682x` | `2029.030` | `0.672x` | `1.411` | `89.0%` | `{"FULL":62,"NONE":2}` | `0.400ms/step` | `0.106ms/step` |
| SR24 route_all + direct CPU score rows | `1949.103` | `0.894x` | `2804.630` | `0.930x` | `1.385` | `80.3%` | `{"FULL":62,"NONE":2}` | `43.282ms/step` | `0.002ms/step` |
| SR24 fixed_prefix H=2 + route_all | `2013.475` | `0.928x` | `2871.347` | `0.920x` | `1.385` | `86.5%` | `{"FULL":62,"NONE":2}` | `0.652ms/step` | `0.002ms/step` |
| SR24 fixed_prefix H=2 + reuse base | `2032.154` | `0.929x` | `2883.661` | `0.954x` | `1.386` | `86.8%` | `{"FULL":62,"NONE":2}` | `0.277ms/step` | `0.003ms/step` |
| SR24 fixed_prefix H=0 + reuse base speed ceiling | `2022.507` | `0.923x` | `2861.618` | `0.945x` | `1.380` | `86.4%` | `{"FULL":62,"NONE":2}` | `0.626ms/step` | n/a |

The current bottleneck is not gross GPU idleness. Older bucket128/reuse-base
rows made scheduler/mask construction look secondary, but the latest route-all
refresh shows a large clean row-index/bucket wall counter. For older selective
rows, accepted draft length was also too low, around `1.06` draft tokens per
speculative step versus dense EAGLE3's `1.42`. The latest prefix-confidence
route-all run repairs accepted length, so its remaining slowdown should be
attributed to row-index/bucket overhead plus mixed sparse/dense Linear useful
work rather than acceptance collapse.

The same-condition `base_only_24` refresh answers the first goal item: in the
current scoped configuration it is not slow. It accepts more draft tokens than
dense EAGLE3 (`1.599` vs `1.423`), has comparable GPU utilization, and keeps
normal CUDA Graph coverage. If a future broader `base_only_24` run is slow, it
should be treated as a scope/operator-shape issue unless a fresh seven-part
report proves accepted-length collapse or GPU underutilization.

The later bs64/max128 non-route-all t08 refresh gives a cleaner answer to the
current "where is it slow" question for the default mixed path. In that report,
dense EAGLE3 is `3025.137` full-batch tok/s, `base_only_24` is `3242.405`
full-batch tok/s with normal graph coverage, and `speclink_t08` is only
`2468.236` full-batch tok/s with all mixed verify steps in CUDA Graph `NONE`.
Clean scheduler/mask build is only `0.674ms/step`, so it is not the main clean
bottleneck for this path. The instrumented row localizes the operator-side cost
to sparse base `0.486ms/call`, dense residual correction `0.591ms/call`, and
gather/scatter `0.078ms/call`, with draft residual/base rows `17060/9484`.
Thus the immediate non-route-all t08 priority is graph-safe mixed execution
plus less or fused dense correction, not another threshold-only controller
sweep or ordinary stats-overhead cleanup.

The 2026-06-27 user-requested seven-part refresh changes the immediate priority
for the route-all path. With the prefix-confidence/gate-up/route-all candidate,
accepted draft length is not the problem in this run: `speclink_t08` accepts
`1.404` draft tokens/step versus dense EAGLE3 `1.396`. CUDA Graph coverage is
also healthy (`{"FULL":62,"NONE":2}`), and GPU utilization is higher than
dense (`87.7%` versus `80.0%`). The visible clean-path problem is the dynamic
row-index/bucket path: `sr24_scheduler_mask_wall_cpu_ms_per_step=42.339`,
almost all of it in
`sr24_scheduler_row_index_bucket_wall_cpu_ms_per_step=42.083`. By contrast,
the same-condition `base_only_24` row has only `0.207ms/step` scheduler mask
wall time and no row bucket/index work.

This means the next optimization pass should first split and reduce
`_compute_residual_bucket()` plus `_compute_mixed_row_indices()` in
`vllm/vllm/speclink_sr24.py`. The current graph-safe route-all path builds a
static residual bucket, runs top-k/copy for a fixed bucket tensor, and computes
both residual and base row indices from a full mask. The diagnostic row still
shows GPU-side Linear cost too: sparse base is `0.637ms/call`, dense correction
is `0.183ms/call`, and route-all gather/scatter is `0.093ms/linear`. But the
clean wall counter says we should not jump straight to another controller
sweep or only tune the Linear hook; the scheduler row-index/bucket window is
now large enough to be isolated first.

The first isolation pass added `SPECLINK_SR24_ROUTE_ALL_SKIP_BUCKET=1` and
separate clean counters for residual bucket build and mixed row indices. It
proved that the residual bucket itself is not the remaining issue. With skip
bucket enabled, bucket build is only `0.001ms/step`, but
`scheduler_mixed_row_indices_wall_cpu_ms_per_step` remains `42.113ms` in the
graph-enabled clean row and throughput falls to `0.931x` dense. Forcing mixed
steps to CUDA Graph NONE reduces mixed row-index wall time to `13.499ms/step`,
but loses graph coverage and drops throughput further to `0.863x` dense. The
actionable bottleneck is therefore the dynamic `nonzero`/variable-length row
list construction in `_compute_mixed_row_indices()`, not bucket top-k. The next
implementation should avoid dynamic row-list materialization on the serving
path, likely with a fixed-shape mask-aware routed Linear/Triton/CUDA operator
or a compact per-request prefix representation that does not require global
`nonzero` each decode step.

The fixed-bucket control then checked whether avoiding dynamic row-list
materialization is sufficient by itself. It used `prefix_confidence@0.05`,
`gate_up_proj`, bucket512, `SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM=1`, and
normal CUDA Graph coverage. It reduced clean scheduler/mask wall time to
`0.400ms/step`, row-index/bucket to `0.106ms/step`, and mixed row indices to
`0.001ms/step`, so it successfully removes the route-all row-list bottleneck.
But throughput fell to `2029.030` full-batch tok/s versus same-root dense
`3020.696` (`0.672x`) while GPU util stayed high (`89.0%`) and accepted
draft/step stayed healthy (`1.411`). The diagnostic row attributes this to
GPU-side useful work: base sparse `0.577ms/call` plus dense correction
`0.316ms/call` for a fixed 512-row bucket, with gather/scatter only
`0.039ms/call`. Therefore the current slow path is two-stage: route-all has a
dynamic row-list problem, while fixed-bucket avoids it but over-corrects and
pays too much sparse-base plus dense-correction work.

Two direct-row follow-ups narrow this further. First,
`SPECLINK_SR24_DIRECT_CPU_ROUTE_ROWS=1` for `prefix_confidence` moved the
row-list wait from GPU `nonzero` to a small draft-score GPU-to-CPU sync:
`scheduler_row_index_bucket_wall_cpu_ms_per_step` fell to `0.002`, but
`scheduler_direct_cpu_route_rows_wall_cpu_ms_per_step` became `43.200`; clean
throughput was `0.930x` dense. This rejects CPU score readback as a row-list
fix. Second, the new `fixed_prefix` policy avoids draft-score reads entirely
and protects a fixed number of draft prefix rows plus the bonus row. It removes
the scheduler wait (`0.277-0.652ms/step`) but still does not beat dense. With
H=2 and `route_reuse_base_output`, full-batch throughput was `2883.661` versus
dense `3022.426` (`0.954x`) with correction fraction about `0.298`; with H=0,
the correction fraction fell to about `0.099` but throughput was still only
`0.945x`. The bottleneck has moved from scheduler row-list construction to
the operator ceiling: current gate/up semi-structured sparse base plus any
correction does not provide a serving-side `1.2x` path.

The all-corrected no-fastpath compressed-residual row answers the second goal's
first question: it is not a CPU-residual-transfer issue. The clean bs64 run
uses `torch_sparse/compressed_dense@cuda`, `compressed_residual_runtime_on_gpu`
is true, CUDA Graph coverage is healthy (`{"FULL":77,"NONE":2}`), and GPU util
is high (`89.4%`). It is still slower than dense: full-batch output is
`2424.444 tok/s` versus dense `3021.430 tok/s` (`0.802x`). The instrumented
row localizes the extra work to GPU-side Linear components: sparse base
`0.980ms/call`, compressed residual GEMM `0.557ms/call`, residual add
`0.135ms/call`, and materialization only `0.0004ms/call` thanks to the cached
GPU residual weight. So the current all-corrected path is slow because it pays
both sparse base and residual correction, not because it copies residual data
through CPU.

The operator microbench refresh makes the same point without vLLM serving
effects. For Llama `gate_up_proj` (`out=28672, in=4096`), graph-captured base
sparse alone is useful: `0.1668ms` versus `0.2892ms` dense at 256 rows
(`0.58x` dense time), and `0.3551ms` versus `0.5401ms` dense at 512 rows
(`0.66x`). But exact correction consumes that gain quickly. At 10% corrected
rows, the serving-like mixed path is already `1.24x` dense time for 256 rows
and `1.02x` dense time for 512 rows; at 25% corrected rows it is `1.31x` and
`1.15x`. The exact all-corrected sparse backend is `1.34x` dense time at 256
rows and `1.41x` dense time at 512 rows. So the current `speclink_t08`
slowdown is consistent with an operator ceiling: base sparse is fast, but base
sparse plus correction is not.

## Seven-Part Breakdown

| part | what to measure | current evidence | diagnosis |
| --- | --- | --- | --- |
| scheduler / mask build | Per-step residual mask construction, request routing, row bucket assembly. | Latest route-all clean row: `42.339ms/step` mask wall time, `42.083ms/step` row-index/bucket wall time. `base_only_24` same-root is `0.207ms/step`. Skip-bucket route-all shows bucket build `0.001ms/step` but mixed row indices `42.113ms/step`; force-NONE lowers row indices to `13.499ms/step` but throughput worsens to `0.863x`. | The bucket is not the blocker. Dynamic `nonzero`/variable row-list construction is the first scheduler-side target, and simply disabling graph is not acceptable. |
| scheduler / mask build, fixed bucket control | Same counters when dynamic row-list materialization is avoided. | Fixed bucket512 + Triton dense correction: `0.400ms/step` mask wall time, `0.106ms/step` row-index/bucket, `0.104ms/step` bucket build, and `0.001ms/step` mixed row indices. | Fixed shapes can remove the scheduler row-list bottleneck, but this route is still slow because the operator work is too heavy. |
| scheduler / mask build, fixed_prefix control | Mask and route rows without reading draft scores. | fixed_prefix H=2 + reuse base: `0.277ms/step` mask wall, `0.215ms/step` direct row construction, `0.003ms/step` row-index/bucket. | Scheduler overhead is no longer the blocker in this shape. |
| base sparse linear | Sparse/base work for `gate_up_proj=16-31` and matching row-routed MLP base side. | Reuse-base diagnostic: row-routed base side `2.709ms/call`. All-corrected compressed diagnostic: sparse base `0.980ms/call` for `gate_up_proj=16-31`, `489` rows/call. Latest route-all diagnostic: sparse base `0.637ms/call` for `125` base rows/call. Operator microbench: gate_up base sparse is `0.58-0.66x` dense time for rows 256/512. | Base sparse alone is good enough to explain why `base_only_24` can be faster than dense. It is not enough once correction rows are introduced. |
| base sparse linear, fixed_prefix control | Sparse/base work after score-free fixed prefix routing. | fixed_prefix H=2 + reuse base diagnostic: base sparse `0.695ms/call`; H=0 is `0.740ms/call`. | Even with sparse correction, base sparse dominates and does not yield enough end-to-end gain. |
| base sparse linear, fixed bucket control | Sparse/base work when residual rows are corrected through a fixed bucket. | Fixed bucket512 diagnostic: sparse base `0.577ms/call` for `281` base rows/call. | The base sparse side is cheaper than route-all's earlier diagnostic, but still large enough to matter. |
| residual correction | Dense-row or compressed residual correction GEMM for selected residual rows. | Reuse-base diagnostic: dense correction side `0.323ms/call`. All-corrected compressed diagnostic: compressed residual total `0.692ms/call`, mostly GEMM `0.557ms/call` plus add `0.135ms/call`; materialization is negligible (`0.0004ms/call`). Latest route-all diagnostic: dense GEMM `0.183ms/call` for `115` dense rows/call. Operator microbench: only 10% corrected rows pushes gate_up mixed time to `1.02-1.24x` dense. | Correction is additive GPU work, and the current two-pass operator loses the base-sparse gain at modest correction fractions. |
| residual correction, fixed bucket control | Dense correction GEMM for a fixed bucket. | Fixed bucket512 diagnostic: dense correction `0.316ms/call` for 512 bucket rows. | This over-corrects relative to useful accepted rows and is the dominant reason the fixed-shape control is slower than route-all. |
| gather/scatter | `index_select`, assembly, `index_copy_` / `index_add_`, Triton assembly. | Latest route-all diagnostic: gather/scatter aggregate `0.093ms/linear`; raw components are base gather `0.004ms`, dense gather `0.004ms`, base copy `0.074ms`, dense copy `0.010ms` per Linear call. The older summary field `0.023ms/event` is event-averaged, not a per-Linear total. | Linear-side assembly is secondary. Scheduler-side row-index/bucket construction is the higher-priority assembly-like cost. |
| routing statistics | Draft residual rows, non-draft residual rows, bucket fill, accepted base-only risk. | Latest diagnostic route-all row: draft residual/base `6548/7396`, non-draft residual/base `1743/1635`, correction row fraction `0.479`, bucket fill `0.035`, bucket actual/requested `18/8291`. | The mixed path corrects almost half the rows while the active bucket signal is sparse. Row selection and fixed-bucket mechanics are not aligned with useful accepted-token work. |
| CUDA Graph | FULL/NONE graph steps for dense/base-only/SR24. | Latest route-all clean row: `{"FULL":62,"NONE":2}` for both `base_only_24` and `speclink_t08`; all-corrected diagnostic stays eager by design. | Graph loss is not the current clean-serving explanation. It remains a guardrail for new variants. |
| GPU util | Average/peak utilization and full-batch output tok/s. | Latest route-all clean row: SR24 GPU util `87.7%`, dense `80.0%`, base-only `82.5%`; SR24 full-batch is still `0.978x` dense. | The GPU is not idle. The problem is inefficient useful work plus the row-index/bucket window, not simple underutilization. Small kernels can still hurt graph/launch efficiency, but the dominant evidence points to too much base sparse plus correction work. |

## Why It Is Slow

The latest data points to two route-all efficiency problems:

1. In the latest prefix-confidence route-all run, accepted length is repaired:
   SR24 gets `1.404` accepted draft tokens/step versus dense `1.396`.
2. Even with accepted length repaired, SR24 still pays dynamic row-index/bucket
   work: `42.083ms/step` in the broad clean wall counter.
3. During those steps, SR24 still runs sparse/base and dense correction paths.
   The route-all diagnostic shows base sparse `0.637ms/call` and dense
   correction `0.183ms/call`; the older reuse-base diagnostic shows the
   base side can be much worse when routing is not avoiding duplicate work.
4. For exact `all_corrected_24`, accepted length is not the issue; every row is
   corrected. The path is slow because sparse base `0.980ms/call` plus
   compressed residual correction `0.692ms/call` exceeds the dense target
   work it replaces.
5. CUDA Graph is not currently bad enough to explain the route-all slowdown:
   clean `speclink_t08` has only `2/64` NONE steps. Scheduler row-index/bucket
   and mixed Linear useful-work are the immediate bottlenecks.

## Routing Trace Read

The current `all_if_any_low + bucket128 + position-scale10 + reuse-base` trace
shows why `speclink_t08` loses accepted-token value:

| metric | value | read |
| --- | ---: | --- |
| requested residual fraction | `0.9583` | the policy asks to correct almost every draft row |
| effective residual fraction | `0.1629` | the fixed bucket cap excludes most requested residual rows |
| accepted requested-base fraction | `0.0572` | the logical policy itself misses some accepted rows |
| accepted effective-base fraction | `0.4848` | after bucket capping, almost half of accepted draft tokens are effectively base-only |
| mean accepted/step in traced run | `1.1553` | still well below dense EAGLE3 and `base_only_24` |
| mean requested residual rows/request-step | `7.6666` | too many requested rows for bucket128 at bs128 |

Offline projection suggested `critical_prefix@0.4` could reduce requested
residual rows to about `3.96` per request-step with accepted/rejected base-only
risk near `0.09/0.10`. The actual clean run was negative: total tok/s fell to
`1878.287`, full-batch tok/s to `2646.401`, and accepted draft tokens/step
stayed near `1.052`. This means the next route cannot be only "request fewer
residual rows"; it must preserve or improve the verifier's accepted length in
the live SR24 path.

A 2026-06-27 prefix-confidence follow-up made that tradeoff explicit. The
offline projection over the same trace estimated
`SPECLINK_SR24_SELECTIVE_RESIDUAL_POLICY=prefix_confidence` with
`SPECLINK_SR24_PREFIX_THRESHOLD=0.05` at residual draft fraction `0.5351`,
accepted-base risk `0.0482`, rejected-base risk `0.0721`, and `4.281` residual
draft rows/step. The live bs64/math/max128 graph-enabled run with
`down_proj,gate_up_proj` recovered accepted draft/step to `1.416` versus dense
`1.395`, but full-batch throughput was only `2073.671` versus dense
`3019.809` (`0.687x`) with GPU util `88.1%` and server graph profile
`{"FULL": 49}`. Restricting the same policy to `gate_up_proj` improved
full-batch throughput to `2323.575` versus dense `3024.459` (`0.768x`) and kept
accepted draft/step at `1.406` versus dense `1.396`, but it was still far below
dense. This confirms that protecting likely accepted prefix rows can repair
accepted-length value, but the current mixed sparse+residual operator remains
too expensive even with CUDA Graph coverage and a narrower target leaf.

The operator-side follow-up added `SPECLINK_SR24_ROUTE_ALL_RESIDUAL_ROWS=1` for
dense_rows residuals. This routes residual rows directly to dense Linear and
base rows directly to sparse Linear, avoiding the old duplicate sparse-base pass
on rows that would be overwritten. With `prefix_confidence@0.05`, `gate_up_proj`
only, bucket512, and CUDA Graph bucket enabled, the clean bs64/math/max128 run
improved SR24 full-batch throughput to `2961.618 tok/s` versus same-root dense
`3018.108 tok/s` (`0.981x`) while keeping accepted draft/step at `1.399` versus
dense `1.395`. Disabling `route_contiguous_fastpath` was neutral
(`2949.227 tok/s`, `0.974x` same-root dense), so the suspected `torch.equal`
contiguous check is not a first-order CPU-sync issue.

The matching CPU-sync ablation on this route-all path shows that the old
sync-heavy route is still bad (`2195.085` full-batch tok/s), but the low-sync
variants are all in the same short-run band (`2806-2954` full-batch tok/s).
Therefore the next optimization should not be another CPU synchronization
cleanup. The refreshed route-all component report attributes the mixed Linear
cost to base sparse GEMM `0.676ms/call` plus dense correction GEMM
`0.184ms/call`, while route-all gather/scatter is `0.093ms/linear` and remains
secondary. A graph-off dense-zero base ablation did not clearly beat graph-off
torch-sparse (`2615.838` vs `2619.522` full-batch tok/s). A small-base dense
fallback test with `SPECLINK_SR24_ROUTE_MIN_BASE_ROWS=160` was also negative:
SR24 full-batch fell to `2813.306 tok/s` versus same-root dense
`3020.687 tok/s`. So the actionable direction is a fused/packed mixed operator
or a better base-side sparse kernel under CUDA Graph, not a dense-zero
replacement or small-base full-dense fallback path.

## Next Experiments

Use the breakdown above before another controller sweep.

1. Replace dynamic route-all row-list materialization. The split counters show
   bucket build is negligible and `mixed_row_indices` is the large wait. Avoid
   `residual_mask.nonzero()` / `(~residual_mask).nonzero()` in the serving path.
2. Do not use the current fixed bucket512 dense-correction path as the final
   answer. It proves fixed shapes can eliminate scheduler row-list overhead,
   but it drops to `0.672x` dense because it pays base sparse plus 512-row
   dense correction each call.
3. Do not use CPU draft-score readback as the row-list fix. It turns the
   `nonzero` wait into an equivalent GPU-to-CPU score synchronization wait.
4. Treat fixed_prefix H=0/H=2 as operator ceiling probes, not final quality
   fixes. They remove scheduler overhead and reduce correction rows, but still
   stay below dense because the current gate/up sparse base is not fast enough
   in serving.
5. Prototype a true mask-aware route-all Linear/Triton/CUDA operator, or a
   compact per-request prefix representation, so the kernel avoids dynamic row
   tensors without correcting a large fixed bucket of mostly low-value rows.
6. Keep CUDA Graph coverage. The force-NONE control reduced the row-index wait
   but lost all graph steps and was slower overall.
7. Keep accepted draft/step as a hard guardrail. The latest route-all candidate
   repaired accepted length; do not accept a speed optimization that drops it
   back near `1.05`.
8. Add a trace-backed route that accounts for bucket capacity directly. The
   current trace records requested/effective residual mismatch; a useful next
   policy should choose the top bucket rows with a per-request accepted-prefix
   value signal instead of requesting almost all rows and relying on global
   bucket truncation.
9. Use `prefix_confidence@0.05 + gate_up_proj + route_all` as the current
   speed diagnostic, not as a final win. It reaches about `0.98x` same-root
   dense with accepted length intact, proving that avoiding duplicate
   sparse-base work matters, but it is still below dense.
10. Keep every candidate reporting the same seven fields: scheduler/mask,
   base sparse, residual correction, gather/scatter, routing stats, CUDA Graph,
   and GPU util.
11. Treat Triton Linear assembly alone as secondary unless it also reduces the
   base sparse GEMM cost or the scheduler row-index/bucket window. The measured
   Linear gather/scatter is `0.093ms/linear`, below the base sparse and broad
   clean scheduler costs.
12. Treat direct-position bucket construction as a negative/secondary ablation
   unless it is paired with a stronger value signal. The latest device-vector
   implementation reduced part of bucket64 builder overhead, but bs16 smokes
   still accepted only `0.44-0.54` draft tokens/step. Position-major early rows
   alone are not enough to recover useful accepted-token value.
13. The next implementation should be confidence/value-aware first and
   scheduler-efficient second: choose rows that are likely to affect accepted
   draft length, then represent them without dynamic CPU/GPU synchronization.
   Do not start with another threshold-only sweep or an assembly-only kernel
   unless it is explicitly labeled as an ablation.

## Priority-Signal And Row-Routed Check, 2026-06-27

The capped residual-bucket priority now uses the selected policy's value signal:
`high_confidence` ranks by DLM selected-token probability,
`prefix_confidence` ranks by cumulative prefix probability, and
`low_confidence` keeps the previous risk-based severity. The GPU correctness
smoke now asserts both slow/batched priority equivalence and the expected
priority direction for high-confidence and prefix-confidence rows:

```text
conda run -n spec python examples/evaluate/eval-guidellm/scripts/check_speclink_sr24_correctness.py
```

The focused serving rerun that actually motivated this change was neutral:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_prioritysignal_bucket512_bs64_math128_20260627/clean_serving/report.md
```

It used Llama-3.1-8B, `math_reasoning`, bs64, EAGLE3 K=8, max new tokens 128,
`prefix_confidence@0.05`, `gate_up_proj=16-31`, bucket512, CUDA Graph bucket,
and no row-routed MLP. The result was essentially unchanged from the earlier
prefix-confidence bucket512 run: dense full-batch `3032.967 tok/s`, SR24
full-batch `2078.159 tok/s`, accepted draft/step `1.416` for SR24 versus
`1.396` for dense, and CUDA Graph profile `{"FULL":49}`. Therefore, for this
non-row-routed path, better priority ranking does not address the first-order
operator cost.

The explicit row-routed MLP variant was worse:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefixconf_prioritysignal_rowrouted_bucket512_bs64_math128_20260627/clean_serving/report.md
```

With the same setup plus `--sr24-row-routed-mlp`, SR24 full-batch throughput
fell to `1951.156 tok/s`, accepted draft/step dropped to `1.223`, and average
GPU utilization fell to `63.5%`, even though graph profile still reported
`{"FULL":49}`. A new unit check verifies that `row_routed_mlp_output()` is
equivalent to the linear-level mixed path for a small exact-down MLP case, so
the serving regression is not currently explained by an MLP arithmetic
correctness bug. The likely issue is row selection / bucket shape / underfilled
row-routed work in serving. Do not promote row-routed MLP until a breakdown row
shows accepted draft/step and GPU util both recover.

## Corrected Row-Routed Overlap Breakdown, 2026-06-28

The follow-up constrained both residual leafs to the intended late-layer
overlap:

```text
--sr24-target-leafs down_proj,gate_up_proj
--sr24-residual-target-leafs down_proj,gate_up_proj
--sr24-residual-layer-ids-by-leaf 'gate_up_proj=16-31;down_proj=16-31'
--sr24-row-routed-mlp
```

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_overlap16_31_breakdown_bs64_math128_20260628/seven_part_report/report.md
```

Clean serving improved over the earlier unconstrained row-routed check, but it
is still not a speed path:

| method | full-batch tok/s | total tok/s | vs dense full | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 baseline | `3026.088` | `2185.578` | `1.000x` | `84.8%` | `1.403` | n/a |
| corrected row-routed `speclink_t08` | `2597.598` | `1840.189` | `0.858x` | `86.2%` | `1.230` | `{"FULL":62,"NONE":2}` |

Seven-part read:

| part | measured value | read |
| --- | ---: | --- |
| scheduler / mask build | clean `1.310ms/step`; row bucket/index `1.040ms/step`; bucket build `0.097ms/step`; mixed row indices `0.001ms/step` | visible but not large enough to explain the whole gap |
| base sparse linear | row-routed base total `0.577ms/call`; base gate/up `0.572ms/call`; base rows/call `143.7` | sparse base is still a large GPU-side cost |
| residual correction | row-routed dense total `0.760ms/call`; dense gate/up `0.477ms`; dense activation `0.019ms`; dense down `0.257ms`; dense rows/call `434.7` | correction is larger than sparse base, so the mixed MLP does too much dense work |
| gather/scatter | dense gather `0.007ms`, base gather `0.004ms`, assemble `0.022ms` | secondary |
| routing statistics | draft residual/base `12223/18889`; non-draft residual/base `3889/3788`; bucket fill `0.483`; bucket actual/requested `11634/16127` | many rows are corrected, but accepted draft length still drops |
| CUDA Graph | clean `{"FULL":62,"NONE":2}` | graph coverage is healthy |
| GPU util | clean avg `86.2%`, peak `100%` | GPU is busy; the issue is inefficient useful work, not idleness |

This answers the current slowdown question more sharply than the older broad
diagnosis. The corrected row-routed path is not mainly slow because of CUDA
Graph misses, gross GPU underutilization, or Python stats overhead. It is slow
because it reduces useful speculative progress (`1.230` accepted draft
tokens/step versus dense `1.403`) while also paying both sparse base work and
a larger dense correction MLP. In other words, it is a quality/selection problem
and an operator-efficiency problem at the same time.

Do not keep optimizing row-routed MLP assembly alone. A useful next candidate
must first preserve accepted draft length, then prove that dense correction rows
are much smaller than the sparse base rows or that the base/correction work is
fused into one efficient operator. If it cannot meet that guardrail, fall back
to the current best diagnostic line, `prefix_confidence@0.05 + gate_up_proj +
route_all`, which is closer to dense throughput while preserving accepted
length.

## Base-Only And All-Corrected Operator Refresh, 2026-06-28

This refresh uses the same late-MLP scope to separate the first two goal
questions:

```text
--sr24-target-leafs down_proj,gate_up_proj
--sr24-base-only-layer-ids-by-leaf 'gate_up_proj=16-31;down_proj=16-31'
--sr24-residual-layer-ids-by-leaf 'gate_up_proj=16-31;down_proj=16-31'
```

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_latemlp_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_default_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_directcslt_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_operator_probe_20260628/summary.md
```

Clean serving:

| method | residual backend | full-batch tok/s | total tok/s | vs same-root dense full | accepted draft/step | avg GPU util | CUDA Graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| late-MLP dense EAGLE3 | n/a | `3021.525` | `2185.492` | `1.000x` | `1.395` | `80.6%` | n/a |
| late-MLP `base_only_24` | none | `5389.838` | `2739.866` | `1.784x` | `2.444` | `81.8%` | `{"FULL":69,"NONE":2}` |
| late-MLP dense EAGLE3 | n/a | `3016.036` | `2180.028` | `1.000x` | `1.396` | `80.6%` | n/a |
| late-MLP `all_corrected_24` | `torch_sparse` | `2551.468` | `1778.923` | `0.846x` | `1.400` | `88.4%` | `{"FULL":77,"NONE":2}` |
| late-MLP `all_corrected_24` + direct cuSPARSELt | `torch_sparse` | `2515.072` | `1752.827` | `0.804x` | `1.400` | `89.0%` | `{"FULL":77,"NONE":2}` |

Read:

1. Under this scope, `base_only_24` is not slow. It has normal CUDA Graph
   coverage, similar GPU utilization to dense, and higher accepted draft/step.
   The earlier "base_only slow" concern should be treated as a scope/operator
   regression only if a fresh same-condition run contradicts this row.
2. `all_corrected_24` is slow even though accepted draft/step is essentially
   unchanged from dense and GPU utilization is higher. The bottleneck is the
   exact correction operator, not acceptance collapse or GPU idleness.
3. Direct `_cslt_sparse_mm` does not fix serving throughput. It is slightly
   worse than the default PyTorch sparse path in this row, so do not enable
   direct cuSPARSELt for `all_corrected_24` by default.
4. The operator probe says the same thing without vLLM. For Llama gate/up,
   the best current exact graph path is still `1.34-1.40x` dense time; for
   down-proj it is `1.14-1.52x` dense time depending on rows. Cached
   `compressed_dense` is GPU-resident, but its all-corrected graph path is
   still slower than the two-sparse-GEMM path. The next real all-corrected
   optimization needs a fused packed base+residual kernel; another Python
   dispatch flag or direct cuSPARSELt wrapper is not enough.

## Critical-Prefix Quality Gate And Breakdown, 2026-06-28

The quality-safe selector that should be used for the next SR24 iteration is:

```text
--sr24-selective-residual-policy critical_prefix
--sr24-threshold 0.6
--sr24-selective-min-prefix-residual 4
--sr24-selective-extra-after-low 1
--sr24-selective-non-draft-policy bonus
--sr24-target-leafs gate_up_proj,down_proj
--sr24-residual-target-leafs gate_up_proj,down_proj
--sr24-residual-layer-ids-by-leaf 'gate_up_proj=16-31;down_proj=8-15'
```

Quality gate artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_gsm8k50_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_gsm8k50_20260628/trace_analysis/report.md
```

Result:

| run | score | accepted base-only frac | rejected base-only frac | mean accepted/step | mean residual rows/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `critical_prefix@0.6,prefix4,extra1` GSM8K-50 | `0.7200` | `0.0228` | `0.0389` | `1.4501` | `4.4551` |

This matches the previous dense/spec-safe GSM8K-50 aggregate score while
reducing the high-confidence row-routing risk seen in the failed
`high_confidence@0.7,prefix3` gate. That failed gate had score `0.7000`,
accepted base-only fraction `0.0441`, and rejected base-only fraction `0.1524`.

Same-condition CPU-sync serving ablation:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/clean_serving_nosync/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/clean_serving_syncmask/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_bs64_math128_20260628/clean_serving_uniformdirect/report.md
```

All rows are Llama-3.1-8B, `math_reasoning`, client-side concurrency 64,
EAGLE3 K=8, max new tokens 128.

| variant | full-batch tok/s | total tok/s | accepted draft/step | avg GPU util | scheduler mask ms/step | mask-state ms/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense reference from no-sync root | `3014.282` | `2180.503` | `1.395` | `80.3%` | n/a | n/a | n/a |
| `critical_prefix` no-sync | `2807.819` | `1779.830` | `1.632` | `79.9%` | `0.343` | `0.000` | `{"NONE":76}` |
| `critical_prefix` + mask-state sync | `2691.279` | `1716.161` | `1.632` | `74.8%` | `5.411` | `5.043` | `{"NONE":76}` |
| `critical_prefix` + uniform-direct builder | `2769.327` | `1735.971` | `1.580` | `74.2%` | `0.774` | `0.000` | `{"NONE":78}` |

Read: reducing CPU synchronization matters. The explicit mask-state sync costs
about `5ms/step` and lowers full-batch throughput by about `4%` relative to the
same-condition no-sync row. The `uniform-direct` builder is negative in this
configuration because its batched-builder wall time increases. Keep the normal
batched mask builder plus no mask-state sync for the current candidate.

Full seven-part breakdown artifact:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_breakdown_bs64_math128_20260628/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_breakdown_bs64_math128_20260628/component_summary/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_breakdown_bs64_math128_20260628/component_microbench/summary.md
```

Clean serving row:

| method | full-batch tok/s | total tok/s | vs dense full | avg GPU util | accepted draft/step | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3132.894` | `2183.544` | `1.000x` | `86.4%` | `1.401` | n/a |
| `critical_prefix` SR24 | `2804.983` | `1780.379` | `0.895x` | `77.2%` | `1.635` | `{"NONE":64}` |

Seven-part read:

| part | measured value | read |
| --- | ---: | --- |
| scheduler / mask build | clean `0.372ms/step`; batched builder `0.195ms`; bucket/index `0.109ms`; bucket build `0.107ms` | low-sync clean path is already sub-ms |
| base sparse linear | instrumented sparse base `0.985ms/call`; gate_up 16-31 `1.076ms/call`; down 8-15 `0.804ms/call` | largest measured Linear-side cost |
| residual correction | dense-row correction `0.148ms/call`; gate_up dense `0.171ms`; down dense `0.101ms` | secondary in this candidate |
| gather/scatter | `0.012ms/call` | not the current first bottleneck |
| routing stats | draft residual/base `8217/7215`; non-draft residual/base `1929/3801`; bucket fill `0.979` | many draft rows still use residual correction |
| CUDA Graph | clean `{"NONE":64}` | mixed path is entirely graph-off |
| GPU util | clean avg `77.2%`, peak `99%` | not fully idle, but lower than dense; useful-work efficiency is weak |

Component microbench for representative 256-row Llama MLP shapes confirms the
operator direction:

- For `gate_up` shape `256x4096x14336`, base sparse alone is about dense
  (`0.164ms` versus `0.165ms`), but the mixed base+correction path is
  `1.58-2.19x` dense time as residual fraction goes from `0.25` to `1.0`.
- For `down_proj` shape `256x14336x4096`, base sparse alone is faster than
  dense (`0.101ms` versus `0.144ms`), but mixed base+correction is still
  `1.42-2.35x` dense time.

Current conclusion: the CPU-sync reduction should stay enabled, but it is not
enough. The remaining clean gap is mainly the combination of CUDA Graph loss for
mixed SR24 steps and an inefficient mixed sparse-base plus residual-correction
operator. The next optimization should either make this mixed path graph-safe
with static tensors or replace the two-pass mixed Linear with a fused/packed
operator. Threshold-only controller sweeps are lower priority unless they
reduce residual rows without losing accepted draft length.

## Mixed CUDA Graph Positive Check, 2026-06-28

The next check turned on the existing graph-safe mixed path for the current
quality selector:

```text
--sr24-dynamic-auto-cudagraph
--sr24-cudagraph-bucket
--no-sr24-force-cudagraph-none-for-mixed
```

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_bs64_math128_20260628/clean_serving/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_critical_t06_prefix4_extra1_graphon_paired_gsm8k50_20260628/report.md
```

Serving result, Llama-3.1-8B `math_reasoning`, bs64, K=8, max new tokens 128:

| method | full-batch tok/s | total tok/s | vs dense full | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 | `3026.614` | `2184.300` | `1.000x` | `1.396` | `80.75%` | n/a |
| graph-on `critical_prefix` SR24 | `3118.481` | `2003.767` | `1.030x` | `1.569` | `84.75%` | `{"FULL":78,"NONE":2}` |

Paired GSM8K-50:

| mode | score | pair reg | pair imp |
| --- | ---: | ---: | ---: |
| dense EAGLE3 | `0.7200` | `0` | `0` |
| graph-on SR24 | `0.7400` | `1` | `2` |

This changes the current optimization priority. The graph-miss part of the
slowdown is actionable and already recovers enough performance to beat dense
on the full-batch window. However, the result is still only `1.03x`, not the
requested `1.2x`. Further threshold-only work is unlikely to be enough unless
it cuts residual work without hurting accepted draft length. The next useful
experiments should keep this graph-on configuration and reduce the operator
work, starting with narrower residual leafs/layers, then moving to a fused
packed base+residual Linear if scope reduction cannot preserve quality.

Follow-up scope/controller ablations were negative:

| variant | artifact | full-batch tok/s | accepted draft/step | read |
| --- | --- | ---: | ---: | --- |
| `gate_up_proj=16-31` only | `results.bak/sr24_critical_t06_prefix4_extra1_graphon_gateup16_31_bs64_math128_20260628/clean_serving/report.md` | `2968.512` | `1.493` | worse than dense and full-leaf graph-on |
| `down_proj=8-15` only | `results.bak/sr24_critical_t06_prefix4_extra1_graphon_down8_15_bs64_math128_20260628/clean_serving/report.md` | `2953.232` | `1.443` | worse than dense and full-leaf graph-on |
| threshold `0.7`, full leafs | `results.bak/sr24_critical_t07_prefix4_extra1_graphon_full_bs64_math128_20260628/clean_serving/report.md` | `3088.045` | `1.631` | below threshold `0.6` despite higher accepted length |
| bucket16, full leafs | `results.bak/sr24_critical_t06_prefix4_extra1_graphon_bucket16_bs64_math128_20260628/clean_serving/report.md` | `3108.951` | `1.579` | no speed gain over bucket32 |

These rows show that the current `1.2x` gap is not solved by simply correcting
fewer leafs, changing threshold, or capping the bucket smaller. The graph-on
candidate should remain the baseline for the next implementation pass, and the
next pass should target the mixed Linear operator itself.

## All-MLP Bucket And CPU-Sync Follow-Up, 2026-06-28

A later follow-up tested an all-MLP residual scope
(`gate_up_proj,down_proj` across all layers) with graph-capable mixed SR24 on
Llama-3.1-8B `math_reasoning`, GuideLLM client-side concurrency 64, EAGLE3 K=8,
and max new tokens 128.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bs64_math128_20260628_0220/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_stats_off_bs64_math128_20260628_0405/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket16_bs64_math128_20260628_0415/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket64_bs64_math128_20260628_0418/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket32_bonus1_bs64_math128_20260628_0425/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_bucket32_pos10_bs64_math128_20260628_0430/report.md
```

| variant | total tok/s | full-batch tok/s | full vs dense | accepted draft/step | CUDA Graph | read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| bucket16 | `1732.780` | `3394.140` | `1.122x` | `2.397` | `{"FULL":53,"NONE":11}` | better total than bucket32, worse full-batch |
| bucket32 default | `1717.158` | `3545.363` | `1.172x` | `2.345` | `{"FULL":55,"NONE":9}` | best full-batch row |
| bucket32 stats off | `1710.424` | `3462.887` | `1.145x` | `2.385` | `{"FULL":49}` | no hidden CPU-sync win |
| bucket32 bonus1 | `1555.200` | `2746.878` | `0.908x` | `1.872` | `{"FULL":81,"NONE":15}` | accepted length collapses |
| bucket32 pos10 | `1530.576` | `2874.849` | `0.950x` | `1.720` | `{"FULL":85,"NONE":11}` | accepted length collapses |
| bucket64 | `1626.912` | `3284.260` | `1.086x` | `2.067` | `{"FULL":84,"NONE":12}` | larger bucket overcorrects |

The same-root dense row for the default all-MLP run was `3025.159` full-batch
tok/s and `2187.155` total tok/s. The all-MLP bucket32 path reaches
`1.172x` full-batch throughput, but it still loses on total throughput. This
supports the same diagnosis as the seven-part breakdown: CPU synchronization is
worth controlling, but it is not the main bottleneck after the graph-capable
path is enabled. The remaining gap is the mixed Linear useful-work shape:
sparse base is computed for rows that then also pay dense residual correction.

## All-MLP Triton Override Gate, 2026-06-28

I tested the all-MLP graph-capable path with `SPECLINK_SR24_TRITON_BUCKET_OVERRIDE=1`
to distinguish an implementation-speed upper bound from a quality-safe route.

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_bs64_math128_20260628_0500/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_paired_gsm8k50_20260628_0510/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix5_bs64_math128_20260628_0540/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_prefix6_bs64_math128_20260628_0530/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_lowconf_prefix4_bs64_math128_20260628_0550/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_t08_tritonoverride_lowconf_prefix4_paired_gsm8k50_20260628_0600/report.md
```

| variant | full-batch tok/s | full vs dense | total tok/s | accepted draft/step | quality gate |
| --- | ---: | ---: | ---: | ---: | --- |
| bucket32 default | `3545.363` | `1.172x` | `1717.158` | `2.345` | not gated here |
| Triton override, `critical_prefix`, prefix4 | `3645.945` | `1.205x` | `1993.722` | `2.379` | GSM8K-50 `0.7000` vs dense `0.7200`, pair reg `4`, pair imp `3` |
| Triton override, `critical_prefix`, prefix5 | `3560.886` | `1.178x` | `1903.229` | `2.377` | not gated |
| Triton override, `critical_prefix`, prefix6 | `3579.986` | `1.190x` | `1903.661` | `2.377` | not gated |
| Triton override, `low_confidence`, prefix4 | `3622.772` | `1.198x` | `1925.328` | `2.393` | GSM8K-50 `0.6800` vs dense `0.7200`, pair reg `4`, pair imp `2` |

Conclusion: Triton bucket override is a real speed optimization and reaches the
`1.2x` full-batch target for the aggressive all-MLP route, but the paired
accuracy gate fails. Prefix expansion falls below target, and `low_confidence`
is worse on quality. Treat this as a speed upper bound, not the default SR24
candidate.

The doc-id 11 replay artifact below generated the correct answer in offline
single-sample replay, while the paired serving run regressed the same sample:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_replay_doc11_prefix4_debug_20260628_0520/gsm8k_cot_doc11_selective_prefix4.json
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_replay_doc11_prefix4_debug_20260628_0520/speclink_sr24_debug_trace.jsonl
```

So the next correctness pass should collect serving-shape regression traces.
For speed, the next pass should focus on fused/packed mixed Linear work or a
selector that cuts residual-row fraction much more sharply without lowering
accepted draft length. Another scalar threshold/bucket sweep is not the right
main line unless it changes that seven-part breakdown.

## Gate-Up16 Graph And CPU-Sync Check, 2026-06-28

I also reran the narrower `gate_up_proj=16-31` high-confidence selector with
the graph-safe mask path enabled:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_graphon_bs64_math128_20260628/report.md
```

The matching graph-off focused breakdown is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_gateup16_bs64_math_k8_20260628/seven_part_report_direct/report.md
```

| variant | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense EAGLE3 reference | `2368.916` | `2763.202` | `1.402` | `88.64%` | n/a |
| gate-up16 SR24, graph off | `1904.859` | `2294.100` | `1.453` | `91.24%` | `{"NONE":128}` |
| gate-up16 SR24, graph on | `1992.979` | `2314.728` | `1.476` | `93.13%` | `{"FULL":107,"NONE":32}` |
| gate-up16 SR24, graph on, per-step mask-state sync | `1967.624` | `2282.304` | `1.455` | `92.71%` | `{"FULL":109,"NONE":30}` |

Graph-safe mask state improves the total tok/s from `1904.859` to
`1992.979`, and it restores most decode steps to FULL CUDA Graph mode. The
full-batch number barely moves, however, and remains well below dense. This
confirms that graph loss and CPU synchronization are only part of the slowdown.
The larger bottleneck is still the two-pass mixed operator: sparse base work
plus dense-row residual correction.

The explicit per-step mask-state sync ablation is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_graphon_syncstate_bs64_math128_20260628/report.md
```

It is slightly slower than the no-sync graph-on run while keeping almost the
same CUDA Graph coverage. This supports keeping `--no-sr24-sync-mask-state`
and the batched/static mask path enabled for future candidates, but it also
shows that this synchronization is not the large missing speedup.

Correctness replay for the same graph-off/graph-on configuration:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_graph_correctness_replay_20260628/compare/report.md
```

The replay used `gsm8k_cot` doc id `0`, Llama-3.1-8B EAGLE3 K=8, max new
tokens 128, `gate_up_proj=16-31`, `high_confidence`, reduced CPU sync, static
mask buffer, and batched mask builder. Graph-off and graph-on generated the
same `65` token ids, with the same cumulative logprob `-13.578305416107469`.
So this graph-safe/reduced-sync path is correct on the replay smoke, but it is
not fast enough by itself.

## Route-Reuse Row-Index Check, 2026-06-28

I then tested whether avoiding the full dense correction GEMM is more important
than CUDA Graph coverage. The candidate was the same gate-up16 high-confidence
selector plus:

```text
--sr24-route-reuse-base-output
```

Artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_route_reuse_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_highconf_route_reuse_eager_bs64_math128_20260628/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_route_reuse_correctness_replay_20260628/compare_three/report.md
```

| variant | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA Graph | correctness read |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| graph-on, no route-reuse | `1992.979` | `2314.728` | `1.476` | `93.13%` | `{"FULL":107,"NONE":32}` | replay exact vs eager |
| route-reuse with dynamic graph | `2141.540` | `2503.360` | `1.440` | `91.07%` | `{"FULL":110,"NONE":32}` | selected tokens same, logprobs drift |
| route-reuse eager-safe | `1871.695` | `2222.398` | `1.466` | `89.88%` | `{"NONE":139}` | replay exact vs eager |

The three-way replay shows:

```text
eager cumulative logprob:             -13.578305416107469
route-reuse eager cumulative logprob: -13.578305416107469
route-reuse graph cumulative logprob: -16.11578315886254
```

So `route_reuse_base_output` is mathematically safe in eager mode, but the
dynamic graph combination is unsafe because row-index tensors can drift under
graph replay. I updated the matrix runner so `route_bucket_rows`,
`route_all_residual_rows`, and `route_reuse_base_output` are no longer allowed
through the dynamic-auto CUDA Graph gate. The safe eager route-reuse run is
slower than the graph-safe no-route run, so this is not the next speed path.

Read: row-index routing confirms the hypothesis that full dense correction is
wasteful, but the current implementation only wins when it uses an unsafe
graph combination. A useful next implementation would need graph-stable
fixed-shape row routing or a fused packed mixed Linear; the existing eager
index-select/scatter path is not enough.
