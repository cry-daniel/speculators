# SR24 Current Slowdown Breakdown, 2026-06-29

This note records the current pivot: before more controller sweeps, first
measure where SR24 is slow. The newest explicit user-requested breakdown is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_explicit_breakdown_bs64_math256_20260629/
```

Read these reports first:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_explicit_breakdown_bs64_math256_20260629/seven_part_report_with_graph/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_explicit_breakdown_bs64_math256_20260629/component_summary_with_graph/report.md
```

The previous offline reducer output is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_pivot_breakdown_refreshed_20260629/
```

Supporting quality/stability gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_stability_gate_current_plus_unsafe50_20260629/
```

Dynamic mixed CUDA Graph throughput check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_t08_graph_current_bs64_math256_20260629/
```

Small dynamic mixed CUDA Graph quality sanity:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_dynamic_t08_graph_current_quality_gsm8k20_20260629/
```

Low-residual t08 routing candidates:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_clean_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_bucket8_clean_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_bucket4_clean_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_predbonus_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_quality_gsm8k20_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap2_quality_gsm8k50_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_riskcap2_stability_gate_gsm8k50_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_lowresidual_gateup_riskcap1_clean_bs64_math256_20260629/
```

All-MLP speed-target candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_nostats_bs64_math128_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_preset_repro3_bs64_math128_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_streaming_bs64_math128_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_streaming_nochunk_bs64_math128_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_preset_gsm8k20_current_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_paired_gsm8k50_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_stability_gate_20260629/
```

Bucket/copy negative controls from this follow-up pass:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_bucket16_fullcopy_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_bucket8_fullcopy_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_t08_bucket16_activeonly_fixed_bs64_math256_20260629/
```

Existing Triton bucket-kernel sweep:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_param_sweep_bucket16_20260629/
```

Compressed-residual kernel and serving checks:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_profile_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cached_gateup16_31_bs64_math256_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compressed_cached_gateup16_31_bs64_math256_mem97_20260629/
```

## Short Answer

The current slowdown is not mainly caused by low accepted length, normal
scheduler overhead, or idle GPU. In the explicit bs=64/math_reasoning run,
`speclink_t08` has high GPU utilization, while its clean low-sync mask build is
sub-ms per step.

The first-order slowdown is mixed CUDA Graph coverage plus useful-work
efficiency in the sparse-base plus residual-correction path:

- `base_only_24` is faster, so the sparse base has a real speed upper bound.
- `all_corrected_24` dense-fastpath is near dense, so SR24 attachment/accounting
  is not the root problem.
- dense has graph counts `FULL=115, NONE=76, PIECEWISE=1`, `base_only_24` has
  `FULL=126, NONE=2`, but guarded `speclink_t08` has `NONE=128`.
- enabling dynamic mixed CUDA Graph replay changes `speclink_t08` to
  `FULL=126, NONE=2`, but full-batch throughput is still only `2900 tok/s`
  versus dense `3555 tok/s`.
- `speclink_t08` routes many rows to residual correction. In the explicit
  diagnostic row, draft residual/base rows are `8640/4608` and non-draft
  residual/base rows are `1656/1635`.
- Safe graph guard is correctness-safe but very slow because it forces decode
  steps to CUDA Graph `NONE`.
- Dynamic mixed CUDA Graph replay is necessary, but not sufficient. After graph
  coverage is fixed, the remaining bottleneck is still the two-pass
  sparse-base plus dense-row residual-correction operator at the current
  residual-row fraction.
- A lower-residual gate-up-only candidate with `low_confidence`, prefix2, and
  riskcap2 improves clean serving to only `1.07x-1.09x` dense, not the `1.2x`
  target. This confirms that routing improvements help but are not enough with
  the current bucketed dense-row correction path.
- The only measured route that has reached the `1.2x` full-batch target is the
  all-MLP `low_confidence@0.6` prefix5 candidate with Triton bucket override:
  `3650.240` versus dense `3036.970` full-batch tok/s on bs64/math/max128.
  A current 3-repeat preset reproduction reports median dense/SR24 full-batch
  `3038.443/3648.217` tok/s, i.e. `1.201x`, with CUDA Graph `{"FULL":49}`.
  This is not yet a proven final answer because fixed-request total tok/s was
  still lower than dense (`2272.489/1925.203` tok/s in the 3-repeat median),
  and paired GSM8K-50 had a small aggregate drop (`0.7400 -> 0.7200`),
  although dense-repeat stability analysis found zero stable regressions. A
  current GSM8K-20 preset sanity gate reported dense/SR24 `0.6000/0.7000`,
  with paired regressions/improvements `1/3`; treat this only as a smoke
  quality check, not as final accuracy proof.
- Continuous refill serving rejects the current all-MLP preset as a final
  throughput solution: with bs64/math/max128, dense/SR24 steady output tok/s
  is `2622.266/2288.145` (`0.873x`) and full-batch output tok/s is
  `2649.008/2351.428` (`0.888x`). SR24 still accepts more draft tokens
  (`2.392` versus dense `1.398` accepted draft tokens/step), but TTFT is much
  higher (`601ms` versus dense `199ms`) and average GPU util is lower
  (`78.4%` versus `96.2%`). Therefore the fixed-request `1.2x` full-batch
  number is only an early decode-window metric; the next speed work must
  address continuous refill/prefill interference and TTFT/utilization.
- Disabling vLLM chunked prefill did not fix this. In the same continuous
  setting with `--disable-chunked-prefill`, dense/SR24 steady output tok/s is
  `2628.897/2270.428`, and SR24 TTFT/GPU util are `608ms/75.6%`. This makes
  chunked prefill a negative/neutral scheduling ablation, not the root cause.
- The existing Triton bucket dense-GEMM path is not a quick fix. On the
  dominant `gate_up_proj` shape it is about `1.43x` dense and slower than the
  current bucket dense-copy path, so further speedup needs either a different
  fused/packed correction operator or a policy that drives corrected rows much
  lower without losing accuracy.
- `compressed_dense` is GPU-resident in the current implementation when
  `SPECLINK_SR24_RESIDUAL_DEVICE=cuda`, but it is not a speed path. The packed
  Triton residual kernel is `7-9x` slower than materializing the residual dense
  weight and using torch GEMM on the measured Llama MLP shapes. Cache+prewarm
  avoids repeated materialization and gives CUDA Graph `FULL=190,NONE=2`, but
  the serving result is still only `2788 full-batch tok/s` versus same-root
  dense `3489 tok/s` and the earlier `torch_sparse` residual `3033 tok/s`.

## Key Rows

| row | total tok/s | full-batch tok/s | accepted draft tokens/step | GPU util | read |
| --- | ---: | ---: | ---: | ---: | --- |
| dense, explicit clean | 2334.752 | 3482.041 | 1.734 | 88.429% | current same-root baseline |
| `base_only_24`, explicit clean | 2790.475 | 3961.598 | 2.027 | 90.833% | sparse-base upper bound exists |
| `speclink_t08`, explicit clean | 2063.403 | 2847.156 | 1.703 | 86.938% | slower than dense; CUDA Graph `NONE=128` |
| `speclink_t08`, dynamic graph | 2189.570 | 2900.334 | 1.729 | 92.467% | graph fixed to `FULL=126,NONE=2`, still below dense |
| `speclink_t08`, riskcap2 low-residual | 2551.424 | 3793.940 | 2.072 | 89.846% | best current speed row, `1.09x` dense, not `1.2x` |
| `speclink_t08`, riskcap2 clean/no stats | 2565.617 | 3726.061 | 2.014 | 90.154% | removing stats did not close the gap |
| `speclink_t08`, riskcap2 bucket8 clean | 2577.247 | 3733.398 | 2.022 | 90.077% | tiny bucket-size gain only, still `1.07x` dense |
| `speclink_t08`, riskcap2 bucket4 clean | 2638.117 | 3678.525 | 2.001 | 89.846% | lower bucket hurts full-batch throughput |
| `speclink_t08`, riskcap2 predicted bonus | 2333.322 | 3672.706 | 1.935 | 91.857% | avoiding always-correct bonus was slower |
| `speclink_t08`, all-MLP lowconf prefix5 Triton override | 1846.006 | 3650.240 | 2.334 | 84.444% | only current `1.2x` full-batch candidate; quality not fully proven |
| `speclink_t08`, all-MLP lowconf prefix5 preset repro3 median | 1925.203 | 3648.217 | 2.340 | 83.000% | current-code preset reproduction; full-batch `1.201x`, total tok/s still lower |
| `speclink_t08`, all-MLP lowconf prefix5 continuous refill | 2208.417 | 2351.428 | 2.392 | 78.365% | steady `2288.145`, only `0.873x` dense steady; TTFT/utilization problem |
| `speclink_t08`, all-MLP lowconf prefix5 continuous no chunked prefill | 2223.771 | 2342.866 | 2.429 | 75.558% | steady `2270.428`; disabling chunked prefill did not help |
| dense, paired clean | 2339.571 | 3490.544 | 1.734 | 88.429% | baseline |
| `speclink_t08`, paired clean | 2288.631 | 3430.631 | n/a | 91.000% | roughly tied/slower than dense |
| `base_only_24`, eager supplement | 2486.830 | 3940.794 | 2.202 | 78.923% | sparse-base upper bound exists |
| `all_corrected_24`, dense-fastpath | 2350.754 | 3536.688 | 1.735 | 89.429% | SR24 hook/accounting is not slow by itself |
| `all_corrected_24`, early dense eager | 2144.563 | 3320.947 | 1.703 | 90.867% | avoiding correction helps, but eager path costs |
| `all_corrected_24`, early dense graph | 1898.871 | 3120.939 | 1.697 | 77.588% | this graph/runner path was worse |
| `speclink_t08` adaptive fallback check | 2102.813 | 3278.684 | 1.701 | 92.125% | first run was forced eager; not final evidence |

## Seven-Part Breakdown

| part | what to measure | current evidence | current read |
| --- | --- | --- | --- |
| scheduler / mask build | per-step residual mask, bucket rows, routing state | explicit clean `speclink_t08`: `0.455 ms/step`; graph-counter row: `0.352 ms/step`; exact diagnostic row: `6.280 ms/step` | normal serving is not scheduler-bound; exact stats expose sync/routing overhead only |
| base sparse linear | sparse base time, especially `gate_up_proj` layers 16-31 | explicit diagnostic `speclink_t08`: `gate_up_proj[16-31]=0.605 ms/call`; `all_corrected_24=0.741 ms/call` | largest measured GPU-side component |
| residual correction | dense-row correction GEMM time | explicit diagnostic `speclink_t08`: `0.333 ms/call`; `all_corrected_24=0.480 ms/call` | large relative to sparse base, so high residual fraction erases sparse gains |
| gather/scatter | `index_select`, `index_add_`, bucket assembly | explicit diagnostic `speclink_t08=0.036 ms/event`; `all_corrected_24=0.028 ms/event` | not first bottleneck in current config |
| routing statistics | draft residual rows, non-draft residual rows, bucket fill | exact diagnostic `speclink_t08`: draft residual/base `8640/4608`; non-draft residual/base `1656/1635`; graph-counter row has non-draft residual fraction `0.623` | too many rows still take residual correction |
| CUDA Graph | dense/base-only/t08 `FULL` vs `NONE` graph steps | dense `FULL=115, NONE=76, PIECEWISE=1`; base-only `FULL=126, NONE=2`; guarded `speclink_t08 NONE=128`; dynamic-graph `speclink_t08 FULL=126,NONE=2` | graph miss is real, but fixing it alone does not make t08 faster than dense |
| GPU util | full GPU work vs underutilized small kernels | dense `88.429%`; base-only `90.833%`; `speclink_t08` clean `86.938%`, graph-counter `88.188%` | GPU is busy; problem is inefficient work and graph coverage, not idle GPU |

## Operator Evidence

For the dominant `gate_up_proj` shape `512 x 28672 x 4096`, the isolated
microbench shows the current mixed operator only wins when residual rows are
very low:

| residual fraction | dense graph ms | base sparse graph ms | current mixed graph ms | mixed / dense |
| ---: | ---: | ---: | ---: | ---: |
| 0.0625 | 0.538 | 0.353 | 0.532 | 0.99x |
| 0.1250 | 0.539 | 0.352 | 0.553 | 1.03x |
| 0.2500 | 0.539 | 0.352 | 0.617 | 1.14x |
| 0.5000 | 0.540 | 0.352 | 0.823 | 1.52x |

Current routing is around `0.53-0.55` residual fraction, so it lands in the bad
region where sparse base plus correction is much slower than dense for the
dominant gate/up shape. This explains why `base_only_24` can be fast while
`speclink_t08` cannot yet beat dense reliably.

I also swept the existing Triton bucket dense-GEMM path with bucket16 on
2026-06-29. The best rows were:

| shape | residual fraction | dense graph ms | base sparse graph ms | current mixed graph ms | bucket dense-copy graph ms | Triton bucket graph ms | Triton / dense | Triton / current mixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512 x 28672 x 4096` | 0.03125 | 0.539 | 0.352 | 0.539 | 0.534 | 0.770 | 1.43x | 1.43x |
| `512 x 28672 x 4096` | 0.0625 | 0.541 | 0.353 | 0.532 | 0.531 | 0.773 | 1.43x | 1.45x |
| `512 x 28672 x 4096` | 0.125 | 0.541 | 0.352 | 0.553 | 0.531 | 0.773 | 1.43x | 1.40x |
| `512 x 28672 x 4096` | 0.250 | 0.541 | 0.353 | 0.617 | 0.531 | 0.773 | 1.43x | 1.25x |
| `512 x 4096 x 14336` | 0.0625 | 0.292 | 0.166 | 0.263 | 0.271 | 0.291 | 1.00x | 1.11x |
| `512 x 4096 x 14336` | 0.125 | 0.292 | 0.166 | 0.266 | 0.271 | 0.290 | 0.99x | 1.09x |

The existing Triton implementation fuses row gather, dense GEMM, and scatter,
but it gives up cuBLAS efficiency on the very wide `gate_up_proj` output. For
that shape it is slower even when only bucket16 rows are corrected. This
removes one tempting shortcut: simply enabling
`SPECLINK_SR24_TRITON_BUCKET_DENSE_GEMM=1` is not a serving-speed solution.

For `all_corrected_24`, I also checked the `compressed_dense` residual path
directly. The implementation is GPU-resident when requested:

- attach stats report `compressed_residual_runtime_on_gpu=true` and
  `compressed_residual_non_gpu_modules=[]`,
- `residual_values` and `mask_bytes` are on CUDA for
  `SPECLINK_SR24_RESIDUAL_DEVICE=cuda`,
- `_compressed_residual_weight()` materializes the dense residual tensor on
  CUDA and then uses `F.linear`,
- the optional packed Triton path computes directly from packed residual values
  on CUDA.

The problem is not a CPU fallback; it is operator cost and memory pressure.

| check | result |
| --- | --- |
| packed Triton residual, `512 x 28672 x 4096` | best `3.929 ms` vs materialized dense residual GEMM `0.538 ms`, `7.31x` slower, max abs diff `0.5` |
| packed Triton residual, `512 x 4096 x 14336` | best `2.509 ms` vs materialized dense residual GEMM `0.293 ms`, `8.56x` slower, max abs diff `1.0` |
| cached/prewarmed compressed_dense at `gpu_memory_utilization=0.90` | failed startup: model load `20.73 GiB`, graph estimate `1.20 GiB`, available KV `-0.63 GiB` |
| cached/prewarmed compressed_dense at `gpu_memory_utilization=0.97` | started, CUDA Graph `FULL=190,NONE=2`, but only `2788 full-batch tok/s` vs same-root dense `3489 tok/s` |
| earlier `torch_sparse` residual all-corrected | `3033 full-batch tok/s` vs same-root dense `3476 tok/s` |
| dense-fastpath all-corrected control | `3537 full-batch tok/s` vs same-root dense `3483 tok/s`; this is the dense-equivalent/no-op control |

So the current optimized `all_corrected_24` story is:

- if the goal is an exact correctness control, keep the dense fastpath;
- if the goal is a sparse-operator ablation, `torch_sparse` residual is the
  best measured exact sparse path so far, but it is still below dense;
- `compressed_dense` with cache/prewarm proves the GPU path but is not the
  throughput solution;
- packed residual Triton should stay off until it is replaced with a different
  kernel design.

## Graph Safety Update

The earlier dynamic mixed CUDA Graph check had one dense-correct / SR24-wrong
paired difference on GSM8K20. That is no longer strong evidence by itself.
After adding repeated dense references, the stability gate reports:

| experiment | samples | exp acc | dense repeat unstable | stable regressions vs dense repeats |
| --- | ---: | ---: | ---: | ---: |
| dynamic graph unsafe GSM8K20 | 20 | 0.7000 | 5 | 0 |
| safe graph guard GSM8K50 | 50 | 0.7200 | 5 | 0 |
| sync-on guard GSM8K20 | 20 | 0.7000 | 5 | 0 |
| sync-off guard GSM8K20 | 20 | 0.7000 | 5 | 0 |
| dynamic graph unsafe GSM8K50 | 50 | 0.7200 | 5 | 0 |

So dynamic mixed graph replay should not be rejected solely because of the old
paired diff. It is still not globally proven safe; future speed runs need this
stability gate or a deterministic correctness check.

The current dynamic-graph sanity run on GSM8K20 with the same gate-up-only t08
flags reports:

| mode | accuracy | paired regressions | paired improvements | accepted/drafted |
| --- | ---: | ---: | ---: | ---: |
| dense baseline | 0.6000 | 0 | 0 | 1091/5960 |
| dynamic-graph `speclink_t08` | 0.6500 | 0 | 1 | 1133/6032 |

This is only a small sanity check, not a final quality result. It is enough to
continue using dynamic graph as the speed-diagnosis baseline, but not enough to
claim accuracy preservation.

The low-residual riskcap2 candidate also passed the same small paired sanity
check:

| mode | accuracy | paired regressions | paired improvements | accepted/drafted |
| --- | ---: | ---: | ---: | ---: |
| dense baseline | 0.6000 | 0 | 0 | 1091/5960 |
| riskcap2 `speclink_t08` | 0.7000 | 0 | 2 | 1180/6192 |

This makes it a useful speed/quality probe, but not a completed solution. It
needs a larger GSM8K/math gate before it can replace dense, and its throughput
is still below the requested `1.2x` dense target.

I expanded the same riskcap2 candidate to GSM8K50. The aggregate accuracy
matched dense EAGLE3:

| mode | accuracy | paired regressions | paired improvements | dense retention |
| --- | ---: | ---: | ---: | ---: |
| dense baseline | 0.7200 | 0 | 0 | 1.0000 |
| riskcap2 `speclink_t08` | 0.7200 | 1 | 1 | 0.9722 |

The one paired regression is not a stable regression under the repeated-dense
gate. Combining this run with an existing GSM8K50 dense-repeat root gives
`dense_repeat_unstable=2` and `stable_regressions_vs_dense_repeat=0`. The
specific differing samples are doc15 and doc16, where dense EAGLE3 itself
disagrees across repeats. This means the current riskcap2 candidate is not
proven quality-safe, but the latest evidence does not show a stable quality
drop on GSM8K50.

## Low-Residual Routing Update

The first actionable route after the seven-part breakdown was to reduce the
corrected-row budget:

- target only `gate_up_proj` layers 16-31,
- policy `low_confidence` with threshold `0.8`,
- force prefix2 residual rows,
- add at most two additional low-confidence rows by risk,
- keep `bonus` non-draft correction,
- use bucket16 plus dynamic mixed CUDA Graph.

This improved the best same-root full-batch throughput from the dynamic graph
baseline `2900 tok/s` to `3794 tok/s`, while dense in the same root was
`3485 tok/s`. The gain is real but too small: `1.09x`, not `1.2x`.

A `predicted_full_accept` non-draft policy removed non-draft residual work in
the summary, but it also reduced acceptance and throughput (`3673 tok/s`), so
the always-correct bonus row remains the better current route.

Turning off runtime stats did not help; the clean row reached only
`3726 tok/s`. This means the remaining gap is not mainly Python stats overhead.
The next optimization has to reduce actual GPU work: either fewer corrected
rows without losing accuracy, or a fused/packed correction kernel that makes
bucket16 correction cheap enough.

I also tried a more aggressive cap1 variant with
`selective_min_prefix_residual=1` and
`selective_max_residual_draft_rows=1`. It did not improve speed:

| variant | dense full-batch tok/s | t08 full-batch tok/s | speedup | accepted draft tokens/step |
| --- | ---: | ---: | ---: | ---: |
| riskcap2 clean | 3487.836 | 3726.061 | 1.068x | 2.014 |
| riskcap1 clean | 3480.478 | 3725.939 | 1.071x | 2.023 |
| riskcap2 stats run | 3485.076 | 3793.940 | 1.089x | 2.072 |

So the remaining `1.2x` gap is not closed by simply lowering the prefix/cap
from two rows to one row. The likely limit is now the current operator shape and
possibly acceptance/verification dynamics, not just the count of explicitly
corrected draft rows.

I then checked whether the cap1/cap2 plateau was caused by graph-safe static
bucket padding. The answer is no for the current serving path. The comparable
gate-up-only riskcap2 runs show only a tiny bucket-size effect:

| variant | dense full-batch tok/s | t08 full-batch tok/s | speedup | t08 total tok/s | accepted draft tokens/step | GPU util | graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate-up bucket16 clean | 3487.836 | 3726.061 | 1.068x | 2565.617 | 2.014 | 90.154% | `{"FULL": 49}` |
| gate-up bucket8 clean | 3485.364 | 3733.398 | 1.071x | 2577.247 | 2.022 | 90.077% | `{"FULL": 49}` |
| gate-up bucket4 clean | 3489.747 | 3678.525 | 1.054x | 2638.117 | 2.001 | 89.846% | `{"FULL": 49}` |

Bucket8 is the best measured bucket setting, but the gain over bucket16 is
only about `0.3%`. Bucket4 reduces the full-batch result and slightly lowers
accepted draft tokens/step. This means the current `1.2x` gap is not mostly
inactive bucket-tail work.

I also ran a negative-control gate-up-plus-down version with the same short
fixed-request harness. It reached only `0.982x-0.984x` dense for bucket16,
bucket8, and active-only copy, so this is not the old best riskcap2 route. The
old best route is explicitly gate-up-only:

```text
--sr24-target-leafs gate_up_proj
--sr24-residual-target-leafs gate_up_proj
--sr24-residual-layer-ids-by-leaf gate_up_proj=16-31
```

Do not compare gate-up-plus-down bucket controls directly against the
gate-up-only riskcap2 speed rows except as evidence that adding down-proj
correction currently erases the speed benefit.

## Current Bottleneck Order

1. Mixed sparse-base plus residual-correction useful-work efficiency.
2. Residual routing fraction: current residual fraction is far above the
   microbench break-even point.
3. CUDA Graph coverage: safe graph guard is too slow, but dynamic graph only
   moves `speclink_t08` from clearly bad to still below dense.
4. Bucketed dense-row correction cost: low residual budgets still plateau around
   `1.07x-1.09x`; bucket8 is only a tiny improvement over bucket16, bucket4 is
   worse, the existing Triton bucket kernel is slower on the dominant gate/up
   shape, and `compressed_dense` cache/prewarm is slower than `torch_sparse`
   residual. The current correction operators are too expensive for the `1.2x`
   target.
5. CPU mask-state synchronization: remove it by default, but it is secondary
   after graph coverage and operator work.
6. Scheduler construction and gather/scatter are not the dominant clean-serving
   bottlenecks.

## Next Experiments

Run the next iteration as a breakdown-first matrix:

1. Throughput rows: dense, `speclink_t08` safe graph guard, `speclink_t08`
   dynamic mixed graph, and `speclink_t08` dynamic graph plus adaptive dense
   fallback. Use low-sync counters only.
2. Quality gate: for every dynamic-graph candidate, run the stability gate
   against repeated dense EAGLE3 outputs.
3. Routing sweep: cap residual rows or raise confidence threshold so residual
   fraction approaches `<=0.125`, then measure both accuracy and full-batch
   tok/s. Do not spend more time on bucket16->8->4 tuning alone; the
   gate-up-only bucket sweep already showed that bucket size is not the main
   missing `1.2x` lever.
4. Operator work: the low-residual candidate is still below `1.2x`, and neither
   existing Triton bucket GEMM nor packed compressed residual Triton is viable
   for `gate_up_proj`. The next operator candidate should either keep cuBLAS for
   the small dense-row GEMM and remove only avoidable row materialization/scatter
   cost, or use a new packed residual format whose work is lower than dense
   rather than redoing a wide dense GEMM.
5. CPU-sync ablation: keep `--no-sr24-sync-mask-state` in normal runs; only
   enable exact stats in diagnostic-only runs.

The success criterion for continuing this SR24 path is not just a controller
sweep improvement. It needs a candidate that simultaneously has stable quality,
high CUDA Graph `FULL` coverage, residual fraction low enough for the operator
to win, and end-to-end speed above dense EAGLE3.

## 2026-06-29 All-MLP Prefix5 Breakdown Refresh

I added `mlpall_lowconf_prefix5_tritonoverride` expansion to
`scripts/run_sr24_slowdown_breakdown.py` so the breakdown wrapper no longer
overrides that preset with its older manual defaults. The fresh targeted run is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_breakdown_bs64_math128_20260629/
```

Primary reports:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_breakdown_bs64_math128_20260629/seven_part_report/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_breakdown_bs64_math128_20260629/component_summary/report.md
```

Clean fixed-64 rows:

| method | full-batch tok/s | total tok/s | full-batch speedup vs dense | total speedup vs dense | accepted draft tokens/step | GPU util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense EAGLE3 | 3036.752 | 2267.509 | 1.000x | 1.000x | 1.426 | 82.000% |
| `base_only_24` | 5108.797 | 2700.166 | 1.682x | 1.191x | 2.418 | 79.333% |
| all-MLP prefix5 `speclink_t08` | 3661.049 | 1953.180 | 1.206x | 0.861x | 2.382 | 82.778% |

This repeats the same pattern as the earlier 3-repeat and continuous results:
the early full-batch decode window reaches about `1.2x`, but fixed-request
total tok/s and continuous-serving steady tok/s are still below dense. Do not
treat the full-batch window alone as a final serving win.

Component timing from the diagnostic row:

| part | measured value | read |
| --- | --- | --- |
| scheduler / mask build | exact diagnostic mask `10.250ms/step`, request loop `9.968ms/step` | sync-heavy diagnostic path; useful to localize routing overhead, not the clean serving cost |
| base sparse linear | `1.216ms/call`, gate/up layers 16-31 `1.288ms/call` | dominant GPU-side component |
| residual correction | `0.135ms/call`, gate/up dense rows `0.171ms/call` | secondary per call, but still paid on many rows |
| gather/scatter | `0.046ms/event` | not the first bottleneck |
| routing statistics | draft residual/base `10953/1327`, non-draft residual/base `1535/3788`, bucket fill `0.981`, bucket active `31.378/32` | bucket is almost always full and `89.2%` of draft rows still take residual correction |
| CUDA Graph | diagnostic row `{"NONE":47}` because linear timing forces eager; previous clean preset repro had `{"FULL":49}` | use clean rows for graph judgment; instrumented rows are localization only |
| GPU util | clean `speclink_t08` `82.778%`, dense `82.000%` | fixed full-batch row is not idle-GPU limited; it is useful-work limited |

The continuous-serving evidence is stricter. With continuous refill:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_streaming_bs64_math128_20260629/
```

dense/SR24 steady tok/s is `2622.266/2288.145`, full-batch tok/s is
`2649.008/2351.428`, and SR24 GPU util is only `78.365%` versus dense
`96.210%`. SR24 accepts more draft tokens (`2.392` vs `1.398`), so the
continuous slowdown is not caused by accepted length collapse. It is a
refill/TTFT/utilization and useful-work problem.

The nonuniform dense fallback test is also negative:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlpall_lowconf_prefix5_streaming_nonuniform_dense_bs64_math128_20260629/
```

It recovers high GPU util and graph coverage, but dense/SR24 steady tok/s is
`2629.506/2125.526` and full-batch tok/s is `2649.957/2208.027`. SR24 accepted
draft tokens/step drops to `1.515`, close to dense `1.401`, so this fallback
removes the speculative-length advantage while keeping SR24's memory/operator
overhead. It is not a solution.

Current diagnosis:

1. `base_only_24` proves there is a real sparse-base upper bound, so the idea
   is not dead.
2. The mixed all-MLP path corrects too many draft rows. With `89%` draft
   residual rows and a full bucket, the sparse-base pass plus dense-row
   correction is doing too much extra work.
3. Continuous refill exposes an additional serving problem: higher TTFT/lower
   util and much smaller effective KV headroom from SR24's larger storage and
   graph footprint. Dense fallback on nonuniform steps fixes utilization but
   loses acceptance, so the fix must preserve dynamic mixed behavior rather
   than turning it into dense verification.
4. The next optimization should follow the seven-part breakdown before any
   new threshold sweep: scheduler/mask, base sparse linear, residual
   correction, gather/scatter, routing row fractions, CUDA Graph modes, and
   GPU util. The most promising route is reducing residual rows or replacing
   the two-pass sparse-base plus correction operator; bucket and scalar
   threshold tweaks alone have already plateaued.

## 2026-06-29 Clean Graph And Exact All-Corrected Follow-Up

This follow-up records the current pivot: before more controller sweeps, first
measure exactly where the slowdown is. The matrix runner now has
`--sr24-cudagraph-stats`, which writes `cudagraph_stats.jsonl` without enabling
heavy SR24 Linear timing. The slowdown wrapper forwards this switch by default
and preserves explicit `--sr24-residual-bucket-size` /
`--no-sr24-residual-bucket-priority` overrides when expanding presets. This is
important because `mlpall_lowconf_prefix5_tritonoverride` defaults to bucket32;
using it blindly for `all_corrected_24` can profile a bucketed approximation
instead of exact no-bucket correction.

Clean graph smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_clean_cudagraph_stats_smoke_bs64_math64_20260629/
```

| method | total tok/s | full-batch tok/s | accepted draft tokens/step | GPU util | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | 1787.341 | 2631.188 | 1.148 | 76.2% | `{"FULL":6,"NONE":25,"PIECEWISE":1}` |
| `base_only_24` | 1916.811 | 4018.758 | 1.799 | 75.6% | `{"FULL":30,"NONE":2}` |
| `speclink_t08` | 1629.809 | 3342.294 | 1.795 | 79.4% | `{"FULL":30,"NONE":2}` |

Read: in this short fixed-request smoke, `base_only_24` is not slow because of
accepted length or graph misses. It has better accepted length than dense and
mostly uses CUDA Graph. The remaining `speclink_t08` issue is useful work in
the mixed operator and fixed-request/tail behavior, not just graph coverage.

Exact no-bucket all-corrected clean smoke:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_exact_nobucket_clean_graph_smoke_bs64_math64_20260629/
```

| method | total tok/s | full-batch tok/s | accepted draft tokens/step | GPU util | residual bucket | CUDA Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 1786.713 | 2628.996 | 1.148 | 70.6% | n/a | `{"FULL":6,"NONE":25,"PIECEWISE":1}` |
| `all_corrected_24` exact/no-bucket | 826.866 | 1250.198 | 1.129 | 60.7% | 0 | `{"NONE":32}` |

Read: exact `dense_rows` all-corrected is slow even though accepted length is
basically unchanged. The main reason is not draft quality; it is the eager,
two-pass operator path: every corrected Linear computes sparse base plus a
second dense residual/full-dense path, and the current graph guard leaves every
decode step in CUDA Graph `NONE`.

Exact no-bucket component profile:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_exact_nobucket_direct_component_profile_bs64_math64_20260629/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_allcorrected_exact_nobucket_direct_component_profile_bs64_math64_20260629_work/all_corrected_24/bs64/rep1/speclink_sr24_breakdown.json
```

This profile is diagnostic only because it enables Linear CUDA event timing and
exact routing counters. Component timing:

| component | total ms | calls | avg ms/call | ms/decode step |
| --- | ---: | ---: | ---: | ---: |
| scheduler mask build CPU | 1.584 | 43 | 0.037 | 0.037 |
| scheduler materialize counts CPU | 0.343 | 43 | 0.008 | 0.008 |
| base sparse Linear CUDA | 3056.148 | 2944 | 1.038 | 71.073 |
| residual dense full GEMM CUDA | 1980.223 | 2944 | 0.673 | 46.052 |
| residual full select CUDA | 96.215 | 2944 | 0.033 | 2.238 |

Read: scheduler/mask work is negligible in this exact all-corrected profile.
The slowdown is dominated by GPU-side Linear work: sparse base plus a second
dense residual/full-dense pass. There is no gather/scatter cost in the
no-bucket all-residual path because it falls through to full dense GEMM plus
`torch.where` selection. This is currently the clearest answer to "why slow":
exact all-corrected pays more useful work than dense and loses CUDA Graph.

Updated optimization direction:

- keep using the seven-part breakdown before new speed claims;
- do not treat bucketed all-corrected as exact all-corrected;
- for exact all-corrected, either use dense-fastpath as the correctness/no-op
  control or build a fused/packed sparse+residual kernel;
- for `speclink_t08`, graph coverage is necessary, but the next gain must come
  from reducing corrected rows without quality loss or fusing the mixed
  operator, not from another plain scheduler/mask optimization.

## 2026-06-29 User-Pivot Breakdown Refresh

The current user-pivot summary is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWNESS_BREAKDOWN_CURRENT.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_user_pivot_breakdown_20260629/report.md
```

This report joins the explicit seven-part breakdown, exact no-bucket
`all_corrected_24`, and the early-dense/default-vLLM-compile control. The key
read is now:

- `base_only_24` is not the immediate slowdown. It has mostly `FULL` CUDA Graph
  coverage and better accepted draft length than dense in the explicit bs64
  run.
- guarded `speclink_t08` is slow because it combines CUDA Graph `NONE` coverage
  with too many residual-corrected rows. Diagnostic timing points to sparse
  base plus residual correction, not gather/scatter or normal mask build.
- exact no-bucket `all_corrected_24` is slow even with nearly unchanged accepted
  draft length. It pays sparse base plus a second dense residual/full path and
  falls out of CUDA Graph.
- early-dense/default-compile `all_corrected_24` is near dense-equivalent:
  same-root full-batch tok/s is `2626.557` versus dense `2840.707`. This
  confirms that SR24 bookkeeping is not the root problem; the real
  sparse+residual operator path is.

The next optimization should therefore be evaluated by the seven fields:
scheduler/mask build, base sparse Linear, residual correction, gather/scatter,
routing row fractions, CUDA Graph modes, and GPU utilization. A candidate is
not useful unless it reduces residual rows or replaces the mixed operator while
preserving clean CUDA Graph coverage and quality.
