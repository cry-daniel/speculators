# SR24 System Optimization Status - 2026-06-30

## Current Diagnosis

The current SR24 slowdown is not mainly acceptance collapse. In the clean
Llama-3.1 math_reasoning matrix, SR24 accepts about the same or more draft
tokens per step than dense EAGLE3, but bs8/16/32 are still slower. The remaining
problem is useful-work efficiency: current mixed sparse/dense execution keeps
the GPU busy, but much of the work is fragmented sparse-base MLP or
row-routing overhead.

The fixed-prefix contiguous route-table patch removed a real scheduler-side
bottleneck. On the bs64 diagnostic run, route CPU time dropped from about
72 ms/step to 0.81 ms/step and the expensive Triton complement path almost
disappeared. Clean serving still does not reach the target at bs8/16/32, so the
next bottleneck is the operator/data-layout path.

## Experiments From This Pass

Graph-on base-only upper-bound check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_compile_ab_bs32_64_math256_20260630
```

The earlier slow `base_only_24` runs were mostly a graph/eager artifact. With
`--sr24-base-only-allow-compile --sr24-allow-cudagraph`, full-MLP base-only
uses mostly `FULL` CUDA-Graph steps and becomes a real speed upper bound:

| bs | dense total | base-only total | total speedup | dense full | base-only full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2274.533 | 2654.179 | 1.167x | 2645.353 | 3403.346 | 1.287x |
| 64 | 2317.306 | 3829.167 | 1.652x | 3418.704 | 6368.113 | 1.863x |

However, this full-MLP upper bound is not quality-safe:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_mlpall_gsm8k50_accuracy_20260630
```

GSM8K-50 exact match drops from `0.7200` to `0.2200` (`-50 pp`), so broad
base-only all-MLP sparse execution is useful only as a performance ceiling.

Quality-bound base-only scope check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup16_31_baseonly_compile_bs8_16_32_64_math256_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_scope_gate_quality50_speed64_20260630_continue2
```

`gate_up_proj=16-31` is the current pure base-only scope closest to the requested
8pp quality budget: the existing GSM8K-50 scope gate reports `0.7200 -> 0.6400`
(`-8 pp`). With graph-on throughput, it only helps at high batch:

| bs | dense total | base-only total | total speedup | dense full | base-only full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1117.994 | 1072.540 | 0.959x | 1203.710 | 1178.260 | 0.979x |
| 16 | 1727.006 | 1644.908 | 0.952x | 1973.916 | 1918.163 | 0.972x |
| 32 | 2270.104 | 2295.024 | 1.011x | 2639.991 | 2843.749 | 1.077x |
| 64 | 2315.768 | 2792.660 | 1.206x | 3416.253 | 3964.942 | 1.161x |

This is the strongest current evidence for the design constraint: quality-safe
2:4 scopes can reach the target around bs64 total throughput, but not at bs8/16
and not reliably at bs32 without a better operator/data-layout path.

All-corrected residual backend check:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_gateup16_31_denserows_bs64_math256_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_gateup16_31_torchsparse_bs64_math256_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_gateup16_31_compresseddense_bs64_math256_20260630
```

All three runs use the same exact all-corrected scope:
`gate_up_proj=16-31`, Llama-3.1, `math_reasoning`, bs64, K=8,
max tokens 256. They differ only in residual backend.

| residual backend | total tok/s | dense total | total speedup | full tok/s | dense full | full speedup | GPU util | graph | storage/dense |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `dense_rows` | 1709.850 | 2319.031 | 0.737x | 2878.277 | 3423.919 | 0.841x | 83.316% | `NONE` | 1.625 |
| `torch_sparse` | 2318.212 | 2317.835 | 1.000x | 3117.807 | 3424.287 | 0.910x | 91.000% | `FULL` | 1.1875 |
| `compressed_dense` | 1937.187 | 2316.086 | 0.836x | 2982.031 | 3419.160 | 0.872x | 93.529% | `FULL` | 1.125 |

This addresses the compressed-dense question directly: the current
`compressed_dense` run is GPU-resident (`residual_device_counts={"cuda:0": 16}`,
cached, prewarmed, `residual_out_chunk=0`) and still slower than
`torch_sparse` residual. The bottleneck is not CPU transfer. The best exact
all-corrected path today is `torch_sparse` residual with CUDA Graph, but it is
only about dense-equivalent in total throughput and below dense in full-batch
throughput. Therefore exact correction is not enough for the requested 1.2x;
the selective path has to reduce exact-corrected work and/or use a fused/grouped
operator.

No-verify dense MLP fastpath A/B:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_fastpath_off_matched_b8_16_math512_20260630
```

Disabling `SPECLINK_SR24_NOVERIFY_DENSE_MLP_FASTPATH` is not a speed fix.
Matched bs8 total tok/s is `1226.344` versus the earlier fastpath-on
`1238.717`; bs16 is `1960.913` versus `1943.690`, effectively noise. Keep the
fastpath enabled by default.

Lossy smallrow/layer-scope candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_system_gateup_smallrow160_quality50_speed_b8_16_32_64_20260630
```

`gateup_res16_25_base26_31_critical4_smallrow160` passes GSM8K-50 exactly at
the current 8pp quality budget: dense `0.7600`, SR24 `0.6800`, delta `-8.0 pp`.
Throughput is still not good enough:

| bs | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1382.816 | 1088.004 | 0.787x | 1496.997 | 1223.400 | 0.817x |
| 16 | 2091.614 | 1685.990 | 0.806x | 2521.137 | 2078.742 | 0.825x |
| 32 | 2655.585 | 2276.807 | 0.857x | 3438.747 | 3006.291 | 0.874x |
| 64 | 2457.225 | 2882.766 | 1.173x | 4326.795 | 4106.539 | 0.949x |

Conclusion: shrinking residual scope and falling back on tiny sparse groups can
help high-batch total throughput, but it does not solve low/medium batch or
full-batch throughput.

Follow-up fixed-prefix row-routed MLP and overlap checks:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_prefix2_rowrouted_mlp_gate_q50_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_prefix2_rowrouted_mlp_throughput_bs8_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_smallrow160_gate_q50_speed_bs8_64_20260630
```

`lossy_prefix2_rowrouted_mlp` is the current implementation closest to the
desired systems design: fixed prefix2 plus bonus rows are dense, all other rows
are sparse-only, and small dense-important groups are padded to improve
occupancy. It passes GSM8K-50 with dense `0.7400`, SR24 `0.7200`
(`-2 pp`), but the current PyTorch/cuSPARSELt implementation is only useful at
large batch:

| bs | total speedup | full-batch speedup | CUDA graph |
|---:|---:|---:|---|
| 8 | 0.491x | 0.475x | `FULL=8` |
| 16 | 0.519x | 0.496x | `FULL=16` |
| 32 | 0.664x | 0.660x | `FULL=31` |
| 64 | 0.964x | 1.089x | `FULL=49` |

The explicit CUDA-stream overlap ablation with the same candidate also passes
the GSM8K-50 quality gate (`-2 pp`) but collapses throughput at bs64:
`0.331x` total and `0.432x` full-batch. In the current code this is not a
viable path; auxiliary streams increase fragmentation and lose the practical
benefit of the graph-stable route-table path.

The repeated `gateup_res16_25_base26_31_critical4_smallrow160` run also passes
the GSM8K-50 gate with dense `0.7400`, SR24 `0.7200` (`-2 pp`), but remains
below dense throughput at every tested batch:

| bs | total speedup | full-batch speedup | CUDA graph |
|---:|---:|---:|---|
| 8 | 0.707x | 0.702x | `FULL=8` |
| 16 | 0.744x | 0.768x | `FULL=16` |
| 32 | 0.807x | 0.821x | `FULL=31` |
| 64 | 0.861x | 0.887x | `FULL=49` |

Interpretation: relaxing quality to an 8pp budget helps find acceptable
controllers, but strategy-level routing is no longer the main limiter. The
current kernels pay too much for fragmented sparse MLP, row gathering, and
row assembly. Achieving the 1.2x target across bs8/16/32/64 requires changing
the data layout and operator, not only changing which rows are dense.

Follow-up compact-verifier guard and row-count fallback:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noncompact_dense_breakdown_bs8_math64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noncompact_dense_minbase256_speed_bs8_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noncompact_dense_minbase256_quality_q50_20260630
```

The fixed-prefix row-routed MLP path assumed every scheduled row belonged to a
compact speculative verifier block. Breakdown showed that non-uniform steps
such as `sched=9-105` could still enter SR24 routing, so long prompt/prefill
chunks were treated as sparse-base rows. The worker now passes decode-only
draft counts into SR24, and SR24 selective mode falls back to `all_residual`
for non-compact spec batches. This is a correctness and data-layout guard:
fixed-prefix routing is now reserved for uniform `K+1` verifier blocks.

The breakdown check confirms the guard is active:
`scheduler_selective_noncompact_spec_all_residual=8`, and the large
`row_routed_mlp_fixed_block_*` sparse timings disappear from the non-compact
steps. Throughput improves most at bs32 but still does not reach the target:

| bs | old minbase256 total | new total | old full | new full |
|---:|---:|---:|---:|---:|
| 8 | 0.847x | 0.845x | 0.858x | 0.850x |
| 16 | 0.840x | 0.870x | 0.863x | 0.886x |
| 32 | 0.817x | 0.903x | 0.817x | 0.956x |
| 64 | 0.984x | 0.985x | 1.037x | 1.040x |

The quality gate remains inside budget: GSM8K-50 dense `0.7400`, SR24
`0.7200`, delta `-2 pp`.

This guard should remain in the code even if later kernels are replaced,
because it prevents sparse routing from leaking into prompt/prefill rows. It
does not solve the main performance target by itself. The remaining gap is
operator efficiency: the profitable sparse work only appears in compact
verifier blocks, while low/medium batch sizes still spend too much time in
graph overhead, dense fallback, and unfused MLP dispatch.

Follow-up controller and fill-factor checks:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_minbase256_quality_q50_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_minbase256_speed_bs8_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_minbase256_breakdown_bs32_math_corrected_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_minbase32_64_128_speed_bs8_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_minbase128_quality_q50_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_minbase128_speed_bs8_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_k12_minbase256_speed_bs32_64_20260630
```

The direct "do not recompute dense after sparse" A/B is not enough. Disabling
the no-verify dense MLP fastpath preserves the GSM8K-50 quality gate
(`0.7400 -> 0.7200`, `-2 pp`), but throughput stays essentially the same as
the fastpath-on minbase256 route:

| bs | minbase256 total | noverify-off total | minbase256 full | noverify-off full |
|---:|---:|---:|---:|---:|
| 8 | 0.845x | 0.848x | 0.850x | 0.855x |
| 16 | 0.870x | 0.874x | 0.886x | 0.891x |
| 32 | 0.903x | 0.914x | 0.956x | 0.953x |
| 64 | 0.985x | 0.960x | 1.040x | 1.009x |

The corrected bs32 breakdown explains why minbase256 is close to dense but not
faster. With K=8 and prefix2+bonus, bs32 has only about `32*(8-2)=192` sparse
base rows per compact verifier step. The 256-row guard therefore falls back to
full dense for most row-routed MLP calls:
`row_routed_mlp_full_dense_fallback_calls=1248`. Clean serving has good CUDA
Graph coverage (`FULL=62`, `NONE=2`) but SR24 full-batch throughput is still
only `0.928x` of dense and GPU util is lower (`73.0%` vs `79.2%`). The
diagnostic row shows the exact-stats path spends about `18 ms/step` in routing,
but that is a diagnostic synchronization artifact; the clean row points to
operator useful-work efficiency.

Lowering the sparse-base row threshold makes more batches enter the sparse
branch, but current PyTorch semi-structured sparse MLP is too inefficient at
those shapes:

| candidate | bs8 total/full | bs16 total/full | bs32 total/full | bs64 total/full |
|---|---:|---:|---:|---:|
| minbase32 | 0.631/0.616x | 0.720/0.720x | 0.828/0.873x | 0.926/0.999x |
| minbase64 | 0.848/0.853x | 0.709/0.722x | 0.822/0.871x | 0.930/1.002x |
| minbase128 | 0.852/0.850x | 0.876/0.888x | 0.853/0.868x | 0.970/1.030x |
| minbase256 | 0.845/0.850x | 0.870/0.886x | 0.903/0.956x | 0.985/1.040x |

Reducing the protected dense prefix from two draft rows to one draft row also
does not help. `prefix1+bonus` passes the same GSM8K-50 gate (`-2 pp`) but is
slower than prefix2 at bs32/64:

| bs | prefix1 total/full | prefix2 minbase128 total/full |
|---:|---:|---:|
| 8 | 0.881/0.864x | 0.852/0.850x |
| 16 | 0.878/0.897x | 0.876/0.888x |
| 32 | 0.826/0.855x | 0.853/0.868x |
| 64 | 0.906/0.976x | 0.970/1.030x |

Increasing K from 8 to 12 is also not the answer. It fills more sparse rows,
but it slows both dense EAGLE3 and SR24, and SR24 remains below dense at bs32:

| K | bs | total speedup | full-batch speedup |
|---:|---:|---:|---:|
| 12 | 32 | 0.899x | 0.929x |
| 12 | 64 | 0.845x | 1.039x |

Turning on the existing Triton route assembly is not enough either:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_minbase256_triton_assemble_speed_bs32_64_20260630
```

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 32 | 0.909x | 0.943x |
| 64 | 0.963x | 1.013x |

The independent row-routed MLP microbench gives the operator-level reason:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_microbench_rows288_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_microbench_rows576_20260630
```

Rows=288 approximates bs32, K=8 verifier blocks; rows=576 approximates bs64.
The table reports graph replay ms. Exact-down is the quality-conservative route
where important rows are dense through both gate/up and down.

| rows | bucket | dense graph | exact-down Triton | exact-down no final assemble | Triton assemble only |
|---:|---:|---:|---:|---:|---:|
| 288 | 96 | 0.5320 | 0.6542 | 0.6420 | 0.0145 |
| 288 | 192 | 0.5322 | 0.6447 | 0.6386 | 0.0113 |
| 576 | 96 | 0.9577 | 0.8661 | 0.8421 | 0.0249 |
| 576 | 192 | 0.9588 | 0.8025 | 0.7812 | 0.0228 |

For rows=288, even removing final assembly entirely would still be slower than
dense. For rows=576, exact-down can beat dense, and the no-final-assemble
variant reaches about the 1.2x target for bucket 192 (`0.7812 ms` versus dense
`0.9588 ms`). Final assembly costs only about `0.01-0.03 ms`; the larger cost
is the split sparse/dense MLP itself, especially sparse gate/up and sparse down.
Therefore an assembly-only Triton optimization cannot solve bs8/16/32. The
operator path needs a packed verifier-block data format and a fused or
graph-stable grouped MLP route that reduces dispatch and sparse GEMM overhead.

Current conclusion: simple controller changes have been exhausted. The only
measured path that beats dense meaningfully is broad base-only sparse MLP, but
it fails quality. Quality-safe lossy controllers can keep accuracy within the
8pp budget, but they cannot reach 1.2x without a better data format and fused
operator.

## Latest Scope Checks

The user-requested direction was to stop requiring lossless behavior, allow up
to 8 percentage points of accuracy loss, and avoid recomputing dense work for
unimportant tokens after sparse work has already run. The current code already
has the latter property in the row-routed MLP path when
`SPECLINK_SR24_ROUTE_REUSE_BASE_OUTPUT=0`: dense-important rows and sparse-base
rows are disjoint, then the outputs are assembled. The `reuse_base_output` path
does run a full sparse base MLP and overwrite dense rows afterward, so it should
stay disabled for this optimization target.

One experiment-framework issue was fixed before interpreting the newest scope
checks. The SR24 preset application previously reset explicit layer-scope
arguments such as `--sr24-residual-layer-ids-by-leaf` and
`--sr24-base-only-layer-ids-by-leaf`, so a command could look scoped in
`commands.sh` while the actual server environment still used the preset's
default all-MLP scope. The override allow-list now includes:

```text
--sr24-target-leafs
--sr24-residual-target-leafs
--sr24-base-only-layer-ids
--sr24-base-only-layer-ids-by-leaf
--sr24-residual-layer-ids-by-leaf
```

The fixed runs are under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix2_directcslt_limited_layers_quality50_bs32_64_20260630_scopefix
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix2_directcslt_front_residual_tail_base_quality50_bs32_64_20260630
```

Residual-scope-only checks do not make other layers sparse-only; unlisted MLP
layers stay on the normal dense vLLM path. These pass quality but do not remove
enough work:

| candidate | GSM8K-50 dense | GSM8K-50 SR24 | delta | bs32 total/full | bs64 total/full | storage/dense |
|---|---:|---:|---:|---:|---:|---:|
| residual MLP 16-31 only | 0.72 | 0.68 | -4 pp | 0.520/0.584x | 0.676/0.839x | 1.625 |
| residual MLP 24-31 only | 0.72 | 0.72 | 0 pp | 0.769/0.827x | 1.043/0.947x | 1.625 |

The true sparse-only tail checks explicitly attach front layers with dense
residual correction and mark tail layers as base-only sparse. These reduce
storage, but the layer granularity is too coarse for the 8pp quality budget:

| candidate | GSM8K-50 dense | GSM8K-50 SR24 | delta | storage/dense | result |
|---|---:|---:|---:|---:|---|
| residual MLP 0-15, base-only MLP 16-31 | 0.72 | 0.50 | -22 pp | 1.125 | failed gate |
| residual MLP 0-23, base-only MLP 24-31 | 0.72 | 0.60 | -12 pp | 1.375 | failed gate |

This rules out simple layer-level "tail sparse-only" as the main path. The
quality-safe point is token-level or channel-level selectivity inside layers:
important verifier rows need dense MLP, while unimportant rows should use 2:4
only. Current row-routed MLP expresses that policy, but its PyTorch/cuSPARSELt
implementation is too fragmented at bs8/16/32. Therefore the remaining work is
systems work: a graph-stable packed verifier-block format and fused/grouped MLP
operators, not more Python-side controller tuning.

## Latest Lossy Gate And Operator Breakdown

Focused note:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_LOSSY_PPOPP_OPTIMIZATION_PLAN_20260630.md
```

Newest small-scale gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_ppopp_gate_bs8_64_math128_gsm8k50_20260630
```

Setup: Llama-3.1-8B EAGLE3, K=8, GSM8K limit 50 for quality, 8 percentage-point
absolute quality budget, `math_reasoning` bs `8/16/32/64` for throughput.

All checked candidates pass the quality gate at `-2 pp`, but none reaches the
requested `1.2x` target. The best current candidate,
`lossy_prefix2_rowrouted_mlp_minbase64`, reaches only:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 0.859x | 0.855x |
| 16 | 0.721x | 0.719x |
| 32 | 0.839x | 0.879x |
| 64 | 0.966x | 1.006x |

The follow-up breakdowns are:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_rowrouted_breakdown_bs64_math128_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_rowrouted_breakdown_bs64_inst64_20260630
```

Clean bs64 serving shows the same pattern:

| method | total tok/s | full-batch tok/s | GPU util | read |
|---|---:|---:|---:|---|
| dense EAGLE3 | 2242.7 | 3039.2 | 78.4% | baseline |
| base-only 2:4 | 2737.5 | 5412.5 | 82.0% | sparse upper bound is real |
| row-routed SR24 | 1739.6 | 2812.3 | 83.7% | policy correct, operator slower |

The full-batch instrumented run confirms the intended routing policy is active:
draft residual/base rows are `4026/12078`, non-draft residual rows are `2013`,
so only one quarter of draft rows use dense-important MLP. The expensive part is
not dense-important work; row-routed fixed-block timing is dominated by the
sparse branch:

| component | avg ms/call |
|---|---:|
| sparse/base gate-up | 0.950 |
| sparse/base down | 0.975 |
| dense-important gate-up | 0.196 |
| dense-important down | 0.115 |

The isolated microbench is consistent: sparse-only Linear is faster than dense,
but current mixed sparse-base plus correction/route machinery erases the gain.
For gate/up at residual fraction `0.125`, dense graph time is `0.539 ms`,
base-sparse is `0.354 ms`, but the mixed proxy is already `0.555 ms` and the
routed split is `0.789 ms`.

Conclusion: the current row-routed semantics are correct for the user-requested
lossy policy, but the PyTorch/cuSPARSELt operator path is not fast enough. The
next useful work is a fixed-capacity packed verifier-block data format and a
fused or grouped mixed MLP kernel. More lossless gates, scalar threshold sweeps,
or Python-side active-row selection are unlikely to reach the target.

## Next Implementation Direction

The next path should be a fixed-capacity grouped route-table operator plus a
row-count-aware controller:

1. Build graph-stable row metadata: per step, keep fixed-size dense-important
   and sparse-base row buckets with request ids and active masks.
2. Make dense and sparse branches disjoint: rows selected for dense should not
   also pay sparse-base MLP work.
3. Avoid dynamic compaction in Python: active-only `nonzero/index_select` was
   already much slower. Active masks should be consumed inside the kernel or a
   graph-stable grouped plan.
4. Replace the current MLP-level Python route with a packed format:
   `row_kind[B, K+1]`, `request_row[B, K+1]`, fixed dense-prefix and sparse-tail
   views, and per-layer reusable route descriptors. The verifier block is
   already logically `[batch, K+1, hidden]`; preserving that format avoids
   repeated flat `index_select/index_copy` assembly.
5. Fuse the common compact-verifier case: dense important rows and 2:4 sparse
   base rows should be launched from one graph-stable operator wrapper, with
   active masks handled on GPU. If two GEMMs are still required, use a single
   persistent route descriptor and overlap only after row counts are large
   enough; the previous PyTorch auxiliary-stream overlap was negative.
6. Handle too-few sparse rows by stable tile filling or dense fallback based on
   measured fill thresholds. The current data show 32/64/128-row sparse
   branches are net negative, while bs64-sized groups can break even in
   full-batch windows.
7. Keep quality gates lossy but bounded: use GSM8K limit 50+ and an absolute
   `8 pp` budget for quick screening before running full throughput matrices.

For quality screening, continue using GSM8K limit 50 or larger with an
8 percentage-point absolute loss budget before spending time on full
throughput matrices.
