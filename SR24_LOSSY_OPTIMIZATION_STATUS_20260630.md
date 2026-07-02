# SR24 Lossy Optimization Status, 2026-06-30

## Current Goal

The current target is a lossy SpecLink/SR24 path that stays within about
`8 pp` absolute GSM8K accuracy loss and reaches at least `1.2x` dense EAGLE3
serving throughput on most batch sizes `8/16/32/64` and datasets.  The current
implementation has not reached that target yet.

## New Results

All outputs below are in `results.bak/` or `temp/`, not `results/`.

### Low-storage gate_up split

Result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_res16_25_base26_31_critical4_b8_16_32_64_math2048_20260630/
```

This candidate uses `gate_up_proj` only, residual-corrects layers `16-25`,
leaves layers `26-31` sparse-only, and uses `critical_prefix+extra4`.
GSM8K-50 quality passed in:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_res16_25_base26_31_critical4_gsm8k50_20260630/
```

GSM8K-50 dense and SR24 were both `0.72`.

Throughput on `math_reasoning`, K=8, max tokens 2048:

| bs | total speedup | full-batch speedup | read |
|---:|---:|---:|---|
| 8 | 0.866x | 0.880x | slow |
| 16 | 0.811x | 0.849x | slow |
| 32 | 0.753x | 0.896x | slow |
| 64 | 1.345x | 0.963x | total win, full-batch still slow |

This is the best long-output total-throughput point so far, but it does not
solve the stable full-batch operator problem and fails low/mid batch.

### Small-row no-residual dense fallback

Code added:

- `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_NO_RESIDUAL_ONLY`
- `SPECLINK_SR24_ADAPTIVE_DENSE_FALLBACK_SMALL_GATE_UP_NO_RESIDUAL`
- preset alias `gateup_res16_25_base26_31_critical4_smallrow160`

Result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_smallrow160_b8_16_32_math2048_20260630_032223/
```

The first run did not actually trigger fallback because sparse-only
no-residual layers do not retain dense weights under the low-storage preset.
The sanity run confirmed `adaptive_dense_fallback_calls=0`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_smallrow160_stats_sanity_20260630_032223/
```

Keeping dense weights for no-residual layers was tested here:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_smallrow160_keepdense_b8_16_32_math2048_20260630_032223/
```

| bs | total speedup | full-batch speedup | storage/dense | read |
|---:|---:|---:|---:|---|
| 8 | 0.905x | 0.919x | 1.625x | still slow |
| 16 | 0.999x | 0.873x | 1.625x | total only catches up |
| 32 | 0.712x | 0.891x | 1.625x | slow |

Conclusion: low-batch slowdown is not mainly caused by no-residual gate_up
layers lacking dense fallback.

### Active-fused important-token correction

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_mlpall_prefix2_gsm8k50_quality_20260630_032223/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_mlpall_prefix2_bs64_math2048_20260630_032223/
```

GSM8K-50 quality passes the 8pp budget: dense `0.72`, SR24 `0.66`, delta
`-6 pp`.

On bs64, K=8, max tokens 2048:

| metric | dense | SR24 | speedup |
|---|---:|---:|---:|
| total output tok/s | 2394.619 | 2513.040 | 1.049x |
| full-batch output tok/s | 5208.435 | 4630.913 | 0.889x |

The short-output full-batch win does not carry to long-output full-batch.

### Tile-fill bucket32 cuBLAS correction

Code preset added in both throughput and lm-eval runners:

```bash
--sr24-preset mlpall_tilefill_prefix2_bucket32_cublas
```

This is the current data-format probe for the "important rows are too few"
problem. It uses all MLP leaves, direct cuSPARSELt sparse base, low-confidence
prefix2 correction, fixed bucket32 dense overwrite, and cuBLAS for the dense
bucket. It deliberately avoids active-only compaction and the current Triton
dense-GEMM/scatter path.

GSM8K-50 quality passed with no aggregate loss:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix2_gsm8k50_quality_20260630/
```

Dense and SR24 were both `0.72` with paired regressions/improvements `1/1`.

Medium-output throughput is close to the target:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix2_bs64_math512_20260630/
```

| metric | dense | SR24 | speedup |
|---|---:|---:|---:|
| total output tok/s | 2462.304 | 2812.765 | 1.142x |
| full-batch output tok/s | 4379.987 | 5213.951 | 1.190x |
| accepted draft tokens/step | 2.356 | 3.060 | higher |

Long-output throughput still fails the robust target:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix2_bs64_math2048_20260630/
```

| metric | dense | SR24 | speedup |
|---|---:|---:|---:|
| total output tok/s | 2428.125 | 2437.221 | 1.004x |
| full-batch output tok/s | 5388.389 | 4565.464 | 0.847x |
| accepted draft tokens/step | 3.979 | 4.269 | higher |

Interpretation: fixed tile fill is better than active-only tiny GEMMs and is
the right data-layout direction, but it is still not a max2048 serving solution.
The remaining problem is full-batch operator/KV pressure, not GSM8K quality or
accepted-length collapse.

Follow-up diagnosis:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_breakdown_bs64_math128_20260630/
```

Clean bs64/max128 serving shows the same path is graph/util healthy but not a
large enough win: dense full-batch `3029.099`, base-only `5092.301`, and SR24
`3412.998` tok/s (`1.127x`). The diagnostic row localizes the mixed-path cost:
base sparse is `1.173 ms/call`, residual dense correction is only
`0.139 ms/call`, gather/scatter is `0.014 ms/call`, and bucket fill is high
(`31.162/32`). Draft residual fraction is still high (`0.816`). This says the
current problem is not empty correction tiles; it is that most useful rows still
pay the sparse-base pass, and the sparse-base pass dominates.

Two lossy follow-ups did not improve throughput:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix1_gsm8k50_quality_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix1_bs64_math512_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix2_thr04_gsm8k50_quality_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_bucket32_cublas_prefix2_thr04_bs64_math512_20260630/
```

Both passed GSM8K-50 with dense `0.72` and SR24 `0.72`, but both were slower on
bs64/math/max512:

| variant | total speedup | full-batch speedup | read |
|---|---:|---:|---|
| prefix1 threshold0.6 | 0.921x | 0.904x | fewer forced rows hurt acceptance/shape enough to lose |
| prefix2 threshold0.4 | 0.848x | 0.903x | lower threshold also loses |

Do not continue with scalar prefix/threshold reductions as the main path. They
can pass the 8pp quality gate, but they do not reduce the dominant sparse-base
cost in a useful way.

### Row-routed overlap and fixed-prefix routing probes

The explicit two-stream row-routed MLP overlap path was tested as a negative
ablation.  It runs dense rows and sparse rows on separate CUDA streams, but it
is forced out of CUDA Graph mixed replay and does not hide the dominant sparse
branch cost:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_mlp_overlap_eager_breakdown_bs64_math64_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowrouted_mlp_nooverlap_eager_breakdown_bs64_math64_20260630/
```

Both eager runs were about `235` output tok/s.  The overlap counters fired
(`row_routed_mlp_overlap_stream_calls=1280`), but the sparse base branch still
dominated: overlap base `gate_up` and `down` took about `1533 ms` and
`1405 ms` cumulatively, while the dense branch was only about `204 ms` and
`110 ms`.  Do not use Python/Torch stream splitting as the main optimization.

A graph-friendly fixed-prefix route-all probe was added:

```bash
--sr24-preset fixedprefix4_all_rowrouted_graph
```

It protects a fixed prefix plus all non-draft rows, avoids
`direct_cpu_route_rows`, and uses MLP-level row routing so exact dense rows do
not also pay sparse-base work.  The runner guard was relaxed only for this
fixed-prefix/non-draft-all/no-direct-cpu route-all case so vLLM does not force
eager.  The graph path does work, but it is still not fast enough:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedprefix4_all_rowrouted_graph_noforceeager_bs64_math512_20260630/
```

On bs64/math/max512 it achieved `2562.341` total tok/s and `3982.950`
full-batch tok/s versus dense `2818.885` and `4216.231` (`0.909x` total,
`0.945x` full-batch).  CUDA Graph coverage was healthy
(`profile:{"FULL":49}`) and GPU util was `95.45%`, so the remaining loss is
operator/data-layout cost rather than scheduler idleness.

### Active-only bucket correction

The user-requested idea "if an unimportant token already ran sparse, do not run
dense for it too" was tested with the existing active-only bucket knobs:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_prefix2_activeonly_bs64_math512_20260630/
```

This is a strong negative result: dense was `2819.236` total / `4215.199`
full-batch tok/s, while active-only SR24 was `1178.266` / `1911.627`
(`0.418x` total, `0.454x` full-batch).  The run still had CUDA Graph coverage
(`profile:{"FULL":49}`), but GPU util dropped to `42.51%`.  Dynamic active-row
compaction creates small/underfilled work and loses the stable tile benefit.
Future work should keep fixed-capacity route tables and use an active mask
inside a fused/grouped kernel, not Python/Torch active-row compression.

The fixed-shape active-mask fused variant restores CUDA Graph/GPU utilization
but still does not make the correction kernel proportional to the active rows:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_prefix2_active_mask_fused_bs64_math512_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_bucket_param_sweep_gateup_down_bs64_20260630/
```

On bs64/math/max512, active-mask fused SR24 reached `2580.280` total and
`3810.455` full-batch tok/s versus dense `2819.112` and `4217.142`
(`0.915x` total, `0.904x` full-batch), with GPU util `93.10%` and CUDA Graph
`profile:{"FULL":49}`.  The microbench shows why: current Triton bucket
dense-GEMM/scatter is about `0.76 ms` for Llama `gate_up_proj` and `0.29 ms`
for `down_proj` at bucket32, almost independent of residual fraction
`0.0625` vs `0.125`, and often slower than the dense or bucket-delta
microbench baselines.  It fixes execution shape, not useful-work count.

The current fixed tile-fill control also no longer reproduces the older
positive max512 result under the latest code path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_tilefill_prefix2_current_bs64_math512_20260630/
```

It reached `2600.489` total and `3744.343` full-batch tok/s versus dense
`2820.426` and `4218.417` (`0.922x` total, `0.888x` full-batch).  Treat the
older `1.142x/1.190x` max512 tile-fill result as stale unless reproduced in a
fresh same-condition run.

### Coarse base-only lossy bound

A deliberately lossy static candidate was tested to see whether the allowed
`8 pp` accuracy budget could buy enough speed by removing token-level
correction entirely:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_tail16_baseonly_speed_bs64_math512_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_tail16_baseonly_gsm8k50_quality_20260630/
```

The candidate applies `base_only_24` to `gate_up_proj=16-31` and
`down_proj=16-31`.  It is fast on bs64/math/max512: total tok/s `4990.827`
versus dense `3609.462` (`1.383x`), and full-batch tok/s `5158.599` versus
`3750.903` (`1.375x`).  However, GSM8K-50 exact match collapses from `0.72` to
`0.22` (`-50 pp`), with paired regressions/improvements `28/3`, average output
tokens rising from `98.16` to `160.98`, and clipped completions rising from
`1` to `10`.  This is a useful upper-bound negative result: coarse late-layer
MLP base-only can meet the throughput target but destroys reasoning quality
far beyond the allowed-loss budget.

### Base-only scope quality/speed ceiling

The scope sweep helper was fixed so `--output-root` and `--work-root` are
resolved to absolute paths before child `lm-eval` and GuideLLM commands run.
Without this, child commands launched from `eval-guidellm/` could write nested
`examples/evaluate/eval-guidellm/...` output trees that the parent could not
read.

Latest focused sweep:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_scope_gate_quality50_speed64_20260630_continue2/
```

Setup: Llama-3.1-8B, EAGLE3 K=8, GSM8K-50 quality gate with an `8 pp`
accuracy-loss budget, then bs64/math_reasoning/max128 speed ceiling only for
quality-passing scopes.

| scope | GSM8K-50 dense -> base-only | quality | full-batch speedup | total speedup | read |
|---|---:|---|---:|---:|---|
| `gate_up_proj=16-31` | `0.72 -> 0.64` | pass exactly | `1.102x` | `1.007x` | not enough headroom for `speclink_t08` correction |
| `down_proj=8-15` | `0.72 -> 0.58` | fail | skipped | skipped | down-only base-only hurts reasoning |
| `gate_up_proj=31` with `up_sparse` | `0.72 -> 0.74` | pass | `0.967x` | `0.911x` | quality-safe but no speed ceiling |

This rules out another simple layer-scope-only base-only route: scopes that
preserve GSM8K within the allowed budget do not have a `1.2x` base-only
full-batch ceiling, so `speclink_t08` cannot reach `1.2x` after adding any
residual correction unless the mixed sparse/dense operator changes.

### Down-only controller

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downall_prefix2_bucket32_gsm8k50_quality_20260630_032223/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downall_prefix2_bucket32_bs64_math2048_20260630_032223/
```

GSM8K-50 quality is lossless in aggregate: dense `0.72`, SR24 `0.72`.
Throughput is still negative on bs64/max2048:

| metric | dense | SR24 | speedup |
|---|---:|---:|---:|
| total output tok/s | 2419.150 | 2382.584 | 0.985x |
| full-batch output tok/s | 5360.884 | 4562.316 | 0.851x |
| accepted draft tokens/step | 3.938 | 3.268 | lower |

Down-only has a reasonable isolated operator profile, but it reduces EAGLE3
acceptance enough to lose end-to-end.

## Component Breakdown

Result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_component_activefused_shapes_20260630_032223/
```

Microbench summary:

- `gate_up_proj` sparse base is faster than dense (`0.58-0.68x`), but adding
  correction/assembly makes the mixed path slower than dense (`1.07-1.62x`).
- `down_proj` is promising only at larger row counts and low correction
  fractions (`0.78-0.92x` for rows 576 with bucket copy), but small rows are
  often not faster than dense.
- The current Triton bucket dense-GEMM/scatter is not the winning path for
  gate_up; fixed-layout active masks alone are not enough.

Compressed residual GPU probe:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_probe_20260630_continue/
```

The current packed `compressed_dense` Triton residual kernel is GPU-resident but
not fast enough.  Against a materialized residual weight plus torch GEMM, the
best packed Triton result is `7.610x` slower for Llama `gate_up_proj`
(`0.5361 ms` dense residual graph vs `4.0799 ms` Triton) and `9.183x` slower
for `down_proj` (`0.2921 ms` vs `2.6829 ms`).  Therefore the best current
`compressed_dense` all-corrected ablation is cached/prewarmed GPU materialized
residual GEMM, not `SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=1`.  A real
all-corrected speedup requires a fused/grouped kernel that combines the two
2:4 halves without doubling launches and memory traffic; the present packed
Triton kernel should remain diagnostic-only.

## Interpretation

The useful direction is still a systems/operator redesign, not another scalar
threshold sweep:

- unimportant rows should execute only the 2:4 sparse branch;
- important rows should not also pay sparse-base work if they will be dense;
- tiny important-row groups need grouping/tile fill/fallback rather than small
  per-step GEMMs;
- sparse and dense branches should be disjoint and graph-stable before testing
  stream overlap.

The next credible implementation should use fixed-capacity route tables and a
grouped/fused row-routed MLP operator.  The current Python/Torch row routing,
active-only compaction, and bucket fused scatter paths do not yet give a robust
serving win.
