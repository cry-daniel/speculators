# SR24 Lossy Optimization Status, 2026-06-29

## Current Rule

The SR24 optimization target is no longer paired no-loss. New candidates should
use an explicit quality budget:

- primary quality gate: GSM8K `limit=50`
- acceptable loss: no worse than `-8 percentage points` vs dense EAGLE3
- speed target: at least `1.2x` dense baseline on most batch sizes
  `8/16/32/64` and datasets

Use this runner for small gates:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec python scripts/run_sr24_lossy_speed_quality_sweep.py \
  --candidates lowresidual_gateup_riskcap2,lowresidual_gateup_riskcap2_rowrouted \
  --batch-sizes 64 \
  --fixed-total-requests 64
```

The script writes final summaries to `results.bak/` and work directories to
`temp/`.

For pure base-only scope probes, use the quality gate in the scope runner:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
conda run -n spec python scripts/run_sr24_baseonly_scope_sweep.py \
  --quality-gate \
  --quality-limit 50 \
  --max-accuracy-drop-pp 8 \
  --cases gateup16_31,gateup_all,accuracy_tail_gateup31_up_sparse \
  --batch-size 64 \
  --fixed-total-requests 64 \
  --max-tokens 128
```

## Small Probe Results

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260629_233738/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_speed_quality_sweep_20260629_234328/
```

| candidate | GSM8K-50 dense | GSM8K-50 SR24 | delta pp | full-batch speedup | total speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lowresidual_gateup_riskcap2` | 0.7200 | 0.7200 | 0.0 | 1.0185x | 0.9257x |
| `lowresidual_gateup_riskcap2_rowrouted` | 0.7200 | 0.7200 | 0.0 | 0.9906x | 0.9192x |

Both pass the 8pp quality gate, but neither is a useful speed point.

Additional route/operator probe root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/speclink_sr24_mixed_bucket_probe_20260629_235410/
```

This microbenchmark uses a Llama gate_up-shaped Linear
`[rows=512, out=28672, in=4096]`. Best graph timings:

| bucket rows | dense ms | sparse base ms | base/dense | base-first correction/dense | routed no-scatter/dense | routed full/dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5377 | 0.3546 | 0.660x | 0.935x | 0.943x | 1.558x |
| 8 | 0.5412 | 0.3553 | 0.657x | 0.978x | 0.972x | 1.570x |
| 32 | 0.5414 | 0.3560 | 0.657x | 0.988x | 0.953x | 1.512x |
| 64 | 0.5418 | 0.3561 | 0.657x | 1.027x | 0.943x | 1.462x |
| 128 | 0.5422 | 0.3563 | 0.657x | 1.151x | 0.765x | 1.213x |
| 256 | 0.5437 | 0.3562 | 0.655x | 1.542x | 0.879x | 1.225x |

Read: pure sparse base has enough kernel headroom, but the current
base-first+dense-correction path loses the headroom as soon as it protects
rows. Full routed assembly is slow for small important-row groups; a routed
no-scatter lower bound only becomes attractive at large groups, which is
exactly where quality-safe controllers usually become less sparse.

Base-only quality/speed gate root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_scope_sweep_20260629_235858/
```

| scope | GSM8K-50 dense | GSM8K-50 base-only | delta pp | quality pass | bs64 math full speedup | bs64 math total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gateup16_31` | 0.72 | 0.64 | -8.00 | yes | 1.068x | 0.982x |
| `gateup_all` | 0.72 | 0.42 | -30.00 | no | skipped | skipped |
| `accuracy_tail_gateup31_up_sparse` | 0.72 | 0.74 | +2.00 | yes | 0.966x | 0.911x |

Read: a broad gate_up tail scope barely passes the 8pp quality budget but still
does not reach the 1.2x serving target. The whole gate_up scope has speed
potential but is far outside the quality budget. A tiny safe scope is too small
to improve throughput.

## 2026-06-30 Operator Follow-up

Three follow-up probes checked whether the current exact or overlap variants
can be turned into the next speed path.

Compressed residual kernel profile:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_profile_20260630_001736/summary.md
```

| shape | dense residual graph ms | Triton packed residual ms | Triton / dense | read |
| --- | ---: | ---: | ---: | --- |
| `512 x 28672 x 4096` | 0.5358 | 4.0793 | 7.614x | reject |
| `512 x 4096 x 14336` | 0.2925 | 2.6945 | 9.212x | reject |

This confirms that `compressed_dense` is GPU-side, not a CPU-transfer bug, but
the current direct packed Triton residual kernel is much slower than
materializing the residual into a dense GEMM. Do not promote
`SPECLINK_SR24_COMPRESSED_RESIDUAL_TRITON=1`.

Cached stream overlap serving probe:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_route_overlap_cached_streams_bs64_math128_20260630/report.md
```

On Llama-3.1-8B, `math_reasoning`, bs64, K=8, max tokens 128, dense reached
`2263.416` total tok/s and `3152.223` full-batch tok/s. The cached-stream
`speclink_t08` overlap row reached only `1740.457` total tok/s and
`2754.103` full-batch tok/s, or `0.769x` total and `0.874x` full-batch. The
code path now reuses per-device streams behind
`SPECLINK_SR24_ROUTE_OVERLAP_STREAMS=1`, but it remains an ablation, not a
default.

Exact no-fastpath `all_corrected_24` with torch-sparse residual:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_torchsparse_gateup16_31_bs64_math128_20260630/report.md
```

Dense reached `2266.305` total tok/s and `3153.225` full-batch tok/s. The
narrow `all_corrected_24` row with `gate_up_proj=16-31`, direct cuSPARSELt,
torch-sparse residual, reduced CPU sync, static all-residual mask state, and
CUDA Graph coverage `{"FULL": 62, "NONE": 2}` reached `1840.351` total tok/s
and `2652.219` full-batch tok/s, or `0.812x` total and `0.841x` full-batch.

Read: the exact reconstruction path is still slower even when scoped narrowly
and graph-covered. A two-sparse-GEMM exact `all_corrected_24` path should stay
a correctness/control path unless a new fused operator avoids the second launch,
addition, and residual materialization overhead.

## Interpretation

The desired dataflow is still correct:

- unimportant token rows should execute only the 2:4 sparse base branch;
- important token rows should execute dense or base+residual fidelity, but
  should not also pay a redundant sparse-base pass if dense overwrite is used;
- underfilled important-token groups need tile fill or grouping, not a tiny
  per-step GEMM.

The row-routed probe implemented the mutually exclusive row split, but it
launched dynamic gather, small dense GEMM, sparse GEMM over the remaining rows,
and scatter assembly. With only a small number of important rows, the saved
sparse work is smaller than the routing and small-kernel cost. That makes
unconditional row routing a negative result.

## Next Engineering Direction

The next candidate should be a graph-stable packed mixed operator, not another
threshold-only sweep:

1. Data format:
   fixed route slots with `{row_id, route_kind, active}` plus per-layer fixed
   row capacities. CUDA Graph should see stable pointers and shapes.

2. Tile fill:
   if important rows are too few, do not launch a tiny dense route. Either keep
   base-first overwrite or fill the dense tile with near-threshold rows only
   when the planner predicts a larger dense tile beats the base-first path.
   Filled rows count against the 8pp quality budget only through measured
   output, not by assumption.

3. Execution planner:
   choose among base-first overwrite, row-routed split, and full dense fallback
   using row counts, layer shape, and measured kernel latencies.

4. Operator work:
   dense-important and sparse-base branches need grouped execution with
   pre-created graph-safe streams, or a fused packed operator that avoids
   Python dynamic `nonzero()`, repeated `index_select`, tiny dense GEMMs, and
   post-hoc scatter assembly. The final design should explicitly support
   concurrent sparse-base and dense-important branches.

5. Validation:
   every retained candidate must report GSM8K-50 accuracy, total/full-batch
   throughput, accepted tokens/step, CUDA Graph counts, GPU util, and row-route
   component timing.

Do not promote a new SR24 default unless it beats dense in the clean serving
path and stays within the 8pp accuracy budget.

## 2026-06-30 Lossy Direct-cuSPARSELt Prefix Sweep

I added direct-cuSPARSELt all-MLP variants to
`examples/evaluate/eval-guidellm/scripts/run_sr24_lossy_speed_quality_sweep.py`:

- `mlpall_lowconf_prefix5_directcslt`
- `mlpall_direct_prefix3`
- `mlpall_direct_prefix2`
- `mlpall_direct_prefix0`

These keep the all-MLP low-confidence controller and explicitly pass
`--sr24-direct-cslt-linear`. The prefix variants also override
`--sr24-selective-min-prefix-residual`.

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlp_scope_gate_bs64_math128_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlp_directcslt_gate_bs64_math128_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlp_direct_prefix_sweep_bs64_math128_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlp_direct_prefix2_bs64_math512_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlp_direct_prefix2_bs64_math2048_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_mlp_direct_prefix5_bs64_math2048_20260630/
```

Small GSM8K-50 quality and max128 speed gate:

| candidate | GSM8K-50 dense | GSM8K-50 SR24 | delta pp | full-batch speedup | total speedup | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mlpall_lowconf_prefix5_tritonoverride` | 0.72 | 0.72 | 0.0 | 1.077x | 0.882x | quality safe, no direct cuSPARSELt |
| `mlpall_lowconf_prefix5_directcslt` | 0.72 | 0.72 | 0.0 | 1.151x | 0.937x | direct cuSPARSELt helps |
| `mlpall_direct_prefix3` | 0.72 | 0.66 | -6.0 | 1.132x | 0.936x | lower prefix hurts quality and not faster |
| `mlpall_direct_prefix2` | 0.72 | 0.66 | -6.0 | 1.193x | 0.957x | best max128 point, within 8pp |
| `mlpall_direct_prefix0` | 0.72 | 0.52 | -20.0 | skipped | skipped | outside quality budget |

Longer-output checks for `mlpall_direct_prefix2`:

| max tokens | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | accepted draft/step dense -> SR24 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 2483.798 | 2945.462 | 1.186x | 4429.165 | 5121.298 | 1.156x | 2.436 -> 3.067 |
| 2048 | 2428.081 | 2475.675 | 1.020x | 5369.123 | 4419.230 | 0.823x | 4.014 -> 4.176 |

The max2048 result is the key negative result: on long math outputs, dense
EAGLE3 already has high acceptance, so the SR24 controller saves too few
verification steps to cover the per-step mixed sparse/residual operator cost.
The more conservative `prefix5` 2048 check is worse (`2373.043` vs dense
`2428.693` total tok/s).

Current read:

- Direct cuSPARSELt is a real improvement over the previous all-MLP Triton
  override path and should stay available as an experimental candidate.
- Reducing mandatory dense prefix below 2 is outside the current 8pp quality
  budget; prefix3 is not faster than prefix2.
- The 1.2x target is not robust at max2048 with the current two-pass
  sparse-base plus dense-residual implementation. The next optimization should
  be operator/data-format work: graph-stable route tables, grouped sparse and
  dense branches, and a fused assembly path. More scalar threshold sweeps are
  unlikely to solve the long-output case.

## 2026-06-30 Active-Mask Fused Bucket Follow-up

The dynamic active-only bucket path is a negative result. It really avoids
dense GEMM for inactive bucket rows, but the variable-shape
`nonzero()/index_select()` path is too expensive and CUDA-Graph unfriendly:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activeonly_vs_activefused_bs64_math128_20260630/
```

On Llama-3.1-8B `math_reasoning`, bs64, K=8, max128,
`mlpall_direct_prefix2_activeonly` passed the GSM8K-50 budget exactly
(`0.72 -> 0.64`, `-8 pp`) but reached only `0.381x` total and `0.511x`
full-batch throughput versus dense.

I added `--sr24-bucket-dense-active-mask-fused`, which keeps the bucket tensor
shape fixed and lets the Triton fused GEMM+scatter read `bucket_values` as the
GPU active mask. This is a graph-safe data-format probe, not a final operator:
inactive rows still occupy the fixed bucket tile, but Python dynamic compaction
is removed.

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_quality_speclink_only_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_direct_bs64_math128_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_direct_bs64_math512_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_direct_bs64_math2048_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_breakdown_bs64_math128_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_activefused_directpos_bs64_math512_20260630/
```

Quality:

| task | dense reference | active-mask fused SR24 | delta pp |
| --- | ---: | ---: | ---: |
| GSM8K-50 flexible exact match | 0.72 | 0.66 | -6.0 |

Clean serving speed, bs64/K=8 `math_reasoning`:

| max tokens | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | accepted draft/step dense -> SR24 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 2258.660 | 2169.181 | 0.960x | 3026.842 | 3634.799 | 1.201x | 1.426 -> 1.789 |
| 512 | 2482.865 | 2907.552 | 1.171x | 4426.090 | 5240.547 | 1.184x | 2.436 -> 2.995 |
| 2048 | 2427.632 | 2520.563 | 1.038x | 5386.383 | 4681.783 | 0.869x | 3.978 -> 4.400 |

Breakdown read from the diagnostic max128 row:

- CUDA Graph coverage is mostly fixed: `FULL=62,NONE=2`.
- Dense correction is not the bottleneck anymore:
  `bucket_active_mask_fused_dense_gemm_scatter_cuda_ms=7.6 ms`.
- Sparse base Linear dominates:
  `base_sparse_linear_cuda_ms=148.4 ms` across the run.
- Scheduler/bucket overhead is secondary but visible:
  `scheduler_mask_build_cpu_ms=35.8 ms`,
  `scheduler_bucket_build_cpu_ms=9.6 ms`, and
  `scheduler_bucket_topk_cuda_ms=6.3 ms`.
- `--sr24-direct-position-bucket` did not improve max512 throughput
  (`2817.502` total vs `2907.552` for the normal active-mask fused run).

Current read:

- Active-mask fused is the best small implementation step so far and should
  replace dynamic active-only for future graph-safe experiments.
- It is still below the robust `1.2x` target: max512 is close, max2048 is not.
- The next implementation needs a disjoint row-routed/grouped operator that
  avoids running sparse base on rows that will be dense-corrected. Since the
  corrected bucket is a small fraction of total sparse rows, this also needs
  tile-fill/grouping across requests or layers; otherwise saved sparse work is
  smaller than routing and small-kernel overhead.

## 2026-06-30 Disjoint Row-Routed Probe

I tested the direct interpretation of the user-requested optimization:
important rows go straight to dense Linear, unimportant rows go straight to
2:4 sparse Linear, so dense-corrected rows do not also pay the sparse-base
branch.

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_disjoint_rowrouted_min1_bs64_math512_20260630/
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_disjoint_rowrouted_breakdown_bs64_math128_20260630/
```

Clean serving speed, bs64/K=8 `math_reasoning`, max512:

| method | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util |
| --- | ---: | ---: | ---: | ---: |
| dense EAGLE3 | 2481.661 | 4422.480 | 2.436 | 93.1% |
| SR24 disjoint row-routed | 2685.923 | 4564.406 | 3.035 | 90.0% |
| speedup | 1.082x | 1.032x | - | - |

The profiling row confirms that row-routed MLP really executed, but the current
Python/Torch row-list implementation is not the target operator:

| component | total CUDA ms | avg per timed call |
| --- | ---: | ---: |
| row-routed base gate_up 2:4 | 50.49 | 1.578 |
| row-routed base down 2:4 | 38.10 | 1.191 |
| row-routed dense gate_up | 4.94 | 0.154 |
| row-routed dense down | 2.70 | 0.084 |
| row-routed gather/index-copy/act | 3.54 | - |
| bucket top-k + base-row build | 11.44 | - |

Routing counts in the diagnostic row:

- row-routed MLP calls: `1568`
- dense rows: `50176`
- base rows: `442464`
- bucket fill ratio: `0.987`
- CUDA Graph modes: `FULL=58`, `NONE=2`

Read:

- The principle is correct, but the current implementation is the wrong data
  format: it slices rows with `index_select`, runs separate small GEMMs, and
  assembles with `index_copy`/Triton scatter.
- The sparse branch remains dominant after routing because most rows are still
  base rows, and the branch is now a smaller, less efficient 2:4 GEMM than the
  graph-stable full-row sparse base.
- This path should not replace active-mask fused. The next operator should be
  fixed-capacity and grouped: build a persistent route table, pack dense rows
  and sparse rows into large-enough tiles, and only then consider two-stream
  dense/sparse overlap.

Concrete next implementation target:

1. Scheduler emits a fixed-shape route table per decode step:
   `(row_id, lane, active, route=dense|sparse, request_id, draft_position)`.
   This keeps CUDA Graph replay stable and avoids per-layer `nonzero()`.
2. A grouped MLP operator consumes the table directly. Dense rows use dense
   GEMM; sparse rows use 2:4 GEMM; assembly is fused with route metadata.
3. If important rows are too few, fill dense tiles with the next highest-risk
   draft rows or fall back to active-mask fused. The quality budget is still
   GSM8K limit 50 with at most 8 pp absolute accuracy loss.
4. Sparse and dense branches should be overlapped only when both sides exceed a
   measured tile threshold. The existing stream-overlap hook is diagnostic and
   graph-unsafe; it is not a serving default.
