# SR24 Current Slowness Breakdown Direction

This note resets the next SR24/SpecLink tuning pass around an explicit
slowdown breakdown. The immediate goal is to explain where the current path is
slow before running another selector or throughput sweep.

Current reference artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/SR24_SLOWNESS_BREAKDOWN_CURRENT.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/run_sr24_slowdown_breakdown.py
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_breakdown_bs64_math_20260629/seven_part_report_with_base_safe/report.md
```

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
