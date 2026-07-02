# SR24 8pp Optimization Status - 2026-07-02

## Current Results

All paths below are absolute result roots.

0. `base_only_24` slow rows are not acceptance-limited.
   - Diagnostic script:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/analyze_sr24_baseonly_slowdown.py`
   - Result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_baseonly_slow_reason_refresh_20260702`
   - Across 8 representative `base_only_24` rows, 6 are slower than dense, but
     all 6 accept more draft tokens per step than dense. They are slow because
     GPU utilization/CUDA Graph coverage/config scope is bad, not because
     accepted length collapses. Two graph-enabled MLP-only rows are fast:
     `1.697x` and `1.844x` full-batch speedup with healthy acceptance and GPU
     utilization.

0b. `all_corrected_24` compressed-dense is GPU-resident but still not a speed
    path.
   - Diagnostic script:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/analyze_sr24_allcorrected_path.py`
   - Result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_backend_diagnosis_refresh_20260702`
   - The refresh analyzes 5 `all_corrected_24` rows, 4 with serving
     throughput. Three `compressed_dense` rows are proven GPU-resident
     (`cuda:0` residual modules, `compressed_residual_runtime_on_gpu=True`,
     no non-GPU modules). All 4 measured rows are slower than dense with
     accepted draft length intact. The bottleneck is two-pass sparse-base plus
     residual-correction work, not CPU residual fallback or acceptance collapse.

1. Prefix0 verifier-only is not enough.
   - Result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix0_live_candidate_gate_bs64_20260702`
   - GSM8K-20 quality passed, but bs64 math throughput was only `2336.7`
     tok/s vs dense `2326.3` tok/s.

2. Current policy/stream-overlap is not the packed operator.
   - Result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix0_policy_overlap_bs64_20260702`
   - bs64 math throughput dropped to `1926.3` tok/s vs dense `2323.3`
     tok/s, with lower GPU utilization.

3. Broad no-verify sparse is not quality-safe.
   - Result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix0_noverify_sparse_quality_gsm8k50_20260702`
   - GSM8K-50 dropped from dense `0.7800` to `0.1400`.

4. Prefix1/front28 is quality-feasible but not fast enough.
   - Quality source:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_tail_sparse_prefix_quality_gsm8k50_20260701`
   - It keeps ordinary/no-verify MLP rows dense in layers 0-27, uses
     sparse-only ordinary/no-verify rows only in layers 28-31, and protects
     draft position 0 plus the verifier bonus row with dense MLP.
   - GSM8K-50: `0.7800 -> 0.7400` (-4pp), within the 8pp budget.
   - Throughput result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_front28_noverifydense_math_bs8_16_32_64_20260702`

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1146.429 | 1019.400 | 0.889x | 1239.513 | 1195.437 | 0.964x |
| 16 | 1809.907 | 1484.214 | 0.820x | 1959.049 | 1906.322 | 0.973x |
| 32 | 2265.540 | 2343.945 | 1.035x | 2675.476 | 2779.697 | 1.039x |
| 64 | 2651.153 | 2735.618 | 1.032x | 3451.962 | 3577.881 | 1.036x |

5. Python stream overlap is not useful for this candidate.
   - Result:
     `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_front28_overlap_bs64_20260702`
   - bs64 math throughput dropped to `2572.6` tok/s vs dense `2662.1`
     tok/s.

## Interpretation

The current row-routed path already avoids dense recompute for sparse-only
verifier rows. The remaining issue is not the controller threshold alone. The
serving path is still a collection of small PyTorch/cuSPARSELt calls plus dense
fallbacks, so sparse rows do not become a stable system-level win.

The 8pp budget is useful because it identifies acceptable policies, but it does
not make the current operator fast enough. Low batch sizes are especially bad:
the sparse branch is underfilled at bs8/16, and stream overlap increases
overhead instead of hiding it.

## Next Implementation Direction

The next credible path to a PPoPP-style result is:

1. Fixed-capacity verifier-block data format:
   store each verify step as contiguous `[request, K + 1, hidden]`, with a
   compact route descriptor for dense-important rows and sparse-only rows.

2. Grouped verifier queue:
   coalesce verifier blocks until the sparse branch reaches an effective bs64
   shape. For K=8, the prefix0 microbench needs group factors roughly bs8->8,
   bs16->4, bs32->2, bs64->1.

3. Dense bypass underfill policy:
   if a group cannot fill the sparse operator without delaying too long, bypass
   SR24 and run the original dense MLP. This avoids bs8/16 regressions.

4. Fused or packed mixed MLP operator:
   run dense-important rows and 2:4 sparse-unimportant rows through one
   low-overhead operator boundary. Python-level stream overlap should stay a
   diagnostic only.

5. Quality gate:
use GSM8K limit 50 as the first reasoning gate and allow at most 8pp
accuracy loss before spending time on larger throughput matrices.

## 8pp Row-Routed Gate Refresh

I reran a small max-token-128 gate on `math_reasoning` with GSM8K-50 as the
quality filter. Outputs were kept out of `results/`:

- Main result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_8pp_importance_rowrouted_gate_20260702`
- Work logs:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_8pp_importance_rowrouted_gate_20260702_work`

Quality gates:

| candidate | GSM8K-50 dense | GSM8K-50 SR24 | delta pp | quality verdict |
|---|---:|---:|---:|---|
| `lossy_prefix2_down_front14_compile` | 0.7400 | 0.6800 | -6.0 | pass |
| `fixedprefix4_bucket16_directcslt_rowrouted_min128` | 0.7200 | 0.6800 | -4.0 | pass |
| `criticalprefix4_bucket16_directcslt_rowrouted_min128` | 0.7400 | 0.7400 | 0.0 | pass |

Throughput, total output tok/s speedup over dense:

| candidate | bs8 | bs16 | bs32 | bs64 | best total |
|---|---:|---:|---:|---:|---:|
| `lossy_prefix2_down_front14_compile` | 0.783x | 0.741x | 0.447x | 1.028x | 1.028x |
| `fixedprefix4_bucket16_directcslt_rowrouted_min128` | 0.433x | 0.522x | 0.661x | 0.832x | 0.832x |
| `criticalprefix4_bucket16_directcslt_rowrouted_min128` | 0.796x | 0.737x | 0.789x | 0.704x | 0.796x |

The important point is that the fixed-block row-routed implementation already
has disjoint semantics: sparse-only rows are not recomputed through dense in
the normal fixed-block path. The current loss is therefore system overhead:
multiple small dense/sparse launches, row assembly, graph limitations, and
underfilled sparse shapes.

## Output-Buffer And Stream-Overlap Check

I then tested two down-front14 follow-ups without rerunning quality, because
they keep the same dense/sparse token policy as the already passing down-front14
candidate:

- Result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_8pp_outputbuf_overlap_ablation_20260702`
- Work logs:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/temp/sr24_8pp_outputbuf_overlap_ablation_20260702_work`

`lossy_prefix2_down_front14_outputbuf_compile` uses the reusable fixed-block
output workspace and keeps vLLM compile/CUDA Graphs enabled:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 976.630 | 758.113 | 0.776x | 1068.480 | 943.724 | 0.883x |
| 16 | 1513.952 | 1122.636 | 0.742x | 1722.719 | 1509.428 | 0.876x |
| 32 | 1921.317 | 1486.415 | 0.774x | 2310.506 | 2027.906 | 0.878x |
| 64 | 2202.829 | 1988.672 | 0.903x | 3014.295 | 3181.636 | 1.056x |

The output buffer helps the bs64 full-batch window but does not make end-to-end
serving faster. The request-drain and split-operator overhead still dominate.

`lossy_prefix2_down_front14_outputbuf_overlap_compile` adds CUDA stream overlap
for dense-important and 2:4 sparse-base branches. The first SR24 row was much
worse and the remainder was stopped to save GPU time:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 955.617 | 536.453 | 0.561x | 1055.041 | 636.765 | 0.604x |

This makes Python-level stream overlap a diagnostic only. It breaks the
graph-friendly path and adds synchronization/stream overhead before there is
enough work to hide.

## Updated Optimization Direction

The next change should not be another threshold-only or Python routing tweak.
The implementation needs a real system data path:

1. Keep the 8pp quality budget and GSM8K-50 gate. Do not require lossless
   accuracy; otherwise the controller chooses too many dense rows and removes
   the speedup opportunity.
2. Keep the disjoint dense/sparse row semantics: if an unimportant token has
   already gone through the 2:4 sparse branch, do not later recompute it with
   dense.
3. Replace per-step Python routing with a fixed verifier-block descriptor:
   `[request, K + 1, hidden]` plus compact dense/sparse row ranges. The current
   fixed-block path approximates this layout but still launches separate
   high-level ops.
4. Implement a packed/fused mixed MLP boundary for verifier blocks. The
   operator should consume dense-important rows and sparse-unimportant rows in
   one packed call boundary, with fused assembly into the original verifier
   order.
5. Add a grouped verifier queue for underfilled cases. At bs8/16, accumulate
   verifier blocks until the sparse branch has an effective bs64-ish shape, or
   bypass SR24 and run dense if the queue would wait too long.
6. Treat CUDA stream overlap as an operator-internal design, not as Python-level
   `torch.cuda.stream` around separate PyTorch/cuSPARSELt calls.

The current evidence says the 8pp controller can find quality-feasible sparse
policies, but the live serving code needs a data-format/operator change before
it can plausibly reach the requested 1.2x on bs8/16/32/64.

## Down-Front14 Output-Buffer Check

`lossy_prefix2_down_front14_compile` remains the closest 8pp-budget candidate:
on `math_reasoning`, max tokens 256, it reaches total speedups
`0.959x/0.950x/1.025x/1.155x` for bs `8/16/32/64`, with GSM8K-50
`0.7800 -> 0.7000` (`-8pp`). It keeps gate/up dense, protects prefix2 plus
bonus for down_proj, and makes no-verify down_proj sparse-only only in layers
14-31.

The output-buffer follow-up did not improve it:

- Result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downfront14_outputbuf_gate_bs32_64_math256_20260702`
- GSM8K-50 in that sampled run: dense `0.7200`, SR24 `0.6600` (`-6pp`), so the
  quality gate passed.

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2443.751 | 2172.777 | 0.889x | 2663.626 | 2697.775 | 1.013x |
| 64 | 2593.227 | 2708.248 | 1.044x | 3149.479 | 3282.700 | 1.042x |

Conclusion: reusing the fixed-block output buffer is not the missing
optimization for the lossy down-only path. The remaining gap is the split
dense/sparse operator boundary plus PIECEWISE graph coverage, not just output
allocation or scatter assembly overhead.

## Down-Front13 Boundary Check

I checked the untested boundary between the passing `down_front14` and failing
`down_front12` policies by keeping no-verify `down_proj=0-12` dense and using
2:4 sparse-only for no-verify `down_proj=13-31`.

- Quality result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_front13_quality_gsm8k50_20260702`
- Throughput result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_down_front13_throughput_bs32_64_math256_20260702`

GSM8K-50 passed the 8pp budget:

| method | accuracy | delta pp | pair reg / imp |
|---|---:|---:|---:|
| dense baseline | 0.7800 | 0.0 | 0 / 0 |
| down_front13 speclink_t08 | 0.7200 | -6.0 | 6 / 3 |

But serving speed did not improve:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2434.282 | 2241.934 | 0.921x | 2660.992 | 2661.900 | 1.000x |
| 64 | 2822.410 | 2622.946 | 0.929x | 3224.570 | 3248.321 | 1.007x |

Conclusion: `down_front13` is quality-feasible under the 8pp budget, but it is
not a system-speed candidate in the current implementation. Making one more
down_proj layer sparse-only does not overcome the split dense/sparse row-routed
operator overhead.

## All-Corrected CompressedDense Chunk Check

I also checked whether the slow exact `all_corrected_24` compressed-dense path
was mainly caused by gate_up's default seven output chunks. It is not.

- Result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_allcorrected_compresseddense_gateup_chunk0_bs64_math64_20260702`
- Setup: Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, bs64, max tokens 64,
  `target_leafs=gate_up_proj`, `residual_backend=compressed_dense`,
  `residual_device=cuda`, `--no-sr24-all-corrected-dense-fastpath`,
  `--no-sr24-auto-compressed-residual-fastpath`, and
  `--sr24-residual-out-chunk 0`.

| method | total tok/s | full-batch tok/s | avg GPU util | accepted draft/step | graph modes |
|---|---:|---:|---:|---:|---|
| dense baseline | 1789.089 | 2602.309 | 69.2% | 1.146 |  |
| all_corrected compressed_dense chunk0 | 510.243 | 864.218 | 92.8% | 1.083 | `NONE=55` |

The SR24 stats showed `sr24_compressed_residual_runtime_on_gpu=True`,
`sr24_residual_backend_counts={"compressed_dense":32}`, and
`sr24_residual_device_counts={"cuda:0":32}`. So the current all-corrected
compressed path is not slow because residual values are on CPU, and not merely
because gate_up was sliced into seven output chunks. It is slow because exact
correction still executes sparse base plus a separate residual matmul/add path.
That path needs a fused base+residual operator to become useful.

## Rowguard16 Update

Implemented a compile-safe no-verify row guard:

- vLLM env: `SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_MAX_ROWS`
- runner flag: `--sr24-selective-dense-nonverify-max-rows`
- static compile hint: `SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_STATIC_ROWS`
  is set by the throughput runner to the current client batch size and by the
  lm-eval runner to `max_num_seqs`.

The guard makes small no-mask/no-verify forwards use dense weights even when a
layer scope would otherwise allow the 2:4 sparse base. This avoids symbolic
shape comparisons inside torch.compile and prevents low-row sparse kernels from
being forced into the graph.

Results:

- GSM8K-50 quality sanity:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_rowguard16_gsm8k50_quality_20260702`
  - dense `0.7800`
  - speclink_t08 rowguard16 `0.7800`
  - paired regressions/improvements: `1 / 1`
- math_reasoning throughput, bs8/16:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_front28_rowguard16_math_bs8_16_20260702_rerun2`
- math_reasoning throughput, bs32/64:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_front28_rowguard16_math_bs8_16_32_64_20260702_rerun`

| bs | dense total tok/s | SR24 total tok/s | total speedup |
|---:|---:|---:|---:|
| 8 | 1146.401 | 1026.445 | 0.895x |
| 16 | 1747.864 | 1502.938 | 0.860x |
| 32 | 2325.817 | 1911.789 | 0.822x |
| 64 | 2328.871 | 2344.978 | 1.007x |

Conclusion: rowguard16 is worth keeping as a correctness/reliability guard for
compile-safe experiments, but it does not solve the throughput target. It
confirms the real bottleneck is the current operator/data-format path:
small sparse branches and graph breaks/PIECEWISE steps dominate before the 2:4
FLOP reduction can turn into serving-level speedup.

## Near-Full Fixed-Capacity Padding Update

Implemented an explicit near-full packed-data-format ablation:

- vLLM env: `SPECLINK_SR24_SCHEDULER_POLICY_NEAR_FULL_TOLERANCE`
- vLLM env: `SPECLINK_SR24_FIXED_BLOCK_CAPACITY_PADDING`
- runner flags: `--sr24-scheduler-policy-near-full-tolerance` and
  `--sr24-fixed-block-capacity-padding`

When the scheduler policy maps a verifier block to a larger policy batch
within the tolerance, the fixed-block MLP path pads dense-important rows and
2:4 sparse-unimportant rows to the policy capacity with dummy rows, runs the
current mixed branch at the stable capacity, and assembles only real rows. This
keeps important rows dense-only and unimportant rows 2:4-only; it does not add a
dense correction for rows that already went through sparse.

Quality gate:

- Result:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_nearfull_padding_prefix2_gsm8k50_quality_20260702`
- Llama-3.1-8B GSM8K-50:
  - dense flexible exact match: `0.7800`
  - speclink_t08 flexible exact match: `0.7800`
  - speclink_t08 strict exact match: `0.7400`
- This is inside the 8 percentage-point absolute quality budget.

Throughput probes on Llama-3.1-8B, `math_reasoning`, bs64, K=8, max tokens 256:

| config | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup | note |
|---|---:|---:|---:|---:|---:|---:|---|
| previous prefix2 policy | 2839.721 | 2881.606 | 1.015x | 3053.304 | 3178.273 | 1.041x | no capacity padding |
| tolerance=2 + capacity padding | 2842.210 | 2947.745 | 1.037x | 3063.335 | 3212.299 | 1.049x | current best live knob |
| tolerance=2 + capacity padding, no dummy zero-fill | 2850.095 | 2946.487 | 1.034x | 3061.137 | 3208.142 | 1.048x | removes dummy tail zero writes, neutral |
| tolerance=2 + capacity padding + route-overlap graph allow | 2849.958 | 2686.038 | 0.943x | 3069.403 | 3040.931 | 0.991x | do not use |
| tolerance=2 + capacity padding + Triton route assembly | 2838.674 | 2644.320 | 0.932x | 3072.153 | 3025.379 | 0.985x | do not use |
| tolerance=8 + capacity padding | 2844.589 | 2929.737 | 1.030x | 3063.167 | 3183.060 | 1.039x | too many dummy rows |
| tolerance=2 + capacity padding + dense bypass | 2890.019 | 2643.205 | 0.915x | 3115.401 | 3031.498 | 0.973x | do not use |

Trace evidence:

- tolerance=2 maps active verifier counts `62/63/64` to the bs64 policy.
- In the bs64 probe, fixed-block mixed events increased to `189 / 309`; the
  remaining `120 / 309` fixed events still fell back or were underfilled.
- tolerance=8 increased mixed events to `213 / 312`, but extra dummy-row work
  reduced throughput.
- Dense bypass is negative because returning to the original dense path for
  fallback cases lowers GPU utilization and disrupts the current graph/serving
  shape.
- Skipping dummy-row zero-fill is safe for the current row-independent
  dense/sparse MLP branch because dummy outputs are discarded, but it is
  throughput-neutral in the live bs64 probe. Keep
  `SPECLINK_SR24_FIXED_BLOCK_CAPACITY_ZERO_DUMMY=0` as the default and do not
  treat dummy zero-fill as the remaining bottleneck.
- `SPECLINK_SR24_ROUTE_OVERLAP_ALLOW_CUDAGRAPH=1` does not fix CUDA Graph
  coverage for this path: the probe still reports `FULL=44, PIECEWISE=44` and
  regresses to `0.943x` total. Keep it off for serving claims.
- `SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=1` is also negative here. The trace still
  shows 324 mixed opportunities with average dense/base rows around `144/289`,
  but total throughput drops to `0.932x` and full-batch throughput to `0.985x`.
  The extra assembly kernel is not the missing fused operator.

Conclusion: fixed-capacity near-full padding is worth keeping as a small,
system-shaped improvement and as evidence for the data format, but it is not
the 1.2x path. The remaining work must implement a dependency-safe grouped
verifier queue or a fused packed MLP operator so bs8/16/32 can coalesce verifier
blocks to effective bs64 without excessive dummy rows. Further scalar threshold
sweeps are unlikely to close the gap.

## DownProj Fixed-Block Operator Probe

I added a down-proj-only packed operator microbenchmark and a gated live
fixed-block down route. The live route is default-off through:

```text
SPECLINK_SR24_ROW_ROUTED_DOWN_FIXED_BLOCK=0
```

The probe isolates the current down-only 8pp candidate where gate/up stays dense,
prefix+bonus down rows are dense-only, and remaining draft down rows are 2:4
sparse-only.

- Operator probe:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_downproj_operator_probe_20260702`
- Dense-fill probe:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_downproj_densefill128_probe_20260702`
- Live default-off regression:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downfront14_default_after_gate_bs64_math256_20260702`
- Live fixed-block down route:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downfront14_fixedblock_down_fastpath_bs64_math256_20260702`
- Live fixed-block down route plus Python stream overlap:
  `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downfront14_fixedblock_down_overlap_bs64_math256_20260702`

Important operator-level observations for K=8/prefix2:

| shape | packed/dense graph time | parallel/dense graph time | read |
|---|---:|---:|---|
| bs8, coalesce1 | 2.666x | 2.624x | much slower |
| bs16, coalesce1 | 2.390x | 2.337x | much slower |
| bs32, coalesce1 | 1.748x | 1.759x | slower |
| bs64, coalesce1 | 0.999x | 0.816x | only parallel has local headroom |
| bs8, coalesce8 | 1.003x | 0.817x | grouping reaches bs64-like shape |
| bs16, coalesce4 | 1.001x | 0.820x | grouping reaches bs64-like shape |
| bs32, coalesce2 | 0.996x | 0.816x | grouping reaches bs64-like shape |

So the useful data-format lesson is not "promote extra unimportant rows to
dense." The dense-fill=128 probe shows that this either degenerates to all-dense
at bs8 or leaves a tiny sparse branch at bs16/32 and becomes slower. The better
system direction is to coalesce verifier blocks so both branches operate at an
effective bs64 shape, or to bypass SR24 when the branch is underfilled.

The live fastpath did not translate the microbench upper bound:

| config | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---|---:|---:|---:|---:|---:|---:|
| default after gate | 2696.262 | 2698.708 | 1.001x | 3527.636 | 3599.769 | 1.020x |
| fixed-block down route | 2324.030 | 2219.620 | 0.955x | 3431.405 | 3226.095 | 0.940x |
| fixed-block down + stream overlap | 2321.842 | 2076.717 | 0.894x | 3441.876 | 3189.822 | 0.927x |

Conclusion: keep `SPECLINK_SR24_ROW_ROUTED_DOWN_FIXED_BLOCK=0` by default.
Python-level stream overlap is not the PPoPP-quality optimization; it lowers live
GPU utilization. The next real optimization needs a grouped verifier queue or a
fused CUDA/Triton operator that exposes the dense and sparse branches under one
low-overhead scheduling boundary.

## Amdahl Ceiling and Operator-Guard Recheck

I added an offline Amdahl ceiling estimator:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/estimate_sr24_amdahl_ceiling.py
```

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_amdahl_ceiling_llama3_20260702
```

The key result is that down-only scopes cannot reach the 1.2x serving target
even with optimistic local operators:

| scope | target compute share | 1.225x local op | 2.0x local op | free-op ceiling |
|---|---:|---:|---:|---:|
| down_front14 | 15.14% | 1.029x | 1.082x | 1.178x |
| down_front13 | 15.99% | 1.031x | 1.087x | 1.190x |
| all_down | 26.92% | 1.052x | 1.156x | 1.368x |
| all_mlp | 80.77% | 1.174x | 1.677x | 5.200x |

So the 1.2x target requires broad MLP coverage with a real mixed operator, not
another down-proj boundary sweep.

I also rechecked the current 8pp-budget operator-guard path on the current tree.
Quality passed GSM8K-50:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_front28_current_gsm8k50_20260702
```

| method | GSM8K-50 flexible exact match | delta |
|---|---:|---:|
| dense_baseline | 0.7800 | 0.0pp |
| speclink_t08 operator_guard + front28 | 0.7400 | -4.0pp |

The matching throughput sanity with scheduler-policy dense bypass did not meet
the speed target:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_front28_densebypass_math_bs8_64_max256_20260702
```

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1148.004 | 1079.823 | 0.941x | 1204.740 | 1207.395 | 1.002x |
| 16 | 1820.612 | 1606.733 | 0.883x | 1985.459 | 1924.900 | 0.969x |
| 32 | 2339.211 | 2045.577 | 0.874x | 2649.358 | 2619.778 | 0.989x |
| 64 | 2625.870 | 2452.185 | 0.934x | 3303.423 | 3185.121 | 0.964x |

The grouping trace explains why: without a grouped verifier queue, bs16/32
policy rows are `dense_fallback_until_grouped`, and even bs64 falls back for
many non-full steps. The run had lower SR24 GPU utilization (`~80%`) than dense
(`~93-96%`), so dense bypass alone does not remove the hook/graph overhead.

The Python-stream overlap sanity, which makes the current live path accept the
planner's single-block `packed_parallel` row, was also negative:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_front28_overlap_bs64_max256_20260702
```

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2607.407 | 2330.475 | 0.894x | 3227.327 | 3072.812 | 0.952x |

Trace summary: only 49 bs64 steps actually used the single-block mixed path,
while 105 steps still reported `operator_unimplemented` for grouped
`packed_parallel` and 30 fell back as underfilled. This validates the design
direction but rejects Python stream overlap as the implementation.

I also added a diagnostic-only serial fallback for single-block
`packed_parallel`:

```text
SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_SERIAL_PACKED_PARALLEL=1
```

It lets the scheduler execute the current dense-important plus 2:4-sparse
unimportant branches without Python stream overlap when the route is a single
fixed block. The bs64 sanity is negative:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_operator_guard_front28_serialpacked_bs64_max256_20260702
```

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2631.511 | 2310.462 | 0.878x | 3301.062 | 2921.504 | 0.885x |

The trace still shows only 49 `use_mixed_single_block` steps, 107 grouped
`operator_unimplemented` fallbacks, and lower SR24 GPU utilization
(`77.2%` vs dense `93.7%`). This confirms that merely allowing the existing
mixed branch is not enough; the viable path is a real grouped queue or a fused
packed MLP operator.

To make that decision reproducible, I added:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/scripts/analyze_sr24_grouped_operator_need.py
```

Running it on the same bs64 serial-packed grouping trace writes:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouped_operator_need_serialpacked_bs64_20260702
```

The grouped fallback rows have an optimistic local operator-only speedup of
`1.241x`, but policy/capacity selection cannot expose most of it: with
`near_full_tolerance=8`, only `29.8%` of the operator-unimplemented row weight
is near the target effective batch; the remaining `70.2%` requires a real
cross-step grouped verifier queue or a larger fused packed MLP shape. I also
updated new grouping traces to include `compact_tensor_blocks`,
`active_request_verifier_blocks`, and `target_effective_*` fields, so the trace
separates one scheduled compact tensor from the number of request rows inside
it.

I then refreshed the prefix2/front28 packed policy so bs8 is not permanently
classified as dense fallback. The config now has bs8
`dense_fallback_until_grouped` with `min_grouped_verifier_blocks=8`,
`target_effective_batch_size=64`, `grouped_dense_rows=192`, and
`grouped_base_rows=384`:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/configs/sr24_scheduler_policy_prefix2_front28_packed_k8_effective64.json
```

The supporting microbench planner is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_grouped_bucket_k8_bs8_64_prefix12_20260701/operator_planner.md
```

I also extended `analyze_sr24_grouping_queue_trace.py` with `--policy-path`,
so old traces can be re-read under the refreshed operator-local policy. Output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_queue_need_front28_policyoverride_bs8_64_20260702
```

Minimum wait budget to reach `1.2x` local MLP speedup in this optimistic queue
model:

| bs | min wait blocks | local MLP speedup | grouped block % | grouped row % |
|---:|---:|---:|---:|---:|
| 8 | 15 | 1.227x | 95.7 | 99.0 |
| 16 | 7 | 1.204x | 88.3 | 96.7 |
| 32 | 15 | 1.203x | 87.7 | 99.2 |
| 64 | 15 | 1.218x | 92.4 | 99.7 |

This is an upper-bound design signal, not an end-to-end speed claim. The next
live implementation needs a bounded verifier-block queue with dense fallback
for timeout/tail blocks, plus the fused/packed operator itself.

I then extended `analyze_sr24_grouping_queue_trace.py` again so the analysis
writes a concrete offline replay plan:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_grouping_queue_live_plan_front28_policyoverride_20260702
```

New files:

- `queue_plan.csv`
- `queue_plan.jsonl`
- `queue_plan_summary.csv`

Each plan row is one `group` or `fallback` decision with the compatible policy
key, verifier block indices, dense/base row counts, target rows, fill ratios,
wait budget, and reason (`target_reached`, `timeout_underfilled`, or
`tail_underfilled`). This is still trace-local, but it is now the right artifact
to drive a live scheduler implementation.

The refreshed plan confirms the same queue requirement:

| bs | no-wait local MLP speedup | min wait for >=1.2x | speedup at min wait | grouped block % | grouped row % |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.000x | 15 | 1.227x | 95.7 | 99.0 |
| 16 | 1.000x | 7 | 1.204x | 88.3 | 96.7 |
| 32 | 1.000x | 15 | 1.203x | 87.7 | 99.2 |
| 64 | 1.071x | 15 | 1.218x | 92.4 | 99.7 |

Plan-summary read: at bs64, `max_wait=0` can group 67 single-block events but
overall reaches only `1.071x` local MLP because 129 timeout fallbacks remain.
At bs8/16/32, no-wait grouping is effectively absent. Therefore the live path
must implement dependency-safe bounded waiting plus dense fallback; otherwise
the current fixed-block operator cannot turn the 2:4 row split into the user's
`1.2x` serving target.

Code update: `vllm/vllm/speclink_sr24.py` now has a default-off live shadow
queue controlled by:

```text
SPECLINK_SR24_GROUPED_QUEUE_SHADOW=1
SPECLINK_SR24_GROUPED_QUEUE_MAX_WAIT_BLOCKS=15
```

It consumes `sr24_grouping_opportunity` events and writes
`sr24_grouped_queue_shadow_decision` events into the same grouping trace. A
no-GPU helper smoke with three compatible blocks and `max_wait=1` produced one
`target_reached` group of two blocks and one `tail_underfilled` fallback. This
is a replay/parity scaffold only; it does not delay verifier execution and is
not a throughput optimization until a real scheduler queue and grouped/fused
operator are added.

Small code cleanup from this pass:

- `row_routed_mlp_min_dense_rows()` now accepts `0`, matching the
  operator-guard preset's "no dense-fill promotion" semantics.
- SR24 stats now include `fixed_prefix_route_descriptor_only`, so reports can
  distinguish descriptor-only route-table runs from older row-list routes.

## Fixed-Block Assembly / Output-Buffer Ablation

I added one more low-risk systems ablation before moving on to the real grouped
queue/operator work:

- `_triton_fixed_block_assemble()` can now write into an optional preallocated
  output tensor.
- Fixed-block MLP and fixed-block down-proj routes use that workspace when both
  `SPECLINK_SR24_TRITON_ROUTE_ASSEMBLY=1` and
  `SPECLINK_SR24_FIXED_BLOCK_OUTPUT_BUFFER=1` are enabled.
- `scripts/run_sr24_lossy_speed_quality_sweep.py` now has paired front28 policy
  candidates:
  - `lossy_prefix2_front28_policy_outputbuf_compile`
  - `lossy_prefix2_front28_policy_outputbuf_tritonassemble_compile`

These candidates keep the current 8pp-budget front28 semantics: no-verify MLP
rows stay dense in layers 0-27 and use 2:4 sparse only in layers 28-31;
verifier rows use the fixed-prefix2 dense-important / sparse-unimportant data
layout; near-full bs64 steps use capacity padding.

Sanity results:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front28_policy_outputbuf_bs64_fill_smoke_20260702
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front28_policy_outputbuf_tritonassemble_smoke_20260702
```

The filled bs64/max64 output-buffer smoke is negative:

| candidate | dense total | SR24 total | total speedup | dense full | SR24 full | full speedup | SR24 GPU util | graph |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| front28 policy + output buffer | 1793.562 | 1145.098 | 0.638x | 2606.371 | 1446.945 | 0.555x | 62.7% | `FULL=44, PIECEWISE=44` |

The Triton-assembly smoke with only 16 requests is also negative:

| candidate | dense total | SR24 total | total speedup | SR24 GPU util | graph |
|---|---:|---:|---:|---:|---|
| front28 policy + output buffer + Triton assembly | 739.126 | 389.651 | 0.527x | 17.2% | `FULL=44, PIECEWISE=44` |

The 16-request run is underfilled for bs64 and should not be used as a final
throughput claim. It is still useful as an underfill warning: adding another
assembly kernel/workspace option does not hide the split dense/sparse branch
cost. The filled bs64 smoke also rejects this as a mainline optimization.

Conclusion: keep the new output-buffer Triton path as an explicit diagnostic
switch, but do not pursue allocator-free assembly as the 1.2x solution. The
next implementation must change the operator boundary: a grouped verifier queue
that reaches effective bs64 shapes for bs8/16/32/64, plus a fused/packed MLP
operator that runs dense-important rows and 2:4 sparse-unimportant rows under a
single low-overhead scheduling boundary.

## Compressed Residual Triton Refresh

I rechecked the existing residual-only Triton kernel for the exact
`all_corrected_24 + compressed_dense` path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_refresh_20260702
```

The benchmark compares materialized residual weight plus torch GEMM against
the packed `_compressed_residual_matmul_kernel` for typical Llama MLP shapes.
Lower `triton/dense` is better.

| rows | projection shape | dense graph ms | best Triton graph ms | triton/dense | max abs diff |
|---:|---|---:|---:|---:|---:|
| 64 | down `4096 x 14336` | 0.0863 | 0.5198 | 6.026x | 1.000000 |
| 64 | gate_up `28672 x 4096` | 0.1545 | 0.5679 | 3.676x | 0.500000 |
| 256 | down `4096 x 14336` | 0.1694 | 1.3277 | 7.837x | 0.500000 |
| 256 | gate_up `28672 x 4096` | 0.2948 | 2.0457 | 6.938x | 0.250000 |

This rejects residual-only Triton as the `all_corrected_24` speed path. It is
both slower than materialized residual GEMM on all refreshed shapes and has
large max-absolute differences in this diagnostic. I updated the runner help
texts for `--sr24-compressed-residual-triton` to mark it as diagnostic-only,
and updated `profile_speclink_sr24_compressed_residual_kernel.py` so future
summaries print the same warning automatically.

Read: `compressed_dense` being GPU-resident is still true, but exact
`all_corrected_24` remains a two-pass computation: sparse base plus residual
correction. A useful exact path would need a fused base+residual operator, not a
separate residual-only Triton matmul.
