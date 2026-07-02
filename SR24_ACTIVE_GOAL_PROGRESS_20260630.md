# SR24 Active Goal Progress - 2026-06-30

This note tracks the current three-part goal:

1. explain when `base_only_24` is slow,
2. improve and diagnose `all_corrected_24`,
3. use the first two points to make `speclink_t08` quality-safe and at least
   `1.2x` faster than dense.

## 1. `base_only_24` Slowdown Diagnosis

Current evidence:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_baseonly_diagnosis_20260630_current/report.md
```

Read:

- `base_only_24` is not slow because accepted length collapses.
- In the graph-enabled base-only runs, accepted draft tokens per step are
  consistently no worse than dense.
- When it is slow, the issue is graph/config/operator shape, not acceptance.

Representative rows:

| run | bs | full-batch speedup | accepted/step | dense accepted/step | GPU util | graph | read |
|---|---:|---:|---:|---:|---:|---|---|
| full MLP base-only | 32 | 1.287x | 3.216 | 1.700 | 77.2% | `FULL=154,NONE=20` | fast upper bound |
| full MLP base-only | 64 | 1.863x | 3.304 | 1.672 | 88.1% | `FULL=100,NONE=2` | fast upper bound |
| gate_up 16-31 base-only | 8 | 0.979x | 2.043 | 1.678 | 93.7% | `FULL=703,NONE=56` | not acceptance-limited |
| gate_up 16-31 base-only | 16 | 0.972x | 1.946 | 1.682 | 91.8% | `FULL=378,NONE=47` | not acceptance-limited |
| gate_up 16-31 base-only | 32 | 1.077x | 1.962 | 1.702 | 88.4% | `FULL=200,NONE=30` | useful but below target |
| gate_up 16-31 base-only | 64 | 1.161x | 2.027 | 1.672 | 90.8% | `FULL=127,NONE=2` | useful but below target |

Conclusion: base-only confirms the 2:4 sparse upper bound can be real, but
narrow quality-safe scopes do not reach the `1.2x` target at bs8/16 and are
borderline at bs32/64. This points to operator/data-layout efficiency, not to
accepted-length collapse.

## 2. `all_corrected_24` Exact Path

New A/B roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_allcorrected_default_densefastpath_bs64_math64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_allcorrected_compressed_dense_nofastpath_gateup_noautoprewarm_bs64_math64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_allcorrected_torch_sparse_nofastpath_gateup_bs64_math64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_allcorrected_compressed_triton_gateup_bs64_math64_20260630
```

Setup: Llama-3.1-8B, EAGLE3 K=4, `math_reasoning`, bs64, 64 requests,
max tokens 64. The no-fastpath rows scope SR24 to `gate_up_proj`.

| path | status | full-batch speedup vs dense | total speedup vs dense | GPU util | residual device | read |
|---|---|---:|---:|---:|---|---|
| default densefastpath | ok | 1.004x | 1.001x | 60.3% | `none` | exact dense-equivalent control |
| compressed_dense chunked | ok | 0.268x | 0.300x | 87.2% | `cuda:0` | GPU-resident but slow |
| torch_sparse residual | ok | 0.358x | 0.395x | 42.6% | sparse residual tensor | eager/small-kernel limited |
| compressed Triton | ok | 0.189x | 0.169x | 48.3% | `cuda:0` | current Triton kernel is not useful |

One code fix was made in
`examples/evaluate/eval-guidellm/scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py`:
the compressed-dense auto fastpath no longer treats a leaf-only scope such as
`gate_up_proj` as narrow enough for cache+prewarm. Before the fix, that command
attempted to materialize all 32 gate/up residual dense tensors on GPU and OOMed.
After the fix, it runs as chunked GPU `compressed_dense`.

Conclusion:

- `compressed_dense` is on GPU in the tested path; it is not slow because it is
  running on CPU.
- The optimized exact `all_corrected_24` path is the densefastpath because
  all-corrected is algebraically equal to the original dense Linear.
- The sparse-base plus exact-residual operator is currently much slower than
  dense. It needs a different fused/grouped sparse operator, not just a
  compressed_dense flag change.

## 3. Compressed Residual Triton Sweep

Sweep root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_compressed_residual_triton_sweep_gateup_20260630
```

Best points:

| rows | out | in | best block_m | best block_n | best block_g | dense graph ms | Triton graph ms | Triton/dense |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 | 28672 | 4096 | 64 | 128 | 16 | 0.3434 | 1.7144 | 4.993x |
| 512 | 28672 | 4096 | 64 | 128 | 16 | 0.5419 | 2.6688 | 4.925x |

Conclusion: the current packed-values Triton residual kernel is about `5x`
slower than materialized dense residual GEMM for the gate/up shape. It should
stay a diagnostic path. A real operator effort should use cuSPARSELt/CUTLASS or
a new grouped/fused design, not incremental tuning of this kernel.

## 4. `speclink_t08` Quality Status

Current lossy gate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_ppopp_gate_bs8_64_math128_gsm8k50_20260630
```

The current row-routed/fixed-prefix candidates are not suffering a large
accuracy collapse on GSM8K-50:

| candidate | dense acc | SR24 acc | delta | pair reg/imp |
|---|---:|---:|---:|---:|
| `lossy_prefix2_rowrouted_mlp_minbase64` | 0.7400 | 0.7200 | -2pp | 3/2 |
| `lossy_prefix1_rowrouted_mlp_minbase128` | 0.7400 | 0.7200 | -2pp | 3/2 |
| `gateup_res16_25_base26_31_critical4_smallrow160` | 0.7400 | 0.7200 | -2pp | 2/1 |

The large drop came from coarse layer-level sparse-only routes. Example:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_tail16_baseonly_gsm8k50_quality_20260630
```

That path drops from `0.7200` to `0.2200` on GSM8K-50 (`-50pp`) because broad
late-layer MLP sparse-only execution changes accepted reasoning trajectories.

Conclusion: the quality-safe path is token/row-level routing with important
rows dense-only and unimportant rows sparse-only. The remaining gap is speed:
current `speclink_t08` row-routed execution is correct enough for the 8pp gate,
but not operator-efficient enough for `1.2x`.

## Next Work

1. Treat `base_only_24` as the sparse upper-bound signal, not as the final
   quality-safe method.
2. Treat default `all_corrected_24` densefastpath as the optimized exact
   control; keep no-fastpath variants as operator diagnostics only.
3. Replace the current mixed MLP implementation with a packed verifier-block
   grouped/fused operator. The first success criterion should be a standalone
   microbench: rows around `288/576`, gate/up and down shapes, at least `1.25x`
   faster than dense before live vLLM integration.
4. Then rerun `speclink_t08` GSM8K limit 50 and bs `8/16/32/64` throughput.

## 5. Packed Verifier-Block MLP and Live K Sweep

Standalone packed verifier-block microbench roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_packed_verifier_mlp_parallel_bs8_64_prefix124_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_packed_verifier_mlp_parallel_k12_bs8_64_prefix012_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_packed_verifier_mlp_parallel_k16_bs8_64_prefix012_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_packed_verifier_mlp_parallel_k24_bs8_64_prefix012_20260630
```

Key standalone read:

| K | prefix | bs16 speedup | bs32 speedup | bs64 speedup | read |
|---:|---:|---:|---:|---:|---|
| 8 | 1 | 0.716x | 0.977x | 1.198x | only bs64 near target |
| 12 | 1 | 0.880x | 1.200x | 1.235x | bs32/64 viable standalone |
| 16 | 1 | 0.979x | 1.357x | 1.333x | good standalone, more KV pressure |
| 24 | 1 | 1.196x | 1.275x | 1.331x | bs16 starts to work but K is too large |

The double-stream packed MLP confirms the right dataflow for a PPoPP-style
operator: dense-important rows and sparse-only rows should be disjoint, and the
dense and sparse branches can overlap. However, the current PyTorch/cuSPARSELt
split needs roughly hundreds of rows per branch before it beats dense. Small
batch needs either dense fallback, larger verification blocks, cross-step/layer
grouping, or a custom fused/grouped kernel.

Live serving checks:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_lossy_k24_prefix01_overlap_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_lossy_k12_prefix1_overlap_bs32_64_math_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_k12_prefix1_noverify_sparse_quality_gsm8k50_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_k12_prefix1_noverify_sparse_throughput_bs32_64_math_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_k12_prefix1_sparse_non_draft_quality_gsm8k50_20260630
```

Results:

| config | quality | bs32 total speedup | bs64 total speedup | read |
|---|---:|---:|---:|---|
| K24 prefix1 overlap | +2pp on GSM8K-50 | 0.700x | failed bs64 | K too large; EAGLE3 acceptance about 6% |
| K12 prefix1 overlap | not rerun in this root | 0.650x | 0.650x | standalone MLP gain does not survive serving |
| K12 prefix1 noverify sparse | -6pp on GSM8K-50 | 0.544x | 0.650x | quality within 8pp, throughput still negative |
| K12 prefix1 sparse non-draft | -22pp on GSM8K-50 | skipped | skipped | too much quality loss |

Breakdown roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_k12_prefix1_overlap_breakdown_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_k12_prefix1_noverify_sparse_breakdown_bs64_20260630
```

Important breakdown facts:

- In the default K12 prefix1 overlap path, scheduler CPU work is not the main
  issue: `scheduler_mask_build_cpu_ms` is about `21ms` for the short breakdown
  run. The large terms are MLP kernels.
- `noverify_dense_mlp_gate_up_cuda_ms` and `noverify_dense_mlp_down_cuda_ms`
  dominate the default path, so simply routing verifier rows is not enough.
- Turning off `noverify_dense_mlp_fastpath` removes those names, but
  `full_residual_early_dense_rows` becomes huge. This is caused by
  `selective_correct_non_draft=1`, which intentionally keeps no-residual
  non-draft rows dense for quality.
- Turning off `selective_correct_non_draft` removes that dense protection but
  drops GSM8K-50 by `22pp`, outside the allowed 8pp budget.

Conclusion: the viable quality boundary is not "make every non-draft/noverify
row sparse." The current safe boundary is closer to "selectively sparse
non-critical verifier rows while keeping enough non-draft rows dense." The speed
target still needs a real fused/grouped operator or a finer non-draft controller;
the present Python/PyTorch split path is not enough for `1.2x` serving speedup.

## 6. Layer-Scoped Noverify Dense Guard and Compile Ablation

Implemented a layer-scoped noverify dense guard:

```text
SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_LAYER_IDS_BY_LEAF
--sr24-selective-dense-nonverify-layer-ids-by-leaf
```

Empty scope preserves the old behavior. A scope such as
`gate_up_proj=0-7;down_proj=0-7` keeps no-mask/noverify MLPs dense only in the
listed layers; other noverify rows use the 2:4 sparse base, so unimportant rows
are not computed once sparse and then recomputed dense.

Small quality gates:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front8_compile_quality_throughput_bs32_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front8_compile_minbase0_quality_throughput_bs32_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/examples/evaluate/eval-guidellm/results.bak/sr24_scoped_noverify_dense_front16_24_q50_bs32_64_20260630
```

GSM8K-50 results:

| config | dense acc | SR24 acc | delta | read |
|---|---:|---:|---:|---|
| front24 noverify dense scope | 0.7400 | 0.7200 | -2pp | within 8pp |
| front16 noverify dense scope | 0.7400 | 0.7400 | 0pp | within 8pp |
| front8 noverify dense scope + default compile | 0.7400 | 0.7400 | 0pp | within 8pp |
| front8 + `route_min_base_rows=0` + default compile | 0.7400 | 0.7400 | 0pp | within 8pp |

Throughput checks:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_scoped_noverify_dense_front16_24_throughput_bs32_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front16_default_compile_throughput_bs32_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front8_compile_quality_throughput_bs32_64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front8_compile_minbase0_quality_throughput_bs32_64_20260630
```

| config | bs32 speedup | bs64 speedup | read |
|---|---:|---:|---|
| front24, compile off | 0.835x | 0.965x | too much hook/eager overhead |
| front16, compile off | 0.837x | 0.945x | same bottleneck |
| front16, default compile on | 0.997x | 0.997x | compile removes most hook overhead |
| front8, default compile on | 0.997x | 0.997x | more sparse noverify layers do not add speed |
| front8, minbase0, default compile on | 0.992x | 1.000x | tiny verifier sparse branches do not help |

The final noverify boundary tested `scope=none`, so all no-mask/noverify MLP
rows use only the 2:4 sparse base while verifier prefix/bonus rows keep the
selective dense protection. This required a small parser fix so `none` is a
valid special value in
`SPECLINK_SR24_SELECTIVE_DENSE_NONVERIFY_LAYER_IDS_BY_LEAF`.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_compile_quality_throughput_bs32_64_retry_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_compile_throughput_bs32_64_max512_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_compile_throughput_bs8_16_max512_20260630
```

GSM8K-50 quality remains safe: dense `0.7400`, SR24 `0.7400`, `0pp` delta.
Throughput:

| max tokens | bs8 | bs16 | bs32 | bs64 | read |
|---:|---:|---:|---:|---:|---|
| 128 | n/a | n/a | 0.985x | 1.012x | short output still parity |
| 512 | 0.960x | 0.992x | 1.000x | 1.239x | large batch and longer output finally exceed 1.2x |

This is the first quality-safe live configuration in this pass that exceeds
`1.2x`, but only at `bs64/max512/math_reasoning`. It does not satisfy the goal
for most batch sizes. The bottleneck for bs8/16/32 is still branch underfill and
split sparse operator overhead, not quality loss.

Diagnostic breakdown:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front16_instrumented_breakdown_bs32_math_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_front16_instrumented_breakdown_summary_bs32_math_20260630
```

Key facts from the diagnostic row:

- front16 scope is active: noverify dense MLP calls appear only in layers 0-15,
  and 2:4 base sparse calls appear in layers 16-31.
- Linear CUDA-event timing is diagnostic and sync-heavy, but it localizes the
  issue: tail-layer base sparse linears cost about `1.172 ms/call` eager, while
  the dense noverify MLP linears in the first 16 layers cost about
  `0.645 ms/call` for gate/up and `0.343 ms/call` for down.
- Scheduler exact diagnostic time is about `17 ms/step`, but clean compile-on
  runs recover to about parity, so the first speed bottleneck is still operator
  and graph integration, not accepted-length collapse.

Sparse backend probe:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_sparse_backend_probe_rows512_640_llama_mlp_20260630
```

For representative Llama MLP shapes, PyTorch semi-structured sparse eager calls
are slower than dense, but CUDA Graph replay can make the base sparse GEMM
faster than dense for a single base-only Linear. Exact all-corrected paths
remain slower than dense unless a fused packed kernel removes the separate
base/residual passes. This explains the serving behavior: vLLM default compile
is necessary to approach parity, but current split sparse/dense execution still
does not reach the `1.2x` target.

Conclusion: within the 8pp quality budget, noverify sparse is quality-safe, and
default compile is mandatory. Removing all noverify dense work can cross `1.2x`
only for the best large-batch/long-output case. The next implementation should
focus on a graph-stable fixed-capacity route table plus fused/grouped mixed MLP
operator:

1. important rows dense-only, unimportant rows 2:4-only;
2. no duplicated sparse+dense work for the same row;
3. route metadata and buckets resident on GPU;
4. small important-row sets coalesced across requests/layers or forced to dense
   fallback;
5. dense and sparse streams overlapped only after each branch has enough rows to
   occupy the GPU.

## 7. K12 and Stream-Overlap Boundary Checks

Two follow-up checks tested whether the remaining slowdown is simply caused by
too few sparse/dense rows per branch.

First, K was raised from 8 to 12 for the noverify-sparse compile boundary:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_compile_k12_quality_throughput_bs16_64_max512_20260630
```

Quality remains safe on GSM8K-50: dense `0.7400`, SR24 `0.7400`, `0pp` delta.
Throughput on `math_reasoning`, max512:

| K | bs16 total | bs16 full-batch | bs32 total | bs32 full-batch | bs64 total | bs64 full-batch | read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 1.101x | 1.008x | 1.062x | 1.014x | 0.978x | 0.979x | larger K improves drain-sensitive totals at bs16/32 but does not improve core full-batch speed |

Dense EAGLE3 K12 accepts only about `2.43-2.59` draft tokens/step on this
workload, so increasing K mostly increases verifier work rather than filling
useful accepted-token work.

Second, a new candidate was added:

```text
lossy_prefix2_rowrouted_mlp_noverify_sparse_overlap_compile
```

It combines `scope=none`, default vLLM compile, `route_min_base_rows=128`, and
`route_overlap_streams=1` so fixed-block dense-important rows and 2:4
sparse-unimportant rows can run on separate CUDA streams. Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_overlap_compile_k8_quality_throughput_bs16_64_max512_20260630
```

Quality again remains safe on GSM8K-50: dense `0.7400`, SR24 `0.7400`, `0pp`.
Throughput on `math_reasoning`, K8, max512:

| K | bs16 total | bs16 full-batch | bs32 total | bs32 full-batch | bs64 total | bs64 full-batch | read |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 + overlap | 0.999x | 0.982x | 0.984x | 0.998x | 0.950x | 0.948x | direct stream overlap does not help in the current Python/PyTorch split path |

The stats confirm `route_overlap_streams=True`,
`route_contiguous_fastpath=True`, `fixed_prefix_route_fastpath=True`, and
`selective_dense_nonverify_scope=none`. The negative result is therefore not
because overlap was disabled; the current overlap path still pays split-branch,
auxiliary-stream, and graph-shape overhead. The next useful implementation is a
fused/grouped mixed MLP operator or graph-stable fixed-capacity route-table
kernel, not another K-only or stream-flag sweep.

## 8. Triton Fixed-Block Assemble Ablation

I added an opt-in fixed-block Triton assemble path for the row-routed MLP output:

```text
vllm/vllm/speclink_sr24.py
lossy_prefix2_rowrouted_mlp_noverify_sparse_tritonassemble_compile
```

The intent was to remove the three Python/PyTorch slice copies in the common
fixed `[prefix dense rows, sparse middle rows, verifier bonus dense row]` layout.
The new kernel maps each output row directly to dense-prefix, sparse-middle, or
dense-bonus storage. It is gated by `--sr24-triton-route-assembly` and is not
enabled by default.

Smoke result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_tritonassemble_smoke_bs64_20260630
```

The smoke confirms the path is active and quality-safe at the small gate:
GSM8K-10 dense `0.8000`, SR24 `0.8000`.

Focused result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_tritonassemble_quality_throughput_bs32_64_max512_20260630
```

GSM8K-50 remains safe: dense `0.7400`, SR24 `0.7400`, `0pp` delta. Throughput on
`math_reasoning`, K8, max512:

| variant | bs32 total | bs32 full-batch | bs64 total | bs64 full-batch | read |
|---|---:|---:|---:|---:|---|
| no-Triton assemble | 1.000x | 0.998x | 1.239x | 0.956x | current best large-batch boundary |
| Triton assemble | 1.009x | 0.998x | ~0.960x | ~0.961x | negative; bs64 summary row did not enter the generated median table, but raw 64/64 stream records completed and give ~2364.5 total tok/s vs dense 2462.8 |

The bs64 Triton run produced `64` successful stream records with `32768` output
tokens, so the missing generated summary row is a reporting issue, not a serving
failure. Server logs show normal health, requests, metrics, and shutdown. The
result still argues against promoting the Triton assemble path: it removes small
slice copies but adds another custom kernel and does not address the dominant
sparse-base MLP cost.

Conclusion: keep `--sr24-triton-route-assembly` only as an explicit negative
ablation. The default path should remain the no-Triton
`lossy_prefix2_rowrouted_mlp_noverify_sparse_compile` candidate until a real
fused/grouped MLP operator replaces the split sparse/dense execution.

## 9. Low-Row Sparse Branch Boundary

I added and ran:

```text
lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_minbase0
```

This keeps the same quality controller as the current noverify-sparse boundary
but changes `route_min_base_rows` from `128` to `0`. The hypothesis was that
low-batch cases might be slow because the sparse-unimportant branch falls back
instead of running 2:4 when it has few rows.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_compile_minbase0_quality_throughput_bs8_64_max512_20260630
```

GSM8K-50 remains safe: dense `0.7400`, SR24 `0.7400`, `0pp` delta. Throughput
on `math_reasoning`, K8, max512:

| bs | total speedup | full-batch speedup | read |
|---:|---:|---:|---|
| 8 | 1.016x | 1.012x | small improvement over the minbase128 bs8 run, but far below target |
| 16 | 0.954x | 1.011x | total throughput regresses |
| 32 | 1.036x | 0.997x | small total gain, no saturated gain |
| 64 | 0.741x | 0.946x | large regression versus minbase128, where bs64 total was 1.239x |

Conclusion: forcing the sparse-unimportant branch for very small row counts is
not the missing low-batch fix. The `route_min_base_rows=128` guard should stay
for the current split PyTorch/cuSPARSELt path. This supports the broader
operator conclusion: the next useful optimization is not a scalar row-threshold
sweep, but a fixed-capacity grouped/fused MLP data path that can coalesce enough
rows to fill the sparse branch without hurting large-batch serving.

## 10. Actual K8 Row-Shape and cuSPARSELt alg_id Probe

I added:

```text
examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_cslt_algos.py
```

and extended:

```text
examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_row_routed_mlp.py
```

The actual K=8 verifier MLP row shapes are approximately
`rows=batch_size*(K+1)` and the fixed prefix2+bonus dense group is
`3*batch_size`. The microbenchmark result roots are:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_actual_rows_bs8_k8_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_actual_rows_bs16_k8_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_actual_rows_bs32_k8_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_row_routed_mlp_actual_rows_bs64_k8_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_cslt_algos_actual_rows_k8_20260630
```

Key lower-bound operator numbers:

| batch | rows | dense rows | dense graph ms | dense/1.2 target | full sparse graph ms | exact-down contiguous graph ms | exact-down/dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 72 | 24 | 0.2684 | 0.2236 | 0.2935 | 0.5271 | 1.964x |
| 16 | 144 | 48 | 0.3228 | 0.2690 | 0.3554 | 0.5391 | 1.670x |
| 32 | 288 | 96 | 0.5305 | 0.4421 | 0.4315 | 0.6461 | 1.218x |
| 64 | 576 | 192 | 0.9579 | 0.7982 | 0.6174 | 0.7966 | 0.832x |

This is the clearest current blocker: for bs8/16, even the ideal full sparse
MLP is slower than dense with default cuSPARSELt alg0. Row routing cannot
produce a 1.2x end-to-end speedup when the base sparse operator lower bound is
already negative.

The cuSPARSELt alg sweep found only alg0 and alg1 valid. alg1 helps the
bs8-shaped full sparse MLP but hurts larger rows:

| rows | best alg | sparse speedup vs dense |
|---:|---:|---:|
| 72 | 1 | 1.152x |
| 144 | 1 | 1.029x |
| 288 | 0 | 1.240x |
| 576 | 0 | 1.553x |

I implemented an opt-in live switch:

```text
SPECLINK_SR24_CSLT_SMALL_M_ALG_ID_ENABLE=1
SPECLINK_SR24_CSLT_SMALL_M_THRESHOLD=96
SPECLINK_SR24_CSLT_SMALL_M_ALG_ID=1
```

The corresponding runner flags are:

```text
--sr24-cslt-small-m-alg-id-enable
--sr24-cslt-small-m-threshold 96
--sr24-cslt-small-m-alg-id 1
```

This is an operator-selection ablation, not a final default.

## 11. Live Small-M alg1 and K16 Probes

I added the candidate:

```text
lossy_prefix2_rowrouted_mlp_noverify_sparse_compile_smallm_alg1
```

Quality/throughput result:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_smallm_alg1_quality_throughput_bs8_64_max512_20260630
```

GSM8K-50 passed with no observed accuracy delta: dense `0.7400`, SR24 `0.7400`.
Throughput on `math_reasoning`, K8, max512:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 1.015x | 0.993x |
| 16 | 1.002x | 1.008x |
| 32 | 0.990x | 0.997x |
| 64 | 0.963x | 0.947x |

I also tested K16 throughput only:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_smallm_alg1_k16_throughput_bs8_64_max512_20260630
```

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 1.010x | 0.998x |
| 16 | 0.989x | 1.005x |
| 32 | 0.926x | 0.974x |
| 64 | 0.968x | 0.963x |

Conclusion: adaptive alg1 is a useful diagnostic and slightly improves bs8,
but it is not the missing 1.2x path. Increasing K from 8 to 16 also does not
solve the fill problem; lower acceptance and larger verification work dominate.
The next viable path needs a real small-M 2:4 operator/data-layout change or a
serving design that can coalesce sparse rows without increasing verifier work.

## 12. Triton Base 2:4 Prototype and Triton Assembly Gate

I added a standalone scalar Triton base-2:4 MLP prototype:

```text
examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_triton_base24_mlp.py
```

It uses a GPU-resident format:

```text
values[out, group, 2]
k0[out, group]
k1[out, group]
```

and computes the 2:4 base linear by gathering two input channels per group.
This validates the data-layout idea but intentionally does not use sparse
tensor cores. Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_base24_mlp_actual_rows_k8_20260630
```

Key result:

| rows | best Triton base24 graph ms | dense graph ms | cuSPARSELt best graph ms | Triton speedup |
|---:|---:|---:|---:|---:|
| 72 | 7.9858 | 0.2668 | 0.2343 | 0.033x |
| 144 | 15.2555 | 0.3258 | 0.3373 | 0.021x |
| 288 | 30.5315 | 0.5336 | 0.4427 | 0.017x |
| 576 | 60.5686 | 0.9596 | 0.6453 | 0.016x |

Conclusion: a scalar Triton gather-multiply sparse operator is not viable.
The sparse path must stay tensor-core-backed, either via cuSPARSELt/CUTLASS or
a new CUDA kernel with an equivalent compressed sparse tensor-core layout.

I also ran the live Triton fixed-block output-assembly candidate:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_tritonassemble_quality_throughput_bs8_64_20260630
```

Setup:

- Llama-3.1-8B, EAGLE3 K=8
- GSM8K-CoT quality, limit 50, max new tokens 512
- `math_reasoning` throughput, bs `8/16/32/64`, max tokens 128
- candidate: `lossy_prefix2_rowrouted_mlp_noverify_sparse_tritonassemble_compile`

Quality is safe: dense `0.7400`, SR24 `0.7400`, `0pp` delta.

Throughput:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1017.191 | 1013.300 | 0.996x | 1059.481 | 1055.388 | 0.996x |
| 16 | 1623.761 | 1623.867 | 1.000x | 1717.092 | 1721.381 | 1.002x |
| 32 | 2079.998 | 2091.024 | 1.005x | 2280.875 | 2281.426 | 1.000x |
| 64 | 2367.243 | 2371.709 | 1.002x | 2789.705 | 2759.654 | 0.989x |

This removes most of the negative overhead from the output assembly, but it
does not create a real speedup. The dominant term remains the small-M
cuSPARSELt sparse-base MLP plus split-branch launch structure. The next useful
implementation work is a tensor-core-backed grouped/fused mixed MLP operator,
not more Python-side route/list/assembly tuning.

## 13. Gate-Up-Only Noverify-Sparse Probe

I added a focused sweep candidate:

```text
lossy_prefix2_gateup_only_noverify_sparse_compile
```

It reuses the fixed-prefix noverify-sparse policy, but overrides SR24 scope to
`gate_up_proj` only. The down projection stays on the original dense vLLM path.
This tests whether removing the small-M sparse down branch is enough to make
the lossy path profitable.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_noverify_sparse_quality_throughput_bs8_64_20260630
```

Setup:

- Llama-3.1-8B, EAGLE3 K=8
- GSM8K-CoT quality, limit 50, max new tokens 512
- `math_reasoning` throughput, bs `8/16/32/64`, max tokens 128
- noverify rows are sparse-only; verifier prefix/bonus rows are dense

Quality is safe: dense `0.7400`, SR24 `0.7400`, `0pp` delta.

Throughput:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1014.929 | 1012.037 | 0.997x | 1061.893 | 1058.849 | 0.997x |
| 16 | 1618.916 | 1627.940 | 1.006x | 1720.976 | 1722.113 | 1.001x |
| 32 | 2027.130 | 2070.131 | 1.021x | 2285.844 | 2277.375 | 0.996x |
| 64 | 2372.864 | 2374.456 | 1.001x | 2763.542 | 2778.025 | 1.005x |

Conclusion: the sparse down branch is not the only blocker. Removing it makes
the candidate quality-safe and slightly improves total throughput at bs32, but
still does not approach `1.2x`. The remaining bottleneck is the gate/up sparse
branch at small M plus the launch/routing structure. This further supports a
grouped/fused tensor-core sparse operator as the main path.

## 14. Channel-Pair Gate/Up Split Check

I added two channel-split candidates:

```text
gateup_channel_pair_dense25_eager
gateup_channel_pair_dense50_eager
```

They split the gate/up intermediate dimension instead of verifier rows. The
highest-norm 25% or 50% intermediate channels stay dense, the rest use a
full-row 2:4 sparse branch, and `down_proj` stays dense. This was meant to
avoid tiny-M row-routed sparse GEMMs while keeping dense and sparse work
disjoint.

The default-compile run failed at server start while Torch compiled the
channel-pair path and traced a lazily materialized grouped down weight:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_quality_only_20260630
```

The eager quality-only rerun completed:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_quality_only_20260630_eager
```

GSM8K-50 result:

| candidate | dense acc | SR24 acc | delta | pair reg/imp | read |
|---|---:|---:|---:|---:|---|
| channel dense 25% | 0.7200 | 0.5000 | -22pp | 14/3 | fails 8pp budget |
| channel dense 50% | 0.7200 | 0.5600 | -16pp | 10/2 | fails 8pp budget |

Conclusion: full-layer gate/up channel splitting is too accuracy-sensitive for
this workload. It also has a compile-safety issue in the current lazy grouped
down-weight implementation. Do not continue this path unless a new sensitivity
analysis scopes channel splitting to a much smaller set of layers/channels.

## 15. Minbase256 and Prefix1 Boundary Checks

The latest row-routed probes tested whether the quality budget can be spent by
protecting fewer rows or avoiding underfilled sparse splits.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_minbase256_gate_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_noverify_sparse_minbase256_gate_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_noverify_sparse_gate_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_gateup_only_noverify_sparse_bs64_max512_20260630
```

GSM8K-50 and bs64 `math_reasoning` results:

| candidate | quality | max tokens | full-batch speedup | total speedup | read |
|---|---:|---:|---:|---:|---|
| all-MLP noverify sparse, minbase256 | `0.7400 -> 0.7400` | 128 | 1.0005x | 0.9892x | safe but no speedup |
| gate_up-only noverify sparse, minbase256 | `0.7400 -> 0.7400` | 128 | 0.9934x | 1.0060x | safe but parity |
| gate_up-only prefix1 noverify sparse | `0.7400 -> 0.7400` | 128 | 0.9990x | 1.0115x | fewer dense rows still parity |
| all-MLP prefix1 noverify sparse | `0.7400 -> 0.7400` | 128 | 0.9914x | 0.9982x | broader sparse scope still parity |
| gate_up-only prefix1 noverify sparse | quality skipped in root | 512 | 0.9992x | 1.0004x | long-output sanity still parity |

Conclusion: the current path is not blocked by an overly strict lossless
requirement. Even when we protect only draft position 0 plus the verifier bonus
row, the live serving path stays around parity. The execution path needs a
better operator/data layout.

## 16. Gate-Up-Only Noverify-Sparse Breakdown

Latest breakdown root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_noverify_sparse_breakdown_bs64_math_manual_20260630
```

Clean bs64 `math_reasoning`, K8, max128:

| method | total tok/s | full-batch tok/s | avg GPU util | CUDA Graph |
|---|---:|---:|---:|---|
| dense EAGLE3 | 2377.825 | 2768.973 | 88.9% | `FULL=35,NONE=92,PIECEWISE=1` |
| gate_up-only SR24 | 2325.595 | 2784.583 | 92.1% | `FULL=96,NONE=32` |

Diagnostic row-routed gate/up timing:

| item | value |
|---|---:|
| total row-routed gate/up | 1.131 ms/call |
| sparse base | 0.936 ms/call |
| dense gather + dense GEMM | about 0.168 ms/call |
| assembly | 0.023 ms/call |
| rows per call | 256.8 total, 171.2 base, 85.6 dense |
| draft residual/base rows | 2000 / 6000 |
| non-draft residual rows | 1000 |
| correction row fraction | 0.333 |

The diagnostic scheduler/request-loop timing is sync-heavy and should not be
used as clean throughput, but it shows the CPU routing path is still not free:
exact diagnostic `scheduler_mask_wall_cpu_ms_per_step` is `5.181 ms`, almost
all of it from the per-request routing loop.

The component microbench in the same root explains why the clean run remains
parity. For gate/up shape `512x28672x4096`, full-row sparse base is faster than
dense (`0.353 ms` vs `0.539 ms`), but mixed sparse+dense routing at realistic
residual fractions costs about dense or worse. For the down shape, low residual
fractions can be better, but removing sparse down already failed to deliver
speedup, so the remaining gate/up branch is sufficient to erase the gain.

Conclusion: output length, final assembly, and duplicate dense recompute are
not the main remaining bottleneck. The blocker is small-M sparse MLP efficiency
and split-branch launch structure. The next experiment should be a
fixed-capacity grouped/fused operator microbench, not another threshold sweep.

## Updated Next Work

The active target is unchanged: reach about `1.2x` over dense EAGLE3 for most
bs `8/16/32/64` and datasets under an `8pp` GSM8K-style accuracy budget. The
evidence now says the next useful work is:

1. fixed-capacity route descriptors, resident on GPU and compatible with CUDA
   Graph capture;
2. disjoint dense-important rows and 2:4-sparse-unimportant rows, with no
   sparse+dense duplicate work for a row;
3. coalescing across requests, steps, or layers so the sparse side has enough
   rows to use tensor cores efficiently;
4. dense fallback when the sparse side is underfilled;
5. optional dense/sparse overlap only inside a captured grouped/fused operator.

The immediate microbench gate should require the mixed MLP path to beat dense
by at least `1.25x` at actual K8 row counts around `288` and `576` before
another live vLLM matrix is worth running.

## 17. Fixed-Capacity Fill-Aware Packed MLP Probe

I extended the packed verifier-block microbenchmark:

```text
examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_packed_verifier_mlp.py
```

New flags:

```text
--capacity-multiple
--min-dense-capacity
--min-base-capacity
```

The new `packed_capacity_padded` path copies active dense-important and
sparse-only rows into fixed-capacity branch buffers before the MLP GEMMs. This
directly tests a possible systems fix for underfilled branches: keep the route
descriptor CUDA-graph stable and pad branch row counts to a tensor-core-friendly
capacity.

Result roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_fixed_capacity_fill_k8_bs8_64_prefix12_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_fixed_capacity_fill64_k8_bs8_64_prefix12_20260701
```

K8, prefix1/2, Llama MLP shape, graph-captured speedup versus dense:

| capacity multiple | bs | prefix | packed | parallel packed | padded | dense fill | sparse fill | read |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 8 | 1 | 0.482x | 0.604x | 0.454x | 0.125 | 0.438 | padding much worse |
| 128 | 16 | 1 | 0.593x | 0.716x | 0.553x | 0.250 | 0.875 | padding worse |
| 128 | 32 | 1 | 0.851x | 0.977x | 0.810x | 0.500 | 0.875 | padding worse |
| 128 | 64 | 2 | 1.199x | 1.240x | 1.027x | 0.750 | 1.000 | dense padding hurts |
| 64 | 8 | 1 | 0.483x | 0.605x | 0.492x | 0.250 | 0.875 | slightly better, still far below dense |
| 64 | 16 | 1 | 0.592x | 0.716x | 0.587x | 0.500 | 0.875 | parity with unpadded, still bad |
| 64 | 32 | 1 | 0.853x | 0.979x | 0.857x | 1.000 | 0.875 | no real gain |
| 64 | 64 | 2 | 1.198x | 1.241x | 1.199x | 1.000 | 1.000 | works only when no padding is needed |

Conclusion: fixed-capacity route descriptors are still the right live-serving
shape, but padding alone is not the low-batch speed solution. It wastes too much
dense-side work and does not make the sparse branch fast enough at bs8/16/32.
The only standalone row where the mixed path clearly beats dense is bs64, where
the branch row counts are naturally full. The next implementation should use
fixed-capacity descriptors for graph stability, but fill them with real useful
rows by coalescing or grouping; if a branch cannot be filled with useful rows,
dense fallback is better than padding dummy rows.

## 18. Useful-Row Coalescing Upper Bound

I added `--coalesce-factors` to the same packed verifier-block MLP
microbenchmark. A coalesce factor groups several independent verifier blocks
with the same weights into one larger dense/sparse branch. The benchmark reports
two distinct quantities:

- speedup versus one coalesced dense GEMM at the same effective batch;
- speedup versus running the original batch-size dense block serially for each
  coalesced block.

This separates pure dense launch/shape coalescing from the extra value of the
mixed 2:4 path.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_useful_coalesce_k8_bs8_32_prefix12_rerun_20260701
```

Planner files derived from the same summary:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_useful_coalesce_k8_bs8_32_prefix12_rerun_20260701/operator_planner.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_useful_coalesce_k8_bs8_32_prefix12_rerun_20260701/operator_planner_grouping.csv
```

Key K8/prefix2 graph-captured results:

| original bs | coalesce | effective bs | mixed vs coalesced dense | mixed vs serial dense | read |
|---:|---:|---:|---:|---:|---|
| 8 | 1 | 8 | 0.601x | 0.599x | too small |
| 8 | 4 | 32 | 0.945x | 1.899x | serial launch savings dominate |
| 8 | 8 | 64 | 1.242x | 2.768x | finally passes standalone target |
| 16 | 2 | 32 | 0.949x | 1.157x | still below coalesced dense |
| 16 | 4 | 64 | 1.240x | 1.680x | passes at effective bs64 |
| 32 | 1 | 32 | 0.948x | 0.948x | slightly slower than dense |
| 32 | 2 | 64 | 1.241x | 1.381x | passes at effective bs64 |

Read:

- The current mixed MLP begins to beat dense once useful rows reach effective
  bs around 64.
- Dummy padding cannot create that benefit; the extra rows must be useful work.
- Dense coalescing itself gives large speedups versus serial small blocks, so a
  fair live design must compare against whatever dense batching the scheduler
  could also exploit.
- For live speculative decoding, coalescing across future decode iterations is
  not automatically legal because of autoregressive dependencies. Treat this as
  an operator/scheduler upper bound, not a completed serving optimization.

Updated live-path implication:

1. Underfilled SR24 branches should fall back to dense rather than run a tiny
   sparse branch.
2. A real speed path needs grouped useful work: multiple ready request groups,
   cross-microbatch grouping, or a grouped operator over independent work.
3. The first live target should be bs32/64, where only coalesce 2/1-like fill is
   needed. bs8/16 need much more grouping or a different small-M sparse kernel.

## 19. Fixed-Block Dense-Fill Live Serving Probe

The latest serving probe tests the user's "important tokens may be dense, but
unimportant tokens should not pay dense correction after sparse work" direction
with an explicit fill-factor ablation. The clean row-routed policy is still
disjoint: important prefix+bonus rows are dense-only, remaining verifier rows
are 2:4 sparse-only. The new flag
`SPECLINK_SR24_ROW_ROUTED_MLP_FIXED_BLOCK_DENSE_FILL=1` intentionally promotes
some adjacent low-priority rows into dense only to test Tensor Core fill; it is
not the default semantics.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_densefill64_128_quality50_speed_b8_64_math512_20260701
```

Both dense-fill64 and dense-fill128 pass GSM8K-50 with no measured regression
(`0.7400 -> 0.7400`). Throughput remains far below the goal:

| candidate | best total speedup | best full-batch speedup | failure mode |
|---|---:|---:|---|
| dense-fill64 | 1.026x | 1.035x | only small bs16/32 gains; bs64 regresses |
| dense-fill128 | 1.037x | 1.001x | dense overfill hurts high batch |

An overlap-stream serving probe with dense-fill64 is also negative under the
same math_reasoning max512 setup: bs32 total/full-batch speedups are
`0.997x/0.979x`, and bs64 total/full-batch speedups are `0.950x/0.949x`.
The current Python-side stream split and row-routed PyTorch/cuSPARSELt
composition should not be promoted.

Updated active read:

1. The quality budget is not the immediate blocker; GSM8K-50 can tolerate the
   prefix2 row-routed policy and even the dense-fill probe.
2. The speed blocker is still operator efficiency: fragmented sparse-base MLP,
   dense branch overfill, and assembly/launch overhead.
3. The next real implementation step should be a fixed-capacity grouped/fused
   mixed MLP operator that fills branch capacity with useful rows, not dummy
   padding or promoted low-priority dense rows.
4. Dense fallback remains the right policy for underfilled branches until that
   grouped operator exists.

## 20. Current-Tree Clean Recheck After Dense-Fill Changes

I reran a short clean-serving check after the fixed-block dense-fill changes to
make sure the active diagnosis still holds in the current worktree. This run
uses no diagnostic CUDA-event timing, so its tok/s values are serving evidence,
not synchronization-heavy profiler evidence.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_current_clean_recheck_bs64_math64_20260701
```

Setup: Llama-3.1-8B, EAGLE3 K=8, `math_reasoning`, bs64, max tokens 64,
64 fixed requests, one repeat.

| method | total tok/s | full-batch tok/s | accepted draft/step | avg GPU util | CUDA Graph |
|---|---:|---:|---:|---:|---|
| dense baseline | 1790.509 | 2605.266 | 1.146 | 69.2% | `FULL=5,NONE=26,PIECEWISE=1` |
| base_only_24 | 1917.859 | 4015.168 | 1.799 | 71.4% | `FULL=30,NONE=2` |
| all_corrected_24 | 1826.585 | 2801.913 | 1.147 | 79.2% | `FULL=7,NONE=24,PIECEWISE=1` |
| speclink_t08 | 1805.542 | 2700.907 | 1.147 | 72.4% | `FULL=7,NONE=24,PIECEWISE=1` |

Current read:

1. `base_only_24` is not acceptance-limited in this row. It accepts more draft
   tokens per step than dense and reaches `1.541x` full-batch throughput, so
   the sparse upper bound is real when graph coverage is good.
2. `all_corrected_24` is using the dense fastpath exact control here
   (`dense_fastpath`, storage/dense `1.0`). This is the right optimized exact
   behavior. The non-fastpath all-corrected sparse+residual path remains an
   operator diagnostic, not a speed path.
3. `speclink_t08` has the same accepted draft/step as all-corrected in this
   short run, but only `1.037x` full-batch speedup. That points to graph and
   mixed-operator efficiency rather than accepted-length collapse.
4. The next implementation work should target graph-stable useful-row grouped
   MLP execution. More dense-fill, scalar Triton, or Python stream overlap has
   already been tested as insufficient.

## 21. Small-M cuSPARSELt alg1 Threshold 160 Probe

I added one more serving candidate to test the only low-risk cuSPARSELt planner
knob that remained from the small-M operator sweep:

```text
lossy_prefix2_noverify_sparse_smallm_alg1_t160_compile
```

This is the noverify-sparse compile boundary with `alg_id=1` extended from
rows `<=96` to rows `<=160`, because the standalone cuSPARSELt sweep showed
that rows around `144` prefer alg1 while larger rows generally prefer alg0.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_smallm_alg1_t160_quality_throughput_bs8_64_max512_20260701
```

Quality gate:

| dataset | limit | dense acc | SR24 acc | delta |
|---|---:|---:|---:|---:|
| GSM8K | 50 | 0.7400 | 0.7400 | 0pp |

Throughput on `math_reasoning`, max tokens 512:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 1.017x | 1.003x |
| 16 | 1.029x | 1.032x |
| 32 | 1.003x | 1.006x |
| 64 | 0.962x | 0.950x |

Read:

1. The quality boundary is still safe under the current 8pp gate.
2. Extending alg1 to rows `<=160` gives a small low/mid-batch improvement, so
   it is worth keeping as a planner knob for underfilled sparse calls.
3. The gain is only about `1-3%` and bs64 regresses, so alg selection is not a
   route to the required `1.2x` serving speedup.
4. The next implementation work remains a graph-stable grouped/fused mixed MLP
   operator that fills branches with useful rows. Do not keep running
   alg-id-only or scalar Triton sweeps unless they are part of that operator
   design.

## 22. 2026-07-02 Packed-Operator Shape Probe and Current Triage

This pass rechecked the active goal under the current tree:

1. decide whether `base_only_24` is slow because acceptance collapses or because
   GPU/operator shape is inefficient,
2. keep `all_corrected_24` honest about GPU-resident `compressed_dense`, and
3. turn the `speclink_t08` speed target into a concrete grouped-operator
   requirement instead of another threshold sweep.

New outputs:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_packed_operator_shape_probe_20260702
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_packed_operator_policy_20260702
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_current_base_all_t08_triage_20260702
```

The packed verifier-block microbench used K=8, prefix in `{0,1,2,4}`,
batch size `{8,16,32,64}`, coalesce factor `{1,2,4}`, dense capacity multiple
64, minimum dense capacity 64, and minimum sparse-base capacity 128.  The
important semantic constraint is preserved: dense-important rows and
2:4-sparse rows are disjoint; sparse rows are not recomputed by dense.

Best `packed_parallel` local speedups, measured as dense graph ms divided by
packed-parallel graph ms:

| original bs | coalesce | effective bs | prefix | speedup |
|---:|---:|---:|---:|---:|
| 8 | 1 | 8 | 0 | 0.610x |
| 8 | 4 | 32 | 0 | 0.985x |
| 16 | 4 | 64 | 0 | 1.380x |
| 16 | 4 | 64 | 1 | 1.203x |
| 16 | 4 | 64 | 2 | 1.240x |
| 32 | 2 | 64 | 0 | 1.380x |
| 32 | 2 | 64 | 1 | 1.201x |
| 32 | 2 | 64 | 2 | 1.239x |
| 64 | 1 | 64 | 0 | 1.380x |
| 64 | 1 | 64 | 2 | 1.241x |

Planner output for a 1.2x local operator target:

| bs | best single-prefix speedup | planner action | required grouping |
|---:|---:|---|---|
| 8 | 0.610x | dense fallback | no local 1.2x point up to coalesce 4 |
| 16 | 0.736x | dense fallback until grouped | coalesce 4, effective bs64, prefix1 |
| 32 | 0.990x | dense fallback until grouped | coalesce 2, effective bs64, prefix1 |
| 64 | 1.380x | use mixed single block | prefix0 or prefix2 works locally |

Current clean/triage rows still support the same diagnosis:

| method | bs | full-batch tok/s | speedup vs dense | accepted draft/step | GPU util | CUDA Graph |
|---|---:|---:|---:|---:|---:|---|
| dense baseline | 64 | 2605.266 | 1.000x | 1.146 | 69.2% | `FULL=5,NONE=26,PIECEWISE=1` |
| `base_only_24` | 64 | 4015.168 | 1.541x | 1.799 | 71.4% | `FULL=30,NONE=2` |
| `all_corrected_24` | 64 | 2801.913 | 1.075x | 1.147 | 79.2% | `FULL=7,NONE=24,PIECEWISE=1` |
| `speclink_t08` | 64 | 2700.907 | 1.037x | 1.147 | 72.4% | `FULL=7,NONE=24,PIECEWISE=1` |

For `base_only_24`, this rules out accepted-length collapse in the current
clean row.  It accepts more draft tokens than dense and has good graph coverage.
When base-only is slow in narrower rows, it is because the sparse operator shape
is underfilled or graph/config is unfavorable, not because EAGLE3 suddenly stops
accepting drafts.

For `all_corrected_24`, the exact sparse+residual operator remains the wrong
speed path:

| exact path | residual backend | residual device | densefastpath | full speedup vs dense | read |
|---|---|---|---:|---:|---|
| densefastpath | `dense_fastpath` | `none` | yes | 1.004x | correct optimized exact control |
| no-fastpath compressed dense | `compressed_dense` | `cuda:0` | no | 0.268x | GPU-resident but materialize/GEMM path is too expensive |
| no-fastpath torch sparse residual | `torch_sparse` | `none` | no | 0.358x | not enough GPU work and graph coverage |
| no-fastpath compressed Triton | `compressed_dense` | `cuda:0` | no | 0.189x | current Triton residual kernel is worse |

So `compressed_dense` is already GPU-resident when requested, and
`SPECLINK_SR24_REQUIRE_GPU_RESIDUAL=1` now guards both non-GPU residual modules
and CPU extraction fallback modules.  The bottleneck is not CPU placement; it
is the exact residual operator's data movement and small-kernel structure.

One code correction was made in `vllm/vllm/speclink_sr24.py`: scheduler policy
rows with `mixed_operator=packed_parallel` are no longer considered implemented
just because `SPECLINK_SR24_ROUTE_OVERLAP_STREAMS=1` is enabled.  The old
Python-level stream split is not the packed_parallel grouped operator measured
by the microbench.  It can still be forced for ablation with
`SPECLINK_SR24_SCHEDULER_POLICY_ALLOW_LEGACY_MIXED=1`, but the default policy
now correctly falls back dense until a real grouped operator exists.

Updated implementation read:

1. `speclink_t08` should not chase more scalar thresholds first.  Its quality
   issue came from broad sparse-only layer routing; fixed-prefix/token-level
   routing can fit the 8pp GSM8K budget.
2. The speed target needs a grouped/fused verifier MLP.  For K=8, the measured
   local operator needs roughly effective bs64 useful verifier blocks before
   packed dense/sparse wins by 1.2x.
3. bs64 can use a single grouped block if a true packed_parallel operator is
   integrated.  bs32 needs coalesce 2.  bs16 needs coalesce 4.  bs8 does not hit
   local 1.2x in the current search up to coalesce 4 and likely needs a better
   small-M kernel or dense fallback.
4. The next code milestone should be a live grouped verifier queue plus a real
   operator boundary, not another Python stream-overlap or dense-fill variant.

## 2026-07-02 User-Budget Gate Refresh

Latest gate output:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_budget_gate_math256_20260702
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_user_budget_outputbuf_overlap_gate_20260702
```

The current best quality/speed compromise is `lossy_prefix2_down_front14_compile`.
It keeps `gate_up` dense, makes only `down_proj` row-routed, keeps ordinary /
no-verify `down_proj` rows dense in layers 0-13, and routes the same rows
through 2:4 sparse-only in layers 14-31.  On Llama-3.1-8B GSM8K-CoT limit 50:

| candidate | dense acc | SR24 acc | delta | paired reg/imp | gate |
|---|---:|---:|---:|---:|---|
| `lossy_prefix2_front24_serving_policy_compile` | 0.7800 | 0.6800 | -10pp | 6/1 | fail |
| `lossy_prefix1_mlp_noverify_sparse_minbase128_compile` | 0.7800 | 0.1400 | -64pp | 35/3 | fail |
| `lossy_prefix2_down_front14_compile` | 0.7800 | 0.7000 | -8pp | 6/2 | pass |
| `lossy_prefix2_down_front14_outputbuf_compile` | 0.7800 | 0.6800 | -10pp | 6/1 | fail |
| `lossy_prefix2_down_front14_outputbuf_overlap_compile` | 0.7800 | 0.6800 | -10pp | 6/1 | fail |

For the passing `down_front14` candidate, `math_reasoning`, K=8, max tokens 256,
64 fixed requests:

| bs | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1124.988 | 1078.575 | 0.959x | 1201.557 | 1169.594 | 0.973x |
| 16 | 1799.691 | 1709.236 | 0.950x | 1989.181 | 1962.755 | 0.987x |
| 32 | 2319.524 | 2378.370 | 1.025x | 2677.973 | 2777.964 | 1.037x |
| 64 | 2330.556 | 2692.901 | 1.155x | 3441.001 | 3617.463 | 1.051x |

Takeaways:

1. The current disjoint row-routed path already avoids dense recompute for rows
   that are sparse-only.  Do not use `SPECLINK_SR24_ROW_ROUTED_MLP_REUSE_BASE_OUTPUT`
   or fixed-block dense-fill as the main path; those are fill/diagnostic
   ablations that intentionally overlap dense and sparse work.
2. Output-buffer reuse is not yet correctness-transparent.  Both output-buffer
   variants lose an extra 2pp on GSM8K-50 and must be debugged with token-level
   equivalence before any throughput claim.
3. The 8pp budget helps identify a real operating point, but it is not enough
   for 1.2x across bs8/16/32/64 with the current PyTorch/cuSPARSELt split
   operator.  bs32/64 are positive; bs8/16 need dense fallback, larger effective
   verifier groups, or a better small-M kernel.
4. The next credible systems milestone is a packed verifier-block data format
   and grouped/fused MLP operator: contiguous `[request, K+1, hidden]` verifier
   blocks, dense rows for prefix/bonus only, sparse rows for the remaining draft
   positions, one route descriptor per block, and optional queueing only when it
   can fill an effective bs64 operator without violating dependencies.

## 2026-07-02 Goal Refresh: Cause, Correctness, and Next Operator

Latest refreshed artifacts:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_goal_baseonly_cause_refresh_20260702/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixed_block_output_buffer_equivalence_20260702/result.json
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downfront14_no_compile_quality20_20260702/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_outputbuf_no_compile_quality20_20260702/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_downfront14_sample_divergence_20260702/report.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_compressed_residual_kernel_refresh_20260702/summary.md
```

`base_only_24` refreshed read:

- Slow or borderline rows are not caused by accepted-length collapse. The
  refreshed report has slow bs8/16 gate-up base-only rows with accepted/step
  above dense and GPU util around 92-94%.
- Down-proj tail rows show healthy acceptance and GPU util but many CUDA Graph
  `NONE` steps, so the issue is graph/shape/operator efficiency.
- Full MLP base-only remains the speed upper bound when graph coverage is good,
  but it is not quality-safe.

`all_corrected_24` refreshed read:

- `compressed_dense` is already GPU-resident in the measured path. The current
  bottleneck is not CPU placement.
- The current packed Triton residual microbench is not usable: for rows512
  Llama MLP shapes it is about 8-9x slower than a dense residual GEMM and has
  loose numerical error.
- Exact all-corrected should therefore stay a dense-equivalent control until a
  real tensor-core-backed fused residual operator exists, or until the design
  reduces exact residual rows instead of correcting every row.

`speclink_t08` quality read:

- Fixed-block output-buffer local assembly is exact in the new unit check
  (`cross_max_abs_diff=0`).
- Matched GSM8K-20 checks show output-buffer and non-output-buffer
  `down_front14` both drop from dense `0.6000` to SR24 `0.5000`; output-buffer
  is not the accuracy root cause.
- Sample divergence for GSM8K-50 down_front14 shows six paired regressions and
  two improvements. The regressions diverge early or mid response
  (`first_diff_token` median 12.5) and are arithmetic reasoning changes, not
  answer-extraction/formatting failures.

Updated direction:

1. Keep the 8 percentage-point GSM8K budget and use at least GSM8K limit 50 for
   quality gates.
2. Do not chase lossless behavior or broad sparse-only non-important tokens.
   The current boundary is down-proj sparse scope, not output-buffer assembly.
3. The speed target needs a systems change: fixed-capacity verifier-block data
   format plus a fused/grouped dense-important and 2:4-sparse-unimportant MLP
   operator. Python/PyTorch split branches, current packed Triton residual, and
   exact all-corrected residual paths are not credible 1.2x routes.

## 2026-07-02 Prefix0 Packed-Operator Refresh

Focused operator refresh:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix012_packed_operator_refresh_20260702/summary.csv
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix012_packed_operator_refresh_20260702/planner/operator_planner.md
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix012_packed_operator_refresh_20260702/planner_prefix0_rerun/operator_planner.md
```

This adds `prefix=0` to the packed verifier-block microbench. Prefix0 keeps
only the verifier bonus row dense and sends all draft rows through the 2:4 base
branch, so it tests whether "important token count is too small" can be solved
by making the sparse branch larger.

Key operator-local result for K=8, capacity multiple 64:

| bs | prefix0 single-block local speedup | grouped blocks needed for >=1.2x | effective bs | grouped local speedup |
|---:|---:|---:|---:|---:|
| 8 | 0.602x | 8 | 64 | 1.373x |
| 16 | 0.724x | 4 | 64 | 1.373x |
| 32 | 0.992x | 2 | 64 | 1.374x |
| 64 | 1.374x | 1 | 64 | 1.374x |

Read:

- Prefix0 improves the operator upper bound once the useful rows are filled,
  but it does not make low-batch single-block execution viable. bs8/16/32 still
  need grouping to effective bs64.
- A repo-local operator policy was added for this exact shape:
  `examples/evaluate/eval-guidellm/configs/sr24_scheduler_policy_prefix0_grouped_k8_effective64.json`.
  It is intentionally marked as operator-local and not a live serving speed
  claim for the current legacy split PyTorch/cuSPARSELt path.
- This strengthens the systems direction: the next implementation needs a
  dependency-safe verifier-block grouping queue plus a real packed/fused mixed
  MLP operator. Reducing dense-important rows alone is not sufficient.
