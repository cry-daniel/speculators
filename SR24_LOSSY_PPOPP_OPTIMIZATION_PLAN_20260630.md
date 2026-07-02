# SR24 Lossy Systems Optimization Plan - 2026-06-30

## Goal

Target: make SpecLink/SR24 at least `1.2x` faster than dense EAGLE3 for most
batch sizes `8/16/32/64` and datasets, while allowing bounded quality loss.

Working quality gate:

- task: GSM8K exact-match first, `limit >= 50`
- budget: absolute accuracy drop within `8 percentage points`
- no lossless requirement for early optimization

## Current Semantics

The current row-routed MLP path already matches the requested lossy semantics:

- important verifier rows go through dense MLP only
- unimportant verifier rows go through the 2:4 sparse MLP only
- sparse-only rows are not corrected again by dense residual work

This is the right policy shape. The remaining problem is execution efficiency,
not the high-level routing semantics.

## Small-Scale Gate Results

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_lossy_ppopp_gate_bs8_64_math128_gsm8k50_20260630
```

Setup:

- model: Llama-3.1-8B with EAGLE3, K=8
- quality task: GSM8K, limit 50
- throughput task: `math_reasoning`
- batch sizes: `8,16,32,64`
- throughput max tokens: 128

All three candidates pass the 8pp quality gate:

| candidate | dense acc | SR24 acc | delta | best total speedup | best full-batch speedup |
|---|---:|---:|---:|---:|---:|
| `lossy_prefix2_rowrouted_mlp_minbase64` | 0.7400 | 0.7200 | -2pp | 0.966x | 1.006x |
| `lossy_prefix1_rowrouted_mlp_minbase128` | 0.7400 | 0.7200 | -2pp | 0.935x | 0.983x |
| `gateup_res16_25_base26_31_critical4_smallrow160` | 0.7400 | 0.7200 | -2pp | 0.865x | 0.913x |

Best row-routed candidate, `lossy_prefix2_rowrouted_mlp_minbase64`:

| batch | dense total tok/s | SR24 total tok/s | total speedup | dense full tok/s | SR24 full tok/s | full speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1016.4 | 872.7 | 0.859x | 1062.0 | 908.5 | 0.855x |
| 16 | 1628.3 | 1173.3 | 0.721x | 1724.4 | 1240.5 | 0.719x |
| 32 | 2081.7 | 1747.6 | 0.839x | 2282.6 | 2005.3 | 0.879x |
| 64 | 2400.3 | 2317.7 | 0.966x | 2774.3 | 2789.8 | 1.006x |

This confirms that relaxing quality is not enough. The policy can pass the
quality gate, but the current implementation still cannot reach `1.2x`.

## Focused Breakdown

Breakdown roots:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_rowrouted_breakdown_bs64_math128_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_ppopp_rowrouted_breakdown_bs64_inst64_20260630
```

Clean bs64 serving:

| method | total tok/s | full-batch tok/s | avg GPU util | CUDA Graph |
|---|---:|---:|---:|---|
| dense EAGLE3 | 2242.7 | 3039.2 | 78.4% | `FULL=20,NONE=43,PIECEWISE=1` |
| base-only 2:4 | 2737.5 | 5412.5 | 82.0% | `FULL=62,NONE=2` |
| row-routed SR24 | 1739.6 | 2812.3 | 83.7% | `FULL=83,NONE=13` |

Read:

- base-only 2:4 has a real operator upper bound, especially in full-batch
  windows.
- row-routed SR24 has healthy graph coverage and GPU utilization, but is still
  slower than dense.
- Therefore the issue is useful-work efficiency inside the mixed MLP path.

Instrumented bs64 row-routed routing:

- draft residual/base rows: `4026 / 12078`
- non-draft residual rows: `2013`
- draft residual fraction: `0.25`
- this matches the intended fixed prefix2+bonus policy.

Instrumented row-routed MLP fixed-block timing:

| component | avg ms/call |
|---|---:|
| sparse/base gate-up | 0.950 |
| sparse/base down | 0.975 |
| dense-important gate-up | 0.196 |
| dense-important down | 0.115 |
| view/act/assemble combined | about 0.047 |

The sparse branch dominates even after most draft rows are sparse-only. The
dense-important part is not the main problem.

Microbench evidence:

| shape | residual frac | dense graph ms | base sparse graph ms | mixed graph ms | routed split graph ms |
|---|---:|---:|---:|---:|---:|
| gate/up `512x28672x4096` | 0.125 | 0.539 | 0.354 | 0.555 | 0.789 |
| gate/up `512x28672x4096` | 0.250 | 0.540 | 0.354 | 0.621 | 0.657 |
| down `512x4096x14336` | 0.125 | 0.292 | 0.166 | 0.266 | 0.303 |
| down `512x4096x14336` | 0.250 | 0.292 | 0.166 | 0.287 | 0.320 |

Sparse-only is faster than dense, but sparse plus correction/route machinery
erases most of the win. For gate/up, the current routed split is especially bad
at the row counts that matter for bs8/16/32.

## Design Direction

The next credible path is not another threshold/controller sweep. It should be
a systems implementation change with a graph-stable packed data format and a
fused or grouped mixed MLP operator.

### Data Format

Keep verifier rows in the natural block layout:

```text
hidden:   [batch, K + 1, hidden]
row_kind: [batch, K + 1]  # dense-important or sparse-only
```

Do not flatten into dynamic Python row lists on every step. The scheduler should
emit fixed-capacity route descriptors:

```text
dense_rows[B, dense_capacity]
sparse_rows[B, sparse_capacity]
dense_active_mask[B, dense_capacity]
sparse_active_mask[B, sparse_capacity]
```

The capacities should be stable for CUDA Graph capture. Active masks should be
consumed by the GPU operator, not by repeated `nonzero/index_select` on the CPU
or Python side.

### Operator

Replace the current four independent MLP launches:

```text
dense gate_up + dense down + sparse gate_up + sparse down
```

with a grouped or fused wrapper:

1. dense-important rows and sparse-only rows remain disjoint
2. sparse rows are packed enough to hit a profitable cuSPARSELt/CUTLASS tile
3. dense rows are grouped across requests, and optionally across adjacent layers
4. the wrapper is graph-captured with fixed capacities
5. branch overlap is enabled only when both dense and sparse sides exceed a
   measured row threshold

The existing PyTorch auxiliary-stream overlap probe is negative. Concurrency
should be implemented in a captured C++/CUDA/CUTLASS/Triton wrapper, not by
creating or switching Python streams in the hot path.

### Fill Strategy

When important rows are too few:

- pad to a fixed dense capacity only if the padded work stays below dense MLP
  cost
- otherwise group dense rows across requests/layers before launch

When sparse-only rows are too few:

- either defer/group across requests/layers, or fall back to dense for that
  layer/step
- do not launch tiny sparse branches just to preserve the policy shape

The controller should be row-count aware. A route that is quality-correct but
underfilled should not be allowed to make the serving path slower than dense.

### Pipeline

Pipeline opportunity exists, but only after the operator shape is stable:

- scheduler prepares next-step route descriptors while current verifier MLP runs
- dense-important and sparse-only MLP branches can run concurrently only when
  both branches are large enough
- for small branches, use one grouped launch or dense fallback; do not pay
  extra stream/launch overhead

## Immediate Next Experiments

1. Implement a fixed-capacity packed verifier-block MLP microbench first.
   Required rows: bs8/16/32/64 with K=8, approximately rows
   `72/144/288/576`.
2. Require the packed microbench to beat dense by at least `1.25x` at rows
   `288` and `576` before adding another live serving path.
3. After microbench passes, integrate the packed route descriptor into vLLM and
   run the same quick gate:
   GSM8K limit 50, 8pp budget, `math_reasoning` bs `8/16/32/64`.
4. Only after the quick gate reaches near `1.2x`, expand to more datasets and
   max tokens 2048.

## Current Decision

The current code is correct enough for the lossy-row-routing policy, but the
implementation is not fast enough. The useful next work is a packed/fused mixed
MLP operator. More threshold tuning, lossless gates, or Python-side active-row
selection is unlikely to reach the requested systems target.

## 2026-06-30 Addendum

Two additional checks narrow the optimization path.

First, a custom scalar Triton base-2:4 prototype was added:

```text
examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_triton_base24_mlp.py
```

It stores sparse weights as `values[out, group, 2]` plus absolute `k0/k1`
positions and keeps all metadata on GPU. This is a reasonable route-descriptor
format, but the scalar gather-multiply implementation is not a viable operator.
At actual K8 row shapes, it is `30-80x` slower than dense:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_triton_base24_mlp_actual_rows_k8_20260630
```

Therefore, a PPoPP-grade implementation cannot be a plain Triton gather kernel.
It needs a tensor-core-backed sparse base path, either by driving cuSPARSELt/
CUTLASS more directly or by writing a CUDA operator around an equivalent
compressed 2:4 layout.

Second, the noverify-sparse plus Triton fixed-block assembly live candidate was
run end to end:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_tritonassemble_quality_throughput_bs8_64_20260630
```

This candidate matches the desired lossy semantics: important rows are dense,
unimportant/noverify rows are sparse-only, and sparse-only rows are not later
overwritten by dense work. It passes GSM8K-50 with no observed accuracy loss
(`0.7400 -> 0.7400`), but throughput is only parity:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 0.996x | 0.996x |
| 16 | 1.000x | 1.002x |
| 32 | 1.005x | 1.000x |
| 64 | 1.002x | 0.989x |

This means output assembly and duplicate dense-overwrite semantics are not the
remaining blocker. The blocker is the small-M sparse MLP branch itself and the
fact that dense-important and sparse-unimportant work are still launched as
separate high-level PyTorch/cuSPARSELt pieces.

Third, a gate-up-only noverify-sparse probe was run:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_noverify_sparse_quality_throughput_bs8_64_20260630
```

This keeps only `gate_up_proj` under SR24 and leaves `down_proj` dense. It
passes GSM8K-50 with no observed accuracy loss (`0.7400 -> 0.7400`) but still
only reaches:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 0.997x | 0.997x |
| 16 | 1.006x | 1.001x |
| 32 | 1.021x | 0.996x |
| 64 | 1.001x | 1.005x |

This rules out the simple hypothesis that the small-M sparse down projection is
the only reason the row-routed path fails to accelerate. The gate/up sparse
branch itself and the high-level route/launch structure are enough to erase the
compute saving.

The next implementation should be treated as an operator/data-layout project:

- fixed-capacity route descriptors that are CUDA-graph stable,
- GPU-consumed active masks instead of Python-side row-list construction,
- tensor-core sparse base kernels for small M,
- grouped dense-important rows and sparse-unimportant rows with enough fill,
- dense fallback only when the sparse branch cannot be profitably filled,
- optional branch overlap inside a captured C++/CUDA wrapper, not Python stream
  switching.

Short-term threshold sweeps, K-only sweeps, scalar Triton sparse kernels, and
Python-level overlap flags should not be the main line unless the underlying
operator/data path changes.

## 2026-06-30 Final Addendum: Quality Budget Is Not Enough

The latest probes specifically tested the user's requested optimization
direction: unimportant tokens should not run sparse and then dense again, the
accuracy gate can allow up to `8pp` loss, and the system path should focus on
data layout/operator design rather than a lossless controller.

### Channel Split Is Not a Good Shortcut

The `gateup_channel_pair_dense25_eager` and
`gateup_channel_pair_dense50_eager` probes split the gate/up intermediate
channels instead of verifier rows. This avoids tiny-M row splits, but the
quality loss is too large:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_channel_pair_quality_only_20260630_eager
```

| channel dense fraction | dense acc | SR24 acc | delta |
|---:|---:|---:|---:|
| 25% | 0.7200 | 0.5000 | -22pp |
| 50% | 0.7200 | 0.5600 | -16pp |

This rules out broad all-layer channel-level sparsification as the near-term
speed path. It also exposed a compile-safety issue in the current lazy grouped
down-weight path, so channel split should stay disabled unless revisited with a
much narrower sensitivity model.

### Minbase and Prefix Relaxation Stay Near Parity

The current row-routed/noverify-sparse semantics already do the important
thing: important rows are dense-only, unimportant rows are 2:4-only, and
unimportant rows are not overwritten by dense work. Four follow-ups tried to
spend the 8pp quality budget more aggressively:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_noverify_sparse_minbase256_gate_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_noverify_sparse_minbase256_gate_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_noverify_sparse_gate_bs64_20260630
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_prefix1_gateup_only_noverify_sparse_bs64_max512_20260630
```

| variant | GSM8K-50 | max tokens | full speedup | total speedup |
|---|---:|---:|---:|---:|
| all-MLP noverify sparse, minbase256 | `0.7400 -> 0.7400` | 128 | 1.0005x | 0.9892x |
| gate_up-only noverify sparse, minbase256 | `0.7400 -> 0.7400` | 128 | 0.9934x | 1.0060x |
| gate_up-only prefix1 noverify sparse | `0.7400 -> 0.7400` | 128 | 0.9990x | 1.0115x |
| all-MLP prefix1 noverify sparse | `0.7400 -> 0.7400` | 128 | 0.9914x | 0.9982x |
| gate_up-only prefix1 noverify sparse | quality skipped | 512 | 0.9992x | 1.0004x |

Relaxing from prefix2 to prefix1 did not create speed. Increasing the sparse
base fill threshold to 256 did not create speed. Restricting SR24 to gate/up did
not create speed. The current serving path is operator-limited, not
controller-limited.

### Current Bottleneck

The focused gate_up-only noverify-sparse breakdown is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_gateup_only_noverify_sparse_breakdown_bs64_math_manual_20260630
```

Clean serving is essentially parity:

| method | total tok/s | full-batch tok/s | GPU util | graph |
|---|---:|---:|---:|---|
| dense EAGLE3 | 2377.825 | 2768.973 | 88.9% | `FULL=35,NONE=92,PIECEWISE=1` |
| gate_up-only SR24 | 2325.595 | 2784.583 | 92.1% | `FULL=96,NONE=32` |

The diagnostic row-routed gate/up branch costs `1.131 ms/call`; most of that is
the sparse base (`0.936 ms/call`), while the dense-important GEMM is only about
`0.164 ms/call` and assembly is about `0.023 ms/call`. The rows are small:
about `257` total verifier rows per call, with only `171` sparse-base rows and
`86` dense rows. This is too little work per branch for the current
PyTorch/cuSPARSELt split path to beat dense reliably.

### Updated Systems Direction

The near-term implementation should be designed like a systems paper artifact,
not like another policy sweep:

1. **Graph-stable route descriptors.** Emit fixed-capacity tensors such as
   `dense_rows[B, C_d]`, `sparse_rows[B, C_s]`, and active masks. Keep the data
   on GPU and make capacities stable for CUDA Graph capture.
2. **Disjoint row execution.** Important rows run dense-only. Unimportant rows
   run 2:4-only. No row should pay both sparse and dense unless it is an
   explicit fallback for an underfilled branch.
3. **Fill-aware grouping.** Coalesce sparse rows across requests, decode steps,
   or adjacent layers until the branch has enough rows for tensor-core 2:4 to
   win. If it cannot be filled, use dense fallback instead of launching a tiny
   sparse branch.
4. **Tensor-core sparse operator.** Scalar Triton gather kernels are not viable.
   Use cuSPARSELt/CUTLASS or a custom CUDA wrapper around an equivalent 2:4
   tensor-core layout.
5. **Fused/grouped MLP wrapper.** Launch dense-important and sparse-only
   branches through a single graph-captured grouped wrapper. Only enable
   branch overlap inside that wrapper when both sides exceed measured row
   thresholds.
6. **Serving gate.** Before another full vLLM matrix, the standalone mixed MLP
   microbench must beat dense by at least `1.25x` at actual K8 row counts near
   `288` and `576`. Then rerun GSM8K limit 50 and bs `8/16/32/64`.

This is the clearest current path to the requested `1.2x` goal. The present
Python/PyTorch split operator is useful as a correctness and measurement
scaffold, but it should not be the final implementation.

## 2026-07-01 Fixed-Capacity Fill Probe

I extended the packed verifier-block MLP microbenchmark with fixed-capacity
branch buffers:

```text
examples/evaluate/eval-guidellm/scripts/profile_speclink_sr24_packed_verifier_mlp.py
```

New controls:

```text
--capacity-multiple
--min-dense-capacity
--min-base-capacity
```

The new padded path is intentionally close to the desired graph-stable data
format: active dense-important rows and active sparse-only rows are copied into
fixed-capacity branch buffers, then the dense and 2:4 sparse MLP branches run at
those capacities. This tests whether "pad to fill tensor cores" can solve the
important-token-underfill problem.

Results:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_fixed_capacity_fill_k8_bs8_64_prefix12_20260701
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_fixed_capacity_fill64_k8_bs8_64_prefix12_20260701
```

Representative graph-captured speedups versus dense:

| case | unpadded packed | two-stream packed | fixed-capacity padded | read |
|---|---:|---:|---:|---|
| bs8/prefix1, cap64 | 0.483x | 0.605x | 0.492x | sparse branch still too small |
| bs16/prefix1, cap64 | 0.592x | 0.716x | 0.587x | padding does not help |
| bs32/prefix1, cap64 | 0.853x | 0.979x | 0.857x | near parity only |
| bs64/prefix2, cap64 | 1.198x | 1.241x | 1.199x | works only when no padding is needed |
| bs64/prefix2, cap128 | 1.199x | 1.240x | 1.027x | dense-side padding hurts |

Design read:

- fixed-capacity descriptors are still useful for CUDA Graph stability;
- padding dummy rows is not a speed solution for bs8/16/32;
- dense-important rows are especially expensive to pad, because dense gate/up
  work scales with the padded capacity;
- the only successful standalone case is bs64 where active row counts already
  fill the branch buffers;
- the serving implementation should fill capacity with real useful rows by
  grouping/coalescing, or fall back to dense when a branch is underfilled.

Updated operator target:

1. Keep fixed-capacity route buffers as the data format.
2. Do not use padding-only as the live optimization.
3. Add a grouped/fused operator or grouped-GEMM wrapper that can consume several
   verifier row groups as useful work, not dummy rows.
4. Use dense fallback for underfilled branches until the grouped operator exists.
5. Re-run live serving only after the standalone mixed MLP exceeds dense by
   `>=1.25x` at bs32-shaped rows without relying on dummy padding.

## 2026-07-01 Useful-Row Coalescing Result

I added a useful-row coalescing mode to the packed verifier-block MLP
microbenchmark:

```text
--coalesce-factors
```

This mode groups several independent verifier blocks with the same weights into
one larger mixed dense/sparse branch. It is an optimistic upper bound for what a
grouped operator or scheduler could achieve if it can collect real useful rows;
it is not the same as padding dummy rows.

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_useful_coalesce_k8_bs8_32_prefix12_rerun_20260701
```

The derived planner table for this root is:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_packed_useful_coalesce_k8_bs8_32_prefix12_rerun_20260701/operator_planner.md
```

K8/prefix2 graph-captured read:

| original bs | coalesce factor needed for mixed >=1.2x vs coalesced dense | effective bs | mixed vs coalesced dense at that point |
|---:|---:|---:|---:|
| 8 | 8 | 64 | 1.242x |
| 16 | 4 | 64 | 1.240x |
| 32 | 2 | 64 | 1.241x |

The same rows also show that dense coalescing itself is strong. For example,
original bs8/coalesce8 has coalesced dense `2.229x` faster than eight serial
bs8 dense blocks, while mixed reaches `2.768x` versus the serial dense
reference. The mixed path is useful, but only after the operator has enough real
rows.

Design consequence:

- The target fill point for the current tensor-core sparse path is roughly
  effective bs64 for K8/prefix2.
- A low-batch live implementation needs real useful-row grouping, not dummy
  padding.
- If the scheduler cannot legally collect enough independent verifier work,
  dense fallback is the correct low-batch policy.
- Future serving integration should first target bs32/64 because they need
  only 2x/1x grouping to reach the useful row regime; bs8/16 need either more
  aggressive scheduling or a genuinely better small-M sparse kernel.

## 2026-07-01 Live Fixed-Block Dense-Fill Probe

The live vLLM row-routed MLP now has an explicit tile-fill ablation:

```text
SPECLINK_SR24_ROW_ROUTED_MLP_FIXED_BLOCK_DENSE_FILL=1
```

When enabled, the fixed-prefix row-routed MLP keeps the important prefix rows
and verifier bonus row dense, then promotes adjacent sparse/base rows into the
dense branch until `SPECLINK_SR24_ROW_ROUTED_MLP_MIN_DENSE_ROWS` is reached.
This is not the clean sparse-only policy for those promoted rows; it is a
controlled fill-factor probe to test whether the dense branch was too small.

Quality/throughput root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_densefill64_128_quality50_speed_b8_64_math512_20260701
```

Setup: Llama-3.1-8B, EAGLE3 K=8, GSM8K `limit=50`, `math_reasoning`, bs
`8/16/32/64`, max tokens 512. Both candidates pass the 8pp gate exactly:
dense `0.7400`, SR24 `0.7400`.

| candidate | best total speedup | best full-batch speedup | read |
|---|---:|---:|---|
| dense-fill64 | 1.026x | 1.035x | small bs16/32 win, bs64 regresses |
| dense-fill128 | 1.037x | 1.001x | bs32 total win only, bs64 regresses badly |

Detailed dense-fill64 serving rows:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 1.001x | 1.001x |
| 16 | 1.018x | 1.035x |
| 32 | 1.026x | 0.991x |
| 64 | 0.962x | 0.952x |

An auxiliary-stream overlap probe with dense-fill64 is also negative: bs32
total/full-batch speedups are `0.997x/0.979x`, and bs64 total/full-batch
speedups are `0.950x/0.949x`.

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_fixedblock_densefill64_overlap_b32_64_math512_20260701
```

Conclusion: important-token underfill is not the sole bottleneck. Padding or
promoting low-priority rows into the dense branch can pass the quality gate but
does not approach the `1.2x` target and hurts high batch. Continue with a real
fixed-capacity grouped/fused operator that fills sparse and dense branches with
useful rows; keep `fixed_block_dense_fill` as an off-by-default ablation.

## 2026-07-01 Small-M alg1 Threshold 160 Result

I also tested the remaining simple small-M planner knob from the cuSPARSELt alg
sweep: use `alg_id=1` for sparse-base rows `<=160` instead of only `<=96`.
This corresponds to:

```text
lossy_prefix2_noverify_sparse_smallm_alg1_t160_compile
```

Result root:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results.bak/sr24_smallm_alg1_t160_quality_throughput_bs8_64_max512_20260701
```

GSM8K-50 remains unchanged:

```text
dense 0.7400 -> SR24 0.7400
```

Serving throughput on `math_reasoning`, max tokens 512:

| bs | total speedup | full-batch speedup |
|---:|---:|---:|
| 8 | 1.017x | 1.003x |
| 16 | 1.029x | 1.032x |
| 32 | 1.003x | 1.006x |
| 64 | 0.962x | 0.950x |

Conclusion: alg selection is useful as a small backend-planner knob for
underfilled sparse MLP calls, but it does not change the optimization plan. It
does not solve branch fragmentation, launch overhead, or the lack of useful-row
grouping, and it is not enough for the `1.2x` target. The next real design step
is still a fixed-capacity grouped/fused mixed MLP operator with dense fallback
for underfilled branches.

## 2026-07-01 Addendum: Compile and Data-Format Guardrails

Detailed status:

```text
SR24_SYSTEM_OPTIMIZATION_STATUS_20260701.md
```

This pass tested two remaining plausible shortcuts and both failed as serving
paths.

First, the full MLP `base_only_24` upper bound remains strong on bs64
`math_reasoning`, K=8, max tokens 256:

| method | total tok/s | full-batch tok/s | GSM8K-50 acc |
|---|---:|---:|---:|
| dense EAGLE3 | 2786.130 | 3174.794 | 0.72 |
| base-only 2:4 | 3855.742 | 4830.710 | 0.22 |

The speed signal is real (`1.384x` total, `1.522x` full-batch), but the
quality loss is `-50pp`, far outside the allowed budget.

Second, a layer-scoped gate/up split is quality-safe only in eager mode. The
candidate keeps layers 16-25 corrected and makes layers 26-31 base-only for
`gate_up_proj`; it gets GSM8K-50 `0.72 -> 0.72` in enforce-eager mode. Under
default vLLM compile, however, the shared decoder-layer graph can be compiled
for a dense layer and then reused on a sparse-storage layer, producing dense
`extern_kernels.mm` on sparse weights. This means layer-heterogeneous SR24
formats are not a safe default-compile path without graph-key or data-format
changes.

Third, a homogeneous channel-pair split avoids that compile failure but is not
fast. Dense 25% and 50% channel splits fail quality (`-22pp` and `-16pp`).
Dense 75% passes quality (`0.72 -> 0.68`) but is slower than dense on
bs64/math/max256 with direct cuSPARSELt and graph enabled: `0.799x` total and
`0.852x` full-batch speedup.

The current systems conclusion is now stricter: the next useful work is not a
new threshold, channel fraction, dummy fill policy, or layer-level split. It is
a uniform graph-stable route format plus a grouped/fused operator that can
coalesce real useful verifier rows and fall back to dense when the sparse branch
is underfilled.
