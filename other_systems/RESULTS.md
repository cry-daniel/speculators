# RTX 5090 BF16 N:M results (2026-07-23)

## Protocol

- GPU: NVIDIA GeForce RTX 5090 (`sm_120`), no other compute process.
- Models: Qwen3-8B and Llama-3.1-8B, TP=1 synthetic weights with real
  `qkv/o/gate_up/down` shapes.
- Formats: exact 5:8 and 3:4, identical N:M values for every method.
- Kernel activation rows: `M=512/1024/2048`.
- Layer batches: `B=64/128/256`, seven draft tokens plus one current token per
  request, context length 128; therefore the same three M values.
- BF16 only; reduced-precision BF16 accumulation and TF32 are disabled.
- Split-K is screened outside formal timing.
- Formal timing: 100 CUDA Graph warmups, 10 independent trials, 1000 replays
  per trial, CUDA Event total divided by 1000.  A 256 MiB eviction precedes
  each independent trial.  Tables below report geometric mean speedup, where
  values above 1.0 would beat BF16 cuBLAS.

All 192 kernel rows and all 36 layer rows passed numerical checks.

## Kernel result

| N:M | Method | Qwen3-8B | Llama-3.1-8B | Combined |
|---|---:|---:|---:|---:|
| 5:8 | Flash-LLM | 0.770x | 0.773x | 0.772x |
| 5:8 | SpInfer | 0.790x | 0.784x | 0.787x |
| 5:8 | SparTA artifact | 0.539x | 0.537x | 0.538x |
| 3:4 | Flash-LLM | 0.745x | 0.744x | 0.745x |
| 3:4 | SpInfer | 0.774x | 0.772x | 0.773x |
| 3:4 | SparTA artifact | 0.536x | 0.532x | 0.534x |

Formal artifacts:

- `examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b_20260723/kernel_results.csv`
- `examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b_20260723/kernel_raw_trials.csv`
- `examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b_20260723/split_screen.csv`
- `examples/evaluate/eval-guidellm/results/other_systems_nm_kernel_8b_20260723/figures/kernel_speedup.png`

## One-layer result

| N:M | Method | Qwen3-8B | Llama-3.1-8B | Combined |
|---|---:|---:|---:|---:|
| 5:8 | Flash-LLM | 0.814x | 0.812x | 0.813x |
| 5:8 | SpInfer | 0.824x | 0.816x | 0.820x |
| 5:8 | SparTA artifact | 0.560x | 0.544x | 0.552x |
| 3:4 | Flash-LLM | 0.794x | 0.786x | 0.790x |
| 3:4 | SpInfer | 0.817x | 0.797x | 0.807x |
| 3:4 | SparTA artifact | 0.559x | 0.538x | 0.549x |

Formal artifacts:

- `examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b_20260723/layer_results.csv`
- `examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b_20260723/layer_raw_trials.csv`
- `examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b_20260723/split_screen.csv`
- `examples/evaluate/eval-guidellm/results/other_systems_nm_layer_8b_20260723/figures/layer_speedup.png`

## Conclusion

For these 8B shapes and M values, none of the reproduced external N:M paths
beats BF16 cuBLAS.  SpInfer is consistently the strongest of the three, but its
combined geometric mean is still only 0.787x/0.773x at kernel level and
0.820x/0.807x at one-layer level for 5:8/3:4.  SparTA is slowest because the
at-most-2:4 base and residual remain two serial weight streams and two
dependent kernels.  These results support keeping vLLM integration opt-in and
starting with SpInfer only; they do not justify replacing the dense vLLM path
by default.
