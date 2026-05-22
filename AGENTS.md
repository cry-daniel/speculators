# Speculator Throughput Experiments

This repo is currently set up to benchmark Qwen3-8B with EAGLE3 and P-EAGLE
speculators through vLLM and GuideLLM.

## Environment

Use the existing conda environment:

```bash
conda activate spec
```

Known working environment:

- Python: `/ACALAB/stu1/miniconda3/envs/spec/bin/python`
- vLLM: `0.20.0`
- GuideLLM: `0.6.0`
- Torch: `2.11.0+cu130`
- Local package: this repo installed editable into `spec`

If recreating the environment, use Python 3.12 and install vLLM, GuideLLM,
Hugging Face Hub, then install this repo editable:

```bash
conda create -n spec python=3.12
conda activate spec
pip install vllm guidellm huggingface-hub
pip install -e /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators --no-deps
```

GPU commands must run with real GPU access. In Codex sandboxed commands,
`torch.cuda.is_available()` may return false even when `nvidia-smi` sees the GPU;
run vLLM/GuideLLM commands with escalated GPU access.

## Models

Use local paths for speculator checkpoints. The Qwen3-8B base model defaults to
the Hugging Face ID and will use the local Hugging Face cache when available.

```bash
BASE_MODEL=Qwen/Qwen3-8B
EAGLE3_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator
PEAGLE_SPECULATOR_MODEL=/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator
```

The current local paths are:

- EAGLE3: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator`
- P-EAGLE: `/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator`

To download the speculator checkpoints again:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
./download_qwen3_8b_speculators.sh
```

The base model is not copied into `../models` by default. Override it with
`QWEN3_8B_MODEL=/path/to/local/qwen3-8b` only if you want to force a specific
local base model directory.

## P-EAGLE vLLM Compatibility

The `spec` environment's vLLM package has a local Python compatibility patch for
P-EAGLE speculators. It adds:

- `peagle` to `vllm.transformers_utils.configs.speculators.algos.SUPPORTED_SPECULATORS_TYPES`
- conversion from `speculators_model_type=peagle` to `method=eagle3` with
  `parallel_drafting=true`
- PEAGLE config fields required by the existing EAGLE3 parallel drafting path:
  `pard_token`, `draft_vocab_size`, `norm_before_residual`,
  `norm_before_fc`, and `eagle_aux_hidden_state_layer_ids`

The patch lives in the installed vLLM package under:

```text
/ACALAB/stu1/miniconda3/envs/spec/lib/python3.12/site-packages/vllm/transformers_utils/configs/speculators/
```

Original backups were saved as:

```text
algos.py.bak-peagle
base.py.bak-peagle
```

If vLLM is reinstalled, reapply this compatibility patch or switch to a vLLM
build that already supports `peagle`.

Expected PEAGLE config parse:

```text
supported ['dflash', 'eagle3', 'peagle']
architectures ['Eagle3LlamaForCausalLM']
pard_token 151669
aux_layers [2, 18, 33]
spec_config {'method': 'eagle3', 'num_speculative_tokens': 4, 'parallel_drafting': True}
```

## Dataset

For the current smoke benchmark, use the repo-local `math_reasoning.jsonl` file:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
mkdir -p data
hf download RedHatAI/speculator_benchmarks \
  --repo-type dataset \
  --include math_reasoning.jsonl \
  --local-dir data \
  --max-workers 8
```

Dataset path:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/data/math_reasoning.jsonl
```

The evaluation scripts also accept Hugging Face dataset syntax:

```text
RedHatAI/speculator_benchmarks:math_reasoning.jsonl
```

`scripts/run_guidellm.sh` strips the `path=` prefix emitted by this HF CLI
version before searching downloaded files.

## Running Benchmarks

Run commands from:

```bash
cd /ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm
```

By default, experiment outputs go under:

```text
/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/speculators/examples/evaluate/eval-guidellm/results/
```

`run_evaluation.sh` defaults to `results/eval_results_TIMESTAMP/`. Override the
root with either:

```bash
RESULTS_DIR=/path/to/results ./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env
```

or:

```bash
./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env --results-dir /path/to/results
```

EAGLE3 smoke:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-eagle3.env \
  -o ./results/out_eagle3_smoke_localds_math \
  --port 8010
```

P-EAGLE smoke:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_evaluation.sh \
  -c ./configs/qwen3-8b-peagle.env \
  -o ./results/out_peagle_smoke_localds_math \
  --port 8011
```

Qwen3-8B EAGLE3 vs P-EAGLE comparison:

```bash
GUIDELLM_RATE=1 REQUEST_TYPE=chat_completions conda run -n spec bash ./run_qwen3_8b_eagle3_vs_peagle.sh
```

The comparison script writes to:

```text
results/qwen3_8b_eagle3_vs_peagle_TIMESTAMP/eagle3/
results/qwen3_8b_eagle3_vs_peagle_TIMESTAMP/peagle/
```

The config defaults are:

- EAGLE3 uses `NUM_SPEC_TOKENS=3`
- P-EAGLE uses `NUM_SPEC_TOKENS=4`
- P-EAGLE sets `PARALLEL_DRAFTING=true`
- Both use `BASE_MODEL=Qwen/Qwen3-8B` by default
- Both use `DATASET=data/math_reasoning.jsonl` by default

Expected output files in each output directory:

- `vllm_server.log`
- `guidellm_output.log`
- `guidellm_results.json`
- `acceptance_analysis.txt`

## Current Run Notes

EAGLE3 completed successfully with the local dataset:

```text
Output dir: examples/evaluate/eval-guidellm/out_eagle3_smoke_localds_math
Output throughput: 1086.1 generated tokens/s
Total throughput: 1120.7 tokens/s
Weighted acceptance rates: [0.707 0.476 0.287]
```

P-EAGLE successfully started after the compatibility patch and produced vLLM
`SpecDecoding metrics`, including four-position acceptance rates. The run was
interrupted before GuideLLM wrote `guidellm_results.json`; rerun the P-EAGLE
smoke command above for a complete result file.

## Troubleshooting

If PEAGLE startup fails with:

```text
Expected one of: {'eagle3': ..., 'dflash': ...}
```

then the vLLM PEAGLE compatibility patch is missing or vLLM was reinstalled.

If vLLM fails with:

```text
Failed to infer device type
```

check CUDA visibility:

```bash
conda run -n spec python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

For Codex-run GPU commands, use escalated execution; non-escalated sandbox
commands may not expose `/dev/nvidia*`.

If the Hugging Face dataset download path starts with `path=`, use the patched
`scripts/run_guidellm.sh` or pass a local `.jsonl` path directly.
