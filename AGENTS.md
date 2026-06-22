# Speculator Throughput Experiments

This repo is currently set up to benchmark Qwen3-8B with EAGLE3/P-EAGLE and
Llama-3.1-8B with FastDraft, Smurfs dynamic K, and EAGLE3 through vLLM and
GuideLLM.

## Environment

Use the existing conda environment:

```bash
conda activate spec
```

Known working environment:

- Python: `${CONDA_PREFIX}/bin/python`
- vLLM: `0.20.0`, installed editable from this repo's `vllm/`
- GuideLLM: `0.6.0`
- Torch: `2.11.0+cu130`
- Local package: this repo installed editable into `spec`

If recreating the environment, use Python 3.12, install GuideLLM and Hugging
Face Hub, then install this repo and the vendored vLLM editable. Keep vLLM's
CUDA build aligned with PyTorch's CUDA 13.0 stack; do not build it with a newer
system CUDA such as `${SYSTEM_CUDA_HOME}`, because that can produce PTX the
current driver cannot run.

```bash
conda create -n spec python=3.12
conda activate spec
pip install guidellm huggingface-hub
pip install -e . --no-deps
pip install -r ./vllm/requirements/build/cuda.txt
pip install \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
conda install -n spec -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13

cd .
CU13=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13
LIBCUDA_SO="${LIBCUDA_SO:?set LIBCUDA_SO to the system libcuda.so.1 path}"
ln -sfn lib "${CU13}/lib64"
ln -sfn libcudart.so.13 "${CU13}/lib/libcudart.so"
ln -sfn libnvJitLink.so.13 "${CU13}/lib/libnvJitLink.so"
ln -sfn libnvrtc.so.13 "${CU13}/lib/libnvrtc.so"
ln -sfn libnvvm.so.4 "${CU13}/lib/libnvvm.so"
mkdir -p "${CU13}/lib/stubs"
ln -sfn ${LIBCUDA_SO} "${CU13}/lib/stubs/libcuda.so"

CUDA_HOME="${CU13}" \
CUDACXX="${CU13}/bin/nvcc" \
CUDAHOSTCXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc \
CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
PATH="${CU13}/bin:${PATH}" \
LD_LIBRARY_PATH="${CU13}/lib:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0 \
  pip install -e ./vllm --no-deps --no-build-isolation --force-reinstall
```

GPU commands must run with real GPU access. In Codex sandboxed commands,
`torch.cuda.is_available()` may return false even when `nvidia-smi` sees the GPU;
run vLLM/GuideLLM commands with escalated GPU access.

## Models

Use local paths for speculator checkpoints. The Qwen3-8B base model defaults to
the Hugging Face ID and will use the local Hugging Face cache when available.

```bash
BASE_MODEL=Qwen/Qwen3-8B
EAGLE3_SPECULATOR_MODEL=../models/qwen3-8b-eagle3-speculator
PEAGLE_SPECULATOR_MODEL=../models/qwen3-8b-peagle-speculator
```

The current local paths are:

- EAGLE3: `../models/qwen3-8b-eagle3-speculator`
- P-EAGLE: `../models/qwen3-8b-peagle-speculator`
- Llama-3.1-8B base: `../models/llama-3.1-8b-instruct`
- Llama-3.1-8B FastDraft: `../models/llama-3.1-8b-fastdraft-150m-int8-hf`
- Llama-3.1-8B EAGLE3: `../models/llama-3.1-8b-eagle3-speculator`

To download the speculator checkpoints again:

```bash
cd examples/evaluate/eval-guidellm
./download_qwen3_8b_speculators.sh
./download_llama_3_1_8b_eagle3_speculator.sh
```

The base model is not copied into `../models` by default. Override it with
`QWEN3_8B_MODEL=../models/qwen3-8b` only if you want to force a specific
local base model directory.

## Vendored vLLM Source and Install

vLLM is vendored as a full source tree inside this repository:

```text
./vllm
```

This tree was imported from upstream vLLM `v0.20.0` and pushed to `main` in
commit `9def519`. Future vLLM edits for these experiments should be made in
`speculators/vllm`, not in `site-packages` and not in the older external
`../vllm` checkout.

Current local vLLM changes in `speculators/vllm` are Python-only:

- `peagle` support in
  `vllm/transformers_utils/configs/speculators/algos.py`
- conversion from `speculators_model_type=peagle` to `method=eagle3` with
  `parallel_drafting=true` in
  `vllm/transformers_utils/configs/speculators/base.py`
- motivation-breakdown instrumentation in
  `vllm/v1/worker/gpu_model_runner.py`, gated by
  `SPECLINK_BREAKDOWN=1`, `SPECLINK_BREAKDOWN_OUT`,
  `SPECLINK_BREAKDOWN_ALGO`, `SPECLINK_BREAKDOWN_BATCH_SIZE`, and
  `SPECLINK_BREAKDOWN_NUM_SPEC_TOKENS`
- Qwen3 verifier-detail instrumentation in
  `vllm/model_executor/models/qwen3.py` and
  `vllm/speclink_breakdown.py`, gated by
  `SPECLINK_BREAKDOWN_VERIFY_DETAIL=1`
- confidence/acceptance tracing for the SpecLink first validation experiment in
  `vllm/v1/spec_decode/llm_base_proposer.py`,
  `vllm/v1/sample/rejection_sampler.py`,
  `vllm/v1/worker/gpu_model_runner.py`, and
  `vllm/speclink_confidence_trace.py`, gated by
  `SPECLINK_TRACE_CONFIDENCE=1`
- SpecLink-CV live prefix/suffix verification slice in
  `vllm/speclink_cv.py`,
  `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/sched/output.py`, and
  `vllm/v1/worker/gpu_model_runner.py`, gated by
  `SPECLINK_CV_ENABLE=1`
- Smurfs dynamic K for FastDraft is implemented inside this repo's vendored
  vLLM in
  `vllm/smurfs_dynamic.py`,
  `vllm/v1/spec_decode/llm_base_proposer.py`, and
  `vllm/v1/core/sched/scheduler.py`, gated by
  `SPECLINK_SMURFS_DYNAMIC_ENABLE=1`

The old standalone `Others/Smurfs` tree has been removed and is not used for
current runs. Do not route Smurfs benchmarks through a standalone Smurfs
checkout or a separate Smurfs conda environment. Use the `spec` environment and
the vendored `speculators/vllm` install.

Install or refresh vLLM from the repo root with editable mode. The current
machine uses PyTorch `2.11.0+cu130`, so point the vLLM build at the conda
environment's CUDA 13.0 compiler stack instead of the system CUDA:

```bash
cd .
conda run -n spec python -m pip install -r vllm/requirements/build/cuda.txt
conda run -n spec python -m pip install \
  nvidia-cuda-nvcc==13.0.88 \
  nvidia-nvvm==13.0.88 \
  nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85
conda install -n spec -y -c conda-forge gcc_linux-64=13 gxx_linux-64=13

CU13=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13
LIBCUDA_SO="${LIBCUDA_SO:?set LIBCUDA_SO to the system libcuda.so.1 path}"
ln -sfn lib "${CU13}/lib64"
ln -sfn libcudart.so.13 "${CU13}/lib/libcudart.so"
ln -sfn libnvJitLink.so.13 "${CU13}/lib/libnvJitLink.so"
ln -sfn libnvrtc.so.13 "${CU13}/lib/libnvrtc.so"
ln -sfn libnvvm.so.4 "${CU13}/lib/libnvvm.so"
mkdir -p "${CU13}/lib/stubs"
ln -sfn ${LIBCUDA_SO} "${CU13}/lib/stubs/libcuda.so"

CUDA_HOME="${CU13}" \
CUDACXX="${CU13}/bin/nvcc" \
CUDAHOSTCXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc \
CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++ \
PATH="${CU13}/bin:${PATH}" \
LD_LIBRARY_PATH="${CU13}/lib:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=8 \
NVCC_THREADS=2 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0 \
  conda run -n spec python -m pip install -e ./vllm --no-deps --no-build-isolation --force-reinstall
```

`pip install -e` is the modern replacement for `setup.py develop`. Python edits
under `speculators/vllm/vllm/` take effect on the next process start. C++/CUDA
extension edits still require rerunning the editable install, but CMake/Ninja
can reuse existing build artifacts if `vllm/.deps` and the build cache are kept.

Verify from a directory other than the speculators repo root, for example:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve())"
conda run -n spec vllm --help
```

Expected import path prefix:

```text
./vllm/vllm/
```

Do not use `VLLM_USE_PRECOMPILED=1` for this workflow. The point of the local
source install is that future edits to `speculators/vllm` are the code being
run. Also avoid launching Python from the speculators repo root when checking
`import vllm`: the root contains a `vllm/` source directory and can shadow the
editable package as a namespace package. The motivation breakdown script changes
into `examples/evaluate/eval-guidellm` before validating the import path.

Older runs left direct-edit residue under:

```text
${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/
```

That directory should not exist after the editable install. If it reappears and
`vllm.__file__` is `None` or imports resolve to `site-packages/vllm`, move the
stale directory out of the conda environment and rerun the verification above.

If the install fails after moving a partially built `vllm/.deps` directory and
CMake reports an old path in `CMakeCache.txt`, remove only generated build
subdirectories and retry:

```bash
find vllm/.deps -maxdepth 1 -type d \( -name '*-subbuild' -o -name '*-build' \) -exec rm -rf {} +
CU13=${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13
CUDA_HOME="${CU13}" CUDACXX="${CU13}/bin/nvcc" TORCH_CUDA_ARCH_LIST=12.0 \
MAX_JOBS=8 NVCC_THREADS=2 SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0 \
  conda run -n spec python -m pip install -e ./vllm --no-deps --no-build-isolation --force-reinstall
```

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
cd examples/evaluate/eval-guidellm
mkdir -p data
hf download RedHatAI/speculator_benchmarks \
  --repo-type dataset \
  --include math_reasoning.jsonl \
  --local-dir data \
  --max-workers 8
```

Dataset path:

```text
examples/evaluate/eval-guidellm/data/math_reasoning.jsonl
```

The evaluation scripts also accept Hugging Face dataset syntax:

```text
RedHatAI/speculator_benchmarks:math_reasoning.jsonl
```

`scripts/run_guidellm.sh` strips the `path=` prefix emitted by this HF CLI
version before searching downloaded files.

## Structured 2:4 C4 Calibration

Structured 2:4 quality and layer-sensitivity experiments use a reusable C4
activation-RMS cache for activation-aware masking. Do not implicitly calibrate
from the evaluation datasets. The fixed prompt sample is:

```text
examples/evaluate/eval-guidellm/data/c4_calibration/c4_calibration_512_seed42.jsonl
```

Prepare or refresh the fixed C4 prompt sample with:

```bash
cd .
conda run -n spec python examples/evaluate/eval-guidellm/scripts/prepare_c4_calibration_dataset.py \
  --num-examples 512 \
  --seed 42 \
  --shuffle-buffer 10000 \
  --output examples/evaluate/eval-guidellm/data/c4_calibration/c4_calibration_512_seed42.jsonl \
  --force
```

Run model-specific calibration once, then reuse the generated RMS cache for
later experiments:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/residual_24_feasibility.py calibrate-24 \
  --models qwen3_8b,llama3_1_8b \
  --calibration-prompts examples/evaluate/eval-guidellm/data/c4_calibration/c4_calibration_512_seed42.jsonl \
  --calibration-max-seq-len 512 \
  --calibration-batch-size 1 \
  --dtype bf16 \
  --output-root examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512
```

The default cache root is:

```text
examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512
```

Run the current accuracy-first speculative serving comparison with token-dense
thresholds only:

```bash
cd .
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_token_dense_accuracy.py \
  --models qwen3_8b,llama3_1_8b \
  --methods token_dense_t00,token_dense_t01,token_dense_t02,token_dense_t03,token_dense_t04,token_dense_t05,token_dense_t06,token_dense_t07,token_dense_t08,token_dense_t09,token_dense_t10 \
  --datasets gsm8k,math_reasoning \
  --gsm8k-num-examples 64 \
  --math-num-examples 64 \
  --accuracy-max-tokens 512 \
  --accuracy-concurrency 8 \
  --max-num-seqs 8 \
  --num-spec-tokens 8 \
  --output-root examples/evaluate/eval-guidellm/results/token_dense_accuracy_TIMESTAMP
```

`token_dense_tXX` is the current token-level routing experiment. It keeps the
TLM/base weights dense, attaches the same activation-aware 2:4 masks to Llama
target linears, records DLM `draft_selected_prob`, and during target
verification routes only low-confidence draft-token rows through the 2:4
masked weight. High-confidence draft rows, prefill rows, non-draft rows,
missing-score rows, and verifier bonus rows stay dense. The label threshold is
decimal shorthand: `token_dense_t07` means threshold 0.7, `token_dense_t05`
means 0.5, and `token_dense_t90` also means 0.90. This mode forces
`--enforce-eager` because the token mask changes per decoding step.

To test the low-overhead sensitive-layer preservation path, append
`_keep_first_N` to a method name. For example, the command below keeps the
first two transformer layers dense and applies the selected 2:4 method to all
remaining target-model layers:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_token_dense_accuracy.py \
  --methods activation_aware,activation_aware_keep_first_2,token_dense_t00,token_dense_t01,token_dense_t02,token_dense_t03,token_dense_t04,token_dense_t05,token_dense_t06,token_dense_t07,token_dense_t08,token_dense_t09,token_dense_t10 \
  --output-root examples/evaluate/eval-guidellm/results/token_dense_accuracy_TIMESTAMP \
  --resume
```

For LMeval accuracy, use the local wrapper that starts the same vLLM serving
path and then calls EleutherAI lm-evaluation-harness:

```bash
cd .
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode all \
  --task smoke \
  --limit 4 \
  --output-dir examples/evaluate/eval-guidellm/results/token_dense_llama_math_gsm8k_thresholds16_20260617_145507/lm_eval
```

`--mode all` expands to `dense_ar`, `eagle3_dense`, `activation_aware`, and
`token_dense_t00` through `token_dense_t10`. For LogiQA, use
`agieval_logiqa_en` for the official multiple-choice slot and
`logiqa_generative` for the generate-until path; the older built-in `logiqa`
task depends on a dataset script that the current `datasets` package rejects.

For the current long-output lm-eval TODO, do not use the older LogiQA/MMLU/
HotpotQA task set. On a single RTX 5090, run only `qwen3_8b` and
`llama3_1_8b`; do not run 14B/32B/70B on this machine. The formal task set is:

```text
gsm8k_cot,minerva_math500,gpqa_diamond_cot_zeroshot,ifeval,humaneval_instruct,longbench_multi_news
```

Use fixed lm-eval sample manifests for reproducibility:

```bash
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode all \
  --task gsm8k_cot,minerva_math500,gpqa_diamond_cot_zeroshot,ifeval,humaneval_instruct,longbench_multi_news \
  --models qwen3_8b,llama3_1_8b \
  --use-task-manifests \
  --manifest-size 200 \
  --max-new-tokens 512 \
  --num-spec-tokens 8 \
  --batch-size 1 \
  --num-concurrent 1 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.94 \
  --allow-unsafe-code \
  --humaneval-sandbox auto \
  --resume \
  --output-dir examples/evaluate/eval-guidellm/results/token_dense_lm_eval_long_output_5090x1_TIMESTAMP
```

The wrapper writes manifests to
`examples/evaluate/eval-guidellm/configs/task_manifests/`, uses repo-local
Hugging Face/evaluate caches under `examples/evaluate/eval-guidellm/temp/`, and
sets `SPECLINK_TOKEN_DENSE_STATS_INTERVAL=1` for token-dense modes so routing
fractions appear in `token_dense_stats.jsonl`. `run_meta.json` records
`started_at`, `ended_at`, and `elapsed_seconds` for each case. GPQA is a gated
Hugging Face dataset; the wrapper preflights
`Idavidrein/gpqa:gpqa_diamond` before launching vLLM and records a failed
`run_meta.json` if the available token lacks access. HumanEval must run with
`--allow-unsafe-code`; on this machine `bwrap` is available only with
non-sandboxed execution permissions, so GPU/lm-eval runs must be launched with
real execution permissions rather than Codex's default sandbox.

Smoke and sanity outputs for this path should go under `results.bak/`, for
example:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_token_dense_accuracy.py --smoke
```

`quality` and `layer-sensitivity` load this cache by default when they need
activation-aware RMS. If the cache is missing, generate it with `calibrate-24`
instead of falling back to evaluation prompts. A standard C4-calibrated layer
sensitivity run is:

```bash
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/residual_24_feasibility.py layer-sensitivity \
  --models qwen3_8b,llama3_1_8b \
  --datasets mtbench,dolly,gsm8k,math_reasoning \
  --mtbench-num-examples 40 \
  --dolly-num-examples 64 \
  --gsm8k-num-examples 64 \
  --math-num-examples 64 \
  --generation-batch-size 4 \
  --dtype bf16 \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_layer_sensitivity_qwen_llama_c4calib_TIMESTAMP
```

The command above is an offline Transformers quality check. For current
speculative-inference experiments, use the vLLM/EAGLE3 runner below instead.
It applies 2:4 only to the TLM/base large model at vLLM model-load time; the
EAGLE3 drafter/speculator remains dense. Accuracy is generated through vLLM
speculative decoding, while PPL is dense-vs-sparse TLM reference loss using the
same mask policy.

```bash
cd .
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets mtbench,dolly,gsm8k,math_reasoning \
  --num-spec-tokens 8 \
  --mtbench-num-examples 40 \
  --dolly-num-examples 64 \
  --gsm8k-num-examples 64 \
  --math-num-examples 64 \
  --accuracy-concurrency 8 \
  --max-num-seqs 8 \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_spec_tlm_eagle3_k8_TIMESTAMP
```

Useful variants:

```bash
# Smoke and sanity artifacts go under results.bak/.
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py --smoke

# Resume an interrupted full run.
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_spec_tlm_eagle3_k8_TIMESTAMP \
  --resume

# Re-run only the all-sparse and first/last dense-keep comparison.
MPLCONFIGDIR=temp/matplotlib conda run -n spec python -u \
  examples/evaluate/eval-guidellm/scripts/run_structured_24_spec_quality.py \
  --skip-layer-sensitivity \
  --output-root examples/evaluate/eval-guidellm/results/structured_24_spec_tlm_eagle3_k8_dense_keep_TIMESTAMP
```

The vLLM hook is disabled by default and is controlled by:

```text
SPECLINK_STRUCTURED_24_ENABLE=1
SPECLINK_STRUCTURED_24_MODEL_LABEL=qwen3_8b|llama3_1_8b
SPECLINK_STRUCTURED_24_CALIBRATION_CACHE_ROOT=examples/evaluate/eval-guidellm/data/c4_calibration/activation_rms/c4_512_seed42_bf16_max512
SPECLINK_STRUCTURED_24_POLICY=dense|single_layer|all_sparse|keep_first|keep_last|keep_first_last
SPECLINK_STRUCTURED_24_LAYER_INDEX=0
SPECLINK_STRUCTURED_24_KEEP_N=1
SPECLINK_STRUCTURED_24_STATS_PATH=temp/vllm_structured_24_stats.json
SPECLINK_TOKEN_DENSE_ENABLE=1
SPECLINK_TOKEN_DENSE_MODE=high_confidence_dense
SPECLINK_TOKEN_DENSE_THRESHOLD=0.7
SPECLINK_TOKEN_DENSE_STATS_PATH=temp/token_dense_stats.jsonl
```

Final files from the speculative runner:

- `structured_24_spec_quality.csv`: all dense-vs-sparse PPL and ACC rows.
- `layer_sensitivity.csv`: one-sparse-layer rows.
- `dense_keep_compare.csv`: all-sparse, keep-first, keep-last, and
  keep-first-last rows.
- `figures/layer_sensitivity_spec.png` and `figures/dense_keep_spec.png`.
- `runs/*/*/vllm_structured_24_stats.json`: vLLM-side proof that only TLM
  modules were masked.

## Running Benchmarks

Run commands from:

```bash
cd examples/evaluate/eval-guidellm
```

By default, experiment outputs go under:

```text
examples/evaluate/eval-guidellm/results/
```

`run_evaluation.sh` defaults to `results/eval_results_TIMESTAMP/`. Override the
root with either:

```bash
RESULTS_DIR=results/custom ./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env
```

or:

```bash
./run_evaluation.sh -c ./configs/qwen3-8b-peagle.env --results-dir results/custom
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

## Llama FastDraft/Smurfs Matrix

For the Llama-3.1-8B comparison of AR, FastDraft, Smurfs dynamic K, and EAGLE3,
use:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python ./scripts/run_llama31_vllm_fastdraft_smurfs_eagle3_matrix.py
```

The matrix script compares:

- `vllm_ar`
- `vllm_fastdraft`
- `smurfs_fastdraft`
- `vllm_eagle3`

Defaults:

- datasets: `math_reasoning,mtbench,gsm8k,humaneval`
- client-side batch/concurrency: `8,16,32,64`
- `max_tokens=2048`
- FastDraft K=4
- EAGLE3 K=4
- Smurfs initial K=4, dynamic max K=12 below batch size 32 and 8 otherwise

Final outputs go to:

```text
examples/evaluate/eval-guidellm/results_final/llama31_vllm_fastdraft_smurfs_eagle3_matrix_2048_full_TIMESTAMP/
```

Intermediate server logs and per-run work files go to:

```text
examples/evaluate/eval-guidellm/temp/llama31_vllm_fastdraft_smurfs_eagle3_matrix_2048_full_TIMESTAMP/
```

For Smurfs runs, the server writes the live dynamic-K event log to each run
directory as `smurfs_dynamic_k.jsonl` via `SPECLINK_SMURFS_DYNAMIC_OUT`. The
WebUI Smurfs mode in `vllm_spec_webui.py` reads that log through
`--smurfs-k-log` and displays the current K.

## Motivation Breakdown

`motivation_breakdown.sh` runs the synthetic 1000-token prompt / 1000-token
output experiment requested for EAGLE3 and P-EAGLE:

- `batch_size=1 2 4 8 16`
- `NUM_SPEC_TOKENS=8 16 24`
- default `REQUESTS_PER_RUN=32`
- default `WARMUP_REQUESTS=4`
- default `MAX_NUM_BATCHED_TOKENS=8192`
- vLLM `--max-num-seqs` is set to the current batch/concurrency size
- default `SPECLINK_BREAKDOWN_VERIFY_DETAIL=1`, which also passes
  `--enforce-eager` to vLLM for this experiment

Run it from `examples/evaluate/eval-guidellm`:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec bash ./motivation_breakdown.sh
```

The script no longer accepts `--vllm-dir`, `--skip-vllm-setup`, or
`--setup-only`; vLLM is expected to already be installed editable from
`speculators/vllm`. At startup it verifies that `import vllm` resolves under:

```text
./vllm/vllm/
```

Outputs are written under `results/motivation_breakdown_TIMESTAMP/`, including:

- `status.tsv`
- per-run `vllm_server.log`, `guidellm_output.log`,
  `guidellm_results.json`, and `breakdown_events.jsonl`
- `concise_summary.csv`: the preferred compact result table. It keeps only
  model, batch size, `NUM_SPEC_TOKENS`, decode-stage verify/draft/other
  percentages, generated tokens per decode iteration, and end-to-end mean
  latency.
- `verify_detail_summary.csv`: Qwen3 verifier-only QKV projection, Attention,
  FFN, and verifier-other percentages and per-iteration times. Attention here
  includes q/k norm, RoPE, the attention kernel, and `o_proj`.
- `summary.csv`
- `raw_events.csv`
- `acceptance.csv`
- `motivation_breakdown.xlsx`, with `concise_summary` as the first sheet
- `motivation_breakdown.svg`
- `motivation_verify_breakdown.svg`

For P-EAGLE with `NUM_SPEC_TOKENS=16` or `24`, keep the scheduler budget large
enough. The default `MAX_NUM_BATCHED_TOKENS=8192` is intentional; vLLM otherwise
can fail during startup with `max_num_scheduled_tokens is set to ...`, because
parallel drafting reserves additional draft-token slots. The script also checks
that GuideLLM wrote a result JSON and that vLLM wrote breakdown events before it
marks a run as `ok`.

Set `SPECLINK_BREAKDOWN_VERIFY_DETAIL=0` to disable Qwen3 verify-detail
instrumentation. The detail mode uses CUDA events inside Qwen3 verifier layers
and is meant for breakdown analysis, not for clean throughput-only numbers.

## Confidence Acceptance Experiment

`run_speclink_confidence_acceptance.sh` tests whether DLM draft-token confidence
predicts TLM local acceptance. It does not implement chunked verification
scheduling.

The vLLM trace is off by default and is enabled only by:

```bash
SPECLINK_TRACE_CONFIDENCE=1
SPECLINK_TRACE_OUTPUT=temp/trace.jsonl
SPECLINK_TRACE_RUN_ID=qwen3_8b_eagle3_k4
SPECLINK_TRACE_DATASET_LABEL=math
SPECLINK_TRACE_MODEL_LABEL=qwen3_8b
SPECLINK_TRACE_METHOD=eagle3
SPECLINK_TRACE_NUM_SPEC_TOKENS=4
```

The hooks collect proposer logits in
`vllm/v1/spec_decode/llm_base_proposer.py`, buffer per-request draft records in
`vllm/speclink_confidence_trace.py`, and attach acceptance labels from
`vllm/v1/sample/rejection_sampler.py`. Records are aligned by vLLM request id
plus a per-request speculative-step counter. `dataset_label` separates math and
MTBench, and `model_label` separates Qwen and Llama traces in combined
analysis. `token_text` is left null to avoid tokenizer overhead in the hot path.

Run from `examples/evaluate/eval-guidellm`:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec bash ./run_speclink_confidence_acceptance.sh
```

The default command is the one-click reproduction for the final four-row report:
EAGLE3 `NUM_SPEC_TOKENS=8` on `{qwen3_8b,llama3_1_8b} x {math,mtbench}`. It
runs the four individual cases under:

```text
temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/
```

and writes only the combined final report under:

```text
results/speclink_confidence_acceptance_datasets_TIMESTAMP/
```

Defaults:

- reproduction: EAGLE3 only, `REPRO_NUM_SPEC_TOKENS=8`,
  `REPRO_PROMPTS=80`, `REPRO_MAX_TOKENS=128`, `temperature=0`
- request concurrency defaults to `REPRO_REQUEST_CONCURRENCY=1`
- vLLM is launched with `--enforce-eager` for trace stability
- Qwen uses port `QWEN_PORT=8036`; Llama uses `LLAMA_PORT=8037`
- use `--single-case` for the older configurable single-model/single-dataset
  path

MTBench setup:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec python ./prepare_mt_bench_dataset.py --force
```

This downloads the official FastChat MTBench `question.jsonl` and writes:

```text
data/mt_bench_raw.jsonl
data/mt_bench.jsonl
```

The converted file has 80 rows. Multi-turn MTBench prompts are serialized as
`User turn N:` blocks followed by `Assistant:` so the completions endpoint can
be used consistently with the math dataset.

Useful variants:

```bash
# Preview the four final cases without launching vLLM
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --dry-run

# Regenerate parsed CSV, calibration, figures, and report for an existing final run
conda run -n spec bash ./run_speclink_confidence_acceptance.sh \
  --analyze-only ./results/speclink_confidence_acceptance_TIMESTAMP

# Short main run for debugging
SPECLINK_SINGLE_CASE=1 MAIN_PROMPTS=128 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Smoke only for one configurable case, written under temp/
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --smoke-only

# Llama-3.1-8B EAGLE3 K=8 only
MODEL_LABEL=llama3_1_8b \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
EAGLE3_SPECULATOR_MODEL=../models/llama-3.1-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 PORT=8035 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Qwen3-8B MTBench, EAGLE3 K=8 only
DATASET_LABEL=mtbench \
DATASET=examples/evaluate/eval-guidellm/data/mt_bench.jsonl \
MODEL_LABEL=qwen3_8b \
BASE_MODEL=Qwen/Qwen3-8B \
EAGLE3_SPECULATOR_MODEL=../models/qwen3-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 MAIN_PROMPTS=80 MAIN_MAX_TOKENS=128 \
REQUEST_CONCURRENCY=1 PORT=8036 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Llama-3.1-8B MTBench, EAGLE3 K=8 only
DATASET_LABEL=mtbench \
DATASET=examples/evaluate/eval-guidellm/data/mt_bench.jsonl \
MODEL_LABEL=llama3_1_8b \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
EAGLE3_SPECULATOR_MODEL=../models/llama-3.1-8b-eagle3-speculator \
METHODS=eagle3 MAIN_NUM_SPEC_TOKENS=8 MAIN_PROMPTS=80 MAIN_MAX_TOKENS=128 \
REQUEST_CONCURRENCY=1 PORT=8037 \
conda run -n spec bash ./run_speclink_confidence_acceptance.sh --single-case --main-only

# Combine archived/intermediate case roots into one K=8 report
conda run -n spec python ./scripts/combine_speclink_confidence_results.py \
  --output-root ./results/speclink_confidence_acceptance_datasets_TIMESTAMP \
  --source qwen3_8b:math:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/math_qwen3_8b_eagle3_k8 \
  --source llama3_1_8b:math:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/math_llama3_1_8b_eagle3_k8 \
  --source qwen3_8b:mtbench:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/mtbench_qwen3_8b_eagle3_k8 \
  --source llama3_1_8b:mtbench:./temp/speclink_confidence_acceptance_reproduce_TIMESTAMP/mtbench_llama3_1_8b_eagle3_k8 \
  --method eagle3 \
  --num-spec-tokens 8 \
  --analyze
```

Final outputs include:

- `commands.sh`
- `repro_report.md` for the default four-way combined report
- per-case `env_report.md` files under the corresponding temp work root
- `trace/DATASET_LABEL_MODEL_LABEL_METHOD_trace.jsonl`
- `parsed/*_token_level.csv`, `parsed/*_sanity.md`
- `calibration/*_calibrated.csv`, `calibration/*_summary.json`,
  `calibration/*_model_params.json`
- `figures/acceptance_by_position.png`, `confidence_bins.png`,
  `calibration_curve.png`, `confidence_fit_curve.png`, `reliability.png`,
  `reject_within_h.png`, `chunk_benefit.png`, plus the CSV data used to draw
  each figure. `confidence_fit_curve.png` uses dataset/model facets for combined
  K=8 reports so the actual-vs-fit curves remain readable without a legend.
- `summary.csv`, `summary.json`, `report.md`

The analysis script uses only installed lightweight dependencies:
`numpy`, `pandas`, and `PIL`. The current `spec` env does not include
`sklearn` or `matplotlib`, so logistic regression and calibration metrics are
implemented locally in
`scripts/analyze_speclink_confidence_acceptance.py`.

Current combined K=8 confidence/acceptance result:

```text
results/speclink_confidence_acceptance_datasets_20260525_200725/
```

It contains only EAGLE3 `NUM_SPEC_TOKENS=8` rows for
`{qwen3_8b,llama3_1_8b} x {math,mtbench}`. Key summary values:

```text
math/llama3_1_8b:    acceptance=0.5918, AUROC=0.8603, Spearman=0.6079
math/qwen3_8b:       acceptance=0.6201, AUROC=0.8208, Spearman=0.5296
mtbench/llama3_1_8b: acceptance=0.4787, AUROC=0.8218, Spearman=0.5573
mtbench/qwen3_8b:    acceptance=0.5599, AUROC=0.8135, Spearman=0.5403
```

Older confidence/acceptance case roots from the same development pass were
moved out of `results/` to:

```text
temp/moved_from_results_20260525_203200/
```

## Acceptance Jitter Experiment

`run_acceptance_jitter.sh` is the one-command reproduction for the accepted
draft-token count jitter figure. It uses normal vLLM speculative decoding with
`SPECLINK_TRACE_CONFIDENCE=1`; it does not use scheduler-level chunking or any
`SPECLINK_CHUNK_*` path.

Run from `examples/evaluate/eval-guidellm`:

```bash
cd examples/evaluate/eval-guidellm
conda run -n spec bash ./run_acceptance_jitter.sh
```

Default matrix:

- cases: `qwen3_8b:peagle qwen3_8b:eagle3 llama3_1_8b:eagle3`
- `NUM_SPEC_TOKENS_LIST="8 12 16"`
- workloads: `math mtbench synthetic_1000x1000`
- prompt counts: math=80, MTBench=80, synthetic=8
- real datasets use `max_tokens=128`
- synthetic uses exact token-id prompts with `SYNTHETIC_PROMPT_TOKENS=1000`
  and `SYNTHETIC_MAX_TOKENS=1000`

Final outputs are written under:

```text
results/accepted_count_jitter_TIMESTAMP/
```

Intermediate vLLM logs, responses, and raw trace JSONL files are written under:

```text
temp/accepted_count_jitter_work_TIMESTAMP/
```

Key final files:

- `step_level_acceptance.csv`: one row per decode step with
  `num_accepted`.
- `summary.csv`: mean/std of accepted tokens, `P(accepted < 2)`,
  `P(accepted < 4)`, full-prefix acceptance rate, and jitter metrics by
  workload/case/K.
- `accepted_count_distribution.csv`: empirical accepted-count distribution.
- `figures/math_accepted_count_jitter.png`
- `figures/mtbench_accepted_count_jitter.png`
- `figures/synthetic_1000x1000_accepted_count_jitter.png`
- `report.md`

Useful commands:

```bash
# Preview all 27 cases without launching vLLM
conda run -n spec bash ./run_acceptance_jitter.sh --dry-run

# Fast smoke into temp/
conda run -n spec bash ./run_acceptance_jitter.sh --smoke-only

# Re-analyze an existing intermediate work root into a final results dir
conda run -n spec bash ./run_acceptance_jitter.sh \
  --analyze-only ./temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root ./results/accepted_count_jitter_TIMESTAMP
```

## Threshold Tradeoff Analysis

`scripts/run_threshold_tradeoff.sh` is a one-command experiment for confidence
threshold tradeoff:
1. run `run_acceptance_jitter.sh` (default workloads are `math,mtbench`, no
   synthetic case),
2. run `scripts/threshold_tradeoff.py` to compute the tradeoff.

`scripts/threshold_tradeoff.py` is an offline analysis over the confidence trace
and also does not launch vLLM.

The confidence rule is prefix sequence confidence:

```text
confidence(h) = product(draft_selected_prob[1:h])
```

For each threshold, the predicted token count is the largest prefix length whose
confidence stays above the threshold. If the first draft token is already below
the threshold, the prediction is 0.

Primary metrics:

- `error_probability`: `P(pred_tokens > actual_accept_tokens)` (overrun risk).
- `mismatch_probability`: `P(pred_tokens != actual_accept_tokens)`.
- `compute_efficiency`: `E(pred_tokens / K)`.

Pareto-optimal thresholds are those not dominated by another threshold with both
lower-or-equal `error_probability` and higher-or-equal `compute_efficiency`.

Run from `examples/evaluate/eval-guidellm`:

```bash
conda run -n spec bash ./scripts/run_threshold_tradeoff.sh
```

Analyze an existing trace root:

```bash
conda run -n spec python ./scripts/threshold_tradeoff.py \
  ./temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root ./results/threshold_tradeoff_TIMESTAMP
```

Same offline command with custom cases and thresholds:

```bash
conda run -n spec python ./scripts/threshold_tradeoff.py \
  ./temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root ./results/threshold_tradeoff_TIMESTAMP \
  --workloads math,mtbench \
  --models qwen3_8b,llama3_1_8b \
  --methods eagle3,peagle \
  --num-spec-tokens 8,12,16 \
  --thresholds 0.05,0.10,0.20,0.30,0.40,0.50
```

```bash
conda run -n spec bash ./scripts/run_threshold_tradeoff.sh \
  --workloads math,mtbench \
  --cases qwen3_8b:eagle3,llama3_1_8b:eagle3 \
  --num-spec-tokens 8,16 \
  --thresholds 0.05,0.10,0.15,0.20,0.30,0.40
```

Outputs:

- `threshold_tradeoff.csv`: all threshold points.
- `pareto_thresholds.csv`: non-dominated threshold points.
- `figures/threshold_tradeoff.png`
- `report.md`

## SpecLink-CV Trace Milestone

`tools/speclink_cv/` contains the current SpecLink-CV experiment scaffold:
chunk-size decision logic, request state machine, async verification queue,
roofline-packing policy, confidence calibration tools, unit tests, and a
trace-based experiment runner.

Run the current milestone from the repo root:

```bash
cd .
conda run -n spec python -m tools.speclink_cv.run_trace_experiment \
  --trace-root examples/evaluate/eval-guidellm/temp/accepted_count_jitter_work_TIMESTAMP \
  --output-root examples/evaluate/eval-guidellm/results/speclink_cv_TIMESTAMP \
  --workloads math,mtbench
```

The runner writes the requested result tree with:

- `00_env/env_report.{md,json}`
- `02_unit_tests/unit_test_summary.{md,csv,json}`
- `03_confidence_calibration/`
- `04_baselines/`
- `05_cv_ablation/cv_ablation_summary.csv`
- `06_scheduler_queue/`
- `07_roofline_packing/`
- `08_figures/cv_trace_tradeoff.png`
- `09_reports/SPECLINK_CV_REPORT.md`
- top-level `summary_metrics.{csv,json}`

Important limitation: this is an exact trace-level simulation over existing
one-shot EAGLE3 verification labels. The trace runner itself does not execute a
serving benchmark and must not be used to claim end-to-end throughput or latency
speedup. The separate live vLLM slice below changes scheduled speculative-token
shapes before target logits are computed, and the GuideLLM matrix runner records
real serving metrics for the live implementation.

## SpecLink-CV Live vLLM Slice

There is now a gated live vLLM implementation slice for fixed-half chunked
verification. It is intentionally narrow:

- enabled only with `SPECLINK_CV_ENABLE=1`
- requires the regular vLLM V1 scheduler path with vLLM's own
  `async_scheduling` disabled; the GuideLLM matrix runner automatically adds
  `--no-async-scheduling` for `cv_*` methods
- uses fixed half when confidence sizing is off, e.g. K=8 -> h=4
- when confidence sizing is on, carries proposal-time
  `draft_selected_prob` from the EAGLE3 drafter to the scheduler; without a
  calibration path it uses this as an uncalibrated local-acceptance proxy, and
  with `SPECLINK_CV_CALIBRATION_PATH` it applies the binning calibration model
  produced by `tools.speclink_cv.calibrate_acceptance`
- if the prefix rejects, it skips the suffix draft tokens
- if the prefix fully accepts, it masks the normal speculative bonus token,
  rolls scheduler progress back by that discarded bonus, and schedules the
  suffix for exact TLM verification
- when `SPECLINK_CV_ROOFLINE_PACKING=1`, estimates whether the current prefix
  chunk launch is underfilled using token/sequence budget utilization; if it is
  below `SPECLINK_CV_UTIL_THRESHOLD`, it falls back to exact one-shot
  verification for that step instead of running a small prefix chunk
- when `SPECLINK_CV_ASYNC_QUEUE=1`, prefix chunks enter a live scheduler queue
  before dispatch. The queue uses selected benefit, age, token budget, sequence
  budget, and roofline utilization to decide which queued prefix chunks run in
  the current scheduler step. Age timeout prevents starvation. This is a
  conservative first live queue, not the final full cross-request packing design.

Current environment variables:

```text
SPECLINK_CV_ENABLE=1
SPECLINK_CV_CONFIDENCE_SIZING=0
SPECLINK_CV_ASYNC_QUEUE=0
SPECLINK_CV_ROOFLINE_PACKING=0
SPECLINK_CV_CANDIDATE_CHUNKS=1,2,4,6,8,full
SPECLINK_CV_DEFAULT_HALF_POLICY=floor
SPECLINK_CV_MIN_BENEFIT=0.0
SPECLINK_CV_MAX_VERIFY_TOKENS_PER_STEP=0
SPECLINK_CV_MAX_VERIFY_SEQS_PER_STEP=0
SPECLINK_CV_MAX_QUEUE_WAIT_MS=2
SPECLINK_CV_UTIL_THRESHOLD=0.6
SPECLINK_CV_CALIBRATION_PATH=
SPECLINK_CV_LOG_JSONL=temp/events.jsonl
SPECLINK_CV_PROFILE_JSONL=temp/profile.jsonl
SPECLINK_CV_DEBUG_DUMP=0
```

`SPECLINK_CV_CONFIDENCE_SIZING=1` is wired into the live scheduler. If
`SPECLINK_CV_CALIBRATION_PATH` is empty, `draft_selected_prob` is used directly
as `a_hat` and events report `confidence_source=draft_selected_prob_uncalibrated`.
If the path points to a binning `calibration_model.json`, events report
`confidence_source=calibrated_binning`. `SPECLINK_CV_ROOFLINE_PACKING=1` is
wired as a live utilization gate and emits `roofline_fallback_one_shot` when an
underfilled prefix chunk is converted back to one-shot verification in sync
mode. With `SPECLINK_CV_ASYNC_QUEUE=1`, the scheduler emits
`verify_chunk_queued`, `async_queue_step`, and `verify_chunk_dequeued` profile
events and dispatches queued prefix chunks before exact verification.

`SPECLINK_CV_PROFILE_JSONL` is active in the live scheduler path. It records
newline-delimited JSON events for:

- `schedule_step`: scheduled seq/token counts, spec-token counts,
  prefix-chunk counts, remaining token budget, and config toggles.
- `verify_chunk_queued`, `async_queue_step`, `verify_chunk_dequeued`,
  `verify_chunk_waiting`: live async queue state, selected chunks, wait time,
  dispatch reason, predicted utilization, and budget use.
- `verify_chunk_scheduled`: prefix/suffix chunk length, suffix length,
  selected benefit, scheduled tokens, and budget context.
- `verify_chunk_result`: prefix accepted/rejected outcome, skipped suffix
  tokens, extra TLM forward count, and discarded speculative bonus count.
- `verify_chunk_decision`: roofline fallback decisions and predicted
  token/sequence utilization.

Focused checks from the repo root:

```bash
conda run -n spec python -m tools.speclink_cv.test_chunk_decision
conda run -n spec python -m tools.speclink_cv.test_state_machine
conda run -n spec python -m tools.speclink_cv.test_async_queue
conda run -n spec python -m tools.speclink_cv.test_roofline_packing
conda run -n spec python -m tools.speclink_cv.test_correctness_smoke
conda run -n spec python -m tools.speclink_cv.test_vllm_runtime_config
```

Live smoke status on 2026-05-26: Qwen/Qwen3-8B with the local EAGLE3
speculator, K=8, greedy `temperature=0`, and a 32-token fixed-half generation
matched baseline one-shot EAGLE3 token-for-token. A separate 16-token
confidence-sizing smoke also matched baseline. A calibrated confidence smoke
using the trace milestone `calibration_model.json` matched baseline and showed
`confidence_source=calibrated_binning` in the event JSONL. A roofline fallback
smoke with `--roofline-packing --util-threshold 0.99` also matched baseline and
emitted `roofline_fallback_one_shot`. A live async-queue smoke with
`--async-queue` matched baseline and emitted queue/dequeue/profile events. A
combined `--async-queue --roofline-packing --util-threshold 0.99` smoke also
matched baseline and dispatched via `no_other_ready_work` for the single-request
case. The smoke was run from
`examples/evaluate/eval-guidellm` to avoid repo-root `vllm/` import shadowing.
The event logs are temporary diagnostics under
`examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_20260526_scheduler/`.

Reusable GPU smoke command:

```bash
conda run -n spec python tools/speclink_cv/live_correctness_smoke.py \
  --speculator-model ../models/qwen3-8b-eagle3-speculator \
  --max-tokens 32 \
  --output-json examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke.json \
  --event-jsonl examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_events.jsonl \
  --profile-jsonl examples/evaluate/eval-guidellm/temp/speclink_cv_live_smoke_profile.jsonl
```

Add `--confidence-sizing` to exercise the live uncalibrated confidence path.
Add `--calibration-path path/to/calibration_model.json` with
`--confidence-sizing` to exercise live calibrated binning. Add
`--roofline-packing --util-threshold 0.99` to force the live underfilled-prefix
fallback path in a single-request smoke. Add `--async-queue` to exercise the
live prefix queue.

For batched correctness, run the same smoke over repo-local prompts:

```bash
conda run -n spec python tools/speclink_cv/live_correctness_smoke.py \
  --speculator-model ../models/qwen3-8b-eagle3-speculator \
  --prompts-jsonl examples/evaluate/eval-guidellm/data/math_reasoning.jsonl \
  --num-prompts 8 \
  --max-num-seqs 8 \
  --max-tokens 64 \
  --async-queue \
  --roofline-packing \
  --output-json examples/evaluate/eval-guidellm/temp/speclink_cv_live_batched_smoke.json
```

## SpecLink-CV GuideLLM Matrix Runner

`examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py`
starts `vllm serve`, waits for `/health` with a proxy-safe raw socket check,
runs GuideLLM, parses vLLM speculative metrics, and writes per-run logs plus
top-level `status.csv`, `summary_metrics.csv`, `summary_metrics.json`,
`report.md`, `scripts/run_commands.sh`, the TODO result tree
`00_env/` through `09_reports/`, raw run directories under `runs/`, and figure
source tables under `08_figures/`. For `cv_*` methods it adds
`--no-async-scheduling` because SpecLink-CV's live prefix/suffix scheduler
logic is implemented in the regular V1 scheduler; `SPECLINK_CV_ASYNC_QUEUE` is
the experiment's own verification queue and is independent from vLLM's
scheduler async mode.
The summary includes text-level `exact_match_vs_eagle3` by aligning GuideLLM
successful requests by `request_args` and comparing each `cv_*` output to the
matching `eagle3_oneshot` output for the same model/dataset/K/batch-size case.
Token-id exact-match evidence still comes from
`tools/speclink_cv/live_correctness_smoke.py`. Treat any `cv_*` row with
`exact_match_vs_eagle3 < 1.0` as a correctness warning, not as a valid speedup
claim.

The runner forces `NO_PROXY/no_proxy` to include local addresses and kills the
vLLM process group during cleanup. This matters in the current environment
because local HTTP proxy variables can otherwise trap `127.0.0.1` health checks,
and a failed API server can leave an EngineCore process holding GPU memory.

Smoke command:

```bash
cd .
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --smoke \
  --max-requests 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.75 \
  --port 8051 \
  --output-root examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_TIMESTAMP
```

Full TODO-shaped matrix command:

```bash
conda run -n spec python -u examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_matrix.py \
  --models qwen3_8b,llama3_1_8b \
  --datasets math,mtbench \
  --ks 8,12 \
  --batch-sizes 8,16,32 \
  --methods pure_vllm,eagle3_oneshot,cv_half_sync_simple,cv_half_sync_roofline,cv_half_async_simple,cv_half_async_roofline,cv_conf_sync_simple,cv_conf_sync_roofline,cv_conf_async_simple,cv_conf_async_roofline \
  --max-requests 80 \
  --output-root examples/evaluate/eval-guidellm/results/speclink_cv_guidellm_TIMESTAMP
```

Equivalent one-command wrapper:

```bash
OUTPUT_ROOT=examples/evaluate/eval-guidellm/results/speclink_cv_guidellm_TIMESTAMP \
  examples/evaluate/eval-guidellm/scripts/run_speclink_cv_guidellm_full.sh
```

The wrapper defaults to `--resume`, `--enforce-eager`, and
`--disable-vllm-async-scheduling` for a conservative correctness/fairness run
where EAGLE3 one-shot and CV both use the regular scheduler path. Override with
`ENFORCE_EAGER=0` or `DISABLE_VLLM_ASYNC_SCHEDULING=0` only when you explicitly
want the default vLLM serving mode for the baselines. Use `CASE_OFFSET` and
`CASE_LIMIT` to run chunks of the full matrix.

Use `--dry-run` to write planned commands only, `--analyze-only` to rebuild
summary files from an existing output root, and `--resume` to reuse any run
directory that already contains `guidellm_results.json`. `--case-offset N` and
`--case-limit M` run or analyze a slice of the planned matrix, which is useful
for splitting the 240-case matrix across long GPU sessions. Reusing one output
root with `--resume` is the safest way to continue after interruption. The
runner passes GuideLLM `--random-seed 42` by default for reproducible dataset
sampling. Add `--disable-vllm-async-scheduling` when you need the EAGLE3
one-shot baseline to use the same regular V1 scheduler mode as the CV runs for
correctness/fairness diagnosis.

GuideLLM matrix smoke status on 2026-05-26: Qwen3 math, K=8, batch size 1,
`eagle3_oneshot` and `cv_half_async_roofline` both completed with
`status=ok`; rerun after the `--no-async-scheduling` fix confirmed that the CV
case emits live `verify_chunk_*` profile events. A separate
`cv_conf_async_roofline` smoke also completed with `status=ok` and emitted
`confidence_source=draft_selected_prob_uncalibrated`. A later full-path smoke
also generated `00_env/`, focused unit-test summaries, `08_figures/`, and
`09_reports/SPECLINK_CV_REPORT.md`. Output roots:

```text
examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_20260526_v6/
examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_conf_smoke_20260526/
examples/evaluate/eval-guidellm/temp/speclink_cv_guidellm_smoke_report_20260526_v7/
```

This smoke proves the runner, vLLM server startup, GuideLLM request path, and
live `verify_chunk_*` CV profile logging work. It is not a throughput claim
because it uses one request.

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
With the current workflow, fix this in `speculators/vllm`, reinstall editable if
needed, and verify that `vllm.__file__` points to the vendored source tree.

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
