# LMeval Accuracy Runs

This repo uses EleutherAI `lm-eval==0.4.12` through the same vLLM serving path as
the token-dense accuracy experiments. The wrapper starts `vllm serve`, applies
the requested dense/speculative/hybrid environment, then runs `lm-eval` against
the local OpenAI-compatible completions endpoint.

Supported modes:

- `dense_ar`
- `eagle3_dense`
- `activation_aware`
- `token_dense_t00` through `token_dense_t10`

Install or refresh the harness:

```bash
cd .
examples/evaluate/eval-guidellm/scripts/install_lm_eval.sh
```

Run a smoke matrix:

```bash
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode all \
  --task smoke \
  --limit 4 \
  --output-dir examples/evaluate/eval-guidellm/results/token_dense_lm_eval
```

Run a focused task:

```bash
examples/evaluate/eval-guidellm/scripts/run_lm_eval_accuracy.sh \
  --mode token_dense \
  --task gsm8k_cot \
  --models llama3_1_8b \
  --output-dir examples/evaluate/eval-guidellm/results/token_dense_lm_eval \
  --resume
```

For LogiQA, the built-in `logiqa` task in this environment depends on an old
dataset script that current `datasets` rejects. The wrapper's `--task all`
therefore uses `agieval_logiqa_en` for the official multiple-choice LogiQA slot,
and `logiqa_generative` for the generate-until A/B/C/D path.

`humaneval_instruct` is skipped unless `--allow-unsafe-code` is provided. Do not
use that flag without an isolated execution environment.
