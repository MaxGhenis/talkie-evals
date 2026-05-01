# Raw results

Compressed JSON artifacts for the Talkie-1930 math-evals blog post.

The `lm_eval_full_*.json.gz` files are the benchmark-style artifacts behind the
published tables. They include full `lm-evaluation-harness` outputs, model and
dataset provenance, task configs, samples, raw responses, filtered responses,
and per-row metrics.

The older custom-run artifacts are retained as audit/probe records. They were
generated before the package added the current full provenance block, so read
them alongside the repository's pinned dependency, model, and dataset constants.

## Files

- `lm_eval_full_arithmetic_all_models.json.gz`
  - Full `lm-evaluation-harness` arithmetic run used for the arithmetic table.
  - Models: `talkie-1930-13b-base`, `talkie-1930-13b-it`, `talkie-web-13b-base`.
  - Tasks: all 10 EleutherAI/OpenAI arithmetic tasks.
  - Sample: all 2,000 validation rows per task.
  - Scoring: log-likelihood exact target completion, matching the harness task definitions.

- `lm_eval_full_gsm8k_zero_shot_chat.json.gz`
  - Full `lm-evaluation-harness` GSM8K zero-shot run used for the GSM8K table.
  - Model: `talkie-1930-13b-it`.
  - Sample: all 1,319 GSM8K test rows.
  - Prompting: zero-shot with the Talkie instruction chat template.
  - Decoding: greedy (`do_sample=false`, `temperature=0.0`).
  - Logs prompts, raw completions, filtered answers, gold answers, and per-row metrics.

- `lm_eval_full_gsm8k_5shot_chat.json.gz`
  - Full `lm-evaluation-harness` GSM8K 5-shot run used for the GSM8K table.
  - Model: `talkie-1930-13b-it`.
  - Sample: all 1,319 GSM8K test rows.
  - Prompting: the harness task's standard 5-shot prompt with the Talkie instruction chat template.
  - Decoding: greedy (`do_sample=false`, `temperature=0.0`).
  - Logs prompts, raw completions, filtered answers, gold answers, and per-row metrics.

- `arithmetic_eval_talkie-1930-13b-base_talkie-1930-13b-it_talkie-web-13b-base_20260429_214800.json.gz`
  - Custom arithmetic audit run used for the original sampled arithmetic table.
  - Models: `talkie-1930-13b-base`, `talkie-1930-13b-it`, `talkie-web-13b-base`.
  - Tasks: 10 EleutherAI/OpenAI arithmetic tasks.
  - Sample: 500 rows per task, seed 1930.
  - Logs row-level scoring data, including target and top-token traces.

- `gsm8k_eval_talkie-1930-13b-it_zero_shot_direct_20260429_170239.json.gz`
  - Custom GSM8K direct-answer generation probe from the original audit.
  - Model: `talkie-1930-13b-it`.
  - Sample: 50 rows, seed 1930.
  - Decoding: `temperature=0.18`, `top_k=20`.
  - Logs prompts, raw completions, parsed answers, gold answers, correctness, token counts, and timings.

- `gsm8k_eval_talkie-1930-13b-it_cot_20260429_170728.json.gz`
  - Custom GSM8K reasoning probe from the original audit.
  - Model: `talkie-1930-13b-it`.
  - Conditions: zero-shot chain-of-thought and 4-shot chain-of-thought.
  - Sample: 10 rows per condition, seed 1930.
  - Decoding: `temperature=0.18`, `top_k=20`.
  - Logs prompts, raw completions, parsed answers, gold answers, correctness, token counts, and timings.

Decompress with:

```bash
gzip -dk results/raw/<file>.json.gz
```
