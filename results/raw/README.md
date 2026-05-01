# Raw results

Compressed JSON artifacts for the Talkie-1930 math-evals blog draft.

These are the original custom-run artifacts behind the published tables. They
were generated before the package added the current full provenance block, so
they should be read alongside the repository's pinned dependency, model, and
dataset constants. New runs from the current package include more explicit
provenance metadata in the result JSON.

## Files

- `arithmetic_eval_talkie-1930-13b-base_talkie-1930-13b-it_talkie-web-13b-base_20260429_214800.json.gz`
  - Custom arithmetic audit run used for the arithmetic table.
  - Models: `talkie-1930-13b-base`, `talkie-1930-13b-it`, `talkie-web-13b-base`.
  - Tasks: 10 EleutherAI/OpenAI arithmetic tasks.
  - Sample: 500 rows per task, seed 1930.
  - Logs row-level scoring data, including target and top-token traces.

- `gsm8k_eval_talkie-1930-13b-it_zero_shot_direct_20260429_170239.json.gz`
  - Custom GSM8K direct-answer generation probe used for the GSM8K table.
  - Model: `talkie-1930-13b-it`.
  - Sample: 50 rows, seed 1930.
  - Decoding: `temperature=0.18`, `top_k=20`.
  - Logs prompts, raw completions, parsed answers, gold answers, correctness, token counts, and timings.

- `gsm8k_eval_talkie-1930-13b-it_cot_20260429_170728.json.gz`
  - Custom GSM8K reasoning probe used for the GSM8K table.
  - Model: `talkie-1930-13b-it`.
  - Conditions: zero-shot chain-of-thought and 4-shot chain-of-thought.
  - Sample: 10 rows per condition, seed 1930.
  - Decoding: `temperature=0.18`, `top_k=20`.
  - Logs prompts, raw completions, parsed answers, gold answers, correctness, token counts, and timings.

Decompress with:

```bash
gzip -dk results/raw/<file>.json.gz
```
