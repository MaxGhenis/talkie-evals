# Talkie evals

Reproducible numeracy evaluations for the Talkie language models, run on Modal
CUDA GPUs. The package currently supports:

- An `lm-evaluation-harness` runner for pinned arithmetic and GSM8K task
  definitions.
- A custom arithmetic audit runner that logs token-level target/top-token traces.
- A custom GSM8K probe runner that logs prompts, completions, parsed answers, and
  timing.

This repo is intentionally narrow. It is for math/numeracy checks, not value
forecasting or public-opinion forecasting.

## Setup

Install [`uv`](https://docs.astral.sh/uv/) and authenticate Modal:

```bash
uv sync
uv run modal setup
```

The first CUDA run downloads Talkie checkpoints into the Modal volume
`talkie-hf-cache`. Subsequent runs reuse that cache.

## Pinned inputs

The evaluator records and uses pinned inputs:

- Talkie Python package commit:
  `5b65cfe0d311520898112494aaf583d30f9ac044`
- `talkie-1930-13b-base` model revision:
  `b7c97680791f7fca4262c3c80b36ff7d666faab0`
- `talkie-1930-13b-it` model revision:
  `8033675be6360ae0127fa75f941c12d52064f1dc`
- `talkie-web-13b-base` model revision:
  `1e5b771c9d38d44f54d35e722c5c0d73da418dd8`
- `EleutherAI/arithmetic` dataset revision:
  `cf5ec4512aaa47cdebf02ca032b7af870528c272`
- `openai/gsm8k` dataset revision:
  `740312add88f781978c0658806c59bc2815b9866`

Print the same list from the package:

```bash
uv run talkie-evals provenance
```

## Run lm-eval-harness

Use this path for benchmark-style runs. Arithmetic is scored by log-likelihood,
so there is no generation temperature. GSM8K is generated greedily with
`do_sample: false` and `temperature: 0.0`.

Smoke test one arithmetic task:

```bash
uv run talkie-evals harness \
  --model-names talkie-1930-13b-it \
  --tasks arithmetic_2da \
  --sample-size 2
```

Run a sampled arithmetic comparison:

```bash
uv run talkie-evals harness \
  --model-names talkie-1930-13b-base,talkie-1930-13b-it,talkie-web-13b-base \
  --tasks arithmetic \
  --sample-size 500
```

Run a zero-shot GSM8K generation eval with the Talkie instruction template:

```bash
uv run talkie-evals harness \
  --model-names talkie-1930-13b-it \
  --tasks gsm8k \
  --sample-size 50 \
  --num-fewshot 0 \
  --talkie-chat-template
```

Use `--sample-size 0` to evaluate every example in the selected tasks.

## Run custom arithmetic

Smoke test one task:

```bash
uv run talkie-evals arithmetic \
  --model-names talkie-1930-13b-it \
  --task-names arithmetic_2da \
  --sample-size 5 \
  --log-examples 5
```

Run the main comparison used in the blog draft:

```bash
uv run talkie-evals arithmetic \
  --model-names talkie-1930-13b-base,talkie-1930-13b-it,talkie-web-13b-base \
  --sample-size 500 \
  --log-examples 25
```

Use `--sample-size 0` to run all 2,000 examples per arithmetic task.

## Run custom GSM8K

```bash
uv run talkie-evals gsm8k \
  --model-name talkie-1930-13b-it \
  --sample-size 50 \
  --condition-names zero_shot_direct
```

The GSM8K command logs every prompt, model output, parsed answer, gold answer,
and timing in the result JSON.

## Summarize a result

```bash
uv run talkie-evals summarize results/arithmetic_eval_*.json
```

The saved JSON contains the full provenance, aggregate summaries, and row-level
outputs. Arithmetic row records include the target tokens and top token at each
teacher-forced target position, so failures like a word-form answer beating the
digit target remain inspectable.

## Notes on the arithmetic metric

The arithmetic task definitions in `lm-evaluation-harness` use
`output_type: loglikelihood`, `doc_to_text: "{{context}}"`, and
`doc_to_target: "{{completion}}"`, with `acc` equal to whether the exact target
completion is greedy. This package follows that convention. It does not give
credit when the model prefers a semantically equivalent answer with different
formatting, such as `Forty` instead of ` 40`.
