"""Pinned model, dataset, and dependency metadata."""

from __future__ import annotations

TALKIE_GIT_REVISION = "5b65cfe0d311520898112494aaf583d30f9ac044"
TALKIE_PACKAGE = f"git+https://github.com/talkie-lm/talkie.git@{TALKIE_GIT_REVISION}"

MODAL_VOLUME_NAME = "talkie-hf-cache"

PYTHON_VERSION = "3.11"
MODAL_PIP_PACKAGES = [
    "torch==2.11.0",
    "tiktoken==0.12.0",
    "huggingface_hub[hf_transfer]==1.12.2",
    "hf_transfer==0.1.9",
    "datasets==4.8.5",
    TALKIE_PACKAGE,
]

MODEL_REVISIONS = {
    "talkie-1930-13b-base": "b7c97680791f7fca4262c3c80b36ff7d666faab0",
    "talkie-1930-13b-it": "8033675be6360ae0127fa75f941c12d52064f1dc",
    "talkie-web-13b-base": "1e5b771c9d38d44f54d35e722c5c0d73da418dd8",
}

ARITHMETIC_DATASET = "EleutherAI/arithmetic"
ARITHMETIC_REVISION = "cf5ec4512aaa47cdebf02ca032b7af870528c272"
ARITHMETIC_LM_EVAL_SOURCE = (
    "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/"
    "lm_eval/tasks/arithmetic"
)

GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"

DEFAULT_SEED = 1930

ARITHMETIC_TASKS = [
    {
        "name": "arithmetic_1dc",
        "description": "single-digit three-operation expressions",
        "filename": "data/single_digit_three_ops.jsonl",
    },
    {
        "name": "arithmetic_2da",
        "description": "2-digit addition",
        "filename": "data/two_digit_addition.jsonl",
    },
    {
        "name": "arithmetic_2ds",
        "description": "2-digit subtraction",
        "filename": "data/two_digit_subtraction.jsonl",
    },
    {
        "name": "arithmetic_3da",
        "description": "3-digit addition",
        "filename": "data/three_digit_addition.jsonl",
    },
    {
        "name": "arithmetic_3ds",
        "description": "3-digit subtraction",
        "filename": "data/three_digit_subtraction.jsonl",
    },
    {
        "name": "arithmetic_4da",
        "description": "4-digit addition",
        "filename": "data/four_digit_addition.jsonl",
    },
    {
        "name": "arithmetic_4ds",
        "description": "4-digit subtraction",
        "filename": "data/four_digit_subtraction.jsonl",
    },
    {
        "name": "arithmetic_5da",
        "description": "5-digit addition",
        "filename": "data/five_digit_addition.jsonl",
    },
    {
        "name": "arithmetic_5ds",
        "description": "5-digit subtraction",
        "filename": "data/five_digit_subtraction.jsonl",
    },
    {
        "name": "arithmetic_2dm",
        "description": "2-digit multiplication",
        "filename": "data/two_digit_multiplication.jsonl",
    },
]
