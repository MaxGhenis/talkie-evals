"""Modal CUDA app for Talkie evaluations."""

from __future__ import annotations

import json
import random
import statistics
import time
from importlib import resources
from typing import Any

import modal

from talkie_evals import __version__
from talkie_evals.constants import (
    ARITHMETIC_DATASET,
    ARITHMETIC_LM_EVAL_SOURCE,
    ARITHMETIC_REVISION,
    ARITHMETIC_TASKS,
    DEFAULT_SEED,
    GSM8K_DATASET,
    GSM8K_REVISION,
    MODAL_PIP_PACKAGES,
    MODAL_VOLUME_NAME,
    MODEL_REVISIONS,
    PYTHON_VERSION,
    TALKIE_GIT_REVISION,
)
from talkie_evals.talkie_loader import load_talkie

app = modal.App("talkie-evals")

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install(*MODAL_PIP_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface"})
)

cache = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

GSM8K_TEST_SIZE = 1319
ARITHMETIC_TASK_SIZE = 2000


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _provenance() -> dict[str, Any]:
    return {
        "talkie_evals_version": __version__,
        "talkie_git_revision": TALKIE_GIT_REVISION,
        "model_revisions": MODEL_REVISIONS,
        "arithmetic_dataset_revision": ARITHMETIC_REVISION,
        "gsm8k_dataset_revision": GSM8K_REVISION,
        "modal_python_version": PYTHON_VERSION,
        "modal_pip_packages": MODAL_PIP_PACKAGES,
    }


def _expand_lm_eval_tasks(tasks: str) -> list[str]:
    requested = _parse_csv(tasks)
    if not requested:
        return ["gsm8k"]
    expanded = []
    arithmetic_names = [task["name"] for task in ARITHMETIC_TASKS]
    for task in requested:
        if task == "arithmetic":
            expanded.extend(arithmetic_names)
        else:
            expanded.append(task)
    return expanded


def _sample_indices(population_size: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size >= population_size:
        return list(range(population_size))
    rng = random.Random(seed)
    return sorted(rng.sample(range(population_size), sample_size))


def _lm_eval_samples(task_names: list[str], sample_size: int, seed: int):
    if sample_size <= 0:
        return None
    samples = {}
    arithmetic_names = {task["name"] for task in ARITHMETIC_TASKS}
    for task_name in task_names:
        if task_name in arithmetic_names:
            samples[task_name] = _sample_indices(
                ARITHMETIC_TASK_SIZE, sample_size, seed
            )
        elif task_name == "gsm8k":
            samples[task_name] = _sample_indices(GSM8K_TEST_SIZE, sample_size, seed)
    return samples or None


def _jsonable_lm_eval_result(result: dict[str, Any]) -> dict[str, Any]:
    import json

    from lm_eval.utils import handle_non_serializable

    return json.loads(
        json.dumps(result, default=handle_non_serializable, ensure_ascii=False)
    )


def _sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def _load_arithmetic_rows(filename: str, cache_dir: str) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=ARITHMETIC_DATASET,
        repo_type="dataset",
        revision=ARITHMETIC_REVISION,
        filename=filename,
        cache_dir=cache_dir,
    )
    rows = []
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            data = json.loads(line)
            context = (
                data["context"]
                .strip()
                .replace("\n\n", "\n")
                .replace("Q:", "Question:")
                .replace("A:", "Answer:")
            )
            rows.append(
                {
                    "index": index,
                    "context": context,
                    "completion": data["completion"],
                }
            )
    return rows


def _summarize_arithmetic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    greedy_count = sum(1 for row in rows if row["is_greedy"])
    first_token_count = sum(1 for row in rows if row["first_token_greedy"])
    target_token_counts = [row["target_token_count"] for row in rows]
    return {
        "total": total,
        "correct_count": greedy_count,
        "accuracy": greedy_count / total if total else None,
        "first_token_greedy_count": first_token_count,
        "first_token_greedy_accuracy": first_token_count / total if total else None,
        "mean_logprob": statistics.mean(row["logprob"] for row in rows)
        if rows
        else None,
        "mean_logprob_per_token": statistics.mean(
            row["logprob_per_token"] for row in rows
        )
        if rows
        else None,
        "mean_target_tokens": statistics.mean(target_token_counts)
        if target_token_counts
        else None,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    memory=80_000,
    volumes={"/cache": cache},
    timeout=14_400,
)
def run_arithmetic_eval(
    model_names: str = "talkie-1930-13b-base",
    task_names: str = "",
    sample_size: int = 200,
    seed: int = DEFAULT_SEED,
    log_examples: int = 20,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    requested_models = _parse_csv(model_names)
    if not requested_models:
        raise ValueError("model_names must include at least one model")
    unknown_models = set(requested_models) - set(MODEL_REVISIONS)
    if unknown_models:
        raise ValueError(f"Unknown model_names: {sorted(unknown_models)}")

    requested_tasks = set(_parse_csv(task_names))
    tasks = [
        task
        for task in ARITHMETIC_TASKS
        if not requested_tasks or task["name"] in requested_tasks
    ]
    missing_tasks = requested_tasks - {task["name"] for task in tasks}
    if missing_tasks:
        raise ValueError(f"Unknown task_names: {sorted(missing_tasks)}")

    dataset_started = time.perf_counter()
    task_rows = {
        task["name"]: _sample_rows(
            _load_arithmetic_rows(task["filename"], "/cache/huggingface/datasets"),
            sample_size=sample_size,
            seed=seed,
        )
        for task in tasks
    }
    cache.commit()
    dataset_load_seconds = time.perf_counter() - dataset_started

    def score_row(talkie, row: dict[str, Any]) -> dict[str, Any]:
        context_tokens = talkie.tokenizer.encode(row["context"], allowed_special="all")
        target_tokens = talkie.tokenizer.encode(
            row["completion"], allowed_special="all"
        )
        if not context_tokens:
            raise ValueError(f"Empty context for row {row['index']}")
        if not target_tokens:
            raise ValueError(f"Empty target for row {row['index']}")

        input_tokens = list(context_tokens)
        step_details = []
        total_logprob = 0.0
        greedy_tokens = []

        with torch.no_grad(), talkie._autocast:
            for target_token in target_tokens:
                token_tensor = torch.tensor(
                    input_tokens, dtype=torch.long, device=talkie.device
                ).unsqueeze(0)
                logits = talkie.model(token_tensor)[0]
                logprobs = F.log_softmax(logits, dim=-1)
                top_token = int(torch.argmax(logits).item())
                target_logprob = float(logprobs[target_token].item())
                top_logprob = float(logprobs[top_token].item())
                greedy_tokens.append(top_token)
                step_details.append(
                    {
                        "target_token": int(target_token),
                        "target_text": talkie.tokenizer.decode([int(target_token)]),
                        "target_logprob": target_logprob,
                        "top_token": top_token,
                        "top_text": talkie.tokenizer.decode([top_token]),
                        "top_logprob": top_logprob,
                        "greedy_match": top_token == int(target_token),
                    }
                )
                total_logprob += target_logprob
                input_tokens.append(int(target_token))

        return {
            "index": row["index"],
            "context": row["context"],
            "completion": row["completion"],
            "target_tokens": [int(token) for token in target_tokens],
            "target_token_count": len(target_tokens),
            "logprob": total_logprob,
            "logprob_per_token": total_logprob / len(target_tokens),
            "is_greedy": all(step["greedy_match"] for step in step_details),
            "first_token_greedy": step_details[0]["greedy_match"],
            "teacher_forced_top_text": talkie.tokenizer.decode(greedy_tokens),
            "steps": step_details,
        }

    model_results = []
    for model_name in requested_models:
        load_started = time.perf_counter()
        talkie = load_talkie(model_name, cache_dir="/cache/huggingface")
        cache.commit()
        model_load_seconds = time.perf_counter() - load_started
        task_results = []
        for task in tasks:
            task_started = time.perf_counter()
            rows = []
            for item_number, row in enumerate(task_rows[task["name"]], start=1):
                rows.append(score_row(talkie, row))
                if item_number % 100 == 0:
                    correct_so_far = sum(1 for item in rows if item["is_greedy"])
                    print(
                        f"{model_name} {task['name']}: {item_number}/"
                        f"{len(task_rows[task['name']])} complete, "
                        f"{correct_so_far} greedy matches",
                        flush=True,
                    )
            summary = _summarize_arithmetic(rows)
            task_results.append(
                {
                    "name": task["name"],
                    "description": task["description"],
                    "filename": task["filename"],
                    "summary": summary,
                    "seconds": time.perf_counter() - task_started,
                    "examples": rows[:log_examples],
                    "all_rows": rows,
                }
            )
            print(
                f"{model_name} {task['name']} done: "
                f"{summary['correct_count']}/{summary['total']} "
                f"acc={summary['accuracy']:.3f}",
                flush=True,
            )

        all_rows = [row for task in task_results for row in task["all_rows"]]
        model_results.append(
            {
                "model_name": model_name,
                "style": talkie.spec.style,
                "device": str(talkie.device),
                "cuda_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "model_revision": MODEL_REVISIONS[model_name],
                "model_load_seconds": model_load_seconds,
                "summary": _summarize_arithmetic(all_rows),
                "tasks": task_results,
            }
        )
        del talkie
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "dataset": ARITHMETIC_DATASET,
        "dataset_revision": ARITHMETIC_REVISION,
        "dataset_source": "https://huggingface.co/datasets/EleutherAI/arithmetic",
        "lm_eval_task_source": ARITHMETIC_LM_EVAL_SOURCE,
        "scoring": (
            "lm-eval-style loglikelihood acc: exact completion is greedy under "
            "teacher forcing"
        ),
        "sample_size": sample_size,
        "seed": seed,
        "log_examples": log_examples,
        "tasks": tasks,
        "dataset_load_seconds": dataset_load_seconds,
        "provenance": _provenance(),
        "models": model_results,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    memory=80_000,
    volumes={"/cache": cache},
    timeout=14_400,
)
def run_lm_eval_harness(
    model_names: str = "talkie-1930-13b-base",
    tasks: str = "arithmetic",
    sample_size: int = 0,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    num_fewshot: int | None = None,
    apply_talkie_chat_template: bool = False,
    log_samples: bool = True,
) -> dict[str, Any]:
    import lm_eval
    import lm_eval.evaluator
    import torch
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table

    from talkie_evals.lm_eval_model import TalkieLM

    def no_git_commit_hash() -> None:
        return None

    lm_eval.evaluator.get_git_commit_hash = no_git_commit_hash

    requested_models = _parse_csv(model_names)
    if not requested_models:
        raise ValueError("model_names must include at least one model")
    unknown_models = set(requested_models) - set(MODEL_REVISIONS)
    if unknown_models:
        raise ValueError(f"Unknown model_names: {sorted(unknown_models)}")
    if limit is not None and sample_size > 0:
        raise ValueError("Use either limit or sample_size, not both")

    task_names = _expand_lm_eval_tasks(tasks)
    samples = _lm_eval_samples(task_names, sample_size, seed)
    if sample_size > 0:
        if samples is None:
            raise ValueError(
                "sample_size is only supported for pinned arithmetic and GSM8K "
                "tasks. Use limit for other lm-eval tasks."
            )
        unsampled_tasks = sorted(set(task_names) - set(samples))
        if unsampled_tasks:
            raise ValueError(
                "sample_size is only supported for pinned arithmetic and GSM8K "
                f"tasks. Use limit for: {unsampled_tasks}"
            )
    task_path = str(resources.files("talkie_evals").joinpath("lm_eval_tasks"))

    model_results = []
    for model_name in requested_models:
        load_started = time.perf_counter()
        lm = TalkieLM(
            model_name=model_name,
            cache_dir="/cache/huggingface",
            apply_talkie_chat_template=apply_talkie_chat_template,
        )
        cache.commit()
        model_load_seconds = time.perf_counter() - load_started
        task_manager = TaskManager(include_path=task_path)
        eval_started = time.perf_counter()
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=task_names,
            num_fewshot=num_fewshot,
            limit=limit,
            samples=samples,
            bootstrap_iters=0,
            log_samples=log_samples,
            task_manager=task_manager,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            gen_kwargs={"do_sample": False, "temperature": 0.0},
        )
        if result is None:
            raise RuntimeError("lm_eval.simple_evaluate returned None")
        result = _jsonable_lm_eval_result(result)
        model_results.append(
            {
                "model_name": model_name,
                "model_revision": MODEL_REVISIONS[model_name],
                "style": lm.talkie.spec.style,
                "device": str(lm.talkie.device),
                "cuda_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "apply_talkie_chat_template": apply_talkie_chat_template,
                "model_load_seconds": model_load_seconds,
                "eval_seconds": time.perf_counter() - eval_started,
                "table": make_table(result),
                "result": result,
            }
        )
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "kind": "lm_eval_harness",
        "tasks": task_names,
        "task_path": task_path,
        "sample_size": sample_size,
        "limit": limit,
        "seed": seed,
        "num_fewshot": num_fewshot,
        "samples": samples,
        "log_samples": log_samples,
        "provenance": _provenance(),
        "models": model_results,
    }


def _summarize_gsm8k(items: list[dict[str, Any]]) -> dict[str, Any]:
    correct_count = sum(1 for item in items if item["correct"])
    parsed_count = sum(1 for item in items if item["prediction"] is not None)
    return {
        "correct_count": correct_count,
        "total": len(items),
        "accuracy": correct_count / len(items) if items else None,
        "parsed_count": parsed_count,
        "unparsed_count": len(items) - parsed_count,
        "mean_seconds": statistics.mean(item["seconds"] for item in items)
        if items
        else None,
        "mean_tokens": statistics.mean(item["token_count"] for item in items)
        if items
        else None,
    }


def _few_shot_prefix(train_examples: list[dict[str, Any]], n: int) -> str:
    return "\n\n".join(
        f"Question: {example['question']}\n{example['answer']}"
        for example in train_examples[:n]
    )


@app.function(
    image=image,
    gpu="A100-40GB",
    memory=80_000,
    volumes={"/cache": cache},
    timeout=7_200,
)
def run_gsm8k_eval(
    model_name: str = "talkie-1930-13b-it",
    sample_size: int = 50,
    seed: int = DEFAULT_SEED,
    few_shot: int = 4,
    condition_names: str = "",
    temperature: float = 0.18,
    top_k: int = 20,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset

    from talkie_evals.gsm8k import extract_prediction, gold_answer, is_correct

    if model_name not in MODEL_REVISIONS:
        raise ValueError(f"Unknown model_name: {model_name}")

    load_started = time.perf_counter()
    talkie = load_talkie(model_name, cache_dir="/cache/huggingface")
    cache.commit()
    model_load_seconds = time.perf_counter() - load_started

    dataset_started = time.perf_counter()
    gsm8k = load_dataset(
        GSM8K_DATASET,
        "main",
        revision=GSM8K_REVISION,
        cache_dir="/cache/huggingface/datasets",
    )
    train = list(gsm8k["train"])
    test = list(gsm8k["test"])
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(test)), min(sample_size, len(test))))
    examples = [test[index] for index in indices]
    dataset_load_seconds = time.perf_counter() - dataset_started

    prefix = _few_shot_prefix(train, few_shot)
    conditions = [
        {
            "name": "zero_shot_direct",
            "max_tokens": 40,
            "prompt": (
                "Solve the following grade-school arithmetic problem. Give only "
                "the final numeric answer.\n\nQuestion: {question}\nAnswer:"
            ),
        },
        {
            "name": "zero_shot_cot",
            "max_tokens": 256,
            "prompt": (
                "Solve the following grade-school arithmetic problem. Reason "
                "briefly, then give the final numeric answer.\n\n"
                "Question: {question}\nSolution:"
            ),
        },
        {
            "name": f"{few_shot}_shot_cot",
            "max_tokens": 256,
            "prompt": prefix + "\n\nQuestion: {question}\n",
        },
    ]
    requested_conditions = set(_parse_csv(condition_names))
    if requested_conditions:
        conditions = [
            condition
            for condition in conditions
            if condition["name"] in requested_conditions
        ]
        if not conditions:
            raise ValueError(
                f"No matching conditions for {sorted(requested_conditions)}"
            )

    def generate(prompt: str, max_tokens: int, gold: str | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        result = talkie.generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        text = result.text.strip()
        prediction = extract_prediction(text)
        return {
            "text": text,
            "prediction": prediction,
            "gold": gold,
            "correct": is_correct(prediction, gold),
            "token_count": result.token_count,
            "seconds": time.perf_counter() - t0,
        }

    condition_results = []
    for condition in conditions:
        generations = []
        for item_number, (index, example) in enumerate(
            zip(indices, examples, strict=True), start=1
        ):
            gold = gold_answer(example["answer"])
            prompt = condition["prompt"].format(question=example["question"])
            output = generate(prompt, condition["max_tokens"], gold)
            generations.append(
                {
                    "index": index,
                    "question": example["question"],
                    "gold_answer_text": example["answer"],
                    "prompt": prompt,
                    **output,
                }
            )
            if item_number % 10 == 0:
                correct_so_far = sum(1 for item in generations if item["correct"])
                print(
                    f"{condition['name']}: {item_number}/{len(examples)} "
                    f"complete, {correct_so_far} correct",
                    flush=True,
                )
        condition_results.append(
            {
                "name": condition["name"],
                "max_tokens": condition["max_tokens"],
                "summary": _summarize_gsm8k(generations),
                "generations": generations,
            }
        )

    return {
        "model_name": model_name,
        "style": talkie.spec.style,
        "device": str(talkie.device),
        "cuda_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "model_revision": MODEL_REVISIONS[model_name],
        "model_load_seconds": model_load_seconds,
        "dataset_load_seconds": dataset_load_seconds,
        "dataset": "gsm8k/main/test",
        "dataset_hf_repo": GSM8K_DATASET,
        "dataset_revision": GSM8K_REVISION,
        "sample_size": len(examples),
        "seed": seed,
        "indices": indices,
        "few_shot": few_shot,
        "condition_names": condition_names,
        "temperature": temperature,
        "top_k": top_k,
        "provenance": _provenance(),
        "conditions": condition_results,
    }
