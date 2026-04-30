"""Summaries for saved Talkie eval result JSON."""

from __future__ import annotations

from typing import Any


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100 * value:.1f}%"


def arithmetic_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in result.get("models", []):
        for task in model.get("tasks", []):
            summary = task["summary"]
            rows.append(
                {
                    "model": model["model_name"],
                    "task": task["name"],
                    "description": task.get("description", ""),
                    "accuracy": summary.get("accuracy"),
                    "correct": summary.get("correct_count"),
                    "total": summary.get("total"),
                    "first_token_accuracy": summary.get("first_token_greedy_accuracy"),
                    "mean_logprob_per_token": summary.get("mean_logprob_per_token"),
                }
            )
    return rows


def gsm8k_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in result.get("conditions", []):
        summary = condition["summary"]
        rows.append(
            {
                "model": result.get("model_name"),
                "condition": condition["name"],
                "accuracy": summary.get("accuracy"),
                "correct": summary.get("correct_count"),
                "total": summary.get("total"),
                "parsed": summary.get("parsed_count"),
                "mean_seconds": summary.get("mean_seconds"),
                "mean_tokens": summary.get("mean_tokens"),
            }
        )
    return rows


def markdown_table(result: dict[str, Any]) -> str:
    if result.get("dataset") == "EleutherAI/arithmetic":
        rows = arithmetic_rows(result)
        header = (
            "| model | task | description | accuracy | correct / total | "
            "first-token acc |"
        )
        lines = [
            header,
            "|---|---:|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {model} | {task} | {description} | {accuracy} | {correct}/{total} | "
                "{first_token_accuracy} |".format(
                    model=row["model"],
                    task=row["task"],
                    description=row["description"],
                    accuracy=_pct(row["accuracy"]),
                    correct=row["correct"],
                    total=row["total"],
                    first_token_accuracy=_pct(row["first_token_accuracy"]),
                )
            )
        return "\n".join(lines)

    if str(result.get("dataset", "")).startswith("gsm8k"):
        rows = gsm8k_rows(result)
        header = (
            "| model | condition | accuracy | correct / total | parsed | "
            "mean seconds | mean tokens |"
        )
        lines = [
            header,
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {model} | {condition} | {accuracy} | {correct}/{total} | {parsed} | "
                "{mean_seconds:.2f} | {mean_tokens:.1f} |".format(
                    model=row["model"],
                    condition=row["condition"],
                    accuracy=_pct(row["accuracy"]),
                    correct=row["correct"],
                    total=row["total"],
                    parsed=row["parsed"],
                    mean_seconds=row["mean_seconds"] or 0,
                    mean_tokens=row["mean_tokens"] or 0,
                )
            )
        return "\n".join(lines)

    raise ValueError(f"Unknown result format: {result.get('dataset')!r}")
