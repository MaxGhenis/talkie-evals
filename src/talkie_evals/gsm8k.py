"""GSM8K answer parsing helpers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().replace(",", "").replace("$", "").rstrip(".")
    if value.endswith("%"):
        value = value[:-1]
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        return None
    normalized = decimal.normalize()
    if normalized == normalized.to_integral():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")


def gold_answer(answer: str) -> str | None:
    match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?%?)", answer)
    if match:
        return normalize_number(match.group(1))
    return None


def extract_prediction(text: str) -> str | None:
    final_patterns = [
        r"(?:therefore|thus|so|answer|final answer|the answer is)[:\s$]*"
        r"([-+]?\$?[\d,]+(?:\.\d+)?%?)",
        r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?%?)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            parsed = normalize_number(matches[-1])
            if parsed is not None:
                return parsed

    matches = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?%?", text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def is_correct(prediction: str | None, gold: str | None) -> bool:
    return prediction is not None and gold is not None and prediction == gold
