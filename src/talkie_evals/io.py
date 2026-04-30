"""Local result file helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return value.replace(",", "_").replace("/", "-").replace(" ", "")
