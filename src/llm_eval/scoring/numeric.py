from __future__ import annotations

import re
from typing import Any

from llm_eval.config import NumericTask

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(text: str) -> float | None:
    m = _NUM_RE.search(text)
    return float(m.group(0)) if m else None


def score_numeric(task: NumericTask, output: str) -> tuple[float, dict[str, Any]]:
    actual = _extract_number(output)
    if actual is None:
        return 0.0, {"type": task.type, "notes": ["no_number_in_output"]}

    tol = task.numeric_tolerance
    bound = max(tol.abs, tol.rel * abs(task.expected))
    if abs(actual - task.expected) <= bound:
        return 1.0, {"type": task.type, "notes": [], "extracted": actual}
    return 0.0, {"type": task.type, "notes": [f"got {actual}, expected {task.expected}"]}
