from __future__ import annotations

import re
from typing import Any

from llm_eval.config import ExactTask


def score_exact(task: ExactTask, output: str) -> tuple[float, dict[str, Any]]:
    actual = output.strip()
    expected = task.expected_text.strip()
    if task.normalize_whitespace:
        actual = re.sub(r"\s+", " ", actual)
        expected = re.sub(r"\s+", " ", expected)
    if not task.case_sensitive:
        actual = actual.lower()
        expected = expected.lower()
    score_val = 1.0 if actual == expected else 0.0
    return score_val, {"type": task.type, "notes": []}
