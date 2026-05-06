from __future__ import annotations

import re
from typing import Any

from llm_eval.config import RegexTask


def score_regex(task: RegexTask, output: str) -> tuple[float, dict[str, Any]]:
    flags = re.UNICODE | (0 if task.case_sensitive else re.IGNORECASE)
    if re.search(task.expected_regex, output, flags):
        return 1.0, {"type": task.type, "notes": []}
    return 0.0, {"type": task.type, "notes": []}
