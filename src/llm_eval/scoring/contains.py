from __future__ import annotations

import re
from typing import Any

from llm_eval.config import ContainsTask


def score_contains(task: ContainsTask, output: str) -> tuple[float, dict[str, Any]]:
    flags = 0 if task.case_sensitive else re.IGNORECASE
    flags |= re.UNICODE
    hits = 0
    for term in task.expected_contains:
        if not term:
            continue
        if task.word_boundary:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, output, flags):
                hits += 1
        else:
            hay = output if task.case_sensitive else output.lower()
            needle = term if task.case_sensitive else term.lower()
            if needle in hay:
                hits += 1
    n = len(task.expected_contains)
    if n == 0:
        return 0.0, {"type": task.type, "notes": ["empty expected_contains"]}
    if task.match_all_required:
        return (1.0 if hits == n else 0.0), {"type": task.type, "notes": []}
    return hits / n, {"type": task.type, "notes": []}
