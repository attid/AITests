from __future__ import annotations

import json
import re
from typing import Any

from llm_eval.config import TextQualityTask

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)
_MARKDOWN_RE = re.compile(r"```|^#{1,6}\s|\*\*|^\s*[-*]\s", re.MULTILINE)


def _script_ratio(text: str, script: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    if script == "cyrillic":
        matching = sum(1 for ch in letters if _CYRILLIC_RE.fullmatch(ch))
    elif script == "latin":
        matching = sum(1 for ch in letters if _LATIN_RE.fullmatch(ch))
    else:
        return 1.0
    return matching / len(letters)


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True


def score_text_quality(task: TextQualityTask, output: str) -> tuple[float, dict[str, Any]]:
    text = output.strip()
    notes: list[str] = []

    if not text:
        notes.append("empty")
    if task.disallow_cjk and _CJK_RE.search(text):
        notes.append("contains_cjk")
    if task.disallow_json and _looks_like_json(text):
        notes.append("looks_like_json")
    if task.disallow_markdown and _MARKDOWN_RE.search(text):
        notes.append("markdown_artifact")

    words = _WORD_RE.findall(text)
    if len(words) < task.min_words:
        notes.append(f"too_few_words:{len(words)}<{task.min_words}")

    if task.script != "any" and _script_ratio(text, task.script) < task.min_script_ratio:
        notes.append(f"script_mismatch:{task.script}")

    if not notes:
        return 1.0, {"type": task.type, "notes": []}

    hard_failures = {"empty", "contains_cjk", "looks_like_json", "markdown_artifact"}
    if any(note in hard_failures for note in notes):
        return 0.0, {"type": task.type, "notes": notes}

    score = max(0.0, 1.0 - len(notes) / 3)
    return score, {"type": task.type, "notes": notes}
