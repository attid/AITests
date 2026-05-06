"""Universal preprocessing for scorers."""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (some providers inline reasoning)."""
    return _THINK_RE.sub("", text)


def contains_forbidden(
    text: str,
    forbidden: list[str],
    *,
    word_boundary: bool,
    case_sensitive: bool,
) -> str | None:
    """Return the first matching forbidden term, or None."""
    flags = 0 if case_sensitive else re.IGNORECASE
    for term in forbidden:
        if not term:
            continue
        if word_boundary:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, text, flags | re.UNICODE):
                return term
        else:
            haystack = text if case_sensitive else text.lower()
            needle = term if case_sensitive else term.lower()
            if needle in haystack:
                return term
    return None
