from llm_eval.config import parse_task
from llm_eval.scoring import score


def test_exact_match() -> None:
    task = parse_task({"id": "t", "type": "exact", "prompt": "p", "expected_text": "OK"})
    s, details = score(task, "OK")
    assert s == 1.0
    assert details["type"] == "exact"


def test_exact_case_insensitive_by_default() -> None:
    task = parse_task({"id": "t", "type": "exact", "prompt": "p", "expected_text": "OK"})
    # case_sensitive default is False, so "ok" matches "OK"
    s, _ = score(task, "ok")
    assert s == 1.0
    s2, _ = score(task, "Okay")
    assert s2 == 0.0


def test_exact_strips_whitespace() -> None:
    task = parse_task({"id": "t", "type": "exact", "prompt": "p", "expected_text": "OK"})
    s, _ = score(task, "  OK  \n")
    assert s == 1.0


def test_exact_with_thinking_stripped() -> None:
    task = parse_task({"id": "t", "type": "exact", "prompt": "p", "expected_text": "OK"})
    s, _ = score(task, "<think>let me think</think>OK")
    assert s == 1.0


def test_forbidden_overrides_match() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "exact",
            "prompt": "p",
            "expected_text": "OK",
            "forbidden_contains": ["OK"],
        }
    )
    s, details = score(task, "OK")
    assert s == 0.0
    assert any("forbidden" in n for n in details["notes"])
