import pytest

from llm_eval.config import parse_task
from llm_eval.scoring import score


def test_contains_all_hit() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "contains",
            "prompt": "p",
            "expected_contains": ["сервер", "памяти"],
        }
    )
    s, _ = score(task, "Сервер упал из-за нехватки памяти")
    assert s == 1.0


def test_contains_partial() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "contains",
            "prompt": "p",
            "expected_contains": ["сервер", "памяти", "мониторинг"],
        }
    )
    s, _ = score(task, "Сервер упал из-за памяти")
    assert s == pytest.approx(2 / 3)


def test_contains_word_boundary() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "contains",
            "prompt": "p",
            "expected_contains": ["да"],
        }
    )
    # word_boundary default True: "правда" must NOT count as a hit
    # (the "да" in "правда" is part of a larger word)
    s, _ = score(task, "правда что")
    assert s == 0.0


def test_contains_match_all_required() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "contains",
            "prompt": "p",
            "expected_contains": ["a", "b", "c"],
            "match_all_required": True,
        }
    )
    assert score(task, "a b c")[0] == 1.0
    assert score(task, "a b")[0] == 0.0
