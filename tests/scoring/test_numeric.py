from llm_eval.config import parse_task
from llm_eval.scoring import score


def test_numeric_exact() -> None:
    task = parse_task({"id": "t", "type": "numeric", "prompt": "p", "expected": 391})
    assert score(task, "391")[0] == 1.0
    assert score(task, "the answer is 391.")[0] == 1.0


def test_numeric_float() -> None:
    task = parse_task({"id": "t", "type": "numeric", "prompt": "p", "expected": 1.5})
    assert score(task, "1.5")[0] == 1.0
    assert score(task, "1.50")[0] == 1.0


def test_numeric_with_abs_tolerance() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "numeric",
            "prompt": "p",
            "expected": 100.0,
            "numeric_tolerance": {"abs": 0.5, "rel": 0},
        }
    )
    assert score(task, "100.4")[0] == 1.0
    assert score(task, "100.6")[0] == 0.0


def test_numeric_with_rel_tolerance() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "numeric",
            "prompt": "p",
            "expected": 1000.0,
            "numeric_tolerance": {"abs": 0, "rel": 0.01},  # 1%
        }
    )
    assert score(task, "1009")[0] == 1.0
    assert score(task, "1011")[0] == 0.0


def test_numeric_extract_first_number() -> None:
    task = parse_task({"id": "t", "type": "numeric", "prompt": "p", "expected": 42})
    assert score(task, "result: 42 with conf 0.9")[0] == 1.0


def test_numeric_no_number_in_output() -> None:
    task = parse_task({"id": "t", "type": "numeric", "prompt": "p", "expected": 42})
    s, details = score(task, "no number here")
    assert s == 0.0
    assert any("no_number" in n for n in details["notes"])
