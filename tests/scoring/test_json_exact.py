from llm_eval.config import parse_task
from llm_eval.scoring import score


def test_json_exact_simple() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"answer": 391},
        }
    )
    assert score(task, '{"answer": 391}')[0] == 1.0


def test_json_exact_extracts_from_code_block() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"answer": 391},
        }
    )
    s, _ = score(task, '```json\n{"answer": 391}\n```')
    assert s == 1.0


def test_json_exact_partial_match() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"a": 1, "b": 2},
        }
    )
    s, _ = score(task, '{"a": 1, "b": 99}')
    assert s == 0.5  # one of two keys correct


def test_json_exact_missing_key() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"a": 1, "b": 2},
        }
    )
    s, _ = score(task, '{"a": 1}')
    assert s == 0.5


def test_json_exact_extra_keys_ignored() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"a": 1},
        }
    )
    s, _ = score(task, '{"a": 1, "extra": "ok"}')
    assert s == 1.0


def test_json_exact_numeric_tolerance() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"x": 1.0},
            "numeric_tolerance": {"abs": 0.01, "rel": 0},
        }
    )
    assert score(task, '{"x": 1.005}')[0] == 1.0
    assert score(task, '{"x": 1.5}')[0] == 0.0


def test_json_exact_string_case_insensitive_default() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"category": "billing"},
        }
    )
    assert score(task, '{"category": "BILLING"}')[0] == 1.0


def test_json_exact_string_case_sensitive_when_set() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"category": "billing"},
            "case_sensitive_strings": True,
        }
    )
    assert score(task, '{"category": "BILLING"}')[0] == 0.0


def test_json_exact_parse_failure() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"a": 1},
        }
    )
    s, details = score(task, "not json at all")
    assert s == 0.0
    assert details["json_ok"] is False
    assert any("json_parse_error" in n for n in details["notes"])


def test_json_exact_list() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_exact",
            "prompt": "p",
            "expected": [1, 2, 3],
        }
    )
    assert score(task, "[1, 2, 3]")[0] == 1.0
    assert score(task, "[1, 2]")[0] == 0.0  # length mismatch
