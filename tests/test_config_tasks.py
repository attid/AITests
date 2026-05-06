import json

import pytest
from pydantic import ValidationError

from llm_eval.config import (
    ContainsTask,
    ExactTask,
    JsonExactTask,
    JsonSchemaTask,
    LlmJudgeTask,
    NumericTask,
    RegexTask,
    parse_task,
)


def test_exact_task() -> None:
    task = parse_task({"id": "t1", "type": "exact", "prompt": "p", "expected_text": "OK"})
    assert isinstance(task, ExactTask)
    assert task.expected_text == "OK"
    assert task.weight == 1.0
    assert task.word_boundary is True
    assert task.case_sensitive is False


def test_contains_task_with_forbidden() -> None:
    task = parse_task(
        {
            "id": "t2",
            "type": "contains",
            "prompt": "p",
            "expected_contains": ["нет данных"],
            "forbidden_contains": ["@"],
        }
    )
    assert isinstance(task, ContainsTask)
    assert task.expected_contains == ["нет данных"]
    assert task.forbidden_contains == ["@"]


def test_regex_task() -> None:
    task = parse_task({"id": "t3", "type": "regex", "prompt": "p", "expected_regex": r"^OK$"})
    assert isinstance(task, RegexTask)


def test_numeric_task_with_tolerance() -> None:
    task = parse_task(
        {
            "id": "t4",
            "type": "numeric",
            "prompt": "p",
            "expected": 391,
            "numeric_tolerance": {"abs": 0.01, "rel": 0.001},
        }
    )
    assert isinstance(task, NumericTask)
    assert task.numeric_tolerance.abs == 0.01
    assert task.numeric_tolerance.rel == 0.001


def test_json_exact_task() -> None:
    task = parse_task(
        {
            "id": "t5",
            "type": "json_exact",
            "prompt": "p",
            "expected": {"answer": 391},
        }
    )
    assert isinstance(task, JsonExactTask)
    assert task.expected == {"answer": 391}


def test_json_schema_task() -> None:
    task = parse_task(
        {
            "id": "t6",
            "type": "json_schema",
            "prompt": "p",
            "schema": {"type": "object", "required": ["name"]},
        }
    )
    assert isinstance(task, JsonSchemaTask)


def test_llm_judge_task() -> None:
    task = parse_task(
        {
            "id": "t7",
            "type": "llm_judge",
            "prompt": "p",
            "rubric": "Answer must mention X",
        }
    )
    assert isinstance(task, LlmJudgeTask)


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_task({"id": "t", "type": "nonsense", "prompt": "p"})


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_task({"id": "t", "type": "exact", "prompt": "p"})  # no expected_text


def test_tags_and_weight() -> None:
    task = parse_task(
        {
            "id": "t8",
            "type": "exact",
            "prompt": "p",
            "expected_text": "OK",
            "tags": ["json", "anti_hallucination"],
            "weight": 1.5,
        }
    )
    assert task.tags == ["json", "anti_hallucination"]
    assert task.weight == 1.5


def test_jsonl_roundtrip(tmp_path) -> None:
    src = tmp_path / "tasks.jsonl"
    rows = [
        {"id": "a", "type": "exact", "prompt": "p", "expected_text": "OK"},
        {"id": "b", "type": "regex", "prompt": "p", "expected_regex": "^x$"},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    tasks = [parse_task(json.loads(line)) for line in src.read_text().splitlines() if line.strip()]
    assert len(tasks) == 2
