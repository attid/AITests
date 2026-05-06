from llm_eval.config import parse_task
from llm_eval.scoring import score


def test_schema_valid_no_expected_values() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_schema",
            "prompt": "p",
            "schema": {
                "type": "object",
                "required": ["name", "amount"],
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "number"},
                },
            },
        }
    )
    assert score(task, '{"name": "x", "amount": 1.5}')[0] == 1.0


def test_schema_invalid() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_schema",
            "prompt": "p",
            "schema": {"type": "object", "required": ["name"]},
        }
    )
    s, details = score(task, '{"other": "x"}')
    assert s == 0.0
    assert any("schema" in n for n in details["notes"])


def test_schema_with_expected_values_full() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_schema",
            "prompt": "p",
            "schema": {"type": "object", "required": ["name"]},
            "expected_values": {"name": "Ivan"},
        }
    )
    assert score(task, '{"name": "Ivan"}')[0] == 1.0
    assert score(task, '{"name": "Other"}')[0] == 0.5  # schema valid, values mismatch


def test_schema_parse_failure() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "json_schema",
            "prompt": "p",
            "schema": {"type": "object"},
        }
    )
    s, details = score(task, "not json")
    assert s == 0.0
    assert details["json_ok"] is False
