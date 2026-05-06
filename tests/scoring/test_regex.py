from llm_eval.config import parse_task
from llm_eval.scoring import score


def test_regex_hit() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "regex",
            "prompt": "p",
            "expected_regex": r"^OK\s*$",
        }
    )
    assert score(task, "OK")[0] == 1.0
    assert score(task, "OK\n")[0] == 1.0


def test_regex_miss() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "regex",
            "prompt": "p",
            "expected_regex": r"^OK\s*$",
        }
    )
    assert score(task, "Okay")[0] == 0.0


def test_regex_case_sensitive() -> None:
    task = parse_task(
        {
            "id": "t",
            "type": "regex",
            "prompt": "p",
            "expected_regex": "ok",
            "case_sensitive": True,
        }
    )
    assert score(task, "OK")[0] == 0.0
