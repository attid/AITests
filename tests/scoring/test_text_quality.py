import pytest

from llm_eval.config import TextQualityTask, parse_task
from llm_eval.scoring import score


def _task(**overrides) -> TextQualityTask:
    data = {
        "id": "txt",
        "type": "text_quality",
        "prompt": "Ответь одним человеческим предложением.",
        "min_words": 4,
        "script": "cyrillic",
    }
    data.update(overrides)
    task = parse_task(data)
    assert isinstance(task, TextQualityTask)
    return task


def test_text_quality_passes_human_russian_sentence() -> None:
    s, details = score(_task(), "Платёж отклонён из-за недостатка средств на карте.")
    assert s == 1.0
    assert details["notes"] == []


def test_text_quality_rejects_cjk_characters() -> None:
    s, details = score(_task(), "Платёж отклонён 因为余额不足.")
    assert s == 0.0
    assert "contains_cjk" in details["notes"]


def test_text_quality_rejects_json_markdown_and_too_short_text() -> None:
    s, details = score(_task(disallow_json=True), '{"answer": "OK"}')
    assert s == 0.0
    assert "looks_like_json" in details["notes"]

    s, details = score(_task(disallow_markdown=True), "```text\nOK\n```")
    assert s == 0.0
    assert "markdown_artifact" in details["notes"]

    s, details = score(_task(min_words=3), "\u041e\u041a")
    assert s == pytest.approx(2 / 3)
    assert "too_few_words:1<3" in details["notes"]


def test_text_quality_script_ratio() -> None:
    s, details = score(_task(script="latin"), "This response is readable and natural.")
    assert s == 1.0
    assert details["notes"] == []

    s, details = score(_task(script="cyrillic"), "This response is readable and natural.")
    assert s == pytest.approx(2 / 3)
    assert "script_mismatch:cyrillic" in details["notes"]
