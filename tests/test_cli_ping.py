from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_eval.cli import app

runner = CliRunner()
DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")


def test_ping_success(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )
    result = runner.invoke(
        app,
        ["ping", "--config", str(DATA / "models_minimal.yaml"), "--only", "gpt-mini"],
    )
    assert result.exit_code == 0, result.output
    assert "gpt-mini" in result.output
    assert "OK" in result.output
    request = httpx_mock.get_request()
    assert request is not None
    assert request.read()
    assert b'"max_tokens":128' in request.content


def test_ping_failure_exits_nonzero(httpx_mock) -> None:
    httpx_mock.add_response(status_code=401, json={"error": "invalid key"})
    result = runner.invoke(
        app,
        ["ping", "--config", str(DATA / "models_minimal.yaml"), "--only", "gpt-mini"],
    )
    assert result.exit_code == 1
    assert "gpt-mini" in result.output


def test_ping_empty_or_wrong_content_exits_nonzero(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        },
    )
    result = runner.invoke(
        app,
        ["ping", "--config", str(DATA / "models_minimal.yaml"), "--only", "gpt-mini"],
    )
    assert result.exit_code == 1
    assert "unexpected output" in result.output


def test_ping_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(
        app,
        ["ping", "--config", str(DATA / "models_minimal.yaml"), "--only", "gpt-mini"],
    )
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output
