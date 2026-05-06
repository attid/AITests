import asyncio

import pytest

from llm_eval.client import CallError, LLMClient
from llm_eval.config import ModelConfig


def _model(**overrides) -> ModelConfig:
    base = {
        "id": "m",
        "base_url": "https://api.example.com/v1",
        "api_key_env": "FAKE_KEY",
        "model": "x",
        "concurrency": 1,
        "temperature": 0.0,
    }
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(autouse=True)
def _key(monkeypatch) -> None:
    monkeypatch.setenv("FAKE_KEY", "sk-test")


async def test_429_with_retry_after(httpx_mock, monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(t: float) -> None:
        sleeps.append(t)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    httpx_mock.add_response(status_code=429, headers={"Retry-After": "2"})
    httpx_mock.add_response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    client = LLMClient(_model())
    result = await client.call(prompt="p", system_prompt="s", max_tokens=10)
    assert result.output == "ok"
    assert sleeps and sleeps[0] >= 1.5  # respected Retry-After


async def test_5xx_exponential_backoff(httpx_mock, monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(t: float) -> None:
        sleeps.append(t)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=503)
    httpx_mock.add_response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    client = LLMClient(_model())
    result = await client.call(prompt="p", system_prompt="s", max_tokens=10)
    assert result.output == "ok"
    assert len(sleeps) == 2
    # second sleep > first (exponential)
    assert sleeps[1] > sleeps[0]


async def test_4xx_no_retry(httpx_mock) -> None:
    httpx_mock.add_response(status_code=400, json={"error": "bad request"})
    client = LLMClient(_model())
    with pytest.raises(CallError):
        await client.call(prompt="p", system_prompt="s", max_tokens=10)


async def test_max_attempts_exceeded(httpx_mock) -> None:
    for _ in range(5):
        httpx_mock.add_response(status_code=500)
    client = LLMClient(_model())
    with pytest.raises(CallError, match="500"):
        await client.call(prompt="p", system_prompt="s", max_tokens=10)


async def test_transport_error_retried(httpx_mock, monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(t: float) -> None:
        sleeps.append(t)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    import httpx

    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    httpx_mock.add_response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    client = LLMClient(_model())
    result = await client.call(prompt="p", system_prompt="s", max_tokens=10)
    assert result.output == "ok"
    assert len(sleeps) == 1


async def test_transport_error_exhausts(monkeypatch, httpx_mock) -> None:
    """All attempts fail with transport error → CallError raised."""

    async def fake_sleep(t: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    import httpx

    for _ in range(3):
        httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    client = LLMClient(_model())
    with pytest.raises(CallError, match="transport error after 3 attempts"):
        await client.call(prompt="p", system_prompt="s", max_tokens=10)
