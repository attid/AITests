import pytest

from llm_eval.client import CallError, CallResult, LLMClient
from llm_eval.config import ModelConfig


def _model(**overrides) -> ModelConfig:
    base = {
        "id": "m1",
        "base_url": "https://api.example.com/v1",
        "api_key_env": "FAKE_KEY",
        "model": "fake-model",
        "concurrency": 1,
        "temperature": 0.0,
    }
    base.update(overrides)
    return ModelConfig(**base)


async def test_basic_call(monkeypatch, httpx_mock) -> None:
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    client = LLMClient(_model())
    result = await client.call(prompt="hi", system_prompt="be brief", max_tokens=50)
    assert isinstance(result, CallResult)
    assert result.output == "hello"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.reasoning_content is None
    assert result.provider_cost is None


async def test_reasoning_content_extracted(monkeypatch, httpx_mock) -> None:
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "step 1; step 2",
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )
    client = LLMClient(_model(is_reasoning=True))
    result = await client.call(prompt="hi", system_prompt="s", max_tokens=10)
    assert result.output == "answer"
    assert result.reasoning_content == "step 1; step 2"


async def test_provider_cost_extracted(monkeypatch, httpx_mock) -> None:
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "x"}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0042,
            },
        },
    )
    client = LLMClient(_model(use_provider_cost=True))
    result = await client.call(prompt="hi", system_prompt="s", max_tokens=10)
    assert result.provider_cost == 0.0042


async def test_response_format_passed_when_set(monkeypatch, httpx_mock) -> None:
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    client = LLMClient(_model(response_format="json_object"))
    await client.call(prompt="p", system_prompt="s", max_tokens=10)
    request = httpx_mock.get_requests()[0]
    body = request.read()
    assert b'"response_format"' in body
    assert b'"json_object"' in body


async def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("FAKE_KEY", raising=False)
    client = LLMClient(_model())
    with pytest.raises(CallError, match="FAKE_KEY"):
        await client.call(prompt="p", system_prompt="s", max_tokens=10)


async def test_malformed_response_raises(monkeypatch, httpx_mock) -> None:
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    httpx_mock.add_response(
        url="https://api.example.com/v1/chat/completions",
        json={"choices": []},  # missing [0].message
    )
    client = LLMClient(_model())
    with pytest.raises(CallError, match="malformed response"):
        await client.call(prompt="p", system_prompt="s", max_tokens=10)
