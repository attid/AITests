"""OpenAI-compatible client over httpx.

Why httpx directly instead of the openai SDK: we need fine-grained control
over retries (Retry-After) and provider-specific fields (usage.cost from
OpenRouter, reasoning_content from DeepSeek/Qwen). The openai SDK abstracts
those away.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx

from llm_eval.config import ModelConfig
from llm_eval.storage import Usage


class CallError(Exception):
    """Wraps transport, rate-limit, server, and validation failures."""


@dataclass(frozen=True)
class CallResult:
    output: str
    reasoning_content: str | None
    usage: Usage
    latency_sec: float
    provider_cost: float | None  # from usage.cost when present


async def _sleep_for_retry(
    attempt: int,
    *,
    base: float,
    max_wait: float = 60.0,
    retry_after: float | None = None,
) -> None:
    if retry_after is not None:
        await asyncio.sleep(min(retry_after, max_wait))
        return
    # Exponential: base * 4**attempt with ±20% jitter.
    delay = min(max_wait, base * (4**attempt))
    jitter = delay * 0.2 * (random.random() * 2 - 1)
    await asyncio.sleep(max(0.0, delay + jitter))


class LLMClient:
    def __init__(self, model: ModelConfig, *, timeout: float = 60.0) -> None:
        self.model = model
        self._timeout = timeout

    def _api_key(self) -> str:
        key = os.environ.get(self.model.api_key_env)
        if not key:
            raise CallError(f"env var {self.model.api_key_env} not set for model {self.model.id}")
        return key

    async def call(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float | None = None,
        attempts: int = 3,
        backoff_base: float = 1.0,
        respect_retry_after: bool = True,
    ) -> CallResult:
        api_key = self._api_key()
        body: dict[str, Any] = {
            "model": self.model.model,
            "max_tokens": max_tokens,
            "temperature": self.model.temperature if temperature is None else temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.model.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}

        url = f"{self.model.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.model.extra_headers,
        }

        last_error: Exception | None = None
        for attempt in range(attempts):
            t0 = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as http:
                    response = await http.post(url, json=body, headers=headers)
                latency = time.monotonic() - t0
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                if attempt == attempts - 1:
                    raise CallError(f"transport error after {attempts} attempts: {e}") from e
                await _sleep_for_retry(attempt, base=backoff_base)
                continue

            if response.status_code == 429:
                last_error = CallError(f"rate limited: {response.text[:200]}")
                if attempt == attempts - 1:
                    raise last_error
                ra = response.headers.get("Retry-After") if respect_retry_after else None
                ra_val = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
                await _sleep_for_retry(attempt, base=backoff_base, retry_after=ra_val)
                continue

            if 500 <= response.status_code < 600:
                last_error = CallError(
                    f"server error {response.status_code}: {response.text[:200]}"
                )
                if attempt == attempts - 1:
                    raise last_error
                await _sleep_for_retry(attempt, base=backoff_base)
                continue

            if not response.is_success:
                raise CallError(f"http {response.status_code}: {response.text[:200]}")

            data = response.json()
            return _parse_response(data, latency=latency)

        # unreachable, but keeps the type checker happy
        raise CallError(f"unreachable; last error: {last_error}")


def _parse_response(data: dict[str, Any], *, latency: float) -> CallResult:
    try:
        msg = cast(dict[str, Any], data["choices"][0]["message"])
        content_raw = msg.get("content")
        content: str = content_raw if isinstance(content_raw, str) else ""
        reasoning_raw = msg.get("reasoning_content")
        reasoning: str | None = reasoning_raw if isinstance(reasoning_raw, str) else None
        usage_raw_obj = data.get("usage")
        usage_raw: dict[str, Any] = (
            cast(dict[str, Any], usage_raw_obj) if isinstance(usage_raw_obj, dict) else {}
        )
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_raw.get("total_tokens", 0) or 0),
            reasoning_tokens=int(usage_raw.get("reasoning_tokens", 0) or 0),
        )
        provider_cost_raw = usage_raw.get("cost")
        provider_cost = float(provider_cost_raw) if provider_cost_raw is not None else None
        return CallResult(
            output=content,
            reasoning_content=reasoning,
            usage=usage,
            latency_sec=latency,
            provider_cost=provider_cost,
        )
    except (KeyError, IndexError, TypeError) as e:
        raise CallError(f"malformed response: {e}; raw={data!r}") from e
