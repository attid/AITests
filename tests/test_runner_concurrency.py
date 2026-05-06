import asyncio
from pathlib import Path

import pytest

from llm_eval.config import ModelConfig, Pricing, RunConfig, parse_task
from llm_eval.runner import Runner
from llm_eval.storage import ResultStore


def _two_models() -> list[ModelConfig]:
    return [
        ModelConfig(
            id="m1",
            base_url="https://a.test/v1",
            api_key_env="K1",
            model="m1",
            pricing=Pricing(input_per_million=1.0, output_per_million=1.0),
            concurrency=2,
            temperature=0.0,
        ),
        ModelConfig(
            id="m2",
            base_url="https://b.test/v1",
            api_key_env="K2",
            model="m2",
            pricing=Pricing(input_per_million=1.0, output_per_million=1.0),
            concurrency=1,
            temperature=0.0,
        ),
    ]


@pytest.fixture(autouse=True)
def _keys(monkeypatch) -> None:
    monkeypatch.setenv("K1", "x")
    monkeypatch.setenv("K2", "x")


async def test_concurrency_capped_per_model(monkeypatch, tmp_path: Path) -> None:
    in_flight: dict[str, int] = {"m1": 0, "m2": 0}
    peak: dict[str, int] = {"m1": 0, "m2": 0}
    lock = asyncio.Lock()

    async def fake_call(self, **kwargs):
        async with lock:
            in_flight[self.model.id] += 1
            peak[self.model.id] = max(peak[self.model.id], in_flight[self.model.id])
        await asyncio.sleep(0.05)
        async with lock:
            in_flight[self.model.id] -= 1
        from llm_eval.client import CallResult
        from llm_eval.storage import Usage

        return CallResult(
            output="OK",
            reasoning_content=None,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_sec=0.05,
            provider_cost=None,
        )

    from llm_eval.client import LLMClient

    monkeypatch.setattr(LLMClient, "call", fake_call)

    models = _two_models()
    cfg = RunConfig(models=models)
    tasks = [
        parse_task(
            {
                "id": f"t{i}",
                "type": "exact",
                "prompt": "p",
                "expected_text": "OK",
            }
        )
        for i in range(8)
    ]
    runner = Runner(config=cfg, models=models, tasks=tasks, out_dir=tmp_path / "run")
    await runner.run()

    assert peak["m1"] <= 2
    assert peak["m2"] <= 1
    rows = list(ResultStore(tmp_path / "run" / "results.jsonl").read())
    assert len(rows) == 16  # 8 tasks * 2 models * 1 repeat


async def test_one_model_failure_does_not_block_others(monkeypatch, tmp_path: Path) -> None:
    from llm_eval.client import CallError, CallResult, LLMClient
    from llm_eval.storage import Usage

    async def fake_call(self, **kwargs):
        if self.model.id == "m1":
            raise CallError("boom")
        return CallResult(
            output="OK",
            reasoning_content=None,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_sec=0.01,
            provider_cost=None,
        )

    monkeypatch.setattr(LLMClient, "call", fake_call)

    models = _two_models()
    cfg = RunConfig(models=models)
    tasks = [
        parse_task(
            {
                "id": "t1",
                "type": "exact",
                "prompt": "p",
                "expected_text": "OK",
            }
        )
    ]
    runner = Runner(config=cfg, models=models, tasks=tasks, out_dir=tmp_path / "run")
    await runner.run()

    rows = list(ResultStore(tmp_path / "run" / "results.jsonl").read())
    assert len(rows) == 2
    by_model = {r.model_id: r for r in rows}
    assert by_model["m1"].score == 0.0
    assert by_model["m2"].score == 1.0
