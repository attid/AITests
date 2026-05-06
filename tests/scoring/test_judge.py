from unittest.mock import AsyncMock

from llm_eval.client import CallResult
from llm_eval.config import LlmJudgeTask, ModelConfig, parse_task
from llm_eval.scoring.judge import JudgeRunner
from llm_eval.storage import Usage


def _judge_model() -> ModelConfig:
    return ModelConfig(
        id="judge-m",
        base_url="http://x",
        api_key_env="K",
        model="m",
        concurrency=1,
        temperature=0.0,
    )


def _result(content: str) -> CallResult:
    return CallResult(
        output=content,
        reasoning_content=None,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency_sec=0.1,
        provider_cost=None,
    )


def _task(**overrides) -> LlmJudgeTask:
    base = {
        "id": "t",
        "type": "llm_judge",
        "prompt": "user prompt here",
        "rubric": "Answer must be brief and accurate",
    }
    base.update(overrides)
    task = parse_task(base)
    assert isinstance(task, LlmJudgeTask)
    return task


async def test_judge_pass() -> None:
    client = AsyncMock()
    client.call = AsyncMock(
        return_value=_result('{"score": 1, "reasoning": "matches all rubric criteria"}')
    )
    runner = JudgeRunner(client=client, judge_model=_judge_model())

    s, details = await runner.score(_task(), "candidate answer")
    assert s == 1.0
    assert details["judge_reasoning"] == "matches all rubric criteria"
    assert details["judge_cost_usd"] == 0.0


async def test_judge_partial() -> None:
    client = AsyncMock()
    client.call = AsyncMock(return_value=_result('{"score": 0.5, "reasoning": "ok-ish"}'))
    runner = JudgeRunner(client=client, judge_model=_judge_model())
    s, _ = await runner.score(_task(), "candidate")
    assert s == 0.5


async def test_judge_invalid_json() -> None:
    client = AsyncMock()
    client.call = AsyncMock(return_value=_result("not json at all"))
    runner = JudgeRunner(client=client, judge_model=_judge_model())
    s, details = await runner.score(_task(), "candidate")
    assert s == 0.0
    assert any("judge_error" in n for n in details["notes"])


async def test_judge_call_error() -> None:
    from llm_eval.client import CallError

    client = AsyncMock()
    client.call = AsyncMock(side_effect=CallError("network down"))
    runner = JudgeRunner(client=client, judge_model=_judge_model())
    s, details = await runner.score(_task(), "candidate")
    assert s == 0.0
    assert any("judge_error" in n for n in details["notes"])


async def test_judge_clamps_invalid_score() -> None:
    client = AsyncMock()
    client.call = AsyncMock(return_value=_result('{"score": 0.7, "reasoning": "?"}'))
    runner = JudgeRunner(client=client, judge_model=_judge_model())
    s, _ = await runner.score(_task(), "candidate")
    # 0.7 is not in {0, 0.5, 1}; runner snaps to nearest valid (0.5).
    assert s == 0.5
