"""LLM-as-judge scorer. Runs a separate model with a rubric prompt."""

from __future__ import annotations

from typing import Any, Protocol, cast

from llm_eval.client import CallError, CallResult
from llm_eval.config import LlmJudgeTask, ModelConfig
from llm_eval.scoring.json_match import extract_json

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator. Given a task description, the model's response, and a rubric, "
    "return strict JSON with no markdown: "
    '{"score": 0|0.5|1, "reasoning": "..."}. '
    "Score 1 if the response fully meets the rubric, 0.5 if partially, 0 otherwise."
)


def _build_user_prompt(task_prompt: str, output: str, rubric: str) -> str:
    return f"Task:\n{task_prompt}\n\nResponse:\n{output}\n\nRubric:\n{rubric}"


def _snap_score(value: float) -> float:
    """Snap to {0, 0.5, 1} — judge models occasionally pick 0.7 etc."""
    options = (0.0, 0.5, 1.0)
    return min(options, key=lambda x: abs(x - value))


class _AsyncCaller(Protocol):
    async def call(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> CallResult: ...


class JudgeRunner:
    def __init__(self, *, client: _AsyncCaller, judge_model: ModelConfig) -> None:
        self.client = client
        self.judge_model = judge_model

    async def score(
        self, task: LlmJudgeTask, candidate_output: str
    ) -> tuple[float, dict[str, Any]]:
        user_prompt = _build_user_prompt(task.prompt, candidate_output, task.rubric)
        try:
            result = await self.client.call(
                prompt=user_prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                max_tokens=300,
                temperature=0.0,
            )
        except CallError as e:
            return 0.0, {
                "type": "llm_judge",
                "notes": [f"judge_error:{e}"],
                "judge_cost_usd": 0.0,
            }

        try:
            parsed = extract_json(result.output)
            parsed_dict = cast(dict[str, Any], parsed)
            raw_score = float(parsed_dict.get("score", -1))
            reasoning = str(parsed_dict.get("reasoning", ""))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            return 0.0, {
                "type": "llm_judge",
                "notes": [f"judge_error:invalid_json:{e}"],
                "judge_cost_usd": 0.0,
            }

        return _snap_score(raw_score), {
            "type": "llm_judge",
            "judge_reasoning": reasoning,
            "judge_usage": result.usage.model_dump(),
            "judge_provider_cost": result.provider_cost,
            "judge_cost_usd": 0.0,
            "notes": [],
        }
