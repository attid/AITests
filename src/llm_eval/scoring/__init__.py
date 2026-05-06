"""Score task outputs. Dispatch by task type."""

from __future__ import annotations

from typing import Any

from llm_eval.config import (
    ContainsTask,
    ExactTask,
    JsonExactTask,
    JsonSchemaTask,
    LlmJudgeTask,
    NumericTask,
    RegexTask,
    Task,
)
from llm_eval.scoring.contains import score_contains
from llm_eval.scoring.exact import score_exact
from llm_eval.scoring.json_match import score_json_exact, score_json_schema
from llm_eval.scoring.judge import JudgeRunner
from llm_eval.scoring.numeric import score_numeric
from llm_eval.scoring.preprocess import contains_forbidden, strip_thinking
from llm_eval.scoring.regex import score_regex


def score(task: Task, raw_output: str) -> tuple[float, dict[str, Any]]:
    """Universal entry point. Strips reasoning, checks forbidden, then dispatches."""
    output = strip_thinking(raw_output)

    if task.forbidden_contains:
        hit = contains_forbidden(
            output,
            task.forbidden_contains,
            word_boundary=task.word_boundary,
            case_sensitive=task.case_sensitive,
        )
        if hit is not None:
            return 0.0, {"type": task.type, "notes": [f"forbidden_contains:{hit}"]}

    if isinstance(task, ExactTask):
        return score_exact(task, output)
    if isinstance(task, ContainsTask):
        return score_contains(task, output)
    if isinstance(task, RegexTask):
        return score_regex(task, output)
    if isinstance(task, NumericTask):
        return score_numeric(task, output)
    if isinstance(task, JsonExactTask):
        return score_json_exact(task, output)
    if isinstance(task, JsonSchemaTask):
        return score_json_schema(task, output)

    raise NotImplementedError(f"scoring for {task.type} not yet implemented")


async def score_async(
    task: Task,
    raw_output: str,
    *,
    judge_runner: JudgeRunner | None = None,
) -> tuple[float, dict[str, Any]]:
    """Async dispatch — handles llm_judge, delegates everything else to sync score()."""
    if isinstance(task, LlmJudgeTask):
        if judge_runner is None:
            return 0.0, {
                "type": "llm_judge",
                "notes": ["judge_runner_not_configured"],
            }
        # Apply universal preprocessing first.
        output = strip_thinking(raw_output)
        if task.forbidden_contains:
            hit = contains_forbidden(
                output,
                task.forbidden_contains,
                word_boundary=task.word_boundary,
                case_sensitive=task.case_sensitive,
            )
            if hit is not None:
                return 0.0, {
                    "type": "llm_judge",
                    "notes": [f"forbidden_contains:{hit}"],
                }
        return await judge_runner.score(task, output)

    return score(task, raw_output)
