"""Async orchestrator: dispatches tasks per model, scores, streams to storage."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rich.progress import Progress, TaskID

from llm_eval.client import CallError, LLMClient
from llm_eval.config import LlmJudgeTask, ModelConfig, RunConfig, Task
from llm_eval.pricing import compute_cost
from llm_eval.scoring import score_async
from llm_eval.scoring.judge import JudgeRunner
from llm_eval.storage import ResultRecord, ResultStore, Usage

_SYSTEM_PROMPT = (
    "You are being evaluated. Follow the user instruction exactly. "
    "Do not add markdown unless explicitly requested. "
    "If the answer is not in the provided data, say there is not enough data."
)


class Runner:
    def __init__(
        self,
        *,
        config: RunConfig,
        models: list[ModelConfig],
        tasks: list[Task],
        out_dir: Path,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.models = models
        self.tasks = tasks
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.store = ResultStore(self.out_dir / "results.jsonl")
        self.done: set[tuple[str, str, int]] = self.store.done_set() if resume else set()

        self.judge_runner = self._build_judge() if self._has_judge_tasks() else None

    def _has_judge_tasks(self) -> bool:
        return any(isinstance(t, LlmJudgeTask) for t in self.tasks)

    def _build_judge(self) -> JudgeRunner | None:
        if self.config.judge is None:
            return None
        judge_model = next(
            (m for m in self.config.models if m.id == self.config.judge.model_id),
            None,
        )
        if judge_model is None:
            raise ValueError(f"judge.model_id={self.config.judge.model_id!r} not found in models")
        return JudgeRunner(client=LLMClient(judge_model), judge_model=judge_model)

    async def run(self) -> None:
        repeats = self.config.defaults.repeats
        with Progress() as progress:
            coros: list[Coroutine[Any, Any, None]] = []
            for model in self.models:
                client = LLMClient(model)
                concurrency = model.concurrency or self.config.defaults.concurrency
                sem = asyncio.Semaphore(concurrency)
                bar = progress.add_task(f"[cyan]{model.id}", total=len(self.tasks) * repeats)
                for task in self.tasks:
                    for repeat in range(1, repeats + 1):
                        if (model.id, task.id, repeat) in self.done:
                            progress.advance(bar)
                            continue
                        coros.append(self._run_one(client, model, task, repeat, sem, progress, bar))
            if coros:
                await asyncio.gather(*coros)

    async def _run_one(
        self,
        client: LLMClient,
        model: ModelConfig,
        task: Task,
        repeat: int,
        sem: asyncio.Semaphore,
        progress: Progress,
        bar: TaskID,
    ) -> None:
        async with sem:
            ts = datetime.now(UTC).isoformat()
            try:
                call = await client.call(
                    prompt=task.prompt,
                    system_prompt=_SYSTEM_PROMPT,
                    max_tokens=task.max_tokens,
                    attempts=self.config.defaults.retries.attempts,
                    backoff_base=self.config.defaults.retries.backoff_base,
                    respect_retry_after=self.config.defaults.retries.respect_retry_after,
                )
            except CallError as e:
                await self.store.append(self._error_record(model, task, repeat, ts, str(e)))
                progress.advance(bar)
                return

            try:
                score_val, details = await score_async(
                    task, call.output, judge_runner=self.judge_runner
                )
                details["tags"] = list(task.tags)
                details["weight"] = task.weight
                cost, source = compute_cost(
                    model,
                    prompt_tokens=call.usage.prompt_tokens,
                    completion_tokens=call.usage.completion_tokens,
                    provider_cost=call.provider_cost,
                )
                judge_cost = 0.0
                judge_usage = cast(dict[str, int] | None, details.get("judge_usage"))
                if judge_usage is not None and self.judge_runner is not None:
                    judge_provider_cost = cast(float | None, details.get("judge_provider_cost"))
                    jc, _ = compute_cost(
                        self.judge_runner.judge_model,
                        prompt_tokens=judge_usage["prompt_tokens"],
                        completion_tokens=judge_usage["completion_tokens"],
                        provider_cost=judge_provider_cost,
                    )
                    judge_cost = jc

                notes = cast(list[str], details.get("notes", []))
                record = ResultRecord(
                    model_id=model.id,
                    task_id=task.id,
                    repeat=repeat,
                    ts=ts,
                    score=score_val,
                    details=details,
                    latency_sec=call.latency_sec,
                    usage=call.usage.model_dump(),
                    cost_usd=cost,
                    cost_source=source,
                    judge_cost_usd=judge_cost,
                    output=call.output,
                    reasoning_content=call.reasoning_content if model.is_reasoning else None,
                    notes=notes,
                )
                await self.store.append(record)
            except Exception as e:  # runner must not crash on per-task errors
                await self.store.append(
                    self._error_record(
                        model, task, repeat, ts, f"unexpected:{type(e).__name__}:{e}"
                    )
                )
            finally:
                progress.advance(bar)

    def _error_record(
        self,
        model: ModelConfig,
        task: Task,
        repeat: int,
        ts: str,
        error: str,
    ) -> ResultRecord:
        return ResultRecord(
            model_id=model.id,
            task_id=task.id,
            repeat=repeat,
            ts=ts,
            score=0.0,
            details={
                "type": task.type,
                "notes": [f"error:{error}"],
                "tags": list(task.tags),
                "weight": task.weight,
            },
            latency_sec=0.0,
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0).model_dump(),
            cost_usd=0.0,
            cost_source="unknown",
            judge_cost_usd=0.0,
            output="",
            reasoning_content=None,
            notes=[f"error:{error}"],
        )
