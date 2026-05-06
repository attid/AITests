"""JSONL stream storage with resume support."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import aiofiles
from pydantic import BaseModel, ConfigDict


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    task_id: str
    repeat: int
    ts: str
    score: float
    details: dict[str, Any]
    latency_sec: float
    usage: dict[str, Any]
    cost_usd: float
    cost_source: Literal["provider", "config", "unknown"]
    judge_cost_usd: float
    output: str
    reasoning_content: str | None
    notes: list[str]


class ResultStore:
    """Append-only JSONL store with resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, record: ResultRecord) -> None:
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write(record.model_dump_json() + "\n")
            await f.flush()

    def read(self) -> Iterator[ResultRecord]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{self.path}: line {line_no} corrupt: {e}") from e
                yield ResultRecord.model_validate(data)

    def done_set(self) -> set[tuple[str, str, int]]:
        return {(r.model_id, r.task_id, r.repeat) for r in self.read()}
