import json
from pathlib import Path

import pytest

from llm_eval.storage import ResultRecord, ResultStore


@pytest.fixture
def record() -> ResultRecord:
    return ResultRecord(
        model_id="m1",
        task_id="t1",
        repeat=1,
        ts="2026-05-06T18:00:00Z",
        score=1.0,
        details={"json_ok": True},
        latency_sec=0.42,
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 0,
        },
        cost_usd=0.0001,
        cost_source="config",
        judge_cost_usd=0.0,
        output="hello",
        reasoning_content=None,
        notes=[],
    )


async def test_append_and_read(tmp_path: Path, record: ResultRecord) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    await store.append(record)
    rec2 = record.model_copy(update={"task_id": "t2"})
    await store.append(rec2)

    rows = list(store.read())
    assert len(rows) == 2
    assert rows[0].task_id == "t1"
    assert rows[1].task_id == "t2"


async def test_done_set_after_resume(tmp_path: Path, record: ResultRecord) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    await store.append(record)
    await store.append(record.model_copy(update={"task_id": "t2", "repeat": 2}))

    store2 = ResultStore(tmp_path / "results.jsonl")
    done = store2.done_set()
    assert done == {("m1", "t1", 1), ("m1", "t2", 2)}


def test_done_set_empty_when_no_file(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "results.jsonl")
    assert store.done_set() == set()


def test_corrupt_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "results.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    store = ResultStore(p)
    with pytest.raises(ValueError, match="line 1"):
        store.done_set()


async def test_flush_per_record(tmp_path: Path, record: ResultRecord) -> None:
    """File contents must be readable after each append (no buffering surprises)."""
    p = tmp_path / "results.jsonl"
    store = ResultStore(p)
    await store.append(record)
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n")
    parsed = json.loads(text.splitlines()[0])
    assert parsed["task_id"] == "t1"
