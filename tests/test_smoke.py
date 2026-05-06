"""Real-API smoke test. Off by default; enable with `pytest -m smoke`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_eval.cli import _load_tasks
from llm_eval.config import load_run_config
from llm_eval.runner import Runner
from llm_eval.storage import ResultStore


@pytest.mark.smoke
async def test_smoke_one_task_one_model(tmp_path: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    config = load_run_config(Path("models.yaml"))
    selected = config.select_models(only=["gpt-4-1-mini"], skip=[])
    assert selected, "gpt-4-1-mini should be in models.yaml"

    tasks = _load_tasks(Path("tasks.jsonl"))[:1]

    runner = Runner(
        config=config,
        models=selected,
        tasks=tasks,
        out_dir=tmp_path / "smoke",
    )
    await runner.run()

    rows = list(ResultStore(tmp_path / "smoke" / "results.jsonl").read())
    assert len(rows) == 1
    assert rows[0].model_id == "gpt-4-1-mini"
    # We don't assert score>0; the model might fail, that's fine.
    # We assert the pipeline produced a record.
    assert rows[0].cost_usd >= 0
