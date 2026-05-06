import json
from datetime import datetime
from pathlib import Path

import pytest

from llm_eval.config import load_run_config, parse_task
from llm_eval.runner import Runner
from llm_eval.storage import ResultStore

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


async def test_runner_single_model_single_task(httpx_mock, tmp_path: Path) -> None:
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )

    cfg = load_run_config(DATA / "models_minimal.yaml")
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
    tasks = [
        parse_task(
            {
                "id": "t1",
                "type": "exact",
                "prompt": "say OK",
                "expected_text": "OK",
            }
        )
    ]

    out_dir = tmp_path / "run1"
    runner = Runner(
        config=cfg,
        models=selected,
        tasks=tasks,
        out_dir=out_dir,
    )
    await runner.run()

    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 1
    assert rows[0].score == 1.0
    assert rows[0].cost_source == "config"
    assert rows[0].cost_usd > 0


async def test_runner_repeats(httpx_mock, tmp_path: Path) -> None:
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        is_reusable=True,
    )

    cfg = load_run_config(DATA / "models_minimal.yaml")
    cfg.defaults.repeats = 3
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
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

    out_dir = tmp_path / "run"
    runner = Runner(config=cfg, models=selected, tasks=tasks, out_dir=out_dir)
    await runner.run()

    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 3
    assert {r.repeat for r in rows} == {1, 2, 3}


async def test_runner_resume_skips_done(httpx_mock, tmp_path: Path) -> None:
    cfg = load_run_config(DATA / "models_minimal.yaml")
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
    tasks = [
        parse_task(
            {
                "id": "t1",
                "type": "exact",
                "prompt": "p",
                "expected_text": "OK",
            }
        ),
        parse_task(
            {
                "id": "t2",
                "type": "exact",
                "prompt": "p",
                "expected_text": "OK",
            }
        ),
    ]

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    # Pre-seed JSONL: t1 already done.
    pre = {
        "model_id": "gpt-mini",
        "task_id": "t1",
        "repeat": 1,
        "ts": datetime.now().isoformat(),
        "score": 1.0,
        "details": {},
        "latency_sec": 0.1,
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "reasoning_tokens": 0,
        },
        "cost_usd": 0.0,
        "cost_source": "config",
        "judge_cost_usd": 0.0,
        "output": "OK",
        "reasoning_content": None,
        "notes": [],
    }
    (out_dir / "results.jsonl").write_text(json.dumps(pre) + "\n", encoding="utf-8")

    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    runner = Runner(config=cfg, models=selected, tasks=tasks, out_dir=out_dir, resume=True)
    await runner.run()

    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 2  # 1 pre-seeded + 1 new
    new_rows = [r for r in rows if r.task_id == "t2"]
    assert len(new_rows) == 1


async def test_runner_records_call_error(httpx_mock, tmp_path: Path) -> None:
    httpx_mock.add_response(status_code=400, json={"error": "bad"})

    cfg = load_run_config(DATA / "models_minimal.yaml")
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
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

    out_dir = tmp_path / "run"
    runner = Runner(config=cfg, models=selected, tasks=tasks, out_dir=out_dir)
    await runner.run()

    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 1
    assert rows[0].score == 0.0
    assert any("error" in n for n in rows[0].notes)


async def test_runner_records_unexpected_scoring_error(
    httpx_mock, tmp_path: Path, monkeypatch
) -> None:
    """If scoring blows up unexpectedly, runner records error and continues."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("scoring exploded")

    import llm_eval.runner

    monkeypatch.setattr(llm_eval.runner, "score_async", boom)

    cfg = load_run_config(DATA / "models_minimal.yaml")
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
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
    out_dir = tmp_path / "run"
    runner = Runner(config=cfg, models=selected, tasks=tasks, out_dir=out_dir)
    await runner.run()  # must NOT raise

    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 1
    assert rows[0].score == 0.0
    assert any("unexpected" in n for n in rows[0].notes)
    assert any("RuntimeError" in n for n in rows[0].notes)


async def test_runner_propagates_tags_and_weight(httpx_mock, tmp_path: Path) -> None:
    """Runner must inject task.tags and task.weight into details for reporting."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    cfg = load_run_config(DATA / "models_minimal.yaml")
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
    tasks = [
        parse_task(
            {
                "id": "t1",
                "type": "exact",
                "prompt": "p",
                "expected_text": "OK",
                "tags": ["json_format", "anti_hallucination"],
                "weight": 1.5,
            }
        )
    ]
    out_dir = tmp_path / "run"
    runner = Runner(config=cfg, models=selected, tasks=tasks, out_dir=out_dir)
    await runner.run()

    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 1
    assert rows[0].details.get("tags") == ["json_format", "anti_hallucination"]
    assert rows[0].details.get("weight") == 1.5
