import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_eval.cli import app
from llm_eval.storage import ResultStore

runner = CliRunner()
DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")


def test_run_creates_jsonl(tmp_path: Path, httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "t1",
                "type": "exact",
                "prompt": "p",
                "expected_text": "OK",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "run1"

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DATA / "models_minimal.yaml"),
            "--tasks",
            str(tasks),
            "--out",
            str(out_dir),
            "--only",
            "gpt-mini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "results.jsonl").exists()
    rows = list(ResultStore(out_dir / "results.jsonl").read())
    assert len(rows) == 1


def test_run_copies_config_to_run_dir(tmp_path: Path, httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "t1", "type": "exact", "prompt": "p", "expected_text": "OK"}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DATA / "models_minimal.yaml"),
            "--tasks",
            str(tasks),
            "--out",
            str(out_dir),
            "--only",
            "gpt-mini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "models.yaml").exists()
    # Sanity: re-loadable
    from llm_eval.config import load_run_config

    cfg = load_run_config(out_dir / "models.yaml")
    assert any(m.id == "gpt-mini" for m in cfg.models)


def test_run_fails_fast_on_missing_api_key(tmp_path: Path, monkeypatch) -> None:
    """Run must abort with a clear message before creating a run dir if any
    selected model's api_key_env is unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "t1", "type": "exact", "prompt": "p", "expected_text": "OK"}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(DATA / "models_minimal.yaml"),
            "--tasks",
            str(tasks),
            "--out",
            str(out_dir),
            "--only",
            "gpt-mini",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "OPENAI_API_KEY" in result.output
    assert "Missing API keys" in result.output
    # Must not create the run dir when bailing out early.
    assert not out_dir.exists()
