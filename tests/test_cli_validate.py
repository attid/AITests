import json
from pathlib import Path

from typer.testing import CliRunner

from llm_eval.cli import app

runner = CliRunner()
DATA = Path(__file__).parent / "data"


def _write_tasks(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_validate_ok(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(
        tasks,
        [
            {"id": "a", "type": "exact", "prompt": "p", "expected_text": "OK"},
        ],
    )
    result = runner.invoke(
        app,
        [
            "validate",
            "--config",
            str(DATA / "models_minimal.yaml"),
            "--tasks",
            str(tasks),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 tasks" in result.output
    assert "2 models" in result.output


def test_validate_bad_task(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"a","type":"exact","prompt":"p"}\n', encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "validate",
            "--config",
            str(DATA / "models_minimal.yaml"),
            "--tasks",
            str(tasks),
        ],
    )
    assert result.exit_code != 0
    assert "expected_text" in result.output


def test_validate_duplicate_ids(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(
        tasks,
        [
            {"id": "a", "type": "exact", "prompt": "p", "expected_text": "OK"},
            {"id": "a", "type": "exact", "prompt": "p", "expected_text": "OK"},
        ],
    )
    result = runner.invoke(
        app,
        [
            "validate",
            "--config",
            str(DATA / "models_minimal.yaml"),
            "--tasks",
            str(tasks),
        ],
    )
    assert result.exit_code != 0
    assert "duplicate" in result.output.lower()
