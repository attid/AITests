import json
from pathlib import Path

from typer.testing import CliRunner

from llm_eval.cli import app

runner = CliRunner()


def _write_record(
    run_dir: Path,
    *,
    model_id: str,
    task_id: str,
    score: float,
    cost_usd: float = 0.001,
    latency_sec: float = 0.5,
    notes: list[str] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "model_id": model_id,
        "task_id": task_id,
        "repeat": 1,
        "ts": "2026-05-06T18:00:00Z",
        "score": score,
        "details": {"type": "exact", "tags": [], "weight": 1.0},
        "latency_sec": latency_sec,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 0,
        },
        "cost_usd": cost_usd,
        "cost_source": "config",
        "judge_cost_usd": 0.0,
        "output": "OK",
        "reasoning_content": None,
        "notes": notes or [],
    }
    with open(run_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def test_compare_reports_score_cost_and_latency_deltas(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_record(old, model_id="m1", task_id="t1", score=0.5, cost_usd=0.01, latency_sec=1.0)
    _write_record(old, model_id="m1", task_id="t2", score=1.0, cost_usd=0.01, latency_sec=3.0)
    _write_record(new, model_id="m1", task_id="t1", score=1.0, cost_usd=0.02, latency_sec=2.0)
    _write_record(new, model_id="m1", task_id="t2", score=1.0, cost_usd=0.02, latency_sec=4.0)

    result = runner.invoke(app, ["compare", str(old), str(new)])

    assert result.exit_code == 0, result.output
    assert "m1" in result.output
    assert "+0.2500" in result.output
    assert "+0.020000" in result.output
    assert "+1.000" in result.output


def test_compare_fails_when_results_missing(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()

    result = runner.invoke(app, ["compare", str(old), str(new)])

    assert result.exit_code == 1
    assert "No results.jsonl" in result.output
