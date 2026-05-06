import json
from pathlib import Path

from typer.testing import CliRunner

from llm_eval.cli import app

runner = CliRunner()
DATA = Path(__file__).parent / "data"


def test_report_generates_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rec = {
        "model_id": "gpt-mini",
        "task_id": "t1",
        "repeat": 1,
        "ts": "2026-05-06T18:00:00Z",
        "score": 1.0,
        "details": {"type": "exact", "tags": ["instruction_following"]},
        "latency_sec": 0.4,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "reasoning_tokens": 0,
        },
        "cost_usd": 0.0001,
        "cost_source": "config",
        "judge_cost_usd": 0.0,
        "output": "OK",
        "reasoning_content": None,
        "notes": [],
    }
    (run_dir / "results.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    # Provide config so the markdown filter section can render.
    (run_dir / "models.yaml").write_text(
        (DATA / "models_minimal.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    result = runner.invoke(app, ["report", str(run_dir), "--config", str(run_dir / "models.yaml")])
    assert result.exit_code == 0, result.output
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "leaderboard.csv").exists()
    assert (run_dir / "report.md").exists()
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "gpt-mini" in md
