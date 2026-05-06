from pathlib import Path

from llm_eval.config import ModelConfig, ReportingConfig, ReportingFilter, RunConfig
from llm_eval.reporting import write_markdown_report
from llm_eval.storage import ResultRecord


def _rec(
    model_id: str,
    task_id: str,
    *,
    score: float,
    task_type: str = "exact",
    tags: list[str] | None = None,
    json_ok: bool | None = None,
) -> ResultRecord:
    details: dict = {"type": task_type, "tags": tags or []}
    if json_ok is not None:
        details["json_ok"] = json_ok
    return ResultRecord(
        model_id=model_id,
        task_id=task_id,
        repeat=1,
        ts="2026-05-06T18:00:00Z",
        score=score,
        details=details,
        latency_sec=0.5,
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "total_tokens": 100,
            "reasoning_tokens": 0,
        },
        cost_usd=0.0001,
        cost_source="config",
        judge_cost_usd=0.0,
        output="ok",
        reasoning_content=None,
        notes=[],
    )


def _config_with_models() -> RunConfig:
    return RunConfig(
        models=[
            ModelConfig(
                id="m1",
                base_url="http://x",
                api_key_env="K",
                model="x",
                concurrency=1,
                temperature=0.0,
            ),
            ModelConfig(
                id="m2",
                base_url="http://y",
                api_key_env="K",
                model="y",
                concurrency=1,
                temperature=0.0,
            ),
        ],
        reporting=ReportingConfig(
            filter=ReportingFilter(
                avg_score_min=0.75,
                json_ok_rate_min=0.9,
                forbidden_fail_max=2,
            )
        ),
    )


def test_markdown_has_header_and_leaderboard(tmp_path: Path) -> None:
    rows = [
        _rec("m1", "t1", score=1.0, task_type="json_exact", tags=["json"], json_ok=True),
        _rec("m1", "t2", score=0.8, task_type="exact", tags=["instruction_following"]),
        _rec("m2", "t1", score=0.5, task_type="json_exact", tags=["json"], json_ok=True),
    ]
    out = tmp_path / "report.md"
    write_markdown_report(rows, _config_with_models(), out, run_dir=tmp_path)
    text = out.read_text(encoding="utf-8")

    assert "# Eval Report" in text
    assert "## Leaderboard" in text
    assert "| m1 |" in text
    assert "| m2 |" in text


def test_markdown_filter_section(tmp_path: Path) -> None:
    rows = [
        _rec("good", "t1", score=0.9, task_type="json_exact", json_ok=True),
        _rec("good", "t2", score=0.9, task_type="json_exact", json_ok=True),
        _rec("bad", "t1", score=0.5, task_type="json_exact", json_ok=False),
        _rec("bad", "t2", score=0.5, task_type="json_exact", json_ok=False),
    ]
    config = RunConfig(
        models=[
            ModelConfig(
                id="good",
                base_url="http://x",
                api_key_env="K",
                model="x",
                concurrency=1,
                temperature=0.0,
            ),
            ModelConfig(
                id="bad",
                base_url="http://y",
                api_key_env="K",
                model="y",
                concurrency=1,
                temperature=0.0,
            ),
        ],
        reporting=ReportingConfig(
            filter=ReportingFilter(
                avg_score_min=0.75,
                json_ok_rate_min=0.9,
                forbidden_fail_max=2,
            )
        ),
    )
    out = tmp_path / "report.md"
    write_markdown_report(rows, config, out, run_dir=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "Passed:" in text
    assert "good" in text.split("Passed:")[1].splitlines()[0]


def test_markdown_top_failures(tmp_path: Path) -> None:
    rows = [_rec("m", f"t{i}", score=0.0) for i in range(3)]
    rows.extend([_rec("m", f"ok{i}", score=1.0) for i in range(2)])
    out = tmp_path / "report.md"
    write_markdown_report(rows, _config_with_models(), out, run_dir=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "Top failures" in text
    assert "t0" in text  # at least one failure listed
