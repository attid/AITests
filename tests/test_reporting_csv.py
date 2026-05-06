import csv
from pathlib import Path

from llm_eval.reporting import write_leaderboard_csv, write_results_csv
from llm_eval.storage import ResultRecord


def _rec(
    model_id: str,
    task_id: str,
    repeat: int = 1,
    *,
    score: float = 1.0,
    total_tokens: int = 100,
    cost: float = 0.0001,
    latency: float = 0.5,
    json_ok: bool | None = None,
    task_type: str = "exact",
    tags: list[str] | None = None,
    notes: list[str] | None = None,
) -> ResultRecord:
    details: dict = {"type": task_type, "tags": tags or [], "notes": notes or []}
    if json_ok is not None:
        details["json_ok"] = json_ok
    return ResultRecord(
        model_id=model_id,
        task_id=task_id,
        repeat=repeat,
        ts="2026-05-06T18:00:00Z",
        score=score,
        details=details,
        latency_sec=latency,
        usage={
            "prompt_tokens": total_tokens // 2,
            "completion_tokens": total_tokens // 2,
            "total_tokens": total_tokens,
            "reasoning_tokens": 0,
        },
        cost_usd=cost,
        cost_source="config",
        judge_cost_usd=0.0,
        output="ok" * 50,
        reasoning_content=None,
        notes=notes or [],
    )


def test_results_csv(tmp_path: Path) -> None:
    rows = [_rec("m1", "t1"), _rec("m1", "t2", score=0.0)]
    out = tmp_path / "results.csv"
    write_results_csv(rows, out)

    with open(out) as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[0]["model_id"] == "m1"
    assert "output_preview" in reader[0]
    assert len(reader[0]["output_preview"]) <= 200


def test_leaderboard_aggregates(tmp_path: Path) -> None:
    rows = [
        _rec(
            "m1",
            "t1",
            score=1.0,
            json_ok=True,
            task_type="json_exact",
            total_tokens=100,
            cost=0.001,
            latency=0.4,
        ),
        _rec(
            "m1",
            "t2",
            score=0.5,
            json_ok=True,
            task_type="json_exact",
            total_tokens=120,
            cost=0.002,
            latency=0.6,
        ),
        _rec(
            "m2",
            "t1",
            score=1.0,
            json_ok=False,
            task_type="json_exact",
            total_tokens=200,
            cost=0.0005,
            latency=1.0,
            notes=["forbidden_contains:x"],
        ),
    ]
    out = tmp_path / "leaderboard.csv"
    write_leaderboard_csv(rows, out)

    with open(out) as f:
        reader = list(csv.DictReader(f))
    by_model = {r["model_id"]: r for r in reader}

    assert float(by_model["m1"]["avg_score"]) == 0.75
    assert float(by_model["m2"]["avg_score"]) == 1.0
    assert float(by_model["m1"]["json_ok_rate"]) == 1.0
    assert float(by_model["m2"]["json_ok_rate"]) == 0.0
    assert int(by_model["m2"]["forbidden_fail_count"]) == 1
    # leaderboard sorted by weighted_score DESC
    assert reader[0]["model_id"] == "m2"


def test_leaderboard_value_score(tmp_path: Path) -> None:
    rows = [
        _rec("expensive", "t", score=1.0, cost=1.0),
        _rec("cheap", "t", score=0.9, cost=0.001),
    ]
    out = tmp_path / "lb.csv"
    write_leaderboard_csv(rows, out)
    with open(out) as f:
        reader = list(csv.DictReader(f))
    by_model = {r["model_id"]: r for r in reader}
    assert float(by_model["cheap"]["value_score"]) > float(by_model["expensive"]["value_score"])
