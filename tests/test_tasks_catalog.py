import json
from pathlib import Path

from llm_eval.config import JsonExactTask, TextQualityTask, parse_task

ROOT = Path(__file__).resolve().parents[1]


def _load_repo_tasks() -> list:
    tasks_path = ROOT / "tasks.jsonl"
    return [
        parse_task(json.loads(line))
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_skill_readiness_tasks_are_present_and_valid() -> None:
    tasks = _load_repo_tasks()
    by_id = {task.id: task for task in tasks}

    expected_ids = {
        "skill_select_001",
        "skill_select_002",
        "skill_select_003",
        "tool_plan_001",
        "tool_plan_002",
        "skill_guard_001",
        "skill_guard_002",
    }

    assert expected_ids <= set(by_id)
    for task_id in expected_ids:
        task = by_id[task_id]
        assert isinstance(task, JsonExactTask)
        assert "skill_readiness" in task.tags


def test_repo_tasks_have_unique_ids() -> None:
    tasks = _load_repo_tasks()
    ids = [task.id for task in tasks]
    assert len(ids) == len(set(ids))


def test_text_quality_tasks_are_present_and_valid() -> None:
    tasks = _load_repo_tasks()
    by_id = {task.id: task for task in tasks}

    expected_ids = {"textq_001", "textq_002", "textq_003"}

    assert expected_ids <= set(by_id)
    for task_id in expected_ids:
        task = by_id[task_id]
        assert isinstance(task, TextQualityTask)
        assert "text_quality" in task.tags
