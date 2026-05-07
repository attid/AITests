"""Typer-based CLI: run | report | validate."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from llm_eval.client import CallError, LLMClient
from llm_eval.config import LlmJudgeTask, ModelConfig, RunConfig, Task, load_run_config, parse_task
from llm_eval.pricing import compute_cost
from llm_eval.reporting import (
    write_leaderboard_csv,
    write_markdown_report,
    write_results_csv,
)
from llm_eval.runner import Runner
from llm_eval.storage import ResultStore

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _check_api_keys(
    cfg: RunConfig, selected: list[ModelConfig], tasks_have_judge: bool
) -> list[str]:
    """Return list of human-readable error lines for any missing API keys."""
    needed: dict[str, str] = {}  # env_var -> first model id requiring it
    for m in selected:
        needed.setdefault(m.api_key_env, m.id)
    if tasks_have_judge and cfg.judge is not None:
        judge_model = next((m for m in cfg.models if m.id == cfg.judge.model_id), None)
        if judge_model is not None:
            needed.setdefault(judge_model.api_key_env, f"judge:{judge_model.id}")
    missing = [(var, mid) for var, mid in needed.items() if not os.environ.get(var)]
    return [f"  - {var} (needed by {mid})" for var, mid in missing]


def _load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(f"{path}:{line_no} bad JSON: {e}") from e
            try:
                task = parse_task(data)
            except ValidationError as e:
                raise typer.BadParameter(f"{path}:{line_no} bad task: {e}") from e
            if task.id in seen_ids:
                raise typer.BadParameter(f"{path}:{line_no} duplicate task id: {task.id}")
            seen_ids.add(task.id)
            tasks.append(task)
    return tasks


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    tasks: Path = typer.Option(..., "--tasks", exists=True, dir_okay=False, readable=True),
) -> None:
    """Validate models.yaml + tasks.jsonl. Exit 0 on success."""
    cfg = load_run_config(config)
    task_list = _load_tasks(tasks)
    console.print(f"[green]OK[/green]: {len(task_list)} tasks, {len(cfg.models)} models")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    tasks: Path = typer.Option(..., "--tasks", exists=True, dir_okay=False, readable=True),
    out: Path | None = typer.Option(None, "--out", help="Output dir; default runs/<timestamp>"),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing JSONL in --out"),
    only: str = typer.Option("", "--only", help="Comma-separated model IDs to keep"),
    skip: str = typer.Option("", "--skip", help="Comma-separated model IDs to exclude"),
) -> None:
    """Run all enabled models against tasks; write JSONL stream."""
    cfg = load_run_config(config)
    task_list = _load_tasks(tasks)

    only_ids = [s for s in only.split(",") if s]
    skip_ids = [s for s in skip.split(",") if s]
    selected = cfg.select_models(only=only_ids, skip=skip_ids)
    if not selected:
        console.print("[red]No models selected[/red]")
        raise typer.Exit(code=2)

    has_judge = any(isinstance(t, LlmJudgeTask) for t in task_list)
    missing_lines = _check_api_keys(cfg, selected, has_judge)
    if missing_lines:
        console.print("[red]Missing API keys:[/red]")
        for line in missing_lines:
            console.print(line)
        console.print(
            "[yellow]Set them via env (export FOO=...) or disable models in models.yaml.[/yellow]"
        )
        raise typer.Exit(code=2)

    if out is None:
        # Microseconds avoid collisions when two runs start in the same second.
        out = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Run dir:[/cyan] {out}")
    console.print(f"[cyan]Models:[/cyan] {', '.join(m.id for m in selected)}")
    console.print(f"[cyan]Tasks:[/cyan] {len(task_list)} x {cfg.defaults.repeats} repeats")
    console.print(
        f"[dim]If it hangs, Ctrl+C and resume with:[/dim] "
        f"[bold]just run --out {out} --resume[/bold]"
    )

    shutil.copyfile(config, out / "models.yaml")

    runner_obj = Runner(config=cfg, models=selected, tasks=task_list, out_dir=out, resume=resume)
    asyncio.run(runner_obj.run())
    console.print(f"[green]Done.[/green] Run report with: [bold]llm-eval report {out}[/bold]")


@app.command()
def report(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="models.yaml for filter thresholds; defaults to <run_dir>/models.yaml",
    ),
) -> None:
    """Generate results.csv, leaderboard.csv, report.md from run_dir/results.jsonl."""
    jsonl = run_dir / "results.jsonl"
    if not jsonl.exists():
        console.print(f"[red]No results.jsonl in {run_dir}[/red]")
        raise typer.Exit(code=1)

    records = list(ResultStore(jsonl).read())
    if not records:
        console.print("[yellow]No records to report[/yellow]")
        raise typer.Exit(code=1)

    write_results_csv(records, run_dir / "results.csv")
    write_leaderboard_csv(records, run_dir / "leaderboard.csv")

    cfg_path = config or (run_dir / "models.yaml")
    cfg = load_run_config(cfg_path) if cfg_path.exists() else RunConfig(models=[])

    write_markdown_report(records, cfg, run_dir / "report.md", run_dir=run_dir)
    console.print(f"[green]Wrote:[/green] results.csv, leaderboard.csv, report.md in {run_dir}")


@app.command()
def ping(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
    only: str = typer.Option("", "--only", help="Comma-separated model IDs to keep"),
    skip: str = typer.Option("", "--skip", help="Comma-separated model IDs to exclude"),
) -> None:
    """Send a 1-token request to each enabled model. Verifies keys + connectivity."""
    cfg = load_run_config(config)
    only_ids = [s for s in only.split(",") if s]
    skip_ids = [s for s in skip.split(",") if s]
    selected = cfg.select_models(only=only_ids, skip=skip_ids)
    if not selected:
        console.print("[red]No models selected[/red]")
        raise typer.Exit(code=2)

    missing_lines = _check_api_keys(cfg, selected, tasks_have_judge=False)
    if missing_lines:
        console.print("[red]Missing API keys:[/red]")
        for line in missing_lines:
            console.print(line)
        raise typer.Exit(code=2)

    async def _ping_one(model: ModelConfig) -> tuple[ModelConfig, str | None, object]:
        client = LLMClient(model)
        try:
            result = await client.call(
                prompt="Reply with the single word: OK",
                system_prompt="Respond with exactly the word OK and nothing else.",
                max_tokens=10,
                attempts=1,
            )
        except CallError as e:
            return model, str(e), None
        return model, None, result

    async def _run_all() -> list[tuple[ModelConfig, str | None, object]]:
        return await asyncio.gather(*(_ping_one(m) for m in selected))

    results = asyncio.run(_run_all())

    any_failed = False
    for model, err, result in results:
        if err is not None:
            console.print(f"[red]✗[/red] {model.id}: {err}")
            any_failed = True
            continue
        # result is CallResult here
        cost, source = compute_cost(
            model,
            prompt_tokens=result.usage.prompt_tokens,  # type: ignore[union-attr]
            completion_tokens=result.usage.completion_tokens,  # type: ignore[union-attr]
            provider_cost=result.provider_cost,  # type: ignore[union-attr]
        )
        preview = result.output.replace("\n", " ").strip()[:40]  # type: ignore[union-attr]
        latency = result.latency_sec  # type: ignore[union-attr]
        console.print(
            f"[green]✓[/green] {model.id}: {preview!r} ({latency:.2f}s, ${cost:.6f} {source})"
        )

    raise typer.Exit(code=1 if any_failed else 0)


if __name__ == "__main__":
    app()
