"""Typer-based CLI: run | report | validate."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from llm_eval.client import CallError, CallResult, LLMClient
from llm_eval.compare import compare_model_summaries, summarize_by_model
from llm_eval.config import LlmJudgeTask, ModelConfig, RunConfig, Task, load_run_config, parse_task
from llm_eval.pricing import compute_cost
from llm_eval.reporting import (
    write_leaderboard_csv,
    write_markdown_report,
    write_results_csv,
)
from llm_eval.runner import Runner
from llm_eval.scoring.preprocess import strip_thinking
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
    # On --resume, the run dir already has a snapshot of which models were
    # selected. Use it as source of truth so we don't accidentally start
    # firing requests at models that weren't part of the original run.
    snapshot = out / "models.yaml" if out is not None else None
    if resume and snapshot is not None and snapshot.exists():
        cfg = load_run_config(snapshot)
        console.print(f"[dim]Resume: using config snapshot at {snapshot}[/dim]")
    else:
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

    # Snapshot the run config — but only the models actually being run, plus
    # the judge model if separate. Mirrors what really happened, so a future
    # `llm-eval report` or `--resume` reads a faithful record.
    selected_ids = {m.id for m in selected}
    judge_id = cfg.judge.model_id if cfg.judge else None
    snapshot_models = list(selected)
    if judge_id and judge_id not in selected_ids:
        judge_model = next((m for m in cfg.models if m.id == judge_id), None)
        if judge_model is not None:
            snapshot_models.append(judge_model)
    snapshot = cfg.model_dump(exclude_none=True, by_alias=True)
    snapshot["models"] = [m.model_dump(exclude_none=True, by_alias=True) for m in snapshot_models]
    (out / "models.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    runner_obj = Runner(config=cfg, models=selected, tasks=task_list, out_dir=out, resume=resume)
    asyncio.run(runner_obj.run())

    # Auto-generate the report so the user doesn't have to remember a second step.
    records = list(ResultStore(out / "results.jsonl").read())
    if records:
        write_results_csv(records, out / "results.csv")
        write_leaderboard_csv(records, out / "leaderboard.csv")
        write_markdown_report(records, cfg, out / "report.md", run_dir=out)
        console.print(
            f"[green]Done.[/green] Report: [bold]{out / 'report.md'}[/bold] "
            f"(also results.csv, leaderboard.csv)"
        )
    else:
        console.print(f"[yellow]Done but no records to report.[/yellow] {out}")


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


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _fmt_delta(value: float | None, digits: int) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


@app.command()
def compare(
    old_run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    new_run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Compare model-level score, cost, and latency between two run dirs."""
    old_jsonl = old_run_dir / "results.jsonl"
    new_jsonl = new_run_dir / "results.jsonl"
    for jsonl in (old_jsonl, new_jsonl):
        if not jsonl.exists():
            console.print(f"[red]No results.jsonl in {jsonl.parent}[/red]")
            raise typer.Exit(code=1)

    old_records = list(ResultStore(old_jsonl).read())
    new_records = list(ResultStore(new_jsonl).read())
    if not old_records or not new_records:
        console.print("[yellow]Both runs must contain at least one record[/yellow]")
        raise typer.Exit(code=1)

    rows = compare_model_summaries(
        summarize_by_model(old_records),
        summarize_by_model(new_records),
    )
    console.print(
        "model,old_score,new_score,score_delta,old_cost,new_cost,cost_delta,"
        "old_latency,new_latency,latency_delta,old_records,new_records,old_errors,new_errors"
    )
    for row in rows:
        old = row.old
        new = row.new
        console.print(
            ",".join(
                [
                    row.model_id,
                    _fmt(old.weighted_score if old else None, 4),
                    _fmt(new.weighted_score if new else None, 4),
                    _fmt_delta(row.score_delta, 4),
                    _fmt(old.total_cost_usd if old else None, 6),
                    _fmt(new.total_cost_usd if new else None, 6),
                    _fmt_delta(row.cost_delta, 6),
                    _fmt(old.avg_latency_sec if old else None, 3),
                    _fmt(new.avg_latency_sec if new else None, 3),
                    _fmt_delta(row.latency_delta, 3),
                    str(old.records if old else 0),
                    str(new.records if new else 0),
                    str(old.errors if old else 0),
                    str(new.errors if new else 0),
                ]
            )
        )


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

    async def _ping_one(model: ModelConfig) -> tuple[ModelConfig, str | None, CallResult | None]:
        client = LLMClient(model)
        try:
            result = await client.call(
                prompt="Reply with the single word: OK",
                system_prompt="Respond with exactly the word OK and nothing else.",
                max_tokens=128,
                attempts=1,
            )
        except CallError as e:
            return model, str(e), None
        return model, None, result

    async def _run_all() -> list[tuple[ModelConfig, str | None, CallResult | None]]:
        return await asyncio.gather(*(_ping_one(m) for m in selected))

    results = asyncio.run(_run_all())

    any_failed = False
    for model, err, result in results:
        if err is not None:
            console.print(f"[red]✗[/red] {model.id}: {err}")
            any_failed = True
            continue
        if result is None:
            console.print(f"[red]✗[/red] {model.id}: empty result")
            any_failed = True
            continue
        cleaned = strip_thinking(result.output).strip()
        if cleaned != "OK":
            preview = result.output.replace("\n", " ").strip()[:80]
            reasoning = (result.reasoning_content or "").replace("\n", " ").strip()[:80]
            console.print(
                f"[red]✗[/red] {model.id}: unexpected output {preview!r}"
                + (f" reasoning={reasoning!r}" if reasoning else "")
            )
            any_failed = True
            continue
        cost, source = compute_cost(
            model,
            prompt_tokens=result.usage.prompt_tokens,  # type: ignore[union-attr]
            completion_tokens=result.usage.completion_tokens,  # type: ignore[union-attr]
            provider_cost=result.provider_cost,  # type: ignore[union-attr]
        )
        latency = result.latency_sec  # type: ignore[union-attr]
        console.print(
            f"[green]✓[/green] {model.id}: {cleaned!r} ({latency:.2f}s, ${cost:.6f} {source})"
        )

    raise typer.Exit(code=1 if any_failed else 0)


if __name__ == "__main__":
    app()
