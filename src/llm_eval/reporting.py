"""Generate CSV / leaderboard / markdown from JSONL results."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llm_eval.config import RunConfig
from llm_eval.storage import ResultRecord

_RESULTS_FIELDS = [
    "model_id",
    "task_id",
    "task_type",
    "tags",
    "repeat",
    "score",
    "weight",
    "latency_sec",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cost_usd",
    "cost_source",
    "judge_cost_usd",
    "json_ok",
    "forbidden_fail",
    "notes",
    "output_preview",
]

_LEADERBOARD_FIELDS = [
    "model_id",
    "avg_score",
    "weighted_score",
    "json_ok_rate",
    "forbidden_fail_count",
    "judge_pass_rate",
    "avg_latency_sec",
    "p50_latency",
    "p95_latency",
    "total_tokens",
    "total_cost_usd",
    "value_score",
    "tasks_completed",
    "tasks_errored",
]


def _is_forbidden_fail(rec: ResultRecord) -> bool:
    return any(n.startswith("forbidden_contains") for n in rec.notes)


def _output_preview(text: str, limit: int = 200) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r")[:limit]


def write_results_csv(records: Iterable[ResultRecord], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULTS_FIELDS)
        writer.writeheader()
        for r in records:
            details = r.details
            tags = ",".join(details.get("tags", []) or [])
            json_ok = details.get("json_ok", "")
            writer.writerow(
                {
                    "model_id": r.model_id,
                    "task_id": r.task_id,
                    "task_type": details.get("type", ""),
                    "tags": tags,
                    "repeat": r.repeat,
                    "score": round(r.score, 4),
                    "weight": details.get("weight", 1.0),
                    "latency_sec": round(r.latency_sec, 3),
                    "prompt_tokens": r.usage.get("prompt_tokens", 0),
                    "completion_tokens": r.usage.get("completion_tokens", 0),
                    "reasoning_tokens": r.usage.get("reasoning_tokens", 0),
                    "total_tokens": r.usage.get("total_tokens", 0),
                    "cost_usd": round(r.cost_usd, 8),
                    "cost_source": r.cost_source,
                    "judge_cost_usd": round(r.judge_cost_usd, 8),
                    "json_ok": json_ok,
                    "forbidden_fail": int(_is_forbidden_fail(r)),
                    "notes": ";".join(r.notes),
                    "output_preview": _output_preview(r.output),
                }
            )


def write_leaderboard_csv(records: Iterable[ResultRecord], out: Path) -> None:
    by_model: dict[str, list[ResultRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model_id].append(r)

    rows: list[dict[str, Any]] = []
    for model_id, recs in by_model.items():
        scores = [r.score for r in recs]
        weights = [float(r.details.get("weight", 1.0)) for r in recs]
        avg = sum(scores) / len(scores)
        weighted = (
            sum(s * w for s, w in zip(scores, weights, strict=True)) / sum(weights)
            if sum(weights) > 0
            else 0.0
        )
        json_recs = [r for r in recs if str(r.details.get("type", "")).startswith("json_")]
        json_ok_rate = (
            sum(1 for r in json_recs if r.details.get("json_ok") is True) / len(json_recs)
            if json_recs
            else 0.0
        )
        judge_recs = [r for r in recs if r.details.get("type") == "llm_judge"]
        judge_pass_rate = (
            sum(1 for r in judge_recs if r.score >= 0.5) / len(judge_recs) if judge_recs else 0.0
        )
        latencies = [r.latency_sec for r in recs if r.latency_sec > 0]
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted) if latencies_sorted else 0.0
        p95 = (
            latencies_sorted[max(0, int(len(latencies_sorted) * 0.95) - 1)]
            if latencies_sorted
            else 0.0
        )
        total_tokens = sum(int(r.usage.get("total_tokens", 0)) for r in recs)
        total_cost = sum(r.cost_usd + r.judge_cost_usd for r in recs)
        errored = sum(1 for r in recs if any(n.startswith("error:") for n in r.notes))
        forbidden_fail = sum(1 for r in recs if _is_forbidden_fail(r))

        rows.append(
            {
                "model_id": model_id,
                "avg_score": round(avg, 4),
                "weighted_score": round(weighted, 4),
                "json_ok_rate": round(json_ok_rate, 4),
                "forbidden_fail_count": forbidden_fail,
                "judge_pass_rate": round(judge_pass_rate, 4),
                "avg_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                "p50_latency": round(p50, 3),
                "p95_latency": round(p95, 3),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "value_score": round(weighted / max(total_cost, 1e-4), 2),
                "tasks_completed": len(recs) - errored,
                "tasks_errored": errored,
            }
        )

    rows.sort(key=lambda r: (-float(r["weighted_score"]), -float(r["value_score"])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _leaderboard_summary(records: list[ResultRecord]) -> list[dict[str, Any]]:
    by_model: dict[str, list[ResultRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model_id].append(r)
    summary: list[dict[str, Any]] = []
    for model_id, recs in by_model.items():
        scores = [r.score for r in recs]
        weights = [float(r.details.get("weight", 1.0)) for r in recs]
        avg = sum(scores) / len(scores) if scores else 0.0
        weighted = (
            sum(s * w for s, w in zip(scores, weights, strict=True)) / sum(weights)
            if sum(weights) > 0
            else 0.0
        )
        json_recs = [r for r in recs if str(r.details.get("type", "")).startswith("json_")]
        json_ok_rate = (
            sum(1 for r in json_recs if r.details.get("json_ok") is True) / len(json_recs)
            if json_recs
            else 0.0
        )
        forbidden_fail = sum(1 for r in recs if _is_forbidden_fail(r))
        total_cost = sum(r.cost_usd + r.judge_cost_usd for r in recs)
        summary.append(
            {
                "model_id": model_id,
                "avg_score": avg,
                "weighted_score": weighted,
                "json_ok_rate": json_ok_rate,
                "forbidden_fail": forbidden_fail,
                "total_cost": total_cost,
                "value_score": weighted / max(total_cost, 1e-4),
                "records": recs,
            }
        )
    summary.sort(key=lambda s: (-float(s["weighted_score"]), -float(s["value_score"])))
    return summary


def _by_tag_table(records: list[ResultRecord]) -> str:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_tags: set[str] = set()
    for r in records:
        raw_tags: Any = r.details.get("tags")
        tags: list[str] = []
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if not tags:
            tags = ["(untagged)"]
        for tag in tags:
            by_model[r.model_id][tag].append(r.score)
            all_tags.add(tag)
    if not all_tags:
        return "_(no tags)_"
    sorted_tags = sorted(all_tags)
    headers = ["model", *sorted_tags]
    rows: list[list[str]] = []
    for model_id in sorted(by_model.keys()):
        cells = [model_id]
        for tag in sorted_tags:
            scores = by_model[model_id].get(tag, [])
            cell = f"{(sum(scores) / len(scores)):.2f}" if scores else "—"
            cells.append(cell)
        rows.append(cells)
    return _format_table(headers, rows)


def _top_failures(records: list[ResultRecord], k: int = 5) -> str:
    by_model: dict[str, list[ResultRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model_id].append(r)
    lines: list[str] = []
    for model_id in sorted(by_model.keys()):
        recs = sorted(by_model[model_id], key=lambda r: r.score)[:k]
        failures = [r for r in recs if r.score < 1.0]
        if not failures:
            continue
        lines.append(f"### {model_id}")
        for r in failures:
            note = ";".join(r.notes[:2]) or "—"
            preview = r.output.replace("\n", " ")[:80]
            lines.append(f"- `{r.task_id}` (score={r.score:.2f}): {note} | output: {preview!r}")
        lines.append("")
    return "\n".join(lines) if lines else "_(none)_"


def write_markdown_report(
    records: list[ResultRecord],
    config: RunConfig,
    out: Path,
    *,
    run_dir: Path,
) -> None:
    if not records:
        out.write_text("# Eval Report\n\n_(empty)_\n", encoding="utf-8")
        return

    summary = _leaderboard_summary(records)
    total_runs = len(records)
    total_cost = sum(float(s["total_cost"]) for s in summary)
    distinct_tasks = len({r.task_id for r in records})
    distinct_models = len(summary)

    lb_rows = [
        [
            str(s["model_id"]),
            f"{float(s['avg_score']):.2f}",
            f"{float(s['json_ok_rate']) * 100:.0f}%",
            str(s["forbidden_fail"]),
            f"${float(s['total_cost']):.4f}",
            f"{float(s['value_score']):.1f}",
        ]
        for s in summary
    ]
    leaderboard_md = _format_table(
        ["model", "avg_score", "json_ok", "forbidden_fail", "cost", "value"],
        lb_rows,
    )

    f = config.reporting.filter
    passing = [
        s
        for s in summary
        if float(s["avg_score"]) >= f.avg_score_min
        and float(s["json_ok_rate"]) >= f.json_ok_rate_min
        and int(s["forbidden_fail"]) <= f.forbidden_fail_max
    ]
    if passing:
        recommendation = str(min(passing, key=lambda s: float(s["total_cost"]))["model_id"])
    else:
        recommendation = "(none meets filter)"

    parts = [
        f"# Eval Report — {run_dir.name}",
        "",
        (
            f"**Tasks:** {distinct_tasks} · **Models:** {distinct_models} "
            f"· **Total runs:** {total_runs}"
        ),
        f"**Total cost:** ${total_cost:.4f}",
        "",
        "## Leaderboard",
        "",
        leaderboard_md,
        "",
        "## By tag",
        "",
        _by_tag_table(records),
        "",
        "## Top failures (per model)",
        "",
        _top_failures(records),
        "",
        "## Filter applied",
        "",
        (
            f"`avg_score >= {f.avg_score_min} AND "
            f"json_ok_rate >= {f.json_ok_rate_min} AND "
            f"forbidden_fail <= {f.forbidden_fail_max}`"
        ),
        "",
        "**Passed:** " + (", ".join(str(s["model_id"]) for s in passing) if passing else "(none)"),
        f"**Recommendation:** {recommendation}",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
