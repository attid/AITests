"""Compare two run directories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from llm_eval.storage import ResultRecord


@dataclass(frozen=True)
class ModelSummary:
    model_id: str
    weighted_score: float
    avg_latency_sec: float
    total_cost_usd: float
    records: int
    errors: int


@dataclass(frozen=True)
class ModelComparison:
    model_id: str
    old: ModelSummary | None
    new: ModelSummary | None

    @property
    def score_delta(self) -> float | None:
        if self.old is None or self.new is None:
            return None
        return self.new.weighted_score - self.old.weighted_score

    @property
    def cost_delta(self) -> float | None:
        if self.old is None or self.new is None:
            return None
        return self.new.total_cost_usd - self.old.total_cost_usd

    @property
    def latency_delta(self) -> float | None:
        if self.old is None or self.new is None:
            return None
        return self.new.avg_latency_sec - self.old.avg_latency_sec


def summarize_by_model(records: list[ResultRecord]) -> dict[str, ModelSummary]:
    by_model: dict[str, list[ResultRecord]] = defaultdict(list)
    for record in records:
        by_model[record.model_id].append(record)

    summaries: dict[str, ModelSummary] = {}
    for model_id, recs in by_model.items():
        weights = [float(r.details.get("weight", 1.0)) for r in recs]
        total_weight = sum(weights)
        weighted_score = (
            sum(r.score * w for r, w in zip(recs, weights, strict=True)) / total_weight
            if total_weight > 0
            else 0.0
        )
        latencies = [r.latency_sec for r in recs if r.latency_sec > 0]
        total_cost = sum(r.cost_usd + r.judge_cost_usd for r in recs)
        errors = sum(1 for r in recs if any(n.startswith("error:") for n in r.notes))
        summaries[model_id] = ModelSummary(
            model_id=model_id,
            weighted_score=weighted_score,
            avg_latency_sec=sum(latencies) / len(latencies) if latencies else 0.0,
            total_cost_usd=total_cost,
            records=len(recs),
            errors=errors,
        )
    return summaries


def compare_model_summaries(
    old: dict[str, ModelSummary], new: dict[str, ModelSummary]
) -> list[ModelComparison]:
    rows = [
        ModelComparison(model_id, old.get(model_id), new.get(model_id)) for model_id in old | new
    ]
    rows.sort(
        key=lambda r: (
            r.score_delta is None,
            -(r.score_delta or 0.0),
            r.model_id,
        )
    )
    return rows
