"""Resolve per-call cost. Priority: provider-reported > config-declared > unknown."""

from __future__ import annotations

from typing import Literal

from llm_eval.config import ModelConfig

CostSource = Literal["provider", "config", "unknown"]


def compute_cost(
    model: ModelConfig,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    provider_cost: float | None,
) -> tuple[float, CostSource]:
    if model.use_provider_cost and provider_cost is not None:
        return provider_cost, "provider"
    if model.pricing is not None:
        cost = (
            prompt_tokens / 1_000_000 * model.pricing.input_per_million
            + completion_tokens / 1_000_000 * model.pricing.output_per_million
        )
        return cost, "config"
    return 0.0, "unknown"
