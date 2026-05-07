"""Pydantic config models for tasks, models, and runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _TaskBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    max_tokens: int = 500
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0
    forbidden_contains: list[str] = Field(default_factory=list)
    word_boundary: bool = True
    case_sensitive: bool = False


class ExactTask(_TaskBase):
    type: Literal["exact"]
    expected_text: str
    normalize_whitespace: bool = True


class ContainsTask(_TaskBase):
    type: Literal["contains"]
    expected_contains: list[str]
    match_all_required: bool = False


class RegexTask(_TaskBase):
    type: Literal["regex"]
    expected_regex: str


class NumericTolerance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    abs: float = 1e-6
    rel: float = 0.0


class NumericTask(_TaskBase):
    type: Literal["numeric"]
    expected: float
    numeric_tolerance: NumericTolerance = Field(default_factory=NumericTolerance)


class JsonExactTask(_TaskBase):
    type: Literal["json_exact"]
    expected: dict[str, Any] | list[Any]
    numeric_tolerance: NumericTolerance = Field(default_factory=NumericTolerance)
    case_sensitive_strings: bool = False


class JsonSchemaTask(_TaskBase):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")
    expected_values: dict[str, Any] | None = None


class LlmJudgeTask(_TaskBase):
    type: Literal["llm_judge"]
    rubric: str
    judge_model: str | None = None  # overrides global judge


class TextQualityTask(_TaskBase):
    type: Literal["text_quality"]
    min_words: int = 3
    script: Literal["any", "cyrillic", "latin"] = "any"
    min_script_ratio: float = 0.7
    disallow_cjk: bool = True
    disallow_json: bool = True
    disallow_markdown: bool = True


Task = Annotated[
    ExactTask
    | ContainsTask
    | RegexTask
    | NumericTask
    | JsonExactTask
    | JsonSchemaTask
    | LlmJudgeTask
    | TextQualityTask,
    Field(discriminator="type"),
]


_task_adapter: TypeAdapter[Task] = TypeAdapter(Task)


def parse_task(data: dict[str, Any]) -> Task:
    """Parse a single task dict into the appropriate Task model."""
    return _task_adapter.validate_python(data)


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_per_million: float
    output_per_million: float


class RetriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempts: int = 3
    backoff_base: float = 1.0
    respect_retry_after: bool = True


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concurrency: int = 3
    temperature: float = 0.0
    repeats: int = 1
    retries: RetriesConfig = Field(default_factory=RetriesConfig)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str  # references a ModelConfig.id


class ReportingFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    avg_score_min: float = 0.75
    json_ok_rate_min: float = 0.90
    forbidden_fail_max: int = 2


class ReportingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filter: ReportingFilter = Field(default_factory=ReportingFilter)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    enabled: bool = True
    base_url: str
    api_key_env: str
    model: str
    pricing: Pricing | None = None
    use_provider_cost: bool = False
    is_reasoning: bool = False
    response_format: Literal["json_object"] | None = None
    concurrency: int | None = None
    temperature: float | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # Reasoning models burn tokens on chain-of-thought; per-task max_tokens
    # can starve them. If set, raise effective max_tokens to at least this.
    min_max_tokens: int | None = None


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: Defaults = Field(default_factory=Defaults)
    judge: JudgeConfig | None = None
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    models: list[ModelConfig]

    def model_post_init(self, _: Any) -> None:
        # Apply defaults to per-model fields that were left unset.
        for m in self.models:
            if m.concurrency is None:
                m.concurrency = self.defaults.concurrency
            if m.temperature is None:
                m.temperature = self.defaults.temperature

    def select_models(self, *, only: list[str], skip: list[str]) -> list[ModelConfig]:
        """
        Filter models for a run.

        - --only id1,id2: keep only listed IDs (overrides `enabled`).
        - --skip id3: exclude listed IDs (applied after --only).
        - When --only is empty, respect `enabled`.
        """
        if only:
            chosen = [m for m in self.models if m.id in only]
        else:
            chosen = [m for m in self.models if m.enabled]
        return [m for m in chosen if m.id not in skip]


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a models.yaml file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
