from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_eval.config import RunConfig, load_run_config

DATA = Path(__file__).parent / "data"


def test_load_minimal() -> None:
    cfg = load_run_config(DATA / "models_minimal.yaml")
    assert isinstance(cfg, RunConfig)
    assert len(cfg.models) == 2
    assert cfg.defaults.concurrency == 3
    assert cfg.defaults.repeats == 1


def test_model_inherits_defaults() -> None:
    cfg = load_run_config(DATA / "models_minimal.yaml")
    cheap = next(m for m in cfg.models if m.id == "cheap-via-or")
    gpt = next(m for m in cfg.models if m.id == "gpt-mini")
    # cheap overrides concurrency, gpt does not
    assert cheap.concurrency == 2
    assert gpt.concurrency == 3  # from defaults


def test_pricing_optional() -> None:
    cfg = load_run_config(DATA / "models_minimal.yaml")
    cheap = next(m for m in cfg.models if m.id == "cheap-via-or")
    assert cheap.use_provider_cost is True
    assert cheap.pricing is None


def test_unknown_field_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("models:\n  - id: x\n    extra: nope\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_run_config(bad)


def test_filter_models() -> None:
    cfg = load_run_config(DATA / "models_minimal.yaml")
    selected = cfg.select_models(only=["gpt-mini"], skip=[])
    assert [m.id for m in selected] == ["gpt-mini"]

    selected = cfg.select_models(only=[], skip=["gpt-mini"])
    assert [m.id for m in selected] == []  # cheap-via-or is disabled

    selected = cfg.select_models(only=["cheap-via-or"], skip=[])
    assert [m.id for m in selected] == ["cheap-via-or"]  # --only ignores enabled flag
