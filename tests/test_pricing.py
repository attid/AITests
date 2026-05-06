from llm_eval.config import ModelConfig, Pricing
from llm_eval.pricing import compute_cost


def _model(**overrides) -> ModelConfig:
    base = {
        "id": "m",
        "base_url": "http://x",
        "api_key_env": "K",
        "model": "x",
        "concurrency": 1,
        "temperature": 0.0,
    }
    base.update(overrides)
    return ModelConfig(**base)


def test_provider_cost_used_when_present() -> None:
    m = _model(use_provider_cost=True)
    cost, source = compute_cost(
        model=m,
        prompt_tokens=100,
        completion_tokens=200,
        provider_cost=0.0042,
    )
    assert cost == 0.0042
    assert source == "provider"


def test_config_pricing_when_no_provider_cost() -> None:
    m = _model(pricing=Pricing(input_per_million=0.40, output_per_million=1.60))
    cost, source = compute_cost(
        m, prompt_tokens=1_000_000, completion_tokens=1_000_000, provider_cost=None
    )
    assert cost == 0.40 + 1.60
    assert source == "config"


def test_config_used_when_provider_flag_off() -> None:
    m = _model(
        use_provider_cost=False,
        pricing=Pricing(input_per_million=1.0, output_per_million=2.0),
    )
    cost, source = compute_cost(
        m, prompt_tokens=1_000_000, completion_tokens=500_000, provider_cost=0.99
    )
    # provider cost ignored because use_provider_cost is False
    assert cost == 1.0 + 1.0
    assert source == "config"


def test_unknown_when_neither() -> None:
    m = _model()  # no pricing, no use_provider_cost
    cost, source = compute_cost(m, prompt_tokens=10, completion_tokens=20, provider_cost=None)
    assert cost == 0.0
    assert source == "unknown"


def test_provider_flag_on_but_no_cost_falls_back_to_config() -> None:
    m = _model(
        use_provider_cost=True,
        pricing=Pricing(input_per_million=1.0, output_per_million=1.0),
    )
    cost, source = compute_cost(
        m, prompt_tokens=500_000, completion_tokens=500_000, provider_cost=None
    )
    assert cost == 1.0
    assert source == "config"
