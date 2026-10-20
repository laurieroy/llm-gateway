from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    provider: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


PRICING_USD_PER_MILLION: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(
        provider="openai",
        input_usd_per_million_tokens=0.15,
        output_usd_per_million_tokens=0.60,
    ),
    "gpt-4o": ModelPricing(
        provider="openai",
        input_usd_per_million_tokens=2.50,
        output_usd_per_million_tokens=10.00,
    ),
    "claude-sonnet-5-5": ModelPricing(
        provider="anthropic",
        input_usd_per_million_tokens=2.00,
        output_usd_per_million_tokens=10.00,
    ),
    "claude-haiku-5-5": ModelPricing(
        provider="anthropic",
        input_usd_per_million_tokens=0.10,
        output_usd_per_million_tokens=0.50,
    ),
    "claude-opus-5-5": ModelPricing(
        provider="anthropic",
        input_usd_per_million_tokens=4.00,
        output_usd_per_million_tokens=20.00,
    ),
    "claude-sonnet-4-6": ModelPricing(
        provider="anthropic",
        input_usd_per_million_tokens=3.00,
        output_usd_per_million_tokens=15.00,
    ),
    "claude-haiku-4-5": ModelPricing(
        provider="anthropic",
        input_usd_per_million_tokens=1.00,
        output_usd_per_million_tokens=5.00,
    ),
    "claude-opus-4-5": ModelPricing(
        provider="anthropic",
        input_usd_per_million_tokens=5.00,
        output_usd_per_million_tokens=25.00,
    ),
}


def get_model_pricing(model: str) -> ModelPricing | None:
    """Resolve a supported model or a dated model ID to its listed pricing."""
    pricing = PRICING_USD_PER_MILLION.get(model)
    if pricing is not None:
        return pricing

    for model_prefix, model_pricing in PRICING_USD_PER_MILLION.items():
        if model.startswith(f"{model_prefix}-"):
            return model_pricing
    return None
