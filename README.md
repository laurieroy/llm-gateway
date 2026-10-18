# Multi-provider LLM gateway

An asynchronous streaming wrapper for the supported OpenAI and Anthropic models.

## Configuration

Set the API key for the provider you plan to use:

```dotenv
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

The gateway creates each official async client lazily, so configuring one provider is sufficient when only using that provider. Call `await gateway.aclose()` when finished to close initialized clients.

## Streaming

```python
import asyncio

from multi_provider_llm_gateway import UnifiedLLMGateway


async def main() -> None:
    gateway = UnifiedLLMGateway()
    try:
        async for chunk in gateway.stream(
            "Explain async generators briefly.",
            model="gpt-4o-mini",
        ):
            if chunk.is_final:
                print("\nUsage:", chunk.telemetry)
            else:
                print(chunk.delta_text, end="", flush=True)
    finally:
        await gateway.aclose()


asyncio.run(main())
```

Every response ends with a final chunk containing provider-reported token counts, latency, time-to-first-token, throughput, and estimated cost. OpenAI telemetry also includes `prompt_tokens_estimate`, a local tiktoken count of the prompt;
`input_tokens` remains the provider-reported usage value.

Supported model IDs include `gpt-4o-mini`, `gpt-4o`,
`claude-3-5-sonnet`, and `claude-3-5-haiku`. Dated model IDs beginning with these names use the same pricing entry. The pricing table is a static estimate and should be updated when provider rates change.