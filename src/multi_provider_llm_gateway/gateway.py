from collections.abc import AsyncGenerator
from time import perf_counter

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from multi_provider_llm_gateway.pricing import ModelPricing, get_model_pricing
from multi_provider_llm_gateway.schemas import ChunkTelemetry, StreamChunk


class GatewayError(Exception):
    """Base class for gateway errors safe to expose to application callers."""


class GatewayProviderError(GatewayError):
    """A provider request failed for a reason other than rate limiting or auth."""


class RateLimitExceededError(GatewayError):
    """A provider rejected the request because of rate limiting."""


class InvalidAPIKeyError(GatewayError):
    """A provider rejected the configured API key."""


class ProviderConnectionError(GatewayError):
    """The gateway could not establish or maintain a provider connection."""


class UnsupportedModelError(GatewayError):
    """The requested model does not have configured provider and pricing data."""


class UnifiedLLMGateway:
    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        anthropic_client: AsyncAnthropic | None = None,
    ) -> None:
        self._openai_client = openai_client
        self._anthropic_client = anthropic_client

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        pricing = get_model_pricing(model)
        if pricing is None:
            raise UnsupportedModelError(f"Unsupported model: {model}")

        started_at = perf_counter()
        if pricing.provider == "openai":
            async for chunk in self._stream_openai(
                prompt, model, temperature, pricing, started_at
            ):
                yield chunk
        else:
            async for chunk in self._stream_anthropic(
                prompt, model, temperature, pricing, started_at
            ):
                yield chunk

    async def aclose(self) -> None:
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None

    async def _stream_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        pricing: ModelPricing,
        started_at: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        first_token_at: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        prompt_tokens_estimate = len(
            tiktoken.encoding_for_model(model).encode(prompt)
        )

        try:
            client = self._openai_client
            if client is None:
                try:
                    client = AsyncOpenAI()
                except openai.OpenAIError:
                    raise InvalidAPIKeyError(
                        "OpenAI API key is missing or invalid."
                    ) from None
                self._openai_client = client

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in response:
                if event.usage is not None:
                    input_tokens = event.usage.prompt_tokens
                    output_tokens = event.usage.completion_tokens

                for choice in event.choices:
                    text = choice.delta.content
                    if isinstance(text, str) and text:
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        yield StreamChunk(
                            delta_text=text,
                            provider="openai",
                            model=model,
                            is_final=False,
                        )
        except openai.RateLimitError:
            raise RateLimitExceededError("OpenAI rate limit exceeded.") from None
        except openai.AuthenticationError:
            raise InvalidAPIKeyError("OpenAI API key is invalid or missing.") from None
        except openai.APIConnectionError:
            raise ProviderConnectionError(
                "Could not connect to the OpenAI API."
            ) from None
        except openai.APIError:
            raise GatewayProviderError("OpenAI request failed.") from None
        except openai.OpenAIError:
            raise GatewayProviderError(
                "OpenAI client setup or request failed."
            ) from None

        if input_tokens is None or output_tokens is None:
            raise GatewayProviderError(
                "OpenAI stream ended without token usage information."
            )

        completed_at = perf_counter()
        yield self._final_chunk(
            provider="openai",
            model=model,
            pricing=pricing,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            started_at=started_at,
            completed_at=completed_at,
            first_token_at=first_token_at,
            prompt_tokens_estimate=prompt_tokens_estimate,
        )

    async def _stream_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
        pricing: ModelPricing,
        started_at: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        first_token_at: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            client = self._anthropic_client
            if client is None:
                try:
                    client = AsyncAnthropic()
                except anthropic.AnthropicError:
                    raise InvalidAPIKeyError(
                        "Anthropic API key is missing or invalid."
                    ) from None
                self._anthropic_client = client

            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
            )
            async for event in response:
                if event.type == "message_start":
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta" and event.delta.text:
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        yield StreamChunk(
                            delta_text=event.delta.text,
                            provider="anthropic",
                            model=model,
                            is_final=False,
                        )
                elif event.type == "message_delta":
                    event_output_tokens = event.usage.output_tokens
                    if event_output_tokens is not None:
                        output_tokens = max(
                            output_tokens or 0, event_output_tokens
                        )
        except anthropic.RateLimitError:
            raise RateLimitExceededError("Anthropic rate limit exceeded.") from None
        except anthropic.AuthenticationError:
            raise InvalidAPIKeyError(
                "Anthropic API key is invalid or missing."
            ) from None
        except anthropic.APIConnectionError:
            raise ProviderConnectionError(
                "Could not connect to the Anthropic API."
            ) from None
        except anthropic.APIError:
            raise GatewayProviderError("Anthropic request failed.") from None
        except anthropic.AnthropicError:
            raise GatewayProviderError(
                "Anthropic client setup or request failed."
            ) from None

        if input_tokens is None or output_tokens is None:
            raise GatewayProviderError(
                "Anthropic stream ended without token usage information."
            )

        completed_at = perf_counter()
        yield self._final_chunk(
            provider="anthropic",
            model=model,
            pricing=pricing,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            started_at=started_at,
            completed_at=completed_at,
            first_token_at=first_token_at,
        )

    @staticmethod
    def _final_chunk(
        *,
        provider: str,
        model: str,
        pricing: ModelPricing,
        input_tokens: int,
        output_tokens: int,
        started_at: float,
        completed_at: float,
        first_token_at: float | None,
        prompt_tokens_estimate: int | None = None,
    ) -> StreamChunk:
        latency_seconds = completed_at - started_at
        ttft_seconds = (
            first_token_at - started_at
            if first_token_at is not None
            else latency_seconds
        )
        generation_seconds = max(latency_seconds - ttft_seconds, 0.0)
        tokens_per_second = (
            output_tokens / generation_seconds if generation_seconds else 0.0
        )
        total_cost_usd = (
            input_tokens * pricing.input_usd_per_million_tokens
            + output_tokens * pricing.output_usd_per_million_tokens
        ) / 1_000_000

        return StreamChunk(
            delta_text="",
            provider=provider,
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_seconds=latency_seconds,
                ttft_seconds=ttft_seconds,
                tokens_per_second=tokens_per_second,
                total_cost_usd=total_cost_usd,
                prompt_tokens_estimate=prompt_tokens_estimate,
            ),
        )
