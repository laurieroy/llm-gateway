import asyncio
from collections.abc import AsyncGenerator
import os
from time import perf_counter
import re
from typing import Literal

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from multi_provider_llm_gateway.pricing import ModelPricing, get_model_pricing
from multi_provider_llm_gateway.schemas import ChunkTelemetry, StreamChunk


class GatewayError(Exception):
    """Base class for gateway errors safe to expose to application callers."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class GatewayProviderError(GatewayError):
    """A provider request failed for a reason other than rate limiting or auth."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.status_code = status_code


class RateLimitExceededError(GatewayProviderError):
    """A provider rejected the request because of rate limiting."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after


class InvalidAPIKeyError(GatewayProviderError):
    """A provider rejected the configured API key."""


class ProviderConnectionError(GatewayProviderError):
    """The gateway could not establish or maintain a provider connection."""


class ProviderTimeoutError(GatewayProviderError):
    """A provider request exceeded the configured timeout."""


class UnsupportedModelError(GatewayError):
    """The requested model does not have configured provider and pricing data."""


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"api[_-]?key[=:]\s*\S+", re.IGNORECASE),
)


def _sanitize(message: object) -> str:
    """Strip anything resembling a credential from an error string."""
    text = str(message)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("***", text)
    return text[:500]


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (ValueError, AttributeError):
        return None


def _status_error_suffix(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    detail = ""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and isinstance(
            inner.get("message"), str
        ):
            detail = _sanitize(inner["message"])
    suffix = f" (status {status_code})" if status_code is not None else ""
    return f"{suffix}: {detail}" if detail else suffix


def _translate_error(
    error: Exception,
    provider_label: str,
    provider: str,
    model: str,
) -> GatewayError | None:
    """Map SDK exceptions onto typed domain errors.

    Returns None for unexpected (non-SDK) exceptions so callers can let them
    propagate raw — those are bugs, not provider failures.
    """
    context = {"provider": provider, "model": model}
    if isinstance(error, (openai.RateLimitError, anthropic.RateLimitError)):
        return RateLimitExceededError(
            f"{provider_label} rate limit exceeded.",
            retry_after=_retry_after(error),
            **context,
        )
    if isinstance(
        error, (openai.AuthenticationError, anthropic.AuthenticationError)
    ):
        return InvalidAPIKeyError(
            f"{provider_label} API key is invalid or missing.",
            status_code=401,
            **context,
        )
    # Timeout subclasses the connection error in both SDKs — check it first.
    if isinstance(error, (openai.APITimeoutError, anthropic.APITimeoutError)):
        return ProviderTimeoutError(
            f"{provider_label} request timed out.", **context
        )
    if isinstance(
        error, (openai.APIConnectionError, anthropic.APIConnectionError)
    ):
        return ProviderConnectionError(
            f"Could not connect to the {provider_label} API.", **context
        )
    if isinstance(error, (openai.APIStatusError, anthropic.APIStatusError)):
        return GatewayProviderError(
            f"{provider_label} request failed{_status_error_suffix(error)}",
            status_code=error.status_code,
            **context,
        )
    if isinstance(error, (openai.APIError, anthropic.APIError)):
        return GatewayProviderError(
            f"{provider_label} request failed.", **context
        )
    if isinstance(error, (openai.OpenAIError, anthropic.AnthropicError)):
        return GatewayProviderError(
            f"{provider_label} client setup or request failed.", **context
        )
    return None


def _encoding_for(model: str) -> tiktoken.Encoding | None:
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


class UnifiedLLMGateway:
    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        anthropic_client: AsyncAnthropic | None = None,
        max_tokens: int = 4096,
        request_timeout: float = 60.0,
    ) -> None:
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self._openai_client = openai_client
        self._anthropic_client = anthropic_client
        self._max_tokens = max_tokens
        self._timeout = request_timeout

    async def __aenter__(self) -> "UnifiedLLMGateway":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None
        if self._anthropic_client is not None:
            await self._anthropic_client.close()
            self._anthropic_client = None

    def _get_openai(self) -> AsyncOpenAI:
        if self._openai_client is None:
            api_key = (
                self._openai_api_key
                if self._openai_api_key is not None
                else os.getenv("OPENAI_API_KEY", "")
            )
            if not api_key.strip():
                raise InvalidAPIKeyError(
                    "OpenAI API key is missing or invalid."
                )
            try:
                self._openai_client = AsyncOpenAI(
                    api_key=api_key, timeout=self._timeout
                )
            except openai.OpenAIError:
                raise InvalidAPIKeyError(
                    "OpenAI API key is missing or invalid."
                ) from None
        return self._openai_client

    def _get_anthropic(self) -> AsyncAnthropic:
        if self._anthropic_client is None:
            api_key = (
                self._anthropic_api_key
                if self._anthropic_api_key is not None
                else os.getenv("ANTHROPIC_API_KEY", "")
            )
            if not api_key.strip():
                raise InvalidAPIKeyError(
                    "Anthropic API key is missing or invalid."
                )
            try:
                self._anthropic_client = AsyncAnthropic(
                    api_key=api_key, timeout=self._timeout
                )
            except anthropic.AnthropicError:
                raise InvalidAPIKeyError(
                    "Anthropic API key is missing or invalid."
                ) from None
        return self._anthropic_client

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
        parts: list[str] = []

        try:
            client = self._get_openai()
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
                        parts.append(text)
                        yield StreamChunk(
                            delta_text=text,
                            provider="openai",
                            model=model,
                            is_final=False,
                        )
        except Exception as error:
            translated = _translate_error(
                error, "OpenAI", pricing.provider, model
            )
            if translated is None:
                raise
            raise translated from None

        # tiktoken is sync CPU work — offload so the loop never stalls. The
        # first call may also fetch BPE ranks over the network.
        encoding = await asyncio.to_thread(_encoding_for, model)
        prompt_tokens_estimate = (
            len(await asyncio.to_thread(encoding.encode, prompt))
            if encoding is not None
            else None
        )

        if input_tokens is not None and output_tokens is not None:
            token_count_source = "provider_api"
        else:
            # Stream ended before the usage chunk — estimate locally.
            input_tokens = prompt_tokens_estimate or 0
            output_tokens = (
                len(
                    await asyncio.to_thread(
                        encoding.encode, "".join(parts)
                    )
                )
                if encoding is not None
                else 0
            )
            token_count_source = "tiktoken_estimate"

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
            token_count_source=token_count_source,
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
        parts: list[str] = []

        try:
            client = self._get_anthropic()
            response = await client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
            )
            async for event in response:
                if event.type == "message_start":
                    usage = event.message.usage
                    input_tokens = (
                        (usage.input_tokens or 0)
                        + (
                            getattr(
                                usage, "cache_creation_input_tokens", None
                            )
                            or 0
                        )
                        + (
                            getattr(usage, "cache_read_input_tokens", None)
                            or 0
                        )
                    )
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta" and event.delta.text:
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        parts.append(event.delta.text)
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
        except Exception as error:
            translated = _translate_error(
                error, "Anthropic", pricing.provider, model
            )
            if translated is None:
                raise
            raise translated from None

        prompt_tokens_estimate: int | None = None
        if input_tokens is not None and output_tokens is not None:
            token_count_source = "provider_api"
        else:
            # Stream ended before usage events arrived — estimate locally.
            encoding = await asyncio.to_thread(_encoding_for, model)
            prompt_tokens_estimate = (
                len(await asyncio.to_thread(encoding.encode, prompt))
                if encoding is not None
                else None
            )
            if input_tokens is None:
                input_tokens = prompt_tokens_estimate or 0
            if output_tokens is None:
                output_tokens = (
                    len(
                        await asyncio.to_thread(
                            encoding.encode, "".join(parts)
                        )
                    )
                    if encoding is not None
                    else 0
                )
            token_count_source = "tiktoken_estimate"

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
            prompt_tokens_estimate=prompt_tokens_estimate,
            token_count_source=token_count_source,
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
        token_count_source: Literal[
            "provider_api", "tiktoken_estimate"
        ] = "provider_api",
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
        total_cost_usd = round(
            (
                input_tokens * pricing.input_usd_per_million_tokens
                + output_tokens * pricing.output_usd_per_million_tokens
            )
            / 1_000_000,
            6,
        )

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
                token_count_source=token_count_source,
            ),
        )
