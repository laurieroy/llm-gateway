from collections.abc import AsyncIterator
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from multi_provider_llm_gateway import (
    GatewayProviderError,
    StreamChunk,
    UnifiedLLMGateway,
)
from multi_provider_llm_gateway import streaming


async def _events(*events: object) -> AsyncIterator[object]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_openai_stream_normalizes_text_and_includes_usage() -> None:
    stream = _events(
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="Hello"))
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=None))
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        ),
    )
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )

    chunks = [
        chunk
        async for chunk in UnifiedLLMGateway(openai_client=client).stream(
            "Say hello", "gpt-4o-mini"
        )
    ]

    assert [chunk.delta_text for chunk in chunks] == ["Hello", ""]
    assert all(isinstance(chunk, StreamChunk) for chunk in chunks)
    assert chunks[-1].is_final
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.input_tokens == 12
    assert chunks[-1].telemetry.output_tokens == 3
    assert chunks[-1].telemetry.prompt_tokens_estimate is not None
    assert chunks[-1].telemetry.total_cost_usd == pytest.approx(0.0000036)
    create.assert_awaited_once()
    assert create.await_args.kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_anthropic_stream_uses_message_usage_events() -> None:
    stream = _events(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10)),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hi"),
        ),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=4),
        ),
        SimpleNamespace(type="message_stop"),
    )
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    chunks = [
        chunk
        async for chunk in UnifiedLLMGateway(anthropic_client=client).stream(
            "Say hi", "claude-3-5-haiku"
        )
    ]

    assert [chunk.delta_text for chunk in chunks] == ["Hi", ""]
    assert chunks[-1].is_final
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.input_tokens == 10
    assert chunks[-1].telemetry.output_tokens == 4
    assert chunks[-1].telemetry.total_cost_usd == pytest.approx(0.000024)
    create.assert_awaited_once()
    assert create.await_args.kwargs == {
        "model": "claude-3-5-haiku",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": True,
    }


@pytest.mark.asyncio
async def test_missing_provider_usage_raises_instead_of_returning_estimates() -> None:
    stream = _events(
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="Hello"))
            ],
            usage=None,
        )
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=stream)),
        )
    )
    chunks: list[StreamChunk] = []

    with pytest.raises(GatewayProviderError, match="without token usage"):
        async for chunk in UnifiedLLMGateway(openai_client=client).stream(
            "Say hello", "gpt-4o-mini"
        ):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert not chunks[0].is_final


def test_unexpected_stream_error_is_logged_without_details_or_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingGateway:
        async def stream(
            self, prompt: str, model: str, temperature: float
        ) -> AsyncIterator[StreamChunk]:
            raise RuntimeError("sensitive error details")
            yield

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(streaming, "UnifiedLLMGateway", FailingGateway)

    events = streaming.start_stream("prompt", "gpt-4o-mini", 0.7)
    emitted = []
    while (event := events.get(timeout=2)) is not None:
        emitted.append(event)

    assert len(emitted) == 1
    assert isinstance(emitted[0], RuntimeError)
    assert "RuntimeError" in caplog.text
    assert "sensitive error details" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
