from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import pytest
import tiktoken

from multi_provider_llm_gateway import (
    GatewayProviderError,
    StreamChunk,
    UnifiedLLMGateway,
)
from multi_provider_llm_gateway import gateway as gateway_module


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
    assert "".join(chunk.delta_text for chunk in chunks) == "Hello"
    assert chunks[-1].is_final
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.input_tokens == 12
    assert chunks[-1].telemetry.output_tokens == 3
    assert chunks[-1].telemetry.prompt_tokens_estimate is not None
    assert chunks[-1].telemetry.total_cost_usd == 0.000004
    create.assert_awaited_once()
    assert create.await_args.kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_openai_prompt_token_estimate_matches_tiktoken_within_ten_percent() -> None:
    prompt = " ".join(f"word{index}" for index in range(50))
    expected_tokens = len(
        tiktoken.encoding_for_model("gpt-4o-mini").encode(prompt)
    )
    stream = _events(
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Done"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=expected_tokens,
                completion_tokens=2,
            ),
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=stream),
            )
        )
    )

    chunks = [
        chunk
        async for chunk in UnifiedLLMGateway(openai_client=client).stream(
            prompt, "gpt-4o-mini"
        )
    ]

    telemetry = chunks[-1].telemetry
    assert telemetry is not None
    assert telemetry.prompt_tokens_estimate is not None
    assert expected_tokens > 0
    assert (
        abs(telemetry.prompt_tokens_estimate - expected_tokens)
        / expected_tokens
        <= 0.10
    )
    assert telemetry.input_tokens == expected_tokens


@pytest.mark.asyncio
async def test_openai_ttft_runs_from_request_start_to_first_nonempty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _events(
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=""))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=stream),
            )
        )
    )
    clock = iter((100.0, 108.0, 112.0))
    monkeypatch.setattr(gateway_module, "perf_counter", lambda: next(clock))

    chunks = [
        chunk
        async for chunk in UnifiedLLMGateway(openai_client=client).stream(
            "Prompt", "gpt-4o-mini"
        )
    ]

    telemetry = chunks[-1].telemetry
    assert telemetry is not None
    assert telemetry.ttft_seconds == 8.0
    assert telemetry.latency_seconds == 12.0


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
            "Say hi", "claude-haiku-5-5", temperature=0.3
        )
    ]

    assert [chunk.delta_text for chunk in chunks] == ["Hi", ""]
    assert all(isinstance(chunk, StreamChunk) for chunk in chunks)
    assert "".join(chunk.delta_text for chunk in chunks) == "Hi"
    assert chunks[-1].is_final
    assert chunks[-1].telemetry is not None
    assert chunks[-1].telemetry.input_tokens == 10
    assert chunks[-1].telemetry.output_tokens == 4
    assert chunks[-1].telemetry.total_cost_usd == pytest.approx(0.000003)
    create.assert_awaited_once()
    assert create.await_args.kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Say hi"}],
        "temperature": 0.3,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_anthropic_ttft_runs_to_first_nonempty_text_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _events(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=1)),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text=""),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hi"),
        ),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(output_tokens=2),
        ),
        SimpleNamespace(type="message_stop"),
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=stream),
        )
    )
    clock = iter((100.0, 108.0, 112.0))
    monkeypatch.setattr(gateway_module, "perf_counter", lambda: next(clock))

    chunks = [
        chunk
        async for chunk in UnifiedLLMGateway(anthropic_client=client).stream(
            "Prompt", "claude-haiku-5-5"
        )
    ]

    telemetry = chunks[-1].telemetry
    assert telemetry is not None
    assert telemetry.ttft_seconds == 8.0
    assert telemetry.latency_seconds == 12.0


@pytest.mark.asyncio
async def test_anthropic_status_error_surfaces_provider_detail() -> None:
    response = SimpleNamespace(
        request=object(),
        status_code=404,
        headers={},
    )
    provider_error = anthropic.NotFoundError(
        "Error code: 404",
        response=response,
        body={
            "type": "error",
            "error": {
                "type": "not_found_error",
                "message": "model: claude-3-5-haiku",
            },
        },
    )
    create = AsyncMock(side_effect=provider_error)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    with pytest.raises(
        GatewayProviderError, match="status 404.*claude-3-5-haiku"
    ):
        async for _ in UnifiedLLMGateway(anthropic_client=client).stream(
            "Say hi", "claude-haiku-5-5"
        ):
            pass


def test_final_chunk_rounds_cost_to_six_decimal_places() -> None:
    pricing = gateway_module.get_model_pricing("gpt-4o-mini")
    assert pricing is not None

    final_chunk = UnifiedLLMGateway._final_chunk(
        provider="openai",
        model="gpt-4o-mini",
        pricing=pricing,
        input_tokens=1,
        output_tokens=3,
        started_at=0.0,
        completed_at=2.0,
        first_token_at=1.0,
    )

    assert final_chunk.telemetry is not None
    telemetry = final_chunk.telemetry
    assert final_chunk.is_final
    assert telemetry.total_cost_usd == 0.000002
    assert telemetry.total_cost_usd > 0
    assert telemetry.tokens_per_second > 0
    assert telemetry.ttft_seconds * 1000 > 0


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
