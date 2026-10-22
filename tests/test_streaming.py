from queue import Queue
from threading import Event

import pytest

from multi_provider_llm_gateway import streaming
from multi_provider_llm_gateway.pricing import ModelPricing
from multi_provider_llm_gateway.schemas import ChunkTelemetry, StreamChunk


def test_tokenization_does_not_delay_text_and_final_metrics_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenization_started = Event()
    finish_tokenization = Event()

    def slow_count_tokens(_model: str, text: str) -> int:
        if text == "prompt":
            tokenization_started.set()
            assert finish_tokenization.wait(timeout=2)
        return 1

    monkeypatch.setattr(streaming, "_count_tokens", slow_count_tokens)
    events: Queue[StreamChunk | Exception | None] = Queue()
    events.put(
        StreamChunk(
            delta_text="Hello",
            provider="demo",
            model="demo-sim",
            is_final=False,
        )
    )
    events.put(
        StreamChunk(
            delta_text="",
            provider="demo",
            model="demo-sim",
            is_final=True,
            telemetry=ChunkTelemetry(
                input_tokens=2,
                output_tokens=1,
                total_tokens=3,
                latency_seconds=1.0,
                ttft_seconds=0.2,
                tokens_per_second=1.25,
                total_cost_usd=0.0,
            ),
        )
    )
    events.put(None)
    metrics = []
    stream, result = streaming.stream_response(
        events,
        "prompt",
        "demo-sim",
        ModelPricing("demo", 0.0, 0.0),
        metrics.append,
    )

    try:
        assert next(stream) == "Hello"
        assert tokenization_started.wait(timeout=2)
        assert next(stream, None) is None
    finally:
        finish_tokenization.set()

    assert result.telemetry is not None
    assert len(metrics) == 1
    assert metrics[0].prompt_tokens == 2
    assert metrics[0].completion_tokens == 1
    assert metrics[0].is_estimate is False
