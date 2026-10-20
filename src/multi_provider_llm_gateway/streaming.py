import asyncio
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import tiktoken

from multi_provider_llm_gateway.gateway import UnifiedLLMGateway
from multi_provider_llm_gateway.pricing import ModelPricing
from multi_provider_llm_gateway.schemas import ChunkTelemetry, StreamChunk


@dataclass(frozen=True, slots=True)
class StreamMetrics:
    ttft_ms: float
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    is_estimate: bool


@dataclass(slots=True)
class StreamResult:
    telemetry: ChunkTelemetry | None = None
    error: Exception | None = None


def start_stream(
    prompt: str, model: str, temperature: float
) -> queue.Queue[StreamChunk | Exception | None]:
    events: queue.Queue[StreamChunk | Exception | None] = queue.Queue()

    async def produce() -> None:
        gateway = UnifiedLLMGateway()
        try:
            async for chunk in gateway.stream(prompt, model, temperature):
                events.put(chunk)
        finally:
            await gateway.aclose()

    def run() -> None:
        try:
            asyncio.run(produce())
        except Exception as error:
            events.put(error)
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()
    return events


def stream_response(
    events: queue.Queue[StreamChunk | Exception | None],
    prompt: str,
    model: str,
    pricing: ModelPricing,
    on_metrics: Callable[[StreamMetrics], None],
) -> tuple[Iterator[str], StreamResult]:
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    prompt_tokens = len(tokenizer.encode(prompt))
    result = StreamResult()
    response = ""
    started_at = time.perf_counter()
    first_token_at: float | None = None
    last_metrics_update = 0.0

    def generate() -> Iterator[str]:
        nonlocal response, first_token_at, last_metrics_update

        while True:
            event = events.get()
            if event is None:
                return
            if isinstance(event, Exception):
                result.error = event
                continue
            if event.is_final:
                if event.telemetry is not None:
                    result.telemetry = event.telemetry
                    telemetry = event.telemetry
                    on_metrics(
                        StreamMetrics(
                            ttft_ms=telemetry.ttft_seconds * 1000,
                            tokens_per_second=telemetry.tokens_per_second,
                            prompt_tokens=telemetry.input_tokens,
                            completion_tokens=telemetry.output_tokens,
                            cost_usd=telemetry.total_cost_usd,
                            is_estimate=False,
                        )
                    )
                continue

            response += event.delta_text
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now

            if now - last_metrics_update >= 0.2:
                completion_tokens = len(tokenizer.encode(response))
                generation_seconds = max(now - first_token_at, 0.001)
                cost_usd = (
                    prompt_tokens * pricing.input_usd_per_million_tokens
                    + completion_tokens * pricing.output_usd_per_million_tokens
                ) / 1_000_000
                on_metrics(
                    StreamMetrics(
                        ttft_ms=(first_token_at - started_at) * 1000,
                        tokens_per_second=completion_tokens / generation_seconds,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=cost_usd,
                        is_estimate=True,
                    )
                )
                last_metrics_update = now
            yield event.delta_text

    return generate(), result
