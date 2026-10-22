import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import queue
import sys
import threading
import time
import traceback
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass

import tiktoken

from multi_provider_llm_gateway.gateway import GatewayError, UnifiedLLMGateway
from multi_provider_llm_gateway.pricing import ModelPricing
from multi_provider_llm_gateway.schemas import ChunkTelemetry, StreamChunk

logger = logging.getLogger(__name__)
_TOKENIZER_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="llm-tokenizer"
)


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


async def _gateway_source(
    prompt: str, model: str, temperature: float
) -> AsyncIterator[StreamChunk]:
    gateway = UnifiedLLMGateway()
    try:
        async for chunk in gateway.stream(prompt, model, temperature):
            yield chunk
    finally:
        await gateway.aclose()


def start_stream(
    prompt: str,
    model: str,
    temperature: float,
    source: Callable[[], AsyncIterator[StreamChunk]] | None = None,
) -> queue.Queue[StreamChunk | Exception | None]:
    """Feed an async chunk producer through a queue on a worker thread.

    `source` defaults to the real gateway; tests and the demo provider can
    inject any async iterator of StreamChunks and reuse the same plumbing.
    """
    events: queue.Queue[StreamChunk | Exception | None] = queue.Queue()
    factory = (
        source
        if source is not None
        else lambda: _gateway_source(prompt, model, temperature)
    )

    async def produce() -> None:
        async for chunk in factory():
            events.put(chunk)

    def run() -> None:
        try:
            asyncio.run(produce())
        except Exception as error:
            if isinstance(error, GatewayError):
                logger.error("LLM stream failed: %s", error)
            else:
                error_type = type(error)
                error_module = sys.modules.get(error_type.__module__)
                gateway_module = sys.modules.get(GatewayError.__module__)
                traceback_summary = " <- ".join(
                    f"{frame.filename}:{frame.lineno} in {frame.name}"
                    for frame in traceback.extract_tb(error.__traceback__)
                )
                logger.error(
                    "LLM stream failed with unexpected exception type "
                    "%s.%s (module_file=%s, expected_gateway_error=%s@%x "
                    "from %s, exception_mro=%s, traceback=%s)",
                    error_type.__module__,
                    error_type.__qualname__,
                    getattr(error_module, "__file__", "<unknown>"),
                    GatewayError.__qualname__,
                    id(GatewayError),
                    getattr(gateway_module, "__file__", "<unknown>"),
                    " -> ".join(
                        f"{base.__module__}.{base.__qualname__}@{id(base):x}"
                        for base in error_type.__mro__
                    ),
                    traceback_summary or "<no traceback>",
                )
            events.put(error)
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()
    return events


def _count_tokens(model: str, text: str) -> int:
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def stream_response(
    events: queue.Queue[StreamChunk | Exception | None],
    prompt: str,
    model: str,
    pricing: ModelPricing,
    on_metrics: Callable[[StreamMetrics], None],
) -> tuple[Iterator[str], StreamResult]:
    result = StreamResult()
    response = ""
    started_at = time.perf_counter()
    first_token_at: float | None = None
    prompt_tokens: int | None = None
    last_metrics_update: float | None = None
    prompt_tokens_future: Future[int] | None = _TOKENIZER_EXECUTOR.submit(
        _count_tokens, model, prompt
    )
    completion_tokens_future: Future[int] | None = None
    completion_snapshot_at: float | None = None

    def generate() -> Iterator[str]:
        nonlocal response
        nonlocal first_token_at
        nonlocal last_metrics_update
        nonlocal prompt_tokens_future
        nonlocal completion_tokens_future
        nonlocal completion_snapshot_at

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
                if prompt_tokens_future is not None:
                    prompt_tokens_future.cancel()
                    prompt_tokens_future = None
                if completion_tokens_future is not None:
                    completion_tokens_future.cancel()
                    completion_tokens_future = None
                continue

            response += event.delta_text
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now

            if (
                prompt_tokens_future is not None
                and prompt_tokens_future.done()
            ):
                prompt_tokens = prompt_tokens_future.result()
                prompt_tokens_future = None

            if (
                completion_tokens_future is not None
                and completion_tokens_future.done()
            ):
                completion_tokens = completion_tokens_future.result()
                completion_tokens_future = None
                if prompt_tokens is not None:
                    assert completion_snapshot_at is not None
                    assert first_token_at is not None
                    generation_seconds = max(
                        completion_snapshot_at - first_token_at, 0.001
                    )
                    cost_usd = (
                        prompt_tokens
                        * pricing.input_usd_per_million_tokens
                        + completion_tokens
                        * pricing.output_usd_per_million_tokens
                    ) / 1_000_000
                    on_metrics(
                        StreamMetrics(
                            ttft_ms=(first_token_at - started_at) * 1000,
                            tokens_per_second=(
                                completion_tokens / generation_seconds
                            ),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cost_usd=cost_usd,
                            is_estimate=True,
                        )
                    )

            if (
                prompt_tokens_future is None
                and prompt_tokens is not None
                and completion_tokens_future is None
                and (
                    last_metrics_update is None
                    or now - last_metrics_update >= 0.2
                )
            ):
                completion_tokens_future = _TOKENIZER_EXECUTOR.submit(
                    _count_tokens, model, response
                )
                completion_snapshot_at = now
                last_metrics_update = now
            yield event.delta_text

    return generate(), result
