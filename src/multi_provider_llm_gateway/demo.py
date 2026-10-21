"""Canned stream provider — exercises the UI without provider API keys.

Only reachable when LLM_GATEWAY_DEMO is set; see app._provider_catalog.
"""

import asyncio
from collections.abc import AsyncIterator
import time

from multi_provider_llm_gateway.schemas import ChunkTelemetry, StreamChunk

DEMO_PROVIDER = "Demo"
DEMO_MODEL = "demo-sim"


async def demo_stream(
    prompt: str, model: str = DEMO_MODEL
) -> AsyncIterator[StreamChunk]:
    """Emit a fixed response word-by-word followed by synthetic telemetry."""
    started_at = time.perf_counter()
    text = (
        f"This is a simulated response from '{model}'. No provider API was "
        "called — set OPENAI_API_KEY or ANTHROPIC_API_KEY in a .env file and "
        "pick a real model to stream live output. Your prompt was "
        f"{len(prompt)} characters long."
    )
    words = text.split(" ")
    first_token_at: float | None = None
    for index, word in enumerate(words):
        await asyncio.sleep(0.04)
        if first_token_at is None:
            first_token_at = time.perf_counter()
        yield StreamChunk(
            delta_text=word + (" " if index < len(words) - 1 else ""),
            provider=DEMO_PROVIDER,
            model=model,
            is_final=False,
        )

    completed_at = time.perf_counter()
    latency_seconds = completed_at - started_at
    ttft_seconds = (
        first_token_at - started_at if first_token_at is not None else 0.0
    )
    generation_seconds = max(latency_seconds - ttft_seconds, 0.0)
    input_tokens = max(1, len(prompt) // 4)
    yield StreamChunk(
        delta_text="",
        provider=DEMO_PROVIDER,
        model=model,
        is_final=True,
        telemetry=ChunkTelemetry(
            input_tokens=input_tokens,
            output_tokens=len(words),
            total_tokens=input_tokens + len(words),
            latency_seconds=latency_seconds,
            ttft_seconds=ttft_seconds,
            tokens_per_second=(
                len(words) / generation_seconds if generation_seconds else 0.0
            ),
            total_cost_usd=0.0,
            prompt_tokens_estimate=input_tokens,
            token_count_source="tiktoken_estimate",
        ),
    )
