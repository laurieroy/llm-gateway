import asyncio
import queue
import threading
import time
from collections.abc import Iterator

import streamlit as st
import tiktoken
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator

from multi_provider_llm_gateway import GatewayError, StreamChunk, UnifiedLLMGateway
from multi_provider_llm_gateway.pricing import ModelPricing, get_model_pricing
from multi_provider_llm_gateway.schemas import ChunkTelemetry

MODEL_OPTIONS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
}


def _tokenizer_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _estimated_cost(
    prompt_tokens: int, completion_tokens: int, pricing: ModelPricing
) -> float:
    return (
        prompt_tokens * pricing.input_usd_per_million_tokens
        + completion_tokens * pricing.output_usd_per_million_tokens
    ) / 1_000_000


def _start_stream(
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


def _render_dashboard(
    slots: list[DeltaGenerator],
    ttft: str,
    speed: str,
    prompt_tokens: str,
    completion_tokens: str,
    cost: str,
    *,
    prompt_label: str = "Prompt tokens",
    completion_label: str = "Completion tokens",
    cost_label: str = "Execution cost",
) -> None:
    labels = (
        "TTFT",
        "Generation speed",
        prompt_label,
        completion_label,
        cost_label,
    )
    values = (ttft, speed, prompt_tokens, completion_tokens, cost)
    for slot, label, value in zip(slots, labels, values, strict=True):
        slot.metric(label, value)


def _stream_response(
    events: queue.Queue[StreamChunk | Exception | None],
    prompt: str,
    model: str,
    pricing: ModelPricing,
    dashboard: list[DeltaGenerator],
) -> tuple[Iterator[str], dict[str, ChunkTelemetry | Exception]]:
    tokenizer = _tokenizer_for(model)
    prompt_token_estimate = len(tokenizer.encode(prompt))
    result: dict[str, ChunkTelemetry | Exception] = {}
    response = ""
    started_at = time.perf_counter()
    first_token_at: float | None = None
    last_dashboard_update = 0.0

    def generate() -> Iterator[str]:
        nonlocal response, first_token_at, last_dashboard_update

        while True:
            event = events.get()
            if event is None:
                return
            if isinstance(event, Exception):
                result["error"] = event
                continue
            if event.is_final:
                if event.telemetry is not None:
                    result["telemetry"] = event.telemetry
                    telemetry = event.telemetry
                    _render_dashboard(
                        dashboard,
                        f"{telemetry.ttft_seconds * 1000:.0f} ms",
                        f"{telemetry.tokens_per_second:.2f} tokens/s",
                        str(telemetry.input_tokens),
                        str(telemetry.output_tokens),
                        f"${telemetry.total_cost_usd:.6f}",
                    )
                continue

            response += event.delta_text
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now

            if now - last_dashboard_update >= 0.2:
                completion_estimate = len(tokenizer.encode(response))
                generation_seconds = max(now - first_token_at, 0.001)
                speed_estimate = completion_estimate / generation_seconds
                estimated_cost = _estimated_cost(
                    prompt_token_estimate, completion_estimate, pricing
                )
                _render_dashboard(
                    dashboard,
                    f"~{(first_token_at - started_at) * 1000:.0f} ms",
                    f"~{speed_estimate:.2f} tokens/s",
                    f"~{prompt_token_estimate}",
                    f"~{completion_estimate}",
                    f"~${estimated_cost:.6f}",
                    prompt_label="Prompt tokens (estimate)",
                    completion_label="Completion tokens (estimate)",
                    cost_label="Execution cost (estimate)",
                )
                last_dashboard_update = now
            yield event.delta_text

    return generate(), result


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="LLM Gateway", layout="wide")
    st.title("Multi-provider LLM Gateway")
    st.caption("Stream a response and inspect its live generation telemetry.")

    st.sidebar.header("Live telemetry")
    dashboard = [st.sidebar.empty() for _ in range(5)]
    _render_dashboard(dashboard, "—", "—", "—", "—", "—")
    st.sidebar.caption(
        "Streaming metrics are estimates until provider-reported usage arrives. "
        "Cost uses the gateway's configured model pricing."
    )

    provider_column, model_column = st.columns(2)
    with provider_column:
        provider = st.selectbox("Provider", list(MODEL_OPTIONS))
    with model_column:
        model = st.selectbox("Model", MODEL_OPTIONS[provider])

    with st.form("generation_form"):
        prompt = st.text_area(
            "Prompt",
            placeholder="What would you like the model to do?",
            height=180,
        )
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1
        )
        submitted = st.form_submit_button(
            "Generate response", type="primary", use_container_width=True
        )

    if not submitted:
        return
    if not prompt.strip():
        st.warning("Enter a prompt before generating a response.")
        return

    pricing = get_model_pricing(model)
    if pricing is None:
        st.error(f"No pricing is configured for {model}.")
        return

    st.subheader("Response")
    events = _start_stream(prompt, model, temperature)
    stream, result = _stream_response(events, prompt, model, pricing, dashboard)
    st.write_stream(stream)

    error = result.get("error")
    if isinstance(error, GatewayError):
        st.error(str(error))
    elif isinstance(error, Exception):
        raise error
    elif "telemetry" not in result:
        st.error("The stream ended without final usage telemetry.")


if __name__ == "__main__":
    main()
