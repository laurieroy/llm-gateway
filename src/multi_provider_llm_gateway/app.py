import streamlit as st
from dotenv import load_dotenv
from streamlit.delta_generator import DeltaGenerator

from multi_provider_llm_gateway import GatewayError
from multi_provider_llm_gateway.pricing import get_model_pricing
from multi_provider_llm_gateway.streaming import (
    StreamMetrics,
    start_stream,
    stream_response,
)

MODEL_OPTIONS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku"],
}


def _render_dashboard(
    slots: list[DeltaGenerator],
    metrics: StreamMetrics | None = None,
) -> None:
    if metrics is None:
        values = ("—", "—", "—", "—", "—")
        labels = (
            "TTFT",
            "Generation speed",
            "Prompt tokens",
            "Completion tokens",
            "Execution cost",
        )
    else:
        prefix = "~" if metrics.is_estimate else ""
        suffix = " (estimate)" if metrics.is_estimate else ""
        values = (
            f"{prefix}{metrics.ttft_ms:.0f} ms",
            f"{prefix}{metrics.tokens_per_second:.2f} tokens/s",
            f"{prefix}{metrics.prompt_tokens}",
            f"{prefix}{metrics.completion_tokens}",
            f"{prefix}${metrics.cost_usd:.6f}",
        )
        labels = (
            "TTFT",
            "Generation speed",
            f"Prompt tokens{suffix}",
            f"Completion tokens{suffix}",
            f"Execution cost{suffix}",
        )

    for slot, label, value in zip(slots, labels, values, strict=True):
        slot.metric(label, value)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="LLM Gateway", layout="wide")
    st.title("Multi-provider LLM Gateway")
    st.caption("Stream a response and inspect its live generation telemetry.")

    st.sidebar.header("Live telemetry")
    dashboard = [st.sidebar.empty() for _ in range(5)]
    _render_dashboard(dashboard)
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
    events = start_stream(prompt, model, temperature)
    stream, result = stream_response(
        events,
        prompt,
        model,
        pricing,
        lambda metrics: _render_dashboard(dashboard, metrics),
    )
    st.write_stream(stream)

    error = result.error
    if isinstance(error, GatewayError):
        st.error(str(error))
    elif isinstance(error, Exception):
        raise error
    elif result.telemetry is None:
        st.error("The stream ended without final usage telemetry.")


if __name__ == "__main__":
    main()
