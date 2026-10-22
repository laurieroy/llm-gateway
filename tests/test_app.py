from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from multi_provider_llm_gateway import streaming
from multi_provider_llm_gateway.demo import DEMO_MODEL, demo_stream
from multi_provider_llm_gateway.schemas import StreamChunk

APP_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "multi_provider_llm_gateway"
    / "app.py"
)


@pytest.mark.parametrize(
    ("provider", "environment_variable"),
    [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_missing_provider_api_key_shows_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    environment_variable: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception

    app.selectbox[0].select(provider).run()
    app.text_area[0].set_value("Test prompt")
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert any(environment_variable in error.value for error in app.error)


def test_empty_prompt_shows_validation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("  \n")
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert any(
        "Enter a prompt before generating a response." in warning.value
        for warning in app.warning
    )


def test_temperature_and_telemetry_have_help_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH)).run()

    assert "Controls variation" in app.slider[0].help
    assert len(app.metric) == 5
    assert all(metric.help for metric in app.metric)


@pytest.mark.parametrize(
    ("provider", "max_value"),
    [("OpenAI", 2.0), ("Anthropic", 1.0)],
)
def test_temperature_range_matches_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    max_value: float,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH)).run()
    app.selectbox[0].select(provider).run()

    assert not app.slider[0].disabled
    assert app.slider[0].min == 0.0
    assert app.slider[0].max == max_value


def test_demo_provider_hidden_unless_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_GATEWAY_DEMO", raising=False)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert "Demo" not in app.selectbox[0].options

    monkeypatch.setenv("LLM_GATEWAY_DEMO", "1")
    app = AppTest.from_file(str(APP_PATH)).run()
    assert "Demo" in app.selectbox[0].options


@pytest.mark.asyncio
async def test_demo_stream_emits_text_then_final_telemetry() -> None:
    chunks = [chunk async for chunk in demo_stream("hi there")]

    assert chunks[-1].is_final
    assert "".join(chunk.delta_text for chunk in chunks[:-1])
    telemetry = chunks[-1].telemetry
    assert telemetry is not None
    assert telemetry.output_tokens > 0
    assert telemetry.token_count_source == "tiktoken_estimate"
    assert telemetry.total_cost_usd == 0.0


def test_start_stream_consumes_custom_source() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            delta_text="hi",
            provider="demo",
            model=DEMO_MODEL,
            is_final=False,
        )

    events = streaming.start_stream("prompt", DEMO_MODEL, 0.7, source)
    emitted = []
    while (event := events.get(timeout=2)) is not None:
        emitted.append(event)

    assert [chunk.delta_text for chunk in emitted] == ["hi"]
