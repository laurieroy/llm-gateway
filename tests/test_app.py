from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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


def test_temperature_and_telemetry_have_help_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH)).run()

    assert "Controls variation" in app.slider[0].help
    assert len(app.metric) == 5
    assert all(metric.help for metric in app.metric)


def test_temperature_is_disabled_for_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    app = AppTest.from_file(str(APP_PATH)).run()
    app.selectbox[0].select("Anthropic").run()

    assert app.slider[0].disabled
    assert "Anthropic's Messages API does not currently accept temperature" in (
        app.slider[0].help
    )
