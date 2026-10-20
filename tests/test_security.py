from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import openai
import pytest
from streamlit.testing.v1 import AppTest

from multi_provider_llm_gateway import InvalidAPIKeyError, UnifiedLLMGateway
from multi_provider_llm_gateway import gateway as gateway_module
from multi_provider_llm_gateway import streaming


@pytest.mark.parametrize(
    ("provider", "environment_variable", "model"),
    [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4o-mini"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-3-5-haiku"),
    ],
)
@pytest.mark.parametrize("api_key", [None, "", "   "])
@pytest.mark.asyncio
async def test_missing_or_blank_api_key_raises_descriptive_error(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    environment_variable: str,
    model: str,
    api_key: str | None,
) -> None:
    if api_key is None:
        monkeypatch.delenv(environment_variable, raising=False)
    else:
        monkeypatch.setenv(environment_variable, api_key)

    gateway = UnifiedLLMGateway()
    with pytest.raises(InvalidAPIKeyError, match=f"{provider} API key"):
        async for _ in gateway.stream("Test prompt", model):
            pass


@pytest.mark.parametrize(
    ("provider", "model", "error_type"),
    [
        ("OpenAI", "gpt-4o-mini", openai.AuthenticationError),
        ("Anthropic", "claude-3-5-haiku", anthropic.AuthenticationError),
    ],
)
@pytest.mark.asyncio
async def test_provider_authentication_errors_redact_api_key(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    error_type: type[Exception],
) -> None:
    secret = "sk-test-secret-substring"
    malformed_api_key = f"malformed-{secret}"
    response = SimpleNamespace(
        request=object(),
        status_code=401,
        headers={},
    )
    provider_error = error_type(
        f"Invalid credential {malformed_api_key}",
        response=response,
        body=None,
    )
    create = AsyncMock(side_effect=provider_error)

    if provider == "OpenAI":
        monkeypatch.setenv("OPENAI_API_KEY", malformed_api_key)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )
        monkeypatch.setattr(
            gateway_module, "AsyncOpenAI", lambda: client
        )
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", malformed_api_key)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        monkeypatch.setattr(
            gateway_module, "AsyncAnthropic", lambda: client
        )

    with pytest.raises(InvalidAPIKeyError) as exc_info:
        async for _ in UnifiedLLMGateway().stream("Test prompt", model):
            pass

    assert "API key" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert exc_info.value.__suppress_context__


def test_unexpected_stream_error_is_logged_without_details_or_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-secret-substring"

    class FailingGateway:
        async def stream(
            self, prompt: str, model: str, temperature: float
        ) -> AsyncIterator[object]:
            raise RuntimeError(f"Invalid credential {secret}")
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
    assert secret not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_unexpected_stream_error_displays_safe_message() -> None:
    secret = "sk-test-secret-substring"
    app = AppTest.from_string(
        "from multi_provider_llm_gateway.app import _render_stream_error\n"
        f"_render_stream_error(RuntimeError('Invalid credential {secret}'))"
    ).run()

    assert not app.exception
    assert len(app.error) == 1
    assert "failed unexpectedly" in app.error[0].value
    assert secret not in app.error[0].value
