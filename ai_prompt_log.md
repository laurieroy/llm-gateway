# AI Prompt Log

Note:
I had been using windsurf for a while. It switched to being Devin this summer, and this is my first use of that IDE. The history, as far as I can tell, is not maintained as well as I was used to. So some of these prompts are not recorded verbatim.

I also started the project in VS-code. But on reflection of the requirements, noted that it was not using FastAPI and started over in devin. It built the project similarly, so I merged the two together.

Also, turns out I was coming down with a cold so I wasn't thinking as clearly as I should have been. 

Log of some of the prompts sent to the AI pair-programmer (Devin, SWE-2) and VS-code (auto)

## Initial Prompt
> (from assignment spec)
You are an expert Python AI Infrastructure Engineer. I need an asynchronous, streaming multi-provider SDK wrapper supporting both OpenAI (e.g., gpt-4o-mini, gpt-4o) and Anthropic (e.g., claude-3-5-sonnet, claude-3-5-haiku) or the most recent available models.

Requirements:

Environment: Python 3.11+, typed with Pydantic v2.
Architecture:

A unified class UnifiedLLMGateway with an async generator method:
async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:

Use official async clients (AsyncOpenAI and AsyncAnthropic).
Normalization:

Define a Pydantic schema StreamChunk containing:
delta_text: str (the new text token)
provider: str (e.g., "openai" or "anthropic")
model: str
is_final: bool
telemetry: Optional[ChunkTelemetry]
Normalize streaming events so that differences in chunk structure between OpenAI and Anthropic are completely hidden from the consumer.
Token & Cost Telemetry:

Include a static pricing dictionary with USD rates per 1,000,000 input and output tokens for supported models.
For OpenAI: enable stream usage tracking (stream_options={"include_usage": True}) and use tiktoken for prompt token verification.
For Anthropic: capture message_start for input tokens and accumulate output tokens from stream events.
On the final chunk (is_final=True), provide exact input tokens, output tokens, total latency (seconds), Time-to-First-Token (TTFT), tokens per second, and calculated total request cost ($USD).
Error Handling:

Handle rate limits, invalid API keys, and connection errors by raising typed, custom domain exceptions (e.g., GatewayProviderError, RateLimitExceededError).
Do not leak raw API keys or unhandled stack traces in logs or outputs.
 
Provide clean, production-ready code split into schemas.py, pricing.py, and gateway.py.

## Asked AI abut bugs
- searched manually for openai.OpenAI() and openai.AsyncOpenAI()
- Asked AI to about the OpenAI bug and Anthropic bug.
- Checked Telemetry & Timing Drift
- Asked if Client Instantiation Anti-Pattern

### Bugs noted (some)

1. Anthropic temperature claim — draft disabled the UI slider and dropped the
   parameter on the claim that the Messages API doesn't accept `temperature`.
   It does (0.0–1.0).
2. Retired model IDs — draft catalog listed `claude-3-5-sonnet` /
   `claude-3-5-haiku`, which return 404 `not_found_error` from the Anthropic API.
3. Over-broad error handling — `except anthropic.APIError` flattened
   `NotFoundError` into a generic message, hiding the status code and provider
   detail that made the retired-model bug diagnosable.
4. Hard failure on missing usage — streams that ended before the usage chunk
   raised an error, losing all partial output; now falls back to tiktoken
   estimates marked with `token_count_source="tiktoken_estimate"`.

> Anthropic requests are failing — help me debug why.

**Outcome:** Root-caused two issues. First, the model catalog listed retired
Anthropic model IDs (`claude-3-5-sonnet`, `claude-3-5-haiku`), which return 404
`not_found_error` responses. Second, a broad `except anthropic.APIError`
handler flattened `NotFoundError` into the generic `"Anthropic request
failed."`, hiding the status code and provider detail. Fixes: added current
model pricing entries (`claude-sonnet-5-5`, `claude-haiku-5-5`,
`claude-opus-5-5`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-5`),
expanded the Anthropic dropdown to all six models, added `APIStatusError`
handling that surfaces status code + sanitized provider detail, and made the
streaming layer log sanitized `GatewayError` messages in full while logging
only the type name for unexpected exceptions.

### Prompt

> when i select anthropic, shouldn't i be able to adjust the temperature

**Outcome:** Confirmed a real bug — the UI disabled the temperature slider for
Anthropic based on an incorrect claim that the Messages API doesn't accept
`temperature`, and the gateway silently dropped the parameter anyway. Fixed
`_stream_anthropic` to forward `temperature` to `client.messages.create`,
enabled the slider with the correct 0.0–1.0 range (vs. OpenAI's 0.0–2.0), and
updated tests.

### Prompt

> can open ai really adjust temperature from 0-2

**Outcome:** Verification question. Confirmed against current OpenAI API docs
that Chat Completions accepts `temperature` 0–2 (default 1), with caveats:
reasoning models (o1, o3, GPT-5 family) reject or ignore it, and values above
~1.2 produce degraded output in practice. No code change needed — the slider
range was already correct.

### Prompt

> just to confirm, you are using fastapi, right?

**Outcome:** Correction — the app is Streamlit, not FastAPI. `fastapi` and
`uvicorn` are declared in `pyproject.toml` but unused (the assignment's `uv add`
command lists them, so they were kept).

** At this point I rebuilt the project **




### Prompt

> compare this repo to ../ad331/multi-provider-llm-gateway is one better?  Would you review both projects as far as which is better code style form a season python developer view, has better security and cleaner


### Prompt

> Go ahead

**Outcome:** Executed the port into this repo. Changes: structured error
taxonomy (`status_code`, `retry_after`, `provider`, `model` attrs; new
`ProviderTimeoutError`), single `_translate_error` mapping both SDKs,
`_sanitize` regex scrubbing of provider error detail, lazy `_get_openai` /
`_get_anthropic` client getters with API-key injection and configurable
timeout/`max_tokens`, tiktoken fallback (`token_count_source="tiktoken_estimate"`)
when a stream ends without usage events, `asyncio.to_thread` for encoding, new
`demo.py` canned provider gated by `LLM_GATEWAY_DEMO`, and pluggable `source`
param on `start_stream`. Test suite grew 24 → 32 tests, all passing.

### Prompt
> in general, please review this readme as you were a hiring professional with 
> python knowledge. what would you want to see? not see?

> go ahead and rewrite the readme with your suggestions. I'll update it later with a live link and screenshot. > For right now i don't need to show implementation details. Add how to set up, run locally, run tests
> Move the AI assistance to bottom. perhaps add for AD331, North Seattle College, Oct 2026

### Prompt
> confirm tests: test_gateway.py:
> Test 1: Unified Stream Protocol: Assert that streaming from both providers yields valid StreamChunk objects and concatenates into the expected text response.
> Test 2: Token Count Validation: Feed a deterministic prompt (e.g., 50 specific words) and assert that recorded prompt tokens match tiktoken calculations within a ± 10% tolerance.
> Test 3: Telemetry Sanity: Assert that when is_final=True, total_cost_usd > 0, tokens_per_second > 0, and ttft_ms > 0.

### Prompt
> Also create test_security.py, move security type tests there.
> be sure to cover
> Missing Key Handling: Assert that missing or malformed environment variables raise a descriptive AuthenticationError rather than crashing with unhandled exceptions.
>
> Secret Redaction: Verify that logging or string representations of exceptions never output raw API key substrings.

### Prompt
> thanks. are there any security gaps not checked

### Prompt

> Please examine the repo against these requirements. Tell me what is missing,
> or was done incorrectly. [assignment spec]

**Outcome:** Gap analysis against the assignment rubric. Code satisfies all
technical items (async architecture, telemetry, tests, UI). Missing artifacts:
this log (was empty), deployment (`Dockerfile`/`docker-compose.yml` or a cloud
URL), and three required README sections (Mermaid diagram, pricing matrix
table, audit & verification summary). Minor nits: no `tests/__init__.py`,
trailing space on `.env` in `.gitignore`.

### Prompt

> Can you add the prompts i've sent you to the ai_prompt_log.md?

**Outcome:** This file.

### Prompt 
> Please add **Project Overview & Architecture Diagram:** A Mermaid.js sequence or flow diagram illustrating the client-to-gateway-to-provider streaming pipeline.
