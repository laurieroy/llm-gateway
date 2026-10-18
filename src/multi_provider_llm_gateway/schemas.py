from pydantic import BaseModel, ConfigDict, Field


class ChunkTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    ttft_seconds: float = Field(ge=0)
    tokens_per_second: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    prompt_tokens_estimate: int | None = Field(default=None, ge=0)


class StreamChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_text: str
    provider: str
    model: str
    is_final: bool
    telemetry: ChunkTelemetry | None = None
