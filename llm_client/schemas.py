import typing

from pydantic import BaseModel, Field, SecretStr, StringConstraints

NotEmptyStr = typing.Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
Provider = typing.Literal["openai", "anthropic", "gemini"]


class ChatMessage(BaseModel):
    role: typing.Literal["user", "system", "assistant"]
    content: NotEmptyStr


class LLMConfig(BaseModel):
    provider: Provider
    model: NotEmptyStr
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    api_key: SecretStr = Field(min_length=1)
    timeout: float = Field(default=30, gt=0)


class ModelResponse(BaseModel):
    content: str | None = None
    error: str | None = None
    provider: Provider
    model: str


class StreamChunk(BaseModel):
    content: str | None = None
    error: str | None = None
