import asyncio
from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,
    APIError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
    omit,
)

from llm_client.clients.base import BaseLLMClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, StreamChunk


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = AsyncAnthropic(api_key=self.config.api_key.get_secret_value())

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict], str | None]:
        anthropic_messages = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
                continue
            anthropic_messages.append(m.model_dump())
        return anthropic_messages, system_instruction

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        anthropic_messages, system_instruction = self._convert_messages(messages)
        try:
            async with asyncio.timeout(self.config.timeout):
                response = await self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=system_instruction
                    if system_instruction is not None
                    else omit,
                    messages=anthropic_messages,
                )

                return ModelResponse(
                    provider=self.config.provider,
                    model=self.config.model,
                    content=response.content[0].text,
                )
        except (TimeoutError, APIError) as e:
            return self._error_response(self._error_message(e))

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        anthropic_messages, system_instruction = self._convert_messages(messages)
        try:
            async with asyncio.timeout(self.config.timeout):
                async with self._client.messages.stream(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=system_instruction
                    if system_instruction is not None
                    else omit,
                    messages=anthropic_messages,
                ) as stream:
                    async for text in stream.text_stream:
                        yield StreamChunk(content=text)
        except (TimeoutError, APIError) as e:
            yield StreamChunk(error=self._error_message(e))

    def _error_message(self, error: Exception) -> str:
        if isinstance(error, TimeoutError):
            return f"Timeout: no response after {self.config.timeout}s"
        if isinstance(error, RateLimitError):
            return f"Rate limit exceeded: {error}"
        if isinstance(error, APIConnectionError):
            return f"Connection error: {error}"
        if isinstance(error, InternalServerError):
            return f"Anthropic  server error: {error}"
        if isinstance(error, APIError):
            return f"Invalid request: {error}"
        return f"Unexpected error: {error}"
