import asyncio
from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from llm_client.clients.base import BaseLLMClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, StreamChunk


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = AsyncOpenAI(api_key=self.config.api_key.get_secret_value())

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        openai_messages = [m.model_dump() for m in messages]
        try:
            async with asyncio.timeout(self.config.timeout):
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=openai_messages,
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                )

                return ModelResponse(
                    provider=self.config.provider,
                    model=self.config.model,
                    content=response.choices[0].message.content,
                )
        except (TimeoutError, APIError) as e:
            return self._error_response(self._error_message(e))

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        openai_messages = [m.model_dump() for m in messages]
        try:
            async with asyncio.timeout(self.config.timeout):
                stream = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=openai_messages,
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield StreamChunk(content=chunk.choices[0].delta.content)
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
            return f"OpenAI server error: {error}"
        if isinstance(error, APIError):
            return f"Invalid request: {error}"
        return f"Unexpected error: {error}"
