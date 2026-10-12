import asyncio
from collections.abc import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from llm_client.clients.base import BaseLLMClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, StreamChunk


class GeminiClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = genai.Client(
            api_key=self.config.api_key.get_secret_value(),
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=3),
            ),
        )

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[types.Content], str | None]:
        gemini_messages = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
                continue
            elif m.role == "assistant":
                role = "model"
            else:
                role = m.role
            gemini_messages.append(
                types.Content(role=role, parts=[types.Part(text=m.content)])
            )
        return gemini_messages, system_instruction

    def _build_config(
        self, system_instruction: str | None
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system_instruction,
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        gemini_messages, system_instruction = self._convert_messages(messages)
        try:
            async with asyncio.timeout(self.config.timeout):
                response = await self._client.aio.models.generate_content(
                    model=self.config.model,
                    contents=gemini_messages,
                    config=self._build_config(system_instruction=system_instruction),
                )

                return ModelResponse(
                    provider=self.config.provider,
                    model=self.config.model,
                    content=response.text,
                )
        except (TimeoutError, errors.APIError, httpx.TransportError) as e:
            return self._error_response(self._error_message(e))

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        gemini_messages, system_instruction = self._convert_messages(messages)
        try:
            async with asyncio.timeout(self.config.timeout):
                stream = await self._client.aio.models.generate_content_stream(
                    model=self.config.model,
                    contents=gemini_messages,
                    config=self._build_config(system_instruction=system_instruction),
                )
                async for chunk in stream:
                    if chunk.text:
                        yield StreamChunk(content=chunk.text)
        except (TimeoutError, errors.APIError, httpx.TransportError) as e:
            yield StreamChunk(error=self._error_message(e))

    def _error_message(self, error: Exception) -> str:
        if isinstance(error, TimeoutError):
            return f"Timeout: no response after {self.config.timeout}s"
        if isinstance(error, errors.ClientError) and error.code == 429:
            return f"Rate limit exceeded: {error}"
        if isinstance(error, errors.ClientError):
            return f"Invalid request: {error}"
        if isinstance(error, errors.ServerError):
            return f"Gemini server error: {error}"
        if isinstance(error, httpx.TransportError):
            return f"Connection error: {error}"
        return f"Unexpected error: {error}"
