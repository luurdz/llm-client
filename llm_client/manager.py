from collections.abc import AsyncIterator

from llm_client.clients.anthropic_client import AnthropicClient
from llm_client.clients.base import BaseLLMClient
from llm_client.clients.gemini_client import GeminiClient
from llm_client.clients.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, StreamChunk

CLIENTS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
}


class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        client_class = CLIENTS[config.provider]
        self._client = client_class(config)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in self._client.generate_stream(messages):
            yield chunk
