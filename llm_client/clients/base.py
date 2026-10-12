from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from llm_client.schemas import ChatMessage, LLMConfig, ModelResponse, StreamChunk


class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse: ...

    @abstractmethod
    def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    def _error_message(self, error: Exception) -> str: ...

    def _error_response(self, message: str) -> ModelResponse:
        return ModelResponse(
            provider=self.config.provider, model=self.config.model, error=message
        )
