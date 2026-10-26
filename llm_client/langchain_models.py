from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from llm_client.schemas import LLMConfig

CHAT_MODELS: dict[str, type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
    "gemini": ChatGoogleGenerativeAI,
}

EMBEDDING_MODELS: dict[str, type[Embeddings]] = {
    "openai": OpenAIEmbeddings,
    "gemini": GoogleGenerativeAIEmbeddings,
}


def build_embeddings(config: LLMConfig) -> Embeddings:
    if config.provider not in EMBEDDING_MODELS:
        raise ValueError(
            f"Provider '{config.provider}' has no embedding model. Use one of {', '.join(EMBEDDING_MODELS)}"
        )

    model_class = EMBEDDING_MODELS[config.provider]

    if config.embedding_model is None:
        raise ValueError("EMBEDDING_MODEL is not set in the .env file")
    params = {
        "model": config.embedding_model,
        "timeout": config.timeout,
        "api_key": config.api_key,
    }
    return model_class(**params)


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    model_class = CHAT_MODELS[config.provider]

    params = {
        "model": config.model,
        "timeout": config.timeout,
        "api_key": config.api_key,
        "max_tokens": config.max_tokens,
    }
    if config.provider != "anthropic":
        params["temperature"] = config.temperature

    return model_class(**params)
