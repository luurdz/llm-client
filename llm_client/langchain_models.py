from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from llm_client.schemas import LLMConfig

CHAT_MODELS: dict[str, type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
    "gemini": ChatGoogleGenerativeAI,
}


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


if __name__ == "__main__":
    from dotenv import load_dotenv

    from llm_client.config import load_config

    load_dotenv()
    model = build_chat_model(load_config())
    print(type(model).__name__)