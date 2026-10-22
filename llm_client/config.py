import os

from llm_client.schemas import LLMConfig

API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def load_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider not in API_KEY_ENV_VARS:
        raise ValueError(

            f"Unsupported LLM_PROVIDER '{provider}'. Use one of {', '.join(API_KEY_ENV_VARS)}"
        )
    api_key = os.getenv(API_KEY_ENV_VARS[provider])
    return LLMConfig(provider=provider, model=os.getenv("LLM_MODEL"), api_key=api_key, fallback_model=os.getenv("LLM_FALLBACK_MODEL") or None,)
