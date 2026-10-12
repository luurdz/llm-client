import asyncio
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, LLMConfig

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
    return LLMConfig(provider=provider, model=os.getenv("LLM_MODEL"), api_key=api_key)


async def main():
    load_dotenv()
    try:
        config = load_config()
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return
    manager = AsyncLLMManager(config)
    messages = [
        ChatMessage(role="system", content="Respondé en 2 o 3 líneas."),
        ChatMessage(role="user", content="¿Qué es la entropía?"),
    ]

    # 1. Normal Mode
    print("=== Normal mode ===")
    response = await manager.generate(messages=messages)

    if response.error:
        print(f"Error: {response.error}")
    else:
        print(response.content)

    print("=== Streaming mode ===")
    # 2. Streaming Mode
    async for chunk in manager.generate_stream(messages=messages):
        if chunk.error:
            print(f"Error: {chunk.error}", end="", flush=True)
        else:
            print(chunk.content, end="", flush=True)
    print()

    print("=== Resilience test (invalid API key) ===")
    # 3. Resilience test
    invalid_config = LLMConfig(
        provider=config.provider, model=config.model, api_key="invalid-key"
    )
    invalid_manager = AsyncLLMManager(invalid_config)
    response = await invalid_manager.generate(messages)
    print(f"Error: {response.error}")
    print("The program is still running after the error.")


if __name__ == "__main__":
    asyncio.run(main())
