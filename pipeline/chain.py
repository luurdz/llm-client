import asyncio

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from llm_client.config import load_config
from llm_client.langchain_models import build_chat_model
from pipeline.prompts import EXTRACTION_PROMPT
from pipeline.schemas import TechnicalEntities

TRUNCATION_REASONS = {"MAX_TOKENS", "max_tokens"}

class IncompleteOutputError (Exception):
    """Raised when the model response was cut off by the max tokens limit."""

def check_output(output: dict) -> TechnicalEntities:
    reason= output["raw"].response_metadata.get("stop_reason") or output["raw"].response_metadata.get("finish_reason")
    if reason in TRUNCATION_REASONS:
        raise IncompleteOutputError(
            f"The model response was cut off (reason: {reason}). Increase max_tokens."
        )
    if output["parsing_error"]:
        raise output["parsing_error"]
    return output["parsed"]


def build_chain() -> Runnable:
    config = load_config()
    model = build_chat_model(config.model_copy(update={"temperature": 0}))
    chain = (EXTRACTION_PROMPT | model.with_structured_output(TechnicalEntities, include_raw=True) | check_output).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=3
    )
    return chain


async def process_text(text: str) -> TechnicalEntities:
    chain = build_chain()
    result = await chain.ainvoke({"text": text})
    return result


async def main():
    load_dotenv()

    example = "SQL Server está caido"
    result = await process_text(example)

    print(result.model_dump_json(by_alias=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())