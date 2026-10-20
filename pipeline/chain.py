import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from llm_client.config import load_config
from llm_client.langchain_models import build_chat_model
from pipeline.prompts import EXTRACTION_PROMPT
from pipeline.schemas import TechnicalEntities

logger = logging.getLogger(__name__)

TRUNCATION_REASONS = {"MAX_TOKENS", "max_tokens"}


class IncompleteOutputError(Exception):
    """Raised when the model response was cut off by the max tokens limit."""


def check_output(output: dict) -> TechnicalEntities:
    reason = output["raw"].response_metadata.get("stop_reason") or output[
        "raw"
    ].response_metadata.get("finish_reason")
    if reason in TRUNCATION_REASONS:
        logger.error(f"The response was truncated. Reason: {reason}")
        raise IncompleteOutputError(
            f"The model response was cut off (reason: {reason}). Increase max_tokens."
        )
    if output["parsing_error"]:
        logger.warning("The response was invalid. Retry")
        raise output["parsing_error"]
    entities = output["parsed"]
    logger.info(
        f"Validation was successful, Technologies found: {len(entities.technologies)} with criticality level {entities.criticality_level.value}"
    )
    return entities


def build_chain() -> Runnable:
    config = load_config()
    model = build_chat_model(config.model_copy(update={"temperature": 0}))
    chain = (
        EXTRACTION_PROMPT
        | model.with_structured_output(TechnicalEntities, include_raw=True)
        | check_output
    ).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=3,
    )
    return chain


async def process_text(text: str) -> TechnicalEntities:
    chain = build_chain()
    logger.info(f"Starting chain with {len(text)} characters")
    result = await chain.ainvoke({"text": text})
    return result
