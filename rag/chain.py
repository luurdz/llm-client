import logging

from langchain_core.documents import Document
from langchain_core.exceptions import (
    OutputParserException,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import ValidationError

from llm_client.config import load_config
from llm_client.langchain_models import build_chat_model
from llm_client.schemas import LLMConfig
from rag.prompts import RAG_PROMPT
from rag.schemas import RagResponse
from rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)


def format_docs(docs: list[Document]) -> str:
    logger.info(
        f"Retrieved {len(docs)} chunks from: {[doc.metadata['source'] for doc in docs]}"
    )
    texts = []
    for doc in docs:
        texts.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content}")
    return "\n\n".join(texts)


def build_rag_chain(config: LLMConfig) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=RagResponse)
    prompt = RAG_PROMPT.partial(format_instructions=parser.get_format_instructions())
    model = build_chat_model(config.model_copy(update={"temperature": 0}))
    retriever = get_retriever(config)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    ).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=3,
    )
    return chain


async def get_rag_response(query: str) -> RagResponse:
    logger.info(f"Question: {query}")
    chain = build_rag_chain(load_config())
    result = await chain.ainvoke(query)
    return result
