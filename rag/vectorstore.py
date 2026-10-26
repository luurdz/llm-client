from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from llm_client.langchain_models import build_embeddings
from llm_client.schemas import LLMConfig

VECTORSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "rumbo_sur"
TOP_K = 4


def get_vectorstore(config: LLMConfig) -> Chroma:
    embeddings = build_embeddings(config)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


def get_retriever(config: LLMConfig) -> VectorStoreRetriever:
    vectorstore = get_vectorstore(config)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return retriever
