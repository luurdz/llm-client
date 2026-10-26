from pathlib import Path

from langchain_chroma import Chroma

from llm_client.langchain_models import build_embeddings
from llm_client.schemas import LLMConfig

VECTORSTORE_DIR = Path("vectorstore")
COLLECTION_NAME = "rumbo_sur"

def get_vectorstore(config: LLMConfig) -> Chroma:
    embeddings= build_embeddings(config)
    vectorstore= Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore
