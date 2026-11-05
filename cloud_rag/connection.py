from langchain_core.embeddings import Embeddings
from pinecone import Index, Pinecone

from cloud_rag.config import load_pinecone_settings
from llm_client.config import load_config
from llm_client.langchain_models import build_embeddings


def connect_pinecone() -> tuple[Index, Embeddings, str]:
    pinecone_settings = load_pinecone_settings()
    config = load_config()
    embeddings = build_embeddings(config)
    pc = Pinecone(api_key=pinecone_settings.api_key)
    index = pc.index(pinecone_settings.index_name)
    return index, embeddings, pinecone_settings.namespace