from pathlib import Path

import pytest
from dotenv import load_dotenv

from cloud_rag.ingest import build_chunk_metadata, load_documents, split_documents
from cloud_rag.schemas import ChunkMetadata

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def metas() -> list[ChunkMetadata]:
    """Chunks built from data/ with the same pipeline used by ingest_pinecone.py."""
    return build_chunk_metadata(split_documents(load_documents(DATA_DIR)))


@pytest.fixture(scope="session")
def pinecone_context():
    """Pinecone index, embeddings and namespace from .env. Skips the test if the keys are missing."""
    load_dotenv()
    from pinecone import Pinecone

    from cloud_rag.config import load_pinecone_settings
    from llm_client.config import load_config
    from llm_client.langchain_models import build_embeddings

    try:
        settings = load_pinecone_settings()
        embeddings = build_embeddings(load_config())
    except Exception as e:
        pytest.skip(f"Pinecone or embedding configuration not available: {e}")
    index = Pinecone(api_key=settings.api_key).index(settings.index_name)
    return index, embeddings, settings.namespace
