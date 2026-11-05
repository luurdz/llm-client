from pathlib import Path

import pytest
from dotenv import load_dotenv

from cloud_rag.ingest import load_chunk_metas
from cloud_rag.schemas import ChunkMetadata

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def metas() -> list[ChunkMetadata]:
    """Chunks built from data/ with the same pipeline used by ingest_pinecone.py."""
    return load_chunk_metas(DATA_DIR)


@pytest.fixture(scope="session")
def pinecone_context():
    """Pinecone index, embeddings and namespace from .env. Skips the test if the keys are missing."""
    load_dotenv()
    from cloud_rag.connection import connect_pinecone

    try:
        return connect_pinecone()
    except Exception as e:
        pytest.skip(f"Pinecone or embedding configuration not available: {e}")

