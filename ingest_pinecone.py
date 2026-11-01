from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import ValidationError

from cloud_rag.config import load_pinecone_settings
from cloud_rag.ingest import (
    build_chunk_metadata,
    fetch_existing_hashes,
    load_documents,
    plan_changes,
    split_documents,
    upsert_chunks,
)
from llm_client.config import load_config
from llm_client.langchain_models import build_embeddings

DATA_DIR = Path("data")


def main():
    load_dotenv()
    try:
        pinecone_settings = load_pinecone_settings()
        config = load_config()
        embeddings = build_embeddings(config)
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents=documents)
    metas = build_chunk_metadata(chunks=chunks)
    pc = Pinecone(api_key=pinecone_settings.api_key)
    index = pc.index(pinecone_settings.index_name)

    existing = fetch_existing_hashes(index=index, namespace=pinecone_settings.namespace)
    to_upsert, to_delete = plan_changes(metas=metas, existing=existing)

    upsert_chunks(index=index, embeddings=embeddings, metas=to_upsert, namespace=pinecone_settings.namespace)

    if to_delete:
        index.delete(ids=to_delete, namespace=pinecone_settings.namespace)

    unchanged = len(metas) - len(to_upsert)
    print(f"Upserted: {len(to_upsert)}, Deleted: {len(to_delete)}, Unchanged: {unchanged}")


if __name__ == "__main__":
    main()