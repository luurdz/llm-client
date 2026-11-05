from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from cloud_rag.connection import connect_pinecone
from cloud_rag.ingest import (
    fetch_existing_hashes,
    load_chunk_metas,
    plan_changes,
    upsert_chunks,
)

DATA_DIR = Path("data")


def main():
    load_dotenv()
    try:
        index, embeddings, namespace = connect_pinecone()
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return

    metas = load_chunk_metas(DATA_DIR)

    existing = fetch_existing_hashes(index=index, namespace=namespace)
    to_upsert, to_delete = plan_changes(metas=metas, existing=existing)

    upsert_chunks(index=index, embeddings=embeddings, metas=to_upsert, namespace=namespace)

    if to_delete:
        index.delete(ids=to_delete, namespace=namespace)

    unchanged = len(metas) - len(to_upsert)
    print(f"Upserted: {len(to_upsert)}, Deleted: {len(to_delete)}, Unchanged: {unchanged}")


if __name__ == "__main__":
    main()
