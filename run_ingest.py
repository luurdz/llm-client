import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client.config import load_config
from rag.ingest import index_documents, load_documents, split_documents
from rag.vectorstore import get_vectorstore

DATA_DIR = Path("data")

async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    load_dotenv()
    try:
        config = load_config()
        vectorstore = get_vectorstore(config)
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    count = await index_documents(vectorstore=vectorstore, chunks=chunks)
    print(f"Chunks indexed: {count}")


if __name__ == "__main__":
    asyncio.run(main())


