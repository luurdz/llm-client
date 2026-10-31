import os

from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import ValidationError

from cloud_rag.pinecone_index import ensure_index
from llm_client.config import load_config
from llm_client.langchain_models import build_embeddings


def main():
    load_dotenv()
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set in the .env file")
        index_name = os.getenv("INDEX_NAME")
        if not index_name:
            raise ValueError("INDEX_NAME is not set in the .env file")
        config = load_config()
        embeddings = build_embeddings(config)
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return

    dimension = len(embeddings.embed_query("dimension check"))

    pc = Pinecone(api_key=api_key)
    ensure_index(pc, index_name, dimension)

    info = pc.describe_index(index_name)
    print(f"Index '{info.name}' ready. Dimension: {info.dimension}, metric: {info.metric}")


if __name__ == "__main__":
    main()