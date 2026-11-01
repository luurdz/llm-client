from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import ValidationError

from cloud_rag.config import load_pinecone_settings
from cloud_rag.pinecone_index import ensure_index
from llm_client.config import load_config
from llm_client.langchain_models import build_embeddings


def main():
    load_dotenv()
    try:
        pinecone_settings = load_pinecone_settings()
        config = load_config()
        embeddings = build_embeddings(config)
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return

    dimension = len(embeddings.embed_query("dimension check"))

    pc = Pinecone(api_key=pinecone_settings.api_key)
    ensure_index(pc, pinecone_settings.index_name, dimension)

    info = pc.describe_index(pinecone_settings.index_name)
    print(f"Index '{info.name}' ready. Dimension: {info.dimension}, metric: {info.metric}")


if __name__ == "__main__":
    main()