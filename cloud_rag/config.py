import os

from pydantic import BaseModel


class PineconeSettings(BaseModel):
    api_key: str
    index_name: str
    namespace: str


def load_pinecone_settings() -> PineconeSettings:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set in the .env file")
    index_name = os.getenv("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME is not set in the .env file")
    namespace = os.getenv("PINECONE_NAMESPACE")
    if not namespace:
        raise ValueError("PINECONE_NAMESPACE is not set in the .env file")
    return PineconeSettings(api_key=api_key, index_name=index_name, namespace=namespace)
