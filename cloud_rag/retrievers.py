from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pinecone import Index
from pydantic import Field

from cloud_rag.schemas import ChunkMetadata


def search(index: Index, embeddings: Embeddings, question: str, top_k: int, namespace: str, metadata_filter: dict | None = None) -> list[tuple[ChunkMetadata, float]]:
    vector = embeddings.embed_query(text=question)
    response = index.query(vector=vector, top_k= top_k, namespace=namespace, include_metadata=True, filter=metadata_filter)
    results = []
    for match in response.matches:
        results.append((ChunkMetadata(**match.metadata), match.score))
    return results

class PineconeRetriever(BaseRetriever):
    index: Index
    embeddings: Embeddings
    namespace: str
    top_k: int = Field(default=5)
    metadata_filter: dict | None = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        results = search(index=self.index, embeddings=self.embeddings, question=query, top_k=self.top_k, metadata_filter=self.metadata_filter, namespace=self.namespace)
        documents = []
        for meta, score in results:
            documents.append(Document(page_content=meta.text, metadata=meta.model_dump(exclude={"text"}) | {"score": score}))
        return documents