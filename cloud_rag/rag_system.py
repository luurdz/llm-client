from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pinecone import Index

from cloud_rag.retrievers import PineconeRetriever, build_bm25_retriever
from cloud_rag.schemas import ChunkMetadata


class RAGSystem:
    def __init__(self, index: Index, embeddings: Embeddings, namespace: str, metas: list[ChunkMetadata], top_k: int = 5, weights: tuple[float, float] = (0.3, 0.7)):
        self.top_k = top_k
        bm25_retriever = build_bm25_retriever(metas=metas, top_k=top_k)
        pinecone_retriever = PineconeRetriever(index=index, embeddings=embeddings, namespace=namespace, top_k=top_k)
        self.ensemble = EnsembleRetriever(retrievers=[bm25_retriever, pinecone_retriever], weights=list(weights))

    def retrieve(self, question: str) -> list[Document]:
        result = self.ensemble.invoke(question)
        return result[:self.top_k]
