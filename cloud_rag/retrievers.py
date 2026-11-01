import re
import unicodedata

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pinecone import Index
from pydantic import Field

from cloud_rag.schemas import ChunkMetadata


def search(index: Index, embeddings: Embeddings, question: str, top_k: int, namespace: str, metadata_filter: dict | None = None) -> list[tuple[ChunkMetadata, float]]:
    vector = embeddings.embed_query(text=question)
    response = index.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=True, filter=metadata_filter)
    results = []
    for match in response.matches:
        results.append((ChunkMetadata(**match.metadata), match.score))
    return results


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"[\w-]+", text)


def to_document(meta: ChunkMetadata) -> Document:
    return Document(page_content=meta.text, metadata=meta.model_dump(exclude={"text"}))


class PositiveScoreBM25Retriever(BM25Retriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        scored_docs = [(doc, score) for doc, score in zip(self.docs, scores) if score > 0]
        ranked = sorted(scored_docs, key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:self.k]]


def build_bm25_retriever(metas: list[ChunkMetadata], top_k: int) -> BM25Retriever:
    documents = []
    for meta in metas:
        documents.append(to_document(meta))

    return PositiveScoreBM25Retriever.from_documents(documents, preprocess_func=tokenize, k=top_k)


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
            doc = to_document(meta)
            doc.metadata["score"] = score
            documents.append(doc)
        return documents
