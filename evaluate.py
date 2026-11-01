import json
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone
from pydantic import ValidationError

from cloud_rag.config import load_pinecone_settings
from cloud_rag.ingest import build_chunk_metadata, load_documents, split_documents
from cloud_rag.rag_system import RAGSystem
from cloud_rag.retrievers import PineconeRetriever, build_bm25_retriever
from llm_client.config import load_config
from llm_client.langchain_models import build_embeddings

DATA_DIR = Path("data")
GOLDEN_SET_PATH = Path("benchmark/golden_set.json")
TOP_K = 5


def load_golden_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_at_k(docs: list[Document], expected: str) -> int:
    return int(any(doc.metadata["source"] == expected for doc in docs))


def precision_at_k(docs: list[Document], expected: str, k: int) -> float:
    return sum(1 for doc in docs if doc.metadata["source"] == expected) / k


def evaluate_retriever(retrieve: Callable[[str], list[Document]], golden_set: list[dict]) -> tuple[float, float]:
    recalls = []
    precisions = []
    for item in golden_set:
        docs = retrieve(item["pregunta"])
        recall = recall_at_k(docs, item["documento_id_esperado"])
        recalls.append(recall)
        precision = precision_at_k(docs, item["documento_id_esperado"], TOP_K)
        precisions.append(precision)
        print(item["pregunta"])
        print(f"   recall@5={recall}  precision@5={precision:.2f}")
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    return avg_recall, avg_precision


def main():
    load_dotenv()
    try:
        pinecone_settings = load_pinecone_settings()
        config = load_config()
        embeddings = build_embeddings(config)
    except (ValueError, ValidationError) as e:
        print(f"Configuration error: {e}")
        return

    metas = build_chunk_metadata(split_documents(load_documents(DATA_DIR)))
    pc = Pinecone(api_key=pinecone_settings.api_key)
    index = pc.index(pinecone_settings.index_name)
    rag = RAGSystem(index=index, embeddings=embeddings, namespace=pinecone_settings.namespace, metas=metas, top_k=TOP_K)
    pinecone_retriever = PineconeRetriever(index=index, embeddings=embeddings, namespace=pinecone_settings.namespace, top_k=TOP_K)
    bm25 = build_bm25_retriever(metas, TOP_K)
    golden_set = load_golden_set(GOLDEN_SET_PATH)

    retrievers = {
        "Pinecone": pinecone_retriever.invoke,
        "BM25": bm25.invoke,
        "Hybrid": rag.retrieve,
    }
    summary = {}
    for name, retrieve in retrievers.items():
        print(f"\n=== {name} ===")
        summary[name] = evaluate_retriever(retrieve, golden_set)

    print(f"\n{'Retriever':<10}  {'Recall@5':>8}  {'Precision@5':>11}")
    for name, (recall, precision) in summary.items():
        print(f"{name:<10}  {recall:>8.2f}  {precision:>11.2f}")


if __name__ == "__main__":
    main()
