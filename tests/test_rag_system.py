import pytest

from cloud_rag.rag_system import RAGSystem
from cloud_rag.retrievers import search

pytestmark = pytest.mark.integration

# Measured on the 47 chunks: on-topic questions score >= 0.70, off-topic ones <= 0.61.
RELEVANCE_THRESHOLD = 0.65


@pytest.fixture(scope="module")
def rag(pinecone_context, metas) -> RAGSystem:
    index, embeddings, namespace = pinecone_context
    return RAGSystem(index=index, embeddings=embeddings, namespace=namespace, metas=metas)


def test_retrieve_returns_top_5_without_duplicates(rag):
    docs = rag.retrieve("¿Qué hago con el error RS-1042?")
    texts = [doc.page_content for doc in docs]
    assert len(docs) == 5
    assert len(texts) == len(set(texts))


def test_retrieve_respects_top_k(pinecone_context, metas):
    index, embeddings, namespace = pinecone_context
    rag = RAGSystem(index=index, embeddings=embeddings, namespace=namespace, metas=metas, top_k=3)
    assert len(rag.retrieve("¿Qué hago con el error RS-1042?")) == 3


def test_exact_code_is_ranked_first(rag):
    docs = rag.retrieve("¿Qué hago con el error RS-1042?")
    assert "RS-1042" in docs[0].page_content


def test_question_without_shared_words_is_answered_by_meaning(rag):
    # The documents say "despachar" and "kilos", never "peso": BM25 alone does not solve it.
    docs = rag.retrieve("¿Cuánto peso puedo despachar?")
    assert docs[0].metadata["source"] == "equipaje_y_checkin.md"


def test_bm25_padding_does_not_reach_the_top_5(rag):
    docs = rag.retrieve("¿Cuánto peso puedo despachar?")
    assert all(doc.metadata["source"] == "equipaje_y_checkin.md" for doc in docs)


# Relevance checks

@pytest.mark.parametrize("question", [
    "¿Cuántos kilos puedo llevar en la valija?",
    "¿Qué cubre el seguro?",
])
def test_on_topic_question_scores_above_threshold(pinecone_context, question):
    index, embeddings, namespace = pinecone_context
    results = search(index, embeddings, question, 1, namespace)
    assert results[0][1] >= RELEVANCE_THRESHOLD


@pytest.mark.parametrize("question", [
    "pizza margarita",
    "¿Quién ganó el mundial de fútbol 2022?",
    "receta de empanadas salteñas",
])
def test_off_topic_question_scores_below_threshold(pinecone_context, question):
    # Pinecone always returns top_k results; only the score tells that none is relevant.
    index, embeddings, namespace = pinecone_context
    results = search(index, embeddings, question, 1, namespace)
    assert results[0][1] < RELEVANCE_THRESHOLD


def test_ambiguous_question_returns_chunks_from_several_documents(rag):
    # "Cancelar" applies to trips, insurance and loyalty miles: the context is mixed.
    docs = rag.retrieve("¿Cómo cancelo?")
    assert len({doc.metadata["source"] for doc in docs}) > 1
