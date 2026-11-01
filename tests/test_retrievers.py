import pytest

from cloud_rag.retrievers import PineconeRetriever, build_bm25_retriever, search, tokenize


# tokenize

def test_tokenize_lowercases_and_removes_accents():
    assert tokenize("Cuántos KILOS") == ["cuantos", "kilos"]


def test_tokenize_removes_punctuation():
    assert tokenize("¿Qué pasa?") == ["que", "pasa"]


def test_tokenize_keeps_error_codes():
    assert tokenize("error RS-1042:") == ["error", "rs-1042"]


# BM25

def test_bm25_finds_exact_error_code(metas):
    bm25 = build_bm25_retriever(metas, top_k=5)
    docs = bm25.invoke("¿Qué hago con el error RS-1042?")
    assert docs[0].metadata["source"] == "manual_sistema_de_reservas.md"
    assert "RS-1042" in docs[0].page_content


def test_bm25_returns_only_chunks_with_matching_words(metas):
    bm25 = build_bm25_retriever(metas, top_k=5)
    docs = bm25.invoke("¿Cuánto peso puedo despachar?")
    for doc in docs:
        assert "despachar" in tokenize(doc.page_content)


def test_bm25_returns_nothing_for_unrelated_words(metas):
    bm25 = build_bm25_retriever(metas, top_k=5)
    assert bm25.invoke("pizza margarita") == []


# Pinecone

@pytest.mark.integration
def test_search_returns_results_ordered_by_score(pinecone_context):
    index, embeddings, namespace = pinecone_context
    results = search(index, embeddings, "¿Cuántos kilos puedo llevar en la valija?", 3, namespace)
    scores = [score for _, score in results]
    assert len(results) == 3
    assert scores == sorted(scores, reverse=True)
    assert results[0][0].source == "equipaje_y_checkin.md"


@pytest.mark.integration
def test_search_metadata_filter_is_applied(pinecone_context):
    index, embeddings, namespace = pinecone_context
    results = search(index, embeddings, "¿Cuántos kilos puedo llevar en la valija?", 3, namespace, {"category": {"$eq": "pagos"}})
    assert len(results) > 0
    for meta, _ in results:
        assert meta.category == "pagos"


@pytest.mark.integration
def test_pinecone_retriever_returns_documents_with_score(pinecone_context):
    index, embeddings, namespace = pinecone_context
    retriever = PineconeRetriever(index=index, embeddings=embeddings, namespace=namespace, top_k=3)
    docs = retriever.invoke("¿Cuántos kilos puedo llevar en la valija?")
    assert len(docs) == 3
    for doc in docs:
        assert doc.page_content
        assert "score" in doc.metadata
        assert "text" not in doc.metadata
