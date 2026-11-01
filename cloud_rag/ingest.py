import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pinecone import Index

from cloud_rag.schemas import ChunkMetadata
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, clean_text

BATCH_SIZE = 20


def load_documents(data_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(data_dir.glob("*.md")):
        text = clean_text(path.read_text(encoding="utf-8"))
        if not text.startswith("---"):
            raise ValueError(f"{path.name} has no front matter")
        _, front_matter, body = text.split("---", 2)
        category = front_matter.strip().split(":", 1)[1].strip()
        body = body.strip()
        documents.append(Document(page_content=body, metadata={"source": path.name, "category": category}))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")], strip_headers=False)
    sections = []
    for document in documents:
        sections_documents = header_splitter.split_text(document.page_content)
        for section in sections_documents:
            sections.append(Document(page_content=section.page_content, metadata={**document.metadata, **section.metadata}))
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(sections)


def build_chunk_metadata(chunks: list[Document]) -> list[ChunkMetadata]:
    results = []
    counters = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index = counters.get(source, 0)
        counters[source] = chunk_index + 1
        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        category = chunk.metadata["category"]
        section = chunk.metadata["section"]
        text = chunk.page_content
        results.append(ChunkMetadata(source=source, category=category, section=section, chunk_index=chunk_index, content_hash=content_hash, text=text))
    return results


def chunk_id(meta: ChunkMetadata) -> str:
    return f"{meta.source}-{meta.chunk_index}"


def fetch_existing_hashes(index: Index, namespace: str) -> dict[str, str]:
    hashes = {}
    for page in index.list(namespace=namespace):
        ids = [item.id for item in page]
        response = index.fetch(ids=ids, namespace=namespace)
        for vector_id, vector in response.vectors.items():
            hashes[vector_id] = vector.metadata["content_hash"]
    return hashes


def upsert_chunks(index: Index, embeddings: Embeddings, metas: list[ChunkMetadata], namespace: str) -> int:
    for start in range(0, len(metas), BATCH_SIZE):
        batch = metas[start:start + BATCH_SIZE]
        texts = [meta.text for meta in batch]
        vectors = embeddings.embed_documents(texts)
        records = []
        for meta, vector in zip(batch, vectors):
            records.append({"id": chunk_id(meta=meta), "values": vector, "metadata": meta.model_dump()})
        index.upsert(vectors=records, namespace=namespace)
    return len(metas)


def plan_changes(metas: list[ChunkMetadata], existing: dict[str, str]) -> tuple[list[ChunkMetadata], list[str]]:
    to_upsert = []
    for meta in metas:
        cid = chunk_id(meta=meta)
        if cid not in existing or existing[cid] != meta.content_hash:
            to_upsert.append(meta)

    local_ids = {chunk_id(meta) for meta in metas}
    to_delete = [cid for cid in existing if cid not in local_ids]

    return to_upsert, to_delete
