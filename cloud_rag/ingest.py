import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from cloud_rag.schemas import ChunkMetadata
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, clean_text


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