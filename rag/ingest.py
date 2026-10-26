import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DOCUMENT_FOOTER = "Rumbo Sur Viajes · Documento interno · Versión 2026.2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    text = text.replace(DOCUMENT_FOOTER, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def load_documents(data_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(data_dir.glob("*.md")):
        text = clean_text(path.read_text(encoding="utf-8"))
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


async def index_documents(vectorstore: Chroma, chunks: list[Document]) -> int:
    existing = vectorstore.get(limit=1)
    if existing["ids"]:
        logger.info("Vector store already has data, skipping indexing")
        return 0
    ids = [f"{chunk.metadata['source']}-{i}" for i, chunk in enumerate(chunks)]
    await vectorstore.aadd_documents(chunks, ids=ids)
    logger.info(f"Indexed {len(chunks)} chunks")
    return len(chunks)
