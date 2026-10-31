from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["cancelaciones", "equipaje", "seguros", "paquetes", "sistemas", "fidelidad", "pagos", "emergencias"]


class ChunkMetadata(BaseModel):
    """Metadata stored in Pinecone with each chunk vector."""

    source: str = Field(
        description="File name of the document the chunk comes from, for example 'seguro_de_viaje.md'.",
    )

    category: Category = Field(
        description="Category of the document, taken from its front matter.",
    )

    section: str = Field(
        description="Title of the '##' section of the document the chunk belongs to.",
    )

    chunk_index: int = Field(
        ge=0,
        description="Position of the chunk inside its document, starting at 0.",
    )

    content_hash: str = Field(
        description="Hash of the chunk text, used to detect changes in incremental ingestion.",
    )

    text: str = Field(
        description="Original text of the chunk, returned with the search results.",
    )
