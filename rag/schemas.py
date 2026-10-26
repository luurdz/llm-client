from pydantic import BaseModel, Field


class RagResponse(BaseModel):
    """Answer of the RAG system with the source documents it was based on."""

    answer: str = Field(
        description=(
            "Answer to the question, written in Spanish, using only the information in the context. "
            "If the context does not contain the answer, this field must be exactly 'No lo sé'."
        ),
    )
    sources: list[str] = Field(
        description=(
            "File names of the context fragments actually used to write the answer, "
            "for example 'seguro_de_viaje.md', without duplicates. "
            "Empty list if the answer is 'No lo sé'."
        ),
    )
