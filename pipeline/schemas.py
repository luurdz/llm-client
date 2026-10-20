from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CriticalityLevel(str, Enum):
    LOW = "baja"
    MEDIUM = "media"
    HIGH = "alta"


class TechnicalEntities(BaseModel):
    """Technical information extracted from a software architecture description or an error log."""

    technologies: list[str] = Field(
        serialization_alias="tecnologias",
        description=(
            "Technologies explicitly mentioned in the text: programming languages, "
            "frameworks, databases, message brokers, cloud services and tools. "
            "Use the official name of each technology, without duplicates. "
            "Only include specific products with a proper name, such as PostgreSQL or Redis; "
            "never generic categories such as 'database' or 'cache', and never placeholders such as 'unknown' or 'N/A'."
        ),
    )
    criticality_level: CriticalityLevel = Field(
        serialization_alias="nivel_de_criticidad",
        description=(
            "Criticality of the situation described in the text. "
            "'alta': service outage, data loss, security breach or errors affecting users in production. "
            "'media': the system works but is degraded, slow or has partial failures. "
            "'baja': warnings, informational messages or a description without problems."
        ),
    )
    technical_summary: str = Field(
        min_length=1,
        serialization_alias="resumen_tecnico",
        description=(
            "Technical summary in one or two sentences, written in Spanish. "
            "Describe the architecture or the main problem, and its likely cause if it is mentioned."
        ),
    )

    @field_validator("technologies")
    @classmethod
    def validate_technologies(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one specific technology is required")
        return value
