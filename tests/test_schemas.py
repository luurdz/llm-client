import pytest
from pydantic import ValidationError

from cloud_rag.schemas import ChunkMetadata

VALID_FIELDS = {
    "source": "pagos_y_facturacion.md",
    "category": "pagos",
    "section": "3. Pagos rechazados",
    "chunk_index": 0,
    "content_hash": "abc123",
    "text": "La pasarela rechazó la tarjeta.",
}


def test_valid_metadata_is_accepted():
    meta = ChunkMetadata(**VALID_FIELDS)
    assert meta.category == "pagos"


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        ChunkMetadata(**{**VALID_FIELDS, "category": "marketing"})


def test_negative_chunk_index_is_rejected():
    with pytest.raises(ValidationError):
        ChunkMetadata(**{**VALID_FIELDS, "chunk_index": -1})


def test_missing_field_is_rejected():
    fields = {key: value for key, value in VALID_FIELDS.items() if key != "text"}
    with pytest.raises(ValidationError):
        ChunkMetadata(**fields)


def test_float_chunk_index_from_pinecone_is_converted_to_int():
    # Pinecone returns numeric metadata as float (2.0).
    meta = ChunkMetadata(**{**VALID_FIELDS, "chunk_index": 2.0})
    assert meta.chunk_index == 2
