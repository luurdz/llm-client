from cloud_rag.ingest import chunk_id, plan_changes
from cloud_rag.schemas import ChunkMetadata


def make_meta(source: str, chunk_index: int, content_hash: str) -> ChunkMetadata:
    return ChunkMetadata(
        source=source,
        category="pagos",
        section="1. Medios de pago",
        chunk_index=chunk_index,
        content_hash=content_hash,
        text="texto",
    )


# Loader and splitter, on the real data/ folder

def test_every_chunk_has_category_and_section(metas):
    assert len(metas) > 0
    for meta in metas:
        assert meta.category
        assert meta.section


def test_front_matter_is_not_in_chunk_text(metas):
    for meta in metas:
        assert not meta.text.startswith("---")
        assert "category:" not in meta.text


def test_chunk_ids_are_unique(metas):
    ids = [chunk_id(meta) for meta in metas]
    assert len(ids) == len(set(ids))


def test_chunk_index_restarts_for_each_file(metas):
    sources = {meta.source for meta in metas}
    for source in sources:
        indexes = [meta.chunk_index for meta in metas if meta.source == source]
        assert indexes == list(range(len(indexes)))


# Incremental ingestion

def test_plan_changes_nothing_changed():
    meta = make_meta("a.md", 0, "h1")
    assert plan_changes([meta], {"a.md-0": "h1"}) == ([], [])


def test_plan_changes_modified_chunk_is_upserted():
    meta = make_meta("a.md", 0, "h1")
    assert plan_changes([meta], {"a.md-0": "old"}) == ([meta], [])


def test_plan_changes_new_chunk_is_upserted():
    meta = make_meta("a.md", 0, "h1")
    assert plan_changes([meta], {}) == ([meta], [])


def test_plan_changes_removed_chunk_is_deleted():
    meta = make_meta("a.md", 0, "h1")
    assert plan_changes([meta], {"a.md-0": "h1", "b.md-0": "h9"}) == ([], ["b.md-0"])
