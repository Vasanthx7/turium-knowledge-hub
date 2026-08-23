"""Integration tests for the ingest -> retrieve -> answer flow.

These exercise the real services wired by the container, using the mock AI
providers and an in-memory DB.
"""

from __future__ import annotations

import pytest

from app.container import Container
from app.domain.errors import EmptyKnowledgeBaseError, ValidationError


@pytest.mark.asyncio
async def test_ingest_note_creates_item_and_chunks(container: Container):
    item, n_chunks = await container.ingest_service.ingest_note(
        "The capital of France is Paris. Paris is known for the Eiffel Tower.",
        title=None,
    )
    assert item.id
    assert n_chunks >= 1
    assert container.repository.count_items() == 1
    assert container.vector_index.size() == n_chunks


@pytest.mark.asyncio
async def test_empty_note_rejected(container: Container):
    with pytest.raises(ValidationError):
        await container.ingest_service.ingest_note("   ", title=None)


@pytest.mark.asyncio
async def test_query_before_ingest_raises(container: Container):
    with pytest.raises(EmptyKnowledgeBaseError):
        await container.rag_service.answer("anything?", top_k=3)


@pytest.mark.asyncio
async def test_query_returns_relevant_citation(container: Container):
    await container.ingest_service.ingest_note(
        "Photosynthesis lets plants convert sunlight into chemical energy.",
        title="Biology note",
    )
    await container.ingest_service.ingest_note(
        "The Roman Empire fell in the year 476 AD.", title="History note"
    )

    answer = await container.rag_service.answer(
        "How do plants use sunlight?", top_k=3
    )

    assert answer.citations, "expected at least one citation"
    # The biology note should outrank the history note for this question.
    assert answer.citations[0].title == "Biology note"
    assert answer.answer  # non-empty text


@pytest.mark.asyncio
async def test_index_rebuilds_from_storage(settings):
    """A fresh container over the same (in-memory shared) DB re-warms the index.

    Uses a file-backed temp DB to prove persistence across container instances.
    """
    import tempfile
    import os

    from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

    def make_container(db_path: str) -> Container:
        return Container(
            settings.model_copy(update={"database_path": db_path}),
            embedder=FakeEmbeddingProvider(),
            llm=FakeLLMProvider(),
        )

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        first = make_container(path)
        first.warm_up()
        await first.ingest_service.ingest_note("Durable knowledge.", title="Note")
        assert first.vector_index.size() >= 1

        # New container, same DB file: index must be rebuilt from storage.
        second = make_container(path)
        second.warm_up()
        assert second.repository.count_items() == 1
        assert second.vector_index.size() >= 1
    finally:
        os.remove(path)
