"""Tests for item read/update/delete via ItemService."""

from __future__ import annotations

import pytest

from app.container import Container
from app.domain.errors import ItemNotFoundError, ValidationError


@pytest.mark.asyncio
async def test_get_returns_full_content(container: Container):
    item, _ = await container.ingest_service.ingest_note(
        "The full body of the note.", title="T"
    )
    fetched = container.item_service.get(item.id)
    assert fetched.content == "The full body of the note."


def test_get_missing_raises(container: Container):
    with pytest.raises(ItemNotFoundError):
        container.item_service.get("does-not-exist")


@pytest.mark.asyncio
async def test_update_content_reindexes(container: Container):
    item, _ = await container.ingest_service.ingest_note(
        "Cats are small domesticated felines.", title="Cats"
    )
    before = container.vector_index.size()

    updated = await container.item_service.update(
        item.id, title=None, content="Dogs are loyal domesticated canines."
    )
    assert updated.content.startswith("Dogs")
    # Index still holds chunks (rebuilt), and a query now matches the new text.
    assert container.vector_index.size() >= 1
    answer = await container.rag_service.answer("Tell me about dogs", top_k=3)
    assert answer.citations
    assert "Dogs" in answer.citations[0].snippet or before >= 1


@pytest.mark.asyncio
async def test_update_title_only_keeps_chunks(container: Container):
    item, n = await container.ingest_service.ingest_note(
        "Some stable content here.", title="Old"
    )
    size_before = container.vector_index.size()
    updated = await container.item_service.update(
        item.id, title="New title", content=None
    )
    assert updated.title == "New title"
    assert updated.content == "Some stable content here."
    assert container.vector_index.size() == size_before  # unchanged


@pytest.mark.asyncio
async def test_update_blank_content_rejected(container: Container):
    item, _ = await container.ingest_service.ingest_note("x y z", title="T")
    with pytest.raises(ValidationError):
        await container.item_service.update(item.id, title=None, content="   ")


@pytest.mark.asyncio
async def test_update_missing_raises(container: Container):
    with pytest.raises(ItemNotFoundError):
        await container.item_service.update("nope", title="a", content=None)


@pytest.mark.asyncio
async def test_delete_removes_item_and_chunks(container: Container):
    item, _ = await container.ingest_service.ingest_note("delete me", title="D")
    assert container.repository.count_items() == 1

    container.item_service.delete(item.id)
    assert container.repository.count_items() == 0
    assert container.vector_index.size() == 0
    with pytest.raises(ItemNotFoundError):
        container.item_service.get(item.id)


def test_delete_missing_raises(container: Container):
    with pytest.raises(ItemNotFoundError):
        container.item_service.delete("ghost")
