"""Item CRUD use-cases: read, edit and delete saved items.

Editing re-runs the chunk+embed pipeline only when content actually changes, so
title-only edits skip embedding calls.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from app.domain.errors import ItemNotFoundError, ValidationError
from app.domain.interfaces import ItemRepository
from app.domain.models import Item, SourceType
from app.repositories.vector_index import VectorIndex
from app.services.chunk_embedder import ChunkEmbedder
from app.services.text_utils import derive_note_title

logger = logging.getLogger(__name__)


class ItemService:
    """Read/update/delete operations over saved items."""

    def __init__(
        self,
        repository: ItemRepository,
        vector_index: VectorIndex,
        chunk_embedder: ChunkEmbedder,
    ) -> None:
        self._repo = repository
        self._index = vector_index
        self._chunk_embedder = chunk_embedder

    def get(self, item_id: str) -> Item:
        """Return an item (with full content) or raise if it doesn't exist."""
        item = self._repo.get(item_id)
        if item is None:
            raise ItemNotFoundError(f"No item with id '{item_id}'.")
        return item

    async def update(
        self, item_id: str, title: str | None, content: str | None
    ) -> Item:
        """Edit an item's title and/or content, re-indexing if content changed."""
        existing = self.get(item_id)

        new_content = existing.content
        if content is not None:
            new_content = content.strip()
            if not new_content:
                raise ValidationError("Content must not be empty.")

        new_title = self._resolve_title(existing, title, new_content)
        updated = replace(existing, title=new_title, content=new_content)

        content_changed = new_content != existing.content
        if content_changed:
            chunks = await self._chunk_embedder.build(item_id, new_content)
            self._repo.update(updated, chunks)
            self._index.remove_item(item_id)
            self._index.add(chunks)
        else:
            self._repo.update(updated, None)

        logger.info(
            "item updated",
            extra={"item_id": item_id, "content_changed": content_changed},
        )
        return updated

    def delete(self, item_id: str) -> None:
        """Delete an item, its chunks and its vector-index entries."""
        self.get(item_id)  # raises ItemNotFoundError if missing
        self._repo.delete(item_id)
        self._index.remove_item(item_id)
        logger.info("item deleted", extra={"item_id": item_id})

    @staticmethod
    def _resolve_title(existing: Item, title: str | None, content: str) -> str:
        """Decide the effective title after an edit.

        Explicit non-blank title wins; a blanked note title is re-derived from
        content; URLs keep their existing title.
        """
        if title is not None:
            stripped = title.strip()
            if stripped:
                return stripped
            if existing.source_type == SourceType.NOTE:
                return derive_note_title(content)
        return existing.title
