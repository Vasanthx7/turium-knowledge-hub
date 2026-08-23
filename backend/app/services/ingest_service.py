"""Ingestion use-case: save content, chunk it, embed it, index it."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.domain.errors import ValidationError
from app.domain.interfaces import ContentFetcher, ItemRepository
from app.domain.models import Item, SourceType
from app.observability import span, sync_span
from app.repositories.vector_index import VectorIndex
from app.services.chunk_embedder import ChunkEmbedder
from app.services.text_utils import derive_note_title

logger = logging.getLogger(__name__)


class IngestService:
    """Turns raw user input (note or URL) into a searchable, stored item."""

    def __init__(
        self,
        repository: ItemRepository,
        vector_index: VectorIndex,
        chunk_embedder: ChunkEmbedder,
        fetcher: ContentFetcher,
    ) -> None:
        self._repo = repository
        self._index = vector_index
        self._chunk_embedder = chunk_embedder
        self._fetcher = fetcher

    async def ingest_note(self, text: str, title: str | None) -> tuple[Item, int]:
        """Ingest a plain-text note."""
        if not text or not text.strip():
            raise ValidationError("Note text must not be empty.")
        resolved_title = title or derive_note_title(text)
        return await self._ingest(
            source_type=SourceType.NOTE,
            title=resolved_title,
            content=text.strip(),
            source_url=None,
        )

    async def ingest_url(self, url: str) -> tuple[Item, int]:
        """Fetch a URL server-side and ingest its extracted text."""
        async with span("fetch_url") as s:
            title, content = await self._fetcher.fetch(url)
            s["chars"] = len(content)
        return await self._ingest(
            source_type=SourceType.URL,
            title=title,
            content=content,
            source_url=url,
        )

    async def _ingest(
        self,
        source_type: SourceType,
        title: str,
        content: str,
        source_url: str | None,
    ) -> tuple[Item, int]:
        item_id = uuid.uuid4().hex
        item = Item(
            id=item_id,
            source_type=source_type,
            title=title,
            content=content,
            created_at=datetime.now(timezone.utc),
            source_url=source_url,
        )

        async with span("chunk_embed", source_type=source_type.value) as s:
            chunks = await self._chunk_embedder.build(item_id, content)
            s["chunks"] = len(chunks)

        # Persist first (durable), then update the in-memory index.
        with sync_span("persist", chunks=len(chunks)):
            self._repo.add(item, chunks)
            self._index.add(chunks)

        logger.info(
            "item ingested",
            extra={
                "item_id": item_id,
                "source_type": source_type.value,
                "chunks": len(chunks),
            },
        )
        return item, len(chunks)
