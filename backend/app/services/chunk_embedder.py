"""Chunk + embed pipeline shared by ingestion and editing."""

from __future__ import annotations

import uuid

from app.domain.errors import ValidationError
from app.domain.interfaces import ChunkingStrategy, EmbeddingProvider
from app.domain.models import Chunk


class ChunkEmbedder:
    """Turns an item's raw content into embedded, index-ready chunks."""

    def __init__(
        self, chunker: ChunkingStrategy, embedder: EmbeddingProvider
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder

    async def build(self, item_id: str, content: str) -> list[Chunk]:
        """Split ``content`` and embed each chunk, tagged to ``item_id``."""
        texts = self._chunker.split(content)
        if not texts:
            raise ValidationError("Content produced no chunks after cleaning.")

        embeddings = await self._embedder.embed(texts)
        return [
            Chunk(
                id=uuid.uuid4().hex,
                item_id=item_id,
                index=i,
                text=text,
                embedding=embedding,
            )
            for i, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
