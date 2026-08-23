"""Retrieval use-case: embed a question and fetch the most relevant chunks.

Resolves chunk hits back to their parent items for context and citations.
"""

from __future__ import annotations

import logging

from app.domain.interfaces import EmbeddingProvider, ItemRepository
from app.domain.models import RetrievedChunk
from app.observability import span
from app.repositories.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class RetrievalService:
    """Semantic retrieval over the ingested corpus."""

    def __init__(
        self,
        repository: ItemRepository,
        vector_index: VectorIndex,
        embedder: EmbeddingProvider,
    ) -> None:
        self._repo = repository
        self._index = vector_index
        self._embedder = embedder

    async def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks most relevant to ``question``."""
        async with span("retrieve", top_k=top_k) as s:
            async with span("embed_query"):
                query_embedding = (await self._embedder.embed([question]))[0]
            hits = self._index.search(query_embedding, top_k)

            # cache items so each distinct parent is fetched only once.
            item_cache = {}
            results: list[RetrievedChunk] = []
            for chunk, score in hits:
                item = item_cache.get(chunk.item_id)
                if item is None:
                    item = self._repo.get(chunk.item_id)
                    if item is None:
                        continue  # orphaned chunk; skip defensively
                    item_cache[chunk.item_id] = item
                results.append(RetrievedChunk(chunk=chunk, item=item, score=score))

            s["index_size"] = self._index.size()
            s["hits"] = len(results)
            s["top"] = [
                {"item": rc.item.title[:50], "score": round(rc.score, 3)}
                for rc in results[:5]
            ]
        return results
