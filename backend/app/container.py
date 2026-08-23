"""Composition root: the single place where the object graph is assembled."""

from __future__ import annotations

import logging

from app.config import Settings
from app.db.database import Database
from app.domain.interfaces import EmbeddingProvider, LLMProvider
from app.providers.factory import build_embedding_provider, build_llm_provider
from app.repositories.sqlite_item_repository import SqliteItemRepository
from app.repositories.vector_index import VectorIndex
from app.services.chunk_embedder import ChunkEmbedder
from app.services.chunking import OverlappingCharacterChunker
from app.services.content_fetcher import HttpContentFetcher
from app.services.ingest_service import IngestService
from app.services.item_service import ItemService
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class Container:
    """Holds application-wide singletons and their wiring."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.settings = settings

        # Persistence.
        self.database = Database(settings.database_path)
        self.database.initialize()
        self.repository = SqliteItemRepository(self.database)
        self.vector_index = VectorIndex()

        # AI adapters: injected (tests) or built from settings by the factory.
        self.embedder = embedder or build_embedding_provider(settings)
        self.llm = llm or build_llm_provider(settings)

        # Strategies.
        self.chunker = OverlappingCharacterChunker(
            chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        self.fetcher = HttpContentFetcher()
        self.chunk_embedder = ChunkEmbedder(self.chunker, self.embedder)

        # Services (use-cases).
        self.ingest_service = IngestService(
            repository=self.repository,
            vector_index=self.vector_index,
            chunk_embedder=self.chunk_embedder,
            fetcher=self.fetcher,
        )
        self.item_service = ItemService(
            repository=self.repository,
            vector_index=self.vector_index,
            chunk_embedder=self.chunk_embedder,
        )
        self.retrieval_service = RetrievalService(
            repository=self.repository,
            vector_index=self.vector_index,
            embedder=self.embedder,
        )
        self.rag_service = RagService(
            retrieval=self.retrieval_service,
            llm=self.llm,
            repository=self.repository,
            min_score=settings.min_relevance_score,
            relevance_gate=settings.relevance_gate,
        )

    def warm_up(self) -> None:
        """Rebuild the in-memory vector index from durable storage on boot."""
        chunks = self.repository.all_chunks()
        self.vector_index.load(chunks)
        logger.info(
            "vector index warmed",
            extra={"chunks": self.vector_index.size(),
                   "items": self.repository.count_items()},
        )
