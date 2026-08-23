"""Port interfaces (abstract base classes) implemented by concrete adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import Chunk, Item


class EmbeddingProvider(ABC):
    """Turns text into vector embeddings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier, used in logs/health."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input."""


class LLMProvider(ABC):
    """Generates a natural-language answer from a prompt."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier, used in logs/health."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion for the given prompts."""


class ChunkingStrategy(ABC):
    """Splits raw item content into retrievable chunks."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split ``text`` into an ordered list of chunk strings."""


class ContentFetcher(ABC):
    """Fetches and extracts readable text from a URL."""

    @abstractmethod
    async def fetch(self, url: str) -> tuple[str, str]:
        """Return ``(title, extracted_text)`` for the given URL."""


class ItemRepository(ABC):
    """Persistence for :class:`Item` aggregates and their chunks."""

    @abstractmethod
    def add(self, item: Item, chunks: list[Chunk]) -> None:
        """Persist an item together with its chunks atomically."""

    @abstractmethod
    def update(self, item: Item, chunks: list[Chunk] | None) -> None:
        """Update an item's row; replace chunks when given, else leave them."""

    @abstractmethod
    def delete(self, item_id: str) -> None:
        """Delete an item and its chunks. No-op if it doesn't exist."""

    @abstractmethod
    def get(self, item_id: str) -> Item | None:
        """Return an item by id, or ``None`` if it does not exist."""

    @abstractmethod
    def list_items(self) -> list[Item]:
        """Return all items, newest first."""

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to build the vector index)."""

    @abstractmethod
    def count_items(self) -> int:
        """Return the number of stored items."""
