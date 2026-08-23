"""Core domain entities, free of persistence or transport concerns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """How a piece of content entered the inbox."""

    NOTE = "note"
    URL = "url"


@dataclass
class Item:
    """A single saved piece of knowledge (a note or a fetched URL)."""

    id: str
    source_type: SourceType
    title: str
    content: str
    created_at: datetime
    # Present only for URL items.
    source_url: str | None = None

    @property
    def preview(self) -> str:
        """A short snippet of the content for list views."""
        snippet = self.content.strip().replace("\n", " ")
        return snippet[:200] + ("…" if len(snippet) > 200 else "")


@dataclass
class Chunk:
    """A contiguous slice of an item's content, the unit of retrieval."""

    id: str
    item_id: str
    # 0-based position within the parent item.
    index: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, paired with its relevance score."""

    chunk: Chunk
    item: Item
    score: float


@dataclass
class Citation:
    """A source reference attached to an answer."""

    item_id: str
    title: str
    source_type: SourceType
    source_url: str | None
    snippet: str
    score: float


@dataclass
class Answer:
    """The result of a RAG query: generated text plus its grounding sources."""

    question: str
    answer: str
    citations: list[Citation]
