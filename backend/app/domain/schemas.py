"""Transport DTOs (Pydantic models) for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.models import SourceType


# --------------------------------------------------------------------------- #
# Ingestion                                                                    #
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``. Exactly one of ``text`` or ``url``."""

    text: str | None = Field(
        default=None, description="Plain-text note content.", max_length=100_000
    )
    url: str | None = Field(
        default=None, description="A URL whose page content will be fetched."
    )
    title: str | None = Field(
        default=None, description="Optional title for a note.", max_length=300
    )

    @field_validator("text", "url", "title")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "IngestRequest":
        has_text = bool(self.text)
        has_url = bool(self.url)
        if has_text == has_url:
            raise ValueError("Provide exactly one of 'text' or 'url'.")
        if has_url and not (
            self.url.startswith("http://") or self.url.startswith("https://")
        ):
            raise ValueError("'url' must start with http:// or https://.")
        return self


class ItemResponse(BaseModel):
    """A saved item as returned by ``/items`` and ``/ingest``."""

    id: str
    source_type: SourceType
    title: str
    preview: str
    source_url: str | None
    created_at: datetime


class ItemDetailResponse(ItemResponse):
    """A single item including its full content (for the detail view)."""

    content: str


class IngestResponse(BaseModel):
    """Result of a successful ingestion."""

    item: ItemResponse
    chunks_created: int
    # Populated only when ?debug=true.
    trace: dict | None = None


class UpdateItemRequest(BaseModel):
    """Request body for ``PATCH /items/{id}``; at least one field required."""

    title: str | None = Field(default=None, max_length=300)
    content: str | None = Field(default=None, max_length=100_000)

    @field_validator("title", "content")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _at_least_one(self) -> "UpdateItemRequest":
        if self.title is None and self.content is None:
            raise ValueError("Provide at least one of 'title' or 'content'.")
        if self.content is not None and not self.content:
            raise ValueError("'content' must not be empty.")
        return self


class ItemListResponse(BaseModel):
    """Envelope for the item list, with a count."""

    items: list[ItemResponse]
    count: int


# --------------------------------------------------------------------------- #
# Query / RAG                                                                  #
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    question: str = Field(max_length=2_000)
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override how many chunks to retrieve (defaults to server config).",
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        stripped = v.strip()
        # Enforce minimum length on the trimmed value.
        if len(stripped) < 3:
            raise ValueError("Question must be at least 3 characters.")
        return stripped


class CitationResponse(BaseModel):
    """A single cited source backing an answer.

    The internal relevance score is intentionally not exposed here; it stays in
    the domain model, logs and the ``?debug=true`` trace.
    """

    item_id: str
    title: str
    source_type: SourceType
    source_url: str | None
    snippet: str


class QueryResponse(BaseModel):
    """The answer to a question plus its supporting citations."""

    question: str
    answer: str
    citations: list[CitationResponse]
    # Populated only when ?debug=true: full step trace with per-step timings.
    trace: dict | None = None
