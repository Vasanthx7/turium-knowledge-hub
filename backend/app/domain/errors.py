"""Transport-agnostic domain exceptions; mapped to HTTP in ``app/api/errors.py``."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all expected, business-level errors."""


class ValidationError(DomainError):
    """Input failed a business rule (maps to HTTP 422/400)."""


class ItemNotFoundError(DomainError):
    """A requested item does not exist (maps to HTTP 404)."""


class ContentFetchError(DomainError):
    """A URL could not be fetched or yielded no usable text (maps to 400/502)."""


class ProviderError(DomainError):
    """An embedding/LLM provider failed (maps to HTTP 502)."""


class EmptyKnowledgeBaseError(DomainError):
    """A query was issued with nothing ingested yet (maps to HTTP 409)."""
