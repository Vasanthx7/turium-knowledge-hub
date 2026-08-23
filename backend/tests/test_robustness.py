"""Robustness / failure-path tests.

These assert the app fails *gracefully* — clear domain errors that map to
sensible HTTP codes — rather than leaking 500s.
"""

from __future__ import annotations

import pytest

from app.container import Container
from app.domain.errors import ContentFetchError
from app.domain.interfaces import ContentFetcher
from app.services.content_fetcher import HttpContentFetcher
from app.services.ingest_service import IngestService


class _FailingFetcher(ContentFetcher):
    """A fetcher that always fails, standing in for an unreachable/invalid URL."""

    async def fetch(self, url: str) -> tuple[str, str]:
        raise ContentFetchError("could not fetch (simulated)")


@pytest.mark.asyncio
async def test_url_fetch_error_propagates(container: Container):
    """A fetch failure surfaces as ContentFetchError (→ HTTP 400), not a 500."""
    service = IngestService(
        repository=container.repository,
        vector_index=container.vector_index,
        chunk_embedder=container.chunk_embedder,
        fetcher=_FailingFetcher(),
    )
    with pytest.raises(ContentFetchError):
        await service.ingest_url("https://unreachable.example")

    # A failed ingest must not leave partial state behind.
    assert container.repository.count_items() == 0
    assert container.vector_index.size() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",              # loopback
        "http://169.254.169.254/",        # link-local (cloud metadata)
        "http://[::1]/",                  # IPv6 loopback
        "http://10.0.0.1/",               # private
        "ftp://example.com/file",         # non-HTTP scheme
    ],
)
async def test_fetch_blocks_internal_and_non_http(url: str):
    """SSRF guard: internal-address and non-HTTP targets are refused."""
    with pytest.raises(ContentFetchError):
        await HttpContentFetcher().fetch(url)
