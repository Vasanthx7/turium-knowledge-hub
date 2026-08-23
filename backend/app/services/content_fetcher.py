"""Fetch a URL server-side and extract its readable text.

SSRF guard: redirects are followed manually and every hop's host is rejected if
it resolves to a non-global address. The body is streamed and aborted past
``max_bytes`` to bound memory.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.domain.errors import ContentFetchError
from app.domain.interfaces import ContentFetcher

logger = logging.getLogger(__name__)

# Tags whose text is boilerplate, not content.
_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
               "aside", "form", "iframe", "svg"]

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KnowledgeInboxBot/1.0; "
        "+https://example.com/bot)"
    )
}

_MAX_REDIRECTS = 5


class HttpContentFetcher(ContentFetcher):
    """Fetch and extract page text via HTTP + BeautifulSoup."""

    def __init__(self, timeout_seconds: float = 15.0, max_bytes: int = 5_000_000):
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes

    async def fetch(self, url: str) -> tuple[str, str]:
        html = await self._download(url)
        title, text = self._extract(html)
        if not text:
            raise ContentFetchError(
                "Fetched the page but found no readable text (it may be "
                "JavaScript-rendered or empty)."
            )
        logger.info(
            "url fetched",
            extra={"url": url, "title": title, "chars": len(text)},
        )
        return title or url, text

    async def _download(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,  # handled manually so each hop is validated
            headers=_DEFAULT_HEADERS,
        ) as client:
            current = url
            for _hop in range(_MAX_REDIRECTS + 1):
                await self._guard_url(current)
                try:
                    async with client.stream("GET", current) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                raise ContentFetchError(
                                    "URL redirected without a target."
                                )
                            current = str(resp.url.join(location))
                            continue
                        resp.raise_for_status()
                        self._check_content_type(resp)
                        return await self._read_limited(resp)
                except httpx.HTTPStatusError as exc:
                    raise ContentFetchError(
                        f"URL returned HTTP {exc.response.status_code}."
                    ) from exc
                except httpx.HTTPError as exc:
                    raise ContentFetchError(f"Could not fetch URL: {exc}") from exc
        raise ContentFetchError("Too many redirects.")

    async def _guard_url(self, url: str) -> None:
        """Reject non-HTTP schemes and hosts that resolve to internal addresses."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ContentFetchError("Only http:// and https:// URLs are supported.")
        host = parsed.hostname
        if not host:
            raise ContentFetchError("URL has no host.")

        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo, host, None, 0, socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise ContentFetchError("Could not resolve URL host.") from exc

        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise ContentFetchError(
                    "URL host is not allowed (it points to a private or "
                    "internal address)."
                )

    def _check_content_type(self, resp: httpx.Response) -> None:
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            raise ContentFetchError(
                f"Unsupported content type '{content_type or 'unknown'}'. "
                "Only HTML/text pages are supported."
            )

    async def _read_limited(self, resp: httpx.Response) -> str:
        """Stream the body, aborting as soon as it exceeds ``max_bytes``."""
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max_bytes:
            raise ContentFetchError("Page is too large to ingest.")

        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > self._max_bytes:
                raise ContentFetchError("Page is too large to ingest.")
            chunks.append(chunk)
        return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")

    @staticmethod
    def _extract(html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        main = soup.find("main") or soup.find("article") or soup.body or soup
        raw = main.get_text(separator="\n")

        lines = [ln.strip() for ln in raw.splitlines()]
        text = "\n".join(ln for ln in lines if ln)
        return title, text
