"""OpenAI adapters for embeddings and chat completion.

Translate SDK errors into domain ProviderError so the rest of the app never
handles OpenAI types directly.
"""

from __future__ import annotations

import logging

from app.domain.errors import ProviderError
from app.domain.interfaces import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via the OpenAI (or any OpenAI-compatible) endpoint.

    ``base_url`` points the SDK at a compatible server (e.g. local Ollama);
    ``label`` names the provider in logs/health.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        label: str = "openai",
    ) -> None:
        from openai import AsyncOpenAI  # lazy import: optional dependency

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._label = label

    @property
    def name(self) -> str:
        return f"{self._label}:{self._model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = await self._client.embeddings.create(
                model=self._model, input=texts
            )
        except Exception as exc:  # noqa: BLE001 — normalise to a domain error
            logger.error("openai embedding failed", extra={"error": str(exc)})
            raise ProviderError(f"Embedding request failed: {exc}") from exc
        # API preserves input order.
        return [d.embedding for d in resp.data]


class OpenAILLMProvider(LLMProvider):
    """Chat completion via the OpenAI (or any OpenAI-compatible) endpoint.

    ``base_url`` / ``label`` behave as for :class:`OpenAIEmbeddingProvider`.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        label: str = "openai",
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._label = label

    @property
    def name(self) -> str:
        return f"{self._label}:{self._model}"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("openai completion failed", extra={"error": str(exc)})
            raise ProviderError(f"Completion request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()
