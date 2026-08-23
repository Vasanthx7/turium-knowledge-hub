"""Maps configuration to a concrete embedding/LLM adapter.

Both providers speak the OpenAI protocol, so one adapter serves a local Ollama
server (default) and the hosted OpenAI API.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.domain.errors import ProviderError
from app.domain.interfaces import EmbeddingProvider, LLMProvider
from app.providers.openai_provider import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
)

logger = logging.getLogger(__name__)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Return the embedding provider selected by ``settings``."""
    if settings.use_openai:
        _require_key(settings)
        provider = OpenAIEmbeddingProvider(
            settings.openai_api_key, settings.openai_embedding_model
        )
    else:
        provider = OpenAIEmbeddingProvider(
            api_key="ollama",  # required by the SDK, ignored by Ollama
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            label="ollama",
        )
    logger.info("embedding provider selected", extra={"provider": provider.name})
    return provider


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Return the LLM provider selected by ``settings``."""
    if settings.use_openai:
        _require_key(settings)
        provider = OpenAILLMProvider(
            settings.openai_api_key, settings.openai_chat_model
        )
    else:
        provider = OpenAILLMProvider(
            api_key="ollama",
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            label="ollama",
        )
    logger.info("llm provider selected", extra={"provider": provider.name})
    return provider


def _require_key(settings: Settings) -> None:
    """Fail fast with a clear message if OpenAI is selected but has no key."""
    if not settings.openai_api_key:
        raise ProviderError(
            "AI_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Set the key, or use AI_PROVIDER=ollama for a local model."
        )
