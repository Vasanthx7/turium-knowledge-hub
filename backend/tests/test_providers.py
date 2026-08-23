"""Provider selection tests.

Documents the shipped configuration: the app defaults to a local Ollama model,
and OpenAI is a supported alternative that requires an API key. Building a
provider does not perform any network I/O, so these run offline.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.domain.errors import ProviderError
from app.providers.factory import build_embedding_provider, build_llm_provider


def test_default_provider_is_local_ollama():
    settings = Settings(_env_file=None,)
    assert settings.ai_provider == "ollama"
    assert settings.use_ollama and not settings.use_openai


def test_ollama_builds_local_adapters():
    settings = Settings(_env_file=None,ai_provider="ollama")
    embed = build_embedding_provider(settings)
    llm = build_llm_provider(settings)
    # Labelled "ollama" and pointed at the local server, no key required.
    assert embed.name == f"ollama:{settings.ollama_embedding_model}"
    assert llm.name == f"ollama:{settings.ollama_chat_model}"


def test_openai_requires_api_key():
    settings = Settings(_env_file=None,ai_provider="openai", openai_api_key="")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_embedding_provider(settings)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_llm_provider(settings)


def test_openai_builds_when_key_present():
    settings = Settings(_env_file=None,ai_provider="openai", openai_api_key="sk-test")
    embed = build_embedding_provider(settings)
    llm = build_llm_provider(settings)
    assert embed.name == f"openai:{settings.openai_embedding_model}"
    assert llm.name == f"openai:{settings.openai_chat_model}"


def test_unknown_provider_rejected():
    # ai_provider is a constrained Literal, so a bad value fails validation.
    with pytest.raises(Exception):
        Settings(ai_provider="banana")
