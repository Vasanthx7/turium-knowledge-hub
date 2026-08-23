"""Shared test fixtures.

Tests run against deterministic fake providers and an in-memory SQLite DB, so
they need no model server or API key and leave no files behind.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.container import Container
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps tests hermetic — they use code defaults, not the
    # developer's local .env.
    return Settings(
        _env_file=None,
        database_path=":memory:",
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    c = Container(
        settings,
        embedder=FakeEmbeddingProvider(),
        llm=FakeLLMProvider(),
    )
    c.warm_up()
    return c
