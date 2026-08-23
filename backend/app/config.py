"""Application configuration: a typed ``Settings`` object from env or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "openai"]


class Settings(BaseSettings):
    """Typed application settings; env vars map case-insensitively to fields."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AI provider: "ollama" (local, default) or "openai".
    ai_provider: Provider = "ollama"

    # OpenAI configuration (used when ai_provider="openai").
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Ollama exposes an OpenAI-compatible endpoint, so the same adapter drives it.
    # Use exact tags from ``ollama list``; untagged names resolve to ":latest".
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "qwen2.5:7b-instruct"

    # Retrieval / chunking tuning.
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    # Coarse cosine floor for relevance; tune per embedding model.
    min_relevance_score: float = 0.05
    # Yes/no LLM check that context answers the question before generating;
    # rejects out-of-scope questions at the cost of one LLM call.
    relevance_gate: bool = True

    # Storage.
    database_path: str = "knowledge_inbox.db"

    # Server / observability.
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_ollama(self) -> bool:
        return self.ai_provider == "ollama"

    @property
    def use_openai(self) -> bool:
        return self.ai_provider == "openai"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
