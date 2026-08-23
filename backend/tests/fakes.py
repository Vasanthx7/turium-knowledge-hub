"""Deterministic in-memory providers for tests.

These let the suite run offline with no model server or API key while still
exercising the real pipeline. They live in the test tree, not the application,
so the shipped code only knows about real providers.
"""

from __future__ import annotations

import hashlib
import re

from app.domain.interfaces import EmbeddingProvider, LLMProvider
from app.services.prompt import GATE_SYSTEM_PROMPT

_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbeddingProvider(EmbeddingProvider):
    """Hashing bag-of-words embeddings: stable and cheap, no network.

    Not semantically clever, but consistent and discriminative enough on
    vocabulary overlap to drive retrieval end-to-end in tests.
    """

    @property
    def name(self) -> str:
        return "fake-embedding"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vec = [0.0] * _DIM
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % _DIM
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec


class FakeLLMProvider(LLMProvider):
    """Extractive stand-in for a chat model.

    Answers by echoing the most relevant retrieved text so grounding is
    obvious. The relevance gate is identified by its system prompt and answered
    with ``gate_reply`` (default "YES", i.e. allow).
    """

    def __init__(self, gate_reply: str = "YES") -> None:
        self._gate_reply = gate_reply
        self.answered = False

    @property
    def name(self) -> str:
        return "fake-llm"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if system_prompt == GATE_SYSTEM_PROMPT:
            return self._gate_reply
        self.answered = True
        match = re.search(r"\[Source \d+[^\]]*\]\s*(.+)", user_prompt, re.DOTALL)
        snippet = match.group(1).strip()[:200] if match else "no sources"
        return f"Based on your saved content: {snippet}"
