"""Fixed character-window chunking with overlap and boundary snapping.

Snaps chunk boundaries to the nearest paragraph/sentence break; overlap
preserves context for answers that straddle a boundary.
"""

from __future__ import annotations

import re

from app.domain.interfaces import ChunkingStrategy

# break preference: paragraph, then sentence end, then whitespace.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_COLLAPSE = re.compile(r"[ \t]+")


class OverlappingCharacterChunker(ChunkingStrategy):
    """Fixed-size character chunker with overlap and boundary snapping."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[str]:
        text = self._normalize(text)
        if not text:
            return []
        if len(text) <= self._size:
            return [text]

        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self._size, n)
            if end < n:
                end = self._snap_boundary(text, start, end)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            # advance, keeping ``overlap`` chars of trailing context; +1 floor
            # guarantees forward progress.
            start = max(end - self._overlap, start + 1)
        return chunks

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _WHITESPACE_COLLAPSE.sub(" ", text)
        return text.strip()

    def _snap_boundary(self, text: str, start: int, end: int) -> int:
        """Move ``end`` back to the nearest break within ~20% of chunk size.

        Falls back to the hard ``end`` if no break is close enough.
        """
        window = text[start:end]
        lookback = max(int(self._size * 0.2), 1)

        para = window.rfind("\n\n")
        if para >= len(window) - lookback and para > 0:
            return start + para

        sentence_breaks = list(_SENTENCE_END.finditer(window))
        for match in reversed(sentence_breaks):
            if match.end() >= len(window) - lookback:
                return start + match.end()

        space = window.rfind(" ")
        if space >= len(window) - lookback and space > 0:
            return start + space

        return end
