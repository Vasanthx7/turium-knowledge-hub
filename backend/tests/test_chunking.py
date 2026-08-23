"""Unit tests for the chunking strategy."""

from __future__ import annotations

import pytest

from app.services.chunking import OverlappingCharacterChunker


def test_short_text_is_single_chunk():
    chunker = OverlappingCharacterChunker(chunk_size=100, overlap=20)
    chunks = chunker.split("A short note.")
    assert chunks == ["A short note."]


def test_empty_text_yields_no_chunks():
    chunker = OverlappingCharacterChunker(chunk_size=100, overlap=20)
    assert chunker.split("   \n  ") == []


def test_long_text_is_split_with_overlap():
    chunker = OverlappingCharacterChunker(chunk_size=120, overlap=30)
    text = " ".join(f"sentence number {i}." for i in range(60))
    chunks = chunker.split(text)

    assert len(chunks) > 1
    # No chunk materially exceeds the size bound (allow slack for snapping).
    assert all(len(c) <= 120 + 30 for c in chunks)
    # Chunks reassemble to cover the whole input (overlap means duplication).
    assert "sentence number 0" in chunks[0]
    assert "sentence number 59" in chunks[-1]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        OverlappingCharacterChunker(chunk_size=100, overlap=100)
