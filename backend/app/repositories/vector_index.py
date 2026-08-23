"""In-memory vector index with vectorised cosine-similarity search.

Rebuilt from SQLite (the durable source of truth) on boot and updated on ingest.

Embeddings live in one contiguous ``float32`` matrix whose rows are L2-normalised
at insert time, so cosine similarity for a query is a single matrix-vector dot
product (``matrix @ q``) backed by BLAS rather than a Python loop per chunk. This
is ~1000x faster than a pure-Python scan and ~8x smaller than storing each vector
as a Python ``list[float]`` (4-byte float32 vs 24-byte float objects).

The chunk metadata kept alongside the matrix is stored *without* embeddings — the
durable copy lives in SQLite and nothing downstream reads ``chunk.embedding`` off
a retrieved chunk, so dropping it here is pure memory saved.

Interface is unchanged (``load``/``add``/``remove_item``/``size``/``search``), so
callers and tests are unaffected.
"""

from __future__ import annotations

import dataclasses
import threading

import numpy as np

from app.domain.models import Chunk


class VectorIndex:
    """A thread-safe, append-only in-memory store of embedded chunks."""

    def __init__(self) -> None:
        # Metadata rows (embeddings stripped), parallel to ``_matrix`` rows.
        self._chunks: list[Chunk] = []
        # (N, D) float32 matrix of L2-normalised embeddings; None until non-empty.
        self._matrix: np.ndarray | None = None
        self._lock = threading.Lock()

    def load(self, chunks: list[Chunk]) -> None:
        """Replace the index contents (used to rebuild from storage on boot)."""
        meta, matrix = self._prepare(chunks)
        with self._lock:
            self._chunks = meta
            self._matrix = matrix

    def add(self, chunks: list[Chunk]) -> None:
        """Append newly-embedded chunks to the index."""
        meta, matrix = self._prepare(chunks)
        if not meta:
            return
        with self._lock:
            self._chunks.extend(meta)
            if self._matrix is None:
                self._matrix = matrix
            elif self._matrix.shape[1] == matrix.shape[1]:
                self._matrix = np.vstack((self._matrix, matrix))
            else:
                # Dimension changed (e.g. embedding model swapped) — the old rows
                # can no longer be compared, so start fresh from this batch.
                self._chunks = list(meta)
                self._matrix = matrix

    def remove_item(self, item_id: str) -> None:
        """Drop every chunk belonging to ``item_id`` (on edit or delete)."""
        with self._lock:
            if not self._chunks:
                return
            keep = [i for i, c in enumerate(self._chunks) if c.item_id != item_id]
            if len(keep) == len(self._chunks):
                return  # nothing to remove
            self._chunks = [self._chunks[i] for i in keep]
            self._matrix = self._matrix[keep] if keep and self._matrix is not None else None

    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks by descending cosine similarity."""
        with self._lock:
            matrix = self._matrix
            chunks = self._chunks

        if matrix is None or not chunks or top_k <= 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.shape[0] != matrix.shape[1]:
            return []  # dimension mismatch — cannot compare
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q /= q_norm

        # Rows are pre-normalised, so cosine == dot product.
        scores = matrix @ q  # (N,)

        k = min(top_k, scores.shape[0])
        # argpartition gives the top-k unordered in O(N); sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(chunks[i], float(scores[i])) for i in top_idx]

    @staticmethod
    def _prepare(chunks: list[Chunk]) -> tuple[list[Chunk], np.ndarray | None]:
        """Filter to embedded chunks, returning stripped metadata + a normalised
        float32 matrix. Chunks whose dimension differs from the batch's first are
        skipped defensively (they can't be compared against the rest)."""
        meta: list[Chunk] = []
        vectors: list[list[float]] = []
        dim: int | None = None
        for c in chunks:
            if not c.embedding:
                continue
            if dim is None:
                dim = len(c.embedding)
            elif len(c.embedding) != dim:
                continue
            meta.append(dataclasses.replace(c, embedding=[]))
            vectors.append(c.embedding)

        if not vectors:
            return [], None

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid divide-by-zero: zero-norm rows stay zero (dot product -> 0).
        np.divide(matrix, norms, out=matrix, where=norms != 0)
        return meta, matrix
