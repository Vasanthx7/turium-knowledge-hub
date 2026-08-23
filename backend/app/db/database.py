"""SQLite connection factory and schema bootstrap.

Source of truth for items, chunks and embeddings; embeddings stored as JSON
float arrays. Nearest-neighbour search runs in-memory (see VectorIndex).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id           TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL,
    title        TEXT NOT NULL,
    content      TEXT NOT NULL,
    source_url   TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    item_id     TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,  -- JSON array of floats
    FOREIGN KEY (item_id) REFERENCES items (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_item_id ON chunks (item_id);
CREATE INDEX IF NOT EXISTS idx_items_created_at ON items (created_at DESC);
"""


class Database:
    """Owns the SQLite database file and hands out configured connections."""

    def __init__(self, path: str) -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # In-memory DB needs one shared connection kept alive, or the schema
        # vanishes between calls; check_same_thread=False for the threadpool.
        self._shared = (
            sqlite3.connect(path, check_same_thread=False)
            if path == ":memory:"
            else None
        )
        if self._shared is not None:
            self._configure(self._shared)

    def _configure(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection, closing it if it is not shared.

        File-backed connections are opened per operation and closed on exit;
        leaked handles would lock the DB file on Windows.
        """
        if self._shared is not None:
            yield self._shared
            return
        conn = sqlite3.connect(self._path)
        self._configure(conn)
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create tables and indexes if they do not already exist."""
        with self.connection() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()
