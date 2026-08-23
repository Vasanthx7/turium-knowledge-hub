"""SQLite-backed ItemRepository: all SQL for items and chunks."""

from __future__ import annotations

import json
from datetime import datetime

from app.db.database import Database
from app.domain.interfaces import ItemRepository
from app.domain.models import Chunk, Item, SourceType


class SqliteItemRepository(ItemRepository):
    """Persist items and their chunks in SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, item: Item, chunks: list[Chunk]) -> None:
        with self._db.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO items
                        (id, source_type, title, content, source_url, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.source_type.value,
                        item.title,
                        item.content,
                        item.source_url,
                        item.created_at.isoformat(),
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO chunks (id, item_id, chunk_index, text, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (c.id, c.item_id, c.index, c.text, json.dumps(c.embedding))
                        for c in chunks
                    ],
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def update(self, item: Item, chunks: list[Chunk] | None) -> None:
        with self._db.connection() as conn:
            try:
                conn.execute(
                    """
                    UPDATE items
                    SET title = ?, content = ?, source_url = ?
                    WHERE id = ?
                    """,
                    (item.title, item.content, item.source_url, item.id),
                )
                if chunks is not None:
                    # Replace the item's chunks wholesale (item row is kept).
                    conn.execute(
                        "DELETE FROM chunks WHERE item_id = ?", (item.id,)
                    )
                    conn.executemany(
                        """
                        INSERT INTO chunks
                            (id, item_id, chunk_index, text, embedding)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (c.id, c.item_id, c.index, c.text, json.dumps(c.embedding))
                            for c in chunks
                        ],
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def delete(self, item_id: str) -> None:
        with self._db.connection() as conn:
            # ON DELETE CASCADE removes the chunks with the item.
            conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
            conn.commit()

    def get(self, item_id: str) -> Item | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM items WHERE id = ?", (item_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    def list_items(self) -> list[Item]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM items ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, item_id, chunk_index, text, embedding FROM chunks"
            ).fetchall()
        return [
            Chunk(
                id=r["id"],
                item_id=r["item_id"],
                index=r["chunk_index"],
                text=r["text"],
                embedding=json.loads(r["embedding"]),
            )
            for r in rows
        ]

    def count_items(self) -> int:
        with self._db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) AS n FROM items"
            ).fetchone()["n"]

    @staticmethod
    def _row_to_item(row) -> Item:
        return Item(
            id=row["id"],
            source_type=SourceType(row["source_type"]),
            title=row["title"],
            content=row["content"],
            source_url=row["source_url"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
