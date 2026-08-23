"""Small text helpers shared across services."""

from __future__ import annotations


def derive_note_title(text: str) -> str:
    """Use the first non-empty line (truncated) as a note's title."""
    first_line = next(
        (ln.strip() for ln in text.splitlines() if ln.strip()), "Untitled note"
    )
    return first_line[:80] + ("…" if len(first_line) > 80 else "")
