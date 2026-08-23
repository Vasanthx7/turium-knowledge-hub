"""Prompt construction for the RAG answer and relevance-gate steps."""

from __future__ import annotations

from app.domain.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions strictly from the user's "
    "saved notes and pages. Use only the provided sources. If the sources do "
    "not contain the answer, say so plainly. Cite sources inline as [Source N] "
    "where relevant. Be concise."
)

# yes/no relevance gate, run before generating to catch out-of-scope questions
# that slip past the cosine floor.
GATE_SYSTEM_PROMPT = (
    "You are a relevance classifier. Decide whether the provided context "
    "contains information that answers the question. The context may span "
    "several sources; answer YES if the answer can be assembled by combining "
    "them, even when no single source contains it on its own. Reply with "
    "exactly one word: YES or NO. Answer NO only when the context is unrelated "
    "to the question or lacks the information needed to answer it, even if it "
    "shares some keywords."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered source blocks shared by the answer and gate prompts."""
    blocks = [
        f"[Source {i}: {rc.item.title}]\n{rc.chunk.text}"
        for i, rc in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks) if blocks else "(no sources found)"


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the context block + question sent to the LLM.

    Sources are numbered so the model and citation layer refer to them
    consistently ([Source 1], [Source 2], ...).
    """
    return f"Context:\n{_format_context(chunks)}\n\nQuestion: {question}"


def build_gate_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the yes/no relevance-check prompt."""
    return (
        f"Context:\n{_format_context(chunks)}\n\nQuestion: {question}\n\n"
        "Does the context contain information that answers this question? "
        "Reply YES or NO."
    )
