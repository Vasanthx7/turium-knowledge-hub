"""RAG use-case: retrieve context, ask the LLM, return a cited answer."""

from __future__ import annotations

import logging
import re

from app.domain.errors import EmptyKnowledgeBaseError
from app.domain.interfaces import ItemRepository, LLMProvider
from app.domain.models import Answer, Citation, RetrievedChunk
from app.observability import record, span
from app.services.prompt import (
    GATE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_gate_prompt,
    build_user_prompt,
)
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

_NO_ANSWER = (
    "I couldn't find anything in your saved notes or pages that answers this. "
    "I only answer from content you've saved — try rephrasing your question, or "
    "add a note or link that covers this topic and ask again."
)


def _is_negative_verdict(reply: str) -> bool:
    """True only when the whole gate reply reduces to a bare NO.

    Replies longer than two tokens fail open: a verbose reply starting with
    "No" (e.g. "No single source has it, but together they do") is a synthesis
    answer, not a decline.
    """
    words = reply.strip().lower().split()
    if not words or len(words) > 2:
        return False
    return re.sub(r"[^a-z]", "", words[0]) == "no"

# Coarse floor for dropping low-similarity chunks. Kept low because relevant
# and unrelated score distributions overlap; rejecting out-of-scope questions
# is the relevance gate's job.
_DEFAULT_MIN_SCORE = 0.05

# Cap the question text written to logs, keeping log lines bounded and limiting
# retained free-text user input.
_QUERY_LOG_MAX = 300


def _for_log(question: str) -> str:
    """Trim a question for inclusion in a log line."""
    q = question.strip()
    return q if len(q) <= _QUERY_LOG_MAX else q[:_QUERY_LOG_MAX] + "…"


class RagService:
    """Answer questions over the ingested corpus with cited sources."""

    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMProvider,
        repository: ItemRepository,
        min_score: float = _DEFAULT_MIN_SCORE,
        relevance_gate: bool = True,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._repo = repository
        self._min_score = min_score
        self._relevance_gate = relevance_gate

    async def answer(self, question: str, top_k: int) -> Answer:
        # log first so requests that bail out below are still traceable.
        record("query", question=_for_log(question), chars=len(question), top_k=top_k)

        if self._repo.count_items() == 0:
            raise EmptyKnowledgeBaseError(
                "No content has been ingested yet. Add a note or URL first."
            )

        retrieved = await self._retrieval.retrieve(question, top_k)
        relevant = [rc for rc in retrieved if rc.score >= self._min_score]
        record(
            "filter",
            candidates=len(retrieved),
            kept=len(relevant),
            min_score=self._min_score,
        )

        if not relevant:
            return self._declined(question, reason="no_chunks_above_floor")

        # cosine floor can't separate out-of-scope from weak-but-valid chunks,
        # so ask the LLM whether the context answers before generating.
        if self._relevance_gate:
            async with span("gate") as s:
                passed = await self._context_answers(question, relevant)
                s["verdict"] = "pass" if passed else "decline"
            if not passed:
                return self._declined(question, reason="gate_declined")

        user_prompt = build_user_prompt(question, relevant)
        async with span("generate", llm=self._llm.name, chars_in=len(user_prompt)) as s:
            text = await self._llm.complete(SYSTEM_PROMPT, user_prompt)
            s["chars_out"] = len(text)

        answer = Answer(
            question=question,
            answer=text,
            citations=self._build_citations(relevant),
        )
        record("answer", citations=len(answer.citations))
        return answer

    async def _context_answers(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> bool:
        """Ask the LLM whether the context can answer the question.

        Fails open on error or unparseable reply.
        """
        prompt = build_gate_prompt(question, chunks)
        try:
            reply = await self._llm.complete(GATE_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001 — never block answering on the gate
            logger.warning("relevance gate errored; failing open",
                           extra={"error": str(exc)})
            return True
        return not _is_negative_verdict(reply)

    def _declined(self, question: str, reason: str) -> Answer:
        record("declined", reason=reason)
        return Answer(question=question, answer=_NO_ANSWER, citations=[])

    @staticmethod
    def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
        """One citation per distinct source item, keeping its best-scoring chunk."""
        best_by_item: dict[str, RetrievedChunk] = {}
        for rc in chunks:
            existing = best_by_item.get(rc.item.id)
            if existing is None or rc.score > existing.score:
                best_by_item[rc.item.id] = rc

        ordered = sorted(
            best_by_item.values(), key=lambda rc: rc.score, reverse=True
        )
        return [
            Citation(
                item_id=rc.item.id,
                title=rc.item.title,
                source_type=rc.item.source_type,
                source_url=rc.item.source_url,
                snippet=_snippet(rc.chunk.text),
                score=round(rc.score, 4),
            )
            for rc in ordered
        ]


def _snippet(text: str, limit: int = 300) -> str:
    clean = text.strip().replace("\n", " ")
    return clean[:limit] + ("…" if len(clean) > limit else "")
