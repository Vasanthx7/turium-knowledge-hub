"""Tests for the LLM relevance gate in RagService.

The gate call is identified by its system prompt, so the fake LLM returns a
fixed verdict: the answer is declined on NO and produced on YES.
"""

from __future__ import annotations

import pytest

from app.container import Container
from app.services.rag_service import RagService
from tests.fakes import FakeLLMProvider


def _rag_with(
    container: Container, gate_reply: str
) -> tuple[RagService, FakeLLMProvider]:
    llm = FakeLLMProvider(gate_reply=gate_reply)
    service = RagService(
        retrieval=container.retrieval_service,
        llm=llm,
        repository=container.repository,
        relevance_gate=True,
    )
    return service, llm


@pytest.mark.asyncio
async def test_gate_blocks_when_llm_says_no(container: Container):
    await container.ingest_service.ingest_note(
        "Paris is the capital of France.", title="Geo"
    )
    service, llm = _rag_with(container, "NO")

    answer = await service.answer("What is the capital of France?", top_k=3)

    assert answer.citations == []          # declined
    assert "couldn't find" in answer.answer.lower()
    assert llm.answered is False           # gate blocked before generation


@pytest.mark.asyncio
async def test_gate_allows_when_llm_says_yes(container: Container):
    await container.ingest_service.ingest_note(
        "Paris is the capital of France.", title="Geo"
    )
    service, llm = _rag_with(container, "YES")

    answer = await service.answer("What is the capital of France?", top_k=3)

    assert answer.citations, "expected citations when the gate allows"
    assert llm.answered is True


@pytest.mark.asyncio
async def test_gate_fails_open_on_garbled_reply(container: Container):
    """An unparseable verdict must not swallow a valid answer."""
    await container.ingest_service.ingest_note(
        "Paris is the capital of France.", title="Geo"
    )
    service, llm = _rag_with(container, "hmm, maybe?")

    answer = await service.answer("What is the capital of France?", top_k=3)

    assert llm.answered is True            # failed open → answered
