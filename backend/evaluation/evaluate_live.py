"""Live-data evaluation — runs over the items *you* have actually saved.

The golden-set evaluators use labelled synthetic corpora. This one instead
reads your real SQLite database and health-checks retrieval on it, with **no
manual labels**, using each item as its own ground truth:

  for every saved item →
    1. ask the LLM to invent a natural question that item answers
    2. run the real retrieval + RAG pipeline on that question
    3. check the item is retrieved (self-recall: Hit@1 / MRR) and that the
       answer cites it

It is fully read-only — it does not ingest, edit or delete anything.

Usage (from backend/, with your normal .env / provider running):

    python -m evaluation.evaluate_live
    EVAL_LIMIT=10 python -m evaluation.evaluate_live     # cap how many items

Because it uses a real LLM to generate questions, it needs the configured
provider (Ollama by default) to be running.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys

from app.config import get_settings
from app.container import Container
from app.domain.models import Item

_RANK_K = int(os.getenv("EVAL_RANK_CHUNKS", "10"))
_ANSWER_K = int(os.getenv("EVAL_ANSWER_K", "4"))

_Q_SYSTEM = (
    "You write exactly one natural question that the following text answers. "
    "Output only the question — no preamble, no quotes."
)


async def _make_question(llm, item: Item) -> str:
    """Have the LLM synthesise a question this item should answer."""
    try:
        reply = await llm.complete(_Q_SYSTEM, item.content[:1500])
        q = reply.strip().splitlines()[0].strip().strip('"') if reply.strip() else ""
    except Exception:  # noqa: BLE001
        q = ""
    return q or f"What does the note titled '{item.title}' describe?"


def _ranked_item_ids(retrieved) -> list[str]:
    seen: list[str] = []
    for rc in retrieved:
        if rc.item.id not in seen:
            seen.append(rc.item.id)
    return seen


async def run() -> None:
    settings = get_settings()  # real DB + provider from .env
    container = Container(settings)
    container.warm_up()

    items = container.repository.list_items()
    limit = int(os.getenv("EVAL_LIMIT", "0"))
    if limit > 0:
        items = items[:limit]

    print(f"\nProvider  : embed={container.embedder.name}  llm={container.llm.name}")
    print(f"Database  : {settings.database_path}")
    print(f"Items     : {len(items)} (chunks indexed: {container.vector_index.size()})\n")

    if not items:
        print("No saved items to evaluate. Add notes/URLs in the app first.\n")
        return

    rr, hits = [], {1: 0, 3: 0}
    cited = declined = 0

    print("═══ PER ITEM ══════════════════════════════════════════════════════")
    for item in items:
        question = await _make_question(container.llm, item)
        retrieved = await container.retrieval_service.retrieve(question, _RANK_K)
        ranked = _ranked_item_ids(retrieved)
        rank = ranked.index(item.id) + 1 if item.id in ranked else 0
        rr.append(1.0 / rank if rank else 0.0)
        for k in hits:
            if rank and rank <= k:
                hits[k] += 1

        answer = await container.rag_service.answer(question, _ANSWER_K)
        did_cite = any(c.item_id == item.id for c in answer.citations)
        was_declined = len(answer.citations) == 0
        cited += did_cite
        declined += was_declined

        mark = "✓" if rank == 1 else ("~" if rank else "✗")
        flag = "cited" if did_cite else ("DECLINED" if was_declined else "not-cited")
        print(f"\n  {mark} rank={rank or '—'} [{flag}] {item.title[:60]}")
        print(f"      q: {question[:90]}")

    n = len(items)
    print("\n══ RESULTS (live data) ════════════════════════════════════════════")
    print(f"  Self-recall Hit@1 : {hits[1]}/{n} = {hits[1]/n:.0%}")
    print(f"  Self-recall Hit@3 : {hits[3]}/{n} = {hits[3]/n:.0%}")
    print(f"  MRR               : {statistics.mean(rr):.3f}")
    print(f"  Answer cites item : {cited}/{n} = {cited/n:.0%}")
    print(f"  Declined by gate  : {declined}/{n} = {declined/n:.0%}")
    print("═══════════════════════════════════════════════════════════════════\n")
    print("Note: questions are LLM-generated from each item, so a low Hit@1 points\n"
          "to weak embeddings/index or near-duplicate items competing for the top\n"
          "rank; frequent 'DECLINED' means the relevance gate is too strict.\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    get_settings.cache_clear()
    asyncio.run(run())


if __name__ == "__main__":
    main()
