"""RAG evaluation harness.

Runs the *real* pipeline (via the app's ``Container``) over a labelled golden
set and reports the two independent RAG failure modes separately:

  1. RETRIEVAL — did the right source(s) come back, and how highly ranked?
       Hit@k / MRR for single-source, plus CONTEXT RECALL and ALL-SOURCES-HIT
       for multi-source (cross-content) queries.
  2. GENERATION — is the answer grounded in the retrieved context?
       cites-expected-source (any / all), lexical groundedness, and an
       LLM-as-judge faithfulness + relevance score (when a real LLM is set).

It also checks that out-of-scope questions are correctly *declined*.

Usage (from the backend/ directory; uses the configured provider, Ollama by
default):

    python -m evaluation.evaluate                       # easy set
    python -m evaluation.evaluate golden_set_hard.json  # hard set
    EVAL_ANSWER_K=8 python -m evaluation.evaluate golden_set_hard.json

Env knobs: AI_PROVIDER (embed+llm), EVAL_ANSWER_K (chunks sent to the LLM),
EVAL_RANK_CHUNKS (chunks pulled for ranking), EVAL_RECALL_K (distinct docs the
recall/all-sources metrics look within).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys
from pathlib import Path

from app.config import Settings
from app.container import Container
from app.domain.models import RetrievedChunk

_HERE = Path(__file__).parent

_WORD = re.compile(r"[a-zA-Z]{4,}")
_STOP = {
    "this", "that", "with", "from", "your", "which", "into", "using", "used",
    "based", "content", "saved", "source", "sources", "answer", "question",
    "there", "their", "about", "most", "such", "here", "what", "when", "where",
    "they", "them", "then", "than", "have", "does", "would", "could", "should",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _expected_ids(query: dict) -> list[str]:
    """Normalise a query's expected source(s) to a list of golden doc ids."""
    if query.get("expected_doc_ids") is not None:
        return list(query["expected_doc_ids"])
    single = query.get("expected_doc_id")
    return [single] if single else []


def _ranked_docs(
    retrieved: list[RetrievedChunk], item_to_doc: dict[str, str]
) -> list[tuple[str, float]]:
    """Distinct docs in retrieval order with their best chunk score."""
    best: dict[str, float] = {}
    order: list[str] = []
    for rc in retrieved:
        doc = item_to_doc.get(rc.item.id, rc.item.id)
        if doc not in best:
            best[doc] = rc.score
            order.append(doc)
    return [(doc, best[doc]) for doc in order]


def _lexical_groundedness(answer: str, context: str) -> float:
    ctx = {w.lower() for w in _WORD.findall(context)}
    words = [w.lower() for w in _WORD.findall(answer) if w.lower() not in _STOP]
    if not words:
        return 0.0
    return sum(1 for w in words if w in ctx) / len(words)


_JUDGE_SYSTEM = (
    "You are a strict grader for a retrieval-augmented answer. You are given a "
    "QUESTION, the retrieved CONTEXT, and an ANSWER. Rate two things from 1 to "
    "5: FAITHFUL = how fully the answer is supported by the context (5 = every "
    "claim supported, 1 = unsupported/hallucinated), and RELEVANT = how well it "
    "addresses the question. Reply with EXACTLY one line: FAITHFUL=<n> RELEVANT=<n>"
)


async def _llm_judge(container, question, answer, context):
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    try:
        raw = await container.llm.complete(_JUDGE_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001
        print(f"      (judge failed: {exc})")
        return None
    f = re.search(r"FAITHFUL\s*=\s*([1-5])", raw)
    r = re.search(r"RELEVANT\s*=\s*([1-5])", raw)
    return (int(f.group(1)), int(r.group(1))) if f and r else None


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #
async def run(golden: dict) -> None:
    rank_chunks = int(os.getenv("EVAL_RANK_CHUNKS", "20"))
    recall_k = int(os.getenv("EVAL_RECALL_K", "6"))

    settings = Settings(database_path=":memory:")
    answer_k = int(os.getenv("EVAL_ANSWER_K", str(settings.top_k)))
    container = Container(settings)
    container.warm_up()

    print(f"\nProvider     : embed={container.embedder.name}  llm={container.llm.name}")
    print(f"Corpus       : {len(golden['documents'])} docs, "
          f"{len(golden['queries'])} queries")
    print(f"Params       : rank_chunks={rank_chunks}  recall_k={recall_k}  "
          f"answer_k={answer_k}\n")

    # Ingest corpus; map golden doc id <-> stored item id.
    doc_to_item: dict[str, str] = {}
    for doc in golden["documents"]:
        item, n = await container.ingest_service.ingest_note(doc["text"], doc["title"])
        doc_to_item[doc["id"]] = item.id
        print(f"  ingested {doc['id']:<14} ({n} chunk{'s' if n != 1 else ''})")
    item_to_doc = {v: k for k, v in doc_to_item.items()}
    print()

    positives = [q for q in golden["queries"] if _expected_ids(q)]
    negatives = [q for q in golden["queries"] if not _expected_ids(q)]

    rr, hits = [], {1: 0, 3: 0, 5: 0}
    recalls, all_hits = [], 0
    cites_any = cites_all = 0
    lexical, jf, jr = [], [], []

    print("═══ POSITIVES ═════════════════════════════════════════════════════")
    for q in positives:
        expected_docs = set(_expected_ids(q))

        retrieved = await container.retrieval_service.retrieve(q["question"], rank_chunks)
        ranked = _ranked_docs(retrieved, item_to_doc)
        ranked_docs = [d for d, _ in ranked]

        first_rank = next((i + 1 for i, d in enumerate(ranked_docs)
                           if d in expected_docs), 0)
        rr.append(1.0 / first_rank if first_rank else 0.0)
        for k in hits:
            if first_rank and first_rank <= k:
                hits[k] += 1

        top_set = set(ranked_docs[:recall_k])
        found = expected_docs & top_set
        recall = len(found) / len(expected_docs)
        recalls.append(recall)
        all_hit = expected_docs.issubset(top_set)
        all_hits += all_hit

        answer = await container.rag_service.answer(q["question"], answer_k)
        cited_docs = {item_to_doc.get(c.item_id, c.item_id) for c in answer.citations}
        context = "\n".join(rc.chunk.text for rc in retrieved[:answer_k])
        lexical.append(_lexical_groundedness(answer.answer, context))
        any_cited = bool(expected_docs & cited_docs)
        all_cited = expected_docs.issubset(cited_docs)
        cites_any += any_cited
        cites_all += all_cited

        judge = await _llm_judge(container, q["question"], answer.answer, context)
        if judge:
            jf.append(judge[0])
            jr.append(judge[1])

        top_str = ", ".join(f"{d}({s:.2f})" for d, s in ranked[:4])
        mark = "✓" if all_hit else ("~" if found else "✗")
        print(f"\n  {mark} [{q.get('type','?')}] {q['question']}")
        print(f"      expected : {', '.join(sorted(expected_docs))}")
        print(f"      top docs : {top_str}")
        judge_str = f" | judge F={judge[0]} R={judge[1]}" if judge else ""
        print(f"      rank={first_rank or '—'} | recall@{recall_k}="
              f"{len(found)}/{len(expected_docs)} | all-sources={'Y' if all_hit else 'N'}"
              f" | cites={'/'.join(sorted(cited_docs)) or '—'} "
              f"(all={'Y' if all_cited else 'N'}){judge_str}")

    print("\n═══ NEGATIVES (should be declined) ════════════════════════════════")
    correct_rej = 0
    for q in negatives:
        answer = await container.rag_service.answer(q["question"], answer_k)
        declined = len(answer.citations) == 0
        correct_rej += declined
        leaked = "" if declined else \
            f"  → cited {', '.join(item_to_doc.get(c.item_id, '?') for c in answer.citations)}"
        print(f"  {'✓' if declined else '✗'} "
              f"{'declined' if declined else 'ANSWERED'}  {q['question']}{leaked}")

    # ---- summary ------------------------------------------------------- #
    np_ = len(positives) or 1
    nn = len(negatives) or 1
    multi = [q for q in positives if len(_expected_ids(q)) > 1]
    print("\n══ RESULTS ════════════════════════════════════════════════════════")
    print(f"Queries: {len(positives)} positive ({len(multi)} multi-source), "
          f"{len(negatives)} negative")
    print("Retrieval:")
    print(f"  Hit@1 (≥1 source) : {hits[1]}/{len(positives)} = {hits[1]/np_:.0%}")
    print(f"  Hit@3 (≥1 source) : {hits[3]}/{len(positives)} = {hits[3]/np_:.0%}")
    print(f"  Hit@5 (≥1 source) : {hits[5]}/{len(positives)} = {hits[5]/np_:.0%}")
    print(f"  MRR               : {statistics.mean(rr):.3f}")
    print(f"  Context recall@{recall_k}  : {statistics.mean(recalls):.0%} "
          f"(avg fraction of needed sources retrieved)")
    print(f"  All-sources hit@{recall_k} : {all_hits}/{len(positives)} = "
          f"{all_hits/np_:.0%} (every needed source retrieved)")
    print("Generation:")
    print(f"  Cites ≥1 expected : {cites_any}/{len(positives)} = {cites_any/np_:.0%}")
    print(f"  Cites ALL expected: {cites_all}/{len(positives)} = {cites_all/np_:.0%}")
    print(f"  Lexical grounding : {statistics.mean(lexical):.2f}")
    if jf:
        print(f"  Judge faithfulness: {statistics.mean(jf):.2f} / 5")
        print(f"  Judge relevance   : {statistics.mean(jr):.2f} / 5")
    else:
        print("  LLM-judge         : no parseable verdicts")
    print("Robustness:")
    print(f"  Correct rejections: {correct_rej}/{len(negatives)} = {correct_rej/nn:.0%}")
    print("═══════════════════════════════════════════════════════════════════\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    set_name = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
        "EVAL_SET", "golden_set.json"
    )
    path = Path(set_name)
    if not path.is_absolute():
        path = _HERE / path
    golden = json.loads(path.read_text(encoding="utf-8"))
    print(f"Golden set   : {path.name}")

    from app.config import get_settings

    get_settings.cache_clear()
    asyncio.run(run(golden))


if __name__ == "__main__":
    main()
