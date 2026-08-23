"""Unit tests for the NumPy-backed in-memory VectorIndex.

Cover the retrieval contract (cosine-correct top-k), the mutation methods
(load / add / remove_item / size), and the defensive edge cases the search
path must survive (empty index, zero/degenerate query, dimension mismatch,
embedding-less chunks). A concurrency smoke test guards the locking.
"""

from __future__ import annotations

import math
import threading

import pytest

from app.domain.models import Chunk
from app.repositories.vector_index import VectorIndex


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _chunk(cid: str, embedding: list[float], item_id: str = "item", index: int = 0,
           text: str = "text") -> Chunk:
    return Chunk(id=cid, item_id=item_id, index=index, text=text,
                 embedding=embedding)


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# --------------------------------------------------------------------------- #
# Search correctness                                                          #
# --------------------------------------------------------------------------- #
def test_search_returns_cosine_ranked_topk():
    index = VectorIndex()
    index.add([
        _chunk("a", [1.0, 0.0, 0.0]),
        _chunk("b", [0.0, 1.0, 0.0]),
        _chunk("c", [0.9, 0.1, 0.0]),
    ])
    query = [1.0, 0.0, 0.0]
    results = index.search(query, top_k=2)

    assert [c.id for c, _ in results] == ["a", "c"]  # a exact, c close
    # Scores agree with a reference cosine implementation.
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[1][1] == pytest.approx(_cosine(query, [0.9, 0.1, 0.0]), abs=1e-5)


def test_search_matches_reference_ranking_on_many_vectors():
    import random

    random.seed(7)
    dim = 32
    chunks = [
        _chunk(f"{i:03d}", [random.gauss(0, 1) for _ in range(dim)], item_id=f"i{i}")
        for i in range(150)
    ]
    index = VectorIndex()
    index.add(chunks)

    query = [random.gauss(0, 1) for _ in range(dim)]
    got = index.search(query, top_k=5)
    expected = sorted(
        ((c, _cosine(query, c.embedding)) for c in chunks),
        key=lambda p: -p[1],
    )[:5]

    assert [c.id for c, _ in got] == [c.id for c, _ in expected]
    for (_, gs), (_, es) in zip(got, expected):
        assert gs == pytest.approx(es, abs=1e-4)


def test_topk_larger_than_index_returns_all():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    results = index.search([1.0, 1.0], top_k=10)
    assert len(results) == 2


def test_scores_are_plain_floats():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0])])
    (_, score), = index.search([1.0, 0.0], top_k=1)
    assert isinstance(score, float)  # not numpy.float32


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #
def test_empty_index_returns_empty():
    assert VectorIndex().search([1.0, 0.0], top_k=3) == []


def test_zero_query_returns_empty():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0])])
    assert index.search([0.0, 0.0], top_k=3) == []


def test_dimension_mismatch_query_returns_empty():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0, 0.0])])
    assert index.search([1.0, 0.0], top_k=3) == []


def test_non_positive_topk_returns_empty():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0])])
    assert index.search([1.0, 0.0], top_k=0) == []
    assert index.search([1.0, 0.0], top_k=-1) == []


def test_chunks_without_embeddings_are_skipped():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0]), _chunk("b", [])])
    assert index.size() == 1
    assert [c.id for c, _ in index.search([1.0, 0.0], top_k=5)] == ["a"]


def test_zero_norm_row_does_not_break_search():
    index = VectorIndex()
    index.add([_chunk("zero", [0.0, 0.0]), _chunk("real", [1.0, 0.0])])
    results = index.search([1.0, 0.0], top_k=2)
    assert results[0][0].id == "real"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Mutation                                                                     #
# --------------------------------------------------------------------------- #
def test_add_appends_across_batches():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0])])
    index.add([_chunk("b", [0.0, 1.0])])
    assert index.size() == 2
    assert {c.id for c, _ in index.search([1.0, 1.0], top_k=2)} == {"a", "b"}


def test_remove_item_drops_only_matching_chunks():
    index = VectorIndex()
    index.add([
        _chunk("a1", [1.0, 0.0], item_id="A"),
        _chunk("a2", [0.9, 0.1], item_id="A"),
        _chunk("b1", [0.0, 1.0], item_id="B"),
    ])
    index.remove_item("A")
    assert index.size() == 1
    assert [c.id for c, _ in index.search([0.0, 1.0], top_k=5)] == ["b1"]


def test_remove_missing_item_is_noop():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0], item_id="A")])
    index.remove_item("does-not-exist")
    assert index.size() == 1


def test_remove_all_then_search_is_empty():
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0], item_id="A")])
    index.remove_item("A")
    assert index.size() == 0
    assert index.search([1.0, 0.0], top_k=3) == []


def test_load_replaces_contents():
    index = VectorIndex()
    index.add([_chunk("old", [1.0, 0.0])])
    index.load([_chunk("new1", [0.0, 1.0]), _chunk("new2", [1.0, 1.0])])
    assert index.size() == 2
    assert {c.id for c, _ in index.search([0.0, 1.0], top_k=5)} == {"new1", "new2"}


def test_search_returns_metadata_without_embedding():
    """Retrieved chunks keep their metadata but drop the embedding (memory)."""
    index = VectorIndex()
    index.add([_chunk("a", [1.0, 0.0], item_id="I", index=3, text="hello")])
    (chunk, _), = index.search([1.0, 0.0], top_k=1)
    assert chunk.id == "a"
    assert chunk.item_id == "I"
    assert chunk.index == 3
    assert chunk.text == "hello"
    assert chunk.embedding == []  # stripped; durable copy lives in SQLite


def test_dimension_change_on_add_resets_index():
    """Swapping embedding model (new dim) can't compare to old rows; reset."""
    index = VectorIndex()
    index.add([_chunk("old", [1.0, 0.0])])
    index.add([_chunk("new", [1.0, 0.0, 0.0])])  # different dimension
    assert index.size() == 1
    assert [c.id for c, _ in index.search([1.0, 0.0, 0.0], top_k=5)] == ["new"]


# --------------------------------------------------------------------------- #
# Concurrency smoke test                                                       #
# --------------------------------------------------------------------------- #
def test_concurrent_add_and_search_is_safe():
    index = VectorIndex()
    dim = 16
    errors: list[Exception] = []

    def writer(start: int) -> None:
        try:
            for i in range(start, start + 50):
                vec = [0.0] * dim
                vec[i % dim] = 1.0
                index.add([_chunk(f"{i}", vec, item_id=f"i{i}")])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(100):
                index.search([1.0] + [0.0] * (dim - 1), top_k=3)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert index.size() == 150
