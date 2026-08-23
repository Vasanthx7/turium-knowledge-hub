"""API-level tests using FastAPI's TestClient with fake providers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container
from app.main import create_app
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def client(tmp_path):
    # Build the app with a container wired to fake providers and an isolated DB.
    container = Container(
        Settings(_env_file=None, database_path=str(tmp_path / "test.db")),
        embedder=FakeEmbeddingProvider(),
        llm=FakeLLMProvider(),
    )
    with TestClient(create_app(container)) as c:
        yield c


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["embedding_provider"] == "fake-embedding"


def test_ingest_validation_rejects_both_fields(client: TestClient):
    resp = client.post("/ingest", json={"text": "hi", "url": "http://x.com"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "RequestValidationError"


def test_full_flow(client: TestClient):
    # Ingest.
    r1 = client.post("/ingest", json={"text": "Mars is the fourth planet."})
    assert r1.status_code == 201
    assert r1.json()["chunks_created"] >= 1

    # List.
    r2 = client.get("/items")
    assert r2.status_code == 200
    assert r2.json()["count"] == 1

    # Query.
    r3 = client.post("/query", json={"question": "Which planet is fourth?"})
    assert r3.status_code == 200
    body = r3.json()
    assert body["answer"]
    assert len(body["citations"]) >= 1


def test_query_empty_kb_returns_409(client: TestClient):
    resp = client.post("/query", json={"question": "anything here?"})
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "EmptyKnowledgeBaseError"


def test_item_detail_edit_delete_flow(client: TestClient):
    item_id = client.post(
        "/ingest", json={"text": "Original body.", "title": "Orig"}
    ).json()["item"]["id"]

    # Detail returns full content.
    detail = client.get(f"/items/{item_id}")
    assert detail.status_code == 200
    assert detail.json()["content"] == "Original body."

    # Edit content.
    edited = client.patch(f"/items/{item_id}", json={"content": "New body text."})
    assert edited.status_code == 200
    assert edited.json()["content"] == "New body text."

    # Delete.
    assert client.delete(f"/items/{item_id}").status_code == 204
    assert client.get(f"/items/{item_id}").status_code == 404
    assert client.get("/items").json()["count"] == 0


def test_get_missing_item_returns_404(client: TestClient):
    assert client.get("/items/nope").status_code == 404


def test_update_requires_a_field(client: TestClient):
    item_id = client.post("/ingest", json={"text": "body"}).json()["item"]["id"]
    resp = client.patch(f"/items/{item_id}", json={})
    assert resp.status_code == 422


def test_note_response_keeps_null_source_url(client: TestClient):
    # A note has no source_url; the field must be present as null, not dropped.
    item = client.post("/ingest", json={"text": "a note body"}).json()["item"]
    assert "source_url" in item and item["source_url"] is None


def test_query_short_after_strip_rejected(client: TestClient):
    # "  a  " is long enough raw but trims to 1 char — must fail validation.
    resp = client.post("/query", json={"question": "  a  "})
    assert resp.status_code == 422


def test_request_id_valid_is_echoed(client: TestClient):
    resp = client.get("/health", headers={"X-Request-ID": "abc-123_XYZ"})
    assert resp.headers.get("x-request-id") == "abc-123_XYZ"


def test_request_id_malicious_is_replaced(client: TestClient):
    resp = client.get("/health", headers={"X-Request-ID": "bad id with spaces!!"})
    returned = resp.headers.get("x-request-id")
    assert returned and returned != "bad id with spaces!!"


def test_query_debug_returns_trace(client: TestClient):
    client.post("/ingest", json={"text": "Mars is the fourth planet from the Sun."})
    resp = client.post("/query?debug=true", json={"question": "Which planet is fourth?"})
    assert resp.status_code == 200

    # Every response carries a correlation id.
    assert resp.headers.get("x-request-id")

    trace = resp.json().get("trace")
    assert trace and trace["trace_id"]
    steps = [s["step"] for s in trace["steps"]]
    # The RAG flow is visible: retrieve → gate → generate.
    assert "retrieve" in steps and "gate" in steps and "generate" in steps
    assert all("duration_ms" in s for s in trace["steps"])


def test_query_without_debug_omits_trace(client: TestClient):
    client.post("/ingest", json={"text": "Some content."})
    body = client.post("/query", json={"question": "what content is here?"}).json()
    assert "trace" not in body  # excluded when not requested
