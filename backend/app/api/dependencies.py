"""FastAPI dependency providers exposing container services to route handlers."""

from __future__ import annotations

from fastapi import Request

from app.container import Container
from app.services.ingest_service import IngestService
from app.services.item_service import ItemService
from app.services.rag_service import RagService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_ingest_service(request: Request) -> IngestService:
    return get_container(request).ingest_service


def get_item_service(request: Request) -> ItemService:
    return get_container(request).item_service


def get_rag_service(request: Request) -> RagService:
    return get_container(request).rag_service
