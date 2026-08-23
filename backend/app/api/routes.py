"""HTTP routes for the Turium Knowledge Hub."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies import (
    get_ingest_service,
    get_item_service,
    get_rag_service,
)
from app.domain.models import Answer, Item
from app.domain.schemas import (
    CitationResponse,
    IngestRequest,
    IngestResponse,
    ItemDetailResponse,
    ItemListResponse,
    ItemResponse,
    QueryRequest,
    QueryResponse,
    UpdateItemRequest,
)
from app.observability import current_trace
from app.services.ingest_service import IngestService
from app.services.item_service import ItemService
from app.services.rag_service import RagService


def _json(body: BaseModel, debug: bool, status_code: int = 200) -> JSONResponse:
    """Serialise a response model, attaching the trace only when debug is set."""
    data = body.model_dump(mode="json")
    trace = current_trace() if debug else None
    if trace is not None:
        data["trace"] = trace.to_dict()
    else:
        data.pop("trace", None)
    return JSONResponse(content=data, status_code=status_code)

router = APIRouter()


# Mappers (domain -> transport)
def _to_item_response(item: Item) -> ItemResponse:
    return ItemResponse(
        id=item.id,
        source_type=item.source_type,
        title=item.title,
        preview=item.preview,
        source_url=item.source_url,
        created_at=item.created_at,
    )


def _to_item_detail(item: Item) -> ItemDetailResponse:
    return ItemDetailResponse(
        id=item.id,
        source_type=item.source_type,
        title=item.title,
        preview=item.preview,
        source_url=item.source_url,
        created_at=item.created_at,
        content=item.content,
    )


def _to_query_response(answer: Answer) -> QueryResponse:
    return QueryResponse(
        question=answer.question,
        answer=answer.answer,
        citations=[
            CitationResponse(
                item_id=c.item_id,
                title=c.title,
                source_type=c.source_type,
                source_url=c.source_url,
                snippet=c.snippet,
            )
            for c in answer.citations
        ],
    )


# Routes
@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a note or fetch & save a URL",
)
async def ingest(
    payload: IngestRequest,
    debug: bool = False,
    service: IngestService = Depends(get_ingest_service),
) -> Response:
    if payload.url:
        item, chunks = await service.ingest_url(payload.url)
    else:
        item, chunks = await service.ingest_note(payload.text, payload.title)
    body = IngestResponse(item=_to_item_response(item), chunks_created=chunks)
    return _json(body, debug, status.HTTP_201_CREATED)


@router.get(
    "/items",
    response_model=ItemListResponse,
    summary="List all saved items, newest first",
)
async def list_items(request: Request) -> ItemListResponse:
    repo = request.app.state.container.repository
    items = [_to_item_response(i) for i in repo.list_items()]
    return ItemListResponse(items=items, count=len(items))


@router.get(
    "/items/{item_id}",
    response_model=ItemDetailResponse,
    summary="Get a single saved item with its full content",
)
async def get_item(
    item_id: str,
    service: ItemService = Depends(get_item_service),
) -> ItemDetailResponse:
    return _to_item_detail(service.get(item_id))


@router.patch(
    "/items/{item_id}",
    response_model=ItemDetailResponse,
    summary="Edit an item's title and/or content",
)
async def update_item(
    item_id: str,
    payload: UpdateItemRequest,
    service: ItemService = Depends(get_item_service),
) -> ItemDetailResponse:
    item = await service.update(item_id, payload.title, payload.content)
    return _to_item_detail(item)


@router.delete(
    "/items/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a saved item and its indexed chunks",
)
async def delete_item(
    item_id: str,
    service: ItemService = Depends(get_item_service),
) -> Response:
    service.delete(item_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question over saved content (RAG)",
)
async def query(
    payload: QueryRequest,
    request: Request,
    debug: bool = False,
    service: RagService = Depends(get_rag_service),
) -> Response:
    top_k = payload.top_k or request.app.state.container.settings.top_k
    answer = await service.answer(payload.question, top_k)
    return _json(_to_query_response(answer), debug)


@router.get("/health", summary="Liveness & provider/index status")
async def health(request: Request) -> dict:
    container = request.app.state.container
    return {
        "status": "ok",
        "embedding_provider": container.embedder.name,
        "llm_provider": container.llm.name,
        "items": container.repository.count_items(),
        "indexed_chunks": container.vector_index.size(),
    }
