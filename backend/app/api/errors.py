"""Maps domain errors to HTTP responses with a uniform error envelope."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.errors import (
    ContentFetchError,
    DomainError,
    EmptyKnowledgeBaseError,
    ItemNotFoundError,
    ProviderError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Domain error -> HTTP status code.
_STATUS_MAP: dict[type[DomainError], int] = {
    ValidationError: 422,
    ItemNotFoundError: 404,
    EmptyKnowledgeBaseError: 409,
    ContentFetchError: 400,
    ProviderError: 502,
}


def _envelope(error_type: str, message: str) -> dict:
    return {"error": {"type": error_type, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that translate exceptions into JSON error responses."""

    @app.exception_handler(DomainError)
    async def _handle_domain_error(_: Request, exc: DomainError) -> JSONResponse:
        status = _STATUS_MAP.get(type(exc), 400)
        # 5xx logs at error level; 4xx are client faults.
        log = logger.error if status >= 500 else logger.warning
        log("domain error", extra={"error_type": type(exc).__name__,
                                   "status": status, "detail": str(exc)})
        return JSONResponse(
            status_code=status,
            content=_envelope(type(exc).__name__, str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Surface only the first validation problem.
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        message = first.get("msg", "Invalid request.")
        detail = f"{loc}: {message}" if loc else message
        return JSONResponse(
            status_code=422,
            content=_envelope("RequestValidationError", detail),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", extra={"error": str(exc)})
        return JSONResponse(
            status_code=500,
            content=_envelope(
                "InternalServerError",
                "An unexpected error occurred. Check server logs for details.",
            ),
        )
