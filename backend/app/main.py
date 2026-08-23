"""FastAPI application entrypoint.

Run with: ``uvicorn app.main:app --reload``
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.config import get_settings
from app.container import Container
from app.logging_config import configure_logging
from app.observability.middleware import TraceMiddleware

logger = logging.getLogger(__name__)


def _lifespan(container: Container | None):
    """Build the lifespan handler; tests inject a container, else build from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        configure_logging(settings.log_level)

        app.state.container = container or Container(settings)
        app.state.container.warm_up()

        logger.info("application started", extra={"version": app.version})
        yield
        logger.info("application shutting down")

    return lifespan


def create_app(container: Container | None = None) -> FastAPI:
    """Application factory — keeps construction testable and side-effect free."""
    app = FastAPI(
        title="Turium Knowledge Hub",
        version="1.0.0",
        description="Save notes/URLs and ask questions over them with RAG.",
        lifespan=_lifespan(container),
    )

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Added last so it wraps outermost; trace covers CORS handling too.
    app.add_middleware(TraceMiddleware)

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
