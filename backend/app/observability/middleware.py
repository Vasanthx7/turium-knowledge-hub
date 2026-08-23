"""Trace middleware: assigns/echoes a trace_id, times the request, sets X-Request-ID.

Pure ASGI (not BaseHTTPMiddleware) so the ContextVar set here stays visible to
the route handler and services; BaseHTTPMiddleware runs the app in a separate
task and breaks that propagation.
"""

from __future__ import annotations

import logging
import re
import time

from app.observability.tracing import new_trace

logger = logging.getLogger("app.trace")

# Client-supplied X-Request-ID is untrusted: accept only a short, safe token.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _clean_incoming_id(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    return value if _SAFE_ID.match(value) else None


class TraceMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        trace = new_trace(_clean_incoming_id(headers.get(b"x-request-id")))

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.perf_counter()
        status_code = {"value": 0}

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status_code["value"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append(
                    (b"x-request-id", trace.trace_id.encode())
                )
            await send(message)

        logger.info("request.start", extra={"method": method, "path": path})
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request.end",
                extra={
                    "method": method,
                    "path": path,
                    "status": status_code["value"],
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "steps": len(trace.events),
                },
            )
