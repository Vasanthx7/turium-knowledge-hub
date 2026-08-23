"""Request-scoped trace: spans, timings and a correlation id.

The active Trace lives in a ContextVar set per request by TraceMiddleware. With
no active trace, span/record calls degrade to plain logging.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger("app.trace")

_current: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)


@dataclass
class Event:
    """One recorded step: name, duration, and structured data."""

    step: str
    duration_ms: float
    data: dict[str, Any]


@dataclass
class Trace:
    """All steps recorded during a single request, sharing one trace_id."""

    trace_id: str
    events: list[Event] = field(default_factory=list)

    def add(self, step: str, duration_ms: float, data: dict[str, Any]) -> None:
        self.events.append(Event(step, round(duration_ms, 2), data))

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view, e.g. for a ?debug response."""
        return {
            "trace_id": self.trace_id,
            "steps": [
                {"step": e.step, "duration_ms": e.duration_ms, **e.data}
                for e in self.events
            ],
            "total_ms": round(sum(e.duration_ms for e in self.events), 2),
        }


def new_trace(trace_id: str | None = None) -> Trace:
    """Start a trace and make it the active one for this context."""
    trace = Trace(trace_id=trace_id or uuid.uuid4().hex[:12])
    _current.set(trace)
    return trace


def current_trace() -> Trace | None:
    """The trace active in this context, or None outside a request."""
    return _current.get()


def current_trace_id() -> str | None:
    trace = _current.get()
    return trace.trace_id if trace else None


def _emit(step: str, duration_ms: float, data: dict[str, Any]) -> None:
    trace = _current.get()
    if trace is not None:
        trace.add(step, duration_ms, data)
    logger.info(
        "span",
        extra={"step": step, "duration_ms": round(duration_ms, 2), **data},
    )


@asynccontextmanager
async def span(step: str, **data: Any) -> Iterator[dict[str, Any]]:
    """Time an async block as a span.

    Yields a mutable dict for attaching results discovered during the span.
    """
    start = time.perf_counter()
    fields: dict[str, Any] = dict(data)
    try:
        yield fields
    finally:
        _emit(step, (time.perf_counter() - start) * 1000.0, fields)


@contextmanager
def sync_span(step: str, **data: Any) -> Iterator[dict[str, Any]]:
    """Synchronous counterpart of span()."""
    start = time.perf_counter()
    fields: dict[str, Any] = dict(data)
    try:
        yield fields
    finally:
        _emit(step, (time.perf_counter() - start) * 1000.0, fields)


def record(step: str, **data: Any) -> None:
    """Record a zero-duration point event (no timing)."""
    _emit(step, 0.0, data)
