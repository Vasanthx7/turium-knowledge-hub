"""Request-scoped tracing built on a ContextVar; spans are timed and logged by trace_id."""

from app.observability.tracing import (
    Trace,
    current_trace,
    current_trace_id,
    new_trace,
    record,
    span,
    sync_span,
)

__all__ = [
    "Trace",
    "current_trace",
    "current_trace_id",
    "new_trace",
    "record",
    "span",
    "sync_span",
]
