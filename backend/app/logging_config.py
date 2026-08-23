"""Structured single-line JSON logging setup."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.observability.tracing import current_trace_id


class TraceIdFilter(logging.Filter):
    """Attach the active request's trace_id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = current_trace_id()
        if trace_id is not None:
            record.trace_id = trace_id
        return True


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON objects."""

    # Attributes on every LogRecord; anything else is treated as extra context.
    _RESERVED = set(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge structured context passed through extra=.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(TraceIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quieten noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
