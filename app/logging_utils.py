"""Structured (JSON-line) logging for the API process.

Every log record is emitted as a single JSON object, which is what log
shippers (Loki, ELK, Vector, ...) consume natively. A request id is bound to a
context variable by the HTTP middleware and automatically included in every
record logged while a request is being handled, so one trace is easy to follow.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON line, always including the request id."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("method", "path", "status", "duration_ms", "tenant_id", "actor"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, separators=(",", ":"))


def setup_logging() -> None:
    """Idempotent: install the JSON handler on the ``app`` and ``uvicorn``
    loggers. Called once at API startup."""
    root = logging.getLogger("app")
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False

    # The app already logs each request with structured JSON; silence the
    # default uvicorn access log (which duplicates it, in a different format).
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    if not any(
        isinstance(h.formatter, JsonFormatter)
        for h in logging.getLogger("uvicorn.error").handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logging.getLogger("uvicorn.error").addHandler(handler)
        logging.getLogger("uvicorn.error").propagate = False


log = logging.getLogger("app.main")
