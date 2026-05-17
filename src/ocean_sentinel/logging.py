import logging
import os

import structlog

from ocean_sentinel.api.routes.logs import broadcast_processor


def _resolve_level() -> int:
    """OS_LOG_LEVEL overrides the default (NOTSET = show everything).
    Accepts both a name ("warning") and a numeric value ("30")."""
    raw = os.environ.get("OS_LOG_LEVEL", "").strip()
    if not raw:
        return 0
    if raw.isdigit():
        return int(raw)
    return getattr(logging, raw.upper(), 0)


def configure_logging(*, json_output: bool = False) -> None:
    """Call once at app startup. JSON in prod, colored console in dev."""

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        broadcast_processor,  # push every log to SSE subscribers
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_resolve_level()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )