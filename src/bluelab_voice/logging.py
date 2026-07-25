"""Structured logging (structlog) with attempt/request correlation (OBS-1 / 01 §5).

Mirrors `apps/api` logging but the worker's correlation key is `attempt_id` (one worker process
per attempt, 07 §6), plus an optional `request_id` for the outbound api calls (bundle fetch,
callbacks). Both are carried in contextvars so every log line in a call is automatically tagged.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

import structlog

_attempt_id: ContextVar[str | None] = ContextVar("attempt_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_attempt_id(value: str | None) -> None:
    _attempt_id.set(value)


def get_attempt_id() -> str | None:
    return _attempt_id.get()


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


def _add_correlation(_logger, _method, event_dict):
    aid = _attempt_id.get()
    if aid:
        event_dict["attempt_id"] = aid
    rid = _request_id.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_correlation,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


class _PromotePreemptiveDebug(logging.Filter):
    """Surface the SDK's DEBUG 'using preemptive generation' line at INFO.

    livekit-agents logs preemptive-generation HITS only at DEBUG (misses are WARNING), so at the
    production INFO level we could never confirm the feature engages. Attached to the
    `livekit.agents` logger, this promotes that one record to INFO (its `preemptive_lead_time`
    extra shows how much head start the LLM got) and drops every other DEBUG record, so raising
    the logger to DEBUG does not flood INFO-level handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.DEBUG:
            return True
        if "using preemptive generation" in str(record.msg):
            record.levelno = logging.INFO
            record.levelname = "INFO"
            return True
        return False


def surface_preemptive_generation_logs() -> None:
    """Make preemptive-generation hits visible at INFO on the `livekit.agents` logger.

    Must run in the JOB process (e.g. from the prewarm hook) — that is where AgentActivity logs.
    """
    lk = logging.getLogger("livekit.agents")
    lk.setLevel(logging.DEBUG)
    lk.addFilter(_PromotePreemptiveDebug())
