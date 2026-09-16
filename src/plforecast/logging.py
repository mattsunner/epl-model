"""structlog setup: JSON lines on stderr, no bare prints anywhere in the pipeline.

Logs go to stderr so `plforecast evaluate > results.txt` captures results, not
interleaved JSON (story A-12). The logger looks up `sys.stderr` at every write rather
than binding the stream once: structlog caches loggers on first use, and a stream bound
under a test runner's capture would be closed by the time the next test logs.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


class _StderrLogger:
    """Minimal structlog logger: writes each rendered line to the *current* sys.stderr."""

    def msg(self, message: str) -> None:
        print(message, file=sys.stderr)

    log = debug = info = warning = error = critical = exception = fatal = msg


class _StderrLoggerFactory:
    def __call__(self, *args: Any) -> _StderrLogger:
        return _StderrLogger()


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=_StderrLoggerFactory(),
        cache_logger_on_first_use=True,
    )
