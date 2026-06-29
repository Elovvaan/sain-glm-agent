"""Structured logging setup for SAIN GLM Agent.

Supports three output formats:

* ``rich``  — coloured, human-friendly console output (default).
* ``plain`` — plain-text console output.
* ``json``  — newline-delimited JSON for log aggregators.

Usage::

    from sain_glm_agent.logging_ import setup_logging, get_logger

    setup_logging()                 # configure once at startup
    log = get_logger(__name__)
    log.info("Agent started")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Rich is an optional but listed dependency; guard import for safety.
try:
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_configured = False


def setup_logging(
    *,
    level: str = "INFO",
    log_format: str = "rich",
    log_file: Path | None = None,
) -> None:
    """Configure the root logger for the application.

    This function is idempotent — subsequent calls are no-ops unless you pass
    ``force=True`` via the standard :func:`logging.basicConfig` mechanism.

    Args:
        level: Log level string (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        log_format: Output style — ``rich``, ``plain``, or ``json``.
        log_file: If supplied, log records are also written to this file.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return
    _configured = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []

    if log_format == "rich" and _RICH_AVAILABLE:
        console_handler: logging.Handler = RichHandler(
            level=numeric_level,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
    elif log_format == "json":
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(_JsonFormatter())
        json_handler.setLevel(numeric_level)
        console_handler = json_handler
    else:
        plain_handler = logging.StreamHandler(sys.stdout)
        plain_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        plain_handler.setLevel(numeric_level)
        console_handler = plain_handler

    handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, inheriting the root configuration.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
