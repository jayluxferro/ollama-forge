"""Shared logger for ollama-forge; control verbosity via set_verbose() from CLI."""

from __future__ import annotations

import logging
import sys


def get_logger() -> logging.Logger:
    """Return the package logger. Handler is added on first use if not already configured."""
    logger = logging.getLogger("ollama_forge")
    _ensure_handler(logger)
    return logger


def _ensure_handler(logger: logging.Logger) -> None:
    if logger.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def set_verbose(verbose: bool) -> None:
    """Set log level to DEBUG if verbose else INFO. Call from CLI after parsing --verbose."""
    logger = get_logger()
    _ensure_handler(logger)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
