"""Logging setup for the VISHUSTRA laboratory."""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "vishustra"
_LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOGGER_FORMAT))
    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")