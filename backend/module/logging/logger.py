from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from backend.module.config_handler import settings

_LOGGER_CACHE: Dict[str, logging.Logger] = {}


class JsonLineFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            payload["extra"] = extra_data
        return json.dumps(payload, ensure_ascii=True)


def _ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_json_logger(module_name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped JSONL logger writing to logs/{module_name}.jsonl."""
    if module_name in _LOGGER_CACHE:
        return _LOGGER_CACHE[module_name]

    log_dir = settings.logging_directory
    _ensure_log_dir(log_dir)
    logfile = log_dir / f"{module_name}.jsonl"

    logger = logging.getLogger(f"backend.{module_name}")
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)

    _LOGGER_CACHE[module_name] = logger
    return logger


class LoggingInterceptor:
    """Thin convenience wrapper for consistent structured logging."""

    def __init__(self, module_name: str, *, level: int = logging.INFO) -> None:
        self._logger = get_json_logger(module_name, level=level)

    def info(self, message: str, **extra: Any) -> None:
        self._logger.info(message, extra={"extra_data": extra or None})

    def warning(self, message: str, **extra: Any) -> None:
        self._logger.warning(message, extra={"extra_data": extra or None})

    def error(self, message: str, **extra: Any) -> None:
        self._logger.error(message, extra={"extra_data": extra or None})

    def exception(self, message: str, **extra: Any) -> None:
        self._logger.exception(message, extra={"extra_data": extra or None})

