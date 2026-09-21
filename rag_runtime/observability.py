from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

SERVICE_NAME = "civicguide-ai"
LOGGER_NAME = "civicguide.operations"


def _logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def emit_event(event: str, *, level: str = "info", **fields: Any) -> None:
    """Emit one structured operational event without conversation content."""

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": SERVICE_NAME,
        "environment": os.getenv("VERCEL_ENV") or os.getenv("APP_ENV", "local"),
        "event": event,
        **fields,
    }
    message = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    log_method = getattr(_logger(), level, _logger().info)
    log_method(message)
