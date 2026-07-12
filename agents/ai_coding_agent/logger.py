import json
import logging
import os
import uuid
from datetime import datetime, timezone

LOGS_DIR = "logs"


def new_session_id() -> str:
    return uuid.uuid4().hex


def setup_session_logger(session_id: str, enabled: bool) -> logging.Logger:
    """Create a per-session logger that writes JSONL entries to logs/.

    When `enabled` is False, the logger has no handlers and never touches disk.
    """
    session_logger = logging.getLogger(f"ai_coding_agent.session.{session_id}")
    session_logger.setLevel(logging.DEBUG)
    session_logger.propagate = False

    if enabled:
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOGS_DIR, f"{timestamp}_{session_id}.jsonl")
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        session_logger.addHandler(handler)

    return session_logger


def serialize(obj):
    """Best-effort conversion of SDK/pydantic objects into plain JSON-able data."""
    if isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def log_event(session_logger: logging.Logger, session_id: str, event: str, data) -> None:
    if not session_logger.handlers:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "event": event,
        "data": serialize(data),
    }
    session_logger.debug(json.dumps(entry, default=str))
