from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

RUNS_DIR = Path("observability_runs")
RUNS_DIR.mkdir(exist_ok=True)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 6),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "stage", "event", "extra_fields"):
            val = getattr(record, key, None)
            if val is not None:
                if key == "extra_fields" and isinstance(val, dict):
                    payload.update(val)
                else:
                    payload[key] = val
        return json.dumps(payload, default=str)


def get_logger(request_id: str) -> logging.Logger:
    """Returns a logger that writes JSON lines to stdout AND to a
    per-request file under observability_runs/, tagged with request_id."""
    logger = logging.getLogger(f"delta_chat.{request_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(RUNS_DIR / f"{request_id}.log.jsonl")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def log_event(logger: logging.Logger, request_id: str, stage: str, event: str, **fields):
    logger.info(event, extra={"request_id": request_id, "stage": stage, "event": event, "extra_fields": fields})
