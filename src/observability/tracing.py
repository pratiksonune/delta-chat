from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

TRACES_DIR = Path("observability_runs")
TRACES_DIR.mkdir(exist_ok=True)


@dataclass
class Span:
    name: str
    start_ts: float
    end_ts: float | None = None
    status: str = "ok"
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.end_ts is None:
            return None
        return round((self.end_ts - self.start_ts) * 1000, 3)

    def to_dict(self):
        return {
            "name": self.name,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    request_id: str
    started_at: float = field(default_factory=time.time)
    spans: list[Span] = field(default_factory=list)
    trace_meta: dict = field(default_factory=dict)

    @contextmanager
    def span(self, name: str, **metadata):
        s = Span(name=name, start_ts=time.time(), metadata=dict(metadata))
        self.spans.append(s)
        try:
            yield s
            s.status = "ok"
        except Exception as e:
            s.status = "error"
            s.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            s.end_ts = time.time()

    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        return round((max((s.end_ts or s.start_ts) for s in self.spans) - self.started_at) * 1000, 3)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "started_at": self.started_at,
            "total_duration_ms": self.total_duration_ms(),
            "spans": [s.to_dict() for s in self.spans],
            "trace_meta": self.trace_meta,
        }

    def flush(self):
        path = TRACES_DIR / f"{self.request_id}.trace.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
