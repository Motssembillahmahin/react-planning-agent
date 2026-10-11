"""JSONL trace writer/reader. Every phase transition of a run becomes one line."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from react_agent.types import Phase


class Tracer:
    """Appends events to an in-memory list and, if `path` is set, to a JSONL file."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        run_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self._clock = clock
        self.phase: Phase | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def emit(self, event: str, **data: Any) -> None:
        record: dict[str, Any] = {
            "seq": len(self.events),
            "ts": round(self._clock(), 6),
            "run_id": self.run_id,
            "event": event,
            **data,
        }
        self.events.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    def transition(self, to: Phase, **data: Any) -> None:
        """Record a state-machine transition from the current phase to `to`."""
        src = self.phase
        self.phase = to
        self.emit("transition", **{"from": src.value if src else None, "to": to.value}, **data)

    def transitions(self) -> list[tuple[str | None, str]]:
        return [(e["from"], e["to"]) for e in self.events if e["event"] == "transition"]


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL trace, skipping blank lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
