"""Resolver shadow log - stdlib JSONL, one file per day.

Deliberately independent from live/recorder: the production resolver must
not depend on the research sidecar (recorder down != bot down).

Rows:
- heartbeat: one per scan, always written (scanner liveness + observed state)
- lock: one per LOCKED bucket per scan (with executable book snapshot)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlAppender:
    """Append dict rows to a per-day JSONL file under out_root."""

    def __init__(self, out_root: str | Path, prefix: str = "shadow") -> None:
        self.root = Path(out_root)
        self.prefix = prefix
        self.root.mkdir(parents=True, exist_ok=True)
        self._fh: Any = None
        self._day: str = ""

    def _rotate(self) -> None:
        day = datetime.now(UTC).date().isoformat()
        if self._day != day:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.root / f"{self.prefix}-{day}.jsonl", "a")  # noqa: SIM115 - long-lived append handle
            self._day = day

    def append(self, row: dict) -> None:
        self._rotate()
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
