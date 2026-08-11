"""Live data recording — v2 feature, recorder format only.

Live L2/trade data is append-only JSONL.zst, never DuckDB. Batched into
Parquet later. See adapters/kalshi/websocket.py for the socket skeleton.
"""

from __future__ import annotations

import json
from pathlib import Path

import zstandard as zstd


class JsonlZstAppender:
    """Append dict rows to a compressed JSONL file (one file per hour bucket)."""

    def __init__(self, out_root: str | Path, series: str) -> None:
        self.root = Path(out_root)
        self.series = series
        self._fh: object | None = None
        self._current: str | None = None

    def append(self, timestamp: str, row: dict) -> None:
        """timestamp: ISO UTC; rows are bucketed by date/hour for replay."""
        hour_key = timestamp[:13]  # YYYY-MM-DDTHH
        if self._current != hour_key:
            self._rotate(hour_key)
        payload = {"ts": timestamp, **row}
        # mypy: stream_writer on self._fh — see _rotate
        self._fh.write((json.dumps(payload, default=str) + "\n").encode())  # type: ignore[attr-defined]

    def _rotate(self, hour_key: str) -> None:
        if self._fh is not None:
            self._fh.close()  # type: ignore[attr-defined]
        out_dir = self.root / hour_key[:10] / self.series
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{hour_key[11:]}.jsonl.zst"
        compressor = zstd.ZstdCompressor(level=3)
        # the stream writer must outlive a single append, so the file handle
        # is deliberately kept open across calls (not context-managed per write)
        self._fh = compressor.stream_writer(open(path, "ab"))  # noqa: SIM115
        self._current = hour_key

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()  # type: ignore[attr-defined]
            self._fh = None
            self._current = None
