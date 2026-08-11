"""Kalshi WebSocket recorder — v2 feature, skeleton only.

v0 deliberately does NOT stream L2. It is listed here so the adapter surface
is complete and the future live recorder has a defined home. Live data format
is append-only JSONL.zst (never DuckDB) so the historical lake stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KalshiWebSocket:
    """Placeholder for the v2 order-book recorder.

    The v1 historical pipeline is the source of truth; this class exists to
    keep the adapter layout stable and to document the intended wire format:

        data/raw/kalshi_ws/<YYYY-MM-DD>/<series>/<HH>.jsonl.zst

    Each line: {"ts": ..., "market": ..., "type": "book|trade|status", ...}
    """

    api_key: str = ""
    api_secret: str = ""
    out_root: Path | None = None
    _open: bool = field(default=False)

    async def connect(self, series_ticker: str) -> None:
        raise NotImplementedError(
            "KalshiWebSocket is a v2 component. "
            "v0 = historical research only; do not stream L2 before the "
            "alpha existence test has passed."
        )
