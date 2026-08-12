"""Kalshi trade-print adapter -> trades canonical frame.

Verified wire shape (2026-08-12 against both endpoints):

    live:        GET /markets/trades?ticker=...&limit&cursor
    historical:  GET /historical/trades?ticker=...&limit&cursor
                 (no /markets segment in the historical path)

Both return {"trades": [...], "cursor": ...}. Each print carries
`count_fp` / `yes_price_dollars` / `no_price_dollars` as fixed-point
strings, `created_time` as ISO-8601 with sub-second precision,
`taker_side` ("yes" = aggressor bought YES, "no" = aggressor bought NO,
i.e. hit the YES bid) and `is_block_trade`.

The tape is ground truth for executability: a quote OHLC bar proves a
quote existed; only a print proves somebody could actually trade at it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient, parse_api_ts
from weadge.domain.time import utc_now
from weadge.storage.schema import TRADE_SCHEMA


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def trades_frame(
    client: KalshiClient,
    market_ticker: str,
    *,
    close_at: datetime | None = None,
) -> pl.DataFrame:
    """Fetch every public trade print for one market, oldest first.

    close_at (the market's close time) routes live vs historical; see
    KalshiClient.get_market_trades.
    """
    raw = client.get_market_trades(market_ticker, close_at=close_at)
    now = utc_now()
    rows = [
        {
            "market_ticker": market_ticker,
            "trade_id": t.get("trade_id"),
            "created_at": parse_api_ts(t["created_time"]),
            "yes_price": _f(t.get("yes_price_dollars")),
            "no_price": _f(t.get("no_price_dollars")),
            "count": _f(t.get("count_fp") or t.get("count")) or 0.0,
            "taker_side": t.get("taker_side"),
            "is_block_trade": bool(t.get("is_block_trade", False)),
            "ingested_at": now,
        }
        for t in raw
    ]
    return pl.DataFrame(rows, schema=TRADE_SCHEMA).sort("created_at")
