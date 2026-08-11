"""Kalshi 1-minute candlestick adapter -> quote_1m canonical frame."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient
from weadge.domain.time import from_timestamp, utc_now
from weadge.storage.schema import QUOTE_1M_SCHEMA


def _ohlc(candle: dict[str, Any], side: str) -> tuple[float | None, ...]:
    block = candle.get(f"yes_{side}") or {}
    return (
        _f(block.get("open")),
        _f(block.get("high")),
        _f(block.get("low")),
        _f(block.get("close")),
    )


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def candles_frame(
    client: KalshiClient,
    market_ticker: str,
    start: datetime,
    end: datetime,
    *,
    interval: str = "1m",
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch 1-minute YES bid/ask OHLC candles for a market."""
    raw_rows = client.get_market_candles(market_ticker, start, end, period_interval=interval)
    rows = []
    for c in raw_rows:
        bid_open, bid_high, bid_low, bid_close = _ohlc(c, "bid")
        ask_open, ask_high, ask_low, ask_close = _ohlc(c, "ask")
        start = from_timestamp(int(c["start_ts"]))
        rows.append(
            {
                "market_ticker": market_ticker,
                "ts": start,
                "bar_start_at": start,
                "bar_end_at": start + timedelta(minutes=1),
                "yes_bid_open": bid_open,
                "yes_bid_high": bid_high,
                "yes_bid_low": bid_low,
                "yes_bid_close": bid_close,
                "yes_ask_open": ask_open,
                "yes_ask_high": ask_high,
                "yes_ask_low": ask_low,
                "yes_ask_close": ask_close,
                "volume": int(c.get("volume") or 0),
                "open_interest": int(c.get("open_interest") or 0),
                "ingested_at": utc_now(),
            }
        )
    if save_raw and raw_rows:
        lake = getattr(client, "_lake", None)
        if lake is not None:
            lake.save_raw_jsonl("kalshi", f"candles/{market_ticker}", raw_rows)
    return pl.DataFrame(rows, schema=QUOTE_1M_SCHEMA)
