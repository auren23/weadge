"""Kalshi 1-minute candlestick adapter -> quote_1m canonical frame.

Verified wire shape (2026-08-11): candles carry `end_period_ts` (epoch s) —
the bar's END — not start_ts. The bar bounds are therefore derived:

    bar_end_at   = end_period_ts
    bar_start_at = bar_end_at - period_interval_s
    ts           = bar_start_at   (canonical bar-start timestamp)

Bid/ask OHLC come back in two shapes that must both be parsed:
    live:        yes_ask = {close_dollars, high_dollars, low_dollars, open_dollars}
    historical:  yes_ask = {close, high, low, open}
Both are dollar-denominated strings (e.g. "0.0300"). Volume arrives as
`volume_fp` (live) or `volume` (historical), also a string.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient
from weadge.domain.time import from_timestamp, utc_now
from weadge.storage.schema import QUOTE_1M_SCHEMA


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ohlc(block: dict[str, Any] | None) -> tuple[float | None, ...]:
    """Bid/ask OHLC from either the live (*_dollars) or historical (bare)
    field naming. Both are dollar-denominated strings."""
    block = block or {}
    open_v = _f(block.get("open_dollars") or block.get("open"))
    high_v = _f(block.get("high_dollars") or block.get("high"))
    low_v = _f(block.get("low_dollars") or block.get("low"))
    close_v = _f(block.get("close_dollars") or block.get("close"))
    return open_v, high_v, low_v, close_v


def _int_fp(v: Any) -> int:
    """volume / open_interest arrive as fixed-point strings (e.g. "93.51")."""
    f = _f(v)
    return int(f) if f is not None else 0


def candles_frame(
    client: KalshiClient,
    market_ticker: str,
    start: datetime,
    end: datetime,
    *,
    series_ticker: str | None = None,
    period_interval_s: int = 60,
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch 1-minute YES bid/ask OHLC candles for a market.

    Live candles need series_ticker (the live path is
    /series/{series}/markets/{ticker}/candlesticks); historical candles do
    not. The client routes by the historical cutoff automatically.
    """
    raw_rows = client.get_market_candles(
        market_ticker,
        series_ticker,
        start,
        end,
        period_interval_s=period_interval_s,
    )
    rows = []
    for c in raw_rows:
        bid_open, bid_high, bid_low, bid_close = _ohlc(c.get("yes_bid"))
        ask_open, ask_high, ask_low, ask_close = _ohlc(c.get("yes_ask"))
        bar_end = from_timestamp(int(c["end_period_ts"]))
        bar_start = bar_end - timedelta(seconds=period_interval_s)
        rows.append(
            {
                "market_ticker": market_ticker,
                "ts": bar_start,
                "bar_start_at": bar_start,
                "bar_end_at": bar_end,
                "yes_bid_open": bid_open,
                "yes_bid_high": bid_high,
                "yes_bid_low": bid_low,
                "yes_bid_close": bid_close,
                "yes_ask_open": ask_open,
                "yes_ask_high": ask_high,
                "yes_ask_low": ask_low,
                "yes_ask_close": ask_close,
                "volume": _int_fp(c.get("volume_fp") or c.get("volume")),
                "open_interest": _int_fp(c.get("open_interest_fp") or c.get("open_interest")),
                "ingested_at": utc_now(),
            }
        )
    if save_raw and raw_rows:
        lake = getattr(client, "_lake", None)
        if lake is not None:
            lake.save_raw_jsonl("kalshi", f"candles/{market_ticker}", raw_rows)
    return pl.DataFrame(rows, schema=QUOTE_1M_SCHEMA)
