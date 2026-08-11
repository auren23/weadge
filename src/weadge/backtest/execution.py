"""Executable-quote execution model.

The core anti-leakage rule: a signal decided at bar T may only fill at the
OPEN of bar T+delay (delay >= 1). Same-bar fills are forbidden, because a
1m candle's close was not executable when the decision was made.

Fill prices come from real bid/ask OHLC (ask_open for buys, bid_open for
sells), never from mid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from weadge.domain.time import ensure_utc


@dataclass(frozen=True)
class Fill:
    market_ticker: str
    signal_at: datetime
    execute_at: datetime | None
    price: float | None
    side: str  # "buy_yes" | "sell_no"

    @property
    def filled(self) -> bool:
        return self.price is not None


def execute_delayed(
    quotes: pl.DataFrame,
    market_ticker: str,
    signal_at: datetime,
    side: str = "buy_yes",
    delay_min: int = 1,
    ts_col: str = "ts",
) -> Fill:
    """First executable quote strictly after signal_at + delay.

    buy_yes  -> yes_ask_open   (you pay the ask)
    sell_no  -> yes_bid_open   (you receive the bid)
    """
    signal_at = ensure_utc(signal_at)
    target = signal_at + timedelta(minutes=delay_min)
    sub = (
        quotes.filter(pl.col("market_ticker") == market_ticker)
        .filter(pl.col(ts_col) >= target)
        .sort(ts_col)
    )
    if sub.is_empty():
        return Fill(market_ticker, signal_at, None, None, side)
    row = sub.head(1).row(0, named=True)
    price = row["yes_ask_open"] if side == "buy_yes" else row["yes_bid_open"]
    return Fill(market_ticker, signal_at, row[ts_col], price, side)
