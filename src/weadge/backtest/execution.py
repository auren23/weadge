"""Executable-quote execution model.

The core anti-leakage rule: a signal decided at bar T may only fill at the
OPEN of bar T+delay (delay >= 1). Same-bar fills are forbidden, because a
1m candle's close was not executable when the decision was made.

Candle availability semantics (quotes carry bar_start_at / bar_end_at):

    open of bar B  is knowable when  bar_start_at(B) <= decision_at
    close of bar B is knowable when  bar_end_at(B)   <= decision_at
        (a bar's close only exists once the bar has completed)

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

    The fill is the OPEN of the first bar whose start is at/after
    signal_at + delay: a bar's open is knowable exactly at its start, so
    this is the earliest quote the signal could actually trade against.
    """
    signal_at = ensure_utc(signal_at)
    target = signal_at + timedelta(minutes=delay_min)
    start_col = "bar_start_at" if "bar_start_at" in quotes.columns else ts_col
    sub = (
        quotes.filter(pl.col("market_ticker") == market_ticker)
        .filter(pl.col(start_col) >= target)
        .sort(start_col)
    )
    if sub.is_empty():
        return Fill(market_ticker, signal_at, None, None, side)
    row = sub.head(1).row(0, named=True)
    price = row["yes_ask_open"] if side == "buy_yes" else row["yes_bid_open"]
    return Fill(market_ticker, signal_at, row[start_col], price, side)
