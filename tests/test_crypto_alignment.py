"""Spot look-ahead guard: p* at decision time t may only use spot bars
whose CLOSE already happened (bar label + 60 <= t for open-time labels).

The 2026-08-12 audit found the scratch study indexing the bar still in
progress at t — 60 seconds of future BTC — which manufactured the entire
"ALIVE" edge. These tests pin the leak-free semantics."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from weadge.research.crypto_executability import (
    derive_no_signals,
    last_knowable_spot_idx,
)

T = 1_770_000_000 * 60 // 60  # any minute-aligned epoch
BASE = T - 69 * 60


def test_last_knowable_spot_idx() -> None:
    s_ts = np.array([0, 60, 120, 180], dtype="int64")
    assert last_knowable_spot_idx(s_ts, 120) == 1  # label 60 closed AT 120
    assert last_knowable_spot_idx(s_ts, 119) == 0  # label 60 closes in the future
    assert last_knowable_spot_idx(s_ts, 180) == 2
    assert last_knowable_spot_idx(s_ts, 59) == -1  # nothing closed yet
    # the naive index at t=120 is the in-progress bar — one to the right
    assert int(np.searchsorted(s_ts, 120, side="right")) - 1 == 2


def _fixtures(bid_close: float) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """One market, strike 100. Spot sits at ~100 through the signal bar,
    then CRASHES to 50 inside the fill minute [T, T+60). Leak-free p* at T
    must be ~0.5 (S=K exactly at the last knowable close); the leaked
    variant would see S=50 -> p* ~ 0 -> a huge fake NO edge."""

    def dt(s: int) -> datetime:
        return datetime.fromtimestamp(s, tz=UTC)

    labels = [BASE + i * 60 for i in range(70)]  # open-time labels, last = T
    closes = [100.0 if i % 2 else 100.01 for i in range(70)]
    closes[68] = 100.0  # last knowable close at T == strike -> p* = 0.5
    closes[69] = 50.0  # the future crash the signal must NOT see
    spot = pl.DataFrame({"ts": labels, "close": closes})

    markets = pl.DataFrame(
        {
            "market_ticker": ["KXBTC15M-TEST"],
            "floor_strike": [100.0],
            "close_at": [dt(T + 300)],
            "result": ["no"],
        }
    )
    quotes = pl.DataFrame(
        {
            "market_ticker": ["KXBTC15M-TEST"] * 2,
            "bar_start_at": [dt(T - 60), dt(T)],
            "bar_end_at": [dt(T), dt(T + 60)],
            "yes_bid_open": [bid_close, 0.50],
            "yes_bid_close": [bid_close, 0.10],
        }
    )
    return markets, quotes, spot


def test_future_crash_does_not_create_a_signal() -> None:
    # leak-free edge = 0.52 - 0.5 = 0.02 < 5%; the leaked edge would be ~0.52
    signals = derive_no_signals(*_fixtures(bid_close=0.52))
    assert signals.is_empty()


def test_leak_free_signal_uses_last_completed_bar() -> None:
    # leak-free edge = 0.60 - 0.5 = 0.10 >= 5% -> exactly one signal
    signals = derive_no_signals(*_fixtures(bid_close=0.60))
    assert signals.height == 1
    row = signals.row(0, named=True)
    assert row["fill_bid"] == 0.50
    assert row["fill_at"] == datetime.fromtimestamp(T, tz=UTC)
    assert row["result"] == 0
