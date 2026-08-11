"""Execution model: no same-bar fills, delayed fills use real quotes."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from conftest import make_quotes
from weadge.backtest.execution import execute_delayed


@pytest.fixture
def quotes() -> pl.DataFrame:
    start = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    return make_quotes("MKT1", start, 30, bid_base=0.40, ask_base=0.45)


class TestDelayedExecution:
    def test_fills_one_bar_later_ask_open(self, quotes) -> None:
        signal = datetime(2026, 7, 1, 12, 5, tzinfo=UTC)
        fill = execute_delayed(quotes, "MKT1", signal, side="buy_yes", delay_min=1)
        assert fill.filled
        # signal at 12:05 bar -> fill at 12:06 bar's ask OPEN
        assert fill.execute_at == datetime(2026, 7, 1, 12, 6, tzinfo=UTC)
        row = quotes.filter(pl.col("ts") == fill.execute_at).row(0, named=True)
        assert fill.price == row["yes_ask_open"]

    def test_no_same_bar_fill(self, quotes) -> None:
        """A decision at 12:05 must NOT fill at 12:05's own ask."""
        signal = datetime(2026, 7, 1, 12, 5, tzinfo=UTC)
        fill = execute_delayed(quotes, "MKT1", signal, delay_min=1)
        assert fill.execute_at == datetime(2026, 7, 1, 12, 6, tzinfo=UTC)
        # delay_min=0 is forbidden by design — caller must pass >= 1
        assert fill.execute_at > signal

    def test_sell_no_uses_bid_open(self, quotes) -> None:
        signal = datetime(2026, 7, 1, 12, 5, tzinfo=UTC)
        fill = execute_delayed(quotes, "MKT1", signal, side="sell_no", delay_min=1)
        row = quotes.filter(pl.col("ts") == fill.execute_at).row(0, named=True)
        assert fill.price == row["yes_bid_open"]

    def test_no_quote_returns_unfilled(self, quotes) -> None:
        signal = datetime(2026, 7, 1, 12, 50, tzinfo=UTC)  # beyond last bar
        fill = execute_delayed(quotes, "MKT1", signal, delay_min=1)
        assert not fill.filled
        assert fill.price is None

    def test_delay_skips_bars_with_no_trade(self) -> None:
        """If the +1 bar is missing, the first bar at/after signal+delay wins."""
        sparse = make_quotes("MKT1", datetime(2026, 7, 1, 12, 0, tzinfo=UTC), 10)
        # drop the 12:06 bar to simulate a quiet minute
        sparse = sparse.filter(pl.col("ts") != datetime(2026, 7, 1, 12, 6, tzinfo=UTC))
        signal = datetime(2026, 7, 1, 12, 5, tzinfo=UTC)
        fill = execute_delayed(sparse, "MKT1", signal, delay_min=1)
        assert fill.execute_at == datetime(2026, 7, 1, 12, 7, tzinfo=UTC)
