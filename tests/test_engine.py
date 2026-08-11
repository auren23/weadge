"""Taker backtest engine: candle availability semantics + honest fills.

The anti-leakage contract under test:

    open of bar B  is knowable when  bar_start_at(B) <= decision_at
    close of bar B is knowable when  bar_end_at(B)   <= decision_at

The v0 strategy gates on the last COMPLETED bar's ask close and fills at
the next bar's ask open — deliberately one minute slower than the
theoretical earliest fill, never faster.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from conftest import make_quotes
from weadge.backtest.engine import run_taker_backtest
from weadge.backtest.fees import FeeSchedule

T0 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _fee_schedule() -> FeeSchedule:
    return FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])


def _signals(decision_at: datetime, p_model: float = 0.90, result: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_ticker": ["MKT1"],
            "decision_at": [decision_at],
            "p_model": [p_model],
            "result": [result],
        }
    )


class TestCompletedBarGate:
    def _quotes(self) -> pl.DataFrame:
        """Distinctive prices per bar so any off-by-one is visible:
        12:04 ask_close=0.41, 12:05 ask_open=0.50 / ask_close=0.99 (leak bait),
        12:06 ask_open=0.60 / ask_close=0.61."""
        rows = []
        for minute, (ask_open, ask_close) in {
            4: (0.40, 0.41),
            5: (0.50, 0.99),
            6: (0.60, 0.61),
            7: (0.70, 0.71),
        }.items():
            start = T0 + timedelta(minutes=minute)
            rows.append(
                {
                    "market_ticker": "MKT1",
                    "ts": start,
                    "bar_start_at": start,
                    "bar_end_at": start + timedelta(minutes=1),
                    "yes_bid_open": ask_open - 0.05,
                    "yes_bid_high": ask_open - 0.05,
                    "yes_bid_low": ask_open - 0.05,
                    "yes_bid_close": ask_open - 0.05,
                    "yes_ask_open": ask_open,
                    "yes_ask_high": ask_open,
                    "yes_ask_low": ask_open,
                    "yes_ask_close": ask_close,
                    "volume": 100,
                    "open_interest": 500,
                    "ingested_at": start + timedelta(minutes=1),
                }
            )
        return pl.DataFrame(rows)

    def test_gate_uses_last_completed_bar_close(self) -> None:
        """Decision at 12:05:00: the 12:05 bar is still open, so the gate is
        the 12:04 bar's ask CLOSE (0.41), never 12:05's open or close."""
        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=5)),
            self._quotes(),
            _fee_schedule(),
            threshold=0.06,
        )
        assert report.trades == 1
        row = report.trades_table.row(0, named=True)
        assert row["gross_edge"] == pytest.approx(0.90 - 0.41, abs=1e-9)
        assert row["fill_price"] == pytest.approx(0.60)  # 12:06 ask OPEN
        assert row["execute_at"] == T0 + timedelta(minutes=6)

    def test_bar_end_boundary_is_knowable(self) -> None:
        """Decision at 12:06:00 exactly: the 12:05 bar ENDED at 12:06:00, so
        its close (0.99) is now knowable — the gate moves to it (and a 0.06
        threshold on a 0.90 model correctly does NOT trade against 0.99)."""
        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=6)),
            self._quotes(),
            _fee_schedule(),
            threshold=0.06,
        )
        assert report.trades == 0  # gate 0.99 => gross -0.09, below threshold

        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=6), p_model=1.0),
            self._quotes(),
            _fee_schedule(),
            threshold=0.0,
        )
        row = report.trades_table.row(0, named=True)
        assert row["gross_edge"] == pytest.approx(1.0 - 0.99, abs=1e-9)
        assert row["execute_at"] == T0 + timedelta(minutes=7)

    def test_mid_bar_decision_still_uses_completed_bar(self) -> None:
        """Decision at 12:05:30: 12:05 bar not complete (ends 12:06:00); gate
        stays on 12:04 close; fill is the first bar starting at/after 12:06:30
        -> 12:07 open."""
        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=5, seconds=30)),
            self._quotes(),
            _fee_schedule(),
            threshold=0.06,
        )
        row = report.trades_table.row(0, named=True)
        assert row["gross_edge"] == pytest.approx(0.90 - 0.41, abs=1e-9)
        assert row["execute_at"] == T0 + timedelta(minutes=7)

    def test_no_trade_when_completed_bar_below_threshold(self) -> None:
        """p_model below the completed-bar ask must not trade even if the
        in-progress bar's open would have passed."""
        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=5), p_model=0.45),
            self._quotes(),
            _fee_schedule(),
            threshold=0.06,
        )
        assert report.trades == 0

    def test_legacy_frames_without_bar_columns_still_work(self) -> None:
        """Frames predating bar_start_at/bar_end_at derive bounds from ts."""
        quotes = make_quotes("MKT1", T0, 30, bid_base=0.40, ask_base=0.45).drop(
            ["bar_start_at", "bar_end_at"]
        )
        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=5), p_model=0.90),
            quotes,
            _fee_schedule(),
            threshold=0.06,
        )
        assert report.trades == 1
        assert report.trades_table["execute_at"][0] == T0 + timedelta(minutes=6)


class TestEdgeBins:
    """Predicted-edge binning must be exact: no digitize off-by-one, no
    dropped top tail, every trade in exactly one explicit interval."""

    def _flat_quotes(self) -> pl.DataFrame:
        """Constant ask 0.50 / bid 0.45 across 60 bars -> gross = p_model - 0.50."""
        rows = []
        for i in range(60):
            start = T0 + timedelta(minutes=i)
            rows.append(
                {
                    "market_ticker": "MKT1",
                    "ts": start,
                    "bar_start_at": start,
                    "bar_end_at": start + timedelta(minutes=1),
                    "yes_bid_open": 0.45,
                    "yes_bid_high": 0.45,
                    "yes_bid_low": 0.45,
                    "yes_bid_close": 0.45,
                    "yes_ask_open": 0.50,
                    "yes_ask_high": 0.50,
                    "yes_ask_low": 0.50,
                    "yes_ask_close": 0.50,
                    "volume": 100,
                    "open_interest": 500,
                    "ingested_at": start + timedelta(minutes=1),
                }
            )
        return pl.DataFrame(rows)

    def _run(self, signals: pl.DataFrame) -> pl.DataFrame:
        report = run_taker_backtest(
            signals, self._flat_quotes(), _fee_schedule(), threshold=0.0
        )
        return report.edge_bins

    def test_all_trades_land_in_exactly_one_bin(self) -> None:
        # gate ask ~0.50 -> gross = p_model - ~0.50 across every bin
        signals = pl.DataFrame(
            {
                "market_ticker": ["MKT1"] * 7,
                "decision_at": [T0 + timedelta(minutes=5 + i) for i in range(7)],
                "p_model": [0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.71],
                "result": [0, 0, 0, 1, 1, 1, 1],
            }
        )
        bins = self._run(signals)
        assert bins["edge_bin"].to_list() == ["<2%", "2-4%", "4-6%", "6-8%", "8-10%", "10%+"]
        assert bins["n"].to_list() == [1, 1, 1, 1, 1, 2]
        assert int(bins["n"].sum()) == 7  # no trade dropped, none double-counted

    def test_top_tail_not_dropped(self) -> None:
        """Regression: the old digitize mapping silently dropped every trade
        with gross >= 12% (the '12%+' label had no bin index)."""
        signals = pl.DataFrame(
            {
                "market_ticker": ["MKT1"] * 2,
                "decision_at": [T0 + timedelta(minutes=5), T0 + timedelta(minutes=6)],
                "p_model": [0.65, 0.99],  # gross ~0.15 and ~0.49
                "result": [1, 1],
            }
        )
        bins = self._run(signals)
        top = bins.filter(pl.col("edge_bin") == "10%+")
        assert len(top) == 1
        assert top["n"][0] == 2  # both high-edge trades present

    def test_realized_edge_is_monotone_in_predicted_edge(self) -> None:
        """The monotonicity evidence this table exists for: bins with higher
        predicted edge must show higher realized edge (wins sit in the top)."""
        signals = pl.DataFrame(
            {
                "market_ticker": ["MKT1"] * 6,
                "decision_at": [T0 + timedelta(minutes=5 + i) for i in range(6)],
                "p_model": [0.51, 0.55, 0.57, 0.59, 0.61, 0.71],
                "result": [0, 0, 1, 1, 1, 1],
            }
        )
        bins = self._run(signals)
        order = ["<2%", "2-4%", "4-6%", "6-8%", "8-10%", "10%+"]
        realized = []
        for label in order:
            rows = bins.filter(pl.col("edge_bin") == label)
            if len(rows):
                realized.append(rows["realized_edge"][0])
        assert realized == sorted(realized)  # non-decreasing across bins

    def test_below_threshold_trades_absent(self) -> None:
        signals = pl.DataFrame(
            {
                "market_ticker": ["MKT1"] * 2,
                "decision_at": [T0 + timedelta(minutes=5), T0 + timedelta(minutes=6)],
                "p_model": [0.55, 0.61],  # gross 0.05 and 0.11
                "result": [1, 1],
            }
        )
        report = run_taker_backtest(
            signals, self._flat_quotes(), _fee_schedule(), threshold=0.06
        )
        assert report.trades == 1
        assert report.edge_bins["edge_bin"].to_list() == ["10%+"]


class TestFeeIntegration:
    def test_fee_uses_kalshi_formula_at_execution_time(self) -> None:
        quotes = make_quotes("MKT1", T0, 30, bid_base=0.40, ask_base=0.45)
        report = run_taker_backtest(
            _signals(T0 + timedelta(minutes=5), p_model=0.90),
            quotes,
            _fee_schedule(),
            threshold=0.06,
        )
        from weadge.backtest.fees import round_up_to_cent

        row = report.trades_table.row(0, named=True)
        expected = round_up_to_cent(0.07 * row["fill_price"] * (1 - row["fill_price"]))
        assert row["fee"] == pytest.approx(expected, abs=1e-9)
