"""Fee replay: the multiplier in effect at execution time is DATA, not a constant."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from weadge.backtest.fees import FeeSchedule, series_fee_schedule


def _changes() -> list[tuple[datetime, float]]:
    return [
        (datetime(2026, 1, 1, 0, 0, tzinfo=UTC), 0.10),
        (datetime(2026, 6, 1, 0, 0, tzinfo=UTC), 0.07),
        (datetime(2026, 9, 1, 0, 0, tzinfo=UTC), 0.05),
    ]


class TestFeeSchedule:
    def test_multiplier_at_steps(self) -> None:
        sched = FeeSchedule(_changes(), fallback_multiplier=None)
        assert sched.multiplier_at(datetime(2026, 1, 15, 0, 0, tzinfo=UTC)) == pytest.approx(0.10)
        assert sched.multiplier_at(datetime(2026, 6, 15, 0, 0, tzinfo=UTC)) == pytest.approx(0.07)
        assert sched.multiplier_at(datetime(2026, 9, 15, 0, 0, tzinfo=UTC)) == pytest.approx(0.05)

    def test_before_first_change_uses_fallback(self) -> None:
        sched = FeeSchedule(_changes(), fallback_multiplier=0.07)
        assert sched.multiplier_at(datetime(2025, 12, 1, 0, 0, tzinfo=UTC)) == pytest.approx(0.07)

    def test_exact_change_time_applies(self) -> None:
        """A change at T is in effect at T (effective_at <= ts)."""
        sched = FeeSchedule(_changes())
        assert sched.multiplier_at(datetime(2026, 6, 1, 0, 0, tzinfo=UTC)) == pytest.approx(0.07)

    def test_no_multiplier_raises(self) -> None:
        sched = FeeSchedule([])  # no fallback, no changes
        with pytest.raises(ValueError):
            sched.fee_cost(0.5, datetime(2026, 6, 1, 0, 0, tzinfo=UTC))

    def test_fee_cost_is_price_times_multiplier(self) -> None:
        sched = FeeSchedule(_changes())
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        assert sched.fee_cost(0.50, ts) == pytest.approx(0.50 * 0.07)

    def test_from_frame_roundtrip(self) -> None:
        frame = pl.DataFrame(
            [
                {"series_ticker": "KXHIGHNY", "effective_at": t, "fee_type": "taker",
                 "fee_multiplier": m, "ingested_at": t}
                for t, m in _changes()
            ]
        )
        sched = FeeSchedule.from_frame(frame, fallback_multiplier=0.07)
        assert sched.multiplier_at(datetime(2026, 6, 15, 0, 0, tzinfo=UTC)) == pytest.approx(0.07)


class TestSeriesFeeSchedule:
    def test_history_wins_over_metadata_fallback(self) -> None:
        frame = pl.DataFrame(
            [
                {"series_ticker": "KXHIGHNY", "effective_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                 "fee_type": "taker", "fee_multiplier": 0.05, "ingested_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC)},
            ]
        )
        sched = series_fee_schedule({"fee_multiplier": 0.07}, frame)
        assert sched.multiplier_at(datetime(2026, 6, 1, 0, 0, tzinfo=UTC)) == pytest.approx(0.05)

    def test_metadata_only(self) -> None:
        sched = series_fee_schedule({"fee_multiplier": 0.07}, None)
        assert sched.multiplier_at(datetime(2026, 6, 1, 0, 0, tzinfo=UTC)) == pytest.approx(0.07)
