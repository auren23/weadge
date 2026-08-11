"""Fee replay: the multiplier in effect at execution time is DATA, not a constant.

The formula is the official Kalshi schedule (effective 2026-07-07):

    Fee = round_up_to_cent( M * base_rate(fee_type) * C * P * (1 - P) )
    base_rate: taker 0.07, maker 0.0175 ; M = API fee_multiplier (usually 1)

Golden values below are taken directly from Kalshi's published fee schedule.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from weadge.backtest.fees import (
    FEE_FORMULA_VERSION,
    FeeChange,
    FeeSchedule,
    round_up_to_cent,
    series_fee_schedule,
)


def _changes() -> list[tuple[datetime, str, float]]:
    return [
        (datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0),
        (datetime(2026, 6, 1, 0, 0, tzinfo=UTC), "taker", 2.0),
        (datetime(2026, 9, 1, 0, 0, tzinfo=UTC), "taker", 0.5),
    ]


class TestKalshiFormula:
    """Official fee-schedule golden values (taker, M=1, C=100)."""

    GOLDEN = [(0.10, 0.63), (0.25, 1.32), (0.50, 1.75), (0.90, 0.63)]

    def test_official_golden_table(self) -> None:
        sched = FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        for price, expected in self.GOLDEN:
            assert sched.fee_cost(price, ts, contracts=100) == pytest.approx(expected, abs=1e-9), (
                f"P={price} C=100 must fee ${expected}, got {sched.fee_cost(price, ts, contracts=100)}"
            )

    def test_fee_scales_with_contracts(self) -> None:
        sched = FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        one = sched.fee_cost(0.50, ts, contracts=1)          # 1.75c -> $0.02
        ten = sched.fee_cost(0.50, ts, contracts=10)         # 17.5c -> $0.18
        assert one == pytest.approx(0.02)
        assert ten == pytest.approx(0.18)

    def test_maker_uses_half_rate(self) -> None:
        sched = FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "maker", 1.0)])
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        # 0.0175 * 100 * 0.5 * 0.5 = 0.4375 -> $0.44 (rounded up from 43.75c)
        assert sched.fee_cost(0.50, ts, contracts=100) == pytest.approx(0.44)

    def test_multiplier_m_scales_fee(self) -> None:
        sched = FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 2.0)])
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        # M=2: 2 * 0.07 * 1 * 0.5 * 0.5 = 0.035 -> $0.04
        assert sched.fee_cost(0.50, ts, contracts=1) == pytest.approx(0.04)

    def test_m_is_not_a_price_multiplier(self) -> None:
        """Regression: M must enter via P(1-P), never as a direct price factor."""
        sched = FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        assert sched.fee_cost(0.10, ts, contracts=1) == pytest.approx(0.01)  # 0.63c -> 1c
        assert sched.fee_cost(0.90, ts, contracts=1) == pytest.approx(0.01)  # symmetric

    def test_subpenny_price_is_supported(self) -> None:
        """Kalshi's fee doc trades at $0.055; sub-cent prices must not break fees."""
        sched = FeeSchedule([(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        ts = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        # 0.07 * 1 * 0.055 * 0.945 = 0.003638 -> 0.364c -> $0.01
        assert sched.fee_cost(0.055, ts, contracts=1) == pytest.approx(0.01)


class TestRounding:
    def test_round_up_to_cent(self) -> None:
        assert round_up_to_cent(1.3125) == pytest.approx(1.32)
        assert round_up_to_cent(0.6300) == pytest.approx(0.63)
        assert round_up_to_cent(0.0001) == pytest.approx(0.01)
        assert round_up_to_cent(0.0) == pytest.approx(0.0)

    def test_exact_cent_not_inflated_by_float_noise(self) -> None:
        """0.07*100*0.1*0.9 = 0.6300000000000001 in floats; must still be $0.63."""
        assert round_up_to_cent(0.07 * 100 * 0.1 * 0.9) == pytest.approx(0.63)
        assert round_up_to_cent(0.07 * 100 * 0.25 * 0.75) == pytest.approx(1.32)


class TestFeeSchedule:
    def test_change_at_steps(self) -> None:
        sched = FeeSchedule(_changes())
        assert sched.change_at(datetime(2026, 1, 15, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(1.0)
        assert sched.change_at(datetime(2026, 6, 15, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(2.0)
        assert sched.change_at(datetime(2026, 9, 15, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(0.5)

    def test_before_first_change_uses_fallback(self) -> None:
        sched = FeeSchedule(_changes(), fallback=FeeChange(datetime(1970, 1, 1, tzinfo=UTC), "taker", 1.0))
        assert sched.change_at(datetime(2025, 12, 1, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(1.0)

    def test_exact_change_time_applies(self) -> None:
        """A change at T is in effect at T (effective_at <= ts)."""
        sched = FeeSchedule(_changes())
        assert sched.change_at(datetime(2026, 6, 1, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(2.0)

    def test_fee_type_travels_with_change(self) -> None:
        sched = FeeSchedule(
            [
                (datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "taker", 1.0),
                (datetime(2026, 3, 1, 0, 0, tzinfo=UTC), "maker", 1.0),
            ]
        )
        assert sched.change_at(datetime(2026, 2, 1, 0, 0, tzinfo=UTC)).fee_type == "taker"
        assert sched.change_at(datetime(2026, 4, 1, 0, 0, tzinfo=UTC)).fee_type == "maker"

    def test_no_multiplier_raises(self) -> None:
        sched = FeeSchedule([])  # no fallback, no changes
        with pytest.raises(ValueError):
            sched.fee_cost(0.5, datetime(2026, 6, 1, 0, 0, tzinfo=UTC))

    def test_unknown_fee_type_rejected(self) -> None:
        with pytest.raises(ValueError):
            FeeChange(datetime(2026, 1, 1, 0, 0, tzinfo=UTC), "liquidity", 1.0)  # type: ignore[arg-type]

    def test_from_frame_roundtrip(self) -> None:
        frame = pl.DataFrame(
            [
                {"series_ticker": "KXHIGHNY", "effective_at": t, "fee_type": ft,
                 "fee_multiplier": m, "ingested_at": t}
                for t, ft, m in _changes()
            ]
        )
        sched = FeeSchedule.from_frame(frame, fallback_multiplier=1.0)
        assert sched.change_at(datetime(2026, 6, 15, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(2.0)
        assert sched.formula_version == FEE_FORMULA_VERSION


class TestSeriesFeeSchedule:
    def test_history_wins_over_metadata_fallback(self) -> None:
        frame = pl.DataFrame(
            [
                {"series_ticker": "KXHIGHNY", "effective_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                 "fee_type": "taker", "fee_multiplier": 0.5, "ingested_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC)},
            ]
        )
        sched = series_fee_schedule({"fee_multiplier": 1.0, "fee_type": "taker"}, frame)
        assert sched.change_at(datetime(2026, 6, 1, 0, 0, tzinfo=UTC)).multiplier == pytest.approx(0.5)

    def test_metadata_only(self) -> None:
        sched = series_fee_schedule({"fee_multiplier": 1.0, "fee_type": "taker"}, None)
        ts = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        assert sched.change_at(ts).multiplier == pytest.approx(1.0)
        # default M=1, taker: 0.07 * 1 * 1 * 0.5 * 0.5 = 1.75c -> $0.02
        assert sched.fee_cost(0.5, ts, contracts=1) == pytest.approx(0.02)

    def test_metadata_missing_multiplier_raises_on_use(self) -> None:
        sched = series_fee_schedule(None, None)
        with pytest.raises(ValueError):
            sched.fee_cost(0.5, datetime(2026, 6, 1, 0, 0, tzinfo=UTC))
