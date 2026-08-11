"""THE invariant: no information may be used before it was available.

Every test here enforces some form of as-of time. This file is the contract
that makes every downstream result honest.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from conftest import make_forecast_snapshots, make_percentile_history, make_quotes
from weadge.dataset.alignment import (
    latest_knowable,
    latest_quote_at_or_before,
    snapshot_forecast_percentiles,
)
from weadge.domain.forecast import Distribution, ForecastSnapshot, LookaheadError, validate_asof
from weadge.domain.time import shift


def fc(available_at: datetime, ingested_at: datetime | None = None) -> ForecastSnapshot:
    """Minimal valid forecast snapshot."""
    valid_start = available_at + timedelta(hours=6)
    return ForecastSnapshot(
        source="nbm",
        model="nbm_v5",
        run_init_at=available_at - timedelta(hours=1),
        available_at=available_at,
        ingested_at=ingested_at or (available_at + timedelta(minutes=1)),
        valid_start=valid_start,
        valid_end=valid_start + timedelta(hours=24),
        location_id="KXHIGHNY",
        distribution=Distribution(mean=90.0, std=2.0),
    )


class TestForecastValidation:
    def test_available_after_run_init(self) -> None:
        with pytest.raises(ValueError):
            ForecastSnapshot(
                source="nbm", model="nbm_v5",
                run_init_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
                available_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),  # before init
                ingested_at=datetime(2026, 7, 1, 11, 0, tzinfo=UTC),
                valid_start=datetime(2026, 7, 2, 0, 0, tzinfo=UTC),
                valid_end=datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
                location_id="KXHIGHNY",
            )

    def test_ingested_after_available(self) -> None:
        with pytest.raises(ValueError):
            ForecastSnapshot(
                source="nbm", model="nbm_v5",
                run_init_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
                available_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
                ingested_at=datetime(2026, 7, 1, 11, 0, tzinfo=UTC),  # before available
                valid_start=datetime(2026, 7, 2, 0, 0, tzinfo=UTC),
                valid_end=datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
                location_id="KXHIGHNY",
            )

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError):
            ForecastSnapshot(
                source="nbm", model="nbm_v5",
                run_init_at=datetime(2026, 7, 1, 10, 0),  # naive
                available_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
                ingested_at=datetime(2026, 7, 1, 12, 1, tzinfo=UTC),
                valid_start=datetime(2026, 7, 2, 0, 0, tzinfo=UTC),
                valid_end=datetime(2026, 7, 2, 12, 0, tzinfo=UTC),
                location_id="KXHIGHNY",
            )

    def test_valid_end_after_start(self) -> None:
        with pytest.raises(ValueError):
            ForecastSnapshot(
                source="nbm", model="nbm_v5",
                run_init_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
                available_at=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
                ingested_at=datetime(2026, 7, 1, 12, 1, tzinfo=UTC),
                valid_start=datetime(2026, 7, 2, 0, 0, tzinfo=UTC),
                valid_end=datetime(2026, 7, 2, 0, 0, tzinfo=UTC),  # == start
                location_id="KXHIGHNY",
            )


class TestAsOfGate:
    def test_knowable_after_available(self) -> None:
        f = fc(datetime(2026, 7, 1, 12, 0, tzinfo=UTC))
        assert f.is_knowable_at(datetime(2026, 7, 1, 12, 30, tzinfo=UTC))
        assert not f.is_knowable_at(datetime(2026, 7, 1, 11, 59, tzinfo=UTC))

    def test_validate_asof_raises_on_future(self) -> None:
        f = fc(datetime(2026, 7, 1, 12, 0, tzinfo=UTC))
        with pytest.raises(LookaheadError):
            validate_asof(datetime(2026, 7, 1, 11, 0, tzinfo=UTC), f)

    def test_run_init_is_not_the_gate(self) -> None:
        """run_init_at <= decision_at must NOT be sufficient — available_at is."""
        f = fc(datetime(2026, 7, 1, 12, 0, tzinfo=UTC))
        decision = datetime(2026, 7, 1, 11, 30, tzinfo=UTC)
        assert f.run_init_at <= decision          # naive gate would pass
        assert not f.is_knowable_at(decision)     # correct gate fails


class TestFrameAlignment:
    def test_latest_knowable_excludes_future(self, t0) -> None:
        df = make_forecast_snapshots("KXHIGHNY", t0, base_mean=88.0)
        # add a second, newer run that becomes available later
        late = make_forecast_snapshots("KXHIGHNY", t0, base_mean=95.0)
        late = late.with_columns(pl.col("available_at").map_batches(lambda s: s + timedelta(hours=2)))
        both = pl.concat([df, late])
        decision = shift(t0, hours=-30 + 0.25)  # before the late run is available
        known = latest_knowable(both, decision, key_cols=["location_id", "valid_start", "source"])
        assert known.height == 1
        assert known["mean"][0] == 88.0  # NOT the 95.0 leak

    def test_latest_knowable_empty_before_any_available(self, t0) -> None:
        df = make_forecast_snapshots("KXHIGHNY", t0)
        decision = shift(t0, hours=-48)
        assert latest_knowable(df, decision, key_cols=["location_id", "valid_start", "source"]).is_empty()

    def test_latest_quote_at_or_before(self, t0) -> None:
        quotes = make_quotes("MKT1", t0, 10)
        decision = shift(t0, minutes=3, seconds=30)
        got = latest_quote_at_or_before(quotes, decision)
        assert got["ts"][0] == shift(t0, minutes=3)

    def test_snapshot_percentiles_picks_latest_period(self, t0) -> None:
        hist = make_percentile_history(
            "EVENT1", periods=3, period_step_min=120, start=shift(t0, hours=-3)
        )
        decision = shift(t0, minutes=1)  # periods at t0-3h, t0-1h available; t0+1h not yet
        snap = snapshot_forecast_percentiles(hist, decision)
        # 5 percentiles for the single event, all from the LATEST knowable period
        assert snap.height == 5
        assert snap["end_period_ts"].n_unique() == 1
        assert snap["end_period_ts"][0] == shift(t0, hours=-1)
