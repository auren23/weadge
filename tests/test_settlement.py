"""Settlement oracle: reproduce Kalshi's result from official observations."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from weadge.dataset.settlement import (
    AuditReport,
    SettlementOracle,
    SettlementSpec,
    bucket_hit,
)


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"event_ticker": "KXHIGHNY-26JUL01", "series_ticker": "KXHIGHNY",
             "target_date": datetime(2026, 7, 1, 0, 0, tzinfo=UTC), "location_id": "KXHIGHNY"},
            {"event_ticker": "KXHIGHNY-26JUL02", "series_ticker": "KXHIGHNY",
             "target_date": datetime(2026, 7, 2, 0, 0, tzinfo=UTC), "location_id": "KXHIGHNY"},
        ]
    )


def _markets() -> pl.DataFrame:
    rows = []
    for ev, lo, hi, res in [
        ("KXHIGHNY-26JUL01", 88.0, 90.0, "no"),    # obs 91 -> miss
        ("KXHIGHNY-26JUL01", 90.0, 92.0, "yes"),   # obs 91 -> hit
        ("KXHIGHNY-26JUL01", 92.0, None, "no"),    # obs 91 -> miss (unbounded cap)
        ("KXHIGHNY-26JUL02", 90.0, 92.0, "no"),    # missing observation
    ]:
        rows.append(
            {"market_ticker": f"{ev}-{int(lo or 0)}-{int(hi or 999)}",
             "event_ticker": ev, "series_ticker": "KXHIGHNY",
             "floor_strike": lo, "cap_strike": hi, "result": res}
        )
    return pl.DataFrame(rows)


def _observations() -> pl.DataFrame:
    # 2026-07-01 23:59 EDT == 2026-07-02 03:59 UTC — observed just before the
    # local midnight boundary, so it still belongs to the 07-01 target date.
    return pl.DataFrame(
        [
            {"station_id": "KNYC", "observed_at": datetime(2026, 7, 2, 3, 59, tzinfo=UTC),
             "value": 91.0, "unit": "fahrenheit", "source": "NWS ASOS"},
        ]
    )


def _spec() -> SettlementSpec:
    return SettlementSpec(series="KXHIGHNY", station_id="KNYC", timezone="America/New_York")


class TestBucketHit:
    def test_half_open_intervals(self) -> None:
        assert bucket_hit(90.0, 90.0, 92.0)          # low inclusive
        assert bucket_hit(91.999, 90.0, 92.0)
        assert not bucket_hit(92.0, 90.0, 92.0)      # cap exclusive
        assert not bucket_hit(89.999, 90.0, 92.0)
        assert bucket_hit(93.0, 92.0, None)          # unbounded tail
        assert not bucket_hit(91.0, 92.0, None)


class TestAudit:
    def test_audit_counts(self) -> None:
        report = SettlementOracle(_spec(), _events(), _markets(), _observations()).audit()
        assert isinstance(report, AuditReport)
        assert report.events_checked == 1           # only 07-01 has obs
        assert report.markets_checked == 4
        assert report.matched == 3                  # 3 correct, 1 missing
        assert report.mismatched == 0
        assert report.missing == 1                  # 07-02 market, no obs
        assert not report.clean                     # missing blocks research

    def test_mismatch_detected(self) -> None:
        bad = _markets().with_columns(
            pl.when(pl.col("event_ticker") == "KXHIGHNY-26JUL01")
            .then(pl.lit("yes"))
            .otherwise(pl.col("result"))
            .alias("result")
        )
        report = SettlementOracle(_spec(), _events(), bad, _observations()).audit()
        assert report.mismatched >= 1

    def test_timezone_boundary(self) -> None:
        """Obs at 03:59 UTC (Jul 1 23:59 EDT) still belongs to Jul 1 local."""
        obs = pl.DataFrame(
            [
                {"station_id": "KNYC", "observed_at": datetime(2026, 7, 2, 3, 59, tzinfo=UTC),
                 "value": 91.0, "unit": "fahrenheit", "source": "NWS ASOS"},
            ]
        )
        report = SettlementOracle(_spec(), _events(), _markets(), obs).audit()
        assert report.events_checked == 1
