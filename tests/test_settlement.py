"""Settlement oracle: reproduce Kalshi's result from the official NWS
Daily Climate Report — never from hourly METAR observations."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from weadge.dataset.nws_daily_climate import DCR_SOURCE, daily_climate_frame
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
    # DCR record for 2026-07-01, stamped at the window midpoint (17:00 UTC).
    return daily_climate_frame([{"report_date": date(2026, 7, 1), "value": 91.0}])


def _spec() -> SettlementSpec:
    return SettlementSpec(series="KXHIGHNY", station_id="KNYC", timezone="America/New_York")


class TestBucketHit:
    def test_between_is_closed_interval(self) -> None:
        """Kalshi rules say 'between X-Y°' — both edges are inclusive."""
        assert bucket_hit(90.0, 90.0, 92.0)          # low edge included
        assert bucket_hit(91.0, 90.0, 92.0)
        assert bucket_hit(92.0, 90.0, 92.0)          # high edge included (was half-open)
        assert not bucket_hit(89.0, 90.0, 92.0)
        assert not bucket_hit(93.0, 90.0, 92.0)

    def test_less_than_is_strict(self) -> None:
        """T-bucket 'less than 92°': 92 itself is NOT in the bucket."""
        assert bucket_hit(91.0, None, 92.0)
        assert not bucket_hit(92.0, None, 92.0)
        assert not bucket_hit(93.0, None, 92.0)

    def test_greater_than_is_strict(self) -> None:
        """T-bucket 'greater than 99°': 99 itself is NOT in the bucket."""
        assert bucket_hit(100.0, 99.0, None)
        assert not bucket_hit(99.0, 99.0, None)
        assert not bucket_hit(98.0, 99.0, None)

    def test_partition_is_exact(self) -> None:
        """The KXHIGHNY ladder partitions all integers: T92 (<92), B92.5
        {92,93}, B94.5 {94,95}, ..., T99 (>99)."""
        for v in range(80, 105):
            hits = [
                bucket_hit(v, None, 92.0),
                bucket_hit(v, 92.0, 93.0),
                bucket_hit(v, 94.0, 95.0),
                bucket_hit(v, 96.0, 97.0),
                bucket_hit(v, 98.0, 99.0),
                bucket_hit(v, 99.0, None),
            ]
            assert sum(hits) == 1, f"value {v} hits {sum(hits)} buckets"


class TestSettlementDay:
    def test_standard_time_window(self) -> None:
        """[D 05:00 UTC, D+1 05:00 UTC) belongs to report date D."""
        spec = _spec()
        assert spec.settlement_day(datetime(2026, 12, 1, 5, 0, tzinfo=UTC)) == date(2026, 12, 1)
        assert spec.settlement_day(datetime(2026, 12, 1, 4, 59, tzinfo=UTC)) == date(2026, 11, 30)
        assert spec.settlement_day(datetime(2026, 12, 2, 4, 59, tzinfo=UTC)) == date(2026, 12, 1)

    def test_dst_midnight_hour_belongs_to_previous_report_day(self) -> None:
        """Regression: 00:30 EDT on Jul 2 (04:30 UTC) is still within Jul 1's
        standard-time window [Jul 1 05:00 UTC, Jul 2 05:00 UTC). The old
        DST-aware local calendar day placed it on Jul 2."""
        spec = _spec()
        assert spec.settlement_day(datetime(2026, 7, 2, 4, 30, tzinfo=UTC)) == date(2026, 7, 1)
        # 05:00 UTC exactly starts the next report day
        assert spec.settlement_day(datetime(2026, 7, 2, 5, 0, tzinfo=UTC)) == date(2026, 7, 2)


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

    def test_dst_window_assigns_obs_to_correct_report_day(self) -> None:
        """Obs at Jul 2 04:30 UTC (00:30 EDT Jul 2) settles Jul 1's market:
        the DCR standard-time window for Jul 1 runs until Jul 2 05:00 UTC."""
        events = pl.concat(
            [
                _events(),
                pl.DataFrame(
                    [
                        {"event_ticker": "KXHIGHNY-26JUL02", "series_ticker": "KXHIGHNY",
                         "target_date": datetime(2026, 7, 2, 0, 0, tzinfo=UTC),
                         "location_id": "KXHIGHNY"},
                    ]
                ),
            ]
        ).unique(subset=["event_ticker"])
        markets = pl.concat(
            [
                _markets(),
                pl.DataFrame(
                    [
                        {"market_ticker": "KXHIGHNY-26JUL02-88-90", "event_ticker": "KXHIGHNY-26JUL02",
                         "series_ticker": "KXHIGHNY", "floor_strike": 88.0, "cap_strike": 90.0,
                         "result": "no"},
                    ]
                ),
            ]
        )
        obs = pl.DataFrame(
            [
                {"station_id": "KNYC",
                 "observed_at": datetime(2026, 7, 2, 4, 30, tzinfo=UTC),
                 "value": 91.0, "unit": "fahrenheit", "source": DCR_SOURCE},
            ]
        )
        report = SettlementOracle(_spec(), events, markets, obs).audit()
        # Jul 1 market: 91 in [90,92) -> hit -> matches "yes"; Jul 2 market:
        # no observation (91 belongs to Jul 1's report) -> missing, not mismatched
        assert report.mismatched == 0
        assert report.matched >= 1
        assert report.missing >= 1


class TestSourceTrust:
    def test_metar_is_never_settlement_truth(self) -> None:
        """Hourly METAR rows must be ignored entirely: the audit reports
        missing markets (research frozen) instead of faking settlements."""
        obs = pl.DataFrame(
            [
                {"station_id": "KNYC",
                 "observed_at": datetime(2026, 7, 2, 3, 59, tzinfo=UTC),  # 23:59 EDT Jul 1
                 "value": 91.0, "unit": "fahrenheit", "source": "NWS ASOS"},
            ]
        )
        report = SettlementOracle(_spec(), _events(), _markets(), obs).audit()
        assert report.matched == 0
        assert report.missing == 4
        assert report.mismatched == 0
        assert not report.clean

    def test_foreign_station_ignored(self) -> None:
        obs = daily_climate_frame([{"report_date": date(2026, 7, 1), "value": 91.0}],
                                  station_id="KJFK")
        report = SettlementOracle(_spec(), _events(), _markets(), obs).audit()
        assert report.matched == 0
        assert report.missing == 4


class TestDailyClimateFrame:
    def test_roundtrip_to_report_date(self) -> None:
        """DCR records stamped at the window midpoint map back to the exact
        report date in both DST (July) and standard (December) seasons."""
        for report_date in (date(2026, 7, 1), date(2026, 12, 1)):
            obs = daily_climate_frame([{"report_date": report_date, "value": 91.0}])
            assert obs["source"][0] == DCR_SOURCE
            assert _spec().settlement_day(obs["observed_at"][0]) == report_date

    def test_audit_with_dcr_records(self) -> None:
        obs = daily_climate_frame(
            [{"report_date": date(2026, 7, 1), "value": 91.0},
             {"report_date": date(2026, 7, 2), "value": 84.0}]
        )
        report = SettlementOracle(_spec(), _events(), _markets(), obs).audit()
        assert report.matched == 4
        assert report.mismatched == 0
        assert report.missing == 0
        assert report.clean
