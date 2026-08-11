"""Alpha dataset builder integration: bronze frames -> gold rows, as-of clean."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from conftest import make_forecast_snapshots, make_percentile_history, make_quotes
from weadge.backtest.fees import FeeSchedule
from weadge.dataset.builder import AlphaDatasetBuilder
from weadge.domain.time import shift

CLOSE = datetime(2026, 7, 1, 23, 0, tzinfo=UTC)
SERIES = "KXHIGHNY"
EVENT = "KXHIGHNY-26JUL01"
SNAPSHOTS = (24, 12, 6, 3, 1)


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"event_ticker": EVENT, "series_ticker": SERIES,
             "target_date": datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
             "location_id": SERIES},
        ]
    )


def _markets() -> pl.DataFrame:
    rows = []
    for ticker, lo, hi, result in [
        (f"{EVENT}-M1", 88.0, 90.0, "no"),
        (f"{EVENT}-M2", 90.0, 92.0, "yes"),
        (f"{EVENT}-M3", 92.0, None, "no"),
    ]:
        rows.append(
            {
                "market_ticker": ticker,
                "event_ticker": EVENT,
                "series_ticker": SERIES,
                "floor_strike": lo,
                "cap_strike": hi,
                "open_at": shift(CLOSE, hours=-48),
                "close_at": CLOSE,
                "settled_at": shift(CLOSE, hours=6),
                "result": result,
                "settlement_value": result,
                "rules_primary": "",
                "rules_secondary": "",
                "ingested_at": CLOSE,
            }
        )
    return pl.DataFrame(rows)


def _quotes() -> pl.DataFrame:
    frames = []
    for i, (bid, ask) in enumerate([(0.20, 0.25), (0.70, 0.75), (0.10, 0.15)]):
        frames.append(
            make_quotes(
                f"{EVENT}-M{i + 1}",
                shift(CLOSE, hours=-26),
                26 * 60,
                bid_base=bid,
                ask_base=ask,
            )
        )
    return pl.concat(frames)


def _percentiles() -> pl.DataFrame:
    # one period every 2 hours from close-26h to close-1h
    frames = []
    for k in range(0, 26, 2):
        frames.append(
            make_percentile_history(
                EVENT, periods=1, start=shift(CLOSE, hours=-(26 - k)), center=90.0
            )
        )
    return pl.concat(frames)


def _forecasts() -> pl.DataFrame:
    """Three runs with different availability:

      early (mean 88)  available ~44h before close  -> used by 24h/12h/6h/3h snapshots
      mid   (mean 91)  available 2h before close    -> used by the 1h snapshot only
      late  (mean 99)  available 1h AFTER close     -> leak bait, never used
    """
    early = make_forecast_snapshots(SERIES, datetime(2026, 7, 1, 0, 0, tzinfo=UTC), base_mean=88.0)
    mid = make_forecast_snapshots(SERIES, datetime(2026, 7, 1, 0, 0, tzinfo=UTC), base_mean=91.0)
    mid = mid.with_columns(pl.lit(shift(CLOSE, hours=-2)).alias("available_at"))
    late = make_forecast_snapshots(SERIES, datetime(2026, 7, 1, 0, 0, tzinfo=UTC), base_mean=99.0)
    late = late.with_columns(pl.lit(shift(CLOSE, hours=1)).alias("available_at"))
    return pl.concat([early, mid, late])


@pytest.fixture
def builder() -> AlphaDatasetBuilder:
    fee = FeeSchedule([(datetime(2026, 6, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
    return AlphaDatasetBuilder(
        events=_events(),
        markets=_markets(),
        quotes=_quotes(),
        forecast_percentiles=_percentiles(),
        forecasts=_forecasts(),
        fee_schedule=fee,
        snapshots_lead_hours=SNAPSHOTS,
    )


class TestAlphaDataset:
    def test_shape(self, builder) -> None:
        df = builder.build()
        assert df.height == 3 * len(SNAPSHOTS)   # 3 markets x 5 snapshots
        assert sorted(df["lead_hours"].to_list()) == sorted([24.0, 12.0, 6.0, 3.0, 1.0] * 3)

    def test_result_column(self, builder) -> None:
        df = builder.build()
        by_market = df.group_by("market_ticker").first().select(
            "market_ticker", "result"
        ).sort("market_ticker")
        assert by_market["result"].to_list() == [0, 1, 0]

    def test_market_probability_from_quote(self, builder) -> None:
        df = builder.build()
        row = df.filter(pl.col("market_ticker") == f"{EVENT}-M2").filter(
            pl.col("lead_hours") == 24.0
        ).row(0, named=True)
        # at close-24h the quote drift = i/n with i = (close-24h - start)/1min
        start = shift(CLOSE, hours=-26)
        i = int((shift(CLOSE, hours=-24) - start).total_seconds() // 60)
        drift = i / (26 * 60) * 0.1
        expected_mid = (0.70 + drift + 0.75 + drift) / 2.0
        assert row["market_mid"] == pytest.approx(expected_mid, abs=1e-9)
        assert row["p_market"] == pytest.approx(expected_mid, abs=1e-9)

    def test_no_leak_bait_used(self, builder) -> None:
        """The forecast available only after close must never appear."""
        df = builder.build()
        assert not (df["p_nbm"] > 0.99).any()  # leak bait mean=99 -> p(bucket)~1

    def test_latest_knowable_forecast_per_snapshot(self, builder) -> None:
        df = builder.build()
        # early run (mean 88) is the only one knowable at close-24h:
        #   N(88,2) -> P(90<=X<92) ~ 0.16
        # mid run (mean 91) is knowable at close-1h:
        #   N(91,2) -> P(90<=X<92) ~ 0.38
        early_row = df.filter(pl.col("market_ticker") == f"{EVENT}-M2").filter(
            pl.col("lead_hours") == 24.0
        ).row(0, named=True)
        late_row = df.filter(pl.col("market_ticker") == f"{EVENT}-M2").filter(
            pl.col("lead_hours") == 1.0
        ).row(0, named=True)
        assert late_row["p_nbm"] > early_row["p_nbm"] + 0.1

    def test_fee_from_schedule_at_decision_time(self, builder) -> None:
        """Fee = round_up_to_cent(M * 0.07 * P * (1-P)), NOT price * M."""
        from weadge.backtest.fees import round_up_to_cent

        df = builder.build()
        row = df.filter(pl.col("market_ticker") == f"{EVENT}-M2").filter(
            pl.col("lead_hours") == 3.0
        ).row(0, named=True)
        expected = round_up_to_cent(0.07 * row["market_ask"] * (1 - row["market_ask"]))
        assert row["fee"] == pytest.approx(expected, abs=1e-9)
        assert row["fee"] < row["market_ask"] * 0.07  # P(1-P) shrinks the old price*M fee

    def test_spread(self, builder) -> None:
        df = builder.build()
        row = df.filter(pl.col("market_ticker") == f"{EVENT}-M2").head(1).row(0, named=True)
        assert row["spread"] == pytest.approx(row["market_ask"] - row["market_bid"], abs=1e-9)

    def test_all_decision_times_are_before_close(self, builder) -> None:
        df = builder.build()
        assert (df["decision_at"] < CLOSE).all()
