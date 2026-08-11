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
            {
                "event_ticker": EVENT,
                "series_ticker": SERIES,
                "target_date": datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
                "location_id": SERIES,
            },
        ]
    )


def _markets() -> pl.DataFrame:
    # Realistic Kalshi ladder: B-buckets closed [floor, cap], T-buckets
    # strict; the integer sets {<=87},{88,89},{90,91},{>=92} tile.
    rows = []
    for ticker, lo, hi, result in [
        (f"{EVENT}-M1", None, 88.0, "no"),  # T88: less than 88
        (f"{EVENT}-M2", 88.0, 89.0, "no"),  # B88.5: between 88-89
        (f"{EVENT}-M3", 90.0, 91.0, "yes"),  # B90.5: between 90-91
        (f"{EVENT}-M4", 91.0, None, "no"),  # T91: greater than 91
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
    for i, (bid, ask) in enumerate([(0.10, 0.15), (0.20, 0.25), (0.70, 0.75), (0.10, 0.15)]):
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
        assert df.height == 4 * len(SNAPSHOTS)  # 4 markets x 5 snapshots
        assert sorted(df["lead_hours"].to_list()) == sorted([24.0, 12.0, 6.0, 3.0, 1.0] * 4)

    def test_result_column(self, builder) -> None:
        df = builder.build()
        by_market = (
            df.group_by("market_ticker")
            .first()
            .select("market_ticker", "result")
            .sort("market_ticker")
        )
        assert by_market["result"].to_list() == [0, 0, 1, 0]

    def test_market_probability_from_quote(self, builder) -> None:
        df = builder.build()
        row = (
            df.filter(pl.col("market_ticker") == f"{EVENT}-M3")
            .filter(pl.col("lead_hours") == 24.0)
            .row(0, named=True)
        )
        # at close-24h the latest COMPLETED bar is close-24h-1m (a bar whose
        # ts == decision_at is still open and its close is not knowable)
        start = shift(CLOSE, hours=-26)
        completed = shift(CLOSE, hours=-24, minutes=-1)
        i = int((completed - start).total_seconds() // 60)
        drift = i / (26 * 60) * 0.1
        expected_mid = (0.70 + drift + 0.75 + drift) / 2.0
        assert row["market_mid"] == pytest.approx(expected_mid, abs=1e-9)
        assert row["p_market_raw"] == pytest.approx(expected_mid, abs=1e-9)

    def test_no_leak_bait_used(self, builder) -> None:
        """The forecast available only after close must never appear."""
        df = builder.build()
        assert not (df["p_nbm"] > 0.99).any()  # leak bait mean=99 -> p(bucket)~1

    def test_latest_knowable_forecast_per_snapshot(self, builder) -> None:
        df = builder.build()
        # early run (mean 88) is the only one knowable at close-24h:
        #   N(88,2) -> P(90<=X<92) ~ 0.14
        # mid run (mean 91) is knowable at close-1h:
        #   N(91,2) -> P(90<=X<92) ~ 0.38
        early_row = (
            df.filter(pl.col("market_ticker") == f"{EVENT}-M3")
            .filter(pl.col("lead_hours") == 24.0)
            .row(0, named=True)
        )
        late_row = (
            df.filter(pl.col("market_ticker") == f"{EVENT}-M3")
            .filter(pl.col("lead_hours") == 1.0)
            .row(0, named=True)
        )
        assert late_row["p_nbm"] > early_row["p_nbm"] + 0.1

    def test_bucket_sum_guard_rejects_partial_partition(self) -> None:
        """A market ladder missing a bucket must fail the sum~1 guard."""
        markets = _markets().filter(pl.col("floor_strike").is_not_null())  # drop lower tail
        fee = FeeSchedule([(datetime(2026, 6, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        broken = AlphaDatasetBuilder(
            events=_events(),
            markets=markets,
            quotes=_quotes(),
            forecast_percentiles=_percentiles(),
            forecasts=_forecasts(),
            fee_schedule=fee,
            snapshots_lead_hours=SNAPSHOTS,
        )
        with pytest.raises(ValueError, match="sum to 1"):
            broken.build()

    def test_nbm_priority_is_mean_std_then_percentiles(self, builder) -> None:
        """With mean/std present the Normal baseline wins; percentiles are
        only a fallback — never an override."""
        df = builder.build()
        row = (
            df.filter(pl.col("market_ticker") == f"{EVENT}-M3")
            .filter(pl.col("lead_hours") == 24.0)
            .row(0, named=True)
        )
        from scipy import stats as _stats

        # M3 = [90, 91] over reported whole degrees; the continuous forecast
        # is rounded the same way the DCR reports: [89.5, 91.5)
        expected = _stats.norm.cdf(91.5, 88.0, 2.0) - _stats.norm.cdf(89.5, 88.0, 2.0)
        assert row["p_nbm"] == pytest.approx(expected, abs=1e-9)

    def test_fee_from_schedule_at_decision_time(self, builder) -> None:
        """Fee = round_up_to_cent(M * 0.07 * P * (1-P)), NOT price * M."""
        from weadge.backtest.fees import round_up_to_cent

        df = builder.build()
        row = (
            df.filter(pl.col("market_ticker") == f"{EVENT}-M2")
            .filter(pl.col("lead_hours") == 3.0)
            .row(0, named=True)
        )
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

    def test_market_baselines_present(self, builder) -> None:
        """Phase B fills partition-level baselines. The default fixture's
        books (bid base sum 1.10 > 1, before drift) are a REAL infeasible
        partition: the finding must be preserved, not smoothed away."""
        df = builder.build()
        assert "p_market_raw" in df.columns
        assert "p_market_normalized" in df.columns
        assert "p_market_simplex" in df.columns
        # raw is per-bucket mid; normalized exists (all buckets quoted) and
        # each (event, snapshot) partition sums to 1
        assert df["p_market_normalized"].null_count() == 0
        n_partitions = df.select(["series_ticker", "event_ticker", "decision_at"]).unique().height
        assert df["p_market_normalized"].to_numpy().sum() == pytest.approx(n_partitions, abs=1e-9)
        # infeasible books: bid floor alone exceeds 1 -> simplex fails closed
        assert df["market_simplex_feasible"].to_list() == [False] * df.height
        assert df["p_market_simplex"].null_count() == df.height
        assert (df["market_bid_sum"] > 1.0).all()
        # the diagnostics eyeball still holds: bid_sum <= prob_sum <= ask_sum
        assert (df["market_bid_sum"] <= df["market_prob_sum_raw"]).all()
        assert (df["market_prob_sum_raw"] <= df["market_ask_sum"]).all()

    def test_market_baselines_feasible_partition(self) -> None:
        """With books satisfying bid_sum <= 1 <= ask_sum, the simplex
        projection exists: sum q = 1 and bid <= q <= ask for every row.
        Quote bases are chosen so the fixture's price drift keeps EVERY
        partition feasible: bid sum 0.53 + 4*0.0961 <= 1 and
        ask sum 0.98 + 4*0.0076 >= 1 across all five snapshots."""
        frames = []
        for i, (bid, ask) in enumerate([(0.10, 0.22), (0.12, 0.24), (0.15, 0.25), (0.16, 0.27)]):
            frames.append(
                make_quotes(
                    f"{EVENT}-M{i + 1}",
                    shift(CLOSE, hours=-26),
                    26 * 60,
                    bid_base=bid,
                    ask_base=ask,
                )
            )
        quotes = pl.concat(frames)
        fee = FeeSchedule([(datetime(2026, 6, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        b = AlphaDatasetBuilder(
            events=_events(),
            markets=_markets(),
            quotes=quotes,
            forecast_percentiles=_percentiles(),
            forecasts=_forecasts(),
            fee_schedule=fee,
            snapshots_lead_hours=SNAPSHOTS,
        )
        df = b.build()
        assert df["market_simplex_feasible"].to_list() == [True] * df.height
        by_partition = df.group_by(["series_ticker", "event_ticker", "decision_at"]).agg(
            pl.col("p_market_simplex").sum().alias("simplex_sum"),
            pl.col("p_market_normalized").sum().alias("norm_sum"),
        )
        assert (by_partition["simplex_sum"] - 1.0).abs().max() < 1e-9
        assert (by_partition["norm_sum"] - 1.0).abs().max() < 1e-9
        assert (df["p_market_simplex"] >= df["market_bid"] - 1e-12).all()
        assert (df["p_market_simplex"] <= df["market_ask"] + 1e-12).all()

    def test_missing_market_quote_fails_closed(self) -> None:
        """One bucket with no quote -> normalized/simplex NULL for the whole
        (event, snapshot), even though other buckets are fully quoted."""
        quotes = _quotes().filter(pl.col("market_ticker") != f"{EVENT}-M3")
        fee = FeeSchedule([(datetime(2026, 6, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        b = AlphaDatasetBuilder(
            events=_events(),
            markets=_markets(),
            quotes=quotes,
            forecast_percentiles=_percentiles(),
            forecasts=_forecasts(),
            fee_schedule=fee,
            snapshots_lead_hours=SNAPSHOTS,
        )
        df = b.build()
        assert df["market_mid"].null_count() == df.height // 4  # only M3 rows lack quotes
        assert df["p_market_normalized"].null_count() == df.height  # whole partition fails closed
        assert df["p_market_simplex"].null_count() == df.height
        assert df["market_simplex_feasible"].to_list() == [False] * df.height
        assert df["market_prob_sum_raw"].null_count() == df.height
        # the accounting sees exactly the missing M3 cells (4 markets x 5 snapshots)
        s = b.drop_stats
        assert s["cells_total"] == 4 * 5 and s["rows_built"] == 4 * 5
        assert s["missing_market_quote"] == 5
        assert s["market_partition_incomplete"] == 5  # every partition fails closed
        assert s["simplex_infeasible"] == 5

    def test_drop_stats_accounting(self) -> None:
        """Quotes/percentiles/early NBM all start at close-20h: the 24h
        snapshot has nothing knowable -> 4 rows dropped, reasons tallied."""
        quotes = _quotes().filter(pl.col("ts") >= shift(CLOSE, hours=-20))
        pcts = make_percentile_history(EVENT, periods=1, start=shift(CLOSE, hours=-20), center=90.0)
        # only the mid run (available close-2h) and leak bait (close+1h) remain
        forecasts = _forecasts().filter(pl.col("available_at") >= shift(CLOSE, hours=-2))
        fee = FeeSchedule([(datetime(2026, 6, 1, 0, 0, tzinfo=UTC), "taker", 1.0)])
        b = AlphaDatasetBuilder(
            events=_events(),
            markets=_markets(),
            quotes=quotes,
            forecast_percentiles=pcts,
            forecasts=forecasts,
            fee_schedule=fee,
            snapshots_lead_hours=SNAPSHOTS,
        )
        b.build()
        s = b.drop_stats
        assert s["cells_total"] == 4 * 5
        assert s["rows_built"] == 16 and s["rows_dropped"] == 4
        assert s["missing_market_quote"] == 4  # 24h cells only
        assert s["missing_kalshi_forecast"] == 4  # 24h cells only
        assert s["missing_nbm"] == 16  # 24/12/6/3h cells
        assert s["market_partition_incomplete"] == 0  # quotes exist at 12/6/3/1h
        assert s["simplex_infeasible"] == 4  # fixture books sum > 1
