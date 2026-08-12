"""Alpha dataset builder: bronze tables -> gold/alpha_dataset.parquet.

Each row is one (event, market, snapshot): a fixed lead-time decision point,
never every minute. This is the statistical guardrail — minute-level rows for
the same event are not independent samples, and the dataset must not pretend
otherwise.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import polars as pl

from weadge.backtest.fees import FeeSchedule, series_fee_schedule
from weadge.dataset.alignment import latest_completed_quote_at_or_before
from weadge.dataset.market_probability import add_market_baselines
from weadge.dataset.probability import (
    kalshi_forecast_probability,
    market_probability_from_quote,
    nbm_bucket_probability,
)
from weadge.domain.time import ensure_utc, shift
from weadge.storage.parquet import DataLake
from weadge.storage.schema import ALPHA_DATASET_SCHEMA, empty_frame


class AlphaDatasetBuilder:
    """Assemble the research dataset from bronze/silver frames.

    Frames may be passed in directly (tests) or read from the lake (CLI).
    """

    def __init__(
        self,
        events: pl.DataFrame,
        markets: pl.DataFrame,
        quotes: pl.DataFrame,
        forecast_percentiles: pl.DataFrame,
        forecasts: pl.DataFrame,
        fee_schedule: FeeSchedule,
        snapshots_lead_hours: list[int] | tuple[int, ...] = (24, 12, 6, 3, 1),
    ) -> None:
        self.events = events
        self.markets = markets
        self.quotes = quotes
        self.forecast_percentiles = forecast_percentiles
        self.forecasts = forecasts
        self.fee_schedule = fee_schedule
        self.snapshots = sorted(snapshots_lead_hours)
        # honest accounting: why the theoretical-max cells are missing
        self.drop_stats: dict[str, int] = {}
        self._cell_missing: list[dict[str, bool]] = []
        self._unsettled = 0

    # ---------------------------------------------------------------- build
    def build(self) -> pl.DataFrame:
        ev_map = {str(r["event_ticker"]): r for r in self.events.iter_rows(named=True)}
        event_dates = {t: r.get("target_date") for t, r in ev_map.items()}

        # pre-compute mid per quote row
        quotes = self.quotes.with_columns(
            ((pl.col("yes_bid_close") + pl.col("yes_ask_close")) / 2.0).alias("mid_close")
        )

        n_cells = 0
        rows: list[dict] = []
        for m in self.markets.iter_rows(named=True):
            if m.get("result") not in ("yes", "no"):
                # unsettled / void / cancelled market: no settlement truth.
                # Scoring it as a loss (result=0) would poison the dataset.
                self._unsettled += 1
                continue
            ev = str(m["event_ticker"])
            close_at = m.get("close_at")
            target = event_dates.get(ev)
            if close_at is None or target is None:
                continue
            close_at = ensure_utc(close_at)
            low = m.get("floor_strike")
            high = m.get("cap_strike")
            n_cells += len(self.snapshots)
            for lead_h in self.snapshots:
                decision_at = shift(close_at, hours=-lead_h)
                row = self._build_row(
                    m, ev, target, close_at, decision_at, lead_h, low, high, quotes
                )
                if row is not None:
                    rows.append(row)

        if not rows:
            self._finalize_stats(n_cells, 0, pl.DataFrame(schema=ALPHA_DATASET_SCHEMA))
            return empty_frame("alpha_dataset")
        df = pl.DataFrame(rows, schema=ALPHA_DATASET_SCHEMA)
        # Phase B: partition-level market baselines (normalized / simplex /
        # partition diagnostics) computed per (series_ticker, event_ticker,
        # decision_at) — never per row.
        df = add_market_baselines(df)
        df = df.sort(["event_date", "market_ticker", "decision_at"])
        self._assert_bucket_distributions(df)
        self._finalize_stats(n_cells, len(rows), df)
        return df

    def _finalize_stats(self, n_cells: int, n_rows: int, df: pl.DataFrame) -> None:
        """Why the gold frame is smaller than the theoretical max."""
        missing = Counter()
        for cell in self._cell_missing:
            for k, v in cell.items():
                missing[k] += int(v)
        parts = df.group_by(["series_ticker", "event_ticker", "decision_at"]).agg(
            pl.col("p_market_normalized").is_null().all().alias("norm_incomplete"),
            pl.col("market_simplex_feasible").eq(False).all().alias("simplex_infeasible"),
        )
        self.drop_stats = {
            "cells_total": n_cells,
            "rows_built": n_rows,
            "rows_dropped": n_cells - n_rows,
            "missing_market_quote": missing["missing_market_quote"],
            "missing_nbm": missing["missing_nbm"],
            "missing_kalshi_forecast": missing["missing_kalshi_forecast"],
            "market_partition_incomplete": int(parts.filter("norm_incomplete").height),
            "simplex_infeasible": int(parts.filter("simplex_infeasible").height),
            "unsettled_market": self._unsettled,
        }

    def _assert_bucket_distributions(self, df: pl.DataFrame, tolerance: float = 1e-6) -> None:
        """Every (series, event, snapshot) partition of p_nbm must sum to ~1.

        The markets of one KXHIGH event are mutually exclusive buckets of the
        same outcome; if their model probabilities do not form a distribution,
        every downstream comparison (including vs raw market mids) is biased.
        Keyed by event_ticker (not event_date) so different series on the
        same day can never collide.
        """
        sums = (
            df.filter(pl.col("p_nbm").is_not_null())
            .group_by(["series_ticker", "event_ticker", "decision_at"])
            .agg(pl.col("p_nbm").sum().alias("sum_p"))
            .filter((pl.col("sum_p") - 1.0).abs() > tolerance)
        )
        if not sums.is_empty():
            offenders = sums.head(5).to_dicts()
            raise ValueError(
                "p_nbm bucket probabilities must sum to 1 per (series, event, snapshot), "
                f"got sums {offenders} (tolerance {tolerance})"
            )

    # ----------------------------------------------------------------- row
    def _build_row(
        self,
        market: dict,
        ev: str,
        target: datetime,
        close_at: datetime,
        decision_at: datetime,
        lead_h: float,
        low: float | None,
        high: float | None,
        quotes: pl.DataFrame,
    ) -> dict | None:
        quote = latest_completed_quote_at_or_before(quotes, decision_at).filter(
            pl.col("market_ticker") == market["market_ticker"]
        )
        if quote.is_empty():
            bid = ask = mid = None
        else:
            q = quote.row(0, named=True)
            bid, ask = q.get("yes_bid_close"), q.get("yes_ask_close")
            mid = q.get("mid_close")

        p_market_raw = market_probability_from_quote(quote)
        p_kf = kalshi_forecast_probability(self.forecast_percentiles, decision_at, ev, low, high)
        p_nbm = nbm_bucket_probability(
            self.forecasts, decision_at, str(market.get("series_ticker", "")), low, high
        )
        self._cell_missing.append(
            {
                "missing_market_quote": p_market_raw is None,
                "missing_kalshi_forecast": p_kf is None,
                "missing_nbm": p_nbm is None,
            }
        )
        if p_market_raw is None and p_kf is None and p_nbm is None:
            return None  # nothing knowable at this decision time — drop the row

        result = 1 if market.get("result") == "yes" else 0
        price_for_fee = ask if ask is not None else (mid if mid is not None else 0.5)
        try:
            fee = self.fee_schedule.fee_cost(float(price_for_fee), decision_at)
        except ValueError:
            fee = float("nan")

        return {
            "series_ticker": str(market.get("series_ticker", "")),
            "event_ticker": ev,
            "event_date": target,
            "city": str(market.get("series_ticker", "")),
            "decision_at": decision_at,
            "lead_hours": float(lead_h),
            "market_ticker": market["market_ticker"],
            "bucket_low": low,
            "bucket_high": high,
            "market_bid": bid,
            "market_ask": ask,
            "market_mid": mid,
            "p_market_raw": p_market_raw,
            "p_kalshi_forecast": p_kf,
            "p_nbm": p_nbm,
            "p_gefs": None,
            "p_emos": None,
            "p_stack": None,
            "result": result,
            "fee": fee,
            "spread": (ask - bid) if (ask is not None and bid is not None) else None,
        }


def build_from_lake(
    lake: DataLake,
    series_ticker: str,
    snapshots_lead_hours: list[int] | tuple[int, ...] = (24, 12, 6, 3, 1),
    *,
    fallback_fee_multiplier: float | None = 1.0,
    fallback_fee_type: str = "taker",
) -> pl.DataFrame:
    """Read bronze tables from the lake and build the gold dataset.

    `fallback_fee_multiplier` is the API multiplier M (usually 1.0), NOT a
    base rate: the formula is fee = M * base_rate * C * P * (1 - P).
    """
    events = lake.read("events").filter(pl.col("series_ticker") == series_ticker)
    markets = lake.read("markets").filter(pl.col("series_ticker") == series_ticker)
    quotes = lake.read("quote_1m")
    pcts = lake.read("forecast_percentiles")
    forecasts = lake.read("forecasts").filter(pl.col("location_id") == series_ticker)
    fee_schedule = series_fee_schedule(
        {"fee_multiplier": fallback_fee_multiplier, "fee_type": fallback_fee_type},
        lake.read("fee_changes"),
    )
    builder = AlphaDatasetBuilder(
        events=events,
        markets=markets,
        quotes=quotes,
        forecast_percentiles=pcts,
        forecasts=forecasts,
        fee_schedule=fee_schedule,
        snapshots_lead_hours=snapshots_lead_hours,
    )
    df = builder.build()
    return df, builder.drop_stats
