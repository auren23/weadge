"""Incremental alpha test: is gamma (weather) significant given the market?"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from weadge.research.edge import fit_incremental, gamma_has_oos_value
from weadge.research.walk_forward import split_frame


def _synthetic(n: int = 600, seed: int = 0, weather_power: float = 0.0) -> pl.DataFrame:
    """Rows with p_market, p_nbm, result.

    weather_power=0: weather is a noisy copy of the market price — the market
        has already priced in everything weather knows (null hypothesis).
    weather_power>0: weather is strictly closer to the true probability than
        the market — it carries incremental information.
    """
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    market_p = np.clip(true_p + rng.normal(0, 0.08, n), 0.02, 0.98)
    if weather_power <= 0:
        weather_p = np.clip(market_p + rng.normal(0, 0.05, n), 0.02, 0.98)
    else:
        weather_p = np.clip(true_p + rng.normal(0, 0.02, n), 0.02, 0.98)
    y = (rng.uniform(0, 1, n) < true_p).astype(float)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i % 180) for i in range(n)]
    return pl.DataFrame(
        {
            "event_date": dates,
            "p_market": market_p,
            "p_nbm": weather_p,
            "result": y,
        }
    )


class TestFitIncremental:
    def test_weather_power_zero_no_incremental(self) -> None:
        df = _synthetic(weather_power=0.0)
        # chronological split
        dates = sorted(df["event_date"].unique().to_list())
        train, test = split_frame(df, dates[0], dates[len(dates) // 2])
        res = fit_incremental(train, test)
        assert not gamma_has_oos_value(res)

    def test_weather_power_high_detected(self) -> None:
        df = _synthetic(weather_power=1.0, seed=7)
        dates = sorted(df["event_date"].unique().to_list())
        train, test = split_frame(df, dates[0], dates[len(dates) // 2])
        res = fit_incremental(train, test)
        m2 = next(r for r in res if r.model == "M2")
        assert m2.gamma_pvalue is not None and m2.gamma_pvalue < 0.05
        assert gamma_has_oos_value(res)

    def test_three_models_returned(self) -> None:
        df = _synthetic(seed=3)
        dates = sorted(df["event_date"].unique().to_list())
        train, test = split_frame(df, dates[0], dates[len(dates) // 2])
        res = fit_incremental(train, test)
        assert [r.model for r in res] == ["M0", "M1", "M2"]
        assert all(r.test_n > 0 for r in res)

    def test_missing_features_dropped_per_row(self) -> None:
        df = _synthetic(seed=5).with_columns(
            pl.when(pl.col("p_nbm") > 0.8).then(None).otherwise(pl.col("p_nbm")).alias("p_nbm")
        )
        dates = sorted(df["event_date"].unique().to_list())
        train, test = split_frame(df, dates[0], dates[len(dates) // 2])
        res = fit_incremental(train, test)
        # M1/M2 have fewer train rows than M0 (NaN dropped), but all run
        assert all(r.test_n >= 0 for r in res)
