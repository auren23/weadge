"""Incremental alpha test: paired samples + clustered OOS inference.

The gate is NOT an in-sample IID p-value: it is the OOS delta log-loss
(M0 vs M2) with an event/date clustered bootstrap. All models are fit and
scored on the SAME rows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from weadge.research.edge import (
    IncrementalGateResult,
    fit_incremental,
    paired_incremental_gate,
)
from weadge.research.walk_forward import split_frame


def _synthetic(
    n: int = 600, seed: int = 0, weather_power: float = 0.0, n_events: int = 60
) -> pl.DataFrame:
    """Rows with p_market, p_nbm, result, clustered by event_date.

    Data generating process is the model under test (correctly specified):

        logit(P(Y=1)) = -0.2 + 1.2*logit(p_market) + g*logit(p_nbm)

    weather_power=0 -> g=0: the market already prices everything; weather
        is a noisy copy and M2 has nothing incremental to add (null).
    weather_power>0 -> g>0: weather carries information the market does
        not, so M2 is strictly better specified and must win OOS.

    Each event_date carries n // n_events rows (buckets x lead times) so
    the sample is genuinely clustered, as in the real dataset.
    """
    rng = np.random.default_rng(seed)
    market_p = rng.uniform(0.05, 0.95, n)
    weather_p = np.clip(market_p + rng.normal(0, 0.05, n), 0.01, 0.99)
    g = 1.0 if weather_power > 0 else 0.0
    z = -0.2 + 1.2 * _logit(market_p) + g * _logit(weather_p)
    p_y = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p_y).astype(float)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i % n_events) for i in range(n)]
    return pl.DataFrame(
        {
            "event_date": dates,
            "p_market": market_p,
            "p_nbm": weather_p,
            "result": y,
        }
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _split(df: pl.DataFrame):
    dates = sorted(df["event_date"].unique().to_list())
    return split_frame(df, dates[0], dates[len(dates) // 2])


class TestFitIncremental:
    def test_weather_power_zero_no_incremental(self) -> None:
        df = _synthetic(weather_power=0.0)
        train, test = _split(df)
        res = fit_incremental(train, test)
        gate = paired_incremental_gate(train, test)
        assert not gate.has_incremental_alpha
        assert [r.model for r in res] == ["M0", "M1", "M2"]

    def test_weather_power_high_detected(self) -> None:
        df = _synthetic(weather_power=1.0, seed=7)
        train, test = _split(df)
        gate = paired_incremental_gate(train, test)
        assert gate.delta_ll > 0
        assert gate.ci_lower > 0
        assert gate.has_incremental_alpha
        # gamma is reported but as auxiliary evidence
        assert gate.gamma is not None and gate.gamma > 0

    def test_three_models_returned_with_paired_rows(self) -> None:
        df = _synthetic(seed=3)
        train, test = _split(df)
        res = fit_incremental(train, test)
        assert [r.model for r in res] == ["M0", "M1", "M2"]
        assert all(r.test_n > 0 for r in res)
        # paired: identical sample sizes across models
        assert len({r.train_n for r in res}) == 1
        assert len({r.test_n for r in res}) == 1

    def test_missing_weather_drops_rows_from_ALL_models(self) -> None:
        """Regression: M0 used to keep rows M2 had to drop, comparing two
        different test sets. Now every model sees the same rows."""
        df = _synthetic(seed=5).with_columns(
            pl.when(pl.col("p_nbm") > 0.8).then(None).otherwise(pl.col("p_nbm")).alias("p_nbm")
        )
        train, test = _split(df)
        res = fit_incremental(train, test)
        assert all(r.test_n == res[0].test_n for r in res)
        assert all(r.train_n == res[0].train_n for r in res)
        # the common mask really did drop rows
        assert res[0].test_n < test.height
        assert res[0].train_n < train.height


class TestPairedGate:
    def test_cluster_count_and_test_n(self) -> None:
        df = _synthetic(seed=1, weather_power=0.5, n_events=40)
        train, test = _split(df)
        gate = paired_incremental_gate(train, test)
        test_dates = test["event_date"].unique().to_list()
        assert gate.n_clusters == len(test_dates)
        assert gate.test_n == test.height  # no missing features in synthetic data

    def test_ci_sandwiches_delta_ll(self) -> None:
        df = _synthetic(seed=2, weather_power=0.8)
        train, test = _split(df)
        gate = paired_incremental_gate(train, test)
        assert gate.ci_lower <= gate.delta_ll <= gate.ci_upper

    def test_single_cluster_fails_closed(self) -> None:
        """One cluster cannot support clustered inference — gate must fail."""
        df = _synthetic(seed=3, weather_power=1.0, n_events=2)
        # keep only one event date in the test half
        dates = sorted(df["event_date"].unique().to_list())
        train = df.filter(pl.col("event_date") < dates[1])
        test = df.filter(pl.col("event_date") == dates[1])
        if train.height < 10 or test.height == 0:
            return  # window too degenerate to be informative — skip
        gate = paired_incremental_gate(train, test)
        assert gate.n_clusters == 1
        assert not gate.has_incremental_alpha

    def test_too_few_train_rows_fails_closed(self) -> None:
        df = _synthetic(seed=4, weather_power=1.0, n=15, n_events=8)
        train, test = _split(df)
        gate = paired_incremental_gate(train, test)
        assert not gate.has_incremental_alpha  # degenerate window -> NaN gate

    def test_gate_requires_both_mean_and_ci(self) -> None:
        assert not IncrementalGateResult(0.01, -0.001, 0.02, None, None, 10, 3, 0).has_incremental_alpha
        assert IncrementalGateResult(0.01, 0.001, 0.02, None, None, 10, 3, 0).has_incremental_alpha
        assert not IncrementalGateResult(-0.01, 0.001, 0.02, None, None, 10, 3, 0).has_incremental_alpha
        assert not IncrementalGateResult(float("nan"), 0.001, 0.02, None, None, 10, 3, 0).has_incremental_alpha

    def test_gate_is_deterministic(self) -> None:
        df = _synthetic(seed=6, weather_power=0.6)
        train, test = _split(df)
        g1 = paired_incremental_gate(train, test, seed=42)
        g2 = paired_incremental_gate(train, test, seed=42)
        assert g1 == g2
