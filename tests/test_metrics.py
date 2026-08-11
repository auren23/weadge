"""Event-cluster bootstrap: the independence unit is the event/day, not the trade."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from weadge.backtest.metrics import (
    event_cluster_bootstrap,
    hit_rate,
    mean_pnl,
    total_pnl,
)


def _trades(n_events: int = 20, rows_per_event: int = 3, seed: int = 1) -> pl.DataFrame:
    """Trades grouped into events, each with a few correlated rows."""
    rng = np.random.default_rng(seed)
    dates = []
    pnls = []
    for e in range(n_events):
        # within-event correlation: same event-level shock
        shock = rng.normal(0.02, 0.1)
        for _r in range(rows_per_event):
            dates.append(f"2026-01-{e + 1:02d}")
            pnls.append(shock + rng.normal(0, 0.02))
    return pl.DataFrame({"event_date": dates, "pnl": pnls})


class TestEventClusterBootstrap:
    def test_point_equals_raw_metric(self) -> None:
        df = _trades()
        ci = event_cluster_bootstrap(df, mean_pnl, n_boot=100)
        assert ci.point == pytest.approx(mean_pnl(df))

    def test_ci_contains_point(self) -> None:
        df = _trades(seed=3)
        ci = event_cluster_bootstrap(df, mean_pnl, n_boot=200)
        assert ci.lower <= ci.point <= ci.upper

    def test_cluster_count_reported(self) -> None:
        df = _trades(n_events=17)
        ci = event_cluster_bootstrap(df, total_pnl, n_boot=50)
        assert ci.n_clusters == 17

    def test_per_trade_bootstrap_overstates_precision(self) -> None:
        """The whole point: per-trade resampling gives a falsely narrow CI.

        Cluster bootstrap must produce a WIDER (or equal) CI than a naive
        per-row bootstrap, because it preserves within-event dependence.
        """
        df = _trades(n_events=40, rows_per_event=5, seed=9)
        ci_cluster = event_cluster_bootstrap(df, mean_pnl, n_boot=500)

        pnls = df["pnl"].to_numpy()
        rng = np.random.default_rng(0)
        boot = np.array(
            [rng.choice(pnls, size=len(pnls), replace=True).mean() for _ in range(500)]
        )
        width_naive = np.quantile(boot, [0.025, 0.975])
        width_cluster = np.array([ci_cluster.lower, ci_cluster.upper])
        assert width_cluster[1] - width_cluster[0] >= width_naive[1] - width_naive[0] - 1e-9

    def test_hit_rate_bounded(self) -> None:
        df = _trades()
        assert 0.0 <= hit_rate(df) <= 1.0
