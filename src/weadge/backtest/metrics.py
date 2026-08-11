"""Statistical inference for backtest metrics.

Critical rule from the research plan: the independence unit is the EVENT/DAY,
never the trade, bucket, or minute-level snapshot. Multiple buckets of the
same day are mutually exclusive and share the same weather outcome; minute
snapshots of the same event are near-duplicates. Bootstrap accordingly.

This is the guardrail that prevents fake p < 0.001 results.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    n_clusters: int
    ci_level: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}] (n_clusters={self.n_clusters})"


def event_cluster_bootstrap(
    df: pl.DataFrame,
    metric: Callable[[pl.DataFrame], float],
    *,
    cluster_col: str = "event_date",
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap a metric by resampling event clusters with replacement.

    All rows sharing a cluster value are resampled together — this preserves
    the within-event dependence (same weather, mutually exclusive buckets,
    correlated snapshots).
    """
    clusters = df[cluster_col].unique().to_list()
    n_clusters = len(clusters)
    if n_clusters == 0:
        raise ValueError("no clusters to bootstrap")
    rng = np.random.default_rng(seed)

    point = metric(df)
    alpha = (1.0 - ci_level) / 2.0

    boot = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(clusters, size=n_clusters, replace=True)
        mask = df[cluster_col].is_in(picked)
        sub = df.filter(mask)
        boot[i] = metric(sub)
    lower, upper = np.quantile(boot, [alpha, 1.0 - alpha])
    return BootstrapCI(point=float(point), lower=float(lower), upper=float(upper),
                       n_clusters=n_clusters, ci_level=ci_level)


def mean_pnl(df: pl.DataFrame) -> float:
    return float(df["pnl"].to_numpy().mean())


def total_pnl(df: pl.DataFrame) -> float:
    return float(df["pnl"].to_numpy().sum())


def hit_rate(df: pl.DataFrame) -> float:
    if df.height == 0:
        return float("nan")
    return float(df["pnl"].gt(0).to_numpy().mean())
