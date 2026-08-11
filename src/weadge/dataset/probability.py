"""Probability features for the alpha dataset.

    p_market          — mid price of the latest 1m quote (clamped)
    p_kalshi_forecast — Kalshi's own percentile history, interpolated to the bucket
    p_nbm             — NBM distribution (normal or percentile) over the bucket

The Kalshi forecast is labelled `kalshi_forecast`, never `nbm` — provenance
is undocumented.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from weadge.dataset.alignment import snapshot_forecast_percentiles
from weadge.domain.probability import (
    bucket_probability_from_normal,
    bucket_probability_from_percentiles,
    clamp_price,
    mid_to_prob,
)


def market_probability_from_quote(quote: pl.DataFrame | None) -> float | None:
    """Mid close of the quote snapshot -> probability."""
    if quote is None or quote.is_empty():
        return None
    mid = quote["mid_close"][0]
    return mid_to_prob(mid)


def kalshi_forecast_probability(
    percentiles: pl.DataFrame,
    decision_at: datetime,
    event_ticker: str,
    bucket_low: float | None,
    bucket_high: float | None,
) -> float | None:
    """Interpolate P(bucket) from the Kalshi forecast percentile history.

    Uses the latest end_period bucket at or before `decision_at` for the event.
    """
    snap = snapshot_forecast_percentiles(percentiles, decision_at)
    if snap.is_empty():
        return None
    ev = snap.filter(pl.col("event_ticker") == event_ticker)
    if ev.is_empty():
        return None
    pairs = {
        float(p): float(v)
        for p, v in ev.select("percentile", "numerical_forecast").iter_rows()
        if v is not None and p is not None
    }
    if len(pairs) < 2:
        return None
    return bucket_probability_from_percentiles(pairs, bucket_low, bucket_high)


def nbm_bucket_probability(
    forecasts: pl.DataFrame,
    decision_at: datetime,
    location_id: str,
    bucket_low: float | None,
    bucket_high: float | None,
    source: str = "nbm",
) -> float | None:
    """Latest NBM (or other source) forecast knowable at T, bucket probability.

    Uses percentiles p10..p90 when present, else Normal(mean, std).
    """
    from weadge.dataset.alignment import latest_knowable

    knowable = latest_knowable(
        forecasts.filter(pl.col("source") == source),
        decision_at,
        key_cols=["location_id", "valid_start", "source"],
    )
    if knowable.is_empty():
        return None
    row = (
        knowable.filter(pl.col("location_id") == location_id)
        .sort("available_at", descending=True)
        .head(1)
    )
    if row.is_empty():
        return None
    r = row.row(0, named=True)
    pcts = {p: r.get(col) for p, col in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]}
    pcts = {p: v for p, v in pcts.items() if v is not None}
    if len(pcts) >= 2:
        return bucket_probability_from_percentiles(pcts, bucket_low, bucket_high)
    if r.get("mean") is not None and r.get("std") is not None:
        return bucket_probability_from_normal(r["mean"], r["std"], bucket_low, bucket_high)
    return None


__all__ = [
    "clamp_price",
    "kalshi_forecast_probability",
    "market_probability_from_quote",
    "nbm_bucket_probability",
]
