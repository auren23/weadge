"""Probability features for the alpha dataset.

    p_market          — mid price of the latest completed 1m quote (exact, no cent clamp)
    p_kalshi_forecast — Kalshi's own percentile history, fit to a Normal over the bucket
    p_nbm             — NBM distribution over the bucket (Normal mean/std first,
                        percentile-fit Normal as fallback)

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


def _reported_bounds(
    bucket_low: float | None, bucket_high: float | None
) -> tuple[float | None, float | None]:
    """Continuous forecast -> probability of the DCR's WHOLE-DEGREE report.

    Kalshi temperature buckets settle on the NWS daily report, which is
    rounded to whole degrees (T79 = "less than 79°" -> reported <= 78;
    B79.5 = "between 79-80°" -> reported 79 or 80; T86 = "greater than
    86°" -> reported >= 87). Rounding a continuous distribution the same
    way shifts every strike boundary by half a degree, TOWARDS the bucket
    for closed B-buckets and AWAY for strict T-buckets:

        B [floor, cap]   -> [floor-0.5, cap+0.5)
        T "less than c"  -> (-inf, c-0.5)
        T "greater than f" -> [f+0.5, +inf)

    This makes the bucket probabilities tile the real line (sum to 1) on
    the 1-degree-wide ladders Kalshi actually runs (e.g. {<=78}, {79,80},
    {81,82}, ..., {>=87}); without it the mass between buckets belongs to
    no market and every partition sums to <1, silently handicapping the
    model in the comparison.
    """
    if bucket_low is None:  # T lower: strictly less than the cap
        return None, bucket_high - 0.5
    if bucket_high is None:  # T upper: strictly greater than the floor
        return bucket_low + 0.5, None
    return bucket_low - 0.5, bucket_high + 0.5


def market_probability_from_quote(quote: pl.DataFrame | None) -> float | None:
    """Mid close of the quote snapshot -> probability (exact, no cent clamp)."""
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
    The percentile pairs are fit to a Normal (no flat tail extrapolation).
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
    low, high = _reported_bounds(bucket_low, bucket_high)
    return bucket_probability_from_percentiles(pairs, low, high)


def nbm_bucket_probability(
    forecasts: pl.DataFrame,
    decision_at: datetime,
    location_id: str,
    bucket_low: float | None,
    bucket_high: float | None,
    source: str = "nbm",
) -> float | None:
    """Latest NBM (or other source) forecast knowable at T, bucket probability.

    Priority: Normal(mean, std) baseline first; only when mean/std are
    missing do we fall back to a Normal fit on p10..p90. Percentile
    interpolation must never override a real distribution.
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
    low, high = _reported_bounds(bucket_low, bucket_high)
    if r.get("mean") is not None and r.get("std") is not None:
        return bucket_probability_from_normal(r["mean"], r["std"], low, high)
    pcts = {
        p: r.get(col)
        for p, col in [(10, "p10"), (25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]
    }
    pcts = {p: v for p, v in pcts.items() if v is not None}
    if len(pcts) >= 2:
        return bucket_probability_from_percentiles(pcts, low, high)
    return None


__all__ = [
    "clamp_price",
    "kalshi_forecast_probability",
    "market_probability_from_quote",
    "nbm_bucket_probability",
]
