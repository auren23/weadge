"""Probability math shared by research, backtest, and dataset layers.

Everything here is pure and deterministic — no I/O, no random state.
"""

from __future__ import annotations

import math

import numpy as np

# Kalshi prices are bounded to [1, 99] cents in practice; we use [0.01, 0.99].
MIN_P = 0.01
MAX_P = 0.99
_LOGIT_EPS = 1e-6


def clamp_price(p: float | np.ndarray) -> float | np.ndarray:
    """Clamp a probability into the tradable Kalshi price range."""
    return float(np.clip(p, MIN_P, MAX_P)) if np.ndim(p) == 0 else np.clip(p, MIN_P, MAX_P)


def prob_to_logit(p: float | np.ndarray) -> float | np.ndarray:
    """p -> log(p/(1-p)), clipped away from the endpoints."""
    p = float(np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)) if np.ndim(p) == 0 else np.clip(
        p, _LOGIT_EPS, 1.0 - _LOGIT_EPS
    )
    return float(np.log(p / (1.0 - p))) if np.ndim(p) == 0 else np.log(p / (1.0 - p))


def logit_to_prob(x: float | np.ndarray) -> float | np.ndarray:
    if np.ndim(x) == 0:
        return float(1.0 / (1.0 + math.exp(-x)))
    return 1.0 / (1.0 + np.exp(-x))


def mid_to_prob(mid: float | None) -> float | None:
    """Convert a mid price (cents on [0,1]) to a probability."""
    if mid is None:
        return None
    return float(np.clip(mid, MIN_P, MAX_P))


def bucket_probability_from_normal(
    mean: float, std: float, bucket_low: float | None, bucket_high: float | None
) -> float:
    """P(bucket_low <= X < bucket_high) for X ~ Normal(mean, std).

    Kalshi temperature buckets are half-open intervals [floor, cap) with the
    cap strike being the next integer; a missing cap means an unbounded tail.
    """
    if std is None or std <= 0:
        raise ValueError(f"std must be > 0 for bucket probability, got {std}")
    cdf_high = 1.0 if bucket_high is None else _normal_cdf(bucket_high, mean, std)
    cdf_low = 0.0 if bucket_low is None else _normal_cdf(bucket_low, mean, std)
    return float(np.clip(cdf_high - cdf_low, 0.0, 1.0))


def _normal_cdf(x: float, mean: float, std: float) -> float:
    from scipy import stats  # local import keeps module import cheap

    return float(stats.norm.cdf(x, loc=mean, scale=std))


def bucket_probability_from_percentiles(
    percentiles: dict[float, float],  # {p: value}, p in (0, 1)
    bucket_low: float | None,
    bucket_high: float | None,
) -> float:
    """Interpolate P(low <= X < high) from a CDF given as (percentile, value) pairs.

    Values are sorted, duplicates removed; linear interpolation in value space.
    CDF semantics: a percentile p at value v means P(X <= v) = p/100, so the
    CDF at the lowest value is its percentile (not 0) and beyond the highest
    value the CDF saturates at 1. Used for Kalshi forecast percentile history
    and NBM p10..p90.
    """
    if not percentiles:
        raise ValueError("percentiles must be non-empty")
    keys = sorted(percentiles)
    ps = np.asarray(keys, dtype=float) / 100.0   # percent units -> probabilities
    xs = np.asarray([percentiles[k] for k in keys], dtype=float)

    def cdf(x: float) -> float:
        if x <= xs[0]:
            return float(ps[0])
        if x > xs[-1]:
            return 1.0
        return float(np.interp(x, xs, ps))

    hi = 1.0 if bucket_high is None else cdf(bucket_high)
    lo = 0.0 if bucket_low is None else cdf(bucket_low)
    return float(np.clip(hi - lo, 0.0, 1.0))


def edge(p_model: float, quote_price: float) -> float:
    """Model minus executable price. Positive means the model sees value."""
    return p_model - quote_price
