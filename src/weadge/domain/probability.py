"""Probability math shared by research, backtest, and dataset layers.

Everything here is pure and deterministic — no I/O, no random state.
"""

from __future__ import annotations

import math

import numpy as np

# Kalshi prices are NOT restricted to whole cents: sub-cent contract prices
# exist (e.g. $0.055) and weather tails trade below 1 cent. Market
# probabilities must be taken at face value. The ONLY clipping in this module
# is numerical stability for the logit transform; market-rule clamps are a
# different concept and must never be applied here.
_LOGIT_EPS = 1e-6


def clamp_price(p: float | np.ndarray) -> float | np.ndarray:
    """Clip a probability into the valid [0, 1] domain.

    This is a domain-validity clip, NOT a market rule: it never imposes a
    1-cent minimum or 99-cent maximum on a price.
    """
    return float(np.clip(p, 0.0, 1.0)) if np.ndim(p) == 0 else np.clip(p, 0.0, 1.0)


def prob_to_logit(p: float | np.ndarray) -> float | np.ndarray:
    """p -> log(p/(1-p)), clipped away from the endpoints (numerical only)."""
    p = float(np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)) if np.ndim(p) == 0 else np.clip(
        p, _LOGIT_EPS, 1.0 - _LOGIT_EPS
    )
    return float(np.log(p / (1.0 - p))) if np.ndim(p) == 0 else np.log(p / (1.0 - p))


def logit_to_prob(x: float | np.ndarray) -> float | np.ndarray:
    if np.ndim(x) == 0:
        return float(1.0 / (1.0 + math.exp(-x)))
    return 1.0 / (1.0 + np.exp(-x))


def mid_to_prob(mid: float | None) -> float | None:
    """Mid price (dollars on [0,1]) -> probability, exactly — no cent clamp.

    A market at $0.004 stays 0.004; only the domain bounds [0, 1] apply.
    """
    if mid is None:
        return None
    return float(np.clip(mid, 0.0, 1.0))


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


def fit_normal_from_percentiles(percentiles: dict[float, float]) -> tuple[float, float]:
    """Least-squares Normal fit of percentile pairs: x_p = mu + sigma*Phi^-1(p).

    Returns (mu, sigma). Requires at least 2 distinct (percentile, value)
    pairs; a degenerate fit (sigma <= 0 or non-finite) raises ValueError.

    This is the v0 repair for percentile tails: instead of linearly
    interpolating the CDF and flat-extrapolating beyond the extreme values
    (which compressed the whole right tail into (p90, p90+eps] and reported
    P(T <= p10) for EVERY value below p10), we fit a Gaussian and let the
    tails follow it.
    """
    if not percentiles or len(percentiles) < 2:
        raise ValueError("percentiles must contain at least 2 distinct points")
    from scipy import stats  # local import keeps module import cheap

    keys = sorted(percentiles)
    ps = np.asarray(keys, dtype=float) / 100.0  # percent units -> probabilities
    xs = np.asarray([percentiles[k] for k in keys], dtype=float)
    zs = stats.norm.ppf(ps)
    z_bar, x_bar = float(zs.mean()), float(xs.mean())
    denom = float(np.sum((zs - z_bar) ** 2))
    if denom == 0 or not np.isfinite(denom):
        raise ValueError("cannot fit a Normal from identical percentile values")
    sigma = float(np.sum((zs - z_bar) * (xs - x_bar)) / denom)
    mu = float(x_bar - sigma * z_bar)
    if not (sigma > 0 and np.isfinite(mu) and np.isfinite(sigma)):
        raise ValueError(f"degenerate Normal fit from percentiles: mu={mu}, sigma={sigma}")
    return mu, sigma


def bucket_probability_from_percentiles(
    percentiles: dict[float, float],  # {p: value}, p in (0, 1)
    bucket_low: float | None,
    bucket_high: float | None,
) -> float:
    """P(low <= X < high) from a CDF given as (percentile, value) pairs.

    A Gaussian is fit to the pairs (x_p = mu + sigma*Phi^-1(p)) and the
    bucket probability is read off the fitted Normal. There is no linear
    interpolation in value space and no flat tail extrapolation: with only
    p10=85F and p90=93F, P(T <= 50F) is ~0 and P(T <= 93.0001F) is ~0.9,
    never 0.1 / 1.0.
    """
    mu, sigma = fit_normal_from_percentiles(percentiles)
    return bucket_probability_from_normal(mu, sigma, bucket_low, bucket_high)


def assert_bucket_distribution(probs: list[float], tolerance: float = 1e-6) -> None:
    """Mutually exclusive KXHIGH buckets must form a probability distribution.

    Every partition of the outcome space (all markets of one event at one
    snapshot) must have P probabilities summing to ~1; anything else means
    the feature pipeline is broken and downstream alpha is meaningless.
    """
    total = float(sum(probs))
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"bucket probabilities must sum to 1, got {total:.9f} (tolerance {tolerance}); "
            "mutually exclusive event buckets do not form a distribution"
        )


def edge(p_model: float, quote_price: float) -> float:
    """Model minus executable price. Positive means the model sees value."""
    return p_model - quote_price
