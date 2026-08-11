"""Categorical market probability baselines (G0.5).

One KXHIGHNY event is a ladder of mutually exclusive binary markets whose
independent mids do NOT necessarily form a probability distribution (spread,
asynchronous books, aggregation artifacts). Three baselines answer three
different questions, and research code must never collapse them back into a
bare `p_market`:

    p_market_raw        — the bucket's own mid, no cross-bucket adjustment.
                          Kept because "what do the independent books look
                          like" is itself the research object.
    p_market_normalized — raw / sum(raw) over the partition. FAILS CLOSED:
                          if any bucket lacks a valid mid, the whole
                          partition is NULL — a missing book must never be
                          interpreted as "other buckets are more probable".
    p_market_simplex    — box-constrained simplex projection:
                          min  sum (q_i - mid_i)^2
                          s.t. sum q = 1,  bid_i <= q_i <= ask_i
                          Solved by bisection on the Lagrange multiplier
                          (q_i = clip(mid_i - lambda, bid_i, ask_i)); no
                          optimizer dependency. FAILS CLOSED on infeasible
                          bounds: "the books contain no categorical
                          distribution" is information, not an error to fix.

Partition-level diagnostics: market_bid_sum / market_prob_sum_raw /
market_ask_sum give the quick eyeball "bid_sum <= prob_sum <= ask_sum".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

_PARTITION_KEYS = ["series_ticker", "event_ticker", "decision_at"]
_SUM_EPS = 1e-12
_BISECTION_ITERS = 100


def normalize_partition(mids: Sequence[float | None]) -> list[float] | None:
    """p_raw / sum(p_raw) over the partition; None (fail closed) unless every
    bucket has a valid mid and the sum is positive and finite."""
    if any(m is None for m in mids):
        return None
    arr = np.asarray([float(m) for m in mids], dtype=float)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= _SUM_EPS:
        return None
    return (arr / total).tolist()


def project_bounded_simplex(
    mid: Sequence[float | None],
    lower: Sequence[float | None],
    upper: Sequence[float | None],
    iters: int = _BISECTION_ITERS,
) -> np.ndarray | None:
    """Box-constrained projection of mids onto the probability simplex.

    Returns q with sum(q) == 1 and bid <= q <= ask, or None when the
    problem is infeasible or the partition is incomplete. Never silently
    relaxes the bounds — an infeasible book is a finding, not a bug.

    Structure: q_i = clip(mid_i - lambda, lower_i, upper_i) with sum(q)
    monotone decreasing in lambda, so one bisection finds the multiplier.
    """
    if any(x is None for x in mid) or any(x is None for x in lower) or any(x is None for x in upper):
        return None
    m = np.asarray([float(x) for x in mid], dtype=float)
    lo = np.asarray([float(x) for x in lower], dtype=float)
    hi = np.asarray([float(x) for x in upper], dtype=float)

    # fail closed: invalid bounds, or no q can sum to 1 inside the box
    if np.any(lo < 0.0) or np.any(hi > 1.0):
        return None
    if np.any(lo > hi):
        return None
    if float(lo.sum()) > 1.0:
        return None
    if float(hi.sum()) < 1.0:
        return None

    # exact case: mids already form a distribution inside the box — do not
    # move market probabilities for no reason
    if abs(float(m.sum()) - 1.0) <= _SUM_EPS and np.all(m >= lo) and np.all(m <= hi):
        return m.copy()

    lam_lo = float(np.min(m - hi)) - 1.0   # at lam_lo, q = upper -> sum >= 1
    lam_hi = float(np.max(m - lo)) + 1.0   # at lam_hi, q = lower -> sum <= 1
    q = np.clip(m, lo, hi)
    for _ in range(iters):
        lam = (lam_lo + lam_hi) / 2.0
        q = np.clip(m - lam, lo, hi)
        if q.sum() > 1.0:
            lam_lo = lam
        else:
            lam_hi = lam
    return q


def add_market_baselines(df: pl.DataFrame) -> pl.DataFrame:
    """Phase B: partition-level market baselines, computed per
    (series_ticker, event_ticker, decision_at).

    Adds: p_market_normalized, p_market_simplex, market_prob_sum_raw,
    market_bid_sum, market_ask_sum, market_simplex_feasible.
    Any partition that is incomplete or infeasible fails closed to NULL /
    False — never partially normalized.
    """
    out: list[pl.DataFrame] = []
    for group in df.partition_by(_PARTITION_KEYS):
        bids = group["market_bid"].to_list()
        asks = group["market_ask"].to_list()
        mids = group["market_mid"].to_list()

        complete = all(
            b is not None and a is not None and m is not None
            for b, a, m in zip(bids, asks, mids, strict=True)
        )
        if not complete:
            n = group.height
            group = group.with_columns(
                pl.Series("p_market_normalized", [None] * n, dtype=pl.Float64),
                pl.Series("p_market_simplex", [None] * n, dtype=pl.Float64),
                pl.Series("market_prob_sum_raw", [None] * n, dtype=pl.Float64),
                pl.Series("market_bid_sum", [None] * n, dtype=pl.Float64),
                pl.Series("market_ask_sum", [None] * n, dtype=pl.Float64),
                pl.Series("market_simplex_feasible", [False] * n, dtype=pl.Boolean),
            )
            out.append(group)
            continue

        b = np.asarray(bids, dtype=float)
        a = np.asarray(asks, dtype=float)
        m = np.asarray(mids, dtype=float)
        bid_sum = float(b.sum())
        ask_sum = float(a.sum())
        prob_sum_raw = float(m.sum())
        n = group.height
        normalized = normalize_partition(mids)
        simplex = project_bounded_simplex(mids, bids, asks)
        feasible = simplex is not None
        norm_list = normalized if normalized is not None else [None] * n
        simplex_list = simplex.tolist() if simplex is not None else [None] * n

        group = group.with_columns(
            pl.Series("p_market_normalized", norm_list, dtype=pl.Float64),
            pl.Series("p_market_simplex", simplex_list, dtype=pl.Float64),
            pl.Series("market_prob_sum_raw", [prob_sum_raw] * n, dtype=pl.Float64),
            pl.Series("market_bid_sum", [bid_sum] * n, dtype=pl.Float64),
            pl.Series("market_ask_sum", [ask_sum] * n, dtype=pl.Float64),
            pl.Series("market_simplex_feasible", [feasible] * n, dtype=pl.Boolean),
        )
        out.append(group)
    return pl.concat(out)


__all__ = ["add_market_baselines", "normalize_partition", "project_bounded_simplex"]
