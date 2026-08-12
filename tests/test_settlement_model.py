"""p_fair: the average-settlement variance correction (tau - 2/3)."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from weadge.research.crypto_settlement_model import p_fair


def test_average_model_is_sharper_near_expiry() -> None:
    """Final-minute average has 1/3 the endpoint variance: with S above K
    the average model must be MORE confident in YES, and the gap must be
    biggest at tau=1 and vanish as tau grows."""
    s, k, sig = 100.05, 100.0, 1e-3
    gaps = []
    for tau in (1.0, 2.0, 15.0):
        p_end = p_fair(s, k, sig, tau, settle_avg=False)
        p_avg = p_fair(s, k, sig, tau, settle_avg=True)
        assert p_avg > p_end > 0.5
        gaps.append(p_avg - p_end)
    assert gaps[0] > gaps[1] > gaps[2]


def test_exact_values() -> None:
    s, k, sig, tau = 100.05, 100.0, 1e-3, 1.0
    z = math.log(s / k) / sig
    assert p_fair(s, k, sig, tau, settle_avg=False) == pytest.approx(norm.cdf(z))
    assert p_fair(s, k, sig, tau, settle_avg=True) == pytest.approx(
        norm.cdf(z / math.sqrt(1.0 / 3.0))
    )


def test_at_the_money_is_half_under_both() -> None:
    for settle_avg in (False, True):
        assert p_fair(100.0, 100.0, 1e-3, 5.0, settle_avg=settle_avg) == 0.5
