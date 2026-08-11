"""Bucket probability math — the feature every downstream result depends on."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from weadge.domain.probability import (
    assert_bucket_distribution,
    bucket_probability_from_normal,
    bucket_probability_from_percentiles,
    clamp_price,
    fit_normal_from_percentiles,
    logit_to_prob,
    mid_to_prob,
    prob_to_logit,
)


class TestNormalBucket:
    def test_symmetric_bucket(self) -> None:
        # Normal(0,1): P(-1 <= X < 1) = 2*Phi(1) - 1
        p = bucket_probability_from_normal(0.0, 1.0, -1.0, 1.0)
        assert p == pytest.approx(0.68268949, abs=1e-6)

    def test_full_range_sums_to_one(self) -> None:
        assert bucket_probability_from_normal(50.0, 5.0, None, None) == pytest.approx(1.0)
        assert bucket_probability_from_normal(50.0, 5.0, None, 60.0) + bucket_probability_from_normal(
            50.0, 5.0, 60.0, None
        ) == pytest.approx(1.0)

    def test_exact_low_included(self) -> None:
        # P(X < 0) = 0.5, so P(0 <= X < inf) = 0.5
        assert bucket_probability_from_normal(0.0, 1.0, 0.0, None) == pytest.approx(0.5)

    def test_high_exclusive(self) -> None:
        # P(-inf <= X < 0) = 0.5 — the upper edge is exclusive
        assert bucket_probability_from_normal(0.0, 1.0, None, 0.0) == pytest.approx(0.5)

    def test_bad_std_raises(self) -> None:
        with pytest.raises(ValueError):
            bucket_probability_from_normal(50.0, 0.0, 0.0, 1.0)


class TestPercentileFit:
    """Percentiles are fit to a Normal — never linearly interpolated/extrapolated."""

    P10 = 88.0
    P50 = 90.0
    P90 = 92.0

    def _pcts(self) -> dict[float, float]:
        return {10.0: self.P10, 50.0: self.P50, 90.0: self.P90}

    def test_fit_reconstructs_normal_moments(self) -> None:
        """Five percentiles of N(90, 2) must fit back to mu~90, sigma~2."""
        pcts = {p: float(stats.norm.ppf(p / 100, loc=90.0, scale=2.0)) for p in (10, 25, 50, 75, 90)}
        mu, sigma = fit_normal_from_percentiles(pcts)
        assert mu == pytest.approx(90.0, abs=1e-6)
        assert sigma == pytest.approx(2.0, abs=1e-6)

    def test_bucket_probability_matches_fitted_normal(self) -> None:
        """P(89 <= X < 91) must equal the fitted Gaussian's, not linear CDF interp."""
        pcts = self._pcts()  # exactly N(90, 2/z_90) with z_90 = 1.28155...
        mu, sigma = fit_normal_from_percentiles(pcts)
        expected = stats.norm.cdf(91.0, mu, sigma) - stats.norm.cdf(89.0, mu, sigma)
        got = bucket_probability_from_percentiles(pcts, 89.0, 91.0)
        assert got == pytest.approx(expected, abs=1e-12)
        # sanity: not the old linear-CDF value 0.40
        assert got != pytest.approx(0.40, abs=1e-3)

    def test_monotonic_in_bucket_low(self) -> None:
        pcts = self._pcts()
        ps = [
            bucket_probability_from_percentiles(pcts, lo, 91.0)
            for lo in (87.0, 88.0, 89.0, 90.0)
        ]
        assert ps == sorted(ps, reverse=True)  # raising low edge => non-increasing prob

    def test_tail_buckets_follow_fitted_gaussian(self) -> None:
        """Tails are Gaussian, not flat: P(X<88)=0.10 and P(X>=92)=0.10 here
        only because the fitted sigma passes exactly through p10/p90."""
        pcts = self._pcts()
        mu, sigma = fit_normal_from_percentiles(pcts)
        below = bucket_probability_from_percentiles(pcts, None, 88.0)
        above = bucket_probability_from_percentiles(pcts, 92.0, None)
        assert below == pytest.approx(stats.norm.cdf(88.0, mu, sigma), abs=1e-12)
        assert above == pytest.approx(1 - stats.norm.cdf(92.0, mu, sigma), abs=1e-12)
        total = below + bucket_probability_from_percentiles(pcts, 88.0, 92.0) + above
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_sparse_percentiles_do_not_collapse_tails(self) -> None:
        """Regression: p10=85, p90=93 only. The old code flat-extrapolated:
        P(T<=50) = 0.10 and P(T<=93.0001) = 1.0. The fitted Gaussian gives
        ~0 and ~0.90 — the right tail stays a real tail."""
        pcts = {10.0: 85.0, 90.0: 93.0}
        p_below_50 = bucket_probability_from_percentiles(pcts, None, 50.0)
        p_below_930001 = bucket_probability_from_percentiles(pcts, None, 93.0001)
        assert p_below_50 < 1e-4          # was 0.10
        assert p_below_930001 < 0.95      # was 1.00
        assert p_below_930001 > 0.85      # the mass is still where the data is
        # and the complement: P(X > 93) is no longer compressed into (93, 93.0001]
        assert bucket_probability_from_percentiles(pcts, 93.0001, None) > 0.05

    def test_two_points_fit_exact_line(self) -> None:
        pcts = {10.0: 85.0, 90.0: 93.0}
        mu, sigma = fit_normal_from_percentiles(pcts)
        # sigma = (x90 - x10) / (z90 - z10), mu = x50 of the fitted line
        z10, z90 = stats.norm.ppf(0.10), stats.norm.ppf(0.90)
        assert sigma == pytest.approx((93.0 - 85.0) / (z90 - z10), abs=1e-9)
        assert mu == pytest.approx(85.0 - sigma * z10, abs=1e-9)

    def test_less_than_two_points_raises(self) -> None:
        with pytest.raises(ValueError):
            bucket_probability_from_percentiles({10.0: 85.0}, 0.0, 1.0)
        with pytest.raises(ValueError):
            bucket_probability_from_percentiles({}, 0.0, 1.0)

    def test_degenerate_identical_values_raise(self) -> None:
        with pytest.raises(ValueError):
            fit_normal_from_percentiles({10.0: 85.0, 90.0: 85.0})


class TestBucketDistribution:
    def test_partition_sums_to_one(self) -> None:
        probs = [
            bucket_probability_from_normal(90.0, 2.0, None, 88.0),
            bucket_probability_from_normal(90.0, 2.0, 88.0, 90.0),
            bucket_probability_from_normal(90.0, 2.0, 90.0, 92.0),
            bucket_probability_from_normal(90.0, 2.0, 92.0, None),
        ]
        assert_bucket_distribution(probs)  # must not raise

    def test_partition_off_by_one_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_bucket_distribution([0.3, 0.3, 0.3])  # sums to 0.9

    def test_empty_partition_raises(self) -> None:
        with pytest.raises(ValueError):
            assert_bucket_distribution([])


class TestTransforms:
    def test_logit_roundtrip(self) -> None:
        for p in (0.02, 0.25, 0.5, 0.75, 0.98):
            assert logit_to_prob(prob_to_logit(p)) == pytest.approx(p, abs=1e-6)

    def test_clamp_is_domain_only(self) -> None:
        """clamp_price enforces [0,1] validity, never a 1-cent market rule."""
        assert clamp_price(-1.0) == 0.0
        assert clamp_price(2.0) == 1.0
        assert clamp_price(0.5) == 0.5

    def test_subpenny_market_prices_preserved(self) -> None:
        """Regression: market probabilities must NOT be clamped to >= 1 cent."""
        assert mid_to_prob(0.004) == pytest.approx(0.004)
        assert mid_to_prob(0.055) == pytest.approx(0.055)
        assert mid_to_prob(0.997) == pytest.approx(0.997)

    def test_mid_to_prob_domain_bounds_only(self) -> None:
        assert mid_to_prob(-0.1) == 0.0
        assert mid_to_prob(1.5) == 1.0
        assert mid_to_prob(None) is None

    def test_prob_to_logit_stable_at_extremes(self) -> None:
        assert np.isfinite(prob_to_logit(0.0))
        assert np.isfinite(prob_to_logit(1.0))
