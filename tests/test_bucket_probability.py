"""Bucket probability math — the feature every downstream result depends on."""

from __future__ import annotations

import pytest

from weadge.domain.probability import (
    bucket_probability_from_normal,
    bucket_probability_from_percentiles,
    clamp_price,
    logit_to_prob,
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


class TestPercentileBucket:
    P10 = 88.0
    P50 = 90.0
    P90 = 92.0

    def _pcts(self) -> dict[float, float]:
        return {10.0: self.P10, 50.0: self.P50, 90.0: self.P90}

    def test_interpolation_midpoint(self) -> None:
        p = bucket_probability_from_percentiles(self._pcts(), 89.0, 91.0)
        # 89 is halfway between P10(88) and P50(90) -> CDF(89)=0.30
        # 91 is halfway between P50(90) and P90(92) -> CDF(91)=0.70
        assert p == pytest.approx(0.40, abs=1e-9)

    def test_monotonic_in_bucket_low(self) -> None:
        pcts = self._pcts()
        ps = [
            bucket_probability_from_percentiles(pcts, lo, 91.0)
            for lo in (87.0, 88.0, 89.0, 90.0)
        ]
        assert ps == sorted(ps, reverse=True)  # raising low edge => non-increasing prob

    def test_tail_buckets(self) -> None:
        pcts = self._pcts()
        below = bucket_probability_from_percentiles(pcts, None, 88.0)
        above = bucket_probability_from_percentiles(pcts, 92.0, None)
        assert below == pytest.approx(0.10, abs=1e-9)   # flat-extrapolated
        assert above == pytest.approx(0.10, abs=1e-9)
        total = below + bucket_probability_from_percentiles(pcts, 88.0, 92.0) + above
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            bucket_probability_from_percentiles({}, 0.0, 1.0)


class TestTransforms:
    def test_logit_roundtrip(self) -> None:
        for p in (0.02, 0.25, 0.5, 0.75, 0.98):
            assert logit_to_prob(prob_to_logit(p)) == pytest.approx(p, abs=1e-6)

    def test_clamp(self) -> None:
        assert clamp_price(-1.0) == 0.01
        assert clamp_price(2.0) == 0.99
        assert clamp_price(0.5) == 0.5
