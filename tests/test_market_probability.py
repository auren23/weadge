"""Categorical market probability baselines (G0.5).

p_market_raw (no cross-bucket fix), p_market_normalized (fail closed on
incomplete partitions), p_market_simplex (box-constrained projection, fail
closed on infeasible bounds), plus partition diagnostics.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from weadge.dataset.market_probability import (
    add_market_baselines,
    normalize_partition,
    project_bounded_simplex,
)

T0 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _frame(bid: list, ask: list, mid: list, event: str = "EVENT_A", decision: datetime = T0) -> pl.DataFrame:
    n = len(mid)
    return pl.DataFrame(
        {
            "series_ticker": ["KXHIGHNY"] * n,
            "event_ticker": [event] * n,
            "decision_at": [decision] * n,
            "market_bid": bid,
            "market_ask": ask,
            "market_mid": mid,
            "p_market_raw": mid,
        }
    )


class TestNormalizePartition:
    def test_basic_normalization(self) -> None:
        out = normalize_partition([0.10, 0.20, 0.30, 0.50])
        assert out is not None
        assert out == pytest.approx([0.090909, 0.181818, 0.272727, 0.454545], abs=1e-6)
        assert sum(out) == pytest.approx(1.0, abs=1e-12)

    def test_missing_bucket_fails_closed(self) -> None:
        assert normalize_partition([0.1, 0.2, None, 0.5]) is None

    def test_zero_sum_fails_closed(self) -> None:
        assert normalize_partition([0.0, 0.0]) is None

    def test_single_bucket(self) -> None:
        assert normalize_partition([0.4]) == pytest.approx([1.0])

    def test_empty_fails_closed(self) -> None:
        assert normalize_partition([]) is None


class TestProjectBoundedSimplex:
    def test_feasible_projection(self) -> None:
        q = project_bounded_simplex([0.2, 0.3, 0.35], [0.1, 0.2, 0.2], [0.3, 0.4, 0.5])
        assert q is not None
        assert q.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(q >= np.asarray([0.1, 0.2, 0.2]) - 1e-12)
        assert np.all(q <= np.asarray([0.3, 0.4, 0.5]) + 1e-12)

    def test_exact_simplex_untouched(self) -> None:
        """If mids already form a distribution inside the box, q == mid —
        never move market probabilities for no reason."""
        mid = [0.2, 0.3, 0.5]
        q = project_bounded_simplex(mid, [0.1, 0.2, 0.4], [0.3, 0.4, 0.6])
        assert q is not None
        assert q == pytest.approx(mid, abs=1e-12)

    def test_infeasible_lower_bound(self) -> None:
        """sum(bid) = 1.03 > 1: no distribution fits the books."""
        assert project_bounded_simplex([0.2, 0.3, 0.35], [0.33, 0.34, 0.36], [0.4, 0.45, 0.5]) is None

    def test_infeasible_upper_bound(self) -> None:
        """sum(ask) = 0.96 < 1: same fail-closed."""
        assert project_bounded_simplex([0.2, 0.3, 0.35], [0.1, 0.2, 0.2], [0.3, 0.33, 0.33]) is None

    def test_missing_bucket_fails_closed(self) -> None:
        assert project_bounded_simplex([0.1, 0.2, None], [0.1, 0.2, 0.2], [0.3, 0.4, 0.5]) is None

    def test_subpenny_preserved(self) -> None:
        """A 0.4-cent mid enters the math untouched — no cent clamp anywhere."""
        mid = [0.004, 0.996]
        q = project_bounded_simplex(mid, [0.003, 0.99], [0.005, 1.0])
        assert q is not None
        assert q == pytest.approx(mid, abs=1e-12)  # exact shortcut: sum == 1

    def test_out_of_unit_bounds_fails_closed(self) -> None:
        assert project_bounded_simplex([0.5, 0.5], [-0.1, 0.1], [0.6, 0.6]) is None
        assert project_bounded_simplex([0.5, 0.5], [0.1, 0.1], [0.6, 1.5]) is None


class TestAddMarketBaselines:
    def test_complete_partition(self) -> None:
        df = add_market_baselines(_frame([0.1, 0.2, 0.2], [0.3, 0.4, 0.5], [0.2, 0.3, 0.35]))
        assert df["market_bid_sum"][0] == pytest.approx(0.5)
        assert df["market_ask_sum"][0] == pytest.approx(1.2)
        assert df["market_prob_sum_raw"][0] == pytest.approx(0.85)
        assert df["market_simplex_feasible"].to_list() == [True] * 3
        assert df["p_market_simplex"].to_numpy().sum() == pytest.approx(1.0, abs=1e-9)
        assert df["p_market_normalized"].to_numpy().sum() == pytest.approx(1.0, abs=1e-12)

    def test_missing_bucket_nulls_whole_partition(self) -> None:
        """One bucket without a book => normalized/simplex/diagnostics are
        NULL for the WHOLE (event, decision_at) — never partial normalization."""
        df = add_market_baselines(
            _frame([0.1, 0.2, None, 0.2], [0.3, 0.4, None, 0.5], [0.2, 0.3, None, 0.35])
        )
        assert df["p_market_normalized"].null_count() == 4
        assert df["p_market_simplex"].null_count() == 4
        assert df["market_prob_sum_raw"].null_count() == 4
        assert df["market_bid_sum"].null_count() == 4
        assert df["market_ask_sum"].null_count() == 4
        assert df["market_simplex_feasible"].to_list() == [False] * 4

    def test_infeasible_partition_fails_closed(self) -> None:
        """sum(bid) = 1.10 > 1: books contain no categorical distribution —
        the finding is preserved, never smoothed away."""
        df = add_market_baselines(_frame([0.30, 0.40, 0.40], [0.35, 0.45, 0.50], [0.325, 0.425, 0.45]))
        assert df["market_simplex_feasible"].to_list() == [False] * 3
        assert df["p_market_simplex"].null_count() == 3
        assert df["market_bid_sum"][0] == pytest.approx(1.10)
        # raw mid stays fully present even when simplex is infeasible
        assert df["p_market_raw"].to_list() == pytest.approx([0.325, 0.425, 0.45])

    def test_partition_isolation(self) -> None:
        """Two events on the same day must never normalize across each other."""
        a = _frame([0.1, 0.2, 0.2], [0.3, 0.4, 0.5], [0.2, 0.3, 0.35], event="EVENT_A")
        b = _frame([0.1, 0.2, 0.7], [0.3, 0.4, 0.8], [0.2, 0.3, 0.75], event="EVENT_B")
        df = add_market_baselines(pl.concat([a, b]))
        a_norm = df.filter(pl.col("event_ticker") == "EVENT_A")["p_market_normalized"].to_numpy()
        b_norm = df.filter(pl.col("event_ticker") == "EVENT_B")["p_market_normalized"].to_numpy()
        assert a_norm.sum() == pytest.approx(1.0, abs=1e-12)
        assert b_norm.sum() == pytest.approx(1.0, abs=1e-12)
        # EVENT_A's values are its own raw / 0.85, unaffected by EVENT_B
        assert a_norm == pytest.approx([0.2 / 0.85, 0.3 / 0.85, 0.35 / 0.85], abs=1e-12)

    def test_series_isolation(self) -> None:
        """Same event_ticker string on different series is still two partitions."""
        a = _frame([0.1, 0.2, 0.2], [0.3, 0.4, 0.5], [0.2, 0.3, 0.35], event="EVENT_A")
        b = a.with_columns(pl.lit("KXHIGHCHI").alias("series_ticker"))
        df = add_market_baselines(pl.concat([a, b]))
        assert df.group_by("series_ticker").agg(
            pl.col("p_market_normalized").sum()
        )["p_market_normalized"].to_list() == pytest.approx([1.0, 1.0], abs=1e-12)
