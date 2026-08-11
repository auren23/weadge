"""Walk-forward: strictly chronological, no shuffle, growing windows."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from weadge.research.walk_forward import (
    city_holdout_split,
    split_frame,
    walk_forward_splits,
)


def _dates() -> list[datetime]:
    return [
        datetime(2026, m, 1, 0, 0, tzinfo=UTC)
        for m in range(1, 13)
    ]


class TestWalkForwardSplits:
    def test_expanding_windows(self) -> None:
        """train grows from the first date; test windows step forward by 1 month."""
        dates = _dates()
        cuts = list(walk_forward_splits(dates, train_months=3, test_months=1))
        assert cuts[0] == (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC))
        assert cuts[1] == (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC))
        assert cuts[2] == (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC))
        # train windows never overlap test windows
        for (tr_start, te_start) in cuts:
            assert tr_start < te_start

    def test_last_window_reaches_year_end(self) -> None:
        dates = _dates()
        cuts = list(walk_forward_splits(dates, train_months=3, test_months=1))
        # with 12 monthly dates, 3-month train + 1-month test gives Apr..Dec tests
        assert cuts[-1][1] == datetime(2026, 12, 1, tzinfo=UTC)

    def test_empty_input(self) -> None:
        assert list(walk_forward_splits([], 3, 1)) == []


class TestSplitFrame:
    def test_half_open_windows(self) -> None:
        df = pl.DataFrame(
            {
                "event_date": [
                    datetime(2026, 3, 15, tzinfo=UTC),   # train
                    datetime(2026, 4, 1, tzinfo=UTC),    # test (start)
                    datetime(2026, 4, 15, tzinfo=UTC),   # test
                    datetime(2026, 5, 1, tzinfo=UTC),    # after test window
                ]
            }
        )
        train, test = split_frame(
            df,
            datetime(2026, 3, 1, tzinfo=UTC),
            datetime(2026, 4, 1, tzinfo=UTC),
        )
        assert train.height == 1
        assert test.height == 2

    def test_test_window_is_one_month(self) -> None:
        df = pl.DataFrame(
            {
                "event_date": [
                    datetime(2026, 4, 15, tzinfo=UTC),
                    datetime(2026, 4, 30, tzinfo=UTC),
                ]
            }
        )
        _, test = split_frame(
            df,
            datetime(2026, 3, 1, tzinfo=UTC),
            datetime(2026, 4, 1, tzinfo=UTC),
        )
        assert test.height == 2  # both inside April


class TestCityHoldout:
    def test_holdout_is_disjoint_and_exhaustive(self) -> None:
        df = pl.DataFrame(
            {
                "city": ["NY", "NY", "CHI", "MIA"],
                "x": [1, 2, 3, 4],
            }
        )
        train, test = city_holdout_split(df, "NY")
        assert test["city"].to_list() == ["NY", "NY"]
        assert set(train["city"].to_list()) == {"CHI", "MIA"}
        assert train.height + test.height == df.height


class TestMonthEndSafety:
    """Regression: a cut point on Jan 31 must not crash on 'Feb 31'."""

    def test_split_frame_clamps_test_end(self) -> None:
        df = pl.DataFrame(
            {
                "event_date": [
                    datetime(2026, 1, 31, tzinfo=UTC),  # test (window start)
                    datetime(2026, 2, 15, tzinfo=UTC),  # test
                ]
            }
        )
        train, test = split_frame(
            df,
            datetime(2026, 1, 31, tzinfo=UTC),
            datetime(2026, 1, 31, tzinfo=UTC),
        )
        assert train.height == 0
        assert test.height == 2  # half-open window Jan 31 -> Feb 28 (clamped)

    def test_walk_forward_splits_clamps_month_end(self) -> None:
        dates = [
            datetime(2026, 1, 31, tzinfo=UTC),
            datetime(2026, 3, 1, tzinfo=UTC),
        ]
        cuts = list(walk_forward_splits(dates, train_months=1, test_months=1))
        assert cuts[0] == (
            datetime(2026, 1, 31, tzinfo=UTC),
            datetime(2026, 2, 28, tzinfo=UTC),  # not Feb 31
        )
