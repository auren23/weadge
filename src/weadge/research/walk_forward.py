"""Walk-forward validation.

Rules:
  * never shuffle — the split is strictly chronological
  * train grows: [Jan-Mar] -> test [Apr]; [Jan-Apr] -> test [May]; ...
  * optional city-holdout: train on N-1 cities, test on the held-out city,
    which tests "weather-market structure" vs "memorized city bias".
"""

from __future__ import annotations

import calendar
from collections.abc import Iterator
from datetime import datetime

import polars as pl

from weadge.domain.time import ensure_utc


def _add_months_clamped(dt: datetime, months: int) -> datetime:
    """dt + months, clamping the day to the target month's length.

    Plain replace() raises on e.g. Jan 31 -> Feb 31; real event dates
    (a month can end on the 31st) must not crash the walk-forward split.
    """
    y, m = dt.year, dt.month + months
    while m > 12:
        m -= 12
        y += 1
    day = min(dt.day, calendar.monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=day)


def walk_forward_splits(
    dates: list[datetime],
    train_months: int = 3,
    test_months: int = 1,
) -> Iterator[tuple[datetime, datetime]]:
    """Yield (train_start, test_start) cut points.

    The design is EXPANDING train, fixed-length test, strictly chronological:

        train Jan->Mar  test Apr
        train Jan->Apr  test May
        train Jan->May  test Jun
        ...

    Dates must be sorted. No shuffling, no overlap between windows.
    """

    dates = sorted(ensure_utc(d) for d in dates)
    if not dates:
        return
    first = dates[0]
    test_start = _add_months_clamped(first, train_months)
    while test_start <= dates[-1]:
        yield first, test_start
        test_start = _add_months_clamped(test_start, test_months)


def split_frame(
    df: pl.DataFrame,
    train_start: datetime,
    test_start: datetime,
    time_col: str = "event_date",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a frame into train / test by a timestamp column (half-open windows)."""
    train_start = ensure_utc(train_start)
    test_start = ensure_utc(test_start)
    train = df.filter(
        (pl.col(time_col) >= train_start) & (pl.col(time_col) < test_start)
    )
    # test window: next month after test_start (day clamped to month length)
    test_end = _add_months_clamped(test_start, 1)
    test = df.filter((pl.col(time_col) >= test_start) & (pl.col(time_col) < test_end))
    return train, test


def city_holdout_split(
    df: pl.DataFrame,
    held_out_city: str,
    city_col: str = "city",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train on all cities except `held_out_city`; test on it.

    This is the strictest generalization check: it separates "learned market
    structure" from "memorized per-city bias".
    """
    train = df.filter(pl.col(city_col) != held_out_city)
    test = df.filter(pl.col(city_col) == held_out_city)
    return train, test
