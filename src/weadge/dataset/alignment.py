"""As-of alignment.

The one invariant that makes every weadge result honest:

    any data used at decision time T must have available_at <= T.

These helpers gate on the bronze/silver frames. They are intentionally dumb
and testable: filter by availability, then take the latest row per key.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from weadge.domain.time import ensure_utc


def _as_utc(dt: datetime) -> datetime:
    if isinstance(dt, datetime):
        return ensure_utc(dt)
    return ensure_utc(datetime.fromisoformat(str(dt)))


def latest_knowable(
    df: pl.DataFrame,
    decision_at: datetime,
    key_cols: list[str] | None = None,
    available_col: str = "available_at",
) -> pl.DataFrame:
    """Rows of `df` knowable at `decision_at`: available_at <= T, then the
    latest available row per key (default key: all columns except availability
    and value columns — pass key_cols explicitly for clarity)."""
    decision_at = _as_utc(decision_at)
    knowable = df.filter(pl.col(available_col) <= decision_at)
    if knowable.is_empty():
        return knowable
    keys = key_cols or [c for c in df.columns if c != available_col]
    order = knowable.sort(available_col)
    # dedupe keeping the LAST row per key: reverse, take first, restore order
    last = order.sort(available_col, descending=True).unique(subset=keys, keep="first")
    return last.sort(available_col)


def latest_quote_at_or_before(
    quotes: pl.DataFrame,
    decision_at: datetime,
    quote_ts_col: str = "ts",
) -> pl.DataFrame:
    """Last 1m quote at or before `decision_at` per market."""
    decision_at = _as_utc(decision_at)
    before = quotes.filter(pl.col(quote_ts_col) <= decision_at)
    if before.is_empty():
        return before
    return (
        before.sort(quote_ts_col, descending=True)
        .unique(subset=["market_ticker"], keep="first")
        .sort(quote_ts_col)
    )


def snapshot_forecast_percentiles(
    percentiles: pl.DataFrame,
    decision_at: datetime,
    end_period_col: str = "end_period_ts",
) -> pl.DataFrame:
    """Percentile rows as of `decision_at`: all percentiles of the latest
    end_period bucket at or before T, per event."""
    decision_at = _as_utc(decision_at)
    before = percentiles.filter(pl.col(end_period_col) <= decision_at)
    if before.is_empty():
        return before
    latest = (
        before.sort(end_period_col, descending=True)
        .unique(subset=["event_ticker", "percentile"], keep="first")
        .sort(end_period_col)
    )
    return latest


def assert_no_lookahead(
    frames: dict[str, pl.DataFrame],
    decision_at: datetime,
    ts_col: str = "ts",
) -> None:
    """Test helper: fail loudly if any frame has a row after the decision time
    that would be visible to the researcher (raw ingestion timestamps are
    excluded — only market-visible timestamps count)."""
    decision_at = _as_utc(decision_at)
    for name, df in frames.items():
        if ts_col in df.columns and not df.filter(pl.col(ts_col) > decision_at).is_empty():
            raise LookaheadViolation(name, decision_at)


class LookaheadViolation(ValueError):
    def __init__(self, table: str, decision_at: datetime) -> None:
        super().__init__(f"lookahead: table {table} contains rows after decision_at {decision_at}")
