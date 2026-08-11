"""Kalshi series fee-change history -> canonical frame.

Fee schedules are DATA: the backtest must replay the multiplier that was
actually in effect at execution time, never a hardcoded constant.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient
from weadge.domain.time import from_timestamp, utc_now
from weadge.storage.schema import FEE_CHANGES_SCHEMA


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fee_changes_frame(
    client: KalshiClient,
    series_ticker: str,
    *,
    show_historical: bool = True,
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch fee changes for a series (past and future) as a canonical frame."""
    raw_rows = client.get_series_fee_changes(series_ticker, show_historical=show_historical)
    rows = [
        {
            "series_ticker": series_ticker,
            "effective_at": from_timestamp(int(r["effective_time"]))
            if r.get("effective_time") is not None
            else (from_timestamp(int(r["effective_at"])) if r.get("effective_at") is not None else None),
            "fee_type": r.get("fee_type", ""),
            "fee_multiplier": _f(r.get("fee_multiplier")),
            "ingested_at": utc_now(),
        }
        for r in raw_rows
        if r.get("effective_time") is not None or r.get("effective_at") is not None
    ]
    if save_raw and raw_rows:
        lake = getattr(client, "_lake", None)
        if lake is not None:
            lake.save_raw_jsonl("kalshi", f"fee_changes/{series_ticker}", raw_rows)
    return pl.DataFrame(rows, schema=FEE_CHANGES_SCHEMA)
