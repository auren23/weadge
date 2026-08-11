"""Kalshi event forecast percentile history -> canonical frame.

NOTE ON PROVENANCE: Kalshi does not document the weather model behind this
endpoint. weadge names it `kalshi_forecast` — never `p_nbm`. Only when the
underlying model is confirmed may the column be relabeled.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient
from weadge.domain.time import from_timestamp, utc_now
from weadge.storage.schema import FORECAST_PERCENTILE_SCHEMA


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def forecast_percentile_frame(
    client: KalshiClient,
    event_ticker: str,
    series_ticker: str,
    *,
    interval: str = "1m",
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch forecast percentile history for one event."""
    raw_rows = client.get_event_forecast_percentile_history(
        event_ticker, series_ticker, interval=interval
    )
    rows = [
        {
            "event_ticker": event_ticker,
            "percentile": _f(r.get("percentile")),
            "numerical_forecast": _f(r.get("numerical_forecast") or r.get("raw_numerical_forecast")),
            "end_period_ts": from_timestamp(int(r["end_period_ts"])),
            "ingested_at": utc_now(),
        }
        for r in raw_rows
        if r.get("end_period_ts") is not None
    ]
    if save_raw and raw_rows:
        lake = getattr(client, "_lake", None)
        if lake is not None:
            lake.save_raw_jsonl("kalshi", f"forecast_percentiles/{event_ticker}", raw_rows)
    return pl.DataFrame(rows, schema=FORECAST_PERCENTILE_SCHEMA)
