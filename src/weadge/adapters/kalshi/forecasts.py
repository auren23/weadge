"""Kalshi event forecast percentile history -> canonical frame.

NOTE ON PROVENANCE: Kalshi does not document the weather model behind this
endpoint. weadge names it `kalshi_forecast` — never `p_nbm`. Only when the
underlying model is confirmed may the column be relabeled.

Wire shape (2026-08-11): the endpoint is
    /series/{series}/events/{event}/forecast_percentile_history
returning `forecast_history[]` where each entry carries `end_period_ts` and a
`percentile_points[]` array of {percentile, numerical_forecast}. The legacy
flat shape ({end_period_ts, percentile, numerical_forecast} rows) is still
tolerated.
"""

from __future__ import annotations

from datetime import datetime
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


def _flatten(history: list[dict[str, Any]], event_ticker: str) -> list[dict[str, Any]]:
    """forecast_history[] -> one row per (end_period_ts, percentile)."""
    rows: list[dict[str, Any]] = []
    for h in history:
        ep = h.get("end_period_ts")
        if ep is None:
            continue
        points = h.get("percentile_points") or []
        if points:
            for pt in points:
                rows.append(
                    {
                        "event_ticker": event_ticker,
                        "percentile": _f(pt.get("percentile")),
                        "numerical_forecast": _f(
                            pt.get("numerical_forecast") or pt.get("raw_numerical_forecast")
                        ),
                        "end_period_ts": from_timestamp(int(ep)),
                        "ingested_at": utc_now(),
                    }
                )
        else:
            # legacy flat shape tolerated
            rows.append(
                {
                    "event_ticker": event_ticker,
                    "percentile": _f(h.get("percentile")),
                    "numerical_forecast": _f(
                        h.get("numerical_forecast") or h.get("raw_numerical_forecast")
                    ),
                    "end_period_ts": from_timestamp(int(ep)),
                    "ingested_at": utc_now(),
                }
            )
    return rows


def forecast_percentile_frame(
    client: KalshiClient,
    event_ticker: str,
    series_ticker: str,
    *,
    start: datetime,
    end: datetime,
    period_interval_s: int = 60,
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch forecast percentile history for one event's [start, end] window."""
    raw_rows = client.get_event_forecast_percentile_history(
        event_ticker,
        series_ticker,
        start=start,
        end=end,
        period_interval_s=period_interval_s,
    )
    rows = [r for r in _flatten(raw_rows, event_ticker) if r["end_period_ts"] is not None]
    if save_raw and raw_rows:
        lake = getattr(client, "_lake", None)
        if lake is not None:
            lake.save_raw_jsonl("kalshi", f"forecast_percentiles/{event_ticker}", raw_rows)
    return pl.DataFrame(rows, schema=FORECAST_PERCENTILE_SCHEMA)
