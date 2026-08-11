"""Typed Kalshi market-data adapters: series / events / markets -> canonical frames.

Every adapter returns a polars frame in the canonical schema (storage.schema)
and can optionally persist the raw JSON payload into the data lake.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient
from weadge.domain.time import ensure_utc, from_timestamp
from weadge.storage.schema import (
    EVENT_SCHEMA,
    MARKET_SCHEMA,
    SERIES_SCHEMA,
)


def _coerce_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dt(v: Any) -> datetime | None:
    if v is None or v == "":
        return None
    return from_timestamp(int(v))


def _target_date_to_utc(date_str: str) -> datetime | None:
    """'2026-08-12' -> UTC midnight. Best-effort: the true settlement window
    is local-time-based; the settlement oracle fixes this later."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def series_frame(client: KalshiClient, series_ticker: str, *, save_raw: bool = False) -> pl.DataFrame:
    """Fetch series metadata for one ticker, return canonical frame."""
    raw = client.get_series(series_ticker)
    row = {
        "series_ticker": series_ticker,
        "title": raw.get("title", ""),
        "settlement_source": raw.get("settlement_source", ""),
        "fee_type": raw.get("fee_type", ""),
        "fee_multiplier": _coerce_float(raw.get("fee_multiplier")),
        "ingested_at": ensure_utc(datetime.now(UTC)),
    }
    df = pl.DataFrame([row], schema=SERIES_SCHEMA)
    if save_raw:
        _save_raw(client, f"series/{series_ticker}.json", raw)
    return df


def events_frame(
    client: KalshiClient,
    series_ticker: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    status: str | None = None,
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch all events for a series, return canonical events frame."""
    raw_rows = client.get_events(series_ticker, start=start, end=end, status=status)
    rows = [
        {
            "event_ticker": e.get("event_ticker", ""),
            "series_ticker": series_ticker,
            "target_date": _target_date_to_utc(e.get("target_date", "")),
            "location_id": e.get("location", "") or series_ticker,
            "ingested_at": ensure_utc(datetime.now(UTC)),
        }
        for e in raw_rows
    ]
    if save_raw and raw_rows:
        _save_raw(client, f"events/{series_ticker}.jsonl", raw_rows)
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)


def markets_frame(
    client: KalshiClient,
    *,
    series_ticker: str | None = None,
    event_ticker: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    status: str | None = None,
    save_raw: bool = False,
) -> pl.DataFrame:
    """Fetch markets (buckets) matching the filter, return canonical markets frame."""
    raw_rows = client.get_markets(
        series_ticker=series_ticker,
        event_ticker=event_ticker,
        start=start,
        end=end,
        status=status,
    )
    rows = [
        {
            "market_ticker": m.get("ticker", ""),
            "event_ticker": m.get("event_ticker", ""),
            "series_ticker": m.get("series_ticker", "") or series_ticker or "",
            "floor_strike": _coerce_float(m.get("floor_strike")),
            "cap_strike": _coerce_float(m.get("cap_strike")),
            "open_at": _dt(m.get("open_time")),
            "close_at": _dt(m.get("close_time")),
            "settled_at": _dt(m.get("settlement_time")),
            "result": m.get("result") or None,
            "settlement_value": m.get("settlement_value") or None,
            "rules_primary": m.get("rules_primary", ""),
            "rules_secondary": m.get("rules_secondary", ""),
            "ingested_at": ensure_utc(datetime.now(UTC)),
        }
        for m in raw_rows
    ]
    if save_raw and raw_rows:
        _save_raw(client, f"markets/{event_ticker or series_ticker or 'all'}.jsonl", raw_rows)
    return pl.DataFrame(rows, schema=MARKET_SCHEMA)


def _save_raw(client: KalshiClient, key: str, payload: Any) -> None:
    """Persist raw JSON to the lake when the client carries a lake handle."""
    lake = getattr(client, "_lake", None)
    if lake is None:
        return

    rows = payload if isinstance(payload, list) else [payload]
    lake.save_raw_jsonl("kalshi", key.rsplit("/", 1)[0], rows)
