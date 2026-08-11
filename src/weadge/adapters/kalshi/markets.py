"""Typed Kalshi market-data adapters: series / events / markets -> canonical frames.

Every adapter returns a polars frame in the canonical schema (storage.schema)
and can optionally persist the raw JSON payload into the data lake.

Verified wire notes (2026-08-11):
  * events carry `strike_date` (ISO datetime), NOT `target_date`. The target
    date is the strike's LOCAL calendar date in the series timezone (e.g.
    strike 2026-08-11T03:59:00Z = 23:59 EDT Aug 10 -> target Aug 10).
  * markets carry ISO datetime strings (open_time/close_time), `settlement_ts`
    (epoch) and `settlement_value_dollars`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from weadge.adapters.kalshi.client import KalshiClient, parse_api_ts
from weadge.domain.time import ensure_utc
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
    return parse_api_ts(v)


def _strike_date_to_target(date_str: str, timezone: str = "America/New_York") -> datetime | None:
    """Kalshi event strike_date (ISO) -> local target date at UTC midnight.

    The event's `target_date` IS the local calendar date the market settles
    for; the strike instant expressed in the series timezone gives it
    directly (Kalshi stores strikes at the local midnight boundary).
    """
    if not date_str:
        return None
    try:
        strike = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        local_day = strike.astimezone(ZoneInfo(timezone)).date()
        return datetime(local_day.year, local_day.month, local_day.day, tzinfo=UTC)
    except (ValueError, TypeError):
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
            "target_date": _strike_date_to_target(e.get("strike_date", "")),
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
            "settled_at": _dt(m.get("settlement_ts")),
            "result": m.get("result") or None,
            "settlement_value": m.get("settlement_value_dollars") or m.get("settlement_value"),
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
