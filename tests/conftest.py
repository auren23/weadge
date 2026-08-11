"""Shared synthetic-data helpers for tests.

All fixtures are timezone-aware UTC datetimes (weadge rejects naive times).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from weadge.domain.time import shift


@pytest.fixture
def t0() -> datetime:
    return datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def make_quotes(market_ticker: str, start: datetime, n: int, *, step_min: int = 1,
                bid_base: float = 0.4, ask_base: float = 0.45) -> pl.DataFrame:
    """Synthetic 1m candles with slow price drift toward 0.5 over `n` bars."""
    rows = []
    ts = start
    for i in range(n):
        drift = i / max(n, 1) * 0.1
        rows.append(
            {
                "market_ticker": market_ticker,
                "ts": ts,
                "yes_bid_open": bid_base + drift,
                "yes_bid_high": bid_base + drift + 0.01,
                "yes_bid_low": bid_base + drift - 0.01,
                "yes_bid_close": bid_base + drift,
                "yes_ask_open": ask_base + drift,
                "yes_ask_high": ask_base + drift + 0.01,
                "yes_ask_low": ask_base + drift - 0.01,
                "yes_ask_close": ask_base + drift,
                "volume": 100 + i,
                "open_interest": 500,
                "ingested_at": ts + timedelta(minutes=1),
            }
        )
        ts = shift(ts, minutes=step_min)
    return pl.DataFrame(rows)


def make_forecast_snapshots(
    location_id: str,
    valid_start: datetime,
    *,
    sources: tuple[str, ...] = ("nbm",),
    base_mean: float = 90.0,
    base_std: float = 2.0,
    avail_offset_min: int = 15,
) -> pl.DataFrame:
    """One NBM forecast per run, availability = run_init + offset."""
    rows = []
    for src in sources:
        run_init = shift(valid_start, hours=-30)
        avail = shift(run_init, minutes=avail_offset_min)
        rows.append(
            {
                "source": src,
                "model": "nbm_v5" if src == "nbm" else src,
                "model_version": "5.0",
                "run_id": f"{src}-{run_init.isoformat()}",
                "run_init_at": run_init,
                "available_at": avail,
                "ingested_at": shift(avail, minutes=1),
                "valid_start": valid_start,
                "valid_end": shift(valid_start, hours=24),
                "location_id": location_id,
                "station_id": "KNYC",
                "lat": 40.7790,
                "lon": -73.9692,
                "mean": base_mean,
                "std": base_std,
                "p10": base_mean - 1.28 * base_std,
                "p25": base_mean - 0.67 * base_std,
                "p50": base_mean,
                "p75": base_mean + 0.67 * base_std,
                "p90": base_mean + 1.28 * base_std,
                "raw_payload_path": f"data/raw/{src}/{run_init.date()}.grib2",
            }
        )
    return pl.DataFrame(rows)


def make_percentile_history(
    event_ticker: str,
    *,
    periods: int = 3,
    period_step_min: int = 60,
    start: datetime | None = None,
    center: float = 90.0,
) -> pl.DataFrame:
    """Kalshi forecast percentile history: `periods` buckets x 5 percentiles."""
    rows = []
    ts = start or datetime(2026, 6, 30, 0, 0, tzinfo=UTC)
    pcts = {10: -1.28, 25: -0.67, 50: 0.0, 75: 0.67, 90: 1.28}
    for _p in range(periods):
        for pct, z in pcts.items():
            rows.append(
                {
                    "event_ticker": event_ticker,
                    "percentile": float(pct),
                    "numerical_forecast": center + z * 2.0,
                    "end_period_ts": ts,
                    "ingested_at": shift(ts, minutes=1),
                }
            )
        ts = shift(ts, minutes=period_step_min)
    return pl.DataFrame(rows)
