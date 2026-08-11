"""Forecast domain model.

The three timestamps are distinct and must never be conflated:

    run_init_at  — model initialization time (e.g. 00Z GFS run)
    available_at — earliest moment the outside world could obtain the data
    ingested_at  — when weadge actually captured it

Backtests only ever gate on `available_at`. Raw model payloads are always
kept (raw_payload_path) so future recalibration does not require re-download.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from weadge.domain.time import ensure_utc

Source = Literal["nbm", "gefs", "kalshi_forecast", "emos", "metar", "openmeteo"]


class Distribution(BaseModel):
    """Summary distribution of a temperature forecast (Fahrenheit unless unit set)."""

    mean: float | None = None
    std: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    unit: str = "fahrenheit"

    @field_validator("std")
    @classmethod
    def _std_nonneg(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"std must be >= 0, got {v}")
        return v


class ForecastSnapshot(BaseModel):
    """One model's forecast valid for a location window, frozen in time.

    Invariant (enforced by `validate_asof`):
        run_init_at <= available_at <= ingested_at <= valid_start
    """

    source: Source
    model: str
    model_version: str = ""
    run_id: str = ""

    run_init_at: datetime
    available_at: datetime
    ingested_at: datetime

    valid_start: datetime
    valid_end: datetime

    location_id: str = ""          # e.g. "KXHIGHNY" or "KNYC"
    lat: float | None = None
    lon: float | None = None
    station_id: str = ""

    distribution: Distribution | None = None

    raw_payload_path: str = ""     # pointer into data/raw — never inline the payload

    @field_validator("run_init_at", "available_at", "ingested_at", "valid_start", "valid_end")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @field_validator("available_at")
    @classmethod
    def _avail_after_init(cls, v: datetime, info) -> datetime:  # type: ignore[no-untyped-def]
        run_init = info.data.get("run_init_at")
        if run_init is not None and v < run_init:
            raise ValueError(f"available_at {v} < run_init_at {run_init}")
        return v

    @field_validator("ingested_at")
    @classmethod
    def _ingested_after_avail(cls, v: datetime, info) -> datetime:  # type: ignore[no-untyped-def]
        avail = info.data.get("available_at")
        if avail is not None and v < avail:
            raise ValueError(f"ingested_at {v} < available_at {avail}")
        return v

    @field_validator("valid_end")
    @classmethod
    def _valid_after_start(cls, v: datetime, info) -> datetime:  # type: ignore[no-untyped-def]
        start = info.data.get("valid_start")
        if start is not None and v <= start:
            raise ValueError(f"valid_end {v} <= valid_start {start}")
        return v

    def is_knowable_at(self, decision_at: datetime) -> bool:
        """THE as-of gate. A forecast may be used at `decision_at` iff it was
        available no later than `decision_at`."""
        return self.available_at <= ensure_utc(decision_at)


def validate_asof(decision_at: datetime, *forecasts: ForecastSnapshot) -> None:
    """Raise if any forecast leaks future information into `decision_at`."""
    decision_at = ensure_utc(decision_at)
    for fc in forecasts:
        if not fc.is_knowable_at(decision_at):
            raise LookaheadError(fc, decision_at)


class LookaheadError(ValueError):
    """Raised when a forecast would leak information beyond its available_at."""

    def __init__(self, forecast: ForecastSnapshot, decision_at: datetime) -> None:
        self.forecast = forecast
        self.decision_at = decision_at
        super().__init__(
            f"lookahead: forecast available_at={forecast.available_at} used at decision_at={decision_at}"
        )
