"""Market domain model (canonical schema).

Series metadata (settlement_source, fee_type, fee_multiplier) comes from the
Kalshi series endpoint — it is data, never hardcoded constants.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator

from weadge.domain.time import ensure_utc


class SeriesInfo(BaseModel):
    series_ticker: str
    title: str = ""
    settlement_source: str = ""
    fee_type: str = ""              # e.g. "taker"
    fee_multiplier: float | None = None

    @field_validator("fee_multiplier")
    @classmethod
    def _fee_nonneg(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"fee_multiplier must be >= 0, got {v}")
        return v


class EventInfo(BaseModel):
    event_ticker: str
    series_ticker: str = ""
    target_date: datetime | None = None
    location_id: str = ""

    @field_validator("target_date")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return ensure_utc(v) if v is not None else None


class MarketInfo(BaseModel):
    """A single binary contract (bucket) within an event."""

    market_ticker: str
    event_ticker: str = ""
    series_ticker: str = ""

    floor_strike: float | None = None
    cap_strike: float | None = None   # None cap => "at least floor" bucket

    open_at: datetime | None = None
    close_at: datetime | None = None
    settled_at: datetime | None = None

    result: str | None = None                  # "yes" | "no"
    settlement_value: str | None = None        # Kalshi's settlement string

    rules_primary: str = ""
    rules_secondary: str = ""

    @field_validator("open_at", "close_at", "settled_at")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return ensure_utc(v) if v is not None else None

    @property
    def bucket_low(self) -> float | None:
        return self.floor_strike

    @property
    def bucket_high(self) -> float | None:
        # Kalshi cap strike is exclusive upper bound: [floor, cap)
        return self.cap_strike

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class Quote1m(BaseModel):
    """One 1-minute candlestick of the YES side, as served by Kalshi."""

    market_ticker: str
    ts: datetime                       # bucket start (UTC)
    yes_bid_open: float | None = None
    yes_bid_high: float | None = None
    yes_bid_low: float | None = None
    yes_bid_close: float | None = None
    yes_ask_open: float | None = None
    yes_ask_high: float | None = None
    yes_ask_low: float | None = None
    yes_ask_close: float | None = None
    volume: int = 0
    open_interest: int = 0

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @property
    def mid_close(self) -> float | None:
        b, a = self.yes_bid_close, self.yes_ask_close
        if b is None or a is None:
            return None
        return (b + a) / 2.0
