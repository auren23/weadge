"""Time handling: the as-of invariant lives here.

weadge's single most important rule:

    a forecast may influence a decision at time T only if
        forecast.available_at <= T

run_init_at (model initialization) is NOT knowable to a researcher at T,
so it can never be used as the gate. available_at is the earliest moment
the outside world could actually obtain the data; ingested_at is when
weadge actually captured it.

All datetimes in weadge are timezone-aware UTC. Naive datetimes are a bug.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# Canonical datetime: timezone-aware, normalized to UTC.
UtcDatetime = datetime

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Coerce any aware datetime to UTC; reject naive datetimes."""
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime is a bug in weadge: {dt!r} — must be timezone-aware")
    return dt.astimezone(UTC)


def utc_now() -> datetime:
    """Current UTC time (the only allowed source of 'now')."""
    return datetime.now(UTC)


def from_timestamp(ts: float | int) -> datetime:
    """Unix epoch seconds -> UTC datetime (handles both s and ms heuristically)."""
    if isinstance(ts, (int, float)):
        # Kalshi returns epoch seconds in some endpoints and milliseconds in others.
        if abs(ts) > 1e11:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=UTC)
    raise TypeError(f"expected int/float timestamp, got {type(ts)}")


def to_timestamp(dt: datetime, ms: bool = False) -> int:
    """UTC datetime -> epoch (seconds or ms)."""
    dt = ensure_utc(dt)
    seconds = int((dt - _EPOCH).total_seconds())
    return seconds * 1000 if ms else seconds


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string, enforcing UTC normalization."""
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def floor_to_interval(dt: datetime, interval: timedelta) -> datetime:
    """Floor a timestamp to the start of its bucket (used for candle alignment)."""
    dt = ensure_utc(dt)
    epoch = to_timestamp(dt)
    bucket = (epoch // int(interval.total_seconds())) * int(interval.total_seconds())
    return from_timestamp(bucket)


def shift(dt: datetime, **kwargs: float) -> datetime:
    """Shift a datetime by timedelta kwargs (e.g. hours=-6)."""
    return ensure_utc(dt) + timedelta(**kwargs)
