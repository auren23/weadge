"""Settlement oracle.

Priority above weather models: before ANY alpha research, weadge must prove it
can reproduce Kalshi's settlement from official observations. The audit keeps
BOTH the Kalshi result and the official observation, and reports any
mismatch. Research is frozen while mismatch > 0.

Settlement truth is the NWS Daily Climate Report (CLINYC for Central Park) —
the final daily maximum Kalshi's weather markets settle on. Hourly METAR
observations serve same-day research only and are NEVER accepted as
settlement truth: the oracle ignores any observation whose `source` does not
match the spec, so feeding METAR silently produces `missing` markets and
freezes research instead of faking a settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import polars as pl
from pydantic import BaseModel

from weadge.domain.time import ensure_utc

# NWS Daily Climate Report day windows are midnight-to-midnight LOCAL
# STANDARD time. In UTC the window for report date D is always
# [D 05:00, D+1 05:00), summer or winter: during DST the report day runs
# 01:00 EDT -> next 00:59 EDT, and a DST-aware local calendar day would
# misplace the 00:00-01:00 EDT hour. Bucketing with a fixed UTC-5 offset
# reproduces the report exactly.
_EST_OFFSET_HOURS = 5


class SettlementSpec(BaseModel):
    """How a series' settlement value is computed from observations."""

    series: str
    location: str = ""
    station_id: str = ""
    timezone: str = "America/New_York"
    source: str = "NWS Daily Climate Report"  # the ONLY trusted settlement truth
    day_window: Literal["local_standard"] = "local_standard"
    rounding: float = 0.5  # half-open buckets [floor, cap); strikes are integers

    def settlement_day(self, dt_utc: datetime) -> date:
        """Report date owning an observation timestamp (local-standard window).

        obs in [D 05:00 UTC, D+1 05:00 UTC) belongs to report date D,
        which equals (obs - 5h) in UTC calendar days. During DST this maps
        00:00-01:00 EDT to the PREVIOUS report day, matching the NWS
        standard-time window that Kalshi settles on.
        """
        return (ensure_utc(dt_utc) - timedelta(hours=_EST_OFFSET_HOURS)).date()


@dataclass(frozen=True)
class AuditReport:
    events_checked: int
    markets_checked: int
    matched: int
    mismatched: int
    missing: int

    @property
    def clean(self) -> bool:
        return self.mismatched == 0 and self.missing == 0

    def __str__(self) -> str:
        return (
            f"events checked      {self.events_checked}\n"
            f"markets checked     {self.markets_checked}\n"
            f"matched             {self.matched}\n"
            f"mismatched          {self.mismatched}\n"
            f"missing             {self.missing}"
        )


def bucket_hit(value: float, floor_strike: float | None, cap_strike: float | None) -> bool:
    """Does `value` fall in the half-open bucket [floor, cap)?"""
    if floor_strike is not None and value < floor_strike:
        return False
    return not (cap_strike is not None and value >= cap_strike)


class SettlementOracle:
    """Recompute the expected market result from official observations."""

    def __init__(
        self,
        spec: SettlementSpec,
        events: pl.DataFrame,
        markets: pl.DataFrame,
        observations: pl.DataFrame,
    ) -> None:
        self.spec = spec
        self.events = events
        self.markets = markets
        self.observations = observations

    def _event_date_map(self) -> dict[str, date]:
        """event_ticker -> local target date.

        Kalshi's event `target_date` IS the local calendar date the market
        settles for (e.g. "2026-07-01" = July 1 in the series' timezone), so
        we take its date component directly instead of converting the UTC
        midnight we stored it at.
        """
        out: dict[str, date] = {}
        for row in self.events.iter_rows(named=True):
            td = row.get("target_date")
            if td is not None:
                out[str(row["event_ticker"])] = td.date()
        return out

    def _observation_map(self) -> dict[date, float]:
        """report date -> observed value (latest per date wins).

        ONLY rows whose `source` matches the spec count as settlement truth.
        METAR or any other source is ignored: settlement can never be derived
        from hourly observations (their daily max is not the report's daily
        max, and their day window does not match the report's).
        """
        out: dict[date, float] = {}
        for row in self.observations.iter_rows(named=True):
            if str(row.get("station_id", "")) != self.spec.station_id:
                continue
            if str(row.get("source", "")) != self.spec.source:
                continue
            obs_at = row.get("observed_at")
            if obs_at is None:
                continue
            report_day = self.spec.settlement_day(obs_at)
            out[report_day] = float(row["value"])
        return out

    def audit(self) -> AuditReport:
        """Compare expected settlement vs Kalshi's stored `result`."""
        event_dates = self._event_date_map()
        obs = self._observation_map()

        markets: list[tuple[str, str, date | None, float | None, float | None, str | None]] = []
        for m in self.markets.iter_rows(named=True):
            ev = str(m.get("event_ticker", ""))
            markets.append(
                (
                    str(m["market_ticker"]),
                    ev,
                    event_dates.get(ev),
                    m.get("floor_strike"),
                    m.get("cap_strike"),
                    m.get("result"),
                )
            )

        checked_events: set[str] = set()
        matched = mismatched = missing = 0
        for _ticker, ev, local_day, low, high, kalshi_result in markets:
            if local_day is None or kalshi_result is None:
                missing += 1
                continue
            if local_day not in obs:
                missing += 1
                continue
            checked_events.add(ev)
            expected = "yes" if bucket_hit(obs[local_day], low, high) else "no"
            if expected == kalshi_result:
                matched += 1
            else:
                mismatched += 1

        return AuditReport(
            events_checked=len(checked_events),
            markets_checked=len(markets),
            matched=matched,
            mismatched=mismatched,
            missing=missing,
        )


__all__ = ["AuditReport", "SettlementOracle", "SettlementSpec", "bucket_hit"]
