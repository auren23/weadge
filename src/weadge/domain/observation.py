"""Official observations (settlement oracle inputs).

The observation is the ground truth the market settles against. weadge always
stores BOTH the Kalshi settlement result and the official observation, so the
settlement audit can catch data-quality mismatches before any research happens.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

from weadge.domain.time import ensure_utc


class Observation(BaseModel):
    station_id: str
    observed_at: datetime            # local "target date" boundary in UTC
    value: float                     # observed value in `unit`
    unit: str = "fahrenheit"
    source: str = ""                 # e.g. "NWS ASOS"

    @field_validator("observed_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)
