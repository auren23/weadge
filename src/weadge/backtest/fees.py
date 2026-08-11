"""Historical fee replay.

Architecture rule: fees are NEVER a hardcoded constant. Kalshi provides
series.fee_multiplier + /series/{ticker}/fee_changes (with show_historical);
the backtest looks up the multiplier actually in effect at execution time.

Kalshi taker fee model (per series): fee = price * fee_multiplier.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from weadge.domain.time import ensure_utc


class FeeSchedule:
    """Step function of fee multiplier over time for one series."""

    def __init__(
        self,
        changes: list[tuple[datetime, float]],  # (effective_at, multiplier), ascending
        fallback_multiplier: float | None = None,
    ) -> None:
        self.changes = sorted((ensure_utc(t), float(m)) for t, m in changes)
        self.fallback = fallback_multiplier

    @classmethod
    def from_frame(
        cls,
        fee_changes: pl.DataFrame,
        fallback_multiplier: float | None = None,
    ) -> FeeSchedule:
        rows = [
            (r["effective_at"], r["fee_multiplier"])
            for r in fee_changes.iter_rows(named=True)
            if r.get("effective_at") is not None and r.get("fee_multiplier") is not None
        ]
        return cls(rows, fallback_multiplier=fallback_multiplier)

    @classmethod
    def from_series_metadata(cls, fee_multiplier: float | None) -> FeeSchedule:
        """Flat schedule when no fee history exists yet (fallback)."""
        return cls([], fallback_multiplier=fee_multiplier)

    def multiplier_at(self, ts: datetime) -> float | None:
        ts = ensure_utc(ts)
        # last change at or before ts
        current: float | None = self.fallback
        for eff_at, mult in self.changes:
            if eff_at <= ts:
                current = mult
            else:
                break
        return current

    def fee_cost(self, price: float, ts: datetime) -> float:
        """Dollar fee paid by a taker for one contract bought at `price`."""
        mult = self.multiplier_at(ts)
        if mult is None:
            raise ValueError(f"no fee multiplier in effect at {ts} and no fallback configured")
        return price * mult

    def __repr__(self) -> str:  # pragma: no cover
        return f"FeeSchedule(changes={len(self.changes)}, fallback={self.fallback})"


def series_fee_schedule(
    series_meta: dict[str, Any] | None,
    fee_changes: pl.DataFrame | None = None,
) -> FeeSchedule:
    """Build a schedule from series metadata + optional fee change history."""
    fallback = None
    if series_meta and series_meta.get("fee_multiplier") is not None:
        fallback = float(series_meta["fee_multiplier"])
    if fee_changes is not None and not fee_changes.is_empty():
        return FeeSchedule.from_frame(fee_changes, fallback_multiplier=fallback)
    return FeeSchedule.from_series_metadata(fallback)
