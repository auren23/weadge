"""Historical fee replay.

Architecture rule: fees are NEVER a hardcoded constant. Kalshi provides
series.fee_multiplier + /series/{ticker}/fee_changes (with show_historical);
the backtest looks up the multiplier actually in effect at execution time.

Kalshi fee model (official fee schedule, effective 2026-07-07):

    Fee_taker = round_up_to_cent( M * 0.07   * C * P * (1 - P) )
    Fee_maker = round_up_to_cent( M * 0.0175 * C * P * (1 - P) )

M is the API `fee_multiplier` (usually 1.0 — never a base rate), C is the
number of contracts, P is the price in dollars on [0, 1]. The fee is rounded
UP to the nearest cent, applied to the total (not per contract).

Official taker examples (M=1, C=100):
    P=0.10 -> $0.63 ; P=0.25 -> $1.32 ; P=0.50 -> $1.75 ; P=0.90 -> $0.63
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import polars as pl

from weadge.domain.time import ensure_utc

# Base rates by fee type, from the 2026-07-07 fee schedule.
FEE_BASE_RATES: dict[str, float] = {"taker": 0.07, "maker": 0.0175}
FEE_FORMULA_VERSION = "2026-07-07"

FeeType = Literal["taker", "maker"]

# Sentinel effective_at for fallback regimes: strictly before any real change.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _fallback_change(multiplier: float | None, fee_type: str) -> FeeChange | None:
    """Build a fallback FeeChange (or None) from a flat multiplier."""
    if multiplier is None:
        return None
    return FeeChange(effective_at=_EPOCH, fee_type=fee_type, multiplier=multiplier)


@dataclass(frozen=True)
class FeeChange:
    """One fee regime: (fee_type, API multiplier M) in effect from effective_at."""

    effective_at: datetime
    fee_type: FeeType
    multiplier: float  # the API's fee_multiplier (M), usually 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_at", ensure_utc(self.effective_at))
        if self.fee_type not in FEE_BASE_RATES:
            raise ValueError(f"unknown fee_type {self.fee_type!r}, expected one of {list(FEE_BASE_RATES)}")
        if self.multiplier is None or self.multiplier <= 0:
            raise ValueError(f"fee multiplier M must be > 0, got {self.multiplier}")


def _coerce_change(item: Any) -> FeeChange:
    """Accept FeeChange or (effective_at, fee_type, multiplier) tuples."""
    if isinstance(item, FeeChange):
        return item
    if isinstance(item, (tuple, list)):
        if len(item) != 3:
            raise ValueError(f"fee change must be (effective_at, fee_type, multiplier), got {item!r}")
        eff_at, fee_type, mult = item
        return FeeChange(effective_at=eff_at, fee_type=str(fee_type), multiplier=float(mult))
    raise TypeError(f"unsupported fee change: {item!r}")


def round_up_to_cent(dollars: float) -> float:
    """Kalshi rounding: fee rounded UP to the nearest cent, on the total."""
    return math.ceil(dollars * 100 - 1e-9) / 100


class FeeSchedule:
    """Step function of (fee_type, multiplier) over time for one series."""

    formula_version: str = FEE_FORMULA_VERSION

    def __init__(
        self,
        changes: list[FeeChange | tuple[Any, Any, Any]],
        fallback: FeeChange | None = None,
    ) -> None:
        self.changes = sorted((_coerce_change(c) for c in changes), key=lambda c: c.effective_at)
        self.fallback = fallback

    @classmethod
    def from_frame(
        cls,
        fee_changes: pl.DataFrame,
        fallback_multiplier: float | None = None,
        fallback_fee_type: str = "taker",
    ) -> FeeSchedule:
        rows = []
        for r in fee_changes.iter_rows(named=True):
            if r.get("effective_at") is None or r.get("fee_multiplier") is None:
                continue
            rows.append(
                FeeChange(
                    effective_at=r["effective_at"],
                    fee_type=str(r.get("fee_type") or fallback_fee_type),
                    multiplier=float(r["fee_multiplier"]),
                )
            )
        fallback = _fallback_change(fallback_multiplier, fallback_fee_type)
        return cls(rows, fallback=fallback)

    @classmethod
    def from_series_metadata(cls, fee_multiplier: float | None, fee_type: str = "taker") -> FeeSchedule:
        """Flat schedule when no fee history exists yet (fallback)."""
        return cls([], fallback=_fallback_change(fee_multiplier, fee_type))

    def change_at(self, ts: datetime) -> FeeChange | None:
        """(fee_type, multiplier) in effect at `ts`: last change at or before ts."""
        ts = ensure_utc(ts)
        current: FeeChange | None = self.fallback
        for c in self.changes:
            if c.effective_at <= ts:
                current = c
            else:
                break
        return current

    def fee_cost(self, price: float, ts: datetime, contracts: float = 1.0) -> float:
        """Dollar fee paid by a taker for `contracts` contracts at `price`.

        Fee = round_up_to_cent(M * base_rate(fee_type) * contracts * P * (1 - P)).
        """
        change = self.change_at(ts)
        if change is None:
            raise ValueError(f"no fee regime in effect at {ts} and no fallback configured")
        base = FEE_BASE_RATES[change.fee_type]
        raw = change.multiplier * base * contracts * price * (1.0 - price)
        return round_up_to_cent(raw)

    def __repr__(self) -> str:  # pragma: no cover
        return f"FeeSchedule(changes={len(self.changes)}, fallback={self.fallback})"


def series_fee_schedule(
    series_meta: dict[str, Any] | None,
    fee_changes: pl.DataFrame | None = None,
) -> FeeSchedule:
    """Build a schedule from series metadata + optional fee change history.

    series_meta["fee_multiplier"] is the API multiplier M (default 1.0),
    series_meta["fee_type"] is "taker" (default) or "maker".
    """
    fallback_multiplier = None
    fallback_fee_type = "taker"
    if series_meta:
        if series_meta.get("fee_multiplier") is not None:
            fallback_multiplier = float(series_meta["fee_multiplier"])
        if series_meta.get("fee_type"):
            fallback_fee_type = str(series_meta["fee_type"])
    if fee_changes is not None and not fee_changes.is_empty():
        return FeeSchedule.from_frame(
            fee_changes,
            fallback_multiplier=fallback_multiplier,
            fallback_fee_type=fallback_fee_type,
        )
    return FeeSchedule.from_series_metadata(fallback_multiplier, fallback_fee_type)
