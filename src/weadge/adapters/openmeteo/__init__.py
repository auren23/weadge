"""Open-Meteo adapter — lightweight observation/forecast fallback (optional).

Only used for exploratory convenience data (e.g. quick station checks).
It is NOT a settlement source and must never be used as ground truth.
"""

from __future__ import annotations

from weadge.domain.forecast import ForecastSnapshot  # re-export for typing


def fetch_historical_daily(station: str, start: str, end: str) -> list[float]:
    """Placeholder: Open-Meteo historical API daily max temperatures."""
    raise NotImplementedError(
        "openmeteo is an exploratory convenience adapter (v1+); "
        "it must never feed the settlement oracle."
    )


__all__ = ["ForecastSnapshot", "fetch_historical_daily"]
