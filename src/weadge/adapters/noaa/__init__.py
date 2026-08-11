"""NOAA adapters — stubs for v1.

v0 needs NBM only at the dataset layer (p_nbm). The full GRIB pipeline
(xarray + cfgrib + ecCodes) is explicitly deferred: it is not required to
answer the v0 question, so it must not block the alpha existence test.
"""

from __future__ import annotations

from weadge.domain.forecast import ForecastSnapshot


def placeholder() -> None:
    """Document the intended v1 surface: nbm.py / observations.py / climate.py."""
    raise NotImplementedError(
        "NOAA GRIB ingestion is a v1 component (NBM v5 probabilistic products). "
        "v0 consumes pre-computed NBM percentiles via dataset/probability.py."
    )


__all__ = ["ForecastSnapshot", "placeholder"]
