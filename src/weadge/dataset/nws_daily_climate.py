"""NWS Daily Climate Report (CLINYC) ingest — the settlement ground truth.

Kalshi weather markets settle on the FINAL daily maximum from the NWS
Daily Climate Report for Central Park (CLINYC / KNYC), never on a
METAR-derived max. This module is the only sanctioned path into the
settlement oracle's observations table:

    dataset/nws_daily_climate.py  -> audit ground truth (settlement oracle)
    observations/ (METAR)          -> same-day research only, NEVER audit

The report covers midnight-to-midnight LOCAL STANDARD time: during DST the
window is 01:00 EDT -> next 00:59 EDT, i.e. [05:00 UTC, next 05:00 UTC)
all year round.

Raw DCR records are one station-day: {"report_date": date, "value": float}.
`observed_at` is stamped at the window midpoint (report_date 12:00 EST ==
17:00 UTC) so that SettlementSpec.settlement_day maps it back to the exact
report date in any season.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any

import polars as pl

from weadge.domain.time import utc_now
from weadge.storage.schema import OBSERVATION_SCHEMA

DCR_SOURCE = "NWS Daily Climate Report"


def daily_climate_frame(
    records: list[dict[str, Any]],
    *,
    station_id: str = "KNYC",
    source: str = DCR_SOURCE,
) -> pl.DataFrame:
    """Normalize raw DCR records -> OBSERVATION_SCHEMA frame.

    records: [{"report_date": date (or datetime), "value": float}, ...]
    One row per station-day; observed_at = window midpoint (17:00 UTC).
    """
    rows = []
    for r in records:
        report_date = r["report_date"]
        if isinstance(report_date, datetime):
            report_date = report_date.date()
        observed_at = datetime.combine(report_date, time(12, 0), tzinfo=UTC) + timedelta(hours=5)
        rows.append(
            {
                "station_id": station_id,
                "observed_at": observed_at,
                "value": float(r["value"]),
                "unit": str(r.get("unit", "fahrenheit")),
                "source": source,
                "ingested_at": utc_now(),
            }
        )
    return pl.DataFrame(rows, schema=OBSERVATION_SCHEMA)


__all__ = ["DCR_SOURCE", "daily_climate_frame"]
