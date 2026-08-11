"""NWS CLINYC Daily Climate Report ingest (G0 settlement truth).

DATA SOURCE DESIGN
    actual source:   NWS CLINYC Daily Climate Report (Central Park)
    transport:       IEM NWS Text Product Archive (https://mesonet.agron.iastate.edu)
                     IEM stores the original NWS text bulletins, which the
                     forecast.weather.gov product viewer only keeps ~168h.
    observations always carry source="NWS Daily Climate Report"; IEM is only
    an acquisition detail (kept in raw metadata as acquired_via).

PRODUCT SHAPE (verified on real July 2026 bulletins)
    per report date there are two CLINYC products:
      06:20Z  -> full daily for YESTERDAY (report date D-1 in local terms)
      20:35Z  -> preliminary, "VALID TODAY AS OF"
    Classification is CONTENT-BASED, never issuance-time based:
      is_preliminary = "VALID TODAY AS OF" in text
      is_complete    = "YESTERDAY" inside the TEMPERATURE section
                      (a bare text-wide search is WRONG: preliminaries also
                       mention YESTERDAY under PRECIPITATION)
    For one report_date with several complete versions (e.g. a correction),
    the LATEST issued_at wins.

MAXIMUM PARSING (fail closed)
    MAXIMUM         87       -> 87
    MAXIMUM         58R      -> 58   (record flag)
    MAXIMUM         -2       -> -2
    MAXIMUM         M / MM   -> None (missing — never guessed)
    anything else            -> None
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from weadge.dataset.nws_daily_climate import daily_climate_frame as _dataset_daily_climate_frame

PIL_CLINYC = "CLINYC"
ISSUING_CENTER = "OKX"
IEM_LIST_URL = "https://mesonet.agron.iastate.edu/wx/afos/list.phtml"
IEM_TEXT_URL = "https://mesonet.agron.iastate.edu/api/1/nwstext/{pid}"

IDENTITY_MARKER = "THE CENTRAL PARK NY CLIMATE SUMMARY"
_HEADING_RE = re.compile(
    r"THE CENTRAL PARK NY CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})"
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
     "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"])}
_TEMP_SECTION_RE = re.compile(r"TEMPERATURE \(F\)(.*?)(?:PRECIPITATION \(IN\)|$)", re.S)
_MAXIMUM_RE = re.compile(r"MAXIMUM\s+(-?\d+)R?(?=\s|$)")
_PID_RE = re.compile(r"pid=(\d{12}-[A-Z]{4}-[A-Z0-9]{6}-CLINYC)")


@dataclass(frozen=True)
class DCRRecord:
    """One canonical CLINYC product, parsed."""

    product_id: str
    pil: str
    issued_at: datetime
    report_date: date | None
    is_preliminary: bool
    is_complete: bool
    maximum_f: int | None
    raw_payload_path: str | None = None


def _issued_at_from_pid(product_id: str) -> datetime:
    """pid embeds the issue time as YYYYMMDDHHMM (UTC)."""
    m = re.match(r"(\d{12})", product_id)
    if not m:
        raise ValueError(f"product_id does not start with YYYYMMDDHHMM: {product_id!r}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC)


def parse_clinyc(text: str, product_id: str) -> DCRRecord:
    """Parse one CLINYC bulletin into a DCRRecord.

    Raises ValueError when the text is not a Central Park climate summary
    (identity validation — an archive/filter mixup must never be ingested).
    """
    if IDENTITY_MARKER not in text:
        raise ValueError(
            f"{product_id}: not a CLINYC summary — missing {IDENTITY_MARKER!r}; "
            "refusing to parse a foreign product"
        )

    heading = _HEADING_RE.search(text)
    report_date = None
    if heading:
        month, day, year = heading.group(1), int(heading.group(2)), int(heading.group(3))
        month_num = _MONTHS.get(month)
        if month_num is not None:
            report_date = date(year, month_num, day)

    is_preliminary = "VALID TODAY AS OF" in text
    temp_section = ""
    m = _TEMP_SECTION_RE.search(text)
    if m:
        temp_section = m.group(1)
    is_complete = "YESTERDAY" in temp_section

    maximum_f: int | None = None
    if is_complete:
        mm = _MAXIMUM_RE.search(temp_section)
        if mm:
            maximum_f = int(mm.group(1))

    return DCRRecord(
        product_id=product_id,
        pil=PIL_CLINYC,
        issued_at=_issued_at_from_pid(product_id),
        report_date=report_date,
        is_preliminary=is_preliminary,
        is_complete=is_complete,
        maximum_f=maximum_f,
    )


def select_final(records: list[DCRRecord]) -> dict[date, DCRRecord]:
    """Best complete version per report_date: latest issued_at wins, so a
    correction naturally supersedes the original daily."""
    final: dict[date, DCRRecord] = {}
    for r in records:
        if r.is_preliminary or not r.is_complete or r.report_date is None:
            continue
        cur = final.get(r.report_date)
        if cur is None or r.issued_at > cur.issued_at:
            final[r.report_date] = r
    return final


# ------------------------------------------------------------------ IEM fetch
class IEMClient:
    def __init__(self, timeout_s: float = 30.0, max_retries: int = 3) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s), follow_redirects=True)
        self.max_retries = max_retries

    def _get(self, url: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.get(url)
                if resp.status_code >= 500:
                    last = RuntimeError(f"{resp.status_code} on {url}")
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TransportError as exc:
                last = exc
        raise RuntimeError(f"failed to fetch {url} after {self.max_retries} retries") from last

    def list_pids(self, start: date, end: date) -> list[str]:
        """CLINYC product ids issued in [start, end].

        Chunks are 5 days: the IEM list page truncates long ranges (a 10-day
        query dropped whole days; 5-day queries return every product).
        """
        pids: list[str] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=4), end)
            url = (
                f"{IEM_LIST_URL}?by=cccc&source={ISSUING_CENTER}&pil={PIL_CLINYC}"
                f"&year={cursor.year}&month={cursor.month}&day={cursor.day}"
                f"&drange=yes&year2={chunk_end.year}&month2={chunk_end.month}"
                f"&day2={chunk_end.day}"
            )
            html = self._get(url).text
            for pid in _PID_RE.findall(html):
                if pid not in pids:
                    pids.append(pid)
            cursor = chunk_end + timedelta(days=1)
        return sorted(pids)

    def fetch_text(self, product_id: str) -> str:
        """Raw bulletin text. IEM prepends a one-line byte-length header;
        it is harmless (never matches the CLINYC identity) but stripped."""
        resp = self._get(IEM_TEXT_URL.format(pid=product_id))
        lines = resp.text.split("\n", 1)
        if len(lines) == 2 and lines[0].strip().isdigit():
            return lines[1]
        return resp.text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> IEMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------------ backfill
def backfill_dcr(
    start: date,
    end: date,
    lake,
    *,
    pil: str = PIL_CLINYC,
    station_id: str = "KNYC",
    client: IEMClient | None = None,
) -> dict[str, int]:
    """Fetch CLINYC bulletins, parse, select the final daily per report date,
    and append them to the observations table. Returns a summary dict.

    `end` is the last REPORT date wanted; the fetch window extends one day
    beyond it because the final daily for date D is issued on D+1 (~06:20Z).
    """
    own = client is None
    client = client or IEMClient()
    try:
        # the final daily for report date D is issued the NEXT morning
        # (~06:20Z), so the fetch window extends one day past `end`.
        fetch_end = end + timedelta(days=1)
        pids = client.list_pids(start, fetch_end)
        records: list[DCRRecord] = []
        raw_dir = Path(lake.root) / "raw" / "noaa" / "dcr" / pil
        raw_dir.mkdir(parents=True, exist_ok=True)
        for pid in pids:
            text = client.fetch_text(pid)
            # raw capture FIRST — the bulletin is the audit trail
            raw_path = raw_dir / f"{pid}.txt"
            raw_path.write_text(text, encoding="utf-8")
            try:
                rec = parse_clinyc(text, pid)
            except ValueError:
                rec = None
            if rec is not None:
                records.append(DCRRecord(
                    product_id=rec.product_id, pil=rec.pil, issued_at=rec.issued_at,
                    report_date=rec.report_date, is_preliminary=rec.is_preliminary,
                    is_complete=rec.is_complete, maximum_f=rec.maximum_f,
                    raw_payload_path=str(raw_path),
                ))
        final = select_final(records)
        obs = daily_climate_frame(
            [{"report_date": d, "value": r.maximum_f} for d, r in sorted(final.items())
             if r.maximum_f is not None],
            station_id=station_id,
        )
        if not obs.is_empty():
            lake.write_parquet("observations", obs, layer="bronze", partition_by="station_id")
        complete = [r for r in records if r.is_complete and not r.is_preliminary]
        return {
            "products_fetched": len(pids),
            "preliminary_rejected": sum(1 for r in records if r.is_preliminary),
            "complete_daily": len(complete),
            "corrections": len(complete) - len(final),
            "unique_report_days": len(final),
            "parsed_maximum": sum(1 for r in final.values() if r.maximum_f is not None),
            "missing_maximum": sum(1 for r in final.values() if r.maximum_f is None),
            "foreign_rejected": len(pids) - len(records),
        }
    finally:
        if own:
            client.close()


def daily_climate_frame(  # re-exported from dataset layer (canonical normalizer)
    records: list[dict],
    *,
    station_id: str = "KNYC",
    source: str = "NWS Daily Climate Report",
) -> pl.DataFrame:
    return _dataset_daily_climate_frame(records, station_id=station_id, source=source)


__all__ = [
    "DCRRecord",
    "IEMClient",
    "backfill_dcr",
    "daily_climate_frame",
    "parse_clinyc",
    "select_final",
]
