"""NOAA NBM archived probabilistic guidance — probe adapter (v0).

DATA CHAIN (all verified live on July 2026 data, AWS noaa-nbm-grib2-pds):

  * QMD files live at
        blend.YYYYMMDD/{CC}/qmd/blend.t{CC}z.qmd.f{HHH}.{domain}.grib2
    with sibling .idx files (record:byte_offset:d=...:VAR:level:fcst:desc).
  * The MaxT QMD (NBP station-card TXN*) is the TMP 2 m above ground
    "12-30 hour max fcst" family:
        bare desc      = QMD MEAN        (template 8, typeOfStatisticalProcessing 2)
        "StdDev fcst"  = QMD std dev
        "NN% level"    = percentile      (template 10)
  * The day-1 MaxT window of cycle CC is [CC+12h, CC+30h) — an 18-hour
    window ending 06Z; day-2 is [CC+36h, CC+54h) (f054), the SAME calendar
    window as the next run's day-1 but knowable 24h earlier. The day-D
    window [D 12Z, D+1 06Z) does NOT equal the Kalshi DCR settlement day
    [D 05Z, D+1 05Z): 17 of 24 hours overlap (7h late start, 1h late
    end). The mismatch is a recorded research fact (potential
    station/window-correction alpha), never silently papered over.
  * Cross-checks vs the NBP station card (same run): max_2t p50 at KNYC
    = 96.0°F == TXNP5; day-2/day-3 columns match within rounding. The
    card's TXNP1/TXNSD at KNYC day-1 (93/2) sit OUTSIDE the local GRIB
    field range (p10 field minimum at the station is 94.4) — a
    text-product artifact, documented and not chased; the GRIB QMD
    family is authoritative. At coastal stations (PHNL) the card tracks
    the local land-patch max rather than the nearest (often water) grid
    point; KNYC's nearest point is itself the right cell (matches the
    card and the 2026-07-15 observed 95F vs forecast mean 96F).
  * Stored units: mean/p10..p90 and std are °F — the unit of Kalshi
    strikes, DCR observations, and the NBP text product (a °C value fed
    against °F strikes silently dumps the whole distribution into the
    wrong bucket — the sum-to-1 guard cannot catch it).
  * Observed archive availability: the 00Z run's f030 qmd object appears
    ~07:15Z (S3 Last-Modified). That is AFTER the T-24h snapshot
    (D 04:59Z), so T-24h is served by the D-1 run's f054 (available
    D-1 ~07:15Z, same day-D window); T-12h and closer by the D run's
    f030. Both are ingested per run.

This module is the probe/ingest layer only; it writes nothing to research.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from weadge.domain.time import utc_now

BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
DOMAIN_CO = "co"
NBM_PERCENTILES = (10, 25, 50, 75, 90)

# NBM v5 sub-version schedule (official PNS): v5.0.14 went live 2026-07-28 12Z.
V5_0_14_RUN_INIT = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def model_version_for(run_init_at: datetime) -> str:
    """NBM model version for a run init (official schedule, not in the GRIB)."""
    return "nbm_v5.0.14" if run_init_at >= V5_0_14_RUN_INIT else "nbm_v5.0.x"


def qmd_url(run_date, cycle: int, fhour: int, domain: str = DOMAIN_CO) -> str:
    return (
        f"{BUCKET}/blend.{run_date:%Y%m%d}/{cycle:02d}/qmd/"
        f"blend.t{cycle:02d}z.qmd.f{fhour:03d}.{domain}.grib2"
    )


# ------------------------------------------------------------------- .idx
@dataclass(frozen=True)
class IdxRecord:
    record: int
    offset: int
    next_offset: int | None
    var: str
    level: str
    fcst: str
    desc: str

    def __repr__(self) -> str:  # pragma: no cover
        return f"IdxRecord({self.record}, {self.var} {self.level} {self.fcst} {self.desc!r})"


def parse_idx(text: str) -> list[IdxRecord]:
    """Parse an NBM .idx file: 'record:offset:d=...:VAR:level:fcst:desc'."""
    rows: list[IdxRecord] = []
    for ln in text.splitlines():
        p = ln.split(":")
        if len(p) < 6 or not p[0].isdigit():
            continue
        rows.append(
            IdxRecord(
                record=int(p[0]),
                offset=int(p[1]),
                next_offset=None,
                var=p[3],
                level=p[4],
                fcst=p[5],
                desc=":".join(p[6:]) if len(p) > 6 else "",
            )
        )
    for a, b in itertools.pairwise(rows):
        object.__setattr__(a, "next_offset", b.offset)
    return rows


def max_2t_records(rows: list[IdxRecord]) -> dict[str, IdxRecord]:
    """The MaxT QMD family of ONE qmd file: TMP 2m 'N-M hour max fcst' records.

    Returns {"mean": ..., "std": ..., 10: ..., 25: ..., 50: ..., ...} —
    percentile keys are ints. The std record carries fcst='N-M hour StdDev
    fcst' (not 'max fcst'); instantaneous 2t and MinT are excluded. A qmd
    file holds exactly one MaxT window (f030: 12-30, f054: 36-54,
    f078: 60-78, ...), so the window is not part of the key.
    """
    out: dict[str, IdxRecord] = {}
    for r in rows:
        if r.var != "TMP" or r.level != "2 m above ground":
            continue
        if "StdDev fcst" in r.fcst:
            if "hour" in r.fcst:
                out["std"] = r
            continue
        if "max fcst" not in r.fcst or "hour" not in r.fcst:
            continue
        if r.desc == "":
            out["mean"] = r
        else:
            m = re.fullmatch(r"(\d+)% level", r.desc)
            if m:
                out[int(m.group(1))] = r
    return out


# ------------------------------------------------------------------ decode
@dataclass
class DecodedField:
    name: str
    short_name: str
    percentile: int | None
    start_step: int
    end_step: int
    valid_at: datetime  # run_init + end_step
    stat_proc: int | None
    values: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


def _get(g: Any, key: str, default: Any = None) -> Any:
    try:
        return g[key]
    except Exception:  # pragma: no cover - missing key on unusual messages
        return default


def decode_blob(blob: bytes, run_init_at: datetime) -> list[DecodedField]:
    """Decode concatenated GRIB2 messages (range-fetched records).

    A temporary file is used because pygrib.fromstring only handles a
    single message; the blob usually contains several records.
    """
    import tempfile

    import pygrib

    out: list[DecodedField] = []
    with tempfile.NamedTemporaryFile(suffix=".grib2") as fh:
        fh.write(blob)
        fh.flush()
        for g in pygrib.open(fh.name):
            pct = _get(g, "percentileValue")
            stat = _get(g, "typeOfStatisticalProcessing")
            end = int(_get(g, "endStep", 0))
            lats, lons = g.latlons()
            out.append(
                DecodedField(
                    name=str(_get(g, "name", "?")),
                    short_name=str(_get(g, "shortName", "?")),
                    percentile=int(pct) if isinstance(pct, int) else None,
                    start_step=int(_get(g, "startStep", 0)),
                    end_step=end,
                    valid_at=run_init_at + timedelta(hours=end),
                    stat_proc=int(stat) if isinstance(stat, int) else None,
                    values=g.values,
                    lats=lats,
                    lons=lons,
                )
            )
    return out


def nearest_value(field: DecodedField, lat: float, lon: float) -> float:
    """Value at the nearest grid point to (lat, lon)."""
    dlon = np.abs(field.lons - lon)
    dlon = np.minimum(dlon, 360.0 - dlon)
    d = (field.lats - lat) ** 2 + (dlon * np.cos(np.radians(lat))) ** 2
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return float(field.values[j, i])


def neighborhood_max(field: DecodedField, lat: float, lon: float, radius: int = 3) -> float:
    """Max over a (2r+1)^2 patch — tolerates coastal stations whose nearest
    grid point falls on water."""
    dlon = np.abs(field.lons - lon)
    dlon = np.minimum(dlon, 360.0 - dlon)
    d = (field.lats - lat) ** 2 + (dlon * np.cos(np.radians(lat))) ** 2
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return float(field.values[j - radius : j + radius + 1, i - radius : i + radius + 1].max())


# ------------------------------------------------------------------ archive
class NbmArchive:
    """Range-fetched access to the NBM archive: only the needed records are
    downloaded (a full CONUS qmd file is ~650MB)."""

    def __init__(self, timeout_s: float = 90.0, max_retries: int = 4) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s), follow_redirects=True)
        self.max_retries = max_retries

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, url, **kwargs)
                if resp.status_code >= 500:
                    last = RuntimeError(f"{resp.status_code} on {url}")
                    time.sleep(0.5 * (2**attempt))
                    continue
                return resp
            except httpx.TransportError as exc:
                last = exc
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"failed {method} {url} after {self.max_retries} retries") from last

    def idx(self, run_date, cycle: int, fhour: int, domain: str = DOMAIN_CO) -> list[IdxRecord]:
        url = qmd_url(run_date, cycle, fhour, domain) + ".idx"
        resp = self._request("GET", url)
        resp.raise_for_status()
        return parse_idx(resp.text)

    def fetch_records(
        self, run_date, cycle: int, fhour: int, domain: str, records: list[IdxRecord]
    ) -> bytes:
        """Fetch the byte ranges of `records` and concatenate them into one
        valid (partial) GRIB2 blob."""
        url = qmd_url(run_date, cycle, fhour, domain)
        parts: list[bytes] = []
        for r in records:
            if r.next_offset is None:
                raise ValueError(f"record {r.record} has no end offset")
            resp = self._request(
                "GET", url, headers={"Range": f"bytes={r.offset}-{r.next_offset - 1}"}
            )
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(f"expected 206 partial content, got {resp.status_code}")
            parts.append(resp.content)
        return b"".join(parts)

    def max_2t_records_raw(
        self, run_date, cycle: int, fhour: int, domain: str = DOMAIN_CO
    ) -> tuple[bytes, dict[str, DecodedField]]:
        """Fetch and decode the full MaxT QMD family (mean/std/p10..p90) for
        one run and fhour. Returns (raw blob, decoded fields by name)."""
        rows = max_2t_records(self.idx(run_date, cycle, fhour, domain))
        names = ["mean", "std"] + [f"p{p}" for p in NBM_PERCENTILES]
        want: list[tuple[str, IdxRecord]] = []
        for name in names:
            key: Any = "std" if name == "std" else ("mean" if name == "mean" else int(name[1:]))
            r = rows.get(key)
            if r is not None:
                want.append((name, r))
        blob = self.fetch_records(run_date, cycle, fhour, domain, [r for _, r in want])
        run_init = datetime(run_date.year, run_date.month, run_date.day, cycle, tzinfo=UTC)
        fields = decode_blob(blob, run_init)
        return blob, {name: field for (name, _r), field in zip(want, fields, strict=True)}

    def max_2t_qmd(
        self,
        run_date,
        cycle: int,
        fhour: int,
        domain: str = DOMAIN_CO,
    ) -> dict[str, DecodedField]:
        """Decoded MaxT QMD fields only (the raw blob is discarded)."""
        _blob, fields = self.max_2t_records_raw(run_date, cycle, fhour, domain)
        return fields

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NbmArchive:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------- NBP cross-check
def nbp_max_t_values(text: str) -> dict[str, int]:
    """Parse a station-card section's MaxT QMD rows (TXNMN/TXNSD/TXNP*) into
    {key: value} — used to cross-check the GRIB extraction against the NBP
    text product of the same run. Only the FHR24 column is read (the first
    day column), matching the day-1 MaxT window."""
    out: dict[str, int] = {}
    for key in ("TXNMN", "TXNSD", "TXNP1", "TXNP2", "TXNP5", "TXNP7", "TXNP9"):
        m = re.search(rf"^\s*{key}\s+(\d+)", text, re.M)
        if m:
            out[key] = int(m.group(1))
    return out


# ------------------------------------------------------------------ backfill
MAXT_CYCLE = 0  # canonical choice: the D-00Z run
# MaxT QMD windows per run (18h diurnal span, ending 06Z):
#   f030 -> [init+12h, init+30h)  day-1 (the run's own day)
#   f054 -> [init+36h, init+54h)  day-2 (== the next run's day-1, 24h earlier)
MAXT_FHOURS = (30, 54)
MAXT_FHOUR = 30  # day-1 fhour (probe default)
NBM_WINDOW_START_HOUR = 12  # day-D MaxT window [D 12Z, D+1 06Z)


def _window_for(run_init: datetime, fhour: int) -> tuple[datetime, datetime]:
    """MaxT valid window of one run's fhour file: [init+fhour-18h, init+fhour)."""
    return run_init + timedelta(hours=fhour - 18), run_init + timedelta(hours=fhour)


def _f_to_f(kelvin: float) -> float:
    """Kelvin -> Fahrenheit (storage unit of the forecasts table: Kalshi
    strikes, DCR observations, and the NBP text product are all °F)."""
    return (kelvin - 273.15) * 9 / 5 + 32


def _object_last_modified(url: str, cache: dict[str, datetime], client: Any) -> datetime | None:
    """S3 Last-Modified of an object = observed archive availability."""
    if url in cache:
        return cache[url]
    resp = client._request("HEAD", url)
    if resp.status_code != 200:
        cache[url] = None
        return None
    lm = resp.headers.get("last-modified")
    if not lm:
        cache[url] = None
        return None
    dt = datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=UTC)
    cache[url] = dt
    return dt


def backfill_nbm(
    start: date,
    end: date,
    lake,
    *,
    series: str,
    station_id: str,
    lat: float,
    lon: float,
    archive: NbmArchive | None = None,
) -> dict[str, int]:
    """Fetch the MaxT QMD family for each target date and write FORECAST_SCHEMA rows.

    Canonical choice per target date D: the D-00Z run's f030 (day-1, window
    [D 12Z, D+1 06Z)) AND the (D-1)-00Z run's f054 (day-2 — the SAME window,
    knowable 24h earlier). Together they cover every snapshot of the smoke
    audit: T-24h is served by the D-1 run's f054 (available D-1 ~07:15Z),
    T-12h and closer by the D run's f030 (available D ~07:15Z). One leading
    day is fetched so the first event's T-24h has its D-1 run.

    availability = observed S3 Last-Modified, stored in the raw metadata
    sidecar — never guessed. Only if the HEAD fails do we fall back to
    run_init + 8h with availability_source='conservative_offset'. The NBM
    window is deliberately NOT the DCR settlement window — the difference
    is recorded in valid_start/valid_end, not aligned away.
    """
    own = archive is None
    archive = archive or NbmArchive()
    cache: dict[str, datetime] = {}
    rows: list[dict] = []
    summary = {
        "days_requested": 0,
        "f030_fetched": 0,
        "f030_missing": 0,
        "f054_fetched": 0,
        "f054_missing": 0,
        "v5_0_x": 0,
        "v5_0_14": 0,
    }
    try:
        day = start - timedelta(days=1)  # leading day for the first T-24h
        while day <= end:
            in_range = start <= day <= end
            if in_range:
                summary["days_requested"] += 1
            run_init = datetime(day.year, day.month, day.day, MAXT_CYCLE, tzinfo=UTC)
            version = model_version_for(run_init)
            for fhour in MAXT_FHOURS:
                if fhour == 30 and not in_range:
                    continue  # leading day only supplies the day-2 window
                try:
                    raw_blob, qmd = archive.max_2t_records_raw(day, MAXT_CYCLE, fhour)
                except httpx.HTTPStatusError:
                    summary[f"f{fhour:03d}_missing"] += 1
                    continue
                if not qmd or "p50" not in qmd:
                    summary[f"f{fhour:03d}_missing"] += 1
                    continue

                url = qmd_url(day, MAXT_CYCLE, fhour)
                observed_at = _object_last_modified(url, cache, archive)
                if observed_at is None:
                    available_at = run_init + timedelta(hours=8)
                    availability_source = "conservative_offset"
                else:
                    available_at, availability_source = observed_at, "observed"

                # raw capture: grib blob + metadata sidecar (availability truth)
                raw_dir = (
                    Path(lake.root)
                    / "raw"
                    / "noaa"
                    / "nbm"
                    / series
                    / f"{run_init:%Y%m%d%H}Z"
                    / f"f{fhour:03d}"
                )
                raw_dir.mkdir(parents=True, exist_ok=True)
                payload = raw_dir / f"qmd_f{fhour:03d}_co_records.grib2"
                payload.write_bytes(raw_blob)
                valid_start, valid_end = _window_for(run_init, fhour)
                raw_dir.joinpath("metadata.json").write_text(
                    json.dumps(
                        {
                            "source_url": url,
                            "product_created_at": (
                                available_at.isoformat()
                                if availability_source == "observed"
                                else None
                            ),
                            "availability_source": availability_source,
                            "model_version": version,
                            "valid_start": valid_start.isoformat(),
                            "valid_end": valid_end.isoformat(),
                            "percentiles": [10, 25, 50, 75, 90],
                            "window_note": "NBM MaxT day window [init+12h, init+30h) "
                            "(f030) / [init+36h, init+54h) (f054); differs "
                            "from the Kalshi DCR settlement day [D 05Z, "
                            "D+1 05Z) — recorded, not aligned",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                summary[f"f{fhour:03d}_fetched"] += 1
                rows.append(
                    {
                        "source": "nbm",
                        "model": "nbm",
                        "model_version": version,
                        "run_id": f"{run_init:%Y%m%d%H}Z",
                        "run_init_at": run_init,
                        "available_at": available_at,
                        "ingested_at": utc_now(),
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "location_id": series,
                        "station_id": station_id,
                        "lat": lat,
                        "lon": lon,
                        # stored in °F — the unit of Kalshi strikes, DCR
                        # observations, and the NBP text product
                        "mean": _f_to_f(nearest_value(qmd["mean"], lat, lon))
                        if "mean" in qmd
                        else None,
                        "std": nearest_value(qmd["std"], lat, lon) * 9 / 5
                        if "std" in qmd
                        else None,
                        "p10": _f_to_f(nearest_value(qmd["p10"], lat, lon)),
                        "p25": _f_to_f(nearest_value(qmd["p25"], lat, lon)),
                        "p50": _f_to_f(nearest_value(qmd["p50"], lat, lon)),
                        "p75": _f_to_f(nearest_value(qmd["p75"], lat, lon)),
                        "p90": _f_to_f(nearest_value(qmd["p90"], lat, lon)),
                        "raw_payload_path": str(payload),
                    }
                )
            if in_range:
                summary["v5_0_14" if version == "nbm_v5.0.14" else "v5_0_x"] += 1
            day += timedelta(days=1)

        if rows:
            import polars as pl

            from weadge.storage.schema import FORECAST_SCHEMA

            # re-backfill replaces the series partition (append-only lake,
            # but a full re-fetch must not leave stale duplicate part files)
            lake.delete_partition("forecasts", layer="bronze", location_id=series)
            lake.write_parquet(
                "forecasts",
                pl.DataFrame(rows, schema=FORECAST_SCHEMA),
                layer="bronze",
                partition_by="location_id",
            )
        return summary
    finally:
        if own:
            archive.close()


# ------------------------------------------------------------------ smoke audit
SNAPSHOT_HOURS = (24, 12, 6, 3, 1)
# NBM day-D MaxT window [D 12Z, D+1 06Z) vs Kalshi DCR settlement day
# [D 05Z, D+1 05Z): 17 of 24 hours overlap (70.8%) — recorded, not aligned.
NBM_WINDOW_END_HOUR = 30  # D+1 06Z
SETTLEMENT_START_HOUR = 5
SETTLEMENT_END_HOUR = 29


def nbm_smoke_audit(
    events,
    markets,
    forecasts,
    *,
    series: str,
    snapshot_hours: tuple[int, ...] = SNAPSHOT_HOURS,
) -> dict:
    """Coverage / ordering / as-of / window audit of the ingested NBM rows.

    Coverage is per EVENT DAY, never per snapshot time: at lead h a forecast
    counts iff its valid window is the event's day-D MaxT window (valid_start
    == D 12Z, valid_end == D+1 06Z) and it was available at the snapshot
    (last market close - h). A forecast whose window merely overlaps the
    snapshot clock but not the event day (e.g. the D-1 run's day-1 window)
    never counts — it is the 'wrong target-window' class, tallied separately
    as the difference from a naive same-date join. The NBM day window vs the
    DCR settlement window is reported, not aligned.
    """
    import polars as pl

    from weadge.domain.time import ensure_utc

    n_events = events.height
    fc = forecasts.filter(pl.col("location_id") == series)
    close_by_event = markets.group_by("event_ticker").agg(
        pl.col("close_at").max().alias("close_at")
    )
    ev = events.join(close_by_event, on="event_ticker", how="left")

    events_with_fc = 0
    coverage: dict[int, int] = {}
    wrong_target = 0
    for h in snapshot_hours:
        covered = 0
        for row in ev.iter_rows(named=True):
            close = row.get("close_at")
            target = row.get("target_date")
            if close is None or target is None:
                continue
            target = ensure_utc(target)
            snap = ensure_utc(close) - timedelta(hours=h)
            day_start = target + timedelta(hours=NBM_WINDOW_START_HOUR)
            day_end = target + timedelta(hours=NBM_WINDOW_END_HOUR)
            has = fc.filter(
                (pl.col("available_at") <= snap)
                & (pl.col("valid_start") == day_start)
                & (pl.col("valid_end") == day_end)
            ).height
            if has > 0:
                covered += 1
            # wrong target-window: any same-day (naive date-join) forecast at
            # the snapshot that is NOT the canonical day-D window. With the
            # f030/f054 ingest this is 0 by construction; the audit proves it.
            naive = fc.filter(
                (pl.col("available_at") <= snap)
                & (pl.col("valid_start").dt.date() == target.date())
            ).height
            wrong_target += max(naive - has, 0)
        coverage[h] = covered

    # events with >= 1 day-D forecast row at all (any snapshot)
    for row in ev.iter_rows(named=True):
        target = row.get("target_date")
        if target is None:
            continue
        day_start = ensure_utc(target) + timedelta(hours=NBM_WINDOW_START_HOUR)
        if fc.filter(pl.col("valid_start") == day_start).height > 0:
            events_with_fc += 1

    def pct(col: str) -> float:
        n = fc.height
        return 100.0 * fc.filter(pl.col(col).is_not_null()).height / n if n else 0.0

    ordering = 0
    for a, b in itertools.pairwise(("p10", "p25", "p50", "p75", "p90")):
        ordering += int(fc.filter(pl.col(a) > pl.col(b)).height)
    asof = int(
        fc.filter(
            (pl.col("available_at") < pl.col("run_init_at"))
            | (pl.col("valid_start") >= pl.col("valid_end"))
        ).height
    )
    versions = fc.group_by("model_version").len().to_dicts()
    version_runs = (
        fc.select("run_init_at", "model_version")
        .unique()
        .group_by("model_version")
        .len()
        .to_dicts()
    )

    return {
        "events": n_events,
        "events_with_forecast": events_with_fc,
        "coverage": coverage,
        "pct_mean": pct("mean"),
        "pct_std": pct("std"),
        "pct_p": min(pct(f"p{p}") for p in (10, 25, 50, 75, 90)),
        "ordering_violations": ordering,
        "asof_violations": asof,
        "wrong_target_window": wrong_target,
        "window_note": (
            "NBM day-D MaxT window [D 12Z, D+1 06Z) vs Kalshi DCR settlement "
            f"day [D 05Z, D+1 05Z): {SETTLEMENT_END_HOUR - NBM_WINDOW_START_HOUR}/24h "
            "overlap — recorded, not aligned"
        ),
        "versions": versions,
        "version_runs": version_runs,
    }


__all__ = [
    "BUCKET",
    "MAXT_CYCLE",
    "MAXT_FHOUR",
    "MAXT_FHOURS",
    "NBM_WINDOW_START_HOUR",
    "DecodedField",
    "IdxRecord",
    "NbmArchive",
    "backfill_nbm",
    "decode_blob",
    "max_2t_records",
    "model_version_for",
    "nbm_smoke_audit",
    "nbp_max_t_values",
    "nearest_value",
    "neighborhood_max",
    "parse_idx",
    "qmd_url",
]
