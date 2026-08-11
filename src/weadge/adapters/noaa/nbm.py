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
    window ending 06Z. It does NOT equal the Kalshi DCR settlement day
    [D 05Z, D+1 05Z): 7h late start, 1h late end. The mismatch is a
    recorded research fact (potential station/window-correction alpha),
    never silently papered over.
  * Cross-checks vs the NBP station card (same run): max_2t p50 at KNYC
    = 96.0°F == TXNP5; max_2t mean neighborhood at PHNL = 87.5°F vs
    TXNMN 88 (grid resolution).
  * Observed archive availability: the 00Z run's f030 qmd object appears
    ~07:15Z (S3 Last-Modified) — AFTER the T-24h snapshot (D 04:59Z), so
    T-24h cannot use the D-00Z MaxT (an earlier run must, or it is missing).

This module is the probe/ingest layer only; it writes nothing to research.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np

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
    """The MaxT QMD family: TMP 2m '12-30 hour max fcst' records.

    Returns {"mean": ..., "std": ..., 10: ..., 25: ..., 50: ..., ...} —
    percentile keys are ints. The std record carries fcst='12-30 hour
    StdDev fcst' (not 'max fcst'); instantaneous 2t and MinT are excluded.
    """
    out: dict[str, IdxRecord] = {}
    for r in rows:
        if r.var != "TMP" or r.level != "2 m above ground" or "12-30 hour" not in r.fcst:
            continue
        if "StdDev fcst" in r.fcst:
            out["std"] = r
            continue
        if "max fcst" not in r.fcst:
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

    def __init__(self, timeout_s: float = 90.0) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s), follow_redirects=True)

    def idx(self, run_date, cycle: int, fhour: int, domain: str = DOMAIN_CO) -> list[IdxRecord]:
        url = qmd_url(run_date, cycle, fhour, domain) + ".idx"
        resp = self._client.get(url)
        resp.raise_for_status()
        return parse_idx(resp.text)

    def fetch_records(self, run_date, cycle: int, fhour: int, domain: str,
                      records: list[IdxRecord]) -> bytes:
        """Fetch the byte ranges of `records` and concatenate them into one
        valid (partial) GRIB2 blob."""
        url = qmd_url(run_date, cycle, fhour, domain)
        parts: list[bytes] = []
        for r in records:
            if r.next_offset is None:
                raise ValueError(f"record {r.record} has no end offset")
            resp = self._client.get(
                url, headers={"Range": f"bytes={r.offset}-{r.next_offset - 1}"}
            )
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(f"expected 206 partial content, got {resp.status_code}")
            parts.append(resp.content)
        return b"".join(parts)

    def max_2t_qmd(
        self,
        run_date,
        cycle: int,
        fhour: int,
        domain: str = DOMAIN_CO,
        percentiles: tuple[int, ...] = NBM_PERCENTILES,
    ) -> dict[str, DecodedField]:
        """MaxT QMD fields (mean/std/percentiles) for one run and fhour.

        The mean and std records carry identical GRIB metadata (template 8,
        no percentile), so they are identified by their fetch ORDER — the
        .idx desc distinguishes them, the decoded fields do not.
        """
        rows = max_2t_records(self.idx(run_date, cycle, fhour, domain))
        names = ["mean", "std"] + [f"p{p}" for p in percentiles]
        want: list[tuple[str, IdxRecord]] = []
        for name in names:
            key: Any = "std" if name == "std" else ("mean" if name == "mean" else int(name[1:]))
            r = rows.get(key)
            if r is not None:
                want.append((name, r))
        blob = self.fetch_records(run_date, cycle, fhour, domain, [r for _, r in want])
        run_init = datetime(run_date.year, run_date.month, run_date.day, cycle, tzinfo=UTC)
        fields = decode_blob(blob, run_init)
        return {name: field for (name, _r), field in zip(want, fields, strict=True)}

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


__all__ = [
    "BUCKET",
    "DecodedField",
    "IdxRecord",
    "NbmArchive",
    "decode_blob",
    "max_2t_records",
    "model_version_for",
    "nbp_max_t_values",
    "nearest_value",
    "neighborhood_max",
    "parse_idx",
    "qmd_url",
]
