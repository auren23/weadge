"""Observation race + WU resolution audit.

Question: which public feed first shows a settlement-faithful threshold
crossing? Latency and fidelity are separate axes — this tool records both.

  aviationweather METAR (10s, race process only)
  IEM current.py
      -> first_seen_at per (station, observation_at, source)

  METAR-derived local-day max vs Wunderground Daily Observations max
      -> per-city exact / ±1 / larger

Deliberately standalone: tools/, never imported by the resolver.
Production weadge serve stays 30s and LOCKED-only.

Run:
  uv run python tools/observation_race.py serve --interval 10
  uv run python tools/observation_race.py serve --once
  uv run python tools/observation_race.py summary
  uv run python tools/observation_race.py audit --backfill 7
  uv run python tools/observation_race.py audit --date 2026-08-12 --city paris
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from rich.console import Console
from rich.table import Table

from weadge.config import ResolverCityConfig, ResolverConfig, data_root, load_resolver
from weadge.domain.time import from_timestamp, parse_iso, utc_now
from weadge.resolver.log import JsonlAppender
from weadge.resolver.observations import MetarClient, evaluate_observed

USER_AGENT = "weadge-observation-race/0.1"
IEM_CURRENT = "https://mesonet.agron.iastate.edu/json/current.py"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
WU_HIST = "https://api.weather.com/v1/location/{loc}/observations/historical.json"
# ponytail: WU history-page frontend key; rotates. Override with WUNDERGROUND_API_KEY.
WU_DEFAULT_KEY = "f6d2efe5720d47ea92efe5720df7eaa8"
AWC_MAX_HOURS = 360  # documented ~15 day archive
IEM_SOURCE_GRADE = "B"
# Skip cache-history on first poll so first_seen_at is not "now" for 2h-old METARs.
MAX_OBS_AGE = timedelta(minutes=20)
console = Console()


# ---------------------------------------------------------------- pure


def c_to_f(c: float) -> float:
    """Whole-degree F via round(C * 9/5 + 32). Not ASOS 2-min averaging."""
    return float(round(c * 9 / 5 + 32))


def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def wu_api_units(unit: str) -> str:
    return "e" if unit.lower().startswith("f") else "m"


def settlement_unit_letter(unit: str) -> str:
    return "F" if unit.lower().startswith("f") else "C"


def iem_current_station(cfg: ResolverCityConfig) -> str:
    return cfg.iem_station or cfg.station_icao


def report_id(station: str, observation_at: datetime) -> str:
    return f"{station}-{observation_at.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"


def seen_key(station: str, observation_at: datetime, source: str) -> tuple[str, str, str]:
    return (station, observation_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), source)


def wu_daily_max(observations: list[dict]) -> float | None:
    temps = [float(o["temp"]) for o in observations if o.get("temp") is not None]
    return max(temps) if temps else None


def audit_bucket(metar_max: float | None, wu_max: float | None) -> str:
    if wu_max is None:
        return "wu_missing"
    if metar_max is None:
        return "metar_missing"
    delta = abs(metar_max - wu_max)
    if delta == 0:
        return "exact"
    if delta <= 1:
        return "pm1"
    return "larger"


def audit_dates(n: int, *, today: date) -> list[date]:
    return [today - timedelta(days=i) for i in range(1, n + 1)]


def fresh_enough(observation_at: datetime, now: datetime, max_age: timedelta = MAX_OBS_AGE) -> bool:
    return now - observation_at.astimezone(UTC) <= max_age


def local_day_range(local_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime(local_date.year, local_date.month, local_date.day, tzinfo=tz).astimezone(UTC)
    return start, start + timedelta(days=1)


class FirstSeen:
    """First (station, observation_at, source) wins. Restart-safe via load_row."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str, str]] = set()

    def offer(self, station: str, observation_at: datetime, source: str) -> bool:
        key = seen_key(station, observation_at, source)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def load_row(self, row: dict) -> None:
        if row.get("event") or not (
            row.get("station") and row.get("observation_at") and row.get("source")
        ):
            return
        ts = parse_iso(row["observation_at"])
        self._seen.add(seen_key(row["station"], ts, row["source"]))


def race_row(
    *,
    station: str,
    observation_at: datetime,
    source: str,
    source_grade: str,
    first_seen_at: datetime,
    decoded_temp: float,
    temp_unit: str,
    raw_temp: str,
    raw_metar: str | None = None,
    provider_receipt_at: datetime | None = None,
) -> dict:
    return {
        "station": station,
        "report_id": report_id(station, observation_at),
        "observation_at": observation_at.astimezone(UTC).isoformat(),
        "source": source,
        "source_grade": source_grade,
        "first_seen_at": first_seen_at.astimezone(UTC).isoformat(),
        "provider_receipt_at": (
            provider_receipt_at.astimezone(UTC).isoformat() if provider_receipt_at else None
        ),
        "raw_temp": raw_temp,
        "decoded_temp": decoded_temp,
        "temp_unit": temp_unit,
        "raw_metar": raw_metar or "",
    }


def group_first_seen(rows: list[dict]) -> dict[tuple[str, str], dict[str, str]]:
    """(station, observation_at) -> {source: first_seen_at}."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("event") or not row.get("source"):
            continue
        key = (row["station"], row["observation_at"])
        out.setdefault(key, {})[row["source"]] = row["first_seen_at"]
    return out


def decode_awc(row: dict, source_grade: str, now: datetime) -> dict | None:
    if row.get("temp") is None or row.get("obsTime") is None or not row.get("icaoId"):
        return None
    obs_at = from_timestamp(row["obsTime"])
    receipt = parse_iso(row["receiptTime"]) if row.get("receiptTime") else None
    return race_row(
        station=row["icaoId"],
        observation_at=obs_at,
        source="aviationweather",
        source_grade=source_grade,
        first_seen_at=now,
        decoded_temp=float(row["temp"]),
        temp_unit="C",
        raw_temp=str(row["temp"]),
        raw_metar=row.get("rawOb") or "",
        provider_receipt_at=receipt,
    )


def decode_iem_current(payload: dict, cfg: ResolverCityConfig, now: datetime) -> dict | None:
    ob = payload.get("last_ob") or {}
    raw_valid = ob.get("utc_valid")
    tmpf = ob.get("airtemp[F]")
    if not raw_valid or tmpf is None:
        return None
    return race_row(
        station=cfg.station_icao,
        observation_at=parse_iso(raw_valid),
        source="iem",
        source_grade=IEM_SOURCE_GRADE,
        first_seen_at=now,
        decoded_temp=float(tmpf),
        temp_unit="F",
        raw_temp=str(tmpf),
        raw_metar="",
    )


def grade_for(icao: str, stations: list[ResolverCityConfig]) -> str:
    for s in stations:
        if s.station_icao == icao:
            return s.source_grade
    return "C"


# ---------------------------------------------------------------- io helpers


def _http() -> httpx.Client:
    return httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def race_dir() -> Path:
    return data_root() / "race"


def hydrate_seen(seen: FirstSeen, root: Path | None = None) -> None:
    root = root or race_dir()
    if not root.exists():
        return
    for path in sorted(root.glob("race-*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                seen.load_row(json.loads(line))


def _error_row(source: str, station: str | None, exc: BaseException) -> dict:
    return {
        "event": "error",
        "source": source,
        "station": station,
        "ts": utc_now().isoformat(),
        "error": f"{type(exc).__name__}: {exc}",
    }


def fetch_iem_current(http: httpx.Client, cfg: ResolverCityConfig) -> dict:
    if not cfg.iem_network:
        raise RuntimeError(f"{cfg.slug}: iem_network missing")
    resp = http.get(
        IEM_CURRENT,
        params={"station": iem_current_station(cfg), "network": cfg.iem_network},
    )
    resp.raise_for_status()
    return resp.json()


def fetch_wu_observations(
    http: httpx.Client, wu_location: str, local_date: date, unit: str
) -> list[dict]:
    key = os.environ.get("WUNDERGROUND_API_KEY") or WU_DEFAULT_KEY
    resp = http.get(
        WU_HIST.format(loc=wu_location),
        params={
            "apiKey": key,
            "units": wu_api_units(unit),
            "startDate": local_date.strftime("%Y%m%d"),
            "endDate": local_date.strftime("%Y%m%d"),
        },
    )
    resp.raise_for_status()
    if not resp.content:
        return []
    return list((resp.json() or {}).get("observations") or [])


def iem_asos_to_awc_rows(text: str) -> list[dict]:
    """IEM asos.py CSV (valid,tmpf) -> aviationweather-shaped rows with temp in C."""
    rows: list[dict] = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            t = datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        try:
            tmpf = float(parts[2])
        except (TypeError, ValueError):
            continue
        rows.append({"obsTime": int(t.timestamp()), "temp": f_to_c(tmpf)})
    return rows


def fetch_iem_asos(http: httpx.Client, station_icao: str, start: date, end: date) -> list[dict]:
    resp = http.get(
        IEM_ASOS,
        params={
            "station": station_icao,
            "data": "tmpf",
            "year1": start.year,
            "month1": start.month,
            "day1": start.day,
            "year2": end.year,
            "month2": end.month,
            "day2": end.day,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "report_type": ["3", "4"],
            "missing": "M",
            "trace": "null",
            "direct": "yes",
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.text
    if text.lstrip().lower().startswith(("<html", "error")):
        raise RuntimeError(f"IEM asos rejected {station_icao}: {text[:200]}")
    return iem_asos_to_awc_rows(text)


def metar_max_c_for_day(
    rows: list[dict], station: str, tz: ZoneInfo, local_date: date
) -> float | None:
    start, end = local_day_range(local_date, tz)
    day_rows = [
        r for r in rows if r.get("temp") is not None and start <= from_timestamp(r["obsTime"]) < end
    ]
    obs = evaluate_observed(
        station, day_rows, tz, now=end - timedelta(seconds=1), stale_after_min=10**6
    )
    return obs.observed_max_c


def filter_stations(cfg: ResolverConfig, slug: str | None) -> list[ResolverCityConfig]:
    stations = cfg.observation_stations()
    if slug is None:
        return stations
    picked = [s for s in stations if s.slug == slug]
    if not picked:
        raise SystemExit(f"unknown observation city {slug}")
    return picked


# ---------------------------------------------------------------- serve / summary / audit


def _record_new(built: dict, seen: FirstSeen, log: JsonlAppender, now: datetime) -> bool:
    obs_at = parse_iso(built["observation_at"])
    if not fresh_enough(obs_at, now):
        return False
    if not seen.offer(built["station"], obs_at, built["source"]):
        return False
    log.append(built)
    return True


def poll_once(
    cfg: ResolverConfig,
    seen: FirstSeen,
    log: JsonlAppender,
    http: httpx.Client,
    metar: MetarClient,
) -> int:
    stations = cfg.observation_stations()
    icaos = [s.station_icao for s in stations]
    now = utc_now()
    recorded = 0
    try:
        rows = metar.fetch(",".join(icaos), hours=2)
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            built = decode_awc(row, grade_for(row.get("icaoId") or "", stations), now)
            if built is None:
                continue
            if _record_new(built, seen, log, now):
                recorded += 1
    except Exception as exc:
        log.append(_error_row("aviationweather", None, exc))
        console.print(f"[red]awc: {exc}[/red]")

    for st in stations:
        try:
            payload = fetch_iem_current(http, st)
            built = decode_iem_current(payload, st, now)
            if built is None:
                continue
            if _record_new(built, seen, log, now):
                recorded += 1
        except Exception as exc:
            log.append(_error_row("iem", st.station_icao, exc))
            console.print(f"[red]iem {st.station_icao}: {exc}[/red]")
    return recorded


def cmd_serve(interval: int, once: bool) -> None:
    cfg = load_resolver()
    seen = FirstSeen()
    hydrate_seen(seen)
    log = JsonlAppender(race_dir(), prefix="race")
    console.print(
        f"[dim]observation race — {len(cfg.observation_stations())} stations "
        f"every {interval}s (resolver serve stays 30s)[/dim]"
    )
    try:
        with _http() as http, MetarClient() as metar:
            while True:
                n = poll_once(cfg, seen, log, http, metar)
                if n:
                    console.print(f"[green]recorded {n} new first_seen[/green]")
                if once:
                    return
                time.sleep(max(1, interval))
    finally:
        log.close()


def _load_race_rows(root: Path | None = None) -> list[dict]:
    root = root or race_dir()
    rows: list[dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("race-*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def cmd_summary() -> None:
    grouped = group_first_seen(_load_race_rows())
    if not grouped:
        console.print("[dim]no race rows yet[/dim]")
        return
    table = Table(title=f"observation race ({len(grouped)} reports)")
    table.add_column("station")
    table.add_column("observation_at")
    table.add_column("aviationweather")
    table.add_column("iem")
    table.add_column("awc lead s")
    for (station, obs_at), sources in sorted(grouped.items()):
        awc = sources.get("aviationweather")
        iem = sources.get("iem")
        lead = ""
        if awc and iem:
            lead = f"{(parse_iso(awc) - parse_iso(iem)).total_seconds():.0f}"
        table.add_row(station, obs_at, awc or "-", iem or "-", lead)
    console.print(table)


def _metar_rows_for_day(
    http: httpx.Client,
    metar: MetarClient,
    station: str,
    tz: ZoneInfo,
    local_date: date,
    now: datetime,
) -> tuple[list[dict], str]:
    start, _end = local_day_range(local_date, tz)
    hours = int((now - start).total_seconds() // 3600) + 2
    if 0 < hours <= AWC_MAX_HOURS:
        rows = metar.fetch(station, hours=hours)
        if not isinstance(rows, list):
            rows = []
        return rows, "aviationweather"
    pad_start = local_date - timedelta(days=1)
    pad_end = local_date + timedelta(days=1)
    return fetch_iem_asos(http, station, pad_start, pad_end), "iem"


def cmd_audit(backfill: int, day: date | None, city: str | None) -> None:
    cfg = load_resolver()
    stations = filter_stations(cfg, city)
    today_utc = utc_now().date()
    days = [day] if day is not None else audit_dates(backfill, today=today_utc)
    log = JsonlAppender(race_dir(), prefix="audit")
    counts: dict[str, dict[str, int]] = {}
    try:
        with _http() as http, MetarClient() as metar:
            now = utc_now()
            for st in stations:
                if not st.wu_location:
                    console.print(f"[red]{st.slug}: wu_location missing[/red]")
                    log.append(
                        _error_row(
                            "wunderground", st.station_icao, RuntimeError("wu_location missing")
                        )
                    )
                    continue
                tz = ZoneInfo(st.timezone)
                tallies = counts.setdefault(
                    st.slug,
                    {
                        "exact": 0,
                        "pm1": 0,
                        "larger": 0,
                        "wu_missing": 0,
                        "metar_missing": 0,
                        "n": 0,
                    },
                )
                for local_date in days:
                    wu_max: float | None = None
                    try:
                        wu_max = wu_daily_max(
                            fetch_wu_observations(http, st.wu_location, local_date, st.unit)
                        )
                    except Exception as exc:
                        log.append(_error_row("wunderground", st.station_icao, exc))
                        console.print(f"[red]wu {st.slug} {local_date}: {exc}[/red]")
                    metar_c: float | None = None
                    metar_source = "aviationweather"
                    try:
                        rows, metar_source = _metar_rows_for_day(
                            http, metar, st.station_icao, tz, local_date, now
                        )
                        metar_c = metar_max_c_for_day(rows, st.station_icao, tz, local_date)
                    except Exception as exc:
                        log.append(_error_row("metar", st.station_icao, exc))
                        console.print(f"[red]metar {st.slug} {local_date}: {exc}[/red]")
                    letter = settlement_unit_letter(st.unit)
                    metar_settlement = (
                        None if metar_c is None else (c_to_f(metar_c) if letter == "F" else metar_c)
                    )
                    bucket = audit_bucket(metar_settlement, wu_max)
                    delta = (
                        None
                        if metar_settlement is None or wu_max is None
                        else metar_settlement - wu_max
                    )
                    row = {
                        "event": "audit",
                        "station": st.station_icao,
                        "slug": st.slug,
                        "local_date": local_date.isoformat(),
                        "unit": letter,
                        "metar_source": metar_source,
                        "metar_max": metar_settlement,
                        "metar_max_c": metar_c,
                        "wu_max": wu_max,
                        "delta": delta,
                        "bucket": bucket,
                    }
                    log.append(row)
                    tallies[bucket] = tallies.get(bucket, 0) + 1
                    tallies["n"] += 1
    finally:
        log.close()

    table = Table(title="resolution audit")
    table.add_column("city")
    table.add_column("n")
    table.add_column("exact")
    table.add_column("±1")
    table.add_column("larger")
    table.add_column("wu miss")
    table.add_column("metar miss")
    for slug, t in counts.items():
        table.add_row(
            slug,
            str(t["n"]),
            str(t["exact"]),
            str(t["pm1"]),
            str(t["larger"]),
            str(t["wu_missing"]),
            str(t["metar_missing"]),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="poll AWC+IEM and record first_seen")
    sp.add_argument("--interval", type=int, default=10)
    sp.add_argument("--once", action="store_true")

    sub.add_parser("summary", help="per-observation first_seen table")

    ap = sub.add_parser("audit", help="METAR max vs WU Daily Observations max")
    ap.add_argument("--backfill", type=int, default=7)
    ap.add_argument("--date", type=date.fromisoformat, default=None)
    ap.add_argument("--city", default=None, help="slug; default all observation stations")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        cmd_serve(args.interval, args.once)
    elif args.cmd == "summary":
        cmd_summary()
    else:
        cmd_audit(args.backfill, args.date, args.city)


if __name__ == "__main__":
    main(sys.argv[1:])
