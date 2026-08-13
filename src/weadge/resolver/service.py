"""Resolver service - scan orchestration (shadow/alert), serve loop, kill-test stats.

    live data -> evaluate() -> executable book -> assess -> log + alert

evaluate()/find_edges() are pure; live and replay share the same path.
Every scan writes a heartbeat row; every LOCKED bucket writes a lock row
with the executable book snapshot - this is what the V0 kill test measures.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.table import Table

from weadge.config import (
    ResolverCityConfig,
    ResolverConfig,
    ResolverEdgeConfig,
    data_root,
    load_resolver,
)
from weadge.domain.time import utc_now
from weadge.resolver.edge import LockAssessment, find_edges
from weadge.resolver.log import JsonlAppender
from weadge.resolver.markets import ClobClient, PMClient
from weadge.resolver.observations import MetarClient, evaluate_observed
from weadge.resolver.state import ResolutionState, evaluate_event

console = Console()
MODES = ("shadow", "alert", "trade")


class TelegramNotifier:
    """V0 stub: no token -> silently skip."""

    def __init__(self, cfg: ResolverConfig) -> None:
        import os

        self._token = os.getenv(cfg.telegram.bot_token_env)
        self._chat = os.getenv(cfg.telegram.chat_id_env)

    def send(self, text: str) -> None:
        if not (self._token and self._chat):
            return  # ponytail: alert without a channel degrades to shadow
        import httpx

        httpx.post(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            json={"chat_id": self._chat, "text": text},
            timeout=10,
        )


def _today_event(events, tz: ZoneInfo):
    """Event for today in station-local time; None if not listed yet."""
    local_today = utc_now().astimezone(tz).date()
    for e in events:
        if e.target_date == local_today:
            return e
    return None


def _in_scan_window(tz: ZoneInfo, scan_hours: list[int]) -> bool:
    local_hour = utc_now().astimezone(tz).hour
    return scan_hours[0] <= local_hour < scan_hours[1]


def _lock_row(ts: str, city: str, station: str, a: LockAssessment, signal: bool, observation_ts: str | None = None) -> dict:
    return {
        "event": "lock",
        "ts": ts,
        "city": city,
        "station": station,
        "observation_ts": observation_ts,
        "bucket": a.bucket.label,
        "market_id": a.bucket.market_id,
        "state": a.state.value,
        "no_best_ask": a.no_ask,
        "no_ask_size": a.no_ask_size,
        "book_ts": a.book_ts.isoformat() if a.book_ts else None,
        "net_edge": a.net_edge,
        "signal": signal,
    }


def _render(signals, station: str, observed_max: float | None) -> None:
    table = Table(title=f"resolver — {station} max={observed_max}C")
    table.add_column("bucket")
    table.add_column("no_ask")
    table.add_column("size")
    table.add_column("fee")
    table.add_column("net_edge")
    for s in signals:
        table.add_row(
            s.bucket.label,
            f"{s.no_ask:.4f}",
            f"{s.no_ask_size:.1f}",
            f"{s.fee:.4f}",
            f"{s.net_edge:.4f}",
        )
    console.print(table)


def scan(city_slug: str, mode: str, cfg: ResolverConfig | None = None) -> None:
    """One scan: events -> observations -> LOCKED -> executable book -> log/alert."""
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {MODES}")
    if mode == "trade":
        raise SystemExit("trade mode is v1+ — shadow/alert first (execution.py)")
    cfg = cfg or load_resolver()
    edge: ResolverEdgeConfig = cfg.edge
    city: ResolverCityConfig = cfg.by_slug(city_slug)
    tz = ZoneInfo(city.timezone)

    if not _in_scan_window(tz, city.scan_hours):
        return

    log = JsonlAppender(data_root() / "resolver", prefix=f"shadow-{city.slug}")
    try:
        with PMClient() as pm, MetarClient() as metar, ClobClient() as clob:
            events = pm.fetch_daily_high(city.slug)
            event = _today_event(events, tz)
            if event is None:
                return
            rows = metar.fetch(city.station_icao)
            obs = evaluate_observed(city.station_icao, rows, tz, stale_after_min=edge.stale_after_min)

            if obs.stale:
                log.append(
                    {
                        "event": "heartbeat",
                        "ts": utc_now().isoformat(),
                        "city": city.city,
                        "station": city.station_icao,
                        "observed_max": obs.observed_max_c,
                        "observation_ts": obs.ts.isoformat(),
                        "stale": True,
                        "locked_count": 0,
                        "signal_count": 0,
                    }
                )
                console.print(f"[yellow]observation stale ({obs.ts}) — heartbeat only[/yellow]")
                return

            bucket_states = evaluate_event(event.buckets, obs, locked_buffer_c=edge.locked_buffer_c)
            locked = [bs for bs in bucket_states if bs.state is ResolutionState.LOCKED]

            # executable book only for LOCKED buckets (token via authoritative /markets/{cid})
            books: dict[str, object] = {}
            for bs in locked:
                token = clob.resolve_no_token(bs.bucket.condition_id)
                if token:
                    books[bs.bucket.market_id] = clob.fetch_book(token)

            assessments = find_edges(
                bucket_states,
                books,  # type: ignore[arg-type]
                min_net_edge=edge.min_net_edge,
                exec_buffer=edge.exec_buffer,
            )
            signals = [a for a in assessments if a.signal]

        now_iso = utc_now().isoformat()
        log.append(
            {
                "event": "heartbeat",
                "ts": now_iso,
                "city": city.city,
                "station": city.station_icao,
                "observed_max": obs.observed_max_c,
                "observation_ts": obs.ts.isoformat(),
                "stale": False,
                "locked_count": len(locked),
                "signal_count": len(signals),
            }
        )
        for a in assessments:
            log.append(
                _lock_row(
                    now_iso, city.city, city.station_icao, a,
                    signal=a.signal, observation_ts=obs.ts.isoformat(),
                )
            )
    finally:
        log.close()

    if signals:
        _render(signals, city.station_icao, obs.observed_max_c)
        if mode == "alert":
            TelegramNotifier(cfg).send(
                "\n".join(f"{s.bucket.label} NO@{s.no_ask:.3f} edge={s.net_edge:.3f}" for s in signals)
            )


def serve(city_slug: str, mode: str, interval_s: int = 30) -> None:
    """Loop scan every interval_s within the scan window (no WS needed for V0)."""
    cfg = load_resolver()
    city = cfg.by_slug(city_slug)
    console.print(f"[dim]resolver serve — {city.city} every {interval_s}s (window {city.scan_hours})[/dim]")
    while True:
        started = time.monotonic()
        try:
            scan(city_slug, mode, cfg)
        except Exception as exc:
            console.print(f"[red]scan error: {exc}[/red]")
        elapsed = time.monotonic() - started
        time.sleep(max(1, interval_s - elapsed))


def stats(city_slug: str, root=None, min_net_edge: float = 0.02) -> None:
    """Kill-test summary from shadow logs: lock events, ask at lock, reaction latency."""
    import json
    import statistics

    root = root or data_root() / "resolver"
    first_lock: dict[str, dict] = {}     # market_id -> first lock row
    ask_series: dict[str, list[tuple[str, float | None]]] = {}

    for f in sorted(root.glob(f"shadow-{city_slug}-*.jsonl")):
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("event") != "lock":
                    continue
                mid = row["market_id"]
                if mid not in first_lock:
                    first_lock[mid] = row
                ask_series.setdefault(mid, []).append((row["ts"], row["no_best_ask"]))

    if not first_lock:
        console.print("[dim]no lock events yet — keep scanning[/dim]")
        return

    lock_asks = [r["no_best_ask"] for r in first_lock.values() if r["no_best_ask"] is not None]
    cheap = {mid: r for mid, r in first_lock.items() if r["no_best_ask"] is not None and r["no_best_ask"] < 0.97}

    def notional(r: dict) -> float | None:
        if r["no_best_ask"] is None:
            return None
        return r["no_best_ask"] * (r["no_ask_size"] or 0)

    exec5 = [r for r in first_lock.values() if (notional(r) or 0) >= 5 and (r["net_edge"] or 0) >= min_net_edge]
    exec20 = [r for r in first_lock.values() if (notional(r) or 0) >= 20 and (r["net_edge"] or 0) >= min_net_edge]

    # observation age at first lock: scan_ts - METAR observation_ts (only rows logged after this patch)
    ages = []
    for r in first_lock.values():
        ots = r.get("observation_ts")
        if ots:
            ages.append(
                (datetime.fromisoformat(r["ts"]) - datetime.fromisoformat(ots)).total_seconds()
            )

    def reaction(threshold: float) -> list[float]:
        out = []
        for mid, rows in ask_series.items():
            if mid not in cheap:
                continue
            lock_ts = first_lock[mid]["ts"]
            for ts, ask in rows:
                if ask is not None and ask >= threshold and ts > lock_ts:
                    t0 = datetime.fromisoformat(lock_ts)
                    t1 = datetime.fromisoformat(ts)
                    out.append((t1 - t0).total_seconds())
                    break
        return out

    def med(vals: list[float]) -> float | None:
        return statistics.median(vals) if vals else None

    def p90(vals: list[float]) -> float | None:
        vals = sorted(vals)
        if len(vals) < 10:
            return None
        return statistics.quantiles(vals, n=100)[89]

    r97 = reaction(0.97)
    r99 = reaction(0.99)
    table = Table(title=f"resolver stats — {city_slug} ({len(first_lock)} lock events)")
    table.add_column("metric")
    table.add_column("value")
    rows = [
        ("LOCK EVENTS (first lock per bucket)", str(len(first_lock))),
        ("at lock: median NO ask", f"{med(lock_asks):.4f}" if lock_asks else "-"),
        ("at lock: p90 NO ask", f"{p90(lock_asks):.4f}" if p90(lock_asks) is not None else "-"),
        ("lock ask < 0.97", str(len(cheap))),
        ("lock ask < 0.95", str(sum(1 for a in lock_asks if a < 0.95))),
        ("lock ask < 0.90", str(sum(1 for a in lock_asks if a < 0.90))),
        ("observation age at lock: median", f"{med(ages):.0f}s" if ages else "-"),
        ("observation age at lock: p90", f"{p90(ages):.0f}s" if p90(ages) is not None else "-"),
        ("reaction to 0.97: median s", f"{med(r97):.0f}" if r97 else "-"),
        ("reaction to 0.97: p90 s", f"{p90(r97):.0f}" if p90(r97) is not None else "-"),
        ("reaction to 0.99: median s", f"{med(r99):.0f}" if r99 else "-"),
        ("executable $5 at lock", str(len(exec5))),
        ("executable $20 at lock", str(len(exec20))),
    ]
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)
