"""Resolver service - shadow scan orchestration.

    live data -> evaluate() -> find_edges() -> shadow record / alert

evaluate() is a pure function; live and historical replay share the same path.
"""

from __future__ import annotations

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
from weadge.live.recorder import JsonlZstAppender
from weadge.resolver.edge import Signal, find_edges
from weadge.resolver.markets import DailyHighEvent, PMClient
from weadge.resolver.observations import MetarClient, ObservedState, evaluate_observed
from weadge.resolver.state import evaluate_event

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


def _today_event(events: list[DailyHighEvent], tz: ZoneInfo) -> DailyHighEvent | None:
    """Event for today in station-local time; None if not listed yet."""
    local_today = utc_now().astimezone(tz).date()
    for e in events:
        if e.target_date == local_today:
            return e
    return None


def _in_scan_window(tz: ZoneInfo, scan_hours: list[int]) -> bool:
    local_hour = utc_now().astimezone(tz).hour
    return scan_hours[0] <= local_hour < scan_hours[1]


def _record(recorder: JsonlZstAppender, signal: Signal) -> None:
    recorder.append(
        utc_now().isoformat(),
        {
            "mode": "shadow",
            "city": signal.city,
            "target_date": signal.target_date.isoformat(),
            "market_id": signal.bucket.market_id,
            "question": signal.bucket.question,
            "no_ask": signal.no_ask,
            "fee": signal.fee,
            "net_edge": signal.net_edge,
        },
    )


def _render(signals: list[Signal], obs: ObservedState) -> None:
    table = Table(title=f"resolver shadow — {obs.station} max={obs.observed_max_c}C (stale={obs.stale})")
    table.add_column("bucket")
    table.add_column("no_ask")
    table.add_column("fee")
    table.add_column("net_edge")
    for s in signals:
        table.add_row(s.bucket.question[:40], f"{s.no_ask:.4f}", f"{s.fee:.4f}", f"{s.net_edge:.4f}")
    console.print(table)


def scan(city_slug: str, mode: str, cfg: ResolverConfig | None = None) -> None:
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {MODES}")
    if mode == "trade":
        raise SystemExit("trade mode is v1+ — shadow/alert first (execution.py)")
    cfg = cfg or load_resolver()
    edge: ResolverEdgeConfig = cfg.edge
    city: ResolverCityConfig = cfg.by_slug(city_slug)
    tz = ZoneInfo(city.timezone)

    if not _in_scan_window(tz, city.scan_hours):
        console.print(f"[dim]outside scan window ({city.scan_hours}) — skip[/dim]")
        return

    with PMClient() as pm, MetarClient() as metar:
        events = pm.fetch_daily_high(city.slug)
        event = _today_event(events, tz)
        if event is None:
            console.print("[dim]no daily-high event for today — skip[/dim]")
            return
        rows = metar.fetch(city.station_icao)
        obs = evaluate_observed(city.station_icao, rows, tz, stale_after_min=edge.stale_after_min)

    if obs.stale:
        console.print(f"[yellow]observation stale ({obs.ts}) — no signal[/yellow]")
        return

    bucket_states = evaluate_event(event.buckets, obs, locked_buffer_c=edge.locked_buffer_c)
    signals = find_edges(
        city.city,
        event.target_date,
        bucket_states,
        obs,
        min_net_edge=edge.min_net_edge,
        exec_buffer=edge.exec_buffer,
    )

    if not signals:
        console.print("[dim]no LOCKED edge — clean[/dim]")
        return

    recorder = JsonlZstAppender(data_root() / "resolver", f"shadow-{city.slug}")
    for s in signals:
        _record(recorder, s)
    _render(signals, obs)

    if mode == "alert":
        TelegramNotifier(cfg).send(
            "\n".join(f"{s.bucket.question} NO@{s.no_ask:.3f} edge={s.net_edge:.3f}" for s in signals)
        )
