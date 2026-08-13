"""Kalshi LOCKED latency probe — one-off structural falsification tool.

Question: after a Daily High bucket is confirmed dead by public METAR
observations, does the market price lag?

  KNYC METAR (IEM routine + SPECI)  ->  first LOCKED instant per bucket
  Kalshi 1m YES bid/ask candles     ->  first executable quote after lock
  Kalshi trade prints               ->  trade-confirmed stale prints

No-lookahead: the earliest usable quote is the first 1m bar that STARTS at
or after lock_at + 1 minute — never the bar containing the observation.

Evidence is reported in three tiers:
  trade-confirmed  (a print at NO <= 0.97 after lock)  — hardest
  quote-indicated  (candle bid implies NO ask < 0.97)
  unknown          (no quote after lock)

Deliberately standalone: tools/, never imported by the resolver. Uses only
the existing Kalshi client (runtime deps: httpx + rich + stdlib).

Run:
  uv run python tools/kalshi_lock_probe.py --series KXHIGHNY \
      --start 2025-01-01 --end 2026-07-31
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import statistics
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from rich.console import Console
from rich.table import Table

from weadge.adapters.kalshi.client import KalshiClient, KalshiError, parse_api_ts
from weadge.config import data_root, load_cities

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
console = Console()


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- METAR
def fetch_metar(
    station: str, start: date, end: date, cache_dir: Path
) -> list[tuple[datetime, float]]:
    """Routine METAR + SPECI temps for [start, end], per-year cached CSV.

    report_type=3 (routine hourly) + 4 (specials) — the information flow a
    market participant actually had. NOT the 1-minute ASOS archive (may be
    a post-hoc series and would leak information).
    """
    rows: list[tuple[datetime, float]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start.year, end.year + 1):
        cache = cache_dir / f"metar-{station}-{year}.csv"
        if cache.exists():
            text = cache.read_text()
        else:
            y1 = max(start, date(year, 1, 1))
            y2 = min(end, date(year, 12, 31))
            resp = None
            for attempt in range(6):
                resp = httpx.get(
                    IEM_ASOS,
                    params={
                        "station": station,
                        "data": "tmpf",
                        "year1": y1.year,
                        "month1": y1.month,
                        "day1": y1.day,
                        "year2": y2.year,
                        "month2": y2.month,
                        "day2": y2.day,
                        "tz": "Etc/UTC",
                        "format": "onlycomma",
                        "report_type": ["3", "4"],
                        "missing": "M",
                        "trace": "null",
                        "direct": "yes",
                    },
                    timeout=300,
                    follow_redirects=True,
                )
                if resp.status_code != 429:
                    break
                # IEM enforces an IP-based rate limit; back off hard
                time.sleep(30 * (attempt + 1))
            assert resp is not None
            resp.raise_for_status()
            text = resp.text
            if text.lstrip().lower().startswith(("<html", "error")):
                raise RuntimeError(f"IEM rejected request for {station} {year}: {text[:200]}")
            cache.write_text(text)
            console.print(
                f"[dim]METAR {station} {year}: {len(text.splitlines()):,} rows cached[/dim]"
            )
        for line in text.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                t = datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            except ValueError:
                continue
            tmpf = _f(parts[2])
            if tmpf is None:
                continue
            rows.append((t, tmpf))
    rows.sort()
    return rows


# ---------------------------------------------------------------- lock
def locked_floor(market: dict, buffer_f: float) -> float | None:
    """Observed-max threshold (F) at which this bucket is confirmed dead.

    cap-only ("less than X°F"): observed >= cap  -> final < cap impossible.
    both ("between X-Y°F"):      observed >  cap  -> final <= cap impossible
                                 (integer temps, so cap + 1).
    floor-only ("greater than X°F"): never cold-locks.
    """
    cap = market.get("cap_strike")
    if cap is None:
        return None
    if market.get("floor_strike") is not None:
        cap += 1.0
    return cap + buffer_f


def first_lock(
    metar: list[tuple[datetime, float]],
    day: date,
    floor: float,
    tz: ZoneInfo,
) -> tuple[datetime, float] | None:
    """(lock_at, observed_max) when running daily METAR max first hits floor.

    Day boundary is the station-local calendar day (the day the market
    settles for), matching the resolver's running-max semantics.
    """
    start_utc = datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(UTC)
    end_utc = start_utc + timedelta(days=1)
    running = -math.inf
    for t, tmpf in metar:
        if t < start_utc:
            continue
        if t >= end_utc:
            break
        if tmpf > running:
            running = tmpf
        if running >= floor:
            return t, running
    return None


# ---------------------------------------------------------------- market
def probe_market(
    client: KalshiClient,
    series: str,
    market: dict,
    lock_at: datetime,
    close_at: datetime,
) -> dict:
    """Candles + trades for one locked bucket, no-lookahead quotes."""
    ticker = market["ticker"]
    first_ts = lock_at + timedelta(minutes=1)  # never the bar containing the observation

    bars: list[tuple[datetime, float]] = []  # (bar_start, yes_bid_open)
    for c in client.get_market_candles(
        ticker, series, lock_at - timedelta(minutes=10), close_at, period_interval_min=1
    ):
        bid = c.get("yes_bid") or {}
        open_bid = _f(bid.get("open_dollars") or bid.get("open"))
        if open_bid is None:
            continue
        bar_end = parse_api_ts(c["end_period_ts"])
        bars.append((bar_end - timedelta(minutes=1), open_bid))
    bars.sort()

    first_quote: datetime | None = None
    no_ask_at_lock: float | None = None
    reaction_97: float | None = None  # seconds lock_at -> quote implies NO ask <= 0.97
    reaction_99: float | None = None  # ... NO ask <= 0.99
    for bs, bid in bars:
        if bs < first_ts:
            continue
        if first_quote is None:
            first_quote = bs
            no_ask_at_lock = 1.0 - bid
        if reaction_97 is None and bid >= 0.03:
            reaction_97 = (bs - lock_at).total_seconds()
        if reaction_99 is None and bid >= 0.01:
            reaction_99 = (bs - lock_at).total_seconds()
        if first_quote is not None and reaction_97 is not None and reaction_99 is not None:
            break

    # trade-confirmed stale: prints at NO <= 0.97 within 5m of lock
    stale_n = 0
    stale_vol = 0.0
    worst: float | None = None
    for p in client.get_market_trades(ticker, close_at=close_at):
        ts = parse_api_ts(p["created_time"])
        if ts < lock_at or ts >= lock_at + timedelta(minutes=5):
            continue
        np_ = _f(p.get("no_price_dollars"))
        if np_ is None or np_ > 0.97:
            continue
        stale_n += 1
        stale_vol += _f(p.get("count_fp") or p.get("count")) or 0.0
        worst = np_ if worst is None else min(worst, np_)

    return {
        "date": lock_at.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
        "market_ticker": ticker,
        "bucket": bucket_label(market),
        "lock_at": lock_at.isoformat(),
        "observed_max": None,  # filled by caller (lock instant running max)
        "first_quote_at": first_quote.isoformat() if first_quote else "",
        "no_ask_at_lock": no_ask_at_lock,
        "reaction_97_s": reaction_97,
        "reaction_99_s": reaction_99,
        "stale_trade_count_5m": stale_n,
        "stale_trade_volume_5m": round(stale_vol, 1),
        "worst_stale_trade_price": worst,
        "final_result": market.get("result"),
    }


def bucket_label(market: dict) -> str:
    lo, hi = market.get("floor_strike"), market.get("cap_strike")
    if lo is not None and hi is not None:
        return f"between {lo:g}-{hi:g}F"
    if hi is not None:
        return f"less than {hi:g}F"
    if lo is not None:
        return f"greater than {lo:g}F"
    return "?"


def _fmt_td(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------- summary
def summary(rows: list[dict], total_events: int, skipped: int) -> None:
    asks = [r["no_ask_at_lock"] for r in rows if r["no_ask_at_lock"] is not None]
    no_quote = sum(1 for r in rows if not r["first_quote_at"])
    r97 = [r["reaction_97_s"] for r in rows if r["reaction_97_s"] is not None]
    r99 = [r["reaction_99_s"] for r in rows if r["reaction_99_s"] is not None]
    t97 = [
        r
        for r in rows
        if r["stale_trade_count_5m"] > 0 and (r["worst_stale_trade_price"] or 1) <= 0.97
    ]
    t95 = [r for r in t97 if (r["worst_stale_trade_price"] or 1) <= 0.95]
    t90 = [r for r in t97 if (r["worst_stale_trade_price"] or 1) <= 0.90]

    def p90(vals: list[float]) -> float | None:
        if len(vals) < 10:
            return None
        return statistics.quantiles(sorted(vals), n=100)[89]

    table = Table(title=f"Kalshi LOCKED latency — {len(rows)} locked buckets")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("events (settled, window)", str(total_events))
    table.add_row("locked buckets", str(len(rows)))
    table.add_row("skipped (api errors)", str(skipped))
    table.add_row("")
    table.add_row("first quote after lock (>= lock+1m):", "")
    table.add_row("  no ask: median", f"{statistics.median(asks):.4f}" if asks else "-")
    table.add_row(
        "  no ask: p10",
        f"{statistics.quantiles(sorted(asks), n=100)[9]:.4f}" if len(asks) >= 10 else "-",
    )
    table.add_row("  no ask < .97", str(sum(1 for a in asks if a < 0.97)))
    table.add_row("  no ask < .95", str(sum(1 for a in asks if a < 0.95)))
    table.add_row("  no ask < .90", str(sum(1 for a in asks if a < 0.90)))
    table.add_row("  no quote after lock (unknown)", str(no_quote))
    table.add_row("")
    table.add_row(
        "reaction -> .97: median / p90",
        f"{_fmt_td(statistics.median(r97))} / {_fmt_td(p90(r97))}" if r97 else "-",
    )
    table.add_row(
        "reaction -> .99: median / p90",
        f"{_fmt_td(statistics.median(r99))} / {_fmt_td(p90(r99))}" if r99 else "-",
    )
    table.add_row("")
    table.add_row("trade-confirmed stale (5m window):", "")
    table.add_row("  NO <= .97", str(len(t97)))
    table.add_row("  NO <= .95", str(len(t95)))
    table.add_row("  NO <= .90", str(len(t90)))
    console.print(table)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kalshi LOCKED latency probe (structural falsification)"
    )
    ap.add_argument("--series", default="KXHIGHNY")
    ap.add_argument("--start", default="2025-01-01", help="first market date (inclusive)")
    ap.add_argument("--end", default="2026-07-31", help="last market date (inclusive)")
    ap.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="extra F above the cap before a bucket counts as locked (default 0: METAR is the info flow)",
    )
    ap.add_argument(
        "--output", default=None, help="csv path (default data/resolver/kalshi-lock-probe.csv)"
    )
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    try:
        city = load_cities().by_series(args.series)
        station, tz_name = city.station_id, city.timezone
    except KeyError:
        console.print(
            f"[yellow]series {args.series} not in config/cities.yaml — using KNYC / America/New_York[/yellow]"
        )
        station, tz_name = "KNYC", "America/New_York"
    tz = ZoneInfo(tz_name)

    out_path = (
        Path(args.output) if args.output else data_root() / "resolver" / "kalshi-lock-probe.csv"
    )
    metar = fetch_metar(
        station, start - timedelta(days=1), end + timedelta(days=1), out_path.parent
    )
    console.print(f"[dim]METAR rows: {len(metar):,}[/dim]")

    win_start = datetime(start.year, start.month, start.day, tzinfo=UTC)
    win_end = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)

    rows: list[dict] = []
    skipped = 0
    with KalshiClient() as client:
        ev_date: dict[str, date] = {}
        for e in client.get_events(args.series, status="settled"):
            sd = e.get("strike_date")
            if sd:
                with contextlib.suppress(ValueError, TypeError):
                    ev_date[e["event_ticker"]] = parse_api_ts(sd).astimezone(tz).date()
        # split at the historical cutoff: get_markets refuses straddling windows
        cutoff = client.historical_cutoff()
        markets: list[dict] = []
        if win_start < cutoff:
            markets += client.get_markets(
                series_ticker=args.series,
                status="settled",
                start=win_start,
                end=min(win_end, cutoff),
            )
        if win_end > cutoff:
            markets += client.get_markets(
                series_ticker=args.series,
                status="settled",
                start=max(win_start, cutoff) + timedelta(seconds=1),
                end=win_end,
            )
        console.print(f"[dim]markets in window: {len(markets)}[/dim]")
        total_events = len({m["event_ticker"] for m in markets})
        locked_seen = 0
        for m in markets:
            day = ev_date.get(m["event_ticker"])
            floor = locked_floor(m, args.buffer)
            if day is None or floor is None:
                continue
            lock = first_lock(metar, day, floor, tz)
            if lock is None:
                continue
            locked_seen += 1
            lock_at, observed_max = lock
            try:
                rec = probe_market(client, args.series, m, lock_at, parse_api_ts(m["close_time"]))
            except KalshiError as exc:
                skipped += 1
                console.print(f"[dim]skip {m['ticker']}: {exc}[/dim]")
                continue
            rec["observed_max"] = observed_max
            rows.append(rec)
            if locked_seen % 50 == 0:
                console.print(f"[dim]{locked_seen} locked buckets probed...[/dim]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "market_ticker",
        "bucket",
        "lock_at",
        "observed_max",
        "first_quote_at",
        "no_ask_at_lock",
        "reaction_97_s",
        "reaction_99_s",
        "stale_trade_count_5m",
        "stale_trade_volume_5m",
        "worst_stale_trade_price",
        "final_result",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    console.print(f"[dim]{len(rows)} rows -> {out_path}[/dim]")
    summary(rows, total_events, skipped)


if __name__ == "__main__":
    sys.exit(main())
