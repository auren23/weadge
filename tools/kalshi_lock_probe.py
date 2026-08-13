"""Kalshi LOCKED latency probe — structural falsification + econ simulation.

Question: after a Daily High bucket is confirmed dead by public METAR
observations, does the market price lag?

  KNYC METAR (IEM routine + SPECI)  ->  first LOCKED instant per bucket
  Kalshi 1m YES bid/ask candles     ->  first executable quote after lock
  Kalshi trade prints               ->  trade-confirmed stale prints

No-lookahead: the earliest usable quote is the first 1m bar that STARTS at
or after effective_lock_at + 1 minute — never the bar containing the
observation.

Evidence tiers: trade-confirmed (hardest) / quote-indicated / unknown.

Deliberately standalone: tools/, never imported by the resolver.

Run:
  uv run python tools/kalshi_lock_probe.py --series KXHIGHNY \
      --start 2025-01-01 --end 2026-07-31                # single cell
  ... --matrix                                           # 3 buffer x 4 delay grid
  ... --simulate --max-entry 0.90,0.95,0.97              # $5 hold-to-settlement

Raw bars/trades are cached in probe-cache.json so the matrix and the
simulation rerun locally without touching the Kalshi API.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
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
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
console = Console()


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ts(dt: datetime) -> float:
    return (dt - EPOCH).total_seconds()


def _from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


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
                time.sleep(30 * (attempt + 1))  # IEM IP-based rate limit
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
    """(metar_ts, observed_max) when running METAR max first hits floor.

    Day boundary follows the settlement window, NOT the local calendar day:
    NWS Daily Climate Report / Kalshi settle on midnight-to-midnight local
    STANDARD time, i.e. [D 05:00 UTC, D+1 05:00 UTC) for America/New_York
    year-round. A naive local-day window misplaces the 00:00-01:00 DST hour
    into the wrong report day (verified: 2/1138 false LOCKED buckets in the
    original run, e.g. KXHIGHNY-26MAR27-B64.5 locked on a 66F METAR that
    belonged to the previous report day).
    """
    std_offset = tz.utcoffset(datetime(day.year, 1, 1))  # standard (non-DST) offset
    start_utc = datetime(day.year, day.month, day.day, tzinfo=UTC) - std_offset
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


# ---------------------------------------------------------------- cache
def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {"markets": {}, "data": {}}


def save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    tmp.replace(path)


def fetch_raw(
    client: KalshiClient,
    series: str,
    market: dict,
    lock_at: datetime,
    close_at: datetime,
) -> dict:
    """bars: [[bar_start_epoch, yes_bid_open]...] sorted; trades: [[ts, no_price, count]...]."""
    ticker = market["ticker"]
    bars: list[list[float]] = []
    for c in client.get_market_candles(
        ticker, series, lock_at - timedelta(minutes=10), close_at, period_interval_min=1
    ):
        bid = c.get("yes_bid") or {}
        open_bid = _f(bid.get("open_dollars") or bid.get("open"))
        if open_bid is None:
            continue
        bar_end = parse_api_ts(c["end_period_ts"])
        bars.append([_ts(bar_end - timedelta(minutes=1)), open_bid])
    bars.sort(key=lambda b: b[0])

    trades: list[list[float]] = []
    for p in client.get_market_trades(ticker, close_at=close_at):
        trades.append(
            [
                _ts(parse_api_ts(p["created_time"])),
                _f(p.get("no_price_dollars")) or -1.0,
                _f(p.get("count_fp") or p.get("count")) or 0.0,
            ]
        )
    trades.sort(key=lambda t: t[0])
    return {"bars": bars, "trades": trades}


# ---------------------------------------------------------------- compute
def compute(
    metar_lock_at: datetime,
    observed_max: float,
    floor: float,
    delay_min: int,
    raw: dict,
) -> dict:
    """All lock-relative metrics for one bucket under a given info delay.

    effective_lock_at = metar_ts + info_delay — the moment we assume the
    information was actually usable by a trader. Everything (first quote,
    reaction, stale window) is measured from it.
    """
    eff = metar_lock_at + timedelta(minutes=delay_min)
    first_ts = eff + timedelta(minutes=1)  # never the bar containing the observation

    first_quote: datetime | None = None
    no_ask_at_lock: float | None = None
    reaction_97: float | None = None  # seconds eff -> quote implies NO ask >= 0.97
    reaction_99: float | None = None  # ... NO ask >= 0.99
    for bs, bid in raw["bars"]:
        bar_start = _from_ts(bs)
        if bar_start < first_ts:
            continue
        if first_quote is None:
            first_quote = bar_start
            no_ask_at_lock = round(1.0 - bid, 4)
        if reaction_97 is None and bid <= 0.03:  # NO ask >= 0.97  <=>  YES bid <= 0.03
            reaction_97 = (bar_start - eff).total_seconds()
        if reaction_99 is None and bid <= 0.01:  # NO ask >= 0.99  <=>  YES bid <= 0.01
            reaction_99 = (bar_start - eff).total_seconds()
        if first_quote is not None and reaction_97 is not None and reaction_99 is not None:
            break

    win_start = _ts(eff)
    win_end = _ts(eff + timedelta(minutes=5))
    stale = [
        (no_p, cnt)
        for ts, no_p, cnt in raw["trades"]
        if win_start <= ts < win_end and 0 < no_p <= 0.97
    ]
    return {
        "lock_at": eff,
        "observed_max": observed_max,
        "distance_f": round(observed_max - floor, 2),
        "first_quote_at": first_quote,
        "no_ask_at_lock": no_ask_at_lock,
        "reaction_97_s": reaction_97,
        "reaction_99_s": reaction_99,
        "stale_trade_count_5m": len(stale),
        "stale_trade_volume_5m": round(sum(c for _, c in stale), 1),
        "worst_stale_trade_price": min((p for p, _ in stale), default=None),
        "stale_prints": stale,  # [(no_price, count)...] for the econ sim
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
def summary(rows: list[dict], total_events: int, skipped: int, delay_min: int) -> None:
    asks = [r["no_ask_at_lock"] for r in rows if r["no_ask_at_lock"] is not None]
    no_quote = sum(1 for r in rows if r["first_quote_at"] is None)
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

    table = Table(
        title=f"Kalshi LOCKED latency (info delay {delay_min}m) — {len(rows)} locked buckets"
    )
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

    # half-year decay
    hy = Table(title="trade-confirmed stale by half-year")
    hy.add_column("window")
    for label in ("buckets", "<=.97", "<=.95", "<=.90"):
        hy.add_column(label)
    for lo, hi, label in (
        (date(2025, 1, 1), date(2025, 7, 1), "2025 H1"),
        (date(2025, 7, 1), date(2026, 1, 1), "2025 H2"),
        (date(2026, 1, 1), date(2026, 7, 1), "2026 H1"),
        (date(2026, 7, 1), date(2027, 1, 1), "2026 H2"),
    ):
        sub = [r for r in rows if lo <= r["lock_at"].date() < hi]
        if not sub:
            continue
        hy.add_row(
            label,
            str(len(sub)),
            str(sum(1 for r in sub if (r["worst_stale_trade_price"] or 1) <= 0.97)),
            str(sum(1 for r in sub if (r["worst_stale_trade_price"] or 1) <= 0.95)),
            str(sum(1 for r in sub if (r["worst_stale_trade_price"] or 1) <= 0.90)),
        )
    console.print(hy)

    # local hour of lock
    hour_buckets = [(10, 12), (12, 14), (14, 16), (16, 18), (18, 30)]
    hh = Table(title="lock hour (ET) -> stale probability")
    hh.add_column("hour")
    for label in ("buckets", "stale<=.95", "stale%", "avg worst price"):
        hh.add_column(label)
    for lo, hi in hour_buckets:
        sub = [
            r for r in rows if lo <= r["lock_at"].astimezone(ZoneInfo("America/New_York")).hour < hi
        ]
        if not sub:
            continue
        st = [r for r in sub if (r["worst_stale_trade_price"] or 1) <= 0.95]
        avg_worst = statistics.mean(r["worst_stale_trade_price"] for r in st) if st else None
        hh.add_row(
            f"{lo}-{hi}",
            str(len(sub)),
            str(len(st)),
            f"{100 * len(st) / len(sub):.0f}%",
            f"{avg_worst:.3f}" if avg_worst is not None else "-",
        )
    console.print(hh)

    # distance past the lock threshold
    dd = Table(title="distance past lock threshold (F) -> stale probability")
    dd.add_column("distance")
    for label in ("buckets", "stale<=.95", "stale%", "avg worst price"):
        dd.add_column(label)
    for lo, hi, label in ((0, 0.5, "0-0.5"), (0.5, 1, "0.5-1"), (1, 2, "1-2"), (2, 1e9, ">2")):
        sub = [r for r in rows if lo <= r["distance_f"] < hi]
        if not sub:
            continue
        st = [r for r in sub if (r["worst_stale_trade_price"] or 1) <= 0.95]
        avg_worst = statistics.mean(r["worst_stale_trade_price"] for r in st) if st else None
        dd.add_row(
            label,
            str(len(sub)),
            str(len(st)),
            f"{100 * len(st) / len(sub):.0f}%",
            f"{avg_worst:.3f}" if avg_worst is not None else "-",
        )
    console.print(dd)


# ---------------------------------------------------------------- econ sim
def taker_fee(price: float) -> float:
    """Kalshi taker fee per share: max($0.01, 7% x min(p, 1-p)).

    Conservative approximation; a fee_changes replay belongs to a later
    economic pass, not this probe.
    """
    return max(0.01, 0.07 * min(price, 1.0 - price))


def simulate(rows: list[dict], max_entries: list[float]) -> None:
    """$5 fixed-size, buy first trade-confirmed NO print <= max_entry within
    5m of effective lock, hold to settlement (LOCKED -> NO resolves $1).

    Fill model (conservative): a fill is only allowed where a real print at
    <= max_entry exists after lock (trade-confirmed). Fill price = that
    print's price; size = floor($5 / price). No exit assumptions needed.
    """
    table = Table(title="econ sim: $5 fixed, hold to settlement (fee = max($0.01, 7%*min(p,1-p)))")
    table.add_column("max_entry")
    for label in ("fills", "pnl", "win%", "2025H1", "2025H2", "2026H1", "2026H2"):
        table.add_column(label)
    for me in max_entries:
        fills = 0
        pnl_total = 0.0
        wins = 0
        per_hy: dict[str, float] = {}
        for r in rows:
            # first print at <= max_entry after effective lock (tape order)
            fill = next((p for p, _ in r["stale_prints"] if p <= me), None)
            if fill is None:
                continue
            fills += 1
            shares = math.floor(5.0 / fill)
            pnl = (1.0 - fill - taker_fee(fill)) * shares
            pnl_total += pnl
            wins += 1
            hy = f"{r['lock_at'].year} {'H1' if r['lock_at'].month <= 6 else 'H2'}"
            per_hy[hy] = per_hy.get(hy, 0.0) + pnl
        table.add_row(
            f"{me:.2f}",
            str(fills),
            f"${pnl_total:,.0f}",
            f"{100 * wins / fills:.0f}%" if fills else "-",
            *(f"${per_hy.get(k, 0):,.0f}" for k in ("2025 H1", "2025 H2", "2026 H1", "2026 H2")),
        )
    console.print(table)


# ---------------------------------------------------------------- matrix
def matrix(rows_by_cell: dict[tuple[float, int], list[dict]]) -> None:
    """buffer x info-delay grid of trade-confirmed stale counts (local)."""
    buffers = (0.0, 0.5, 1.0)
    delays = (0, 1, 2, 5)
    for threshold in (0.97, 0.95, 0.90):
        table = Table(title=f"trade-confirmed stale NO<={threshold:.2f} (5m window)")
        table.add_column("buffer \\ delay")
        for d in delays:
            table.add_column(f"{d}m")
        for b in buffers:
            row = []
            for d in delays:
                rows_ = rows_by_cell.get((b, d), [])
                row.append(
                    str(sum(1 for r in rows_ if (r["worst_stale_trade_price"] or 1) <= threshold))
                )
            table.add_row(f"{b:g}F", *row)
        console.print(table)


def run_cell(
    metar: list[tuple[datetime, float]],
    cache: dict,
    tz: ZoneInfo,
    buffer_f: float,
    delay_min: int,
) -> list[dict]:
    """Compute one buffer x delay cell entirely from cache (no API)."""
    out = []
    for ticker, m in cache["markets"].items():
        floor = locked_floor(m, buffer_f)
        if floor is None:
            continue
        lock = first_lock(metar, date.fromisoformat(m["day"]), floor, tz)
        if lock is None:
            continue
        metar_ts, observed = lock
        raw = cache["data"].get(ticker)
        if raw is None:
            continue
        rec = compute(metar_ts, observed, floor, delay_min, raw)
        rec["market_ticker"] = ticker
        rec["bucket"] = bucket_label(m)
        rec["final_result"] = m.get("result")
        rec["day"] = m["day"]
        out.append(rec)
    return out


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
        help="extra F above the cap before a bucket counts as locked",
    )
    ap.add_argument(
        "--info-delay",
        type=int,
        default=0,
        choices=(0, 1, 2, 5),
        help="assumed minutes between METAR valid time and trader-usable data",
    )
    ap.add_argument(
        "--matrix", action="store_true", help="3 buffer x 4 delay grid (offline, uses cache)"
    )
    ap.add_argument("--simulate", action="store_true", help="$5 fixed hold-to-settlement econ sim")
    ap.add_argument("--max-entry", default="0.90,0.95,0.97")
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
    cache_path = out_path.with_name("probe-cache.json")
    metar = fetch_metar(
        station, start - timedelta(days=1), end + timedelta(days=1), out_path.parent
    )
    console.print(f"[dim]METAR rows: {len(metar):,}[/dim]")

    win_start = datetime(start.year, start.month, start.day, tzinfo=UTC)
    win_end = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)

    cache = load_cache(cache_path)
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

        # cache market metadata (strikes/result/day) once
        for m in markets:
            day = ev_date.get(m["event_ticker"])
            if day is None:
                continue
            cache["markets"].setdefault(
                m["ticker"],
                {
                    "floor_strike": m.get("floor_strike"),
                    "cap_strike": m.get("cap_strike"),
                    "result": m.get("result"),
                    "close_time": m["close_time"],
                    "day": day.isoformat(),
                },
            )

        # fetch raw bars/trades for locked buckets only, cached
        skipped = 0
        for m in markets:
            day = ev_date.get(m["event_ticker"])
            floor = locked_floor(m, args.buffer)
            if day is None or floor is None:
                continue
            lock = first_lock(metar, day, floor, tz)
            if lock is None:
                continue
            if m["ticker"] in cache["data"]:
                continue
            try:
                raw = fetch_raw(client, args.series, m, lock[0], parse_api_ts(m["close_time"]))
            except KalshiError as exc:
                skipped += 1
                console.print(f"[dim]skip {m['ticker']}: {exc}[/dim]")
                continue
            cache["data"][m["ticker"]] = raw
            if len(cache["data"]) % 50 == 0:
                save_cache(cache_path, cache)
    total_events = len({m["event_ticker"] for m in markets})
    save_cache(cache_path, cache)

    if args.matrix:
        cells: dict[tuple[float, int], list[dict]] = {}
        for b in (0.0, 0.5, 1.0):
            for d in (0, 1, 2, 5):
                cells[(b, d)] = run_cell(metar, cache, tz, b, d)
        matrix(cells)
        return

    rows = run_cell(metar, cache, tz, args.buffer, args.info_delay)
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
            w.writerow(
                {
                    "date": r["day"],
                    "market_ticker": r["market_ticker"],
                    "bucket": r["bucket"],
                    "lock_at": r["lock_at"].isoformat(),
                    "observed_max": r["observed_max"],
                    "first_quote_at": r["first_quote_at"].isoformat()
                    if r["first_quote_at"]
                    else "",
                    "no_ask_at_lock": r["no_ask_at_lock"],
                    "reaction_97_s": r["reaction_97_s"],
                    "reaction_99_s": r["reaction_99_s"],
                    "stale_trade_count_5m": r["stale_trade_count_5m"],
                    "stale_trade_volume_5m": r["stale_trade_volume_5m"],
                    "worst_stale_trade_price": r["worst_stale_trade_price"],
                    "final_result": r["final_result"],
                }
            )
    console.print(f"[dim]{len(rows)} rows -> {out_path}[/dim]")
    summary(rows, total_events, skipped, args.info_delay)
    if args.simulate:
        simulate(rows, [float(x) for x in args.max_entry.split(",")])


if __name__ == "__main__":
    sys.exit(main())
