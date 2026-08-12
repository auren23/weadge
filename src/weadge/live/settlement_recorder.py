"""Settlement-grade live recorder for the KXBTC15M research program.

One process records, timestamps and append-only persists every message
from the venues that jointly determine whether sub-minute crypto alpha
exists — the question 1m historical candles can never answer:

    kalshi    REST poll (PUBLIC, verified 2026-08-12): the single open
              KXBTC15M market's full orderbook every second + trade prints
              + market metadata on roll. WS L2 needs an API key; the REST
              book at 1s is enough for basis/absorption work at 1s
              resolution. ponytail: sub-second Kalshi absorption stays
              unmeasurable until KALSHI_API_KEY exists — upgrade path is a
              ws task alongside the poller, format unchanged.
    kraken    public WS v2, BTC/USD trades + ticker   (BRTI constituent)
    coinbase  public WS, BTC-USD matches + ticker     (BRTI constituent)
    bitstamp  public WS, btcusd live trades           (BRTI constituent)
    cfb       BRTI value stream at PER_SECOND — REQUIRES a CF Benchmarks
              license (Basic auth on wss://www.cfbenchmarks.com/ws/v4).
              Enabled only when CFB_API_USER / CFB_API_KEY are set; until
              then the constituent feeds above are the composite proxy.

Format: data/raw/live/<YYYY-MM-DD>/<source>/<HH>.jsonl.zst, one line per
message: {"ts": <received_at UTC ISO>, "raw": <payload untouched>}.
Nothing is parsed, normalized or dropped at record time — interpretation
belongs to replay code, recording is evidence capture. No trading.

    uv run python -m weadge.live.settlement_recorder [data_root]
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from weadge.domain.time import utc_now
from weadge.live.recorder import JsonlZstAppender

logger = logging.getLogger("weadge.live.settlement_recorder")

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"

BOOK_POLL_S = 1.0
TRADES_POLL_S = 2.0
MARKETS_POLL_S = 10.0
FLUSH_S = 5.0
SEEN_TRADES_MAX = 5000

# BRTI constituent venues, public feeds. Adding a venue = one more entry;
# payloads are recorded raw so no per-venue parsing exists anywhere.
EXCHANGE_FEEDS: dict[str, tuple[str, list[dict]]] = {
    "kraken": (
        "wss://ws.kraken.com/v2",
        [
            {"method": "subscribe", "params": {"channel": "trade", "symbol": ["BTC/USD"]}},
            {"method": "subscribe", "params": {"channel": "ticker", "symbol": ["BTC/USD"]}},
        ],
    ),
    "coinbase": (
        "wss://ws-feed.exchange.coinbase.com",
        [{"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker", "matches"]}],
    ),
    "bitstamp": (
        "wss://ws.bitstamp.net",
        [{"event": "bts:subscribe", "data": {"channel": "live_trades_btcusd"}}],
    ),
}

CFB_URL = "wss://www.cfbenchmarks.com/ws/v4"


def cfb_subscribe(index_id: str = "BRTI") -> dict:
    """Value-channel subscribe per docs.cfbenchmarks.com/api/websocket/value."""
    return {"type": "subscribe", "stream": "value", "id": index_id, "maxResolution": "PER_SECOND"}


def cfb_auth_header(user: str, key: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def new_trades(seen: set[str], trades: list[dict]) -> list[dict]:
    """Prints not yet recorded, oldest first; mutates `seen` (bounded by
    caller). The trades endpoint has no incremental cursor we verified, so
    dedupe on trade_id is the replay-safe way to poll it."""
    fresh = [t for t in reversed(trades) if t.get("trade_id") not in seen]
    seen.update(t["trade_id"] for t in fresh if t.get("trade_id"))
    return fresh


def _now_iso() -> str:
    return utc_now().isoformat()


def _to_raw(message: str | bytes) -> Any:
    if isinstance(message, bytes):
        message = message.decode(errors="replace")
    try:
        return json.loads(message)
    except ValueError:
        return message


async def record_ws(
    name: str,
    url: str,
    subscriptions: list[dict],
    out: JsonlZstAppender,
    headers: dict[str, str] | None = None,
) -> None:
    """Connect, subscribe, append every message raw. Reconnect forever with
    capped exponential backoff — a recorder's only job is to still be
    running two weeks from now."""
    import websockets

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                url, additional_headers=headers, ping_interval=20, ping_timeout=20
            ) as ws:
                for sub in subscriptions:
                    await ws.send(json.dumps(sub))
                logger.info("%s connected", name)
                backoff = 1.0
                async for message in ws:
                    out.append(_now_iso(), {"raw": _to_raw(message)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # any disconnect: log and rejoin
            logger.warning("%s dropped (%s); reconnecting in %.0fs", name, exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def record_kalshi(out: JsonlZstAppender) -> None:
    """Poll the open KXBTC15M market: 1s orderbook snapshots, 2s trade
    prints (deduped), market metadata on every roll."""
    seen: set[str] = set()
    open_markets: list[dict] = []
    last_markets = last_trades = 0.0
    async with httpx.AsyncClient(base_url=KALSHI_BASE, timeout=10.0) as client:
        while True:
            started = asyncio.get_event_loop().time()
            try:
                if started - last_markets >= MARKETS_POLL_S or not open_markets:
                    body = (
                        await client.get(
                            "/markets", params={"series_ticker": SERIES, "status": "open"}
                        )
                    ).json()
                    fresh = body.get("markets", [])
                    if {m["ticker"] for m in fresh} != {m["ticker"] for m in open_markets}:
                        out.append(_now_iso(), {"raw": {"type": "markets", "markets": fresh}})
                    open_markets = fresh
                    last_markets = started
                for market in open_markets:
                    ticker = market["ticker"]
                    book = (await client.get(f"/markets/{ticker}/orderbook")).json()
                    out.append(_now_iso(), {"raw": {"type": "book", "ticker": ticker, **book}})
                    if started - last_trades >= TRADES_POLL_S:
                        body = (
                            await client.get(
                                "/markets/trades", params={"ticker": ticker, "limit": 100}
                            )
                        ).json()
                        for t in new_trades(seen, body.get("trades", [])):
                            out.append(_now_iso(), {"raw": {"type": "trade", **t}})
                if started - last_trades >= TRADES_POLL_S:
                    last_trades = started
                if len(seen) > SEEN_TRADES_MAX:
                    seen.clear()  # ids roll off the 100-deep poll long before this
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep polling through blips
                logger.warning("kalshi poll error: %s", exc)
            elapsed = asyncio.get_event_loop().time() - started
            await asyncio.sleep(max(BOOK_POLL_S - elapsed, 0.05))


async def flush_forever(appenders: list[JsonlZstAppender]) -> None:
    while True:
        await asyncio.sleep(FLUSH_S)
        for a in appenders:
            a.flush()


async def run(data_root: Path) -> None:
    out_root = data_root / "raw" / "live"
    appenders: dict[str, JsonlZstAppender] = {}

    def out(name: str) -> JsonlZstAppender:
        appenders[name] = JsonlZstAppender(out_root, name)
        return appenders[name]

    tasks = [record_kalshi(out("kalshi"))]
    tasks += [record_ws(name, url, subs, out(name)) for name, (url, subs) in EXCHANGE_FEEDS.items()]
    cfb_user, cfb_key = os.environ.get("CFB_API_USER"), os.environ.get("CFB_API_KEY")
    if cfb_user and cfb_key:
        tasks.append(
            record_ws(
                "cfb", CFB_URL, [cfb_subscribe()], out("cfb"), cfb_auth_header(cfb_user, cfb_key)
            )
        )
    else:
        logger.warning(
            "CFB_API_USER/CFB_API_KEY not set — BRTI stream disabled, "
            "recording constituent proxies only"
        )
    tasks.append(flush_forever(list(appenders.values())))
    try:
        await asyncio.gather(*tasks)
    finally:
        for a in appenders.values():
            a.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    try:
        asyncio.run(run(data_root))
    except KeyboardInterrupt:
        logger.info("recorder stopped")


if __name__ == "__main__":
    main()
