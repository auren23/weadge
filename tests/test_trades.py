"""Trade-print endpoint: routing, pagination params, and frame parsing.

Mocks mirror the VERIFIED 2026-08-12 wire contract: live /markets/trades
vs historical /historical/trades (no /markets segment), trades[] with
count_fp / *_price_dollars fixed-point strings and sub-second ISO
created_time, newest first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import respx

from weadge.adapters.kalshi.client import LIVE_BASE, KalshiClient, RateLimiter
from weadge.adapters.kalshi.trades import trades_frame

CUTOFF = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def make_client() -> KalshiClient:
    return KalshiClient(
        base_url=LIVE_BASE,
        min_interval_s=0.0,
        limiter=RateLimiter(min_interval_s=0.0, backoff_base_s=0.005, max_retries=3),
    )


def mock_cutoff() -> None:
    respx.get(f"{LIVE_BASE}/historical/cutoff").mock(
        return_value=httpx.Response(
            200,
            json={"market_settled_ts": CUTOFF.strftime("%Y-%m-%dT%H:%M:%SZ")},
        )
    )


def _print(created: str, yes: str, taker: str, count: str = "1.00") -> dict:
    return {
        "count_fp": count,
        "created_time": created,
        "is_block_trade": False,
        "no_price_dollars": f"{1 - float(yes):.4f}",
        "taker_book_side": "bid",
        "taker_outcome_side": taker,
        "taker_side": taker,
        "ticker": "M1",
        "trade_id": f"id-{created}",
        "yes_price_dollars": yes,
    }


class TestTradeRouting:
    @respx.mock
    def test_pre_cutoff_close_uses_historical_trades_path(self) -> None:
        """Historical trades live at /historical/trades — NOT
        /historical/markets/trades (verified 2026-08-12)."""
        mock_cutoff()
        hist = respx.get(url__startswith=f"{LIVE_BASE}/historical/trades").mock(
            return_value=httpx.Response(200, json={"trades": [], "cursor": ""})
        )
        live = respx.get(url__startswith=f"{LIVE_BASE}/markets/trades").mock(
            return_value=httpx.Response(200, json={"trades": [], "cursor": ""})
        )
        with make_client() as c:
            c.get_market_trades("M1", close_at=CUTOFF - timedelta(days=1))
        assert hist.called
        assert not live.called
        query = str(hist.calls[0].request.url)
        assert "ticker=M1" in query
        assert "limit=1000" in query

    @respx.mock
    def test_post_cutoff_or_unknown_close_uses_live_path(self) -> None:
        mock_cutoff()
        live = respx.get(url__startswith=f"{LIVE_BASE}/markets/trades").mock(
            return_value=httpx.Response(200, json={"trades": [], "cursor": ""})
        )
        with make_client() as c:
            c.get_market_trades("M1", close_at=CUTOFF + timedelta(days=1))
            c.get_market_trades("M1")  # no close_at -> live, no cutoff call needed
        assert live.call_count == 2

    @respx.mock
    def test_cursor_pagination_concatenates(self) -> None:
        mock_cutoff()
        route = respx.get(url__startswith=f"{LIVE_BASE}/markets/trades").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "trades": [_print("2026-07-20T02:59:59.52758Z", "0.0010", "yes")],
                        "cursor": "abc",
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "trades": [_print("2026-07-20T02:59:54.919509Z", "0.0010", "yes")],
                        "cursor": "",
                    },
                ),
            ]
        )
        with make_client() as c:
            out = c.get_market_trades("M1")
        assert len(out) == 2
        assert "cursor=abc" in str(route.calls[1].request.url)


class TestTradesFrame:
    @respx.mock
    def test_parses_verified_wire_shape_and_sorts_ascending(self) -> None:
        """API returns newest first; the frame is oldest first with floats
        parsed from the *_dollars / count_fp fixed-point strings."""
        mock_cutoff()
        respx.get(url__startswith=f"{LIVE_BASE}/markets/trades").mock(
            return_value=httpx.Response(
                200,
                json={
                    "trades": [
                        _print("2026-07-20T02:59:59.527580Z", "0.3000", "yes", "4.36"),
                        _print("2026-07-20T02:59:54.919509Z", "0.2800", "no", "925.40"),
                    ],
                    "cursor": "",
                },
            )
        )
        with make_client() as c:
            df = trades_frame(c, "M1")
        assert df.height == 2
        first = df.row(0, named=True)  # oldest print first
        assert first["yes_price"] == 0.28
        assert first["no_price"] == 0.72
        assert first["count"] == 925.40
        assert first["taker_side"] == "no"
        assert first["created_at"] == datetime(2026, 7, 20, 2, 59, 54, 919509, tzinfo=UTC)
        assert first["is_block_trade"] is False
        assert df["created_at"].is_sorted()
