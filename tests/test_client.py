"""Kalshi client: live/historical routing, retry/backoff, pagination, parsing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from weadge.adapters.kalshi.candles import candles_frame
from weadge.adapters.kalshi.client import LIVE_BASE, KalshiClient, KalshiError, RateLimiter
from weadge.adapters.kalshi.fees import fee_changes_frame
from weadge.adapters.kalshi.forecasts import forecast_percentile_frame
from weadge.adapters.kalshi.markets import events_frame, markets_frame, series_frame

CUTOFF = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def make_client() -> KalshiClient:
    return KalshiClient(
        base_url=LIVE_BASE,
        min_interval_s=0.0,
        limiter=RateLimiter(min_interval_s=0.0, backoff_base_s=0.005, max_retries=3),
    )


def mock_cutoff() -> None:
    respx.get(f"{LIVE_BASE}/historical/cutoff").mock(
        return_value=httpx.Response(200, json={"cutoff": int(CUTOFF.timestamp())})
    )


class TestRouting:
    @respx.mock
    def test_window_before_cutoff_uses_historical(self) -> None:
        mock_cutoff()
        hist = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
        )
        live = respx.get(url__startswith=f"{LIVE_BASE}/markets").mock(
            return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
        )
        with make_client() as c:
            c.get_markets(
                series_ticker="KXHIGHNY",
                start=CUTOFF - timedelta(days=10),
                end=CUTOFF - timedelta(days=1),
            )
        assert hist.called
        assert not live.called

    @respx.mock
    def test_window_after_cutoff_uses_live(self) -> None:
        mock_cutoff()
        hist = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
        )
        live = respx.get(url__startswith=f"{LIVE_BASE}/markets").mock(
            return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
        )
        with make_client() as c:
            c.get_markets(
                series_ticker="KXHIGHNY",
                start=CUTOFF + timedelta(days=1),
                end=CUTOFF + timedelta(days=10),
            )
        assert live.called
        assert not hist.called

    @respx.mock
    def test_straddling_window_raises(self) -> None:
        mock_cutoff()
        with make_client() as c, pytest.raises(Exception, match="straddles"):
            c.get_markets(
                series_ticker="KXHIGHNY",
                start=CUTOFF - timedelta(days=1),
                end=CUTOFF + timedelta(days=1),
            )


class TestRetry:
    @respx.mock
    def test_429_retries_then_succeeds(self) -> None:
        mock_cutoff()
        respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limited"}),
                httpx.Response(200, json={"markets": [{"ticker": "M1"}], "cursor": ""}),
            ]
        )
        with make_client() as c:
            out = c.get_markets(series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=2),
                                end=CUTOFF - timedelta(days=1))
        assert len(out) == 1

    @respx.mock
    def test_5xx_retries_then_raises(self) -> None:
        mock_cutoff()
        route = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            return_value=httpx.Response(503, json={"error": "down"})
        )
        with make_client() as c, pytest.raises(KalshiError):
            c.get_markets(series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=2),
                          end=CUTOFF - timedelta(days=1))
        assert route.call_count >= 2  # retried, not just failed once

    @respx.mock
    def test_400_is_not_retried(self) -> None:
        mock_cutoff()
        route = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            return_value=httpx.Response(400, json={"error": "bad params"})
        )
        with make_client() as c, pytest.raises(KalshiError):
            c.get_markets(series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=2),
                          end=CUTOFF - timedelta(days=1))
        assert route.call_count == 1  # client error: no retry


class TestPagination:
    @respx.mock
    def test_cursor_pagination_concatenates(self) -> None:
        mock_cutoff()
        route = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "markets": [{"ticker": f"M{i}"} for i in range(2)],
                        "cursor": "abc",
                    },
                ),
                httpx.Response(200, json={"markets": [{"ticker": "M9"}], "cursor": ""}),
            ]
        )
        with make_client() as c:
            out = c.get_markets(series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=2),
                                end=CUTOFF - timedelta(days=1))
        assert [m["ticker"] for m in out] == ["M0", "M1", "M9"]
        assert route.call_count == 2
        second_req = route.calls[1].request
        assert "cursor" in str(second_req.url)


class TestParsing:
    @respx.mock
    def test_series_frame(self) -> None:
        respx.get(f"{LIVE_BASE}/series/KXHIGHNY").mock(
            return_value=httpx.Response(
                200,
                json={
                    "series": {
                        "ticker": "KXHIGHNY",
                        "title": "NY High Temp",
                        "settlement_source": "NWS",
                        "fee_type": "taker",
                        "fee_multiplier": "0.07",
                    }
                },
            )
        )
        with make_client() as c:
            df = series_frame(c, "KXHIGHNY")
        assert df["series_ticker"][0] == "KXHIGHNY"
        assert df["fee_multiplier"][0] == 0.07  # string coerced to float

    @respx.mock
    def test_events_frame(self) -> None:
        respx.get(url__startswith=f"{LIVE_BASE}/events").mock(
            return_value=httpx.Response(
                200,
                json={
                    "events": [
                        {"event_ticker": "E1", "target_date": "2026-07-01", "location": "NY"},
                    ],
                    "cursor": "",
                },
            )
        )
        with make_client() as c:
            df = events_frame(c, "KXHIGHNY")
        assert df["event_ticker"][0] == "E1"
        assert df["target_date"][0] == datetime(2026, 7, 1, tzinfo=UTC)

    @respx.mock
    def test_markets_frame_epoch_to_utc(self) -> None:
        mock_cutoff()
        ts = int(CUTOFF.timestamp())
        respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            return_value=httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "M1",
                            "event_ticker": "E1",
                            "series_ticker": "KXHIGHNY",
                            "floor_strike": 90.0,
                            "cap_strike": 92.0,
                            "open_time": ts,
                            "close_time": ts + 86400,
                            "settlement_time": ts + 172800,
                            "result": "yes",
                            "settlement_value": "yes",
                        }
                    ],
                    "cursor": "",
                },
            )
        )
        with make_client() as c:
            df = markets_frame(c, series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=1),
                               end=CUTOFF)
        assert df["result"][0] == "yes"
        assert df["open_at"][0] == CUTOFF

    @respx.mock
    def test_candles_frame(self) -> None:
        mock_cutoff()
        ts = int(CUTOFF.timestamp()) + 86400  # one day after cutoff -> live API
        start = CUTOFF + timedelta(days=1)
        end = CUTOFF + timedelta(days=2)
        respx.get(url__startswith=f"{LIVE_BASE}/markets/M1/candlesticks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "start_ts": ts,
                            "end_ts": ts + 60,
                            "yes_bid": {"open": 0.40, "high": 0.41, "low": 0.39, "close": 0.405},
                            "yes_ask": {"open": 0.45, "high": 0.46, "low": 0.44, "close": 0.455},
                            "volume": 12,
                            "open_interest": 99,
                        }
                    ]
                },
            )
        )
        with make_client() as c:
            df = candles_frame(c, "M1", start, end)
        assert df["yes_ask_close"][0] == 0.455
        assert df["ts"][0] == start

    @respx.mock
    def test_forecast_percentile_frame(self) -> None:
        ts = int(CUTOFF.timestamp())
        respx.get(url__startswith=f"{LIVE_BASE}/events/E1/forecast_percentile_history").mock(
            return_value=httpx.Response(
                200,
                json={
                    "forecast_percentile_history": [
                        {"end_period_ts": ts, "percentile": 50.0,
                         "numerical_forecast": 90.0, "formatted_forecast": "90"},
                        {"end_period_ts": ts, "percentile": 90.0,
                         "numerical_forecast": 92.0, "formatted_forecast": "92"},
                    ]
                },
            )
        )
        with make_client() as c:
            df = forecast_percentile_frame(c, "E1", "KXHIGHNY")
        assert df.height == 2
        assert df["percentile"].to_list() == [50.0, 90.0]

    @respx.mock
    def test_fee_changes_frame(self) -> None:
        ts = int(CUTOFF.timestamp())
        respx.get(url__startswith=f"{LIVE_BASE}/series/KXHIGHNY/fee_changes").mock(
            return_value=httpx.Response(
                200,
                json={
                    "fee_changes": [
                        {"effective_time": ts, "fee_multiplier": "0.07", "fee_type": "taker"},
                    ]
                },
            )
        )
        with make_client() as c:
            df = fee_changes_frame(c, "KXHIGHNY")
        assert df["fee_multiplier"][0] == 0.07
        assert df["effective_at"][0] == CUTOFF
