"""Kalshi client: live/historical routing, retry/backoff, pagination, parsing.

Mocks mirror the VERIFIED 2026-08-11 wire contract: ISO cutoff fields,
limit+cursor pagination, end_period_ts candles with *_dollars / bare OHLC
shapes, /series/{s}/events/{e}/forecast_percentile_history with repeated
percentiles, /series/fee_changes with series_fee_change_arr.
"""

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
    """Official cutoff shape: ISO timestamps, no 'cutoff' key."""
    respx.get(f"{LIVE_BASE}/historical/cutoff").mock(
        return_value=httpx.Response(
            200,
            json={
                "market_settled_ts": CUTOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "trades_created_ts": CUTOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "orders_updated_ts": CUTOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
    )


def _empty(path: str) -> None:
    respx.get(url__startswith=path).mock(
        return_value=httpx.Response(200, json={"markets": [], "events": [], "cursor": ""})
    )


class TestRouting:
    @respx.mock
    def test_window_before_cutoff_uses_historical_without_time_filters(self) -> None:
        """Historical markets accept NO min/max_close_ts — the request must
        not carry them; the window is applied client-side."""
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
        query = str(hist.calls[0].request.url)
        assert "min_close_ts" not in query
        assert "max_close_ts" not in query
        assert "limit=" in query

    @respx.mock
    def test_window_after_cutoff_uses_live_with_time_filters(self) -> None:
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
        query = str(live.calls[0].request.url)
        assert "min_close_ts" in query
        assert "max_close_ts" in query

    @respx.mock
    def test_straddling_window_raises(self) -> None:
        mock_cutoff()
        with make_client() as c, pytest.raises(Exception, match="straddles"):
            c.get_markets(
                series_ticker="KXHIGHNY",
                start=CUTOFF - timedelta(days=1),
                end=CUTOFF + timedelta(days=1),
            )

    @respx.mock
    def test_historical_markets_window_filtered_client_side(self) -> None:
        """The series' full market history is paged, then restricted to the
        close window [start, end] locally."""
        mock_cutoff()
        iso = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
        markets = [
            {"ticker": "M1", "event_ticker": "E1", "series_ticker": "KXHIGHNY",
             "close_time": iso(CUTOFF - timedelta(days=5))},   # inside window
            {"ticker": "M2", "event_ticker": "E2", "series_ticker": "KXHIGHNY",
             "close_time": iso(CUTOFF - timedelta(days=50))},  # before window
            {"ticker": "M3", "event_ticker": "E3", "series_ticker": "KXHIGHNY",
             "close_time": iso(CUTOFF - timedelta(days=20))},  # before window
        ]
        respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            return_value=httpx.Response(200, json={"markets": markets, "cursor": ""})
        )
        with make_client() as c:
            out = c.get_markets(
                series_ticker="KXHIGHNY",
                start=CUTOFF - timedelta(days=10),
                end=CUTOFF - timedelta(days=1),
            )
        assert [m["ticker"] for m in out] == ["M1"]


class TestCandleRouting:
    @respx.mock
    def test_pre_cutoff_window_uses_historical_path(self) -> None:
        mock_cutoff()
        hist = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets/M1/candlesticks").mock(
            return_value=httpx.Response(200, json={"candlesticks": []})
        )
        live = respx.get(url__startswith=f"{LIVE_BASE}/series/KXHIGHNY/markets/M1/candlesticks").mock(
            return_value=httpx.Response(200, json={"candlesticks": []})
        )
        with make_client() as c:
            c.get_market_candles(
                "M1", "KXHIGHNY",
                start=CUTOFF - timedelta(days=2),
                end=CUTOFF - timedelta(days=1),
                period_interval_s=60,
            )
        assert hist.called
        assert not live.called
        query = str(hist.calls[0].request.url)
        assert "period_interval=60" in query
        assert "start_ts" in query and "end_ts" in query

    @respx.mock
    def test_post_cutoff_window_uses_live_series_path(self) -> None:
        mock_cutoff()
        hist = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets/M1/candlesticks").mock(
            return_value=httpx.Response(200, json={"candlesticks": []})
        )
        live = respx.get(url__startswith=f"{LIVE_BASE}/series/KXHIGHNY/markets/M1/candlesticks").mock(
            return_value=httpx.Response(200, json={"candlesticks": []})
        )
        with make_client() as c:
            c.get_market_candles(
                "M1", "KXHIGHNY",
                start=CUTOFF + timedelta(days=1),
                end=CUTOFF + timedelta(days=2),
                period_interval_s=60,
            )
        assert live.called
        assert not hist.called

    @respx.mock
    def test_live_candles_require_series_ticker(self) -> None:
        mock_cutoff()
        with make_client() as c, pytest.raises(KalshiError, match="series"):
            c.get_market_candles(
                "M1", None,
                start=CUTOFF + timedelta(days=1),
                end=CUTOFF + timedelta(days=2),
            )


class TestRetry:
    @respx.mock
    def test_429_retries_then_succeeds(self) -> None:
        mock_cutoff()
        iso = (CUTOFF - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limited"}),
                httpx.Response(
                    200,
                    json={"markets": [{"ticker": "M1", "close_time": iso}], "cursor": ""},
                ),
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
    def test_cursor_pagination_uses_limit_and_concatenates(self) -> None:
        mock_cutoff()
        iso = (CUTOFF - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        route = respx.get(url__startswith=f"{LIVE_BASE}/historical/markets").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "markets": [{"ticker": f"M{i}", "close_time": iso} for i in range(2)],
                        "cursor": "abc",
                    },
                ),
                httpx.Response(200, json={"markets": [{"ticker": "M9", "close_time": iso}], "cursor": ""}),
            ]
        )
        with make_client() as c:
            out = c.get_markets(series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=2),
                                end=CUTOFF - timedelta(days=1))
        assert [m["ticker"] for m in out] == ["M0", "M1", "M9"]
        assert route.call_count == 2
        assert "limit=1000" in str(route.calls[0].request.url)
        assert "cursor" in str(route.calls[1].request.url)

    @respx.mock
    def test_events_use_events_limit_cap(self) -> None:
        route = respx.get(url__startswith=f"{LIVE_BASE}/events").mock(
            return_value=httpx.Response(200, json={"events": [], "cursor": ""})
        )
        with make_client() as c:
            c.get_events("KXHIGHNY")
        assert "limit=200" in str(route.calls[0].request.url)

    @respx.mock
    def test_events_window_filtered_client_side_on_strike_date(self) -> None:
        """The live events endpoint ignores time filters — the window is
        applied locally on strike_date (there is no /historical/events)."""
        events = [
            {"event_ticker": "E1", "strike_date": "2026-07-15T03:59:00Z"},   # inside
            {"event_ticker": "E2", "strike_date": "2026-06-01T03:59:00Z"},   # before
            {"event_ticker": "E3", "strike_date": "2026-08-20T03:59:00Z"},   # after
        ]
        route = respx.get(url__startswith=f"{LIVE_BASE}/events").mock(
            return_value=httpx.Response(200, json={"events": events, "cursor": ""})
        )
        with make_client() as c:
            out = c.get_events(
                "KXHIGHNY",
                start=datetime(2026, 7, 1, tzinfo=UTC),
                end=datetime(2026, 8, 1, tzinfo=UTC),
            )
        assert [e["event_ticker"] for e in out] == ["E1"]
        query = str(route.calls[0].request.url)
        assert "min_close_ts" not in query  # never sent — live endpoint ignores it


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
                        "fee_type": "quadratic",
                        "fee_multiplier": "1",
                    }
                },
            )
        )
        with make_client() as c:
            df = series_frame(c, "KXHIGHNY")
        assert df["series_ticker"][0] == "KXHIGHNY"
        assert df["fee_multiplier"][0] == 1.0  # string coerced to float

    @respx.mock
    def test_events_frame_derives_target_date_from_strike_date(self) -> None:
        """Regression: the live API has NO target_date field — the target is
        the strike's local (America/New_York) calendar date."""
        respx.get(url__startswith=f"{LIVE_BASE}/events").mock(
            return_value=httpx.Response(
                200,
                json={
                    "events": [
                        # strike 2026-08-11T03:59:00Z == 23:59 EDT Aug 10 -> target Aug 10
                        {"event_ticker": "E1", "strike_date": "2026-08-11T03:59:00Z",
                         "location": "NY"},
                    ],
                    "cursor": "",
                },
            )
        )
        with make_client() as c:
            df = events_frame(c, "KXHIGHNY")
        assert df["event_ticker"][0] == "E1"
        assert df["target_date"][0] == datetime(2026, 8, 10, tzinfo=UTC)

    @respx.mock
    def test_markets_frame_iso_times_and_new_fields(self) -> None:
        mock_cutoff()
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
                            "open_time": "2026-05-30T14:00:00Z",
                            "close_time": "2026-05-31T04:59:00Z",
                            "settlement_ts": int(CUTOFF.timestamp()),
                            "result": "yes",
                            "settlement_value_dollars": "92.00",
                        }
                    ],
                    "cursor": "",
                },
            )
        )
        with make_client() as c:
            df = markets_frame(c, series_ticker="KXHIGHNY", start=CUTOFF - timedelta(days=3),
                               end=CUTOFF)
        assert df["result"][0] == "yes"
        assert df["open_at"][0] == datetime(2026, 5, 30, 14, 0, tzinfo=UTC)
        assert df["settled_at"][0] == CUTOFF
        assert df["settlement_value"][0] == "92.00"

    @respx.mock
    def test_candles_frame_live_dollars_shape(self) -> None:
        """Live candles: end_period_ts + *_dollars strings + volume_fp."""
        mock_cutoff()
        end_ts = int((CUTOFF + timedelta(days=1, minutes=1)).timestamp())
        respx.get(url__startswith=f"{LIVE_BASE}/series/KXHIGHNY/markets/M1/candlesticks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "end_period_ts": end_ts,
                            "yes_bid": {"open_dollars": "0.4000", "high_dollars": "0.4100",
                                        "low_dollars": "0.3900", "close_dollars": "0.4050"},
                            "yes_ask": {"open_dollars": "0.4500", "high_dollars": "0.4600",
                                        "low_dollars": "0.4400", "close_dollars": "0.4550"},
                            "volume_fp": "12.00",
                            "open_interest_fp": "99.00",
                        }
                    ]
                },
            )
        )
        with make_client() as c:
            df = candles_frame(c, "M1", CUTOFF + timedelta(days=1), CUTOFF + timedelta(days=2),
                               series_ticker="KXHIGHNY")
        row = df.row(0, named=True)
        assert row["yes_ask_close"] == 0.455
        # bar bounds are derived from end_period_ts, not read off the payload
        assert row["bar_end_at"] == datetime.fromtimestamp(end_ts, tz=UTC)
        assert row["bar_start_at"] == row["bar_end_at"] - timedelta(seconds=60)
        assert row["ts"] == row["bar_start_at"]
        assert row["volume"] == 12
        assert row["open_interest"] == 99

    @respx.mock
    def test_candles_frame_historical_bare_shape(self) -> None:
        """Historical candles: end_period_ts + bare string OHLC keys."""
        mock_cutoff()
        end_ts = int((CUTOFF - timedelta(days=1, minutes=1)).timestamp())
        respx.get(url__startswith=f"{LIVE_BASE}/historical/markets/M1/candlesticks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "candlesticks": [
                        {
                            "end_period_ts": end_ts,
                            "yes_bid": {"open": "0.0300", "high": "0.0300",
                                        "low": "0.0300", "close": "0.0300"},
                            "yes_ask": {"open": "0.0500", "high": "0.0500",
                                        "low": "0.0400", "close": "0.0400"},
                            "volume": "266.00",
                            "open_interest": "888.00",
                        }
                    ]
                },
            )
        )
        with make_client() as c:
            df = candles_frame(c, "M1", CUTOFF - timedelta(days=2), CUTOFF - timedelta(days=1))
        row = df.row(0, named=True)
        assert row["yes_ask_close"] == 0.04
        assert row["yes_ask_open"] == 0.05
        assert row["volume"] == 266
        assert row["open_interest"] == 888

    @respx.mock
    def test_forecast_percentile_frame_new_endpoint_and_flatten(self) -> None:
        """/series/{s}/events/{e}/forecast_percentile_history with repeated
        percentiles; forecast_history[].percentile_points[] is flattened."""
        ts = int(CUTOFF.timestamp())
        respx.get(
            url__startswith=(
                f"{LIVE_BASE}/series/KXHIGHNY/events/E1/forecast_percentile_history"
            )
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "forecast_history": [
                        {
                            "end_period_ts": ts,
                            "percentile_points": [
                                {"percentile": 50.0, "numerical_forecast": 90.0},
                                {"percentile": 90.0, "numerical_forecast": 92.0},
                            ],
                        },
                        {
                            "end_period_ts": ts + 60,
                            "percentile_points": [
                                {"percentile": 50.0, "numerical_forecast": 91.0},
                            ],
                        },
                    ]
                },
            )
        )
        with make_client() as c:
            df = forecast_percentile_frame(
                c, "E1", "KXHIGHNY",
                start=CUTOFF - timedelta(hours=1), end=CUTOFF,
            )
        assert df.height == 3
        assert df["percentile"].to_list() == [50.0, 90.0, 50.0]
        assert df["numerical_forecast"].to_list() == [90.0, 92.0, 91.0]
        # the request carried repeated percentiles + int period_interval
        url = str(respx.calls[-1].request.url)
        assert "percentiles=10" in url and "percentiles=90" in url
        assert "period_interval=60" in url

    @respx.mock
    def test_forecast_percentile_microdegree_rescale(self) -> None:
        """Live weather values arrive scaled by 1e6 (87.6° -> 87_600_000).
        raw_numerical_forecast (exact) is preferred; both are rescaled."""
        ts = int(CUTOFF.timestamp())
        respx.get(
            url__startswith=(
                f"{LIVE_BASE}/series/KXHIGHNY/events/E1/forecast_percentile_history"
            )
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "forecast_history": [
                        {
                            "end_period_ts": ts,
                            "percentile_points": [
                                {
                                    "percentile": 10.0,
                                    "numerical_forecast": 87_600_000,
                                    "raw_numerical_forecast": 87_578_400,
                                },
                                {
                                    "percentile": 90.0,
                                    "numerical_forecast": 88_200_000,
                                    "raw_numerical_forecast": 88_205_600,
                                },
                            ],
                        }
                    ]
                },
            )
        )
        with make_client() as c:
            df = forecast_percentile_frame(
                c, "E1", "KXHIGHNY",
                start=CUTOFF - timedelta(hours=1), end=CUTOFF,
            )
        assert df["numerical_forecast"].to_list() == [87.5784, 88.2056]

    @respx.mock
    def test_fee_changes_frame_new_path_and_key(self) -> None:
        ts = int(CUTOFF.timestamp())
        respx.get(url__startswith=f"{LIVE_BASE}/series/fee_changes").mock(
            return_value=httpx.Response(
                200,
                json={
                    "series_fee_change_arr": [
                        {"effective_time": ts, "fee_multiplier": "1.0", "fee_type": "quadratic"},
                    ]
                },
            )
        )
        with make_client() as c:
            df = fee_changes_frame(c, "KXHIGHNY")
        assert df["fee_multiplier"][0] == 1.0
        assert df["effective_at"][0] == CUTOFF
