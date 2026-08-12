"""Kalshi HTTP client.

Responsibilities (so research code never sees them):
  * live vs historical API routing, driven by the /historical/cutoff endpoint
  * token-based rate limiting (read/write budgets kept separate)
  * retry with exponential backoff on 429 / 5xx / connection resets
  * cursor pagination (limit + cursor)
  * optional authenticated endpoints (headers only — keys live in env, never in code)

The caller only sees logical methods: client.get_candles(...), client.get_markets(...).

API contract notes (verified 2026-08-11 against the live API):
  * cutoff response carries ISO timestamps: market_settled_ts /
    trades_created_ts / orders_updated_ts — there is no "cutoff" key.
  * pagination is `limit` + `cursor` (events max 200, markets max 1000).
  * live candlesticks:  /series/{series}/markets/{ticker}/candlesticks
    historical:         /historical/markets/{ticker}/candlesticks
    period_interval is an INTEGER number of minutes (1/60/1440), not
    seconds — the 2026-08-11 "60 = 1m" note was WRONG (60 = 60-minute
    bars); fixed 2026-08-12 after empirical verification.
  * forecast percentile history:
    /series/{series}/events/{event}/forecast_percentile_history
    with repeated `percentiles` params + start_ts/end_ts/period_interval.
  * historical markets accept NO time-range filters — windows are applied
    client-side after fetching the series' markets. There is no historical
    events endpoint at all: live /events serves full history and ignores
    time filters, so event windows are applied client-side on strike_date.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from weadge.adapters.kalshi.auth import API_PATH_PREFIX, sign_headers
from weadge.domain.time import from_timestamp, to_timestamp

logger = logging.getLogger("weadge.kalshi")

# Official recommended environment (2026). api.elections.kalshi.com remains
# supported but is the legacy address.
LIVE_BASE = "https://external-api.kalshi.com/trade-api/v2"
LEGACY_LIVE_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HISTORICAL_PREFIX = "/historical"

# Per-endpoint pagination caps (official).
LIMIT_EVENTS = 200
LIMIT_MARKETS = 1000

# Forecast percentile history: 1m granularity, standard weather percentiles.
FORECAST_PERCENTILES = (10, 25, 50, 75, 90)
FORECAST_PERIOD_INTERVAL_MIN = 1


class KalshiError(RuntimeError):
    pass


class RateLimitedError(KalshiError):
    pass


@dataclass
class RateLimiter:
    """Token-bucket-ish limiter per budget (read/write), plus min-request spacing.

    Kalshi issues token-based limits with separate read/write budgets and no
    Retry-After header; the only safe client behavior is backoff + spacing.
    """

    min_interval_s: float = 0.05
    backoff_base_s: float = 0.5
    backoff_factor: float = 2.0
    max_retries: int = 6
    max_backoff_s: float = 60.0
    _last_call: dict[str, float] = field(default_factory=dict)

    def wait(self, budget: str) -> None:
        now = time.monotonic()
        last = self._last_call.get(budget, 0.0)
        delay = self.min_interval_s - (now - last)
        if delay > 0:
            time.sleep(delay)
        self._last_call[budget] = time.monotonic()

    def sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_base_s * (self.backoff_factor**attempt), self.max_backoff_s)
        delay *= 0.5 + random.random()  # jitter
        logger.debug("backing off %.2fs after attempt %d", delay, attempt)
        time.sleep(delay)


def parse_api_ts(v: Any) -> datetime:
    """Parse a Kalshi timestamp that may be epoch seconds (int) or an ISO
    string (e.g. "2026-06-12T00:00:00Z" or "2026-08-11T04:59:00Z")."""
    if isinstance(v, (int, float)):
        return from_timestamp(int(v))
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(UTC)
    raise TypeError(f"cannot parse Kalshi timestamp {v!r}")


class KalshiClient:
    """Typed wrapper over the Kalshi trade-api v2 endpoints.

    Args:
        base_url: live API base. Historical endpoints are derived by prefixing.
        api_key / api_secret: optional, only needed for authenticated calls
            (websockets, orders). Public market-data endpoints need no auth.
        min_interval_s: minimum spacing between requests per budget.
    """

    def __init__(
        self,
        base_url: str = LIVE_BASE,
        api_key: str | None = None,
        api_secret: str | None = None,
        min_interval_s: float = 0.05,
        timeout_s: float = 30.0,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.limiter = limiter or RateLimiter(min_interval_s=min_interval_s)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=True,
        )
        self._cutoff: datetime | None = None  # cached live/historical boundary
        self._cutoff_fetched: float | None = None
        self._cutoff_ttl_s = 300.0

    # ------------------------------------------------------------------ auth
    def _headers(self, method: str, path: str) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key and self.api_secret:
            headers.update(
                sign_headers(self.api_key, self.api_secret, method=method, path=path)
            )
        return headers

    # ------------------------------------------------------------- cutoff
    def historical_cutoff(self, force: bool = False) -> datetime:
        """Timestamp separating live from historical storage.

        Market data at or before the cutoff lives in the /historical API;
        data after it lives in the live API. The official response has NO
        "cutoff" key — markets/candles use `market_settled_ts`
        (fallbacks: market_positions_last_updated_ts, trades_created_ts).
        Cached for TTL seconds.
        """
        now = time.monotonic()
        if (
            not force
            and self._cutoff is not None
            and self._cutoff_fetched is not None
            and now - self._cutoff_fetched < self._cutoff_ttl_s
        ):
            return self._cutoff
        body = self._request("GET", "/historical/cutoff", budget="read", auth=False)
        raw = (
            body.get("market_settled_ts")
            or body.get("market_positions_last_updated_ts")
            or body.get("trades_created_ts")
            or body.get("cutoff")  # legacy shape, tolerated
        )
        self._cutoff = parse_api_ts(raw) if raw is not None else datetime.now(UTC)
        self._cutoff_fetched = time.monotonic()
        return self._cutoff

    def _route(self, logical_path: str, start: datetime | None, end: datetime | None) -> str:
        """Map a logical path (e.g. "/markets") to live or historical.

        Requests without a time window (series metadata, fee changes, forecast
        history) always use the live API and never touch the cutoff endpoint.
        """
        if logical_path.startswith(HISTORICAL_PREFIX):
            return logical_path
        if start is None and end is None:
            return logical_path
        cutoff = self.historical_cutoff()
        if end is not None and end <= cutoff:
            return HISTORICAL_PREFIX + logical_path
        if start is not None and start > cutoff:
            return logical_path
        if start is not None and end is not None and start <= cutoff < end:
            # window straddles the boundary — caller must split; raise to force explicit handling
            raise KalshiError(
                f"request window [{start}, {end}] straddles historical cutoff {cutoff}; "
                "split the window or use get_*_windowed()"
            )
        return logical_path

    # ------------------------------------------------------------ request
    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        budget: str = "read",
        auth: bool = True,
        retries: int | None = None,
    ) -> dict[str, Any]:
        """Perform one logical request: rate-limit, retry, backoff, paginate-ready.

        Returns the JSON body as dict. Callers who need pagination use
        _request_paged() instead. The signature covers the FULL path as
        sent (including the /trade-api/v2 prefix, excluding the query).
        """
        retries = retries if retries is not None else self.limiter.max_retries
        signed_path = f"{API_PATH_PREFIX}{path}"
        for attempt in range(retries + 1):
            self.limiter.wait(budget)
            headers = self._headers(method, signed_path) if auth else {"Accept": "application/json"}
            try:
                resp = self._client.request(method, path, params=params, headers=headers)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                if attempt < retries:
                    self.limiter.sleep_backoff(attempt)
                    continue
                raise KalshiError(f"connection failed after {retries} retries: {path}") from exc

            if resp.status_code == 429:
                if attempt < retries:
                    self.limiter.sleep_backoff(attempt)
                    continue
                raise RateLimitedError(f"429 on {path}")
            if resp.status_code >= 500:
                if attempt < retries:
                    self.limiter.sleep_backoff(attempt)
                    continue
                raise KalshiError(f"{resp.status_code} on {path}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise KalshiError(f"{resp.status_code} on {path}: {resp.text[:300]}")
            return resp.json()
        raise KalshiError(f"request failed after {retries} retries: {path}")  # pragma: no cover

    def _request_paged(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        list_key: str,
        budget: str = "read",
        auth: bool = True,
        limit: int | None = None,
        max_pages: int = 10_000,
    ) -> list[dict[str, Any]]:
        """GET a cursor-paginated list endpoint (`limit` + `cursor`)."""
        params = dict(params or {})
        if limit is not None:
            params.setdefault("limit", limit)
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            body = self._request(method, path, params=params, budget=budget, auth=auth)
            rows.extend(body.get(list_key, []))
            cursor = body.get("cursor")
            if not cursor:
                break
        else:  # pragma: no cover
            raise KalshiError(f"pagination exceeded {max_pages} pages on {path}")
        return rows

    # ------------------------------------------------------- logical methods
    def get_series(self, series_ticker: str) -> dict[str, Any]:
        """Series metadata (settlement_source, fee_type, fee_multiplier...)."""
        path = self._route(f"/series/{series_ticker}", None, None)
        body = self._request("GET", path, budget="read")
        return body["series"]

    def get_events(
        self,
        series_ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Events for a series, optionally restricted to [start, end].

        There is NO /historical/events endpoint — the live /events endpoint
        serves the series' full history, and (verified 2026-08-11) it
        IGNORES min_close_ts/max_close_ts. Windows are therefore applied
        client-side on each event's strike_date.
        """
        path = "/events"
        params: dict[str, Any] = {"series_ticker": series_ticker}
        if status is not None:
            params["status"] = status
        rows = self._request_paged("GET", path, params, list_key="events", limit=LIMIT_EVENTS)
        if start is not None or end is not None:
            start_u = start.astimezone(UTC) if start is not None else None
            end_u = end.astimezone(UTC) if end is not None else None
            kept: list[dict[str, Any]] = []
            for r in rows:
                sd = r.get("strike_date")
                if not sd:
                    continue
                try:
                    t = parse_api_ts(sd)
                except (TypeError, ValueError):
                    continue
                if start_u is not None and t < start_u:
                    continue
                if end_u is not None and t > end_u:
                    continue
                kept.append(r)
            rows = kept
        return rows

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Markets matching the filter, optionally restricted to the close
        window [start, end].

        Historical markets accept NO time-range filters, so when the request
        routes to /historical/markets the window is applied client-side on
        each market's close_time (the series' full market history is paged).
        """
        path = self._route("/markets", start, end)
        params: dict[str, Any] = {}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status is not None:
            params["status"] = status
        historical = path.startswith(HISTORICAL_PREFIX)
        if not historical:
            if start is not None:
                params["min_close_ts"] = to_timestamp(start)
            if end is not None:
                params["max_close_ts"] = to_timestamp(end)
        rows = self._request_paged(
            "GET", path, params, list_key="markets", limit=LIMIT_MARKETS
        )
        if historical and (start is not None or end is not None):
            start_u = start.astimezone(UTC) if start is not None else None
            end_u = end.astimezone(UTC) if end is not None else None
            rows = [
                r
                for r in rows
                if r.get("close_time") is not None
                and (start_u is None or parse_api_ts(r["close_time"]) >= start_u)
                and (end_u is None or parse_api_ts(r["close_time"]) <= end_u)
            ]
        return rows

    def get_market_candles(
        self,
        market_ticker: str,
        series_ticker: str | None,
        start: datetime,
        end: datetime,
        period_interval_min: int = 1,
    ) -> list[dict[str, Any]]:
        """YES bid/ask OHLC candles for one market.

        Live:  /series/{series}/markets/{ticker}/candlesticks
        Historical: /historical/markets/{ticker}/candlesticks
        `period_interval` is an integer number of MINUTES (official valid
        values 1/60/1440) — NOT seconds. Verified 2026-08-12: pi=1 over
        one hour returns per-minute bars, pi=60 returns hourly bars.
        """
        start = start.astimezone(UTC)
        end = end.astimezone(UTC)
        if end <= self.historical_cutoff():
            path = f"{HISTORICAL_PREFIX}/markets/{market_ticker}/candlesticks"
        else:
            if series_ticker is None:
                raise KalshiError(
                    "live candlesticks need the series ticker "
                    "(path is /series/{series}/markets/{ticker}/candlesticks)"
                )
            path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {
            "start_ts": to_timestamp(start),
            "end_ts": to_timestamp(end),
            "period_interval": period_interval_min,
        }
        body = self._request("GET", path, params=params, budget="read")
        return body.get("candlesticks", [])

    def get_event_forecast_percentile_history(
        self,
        event_ticker: str,
        series_ticker: str,
        start: datetime,
        end: datetime,
        percentiles: tuple[int, ...] = FORECAST_PERCENTILES,
        period_interval_min: int = FORECAST_PERIOD_INTERVAL_MIN,
    ) -> list[dict[str, Any]]:
        """Kalshi's own forecast percentile history for one event.

        Endpoint: /series/{series}/events/{event}/forecast_percentile_history
        with repeated `percentiles` params and an integer period_interval in
        MINUTES (verified 2026-08-12). Returns `forecast_history[]` —
        flattening happens in the adapter layer.
        """
        path = f"/series/{series_ticker}/events/{event_ticker}/forecast_percentile_history"
        params: dict[str, Any] = {
            "start_ts": to_timestamp(start),
            "end_ts": to_timestamp(end),
            "period_interval": period_interval_min,
        }
        for p in percentiles:
            params.setdefault("percentiles", []).append(str(p))
        body = self._request("GET", path, params=params, budget="read")
        return body.get("forecast_history", [])

    def get_series_fee_changes(
        self,
        series_ticker: str,
        show_historical: bool = True,
    ) -> list[dict[str, Any]]:
        path = "/series/fee_changes"
        params = {
            "series_ticker": series_ticker,
            "show_historical": "true" if show_historical else "false",
        }
        body = self._request("GET", path, params=params, budget="read")
        return body.get("series_fee_change_arr", [])

    # ------------------------------------------------------------------ misc
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Re-exported for convenience (typed adapters live in sibling modules).
def with_backoff(fn: Callable[..., Any], attempts: int = 3) -> Callable[..., Any]:
    """Small decorator for one-off calls that should tolerate transient errors."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last: Exception | None = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except (RateLimitedError, httpx.TransportError) as exc:
                last = exc
                time.sleep(0.5 * (2**i))
        raise last  # type: ignore[misc]
    return wrapper
