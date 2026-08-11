"""Kalshi HTTP client.

Responsibilities (so research code never sees them):
  * live vs historical API routing, driven by the /historical/cutoff endpoint
  * token-based rate limiting (read/write budgets kept separate)
  * retry with exponential backoff on 429 / 5xx / connection resets
  * cursor pagination
  * optional authenticated endpoints (headers only — keys live in env, never in code)

The caller only sees logical methods: client.get_candles(...), client.get_markets(...).
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

from weadge.domain.time import from_timestamp, to_timestamp

logger = logging.getLogger("weadge.kalshi")

LIVE_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HISTORICAL_PREFIX = "/historical"


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
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key and self.api_secret:
            from weadge.adapters.kalshi.auth import sign_headers

            headers.update(sign_headers(self.api_key, self.api_secret))
        return headers

    # ------------------------------------------------------------- cutoff
    def historical_cutoff(self, force: bool = False) -> datetime:
        """Timestamp separating live from historical storage.

        Data at or before the cutoff lives in the /historical API; data after
        it lives in the live API. Cached for TTL seconds.
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
        raw = body.get("cutoff")
        self._cutoff = from_timestamp(int(raw)) if raw is not None else datetime.now(UTC)
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
        _request_paged() instead.
        """
        retries = retries if retries is not None else self.limiter.max_retries
        for attempt in range(retries + 1):
            self.limiter.wait(budget)
            headers = self._headers() if auth else {"Accept": "application/json"}
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
        page_size: int = 200,
        max_pages: int = 10_000,
    ) -> list[dict[str, Any]]:
        """GET a cursor-paginated list endpoint, following `cursor` until empty."""
        params = dict(params or {})
        params.setdefault("page_size", page_size)
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
        path = self._route("/events", start, end)
        params: dict[str, Any] = {"series_ticker": series_ticker}
        if start is not None:
            params["min_close_ts"] = to_timestamp(start)
        if end is not None:
            params["max_close_ts"] = to_timestamp(end)
        if status is not None:
            params["status"] = status
        return self._request_paged("GET", path, params, list_key="events")

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        path = self._route("/markets", start, end)
        params: dict[str, Any] = {}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if start is not None:
            params["min_close_ts"] = to_timestamp(start)
        if end is not None:
            params["max_close_ts"] = to_timestamp(end)
        if status is not None:
            params["status"] = status
        return self._request_paged("GET", path, params, list_key="markets")

    def get_market_candles(
        self,
        market_ticker: str,
        start: datetime,
        end: datetime,
        period_interval: str = "1m",
    ) -> list[dict[str, Any]]:
        """1-minute YES bid/ask OHLC candles for one market."""
        path = self._route(f"/markets/{market_ticker}/candlesticks", start, end)
        params = {
            "start_ts": to_timestamp(start),
            "end_ts": to_timestamp(end),
            "period_interval": period_interval,
        }
        body = self._request("GET", path, params=params, budget="read")
        return body.get("candlesticks", [])

    def get_event_forecast_percentile_history(
        self,
        event_ticker: str,
        series_ticker: str,
        interval: str = "1m",
    ) -> list[dict[str, Any]]:
        """Kalshi's own forecast percentile history for an event."""
        path = self._route(f"/events/{event_ticker}/forecast_percentile_history", None, None)
        params = {"series_ticker": series_ticker, "interval": interval}
        body = self._request("GET", path, params=params, budget="read")
        return body.get("forecast_percentile_history", [])

    def get_series_fee_changes(
        self,
        series_ticker: str,
        show_historical: bool = True,
    ) -> list[dict[str, Any]]:
        path = self._route(f"/series/{series_ticker}/fee_changes", None, None)
        params = {"show_historical": "true" if show_historical else "false"}
        body = self._request("GET", path, params=params, budget="read")
        return body.get("fee_changes", [])

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
