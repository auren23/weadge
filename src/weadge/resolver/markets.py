"""PM Daily High market discovery and normalization (gamma + CLOB API).

Verified facts (2026-08-13):
- event: highest-temperature-in-paris-on-2026-08-13, tag `daily-temperature`
- one binary market (YES/NO) per temperature bucket, negRisk, mutually exclusive
- bucket win range: "be 33C" = [33, 34); "be 32C or below" = (-inf, 33); "or above" = [X, inf)
- Paris resolution station is fixed in the rules: wunderground.com/history/daily/fr/bonneuil-en-france/LFPB
- gamma `clobTokenIds` array order is NOT reliable (issue py-clob-client#276);
  resolve token ids via CLOB `/markets/{condition_id}` tokens node instead
- gamma `outcomePrices` is display/mid pricing, NOT executable:
  Paris 37C bucket showed 0.415/0.585 while the real book was 0.01/0.99.
  executable ask must come from `/book`
- public market data needs no credentials"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel

from weadge.domain.time import from_timestamp, parse_iso

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DAILY_TEMP_TAG = "daily-temperature"


class Bucket(BaseModel):
    """One temperature-bucket binary market (CLOB contract)."""

    market_id: str
    question: str
    condition_id: str = ""
    cap_high: float | None        # win range [cap_low, cap_high); None = or-above (no cap)
    no_token_id: str = ""         # resolved via /markets/{condition_id}; "" until fetched

    @property
    def label(self) -> str:
        return self.question.split(" be ")[1].split(" on ")[0] if " be " in self.question else self.question


class DailyHighEvent(BaseModel):
    """All temperature buckets for one city-day."""

    slug: str
    city: str
    target_date: date
    buckets: list[Bucket] = field(default_factory=list)  # type: ignore[assignment]


@dataclass(frozen=True)
class Book:
    """Executable top-of-book for one token."""

    token_id: str
    best_ask: float | None      # None = no ask (cannot taker-buy)
    best_ask_size: float = 0.0
    ts: Any = None              # book timestamp (ms epoch)

    @property
    def has_ask(self) -> bool:
        return self.best_ask is not None


def bucket_cap_high(question: str) -> float | None:
    """Parse bucket upper bound (C) from the market question.

    "be 33C on August 13"      -> 34.0  (win range [33,34))
    "be 32C or below on ..."   -> 33.0  ((-inf,33))
    "be 38C or above on ..."   -> None  (no cap, never locks)
    """
    q = question.lower()
    if "or above" in q:
        return None
    if "or below" in q:
        # "be 32C or below" -> upper bound 33.0 (bucket wins iff daily max < 33)
        return float(q.split("be")[1].strip().split("°c")[0]) + 1.0
    # "be 33°C" -> [33, 34)
    return float(q.split("be")[1].strip().split("°c")[0]) + 1.0


def parse_event(payload: dict[str, Any], city: str) -> DailyHighEvent:
    """gamma /events single event JSON -> DailyHighEvent. Pure."""

    slug = payload["slug"]
    target_date = parse_iso(payload["endDate"]).date()
    buckets: list[Bucket] = []
    for m in payload.get("markets") or []:
        buckets.append(
            Bucket(
                market_id=str(m["id"]),
                question=m.get("question") or "",
                condition_id=m.get("conditionId") or "",
                cap_high=bucket_cap_high(m.get("question") or ""),
            )
        )
    return DailyHighEvent(slug=slug, city=city, target_date=target_date, buckets=buckets)


class PMClient:
    """gamma API read-only client (public data, no credentials)."""

    def __init__(self, base: str = GAMMA_API, timeout: float = 15.0) -> None:
        self._client = httpx.Client(base_url=base, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_daily_high(self, city: str, limit: int = 200) -> list[DailyHighEvent]:
        """All unclosed daily-high events for a city (slug prefix filter)."""
        resp = self._client.get(
            "/events",
            params={"tag_slug": DAILY_TEMP_TAG, "closed": "false", "limit": limit},
        )
        resp.raise_for_status()
        prefix = f"highest-temperature-in-{city}-on-"
        return [
            parse_event(e, city)
            for e in resp.json()
            if isinstance(e, dict) and e.get("slug", "").startswith(prefix)
        ]


class ClobClient:
    """CLOB read-only client (public book/tokens data, no credentials)."""

    def __init__(self, base: str = CLOB_API, timeout: float = 15.0) -> None:
        self._client = httpx.Client(base_url=base, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ClobClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def resolve_no_token(self, condition_id: str) -> str:
        """NO token id via /markets/{condition_id} tokens node (authoritative)."""
        if not condition_id:
            return ""
        resp = self._client.get(f"/markets/{condition_id}")
        resp.raise_for_status()
        for t in resp.json().get("tokens") or []:
            if t.get("outcome") == "No":
                return t["token_id"]
        return ""

    def fetch_book(self, token_id: str) -> Book:
        """Top of book for a token; best_ask None when no ask is resting."""
        resp = self._client.get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
        asks = data.get("asks") or []
        best = asks[0] if asks else None
        return Book(
            token_id=token_id,
            best_ask=float(best["price"]) if best else None,
            best_ask_size=float(best["size"]) if best else 0.0,
            ts=from_timestamp(int(data["timestamp"])) if data.get("timestamp") else None,
        )
