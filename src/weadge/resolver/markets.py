"""PM Daily High market discovery and normalization (gamma API).

Verified facts (2026-08-13):
- event: highest-temperature-in-paris-on-2026-08-13, tag `daily-temperature`
- one binary market (YES/NO) per temperature bucket, negRisk, mutually exclusive
- bucket win range: "be 33C" = [33, 34); "be 32C or below" = (-inf, 33); "or above" = [X, inf)
- Paris resolution station is fixed in the rules: wunderground.com/history/daily/fr/bonneuil-en-france/LFPB
- public market data needs no credentials"""

from __future__ import annotations

from dataclasses import field
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, field_validator

from weadge.domain.time import parse_iso

GAMMA_API = "https://gamma-api.polymarket.com"
DAILY_TEMP_TAG = "daily-temperature"


class Bucket(BaseModel):
    """一个温度桶的二元市场(clob 合约)。"""

    market_id: str
    question: str
    clob_token_ids: list[str] = []
    cap_high: float | None        # 赢区间 [cap_low, cap_high); None = or-above 桶(无上限)
    yes_price: float = 0.0
    no_price: float = 0.0

    @field_validator("yes_price", "no_price")
    @classmethod
    def _pct(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"price out of range: {v}")
        return v


class DailyHighEvent(BaseModel):
    """一天一个城市的全部温度桶市场。"""

    slug: str
    city: str
    target_date: date
    buckets: list[Bucket] = field(default_factory=list)  # type: ignore[assignment]

    @property
    def highest_cap(self) -> float | None:
        caps = [b.cap_high for b in self.buckets if b.cap_high is not None]
        return max(caps) if caps else None


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
    """gamma /events 单事件 JSON -> DailyHighEvent。纯函数。"""
    import json

    slug = payload["slug"]
    target_date = parse_iso(payload["endDate"]).date()
    buckets: list[Bucket] = []
    for m in payload.get("markets") or []:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
        yes_idx = outcomes.index("Yes")
        buckets.append(
            Bucket(
                market_id=str(m["id"]),
                question=m.get("question") or "",
                clob_token_ids=list(m.get("clobTokenIds") or []),
                cap_high=bucket_cap_high(m.get("question") or ""),
                yes_price=float(prices[yes_idx]),
                no_price=float(prices[1 - yes_idx]),
            )
        )
    return DailyHighEvent(slug=slug, city=city, target_date=target_date, buckets=buckets)


class PMClient:
    """gamma API 只读客户端(公共数据, 免凭证)。"""

    def __init__(self, base: str = GAMMA_API, timeout: float = 15.0) -> None:
        self._client = httpx.Client(base_url=base, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_daily_high(self, city: str, limit: int = 200) -> list[DailyHighEvent]:
        """拉取指定城市全部未关闭的 daily-high 事件(按 slug 前缀过滤)。"""
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
