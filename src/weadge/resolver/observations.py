"""Station observations (NOAA aviationweather METAR, free, no key, global ICAO).

Sample: {"icaoId":"LFPB","obsTime":1786617000,"temp":34}  # temp in C

Observation source (METAR) != settlement source (Wunderground station page);
both read the same airport ASOS, deviation is usually <0.5C, absorbed by
locked_buffer_c in state.py."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from weadge.domain.time import from_timestamp, utc_now

METAR_API = "https://aviationweather.gov/api/data/metar"


@dataclass(frozen=True)
class ObservedState:
    """某结算日到此刻的站观测汇总。"""

    station: str
    ts: datetime                 # 最新观测时间 (UTC)
    temp_c: float | None         # 当前温度
    observed_max_c: float | None # max temp observed on the settlement local day
    stale: bool                  # 最新观测太旧


def _local_day_start(obs_ts: datetime, tz: ZoneInfo) -> datetime:
    local = obs_ts.astimezone(tz)
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.astimezone(UTC)


def evaluate_observed(
    station: str,
    rows: list[dict],
    tz: ZoneInfo,
    now: datetime | None = None,
    stale_after_min: int = 30,
) -> ObservedState:
    """METAR 行列表 -> ObservedState。纯函数。

    rows: aviationweather JSON(含 obsTime 秒、temp degC)。只统计站本地
    自然日 [00:00, 23:59) 内的观测 ---- 与 PM 结算日对齐。
    """
    now = now or utc_now()
    # 只统计不晚于 now 的观测 (API 理论上不会返回未来, 纯函数防御)
    valid = [
        (from_timestamp(r["obsTime"]), float(r["temp"]))
        for r in rows
        if r.get("temp") is not None and from_timestamp(r["obsTime"]) <= now
    ]
    if not valid:
        return ObservedState(station=station, ts=now, temp_c=None, observed_max_c=None, stale=True)

    # 结算日 = 最新观测的站本地日 (扫描发生在结算日当天)
    latest_ts = max(ts for ts, _ in valid)
    day_start = _local_day_start(latest_ts, tz)
    day_rows = [(ts, t) for ts, t in valid if day_start <= ts < day_start + timedelta(days=1)]

    day_latest_ts, day_latest_temp = max(day_rows, key=lambda x: x[0])
    observed_max = max(t for _, t in day_rows)
    stale = (now - day_latest_ts) > timedelta(minutes=stale_after_min)
    return ObservedState(
        station=station,
        ts=day_latest_ts,
        temp_c=day_latest_temp,
        observed_max_c=observed_max,
        stale=stale,
    )


class MetarClient:
    """aviationweather METAR 只读客户端。"""

    def __init__(self, base: str = METAR_API, timeout: float = 15.0) -> None:
        self._client = httpx.Client(base_url=base, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MetarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, icao: str, hours: int = 24) -> list[dict]:
        resp = self._client.get(
            "/",
            params={"ids": icao, "format": "json", "taf": "false", "hours": hours},
        )
        resp.raise_for_status()
        return resp.json()


def tz_for(city_cfg: dict) -> ZoneInfo:
    return ZoneInfo(city_cfg["timezone"])

