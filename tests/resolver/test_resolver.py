"""Resolver V0 pure-function tests (offline; fixture from real gamma API sample 2026-08-13)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from weadge.resolver.edge import find_edges, taker_fee
from weadge.resolver.markets import Book, bucket_cap_high, parse_event
from weadge.resolver.observations import ObservedState, evaluate_observed
from weadge.resolver.state import ResolutionState, evaluate_event

FIXTURES = Path(__file__).parent / "fixtures"
PARIS_TZ = ZoneInfo("Europe/Paris")


@pytest.fixture(scope="module")
def paris_event():
    with open(FIXTURES / "paris_event.json") as fh:
        return parse_event(json.load(fh), "paris")


# --------------------------------------------------------------- markets

def test_bucket_cap_high():
    assert bucket_cap_high("Will the highest temperature in Paris be 32°C or below on August 13?") == 33.0
    assert bucket_cap_high("Will the highest temperature in Paris be 33°C on August 13?") == 34.0
    assert bucket_cap_high("Will the highest temperature in Paris be 38°C or above on August 13?") is None


def test_parse_event(paris_event):
    assert paris_event.slug == "highest-temperature-in-paris-on-august-13-2026"
    assert paris_event.target_date == datetime(2026, 8, 13, tzinfo=UTC).date()
    assert len(paris_event.buckets) == 6
    # 首个桶 "32°C or below" -> cap 33.0; 末端 "37°C" -> cap 38.0
    assert paris_event.buckets[0].cap_high == 33.0
    assert paris_event.buckets[-1].cap_high == 38.0
    # token id 不随 gamma 预解析 —— 由 CLOB /markets/{cid} 权威解析 (顺序不可靠, issue #276)
    assert paris_event.buckets[0].no_token_id == ""
    assert paris_event.buckets[0].condition_id.startswith("0x")
    assert paris_event.buckets[0].label == "32°C or below"


# --------------------------------------------------------------- observations

def _metar_row(hour_utc: int, temp: float) -> dict:
    ts = datetime(2026, 8, 13, hour_utc, tzinfo=UTC)
    return {"obsTime": int(ts.timestamp()), "temp": temp, "rawOb": f"METAR LFPB {hour_utc:02d}000Z"}


def _afternoon_obs(rows: list[dict]) -> ObservedState:
    """固定 now=8/13 14:10Z (巴黎 16:10, 日高温时段) 的观测。"""
    now = datetime(2026, 8, 13, 14, 10, tzinfo=UTC)
    return evaluate_observed("LFPB", rows, PARIS_TZ, now=now)


def test_evaluate_observed_local_day():
    # 巴黎 8/13 本地日 = [8/12 22:00Z, 8/13 22:00Z); 边界在 22:00 UTC
    rows = [
        _metar_row(23, 29.0),  # 8/12 23:00Z = 巴黎 8/13 01:00, 属于 8/13
        _metar_row(8, 30.0),   # 巴黎 10:00
        _metar_row(12, 32.0),  # 巴黎 14:00
        _metar_row(16, 33.0),  # 巴黎 18:00 -> 日最大
        _metar_row(20, 32.0),  # 巴黎 22:00, 仍属 8/13
        _metar_row(23, 31.0),  # 未来观测 (now=21:00Z), 排除
    ]
    now = datetime(2026, 8, 13, 20, 10, tzinfo=UTC)
    obs = evaluate_observed("LFPB", rows, PARIS_TZ, now=now)
    assert obs.observed_max_c == 33.0
    assert obs.temp_c == 32.0  # 最新属于当天的观测 (20:00Z)
    assert not obs.stale


def test_evaluate_observed_stale_and_empty():
    rows = [_metar_row(8, 30.0)]
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)  # 4h 后
    obs = evaluate_observed("LFPB", rows, PARIS_TZ, now=now)
    assert obs.stale
    empty = evaluate_observed("LFPB", [], PARIS_TZ, now=now)
    assert empty.observed_max_c is None and empty.stale


# --------------------------------------------------------------- state

def test_locked_cold_side(paris_event):
    # observed_max 34.5 >= 34.0(33°C 桶 cap) + 0.5 -> "33°C" LOCKED
    obs = _afternoon_obs([_metar_row(14, 34.5)])
    states = {bs.bucket.cap_high: bs.state for bs in evaluate_event(paris_event.buckets, obs)}
    assert states[33.0] is ResolutionState.LOCKED      # 32°C or below (cap 33)
    assert states[34.0] is ResolutionState.LOCKED      # 33°C (cap 34)
    assert states[35.0] is ResolutionState.OPEN        # 34°C (cap 35): 34.5 < 35.5
    assert states[38.0] is ResolutionState.OPEN        # 37°C (cap 38)


def test_locked_buffer_respects_settlement_deviation(paris_event):
    # 观测 34.2, buffer 0.5 -> "33°C" (cap 34.0) 未到 34.5, 不锁 —— 防 METAR/Wunderground 偏差
    obs = _afternoon_obs([_metar_row(14, 34.2)])
    states = {bs.bucket.cap_high: bs.state for bs in evaluate_event(paris_event.buckets, obs)}
    assert states[34.0] is ResolutionState.OPEN


# --------------------------------------------------------------- edge

def _book(no_ask: float | None, size: float = 100.0) -> Book:
    return Book(token_id="t", best_ask=no_ask, best_ask_size=size, ts=datetime(2026, 8, 13, 14, 0, tzinfo=UTC))


def test_find_edges(paris_event):
    obs = _afternoon_obs([_metar_row(14, 34.5)])
    bucket_states = evaluate_event(paris_event.buckets, obs)
    locked = [bs for bs in bucket_states if bs.bucket.cap_high in (33.0, 34.0)]
    # "33°C" 桶 (cap 34.0) 可执行 NO@0.94: 理论 1.0, fee≈0.0028, exec 0.01 -> edge≈0.047
    books = {bs.bucket.market_id: _book(0.94) if bs.bucket.cap_high == 34.0 else _book(None) for bs in locked}
    assessments = find_edges(bucket_states, books, min_net_edge=0.02)
    signals = [a for a in assessments if a.signal]
    assert len(signals) == 1
    s = signals[0]
    assert s.bucket.cap_high == 34.0
    assert s.no_ask == 0.94
    assert s.no_ask_size == 100.0
    assert abs(s.fee - taker_fee(0.94)) < 1e-9
    assert abs(s.net_edge - (1.0 - 0.94 - taker_fee(0.94) - 0.01)) < 1e-9
    assert s.net_edge >= 0.02
    # 无 resting ask 的 LOCKED 桶也返回评估 (untradeable) —— kill test 需要看见它
    untradeable = [a for a in assessments if a.no_ask is None]
    assert len(untradeable) == 1 and untradeable[0].signal is False


def test_find_edges_no_signal_when_price_fair(paris_event):
    obs = _afternoon_obs([_metar_row(14, 34.5)])
    bucket_states = evaluate_event(paris_event.buckets, obs)
    locked = [bs for bs in bucket_states if bs.bucket.cap_high in (33.0, 34.0)]
    # 盘口已反应 (NO@0.99): fee≈0.0005, edge≈-0.0005 < 0.02 -> 无 signal
    books = {bs.bucket.market_id: _book(0.99) for bs in locked}
    assessments = find_edges(bucket_states, books, min_net_edge=0.02)
    assert all(not a.signal for a in assessments)
    assert len(assessments) == 2  # 评估仍被产出 (kill test 需要 lock 记录)


def test_taker_fee_extremes():
    assert abs(taker_fee(0.5) - 0.0125) < 1e-9
    assert taker_fee(0.99) < 0.001
    assert taker_fee(0.0) == 0.0


# --------------------------------------------------------------- execution stub

def test_trade_mode_stubbed():
    from weadge.resolver import execution

    with pytest.raises(NotImplementedError):
        execution.place_limit("1", "NO", 0.97, 100)
    with pytest.raises(NotImplementedError):
        execution.cancel("x")
    with pytest.raises(NotImplementedError):
        execution.get_open_orders()
