"""Pure-logic checks for tools/kalshi_lock_probe.py (no network)."""

from __future__ import annotations

import importlib.util
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

spec = importlib.util.spec_from_file_location("probe", "tools/kalshi_lock_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

ET = ZoneInfo("America/New_York")


def _raw(bars, trades=()):
    return {"bars": bars, "trades": trades}


def test_reaction_inequality_direction():
    """NO ask >= .97  <=>  YES bid <= .03 (P0 regression: was >=)."""
    lock = datetime(2025, 6, 1, 14, 51, tzinfo=UTC)
    # bid 0.04 -> NO ask 0.96, not yet reacted; bid 0.02 -> NO ask 0.98, reacted
    bars = [[_ts(lock + timedelta(minutes=1)), 0.04], [_ts(lock + timedelta(minutes=2)), 0.02]]
    r = probe.compute(lock, 85.0, 84.0, 0, _raw(bars))
    assert r["reaction_97_s"] == 120.0  # only the second bar qualifies
    assert r["reaction_99_s"] is None  # bid never <= 0.01
    assert r["no_ask_at_lock"] == 0.96


def test_first_quote_never_same_bar():
    lock = datetime(2025, 6, 1, 14, 51, tzinfo=UTC)
    bars = [
        [_ts(lock - timedelta(minutes=1)), 0.5],
        [_ts(lock), 0.3],
        [_ts(lock + timedelta(minutes=1)), 0.1],
    ]
    r = probe.compute(lock, 85.0, 84.0, 0, _raw(bars))
    assert r["first_quote_at"] == lock + timedelta(minutes=1)
    assert r["no_ask_at_lock"] == 0.9


def test_info_delay_shifts_effective_lock():
    lock = datetime(2025, 6, 1, 14, 51, tzinfo=UTC)
    bars = [[_ts(lock + timedelta(minutes=1)), 0.02]]
    r = probe.compute(lock, 85.0, 84.0, 2, _raw(bars))
    assert r["lock_at"] == lock + timedelta(minutes=2)
    # bar at lock+1m is now before effective lock+1m -> not usable
    assert r["first_quote_at"] is None
    assert r["no_ask_at_lock"] is None


def test_stale_window_is_effective_5m():
    lock = datetime(2025, 6, 1, 14, 51, tzinfo=UTC)
    trades = [
        [_ts(lock + timedelta(seconds=30)), 0.90, 10.0],  # in window
        [_ts(lock + timedelta(minutes=6)), 0.80, 10.0],  # outside
    ]
    r = probe.compute(lock, 85.0, 84.0, 0, _raw([], trades))
    assert r["stale_trade_count_5m"] == 1
    assert r["stale_trade_volume_5m"] == 10.0
    assert r["worst_stale_trade_price"] == 0.90


def test_locked_floor_semantics():
    assert probe.locked_floor({"floor_strike": None, "cap_strike": 85}, 0.0) == 85.0
    assert (
        probe.locked_floor({"floor_strike": 80, "cap_strike": 85}, 0.0) == 86.0
    )  # must exceed cap
    assert probe.locked_floor({"floor_strike": 85, "cap_strike": None}, 0.0) is None  # never locks
    assert probe.locked_floor({"floor_strike": None, "cap_strike": 85}, 0.5) == 85.5


def test_lock_day_window_is_standard_time():
    """DST: a 00:30 EDT METAR belongs to the PREVIOUS report day.

    On 2025-03-09 (DST switch day) the standard-time window is
    [2025-03-09 05:00, 2025-03-10 05:00) UTC. 2025-03-10 04:30 UTC is
    00:30 EDT on 03-10, which the NWS report still assigns to 03-09.
    """
    metar = [
        (datetime(2025, 3, 10, 4, 30, tzinfo=UTC), 50.0),  # 00:30 EDT 03-10 -> report day 03-09
        (datetime(2025, 3, 10, 13, 0, tzinfo=UTC), 60.0),  # 09:00 EDT 03-10 -> report day 03-10
    ]
    # report day 03-09: the 04:30 UTC METAR is inside the window
    assert probe.first_lock(metar, date(2025, 3, 9), 50.0, ET) is not None
    # report day 03-10: 04:30 UTC is before the window start -> only 13:00 counts
    assert probe.first_lock(metar, date(2025, 3, 10), 50.0, ET) is not None  # 60F at 13:00
    assert probe.first_lock(metar, date(2025, 3, 10), 61.0, ET) is None  # 60 < 61, 04:30 excluded


def _ts(dt: datetime) -> float:
    return (dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()
