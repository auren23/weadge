"""confirm_no_fills: a fill is confirmed only by a taker-NO print at or
above the assumed bid, inside the [fill_at, fill_at + window) window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from weadge.research.executability import confirm_no_fills

T0 = datetime(2026, 7, 20, 2, 50, 0, tzinfo=UTC)


def _signals() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_ticker": ["A", "B", "C", "D"],
            "fill_at": [T0, T0, T0, T0],
            "fill_bid": [0.60, 0.60, 0.60, 0.60],
            "result": [0, 0, 1, 0],  # extra column must pass through
        }
    )


def _trade(mk: str, dt_s: float, yes: float, taker: str, count: float = 5.0) -> dict:
    return {
        "market_ticker": mk,
        "created_at": T0 + timedelta(seconds=dt_s),
        "yes_price": yes,
        "taker_side": taker,
        "count": count,
    }


def test_confirmation_semantics() -> None:
    trades = pl.DataFrame(
        [
            _trade("A", 10, 0.60, "no"),  # at bid, NO taker -> confirms
            _trade("A", 20, 0.55, "no"),  # below bid -> counted, not confirming
            _trade("B", 10, 0.70, "yes"),  # rich print but YES taker -> no
            _trade("B", 30, 0.58, "no"),  # below bid -> no
            _trade("C", 60, 0.65, "no"),  # exactly window end -> excluded
            _trade("C", -1, 0.65, "no"),  # before fill -> excluded
        ]
    )
    out = confirm_no_fills(_signals(), trades, window_s=60).sort("market_ticker")

    a, b, c, d = out.to_dicts()
    assert a["confirmed"] and a["n_confirm"] == 1 and a["n_prints"] == 2
    assert a["best_sell_yes"] == 0.60
    assert a["traded_count"] == 10.0
    assert not b["confirmed"] and b["n_prints"] == 2 and b["best_sell_yes"] == 0.58
    assert not c["confirmed"] and c["n_prints"] == 0  # both prints outside window
    assert not d["confirmed"] and d["n_prints"] == 0  # no tape at all
    assert out["result"].to_list() == [0, 0, 1, 0]  # passthrough intact


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="fill_bid"):
        confirm_no_fills(pl.DataFrame({"market_ticker": ["A"], "fill_at": [T0]}), pl.DataFrame())
