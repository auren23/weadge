"""Average-settlement fair value for KXBTC15M — calibration + edge test.

Context. The leak-free re-run of the H2 study (crypto_executability) killed
the latency candidate: net negative in all phases. The next honest
hypothesis is a MODELING edge, not a speed edge: KXBTC15M settles on the
CFB RTI — a ~60-second average — while the natural mental model (and the
scratch study's p*) prices the ENDPOINT. For a Brownian endpoint the final
minute contributes variance sig^2 * 1; for the final-minute AVERAGE it
contributes sig^2 / 3. Correct fair value at tau >= 1 minute:

    p_avg = Phi( ln(S/K) / (sig * sqrt(tau - 2/3)) )      [minutes]

vs the endpoint model p_end = Phi( ln(S/K) / (sig * sqrt(tau)) ). The gap
concentrates exactly where the book turns over most (last few minutes), so
if the market prices endpoints, an average-aware model should show a
calibration and edge advantage there.

Also recorded as a first-class negative result (checked 2026-08-12): all
22,815 KXBTC15M events carry exactly ONE strike, so there is no intra-event
ladder and no structural-arb direction on this series.

Discipline inherited from crypto_executability: spot bars are labeled by
OPEN time and only closes with label + 60 <= t are usable
(last_knowable_spot_idx); fills are next-bar opens and the fill bar must
start exactly at the signal bar's end; fees replay the series multiplier.
tau < 1 minute is excluded — modeling a partially realized average needs
sub-second data we don't have yet.

    uv run python -m weadge.research.crypto_settlement_model
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from weadge.research.crypto_executability import (
    MIN_SIGMA_OBS,
    PHASES,
    SERIES,
    SIGMA_WINDOW_S,
    _epoch_s,
    fee_multiplier,
    fee_quad,
    last_knowable_spot_idx,
)

TAU_BINS = ((1, 2), (2, 5), (5, 10), (10, 15))
EDGE_MIN = 0.03


def p_fair(s: float, k: float, sig: float, tau_min: float, *, settle_avg: bool) -> float:
    """Lognormal fair value of YES (settle >= K) at tau_min minutes out.

    settle_avg=True prices settlement as the final-minute average
    (variance tau - 2/3 in minute units); False prices the endpoint
    (variance tau). Requires tau_min >= 1."""
    var = (tau_min - 2.0 / 3.0) if settle_avg else tau_min
    return float(norm.cdf(math.log(s / k) / (sig * math.sqrt(var))))


def build_observations(
    markets: pl.DataFrame, quotes: pl.DataFrame, spot: pl.DataFrame
) -> pl.DataFrame:
    """One row per (market, completed bar) with both models, leak-free."""
    st = spot.sort("ts")
    s_ts = _epoch_s(st["ts"])
    s_cl = st["close"].to_numpy()

    rows: list[dict] = []
    quotes = quotes.filter(pl.col("market_ticker").str.starts_with(SERIES))
    by_mk = {
        (k[0] if isinstance(k, tuple) else k): g.sort("bar_end_at")
        for k, g in quotes.group_by("market_ticker")
    }
    for r in markets.iter_rows(named=True):
        g = by_mk.get(r["market_ticker"])
        strike = r["floor_strike"]
        if g is None or strike is None:
            continue
        close_s = _epoch_s(pl.Series([r["close_at"]]))[0]
        result = 1 if r["result"] == "yes" else 0
        ts = _epoch_s(g["bar_end_at"])
        starts = _epoch_s(g["bar_start_at"])
        bid_close = g["yes_bid_close"].to_numpy()
        ask_close = g["yes_ask_close"].to_numpy()
        bid_open = g["yes_bid_open"].to_numpy()
        ask_open = g["yes_ask_open"].to_numpy()
        for i in range(len(ts) - 1):
            t = int(ts[i])
            if starts[i + 1] != t:  # fill must be the immediately next minute
                continue
            tau_min = (close_s - t) / 60.0
            if tau_min < 1.0:
                continue
            si = last_knowable_spot_idx(s_ts, t)
            s0 = int(np.searchsorted(s_ts, t - SIGMA_WINDOW_S, side="left"))
            if si < 0 or si - s0 < MIN_SIGMA_OBS:
                continue
            sig = float(np.diff(np.log(s_cl[s0 : si + 1])).std())
            if sig <= 0:
                continue
            s_now = float(s_cl[si])
            rows.append(
                {
                    "market_ticker": r["market_ticker"],
                    "t": t,
                    "tau": tau_min,
                    "p_end": p_fair(s_now, strike, sig, tau_min, settle_avg=False),
                    "p_avg": p_fair(s_now, strike, sig, tau_min, settle_avg=True),
                    "bid": float(bid_close[i]) if bid_close[i] is not None else None,
                    "ask": float(ask_close[i]) if ask_close[i] is not None else None,
                    "fill_bid": float(bid_open[i + 1]) if bid_open[i + 1] is not None else None,
                    "fill_ask": float(ask_open[i + 1]) if ask_open[i + 1] is not None else None,
                    "result": result,
                }
            )
    out = pl.DataFrame(rows)
    if out.is_empty():
        return out
    phase_col = pl.col("t")
    expr = pl.lit(PHASES[-1][1])
    for bound, name in reversed(PHASES[:-1]):
        expr = pl.when(phase_col < bound).then(pl.lit(name)).otherwise(expr)
    return out.with_columns(expr.alias("phase"))


def _calibration(obs: pl.DataFrame) -> None:
    print(f"\ncalibration (leak-free): decile -> realized YES rate  n={obs.height}")
    print(f"{'decile':<8} {'p_end':>7} {'p_avg':>7} {'realized':>9} {'n':>8}")
    for d in range(10):
        lo, hi = d / 10, (d + 1) / 10
        g = obs.filter((pl.col("p_avg") >= lo) & (pl.col("p_avg") < hi))
        if g.height < 200:
            continue
        print(
            f"[{lo:.1f},{hi:.1f}) {g['p_end'].mean():>7.3f} {g['p_avg'].mean():>7.3f} "
            f"{g['result'].mean():>9.3f} {g.height:>8}"
        )
    for name in ("p_end", "p_avg"):
        brier = ((obs[name] - obs["result"]) ** 2).mean()
        print(f"brier {name}: {brier:.5f}")


def _edge_report(obs: pl.DataFrame, multiplier: float) -> None:
    """Taker edge test on the average model, tau-stratified, per phase."""
    print(
        f"\nedge >= {EDGE_MIN:.0%} on p_avg, next-adjacent-bar fills, fee x{multiplier}"
        f"\n{'phase':<10} {'tau':>6} {'side':<4} {'n':>6} {'net/ct':>8}"
    )
    for ph in ("DISCOVERY", "OOS1", "OOS2"):
        for lo, hi in TAU_BINS:
            base = obs.filter(
                (pl.col("phase") == ph) & (pl.col("tau") >= lo) & (pl.col("tau") < hi)
            )
            for side in ("YES", "NO"):
                if side == "YES":
                    g = base.filter(
                        ((pl.col("p_avg") - pl.col("ask")) >= EDGE_MIN)
                        & pl.col("fill_ask").is_not_null()
                    )
                    if g.is_empty():
                        continue
                    fill = g["fill_ask"].to_numpy()
                    win = g["result"].to_numpy()
                else:
                    g = base.filter(
                        ((pl.col("bid") - pl.col("p_avg")) >= EDGE_MIN)
                        & pl.col("fill_bid").is_not_null()
                    )
                    if g.is_empty():
                        continue
                    fill = 1.0 - g["fill_bid"].to_numpy()
                    win = 1.0 - g["result"].to_numpy()
                fees = np.array([fee_quad(p, multiplier) for p in fill])
                net = (win - (fill + fees)).mean()
                print(f"{ph:<10} {f'{lo}-{hi}m':>6} {side:<4} {g.height:>6} {net:>+8.4f}")


def main(data_root: Path = Path("data")) -> None:
    markets = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(glob.glob(str(data_root / "bronze/markets/*/*.parquet")))
        ]
    ).filter((pl.col("series_ticker") == SERIES) & pl.col("result").is_not_null())
    quotes = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(glob.glob(str(data_root / "bronze/quote_1m/*/*.parquet")))
        ]
    )
    spot = pl.read_parquet(data_root / "spot/btc_1m.parquet")

    obs = build_observations(markets, quotes, spot)
    print(f"observations: {obs.height} across {obs['market_ticker'].n_unique()} markets")
    _calibration(obs)
    _edge_report(obs, fee_multiplier(data_root))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data"))
