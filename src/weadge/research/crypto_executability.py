"""Executability + alignment-corrected verdict for the KXBTC15M H2-NO candidate.

The scratch H1-H4 study (scratch/crypto_h1h4.py) reported an ALIVE
candidate: buy NO when the YES bid sits >= 5c above lognormal fair value.
Two threats had to be tested against it, both handled here:

1. FILL REALISM — candle-quote fills on a book where ~88% of 1m bars never
   trade. Tested via the public trade tape (adapters.kalshi.trades): a fill
   counts only if a real taker bought NO at our price or better inside the
   fill window (research.executability.confirm_no_fills).

2. SPOT LOOK-AHEAD (audit finding, 2026-08-12) — the Binance 1m spot
   parquet labels bars by OPEN time (verified: consecutive closes chain to
   the next open). The scratch study indexed spot with
   searchsorted(s_ts, t) at signal time t, which selects the bar covering
   [t, t+60) — its close is 60 SECONDS OF FUTURE. The reported H1 "1-minute
   cross-venue lag" is the same artifact read backwards: corr 0.54 at "lag
   1" is contemporaneous co-movement under mislabeled bars. Fixed here in
   last_knowable_spot_idx(): a bar's close is usable at t only if
   label + 60 <= t.

Pre-registered verdict (unchanged): the candidate survives only if net
stays positive in tape-CONFIRMED fills under leak-free alignment.

Run on a machine with the crypto lake populated (markets + quote_1m for
KXBTC15M under data/bronze, Binance 1m spot at data/spot/btc_1m.parquet):

    uv run python -m weadge.research.crypto_executability [n_markets]

n_markets = 0 skips the tape entirely and prints the offline phase report
over ALL signals (fast leak-free re-verdict of the H2 backtest).
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from weadge.research.executability import confirm_no_fills

SERIES = "KXBTC15M"
EDGE_MIN = 0.05
SIGMA_WINDOW_S = 3600
MIN_SIGMA_OBS = 20
SPOT_BAR_S = 60  # Binance 1m klines, labeled by bar OPEN time
FEE_RATE = 0.07  # Kalshi quadratic taker fee, scaled by the series multiplier

PHASES = ((1774915200, "DISCOVERY"), (1780272000, "OOS1"), (2**62, "OOS2"))


def fee_quad(p: float, multiplier: float = 1.0) -> float:
    return math.ceil(multiplier * FEE_RATE * p * (1.0 - p) * 100 - 1e-9) / 100


def fee_multiplier(data_root: Path) -> float:
    """Series fee replay (README rule 5). Verified 2026-08-12 against the
    live API: KXBTC15M is fee_type=quadratic, fee_multiplier=1 and
    /series/fee_changes is EMPTY, so replay degenerates to one constant —
    but read it from the lake and fail loudly the day a fee history
    appears, instead of silently backtesting with a stale constant."""
    from weadge.storage.parquet import DataLake

    lake = DataLake(data_root)
    changes = lake.read("fee_changes")
    if not changes.is_empty() and changes.filter(pl.col("series_ticker") == SERIES).height:
        raise NotImplementedError(
            f"{SERIES} now has a fee-change history — implement time-based "
            "fee replay before trusting any PnL from this module"
        )
    series = lake.read("series").filter(pl.col("series_ticker") == SERIES)
    return float(series["fee_multiplier"][0]) if series.height else 1.0


def last_knowable_spot_idx(s_ts: np.ndarray, t: int) -> int:
    """Index of the last spot bar whose CLOSE is knowable at decision time
    t, for bars labeled by OPEN time: label + SPOT_BAR_S <= t. Returns -1
    when no bar qualifies. searchsorted(s_ts, t) without this shift selects
    the bar still in progress at t — that close is future data."""
    return int(np.searchsorted(s_ts, t - SPOT_BAR_S, side="right")) - 1


def _epoch_s(col: pl.Series) -> np.ndarray:
    """Column -> epoch seconds. Handles datetime and bare-int columns of any
    unit — the us-vs-s mismatch is exactly the bug that voided one run of
    the scratch study, so units are normalized in ONE place."""
    if col.dtype.is_temporal():
        a = col.cast(pl.Datetime("us", time_zone="UTC")).cast(pl.Int64).to_numpy()
        return a // 1_000_000
    a = col.to_numpy().astype("int64")
    top = int(a.max())
    for scale in (1_000_000_000, 1_000_000, 1_000):  # ns / us / ms
        if top > 100 * scale * 10**9:
            return a // scale
    return a


def derive_no_signals(
    markets: pl.DataFrame, quotes: pl.DataFrame, spot: pl.DataFrame
) -> pl.DataFrame:
    """H2-NO signal set: signal on completed bar t when
    yes_bid_close - p*(t) >= EDGE_MIN, assumed fill = next bar's
    yes_bid_open. Same semantics as scratch/crypto_h1h4.py EXCEPT the spot
    index, which is leak-free here (see last_knowable_spot_idx)."""
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
        bid_close = g["yes_bid_close"].to_numpy()
        bid_open = g["yes_bid_open"].to_numpy()
        fill_at = g["bar_start_at"].to_list()
        for i in range(len(ts) - 1):
            bid = bid_close[i]
            fill_bid = bid_open[i + 1]
            if bid is None or fill_bid is None or np.isnan(bid) or np.isnan(fill_bid):
                continue
            t = int(ts[i])
            si = last_knowable_spot_idx(s_ts, t)
            s0 = int(np.searchsorted(s_ts, t - SIGMA_WINDOW_S, side="left"))
            if si < 0 or si - s0 < MIN_SIGMA_OBS:
                continue
            sig = float(np.diff(np.log(s_cl[s0 : si + 1])).std())
            tau_min = max((close_s - t) / 60.0, 0.1)
            if sig <= 0:
                continue
            p_star = float(norm.cdf(math.log(s_cl[si] / strike) / (sig * math.sqrt(tau_min))))
            if bid - p_star < EDGE_MIN:
                continue
            rows.append(
                {
                    "market_ticker": r["market_ticker"],
                    "t": t,
                    "fill_at": fill_at[i + 1],
                    "fill_bid": float(fill_bid),
                    "result": result,
                }
            )
    out = pl.DataFrame(rows)
    if out.is_empty():
        return out
    phase = pl.col("t")
    expr = pl.lit(PHASES[-1][1])
    for bound, name in reversed(PHASES[:-1]):
        expr = pl.when(phase < bound).then(pl.lit(name)).otherwise(expr)
    return out.with_columns(expr.alias("phase"))


def _sample_tickers(signals: pl.DataFrame, n_markets: int) -> list[str]:
    """Deterministic spread across the whole period: sort by first signal
    time, take every k-th ticker (no RNG, rerun-stable)."""
    order = (
        signals.group_by("market_ticker")
        .agg(pl.col("t").min())
        .sort(["t", "market_ticker"])["market_ticker"]
        .to_list()
    )
    if len(order) <= n_markets:
        return order
    step = len(order) / n_markets
    return [order[int(i * step)] for i in range(n_markets)]


def _fetch_trades(tickers: list[str], markets: pl.DataFrame, data_root: Path) -> pl.DataFrame:
    from weadge.adapters.kalshi.client import KalshiClient
    from weadge.adapters.kalshi.trades import trades_frame
    from weadge.storage.parquet import DataLake

    lake = DataLake(data_root)
    close_by_mk = dict(zip(markets["market_ticker"], markets["close_at"], strict=True))
    with KalshiClient() as client:
        for i, ticker in enumerate(tickers):
            part = data_root / "bronze" / "trades" / f"market_ticker={ticker}"
            if part.exists():
                continue
            frame = trades_frame(client, ticker, close_at=close_by_mk.get(ticker))
            if frame.is_empty():
                part.mkdir(parents=True, exist_ok=True)  # cache the emptiness too
            else:
                lake.write_parquet("trades", frame, partition_by="market_ticker")
            if (i + 1) % 25 == 0:
                print(f"  trades fetched for {i + 1}/{len(tickers)} markets")
    trades = lake.read("trades")
    return trades.filter(pl.col("market_ticker").is_in(tickers))


def _with_net(df: pl.DataFrame, multiplier: float) -> pl.DataFrame:
    fees = np.array([fee_quad(1.0 - b, multiplier) for b in df["fill_bid"]])
    return df.with_columns(
        ((1 - pl.col("result")) - ((1 - pl.col("fill_bid")) + pl.Series(fees))).alias("net")
    )


def _report_offline(signals: pl.DataFrame, multiplier: float) -> None:
    """Leak-free H2-NO phase report over all signals, no tape needed."""
    signals = _with_net(signals, multiplier)
    print(f"\n{'phase':<12} {'n':>7} {'net/ct':>8}")
    for ph in ("DISCOVERY", "OOS1", "OOS2"):
        g = signals.filter(pl.col("phase") == ph)
        if g.is_empty():
            print(f"{ph:<12} {0:>7}")
            continue
        print(f"{ph:<12} {g.height:>7} {g['net'].mean():>+8.4f}")
    print(f"{'ALL':<12} {signals.height:>7} {signals['net'].mean():>+8.4f}")


def _report(checked: pl.DataFrame, multiplier: float) -> None:
    checked = _with_net(checked, multiplier)
    print(f"\n{'group':<22} {'n':>7} {'net/ct':>8} {'traded_ct(sum)':>14}")
    groups: list[tuple[str, pl.DataFrame]] = [("ALL sampled", checked)]
    groups += [(f"confirmed={v}", checked.filter(pl.col("confirmed") == v)) for v in (True, False)]
    groups += [
        (f"{ph} confirmed", checked.filter((pl.col("phase") == ph) & pl.col("confirmed")))
        for ph in ("DISCOVERY", "OOS1", "OOS2")
    ]
    for label, g in groups:
        if g.is_empty():
            print(f"{label:<22} {0:>7}")
            continue
        print(
            f"{label:<22} {g.height:>7} {g['net'].mean():>+8.4f} {g['traded_count'].sum():>14.0f}"
        )
    rate = checked["confirmed"].mean()
    print(f"\nconfirm rate: {rate:.1%}  (window 60s, taker-NO print at >= assumed bid)")
    print("verdict: edge must hold in `confirmed=true` — unconfirmed-only edge is a quote artifact")


def main(n_markets: int = 400, data_root: Path = Path("data")) -> None:
    markets = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(glob.glob(str(data_root / "bronze/markets/*/*.parquet")))
        ]
    )
    markets = markets.filter((pl.col("series_ticker") == SERIES) & pl.col("result").is_not_null())
    quotes = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(glob.glob(str(data_root / "bronze/quote_1m/*/*.parquet")))
        ]
    )
    spot = pl.read_parquet(data_root / "spot/btc_1m.parquet")

    signals = derive_no_signals(markets, quotes, spot)
    multiplier = fee_multiplier(data_root)
    print(
        f"signals (NO edge >= {EDGE_MIN:.0%}, leak-free spot): {signals.height} across "
        f"{signals['market_ticker'].n_unique()} markets  (fee multiplier {multiplier})"
    )
    if n_markets == 0:
        _report_offline(signals, multiplier)
        return
    tickers = _sample_tickers(signals, n_markets)
    sampled = signals.filter(pl.col("market_ticker").is_in(tickers))
    print(f"sampled {len(tickers)} markets -> {sampled.height} signals")

    trades = _fetch_trades(tickers, markets, data_root)
    print(f"tape: {trades.height} prints")
    _report(confirm_no_fills(sampled, trades), multiplier)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
