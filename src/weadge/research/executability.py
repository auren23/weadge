"""Fill-realism check: cross-examine assumed backtest fills against the tape.

A taker backtest on candle data "fills" at the next bar's quote open and
thereby assumes the recorded quote was hittable. On thin books that
assumption is the main way a stale-quote artifact turns into fake alpha:
conditioning on "bid rich vs fair value" preferentially selects bids that
were about to be cancelled, not hit (only ~12% of KXBTC15M 1m bars show
any volume at all).

confirm_no_fills() takes assumed NO-side fills (sell into the YES bid) and
the public trade prints, and marks a fill CONFIRMED only if, inside the
fill window, some real taker also bought NO at a YES price at or above our
assumed bid — i.e. an actual transaction happened at our price or better.

Pre-registered read of the output:
  * edge that survives only in UNCONFIRMED fills is a quote artifact;
  * edge in CONFIRMED fills, sized by `traded_count`, is executable alpha
    (upper bound — we still assume we could replace that taker).
"""

from __future__ import annotations

import polars as pl

REQUIRED_SIGNAL_COLS = ("market_ticker", "fill_at", "fill_bid")


def confirm_no_fills(
    signals: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    window_s: int = 60,
    price_tol: float = 1e-9,
) -> pl.DataFrame:
    """Classify each assumed NO fill as tape-confirmed or not.

    Args:
        signals: one row per assumed fill, with at least
            market_ticker, fill_at (UTC datetime, start of the fill bar)
            and fill_bid (the YES bid open the backtest assumes to hit).
            Extra columns pass through untouched.
        trades: canonical trades frame (see TRADE_SCHEMA) — needs
            market_ticker, created_at, yes_price, taker_side, count.
        window_s: fill window length starting at fill_at.
        price_tol: float tolerance on the price comparison.

    Returns:
        signals plus n_prints / traded_count (all prints in the window),
        n_confirm / best_sell_yes (taker-NO prints; best_sell_yes is the
        highest YES price any NO-buyer paid through) and confirmed.
    """
    missing = [c for c in REQUIRED_SIGNAL_COLS if c not in signals.columns]
    if missing:
        raise ValueError(f"signals frame missing columns {missing}")

    sig = signals.with_row_index("_sig_id")
    in_window = (pl.col("created_at") >= pl.col("fill_at")) & (
        pl.col("created_at") < pl.col("fill_at") + pl.duration(seconds=window_s)
    )
    sell = pl.col("taker_side") == "no"
    joined = (
        sig.select("_sig_id", "market_ticker", "fill_at", "fill_bid")
        .join(
            trades.select("market_ticker", "created_at", "yes_price", "taker_side", "count"),
            on="market_ticker",
            how="inner",
        )
        .filter(in_window)
    )
    agg = joined.group_by("_sig_id").agg(
        pl.len().alias("n_prints"),
        pl.col("count").sum().alias("traded_count"),
        (sell & (pl.col("yes_price") >= pl.col("fill_bid") - price_tol)).sum().alias("n_confirm"),
        pl.col("yes_price").filter(sell).max().alias("best_sell_yes"),
    )
    return (
        sig.join(agg, on="_sig_id", how="left")
        .drop("_sig_id")
        .with_columns(
            pl.col("n_prints").fill_null(0),
            pl.col("traded_count").fill_null(0.0),
            pl.col("n_confirm").fill_null(0),
        )
        .with_columns((pl.col("n_confirm") > 0).alias("confirmed"))
    )
