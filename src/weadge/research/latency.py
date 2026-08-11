"""Latency stress test.

Execution must be delayed by at least one bar: a signal at 12:01 executes at
12:02's ask OPEN. This destroys any alpha that only exists because you used
the same candle you "decided" on.

Stress test: recompute realized edges for +1/+2/+5/+10 minute delays.
If edge collapses from +8% at +0m to +0.4% at +1m, the result was a data
artifact, not a tradable edge.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from weadge.domain.probability import edge as edge_func


def delayed_execution_edges(
    signals: pl.DataFrame,
    quotes: pl.DataFrame,
    delays_min: list[int] | tuple[int, ...] = (1, 2, 5, 10),
    ts_col: str = "ts",
    signal_col: str = "signal_at",
    quote_key: str = "market_ticker",
    ask_col: str = "yes_ask_open",
) -> pl.DataFrame:
    """For each signal, the ask_open `delay_min` after the signal bar.

    `signals` must carry: market_ticker, signal_at, p_model.
    Returns one row per (signal, delay): price, edge_at_delay, available.
    """
    quotes = quotes.sort(ts_col)
    rows = []
    for sig in signals.iter_rows(named=True):
        mk = sig[quote_key]
        sig_ts = sig[signal_col]
        p_model = float(sig["p_model"])
        sub = quotes.filter(pl.col(quote_key) == mk).sort(ts_col)
        for delay in delays_min:
            target = sig_ts + timedelta(minutes=delay)
            # exact or first bar at/after target
            after = sub.filter(pl.col(ts_col) >= target)
            fill = after.head(1)
            if fill.is_empty():
                rows.append(
                    {"market_ticker": mk, "delay_min": delay, "price": None,
                     "edge": None, "available": False}
                )
                continue
            price = float(fill[ask_col][0])
            rows.append(
                {
                    "market_ticker": mk,
                    "delay_min": delay,
                    "price": price,
                    "edge": edge_func(p_model, price),
                    "available": True,
                }
            )
    return pl.DataFrame(rows)


def edge_by_delay_summary(delayed: pl.DataFrame) -> pl.DataFrame:
    """Mean edge and fill rate per delay minute."""
    return (
        delayed.group_by("delay_min")
        .agg(
            pl.col("edge").mean().alias("mean_edge"),
            pl.col("available").mean().alias("fill_rate"),
            pl.len().alias("n"),
        )
        .sort("delay_min")
    )
