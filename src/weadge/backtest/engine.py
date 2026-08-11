"""Taker backtest engine (v0, deliberately minimal).

Only one strategy exists in v0:
    if p_model - ask >= threshold   -> BUY YES, hold to settlement.

No laddering, no take-profit, no stop-loss, no dynamic Kelly. If this
cannot beat the market after costs, no amount of strategy machinery will.

Every trade pays the fee that was actually in effect at execution time
(FeeSchedule), and fills at the delayed ask open (no same-bar fills).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from weadge.backtest.execution import execute_delayed
from weadge.backtest.fees import FeeSchedule


@dataclass
class TradeRecord:
    market_ticker: str
    signal_at: datetime
    execute_at: datetime | None
    fill_price: float | None
    fee: float | None
    p_model: float
    result: int  # 1 = YES settles $1
    gross_edge: float | None      # p_model - ask (pre-fee)
    net_ev: float | None          # p_model - (price + fee)
    pnl: float | None             # realized: result - (price + fee)


@dataclass
class BacktestReport:
    signals: int
    trades: int
    filled: int
    gross_edge: float            # mean p_model - ask over filled trades
    fee_cost: float              # mean fee over filled trades
    slippage: float              # mean (ask_at_delay - ask_at_signal) over filled trades
    net_realized_edge: float     # mean pnl over filled trades
    net_pnl: float
    max_drawdown: float
    edge_bins: pl.DataFrame      # predicted edge bin -> realized edge
    trades_table: pl.DataFrame

    def as_text(self) -> str:
        return (
            f"signals              {self.signals}\n"
            f"trades               {self.trades}\n"
            f"filled               {self.filled}\n"
            f"gross edge           {self.gross_edge:+.2%}\n"
            f"fees                 {self.fee_cost:.2%}\n"
            f"slippage             {self.slippage:.2%}\n"
            f"net realized edge    {self.net_realized_edge:+.2%}\n"
            f"net PnL              {self.net_pnl:+.2f}\n"
            f"max drawdown         {self.max_drawdown:.2%}"
        )


def run_taker_backtest(
    signals: pl.DataFrame,
    quotes: pl.DataFrame,
    fee_schedule: FeeSchedule,
    *,
    threshold: float = 0.06,
    delay_min: int = 1,
    p_model_col: str = "p_model",
    result_col: str = "result",
    signal_ts_col: str = "decision_at",
    market_col: str = "market_ticker",
    ask_col: str = "yes_ask_close",
) -> BacktestReport:
    """Run the v0 taker strategy and return a report.

    `signals` rows must carry p_model, result (0/1) and decision_at.

    Candle availability: the ask used for the threshold decision is the ask
    CLOSE of the last COMPLETED bar at or before decision_at
    (bar_end_at <= decision_at) — a bar's close is not knowable until the
    bar has ended. The fill is the ask OPEN of the first bar starting at or
    after decision_at + delay (default +1 minute), i.e. the next bar's open
    after the signal. This deliberately trades one minute slower than the
    theoretical earliest fill; honesty beats hypothetical latency edge.
    """
    if signals.is_empty():
        return _empty_report(signals)

    quotes = _ensure_bar_bounds(quotes)
    trades: list[TradeRecord] = []
    candidates = 0
    for sig in signals.iter_rows(named=True):
        mk = sig[market_col]
        sig_ts = sig[signal_ts_col]
        p_model = float(sig[p_model_col])
        result = int(sig[result_col])
        candidates += 1

        # gate ask: last COMPLETED bar's ask close (bar_end_at <= decision_at)
        ask_signal = _ask_at_completed_bar(quotes, mk, sig_ts, ask_col)
        if ask_signal is None or np.isnan(ask_signal):
            continue
        gross = p_model - ask_signal
        if gross < threshold:
            continue

        fill = execute_delayed(quotes, mk, sig_ts, side="buy_yes", delay_min=delay_min)
        if fill.price is None or fill.execute_at is None:
            continue
        fee = fee_schedule.fee_cost(fill.price, fill.execute_at)
        pnl = result * 1.0 - (fill.price + fee)
        trades.append(
            TradeRecord(
                market_ticker=mk,
                signal_at=sig_ts,
                execute_at=fill.execute_at,
                fill_price=fill.price,
                fee=fee,
                p_model=p_model,
                result=result,
                gross_edge=gross,
                net_ev=p_model - (fill.price + fee),
                pnl=pnl,
            )
        )

    if not trades:
        return _empty_report(signals)

    df = pl.DataFrame(
        [
            {
                "market_ticker": t.market_ticker,
                "signal_at": t.signal_at,
                "execute_at": t.execute_at,
                "fill_price": t.fill_price,
                "fee": t.fee,
                "p_model": t.p_model,
                "result": t.result,
                "gross_edge": t.gross_edge,
                "net_ev": t.net_ev,
                "pnl": t.pnl,
            }
            for t in trades
        ]
    )
    # recompute slippage = fill_price - ask_at_signal (need ask_signal per trade)
    df = _add_slippage(df, trades)

    pnls = df["pnl"].to_numpy()
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = float(np.max(peak - equity)) if len(equity) else 0.0
    max_dd = float(dd / max(1.0, float(np.max(peak))))

    report = BacktestReport(
        signals=candidates,
        trades=len(trades),
        filled=len(trades),
        gross_edge=float(df["gross_edge"].to_numpy().mean()),
        fee_cost=float(df["fee"].to_numpy().mean()),
        slippage=float(df["slippage"].to_numpy().mean()),
        net_realized_edge=float(df["pnl"].to_numpy().mean()),
        net_pnl=float(df["pnl"].to_numpy().sum()),
        max_drawdown=max_dd,
        edge_bins=_edge_bins(df),
        trades_table=df,
    )
    return report


def _add_slippage(df: pl.DataFrame, trades: list[TradeRecord]) -> pl.DataFrame:
    """Attach slippage = fill_price - gate ask (last completed bar's ask close)."""
    gross = df["gross_edge"].to_list()
    # signal-bar ask = p_model - gross_edge
    p_model = df["p_model"].to_list()
    ask_signal = [p - g if g is not None else None for p, g in zip(p_model, gross, strict=True)]
    fill = df["fill_price"].to_list()
    slippage = [
        (f - a) if (f is not None and a is not None) else None
        for f, a in zip(fill, ask_signal, strict=True)
    ]
    return df.with_columns(pl.Series("slippage", slippage, dtype=pl.Float64))


def _ensure_bar_bounds(quotes: pl.DataFrame) -> pl.DataFrame:
    """Attach bar_start_at / bar_end_at, deriving them from ts when absent
    (legacy frames predating the columns): ts is the bar start, a 1m bar
    completes one minute later."""
    if "bar_end_at" in quotes.columns:
        return quotes
    return quotes.with_columns(
        pl.col("ts").alias("bar_start_at"),
        (pl.col("ts") + timedelta(minutes=1)).alias("bar_end_at"),
    )


def _ask_at_completed_bar(
    quotes: pl.DataFrame,
    market_ticker: str,
    ts: datetime,
    ask_col: str,
) -> float | None:
    """Ask of the last COMPLETED bar at or before `ts` (bar_end_at <= ts).

    A bar's close only exists once the bar has ended: with decision at
    12:05:00, the 12:05 bar (covering 12:05-12:06) is NOT complete, so the
    gate is the 12:04 bar's close.
    """
    before = (
        quotes.filter(pl.col("market_ticker") == market_ticker)
        .filter(pl.col("bar_end_at") <= ts)
        .sort("bar_end_at")
    )
    if before.is_empty():
        return None
    return float(before.tail(1)[ask_col][0])


def _edge_bins(trades: pl.DataFrame) -> pl.DataFrame:
    """Predicted-edge bin -> realized edge (monotonicity check)."""
    edges = [0.06, 0.08, 0.10, 0.12, np.inf]
    labels = ["6-8%", "8-10%", "10-12%", "12%+"]
    bins = np.digitize(trades["gross_edge"].to_numpy(), edges[:-1], right=False)
    rows = []
    for i, label in enumerate(labels):
        mask = bins == i
        if not mask.any():
            continue
        rows.append(
            {
                "edge_bin": label,
                "n": int(mask.sum()),
                "realized_edge": float(trades["pnl"].to_numpy()[mask].mean()),
            }
        )
    return pl.DataFrame(rows)


def _empty_report(signals: pl.DataFrame) -> BacktestReport:
    return BacktestReport(
        signals=int(signals.height),
        trades=0,
        filled=0,
        gross_edge=0.0,
        fee_cost=0.0,
        slippage=0.0,
        net_realized_edge=0.0,
        net_pnl=0.0,
        max_drawdown=0.0,
        edge_bins=pl.DataFrame(schema={"edge_bin": pl.Utf8, "n": pl.Int64, "realized_edge": pl.Float64}),
        trades_table=pl.DataFrame(
            schema={
                "market_ticker": pl.Utf8, "signal_at": pl.Datetime("us", time_zone="UTC"),
                "execute_at": pl.Datetime("us", time_zone="UTC"), "fill_price": pl.Float64,
                "fee": pl.Float64, "p_model": pl.Float64, "result": pl.Int64,
                "gross_edge": pl.Float64, "net_ev": pl.Float64, "pnl": pl.Float64,
            }
        ),
    )
