"""Edge scan - locked buckets vs book, minus fee/buffer, signal if net edge >= threshold.

PM Weather taker fee (5%): fee = 0.05 x p x (1-p), p = NO fill price
net_edge = 1.0 - no_ask - fee - exec_buffer"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from weadge.resolver.markets import Bucket
from weadge.resolver.observations import ObservedState
from weadge.resolver.state import BucketState, ResolutionState

TAKER_FEE_RATE = 0.05  # PM Weather 分类, 见 docs.polymarket.com/trading/fees


def taker_fee(price: float) -> float:
    """100 股 1 股对应的 fee(价格单位)。极端价格下 ≈ 0。"""
    return TAKER_FEE_RATE * price * (1.0 - price)


@dataclass(frozen=True)
class Signal:
    city: str
    target_date: date
    bucket: Bucket
    no_ask: float
    fee: float
    net_edge: float


def find_edges(
    city: str,
    target_date: date,
    buckets: list[BucketState],
    obs: ObservedState,
    min_net_edge: float = 0.02,
    exec_buffer: float = 0.01,
) -> list[Signal]:
    """LOCKED 桶中 NO ask 明显低于理论价 1.0 的 -> Signal。纯函数。

    不含 stale 观测的过滤 ---- 由 service 层在信号发出前统一把关
    (obs.stale 时整批丢弃, 不逐桶判断)。
    """
    signals: list[Signal] = []
    for bs in buckets:
        if bs.state is not ResolutionState.LOCKED:
            continue
        no_ask = bs.bucket.no_price
        fee = taker_fee(no_ask)
        net_edge = 1.0 - no_ask - fee - exec_buffer
        if net_edge >= min_net_edge:
            signals.append(
                Signal(
                    city=city,
                    target_date=target_date,
                    bucket=bs.bucket,
                    no_ask=no_ask,
                    fee=fee,
                    net_edge=net_edge,
                )
            )
    return signals
