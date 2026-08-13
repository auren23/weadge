"""Edge scan - locked buckets vs executable book, minus fee/buffer.

PM Weather taker fee (5%): fee = 0.05 x p x (1-p), p = NO fill price
net_edge = 1.0 - executable_no_ask - fee - exec_buffer

gamma outcomePrices is display pricing only; the executable ask comes from
the CLOB /book top of book (Paris 37C bucket showed 0.415/0.585 display vs
0.01/0.99 real book on 2026-08-13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from weadge.resolver.markets import Book, Bucket
from weadge.resolver.state import BucketState, ResolutionState

TAKER_FEE_RATE = 0.05  # PM Weather category, docs.polymarket.com/trading/fees


def taker_fee(price: float) -> float:
    """Fee per share (price units); ~0 at extreme prices."""
    return TAKER_FEE_RATE * price * (1.0 - price)


@dataclass(frozen=True)
class LockAssessment:
    """One LOCKED bucket with its executable book snapshot and net edge."""

    bucket: Bucket
    state: ResolutionState
    no_ask: float | None      # executable best ask; None = no resting ask
    no_ask_size: float
    book_ts: datetime | None
    fee: float
    net_edge: float | None
    signal: bool              # net_edge >= min_net_edge


def find_edges(
    buckets: list[BucketState],
    books: dict[str, Book],
    min_net_edge: float = 0.02,
    exec_buffer: float = 0.01,
) -> list[LockAssessment]:
    """Assess every LOCKED bucket against its executable book. Pure.

    Buckets without a book (token resolution failed / no resting ask) are
    still returned as untradeable assessments so the kill test sees them.
    """
    assessments: list[LockAssessment] = []
    for bs in buckets:
        if bs.state is not ResolutionState.LOCKED:
            continue
        book = books.get(bs.bucket.market_id)
        if book is None or book.best_ask is None:
            assessments.append(
                LockAssessment(bs.bucket, bs.state, None, 0.0, None, 0.0, None, False)
            )
            continue
        no_ask = book.best_ask
        fee = taker_fee(no_ask)
        net_edge = 1.0 - no_ask - fee - exec_buffer
        assessments.append(
            LockAssessment(
                bs.bucket,
                bs.state,
                no_ask,
                book.best_ask_size,
                book.ts,
                fee,
                net_edge,
                signal=net_edge >= min_net_edge,
            )
        )
    return assessments
