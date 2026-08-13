"""Execution - deliberately thin. V0 keeps only the interface; trade mode is NotImplementedError.

shadow/alert/trade share one signal path, differing only in the last step:
- shadow: record simulated fills (JSONL)
- alert: notify (Telegram when token present)
- trade: real orders (v1+)"""

from __future__ import annotations


def place_limit(market_id: str, side: str, price: float, size: int) -> str:
    """v1+(PM CLOB 下单, 需 API key)。"""
    raise NotImplementedError("trade mode is v1+ — shadow/alert first")


def cancel(order_id: str) -> None:
    raise NotImplementedError("trade mode is v1+ — shadow/alert first")


def get_open_orders() -> list[str]:
    raise NotImplementedError("trade mode is v1+ — shadow/alert first")
