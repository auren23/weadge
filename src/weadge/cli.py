"""weadge CLI - thin production entry.

    weadge scan                     # one resolver scan (shadow)
    weadge serve                    # resolver loop (30s)
    weadge stats                    # kill-test summary from shadow logs

Frozen research v1 commands (kalshi/dataset/noaa/research/backtest/live) are
registered lazily from weadge.research_cli: heavy deps (polars, duckdb, ...)
load per-command, never at startup. `weadge --help` stays fast.
"""

from __future__ import annotations

from typing import Annotated

import typer

from weadge.research_cli import (
    backtest_app,
    dataset_app,
    kalshi_app,
    live_app,
    noaa_app,
    research_app,
)

app = typer.Typer(
    help="weadge — weather resolution-mispricing trader (resolver lane active, research v1 frozen)",
    no_args_is_help=True,
)

# frozen research v1 lane (per-command heavy imports)
app.add_typer(kalshi_app, name="kalshi")
app.add_typer(dataset_app, name="dataset")
app.add_typer(noaa_app, name="noaa")
app.add_typer(research_app, name="research")
app.add_typer(backtest_app, name="backtest")
app.add_typer(live_app, name="live")

_city = Annotated[str, typer.Option("--city", help="city slug from config/resolver.yaml")]
_mode = Annotated[str, typer.Option("--mode", help="shadow | alert | trade(v1+)")]


@app.command("scan")
def scan(
    city: _city = "paris",
    mode: _mode = "shadow",
) -> None:
    """One resolver scan: events -> observations -> LOCKED -> book -> log."""
    from weadge.resolver.service import scan as run_scan

    run_scan(city, mode)


@app.command("serve")
def serve(
    city: _city = "paris",
    mode: _mode = "shadow",
    interval: Annotated[int, typer.Option("--interval", help="seconds between scans")] = 30,
) -> None:
    """Loop scans within the scan window (no WS needed for V0)."""
    from weadge.resolver.service import serve as run_serve

    run_serve(city, mode, interval)


@app.command("stats")
def stats(city: _city = "paris") -> None:
    """Kill-test summary from shadow logs (lock events, ask at lock, reaction)."""
    from weadge.resolver.service import stats as run_stats

    run_stats(city)


if __name__ == "__main__":
    app()
