"""Paper trading — v2 feature, intentionally not implemented in v0."""

from __future__ import annotations


def paper_run(series: str) -> None:
    """Placeholder for the paper-trading loop (after live recorder + live gate)."""
    raise NotImplementedError(
        "paper trading is a v2 component. It may only start after gates "
        "G0..G5 (settlement -> forecast -> incremental -> economic -> robustness) pass."
    )
