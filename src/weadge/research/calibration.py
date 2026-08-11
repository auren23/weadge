"""Calibration analysis: reliability curves and reliability index."""

from __future__ import annotations

import numpy as np
import polars as pl


def calibration_curve(
    y: np.ndarray,
    p: np.ndarray,
    bins: int | np.ndarray = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin_centers, observed_freq, bin_counts) for a fixed-bin calibration curve."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if isinstance(bins, int):
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    observed = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(centers) - 1)
    for i in range(len(centers)):
        mask = idx == i
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            observed[i] = y[mask].mean()
    return centers, observed, counts


def reliability_index(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Mean |predicted - observed| across bins weighted by bin count."""
    centers, observed, counts = calibration_curve(y, p, bins)
    mask = ~np.isnan(observed) & (counts > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(np.abs(centers[mask] - observed[mask]), weights=counts[mask]))


def calibration_table(
    df: pl.DataFrame,
    prob_columns: list[str],
    label_col: str = "result",
    bins: int = 10,
) -> pl.DataFrame:
    """Reliability index per model, plus overall observed frequency."""
    y = df[label_col].to_numpy()
    rows = []
    for col in prob_columns:
        p = df[col].to_numpy()
        rows.append(
            {
                "model": col,
                "n": len(p),
                "reliability_index": reliability_index(y, p, bins),
                "mean_p": float(np.nanmean(p)) if len(p) else float("nan"),
                "base_rate": float(np.nanmean(y)) if len(y) else float("nan"),
            }
        )
    return pl.DataFrame(rows)
