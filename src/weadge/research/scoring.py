"""Scoring rules for probability forecasts.

Brier and LogLoss are computed on the raw probability, with a floor to keep
log loss finite. Scores are aggregated per (model, lead bucket) and reported
as a plain table — calibration/edge analysis is separate.
"""

from __future__ import annotations

import numpy as np
import polars as pl

LOG_FLOOR = 1e-6


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), LOG_FLOOR, 1.0 - LOG_FLOOR)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score_frame(
    df: pl.DataFrame,
    prob_columns: list[str],
    label_col: str = "result",
) -> pl.DataFrame:
    """One row per probability column: Brier + LogLoss over all rows.

    Fail-closed columns (e.g. p_market_normalized / p_market_simplex when a
    partition is incomplete) carry nulls; each model is scored on its
    NON-NULL rows only, and `n` reports that count.
    """
    y = df[label_col].to_numpy().astype(float)
    rows = []
    for col in prob_columns:
        p_full = df[col].to_numpy().astype(float)
        mask = ~np.isnan(p_full)
        p = p_full[mask]
        if len(p) == 0:
            rows.append({"model": col, "n": 0, "brier": float("nan"), "log_loss": float("nan")})
            continue
        rows.append(
            {
                "model": col,
                "n": len(p),
                "brier": brier_score(y[mask], p),
                "log_loss": log_loss(y[mask], p),
            }
        )
    return pl.DataFrame(rows)


def score_by_lead_bucket(
    df: pl.DataFrame,
    prob_columns: list[str],
    lead_buckets: list[tuple[str, float, float]],  # (label, min_hours, max_hours)
    label_col: str = "result",
) -> pl.DataFrame:
    """Brier/LogLoss per model per lead-time bucket.

    A row belongs to the FIRST bucket with min <= lead < max.
    """
    out: list[pl.DataFrame] = []
    for label, lo, hi in lead_buckets:
        sub = df.filter((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
        if sub.is_empty():
            continue
        sc = score_frame(sub, prob_columns, label_col).with_columns(
            pl.lit(label).alias("lead_bucket")
        )
        out.append(sc)
    return pl.concat(out) if out else pl.DataFrame(
        schema={"model": pl.Utf8, "n": pl.Int64, "brier": pl.Float64,
                "log_loss": pl.Float64, "lead_bucket": pl.Utf8}
    )
