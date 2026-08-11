"""Model layer.

v0 model policy (from the research plan):
  * Market       — the baseline; p_market_simplex (box-constrained projection
                   of the bucket mids) is the default market construction
  * NBM          — the free calibrated baseline (consumed as p_nbm)
  * Stack        — logistic stacking of market + weather (M2 in edge.py)
  * EMOS/GEFS    — challengers ONLY after the NBM baseline is beaten OOS

EMOS is deliberately NOT implemented here. See research/edge.py for the
incremental test that decides when a challenger earns a place.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm

from weadge.domain.probability import prob_to_logit


def stack_probability(
    train: pl.DataFrame,
    test: pl.DataFrame,
    market_col: str = "p_market_simplex",
    weather_col: str = "p_nbm",
    label_col: str = "result",
) -> pl.Series:
    """OOS stacking probability: fit logit(p) = a + b*logit(p_market) + g*logit(p_weather)
    on train, predict on test. Returns a Series aligned with `test` rows that
    have both features present (others are NaN)."""
    X_tr = np.column_stack(
        [prob_to_logit(train[c].to_numpy()) for c in (market_col, weather_col)]
    )
    y_tr = train[label_col].to_numpy()
    keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    if keep.sum() < 20 or len(np.unique(y_tr[keep])) < 2:
        return pl.Series("p_stack", [float("nan")] * test.height, dtype=pl.Float64)

    logit = sm.Logit(y_tr[keep], sm.add_constant(X_tr[keep])).fit(disp=0)
    X_te = np.column_stack(
        [prob_to_logit(test[c].to_numpy()) for c in (market_col, weather_col)]
    )
    pred = logit.predict(sm.add_constant(X_te))
    pred[~np.isfinite(pred)] = float("nan")
    return pl.Series("p_stack", pred, dtype=pl.Float64)


def emos_placeholder() -> None:
    """EMOS is a v1 challenger. Do not call before G2 (incremental alpha) passes."""
    raise NotImplementedError(
        "EMOS is deliberately deferred: it may only enter after the NBM "
        "baseline beats the market OOS (alpha gate G2)."
    )


__all__ = ["emos_placeholder", "stack_probability"]
