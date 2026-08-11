"""Incremental alpha existence test.

The research question is NOT "is the weather forecast accurate?" — it is:

    logit(P(Y=1)) = a + b*logit(p_market) + g*logit(p_weather)

    Is gamma still significant OUT of sample once the market is in the model?

M0: market only | M1: weather only | M2: market + weather.
Fit on train, score on test (walk-forward). No shuffling, ever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm

from weadge.domain.probability import prob_to_logit


@dataclass(frozen=True)
class IncrementalResult:
    model: str
    features: tuple[str, ...]
    train_n: int
    test_n: int
    gamma: float | None          # weather coefficient in M2
    gamma_pvalue: float | None   # two-sided p-value of gamma in M2
    test_log_loss: float
    test_brier: float


def _logit_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = np.column_stack([prob_to_logit(df[c].to_numpy()) for c in feature_cols])
    return np.asarray(X, dtype=float)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_incremental(
    train: pl.DataFrame,
    test: pl.DataFrame,
    market_col: str = "p_market",
    weather_col: str = "p_nbm",
    label_col: str = "result",
) -> list[IncrementalResult]:
    """Fit M0 (market), M1 (weather), M2 (market+weather) on train; eval on test.

    Rows with missing features or label are dropped per model. Requires
    train/test to be temporally disjoint (enforced by the caller's walk-forward
    split — this function does not re-split).
    """
    y_tr = train[label_col].to_numpy()
    y_te = test[label_col].to_numpy()
    results: list[IncrementalResult] = []

    for name, feats in (("M0", [market_col]), ("M1", [weather_col]), ("M2", [market_col, weather_col])):
        X_tr = _logit_matrix(train, feats)
        X_te = _logit_matrix(test, feats)
        # drop rows with NaN in features
        keep_tr = ~np.isnan(X_tr).any(axis=1)
        keep_te = ~np.isnan(X_te).any(axis=1)
        X_tr, y_tr_m = X_tr[keep_tr], y_tr[keep_tr]
        X_te, y_te_m = X_te[keep_te], y_te[keep_te]
        if len(np.unique(y_tr_m)) < 2 or len(y_tr_m) < 10 or len(y_te_m) == 0:
            results.append(
                IncrementalResult(name, tuple(feats), len(y_tr_m), len(y_te_m),
                                  None, None, float("nan"), float("nan"))
            )
            continue

        logit = sm.Logit(y_tr_m, sm.add_constant(X_tr)).fit(disp=0)
        pred = logit.predict(sm.add_constant(X_te))
        gamma = None
        gamma_p = None
        if name == "M2":
            gamma = float(logit.params[-1])
            gamma_p = float(logit.pvalues[-1])
        results.append(
            IncrementalResult(
                model=name,
                features=tuple(feats),
                train_n=len(y_tr_m),
                test_n=len(y_te_m),
                gamma=gamma,
                gamma_pvalue=gamma_p,
                test_log_loss=_log_loss(y_te_m, pred),
                test_brier=_brier(y_te_m, pred),
            )
        )
    return results


def gamma_has_oos_value(results: list[IncrementalResult], alpha: float = 0.05) -> bool:
    """True iff M2's weather coefficient is significant in-sample AND the
    combined model beats market-only OOS log loss (not just gamma != 0)."""
    m2 = next((r for r in results if r.model == "M2"), None)
    m0 = next((r for r in results if r.model == "M0"), None)
    if m2 is None or m0 is None or m2.gamma_pvalue is None:
        return False
    significant = m2.gamma_pvalue < alpha
    better_oos = m2.test_log_loss < m0.test_log_loss
    return significant and better_oos
