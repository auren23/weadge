"""Incremental alpha existence test.

The research question is NOT "is the weather forecast accurate?" — it is:

    logit(P(Y=1)) = a + b*logit(p_market) + g*logit(p_weather)

    Does weather add OUT-OF-SAMPLE information the market does not have?

M0: market only | M1: weather only | M2: market + weather.
Fit on train, score on test (walk-forward). No shuffling, ever.

Statistical discipline (both are P0 for this file):

  * PAIRED SAMPLES: M0/M1/M2 are fit and scored on the SAME rows. A row
    with any missing feature (or label) is dropped from ALL models, never
    per model — comparing M0 on 1000 rows with M2 on 700 rows is not a
    comparison at all.

  * CLUSTERED OOS INFERENCE: one event produces many rows (buckets x lead
    times), so rows are not independent. The gate is the OOS delta
    log-loss with an event/date clustered bootstrap (mean > 0 AND 95%
    cluster CI lower bound > 0). An in-sample IID coefficient p-value is
    supporting evidence only — a gamma p=1e-5 can be one day's information
    counted dozens of times.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm

from weadge.domain.probability import prob_to_logit

# Market probabilities are midpoints of 1-cent quotes: certainty beyond
# ~0.999 (logit +-7) is quote noise, not signal. Clipping the logit
# FEATURES keeps the logistic fit finite under the near-perfect
# separation that short-lead market prices produce (Newton's Hessian
# goes singular when a coefficient is unbounded). The same transform is
# applied to train and test, so the paired M0/M1/M2 comparison is
# unchanged.
LOGIT_FEATURE_CLIP = 7.0


@dataclass(frozen=True)
class IncrementalResult:
    model: str
    features: tuple[str, ...]
    train_n: int
    test_n: int
    gamma: float | None  # weather coefficient in M2 (in-sample, auxiliary)
    gamma_pvalue: float | None  # two-sided p-value of gamma in M2 (auxiliary)
    test_log_loss: float
    test_brier: float


@dataclass(frozen=True)
class IncrementalGateResult:
    """OOS paired delta log-likelihood with clustered bootstrap CI.

    delta_ll = mean per-row (ll_M2 - ll_M0) over the test set, which equals
    OOS log-loss(M0) - log-loss(M2): positive means the market+weather
    model is better OOS. ci_lower/ci_upper are the 2.5/97.5 percentiles of
    the event/date clustered bootstrap. NaN deltas mean the window was too
    small or degenerate to test (gate fails closed).
    """

    delta_ll: float
    ci_lower: float
    ci_upper: float
    gamma: float | None  # in-sample M2 weather coefficient (auxiliary)
    gamma_pvalue: float | None  # in-sample p-value (auxiliary, NOT the gate)
    test_n: int
    n_clusters: int
    seed: int

    @property
    def has_incremental_alpha(self) -> bool:
        """The gate: OOS mean delta_ll > 0 AND 95% cluster CI lower bound > 0.

        NaN deltas (degenerate window) compare False and fail closed.
        """
        return self.delta_ll > 0 and self.ci_lower > 0


def _logit_matrix(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = np.column_stack([prob_to_logit(df[c].to_numpy()) for c in feature_cols])
    X = np.clip(X, -LOGIT_FEATURE_CLIP, LOGIT_FEATURE_CLIP)
    return np.asarray(X, dtype=float)


def _fit_logit(y: np.ndarray, X: np.ndarray):
    """Logistic fit robust to near-perfect separation: BFGS + clipped
    features keep the optimum finite. Returns None on failure — the caller
    then fails closed (the window is degenerate for inference)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PerfectSeparationWarning etc.
            return sm.Logit(y, sm.add_constant(X)).fit(disp=0, method="bfgs", maxiter=200)
    except Exception:
        return None


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _row_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row log likelihood contribution (higher = better)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def _paired_masks(
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Common valid-row masks for train and test across ALL features.

    A row with NaN in ANY feature (or the label) is dropped for every
    model, so all models see identical observations (paired comparison).
    """
    y_tr = train[label_col].to_numpy().astype(float)
    y_te = test[label_col].to_numpy().astype(float)
    X_tr = _logit_matrix(train, feature_cols)
    X_te = _logit_matrix(test, feature_cols)
    keep_tr = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    keep_te = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te)
    return X_tr, X_te, keep_tr, keep_te


def fit_incremental(
    train: pl.DataFrame,
    test: pl.DataFrame,
    market_col: str = "p_market_simplex",
    weather_col: str = "p_nbm",
    label_col: str = "result",
) -> list[IncrementalResult]:
    """Fit M0 (market), M1 (weather), M2 (market+weather) on train; eval on test.

    PAIRED: all three models use the same valid rows on both train and test
    (a row missing any feature is dropped from every model), so their OOS
    metrics are directly comparable. Requires train/test to be temporally
    disjoint (enforced by the caller's walk-forward split — this function
    does not re-split).
    """
    all_feats = [market_col, weather_col]
    X_tr, X_te, keep_tr, keep_te = _paired_masks(train, test, all_feats, label_col)
    y_tr = train[label_col].to_numpy().astype(float)[keep_tr]
    y_te = test[label_col].to_numpy().astype(float)[keep_te]
    idx_tr, idx_te = np.where(keep_tr)[0], np.where(keep_te)[0]
    results: list[IncrementalResult] = []

    for name, feats in (("M0", [market_col]), ("M1", [weather_col]), ("M2", all_feats)):
        cols = [all_feats.index(f) for f in feats]
        X_tr_m = X_tr[idx_tr][:, cols]
        X_te_m = X_te[idx_te][:, cols]
        if len(np.unique(y_tr)) < 2 or len(y_tr) < 10 or len(y_te) == 0:
            results.append(
                IncrementalResult(
                    name, tuple(feats), len(y_tr), len(y_te), None, None, float("nan"), float("nan")
                )
            )
            continue

        logit = _fit_logit(y_tr, X_tr_m)
        if logit is None:  # degenerate fit — fail closed, paired mask intact
            results.append(
                IncrementalResult(
                    name, tuple(feats), len(y_tr), len(y_te), None, None, float("nan"), float("nan")
                )
            )
            continue
        pred = logit.predict(sm.add_constant(X_te_m))
        gamma = None
        gamma_p = None
        if name == "M2":
            gamma = float(logit.params[-1])
            gamma_p = float(logit.pvalues[-1])
        results.append(
            IncrementalResult(
                model=name,
                features=tuple(feats),
                train_n=len(y_tr),
                test_n=len(y_te),
                gamma=gamma,
                gamma_pvalue=gamma_p,
                test_log_loss=_log_loss(y_te, pred),
                test_brier=_brier(y_te, pred),
            )
        )
    return results


def paired_incremental_gate(
    train: pl.DataFrame,
    test: pl.DataFrame,
    market_col: str = "p_market_simplex",
    weather_col: str = "p_nbm",
    label_col: str = "result",
    cluster_col: str = "event_date",
    n_boot: int = 2000,
    seed: int = 0,
) -> IncrementalGateResult:
    """The G2 gate: OOS delta log-loss (M0 vs M2) with a clustered bootstrap.

    Procedure:
      1. common (paired) valid rows on train and test
      2. fit M0 and M2 on the same train rows
      3. per test row: delta_i = ll_M0_i - ll_M2_i (positive = M2 better)
      4. mean delta over the test set
      5. 95% CI by resampling CLUSTERS (`cluster_col`, e.g. event_date)
         with replacement, n_boot draws

    Passes iff mean delta_ll > 0 AND the 95% cluster CI lower bound > 0.
    gamma keeps its role as supporting evidence only (direction/sign).
    A degenerate window (too few rows, one class, no test rows, < 2
    clusters) fails closed with NaN metrics.
    """
    all_feats = [market_col, weather_col]
    X_tr, X_te, keep_tr, keep_te = _paired_masks(train, test, all_feats, label_col)
    y_tr = train[label_col].to_numpy().astype(float)[keep_tr]
    y_te = test[label_col].to_numpy().astype(float)[keep_te]
    X_tr_k, X_te_k = X_tr[keep_tr], X_te[keep_te]

    empty = IncrementalGateResult(
        float("nan"), float("nan"), float("nan"), None, None, len(y_te), 0, seed
    )
    if len(y_tr) < 10 or len(np.unique(y_tr)) < 2 or len(y_te) == 0:
        return empty

    m0 = _fit_logit(y_tr, X_tr_k[:, [0]])
    m2 = _fit_logit(y_tr, X_tr_k)
    if m0 is None or m2 is None:  # degenerate fit — fail closed
        return empty
    p0 = m0.predict(sm.add_constant(X_te_k[:, [0]]))
    p2 = m2.predict(sm.add_constant(X_te_k))

    delta = _row_ll(y_te, p2) - _row_ll(y_te, p0)
    # per-row log-likelihood gain of M2 over M0; equivalently
    # log-loss(M0) - log-loss(M2). Positive = M2 better.
    mean_delta = float(delta.mean())

    clusters = test[cluster_col].to_numpy()[keep_te]
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        # a single cluster cannot support clustered inference — fail closed
        return IncrementalGateResult(
            mean_delta,
            float("nan"),
            float("nan"),
            float(m2.params[-1]),
            float(m2.pvalues[-1]),
            len(y_te),
            n_clusters,
            seed,
        )

    sums = np.zeros(n_clusters)
    counts = np.zeros(n_clusters)
    np.add.at(sums, inverse, delta)
    np.add.at(counts, inverse, 1.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    return IncrementalGateResult(
        delta_ll=mean_delta,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        gamma=float(m2.params[-1]),
        gamma_pvalue=float(m2.pvalues[-1]),
        test_n=len(y_te),
        n_clusters=n_clusters,
        seed=seed,
    )
