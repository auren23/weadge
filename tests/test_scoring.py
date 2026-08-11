"""Scoring rules: Brier and LogLoss behave correctly."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from weadge.research.scoring import brier_score, log_loss, score_by_lead_bucket, score_frame


class TestScoring:
    def test_perfect_predictions(self) -> None:
        y = np.array([1.0, 1.0, 0.0, 0.0])
        p = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y, p) == pytest.approx(0.0)
        # log loss is clipped at 1e-6 to stay finite, so near-zero but not exact
        assert log_loss(y, p) < 1e-3

    def test_coin_flip_brier(self) -> None:
        y = np.array([1.0, 0.0])
        p = np.array([0.5, 0.5])
        assert brier_score(y, p) == pytest.approx(0.25)

    def test_log_loss_flips_penalized(self) -> None:
        # confident wrong is worse than uncertain
        y = np.array([1.0])
        assert log_loss(y, np.array([0.99])) < log_loss(y, np.array([0.5]))

    def test_clipping_keeps_log_loss_finite(self) -> None:
        assert np.isfinite(log_loss(np.array([1.0]), np.array([0.0])))
        assert np.isfinite(log_loss(np.array([0.0]), np.array([1.0])))

    def test_score_frame(self) -> None:
        df = pl.DataFrame(
            {
                "result": [1, 0, 1, 0],
                "p_market": [0.6, 0.4, 0.7, 0.3],
                "p_nbm": [0.9, 0.1, 0.8, 0.2],
            }
        )
        table = score_frame(df, ["p_market", "p_nbm"])
        assert table["model"].to_list() == ["p_market", "p_nbm"]
        # nbm is better calibrated here -> lower brier
        assert table.filter(pl.col("model") == "p_nbm")["brier"][0] < table.filter(
            pl.col("model") == "p_market"
        )["brier"][0]

    def test_score_frame_scores_non_null_rows_only(self) -> None:
        """Fail-closed columns (e.g. p_market_simplex) can be partially null;
        each model must be scored on its own non-null rows, with n reporting
        that count — never NaN-poisoned aggregates."""
        df = pl.DataFrame(
            {
                "result": [1, 0, 1, 0],
                "p_market_raw": [0.6, 0.4, 0.7, 0.3],
                "p_market_simplex": [0.6, None, 0.7, None],
            }
        )
        table = score_frame(df, ["p_market_raw", "p_market_simplex"])
        raw = table.filter(pl.col("model") == "p_market_raw").row(0, named=True)
        sim = table.filter(pl.col("model") == "p_market_simplex").row(0, named=True)
        assert raw["n"] == 4
        assert sim["n"] == 2
        assert np.isfinite(sim["brier"])
        assert np.isfinite(sim["log_loss"])
        # the two non-null simplex rows align with the same result labels
        assert sim["brier"] == pytest.approx(brier_score(np.array([1.0, 1.0]), np.array([0.6, 0.7])))

    def test_score_by_lead_bucket(self) -> None:
        df = pl.DataFrame(
            {
                "result": [1, 0, 1, 0],
                "lead_hours": [30.0, 15.0, 5.0, 1.0],
                "p_market": [0.6, 0.4, 0.7, 0.3],
            }
        )
        buckets = [("24-48h", 24, 48), ("12-24h", 12, 24), ("6-12h", 6, 12), ("1-6h", 1, 6)]
        table = score_by_lead_bucket(df, ["p_market"], buckets)
        assert table["lead_bucket"].to_list() == ["24-48h", "12-24h", "1-6h"]
