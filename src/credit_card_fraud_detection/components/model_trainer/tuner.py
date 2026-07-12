"""
components/model_trainer/tuner.py
==================================
Hyperparameter tuning for any TrainableModel using GridSearchCV,
RandomizedSearchCV, or (optionally) Bayesian search via scikit-optimize.

Design decisions:
  • CV is always StratifiedKFold — preserves class ratio in every fold,
    critical for fraud detection where folds can otherwise end up with
    very few minority samples.
  • Default scoring is AUPRC (average_precision) — the correct primary
    metric for imbalanced binary classification, not accuracy.
  • Tuning runs on a stratified subsample (tuning_sample_size) for speed;
    the FINAL model is always refit on the FULL training set with the
    best hyperparameters found, so production performance is not
    compromised by the subsample.
  • Falls back gracefully from "bayes" to "random" if scikit-optimize
    is not installed.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from credit_card_fraud_detection.components.model_trainer.interface import TrainableModel
from credit_card_fraud_detection.entity.config_entity import ModelTrainerConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class HyperparameterTuner:
    """
    Runs a hyperparameter search for a single TrainableModel and returns
    the best fitted pipeline, best params, and a full results report.
    """

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def tune(
        self,
        model: TrainableModel,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[TrainableModel, Dict[str, Any]]:
        """
        Tune *model* on a stratified subsample of (X, y), then refit the
        best estimator on the FULL (X, y).

        Returns:
            (model_with_best_pipeline, tuning_report_dict)
        """
        cfg        = self.config
        param_grid = model.get_param_grid()

        if not cfg.tuning_enabled or not param_grid:
            logger.info(f"  {model.name}: tuning disabled or empty grid — fitting with defaults")
            t0 = time.perf_counter()
            model.fit(X, y)
            elapsed = round(time.perf_counter() - t0, 2)
            return model, {
                "model": model.name,
                "tuning_enabled": False,
                "fit_duration_seconds": elapsed,
                "best_params": model.best_params,
            }

        # Subsample for the search itself (speed)
        X_search, y_search = self._subsample(X, y, cfg.tuning_sample_size)

        logger.info(
            f"  {model.name}: tuning on {len(X_search):,} rows "
            f"(strategy={cfg.tuning_strategy}, cv={cfg.tuning_cv_folds}, "
            f"scoring={cfg.tuning_scoring})"
        )

        pipeline = model.build_pipeline()
        cv       = StratifiedKFold(
            n_splits=cfg.tuning_cv_folds, shuffle=True, random_state=cfg.tuning_random_state
        )

        searcher = self._build_searcher(pipeline, param_grid, cv)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            searcher.fit(X_search, y_search)
        search_elapsed = round(time.perf_counter() - t0, 2)

        best_params = searcher.best_params_
        best_score_search = round(float(searcher.best_score_), 6)

        logger.info(
            f"  {model.name}: search done in {search_elapsed}s — "
            f"best {cfg.tuning_scoring}={best_score_search} | params={best_params}"
        )

        # ── Refit on FULL training data with best params ────────────
        logger.info(f"  {model.name}: refitting on full training set ({len(X):,} rows)")
        final_pipeline = model.build_pipeline()
        final_pipeline.set_params(**best_params)

        t1 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipeline.fit(X, y)
        refit_elapsed = round(time.perf_counter() - t1, 2)

        model.set_fitted_pipeline(final_pipeline)

        report = {
            "model":                  model.name,
            "tuning_enabled":         True,
            "tuning_strategy":        cfg.tuning_strategy,
            "search_sample_size":     len(X_search),
            "cv_folds":               cfg.tuning_cv_folds,
            "scoring":                cfg.tuning_scoring,
            "search_duration_seconds": search_elapsed,
            "refit_duration_seconds":  refit_elapsed,
            "best_cv_score":          best_score_search,
            "best_params":            best_params,
            "n_param_combinations_tried": len(searcher.cv_results_["params"]),
        }
        return model, report

    # ─────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────

    def _subsample(
        self, X: pd.DataFrame, y: pd.Series, max_n: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if len(X) <= max_n:
            return X, y
        # Stratified subsample to preserve class ratio
        idx_pos = y[y == 1].index
        idx_neg = y[y == 0].index
        frac    = max_n / len(y)
        n_pos   = max(10, int(len(idx_pos) * frac))
        n_neg   = max_n - n_pos

        rng = np.random.default_rng(self.config.tuning_random_state)
        sel_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
        sel_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
        sel_idx = np.concatenate([sel_pos, sel_neg])

        return X.loc[sel_idx], y.loc[sel_idx]

    def _build_searcher(self, pipeline, param_grid: dict, cv):
        cfg = self.config

        if cfg.tuning_strategy == "grid":
            return GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=cfg.tuning_scoring,
                n_jobs=cfg.tuning_n_jobs,
                refit=True,
            )

        if cfg.tuning_strategy == "random":
            return RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=cfg.tuning_n_iter,
                cv=cv,
                scoring=cfg.tuning_scoring,
                n_jobs=cfg.tuning_n_jobs,
                random_state=cfg.tuning_random_state,
                refit=True,
            )

        if cfg.tuning_strategy == "bayes":
            try:
                from skopt import BayesSearchCV
                return BayesSearchCV(
                    pipeline,
                    search_spaces=param_grid,
                    n_iter=cfg.tuning_n_iter,
                    cv=cv,
                    scoring=cfg.tuning_scoring,
                    n_jobs=cfg.tuning_n_jobs,
                    random_state=cfg.tuning_random_state,
                    refit=True,
                )
            except ImportError:
                logger.warning(
                    "scikit-optimize not installed — falling back to RandomizedSearchCV. "
                    "Install with: pip install scikit-optimize"
                )
                return RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_grid,
                    n_iter=cfg.tuning_n_iter,
                    cv=cv,
                    scoring=cfg.tuning_scoring,
                    n_jobs=cfg.tuning_n_jobs,
                    random_state=cfg.tuning_random_state,
                    refit=True,
                )

        raise ValueError(
            f"Unknown tuning_strategy: '{cfg.tuning_strategy}'. "
            "Choose from: grid | random | bayes"
        )