"""
components/model_trainer/trainer.py
====================================
Stage — Model Training & Selection

Trains every model listed in config.models_to_train, tunes each via
HyperparameterTuner, evaluates all candidates on the held-out validation
set, and selects the single best model by model_selection_metric.

Workflow:
  1. Load train (or resampled train) and val splits.
  2. For each model in models_to_train:
       a. Build + tune via HyperparameterTuner (CV on train subsample,
          final refit on full train set).
       b. Evaluate the refit model on the validation set (full metric suite).
  3. Rank all candidates by model_selection_metric on validation AUPRC/MCC/etc.
  4. Save every fitted model individually + persist the single best as
     best_model.pkl for the evaluation stage.
  5. Write train_metrics.json (per-model val scores) and
     tuning_results.json (full hyperparameter search details).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_card_fraud_detection.components.model_trainer.interface import TrainableModel
from credit_card_fraud_detection.components.model_trainer.model_factory import ModelFactory
from credit_card_fraud_detection.components.model_trainer.tuner import HyperparameterTuner
from credit_card_fraud_detection.entity.config_entity import ModelTrainerConfig
from credit_card_fraud_detection.utils.logging_setup import logger


def _compute_val_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "f2":        round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 6),
        "mcc":       round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 6),
        "auprc":     round(float(average_precision_score(y_true, y_prob)), 6),
    }


class ModelTrainer:
    """
    Orchestrates training, tuning, and selection across multiple model
    candidates for the CC Fraud 2023 dataset.
    """

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config  = config
        self.tuner   = HyperparameterTuner(config)
        self._candidates: Dict[str, TrainableModel] = {}
        self._val_metrics: Dict[str, dict]          = {}
        self._tuning_reports: Dict[str, dict]       = {}
        self._best_model_name: str | None = None
        self._report: dict = {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        param_grids: Dict[str, dict],
    ) -> None:
        """
        Train, tune, and validate every model in config.models_to_train.

        Args:
            X_train, y_train: training features/target (or resampled versions)
            X_val,   y_val:   held-out validation features/target
            param_grids:      {model_name: param_grid} sourced from params.yaml
        """
        cfg = self.config
        logger.info(
            f"ModelTrainer: training {len(cfg.models_to_train)} candidates "
            f"on {len(X_train):,} train rows, validating on {len(X_val):,} rows"
        )

        for model_name in cfg.models_to_train:
            logger.info(f"\n-- Training: {model_name.upper()} {'-'*40}")
            t0 = time.perf_counter()
            try:
                grid  = param_grids.get(model_name, {})
                model = ModelFactory.create(model_name, param_grid=grid, random_state=cfg.random_state)

                model, tuning_report = self.tuner.tune(model, X_train, y_train)
                self._tuning_reports[model_name] = tuning_report

                y_prob = model.predict_proba(X_val)
                y_pred = model.predict(X_val)
                metrics = _compute_val_metrics(y_val.values, y_pred, y_prob)
                self._val_metrics[model_name] = metrics
                self._candidates[model_name]  = model

                elapsed = round(time.perf_counter() - t0, 2)
                logger.info(
                    f"  [OK] {model_name} done in {elapsed}s | "
                    f"val AUPRC={metrics['auprc']:.4f}  recall={metrics['recall']:.4f}  "
                    f"f2={metrics['f2']:.4f}  mcc={metrics['mcc']:.4f}"
                )
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 2)
                logger.error(f"  [FAIL] {model_name} after {elapsed}s: {exc}")
                self._val_metrics[model_name] = {"error": str(exc)}

        self._select_best_model()
        self._build_report()

    def save_all_models(self) -> Dict[str, Path]:
        """Persist every successfully trained model to model_dir."""
        cfg   = self.config
        paths = {}
        for name, model in self._candidates.items():
            path = cfg.model_dir / f"{name}.pkl"
            model.save(path)
            paths[name] = path
        return paths

    def save_best_model(self) -> Path:
        """Persist the single best model (by model_selection_metric) to best_model_path."""
        if self._best_model_name is None:
            raise RuntimeError("No best model selected -- call train_all() first.")
        best_model = self._candidates[self._best_model_name]
        return best_model.save(self.config.best_model_path)

    def save_metrics(self) -> Path:
        out = self.config.metrics_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._report, f, indent=2, default=str)
        logger.info(f"ModelTrainer: metrics saved -> {out}")
        return out

    def save_tuning_results(self) -> Path:
        out = self.config.tuning_results_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._tuning_reports, f, indent=2, default=str)
        logger.info(f"ModelTrainer: tuning results saved -> {out}")
        return out

    @property
    def best_model_name(self) -> str | None:
        return self._best_model_name

    @property
    def report(self) -> dict:
        return dict(self._report)

    @property
    def candidates(self) -> Dict[str, TrainableModel]:
        return dict(self._candidates)

    # ─────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────

    def _select_best_model(self) -> None:
        cfg    = self.config
        metric = cfg.model_selection_metric

        valid = {
            name: m[metric]
            for name, m in self._val_metrics.items()
            if "error" not in m and metric in m
        }
        if not valid:
            raise RuntimeError(
                f"No model produced a valid '{metric}' score -- all candidates failed. "
                f"Errors: { {k: v.get('error') for k, v in self._val_metrics.items()} }"
            )

        self._best_model_name = max(valid, key=valid.get)
        logger.info(
            f"\nBest model: {self._best_model_name} "
            f"({metric}={valid[self._best_model_name]:.6f})"
        )

    def _build_report(self) -> None:
        ranking = sorted(
            [
                {"model": name, **metrics}
                for name, metrics in self._val_metrics.items()
                if "error" not in metrics
            ],
            key=lambda r: r.get(self.config.model_selection_metric, -1),
            reverse=True,
        )
        self._report = {
            "models_trained":          self.config.models_to_train,
            "model_selection_metric":  self.config.model_selection_metric,
            "best_model":              self._best_model_name,
            "validation_metrics":      self._val_metrics,
            "ranking":                 ranking,
            "failed_models":           [
                name for name, m in self._val_metrics.items() if "error" in m
            ],
        }