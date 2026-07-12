"""
components/model_evaluation/evaluator.py
=========================================
Stage — Model Evaluation

Orchestrates the full evaluation suite against the held-out test set:
  1. MetricsEvaluator      — core metric suite + confusion matrix
  2. ThresholdOptimizer    — optimal decision threshold (run on val, applied to test)
  3. CostAnalyzer          — financial cost interpretation
  4. CalibrationEvaluator  — reliability diagram + ECE
  5. SubgroupEvaluator     — performance by Amount bucket

Also supports comparing every saved model side-by-side (evaluate_all_models)
to produce a single ranked comparison table.

IMPORTANT: The test set must NEVER be used for threshold selection.
This evaluator expects (X_val, y_val) for threshold tuning and
(X_test, y_test) for the final, reported metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from credit_card_fraud_detection.components.model_evaluation.calibration_evaluator import (
    CalibrationEvaluator,
    SubgroupEvaluator,
)
from credit_card_fraud_detection.components.model_evaluation.metrics_evaluator import (
    CostAnalyzer,
    MetricsEvaluator,
    ThresholdOptimizer,
)
from credit_card_fraud_detection.components.model_trainer.interface import TrainableModel
from credit_card_fraud_detection.components.model_trainer.model_factory import ModelFactory
from credit_card_fraud_detection.entity.config_entity import ModelEvaluationConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class ModelEvaluator:
    """
    Full evaluation pipeline for a single fitted model (or, when
    evaluate_all_models=True, every saved model for comparison).
    """

    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        self._report: dict = {}
        self._comparison: list = []

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def evaluate_model(
        self,
        model: TrainableModel,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test_raw: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Evaluate a single fitted model end-to-end.

        Args:
            model:       fitted TrainableModel
            X_val, y_val:   validation split (used ONLY for threshold tuning)
            X_test, y_test: test split (used for the final reported metrics)
            df_test_raw:    optional original test DataFrame with 'Amount'
                             for cost/subgroup analysis

        Returns:
            Full per-model evaluation report dict.
        """
        cfg = self.config
        logger.info(f"Evaluating model '{model.name}' on {len(X_test):,} test rows")

        report: dict = {"model": model.name}

        # ── 1. Threshold optimisation on VALIDATION data ─────────────
        threshold = 0.5
        if cfg.optimize_threshold:
            y_val_prob = model.predict_proba(X_val)
            y_val_pred = (y_val_prob >= 0.5).astype(int)
            thresh_result = ThresholdOptimizer().evaluate(
                y_val.values, y_val_pred, y_val_prob, cfg
            )
            report.update(thresh_result)
            threshold = thresh_result["threshold_optimization"]["optimal_threshold"]

        # ── 2. Apply chosen threshold to TEST data ───────────────────
        y_test_prob = model.predict_proba(X_test)
        y_test_pred = (y_test_prob >= threshold).astype(int)

        # ── 3. Core metrics ───────────────────────────────────────────
        metrics_result = MetricsEvaluator().evaluate(
            y_test.values, y_test_pred, y_test_prob, cfg
        )
        report.update(metrics_result)
        report["threshold_used_for_test"] = threshold

        # ── 4. Cost analysis ──────────────────────────────────────────
        cost_result = CostAnalyzer().evaluate(
            y_test.values, y_test_pred, y_test_prob, cfg, df=df_test_raw
        )
        report.update(cost_result)

        # ── 5. Calibration ────────────────────────────────────────────
        if cfg.run_calibration_check:
            calib_result = CalibrationEvaluator().evaluate(
                y_test.values, y_test_pred, y_test_prob, cfg
            )
            report.update(calib_result)

        # ── 6. Subgroup analysis ──────────────────────────────────────
        if cfg.run_subgroup_analysis:
            subgroup_result = SubgroupEvaluator().evaluate(
                y_test.values, y_test_pred, y_test_prob, cfg, df=df_test_raw
            )
            report.update(subgroup_result)

        logger.info(
            f"  '{model.name}' test AUPRC={report['metrics']['auprc']:.4f}  "
            f"recall={report['metrics']['recall']:.4f}  "
            f"f2={report['metrics']['f2']:.4f}"
        )
        return report

    def evaluate_all(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        df_test_raw: Optional[pd.DataFrame] = None,
    ) -> Dict[str, dict]:
        """
        Evaluate every model saved under config.model_dir and build a
        ranked comparison table.
        """
        cfg = self.config
        model_files = sorted(cfg.model_dir.glob("*.pkl"))
        model_files = [p for p in model_files if p.stem != "best_model"]

        if not model_files:
            raise FileNotFoundError(f"No saved models found in {cfg.model_dir}")

        all_reports = {}
        for path in model_files:
            name = path.stem
            try:
                model = ModelFactory.load_model(name, path)
            except Exception as exc:
                logger.warning(f"  Could not load model '{name}': {exc}")
                continue
            all_reports[name] = self.evaluate_model(
                model, X_val, y_val, X_test, y_test, df_test_raw
            )

        self._comparison = sorted(
            [
                {"model": name, **r["metrics"], "threshold": r["threshold_used_for_test"]}
                for name, r in all_reports.items()
            ],
            key=lambda r: r["auprc"],
            reverse=True,
        )
        self._report = all_reports
        return all_reports

    def save_report(self, path: Path | None = None) -> Path:
        out = Path(path or self.config.evaluation_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._report, f, indent=2, default=str)
        logger.info(f"ModelEvaluator: report saved -> {out}")
        return out

    def save_comparison_table(self, path: Path | None = None) -> Path:
        if not self._comparison:
            raise RuntimeError("No comparison data -- call evaluate_all() first.")
        out = Path(path or self.config.comparison_table_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._comparison, f, indent=2, default=str)
        logger.info(f"ModelEvaluator: comparison table saved -> {out}")
        return out

    @property
    def report(self) -> dict:
        return dict(self._report)

    @property
    def comparison_table(self) -> list:
        return list(self._comparison)