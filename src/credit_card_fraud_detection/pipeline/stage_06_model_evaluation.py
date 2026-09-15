"""
pipeline/evaluation_pipeline.py
=================================
Orchestrates the full model evaluation stage:
  1. Load val and test splits (plus original test data for Amount-based
     cost/subgroup analysis, if available)
  2. Evaluate either the single best model or every saved model
     (config.evaluate_all_models)
  3. Generate plots: ROC, PR curve, confusion matrix, calibration,
     model comparison bar chart, feature importance
  4. Persist evaluation_report.json and model_comparison.json

Usage:
    cfg      = ConfigurationManager().get_model_evaluation_config()
    pipeline = EvaluationPipeline(cfg, val_path)
    pipeline.run()
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from credit_card_fraud_detection.components.model_evaluation import plots
from credit_card_fraud_detection.components.model_evaluation.evaluator import ModelEvaluator
from credit_card_fraud_detection.components.model_trainer.model_factory import ModelFactory
from credit_card_fraud_detection.entity.config_entity import ModelEvaluationConfig
from credit_card_fraud_detection.utils.common import load_data
from credit_card_fraud_detection.utils.logging_setup import logger


class EvaluationPipeline:
    """End-to-end orchestrator for the model evaluation stage."""

    def __init__(self, config: ModelEvaluationConfig, val_path: Path) -> None:
        """
        Args:
            config:   ModelEvaluationConfig
            val_path: path to the validation split, needed for leakage-safe
                      threshold optimisation (kept separate from test_path
                      in config to avoid implicitly coupling the two stages).
        """
        self.config   = config
        self.val_path = val_path
        self.evaluator = ModelEvaluator(config)

    def run(self) -> None:
        cfg = self.config
        logger.info("=" * 70)
        logger.info("MODEL EVALUATION PIPELINE -- START")
        logger.info("=" * 70)
        t0 = time.perf_counter()

        # ── 1. Load data ──────────────────────────────────────────
        val_df  = load_data(self.val_path)
        test_df = load_data(cfg.test_path)

        X_val  = val_df.drop(columns=[cfg.target_column])
        y_val  = val_df[cfg.target_column]
        X_test = test_df.drop(columns=[cfg.target_column])
        y_test = test_df[cfg.target_column]

        logger.info(f"Val: {len(X_val):,} rows | Test: {len(X_test):,} rows")

        # ── 2. Evaluate ────────────────────────────────────────────
        if cfg.evaluate_all_models:
            all_reports = self.evaluator.evaluate_all(X_val, y_val, X_test, y_test, df_test_raw=test_df)
            self.evaluator.save_report()
            self.evaluator.save_comparison_table()

            # Plot model comparison bar chart
            plots.plot_model_comparison(self.evaluator.comparison_table, "auprc", cfg.plots_dir)
            plots.plot_model_comparison(self.evaluator.comparison_table, "recall", cfg.plots_dir)

            # Per-model diagnostic plots
            for name, report in all_reports.items():
                self._plot_for_model(name, X_test, y_test, report, test_df)

            logger.info("\nModel ranking (by AUPRC):")
            for i, row in enumerate(self.evaluator.comparison_table, start=1):
                logger.info(
                    f"  {i}. {row['model']:<20} AUPRC={row['auprc']:.4f}  "
                    f"recall={row['recall']:.4f}  f2={row['f2']:.4f}"
                )
        else:
            model = ModelFactory.load_model("best_model", cfg.best_model_path)
            report = self.evaluator.evaluate_model(model, X_val, y_val, X_test, y_test, df_test_raw=test_df)
            self.evaluator._report = {"best_model": report}
            self.evaluator.save_report()
            self._plot_for_model(model.name, X_test, y_test, report, test_df)

        total = round(time.perf_counter() - t0, 2)
        logger.info("=" * 70)
        logger.info(f"MODEL EVALUATION PIPELINE -- DONE in {total}s")
        logger.info("=" * 70)

    def _plot_for_model(self, name: str, X_test, y_test, report: dict, test_df: pd.DataFrame) -> None:
        cfg = self.config
        try:
            model = ModelFactory.load_model(name, cfg.model_dir / f"{name}.pkl")
        except Exception as exc:
            logger.warning(f"  Could not reload '{name}' for plotting: {exc}")
            return

        y_prob = model.predict_proba(X_test)
        threshold = report.get("threshold_used_for_test", 0.5)
        y_pred = (y_prob >= threshold).astype(int)

        plots.plot_roc_curve(y_test.values, y_prob, name, cfg.plots_dir)
        plots.plot_pr_curve(y_test.values, y_prob, name, cfg.plots_dir)
        plots.plot_confusion_matrix(y_test.values, y_pred, name, cfg.plots_dir)
        if cfg.run_calibration_check:
            plots.plot_calibration_curve(y_test.values, y_prob, name, cfg.plots_dir)
        plots.plot_feature_importance(model, X_test, name, cfg.plots_dir)

    @property
    def report(self) -> dict:
        return self.evaluator.report

    @property
    def comparison_table(self) -> list:
        return self.evaluator.comparison_table