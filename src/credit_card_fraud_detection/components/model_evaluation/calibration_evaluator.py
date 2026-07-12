"""
components/model_evaluation/calibration_evaluator.py
=====================================================
Probability calibration check + subgroup performance breakdown.

CalibrationEvaluator — reliability diagram, ECE, Brier skill score.
SubgroupEvaluator    — performance metrics broken down by Amount risk bucket,
                       so the team can see whether the model under-performs
                       on specific transaction-size segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score

from credit_card_fraud_detection.components.model_evaluation.interface import EvaluationComponent
from credit_card_fraud_detection.entity.config_entity import ModelEvaluationConfig
from credit_card_fraud_detection.utils.logging_setup import logger


class CalibrationEvaluator(EvaluationComponent):
    """Reliability diagram + Expected Calibration Error (ECE) + Brier skill score."""

    N_BINS = 10

    def evaluate(self, y_true, y_pred, y_prob, config: ModelEvaluationConfig, **kwargs) -> dict:
        logger.info("Evaluating: Probability Calibration")

        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=self.N_BINS, strategy="uniform"
        )
        bin_counts = np.histogram(y_prob, bins=self.N_BINS, range=(0, 1))[0]
        bin_counts = bin_counts[bin_counts > 0]
        ece = float(np.sum(np.abs(fraction_pos - mean_pred) * bin_counts / len(y_true)))
        brier          = float(brier_score_loss(y_true, y_prob))
        fraud_rate     = float(np.mean(y_true))
        brier_baseline = fraud_rate * (1 - fraud_rate)
        skill_score    = round(1 - brier / brier_baseline, 4) if brier_baseline > 0 else None

        quality = "GOOD" if ece < 0.05 else "MODERATE" if ece < 0.10 else "POOR"

        reliability = [
            {"mean_predicted_prob": round(float(mp), 4), "fraction_positive": round(float(fp), 4)}
            for mp, fp in zip(mean_pred, fraction_pos)
        ]

        return {
            "calibration": {
                "brier_score":           round(brier, 6),
                "brier_baseline_random": round(brier_baseline, 6),
                "brier_skill_score":     skill_score,
                "ece":                   round(ece, 6),
                "quality":               quality,
                "reliability_diagram":   reliability,
                "recommendation": (
                    "Calibration is good — probabilities are reliable for risk ranking."
                    if quality == "GOOD" else
                    "Calibration is moderate — consider Platt scaling or isotonic regression."
                    if quality == "MODERATE" else
                    "Calibration is poor — apply CalibratedClassifierCV(method='isotonic')."
                ),
            }
        }


class SubgroupEvaluator(EvaluationComponent):
    """Performance and financial loss broken down by Amount risk bucket."""

    def evaluate(self, y_true, y_pred, y_prob, config: ModelEvaluationConfig, **kwargs) -> dict:
        df = kwargs.get("df")
        logger.info("Evaluating: Subgroup Performance (by Amount)")

        if df is None or "Amount" not in df.columns:
            return {"subgroup_analysis": "Amount column not available — skipped."}

        amount = df["Amount"].reset_index(drop=True)
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)
        y_prob_s = pd.Series(y_prob).reset_index(drop=True)

        try:
            buckets = pd.qcut(amount, q=5, labels=["very_low", "low", "medium", "high", "very_high"], duplicates="drop")
        except ValueError:
            return {"subgroup_analysis": "Insufficient Amount variance for quantile buckets."}

        results = {}
        for bucket in buckets.cat.categories:
            mask = (buckets == bucket).values
            if mask.sum() < 5:
                continue
                
            yt, yp, ypr = y_true_s[mask], y_pred_s[mask], y_prob_s[mask]
            amount_mask = amount[mask]
            
            n_fraud = int(yt.sum())
            fn_mask = (yt == 1) & (yp == 0)
            euro_loss = float(amount_mask[fn_mask].sum())
            
            results[str(bucket)] = {
                "n_samples":        int(mask.sum()),
                "n_fraud":          n_fraud,
                "amount_range_eur": [round(float(amount_mask.min()), 2), round(float(amount_mask.max()), 2)],
                "recall":           round(float(recall_score(yt, yp, zero_division=0)), 4) if n_fraud > 0 else None,
                "auprc":            round(float(average_precision_score(yt, ypr)), 4) if n_fraud > 0 else None,
                "total_euro_loss":  round(euro_loss, 2)
            }

        worst_bucket = min(
            (k for k, v in results.items() if v["recall"] is not None),
            key=lambda k: results[k]["recall"],
            default=None,
        )

        return {
            "subgroup_analysis": {
                "by_amount_bucket": results,
                "worst_performing_bucket": worst_bucket,
                "interpretation": (
                    f"Lowest recall observed in the '{worst_bucket}' bucket. "
                    f"Review 'total_euro_loss' to assess the true financial impact."
                )
            }
        }